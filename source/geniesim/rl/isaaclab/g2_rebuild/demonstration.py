# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable HDF5 storage and migration for G2 demonstrations.

The controlled-rebuild format is self-contained: migration materializes data
instead of creating HDF5 external links.  All writers use exclusive creation,
record contract/provenance hashes, and keep a per-row validity mask for every
field.  Legacy files are opened read-only and verified byte-for-byte unchanged
after migration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

import h5py
import numpy as np

from .data_contract import DatasetSource, EpisodeRole, G2DataContract, SchemaVersion


DEMONSTRATION_FORMAT = "g2_controlled_rebuild_demonstration_hdf5_v2"
ANNOTATION_FORMAT = "g2_controlled_rebuild_annotation_sidecar_v1"
EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

# One transition row always means ``(s_t, a_t, outcome_{t+1})``.  Keep this
# inventory next to the storage contract so independent HUMAN and live-Mimic
# producers cannot silently give an identically named physical-event column a
# different temporal meaning.
TRANSITION_ALIGNMENT_FORMAT = "g2_controlled_rebuild_s_t_a_t_outcome_t_plus_1_v1"
PRE_ACTION_STATE_FIELDS = (
    "head_rgb",
    "head_depth_m",
    "head_depth_valid",
    "right_wrist_rgb",
    "right_wrist_depth_m",
    "right_wrist_depth_valid",
    "head_camera_pose_robot_root_xyzw",
    "right_wrist_camera_pose_robot_root_xyzw",
    "camera_timestamp_s",
    "camera_sequence_id",
    "head_cube_gt_pose_at_capture_robot_root_xyzw",
    "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw",
    "head_cube_visible",
    "right_wrist_cube_visible",
    "head_cube_visibility_valid",
    "right_wrist_cube_visibility_valid",
    "robot_joint_position_rad",
    "robot_joint_velocity_rad_s",
    "ee_pose_robot_root_xyzw",
    "distal_pad_midpoint_robot_root_m",
    "cube_gt_pose_robot_root_xyzw",
)
ACTION_FIELDS = (
    "gripper_command",
    "keyboard_physical_command",
    "operator_action_8d",
    "canonical_policy_action_7d",
    "applied_actuator_joint_target_rad",
    "controller_target_ee_pose_robot_root_xyzw",
)
POST_ACTION_OUTCOME_FIELDS = (
    "right_gripper_inner_outer_contact_force_n",
    "right_gripper_inner_outer_contact_valid",
    "exact_inner_contact",
    "exact_outer_contact",
    "exact_bilateral_contact",
    "stable_grasp",
    "physical_lift_start",
    "cube_center_height_world_m",
    "forbidden_collision",
    "safety_violation",
    "success",
    "failure",
)

# Absence is reported, never synthesized.  Some migrated episodes are useful
# for BC/contact learning even when this inventory remains incomplete.
DEMONSTRATION_REQUIRED_FIELDS = frozenset(
    {
        "timestamp_s",
        "control_step",
        "head_rgb",
        "head_depth_m",
        "head_depth_valid",
        "right_wrist_rgb",
        "right_wrist_depth_m",
        "right_wrist_depth_valid",
        "head_camera_pose_robot_root_xyzw",
        "right_wrist_camera_pose_robot_root_xyzw",
        "camera_timestamp_s",
        "camera_sequence_id",
        "camera_frame_age_s",
        "robot_joint_position_rad",
        "robot_joint_velocity_rad_s",
        "ee_pose_robot_root_xyzw",
        "ee_linear_velocity_robot_root_m_s",
        "controller_target_ee_pose_robot_root_xyzw",
        "distal_pad_midpoint_robot_root_m",
        "gripper_command",
        "gripper_state",
        "gripper_master_raw_rad",
        "right_gripper_inner_outer_contact_force_n",
        "right_gripper_inner_outer_contact_valid",
        "exact_inner_contact",
        "exact_outer_contact",
        "exact_bilateral_contact",
        "stable_grasp",
        "physical_lift_start",
        "success",
        "failure",
        "forbidden_collision",
        "safety_violation",
        "keyboard_physical_command",
        "operator_action_8d",
        "canonical_policy_action_7d",
        "applied_actuator_joint_target_rad",
        "cube_gt_pose_robot_root_xyzw",
        "head_cube_gt_pose_at_capture_robot_root_xyzw",
        "right_wrist_cube_gt_pose_at_capture_robot_root_xyzw",
        "head_cube_visible",
        "right_wrist_cube_visible",
        "head_cube_visibility_valid",
        "right_wrist_cube_visibility_valid",
        "cube_center_height_world_m",
    }
)

LEGACY_FIELD_ALIASES = {
    "timestamp": "timestamp_s",
    "sequence_step": "control_step",
    "head_depth": "head_depth_m",
    "right_wrist_depth": "right_wrist_depth_m",
    "applied_teleop_action": "operator_action_8d",
    "canonical_policy_action": "canonical_policy_action_7d",
}


class DemonstrationError(RuntimeError):
    """Base class for storage and migration failures."""


class SourceMutationError(DemonstrationError):
    """Raised if a source file changes while a read-only migration runs."""


def _fsync_finalized_file_and_parent(path: Path) -> None:
    """Make a closed HDF5 artifact and its directory entry crash-durable."""

    resolved = Path(path).resolve()
    descriptor = os.open(resolved, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(resolved.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def canonical_transition_alignment(row_count: int) -> dict[str, Any]:
    """Return the single accepted temporal interpretation of a dataset row."""

    if row_count <= 0:
        raise DemonstrationError("transition alignment requires at least one row")
    return {
        "schema": TRANSITION_ALIGNMENT_FORMAT,
        "state_index": "s_t",
        "action_index": "a_t",
        "outcome_index": "t_plus_1",
        "post_minus_pre_control_steps": 1,
        "all_rows_validated": True,
        "validated_row_count": int(row_count),
        "pre_action_state_fields": list(PRE_ACTION_STATE_FIELDS),
        "action_fields": list(ACTION_FIELDS),
        "post_action_outcome_fields": list(POST_ACTION_OUTCOME_FIELDS),
    }


def episode_transition_alignment(group: h5py.Group) -> Mapping[str, Any]:
    """Read either HUMAN or live-Mimic alignment evidence without guessing."""

    raw = group.attrs.get("episode_metadata_json")
    if raw is None:
        raise DemonstrationError("episode transition alignment metadata is missing")
    try:
        metadata = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise DemonstrationError("episode metadata is not valid JSON") from error
    if not isinstance(metadata, Mapping):
        raise DemonstrationError("episode metadata must be a JSON object")
    alignment = metadata.get("transition_alignment")
    if not isinstance(alignment, Mapping):
        mimic = metadata.get("mimic_live_recapture")
        alignment = mimic.get("transition_alignment") if isinstance(mimic, Mapping) else None
    if not isinstance(alignment, Mapping):
        raise DemonstrationError("episode transition alignment attestation is missing")
    return alignment


def assert_canonical_transition_alignment(group: h5py.Group) -> None:
    """Fail closed unless every row is attested as ``s_t,a_t,outcome_(t+1)``."""

    rows = int(group.attrs.get("transition_count", -1))
    expected = canonical_transition_alignment(rows)
    observed = episode_transition_alignment(group)
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise DemonstrationError(
            "episode transition alignment mismatch: " + repr(mismatches)
        )


def assert_complete_canonical_dataset(file: h5py.File) -> None:
    """Validate immutable file-level authority before a learning consumer reads it."""

    contract = G2DataContract.canonical()
    if file.attrs.get("format", "") != DEMONSTRATION_FORMAT:
        raise DemonstrationError("controlled-rebuild dataset format mismatch")
    contract.assert_schema(file.attrs.get("schema_version", ""))
    if file.attrs.get("contract_sha256", "") != contract.sha256():
        raise DemonstrationError("controlled-rebuild contract hash mismatch")
    raw_metadata = file.attrs.get("contract_metadata_json")
    try:
        stored_metadata = json.loads(raw_metadata) if raw_metadata is not None else None
    except (TypeError, json.JSONDecodeError) as error:
        raise DemonstrationError("controlled-rebuild contract metadata is invalid") from error
    expected_metadata = json.loads(json.dumps(contract.metadata()))
    if stored_metadata != expected_metadata:
        raise DemonstrationError("controlled-rebuild contract metadata mismatch")
    if not bool(file.attrs.get("complete", False)):
        raise DemonstrationError("controlled-rebuild dataset is incomplete")
    try:
        DatasetSource(str(file.attrs.get("dataset_source", "")))
    except ValueError as error:
        raise DemonstrationError("controlled-rebuild dataset source is invalid") from error
    if not bool(file.attrs.get("source_is_immutable", False)):
        raise DemonstrationError("controlled-rebuild source immutability is not attested")
    if not bool(file.attrs.get("external_links_forbidden", False)):
        raise DemonstrationError("controlled-rebuild external-link ban is not attested")
    if not math.isclose(float(file.attrs.get("control_hz", -1.0)), contract.control_hz):
        raise DemonstrationError("controlled-rebuild control rate mismatch")
    if not math.isclose(float(file.attrs.get("dt_s", -1.0)), contract.dt_s):
        raise DemonstrationError("controlled-rebuild dt mismatch")
    observed_episode_count = sum(1 for _ in iter_episode_locations(file))
    declared_episode_count = int(file.attrs.get("episode_count", -1))
    if observed_episode_count <= 0 or declared_episode_count != observed_episode_count:
        raise DemonstrationError(
            "controlled-rebuild episode inventory mismatch: "
            f"declared={declared_episode_count}:observed={observed_episode_count}"
        )


def assert_registered_field_metadata(
    group: h5py.Group, names: Iterator[str] | tuple[str, ...] | list[str] | set[str]
) -> None:
    """Prove stored units/frames/dtypes/timestamps still match the contract."""

    contract = G2DataContract.canonical()
    fields = group.get("fields")
    if fields is None:
        raise DemonstrationError("episode fields group is missing")
    for name in names:
        if name not in contract.tensors:
            raise DemonstrationError(f"unregistered field requested: {name}")
        if name not in fields:
            raise DemonstrationError(f"registered episode field is missing: {name}")
        dataset = fields[name]
        if not bool(dataset.attrs.get("contract_registered", False)):
            raise DemonstrationError(f"field is not contract registered: {name}")
        raw = dataset.attrs.get("tensor_metadata_json")
        try:
            observed = json.loads(raw) if raw is not None else None
        except (TypeError, json.JSONDecodeError) as error:
            raise DemonstrationError(f"invalid tensor metadata JSON: {name}") from error
        if not isinstance(observed, Mapping):
            raise DemonstrationError(f"tensor metadata is missing: {name}")
        # JSON storage canonicalizes tuples to lists.  Compare against the
        # same serialized representation rather than Python container types.
        expected = json.loads(json.dumps(contract.spec(name).metadata()))
        mismatches = {
            key: {"expected": value, "observed": observed.get(key)}
            for key, value in expected.items()
            if observed.get(key) != value
        }
        if mismatches:
            raise DemonstrationError(
                f"tensor metadata contract mismatch:{name}:{mismatches}"
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enum_member(enum_type: type, value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value).strip()
    for member in enum_type:
        if text.lower() in {str(member.name).lower(), str(member.value).lower()}:
            return member
    raise ValueError(f"unsupported {enum_type.__name__}: {value!r}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _row_validity(value: np.ndarray) -> np.ndarray:
    if value.ndim == 0:
        raise DemonstrationError("episode fields must have a transition dimension")
    if np.issubdtype(value.dtype, np.number):
        finite = np.isfinite(value)
        if finite.ndim == 1:
            return finite.astype(np.bool_, copy=False)
        return finite.reshape(value.shape[0], -1).all(axis=1)
    return np.ones(value.shape[0], dtype=np.bool_)


def _normalize_validity(name: str, value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise DemonstrationError(
            f"field validity for {name!r} must be canonical bool; "
            "implicit numeric conversion is forbidden"
        )
    if array.ndim == 0:
        return np.full(rows, bool(array), dtype=np.bool_)
    if array.shape != (rows,):
        raise DemonstrationError(
            f"field validity for {name!r} must be scalar or ({rows},), got {array.shape}"
        )
    return array


@dataclass(frozen=True)
class EpisodeLocation:
    role: EpisodeRole
    episode_id: str
    hdf5_path: str


@dataclass(frozen=True)
class MigrationReport:
    source: str
    source_sha256: str
    destination: str
    destination_sha256: str
    episodes: int
    transitions: int
    role_counts: Mapping[str, int]
    missing_required_fields: tuple[str, ...]
    source_unchanged: bool
    self_contained: bool = True
    lift_event_migrated_as_strict_authority: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "destination": self.destination,
            "destination_sha256": self.destination_sha256,
            "episodes": self.episodes,
            "transitions": self.transitions,
            "role_counts": dict(self.role_counts),
            "missing_required_fields": list(self.missing_required_fields),
            "source_unchanged": self.source_unchanged,
            "self_contained": self.self_contained,
            "lift_event_migrated_as_strict_authority": (
                self.lift_event_migrated_as_strict_authority
            ),
        }


class G2DemonstrationWriter:
    """Exclusive-create writer for the controlled rebuild dataset."""

    def __init__(
        self,
        destination: str | Path,
        *,
        dataset_source: DatasetSource | str,
        contract: G2DataContract | None = None,
        source_path: str | Path | None = None,
        source_sha256: str | None = None,
    ) -> None:
        self.destination = Path(destination).resolve()
        self.contract = contract or G2DataContract.canonical()
        self.dataset_source = _enum_member(DatasetSource, dataset_source)
        if source_path is not None and self.destination == Path(source_path).resolve():
            raise DemonstrationError("source and destination must be different files")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        # h5py mode "x" is the immutability guard: an existing artifact is
        # never truncated or updated, including an earlier migration result.
        self._file = h5py.File(self.destination, "x")
        self._closed = False
        self._episode_ids: set[str] = set()
        self._file.attrs.update(
            {
                "format": DEMONSTRATION_FORMAT,
                "schema_version": self.contract.schema_version.value,
                "contract_sha256": self.contract.sha256(),
                "contract_metadata_json": json.dumps(
                    self.contract.metadata(), sort_keys=True, separators=(",", ":")
                ),
                "dataset_source": self.dataset_source.value,
                "control_hz": self.contract.control_hz,
                "dt_s": self.contract.dt_s,
                "complete": False,
                "source_is_immutable": True,
                "external_links_forbidden": True,
                "strict_lift_requires_physical_authority": True,
            }
        )
        if source_path is not None:
            self._file.attrs["source_path"] = str(Path(source_path).resolve())
        if source_sha256 is not None:
            self._file.attrs["source_sha256"] = source_sha256
        dataset = self._file.create_group("dataset")
        for role in EpisodeRole:
            dataset.create_group(role.value)

    def __enter__(self) -> "G2DemonstrationWriter":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close(complete=exception_type is None)

    def write_episode(
        self,
        episode_id: str,
        *,
        role: EpisodeRole | str,
        payload: Mapping[str, Any],
        field_validity: Mapping[str, Any] | None = None,
        field_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        validate_contract_fields: bool = True,
        allow_unregistered_fields: bool = False,
    ) -> EpisodeLocation:
        if self._closed:
            raise DemonstrationError("writer is closed")
        if not EPISODE_ID_PATTERN.fullmatch(episode_id):
            raise DemonstrationError(f"invalid episode id: {episode_id!r}")
        if episode_id in self._episode_ids:
            raise DemonstrationError(f"duplicate episode id: {episode_id}")
        if not payload:
            raise DemonstrationError("episode payload cannot be empty")
        episode_role = _enum_member(EpisodeRole, role)
        arrays = {name: np.asarray(value) for name, value in payload.items()}
        rows = int(next(iter(arrays.values())).shape[0]) if next(iter(arrays.values())).ndim else 0
        if rows <= 0:
            raise DemonstrationError("episode must contain at least one transition")
        for name, array in arrays.items():
            if array.ndim == 0 or array.shape[0] != rows:
                raise DemonstrationError(
                    f"episode field {name!r} has inconsistent transition dimension {array.shape}"
                )
            if name not in self.contract.tensors and not allow_unregistered_fields:
                raise DemonstrationError(f"unregistered contract field: {name}")
        registered = tuple(name for name in arrays if name in self.contract.tensors)
        if validate_contract_fields and registered:
            self.contract.validate_payload(arrays, names=registered)

        validity_input = field_validity or {}
        unexpected_validity = set(validity_input).difference(arrays)
        if unexpected_validity:
            raise DemonstrationError(
                f"validity supplied for missing fields: {sorted(unexpected_validity)}"
            )
        validity = {
            name: (
                _normalize_validity(name, validity_input[name], rows)
                if name in validity_input
                else _row_validity(array)
            )
            for name, array in arrays.items()
        }

        path = f"dataset/{episode_role.value}/{episode_id}"
        group = self._file.create_group(path)
        group.attrs.update(
            {
                "episode_id": episode_id,
                "episode_role": episode_role.value,
                "dataset_source": self.dataset_source.value,
                "transition_count": rows,
                "schema_version": self.contract.schema_version.value,
                "contract_sha256": self.contract.sha256(),
                "field_validity_required": True,
                "strict_lift_event_authority": False,
            }
        )
        if metadata:
            group.attrs["episode_metadata_json"] = json.dumps(
                dict(metadata), sort_keys=True, default=_json_default
            )
        field_group = group.create_group("fields")
        validity_group = group.create_group("field_validity")
        supplied_metadata = field_metadata or {}
        for name, array in arrays.items():
            kwargs: dict[str, Any] = {}
            if array.ndim > 0 and array.size > 0:
                kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
            dataset = field_group.create_dataset(name, data=array, **kwargs)
            registered_field = name in self.contract.tensors
            dataset.attrs["contract_registered"] = registered_field
            dataset.attrs["record_source"] = self.dataset_source.value
            if registered_field:
                spec_metadata = self.contract.spec(name).metadata()
                extra_metadata = dict(supplied_metadata.get(name, {}))
                for key in (
                    "name",
                    "shape",
                    "dtype",
                    "unit",
                    "frame",
                    "normalization",
                    "timestamp",
                    "source",
                    "schema_version",
                ):
                    if key in extra_metadata and extra_metadata[key] != spec_metadata[key]:
                        raise DemonstrationError(
                            f"field metadata cannot override contract authority: {name}:{key}"
                        )
                spec_metadata = {**spec_metadata, **extra_metadata}
                dataset.attrs["tensor_metadata_json"] = json.dumps(
                    spec_metadata, sort_keys=True, separators=(",", ":")
                )
                for key in (
                    "unit",
                    "frame",
                    "normalization",
                    "timestamp",
                    "source",
                    "schema_version",
                ):
                    dataset.attrs[key] = spec_metadata[key]
            else:
                legacy_metadata = dict(supplied_metadata.get(name, {}))
                legacy_metadata = {
                    "name": name,
                    "shape": list(array.shape[1:]),
                    "dtype": str(array.dtype),
                    "unit": "unknown_legacy",
                    "frame": "unknown_legacy",
                    "normalization": "unknown_legacy",
                    "timestamp": "unknown_legacy",
                    "source": DatasetSource.MIGRATED.value,
                    "schema_version": SchemaVersion.CONTROLLED_REBUILD_V2.value,
                    **legacy_metadata,
                }
                dataset.attrs["tensor_metadata_json"] = json.dumps(
                    legacy_metadata, sort_keys=True, default=_json_default
                )
                dataset.attrs["schema_version"] = SchemaVersion.CONTROLLED_REBUILD_V2.value
            validity_group.create_dataset(
                name,
                data=validity[name],
                dtype=np.bool_,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

        missing = sorted(DEMONSTRATION_REQUIRED_FIELDS.difference(arrays))
        group.attrs["missing_required_fields_json"] = json.dumps(missing)
        group.attrs["contract_complete"] = not missing
        self._episode_ids.add(episode_id)
        return EpisodeLocation(episode_role, episode_id, f"/{path}")

    def close(self, *, complete: bool = True) -> None:
        if self._closed:
            return
        self._file.attrs["episode_count"] = len(self._episode_ids)
        self._file.attrs["complete"] = bool(complete)
        self._file.flush()
        self._file.close()
        self._closed = True
        _fsync_finalized_file_and_parent(self.destination)

    def flush(self) -> None:
        """Durably flush completed episodes while keeping the writer alive."""

        if self._closed:
            raise DemonstrationError("writer is closed")
        self._file.flush()


def iter_episode_locations(file: h5py.File) -> Iterator[EpisodeLocation]:
    """Iterate controlled-rebuild episode locations in deterministic order."""

    if "dataset" not in file:
        return
    dataset = file["dataset"]
    for role in EpisodeRole:
        if role.value not in dataset:
            continue
        for episode_id in sorted(dataset[role.value]):
            group = dataset[role.value][episode_id]
            if isinstance(group, h5py.Group):
                yield EpisodeLocation(role, episode_id, group.name)


def load_episode_fields(
    file: h5py.File, location: EpisodeLocation
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    group = file[location.hdf5_path]
    if "fields" not in group or "field_validity" not in group:
        raise DemonstrationError("episode fields/field_validity group is missing")
    fields = {name: np.asarray(value) for name, value in group["fields"].items()}
    validity = {name: np.asarray(value) for name, value in group["field_validity"].items()}
    missing_validity = sorted(set(fields).difference(validity))
    extra_validity = sorted(set(validity).difference(fields))
    if missing_validity or extra_validity:
        raise DemonstrationError(
            "episode field/validity inventory mismatch: "
            f"missing_validity={missing_validity}:extra_validity={extra_validity}"
        )
    invalid = {
        name: (str(value.dtype), tuple(value.shape))
        for name, value in validity.items()
        if value.dtype != np.bool_
        or name not in fields
        or value.shape != (fields[name].shape[0],)
    }
    if invalid:
        raise DemonstrationError(
            "canonical field validity is corrupt or not bool: " + repr(invalid)
        )
    rows = int(group.attrs.get("transition_count", -1))
    inconsistent_rows = {
        name: int(value.shape[0])
        for name, value in fields.items()
        if value.ndim == 0 or int(value.shape[0]) != rows
    }
    if rows <= 0 or inconsistent_rows:
        raise DemonstrationError(
            f"episode transition count mismatch: expected={rows}:fields={inconsistent_rows}"
        )
    return fields, validity


def _legacy_episode_groups(file: h5py.File) -> Iterator[tuple[str, h5py.Group, EpisodeRole | None]]:
    if file.attrs.get("format", "") == DEMONSTRATION_FORMAT and "dataset" in file:
        for location in iter_episode_locations(file):
            yield location.episode_id, file[location.hdf5_path], location.role
        return
    if "data" not in file:
        raise DemonstrationError("legacy source has no /data episode group")
    for name in sorted(file["data"]):
        group = file["data"][name]
        if isinstance(group, h5py.Group):
            yield name, group, None


def _source_tensor_metadata(dataset: h5py.Dataset) -> dict[str, Any]:
    """Read source-declared tensor semantics without inferring them.

    A matching shape is not evidence that a legacy value is expressed in the
    canonical unit or frame.  Only explicit dataset metadata may authorize a
    canonical alias during migration.
    """

    raw = dataset.attrs.get("tensor_metadata_json")
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _legacy_arrays(
    group: h5py.Group,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Mapping[str, Any]]]:
    if "fields" in group and isinstance(group["fields"], h5py.Group):
        arrays = {name: np.asarray(value) for name, value in group["fields"].items()}
        metadata = {
            name: _source_tensor_metadata(value)
            for name, value in group["fields"].items()
            if isinstance(value, h5py.Dataset)
        }
        validity = (
            {
                name: np.asarray(value)
                for name, value in group["field_validity"].items()
            }
            if "field_validity" in group
            else {}
        )
        return arrays, validity, metadata
    arrays = {
        name: np.asarray(value)
        for name, value in group.items()
        if isinstance(value, h5py.Dataset)
    }
    metadata = {
        name: _source_tensor_metadata(value)
        for name, value in group.items()
        if isinstance(value, h5py.Dataset)
    }
    return arrays, {}, metadata


def _infer_role(
    group: h5py.Group,
    arrays: Mapping[str, np.ndarray],
    explicit_role: EpisodeRole | str | None,
    inherited_role: EpisodeRole | None,
) -> EpisodeRole:
    if explicit_role is not None:
        return _enum_member(EpisodeRole, explicit_role)
    if inherited_role is not None:
        return inherited_role
    if "episode_role" in group.attrs:
        return _enum_member(EpisodeRole, group.attrs["episode_role"])
    if bool(group.attrs.get("recovery", False)):
        return EpisodeRole.RECOVERY
    success_name = (
        "success"
        if "success" in arrays
        else "legacy_unattested__success"
        if "legacy_unattested__success" in arrays
        else None
    )
    if success_name is not None and bool(
        np.asarray(arrays[success_name], dtype=np.bool_).any()
    ):
        return EpisodeRole.SUCCESS
    # Unknown/ongoing legacy outcomes are conservatively retained as failure,
    # never promoted into the BC-success partition.
    return EpisodeRole.FAILURE


def _canonicalize_legacy_arrays(
    arrays: Mapping[str, np.ndarray],
    validity: Mapping[str, np.ndarray],
    source_metadata: Mapping[str, Mapping[str, Any]],
    contract: G2DataContract,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Mapping[str, Any]]]:
    converted: dict[str, np.ndarray] = {}
    converted_validity: dict[str, np.ndarray] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for old_name, original in arrays.items():
        candidate = LEGACY_FIELD_ALIASES.get(old_name, old_name)
        value = np.asarray(original)
        if candidate in contract.tensors:
            spec = contract.spec(candidate)
            if spec.shape == (1,) and value.ndim == 1:
                value = value[:, None]
            declared = dict(source_metadata.get(old_name, {}))
            expected = spec.metadata()
            semantic_keys = ("unit", "frame", "normalization", "timestamp")
            semantics_attested = all(
                declared.get(key) == expected[key] for key in semantic_keys
            )
            try:
                spec.validate(value)
            except (TypeError, ValueError):
                # Retain incompatible information under its exact legacy name;
                # migration never silently casts units, frames, or dtypes.
                candidate = f"legacy_incompatible__{old_name}"
                value = np.asarray(original)
            else:
                if not semantics_attested:
                    candidate = f"legacy_unattested__{old_name}"
                    value = np.asarray(original)
        if candidate in converted:
            raise DemonstrationError(f"legacy field alias collision: {candidate}")
        converted[candidate] = value
        source_validity = validity.get(old_name, validity.get(candidate))
        converted_validity[candidate] = (
            _normalize_validity(candidate, source_validity, value.shape[0])
            if source_validity is not None
            else _row_validity(value)
        )
        if candidate not in contract.tensors:
            metadata[candidate] = {
                "legacy_name": old_name,
                "source_hdf5_path": group_path_placeholder(old_name),
                "contract_status": "unregistered_or_incompatible_preserved_verbatim",
                "source_tensor_metadata": dict(source_metadata.get(old_name, {})),
                "provenance": DatasetSource.MIGRATED.value,
            }
    return converted, converted_validity, metadata


def group_path_placeholder(field_name: str) -> str:
    """Return a non-misleading relative lineage marker for legacy metadata."""

    return f"<source_episode>/{field_name}"


def migrate_legacy_demonstrations(
    source: str | Path,
    destination: str | Path,
    *,
    default_role: EpisodeRole | str | None = None,
    contract: G2DataContract | None = None,
) -> MigrationReport:
    """Materialize a legacy file into a new, self-contained v1 artifact.

    Existing lift/success phase bits are preserved as legacy fields only.  No
    migrated value is declared to be a strict physical ``LIFT_START`` event.
    """

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise DemonstrationError("source and destination must be different files")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    authority = contract or G2DataContract.canonical()
    source_stat_before = source_path.stat()
    source_hash_before = sha256_file(source_path)
    episode_count = 0
    transition_count = 0
    roles = {role.value: 0 for role in EpisodeRole}
    observed_fields: set[str] = set()

    writer = G2DemonstrationWriter(
        destination_path,
        dataset_source=DatasetSource.MIGRATED,
        contract=authority,
        source_path=source_path,
        source_sha256=source_hash_before,
    )
    try:
        with h5py.File(source_path, "r") as legacy:
            for index, (legacy_id, group, inherited_role) in enumerate(
                _legacy_episode_groups(legacy)
            ):
                arrays, validity, source_metadata = _legacy_arrays(group)
                if not arrays:
                    continue
                converted, converted_validity, field_metadata = _canonicalize_legacy_arrays(
                    arrays, validity, source_metadata, authority
                )
                # Partition lineage from the source outcome bit before conservative
                # field renaming (for example uint8 legacy success is preserved as
                # incompatible rather than silently cast to canonical bool).
                role = _infer_role(group, arrays, default_role, inherited_role)
                episode_id = legacy_id if EPISODE_ID_PATTERN.fullmatch(legacy_id) else f"episode_{index:06d}"
                rows = int(next(iter(converted.values())).shape[0])
                writer.write_episode(
                    episode_id,
                    role=role,
                    payload=converted,
                    field_validity=converted_validity,
                    field_metadata=field_metadata,
                    metadata={
                        "legacy_episode_path": group.name,
                        "legacy_episode_id": legacy_id,
                        "provenance": DatasetSource.MIGRATED.value,
                        "strict_lift_event_authority": False,
                        "legacy_lift_must_not_be_used_as_strict_authority": True,
                    },
                    validate_contract_fields=True,
                    allow_unregistered_fields=True,
                )
                episode_count += 1
                transition_count += rows
                roles[role.value] += 1
                observed_fields.update(converted)
            if episode_count == 0:
                raise DemonstrationError("source contains no episodes with array fields")

        # Revalidate the source before the destination's complete bit is
        # committed.  A mutation now exits through close(complete=False), so
        # a failed migration can never leave promotable evidence behind.
        source_stat_after = source_path.stat()
        source_hash_after = sha256_file(source_path)
        unchanged = (
            source_hash_before == source_hash_after
            and source_stat_before.st_size == source_stat_after.st_size
            and source_stat_before.st_mtime_ns == source_stat_after.st_mtime_ns
        )
        if not unchanged:
            raise SourceMutationError(
                f"source changed during migration: {source_path}"
            )
        writer.close(complete=True)
    except BaseException:
        writer.close(complete=False)
        raise
    return MigrationReport(
        source=str(source_path),
        source_sha256=source_hash_before,
        destination=str(destination_path),
        destination_sha256=sha256_file(destination_path),
        episodes=episode_count,
        transitions=transition_count,
        role_counts=roles,
        missing_required_fields=tuple(sorted(DEMONSTRATION_REQUIRED_FIELDS - observed_fields)),
        source_unchanged=True,
    )


def write_annotation_sidecar(
    source: str | Path,
    destination: str | Path,
    annotations: Mapping[EpisodeLocation, Any],
    *,
    contract: G2DataContract | None = None,
) -> str:
    """Write annotations to a new file without changing the demonstration."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise DemonstrationError("annotation sidecar cannot overwrite its source")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    source_hash_before = sha256_file(source_path)
    source_stat_before = source_path.stat()
    authority = contract or G2DataContract.canonical()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination_path, "x") as output:
        output.attrs.update(
            {
                "format": ANNOTATION_FORMAT,
                "schema_version": authority.schema_version.value,
                "contract_sha256": authority.sha256(),
                "source_path": str(source_path),
                "source_sha256": source_hash_before,
                "complete": False,
                "strict_lift_requires_physical_authority": True,
            }
        )
        root = output.create_group("annotations")
        for location, annotation in annotations.items():
            group = root.create_group(f"{location.role.value}/{location.episode_id}")
            for name, value in annotation.as_arrays().items():
                group.create_dataset(
                    name,
                    data=np.asarray(value),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            group.attrs.update(
                {
                    "source_episode_path": location.hdf5_path,
                    "episode_role": location.role.value,
                    "strict_order_valid": annotation.strict_order_valid,
                    "strict_order_reason": annotation.strict_order_reason,
                    "lift_has_physical_authority": annotation.lift_has_physical_authority,
                    "first_close_step": -1 if annotation.first_close_step is None else annotation.first_close_step,
                    "contact_step": -1 if annotation.contact_step is None else annotation.contact_step,
                    "stable_step": -1 if annotation.stable_step is None else annotation.stable_step,
                    "lift_step": -1 if annotation.lift_step is None else annotation.lift_step,
                    "close_to_contact_steps": -1 if annotation.close_to_contact_steps is None else annotation.close_to_contact_steps,
                    "close_to_stable_steps": -1 if annotation.close_to_stable_steps is None else annotation.close_to_stable_steps,
                    "close_timing_support_steps": annotation.close_timing_support_steps,
                    "close_timing_temperature_steps": annotation.close_timing_temperature_steps,
                    "future_safe_close_proximity_steps": annotation.future_safe_close_proximity_steps,
                    "future_contact_horizon_steps": annotation.future_contact_horizon_steps,
                    "future_stable_horizon_steps": annotation.future_stable_horizon_steps,
                    "gripper_open_minimum": annotation.gripper_open_minimum,
                    "source_transition_alignment": TRANSITION_ALIGNMENT_FORMAT,
                    "event_marker_alignment": "transition_row_that_produced_event",
                    "first_close_step_alignment": "action_t",
                    "physical_event_step_alignment": "outcome_t_plus_1",
                    "physical_event_timestamp_offset_s": authority.dt_s,
                }
            )
        output.attrs["episode_count"] = len(annotations)
        # Source binding is part of the commit condition, not a post-hoc
        # diagnostic.  If it changed, the context closes with complete=False.
        source_stat_after = source_path.stat()
        if (
            source_hash_before != sha256_file(source_path)
            or source_stat_before.st_size != source_stat_after.st_size
            or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
        ):
            raise SourceMutationError(
                f"source changed while annotating: {source_path}"
            )
        output.attrs["complete"] = True
        output.flush()
    _fsync_finalized_file_and_parent(destination_path)
    return sha256_file(destination_path)


__all__ = [
    "ANNOTATION_FORMAT",
    "DEMONSTRATION_FORMAT",
    "DEMONSTRATION_REQUIRED_FIELDS",
    "DemonstrationError",
    "EpisodeLocation",
    "G2DemonstrationWriter",
    "MigrationReport",
    "SourceMutationError",
    "iter_episode_locations",
    "load_episode_fields",
    "migrate_legacy_demonstrations",
    "sha256_file",
    "write_annotation_sidecar",
]
