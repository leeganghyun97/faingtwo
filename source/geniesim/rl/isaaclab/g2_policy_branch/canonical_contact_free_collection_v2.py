# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical v2 OPEN-only collection schema; static and controller-neutral.

It deliberately records one already-consumed packet/epoch, not an alternate
control path.  The later live runner remains solely responsible for invoking
the existing ``env.step`` route exactly once before constructing a row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ..g2_lift_methodology import RIGHT_ARM_JOINTS
from .dls_preview_api_contract import ContactFreeMetricOpenAction
from .rgbd_logging_contract import (
    RGBD_CAMERA_NAMES,
    RGBD_EVIDENCE_SCHEMA,
    validate_camera_evidence,
)


CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA = "g2_canonical_contact_free_collection_v2"
CANONICAL_FRAME = "robot_root"
CANONICAL_QUATERNION_ORDER = "xyzw"
CANONICAL_DTYPE = "float32"
POLICY_DT_S = 0.020
PHYSICS_DT_S = 0.002
TRANSLATION_SCALE_M = 0.0225
# HDF5 stores metric actions as float32.  A source action whose float64 norm
# is exactly 4.5 mm can acquire a sub-nanometre excess after float32 storage
# and float64 norm evaluation.  This is a dataset *readback* tolerance only:
# action construction and the controller boundary continue to enforce the
# exact 4.5-mm command contract before serialization.
SERIALIZED_FLOAT32_ACTION_ROUNDTRIP_TOLERANCE_M = 1.0e-9
ACTOR_FIELDS = (
    "right_wrist_rgb",
    "right_wrist_depth_m",
    "right_wrist_depth_valid",
    "ee_pose_robot_root_m_xyzw",
    "right_arm_joint_position_rad",
    "right_arm_joint_velocity_rad_s",
    "right_arm_joint_acceleration_rad_s2",
    "gripper_state_open",
    "previous_policy_action_4d_metric_root_m",
)
REQUIRED_ROW_FIELDS = (*ACTOR_FIELDS, "policy_action_4d_metric_root_m", "packet_receipt")
FORBIDDEN_FIELD_TOKENS = ("cube", "contact", "phase", "future", "next_", "planner_target", "clearance")


class CanonicalContactFreeCollectionError(ValueError):
    pass


def validate_serialized_policy_action_4d_metric_root_m(value: Any) -> np.ndarray:
    """Validate a float32 action read from canonical HDF5 storage.

    The stored value is never sent back to the controller directly.  A loader
    must still construct any future policy command through the strict live
    action contract.  This helper distinguishes harmless IEEE-754 round-trip
    error from a genuinely over-bound recorded label.
    """

    action = _finite(
        "serialized_policy_action_4d_metric_root_m", value, (4,), np.float32
    )
    if float(action[3]) != 0.0:
        raise CanonicalContactFreeCollectionError(
            "serialized contact-free action must retain HOLD_OPEN=0"
        )
    if float(np.linalg.norm(action[:3].astype(np.float64))) > (
        0.0045 + SERIALIZED_FLOAT32_ACTION_ROUNDTRIP_TOLERANCE_M
    ):
        raise CanonicalContactFreeCollectionError(
            "serialized action exceeds 4.5-mm bound beyond float32 round-trip tolerance"
        )
    return action


def _finite(name: str, value: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise CanonicalContactFreeCollectionError(f"{name} must be finite {dtype} with exact shape {shape}")
    return array


def _sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise CanonicalContactFreeCollectionError(f"{name} must be a full SHA-256")
    return value.lower()


def _hash_packet(epoch_id: str, full8: Sequence[float]) -> str:
    raw = json.dumps({"epoch": epoch_id, "full8": list(full8), "schema": CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _unit_xyzw(pose: Any) -> np.ndarray:
    result = _finite("ee_pose_robot_root_m_xyzw", pose, (7,), np.float32)
    norm = float(np.linalg.norm(result[3:]))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
        raise CanonicalContactFreeCollectionError("EE pose must carry a unit XYZW quaternion")
    return result


def _actor_inventory(fields: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(item) for item in fields)
    forbidden = sorted(name for name in names if any(token in name.lower() for token in FORBIDDEN_FIELD_TOKENS))
    if forbidden or set(names) != set(ACTOR_FIELDS):
        raise CanonicalContactFreeCollectionError(f"actor inventory mismatch: forbidden={forbidden}; missing={sorted(set(ACTOR_FIELDS)-set(names))}")
    return names


@dataclass(frozen=True)
class PacketEpochReceipt:
    epoch_id: str
    full8_packet: tuple[float, float, float, float, float, float, float, float]
    packet_hash: str
    env_step_count: int
    process_action_count: int
    controller_consumption_count: int
    snapshot_hook_enabled: bool
    snapshot_hook_receipt: str | None
    source_sha256: str
    asset_sha256: str
    trajectory_sha256: str
    terminal: bool = False
    contact: bool = False
    gripper_close: bool = False
    forbidden_collision: bool = False
    schema: str = CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA or not self.epoch_id:
            raise CanonicalContactFreeCollectionError("packet receipt schema/epoch mismatch")
        packet = _finite("full8_packet", self.full8_packet, (8,), np.float64)
        if any(abs(float(value)) > 1.0 for value in packet) or tuple(packet[3:7]) != (0.0, 0.0, 0.0, 0.0) or float(packet[7]) != 1.0:
            raise CanonicalContactFreeCollectionError("packet must be arm7 fixed-rotation/elbow plus full8 OPEN=+1")
        object.__setattr__(self, "full8_packet", tuple(float(value) for value in packet))
        if self.packet_hash != _hash_packet(self.epoch_id, packet):
            raise CanonicalContactFreeCollectionError("packet hash does not bind epoch/full8 identity")
        if (self.env_step_count, self.process_action_count, self.controller_consumption_count) != (1, 1, 1):
            raise CanonicalContactFreeCollectionError("packet receipt requires exactly one env.step/process_action/controller consumption")
        if self.snapshot_hook_enabled != bool(self.snapshot_hook_receipt):
            raise CanonicalContactFreeCollectionError("snapshot hook requires one explicit receipt when enabled")
        for name in ("source_sha256", "asset_sha256", "trajectory_sha256"):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if any((self.terminal, self.contact, self.gripper_close, self.forbidden_collision)):
            raise CanonicalContactFreeCollectionError("OPEN-only receipt rejects terminal/contact/close/collision")


@dataclass(frozen=True)
class CanonicalContactFreeRow:
    timestamp_s: float
    control_step: int
    actor_fields: tuple[str, ...]
    right_wrist_rgb: Any
    right_wrist_depth_m: Any
    right_wrist_depth_valid: Any
    ee_pose_robot_root_m_xyzw: Any
    right_arm_joint_position_rad: Any
    right_arm_joint_velocity_rad_s: Any
    right_arm_joint_acceleration_rad_s2: Any
    gripper_state_open: float
    previous_policy_action_4d_metric_root_m: tuple[float, float, float, float]
    policy_action_4d_metric_root_m: tuple[float, float, float, float]
    packet_receipt: PacketEpochReceipt
    jacobian_6x7: Any
    tensor_dtype: str
    runtime_device: str
    camera_evidence: Mapping[str, Any] | None = None
    frame: str = CANONICAL_FRAME
    quaternion_order: str = CANONICAL_QUATERNION_ORDER

    def validate(self) -> None:
        _actor_inventory(self.actor_fields)
        if not math.isfinite(float(self.timestamp_s)) or self.timestamp_s < 0.0 or type(self.control_step) is not int or self.control_step < 0:
            raise CanonicalContactFreeCollectionError("timestamp/control step invalid")
        rgb = np.asarray(self.right_wrist_rgb, dtype=np.uint8)
        if rgb.shape != (192, 256, 3):
            raise CanonicalContactFreeCollectionError("right_wrist_rgb must be uint8[192,256,3]")
        depth = _finite("right_wrist_depth_m", self.right_wrist_depth_m, (192, 256, 1), np.float32)
        valid = np.asarray(self.right_wrist_depth_valid, dtype=np.bool_)
        if valid.shape != depth.shape or np.any((~valid) & (depth != 0.0)):
            raise CanonicalContactFreeCollectionError("depth validity must be bool[192,256,1] with invalid pixels zero-filled")
        _unit_xyzw(self.ee_pose_robot_root_m_xyzw)
        for name in ("right_arm_joint_position_rad", "right_arm_joint_velocity_rad_s", "right_arm_joint_acceleration_rad_s2"):
            _finite(name, getattr(self, name), (7,), np.float32)
        if float(self.gripper_state_open) != 1.0:
            raise CanonicalContactFreeCollectionError("canonical contact-free rows require measured OPEN gripper")
        for name in ("previous_policy_action_4d_metric_root_m", "policy_action_4d_metric_root_m"):
            values = _finite(name, getattr(self, name), (4,), np.float64)
            ContactFreeMetricOpenAction(tuple(float(item) for item in values[:3]), float(values[3]))
        action = ContactFreeMetricOpenAction(
            tuple(float(item) for item in self.policy_action_4d_metric_root_m[:3]),
            float(self.policy_action_4d_metric_root_m[3]),
        )
        expected_arm = action.normalized_arm7
        if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7) for actual, expected in zip(self.packet_receipt.full8_packet[:3], expected_arm[:3], strict=True)):
            raise CanonicalContactFreeCollectionError("controller boundary scale must be exactly metric/0.0225 once")
        _finite("jacobian_6x7", self.jacobian_6x7, (6, 7), np.float32)
        if self.tensor_dtype != CANONICAL_DTYPE or not isinstance(self.runtime_device, str) or not self.runtime_device:
            raise CanonicalContactFreeCollectionError("row requires float32 and explicit runtime device provenance")
        if self.frame != CANONICAL_FRAME or self.quaternion_order != CANONICAL_QUATERNION_ORDER:
            raise CanonicalContactFreeCollectionError("row frame/quaternion order mismatch")
        if self.camera_evidence is not None:
            try:
                evidence = validate_camera_evidence(self.camera_evidence)
            except ValueError as error:
                raise CanonicalContactFreeCollectionError(
                    f"camera evidence invalid: {error}"
                ) from error
            wrist = evidence["right_wrist"]
            if not np.array_equal(
                np.asarray(wrist["rgb"], dtype=np.uint8), rgb
            ):
                raise CanonicalContactFreeCollectionError(
                    "actor wrist RGB and evidence wrist RGB differ"
                )


def collection_manifest_v2(*, source_sha256: str, asset_sha256: str, trajectory_sha256: str, snapshot_hook_enabled: bool) -> dict[str, object]:
    """A serializable schema manifest; no controller or simulation imports."""
    return {
        "schema": CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA,
        "actor_fields": list(ACTOR_FIELDS),
        "action": {"field": "policy_action_4d_metric_root_m", "frame": CANONICAL_FRAME, "unit": "m,m,m,HOLD_OPEN=0", "max_norm_m": 0.0045},
        "previous_action": {
            "field": "previous_policy_action_4d_metric_root_m",
            "frame": CANONICAL_FRAME,
            "unit": "m,m,m,HOLD_OPEN=0",
            "provenance": "previous accepted canonical packet receipt",
            "normalized_actor_input_forbidden": True,
        },
        "controller_boundary_only": {"arm7": "xyz_m/0.0225;rotvec=0;elbow=0", "full8_tail": "+1_OPEN", "scale_count": 1},
        "state": {"ee": "robot_root,m,XYZW", "right_arm_order": list(RIGHT_ARM_JOINTS), "q_qd_qdd": "rad,rad/s,rad/s^2", "jacobian": "[6,7]", "dtype": "float32", "physics_dt_s": PHYSICS_DT_S, "policy_dt_s": POLICY_DT_S},
        "packet_receipt": {"env_step": 1, "process_action": 1, "controller_consumption": 1, "snapshot_hook_enabled": snapshot_hook_enabled},
        "camera_evidence_contract": {
            "active": False,
            "schema": RGBD_EVIDENCE_SCHEMA,
            "current_actor_inputs_changed": False,
        },
        "source_hashes": {"source": _sha256("source_sha256", source_sha256), "asset": _sha256("asset_sha256", asset_sha256), "trajectory": _sha256("trajectory_sha256", trajectory_sha256)},
        "forbidden_actor_field_tokens": list(FORBIDDEN_FIELD_TOKENS),
    }


class CanonicalContactFreeStreamingWriter:
    """Append validated RGB-D rows to a private HDF5 file, then publish once.

    A contact-free episode can contain hundreds of 192x256 RGB-D frames.  A
    list-of-rows writer creates a second full in-memory copy at final HDF5
    conversion and risks losing both the functional report and valid data to
    an OOM kill.  This writer holds one row at a time, leaves no final dataset
    unless ``publish`` is called, and retains the same atomic exclusive
    publication rule as the original writer.
    """

    def __init__(self, destination: str | Path, *, manifest: Mapping[str, object]) -> None:
        self.target = Path(destination)
        if self.target.exists():
            raise FileExistsError(f"refusing to overwrite existing collection: {self.target}")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.target.name}.", suffix=".tmp", dir=self.target.parent
        )
        os.close(fd)
        self._temporary = Path(temporary_name)
        self._handle: h5py.File | None = None
        self._published = False
        self._row_count = 0
        camera_contract = dict(manifest.get("camera_evidence_contract", {}))
        self._camera_evidence_active = bool(camera_contract.get("active", False))
        try:
            handle = h5py.File(self._temporary, "w")
            handle.attrs["schema"] = CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA
            handle.attrs["manifest_json"] = json.dumps(
                dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            group = handle.create_group("rows")
            self._create_datasets(
                group, camera_evidence_active=self._camera_evidence_active
            )
            self._handle = handle
        except BaseException:
            self.abort()
            raise

    @staticmethod
    def _create_datasets(
        group: h5py.Group, *, camera_evidence_active: bool = False
    ) -> None:
        def dataset(name: str, shape: tuple[int, ...], dtype: object) -> None:
            group.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype=dtype,
                chunks=(1, *shape),
            )

        dataset("timestamp_s", (), np.float64)
        dataset("control_step", (), np.int64)
        dataset("policy_action_4d_metric_root_m", (4,), np.float32)
        dataset("previous_policy_action_4d_metric_root_m", (4,), np.float32)
        dataset("ee_pose_robot_root_m_xyzw", (7,), np.float32)
        dataset("right_arm_joint_position_rad", (7,), np.float32)
        dataset("right_arm_joint_velocity_rad_s", (7,), np.float32)
        dataset("right_arm_joint_acceleration_rad_s2", (7,), np.float32)
        dataset("jacobian_6x7", (6, 7), np.float32)
        dataset("gripper_state_open", (1,), np.float32)
        dataset("right_wrist_rgb", (192, 256, 3), np.uint8)
        dataset("right_wrist_depth_m", (192, 256, 1), np.float32)
        dataset("right_wrist_depth_valid", (192, 256, 1), np.bool_)
        if camera_evidence_active:
            for camera_name in RGBD_CAMERA_NAMES:
                if camera_name == "head":
                    dataset("head_rgb", (192, 256, 3), np.uint8)
                dataset(f"{camera_name}_depth_raw_m", (192, 256, 1), np.float32)
                dataset(
                    f"{camera_name}_depth_source_valid",
                    (192, 256, 1),
                    np.bool_,
                )
                dataset(f"{camera_name}_camera_timestamp_s", (), np.float64)
                dataset(f"{camera_name}_camera_frame_id", (), np.int64)
                dataset(f"associated_{camera_name}_frame_index", (), np.int64)
                dataset(
                    f"{camera_name}_camera_pose_root_m_xyzw", (7,), np.float32
                )
        group.create_dataset(
            "packet_receipt_json",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(1,),
        )

    @property
    def row_count(self) -> int:
        return self._row_count

    def append(self, row: CanonicalContactFreeRow) -> None:
        if self._published or self._handle is None:
            raise CanonicalContactFreeCollectionError("collection writer is closed")
        row.validate()
        if self._camera_evidence_active != (row.camera_evidence is not None):
            raise CanonicalContactFreeCollectionError(
                "manifest/row camera evidence activation mismatch"
            )
        group = self._handle["rows"]
        index = self._row_count
        for dataset in group.values():
            dataset.resize((index + 1, *dataset.shape[1:]))
        group["timestamp_s"][index] = np.float64(row.timestamp_s)
        group["control_step"][index] = np.int64(row.control_step)
        group["policy_action_4d_metric_root_m"][index] = np.asarray(
            row.policy_action_4d_metric_root_m, dtype=np.float32
        )
        group["previous_policy_action_4d_metric_root_m"][index] = np.asarray(
            row.previous_policy_action_4d_metric_root_m, dtype=np.float32
        )
        group["ee_pose_robot_root_m_xyzw"][index] = np.asarray(
            row.ee_pose_robot_root_m_xyzw, dtype=np.float32
        )
        for name in (
            "right_arm_joint_position_rad",
            "right_arm_joint_velocity_rad_s",
            "right_arm_joint_acceleration_rad_s2",
            "jacobian_6x7",
        ):
            group[name][index] = np.asarray(getattr(row, name), dtype=np.float32)
        group["gripper_state_open"][index] = np.asarray(
            (row.gripper_state_open,), dtype=np.float32
        )
        group["right_wrist_rgb"][index] = np.asarray(row.right_wrist_rgb, dtype=np.uint8)
        group["right_wrist_depth_m"][index] = np.asarray(
            row.right_wrist_depth_m, dtype=np.float32
        )
        group["right_wrist_depth_valid"][index] = np.asarray(
            row.right_wrist_depth_valid, dtype=np.bool_
        )
        if self._camera_evidence_active:
            evidence = validate_camera_evidence(row.camera_evidence or {})
            for camera_name in RGBD_CAMERA_NAMES:
                camera = evidence[camera_name]
                if camera_name == "head":
                    group["head_rgb"][index] = camera["rgb"]
                group[f"{camera_name}_depth_raw_m"][index] = camera[
                    "depth_raw_m"
                ]
                group[f"{camera_name}_depth_source_valid"][index] = camera[
                    "depth_source_valid"
                ]
                group[f"{camera_name}_camera_timestamp_s"][index] = np.float64(
                    camera["timestamp_s"]
                )
                group[f"{camera_name}_camera_frame_id"][index] = np.int64(
                    camera["frame_id"]
                )
                group[f"associated_{camera_name}_frame_index"][index] = np.int64(
                    camera["associated_frame_index"]
                )
                group[f"{camera_name}_camera_pose_root_m_xyzw"][index] = camera[
                    "pose_root_m_xyzw"
                ]
        group["packet_receipt_json"][index] = json.dumps(
            row.packet_receipt.__dict__, sort_keys=True, allow_nan=False
        )
        self._handle.flush()
        self._row_count += 1

    def publish(self) -> Path:
        return self._publish_to(self.target)

    def _publish_to(
        self,
        destination: Path,
        *,
        manifest_overlay: Mapping[str, object] | None = None,
    ) -> Path:
        if self._published or self._handle is None:
            raise CanonicalContactFreeCollectionError("collection writer is closed")
        if self._row_count <= 0:
            raise CanonicalContactFreeCollectionError("at least one validated row is required")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing collection: {destination}"
            )
        if destination.parent.resolve() != self._temporary.parent.resolve():
            raise CanonicalContactFreeCollectionError(
                "atomic publication destination must share the staging directory"
            )
        if manifest_overlay:
            manifest = json.loads(str(self._handle.attrs["manifest_json"]))
            manifest.update(dict(manifest_overlay))
            self._handle.attrs["manifest_json"] = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        self._handle.flush()
        self._handle.close()
        self._handle = None
        try:
            os.link(
                self._temporary, destination
            )  # atomic exclusive publish; never replaces an existing dataset.
            self._published = True
        finally:
            self._temporary.unlink(missing_ok=True)
        return destination

    def publish_partial_safe(
        self,
        destination: str | Path,
        *,
        failure_classification: str,
        failure_detail: str,
        removed_failure_tail_rows: int,
        removed_safety_rows: int,
        removed_contract_rows: int,
    ) -> Path:
        """Publish only the already validated prefix of a failed episode.

        Rows reach this writer only after the normal controller packet was
        consumed exactly once and the post-step OPEN/contact/collision checks
        passed.  A failure detected before the next action submission is
        therefore not present in the HDF5 prefix.  The distinct artifact role
        prevents a failed episode from being mistaken for a complete rollout.
        """

        counts = (
            removed_failure_tail_rows,
            removed_safety_rows,
            removed_contract_rows,
        )
        if not failure_classification or not failure_detail:
            raise CanonicalContactFreeCollectionError(
                "partial-safe publication requires failure provenance"
            )
        if any(type(value) is not int or value < 0 for value in counts):
            raise CanonicalContactFreeCollectionError(
                "partial-safe removal counts must be non-negative integers"
            )
        return self._publish_to(
            Path(destination),
            manifest_overlay={
                "artifact_role": "PARTIAL_SAFE_PREFIX",
                "complete_episode": False,
                "eligible_for_success_only": False,
                "eligible_for_partial_safe": True,
                "failure_provenance": {
                    "failure_classification": failure_classification,
                    "failure_detail": failure_detail,
                    "partial_safe_rows": self._row_count,
                    "removed_failure_tail_rows": removed_failure_tail_rows,
                    "removed_safety_rows": removed_safety_rows,
                    "removed_contract_rows": removed_contract_rows,
                    "tail_policy": "failure-triggering transition rejected before append",
                },
            },
        )

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if not self._published:
            self._temporary.unlink(missing_ok=True)


def write_canonical_rows_atomic(destination: str | Path, *, rows: Sequence[CanonicalContactFreeRow], manifest: Mapping[str, object]) -> Path:
    """Create a new HDF5 atomically without retaining all rows in a bulk array."""
    writer = CanonicalContactFreeStreamingWriter(destination, manifest=manifest)
    try:
        for row in rows:
            writer.append(row)
        return writer.publish()
    except BaseException:
        writer.abort()
        raise


__all__ = [
    "ACTOR_FIELDS", "CANONICAL_CONTACT_FREE_COLLECTION_SCHEMA", "CanonicalContactFreeCollectionError",
    "CanonicalContactFreeStreamingWriter",
    "CanonicalContactFreeRow", "PacketEpochReceipt", "SERIALIZED_FLOAT32_ACTION_ROUNDTRIP_TOLERANCE_M",
    "collection_manifest_v2", "validate_serialized_policy_action_4d_metric_root_m", "write_canonical_rows_atomic",
]
