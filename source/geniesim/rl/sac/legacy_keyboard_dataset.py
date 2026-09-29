# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Strict legacy-keyboard classification and zero-copy event-window loading.

The legacy v26 HDF5 files are immutable views over the original RGB-D files.
This module writes JSON references only; image and depth tensors remain in the
source HDF5.  It deliberately keeps action-contract classification separate
from Production-to-Candidate-A mechanics compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from geniesim.rl.isaaclab.g2_lift_methodology import (
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_MASTER,
)

from .keyboard_grasp_contract import canonical_keyboard_grasp_contract


DATASET_MANIFEST_SCHEMA = "g2_legacy_keyboard_dataset_manifest_v1"
EPISODE_MANIFEST_SCHEMA = "g2_legacy_keyboard_episode_manifest_v1"
REPORT_SCHEMA = "g2_legacy_keyboard_dataset_report_v1"
COMPATIBILITY_SCHEMA = "g2_production_candidate_a_keyboard_compatibility_v1"
EXPECTED_LEGACY_SCHEMA = (
    "g2_keyboard_teacher_rgbd_canonical_xyzw_rotvec_rect_object_pad_fk_v26"
)
EXPECTED_PRODUCTION_SHA256 = (
    "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
)
EXPECTED_CANDIDATE_A_SHA256 = (
    "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
)
EXPECTED_ACTION_ORDER = (
    "dx",
    "dy",
    "dz",
    "drotvec_x",
    "drotvec_y",
    "drotvec_z",
    "elbow_nullspace",
    "gripper",
)
EXPECTED_PHYSICAL_LABELS = (
    "dx_m",
    "dy_m",
    "dz_m",
    "drotvec_x_rad",
    "drotvec_y_rad",
    "drotvec_z_rad",
    "elbow_nullspace_normalized",
    "gripper_binary",
)
EXPECTED_PHYSICAL_UNITS = (
    "m",
    "m",
    "m",
    "rad",
    "rad",
    "rad",
    "normalized",
    "binary",
)
EXPECTED_NORMALIZED_LABELS = (
    "dx_normalized",
    "dy_normalized",
    "dz_normalized",
    "drotvec_x_normalized",
    "drotvec_y_normalized",
    "drotvec_z_normalized",
    "elbow_nullspace_normalized",
    "gripper_binary",
)
EXPECTED_CONTROLLED_JOINT_ORDER = (*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER)
EXPECTED_CANDIDATE_OVERRIDES = {
    "/genie/joints/idx73_gripper_r_inner_joint4.physics:lowerLimit": (-2.0, -11.25),
    "/genie/joints/idx73_gripper_r_inner_joint4.physics:upperLimit": (2.0, 11.25),
    "/genie/joints/idx83_gripper_r_outer_joint4.physics:lowerLimit": (-2.0, -11.25),
    "/genie/joints/idx83_gripper_r_outer_joint4.physics:upperLimit": (2.0, 11.25),
}

WINDOW_SPECS = {
    "CLOSE_TIMING": (-10, 10),
    "GRASP_TRAJECTORY": (-10, 52),
}
CONTEXT_CLASSES = ("PRE_CLOSE", "CLOSE_TRANSITION", "POST_CLOSE_CONTEXT")

# These fields are safe to copy lazily from a requested window.  Privileged
# teacher observations and the distal-link midpoint proxy are intentionally
# absent.  The caller cannot accidentally treat the midpoint as pad surface.
LAZY_SOURCE_FIELDS = frozenset(
    {
        "head_rgb",
        "head_depth",
        "head_depth_valid",
        "right_wrist_rgb",
        "right_wrist_depth",
        "right_wrist_depth_valid",
        "timestamp",
        "camera_timestamp",
        "camera_frame_age",
    }
)
FORBIDDEN_LOADER_FIELDS = frozenset(
    {
        "grasp_center_position_root_m",
        "teacher_observation",
        "next_teacher_observation",
        "contact_force_n",
        "bilateral_contact",
        "stable_grasp",
    }
)


class RowClass(StrEnum):
    VALID_DIRECT = "VALID_DIRECT"
    REQUIRES_MIGRATION = "REQUIRES_MIGRATION"
    REJECT = "REJECT"


class CompatibilityStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class LegacyKeyboardDatasetError(RuntimeError):
    """Raised when immutable dataset or manifest authority is invalid."""


@dataclass(frozen=True)
class RowClassification:
    classification: np.ndarray
    reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.classification.ndim != 1:
            raise ValueError("row classification must be rank one")
        if len(self.reasons) != int(self.classification.shape[0]):
            raise ValueError("row classification/reason cardinality mismatch")

    @property
    def valid_direct_mask(self) -> np.ndarray:
        return self.classification == RowClass.VALID_DIRECT.value

    def counts(self) -> dict[str, int]:
        return {
            value.value: int(np.count_nonzero(self.classification == value.value))
            for value in RowClass
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LegacyKeyboardDatasetError(f"JSON authority must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_env_args(handle: h5py.File) -> dict[str, Any]:
    data = handle.get("data")
    if not isinstance(data, h5py.Group) or "env_args" not in data.attrs:
        raise LegacyKeyboardDatasetError("legacy HDF5 has no /data env_args authority")
    try:
        value = json.loads(str(data.attrs["env_args"]))
    except json.JSONDecodeError as error:
        raise LegacyKeyboardDatasetError("legacy env_args is invalid JSON") from error
    if not isinstance(value, dict):
        raise LegacyKeyboardDatasetError("legacy env_args must be an object")
    return value


def legacy_semantics_errors(env: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every mismatch that prevents exact interpretation of legacy 8-D rows."""

    contract = canonical_keyboard_grasp_contract()
    errors: list[str] = []
    if env.get("dataset_schema") != EXPECTED_LEGACY_SCHEMA:
        errors.append("UNKNOWN_DATASET_SCHEMA")
    if tuple(env.get("keyboard_action_order", ())) != EXPECTED_ACTION_ORDER:
        errors.append("UNKNOWN_ACTION_ORDER")
    if env.get("keyboard_action_dim") != contract.legacy_action_dim:
        errors.append("UNKNOWN_ACTION_DIMENSION")
    if env.get("physical_to_normalized_applied_exactly_once") is not True:
        errors.append("UNKNOWN_ACTION_SCALING_COUNT")
    if env.get("environment_translation_scale_m_per_normalized_unit") != (
        contract.legacy_normalized_to_metric_scale_m
    ):
        errors.append("UNKNOWN_TRANSLATION_SCALE")
    rotation_scale = env.get(
        "environment_rotation_vector_scale_rad_per_normalized_unit"
    )
    if (
        isinstance(rotation_scale, bool)
        or not isinstance(rotation_scale, (int, float))
        or not math.isfinite(float(rotation_scale))
        or float(rotation_scale) <= 0.0
    ):
        errors.append("UNKNOWN_ROTATION_SCALE")
    if env.get("policy_dt_s") != contract.policy_dt_s:
        errors.append("UNKNOWN_CONTROL_DT")
    if env.get("physics_dt_s") != contract.physics_dt_s:
        errors.append("UNKNOWN_PHYSICS_DT")
    if env.get("control_decimation") != 10:
        errors.append("UNKNOWN_CONTROL_DECIMATION")
    if env.get("g2_usd_sha256") != EXPECTED_PRODUCTION_SHA256:
        errors.append("UNKNOWN_SOURCE_ASSET")

    tensor = env.get("tensor_contract")
    if not isinstance(tensor, Mapping):
        errors.append("MISSING_TENSOR_CONTRACT")
        return tuple(errors)
    physical = tensor.get("keyboard_physical_action")
    normalized = tensor.get("keyboard_normalized_action")
    teacher_fields = tensor.get("teacher_observation_fields")
    controlled_order = tensor.get("controlled_joint_order")
    if not isinstance(controlled_order, (list, tuple)) or tuple(
        controlled_order
    ) != EXPECTED_CONTROLLED_JOINT_ORDER:
        errors.append("UNKNOWN_CONTROLLED_JOINT_ORDER")
    camera_layout = tensor.get("camera_layout")
    if (
        not isinstance(camera_layout, Mapping)
        or camera_layout.get("metric_depth")
        != "[transition,height,width,1]_float_m"
        or not isinstance(camera_layout.get("camera_order"), (list, tuple))
        or tuple(camera_layout.get("camera_order")) != (
            "head",
            "right_wrist",
        )
    ):
        errors.append("UNKNOWN_METRIC_DEPTH_LAYOUT")
    if env.get("teacher_observation_dim") != 59:
        errors.append("UNKNOWN_TEACHER_OBSERVATION_DIMENSION")
    expected_teacher_fields = {
        "controlled_joint_position_relative_rad": (0, 8, "rad", "robot_joint"),
        "controlled_joint_velocity_rad_s": (8, 16, "rad/s", "robot_joint"),
        "end_effector_pose_root_xyzw": (
            16,
            23,
            "m+unit_quaternion_xyzw",
            "robot_root",
        ),
        "cube_pose_root_xyzw": (
            23,
            30,
            "m+unit_quaternion_xyzw",
            "robot_root",
        ),
    }
    observed_teacher_fields: dict[str, tuple[Any, Any, Any, Any]] = {}
    if isinstance(teacher_fields, list):
        for item in teacher_fields:
            if isinstance(item, Mapping) and isinstance(item.get("name"), str):
                observed_teacher_fields[str(item["name"])] = (
                    item.get("start"),
                    item.get("stop"),
                    item.get("unit"),
                    item.get("frame"),
                )
    for name, expected in expected_teacher_fields.items():
        if observed_teacher_fields.get(name) != expected:
            errors.append(f"UNKNOWN_TEACHER_FIELD:{name}")
    initial_pose = env.get("initial_pose_contract")
    initial_right_arm_q: Any = (
        initial_pose.get("right_arm_q_rad")
        if isinstance(initial_pose, Mapping)
        else None
    )
    try:
        initial_q_array = np.asarray(initial_right_arm_q, dtype=np.float64)
    except (TypeError, ValueError):
        initial_q_array = np.empty((0,), dtype=np.float64)
    if initial_q_array.shape != (7,) or not np.isfinite(initial_q_array).all():
        errors.append("UNKNOWN_INITIAL_RIGHT_ARM_Q")
    if not isinstance(physical, Mapping):
        errors.append("MISSING_PHYSICAL_ACTION_CONTRACT")
    else:
        if physical.get("shape") != [8]:
            errors.append("UNKNOWN_PHYSICAL_ACTION_SHAPE")
        if tuple(physical.get("labels", ())) != EXPECTED_PHYSICAL_LABELS:
            errors.append("UNKNOWN_PHYSICAL_ACTION_LABELS")
        if tuple(physical.get("units", ())) != EXPECTED_PHYSICAL_UNITS:
            errors.append("UNKNOWN_PHYSICAL_ACTION_UNITS")
    if not isinstance(normalized, Mapping):
        errors.append("MISSING_NORMALIZED_ACTION_CONTRACT")
    else:
        if normalized.get("shape") != [8]:
            errors.append("UNKNOWN_NORMALIZED_ACTION_SHAPE")
        if tuple(normalized.get("labels", ())) != EXPECTED_NORMALIZED_LABELS:
            errors.append("UNKNOWN_NORMALIZED_ACTION_LABELS")
        conversion = normalized.get("physical_to_normalized")
        if not isinstance(conversion, Mapping):
            errors.append("MISSING_NORMALIZATION_CONTRACT")
        else:
            if conversion.get("translation_divisor_m") != (
                contract.legacy_normalized_to_metric_scale_m
            ):
                errors.append("UNKNOWN_NORMALIZED_TRANSLATION_DIVISOR")
            if conversion.get("rotation_vector_divisor_rad") != env.get(
                "environment_rotation_vector_scale_rad_per_normalized_unit"
            ):
                errors.append("UNKNOWN_NORMALIZED_ROTATION_DIVISOR")
            if conversion.get("elbow_nullspace") != "identity_then_clip_-1_1":
                errors.append("UNKNOWN_ELBOW_SEMANTICS")
            if conversion.get("gripper") != "sign_to_binary_minus1_plus1":
                errors.append("UNKNOWN_GRIPPER_SEMANTICS")
    return tuple(errors)


def classify_legacy_rows(
    normalized_action: Any,
    physical_action: Any,
    *,
    semantics_errors: Sequence[str] = (),
    rotation_scale_rad: float = 0.28125,
) -> RowClassification:
    """Classify legacy rows without slicing, clipping, or saturation.

    Unknown semantics and corrupt values are ``REJECT``.  Known, finite rows
    that require an action-space conversion are ``REQUIRES_MIGRATION``.
    Only rows already satisfying the frozen target action contract are
    ``VALID_DIRECT``.
    """

    contract = canonical_keyboard_grasp_contract()
    normalized = np.asarray(normalized_action)
    physical = np.asarray(physical_action)
    if normalized.ndim != 2 or physical.ndim != 2:
        raise ValueError("legacy actions must be rank-two row arrays")
    if normalized.shape[0] != physical.shape[0]:
        raise ValueError("legacy normalized/physical row counts differ")
    rows = int(normalized.shape[0])
    classifications = np.full((rows,), RowClass.REJECT.value, dtype="U20")
    reasons: list[tuple[str, ...]] = []
    if normalized.shape[1:] != (8,) or physical.shape[1:] != (8,):
        reason = ("ACTION_SHAPE_INVALID",)
        return RowClassification(classifications, tuple(reason for _ in range(rows)))
    semantic_reasons = tuple(sorted(set(str(value) for value in semantics_errors)))
    if semantic_reasons:
        return RowClassification(
            classifications, tuple(semantic_reasons for _ in range(rows))
        )
    if not math.isfinite(rotation_scale_rad) or rotation_scale_rad <= 0.0:
        raise ValueError("rotation scale must be finite and positive")

    normalized = normalized.astype(np.float64, copy=False)
    physical = physical.astype(np.float64, copy=False)
    for row in range(rows):
        row_reasons: list[str] = []
        norm = normalized[row]
        metric = physical[row]
        if not np.isfinite(norm).all() or not np.isfinite(metric).all():
            reasons.append(("NONFINITE_ACTION",))
            continue
        if not np.allclose(
            metric[:3],
            norm[:3] * contract.legacy_normalized_to_metric_scale_m,
            rtol=0.0,
            atol=1.0e-8,
        ):
            reasons.append(("TRANSLATION_SEMANTICS_MISMATCH",))
            continue
        if not np.allclose(
            metric[3:6], norm[3:6] * rotation_scale_rad, rtol=0.0, atol=1.0e-8
        ):
            reasons.append(("ROTATION_SEMANTICS_MISMATCH",))
            continue
        if not np.array_equal(metric[6:7], norm[6:7]):
            reasons.append(("ELBOW_SEMANTICS_MISMATCH",))
            continue
        if (
            float(metric[7]) not in (-1.0, 1.0)
            or float(norm[7]) not in (-1.0, 1.0)
            or float(metric[7]) != float(norm[7])
        ):
            reasons.append(("GRIPPER_SEMANTICS_INVALID",))
            continue

        translation_norm = float(np.linalg.norm(metric[:3]))
        if translation_norm > contract.maximum_final_xyz_norm_m:
            row_reasons.append("TRANSLATION_NORM_EXCEEDS_0P0045_M")
        if np.any(metric[3:6] != 0.0):
            row_reasons.append("ROTATION_NONZERO")
        if float(metric[6]) != 0.0:
            row_reasons.append("ELBOW_NONZERO")
        if row_reasons:
            classifications[row] = RowClass.REQUIRES_MIGRATION.value
            reasons.append(tuple(row_reasons))
        else:
            classifications[row] = RowClass.VALID_DIRECT.value
            reasons.append(())
    return RowClassification(classifications, tuple(reasons))


def audit_production_candidate_a_compatibility(
    *,
    production_asset: Path,
    candidate_asset: Path,
    mechanics_authority: Path,
) -> dict[str, Any]:
    """Audit the frozen Production-to-Candidate-A boundary property by property."""

    production_asset = Path(production_asset).resolve()
    candidate_asset = Path(candidate_asset).resolve()
    mechanics_authority = Path(mechanics_authority).resolve()
    production_hash = _sha256(production_asset) if production_asset.is_file() else None
    candidate_hash = _sha256(candidate_asset) if candidate_asset.is_file() else None
    authority = _load_json(mechanics_authority) if mechanics_authority.is_file() else {}
    boundary = authority.get("candidate_a_mutation_boundary")
    overrides = boundary.get("direct_overrides") if isinstance(boundary, Mapping) else None
    observed: dict[str, tuple[float, float]] = {}
    if isinstance(overrides, list):
        for item in overrides:
            if not isinstance(item, Mapping):
                continue
            try:
                observed[str(item["property"])] = (
                    float(item["production_root_value"]),
                    float(item["candidate_root_value"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    binding_pass = bool(
        production_hash == EXPECTED_PRODUCTION_SHA256
        and candidate_hash == EXPECTED_CANDIDATE_A_SHA256
        and isinstance(boundary, Mapping)
        and boundary.get("unexpected_override") is False
        and boundary.get("direct_override_count") == 4
        and observed == EXPECTED_CANDIDATE_OVERRIDES
    )

    properties = [
        {
            "property": "control_clock_and_units",
            "status": "PASS" if binding_pass else "FAIL",
            "evidence": "50Hz/0.02s, physics 0.002s, m/rad/N contract is asset-independent",
        },
        {
            "property": "right_arm_7dof_kinematics_and_ee_origin",
            "status": "PASS" if binding_pass else "FAIL",
            "evidence": "no Candidate-A override outside four passive joint4 limits",
        },
        {
            "property": "head_and_right_wrist_rgbd_geometry",
            "status": "PASS" if binding_pass else "FAIL",
            "evidence": "active arm/camera geometry inherited by the frozen mechanics authority",
        },
        {
            "property": "right_inner_joint4_lower_limit_deg",
            "status": "PARTIAL" if binding_pass else "FAIL",
            "production": -2.0,
            "candidate_a": -11.25,
            "evidence": "intentional passive-limit delta; passive state is absent from legacy rows",
        },
        {
            "property": "right_inner_joint4_upper_limit_deg",
            "status": "PARTIAL" if binding_pass else "FAIL",
            "production": 2.0,
            "candidate_a": 11.25,
            "evidence": "intentional passive-limit delta; passive state is absent from legacy rows",
        },
        {
            "property": "right_outer_joint4_lower_limit_deg",
            "status": "PARTIAL" if binding_pass else "FAIL",
            "production": -2.0,
            "candidate_a": -11.25,
            "evidence": "intentional passive-limit delta; passive state is absent from legacy rows",
        },
        {
            "property": "right_outer_joint4_upper_limit_deg",
            "status": "PARTIAL" if binding_pass else "FAIL",
            "production": 2.0,
            "candidate_a": 11.25,
            "evidence": "intentional passive-limit delta; passive state is absent from legacy rows",
        },
        {
            "property": "pre_action_open_hand_observation_and_close_onset_label",
            "status": "PASS" if binding_pass else "FAIL",
            "evidence": "legacy transition alignment records observation before applying g; first close onset is retained",
        },
        {
            "property": "post_close_grasp_trajectory",
            "status": "PARTIAL" if binding_pass else "FAIL",
            "evidence": "Production response may depend on changed passive limits; retained as context but excluded from Candidate-A usable subset",
        },
        {
            "property": "physical_pad_surface_pose",
            "status": "FAIL",
            "evidence": "unavailable in legacy rows; distal-link midpoint substitution is forbidden",
        },
    ]
    status = CompatibilityStatus.PARTIAL if binding_pass else CompatibilityStatus.FAIL
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "status": status.value,
        "production_asset": str(production_asset),
        "production_asset_sha256": production_hash,
        "candidate_a_asset": str(candidate_asset),
        "candidate_a_asset_sha256": candidate_hash,
        "mechanics_authority": str(mechanics_authority),
        "mechanics_authority_sha256": (
            _sha256(mechanics_authority) if mechanics_authority.is_file() else None
        ),
        "exact_four_property_boundary_pass": binding_pass,
        "properties": properties,
        "usable_subset": {
            "xyz_bc": "VALID_DIRECT rows through the first CLOSE onset only",
            "close_timing": "pre-CLOSE rows plus the first CLOSE_TRANSITION row",
            "post_close_context": "PRESERVED_NOT_CANDIDATE_A_TRAINABLE",
            "grasp_trajectory": "PARTIAL_CONTEXT_ONLY_NOT_FULLY_TRANSFER_QUALIFIED",
        },
    }


def _episode_id(source_sha256: str, group_path: str) -> str:
    suffix = group_path.strip("/").replace("/", "__")
    return f"{source_sha256[:16]}__{suffix}"


def _window(begin: int, end: int, close_row: int, kind: str) -> dict[str, Any]:
    before, after = WINDOW_SPECS[kind]
    first = max(begin, close_row + before)
    stop = min(end, close_row + after + 1)
    return {
        "kind": kind,
        "requested_relative_steps_inclusive": [before, after],
        "source_row_start": first,
        "source_row_stop_exclusive": stop,
        "actual_relative_steps_inclusive": [first - close_row, stop - 1 - close_row],
        "row_count": stop - first,
    }


def _close_events(gripper: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(gripper, dtype=np.float64).reshape(-1)
    valid = np.isin(values, (-1.0, 1.0))
    prior = np.concatenate((np.ones((1,), dtype=np.float64), values[:-1]))
    prior_valid = np.concatenate((np.ones((1,), dtype=np.bool_), valid[:-1]))
    onsets = np.flatnonzero(valid & prior_valid & (values == -1.0) & (prior == 1.0))
    events: list[dict[str, Any]] = []
    for event_number, onset in enumerate(onsets.tolist()):
        reopen = np.flatnonzero(values[onset + 1 :] == 1.0)
        persistent_stop = (
            len(values) if not reopen.size else onset + 1 + int(reopen[0])
        )
        events.append(
            {
                "event_id": f"close_{event_number:04d}",
                "close_onset_row": onset,
                "persistent_close_row_start": onset,
                "persistent_close_row_stop_exclusive": persistent_stop,
                "persistent_close_label": -1.0,
                "context_classes": list(CONTEXT_CLASSES),
                "windows": {
                    kind: _window(0, len(values), onset, kind)
                    for kind in WINDOW_SPECS
                },
            }
        )
    return events


def _source_lineage(handle: h5py.File, env: Mapping[str, Any]) -> dict[str, Any]:
    pad_fk = env.get("pad_fk_migration")
    if not isinstance(pad_fk, Mapping):
        pad_fk = {}
    source_path = str(pad_fk.get("source_path", ""))
    declared = str(
        pad_fk.get("source_sha256", handle.attrs.get("pad_fk_source_sha256", ""))
    )
    return {
        "source_path": source_path or None,
        "declared_source_sha256": declared or None,
        "source_exists": bool(source_path and Path(source_path).is_file()),
        "source_sha256_recomputed": False,
        "rgbd_storage": "HDF5_EXTERNAL_LINK_NO_DUPLICATION",
    }


def _build_episode_manifest(
    *,
    view_path: Path,
    view_sha256: str,
    handle: h5py.File,
    group_name: str,
    env: Mapping[str, Any],
    semantics: Sequence[str],
    candidate_boundary_pass: bool,
) -> dict[str, Any]:
    group = handle["data"][group_name]
    required = ("actions", "keyboard_physical_command")
    if any(name not in group for name in required):
        raise LegacyKeyboardDatasetError(
            f"{view_path}::/data/{group_name} missing action fields"
        )
    normalized = np.asarray(group["actions"])
    physical = np.asarray(group["keyboard_physical_command"])
    rotation_scale = float(
        env.get("environment_rotation_vector_scale_rad_per_normalized_unit", math.nan)
    )
    classified = classify_legacy_rows(
        normalized,
        physical,
        semantics_errors=semantics,
        rotation_scale_rad=rotation_scale if math.isfinite(rotation_scale) else 1.0,
    )
    rows = int(classified.classification.shape[0])
    episode_id = _episode_id(view_sha256, f"data/{group_name}")
    events = _close_events(physical[:, 7] if physical.shape == (rows, 8) else np.zeros(rows))
    first_close = events[0]["close_onset_row"] if events else rows - 1
    candidate_mask = np.zeros((rows,), dtype=np.bool_)
    if candidate_boundary_pass and rows:
        candidate_mask[: first_close + 1] = True
    valid_indices = np.flatnonzero(classified.valid_direct_mask).astype(np.int64)
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    xyz_eligible = np.flatnonzero(classified.valid_direct_mask & candidate_mask).astype(
        np.int64
    )
    event_direct_occurrences: dict[str, int] = {}
    event_candidate_occurrences: dict[str, int] = {}
    event_close_timing_occurrences: dict[str, int] = {}
    event_row_occurrences: dict[str, int] = {}
    event_context_occurrences: dict[str, dict[str, int]] = {}
    for kind in WINDOW_SPECS:
        direct_count = candidate_count = close_timing_count = 0
        row_occurrences = 0
        context_counts = {name: 0 for name in CONTEXT_CLASSES}
        for event in events:
            window = event["windows"][kind]
            start = int(window["source_row_start"])
            stop = int(window["source_row_stop_exclusive"])
            onset = int(event["close_onset_row"])
            direct_count += int(classified.valid_direct_mask[start:stop].sum())
            candidate_count += int(
                (classified.valid_direct_mask[start:stop] & candidate_mask[start:stop]).sum()
            )
            close_timing_count += int(
                (
                    (classified.classification[start:stop] != RowClass.REJECT.value)
                    & candidate_mask[start:stop]
                ).sum()
            )
            row_occurrences += stop - start
            relative = np.arange(start - onset, stop - onset, dtype=np.int64)
            context_counts["PRE_CLOSE"] += int(np.count_nonzero(relative < 0))
            context_counts["CLOSE_TRANSITION"] += int(np.count_nonzero(relative == 0))
            context_counts["POST_CLOSE_CONTEXT"] += int(np.count_nonzero(relative > 0))
        event_direct_occurrences[kind] = direct_count
        event_candidate_occurrences[kind] = candidate_count
        event_close_timing_occurrences[kind] = close_timing_count
        event_row_occurrences[kind] = row_occurrences
        event_context_occurrences[kind] = context_counts
    close_onset_classification_counts = {value.value: 0 for value in RowClass}
    for event in events:
        classification = str(
            classified.classification[int(event["close_onset_row"])]
        )
        close_onset_classification_counts[classification] += 1
    initial_pose = env.get("initial_pose_contract")
    initial_right_arm_q = (
        list(initial_pose.get("right_arm_q_rad", ()))
        if isinstance(initial_pose, Mapping)
        else []
    )
    manifest: dict[str, Any] = {
        "schema": EPISODE_MANIFEST_SCHEMA,
        "episode_id": episode_id,
        "split_key": hashlib.sha256(
            f"{view_sha256}:/data/{group_name}".encode("utf-8")
        ).hexdigest(),
        "source": {
            "view_hdf5": str(view_path),
            "view_hdf5_sha256": view_sha256,
            "group": f"/data/{group_name}",
            **_source_lineage(handle, env),
        },
        "source_asset_sha256": env.get("g2_usd_sha256"),
        "row_count": rows,
        "row_classification": classified.classification.tolist(),
        "row_reasons": [list(value) for value in classified.reasons],
        "row_classification_counts": classified.counts(),
        "valid_direct_row_indices": valid_indices.tolist(),
        "candidate_a_compatible_row_indices": candidate_indices.tolist(),
        "xyz_bc_eligible_row_indices": xyz_eligible.tolist(),
        "classification_contract": {
            "valid_direct": (
                "finite exact-known 8D semantics; physical XYZ equals normalized*0.0225m; "
                "rotation/elbow exactly zero; XYZ norm <=0.0045m; g exact +/-1"
            ),
            "requires_migration": "oversize XYZ, nonzero rotation, or nonzero elbow; excluded from XYZ BC",
            "reject": "unknown semantics, corrupt/nonfinite values, or normalized/physical mismatch",
            "silent_slicing": False,
            "silent_clipping": False,
            "silent_saturation": False,
            "direct_finger_target": False,
            "target_action": "[dx_m,dy_m,dz_m,g_high_level_persistent]",
            "causal_previous_action": (
                "[previous_dx_m,previous_dy_m,previous_dz_m,previous_g_close_binary]; "
                "source predecessor must be VALID_DIRECT and Candidate-A compatible"
            ),
            "current_gripper_state": "derived causal command state; OPEN=0,CLOSED=1",
        },
        "observation_authority": {
            "controlled_joint_order": list(EXPECTED_CONTROLLED_JOINT_ORDER),
            "right_arm_joint_order": list(RIGHT_ARM_JOINTS),
            "gripper_master_joint": RIGHT_GRIPPER_MASTER,
            "controlled_joint_position_storage": "relative_to_initial_pose_rad",
            "initial_right_arm_q_rad": initial_right_arm_q,
            "right_arm_joint_position_loader_output": "absolute_rad",
            "right_arm_joint_velocity_loader_output": "rad/s",
            "ee_origin_link": "gripper_r_center_link",
            "ee_frame": "robot_root",
            "metric_depth_layout": "[transition,height,width,1]_float_m",
            "depth_unit": "m",
            "depth_to_meter_scale": 1.0,
        },
        "candidate_a_subset": {
            "status": "PARTIAL" if candidate_boundary_pass else "FAIL",
            "rule": "rows through first CLOSE onset; all post-close rows excluded",
            "post_close_context_preserved": True,
            "post_close_context_trainable_for_candidate_a": False,
        },
        "close_events": events,
        "close_onset_classification_counts": close_onset_classification_counts,
        "window_row_occurrences": event_row_occurrences,
        "window_context_occurrences": event_context_occurrences,
        "window_valid_direct_occurrences": event_direct_occurrences,
        "window_candidate_xyz_eligible_occurrences": event_candidate_occurrences,
        "window_candidate_close_timing_eligible_occurrences": (
            event_close_timing_occurrences
        ),
        "episode_attrs": {
            "success": bool(group.attrs.get("success", False)),
            "seed": (
                int(group.attrs["seed"]) if "seed" in group.attrs else None
            ),
        },
    }
    manifest["classification_sha256"] = _canonical_sha256(
        {
            "classification": manifest["row_classification"],
            "reasons": manifest["row_reasons"],
        }
    )
    return manifest


def build_legacy_keyboard_dataset(
    paths: Iterable[Path],
    *,
    output: Path,
    production_asset: Path,
    candidate_asset: Path,
    mechanics_authority: Path,
) -> dict[str, Any]:
    """Build immutable JSON manifests and a machine-readable report."""

    files = tuple(sorted(Path(path).resolve() for path in paths))
    if not files:
        raise LegacyKeyboardDatasetError("no legacy keyboard HDF5 inputs")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        compatibility = audit_production_candidate_a_compatibility(
            production_asset=production_asset,
            candidate_asset=candidate_asset,
            mechanics_authority=mechanics_authority,
        )
        boundary_pass = bool(compatibility["exact_four_property_boundary_pass"])
        episodes: list[dict[str, Any]] = []
        inputs: list[dict[str, Any]] = []
        for path in files:
            if not path.is_file():
                raise LegacyKeyboardDatasetError(f"legacy keyboard input missing: {path}")
            view_sha = _sha256(path)
            with h5py.File(path, "r") as handle:
                env = _read_env_args(handle)
                semantics = legacy_semantics_errors(env)
                data = handle.get("data")
                if not isinstance(data, h5py.Group):
                    raise LegacyKeyboardDatasetError(f"legacy data group missing: {path}")
                input_record = {
                    "path": str(path),
                    "sha256": view_sha,
                    "dataset_schema": env.get("dataset_schema"),
                    "source_asset_sha256": env.get("g2_usd_sha256"),
                    "semantics_errors": list(semantics),
                    "episode_count": len(data),
                    "lineage": _source_lineage(handle, env),
                }
                inputs.append(input_record)
                for group_name in sorted(data):
                    manifest = _build_episode_manifest(
                        view_path=path,
                        view_sha256=view_sha,
                        handle=handle,
                        group_name=group_name,
                        env=env,
                        semantics=semantics,
                        candidate_boundary_pass=boundary_pass,
                    )
                    relative = Path("episodes") / manifest["episode_id"] / "EPISODE_MANIFEST.json"
                    manifest_path = staging / relative
                    _write_json(manifest_path, manifest)
                    episodes.append(
                        {
                            "episode_id": manifest["episode_id"],
                            "split_key": manifest["split_key"],
                            "manifest": str(relative),
                            "manifest_sha256": _sha256(manifest_path),
                            "row_count": manifest["row_count"],
                            "close_event_count": len(manifest["close_events"]),
                            "row_classification_counts": manifest[
                                "row_classification_counts"
                            ],
                            "close_onset_classification_counts": manifest[
                                "close_onset_classification_counts"
                            ],
                            "xyz_bc_eligible_row_count": len(
                                manifest["xyz_bc_eligible_row_indices"]
                            ),
                            "window_row_occurrences": manifest[
                                "window_row_occurrences"
                            ],
                            "window_context_occurrences": manifest[
                                "window_context_occurrences"
                            ],
                            "window_valid_direct_occurrences": manifest[
                                "window_valid_direct_occurrences"
                            ],
                            "window_candidate_xyz_eligible_occurrences": manifest[
                                "window_candidate_xyz_eligible_occurrences"
                            ],
                            "window_candidate_close_timing_eligible_occurrences": manifest[
                                "window_candidate_close_timing_eligible_occurrences"
                            ],
                        }
                    )
        ids = [item["episode_id"] for item in episodes]
        if len(ids) != len(set(ids)):
            raise LegacyKeyboardDatasetError("episode id collision")
        aggregate_counts = {
            value.value: sum(
                int(item["row_classification_counts"][value.value])
                for item in episodes
            )
            for value in RowClass
        }
        window_occurrences = {
            kind: sum(
                int(item["window_valid_direct_occurrences"][kind])
                for item in episodes
            )
            for kind in WINDOW_SPECS
        }
        close_onset_counts = {
            value.value: sum(
                int(item["close_onset_classification_counts"][value.value])
                for item in episodes
            )
            for value in RowClass
        }
        window_rows = {
            kind: sum(int(item["window_row_occurrences"][kind]) for item in episodes)
            for kind in WINDOW_SPECS
        }
        window_contexts = {
            kind: {
                context: sum(
                    int(item["window_context_occurrences"][kind][context])
                    for item in episodes
                )
                for context in CONTEXT_CLASSES
            }
            for kind in WINDOW_SPECS
        }
        candidate_window_occurrences = {
            kind: sum(
                int(item["window_candidate_xyz_eligible_occurrences"][kind])
                for item in episodes
            )
            for kind in WINDOW_SPECS
        }
        candidate_close_timing_occurrences = {
            kind: sum(
                int(item["window_candidate_close_timing_eligible_occurrences"][kind])
                for item in episodes
            )
            for kind in WINDOW_SPECS
        }
        xyz_bc_eligible_rows = sum(
            int(item["xyz_bc_eligible_row_count"]) for item in episodes
        )
        contract = canonical_keyboard_grasp_contract().as_dict()
        implementation = {
            "builder_module": str(Path(__file__).resolve()),
            "builder_module_sha256": _sha256(Path(__file__).resolve()),
            "shared_contract_module": str(
                Path(__file__).with_name("keyboard_grasp_contract.py").resolve()
            ),
            "shared_contract_module_sha256": _sha256(
                Path(__file__).with_name("keyboard_grasp_contract.py").resolve()
            ),
        }
        top_manifest = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "status": "PASS_WITH_CANDIDATE_A_PARTIAL_SUBSET",
            "contract": contract,
            "implementation": implementation,
            "inputs": inputs,
            "episode_count": len(episodes),
            "row_count": sum(int(item["row_count"]) for item in episodes),
            "close_event_count": sum(
                int(item["close_event_count"]) for item in episodes
            ),
            "row_classification_counts": aggregate_counts,
            "close_onset_classification_counts": close_onset_counts,
            "window_row_occurrences": window_rows,
            "window_context_occurrences": window_contexts,
            "window_valid_direct_occurrences": window_occurrences,
            "window_candidate_xyz_eligible_occurrences": candidate_window_occurrences,
            "window_candidate_close_timing_eligible_occurrences": (
                candidate_close_timing_occurrences
            ),
            "xyz_bc_eligible_row_count": xyz_bc_eligible_rows,
            "episodes": episodes,
            "production_candidate_a_compatibility": compatibility,
            "storage": {
                "rgbd_duplicated": False,
                "manifest_only": True,
                "source_access": "lazy HDF5 slice",
                "pad_surface_available": False,
                "midpoint_proxy_substituted": False,
            },
        }
        _write_json(staging / "DATASET_MANIFEST.json", top_manifest)
        report = {
            "schema": REPORT_SCHEMA,
            "status": top_manifest["status"],
            "dataset_manifest": str(output / "DATASET_MANIFEST.json"),
            "contract": contract,
            "implementation": implementation,
            "counts": {
                "input_files": len(inputs),
                "episodes": len(episodes),
                "rows": top_manifest["row_count"],
                "close_events": top_manifest["close_event_count"],
                "row_classification": aggregate_counts,
                "close_onset_classification": close_onset_counts,
                "window_row_occurrences": window_rows,
                "window_context_occurrences": window_contexts,
                "window_valid_direct_occurrences": window_occurrences,
                "window_candidate_xyz_eligible_occurrences": candidate_window_occurrences,
                "window_candidate_close_timing_eligible_occurrences": (
                    candidate_close_timing_occurrences
                ),
                "xyz_bc_eligible_rows": xyz_bc_eligible_rows,
            },
            "compatibility": compatibility,
            "blockers": [
                "REQUIRES_MIGRATION_ROWS_EXCLUDED_FROM_XYZ_BC",
                "PRODUCTION_POST_CLOSE_TRAJECTORY_NOT_CANDIDATE_A_TRANSFER_QUALIFIED",
                "PASSIVE_JOINT4_STATE_ABSENT_FROM_LEGACY_ROWS",
                "PAD_SURFACE_POSES_UNAVAILABLE_MIDPOINT_SUBSTITUTION_FORBIDDEN",
                "EXTERNAL_RGBD_LINEAGE_HASH_DECLARED_NOT_RECOMPUTED",
            ],
            "training_executed": False,
            "isaac_executed": False,
            "live_collection_executed": False,
        }
        _write_json(staging / "LEGACY_KEYBOARD_DATASET_REPORT.json", report)
        os.replace(staging, output)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


class LegacyKeyboardDataset:
    """Validate manifests and lazily load one close-event window."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = _load_json(self.root / "DATASET_MANIFEST.json")
        if self.manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
            raise LegacyKeyboardDatasetError("dataset manifest schema mismatch")
        contract = canonical_keyboard_grasp_contract().as_dict()
        if self.manifest.get("contract") != contract:
            raise LegacyKeyboardDatasetError("frozen keyboard contract mismatch")
        implementation = self.manifest.get("implementation")
        current_contract_module = Path(__file__).with_name(
            "keyboard_grasp_contract.py"
        ).resolve()
        if (
            not isinstance(implementation, Mapping)
            or implementation.get("builder_module_sha256")
            != _sha256(Path(__file__).resolve())
            or implementation.get("shared_contract_module_sha256")
            != _sha256(current_contract_module)
        ):
            raise LegacyKeyboardDatasetError("dataset builder/contract source hash mismatch")
        records = self.manifest.get("episodes")
        if not isinstance(records, list):
            raise LegacyKeyboardDatasetError("episode inventory missing")
        self._records: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise LegacyKeyboardDatasetError("episode record malformed")
            episode_id = str(record.get("episode_id", ""))
            path = (self.root / str(record.get("manifest", ""))).resolve()
            if not episode_id or episode_id in self._records:
                raise LegacyKeyboardDatasetError("duplicate/missing episode id")
            if not path.is_relative_to(self.root):
                raise LegacyKeyboardDatasetError("episode manifest escapes dataset root")
            if not path.is_file() or _sha256(path) != record.get("manifest_sha256"):
                raise LegacyKeyboardDatasetError("episode manifest hash mismatch")
            self._records[episode_id] = {**record, "path": path}

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def episode_manifest(self, episode_id: str) -> dict[str, Any]:
        try:
            record = self._records[episode_id]
        except KeyError as error:
            raise KeyError(f"unknown episode id: {episode_id}") from error
        manifest = _load_json(record["path"])
        if (
            manifest.get("schema") != EPISODE_MANIFEST_SCHEMA
            or manifest.get("episode_id") != episode_id
        ):
            raise LegacyKeyboardDatasetError("episode manifest identity mismatch")
        expected = _canonical_sha256(
            {
                "classification": manifest.get("row_classification"),
                "reasons": manifest.get("row_reasons"),
            }
        )
        if manifest.get("classification_sha256") != expected:
            raise LegacyKeyboardDatasetError("episode classification hash mismatch")
        return manifest

    def split_key(self, episode_id: str) -> str:
        return str(self.episode_manifest(episode_id)["split_key"])

    def load_window(
        self,
        episode_id: str,
        event_id: str,
        kind: str,
        *,
        fields: Sequence[str] = (),
    ) -> dict[str, Any]:
        if kind not in WINDOW_SPECS:
            raise ValueError(f"unknown window kind: {kind}")
        requested = tuple(str(value) for value in fields)
        forbidden = sorted(set(requested) & FORBIDDEN_LOADER_FIELDS)
        unknown = sorted(set(requested) - LAZY_SOURCE_FIELDS)
        if forbidden:
            raise LegacyKeyboardDatasetError(
                f"privileged/pad-proxy fields are forbidden: {forbidden}"
            )
        if unknown:
            raise LegacyKeyboardDatasetError(f"unknown lazy source fields: {unknown}")
        manifest = self.episode_manifest(episode_id)
        event = next(
            (
                value
                for value in manifest.get("close_events", [])
                if value.get("event_id") == event_id
            ),
            None,
        )
        if not isinstance(event, dict):
            raise KeyError(f"unknown close event: {episode_id}:{event_id}")
        window = event["windows"][kind]
        start = int(window["source_row_start"])
        stop = int(window["source_row_stop_exclusive"])
        close_row = int(event["close_onset_row"])
        source = manifest["source"]
        source_path = Path(source["view_hdf5"]).resolve()
        if not source_path.is_file() or _sha256(source_path) != source["view_hdf5_sha256"]:
            raise LegacyKeyboardDatasetError("legacy view HDF5 hash mismatch")
        with h5py.File(source_path, "r") as handle:
            group = handle[str(source["group"])]
            normalized = np.asarray(group["actions"][start:stop], dtype=np.float32)
            physical = np.asarray(
                group["keyboard_physical_command"][start:stop], dtype=np.float32
            )
            predecessor_start = max(0, start - 1)
            physical_with_predecessor = np.asarray(
                group["keyboard_physical_command"][predecessor_start:stop],
                dtype=np.float32,
            )
            arrays = {name: np.asarray(group[name][start:stop]) for name in requested}
            observation = np.asarray(
                group["teacher_observation"][start:stop], dtype=np.float32
            )
        rows = stop - start
        if normalized.shape != (rows, 8) or physical.shape != (rows, 8):
            raise LegacyKeyboardDatasetError("window action shape mismatch")
        if observation.shape != (rows, 59):
            raise LegacyKeyboardDatasetError("window observation shape mismatch")
        row_class = np.asarray(
            manifest["row_classification"][start:stop], dtype="U20"
        )
        valid_mask = row_class == RowClass.VALID_DIRECT.value
        row_valid_mask = row_class != RowClass.REJECT.value
        candidate_global = np.zeros((int(manifest["row_count"]),), dtype=np.bool_)
        candidate_global[
            np.asarray(manifest["candidate_a_compatible_row_indices"], dtype=np.int64)
        ] = True
        candidate_mask = candidate_global[start:stop]
        xyz_mask = valid_mask & candidate_mask
        close_timing_mask = row_valid_mask & candidate_mask
        relative = np.arange(start - close_row, stop - close_row, dtype=np.int64)
        context = np.where(
            relative < 0,
            "PRE_CLOSE",
            np.where(relative == 0, "CLOSE_TRANSITION", "POST_CLOSE_CONTEXT"),
        )
        valid_local = np.flatnonzero(valid_mask).astype(np.int64)
        xyz_local = np.flatnonzero(xyz_mask).astype(np.int64)
        source_rows = np.arange(start, stop, dtype=np.int64)

        previous_action = np.zeros((rows, 4), dtype=np.float32)
        previous_valid = np.zeros((rows,), dtype=np.bool_)
        previous_source_rows = source_rows - 1
        global_classification = np.asarray(
            manifest["row_classification"], dtype="U20"
        )
        for local_index, source_row in enumerate(source_rows.tolist()):
            if source_row == 0:
                # Episode reset authority is OPEN.  Canonical g is a CLOSE
                # binary, hence the exact causal padding value is zero.
                previous_valid[local_index] = True
                continue
            predecessor = source_row - 1
            if (
                global_classification[predecessor] == RowClass.VALID_DIRECT.value
                and candidate_global[predecessor]
            ):
                physical_index = predecessor - predecessor_start
                predecessor_action = physical_with_predecessor[physical_index]
                previous_action[local_index, :3] = predecessor_action[:3]
                previous_action[local_index, 3] = float(predecessor_action[7] < 0.0)
                previous_valid[local_index] = True
        hidden_reset = ~previous_valid
        if rows:
            # Every independently loaded event window is a fresh GRU sequence.
            # Later resets additionally expose an invalid predecessor.
            hidden_reset[0] = True

        observation_authority = manifest.get("observation_authority")
        if not isinstance(observation_authority, Mapping):
            raise LegacyKeyboardDatasetError("episode observation authority missing")
        controlled_joint_order = tuple(
            observation_authority.get("controlled_joint_order", ())
        )
        right_arm_initial_q = np.asarray(
            observation_authority.get("initial_right_arm_q_rad", ()),
            dtype=np.float32,
        )
        if (
            controlled_joint_order != EXPECTED_CONTROLLED_JOINT_ORDER
            or right_arm_initial_q.shape != (7,)
            or not np.isfinite(right_arm_initial_q).all()
            or observation_authority.get("metric_depth_layout")
            != "[transition,height,width,1]_float_m"
            or observation_authority.get("depth_unit") != "m"
            or observation_authority.get("depth_to_meter_scale") != 1.0
        ):
            raise LegacyKeyboardDatasetError("episode observation authority mismatch")
        right_arm_q = observation[:, 0:7] + right_arm_initial_q[None, :]

        def target(indices: np.ndarray) -> np.ndarray:
            if not len(indices):
                return np.empty((0, 4), dtype=np.float32)
            return np.concatenate(
                (physical[indices, :3], physical[indices, 7:8]), axis=1
            ).astype(np.float32, copy=False)

        return {
            "episode_id": episode_id,
            "split_key": manifest["split_key"],
            "event_id": event_id,
            "kind": kind,
            "source_row_index": source_rows,
            "relative_control_step": relative,
            "context_class": context,
            "row_classification": row_class,
            "row_valid_mask": row_valid_mask,
            "valid_direct_mask": valid_mask,
            "candidate_a_compatible_mask": candidate_mask,
            "xyz_bc_eligible_mask": xyz_mask,
            "close_timing_eligible_mask": close_timing_mask,
            "persistent_gripper_label": physical[:, 7].copy(),
            "previous_policy_action_source_row_index": previous_source_rows,
            "previous_policy_action_4d_metric_root_m": previous_action,
            "previous_policy_action_valid_mask": previous_valid,
            "previous_policy_action_hidden_reset_mask": hidden_reset,
            "hidden_reset_mask": hidden_reset.copy(),
            "current_gripper_state": previous_action[:, 3:4].copy(),
            "valid_direct_local_indices": valid_local,
            "valid_direct_target_action_4d_metric_root_m": target(valid_local),
            "xyz_bc_eligible_local_indices": xyz_local,
            "xyz_bc_target_action_4d_metric_root_m": target(xyz_local),
            "controlled_joint_position_relative_rad": observation[:, 0:8].copy(),
            "controlled_joint_velocity_rad_s": observation[:, 8:16].copy(),
            "controlled_joint_order": controlled_joint_order,
            "right_arm_joint_position_rad": right_arm_q.astype(
                np.float32, copy=False
            ),
            "right_arm_joint_velocity_rad_s": observation[:, 8:15].copy(),
            "ee_pose_robot_root_m_xyzw": observation[:, 16:23].copy(),
            "depth_unit": "m",
            "depth_to_meter_scale": 1.0,
            "source_fields": arrays,
            "pad_surface_available": False,
            "midpoint_proxy_substituted": False,
            "direct_finger_target_exposed": False,
        }


__all__ = [
    "COMPATIBILITY_SCHEMA",
    "DATASET_MANIFEST_SCHEMA",
    "EPISODE_MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "CompatibilityStatus",
    "LegacyKeyboardDataset",
    "LegacyKeyboardDatasetError",
    "RowClass",
    "RowClassification",
    "audit_production_candidate_a_compatibility",
    "build_legacy_keyboard_dataset",
    "classify_legacy_rows",
    "legacy_semantics_errors",
]
