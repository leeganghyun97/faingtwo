# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical keyboard-v3 grasp demonstrations and direct GRU collation.

The v3 format is intentionally not a migration format.  It stores the current
four-dimensional policy action ``[dx, dy, dz, g]`` in robot-root metres and
can be consumed by :mod:`human_grasp_gru_bc` without ever materialising a
legacy 7-D/8-D action.  Planner and privileged values live in disjoint HDF5
groups and are never returned as student inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .human_grasp_gru_bc import (
    CLOSE_EDGE_TARGET,
    HumanGraspGRUConfig,
    HumanGraspSequenceInputs,
    HumanGraspSequenceTargets,
    PrivilegedFeasibilityTargets,
    split_episode_ids,
)
from .keyboard_grasp_contract import (
    LOCAL_GRASP_PHASE_CODE,
    canonical_keyboard_grasp_contract,
    classify_local_grasp_phase,
    nominal_grasp_translation_residual_m,
)
from .privileged_contact_contract import (
    PrivilegedFeasibilityThresholds,
    SimulationContactRaw,
    SimulationContactThresholds,
    close_window_mask_ms,
    derive_privileged_feasibility,
    derive_privileged_grasp_labels,
    human_privileged_confusion,
)
from geniesim.rl.isaaclab.g2_policy_branch.omnipicker_product_contract import (
    OMNIPICKER_MANUAL_URL,
    OMNIPICKER_PRODUCT_CONTRACT,
)
from geniesim.rl.isaaclab.g2_policy_branch.rgbd_logging_contract import (
    CAMERA_TIMESTAMP_SOURCE,
    DEPTH_RAW_UNIT,
    DEPTH_TO_METER_SCALE,
    RGBD_EVIDENCE_SCHEMA,
    RGBDLoggingContractError,
    raw_depth_and_validity,
    validate_camera_calibration_metadata,
)


KEYBOARD_V3_SCHEMA = "keyboard_v3"
KEYBOARD_V3_FILE_SCHEMA = (
    "g2_keyboard_v3_hdf5_v6_wrist_actor_boolean_contact_privileged_left_arm_down_v2"
)
KEYBOARD_V3_VALIDATION_SCHEMA = "g2_keyboard_v3_episode_validation_v5"
KEYBOARD_V3_V1_AGGREGATE_SCHEMA = (
    "g2_keyboard_v3_v1_data_demo_layout_v1_exact_4d_semantics"
)
KEYBOARD_V3_RGBD_TIMESTAMP_SOURCE = "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE"
# Isaac exposes the camera acquisition clock as float32.  Adjacent valid
# 25-Hz captures observed on the frozen runtime differ from an ideal 40-ms
# increment by up to 1.87 us.  This tolerance is deliberately smaller than a
# physics step (2 ms) and a control step (20 ms), so it accepts clock
# quantization without accepting a missed/delayed acquisition.
RGBD_ACQUISITION_INCREMENT_TOLERANCE_S = 5.0e-6
KEYBOARD_V3_CAMERA_NAMES = ("head", "right_wrist")
KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES = ("right_wrist",)
KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES = ("head",)
KEYBOARD_V3_OPERATOR_VIEW_NAMES = ("perspective", "head", "right_wrist")
KEYBOARD_V3_V2_BASELINE_METADATA = {
    "baseline_variant": "LEGACY_CANDIDATE_A_LEFT_ARM_DOWN_V2",
    "left_arm_posture_id": "G2_LEFT_ARM_VERTICAL_DOWN_STRAIGHT_V2",
    "pregrasp_init_source": "CUROBO_DATASET",
    "head_camera": "ENABLED",
    "wrist_camera": "ENABLED",
    "control_source": "TERMINAL_KEYBOARD",
    "omnipicker_product_model": "OmniPicker",
    "omnipicker_manual_authority_url": OMNIPICKER_MANUAL_URL,
    "omnipicker_manual_hardware_version": "1.2",
    "omnipicker_pcba_version": "UNRESOLVED_20_OR_30",
    "omnipicker_firmware_version": "UNRESOLVED_RUNTIME_DEVICE",
    "omnipicker_maximum_gripping_force_n": 30.0,
    "omnipicker_physical_pad_touch_sensor_available": False,
    "omnipicker_protocol_force_feedback_semantics": (
        "CURRENT_MOTOR_TORQUE_NORMALIZED_0_TO_FF"
    ),
    "sim_contact_telemetry_role": (
        "PRIVILEGED_TEACHER_LABEL_AND_M2_DIAGNOSTIC_ONLY"
    ),
}
KEYBOARD_V3_STUDENT_FIELDS = (
    "observations/right_wrist_rgb",
    "observations/right_wrist_depth_m",
    "observations/right_wrist_depth_valid",
    "observations/ee_position_root_m",
    "observations/ee_quat_root_xyzw",
    "observations/arm_q_rad",
    "observations/arm_qd_rad_s",
    "observations/gripper_state",
    "actions/previous_action_4d",
)
KEYBOARD_V3_RECORDED_NON_ACTOR_FIELDS = (
    "observations/head_rgb",
    "observations/head_depth_m",
    "observations/head_depth_valid",
    "observations/head_depth_raw_m",
    "observations/head_depth_source_valid",
    "observations/head_camera_pose_root_m_xyzw",
    "observations/right_wrist_depth_raw_m",
    "observations/right_wrist_depth_source_valid",
    "observations/right_wrist_camera_pose_root_m_xyzw",
)
KEYBOARD_V3_PRIVILEGED_FIELDS = (
    "privileged/cube_center_root_m",
    "privileged/left_pad_surface_position_root_m",
    "privileged/right_pad_surface_position_root_m",
    "privileged/contact_force_left_n",
    "privileged/contact_force_right_n",
    "privileged/total_normal_force_n",
    "privileged/contact_force_valid",
    "privileged/contact_measurement_valid",
    "privileged/contact_timestamp_s",
    "privileged/contact_control_step",
    "privileged/left_contact",
    "privileged/right_contact",
    "privileged/contact",
    "privileged/exact_inner_contact",
    "privileged/exact_outer_contact",
    "privileged/bilateral_contact",
    "privileged/stable_grasp",
    "privileged/physical_lift",
    "privileged/forbidden_collision",
    "privileged/forbidden_collision_valid",
    "privileged/safety_violation",
    "privileged/safety_measurement_valid",
    "privileged/cube_linear_velocity_root_m_s",
    "privileged/cube_pose_root_m_xyzw",
    "privileged/pad_center_relative_to_cube_velocity_root_m_s",
    "privileged/gripper_command",
    "privileged/gripper_master_position_rad",
    "privileged/gripper_master_velocity_rad_s",
    "privileged/right_inner_distal_link_pose_root_m_xyzw",
    "privileged/right_outer_distal_link_pose_root_m_xyzw",
    "planner/nominal_grasp_pose_root_m_xyzw",
    "planner/near_grasp_pose_root_m_xyzw",
    "planner/target_residual_m",
    "planner/current_nominal_grasp_residual_m",
    "planner/lateral_error_m",
    "planner/approach_cosine",
    "planner/ee_speed_m_s",
    "planner/phase_code",
    "planner/local_grasp_phase_code",
)
KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES = {
    "contact_force_left_n": (),
    "contact_force_right_n": (),
    "total_normal_force_n": (),
    "contact_force_valid": (2,),
    "contact_measurement_valid": (),
    "contact_timestamp_s": (),
    "contact_control_step": (),
    "left_contact": (),
    "right_contact": (),
    "contact": (),
    "exact_inner_contact": (),
    "exact_outer_contact": (),
    "bilateral_contact": (),
    "stable_grasp": (),
    "physical_lift": (),
    "slip_speed_m_s": (),
    "forbidden_collision": (),
    "forbidden_collision_valid": (),
    "safety_violation": (),
    "safety_measurement_valid": (),
    "cube_linear_velocity_root_m_s": (3,),
    "cube_pose_root_m_xyzw": (7,),
    "pad_center_relative_to_cube_velocity_root_m_s": (3,),
    "gripper_command": (),
    "gripper_master_position_rad": (),
    "gripper_master_velocity_rad_s": (),
    "right_inner_distal_link_pose_root_m_xyzw": (7,),
    "right_outer_distal_link_pose_root_m_xyzw": (7,),
}
FAILURE_TYPES = frozenset(
    {
        "NONE",
        "EARLY_CLOSE",
        "LATE_CLOSE",
        "LATERAL_MISALIGNMENT",
        "HEIGHT_MISALIGNMENT",
        "OPERATOR_ABORT",
        "FAIL_EARLY_CLOSE",
        "FAIL_LATE_CLOSE",
        "FAIL_LATERAL_MISALIGNMENT",
        "FAIL_HEIGHT_MISALIGNMENT",
        "FAIL_OPERATOR_ABORT",
        "TIMEOUT",
        "OTHER",
    }
)


class KeyboardV3DatasetError(ValueError):
    """Raised before malformed v3 data can enter an HDF5 file or learner."""


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - host dependency
        raise KeyboardV3DatasetError("h5py is required for keyboard-v3") from error
    return h5py


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def preprocess_depth_m(
    source_depth_m: Any, source_valid: Any, *, maximum_depth_m: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the deployable validity rule without metric-depth clipping."""

    depth = np.asarray(source_depth_m, dtype=np.float32)
    valid = np.asarray(source_valid)
    if depth.shape != valid.shape:
        raise KeyboardV3DatasetError("depth and source_valid shapes differ")
    if valid.dtype != np.bool_ and not np.all(np.isin(valid, (0, 1))):
        raise KeyboardV3DatasetError("depth source_valid must be boolean/binary")
    canonical_valid = (
        valid.astype(np.bool_, copy=False)
        & np.isfinite(depth)
        & (depth >= 0.0)
        & (depth <= float(maximum_depth_m))
    )
    canonical = np.where(canonical_valid, depth, np.float32(0.0)).astype(np.float32)
    return canonical, canonical_valid


def canonical_action_4d(value: Any) -> np.ndarray:
    """Validate one metric v3 action; exceeding 4.5 mm is rejected, not clipped."""

    action = np.asarray(value, dtype=np.float64)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise KeyboardV3DatasetError("keyboard-v3 action must be four finite values")
    bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
    if float(np.linalg.norm(action[:3])) > bound + 1.0e-12:
        raise KeyboardV3DatasetError("keyboard-v3 XYZ action exceeds 0.0045 m")
    if action[3] not in (0.0, 1.0):
        raise KeyboardV3DatasetError("keyboard-v3 g must be exact OPEN=0/CLOSE=1")
    return action.astype(np.float32)


def close_edge_from_state(close_state: Any, *, initial_state: bool = False) -> np.ndarray:
    state = np.asarray(close_state)
    if state.ndim != 1 or (
        state.dtype != np.bool_ and not np.all(np.isin(state, (0, 1)))
    ):
        raise KeyboardV3DatasetError("close_state must be a binary vector")
    state = state.astype(np.bool_, copy=False)
    previous = np.concatenate(
        (np.asarray([initial_state], dtype=np.bool_), state[:-1]), axis=0
    )
    return state & ~previous


@dataclass(frozen=True)
class KeyboardV3Episode:
    episode_id: str
    metadata: Mapping[str, Any]
    observations: Mapping[str, Any]
    actions: Mapping[str, Any]
    events: Mapping[str, Any]
    privileged: Mapping[str, Any]
    planner: Mapping[str, Any]
    time: Mapping[str, Any]


@dataclass(frozen=True)
class KeyboardV3Validation:
    episode_id: str
    passed: bool
    rows: int
    nonzero_micro_approach_rows: int
    close_event_count: int
    action_bound_violation_count: int
    student_privileged_input_count: int
    errors: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": KEYBOARD_V3_VALIDATION_SCHEMA,
            "episode_id": self.episode_id,
            "passed": self.passed,
            "training_eligible": self.passed,
            "rows": self.rows,
            "nonzero_micro_approach_rows": self.nonzero_micro_approach_rows,
            "close_event_count": self.close_event_count,
            "action_bound_violation_count": self.action_bound_violation_count,
            "student_privileged_input_count": self.student_privileged_input_count,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class KeyboardV3V1AggregateReceipt:
    """Durable receipt for the V1-style ``/data/demo_N`` compatibility view.

    Only the container layout is V1-compatible.  Actions deliberately remain
    the current typed 4-D metric contract and are never expanded into a fake
    legacy 8-D action.
    """

    path: str
    demo_id: str
    episode_id: str
    rows: int
    total_rows: int
    demo_count: int
    training_eligible: bool

    def payload(self) -> dict[str, Any]:
        return {
            "schema": KEYBOARD_V3_V1_AGGREGATE_SCHEMA,
            **self.__dict__,
        }


@dataclass(frozen=True)
class OfflineGraspLabelThresholds:
    """Analysis-only thresholds selected from historical raw distributions.

    Instances are explicit inputs to offline relabeling and are never embedded
    in the collector or runtime actor.  This prevents a provisional threshold
    from silently becoming production contact authority.
    """

    minimum_total_force_n: float
    maximum_total_force_n: float
    maximum_relative_speed_m_s: float
    stable_dwell_s: float
    distribution_receipt_sha256: str

    def __post_init__(self) -> None:
        numeric = (
            self.minimum_total_force_n,
            self.maximum_total_force_n,
            self.maximum_relative_speed_m_s,
            self.stable_dwell_s,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in numeric):
            raise KeyboardV3DatasetError("offline grasp thresholds must be finite/nonnegative")
        if self.maximum_total_force_n < self.minimum_total_force_n:
            raise KeyboardV3DatasetError("offline force band is inverted")
        if self.stable_dwell_s <= 0.0:
            raise KeyboardV3DatasetError("stable_dwell_s must be positive")
        receipt = self.distribution_receipt_sha256
        if len(receipt) != 64 or any(value not in "0123456789abcdef" for value in receipt):
            raise KeyboardV3DatasetError(
                "distribution_receipt_sha256 must be a lowercase SHA-256"
            )


def derive_offline_grasp_labels(
    episode: KeyboardV3Episode,
    *,
    thresholds: OfflineGraspLabelThresholds,
    cube_retained: Any,
    cube_retained_valid: Any,
    require_lift: bool = False,
) -> dict[str, Any]:
    """Derive provisional CONTACT/STABLE labels without changing raw data."""

    validation = validate_episode(episode)
    if not validation.passed:
        raise KeyboardV3DatasetError(
            "cannot derive labels from invalid episode: " + ",".join(validation.errors)
        )
    raw = SimulationContactRaw(
        timestamp_s=np.asarray(
            episode.privileged["contact_timestamp_s"], dtype=np.float64
        ),
        control_step=np.asarray(
            episode.privileged["contact_control_step"], dtype=np.int64
        ),
        left_contact=np.asarray(episode.privileged["left_contact"]),
        right_contact=np.asarray(episode.privileged["right_contact"]),
        left_normal_force_n=np.asarray(
            episode.privileged["contact_force_left_n"], dtype=np.float64
        ),
        right_normal_force_n=np.asarray(
            episode.privileged["contact_force_right_n"], dtype=np.float64
        ),
        left_pad_pose_root_m_xyzw=np.asarray(
            episode.privileged["right_inner_distal_link_pose_root_m_xyzw"],
            dtype=np.float64,
        ),
        right_pad_pose_root_m_xyzw=np.asarray(
            episode.privileged["right_outer_distal_link_pose_root_m_xyzw"],
            dtype=np.float64,
        ),
        cube_pose_root_m_xyzw=np.asarray(
            episode.privileged["cube_pose_root_m_xyzw"], dtype=np.float64
        ),
        relative_pad_cube_velocity_root_m_s=np.asarray(
            episode.privileged[
                "pad_center_relative_to_cube_velocity_root_m_s"
            ],
            dtype=np.float64,
        ),
        measurement_valid=np.asarray(
            episode.privileged["contact_measurement_valid"]
        ),
        lifted=np.asarray(episode.privileged["physical_lift"]),
    )
    labels = derive_privileged_grasp_labels(
        raw,
        thresholds=SimulationContactThresholds(
            minimum_total_force_n=thresholds.minimum_total_force_n,
            maximum_total_force_n=thresholds.maximum_total_force_n,
            maximum_relative_speed_m_s=thresholds.maximum_relative_speed_m_s,
            stable_dwell_s=thresholds.stable_dwell_s,
            distribution_receipt_sha256=thresholds.distribution_receipt_sha256,
        ),
        cube_retained=cube_retained,
        cube_retained_valid=cube_retained_valid,
        require_lift=require_lift,
    )
    return {
        "schema": "g2_keyboard_v3_offline_grasp_labels_v2_boolean_primary",
        "authority": "ANALYSIS_ONLY_NOT_PRODUCTION_LOCKED",
        "primary_privileged_learning_signal": "BOOLEAN_CONTACT",
        "force_role": "RAW_DIAGNOSTIC_SAFETY_AUXILIARY_ONLY",
        "thresholds": {
            "minimum_total_force_n": float(thresholds.minimum_total_force_n),
            "maximum_total_force_n": float(thresholds.maximum_total_force_n),
            "maximum_relative_speed_m_s": float(thresholds.maximum_relative_speed_m_s),
            "stable_dwell_s": float(thresholds.stable_dwell_s),
            "distribution_receipt_sha256": thresholds.distribution_receipt_sha256,
        },
        "left_pad_contact": raw.left_contact.copy(),
        "right_pad_contact": raw.right_contact.copy(),
        "contact": labels.contact,
        "bilateral_contact": labels.bilateral_contact,
        "force_ok": labels.force_ok,
        "stable": labels.stable,
        "bilateral_dwell_s": labels.bilateral_dwell_s,
        "cube_retained": labels.cube_retained,
        "grasp_success_rows": labels.grasp_success,
        "grasp_success": bool(labels.grasp_success.any()),
        "replay_class": labels.replay_class,
        "first_contact_step": raw.first_contact_step,
        "first_contact_timestamp_s": raw.first_contact_timestamp_s,
    }


def build_privileged_feasibility_targets(
    episode: KeyboardV3Episode,
    *,
    pad_to_cube_distance_m: Any,
    lateral_alignment_error_m: Any,
    orientation_error_rad: Any,
    relative_speed_m_s: Any,
    geometry_valid: Any,
    geometry_receipt_sha256: str,
    thresholds: PrivilegedFeasibilityThresholds,
) -> tuple[PrivilegedFeasibilityTargets, dict[str, int]]:
    """Build the geometry auxiliary label without changing HUMAN_CLOSE.

    Geometry inputs must come from an approved pad-surface calibration.  Raw
    distal-link poses recorded by keyboard-v3 are intentionally insufficient
    for this function and cannot be promoted through an implicit midpoint
    proxy.  The returned confusion matrix uses the exact operator ``k`` edge,
    not the persistent closed state, and disagreement rows are retained.
    """

    validation = validate_episode(episode)
    if not validation.passed:
        raise KeyboardV3DatasetError(
            "cannot derive privileged feasibility from invalid episode: "
            + ",".join(validation.errors)
        )
    metadata = dict(episode.metadata)
    if (
        metadata.get("pad_surface_valid") is not True
        or metadata.get("pad_calibration_verified") is not True
        or metadata.get("pad_calibration_schema") != "g2_pad_surface_calibration_v1"
    ):
        raise KeyboardV3DatasetError(
            "privileged feasibility requires verified pad-surface calibration"
        )
    if geometry_receipt_sha256 != metadata.get("pad_geometry_receipt_sha256"):
        raise KeyboardV3DatasetError("pad geometry receipt does not match episode")
    feasible, valid = derive_privileged_feasibility(
        pad_to_cube_distance_m=pad_to_cube_distance_m,
        lateral_alignment_error_m=lateral_alignment_error_m,
        orientation_error_rad=orientation_error_rad,
        relative_speed_m_s=relative_speed_m_s,
        valid=geometry_valid,
        thresholds=thresholds,
    )
    close_event = np.asarray(episode.events["close_edge"], dtype=np.bool_)
    if feasible.shape != close_event.shape:
        raise KeyboardV3DatasetError(
            "privileged feasibility rows do not match HUMAN_CLOSE rows"
        )
    matrix = human_privileged_confusion(close_event, feasible, valid)
    target = PrivilegedFeasibilityTargets(
        feasible_target=torch.as_tensor(feasible, dtype=torch.float32).view(
            1, feasible.size, 1
        ),
        valid_mask=torch.as_tensor(valid, dtype=torch.bool).view(1, valid.size),
        geometry_receipt_sha256=geometry_receipt_sha256,
        threshold_receipt_sha256=thresholds.distribution_receipt_sha256,
        source_schema=KEYBOARD_V3_FILE_SCHEMA,
        calibration_schema="g2_pad_surface_calibration_v1",
    )
    return target, matrix


def keyboard_close_window_mask(
    episode: KeyboardV3Episode,
    *,
    before_ms: float = 200.0,
    after_ms: float = 200.0,
) -> np.ndarray:
    """Return the exact time-based window around the operator ``k`` edge."""

    return close_window_mask_ms(
        timestamp_s=episode.time["timestamp_s"],
        close_edge=episode.events["close_edge"],
        before_ms=before_ms,
        after_ms=after_ms,
    )


def _array(group: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in group:
        raise KeyboardV3DatasetError(f"missing field: {name}")
    return np.asarray(group[name])


def validate_episode(episode: KeyboardV3Episode) -> KeyboardV3Validation:
    """Validate all cross-field contracts for a single in-memory episode."""

    errors: list[str] = []
    contract = canonical_keyboard_grasp_contract()
    if not isinstance(episode.episode_id, str) or not episode.episode_id:
        errors.append("EPISODE_ID_INVALID")
    metadata = dict(episode.metadata)
    # Historical v3 episodes remain readable as historical evidence.  A V2
    # episode is a distinct baseline family and must carry the complete
    # robot/cube-pair provenance so V1/V2 data cannot merge silently.
    v2_keys = set(KEYBOARD_V3_V2_BASELINE_METADATA) | {
        "pregrasp_sample_id",
        "cube_sample_id",
    }
    if v2_keys.intersection(metadata):
        for name, expected in KEYBOARD_V3_V2_BASELINE_METADATA.items():
            if metadata.get(name) != expected:
                errors.append(f"METADATA_{name.upper()}_MISMATCH")
        if not isinstance(metadata.get("pregrasp_sample_id"), str) or not metadata.get(
            "pregrasp_sample_id"
        ):
            errors.append("METADATA_PREGRASP_SAMPLE_ID_MISSING")
        if metadata.get("cube_sample_id") != metadata.get("pregrasp_sample_id"):
            errors.append("METADATA_ROBOT_CUBE_SAMPLE_PAIR_MISMATCH")
    expected_meta = {
        "schema_version": KEYBOARD_V3_SCHEMA,
        "action_dim": 4,
        "action_frame": contract.geometry_frame,
        "action_unit": contract.action_xyz_unit,
        "joint_position_unit": contract.angle_unit,
        "joint_velocity_unit": "rad/s",
        "depth_unit": "m",
        "control_hz": contract.control_hz,
        "control_dt_s": contract.policy_dt_s,
        "dataset_row_hz": contract.dataset_row_hz,
        "physics_hz": contract.physics_hz,
        "physics_dt_s": contract.physics_dt_s,
        "rgbd_hz": contract.rgbd_acquisition_hz,
        "rgbd_dt_s": contract.rgbd_dt_s,
        "rgbd_timestamp_source": KEYBOARD_V3_RGBD_TIMESTAMP_SOURCE,
        "rgbd_observation_alignment": (
            "LATEST_VALID_ACQUISITION_REFERENCED_BY_FRAME_ID"
        ),
        "camera_capture_contract": (
            "ACTUAL_ACQUISITION_TIMESTAMP_HELD_ACROSS_TWO_CONTROL_ROWS"
        ),
        "ee_frame": contract.ee_origin_link,
        "orientation_policy": "FIXED",
        "elbow_policy": "NON_POLICY",
        "student_privileged_input_count": 0,
        "gru_actor_camera_names": list(KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES),
        "recorded_non_actor_camera_names": list(KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES),
        "outcome_alignment": "POST_ACTION_TRANSITION_T_PLUS_1",
        "outcome_label_authority": "RAW_TELEMETRY_OFFLINE_DERIVATION_ONLY",
        "human_close_label_authority": "KEYBOARD_K_CLOSE_EDGE",
        "privileged_feasible_label_authority": "VERIFIED_SIM_GEOMETRY_ONLY",
        "human_and_privileged_labels_merged": False,
        "primary_privileged_learning_signal": "BOOLEAN_CONTACT",
        "force_learning_role": "RAW_DIAGNOSTIC_SAFETY_AUXILIARY_ONLY",
        "contact_channel_names": ["right_inner", "right_outer"],
        "contact_channel_source": "ISAAC_SIM_CONTACT_SENSOR_PRIVILEGED_ONLY",
        "contact_boolean_extraction": "FILTERED_FORCE_MATRIX_W_GREATER_THAN_1N_TO_BOOLEAN",
        "contact_boolean_threshold_n": 1.0,
        "hardware_pad_touch_sensor_available": False,
        "oem_maximum_gripping_force_n": (
            OMNIPICKER_PRODUCT_CONTRACT.maximum_gripping_force_n
        ),
        "oem_gripping_force_metric": (
            OMNIPICKER_PRODUCT_CONTRACT.m2_gripping_force_metric
        ),
        "sim_contact_to_oem_force_mapping": (
            OMNIPICKER_PRODUCT_CONTRACT.sim_contact_to_oem_gripping_force_mapping
        ),
        "protocol_force_feedback_semantics": (
            OMNIPICKER_PRODUCT_CONTRACT.protocol_force_feedback_semantics
        ),
        "student_contact_input_count": 0,
        "pad_pose_authority": "DISTAL_LINK_FRAME_RAW_NOT_CALIBRATED_PAD_SURFACE",
        "future_outcome_horizon_steps": 64,
    }
    for name, expected in expected_meta.items():
        if metadata.get(name) != expected:
            errors.append(f"METADATA_{name.upper()}_MISMATCH")
    display_names = tuple(
        str(value) for value in metadata.get("operator_display_only_view_names", ())
    )
    if display_names not in (
        ("perspective", "head"),
        KEYBOARD_V3_OPERATOR_VIEW_NAMES,
    ):
        errors.append("METADATA_OPERATOR_DISPLAY_ONLY_VIEW_NAMES_MISMATCH")
    if int(metadata.get("recorded_camera_count", -1)) != len(
        KEYBOARD_V3_CAMERA_NAMES
    ):
        errors.append("METADATA_RECORDED_CAMERA_COUNT_MISMATCH")
    camera_names = tuple(
        str(value) for value in metadata.get("recorded_camera_names", ())
    )
    if camera_names != KEYBOARD_V3_CAMERA_NAMES:
        errors.append("METADATA_RECORDED_CAMERA_NAMES_MISMATCH")
    if int(metadata.get("operator_view_count", -1)) != len(
        KEYBOARD_V3_OPERATOR_VIEW_NAMES
    ):
        errors.append("METADATA_OPERATOR_VIEW_COUNT_MISMATCH")
    operator_view_names = tuple(
        str(value) for value in metadata.get("operator_view_names", ())
    )
    if operator_view_names != KEYBOARD_V3_OPERATOR_VIEW_NAMES:
        errors.append("METADATA_OPERATOR_VIEW_NAMES_MISMATCH")
    if metadata.get("failure_type") not in FAILURE_TYPES:
        errors.append("FAILURE_TYPE_INVALID")
    if bool(metadata.get("success", False)) != (metadata.get("failure_type") == "NONE"):
        errors.append("SUCCESS_FAILURE_TYPE_INCONSISTENT")
    rgbd_hz = metadata.get("rgbd_hz")
    if rgbd_hz != contract.rgbd_acquisition_hz:
        errors.append("RGBD_HZ_MUST_BE_25")
    camera_evidence_active = bool(metadata.get("camera_evidence_extension_active", False))
    if camera_evidence_active:
        if metadata.get("rgbd_evidence_schema") != RGBD_EVIDENCE_SCHEMA:
            errors.append("RGBD_EVIDENCE_SCHEMA_MISMATCH")
        if metadata.get("depth_raw_unit") != DEPTH_RAW_UNIT:
            errors.append("DEPTH_RAW_UNIT_MISMATCH")
        try:
            if float(metadata.get("depth_to_meter_scale")) != DEPTH_TO_METER_SCALE:
                errors.append("DEPTH_TO_METER_SCALE_MISMATCH")
        except (TypeError, ValueError):
            errors.append("DEPTH_TO_METER_SCALE_MISMATCH")
        for camera_name in KEYBOARD_V3_CAMERA_NAMES:
            prefix = f"{camera_name}_camera_"
            calibration = {
                "camera_name": metadata.get(prefix + "camera_name"),
                "source_scene_key": metadata.get(prefix + "source_scene_key"),
                "source_frame_id": metadata.get(prefix + "source_frame_id"),
                "intrinsics_3x3": metadata.get(prefix + "intrinsics_3x3"),
                "image_width": metadata.get(prefix + "image_width"),
                "image_height": metadata.get(prefix + "image_height"),
                "extrinsic_reference_frame": metadata.get(prefix + "extrinsic_reference_frame"),
                "pose_convention": metadata.get(prefix + "pose_convention"),
                "timestamp_source": metadata.get(prefix + "timestamp_source"),
            }
            try:
                validate_camera_calibration_metadata(calibration, camera_name=camera_name)
            except (RGBDLoggingContractError, TypeError, ValueError):
                errors.append(f"{camera_name.upper()}_CAMERA_CALIBRATION_INVALID")
    local_contract_keys = {
        "handoff_distance_m",
        "handoff_band_m",
        "default_handoff_m",
        "local_grasp_band_m",
        "local_grasp_distance_reference",
        "local_grasp_distance_is_student_input",
        "local_grasp_phase_names",
    }
    local_contract_present = bool(local_contract_keys.intersection(metadata))
    collection_target_active = bool(
        metadata.get("collection_target_authority_active", False)
    )
    if collection_target_active and display_names != KEYBOARD_V3_OPERATOR_VIEW_NAMES:
        errors.append("COLLECTION_OPERATOR_THREE_VIEW_CONTRACT_MISMATCH")
    collection_targets_m = tuple(value / 1000.0 for value in range(16, 23))
    backoff = float("nan")
    try:
        backoff = float(metadata.get("curobo_backoff_m"))
        approved_backoffs = (
            collection_targets_m
            if collection_target_active
            else contract.curobo_handoff_band_m
            if local_contract_present
            else (0.010, 0.015, 0.020, *contract.curobo_handoff_band_m)
        )
        if backoff not in approved_backoffs:
            errors.append("CUROBO_BACKOFF_INVALID")
    except (TypeError, ValueError):
        errors.append("CUROBO_BACKOFF_INVALID")
    if collection_target_active:
        expected_target_mm = int(round(backoff * 1000.0)) if np.isfinite(backoff) else -1
        if metadata.get("collection_target_residual_mm") != expected_target_mm:
            errors.append("COLLECTION_TARGET_MM_BACKOFF_MISMATCH")
        if metadata.get("collection_target_residual_m") != backoff:
            errors.append("COLLECTION_TARGET_M_BACKOFF_MISMATCH")
        if metadata.get("collection_target_reference") != (
            "CUROBO_NOMINAL_GRASP_POSE_TRANSLATION_RESIDUAL_NOT_PAD_SURFACE"
        ):
            errors.append("COLLECTION_TARGET_REFERENCE_MISMATCH")
    start_offset = np.asarray(metadata.get("start_offset_xyz_m"), dtype=np.float64)
    if start_offset.shape != (3,) or not np.isfinite(start_offset).all():
        errors.append("START_OFFSET_INVALID")
    if local_contract_present:
        if not local_contract_keys.issubset(metadata):
            errors.append("LOCAL_GRASP_METADATA_INCOMPLETE")
        try:
            if float(metadata.get("handoff_distance_m")) != backoff:
                errors.append("HANDOFF_DISTANCE_BACKOFF_MISMATCH")
            if not np.array_equal(
                np.asarray(metadata.get("handoff_band_m"), dtype=np.float64),
                np.asarray(contract.curobo_handoff_band_m, dtype=np.float64),
            ):
                errors.append("HANDOFF_BAND_MISMATCH")
            if float(metadata.get("default_handoff_m")) != contract.curobo_default_handoff_m:
                errors.append("DEFAULT_HANDOFF_MISMATCH")
            if not np.array_equal(
                np.asarray(metadata.get("local_grasp_band_m"), dtype=np.float64),
                np.asarray(contract.local_grasp_band_m, dtype=np.float64),
            ):
                errors.append("LOCAL_GRASP_BAND_MISMATCH")
            if metadata.get("local_grasp_distance_reference") != contract.local_grasp_distance_reference:
                errors.append("LOCAL_GRASP_DISTANCE_REFERENCE_MISMATCH")
            if bool(metadata.get("local_grasp_distance_is_student_input")):
                errors.append("LOCAL_GRASP_DISTANCE_STUDENT_LEAKAGE")
            if tuple(metadata.get("local_grasp_phase_names", ())) != tuple(
                LOCAL_GRASP_PHASE_CODE
            ):
                errors.append("LOCAL_GRASP_PHASE_NAMES_MISMATCH")
        except (TypeError, ValueError):
            errors.append("LOCAL_GRASP_METADATA_INVALID")
    try:
        action = _array(episode.actions, "action_4d").astype(np.float64)
        previous = _array(episode.actions, "previous_action_4d").astype(np.float64)
    except (KeyboardV3DatasetError, ValueError, TypeError) as error:
        action = np.zeros((0, 4), dtype=np.float64)
        previous = np.zeros((0, 4), dtype=np.float64)
        errors.append(str(error))
    rows = int(action.shape[0]) if action.ndim == 2 else 0
    if action.shape != (rows, 4) or previous.shape != (rows, 4) or rows == 0:
        errors.append("ACTION_SHAPE_INVALID")
    action_bound_violations = 0
    if rows:
        if not np.isfinite(action).all() or not np.isfinite(previous).all():
            errors.append("ACTION_NONFINITE")
        action_norm = np.linalg.norm(action[:, :3], axis=1)
        previous_norm = np.linalg.norm(previous[:, :3], axis=1)
        action_bound_violations = int(
            np.count_nonzero(action_norm > contract.maximum_final_xyz_norm_m + 1.0e-12)
            + np.count_nonzero(previous_norm > contract.maximum_final_xyz_norm_m + 1.0e-12)
        )
        if action_bound_violations:
            errors.append("ACTION_BOUND_VIOLATION")
        if not np.all(np.isin(action[:, 3], (0.0, 1.0))) or not np.all(
            np.isin(previous[:, 3], (0.0, 1.0))
        ):
            errors.append("GRIPPER_ACTION_INVALID")
        expected_previous = np.zeros_like(action)
        expected_previous[1:] = action[:-1]
        if not np.allclose(previous, expected_previous, rtol=0.0, atol=1.0e-8):
            errors.append("PREVIOUS_ACTION_NOT_CAUSAL")

    expected_shapes = {
        "ee_position_root_m": (rows, 3),
        "ee_quat_root_xyzw": (rows, 4),
        "arm_q_rad": (rows, 7),
        "arm_qd_rad_s": (rows, 7),
        "gripper_state": (rows, 1),
    }
    for camera_name in KEYBOARD_V3_CAMERA_NAMES:
        expected_shapes.update(
            {
                f"{camera_name}_rgb": (rows, None, None, 3),
                f"{camera_name}_depth_m": (rows, None, None, 1),
                f"{camera_name}_depth_valid": (rows, None, None, 1),
                f"{camera_name}_rgb_timestamp_s": (rows,),
                f"{camera_name}_depth_timestamp_s": (rows,),
                f"{camera_name}_frame_id": (rows,),
            }
        )
        if camera_evidence_active:
            expected_shapes.update(
                {
                    f"{camera_name}_depth_raw_m": (rows, None, None, 1),
                    f"{camera_name}_depth_source_valid": (rows, None, None, 1),
                    f"associated_{camera_name}_frame_index": (rows,),
                    f"{camera_name}_camera_pose_root_m_xyzw": (rows, 7),
                }
            )
    observation_arrays: dict[str, np.ndarray] = {}
    for name, shape in expected_shapes.items():
        try:
            value = _array(episode.observations, name)
            observation_arrays[name] = value
            if value.ndim != len(shape) or any(
                expected is not None and value.shape[index] != expected
                for index, expected in enumerate(shape)
            ):
                errors.append(f"OBSERVATION_{name.upper()}_SHAPE_INVALID")
        except KeyboardV3DatasetError:
            errors.append(f"OBSERVATION_{name.upper()}_MISSING")
    camera_shapes: set[tuple[int, ...]] = set()
    for camera_name in KEYBOARD_V3_CAMERA_NAMES:
        rgb = observation_arrays.get(f"{camera_name}_rgb")
        if rgb is not None:
            camera_shapes.add(tuple(rgb.shape[1:3]))
            if rgb.dtype != np.uint8:
                errors.append(f"{camera_name.upper()}_RGB_MUST_BE_UINT8")
        depth = observation_arrays.get(f"{camera_name}_depth_m")
        depth_valid = observation_arrays.get(f"{camera_name}_depth_valid")
        if depth is not None and depth_valid is not None and depth.shape == depth_valid.shape:
            if depth_valid.dtype != np.bool_ and not np.all(np.isin(depth_valid, (0, 1))):
                errors.append(f"{camera_name.upper()}_DEPTH_VALID_NOT_BINARY")
            else:
                valid = depth_valid.astype(np.bool_, copy=False)
                if not np.isfinite(depth).all() or np.any(depth < 0.0):
                    errors.append(f"{camera_name.upper()}_DEPTH_NOT_FINITE_NONNEGATIVE")
                if np.any(depth[valid] > contract.maximum_policy_depth_m):
                    errors.append(f"{camera_name.upper()}_DEPTH_VALID_OUT_OF_RANGE")
                if np.any(depth[~valid] != 0.0):
                    errors.append(f"{camera_name.upper()}_DEPTH_INVALID_NOT_FINITE_ZERO")
        if camera_evidence_active:
            raw = observation_arrays.get(f"{camera_name}_depth_raw_m")
            source_valid = observation_arrays.get(f"{camera_name}_depth_source_valid")
            if raw is not None and source_valid is not None:
                try:
                    for row_index in range(rows):
                        _raw, expected_source_valid = raw_depth_and_validity(
                            raw[row_index], source_valid[row_index]
                        )
                        if not np.array_equal(
                            expected_source_valid,
                            np.asarray(source_valid[row_index], dtype=np.bool_),
                        ):
                            raise RGBDLoggingContractError("source validity mismatch")
                except RGBDLoggingContractError:
                    errors.append(f"{camera_name.upper()}_RAW_DEPTH_EVIDENCE_INVALID")
            pose = observation_arrays.get(f"{camera_name}_camera_pose_root_m_xyzw")
            if pose is not None and pose.shape == (rows, 7):
                if not np.isfinite(pose).all() or np.any(
                    np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1.0) > 1.0e-3
                ):
                    errors.append(f"{camera_name.upper()}_CAMERA_POSE_INVALID")
    if len(camera_shapes) > 1:
        errors.append("RECORDED_CAMERA_RESOLUTION_MISMATCH")
    for name in (
        "ee_position_root_m",
        "ee_quat_root_xyzw",
        "arm_q_rad",
        "arm_qd_rad_s",
        "gripper_state",
    ):
        value = observation_arrays.get(name)
        if value is not None and not np.isfinite(value).all():
            errors.append(f"OBSERVATION_{name.upper()}_NONFINITE")
    quat = observation_arrays.get("ee_quat_root_xyzw")
    if quat is not None and quat.shape == (rows, 4) and np.any(
        np.abs(np.linalg.norm(quat, axis=1) - 1.0) > 1.0e-3
    ):
        errors.append("EE_QUATERNION_NOT_UNIT")
    gripper_state = observation_arrays.get("gripper_state")
    if gripper_state is not None and not np.all(np.isin(gripper_state, (0.0, 1.0))):
        errors.append("GRIPPER_STATE_INVALID")

    try:
        close_state = _array(episode.events, "close_state").reshape(-1)
        close_edge = _array(episode.events, "close_edge").reshape(-1)
        if close_state.shape != (rows,) or close_edge.shape != (rows,):
            errors.append("CLOSE_EVENT_SHAPE_INVALID")
        elif not np.all(np.isin(close_state, (0, 1))) or not np.all(
            np.isin(close_edge, (0, 1))
        ):
            errors.append("CLOSE_EVENT_NOT_BINARY")
        else:
            expected_edge = close_edge_from_state(close_state)
            if not np.array_equal(close_edge.astype(bool), expected_edge):
                errors.append("CLOSE_EDGE_STATE_MISMATCH")
            if rows and not np.array_equal(close_state.astype(np.float64), action[:, 3]):
                errors.append("CLOSE_STATE_ACTION_MISMATCH")
    except KeyboardV3DatasetError:
        close_state = np.zeros((rows,), dtype=np.bool_)
        close_edge = np.zeros((rows,), dtype=np.bool_)
        errors.append("CLOSE_EVENTS_MISSING")

    try:
        timestamp = _array(episode.time, "timestamp_s").astype(np.float64)
        control_step = _array(episode.time, "control_step").astype(np.int64)
        if timestamp.shape != (rows,) or control_step.shape != (rows,):
            errors.append("TIME_SHAPE_INVALID")
        elif rows:
            if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0.0):
                errors.append("TIMESTAMP_NOT_STRICTLY_MONOTONIC")
            if not np.array_equal(control_step, np.arange(rows, dtype=np.int64)):
                errors.append("CONTROL_STEP_NOT_CONTIGUOUS")
            if rows > 1 and not np.allclose(
                np.diff(timestamp), contract.policy_dt_s, rtol=0.0, atol=1.0e-6
            ):
                errors.append("CONTROL_DT_MISMATCH")
            for camera_name in KEYBOARD_V3_CAMERA_NAMES:
                frame_id = observation_arrays.get(f"{camera_name}_frame_id")
                rgb_sensor_time = observation_arrays.get(
                    f"{camera_name}_rgb_timestamp_s"
                )
                depth_sensor_time = observation_arrays.get(
                    f"{camera_name}_depth_timestamp_s"
                )
                if frame_id is not None:
                    if frame_id.dtype.kind not in ("i", "u"):
                        errors.append(
                            f"{camera_name.upper()}_FRAME_ID_NOT_INTEGER"
                        )
                    elif np.any(frame_id < 0) or np.any(np.diff(frame_id) < 0):
                        errors.append(
                            f"{camera_name.upper()}_FRAME_ID_NOT_MONOTONIC"
                        )
                    elif frame_id.shape == (rows,):
                        expected_acquisition = np.arange(rows, dtype=np.int64) // 2
                        if not np.array_equal(
                            frame_id - int(frame_id[0]), expected_acquisition
                        ):
                            errors.append(
                                f"{camera_name.upper()}_FRAME_ID_NOT_25HZ_REUSED"
                            )
                    if camera_evidence_active:
                        associated = observation_arrays.get(
                            f"associated_{camera_name}_frame_index"
                        )
                        if associated is None or not np.array_equal(
                            associated.astype(np.int64), frame_id.astype(np.int64)
                        ):
                            errors.append(
                                f"{camera_name.upper()}_CONTROL_FRAME_ASSOCIATION_MISMATCH"
                            )
                if (
                    rgb_sensor_time is not None
                    and depth_sensor_time is not None
                    and rgb_sensor_time.shape == (rows,)
                    and depth_sensor_time.shape == (rows,)
                    and not np.array_equal(rgb_sensor_time, depth_sensor_time)
                ):
                    errors.append(
                        f"{camera_name.upper()}_RGB_DEPTH_TIMESTAMP_MISMATCH"
                    )
                for suffix in ("rgb_timestamp_s", "depth_timestamp_s"):
                    field = f"{camera_name}_{suffix}"
                    sensor_time = observation_arrays.get(field)
                    if sensor_time is not None and sensor_time.shape == (rows,):
                        if not np.isfinite(sensor_time).all():
                            errors.append(f"{field.upper()}_NONFINITE")
                        elif np.any(np.diff(sensor_time) < -1.0e-9):
                            errors.append(f"{field.upper()}_NOT_MONOTONIC")
                        else:
                            # Compare adjacent *sensor acquisitions*, not the
                            # whole float32 clock against an ideal float64
                            # origin.  The latter accumulates quantization and
                            # falsely rejected otherwise exact K,K,K+1,K+1
                            # frame reuse after a few seconds.  Reused frames
                            # must preserve their timestamp exactly; each new
                            # frame must advance by one measured 25-Hz period.
                            sensor_delta = np.diff(sensor_time.astype(np.float64))
                            if frame_id is None or frame_id.shape != (rows,):
                                errors.append(
                                    f"{field.upper()}_FRAME_ID_REQUIRED_FOR_CLOCK"
                                )
                                continue
                            frame_delta = np.diff(frame_id.astype(np.int64))
                            reused = frame_delta == 0
                            acquired = frame_delta == 1
                            clock_valid = bool(
                                np.all(sensor_delta[reused] == 0.0)
                                and np.all(acquired | reused)
                                and np.allclose(
                                    sensor_delta[acquired],
                                    contract.rgbd_dt_s,
                                    rtol=0.0,
                                    atol=RGBD_ACQUISITION_INCREMENT_TOLERANCE_S,
                                )
                            )
                            if not clock_valid:
                                errors.append(
                                    f"{field.upper()}_NOT_25HZ_ACQUISITION_CLOCK"
                                )
    except (KeyboardV3DatasetError, ValueError):
        errors.append("TIME_FIELDS_MISSING_OR_INVALID")

    try:
        cube = _array(episode.privileged, "cube_center_root_m")
        if cube.shape != (rows, 3) or not np.isfinite(cube).all():
            errors.append("CUBE_CENTER_PRIVILEGED_INVALID")
    except KeyboardV3DatasetError:
        errors.append("CUBE_CENTER_PRIVILEGED_MISSING")
    binary_outcomes = {
        "contact_force_valid",
        "contact_measurement_valid",
        "left_contact",
        "right_contact",
        "contact",
        "exact_inner_contact",
        "exact_outer_contact",
        "bilateral_contact",
        "stable_grasp",
        "physical_lift",
        "forbidden_collision",
        "forbidden_collision_valid",
        "safety_violation",
        "safety_measurement_valid",
        "gripper_command",
    }
    for name, trailing in KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES.items():
        try:
            value = _array(episode.privileged, name)
            if value.shape != (rows, *trailing):
                errors.append(f"PRIVILEGED_{name.upper()}_SHAPE_INVALID")
                continue
            if name in binary_outcomes:
                if value.dtype != np.bool_ and not np.all(np.isin(value, (0, 1))):
                    errors.append(f"PRIVILEGED_{name.upper()}_NOT_BINARY")
            elif not np.isfinite(value).all():
                errors.append(f"PRIVILEGED_{name.upper()}_NONFINITE")
        except KeyboardV3DatasetError:
            errors.append(f"PRIVILEGED_{name.upper()}_MISSING")
    if all(name in episode.privileged for name in ("exact_inner_contact", "exact_outer_contact", "bilateral_contact")):
        expected_bilateral = (
            np.asarray(episode.privileged["exact_inner_contact"], dtype=bool)
            & np.asarray(episode.privileged["exact_outer_contact"], dtype=bool)
        )
        if not np.array_equal(expected_bilateral, np.asarray(episode.privileged["bilateral_contact"], dtype=bool)):
            errors.append("BILATERAL_CONTACT_DERIVATION_MISMATCH")
    if all(
        name in episode.privileged
        for name in ("left_contact", "right_contact", "contact")
    ):
        left_contact = np.asarray(episode.privileged["left_contact"], dtype=bool)
        right_contact = np.asarray(episode.privileged["right_contact"], dtype=bool)
        expected_contact = left_contact | right_contact
        if not np.array_equal(
            expected_contact, np.asarray(episode.privileged["contact"], dtype=bool)
        ):
            errors.append("CONTACT_BOOLEAN_DERIVATION_MISMATCH")
        if "bilateral_contact" in episode.privileged and not np.array_equal(
            left_contact & right_contact,
            np.asarray(episode.privileged["bilateral_contact"], dtype=bool),
        ):
            errors.append("BILATERAL_BOOLEAN_DERIVATION_MISMATCH")
        if "exact_inner_contact" in episode.privileged and not np.array_equal(
            left_contact,
            np.asarray(episode.privileged["exact_inner_contact"], dtype=bool),
        ):
            errors.append("LEFT_INNER_CONTACT_ALIAS_MISMATCH")
        if "exact_outer_contact" in episode.privileged and not np.array_equal(
            right_contact,
            np.asarray(episode.privileged["exact_outer_contact"], dtype=bool),
        ):
            errors.append("RIGHT_OUTER_CONTACT_ALIAS_MISMATCH")
    if all(
        name in episode.privileged
        for name in ("contact_force_left_n", "contact_force_right_n", "total_normal_force_n")
    ):
        left_force = np.asarray(episode.privileged["contact_force_left_n"], dtype=np.float64)
        right_force = np.asarray(episode.privileged["contact_force_right_n"], dtype=np.float64)
        total_force = np.asarray(episode.privileged["total_normal_force_n"], dtype=np.float64)
        if np.any(left_force < 0.0) or np.any(right_force < 0.0) or np.any(total_force < 0.0):
            errors.append("NORMAL_FORCE_NEGATIVE")
        if not np.allclose(total_force, left_force + right_force, rtol=0.0, atol=1.0e-6):
            errors.append("TOTAL_NORMAL_FORCE_DERIVATION_MISMATCH")
    if all(
        name in episode.privileged
        for name in ("contact_timestamp_s", "contact_control_step")
    ):
        contact_time = np.asarray(
            episode.privileged["contact_timestamp_s"], dtype=np.float64
        )
        contact_step = np.asarray(episode.privileged["contact_control_step"])
        if contact_step.dtype.kind not in ("i", "u"):
            errors.append("CONTACT_CONTROL_STEP_NOT_INTEGER")
        if "timestamp_s" in episode.time and not np.allclose(
            contact_time,
            np.asarray(episode.time["timestamp_s"], dtype=np.float64)
            + contract.policy_dt_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            errors.append("CONTACT_TIMESTAMP_TIME_GROUP_MISMATCH")
        if "control_step" in episode.time and not np.array_equal(
            contact_step, np.asarray(episode.time["control_step"]) + 1
        ):
            errors.append("CONTACT_CONTROL_STEP_TIME_GROUP_MISMATCH")
    if "cube_pose_root_m_xyzw" in episode.privileged:
        cube_pose = np.asarray(episode.privileged["cube_pose_root_m_xyzw"], dtype=np.float64)
        if cube_pose.shape == (rows, 7) and np.any(
            np.abs(np.linalg.norm(cube_pose[:, 3:7], axis=1) - 1.0) > 1.0e-3
        ):
            errors.append("CUBE_POSE_QUATERNION_NOT_UNIT")
    if "gripper_command" in episode.privileged and rows:
        if not np.array_equal(
            np.asarray(episode.privileged["gripper_command"], dtype=np.float64),
            action[:, 3],
        ):
            errors.append("GRIPPER_COMMAND_ACTION_MISMATCH")
    pad_valid = bool(metadata.get("pad_surface_valid", False))
    pad_fields = (
        "left_pad_surface_position_root_m",
        "right_pad_surface_position_root_m",
        "left_pad_surface_quat_root_xyzw",
        "right_pad_surface_quat_root_xyzw",
    )
    present_pad = [name for name in pad_fields if name in episode.privileged]
    if pad_valid:
        if metadata.get("pad_calibration_verified") is not True or len(present_pad) != 4:
            errors.append("PAD_SURFACE_AUTHORITY_UNVERIFIED")
    elif present_pad:
        errors.append("PAD_SURFACE_FIELDS_PRESENT_WITHOUT_AUTHORITY")
    if any("midpoint" in str(name).lower() for name in episode.privileged):
        errors.append("PAD_MIDPOINT_PROXY_FORBIDDEN")

    for name in ("nominal_grasp_pose_root_m_xyzw", "near_grasp_pose_root_m_xyzw"):
        try:
            pose = _array(episode.planner, name).astype(np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                errors.append(f"PLANNER_{name.upper()}_INVALID")
            elif abs(float(np.linalg.norm(pose[3:])) - 1.0) > 1.0e-3:
                errors.append(f"PLANNER_{name.upper()}_QUATERNION_INVALID")
        except KeyboardV3DatasetError:
            errors.append(f"PLANNER_{name.upper()}_MISSING")
    nominal = np.empty((0,), dtype=np.float64)
    try:
        nominal = _array(episode.planner, "nominal_grasp_pose_root_m_xyzw").astype(np.float64)
        near = _array(episode.planner, "near_grasp_pose_root_m_xyzw").astype(np.float64)
        axis = _array(episode.planner, "approach_axis_root").astype(np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(float(np.linalg.norm(axis)) - 1.0) > 1.0e-6:
            errors.append("PLANNER_APPROACH_AXIS_INVALID")
        elif nominal.shape == (7,) and near.shape == (7,) and start_offset.shape == (3,):
            expected_near = nominal[:3] - axis * backoff + start_offset
            if not np.allclose(near[:3], expected_near, rtol=0.0, atol=1.0e-6):
                errors.append("PLANNER_NEAR_GRASP_BACKOFF_MISMATCH")
            if not np.allclose(near[3:], nominal[3:], rtol=0.0, atol=1.0e-6):
                errors.append("PLANNER_NEAR_GRASP_ORIENTATION_CHANGED")
    except (KeyboardV3DatasetError, UnboundLocalError):
        errors.append("PLANNER_APPROACH_AXIS_MISSING")
    for name in ("nominal_grasp_q_rad", "near_grasp_q_rad"):
        if name in episode.planner:
            q = np.asarray(episode.planner[name])
            if q.shape != (7,) or not np.isfinite(q).all():
                errors.append(f"PLANNER_{name.upper()}_INVALID")
    analysis_fields = {
        "current_nominal_grasp_residual_m",
        "local_grasp_phase_code",
    }
    collection_analysis_fields = {
        "target_residual_m",
        "current_nominal_grasp_residual_m",
        "lateral_error_m",
        "approach_cosine",
        "ee_speed_m_s",
        "phase_code",
    }
    if collection_target_active and not collection_analysis_fields.issubset(
        episode.planner
    ):
        errors.append("COLLECTION_ANALYSIS_FIELDS_MISSING")
    analysis_present = bool(analysis_fields.intersection(episode.planner))
    if local_contract_present and not analysis_fields.issubset(episode.planner):
        errors.append("LOCAL_GRASP_ANALYSIS_FIELDS_MISSING")
    if analysis_present:
        if not analysis_fields.issubset(episode.planner):
            errors.append("LOCAL_GRASP_ANALYSIS_FIELDS_INCOMPLETE")
        else:
            residual = np.asarray(
                episode.planner["current_nominal_grasp_residual_m"],
                dtype=np.float64,
            )
            phase_code = np.asarray(episode.planner["local_grasp_phase_code"])
            if residual.shape != (rows,) or not np.isfinite(residual).all() or np.any(residual < 0.0):
                errors.append("NOMINAL_GRASP_RESIDUAL_INVALID")
            if phase_code.shape != (rows,) or phase_code.dtype.kind not in ("i", "u"):
                errors.append("LOCAL_GRASP_PHASE_CODE_INVALID")
            ee_position = observation_arrays.get("ee_position_root_m")
            if (
                residual.shape == (rows,)
                and ee_position is not None
                and ee_position.shape == (rows, 3)
                and nominal.shape == (7,)
            ):
                expected_residual = np.asarray(
                    [
                        nominal_grasp_translation_residual_m(position, nominal)
                        for position in ee_position
                    ],
                    dtype=np.float64,
                )
                if not np.allclose(residual, expected_residual, rtol=0.0, atol=1.0e-6):
                    errors.append("NOMINAL_GRASP_RESIDUAL_MISMATCH")
                expected_phase = np.asarray(
                    [
                        LOCAL_GRASP_PHASE_CODE[
                            classify_local_grasp_phase(
                                float(distance), gripper_closed=bool(closed)
                            )
                        ]
                        for distance, closed in zip(
                            expected_residual, close_state, strict=True
                        )
                    ],
                    dtype=np.int64,
                )
                if phase_code.shape == (rows,) and not np.array_equal(
                    phase_code.astype(np.int64), expected_phase
                ):
                    errors.append("LOCAL_GRASP_PHASE_CODE_MISMATCH")
    if collection_target_active and collection_analysis_fields.issubset(
        episode.planner
    ):
        target = np.asarray(episode.planner["target_residual_m"], dtype=np.float64)
        lateral = np.asarray(episode.planner["lateral_error_m"], dtype=np.float64)
        cosine = np.asarray(episode.planner["approach_cosine"], dtype=np.float64)
        speed = np.asarray(episode.planner["ee_speed_m_s"], dtype=np.float64)
        phase_alias = np.asarray(episode.planner["phase_code"])
        if target.shape != (rows,) or not np.allclose(
            target, backoff, rtol=0.0, atol=1.0e-7
        ):
            errors.append("TARGET_RESIDUAL_ROWS_MISMATCH")
        for name, value in (
            ("LATERAL_ERROR", lateral),
            ("APPROACH_COSINE", cosine),
            ("EE_SPEED", speed),
        ):
            if value.shape != (rows,) or not np.isfinite(value).all():
                errors.append(f"{name}_ROWS_INVALID")
        if cosine.shape == (rows,) and np.any(np.abs(cosine) > 1.0 + 1.0e-6):
            errors.append("APPROACH_COSINE_RANGE_INVALID")
        if lateral.shape == (rows,) and np.any(lateral < 0.0):
            errors.append("LATERAL_ERROR_NEGATIVE")
        if speed.shape == (rows,) and np.any(speed < 0.0):
            errors.append("EE_SPEED_NEGATIVE")
        if phase_alias.shape != (rows,) or not np.array_equal(
            phase_alias, np.asarray(episode.planner["local_grasp_phase_code"])
        ):
            errors.append("PHASE_CODE_ALIAS_MISMATCH")

    # The input inventory is a fixed allowlist.  Planner/privileged values are
    # available only through the episode object, never this student inventory.
    student_privileged_count = sum(
        int(field in KEYBOARD_V3_STUDENT_FIELDS) for field in KEYBOARD_V3_PRIVILEGED_FIELDS
    )
    if student_privileged_count:
        errors.append("STUDENT_PRIVILEGED_LEAKAGE")
    first_close = int(np.flatnonzero(close_edge)[0]) if rows and np.any(close_edge) else rows
    nonzero = int(
        np.count_nonzero(np.linalg.norm(action[:first_close, :3], axis=1) > 1.0e-9)
    ) if rows else 0
    close_count = int(np.count_nonzero(close_edge)) if rows else 0
    if close_count > 1:
        errors.append("MULTIPLE_CLOSE_EDGES_FORBIDDEN")
    expected_post_close = rows - int(np.flatnonzero(close_edge)[0]) - 1 if close_count else 0
    if metadata.get("post_close_observed_steps") != expected_post_close:
        errors.append("POST_CLOSE_OBSERVED_STEPS_MISMATCH")
    expected_censored = bool(close_count and expected_post_close < 64)
    if bool(metadata.get("future_outcome_right_censored")) != expected_censored:
        errors.append("FUTURE_OUTCOME_CENSORING_MISMATCH")
    if bool(metadata.get("success", False)):
        if close_count != 1:
            errors.append("SUCCESS_REQUIRES_ONE_CLOSE_EDGE")
        # Operator outcome and raw mechanics are preserved independently.
        # CONTACT/STABLE/FORCE_OK/GRASP_SUCCESS are generated offline from
        # the raw traces; collection-time thresholds are diagnostics only and
        # cannot become production label authority here.
    return KeyboardV3Validation(
        episode_id=episode.episode_id,
        passed=not errors,
        rows=rows,
        nonzero_micro_approach_rows=nonzero,
        close_event_count=close_count,
        action_bound_violation_count=action_bound_violations,
        student_privileged_input_count=student_privileged_count,
        errors=tuple(errors),
    )


def _write_mapping(group: Any, values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        array = np.asarray(value)
        if array.dtype.kind in ("U", "O"):
            raise KeyboardV3DatasetError(f"dataset {name} cannot contain object/string arrays")
        group.create_dataset(name, data=array, compression="gzip" if array.ndim >= 3 else None)


def _write_episode_artifact(
    path: str | Path,
    episode: KeyboardV3Episode,
    validation: KeyboardV3Validation,
    *,
    training_eligible: bool,
) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise KeyboardV3DatasetError("keyboard-v3 episode artifact already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    h5py = _h5py()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["file_schema"] = KEYBOARD_V3_FILE_SCHEMA
            handle.attrs["student_privileged_input_count"] = 0
            handle.attrs["training_eligible"] = bool(training_eligible)
            handle.attrs["artifact_class"] = (
                "CANONICAL_TRAINING_EPISODE"
                if training_eligible
                else "REJECTED_EPISODE_EVIDENCE"
            )
            root = handle.require_group("episodes").create_group(episode.episode_id)
            metadata = root.create_group("metadata")
            for name, value in episode.metadata.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    array = np.asarray(value)
                    if array.dtype.kind in ("U", "O"):
                        metadata.attrs[name] = json.dumps(
                            [str(item) for item in array.reshape(-1)],
                            separators=(",", ":"),
                        )
                    else:
                        metadata.create_dataset(name, data=array)
                else:
                    metadata.attrs[name] = value
            for group_name, values in (
                ("observations", episode.observations),
                ("actions", episode.actions),
                ("events", episode.events),
                ("privileged", episode.privileged),
                ("planner", episode.planner),
                ("time", episode.time),
            ):
                _write_mapping(root.create_group(group_name), values)
            receipt = root.create_group("validation")
            for name, value in validation.payload().items():
                if isinstance(value, list):
                    receipt.attrs[name] = json.dumps(value, sort_keys=True)
                else:
                    receipt.attrs[name] = value
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_keyboard_v3_episode(path: str | Path, episode: KeyboardV3Episode) -> KeyboardV3Validation:
    """Validate then atomically create one immutable canonical episode."""

    validation = validate_episode(episode)
    if not validation.passed:
        raise KeyboardV3DatasetError(
            "episode failed v3 validation: " + ",".join(validation.errors)
        )
    _write_episode_artifact(
        path, episode, validation, training_eligible=True
    )
    return validation


def write_rejected_keyboard_v3_episode(
    path: str | Path, episode: KeyboardV3Episode
) -> KeyboardV3Validation:
    """Preserve invalid evidence without admitting it to canonical training."""

    validation = validate_episode(episode)
    if validation.passed:
        raise KeyboardV3DatasetError(
            "valid episode must use canonical writer, not rejected storage"
        )
    _write_episode_artifact(
        path, episode, validation, training_eligible=False
    )
    return validation


def _v1_flat_episode_fields(episode: KeyboardV3Episode) -> dict[str, np.ndarray]:
    """Return a flat V1-writer-style dataset map without semantic coercion."""

    fields: dict[str, np.ndarray] = {
        # V1 readers discover the policy action through ``actions``.  The
        # enclosing attrs make its exact 4-D metric meaning explicit.
        "actions": np.asarray(episode.actions["action_4d"]),
    }
    for namespace, values in (
        ("observations", episode.observations),
        ("actions", episode.actions),
        ("events", episode.events),
        ("privileged", episode.privileged),
        ("planner", episode.planner),
        ("time", episode.time),
    ):
        for name, value in values.items():
            target_name = name
            if target_name == "action_4d":
                target_name = "policy_action_4d_metric_root_m"
            if target_name in fields:
                target_name = f"{namespace}_{target_name}"
            if target_name in fields:
                raise KeyboardV3DatasetError(
                    f"V1 aggregate field collision: {namespace}/{name}"
                )
            fields[target_name] = np.asarray(value)
    for name, value in episode.metadata.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value)
            if array.dtype.kind not in ("U", "O"):
                target_name = f"metadata_{name}"
                if target_name in fields:
                    raise KeyboardV3DatasetError(
                        f"V1 aggregate metadata collision: {name}"
                    )
                fields[target_name] = array
    return fields


def append_keyboard_v3_episode_v1_layout(
    path: str | Path,
    episode: KeyboardV3Episode,
    *,
    training_eligible: bool,
) -> KeyboardV3V1AggregateReceipt:
    """Append one episode to a durable V1-style ``/data/demo_N`` container.

    The immutable per-episode artifact remains the provenance source.  This
    aggregate is the continuous-collection view requested by the operator.
    A process lock prevents two collectors from assigning the same demo ID.
    """

    validation = validate_episode(episode)
    if bool(training_eligible) != bool(validation.passed):
        raise KeyboardV3DatasetError(
            "aggregate eligibility must exactly match v3 validator result"
        )
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    h5py = _h5py()
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            with h5py.File(target, "a") as handle:
                if "file_schema" in handle.attrs and (
                    handle.attrs["file_schema"] != KEYBOARD_V3_V1_AGGREGATE_SCHEMA
                ):
                    raise KeyboardV3DatasetError("V1 aggregate file schema mismatch")
                handle.attrs["format_version"] = 1
                handle.attrs["file_schema"] = KEYBOARD_V3_V1_AGGREGATE_SCHEMA
                handle.attrs["layout_compatibility"] = "V1_DATA_DEMO_N"
                handle.attrs["action_semantics"] = (
                    "EXACT_4D_DX_DY_DZ_G_ROBOT_ROOT_METERS"
                )
                handle.attrs["legacy_v1_8d_action_compatible"] = False
                data = handle.require_group("data")
                data.attrs["env_args"] = json.dumps(
                    {
                        "type": "keyboard_v3",
                        "control_hz": 50,
                        "rgbd_hz": 25,
                        "physics_hz": 500,
                        "action_dim": 4,
                        "action_frame": "robot_root",
                        "action_unit": "m",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for existing in data.values():
                    if str(existing.attrs.get("episode_id", "")) == episode.episode_id:
                        raise KeyboardV3DatasetError(
                            f"episode already exists in V1 aggregate: {episode.episode_id}"
                        )
                indices = [
                    int(name.removeprefix("demo_"))
                    for name in data.keys()
                    if name.startswith("demo_")
                    and name.removeprefix("demo_").isdigit()
                ]
                demo_index = max(indices, default=-1) + 1
                demo_id = f"demo_{demo_index}"
                pending_id = f"__pending_{demo_id}_{os.getpid()}"
                if pending_id in data:
                    del data[pending_id]
                demo = data.create_group(pending_id)
                demo.attrs["episode_id"] = episode.episode_id
                demo.attrs["num_samples"] = validation.rows
                demo.attrs["seed"] = int(episode.metadata.get("seed", -1))
                demo.attrs["success"] = bool(episode.metadata.get("success", False))
                demo.attrs["failure_type"] = str(
                    episode.metadata.get("failure_type", "UNKNOWN")
                )
                demo.attrs["training_eligible"] = bool(training_eligible)
                demo.attrs["action_dim"] = 4
                demo.attrs["action_frame"] = "robot_root"
                demo.attrs["action_unit"] = "m"
                demo.attrs["hidden_legacy_8d_path_count"] = 0
                demo.attrs["silent_clipping_count"] = 0
                for name, value in episode.metadata.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        continue
                    demo.attrs[f"metadata_{name}"] = value
                for name, array in _v1_flat_episode_fields(episode).items():
                    if array.dtype.kind in ("U", "O"):
                        raise KeyboardV3DatasetError(
                            f"V1 aggregate dataset {name} cannot contain strings"
                        )
                    demo.create_dataset(
                        name,
                        data=array,
                        compression="gzip" if array.ndim >= 3 else None,
                    )
                validation_group = demo.create_group("validation")
                for name, value in validation.payload().items():
                    validation_group.attrs[name] = (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, list)
                        else value
                    )
                demo.attrs["write_complete"] = True
                handle.flush()
                data.move(pending_id, demo_id)
                total_rows = int(data.attrs.get("total", 0)) + validation.rows
                demo_count = int(data.attrs.get("num_demos", 0)) + 1
                data.attrs["total"] = total_rows
                data.attrs["num_demos"] = demo_count
                handle.flush()
            with target.open("rb") as aggregate_stream:
                os.fsync(aggregate_stream.fileno())
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return KeyboardV3V1AggregateReceipt(
        path=str(target),
        demo_id=demo_id,
        episode_id=episode.episode_id,
        rows=validation.rows,
        total_rows=total_rows,
        demo_count=demo_count,
        training_eligible=bool(training_eligible),
    )


def _read_keyboard_v3_episode_file(
    path: str | Path,
    episode_id: str | None = None,
    *,
    require_training_eligible: bool,
) -> KeyboardV3Episode:
    source = Path(path).expanduser().resolve()
    h5py = _h5py()
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("file_schema") != KEYBOARD_V3_FILE_SCHEMA:
            raise KeyboardV3DatasetError("keyboard-v3 file schema mismatch")
        if require_training_eligible and not bool(
            handle.attrs.get("training_eligible", False)
        ):
            raise KeyboardV3DatasetError("rejected episode cannot enter canonical loader")
        episodes = handle["episodes"]
        names = sorted(episodes.keys())
        if episode_id is None:
            if len(names) != 1:
                raise KeyboardV3DatasetError("episode_id required for multi-episode file")
            episode_id = names[0]
        if episode_id not in episodes:
            raise KeyboardV3DatasetError("keyboard-v3 episode_id not found")
        root = episodes[episode_id]
        metadata = {name: value for name, value in root["metadata"].attrs.items()}
        metadata.update({name: np.asarray(value) for name, value in root["metadata"].items()})
        for name, value in tuple(metadata.items()):
            if isinstance(value, str) and value.startswith("["):
                try:
                    metadata[name] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        groups = {
            name: {field: np.asarray(value) for field, value in root[name].items()}
            for name in ("observations", "actions", "events", "privileged", "planner", "time")
        }
    episode = KeyboardV3Episode(episode_id, metadata, **groups)
    validation = validate_episode(episode)
    if not validation.passed:
        raise KeyboardV3DatasetError(
            "stored keyboard-v3 episode is invalid: " + ",".join(validation.errors)
        )
    return episode


def read_keyboard_v3_episode(
    path: str | Path, episode_id: str | None = None
) -> KeyboardV3Episode:
    """Read an episode that was canonical at collection time."""

    return _read_keyboard_v3_episode_file(
        path, episode_id, require_training_eligible=True
    )


def read_recovered_keyboard_v3_episode(
    path: str | Path,
    *,
    expected_sha256: str,
    episode_id: str | None = None,
) -> KeyboardV3Episode:
    """Read immutable rejected evidence through an explicit recovery receipt.

    This is intentionally not a fallback in :func:`read_keyboard_v3_episode`.
    The caller must provide the SHA-256 recorded by a versioned recovery
    manifest, and the episode must pass the current complete validator.  The
    original HDF5 ``training_eligible`` attribute remains false and is never
    rewritten.
    """

    source = Path(path).expanduser().resolve()
    observed_sha256 = _sha256_file(source)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise KeyboardV3DatasetError("recovery SHA-256 is missing or malformed")
    if observed_sha256 != expected_sha256:
        raise KeyboardV3DatasetError("recovery source SHA-256 mismatch")
    return _read_keyboard_v3_episode_file(
        source, episode_id, require_training_eligible=False
    )


def list_keyboard_v3_episode_ids(path: str | Path) -> tuple[str, ...]:
    """List canonical episode IDs without materialising camera arrays."""

    source = Path(path).expanduser().resolve()
    h5py = _h5py()
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("file_schema") != KEYBOARD_V3_FILE_SCHEMA:
            raise KeyboardV3DatasetError("keyboard-v3 file schema mismatch")
        if not bool(handle.attrs.get("training_eligible", False)):
            raise KeyboardV3DatasetError(
                "rejected episode cannot enter canonical loader"
            )
        if "episodes" not in handle:
            raise KeyboardV3DatasetError("keyboard-v3 episodes group missing")
        names = tuple(sorted(str(name) for name in handle["episodes"].keys()))
    if not names:
        raise KeyboardV3DatasetError("keyboard-v3 file has no episodes")
    return names


def episode_to_gru(
    episode: KeyboardV3Episode,
    *,
    config: HumanGraspGRUConfig = HumanGraspGRUConfig(),
    close_window: tuple[int, int] | None = None,
) -> tuple[HumanGraspSequenceInputs, HumanGraspSequenceTargets]:
    """Direct v3-to-GRU conversion; no legacy migration/slicing is involved."""

    validation = validate_episode(episode)
    if not validation.passed:
        raise KeyboardV3DatasetError(
            "cannot collate invalid episode: " + ",".join(validation.errors)
        )
    action = np.asarray(episode.actions["action_4d"], dtype=np.float32)
    rows = action.shape[0]
    start, stop = 0, rows
    if close_window is not None:
        before, after = close_window
        if before > 0 or after < 0:
            raise KeyboardV3DatasetError("close window must be [negative_or_zero, positive_or_zero]")
        edges = np.flatnonzero(np.asarray(episode.events["close_edge"], dtype=bool))
        if edges.size == 0:
            raise KeyboardV3DatasetError("close-window collation requires a CLOSE edge")
        center = int(edges[0])
        start, stop = max(0, center + before), min(rows, center + after + 1)
    selected = slice(start, stop)
    obs = episode.observations
    # Head RGB-D remains recorded for operator visualization, detection,
    # recovery, and diagnostics.  It is deliberately absent from the grasp
    # actor input assembled here.
    rgb = torch.as_tensor(np.asarray(obs["right_wrist_rgb"])[selected]).unsqueeze(0)
    depth = torch.as_tensor(np.asarray(obs["right_wrist_depth_m"])[selected], dtype=torch.float32).unsqueeze(0)
    valid = torch.as_tensor(np.asarray(obs["right_wrist_depth_valid"])[selected], dtype=torch.bool).unsqueeze(0)
    ee = np.concatenate(
        (
            np.asarray(obs["ee_position_root_m"])[selected],
            np.asarray(obs["ee_quat_root_xyzw"])[selected],
        ),
        axis=1,
    )
    time_rows = stop - start
    previous_action = np.asarray(
        episode.actions["previous_action_4d"], dtype=np.float32
    )[selected]
    # Strict causal alignment: the decision target at t must never be present
    # in an input at t.  The writer/validator guarantees
    # previous_action_4d[t] == action_4d[t-1], with an all-zero OPEN row at
    # episode start.  Do not use observations/gripper_state[t] here: it is the
    # post-command state and is identical to close_state[t] in this dataset.
    causal_gripper_state = previous_action[:, 3:4]
    inputs = HumanGraspSequenceInputs(
        right_wrist_rgb=rgb,
        right_wrist_depth_m=depth,
        right_wrist_depth_valid=valid,
        ee_pose_robot_root_m_xyzw=torch.as_tensor(ee, dtype=torch.float32).unsqueeze(0),
        right_arm_joint_position_rad=torch.as_tensor(
            np.asarray(obs["arm_q_rad"])[selected], dtype=torch.float32
        ).unsqueeze(0),
        right_arm_joint_velocity_rad_s=torch.as_tensor(
            np.asarray(obs["arm_qd_rad_s"])[selected], dtype=torch.float32
        ).unsqueeze(0),
        current_gripper_state=torch.as_tensor(
            causal_gripper_state, dtype=torch.float32
        ).unsqueeze(0),
        previous_policy_action_4d_metric_root_m=torch.as_tensor(
            previous_action, dtype=torch.float32
        ).unsqueeze(0),
        hidden_reset_mask=torch.tensor(
            [[True] + [False] * (time_rows - 1)], dtype=torch.bool
        ),
    )
    target_action = torch.as_tensor(action[selected], dtype=torch.float32)
    base = torch.ones((1, time_rows), dtype=torch.bool)
    close_edge = torch.as_tensor(
        np.asarray(episode.events["close_edge"])[selected], dtype=torch.float32
    ).view(1, time_rows, 1)
    local_edges = np.flatnonzero(np.asarray(episode.events["close_edge"])[selected])
    first_edge = int(local_edges[0]) if local_edges.size else time_rows
    index = torch.arange(time_rows)
    targets = HumanGraspSequenceTargets(
        xyz_action_robot_root_m=target_action[:, :3].unsqueeze(0),
        close_target=close_edge,
        padding_valid=base,
        row_valid=base.clone(),
        xyz_bc_eligible_mask=base.clone(),
        # CLOSE_EDGE supervision uses every hard negative through the onset
        # row, then excludes all persistence rows.  The runtime latch owns the
        # persistent CLOSED state after the one-shot edge.
        close_temporal_valid_mask=(index <= first_edge).unsqueeze(0),
        early_region_mask=(index < max(0, first_edge - 10)).unsqueeze(0),
        late_region_mask=(index > min(time_rows - 1, first_edge + 10)).unsqueeze(0),
        far_region_mask=torch.zeros((1, time_rows), dtype=torch.bool),
        close_target_semantics=CLOSE_EDGE_TARGET,
    )
    # Run the complete model contract without adding a second implementation.
    from .human_grasp_gru_bc import _validate_inputs, _validate_targets

    batch, time = _validate_inputs(inputs, config)
    _validate_targets(targets, batch=batch, time=time)
    return inputs, targets


def training_readiness(paths: Sequence[str | Path], *, split_seed: int = 42) -> dict[str, Any]:
    """Audit collected files and issue a fail-closed human-only BC receipt."""

    episodes: list[KeyboardV3Episode] = []
    validations: list[KeyboardV3Validation] = []
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        try:
            episode = read_keyboard_v3_episode(path)
            validation = validate_episode(episode)
            episodes.append(episode)
            validations.append(validation)
            hashes[str(path)] = _sha256_file(path)
        except (OSError, KeyboardV3DatasetError) as error:
            errors.append(f"{path}:{error}")
    ids = [episode.episode_id for episode in episodes]
    baseline_variants = {
        str(episode.metadata.get("baseline_variant", "LEGACY_KEYBOARD_V3_V1"))
        for episode in episodes
    }
    if len(baseline_variants) > 1:
        errors.append("BASELINE_VARIANT_SILENT_MERGE_FORBIDDEN")
    split_payload: dict[str, Any] | None = None
    try:
        split_payload = split_episode_ids(ids, seed=split_seed).as_dict()
    except Exception as error:
        errors.append(f"EPISODE_SPLIT:{error}")
    nonzero = sum(item.nonzero_micro_approach_rows for item in validations)
    closes = sum(item.close_event_count for item in validations)
    violations = sum(item.action_bound_violation_count for item in validations)
    ready = (
        not errors
        and bool(validations)
        and all(item.passed for item in validations)
        and nonzero > 0
        and closes > 0
        and violations == 0
        and split_payload is not None
    )
    payload = {
        "schema": "g2_keyboard_v3_training_readiness_v1",
        "keyboard_v3_schema": "PASS" if validations and all(item.passed for item in validations) else "FAIL",
        "action_contract": "PASS" if violations == 0 else "FAIL",
        "unit_contract": "PASS" if validations else "FAIL",
        "frame_contract": "PASS" if validations else "FAIL",
        "depth_preprocess": "PASS" if validations else "FAIL",
        "student_privileged_input_count": 0,
        "episode_count": len(validations),
        "nonzero_micro_approach_rows": nonzero,
        "close_event_count": closes,
        "no_action_bound_violations": violations == 0,
        "train_val_test_episode_split": "PASS" if split_payload else "FAIL",
        "split": split_payload,
        "episode_sha256": hashes,
        "errors": errors,
        "gru_bc_training_ready": ready,
        "privileged_feasibility_loss_when_unavailable": 0.0,
        "legacy_migration_required": False,
        "baseline_variants": sorted(baseline_variants),
        "baseline_variant_parity": "PASS" if len(baseline_variants) <= 1 else "FAIL",
    }
    payload["receipt_sha256"] = _json_sha256(payload)
    return payload


__all__ = [
    "FAILURE_TYPES",
    "KEYBOARD_V3_CAMERA_NAMES",
    "KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES",
    "KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES",
    "KEYBOARD_V3_RECORDED_NON_ACTOR_FIELDS",
    "KEYBOARD_V3_FILE_SCHEMA",
    "KEYBOARD_V3_SCHEMA",
    "KEYBOARD_V3_STUDENT_FIELDS",
    "KEYBOARD_V3_OPERATOR_VIEW_NAMES",
    "KEYBOARD_V3_REQUIRED_OUTCOME_SHAPES",
    "KEYBOARD_V3_V2_BASELINE_METADATA",
    "RGBD_ACQUISITION_INCREMENT_TOLERANCE_S",
    "KeyboardV3DatasetError",
    "KeyboardV3Episode",
    "KeyboardV3Validation",
    "OfflineGraspLabelThresholds",
    "build_privileged_feasibility_targets",
    "list_keyboard_v3_episode_ids",
    "canonical_action_4d",
    "close_edge_from_state",
    "derive_offline_grasp_labels",
    "keyboard_close_window_mask",
    "episode_to_gru",
    "preprocess_depth_m",
    "read_keyboard_v3_episode",
    "read_recovered_keyboard_v3_episode",
    "training_readiness",
    "validate_episode",
    "write_keyboard_v3_episode",
    "write_rejected_keyboard_v3_episode",
]
