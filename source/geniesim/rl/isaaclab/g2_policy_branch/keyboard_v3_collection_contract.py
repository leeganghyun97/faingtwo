# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Simulator-free keyboard-v3 collection and near-grasp handoff contract.

cuRobo owns only the nominal pose and approach to a bounded, contact-free
near-grasp start.  The operator then owns metric robot-root XYZ and abstract
OPEN/CLOSE.  This module contains no Isaac imports and cannot write a legacy
8-D action packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from geniesim.rl.sac.keyboard_grasp_contract import (
    LOCAL_GRASP_PHASE_CODE,
    canonical_keyboard_grasp_contract,
    classify_local_grasp_phase,
    nominal_grasp_translation_residual_m,
)
from geniesim.rl.sac.keyboard_v2_geometry_schema import CANDIDATE_A_SHA256
from geniesim.rl.sac.keyboard_v3_dataset import canonical_action_4d


KEYBOARD_V3_COLLECTION_CONTRACT_SCHEMA = "g2_keyboard_v3_collection_contract_v2"
TELEOP_COLLECTION_AUTHORITY = "BOUNDED_DEMONSTRATION_ONLY"
_COMMON = canonical_keyboard_grasp_contract()
NEAR_GRASP_BACKOFF_SET_M = _COMMON.curobo_handoff_band_m
DEFAULT_NEAR_GRASP_BACKOFF_M = _COMMON.curobo_default_handoff_m
COLLECTION_TARGETS_MM = (16, 17, 18, 19, 20, 21, 22)
COLLECTION_TARGETS_M = tuple(value / 1000.0 for value in COLLECTION_TARGETS_MM)
COLLECTION_TARGET_REFERENCE = (
    "CUROBO_NOMINAL_GRASP_POSE_TRANSLATION_RESIDUAL_NOT_PAD_SURFACE"
)
LOCAL_GRASP_BAND_M = _COMMON.local_grasp_band_m
LOCAL_GRASP_DISTANCE_REFERENCE = _COMMON.local_grasp_distance_reference
PILOT_OFFSET_MAGNITUDE_M = 0.003
PILOT_OFFSET_LABELS = ("center", "left", "right", "up", "down", "near", "far")


class KeyboardV3CollectionContractError(ValueError):
    pass


def _finite(value: Sequence[float], width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise KeyboardV3CollectionContractError(f"{name} must be {width} finite values")
    return result


@dataclass(frozen=True)
class NearGraspCondition:
    condition_id: str
    backoff_m: float
    start_offset_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.backoff_m not in (*NEAR_GRASP_BACKOFF_SET_M, *COLLECTION_TARGETS_M):
            raise KeyboardV3CollectionContractError("near-grasp backoff is not approved")
        offset = _finite(self.start_offset_xyz_m, 3, "start_offset_xyz_m")
        if float(np.max(np.abs(offset))) > PILOT_OFFSET_MAGNITUDE_M + 1.0e-12:
            raise KeyboardV3CollectionContractError("pilot start offset exceeds 3 mm")


def pilot_conditions() -> tuple[NearGraspCondition, ...]:
    """Return 21 deterministic small-offset pilot conditions."""

    d = PILOT_OFFSET_MAGNITUDE_M
    offsets = {
        "center": (0.0, 0.0, 0.0),
        "left": (0.0, d, 0.0),
        "right": (0.0, -d, 0.0),
        "up": (0.0, 0.0, d),
        "down": (0.0, 0.0, -d),
        "near": (d, 0.0, 0.0),
        "far": (-d, 0.0, 0.0),
    }
    return tuple(
        NearGraspCondition(
            condition_id=f"backoff_{int(round(backoff * 1000)):02d}mm_{label}",
            backoff_m=backoff,
            start_offset_xyz_m=offsets[label],
        )
        for backoff in NEAR_GRASP_BACKOFF_SET_M
        for label in PILOT_OFFSET_LABELS
    )


def collection_target_conditions() -> tuple[NearGraspCondition, ...]:
    """Return the balanced Keyboard-v3 16--22 mm target authority."""

    return tuple(
        NearGraspCondition(
            condition_id=f"target_{target_mm:02d}mm_center",
            backoff_m=target_mm / 1000.0,
            start_offset_xyz_m=(0.0, 0.0, 0.0),
        )
        for target_mm in COLLECTION_TARGETS_MM
    )


def target_distribution_counts(receipts: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Count attempted episodes by target without inferring their outcome."""

    counts = {target: 0 for target in COLLECTION_TARGETS_MM}
    for receipt in receipts:
        try:
            target = int(receipt["target_residual_mm"])
        except (KeyError, TypeError, ValueError) as error:
            raise KeyboardV3CollectionContractError(
                "result receipt lacks target_residual_mm"
            ) from error
        if target not in counts:
            raise KeyboardV3CollectionContractError(
                f"result receipt target is outside 16--22 mm: {target}"
            )
        counts[target] += 1
    return counts


def select_next_target_mm(receipts: Sequence[Mapping[str, Any]]) -> int:
    """Least-sampled deterministic scheduler; outcomes never affect selection."""

    counts = target_distribution_counts(receipts)
    return min(COLLECTION_TARGETS_MM, key=lambda target: (counts[target], target))


def preclose_geometry_metrics(
    *,
    ee_position_root_m: Sequence[float],
    previous_ee_position_root_m: Sequence[float] | None,
    cube_center_root_m: Sequence[float],
    nominal_grasp_pose_root_m_xyzw: Sequence[float],
    approach_axis_root: Sequence[float],
    action_xyz_root_m: Sequence[float],
    control_dt_s: float = 0.02,
) -> dict[str, float]:
    """One shared metric/frame implementation for HDF and terminal display."""

    ee = _finite(ee_position_root_m, 3, "ee_position_root_m")
    cube = _finite(cube_center_root_m, 3, "cube_center_root_m")
    nominal = _finite(
        nominal_grasp_pose_root_m_xyzw, 7, "nominal_grasp_pose_root_m_xyzw"
    )
    axis = _finite(approach_axis_root, 3, "approach_axis_root")
    action = _finite(action_xyz_root_m, 3, "action_xyz_root_m")
    if abs(float(np.linalg.norm(axis)) - 1.0) > 1.0e-6:
        raise KeyboardV3CollectionContractError("approach axis must be unit")
    if not np.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise KeyboardV3CollectionContractError("control_dt_s must be positive")
    error = nominal[:3] - ee
    residual = float(np.linalg.norm(error))
    longitudinal = float(np.dot(error, axis))
    lateral = float(np.linalg.norm(error - longitudinal * axis))
    action_norm = float(np.linalg.norm(action))
    approach_cosine = (
        float(np.dot(action, error) / (action_norm * residual))
        if action_norm > 1.0e-12 and residual > 1.0e-12
        else 0.0
    )
    speed = 0.0
    if previous_ee_position_root_m is not None:
        previous = _finite(
            previous_ee_position_root_m, 3, "previous_ee_position_root_m"
        )
        speed = float(np.linalg.norm(ee - previous) / control_dt_s)
    return {
        "current_nominal_grasp_residual_m": residual,
        "ee_to_cube_center_distance_m": float(np.linalg.norm(cube - ee)),
        "lateral_error_m": lateral,
        "approach_cosine": approach_cosine,
        "ee_speed_m_s": speed,
        "action_norm_m": action_norm,
    }


def format_terminal_status(
    *, episode_id: str, control_step: int, target_residual_mm: int,
    metrics: Mapping[str, float], action_xyz_root_m: Sequence[float],
    gripper_state: str, close_edge: bool, close_state: bool, phase: str,
) -> str:
    """Unambiguous operator display; all displayed distances are named."""

    if target_residual_mm not in COLLECTION_TARGETS_MM:
        raise KeyboardV3CollectionContractError("terminal target is outside 16--22 mm")
    action = _finite(action_xyz_root_m, 3, "action_xyz_root_m")
    return "\n".join(
        (
            f"[EP {episode_id}][STEP {int(control_step):04d}]",
            f"Target nominal residual : {target_residual_mm:.1f} mm",
            "Current nominal residual: "
            f"{metrics['current_nominal_grasp_residual_m'] * 1000.0:.3f} mm",
            "EE->Cube center distance: "
            f"{metrics['ee_to_cube_center_distance_m'] * 1000.0:.3f} mm",
            f"Lateral error           : {metrics['lateral_error_m'] * 1000.0:.3f} mm",
            f"Approach cosine         : {metrics['approach_cosine']:.6f}",
            f"EE speed                : {metrics['ee_speed_m_s'] * 1000.0:.3f} mm/s",
            "XYZ command robot_root  : "
            f"[{action[0]*1000.0:.3f}, {action[1]*1000.0:.3f}, {action[2]*1000.0:.3f}] mm",
            f"Action norm             : {metrics['action_norm_m'] * 1000.0:.3f} mm",
            f"Gripper                 : {gripper_state}",
            f"Close edge/state         : {int(close_edge)}/{int(close_state)}",
            f"Phase                    : {phase}",
            "Pad-surface distance     : UNAVAILABLE_NO_APPROVED_CALIBRATION",
        )
    )


def near_grasp_pose_root_m_xyzw(
    nominal_grasp_pose_root_m_xyzw: Sequence[float],
    approach_axis_root: Sequence[float],
    condition: NearGraspCondition,
) -> tuple[float, ...]:
    """Back off from the nominal pose along an explicit robot-root axis."""

    pose = _finite(nominal_grasp_pose_root_m_xyzw, 7, "nominal_grasp_pose")
    axis = _finite(approach_axis_root, 3, "approach_axis_root")
    axis_norm = float(np.linalg.norm(axis))
    if abs(axis_norm - 1.0) > 1.0e-6:
        raise KeyboardV3CollectionContractError("approach axis must be unit robot-root")
    if abs(float(np.linalg.norm(pose[3:])) - 1.0) > 1.0e-3:
        raise KeyboardV3CollectionContractError("nominal grasp quaternion must be unit XYZW")
    position = (
        pose[:3]
        - axis * condition.backoff_m
        + np.asarray(condition.start_offset_xyz_m, dtype=np.float64)
    )
    return tuple(float(value) for value in np.concatenate((position, pose[3:])))


@dataclass
class PersistentGripperCommand:
    """Edge-triggered persistent OPEN/CLOSE state; no direct finger targets."""

    closed: bool = False

    def apply(self, *, close: bool = False, open_: bool = False) -> tuple[float, bool]:
        if close and open_:
            raise KeyboardV3CollectionContractError("OPEN and CLOSE edges are mutually exclusive")
        previous = self.closed
        if close:
            self.closed = True
        elif open_:
            self.closed = False
        return float(self.closed), bool(self.closed and not previous)


def keyboard_action(
    xyz_delta_root_m: Sequence[float], gripper: PersistentGripperCommand
) -> np.ndarray:
    xyz = _finite(xyz_delta_root_m, 3, "keyboard_xyz_delta_root_m")
    return canonical_action_4d([*xyz, float(gripper.closed)])


def assert_candidate_a_asset(path: str | Path) -> str:
    import hashlib

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != CANDIDATE_A_SHA256:
        raise KeyboardV3CollectionContractError("Candidate A hash mismatch")
    return digest


def collection_contract_payload() -> dict[str, Any]:
    common = canonical_keyboard_grasp_contract()
    return {
        "schema": KEYBOARD_V3_COLLECTION_CONTRACT_SCHEMA,
        "candidate_a_sha256": CANDIDATE_A_SHA256,
        "teleop_collection_authority": TELEOP_COLLECTION_AUTHORITY,
        "episode_flow": [
            "RESET",
            "CUBE_RANDOMIZE",
            "CUROBO_SOLVE",
            "MOVE_NEAR_GRASP_OPEN",
            "RECORDING_START",
            "KEYBOARD_MICRO_APPROACH",
            "KEYBOARD_CLOSE",
            "SHORT_POST_CLOSE_CONTEXT",
            "VALIDATE_AND_SAVE",
        ],
        "near_grasp_backoff_set_m": list(NEAR_GRASP_BACKOFF_SET_M),
        "collection_targets_mm": list(COLLECTION_TARGETS_MM),
        "collection_target_reference": COLLECTION_TARGET_REFERENCE,
        "default_near_grasp_backoff_m": DEFAULT_NEAR_GRASP_BACKOFF_M,
        "local_grasp_band_m": list(LOCAL_GRASP_BAND_M),
        "local_grasp_distance_reference": LOCAL_GRASP_DISTANCE_REFERENCE,
        "local_grasp_phase_code": dict(LOCAL_GRASP_PHASE_CODE),
        "local_grasp_distance_is_student_input": False,
        "pilot_condition_count": len(pilot_conditions()),
        "pilot_offset_max_abs_m": PILOT_OFFSET_MAGNITUDE_M,
        "control_hz": common.control_hz,
        "control_dt_s": common.policy_dt_s,
        "physics_dt_s": common.physics_dt_s,
        "dataset_row_hz": common.dataset_row_hz,
        "rgbd_acquisition_hz": common.rgbd_acquisition_hz,
        "rgbd_dt_s": common.rgbd_dt_s,
        "rgbd_timestamp_source": "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE",
        "rgbd_observation_alignment": (
            "LATEST_VALID_ACQUISITION_REFERENCED_BY_FRAME_ID"
        ),
        "camera_capture_contract": (
            "ACTUAL_ACQUISITION_TIMESTAMP_HELD_ACROSS_TWO_CONTROL_ROWS"
        ),
        "physics_hz": common.physics_hz,
        "action_dim": common.bc_action_dim,
        "action": "[dx,dy,dz,g]",
        "action_frame": common.geometry_frame,
        "action_unit": common.action_xyz_unit,
        "maximum_xyz_norm_m": common.maximum_final_xyz_norm_m,
        "silent_clipping": False,
        "orientation_policy": "FIXED",
        "elbow_policy": "NON_POLICY",
        "direct_finger_joint_target": False,
        "hidden_8d_dataset_action": False,
        "gripper_semantic": "OPEN_CLOSE_EDGE_TRIGGERED_PERSISTENT",
        "open_command": "ENABLED_OPERATOR_ONLY",
        "close_command": "ENABLED_OPERATOR_ONLY",
        "autonomous_close": False,
        "residual_sac_during_collection": False,
        "lift": False,
        "place": False,
        "torque_control": False,
        "ee_frame": common.ee_origin_link,
        "student_privileged_input_count": 0,
        "pad_calibration_required_for_collection": False,
        "pad_calibration_required_for_human_only_bc": False,
        "pad_calibration_required_for_privileged_feasibility": True,
        "midpoint_proxy_substituted": False,
        "curobo_cube_gt_scope": "PLANNER_ONLY_NOT_STUDENT_INPUT",
    }


__all__ = [
    "KEYBOARD_V3_COLLECTION_CONTRACT_SCHEMA",
    "DEFAULT_NEAR_GRASP_BACKOFF_M",
    "COLLECTION_TARGETS_MM",
    "COLLECTION_TARGETS_M",
    "COLLECTION_TARGET_REFERENCE",
    "LOCAL_GRASP_BAND_M",
    "LOCAL_GRASP_DISTANCE_REFERENCE",
    "LOCAL_GRASP_PHASE_CODE",
    "NEAR_GRASP_BACKOFF_SET_M",
    "TELEOP_COLLECTION_AUTHORITY",
    "NearGraspCondition",
    "PersistentGripperCommand",
    "assert_candidate_a_asset",
    "collection_contract_payload",
    "classify_local_grasp_phase",
    "collection_target_conditions",
    "format_terminal_status",
    "keyboard_action",
    "near_grasp_pose_root_m_xyzw",
    "nominal_grasp_translation_residual_m",
    "pilot_conditions",
    "preclose_geometry_metrics",
    "select_next_target_mm",
    "target_distribution_counts",
]
