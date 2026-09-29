# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Frozen cross-component contract for keyboard grasp BC + residual SAC.

This module contains no simulator or training side effects.  It is the shared
authority for dataset migration, offline BC and the future 25-environment SAC
runtime so those components cannot silently disagree about time, units,
frames, dimensions or residual scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence


KEYBOARD_GRASP_PIPELINE_SCHEMA = "g2_keyboard_grasp_bc_residual_sac_v2"
LOCAL_GRASP_PHASE_CODE = {
    "FAR_HANDOFF": 0,
    "LOCAL_APPROACH": 1,
    "GRASP_DECISION_BAND": 2,
    "BELOW_GRASP_BAND_OPEN": 3,
    "CLOSE_POST": 4,
}


@dataclass(frozen=True)
class KeyboardGraspPipelineContract:
    control_hz: int = 50
    policy_dt_s: float = 0.02
    dataset_row_hz: int = 50
    rgbd_acquisition_hz: int = 25
    rgbd_dt_s: float = 0.04
    physics_hz: int = 500
    physics_dt_s: float = 0.002
    offline_bc_num_envs: int = 0
    future_sac_num_envs: int = 25

    geometry_frame: str = "robot_root"
    ee_origin_link: str = "gripper_r_center_link"
    cube_reference: str = "cube_center"
    position_unit: str = "m"
    depth_raw_unit: str = "m"
    depth_to_meter_scale: float = 1.0
    maximum_policy_depth_m: float = 2.0
    ee_position_unit: str = "m"
    cube_position_unit: str = "m"
    action_xyz_unit: str = "m"
    angle_unit: str = "rad"
    force_unit: str = "N"
    her_force_proxy_unit: str = "J"

    legacy_action_dim: int = 8
    bc_action_dim: int = 4
    sac_action_dim: int = 3
    legacy_normalized_to_metric_scale_m: float = 0.0225
    legacy_maximum_translation_m: float = 0.015
    maximum_final_xyz_norm_m: float = 0.0045
    fixed_orientation: bool = True
    elbow_is_policy_action: bool = False
    sac_has_gripper_authority: bool = False

    residual_raw_maximum_norm_m: float = 0.0045
    residual_alpha_small: float = 0.10
    residual_alpha_medium: float = 0.25

    # Planner-to-policy ownership boundary.  These are robot-root metric
    # distances relative to the cuRobo nominal grasp pose, never calibrated
    # pad-surface distances.
    curobo_handoff_band_m: tuple[float, float, float] = (0.025, 0.030, 0.035)
    curobo_default_handoff_m: float = 0.030
    local_grasp_band_m: tuple[float, float] = (0.015, 0.022)
    local_grasp_distance_reference: str = (
        "CUROBO_NOMINAL_GRASP_POSE_TRANSLATION_RESIDUAL_NOT_PAD_SURFACE"
    )

    geometric_her_enabled: bool = False
    her_force_enabled: bool = True
    pad_surface_existing_data_available: bool = False
    midpoint_proxy_substituted: bool = False

    def __post_init__(self) -> None:
        expected = {
            "control_hz": 50,
            "policy_dt_s": 0.02,
            "dataset_row_hz": 50,
            "rgbd_acquisition_hz": 25,
            "rgbd_dt_s": 0.04,
            "physics_hz": 500,
            "physics_dt_s": 0.002,
            "offline_bc_num_envs": 0,
            "future_sac_num_envs": 25,
            "geometry_frame": "robot_root",
            "ee_origin_link": "gripper_r_center_link",
            "cube_reference": "cube_center",
            "position_unit": "m",
            "depth_raw_unit": "m",
            "depth_to_meter_scale": 1.0,
            "maximum_policy_depth_m": 2.0,
            "ee_position_unit": "m",
            "cube_position_unit": "m",
            "action_xyz_unit": "m",
            "angle_unit": "rad",
            "force_unit": "N",
            "her_force_proxy_unit": "J",
            "legacy_action_dim": 8,
            "bc_action_dim": 4,
            "sac_action_dim": 3,
            "legacy_normalized_to_metric_scale_m": 0.0225,
            "legacy_maximum_translation_m": 0.015,
            "maximum_final_xyz_norm_m": 0.0045,
            "residual_raw_maximum_norm_m": 0.0045,
            "residual_alpha_small": 0.10,
            "residual_alpha_medium": 0.25,
            "curobo_handoff_band_m": (0.025, 0.030, 0.035),
            "curobo_default_handoff_m": 0.030,
            "local_grasp_band_m": (0.015, 0.022),
            "local_grasp_distance_reference": (
                "CUROBO_NOMINAL_GRASP_POSE_TRANSLATION_RESIDUAL_NOT_PAD_SURFACE"
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"frozen keyboard grasp contract mismatch: {name}")
        if not self.fixed_orientation or self.elbow_is_policy_action:
            raise ValueError("orientation/elbow policy authority mismatch")
        if self.control_hz * self.policy_dt_s != 1.0:
            raise ValueError("control frequency/period mismatch")
        if self.rgbd_acquisition_hz * self.rgbd_dt_s != 1.0:
            raise ValueError("RGB-D frequency/period mismatch")
        if self.physics_hz * self.physics_dt_s != 1.0:
            raise ValueError("physics frequency/period mismatch")
        if self.dataset_row_hz != self.control_hz:
            raise ValueError("dataset rows must follow the control clock")
        if self.control_hz != 2 * self.rgbd_acquisition_hz:
            raise ValueError("each RGB-D acquisition must span exactly two control rows")
        if self.sac_has_gripper_authority:
            raise ValueError("residual SAC cannot own gripper timing")
        if self.geometric_her_enabled or not self.her_force_enabled:
            raise ValueError("HER must be disabled and HER_FORCE mandatory")
        if self.pad_surface_existing_data_available or self.midpoint_proxy_substituted:
            raise ValueError("legacy keyboard data has no pad-surface authority")

    def residual_contribution_bound_m(self, stage: str) -> float:
        alpha = {
            "small": self.residual_alpha_small,
            "medium": self.residual_alpha_medium,
        }.get(stage)
        if alpha is None:
            raise ValueError(f"unknown residual alpha stage: {stage}")
        return alpha * self.residual_raw_maximum_norm_m

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": KEYBOARD_GRASP_PIPELINE_SCHEMA,
            **asdict(self),
            "bc_action": "[dx,dy,dz,g]",
            "sac_action": "[delta_x,delta_y,delta_z]",
            "composition": "final_xyz=bc_xyz+alpha*delta_sac_metric",
            "final_gripper": "g_bc",
            "silent_clipping": False,
            "residual_contribution_small_m": self.residual_contribution_bound_m(
                "small"
            ),
            "residual_contribution_medium_m": self.residual_contribution_bound_m(
                "medium"
            ),
        }


def canonical_keyboard_grasp_contract() -> KeyboardGraspPipelineContract:
    return KeyboardGraspPipelineContract()


def nominal_grasp_translation_residual_m(
    ee_position_root_m: Sequence[float],
    nominal_grasp_pose_root_m_xyzw: Sequence[float],
) -> float:
    """Analysis-only EE-to-nominal translation residual in robot-root metres."""

    ee = tuple(float(value) for value in ee_position_root_m)
    nominal = tuple(float(value) for value in nominal_grasp_pose_root_m_xyzw)
    if len(ee) != 3 or len(nominal) != 7 or not all(
        math.isfinite(value) for value in (*ee, *nominal)
    ):
        raise ValueError("nominal-grasp residual inputs must be finite 3D/7D values")
    return math.sqrt(sum((ee[index] - nominal[index]) ** 2 for index in range(3)))


def classify_local_grasp_phase(
    residual_m: float, *, gripper_closed: bool
) -> str:
    """Classify the analysis phase without adding distance to actor inputs."""

    residual = float(residual_m)
    if not math.isfinite(residual) or residual < 0.0:
        raise ValueError("nominal grasp residual must be finite and nonnegative")
    if gripper_closed:
        return "CLOSE_POST"
    contract = canonical_keyboard_grasp_contract()
    lower, upper = contract.local_grasp_band_m
    if residual < lower:
        return "BELOW_GRASP_BAND_OPEN"
    if residual <= upper:
        return "GRASP_DECISION_BAND"
    if residual <= contract.curobo_default_handoff_m:
        return "LOCAL_APPROACH"
    return "FAR_HANDOFF"


__all__ = [
    "KEYBOARD_GRASP_PIPELINE_SCHEMA",
    "LOCAL_GRASP_PHASE_CODE",
    "KeyboardGraspPipelineContract",
    "canonical_keyboard_grasp_contract",
    "classify_local_grasp_phase",
    "nominal_grasp_translation_residual_m",
]
