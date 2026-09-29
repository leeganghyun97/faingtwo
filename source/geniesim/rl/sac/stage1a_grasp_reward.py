# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Stage-1A privileged grasp reward for the G2 residual-SAC branch.

The actor remains deployable: it receives no contact, force, cube pose, or
contact geometry from this module.  These values are simulation-only reward
authority.  The residual actor owns only an XYZ correction; CLOSE remains the
frozen GRU-BC output.

All distances are metres in ``robot_root``.  Contact evidence is aggregated
from ten 500-Hz PhysX substeps for a normal transition, or from the exact
real prefix ending at a structured runtime hard-stop.
Force magnitude never earns positive reward: force is used only for contact
hysteresis, left/right balance, and an externally classified safety result.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .keyboard_grasp_contract import canonical_keyboard_grasp_contract
from .privileged_contact_contract import DEPLOYABLE_ACTOR_FIELDS, deployment_guard
from .residual_her_force import ResidualHERForceConfig
from .stage1a_exception_safe_transition import elapsed_discount


STAGE1A_GRASP_REWARD_SCHEMA = "g2_stage1a_privileged_grasp_reward_v2"
STAGE1A_GRASP_REWARD_V3_SCHEMA = "g2_stage1a_temporal_progress_reward_v3_smoke"
STAGE1A_GRASP_REWARD_V32_SCHEMA = "g2_stage1a_temporal_progress_reward_v3_2_bounded_smoke"
CONTROL_HZ = 50
CONTROL_DT_S = 0.020
PHYSICS_HZ = 500
PHYSICS_DT_S = 0.002
RGBD_HZ = 25
PHYSICS_SUBSTEPS_PER_CONTROL = 10
STAGE1A_SAC_PBRS_GAMMA = 0.9993


class Stage1AGraspRewardError(ValueError):
    """Raised before an invalid reward sample can enter replay."""


@dataclass(frozen=True)
class Stage1ARewardV3Config:
    """Opt-in smoke candidates; none of these values are production locked."""

    temporal_window_steps: int = 15
    cosine_window_steps: int = 5
    progress_epsilon_m: float = 0.00020
    worsen_epsilon_m: float = 0.00020
    static_range_m: float = 0.00030
    oscillation_path_m: float = 0.00080
    hover_grace_steps: int = 15
    hover_far_penalty_per_step: float = -0.0010
    hover_near_penalty_per_step: float = -0.0016
    oscillation_penalty_per_step: float = -0.0009
    worsening_small_penalty_per_step: float = -0.0002
    worsening_large_penalty_per_step: float = -0.0004
    worsening_large_m: float = 0.00040
    cosine_threshold: float = 0.80
    cosine_min_progress_m: float = 0.00005
    cosine_reward_max_per_step: float = 0.002
    progress_reward_per_mm: float = 0.04
    progress_clip_mm: float = 0.25
    lateral_reward_per_mm: float = 0.015
    lateral_reward_max_per_step: float = 0.002
    time_penalty_per_active_step: float = -0.0003
    residual_penalty_max_per_step: float = -0.0007
    smoothness_penalty_max_per_step: float = -0.0004
    milestone_20mm_reward: float = 0.03
    milestone_18mm_reward: float = 0.05
    milestone_16mm_reward: float = 0.08
    first_contact_reward: float = 0.25
    bilateral_base_reward: float = 1.0
    stable_base_reward: float = 2.0
    success_reward: float = 4.0
    stable_hold_reward: float = 0.005
    single_contact_dwell_penalty: float = -0.005

    def __post_init__(self) -> None:
        if self.temporal_window_steps != 15 or self.cosine_window_steps != 5:
            raise Stage1AGraspRewardError("Reward V3 windows must remain 15/5 steps")
        if self.hover_grace_steps != 15:
            raise Stage1AGraspRewardError("Reward V3 hover grace must remain 15 steps")
        if any(
            value > 0.0
            for value in (
                self.hover_far_penalty_per_step,
                self.hover_near_penalty_per_step,
                self.oscillation_penalty_per_step,
                self.worsening_small_penalty_per_step,
                self.worsening_large_penalty_per_step,
                self.time_penalty_per_active_step,
                self.residual_penalty_max_per_step,
                self.smoothness_penalty_max_per_step,
                self.single_contact_dwell_penalty,
            )
        ):
            raise Stage1AGraspRewardError("Reward V3 penalties must not be positive")

    def payload(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result.update(
            {
                "schema": STAGE1A_GRASP_REWARD_V3_SCHEMA,
                "authority": "BOUNDED_SMOKE_CANDIDATE_NOT_PRODUCTION_LOCKED",
                "control_hz": CONTROL_HZ,
                "temporal_window_ms": 300,
                "cosine_window_ms": 100,
                "active_band_m": [0.015, 0.022],
                "force_magnitude_positive_reward": False,
                "ordinary_her": False,
                "her_force": False,
            }
        )
        result["reward_v3_authority_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return result


@dataclass(frozen=True)
class Stage1ARewardV32Config:
    """Small, phase-scoped V3.2 additions for one bounded smoke.

    This deliberately does not alter the historical V3 configuration.  The
    only added positive shaping is a capped bilateral-maintenance signal: it
    is independent of force magnitude and can never exceed 0.020 per episode.
    """

    bilateral_stability_reward_per_step: float = 0.001
    bilateral_instability_penalty_per_step: float = -0.002
    bilateral_stability_reward_episode_cap: float = 0.020
    strengthened_single_contact_dwell_penalty: float = -0.006
    phase_aware_hover: bool = True

    def __post_init__(self) -> None:
        if not (0.001 <= self.bilateral_stability_reward_per_step <= 0.005):
            raise Stage1AGraspRewardError("V3.2 bilateral reward must stay bounded")
        if not (-0.005 <= self.bilateral_instability_penalty_per_step <= -0.001):
            raise Stage1AGraspRewardError("V3.2 bilateral penalty must stay bounded")
        if self.bilateral_stability_reward_episode_cap <= 0.0:
            raise Stage1AGraspRewardError("V3.2 bilateral reward cap must be positive")
        if self.strengthened_single_contact_dwell_penalty > -0.005:
            raise Stage1AGraspRewardError("V3.2 single-contact dwell must not weaken")

    def payload(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result.update(
            {
                "schema": STAGE1A_GRASP_REWARD_V32_SCHEMA,
                "authority": "BOUNDED_V32_SMOKE_CANDIDATE_NOT_PRODUCTION_LOCKED",
                "bilateral_stability_phase_only": True,
                "force_magnitude_positive_reward": False,
                "farming_protection": "PER_EPISODE_BILATERAL_SHAPING_CAP",
            }
        )
        result["reward_v32_authority_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return result


@dataclass(frozen=True)
class Stage1AGraspRewardConfig:
    """Frozen Stage-1A values; candidate thresholds are marked as such."""

    control_hz: int = CONTROL_HZ
    physics_hz: int = PHYSICS_HZ
    rgbd_hz: int = RGBD_HZ
    control_dt_s: float = CONTROL_DT_S
    physics_dt_s: float = PHYSICS_DT_S
    physics_substeps_per_control: int = PHYSICS_SUBSTEPS_PER_CONTROL
    gamma: float = STAGE1A_SAC_PBRS_GAMMA
    grasp_band_near_m: float = 0.015
    grasp_band_far_m: float = 0.022
    lateral_reference_m: float = 0.005
    lateral_pbrs_weight: float = 0.30
    contact_on_force_n: float = 1.0
    contact_off_force_n: float = 0.5
    contact_minimum_substeps: int = 3
    single_contact_reward: float = 0.10
    single_contact_dwell_free_steps: int = 10
    single_contact_dwell_penalty: float = -0.01
    bilateral_base_reward: float = 0.80
    bilateral_quality_weight: float = 0.40
    stable_base_reward: float = 1.50
    stable_quality_weight: float = 0.70
    success_reward: float = 3.0
    stable_consecutive_steps: int = 10
    stable_antipodal_minimum: float = 0.70
    stable_force_balance_minimum: float = 0.50
    slip_reference_m_s: float = 0.02
    impact_reference_m_s: float = 0.05
    cube_omega_reference_rad_s: float = 0.50
    residual_alpha: float = 0.10
    raw_residual_maximum_m: float = 0.0045
    effective_residual_maximum_m: float = 0.00045
    residual_penalty_weight: float = 0.03
    smoothness_penalty_weight: float = 0.01
    time_penalty_per_active_step: float = -0.002
    progress_epsilon_m: float = 0.00025
    lateral_improvement_epsilon_m: float = 0.00025
    hover_grace_steps: int = 10
    hover_penalty: float = -0.01
    wrong_direction_grace_steps: int = 5
    wrong_direction_penalty: float = -0.02
    contact_loss_grace_steps: int = 10
    # Source-owned qualification penalty.  It is derived from the existing
    # positive one-shot contact milestones, never supplied by the coordinator.
    runtime_hardstop_terminal_penalty: float = -0.90

    def __post_init__(self) -> None:
        common = canonical_keyboard_grasp_contract()
        if (self.control_hz, self.physics_hz, self.rgbd_hz) != (50, 500, 25):
            raise Stage1AGraspRewardError("Stage-1A timing must remain 50/500/25 Hz")
        if (
            self.control_hz != common.control_hz
            or self.physics_hz != common.physics_hz
            or self.rgbd_hz != common.rgbd_acquisition_hz
            or (self.grasp_band_near_m, self.grasp_band_far_m)
            != common.local_grasp_band_m
            or self.raw_residual_maximum_m != common.residual_raw_maximum_norm_m
            or self.residual_alpha != common.residual_alpha_small
        ):
            raise Stage1AGraspRewardError("Stage-1A diverged from shared grasp authority")
        if not math.isclose(self.control_dt_s, 1.0 / self.control_hz) or not math.isclose(
            self.physics_dt_s, 1.0 / self.physics_hz
        ):
            raise Stage1AGraspRewardError("Stage-1A dt metadata is inconsistent")
        if self.physics_substeps_per_control != 10 or not math.isclose(
            self.control_dt_s / self.physics_dt_s,
            float(self.physics_substeps_per_control),
        ):
            raise Stage1AGraspRewardError("one control transition must contain 10 substeps")
        if not 0.0 < self.gamma <= 1.0:
            raise Stage1AGraspRewardError("gamma must be in (0,1]")
        if not 0.0 < self.grasp_band_near_m < self.grasp_band_far_m:
            raise Stage1AGraspRewardError("invalid 15--22 mm grasp band")
        if self.contact_off_force_n >= self.contact_on_force_n:
            raise Stage1AGraspRewardError("contact hysteresis is inverted")
        if not 1 <= self.contact_minimum_substeps <= self.physics_substeps_per_control:
            raise Stage1AGraspRewardError("invalid contact persistence")
        if self.stable_consecutive_steps != 10:
            raise Stage1AGraspRewardError("stable dwell must remain 10 control steps")
        if self.single_contact_dwell_free_steps != 10 or self.hover_grace_steps != 10:
            raise Stage1AGraspRewardError("single/hover persistence must remain 10 steps")
        if self.wrong_direction_grace_steps != 5:
            raise Stage1AGraspRewardError("wrong-direction grace must remain 5 steps")
        if self.contact_loss_grace_steps != 10:
            raise Stage1AGraspRewardError("contact-loss grace must remain 10 steps")
        if self.progress_epsilon_m <= 0.0 or self.lateral_improvement_epsilon_m <= 0.0:
            raise Stage1AGraspRewardError("progress persistence epsilons must be positive")
        if not math.isclose(
            self.residual_alpha * self.raw_residual_maximum_m,
            self.effective_residual_maximum_m,
            abs_tol=1.0e-12,
        ):
            raise Stage1AGraspRewardError("residual alpha/scale contract mismatch")
        if any(
            value > 0.0
            for value in (
                self.single_contact_dwell_penalty,
                self.hover_penalty,
                self.wrong_direction_penalty,
                self.time_penalty_per_active_step,
            )
        ):
            raise Stage1AGraspRewardError("penalties must not be positive")
        expected_hardstop = -(
            self.single_contact_reward + self.bilateral_base_reward
        )
        if not math.isclose(
            self.runtime_hardstop_terminal_penalty,
            expected_hardstop,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Stage1AGraspRewardError(
                "runtime hard-stop penalty must equal -(contact+bilateral)"
            )

    def payload(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result.update(
            {
                "schema": STAGE1A_GRASP_REWARD_SCHEMA,
                "frame": "robot_root",
                "position_unit": "m",
                "force_unit": "N",
                "angular_velocity_unit": "rad/s",
                "contact_threshold_authority": "INITIAL_CANDIDATE_NOT_PRODUCTION_LOCKED",
                "lateral_reference_authority": "INITIAL_CANDIDATE_NOT_PRODUCTION_LOCKED",
                "stable_threshold_authority": "INITIAL_CANDIDATE_NOT_PRODUCTION_LOCKED",
                "force_role": "BOOLEAN_AND_QUALITY_ONLY",
                "force_magnitude_positive_reward": False,
                "geometric_her_enabled": False,
                "her_force_role": "PRIORITY_ONLY",
                "sac_action_dim": 3,
                "sac_gripper_authority": False,
                "pad_cube_contact_pairs_only": True,
                "contact_normal_convention": (
                    "ACTUAL_PHYSX_NORMAL_ORIENTED_FROM_CONTACT_POINT_TOWARD_CUBE_CENTER"
                ),
                "absolute_distance_reward": False,
                "pbrs_disabled_after_first_contact": True,
                "bilateral_required_for_high_reward": True,
                "single_contact_reward_semantics": "ONE_SHOT_SMALL",
                "bilateral_contact_reward_semantics": "ONE_SHOT_QUALITY_WEIGHTED",
                "stable_grasp_reward_semantics": "ONE_SHOT_QUALITY_WEIGHTED",
                "success_reward_semantics": "ONE_SHOT_TERMINAL",
                "stable_duration_ms": int(
                    1000 * self.stable_consecutive_steps / self.control_hz
                ),
                "persistence_counter_unit": "50_HZ_CONTROL_STEP",
                "persistence_reward_contract": True,
                "stable_threshold_production_locked": False,
                "contact_loss_terminal": True,
                "progress_epsilon_authority": (
                    "HISTORICAL_CANDIDATE_REUSED_CONTACT_FREE_PROGRESS_DEADBAND"
                ),
                "progress_epsilon_production_locked": False,
                "progress_epsilon_adjustment_authority": (
                    "FUTURE_MEASURED_SAC_TELEMETRY_REVIEW_ONLY"
                ),
                "runtime_hardstop_penalty_authority": (
                    "SOURCE_DERIVED_NEGATIVE_SUM_OF_SINGLE_CONTACT_AND_"
                    "BILATERAL_BASE_ONE_SHOT_REWARDS"
                ),
                "runtime_hardstop_penalty_formula": (
                    "-(single_contact_reward + bilateral_base_reward)"
                ),
                "runtime_hardstop_penalty_resolved": (
                    self.runtime_hardstop_terminal_penalty
                ),
            }
        )
        result["reward_authority_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return result


@dataclass(frozen=True)
class Stage1ARewardInputs:
    """One vectorized 50-Hz transition.

    Contact tensors use side order ``[left, right]`` and contain ten physics
    substeps. ``normal_impulse_available`` selects impulse integration for an
    entire side/interval; unavailable intervals use the force-time integral.
    Mixing the two sources within one interval is deliberately impossible.
    """

    current_ee_position_root_m: Tensor
    next_ee_position_root_m: Tensor
    nominal_grasp_position_root_m: Tensor
    nominal_approach_axis_root: Tensor
    stage1a_active: Tensor
    grasp_decision_phase: Tensor
    phase_reset: Tensor
    episode_reset: Tensor
    pad_cube_contact_valid: Tensor
    normal_impulse_ns: Tensor
    normal_impulse_available: Tensor
    normal_force_n: Tensor
    contact_geometry_valid: Tensor
    contact_point_root_m: Tensor
    contact_normal_root: Tensor
    tangential_relative_speed_m_s: Tensor
    normal_relative_speed_m_s: Tensor
    cube_center_root_m: Tensor
    cube_quat_root_xyzw: Tensor
    cube_half_extents_m: Tensor
    cube_linear_velocity_root_m_s: Tensor
    cube_angular_velocity_root_rad_s: Tensor
    effective_residual_root_m: Tensor
    previous_effective_residual_root_m: Tensor
    final_action_xyz_root_m: Tensor
    contract_grasp_state_valid: Tensor
    safety_violation: Tensor
    authoritative_safety_penalty: Tensor
    non_pad_gripper_cube_contact: Tensor
    close_command: Tensor
    consumed_substeps: Tensor | None = None
    runtime_hardstop: Tensor | None = None

    def actor_payload(self) -> Mapping[str, Tensor]:
        raise RuntimeError("STAGE1A_PRIVILEGED_REWARD_TO_ACTOR_FORBIDDEN")


@dataclass(frozen=True)
class Stage1ARewardStep:
    reward_total: Tensor
    done: Tensor
    success: Tensor
    reward_progress: Tensor
    reward_lateral: Tensor
    reward_single_contact: Tensor
    reward_bilateral: Tensor
    reward_stable: Tensor
    reward_success: Tensor
    penalty_single_contact_dwell: Tensor
    penalty_hover: Tensor
    penalty_wrong_direction: Tensor
    residual_penalty: Tensor
    smoothness_penalty: Tensor
    time_penalty: Tensor
    safety_penalty: Tensor
    q_antipodal: Tensor
    q_center: Tensor
    q_force_balance: Tensor
    q_slip: Tensor
    q_impact: Tensor
    q_omega: Tensor
    q_grasp: Tensor
    q_antipodal_available: Tensor
    q_center_available: Tensor
    left_contact_boolean: Tensor
    right_contact_boolean: Tensor
    single_contact_boolean: Tensor
    bilateral_boolean: Tensor
    bilateral_contact_boolean: Tensor
    stable_grasp_boolean: Tensor
    contact_loss_fail: Tensor
    bilateral_stable_counter: Tensor
    single_contact_counter: Tensor
    hover_counter: Tensor
    wrong_direction_counter: Tensor
    contact_loss_counter: Tensor
    cosine_similarity: Tensor
    progress_m: Tensor
    lateral_improvement_m: Tensor
    left_normal_force_n: Tensor
    right_normal_force_n: Tensor
    left_contact_substep_count: Tensor
    right_contact_substep_count: Tensor
    force_aggregation_used_impulse: Tensor
    tangential_slip_m_s: Tensor
    contact_normal_velocity_m_s: Tensor
    cube_linear_velocity_m_s: Tensor
    cube_angular_velocity_rad_s: Tensor
    residual_norm_mm: Tensor
    final_action_norm_mm: Tensor
    contact_latched: Tensor
    non_pad_gripper_cube_contact: Tensor
    no_progress_hovering: Tensor
    band_edge_hovering: Tensor
    runtime_hardstop: Tensor | None = None
    consumed_substeps: Tensor | None = None
    elapsed_time_s: Tensor | None = None
    gamma_effective: Tensor | None = None
    reward_cosine: Tensor | None = None
    reward_milestone_20mm: Tensor | None = None
    reward_milestone_18mm: Tensor | None = None
    reward_milestone_16mm: Tensor | None = None
    reward_stable_hold: Tensor | None = None
    penalty_oscillation: Tensor | None = None
    penalty_worsening: Tensor | None = None
    temporal_progressing: Tensor | None = None
    temporal_static_hover: Tensor | None = None
    temporal_oscillatory_hover: Tensor | None = None
    temporal_worsening: Tensor | None = None
    temporal_ambiguous: Tensor | None = None
    temporal_window_valid: Tensor | None = None
    cosine_window_valid: Tensor | None = None
    cosine_sustained: Tensor | None = None
    cosine_with_progress: Tensor | None = None
    net_progress_0p3_m: Tensor | None = None
    path_motion_0p3_m: Tensor | None = None
    range_0p3_m: Tensor | None = None
    cosine_mean_0p1: Tensor | None = None
    progress_0p1_m: Tensor | None = None
    nominal_residual_mm: Tensor | None = None
    reward_v3_active: Tensor | None = None
    reward_bilateral_stability_shaping: Tensor | None = None
    penalty_bilateral_instability: Tensor | None = None

    def telemetry(self) -> Mapping[str, Tensor]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _quat_xyzw_to_matrix(quaternion: Tensor) -> Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


class Stage1AGraspReward:
    """Vectorized stateful Stage-1A reward with monotonic event latches."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
        config: Stage1AGraspRewardConfig = Stage1AGraspRewardConfig(),
        v3_config: Stage1ARewardV3Config | None = None,
        v32_config: Stage1ARewardV32Config | None = None,
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise Stage1AGraspRewardError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.config = config
        self.v3_config = v3_config
        if v32_config is not None and v3_config is None:
            raise Stage1AGraspRewardError("Reward V3.2 requires the immutable V3 base")
        self.v32_config = v32_config
        self._shape = (num_envs,)
        self._left_contact = torch.zeros(self._shape, dtype=torch.bool, device=self.device)
        self._right_contact = torch.zeros_like(self._left_contact)
        self._contact_latched = torch.zeros_like(self._left_contact)
        self._single_emitted = torch.zeros_like(self._left_contact)
        self._bilateral_emitted = torch.zeros_like(self._left_contact)
        self._stable_emitted = torch.zeros_like(self._left_contact)
        self._success_emitted = torch.zeros_like(self._left_contact)
        self._single_dwell = torch.zeros(self._shape, dtype=torch.int64, device=self.device)
        self._bilateral_dwell = torch.zeros_like(self._single_dwell)
        self._hover_counter = torch.zeros_like(self._single_dwell)
        self._wrong_direction_counter = torch.zeros_like(self._single_dwell)
        self._contact_loss_counter = torch.zeros_like(self._single_dwell)
        self._grasp_phase_active = torch.zeros_like(self._left_contact)
        self._initial_impact_speed_m_s = torch.zeros(
            (num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._impact_recorded = torch.zeros(
            (num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._distance_history = torch.full(
            (num_envs, 16), float("nan"), dtype=torch.float32, device=self.device
        )
        self._distance_history_count = torch.zeros_like(self._single_dwell)
        self._cosine_history = torch.full(
            (num_envs, 5), float("nan"), dtype=torch.float32, device=self.device
        )
        self._cosine_history_count = torch.zeros_like(self._single_dwell)
        self._milestone_20mm_emitted = torch.zeros_like(self._left_contact)
        self._milestone_18mm_emitted = torch.zeros_like(self._left_contact)
        self._milestone_16mm_emitted = torch.zeros_like(self._left_contact)
        self._v32_bilateral_stability_shaping_paid = torch.zeros(
            self._shape, dtype=torch.float32, device=self.device
        )

    def reset(self, mask: Tensor | None = None) -> None:
        selected = (
            torch.ones(self._shape, dtype=torch.bool, device=self.device)
            if mask is None
            else self._bool(mask, "reset mask")
        )
        for value in (
            self._left_contact,
            self._right_contact,
            self._contact_latched,
            self._single_emitted,
            self._bilateral_emitted,
            self._stable_emitted,
            self._success_emitted,
            self._grasp_phase_active,
            self._milestone_20mm_emitted,
            self._milestone_18mm_emitted,
            self._milestone_16mm_emitted,
        ):
            value.masked_fill_(selected, False)
        self._single_dwell.masked_fill_(selected, 0)
        self._bilateral_dwell.masked_fill_(selected, 0)
        self._hover_counter.masked_fill_(selected, 0)
        self._wrong_direction_counter.masked_fill_(selected, 0)
        self._contact_loss_counter.masked_fill_(selected, 0)
        self._initial_impact_speed_m_s[selected] = 0.0
        self._impact_recorded[selected] = False
        self._distance_history[selected] = float("nan")
        self._distance_history_count.masked_fill_(selected, 0)
        self._cosine_history[selected] = float("nan")
        self._cosine_history_count.masked_fill_(selected, 0)
        self._v32_bilateral_stability_shaping_paid.masked_fill_(selected, 0.0)

    def _float(self, value: Tensor, name: str, shape: tuple[int, ...]) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tuple(result.shape) != shape or not bool(torch.isfinite(result).all()):
            raise Stage1AGraspRewardError(f"{name} must be finite with shape {shape}")
        return result

    def _bool(self, value: Tensor, name: str, shape: tuple[int, ...] | None = None) -> Tensor:
        result = torch.as_tensor(value, device=self.device)
        expected = self._shape if shape is None else shape
        if tuple(result.shape) != expected or result.dtype is not torch.bool:
            raise Stage1AGraspRewardError(f"{name} must be bool with shape {expected}")
        return result

    def _validate(self, sample: Stage1ARewardInputs) -> dict[str, Tensor]:
        e = self.num_envs
        normal_force_shape = tuple(torch.as_tensor(sample.normal_force_n).shape)
        if (
            len(normal_force_shape) != 3
            or normal_force_shape[:2] != (e, 2)
            or not 1 <= normal_force_shape[2] <= self.config.physics_substeps_per_control
        ):
            raise Stage1AGraspRewardError(
                "normal_force_n must be [env,2,1..10]"
            )
        s = normal_force_shape[2]
        vector_names = (
            "current_ee_position_root_m",
            "next_ee_position_root_m",
            "nominal_grasp_position_root_m",
            "nominal_approach_axis_root",
            "cube_center_root_m",
            "cube_half_extents_m",
            "cube_linear_velocity_root_m_s",
            "cube_angular_velocity_root_rad_s",
            "effective_residual_root_m",
            "previous_effective_residual_root_m",
            "final_action_xyz_root_m",
        )
        values: dict[str, Tensor] = {
            name: self._float(getattr(sample, name), name, (e, 3)) for name in vector_names
        }
        values["cube_quat_root_xyzw"] = self._float(
            sample.cube_quat_root_xyzw, "cube_quat_root_xyzw", (e, 4)
        )
        for name in (
            "stage1a_active",
            "grasp_decision_phase",
            "phase_reset",
            "episode_reset",
            "contract_grasp_state_valid",
            "safety_violation",
            "non_pad_gripper_cube_contact",
            "close_command",
        ):
            values[name] = self._bool(getattr(sample, name), name)
        values["pad_cube_contact_valid"] = self._bool(
            sample.pad_cube_contact_valid, "pad_cube_contact_valid", (e, 2, s)
        )
        values["normal_impulse_available"] = self._bool(
            sample.normal_impulse_available, "normal_impulse_available", (e, 2)
        )
        values["contact_geometry_valid"] = self._bool(
            sample.contact_geometry_valid, "contact_geometry_valid", (e, 2, s)
        )
        for name in (
            "normal_impulse_ns",
            "normal_force_n",
            "tangential_relative_speed_m_s",
            "normal_relative_speed_m_s",
        ):
            values[name] = self._float(getattr(sample, name), name, (e, 2, s))
        values["contact_point_root_m"] = self._float(
            sample.contact_point_root_m, "contact_point_root_m", (e, 2, s, 3)
        )
        values["contact_normal_root"] = self._float(
            sample.contact_normal_root, "contact_normal_root", (e, 2, s, 3)
        )
        values["authoritative_safety_penalty"] = self._float(
            sample.authoritative_safety_penalty,
            "authoritative_safety_penalty",
            (e,),
        )
        if sample.consumed_substeps is None:
            consumed = torch.full(
                (e,), s, dtype=torch.int64, device=self.device
            )
        else:
            consumed = torch.as_tensor(
                sample.consumed_substeps, dtype=torch.int64, device=self.device
            )
        if tuple(consumed.shape) != (e,) or bool((consumed != s).any()):
            raise Stage1AGraspRewardError(
                "consumed_substeps must equal the real contact interval width"
            )
        values["consumed_substeps"] = consumed
        if sample.runtime_hardstop is None:
            runtime_hardstop = torch.zeros(
                (e,), dtype=torch.bool, device=self.device
            )
        else:
            runtime_hardstop = self._bool(
                sample.runtime_hardstop, "runtime_hardstop"
            )
        if bool((~runtime_hardstop & (consumed != self.config.physics_substeps_per_control)).any()):
            raise Stage1AGraspRewardError(
                "partial interval is valid only for runtime hard-stop"
            )
        if bool((runtime_hardstop & ~values["safety_violation"]).any()):
            raise Stage1AGraspRewardError(
                "runtime hard-stop requires safety_violation"
            )
        values["runtime_hardstop"] = runtime_hardstop
        values["execution_fraction"] = consumed.to(torch.float32) / float(
            self.config.physics_substeps_per_control
        )
        values["elapsed_time_s"] = consumed.to(torch.float32) * self.config.physics_dt_s
        values["gamma_effective"] = torch.pow(
            torch.full((e,), self.config.gamma, dtype=torch.float32, device=self.device),
            values["execution_fraction"],
        )
        if bool((values["normal_impulse_ns"] < 0).any()) or bool(
            (values["normal_force_n"] < 0).any()
        ):
            raise Stage1AGraspRewardError("normal impulse/force cannot be negative")
        for name in ("tangential_relative_speed_m_s", "normal_relative_speed_m_s"):
            if bool((values[name] < 0).any()):
                raise Stage1AGraspRewardError(f"{name} cannot be negative")
        if bool((values["cube_half_extents_m"] <= 0).any()):
            raise Stage1AGraspRewardError("cube half extents must be positive metres")
        axis_norm = torch.linalg.vector_norm(values["nominal_approach_axis_root"], dim=1)
        quat_norm = torch.linalg.vector_norm(values["cube_quat_root_xyzw"], dim=1)
        if bool((torch.abs(axis_norm - 1.0) > 1.0e-4).any()):
            raise Stage1AGraspRewardError("nominal approach axis must be unit length")
        if bool((torch.abs(quat_norm - 1.0) > 1.0e-3).any()):
            raise Stage1AGraspRewardError("cube quaternion must be unit length")
        if bool((values["authoritative_safety_penalty"] > 0).any()):
            raise Stage1AGraspRewardError("authoritative safety penalty cannot be positive")
        if bool(
            ((~values["safety_violation"]) & (values["authoritative_safety_penalty"] != 0)).any()
        ):
            raise Stage1AGraspRewardError("safety penalty requires existing authority violation")
        residual_norm = torch.linalg.vector_norm(values["effective_residual_root_m"], dim=1)
        if bool((residual_norm > self.config.effective_residual_maximum_m + 1.0e-9).any()):
            raise Stage1AGraspRewardError("effective residual exceeds 0.45 mm")
        final_norm = torch.linalg.vector_norm(values["final_action_xyz_root_m"], dim=1)
        if bool((final_norm > 0.0045 + 1.0e-9).any()):
            raise Stage1AGraspRewardError("final XYZ action exceeds 4.5 mm without clipping")
        return values

    @staticmethod
    def _geometric_mean(values: Tensor, available: Tensor) -> Tensor:
        safe = torch.clamp(values, 0.0, 1.0)
        count = available.sum(dim=1)
        log_sum = torch.where(
            available,
            torch.log(torch.clamp(safe, min=1.0e-12)),
            torch.zeros_like(safe),
        ).sum(dim=1)
        return torch.where(count > 0, torch.exp(log_sum / count.clamp(min=1)), torch.zeros_like(log_sum))

    def _aggregate_contacts(self, values: Mapping[str, Tensor]) -> dict[str, Tensor]:
        valid = values["pad_cube_contact_valid"]
        impulse_available = values["normal_impulse_available"]
        impulse = values["normal_impulse_ns"] * valid
        force_impulse = values["normal_force_n"] * self.config.physics_dt_s * valid
        interval_impulse = torch.where(
            impulse_available,
            impulse.sum(dim=2),
            force_impulse.sum(dim=2),
        )
        actual_dt = values["elapsed_time_s"].unsqueeze(1)
        effective_force = interval_impulse / actual_dt
        persistence = valid.sum(dim=2)
        on = (effective_force >= self.config.contact_on_force_n) & (
            persistence >= self.config.contact_minimum_substeps
        )
        remain = (effective_force >= self.config.contact_off_force_n) & (
            persistence >= self.config.contact_minimum_substeps
        )
        previous = torch.stack((self._left_contact, self._right_contact), dim=1)
        contact = torch.where(previous, remain, on)

        geometry_valid = valid & values["contact_geometry_valid"]
        weights = torch.where(
            impulse_available.unsqueeze(-1), impulse, force_impulse
        ) * geometry_valid
        weight_sum = weights.sum(dim=2)
        geometry_available = weight_sum > 1.0e-12
        denom = weight_sum.clamp(min=1.0e-12).unsqueeze(-1)
        point = (values["contact_point_root_m"] * weights.unsqueeze(-1)).sum(dim=2) / denom
        normal_raw = (values["contact_normal_root"] * weights.unsqueeze(-1)).sum(dim=2) / denom
        normal_norm = torch.linalg.vector_norm(normal_raw, dim=2)
        geometry_available &= normal_norm > 1.0e-8
        normal = normal_raw / normal_norm.clamp(min=1.0e-8).unsqueeze(-1)
        slip_side = torch.where(
            valid,
            values["tangential_relative_speed_m_s"],
            torch.zeros_like(values["tangential_relative_speed_m_s"]),
        ).amax(dim=2)
        impact_side = torch.where(
            valid,
            values["normal_relative_speed_m_s"],
            torch.zeros_like(values["normal_relative_speed_m_s"]),
        ).amax(dim=2)
        return {
            "force": effective_force,
            "persistence": persistence,
            "contact": contact,
            "geometry_available": geometry_available,
            "point": point,
            "normal": normal,
            "slip": slip_side.amax(dim=1),
            "impact": impact_side.amax(dim=1),
            "slip_side": slip_side,
            "impact_side": impact_side,
            "impulse_available": impulse_available,
        }

    def _quality(
        self, values: Mapping[str, Tensor], contact: Mapping[str, Tensor], bilateral: Tensor
    ) -> dict[str, Tensor]:
        points, normals = contact["point"], contact["normal"]
        both_geometry = contact["geometry_available"].all(dim=1) & bilateral
        # PhysX normal sign depends on pair ordering.  Canonicalize each actual
        # normal toward the cube centre before comparing the two sides; no pad
        # midpoint or fabricated contact point is introduced.
        toward_cube = values["cube_center_root_m"].unsqueeze(1) - points
        flip = (normals * toward_cube).sum(dim=2) < 0.0
        normals = torch.where(flip.unsqueeze(-1), -normals, normals)
        contact_axis_raw = points[:, 1] - points[:, 0]
        contact_axis_norm = torch.linalg.vector_norm(contact_axis_raw, dim=1)
        antipodal_available = both_geometry & (contact_axis_norm > 1.0e-8)
        contact_axis = contact_axis_raw / contact_axis_norm.clamp(min=1.0e-8).unsqueeze(-1)
        opposition = torch.clamp(-(normals[:, 0] * normals[:, 1]).sum(dim=1), 0.0, 1.0)
        left_alignment = torch.abs((normals[:, 0] * contact_axis).sum(dim=1))
        right_alignment = torch.abs((normals[:, 1] * contact_axis).sum(dim=1))
        q_antipodal = torch.pow(
            torch.clamp(opposition * left_alignment * right_alignment, min=0.0),
            1.0 / 3.0,
        )
        q_antipodal = torch.where(antipodal_available, q_antipodal, torch.zeros_like(q_antipodal))

        rotation = _quat_xyzw_to_matrix(values["cube_quat_root_xyzw"])
        relative = points - values["cube_center_root_m"].unsqueeze(1)
        # ``rotation`` maps cube-local vectors to robot-root, so contact
        # points use its transpose for root -> cube-local conversion.
        local = torch.einsum("eij,esi->esj", rotation, relative)
        normalized_abs = torch.abs(local) / values["cube_half_extents_m"].unsqueeze(1)
        face_axis = normalized_abs.argmax(dim=2)
        axis_one_hot = torch.nn.functional.one_hot(face_axis, num_classes=3).to(local.dtype)
        tangential_normalized = (local / values["cube_half_extents_m"].unsqueeze(1)) * (
            1.0 - axis_one_hot
        )
        radial = torch.linalg.vector_norm(tangential_normalized, dim=2) / math.sqrt(2.0)
        centered_each = torch.clamp(1.0 - radial, 0.0, 1.0)
        q_center = torch.sqrt(centered_each[:, 0] * centered_each[:, 1])
        center_available = both_geometry
        q_center = torch.where(center_available, q_center, torch.zeros_like(q_center))

        force = contact["force"]
        q_force_balance = torch.clamp(
            1.0 - torch.abs(force[:, 0] - force[:, 1]) / (force.sum(dim=1) + 1.0e-6),
            0.0,
            1.0,
        )
        q_slip = torch.exp(-torch.square(contact["slip"] / self.config.slip_reference_m_s))
        q_impact = torch.exp(-torch.square(contact["impact"] / self.config.impact_reference_m_s))
        omega = torch.linalg.vector_norm(values["cube_angular_velocity_root_rad_s"], dim=1)
        q_omega = torch.exp(-torch.square(omega / self.config.cube_omega_reference_rad_s))
        components = torch.stack(
            (q_antipodal, q_center, q_force_balance, q_slip, q_impact, q_omega), dim=1
        )
        always = bilateral.unsqueeze(1).expand(-1, 4)
        available = torch.cat(
            (antipodal_available[:, None], center_available[:, None], always), dim=1
        )
        q_grasp = torch.where(
            bilateral, self._geometric_mean(components, available), torch.zeros_like(omega)
        )
        return {
            "q_antipodal": q_antipodal,
            "q_center": q_center,
            "q_force_balance": torch.where(bilateral, q_force_balance, torch.zeros_like(q_force_balance)),
            "q_slip": torch.where(bilateral, q_slip, torch.zeros_like(q_slip)),
            "q_impact": torch.where(bilateral, q_impact, torch.zeros_like(q_impact)),
            "q_omega": torch.where(bilateral, q_omega, torch.zeros_like(q_omega)),
            "q_grasp": q_grasp,
            "antipodal_available": antipodal_available,
            "center_available": center_available,
            "omega": omega,
        }

    def step(self, sample: Stage1ARewardInputs) -> Stage1ARewardStep:
        values = self._validate(sample)
        reset = values["episode_reset"]
        self.reset(reset)
        contact = self._aggregate_contacts(values)
        left, right = contact["contact"][:, 0], contact["contact"][:, 1]
        current_sample = ~reset
        left &= current_sample
        right &= current_sample
        side_contact = torch.stack((left, right), dim=1)
        previous_side_contact = torch.stack(
            (self._left_contact, self._right_contact), dim=1
        )
        new_side_impact = side_contact & ~previous_side_contact & ~self._impact_recorded
        self._initial_impact_speed_m_s.copy_(
            torch.where(
                new_side_impact,
                contact["impact_side"],
                self._initial_impact_speed_m_s,
            )
        )
        self._impact_recorded |= new_side_impact
        # The impact term describes contact acquisition, not quiet force
        # maintenance ten steps later.  Preserve each side's first measured
        # impact for all subsequent quality milestones in the episode.
        contact["impact"] = torch.where(
            self._impact_recorded.any(dim=1),
            self._initial_impact_speed_m_s.amax(dim=1),
            contact["impact"],
        )
        single = torch.logical_xor(left, right)
        bilateral = left & right
        first_valid_contact = left | right
        new_first_valid_contact = first_valid_contact & ~self._contact_latched
        self._contact_latched |= first_valid_contact
        contact_loss_condition = (
            self._contact_latched
            & ~left
            & ~right
            & current_sample
        )
        self._contact_loss_counter = torch.where(
            contact_loss_condition,
            self._contact_loss_counter + 1,
            torch.zeros_like(self._contact_loss_counter),
        )
        contact_loss_fail = (
            self._contact_loss_counter >= self.config.contact_loss_grace_steps
        )

        current_error = values["nominal_grasp_position_root_m"] - values["current_ee_position_root_m"]
        next_error = values["nominal_grasp_position_root_m"] - values["next_ee_position_root_m"]
        current_distance = torch.linalg.vector_norm(current_error, dim=1)
        next_distance = torch.linalg.vector_norm(next_error, dim=1)
        progress_m = current_distance - next_distance
        phase_distance_valid = (
            (current_distance >= self.config.grasp_band_near_m - 1.0e-6)
            & (current_distance <= self.config.grasp_band_far_m + 1.0e-6)
        )
        if bool((values["grasp_decision_phase"] & ~phase_distance_valid).any()):
            raise Stage1AGraspRewardError(
                "GRASP_DECISION phase requires 15--22 mm nominal residual"
            )
        denominator = self.config.grasp_band_far_m - self.config.grasp_band_near_m
        phi_current = torch.clamp(
            (self.config.grasp_band_far_m - current_distance) / denominator, 0.0, 1.0
        )
        phi_next = torch.clamp(
            (self.config.grasp_band_far_m - next_distance) / denominator, 0.0, 1.0
        )
        axis = values["nominal_approach_axis_root"]
        lateral_current_vector = current_error - (current_error * axis).sum(dim=1, keepdim=True) * axis
        lateral_next_vector = next_error - (next_error * axis).sum(dim=1, keepdim=True) * axis
        lateral_current = torch.linalg.vector_norm(lateral_current_vector, dim=1)
        lateral_next = torch.linalg.vector_norm(lateral_next_vector, dim=1)
        lateral_improvement_m = lateral_current - lateral_next
        lateral_phi_current = torch.clamp(
            1.0 - lateral_current / self.config.lateral_reference_m, 0.0, 1.0
        )
        lateral_phi_next = torch.clamp(
            1.0 - lateral_next / self.config.lateral_reference_m, 0.0, 1.0
        )
        automatic_phase_transition = (
            values["grasp_decision_phase"] != self._grasp_phase_active
        )
        phase_transition = values["phase_reset"] | automatic_phase_transition
        shaping_allowed = (
            values["stage1a_active"]
            & values["grasp_decision_phase"]
            & ~phase_transition
            & ~self._contact_latched
            & current_sample
        )
        reward_progress = torch.where(
            shaping_allowed,
            values["gamma_effective"] * phi_next - phi_current,
            torch.zeros_like(phi_current),
        )
        reward_lateral = torch.where(
            shaping_allowed,
            self.config.lateral_pbrs_weight
            * (values["gamma_effective"] * lateral_phi_next - lateral_phi_current),
            torch.zeros_like(phi_current),
        )

        # A RuntimeHardstop may occur after a real contact milestone inside
        # the same partial interval.  Preserve that observed event and apply
        # the terminal penalty separately; generic safety failures still
        # cannot earn a new milestone.
        milestone_safety_eligible = (
            ~values["safety_violation"] | values["runtime_hardstop"]
        )
        new_single = (
            single
            & ~self._single_emitted
            & ~self._bilateral_emitted
            & milestone_safety_eligible
            & current_sample
        )
        self._single_emitted |= single
        reward_single = new_single.to(torch.float32) * self.config.single_contact_reward
        self._single_dwell = torch.where(single, self._single_dwell + 1, torch.zeros_like(self._single_dwell))
        penalty_single_contact_dwell = (
            single & (self._single_dwell > self.config.single_contact_dwell_free_steps)
        ).to(torch.float32) * self.config.single_contact_dwell_penalty

        quality = self._quality(values, contact, bilateral)
        new_bilateral = (
            bilateral
            & ~self._bilateral_emitted
            & milestone_safety_eligible
            & current_sample
        )
        self._bilateral_emitted |= bilateral
        reward_bilateral = new_bilateral.to(torch.float32) * (
            self.config.bilateral_base_reward
            + self.config.bilateral_quality_weight * quality["q_grasp"]
        )
        stable_candidate = (
            bilateral
            & (contact["force"] >= self.config.contact_on_force_n).all(dim=1)
            & quality["antipodal_available"]
            & (quality["q_antipodal"] >= self.config.stable_antipodal_minimum)
            & (quality["q_force_balance"] >= self.config.stable_force_balance_minimum)
            & (contact["slip"] <= self.config.slip_reference_m_s)
            & (quality["omega"] <= self.config.cube_omega_reference_rad_s)
            & milestone_safety_eligible
            & current_sample
        )
        self._bilateral_dwell = torch.where(
            stable_candidate, self._bilateral_dwell + 1, torch.zeros_like(self._bilateral_dwell)
        )
        stable = stable_candidate & (
            self._bilateral_dwell >= self.config.stable_consecutive_steps
        )
        new_stable = stable & ~self._stable_emitted
        self._stable_emitted |= stable
        reward_stable = new_stable.to(torch.float32) * (
            self.config.stable_base_reward
            + self.config.stable_quality_weight * quality["q_grasp"]
        )
        success = (
            stable
            & values["contract_grasp_state_valid"]
            & ~self._success_emitted
            & ~values["runtime_hardstop"]
        )
        self._success_emitted |= success
        reward_success = success.to(torch.float32) * self.config.success_reward

        active = values["stage1a_active"] & current_sample
        final_command = values["final_action_xyz_root_m"]
        final_norm = torch.linalg.vector_norm(final_command, dim=1)
        target_error_norm = torch.linalg.vector_norm(current_error, dim=1)
        cosine_similarity = (final_command * current_error).sum(dim=1) / (
            final_norm * target_error_norm + 1.0e-12
        )
        new_contact_milestone = new_single | new_bilateral | new_stable | success
        hover_condition = (
            active
            & ~phase_transition
            & (torch.abs(progress_m) < self.config.progress_epsilon_m)
            & (
                lateral_improvement_m
                < self.config.lateral_improvement_epsilon_m
            )
            & ~new_contact_milestone
        )
        if self.v3_config is None:
            self._hover_counter = torch.where(
                hover_condition,
                self._hover_counter + 1,
                torch.zeros_like(self._hover_counter),
            )
            penalty_hover = (
                self._hover_counter > self.config.hover_grace_steps
            ).to(torch.float32) * self.config.hover_penalty
        else:
            penalty_hover = torch.zeros_like(progress_m)
        wrong_direction_condition = (
            active
            & ~phase_transition
            & (cosine_similarity < 0.0)
        )
        if self.v3_config is None:
            self._wrong_direction_counter = torch.where(
                wrong_direction_condition,
                self._wrong_direction_counter + 1,
                torch.zeros_like(self._wrong_direction_counter),
            )
            penalty_wrong_direction = (
                self._wrong_direction_counter > self.config.wrong_direction_grace_steps
            ).to(torch.float32) * self.config.wrong_direction_penalty
        else:
            self._wrong_direction_counter.zero_()
            penalty_wrong_direction = torch.zeros_like(progress_m)
        residual_norm = torch.linalg.vector_norm(values["effective_residual_root_m"], dim=1)
        residual_delta = torch.linalg.vector_norm(
            values["effective_residual_root_m"] - values["previous_effective_residual_root_m"],
            dim=1,
        )
        residual_penalty = torch.where(
            active,
            -self.config.residual_penalty_weight
            * torch.square(residual_norm / self.config.effective_residual_maximum_m),
            torch.zeros_like(residual_norm),
        )
        smoothness_penalty = torch.where(
            active,
            -self.config.smoothness_penalty_weight
            * torch.square(residual_delta / self.config.effective_residual_maximum_m),
            torch.zeros_like(residual_norm),
        )
        time_penalty = (
            active.to(torch.float32)
            * self.config.time_penalty_per_active_step
            * values["execution_fraction"]
        )
        safety_penalty = values["authoritative_safety_penalty"] + (
            values["runtime_hardstop"].to(torch.float32)
            * self.config.runtime_hardstop_terminal_penalty
        )
        # Reward V3 is an explicit bounded-smoke opt-in.  V2 remains byte-for-
        # byte the default behavior for historical 1K/15K artifacts.
        zero = torch.zeros_like(residual_norm)
        reward_cosine = zero
        reward_milestone_20mm = zero
        reward_milestone_18mm = zero
        reward_milestone_16mm = zero
        reward_stable_hold = zero
        reward_bilateral_stability_shaping = zero
        penalty_bilateral_instability = zero
        penalty_oscillation = zero
        penalty_worsening = zero
        temporal_progressing = torch.zeros_like(active)
        temporal_static_hover = torch.zeros_like(active)
        temporal_oscillatory_hover = torch.zeros_like(active)
        temporal_worsening = torch.zeros_like(active)
        temporal_ambiguous = torch.zeros_like(active)
        temporal_window_valid = torch.zeros_like(active)
        cosine_window_valid = torch.zeros_like(active)
        cosine_sustained = torch.zeros_like(active)
        cosine_with_progress = torch.zeros_like(active)
        net_progress_0p3_m = zero
        path_motion_0p3_m = zero
        range_0p3_m = zero
        cosine_mean_0p1 = zero
        progress_0p1_m = zero
        reward_v3_active = torch.zeros_like(active)
        if self.v3_config is not None:
            v3 = self.v3_config
            reward_v3_active = shaping_allowed

            clear_temporal = ~reward_v3_active | phase_transition
            self._distance_history[clear_temporal] = float("nan")
            self._distance_history_count.masked_fill_(clear_temporal, 0)
            self._cosine_history[clear_temporal] = float("nan")
            self._cosine_history_count.masked_fill_(clear_temporal, 0)

            first_distance = reward_v3_active & (self._distance_history_count == 0)
            continuing_distance = reward_v3_active & ~first_distance
            if bool(continuing_distance.any()):
                self._distance_history[continuing_distance] = torch.roll(
                    self._distance_history[continuing_distance], shifts=-1, dims=1
                )
                self._distance_history[continuing_distance, -1] = next_distance[
                    continuing_distance
                ]
                self._distance_history_count[continuing_distance] = torch.clamp(
                    self._distance_history_count[continuing_distance] + 1, max=16
                )
            if bool(first_distance.any()):
                self._distance_history[first_distance, -2] = current_distance[
                    first_distance
                ]
                self._distance_history[first_distance, -1] = next_distance[
                    first_distance
                ]
                self._distance_history_count[first_distance] = 2

            continuing_cosine = reward_v3_active & (self._cosine_history_count > 0)
            first_cosine = reward_v3_active & ~continuing_cosine
            if bool(continuing_cosine.any()):
                self._cosine_history[continuing_cosine] = torch.roll(
                    self._cosine_history[continuing_cosine], shifts=-1, dims=1
                )
                self._cosine_history[continuing_cosine, -1] = cosine_similarity[
                    continuing_cosine
                ]
                self._cosine_history_count[continuing_cosine] = torch.clamp(
                    self._cosine_history_count[continuing_cosine] + 1, max=5
                )
            if bool(first_cosine.any()):
                self._cosine_history[first_cosine, -1] = cosine_similarity[first_cosine]
                self._cosine_history_count[first_cosine] = 1

            temporal_window_valid = reward_v3_active & (
                self._distance_history_count >= 16
            )
            cosine_window_valid = reward_v3_active & (
                self._cosine_history_count >= v3.cosine_window_steps
            )
            safe_distance_history = torch.nan_to_num(self._distance_history, nan=0.0)
            net_progress_0p3_m = torch.where(
                temporal_window_valid,
                safe_distance_history[:, 0] - safe_distance_history[:, -1],
                zero,
            )
            path_motion_0p3_m = torch.where(
                temporal_window_valid,
                torch.abs(safe_distance_history[:, 1:] - safe_distance_history[:, :-1]).sum(dim=1),
                zero,
            )
            range_0p3_m = torch.where(
                temporal_window_valid,
                safe_distance_history.amax(dim=1) - safe_distance_history.amin(dim=1),
                zero,
            )
            progress_0p1_m = torch.where(
                temporal_window_valid,
                safe_distance_history[:, -6] - safe_distance_history[:, -1],
                zero,
            )
            safe_cosine_history = torch.nan_to_num(self._cosine_history, nan=-1.0)
            cosine_mean_0p1 = torch.where(
                cosine_window_valid, safe_cosine_history.mean(dim=1), zero
            )
            cosine_sustained = cosine_window_valid & (
                safe_cosine_history >= v3.cosine_threshold
            ).all(dim=1)
            cosine_with_progress = cosine_sustained & (
                progress_0p1_m >= v3.cosine_min_progress_m
            )

            temporal_worsening = temporal_window_valid & (
                net_progress_0p3_m < -v3.worsen_epsilon_m
            )
            near_zero_net = torch.abs(net_progress_0p3_m) <= v3.progress_epsilon_m
            temporal_oscillatory_hover = (
                temporal_window_valid
                & ~temporal_worsening
                & near_zero_net
                & (path_motion_0p3_m >= v3.oscillation_path_m)
            )
            temporal_static_hover = (
                temporal_window_valid
                & ~temporal_worsening
                & ~temporal_oscillatory_hover
                & near_zero_net
                & (range_0p3_m <= v3.static_range_m)
            )
            temporal_progressing = (
                temporal_window_valid
                & ~temporal_worsening
                & ~temporal_oscillatory_hover
                & ~temporal_static_hover
                & (net_progress_0p3_m > v3.progress_epsilon_m)
            )
            temporal_ambiguous = temporal_window_valid & ~(
                temporal_worsening
                | temporal_oscillatory_hover
                | temporal_static_hover
                | temporal_progressing
            )

            hover_condition = temporal_static_hover | temporal_oscillatory_hover
            # V3.2 must not classify low-speed contact maintenance as the
            # pre-CLOSE hover exploit.  Single-contact dwell remains its own
            # authority, while bilateral/stable motion is evaluated only by
            # the explicitly bounded stability terms below.
            if self.v32_config is not None and self.v32_config.phase_aware_hover:
                hover_condition = hover_condition & ~self._contact_latched
            self._hover_counter = torch.where(
                hover_condition,
                self._hover_counter + 1,
                torch.zeros_like(self._hover_counter),
            )
            hover_penalty_active = self._hover_counter > v3.hover_grace_steps
            hover_rate = torch.where(
                next_distance < 0.019,
                torch.full_like(next_distance, v3.hover_near_penalty_per_step),
                torch.full_like(next_distance, v3.hover_far_penalty_per_step),
            )
            penalty_hover = torch.where(hover_penalty_active, hover_rate, zero)
            penalty_oscillation = torch.where(
                hover_penalty_active & temporal_oscillatory_hover,
                torch.full_like(zero, v3.oscillation_penalty_per_step),
                zero,
            )
            worsening_rate = torch.where(
                -net_progress_0p3_m > v3.worsening_large_m,
                torch.full_like(zero, v3.worsening_large_penalty_per_step),
                torch.full_like(zero, v3.worsening_small_penalty_per_step),
            )
            penalty_worsening = torch.where(temporal_worsening, worsening_rate, zero)

            reward_progress = torch.where(
                reward_v3_active,
                v3.progress_reward_per_mm
                * torch.clamp(progress_m * 1000.0, min=0.0, max=v3.progress_clip_mm),
                zero,
            )
            reward_lateral = torch.where(
                reward_v3_active & (progress_m > 0.0),
                torch.clamp(
                    v3.lateral_reward_per_mm
                    * torch.clamp(lateral_improvement_m * 1000.0, min=0.0),
                    max=v3.lateral_reward_max_per_step,
                ),
                zero,
            )
            reward_cosine = torch.where(
                reward_v3_active & cosine_with_progress,
                v3.cosine_reward_max_per_step
                * torch.clamp(
                    (cosine_mean_0p1 - v3.cosine_threshold)
                    / (1.0 - v3.cosine_threshold),
                    0.0,
                    1.0,
                ),
                zero,
            )

            def _crossing_reward(
                threshold_m: float, emitted: Tensor, amount: float
            ) -> Tensor:
                event = (
                    reward_v3_active
                    & (current_distance > threshold_m)
                    & (next_distance <= threshold_m)
                    & ~emitted
                )
                emitted |= event
                return event.to(torch.float32) * amount

            reward_milestone_20mm = _crossing_reward(
                0.020, self._milestone_20mm_emitted, v3.milestone_20mm_reward
            )
            reward_milestone_18mm = _crossing_reward(
                0.018, self._milestone_18mm_emitted, v3.milestone_18mm_reward
            )
            reward_milestone_16mm = _crossing_reward(
                0.016, self._milestone_16mm_emitted, v3.milestone_16mm_reward
            )

            reward_single = (
                new_first_valid_contact
                & milestone_safety_eligible
                & current_sample
            ).to(torch.float32) * v3.first_contact_reward
            reward_bilateral = new_bilateral.to(torch.float32) * v3.bilateral_base_reward
            reward_stable = new_stable.to(torch.float32) * v3.stable_base_reward
            reward_stable_hold = (stable & ~new_stable).to(torch.float32) * v3.stable_hold_reward
            reward_success = success.to(torch.float32) * v3.success_reward
            penalty_single_contact_dwell = (
                single & (self._single_dwell > self.config.single_contact_dwell_free_steps)
            ).to(torch.float32) * (
                self.v32_config.strengthened_single_contact_dwell_penalty
                if self.v32_config is not None
                else v3.single_contact_dwell_penalty
            )
            if self.v32_config is not None:
                # This is deliberately contact-quality based, never force
                # magnitude based.  A small positive signal is available
                # only while bilateral contact is retained and all three
                # stability proxies are within the existing contract's
                # references.  Its fixed per-episode budget prevents a
                # stationary bilateral-contact farming loop.
                bilateral_quiet = (
                    bilateral
                    & ~stable
                    & (contact["slip"] <= self.config.slip_reference_m_s)
                    & (contact["impact"] <= self.config.impact_reference_m_s)
                    & (quality["omega"] <= self.config.cube_omega_reference_rad_s)
                )
                remaining_budget = torch.clamp(
                    self.v32_config.bilateral_stability_reward_episode_cap
                    - self._v32_bilateral_stability_shaping_paid,
                    min=0.0,
                )
                reward_bilateral_stability_shaping = torch.where(
                    bilateral_quiet & (remaining_budget > 0.0),
                    torch.minimum(
                        torch.full_like(
                            zero,
                            self.v32_config.bilateral_stability_reward_per_step,
                        ),
                        remaining_budget,
                    ),
                    zero,
                )
                self._v32_bilateral_stability_shaping_paid += (
                    reward_bilateral_stability_shaping
                )
                bilateral_unstable = bilateral & ~stable & ~bilateral_quiet
                penalty_bilateral_instability = torch.where(
                    bilateral_unstable,
                    torch.full_like(
                        zero,
                        self.v32_config.bilateral_instability_penalty_per_step,
                    ),
                    zero,
                )
            penalty_wrong_direction = zero
            residual_penalty = torch.where(
                active,
                v3.residual_penalty_max_per_step
                * torch.square(residual_norm / self.config.effective_residual_maximum_m),
                zero,
            )
            smoothness_penalty = torch.where(
                active,
                v3.smoothness_penalty_max_per_step
                * torch.square(residual_delta / self.config.effective_residual_maximum_m),
                zero,
            )
            time_penalty = (
                active.to(torch.float32)
                * v3.time_penalty_per_active_step
                * values["execution_fraction"]
            )
        reward_total = (
            reward_progress
            + reward_lateral
            + reward_cosine
            + reward_milestone_20mm
            + reward_milestone_18mm
            + reward_milestone_16mm
            + reward_single
            + reward_bilateral
            + reward_stable
            + reward_stable_hold
            + reward_bilateral_stability_shaping
            + penalty_bilateral_instability
            + reward_success
            + penalty_single_contact_dwell
            + penalty_hover
            + penalty_oscillation
            + penalty_worsening
            + penalty_wrong_direction
            + residual_penalty
            + smoothness_penalty
            + time_penalty
            + safety_penalty
        )
        done = (
            success
            | values["safety_violation"]
            | contact_loss_fail
            | values["runtime_hardstop"]
        )
        hover = hover_condition
        band_edge = active & ~first_valid_contact & (
            ((current_distance - self.config.grasp_band_near_m).abs() <= 0.00025)
            | ((current_distance - self.config.grasp_band_far_m).abs() <= 0.00025)
        ) & hover

        self._left_contact.copy_(left)
        self._right_contact.copy_(right)
        self._grasp_phase_active.copy_(
            values["grasp_decision_phase"] & current_sample
        )
        cube_linear_speed = torch.linalg.vector_norm(values["cube_linear_velocity_root_m_s"], dim=1)
        return Stage1ARewardStep(
            reward_total=reward_total,
            done=done,
            success=success,
            reward_progress=reward_progress,
            reward_lateral=reward_lateral,
            reward_single_contact=reward_single,
            reward_bilateral=reward_bilateral,
            reward_stable=reward_stable,
            reward_success=reward_success,
            penalty_single_contact_dwell=penalty_single_contact_dwell,
            penalty_hover=penalty_hover,
            penalty_wrong_direction=penalty_wrong_direction,
            residual_penalty=residual_penalty,
            smoothness_penalty=smoothness_penalty,
            time_penalty=time_penalty,
            safety_penalty=safety_penalty,
            q_antipodal=quality["q_antipodal"],
            q_center=quality["q_center"],
            q_force_balance=quality["q_force_balance"],
            q_slip=quality["q_slip"],
            q_impact=quality["q_impact"],
            q_omega=quality["q_omega"],
            q_grasp=quality["q_grasp"],
            q_antipodal_available=quality["antipodal_available"],
            q_center_available=quality["center_available"],
            left_contact_boolean=left,
            right_contact_boolean=right,
            single_contact_boolean=single,
            bilateral_boolean=bilateral,
            bilateral_contact_boolean=bilateral,
            stable_grasp_boolean=stable,
            contact_loss_fail=contact_loss_fail,
            bilateral_stable_counter=self._bilateral_dwell.clone(),
            single_contact_counter=self._single_dwell.clone(),
            hover_counter=self._hover_counter.clone(),
            wrong_direction_counter=self._wrong_direction_counter.clone(),
            contact_loss_counter=self._contact_loss_counter.clone(),
            cosine_similarity=cosine_similarity,
            progress_m=progress_m,
            lateral_improvement_m=lateral_improvement_m,
            left_normal_force_n=contact["force"][:, 0],
            right_normal_force_n=contact["force"][:, 1],
            left_contact_substep_count=contact["persistence"][:, 0],
            right_contact_substep_count=contact["persistence"][:, 1],
            force_aggregation_used_impulse=contact["impulse_available"],
            tangential_slip_m_s=contact["slip"],
            contact_normal_velocity_m_s=contact["impact"],
            cube_linear_velocity_m_s=cube_linear_speed,
            cube_angular_velocity_rad_s=quality["omega"],
            residual_norm_mm=residual_norm * 1000.0,
            final_action_norm_mm=final_norm * 1000.0,
            contact_latched=self._contact_latched.clone(),
            non_pad_gripper_cube_contact=values["non_pad_gripper_cube_contact"],
            no_progress_hovering=hover,
            band_edge_hovering=band_edge,
            runtime_hardstop=values["runtime_hardstop"],
            consumed_substeps=values["consumed_substeps"],
            elapsed_time_s=values["elapsed_time_s"],
            gamma_effective=values["gamma_effective"],
            reward_cosine=reward_cosine,
            reward_milestone_20mm=reward_milestone_20mm,
            reward_milestone_18mm=reward_milestone_18mm,
            reward_milestone_16mm=reward_milestone_16mm,
            reward_stable_hold=reward_stable_hold,
            reward_bilateral_stability_shaping=reward_bilateral_stability_shaping,
            penalty_bilateral_instability=penalty_bilateral_instability,
            penalty_oscillation=penalty_oscillation,
            penalty_worsening=penalty_worsening,
            temporal_progressing=temporal_progressing,
            temporal_static_hover=temporal_static_hover,
            temporal_oscillatory_hover=temporal_oscillatory_hover,
            temporal_worsening=temporal_worsening,
            temporal_ambiguous=temporal_ambiguous,
            temporal_window_valid=temporal_window_valid,
            cosine_window_valid=cosine_window_valid,
            cosine_sustained=cosine_sustained,
            cosine_with_progress=cosine_with_progress,
            net_progress_0p3_m=net_progress_0p3_m,
            path_motion_0p3_m=path_motion_0p3_m,
            range_0p3_m=range_0p3_m,
            cosine_mean_0p1=cosine_mean_0p1,
            progress_0p1_m=progress_0p1_m,
            nominal_residual_mm=next_distance * 1000.0,
            reward_v3_active=reward_v3_active,
        )


def stage1a_reward_contract(
    *, reward_v3: bool = False, reward_v32: bool = False
) -> dict[str, Any]:
    """Machine-readable proof of actor/replay boundaries."""

    # Reuse the source-owned deployable field allow-list.  The keyboard grasp
    # contract owns units/timing/action shape, while this list owns the
    # actor-versus-privileged observation boundary.
    canonical_keyboard_grasp_contract()
    actor_fields = DEPLOYABLE_ACTOR_FIELDS
    guard = deployment_guard(actor_fields)
    if not guard["passed"]:
        raise Stage1AGraspRewardError("student observation contains privileged reward data")
    her_force = ResidualHERForceConfig().payload()
    result = {
        **Stage1AGraspRewardConfig().payload(),
        "student_privileged_input_count": guard["student_privileged_input_count"],
        "actor_input_fields": tuple(actor_fields),
        "privileged_reward_authority": True,
        "privileged_student_observation": False,
        "her_force": her_force,
    }
    if reward_v32 and not reward_v3:
        raise Stage1AGraspRewardError("Reward V3.2 contract requires Reward V3")
    if reward_v3:
        result["reward_v3"] = Stage1ARewardV3Config().payload()
    if reward_v32:
        result["reward_v32"] = Stage1ARewardV32Config().payload()
    return result


def summarize_reward_components(
    episodes: Sequence[Sequence[Stage1ARewardStep]],
) -> dict[str, dict[str, Any]]:
    """Offline scale audit; never changes weights or runtime state."""

    component_names = (
        "reward_total",
        "reward_progress",
        "reward_lateral",
        "reward_single_contact",
        "reward_bilateral",
        "reward_stable",
        "reward_bilateral_stability_shaping",
        "reward_success",
        "penalty_single_contact_dwell",
        "penalty_bilateral_instability",
        "penalty_hover",
        "penalty_wrong_direction",
        "residual_penalty",
        "smoothness_penalty",
        "time_penalty",
        "safety_penalty",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in component_names:
        flattened: list[float] = []
        episode_sums: list[float] = []
        for episode in episodes:
            values = np.concatenate(
                [getattr(step, name).detach().cpu().numpy().reshape(-1) for step in episode]
            ) if episode else np.asarray([], dtype=np.float64)
            flattened.extend(float(value) for value in values)
            episode_sums.append(float(values.sum()))
        array = np.asarray(flattened, dtype=np.float64)
        if array.size == 0:
            raise Stage1AGraspRewardError("scale audit requires at least one reward row")
        result[name] = {
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
            "episode_sum": episode_sums,
            "sum_per_episode_mean": float(np.mean(episode_sums)),
            "sum_per_episode_p50": float(np.percentile(episode_sums, 50)),
            "sum_per_episode_p95": float(np.percentile(episode_sums, 95)),
            "sum_per_episode_max": float(np.max(episode_sums)),
        }
    return result


__all__ = [
    "CONTROL_DT_S",
    "CONTROL_HZ",
    "PHYSICS_DT_S",
    "PHYSICS_HZ",
    "PHYSICS_SUBSTEPS_PER_CONTROL",
    "RGBD_HZ",
    "STAGE1A_GRASP_REWARD_SCHEMA",
    "STAGE1A_GRASP_REWARD_V3_SCHEMA",
    "STAGE1A_GRASP_REWARD_V32_SCHEMA",
    "Stage1AGraspReward",
    "Stage1AGraspRewardConfig",
    "Stage1ARewardV3Config",
    "Stage1ARewardV32Config",
    "Stage1AGraspRewardError",
    "Stage1ARewardInputs",
    "Stage1ARewardStep",
    "stage1a_reward_contract",
    "summarize_reward_components",
]
