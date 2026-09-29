# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""G2 keyboard teleoperation and demonstration-dataset contracts.

The Isaac Lab ``Se3Keyboard`` emits physical task-space deltas while the G2
Lift action term consumes normalized commands.  Keeping this conversion in a
small, testable adapter prevents applying the environment action scale twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import torch


G2_TELEOP_ACTION_DIM = 7
G2_ARM_ACTION_DIM = 6
# Translation and rotation are deliberately separate physical scales.  A
# terminal key requests 22.5 mm while rotation uses its independently audited
# 281.25 mrad scale. Both are exactly 1.5 times the preceding contract;
# using one shared constant would silently couple translation and rotation.
G2_TRANSLATION_ACTION_SCALE_M = 0.0225
G2_ROTATION_ACTION_SCALE_RAD = 0.28125
# Operator-key sensitivity is intentionally separate from the maximum
# teacher/student rotation-vector action scale.  The last live keyboard run
# used 0.005 rad per key; 0.0075 rad is exactly 1.5 times that response while
# remaining well inside the normalized action contract.
G2_KEYBOARD_ROTATION_STEP_RAD = 0.0075
# Compatibility alias for consumers that only describe the historical
# rotation-vector scale.  New code must use the dimension-specific names.
G2_ARM_ACTION_SCALE = G2_ROTATION_ACTION_SCALE_RAD
G2_REDUNDANCY_TELEOP_ACTION_DIM = 8


def contact_latched_position_target(
    desired: torch.Tensor,
    latched_target: torch.Tensor,
    hold_mask: torch.Tensor,
) -> torch.Tensor:
    """Select a previously emitted actuator target for contact-held rows."""

    if desired.ndim != 2 or latched_target.shape != desired.shape:
        raise ValueError("desired and latched gripper targets must be equal 2-D tensors")
    if hold_mask.shape != (desired.shape[0],) or hold_mask.dtype != torch.bool:
        raise ValueError("gripper contact hold mask must be boolean [N]")
    return torch.where(hold_mask.unsqueeze(-1), latched_target, desired)


def reset_safe_gripper_open_mask(
    actions: torch.Tensor,
    open_hold_remaining: torch.Tensor,
    close_command_armed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reset protection mask and updated per-episode close arming.

    A close request inherited from the previous episode is not new operator or
    policy intent. The gripper remains open for the configured reset settling
    interval and until an OPEN request has been observed in the new episode.
    """

    if actions.ndim != 2 or actions.shape[1] != 1:
        raise ValueError("binary gripper actions must have shape [N,1]")
    if open_hold_remaining.shape != (actions.shape[0],):
        raise ValueError("gripper open-hold counter must have shape [N]")
    if close_command_armed.shape != (actions.shape[0],):
        raise ValueError("gripper close-arm state must have shape [N]")
    requested_open = actions[:, 0] if actions.dtype == torch.bool else actions[:, 0] >= 0.0
    armed = close_command_armed | requested_open
    protected = (open_hold_remaining > 0) | (~armed)
    return protected, armed


def protected_reference_reset_settle_arm_hold_mask(
    reference_behavior_mask: torch.Tensor,
    protected_near_mask: torch.Tensor,
    gripper_open_hold_active: torch.Tensor,
) -> torch.Tensor:
    """Select protected reference rows that must wait for reset settling.

    A partial articulation reset restores the live-attested measured
    open-hand cache, but PhysX still needs the already configured gripper open-hold
    interval to settle its passive four-bar.  Starting differential IK during
    that interval couples the transient into the arm and makes an otherwise
    identical cached near reset depend on which vector row reset first.  This
    mask holds only the arm request for those protected reference rows; peer
    environments and every post-promotion policy row remain untouched.
    """

    expected_shape = reference_behavior_mask.shape
    for name, value in (
        ("reference_behavior_mask", reference_behavior_mask),
        ("protected_near_mask", protected_near_mask),
        ("gripper_open_hold_active", gripper_open_hold_active),
    ):
        if value.ndim != 1 or value.shape != expected_shape:
            raise ValueError(f"{name} must have the shared one-dimensional shape")
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    return reference_behavior_mask & protected_near_mask & gripper_open_hold_active


class G2ProtectedResetSettleEvidence:
    """Audit protected per-row gripper-settle/arm-hold intervals.

    The evidence object observes policy boundaries only.  It never changes an
    action, reset state, or safety threshold.  Runtime callers remain
    responsible for failing closed before applying an invalid action.
    """

    def __init__(
        self,
        num_envs: int,
        arm_joint_count: int,
        *,
        arm_action_dim: int,
        device,
    ) -> None:
        if num_envs <= 0 or arm_joint_count <= 0 or arm_action_dim <= 0:
            raise ValueError("settle evidence dimensions must be positive")
        self.num_envs = int(num_envs)
        self.arm_joint_count = int(arm_joint_count)
        self.arm_action_dim = int(arm_action_dim)
        self.device = torch.device(device)
        self.active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_duration_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.entry_count = torch.zeros_like(self.current_duration_steps)
        self.release_count = torch.zeros_like(self.current_duration_steps)
        self.completed_duration_count = torch.zeros_like(
            self.current_duration_steps
        )
        self.completed_duration_sum_steps = torch.zeros_like(
            self.current_duration_steps
        )
        self.completed_duration_min_steps = torch.full_like(
            self.current_duration_steps, torch.iinfo(torch.long).max
        )
        self.completed_duration_max_steps = torch.zeros_like(
            self.current_duration_steps
        )
        self.start_arm_q_rad = torch.zeros(
            (self.num_envs, self.arm_joint_count), device=self.device
        )
        self.start_cube_minus_ee_m = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.maximum_arm_q_component_drift_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self.maximum_cube_minus_ee_component_drift_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.applied_arm_attestation_rows = 0
        self.applied_arm_nonzero_rows = 0
        self.maximum_applied_arm_absolute = 0.0

    def _validate_state(
        self,
        hold_mask: torch.Tensor,
        arm_q_rad: torch.Tensor,
        cube_minus_ee_m: torch.Tensor,
    ) -> None:
        if hold_mask.shape != (self.num_envs,) or hold_mask.dtype != torch.bool:
            raise ValueError("settle hold mask must be boolean [N]")
        if arm_q_rad.shape != (self.num_envs, self.arm_joint_count):
            raise ValueError("settle arm state shape mismatch")
        if cube_minus_ee_m.shape != (self.num_envs, 3):
            raise ValueError("settle relative geometry shape mismatch")
        if (
            hold_mask.device != self.device
            or arm_q_rad.device != self.device
            or cube_minus_ee_m.device != self.device
        ):
            raise ValueError("settle evidence tensors must share its device")

    def observe_hold_state(
        self,
        *,
        hold_mask: torch.Tensor,
        arm_q_rad: torch.Tensor,
        cube_minus_ee_m: torch.Tensor,
        maximum_relative_component_drift_m: float,
    ) -> dict[str, torch.Tensor]:
        """Observe one pre-action boundary and return fail-closed row masks."""

        self._validate_state(hold_mask, arm_q_rad, cube_minus_ee_m)
        bound = float(maximum_relative_component_drift_m)
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("settle relative drift bound must be finite and positive")
        entering = hold_mask & ~self.active
        releasing = ~hold_mask & self.active
        self.start_arm_q_rad[entering] = arm_q_rad[entering]
        self.start_cube_minus_ee_m[entering] = cube_minus_ee_m[entering]
        self.entry_count += entering.to(torch.long)

        # Releasing rows are included so the last post-physics sample of an
        # interval cannot evade the drift attestation.
        measured_rows = self.active | entering
        finite_rows = torch.isfinite(arm_q_rad).all(dim=-1) & torch.isfinite(
            cube_minus_ee_m
        ).all(dim=-1)
        nonfinite_rows = measured_rows & ~finite_rows
        safe_arm_q = torch.where(
            finite_rows.unsqueeze(-1), arm_q_rad, self.start_arm_q_rad
        )
        safe_relative = torch.where(
            finite_rows.unsqueeze(-1),
            cube_minus_ee_m,
            self.start_cube_minus_ee_m,
        )
        arm_delta = torch.atan2(
            torch.sin(safe_arm_q - self.start_arm_q_rad),
            torch.cos(safe_arm_q - self.start_arm_q_rad),
        ).abs()
        arm_component_drift = arm_delta.amax(dim=-1)
        relative_component_drift = (
            safe_relative - self.start_cube_minus_ee_m
        ).abs().amax(dim=-1)
        self.maximum_arm_q_component_drift_rad = torch.where(
            measured_rows,
            torch.maximum(
                self.maximum_arm_q_component_drift_rad, arm_component_drift
            ),
            self.maximum_arm_q_component_drift_rad,
        )
        self.maximum_cube_minus_ee_component_drift_m = torch.where(
            measured_rows,
            torch.maximum(
                self.maximum_cube_minus_ee_component_drift_m,
                relative_component_drift,
            ),
            self.maximum_cube_minus_ee_component_drift_m,
        )
        drift_violation_rows = measured_rows & (
            relative_component_drift > bound
        )

        released_duration = self.current_duration_steps.clone()
        self.release_count += releasing.to(torch.long)
        self.completed_duration_count += releasing.to(torch.long)
        self.completed_duration_sum_steps += torch.where(
            releasing, released_duration, torch.zeros_like(released_duration)
        )
        self.completed_duration_min_steps = torch.where(
            releasing,
            torch.minimum(self.completed_duration_min_steps, released_duration),
            self.completed_duration_min_steps,
        )
        self.completed_duration_max_steps = torch.where(
            releasing,
            torch.maximum(self.completed_duration_max_steps, released_duration),
            self.completed_duration_max_steps,
        )
        self.current_duration_steps = torch.where(
            hold_mask,
            self.current_duration_steps + 1,
            torch.zeros_like(self.current_duration_steps),
        )
        self.active.copy_(hold_mask)
        return {
            "entering": entering,
            "releasing": releasing,
            "nonfinite_rows": nonfinite_rows,
            "drift_violation_rows": drift_violation_rows,
            "arm_component_drift_rad": arm_component_drift,
            "relative_component_drift_m": relative_component_drift,
        }

    def attest_applied_arm_zero(
        self, hold_mask: torch.Tensor, applied_arm: torch.Tensor
    ) -> torch.Tensor:
        """Return protected rows whose post-transform arm action is not zero."""

        if hold_mask.shape != (self.num_envs,) or hold_mask.dtype != torch.bool:
            raise ValueError("settle hold mask must be boolean [N]")
        if applied_arm.shape != (self.num_envs, self.arm_action_dim):
            raise ValueError("settle applied arm shape mismatch")
        if hold_mask.device != self.device or applied_arm.device != self.device:
            raise ValueError("settle applied tensors must share its device")
        finite = torch.isfinite(applied_arm).all(dim=-1)
        nonzero = (applied_arm != 0.0).any(dim=-1)
        violating = hold_mask & (~finite | nonzero)
        self.applied_arm_attestation_rows += int(hold_mask.sum())
        self.applied_arm_nonzero_rows += int(violating.sum())
        if bool(hold_mask.any()):
            selected = torch.nan_to_num(
                applied_arm[hold_mask].abs(), nan=float("inf"), posinf=float("inf")
            )
            self.maximum_applied_arm_absolute = max(
                self.maximum_applied_arm_absolute, float(selected.max())
            )
        return violating

    def serializable(self) -> dict[str, object]:
        rows = []
        for env_id in range(self.num_envs):
            completed = int(self.completed_duration_count[env_id])
            rows.append(
                {
                    "environment_id": env_id,
                    "entry_count": int(self.entry_count[env_id]),
                    "release_count": int(self.release_count[env_id]),
                    "completed_duration_count": completed,
                    "completed_duration_min_steps": (
                        int(self.completed_duration_min_steps[env_id])
                        if completed
                        else None
                    ),
                    "completed_duration_max_steps": (
                        int(self.completed_duration_max_steps[env_id])
                        if completed
                        else None
                    ),
                    "completed_duration_mean_steps": (
                        float(self.completed_duration_sum_steps[env_id]) / completed
                        if completed
                        else None
                    ),
                    "active": bool(self.active[env_id]),
                    "current_duration_steps": int(
                        self.current_duration_steps[env_id]
                    ),
                    "maximum_arm_q_component_drift_rad": float(
                        self.maximum_arm_q_component_drift_rad[env_id]
                    ),
                    "maximum_cube_minus_ee_component_drift_m": float(
                        self.maximum_cube_minus_ee_component_drift_m[env_id]
                    ),
                }
            )
        completed_total = int(self.completed_duration_count.sum())
        completed_minimum = self.completed_duration_min_steps[
            self.completed_duration_count > 0
        ]
        return {
            "schema": "g2_protected_reset_settle_evidence_v1",
            "entry_count": int(self.entry_count.sum()),
            "release_count": int(self.release_count.sum()),
            "completed_duration_count": completed_total,
            "completed_duration_min_steps": (
                int(completed_minimum.min()) if completed_total else None
            ),
            "completed_duration_max_steps": (
                int(self.completed_duration_max_steps.max())
                if completed_total
                else None
            ),
            "completed_duration_mean_steps": (
                float(self.completed_duration_sum_steps.sum()) / completed_total
                if completed_total
                else None
            ),
            "maximum_arm_q_component_drift_rad": float(
                self.maximum_arm_q_component_drift_rad.max()
            ),
            "maximum_cube_minus_ee_component_drift_m": float(
                self.maximum_cube_minus_ee_component_drift_m.max()
            ),
            "applied_arm_attestation_rows": self.applied_arm_attestation_rows,
            "applied_arm_nonzero_rows": self.applied_arm_nonzero_rows,
            "maximum_applied_arm_absolute": self.maximum_applied_arm_absolute,
            "applied_arm_exact_zero_attestation": (
                self.applied_arm_nonzero_rows == 0
            ),
            "rows": rows,
        }


class G2EpisodeDisposition(str, Enum):
    CONTINUE = "CONTINUE"
    SAVE = "SAVE"
    SAVE_FAILURE = "SAVE_FAILURE"
    DISCARD = "DISCARD"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class G2TeleopEpisodeControl:
    """State machine for the collection-only Enter/Backspace/Escape keys."""

    disposition: G2EpisodeDisposition = G2EpisodeDisposition.CONTINUE

    def handle_key(self, key_name: str) -> G2EpisodeDisposition:
        key = key_name.upper()
        if key in {"ENTER", "NUMPAD_ENTER"}:
            self.disposition = G2EpisodeDisposition.SAVE
        elif key == "F":
            self.disposition = G2EpisodeDisposition.SAVE_FAILURE
        elif key == "BACKSPACE":
            self.disposition = G2EpisodeDisposition.DISCARD
        elif key in {"ESC", "ESCAPE"}:
            self.disposition = G2EpisodeDisposition.EMERGENCY_STOP
        return self.disposition

    def reset(self) -> None:
        self.disposition = G2EpisodeDisposition.CONTINUE


MILESTONE6_TRANSITION_KEYS = frozenset({
    "observation_before_action", "next_observation", "teacher_observation",
    "student_observation", "actions", "previous_action", "achieved_goal",
    "desired_goal", "next_achieved_goal", "reward_components", "terminated",
    "truncated", "success", "contact_feature", "force_impulse_energy_feature",
    "curriculum_phase", "timestamp", "camera_timestamp", "camera_frame_age",
    "episode_id", "environment_seed", "task_preset", "g2_urdf_hash",
    "camera_extrinsic_hash", "action_schema_version", "observation_schema_version",
    "controller_mode",
    "episode_disposition", "contact_measurement_valid",
    "head_rgb", "head_depth", "head_depth_valid",
    "right_wrist_rgb", "right_wrist_depth", "right_wrist_depth_valid",
})


@dataclass(frozen=True)
class G2KeyboardTeleopContract:
    """Convert physical SE(3) keyboard commands to the G2 action contract."""

    translation_scale_m: float = G2_TRANSLATION_ACTION_SCALE_M
    rotation_scale_rad: float = G2_ROTATION_ACTION_SCALE_RAD
    normalized_limit: float = 1.0

    def validate(self) -> "G2KeyboardTeleopContract":
        if self.translation_scale_m <= 0.0 or self.rotation_scale_rad <= 0.0:
            raise ValueError("teleoperation scales must be positive")
        if self.normalized_limit <= 0.0:
            raise ValueError("normalized action limit must be positive")
        return self

    def normalize(self, physical_command: torch.Tensor) -> torch.Tensor:
        """Return normalized ``[dpos, drot, gripper]`` without changing order."""

        self.validate()
        if physical_command.shape[-1] != G2_TELEOP_ACTION_DIM:
            raise ValueError(
                f"keyboard command must end in {G2_TELEOP_ACTION_DIM} values; "
                f"got shape {tuple(physical_command.shape)}"
            )
        action = physical_command.clone()
        action[..., :3] /= self.translation_scale_m
        action[..., 3:6] /= self.rotation_scale_rad
        action[..., 6] = torch.where(
            action[..., 6] >= 0.0,
            torch.ones_like(action[..., 6]),
            -torch.ones_like(action[..., 6]),
        )
        return torch.clamp(action, -self.normalized_limit, self.normalized_limit)


@dataclass(frozen=True)
class G2RedundancyKeyboardTeleopContract:
    """Normalize ``SE(3) + elbow-nullspace + gripper`` teleoperation input.

    The eighth value does not command a physical joint directly. It is a
    signed redundancy request projected through the live 6-by-7 arm Jacobian.
    The existing seven-value policy/dataset contract remains unchanged.
    """

    translation_scale_m: float = G2_TRANSLATION_ACTION_SCALE_M
    rotation_scale_rad: float = G2_ROTATION_ACTION_SCALE_RAD
    normalized_limit: float = 1.0

    def validate(self) -> "G2RedundancyKeyboardTeleopContract":
        if self.translation_scale_m <= 0.0 or self.rotation_scale_rad <= 0.0:
            raise ValueError("teleoperation scales must be positive")
        if self.normalized_limit <= 0.0:
            raise ValueError("normalized action limit must be positive")
        return self

    def normalize(self, physical_command: torch.Tensor) -> torch.Tensor:
        """Return normalized ``[dpos, drot, elbow, gripper]`` commands."""

        self.validate()
        if physical_command.shape[-1] != G2_REDUNDANCY_TELEOP_ACTION_DIM:
            raise ValueError(
                "redundancy keyboard command must end in 8 values; "
                f"got shape {tuple(physical_command.shape)}"
            )
        action = physical_command.clone()
        action[..., :3] /= self.translation_scale_m
        action[..., 3:6] /= self.rotation_scale_rad
        action[..., 6] = torch.clamp(
            action[..., 6], -self.normalized_limit, self.normalized_limit
        )
        action[..., 7] = torch.where(
            action[..., 7] >= 0.0,
            torch.ones_like(action[..., 7]),
            -torch.ones_like(action[..., 7]),
        )
        return torch.clamp(action, -self.normalized_limit, self.normalized_limit)


def rate_limit_position_target(
    previous_target: torch.Tensor,
    desired_target: torch.Tensor,
    *,
    maximum_speed_rad_s: float,
    physics_dt_s: float,
) -> torch.Tensor:
    """Rate-limit an emitted joint-position target without changing hard limits.

    The limit applies to the command target derivative.  Measured velocity
    remains a separate safety authority because an authored USD drive may
    overshoot its target.
    """

    if previous_target.shape != desired_target.shape:
        raise ValueError("previous and desired joint targets must have identical shapes")
    if maximum_speed_rad_s <= 0.0 or physics_dt_s <= 0.0:
        raise ValueError("speed and physics dt must be positive")
    if not bool(torch.isfinite(previous_target).all() and torch.isfinite(desired_target).all()):
        raise ValueError("joint targets must be finite")
    if not previous_target.is_floating_point():
        raise ValueError("joint targets must use a floating-point dtype")

    # The target is ultimately authored in the asset tensor dtype (normally
    # float32). Adding an exact-looking Python ``speed * dt`` to a float32
    # joint angle can round the emitted delta one ULP away from the requested
    # hard limit. Reserve that arithmetic round-off at the producer so the
    # actual representable target stays below the configured limit; do not
    # compensate for it later with a relaxed safety comparison.
    maximum_delta = torch.as_tensor(
        maximum_speed_rad_s * physics_dt_s,
        dtype=previous_target.dtype,
        device=previous_target.device,
    )
    scale = torch.maximum(
        torch.maximum(previous_target.abs(), desired_target.abs()),
        torch.ones_like(previous_target),
    )
    roundoff_guard = 2.0 * torch.finfo(previous_target.dtype).eps * scale
    representable_maximum_delta = torch.clamp(
        maximum_delta - roundoff_guard, min=0.0
    )
    return previous_target + torch.clamp(
        desired_target - previous_target,
        -representable_maximum_delta,
        representable_maximum_delta,
    )


def synchronize_reset_position_target_state(
    previous_target: torch.Tensor,
    previous_target_velocity: torch.Tensor,
    target_initialized: torch.Tensor,
    measured_target: torch.Tensor,
    reset_mask: torch.Tensor,
) -> None:
    """Rebase a target generator on measured state for reset rows only.

    An Isaac Lab auto-reset can replace one articulation while peer
    environments continue their episodes.  The target and target-velocity
    history therefore cannot be cleared globally.  Replacing the completed
    rows with their newly measured joint positions makes the first target of
    the next episode continuous and gives it a zero command-velocity baseline.
    """

    if not (
        previous_target.shape
        == previous_target_velocity.shape
        == measured_target.shape
    ):
        raise ValueError("target, target velocity, and measurement shapes must match")
    if previous_target.ndim != 2:
        raise ValueError("reset target state must have shape (N, J)")
    if target_initialized.shape != (previous_target.shape[0],):
        raise ValueError("target initialized state must have shape (N,)")
    if reset_mask.shape != (previous_target.shape[0],) or reset_mask.dtype != torch.bool:
        raise ValueError("reset mask must be boolean with shape (N,)")
    if not (
        previous_target.is_floating_point()
        and previous_target_velocity.is_floating_point()
        and measured_target.is_floating_point()
    ):
        raise ValueError("reset target state must use floating-point tensors")
    if not bool(
        torch.isfinite(previous_target).all()
        and torch.isfinite(previous_target_velocity).all()
        and torch.isfinite(measured_target).all()
    ):
        raise ValueError("reset target state must be finite")
    if bool(reset_mask.any()):
        previous_target[reset_mask] = measured_target[reset_mask]
        previous_target_velocity[reset_mask] = 0.0
        target_initialized[reset_mask] = True


def synchronized_rate_limit_position_target(
    previous_target: torch.Tensor,
    desired_target: torch.Tensor,
    previous_target_velocity: torch.Tensor,
    *,
    maximum_speed_rad_s: float | torch.Tensor,
    maximum_acceleration_rad_s2: float,
    physics_dt_s: float,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rate-limit a coupled joint target with one scale per environment.

    Differential IK produces one *coupled* joint-space direction for a
    Cartesian request.  Clipping every joint independently changes that
    direction and can therefore make a pure ``-Z`` request move the tool in
    ``+Z``.  This helper applies a common speed scale and a common endpoint
    reference schedule to every joint in a row.  The dynamic velocity state
    approaches that reference through a common acceleration-bounded slew.  A
    minimal per-joint bounded deceleration guard is used only when a moving
    endpoint would otherwise be crossed; it never snaps position or velocity.

    The returned velocity is the state to use on the next physics step.  This
    function limits command targets only; measured finite-difference velocity
    remains the physical-motion safety authority.
    """

    if not (
        previous_target.shape
        == desired_target.shape
        == previous_target_velocity.shape
    ):
        raise ValueError("target and target-velocity tensors must have identical shapes")
    if previous_target.ndim != 2:
        raise ValueError("synchronized joint targets must have shape (N, J)")
    if maximum_acceleration_rad_s2 <= 0.0 or physics_dt_s <= 0.0:
        raise ValueError("acceleration and physics dt must be positive")
    if not previous_target.is_floating_point():
        raise ValueError("joint targets must use a floating-point dtype")
    if not bool(
        torch.isfinite(previous_target).all()
        and torch.isfinite(desired_target).all()
        and torch.isfinite(previous_target_velocity).all()
    ):
        raise ValueError("joint target state must be finite")

    speed_limit = torch.as_tensor(
        maximum_speed_rad_s,
        dtype=previous_target.dtype,
        device=previous_target.device,
    )
    if speed_limit.ndim == 0:
        speed_limit = speed_limit.expand(previous_target.shape[0], 1)
    elif speed_limit.shape == (previous_target.shape[0],):
        speed_limit = speed_limit.unsqueeze(-1)
    elif speed_limit.shape != (previous_target.shape[0], 1):
        raise ValueError("speed limit must be scalar, (N,), or (N, 1)")
    if not bool(torch.isfinite(speed_limit).all() and (speed_limit > 0.0).all()):
        raise ValueError("speed limits must be finite and positive")

    target_delta = desired_target - previous_target
    desired_velocity = target_delta / physics_dt_s
    peak_desired_speed = torch.amax(torch.abs(desired_velocity), dim=-1, keepdim=True)
    # Reserve float arithmetic round-off at the producer.  The reserve is
    # common to the row and therefore does not alter the coupled direction.
    speed_guard = 4.0 * torch.finfo(previous_target.dtype).eps * torch.maximum(
        speed_limit, torch.ones_like(speed_limit)
    )
    representable_speed_limit = torch.clamp(speed_limit - speed_guard, min=0.0)
    speed_scale = torch.clamp(
        representable_speed_limit / torch.clamp(peak_desired_speed, min=1.0e-12),
        max=1.0,
    )
    speed_limited_velocity = desired_velocity * speed_scale

    # Approach a moving IK endpoint through a discrete braking envelope before
    # updating the dynamic velocity state.  The former implementation first
    # applied the acceleration slew and then multiplied the resulting velocity
    # by a common endpoint-crossing ratio.  A joint only a few microradians
    # from its endpoint could therefore collapse every joint's accumulator in
    # one physics step, bypassing the acceleration bound.
    #
    # For constant acceleration ``a`` and step ``dt``, require
    #
    #   v * dt + v**2 / (2*a) <= remaining_distance
    #
    # which gives the positive safe reference speed below.  The minimum ratio
    # remains common to the row, preserving synchronized IK progress, but it
    # now modifies only the reference velocity.  The accumulator reaches that
    # reference exclusively through the existing acceleration-bounded slew.
    acceleration = torch.as_tensor(
        maximum_acceleration_rad_s2,
        dtype=previous_target.dtype,
        device=previous_target.device,
    )
    step_velocity = acceleration * physics_dt_s
    braking_speed = torch.sqrt(
        step_velocity.square() + 2.0 * acceleration * torch.abs(target_delta)
    ) - step_velocity
    moving_toward_endpoint = target_delta * speed_limited_velocity > 0.0
    per_joint_endpoint_scale = torch.where(
        moving_toward_endpoint & (torch.abs(speed_limited_velocity) > 1.0e-12),
        braking_speed
        / torch.clamp(torch.abs(speed_limited_velocity), min=1.0e-12),
        torch.ones_like(target_delta),
    )
    per_joint_endpoint_scale = torch.clamp(per_joint_endpoint_scale, max=1.0)
    endpoint_scale = torch.amin(per_joint_endpoint_scale, dim=-1, keepdim=True)
    endpoint_scheduled_velocity = speed_limited_velocity * endpoint_scale

    velocity_delta = endpoint_scheduled_velocity - previous_target_velocity
    peak_velocity_delta = torch.amax(torch.abs(velocity_delta), dim=-1, keepdim=True)
    maximum_velocity_delta = torch.as_tensor(
        maximum_acceleration_rad_s2 * physics_dt_s,
        dtype=previous_target.dtype,
        device=previous_target.device,
    )
    acceleration_scale = torch.clamp(
        maximum_velocity_delta / torch.clamp(peak_velocity_delta, min=1.0e-12),
        max=1.0,
    )
    target_velocity = previous_target_velocity + velocity_delta * acceleration_scale

    # A moving IK endpoint can shift by a few microradians between physics
    # steps.  If the common slew would cross that updated endpoint, decelerate
    # only the affected joint toward the already scheduled reference, still
    # bounded by the same acceleration delta.  This is not an endpoint snap:
    # the accumulator remains continuous and every component retains the
    # configured acceleration bound.
    proposed_delta = target_velocity * physics_dt_s
    crossing_after_common_slew = (
        target_delta * proposed_delta > 0.0
    ) & (torch.abs(proposed_delta) > torch.abs(target_delta))
    per_joint_velocity_delta = torch.clamp(
        velocity_delta,
        min=-maximum_velocity_delta,
        max=maximum_velocity_delta,
    )
    per_joint_decelerated_velocity = (
        previous_target_velocity + per_joint_velocity_delta
    )
    target_velocity = torch.where(
        crossing_after_common_slew,
        per_joint_decelerated_velocity,
        target_velocity,
    )
    final_proposed_delta = target_velocity * physics_dt_s
    unavoidable_endpoint_overshoot = (
        target_delta * final_proposed_delta > 0.0
    ) & (torch.abs(final_proposed_delta) > torch.abs(target_delta))

    # At this point every row already satisfies the configured target-speed
    # and target-acceleration limits.  With a moving DLS endpoint it can be
    # physically impossible, in one 2 ms step, to satisfy those limits *and*
    # stop exactly at the newly recomputed endpoint.  Raising here turns a
    # bounded deceleration into a controller crash even though the next target
    # is safe to emit.  Keep the acceleration-bounded trajectory continuous;
    # callers retain their joint-safe envelope checks and telemetry records
    # this exceptional, unavoidable micro-overshoot explicitly.
    #
    # This is not endpoint clipping or a target snap.  ``limited_target`` is
    # still the integral of the bounded velocity state.  The following DLS
    # update will bring it back to the Cartesian target under the same limits.

    limited_target = previous_target + target_velocity * physics_dt_s
    if diagnostics is not None:
        diagnostics.update(
            {
                "speed_scale": speed_scale.detach().clone(),
                "per_joint_endpoint_scale": per_joint_endpoint_scale.detach().clone(),
                "common_endpoint_scale": endpoint_scale.detach().clone(),
                "acceleration_scale": acceleration_scale.detach().clone(),
                "per_joint_endpoint_deceleration_mask": (
                    crossing_after_common_slew.detach().clone()
                ),
                "braking_speed_rad_s": braking_speed.detach().clone(),
                "endpoint_scheduled_velocity_rad_s": (
                    endpoint_scheduled_velocity.detach().clone()
                ),
                "unavoidable_endpoint_overshoot_mask": (
                    unavoidable_endpoint_overshoot.detach().clone()
                ),
            }
        )
    return limited_target, target_velocity


def clamp_outstanding_cartesian_position_target(
    measured_position: torch.Tensor,
    desired_position: torch.Tensor,
    *,
    maximum_axis_error_m: float,
) -> torch.Tensor:
    """Bound queued Cartesian motion without changing the requested axis sign.

    Keyboard packets are endpoint increments.  A fast key repeat must not
    queue an unreachable endpoint hundreds of millimetres beyond the live
    gripper.  The bound is component-wise so an ``E`` packet remains a pure
    root-frame ``-Z`` request; it never creates an X/Y command.
    """

    if measured_position.shape != desired_position.shape:
        raise ValueError("measured and desired Cartesian positions must match")
    if measured_position.shape[-1] != 3:
        raise ValueError("Cartesian position must end in XYZ")
    if maximum_axis_error_m <= 0.0:
        raise ValueError("maximum_axis_error_m must be positive")
    if not bool(
        torch.isfinite(measured_position).all()
        and torch.isfinite(desired_position).all()
    ):
        raise ValueError("Cartesian positions must be finite")
    return measured_position + torch.clamp(
        desired_position - measured_position,
        -maximum_axis_error_m,
        maximum_axis_error_m,
    )


def axis_isolated_translation_command_origin(
    measured_position: torch.Tensor,
    previous_desired_position: torch.Tensor,
    previous_target_valid: torch.Tensor,
    previous_axis: torch.Tensor,
    translation_command: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose an endpoint origin that never carries motion across key axes.

    A neutral heartbeat retains the previous endpoint.  A repeated command on
    the same single axis accumulates only that axis.  Switching axes rebases
    all coordinates on the measured gripper pose, cancelling unfinished motion
    from the old axis.  Multi-axis policy commands are supported but are not
    labelled as a keyboard axis (``-2``).
    """

    if measured_position.shape != previous_desired_position.shape:
        raise ValueError("measured and previous desired position shapes must match")
    if measured_position.ndim != 2 or measured_position.shape[-1] != 3:
        raise ValueError("Cartesian positions must have shape (N, 3)")
    if translation_command.shape != measured_position.shape:
        raise ValueError("translation command must have shape (N, 3)")
    if previous_target_valid.shape != (measured_position.shape[0],):
        raise ValueError("previous_target_valid must have shape (N,)")
    if previous_axis.shape != previous_target_valid.shape:
        raise ValueError("previous_axis must have shape (N,)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    requested = torch.abs(translation_command) > epsilon
    requested_count = requested.sum(dim=-1)
    single_axis = requested_count == 1
    any_translation = requested_count > 0
    requested_axis = torch.argmax(requested.to(torch.int64), dim=-1)

    origin = torch.where(
        previous_target_valid.unsqueeze(-1),
        previous_desired_position,
        measured_position,
    ).clone()
    # Every explicit translation starts with all uncommanded coordinates at
    # the current measurement.  Only a same-axis repeat retains that axis's
    # previous endpoint for deliberate accumulation.
    origin[any_translation] = measured_position[any_translation]
    same_axis_repeat = (
        single_axis
        & previous_target_valid
        & (previous_axis == requested_axis)
    )
    rows = torch.nonzero(same_axis_repeat, as_tuple=False).flatten()
    if rows.numel() > 0:
        axes = requested_axis[rows]
        origin[rows, axes] = previous_desired_position[rows, axes]

    next_axis = previous_axis.clone()
    next_axis[single_axis] = requested_axis[single_axis]
    next_axis[any_translation & ~single_axis] = -2
    return origin, next_axis


def validate_milestone6_episode(
    episode: Any,
    *,
    expected_steps: int,
    observation_dim: int,
) -> dict[str, Any]:
    """Validate the public Isaac Lab HDF5 episode contract after loading."""

    data = episode.data
    required = {"initial_state", "states", "actions", "obs", "processed_actions"}
    missing = sorted(required.difference(data))
    actions = data.get("actions")
    observations = data.get("obs")
    processed = data.get("processed_actions")
    actions_shape = list(actions.shape) if isinstance(actions, torch.Tensor) else None
    observations_shape = (
        list(observations.shape) if isinstance(observations, torch.Tensor) else None
    )
    processed_shape = (
        list(processed.shape) if isinstance(processed, torch.Tensor) else None
    )
    finite = bool(
        isinstance(actions, torch.Tensor)
        and isinstance(observations, torch.Tensor)
        and isinstance(processed, torch.Tensor)
        and torch.isfinite(actions).all()
        and torch.isfinite(observations).all()
        and torch.isfinite(processed).all()
    )
    passed = bool(
        not missing
        and actions_shape == [expected_steps, G2_TELEOP_ACTION_DIM]
        and observations_shape == [expected_steps, observation_dim]
        and processed_shape == [expected_steps, G2_TELEOP_ACTION_DIM]
        and finite
    )
    return {
        "pass": passed,
        "required_keys": sorted(required),
        "missing_keys": missing,
        "actions_shape": actions_shape,
        "observations_shape": observations_shape,
        "processed_actions_shape": processed_shape,
        "all_finite": finite,
    }


def validate_full_demonstration_episode(episode: Any, *, expected_steps: int) -> dict[str, Any]:
    """Validate the complete G2 teacher/student transition schema."""

    data = episode.data
    missing = sorted(MILESTONE6_TRANSITION_KEYS.difference(data))
    lengths: dict[str, int | None] = {}
    finite = True
    raw_depth_keys = {"head_depth", "right_wrist_depth"}
    for key in sorted(MILESTONE6_TRANSITION_KEYS.intersection(data)):
        value = data[key]
        if not isinstance(value, torch.Tensor):
            lengths[key] = None
            finite = False
            continue
        lengths[key] = int(value.shape[0]) if value.ndim else None
        if key not in raw_depth_keys:
            finite = finite and bool(torch.isfinite(value).all())
    depth_validity: dict[str, bool] = {}
    for prefix in ("head", "right_wrist"):
        depth = data.get(f"{prefix}_depth")
        valid = data.get(f"{prefix}_depth_valid")
        pair_pass = bool(
            isinstance(depth, torch.Tensor)
            and isinstance(valid, torch.Tensor)
            and depth.shape == valid.shape
            and bool(valid.to(torch.bool).any())
            and bool(torch.isfinite(depth[valid.to(torch.bool)]).all())
            and bool((depth[valid.to(torch.bool)] > 0.0).all())
        )
        depth_validity[prefix] = pair_pass
        finite = finite and pair_pass
    wrong_lengths = sorted(key for key, length in lengths.items() if length != expected_steps)
    return {
        "pass": not missing and not wrong_lengths and finite,
        "required_keys": sorted(MILESTONE6_TRANSITION_KEYS),
        "missing_keys": missing,
        "wrong_length_keys": wrong_lengths,
        "lengths": lengths,
        "all_finite": finite,
        "depth_valid_mask_contract": depth_validity,
    }


__all__ = [
    "G2_ARM_ACTION_DIM",
    "G2_ARM_ACTION_SCALE",
    "G2ProtectedResetSettleEvidence",
    "G2_KEYBOARD_ROTATION_STEP_RAD",
    "G2_TRANSLATION_ACTION_SCALE_M",
    "G2_ROTATION_ACTION_SCALE_RAD",
    "contact_latched_position_target",
    "protected_reference_reset_settle_arm_hold_mask",
    "reset_safe_gripper_open_mask",
    "G2_REDUNDANCY_TELEOP_ACTION_DIM",
    "G2_TELEOP_ACTION_DIM",
    "G2KeyboardTeleopContract",
    "G2RedundancyKeyboardTeleopContract",
    "G2EpisodeDisposition",
    "G2TeleopEpisodeControl",
    "MILESTONE6_TRANSITION_KEYS",
    "clamp_outstanding_cartesian_position_target",
    "axis_isolated_translation_command_origin",
    "rate_limit_position_target",
    "synchronize_reset_position_target_state",
    "synchronized_rate_limit_position_target",
    "validate_milestone6_episode",
    "validate_full_demonstration_episode",
]
