# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Lab action/device adapters for G2 7-DoF keyboard teleoperation."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import inspect
import math
from pathlib import Path

import torch

from isaaclab.devices import Se3Keyboard
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
# Import the decorator from its defining module.  Importing the module through
# another Isaac Lab path can otherwise bind ``isaaclab.utils.configclass`` as a
# module attribute before this file is evaluated, which makes the package-level
# import below non-callable during AppLauncher startup.
from isaaclab.utils.configclass import configclass
from isaaclab.utils import math as math_utils

from .g2_gripper_reset_contract import (
    G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
    PassiveLimitMasterGovernor,
    acceleration_limited_position_target_step,
)
from .g2_lift_methodology import RIGHT_ARM_JOINTS
from .g2_redundancy_teleop import (
    G2ElbowKeyState,
    damped_nullspace_project,
    ee_rotation_only_mask,
    tensor_value,
    wrist_only_rotation_jacobian,
)
from .g2_reference_recovery import exact_emitted_endpoint_mask
from .g2_quaternion import (
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
)
from .g2_teleop_dataset import (
    G2RedundancyKeyboardTeleopContract,
    axis_isolated_translation_command_origin,
    clamp_outstanding_cartesian_position_target,
    contact_latched_position_target,
    reset_safe_gripper_open_mask,
    synchronize_reset_position_target_state,
    synchronized_rate_limit_position_target,
)
from .g2_policy_branch.dls_read_only_preview_api import (
    G2DLSReadOnlyPreviewError,
    G2DLSReadOnlySnapshot,
    source_sha256,
)
from .g2_policy_branch.dls_read_only_preview_capture import (
    G2DLSReadOnlyPreviewCaptureQueue,
    G2DLSReadOnlyPreviewCaptureBundle,
    G2DLSReadOnlyPreviewCaptureReceipt,
    G2DLSReadOnlyPreviewCaptureRequest,
)


G2_DIFFERENTIAL_IK_TARGET_RATE_LIMIT_MODE = "SYNCHRONIZED_COMMON_ROW_SCALE"


class _G2PositionTargetRateLimit:
    """Shared per-physics-step target rate limiter for G2 action terms."""

    def _initialize_target_rate_limit(self, width: int, env) -> None:
        speed = float(self.cfg.maximum_joint_target_speed_rad_s)
        if speed <= 0.0:
            raise ValueError("maximum_joint_target_speed_rad_s must be positive")
        acceleration = float(self.cfg.maximum_joint_target_acceleration_rad_s2)
        if acceleration <= 0.0:
            raise ValueError(
                "maximum_joint_target_acceleration_rad_s2 must be positive"
            )
        self._g2_physics_dt_s = float(env.physics_dt)
        self._g2_previous_target = torch.zeros(
            (self.num_envs, width), device=self.device
        )
        self._g2_target_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._g2_previous_target_velocity = torch.zeros(
            (self.num_envs, width), device=self.device
        )
        self._g2_environment_speed_limit_rad_s = torch.full(
            (self.num_envs, 1), speed, device=self.device
        )
        self._g2_maximum_emitted_target_speed_rad_s = torch.zeros(
            (), device=self.device
        )
        self._g2_maximum_emitted_target_acceleration_rad_s2 = torch.zeros(
            (), device=self.device
        )
        self._g2_last_synchronized_speed_scale = torch.ones(
            (self.num_envs, 1), device=self.device
        )
        self._g2_last_synchronized_endpoint_scale = torch.ones(
            (self.num_envs, 1), device=self.device
        )
        self._g2_last_synchronized_per_joint_endpoint_scale = torch.ones(
            (self.num_envs, width), device=self.device
        )
        self._g2_last_synchronized_acceleration_scale = torch.ones(
            (self.num_envs, 1), device=self.device
        )
        self._g2_last_synchronized_endpoint_deceleration_mask = torch.zeros(
            (self.num_envs, width), dtype=torch.bool, device=self.device
        )
        self._g2_last_synchronized_unavoidable_overshoot_mask = torch.zeros(
            (self.num_envs, width), dtype=torch.bool, device=self.device
        )

    def _rate_limited_target(
        self,
        desired: torch.Tensor,
        measured: torch.Tensor,
        *,
        maximum_speed_override_rad_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uninitialized = ~self._g2_target_initialized
        if bool(uninitialized.any()):
            self._g2_previous_target[uninitialized] = measured[uninitialized]
            self._g2_previous_target_velocity[uninitialized] = 0.0
            self._g2_target_initialized[uninitialized] = True
        speed_limit = self._g2_environment_speed_limit_rad_s
        if maximum_speed_override_rad_s is not None:
            if maximum_speed_override_rad_s.shape != (
                self.num_envs,
                1,
            ):
                raise ValueError("maximum speed override shape mismatch")
            speed_limit = torch.minimum(
                speed_limit, maximum_speed_override_rad_s
            )
        limited, target_velocity = acceleration_limited_position_target_step(
            self._g2_previous_target,
            desired,
            self._g2_previous_target_velocity,
            maximum_speed_rad_s=speed_limit,
            maximum_acceleration_rad_s2=float(
                self.cfg.maximum_joint_target_acceleration_rad_s2
            ),
            dt_s=self._g2_physics_dt_s,
        )
        emitted_speed = torch.max(
            torch.abs(limited - self._g2_previous_target)
        ) / self._g2_physics_dt_s
        emitted_acceleration = torch.max(
            torch.abs(target_velocity - self._g2_previous_target_velocity)
        ) / self._g2_physics_dt_s
        self._g2_maximum_emitted_target_speed_rad_s = torch.maximum(
            self._g2_maximum_emitted_target_speed_rad_s, emitted_speed
        )
        self._g2_maximum_emitted_target_acceleration_rad_s2 = torch.maximum(
            self._g2_maximum_emitted_target_acceleration_rad_s2,
            emitted_acceleration,
        )
        self._g2_previous_target.copy_(limited)
        self._g2_previous_target_velocity.copy_(target_velocity)
        return limited

    def _synchronized_rate_limited_target(
        self, desired: torch.Tensor, measured: torch.Tensor
    ) -> torch.Tensor:
        """Limit an IK target without destroying its coupled joint direction."""

        uninitialized = ~self._g2_target_initialized
        if bool(uninitialized.any()):
            self._g2_previous_target[uninitialized] = measured[uninitialized]
            self._g2_previous_target_velocity[uninitialized] = 0.0
            self._g2_target_initialized[uninitialized] = True
        previous_velocity = self._g2_previous_target_velocity.clone()
        diagnostics: dict[str, torch.Tensor] = {}
        limited, target_velocity = synchronized_rate_limit_position_target(
            self._g2_previous_target,
            desired,
            self._g2_previous_target_velocity,
            maximum_speed_rad_s=self._g2_environment_speed_limit_rad_s,
            maximum_acceleration_rad_s2=float(
                self.cfg.maximum_joint_target_acceleration_rad_s2
            ),
            physics_dt_s=self._g2_physics_dt_s,
            diagnostics=diagnostics,
        )
        self._g2_last_synchronized_speed_scale.copy_(diagnostics["speed_scale"])
        self._g2_last_synchronized_endpoint_scale.copy_(
            diagnostics["common_endpoint_scale"]
        )
        self._g2_last_synchronized_per_joint_endpoint_scale.copy_(
            diagnostics["per_joint_endpoint_scale"]
        )
        self._g2_last_synchronized_acceleration_scale.copy_(
            diagnostics["acceleration_scale"]
        )
        self._g2_last_synchronized_endpoint_deceleration_mask.copy_(
            diagnostics["per_joint_endpoint_deceleration_mask"]
        )
        self._g2_last_synchronized_unavoidable_overshoot_mask.copy_(
            diagnostics["unavoidable_endpoint_overshoot_mask"]
        )
        emitted_speed = torch.amax(
            torch.abs(limited - self._g2_previous_target)
        ) / self._g2_physics_dt_s
        emitted_acceleration = torch.amax(
            torch.abs(target_velocity - previous_velocity)
        ) / self._g2_physics_dt_s
        self._g2_maximum_emitted_target_speed_rad_s = torch.maximum(
            self._g2_maximum_emitted_target_speed_rad_s, emitted_speed
        )
        self._g2_maximum_emitted_target_acceleration_rad_s2 = torch.maximum(
            self._g2_maximum_emitted_target_acceleration_rad_s2,
            emitted_acceleration,
        )
        self._g2_previous_target.copy_(limited)
        self._g2_previous_target_velocity.copy_(target_velocity)
        return limited

    def _reset_target_rate_limit(self, env_ids) -> None:
        self._g2_target_initialized[env_ids] = False
        self._g2_previous_target_velocity[env_ids] = 0.0
        self._g2_environment_speed_limit_rad_s[env_ids] = float(
            self.cfg.maximum_joint_target_speed_rad_s
        )

    def set_environment_speed_limit_rad_s(self, limit: torch.Tensor) -> None:
        """Set a per-environment joint-target speed cap.

        Values may only reduce the configured maximum; this method cannot
        raise actuator authority above the audited command limit.
        """

        if limit.shape != (self.num_envs,):
            raise ValueError("environment speed limit shape mismatch")
        limit = limit.to(self.device, dtype=self._g2_previous_target.dtype)
        if bool((limit <= 0.0).any()):
            raise ValueError("environment speed limits must be positive")
        self._g2_environment_speed_limit_rad_s[:, 0].copy_(
            torch.clamp(
                limit,
                max=float(self.cfg.maximum_joint_target_speed_rad_s),
            )
        )

    def synchronize_target_to_measured(self, env_ids: torch.Tensor) -> None:
        """Start a zero-velocity hold from the live measured joint state.

        A zero Cartesian action does not by itself cancel velocity retained by
        the acceleration-limited target generator.  Reset initialization uses
        this explicit hand-off only after the task-space endpoint has remained
        in tolerance.  It does not teleport articulation state or relax any
        velocity/acceleration limit.
        """

        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        measured = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        reset_mask[env_ids] = True
        synchronize_reset_position_target_state(
            self._g2_previous_target,
            self._g2_previous_target_velocity,
            self._g2_target_initialized,
            measured,
            reset_mask,
        )

    @property
    def current_target_velocity_rad_s(self) -> torch.Tensor:
        """Live velocity state of the acceleration-limited target generator."""

        return self._g2_previous_target_velocity

    @property
    def last_emitted_joint_position_target(self) -> torch.Tensor:
        """Final post-controller, post-rate-limit actuator target in radians."""

        return self._g2_previous_target

    @property
    def maximum_emitted_target_acceleration_rad_s2(self) -> torch.Tensor:
        """Peak commanded target acceleration; distinct from measured FD acceleration."""

        return self._g2_maximum_emitted_target_acceleration_rad_s2

    @property
    def last_synchronized_endpoint_scale(self) -> torch.Tensor:
        """Common braking-reference scale from the latest physics step."""

        return self._g2_last_synchronized_endpoint_scale

    @property
    def last_synchronized_per_joint_endpoint_scale(self) -> torch.Tensor:
        """Per-joint candidates used to form the common braking scale."""

        return self._g2_last_synchronized_per_joint_endpoint_scale

    @property
    def last_synchronized_speed_scale(self) -> torch.Tensor:
        """Common maximum-speed scale from the latest physics step."""

        return self._g2_last_synchronized_speed_scale

    @property
    def last_synchronized_acceleration_scale(self) -> torch.Tensor:
        """Common velocity-slew scale from the latest physics step."""

        return self._g2_last_synchronized_acceleration_scale

    @property
    def last_synchronized_endpoint_deceleration_mask(self) -> torch.Tensor:
        """Joints receiving bounded endpoint-specific deceleration."""

        return self._g2_last_synchronized_endpoint_deceleration_mask

    @property
    def last_synchronized_unavoidable_overshoot_mask(self) -> torch.Tensor:
        """Joints that cannot stop at a moving IK endpoint in one substep.

        The emitted target remains speed/acceleration bounded.  This is
        telemetry only; it is not an authorization to cross joint limits.
        """

        return self._g2_last_synchronized_unavoidable_overshoot_mask


class G2RateLimitedDifferentialIKAction(
    _G2PositionTargetRateLimit, DifferentialInverseKinematicsAction
):
    """Six-dimensional differential IK with a configurable joint-target speed."""

    cfg: "G2RateLimitedDifferentialIKActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._initialize_target_rate_limit(len(self._joint_ids), env)

    @property
    def target_rate_limit_mode(self) -> str:
        """Attest how the seven-joint differential-IK direction is limited."""

        return G2_DIFFERENTIAL_IK_TARGET_RATE_LIMIT_MODE

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        if bool(torch.linalg.vector_norm(ee_quat_curr, dim=-1).min() > 0.0):
            jacobian = self._compute_frame_jacobian()
            desired = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            desired = joint_pos.clone()
        # Differential IK can emit a setpoint beyond the articulation limit.
        # Rate limiting that target alone does not prevent the measured joint
        # from striking the hard stop (observed at joint7 == pi/2).  Clamp to
        # the audited soft limits with the same internal margin used by the
        # redundancy action; do not alter the URDF/USD limits themselves.
        soft_limits = tensor_value(self._asset.data.soft_joint_pos_limits)[
            :, self._joint_ids
        ]
        hard_limits = tensor_value(self._asset.data.joint_pos_limits)[
            :, self._joint_ids
        ]
        margin = float(self.cfg.joint_limit_margin_rad)
        # The physics hard limit is the final authority.  Keep the learning
        # soft region only where it is stricter.  In a 25-env run joint7 was
        # measured at its -1.571 rad hard stop while the emitted target was
        # -1.597 rad; the resulting constraint stop changed measured velocity
        # from -0.206 to 0 in one 20 ms control interval.  Intersecting both
        # sources fixes the target, without modifying either asset limit.
        limits = torch.stack(
            (
                torch.maximum(soft_limits[..., 0], hard_limits[..., 0]),
                torch.minimum(soft_limits[..., 1], hard_limits[..., 1]),
            ),
            dim=-1,
        )
        safe_lower = limits[..., 0] + margin
        safe_upper = limits[..., 1] - margin
        if bool((safe_lower >= safe_upper).any()):
            raise RuntimeError("G2_ARM_JOINT_LIMIT_MARGIN_EMPTY")
        # A stale internal target from an earlier IK request must not remain
        # beyond a hard stop while the desired target has already been
        # clamped.  Fresh episodes normally make this a no-op.
        bounded_previous = torch.clamp(
            self._g2_previous_target, safe_lower, safe_upper
        )
        previous_was_clamped = bounded_previous != self._g2_previous_target
        self._g2_previous_target.copy_(bounded_previous)
        self._g2_previous_target_velocity.masked_fill_(previous_was_clamped, 0.0)
        desired = torch.clamp(
            desired, safe_lower, safe_upper
        )
        # A differential-IK target is one coupled seven-joint direction.
        # Component-wise target clipping lets the largest joints saturate
        # independently and rotates that direction (observed as a requested
        # root-frame -Z move producing +Z EE motion).  Use one common row
        # scale, as the redundancy action already does, so speed and
        # acceleration bounds preserve the DLS solution.
        desired = self._synchronized_rate_limited_target(desired, joint_pos)
        desired = torch.clamp(desired, safe_lower, safe_upper)
        if bool(((desired < safe_lower) | (desired > safe_upper)).any()):
            raise RuntimeError("G2_ARM_EMITTED_TARGET_OUTSIDE_SAFE_LIMIT")
        self._g2_previous_target.copy_(desired)
        self._asset.set_joint_position_target_index(
            target=desired, joint_ids=self._joint_ids
        )

    def synchronize_reset_to_measured(self, env_ids: torch.Tensor) -> None:
        """Rebase every arm-controller cache after a partial auto-reset.

        ``ActionManager.reset`` clears the action term, but the reset event has
        already teleported the completed articulation by then.  The runtime
        calls this method after ``env.step`` returns so the IK endpoint, joint
        target, and target velocity all describe that newly measured pose.
        Peer rows are intentionally untouched.
        """

        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        ee_position, ee_orientation = self._compute_frame_pose()
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._ik_controller.ee_pos_des[env_ids] = ee_position[env_ids]
        self._ik_controller.ee_quat_des[env_ids] = ee_orientation[env_ids]
        self.synchronize_target_to_measured(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._reset_target_rate_limit(env_ids)


@configclass
class G2RateLimitedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = G2RateLimitedDifferentialIKAction
    maximum_joint_target_speed_rad_s: float = 0.6
    joint_limit_margin_rad: float = 0.02
    # A 25-env live attribution measured 13.046 rad/s^2 at joint6 from a
    # 5.0 rad/s^2 emitted-target limit without contact produced 13.046 rad/s^2
    # measured acceleration.  A later 25-env run still produced 11.899 at a
    # 3.0 target limit.  Retiming to 2.0 preserves the measured-motion hard
    # gate at 10 rad/s^2 and adds margin for close-range drive reversal.
    maximum_joint_target_acceleration_rad_s2: float = 2.0


class G2RateLimitedBinaryJointPositionAction(
    _G2PositionTargetRateLimit, BinaryJointPositionAction
):
    """Binary gripper goal with a bounded emitted master-joint target speed."""

    cfg: "G2RateLimitedBinaryJointPositionActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if int(self.cfg.minimum_open_policy_steps_after_reset) < 0:
            raise ValueError("minimum_open_policy_steps_after_reset must be nonnegative")
        # Isaac Lab initializes BinaryJointAction._processed_actions to zero,
        # which is the G2 close target.  Seed it explicitly to OPEN so the
        # first apply and every reset cannot inherit a closed command from a
        # previous episode before the policy's next action is processed.
        self._raw_actions.fill_(1.0)
        self._processed_actions.copy_(
            self._open_command.unsqueeze(0).expand_as(self._processed_actions)
        )
        self._initialize_target_rate_limit(self._num_joints, env)
        self._last_rate_limited_target = self._processed_actions.clone()
        self._g2_external_hold_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Keep the last rate-limited actuator target when an external contact
        # controller takes authority.  Holding the measured position removes
        # the drive preload on every physics step and can release a valid
        # bilateral grasp; continuing toward the binary closed endpoint can
        # instead over-compress the object.  A latched emitted target preserves
        # the bounded preload without either behavior.
        self._g2_external_hold_target = self._last_rate_limited_target.clone()
        self._g2_open_hold_physics_steps = int(
            math.ceil(
                float(self.cfg.minimum_open_policy_steps_after_reset)
                * float(env.step_dt)
                / float(env.physics_dt)
            )
        )
        self._g2_open_hold_remaining = torch.full(
            (self.num_envs,),
            self._g2_open_hold_physics_steps,
            dtype=torch.long,
            device=self.device,
        )
        self._g2_close_command_armed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        passive_names = tuple(G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES)
        missing_passive = set(passive_names).difference(self._asset.joint_names)
        if missing_passive:
            raise ValueError(
                "G2_PASSIVE_LIMIT_GUARD_JOINT_MISSING:"
                + ",".join(sorted(missing_passive))
            )
        self._g2_passive_joint_ids = [
            self._asset.joint_names.index(name) for name in passive_names
        ]
        self._g2_passive_previous_position = torch.zeros(
            (self.num_envs, len(self._g2_passive_joint_ids)),
            dtype=self._g2_previous_target.dtype,
            device=self.device,
        )
        self._g2_passive_position_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._g2_passive_limit_governor = PassiveLimitMasterGovernor(
            num_envs=self.num_envs,
            passive_joint_count=len(self._g2_passive_joint_ids),
            device=self.device,
            dtype=self._g2_previous_target.dtype,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Reject inherited close state until this episode has requested OPEN."""

        super().process_actions(actions)
        protected, armed = reset_safe_gripper_open_mask(
            actions,
            self._g2_open_hold_remaining,
            self._g2_close_command_armed,
        )
        self._g2_close_command_armed.copy_(armed)
        self._processed_actions[protected] = self._open_command

    def set_external_hold_mask(self, mask: torch.Tensor) -> None:
        """Latch the emitted gripper target after bilateral object contact.

        The binary close intent remains in the replay action.  This actuator
        safety layer prevents the position drive from continuing to compress
        an already contacted rigid object while retaining the contact preload.
        """

        if mask.shape != (self.num_envs,):
            raise ValueError("gripper external hold mask shape mismatch")
        resolved = mask.to(self.device, dtype=torch.bool)
        newly_held = resolved & ~self._g2_external_hold_mask
        if bool(newly_held.any()):
            self._g2_external_hold_target[newly_held] = (
                self._last_rate_limited_target[newly_held]
            )
        self._g2_external_hold_mask.copy_(resolved)

    @property
    def exact_close_target_emitted(self) -> torch.Tensor:
        """Rows whose rate-limited target exactly equals configured CLOSE.

        This is emitted-command telemetry, not a claim about measured finger
        position.  Exact equality is intentional: recovery timing must not
        introduce a new position tolerance outside the actuator contract.
        """

        return exact_emitted_endpoint_mask(
            self._last_rate_limited_target, self._close_command
        )

    @property
    def close_command_active(self) -> torch.Tensor:
        """Rows whose accepted binary command requests CLOSE.

        This reports command intent rather than the mechanical close endpoint.
        A correctly grasped rigid object prevents the master joint from
        reaching the empty-hand endpoint, and the contact hold layer freezes
        its last bounded target to avoid over-compression. Reward code can
        therefore require an intentional closing grasp without rewarding an
        empty fully-closed gripper.
        """

        return exact_emitted_endpoint_mask(
            self._processed_actions, self._close_command
        )

    @property
    def exact_open_target_emitted(self) -> torch.Tensor:
        """Rows whose rate-limited target exactly equals configured OPEN."""

        return exact_emitted_endpoint_mask(
            self._last_rate_limited_target, self._open_command
        )

    @property
    def reset_open_hold_active(self) -> torch.Tensor:
        """Per-environment reset settling state for the passive four-bar.

        This exposes the existing open-hold clock without introducing another
        duration or threshold.  The deterministic reference runtime uses it
        to avoid starting arm IK while a freshly reset OmniPicker is still
        resolving its passive constraint state.
        """

        return self._g2_open_hold_remaining > 0

    def apply_actions(self) -> None:
        measured = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        desired = contact_latched_position_target(
            self._processed_actions,
            self._g2_external_hold_target,
            self._g2_external_hold_mask,
        )
        full_position = tensor_value(self._asset.data.joint_pos)
        passive_position = full_position[:, self._g2_passive_joint_ids]
        passive_uninitialized = ~self._g2_passive_position_initialized
        if bool(passive_uninitialized.any()):
            self._g2_passive_previous_position[passive_uninitialized] = (
                passive_position[passive_uninitialized]
            )
            self._g2_passive_position_initialized[passive_uninitialized] = True
        passive_velocity = torch.atan2(
            torch.sin(passive_position - self._g2_passive_previous_position),
            torch.cos(passive_position - self._g2_passive_previous_position),
        ) / self._g2_physics_dt_s
        passive_velocity[passive_uninitialized] = 0.0
        all_limits = tensor_value(self._asset.data.joint_pos_limits)
        passive_limits = (
            all_limits[:, self._g2_passive_joint_ids]
            if all_limits.ndim == 3
            else all_limits[self._g2_passive_joint_ids]
        )
        target_initialized = self._g2_target_initialized.unsqueeze(-1)
        current_target = torch.where(
            target_initialized, self._g2_previous_target, measured
        )
        current_target_velocity = torch.where(
            target_initialized,
            self._g2_previous_target_velocity,
            torch.zeros_like(self._g2_previous_target_velocity),
        )
        passive_guard_cap = self._g2_passive_limit_governor.compute_speed_cap(
            passive_position,
            passive_velocity,
            passive_limits,
            current_target_velocity,
            desired - current_target,
            maximum_master_speed_rad_s=float(
                self.cfg.maximum_joint_target_speed_rad_s
            ),
            maximum_master_target_acceleration_rad_s2=(
                float(self.cfg.maximum_joint_target_acceleration_rad_s2)
            ),
            dt_s=self._g2_physics_dt_s,
        )
        target = self._rate_limited_target(
            desired,
            measured,
            maximum_speed_override_rad_s=passive_guard_cap,
        )
        self._g2_passive_previous_position.copy_(passive_position)
        self._last_rate_limited_target.copy_(target)
        self._asset.set_joint_position_target_index(
            target=target, joint_ids=self._joint_ids
        )
        self._g2_open_hold_remaining.sub_(1).clamp_(min=0)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Isaac Lab's binary base action uses ``tensor[env_ids]`` directly.
        # A Python tuple is interpreted by PyTorch as a multi-axis index
        # (e.g. ``(0, 1, ..., 24)``), rather than as a batch-row selection.
        # The vector runtime deliberately resets all gripper rows after a
        # direct source restore, so normalize public Sequence input to the
        # one-dimensional tensor index expected by the base action before it
        # clears its buffers.  This changes only Python-side action caches;
        # it does not write articulation/follower joints or alter mechanics.
        if env_ids is not None and not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super().reset(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        self._raw_actions[selection] = 1.0
        self._processed_actions[selection] = self._open_command
        self._last_rate_limited_target[selection] = self._open_command
        self._reset_target_rate_limit(env_ids)
        self._g2_external_hold_mask[selection] = False
        self._g2_external_hold_target[selection] = self._open_command
        self._g2_open_hold_remaining[selection] = self._g2_open_hold_physics_steps
        self._g2_close_command_armed[selection] = False
        self._g2_passive_position_initialized[selection] = False
        self._g2_passive_limit_governor.reset(env_ids)


@configclass
class G2RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type = G2RateLimitedBinaryJointPositionAction
    maximum_joint_target_speed_rad_s: float = 0.8
    maximum_joint_target_acceleration_rad_s2: float = 10.0
    minimum_open_policy_steps_after_reset: int = 60


class G2RedundancyDifferentialIKAction(
    _G2PositionTargetRateLimit, DifferentialInverseKinematicsAction
):
    """Differential IK plus a bounded Jacobian-nullspace elbow request."""

    cfg: "G2RedundancyDifferentialIKActionCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._initialize_target_rate_limit(len(self._joint_ids), env)
        if len(self._joint_ids) != 7:
            raise ValueError("G2 wrist-only rotation requires seven ordered arm joints")
        if not 1 <= int(self.cfg.rotation_only_wrist_start_joint_index) < 7:
            raise ValueError(
                "rotation_only_wrist_start_joint_index must be in [1, 6]"
            )
        # Pure roll/pitch/yaw teleoperation holds the EE position captured at
        # the beginning of the rotation sequence.  Without this persistent
        # anchor, each relative command uses the slightly drifted measured
        # position as its next origin and converts IK tracking error into
        # cumulative XYZ motion.
        self._g2_rotation_position_lock = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._g2_rotation_position_lock_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Relative Cartesian key presses are pose increments, not one-frame
        # velocity commands.  The joint target limiter generally needs many
        # physics steps to realize one increment.  Retain the IK endpoint on
        # neutral heartbeat packets; otherwise every heartbeat re-bases the
        # desired pose on a partially tracked measurement and silently
        # cancels the outstanding X/Y/Z or R/P/Y request.
        self._g2_pose_target_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._g2_active_translation_axis = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._g2_effective_elbow_command = torch.zeros(
            self.num_envs, device=self.device
        )
        self._g2_rotation_only_request = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Disabled by default.  This queue is diagnostic-only and cannot
        # create a second action path or command authority.
        self._g2_read_only_dls_preview_capture = (
            G2DLSReadOnlyPreviewCaptureQueue()
        )

    @property
    def action_dim(self) -> int:
        return 7

    def set_read_only_dls_preview_capture_enabled(self, enabled: bool) -> None:
        """Enable/disable only the bounded diagnostic receipt interface."""

        self._g2_read_only_dls_preview_capture.set_enabled(enabled)

    def stage_read_only_dls_preview_capture(
        self, request: G2DLSReadOnlyPreviewCaptureRequest
    ) -> None:
        """Stage one immutable full-8D contact-free packet before env.step."""

        self._g2_read_only_dls_preview_capture.stage(request)

    def take_read_only_dls_preview_capture(
        self, *, control_epoch: int
    ) -> G2DLSReadOnlyPreviewCaptureReceipt:
        """Take one immutable receipt; this is never a command result."""

        return self._g2_read_only_dls_preview_capture.take_receipt(
            control_epoch=control_epoch
        )

    def take_read_only_dls_preview_capture_bundle(
        self, *, control_epoch: int
    ) -> G2DLSReadOnlyPreviewCaptureBundle:
        """Take pre-apply plus following-normal-target read-only evidence."""

        return self._g2_read_only_dls_preview_capture.take_bundle(
            control_epoch=control_epoch
        )

    def capture_read_only_dls_preview_snapshot(
        self,
        *,
        control_epoch: int,
        metric_action_4d_root_m: torch.Tensor,
        full_action_packet_8d: torch.Tensor,
        environment_index: int = 0,
    ) -> G2DLSReadOnlySnapshot:
        """Clone one contact-free action state for an offline DLS preview.

        This is deliberately a *query*, not an alternate action path.  It
        must be invoked by a caller that has already routed the same packet
        through the canonical ActionManager processing phase and before the
        normal application phase.  The method never processes an action,
        sets an IK command, advances a limiter, invokes an articulation
        writer, or grants command authority.

        The external/persisted pose convention is ``robot_root + XYZW``.
        Isaac Lab's native controller quaternion order is copied separately
        and converted through the central G2 quaternion boundary, so a future
        runtime-version change cannot silently swap quaternion components.
        """

        if type(control_epoch) is not int or control_epoch < 0:
            raise G2DLSReadOnlyPreviewError("control_epoch must be non-negative int")
        if not 0 <= environment_index < self.num_envs:
            raise G2DLSReadOnlyPreviewError("environment index is out of range")
        expected_metric_shape = (self.num_envs, 4)
        expected_packet_shape = (self.num_envs, 8)
        if metric_action_4d_root_m.shape != expected_metric_shape:
            raise G2DLSReadOnlyPreviewError(
                f"metric 4-D action must have shape {expected_metric_shape}"
            )
        if full_action_packet_8d.shape != expected_packet_shape:
            raise G2DLSReadOnlyPreviewError(
                f"full 8-D packet must have shape {expected_packet_shape}"
            )
        if (
            metric_action_4d_root_m.dtype != torch.float32
            or full_action_packet_8d.dtype != torch.float32
            or metric_action_4d_root_m.device != self._raw_actions.device
            or full_action_packet_8d.device != self._raw_actions.device
        ):
            raise G2DLSReadOnlyPreviewError(
                "preview packet tensors must preserve the action-term float32/device identity"
            )
        if tuple(self._joint_names) != RIGHT_ARM_JOINTS:
            raise G2DLSReadOnlyPreviewError("runtime right-arm joint order is not canonical")
        row = environment_index
        packet_arm = full_action_packet_8d[:, :7]
        if not torch.equal(packet_arm, self._raw_actions):
            raise G2DLSReadOnlyPreviewError(
                "full packet arm7 does not equal the current canonical processed raw action"
            )
        expected_processed = self._raw_actions * self._scale
        if not torch.equal(expected_processed, self._processed_actions):
            raise G2DLSReadOnlyPreviewError(
                "action-term processed arm state is not exactly raw action times source scale"
            )
        if (
            bool(self._g2_rotation_only_request[row])
            or float(self._g2_effective_elbow_command[row]) != 0.0
        ):
            raise G2DLSReadOnlyPreviewError(
                "contact-free 4-D preview rejects nonzero orientation/elbow runtime state"
            )
        controller = self.cfg.controller
        if getattr(controller, "ik_method", None) != "dls":
            raise G2DLSReadOnlyPreviewError("preview only supports source DLS controller")
        params = getattr(controller, "ik_params", None)
        if not isinstance(params, dict) or "lambda_val" not in params:
            raise G2DLSReadOnlyPreviewError("source DLS lambda is unavailable")

        # ``_compute_frame_pose`` is read-only.  The Jacobian helper is not
        # used because it may apply a body-offset transform in-place to a
        # cached source tensor; reproduce that exact transform on a clone.
        ee_position_native, ee_quaternion_native = self._compute_frame_pose()
        ee_position_native = ee_position_native.detach().clone()
        ee_quaternion_native = ee_quaternion_native.detach().clone()
        jacobian = self.jacobian_b.detach().clone()
        if self.cfg.body_offset is not None:
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos),
                jacobian[:, 3:, :],
            )
            jacobian[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :]
            )
        native_order = isaaclab_native_quaternion_order()
        canonical_current = quaternion_native_to_xyzw(
            ee_quaternion_native, native_order
        ).detach().clone()
        desired_position_native = self._ik_controller.ee_pos_des.detach().clone()
        desired_quaternion_native = self._ik_controller.ee_quat_des.detach().clone()
        canonical_desired = quaternion_native_to_xyzw(
            desired_quaternion_native, native_order
        ).detach().clone()
        joint_position = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids].detach().clone()
        joint_velocity = tensor_value(self._asset.data.joint_vel)[:, self._joint_ids].detach().clone()
        joint_acceleration = tensor_value(self._asset.data.joint_acc)[:, self._joint_ids].detach().clone()
        soft_limits = tensor_value(self._asset.data.soft_joint_pos_limits)[:, self._joint_ids].detach().clone()
        runtime_tensors = (
            ee_position_native,
            ee_quaternion_native,
            jacobian,
            desired_position_native,
            desired_quaternion_native,
            joint_position,
            joint_velocity,
            joint_acceleration,
            soft_limits,
            self._g2_previous_target,
            self._g2_previous_target_velocity,
            self._g2_environment_speed_limit_rad_s,
        )
        if any(value.dtype != torch.float32 or value.device != self._raw_actions.device for value in runtime_tensors):
            raise G2DLSReadOnlyPreviewError("runtime preview source tensors do not share float32/device identity")

        def _vector(value: torch.Tensor) -> tuple[float, ...]:
            return tuple(float(item) for item in value[row].detach().cpu().tolist())

        action_source = Path(__file__).resolve()
        dls_source = Path(inspect.getsourcefile(type(self._ik_controller)) or "").resolve()
        return G2DLSReadOnlySnapshot(
            control_epoch=control_epoch,
            environment_index=row,
            metric_action_4d_root_m=tuple(
                float(item) for item in metric_action_4d_root_m[row].detach().cpu().tolist()
            ),
            full_action_packet_8d=tuple(
                float(item) for item in full_action_packet_8d[row].detach().cpu().tolist()
            ),
            packet_hash=hashlib.sha256(
                ",".join(format(float(item), ".9g") for item in full_action_packet_8d[row].detach().cpu().tolist()).encode("ascii")
            ).hexdigest(),
            packet_hash_schema="full8_packet_only_sha256_v1",
            ordered_joint_names=tuple(self._joint_names),
            frame="robot_root",
            quaternion_order="xyzw",
            native_dls_quaternion_order=native_order,
            native_to_canonical_quaternion_converter="g2_quaternion.quaternion_native_to_xyzw",
            translation_m_per_normalized=float(self._scale[row, 0]),
            runtime_dtype="float32",
            runtime_device=str(self._raw_actions.device),
            measured_ee_position_root_m=_vector(ee_position_native),
            measured_ee_quaternion_root_xyzw=_vector(canonical_current),
            desired_ee_position_root_m=_vector(desired_position_native),
            desired_ee_quaternion_root_xyzw=_vector(canonical_desired),
            measured_ee_quaternion_native=_vector(ee_quaternion_native),
            desired_ee_quaternion_native=_vector(desired_quaternion_native),
            measured_joint_position_rad=_vector(joint_position),
            measured_joint_velocity_rad_s=_vector(joint_velocity),
            measured_joint_acceleration_rad_s2=_vector(joint_acceleration),
            frame_jacobian_root=tuple(_vector(jacobian[:, index, :]) for index in range(6)),
            soft_joint_limits_rad=tuple(
                tuple(float(item) for item in soft_limits[row, index].detach().cpu().tolist())
                for index in range(7)
            ),
            limiter_previous_target_rad=_vector(self._g2_previous_target),
            limiter_previous_velocity_rad_s=_vector(self._g2_previous_target_velocity),
            limiter_target_initialized=bool(self._g2_target_initialized[row]),
            limiter_speed_rad_s=float(self._g2_environment_speed_limit_rad_s[row, 0]),
            limiter_acceleration_rad_s2=float(self.cfg.maximum_joint_target_acceleration_rad_s2),
            physics_dt_s=float(self._g2_physics_dt_s),
            policy_dt_s=float(self._g2_physics_dt_s * 10),
            dls_lambda=float(params["lambda_val"]),
            nullspace_seed_joint_index=int(self.cfg.redundancy_seed_joint_index),
            nullspace_damping=float(self.cfg.nullspace_damping),
            maximum_nullspace_joint_delta_rad_per_physics_step=float(
                self.cfg.maximum_nullspace_joint_delta_rad_per_physics_step
            ),
            joint_limit_margin_rad=float(self.cfg.joint_limit_margin_rad),
            effective_elbow_command=float(self._g2_effective_elbow_command[row]),
            rotation_only_request=bool(self._g2_rotation_only_request[row]),
            source_hashes=(
                ("g2_redundancy_action_source", source_sha256(action_source)),
                ("g2_redundancy_teleop_source", source_sha256(inspect.getsourcefile(damped_nullspace_project) or "")),
                ("g2_teleop_dataset_source", source_sha256(inspect.getsourcefile(synchronized_rate_limit_position_target) or "")),
                ("differential_ik_source", source_sha256(dls_source)),
            ),
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 7):
            raise ValueError(f"G2 redundancy arm action must be (N, 7), got {tuple(actions.shape)}")
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        cartesian = self._processed_actions[:, :6]
        translation_requested = torch.any(
            torch.abs(cartesian[:, :3]) > 1.0e-8, dim=-1
        )
        elbow_requested = torch.abs(self._processed_actions[:, 6]) > 1.0e-8
        rotation_only = ee_rotation_only_mask(cartesian) & ~elbow_requested
        cartesian_requested = translation_requested | rotation_only
        previous_pose_target_valid = self._g2_pose_target_valid.clone()

        # An explicit XYZ request begins a new positional command and releases
        # the old rotation anchor.  Neutral packets intentionally keep a live
        # anchor so the controller corrects residual XYZ drift between key
        # presses instead of accepting it as the next origin.
        self._g2_rotation_position_lock_valid[
            translation_requested | elbow_requested
        ] = False
        capture = rotation_only & ~self._g2_rotation_position_lock_valid
        self._g2_rotation_position_lock[capture] = ee_pos_curr[capture]
        self._g2_rotation_position_lock_valid[capture] = True
        # New deltas accumulate from the last requested endpoint.  A neutral
        # packet is a zero delta about that same endpoint, so the controller
        # continues converging instead of accepting tracking lag as the next
        # origin.  Elbow null-space requests intentionally start from the
        # measured Cartesian pose because they are not Cartesian increments.
        command_origin_position, next_translation_axis = (
            axis_isolated_translation_command_origin(
                ee_pos_curr,
                self._ik_controller.ee_pos_des,
                previous_pose_target_valid,
                self._g2_active_translation_axis,
                cartesian[:, :3],
            )
        )
        command_origin_position[elbow_requested] = ee_pos_curr[elbow_requested]
        # Switching from translation to pure R/P/Y ends the outstanding
        # translation at the measured pose.  The wrist-only solver then owns
        # only orientation and cannot unexpectedly continue an old XYZ goal.
        command_origin_position = torch.where(
            rotation_only.unsqueeze(-1),
            self._g2_rotation_position_lock,
            command_origin_position,
        )
        keep_orientation_target = previous_pose_target_valid & ~(
            translation_requested | elbow_requested
        )
        command_origin_orientation = torch.where(
            keep_orientation_target.unsqueeze(-1),
            self._ik_controller.ee_quat_des,
            ee_quat_curr,
        )

        # Do not combine a pure EE orientation request with an elbow
        # null-space request.  The latter is a separate 7-DoF control and can
        # introduce small task-space leakage even though its projection is
        # bounded.
        # Keep wrist-only authority active on neutral heartbeat packets.  An
        # XYZ or explicit elbow request intentionally releases it.
        wrist_only_authority = self._g2_rotation_position_lock_valid & ~(
            translation_requested | elbow_requested
        )
        self._g2_effective_elbow_command.copy_(self._processed_actions[:, 6])
        self._g2_effective_elbow_command[wrist_only_authority] = 0.0
        self._g2_rotation_only_request.copy_(wrist_only_authority)
        self._ik_controller.set_command(
            cartesian, command_origin_position, command_origin_orientation
        )
        # Key-repeat may arrive faster than the physical arm can settle.  Keep
        # the outstanding endpoint local to the measured gripper so dozens of
        # repeated E/W/etc. packets cannot queue an unreachable target.  This
        # is an axis-preserving queue bound, not a workspace/tolerance bypass.
        self._ik_controller.ee_pos_des.copy_(
            clamp_outstanding_cartesian_position_target(
                ee_pos_curr,
                self._ik_controller.ee_pos_des,
                maximum_axis_error_m=float(
                    self.cfg.maximum_outstanding_translation_axis_error_m
                ),
            )
        )
        self._g2_pose_target_valid[cartesian_requested] = True
        self._g2_pose_target_valid[elbow_requested] = False
        self._g2_active_translation_axis[translation_requested] = (
            next_translation_axis[translation_requested]
        )
        self._g2_active_translation_axis[rotation_only | elbow_requested] = -1
        # Additive capture point: canonical arm processing is complete above,
        # while ActionManager.apply_action() has not yet entered its normal
        # physics-rate controller/writer phase.  The queue verifies the exact
        # staged full8 arm slice and captures clone-only state; it invokes no
        # action processing, writer, limiter, or controller operation itself.
        self._g2_read_only_dls_preview_capture.capture_after_normal_process(
            processed_arm7=self._raw_actions,
            snapshot_factory=lambda request: self.capture_read_only_dls_preview_snapshot(
                control_epoch=request.control_epoch,
                metric_action_4d_root_m=request.metric_tensor(
                    device=self._raw_actions.device
                ),
                full_action_packet_8d=request.full8_tensor(
                    device=self._raw_actions.device
                ),
                environment_index=request.environment_index,
            ),
        )

    def cancel_pose_target_to_measured(
        self, env_ids: torch.Tensor | Sequence[int]
    ) -> None:
        """Cancel outstanding Cartesian motion at the measured EE pose.

        This is used only for explicit operator control-plane requests such as
        ``L`` (clear queued motion) and viewport changes.  Ordinary neutral
        transport heartbeats deliberately do *not* call it.
        """

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if env_ids.numel() == 0:
            return
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.ee_pos_des[env_ids] = ee_pos_curr[env_ids]
        self._ik_controller.ee_quat_des[env_ids] = ee_quat_curr[env_ids]
        self._g2_pose_target_valid[env_ids] = False
        self._g2_active_translation_axis[env_ids] = -1
        self._g2_rotation_position_lock[env_ids] = 0.0
        self._g2_rotation_position_lock_valid[env_ids] = False
        self._g2_effective_elbow_command[env_ids] = 0.0
        self._g2_rotation_only_request[env_ids] = False
        self.synchronize_target_to_measured(env_ids)

    def synchronize_reset_to_measured(self, env_ids: torch.Tensor) -> None:
        """Rebase every redundancy-controller cache after a row reset.

        ``ManagerBasedRLEnv.step`` may auto-reset only a subset of vector
        rows.  The articulation then contains the new measured state while
        this term can still contain the previous episode's raw packet, DLS
        endpoint, persistent-axis state, and acceleration-limited joint
        target.  Reusing that state would apply an unrequested command to the
        reset row.  This method mirrors the established rate-limited IK reset
        hand-off, but also clears the redundancy-specific temporal state via
        the existing measured-pose cancellation path.  Peer rows are left
        untouched and no physics step or articulation write occurs here.
        """

        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.cancel_pose_target_to_measured(env_ids)

    def apply_actions(self) -> None:
        # A staged diagnostic request must have been captured during canonical
        # process_actions before this unchanged normal controller/writer path.
        self._g2_read_only_dls_preview_capture.assert_no_pending_before_apply()
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = tensor_value(self._asset.data.joint_pos)[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()
        if bool(torch.linalg.vector_norm(ee_quat_curr, dim=-1).min() > 0.0):
            desired = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            if bool(self._g2_rotation_only_request.any()):
                wrist_jacobian = wrist_only_rotation_jacobian(
                    jacobian,
                    wrist_start_joint_index=(
                        self.cfg.rotation_only_wrist_start_joint_index
                    ),
                )
                wrist_desired = self._ik_controller.compute(
                    ee_pos_curr,
                    ee_quat_curr,
                    wrist_jacobian,
                    joint_pos,
                )
                wrist_start = int(self.cfg.rotation_only_wrist_start_joint_index)
                wrist_desired[:, :wrist_start] = joint_pos[:, :wrist_start]
                desired = torch.where(
                    self._g2_rotation_only_request.unsqueeze(-1),
                    wrist_desired,
                    desired,
                )
        else:
            desired = joint_pos.clone()

        projection = damped_nullspace_project(
            jacobian,
            self._g2_effective_elbow_command,
            seed_joint_index=self.cfg.redundancy_seed_joint_index,
            damping=self.cfg.nullspace_damping,
            maximum_joint_delta_rad=self.cfg.maximum_nullspace_joint_delta_rad_per_physics_step,
        )
        desired = desired + projection.joint_delta
        limits = tensor_value(self._asset.data.soft_joint_pos_limits)[:, self._joint_ids]
        margin = self.cfg.joint_limit_margin_rad
        desired = torch.clamp(desired, limits[..., 0] + margin, limits[..., 1] - margin)
        if bool(self._g2_rotation_only_request.any()):
            wrist_start = int(self.cfg.rotation_only_wrist_start_joint_index)
            rows = self._g2_rotation_only_request
            # Matching proximal targets to measured q cancels any outstanding
            # seven-joint target without introducing a drive-position jump.
            self._g2_previous_target[rows, :wrist_start] = joint_pos[
                rows, :wrist_start
            ]
            self._g2_previous_target_velocity[rows, :wrist_start] = 0.0
        # One common row scale preserves the 7-joint DLS solution.  Independent
        # clipping made joints 1/4/7 saturate at the same delta in live E-key
        # traces and turned a pure root-frame -Z request into +X/+Z motion.
        desired = self._synchronized_rate_limited_target(desired, joint_pos)
        self._last_nullspace_joint_delta = projection.joint_delta
        self._last_nullspace_task_leakage = projection.task_leakage
        self._last_nullspace_clipped = projection.clipped
        # Clone-only same-epoch target evidence.  This reads the normal
        # limiter output immediately before the unchanged articulation writer;
        # it does not write a target or alter controller/limiter state.
        self._g2_read_only_dls_preview_capture.record_following_normal_controller_target(
            emitted_target_rad=desired,
            joint_position_rad=joint_pos,
            joint_velocity_rad_s=tensor_value(self._asset.data.joint_vel)[:, self._joint_ids],
            joint_acceleration_rad_s2=tensor_value(self._asset.data.joint_acc)[:, self._joint_ids],
            ordered_joint_names=tuple(self._joint_names),
        )

        if hasattr(self._asset, "set_joint_position_target_index"):
            self._asset.set_joint_position_target_index(target=desired, joint_ids=self._joint_ids)
        else:
            self._asset.set_joint_position_target(desired, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._g2_rotation_position_lock[env_ids] = 0.0
        self._g2_rotation_position_lock_valid[env_ids] = False
        self._g2_pose_target_valid[env_ids] = False
        self._g2_active_translation_axis[env_ids] = -1
        self._g2_effective_elbow_command[env_ids] = 0.0
        self._g2_rotation_only_request[env_ids] = False
        self._reset_target_rate_limit(env_ids)

    @property
    def ee_rotation_position_lock_active(self) -> torch.Tensor:
        """Per-environment attestation for rotation-only EE position hold."""

        return self._g2_rotation_position_lock_valid

    @property
    def ee_rotation_only_request(self) -> torch.Tensor:
        """Rows whose latest command changes EE orientation only."""

        return self._g2_rotation_only_request

    @property
    def ee_pose_target_active(self) -> torch.Tensor:
        """Rows retaining an outstanding Cartesian endpoint across heartbeats."""

        return self._g2_pose_target_valid

    @property
    def active_translation_axis(self) -> torch.Tensor:
        """Root-frame XYZ axis retained by the current keyboard endpoint."""

        return self._g2_active_translation_axis

    @property
    def ee_desired_position(self) -> torch.Tensor:
        """Persistent gripper-center position target in robot-root axes."""

        return self._ik_controller.ee_pos_des

    @property
    def ee_desired_orientation(self) -> torch.Tensor:
        """Persistent gripper-center orientation target as root-frame WXYZ."""

        return self._ik_controller.ee_quat_des


@configclass
class G2RedundancyDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = G2RedundancyDifferentialIKAction
    redundancy_seed_joint_index: int = 2
    nullspace_damping: float = 0.05
    maximum_nullspace_joint_delta_rad_per_physics_step: float = 0.0002
    # idx61..idx64 are held for pure RPY; idx65..idx67 retain authority.
    rotation_only_wrist_start_joint_index: int = 4
    joint_limit_margin_rad: float = 0.02
    maximum_joint_target_speed_rad_s: float = 0.6
    maximum_joint_target_acceleration_rad_s2: float = 2.0
    # At the default 15 mm translation key step this permits two outstanding
    # increments while preventing unbounded key-repeat accumulation.
    maximum_outstanding_translation_axis_error_m: float = 0.03


class G2RedundancySe3Keyboard(Se3Keyboard):
    """Official SE(3) keyboard with bracket-key elbow swivel."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._g2_elbow_state = G2ElbowKeyState()
        self._g2_normalization = G2RedundancyKeyboardTeleopContract()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_g2_elbow_state"):
            self._g2_elbow_state.reset()

    @property
    def input_device_name(self) -> str:
        """Return the live Omniverse keyboard name for GUI attestation."""

        return str(self._input.get_keyboard_name(self._keyboard))

    def advance_physical(self) -> torch.Tensor:
        """Return physical SE(3), normalized elbow request, and gripper."""

        original = super().advance()
        elbow = torch.tensor(
            [self._g2_elbow_state.command], dtype=original.dtype, device=original.device
        )
        return torch.cat((original[:6], elbow, original[6:7]))

    def advance(self) -> torch.Tensor:
        """Return the normalized 8-D action expected by the environment."""

        return self.sample()[1]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the keyboard once and return physical and normalized commands.

        A recorder must not call :meth:`advance_physical` and :meth:`advance`
        separately: that would read the device twice and could pair camera data
        with a different key state.  This method is the single sampling point.
        """

        physical = self.advance_physical()
        return physical, self._g2_normalization.normalize(physical)

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb

        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        if pressed or released:
            self._g2_elbow_state.handle(event.input.name, pressed=pressed)
        return super()._on_keyboard_event(event, *args, **kwargs)


__all__ = [
    "G2_DIFFERENTIAL_IK_TARGET_RATE_LIMIT_MODE",
    "G2RateLimitedBinaryJointPositionAction",
    "G2RateLimitedBinaryJointPositionActionCfg",
    "G2RateLimitedDifferentialIKAction",
    "G2RateLimitedDifferentialIKActionCfg",
    "G2RedundancyDifferentialIKAction",
    "G2RedundancyDifferentialIKActionCfg",
    "G2RedundancySe3Keyboard",
]
