# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""G2 exhibition lift shaping and recovery terms.

The functions in this module keep hard safety separate from reward shaping.
The first contact is never a failure: pushing is ignored for a short grace
period and later receives a bounded recovery potential.  Only a cube that has
left the tabletop for a persistent interval is terminated here.
"""

from __future__ import annotations

import math

import torch

from .g2_collision_authority import G2ForbiddenCollisionEvaluator
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

from .g2_lift_methodology import (
    EXHIBITION_HEAD_Q,
    FIXED_TORSO_Q,
    G2LiftSandboxContract,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_EE_BODY,
    RIGHT_GRIPPER_MASTER,
    TORSO_JOINTS,
)
from .g2_redundancy_teleop import tensor_value
from .g2_quaternion import (
    isaaclab_native_quaternion_order,
    quaternion_xyzw_to_native,
)


TASK = G2LiftSandboxContract().validated()
DEFAULT_SAC_REWARD_GAMMA = 0.9993


def _ensure_state(env) -> None:
    if hasattr(env, "_g2_task_initial_cube_w"):
        return
    device = env.device
    env._g2_task_initial_cube_w = torch.zeros((env.num_envs, 3), device=device)
    env._g2_task_previous_ee_distance = torch.full((env.num_envs,), torch.nan, device=device)
    env._g2_task_previous_goal_distance = torch.full((env.num_envs,), torch.nan, device=device)
    env._g2_task_previous_push_distance = torch.zeros(env.num_envs, device=device)
    env._g2_task_previous_ee_phase = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=device
    )
    env._g2_task_previous_goal_phase = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=device
    )
    env._g2_task_previous_push_phase = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=device
    )
    env._g2_task_ever_bilateral_contact = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_ever_closing_finger_contact = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_stable_time_s = torch.zeros(env.num_envs, device=device)
    env._g2_task_ever_stable_grasp = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_ever_lifted_while_stable = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_partial_credit_emitted = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_touch_milestone_emitted = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_contact_milestone_emitted = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_stable_milestone_emitted = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_lift_milestone_emitted = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    env._g2_task_outside_table_time_s = torch.zeros(env.num_envs, device=device)
    env._g2_task_first_touch_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=device
    )
    names = (*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER)
    env._g2_task_controlled_joint_ids = torch.tensor(
        [env.scene["robot"].joint_names.index(name) for name in names],
        dtype=torch.long,
        device=device,
    )
    env._g2_task_previous_joint_velocity = torch.zeros(
        (env.num_envs, len(names)), device=device
    )


def reset_task_progress_state(env, env_ids: torch.Tensor) -> None:
    """Reset all history after the object and robot reset events."""

    _ensure_state(env)
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    env._g2_task_initial_cube_w[env_ids] = cube[env_ids]
    env._g2_task_previous_ee_distance[env_ids] = torch.nan
    env._g2_task_previous_goal_distance[env_ids] = torch.nan
    env._g2_task_previous_push_distance[env_ids] = 0.0
    env._g2_task_previous_ee_phase[env_ids] = -1
    env._g2_task_previous_goal_phase[env_ids] = -1
    env._g2_task_previous_push_phase[env_ids] = -1
    env._g2_task_ever_bilateral_contact[env_ids] = False
    env._g2_task_ever_closing_finger_contact[env_ids] = False
    env._g2_task_stable_time_s[env_ids] = 0.0
    env._g2_task_ever_stable_grasp[env_ids] = False
    env._g2_task_ever_lifted_while_stable[env_ids] = False
    env._g2_task_partial_credit_emitted[env_ids] = False
    env._g2_task_touch_milestone_emitted[env_ids] = False
    env._g2_task_contact_milestone_emitted[env_ids] = False
    env._g2_task_stable_milestone_emitted[env_ids] = False
    env._g2_task_lift_milestone_emitted[env_ids] = False
    env._g2_task_outside_table_time_s[env_ids] = 0.0
    env._g2_task_first_touch_step[env_ids] = -1
    robot = env.scene["robot"]
    qd = tensor_value(robot.data.joint_vel).index_select(
        1, env._g2_task_controlled_joint_ids
    )
    env._g2_task_previous_joint_velocity[env_ids] = qd[env_ids]


def reset_fixed_torso_and_head_targets(env, env_ids: torch.Tensor) -> None:
    """Restore held-joint targets after the generic asset reset.

    The torso, head and left arm remain outside this task's action space.
    Isaac Lab's generic reset restores their measured positions but, by
    default, does not restore drive targets.  Retaining the old targets made
    the non-controlled left arm accelerate toward the USD defaults during the
    gripper-open settle.  Bind only these independent held joints to the
    configured reset position and zero velocity.  The right arm and right
    master remain owned by their ActionTerms, and no passive/mimic OmniPicker
    coordinate is written here.
    """

    robot = env.scene["robot"]
    names = (
        *TORSO_JOINTS,
        *(f"idx1{i}_head_joint{i}" for i in range(1, 4)),
        *LEFT_ARM_JOINTS,
    )
    ids = [robot.joint_names.index(name) for name in names]
    default = tensor_value(robot.data.default_joint_pos)
    ids_tensor = torch.tensor(ids, dtype=torch.long, device=env.device)
    target = default.index_select(1, ids_tensor)[env_ids]
    expected_fixed = torch.tensor(
        (*FIXED_TORSO_Q, *EXHIBITION_HEAD_Q),
        dtype=target.dtype,
        device=target.device,
    )
    if not bool(torch.allclose(target[:, : len(expected_fixed)], expected_fixed)):
        raise RuntimeError("G2_HELD_TORSO_HEAD_RESET_TARGET_MISMATCH")
    # A full-width target write would author targets for both OmniPicker
    # mimic/passive four-bars, even though this event has no hand authority.
    robot.set_joint_position_target(target, joint_ids=ids, env_ids=env_ids)
    robot.set_joint_velocity_target(
        torch.zeros_like(target), joint_ids=ids, env_ids=env_ids
    )


def fixed_torso_drift(env, maximum_drift_rad: float = 0.01) -> torch.Tensor:
    """Fail if the held torso departs from its sealed zero task posture."""

    if maximum_drift_rad <= 0.0:
        raise ValueError("maximum torso drift must be positive")
    robot = env.scene["robot"]
    if not hasattr(env, "_g2_fixed_torso_joint_ids"):
        env._g2_fixed_torso_joint_ids = torch.tensor(
            [robot.joint_names.index(name) for name in TORSO_JOINTS],
            dtype=torch.long,
            device=env.device,
        )
    q = tensor_value(robot.data.joint_pos).index_select(
        1, env._g2_fixed_torso_joint_ids
    )
    return q.abs().amax(dim=-1) > float(maximum_drift_rad)


def _ee_cube_distance(env) -> torch.Tensor:
    ee = tensor_value(env.scene["ee_frame"].data.target_pos_w)[:, 0]
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    return torch.linalg.vector_norm(cube - ee, dim=-1)


def configure_grasp_reward_target(
    env,
    cube_minus_ee_root_m: tuple[float, float, float],
    *,
    calibration_rows: int,
) -> None:
    """Install the calibrated side-pinch target before the first env step.

    The reference controller, grasp-ready interlock and reward must consume
    one robot-root-frame target.  Missing or stale calibration is therefore a
    hard configuration error rather than a silent zero-centroid fallback.
    """

    if calibration_rows <= 0:
        raise ValueError("grasp reward target requires calibration rows")
    target = torch.as_tensor(
        cube_minus_ee_root_m,
        dtype=tensor_value(env.scene["object"].data.root_pos_w).dtype,
        device=env.device,
    )
    if target.shape != (3,) or not bool(torch.isfinite(target).all()):
        raise ValueError("grasp reward target must be one finite 3-vector")
    env._g2_task_grasp_cube_minus_ee_root_m = target
    env._g2_task_grasp_target_calibration_rows = int(calibration_rows)


def _cube_minus_ee_root(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ee = tensor_value(env.scene["ee_frame"].data.target_pos_w)[:, 0]
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    return quat_apply_inverse(
        tensor_value(robot.data.root_quat_w),
        cube - ee,
    )


def _grasp_offset_error_distance(env) -> torch.Tensor:
    target = getattr(env, "_g2_task_grasp_cube_minus_ee_root_m", None)
    rows = int(getattr(env, "_g2_task_grasp_target_calibration_rows", 0))
    if not isinstance(target, torch.Tensor) or target.shape != (3,) or rows <= 0:
        raise RuntimeError("G2_GRASP_REWARD_TARGET_NOT_CONFIGURED")
    if target.device != torch.device(env.device) or not bool(
        torch.isfinite(target).all()
    ):
        raise RuntimeError("G2_GRASP_REWARD_TARGET_INVALID")
    return torch.linalg.vector_norm(_cube_minus_ee_root(env) - target, dim=-1)


def _goal_distance(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    command = env.command_manager.get_command("object_pose")[:, :3]
    # UniformPoseCommand is root-local. Use the framework's native transform
    # authority; persisted policy quaternions are converted elsewhere.
    goal_w, _ = combine_frame_transforms(
        tensor_value(robot.data.root_pos_w),
        tensor_value(robot.data.root_quat_w),
        command,
    )
    return torch.linalg.vector_norm(goal_w - cube, dim=-1)


def _pbrs(
    env,
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    scale: float,
    gamma: float,
    phase: torch.Tensor,
    previous_phase: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Bounded potential-difference shaping with distance potential -d/scale.

    Clipping is intentional for critic stability, so this is not advertised
    as an exact policy-invariance theorem despite the historic ``pbrs`` name.
    """

    if not 0.0 < gamma < 1.0:
        raise ValueError("PBRS gamma must be in (0, 1)")
    if scale <= 0.0:
        raise ValueError("PBRS distance scale must be positive")
    phase_changed = (previous_phase >= 0) & (phase != previous_phase)
    previous[phase_changed] = torch.nan
    previous_phase.copy_(phase)
    initialized = torch.isfinite(previous) & active
    old = torch.where(initialized, previous, current)
    transition_reward = torch.clamp((-gamma * current + old) / scale, -1.0, 1.0)
    previous.copy_(torch.where(active, current, torch.full_like(current, torch.nan)))
    transition_reward = torch.where(
        initialized, transition_reward, torch.zeros_like(transition_reward)
    )
    # Isaac Lab's RewardManager multiplies every term by env.step_dt. PBRS is
    # already a discrete transition credit, so expose its rate-equivalent here
    # to avoid applying dt twice.
    return transition_reward / float(env.step_dt)


def ee_cube_progress_pbrs(
    env, scale_m: float = 0.10, gamma: float = DEFAULT_SAC_REWARD_GAMMA
) -> torch.Tensor:
    """Progress toward the calibrated side-pinch pose, not cube centroid.

    The public name is retained for CLI/checkpoint-metadata compatibility;
    W&B publishes the explicit grasp-offset alias.
    """

    _ensure_state(env)
    closing_touch, closing_bilateral = _closing_contact_evidence(env)
    env._g2_task_ever_closing_finger_contact |= closing_touch
    env._g2_task_ever_bilateral_contact |= closing_bilateral
    # Reach shaping owns only free-space approach. The first intentional
    # finger/object contact transfers authority to physical grasp rewards so
    # approach credit cannot be farmed while squeezing or hovering in contact.
    active = ~env._g2_task_ever_closing_finger_contact
    phase = env._g2_task_ever_closing_finger_contact.to(torch.long)
    return _pbrs(
        env,
        env._g2_task_previous_ee_distance,
        _grasp_offset_error_distance(env),
        scale=scale_m,
        gamma=gamma,
        phase=phase,
        previous_phase=env._g2_task_previous_ee_phase,
        active=active,
    )


def cube_goal_progress_pbrs(
    env, scale_m: float = 0.10, gamma: float = DEFAULT_SAC_REWARD_GAMMA
) -> torch.Tensor:
    _ensure_state(env)
    cube_z = tensor_value(env.scene["object"].data.root_pos_w)[:, 2]
    current = _goal_distance(env)
    active = cube_z >= TASK.table_surface_height_m + TASK.lift_height_above_table_m
    shaped = _pbrs(
        env,
        env._g2_task_previous_goal_distance,
        current,
        scale=scale_m,
        gamma=gamma,
        phase=active.to(torch.long),
        previous_phase=env._g2_task_previous_goal_phase,
        active=active,
    )
    return shaped


CONTACT_BOOLEAN_FORCE_THRESHOLD_N = 1.0


def _contact(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inner = torch.linalg.vector_norm(
        tensor_value(env.scene["right_inner_finger_contact"].data.force_matrix_w)[:, 0, 0], dim=-1
    )
    outer = torch.linalg.vector_norm(
        tensor_value(env.scene["right_outer_finger_contact"].data.force_matrix_w)[:, 0, 0], dim=-1
    )
    return inner, outer, (
        (inner > CONTACT_BOOLEAN_FORCE_THRESHOLD_N)
        & (outer > CONTACT_BOOLEAN_FORCE_THRESHOLD_N)
    )


def contact_boolean_telemetry(
    env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source-owned side/any/bilateral boolean contact telemetry.

    Downstream label, reward and replay code consumes these booleans rather
    than re-thresholding raw force.  Raw force remains available separately
    through :func:`contact_grasp_telemetry` for diagnostics and safety.
    """

    inner_force, outer_force, _ = _contact(env)
    inner_contact = inner_force > CONTACT_BOOLEAN_FORCE_THRESHOLD_N
    outer_contact = outer_force > CONTACT_BOOLEAN_FORCE_THRESHOLD_N
    return (
        inner_contact,
        outer_contact,
        inner_contact | outer_contact,
        inner_contact & outer_contact,
    )


def _gripper_close_command_active(env) -> torch.Tensor:
    """Return accepted CLOSE intent from the actuator action authority.

    A measured aperture or exact mechanical endpoint is intentionally not
    used: an object held between the pads prevents empty-hand closure. The
    custom G2 action term also accounts for reset-time forced-open protection.
    Missing or malformed command evidence is a configuration error rather
    than an implicit grasp claim.
    """

    action_manager = getattr(env, "action_manager", None)
    terms = getattr(action_manager, "_terms", None)
    gripper_term = terms.get("gripper_action") if isinstance(terms, dict) else None
    close_active = getattr(gripper_term, "close_command_active", None)
    if not isinstance(close_active, torch.Tensor):
        raise RuntimeError("G2_GRIPPER_CLOSE_COMMAND_EVIDENCE_UNAVAILABLE")
    if (
        close_active.shape != (env.num_envs,)
        or close_active.dtype != torch.bool
        or close_active.device != torch.device(env.device)
    ):
        raise RuntimeError("G2_GRIPPER_CLOSE_COMMAND_EVIDENCE_INVALID")
    return close_active


def _closing_contact_evidence(
    env, force_threshold_n: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return intentional any-finger and bilateral cube contact evidence."""

    if force_threshold_n <= 0.0:
        raise ValueError("finger contact force threshold must be positive")
    inner_force, outer_force, _ = _contact(env)
    close_active = _gripper_close_command_active(env)
    if float(force_threshold_n) == CONTACT_BOOLEAN_FORCE_THRESHOLD_N:
        inner_contact, outer_contact, _, _ = contact_boolean_telemetry(env)
    else:
        # Non-canonical thresholds are retained only for explicit diagnostics.
        inner_contact = inner_force > float(force_threshold_n)
        outer_contact = outer_force > float(force_threshold_n)
    any_touch = close_active & (inner_contact | outer_contact)
    bilateral = close_active & inner_contact & outer_contact
    return any_touch, bilateral


def _rigid_point_slip_speed_m_s(env) -> torch.Tensor:
    """Cube velocity relative to the gripper rigid-point velocity."""

    robot = env.scene["robot"]
    cube = env.scene["object"]
    ee_id = robot.body_names.index(RIGHT_EE_BODY)
    gripper_origin = tensor_value(robot.data.body_pos_w)[:, ee_id]
    cube_position = tensor_value(cube.data.root_pos_w)
    gripper_point_velocity = (
        tensor_value(robot.data.body_lin_vel_w)[:, ee_id]
        + torch.linalg.cross(
            tensor_value(robot.data.body_ang_vel_w)[:, ee_id],
            cube_position - gripper_origin,
            dim=-1,
        )
    )
    return torch.linalg.vector_norm(
        tensor_value(cube.data.root_lin_vel_w) - gripper_point_velocity,
        dim=-1,
    )


def _elapsed_at_least(elapsed_s: torch.Tensor, duration_s: float) -> torch.Tensor:
    """Compare elapsed physical time without adding a control-step delay."""

    if duration_s <= 0.0 or not torch.is_floating_point(elapsed_s):
        raise ValueError("duration must be positive and elapsed time floating point")
    roundoff_s = (
        8.0
        * torch.finfo(elapsed_s.dtype).eps
        * max(1.0, abs(float(duration_s)))
    )
    return elapsed_s >= float(duration_s) - roundoff_s


def contact_grasp_telemetry(
    env,
    slip_limit_m_s: float = 0.06,
    stable_time_s: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read contact, rigid-point slip and latched stable-grasp state.

    This function never advances the stable timer; the reward term remains
    its sole update authority, so recorders cannot change task dynamics.
    """

    _ensure_state(env)
    inner, outer, bilateral = _contact(env)
    slip_speed = _rigid_point_slip_speed_m_s(env)
    close_active = _gripper_close_command_active(env)
    stable = (
        close_active
        & bilateral
        & (slip_speed < float(slip_limit_m_s))
        & _elapsed_at_least(env._g2_task_stable_time_s, stable_time_s)
    )
    return inner, outer, bilateral, slip_speed, stable


def _after_first_touch_grace(env, grace_s: float) -> torch.Tensor:
    """Latch first finger touch and preserve a real post-touch recovery grace."""

    _ensure_state(env)
    inner_contact, outer_contact, _, _ = contact_boolean_telemetry(env)
    step = tensor_value(env.episode_length_buf).to(torch.long)
    first_touch = env._g2_task_first_touch_step
    newly_touched = (first_touch < 0) & (inner_contact | outer_contact)
    first_touch.copy_(torch.where(newly_touched, step, first_touch))
    grace_steps = max(1, int(math.ceil(grace_s / float(env.step_dt))))
    post_touch = (first_touch >= 0) & ((step - first_touch) >= grace_steps)
    # Motion without any finger-contact evidence is not called a recoverable
    # first touch, but reset settling still receives the same bounded grace.
    no_touch_reset_grace_elapsed = (first_touch < 0) & (step >= grace_steps)
    return post_touch | no_touch_reset_grace_elapsed


def _one_shot_credit_rate(
    env, event: torch.Tensor, emitted_latch: torch.Tensor
) -> torch.Tensor:
    """Return a fixed event credit through RewardManager's rate interface."""

    newly_emitted = event & (~emitted_latch)
    emitted_latch |= event
    return newly_emitted.float() / float(env.step_dt)


def bilateral_contact_reward(env) -> torch.Tensor:
    """One-shot credit for bilateral cube contact during a close command."""

    _ensure_state(env)
    _, bilateral = _closing_contact_evidence(env)
    env._g2_task_ever_bilateral_contact |= bilateral
    return _one_shot_credit_rate(
        env, bilateral, env._g2_task_contact_milestone_emitted
    )


def bilateral_contact_maintenance_reward(env) -> torch.Tensor:
    """Small rate while intentional bilateral contact is maintained."""

    return _closing_contact_evidence(env)[1].float()


def any_finger_touch_reward(env, force_threshold_n: float = 1.0) -> torch.Tensor:
    """One-shot closing-finger touch milestone, below bilateral grasp."""

    _ensure_state(env)
    touched, _ = _closing_contact_evidence(
        env, force_threshold_n=force_threshold_n
    )
    return _one_shot_credit_rate(
        env, touched, env._g2_task_touch_milestone_emitted
    )


def stable_grasp_reward(env, slip_limit_m_s: float = 0.06, stable_time_s: float = 0.10) -> torch.Tensor:
    """Credit an intentional, bilateral, low-slip grasp exactly once."""

    _ensure_state(env)
    _, bilateral = _closing_contact_evidence(env)
    relative_speed = _rigid_point_slip_speed_m_s(env)
    valid = bilateral & (relative_speed < slip_limit_m_s)
    env._g2_task_stable_time_s = torch.where(
        valid,
        env._g2_task_stable_time_s + float(env.step_dt),
        torch.zeros_like(env._g2_task_stable_time_s),
    )
    stable = _elapsed_at_least(env._g2_task_stable_time_s, stable_time_s)
    env._g2_task_ever_stable_grasp |= stable
    lifted = tensor_value(env.scene["object"].data.root_pos_w)[:, 2] >= (
        TASK.table_surface_height_m + TASK.lift_height_above_table_m
    )
    env._g2_task_ever_lifted_while_stable |= stable & lifted
    return _one_shot_credit_rate(
        env, stable, env._g2_task_stable_milestone_emitted
    )


def grasp_slip_excess(
    env,
    slip_limit_m_s: float = 0.06,
    saturation_span_m_s: float = 0.06,
) -> torch.Tensor:
    """Bounded slip excess during an intentional bilateral grasp.

    The term is zero before close/contact and is a dimensionless rate in
    ``[0, 1]``. Configuration supplies the negative weight. Saturating one
    existing slip-limit above the stable threshold keeps isolated velocity
    noise bounded while penalizing sustained physical slip.
    """

    if slip_limit_m_s <= 0.0 or saturation_span_m_s <= 0.0:
        raise ValueError("slip limit and saturation span must be positive")
    _, bilateral = _closing_contact_evidence(env)
    slip_speed = _rigid_point_slip_speed_m_s(env)
    excess = torch.clamp(
        (slip_speed - float(slip_limit_m_s)) / float(saturation_span_m_s),
        0.0,
        1.0,
    )
    return torch.where(bilateral, excess, torch.zeros_like(excess))


def stable_grasp_maintenance_reward(
    env, slip_limit_m_s: float = 0.06, stable_time_s: float = 0.10
) -> torch.Tensor:
    """Small rate reward for a currently maintained stable grasp.

    This reader deliberately does not advance the stable timer. The preceding
    ``stable_grasp_reward`` term remains the sole timer update authority.
    """

    return contact_grasp_telemetry(
        env,
        slip_limit_m_s=slip_limit_m_s,
        stable_time_s=stable_time_s,
    )[4].float()


def stable_lift_milestone_reward(
    env, slip_limit_m_s: float = 0.06, stable_time_s: float = 0.10
) -> torch.Tensor:
    """Emit one fixed credit when a stable grasp first reaches lift height."""

    _ensure_state(env)
    stable = contact_grasp_telemetry(
        env,
        slip_limit_m_s=slip_limit_m_s,
        stable_time_s=stable_time_s,
    )[4]
    lifted = tensor_value(env.scene["object"].data.root_pos_w)[:, 2] >= (
        TASK.table_surface_height_m + TASK.lift_height_above_table_m
    )
    event = stable & lifted
    env._g2_task_ever_lifted_while_stable |= event
    return _one_shot_credit_rate(
        env, event, env._g2_task_lift_milestone_emitted
    )


def partial_grasp_terminal_credit(env) -> torch.Tensor:
    """One-shot, low terminal credit after real grasp progress was lost.

    The credit is deliberately emitted only at timeout/drop, not at the loss
    instant, so the policy is never rewarded for deliberately releasing the
    object.  It preserves scarce touch/grasp experience below full lift
    success without weakening any termination or safety authority.
    """

    _ensure_state(env)
    cube_z = tensor_value(env.scene["object"].data.root_pos_w)[:, 2]
    dropped = cube_z < TASK.table_surface_height_m - TASK.drop_margin_below_table_m
    timed_out = env.episode_length_buf >= env.max_episode_length
    eligible = (
        env._g2_task_ever_stable_grasp
        & (dropped | timed_out)
        & (~env._g2_task_partial_credit_emitted)
    )
    env._g2_task_partial_credit_emitted |= eligible
    return eligible.float()


def push_recovery_pbrs(env, grace_s: float = 0.10, scale_m: float = 0.02) -> torch.Tensor:
    """Reward returning a prematurely pushed cube; do nothing on first touch."""

    _ensure_state(env)
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    displacement = torch.linalg.vector_norm(
        cube[:, :2] - env._g2_task_initial_cube_w[:, :2], dim=-1
    )
    previous = env._g2_task_previous_push_distance.clone()
    bilateral = _contact(env)[2]
    active = _after_first_touch_grace(env, grace_s) & (~bilateral)
    phase = active.to(torch.long)
    initialized = env._g2_task_previous_push_phase >= 0
    phase_changed = initialized & (phase != env._g2_task_previous_push_phase)
    env._g2_task_previous_push_phase.copy_(phase)
    transition_reward = torch.clamp(
        (previous - displacement) / scale_m, -1.0, 1.0
    )
    valid = active & initialized & (~phase_changed)
    env._g2_task_previous_push_distance.copy_(displacement)
    # This is also a discrete potential delta; compensate RewardManager's dt.
    return torch.where(
        valid, transition_reward, torch.zeros_like(displacement)
    ) / float(env.step_dt)


def persistent_push_magnitude(env, grace_s: float = 0.10, free_motion_m: float = 0.003) -> torch.Tensor:
    _ensure_state(env)
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    displacement = torch.linalg.vector_norm(
        cube[:, :2] - env._g2_task_initial_cube_w[:, :2], dim=-1
    )
    bilateral = _contact(env)[2]
    magnitude = torch.clamp((displacement - free_motion_m) / 0.02, 0.0, 1.0)
    return torch.where(
        _after_first_touch_grace(env, grace_s) & (~bilateral),
        magnitude,
        torch.zeros_like(magnitude),
    )


def certified_region_violation(
    env,
    reachable_x_range_m: tuple[float, float] = TASK.reachable_cube_x_range_m,
    reachable_y_range_m: tuple[float, float] = TASK.reachable_cube_y_range_m,
) -> torch.Tensor:
    """Return workspace violation against an explicit environment contract."""

    if not reachable_x_range_m[0] < reachable_x_range_m[1]:
        raise ValueError("reachable X range must be increasing")
    if not reachable_y_range_m[0] < reachable_y_range_m[1]:
        raise ValueError("reachable Y range must be increasing")
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    robot_root = tensor_value(env.scene["robot"].data.root_pos_w)
    local = cube - robot_root
    dx = torch.relu(reachable_x_range_m[0] - local[:, 0]) + torch.relu(
        local[:, 0] - reachable_x_range_m[1]
    )
    dy = torch.relu(reachable_y_range_m[0] - local[:, 1]) + torch.relu(
        local[:, 1] - reachable_y_range_m[1]
    )
    return torch.clamp(torch.sqrt(dx.square() + dy.square()) / 0.02, 0.0, 1.0)


def grasp_orientation_excess(
    env,
    free_angle_rad: float = math.radians(5.0),
    desired_orientation_root_xyzw: tuple[float, float, float, float] = (
        0.0, 1.0, 0.0, 0.0
    ),
) -> torch.Tensor:
    quaternion = tensor_value(env.scene["ee_frame"].data.target_quat_w)[:, 0]
    # The shared task keeps its canonical default. Keyboard collection may
    # supply its audited 90-degree reset-to-grasp endpoint explicitly so its
    # navigation and recorded reward use the same orientation authority.
    desired_xyzw = quaternion.new_tensor(
        desired_orientation_root_xyzw
    ).expand_as(quaternion)
    desired = quaternion_xyzw_to_native(
        desired_xyzw, isaaclab_native_quaternion_order()
    )
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(torch.sum(quaternion * desired, dim=-1)), 0.0, 1.0))
    _, bilateral = _closing_contact_evidence(env)
    lifted = tensor_value(env.scene["object"].data.root_pos_w)[:, 2] >= (
        TASK.table_surface_height_m + TASK.lift_height_above_table_m
    )
    excess = torch.clamp(
        (angle - free_angle_rad) / max(TASK.maximum_grasp_orientation_error_rad - free_angle_rad, 1.0e-6),
        0.0,
        1.0,
    )
    return torch.where(bilateral | lifted, excess, torch.zeros_like(excess))


def controlled_joint_acceleration_excess(env, free_rad_s2: float = 5.0) -> torch.Tensor:
    _ensure_state(env)
    qd = tensor_value(env.scene["robot"].data.joint_vel).index_select(
        1, env._g2_task_controlled_joint_ids
    )
    acceleration = (qd - env._g2_task_previous_joint_velocity) / float(env.step_dt)
    env._g2_task_previous_joint_velocity.copy_(qd)
    peak = acceleration.abs().amax(dim=-1)
    return torch.clamp((peak - free_rad_s2) / (10.0 - free_rad_s2), 0.0, 1.0)


def cube_left_table_persistently(env, persistence_s: float = 0.20) -> torch.Tensor:
    """Terminate only after the cube leaves the physical table, never on touch."""

    _ensure_state(env)
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    robot_root = tensor_value(env.scene["robot"].data.root_pos_w)
    local = cube - robot_root
    half_x = TASK.table_size_m[0] / 2.0 - TASK.cube_half_extents_m[0]
    half_y = TASK.table_size_m[1] / 2.0 - TASK.cube_half_extents_m[1]
    # A task-specific environment may move the table without mutating the
    # shared training contract.  Its runtime must then publish the matching
    # robot-root-frame center.  Falling back to TASK preserves every existing
    # environment, while preventing keyboard-only geometry from being judged
    # against a stale table location.
    table_center = getattr(env, "_g2_task_table_center_root_m", None)
    if table_center is None:
        table_center_x = TASK.table_center_world_m[0]
        table_center_y = TASK.table_center_world_m[1]
    else:
        table_center = torch.as_tensor(
            table_center, dtype=local.dtype, device=local.device
        )
        if table_center.shape != (3,) or not bool(torch.isfinite(table_center).all()):
            raise RuntimeError("G2_TASK_TABLE_CENTER_ROOT_INVALID")
        table_center_x = table_center[0]
        table_center_y = table_center[1]
    outside = (
        (torch.abs(local[:, 0] - table_center_x) > half_x)
        | (torch.abs(local[:, 1] - table_center_y) > half_y)
    )
    env._g2_task_outside_table_time_s = torch.where(
        outside,
        env._g2_task_outside_table_time_s + float(env.step_dt),
        torch.zeros_like(env._g2_task_outside_table_time_s),
    )
    return env._g2_task_outside_table_time_s >= persistence_s


def object_reached_lifted_goal(env, threshold: float = 0.005) -> torch.Tensor:
    """Canonical Level-1 success: lifted cube inside the goal tolerance.

    This predicate is shared conceptually with teacher logging, HER and data
    collection.  It avoids declaring a low-altitude coincidence a success and
    deliberately uses ``<=`` at the sealed tolerance boundary.
    """

    if threshold <= 0.0:
        raise ValueError("goal threshold must be positive")
    robot = env.scene["robot"]
    cube = tensor_value(env.scene["object"].data.root_pos_w)
    command = env.command_manager.get_command("object_pose")[:, :3]
    goal_w, _ = combine_frame_transforms(
        tensor_value(robot.data.root_pos_w),
        tensor_value(robot.data.root_quat_w),
        command,
    )
    lifted = cube[:, 2] >= (
        TASK.table_surface_height_m + TASK.lift_height_above_table_m
    )
    return lifted & (torch.linalg.vector_norm(goal_w - cube, dim=-1) <= threshold)


def object_stably_grasped_and_lifted(
    env, slip_limit_m_s: float = 0.06, stable_time_s: float = 0.10
) -> torch.Tensor:
    """Level-1 physical success, independent of placement-centroid error."""

    _ensure_state(env)
    _, bilateral = _closing_contact_evidence(env)
    stable = (
        bilateral
        & (_rigid_point_slip_speed_m_s(env) < float(slip_limit_m_s))
        & _elapsed_at_least(env._g2_task_stable_time_s, stable_time_s)
    )
    lifted = tensor_value(env.scene["object"].data.root_pos_w)[:, 2] >= (
        TASK.table_surface_height_m + TASK.lift_height_above_table_m
    )
    return stable & lifted


def time_out_without_success(env, success_threshold: float = 0.005) -> torch.Tensor:
    """Time out only non-successful rows, preserving one terminal category."""

    timed_out = env.episode_length_buf >= env.max_episode_length
    del success_threshold  # retained for config/checkpoint CLI compatibility
    return timed_out & (~object_stably_grasped_and_lifted(env))


def forbidden_collision(env, force_threshold_n: float = 1.0e-6) -> torch.Tensor:
    """Terminate only from a complete, live forbidden-contact partition."""

    evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
    if evaluator is None:
        evaluator = G2ForbiddenCollisionEvaluator(
            env, force_threshold_n=force_threshold_n
        )
        env._g2_forbidden_collision_evaluator = evaluator
    return evaluator()


__all__ = [name for name in globals() if not name.startswith("_")]
