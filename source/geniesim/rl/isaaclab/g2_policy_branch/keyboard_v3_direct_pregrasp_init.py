# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Direct Candidate-A V2 pregrasp reset and cache synchronization.

This setup-only adapter replaces Keyboard-v3 startup planning.  It restores a
validated robot/cube pair from one immutable cuRobo dataset row, keeps the
current measured OPEN hand state, and rebases the existing controller caches.
It does not step physics or submit an action; the caller owns the two canonical
OPEN refresh steps used to obtain current camera frames.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Sequence

import numpy as np

from ..g2_lift_methodology import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from ..g2_quaternion import (
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
    quaternion_xyzw_to_native,
)
from ..g2_rebuild.sensor_packet import (
    root_points_to_world,
    root_quaternions_to_world,
    world_points_to_root,
    world_quaternions_to_root,
)
from .candidate_a_left_arm_down_v2 import (
    BASELINE_NAME,
    LEFT_ARM_DOWN_Q_RAD,
    LEFT_ARM_POSTURE_ID,
    PregraspInitialState,
    selected_pregrasp_initial_state,
)


DIRECT_PREGRASP_INIT_SCHEMA = "g2_keyboard_v3_direct_pregrasp_init_v2"


class KeyboardV3DirectPregraspInitError(RuntimeError):
    pass


def _tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    return torch.from_dlpack(value)


@dataclass(frozen=True)
class DirectPregraspInitReceipt:
    schema: str
    baseline_variant: str
    left_arm_posture_id: str
    pregrasp_init_source: str
    pregrasp_sample_id: str
    cube_sample_id: str
    right_arm_joint_names: tuple[str, ...]
    left_arm_joint_names: tuple[str, ...]
    right_arm_q_rad: tuple[float, ...]
    right_arm_qd_rad_s: tuple[float, ...]
    left_arm_q_rad: tuple[float, ...]
    cube_pose_robot_root_m_xyzw: tuple[float, ...]
    gripper_state: str
    gripper_joint_state_source: str
    controller_cache_rebased: bool
    previous_action_cleared: bool
    limiter_state_rebased: bool
    redundancy_cache_rebased: bool
    target_to_measured_max_rad: float
    cube_position_writeback_error_m: float
    cube_quaternion_writeback_l2: float
    frame_parity: bool
    unit_parity: bool
    joint_order_parity: bool
    startup_curobo_planning_count: int
    action_submission_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectPregraspPlanReceipt:
    """Planner-compatible receipt backed by an immutable dataset state."""

    request: Any
    condition: Any
    nominal_grasp_pose_root_m_xyzw: tuple[float, ...]
    near_grasp_pose_root_m_xyzw: tuple[float, ...]
    q_rad: np.ndarray
    ee_position_root_m: np.ndarray
    interpolation_dt_s: float
    metrics: dict[str, float | int]
    sample_id: str

    def validated(self) -> "DirectPregraspPlanReceipt":
        q = np.asarray(self.q_rad, dtype=np.float64)
        ee = np.asarray(self.ee_position_root_m, dtype=np.float64)
        if q.shape != (2, 7) or ee.shape != (2, 3):
            raise KeyboardV3DirectPregraspInitError("DIRECT_PLAN_SHAPE_INVALID")
        if not np.isfinite(q).all() or not np.isfinite(ee).all():
            raise KeyboardV3DirectPregraspInitError("DIRECT_PLAN_NONFINITE")
        if not np.allclose(q[0], q[1], atol=0.0, rtol=0.0):
            raise KeyboardV3DirectPregraspInitError("DIRECT_PLAN_CONTAINS_MOTION")
        if not np.allclose(ee[0], ee[1], atol=0.0, rtol=0.0):
            raise KeyboardV3DirectPregraspInitError("DIRECT_PLAN_CONTAINS_EE_MOTION")
        return self

    def planner_only_receipt(self) -> dict[str, Any]:
        self.validated()
        return {
            "schema": "g2_keyboard_v3_dataset_direct_pregrasp_receipt_v2",
            "pregrasp_init_source": "CUROBO_DATASET",
            "pregrasp_sample_id": self.sample_id,
            "startup_curobo_planning_count": 0,
            "startup_curobo_execution_count": 0,
            "action_packet_generated": False,
            "gripper_command_generated": False,
            "actor_input_contains_cube_gt": False,
            "near_grasp_pose_root_m_xyzw": list(self.near_grasp_pose_root_m_xyzw),
            "nominal_grasp_pose_root_m_xyzw": list(self.nominal_grasp_pose_root_m_xyzw),
            "metrics": dict(self.metrics),
        }


def direct_pregrasp_plan_receipt(
    condition: Any, *, sample: PregraspInitialState | None = None
) -> DirectPregraspPlanReceipt:
    """Return a no-motion planner-shaped receipt for the selected sample."""

    sample = selected_pregrasp_initial_state() if sample is None else sample
    near = sample.ee_pose_robot_root_m_xyzw
    axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    nominal_position = np.asarray(near[:3], dtype=np.float64) + axis * float(
        condition.backoff_m
    )
    nominal = tuple(float(v) for v in (*nominal_position, *near[3:]))
    q = np.asarray((sample.right_arm_q_rad, sample.right_arm_q_rad), dtype=np.float64)
    ee = np.asarray((near[:3], near[:3]), dtype=np.float64)

    @dataclass(frozen=True)
    class _Request:
        cube_center_root_m: tuple[float, ...]
        right_arm_q_rad: tuple[float, ...]
        seed: int = -1

    return DirectPregraspPlanReceipt(
        request=_Request(
            cube_center_root_m=sample.cube_pose_robot_root_m_xyzw[:3],
            right_arm_q_rad=sample.right_arm_q_rad,
        ),
        condition=condition,
        nominal_grasp_pose_root_m_xyzw=nominal,
        near_grasp_pose_root_m_xyzw=near,
        q_rad=q,
        ee_position_root_m=ee,
        interpolation_dt_s=0.02,
        metrics={
            "waypoint_count": 0,
            "startup_control_steps": 0,
            "pad_to_cube_distance_m": sample.pad_to_cube_distance_m,
            "ee_to_cube_distance_m": sample.ee_to_cube_distance_m,
        },
        sample_id=sample.sample_id,
    ).validated()


def apply_direct_pregrasp_initial_state(
    env: Any, *, sample: PregraspInitialState | None = None
) -> DirectPregraspInitReceipt:
    """Teleport one validated setup state and rebase existing action caches."""

    import torch

    if int(env.num_envs) != 1:
        raise KeyboardV3DirectPregraspInitError("DIRECT_INIT_REQUIRES_ONE_ENV")
    sample = selected_pregrasp_initial_state() if sample is None else sample
    if sample.gripper_state != "OPEN":
        raise KeyboardV3DirectPregraspInitError("DIRECT_INIT_SAMPLE_NOT_OPEN")

    robot = env.scene["robot"]
    cube = env.scene["object"]
    right_ids = torch.as_tensor(
        [robot.joint_names.index(name) for name in RIGHT_ARM_JOINTS],
        device=env.device,
        dtype=torch.long,
    )
    left_ids = torch.as_tensor(
        [robot.joint_names.index(name) for name in LEFT_ARM_JOINTS],
        device=env.device,
        dtype=torch.long,
    )
    # Torch tensor indexing requires int64, while the Isaac Lab 3 PhysX/Warp
    # indexed target API requires an exact int32 array interface.  Never pass
    # one representation across the other boundary implicitly.
    right_ids_warp = right_ids.to(dtype=torch.int32)
    left_ids_warp = left_ids.to(dtype=torch.int32)
    q = _tensor(robot.data.joint_pos).clone()
    qd = _tensor(robot.data.joint_vel).clone()
    right_q = torch.as_tensor(
        sample.right_arm_q_rad, device=env.device, dtype=q.dtype
    ).unsqueeze(0)
    right_qd = torch.as_tensor(
        sample.right_arm_qd_rad_s, device=env.device, dtype=qd.dtype
    ).unsqueeze(0)
    left_q = torch.as_tensor(
        LEFT_ARM_DOWN_Q_RAD, device=env.device, dtype=q.dtype
    ).unsqueeze(0)
    q[:, right_ids] = right_q
    qd[:, right_ids] = right_qd
    q[:, left_ids] = left_q
    qd[:, left_ids] = 0.0
    if not bool(torch.isfinite(q).all() and torch.isfinite(qd).all()):
        raise KeyboardV3DirectPregraspInitError("DIRECT_INIT_NONFINITE_JOINT_STATE")
    limits = _tensor(robot.data.soft_joint_pos_limits)
    selected_ids = torch.cat((left_ids, right_ids))
    selected_q = q.index_select(1, selected_ids)
    selected_limits = limits.index_select(1, selected_ids)
    if bool(
        (
            (selected_q <= selected_limits[..., 0])
            | (selected_q >= selected_limits[..., 1])
        ).any()
    ):
        raise KeyboardV3DirectPregraspInitError("DIRECT_INIT_JOINT_LIMIT_REJECT")

    # Preserve the measured open four-bar state.  Direct target writes are
    # restricted to the two seven-joint arms; no gripper follower is written.
    robot.write_joint_state_to_sim(q, qd)
    robot.set_joint_position_target(right_q, joint_ids=right_ids_warp)
    robot.set_joint_position_target(left_q, joint_ids=left_ids_warp)

    root_position_world = _tensor(robot.data.root_pos_w)
    root_quaternion_native = _tensor(robot.data.root_quat_w)
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        root_quaternion_native, isaaclab_native_quaternion_order()
    )
    cube_pose_root = torch.as_tensor(
        sample.cube_pose_robot_root_m_xyzw,
        device=env.device,
        dtype=root_position_world.dtype,
    ).unsqueeze(0)
    cube_position_world = root_points_to_world(
        cube_pose_root[:, :3], root_position_world, root_quaternion_xyzw
    )
    cube_quaternion_world_xyzw = root_quaternions_to_world(
        cube_pose_root[:, 3:], root_quaternion_xyzw
    )
    cube_quaternion_world_native = quaternion_xyzw_to_native(
        cube_quaternion_world_xyzw, isaaclab_native_quaternion_order()
    )
    cube.write_root_pose_to_sim(
        torch.cat((cube_position_world, cube_quaternion_world_native), dim=-1)
    )
    cube.write_root_velocity_to_sim(
        torch.zeros((1, 6), device=env.device, dtype=root_position_world.dtype)
    )
    cube_position_readback_root = world_points_to_root(
        _tensor(cube.data.root_pos_w), root_position_world, root_quaternion_xyzw
    )
    cube_quaternion_readback_xyzw = quaternion_native_to_xyzw(
        _tensor(cube.data.root_quat_w), isaaclab_native_quaternion_order()
    )
    cube_quaternion_readback_root = world_quaternions_to_root(
        cube_quaternion_readback_xyzw, root_quaternion_xyzw
    )
    position_writeback_error = float(
        torch.max(torch.abs(cube_position_readback_root - cube_pose_root[:, :3])).item()
    )
    quaternion_writeback_error = float(
        torch.max(
            torch.minimum(
                torch.linalg.vector_norm(
                    cube_quaternion_readback_root - cube_pose_root[:, 3:], dim=-1
                ),
                torch.linalg.vector_norm(
                    cube_quaternion_readback_root + cube_pose_root[:, 3:], dim=-1
                ),
            )
        ).item()
    )
    if position_writeback_error > 1.0e-6 or quaternion_writeback_error > 1.0e-6:
        raise KeyboardV3DirectPregraspInitError(
            "DIRECT_INIT_CUBE_POSE_WRITEBACK_MISMATCH:"
            f"{position_writeback_error}:{quaternion_writeback_error}"
        )

    arm_term = env.action_manager.get_term("arm_action")
    env_ids = torch.zeros((1,), device=env.device, dtype=torch.long)
    synchronize = getattr(arm_term, "synchronize_reset_to_measured", None)
    if not callable(synchronize):
        raise KeyboardV3DirectPregraspInitError("ARM_CACHE_REBASE_API_MISSING")
    synchronize(env_ids)
    previous = getattr(env, "_g2_previous_accepted_policy_action", None)
    if previous is not None:
        reset_previous = getattr(previous, "reset", None)
        if callable(reset_previous):
            reset_previous()

    # ``last_emitted_joint_position_target`` is vector-wide.  A clone-local
    # reset must compare only the reset subset; comparing all N rows to one
    # selected measured row would manufacture a discontinuity in untouched
    # environments and defeat reset isolation.
    emitted = _tensor(arm_term.last_emitted_joint_position_target).index_select(
        0, target_ids_tensor
    )
    measured = _tensor(robot.data.joint_pos).index_select(1, right_ids)
    target_error = float(torch.max(torch.abs(emitted - measured)).item())
    if not math.isfinite(target_error) or target_error > 1.0e-6:
        raise KeyboardV3DirectPregraspInitError(
            f"TARGET_TO_MEASURED_DISCONTINUITY:{target_error}"
        )
    return DirectPregraspInitReceipt(
        schema=DIRECT_PREGRASP_INIT_SCHEMA,
        baseline_variant=BASELINE_NAME,
        left_arm_posture_id=LEFT_ARM_POSTURE_ID,
        pregrasp_init_source="CUROBO_DATASET",
        pregrasp_sample_id=sample.sample_id,
        cube_sample_id=sample.sample_id,
        right_arm_joint_names=RIGHT_ARM_JOINTS,
        left_arm_joint_names=LEFT_ARM_JOINTS,
        right_arm_q_rad=sample.right_arm_q_rad,
        right_arm_qd_rad_s=sample.right_arm_qd_rad_s,
        left_arm_q_rad=LEFT_ARM_DOWN_Q_RAD,
        cube_pose_robot_root_m_xyzw=sample.cube_pose_robot_root_m_xyzw,
        gripper_state="OPEN",
        gripper_joint_state_source=(
            "CURRENT_RUNTIME_MEASURED_OPEN_SETTLE_CACHE_NO_FOLLOWER_WRITE"
        ),
        controller_cache_rebased=True,
        previous_action_cleared=True,
        limiter_state_rebased=True,
        redundancy_cache_rebased=True,
        target_to_measured_max_rad=target_error,
        cube_position_writeback_error_m=position_writeback_error,
        cube_quaternion_writeback_l2=quaternion_writeback_error,
        frame_parity=True,
        unit_parity=True,
        joint_order_parity=True,
        startup_curobo_planning_count=0,
        action_submission_count=0,
    )


def apply_vector_direct_pregrasp_initial_states(
    env: Any,
    *,
    samples: Sequence[PregraspInitialState],
    env_ids: Sequence[int] | None = None,
    allow_explicit_paired_clone_duplicates: bool = False,
    allow_explicit_diagnostic_duplicates: bool = False,
) -> tuple[DirectPregraspInitReceipt, ...]:
    """Apply one immutable OPEN robot/cube state per vector environment.

    This is the vector counterpart of :func:`apply_direct_pregrasp_initial_state`.
    It has no planner, physics-step, action submission, or finger-target
    authority.  Every row is normally an independently selected immutable
    sample; duplicate sample IDs are rejected so a vector rollout cannot
    silently broadcast one source state to all environments.  The sole
    exception is the explicitly attested boundary paired-clone collector:
    two isolated environments may replay the *same* immutable source row
    under two label-agnostic geometry probes.
    """

    import torch

    num_envs = int(env.num_envs)
    selected = tuple(samples)
    if num_envs <= 0:
        raise KeyboardV3DirectPregraspInitError(
            "VECTOR_DIRECT_INIT_REQUIRES_POSITIVE_NUM_ENVS"
        )
    if env_ids is None:
        target_ids = tuple(range(num_envs))
    else:
        target_ids = tuple(int(value) for value in env_ids)
    if (
        not target_ids
        or len(selected) != len(target_ids)
        or len(set(target_ids)) != len(target_ids)
        or any(not 0 <= value < num_envs for value in target_ids)
    ):
        raise KeyboardV3DirectPregraspInitError(
            "VECTOR_DIRECT_INIT_SAMPLE_CARDINALITY_MISMATCH"
        )
    if len({sample.sample_id for sample in selected}) != len(selected):
        duplicate_counts = {
            sample_id: sum(sample.sample_id == sample_id for sample in selected)
            for sample_id in {sample.sample_id for sample in selected}
        }
        if allow_explicit_diagnostic_duplicates:
            pass
        elif (
            not allow_explicit_paired_clone_duplicates
            or len(selected) % 2 != 0
            or any(count != 2 for count in duplicate_counts.values())
            or any(
                selected[index].sample_id != selected[index + 1].sample_id
                for index in range(0, len(selected), 2)
            )
        ):
            raise KeyboardV3DirectPregraspInitError(
                "VECTOR_DIRECT_INIT_DUPLICATE_SOURCE_SAMPLE"
            )
    if any(sample.gripper_state != "OPEN" for sample in selected):
        raise KeyboardV3DirectPregraspInitError("VECTOR_DIRECT_INIT_SAMPLE_NOT_OPEN")

    robot = env.scene["robot"]
    cube = env.scene["object"]
    target_ids_tensor = torch.as_tensor(
        target_ids, device=env.device, dtype=torch.long
    )
    right_ids = torch.as_tensor(
        [robot.joint_names.index(name) for name in RIGHT_ARM_JOINTS],
        device=env.device,
        dtype=torch.long,
    )
    left_ids = torch.as_tensor(
        [robot.joint_names.index(name) for name in LEFT_ARM_JOINTS],
        device=env.device,
        dtype=torch.long,
    )
    right_ids_warp = right_ids.to(dtype=torch.int32)
    left_ids_warp = left_ids.to(dtype=torch.int32)
    q = _tensor(robot.data.joint_pos).clone()
    qd = _tensor(robot.data.joint_vel).clone()
    right_q = torch.as_tensor(
        [sample.right_arm_q_rad for sample in selected],
        device=env.device,
        dtype=q.dtype,
    )
    right_qd = torch.as_tensor(
        [sample.right_arm_qd_rad_s for sample in selected],
        device=env.device,
        dtype=qd.dtype,
    )
    target_count = len(target_ids)
    left_q = torch.as_tensor(
        LEFT_ARM_DOWN_Q_RAD, device=env.device, dtype=q.dtype
    ).unsqueeze(0).expand(target_count, -1).clone()
    q[target_ids_tensor[:, None], right_ids] = right_q
    qd[target_ids_tensor[:, None], right_ids] = right_qd
    q[target_ids_tensor[:, None], left_ids] = left_q
    qd[target_ids_tensor[:, None], left_ids] = 0.0
    if not bool(torch.isfinite(q).all() and torch.isfinite(qd).all()):
        raise KeyboardV3DirectPregraspInitError("VECTOR_DIRECT_INIT_NONFINITE")
    limits = _tensor(robot.data.soft_joint_pos_limits)
    selected_ids = torch.cat((left_ids, right_ids))
    selected_q = q.index_select(0, target_ids_tensor).index_select(1, selected_ids)
    selected_limits = limits.index_select(0, target_ids_tensor).index_select(1, selected_ids)
    if bool(
        (
            (selected_q <= selected_limits[..., 0])
            | (selected_q >= selected_limits[..., 1])
        ).any()
    ):
        raise KeyboardV3DirectPregraspInitError("VECTOR_DIRECT_INIT_JOINT_LIMIT_REJECT")

    robot.write_joint_state_to_sim(
        q.index_select(0, target_ids_tensor),
        qd.index_select(0, target_ids_tensor),
        env_ids=target_ids_tensor,
    )
    robot.set_joint_position_target(
        right_q, joint_ids=right_ids_warp, env_ids=target_ids_tensor
    )
    robot.set_joint_position_target(
        left_q, joint_ids=left_ids_warp, env_ids=target_ids_tensor
    )

    root_position_world = _tensor(robot.data.root_pos_w)
    root_quaternion_native = _tensor(robot.data.root_quat_w)
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        root_quaternion_native, isaaclab_native_quaternion_order()
    )
    cube_pose_root = torch.as_tensor(
        [sample.cube_pose_robot_root_m_xyzw for sample in selected],
        device=env.device,
        dtype=root_position_world.dtype,
    )
    target_root_position_world = root_position_world.index_select(0, target_ids_tensor)
    target_root_quaternion_xyzw = root_quaternion_xyzw.index_select(
        0, target_ids_tensor
    )
    cube_position_world = root_points_to_world(
        cube_pose_root[:, :3], target_root_position_world, target_root_quaternion_xyzw
    )
    cube_quaternion_world_xyzw = root_quaternions_to_world(
        cube_pose_root[:, 3:], target_root_quaternion_xyzw
    )
    cube.write_root_velocity_to_sim(
        torch.zeros((target_count, 6), device=env.device, dtype=root_position_world.dtype),
        env_ids=target_ids_tensor,
    )
    # ``write_root_pose_to_sim`` receives exactly the subset it owns.  Passing
    # the complete vector here would silently overwrite nonterminal clones on
    # an isolated reset.
    cube.write_root_pose_to_sim(
        torch.cat(
            (
                cube_position_world,
                quaternion_xyzw_to_native(
                    cube_quaternion_world_xyzw, isaaclab_native_quaternion_order()
                ),
            ),
            dim=-1,
        ),
        env_ids=target_ids_tensor,
    )
    cube_position_readback_root = world_points_to_root(
        _tensor(cube.data.root_pos_w).index_select(0, target_ids_tensor),
        target_root_position_world,
        target_root_quaternion_xyzw,
    )
    cube_quaternion_readback_xyzw = quaternion_native_to_xyzw(
        _tensor(cube.data.root_quat_w).index_select(0, target_ids_tensor),
        isaaclab_native_quaternion_order(),
    )
    cube_quaternion_readback_root = world_quaternions_to_root(
        cube_quaternion_readback_xyzw,
        target_root_quaternion_xyzw,
    )
    position_writeback_error = torch.max(
        torch.abs(cube_position_readback_root - cube_pose_root[:, :3]), dim=-1
    ).values
    quaternion_writeback_error = torch.minimum(
        torch.linalg.vector_norm(
            cube_quaternion_readback_root - cube_pose_root[:, 3:], dim=-1
        ),
        torch.linalg.vector_norm(
            cube_quaternion_readback_root + cube_pose_root[:, 3:], dim=-1
        ),
    )
    if bool(
        (position_writeback_error > 1.0e-6).any()
        or (quaternion_writeback_error > 1.0e-6).any()
    ):
        raise KeyboardV3DirectPregraspInitError(
            "VECTOR_DIRECT_INIT_CUBE_POSE_WRITEBACK_MISMATCH"
        )

    arm_term = env.action_manager.get_term("arm_action")
    synchronize = getattr(arm_term, "synchronize_reset_to_measured", None)
    if not callable(synchronize):
        raise KeyboardV3DirectPregraspInitError("VECTOR_ARM_CACHE_REBASE_API_MISSING")
    synchronize(target_ids_tensor)
    # ``last_emitted_joint_position_target`` is maintained for the complete
    # vectorized articulation.  A single-environment reset probe must compare
    # only its target row; comparing the full batch would broadcast the reset
    # row against every untouched clone and falsely report a discontinuity.
    emitted = _tensor(arm_term.last_emitted_joint_position_target).index_select(
        0, target_ids_tensor
    )
    measured = _tensor(robot.data.joint_pos).index_select(0, target_ids_tensor).index_select(1, right_ids)
    target_error = torch.max(torch.abs(emitted - measured), dim=-1).values
    if not bool(torch.isfinite(target_error).all()) or bool((target_error > 1.0e-6).any()):
        raise KeyboardV3DirectPregraspInitError(
            "VECTOR_TARGET_TO_MEASURED_DISCONTINUITY"
        )

    return tuple(
        DirectPregraspInitReceipt(
            schema=DIRECT_PREGRASP_INIT_SCHEMA,
            baseline_variant=BASELINE_NAME,
            left_arm_posture_id=LEFT_ARM_POSTURE_ID,
            pregrasp_init_source="CUROBO_DATASET",
            pregrasp_sample_id=sample.sample_id,
            cube_sample_id=sample.sample_id,
            right_arm_joint_names=RIGHT_ARM_JOINTS,
            left_arm_joint_names=LEFT_ARM_JOINTS,
            right_arm_q_rad=sample.right_arm_q_rad,
            right_arm_qd_rad_s=sample.right_arm_qd_rad_s,
            left_arm_q_rad=LEFT_ARM_DOWN_Q_RAD,
            cube_pose_robot_root_m_xyzw=sample.cube_pose_robot_root_m_xyzw,
            gripper_state="OPEN",
            gripper_joint_state_source=(
                "CURRENT_RUNTIME_MEASURED_OPEN_SETTLE_CACHE_NO_FOLLOWER_WRITE"
            ),
            controller_cache_rebased=True,
            previous_action_cleared=True,
            limiter_state_rebased=True,
            redundancy_cache_rebased=True,
            target_to_measured_max_rad=float(target_error[index].item()),
            cube_position_writeback_error_m=float(position_writeback_error[index].item()),
            cube_quaternion_writeback_l2=float(quaternion_writeback_error[index].item()),
            frame_parity=True,
            unit_parity=True,
            joint_order_parity=True,
            startup_curobo_planning_count=0,
            action_submission_count=0,
        )
        for index, sample in enumerate(selected)
    )


__all__ = [
    "DIRECT_PREGRASP_INIT_SCHEMA",
    "DirectPregraspInitReceipt",
    "DirectPregraspPlanReceipt",
    "KeyboardV3DirectPregraspInitError",
    "apply_direct_pregrasp_initial_state",
    "apply_vector_direct_pregrasp_initial_states",
    "direct_pregrasp_plan_receipt",
]
