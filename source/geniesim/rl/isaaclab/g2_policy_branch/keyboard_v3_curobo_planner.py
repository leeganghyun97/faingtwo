# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Planner-only cuRobo handoff for one keyboard-v3 demonstration.

The planner owns only the right-arm path to a source-defined, fixed-
orientation near-grasp pose.  It never creates an action packet, writes a
finger target, or exposes cube ground truth to the student observation.  The
live runner remains the sole owner of ``env.step`` and of the operator's 4-D
``[dx, dy, dz, g]`` packet.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from ..g2_keyboard_pose import (
    G2_KEYBOARD_APPROACH_DIRECTION_ROOT,
    G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M,
)
from .contact_free_runtime_replan import (
    ContactFreeRuntimeReplanRequest,
    MOTIONGEN_POSITION_TOLERANCE_M,
    MOTIONGEN_ROTATION_TOLERANCE_RAD,
    POLICY_DT_S,
)
from .curobo_collision_world import (
    build_collision_world_dict,
    validated_cube_center_root_m,
)
from .curobo_planner_authority import (
    EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2,
    EXISTING_HARD_VELOCITY_LIMIT_RAD_S,
    FIXED_WRIST_ORIENTATION_XYZW,
    RIGHT_ARM_7DOF_JOINTS,
    build_right_arm_7dof_robot_config,
)
from .keyboard_v3_collection_contract import (
    NearGraspCondition,
    near_grasp_pose_root_m_xyzw,
)


KEYBOARD_V3_CUROBO_PLAN_SCHEMA = "g2_keyboard_v3_curobo_near_grasp_plan_v1"


class KeyboardV3CuroboPlannerError(RuntimeError):
    pass


def nominal_grasp_pose_root_m_xyzw_for_cube(
    cube_center_root_m: Sequence[float],
) -> tuple[float, ...]:
    """Return the source side-grasp EE target in robot-root metres/XYZW."""

    cube = validated_cube_center_root_m(cube_center_root_m)
    return (
        cube[0],
        cube[1],
        cube[2] + float(G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M),
        *tuple(float(value) for value in FIXED_WRIST_ORIENTATION_XYZW),
    )


@dataclass(frozen=True)
class KeyboardV3CuroboPlan:
    request: ContactFreeRuntimeReplanRequest
    condition: NearGraspCondition
    nominal_grasp_pose_root_m_xyzw: tuple[float, ...]
    near_grasp_pose_root_m_xyzw: tuple[float, ...]
    q_rad: np.ndarray
    ee_position_root_m: np.ndarray
    interpolation_dt_s: float
    metrics: dict[str, float | int]

    def validated(self) -> "KeyboardV3CuroboPlan":
        request = self.request.validated()
        q = np.asarray(self.q_rad, dtype=np.float64)
        ee = np.asarray(self.ee_position_root_m, dtype=np.float64)
        if (
            q.ndim != 2
            or q.shape[0] < 2
            or q.shape[1] != len(RIGHT_ARM_7DOF_JOINTS)
            or ee.shape != (q.shape[0], 3)
            or not np.isfinite(q).all()
            or not np.isfinite(ee).all()
        ):
            raise KeyboardV3CuroboPlannerError("KEYBOARD_V3_PLAN_PATH_INVALID")
        nominal = nominal_grasp_pose_root_m_xyzw_for_cube(
            request.cube_center_root_m
        )
        if not np.allclose(
            self.nominal_grasp_pose_root_m_xyzw, nominal, rtol=0.0, atol=1.0e-12
        ):
            raise KeyboardV3CuroboPlannerError("KEYBOARD_V3_NOMINAL_POSE_MISMATCH")
        expected_near = near_grasp_pose_root_m_xyzw(
            nominal,
            G2_KEYBOARD_APPROACH_DIRECTION_ROOT,
            self.condition,
        )
        if not np.allclose(
            self.near_grasp_pose_root_m_xyzw,
            expected_near,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise KeyboardV3CuroboPlannerError("KEYBOARD_V3_NEAR_POSE_MISMATCH")
        if float(np.linalg.norm(ee[-1] - np.asarray(expected_near[:3]))) > float(
            MOTIONGEN_POSITION_TOLERANCE_M
        ):
            raise KeyboardV3CuroboPlannerError("KEYBOARD_V3_FINAL_FK_OUT_OF_TOLERANCE")
        if not math.isfinite(float(self.interpolation_dt_s)) or self.interpolation_dt_s <= 0:
            raise KeyboardV3CuroboPlannerError("KEYBOARD_V3_PLAN_DT_INVALID")
        return self

    def planner_only_receipt(self) -> dict[str, Any]:
        self.validated()
        return {
            "schema": KEYBOARD_V3_CUROBO_PLAN_SCHEMA,
            "planner_input_cube_center_root_m": list(self.request.cube_center_root_m),
            "planner_input_right_arm_q_rad": list(self.request.right_arm_q_rad),
            "planner_seed": int(self.request.seed),
            "nominal_grasp_pose_root_m_xyzw": list(
                self.nominal_grasp_pose_root_m_xyzw
            ),
            "near_grasp_pose_root_m_xyzw": list(self.near_grasp_pose_root_m_xyzw),
            "approach_axis_root": list(G2_KEYBOARD_APPROACH_DIRECTION_ROOT),
            "backoff_m": float(self.condition.backoff_m),
            "condition_id": self.condition.condition_id,
            "planning_joints": list(RIGHT_ARM_7DOF_JOINTS),
            "waypoint_count": int(self.q_rad.shape[0]),
            "interpolation_dt_s": float(self.interpolation_dt_s),
            "metrics": dict(self.metrics),
            "actor_input_contains_cube_gt": False,
            "replay_row_contains_planner_cube_gt": False,
            "gripper_command_generated": False,
            "action_packet_generated": False,
        }


def _result_success(value: Any) -> bool:
    if hasattr(value, "detach"):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def plan_keyboard_v3_near_grasp(
    request: ContactFreeRuntimeReplanRequest,
    condition: NearGraspCondition,
) -> KeyboardV3CuroboPlan:
    """Plan from the measured reset arm state to the requested near pose."""

    request = request.validated()
    if any(abs(float(value)) > 1.0e-12 for value in condition.start_offset_xyz_m):
        raise KeyboardV3CuroboPlannerError(
            "ONE_SHOT_REQUIRES_CENTER_NEAR_GRASP_CONDITION"
        )
    nominal = nominal_grasp_pose_root_m_xyzw_for_cube(request.cube_center_root_m)
    near = near_grasp_pose_root_m_xyzw(
        nominal, G2_KEYBOARD_APPROACH_DIRECTION_ROOT, condition
    )

    # Lazy imports preserve simulator-free contract and unit-test paths.
    import torch
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )

    numpy_state = np.random.get_state()
    try:
        np.random.seed(request.seed)
        with torch.random.fork_rng(devices=[0], enabled=True):
            torch.manual_seed(request.seed)
            tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
            robot_config = build_right_arm_7dof_robot_config()
            world_config = build_collision_world_dict(
                cube_center_root_m=request.cube_center_root_m
            )
            orientation_wxyz = (
                near[6],
                near[3],
                near[4],
                near[5],
            )
            goal_pose = Pose.from_list([*near[:3], *orientation_wxyz], tensor_args)
            start_q = torch.tensor(
                [request.right_arm_q_rad], dtype=torch.float32, device="cuda:0"
            )
            start_state = JointState.from_position(
                start_q, joint_names=list(RIGHT_ARM_7DOF_JOINTS)
            )
            motion_config = MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_config,
                world_model=world_config,
                tensor_args=tensor_args,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                use_cuda_graph=False,
                num_ik_seeds=64,
                num_graph_seeds=4,
                num_trajopt_seeds=8,
                num_batch_ik_seeds=64,
                num_batch_trajopt_seeds=8,
                interpolation_dt=POLICY_DT_S,
                interpolation_steps=1000,
                trajopt_tsteps=32,
                collision_cache={"obb": 8, "mesh": 0},
                optimize_dt=True,
                position_threshold=MOTIONGEN_POSITION_TOLERANCE_M,
                rotation_threshold=MOTIONGEN_ROTATION_TOLERANCE_RAD,
                self_collision_check=True,
                self_collision_opt=True,
                ik_seed=request.seed,
            )
            motion_gen = MotionGen(motion_config)
            robot_world = RobotWorld(
                RobotWorldConfig.load_from_config(
                    robot_config,
                    world_config,
                    tensor_args=tensor_args,
                    collision_checker_type=CollisionCheckerType.PRIMITIVE,
                    collision_activation_distance=0.0,
                    self_collision_activation_distance=0.0,
                    max_collision_distance=1.0,
                    n_cuboids=8,
                )
            )
            result = motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(
                    enable_graph=True,
                    enable_opt=True,
                    max_attempts=8,
                    enable_finetune_trajopt=True,
                    parallel_finetune=True,
                    check_start_validity=True,
                    timeout=45.0,
                ),
            )
            if not _result_success(result.success):
                raise KeyboardV3CuroboPlannerError(
                    f"KEYBOARD_V3_MOTIONGEN_FAILED:{result.status}"
                )
            trajectory = result.get_interpolated_plan()
            q_tensor = trajectory.position.detach().reshape(-1, 7)
            q = q_tensor.cpu().numpy().astype(np.float64)
            ee = (
                motion_gen.kinematics.get_state(q_tensor.contiguous())
                .ee_position.detach()
                .cpu()
                .reshape(-1, 3)
                .numpy()
                .astype(np.float64)
            )
            dt = float(result.interpolation_dt)
            qd = np.diff(q, axis=0) / dt
            qdd = np.diff(qd, axis=0) / dt if qd.shape[0] > 1 else np.zeros((0, 7))
            limits = robot_world.kinematics.get_joint_limits().position
            limit_np = limits.detach().cpu().numpy().astype(np.float64)
            feasible = robot_world.validate_trajectory(q_tensor.unsqueeze(0)).detach().cpu()
            world_cost, self_cost = (
                robot_world.get_world_self_collision_distance_from_joint_trajectory(
                    q_tensor.unsqueeze(0)
                )
            )
            margins = np.minimum(q - limit_np[0], limit_np[1] - q)
            metrics: dict[str, float | int] = {
                "collision_waypoint_count": int((~feasible.bool()).sum().item()),
                "maximum_world_collision_cost": float(world_cost.detach().max().cpu().item()),
                "maximum_self_collision_cost": float(self_cost.detach().max().cpu().item()),
                "minimum_joint_limit_margin_rad": float(margins.min()),
                "maximum_fd_velocity_rad_s": float(np.abs(qd).max(initial=0.0)),
                "maximum_fd_acceleration_rad_s2": float(np.abs(qdd).max(initial=0.0)),
                "final_fk_position_error_m": float(
                    np.linalg.norm(ee[-1] - np.asarray(near[:3]))
                ),
                "waypoint_count": int(q.shape[0]),
            }
    finally:
        np.random.set_state(numpy_state)

    if (
        metrics["collision_waypoint_count"] != 0
        or metrics["maximum_world_collision_cost"] != 0.0
        or metrics["maximum_self_collision_cost"] != 0.0
        or metrics["minimum_joint_limit_margin_rad"] <= 0.0
        or metrics["maximum_fd_velocity_rad_s"]
        > EXISTING_HARD_VELOCITY_LIMIT_RAD_S + 1.0e-5
        or metrics["maximum_fd_acceleration_rad_s2"]
        > EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2 + 1.0e-4
    ):
        raise KeyboardV3CuroboPlannerError("KEYBOARD_V3_PLAN_SAFETY_CONTRACT_FAILED")

    return KeyboardV3CuroboPlan(
        request=request,
        condition=condition,
        nominal_grasp_pose_root_m_xyzw=nominal,
        near_grasp_pose_root_m_xyzw=near,
        q_rad=q,
        ee_position_root_m=ee,
        interpolation_dt_s=dt,
        metrics=metrics,
    ).validated()


__all__ = [
    "KEYBOARD_V3_CUROBO_PLAN_SCHEMA",
    "KeyboardV3CuroboPlan",
    "KeyboardV3CuroboPlannerError",
    "nominal_grasp_pose_root_m_xyzw_for_cube",
    "plan_keyboard_v3_near_grasp",
]
