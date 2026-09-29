# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Source-owned cuRobo REACH/PREGRASP replanning for contact-free collection.

This module has two deliberately separate responsibilities:

* it may use the *runtime cube root pose* to construct a planner-only target;
* it never constructs an actor observation, replay row, gripper command, or
  ``env.step`` packet.

The latter separation prevents privileged cube GT needed by a nominal planner
from leaking into the visual policy contract.  Importing this module is safe:
cuRobo/Torch are imported only when :func:`plan_contact_free_reach` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .curobo_collision_world import (
    build_collision_world_dict,
    pregrasp_position_root_m_for_cube,
    validated_cube_center_root_m,
)
from .curobo_planner_authority import (
    EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2,
    EXISTING_HARD_VELOCITY_LIMIT_RAD_S,
    FIXED_WRIST_ORIENTATION_XYZW,
    RIGHT_ARM_7DOF_JOINTS,
    build_right_arm_7dof_robot_config,
)


CONTACT_FREE_RUNTIME_REPLAN_SCHEMA = "g2_contact_free_runtime_replan_v1"
MOTIONGEN_POSITION_TOLERANCE_M = 0.003
MOTIONGEN_ROTATION_TOLERANCE_RAD = 0.03
POLICY_DT_S = 0.020


class ContactFreeRuntimeReplanError(RuntimeError):
    """A malformed or unqualified runtime plan; callers must fail closed."""


def _finite_vector(
    value: Sequence[float], *, width: int, label: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ContactFreeRuntimeReplanError(f"{label}_MUST_BE_NUMERIC_SEQUENCE")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ContactFreeRuntimeReplanError(f"{label}_MUST_BE_NUMERIC_SEQUENCE") from error
    if len(result) != width or not all(math.isfinite(component) for component in result):
        raise ContactFreeRuntimeReplanError(f"{label}_MUST_BE_{width}_FINITE_VALUES")
    return result


@dataclass(frozen=True)
class ContactFreeRuntimeReplanRequest:
    """Planner-only state captured after reset and before the first action."""

    cube_center_root_m: Sequence[float]
    right_arm_q_rad: Sequence[float]
    seed: int

    def validated(self) -> "ContactFreeRuntimeReplanRequest":
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ContactFreeRuntimeReplanError("RUNTIME_REPLAN_SEED_MUST_BE_NONNEGATIVE_INT")
        cube = validated_cube_center_root_m(self.cube_center_root_m)
        q = _finite_vector(
            self.right_arm_q_rad,
            width=len(RIGHT_ARM_7DOF_JOINTS),
            label="RUNTIME_REPLAN_RIGHT_ARM_Q_RAD",
        )
        return ContactFreeRuntimeReplanRequest(cube, q, self.seed)


@dataclass(frozen=True)
class ContactFreeRuntimeReplan:
    """Qualified planner result, intentionally separate from actor data."""

    request: ContactFreeRuntimeReplanRequest
    pregrasp_position_root_m: tuple[float, float, float]
    q_rad: np.ndarray
    ee_position_root_m: np.ndarray
    interpolation_dt_s: float
    metrics: dict[str, float | int]

    def validated(self) -> "ContactFreeRuntimeReplan":
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
            raise ContactFreeRuntimeReplanError("RUNTIME_REPLAN_PATH_SHAPE_OR_FINITE_INVALID")
        if not math.isfinite(float(self.interpolation_dt_s)) or self.interpolation_dt_s <= 0.0:
            raise ContactFreeRuntimeReplanError("RUNTIME_REPLAN_DT_INVALID")
        pregrasp = _finite_vector(
            self.pregrasp_position_root_m,
            width=3,
            label="RUNTIME_REPLAN_PREGRASP_ROOT_M",
        )
        expected_pregrasp = pregrasp_position_root_m_for_cube(request.cube_center_root_m)
        if not np.allclose(pregrasp, expected_pregrasp, rtol=0.0, atol=1.0e-12):
            raise ContactFreeRuntimeReplanError("RUNTIME_REPLAN_CUBE_PREGRASP_BINDING_MISMATCH")
        if float(np.linalg.norm(ee[-1] - np.asarray(pregrasp))) > MOTIONGEN_POSITION_TOLERANCE_M:
            raise ContactFreeRuntimeReplanError("RUNTIME_REPLAN_FINAL_FK_OUTSIDE_SOURCE_TOLERANCE")
        return self

    def planner_only_receipt(self) -> dict[str, Any]:
        """Serialize plan provenance without exposing it as an actor feature."""

        self.validated()
        return {
            "schema": CONTACT_FREE_RUNTIME_REPLAN_SCHEMA,
            "planner_input_cube_center_root_m": list(
                self.request.validated().cube_center_root_m
            ),
            "planner_input_right_arm_q_rad": list(
                self.request.validated().right_arm_q_rad
            ),
            "planner_seed": self.request.seed,
            "pregrasp_position_root_m": list(self.pregrasp_position_root_m),
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


def _joint_margin(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.minimum(q - lower, upper - q).min())


def plan_contact_free_reach(
    request: ContactFreeRuntimeReplanRequest,
) -> ContactFreeRuntimeReplan:
    """Plan contact-free right-arm REACH/PREGRASP after a reset observation.

    The returned plan is not executable by itself.  A separate live runner
    must still bind its cube pose to this receipt, preserve exact-4D packet
    consumption, apply the existing 4.5-mm limiter, and stop OPEN at the
    pre-contact handoff.  Contact/CLOSE are deliberately out of scope.
    """

    request = request.validated()
    # Lazy imports keep all static contract/audit paths CUDA- and Isaac-free.
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

    # Planning must not perturb the simulator's global stochastic state after
    # reset.  cuRobo's random search receives the requested seed inside an
    # isolated Torch RNG scope; NumPy state is restored in the finally block.
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
            pregrasp = pregrasp_position_root_m_for_cube(request.cube_center_root_m)
            orientation_wxyz = (
                FIXED_WRIST_ORIENTATION_XYZW[3],
                FIXED_WRIST_ORIENTATION_XYZW[0],
                FIXED_WRIST_ORIENTATION_XYZW[1],
                FIXED_WRIST_ORIENTATION_XYZW[2],
            )
            goal_pose = Pose.from_list([*pregrasp, *orientation_wxyz], tensor_args)
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
            plan_result = motion_gen.plan_single(
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
            if not _result_success(plan_result.success):
                raise ContactFreeRuntimeReplanError(
                    f"RUNTIME_REPLAN_MOTIONGEN_FAILED:{plan_result.status}"
                )
            trajectory = plan_result.get_interpolated_plan()
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
            dt = float(plan_result.interpolation_dt)
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
            metrics: dict[str, float | int] = {
                "collision_waypoint_count": int((~feasible.bool()).sum().item()),
                "maximum_world_collision_cost": float(world_cost.detach().max().cpu().item()),
                "maximum_self_collision_cost": float(self_cost.detach().max().cpu().item()),
                "minimum_joint_limit_margin_rad": min(
                    _joint_margin(row, limit_np[0], limit_np[1]) for row in q
                ),
                "maximum_fd_velocity_rad_s": float(np.abs(qd).max(initial=0.0)),
                "maximum_fd_acceleration_rad_s2": float(
                    np.abs(qdd).max(initial=0.0)
                ),
                "final_fk_position_error_m": float(
                    np.linalg.norm(ee[-1] - np.asarray(pregrasp))
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
        raise ContactFreeRuntimeReplanError("RUNTIME_REPLAN_SAFETY_CONTRACT_FAILED")
    return ContactFreeRuntimeReplan(
        request=request,
        pregrasp_position_root_m=pregrasp,
        q_rad=q,
        ee_position_root_m=ee,
        interpolation_dt_s=dt,
        metrics=metrics,
    ).validated()


__all__ = [
    "CONTACT_FREE_RUNTIME_REPLAN_SCHEMA",
    "ContactFreeRuntimeReplan",
    "ContactFreeRuntimeReplanError",
    "ContactFreeRuntimeReplanRequest",
    "MOTIONGEN_POSITION_TOLERANCE_M",
    "MOTIONGEN_ROTATION_TOLERANCE_RAD",
    "POLICY_DT_S",
    "plan_contact_free_reach",
]
