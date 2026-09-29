# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ROS-free building blocks for the GenieSim PhysX reinforcement-learning path."""

from .stage1_controller import (
    DampedLeastSquaresUpdate,
    LegacyCartesianControllerConfig,
    LegacyDeltaActionAccumulator,
    MeasuredSe3DeltaTarget,
    cartesian_pose_error,
    compose_pose_wxyz,
    euler_xyz_to_matrix,
    euler_xyz_to_quaternion_wxyz,
    legacy_place_gripper_target,
    matrix_to_euler_xyz,
    matrix_to_quaternion_wxyz,
    measured_se3_delta_target,
    quaternion_error_vector_wxyz,
    quaternion_multiply_wxyz,
    quaternion_wxyz_to_rotation_vector,
    quaternion_wxyz_to_euler_xyz,
    quaternion_wxyz_to_matrix,
    relative_pose_wxyz,
    rotation_vector_to_quaternion_wxyz,
    right_arm_dls_update,
)
from .stage1_place_task import (
    LegacyPlaceTaskConfig,
    RewardOutputMode,
    Stage1PlaceStep,
    Stage1PlaceTaskOracle,
)

# ``stage1_place_env`` is intentionally not imported here: its public class
# requires SimulationApp to have started before Omniverse modules are loaded.

__all__ = [
    "DampedLeastSquaresUpdate",
    "LegacyCartesianControllerConfig",
    "LegacyDeltaActionAccumulator",
    "MeasuredSe3DeltaTarget",
    "LegacyPlaceTaskConfig",
    "RewardOutputMode",
    "Stage1PlaceStep",
    "Stage1PlaceTaskOracle",
    "cartesian_pose_error",
    "compose_pose_wxyz",
    "euler_xyz_to_matrix",
    "euler_xyz_to_quaternion_wxyz",
    "legacy_place_gripper_target",
    "matrix_to_euler_xyz",
    "matrix_to_quaternion_wxyz",
    "measured_se3_delta_target",
    "quaternion_error_vector_wxyz",
    "quaternion_multiply_wxyz",
    "quaternion_wxyz_to_rotation_vector",
    "quaternion_wxyz_to_euler_xyz",
    "quaternion_wxyz_to_matrix",
    "relative_pose_wxyz",
    "rotation_vector_to_quaternion_wxyz",
    "right_arm_dls_update",
]
