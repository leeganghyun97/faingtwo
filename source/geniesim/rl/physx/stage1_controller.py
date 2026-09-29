# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure NumPy reference controller for the Stage 1 legacy place task.

The functions here isolate the controller math that was embedded in the
ROS/MuJoCo path.  They do not launch or import a simulator.  The conventions
match the checked-out implementation:

* the legacy 7-D action is ``delta xyz, delta intrinsic-XYZ RPY, gripper``;
* the Stage 2 action is ``base-frame delta xyz, local rotation vector,
  gripper`` and is applied relative to the measured EE pose;
* every action component is clipped to ``[-1, 1]`` before use;
* xyz and RPY commands accumulate around the legacy right-EE reset pose;
* quaternion arrays use ``(w, x, y, z)``;
* the orientation error is ``sign(q_err.w) * 2 * q_err.xyz``;
* DLS is ``J.T @ solve(J @ J.T + damping**2 I, error)``;
* joint targets are explicitly clamped to finite limits.

The DLS helper performs one update because a new 6x7 Jacobian must be supplied
after each kinematics update.  A simulator adapter can call it repeatedly to
match the legacy ``ik_max_iter=10`` loop without hiding stale-Jacobian reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LegacyCartesianControllerConfig:
    """Checked-out ``PlaceWorkpieceEnv`` Cartesian action constants."""

    reset_position: tuple[float, float, float] = (0.4833, 0.0051, 1.2548)
    reset_rpy_xyz: tuple[float, float, float] = (2.5633, 0.0261, 1.5791)
    position_scale: float = 0.015
    rpy_scale: float = 0.05
    position_half_range: float = 0.15
    rpy_half_range: float = 0.2
    ik_damping: float = 0.05
    ik_max_iterations: int = 10

    def __post_init__(self) -> None:
        reset_position = np.asarray(self.reset_position, dtype=np.float32)
        reset_rpy = np.asarray(self.reset_rpy_xyz, dtype=np.float32)
        if reset_position.shape != (3,):
            raise ValueError("reset_position must contain 3 values")
        if reset_rpy.shape != (3,):
            raise ValueError("reset_rpy_xyz must contain 3 values")
        if not np.all(np.isfinite(reset_position)) or not np.all(
            np.isfinite(reset_rpy)
        ):
            raise ValueError("reset pose values must be finite")
        scalars = np.asarray(
            [
                self.position_scale,
                self.rpy_scale,
                self.position_half_range,
                self.rpy_half_range,
                self.ik_damping,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("controller constants must be finite")
        if self.position_scale < 0.0 or self.rpy_scale < 0.0:
            raise ValueError("action scales cannot be negative")
        if self.position_half_range < 0.0 or self.rpy_half_range < 0.0:
            raise ValueError("safety half-ranges cannot be negative")
        if self.ik_damping < 0.0:
            raise ValueError("ik_damping cannot be negative")
        if self.ik_max_iterations <= 0:
            raise ValueError("ik_max_iterations must be positive")


class LegacyDeltaActionAccumulator:
    """Vectorized copy of the legacy 7-D delta-to-absolute action mapping."""

    def __init__(
        self,
        num_envs: int = 1,
        *,
        config: LegacyCartesianControllerConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.config = config or LegacyCartesianControllerConfig()
        self._reset_pose = np.concatenate(
            [
                np.asarray(self.config.reset_position, dtype=np.float32),
                np.asarray(self.config.reset_rpy_xyz, dtype=np.float32),
            ]
        )
        half_range = np.asarray(
            [
                self.config.position_half_range,
                self.config.position_half_range,
                self.config.position_half_range,
                self.config.rpy_half_range,
                self.config.rpy_half_range,
                self.config.rpy_half_range,
            ],
            dtype=np.float32,
        )
        self._safety_low = self._reset_pose - half_range
        self._safety_high = self._reset_pose + half_range
        self._target = np.tile(self._reset_pose, (self.num_envs, 1)).astype(
            np.float32, copy=False
        )

    @property
    def reset_pose_rpy(self) -> np.ndarray:
        return self._reset_pose.copy()

    @property
    def safety_low(self) -> np.ndarray:
        return self._safety_low.copy()

    @property
    def safety_high(self) -> np.ndarray:
        return self._safety_high.copy()

    @property
    def target_pose_rpy(self) -> np.ndarray:
        return self._target.copy()

    def reset(
        self,
        env_ids: Sequence[int] | np.ndarray | None = None,
        *,
        target_pose_rpy: np.ndarray | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Reset selected accumulators to the legacy pose or supplied reset pose.

        ``target_pose_rpy`` exists to accept the legacy randomized reset sample.
        It is clipped to the same absolute safety box before becoming the new
        accumulator state.
        """

        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.intp)
        else:
            ids = np.asarray(env_ids, dtype=np.intp)
            if ids.ndim == 0:
                ids = ids.reshape(1)
            if ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if np.any(ids < 0) or np.any(ids >= self.num_envs):
                raise IndexError("env_ids contains an out-of-range environment")

        if target_pose_rpy is None:
            targets = np.tile(self._reset_pose, (len(ids), 1))
        else:
            targets = np.asarray(target_pose_rpy, dtype=np.float32)
            if targets.ndim == 1 and len(ids) == 1:
                targets = targets.reshape(1, -1)
            expected = (len(ids), 6)
            if targets.shape != expected:
                raise ValueError(
                    f"target_pose_rpy must have shape {expected}, got {targets.shape}"
                )
            if not np.all(np.isfinite(targets)):
                raise ValueError("target_pose_rpy must contain only finite values")
        self._target[ids] = np.clip(targets, self._safety_low, self._safety_high)
        return self._target[ids].copy()

    def apply(self, actions: np.ndarray | Sequence[float]) -> np.ndarray:
        """Accumulate clipped actions and return absolute pose plus gripper."""

        action = np.asarray(actions, dtype=np.float32)
        if action.ndim == 1 and self.num_envs == 1:
            action = action.reshape(1, -1)
        expected = (self.num_envs, 7)
        if action.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("actions must contain only finite values")

        clipped = np.clip(action, np.float32(-1.0), np.float32(1.0))
        self._target[:, :3] += clipped[:, :3] * np.float32(
            self.config.position_scale
        )
        self._target[:, 3:6] += clipped[:, 3:6] * np.float32(
            self.config.rpy_scale
        )
        self._target[:] = np.clip(
            self._target, self._safety_low, self._safety_high
        )
        return np.concatenate([self._target.copy(), clipped[:, 6:7]], axis=-1)


@dataclass(frozen=True)
class MeasuredSe3DeltaTarget:
    """One measured-pose-relative Stage 2 Cartesian command.

    ``target_pose_rpy`` is the six-value absolute command consumed by the
    existing DLS controller.  ``target_pose_wxyz`` records the equivalent,
    unambiguous quaternion pose for manifests and traces.
    """

    target_pose_rpy: np.ndarray
    target_pose_wxyz: np.ndarray
    clipped_action: np.ndarray
    requested_delta_se3: np.ndarray
    effective_delta_se3: np.ndarray
    safety_clamped: bool


def rotation_vector_to_quaternion_wxyz(
    rotation_vector: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Map an axis-angle rotation vector to a normalized wxyz quaternion."""

    vector = _finite_vector("rotation_vector", rotation_vector, 3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        quaternion = np.concatenate([[1.0], 0.5 * vector])
    else:
        half_angle = 0.5 * angle
        quaternion = np.concatenate(
            [[math.cos(half_angle)], vector * (math.sin(half_angle) / angle)]
        )
    return _normalized_quaternion("rotation_vector_quaternion", quaternion)


def quaternion_wxyz_to_rotation_vector(
    quaternion_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Map a wxyz quaternion to its shortest axis-angle rotation vector."""

    quaternion = _normalized_quaternion("quaternion_wxyz", quaternion_wxyz)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:4]))
    if vector_norm < 1.0e-12:
        return (2.0 * quaternion[1:4]).astype(np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
    return quaternion[1:4] * (angle / vector_norm)


def measured_se3_delta_target(
    current_pose_base_wxyz: np.ndarray | Sequence[float],
    action: np.ndarray | Sequence[float],
    *,
    config: LegacyCartesianControllerConfig | None = None,
) -> MeasuredSe3DeltaTarget:
    """Build a Markov Stage 2 target from the currently measured EE pose.

    Translation deltas are expressed in the robot base frame.  Rotation is a
    local/body-frame rotation vector, so its quaternion is post-multiplied onto
    the measured EE quaternion.  The final absolute pose remains inside the
    audited legacy workspace/orientation safety box.
    """

    controller = config or LegacyCartesianControllerConfig()
    current = _finite_vector("current_pose_base_wxyz", current_pose_base_wxyz, 7)
    current_quaternion = _normalized_quaternion(
        "current_pose_base_wxyz quaternion", current[3:7]
    )
    raw_action = _finite_vector("action", action, 7)
    clipped = np.clip(raw_action, -1.0, 1.0)

    target_position = current[:3] + clipped[:3] * controller.position_scale
    local_rotation_vector = clipped[3:6] * controller.rpy_scale
    delta_quaternion = rotation_vector_to_quaternion_wxyz(local_rotation_vector)
    target_quaternion = _normalized_quaternion(
        "target quaternion",
        quaternion_multiply_wxyz(current_quaternion, delta_quaternion),
    )
    target_rpy = quaternion_wxyz_to_euler_xyz(target_quaternion)
    reset_rpy = np.asarray(controller.reset_rpy_xyz, dtype=np.float64)
    target_rpy = reset_rpy + (target_rpy - reset_rpy + np.pi) % (
        2.0 * np.pi
    ) - np.pi

    reset_pose = np.concatenate(
        [
            np.asarray(controller.reset_position, dtype=np.float64),
            reset_rpy,
        ]
    )
    half_range = np.asarray(
        [controller.position_half_range] * 3 + [controller.rpy_half_range] * 3,
        dtype=np.float64,
    )
    unconstrained_target_pose_rpy = np.concatenate([target_position, target_rpy])
    target_pose_rpy = np.clip(
        unconstrained_target_pose_rpy,
        reset_pose - half_range,
        reset_pose + half_range,
    )
    target_pose_wxyz = np.concatenate(
        [target_pose_rpy[:3], euler_xyz_to_quaternion_wxyz(target_pose_rpy[3:6])]
    )
    final_quaternion = target_pose_wxyz[3:7]
    current_inverse = current_quaternion.copy()
    current_inverse[1:4] *= -1.0
    effective_local_quaternion = quaternion_multiply_wxyz(
        current_inverse, final_quaternion
    )
    requested_delta = np.concatenate(
        [clipped[:3] * controller.position_scale, local_rotation_vector]
    )
    effective_delta = np.concatenate(
        [
            target_pose_wxyz[:3] - current[:3],
            quaternion_wxyz_to_rotation_vector(effective_local_quaternion),
        ]
    )
    return MeasuredSe3DeltaTarget(
        target_pose_rpy=target_pose_rpy.astype(np.float32),
        target_pose_wxyz=target_pose_wxyz.astype(np.float32),
        clipped_action=clipped.astype(np.float32),
        requested_delta_se3=requested_delta.astype(np.float32),
        effective_delta_se3=effective_delta.astype(np.float32),
        safety_clamped=not np.allclose(
            target_pose_rpy,
            unconstrained_target_pose_rpy,
            rtol=0.0,
            atol=1.0e-9,
        ),
    )


def euler_xyz_to_matrix(rpy_xyz: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles to a rotation matrix.

    This is the same ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` expansion used by the
    legacy MuJoCo node.  A single ``(3,)`` value or a batch ``(..., 3)`` is
    accepted.
    """

    rpy = _finite_array("rpy_xyz", rpy_xyz, last_dimension=3)
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    matrix = np.empty(rpy.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cos_yaw * cos_pitch
    matrix[..., 0, 1] = -sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll
    matrix[..., 0, 2] = sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll
    matrix[..., 1, 0] = sin_yaw * cos_pitch
    matrix[..., 1, 1] = cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll
    matrix[..., 1, 2] = -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll
    matrix[..., 2, 0] = -sin_pitch
    matrix[..., 2, 1] = cos_pitch * sin_roll
    matrix[..., 2, 2] = cos_pitch * cos_roll
    return matrix


def matrix_to_euler_xyz(matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Extract intrinsic XYZ Euler angles with the legacy gimbal-lock branch."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"matrix must end in shape (3, 3), got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("matrix must contain only finite values")

    flat = rotation.reshape((-1, 3, 3))
    result = np.empty((flat.shape[0], 3), dtype=np.float64)
    for index, value in enumerate(flat):
        pitch = math.asin(max(-1.0, min(1.0, -float(value[2, 0]))))
        if abs(float(value[2, 0])) < 0.9999:
            roll = math.atan2(float(value[2, 1]), float(value[2, 2]))
            yaw = math.atan2(float(value[1, 0]), float(value[0, 0]))
        else:
            roll = math.atan2(-float(value[1, 2]), float(value[1, 1]))
            yaw = 0.0
        result[index] = (roll, pitch, yaw)
    return result.reshape(rotation.shape[:-2] + (3,))


def quaternion_multiply_wxyz(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Hamilton product of wxyz quaternions, with NumPy broadcasting."""

    left = _finite_array("first", first, last_dimension=4)
    right = _finite_array("second", second, last_dimension=4)
    left, right = np.broadcast_arrays(left, right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def quaternion_wxyz_to_matrix(
    quaternion_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Convert one or more non-zero wxyz quaternions to rotation matrices."""

    quaternion = _normalized_quaternion("quaternion_wxyz", quaternion_wxyz)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def matrix_to_quaternion_wxyz(
    matrix: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Convert rotation matrices to normalized wxyz quaternions.

    The largest-diagonal branches avoid the numerical cancellation of a
    trace-only conversion near 180-degree rotations.
    """

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"matrix must end in shape (3, 3), got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("matrix must contain only finite values")

    flat = rotation.reshape((-1, 3, 3))
    result = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, value in enumerate(flat):
        trace = float(np.trace(value))
        if trace > 0.0:
            scale = 2.0 * math.sqrt(max(0.0, trace + 1.0))
            quaternion = np.array(
                [
                    0.25 * scale,
                    (value[2, 1] - value[1, 2]) / scale,
                    (value[0, 2] - value[2, 0]) / scale,
                    (value[1, 0] - value[0, 1]) / scale,
                ]
            )
        elif value[0, 0] > value[1, 1] and value[0, 0] > value[2, 2]:
            scale = 2.0 * math.sqrt(
                max(0.0, 1.0 + value[0, 0] - value[1, 1] - value[2, 2])
            )
            quaternion = np.array(
                [
                    (value[2, 1] - value[1, 2]) / scale,
                    0.25 * scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                ]
            )
        elif value[1, 1] > value[2, 2]:
            scale = 2.0 * math.sqrt(
                max(0.0, 1.0 + value[1, 1] - value[0, 0] - value[2, 2])
            )
            quaternion = np.array(
                [
                    (value[0, 2] - value[2, 0]) / scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    0.25 * scale,
                    (value[1, 2] + value[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(
                max(0.0, 1.0 + value[2, 2] - value[0, 0] - value[1, 1])
            )
            quaternion = np.array(
                [
                    (value[1, 0] - value[0, 1]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("matrix does not yield a finite non-zero quaternion")
        quaternion /= norm
        # Canonicalize for deterministic tests; q and -q represent the same R.
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        result[index] = quaternion
    return result.reshape(rotation.shape[:-2] + (4,))


def euler_xyz_to_quaternion_wxyz(
    rpy_xyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles to normalized wxyz quaternions."""

    return matrix_to_quaternion_wxyz(euler_xyz_to_matrix(rpy_xyz))


def quaternion_wxyz_to_euler_xyz(
    quaternion_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Convert wxyz quaternions to intrinsic XYZ Euler angles."""

    return matrix_to_euler_xyz(quaternion_wxyz_to_matrix(quaternion_wxyz))


def quaternion_error_vector_wxyz(
    target_quaternion_wxyz: np.ndarray | Sequence[float],
    current_quaternion_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the legacy shortest-sign quaternion error approximation.

    Inputs are normalized before multiplication because simulator quaternions
    are unit quaternions.  The output is the vector part multiplied by two,
    not the exact angle-axis logarithm.
    """

    target = _normalized_quaternion("target_quaternion_wxyz", target_quaternion_wxyz)
    current = _normalized_quaternion(
        "current_quaternion_wxyz", current_quaternion_wxyz
    )
    conjugate_current = current.copy()
    conjugate_current[..., 1:4] *= -1.0
    error = quaternion_multiply_wxyz(target, conjugate_current)
    sign = np.where(error[..., 0:1] >= 0.0, 1.0, -1.0)
    return sign * 2.0 * error[..., 1:4]


def cartesian_pose_error(
    *,
    target_position: np.ndarray | Sequence[float],
    target_quaternion_wxyz: np.ndarray | Sequence[float],
    current_position: np.ndarray | Sequence[float],
    current_quaternion_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Build the legacy 6-D ``[position, quaternion-vector]`` IK error."""

    target_pos = _finite_array("target_position", target_position, last_dimension=3)
    current_pos = _finite_array(
        "current_position", current_position, last_dimension=3
    )
    position_error = target_pos - current_pos
    orientation_error = quaternion_error_vector_wxyz(
        target_quaternion_wxyz, current_quaternion_wxyz
    )
    position_error, orientation_error = np.broadcast_arrays(
        position_error, orientation_error
    )
    return np.concatenate([position_error, orientation_error], axis=-1)


@dataclass(frozen=True)
class DampedLeastSquaresUpdate:
    """One 6x7 damped-least-squares right-arm joint update."""

    joint_target: np.ndarray
    joint_delta: np.ndarray
    unclamped_joint_target: np.ndarray
    clamped: np.ndarray
    delta_limited: np.ndarray
    residual: np.ndarray


def right_arm_dls_update(
    *,
    joint_position: np.ndarray | Sequence[float],
    jacobian: np.ndarray | Sequence[Sequence[float]],
    cartesian_error: np.ndarray | Sequence[float],
    joint_lower: np.ndarray | Sequence[float],
    joint_upper: np.ndarray | Sequence[float],
    damping: float = 0.05,
    max_joint_delta: float | np.ndarray | Sequence[float] | None = None,
) -> DampedLeastSquaresUpdate:
    """Apply one exact legacy-form DLS update and finite joint-limit clamp.

    Args:
        joint_position: Current seven right-arm joint values.
        jacobian: Current geometric Jacobian with shape ``(6, 7)``.
        cartesian_error: Legacy position/quaternion-vector error, shape ``(6,)``.
        joint_lower: Seven finite lower limits.
        joint_upper: Seven finite upper limits.
        damping: DLS lambda.  The checked-out default is ``0.05``.
        max_joint_delta: Optional positive scalar legacy norm-preserving cap,
            or seven positive per-joint absolute caps.
    """

    q = _finite_vector("joint_position", joint_position, 7)
    error = _finite_vector("cartesian_error", cartesian_error, 6)
    lower = _finite_vector("joint_lower", joint_lower, 7)
    upper = _finite_vector("joint_upper", joint_upper, 7)
    jacobian_array = np.asarray(jacobian, dtype=np.float64)
    if jacobian_array.shape != (6, 7):
        raise ValueError(
            f"jacobian must have shape (6, 7), got {jacobian_array.shape}"
        )
    if not np.all(np.isfinite(jacobian_array)):
        raise ValueError("jacobian must contain only finite values")
    if np.any(lower > upper):
        raise ValueError("each joint lower limit must be <= its upper limit")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    delta_limit: float | np.ndarray | None = None
    if max_joint_delta is not None:
        candidate = np.asarray(max_joint_delta, dtype=np.float64)
        if candidate.ndim == 0:
            delta_limit = float(candidate)
            if not np.isfinite(delta_limit) or delta_limit <= 0.0:
                raise ValueError("max_joint_delta must be finite and positive")
        elif candidate.shape == (7,):
            if not np.all(np.isfinite(candidate)) or np.any(candidate <= 0.0):
                raise ValueError(
                    "per-joint max_joint_delta must contain positive finite values"
                )
            delta_limit = candidate
        else:
            raise ValueError("max_joint_delta must be a scalar or shape (7,)")

    regularized = (
        jacobian_array @ jacobian_array.T
        + float(damping) ** 2 * np.eye(6, dtype=np.float64)
    )
    joint_delta = jacobian_array.T @ np.linalg.solve(regularized, error)
    unconstrained_joint_delta = joint_delta.copy()
    if isinstance(delta_limit, float):
        largest_delta = float(np.max(np.abs(joint_delta)))
        if largest_delta > delta_limit:
            joint_delta *= delta_limit / largest_delta
    elif isinstance(delta_limit, np.ndarray):
        joint_delta = np.clip(joint_delta, -delta_limit, delta_limit)
    delta_limited = joint_delta != unconstrained_joint_delta

    unclamped = q + joint_delta
    clamped_target = np.clip(unclamped, lower, upper)
    clamped = clamped_target != unclamped
    residual = error - jacobian_array @ (clamped_target - q)
    return DampedLeastSquaresUpdate(
        joint_target=clamped_target.astype(np.float32),
        joint_delta=joint_delta,
        unclamped_joint_target=unclamped,
        clamped=clamped,
        delta_limited=delta_limited,
        residual=residual,
    )


def compose_pose_wxyz(
    parent_pose_wxyz: np.ndarray | Sequence[float],
    child_local_pose_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compose a fixed child/payload pose under a moving parent pose.

    Both poses are ``xyz + quaternion(wxyz)``.  The active legacy MJCF uses an
    identity local pose for ``workpiece_r`` under ``gripper_r_base_link``;
    passing ``[0, 0, 0, 1, 0, 0, 0]`` therefore returns the gripper pose.
    """

    parent = _finite_array("parent_pose_wxyz", parent_pose_wxyz, last_dimension=7)
    child = _finite_array(
        "child_local_pose_wxyz", child_local_pose_wxyz, last_dimension=7
    )
    parent, child = np.broadcast_arrays(parent, child)
    parent_quaternion = _normalized_quaternion(
        "parent quaternion", parent[..., 3:7]
    )
    child_quaternion = _normalized_quaternion(
        "child quaternion", child[..., 3:7]
    )
    child_world_position = (
        np.einsum(
            "...ij,...j->...i",
            quaternion_wxyz_to_matrix(parent_quaternion),
            child[..., :3],
        )
        + parent[..., :3]
    )
    child_world_quaternion = quaternion_multiply_wxyz(
        parent_quaternion, child_quaternion
    )
    child_world_quaternion /= np.linalg.norm(
        child_world_quaternion, axis=-1, keepdims=True
    )
    return np.concatenate(
        [child_world_position, child_world_quaternion], axis=-1
    )


def relative_pose_wxyz(
    parent_pose_wxyz: np.ndarray | Sequence[float],
    child_world_pose_wxyz: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Express a world pose in ``parent_pose_wxyz`` coordinates.

    This is the inverse operation of :func:`compose_pose_wxyz`.  Keeping it in
    this simulator-independent module lets the Stage 1 base-link action and
    observation frame contract be tested without launching Isaac Sim.
    """

    parent = _finite_array("parent_pose_wxyz", parent_pose_wxyz, last_dimension=7)
    child = _finite_array(
        "child_world_pose_wxyz", child_world_pose_wxyz, last_dimension=7
    )
    parent, child = np.broadcast_arrays(parent, child)
    parent_quaternion = _normalized_quaternion(
        "parent quaternion", parent[..., 3:7]
    )
    child_quaternion = _normalized_quaternion(
        "child quaternion", child[..., 3:7]
    )
    parent_rotation = quaternion_wxyz_to_matrix(parent_quaternion)
    child_local_position = np.einsum(
        "...ji,...j->...i",
        parent_rotation,
        child[..., :3] - parent[..., :3],
    )
    parent_conjugate = parent_quaternion.copy()
    parent_conjugate[..., 1:4] *= -1.0
    child_local_quaternion = quaternion_multiply_wxyz(
        parent_conjugate, child_quaternion
    )
    child_local_quaternion /= np.linalg.norm(
        child_local_quaternion, axis=-1, keepdims=True
    )
    return np.concatenate(
        [child_local_position, child_local_quaternion], axis=-1
    )


def legacy_place_gripper_target(
    gripper_action: np.ndarray | Sequence[float] | float,
) -> np.ndarray:
    """Map the legacy Place gripper command to its MuJoCo ctrl range.

    The checked-in wrapper passes the clipped policy action through without a
    scale or offset.  MuJoCo then clamps the actuator control to ``[0, 0.024]``
    metres.  PhysX must apply that clamp explicitly.
    """

    action = np.asarray(gripper_action, dtype=np.float32)
    if not np.all(np.isfinite(action)):
        raise ValueError("gripper_action must contain only finite values")
    return np.clip(action, np.float32(0.0), np.float32(0.024))


def _finite_array(
    name: str,
    values: np.ndarray | Sequence[float],
    *,
    last_dimension: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != last_dimension:
        raise ValueError(
            f"{name} must end in dimension {last_dimension}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_vector(
    name: str, values: np.ndarray | Sequence[float], length: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalized_quaternion(
    name: str, values: np.ndarray | Sequence[float]
) -> np.ndarray:
    quaternion = _finite_array(name, values, last_dimension=4)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError(f"{name} must contain non-zero quaternions")
    return quaternion / norm


__all__ = [
    "DampedLeastSquaresUpdate",
    "LegacyCartesianControllerConfig",
    "LegacyDeltaActionAccumulator",
    "cartesian_pose_error",
    "compose_pose_wxyz",
    "euler_xyz_to_matrix",
    "euler_xyz_to_quaternion_wxyz",
    "legacy_place_gripper_target",
    "matrix_to_euler_xyz",
    "matrix_to_quaternion_wxyz",
    "quaternion_error_vector_wxyz",
    "quaternion_multiply_wxyz",
    "quaternion_wxyz_to_euler_xyz",
    "quaternion_wxyz_to_matrix",
    "relative_pose_wxyz",
    "right_arm_dls_update",
]
