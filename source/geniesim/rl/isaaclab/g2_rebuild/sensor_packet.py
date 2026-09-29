# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac-independent sensor packet and hard-stop integrity validation for G2.

This module is intentionally limited to canonical tensors.  Isaac sensor
objects are adapted into these dataclasses at the acquisition boundary; the
policy and dataset code never need to inspect private Isaac buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .data_contract import (
    CoordinateFrame,
    G2_RUNTIME_JOINT_ORDER,
    SchemaVersion,
    TensorSource,
)


class SensorFaultCode(str, Enum):
    """Machine-readable reasons for a sensor-integrity hard stop."""

    MISSING_FRAME = "missing_frame"
    INVALID_TIMESTAMP = "invalid_timestamp"
    CADENCE_VIOLATION = "cadence_violation"
    CAMERA_SKEW = "camera_skew"
    STALE_FRAME = "stale_frame"
    CORRUPT_RGB = "corrupt_rgb"
    CORRUPT_DEPTH = "corrupt_depth"
    RGB_DEPTH_MISALIGNMENT = "rgb_depth_misalignment"
    INVALID_TF = "invalid_tf"
    INVALID_FK = "invalid_fk"
    INVALID_ROBOT_STATE = "invalid_robot_state"
    INVALID_CONTACT = "invalid_contact"


class SensorIntegrityError(RuntimeError):
    """Raised when a packet is unsafe to record or pass to a policy."""

    def __init__(
        self,
        code: SensorFaultCode,
        field: str,
        detail: str,
        *,
        environment_indices: tuple[int, ...] = (),
    ) -> None:
        self.code = code
        self.field = field
        self.detail = detail
        self.environment_indices = environment_indices
        suffix = (
            f"; environments={list(environment_indices)}"
            if environment_indices
            else ""
        )
        super().__init__(f"{code.value}: {field}: {detail}{suffix}")


@dataclass(frozen=True)
class CameraFrame:
    """One synchronized RGB-D stream sample.

    ``depth_valid`` makes legitimate missing returns explicit.  Invalid depth
    pixels must be stored as finite zeros; NaN/Inf is treated as corruption,
    not occlusion.
    """

    rgb: torch.Tensor
    depth_m: torch.Tensor
    depth_valid: torch.Tensor
    timestamp_s: torch.Tensor
    sequence_id: torch.Tensor
    optical_pose_robot_root_xyzw: torch.Tensor
    present: torch.Tensor
    optical_frame: str
    source: str


@dataclass(frozen=True)
class RobotStateFrame:
    """Canonical robot/FK sample expressed in the world frame where named."""

    joint_position_rad: torch.Tensor
    joint_velocity_rad_s: torch.Tensor
    root_position_world_m: torch.Tensor
    root_quaternion_world_xyzw: torch.Tensor
    root_linear_velocity_world_m_s: torch.Tensor
    root_angular_velocity_world_rad_s: torch.Tensor
    ee_position_world_m: torch.Tensor
    ee_quaternion_world_xyzw: torch.Tensor
    ee_linear_velocity_world_m_s: torch.Tensor
    distal_inner_link_origin_world_m: torch.Tensor
    distal_outer_link_origin_world_m: torch.Tensor
    gripper_state: torch.Tensor
    timestamp_s: torch.Tensor
    source: str


@dataclass(frozen=True)
class ContactFrame:
    """Privileged exact contact sample for labels and validation only.

    The two columns are the *right gripper inner and outer fingers*, not the
    robot's left and right hands.  Force is optional, but measurement validity
    is mandatory even when force is unavailable.
    """

    inner_outer_contact: torch.Tensor
    measurement_valid: torch.Tensor
    timestamp_s: torch.Tensor
    source: str
    inner_outer_force_n: torch.Tensor | None = None


@dataclass(frozen=True)
class CalibratedPadSurfaceMidpoint:
    """Live-FK pad-surface midpoint backed by an identified calibration.

    Acquisition code may construct this only after applying validated local
    inner/outer pad-surface offsets to the corresponding live link poses.
    Supplying the uncalibrated link-origin midpoint here is a contract breach.
    """

    position_robot_root_m: torch.Tensor
    velocity_robot_root_m_s: torch.Tensor
    timestamp_s: torch.Tensor
    valid: torch.Tensor
    calibration_id: str
    source: TensorSource = TensorSource.LIVE_FK


@dataclass(frozen=True)
class SynchronizedSensorPacket:
    """All control-step inputs sharing one episode-local time base."""

    timestamp_s: torch.Tensor
    control_step: torch.Tensor
    head: CameraFrame
    wrist: CameraFrame
    robot: RobotStateFrame
    contact: ContactFrame | None = None
    distal_pad_midpoint: CalibratedPadSurfaceMidpoint | None = None
    schema_version: SchemaVersion = SchemaVersion.CONTROLLED_REBUILD_V2


@dataclass(frozen=True)
class DeployableSensorView:
    """Validated packet projection that structurally excludes exact contact."""

    timestamp_s: torch.Tensor
    control_step: torch.Tensor
    head: CameraFrame
    wrist: CameraFrame
    robot: RobotStateFrame
    distal_pad_midpoint: CalibratedPadSurfaceMidpoint | None
    schema_version: SchemaVersion


@dataclass(frozen=True)
class SensorValidationConfig:
    """Strict canonical rates and physical validation bounds."""

    control_period_s: float = 0.02
    # The RTX cameras capture every two 2 ms physics steps (250 Hz), while the
    # policy samples the newest completed frame only at its 50 Hz boundary.
    # These are deliberately separate clocks: multiplying a capture sequence
    # delta by the policy period rejects every otherwise-correct live packet.
    camera_capture_period_s: float = 0.004
    camera_policy_sample_period_s: float = 0.02
    robot_period_s: float = 0.02
    contact_period_s: float = 0.02
    cadence_tolerance_s: float = 0.0025
    max_camera_age_s: float = 0.03
    max_robot_age_s: float = 0.02
    max_contact_age_s: float = 0.02
    max_head_wrist_skew_s: float = 0.005
    future_tolerance_s: float = 1.0e-6
    min_depth_m: float = 0.02
    max_depth_m: float = 5.0
    min_valid_depth_fraction: float = 0.01
    max_contact_force_n: float = 500.0
    quaternion_norm_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        positive = {
            "control_period_s": self.control_period_s,
            "camera_capture_period_s": self.camera_capture_period_s,
            "camera_policy_sample_period_s": self.camera_policy_sample_period_s,
            "robot_period_s": self.robot_period_s,
            "contact_period_s": self.contact_period_s,
            "max_camera_age_s": self.max_camera_age_s,
            "max_robot_age_s": self.max_robot_age_s,
            "max_contact_age_s": self.max_contact_age_s,
            "max_head_wrist_skew_s": self.max_head_wrist_skew_s,
            "max_depth_m": self.max_depth_m,
            "max_contact_force_n": self.max_contact_force_n,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.cadence_tolerance_s < 0.0 or self.future_tolerance_s < 0.0:
            raise ValueError("timing tolerances must be non-negative")
        if not 0.0 <= self.min_valid_depth_fraction <= 1.0:
            raise ValueError("min_valid_depth_fraction must be in [0, 1]")
        if self.min_depth_m < 0.0 or self.min_depth_m >= self.max_depth_m:
            raise ValueError("depth bounds are invalid")


def _bad_indices(mask: torch.Tensor) -> tuple[int, ...]:
    if mask.ndim == 0:
        return (0,) if bool(mask.item()) else ()
    reduced = mask.reshape(mask.shape[0], -1).any(dim=1)
    return tuple(int(index) for index in torch.nonzero(reduced).flatten().tolist())


def _raise_if(
    condition: torch.Tensor | bool,
    code: SensorFaultCode,
    field: str,
    detail: str,
) -> None:
    if isinstance(condition, bool):
        bad = condition
        indices: tuple[int, ...] = ()
    else:
        bad = bool(torch.any(condition).item())
        indices = _bad_indices(condition) if bad else ()
    if bad:
        raise SensorIntegrityError(
            code, field, detail, environment_indices=indices
        )


def _require_shape(
    value: torch.Tensor,
    expected: tuple[int | None, ...],
    *,
    field: str,
    code: SensorFaultCode,
) -> None:
    matches = value.ndim == len(expected) and all(
        wanted is None or actual == wanted
        for actual, wanted in zip(value.shape, expected, strict=True)
    )
    _raise_if(
        not matches,
        code,
        field,
        f"expected shape {expected}, got {tuple(value.shape)}",
    )


def _require_finite(
    value: torch.Tensor, *, field: str, code: SensorFaultCode
) -> None:
    _raise_if(~torch.isfinite(value), code, field, "contains NaN or Inf")


def _require_batch(value: torch.Tensor, batch_size: int, field: str) -> None:
    _raise_if(
        value.ndim == 0 or value.shape[0] != batch_size,
        SensorFaultCode.INVALID_ROBOT_STATE,
        field,
        f"expected batch dimension {batch_size}, got {tuple(value.shape)}",
    )


def _validate_timestamp(
    sample_time_s: torch.Tensor,
    packet_time_s: torch.Tensor,
    *,
    field: str,
    max_age_s: float,
    future_tolerance_s: float,
) -> None:
    _require_shape(
        sample_time_s,
        (packet_time_s.shape[0],),
        field=field,
        code=SensorFaultCode.INVALID_TIMESTAMP,
    )
    _require_finite(
        sample_time_s, field=field, code=SensorFaultCode.INVALID_TIMESTAMP
    )
    _raise_if(
        sample_time_s < 0.0,
        SensorFaultCode.INVALID_TIMESTAMP,
        field,
        "timestamp must be non-negative episode-local seconds",
    )
    age = packet_time_s - sample_time_s
    _raise_if(
        age < -future_tolerance_s,
        SensorFaultCode.INVALID_TIMESTAMP,
        field,
        "sample timestamp is in the future",
    )
    _raise_if(
        age > max_age_s,
        SensorFaultCode.STALE_FRAME,
        field,
        f"sample age exceeds {max_age_s:g} s",
    )


def _validate_camera(
    frame: CameraFrame,
    *,
    name: str,
    packet_time_s: torch.Tensor,
    config: SensorValidationConfig,
) -> None:
    batch_size = packet_time_s.shape[0]
    _require_shape(
        frame.rgb,
        (batch_size, 192, 256, 3),
        field=f"{name}.rgb",
        code=SensorFaultCode.CORRUPT_RGB,
    )
    _raise_if(
        frame.rgb.dtype != torch.uint8,
        SensorFaultCode.CORRUPT_RGB,
        f"{name}.rgb",
        f"canonical RGB dtype is uint8, got {frame.rgb.dtype}",
    )
    depth = frame.depth_m
    _require_shape(
        depth,
        (batch_size, 192, 256, 1),
        field=f"{name}.depth_m",
        code=SensorFaultCode.RGB_DEPTH_MISALIGNMENT,
    )
    depth_spatial_shape = depth.shape[:3]
    depth_values = depth[..., 0]
    expected_spatial = frame.rgb.shape[:3]
    _raise_if(
        depth_spatial_shape != expected_spatial,
        SensorFaultCode.RGB_DEPTH_MISALIGNMENT,
        f"{name}.depth_m",
        f"RGB spatial shape {expected_spatial} != depth {depth_spatial_shape}",
    )
    valid = frame.depth_valid
    _require_shape(
        valid,
        (batch_size, 192, 256, 1),
        field=f"{name}.depth_valid",
        code=SensorFaultCode.RGB_DEPTH_MISALIGNMENT,
    )
    valid_spatial_shape = valid.shape[:3]
    valid_values = valid[..., 0]
    _raise_if(
        valid_spatial_shape != expected_spatial,
        SensorFaultCode.RGB_DEPTH_MISALIGNMENT,
        f"{name}.depth_valid",
        f"RGB spatial shape {expected_spatial} != validity {valid_spatial_shape}",
    )
    _raise_if(
        valid_values.dtype != torch.bool,
        SensorFaultCode.CORRUPT_DEPTH,
        f"{name}.depth_valid",
        f"canonical depth validity dtype is bool, got {valid_values.dtype}",
    )
    _raise_if(
        depth_values.dtype != torch.float32,
        SensorFaultCode.CORRUPT_DEPTH,
        f"{name}.depth_m",
        f"canonical depth must be float32 meters, got {depth_values.dtype}",
    )
    _require_finite(
        depth_values,
        field=f"{name}.depth_m",
        code=SensorFaultCode.CORRUPT_DEPTH,
    )
    _raise_if(
        valid_values
        & ((depth_values < config.min_depth_m) | (depth_values > config.max_depth_m)),
        SensorFaultCode.CORRUPT_DEPTH,
        f"{name}.depth_m",
        f"valid depth is outside [{config.min_depth_m:g}, {config.max_depth_m:g}] m",
    )
    _raise_if(
        (~valid_values) & (depth_values != 0.0),
        SensorFaultCode.CORRUPT_DEPTH,
        f"{name}.depth_m",
        "invalid depth pixels must be canonical finite zeros",
    )
    valid_fraction = valid_values.to(torch.float32).reshape(batch_size, -1).mean(dim=1)
    _raise_if(
        valid_fraction < config.min_valid_depth_fraction,
        SensorFaultCode.CORRUPT_DEPTH,
        f"{name}.depth_valid",
        "too few valid depth returns; this is sensor corruption, not local occlusion",
    )
    _require_shape(
        frame.present,
        (batch_size,),
        field=f"{name}.present",
        code=SensorFaultCode.MISSING_FRAME,
    )
    _raise_if(
        frame.present.dtype != torch.bool,
        SensorFaultCode.MISSING_FRAME,
        f"{name}.present",
        "present mask must be bool",
    )
    _raise_if(
        ~frame.present,
        SensorFaultCode.MISSING_FRAME,
        f"{name}.present",
        "camera frame is missing",
    )
    _require_shape(
        frame.sequence_id,
        (batch_size,),
        field=f"{name}.sequence_id",
        code=SensorFaultCode.MISSING_FRAME,
    )
    _raise_if(
        frame.sequence_id.dtype != torch.int64,
        SensorFaultCode.MISSING_FRAME,
        f"{name}.sequence_id",
        f"canonical sequence id must be int64, got {frame.sequence_id.dtype}",
    )
    _raise_if(
        frame.sequence_id < 0,
        SensorFaultCode.MISSING_FRAME,
        f"{name}.sequence_id",
        "sequence id must be non-negative",
    )
    _require_shape(
        frame.optical_pose_robot_root_xyzw,
        (batch_size, 7),
        field=f"{name}.optical_pose_robot_root_xyzw",
        code=SensorFaultCode.INVALID_TF,
    )
    _raise_if(
        frame.optical_pose_robot_root_xyzw.dtype != torch.float32,
        SensorFaultCode.INVALID_TF,
        f"{name}.optical_pose_robot_root_xyzw",
        "canonical capture-time optical pose must be float32",
    )
    _require_finite(
        frame.optical_pose_robot_root_xyzw,
        field=f"{name}.optical_pose_robot_root_xyzw",
        code=SensorFaultCode.INVALID_TF,
    )
    pose_quaternion_norm = torch.linalg.vector_norm(
        frame.optical_pose_robot_root_xyzw[..., 3:7], dim=-1
    )
    _raise_if(
        torch.abs(pose_quaternion_norm - 1.0)
        > config.quaternion_norm_tolerance,
        SensorFaultCode.INVALID_TF,
        f"{name}.optical_pose_robot_root_xyzw",
        "capture-time optical quaternion is not unit XYZW",
    )
    expected_optical_frame = {
        "head": CoordinateFrame.HEAD_OPTICAL.value,
        "wrist": CoordinateFrame.RIGHT_WRIST_OPTICAL.value,
    }.get(name)
    _raise_if(
        expected_optical_frame is None or frame.optical_frame != expected_optical_frame,
        SensorFaultCode.INVALID_TF,
        f"{name}.optical_frame",
        f"expected canonical optical frame {expected_optical_frame!r}, got {frame.optical_frame!r}",
    )
    _raise_if(
        not frame.source.strip(),
        SensorFaultCode.INVALID_TF,
        f"{name}.source",
        "sensor source is empty",
    )
    _validate_timestamp(
        frame.timestamp_s,
        packet_time_s,
        field=f"{name}.timestamp_s",
        max_age_s=config.max_camera_age_s,
        future_tolerance_s=config.future_tolerance_s,
    )


def _validate_robot(
    robot: RobotStateFrame,
    *,
    packet_time_s: torch.Tensor,
    config: SensorValidationConfig,
) -> None:
    batch_size = packet_time_s.shape[0]
    fields = {
        "joint_position_rad": robot.joint_position_rad,
        "joint_velocity_rad_s": robot.joint_velocity_rad_s,
        "root_position_world_m": robot.root_position_world_m,
        "root_quaternion_world_xyzw": robot.root_quaternion_world_xyzw,
        "root_linear_velocity_world_m_s": robot.root_linear_velocity_world_m_s,
        "root_angular_velocity_world_rad_s": robot.root_angular_velocity_world_rad_s,
        "ee_position_world_m": robot.ee_position_world_m,
        "ee_quaternion_world_xyzw": robot.ee_quaternion_world_xyzw,
        "ee_linear_velocity_world_m_s": robot.ee_linear_velocity_world_m_s,
        "distal_inner_link_origin_world_m": robot.distal_inner_link_origin_world_m,
        "distal_outer_link_origin_world_m": robot.distal_outer_link_origin_world_m,
        "gripper_state": robot.gripper_state,
    }
    for field, value in fields.items():
        _require_batch(value, batch_size, f"robot.{field}")
        _require_finite(
            value,
            field=f"robot.{field}",
            code=(
                SensorFaultCode.INVALID_FK
                if "link_origin" in field or field.startswith("ee_")
                else SensorFaultCode.INVALID_ROBOT_STATE
            ),
        )
    for field in (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "gripper_state",
    ):
        value = getattr(robot, field)
        _raise_if(
            value.dtype != torch.float32,
            SensorFaultCode.INVALID_ROBOT_STATE,
            f"robot.{field}",
            f"canonical dtype is float32, got {value.dtype}",
        )
    _require_shape(
        robot.joint_position_rad,
        (batch_size, len(G2_RUNTIME_JOINT_ORDER)),
        field="robot.joint_position_rad",
        code=SensorFaultCode.INVALID_ROBOT_STATE,
    )
    _require_shape(
        robot.joint_velocity_rad_s,
        (batch_size, len(G2_RUNTIME_JOINT_ORDER)),
        field="robot.joint_velocity_rad_s",
        code=SensorFaultCode.INVALID_ROBOT_STATE,
    )
    _raise_if(
        robot.joint_position_rad.shape != robot.joint_velocity_rad_s.shape,
        SensorFaultCode.INVALID_ROBOT_STATE,
        "robot.joint_state",
        "position and velocity shapes differ",
    )
    for field in (
        "root_position_world_m",
        "root_linear_velocity_world_m_s",
        "root_angular_velocity_world_rad_s",
        "ee_position_world_m",
        "ee_linear_velocity_world_m_s",
        "distal_inner_link_origin_world_m",
        "distal_outer_link_origin_world_m",
    ):
        _require_shape(
            getattr(robot, field),
            (batch_size, 3),
            field=f"robot.{field}",
            code=(
                SensorFaultCode.INVALID_FK
                if "link_origin" in field or field.startswith("ee_")
                else SensorFaultCode.INVALID_TF
            ),
        )
    _require_shape(
        robot.root_quaternion_world_xyzw,
        (batch_size, 4),
        field="robot.root_quaternion_world_xyzw",
        code=SensorFaultCode.INVALID_TF,
    )
    quaternion_norm = torch.linalg.vector_norm(
        robot.root_quaternion_world_xyzw, dim=-1
    )
    _raise_if(
        torch.abs(quaternion_norm - 1.0) > config.quaternion_norm_tolerance,
        SensorFaultCode.INVALID_TF,
        "robot.root_quaternion_world_xyzw",
        "root quaternion is not unit length",
    )
    _require_shape(
        robot.ee_quaternion_world_xyzw,
        (batch_size, 4),
        field="robot.ee_quaternion_world_xyzw",
        code=SensorFaultCode.INVALID_FK,
    )
    ee_quaternion_norm = torch.linalg.vector_norm(
        robot.ee_quaternion_world_xyzw, dim=-1
    )
    _raise_if(
        torch.abs(ee_quaternion_norm - 1.0) > config.quaternion_norm_tolerance,
        SensorFaultCode.INVALID_FK,
        "robot.ee_quaternion_world_xyzw",
        "EE quaternion is not unit length",
    )
    _require_shape(
        robot.gripper_state,
        (batch_size, 1),
        field="robot.gripper_state",
        code=SensorFaultCode.INVALID_ROBOT_STATE,
    )
    _raise_if(
        not robot.source.strip(),
        SensorFaultCode.INVALID_ROBOT_STATE,
        "robot.source",
        "robot state source is empty",
    )
    _validate_timestamp(
        robot.timestamp_s,
        packet_time_s,
        field="robot.timestamp_s",
        max_age_s=config.max_robot_age_s,
        future_tolerance_s=config.future_tolerance_s,
    )


def _validate_contact(
    contact: ContactFrame,
    *,
    packet_time_s: torch.Tensor,
    config: SensorValidationConfig,
) -> None:
    batch_size = packet_time_s.shape[0]
    _require_shape(
        contact.inner_outer_contact,
        (batch_size, 2),
        field="contact.inner_outer_contact",
        code=SensorFaultCode.INVALID_CONTACT,
    )
    _raise_if(
        contact.inner_outer_contact.dtype != torch.bool,
        SensorFaultCode.INVALID_CONTACT,
        "contact.inner_outer_contact",
        "exact contact state must be bool",
    )
    _require_shape(
        contact.measurement_valid,
        (batch_size, 2),
        field="contact.measurement_valid",
        code=SensorFaultCode.INVALID_CONTACT,
    )
    _raise_if(
        contact.measurement_valid.dtype != torch.bool,
        SensorFaultCode.INVALID_CONTACT,
        "contact.measurement_valid",
        "contact validity must be bool",
    )
    _raise_if(
        ~contact.measurement_valid,
        SensorFaultCode.INVALID_CONTACT,
        "contact.measurement_valid",
        "contact sensor is invalid; zero cannot be assumed",
    )
    if contact.inner_outer_force_n is not None:
        force = contact.inner_outer_force_n
        _require_shape(
            force,
            (batch_size, 2),
            field="contact.inner_outer_force_n",
            code=SensorFaultCode.INVALID_CONTACT,
        )
        _require_finite(
            force,
            field="contact.inner_outer_force_n",
            code=SensorFaultCode.INVALID_CONTACT,
        )
        _raise_if(
            force.dtype != torch.float32,
            SensorFaultCode.INVALID_CONTACT,
            "contact.inner_outer_force_n",
            f"canonical force dtype is float32, got {force.dtype}",
        )
        _raise_if(
            (force < 0.0) | (force > config.max_contact_force_n),
            SensorFaultCode.INVALID_CONTACT,
            "contact.inner_outer_force_n",
            f"force must be in [0, {config.max_contact_force_n:g}] N",
        )
    _raise_if(
        not contact.source.strip(),
        SensorFaultCode.INVALID_CONTACT,
        "contact.source",
        "contact source is empty",
    )
    _validate_timestamp(
        contact.timestamp_s,
        packet_time_s,
        field="contact.timestamp_s",
        max_age_s=config.max_contact_age_s,
        future_tolerance_s=config.future_tolerance_s,
    )


def _validate_distal_pad_midpoint(
    midpoint: CalibratedPadSurfaceMidpoint,
    *,
    packet_time_s: torch.Tensor,
    config: SensorValidationConfig,
) -> None:
    batch_size = packet_time_s.shape[0]
    _require_shape(
        midpoint.position_robot_root_m,
        (batch_size, 3),
        field="distal_pad_midpoint.position_robot_root_m",
        code=SensorFaultCode.INVALID_FK,
    )
    _require_finite(
        midpoint.position_robot_root_m,
        field="distal_pad_midpoint.position_robot_root_m",
        code=SensorFaultCode.INVALID_FK,
    )
    _require_shape(
        midpoint.velocity_robot_root_m_s,
        (batch_size, 3),
        field="distal_pad_midpoint.velocity_robot_root_m_s",
        code=SensorFaultCode.INVALID_FK,
    )
    _require_finite(
        midpoint.velocity_robot_root_m_s,
        field="distal_pad_midpoint.velocity_robot_root_m_s",
        code=SensorFaultCode.INVALID_FK,
    )
    _raise_if(
        midpoint.position_robot_root_m.dtype != torch.float32,
        SensorFaultCode.INVALID_FK,
        "distal_pad_midpoint.position_robot_root_m",
        "canonical pad position dtype must be float32",
    )
    _raise_if(
        midpoint.velocity_robot_root_m_s.dtype != torch.float32,
        SensorFaultCode.INVALID_FK,
        "distal_pad_midpoint.velocity_robot_root_m_s",
        "canonical pad velocity dtype must be float32",
    )
    _require_shape(
        midpoint.valid,
        (batch_size,),
        field="distal_pad_midpoint.valid",
        code=SensorFaultCode.INVALID_FK,
    )
    _raise_if(
        midpoint.valid.dtype != torch.bool,
        SensorFaultCode.INVALID_FK,
        "distal_pad_midpoint.valid",
        "pad midpoint validity must be bool",
    )
    _raise_if(
        ~midpoint.valid,
        SensorFaultCode.INVALID_FK,
        "distal_pad_midpoint.valid",
        "calibrated pad-surface FK is invalid",
    )
    _raise_if(
        not midpoint.calibration_id.strip(),
        SensorFaultCode.INVALID_FK,
        "distal_pad_midpoint.calibration_id",
        "pad-surface calibration id is required",
    )
    _raise_if(
        midpoint.source is not TensorSource.LIVE_FK,
        SensorFaultCode.INVALID_FK,
        "distal_pad_midpoint.source",
        "pad-surface midpoint must come from live FK, never privileged GT",
    )
    _validate_timestamp(
        midpoint.timestamp_s,
        packet_time_s,
        field="distal_pad_midpoint.timestamp_s",
        max_age_s=config.max_robot_age_s,
        future_tolerance_s=config.future_tolerance_s,
    )


def _validate_cadence(
    current: torch.Tensor,
    previous: torch.Tensor,
    step_delta: torch.Tensor,
    *,
    period_s: float,
    tolerance_s: float,
    field: str,
) -> None:
    delta = current - previous
    expected = step_delta.to(device=delta.device, dtype=delta.dtype) * period_s
    _raise_if(
        torch.abs(delta - expected) > tolerance_s,
        SensorFaultCode.CADENCE_VIOLATION,
        field,
        f"timestamp delta does not match {period_s:g} s cadence",
    )


class SensorPacketValidator:
    """Validate one canonical packet and optional same-episode predecessor."""

    def __init__(self, config: SensorValidationConfig | None = None) -> None:
        self.config = config or SensorValidationConfig()

    def validate_contact_frame(
        self,
        contact: ContactFrame,
        *,
        packet_time_s: torch.Tensor,
    ) -> ContactFrame:
        """Validate one contact sample without consuming packet cadence.

        Outcome recorders sometimes need a post-action contact sample while
        the next full packet remains the next recurrent/pre-action state.
        Validating that contact frame independently avoids inserting a second
        full packet at the same control step into the strict monotonic packet
        history.
        """

        _validate_contact(
            contact,
            packet_time_s=packet_time_s,
            config=self.config,
        )
        return contact

    def validate(
        self,
        packet: SynchronizedSensorPacket,
        *,
        previous: SynchronizedSensorPacket | None = None,
    ) -> SynchronizedSensorPacket:
        if packet.schema_version is not SchemaVersion.CONTROLLED_REBUILD_V2:
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "schema_version",
                "packet schema does not match the canonical controlled rebuild",
            )
        timestamp = packet.timestamp_s
        _require_shape(
            timestamp,
            (None,),
            field="timestamp_s",
            code=SensorFaultCode.INVALID_TIMESTAMP,
        )
        _require_finite(
            timestamp,
            field="timestamp_s",
            code=SensorFaultCode.INVALID_TIMESTAMP,
        )
        _raise_if(
            timestamp < 0.0,
            SensorFaultCode.INVALID_TIMESTAMP,
            "timestamp_s",
            "packet timestamp must be non-negative episode-local seconds",
        )
        batch_size = timestamp.shape[0]
        _require_shape(
            packet.control_step,
            (batch_size,),
            field="control_step",
            code=SensorFaultCode.INVALID_TIMESTAMP,
        )
        _raise_if(
            packet.control_step.dtype != torch.int64,
            SensorFaultCode.INVALID_TIMESTAMP,
            "control_step",
            f"canonical control step must be int64, got {packet.control_step.dtype}",
        )
        _raise_if(
            packet.control_step < 0,
            SensorFaultCode.INVALID_TIMESTAMP,
            "control_step",
            "control step must be non-negative",
        )
        _validate_camera(
            packet.head,
            name="head",
            packet_time_s=timestamp,
            config=self.config,
        )
        _validate_camera(
            packet.wrist,
            name="wrist",
            packet_time_s=timestamp,
            config=self.config,
        )
        _raise_if(
            torch.abs(packet.head.timestamp_s - packet.wrist.timestamp_s)
            > self.config.max_head_wrist_skew_s,
            SensorFaultCode.CAMERA_SKEW,
            "head/wrist.timestamp_s",
            f"capture skew exceeds {self.config.max_head_wrist_skew_s:g} s",
        )
        _validate_robot(packet.robot, packet_time_s=timestamp, config=self.config)
        if packet.contact is not None:
            _validate_contact(
                packet.contact, packet_time_s=timestamp, config=self.config
            )
        if packet.distal_pad_midpoint is not None:
            _validate_distal_pad_midpoint(
                packet.distal_pad_midpoint,
                packet_time_s=timestamp,
                config=self.config,
            )
        if previous is not None:
            self._validate_transition(previous, packet)
        return packet

    def deployable_view(
        self,
        packet: SynchronizedSensorPacket,
        *,
        previous: SynchronizedSensorPacket | None = None,
    ) -> DeployableSensorView:
        """Validate and remove privileged contact before observation building."""

        validated = self.validate(packet, previous=previous)
        return DeployableSensorView(
            timestamp_s=validated.timestamp_s,
            control_step=validated.control_step,
            head=validated.head,
            wrist=validated.wrist,
            robot=validated.robot,
            distal_pad_midpoint=validated.distal_pad_midpoint,
            schema_version=validated.schema_version,
        )

    def _validate_transition(
        self,
        previous: SynchronizedSensorPacket,
        current: SynchronizedSensorPacket,
    ) -> None:
        if previous.timestamp_s.shape != current.timestamp_s.shape:
            raise SensorIntegrityError(
                SensorFaultCode.CADENCE_VIOLATION,
                "timestamp_s",
                "batch size changed between packets",
            )
        step_delta = current.control_step - previous.control_step
        _raise_if(
            step_delta <= 0,
            SensorFaultCode.CADENCE_VIOLATION,
            "control_step",
            "previous must be from the same episode and strictly earlier",
        )
        _validate_cadence(
            current.timestamp_s,
            previous.timestamp_s,
            step_delta,
            period_s=self.config.control_period_s,
            tolerance_s=self.config.cadence_tolerance_s,
            field="timestamp_s",
        )
        for name in ("head", "wrist"):
            current_camera = getattr(current, name)
            previous_camera = getattr(previous, name)
            sequence_delta = current_camera.sequence_id - previous_camera.sequence_id
            _raise_if(
                sequence_delta <= 0,
                SensorFaultCode.MISSING_FRAME,
                f"{name}.sequence_id",
                "no new camera frame for an advanced control step",
            )
            _validate_cadence(
                current_camera.timestamp_s,
                previous_camera.timestamp_s,
                sequence_delta,
                period_s=self.config.camera_capture_period_s,
                tolerance_s=self.config.cadence_tolerance_s,
                field=f"{name}.capture_timestamp_s",
            )
            _validate_cadence(
                current_camera.timestamp_s,
                previous_camera.timestamp_s,
                step_delta,
                period_s=self.config.camera_policy_sample_period_s,
                tolerance_s=self.config.cadence_tolerance_s,
                field=f"{name}.policy_sample_timestamp_s",
            )
        _validate_cadence(
            current.robot.timestamp_s,
            previous.robot.timestamp_s,
            step_delta,
            period_s=self.config.robot_period_s,
            tolerance_s=self.config.cadence_tolerance_s,
            field="robot.timestamp_s",
        )
        if current.contact is not None or previous.contact is not None:
            if current.contact is None or previous.contact is None:
                raise SensorIntegrityError(
                    SensorFaultCode.INVALID_CONTACT,
                    "contact",
                    "contact stream appeared or disappeared within an episode",
                )
            _validate_cadence(
                current.contact.timestamp_s,
                previous.contact.timestamp_s,
                step_delta,
                period_s=self.config.contact_period_s,
                tolerance_s=self.config.cadence_tolerance_s,
                field="contact.timestamp_s",
            )
        if (
            current.distal_pad_midpoint is not None
            or previous.distal_pad_midpoint is not None
        ):
            if (
                current.distal_pad_midpoint is None
                or previous.distal_pad_midpoint is None
            ):
                raise SensorIntegrityError(
                    SensorFaultCode.INVALID_FK,
                    "distal_pad_midpoint",
                    "calibrated pad-surface FK appeared or disappeared within an episode",
                )
            _validate_cadence(
                current.distal_pad_midpoint.timestamp_s,
                previous.distal_pad_midpoint.timestamp_s,
                step_delta,
                period_s=self.config.robot_period_s,
                tolerance_s=self.config.cadence_tolerance_s,
                field="distal_pad_midpoint.timestamp_s",
            )


def _normalized_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool(torch.any(~torch.isfinite(norm) | (norm <= 0.0)).item()):
        raise SensorIntegrityError(
            SensorFaultCode.INVALID_TF,
            "root_quaternion_world_xyzw",
            "quaternion norm is zero or non-finite",
        )
    return quaternion / norm


def world_vectors_to_root(
    vector_world: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame vectors into robot-root coordinates.

    Supports ``[N,3]`` and ``[N,...,3]`` vectors.  Quaternions use XYZW and
    express root orientation in world coordinates.
    """

    if vector_world.ndim < 2 or vector_world.shape[-1] != 3:
        raise ValueError("vector_world must have shape [N,...,3]")
    if (
        root_quaternion_world_xyzw.ndim != 2
        or root_quaternion_world_xyzw.shape
        != (vector_world.shape[0], 4)
    ):
        raise ValueError("root quaternion must have shape [N,4]")
    quaternion = _normalized_quaternion_xyzw(root_quaternion_world_xyzw)
    expand_shape = (quaternion.shape[0],) + (1,) * (vector_world.ndim - 2)
    xyz = quaternion[:, :3].reshape(expand_shape + (3,))
    w = quaternion[:, 3].reshape(expand_shape + (1,))
    # Apply the conjugate quaternion: q^-1 * v * q.
    cross = torch.linalg.cross(xyz, vector_world, dim=-1)
    return vector_world - 2.0 * w * cross + 2.0 * torch.linalg.cross(
        xyz, cross, dim=-1
    )


def world_points_to_root(
    point_world_m: torch.Tensor,
    root_position_world_m: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Apply the full inverse root SE(3) transform to world-frame points."""

    if point_world_m.ndim < 2 or point_world_m.shape[-1] != 3:
        raise ValueError("point_world_m must have shape [N,...,3]")
    if root_position_world_m.shape != (point_world_m.shape[0], 3):
        raise ValueError("root position must have shape [N,3]")
    expand_shape = (root_position_world_m.shape[0],) + (1,) * (
        point_world_m.ndim - 2
    )
    translation = root_position_world_m.reshape(expand_shape + (3,))
    return world_vectors_to_root(
        point_world_m - translation, root_quaternion_world_xyzw
    )


def world_points_to_root_from_native_quaternion(
    point_world_m: torch.Tensor,
    root_position_world_m: torch.Tensor,
    root_quaternion_world_native: torch.Tensor,
    *,
    native_order: str,
) -> torch.Tensor:
    """Canonical world→root SE(3) boundary for framework-native quaternions."""

    from ..g2_quaternion import quaternion_native_to_xyzw

    return world_points_to_root(
        point_world_m,
        root_position_world_m,
        quaternion_native_to_xyzw(root_quaternion_world_native, native_order),
    )


def root_vectors_to_world(
    vector_root: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Rotate robot-root vectors into world coordinates using XYZW quaternions."""

    if vector_root.ndim < 2 or vector_root.shape[-1] != 3:
        raise ValueError("vector_root must have shape [N,...,3]")
    if (
        root_quaternion_world_xyzw.ndim != 2
        or root_quaternion_world_xyzw.shape != (vector_root.shape[0], 4)
    ):
        raise ValueError("root quaternion must have shape [N,4]")
    quaternion = _normalized_quaternion_xyzw(root_quaternion_world_xyzw)
    expand_shape = (quaternion.shape[0],) + (1,) * (vector_root.ndim - 2)
    xyz = quaternion[:, :3].reshape(expand_shape + (3,))
    w = quaternion[:, 3].reshape(expand_shape + (1,))
    cross = torch.linalg.cross(xyz, vector_root, dim=-1)
    return vector_root + 2.0 * w * cross + 2.0 * torch.linalg.cross(
        xyz, cross, dim=-1
    )


def root_points_to_world(
    point_root_m: torch.Tensor,
    root_position_world_m: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Apply the full root SE(3) transform to root-frame points."""

    if point_root_m.ndim < 2 or point_root_m.shape[-1] != 3:
        raise ValueError("point_root_m must have shape [N,...,3]")
    if root_position_world_m.shape != (point_root_m.shape[0], 3):
        raise ValueError("root position must have shape [N,3]")
    expand_shape = (root_position_world_m.shape[0],) + (1,) * (
        point_root_m.ndim - 2
    )
    translation = root_position_world_m.reshape(expand_shape + (3,))
    return root_vectors_to_world(
        point_root_m, root_quaternion_world_xyzw
    ) + translation


def _quaternion_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion operands must share shape [...,4]")
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def world_quaternions_to_root(
    quaternion_world_xyzw: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Express world-oriented poses in robot-root orientation coordinates."""

    world = _normalized_quaternion_xyzw(quaternion_world_xyzw)
    root = _normalized_quaternion_xyzw(root_quaternion_world_xyzw)
    if world.shape != root.shape:
        raise ValueError("world/root quaternion shapes differ")
    conjugate_root = torch.cat((-root[..., :3], root[..., 3:4]), dim=-1)
    return _normalized_quaternion_xyzw(
        _quaternion_multiply_xyzw(conjugate_root, world)
    )


def root_quaternions_to_world(
    quaternion_root_xyzw: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Express root-oriented poses in world orientation coordinates."""

    root_relative = _normalized_quaternion_xyzw(quaternion_root_xyzw)
    root = _normalized_quaternion_xyzw(root_quaternion_world_xyzw)
    if root_relative.shape != root.shape:
        raise ValueError("root-relative/root-world quaternion shapes differ")
    return _normalized_quaternion_xyzw(
        _quaternion_multiply_xyzw(root, root_relative)
    )


def world_point_velocity_to_root(
    point_world_m: torch.Tensor,
    point_velocity_world_m_s: torch.Tensor,
    root_position_world_m: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
    root_linear_velocity_world_m_s: torch.Tensor,
    root_angular_velocity_world_rad_s: torch.Tensor,
) -> torch.Tensor:
    """Return the derivative of a world point expressed in robot-root axes.

    Merely rotating an absolute world velocity is not a root-relative
    velocity when the mobile root translates or rotates.  The derivative of
    ``R_root^T (p_point - p_root)`` is

    ``R_root^T (v_point - v_root - omega_root x (p_point - p_root))``.

    Keeping this transform at the sensor/observation boundary prevents the
    actor from receiving base motion mislabeled as EE motion.
    """

    expected_vector = (point_world_m.shape[0], 3)
    values = {
        "point_world_m": point_world_m,
        "point_velocity_world_m_s": point_velocity_world_m_s,
        "root_position_world_m": root_position_world_m,
        "root_linear_velocity_world_m_s": root_linear_velocity_world_m_s,
        "root_angular_velocity_world_rad_s": root_angular_velocity_world_rad_s,
    }
    if point_world_m.ndim != 2:
        raise ValueError("point_world_m must have shape [N,3]")
    for name, value in values.items():
        if value.shape != expected_vector:
            raise ValueError(f"{name} must have shape {expected_vector}")
    relative_world_m = point_world_m - root_position_world_m
    relative_velocity_world_m_s = (
        point_velocity_world_m_s
        - root_linear_velocity_world_m_s
        - torch.linalg.cross(
            root_angular_velocity_world_rad_s,
            relative_world_m,
            dim=-1,
        )
    )
    return world_vectors_to_root(
        relative_velocity_world_m_s, root_quaternion_world_xyzw
    )


def distal_link_origin_midpoint_root_m(robot: RobotStateFrame) -> torch.Tensor:
    """Return the midpoint of the two distal *link origins* in root frame.

    This deliberately does not call the value a pad midpoint.  A physical pad
    midpoint requires separately validated local pad-surface offsets/frames.
    """

    midpoint_world_m = 0.5 * (
        robot.distal_inner_link_origin_world_m
        + robot.distal_outer_link_origin_world_m
    )
    return world_points_to_root(
        midpoint_world_m,
        robot.root_position_world_m,
        robot.root_quaternion_world_xyzw,
    )


__all__ = [
    "CalibratedPadSurfaceMidpoint",
    "CameraFrame",
    "ContactFrame",
    "DeployableSensorView",
    "RobotStateFrame",
    "SensorFaultCode",
    "SensorIntegrityError",
    "SensorPacketValidator",
    "SensorValidationConfig",
    "SynchronizedSensorPacket",
    "distal_link_origin_midpoint_root_m",
    "root_points_to_world",
    "root_quaternions_to_world",
    "root_vectors_to_world",
    "world_point_velocity_to_root",
    "world_points_to_root",
    "world_points_to_root_from_native_quaternion",
    "world_quaternions_to_root",
    "world_vectors_to_root",
]
