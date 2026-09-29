# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Deployment-compatible derived observations for the controlled G2 rebuild.

Only a validated deployable sensor view and vision predictions enter this
module.  Ground-truth object pose and exact contact are absent from every
public builder signature and from the returned actor payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping

import torch

from .data_contract import G2DataContract, SchemaVersion, TensorSource
from .sensor_packet import (
    DeployableSensorView,
    SensorFaultCode,
    SensorIntegrityError,
    distal_link_origin_midpoint_root_m,
    world_point_velocity_to_root,
    world_points_to_root,
)


@dataclass(frozen=True)
class VisionGeometryPrediction:
    """Deployable metric geometry produced exclusively from RGB-D."""

    predicted_cube_xyz_robot_root_m: torch.Tensor
    predicted_cube_velocity_robot_root_m_s: torch.Tensor
    head_visibility_probability: torch.Tensor
    wrist_visibility_probability: torch.Tensor
    fused_confidence: torch.Tensor
    timestamp_s: torch.Tensor
    valid: torch.Tensor
    model_sha256: str
    validation_sha256: str
    source_rgbd_sha256: str
    source: TensorSource = TensorSource.VISION_PREDICTION


@dataclass(frozen=True)
class ObservationBuilderConfig:
    max_vision_age_s: float = 0.04
    future_tolerance_s: float = 1.0e-6
    approved_vision_model_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.max_vision_age_s <= 0.0:
            raise ValueError("max_vision_age_s must be positive")
        if self.future_tolerance_s < 0.0:
            raise ValueError("future_tolerance_s must be non-negative")
        if self.approved_vision_model_sha256 is not None:
            _require_sha256(
                self.approved_vision_model_sha256,
                "approved_vision_model_sha256",
            )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


def rgbd_source_sha256(sensors: DeployableSensorView) -> str:
    """Hash the exact dual-RGBD sample and capture clocks used by inference."""

    digest = hashlib.sha256()
    for name, value in (
        ("head_rgb", sensors.head.rgb),
        ("head_depth_m", sensors.head.depth_m),
        ("head_depth_valid", sensors.head.depth_valid),
        ("head_timestamp_s", sensors.head.timestamp_s),
        ("head_sequence_id", sensors.head.sequence_id),
        (
            "head_optical_pose_robot_root_xyzw",
            sensors.head.optical_pose_robot_root_xyzw,
        ),
        ("wrist_rgb", sensors.wrist.rgb),
        ("wrist_depth_m", sensors.wrist.depth_m),
        ("wrist_depth_valid", sensors.wrist.depth_valid),
        ("wrist_timestamp_s", sensors.wrist.timestamp_s),
        ("wrist_sequence_id", sensors.wrist.sequence_id),
        (
            "wrist_optical_pose_robot_root_xyzw",
            sensors.wrist.optical_pose_robot_root_xyzw,
        ),
    ):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DeployableObservation:
    """Actor payload plus explicitly non-actor diagnostic geometry."""

    _actor_payload: Mapping[str, torch.Tensor]
    distal_link_origin_midpoint_robot_root_m_diagnostic: torch.Tensor
    schema_version: SchemaVersion
    pad_surface_calibration_id: str

    def actor_payload(self) -> dict[str, torch.Tensor]:
        """Return a copy so downstream code cannot mutate stored inputs."""

        return dict(self._actor_payload)


def _require_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape:
        raise SensorIntegrityError(
            SensorFaultCode.INVALID_ROBOT_STATE,
            name,
            f"expected shape {shape}, got {tuple(value.shape)}",
        )


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise SensorIntegrityError(
            SensorFaultCode.INVALID_ROBOT_STATE,
            name,
            "contains NaN or Inf",
        )


def _quaternion_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def _quaternion_multiply_xyzw(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:]
    right_xyz, right_w = right[..., :3], right[..., 3:]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _ee_pose_robot_root_xyzw(view: DeployableSensorView) -> torch.Tensor:
    robot = view.robot
    position = world_points_to_root(
        robot.ee_position_world_m,
        robot.root_position_world_m,
        robot.root_quaternion_world_xyzw,
    )
    root_q = robot.root_quaternion_world_xyzw
    root_q = root_q / torch.linalg.vector_norm(root_q, dim=-1, keepdim=True)
    ee_q = robot.ee_quaternion_world_xyzw
    ee_q = ee_q / torch.linalg.vector_norm(ee_q, dim=-1, keepdim=True)
    relative_q = _quaternion_multiply_xyzw(
        _quaternion_conjugate_xyzw(root_q), ee_q
    )
    relative_q = relative_q / torch.linalg.vector_norm(
        relative_q, dim=-1, keepdim=True
    )
    # Canonicalize sign to make the same orientation serialize identically.
    relative_q = torch.where(relative_q[..., 3:] < 0.0, -relative_q, relative_q)
    return torch.cat((position, relative_q), dim=-1)


def _depth_with_channel(depth: torch.Tensor) -> torch.Tensor:
    return depth.unsqueeze(-1) if depth.ndim == 3 else depth


class DeployableObservationBuilder:
    """Build one allowlisted actor input without privileged information."""

    def __init__(
        self,
        *,
        contract: G2DataContract | None = None,
        config: ObservationBuilderConfig | None = None,
    ) -> None:
        self.contract = contract or G2DataContract.canonical()
        self.config = config or ObservationBuilderConfig()
        # These operations must remain meaningful in one coordinate frame.
        self.contract.require_same_frame(
            "predicted_cube_xyz_robot_root_m",
            "distal_pad_midpoint_robot_root_m",
            "relative_grasp_xyz_robot_root_m",
        )
        self.contract.require_same_frame(
            "predicted_cube_velocity_robot_root_m_s",
            "ee_linear_velocity_robot_root_m_s",
            "relative_velocity_robot_root_m_s",
        )

    def build(
        self,
        sensors: DeployableSensorView,
        vision: VisionGeometryPrediction,
        *,
        previous_action: torch.Tensor | None = None,
    ) -> DeployableObservation:
        """Create an actor payload.

        Temporal history remains the recurrent policy's hidden state, not a
        sensor tensor.  The signature intentionally has no cube-GT, simulator
        object-state, contact, or phase argument.
        """

        if not isinstance(sensors, DeployableSensorView):
            raise TypeError(
                "DeployableObservationBuilder requires DeployableSensorView; "
                "validate/project the full packet first"
            )
        self.contract.assert_schema(sensors.schema_version)
        batch_size = sensors.timestamp_s.shape[0]
        expected_vector = (batch_size, 3)
        _require_shape(
            vision.predicted_cube_xyz_robot_root_m,
            expected_vector,
            "predicted_cube_xyz_robot_root_m",
        )
        _require_shape(
            vision.predicted_cube_velocity_robot_root_m_s,
            expected_vector,
            "predicted_cube_velocity_robot_root_m_s",
        )
        _require_shape(vision.timestamp_s, (batch_size,), "vision.timestamp_s")
        _require_shape(vision.valid, (batch_size,), "vision.valid")
        for name, value in (
            ("head_visibility_probability", vision.head_visibility_probability),
            ("wrist_visibility_probability", vision.wrist_visibility_probability),
            ("fused_confidence", vision.fused_confidence),
        ):
            _require_shape(value, (batch_size, 1), name)
            _require_finite(value, name)
            if bool(torch.any((value < 0.0) | (value > 1.0)).item()):
                raise SensorIntegrityError(
                    SensorFaultCode.CORRUPT_DEPTH,
                    name,
                    "vision confidence must be in [0,1]",
                )
        _require_finite(
            vision.predicted_cube_xyz_robot_root_m,
            "predicted_cube_xyz_robot_root_m",
        )
        _require_finite(
            vision.predicted_cube_velocity_robot_root_m_s,
            "predicted_cube_velocity_robot_root_m_s",
        )
        _require_finite(vision.timestamp_s, "vision.timestamp_s")
        if vision.source is not TensorSource.VISION_PREDICTION:
            raise RuntimeError(
                "G2_GT_RUNTIME_LEAKAGE: cube geometry source must be vision_prediction"
            )
        for name, value in (
            ("vision.model_sha256", vision.model_sha256),
            ("vision.validation_sha256", vision.validation_sha256),
            ("vision.source_rgbd_sha256", vision.source_rgbd_sha256),
        ):
            _require_sha256(value, name)
        if (
            self.config.approved_vision_model_sha256 is not None
            and vision.model_sha256 != self.config.approved_vision_model_sha256
        ):
            raise RuntimeError("G2_VISION_MODEL_NOT_APPROVED")
        if vision.source_rgbd_sha256 != rgbd_source_sha256(sensors):
            raise RuntimeError(
                "G2_GT_RUNTIME_LEAKAGE: vision prediction is not bound to current dual RGB-D"
            )
        if vision.valid.dtype != torch.bool or not bool(vision.valid.all().item()):
            raise SensorIntegrityError(
                SensorFaultCode.CORRUPT_DEPTH,
                "vision.valid",
                "vision geometry is invalid; GT fallback is forbidden",
            )
        vision_age = sensors.timestamp_s - vision.timestamp_s
        if bool(torch.any(vision_age < -self.config.future_tolerance_s).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "vision.timestamp_s",
                "vision prediction is in the future",
            )
        if bool(torch.any(vision_age > self.config.max_vision_age_s).item()):
            raise SensorIntegrityError(
                SensorFaultCode.STALE_FRAME,
                "vision.timestamp_s",
                f"vision age exceeds {self.config.max_vision_age_s:g} s",
            )

        pad = sensors.distal_pad_midpoint
        if pad is None:
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_FK,
                "distal_pad_midpoint_robot_root_m",
                "validated pad-surface offsets are required; link origins are diagnostic only",
            )
        if pad.source is not TensorSource.LIVE_FK:
            raise RuntimeError(
                "G2_GT_RUNTIME_LEAKAGE: pad midpoint source must be live_fk"
            )
        pad_position = pad.position_robot_root_m
        pad_velocity = pad.velocity_robot_root_m_s
        _require_shape(
            pad_position,
            expected_vector,
            "distal_pad_midpoint_robot_root_m",
        )
        _require_finite(pad_position, "distal_pad_midpoint_robot_root_m")
        _require_shape(
            pad_velocity,
            expected_vector,
            "distal_pad_midpoint_velocity_robot_root_m_s",
        )
        _require_finite(
            pad_velocity, "distal_pad_midpoint_velocity_robot_root_m_s"
        )

        ee_pose = _ee_pose_robot_root_xyzw(sensors)
        ee_velocity = world_point_velocity_to_root(
            sensors.robot.ee_position_world_m,
            sensors.robot.ee_linear_velocity_world_m_s,
            sensors.robot.root_position_world_m,
            sensors.robot.root_quaternion_world_xyzw,
            sensors.robot.root_linear_velocity_world_m_s,
            sensors.robot.root_angular_velocity_world_rad_s,
        )
        relative_grasp = vision.predicted_cube_xyz_robot_root_m - pad_position
        relative_velocity = (
            vision.predicted_cube_velocity_robot_root_m_s - pad_velocity
        )
        camera_age = torch.stack(
            (
                sensors.timestamp_s - sensors.head.timestamp_s,
                sensors.timestamp_s - sensors.wrist.timestamp_s,
            ),
            dim=-1,
        ).to(torch.float32)

        payload: dict[str, torch.Tensor] = {
            "head_rgb": sensors.head.rgb,
            "head_depth_m": _depth_with_channel(sensors.head.depth_m),
            "head_depth_valid": _depth_with_channel(sensors.head.depth_valid),
            "right_wrist_rgb": sensors.wrist.rgb,
            "right_wrist_depth_m": _depth_with_channel(sensors.wrist.depth_m),
            "right_wrist_depth_valid": _depth_with_channel(
                sensors.wrist.depth_valid
            ),
            "head_camera_pose_robot_root_xyzw": (
                sensors.head.optical_pose_robot_root_xyzw
            ),
            "right_wrist_camera_pose_robot_root_xyzw": (
                sensors.wrist.optical_pose_robot_root_xyzw
            ),
            "robot_joint_position_rad": sensors.robot.joint_position_rad,
            "robot_joint_velocity_rad_s": sensors.robot.joint_velocity_rad_s,
            "ee_pose_robot_root_xyzw": ee_pose.to(torch.float32),
            "predicted_cube_xyz_robot_root_m": (
                vision.predicted_cube_xyz_robot_root_m.to(torch.float32)
            ),
            "predicted_cube_velocity_robot_root_m_s": (
                vision.predicted_cube_velocity_robot_root_m_s.to(torch.float32)
            ),
            "predicted_head_visibility_probability": (
                vision.head_visibility_probability.to(torch.float32)
            ),
            "predicted_right_wrist_visibility_probability": (
                vision.wrist_visibility_probability.to(torch.float32)
            ),
            "predicted_fused_confidence": vision.fused_confidence.to(
                torch.float32
            ),
            "distal_pad_midpoint_robot_root_m": pad_position.to(torch.float32),
            "relative_grasp_xyz_robot_root_m": relative_grasp.to(torch.float32),
            "ee_linear_velocity_robot_root_m_s": ee_velocity.to(torch.float32),
            "relative_velocity_robot_root_m_s": relative_velocity.to(torch.float32),
            "gripper_state": sensors.robot.gripper_state,
            "camera_frame_age_s": camera_age,
        }
        if previous_action is not None:
            payload["previous_action"] = previous_action

        # This rejects privileged/unknown keys and validates all canonical
        # tensor shapes, dtypes, ranges and finite-value requirements.
        self.contract.assert_actor_payload(payload)
        diagnostic_midpoint = distal_link_origin_midpoint_root_m(sensors.robot)
        return DeployableObservation(
            _actor_payload=MappingProxyType(payload.copy()),
            distal_link_origin_midpoint_robot_root_m_diagnostic=(
                diagnostic_midpoint
            ),
            schema_version=sensors.schema_version,
            pad_surface_calibration_id=pad.calibration_id,
        )


__all__ = [
    "DeployableObservation",
    "DeployableObservationBuilder",
    "ObservationBuilderConfig",
    "VisionGeometryPrediction",
    "rgbd_source_sha256",
]
