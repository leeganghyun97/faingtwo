# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lazy Isaac-Lab adapter for controlled-rebuild M2/M4 tensors.

The module itself imports no Isaac/Kit package.  Isaac-dependent authorities
are resolved only when a live adapter is constructed without injected test
doubles.  This keeps dataset checks and unit tests runnable on CPU-only hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Mapping

import torch

from .data_contract import G2DataContract, G2_RUNTIME_JOINT_ORDER, TensorSource
from .observations import (
    DeployableObservation,
    DeployableObservationBuilder,
    VisionGeometryPrediction,
)
from .sensor_packet import (
    CalibratedPadSurfaceMidpoint,
    CameraFrame,
    ContactFrame,
    DeployableSensorView,
    RobotStateFrame,
    SensorFaultCode,
    SensorIntegrityError,
    SensorPacketValidator,
    SynchronizedSensorPacket,
    world_points_to_root,
    world_vectors_to_root,
)


RIGHT_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
RIGHT_GRIPPER_MASTER = "idx81_gripper_r_outer_joint1"
RIGHT_EE_BODY = "gripper_r_center_link"
RIGHT_INNER_DISTAL_LINK = "gripper_r_inner_link4"
RIGHT_OUTER_DISTAL_LINK = "gripper_r_outer_link4"
CONTACT_CHANNEL_SEMANTICS = (
    "right_gripper_inner_finger",
    "right_gripper_outer_finger",
)
ActionTensorName = Literal["operator_action_8d", "canonical_policy_action_7d"]


def _tensor_value(value: Any) -> torch.Tensor:
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.from_dlpack(value)
    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
        raise TypeError(f"unsupported Isaac tensor type: {type(value)!r}") from error


@dataclass(frozen=True)
class PadSurfaceCalibration:
    """Identified local pad-surface centers in their distal link frames."""

    inner_link_local_offset_m: tuple[float, float, float]
    outer_link_local_offset_m: tuple[float, float, float]
    calibration_id: str
    source_artifact_sha256: str
    calibration_rows: int
    inner_link_name: str = RIGHT_INNER_DISTAL_LINK
    outer_link_name: str = RIGHT_OUTER_DISTAL_LINK

    def __post_init__(self) -> None:
        if not self.calibration_id.strip() or not self.source_artifact_sha256.strip():
            raise ValueError("pad calibration id and source artifact are required")
        if len(self.source_artifact_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_artifact_sha256.lower()
        ):
            raise ValueError("pad calibration source artifact must be a SHA-256 hex digest")
        if self.calibration_rows <= 0:
            raise ValueError("pad calibration must contain at least one row")
        if self.inner_link_name != RIGHT_INNER_DISTAL_LINK:
            raise ValueError("inner pad offset is bound to the wrong link")
        if self.outer_link_name != RIGHT_OUTER_DISTAL_LINK:
            raise ValueError("outer pad offset is bound to the wrong link")
        offsets = torch.tensor(
            (self.inner_link_local_offset_m, self.outer_link_local_offset_m),
            dtype=torch.float64,
        )
        if offsets.shape != (2, 3) or not bool(torch.isfinite(offsets).all()):
            raise ValueError("pad local offsets must be two finite meter vectors")
        if bool(torch.any(torch.linalg.vector_norm(offsets, dim=-1) <= 1.0e-6).item()):
            raise ValueError(
                "each uncalibrated zero link-origin offset is forbidden for pad surfaces"
            )


@dataclass(frozen=True)
class ActionTimestampRecord:
    """Action value with source and control-clock acceptance timestamps."""

    tensor_name: ActionTensorName
    value: torch.Tensor
    source_timestamp_s: torch.Tensor
    accepted_timestamp_s: torch.Tensor
    source: str


@dataclass(frozen=True)
class IsaacStateCapture:
    """Synchronized state capture with no implied action acceptance."""

    packet: SynchronizedSensorPacket
    deployable_sensors: DeployableSensorView
    camera_world_transform: Mapping[str, torch.Tensor]
    contact_channel_semantics: tuple[str, str] = CONTACT_CHANNEL_SEMANTICS


@dataclass(frozen=True)
class IsaacAdapterCapture(IsaacStateCapture):
    """Backward-compatible atomic pre-action state + accepted action record."""

    action: ActionTimestampRecord | None = None


@dataclass(frozen=True)
class IsaacAdapterObservation:
    capture: IsaacStateCapture
    observation: DeployableObservation


@dataclass(frozen=True)
class IsaacRuntimeAdapterConfig:
    head_camera_scene_key: str = "head_camera"
    wrist_camera_scene_key: str = "right_wrist_camera"
    robot_scene_key: str = "robot"
    ee_frame_scene_key: str = "ee_frame"
    inner_contact_scene_key: str = "right_inner_finger_contact"
    outer_contact_scene_key: str = "right_outer_finger_contact"
    gripper_open_position_rad: float = 0.785398
    gripper_position_tolerance_rad: float = 0.02
    contact_force_threshold_n: float = 1.0
    max_contact_pair_skew_s: float = 0.0025
    max_action_source_age_s: float = 0.25
    action_clock_tolerance_s: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.gripper_open_position_rad <= 0.0:
            raise ValueError("gripper open position must be positive")
        if self.gripper_position_tolerance_rad < 0.0:
            raise ValueError("gripper tolerance must be non-negative")
        positive = (
            self.contact_force_threshold_n,
            self.max_contact_pair_skew_s,
            self.max_action_source_age_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("contact/action limits must be positive")
        if self.action_clock_tolerance_s < 0.0:
            raise ValueError("action clock tolerance must be non-negative")


ContactReader = Callable[
    [Any],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


def calibrated_pad_surface_midpoint_root_m(
    *,
    body_position_world_m: torch.Tensor,
    body_quaternion_world_xyzw: torch.Tensor,
    root_position_world_m: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
    inner_body_index: int,
    outer_body_index: int,
    calibration: PadSurfaceCalibration,
) -> torch.Tensor:
    """Resolve the physical pad-surface midpoint from identified local offsets.

    Distal link origins are intentionally insufficient for this calculation.
    Callers must supply a validated :class:`PadSurfaceCalibration`; no zero or
    guessed offset fallback exists.
    """

    if not isinstance(calibration, PadSurfaceCalibration):
        raise TypeError("explicit PadSurfaceCalibration is required")
    if body_position_world_m.ndim != 3 or body_position_world_m.shape[-1] != 3:
        raise ValueError("body positions must have shape [N,B,3]")
    if body_quaternion_world_xyzw.shape != (*body_position_world_m.shape[:2], 4):
        raise ValueError("body quaternions must have shape [N,B,4]")
    batch, bodies = body_position_world_m.shape[:2]
    if not 0 <= inner_body_index < bodies or not 0 <= outer_body_index < bodies:
        raise ValueError("calibrated distal body index is outside articulation bodies")
    offsets = torch.tensor(
        (
            calibration.inner_link_local_offset_m,
            calibration.outer_link_local_offset_m,
        ),
        dtype=body_position_world_m.dtype,
        device=body_position_world_m.device,
    ).unsqueeze(0).expand(batch, -1, -1)
    indices = torch.tensor(
        (inner_body_index, outer_body_index),
        dtype=torch.long,
        device=body_position_world_m.device,
    )
    origins = body_position_world_m.index_select(1, indices)
    quaternions = body_quaternion_world_xyzw.index_select(1, indices)
    xyz = quaternions[..., :3]
    w = quaternions[..., 3:]
    cross = torch.linalg.cross(xyz, offsets, dim=-1)
    rotated_offsets = offsets + 2.0 * w * cross + 2.0 * torch.linalg.cross(
        xyz, cross, dim=-1
    )
    surface_midpoint_world = (origins + rotated_offsets).mean(dim=1)
    return world_points_to_root(
        surface_midpoint_world,
        root_position_world_m,
        root_quaternion_world_xyzw,
    )


def calibrated_pad_surface_midpoint_kinematics_root(
    *,
    body_position_world_m: torch.Tensor,
    body_quaternion_world_xyzw: torch.Tensor,
    body_linear_velocity_world_m_s: torch.Tensor,
    body_angular_velocity_world_rad_s: torch.Tensor,
    root_position_world_m: torch.Tensor,
    root_quaternion_world_xyzw: torch.Tensor,
    root_linear_velocity_world_m_s: torch.Tensor,
    root_angular_velocity_world_rad_s: torch.Tensor,
    inner_body_index: int,
    outer_body_index: int,
    calibration: PadSurfaceCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return calibrated midpoint position and its true root-frame velocity.

    Each pad point uses rigid-point kinematics ``v + omega x r``.  Root
    translation and rotation are then removed before rotating into robot-root
    coordinates, so the velocity is the derivative of the corresponding
    root-frame midpoint position rather than the EE link-origin velocity.
    """

    position = calibrated_pad_surface_midpoint_root_m(
        body_position_world_m=body_position_world_m,
        body_quaternion_world_xyzw=body_quaternion_world_xyzw,
        root_position_world_m=root_position_world_m,
        root_quaternion_world_xyzw=root_quaternion_world_xyzw,
        inner_body_index=inner_body_index,
        outer_body_index=outer_body_index,
        calibration=calibration,
    )
    if body_linear_velocity_world_m_s.shape != body_position_world_m.shape:
        raise ValueError("body linear velocities must match body positions")
    if body_angular_velocity_world_rad_s.shape != body_position_world_m.shape:
        raise ValueError("body angular velocities must match body positions")
    batch = body_position_world_m.shape[0]
    indices = torch.tensor(
        (inner_body_index, outer_body_index),
        dtype=torch.long,
        device=body_position_world_m.device,
    )
    offsets = torch.tensor(
        (
            calibration.inner_link_local_offset_m,
            calibration.outer_link_local_offset_m,
        ),
        dtype=body_position_world_m.dtype,
        device=body_position_world_m.device,
    ).unsqueeze(0).expand(batch, -1, -1)
    origins = body_position_world_m.index_select(1, indices)
    quaternions = body_quaternion_world_xyzw.index_select(1, indices)
    xyz = quaternions[..., :3]
    w = quaternions[..., 3:]
    cross = torch.linalg.cross(xyz, offsets, dim=-1)
    rotated_offsets = offsets + 2.0 * w * cross + 2.0 * torch.linalg.cross(
        xyz, cross, dim=-1
    )
    surface_points_world = origins + rotated_offsets
    surface_velocity_world = (
        body_linear_velocity_world_m_s.index_select(1, indices)
        + torch.linalg.cross(
            body_angular_velocity_world_rad_s.index_select(1, indices),
            rotated_offsets,
            dim=-1,
        )
    )
    midpoint_world = surface_points_world.mean(dim=1)
    midpoint_velocity_world = surface_velocity_world.mean(dim=1)
    relative_world = midpoint_world - root_position_world_m
    root_relative_velocity_world = (
        midpoint_velocity_world
        - root_linear_velocity_world_m_s
        - torch.linalg.cross(
            root_angular_velocity_world_rad_s, relative_world, dim=-1
        )
    )
    velocity = world_vectors_to_root(
        root_relative_velocity_world, root_quaternion_world_xyzw
    )
    return position, velocity


class G2IsaacRuntimeAdapter:
    """Adapt a live G2 ManagerBasedEnv into M2/M4 contract objects."""

    def __init__(
        self,
        env: Any,
        *,
        pad_surface_calibration: PadSurfaceCalibration,
        config: IsaacRuntimeAdapterConfig | None = None,
        contract: G2DataContract | None = None,
        validator: SensorPacketValidator | None = None,
        observation_builder: DeployableObservationBuilder | None = None,
        camera_pose_resolver: Any | None = None,
        contact_reader: ContactReader | None = None,
        native_quaternion_order: Literal["xyzw", "wxyz"] | None = None,
    ) -> None:
        if not isinstance(pad_surface_calibration, PadSurfaceCalibration):
            raise TypeError("explicit PadSurfaceCalibration is required")
        self.env = env
        self.config = config or IsaacRuntimeAdapterConfig()
        self.contract = contract or G2DataContract.canonical()
        self.validator = validator or SensorPacketValidator()
        self.observation_builder = observation_builder or DeployableObservationBuilder(
            contract=self.contract
        )
        self.pad_calibration = pad_surface_calibration
        self.head = env.scene[self.config.head_camera_scene_key]
        self.wrist = env.scene[self.config.wrist_camera_scene_key]
        self.robot = env.scene[self.config.robot_scene_key]
        self.ee_frame = env.scene[self.config.ee_frame_scene_key]
        self.inner_contact = env.scene[self.config.inner_contact_scene_key]
        self.outer_contact = env.scene[self.config.outer_contact_scene_key]
        if not math.isclose(
            float(env.step_dt),
            self.validator.config.control_period_s,
            abs_tol=self.validator.config.cadence_tolerance_s,
        ):
            raise RuntimeError("G2_POLICY_SAMPLE_PERIOD_CONTRACT_MISMATCH")
        for camera_name, camera in (("head", self.head), ("right_wrist", self.wrist)):
            update_period = float(
                getattr(getattr(camera, "cfg", None), "update_period", -1.0)
            )
            if not math.isclose(
                update_period,
                self.validator.config.camera_capture_period_s,
                abs_tol=1.0e-9,
            ):
                raise RuntimeError(
                    "G2_CAMERA_CAPTURE_PERIOD_CONTRACT_MISMATCH:"
                    f"{camera_name}:{update_period:g}!="
                    f"{self.validator.config.camera_capture_period_s:g}"
                )
        if camera_pose_resolver is None:
            # G2AssetCameraPoseResolver imports omni.usd only while constructing
            # a live resolver; importing this adapter remains CPU/offline safe.
            from ..g2_asset_camera_pose import G2AssetCameraPoseResolver

            camera_pose_resolver = G2AssetCameraPoseResolver(env)
        self.camera_pose_resolver = camera_pose_resolver
        camera_contract = camera_pose_resolver.contract()
        if camera_contract.get("binding_validation_pass") is not True:
            raise RuntimeError("G2_CAMERA_BINDING_VALIDATION_FAILED")
        if camera_contract.get("runtime_pose_override") is True:
            raise RuntimeError("G2_CAMERA_RUNTIME_POSE_OVERRIDE_FORBIDDEN")
        self.camera_pose_authority = str(camera_contract.get("authority", ""))
        if not self.camera_pose_authority:
            raise RuntimeError("G2_CAMERA_POSE_AUTHORITY_MISSING")
        validated_names = set(camera_contract.get("validated_camera_names", ()))
        if validated_names != {"head", "right_wrist"}:
            raise RuntimeError("G2_CAMERA_BINDING_NAMES_INVALID")
        if int(camera_contract.get("validated_environment_count", -1)) != int(
            env.num_envs
        ):
            raise RuntimeError("G2_CAMERA_BINDING_ENVIRONMENT_COUNT_INVALID")
        if contact_reader is None:
            # This public telemetry primitive preserves the current reward and
            # accepted-close semantics.  It is deliberately resolved lazily.
            from ..g2_lift_task_mdp import contact_grasp_telemetry

            contact_reader = contact_grasp_telemetry
        self.contact_reader = contact_reader
        if native_quaternion_order is None:
            from ..g2_quaternion import isaaclab_native_quaternion_order

            native_quaternion_order = isaaclab_native_quaternion_order()
        if native_quaternion_order not in ("xyzw", "wxyz"):
            raise ValueError("native quaternion order must be xyzw or wxyz")
        self.native_quaternion_order = native_quaternion_order
        if tuple(self.robot.joint_names) != G2_RUNTIME_JOINT_ORDER:
            raise RuntimeError("G2_ARTICULATION_JOINT_ORDER_CONTRACT_MISMATCH")
        self._controlled_joint_indices = torch.tensor(
            [
                self.robot.joint_names.index(name)
                for name in (*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER)
            ],
            dtype=torch.long,
            device=env.device,
        )
        self._gripper_master_index = self.robot.joint_names.index(RIGHT_GRIPPER_MASTER)
        self._ee_body_index = self.robot.body_names.index(RIGHT_EE_BODY)
        self._inner_body_index = self.robot.body_names.index(
            pad_surface_calibration.inner_link_name
        )
        self._outer_body_index = self.robot.body_names.index(
            pad_surface_calibration.outer_link_name
        )
        self._episode_start_step = _tensor_value(env.episode_length_buf).clone()
        self._previous_packet: SynchronizedSensorPacket | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        current = _tensor_value(self.env.episode_length_buf)
        if env_ids is None:
            self._episode_start_step = current.clone()
        else:
            self._episode_start_step[env_ids] = current[env_ids]
        # Cadence comparison across even one reset environment is ambiguous;
        # fail closed and begin a fresh transition chain for the full batch.
        self._previous_packet = None

    def _native_to_xyzw(self, quaternion: torch.Tensor) -> torch.Tensor:
        if quaternion.shape[-1] != 4:
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TF,
                "quaternion",
                "Isaac quaternion must have four components",
            )
        converted = (
            quaternion
            if self.native_quaternion_order == "xyzw"
            else quaternion[..., (1, 2, 3, 0)]
        )
        norm = torch.linalg.vector_norm(converted, dim=-1, keepdim=True)
        if bool(torch.any(~torch.isfinite(norm) | (norm <= 0.0)).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TF,
                "quaternion",
                "Isaac quaternion norm is zero or non-finite",
            )
        converted = converted / norm
        return torch.where(converted[..., 3:] < 0.0, -converted, converted)

    @staticmethod
    def _quat_apply_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        xyz = quaternion[..., :3]
        w = quaternion[..., 3:]
        cross = torch.linalg.cross(xyz, vector, dim=-1)
        return vector + 2.0 * w * cross + 2.0 * torch.linalg.cross(
            xyz, cross, dim=-1
        )

    def _episode_clock(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_step = _tensor_value(self.env.episode_length_buf).to(torch.int64)
        control_step = current_step - self._episode_start_step.to(torch.int64)
        if bool(torch.any(control_step < 0).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "control_step",
                "environment step counter moved backwards without adapter.reset()",
            )
        timestamp_s = control_step.to(torch.float32) * float(self.env.step_dt)
        return control_step, timestamp_s

    def _camera_timing(
        self, episode_time_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ..g2_camera_timing import dual_camera_capture_timing

        return dual_camera_capture_timing(
            self.head,
            self.wrist,
            fallback_episode_time_s=episode_time_s,
        )

    def _camera_transform(self, name: str) -> torch.Tensor:
        transform = _tensor_value(self.camera_pose_resolver.world_transform(name))
        expected = (int(self.env.num_envs), 4, 4)
        if transform.shape != expected or not bool(torch.isfinite(transform).all()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TF,
                f"camera.{name}.world_transform",
                f"expected finite {expected} transform",
            )
        identity_row = torch.tensor(
            (0.0, 0.0, 0.0, 1.0),
            dtype=transform.dtype,
            device=transform.device,
        )
        rotation = transform[:, :3, :3]
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        orthogonal = rotation.transpose(-1, -2) @ rotation
        if (
            not bool(torch.allclose(transform[:, 3], identity_row.expand_as(transform[:, 3]), atol=1.0e-5, rtol=0.0))
            or not bool(torch.allclose(orthogonal, identity.expand_as(orthogonal), atol=1.0e-4, rtol=0.0))
            or not bool(torch.all(torch.linalg.det(rotation) > 0.0).item())
        ):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TF,
                f"camera.{name}.world_transform",
                "camera transform is not a proper homogeneous SE(3) matrix",
            )
        return transform.clone()

    def _camera_frame(
        self,
        camera: Any,
        *,
        name: str,
        timestamp_s: torch.Tensor,
        root_position_world_m: torch.Tensor,
        root_quaternion_world_xyzw: torch.Tensor,
    ) -> CameraFrame:
        output = camera.data.output
        if "rgb" not in output or "distance_to_image_plane" not in output:
            raise SensorIntegrityError(
                SensorFaultCode.MISSING_FRAME,
                f"camera.{name}",
                "RGB or distance_to_image_plane is missing",
            )
        rgb = _tensor_value(output["rgb"])
        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise SensorIntegrityError(
                SensorFaultCode.CORRUPT_RGB,
                f"camera.{name}.rgb",
                f"expected [N,H,W,C>=3], got {tuple(rgb.shape)}",
            )
        rgb = rgb[..., :3].clone()
        if rgb.dtype != torch.uint8:
            raise SensorIntegrityError(
                SensorFaultCode.CORRUPT_RGB,
                f"camera.{name}.rgb",
                f"expected uint8 RGB, got {rgb.dtype}",
            )
        raw_depth = _tensor_value(output["distance_to_image_plane"])
        if raw_depth.ndim == 3:
            raw_depth = raw_depth.unsqueeze(-1)
        if raw_depth.ndim != 4 or raw_depth.shape[-1] != 1:
            raise SensorIntegrityError(
                SensorFaultCode.RGB_DEPTH_MISALIGNMENT,
                f"camera.{name}.depth_m",
                f"expected [N,H,W,1], got {tuple(raw_depth.shape)}",
            )
        if raw_depth.shape[:3] != rgb.shape[:3]:
            raise SensorIntegrityError(
                SensorFaultCode.RGB_DEPTH_MISALIGNMENT,
                f"camera.{name}.depth_m",
                "RGB and depth batch/spatial shapes differ",
            )
        raw_depth = raw_depth.to(torch.float32)
        # Isaac RTX depth commonly uses +Inf for a ray with no return.  That is
        # explicit missing data, while NaN, -Inf and finite out-of-range values
        # are treated as corruption and never silently clipped.
        corrupt = (
            torch.isnan(raw_depth)
            | torch.isneginf(raw_depth)
            | (
                torch.isfinite(raw_depth)
                & (raw_depth != 0.0)
                & (
                    (raw_depth < self.validator.config.min_depth_m)
                    | (raw_depth > self.validator.config.max_depth_m)
                )
            )
        )
        if bool(torch.any(corrupt).item()):
            raise SensorIntegrityError(
                SensorFaultCode.CORRUPT_DEPTH,
                f"camera.{name}.depth_m",
                "NaN, -Inf, or finite out-of-range depth returned by sensor",
            )
        depth_valid = torch.isfinite(raw_depth) & (raw_depth > 0.0)
        depth_m = torch.where(depth_valid, raw_depth, torch.zeros_like(raw_depth))
        sequence = _tensor_value(camera.frame).to(
            device=timestamp_s.device, dtype=torch.int64
        )
        if sequence.ndim == 0:
            sequence = sequence.expand(timestamp_s.shape[0])
        # ``CameraData.pos_w`` and ``quat_w_ros`` are updated in the same
        # Camera._update_buffers_impl transaction that increments ``frame``
        # and renders RGB-D.  They are therefore the capture-frame optical
        # pose authority; querying the USD prim here would instead relabel a
        # stale image with the newest articulation pose.
        camera_position_world = _tensor_value(camera.data.pos_w).to(torch.float32)
        camera_quaternion_world = _tensor_value(camera.data.quat_w_ros).to(
            torch.float32
        )
        if (
            camera_position_world.shape != (timestamp_s.shape[0], 3)
            or camera_quaternion_world.shape != (timestamp_s.shape[0], 4)
        ):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TF,
                f"camera.{name}.capture_pose",
                "CameraData capture pose does not match the RGB-D batch",
            )
        camera_position_root = world_points_to_root(
            camera_position_world,
            root_position_world_m,
            root_quaternion_world_xyzw,
        )
        root_inverse = root_quaternion_world_xyzw.clone()
        root_inverse[..., :3] *= -1.0
        root_xyz = root_inverse[..., :3]
        root_w = root_inverse[..., 3:4]
        camera_xyz = camera_quaternion_world[..., :3]
        camera_w = camera_quaternion_world[..., 3:4]
        camera_quaternion_root = torch.cat(
            (
                root_w * camera_xyz
                + camera_w * root_xyz
                + torch.linalg.cross(root_xyz, camera_xyz, dim=-1),
                root_w * camera_w
                - (root_xyz * camera_xyz).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        pose_norm = torch.linalg.vector_norm(
            camera_quaternion_root, dim=-1, keepdim=True
        )
        if bool(torch.any(~torch.isfinite(pose_norm) | (pose_norm <= 0.0)).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TF,
                f"camera.{name}.capture_pose",
                "capture optical quaternion is zero or non-finite",
            )
        camera_quaternion_root = camera_quaternion_root / pose_norm
        camera_quaternion_root = torch.where(
            camera_quaternion_root[..., 3:4] < 0.0,
            -camera_quaternion_root,
            camera_quaternion_root,
        )
        camera_pose_root = torch.cat(
            (camera_position_root, camera_quaternion_root), dim=-1
        ).to(torch.float32)
        return CameraFrame(
            rgb=rgb,
            depth_m=depth_m,
            depth_valid=depth_valid,
            timestamp_s=timestamp_s.clone(),
            sequence_id=sequence.clone(),
            optical_pose_robot_root_xyzw=camera_pose_root.clone(),
            present=torch.ones_like(timestamp_s, dtype=torch.bool),
            optical_frame=f"{name}_optical",
            source=f"isaaclab_rgbd:{self.camera_pose_authority}",
        )

    def _contact_timestamp(
        self, sensor: Any, episode_time_s: torch.Tensor
    ) -> torch.Tensor:
        current = getattr(sensor, "_timestamp", None)
        captured = getattr(sensor, "_timestamp_last_update", None)
        if current is None or captured is None:
            # Contact is read directly at the pre-action control boundary.  A
            # missing sensor clock is recorded as synchronous only for this
            # update_period=0 Isaac contact stream.
            update_period = float(getattr(getattr(sensor, "cfg", None), "update_period", -1.0))
            if update_period != 0.0:
                raise SensorIntegrityError(
                    SensorFaultCode.INVALID_TIMESTAMP,
                    "contact.timestamp_s",
                    "contact clock unavailable and sensor is not physics-synchronous",
                )
            return episode_time_s.clone()
        current_tensor = _tensor_value(current).to(torch.float32)
        captured_tensor = _tensor_value(captured).to(torch.float32)
        age = torch.clamp(current_tensor - captured_tensor, min=0.0)
        if age.shape != episode_time_s.shape:
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "contact.timestamp_s",
                "contact clock shape differs from episode clock",
            )
        return episode_time_s - age.to(device=episode_time_s.device)

    def _robot_state(
        self, *, timestamp_s: torch.Tensor
    ) -> tuple[RobotStateFrame, CalibratedPadSurfaceMidpoint]:
        data = self.robot.data
        joint_position = _tensor_value(data.joint_pos)
        joint_velocity = _tensor_value(data.joint_vel)
        root_position = _tensor_value(data.root_pos_w)
        root_quaternion = self._native_to_xyzw(_tensor_value(data.root_quat_w))
        ee_position = _tensor_value(self.ee_frame.data.target_pos_w)[:, 0]
        ee_quaternion = self._native_to_xyzw(
            _tensor_value(self.ee_frame.data.target_quat_w)[:, 0]
        )
        body_position = _tensor_value(data.body_pos_w)
        body_quaternion = self._native_to_xyzw(_tensor_value(data.body_quat_w))
        required_velocity_fields = (
            "body_lin_vel_w",
            "body_ang_vel_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
        )
        missing_velocity = [
            name for name in required_velocity_fields if not hasattr(data, name)
        ]
        if missing_velocity:
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_FK,
                "distal_pad_midpoint.velocity_robot_root_m_s",
                "missing rigid-body velocity authority: "
                + ",".join(missing_velocity),
            )
        body_linear_velocity = _tensor_value(data.body_lin_vel_w)
        body_angular_velocity = _tensor_value(data.body_ang_vel_w)
        # FrameTransformer reports the configured EE target point, which need
        # not coincide with the owning rigid body's origin.  PhysX body linear
        # velocity is the origin velocity, so include omega x r before the
        # root-relative point-velocity transform downstream.
        ee_body_position = body_position[:, self._ee_body_index]
        ee_linear_velocity = body_linear_velocity[:, self._ee_body_index] + (
            torch.linalg.cross(
                body_angular_velocity[:, self._ee_body_index],
                ee_position - ee_body_position,
                dim=-1,
            )
        )
        inner_origin = body_position[:, self._inner_body_index]
        outer_origin = body_position[:, self._outer_body_index]
        pad_midpoint_root, pad_velocity_root = (
            calibrated_pad_surface_midpoint_kinematics_root(
                body_position_world_m=body_position,
                body_quaternion_world_xyzw=body_quaternion,
                body_linear_velocity_world_m_s=body_linear_velocity,
                body_angular_velocity_world_rad_s=body_angular_velocity,
                root_position_world_m=root_position,
                root_quaternion_world_xyzw=root_quaternion,
                root_linear_velocity_world_m_s=_tensor_value(data.root_lin_vel_w),
                root_angular_velocity_world_rad_s=_tensor_value(data.root_ang_vel_w),
                inner_body_index=self._inner_body_index,
                outer_body_index=self._outer_body_index,
                calibration=self.pad_calibration,
            )
        )
        gripper_position = joint_position[:, self._gripper_master_index : self._gripper_master_index + 1]
        lower = -self.config.gripper_position_tolerance_rad
        upper = (
            self.config.gripper_open_position_rad
            + self.config.gripper_position_tolerance_rad
        )
        if bool(torch.any((gripper_position < lower) | (gripper_position > upper)).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_ROBOT_STATE,
                "gripper_state",
                "master joint lies outside calibrated closed/open interval",
            )
        gripper_state = torch.clamp(
            gripper_position / self.config.gripper_open_position_rad, 0.0, 1.0
        ).to(torch.float32)
        robot = RobotStateFrame(
            joint_position_rad=joint_position.to(torch.float32).clone(),
            joint_velocity_rad_s=joint_velocity.to(torch.float32).clone(),
            root_position_world_m=root_position.to(torch.float32).clone(),
            root_quaternion_world_xyzw=root_quaternion.to(torch.float32).clone(),
            root_linear_velocity_world_m_s=(
                _tensor_value(data.root_lin_vel_w).to(torch.float32).clone()
            ),
            root_angular_velocity_world_rad_s=(
                _tensor_value(data.root_ang_vel_w).to(torch.float32).clone()
            ),
            ee_position_world_m=ee_position.to(torch.float32).clone(),
            ee_quaternion_world_xyzw=ee_quaternion.to(torch.float32).clone(),
            ee_linear_velocity_world_m_s=(
                ee_linear_velocity.to(torch.float32).clone()
            ),
            distal_inner_link_origin_world_m=inner_origin.to(torch.float32).clone(),
            distal_outer_link_origin_world_m=outer_origin.to(torch.float32).clone(),
            gripper_state=gripper_state.clone(),
            timestamp_s=timestamp_s.clone(),
            source="isaaclab_articulation_and_frame_transformer",
        )
        midpoint = CalibratedPadSurfaceMidpoint(
            position_robot_root_m=pad_midpoint_root.to(torch.float32).clone(),
            velocity_robot_root_m_s=pad_velocity_root.to(torch.float32).clone(),
            timestamp_s=timestamp_s.clone(),
            valid=torch.ones_like(timestamp_s, dtype=torch.bool),
            calibration_id=self.pad_calibration.calibration_id,
            source=TensorSource.LIVE_FK,
        )
        return robot, midpoint

    def _contact_frame(self, *, episode_time_s: torch.Tensor) -> ContactFrame:
        values = self.contact_reader(self.env)
        if not isinstance(values, tuple) or len(values) < 2:
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_CONTACT,
                "contact_reader",
                "contact primitive must return inner and outer force first",
            )
        inner_force = _tensor_value(values[0]).to(torch.float32)
        outer_force = _tensor_value(values[1]).to(torch.float32)
        force = torch.stack((inner_force, outer_force), dim=-1)
        inner_time = self._contact_timestamp(self.inner_contact, episode_time_s)
        outer_time = self._contact_timestamp(self.outer_contact, episode_time_s)
        if bool(
            torch.any(
                torch.abs(inner_time - outer_time)
                > self.config.max_contact_pair_skew_s
            ).item()
        ):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "contact.timestamp_s",
                "inner/outer contact samples are not synchronized",
            )
        timestamp = torch.minimum(inner_time, outer_time)
        return ContactFrame(
            inner_outer_contact=force > self.config.contact_force_threshold_n,
            measurement_valid=torch.isfinite(force),
            timestamp_s=timestamp,
            source=(
                "isaaclab_contact_sensor:"
                f"{CONTACT_CHANNEL_SEMANTICS[0]},{CONTACT_CHANNEL_SEMANTICS[1]}"
            ),
            inner_outer_force_n=force,
        )

    def capture_contact_frame(self) -> ContactFrame:
        """Read a fresh validated contact outcome without advancing history.

        This is intended for the post-action outcome side of ``(s_t, a_t,
        outcome_{t+1})`` recording.  It deliberately does not mutate
        ``_previous_packet``; the next full capture remains responsible for
        advancing the recurrent sensor cadence exactly once.
        """

        _, episode_time = self._episode_clock()
        contact = self._contact_frame(episode_time_s=episode_time)
        return self.validator.validate_contact_frame(
            contact,
            packet_time_s=episode_time,
        )

    def _action_record(
        self,
        *,
        tensor_name: ActionTensorName,
        value: torch.Tensor,
        source_timestamp_s: torch.Tensor,
        accepted_timestamp_s: torch.Tensor,
        episode_time_s: torch.Tensor,
        source: str,
    ) -> ActionTimestampRecord:
        self.contract.validate_payload({tensor_name: value})
        batch_size = episode_time_s.shape[0]
        for name, timestamp in (
            ("action.source_timestamp_s", source_timestamp_s),
            ("action.accepted_timestamp_s", accepted_timestamp_s),
        ):
            if timestamp.shape != (batch_size,) or not bool(
                torch.isfinite(timestamp).all()
            ):
                raise SensorIntegrityError(
                    SensorFaultCode.INVALID_TIMESTAMP,
                    name,
                    f"expected finite shape {(batch_size,)}",
                )
        if not source.strip():
            raise ValueError("action source is required")
        if bool(torch.any(source_timestamp_s < 0.0).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "action.source_timestamp_s",
                "action source timestamp must be non-negative",
            )
        if bool(torch.any(source_timestamp_s > accepted_timestamp_s).item()):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "action.source_timestamp_s",
                "action source timestamp is later than acceptance",
            )
        if bool(
            torch.any(
                accepted_timestamp_s - source_timestamp_s
                > self.config.max_action_source_age_s
            ).item()
        ):
            raise SensorIntegrityError(
                SensorFaultCode.STALE_FRAME,
                "action.source_timestamp_s",
                "action command exceeded maximum source-to-acceptance age",
            )
        if bool(
            torch.any(
                torch.abs(accepted_timestamp_s - episode_time_s)
                > self.config.action_clock_tolerance_s
            ).item()
        ):
            raise SensorIntegrityError(
                SensorFaultCode.INVALID_TIMESTAMP,
                "action.accepted_timestamp_s",
                "action acceptance is not aligned with the pre-action sensor packet",
            )
        return ActionTimestampRecord(
            tensor_name=tensor_name,
            value=value.clone(),
            source_timestamp_s=source_timestamp_s.clone(),
            accepted_timestamp_s=accepted_timestamp_s.clone(),
            source=source,
        )

    def accept_action(
        self,
        *,
        action_tensor_name: ActionTensorName,
        action_value: torch.Tensor,
        action_source_timestamp_s: torch.Tensor,
        action_source: str,
        action_accepted_timestamp_s: torch.Tensor | None = None,
    ) -> ActionTimestampRecord:
        """Bind ``a_t`` to the current pre-step control clock without capture."""

        _, episode_time = self._episode_clock()
        accepted_time = (
            episode_time
            if action_accepted_timestamp_s is None
            else action_accepted_timestamp_s
        )
        return self._action_record(
            tensor_name=action_tensor_name,
            value=action_value,
            source_timestamp_s=action_source_timestamp_s,
            accepted_timestamp_s=accepted_time,
            episode_time_s=episode_time,
            source=action_source,
        )

    def _capture_state_at(
        self, control_step: torch.Tensor, episode_time: torch.Tensor
    ) -> IsaacStateCapture:
        """Capture state for an already-resolved episode clock."""

        camera_timestamp, _ = self._camera_timing(episode_time)
        transforms = {
            "head": self._camera_transform("head"),
            "right_wrist": self._camera_transform("right_wrist"),
        }
        robot, pad_midpoint = self._robot_state(timestamp_s=episode_time)
        head = self._camera_frame(
            self.head,
            name="head",
            timestamp_s=camera_timestamp[:, 0],
            root_position_world_m=robot.root_position_world_m,
            root_quaternion_world_xyzw=robot.root_quaternion_world_xyzw,
        )
        wrist = self._camera_frame(
            self.wrist,
            name="right_wrist",
            timestamp_s=camera_timestamp[:, 1],
            root_position_world_m=robot.root_position_world_m,
            root_quaternion_world_xyzw=robot.root_quaternion_world_xyzw,
        )
        contact = self._contact_frame(episode_time_s=episode_time)
        packet = SynchronizedSensorPacket(
            timestamp_s=episode_time,
            control_step=control_step,
            head=head,
            wrist=wrist,
            robot=robot,
            contact=contact,
            distal_pad_midpoint=pad_midpoint,
        )
        view = self.validator.deployable_view(
            packet, previous=self._previous_packet
        )
        self._previous_packet = packet
        return IsaacStateCapture(
            packet=packet,
            deployable_sensors=view,
            camera_world_transform=transforms,
        )

    def capture_state(self) -> IsaacStateCapture:
        """Capture post-step ``s_{t+1}`` without relabeling the prior action."""

        control_step, episode_time = self._episode_clock()
        return self._capture_state_at(control_step, episode_time)

    def capture_pre_action(
        self,
        *,
        action_tensor_name: ActionTensorName,
        action_value: torch.Tensor,
        action_source_timestamp_s: torch.Tensor,
        action_source: str,
        action_accepted_timestamp_s: torch.Tensor | None = None,
    ) -> IsaacAdapterCapture:
        """Atomically capture ``s_t`` and bind ``a_t`` before stepping.

        New runtime code should call :meth:`accept_action` before ``env.step``
        and :meth:`capture_state` afterwards. This combined method remains for
        offline diagnostics and existing callers which need one pre-step row.
        """

        control_step, episode_time = self._episode_clock()
        accepted_time = (
            episode_time
            if action_accepted_timestamp_s is None
            else action_accepted_timestamp_s
        )
        action = self._action_record(
            tensor_name=action_tensor_name,
            value=action_value,
            source_timestamp_s=action_source_timestamp_s,
            accepted_timestamp_s=accepted_time,
            episode_time_s=episode_time,
            source=action_source,
        )
        state = self._capture_state_at(control_step, episode_time)
        return IsaacAdapterCapture(
            packet=state.packet,
            deployable_sensors=state.deployable_sensors,
            camera_world_transform=state.camera_world_transform,
            contact_channel_semantics=state.contact_channel_semantics,
            action=action,
        )

    def build_observation(
        self,
        capture: IsaacStateCapture,
        vision: VisionGeometryPrediction,
        *,
        previous_action: torch.Tensor | None = None,
    ) -> IsaacAdapterObservation:
        observation = self.observation_builder.build(
            capture.deployable_sensors,
            vision,
            previous_action=previous_action,
        )
        return IsaacAdapterObservation(capture=capture, observation=observation)


def static_adapter_contract() -> dict[str, object]:
    """Read-only deployment probe data; safe without Isaac installed."""

    config = IsaacRuntimeAdapterConfig()
    return {
        "scene_keys": {
            "head_camera": config.head_camera_scene_key,
            "right_wrist_camera": config.wrist_camera_scene_key,
            "robot": config.robot_scene_key,
            "ee_frame": config.ee_frame_scene_key,
            "inner_contact": config.inner_contact_scene_key,
            "outer_contact": config.outer_contact_scene_key,
        },
        "controlled_joints": [*RIGHT_ARM_JOINTS, RIGHT_GRIPPER_MASTER],
        "full_joint_order": list(G2_RUNTIME_JOINT_ORDER),
        "ee_body": RIGHT_EE_BODY,
        "distal_links": [RIGHT_INNER_DISTAL_LINK, RIGHT_OUTER_DISTAL_LINK],
        "contact_channel_semantics": list(CONTACT_CHANNEL_SEMANTICS),
        "requires_pad_surface_calibration": True,
        "capture_alignment": "separate_pre_action_acceptance_and_post_action_state",
    }


def main(argv: list[str] | None = None) -> int:
    """Emit the read-only adapter binding contract without launching Isaac."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-probe",
        action="store_true",
        help="print expected scene/body/channel bindings as JSON",
    )
    arguments = parser.parse_args(argv)
    if not arguments.static_probe:
        parser.error("--static-probe is required; this CLI never launches Isaac")
    print(json.dumps(static_adapter_contract(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "ActionTimestampRecord",
    "CONTACT_CHANNEL_SEMANTICS",
    "G2IsaacRuntimeAdapter",
    "IsaacAdapterCapture",
    "IsaacStateCapture",
    "IsaacAdapterObservation",
    "IsaacRuntimeAdapterConfig",
    "PadSurfaceCalibration",
    "calibrated_pad_surface_midpoint_root_m",
    "calibrated_pad_surface_midpoint_kinematics_root",
    "main",
    "static_adapter_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
