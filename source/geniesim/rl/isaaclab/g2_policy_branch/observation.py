# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Deployment-compatible observations for the isolated high-level policy branch.

The contract has no simulator ground-truth cube pose, contact label, joint
target, or actuator value.  If relative grasp geometry is supplied, it must
already be a deployable estimate: a vision-predicted cube position minus live
distal-pad midpoint FK.  Phase labels are deliberately dataset/evaluation
metadata and never appear in this actor payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Iterable

import torch
from torch import Tensor

from .action_interface import (
    CartesianResidualScale,
    GripperHysteresisConfig,
    PolicyActionMode,
)


POLICY_OBSERVATION_SCHEMA = "g2_policy_branch_observation_v2"
G2_RIGHT_ARM_JOINT_STATE_LAYOUT = "g2_right_arm_idx61_to_idx67_rad_v1"
ABSTRACT_GRIPPER_STATE_ENCODING = "abstract_open_0_closed_1_v1"
RGB_ENCODING = "uint8_0_255_or_float_0_1_v1"
POLICY_ACTOR_INPUTS = (
    "right_wrist_rgb",
    "right_wrist_depth_m",
    "right_wrist_depth_valid",
    "ee_pose_robot_root_xyzw",
    "arm_joint_position_rad",
    "arm_joint_velocity_rad_s",
    "current_gripper_state",
    "previous_policy_action",
)
POLICY_RELATIVE_GRASP_ACTOR_INPUT = "predicted_relative_grasp_xyz_robot_root_m"
POLICY_HEAD_ACTOR_INPUTS = (
    "head_rgb",
    "head_depth_m",
    "head_depth_valid",
)


class PolicyObservationContractError(ValueError):
    """Raised for an observation that cannot safely enter the actor."""


class RelativeGraspProvenance(str, Enum):
    """Only permitted provenance for the optional relative grasp feature."""

    VISION_PREDICTED_CUBE_MINUS_LIVE_DISTAL_PAD_FK = (
        "VISION_PREDICTED_CUBE_MINUS_LIVE_DISTAL_PAD_FK"
    )


@dataclass(frozen=True)
class PolicyObservationConfig:
    """Observation widths and optional camera use for one policy checkpoint."""

    action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D
    arm_joint_state_dim: int = 7
    arm_joint_state_layout: str = G2_RIGHT_ARM_JOINT_STATE_LAYOUT
    gripper_state_encoding: str = ABSTRACT_GRIPPER_STATE_ENCODING
    rgb_encoding: str = RGB_ENCODING
    use_head_camera: bool = False
    use_predicted_relative_grasp: bool = False
    maximum_depth_m: float = 2.0
    ee_quaternion_norm_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        if not isinstance(self.action_mode, PolicyActionMode):
            raise PolicyObservationContractError("action_mode must be PolicyActionMode")
        if type(self.arm_joint_state_dim) is not int or self.arm_joint_state_dim <= 0:
            raise PolicyObservationContractError("arm_joint_state_dim must be positive")
        if self.arm_joint_state_layout != G2_RIGHT_ARM_JOINT_STATE_LAYOUT:
            raise PolicyObservationContractError("unsupported right-arm joint-state layout")
        if self.gripper_state_encoding != ABSTRACT_GRIPPER_STATE_ENCODING:
            raise PolicyObservationContractError("unsupported abstract gripper-state encoding")
        if self.rgb_encoding != RGB_ENCODING:
            raise PolicyObservationContractError("unsupported RGB encoding")
        if type(self.use_head_camera) is not bool or type(self.use_predicted_relative_grasp) is not bool:
            raise PolicyObservationContractError("observation feature switches must be bool")
        if not math.isfinite(self.maximum_depth_m) or self.maximum_depth_m <= 0.0:
            raise PolicyObservationContractError("maximum_depth_m must be positive")
        if (
            not math.isfinite(self.ee_quaternion_norm_tolerance)
            or self.ee_quaternion_norm_tolerance <= 0.0
        ):
            raise PolicyObservationContractError("ee_quaternion_norm_tolerance must be positive")

    @property
    def previous_action_dim(self) -> int:
        return self.action_mode.dimension


@dataclass(frozen=True)
class PolicyDataSemantics:
    """Serialized action/observation meaning shared by P1 data and P2 BC.

    Architecture width is intentionally not included here; this contract covers
    the physical and observation interpretation of a row.  Checkpoint metadata
    adds the architecture fingerprint separately.
    """

    action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D
    arm_joint_state_dim: int = 7
    arm_joint_state_layout: str = G2_RIGHT_ARM_JOINT_STATE_LAYOUT
    gripper_state_encoding: str = ABSTRACT_GRIPPER_STATE_ENCODING
    rgb_encoding: str = RGB_ENCODING
    residual_scale: CartesianResidualScale = CartesianResidualScale()
    gripper_hysteresis: GripperHysteresisConfig = GripperHysteresisConfig()
    use_head_camera: bool = False
    use_predicted_relative_grasp: bool = False
    maximum_depth_m: float = 2.0
    ee_quaternion_norm_tolerance: float = 1.0e-3
    observation_schema: str = POLICY_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.action_mode, PolicyActionMode):
            raise PolicyObservationContractError("data semantics action_mode is invalid")
        if not isinstance(self.residual_scale, CartesianResidualScale):
            raise PolicyObservationContractError("data semantics residual_scale is invalid")
        if not isinstance(self.gripper_hysteresis, GripperHysteresisConfig):
            raise PolicyObservationContractError("data semantics gripper_hysteresis is invalid")
        if type(self.arm_joint_state_dim) is not int or self.arm_joint_state_dim <= 0:
            raise PolicyObservationContractError("data semantics arm_joint_state_dim must be positive")
        if self.arm_joint_state_layout != G2_RIGHT_ARM_JOINT_STATE_LAYOUT:
            raise PolicyObservationContractError("data semantics has unsupported arm-state layout")
        if self.gripper_state_encoding != ABSTRACT_GRIPPER_STATE_ENCODING:
            raise PolicyObservationContractError("data semantics has unsupported gripper-state encoding")
        if self.rgb_encoding != RGB_ENCODING:
            raise PolicyObservationContractError("data semantics has unsupported RGB encoding")
        if type(self.use_head_camera) is not bool or type(self.use_predicted_relative_grasp) is not bool:
            raise PolicyObservationContractError("data semantics feature switches must be bool")
        if (
            not math.isfinite(self.maximum_depth_m)
            or self.maximum_depth_m <= 0.0
            or not math.isfinite(self.ee_quaternion_norm_tolerance)
            or self.ee_quaternion_norm_tolerance <= 0.0
        ):
            raise PolicyObservationContractError("data semantics numeric bounds must be positive")
        if self.observation_schema != POLICY_OBSERVATION_SCHEMA:
            raise PolicyObservationContractError("data semantics observation schema is unsupported")

    @property
    def observation_config(self) -> PolicyObservationConfig:
        return PolicyObservationConfig(
            action_mode=self.action_mode,
            arm_joint_state_dim=self.arm_joint_state_dim,
            arm_joint_state_layout=self.arm_joint_state_layout,
            gripper_state_encoding=self.gripper_state_encoding,
            rgb_encoding=self.rgb_encoding,
            use_head_camera=self.use_head_camera,
            use_predicted_relative_grasp=self.use_predicted_relative_grasp,
            maximum_depth_m=self.maximum_depth_m,
            ee_quaternion_norm_tolerance=self.ee_quaternion_norm_tolerance,
        )

    def fingerprint(self) -> str:
        payload = {
            "action_mode": self.action_mode.value,
            "arm_joint_state_dim": self.arm_joint_state_dim,
            "arm_joint_state_layout": self.arm_joint_state_layout,
            "gripper_state_encoding": self.gripper_state_encoding,
            "rgb_encoding": self.rgb_encoding,
            "translation_m_per_normalized": self.residual_scale.translation_m_per_normalized,
            "rotation_rad_per_normalized": self.residual_scale.rotation_rad_per_normalized,
            "control_frame": self.residual_scale.control_frame.value,
            "gripper_open_threshold": self.gripper_hysteresis.open_threshold,
            "gripper_close_threshold": self.gripper_hysteresis.close_threshold,
            "use_head_camera": self.use_head_camera,
            "use_predicted_relative_grasp": self.use_predicted_relative_grasp,
            "maximum_depth_m": self.maximum_depth_m,
            "ee_quaternion_norm_tolerance": self.ee_quaternion_norm_tolerance,
            "observation_schema": self.observation_schema,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_compatible(self, other: "PolicyDataSemantics") -> None:
        if not isinstance(other, PolicyDataSemantics) or self.fingerprint() != other.fingerprint():
            raise PolicyObservationContractError(
                "policy data semantics mismatch between collection and training"
            )


def _finite(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value):
        raise PolicyObservationContractError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise PolicyObservationContractError(f"{name} must be finite")


def _same_batch_time(
    reference: Tensor, values: Iterable[tuple[str, Tensor]]
) -> None:
    for name, value in values:
        if value.shape[:2] != reference.shape[:2]:
            raise PolicyObservationContractError(
                f"{name} must share [batch,time] with right_wrist_rgb"
            )


@dataclass(frozen=True)
class PolicyObservation:
    """One batched temporal actor input sequence.

    Image tensors are ``[B,T,H,W,C]``.  Vector tensors are ``[B,T,F]``.  The
    explicitly named ``predicted_*`` geometry prevents accidentally putting a
    ground-truth cube field into the actor surface.
    """

    right_wrist_rgb: Tensor
    right_wrist_depth_m: Tensor
    right_wrist_depth_valid: Tensor
    ee_pose_robot_root_xyzw: Tensor
    arm_joint_position_rad: Tensor
    arm_joint_velocity_rad_s: Tensor
    current_gripper_state: Tensor
    previous_policy_action: Tensor
    data_semantics: PolicyDataSemantics
    data_semantic_fingerprint: str
    predicted_relative_grasp_xyz_robot_root_m: Tensor | None = None
    relative_grasp_provenance: RelativeGraspProvenance | None = None
    right_wrist_frame_age_s: Tensor | None = None
    head_rgb: Tensor | None = None
    head_depth_m: Tensor | None = None
    head_depth_valid: Tensor | None = None
    head_frame_age_s: Tensor | None = None
    hidden_reset_mask: Tensor | None = None

    def actor_payload(
        self,
        *,
        include_head_camera: bool = False,
        include_predicted_relative_grasp: bool = False,
    ) -> dict[str, Tensor]:
        """Return only deployable actor inputs.

        Head fields are excluded by default, matching the Wrist-first baseline.
        A caller enabling the optional Head branch must opt in explicitly and
        provide a complete Head RGB-D triplet.
        """

        payload = {name: getattr(self, name) for name in POLICY_ACTOR_INPUTS}
        if include_predicted_relative_grasp:
            if (
                self.predicted_relative_grasp_xyz_robot_root_m is None
                or self.relative_grasp_provenance
                is not RelativeGraspProvenance.VISION_PREDICTED_CUBE_MINUS_LIVE_DISTAL_PAD_FK
            ):
                raise PolicyObservationContractError(
                    "relative grasp actor payload requires deployable provenance"
                )
            payload[POLICY_RELATIVE_GRASP_ACTOR_INPUT] = (
                self.predicted_relative_grasp_xyz_robot_root_m
            )
        if include_head_camera:
            if self.head_rgb is None or self.head_depth_m is None or self.head_depth_valid is None:
                raise PolicyObservationContractError("Head actor payload requested without Head RGB-D")
            payload.update(
                {
                    "head_rgb": self.head_rgb,
                    "head_depth_m": self.head_depth_m,
                    "head_depth_valid": self.head_depth_valid,
                }
            )
        return payload

    def assert_compatible_data_semantics(
        self, expected: PolicyDataSemantics
    ) -> None:
        """Bind an observation row to the semantic contract used to create it."""

        if self.data_semantic_fingerprint != self.data_semantics.fingerprint():
            raise PolicyObservationContractError(
                "observation semantic fingerprint does not match its data semantics"
            )
        self.data_semantics.assert_compatible(expected)

    def validate(self, config: PolicyObservationConfig) -> None:
        if self.data_semantic_fingerprint != self.data_semantics.fingerprint():
            raise PolicyObservationContractError(
                "observation semantic fingerprint does not match its data semantics"
            )
        if self.data_semantics.observation_config != config:
            raise PolicyObservationContractError(
                "observation semantics and validation config differ"
            )
        rgb = self.right_wrist_rgb
        if rgb.ndim != 5 or rgb.shape[-1] != 3:
            raise PolicyObservationContractError(
                "right_wrist_rgb must be [batch,time,height,width,3]"
            )
        if rgb.dtype not in (torch.uint8, torch.float16, torch.float32, torch.float64):
            raise PolicyObservationContractError("right_wrist_rgb has unsupported dtype")
        depth = self.right_wrist_depth_m
        valid = self.right_wrist_depth_valid
        if depth.shape != (*rgb.shape[:4], 1):
            raise PolicyObservationContractError(
                "right_wrist_depth_m must align with RGB and have one channel"
            )
        if valid.shape != depth.shape or valid.dtype is not torch.bool:
            raise PolicyObservationContractError(
                "right_wrist_depth_valid must be boolean and match depth"
            )
        _finite("right_wrist_depth_m", depth)
        if bool((depth[valid] < 0.0).any()) or bool(
            (depth[valid] > config.maximum_depth_m).any()
        ):
            raise PolicyObservationContractError("valid wrist depth is outside configured range")
        if torch.is_floating_point(rgb):
            _finite("right_wrist_rgb", rgb)
            if bool((rgb < 0.0).any()) or bool((rgb > 1.0).any()):
                raise PolicyObservationContractError(
                    "floating right_wrist_rgb must be encoded in [0, 1]"
                )

        expected = [
            ("ee_pose_robot_root_xyzw", self.ee_pose_robot_root_xyzw),
            ("arm_joint_position_rad", self.arm_joint_position_rad),
            ("arm_joint_velocity_rad_s", self.arm_joint_velocity_rad_s),
            ("current_gripper_state", self.current_gripper_state),
            ("previous_policy_action", self.previous_policy_action),
        ]
        if config.use_predicted_relative_grasp:
            if (
                self.predicted_relative_grasp_xyz_robot_root_m is None
                or self.relative_grasp_provenance
                is not RelativeGraspProvenance.VISION_PREDICTED_CUBE_MINUS_LIVE_DISTAL_PAD_FK
            ):
                raise PolicyObservationContractError(
                    "relative grasp feature requires vision-plus-live-FK provenance"
                )
            expected.append(
                (
                    "predicted_relative_grasp_xyz_robot_root_m",
                    self.predicted_relative_grasp_xyz_robot_root_m,
                )
            )
        elif (
            self.predicted_relative_grasp_xyz_robot_root_m is not None
            or self.relative_grasp_provenance is not None
        ):
            raise PolicyObservationContractError(
                "relative grasp feature is disabled for this policy schema"
            )
        _same_batch_time(rgb, expected)
        expected_widths = [
            ("ee_pose_robot_root_xyzw", self.ee_pose_robot_root_xyzw, 7),
            ("arm_joint_position_rad", self.arm_joint_position_rad, config.arm_joint_state_dim),
            ("arm_joint_velocity_rad_s", self.arm_joint_velocity_rad_s, config.arm_joint_state_dim),
            ("current_gripper_state", self.current_gripper_state, 1),
            ("previous_policy_action", self.previous_policy_action, config.previous_action_dim),
        ]
        if config.use_predicted_relative_grasp:
            assert self.predicted_relative_grasp_xyz_robot_root_m is not None
            expected_widths.append(
                (
                    "predicted_relative_grasp_xyz_robot_root_m",
                    self.predicted_relative_grasp_xyz_robot_root_m,
                    3,
                )
            )
        for name, value, width in expected_widths:
            if value.ndim != 3 or value.shape[-1] != width:
                raise PolicyObservationContractError(f"{name} must end in width {width}")
            _finite(name, value)
        if bool((self.current_gripper_state < 0.0).any()) or bool(
            (self.current_gripper_state > 1.0).any()
        ):
            raise PolicyObservationContractError("current_gripper_state must be in [0, 1]")
        is_open = torch.isclose(
            self.current_gripper_state,
            torch.zeros((), device=self.current_gripper_state.device, dtype=self.current_gripper_state.dtype),
        )
        is_closed = torch.isclose(
            self.current_gripper_state,
            torch.ones((), device=self.current_gripper_state.device, dtype=self.current_gripper_state.dtype),
        )
        if not bool((is_open | is_closed).all()):
            raise PolicyObservationContractError(
                "current_gripper_state must use abstract OPEN=0/CLOSED=1 encoding"
            )
        if bool((self.previous_policy_action[..., -1] < 0.0).any()) or bool(
            (self.previous_policy_action[..., -1] > 1.0).any()
        ):
            raise PolicyObservationContractError("previous gripper probability must be in [0, 1]")
        if bool((self.previous_policy_action[..., :-1].abs() > 1.0).any()):
            raise PolicyObservationContractError("previous EE action must be normalised")

        quaternion_norm = torch.linalg.vector_norm(
            self.ee_pose_robot_root_xyzw[..., 3:], dim=-1
        )
        if bool(
            (torch.abs(quaternion_norm - 1.0) > config.ee_quaternion_norm_tolerance).any()
        ):
            raise PolicyObservationContractError("EE quaternion must be unit-norm within tolerance")
        if self.right_wrist_frame_age_s is not None:
            if self.right_wrist_frame_age_s.shape != (*rgb.shape[:2], 1):
                raise PolicyObservationContractError("right_wrist_frame_age_s must be [B,T,1]")
            _finite("right_wrist_frame_age_s", self.right_wrist_frame_age_s)
            if bool((self.right_wrist_frame_age_s < 0.0).any()):
                raise PolicyObservationContractError("right_wrist_frame_age_s cannot be negative")
        if self.hidden_reset_mask is not None:
            if self.hidden_reset_mask.shape != rgb.shape[:2] or self.hidden_reset_mask.dtype is not torch.bool:
                raise PolicyObservationContractError("hidden_reset_mask must be bool [B,T]")

        head_fields = (self.head_rgb, self.head_depth_m, self.head_depth_valid)
        if config.use_head_camera:
            if any(value is None for value in head_fields):
                raise PolicyObservationContractError("Head camera is enabled but Head RGB-D is missing")
            assert self.head_rgb is not None
            assert self.head_depth_m is not None
            assert self.head_depth_valid is not None
            if self.head_rgb.ndim != 5 or self.head_rgb.shape[-1] != 3:
                raise PolicyObservationContractError("head_rgb must be [B,T,H,W,3]")
            if self.head_rgb.dtype not in (
                torch.uint8,
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                raise PolicyObservationContractError("head_rgb has unsupported dtype")
            if self.head_depth_m.shape != (*self.head_rgb.shape[:4], 1):
                raise PolicyObservationContractError("head_depth_m must align with head_rgb")
            if self.head_depth_valid.shape != self.head_depth_m.shape or self.head_depth_valid.dtype is not torch.bool:
                raise PolicyObservationContractError("head_depth_valid must be boolean and match head depth")
            _same_batch_time(rgb, (("head_rgb", self.head_rgb),))
            _finite("head_depth_m", self.head_depth_m)
            if bool((self.head_depth_m[self.head_depth_valid] < 0.0).any()) or bool(
                (self.head_depth_m[self.head_depth_valid] > config.maximum_depth_m).any()
            ):
                raise PolicyObservationContractError("valid Head depth is outside configured range")
            if torch.is_floating_point(self.head_rgb):
                _finite("head_rgb", self.head_rgb)
                if bool((self.head_rgb < 0.0).any()) or bool((self.head_rgb > 1.0).any()):
                    raise PolicyObservationContractError(
                        "floating head_rgb must be encoded in [0, 1]"
                    )
            if self.head_frame_age_s is not None:
                if self.head_frame_age_s.shape != (*rgb.shape[:2], 1):
                    raise PolicyObservationContractError("head_frame_age_s must be [B,T,1]")
                _finite("head_frame_age_s", self.head_frame_age_s)
                if bool((self.head_frame_age_s < 0.0).any()):
                    raise PolicyObservationContractError("head_frame_age_s cannot be negative")


__all__ = [
    "ABSTRACT_GRIPPER_STATE_ENCODING",
    "G2_RIGHT_ARM_JOINT_STATE_LAYOUT",
    "POLICY_ACTOR_INPUTS",
    "POLICY_HEAD_ACTOR_INPUTS",
    "POLICY_RELATIVE_GRASP_ACTOR_INPUT",
    "POLICY_OBSERVATION_SCHEMA",
    "RGB_ENCODING",
    "PolicyDataSemantics",
    "PolicyObservation",
    "PolicyObservationConfig",
    "PolicyObservationContractError",
    "RelativeGraspProvenance",
]
