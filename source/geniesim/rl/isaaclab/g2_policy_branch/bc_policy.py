# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Small recurrent Behavioral-Cloning baseline for the high-level policy branch.

The model receives Wrist RGB-D and deployment-compatible state, then emits an
EE residual plus a continuous gripper-close probability.  It intentionally has
no articulation, individual-finger, motor, force, CAN, or ground-truth cube
input.  Its output must still pass through :mod:`action_interface` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .action_interface import (
    CartesianResidualScale,
    GripperHysteresisConfig,
    POLICY_ACTION_SCHEMA_4D,
    POLICY_ACTION_SCHEMA_7D,
    PolicyActionMode,
)
from .observation import (
    PolicyDataSemantics,
    PolicyObservation,
    PolicyObservationConfig,
    PolicyObservationContractError,
)


HIGH_LEVEL_POLICY_MODEL_SCHEMA = "g2_high_level_policy_bc_v2"


class PolicyBCContractError(ValueError):
    """Raised for incompatible high-level BC data, labels, or checkpoint metadata."""


def tensor_state_sha256(module: nn.Module) -> str:
    """Stable, device-independent state hash for freeze/lineage evidence."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _visual_state_sha256(
    wrist_encoder: nn.Module, head_encoder: nn.Module | None
) -> str:
    """Hash every visual tensor, including opt-in Head-camera weights."""

    digest = hashlib.sha256()
    modules = (("wrist", wrist_encoder), ("head", head_encoder))
    for prefix, module in modules:
        if module is None:
            continue
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(prefix.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicyBCConfig:
    """Architecture configuration tied to one new branch-local action schema."""

    action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D
    arm_joint_state_dim: int = 7
    use_head_camera: bool = False
    use_predicted_relative_grasp: bool = False
    vision_feature_dim: int = 48
    state_feature_dim: int = 64
    fusion_dim: int = 96
    gru_hidden_dim: int = 128
    gru_layers: int = 1
    maximum_depth_m: float = 2.0
    ee_quaternion_norm_tolerance: float = 1.0e-3
    residual_scale: CartesianResidualScale = CartesianResidualScale()
    gripper_hysteresis: GripperHysteresisConfig = GripperHysteresisConfig()

    def __post_init__(self) -> None:
        for name in (
            "arm_joint_state_dim",
            "vision_feature_dim",
            "state_feature_dim",
            "fusion_dim",
            "gru_hidden_dim",
            "gru_layers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise PolicyBCContractError(f"{name} must be positive")
        if not math.isfinite(self.maximum_depth_m) or self.maximum_depth_m <= 0.0:
            raise PolicyBCContractError("maximum_depth_m must be positive")
        if (
            not math.isfinite(self.ee_quaternion_norm_tolerance)
            or self.ee_quaternion_norm_tolerance <= 0.0
        ):
            raise PolicyBCContractError("ee_quaternion_norm_tolerance must be positive")
        if type(self.use_head_camera) is not bool or type(self.use_predicted_relative_grasp) is not bool:
            raise PolicyBCContractError("BC observation feature switches must be bool")
        # Constructing the shared semantic contract here validates the action
        # frame/scale, gripper encoding, and canonical right-arm state layout.
        _ = self.data_semantics

    @property
    def action_dim(self) -> int:
        return self.action_mode.dimension

    @property
    def action_schema(self) -> str:
        return (
            POLICY_ACTION_SCHEMA_4D
            if self.action_mode is PolicyActionMode.TRANSLATION_GRIPPER_4D
            else POLICY_ACTION_SCHEMA_7D
        )

    @property
    def data_semantics(self) -> PolicyDataSemantics:
        return PolicyDataSemantics(
            action_mode=self.action_mode,
            arm_joint_state_dim=self.arm_joint_state_dim,
            use_head_camera=self.use_head_camera,
            use_predicted_relative_grasp=self.use_predicted_relative_grasp,
            maximum_depth_m=self.maximum_depth_m,
            ee_quaternion_norm_tolerance=self.ee_quaternion_norm_tolerance,
            residual_scale=self.residual_scale,
            gripper_hysteresis=self.gripper_hysteresis,
        )

    @property
    def observation_config(self) -> PolicyObservationConfig:
        return self.data_semantics.observation_config

    def semantic_fingerprint(self) -> str:
        """Hash every architecture and runtime semantic needed for deployment."""

        payload = {
            "data_semantics_fingerprint": self.data_semantics.fingerprint(),
            "vision_feature_dim": self.vision_feature_dim,
            "state_feature_dim": self.state_feature_dim,
            "fusion_dim": self.fusion_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_layers": self.gru_layers,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_compatible_data_semantics(
        self, semantics: PolicyDataSemantics
    ) -> None:
        try:
            self.data_semantics.assert_compatible(semantics)
        except PolicyObservationContractError as error:
            raise PolicyBCContractError(
                "P1 collection and P2 BC data semantics mismatch"
            ) from error


@dataclass(frozen=True)
class PolicyCheckpointMetadata:
    """Strictly prevents old 7-D legacy checkpoints from being loaded by accident."""

    model_schema: str
    action_schema: str
    action_mode: str
    semantic_fingerprint: str

    @classmethod
    def for_config(cls, config: PolicyBCConfig) -> "PolicyCheckpointMetadata":
        return cls(
            model_schema=HIGH_LEVEL_POLICY_MODEL_SCHEMA,
            action_schema=config.action_schema,
            action_mode=config.action_mode.value,
            semantic_fingerprint=config.semantic_fingerprint(),
        )

    def assert_compatible(self, config: PolicyBCConfig) -> None:
        expected = self.for_config(config)
        if self != expected:
            raise PolicyBCContractError(
                "high-level policy checkpoint schema mismatch; legacy or different "
                "action-mode weights must not be silently reused"
            )


class RGBDEncoder(nn.Module):
    """Compact Wrist/Head RGB-D encoder with an explicit validity channel."""

    def __init__(self, output_dim: int, *, maximum_depth_m: float) -> None:
        super().__init__()
        self.maximum_depth_m = float(maximum_depth_m)
        self.backbone = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(48, output_dim)

    def forward(self, rgb: Tensor, depth_m: Tensor, depth_valid: Tensor) -> Tensor:
        # Validation occurs once at the PolicyObservation boundary.  This
        # method only arranges B,T image data into NCHW for the encoder.
        batch, time, height, width, _ = rgb.shape
        rgb_float = rgb.to(dtype=torch.float32)
        if rgb.dtype == torch.uint8:
            rgb_float = rgb_float / 255.0
        rgb_float = rgb_float.clamp(0.0, 1.0)
        valid = depth_valid.to(dtype=torch.float32)
        depth = torch.where(depth_valid, depth_m, torch.zeros_like(depth_m))
        depth = (depth / self.maximum_depth_m).clamp(0.0, 1.0)
        packed = torch.cat((rgb_float, depth, valid), dim=-1)
        packed = packed.permute(0, 1, 4, 2, 3).reshape(
            batch * time, 5, height, width
        )
        features = self.backbone(packed).flatten(start_dim=1)
        return self.projection(features).reshape(batch, time, -1)


@dataclass(frozen=True)
class PolicyNetworkOutput:
    """Normalised action distribution means and recurrent features."""

    policy_action: Tensor
    translation_action: Tensor
    gripper_probability: Tensor
    rotation_action: Tensor | None
    recurrent_features: Tensor
    final_hidden: Tensor


class HighLevelBCPolicy(nn.Module):
    """Fresh temporal policy head behind a separately owned vision encoder."""

    def __init__(self, config: PolicyBCConfig = PolicyBCConfig()) -> None:
        super().__init__()
        self.config = config
        self.wrist_encoder = RGBDEncoder(
            config.vision_feature_dim, maximum_depth_m=config.maximum_depth_m
        )
        self.head_encoder = (
            RGBDEncoder(config.vision_feature_dim, maximum_depth_m=config.maximum_depth_m)
            if config.use_head_camera
            else None
        )
        state_dim = (
            7
            + 2 * config.arm_joint_state_dim
            + 1
            + config.action_dim
            + (3 if config.use_predicted_relative_grasp else 0)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.state_feature_dim),
            nn.SiLU(),
            nn.Linear(config.state_feature_dim, config.state_feature_dim),
            nn.SiLU(),
        )
        vision_dim = config.vision_feature_dim * (2 if config.use_head_camera else 1)
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + config.state_feature_dim, config.fusion_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=config.fusion_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
        )
        self.translation_head = nn.Linear(config.gru_hidden_dim, 3)
        self.rotation_head = (
            nn.Linear(config.gru_hidden_dim, 3)
            if config.action_mode is PolicyActionMode.SE3_GRIPPER_7D
            else None
        )
        self.gripper_head = nn.Linear(config.gru_hidden_dim, 1)

    def visual_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.wrist_encoder.parameters()
        if self.head_encoder is not None:
            yield from self.head_encoder.parameters()

    def controller_parameters(self) -> Iterable[nn.Parameter]:
        modules = (self.state_encoder, self.fusion, self.gru, self.translation_head, self.gripper_head)
        for module in modules:
            yield from module.parameters()
        if self.rotation_head is not None:
            yield from self.rotation_head.parameters()

    def freeze_visual_encoders(self) -> str:
        """Freeze visual features before BC; returns immutable encoder evidence."""

        before = _visual_state_sha256(self.wrist_encoder, self.head_encoder)
        for parameter in self.visual_parameters():
            parameter.requires_grad_(False)
        self.wrist_encoder.eval()
        if self.head_encoder is not None:
            self.head_encoder.eval()
        return before

    def make_bc_optimizer(self, *, learning_rate: float = 3.0e-4) -> torch.optim.Optimizer:
        if any(parameter.requires_grad for parameter in self.visual_parameters()):
            raise RuntimeError(
                "freeze visual encoders before BC so a moving representation "
                "cannot be trained with fresh temporal/action heads"
            )
        parameters = [parameter for parameter in self.controller_parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("no trainable high-level BC controller parameters")
        return torch.optim.Adam(parameters, lr=learning_rate)

    def _state_features(self, observation: PolicyObservation) -> Tensor:
        features = [
            observation.ee_pose_robot_root_xyzw,
            observation.arm_joint_position_rad,
            observation.arm_joint_velocity_rad_s,
            observation.current_gripper_state,
            observation.previous_policy_action,
        ]
        if self.config.use_predicted_relative_grasp:
            assert observation.predicted_relative_grasp_xyz_robot_root_m is not None
            features.append(observation.predicted_relative_grasp_xyz_robot_root_m)
        return torch.cat(features, dim=-1)

    def _run_gru_with_resets(
        self,
        features: Tensor,
        reset_mask: Tensor | None,
        initial_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, time, _ = features.shape
        if initial_hidden is None and (
            reset_mask is None or not bool(reset_mask[:, 0].all())
        ):
            raise PolicyBCContractError(
                "a non-reset GRU row requires an explicit initial hidden state"
            )
        if initial_hidden is not None:
            expected_shape = (self.config.gru_layers, batch, self.config.gru_hidden_dim)
            if initial_hidden.shape != expected_shape:
                raise PolicyBCContractError(
                    "initial GRU hidden state does not match checkpoint batch/layer/width"
                )
            if initial_hidden.device != features.device or initial_hidden.dtype != features.dtype:
                raise PolicyBCContractError(
                    "initial GRU hidden state device or dtype does not match features"
                )
            if not bool(torch.isfinite(initial_hidden).all()):
                raise PolicyBCContractError("initial GRU hidden state must be finite")
        hidden: Tensor | None = initial_hidden
        outputs: list[Tensor] = []
        for step in range(time):
            if hidden is not None and reset_mask is not None:
                reset_rows = reset_mask[:, step]
                if bool(reset_rows.any()):
                    hidden = hidden.clone()
                    hidden[:, reset_rows, :] = 0.0
            output, hidden = self.gru(features[:, step : step + 1], hidden)
            outputs.append(output)
        assert hidden is not None
        return torch.cat(outputs, dim=1), hidden

    def forward(
        self,
        observation: PolicyObservation,
        *,
        initial_hidden: Tensor | None = None,
    ) -> PolicyNetworkOutput:
        try:
            observation.assert_compatible_data_semantics(self.config.data_semantics)
            observation.validate(self.config.observation_config)
        except PolicyObservationContractError as error:
            raise PolicyBCContractError(
                "P1 collection observation and P2 BC semantics mismatch"
            ) from error
        wrist_feature = self.wrist_encoder(
            observation.right_wrist_rgb,
            observation.right_wrist_depth_m,
            observation.right_wrist_depth_valid,
        )
        visual = wrist_feature
        if self.head_encoder is not None:
            assert observation.head_rgb is not None
            assert observation.head_depth_m is not None
            assert observation.head_depth_valid is not None
            head_feature = self.head_encoder(
                observation.head_rgb,
                observation.head_depth_m,
                observation.head_depth_valid,
            )
            visual = torch.cat((visual, head_feature), dim=-1)
        fused = self.fusion(torch.cat((visual, self.state_encoder(self._state_features(observation))), dim=-1))
        recurrent, hidden = self._run_gru_with_resets(
            fused, observation.hidden_reset_mask, initial_hidden
        )
        translation = torch.tanh(self.translation_head(recurrent))
        gripper = torch.sigmoid(self.gripper_head(recurrent))
        rotation = torch.tanh(self.rotation_head(recurrent)) if self.rotation_head is not None else None
        action = torch.cat((translation, gripper), dim=-1) if rotation is None else torch.cat((translation, rotation, gripper), dim=-1)
        return PolicyNetworkOutput(
            policy_action=action,
            translation_action=translation,
            gripper_probability=gripper,
            rotation_action=rotation,
            recurrent_features=recurrent,
            final_hidden=hidden,
        )


@dataclass(frozen=True)
class PolicyBCTargets:
    """BC targets; ``phase_labels`` is logging metadata, not an actor input."""

    policy_action: Tensor
    padding_valid: Tensor
    data_semantics: PolicyDataSemantics
    data_semantic_fingerprint: str
    phase_labels: Tensor | None = None

    @classmethod
    def from_trajectory(
        cls,
        trajectory: "HighLevelPolicyTrajectory",
        config: PolicyBCConfig,
        *,
        padding_valid: Tensor | None = None,
    ) -> "PolicyBCTargets":
        """Create a training batch only after P1/P2 semantic verification.

        The local import keeps the runtime policy module independent of the
        dataset implementation at import time while making the actual
        trajectory-to-training handoff fail closed.
        """

        from .dataset import HighLevelPolicyTrajectory

        if not isinstance(trajectory, HighLevelPolicyTrajectory):
            raise PolicyBCContractError("BC training requires HighLevelPolicyTrajectory")
        trajectory.validate()
        config.assert_compatible_data_semantics(trajectory.data_semantics)
        if padding_valid is None:
            padding_valid = torch.ones(
                trajectory.policy_action.shape[:2],
                dtype=torch.bool,
                device=trajectory.policy_action.device,
            )
        return cls(
            policy_action=trajectory.policy_action,
            padding_valid=padding_valid,
            data_semantics=trajectory.data_semantics,
            data_semantic_fingerprint=trajectory.data_semantic_fingerprint,
        )

    def validate(self, config: PolicyBCConfig) -> None:
        if self.data_semantic_fingerprint != self.data_semantics.fingerprint():
            raise PolicyBCContractError(
                "BC target semantic fingerprint does not match its data semantics"
            )
        config.assert_compatible_data_semantics(self.data_semantics)
        if self.policy_action.ndim != 3 or self.policy_action.shape[-1] != config.action_dim:
            raise PolicyBCContractError("BC action targets do not match policy action schema")
        if not bool(torch.isfinite(self.policy_action).all()):
            raise PolicyBCContractError("BC action targets must be finite")
        if bool((self.policy_action[..., :-1].abs() > 1.0).any()):
            raise PolicyBCContractError("BC EE targets must be normalised")
        if bool((self.policy_action[..., -1] < 0.0).any()) or bool(
            (self.policy_action[..., -1] > 1.0).any()
        ):
            raise PolicyBCContractError("BC gripper targets must be in [0, 1]")
        if self.padding_valid.shape != self.policy_action.shape[:2] or self.padding_valid.dtype is not torch.bool:
            raise PolicyBCContractError("padding_valid must be bool [B,T]")
        if self.phase_labels is not None and self.phase_labels.shape != self.policy_action.shape[:2]:
            raise PolicyBCContractError("phase_labels must be [B,T] metadata")


@dataclass(frozen=True)
class PolicyBCLossConfig:
    translation_weight: float = 1.0
    gripper_weight: float = 1.0
    rotation_weight: float = 1.0
    gripper_semantic_class_balance: bool = False

    def __post_init__(self) -> None:
        for value in (self.translation_weight, self.gripper_weight, self.rotation_weight):
            if value < 0.0:
                raise PolicyBCContractError("BC loss weights must be non-negative")


@dataclass(frozen=True)
class PolicyBCLosses:
    total: Tensor
    translation: Tensor
    gripper: Tensor
    rotation: Tensor
    valid_rows: int
    weighted_contributions: Mapping[str, Tensor]


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    weight = valid.to(dtype=values.dtype)
    denominator = weight.sum().clamp_min(1.0)
    return (values * weight).sum() / denominator


def compute_policy_bc_losses(
    output: PolicyNetworkOutput,
    targets: PolicyBCTargets,
    config: PolicyBCConfig,
    loss_config: PolicyBCLossConfig = PolicyBCLossConfig(),
    *,
    previous_gripper_command_closed: Tensor | None = None,
) -> PolicyBCLosses:
    """Compute only action imitation losses; no phase/FSM authority is learned."""

    targets.validate(config)
    if output.policy_action.shape != targets.policy_action.shape:
        raise PolicyBCContractError("policy output and target shape differ")
    valid = targets.padding_valid
    if not bool(valid.any()):
        raise PolicyBCContractError("BC batch contains no valid rows")
    translation = _masked_mean(
        F.smooth_l1_loss(
            output.translation_action, targets.policy_action[..., :3], reduction="none"
        ).mean(dim=-1),
        valid,
    )
    gripper_rows = F.binary_cross_entropy(
        output.gripper_probability,
        targets.policy_action[..., -1:],
        reduction="none",
    ).squeeze(-1)
    if loss_config.gripper_semantic_class_balance:
        if previous_gripper_command_closed is None:
            raise PolicyBCContractError(
                "semantic gripper balance requires previous gripper command"
            )
        if previous_gripper_command_closed.shape == (*valid.shape, 1):
            previous_gripper_command_closed = previous_gripper_command_closed[..., 0]
        if previous_gripper_command_closed.shape != valid.shape:
            raise PolicyBCContractError(
                "previous gripper command must have shape [B,T] or [B,T,1]"
            )
        previous_closed = previous_gripper_command_closed.to(dtype=torch.bool)
        target_closed = targets.policy_action[..., -1] >= 0.5
        semantic_masks = (
            valid & ~target_closed,
            valid & target_closed & ~previous_closed,
            valid & target_closed & previous_closed,
        )
        present_losses = [
            _masked_mean(gripper_rows, mask)
            for mask in semantic_masks
            if bool(mask.any())
        ]
        if not present_losses:
            raise PolicyBCContractError("BC batch contains no gripper semantic rows")
        # Macro averaging gives OPEN, CLOSE_ONSET, and HOLD_CLOSED equal
        # authority whenever present.  It is frequency-independent and uses
        # no tuned class weight or new decision threshold.
        gripper = torch.stack(present_losses).mean()
    else:
        gripper = _masked_mean(gripper_rows, valid)
    if config.action_mode is PolicyActionMode.SE3_GRIPPER_7D:
        assert output.rotation_action is not None
        rotation = _masked_mean(
            F.smooth_l1_loss(
                output.rotation_action,
                targets.policy_action[..., 3:6],
                reduction="none",
            ).mean(dim=-1),
            valid,
        )
    else:
        rotation = translation.new_zeros(())
    contributions = {
        "translation": translation * loss_config.translation_weight,
        "gripper": gripper * loss_config.gripper_weight,
        "rotation": rotation * loss_config.rotation_weight,
    }
    return PolicyBCLosses(
        total=sum(contributions.values()),
        translation=translation,
        gripper=gripper,
        rotation=rotation,
        valid_rows=int(valid.sum().item()),
        weighted_contributions=contributions,
    )


__all__ = [
    "HIGH_LEVEL_POLICY_MODEL_SCHEMA",
    "HighLevelBCPolicy",
    "PolicyBCConfig",
    "PolicyBCContractError",
    "PolicyBCLossConfig",
    "PolicyBCLosses",
    "PolicyBCTargets",
    "PolicyCheckpointMetadata",
    "PolicyNetworkOutput",
    "RGBDEncoder",
    "compute_policy_bc_losses",
    "tensor_state_sha256",
]
