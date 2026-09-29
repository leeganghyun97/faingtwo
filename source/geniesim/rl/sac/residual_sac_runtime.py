# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Strict runtime loader for the existing three-dimensional SAC actor.

This module does not define another actor architecture.  It binds the existing
``stage2_sac.SquashedGaussianActor`` to the grasp-GRU recurrent feature and
narrows its authority to a robot-root metric XYZ residual.  The resulting
residual is consumed only by :func:`compose_bc_and_residual_action`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from .bc_residual_sac_contract import BCResidualSACStaticConfig
from .keyboard_grasp_contract import canonical_keyboard_grasp_contract
from .stage2_sac import SACConfig, SquashedGaussianActor


RESIDUAL_SAC_ACTOR_CHECKPOINT_SCHEMA = "g2_human_grasp_residual_sac_actor_v1"
RESIDUAL_SAC_OUTPUT_PARAMETERIZATION = "RADIAL_TANH_UNIT_BALL_X_0.0045_M"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResidualSACRuntimeError(ValueError):
    """Raised when residual-SAC runtime provenance or shape is invalid."""


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidualSACActorReceipt:
    checkpoint_path: str
    checkpoint_sha256: str
    actor_state_sha256: str
    human_grasp_checkpoint_sha256: str
    observation_dim: int
    action_dim: int
    output_parameterization: str
    frame: str
    unit: str
    deterministic_inference: bool


@dataclass(frozen=True)
class ResidualSACInference:
    """One immutable actor-forward receipt before the alpha composition.

    ``normalized_action`` is the replay authority used by Stage-1A.  Keeping it
    beside the analytically mapped metric residual prevents a live runner from
    reconstructing or silently changing the actor action after inference.
    """

    normalized_action: tuple[float, float, float]
    metric_residual_root_m: tuple[float, float, float]


class ResidualSACRuntime:
    """Inference-only adapter around the existing SAC actor implementation."""

    def __init__(
        self,
        actor: SquashedGaussianActor,
        *,
        receipt: ResidualSACActorReceipt,
    ) -> None:
        if not isinstance(actor, SquashedGaussianActor):
            raise ResidualSACRuntimeError("existing SquashedGaussianActor is required")
        if actor.config.action_dim != 3 or tuple(actor.config.action_mask) != (1.0, 1.0, 1.0):
            raise ResidualSACRuntimeError("residual SAC actor must have exactly three XYZ outputs")
        if receipt.action_dim != 3 or receipt.frame != "robot_root" or receipt.unit != "m":
            raise ResidualSACRuntimeError("residual SAC receipt authority mismatch")
        if receipt.output_parameterization != RESIDUAL_SAC_OUTPUT_PARAMETERIZATION:
            raise ResidualSACRuntimeError("residual SAC output parameterization mismatch")
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        self.actor = actor
        self.receipt = receipt

    @property
    def observation_dim(self) -> int:
        return int(self.actor.config.observation_dim)

    def infer(self, recurrent_feature: torch.Tensor) -> ResidualSACInference:
        """Return normalized and metric residual authority without clipping.

        The fixed radial ``tanh`` parameterization maps the existing actor's
        finite 3-D output smoothly into the open 4.5-mm ball.  It is not a
        downstream clamp: every input has one analytic output and no component
        is truncated.
        """

        if tuple(recurrent_feature.shape) != (1, self.observation_dim):
            raise ResidualSACRuntimeError(
                f"residual SAC observation must be [1,{self.observation_dim}]"
            )
        if not bool(torch.isfinite(recurrent_feature).all()):
            raise ResidualSACRuntimeError("residual SAC observation contains NaN/Inf")
        with torch.inference_mode():
            normalized, _log_probability, _mean = self.actor.sample(
                recurrent_feature, deterministic=True
            )
        if tuple(normalized.shape) != (1, 3) or not bool(torch.isfinite(normalized).all()):
            raise ResidualSACRuntimeError("residual SAC actor output must be finite [1,3]")
        vector = normalized[0]
        radius = torch.linalg.vector_norm(vector)
        if float(radius.item()) <= 1.0e-12:
            metric = torch.zeros_like(vector)
        else:
            maximum = canonical_keyboard_grasp_contract().residual_raw_maximum_norm_m
            metric = maximum * torch.tanh(radius) * vector / radius
        metric_norm = float(torch.linalg.vector_norm(metric).item())
        if not math.isfinite(metric_norm) or metric_norm >= (
            canonical_keyboard_grasp_contract().residual_raw_maximum_norm_m + 1.0e-12
        ):
            raise ResidualSACRuntimeError("residual SAC metric output exceeded 4.5 mm")
        return ResidualSACInference(
            normalized_action=tuple(
                float(value) for value in normalized[0].detach().cpu().tolist()
            ),
            metric_residual_root_m=tuple(
                float(value) for value in metric.detach().cpu().tolist()
            ),
        )

    def infer_metric_xyz(self, recurrent_feature: torch.Tensor) -> tuple[float, float, float]:
        """Compatibility API returning only the metric residual."""

        return self.infer(recurrent_feature).metric_residual_root_m


def load_residual_sac_actor_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    expected_human_grasp_checkpoint_sha256: str,
    expected_observation_dim: int,
    device: str | torch.device = "cpu",
) -> ResidualSACRuntime:
    """Strictly load a trained 3-D residual actor; no legacy SAC fallback."""

    if not _SHA256.fullmatch(expected_sha256):
        raise ResidualSACRuntimeError("residual SAC checkpoint SHA-256 is invalid")
    if not _SHA256.fullmatch(expected_human_grasp_checkpoint_sha256):
        raise ResidualSACRuntimeError("human-grasp checkpoint SHA-256 is invalid")
    if type(expected_observation_dim) is not int or expected_observation_dim <= 0:
        raise ResidualSACRuntimeError("residual observation dimension must be positive")
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise ResidualSACRuntimeError("residual SAC checkpoint is missing")
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ResidualSACRuntimeError("residual SAC checkpoint hash mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ResidualSACRuntimeError("residual SAC checkpoint is unreadable") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != RESIDUAL_SAC_ACTOR_CHECKPOINT_SCHEMA:
        raise ResidualSACRuntimeError("residual SAC checkpoint schema mismatch")
    required = {
        "schema",
        "static_config",
        "actor_config",
        "actor_state_dict",
        "actor_state_sha256",
        "human_grasp_checkpoint_sha256",
        "output_parameterization",
    }
    if set(payload) != required:
        raise ResidualSACRuntimeError("residual SAC checkpoint fields mismatch")
    static_config = BCResidualSACStaticConfig.from_payload(payload["static_config"])
    actor_config_payload = payload["actor_config"]
    if not isinstance(actor_config_payload, Mapping):
        raise ResidualSACRuntimeError("residual SAC actor config is invalid")
    actor_config = SACConfig(**dict(actor_config_payload))
    if (
        actor_config.observation_dim != expected_observation_dim
        or actor_config.action_dim != static_config.sac_action_dim
        or tuple(actor_config.action_mask) != (1.0, 1.0, 1.0)
    ):
        raise ResidualSACRuntimeError("residual SAC actor dimensional contract mismatch")
    if payload["human_grasp_checkpoint_sha256"] != expected_human_grasp_checkpoint_sha256:
        raise ResidualSACRuntimeError("residual SAC checkpoint is bound to a different GRU")
    if payload["output_parameterization"] != RESIDUAL_SAC_OUTPUT_PARAMETERIZATION:
        raise ResidualSACRuntimeError("residual SAC checkpoint output mapping mismatch")
    actor = SquashedGaussianActor(actor_config).to(device)
    try:
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
    except Exception as error:
        raise ResidualSACRuntimeError("residual SAC actor state_dict mismatch") from error
    if payload["actor_state_sha256"] != _state_sha256(actor):
        raise ResidualSACRuntimeError("residual SAC actor state hash mismatch")
    if any(not bool(torch.isfinite(value).all()) for value in actor.state_dict().values()):
        raise ResidualSACRuntimeError("residual SAC checkpoint contains NaN/Inf")
    receipt = ResidualSACActorReceipt(
        checkpoint_path=str(path),
        checkpoint_sha256=observed_sha256,
        actor_state_sha256=str(payload["actor_state_sha256"]),
        human_grasp_checkpoint_sha256=expected_human_grasp_checkpoint_sha256,
        observation_dim=actor_config.observation_dim,
        action_dim=actor_config.action_dim,
        output_parameterization=RESIDUAL_SAC_OUTPUT_PARAMETERIZATION,
        frame=static_config.frame,
        unit=static_config.position_unit,
        deterministic_inference=True,
    )
    return ResidualSACRuntime(actor, receipt=receipt)


def residual_sac_actor_checkpoint_payload(
    actor: SquashedGaussianActor,
    *,
    human_grasp_checkpoint_sha256: str,
    static_config: BCResidualSACStaticConfig = BCResidualSACStaticConfig(),
) -> dict[str, Any]:
    """Build the strict payload shape for future training/checkpoint writers."""

    if actor.config.action_dim != 3 or tuple(actor.config.action_mask) != (1.0, 1.0, 1.0):
        raise ResidualSACRuntimeError("checkpoint actor must be XYZ-only")
    if not _SHA256.fullmatch(human_grasp_checkpoint_sha256):
        raise ResidualSACRuntimeError("human-grasp checkpoint SHA-256 is invalid")
    return {
        "schema": RESIDUAL_SAC_ACTOR_CHECKPOINT_SCHEMA,
        "static_config": static_config.payload(),
        "actor_config": asdict(actor.config),
        "actor_state_dict": actor.state_dict(),
        "actor_state_sha256": _state_sha256(actor),
        "human_grasp_checkpoint_sha256": human_grasp_checkpoint_sha256,
        "output_parameterization": RESIDUAL_SAC_OUTPUT_PARAMETERIZATION,
    }


__all__ = [
    "RESIDUAL_SAC_ACTOR_CHECKPOINT_SCHEMA",
    "RESIDUAL_SAC_OUTPUT_PARAMETERIZATION",
    "ResidualSACActorReceipt",
    "ResidualSACInference",
    "ResidualSACRuntime",
    "ResidualSACRuntimeError",
    "load_residual_sac_actor_checkpoint",
    "residual_sac_actor_checkpoint_payload",
]
