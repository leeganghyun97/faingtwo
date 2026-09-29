# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Frozen, deployable Wrist RGB-D CLOSE-readiness scorer.

The score produced here is deliberately *advisory only*.  It contains no
privileged geometry, cube ground truth, family identity, score history, or
FSM state.  The caller must preserve the deterministic FSM CLOSE decision
unchanged; this module is intentionally incapable of submitting an action.

The pooled spatial representation is the exact 1,152-D contract used by the
frozen Conditional-CORAL student checkpoint:

* RGB spatial branch;
* masked-depth spatial branch;
* depth-valid coverage and boundary topology branch.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .stage1a_close_readiness_advisory import (
    CloseReadinessAdvice,
    FrozenCloseReadinessAdvisory,
)


RGB_HEIGHT = 192
RGB_WIDTH = 256
POOL_HEIGHT = 12
POOL_WIDTH = 16
FEATURE_DIM = POOL_HEIGHT * POOL_WIDTH * (3 + 1 + 2)
LATENT_DIM = 32
MARGIN_SCALE = 4.0
DEFAULT_HIGH_READY_THRESHOLD = 0.7932017446


class FrozenSpatialStudentError(ValueError):
    """Raised before an advisory receipt can be used for telemetry."""


def sha256_file(path: Path) -> str:
    """Return a full-file digest without accepting a partial checkpoint."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool(channels: np.ndarray, expected_channels: int) -> np.ndarray:
    if (
        channels.shape != (RGB_HEIGHT, RGB_WIDTH, expected_channels)
        or not np.isfinite(channels).all()
    ):
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_BRANCH_INPUT_INVALID")
    pooled = channels.reshape(
        POOL_HEIGHT, 16, POOL_WIDTH, 16, expected_channels
    ).mean(axis=(1, 3))
    result = pooled.reshape(-1)
    if (
        result.shape != (POOL_HEIGHT * POOL_WIDTH * expected_channels,)
        or not np.isfinite(result).all()
    ):
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_BRANCH_OUTPUT_INVALID")
    return result


def _validity_topology_feature(valid: np.ndarray) -> np.ndarray:
    if valid.shape != (RGB_HEIGHT, RGB_WIDTH):
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_VALID_MASK_SHAPE_INVALID")
    value = valid.astype(np.float64, copy=False)
    horizontal = np.zeros_like(value)
    vertical = np.zeros_like(value)
    horizontal[:, 1:] = np.abs(value[:, 1:] - value[:, :-1])
    vertical[1:, :] = np.abs(value[1:, :] - value[:-1, :])
    result = np.concatenate(
        (_pool(value[..., None], 1), _pool(np.maximum(horizontal, vertical)[..., None], 1))
    )
    if result.shape != (POOL_HEIGHT * POOL_WIDTH * 2,) or not np.isfinite(result).all():
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_TOPOLOGY_INVALID")
    return result


def pooled_spatial_missingness_feature(
    rgb: np.ndarray, depth_m: np.ndarray, depth_valid: np.ndarray
) -> np.ndarray:
    """Build the frozen model's deployable spatial feature exactly once."""

    if rgb.shape != (RGB_HEIGHT, RGB_WIDTH, 3) or depth_m.shape != (
        RGB_HEIGHT,
        RGB_WIDTH,
    ) or depth_valid.shape != (RGB_HEIGHT, RGB_WIDTH):
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_FRAME_SHAPE_INVALID")
    if not np.isfinite(depth_m).all():
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_DEPTH_NONFINITE")
    valid = depth_valid.astype(bool, copy=False)
    rgb_feature = _pool(rgb.astype(np.float64, copy=False) / 255.0, 3)
    masked_depth = np.clip(depth_m.astype(np.float64, copy=False), 0.0, 2.0) * valid
    depth_feature = _pool((masked_depth / 2.0)[..., None], 1)
    result = np.concatenate((rgb_feature, depth_feature, _validity_topology_feature(valid)))
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise FrozenSpatialStudentError("SPATIAL_STUDENT_FEATURE_INVALID")
    return result.astype(np.float32, copy=False)


class FrozenSpatialMissingnessStudent:
    """Read-only Conditional-CORAL checkpoint inference for live telemetry."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        expected_sha256: str,
        high_ready_threshold: float = DEFAULT_HIGH_READY_THRESHOLD,
        device: torch.device | str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FrozenSpatialStudentError("FROZEN_STUDENT_CHECKPOINT_MISSING")
        actual_sha256 = sha256_file(self.checkpoint_path)
        if actual_sha256 != str(expected_sha256):
            raise FrozenSpatialStudentError("FROZEN_STUDENT_CHECKPOINT_HASH_MISMATCH")
        self.checkpoint_sha256 = actual_sha256
        self.device = torch.device(device)
        payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise FrozenSpatialStudentError("FROZEN_STUDENT_PAYLOAD_INVALID")
        if payload.get("schema") != "g2_stage1a_conditional_coral_frozen_student_v1":
            raise FrozenSpatialStudentError("FROZEN_STUDENT_SCHEMA_INVALID")
        if payload.get("family_id_input") is not False or int(
            payload.get("privileged_student_input_count", -1)
        ) != 0:
            raise FrozenSpatialStudentError("FROZEN_STUDENT_PRIVILEGE_CONTRACT_INVALID")
        expected_inputs = [
            "right_wrist_rgb",
            "right_wrist_masked_depth",
            "right_wrist_depth_valid_topology",
        ]
        if list(payload.get("inference_input", ())) != expected_inputs:
            raise FrozenSpatialStudentError("FROZEN_STUDENT_INPUT_CONTRACT_INVALID")
        mean = np.asarray(payload.get("normalizer_mean"), dtype=np.float32)
        std = np.asarray(payload.get("normalizer_std"), dtype=np.float32)
        if (
            mean.shape != (FEATURE_DIM,)
            or std.shape != (FEATURE_DIM,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0.0)
        ):
            raise FrozenSpatialStudentError("FROZEN_STUDENT_NORMALIZER_INVALID")
        self._mean = torch.as_tensor(mean, device=self.device)
        self._std = torch.as_tensor(std, device=self.device)
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, LATENT_DIM),
            torch.nn.LayerNorm(LATENT_DIM),
        ).to(self.device)
        self._margin_head = torch.nn.Linear(LATENT_DIM, 1).to(self.device)
        try:
            self._encoder.load_state_dict(payload["encoder"], strict=True)
            self._margin_head.load_state_dict(payload["margin_head"], strict=True)
        except (KeyError, RuntimeError) as error:
            raise FrozenSpatialStudentError("FROZEN_STUDENT_STATE_INVALID") from error
        self._encoder.eval()
        self._margin_head.eval()
        for parameter in (*self._encoder.parameters(), *self._margin_head.parameters()):
            parameter.requires_grad_(False)
        self.advisory = FrozenCloseReadinessAdvisory(
            high_ready_threshold=float(high_ready_threshold)
        )

    def score(
        self, *, rgb: np.ndarray, depth_m: np.ndarray, depth_valid: np.ndarray
    ) -> float:
        """Return a finite probability using only one current Wrist frame."""

        feature = pooled_spatial_missingness_feature(rgb, depth_m, depth_valid)
        tensor = torch.as_tensor(feature, device=self.device).reshape(1, FEATURE_DIM)
        with torch.inference_mode():
            normalized = (tensor - self._mean) / self._std
            margin = self._margin_head(self._encoder(normalized)).reshape(())
            score = torch.sigmoid(MARGIN_SCALE * margin).item()
        result = float(score)
        if not math.isfinite(result) or not (0.0 <= result <= 1.0):
            raise FrozenSpatialStudentError("FROZEN_STUDENT_SCORE_INVALID")
        return result

    def score_and_advise(
        self, *, rgb: np.ndarray, depth_m: np.ndarray, depth_valid: np.ndarray
    ) -> tuple[float, CloseReadinessAdvice]:
        score = self.score(rgb=rgb, depth_m=depth_m, depth_valid=depth_valid)
        return score, self.advisory.advise(score)


__all__ = [
    "DEFAULT_HIGH_READY_THRESHOLD",
    "FEATURE_DIM",
    "FrozenSpatialMissingnessStudent",
    "FrozenSpatialStudentError",
    "pooled_spatial_missingness_feature",
    "sha256_file",
]
