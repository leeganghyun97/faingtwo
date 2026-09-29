# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable train/eval contract for the deployable CLOSE-readiness student.

The geometry teacher remains outside the actor observation.  This module owns
only the normalization and sampling/loss agreement for the separate auxiliary
student head, so a successful teacher experiment cannot silently change the
CURRENT gripper gate or residual SAC action path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CLOSE_READINESS_TRAINING_CONTRACT_SCHEMA = (
    "g2_stage1a_close_readiness_normalized_balanced_contract_v1"
)
CLOSE_READINESS_STUDENT_INITIALIZATION_SCHEMA = (
    "g2_stage1a_close_readiness_normalized_balanced_initialization_v1"
)


class Stage1ACloseReadinessTrainingContractError(ValueError):
    """Raised before a malformed training contract can affect a student head."""


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_vector(value: Any, *, dimension: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise Stage1ACloseReadinessTrainingContractError(
            f"{label} must be finite [{dimension}]"
        )
    return np.ascontiguousarray(vector)


def validate_close_readiness_training_contract(
    payload: Mapping[str, Any], *, observation_dim: int
) -> dict[str, Any]:
    """Return a normalized, immutable-safe copy of a saved contract."""

    if not isinstance(payload, Mapping):
        raise Stage1ACloseReadinessTrainingContractError("training contract must be mapping")
    if payload.get("schema") != CLOSE_READINESS_TRAINING_CONTRACT_SCHEMA:
        raise Stage1ACloseReadinessTrainingContractError("training contract schema mismatch")
    if int(payload.get("observation_dim", -1)) != observation_dim:
        raise Stage1ACloseReadinessTrainingContractError("training contract observation_dim mismatch")
    if payload.get("student_head_architecture") != "LINEAR_LOGIT_V1":
        raise Stage1ACloseReadinessTrainingContractError(
            "training contract must use the linear-logit student head"
        )
    normalization = payload.get("normalization")
    sampler = payload.get("class_balanced_sampler")
    loss = payload.get("class_balanced_bce")
    split = payload.get("episode_disjoint_split")
    temporal = payload.get("temporal_receipts")
    qualification = payload.get("qualification")
    if not all(
        isinstance(item, Mapping)
        for item in (normalization, sampler, loss, split, temporal, qualification)
    ):
        raise Stage1ACloseReadinessTrainingContractError("training contract sections missing")
    mean = _finite_vector(normalization.get("mean"), dimension=observation_dim, label="mean")
    std = _finite_vector(normalization.get("std"), dimension=observation_dim, label="std")
    if np.any(std < 1.0e-6):
        raise Stage1ACloseReadinessTrainingContractError("normalizer std below floor")
    if sampler.get("mode") != "BALANCED_BINARY_WITH_REPLACEMENT":
        raise Stage1ACloseReadinessTrainingContractError("sampler must be balanced binary")
    if int(sampler.get("batch_size", -1)) <= 1 or int(sampler["batch_size"]) % 2:
        raise Stage1ACloseReadinessTrainingContractError("balanced batch must be positive/even")
    if loss.get("mode") != "INVERSE_CLASS_FREQUENCY_BCE_WITH_LOGITS":
        raise Stage1ACloseReadinessTrainingContractError("loss must be class-balanced BCE")
    positive_weight = float(loss.get("positive_weight", math.nan))
    negative_weight = float(loss.get("negative_weight", math.nan))
    if not all(math.isfinite(value) and value > 0.0 for value in (positive_weight, negative_weight)):
        raise Stage1ACloseReadinessTrainingContractError("class weights must be finite/positive")
    if split.get("unit") not in {"EPISODE", "SOURCE_FAMILY"} or not bool(
        split.get("disjoint", False)
    ):
        raise Stage1ACloseReadinessTrainingContractError(
            "episode- or source-family-disjoint split required"
        )
    if temporal.get("runtime_row_schema") != "g2_stage1a_close_readiness_temporal_receipt_v1":
        raise Stage1ACloseReadinessTrainingContractError(
            "temporal receipt schema mismatch"
        )
    if not all(
        isinstance(temporal.get(name), bool)
        for name in (
            "teacher_geometry_timestamp_logged",
            "student_feature_timestamp_logged",
            "gru_reset_generation_logged",
            "episode_env_identity_logged",
            "source_dataset_complete",
        )
    ):
        raise Stage1ACloseReadinessTrainingContractError(
            "temporal receipt completeness flags missing"
        )
    if not isinstance(qualification.get("offline_contract_pass"), bool):
        raise Stage1ACloseReadinessTrainingContractError(
            "offline qualification receipt missing"
        )
    contract = json.loads(json.dumps(payload))
    contract["normalization"]["mean"] = mean.tolist()
    contract["normalization"]["std"] = std.tolist()
    contract["class_balanced_bce"]["positive_weight"] = positive_weight
    contract["class_balanced_bce"]["negative_weight"] = negative_weight
    expected_sha = contract.pop("contract_sha256", None)
    actual_sha = canonical_json_sha256(contract)
    if expected_sha is not None and expected_sha != actual_sha:
        raise Stage1ACloseReadinessTrainingContractError("training contract sha256 mismatch")
    contract["contract_sha256"] = actual_sha
    return contract


def load_close_readiness_training_contract(
    path: str | Path, *, observation_dim: int
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise Stage1ACloseReadinessTrainingContractError("training contract file missing")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1ACloseReadinessTrainingContractError(
            "training contract file unreadable"
        ) from error
    return validate_close_readiness_training_contract(payload, observation_dim=observation_dim)


def normalize_close_readiness_feature(
    feature: Any, *, contract: Mapping[str, Any], observation_dim: int
) -> np.ndarray:
    vector = _finite_vector(feature, dimension=observation_dim, label="student feature")
    normalization = contract["normalization"]
    mean = _finite_vector(normalization["mean"], dimension=observation_dim, label="mean")
    std = _finite_vector(normalization["std"], dimension=observation_dim, label="std")
    normalized = (vector - mean) / std
    if not np.isfinite(normalized).all():
        raise Stage1ACloseReadinessTrainingContractError("normalized feature nonfinite")
    return np.ascontiguousarray(normalized.astype(np.float32, copy=False))
