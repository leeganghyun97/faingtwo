# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import numpy as np
import pytest

from geniesim.rl.sac.stage1a_close_readiness_training_contract import (
    CLOSE_READINESS_TRAINING_CONTRACT_SCHEMA,
    Stage1ACloseReadinessTrainingContractError,
    canonical_json_sha256,
    normalize_close_readiness_feature,
    validate_close_readiness_training_contract,
)


def _contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": CLOSE_READINESS_TRAINING_CONTRACT_SCHEMA,
        "observation_dim": 4,
        "student_head_architecture": "LINEAR_LOGIT_V1",
        "normalization": {
            "source_split": "TRAIN_ONLY",
            "mean": [1.0, 2.0, 3.0, 4.0],
            "std": [2.0, 2.0, 2.0, 2.0],
            "std_floor": 1.0e-6,
        },
        "class_balanced_sampler": {
            "mode": "BALANCED_BINARY_WITH_REPLACEMENT",
            "batch_size": 64,
            "positive_per_batch": 32,
            "negative_per_batch": 32,
        },
        "class_balanced_bce": {
            "mode": "INVERSE_CLASS_FREQUENCY_BCE_WITH_LOGITS",
            "positive_weight": 0.6,
            "negative_weight": 3.0,
        },
        "episode_disjoint_split": {"unit": "EPISODE", "disjoint": True},
        "temporal_receipts": {
            "runtime_row_schema": "g2_stage1a_close_readiness_temporal_receipt_v1",
            "teacher_geometry_timestamp_logged": True,
            "student_feature_timestamp_logged": True,
            "gru_reset_generation_logged": True,
            "episode_env_identity_logged": True,
            "source_dataset_complete": True,
        },
        "qualification": {"offline_contract_pass": False},
        "student_privileged_input_count": 0,
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def test_train_only_normalizer_is_deterministic_and_finite() -> None:
    contract = validate_close_readiness_training_contract(_contract(), observation_dim=4)
    normalized = normalize_close_readiness_feature(
        np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        contract=contract,
        observation_dim=4,
    )
    assert np.array_equal(normalized, np.ones(4, dtype=np.float32))


def test_contract_rejects_non_disjoint_unit_or_unbalanced_sampler() -> None:
    payload = _contract()
    payload["episode_disjoint_split"] = {"unit": "ROW", "disjoint": True}
    payload.pop("contract_sha256")
    payload["contract_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(Stage1ACloseReadinessTrainingContractError):
        validate_close_readiness_training_contract(payload, observation_dim=4)


def test_contract_accepts_family_disjoint_boundary_dataset() -> None:
    payload = _contract()
    payload["episode_disjoint_split"] = {"unit": "SOURCE_FAMILY", "disjoint": True}
    payload.pop("contract_sha256")
    payload["contract_sha256"] = canonical_json_sha256(payload)
    contract = validate_close_readiness_training_contract(payload, observation_dim=4)
    assert contract["episode_disjoint_split"]["unit"] == "SOURCE_FAMILY"
