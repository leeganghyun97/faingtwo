#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Freeze a PASSed boundary-balanced CLOSE-readiness student offline.

This program is deliberately unavailable until the paired source-family audit
has passed.  It neither imports Isaac nor creates an optimizer: the saved
linear head is the audited logistic probe represented in the runtime's
``LINEAR_LOGIT_V1`` state-dict format.  Geometry remains teacher-only and the
artifact has no CLOSE-gate authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from analyze_g2_stage1a_close_readiness_offline import _fit_logistic
from audit_g2_stage1a_boundary_paired_dataset import (
    _canonical_paired_groups,
    _load_rows,
    _split_units,
)
from geniesim.rl.sac.stage1a_close_readiness_training_contract import (
    CLOSE_READINESS_STUDENT_INITIALIZATION_SCHEMA,
    CLOSE_READINESS_TRAINING_CONTRACT_SCHEMA,
    canonical_json_sha256,
    validate_close_readiness_training_contract,
)
from geniesim.rl.sac.stage1a_real_sac_coordinator import build_close_readiness_head


SCHEMA = "g2_stage1a_boundary_teacher_student_finalize_v1"
BATCH_SIZE = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    positive = sum(bool(row["target"]) for row in rows)
    return {"positive": positive, "negative": len(rows) - positive}


def _families(rows: list[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row["source_family_id"]) for row in rows})


def _matrix(rows: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([np.asarray(row["feature"], dtype=np.float64) for row in rows]),
        np.asarray([int(bool(row["target"])) for row in rows], dtype=np.int64),
    )


def _all_receipts_present(rows: list[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        isinstance(row["raw"].get("teacher_geometry_timestamp_s"), (int, float))
        and isinstance(row["raw"].get("student_feature_timestamp_s"), (int, float))
        and isinstance(row["raw"].get("gru_reset_generation"), int)
        and isinstance(row["raw"].get("episode_id"), str)
        and isinstance(row["raw"].get("env_id"), int)
        for row in rows
    )


def _payloads(sidecars: list[Path]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    raw_rows = [row for sidecar in sidecars for row in _load_rows(sidecar)]
    canonical, _summary = _canonical_paired_groups(raw_rows)
    units = _split_units(canonical)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in canonical:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    split = {
        name: [row for pair_id in pair_ids for row in by_pair[pair_id]]
        for name, pair_ids in units.items()
    }
    return canonical, split


def finalize(*, sidecars: list[Path], audit_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("BOUNDARY_STUDENT_FREEZE_REFUSES_OVERWRITE")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("BOUNDARY_AUDIT_UNREADABLE") from error
    if audit.get("OFFLINE_CONTRACT_PASS") is not True:
        raise ValueError("BOUNDARY_AUDIT_NOT_QUALIFIED")
    if audit.get("THRESHOLD_SOURCE") != "VALIDATION_ONLY":
        raise ValueError("BOUNDARY_AUDIT_CALIBRATION_NOT_VALIDATION_ONLY")
    threshold = float(audit.get("VALIDATION_SELECTED_THRESHOLD"))
    if not 0.0 < threshold < 1.0:
        raise ValueError("BOUNDARY_AUDIT_THRESHOLD_INVALID")

    canonical, split = _payloads(sidecars)
    for name in ("train", "validation", "heldout"):
        if set(bool(row["target"]) for row in split[name]) != {False, True}:
            raise ValueError(f"BOUNDARY_FINALIZE_SPLIT_CLASS_MISSING:{name}")
    if not _all_receipts_present(canonical):
        raise ValueError("BOUNDARY_FINALIZE_TEMPORAL_RECEIPT_MISSING")

    x_train, y_train = _matrix(split["train"])
    weights, bias, scaler = _fit_logistic(x_train, y_train)
    feature_dim = int(x_train.shape[1])
    train_counts = _counts(split["train"])
    contract_without_sha: dict[str, Any] = {
        "schema": CLOSE_READINESS_TRAINING_CONTRACT_SCHEMA,
        "observation_dim": feature_dim,
        "student_head_architecture": "LINEAR_LOGIT_V1",
        "source_sidecars": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in sidecars
        ],
        "source_audit_sha256": _sha256(audit_path),
        "normalization": {
            "source_split": "TRAIN_ONLY_SOURCE_FAMILY_DISJOINT",
            "mean": scaler[0].astype(np.float32).tolist(),
            "std": scaler[1].astype(np.float32).tolist(),
            "std_floor": 1.0e-6,
        },
        "class_balanced_sampler": {
            "mode": "BALANCED_BINARY_WITH_REPLACEMENT",
            "batch_size": BATCH_SIZE,
            "positive_per_batch": BATCH_SIZE // 2,
            "negative_per_batch": BATCH_SIZE // 2,
        },
        "class_balanced_bce": {
            "mode": "INVERSE_CLASS_FREQUENCY_BCE_WITH_LOGITS",
            "positive_weight": float(len(y_train) / (2 * train_counts["positive"])),
            "negative_weight": float(len(y_train) / (2 * train_counts["negative"])),
        },
        "episode_disjoint_split": {
            "unit": "SOURCE_FAMILY",
            "disjoint": True,
            "train_source_family_ids": _families(split["train"]),
            "validation_source_family_ids": _families(split["validation"]),
            "heldout_source_family_ids": _families(split["heldout"]),
        },
        "temporal_receipts": {
            "runtime_row_schema": "g2_stage1a_close_readiness_temporal_receipt_v1",
            "teacher_geometry_timestamp_logged": True,
            "student_feature_timestamp_logged": True,
            "gru_reset_generation_logged": True,
            "episode_env_identity_logged": True,
            "source_dataset_complete": True,
        },
        "qualification": {
            "offline_contract_pass": True,
            "validation_selected_threshold": threshold,
            "threshold_source": "VALIDATION_ONLY",
            "validation_metrics": {
                key: audit.get(f"VALIDATION_{key}")
                for key in ("ROC_AUC", "PR_AUC", "BRIER", "FALSE_ACCEPT", "FALSE_REJECT")
            },
            "heldout_metrics": {
                key: audit.get(f"HELDOUT_{key}")
                for key in ("ROC_AUC", "PR_AUC", "BRIER", "FALSE_ACCEPT", "FALSE_REJECT")
            },
            "runtime_gate_promotion": "NO",
        },
        "split_counts": {name: _counts(rows) for name, rows in split.items()},
        "student_privileged_input_count": 0,
    }
    contract_payload = dict(contract_without_sha)
    contract_payload["contract_sha256"] = canonical_json_sha256(contract_without_sha)
    contract = validate_close_readiness_training_contract(
        contract_payload, observation_dim=feature_dim
    )

    head = build_close_readiness_head(feature_dim, architecture="LINEAR_LOGIT_V1")
    with torch.no_grad():
        head.linear.weight.copy_(torch.from_numpy(weights.astype(np.float32)).unsqueeze(0))
        head.linear.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    initialization = {
        "schema": CLOSE_READINESS_STUDENT_INITIALIZATION_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "observation_dim": feature_dim,
        "head_architecture": "LINEAR_LOGIT_V1",
        "head": head.state_dict(),
        "student_privileged_input_count": 0,
        "frozen": True,
        "construction": "AUDITED_OFFLINE_LINEAR_PROBE_NO_RUNTIME_OPTIMIZER",
        "validation_selected_threshold": threshold,
        "runtime_gate_promotion": "NO",
    }
    output_dir.mkdir(parents=True)
    contract_path = output_dir / "CLOSE_READINESS_TRAINING_CONTRACT.json"
    initialization_path = output_dir / "CLOSE_READINESS_STUDENT_HEAD_FROZEN.pt"
    report_path = output_dir / "FREEZE_REPORT.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(initialization, initialization_path)
    report = {
        "SCHEMA": SCHEMA,
        "OFFLINE_ONLY": True,
        "ISAAC_STARTED": "NO",
        "SAC_UPDATE": 0,
        "STUDENT_OPTIMIZER_UPDATE": 0,
        "PRIVILEGED_HARD_GATE": "NO",
        "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
        "STUDENT_HEAD_FROZEN": "YES",
        "NORMALIZER_FROZEN": "YES",
        "CALIBRATION_FROZEN": "YES",
        "CALIBRATION_SOURCE": "VALIDATION_ONLY",
        "HELDOUT_USED_FOR_CALIBRATION": "NO",
        "CONTRACT_PATH": str(contract_path.resolve()),
        "CONTRACT_SHA256": contract["contract_sha256"],
        "INITIALIZATION_PATH": str(initialization_path.resolve()),
        "INITIALIZATION_SHA256": _sha256(initialization_path),
        "TRAIN_POS_NEG": _counts(split["train"]),
        "VALIDATION_POS_NEG": _counts(split["validation"]),
        "HELDOUT_POS_NEG": _counts(split["heldout"]),
        "SOURCE_FAMILIES": {name: _families(rows) for name, rows in split.items()},
        "NEXT": "DESIGN_STUDENT_ADVISORY_CLOSE_3K",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(sidecars=args.sidecar, audit_path=args.audit, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
