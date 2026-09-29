#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed offline audit for Stage-1A CLOSE-readiness supervision.

This tool never imports Isaac, creates an AppLauncher, trains SAC, or changes a
runtime gate.  It only decides whether a future teacher-only 3K run has a
causal pre-CLOSE dataset suitable for an episode-disjoint student evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "g2_stage1a_close_readiness_offline_audit_v1"
MIN_ROWS_PER_CLASS = 32
MIN_EPISODES_PER_CLASS = 3
EVAL_THRESHOLD = 0.5  # Fixed binary decision readout; never tuned here.
STUDENT_FEATURE_AUTHORITY = (
    "FROZEN_GRU_FEATURE(right_wrist_rgbd,ee_pose_root,right_arm_q,"
    "right_arm_qd,current_gripper_state,previous_action_4d)"
)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"env-{int(row['env_id']):02d}:{row['episode_id']}"


def _legacy_pre_close_candidate(row: Mapping[str, Any]) -> bool:
    """Conservative compatibility view for immutable pre-contract artifacts."""

    return bool(
        row.get("phase") == "LOCAL_GRASP"
        and not _as_bool(row.get("close_latched", "1"))
    )


def _metric_teacher_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        raise ValueError("metrics CSV has no rows")
    required = {
        "env_id",
        "episode_id",
        "phase",
        "close_latched",
        "privileged_teacher_available",
        "privileged_close_ready_target",
        "privileged_close_ready_score",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"metrics CSV missing required columns: {sorted(missing)}")
    canonical = {
        "pre_close_candidate",
        "close_latched_before_supervision",
        "privileged_close_ready_negative_reasons",
        "close_readiness_target_schema",
    }.issubset(rows[0])
    selected: list[dict[str, Any]] = []
    raw_teacher_rows = 0
    for index, row in enumerate(rows):
        if not _as_bool(row["privileged_teacher_available"]):
            continue
        raw_teacher_rows += 1
        target_raw = row["privileged_close_ready_target"]
        if target_raw not in {"0", "1"}:
            continue
        pre_close = (
            _as_bool(row["pre_close_candidate"])
            and not _as_bool(row["close_latched_before_supervision"])
            if canonical
            else _legacy_pre_close_candidate(row)
        )
        if not pre_close:
            continue
        score = float(row["privileged_close_ready_score"])
        target = target_raw == "1"
        reasons = tuple(
            item
            for item in row.get("privileged_close_ready_negative_reasons", "").split(";")
            if item
        )
        selected.append(
            {
                "row_index": index,
                "episode_key": _episode_key(row),
                "target": target,
                "score": score,
                "negative_reasons": reasons,
                "schema": row.get("close_readiness_target_schema", ""),
            }
        )
    return selected, {
        "raw_metric_row_count": len(rows),
        "raw_teacher_row_count": raw_teacher_rows,
        "pre_close_candidate_source": (
            "CANONICAL_EXPLICIT" if canonical else "LEGACY_DERIVED_NONCANONICAL"
        ),
    }


def _student_feature_rows(path: Path) -> list[dict[str, Any]]:
    """Load future durable pre-CLOSE rows without accepting post-latch data."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not (
                row.get("pre_close_candidate") is True
                and row.get("close_latched_before_supervision") is False
            ):
                raise ValueError(f"student row {line_number} is not pre-CLOSE-only")
            if row.get("student_privileged_input_count") != 0:
                raise ValueError(
                    f"student row {line_number} has privileged input leakage"
                )
            target = row.get("privileged_close_ready_target")
            score = row.get("privileged_close_ready_score")
            if type(target) is not bool or score != (1.0 if target else 0.0):
                raise ValueError(f"student row {line_number} has non-binary target semantics")
            feature = np.asarray(row.get("actor_observation"), dtype=np.float64)
            if feature.ndim != 1 or feature.size == 0 or not np.isfinite(feature).all():
                raise ValueError(f"student row {line_number} has invalid deployable feature")
            records.append(
                {
                    "episode_key": str(row["episode_key"]),
                    "target": target,
                    "feature": feature,
                }
            )
    return records


def _episode_split(records: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Build a deterministic, episode-disjoint and label-stratified split.

    A hash-only split can put every rare negative episode in train even when
    enough independent negative episodes exist.  That is a split artifact,
    not evidence that a heldout evaluation is impossible.  This allocator
    never moves individual frames: an episode is assigned exactly once.
    """

    label_sets: dict[str, set[bool]] = {}
    row_counts: Counter[str] = Counter()
    for row in records:
        episode = str(row["episode_key"])
        label_sets.setdefault(episode, set()).add(bool(row["target"]))
        row_counts[episode] += 1

    result = {"train": set(), "val": set(), "heldout": set()}
    bucket_labels = {name: set() for name in result}
    bucket_rows: Counter[str] = Counter()
    # The stable hash is solely a tie breaker.  Missing label coverage always
    # wins, so a rare class is represented in val and heldout when possible.
    episodes = sorted(
        label_sets,
        key=lambda episode: (
            -len(label_sets[episode]),
            hashlib.sha256(episode.encode()).hexdigest(),
        ),
    )
    for episode in episodes:
        labels = label_sets[episode]
        destination = min(
            result,
            key=lambda name: (
                -sum(label not in bucket_labels[name] for label in labels),
                bucket_rows[name],
                {"train": 0, "val": 1, "heldout": 2}[name],
            ),
        )
        result[destination].add(episode)
        bucket_labels[destination].update(labels)
        bucket_rows[destination] += row_counts[episode]
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 1.0e-6)
    x = (x_train - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    positive = max(1, int(y_train.sum()))
    negative = max(1, int(y_train.size - positive))
    sample_weight = np.where(y_train > 0.5, y_train.size / (2.0 * positive), y_train.size / (2.0 * negative))
    for _ in range(600):
        probability = _sigmoid(x @ weights + bias)
        residual = (probability - y_train) * sample_weight
        weights -= 0.05 * ((x.T @ residual) / y_train.size + 1.0e-4 * weights)
        bias -= 0.05 * float(residual.mean())
    return weights, np.asarray([bias]), np.stack((mean, scale), axis=0)


def _predict_logistic(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    return _sigmoid(((x - scaler[0]) / scaler[1]) @ weights + float(bias[0]))


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1, dtype=np.float64)
    return float((ranks[y.astype(bool)].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score)
    labels = y[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, labels.size + 1)
    recall = tp / positives
    return float(np.trapz(precision, recall))


def _evaluation(
    y: np.ndarray, score: np.ndarray, *, threshold: float = EVAL_THRESHOLD
) -> dict[str, Any]:
    if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError("evaluation threshold must be finite and in (0, 1)")
    prediction = score >= float(threshold)
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    tn = int(np.sum(~prediction & (y == 0)))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (score >= start) & (score < end if end < 1.0 else score <= end)
        if mask.any():
            ece += float(mask.mean() * abs(score[mask].mean() - y[mask].mean()))
    return {
        "THRESHOLD": float(threshold),
        "PR_AUC": _pr_auc(y, score),
        "ROC_AUC": _roc_auc(y, score),
        "BRIER": float(np.mean((score - y) ** 2)),
        "FALSE_ACCEPT_RATE": float(fp / max(1, fp + tn)),
        "FALSE_REJECT_RATE": float(fn / max(1, fn + tp)),
        "PRECISION": float(tp / max(1, tp + fp)),
        "RECALL": float(tp / max(1, tp + fn)),
        "F1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "CALIBRATION_ECE": ece,
        "TARGET_0_SCORE": _distribution(score[y == 0]),
        "TARGET_1_SCORE": _distribution(score[y == 1]),
        "ALWAYS_READY_COLLAPSE": bool(prediction.all()),
    }


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def audit(metrics_csv: Path, student_rows: Path | None) -> dict[str, Any]:
    records, source = _metric_teacher_rows(metrics_csv)
    positive = [row for row in records if row["target"]]
    negative = [row for row in records if not row["target"]]
    reasons = Counter(reason for row in negative for reason in row["negative_reasons"])
    by_class_episodes = {
        "positive": {row["episode_key"] for row in positive},
        "negative": {row["episode_key"] for row in negative},
    }
    binary_score_ok = all(
        row["score"] == (1.0 if row["target"] else 0.0) for row in records
    )
    result: dict[str, Any] = {
        "SCHEMA": SCHEMA,
        **source,
        "PRE_CLOSE_ROW_COUNT": len(records),
        "PRE_CLOSE_POSITIVE_COUNT": len(positive),
        "PRE_CLOSE_NEGATIVE_COUNT": len(negative),
        "POSITIVE_RATIO": float(len(positive) / len(records)) if records else None,
        "NEGATIVE_RATIO": float(len(negative) / len(records)) if records else None,
        "NEGATIVE_REASON_COUNTS": dict(sorted(reasons.items())),
        "EPISODE_COUNT": len({row["episode_key"] for row in records}),
        "POSITIVE_EPISODE_COUNT": len(by_class_episodes["positive"]),
        "NEGATIVE_EPISODE_COUNT": len(by_class_episodes["negative"]),
        "TARGET_SEMANTICS_FIXED": bool(binary_score_ok and source["pre_close_candidate_source"] == "CANONICAL_EXPLICIT"),
        "STUDENT_FEATURE_ROWS_AVAILABLE": 0,
        "STUDENT_FEATURE_AUTHORITY": STUDENT_FEATURE_AUTHORITY,
        "STUDENT_PRIVILEGED_INPUT_COUNT": "UNVERIFIED",
        "EPISODE_DISJOINT_HELDOUT": False,
        "SPLIT_EPISODE_COUNTS": None,
        "SPLIT_LABEL_COUNTS": None,
        "HELDOUT_PR_AUC": None,
        "HELDOUT_ROC_AUC": None,
        "HELDOUT_BRIER": None,
        "HELDOUT_FALSE_ACCEPT_RATE": None,
        "HELDOUT_FALSE_REJECT_RATE": None,
        "HELDOUT_CALIBRATION_ECE": None,
        "HELDOUT_TARGET_0_SCORE": None,
        "HELDOUT_TARGET_1_SCORE": None,
        "ALWAYS_READY_COLLAPSE": "YES_BY_LABEL_DEGENERACY" if not negative else "NOT_EVALUATED",
        "OFFLINE_CONTRACT_PASS": False,
        "RUNTIME_GATE_PROMOTION": "NO",
        "3K_RERUN_AUTHORIZED": "NO",
        "AUTO_15K_STARTED": "NO",
    }
    enough_labels = (
        len(positive) >= MIN_ROWS_PER_CLASS
        and len(negative) >= MIN_ROWS_PER_CLASS
        and len(by_class_episodes["positive"]) >= MIN_EPISODES_PER_CLASS
        and len(by_class_episodes["negative"]) >= MIN_EPISODES_PER_CLASS
    )
    if not enough_labels:
        result["NEXT"] = "COLLECT_MORE_PRE_CLOSE_NEGATIVES"
        return result
    if student_rows is None:
        result["NEXT"] = "PERSIST_DEPLOYABLE_PRE_CLOSE_FEATURE_ROWS"
        return result
    feature_rows = _student_feature_rows(student_rows)
    result["STUDENT_FEATURE_ROWS_AVAILABLE"] = len(feature_rows)
    result["STUDENT_PRIVILEGED_INPUT_COUNT"] = 0
    if not feature_rows:
        result["NEXT"] = "PERSIST_DEPLOYABLE_PRE_CLOSE_FEATURE_ROWS"
        return result
    dimensions = {row["feature"].shape for row in feature_rows}
    if len(dimensions) != 1:
        result["NEXT"] = "FIX_FEATURE_SCHEMA"
        return result
    splits = _episode_split(feature_rows)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, episodes in splits.items():
        subset = [row for row in feature_rows if row["episode_key"] in episodes]
        if not subset:
            result["NEXT"] = "COLLECT_MORE_PRE_CLOSE_NEGATIVES"
            return result
        arrays[name] = (
            np.stack([row["feature"] for row in subset]),
            np.asarray([int(row["target"]) for row in subset], dtype=np.int64),
        )
    result["SPLIT_EPISODE_COUNTS"] = {
        name: len(episodes) for name, episodes in splits.items()
    }
    result["SPLIT_LABEL_COUNTS"] = {
        name: {
            "positive": int(labels.sum()),
            "negative": int(labels.size - labels.sum()),
        }
        for name, (_, labels) in arrays.items()
    }
    if any(np.unique(labels).size != 2 for _, labels in arrays.values()):
        result["NEXT"] = "COLLECT_MORE_PRE_CLOSE_NEGATIVES"
        return result
    result["EPISODE_DISJOINT_HELDOUT"] = True
    weights, bias, scaler = _fit_logistic(*arrays["train"])
    heldout_score = _predict_logistic(arrays["heldout"][0], weights, bias, scaler)
    metrics = _evaluation(arrays["heldout"][1], heldout_score)
    result.update({f"HELDOUT_{key}": value for key, value in metrics.items() if key != "ALWAYS_READY_COLLAPSE"})
    result["ALWAYS_READY_COLLAPSE"] = "YES" if metrics["ALWAYS_READY_COLLAPSE"] else "NO"
    result["OFFLINE_CONTRACT_PASS"] = bool(
        not metrics["ALWAYS_READY_COLLAPSE"]
        and metrics["FALSE_ACCEPT_RATE"] < 1.0
        and math.isfinite(metrics["PR_AUC"])
        and math.isfinite(metrics["ROC_AUC"])
    )
    result["3K_RERUN_AUTHORIZED"] = "YES" if result["OFFLINE_CONTRACT_PASS"] else "NO"
    result["NEXT"] = "RUN_BOUNDED_3K" if result["OFFLINE_CONTRACT_PASS"] else "FIX_TARGET_SEMANTICS"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--student-rows-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.metrics_csv, args.student_rows_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
