# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable recovery contract for rejected Keyboard-v3 demonstrations.

The collector intentionally wrote validator failures with
``training_eligible=false``.  Recovery therefore never edits those files and
never weakens the canonical loader.  A versioned manifest binds each source
HDF5 and operator receipt by SHA-256, records the new validation result, and
gives every episode a collection-qualified key so repeated ``episode-000001``
IDs cannot leak across train/validation/test partitions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .keyboard_v3_dataset import (
    KeyboardV3DatasetError,
    read_recovered_keyboard_v3_episode,
)


KEYBOARD_V3_RECOVERY_SCHEMA = "g2_keyboard_v3_immutable_recovery_manifest_v1"
RECOVERED_CLASS = "RECOVERED_VALIDATOR_FALSE_REJECTION"
EXCLUDED_TEMPORAL_CLASS = "EXCLUDED_REAL_RGBD_CADENCE_ANOMALY"
DISCARDED_CLASS = "EXCLUDED_OPERATOR_DISCARD"


class KeyboardV3RecoveryError(RuntimeError):
    """Raised before unbound rejected evidence can enter a learner."""


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_key(collection_root: Path, episode_id: str) -> str:
    return f"{collection_root.name}::{episode_id}"


def _stratified_split(
    records: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    """Split whole collection-qualified episodes within target/outcome strata."""

    strata: dict[tuple[int, str], list[str]] = defaultdict(list)
    for record in records:
        strata[(int(record["target_residual_mm"]), str(record["operator_result"]))].append(
            str(record["training_key"])
        )
    partitions = {"train": [], "validation": [], "test": []}
    stratum_counts: dict[str, dict[str, int]] = {}
    for offset, (stratum, keys) in enumerate(sorted(strata.items())):
        if len(keys) < 3:
            raise KeyboardV3RecoveryError(
                f"stratum {stratum!r} has fewer than three recoverable episodes"
            )
        ordered = sorted(keys)
        random.Random(seed + offset).shuffle(ordered)
        total = len(ordered)
        validation_count = max(1, int(total * 0.15))
        test_count = max(1, int(total * 0.15))
        train_count = total - validation_count - test_count
        if train_count < 1:
            raise KeyboardV3RecoveryError(f"stratum {stratum!r} cannot be split")
        partitions["train"].extend(ordered[:train_count])
        partitions["validation"].extend(
            ordered[train_count : train_count + validation_count]
        )
        partitions["test"].extend(ordered[train_count + validation_count :])
        stratum_name = f"target_{stratum[0]}mm::{stratum[1]}"
        stratum_counts[stratum_name] = {
            "total": total,
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
    for values in partitions.values():
        values.sort()
    sets = {name: set(values) for name, values in partitions.items()}
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        raise KeyboardV3RecoveryError("episode leakage across recovery split")
    expected = {str(record["training_key"]) for record in records}
    if set().union(*sets.values()) != expected:
        raise KeyboardV3RecoveryError("recovery split does not cover eligible episodes")
    return {
        **partitions,
        "seed": int(seed),
        "split_unit": "collection_root_plus_episode_id",
        "stratification": "target_residual_mm_plus_operator_result",
        "strata": stratum_counts,
        "adjacent_row_leakage": False,
    }


def build_recovery_manifest(
    collection_roots: Iterable[str | Path], *, split_seed: int = 42
) -> dict[str, Any]:
    """Audit rejected evidence and return a signed in-memory manifest."""

    roots = sorted({Path(value).expanduser().resolve() for value in collection_roots})
    if not roots:
        raise KeyboardV3RecoveryError("no collection roots supplied")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for root in roots:
        receipt_root = root / "result_receipts"
        if not receipt_root.is_dir():
            continue
        for receipt_path in sorted(receipt_root.glob("*.result.json")):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            episode_id = str(receipt.get("episode_id", ""))
            key = _training_key(root, episode_id)
            if not episode_id or key in seen_keys:
                raise KeyboardV3RecoveryError("missing or duplicate recovery episode key")
            seen_keys.add(key)
            common = {
                "training_key": key,
                "collection_root": str(root),
                "episode_id": episode_id,
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": sha256_file(receipt_path),
                "operator_result": str(receipt.get("result", "")),
                "operator_success": bool(receipt.get("success", False)),
                "failure_type": str(receipt.get("failure_type", "")),
                "target_residual_mm": int(receipt.get("target_residual_mm", -1)),
            }
            source_value = receipt.get("save_path")
            if receipt.get("result") == "DISCARD" or not source_value:
                excluded.append(
                    {
                        **common,
                        "classification": DISCARDED_CLASS,
                        "source_path": None,
                        "validation_errors": [],
                    }
                )
                continue
            source = Path(str(source_value)).expanduser().resolve()
            if not source.is_file():
                excluded.append(
                    {
                        **common,
                        "classification": "EXCLUDED_SOURCE_FILE_MISSING",
                        "source_path": str(source),
                        "validation_errors": ["SOURCE_FILE_MISSING"],
                    }
                )
                continue
            source_sha256 = sha256_file(source)
            try:
                episode = read_recovered_keyboard_v3_episode(
                    source,
                    expected_sha256=source_sha256,
                    episode_id=episode_id,
                )
            except KeyboardV3DatasetError as error:
                message = str(error)
                errors = (
                    message.split(": ", 1)[1].split(",")
                    if ": " in message
                    else [message]
                )
                excluded.append(
                    {
                        **common,
                        "classification": EXCLUDED_TEMPORAL_CLASS,
                        "source_path": str(source),
                        "source_sha256": source_sha256,
                        "validation_errors": errors,
                    }
                )
                continue
            metadata_success = bool(episode.metadata.get("success", False))
            metadata_failure = str(episode.metadata.get("failure_type", ""))
            if metadata_success != common["operator_success"]:
                raise KeyboardV3RecoveryError(f"operator success mismatch for {key}")
            if metadata_failure != common["failure_type"]:
                raise KeyboardV3RecoveryError(f"operator failure label mismatch for {key}")
            rows = int(len(episode.actions["action_4d"]))
            close_count = int(episode.events["close_edge"].astype(bool).sum())
            eligible.append(
                {
                    **common,
                    "classification": RECOVERED_CLASS,
                    "source_path": str(source),
                    "source_sha256": source_sha256,
                    "rows": rows,
                    "close_event_count": close_count,
                    "validation_errors": [],
                }
            )
    if not eligible:
        raise KeyboardV3RecoveryError("no rejected episode passed recovery validation")
    split = _stratified_split(eligible, seed=split_seed)
    label_counts = Counter(record["operator_result"] for record in eligible)
    target_counts = Counter(int(record["target_residual_mm"]) for record in eligible)
    payload: dict[str, Any] = {
        "schema": KEYBOARD_V3_RECOVERY_SCHEMA,
        "status": "PASS",
        "source_mutation_count": 0,
        "collection_roots": [str(root) for root in roots],
        "eligible_episodes": eligible,
        "excluded_episodes": excluded,
        "eligible_episode_count": len(eligible),
        "excluded_episode_count": len(excluded),
        "eligible_row_count": sum(int(record["rows"]) for record in eligible),
        "operator_label_counts": dict(sorted(label_counts.items())),
        "target_counts": {str(key): value for key, value in sorted(target_counts.items())},
        "split": split,
        "contracts": {
            "control_hz": 50,
            "rgbd_hz": 25,
            "physics_hz": 500,
            "action_dim": 4,
            "action_frame": "robot_root",
            "action_unit": "meter",
            "maximum_xyz_norm_m": 0.0045,
            "actor_camera": "right_wrist_rgbd",
            "student_privileged_input_count": 0,
            "original_training_eligible_attribute_rewritten": False,
        },
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


def load_recovery_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != KEYBOARD_V3_RECOVERY_SCHEMA:
        raise KeyboardV3RecoveryError("recovery manifest schema mismatch")
    expected = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if expected != _json_sha256(unsigned):
        raise KeyboardV3RecoveryError("recovery manifest SHA-256 mismatch")
    if payload.get("source_mutation_count") != 0:
        raise KeyboardV3RecoveryError("recovery manifest reports source mutation")
    return payload


__all__ = [
    "DISCARDED_CLASS",
    "EXCLUDED_TEMPORAL_CLASS",
    "KEYBOARD_V3_RECOVERY_SCHEMA",
    "KeyboardV3RecoveryError",
    "RECOVERED_CLASS",
    "build_recovery_manifest",
    "load_recovery_manifest",
    "sha256_file",
]
