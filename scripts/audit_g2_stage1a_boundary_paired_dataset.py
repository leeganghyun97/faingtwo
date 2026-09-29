#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Offline-only audit for Candidate-A boundary/provenance-balanced teacher rows.

Unlike the old new-contract audit, a physical episode is allowed to contain a
single class.  The atomic split unit here is a *paired source sample*: all
variants of that sample stay together.  A source family must itself contain
both labels, but a family is assigned to exactly one split.  This prevents
both frame leakage and source-family identity leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from analyze_g2_stage1a_close_readiness_offline import (
    _evaluation,
    _fit_logistic,
    _predict_logistic,
)
from geniesim.rl.sac.stage1a_boundary_paired_collection import (
    BOUNDARY_PAIRED_ROW_SCHEMA,
    SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
)
from geniesim.rl.sac.stage1a_signed_readiness_margin import (
    SIGNED_MARGIN_TARGET_SCHEMA,
    build_signed_readiness_margin,
)


SCHEMA = "g2_stage1a_boundary_paired_dataset_audit_v1"
MIN_ROWS_PER_CLASS = 50
MIN_SOURCE_FAMILIES_TOTAL = 8
# Fixed source-family-disjoint evaluation allocation requested for this
# boundary-balanced dataset: A/B/C -> train, D/E/F -> validation, G/H -> heldout.
# The concrete family IDs are deterministically selected after collection;
# placeholder letters never become model inputs or labels.
REQUIRED_SOURCE_FAMILIES_BY_SPLIT = {
    "train": 3,
    "validation": 3,
    "heldout": 2,
}
REQUIRED_NUMERIC = (
    "inner_pad_cube_gap_mm",
    "outer_pad_cube_gap_mm",
    "minimum_primary_pad_cube_gap_mm",
    "gripper_aperture_mm",
    "cube_effective_width_mm",
    "orientation_error_deg",
    "teacher_geometry_timestamp_s",
    "student_feature_timestamp_s",
)
REQUIRED_BOOL = (
    "pre_close_candidate",
    "close_latched_before_supervision",
    "cube_between_primary_pads",
    "aperture_geometrically_compatible",
    "owner_valid",
    "geometry_valid",
    "privileged_close_ready_target",
)
SIGNED_MARGIN_REQUIRED_NUMERIC = (
    "inner_containment_margin_mm",
    "outer_containment_margin_mm",
    "primary_pad_containment_margin_mm",
    "min_pad_surface_gap_mm",
    "aperture_margin_mm",
    "orientation_margin_deg",
    "nominal_residual_mm",
    "inner_toward_world_m",
    "outer_toward_world_m",
    "cube_projection_world_m",
)
CAUSAL_STATE_REQUIRED_V4 = (
    "ee_pose_robot_root_m_xyzw",
    "right_arm_joint_position_rad",
    "right_arm_joint_velocity_rad_s",
    "gripper_state_open",
    "measured_aperture_mm",
    "previous_policy_action_4d_metric_root_m",
    "robot_state_timestamp_s",
    "state_timestamp_parity_abs_s",
)
CAUSAL_STATE_FORBIDDEN_V4 = {
    "cube_gt",
    "relative_pose_root_m",
    "privileged_geometry",
}
WRIST_RGBD_REQUIRED_V5 = (
    "wrist_rgb_frame_ref",
    "wrist_depth_frame_ref",
    "right_wrist_frame_store_index",
    "camera_frame_id",
    "camera_timestamp_s",
    "camera_age_ms",
    "wrist_depth_valid_fraction",
    "wrist_rgbd_timestamp_parity_abs_s",
)


class BoundaryAuditError(ValueError):
    """Raised when a raw row cannot be canonical paired supervision."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    records = list(rows)
    positives = sum(bool(row["target"]) for row in records)
    return {"positive": positives, "negative": len(records) - positives}


def _source_provenance(raw: Mapping[str, Any], *, path: Path, line_number: int) -> Mapping[str, Any]:
    provenance = raw.get("source_provenance")
    required = ("hdf5_path", "hdf5_sha256", "planner_sidecar_sha256", "catalog_sha256")
    if not isinstance(provenance, Mapping) or any(
        not isinstance(provenance.get(name), str) or not provenance.get(name)
        for name in required
    ):
        raise BoundaryAuditError(f"SOURCE_PROVENANCE_INVALID:{path}:{line_number}")
    return provenance


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BoundaryAuditError(f"SIDECAR_MISSING:{path}")
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            schema = raw.get("schema")
            if schema not in {
                BOUNDARY_PAIRED_ROW_SCHEMA,
                SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
            }:
                raise BoundaryAuditError(f"ROW_SCHEMA_INVALID:{path}:{line_number}")
            if any(type(raw.get(name)) is not bool for name in REQUIRED_BOOL):
                raise BoundaryAuditError(f"ROW_BOOLEAN_INVALID:{path}:{line_number}")
            if not raw["pre_close_candidate"] or raw["close_latched_before_supervision"]:
                raise BoundaryAuditError(f"ROW_NOT_PRE_CLOSE_UNLATCHED:{path}:{line_number}")
            if raw.get("student_privileged_input_count") != 0:
                raise BoundaryAuditError(f"STUDENT_PRIVILEGED_LEAKAGE:{path}:{line_number}")
            if raw.get("contact_free_at_direct_init") is not True or raw.get("canonical_open_at_direct_init") is not True:
                raise BoundaryAuditError(f"DIRECT_INIT_CONTRACT_INVALID:{path}:{line_number}")
            if raw.get("forbidden_collision_before_supervision") is not False:
                raise BoundaryAuditError(f"PRE_CLOSE_FORBIDDEN_COLLISION:{path}:{line_number}")
            if any(
                not isinstance(raw.get(name), (int, float)) or not math.isfinite(float(raw[name]))
                for name in REQUIRED_NUMERIC
            ):
                raise BoundaryAuditError(f"ROW_NUMERIC_INVALID:{path}:{line_number}")
            if type(raw.get("gru_reset_generation")) is not int or raw["gru_reset_generation"] < 0:
                raise BoundaryAuditError(f"GRU_RESET_RECEIPT_INVALID:{path}:{line_number}")
            if not all(isinstance(raw.get(name), str) and raw[name] for name in (
                "source_family_id", "source_sample_id", "boundary_pair_id", "boundary_variant_id",
                "episode_id", "collection_batch_id",
            )) or type(raw.get("env_id")) is not int:
                raise BoundaryAuditError(f"ROW_PROVENANCE_IDENTITY_INVALID:{path}:{line_number}")
            feature = np.asarray(raw.get("frozen_student_feature_128d"), dtype=np.float64)
            if feature.shape != (128,) or not np.isfinite(feature).all():
                raise BoundaryAuditError(f"FROZEN_128D_FEATURE_INVALID:{path}:{line_number}")
            relative_pose = np.asarray(raw.get("relative_pose_root_m"), dtype=np.float64)
            if relative_pose.shape != (3,) or not np.isfinite(relative_pose).all():
                raise BoundaryAuditError(f"RELATIVE_POSE_ROOT_INVALID:{path}:{line_number}")
            target = raw["privileged_close_ready_target"]
            if raw.get("privileged_close_ready_score") != (1.0 if target else 0.0):
                raise BoundaryAuditError(f"TARGET_SEMANTICS_INVALID:{path}:{line_number}")
            reason = str(raw.get("negative_reason", ""))
            if (target and reason != "NONE") or (not target and not reason):
                raise BoundaryAuditError(f"NEGATIVE_REASON_SEMANTICS_INVALID:{path}:{line_number}")
            signed_margin = None
            if schema in {
                SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
            }:
                if (
                    raw.get("signed_margin_telemetry_schema")
                    != SIGNED_MARGIN_TARGET_SCHEMA
                    or type(raw.get("safety_valid")) is not bool
                    or any(
                        not isinstance(raw.get(name), (int, float))
                        or not math.isfinite(float(raw[name]))
                        for name in SIGNED_MARGIN_REQUIRED_NUMERIC
                    )
                    or not isinstance(raw.get("perturbation_type"), str)
                    or not raw["perturbation_type"]
                    or not isinstance(raw.get("perturbation_value"), Mapping)
                ):
                    raise BoundaryAuditError(
                        f"SIGNED_MARGIN_TELEMETRY_INVALID:{path}:{line_number}"
                    )
                margin_receipt = build_signed_readiness_margin(
                    pad_containment_margin_mm=float(
                        raw["primary_pad_containment_margin_mm"]
                    ),
                    minimum_primary_pad_cube_gap_mm=float(
                        raw["min_pad_surface_gap_mm"]
                    ),
                    gripper_aperture_mm=float(raw["gripper_aperture_mm"]),
                    cube_effective_width_mm=float(raw["cube_effective_width_mm"]),
                    orientation_error_deg=float(raw["orientation_error_deg"]),
                    owner_valid=bool(raw["owner_valid"]),
                    geometry_valid=bool(raw["geometry_valid"]),
                    no_safety_violation=bool(raw["safety_valid"]),
                )
                if (
                    not margin_receipt.recordable
                    or margin_receipt.aggregate_margin is None
                    or bool(margin_receipt.aggregate_margin >= 0.0) != bool(target)
                ):
                    raise BoundaryAuditError(
                        f"SIGNED_MARGIN_BINARY_PARITY_INVALID:{path}:{line_number}"
                    )
                signed_margin = float(margin_receipt.aggregate_margin)
            if schema in {
                SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
            }:
                if (
                    raw.get("student_causal_robot_state_schema")
                    != "g2_deployable_causal_robot_state_v1"
                    or raw.get("student_input_privileged_field_count") != 0
                    or not isinstance(raw.get("student_causal_state_fields"), list)
                    or not isinstance(raw.get("student_causal_state_forbidden_fields"), list)
                    or set(raw["student_causal_state_forbidden_fields"])
                    != CAUSAL_STATE_FORBIDDEN_V4
                    or any(
                        name not in raw["student_causal_state_fields"]
                        for name in CAUSAL_STATE_REQUIRED_V4[:-2]
                    )
                    or any(name not in raw for name in CAUSAL_STATE_REQUIRED_V4)
                ):
                    raise BoundaryAuditError(
                        f"CAUSAL_STATE_SCHEMA_INVALID:{path}:{line_number}"
                    )
                arrays = (
                    ("ee_pose_robot_root_m_xyzw", (7,)),
                    ("right_arm_joint_position_rad", (7,)),
                    ("right_arm_joint_velocity_rad_s", (7,)),
                    ("previous_policy_action_4d_metric_root_m", (4,)),
                )
                if any(
                    np.asarray(raw[name], dtype=np.float64).shape != expected
                    or not np.isfinite(
                        np.asarray(raw[name], dtype=np.float64)
                    ).all()
                    for name, expected in arrays
                ) or any(
                    not isinstance(raw.get(name), (int, float))
                    or not math.isfinite(float(raw[name]))
                    for name in (
                        "gripper_state_open",
                        "measured_aperture_mm",
                        "robot_state_timestamp_s",
                        "state_timestamp_parity_abs_s",
                    )
                ) or not math.isclose(
                    float(raw["robot_state_timestamp_s"]),
                    float(raw["student_feature_timestamp_s"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ) or float(raw["state_timestamp_parity_abs_s"]) > 1.0e-6:
                    raise BoundaryAuditError(
                        f"CAUSAL_STATE_TIMESTAMP_OR_VALUE_INVALID:{path}:{line_number}"
                    )
            if schema in {
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
                SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
            }:
                intrinsics = np.asarray(
                    raw.get("wrist_camera_intrinsics_3x3"), dtype=np.float64
                )
                if (
                    raw.get("wrist_rgbd_receipt_schema")
                    not in {
                        "g2_stage1a_timestamp_aligned_wrist_rgbd_v5",
                        "g2_stage1a_timestamp_aligned_wrist_rgbd_v6",
                    }
                    or raw.get("wrist_camera_binding") != "right_wrist_camera"
                    or raw.get("wrist_depth_unit") != "m"
                    or raw.get("wrist_rgb_shape") != [192, 256, 3]
                    or raw.get("wrist_depth_shape") != [192, 256, 1]
                    or raw.get("wrist_rgb_valid") is not True
                    or raw.get("wrist_depth_valid") is not True
                    or not isinstance(raw.get("wrist_rgbd_hdf5_path"), str)
                    or not raw["wrist_rgbd_hdf5_path"]
                    or any(type(raw.get(name)) is not int or raw[name] < 0 for name in (
                        "wrist_rgb_frame_ref",
                        "wrist_depth_frame_ref",
                        "right_wrist_frame_store_index",
                        "camera_frame_id",
                    ))
                    or raw["wrist_rgb_frame_ref"] != raw["wrist_depth_frame_ref"]
                    or raw["wrist_rgb_frame_ref"] != raw["right_wrist_frame_store_index"]
                    or any(
                        not isinstance(raw.get(name), (int, float))
                        or not math.isfinite(float(raw[name]))
                        for name in WRIST_RGBD_REQUIRED_V5[4:]
                    )
                    or float(raw["camera_age_ms"]) < 0.0
                    or float(raw["wrist_depth_valid_fraction"]) <= 0.0
                    or float(raw["wrist_depth_valid_fraction"]) > 1.0
                    or not math.isclose(
                        float(raw["camera_timestamp_s"]),
                        float(raw["student_feature_timestamp_s"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    or float(raw["wrist_rgbd_timestamp_parity_abs_s"]) > 1.0e-6
                    or intrinsics.shape != (3, 3)
                    or not np.isfinite(intrinsics).all()
                    or float(intrinsics[0, 0]) <= 0.0
                    or float(intrinsics[1, 1]) <= 0.0
                ):
                    raise BoundaryAuditError(
                        f"WRIST_RGBD_RECEIPT_INVALID:{path}:{line_number}"
                    )
            if schema == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA:
                v6_pose = np.asarray(
                    raw.get("right_wrist_camera_world_pose_m_xyzw"), dtype=np.float64
                )
                cube_optical = np.asarray(
                    raw.get("cube_pose_right_wrist_camera_optical_m_xyzw"), dtype=np.float64
                )
                if (
                    raw.get("v6_diagnostic_schema")
                    != "g2_stage1a_wrist_rgbd_camera_mask_diagnostic_v6"
                    or raw.get("student_privileged_input_count") != 0
                    or not isinstance(raw.get("v6_wrist_diagnostic_hdf5_path"), str)
                    or type(raw.get("v6_wrist_diagnostic_frame_ref")) is not int
                    or int(raw["v6_wrist_diagnostic_frame_ref"]) < 0
                    or v6_pose.shape != (7,)
                    or cube_optical.shape != (7,)
                    or not np.isfinite(v6_pose).all()
                    or not np.isfinite(cube_optical).all()
                    or not math.isclose(
                        float(raw["teacher_geometry_timestamp_s"]),
                        float(raw["student_feature_timestamp_s"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                ):
                    raise BoundaryAuditError(
                        f"WRIST_RGBD_V6_DIAGNOSTIC_RECEIPT_INVALID:{path}:{line_number}"
                    )
            output.append({
                "target": bool(target),
                "feature": feature,
                "negative_reason": reason,
                "source_family_id": raw["source_family_id"],
                "source_sample_id": raw["source_sample_id"],
                "pair_id": raw["boundary_pair_id"],
                "variant_id": raw["boundary_variant_id"],
                "collection_batch_id": raw["collection_batch_id"],
                "episode_key": f"{path.resolve()}@{_sha256(path)}::env-{raw['env_id']:02d}:{raw['episode_id']}",
                "source_provenance": _source_provenance(raw, path=path, line_number=line_number),
                "raw": raw,
                "signed_margin": signed_margin,
            })
    return output


def _canonical_paired_groups(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_pair[row["pair_id"]].append(row)
    canonical: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for pair_id, rows in by_pair.items():
        families = {str(row["source_family_id"]) for row in rows}
        samples = {str(row["source_sample_id"]) for row in rows}
        if len(families) != 1 or len(samples) != 1:
            rejected[pair_id] = "PAIR_PROVENANCE_MISMATCH"
            continue
        # A paired boundary unit earns canonical status only after the
        # unchanged teacher actually observed both sides.  Planned offsets
        # never determine membership or labels.
        if {bool(row["target"]) for row in rows} != {False, True}:
            rejected[pair_id] = "PAIR_DOES_NOT_OBSERVE_BOTH_TEACHER_LABELS"
            continue
        canonical.extend(rows)
    return canonical, {
        "raw_pair_count": len(by_pair),
        "canonical_pair_count": len(by_pair) - len(rejected),
        "rejected_pair_count": len(rejected),
        "rejected_pair_reasons": dict(Counter(rejected.values())),
    }


def _split_units(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Atomically split pairs with source families disjoint across splits."""

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    by_family: dict[str, list[str]] = defaultdict(list)
    for pair_id, unit in by_pair.items():
        family = str(unit[0]["source_family_id"])
        if any(str(row["source_family_id"]) != family for row in unit):
            raise BoundaryAuditError("PAIR_FAMILY_MISMATCH")
        by_family[family].append(pair_id)
    buckets = {"train": set(), "validation": set(), "heldout": set()}
    # Source-family-disjoint evaluation requires a family to stay in one
    # bucket.  Since every canonical pair already observed both labels, the
    # family can be safely assigned as an atomic balanced provenance unit.
    counts = {name: {"positive": 0, "negative": 0} for name in buckets}
    base_destinations = (
        *("train",) * REQUIRED_SOURCE_FAMILIES_BY_SPLIT["train"],
        *("validation",) * REQUIRED_SOURCE_FAMILIES_BY_SPLIT["validation"],
        *("heldout",) * REQUIRED_SOURCE_FAMILIES_BY_SPLIT["heldout"],
    )
    for family_index, family in enumerate(
        sorted(by_family, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    ):
        units = sorted(by_family[family], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        if family_index < len(base_destinations):
            destination = base_destinations[family_index]
        else:
            # The requested evaluation allocation is exactly 3/3/2.  Extra
            # usable families are retained in the canonical dataset but stay
            # reserve-only; silently adding them to a split would change the
            # declared source-family generalization contract.
            continue
        for pair_id in units:
            unit_counts = _counts(by_pair[pair_id])
            buckets[destination].add(pair_id)
            for label in ("positive", "negative"):
                counts[destination][label] += unit_counts[label]
    return buckets


def _quality(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["ROC_AUC"]) >= 0.90
        and float(metrics["PR_AUC"]) >= 0.90
        and float(metrics["BRIER"]) <= 0.10
        and float(metrics["FALSE_ACCEPT_RATE"]) <= 0.10
        and float(metrics["FALSE_REJECT_RATE"]) <= 0.10
        and float(metrics["TARGET_1_SCORE"]["mean"]) > float(metrics["TARGET_0_SCORE"]["mean"])
        and not bool(metrics["ALWAYS_READY_COLLAPSE"])
    )


def _select_validation_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Choose a frozen decision threshold from validation only.

    The contract constrains both false-accept and false-reject rates.  When
    more than one threshold satisfies those limits, choose the one with the
    strongest F1, then the lower false-accept rate.  Heldout rows never enter
    this selection.
    """

    candidates: list[tuple[bool, float, dict[str, Any]]] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = _evaluation(y, score, threshold=float(threshold))
        contract_ok = bool(
            metrics["FALSE_ACCEPT_RATE"] <= 0.10
            and metrics["FALSE_REJECT_RATE"] <= 0.10
        )
        candidates.append((contract_ok, float(threshold), metrics))
    eligible = [item for item in candidates if item[0]]
    pool = eligible if eligible else candidates
    # False-admit is the safety-priority tie-breaker when no viable threshold
    # exists; this is still a validation-only choice and never alters a gate.
    selected = max(
        pool,
        key=lambda item: (
            int(item[0]),
            float(item[2]["F1"]),
            -float(item[2]["FALSE_ACCEPT_RATE"]),
            -float(item[2]["FALSE_REJECT_RATE"]),
            -abs(item[1] - 0.5),
        ),
    )
    return selected[1], selected[2]


def audit(sidecars: list[Path]) -> dict[str, Any]:
    raw_records = [row for path in sidecars for row in _load_rows(path)]
    canonical, pair_summary = _canonical_paired_groups(raw_records)
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        family_rows[str(row["source_family_id"])].append(row)
    family_receipt = {
        family: {
            **_counts(rows),
            "split_eligible": bool({bool(row["target"]) for row in rows} == {False, True}),
        }
        for family, rows in sorted(family_rows.items())
    }
    base: dict[str, Any] = {
        "SCHEMA": SCHEMA,
        "OFFLINE_ONLY": True,
        "ISAAC_STARTED": "NO",
        "SAC_UPDATE": 0,
        "STUDENT_OPTIMIZER_UPDATE": 0,
        "PRIVILEGED_HARD_GATE": "NO",
        "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
        "WANDB_TRAINING_RUN": "NO",
        "RAW_ROW_COUNT": len(raw_records),
        "BOUNDARY_PAIRED_ROW_COUNT": len(canonical),
        # "TOTAL" is canonical/usable teacher provenance, not every planned
        # catalog family.  A planned source becomes usable only after the
        # unchanged teacher observed both labels in one paired unit.
        "TOTAL_SOURCE_FAMILY_COUNT": len(family_rows),
        "SOURCE_FAMILY_COUNT": len(family_rows),
        "USABLE_PAIRED_FAMILY_COUNT": sum(
            int(receipt["split_eligible"]) for receipt in family_receipt.values()
        ),
        "FAMILY_LABEL_COUNTS": family_receipt,
        "POSITIVE_COUNT": _counts(canonical)["positive"],
        "NEGATIVE_COUNT": _counts(canonical)["negative"],
        "PAIR_SUMMARY": pair_summary,
        "SOURCE_LABEL_CONFOUNDING": "YES",
        "SOURCE_FAMILY_DISJOINT_SPLIT": "NO",
        "TRAIN_POS_NEG": None,
        "VALIDATION_POS_NEG": None,
        "HELDOUT_POS_NEG": None,
        "VALIDATION_ROC_AUC": None,
        "HELDOUT_ROC_AUC": None,
        "VALIDATION_PR_AUC": None,
        "HELDOUT_PR_AUC": None,
        "VALIDATION_FALSE_ACCEPT": None,
        "HELDOUT_FALSE_ACCEPT": None,
        "VALIDATION_FALSE_REJECT": None,
        "HELDOUT_FALSE_REJECT": None,
        "VALIDATION_SELECTED_THRESHOLD": None,
        "THRESHOLD_SOURCE": "VALIDATION_ONLY",
        "OFFLINE_CONTRACT_PASS": False,
        "3K_AUTHORIZED": "NO",
        "NEXT": "EXPAND_BOUNDARY_BALANCED_DATASET",
    }
    if not canonical:
        return base
    units = _split_units(canonical)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        by_pair[row["pair_id"]].append(row)
    split = {
        name: [row for pair_id in pair_ids for row in by_pair[pair_id]]
        for name, pair_ids in units.items()
    }
    split_counts = {name: _counts(rows) for name, rows in split.items()}
    split_families = {
        name: sorted({str(row["source_family_id"]) for row in rows})
        for name, rows in split.items()
    }
    family_assignment = {
        family: next(
            (name for name, families in split_families.items() if family in families),
            "reserve",
        )
        for family in family_receipt
    }
    for family, assignment in family_assignment.items():
        family_receipt[family]["assigned_split"] = assignment
    # No pair/source-sample or source family can leak across splits.  Since a
    # canonical pair includes both labels, every qualified source family also
    # has label variation independent of its split identity.
    pair_membership = Counter(pair for bucket in units.values() for pair in bucket)
    source_split_count = Counter(family for families in split_families.values() for family in families)
    family_targets: dict[str, set[bool]] = defaultdict(set)
    for row in canonical:
        family_targets[str(row["source_family_id"])].add(bool(row["target"]))
    source_family_disjoint = bool(
        len(family_targets) >= MIN_SOURCE_FAMILIES_TOTAL
        and sum(len(families) for families in split_families.values()) == MIN_SOURCE_FAMILIES_TOTAL
        and all(count == 1 for count in source_split_count.values())
        and all(
            len(split_families[name]) >= REQUIRED_SOURCE_FAMILIES_BY_SPLIT[name]
            for name in split_families
        )
    )
    family_label_balanced = bool(
        all(targets == {False, True} for targets in family_targets.values())
    )
    split_class_coverage = bool(
        all(split_counts[name][label] >= MIN_ROWS_PER_CLASS for name in split for label in ("positive", "negative"))
    )
    batch_label_confounded = not all(
        {bool(row["target"]) for row in canonical if row["collection_batch_id"] == batch}
        == {False, True}
        for batch in {str(row["collection_batch_id"]) for row in canonical}
    )
    data_contract = bool(
        split_class_coverage
        and source_family_disjoint
        and family_label_balanced
        and not batch_label_confounded
        and all(count == 1 for count in pair_membership.values())
        and all({row["target"] for row in rows} == {False, True} for rows in split.values())
    )
    base.update({
        "TRAIN_POS_NEG": split_counts["train"],
        "VALIDATION_POS_NEG": split_counts["validation"],
        "HELDOUT_POS_NEG": split_counts["heldout"],
        "SPLIT_PAIR_COUNTS": {name: len(value) for name, value in units.items()},
        "SPLIT_SOURCE_FAMILIES": split_families,
        "TRAIN_ELIGIBLE_FAMILIES": [
            family for family in split_families["train"]
            if family_receipt[family]["split_eligible"]
        ],
        "VALIDATION_ELIGIBLE_FAMILIES": [
            family for family in split_families["validation"]
            if family_receipt[family]["split_eligible"]
        ],
        "HELDOUT_ELIGIBLE_FAMILIES": [
            family for family in split_families["heldout"]
            if family_receipt[family]["split_eligible"]
        ],
        "RESERVE_ELIGIBLE_FAMILIES": [
            family for family, assignment in family_assignment.items()
            if assignment == "reserve" and family_receipt[family]["split_eligible"]
        ],
        "REQUIRED_SOURCE_FAMILIES_BY_SPLIT": REQUIRED_SOURCE_FAMILIES_BY_SPLIT,
        "PAIR_LEAKAGE_COUNT": sum(count - 1 for count in pair_membership.values() if count > 1),
        "SOURCE_FAMILY_BALANCED_ACROSS_SPLITS": "NO",
        "SOURCE_FAMILY_DISJOINT_SPLIT": "YES" if source_family_disjoint else "NO",
        "SOURCE_FAMILY_LABEL_BOTH_CLASSES": "YES" if family_label_balanced else "NO",
        "COLLECTION_BATCH_LABEL_CONFOUNDING": "YES" if batch_label_confounded else "NO",
        "SOURCE_LABEL_CONFOUNDING": (
            "NO" if source_family_disjoint and family_label_balanced else "YES"
        ),
        "DATA_SPLIT_CONTRACT_PASS": data_contract,
    })
    if not data_contract:
        return base
    # Offline linear probe is only a separability check.  It is not the
    # deployed student, writes no checkpoint and performs no Student optimizer
    # update; SAC and runtime are entirely untouched.
    x_train = np.stack([row["feature"] for row in split["train"]])
    y_train = np.asarray([int(row["target"]) for row in split["train"]], dtype=np.int64)
    weights, bias, scaler = _fit_logistic(x_train, y_train)
    x_validation = np.stack([row["feature"] for row in split["validation"]])
    y_validation = np.asarray(
        [int(row["target"]) for row in split["validation"]], dtype=np.int64
    )
    validation_score = _predict_logistic(x_validation, weights, bias, scaler)
    selected_threshold, validation_evaluation = _select_validation_threshold(
        y_validation, validation_score
    )
    evaluation: dict[str, dict[str, Any]] = {"validation": validation_evaluation}
    x_heldout = np.stack([row["feature"] for row in split["heldout"]])
    y_heldout = np.asarray(
        [int(row["target"]) for row in split["heldout"]], dtype=np.int64
    )
    evaluation["heldout"] = _evaluation(
        y_heldout,
        _predict_logistic(x_heldout, weights, bias, scaler),
        threshold=selected_threshold,
    )
    qualified = bool(_quality(evaluation["validation"]) and _quality(evaluation["heldout"]))
    base.update({
        "OFFLINE_LINEAR_PROBE_ONLY": True,
        "VALIDATION_ROC_AUC": evaluation["validation"]["ROC_AUC"],
        "HELDOUT_ROC_AUC": evaluation["heldout"]["ROC_AUC"],
        "VALIDATION_PR_AUC": evaluation["validation"]["PR_AUC"],
        "HELDOUT_PR_AUC": evaluation["heldout"]["PR_AUC"],
        "VALIDATION_BRIER": evaluation["validation"]["BRIER"],
        "HELDOUT_BRIER": evaluation["heldout"]["BRIER"],
        "VALIDATION_FALSE_ACCEPT": evaluation["validation"]["FALSE_ACCEPT_RATE"],
        "HELDOUT_FALSE_ACCEPT": evaluation["heldout"]["FALSE_ACCEPT_RATE"],
        "VALIDATION_FALSE_REJECT": evaluation["validation"]["FALSE_REJECT_RATE"],
        "HELDOUT_FALSE_REJECT": evaluation["heldout"]["FALSE_REJECT_RATE"],
        "VALIDATION_SELECTED_THRESHOLD": selected_threshold,
        "THRESHOLD_SOURCE": "VALIDATION_ONLY",
        "VALIDATION_POSITIVE_SCORE_MEAN": evaluation["validation"]["TARGET_1_SCORE"]["mean"],
        "VALIDATION_NEGATIVE_SCORE_MEAN": evaluation["validation"]["TARGET_0_SCORE"]["mean"],
        "HELDOUT_POSITIVE_SCORE_MEAN": evaluation["heldout"]["TARGET_1_SCORE"]["mean"],
        "HELDOUT_NEGATIVE_SCORE_MEAN": evaluation["heldout"]["TARGET_0_SCORE"]["mean"],
        "OFFLINE_CONTRACT_PASS": qualified,
        "3K_AUTHORIZED": "YES" if qualified else "NO",
        "NEXT": "RUN_CORRECTED_PRIVILEGED_DISTILLATION_3K" if qualified else "EXPAND_BOUNDARY_BALANCED_DATASET",
    })
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("BOUNDARY_PAIRED_AUDIT_REFUSES_OVERWRITE")
    report = audit(args.sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
