#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only canonical audit for historical Stage-1A grasp methods.

This utility normalizes the heterogeneous, legacy Stage-1A reports into one
episode-level grasp schema.  In particular, it never calls Contact or
Bilateral a grasp success: only a 10-step Stable Grasp is success.

The audit reads JSON artifacts only.  It neither imports Isaac nor opens a
replay buffer, checkpoint, controller, or W&B run.  Optional ``--fair-report``
arguments attach post-reset 10-env/6K reports to the same matrix, allowing a
single immutable audit output to distinguish historical evidence from a fair
retest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "g2_stage1a_canonical_method_audit_v1"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("GENIESIM_OUTPUT_ROOT", ROOT / "output"))


# The paths are deliberately explicit.  They are the approved historical
# artifacts, not globs that might accidentally pick up an unfinished retry.
HISTORICAL_REPORT_RELATIVE_PATHS: dict[str, str | None] = {
    "V3 CURRENT": (
        "g2_stage1a_current_reward_v3_her_force_10env_3k_freshretry_"
        "20260927_r3/attempt-01/runtime/STAGE1A_VECTOR_REPORT.json"
    ),
    "V3.1": "g2_stage1a_her_force_vector10_v31_3k_20260927_r2/runtime/STAGE1A_VECTOR_REPORT.json",
    "V3.1 lateral-off": (
        "g2_stage1a_her_force_vector10_v31_lateral_off_paired_3k_"
        "20260927_r5/runtime/STAGE1A_VECTOR_REPORT.json"
    ),
    "V3.2": "g2_stage1a_her_force_vector10_v32_3k_20260927_r1/runtime/STAGE1A_VECTOR_REPORT.json",
    "Privileged geometry hard gate": (
        "g2_stage1a_privileged_geometry_teacher_her_force_3k_"
        "20260927_r3/runtime/STAGE1A_VECTOR_REPORT.json"
    ),
    "Privileged teacher-only distillation": (
        "g2_stage1a_v31_lateral_off_privileged_distillation_her_force_"
        "3k_20260927_r1/STAGE1A_VECTOR_REPORT.json"
    ),
    "Corrected privileged distillation": (
        "g2_stage1a_v31_lateral_off_corrected_privileged_distillation_"
        "her_force_3k_20260928_r1/STAGE1A_VECTOR_REPORT.json"
    ),
    "Spatial-missingness Wrist RGB-D student": (
        "g2_stage1a_wrist_v6_diagnostic_20260928_140659/"
        "offline_v6_spatial_missingness_ablation/REPORT.json"
    ),
    "Family-invariant Conditional CORAL student": (
        "g2_stage1a_v6_independent_revalidation_20260928_145156/"
        "FAMILY_INVARIANT_RETRAINING_SHARED_MARGIN/REPORT.json"
    ),
    # The 10-env FSM/advisory execution stopped before a final report.  Keep
    # this explicit absent entry in the matrix rather than silently treating
    # the partial CSV as a completed RL result.
    "FSM advisory + SAC": None,
    "Current 15K run": (
        "g2_stage1a_v31_lateral_off_fsm_advisory_25env_15k_20260928/"
        "evaluation/checkpoint_15000.json"
    ),
}


METHOD_ROOT_CAUSES: dict[str, dict[str, Any]] = {
    "V3 CURRENT": {
        "primary_failure": "No isolated historical causal receipt; Contact-to-Bilateral and Bilateral-to-Stable are both limited.",
        "secondary_failure": "Historical reset lifecycle can retain measured four-bar state.",
        "proven_cause": "RESET_CONFOUNDED_HISTORICAL_BASELINE",
        "fix_required": "Apply and validate common measured OPEN-restore/fresh-geometry reset contract only.",
        "retest_required": True,
        "status": "RETEST_AFTER_RESET_PARITY_AS_CONTROL",
        "fair_6k_candidate": True,
    },
    "V3.1": {
        "primary_failure": "lateral<=10 mm hard CLOSE gate rejects valid CLOSE opportunity.",
        "secondary_failure": "Low Stable conversion and common reset confound.",
        "proven_cause": "LATERAL_HARD_GATE_OVER_CONSERVATIVE",
        "fix_required": "Do not restore the lateral hard gate; retain lateral telemetry only.",
        "retest_required": False,
        "status": "REJECT_HARD_LATERAL_GATE",
        "fair_6k_candidate": False,
    },
    "V3.1 lateral-off": {
        "primary_failure": "Historical Contact-to-Bilateral and Bilateral-to-Stable losses remain.",
        "secondary_failure": "Common reset lifecycle can create premature contact before FSM CLOSE.",
        "proven_cause": "LATERAL_GATE_REMOVAL_RECOVERS_CLOSE_OPPORTUNITY_BUT_NOT_STABILITY",
        "fix_required": "Common reset fix before any one-variable contact/stability investigation.",
        "retest_required": True,
        "status": "KEEP_AND_RETEST_AFTER_RESET_PARITY",
        "fair_6k_candidate": True,
    },
    "V3.2": {
        "primary_failure": "Multi-change package is not superior to V3.1 lateral-off at equal 3K budget.",
        "secondary_failure": "Individual effects of shaping, hover, and dwell changes are not causally separable.",
        "proven_cause": "NO_EQUAL_BUDGET_ADVANTAGE_OVER_LATERAL_OFF",
        "fix_required": "Do not expand package; isolate one root cause only after reset parity.",
        "retest_required": False,
        "status": "REJECT_MULTI_CHANGE_PACKAGE",
        "fair_6k_candidate": False,
    },
    "Privileged geometry hard gate": {
        "primary_failure": "Privileged geometry is non-deployable runtime CLOSE authority and reduces opportunity in the historical 3K run.",
        "secondary_failure": "Historical reset/open parity affected paired context outcomes.",
        "proven_cause": "NON_DEPLOYABLE_PRIVILEGED_HARD_AUTHORITY",
        "fix_required": "Restrict privileged geometry to teacher, label, and diagnostic roles.",
        "retest_required": False,
        "status": "REJECT_RUNTIME_HARD_GATE_KEEP_TEACHER",
        "fair_6k_candidate": False,
    },
    "Privileged teacher-only distillation": {
        "primary_failure": "Original pre-CLOSE teacher data had no real negative rows; student collapsed toward always-ready.",
        "secondary_failure": "Bilateral-to-Stable conversion was materially worse than lateral-off.",
        "proven_cause": "PRE_CLOSE_NEGATIVE_CLASS_ABSENT",
        "fix_required": "Do not reuse this head/dataset; use boundary-balanced, provenance-disjoint teacher data only.",
        "retest_required": False,
        "status": "REJECT_EARLY_DISTILLATION_CONTRACT",
        "fair_6k_candidate": False,
    },
    "Corrected privileged distillation": {
        "primary_failure": "Corrected binary labels did not close the offline/runtime score-distribution gap.",
        "secondary_failure": "Online false-accept remained high; stable conversion regressed.",
        "proven_cause": "ONLINE_OFFLINE_DISTILLATION_CONTRACT_MISMATCH",
        "fix_required": "Use only a frozen, independently validated advisory representation; never a hard veto.",
        "retest_required": False,
        "status": "SUPERSEDED_BY_ADVISORY_ONLY_PATH",
        "fair_6k_candidate": False,
    },
    "Spatial-missingness Wrist RGB-D student": {
        "primary_failure": "Independent-family calibration collapse despite preserved score ranking.",
        "secondary_failure": "RGB/topology branch activation creates family score offsets.",
        "proven_cause": "FAMILY_SPECIFIC_BRANCH_ACTIVATION_SHIFT",
        "fix_required": "Do not use as a hard gate; retain only as representation evidence for family-invariant training.",
        "retest_required": False,
        "status": "REJECT_AS_RUNTIME_GATE",
        "fair_6k_candidate": False,
    },
    "Family-invariant Conditional CORAL student": {
        "primary_failure": "Single global hard threshold does not transfer to worst heldout family.",
        "secondary_failure": "High false-reject at conservative operating point.",
        "proven_cause": "SINGLE_THRESHOLD_FAMILY_TRANSFER_LIMITATION",
        "fix_required": "Frozen high-confidence advisory plus deterministic-FSM defer; no student veto.",
        "retest_required": True,
        "status": "KEEP_FROZEN_ADVISORY_ONLY",
        "fair_6k_candidate": True,
    },
    "FSM advisory + SAC": {
        "primary_failure": "Historical 10-env execution has no completed final report.",
        "secondary_failure": "Any partial rows are not comparable to completed-episode metrics.",
        "proven_cause": "INCOMPLETE_HISTORICAL_RUNTIME",
        "fix_required": "Run only after common reset parity PASS; student remains advisory only.",
        "retest_required": True,
        "status": "RETEST_AFTER_RESET_PARITY",
        "fair_6k_candidate": True,
    },
    "Current 15K run": {
        "primary_failure": "Late-window Contact-to-Bilateral and Bilateral-to-Stable degradation.",
        "secondary_failure": "Reset/open parity can produce premature contact before FSM CLOSE; 25-env result is not a direct 10-env comparison.",
        "proven_cause": "RESET_CONFOUNDED_POST_CONTACT_STABILITY_FAILURE",
        "fix_required": "Do not extend; first prove reset parity and re-run a normalized 10-env comparison.",
        "retest_required": True,
        "status": "DESCRIPTIVE_ONLY_RETEST_AFTER_RESET_PARITY",
        "fair_6k_candidate": False,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"AUDIT_REPORT_NOT_OBJECT:{path}")
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _canonical_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy vector reports and checkpoint summaries without guessing.

    The new report schema may contain a nested ``GRASP_EVALUATION`` map.  The
    historical artifacts either expose uppercase vector keys or a lower-case
    checkpoint ``cumulative`` map.  Only absent post-bilateral loss is
    derived, and that derivation is visibly marked as a proxy.
    """

    grasp = report.get("GRASP_EVALUATION")
    if isinstance(grasp, Mapping):
        source: Mapping[str, Any] = grasp
    elif isinstance(report.get("cumulative"), Mapping):
        source = report["cumulative"]
    else:
        source = report

    episode_count = _finite_float(
        _first(source, "COMPLETED_EPISODE_COUNT", "episode_count", "episode_count_observed")
    )
    contact_rate = _finite_float(_first(source, "CONTACT_RATE", "contact_rate", "V3_1_CONTACT_RATE"))
    bilateral_rate = _finite_float(
        _first(source, "BILATERAL_CONTACT_CANDIDATE_RATE", "bilateral_rate", "BILATERAL_RATE", "V3_1_BILATERAL_RATE")
    )
    stable_rate = _finite_float(
        _first(source, "GRASP_SUCCESS_STABLE_RATE", "stable_rate", "STABLE_RATE", "V3_1_STABLE_RATE")
    )
    contact_to_bilateral = _finite_float(
        _first(source, "CONTACT_TO_BILATERAL", "contact_to_bilateral", "CONTACT_TO_BILATERAL_CONVERSION")
    )
    bilateral_to_stable = _finite_float(
        _first(source, "BILATERAL_TO_STABLE", "bilateral_to_stable", "BILATERAL_TO_STABLE_CONVERSION")
    )
    loss = _finite_float(
        _first(source, "POST_BILATERAL_CONTACT_LOSS_RATE", "contact_loss_after_bilateral_rate")
    )
    loss_source = "REPORTED"
    if loss is None and bilateral_to_stable is not None:
        # Legacy reports did not distinguish a loss from a non-loss stable
        # timeout.  Preserve that limitation rather than relabeling it.
        loss = max(0.0, min(1.0, 1.0 - bilateral_to_stable))
        loss_source = "DERIVED_LEGACY_PROXY_1_MINUS_BILATERAL_TO_STABLE"

    return {
        "EVALUATION_DENOMINATOR": _first(
            source, "EVALUATION_DENOMINATOR", "metric_denominator"
        ) or "COMPLETED_EPISODES_ONLY_IF_REPORTED",
        "COMPLETED_EPISODE_COUNT": None if episode_count is None else int(episode_count),
        "CONTACT_RATE": contact_rate,
        "BILATERAL_CONTACT_CANDIDATE_RATE": bilateral_rate,
        "GRASP_SUCCESS_STABLE_RATE": stable_rate,
        "CONTACT_TO_BILATERAL": contact_to_bilateral,
        "BILATERAL_TO_STABLE": bilateral_to_stable,
        "POST_BILATERAL_CONTACT_LOSS_RATE": loss,
        "POST_BILATERAL_CONTACT_LOSS_SOURCE": loss_source,
        "CONTACT_EPISODE_COUNT": _finite_float(_first(source, "contact_episode_count")),
        "BILATERAL_EPISODE_COUNT": _finite_float(_first(source, "bilateral_episode_count")),
        "STABLE_GRASP_EPISODE_COUNT": _finite_float(
            _first(source, "stable_episode_count", "success_episode_count")
        ),
    }


def _safety(report: Mapping[str, Any]) -> dict[str, Any]:
    source = report.get("safety") if isinstance(report.get("safety"), Mapping) else report
    keys = {
        "RUNTIME_HARDSTOP": ("runtime_hardstop", "RUNTIME_HARDSTOP"),
        "FORBIDDEN_COLLISION": ("forbidden_collision", "FORBIDDEN_COLLISION"),
        "ACTION_BOUND_VIOLATION": ("action_bound_violation", "ACTION_BOUND_VIOLATION"),
        "GRIPPER_AUTHORITY_VIOLATION": (
            "gripper_authority_violation",
            "GRIPPER_AUTHORITY_VIOLATION",
        ),
    }
    values = {name: _finite_float(_first(source, *aliases)) for name, aliases in keys.items()}
    known = [value for value in values.values() if value is not None]
    values["SAFETY_STATUS"] = (
        "PASS" if len(known) == len(keys) and all(value == 0.0 for value in known)
        else "INCOMPLETE_OR_NONZERO"
    )
    return values


def _method_entry(
    method: str,
    path: Path | None,
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root_cause = METHOD_ROOT_CAUSES[method]
    entry: dict[str, Any] = {
        "METHOD": method,
        "REPORT_PATH": None if path is None else str(path),
        "HISTORICAL_REPORT_AVAILABLE": report is not None,
        "PRIMARY_FAILURE": root_cause["primary_failure"],
        "SECONDARY_FAILURE": root_cause["secondary_failure"],
        "PROVEN_CAUSE": root_cause["proven_cause"],
        "FIX_REQUIRED": root_cause["fix_required"],
        "RETEST_REQUIRED": bool(root_cause["retest_required"]),
        "STATUS": root_cause["status"],
        "FAIR_6K_CANDIDATE_AFTER_RESET_PARITY": bool(root_cause["fair_6k_candidate"]),
    }
    if report is None:
        entry.update({
            "CANONICAL_METRICS": None,
            "SAFETY": {"SAFETY_STATUS": "NOT_EVALUABLE_NO_FINAL_REPORT"},
        })
    else:
        entry["CANONICAL_METRICS"] = _canonical_metrics(report)
        entry["SAFETY"] = (
            {"SAFETY_STATUS": "NOT_APPLICABLE_OFFLINE"}
            if bool(report.get("OFFLINE_ONLY"))
            else _safety(report)
        )
        entry["RUNTIME_VARIANT"] = report.get("runtime_variant")
        entry["ACCEPTED_TRANSITIONS"] = _finite_float(report.get("accepted_transitions"))
        entry["NUM_ENVS"] = _finite_float(report.get("num_envs"))
    return entry


def _parse_fair_report_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("FAIR_REPORT_MUST_BE_METHOD_EQUALS_PATH")
    method, text_path = value.split("=", 1)
    method = method.strip()
    path = Path(text_path.strip())
    if not method or not str(path):
        raise ValueError("FAIR_REPORT_MUST_BE_METHOD_EQUALS_PATH")
    return method, path


def _bool_or_missing(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.upper() in {"YES", "NO", "PASS", "FAIL"}:
        return value.upper() in {"YES", "PASS"}
    return None


def _reset_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    nested = report.get("RESET_OPEN_RESTORE")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else report
    aliases = {
        "OPEN_RESTORE_PASS": ("OPEN_RESTORE_PASS", "open_restore_pass"),
        "RESET_PARITY_PASS": ("RESET_PARITY_PASS", "reset_parity_pass"),
        "TEACHER_RECEIPT_PARITY_PASS": (
            "TEACHER_RECEIPT_PARITY_PASS",
            "teacher_receipt_parity_pass",
        ),
        "SAME_SOURCE_INITIAL_STATE_PARITY": (
            "SAME_SOURCE_INITIAL_STATE_PARITY",
            "same_source_initial_state_parity",
        ),
    }
    values = {name: _bool_or_missing(_first(source, *names)) for name, names in aliases.items()}
    previous_geometry = _finite_float(
        _first(source, "PREVIOUS_EPISODE_GEOMETRY_USED", "previous_episode_geometry_used")
    )
    values["PREVIOUS_EPISODE_GEOMETRY_USED"] = previous_geometry
    evidence_present = all(value is not None for value in values.values())
    values["RESET_PARITY_EVIDENCE"] = (
        "PASS"
        if evidence_present and all(value is True for name, value in values.items() if name != "PREVIOUS_EPISODE_GEOMETRY_USED") and previous_geometry == 0.0
        else "FAIL" if evidence_present else "MISSING"
    )
    return values


def _fair_entry(method: str, path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _canonical_metrics(report)
    safety = _safety(report)
    accepted = _finite_float(report.get("accepted_transitions"))
    num_envs = _finite_float(report.get("num_envs"))
    source_freeze = _bool_or_missing(report.get("source_freeze_match"))
    reset = _reset_evidence(report)
    shape_ok = accepted == 6000.0 and num_envs == 10.0
    eligible = bool(
        shape_ok
        and source_freeze is True
        and safety["SAFETY_STATUS"] == "PASS"
        and reset["RESET_PARITY_EVIDENCE"] == "PASS"
    )
    return {
        "METHOD": method,
        "REPORT_PATH": str(path),
        "CANONICAL_METRICS": metrics,
        "SAFETY": safety,
        "ACCEPTED_TRANSITIONS": accepted,
        "NUM_ENVS": num_envs,
        "SOURCE_FREEZE_MATCH": source_freeze,
        "RESET_EVIDENCE": reset,
        "FAIR_6K_SHAPE_PASS": shape_ok,
        "FAIR_6K_REPORT_PASS": eligible,
    }


def build_canonical_method_audit(
    *,
    historical_reports: Mapping[str, tuple[Path | None, Mapping[str, Any] | None]],
    fair_reports: Mapping[str, tuple[Path, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable audit without reading or writing files."""

    missing = set(HISTORICAL_REPORT_RELATIVE_PATHS) - set(historical_reports)
    if missing:
        raise ValueError(f"MISSING_HISTORICAL_METHODS:{','.join(sorted(missing))}")
    matrix = [
        _method_entry(method, *historical_reports[method])
        for method in HISTORICAL_REPORT_RELATIVE_PATHS
    ]
    fair = [
        _fair_entry(method, path, report)
        for method, (path, report) in sorted((fair_reports or {}).items())
    ]
    reset_states = [entry["RESET_EVIDENCE"]["RESET_PARITY_EVIDENCE"] for entry in fair]
    if "FAIL" in reset_states:
        common_reset = "FAIL"
    elif "PASS" in reset_states:
        common_reset = "PASS"
    else:
        common_reset = "NOT_REVALIDATED"

    return {
        "SCHEMA": SCHEMA,
        "READ_ONLY": True,
        "ISAAC_STARTED": "NO",
        "SAC_STARTED": "NO",
        "GRU_TRAINING_STARTED": "NO",
        "RUNTIME_GATE_CHANGED": "NO",
        "COMMON_GRASP_DEFINITIONS": {
            "CONTACT": "At least one primary pad contacts cube; not grasp success.",
            "BILATERAL": "Inner and outer primary pads simultaneously contact cube; grasp candidate only.",
            "GRASP_SUCCESS_AUTHORITY": "STABLE_ONLY",
            "STABLE_GRASP": "Bilateral contact maintained for 10 consecutive 50-Hz steps (about 200 ms).",
        },
        "COMMON_RESET_FIX": common_reset,
        "COMMON_RESET_REQUIRED_SEQUENCE": [
            "SOURCE_RESTORE",
            "CANONICAL_OPEN_HOLD",
            "MEASURED_MASTER_SETTLE",
            "PASSIVE_FOLLOWER_SETTLE",
            "APERTURE_PARITY",
            "FRESH_GEOMETRY_RECEIPT",
            "CACHE_FRESHNESS",
            "PREVIOUS_ACTION_CLOSE_LATCH_PERSISTENCE_RESET",
            "EPISODE_ACTIVE",
        ],
        "METHOD_ROOT_CAUSE_MATRIX": matrix,
        "POST_RESET_FAIR_REPORTS": fair,
        "BEST_PRE_RESET_METHOD": "V3.1 lateral-off",
        "BEST_VERIFIED_METHOD": (
            "PENDING_POST_RESET_FAIR_RETEST"
            if common_reset != "PASS"
            else "SELECT_BY_GRASP_SUCCESS_STABLE_RATE_FROM_POST_RESET_FAIR_REPORTS"
        ),
        "PRIMARY_REMAINING_BLOCKER": "RESET" if common_reset != "PASS" else "SELECT_FROM_FAIR_REPORTS",
        "METHODS_TO_KEEP": [
            "V3 CURRENT",
            "V3.1 lateral-off",
            "FSM advisory + SAC",
            "Family-invariant Conditional CORAL student (advisory only)",
        ],
        "METHODS_TO_REJECT": [
            "V3.1 hard lateral gate",
            "V3.2 multi-change package",
            "Privileged geometry hard gate",
            "Early privileged teacher-only distillation",
            "Spatial-missingness student as hard gate",
        ],
        "METHODS_REQUIRING_RETEST": [
            "V3 CURRENT",
            "V3.1 lateral-off",
            "FSM advisory + SAC",
        ],
        "NEXT_MINIMAL_EXPERIMENT": (
            "RESET_REPEAT_DIAGNOSTIC_4_SOURCES_X_3_REPEATS"
            if common_reset != "PASS"
            else "FAIR_10ENV_6K_V3_CURRENT_V31_LATERAL_OFF_FSM_AND_ADVISORY"
        ),
        "30K_LONG_RUN_AUTHORIZED": "NO",
    }


def _load_default_historical_reports(output_root: Path) -> dict[str, tuple[Path | None, Mapping[str, Any] | None]]:
    result: dict[str, tuple[Path | None, Mapping[str, Any] | None]] = {}
    for method, relative in HISTORICAL_REPORT_RELATIVE_PATHS.items():
        if relative is None:
            result[method] = (None, None)
            continue
        path = output_root / relative
        if not path.is_file():
            result[method] = (path, None)
            continue
        report = _read_json(path)
        # The final 15K evaluation keeps the completed-episode denominator in
        # ``evaluation/checkpoint_15000.json`` while the vector report keeps
        # run-shape and source-freeze evidence.  Join those two read-only
        # receipts so the matrix never loses its 25-env/15K provenance.
        if method == "Current 15K run":
            vector_path = path.parents[1] / "STAGE1A_VECTOR_REPORT.json"
            if vector_path.is_file():
                vector = _read_json(vector_path)
                merged = dict(vector)
                merged["cumulative"] = report.get("cumulative", {})
                merged["safety"] = report.get("safety", {})
                merged["checkpoint_step"] = report.get("checkpoint_step")
                report = merged
        result[method] = (path, report)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"AUDIT_OUTPUT_REFUSES_OVERWRITE:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fair-report",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Attach an optional post-reset 10-env/6K JSON report without changing it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        fair_specs = [_parse_fair_report_spec(value) for value in args.fair_report]
        names = [name for name, _path in fair_specs]
        if len(set(names)) != len(names):
            raise ValueError("DUPLICATE_FAIR_REPORT_METHOD")
        fair_reports = {name: (path, _read_json(path)) for name, path in fair_specs}
        report = build_canonical_method_audit(
            historical_reports=_load_default_historical_reports(args.output_root),
            fair_reports=fair_reports,
        )
        _atomic_json(args.output, report, overwrite=bool(args.overwrite))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "OUTPUT": str(args.output),
        "COMMON_RESET_FIX": report["COMMON_RESET_FIX"],
        "HISTORICAL_METHOD_COUNT": len(report["METHOD_ROOT_CAUSE_MATRIX"]),
        "POST_RESET_FAIR_REPORT_COUNT": len(report["POST_RESET_FAIR_REPORTS"]),
        "NEXT_MINIMAL_EXPERIMENT": report["NEXT_MINIMAL_EXPERIMENT"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
