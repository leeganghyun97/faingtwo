# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure regression tests for the historical Stage-1A canonical audit CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_g2_stage1a_canonical_methods.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage1a_canonical_methods_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_report(*, stable: float = 0.2) -> dict[str, object]:
    return {
        "accepted_transitions": 3000,
        "num_envs": 10,
        "episode_count_observed": 10,
        "CONTACT_RATE": 0.8,
        "BILATERAL_RATE": 0.5,
        "STABLE_RATE": stable,
        "CONTACT_TO_BILATERAL_CONVERSION": 0.625,
        "BILATERAL_TO_STABLE_CONVERSION": stable / 0.5,
        "RUNTIME_HARDSTOP": 0,
        "FORBIDDEN_COLLISION": 0,
        "ACTION_BOUND_VIOLATION": 0,
        "GRIPPER_AUTHORITY_VIOLATION": 0,
        "source_freeze_match": True,
    }


def _all_historical(module):
    return {
        method: (Path(f"/{index}.json"), _legacy_report())
        for index, method in enumerate(module.HISTORICAL_REPORT_RELATIVE_PATHS)
    }


def test_historical_matrix_uses_stable_as_the_only_grasp_success() -> None:
    module = _module()
    reports = _all_historical(module)
    # The known incomplete 10-env FSM run remains visibly non-evaluable.
    reports["FSM advisory + SAC"] = (None, None)
    audit = module.build_canonical_method_audit(historical_reports=reports)

    assert audit["READ_ONLY"] is True
    assert audit["COMMON_GRASP_DEFINITIONS"]["GRASP_SUCCESS_AUTHORITY"] == "STABLE_ONLY"
    assert len(audit["METHOD_ROOT_CAUSE_MATRIX"]) == 11
    first = audit["METHOD_ROOT_CAUSE_MATRIX"][0]["CANONICAL_METRICS"]
    assert first["GRASP_SUCCESS_STABLE_RATE"] == 0.2
    assert first["BILATERAL_CONTACT_CANDIDATE_RATE"] == 0.5
    assert first["POST_BILATERAL_CONTACT_LOSS_RATE"] == 0.6
    assert first["POST_BILATERAL_CONTACT_LOSS_SOURCE"].startswith("DERIVED_LEGACY_PROXY")
    incomplete = next(row for row in audit["METHOD_ROOT_CAUSE_MATRIX"] if row["METHOD"] == "FSM advisory + SAC")
    assert incomplete["SAFETY"]["SAFETY_STATUS"] == "NOT_EVALUABLE_NO_FINAL_REPORT"
    assert audit["COMMON_RESET_FIX"] == "NOT_REVALIDATED"


def test_post_reset_fair_report_requires_reset_evidence_and_exact_10env_6k() -> None:
    module = _module()
    fair = _legacy_report(stable=0.4) | {
        "accepted_transitions": 6000,
        "num_envs": 10,
        "OPEN_RESTORE_PASS": "YES",
        "RESET_PARITY_PASS": "YES",
        "TEACHER_RECEIPT_PARITY_PASS": "YES",
        "SAME_SOURCE_INITIAL_STATE_PARITY": "YES",
        "PREVIOUS_EPISODE_GEOMETRY_USED": 0,
    }
    audit = module.build_canonical_method_audit(
        historical_reports=_all_historical(module),
        fair_reports={"V3.1 lateral-off": (Path("/fair.json"), fair)},
    )

    entry = audit["POST_RESET_FAIR_REPORTS"][0]
    assert entry["FAIR_6K_SHAPE_PASS"] is True
    assert entry["FAIR_6K_REPORT_PASS"] is True
    assert entry["RESET_EVIDENCE"]["RESET_PARITY_EVIDENCE"] == "PASS"
    assert audit["COMMON_RESET_FIX"] == "PASS"
    assert audit["NEXT_MINIMAL_EXPERIMENT"].startswith("FAIR_10ENV_6K")


def test_fair_spec_parser_rejects_ambiguous_input() -> None:
    module = _module()
    try:
        module._parse_fair_report_spec("not-a-spec")
    except ValueError as error:
        assert str(error) == "FAIR_REPORT_MUST_BE_METHOD_EQUALS_PATH"
    else:
        raise AssertionError("missing METHOD=PATH validation")
