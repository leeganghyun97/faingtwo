# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Regression coverage for the standalone reset-gated fair 6K auditor."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_g2_stage1a_fair_6k_comparison.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage1a_fair_6k_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset_report() -> dict[str, object]:
    return {
        "SOURCE_RESTORE_RECEIPT_PASS": True,
        "OPEN_RESTORE_PASS": True,
        "TEACHER_RECEIPT_PARITY_PASS": True,
        "SAME_SOURCE_INITIAL_STATE_PARITY": True,
        "RESET_PARITY_PASS": True,
        "SOURCE_FREEZE_MATCH": True,
        "PREVIOUS_EPISODE_GEOMETRY_USED": 0,
        "PREMATURE_CONTACT_COUNT": 0,
        "EXPECTED_TRIALS": 12,
        "TRIAL_COUNT": 12,
        "REPEATS_PER_SOURCE": 3,
        "EXACT_SOURCE_ROWS": [149, 153, 157, 169],
    }


def _fair_report(
    module, *, variant: str, stable: float, bilateral_to_stable: float, contact: float
) -> dict[str, object]:
    contact_count = 20
    bilateral_count = int(round(contact_count * 0.8))
    stable_count = int(round(bilateral_count * bilateral_to_stable))
    reward = {
        "reward_authority_sha256": "reward-hash",
        "schema": "reward-schema",
        "frame": "robot_root",
        "position_unit": "m",
        "control_hz": 50,
        "physics_hz": 500,
        "rgbd_hz": 25,
        "residual_alpha": 0.1,
        "effective_residual_maximum_m": 0.00045,
        "raw_residual_maximum_m": 0.0045,
        "sac_action_dim": 3,
        "sac_gripper_authority": False,
        "her_force_role": "PRIORITY_ONLY",
        "her_force": {
            "schema": "her-force-schema",
            "real_rows_only": True,
            "action_unchanged": True,
            "reward_unchanged": True,
        },
        "reward_v3": {"reward_v3_authority_sha256": "reward-v3-hash", "schema": "reward-v3-schema"},
    }
    return {
        "runtime_variant": variant,
        "fair_comparison_id": module.FAIR_COMPARISON_ID,
        "fair_comparison_contract": {
            "num_envs": 10,
            "accepted_transitions": 6000,
            "replay_strategy": "HER_FORCE",
            "wandb_mode": "online",
            "same_seed_required": True,
            "reset_open_restore_required": True,
        },
        "training_seed": 42,
        "num_envs": 10,
        "accepted_transitions": 6000,
        "TRAINING_COMPLETED": True,
        "source_freeze_match": True,
        "vector_contract_pass": True,
        "ACTOR_CRITIC_ALPHA": "FINITE",
        "wandb": {"id": "fake", "url": "https://wandb.invalid/fake"},
        "SAC_REPLAY_SOURCE": "CURRENT_RUNTIME_ROLLOUT_ONLY",
        "OLD_TEACHER_DATA_USED_IN_SAC": "NO",
        "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
        "coordinator_metrics": {"REPLAY_STRATEGY": "HER_FORCE", "BC_WEIGHTS_CHANGED": False},
        "reward_contract": reward,
        "source_initial_states": {
            "schema": "source-schema",
            "selection_rule": "same-frozen-selection",
            "sample_ids": ["row-001", "row-002"],
            "gripper_open_all": True,
            "distinct_source_sample_ids": True,
        },
        # These receipts deliberately contain no same-source parity field.
        "reset_open_restore": {
            "open_restore_pass": True,
            "reset_parity_pass": True,
            "teacher_receipt_parity_pass": True,
            "previous_episode_geometry_used": 0,
        },
        "GRASP_EVALUATION": {
            "EVALUATION_DENOMINATOR": "COMPLETED_EPISODES_ONLY",
            "COMPLETED_EPISODE_COUNT": 25,
            "CONTACT_RATE": contact,
            "BILATERAL_CONTACT_CANDIDATE_RATE": bilateral_count / 25,
            "GRASP_SUCCESS_STABLE_RATE": stable,
            "CONTACT_TO_BILATERAL": bilateral_count / contact_count,
            "BILATERAL_TO_STABLE": stable_count / bilateral_count,
            "POST_BILATERAL_CONTACT_LOSS_RATE": 0.1,
            "CLOSE_TO_CONTACT": 1.0,
        },
        "RUNTIME_HARDSTOP": 0,
        "FORBIDDEN_COLLISION": 0,
        "ACTION_BOUND_VIOLATION": 0,
        "GRIPPER_AUTHORITY_VIOLATION": 0,
    }


def _checkpoint(*, stable: float, bilateral_to_stable: float, contact: float) -> dict[str, object]:
    contact_count = 20
    bilateral_count = int(round(contact_count * 0.8))
    stable_count = int(round(bilateral_count * bilateral_to_stable))
    return {
        "checkpoint_step": 6000,
        "checkpoint_reload_pass": True,
        "checkpoint_sha256": "checkpoint-hash",
        "metric_denominator": "COMPLETED_EPISODES_ONLY",
        "cumulative": {
            "episode_count": 25,
            "contact_episode_count": contact_count,
            "bilateral_episode_count": bilateral_count,
            "stable_episode_count": stable_count,
            "contact_loss_after_bilateral_count": 1,
            "close_triggered_episode_count": 16,
            "close_to_contact_count": 16,
            "contact_rate": contact,
            "bilateral_rate": bilateral_count / 25,
            "stable_rate": stable,
            "contact_to_bilateral": bilateral_count / contact_count,
            "bilateral_to_stable": bilateral_to_stable,
            "contact_loss_after_bilateral_rate": 0.1,
            "close_to_contact": 1.0,
        },
        "safety": {
            "runtime_hardstop": 0,
            "forbidden_collision": 0,
            "action_bound_violation": 0,
            "gripper_authority_violation": 0,
        },
    }


def test_standalone_reset_evidence_and_checkpoint_metric_precedence() -> None:
    module = _module()
    report_a = _fair_report(
        module,
        variant="V3_CURRENT_HER_FORCE_FAIR_6K",
        stable=0.4,
        bilateral_to_stable=0.625,
        contact=0.8,
    )
    report_b = _fair_report(
        module,
        variant="V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
        stable=0.4,
        bilateral_to_stable=0.625,
        contact=0.8,
    )
    audit = module.build_fair_6k_comparison(
        reset_report_path=Path("/reset.json"),
        reset_report=_reset_report(),
        fair_reports={
            "V3 CURRENT": (Path("/a.json"), report_a),
            "V3.1 lateral-off": (Path("/b.json"), report_b),
        },
        checkpoints_6000={
            "V3 CURRENT": (Path("/a_checkpoint.json"), _checkpoint(stable=0.4, bilateral_to_stable=0.625, contact=0.8)),
            "V3.1 lateral-off": (Path("/b_checkpoint.json"), _checkpoint(stable=0.4, bilateral_to_stable=0.625, contact=0.8)),
        },
    )
    assert audit["STANDALONE_RESET_PARITY"]["RESET_PARITY_CONTRACT_PASS"] is True
    assert audit["STANDALONE_RESET_PARITY"]["SAME_SOURCE_EVIDENCE_NOT_READ_FROM_FAIR_REPORTS"] is True
    assert audit["FAIR_COMPARISON_VALID"] is True
    assert all(item["METRIC_SOURCE"] == "CHECKPOINT_6000" for item in audit["FAIR_METHODS"])
    assert all(item["CHECKPOINT_VECTOR_METRICS_CONSISTENT"] is True for item in audit["FAIR_METHODS"])
    # Same Stable/B->Stable tie is broken by Contact; both checkpoint contact
    # values are equal here, so final close-to-contact is also equal and the
    # deterministic lexical ordering is not used as an implicit metric.
    assert audit["SELECTED_METHOD"] in {"V3 CURRENT", "V3.1 lateral-off"}


def test_stable_then_bilateral_to_stable_then_contact_tie_break() -> None:
    module = _module()
    report_a = _fair_report(
        module,
        variant="V3_CURRENT_HER_FORCE_FAIR_6K",
        stable=0.4,
        bilateral_to_stable=0.5,
        contact=0.95,
    )
    report_b = _fair_report(
        module,
        variant="V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
        stable=0.4,
        bilateral_to_stable=0.6,
        contact=0.8,
    )
    audit = module.build_fair_6k_comparison(
        reset_report_path=Path("/reset.json"),
        reset_report=_reset_report(),
        fair_reports={
            "V3 CURRENT": (Path("/a.json"), report_a),
            "V3.1 lateral-off": (Path("/b.json"), report_b),
        },
    )
    assert audit["FAIR_COMPARISON_VALID"] is True
    assert audit["SELECTED_METHOD"] == "V3.1 lateral-off"
    assert audit["SELECTION_TIE_BREAK_ORDER"][:3] == [
        "GRASP_SUCCESS_STABLE_RATE_DESC",
        "BILATERAL_TO_STABLE_DESC",
        "CONTACT_RATE_DESC",
    ]


def test_reset_failure_blocks_selection_without_using_fair_reset_booleans() -> None:
    module = _module()
    failed_reset = _reset_report() | {"PREMATURE_CONTACT_COUNT": 1, "RESET_PARITY_PASS": False}
    report_a = _fair_report(
        module,
        variant="V3_CURRENT_HER_FORCE_FAIR_6K",
        stable=0.4,
        bilateral_to_stable=0.6,
        contact=0.8,
    )
    report_b = _fair_report(
        module,
        variant="V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
        stable=0.3,
        bilateral_to_stable=0.5,
        contact=0.8,
    )
    audit = module.build_fair_6k_comparison(
        reset_report_path=Path("/reset.json"),
        reset_report=failed_reset,
        fair_reports={
            "V3 CURRENT": (Path("/a.json"), report_a),
            "V3.1 lateral-off": (Path("/b.json"), report_b),
        },
    )
    assert audit["FAIR_COMPARISON_VALID"] is False
    assert audit["SELECTED_METHOD"] == "PENDING_INVALID_OR_INCOMPLETE_FAIR_COMPARISON"
