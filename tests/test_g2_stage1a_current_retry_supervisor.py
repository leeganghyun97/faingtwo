# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_v3_retry_supervisor_preserves_fresh_child_and_zero_transition_hangs() -> None:
    source = (
        ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"
    ).read_text(encoding="utf-8")
    assert '"--max-startup-attempts"' in source
    assert "choices=(1, 2, 3)" in source
    assert "APPLAUNCHER_CONSTRUCTOR_TIMEOUT" in source
    assert "STARTUP_EXIT_BEFORE_CONSTRUCTOR_RETURN" in source
    assert "retryable_startup_failures" in source
    assert '"transition_zero_invalid": True' in source
    assert "close_fds=True" in source
    assert "start_new_session=True" in source
    assert "_capture_hang_snapshot" in source
    assert "POST_APP_LAUNCH" in source
    assert "POST_APP_LAUNCH_STAGES" in source
    assert "PRE_VECTOR_RUNTIME" in source
    # Bootstrap receipts deliberately live adjacent to an attempt directory,
    # so the supervisor must watch the runner's durable marker rather than a
    # non-existent TRAINING_REPORT.json.launch.json path inside it.
    assert 'marker = attempt_root.parent / f"{attempt_root.name}_launch.json"' in source
    assert 'smoke_report = attempt_root.parent / f"{attempt_root.name}_physics_smoke.json"' in source


def test_current_v3_retry_supervisor_requires_pretraining_smoke_before_training_pass() -> None:
    source = (
        ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"
    ).read_text(encoding="utf-8")
    assert '(receipt.get("physics_smoke") or {}).get("PHYSICS_SMOKE") != "PASS"' in source
    assert '"failure_class", "PHYSICS_SMOKE_FAIL"' in source
    assert "V3_BASELINE" in source
    assert "HER_FORCE" in source
    assert '"--wandb-mode",\n        "online"' in source
    assert '"privileged_used": False' in source
    assert '"auto_15k_started": False' in source
    assert "known_shutdown_sigsegv_separate" in source
    assert "completed_report" in source


def test_supervisor_validates_and_pins_policy_checkpoints_before_isaac() -> None:
    source = (
        ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "REQUIRED_POLICY_CHECKPOINT_IDS" in source
    assert "_resolve_required_policy_checkpoints" in source
    assert "POLICY_CHECKPOINT_MISSING" in source
    assert "POLICY_CHECKPOINT_HASH_MISMATCH" in source
    assert "environment.update(args.policy_checkpoint_environment)" in source
    assert '"policy_checkpoint_authority": args.policy_checkpoint_authority' in source
