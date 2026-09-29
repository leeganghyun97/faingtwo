# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Static contract coverage for reset-fixed Stage-1A fair 6K routes.

These tests intentionally do not import Isaac or launch a simulation.  They
guard the immutable identities and authority separation that the live
post-reset comparison relies on.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py"
RUNNER = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"


FAIR_VARIANTS = (
    "V3_CURRENT_HER_FORCE_FAIR_6K",
    "V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
    "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_FAIR_6K",
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K",
)


def test_fair_6k_routes_have_distinct_immutable_runtime_identities() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    for variant in FAIR_VARIANTS:
        assert variant in runtime
        assert variant in runner
    assert "def _is_fair_6k_variant" in runtime
    assert "FAIR_6K_REQUIRES_NUM_ENVS_10_TARGET_6000_AND_ONLINE_WANDB" in runtime
    assert "FAIR_6K_REQUIRES_NUM_ENVS_10_TARGET_6000_AND_ONLINE_WANDB" in runner
    assert "choices=(100, 3000, 6000, 7500, 15000, 30000)" in runner


def test_fair_6k_routes_preserve_lateral_off_fsm_and_advisory_authority() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")

    # The V3.1 routes retain their existing micro controller; the baseline
    # remains a separate current-rule control identity.
    assert "V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT," in runtime
    assert "V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT," in runtime
    assert "V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT," in runtime
    assert "POST_CLOSE_MICRO_RUNTIME_VARIANTS" in runtime
    # The deterministic FSM is declared as the actual CLOSE authority, while
    # Conditional-CORAL remains telemetry/advisory only.
    assert 'else "CURRENT_DISTANCE_ALIGNMENT_FSM"' in runtime
    assert '"FSM_RUNTIME_AUTHORITY": (' in runtime
    assert '"YES" if _uses_fsm_sequence_dataset(runtime_variant) else "NO"' in runtime
    assert "FROZEN_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY" in runtime
    assert '"STUDENT_HARD_GATE": "NO"' in runtime
    assert '"student_privileged_input_count": 0' in runtime


def test_fair_6k_fresh_process_supervisor_can_supply_only_frozen_advisory() -> None:
    supervisor = SUPERVISOR.read_text(encoding="utf-8")

    assert "choices=(3000, 6000, 7500, 15000, 30000)" in supervisor
    assert "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K" in supervisor
    assert "--frozen-student-advisory-checkpoint" in supervisor
    assert "FAIR_6K_ADVISORY_REQUIRES_EXISTING_FROZEN_STUDENT_CHECKPOINT_AND_SHA256" in supervisor
