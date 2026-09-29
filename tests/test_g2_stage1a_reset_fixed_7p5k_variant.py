# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Static contract coverage for the reset-fixed 25-env 7.5K advisory run.

No Isaac import is needed: these assertions protect the bounded identity and
the fact that it remains an advisory-only FSM route before a live run is
authorized.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py"
INITIAL_STATES = ROOT / "source/geniesim/rl/sac/stage1a_vector_initial_states.py"
RUNNER = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"

VARIANT = "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV"


def test_reset_fixed_7p5k_variant_is_distinct_and_bounded() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")

    for source in (runtime, runner, supervisor):
        assert VARIANT in source
    assert "25 * 300 = 7,500" in runtime
    assert "accepted_transition_target not in (100, 3000, 6000, 7500, 15000, 30000)" in runtime
    assert "choices=(100, 3000, 6000, 7500, 15000, 30000)" in runner
    assert "choices=(3000, 6000, 7500, 15000, 30000)" in supervisor
    assert "RESET_FIXED_7P5K_ADVISORY_REQUIRES_NUM_ENVS_25_AND_TARGET_7500" in supervisor
    assert "V31_LATERAL_OFF_FSM_ADVISORY_RESET_FIXED_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB" in runtime
    assert "V31_LATERAL_OFF_FSM_ADVISORY_RESET_FIXED_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB" in runner
    # Reset-fixed source rows must be compatible with controller-owned OPEN
    # restoration from a measured closed four-bar, without changing a runtime
    # CLOSE/safety threshold or injecting privileged data.
    assert "MEASURED_OPEN_RESTORE_MIN_EE_CUBE_CENTER_DISTANCE_M" in runtime
    assert "MEASURED_OPEN_RESTORE_MIN_EE_CUBE_CENTER_DISTANCE_M = 0.050" in INITIAL_STATES.read_text(encoding="utf-8")


def test_reset_fixed_7p5k_preserves_fsm_advisory_and_final_checkpoint() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")

    # The same identity participates in every existing sidecar-only category:
    # V3.1 lateral-off policy, capped post-CLOSE controller, frozen advisory,
    # exact sequence receipt, and episode-level diagnostics.  Nothing here
    # introduces a student hard gate or privileged runtime owner.
    assert runtime.count("V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,") >= 5
    assert "FROZEN_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY" in runtime
    assert '"STUDENT_HARD_GATE": "NO"' in runtime
    assert "periodic.add(int(accepted_transition_target))" in runtime
    assert "allow_bounded_final_boundary=bool(" in runtime
