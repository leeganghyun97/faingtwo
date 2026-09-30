# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Static contracts for the 25-env reset-fixed fair method comparison."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py"
RUNNER = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py"


VARIANTS = (
    "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV",
)

VARIANTS_6K = (
    "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV",
    "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV",
    "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV",
    "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV",
    "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV",
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV",
    "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV",
)


def test_all_existing_canonical_methods_have_distinct_reset_fixed_identities() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (RUNTIME, RUNNER, SUPERVISOR)
    )
    for variant in VARIANTS:
        assert variant in sources
    assert "RESET_FIXED_25ENV_7P5K_RUNTIME_VARIANTS" in sources
    assert "RESET_FIXED_FAIR_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB" in sources


def test_all_methods_have_distinct_reset_fixed_25env_6k_identities() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (RUNTIME, RUNNER, SUPERVISOR)
    )
    for variant in VARIANTS_6K:
        assert variant in sources
    assert "RESET_FIXED_25ENV_6K_RUNTIME_VARIANTS" in sources
    assert (
        "RESET_FIXED_FAIR_6K_REQUIRES_NUM_ENVS_10_OR_25_TARGET_6000_AND_ONLINE_WANDB"
        in sources
    )


def test_common_reset_contract_is_not_advisory_specific() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    assert "if _is_reset_fixed_25env_7p5k_variant(runtime_variant)" in runtime
    assert "measured_open_restore_min_ee_cube_center_distance_m" in runtime
    assert '"stage1a_reset_fixed_25env_7p5k"' in runtime
    assert '"accepted_transitions_per_env": (' in runtime
    assert "accepted_transition_target // num_envs" in runtime
    assert "hold_episode_clocks_for_open_restore" in runtime
    assert "episode_clock_hold_count" in runtime
    assert '"right_censored_pending_count"' in runtime
    assert "reset_failed_attempts" in runtime
    assert "reset_pending_attempts" in runtime
    assert "DeferredWandbRun" in runtime
    assert "FIRST_ACCEPTED_TRANSITION_AFTER_MEASURED_OPEN_GATE" in runtime
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'ROOT / "source/geniesim/rl/sac/stage1a_deferred_wandb.py"' in runner


def test_comparison_fails_closed_before_starting_later_methods() -> None:
    orchestrator = (
        ROOT / "scripts/run_g2_stage1a_reset_fixed_7p5k_comparison.py"
    ).read_text(encoding="utf-8")
    assert 'if entry["training"] != "PASS":' in orchestrator
    assert 'persist("METHOD_FAIL_CLOSED")' in orchestrator
    assert 'static_results["fail_closed_method"] = method.label' in orchestrator


def test_method_behavior_aliases_use_existing_canonical_predicates() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    assert "def _is_v31_variant" in runtime
    assert "def _is_v32_variant" in runtime
    assert "def _uses_privileged_geometry_teacher" in runtime
    assert "V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT" in runtime
    # Old distillation is not admitted into the reset-fixed comparison set.
    reset_set = runtime.split("RESET_FIXED_25ENV_7P5K_RUNTIME_VARIANTS", 1)[1].split(")", 1)[0]
    assert "PRIVILEGED_DISTILLATION_RUNTIME_VARIANT" not in reset_set


def test_current_gru_privileged_is_teacher_only_and_not_old_distillation() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    orchestrator = (
        ROOT / "scripts/run_g2_stage1a_reset_fixed_7p5k_comparison.py"
    ).read_text(encoding="utf-8")
    assert "def _uses_current_gru_privileged" in runtime
    assert '"CURRENT_GRU_PRIVILEGED_USED_IN_SAC_REPLAY": "NO"' in runtime
    assert '"student_privileged_input_count": 0' in runtime
    assert "CURRENT GRU + Privileged" in orchestrator
    assert "STATIC_FAIL_CLOSED" not in orchestrator
