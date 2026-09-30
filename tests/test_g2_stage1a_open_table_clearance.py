# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from pathlib import Path

import pytest

from geniesim.rl.sac.stage1a_open_table_clearance import (
    OPEN_TABLE_MINIMUM_CLEARANCE_M,
    OpenTableClearanceError,
    project_open_precontact_action_for_table_clearance,
    reset_open_clearance_decision,
)


def test_reset_clearance_generates_only_bounded_positive_z() -> None:
    decision = reset_open_clearance_decision(
        source_ee_z_m=0.781,
        current_ee_z_m=0.781,
        outer_link4_table_clearance_m=0.0,
        inner_link4_table_clearance_m=0.004,
    )
    assert decision.command_z_m == pytest.approx(0.0005)
    assert decision.derived_safe_ee_z_m == pytest.approx(0.783)
    assert decision.recovery_active is True
    assert decision.clearance_ready is False
    assert decision.receipt()["direct_joint_or_torque_write"] is False


def test_reset_clearance_passes_without_motion_at_two_mm() -> None:
    decision = reset_open_clearance_decision(
        source_ee_z_m=0.781,
        current_ee_z_m=0.783,
        outer_link4_table_clearance_m=OPEN_TABLE_MINIMUM_CLEARANCE_M,
        inner_link4_table_clearance_m=0.005,
    )
    assert decision.command_z_m == 0.0
    assert decision.clearance_ready is True
    assert decision.recovery_active is False


def test_reset_clearance_fails_closed_when_twenty_mm_cannot_recover() -> None:
    decision = reset_open_clearance_decision(
        source_ee_z_m=0.781,
        current_ee_z_m=0.801,
        outer_link4_table_clearance_m=-0.001,
        inner_link4_table_clearance_m=0.004,
    )
    assert decision.recovery_exhausted is True
    assert decision.command_z_m == 0.0


def test_active_projection_preserves_xy_and_only_reduces_descent() -> None:
    action, receipt = project_open_precontact_action_for_table_clearance(
        (0.003, -0.001, -0.002, 0.0),
        current_minimum_clearance_m=0.003,
    )
    assert action == pytest.approx((0.003, -0.001, -0.001, 0.0))
    assert receipt["mode"] == "CLAMP_DOWNWARD_Z"
    assert receipt["applied_minimum_clearance_m"] == pytest.approx(0.002)


def test_active_projection_recovers_drift_without_xy_motion() -> None:
    action, receipt = project_open_precontact_action_for_table_clearance(
        (0.003, -0.001, -0.002, 0.0),
        current_minimum_clearance_m=0.0017,
    )
    assert action == pytest.approx((0.0, 0.0, 0.0003, 0.0))
    assert receipt["mode"] == "RECOVER_CURRENT_CLEARANCE"


def test_nonfinite_clearance_fails_closed() -> None:
    with pytest.raises(OpenTableClearanceError, match="NONFINITE"):
        project_open_precontact_action_for_table_clearance(
            (0.0, 0.0, 0.0, 0.0), current_minimum_clearance_m=float("nan")
        )


def test_vector_runtime_wires_clearance_into_reset_and_replay_receipts() -> None:
    runtime = (
        Path(__file__).resolve().parents[1]
        / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py"
    ).read_text(encoding="utf-8")
    assert "def _uses_reset_open_table_clearance" in runtime
    assert "measure_all_open_table_clearances" in runtime
    assert "table_clearance_ready=" in runtime
    assert "_open_table_projected_execution_proposal" in runtime
    assert '"open_table_clearance_intervention"' in runtime
    assert '"forbidden_collision_relaxed": False' in (
        Path(__file__).resolve().parents[1]
        / "source/geniesim/rl/sac/stage1a_open_table_clearance.py"
    ).read_text(encoding="utf-8")
