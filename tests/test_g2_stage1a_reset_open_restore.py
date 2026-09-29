# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
    G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS,
)
from geniesim.rl.sac.stage1a_reset_open_restore import ResetOpenRestoreProgress


def _observe(progress: ResetOpenRestoreProgress, **overrides):
    values = {
        "master_q_rad": 0.7853981633974483,
        "master_qd_rad_s": 0.0,
        "passive_q_rad_by_name": {"passive": 0.0},
        "passive_qd_rad_s_by_name": {"passive": 0.0},
        "aperture_mm": 72.0,
        "geometry_valid": True,
        "owner_valid": True,
        "geometry_frame_id": 10,
        "geometry_timestamp_s": 1.0,
        "geometry_age_ms": 20.0,
        "geometry_cache_fresh": True,
    }
    values.update(overrides)
    return progress.observe(**values)


def test_open_restore_cannot_activate_before_sixty_open_packets() -> None:
    progress = ResetOpenRestoreProgress(
        env_id=0, episode_id=1, source_sample_id="row-000153", reset_vector_step=10
    )
    for _ in range(59):
        receipt = _observe(progress)
        assert receipt["open_restore_pass"] is False
        assert receipt["open_restore_expired"] is False

    receipt = _observe(progress)
    assert receipt["open_command_steps"] == 60
    assert receipt["open_restore_pass"] is True
    assert receipt["consecutive_settled_samples"] >= 5


def test_open_restore_requires_raw_fresh_geometry_even_when_dynamics_settle() -> None:
    progress = ResetOpenRestoreProgress(
        env_id=0, episode_id=1, source_sample_id="row-000169", reset_vector_step=10
    )
    for _ in range(60):
        receipt = _observe(progress, geometry_cache_fresh=False)
    assert receipt["dynamics_ready"] is True
    assert receipt["fresh_geometry_receipt"] is False
    assert receipt["open_restore_pass"] is False


def test_open_restore_fails_closed_after_maximum_packets_without_passive_settle() -> None:
    progress = ResetOpenRestoreProgress(
        env_id=0, episode_id=1, source_sample_id="row-000149", reset_vector_step=10
    )
    for _ in range(G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS):
        receipt = _observe(
            progress,
            passive_qd_rad_s_by_name={"passive": 0.051},
        )
    assert receipt["open_restore_pass"] is False
    assert receipt["open_restore_expired"] is True
    assert receipt["failure_reason"] == "PASSIVE_OR_MASTER_VELOCITY_NOT_SETTLED"


def test_open_restore_requires_positive_measured_aperture() -> None:
    progress = ResetOpenRestoreProgress(
        env_id=0, episode_id=1, source_sample_id="row-000157", reset_vector_step=10
    )
    for _ in range(60):
        receipt = _observe(progress, aperture_mm=-7.4)
    assert receipt["master_open_ok"] is True
    assert receipt["aperture_open_ok"] is False
    assert receipt["open_restore_pass"] is False
