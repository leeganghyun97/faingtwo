# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Synthetic/offline qualification for the Stage-1A grasp reward."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from geniesim.rl.sac.stage1a_grasp_reward import (
    Stage1AGraspReward,
    Stage1AGraspRewardConfig,
    Stage1ARewardV3Config,
    Stage1ARewardV32Config,
    Stage1AGraspRewardError,
    Stage1ARewardInputs,
    stage1a_reward_contract,
    summarize_reward_components,
)


def test_reward_v3_contract_is_opt_in_and_not_production_locked() -> None:
    legacy = stage1a_reward_contract()
    assert "reward_v3" not in legacy
    contract = stage1a_reward_contract(reward_v3=True)["reward_v3"]
    assert contract["temporal_window_steps"] == 15
    assert contract["cosine_window_steps"] == 5
    assert contract["hover_grace_steps"] == 15
    assert contract["authority"] == "BOUNDED_SMOKE_CANDIDATE_NOT_PRODUCTION_LOCKED"
    assert contract["ordinary_her"] is False
    assert contract["her_force"] is False


def test_reward_v32_is_opt_in_and_preserves_v3_base_contract() -> None:
    contract = stage1a_reward_contract(reward_v3=True, reward_v32=True)
    assert contract["reward_v3"]["stable_base_reward"] == pytest.approx(2.0)
    assert contract["reward_v32"]["phase_aware_hover"] is True
    assert contract["reward_v32"]["force_magnitude_positive_reward"] is False
    with pytest.raises(Stage1AGraspRewardError):
        stage1a_reward_contract(reward_v32=True)


def test_reward_v32_caps_bilateral_shaping_and_strengthens_single_dwell() -> None:
    reward = Stage1AGraspReward(
        1,
        v3_config=Stage1ARewardV3Config(),
        v32_config=Stage1ARewardV32Config(),
    )
    # The first ten bilateral rows include the immutable stability transition;
    # the bounded V3.2 shaping itself can never exceed its per-episode budget.
    bilateral = [
        reward.step(_sample(left_force_n=1.2, right_force_n=1.2))
        for _ in range(40)
    ]
    shaping_sum = sum(
        float(step.reward_bilateral_stability_shaping.item()) for step in bilateral
    )
    assert shaping_sum <= 0.020 + 1.0e-7
    assert all(
        step.penalty_hover.item() == pytest.approx(0.0) for step in bilateral
    )
    reward.reset()
    singles = [reward.step(_sample(left_force_n=1.2)) for _ in range(11)]
    assert singles[-1].penalty_single_contact_dwell.item() == pytest.approx(-0.006)


def test_reward_v3_static_hover_uses_window_and_grace_then_continuous_penalty() -> None:
    reward = Stage1AGraspReward(1, v3_config=Stage1ARewardV3Config())
    steps = [reward.step(_sample()) for _ in range(40)]
    valid = [step for step in steps if step.temporal_window_valid.item()]
    assert valid
    assert valid[0].temporal_static_hover.item()
    assert all(step.penalty_hover.item() == pytest.approx(0.0) for step in valid[:15])
    assert valid[15].penalty_hover.item() == pytest.approx(-0.0010)
    assert valid[16].penalty_hover.item() == pytest.approx(-0.0010)
    assert all(step.reward_progress.item() == pytest.approx(0.0) for step in steps)


def test_reward_v3_progress_and_cosine_require_actual_motion() -> None:
    reward = Stage1AGraspReward(1, v3_config=Stage1ARewardV3Config())
    distance = 0.0215
    last = None
    for _ in range(24):
        next_distance = distance - 0.00003
        last = reward.step(
            _sample(
                current_distance_m=distance,
                next_distance_m=next_distance,
                final_command_xyz_m=(0.001, 0.0, 0.0),
            )
        )
        distance = next_distance
    assert last is not None
    assert last.temporal_progressing.item()
    assert last.cosine_sustained.item()
    assert last.cosine_with_progress.item()
    assert last.reward_progress.item() > 0.0
    assert last.reward_cosine.item() > 0.0


def test_reward_v3_contact_latch_disables_approach_terms() -> None:
    reward = Stage1AGraspReward(1, v3_config=Stage1ARewardV3Config())
    for _ in range(35):
        reward.step(_sample())
    contact = reward.step(_sample(left_force_n=1.2))
    assert contact.reward_single_contact.item() == pytest.approx(0.25)
    after = reward.step(
        _sample(current_distance_m=0.020, next_distance_m=0.019)
    )
    assert not after.reward_v3_active.item()
    assert after.reward_progress.item() == pytest.approx(0.0)
    assert after.penalty_hover.item() == pytest.approx(0.0)
    assert after.penalty_oscillation.item() == pytest.approx(0.0)


def _sample(
    *,
    left_force_n: float = 0.0,
    right_force_n: float = 0.0,
    slip_m_s: float = 0.0,
    impact_m_s: float = 0.0,
    current_distance_m: float = 0.020,
    next_distance_m: float | None = None,
    lateral_m: float = 0.0,
    next_lateral_m: float | None = None,
    contract_valid: bool = False,
    stage1a_active: bool = True,
    grasp_decision_phase: bool = True,
    phase_reset: bool = False,
    close_command: bool = False,
    effective_residual_m: float = 0.0,
    previous_effective_residual_m: float = 0.0,
    safety_violation: bool = False,
    safety_penalty: float = 0.0,
    geometry_valid: bool = True,
    final_command_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Stage1ARewardInputs:
    next_distance = current_distance_m if next_distance_m is None else next_distance_m
    next_lateral = lateral_m if next_lateral_m is None else next_lateral_m
    # Approach axis +X.  Error parallel to +X with an optional +Y lateral part.
    current = torch.tensor([[-current_distance_m, -lateral_m, 0.0]])
    nxt = torch.tensor([[-next_distance, -next_lateral, 0.0]])
    nominal = torch.zeros(1, 3)
    valid = torch.zeros(1, 2, 10, dtype=torch.bool)
    impulse = torch.zeros(1, 2, 10)
    for side, force in enumerate((left_force_n, right_force_n)):
        if force > 0.0:
            valid[0, side, :3] = True
            # Three impulses aggregate to the requested 50-Hz effective force.
            impulse[0, side, :3] = force * 0.020 / 3.0
    geometry = valid.clone() if geometry_valid else torch.zeros_like(valid)
    points = torch.zeros(1, 2, 10, 3)
    normals = torch.zeros_like(points)
    points[0, 0, :, 0] = -0.02
    points[0, 1, :, 0] = 0.02
    normals[0, 0, :, 0] = 1.0
    normals[0, 1, :, 0] = -1.0
    slip = torch.zeros(1, 2, 10)
    impact = torch.zeros(1, 2, 10)
    slip[valid] = slip_m_s
    impact[valid] = impact_m_s
    return Stage1ARewardInputs(
        current_ee_position_root_m=current,
        next_ee_position_root_m=nxt,
        nominal_grasp_position_root_m=nominal,
        nominal_approach_axis_root=torch.tensor([[1.0, 0.0, 0.0]]),
        stage1a_active=torch.tensor([stage1a_active]),
        grasp_decision_phase=torch.tensor([grasp_decision_phase]),
        phase_reset=torch.tensor([phase_reset]),
        episode_reset=torch.tensor([False]),
        pad_cube_contact_valid=valid,
        normal_impulse_ns=impulse,
        normal_impulse_available=torch.ones(1, 2, dtype=torch.bool),
        normal_force_n=torch.zeros(1, 2, 10),
        contact_geometry_valid=geometry,
        contact_point_root_m=points,
        contact_normal_root=normals,
        tangential_relative_speed_m_s=slip,
        normal_relative_speed_m_s=impact,
        cube_center_root_m=torch.zeros(1, 3),
        cube_quat_root_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        cube_half_extents_m=torch.tensor([[0.02, 0.02, 0.02]]),
        cube_linear_velocity_root_m_s=torch.zeros(1, 3),
        cube_angular_velocity_root_rad_s=torch.zeros(1, 3),
        effective_residual_root_m=torch.tensor([[effective_residual_m, 0.0, 0.0]]),
        previous_effective_residual_root_m=torch.tensor(
            [[previous_effective_residual_m, 0.0, 0.0]]
        ),
        final_action_xyz_root_m=torch.tensor([final_command_xyz_m]),
        contract_grasp_state_valid=torch.tensor([contract_valid]),
        safety_violation=torch.tensor([safety_violation]),
        authoritative_safety_penalty=torch.tensor([safety_penalty]),
        non_pad_gripper_cube_contact=torch.tensor([False]),
        close_command=torch.tensor([close_command]),
    )


def test_contract_freezes_timing_force_role_and_actor_boundary() -> None:
    contract = stage1a_reward_contract()
    assert (contract["control_hz"], contract["physics_hz"], contract["rgbd_hz"]) == (
        50,
        500,
        25,
    )
    assert contract["physics_substeps_per_control"] == 10
    assert contract["contact_on_force_n"] == pytest.approx(1.0)
    assert contract["contact_off_force_n"] == pytest.approx(0.5)
    assert contract["force_role"] == "BOOLEAN_AND_QUALITY_ONLY"
    assert contract["force_magnitude_positive_reward"] is False
    assert contract["progress_epsilon_m"] == pytest.approx(0.00025)
    assert contract["progress_epsilon_authority"].startswith("HISTORICAL_CANDIDATE")
    assert contract["progress_epsilon_production_locked"] is False
    assert contract["progress_epsilon_adjustment_authority"] == (
        "FUTURE_MEASURED_SAC_TELEMETRY_REVIEW_ONLY"
    )
    assert contract["student_privileged_input_count"] == 0
    assert contract["geometric_her_enabled"] is False
    assert contract["her_force"]["role"] == "boolean_milestone_sampling_priority_only"


def test_hover_and_close_spam_are_negative_and_reward_equivalent() -> None:
    quiet_reward = Stage1AGraspReward(1)
    spam_reward = Stage1AGraspReward(1)
    quiet_reward.step(_sample(close_command=False))
    spam_reward.step(_sample(close_command=True))
    quiet = quiet_reward.step(_sample(close_command=False))
    spam = spam_reward.step(_sample(close_command=True))
    assert quiet.reward_total.item() < 0.0
    assert spam.reward_total.item() == pytest.approx(quiet.reward_total.item())
    assert quiet.no_progress_hovering.item()


@pytest.mark.parametrize("side", ("left", "right"))
def test_single_contact_is_small_one_shot_then_dwell_penalty(side: str) -> None:
    reward = Stage1AGraspReward(1)
    kwargs = {f"{side}_force_n": 1.2}
    first = reward.step(_sample(**kwargs))
    assert first.reward_single_contact.item() == pytest.approx(0.10)
    assert first.bilateral_boolean.item() is False
    repeated = [reward.step(_sample(**kwargs)) for _ in range(10)]
    assert all(step.reward_single_contact.item() == pytest.approx(0.0) for step in repeated)
    assert all(
        step.penalty_single_contact_dwell.item() == pytest.approx(0.0)
        for step in repeated[:9]
    )
    assert repeated[9].single_contact_counter.item() == 11
    assert repeated[9].penalty_single_contact_dwell.item() == pytest.approx(-0.01)


def test_bilateral_contact_and_force_magnitude_cannot_farm_reward() -> None:
    low = Stage1AGraspReward(1).step(_sample(left_force_n=1.2, right_force_n=1.1))
    high = Stage1AGraspReward(1).step(_sample(left_force_n=5.0, right_force_n=5.0))
    higher = Stage1AGraspReward(1).step(
        _sample(left_force_n=10.0, right_force_n=10.0)
    )
    assert low.bilateral_boolean.item()
    assert high.bilateral_boolean.item()
    # Force balance differs slightly at 1.2/1.1; equal high force has no direct term.
    equal_low = Stage1AGraspReward(1).step(_sample(left_force_n=1.2, right_force_n=1.2))
    assert high.reward_bilateral.item() == pytest.approx(equal_low.reward_bilateral.item())
    assert higher.reward_bilateral.item() == pytest.approx(equal_low.reward_bilateral.item())
    assert high.q_force_balance.item() == pytest.approx(1.0)
    assert high.reward_bilateral.item() <= 1.2 + 1.0e-6


def test_quality_uses_actual_geometry_and_penalizes_slip() -> None:
    good = Stage1AGraspReward(1).step(
        _sample(left_force_n=1.2, right_force_n=1.2, slip_m_s=0.001)
    )
    bad = Stage1AGraspReward(1).step(
        _sample(left_force_n=1.2, right_force_n=1.2, slip_m_s=0.08)
    )
    unavailable = Stage1AGraspReward(1).step(
        _sample(left_force_n=1.2, right_force_n=1.2, geometry_valid=False)
    )
    assert good.q_antipodal.item() == pytest.approx(1.0)
    assert good.q_center.item() == pytest.approx(1.0)
    assert good.q_grasp.item() > bad.q_grasp.item()
    assert not unavailable.q_antipodal_available.item()
    assert not unavailable.q_center_available.item()
    assert 0.0 <= unavailable.q_grasp.item() <= 1.0


def test_initial_contact_impact_is_preserved_for_later_stable_quality() -> None:
    reward = Stage1AGraspReward(1)
    first = reward.step(
        _sample(
            left_force_n=1.2,
            right_force_n=1.2,
            impact_m_s=0.10,
        )
    )
    later = reward.step(_sample(left_force_n=1.2, right_force_n=1.2, impact_m_s=0.0))
    assert first.q_impact.item() == pytest.approx(torch.exp(torch.tensor(-4.0)).item())
    assert later.q_impact.item() == pytest.approx(first.q_impact.item())


def test_contact_latch_disables_progress_even_when_pushing_deeper() -> None:
    reward = Stage1AGraspReward(1)
    contact = reward.step(
        _sample(
            left_force_n=1.2,
            current_distance_m=0.020,
            next_distance_m=0.019,
        )
    )
    pushed = reward.step(
        _sample(current_distance_m=0.019, next_distance_m=0.016)
    )
    assert contact.reward_progress.item() == pytest.approx(0.0)
    assert pushed.reward_progress.item() == pytest.approx(0.0)
    assert pushed.contact_latched.item()


def test_contact_loss_terminates_on_tenth_consecutive_step_after_first_contact() -> None:
    reward = Stage1AGraspReward(1)
    # Pre-contact absence never counts as contact loss.
    for _ in range(12):
        precontact = reward.step(_sample())
        assert precontact.contact_loss_counter.item() == 0
        assert not precontact.contact_loss_fail.item()
    reward.step(_sample(left_force_n=1.2))
    lost = [reward.step(_sample()) for _ in range(10)]
    assert [step.contact_loss_counter.item() for step in lost] == list(range(1, 11))
    assert not any(step.contact_loss_fail.item() for step in lost[:9])
    assert not any(step.done.item() for step in lost[:9])
    assert lost[9].contact_loss_fail.item()
    assert lost[9].done.item()


def test_contact_loss_counter_resets_when_either_pad_recontacts() -> None:
    reward = Stage1AGraspReward(1)
    reward.step(_sample(left_force_n=1.2))
    for _ in range(9):
        lost = reward.step(_sample())
    assert lost.contact_loss_counter.item() == 9
    recovered = reward.step(_sample(right_force_n=1.2))
    assert recovered.contact_loss_counter.item() == 0
    assert not recovered.contact_loss_fail.item()
    assert not recovered.done.item()


def test_lateral_pbrs_uses_nominal_approach_axis_not_world_axis() -> None:
    reward = Stage1AGraspReward(1)
    base = _sample()
    # Nominal approach is +Y; reducing X error is therefore lateral progress.
    reward.step(
        replace(
            base,
            current_ee_position_root_m=torch.tensor([[-0.003, -0.020, 0.0]]),
            next_ee_position_root_m=torch.tensor([[-0.003, -0.020, 0.0]]),
            nominal_approach_axis_root=torch.tensor([[0.0, 1.0, 0.0]]),
        )
    )
    progressed = reward.step(
        replace(
            base,
            current_ee_position_root_m=torch.tensor([[-0.003, -0.020, 0.0]]),
            next_ee_position_root_m=torch.tensor([[-0.002, -0.020, 0.0]]),
            nominal_approach_axis_root=torch.tensor([[0.0, 1.0, 0.0]]),
        )
    )
    assert progressed.reward_lateral.item() > 0.0


def test_grasp_decision_phase_rejects_distance_outside_15_22_mm() -> None:
    with pytest.raises(Stage1AGraspRewardError, match="15--22 mm"):
        Stage1AGraspReward(1).step(_sample(current_distance_m=0.030))


def test_stable_and_success_are_ten_step_one_shot_events() -> None:
    reward = Stage1AGraspReward(1)
    steps = [
        reward.step(
            _sample(left_force_n=1.2, right_force_n=1.2, contract_valid=True)
        )
        for _ in range(20)
    ]
    assert not any(step.stable_grasp_boolean.item() for step in steps[:9])
    assert steps[9].bilateral_stable_counter.item() == 10
    assert steps[9].stable_grasp_boolean.item()
    assert steps[9].reward_stable.item() > 1.5
    assert steps[9].reward_success.item() == pytest.approx(3.0)
    assert steps[9].done.item()
    assert all(step.reward_stable.item() == pytest.approx(0.0) for step in steps[10:])
    assert all(step.reward_success.item() == pytest.approx(0.0) for step in steps[10:])


def test_contact_loss_recontact_cannot_reissue_milestones() -> None:
    reward = Stage1AGraspReward(1)
    first = reward.step(_sample(left_force_n=1.2, right_force_n=1.2))
    reward.step(_sample())
    again = reward.step(_sample(left_force_n=1.2, right_force_n=1.2))
    assert first.reward_bilateral.item() > 0.8
    assert again.reward_bilateral.item() == pytest.approx(0.0)


def test_contact_hysteresis_uses_off_threshold_and_three_substeps() -> None:
    reward = Stage1AGraspReward(1)
    assert reward.step(_sample(left_force_n=1.2)).left_contact_boolean.item()
    assert reward.step(_sample(left_force_n=0.6)).left_contact_boolean.item()
    assert not reward.step(_sample(left_force_n=0.4)).left_contact_boolean.item()


def test_hover_has_ten_step_grace_then_persistent_penalty() -> None:
    reward = Stage1AGraspReward(1)
    reward.step(_sample())  # phase-entry potential reset; not a dwell sample
    grace = [reward.step(_sample()) for _ in range(10)]
    assert [step.hover_counter.item() for step in grace] == list(range(1, 11))
    assert all(step.penalty_hover.item() == pytest.approx(0.0) for step in grace)
    assert all(step.reward_total.item() <= 0.0 for step in grace)
    penalized = reward.step(_sample())
    assert penalized.hover_counter.item() == 11
    assert penalized.penalty_hover.item() == pytest.approx(-0.01)
    again = reward.step(_sample())
    assert again.penalty_hover.item() == pytest.approx(-0.01)


def test_wrong_direction_has_five_step_grace_then_penalty_and_reset() -> None:
    reward = Stage1AGraspReward(1)
    reward.step(_sample())
    wrong = [
        reward.step(_sample(final_command_xyz_m=(-0.001, 0.0, 0.0)))
        for _ in range(6)
    ]
    assert [step.wrong_direction_counter.item() for step in wrong[:5]] == list(
        range(1, 6)
    )
    assert all(
        step.penalty_wrong_direction.item() == pytest.approx(0.0)
        for step in wrong[:5]
    )
    assert wrong[5].cosine_similarity.item() == pytest.approx(-1.0)
    assert wrong[5].penalty_wrong_direction.item() == pytest.approx(-0.02)
    recovered = reward.step(_sample(final_command_xyz_m=(0.001, 0.0, 0.0)))
    assert recovered.cosine_similarity.item() == pytest.approx(1.0)
    assert recovered.wrong_direction_counter.item() == 0
    assert recovered.penalty_wrong_direction.item() == pytest.approx(0.0)


def test_all_persistence_counters_reset_when_their_conditions_break() -> None:
    reward = Stage1AGraspReward(1)
    reward.step(_sample())
    hovered = reward.step(_sample())
    assert hovered.hover_counter.item() == 1
    progressed = reward.step(_sample(current_distance_m=0.020, next_distance_m=0.019))
    assert progressed.hover_counter.item() == 0

    single = reward.step(_sample(left_force_n=1.2))
    assert single.single_contact_counter.item() == 1
    no_contact = reward.step(_sample())
    assert no_contact.single_contact_counter.item() == 0

    bilateral = reward.step(_sample(left_force_n=1.2, right_force_n=1.2))
    assert bilateral.bilateral_stable_counter.item() == 1
    slipping = reward.step(
        _sample(left_force_n=1.2, right_force_n=1.2, slip_m_s=0.08)
    )
    assert slipping.bilateral_stable_counter.item() == 0

    hysteresis_only = Stage1AGraspReward(1)
    hysteresis_only.step(_sample(left_force_n=1.2, right_force_n=1.2))
    below_stable_on = hysteresis_only.step(
        _sample(left_force_n=0.6, right_force_n=0.6)
    )
    assert below_stable_on.bilateral_boolean.item()
    assert below_stable_on.bilateral_stable_counter.item() == 0


def test_oscillation_boundary_and_max_residual_have_negative_net_return() -> None:
    reward = Stage1AGraspReward(1)
    reward.step(_sample(current_distance_m=0.020, next_distance_m=0.020))
    toward = reward.step(_sample(current_distance_m=0.020, next_distance_m=0.019))
    away = reward.step(_sample(current_distance_m=0.019, next_distance_m=0.020))
    assert toward.reward_progress.item() + away.reward_progress.item() < 0.0
    boundary_reset = reward.step(
        _sample(current_distance_m=0.022, phase_reset=True)
    )
    assert boundary_reset.reward_progress.item() == pytest.approx(0.0)
    assert boundary_reset.reward_total.item() < 0.0
    max_residual = Stage1AGraspReward(1).step(
        _sample(effective_residual_m=0.00045, previous_effective_residual_m=0.00045)
    )
    assert max_residual.residual_penalty.item() == pytest.approx(-0.03)
    assert max_residual.reward_total.item() < -0.03


def test_impulse_persistence_and_force_fallback_are_interval_consistent() -> None:
    reward = Stage1AGraspReward(1)
    too_brief = _sample(left_force_n=1.2)
    brief_valid = too_brief.pad_cube_contact_valid.clone()
    brief_valid[0, 0, 1:] = False
    brief_impulse = too_brief.normal_impulse_ns.clone()
    brief_impulse.zero_()
    brief_impulse[0, 0, 0] = 1.2 * 0.020
    result = reward.step(
        replace(
            too_brief,
            pad_cube_contact_valid=brief_valid,
            contact_geometry_valid=brief_valid,
            normal_impulse_ns=brief_impulse,
        )
    )
    assert result.left_normal_force_n.item() == pytest.approx(1.2)
    assert not result.left_contact_boolean.item()

    fallback = _sample(left_force_n=0.0)
    valid = fallback.pad_cube_contact_valid.clone()
    valid[0, 0, :3] = True
    force = fallback.normal_force_n.clone()
    force[0, 0, :3] = 4.0  # 3 * 4 N * 2 ms / 20 ms == 1.2 N
    fallback_result = Stage1AGraspReward(1).step(
        replace(
            fallback,
            pad_cube_contact_valid=valid,
            contact_geometry_valid=valid,
            normal_force_n=force,
            normal_impulse_available=torch.zeros(1, 2, dtype=torch.bool),
        )
    )
    assert fallback_result.left_normal_force_n.item() == pytest.approx(1.2)
    assert fallback_result.left_contact_boolean.item()


def test_existing_safety_authority_is_required_and_terminal() -> None:
    result = Stage1AGraspReward(1).step(
        _sample(safety_violation=True, safety_penalty=-2.0)
    )
    assert result.done.item()
    assert result.safety_penalty.item() == pytest.approx(-2.0)
    with pytest.raises(Stage1AGraspRewardError, match="requires existing authority"):
        Stage1AGraspReward(1).step(_sample(safety_penalty=-1.0))
    assert not hasattr(Stage1AGraspRewardConfig(), "force_safety_limit_n")


def test_unsafe_contact_cannot_emit_positive_milestone() -> None:
    result = Stage1AGraspReward(1).step(
        _sample(
            left_force_n=1.2,
            right_force_n=1.2,
            safety_violation=True,
            safety_penalty=-1.0,
        )
    )
    assert result.reward_bilateral.item() == pytest.approx(0.0)
    assert result.reward_single_contact.item() == pytest.approx(0.0)
    assert result.reward_total.item() < 0.0


def test_scale_audit_is_observational_and_actor_access_fails_closed() -> None:
    sample = _sample()
    with pytest.raises(RuntimeError, match="PRIVILEGED_REWARD_TO_ACTOR"):
        sample.actor_payload()
    step = Stage1AGraspReward(1).step(sample)
    summary = summarize_reward_components([[step]])
    assert summary["reward_total"]["mean"] == pytest.approx(step.reward_total.item())
    assert summary["time_penalty"]["sum_per_episode_mean"] == pytest.approx(-0.002)
    required_persistence_telemetry = {
        "bilateral_stable_counter",
        "single_contact_counter",
        "hover_counter",
        "wrong_direction_counter",
        "contact_loss_counter",
        "single_contact_boolean",
        "bilateral_contact_boolean",
        "stable_grasp_boolean",
        "contact_loss_fail",
        "cosine_similarity",
        "progress_m",
        "lateral_improvement_m",
        "reward_single_contact",
        "reward_bilateral",
        "reward_stable",
        "reward_success",
        "penalty_single_contact_dwell",
        "penalty_hover",
        "penalty_wrong_direction",
    }
    assert required_persistence_telemetry <= set(step.telemetry())
    assert summary["time_penalty"]["episode_sum"] == pytest.approx([-0.002])
