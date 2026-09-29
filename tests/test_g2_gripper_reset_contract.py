# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch

from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
    FullArticulationFDSafetyAudit,
    G2_LEFT_HAND_RESET_POLICY,
    G2_RIGHT_GRIPPER_MASTER,
    G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES,
    G2_FOUR_BAR_ASSET_SOLVER_POSITION_ITERATIONS,
    G2_FOUR_BAR_ASSET_SOLVER_VELOCITY_ITERATIONS,
    G2_FOUR_BAR_CONFIGURED_SEED_LOOP_CLOSURE_MEAN_MM,
    G2_FOUR_BAR_PAD_DIAGNOSTIC_POSITION_ITERATIONS,
    G2_FOUR_BAR_PAD_DIAGNOSTIC_VELOCITY_ITERATIONS,
    G2_FOUR_BAR_ZERO_SEED_LOOP_CLOSURE_MEAN_MM,
    G2_PASSIVE_LIMIT_GOVERNOR_BRAKE,
    G2_PASSIVE_LIMIT_GOVERNOR_BRAKE_ACCELERATION_RAD_S2,
    G2_PASSIVE_LIMIT_GOVERNOR_CREEP,
    G2_PASSIVE_LIMIT_GOVERNOR_NUMERIC_LANDING_TOLERANCE_RAD,
    G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD,
    G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP,
    G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE,
    G2_PASSIVE_LIMIT_GOVERNOR_SETTLE,
    G2_PASSIVE_LIMIT_GOVERNOR_TRACK,
    G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER,
    G2_RIGHT_HAND_JOINT_NAMES,
    G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
    PassiveLimitMasterGovernor,
    _passive_limit_aware_master_speed_cap_per_passive,
    acceleration_limited_position_target_step,
    capture_measured_right_hand_cache,
    four_bar_initial_state_contract_metadata,
    install_measured_right_hand_cache,
    passive_limit_aware_master_speed_cap,
    shared_gripper_reset_contract_metadata,
    validate_complete_stable_hand_configuration,
)
from geniesim.rl.isaaclab.g2_lift_methodology import stable_original_home
from geniesim.rl.isaaclab.g2_rebuild.data_contract import G2_RUNTIME_JOINT_ORDER


ROOT = Path(__file__).resolve().parents[1]


def test_stable_reset_pose_is_name_complete_and_both_hands_are_stable():
    pose = stable_original_home()
    assert tuple(pose) != G2_RUNTIME_JOINT_ORDER  # dict order is not authority
    assert set(pose) == set(G2_RUNTIME_JOINT_ORDER)
    assert len(pose) == 46
    validate_complete_stable_hand_configuration(pose)


def test_partial_open_overlay_is_rejected_and_not_applied_by_keyboard_pose():
    pose = stable_original_home()
    pose["idx81_gripper_r_outer_joint1"] = 0.7853981633974483
    with pytest.raises(ValueError, match="HAND_NOT_COMPLETE_CONFIGURED_SEED"):
        validate_complete_stable_hand_configuration(pose)
    source = (
        ROOT / "source/geniesim/rl/isaaclab/g2_keyboard_pose.py"
    ).read_text()
    function = source[source.index("def apply_keyboard_collection_initial_pose") :]
    function = function[: function.index("def apply_keyboard_collection_task_geometry")]
    assert "pose.update(G2_KEYBOARD_OPEN_GRIPPER" not in function


def test_full_measured_right_hand_cache_round_trip_preserves_every_coordinate():
    names = list(G2_RUNTIME_JOINT_ORDER)
    measured = torch.arange(2 * 46, dtype=torch.float32).reshape(2, 46)
    audit = FullArticulationFDSafetyAudit.start(
        torch.zeros_like(measured), names, dt_s=0.02
    )
    for step in range(1, 61):
        audit.observe(torch.zeros_like(measured), step=step)
    attestation = audit.result()
    assert attestation["pass"] is True
    cache = capture_measured_right_hand_cache(
        measured,
        names,
        safety_attestation=attestation,
    )
    assert cache.shape == (2, 8)
    default = torch.zeros_like(measured)
    install_measured_right_hand_cache(default, cache, names)
    indices = [names.index(name) for name in G2_RIGHT_HAND_JOINT_NAMES]
    assert torch.equal(default[:, indices], cache)
    untouched = [index for index in range(46) if index not in indices]
    assert torch.count_nonzero(default[:, untouched]) == 0


def test_reset_metadata_targets_only_right_master_and_declares_left_policy():
    metadata = shared_gripper_reset_contract_metadata()
    assert metadata["independently_targeted_right_hand_joint_names"] == [
        G2_RIGHT_GRIPPER_MASTER
    ]
    assert set(metadata["passive_or_mimic_target_write_forbidden"]) == set(
        G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES
    )
    assert set(metadata["right_hand_passive_or_mimic_joint_names"]) == set(
        G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES
    )
    assert metadata["left_hand_policy"] == G2_LEFT_HAND_RESET_POLICY
    assert metadata["left_hand_open_commanded"] is False
    assert metadata["hidden_per_environment_physics_step"] is False


def test_four_bar_seed_is_source_linked_but_never_promoted_as_live_equilibrium():
    audit = four_bar_initial_state_contract_metadata()
    assert audit["usd_authored_passive_position_rad"] == 0.0
    assert (
        G2_FOUR_BAR_CONFIGURED_SEED_LOOP_CLOSURE_MEAN_MM
        < G2_FOUR_BAR_ZERO_SEED_LOOP_CLOSURE_MEAN_MM
    )
    assert audit["configured_seed_residual_reduction_fraction"] > 0.80
    assert audit["configured_seed_is_exact_closed_loop_solution"] is False
    assert audit["configured_seed_is_live_equilibrium_evidence"] is False
    assert audit["configured_seed_provenance_by_side"]["right"].startswith(
        "HISTORICAL_COMPOSED_PHYSX_INITIAL_READBACK"
    )
    assert audit["configured_seed_provenance_by_side"]["left"].startswith(
        "CODE_MIRRORED_FROM_RIGHT_PATTERN"
    )
    assert audit["historical_manifest_contains_right_hand_readback"] is True
    assert audit["historical_manifest_contains_left_hand_readback"] is False
    assert audit["left_configured_seed_is_live_equilibrium_evidence"] is False
    assert audit["historical_evidence_acceptance_eligible"] is False
    assert (
        G2_FOUR_BAR_ASSET_SOLVER_POSITION_ITERATIONS,
        G2_FOUR_BAR_ASSET_SOLVER_VELOCITY_ITERATIONS,
    ) == (32, 1)
    assert (
        G2_FOUR_BAR_PAD_DIAGNOSTIC_POSITION_ITERATIONS,
        G2_FOUR_BAR_PAD_DIAGNOSTIC_VELOCITY_ITERATIONS,
    ) == (8, 0)
    assert audit["pad_diagnostic_matches_asset_solver_contract"] is False
    assert audit["solver_change_authorized_from_offline_audit"] is False
    assert audit["live_q0_q1_q2_revalidation_required"] is True


def test_passive_limit_guard_uses_master_authority_braking_distance():
    # Exact position interval and passive speed at the fresh live hard-stop
    # failure.  The master speed is the causal minimum-jerk authority entering
    # that local interval, not a claimed measured-velocity field (the failed
    # artifact did not preserve such a field).
    upper = 0.03490658476948738
    position = 0.03466477617621422
    passive_speed = 0.12090429663658142
    master_speed = 0.7154729810413818
    dt_s = 0.002
    reaction_time_s = 2 * dt_s
    acceleration = 10.0
    margin = upper - position

    direct_passive_authority_margin = (
        passive_speed * reaction_time_s
        + passive_speed**2 / (2.0 * acceleration)
    )
    master_authority_margin = (
        passive_speed * reaction_time_s
        + 0.5 * passive_speed * master_speed / acceleration
    )
    assert margin == pytest.approx(0.00024180859327316284)
    assert direct_passive_authority_margin == pytest.approx(
        0.0012145096338056494
    )
    assert master_authority_margin == pytest.approx(
        0.004808805063310647
    )
    assert master_authority_margin > 3.9 * direct_passive_authority_margin

    cap = passive_limit_aware_master_speed_cap(
        torch.tensor([[position]], dtype=torch.float64),
        torch.tensor([[passive_speed]], dtype=torch.float64),
        torch.tensor([[-upper, upper]], dtype=torch.float64),
        torch.tensor([[master_speed]], dtype=torch.float64),
        torch.tensor([[0.002]], dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=acceleration,
        dt_s=dt_s,
    )
    # At the recorded sample even the two-step reaction distance exceeds the
    # remaining margin.  The causal fix is therefore to have started braking
    # earlier; it must not pretend this late state is recoverable.
    assert float(cap.item()) == pytest.approx(0.0)


def test_passive_limit_guard_does_not_block_direction_reversal():
    cap = passive_limit_aware_master_speed_cap(
        torch.tensor([[0.0348]], dtype=torch.float64),
        torch.tensor([[0.1]], dtype=torch.float64),
        torch.tensor(
            [[-0.03490658849477768, 0.03490658849477768]],
            dtype=torch.float64,
        ),
        torch.tensor([[0.4]], dtype=torch.float64),
        torch.tensor([[-0.001]], dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert float(cap.item()) == pytest.approx(0.8)


def test_stateful_governor_brake_cap_is_continuous_not_a_zero_step():
    dt_s = 0.002
    passive_limit = 0.03490658476948738
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    state = (
        torch.tensor([[passive_limit - 0.020]], dtype=torch.float64),
        torch.tensor([[0.1209]], dtype=torch.float64),
        torch.tensor([[-passive_limit, passive_limit]], dtype=torch.float64),
        torch.tensor([[0.715]], dtype=torch.float64),
        torch.tensor([[0.1]], dtype=torch.float64),
    )
    trigger_cap = governor.compute_speed_cap(
        *state,
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=dt_s,
    )
    next_cap = governor.compute_speed_cap(
        *state,
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=dt_s,
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE
    assert float(trigger_cap.item()) == pytest.approx(0.715)
    assert float(next_cap.item()) == pytest.approx(
        0.715
        - G2_PASSIVE_LIMIT_GOVERNOR_BRAKE_ACCELERATION_RAD_S2 * dt_s
    )
    assert (float(trigger_cap.item()) - float(next_cap.item())) / dt_s == (
        pytest.approx(G2_PASSIVE_LIMIT_GOVERNOR_BRAKE_ACCELERATION_RAD_S2)
    )


def test_stateful_governor_causal_nonlinear_lag_lands_below_hard_limit():
    """The exact -33.24 live clamp is replaced by stop/settle/creep.

    The surrogate uses only past emitted master velocities, a two-sample
    transport lag, and a transmission ratio that increases near the stop.
    Position projection is retained rather than deleting the impact sample.
    """

    dt_s = 0.002
    hard_acceleration = 10.0
    passive_limit = 0.03490658476948738
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    limits = torch.tensor(
        [[-passive_limit, passive_limit]], dtype=torch.float64
    )
    passive_position = torch.tensor([[0.00249]], dtype=torch.float64)
    initial_passive_margin = passive_limit - float(passive_position.item())
    master_target = torch.tensor([[0.0]], dtype=torch.float64)
    desired_target = torch.full_like(master_target, math.pi / 4.0)
    master_velocity = torch.zeros_like(master_target)
    passive_velocity = torch.zeros_like(master_target)
    delayed_master_speeds = [
        float(master_velocity.item()),
        float(master_velocity.item()),
    ]
    maximum_target_acceleration = 0.0
    maximum_passive_acceleration = 0.0
    maximum_passive_acceleration_context = None
    impact_speed = None
    brake_entry_margin = None
    creep_to_brake_count = 0
    phases_seen = {G2_PASSIVE_LIMIT_GOVERNOR_TRACK}

    for step_index in range(5000):
        previous_phase = int(governor.phase.item())
        cap = governor.compute_speed_cap(
            passive_position,
            passive_velocity,
            limits,
            master_velocity,
            desired_target - master_target,
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=hard_acceleration,
            dt_s=dt_s,
        )
        phases_seen.add(int(governor.phase.item()))
        if (
            previous_phase == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
            and int(governor.phase.item())
            == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE
        ):
            creep_to_brake_count += 1
        if (
            int(governor.phase.item())
            == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE
            and brake_entry_margin is None
        ):
            brake_entry_margin = passive_limit - float(
                passive_position.item()
            )
        next_target, next_master_velocity = (
            acceleration_limited_position_target_step(
                master_target,
                desired_target,
                master_velocity,
                maximum_speed_rad_s=cap,
                maximum_acceleration_rad_s2=hard_acceleration,
                dt_s=dt_s,
            )
        )
        maximum_target_acceleration = max(
            maximum_target_acceleration,
            abs(float((next_master_velocity - master_velocity).item()))
            / dt_s,
        )

        # A causal two-sample lag plus a smooth terminal transmission
        # increase near the stop.  Its own bounded response cannot fabricate a
        # one-sample acceleration violation.
        margin_fraction = max(
            0.0,
            min(
                1.0,
                float(
                    (passive_limit - passive_position.item())
                    / initial_passive_margin
                ),
            ),
        )
        # Mirror the actual authority: TRACK/BRAKE measured response is raw;
        # the evidence multiplier exists only in the predicted CREEP branch.
        # Injecting it into TRACK would test a plant that the governor never
        # observes and would not model either landing10 or landing11.
        terminal_factor = (
            1.0
            + (
                G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER
                - 1.0
            )
            * (1.0 - margin_fraction)
            if int(governor.phase.item())
            == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
            else 1.0
        )
        nonlinear_ratio = 0.168985132688873 * terminal_factor
        desired_passive_velocity = (
            nonlinear_ratio * delayed_master_speeds.pop(0)
        )
        delayed_master_speeds.append(float(next_master_velocity.item()))
        passive_velocity_delta = max(
            -hard_acceleration * dt_s,
            min(
                hard_acceleration * dt_s,
                desired_passive_velocity - float(passive_velocity.item()),
            ),
        )
        unconstrained_passive_velocity = float(
            passive_velocity.item()
        ) + passive_velocity_delta
        unconstrained_position = float(passive_position.item()) + (
            unconstrained_passive_velocity * dt_s
        )
        if unconstrained_position >= passive_limit:
            next_passive_position = passive_limit
            next_passive_velocity = (
                next_passive_position - float(passive_position.item())
            ) / dt_s
            impact_speed = unconstrained_passive_velocity
        else:
            next_passive_position = unconstrained_position
            next_passive_velocity = unconstrained_passive_velocity
        passive_acceleration = abs(
            next_passive_velocity - float(passive_velocity.item())
        ) / dt_s
        if passive_acceleration > maximum_passive_acceleration:
            maximum_passive_acceleration = passive_acceleration
            maximum_passive_acceleration_context = (
                step_index,
                int(governor.phase.item()),
                float(passive_velocity.item()),
                next_passive_velocity,
                float(passive_position.item()),
                unconstrained_position,
            )
        passive_position.fill_(next_passive_position)
        passive_velocity.fill_(next_passive_velocity)
        master_target = next_target
        master_velocity = next_master_velocity

        if (
            int(governor.phase.item())
            == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP
        ):
            break

    assert impact_speed is not None, (
        int(governor.phase.item()),
        float(passive_position.item()),
        float(master_velocity.item()),
        float(governor.governor_speed_cap.item()),
    )
    assert step_index < 4 * 921
    assert brake_entry_margin is not None
    assert brake_entry_margin > 0.020
    assert impact_speed <= hard_acceleration * dt_s + 1.0e-12
    assert maximum_target_acceleration <= hard_acceleration + 1.0e-9
    assert maximum_passive_acceleration <= hard_acceleration + 1.0e-9, (
        maximum_passive_acceleration_context
    )
    assert G2_PASSIVE_LIMIT_GOVERNOR_BRAKE in phases_seen
    assert G2_PASSIVE_LIMIT_GOVERNOR_SETTLE in phases_seen
    assert G2_PASSIVE_LIMIT_GOVERNOR_CREEP in phases_seen
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP

    # Preserve the original failed sample as evidence, not as a deleted
    # boundary point: it really was a -33.2407646-rad/s^2 clamp.
    failed_previous_velocity = 0.11782161891460419
    failed_current_velocity = 0.05134008824825287
    assert (
        failed_current_velocity - failed_previous_velocity
    ) / dt_s == pytest.approx(-33.24076533317566)


def test_governor_decision_receipt_observes_exact_existing_decision_operands():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    position = torch.tensor([[0.0]], dtype=torch.float64)
    velocity = torch.tensor([[0.10]], dtype=torch.float64)
    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    master_velocity = torch.tensor([[0.20]], dtype=torch.float64)
    desired_delta = torch.tensor([[0.50]], dtype=torch.float64)
    result = governor.compute_speed_cap(
        position,
        velocity,
        limits,
        master_velocity,
        desired_delta,
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    receipt = governor.last_decision_receipt
    assert result.shape == (1, 1)
    assert receipt["per_passive_envelope_cap_rad_s"].shape == (1, 1)
    assert receipt["transmission_ratio_used"].shape == (1, 1)
    assert receipt["passive_risk_mask"].dtype is torch.bool
    assert receipt["active_mask_at_entry"].dtype is torch.bool
    assert receipt["active_mask_after"].dtype is torch.bool
    assert receipt["active_mask_add_reason_code"].shape == (1, 1)
    assert receipt["active_mask_remove_reason_code"].shape == (1, 1)


def test_second_live_trace_requires_threefold_terminal_uncertainty():
    dt_s = 0.002
    passive_limit = 0.03490658476948738
    q_995 = 0.034822527319192886
    q_996 = 0.03490658104419708
    q_997 = passive_limit
    incoming_speed = (q_996 - q_995) / dt_s
    projected_speed = (q_997 - q_996) / dt_s
    projected_acceleration = (projected_speed - incoming_speed) / dt_s

    assert passive_limit - q_995 == pytest.approx(
        8.405745029449463e-05
    )
    assert incoming_speed == pytest.approx(0.042026862502098083)
    assert projected_speed == pytest.approx(1.862645149230957e-06)
    assert projected_acceleration == pytest.approx(-21.012499928474426)
    observed_to_creep_budget = incoming_speed / 0.015
    assert observed_to_creep_budget == pytest.approx(2.8017908334732056)
    assert (
        G2_PASSIVE_LIMIT_GOVERNOR_TRANSMISSION_UNCERTAINTY_MULTIPLIER
        >= observed_to_creep_budget
    )


def test_post_stop_latch_ignores_same_direction_noise_until_reversal():
    dt_s = 0.002
    passive_limit = 0.03490658476948738
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    limits = torch.tensor(
        [[-passive_limit, passive_limit]], dtype=torch.float64
    )
    # Seed the minimal causal state reached after safe landing.  This unit
    # isolates latch semantics; the preceding test proves the full path.
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)

    same_direction_cap = governor.compute_speed_cap(
        torch.tensor([[passive_limit]], dtype=torch.float64),
        torch.tensor([[0.003276]], dtype=torch.float64),
        limits,
        torch.tensor([[0.02]], dtype=torch.float64),
        torch.tensor([[0.1]], dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=dt_s,
    )
    assert float(same_direction_cap.item()) == pytest.approx(0.8)
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP

    reverse_cap = governor.compute_speed_cap(
        torch.tensor([[passive_limit]], dtype=torch.float64),
        torch.tensor([[0.003276]], dtype=torch.float64),
        limits,
        torch.tensor([[0.02]], dtype=torch.float64),
        torch.tensor([[-0.1]], dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=dt_s,
    )
    assert float(reverse_cap.item()) == pytest.approx(0.8)
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_TRACK


def _seven_passive_risk_fixture(*, dtype=torch.float64):
    limit = 1.0
    margins = torch.tensor(
        [[0.0004, 0.0100, 0.0006, 0.0200, 0.0007, 0.0300, 0.0003]],
        dtype=dtype,
    )
    position = limit - margins
    velocity = torch.full_like(position, 0.2)
    limits = torch.tensor([[-limit, limit]] * 7, dtype=dtype)
    master_velocity = torch.zeros((1, 1), dtype=dtype)
    desired_delta = torch.ones((1, 1), dtype=dtype)
    expected = torch.tensor(
        [[True, False, True, False, True, False, True]],
        dtype=torch.bool,
    )
    return position, velocity, limits, master_velocity, desired_delta, expected


def test_multi_passive_latch_identity_uses_same_per_cap_comparison():
    state = _seven_passive_risk_fixture()
    position, velocity, limits, master_velocity, desired_delta, expected = state
    per_cap = _passive_limit_aware_master_speed_cap_per_passive(
        position,
        velocity,
        limits,
        master_velocity,
        desired_delta,
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=2.0,
        dt_s=0.002,
        reaction_steps=2,
    )
    assert torch.equal(per_cap < 0.8, expected)

    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=7,
        device="cpu",
        dtype=torch.float64,
    )
    governor.compute_speed_cap(
        position,
        velocity,
        limits,
        master_velocity,
        desired_delta,
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert torch.equal(governor.active_passive_mask, expected)
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE


def test_multi_passive_latch_is_permutation_equivariant():
    state = _seven_passive_risk_fixture()
    position, velocity, limits, master_velocity, desired_delta, expected = state
    permutation = torch.tensor([6, 2, 0, 5, 3, 1, 4])
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=7,
        device="cpu",
        dtype=torch.float64,
    )
    governor.compute_speed_cap(
        position[:, permutation],
        velocity[:, permutation],
        limits[permutation],
        master_velocity,
        desired_delta,
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert torch.equal(
        governor.active_passive_mask,
        expected[:, permutation],
    )


def _seed_creep_governor(*, passive_count: int):
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=passive_count,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    return governor


def _stationary_governor_sample(governor, position, limits):
    return governor.compute_speed_cap(
        position,
        torch.zeros_like(position),
        limits,
        torch.zeros((position.shape[0], 1), dtype=position.dtype),
        torch.ones((position.shape[0], 1), dtype=position.dtype),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )


def test_multi_passive_simultaneous_landing_latches_every_coordinate():
    governor = _seed_creep_governor(passive_count=2)
    limits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    position = torch.ones((1, 2), dtype=torch.float64)
    _stationary_governor_sample(governor, position, limits)
    _stationary_governor_sample(governor, position, limits)
    assert torch.equal(
        governor.landed_passive_mask,
        torch.ones((1, 2), dtype=torch.bool),
    )
    assert not bool(governor.active_passive_mask.any())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP


def test_multi_passive_sequential_landing_preserves_cumulative_mask():
    governor = _seed_creep_governor(passive_count=2)
    limits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    first_only = torch.tensor([[1.0, 1.0 - 5.0e-6]], dtype=torch.float64)
    _stationary_governor_sample(governor, first_only, limits)
    _stationary_governor_sample(governor, first_only, limits)
    assert torch.equal(
        governor.landed_passive_mask,
        torch.tensor([[True, False]]),
    )
    assert torch.equal(
        governor.active_passive_mask,
        torch.tensor([[False, True]]),
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP

    both = torch.ones((1, 2), dtype=torch.float64)
    _stationary_governor_sample(governor, both, limits)
    _stationary_governor_sample(governor, both, limits)
    assert torch.equal(
        governor.landed_passive_mask,
        torch.ones((1, 2), dtype=torch.bool),
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP


def test_landed_mask_clears_only_on_reversal_or_reset():
    governor = _seed_creep_governor(passive_count=2)
    limits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    position = torch.ones((1, 2), dtype=torch.float64)
    _stationary_governor_sample(governor, position, limits)
    _stationary_governor_sample(governor, position, limits)
    assert bool(governor.landed_passive_mask.all())

    governor.compute_speed_cap(
        position,
        torch.zeros_like(position),
        limits,
        torch.zeros((1, 1), dtype=torch.float64),
        -torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert not bool(governor.landed_passive_mask.any())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_TRACK

    governor._landed_passive.fill_(True)
    governor.reset()
    assert not bool(governor.landed_passive_mask.any())


def _landing6_lower_stop_band_governor(*, passive_count: int = 1):
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=passive_count,
        device="cpu",
        dtype=torch.float32,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(False)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    governor._governor_speed_cap.fill_(0.004)
    return governor


def _landing6_governor_sample(governor, position, velocity, limits, delta=1.0):
    return governor.compute_speed_cap(
        position,
        velocity,
        limits,
        torch.zeros((position.shape[0], 1), dtype=position.dtype),
        torch.full(
            (position.shape[0], 1), float(delta), dtype=position.dtype
        ),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )


def test_landing6_stop_band_remains_active_and_envelope_monitored():
    governor = _landing6_lower_stop_band_governor()
    limit = 0.03490658476948738
    limits = torch.tensor([[-limit, limit]], dtype=torch.float32)
    position = torch.tensor(
        [[-limit + 0.00015449151396751404]], dtype=torch.float32
    )
    velocity = torch.tensor([[-0.001642853021621704]], dtype=torch.float32)

    for _ in range(4):
        cap = _landing6_governor_sample(
            governor, position, velocity, limits
        )
        assert not bool(governor.landed_passive_mask.item())
        assert bool(governor.active_passive_mask.item())
        assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
        assert float(cap.item()) < 0.8


def test_stationary_solver_residual_lands_without_promoting_stop_band():
    """A measured 3.9-µrad solver residual is distinct from stop-band noise."""

    governor = _landing6_lower_stop_band_governor()
    limit = 0.03490658476948738
    limits = torch.tensor([[-limit, limit]], dtype=torch.float32)
    solver_residual = float(
        G2_PASSIVE_LIMIT_GOVERNOR_NUMERIC_LANDING_TOLERANCE_RAD
    ) - 1.0e-7
    position = torch.tensor([[-limit + solver_residual]], dtype=torch.float32)
    for _ in range(2):
        _landing6_governor_sample(
            governor, position, torch.zeros_like(position), limits
        )
    assert bool(governor.landed_passive_mask.item())
    assert not bool(governor.active_passive_mask.item())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP

    outside_numeric_landing = _landing6_lower_stop_band_governor()
    outside_position = torch.tensor(
        [[-limit + G2_PASSIVE_LIMIT_GOVERNOR_NUMERIC_LANDING_TOLERANCE_RAD + 1.0e-6]],
        dtype=torch.float32,
    )
    for _ in range(2):
        _landing6_governor_sample(
            outside_numeric_landing,
            outside_position,
            torch.zeros_like(outside_position),
            limits,
        )
    assert not bool(outside_numeric_landing.landed_passive_mask.item())
    assert bool(outside_numeric_landing.active_passive_mask.item())
    assert int(outside_numeric_landing.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP


def test_creep_existing_risk_stays_in_creep_instead_of_rebraking():
    """Landing8's active joint is not a new sequential risk."""

    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float32,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.4166667)
    governor._governor_speed_cap.fill_(0.012)
    limit = 0.03490658476948738
    cap = governor.compute_speed_cap(
        torch.tensor([[0.034276966]], dtype=torch.float32),
        torch.tensor([[0.0005532056]], dtype=torch.float32),
        torch.tensor([[-limit, limit]], dtype=torch.float32),
        torch.tensor([[0.012]], dtype=torch.float32),
        torch.tensor([[0.524925]], dtype=torch.float32),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
    assert bool(governor.active_passive_mask.item())
    assert 0.0 < float(cap.item()) < 0.8


def test_creep_active_envelope_retimes_cap_before_reactive_overspeed():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    governor._governor_speed_cap.fill_(0.012)
    cap = governor.compute_speed_cap(
        torch.tensor([[1.0 - 1.0e-5]], dtype=torch.float64),
        torch.tensor([[0.019]], dtype=torch.float64),
        torch.tensor([[-1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.012]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
    assert float(cap.item()) <= 0.012 + 1.0e-12
    assert (0.012 - float(cap.item())) / 0.002 <= 2.0 + 1.0e-12


def test_full_authority_release_uses_latched_not_noise_direction():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    governor._governor_speed_cap.fill_(0.004)
    for _ in range(3):
        governor.compute_speed_cap(
            torch.tensor([[0.999]], dtype=torch.float64),
            torch.tensor([[-0.001]], dtype=torch.float64),
            torch.tensor([[-1.0, 1.0]], dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=0.002,
        )
    assert bool(governor.active_passive_mask.item())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP


def test_settle_preserves_active_coordinate_until_creep_release_audit():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_SETTLE)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    governor._stationary_count.fill_(1)
    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    cap = governor.compute_speed_cap(
        torch.tensor([[0.999]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        limits,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
    assert bool(governor.active_passive_mask.item())
    assert bool(governor.active_toward_upper_mask.item())
    assert float(cap.item()) < 0.8


def test_creep_release_requires_nonrestriction_at_next_track_authority():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    governor._governor_speed_cap.fill_(0.004)
    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    position = torch.tensor([[0.999]], dtype=torch.float64)
    for _ in range(3):
        cap = governor.compute_speed_cap(
            position,
            torch.zeros_like(position),
            limits,
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=0.002,
        )
    assert bool(governor.active_passive_mask.item())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
    assert float(cap.item()) < 0.8


def test_full_authority_release_includes_proposed_master_braking_distance():
    """A stopped CREEP target cannot certify a future 0.8-rad/s release."""

    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_CREEP)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.fill_(0.2)
    governor._governor_speed_cap.fill_(0.004)

    maximum_master_speed = 0.8
    brake_acceleration = 2.0
    dt_s = 0.002
    reaction_steps = 2
    predicted_passive_speed = 3.0 * 0.2 * maximum_master_speed
    reaction_distance = predicted_passive_speed * reaction_steps * dt_s
    full_authority_braking_distance = (
        0.5
        * predicted_passive_speed
        * maximum_master_speed
        / brake_acceleration
    )
    margin = 0.01
    assert reaction_distance < margin
    assert margin < reaction_distance + full_authority_braking_distance

    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    position = torch.tensor([[1.0 - margin]], dtype=torch.float64)
    for _ in range(3):
        cap = governor.compute_speed_cap(
            position,
            torch.zeros_like(position),
            limits,
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            maximum_master_speed_rad_s=maximum_master_speed,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=dt_s,
        )

    assert bool(governor.active_passive_mask.item())
    assert int(governor.release_candidate_count.item()) == 0
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
    assert float(cap.item()) < maximum_master_speed


def test_later_unlearned_coordinate_can_learn_after_direction_latch():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=2,
        device="cpu",
        dtype=torch.float64,
    )
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio[0, 0] = 0.2
    governor.compute_speed_cap(
        torch.tensor([[0.0, 0.7]], dtype=torch.float64),
        torch.tensor([[0.0, 0.03]], dtype=torch.float64),
        torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.1]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert float(governor._maximum_transmission_ratio[0, 0].item()) == pytest.approx(0.2)
    assert float(governor._maximum_transmission_ratio[0, 1].item()) == pytest.approx(0.3)

    governor.compute_speed_cap(
        torch.tensor([[0.0, 0.7]], dtype=torch.float64),
        torch.tensor([[0.01, 0.02]], dtype=torch.float64),
        torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.1]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert float(governor._maximum_transmission_ratio[0, 0].item()) == pytest.approx(0.2)
    assert float(governor._maximum_transmission_ratio[0, 1].item()) == pytest.approx(0.3)


def test_stop_band_outside_stays_latched_when_full_track_is_unsafe():
    limit = 0.03490658476948738
    limits = torch.tensor([[-limit, limit]], dtype=torch.float32)

    outside = _landing6_lower_stop_band_governor()
    outside_position = torch.tensor(
        [[-limit + G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD + 1.0e-5]],
        dtype=torch.float32,
    )
    for _ in range(2):
        _landing6_governor_sample(
            outside,
            outside_position,
            torch.zeros_like(outside_position),
            limits,
        )
    assert not bool(outside.landed_passive_mask.item())
    assert bool(outside.active_passive_mask.item())
    assert int(outside.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP

    overspeed = _landing6_lower_stop_band_governor()
    inside_position = torch.tensor(
        [[-limit + 0.00015449151396751404]], dtype=torch.float32
    )
    for _ in range(2):
        _landing6_governor_sample(
            overspeed,
            inside_position,
            torch.tensor([[-0.006]], dtype=torch.float32),
            limits,
        )
    assert not bool(overspeed.landed_passive_mask.item())
    assert bool(overspeed.active_passive_mask.item())


def test_exact_landing_then_same_direction_stop_band_noise_does_not_retrigger():
    governor = _landing6_lower_stop_band_governor()
    limit = 0.03490658476948738
    limits = torch.tensor([[-limit, limit]], dtype=torch.float32)
    landing_position = torch.tensor([[-limit]], dtype=torch.float32)
    landing_velocity = torch.zeros_like(landing_position)
    for _ in range(2):
        _landing6_governor_sample(
            governor, landing_position, landing_velocity, limits
        )

    post_projection_position = torch.tensor(
        [[-limit + 0.00014340132474899292]], dtype=torch.float32
    )
    cap = _landing6_governor_sample(
        governor,
        post_projection_position,
        torch.tensor([[-0.0021941959857940674]], dtype=torch.float32),
        limits,
    )
    assert float(cap.item()) == pytest.approx(0.8)
    assert bool(governor.landed_passive_mask.item())
    assert not bool(governor.active_passive_mask.item())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP

    _landing6_governor_sample(
        governor,
        post_projection_position,
        torch.zeros_like(post_projection_position),
        limits,
        delta=-1.0,
    )
    assert not bool(governor.landed_passive_mask.item())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_TRACK


def test_exact_landed_coordinate_rearms_envelope_after_rebound_beyond_band():
    governor = _landing6_lower_stop_band_governor()
    limit = 0.03490658476948738
    limits = torch.tensor([[-limit, limit]], dtype=torch.float32)
    exact_lower = torch.tensor([[-limit]], dtype=torch.float32)
    for _ in range(2):
        _landing6_governor_sample(
            governor, exact_lower, torch.zeros_like(exact_lower), limits
        )
    assert bool(governor.landed_passive_mask.item())

    rebound = torch.tensor(
        [[-limit + G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD + 1.0e-4]],
        dtype=torch.float32,
    )
    cap = _landing6_governor_sample(
        governor,
        rebound,
        torch.tensor([[-0.2]], dtype=torch.float32),
        limits,
    )
    assert bool(governor.landed_passive_mask.item())  # historical latch remains
    assert bool(governor.active_passive_mask.item())  # envelope is re-armed
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_BRAKE
    assert float(cap.item()) < 0.8


def test_landing6_seven_joint_upper_lower_stop_band_identity_is_preserved():
    governor = _landing6_lower_stop_band_governor(passive_count=7)
    governor._active_passive.zero_()
    governor._active_passive[:, 4] = True
    governor._landed_passive[:, 3] = True
    governor._active_toward_upper.zero_()
    governor._landed_toward_upper[:, 3] = True
    limit = 0.03490658476948738
    limits = torch.tensor([[-1.0, 1.0]] * 7, dtype=torch.float32)
    limits[3] = torch.tensor([-limit, limit], dtype=torch.float32)
    limits[4] = torch.tensor([-limit, limit], dtype=torch.float32)
    position = torch.zeros((1, 7), dtype=torch.float32)
    position[:, 3] = limit - 0.00014340132474899292
    position[:, 4] = -limit + 0.00015449151396751404
    velocity = torch.zeros_like(position)
    velocity[:, 4] = -0.001642853021621704

    _landing6_governor_sample(governor, position, velocity, limits)
    _landing6_governor_sample(governor, position, velocity, limits)
    expected_landed = torch.tensor(
        [[False, False, False, True, False, False, False]],
        dtype=torch.bool,
    )
    expected_active = torch.tensor(
        [[False, False, False, False, True, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(governor.landed_passive_mask, expected_landed)
    assert torch.equal(governor.active_passive_mask, expected_active)
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP

    position[:, 4] = -limit
    velocity[:, 4] = 0.0
    _landing6_governor_sample(governor, position, velocity, limits)
    _landing6_governor_sample(governor, position, velocity, limits)
    expected_landed[:, 4] = True
    assert torch.equal(governor.landed_passive_mask, expected_landed)
    assert not bool(governor.active_passive_mask.any())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP


def test_non_limit_trigger_settles_back_to_track_without_permanent_ratchet():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    governor._phase.fill_(G2_PASSIVE_LIMIT_GOVERNOR_SETTLE)
    governor._stationary_count.fill_(1)
    governor._active_passive.fill_(True)
    governor._active_toward_upper.fill_(True)
    governor._active_master_direction.fill_(1.0)
    governor._maximum_transmission_ratio.zero_()
    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    cap = governor.compute_speed_cap(
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        limits,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_CREEP
    assert bool(governor.active_passive_mask.any())
    assert float(cap.item()) < 0.8
    for _ in range(2):
        cap = governor.compute_speed_cap(
            torch.zeros((1, 1), dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            limits,
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=0.002,
        )
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_TRACK
    assert not bool(governor.active_passive_mask.any())
    assert float(cap.item()) == pytest.approx(0.8)
    assert float(governor._maximum_transmission_ratio.item()) == 0.0
    governor.compute_speed_cap(
        torch.zeros((1, 1), dtype=torch.float64),
        torch.full((1, 1), 0.2, dtype=torch.float64),
        limits,
        torch.full((1, 1), 0.2, dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    # Every clean TRACK sample updates a monotone maximum.  This prevents a
    # sequential linkage joint from retaining a zero/underestimated CREEP
    # authority estimate after an earlier joint triggered.
    assert float(governor._maximum_transmission_ratio.item()) == pytest.approx(1.0)


def test_boundary_hit_then_two_causal_away_samples_release_without_landing():
    governor = _seed_creep_governor(passive_count=1)
    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    # A moving sample at the boundary is not a settled landing and is not
    # released on one observation.
    governor.compute_speed_cap(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([[-0.01]], dtype=torch.float64),
        limits,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert bool(governor.active_passive_mask.item())
    assert not bool(governor.landed_passive_mask.item())

    away_position = torch.tensor([[0.9997]], dtype=torch.float64)
    for expected_count in (1, 0):
        governor.compute_speed_cap(
            away_position,
            torch.tensor([[-0.01]], dtype=torch.float64),
            limits,
            torch.zeros((1, 1), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=0.002,
        )
        assert int(governor.release_candidate_count.item()) == expected_count
    assert not bool(governor.active_passive_mask.item())
    assert not bool(governor.landed_passive_mask.item())
    # Moving away clears the active mask but the full-authority cap is still
    # restrictive near the former stop.  It must remain conservatively capped
    # rather than returning to TRACK and immediately re-entering BRAKE.
    assert int(governor.phase.item()) == (
        G2_PASSIVE_LIMIT_GOVERNOR_POST_RELEASE_CONSERVATIVE
    )
    assert bool(governor.post_release_passive_mask.item())
    # The retained coordinate does not re-enter the active mask on the next
    # normal decision while its full-authority proof is incomplete.
    governor.compute_speed_cap(
        away_position,
        torch.tensor([[-0.01]], dtype=torch.float64),
        limits,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert not bool(governor.active_passive_mask.item())
    assert bool(governor.post_release_passive_mask.item())
    # Release is not a command reversal: the direction/ratio latch persists.
    assert float(governor._active_master_direction.item()) == 1.0


def test_stop_band_toward_and_away_noise_cannot_release_active_joint():
    governor = _seed_creep_governor(passive_count=1)
    governor._governor_speed_cap.fill_(0.004)
    limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    position = torch.tensor(
        [[1.0 - G2_PASSIVE_LIMIT_GOVERNOR_STOP_BAND_RAD + 1.0e-6]],
        dtype=torch.float64,
    )

    governor.compute_speed_cap(
        position,
        torch.tensor([[-0.001]], dtype=torch.float64),
        limits,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert int(governor.release_candidate_count.item()) == 0
    assert bool(governor.active_passive_mask.item())

    for _ in range(2):
        governor.compute_speed_cap(
            position,
            torch.tensor([[0.001]], dtype=torch.float64),
            limits,
            torch.full((1, 1), 0.8, dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=0.002,
        )
        assert bool(governor.active_passive_mask.item())
        assert int(governor.release_candidate_count.item()) == 0
    assert not bool(governor.landed_passive_mask.item())


def test_seven_joint_sequential_land_and_nonrestrictive_release_are_distinct():
    governor = _seed_creep_governor(passive_count=7)
    governor._governor_speed_cap.fill_(0.004)
    limits = torch.tensor([[-1.0, 1.0]] * 7, dtype=torch.float64)
    position = torch.tensor(
        [[
            1.0,
            0.5,
            1.0 - 5.0e-6,
            1.0,
            1.0 - 5.0e-6,
            0.5,
            1.0 - 5.0e-6,
        ]],
        dtype=torch.float64,
    )
    _stationary_governor_sample(governor, position, limits)
    _stationary_governor_sample(governor, position, limits)
    assert torch.equal(
        governor.landed_passive_mask,
        torch.tensor([[True, False, False, True, False, False, False]]),
    )
    assert torch.equal(
        governor.active_passive_mask,
        torch.tensor([[False, False, True, False, True, False, True]]),
    )

    position[:, [2, 4, 6]] = 1.0
    _stationary_governor_sample(governor, position, limits)
    _stationary_governor_sample(governor, position, limits)
    assert torch.equal(
        governor.landed_passive_mask,
        torch.tensor([[True, False, True, True, True, False, True]]),
    )
    assert not bool(governor.active_passive_mask.any())
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_POST_STOP


def test_float32_batched_multi_hot_masks_preserve_shape_dtype_and_identity():
    position, velocity, limits, _, _, expected = _seven_passive_risk_fixture(
        dtype=torch.float32
    )
    position = position.repeat(3, 1)
    velocity = velocity.repeat(3, 1)
    velocity[1].zero_()
    position[2] = -position[2]
    velocity[2] = -velocity[2]
    governor = PassiveLimitMasterGovernor(
        num_envs=3,
        passive_joint_count=7,
        device="cpu",
        dtype=torch.float32,
    )
    cap = governor.compute_speed_cap(
        position,
        velocity,
        limits,
        torch.zeros((3, 1), dtype=torch.float32),
        torch.ones((3, 1), dtype=torch.float32),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert cap.shape == (3, 1)
    assert cap.dtype == torch.float32
    assert governor.active_passive_mask.dtype == torch.bool
    assert torch.equal(governor.active_passive_mask[0:1], expected)
    assert not bool(governor.active_passive_mask[1].any())
    assert torch.equal(governor.active_passive_mask[2:3], expected)


def test_track_measured_speed_is_not_tripled_by_terminal_uncertainty():
    governor = PassiveLimitMasterGovernor(
        num_envs=1,
        passive_joint_count=1,
        device="cpu",
        dtype=torch.float64,
    )
    limit = 1.0
    cap = governor.compute_speed_cap(
        torch.tensor([[limit - 0.00008]], dtype=torch.float64),
        torch.tensor([[0.01]], dtype=torch.float64),
        torch.tensor([[-limit, limit]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=10.0,
        dt_s=0.002,
    )
    assert float(cap.item()) == pytest.approx(0.8)
    assert int(governor.phase.item()) == G2_PASSIVE_LIMIT_GOVERNOR_TRACK


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_capture_endpoint_predicate_completes_within_four_times_921(dtype):
    dt_s = 0.002
    nominal_steps = 921
    endpoint = torch.full((4, 1), math.pi / 4.0, dtype=dtype)
    target = torch.zeros_like(endpoint)
    velocity = torch.zeros_like(endpoint)
    governor = PassiveLimitMasterGovernor(
        num_envs=4,
        passive_joint_count=7,
        device="cpu",
        dtype=dtype,
    )
    passive_position = torch.zeros((4, 7), dtype=dtype)
    passive_velocity = torch.zeros_like(passive_position)
    passive_limits = torch.tensor(
        [[-2.0, 2.0]] * 7,
        dtype=dtype,
    )
    maximum_speed = 0.0
    maximum_acceleration = 0.0
    completed_step = None
    for step in range(1, 4 * nominal_steps + 1):
        tau = min(float(step) / float(nominal_steps), 1.0)
        alpha = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        desired = endpoint * alpha if step <= nominal_steps else endpoint
        speed_cap = governor.compute_speed_cap(
            passive_position,
            passive_velocity,
            passive_limits,
            velocity,
            desired - target,
            maximum_master_speed_rad_s=0.8,
            maximum_master_target_acceleration_rad_s2=10.0,
            dt_s=dt_s,
        )
        next_target, next_velocity = acceleration_limited_position_target_step(
            target,
            desired,
            velocity,
            maximum_speed_rad_s=speed_cap,
            maximum_acceleration_rad_s2=10.0,
            dt_s=dt_s,
        )
        assert bool((next_target >= target).all())
        assert bool((next_target <= endpoint).all())
        maximum_speed = max(
            maximum_speed,
            float(torch.max(torch.abs(next_velocity))),
        )
        maximum_acceleration = max(
            maximum_acceleration,
            float(torch.max(torch.abs(next_velocity - velocity))) / dt_s,
        )
        target = next_target
        velocity = next_velocity
        if (
            float(torch.max(torch.abs(endpoint - target))) <= 1.0e-7
            and float(torch.max(torch.abs(velocity))) <= 1.0e-7
        ):
            completed_step = step
            break
    assert completed_step is not None
    assert completed_step <= 4 * nominal_steps
    assert maximum_speed <= 0.8 + 1.0e-6
    assert maximum_acceleration <= 10.0 + 1.0e-4


def test_passive_limit_guard_zero_speed_restart_is_acceleration_limited():
    dt_s = 0.002
    acceleration = 10.0
    passive_limit = 0.03490658476948738
    cap = passive_limit_aware_master_speed_cap(
        torch.tensor([[passive_limit - 1.0e-6]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.tensor([[-passive_limit, passive_limit]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        maximum_master_speed_rad_s=0.8,
        maximum_master_target_acceleration_rad_s2=acceleration,
        dt_s=dt_s,
    )
    _, next_velocity = acceleration_limited_position_target_step(
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        maximum_speed_rad_s=cap,
        maximum_acceleration_rad_s2=acceleration,
        dt_s=dt_s,
    )
    assert float(cap.item()) == pytest.approx(0.8)
    assert float(next_velocity.item()) == pytest.approx(
        acceleration * dt_s
    )


def test_passive_limit_guard_contract_is_conservative_not_a_limit_change():
    metadata = shared_gripper_reset_contract_metadata()
    guard = metadata["passive_limit_aware_open_retiming"]
    assert guard["enabled"] is True
    assert guard["passive_coordinates_are_targeted"] is False
    assert guard["velocity_source"] == (
        "PER_PHYSICS_STEP_POSITION_FINITE_DIFFERENCE"
    )
    assert guard["braking_authority"] == "MASTER_TARGET_ACCELERATION"
    assert guard["braking_model"] == (
        "PASSIVE_DISTANCE_DURING_MASTER_TARGET_STOP"
    )
    assert guard["master_target_braking_acceleration_rad_s2"] == 10.0
    assert guard["governor_command_braking_acceleration_rad_s2"] == 2.0
    assert guard["transmission_uncertainty_multiplier"] == 3.0
    assert "2.80179" in guard["transmission_uncertainty_evidence"]
    assert guard["transmission_uncertainty_application"] == (
        "CREEP_PREDICTED_BRANCH_ONLY_MAX_MEASURED_3X_PREDICTED"
    )
    assert guard["track_prediction_uncertainty_multiplier"] == 1.0
    assert guard["transmission_ratio_learning"] == (
        "CLEAN_TRACK_MONOTONIC_PER_COORDINATE_MAX_INCLUDING_LATE_"
        "SEQUENTIAL_MOTION"
    )
    assert guard["settle_release_authority"] == (
        "TWO_SAMPLE_NONRESTRICTIVE_PROOF_AT_NEXT_FULL_TRACK_CAP"
    )
    assert guard["creep_existing_risk_authority"] == (
        "CAUSAL_ACTIVE_ENVELOPE_CAP_AND_MEASURED_LANDING_OVERSPEED_"
        "BRAKE_ONLY_NEW_SEQUENTIAL_RISK"
    )
    assert guard["risk_latch"] == "PER_CAP_MULTI_HOT_ORDER_EQUIVARIANT"
    assert guard["landed_latch"].startswith("CUMULATIVE_DIRECTION_SPECIFIC")
    assert guard["position_target_generator"] == (
        "STOPPING_DISTANCE_AWARE_MONOTONIC_NO_OVERSHOOT"
    )
    assert guard["state_machine"] == (
        "TRACK_BRAKE_SETTLE_CREEP_POST_STOP_POST_RELEASE_CONSERVATIVE"
    )
    assert guard["settle_speed_rad_s"] == 0.005
    assert guard["settle_consecutive_samples"] == 2
    assert guard["stop_band_hysteresis_rad"] == pytest.approx(2.0e-4)
    assert "0.000154492" in guard["stop_band_hysteresis_evidence"]
    assert "NOT_LANDED_AUTHORITY" in guard["stop_band_hysteresis_evidence"]
    assert guard["post_stop_latch_clear_events"] == [
        "DIRECTION_REVERSAL",
        "RESET",
    ]
    assert guard["hard_velocity_limit_rad_s"] == 0.8
    assert guard["hard_acceleration_limit_rad_s2"] == 10.0
    assert guard["joint_limits_changed"] is False
    assert guard["drive_gain_or_effort_changed"] is False


def test_passive_limit_guard_has_no_per_step_cpu_materialization():
    source = inspect.getsource(passive_limit_aware_master_speed_cap)
    for forbidden in (".cpu(", ".numpy(", ".tolist(", ".item("):
        assert forbidden not in source
    per_passive_source = inspect.getsource(
        _passive_limit_aware_master_speed_cap_per_passive
    )
    for forbidden in (".cpu(", ".numpy(", ".tolist(", ".item("):
        assert forbidden not in per_passive_source
    governor_source = inspect.getsource(
        PassiveLimitMasterGovernor.compute_speed_cap
    )
    for forbidden in (".cpu(", ".numpy(", ".tolist(", ".item("):
        assert forbidden not in governor_source


def test_full46_fd_audit_keeps_first_interval_without_fabricating_acceleration():
    initial = torch.zeros((1, 46), dtype=torch.float32)
    audit = FullArticulationFDSafetyAudit.start(
        initial, G2_RUNTIME_JOINT_ORDER, dt_s=0.02
    )
    next_position = initial.clone()
    next_position[0, 0] = 0.001
    audit.observe(next_position, step=1)
    result = audit.result()
    assert result["joint_count"] == 46
    assert result["sample_count"] == 1
    assert result["first_post_reset_46dof_fd_sample_included"] is True
    assert result["first_post_reset_position_interval_included"] is True
    assert result["first_post_reset_acceleration_resolved"] is False
    assert (
        result["first_post_reset_acceleration_status"]
        == "UNRESOLVED_NO_PRIOR_POSITION_INTERVAL"
    )
    assert result["first_interval_maximum_position_delta_rad"] == pytest.approx(
        0.001
    )
    assert result["maximum_fd_velocity_rad_s"] == pytest.approx(0.05)
    assert result["maximum_fd_acceleration_rad_s2"] is None
    assert result["full46_validated_fd_acceleration_sample_count"] == 0
    assert result["pass"] is False


def test_full46_fd_first_valid_acceleration_uses_q0_q1_q2_and_requires_settle():
    initial = torch.zeros((1, 46), dtype=torch.float32)
    audit = FullArticulationFDSafetyAudit.start(
        initial, G2_RUNTIME_JOINT_ORDER, dt_s=0.02
    )
    positions = (0.001, 0.0018, 0.0026, 0.0034, 0.0042, 0.0050)
    for step, value in enumerate(positions, 1):
        position = initial.clone()
        position[0, 0] = value
        audit.observe(position, step=step)
    result = audit.result()
    # v01=0.05, v12=0.04, so the first real FD acceleration is -0.5.
    # The former zero-velocity assumption would have reported 2.5 at step 1.
    assert result["maximum_fd_velocity_rad_s"] == pytest.approx(0.05)
    assert result["maximum_fd_acceleration_rad_s2"] == pytest.approx(0.5)
    assert result["maximum_acceleration_event"]["step"] == 2
    assert result["full46_validated_fd_acceleration_sample_count"] == 5
    assert result["consecutive_settled_full46_samples"] == 5
    assert result["pass"] is True


def test_measured_cache_is_rejected_without_full46_settle_attestation():
    names = list(G2_RUNTIME_JOINT_ORDER)
    measured = torch.zeros((1, 46), dtype=torch.float32)
    with pytest.raises(TypeError):
        capture_measured_right_hand_cache(measured, names)
    with pytest.raises(ValueError, match="REQUIRES_SAFE_SETTLE_PASS"):
        capture_measured_right_hand_cache(
            measured,
            names,
            safety_attestation={
                "schema": "g2_constraint_stable_gripper_reset_v2",
                "pass": False,
            },
        )


def test_measured_cache_rechecks_raw_limits_and_minimum_settle_without_pass_masking():
    names = list(G2_RUNTIME_JOINT_ORDER)
    measured = torch.zeros((1, 46), dtype=torch.float32)
    audit = FullArticulationFDSafetyAudit.start(
        measured, names, dt_s=0.02
    )
    for step in range(1, 61):
        audit.observe(measured, step=step)
    valid = audit.result()
    assert valid["pass"] is True

    too_short = dict(valid, sample_count=59)
    with pytest.raises(ValueError, match="REQUIRES_MINIMUM_OPEN_SETTLE"):
        capture_measured_right_hand_cache(
            measured, names, safety_attestation=too_short
        )

    masked_velocity_failure = dict(
        valid,
        maximum_fd_velocity_rad_s=0.800001,
    )
    with pytest.raises(ValueError, match="ATTESTATION_VELOCITY_FAILED"):
        capture_measured_right_hand_cache(
            measured, names, safety_attestation=masked_velocity_failure
        )

    masked_acceleration_failure = dict(
        valid,
        maximum_fd_acceleration_rad_s2=10.000001,
    )
    with pytest.raises(ValueError, match="ATTESTATION_ACCELERATION_FAILED"):
        capture_measured_right_hand_cache(
            measured, names, safety_attestation=masked_acceleration_failure
        )


def test_first_interval_velocity_and_second_interval_acceleration_both_fail_closed():
    initial = torch.zeros((1, 46), dtype=torch.float32)
    velocity_failure = FullArticulationFDSafetyAudit.start(
        initial, G2_RUNTIME_JOINT_ORDER, dt_s=0.02
    )
    position = initial.clone()
    position[0, 7] = 0.02  # 1 rad/s in the first retained interval.
    velocity_failure.observe(position, step=1)
    for step in range(2, 8):
        velocity_failure.observe(position, step=step)
    velocity_result = velocity_failure.result()
    assert velocity_result["maximum_fd_velocity_rad_s"] == pytest.approx(1.0)
    assert velocity_result["pass"] is False

    acceleration_failure = FullArticulationFDSafetyAudit.start(
        initial, G2_RUNTIME_JOINT_ORDER, dt_s=0.02
    )
    q1 = initial.clone()
    q1[0, 9] = 0.001
    acceleration_failure.observe(q1, step=1)
    q2 = initial.clone()
    q2[0, 9] = 0.0062  # v jumps 0.05 -> 0.26: 10.5 rad/s^2.
    acceleration_failure.observe(q2, step=2)
    for step in range(3, 9):
        continued = initial.clone()
        continued[0, 9] = 0.0062 + (step - 2) * 0.0052
        acceleration_failure.observe(continued, step=step)
    acceleration_result = acceleration_failure.result()
    assert acceleration_result["maximum_fd_acceleration_rad_s2"] >= 10.0
    assert acceleration_result["maximum_acceleration_event"]["step"] == 2
    assert acceleration_result["pass"] is False


def test_async_reset_event_never_steps_or_targets_full_hand():
    curriculum = (
        ROOT / "source/geniesim/rl/isaaclab/g2_visual_curriculum.py"
    ).read_text()
    assert "env.step(" not in curriculum
    assert "q[:, arm_ids], joint_ids=arm_ids" in curriculum
    assert "robot.set_joint_position_target(q, env_ids=env_ids)" not in curriculum


def test_keyboard_teacher_student_share_reset_contract_and_cache_helper():
    keyboard = (ROOT / "scripts/run_g2_keyboard_teacher_collection.py").read_text()
    teacher = (
        ROOT / "scripts/diagnostics/g2_milestone7_teacher_sac_phase.py"
    ).read_text()
    student = (
        ROOT / "scripts/diagnostics/g2_visual_reverse_sac_phase.py"
    ).read_text()
    m2 = (
        ROOT
        / "source/geniesim/rl/isaaclab/g2_rebuild/sensor_validation_live.py"
    ).read_text()
    for source in (keyboard, teacher, student, m2):
        assert "FullArticulationFDSafetyAudit" in source
        assert "capture_measured_right_hand_cache" in source
        assert "safety_attestation=" in source
        assert "install_measured_right_hand_cache" in source
        assert "initial_joint_position_seed_rad" not in source
    for config_path in (
        "source/geniesim/rl/isaaclab/g2_teacher_sac_env_cfg.py",
        "source/geniesim/rl/isaaclab/g2_visual_sac_env_cfg.py",
        "source/geniesim/rl/isaaclab/g2_redundancy_teleop_env_cfg.py",
    ):
        source = (ROOT / config_path).read_text()
        assert "apply_keyboard_collection_initial_pose" in source


def test_reset_target_writes_exclude_passive_and_mimic_hand_coordinates():
    task_mdp = (
        ROOT / "source/geniesim/rl/isaaclab/g2_lift_task_mdp.py"
    ).read_text()
    assert "set_joint_position_target(target, joint_ids=ids, env_ids=env_ids)" in task_mdp
    curriculum = (
        ROOT / "source/geniesim/rl/isaaclab/g2_visual_curriculum.py"
    ).read_text()
    assert "q[:, arm_ids], joint_ids=arm_ids" in curriculum


def test_canonical_binary_action_uses_same_passive_limit_guard_as_capture():
    action_source = (
        ROOT / "source/geniesim/rl/isaaclab/g2_redundancy_action.py"
    ).read_text()
    binary_start = action_source.index(
        "class G2RateLimitedBinaryJointPositionAction("
    )
    binary_end = action_source.index(
        "class G2RateLimitedBinaryJointPositionActionCfg", binary_start
    )
    binary = action_source[binary_start:binary_end]
    assert "G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES" in binary
    assert "PassiveLimitMasterGovernor" in binary
    assert "self._g2_passive_limit_governor.compute_speed_cap" in binary
    assert "self._g2_passive_limit_governor.reset(env_ids)" in binary
    assert "maximum_master_target_acceleration_rad_s2" in binary
    assert "self.cfg.maximum_joint_target_acceleration_rad_s2" in binary
    assert "maximum_speed_override_rad_s=passive_guard_cap" in binary
    assert "set_joint_position_target_index" in binary
    assert "target=target, joint_ids=self._joint_ids" in binary
