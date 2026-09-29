# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU-only contract tests for the real-row Stage-1A coordinator."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from geniesim.rl.sac.human_grasp_gru_bc import (
    HumanGraspGRUBC,
    HumanGraspSequenceInputs,
)
from geniesim.rl.sac.stage1a_grasp_reward import Stage1ARewardStep
from geniesim.rl.sac.residual_sac_runtime import (
    residual_sac_actor_checkpoint_payload,
)
from geniesim.rl.sac.stage2_sac import SACConfig, SquashedGaussianActor
from geniesim.rl.sac.stage1a_real_sac_coordinator import (
    Stage1AAcceptedRealRow,
    Stage1ARealSACCoordinator,
    Stage1ARealSACError,
    CANDIDATE_A_DYNAMICS_AUTHORITY,
    REPLAY_SOURCE_EXISTING_CLOSE,
    REPLAY_SOURCE_ONLINE_SAC,
    REPLAY_STRATEGY_HER,
    REPLAY_STRATEGY_HER_FORCE,
    REPLAY_STRATEGY_SAC,
    RUNTIME_HARDSTOP_FAILURE,
    STAGE1A_SAC_GAMMA,
    radial_metric_residual,
)


def _coordinator(tmp_path: Path) -> Stage1ARealSACCoordinator:
    checkpoint = tmp_path / "frozen_bc.pt"
    checkpoint.write_bytes(b"immutable BC provenance fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return Stage1ARealSACCoordinator(
        frozen_bc=HumanGraspGRUBC(),
        bc_checkpoint_path=checkpoint,
        expected_bc_sha256=digest,
        replay_capacity=128,
        seed=7,
    )


def _coordinator_with_strategy(
    tmp_path: Path, strategy: str
) -> Stage1ARealSACCoordinator:
    checkpoint = tmp_path / f"frozen_bc_{strategy}.pt"
    checkpoint.write_bytes(b"immutable BC provenance fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return Stage1ARealSACCoordinator(
        frozen_bc=HumanGraspGRUBC(),
        bc_checkpoint_path=checkpoint,
        expected_bc_sha256=digest,
        replay_capacity=128,
        seed=7,
        replay_strategy=strategy,
    )


def _inputs() -> HumanGraspSequenceInputs:
    return HumanGraspSequenceInputs(
        right_wrist_rgb=torch.zeros(1, 1, 48, 64, 3),
        right_wrist_depth_m=torch.ones(1, 1, 48, 64, 1),
        right_wrist_depth_valid=torch.ones(1, 1, 48, 64, 1, dtype=torch.bool),
        ee_pose_robot_root_m_xyzw=torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]
        ),
        right_arm_joint_position_rad=torch.zeros(1, 1, 7),
        right_arm_joint_velocity_rad_s=torch.zeros(1, 1, 7),
        current_gripper_state=torch.zeros(1, 1, 1),
        previous_policy_action_4d_metric_root_m=torch.zeros(1, 1, 4),
        hidden_reset_mask=torch.ones(1, 1, dtype=torch.bool),
    )


def _reward(*, contact: bool = False, stable: bool = False) -> Stage1ARewardStep:
    zero = lambda: torch.zeros(1)
    boolean = lambda value=False: torch.tensor([value], dtype=torch.bool)
    return Stage1ARewardStep(
        reward_total=torch.tensor([2.0 if stable else 0.1]),
        done=boolean(stable), success=boolean(stable),
        reward_progress=zero(), reward_lateral=zero(),
        reward_single_contact=zero(),
        reward_bilateral=torch.tensor([1.0 if contact else 0.0]),
        reward_stable=torch.tensor([1.5 if stable else 0.0]),
        reward_success=torch.tensor([3.0 if stable else 0.0]),
        penalty_single_contact_dwell=zero(), penalty_hover=zero(),
        penalty_wrong_direction=zero(), residual_penalty=zero(),
        smoothness_penalty=zero(), time_penalty=zero(), safety_penalty=zero(),
        q_antipodal=zero(), q_center=zero(), q_force_balance=zero(),
        q_slip=zero(), q_impact=zero(), q_omega=zero(), q_grasp=zero(),
        q_antipodal_available=boolean(), q_center_available=boolean(),
        left_contact_boolean=boolean(contact), right_contact_boolean=boolean(contact),
        single_contact_boolean=boolean(), bilateral_boolean=boolean(contact),
        bilateral_contact_boolean=boolean(contact), stable_grasp_boolean=boolean(stable),
        contact_loss_fail=boolean(), bilateral_stable_counter=zero(),
        single_contact_counter=zero(), hover_counter=zero(),
        wrong_direction_counter=zero(), contact_loss_counter=zero(),
        cosine_similarity=zero(), progress_m=zero(), lateral_improvement_m=zero(),
        left_normal_force_n=torch.tensor([1.2 if contact else 0.0]),
        right_normal_force_n=torch.tensor([1.2 if contact else 0.0]),
        left_contact_substep_count=zero(), right_contact_substep_count=zero(),
        force_aggregation_used_impulse=torch.zeros(1, 2, dtype=torch.bool),
        tangential_slip_m_s=zero(), contact_normal_velocity_m_s=zero(),
        cube_linear_velocity_m_s=zero(), cube_angular_velocity_rad_s=zero(),
        residual_norm_mm=zero(), final_action_norm_mm=zero(),
        contact_latched=boolean(contact), non_pad_gripper_cube_contact=boolean(),
        no_progress_hovering=boolean(), band_edge_hovering=boolean(),
    )


def test_radial_mapping_is_smooth_bounded_and_not_component_clipping() -> None:
    zero = radial_metric_residual([0.0, 0.0, 0.0])
    mapped = np.asarray(radial_metric_residual([1.0, 1.0, 0.0]))
    assert zero == (0.0, 0.0, 0.0)
    assert np.linalg.norm(mapped) < 0.0045
    assert mapped[0] == pytest.approx(mapped[1])


def test_sac_and_her_force_are_distinct_real_replay_strategies(
    tmp_path: Path,
) -> None:
    baseline = _coordinator_with_strategy(tmp_path, REPLAY_STRATEGY_SAC)
    her_force = _coordinator_with_strategy(tmp_path, REPLAY_STRATEGY_HER_FORCE)
    assert baseline.replay_strategy == "SAC"
    assert not baseline.replay.prioritized
    assert her_force.replay_strategy == "HER_FORCE"
    assert her_force.replay.prioritized


def test_ordinary_her_fails_closed_without_goal_conditioned_actor_observation(
    tmp_path: Path,
) -> None:
    with pytest.raises(Stage1ARealSACError, match="goal-conditioned"):
        _coordinator_with_strategy(tmp_path, REPLAY_STRATEGY_HER)


def test_verified_residual_actor_checkpoint_initializes_training_actor(
    tmp_path: Path,
) -> None:
    bc_checkpoint = tmp_path / "frozen_bc_actor_init.pt"
    bc_checkpoint.write_bytes(b"immutable BC provenance fixture")
    bc_sha = hashlib.sha256(bc_checkpoint.read_bytes()).hexdigest()
    actor = SquashedGaussianActor(
        SACConfig(
            observation_dim=128,
            action_dim=3,
            action_mask=(1.0, 1.0, 1.0),
            gamma=0.9993,
            initial_alpha=0.01,
            pose_auxiliary_weight=0.1,
            seed=7,
        )
    )
    with torch.no_grad():
        actor.mean.bias.copy_(torch.tensor((0.1, -0.2, 0.3)))
    actor_checkpoint = tmp_path / "residual_actor.pt"
    torch.save(
        residual_sac_actor_checkpoint_payload(
            actor, human_grasp_checkpoint_sha256=bc_sha
        ),
        actor_checkpoint,
    )
    actor_sha = hashlib.sha256(actor_checkpoint.read_bytes()).hexdigest()
    coordinator = Stage1ARealSACCoordinator(
        frozen_bc=HumanGraspGRUBC(),
        bc_checkpoint_path=bc_checkpoint,
        expected_bc_sha256=bc_sha,
        replay_capacity=128,
        seed=7,
        replay_strategy=REPLAY_STRATEGY_SAC,
        residual_actor_checkpoint_path=actor_checkpoint,
        residual_actor_checkpoint_sha256=actor_sha,
    )
    assert coordinator.actor_initialization["authority"] == (
        "VERIFIED_RESIDUAL_ACTOR_CHECKPOINT"
    )
    assert torch.equal(coordinator.agent.actor.mean.bias, actor.mean.bias)


def test_replay_action_is_exact_post_gate_residual_semantics(tmp_path: Path) -> None:
    """Replay owns the residual that can actually reach composition, not a shadow sample."""

    coordinator = _coordinator(tmp_path)
    with torch.no_grad():
        # Make the deterministic actor observably non-zero.  The inactive path
        # must still store/apply zero because the phase gate precedes residual
        # composition; the active path must preserve the exact normalized
        # action and its analytic metric mapping.
        coordinator.agent.actor.mean.weight.zero_()
        coordinator.agent.actor.mean.bias.copy_(
            torch.tensor((0.25, -0.125, 0.0625))
        )

    inactive = coordinator.propose(
        _inputs(), deterministic=True, residual_active=False
    )
    assert np.array_equal(
        inactive.normalized_sac_action, np.zeros(3, dtype=np.float32)
    )
    assert inactive.raw_residual_metric_root_m == (0.0, 0.0, 0.0)
    assert inactive.composition.scaled_residual_contribution_m == (0.0, 0.0, 0.0)

    coordinator.reset_episode()
    active = coordinator.propose(
        _inputs(), deterministic=True, residual_active=True
    )
    assert np.linalg.norm(active.normalized_sac_action) > 0.0
    assert np.asarray(active.raw_residual_metric_root_m) == pytest.approx(
        radial_metric_residual(active.normalized_sac_action), abs=1.0e-12
    )
    assert np.asarray(active.composition.scaled_residual_contribution_m) == pytest.approx(
        0.10 * np.asarray(active.raw_residual_metric_root_m), abs=1.0e-12
    )

    coordinator.accept_real_transition(
        Stage1AAcceptedRealRow(
            row_id="active-post-gate-action",
            proposal=active,
            next_actor_observation=active.actor_observation,
            reward_step=_reward(),
        )
    )
    assert np.array_equal(coordinator.replay.actions[0], active.normalized_sac_action)


def test_far_nominal_override_keeps_gru_history_but_cannot_close(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(
        _inputs(),
        deterministic=True,
        residual_active=False,
        nominal_action_override_4d_metric_root_m=(0.001, 0.0, 0.0, 0.0),
        gripper_authority_active=False,
    )
    assert proposal.actor_observation.shape == (128,)
    assert coordinator.hidden is not None
    assert proposal.bc_action_4d_metric_root_m == pytest.approx(
        (0.001, 0.0, 0.0, 0.0)
    )
    assert proposal.composition.final_gripper_probability == 0.0
    assert np.array_equal(
        proposal.normalized_sac_action, np.zeros(3, dtype=np.float32)
    )
    with pytest.raises(Stage1ARealSACError, match="preserve OPEN"):
        coordinator.propose(
            _inputs(),
            residual_active=False,
            nominal_action_override_4d_metric_root_m=(0.0, 0.0, 0.0, 1.0),
            gripper_authority_active=False,
        )


def test_gru_xyz_can_remain_nominal_owner_while_close_is_logging_only(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    coordinator._closed = True
    proposal = coordinator.propose(
        _inputs(),
        deterministic=True,
        residual_active=False,
        gripper_authority_active=False,
    )
    assert proposal.actor_observation.shape == (128,)
    assert proposal.bc_action_4d_metric_root_m[:3] == pytest.approx(
        proposal.composition.bc_action_metric_root_m[:3]
    )
    assert proposal.bc_action_4d_metric_root_m[3] == 0.0
    assert proposal.composition.final_gripper_probability == 0.0
    # Logging-only inference must not mutate the coordinator's legacy CLOSE
    # latch.  The simplified external gate owns the sole canonical edge.
    assert coordinator._closed is True


def test_real_rows_drive_priority_update_and_checkpoint_reload(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    assert np.linalg.norm(proposal.raw_residual_metric_root_m) == pytest.approx(0.0)
    assert proposal.composition.final_gripper_probability == proposal.bc_action_4d_metric_root_m[3]
    for index in range(64):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(
                row_id=f"real-{index}", proposal=proposal,
                next_actor_observation=proposal.actor_observation,
                reward_step=_reward(contact=index == 63),
            )
        )
    assert coordinator.accepted_transitions == 64
    assert coordinator.sac_update_count == 1
    assert coordinator.replay.priorities[63] > coordinator.replay.priorities[0]
    metrics = coordinator.metrics()
    assert metrics["FINAL_ACTION_BOUND_VIOLATION"] == 0
    assert metrics["GRIPPER_AUTHORITY_VIOLATION"] == 0
    assert metrics["BC_WEIGHTS_CHANGED"] is False
    assert all(np.isfinite(value) for key, value in metrics.items() if key.startswith("optimizer/"))
    receipt = coordinator.export_checkpoints(tmp_path / "checkpoints")
    assert receipt.actor_reload_pass and receipt.learner_reload_pass


def test_periodic_checkpoint_is_full_resumable_and_boundary_locked(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    with pytest.raises(Stage1ARealSACError, match="3000-transition"):
        coordinator.export_periodic_training_checkpoint(
            tmp_path / "checkpoint_0.pt",
            source_freeze={"source": "frozen"},
            reward_config={"reward": "v3"},
            runtime_config={"stable_only": True},
        )
    coordinator.accepted_transitions = 3000
    receipt = coordinator.export_periodic_training_checkpoint(
        tmp_path / "checkpoint_3000.pt",
        source_freeze={"source": "frozen"},
        reward_config={"reward": "v3"},
        runtime_config={"stable_only": True},
    )
    assert receipt.reload_pass
    payload = torch.load(receipt.path, map_location="cpu", weights_only=False)
    assert payload["accepted_transitions"] == 3000
    assert payload["contents"] == {
        "actor": True,
        "critic": True,
        "alpha": True,
        "optimizer_state": True,
        "replay_buffer": True,
        "rng_state": True,
    }
    assert payload["replay_strategy"] == REPLAY_STRATEGY_HER_FORCE


def test_bounded_final_checkpoint_explicitly_allows_non_3k_boundary(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    coordinator.accepted_transitions = 7500
    with pytest.raises(Stage1ARealSACError, match="3000-transition"):
        coordinator.export_periodic_training_checkpoint(
            tmp_path / "checkpoint_7500_rejected.pt",
            source_freeze={"source": "frozen"},
            reward_config={"reward": "v3"},
            runtime_config={"stable_only": True},
        )
    receipt = coordinator.export_periodic_training_checkpoint(
        tmp_path / "checkpoint_7500.pt",
        source_freeze={"source": "frozen"},
        reward_config={"reward": "v3"},
        runtime_config={"stable_only": True},
        allow_bounded_final_boundary=True,
    )
    assert receipt.reload_pass
    payload = torch.load(receipt.path, map_location="cpu", weights_only=False)
    assert payload["accepted_transitions"] == 7500
    assert payload["checkpoint_boundary_kind"] == "BOUNDED_FINAL"


def test_live_one_row_gru_continuation_preserves_hidden_state(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    first = coordinator.propose(_inputs(), deterministic=True)
    carried = coordinator.hidden.detach().clone()
    continuation = _inputs()
    continuation = HumanGraspSequenceInputs(
        **{
            **continuation.__dict__,
            "hidden_reset_mask": torch.zeros(1, 1, dtype=torch.bool),
        }
    )
    second = coordinator.propose(continuation, deterministic=True)
    assert first.actor_observation.shape == second.actor_observation.shape == (128,)
    assert torch.isfinite(coordinator.hidden).all()
    assert not torch.equal(carried, coordinator.hidden)


def test_unconfirmed_nonreal_and_duplicate_rows_fail_before_replay(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    base = dict(
        row_id="row", proposal=proposal,
        next_actor_observation=proposal.actor_observation,
        reward_step=_reward(),
    )
    with pytest.raises(Stage1ARealSACError, match="consumption"):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(**base, canonical_consumption_confirmed=False)
        )
    with pytest.raises(Stage1ARealSACError, match="synthetic"):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(**base, real_telemetry=False)
        )
    assert len(coordinator.replay) == 0
    coordinator.accept_real_transition(Stage1AAcceptedRealRow(**base))
    with pytest.raises(Stage1ARealSACError, match="unique"):
        coordinator.accept_real_transition(Stage1AAcceptedRealRow(**base))
    assert len(coordinator.replay) == 1


def test_bc_file_and_parameters_remain_frozen(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    coordinator.bc_checkpoint_path.write_bytes(b"changed")
    with pytest.raises(Stage1ARealSACError, match="file changed"):
        coordinator.propose(_inputs(), deterministic=True)


def _hardstop_reward() -> Stage1ARewardStep:
    # Preserve the already-earned bilateral milestone while applying a
    # source-owned negative safety sample and failing terminally.
    return replace(
        _reward(contact=True, stable=False),
        reward_total=torch.tensor([0.95]),
        done=torch.tensor([True]),
        success=torch.tensor([False]),
        safety_penalty=torch.tensor([-0.05]),
    )


def test_candidate_a_gamma_matches_requested_pbrs_authority(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    assert coordinator.agent.config.gamma == pytest.approx(STAGE1A_SAC_GAMMA)
    assert STAGE1A_SAC_GAMMA == pytest.approx(0.9993)


def test_hardstop_transition_is_failed_terminal_and_preserves_milestone(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    coordinator.accept_real_transition(
        Stage1AAcceptedRealRow(
            row_id="hardstop-1",
            proposal=proposal,
            next_actor_observation=proposal.actor_observation,
            reward_step=_hardstop_reward(),
            source=REPLAY_SOURCE_ONLINE_SAC,
            episode_id="episode-hardstop",
            success=False,
            failure_reason=RUNTIME_HARDSTOP_FAILURE,
            hardstop=True,
            contact=True,
            bilateral=True,
            stable=False,
            safety_pass=False,
            terminated=True,
            action_consumption="PARTIAL_TERMINATED",
            physics_step_end=5,
            consumed_substeps=5,
            execution_fraction=0.5,
            terminal_timestamp=0.01,
            camera_age=0.01,
            hardstop_joint="idx83_gripper_r_outer_joint4",
            idx83_min_margin=0.0,
        )
    )
    assert len(coordinator.replay) == 1
    assert coordinator.replay.terminated[0, 0] == pytest.approx(1.0)
    assert coordinator.replay.rewards[0, 0] == pytest.approx(0.95)
    assert coordinator.metrics()["BILATERAL_REWARD_SUM"] == pytest.approx(1.0)
    receipt = coordinator.replay_provenance[0]
    assert receipt["success"] is False
    assert receipt["terminated"] is True
    assert receipt["hardstop"] is True
    assert receipt["failure_reason"] == RUNTIME_HARDSTOP_FAILURE


@pytest.mark.parametrize(
    "overrides,pattern",
    (
        ({"success": True}, "success differs"),
        ({"terminated": False}, "terminated flag differs"),
        ({"safety_pass": True}, "failed terminal"),
        ({"failure_reason": "NONE"}, "failed terminal"),
    ),
)
def test_malformed_hardstop_rows_fail_closed(
    tmp_path: Path, overrides: dict[str, object], pattern: str
) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    fields = dict(
        row_id="hardstop-invalid",
        proposal=proposal,
        next_actor_observation=proposal.actor_observation,
        reward_step=_hardstop_reward(),
        episode_id="episode-hardstop",
        success=False,
        failure_reason=RUNTIME_HARDSTOP_FAILURE,
        hardstop=True,
        contact=True,
        bilateral=True,
        stable=False,
        safety_pass=False,
        terminated=True,
    )
    fields.update(overrides)
    with pytest.raises(Stage1ARealSACError, match=pattern):
        coordinator.accept_real_transition(Stage1AAcceptedRealRow(**fields))
    assert len(coordinator.replay) == 0


def test_safety_failure_cannot_be_her_success(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    with pytest.raises(Stage1ARealSACError, match="safety failure"):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(
                row_id="unsafe-success",
                proposal=proposal,
                next_actor_observation=proposal.actor_observation,
                reward_step=_reward(stable=True),
                episode_id="episode-unsafe-success",
                success=True,
                safety_pass=False,
                terminated=True,
            )
        )


def test_candidate_a_provenance_is_stored_and_e1_is_rejected(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    with pytest.raises(Stage1ARealSACError, match="E1/non-Candidate-A"):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(
                row_id="e1",
                proposal=proposal,
                next_actor_observation=proposal.actor_observation,
                reward_step=_reward(),
                source=REPLAY_SOURCE_EXISTING_CLOSE,
                episode_id="legacy-e1",
                dynamics_authority="E1_DIAGNOSTIC",
            )
        )
    coordinator.accept_real_transition(
        Stage1AAcceptedRealRow(
            row_id="candidate-a",
            proposal=proposal,
            next_actor_observation=proposal.actor_observation,
            reward_step=_reward(),
            episode_id="candidate-a-online",
            dynamics_authority=CANDIDATE_A_DYNAMICS_AUTHORITY,
        )
    )
    assert coordinator.replay_provenance[0]["dynamics_authority"] == "CANDIDATE_A"


@pytest.mark.parametrize(
    "overrides",
    (
        {"transition_schema_complete": False},
        {"policy_observation_complete": False},
        {"mechanics_only_telemetry": True},
    ),
)
def test_mechanics_only_or_incomplete_close_artifact_cannot_enter_replay(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    coordinator = _coordinator(tmp_path)
    proposal = coordinator.propose(_inputs(), deterministic=True)
    fields = dict(
        row_id="incomplete-close",
        proposal=proposal,
        next_actor_observation=proposal.actor_observation,
        reward_step=_reward(),
        source=REPLAY_SOURCE_EXISTING_CLOSE,
        episode_id="existing-close",
    )
    fields.update(overrides)
    with pytest.raises(Stage1ARealSACError, match="mechanics-only/incomplete"):
        coordinator.accept_real_transition(Stage1AAcceptedRealRow(**fields))
    assert len(coordinator.replay) == 0


def test_privileged_close_readiness_is_opt_in_teacher_only_student_supervision(
    tmp_path: Path,
) -> None:
    """Teacher labels stay out of actor input and train only the opt-in head."""

    blocked = _coordinator(tmp_path)
    blocked_proposal = blocked.propose(_inputs(), deterministic=True)
    with pytest.raises(Stage1ARealSACError, match="opt-in distillation"):
        blocked.accept_real_transition(
            Stage1AAcceptedRealRow(
                row_id="teacher-label-without-opt-in",
                proposal=blocked_proposal,
                next_actor_observation=blocked_proposal.actor_observation,
                reward_step=_reward(),
                pre_close_candidate=True,
                close_latched_before_supervision=False,
                privileged_close_ready_target=True,
                privileged_close_ready_score=1.0,
            )
        )
    assert len(blocked.replay) == 0

    checkpoint = tmp_path / "frozen_bc_distillation.pt"
    checkpoint.write_bytes(b"immutable BC provenance fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    coordinator = Stage1ARealSACCoordinator(
        frozen_bc=HumanGraspGRUBC(),
        bc_checkpoint_path=checkpoint,
        expected_bc_sha256=digest,
        replay_capacity=128,
        seed=19,
        close_readiness_distillation=True,
    )
    proposal = coordinator.propose(_inputs(), deterministic=True)
    for index in range(64):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(
                row_id=f"teacher-row-{index}",
                proposal=proposal,
                next_actor_observation=proposal.actor_observation,
                reward_step=_reward(),
                episode_id=f"teacher-episode-{index}",
                pre_close_candidate=True,
                close_latched_before_supervision=False,
                privileged_close_ready_target=bool(index % 2),
                privileged_close_ready_score=1.0 if index % 2 else 0.0,
            )
        )
    metrics = coordinator.metrics()
    assert metrics["CLOSE_DISTILLATION_ENABLED"] is True
    assert metrics["CLOSE_DISTILLATION_ROWS"] == 64
    assert metrics["CLOSE_DISTILLATION_UPDATE_COUNT"] == 1
    assert metrics["STUDENT_FALSE_REJECT_RATE"] >= 0.0
    assert metrics["STUDENT_FALSE_ACCEPT_RATE"] >= 0.0
    assert coordinator.replay_provenance[-1]["student_privileged_input_count"] == 0


@pytest.mark.parametrize(
    ("pre_close_candidate", "close_latched_before_supervision", "score", "message"),
    (
        (False, False, 0.0, "post-CLOSE/non-candidate"),
        (True, True, 0.0, "post-CLOSE/non-candidate"),
        (True, False, 0.75, "binary admission semantics"),
    ),
)
def test_close_readiness_rejects_postclose_or_predicate_fraction_labels(
    tmp_path: Path,
    pre_close_candidate: bool,
    close_latched_before_supervision: bool,
    score: float,
    message: str,
) -> None:
    checkpoint = tmp_path / "frozen_bc_preclose_contract.pt"
    checkpoint.write_bytes(b"immutable BC provenance fixture")
    coordinator = Stage1ARealSACCoordinator(
        frozen_bc=HumanGraspGRUBC(),
        bc_checkpoint_path=checkpoint,
        expected_bc_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        replay_capacity=32,
        seed=27,
        close_readiness_distillation=True,
    )
    proposal = coordinator.propose(_inputs(), deterministic=True)
    with pytest.raises(Stage1ARealSACError, match=message):
        coordinator.accept_real_transition(
            Stage1AAcceptedRealRow(
                row_id=f"invalid-preclose-{pre_close_candidate}-{close_latched_before_supervision}-{score}",
                proposal=proposal,
                next_actor_observation=proposal.actor_observation,
                reward_step=_reward(),
                pre_close_candidate=pre_close_candidate,
                close_latched_before_supervision=close_latched_before_supervision,
                privileged_close_ready_target=False,
                privileged_close_ready_score=score,
            )
        )
    assert len(coordinator.replay) == 0
