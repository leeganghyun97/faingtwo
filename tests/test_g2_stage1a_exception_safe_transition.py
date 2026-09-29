# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Regression closure for exception-safe Candidate-A hard-stop replay."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geniesim.rl.sac.human_grasp_gru_bc import (
    HumanGraspGRUBC,
    HumanGraspSequenceInputs,
)
from geniesim.rl.sac.stage1a_exception_safe_transition import (
    ActionConsumptionState,
    RuntimeHardstop,
    RuntimeHardstopReceipt,
    elapsed_discount,
)
from geniesim.rl.sac.stage1a_grasp_reward import (
    Stage1AGraspReward,
    Stage1AGraspRewardConfig,
    Stage1ARewardInputs,
    stage1a_reward_contract,
)
from geniesim.rl.sac.stage1a_real_sac_coordinator import (
    Stage1AAcceptedRealRow,
    Stage1ARealSACCoordinator,
    Stage1ARealSACError,
)
from geniesim.rl.sac.stage2_sac import soft_bellman_target


ROOT = Path(__file__).resolve().parents[1]


def _preflight_module():
    path = ROOT / "scripts/diagnostics/run_g2_policy_4d_training_runtime_preflight.py"
    spec = importlib.util.spec_from_file_location("g2_exception_safe_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt(k: int) -> RuntimeHardstopReceipt:
    qdd = [0.0, 12.0]
    return RuntimeHardstopReceipt(
        physics_step_index=k,
        physics_substep_index=k - 1,
        control_step_index=7,
        consumed_substeps=k,
        nominal_substeps=10,
        terminal_physics_timestamp_s=1.0 + 0.002 * k,
        q_rad=(0.1, 0.2),
        qd_rad_s=(0.0, 0.1),
        qdd_rad_s2=tuple(qdd),
        idx83_limit_margin_rad=0.0,
        hardstop_joint="idx83_gripper_r_outer_joint4",
        hardstop_joint_index=1,
        hardstop_reason="RUNTIME_HARDSTOP",
        contact=True,
        bilateral=False,
        stable=False,
        last_real_sensor_timestamps={"right_wrist": 1.0},
    )


class _Envelope:
    packet_id = "packet-1"
    content_fingerprint = "fingerprint"

    @staticmethod
    def as_env_step_tensor():
        return torch.zeros(1, 8)


class _Packet:
    values = (0.0,) * 8


class _Port:
    def __init__(self) -> None:
        self.partial = None
        self.unknown = False

    def defer_normalized_action_manager_packet(self, _packet):
        return _Envelope()

    def claim_for_env_step(self, _packet_id):
        return _Envelope()

    def acknowledge_partial_termination(self, _packet_id, **fields):
        self.partial = fields
        return SimpleNamespace(
            state=SimpleNamespace(value="PARTIAL_TERMINATED"),
            action_consumption="PARTIAL_TERMINATED",
            **fields,
        )

    def mark_env_step_consumption_unknown(self, _packet_id):
        self.unknown = True


class _Counter:
    context = ""

    def __init__(self) -> None:
        self.env = 0
        self.process = 0

    def checkpoint(self):
        return (self.env, self.process)

    def interval(self, before):
        return {
            "env_step_calls": self.env - before[0],
            "action_manager_process_action_calls": self.process - before[1],
            "env_step_packets": [{"sha256": "x", "values": [0.0] * 8}],
            "process_action_packets": [{"sha256": "x", "values": [0.0] * 8}],
        }


class _HardstopEnv:
    def __init__(self, k: int, counter: _Counter) -> None:
        self.k = k
        self.counter = counter
        self.physics_steps = 0

    def step(self, _action):
        self.counter.env += 1
        self.counter.process += 1
        for _ in range(self.k):
            self.physics_steps += 1
        raise RuntimeHardstop(_receipt(self.k))


@pytest.mark.parametrize("k", (1, 5, 10))
def test_structured_hardstop_is_partial_terminated_at_exact_substep(k: int) -> None:
    preflight = _preflight_module()
    counter = _Counter()
    port = _Port()
    env = _HardstopEnv(k, counter)
    with pytest.raises(RuntimeHardstop) as captured:
        preflight._consume_once(
            env=env,
            counter=counter,
            deferred_port=port,
            packet=_Packet(),
            label=f"hardstop-{k}",
        )
    assert env.physics_steps == k
    assert port.unknown is False
    assert port.partial == {
        "consumed_substeps": k,
        "nominal_substeps": 10,
        "termination_reason": "RUNTIME_HARDSTOP",
    }
    canonical = captured.value.canonical_consumption_receipt
    assert canonical["action_consumption"] == "PARTIAL_TERMINATED"
    assert canonical["consumed_substeps"] == k
    assert canonical["action_manager_process_action_count"] == 1


def _reward_inputs(k: int, *, contact: bool = False) -> Stage1ARewardInputs:
    valid = torch.zeros(1, 2, k, dtype=torch.bool)
    impulse = torch.zeros(1, 2, k)
    points = torch.zeros(1, 2, k, 3)
    normals = torch.zeros_like(points)
    if contact:
        valid[:, :, : min(k, 3)] = True
        # 1.2-N effective force over the actual k*2-ms interval.
        impulse[:, :, : min(k, 3)] = 1.2 * (k * 0.002) / min(k, 3)
        points[:, 0, :, 0] = -0.02
        points[:, 1, :, 0] = 0.02
        normals[:, 0, :, 0] = 1.0
        normals[:, 1, :, 0] = -1.0
    return Stage1ARewardInputs(
        current_ee_position_root_m=torch.tensor([[-0.020, 0.0, 0.0]]),
        next_ee_position_root_m=torch.tensor([[-0.019, 0.0, 0.0]]),
        nominal_grasp_position_root_m=torch.zeros(1, 3),
        nominal_approach_axis_root=torch.tensor([[1.0, 0.0, 0.0]]),
        stage1a_active=torch.tensor([True]),
        grasp_decision_phase=torch.tensor([True]),
        phase_reset=torch.tensor([False]),
        episode_reset=torch.tensor([False]),
        pad_cube_contact_valid=valid,
        normal_impulse_ns=impulse,
        normal_impulse_available=torch.ones(1, 2, dtype=torch.bool),
        normal_force_n=torch.zeros(1, 2, k),
        contact_geometry_valid=valid.clone(),
        contact_point_root_m=points,
        contact_normal_root=normals,
        tangential_relative_speed_m_s=torch.zeros(1, 2, k),
        normal_relative_speed_m_s=torch.zeros(1, 2, k),
        cube_center_root_m=torch.zeros(1, 3),
        cube_quat_root_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        cube_half_extents_m=torch.tensor([[0.02, 0.02, 0.02]]),
        cube_linear_velocity_root_m_s=torch.zeros(1, 3),
        cube_angular_velocity_root_rad_s=torch.zeros(1, 3),
        effective_residual_root_m=torch.zeros(1, 3),
        previous_effective_residual_root_m=torch.zeros(1, 3),
        final_action_xyz_root_m=torch.zeros(1, 3),
        contract_grasp_state_valid=torch.tensor([False]),
        safety_violation=torch.tensor([True]),
        authoritative_safety_penalty=torch.tensor([0.0]),
        non_pad_gripper_cube_contact=torch.tensor([False]),
        close_command=torch.tensor([True]),
        consumed_substeps=torch.tensor([k]),
        runtime_hardstop=torch.tensor([True]),
    )


def test_partial_reward_uses_actual_dt_partial_gamma_and_unscaled_milestone() -> None:
    config = Stage1AGraspRewardConfig()
    step = Stage1AGraspReward(1, config=config).step(
        _reward_inputs(5, contact=True)
    )
    assert step.elapsed_time_s.item() == pytest.approx(0.010)
    assert step.gamma_effective.item() == pytest.approx(
        elapsed_discount(config.gamma, 5)
    )
    assert step.time_penalty.item() == pytest.approx(
        config.time_penalty_per_active_step * 0.5
    )
    # Bilateral is an event, not a rate; it is not multiplied by 0.5.
    assert step.reward_bilateral.item() >= config.bilateral_base_reward
    assert step.safety_penalty.item() == pytest.approx(-0.90)
    assert step.done.item() and not step.success.item()


def test_hardstop_penalty_is_source_owned_and_fingerprinted() -> None:
    contract = stage1a_reward_contract()
    assert contract["runtime_hardstop_terminal_penalty"] == pytest.approx(-0.90)
    assert contract["runtime_hardstop_penalty_resolved"] == pytest.approx(-0.90)
    assert contract["runtime_hardstop_penalty_formula"] == (
        "-(single_contact_reward + bilateral_base_reward)"
    )
    assert len(contract["reward_authority_sha256"]) == 64


def test_terminal_bellman_target_has_zero_bootstrap() -> None:
    reward = torch.tensor([[1.25]])
    target = soft_bellman_target(
        reward, torch.ones_like(reward), 0.9993, torch.tensor([[999.0]])
    )
    assert target.item() == pytest.approx(1.25)


def _coordinator(tmp_path: Path) -> Stage1ARealSACCoordinator:
    checkpoint = tmp_path / "bc.pt"
    checkpoint.write_bytes(b"immutable")
    return Stage1ARealSACCoordinator(
        frozen_bc=HumanGraspGRUBC(),
        bc_checkpoint_path=checkpoint,
        expected_bc_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        replay_capacity=8,
    )


def _hardstop_row(coordinator: Stage1ARealSACCoordinator):
    inputs = HumanGraspSequenceInputs(
        right_wrist_rgb=torch.zeros(1, 1, 48, 64, 3),
        right_wrist_depth_m=torch.ones(1, 1, 48, 64, 1),
        right_wrist_depth_valid=torch.ones(
            1, 1, 48, 64, 1, dtype=torch.bool
        ),
        ee_pose_robot_root_m_xyzw=torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]
        ),
        right_arm_joint_position_rad=torch.zeros(1, 1, 7),
        right_arm_joint_velocity_rad_s=torch.zeros(1, 1, 7),
        current_gripper_state=torch.zeros(1, 1, 1),
        previous_policy_action_4d_metric_root_m=torch.zeros(1, 1, 4),
        hidden_reset_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    proposal = coordinator.propose(inputs, deterministic=True)
    reward_step = Stage1AGraspReward(1).step(
        _reward_inputs(5, contact=True)
    )
    return Stage1AAcceptedRealRow(
        row_id="hardstop",
        proposal=proposal,
        next_actor_observation=proposal.actor_observation.copy(),
        reward_step=reward_step,
        success=False,
        failure_reason="RUNTIME_HARDSTOP",
        hardstop=True,
        contact=True,
        bilateral=True,
        stable=False,
        safety_pass=False,
        terminated=True,
        action_consumption=ActionConsumptionState.PARTIAL_TERMINATED.value,
        physics_step_end=5,
        consumed_substeps=5,
        execution_fraction=0.5,
        terminal_timestamp=0.01,
        camera_age=0.01,
        hardstop_joint="idx83_gripper_r_outer_joint4",
        idx83_min_margin=0.0,
    )


def test_unknown_and_synthetic_terminal_rows_cannot_enter_replay(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    row = _hardstop_row(coordinator)
    with pytest.raises(Stage1ARealSACError, match="unfinished"):
        coordinator.accept_real_transition(
            replace(row, action_consumption="CONSUMPTION_UNKNOWN")
        )
    with pytest.raises(Stage1ARealSACError, match="synthetic"):
        coordinator.accept_real_transition(
            replace(row, row_id="synthetic", terminal_observation_is_real=False)
        )
    assert len(coordinator.replay) == 0


def test_real_partial_terminal_metadata_and_her_safety_are_preserved(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    row = _hardstop_row(coordinator)
    coordinator.accept_real_transition(row)
    assert coordinator.replay.terminated[0, 0] == 1.0
    assert coordinator.replay.truncated[0, 0] == 0.0
    assert coordinator.replay_provenance[0]["success"] is False
    assert coordinator.replay_provenance[0]["terminated"] is True
    assert coordinator.replay_provenance[0]["truncated"] is False
    assert coordinator.replay_provenance[0]["terminal_observation_is_real"] is True
    assert coordinator.metrics()["PARTIAL_TERMINAL_TRANSITIONS"] == 1
    assert coordinator.metrics()["REPLAY_HARDSTOP_COUNT"] == 1


def test_500hz_hardstop_predicate_literals_are_unchanged() -> None:
    source = (ROOT / "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py").read_text()
    assert "HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD = 1.0e-5" in source
    assert "HARD_STOP_ACCELERATION_LIMIT_RAD_S2 = 10.0" in source
    assert "(limit_margin <= HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD)" in source
    assert "(np.abs(qdd) > HARD_STOP_ACCELERATION_LIMIT_RAD_S2)" in source
