# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import pytest
import torch
import numpy as np

from geniesim.rl.isaaclab.g2_policy_branch.action_interface import AbstractGripperIntent
from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
    HighLevelPolicyAction,
    expand_to_existing_controller_8d,
)
from geniesim.rl.isaaclab.g2_policy_branch.contact_free_training_contract import (
    metric_xyz_to_normalized,
)
from geniesim.rl.sac.stage1a_vector_contract import (
    ImmutablePerEnvActionPacket,
    PerEnvCameraCache,
    PerEnvCameraFrame,
    PerEnvCoordinatorStateAdapter,
    PerEnvCanonicalAction,
    PerEnvPacketPort,
    PerEnvReplayIdentity,
    PerEnvStateRegistry,
    Stage1AVectorContractError,
)
from geniesim.rl.sac.stage1a_vector_runtime import (
    Stage1AVectorRuntimeError,
    consume_per_env_packet_once,
    stack_single_env_reward_inputs,
)
from geniesim.rl.sac.stage1a_grasp_reward import Stage1ARewardInputs


def _row(env_id: int, dx: float, *, close: bool = False) -> PerEnvCanonicalAction:
    return PerEnvCanonicalAction(
        env_id=env_id,
        final_action_4d_metric_root_m=(dx, 0.0, 0.0, 1.0 if close else 0.0),
        gripper_intent=(AbstractGripperIntent.CLOSE if close else AbstractGripperIntent.OPEN),
    )


def test_per_env_packet_materializes_distinct_rows_without_broadcast() -> None:
    packet = ImmutablePerEnvActionPacket(
        rows=(_row(0, 0.001), _row(1, -0.002, close=True)),
        binding_id="test-vector",
        sequence_id=1,
        device="cpu",
    )
    tensor = packet.as_env_step_tensor()
    assert tuple(tensor.shape) == (2, 8)
    assert packet.unique_4d_rows == 2
    assert packet.unique_8d_rows == 2
    assert packet.broadcast_detected is False
    assert not torch.equal(tensor[0], tensor[1])
    assert tensor[0, 7].item() == 1.0
    assert tensor[1, 7].item() == -1.0


def test_num_envs_one_packet_has_scalar_p0_action_parity() -> None:
    action = _row(0, 0.001, close=True)
    packet = ImmutablePerEnvActionPacket(
        rows=(action,), binding_id="test-vector", sequence_id=1, device="cpu"
    )
    expected = expand_to_existing_controller_8d(
        HighLevelPolicyAction.from_sequence(
            (*metric_xyz_to_normalized((0.001, 0.0, 0.0)), 1.0)
        ),
        gripper_intent=AbstractGripperIntent.CLOSE,
    )
    assert tuple(packet.as_env_step_tensor()[0].tolist()) == pytest.approx(expected)


def test_packet_treats_equal_independently_constructed_actions_as_nonbroadcast() -> None:
    repeated = ImmutablePerEnvActionPacket(
        rows=(_row(0, 0.001), _row(1, 0.001)),
        binding_id="test-vector",
        sequence_id=1,
        device="cpu",
    )
    # Equal values are not a scalar broadcast when each immutable row was
    # independently built for a different environment.  The vector packet has
    # no repeat/expand materialization API.
    assert repeated.broadcast_detected is False
    with pytest.raises(Stage1AVectorContractError, match="ordered env_id"):
        ImmutablePerEnvActionPacket(
            rows=(_row(1, 0.001), _row(0, -0.001)),
            binding_id="test-vector",
            sequence_id=1,
            device="cpu",
        )


def test_packet_port_enforces_one_batched_claim_and_acknowledgement() -> None:
    port = PerEnvPacketPort(batch_size=2, device="cpu", binding_id="test-vector")
    packet = port.stage((_row(0, 0.001), _row(1, -0.001)))
    assert port.claim(packet.packet_id) is packet
    assert port.acknowledge(packet.packet_id) is packet
    assert port.outstanding_packet is None
    assert (port.stage_count, port.claim_count, port.ack_count) == (1, 2, 1)


def test_reset_isolated_state_preserves_other_environments() -> None:
    registry = PerEnvStateRegistry(5)
    state0 = registry.state(0)
    state3 = registry.state(3)
    state0.close_latched = True
    state0.camera_frame_id = 11
    state3.close_latched = True
    state3.camera_frame_id = 99
    state3.gru_hidden = torch.ones((2, 1, 4))
    registry.reset(3)
    assert registry.state(0).close_latched is True
    assert registry.state(0).camera_frame_id == 11
    assert registry.state(3).episode_index == 1
    assert registry.state(3).close_latched is False
    assert registry.state(3).camera_frame_id is None
    assert registry.state(3).gru_hidden is None


def test_replay_identity_has_environment_scoped_unique_row_id() -> None:
    left = PerEnvReplayIdentity(env_id=0, episode_id="episode-0001", step_in_episode=7)
    right = PerEnvReplayIdentity(env_id=1, episode_id="episode-0001", step_in_episode=7)
    assert left.row_id != right.row_id
    assert left.row_id.startswith("env-00:")


class _FakeSharedCoordinator:
    def __init__(self) -> None:
        self.hidden = None
        self._closed = False
        self._episode_milestones = {}
        self._current_v3_hover_steps = 0
        self._current_max_residual_streak = 0

    def mutate(self, value: int) -> int:
        self.hidden = torch.full((1, 1, 2), float(value))
        self._closed = bool(value % 2)
        self._episode_milestones = {"stable": value}
        self._current_v3_hover_steps = value
        self._current_max_residual_streak = value + 1
        return value

    def reset_episode(self) -> None:
        self.hidden = None
        self._closed = False
        self._episode_milestones = {}
        self._current_v3_hover_steps = 0
        self._current_max_residual_streak = 0


def test_shared_learner_state_adapter_prevents_cross_env_hidden_or_latch_leakage() -> None:
    core = _FakeSharedCoordinator()
    vector = PerEnvCoordinatorStateAdapter(core, num_envs=3)
    assert vector.call(0, "mutate", 3) == 3
    assert vector.call(1, "mutate", 8) == 8
    before_env0 = vector.state_snapshot(0)
    assert before_env0.closed is True
    assert before_env0.hidden is not None and before_env0.hidden[0, 0, 0].item() == 3.0
    vector.reset(1)
    after_env0 = vector.state_snapshot(0)
    after_env1 = vector.state_snapshot(1)
    assert after_env0.closed is True
    assert after_env0.hidden is not None and after_env0.hidden[0, 0, 0].item() == 3.0
    assert after_env1.closed is False
    assert after_env1.hidden is None


def test_camera_cache_is_environment_scoped_and_reset_isolated() -> None:
    cache = PerEnvCameraCache(2)
    frame0 = PerEnvCameraFrame(
        env_id=0, frame_id=7, timestamp_s=0.28,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    frame1 = PerEnvCameraFrame(
        env_id=1, frame_id=99, timestamp_s=3.96,
        rgb=np.full((2, 3, 3), 17, dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    cache.put(frame0)
    cache.put(frame1)
    cache.reset(1)
    assert cache.get(0) is frame0
    assert cache.get(1) is None


def test_camera_cache_reset_permits_a_new_episode_timestamp_epoch() -> None:
    """A clone-local episode reset must not poison another clone's cache."""

    cache = PerEnvCameraCache(2)
    retained = PerEnvCameraFrame(
        env_id=0, frame_id=40, timestamp_s=1.60,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    old_episode = PerEnvCameraFrame(
        env_id=1, frame_id=90, timestamp_s=3.60,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    cache.put(retained)
    cache.put(old_episode)
    cache.reset(1)
    new_episode = PerEnvCameraFrame(
        env_id=1, frame_id=0, timestamp_s=0.0,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    cache.put(new_episode)
    assert cache.get(0) is retained
    assert cache.get(1) is new_episode


def test_camera_cache_permits_one_delayed_sensor_epoch_during_reset_gate() -> None:
    """A sensor may restart one physics tick after its Python cache reset."""

    cache = PerEnvCameraCache(1)
    cache.reset(0)
    pre_restart = PerEnvCameraFrame(
        env_id=0, frame_id=91, timestamp_s=3.64,
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    post_restart = PerEnvCameraFrame(
        env_id=0, frame_id=0, timestamp_s=0.0,
        rgb=np.ones((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3, 1), dtype=np.float32),
        depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    cache.put(pre_restart)
    cache.put(post_restart)
    assert cache.get(0) is post_restart
    assert cache.epoch_transition_count(0) == 1
    assert cache.reset_epoch_transition_allowed(0) is False


def test_camera_cache_rejects_epoch_regression_after_reset_gate_is_sealed() -> None:
    cache = PerEnvCameraCache(1)
    cache.reset(0)
    cache.put(
        PerEnvCameraFrame(
            env_id=0, frame_id=12, timestamp_s=0.48,
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_m=np.ones((2, 3, 1), dtype=np.float32),
            depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
        )
    )
    cache.seal_reset_epoch(0)
    with pytest.raises(Stage1AVectorContractError, match="regressed"):
        cache.put(
            PerEnvCameraFrame(
                env_id=0, frame_id=0, timestamp_s=0.0,
                rgb=np.zeros((2, 3, 3), dtype=np.uint8),
                depth_m=np.ones((2, 3, 1), dtype=np.float32),
                depth_valid=np.ones((2, 3, 1), dtype=np.bool_),
            )
        )


class _FakeManager:
    def __init__(self, counter) -> None:
        self.counter = counter

    def process_action(self, action: torch.Tensor) -> None:
        self.counter.process_records.append(self.counter.record(action))


class _FakeCounter:
    def __init__(self) -> None:
        self.context = ""
        self.env_step_records = []
        self.process_records = []

    @staticmethod
    def record(action: torch.Tensor) -> dict[str, object]:
        tensor = action.detach().to("cpu").contiguous()
        import hashlib
        return {
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }

    def checkpoint(self):
        return len(self.env_step_records), len(self.process_records)

    def interval(self, before):
        env_start, process_start = before
        return {
            "env_step_calls": len(self.env_step_records) - env_start,
            "action_manager_process_action_calls": len(self.process_records) - process_start,
            "env_step_packets": self.env_step_records[env_start:],
            "process_action_packets": self.process_records[process_start:],
        }


class _FakeEnv:
    def __init__(self, counter: _FakeCounter, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.counter = counter
        self.action_manager = _FakeManager(counter)

    def step(self, action: torch.Tensor):
        self.counter.env_step_records.append(self.counter.record(action))
        self.action_manager.process_action(action)
        return "stepped"


def test_vector_consumer_uses_one_env_step_and_one_manager_call_for_distinct_rows() -> None:
    counter = _FakeCounter()
    env = _FakeEnv(counter)
    port = PerEnvPacketPort(batch_size=2, device="cpu", binding_id="vector-test")
    packet = port.stage((_row(0, 0.001), _row(1, -0.002, close=True)))
    result, receipt = consume_per_env_packet_once(
        env=env, counter=counter, port=port, packet=packet, label="TEST",
    )
    assert result == "stepped"
    assert receipt.single_batched_consumption is True
    assert receipt.action_broadcast_detected is False
    assert receipt.unique_4d_action_rows == receipt.unique_8d_packet_rows == 2
    assert port.outstanding_packet is None


def test_vector_consumer_accepts_equal_but_independent_hold_rows() -> None:
    counter = _FakeCounter()
    env = _FakeEnv(counter)
    port = PerEnvPacketPort(batch_size=2, device="cpu", binding_id="vector-test")
    packet = port.stage((_row(0, 0.001), _row(1, 0.001)))
    _result, receipt = consume_per_env_packet_once(
        env=env, counter=counter, port=port, packet=packet, label="TEST",
    )
    assert receipt.action_broadcast_detected is False
    assert len(counter.env_step_records) == 1


def test_reward_inputs_are_stacked_in_env_order_without_recomputing_fields() -> None:
    def row(value: float) -> Stage1ARewardInputs:
        return Stage1ARewardInputs(
            current_ee_position_root_m=torch.full((1, 3), value),
            next_ee_position_root_m=torch.full((1, 3), value + 1),
            nominal_grasp_position_root_m=torch.full((1, 3), value + 2),
            nominal_approach_axis_root=torch.tensor([[1.0, 0.0, 0.0]]),
            stage1a_active=torch.ones(1, dtype=torch.bool),
            grasp_decision_phase=torch.ones(1, dtype=torch.bool),
            phase_reset=torch.zeros(1, dtype=torch.bool),
            episode_reset=torch.zeros(1, dtype=torch.bool),
            pad_cube_contact_valid=torch.ones((1, 2, 10), dtype=torch.bool),
            normal_impulse_ns=torch.zeros((1, 2, 10)),
            normal_impulse_available=torch.ones((1, 2), dtype=torch.bool),
            normal_force_n=torch.zeros((1, 2, 10)),
            contact_geometry_valid=torch.ones((1, 2, 10), dtype=torch.bool),
            contact_point_root_m=torch.zeros((1, 2, 10, 3)),
            contact_normal_root=torch.zeros((1, 2, 10, 3)),
            tangential_relative_speed_m_s=torch.zeros((1, 2, 10)),
            normal_relative_speed_m_s=torch.zeros((1, 2, 10)),
            cube_center_root_m=torch.zeros((1, 3)),
            cube_quat_root_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            cube_half_extents_m=torch.full((1, 3), 0.02),
            cube_linear_velocity_root_m_s=torch.zeros((1, 3)),
            cube_angular_velocity_root_rad_s=torch.zeros((1, 3)),
            effective_residual_root_m=torch.zeros((1, 3)),
            previous_effective_residual_root_m=torch.zeros((1, 3)),
            final_action_xyz_root_m=torch.zeros((1, 3)),
            contract_grasp_state_valid=torch.ones(1, dtype=torch.bool),
            safety_violation=torch.zeros(1, dtype=torch.bool),
            authoritative_safety_penalty=torch.zeros(1),
            non_pad_gripper_cube_contact=torch.zeros(1, dtype=torch.bool),
            close_command=torch.zeros(1, dtype=torch.bool),
            consumed_substeps=torch.full((1,), 10, dtype=torch.int64),
            runtime_hardstop=torch.zeros(1, dtype=torch.bool),
        )

    stacked = stack_single_env_reward_inputs((row(3.0), row(7.0)))
    assert tuple(stacked.current_ee_position_root_m.shape) == (2, 3)
    assert stacked.current_ee_position_root_m[:, 0].tolist() == [3.0, 7.0]
    assert stacked.consumed_substeps.tolist() == [10, 10]
