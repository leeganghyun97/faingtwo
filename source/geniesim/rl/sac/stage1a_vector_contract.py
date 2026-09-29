# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure contracts for the independently-stateful Stage-1A vector runtime.

The original Stage-1A live runner intentionally supports exactly one Isaac
environment.  This module does *not* weaken that runner.  It defines the
additional immutable packet, per-environment state, and replay identities a
separate vector runner must satisfy before it may use a batched ``env.step``.

No class here owns an ActionManager, controller, articulation, or physics
API.  A live runner can only materialize a fresh ``[N, 8]`` tensor after each
row has passed through the existing 4-D high-level action contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
    AbstractGripperIntent,
    HighLevelPolicyAction,
    expand_to_existing_controller_8d,
)
from geniesim.rl.isaaclab.g2_policy_branch.contact_free_training_contract import (
    metric_xyz_to_normalized,
)


VECTOR_STAGE1A_ACTION_SCHEMA = "g2_stage1a_per_env_4d_to_8d_v1"
VECTOR_STAGE1A_REPLAY_SCHEMA = "g2_stage1a_per_env_real_replay_row_v1"


class Stage1AVectorContractError(ValueError):
    """Raised before an invalid vector row can reach a live environment."""


def _finite_tuple(name: str, value: Sequence[float], size: int) -> tuple[float, ...]:
    result = tuple(float(component) for component in value)
    if len(result) != size or not all(math.isfinite(component) for component in result):
        raise Stage1AVectorContractError(f"{name} must be {size} finite values")
    return result


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PerEnvCanonicalAction:
    """One environment's exact 4-D policy result and downstream 8-D row."""

    env_id: int
    final_action_4d_metric_root_m: tuple[float, float, float, float]
    gripper_intent: AbstractGripperIntent
    schema: str = VECTOR_STAGE1A_ACTION_SCHEMA
    normalized_action_4d: tuple[float, float, float, float] = field(init=False)
    canonical_action_8d: tuple[float, float, float, float, float, float, float, float] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise Stage1AVectorContractError("env_id must be a nonnegative int")
        if self.schema != VECTOR_STAGE1A_ACTION_SCHEMA:
            raise Stage1AVectorContractError("unsupported per-env action schema")
        if not isinstance(self.gripper_intent, AbstractGripperIntent):
            raise Stage1AVectorContractError("gripper_intent must be abstract OPEN/CLOSE")
        action = _finite_tuple(
            "final_action_4d_metric_root_m", self.final_action_4d_metric_root_m, 4
        )
        if not 0.0 <= action[3] <= 1.0:
            raise Stage1AVectorContractError("gripper probability must be in [0,1]")
        if float(np.linalg.norm(np.asarray(action[:3], dtype=np.float64))) > 0.0045 + 1.0e-12:
            raise Stage1AVectorContractError("final per-env XYZ action exceeds 4.5 mm")
        normalized_xyz = tuple(float(value) for value in metric_xyz_to_normalized(action[:3]))
        high_level = HighLevelPolicyAction.from_sequence((*normalized_xyz, action[3]))
        canonical = tuple(
            float(value)
            for value in expand_to_existing_controller_8d(
                high_level, gripper_intent=self.gripper_intent
            )
        )
        object.__setattr__(self, "final_action_4d_metric_root_m", action)
        object.__setattr__(self, "normalized_action_4d", (*normalized_xyz, action[3]))
        object.__setattr__(self, "canonical_action_8d", canonical)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                {
                    "schema": self.schema,
                    "env_id": self.env_id,
                    "metric_float_hex": [value.hex() for value in action],
                    "intent": self.gripper_intent.value,
                    "canonical_float_hex": [value.hex() for value in canonical],
                }
            ),
        )


@dataclass(frozen=True)
class ImmutablePerEnvActionPacket:
    """An immutable set of *independent* canonical 8-D rows.

    The public scalar P0 packet deliberately repeats a scalar action across a
    batch.  This separate schema instead records every row in the fingerprint
    and materializes those rows unchanged for exactly one canonical env.step.
    """

    rows: tuple[PerEnvCanonicalAction, ...]
    binding_id: str
    sequence_id: int
    device: str
    schema: str = VECTOR_STAGE1A_ACTION_SCHEMA
    packet_id: str = field(init=False)
    content_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.rows:
            raise Stage1AVectorContractError("per-env action packet cannot be empty")
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise Stage1AVectorContractError("binding_id must be nonempty")
        if type(self.sequence_id) is not int or self.sequence_id <= 0:
            raise Stage1AVectorContractError("sequence_id must be positive")
        if self.schema != VECTOR_STAGE1A_ACTION_SCHEMA:
            raise Stage1AVectorContractError("unsupported vector packet schema")
        expected = tuple(range(len(self.rows)))
        actual = tuple(row.env_id for row in self.rows)
        if actual != expected:
            raise Stage1AVectorContractError("packet rows must be ordered env_id 0..N-1")
        payload = {
            "schema": self.schema,
            "binding_id": self.binding_id,
            "sequence_id": self.sequence_id,
            "batch_shape": [len(self.rows), 8],
            "device": str(torch.device(self.device)),
            "row_fingerprints": [row.fingerprint for row in self.rows],
        }
        digest = _fingerprint(payload)
        object.__setattr__(self, "device", str(torch.device(self.device)))
        object.__setattr__(self, "content_fingerprint", digest)
        object.__setattr__(
            self, "packet_id", f"{self.binding_id}:{self.sequence_id:016d}:{digest[:16]}"
        )

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return self.batch_size, 8

    @property
    def unique_4d_rows(self) -> int:
        return len({row.final_action_4d_metric_root_m for row in self.rows})

    @property
    def unique_8d_rows(self) -> int:
        return len({row.canonical_action_8d for row in self.rows})

    @property
    def broadcast_detected(self) -> bool:
        """Whether this packet was created through a scalar-repeat path.

        Equal numerical actions are not proof of a broadcast: two independent
        clones can legitimately choose an identical OPEN/CLOSE hold action.
        A vector packet is broadcast-free when every row was independently
        constructed with a distinct ``env_id`` and is materialized by stacking
        those rows.  The class rejects an out-of-order/duplicated env identity
        in ``__post_init__``; no ``repeat``/``expand`` API is present here.
        """

        return False

    def as_env_step_tensor(self) -> torch.Tensor:
        """Return a fresh immutable-source tensor for the sole env.step call."""

        tensor = torch.tensor(
            [row.canonical_action_8d for row in self.rows],
            dtype=torch.float32,
            device=self.device,
        )
        if tuple(tensor.shape) != self.shape:
            raise Stage1AVectorContractError("vector action tensor shape mismatch")
        return tensor


class PerEnvPacketPort:
    """One-outstanding-packet ledger with no ActionManager/controller access."""

    def __init__(self, *, batch_size: int, device: str | torch.device, binding_id: str) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise Stage1AVectorContractError("batch_size must be positive")
        self.batch_size = batch_size
        self.device = str(torch.device(device))
        self.binding_id = binding_id
        self._next_sequence_id = 1
        self._outstanding: ImmutablePerEnvActionPacket | None = None
        self.stage_count = 0
        self.claim_count = 0
        self.ack_count = 0

    @property
    def outstanding_packet(self) -> ImmutablePerEnvActionPacket | None:
        return self._outstanding

    def stage(self, rows: Iterable[PerEnvCanonicalAction]) -> ImmutablePerEnvActionPacket:
        if self._outstanding is not None:
            raise Stage1AVectorContractError("previous vector packet is outstanding")
        packet = ImmutablePerEnvActionPacket(
            rows=tuple(rows),
            binding_id=self.binding_id,
            sequence_id=self._next_sequence_id,
            device=self.device,
        )
        if packet.batch_size != self.batch_size:
            raise Stage1AVectorContractError("vector packet batch size mismatch")
        self._next_sequence_id += 1
        self._outstanding = packet
        self.stage_count += 1
        return packet

    def claim(self, packet_id: str) -> ImmutablePerEnvActionPacket:
        if self._outstanding is None or self._outstanding.packet_id != packet_id:
            raise Stage1AVectorContractError("unknown vector packet claim")
        self.claim_count += 1
        return self._outstanding

    def acknowledge(self, packet_id: str) -> ImmutablePerEnvActionPacket:
        packet = self.claim(packet_id)
        self._outstanding = None
        self.ack_count += 1
        return packet


@dataclass
class PerEnvTemporalState:
    """All state that must never be shared between vector environments."""

    episode_index: int = 0
    control_step: int = 0
    gru_hidden: torch.Tensor | None = None
    close_latched: bool = False
    close_persistence_count: int = 0
    first_contact_latched: bool = False
    bilateral_latched: bool = False
    stable_latched: bool = False
    hover_steps: int = 0
    camera_frame_id: int | None = None
    camera_timestamp_s: float | None = None


class PerEnvStateRegistry:
    """Owns reset-isolated temporal state for a fixed vector batch."""

    def __init__(self, num_envs: int) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise Stage1AVectorContractError("num_envs must be positive")
        self.num_envs = num_envs
        self._states = [PerEnvTemporalState() for _ in range(num_envs)]

    def state(self, env_id: int) -> PerEnvTemporalState:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("env_id outside vector batch")
        return self._states[env_id]

    def reset(self, env_id: int) -> None:
        previous = self.state(env_id)
        self._states[env_id] = PerEnvTemporalState(episode_index=previous.episode_index + 1)

    def snapshot(self) -> tuple[PerEnvTemporalState, ...]:
        return tuple(self._states)


@dataclass(frozen=True)
class PerEnvCameraFrame:
    """One env's source-owned 25-Hz camera identity and immutable evidence."""

    env_id: int
    frame_id: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    depth_valid: np.ndarray

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise Stage1AVectorContractError("camera env_id must be nonnegative")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise Stage1AVectorContractError("camera frame_id must be nonnegative")
        if not math.isfinite(float(self.timestamp_s)) or self.timestamp_s < 0.0:
            raise Stage1AVectorContractError("camera timestamp must be finite/nonnegative")
        rgb = np.ascontiguousarray(np.asarray(self.rgb, dtype=np.uint8).copy())
        depth = np.ascontiguousarray(np.asarray(self.depth_m, dtype=np.float32).copy())
        valid = np.ascontiguousarray(np.asarray(self.depth_valid, dtype=np.bool_).copy())
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != valid.shape:
            raise Stage1AVectorContractError("camera frame shapes are invalid")
        rgb.flags.writeable = False
        depth.flags.writeable = False
        valid.flags.writeable = False
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "depth_valid", valid)


class PerEnvCameraCache:
    """25-Hz cache with explicit, clone-local camera reset generations.

    Isaac may restart one camera's source clock after a direct source restore.
    A numeric timestamp/frame comparison across that boundary is therefore not
    a valid freshness proof.  ``reset()`` clears only the selected clone's
    cache and opens one *bounded* post-reset epoch-transition allowance.  The
    caller must seal that allowance when measured reset parity completes; a
    later regression during an active episode remains fail-closed.
    """

    def __init__(self, num_envs: int) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise Stage1AVectorContractError("num_envs must be positive")
        self.num_envs = num_envs
        self._frames: list[PerEnvCameraFrame | None] = [None] * num_envs
        self._reset_generations: list[int] = [0] * num_envs
        self._epoch_transition_allowed: list[bool] = [False] * num_envs
        self._epoch_transition_counts: list[int] = [0] * num_envs

    def put(self, frame: PerEnvCameraFrame) -> None:
        if frame.env_id >= self.num_envs:
            raise Stage1AVectorContractError("camera frame env_id outside vector batch")
        previous = self._frames[frame.env_id]
        if previous is not None:
            epoch_transition = bool(
                frame.frame_id < previous.frame_id
                or frame.timestamp_s < previous.timestamp_s
                or (
                    frame.frame_id != previous.frame_id
                    and frame.timestamp_s == previous.timestamp_s
                )
            )
            if epoch_transition:
                if not self._epoch_transition_allowed[frame.env_id]:
                    raise Stage1AVectorContractError(
                        "camera acquisition timestamp/frame regressed"
                    )
                # A reset-generation transition is causal evidence of a new
                # sensor epoch, never an instruction to reuse the old image.
                self._epoch_transition_allowed[frame.env_id] = False
                self._epoch_transition_counts[frame.env_id] += 1
            if (
                not epoch_transition
                and frame.frame_id == previous.frame_id
                and frame.timestamp_s != previous.timestamp_s
            ):
                raise Stage1AVectorContractError("reused camera frame changed timestamp")
        self._frames[frame.env_id] = frame

    def get(self, env_id: int) -> PerEnvCameraFrame | None:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("camera env_id outside vector batch")
        return self._frames[env_id]

    def reset(self, env_id: int) -> None:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("camera env_id outside vector batch")
        self._frames[env_id] = None
        self._reset_generations[env_id] += 1
        # The first cached frame can still belong to the pre-restore epoch if
        # the sensor has not rendered yet.  Permit exactly one later raw
        # identity rollback while the reset gate is pending.
        self._epoch_transition_allowed[env_id] = True

    def reset_generation(self, env_id: int) -> int:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("camera env_id outside vector batch")
        return int(self._reset_generations[env_id])

    def epoch_transition_count(self, env_id: int) -> int:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("camera env_id outside vector batch")
        return int(self._epoch_transition_counts[env_id])

    def reset_epoch_transition_allowed(self, env_id: int) -> bool:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("camera env_id outside vector batch")
        return bool(self._epoch_transition_allowed[env_id])

    def seal_reset_epoch(self, env_id: int) -> None:
        """Close the reset-only clock-epoch allowance before policy ACTIVE."""

        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("camera env_id outside vector batch")
        self._epoch_transition_allowed[env_id] = False


@dataclass
class _CoordinatorEpisodeState:
    """Private mutable state owned by one environment of a shared learner."""

    hidden: torch.Tensor | None = None
    closed: bool = False
    episode_milestones: dict[str, int] = field(default_factory=dict)
    current_v3_hover_steps: int = 0
    current_max_residual_streak: int = 0


class PerEnvCoordinatorStateAdapter:
    """Give a single shared Stage1A learner independent recurrent episodes.

    The learner's actor/critic/replay/optimizer remain shared.  Only the
    recurrent and episode-latch fields are swapped at an env boundary.  This
    prevents an env reset or CLOSE edge from changing any other environment's
    hidden state while preserving update/data accounting in the one learner.
    """

    _FIELDS = (
        "hidden",
        "_closed",
        "_episode_milestones",
        "_current_v3_hover_steps",
        "_current_max_residual_streak",
    )

    def __init__(self, coordinator: Any, *, num_envs: int) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise Stage1AVectorContractError("num_envs must be positive")
        missing = [name for name in self._FIELDS if not hasattr(coordinator, name)]
        if missing:
            raise Stage1AVectorContractError(
                f"coordinator lacks vector-isolated state: {missing}"
            )
        self.coordinator = coordinator
        self.num_envs = num_envs
        self._states = [_CoordinatorEpisodeState() for _ in range(num_envs)]

    @staticmethod
    def _clone_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
        return None if hidden is None else hidden.detach().clone()

    def _restore(self, env_id: int) -> None:
        state = self._state(env_id)
        self.coordinator.hidden = self._clone_hidden(state.hidden)
        self.coordinator._closed = bool(state.closed)
        self.coordinator._episode_milestones = dict(state.episode_milestones)
        self.coordinator._current_v3_hover_steps = int(state.current_v3_hover_steps)
        self.coordinator._current_max_residual_streak = int(
            state.current_max_residual_streak
        )

    def _capture(self, env_id: int) -> None:
        state = self._state(env_id)
        state.hidden = self._clone_hidden(self.coordinator.hidden)
        state.closed = bool(self.coordinator._closed)
        state.episode_milestones = dict(self.coordinator._episode_milestones)
        state.current_v3_hover_steps = int(self.coordinator._current_v3_hover_steps)
        state.current_max_residual_streak = int(
            self.coordinator._current_max_residual_streak
        )

    def _state(self, env_id: int) -> _CoordinatorEpisodeState:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise Stage1AVectorContractError("env_id outside vector batch")
        return self._states[env_id]

    def call(self, env_id: int, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Call a coordinator method against exactly one env-local state."""

        callback = getattr(self.coordinator, method, None)
        if not callable(callback):
            raise Stage1AVectorContractError(f"coordinator method unavailable: {method}")
        self._restore(env_id)
        try:
            return callback(*args, **kwargs)
        finally:
            self._capture(env_id)

    def reset(self, env_id: int) -> None:
        self.call(env_id, "reset_episode")

    def state_snapshot(self, env_id: int) -> _CoordinatorEpisodeState:
        state = self._state(env_id)
        return _CoordinatorEpisodeState(
            hidden=self._clone_hidden(state.hidden),
            closed=state.closed,
            episode_milestones=dict(state.episode_milestones),
            current_v3_hover_steps=state.current_v3_hover_steps,
            current_max_residual_streak=state.current_max_residual_streak,
        )


@dataclass(frozen=True)
class PerEnvReplayIdentity:
    """The minimum identity needed to prove a row was not batch-broadcast."""

    env_id: int
    episode_id: str
    step_in_episode: int
    schema: str = VECTOR_STAGE1A_REPLAY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise Stage1AVectorContractError("replay env_id must be nonnegative")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise Stage1AVectorContractError("replay episode_id must be nonempty")
        if type(self.step_in_episode) is not int or self.step_in_episode < 0:
            raise Stage1AVectorContractError("step_in_episode must be nonnegative")
        if self.schema != VECTOR_STAGE1A_REPLAY_SCHEMA:
            raise Stage1AVectorContractError("unsupported replay identity schema")

    @property
    def row_id(self) -> str:
        return f"env-{self.env_id:02d}:{self.episode_id}:step-{self.step_in_episode:06d}"
