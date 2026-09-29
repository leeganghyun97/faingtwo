# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure contact-free 4D training contracts for the G2 policy branch.

This module intentionally has no Isaac/PhysX dependency.  It provides the
dataset filter, exact ``[dx, dy, dz, HOLD_OPEN]`` action boundary, replay
buffer, and small CPU-testable actor/critic used before contact mechanics are
qualified.  The legacy 7-D representation is accepted only by an explicit,
validated conversion and never reaches the learner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import nn

from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M


CONTACT_FREE_ACTION_DIM = 4
CONTACT_FREE_MAX_DELTA_M = 0.0045
# This is a policy-side metric safety bound, not the scale of the selected
# production action port.  The latter applies 22.5 mm per normalized unit, so
# a maximum pure-axis contact-free request (4.5 mm) maps to normalized 0.20.
CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M = G2_TRANSLATION_ACTION_SCALE_M
CONTACT_FREE_SAC_GAMMA = 0.9993
CONTACT_FREE_HORIZON = 640
CONTACT_FREE_BATCH_SIZE = 256
CONTACT_FREE_REPLAY_CAPACITY = 100_000


class ContactFreeContractError(ValueError):
    pass


@dataclass(frozen=True)
class ContactFreeSACConfig:
    action_dim: int = CONTACT_FREE_ACTION_DIM
    gamma: float = CONTACT_FREE_SAC_GAMMA
    batch_size: int = CONTACT_FREE_BATCH_SIZE
    replay_capacity: int = CONTACT_FREE_REPLAY_CAPACITY
    learning_starts: int = 1000
    updates_per_step: int = 1
    control_rate_hz: float = 50.0
    horizon: int = CONTACT_FREE_HORIZON
    her_enabled: bool = False
    her_force_enabled: bool = False

    def validated(self) -> "ContactFreeSACConfig":
        if self.action_dim != CONTACT_FREE_ACTION_DIM:
            raise ContactFreeContractError("contact-free action dimension must be 4")
        if not 0.0 < self.gamma <= 1.0:
            raise ContactFreeContractError("gamma must be in (0,1]")
        if self.batch_size <= 0 or self.replay_capacity <= 0:
            raise ContactFreeContractError("batch/replay sizes must be positive")
        if self.learning_starts < 0 or self.updates_per_step <= 0:
            raise ContactFreeContractError("invalid SAC schedule")
        if self.control_rate_hz != 50.0 or self.horizon != 640:
            raise ContactFreeContractError("contact-free control contract mismatch")
        if self.her_enabled or self.her_force_enabled:
            raise ContactFreeContractError("HER is disabled before contact qualification")
        return self


def validate_contact_free_action(action: torch.Tensor | np.ndarray) -> np.ndarray:
    """Validate and return an exact 4-D action without clipping."""

    values = np.asarray(action, dtype=np.float64)
    if values.shape[-1] != CONTACT_FREE_ACTION_DIM:
        raise ContactFreeContractError("action must have exact dimension 4")
    if not np.isfinite(values).all():
        raise ContactFreeContractError("action contains NaN/Inf")
    if np.any(values[..., 3] != 0.0):
        raise ContactFreeContractError("contact-free gripper must remain HOLD_OPEN=0")
    if np.any(np.linalg.norm(values[..., :3], axis=-1) > CONTACT_FREE_MAX_DELTA_M + 1e-12):
        raise ContactFreeContractError("Cartesian command exceeds 4.5 mm bound")
    return values


def metric_xyz_to_normalized(metric_xyz: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert bounded root-frame metres to the existing production port.

    The conversion is representation-only; it does not clip, solve IK, or
    make a controller target.  Crucially, it uses the port's 22.5-mm scale,
    not the 4.5-mm policy bound.
    """
    values = np.asarray(metric_xyz, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ContactFreeContractError("metric XYZ must be finite [...,3]")
    if np.any(np.linalg.norm(values, axis=-1) > CONTACT_FREE_MAX_DELTA_M + 1e-12):
        raise ContactFreeContractError("metric XYZ exceeds 4.5 mm bound")
    return values / CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M


def normalized_xyz_to_metric(normalized_xyz: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert production-port normalized root XYZ to bounded metric metres."""
    values = np.asarray(normalized_xyz, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ContactFreeContractError("normalized XYZ must be finite [...,3]")
    if np.any(np.abs(values) > 1.0 + 1e-12):
        raise ContactFreeContractError("normalized XYZ is outside [-1,1]")
    metric = values * CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M
    if np.any(np.linalg.norm(metric, axis=-1) > CONTACT_FREE_MAX_DELTA_M + 1e-12):
        raise ContactFreeContractError(
            "normalized XYZ would exceed the 4.5-mm contact-free bound"
        )
    return metric


def convert_legacy_7d_open_action(action: np.ndarray) -> np.ndarray:
    """Convert only verified legacy open rows to the 4-D learner contract."""

    values = np.asarray(action, dtype=np.float64)
    if values.shape[-1] != 7:
        raise ContactFreeContractError("legacy source must be exactly 7-D")
    if not np.isfinite(values).all():
        raise ContactFreeContractError("legacy action contains NaN/Inf")
    # Legacy indices 3:6 are orientation residuals.  Any non-zero value is
    # rejected rather than silently dropping an action degree of freedom.
    if np.any(np.abs(values[..., 3:6]) > 1e-12):
        raise ContactFreeContractError("legacy orientation action is non-zero")
    if np.any(values[..., 6] < 0.0):
        raise ContactFreeContractError("legacy row requests CLOSE")
    result = np.concatenate((values[..., :3], np.zeros((*values.shape[:-1], 1))), axis=-1)
    return validate_contact_free_action(result)


@dataclass(frozen=True)
class ContactFreeFilterStats:
    total_transitions: int
    valid_contact_free_transitions: int
    removed_contact: int
    removed_close: int
    removed_safety: int
    removed_action_contract: int
    removed_nan: int

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in self.__dict__.items()}


def filter_contact_free_rows(
    *,
    legacy_action_7d: np.ndarray,
    contact: np.ndarray,
    close_command: np.ndarray | None = None,
    collision: np.ndarray | None = None,
    safety_reject: np.ndarray | None = None,
    hard_limit: np.ndarray | None = None,
    invalid_ik: np.ndarray | None = None,
    nan_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, ContactFreeFilterStats]:
    """Filter demonstration rows and return only validated 4-D open rows."""

    action = np.asarray(legacy_action_7d)
    n = int(action.shape[0]) if action.ndim == 2 else 0
    if action.ndim != 2 or action.shape[-1] != 7:
        raise ContactFreeContractError("legacy action table must be [N,7]")
    def mask(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros(n, dtype=bool)
        arr = np.asarray(value).reshape(-1).astype(bool)
        if arr.shape != (n,):
            raise ContactFreeContractError("filter mask length mismatch")
        return arr
    contact_mask = mask(contact)
    close_mask = mask(close_command) | (action[:, 6] < 0.0)
    safety_mask = mask(collision) | mask(safety_reject) | mask(hard_limit) | mask(invalid_ik)
    nan_mask = mask(nan_rows) | ~np.isfinite(action).all(axis=1)
    contract_mask = np.zeros(n, dtype=bool)
    try:
        converted = convert_legacy_7d_open_action(action)
    except ContactFreeContractError:
        # Preserve row-level accounting: rows with non-zero orientation or
        # other invalid legacy fields are action-contract removals.
        converted = np.zeros((n, CONTACT_FREE_ACTION_DIM), dtype=np.float64)
        contract_mask = (
            np.any(np.abs(action[:, 3:6]) > 1e-12, axis=1)
            | (np.linalg.norm(action[:, :3], axis=1) > CONTACT_FREE_MAX_DELTA_M + 1e-12)
            | (~np.isfinite(action).all(axis=1))
        )
    valid = ~(contact_mask | close_mask | safety_mask | nan_mask | contract_mask)
    if valid.any():
        converted[valid] = convert_legacy_7d_open_action(action[valid])
    stats = ContactFreeFilterStats(
        total_transitions=n,
        valid_contact_free_transitions=int(valid.sum()),
        removed_contact=int((contact_mask & ~nan_mask).sum()),
        removed_close=int((close_mask & ~contact_mask & ~nan_mask).sum()),
        removed_safety=int((safety_mask & ~contact_mask & ~close_mask & ~nan_mask).sum()),
        removed_action_contract=int(contract_mask.sum()),
        removed_nan=int(nan_mask.sum()),
    )
    return converted[valid], valid, stats


class ContactFreeReplay:
    """Small, finite CPU replay with an exact 4-D action boundary."""

    def __init__(self, capacity: int = CONTACT_FREE_REPLAY_CAPACITY) -> None:
        if capacity <= 0:
            raise ContactFreeContractError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._rows: list[tuple[torch.Tensor, ...]] = []

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_state: torch.Tensor, done: torch.Tensor) -> None:
        if state.ndim != 1 or next_state.shape != state.shape:
            raise ContactFreeContractError("state shape mismatch")
        if action.shape != (CONTACT_FREE_ACTION_DIM,):
            raise ContactFreeContractError("replay action must be 4-D")
        validate_contact_free_action(action.detach().cpu().numpy())
        row = tuple(x.detach().cpu().float() for x in (state, action, reward, next_state, done))
        self._rows.append(row)
        if len(self._rows) > self.capacity:
            self._rows.pop(0)

    def __len__(self) -> int:
        return len(self._rows)

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, ...]:
        if len(self._rows) < batch_size:
            raise ContactFreeContractError("replay has insufficient rows")
        indices = torch.randperm(len(self._rows), generator=generator)[:batch_size].tolist()
        return tuple(torch.stack([self._rows[i][j] for i in indices]) for j in range(5))


class ContactFreeActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, CONTACT_FREE_ACTION_DIM))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.net(observation))
        action = action.clone()
        action[..., 3] = 0.0
        # Keep the learner output within the same metric bound as the runtime.
        return action[..., :3] * CONTACT_FREE_MAX_DELTA_M, action[..., 3:4]


class ContactFreeCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim + CONTACT_FREE_ACTION_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        validate_contact_free_action(action.detach().cpu().numpy())
        return self.net(torch.cat((observation, action), dim=-1))


def contact_free_cpu_smoke(observation_dim: int = 12, batch_size: int = 8) -> dict[str, object]:
    """Run loader-equivalent replay, actor/critic forward and one optimizer step."""

    config = ContactFreeSACConfig().validated()
    actor = ContactFreeActor(observation_dim)
    critic = ContactFreeCritic(observation_dim)
    replay = ContactFreeReplay(capacity=32)
    for i in range(batch_size):
        state = torch.randn(observation_dim)
        action = torch.zeros(4); action[:3] = torch.tensor([0.0005, 0.0, 0.0])
        replay.add(state, action, torch.tensor(0.1), state + 0.01, torch.tensor(0.0))
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    arm, grip = actor(states)
    policy_action = torch.cat((arm, grip), dim=-1)
    q = critic(states, policy_action)
    loss = (q - rewards.unsqueeze(-1)).square().mean() + policy_action.square().mean()
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    optimizer.zero_grad(); loss.backward()
    gradients_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in list(actor.parameters()) + list(critic.parameters()))
    optimizer.step()
    return {
        "obs_dim_match": states.shape[-1] == observation_dim,
        "action_dim": int(policy_action.shape[-1]),
        "replay_sample_valid": len(replay) == batch_size and actions.shape[-1] == 4,
        "loss_finite": bool(torch.isfinite(loss)),
        "grad_finite": gradients_finite,
        "her": "DISABLED",
        "her_force": "DISABLED",
        "config": config.__dict__,
    }


__all__ = [
    "CONTACT_FREE_ACTION_DIM", "CONTACT_FREE_MAX_DELTA_M",
    "CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M", "ContactFreeSACConfig",
    "ContactFreeContractError", "ContactFreeFilterStats", "ContactFreeReplay",
    "ContactFreeActor", "ContactFreeCritic", "convert_legacy_7d_open_action",
    "filter_contact_free_rows", "validate_contact_free_action", "contact_free_cpu_smoke",
    "metric_xyz_to_normalized", "normalized_xyz_to_metric",
]
