# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Authoritative four-dimensional SAC surface for the G2 policy branch.

This module is deliberately simulator independent.  It consumes the frozen
observation embedding produced by the high-level policy encoder/temporal
stack and emits exactly ``[dx, dy, dz, gripper_probability]``.  It does not
import or adapt the legacy seven-dimensional Stage-2 runner.

HER and HER_FORCE are disabled in the current contract.  The branch has no
source-owned achieved-goal/desired-goal representation or reward
recomputation authority yet; silently borrowing the legacy task definition
would make hindsight rewards invalid.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .action_interface import POLICY_ACTION_SCHEMA_4D
from .bc_policy import HighLevelBCPolicy
from .runtime_asset_binding import (
    M2_QUALIFICATION_REPORT_SHA256,
    M2_QUALIFIED_CANDIDATE_BINDING_ID,
    M2_QUALIFIED_CANDIDATE_CLASSIFICATION,
    M2_QUALIFIED_CANDIDATE_RELATIVE_PATH,
    M2_QUALIFIED_CANDIDATE_SHA256,
    M2_QUALIFIED_DEPENDENCY_MANIFEST_SHA256,
)


POLICY_SAC_4D_SCHEMA = "g2_policy_branch_sac_4d_v1"
POLICY_REPLAY_4D_SCHEMA = "g2_policy_branch_replay_4d_v1"
POLICY_CHECKPOINT_4D_SCHEMA = "g2_policy_branch_full_checkpoint_4d_v1"
POLICY_ACTION_DIM_4D = 4

M2_VALIDATED_ASSET_RELATIVE_PATH = (
    M2_QUALIFIED_CANDIDATE_RELATIVE_PATH.as_posix()
)
M2_VALIDATED_ASSET_SHA256 = M2_QUALIFIED_CANDIDATE_SHA256
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicySAC4DContractError(ValueError):
    """Raised when training state crosses the frozen four-dimensional boundary."""


class GoalRelabelingStatus(str, Enum):
    """Current branch-local HER authority state."""

    DISABLED_NO_AUTHORITATIVE_GOAL_CONTRACT = (
        "DISABLED_NO_AUTHORITATIVE_GOAL_CONTRACT"
    )
    ENABLED_WITH_AUTHORITATIVE_GOAL_CONTRACT = (
        "ENABLED_WITH_AUTHORITATIVE_GOAL_CONTRACT"
    )


@dataclass(frozen=True)
class GoalRelabelingBinding:
    """Required authority receipt before HER or HER_FORCE can be enabled.

    Fingerprints name external, source-owned contracts.  This module does not
    invent their contents and does not contain a relabeling implementation.
    """

    status: GoalRelabelingStatus = (
        GoalRelabelingStatus.DISABLED_NO_AUTHORITATIVE_GOAL_CONTRACT
    )
    goal_contract_sha256: str | None = None
    reward_recomputation_sha256: str | None = None
    her_force_priority_sha256: str | None = None

    def __post_init__(self) -> None:
        fingerprints = (
            self.goal_contract_sha256,
            self.reward_recomputation_sha256,
            self.her_force_priority_sha256,
        )
        if self.status is GoalRelabelingStatus.DISABLED_NO_AUTHORITATIVE_GOAL_CONTRACT:
            if any(value is not None for value in fingerprints):
                raise PolicySAC4DContractError(
                    "disabled HER binding cannot carry unapproved goal/reward authority"
                )
            return
        if self.status is not GoalRelabelingStatus.ENABLED_WITH_AUTHORITATIVE_GOAL_CONTRACT:
            raise PolicySAC4DContractError("unknown goal relabeling status")
        if not all(value is not None and _SHA256.fullmatch(value) for value in fingerprints):
            raise PolicySAC4DContractError(
                "enabled HER/HER_FORCE requires goal, reward, and force-priority authority hashes"
            )

    @property
    def enabled(self) -> bool:
        return self.status is GoalRelabelingStatus.ENABLED_WITH_AUTHORITATIVE_GOAL_CONTRACT

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "goal_contract_sha256": self.goal_contract_sha256,
            "reward_recomputation_sha256": self.reward_recomputation_sha256,
            "her_force_priority_sha256": self.her_force_priority_sha256,
        }


def _repository_root() -> Path:
    # .../source/geniesim/rl/isaaclab/g2_policy_branch/sac4d.py -> repository
    return Path(__file__).resolve().parents[5]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingAssetBinding:
    """Exact M2-qualified simulation asset identity used by training."""

    relative_path: str = M2_VALIDATED_ASSET_RELATIVE_PATH
    sha256: str = M2_VALIDATED_ASSET_SHA256
    binding_id: str = M2_QUALIFIED_CANDIDATE_BINDING_ID
    classification: str = M2_QUALIFIED_CANDIDATE_CLASSIFICATION
    dependency_manifest_sha256: str = M2_QUALIFIED_DEPENDENCY_MANIFEST_SHA256
    qualification_report_sha256: str = M2_QUALIFICATION_REPORT_SHA256

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise PolicySAC4DContractError(
                "training asset path must be repository-relative and non-escaping"
            )
        expected = {
            "relative_path": M2_VALIDATED_ASSET_RELATIVE_PATH,
            "sha256": M2_VALIDATED_ASSET_SHA256,
            "binding_id": M2_QUALIFIED_CANDIDATE_BINDING_ID,
            "classification": M2_QUALIFIED_CANDIDATE_CLASSIFICATION,
            "dependency_manifest_sha256": M2_QUALIFIED_DEPENDENCY_MANIFEST_SHA256,
            "qualification_report_sha256": M2_QUALIFICATION_REPORT_SHA256,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise PolicySAC4DContractError(
                    f"training asset binding differs from M2-qualified authority: {name}"
                )
        for name in ("sha256", "dependency_manifest_sha256", "qualification_report_sha256"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise PolicySAC4DContractError(f"{name} requires lowercase SHA-256")

    def resolve(self, repository_root: Path | None = None) -> Path:
        return (repository_root or _repository_root()) / self.relative_path

    def assert_exact(self, repository_root: Path | None = None) -> Path:
        path = self.resolve(repository_root)
        if not path.is_file():
            raise PolicySAC4DContractError(f"M2-qualified training asset is missing: {path}")
        actual = _file_sha256(path)
        if actual != self.sha256:
            raise PolicySAC4DContractError(
                f"training asset hash mismatch: expected {self.sha256}, got {actual}"
            )
        return path

    def payload(self) -> dict[str, str]:
        return asdict(self)

    def fingerprint(self) -> str:
        return _canonical_sha256(self.payload())


def _named_module_sha256(modules: Sequence[tuple[str, nn.Module | None]]) -> str:
    digest = hashlib.sha256()
    for prefix, module in modules:
        if module is None:
            continue
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(prefix.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def bc_embedding_producer_sha256(policy: HighLevelBCPolicy) -> str:
    """Hash exactly the modules that produce ``recurrent_features``."""

    return _named_module_sha256(
        (
            ("wrist_encoder", policy.wrist_encoder),
            ("head_encoder", policy.head_encoder),
            ("state_encoder", policy.state_encoder),
            ("fusion", policy.fusion),
            ("gru", policy.gru),
        )
    )


def bc_action_heads_sha256(policy: HighLevelBCPolicy) -> str:
    """Hash the four-dimensional BC action heads transplanted into SAC."""

    if policy.rotation_head is not None or policy.config.action_dim != 4:
        raise PolicySAC4DContractError("BC handoff requires the exact 4-D policy")
    return _named_module_sha256(
        (("translation_head", policy.translation_head), ("gripper_head", policy.gripper_head))
    )


def _bc_state_hashes(state: Mapping[str, Tensor]) -> tuple[str, str]:
    """Hash checkpointed BC tensors with the same namespace as live modules."""

    def digest_prefixes(prefixes: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for prefix in prefixes:
            qualified = prefix + "."
            entries = sorted(
                (name[len(qualified) :], value)
                for name, value in state.items()
                if name.startswith(qualified)
            )
            if not entries and prefix != "head_encoder":
                raise PolicySAC4DContractError(
                    f"checkpointed BC state lacks required module: {prefix}"
                )
            for name, value in entries:
                tensor = value.detach().to(device="cpu").contiguous()
                digest.update(prefix.encode("utf-8"))
                digest.update(name.encode("utf-8"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()

    return (
        digest_prefixes(
            ("wrist_encoder", "head_encoder", "state_encoder", "fusion", "gru")
        ),
        digest_prefixes(("translation_head", "gripper_head")),
    )


@dataclass(frozen=True)
class BCHandoffBinding:
    """Frozen producer and action-head provenance for BC-to-SAC transfer."""

    schema: str
    observation_semantic_fingerprint: str
    embedding_producer_sha256: str
    action_heads_sha256: str
    embedding_dim: int
    method: str = "EXACT_BC_ACTION_HEAD_TRANSPLANT_ON_SHARED_RECURRENT_EMBEDDING"

    def __post_init__(self) -> None:
        if self.schema != "g2_policy_branch_bc_to_sac_handoff_v1":
            raise PolicySAC4DContractError("unsupported BC handoff schema")
        for name in (
            "observation_semantic_fingerprint",
            "embedding_producer_sha256",
            "action_heads_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise PolicySAC4DContractError(f"BC handoff {name} must be SHA-256")
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0:
            raise PolicySAC4DContractError("BC handoff embedding dimension must be positive")
        if self.method != "EXACT_BC_ACTION_HEAD_TRANSPLANT_ON_SHARED_RECURRENT_EMBEDDING":
            raise PolicySAC4DContractError("BC handoff method is not authoritative")

    @classmethod
    def from_policy(cls, policy: HighLevelBCPolicy) -> "BCHandoffBinding":
        if not isinstance(policy, HighLevelBCPolicy):
            raise PolicySAC4DContractError("BC handoff requires HighLevelBCPolicy")
        if policy.config.action_dim != 4 or policy.rotation_head is not None:
            raise PolicySAC4DContractError("BC handoff cannot use a legacy 7-D policy")
        return cls(
            schema="g2_policy_branch_bc_to_sac_handoff_v1",
            observation_semantic_fingerprint=policy.config.semantic_fingerprint(),
            embedding_producer_sha256=bc_embedding_producer_sha256(policy),
            action_heads_sha256=bc_action_heads_sha256(policy),
            embedding_dim=policy.config.gru_hidden_dim,
        )

    def assert_policy(self, policy: HighLevelBCPolicy) -> None:
        if self != BCHandoffBinding.from_policy(policy):
            raise PolicySAC4DContractError("BC policy differs from frozen SAC handoff binding")

    def assert_checkpoint_state(self, state: Mapping[str, Tensor]) -> None:
        embedding_hash, head_hash = _bc_state_hashes(state)
        if embedding_hash != self.embedding_producer_sha256:
            raise PolicySAC4DContractError(
                "checkpoint BC embedding producer differs from handoff binding"
            )
        if head_hash != self.action_heads_sha256:
            raise PolicySAC4DContractError(
                "checkpoint BC action heads differ from handoff binding"
            )

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return _canonical_sha256(self.payload())


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PolicySAC4DConfig:
    """Algorithm and provenance contract for the isolated 4-D learner."""

    observation_embedding_dim: int
    observation_semantic_fingerprint: str
    bc_handoff: BCHandoffBinding
    hidden_sizes: tuple[int, ...] = (256, 256)
    gamma: float = 0.9993
    tau: float = 0.005
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    alpha_learning_rate: float = 3.0e-4
    initial_alpha: float = 0.1
    target_entropy: float = -4.0
    log_standard_deviation_minimum: float = -5.0
    log_standard_deviation_maximum: float = 2.0
    gradient_clip: float = 5.0
    seed: int = 42
    asset: TrainingAssetBinding = TrainingAssetBinding()
    goal_relabeling: GoalRelabelingBinding = GoalRelabelingBinding()

    def __post_init__(self) -> None:
        if type(self.observation_embedding_dim) is not int or self.observation_embedding_dim <= 0:
            raise PolicySAC4DContractError("observation embedding dimension must be positive")
        if not _SHA256.fullmatch(self.observation_semantic_fingerprint):
            raise PolicySAC4DContractError(
                "observation semantics require a lowercase SHA-256 fingerprint"
            )
        if self.bc_handoff.observation_semantic_fingerprint != self.observation_semantic_fingerprint:
            raise PolicySAC4DContractError("BC handoff and SAC observation semantics differ")
        if self.bc_handoff.embedding_dim != self.observation_embedding_dim:
            raise PolicySAC4DContractError("BC handoff and SAC embedding dimensions differ")
        if not self.hidden_sizes or any(type(width) is not int or width <= 0 for width in self.hidden_sizes):
            raise PolicySAC4DContractError("hidden sizes must be positive integers")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "alpha_learning_rate",
            "initial_alpha",
            "gradient_clip",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PolicySAC4DContractError(f"{name} must be finite and positive")
        if not math.isfinite(self.gamma) or not math.isclose(
            self.gamma, 0.9993, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise PolicySAC4DContractError(
                "SAC gamma must equal the current task/PBRS gamma 0.9993"
            )
        if not math.isfinite(self.tau) or not 0.0 < self.tau <= 1.0:
            raise PolicySAC4DContractError("tau must be in (0,1]")
        if not math.isfinite(self.target_entropy):
            raise PolicySAC4DContractError("target entropy must be finite")
        if self.log_standard_deviation_minimum >= self.log_standard_deviation_maximum:
            raise PolicySAC4DContractError("invalid log standard-deviation bounds")
        if self.goal_relabeling.enabled:
            # Binding is necessary but not sufficient.  The current module has
            # no authoritative branch-local recomputation implementation.
            raise PolicySAC4DContractError(
                "HER/HER_FORCE remains unavailable until the current 4-D task owns "
                "an implemented goal and reward-recomputation contract"
            )

    @property
    def action_dim(self) -> int:
        return POLICY_ACTION_DIM_4D

    def payload(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SAC_4D_SCHEMA,
            "action_schema": POLICY_ACTION_SCHEMA_4D,
            "action_dim": self.action_dim,
            "observation_embedding_dim": self.observation_embedding_dim,
            "observation_semantic_fingerprint": self.observation_semantic_fingerprint,
            "bc_handoff": self.bc_handoff.payload(),
            "bc_handoff_fingerprint": self.bc_handoff.fingerprint(),
            "hidden_sizes": list(self.hidden_sizes),
            "gamma": self.gamma,
            "tau": self.tau,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "alpha_learning_rate": self.alpha_learning_rate,
            "initial_alpha": self.initial_alpha,
            "target_entropy": self.target_entropy,
            "log_standard_deviation_minimum": self.log_standard_deviation_minimum,
            "log_standard_deviation_maximum": self.log_standard_deviation_maximum,
            "gradient_clip": self.gradient_clip,
            "seed": self.seed,
            "asset": self.asset.payload(),
            "asset_fingerprint": self.asset.fingerprint(),
            "goal_relabeling": self.goal_relabeling.payload(),
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.payload())


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layer = nn.Linear(previous, width)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.SiLU()))
        previous = width
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class PolicyActionSample4D:
    action: Tensor
    log_probability: Tensor
    mean_action: Tensor


class PolicyActor4D(nn.Module):
    """Gaussian actor with XYZ in [-1,1] and gripper probability in [0,1]."""

    def __init__(self, config: PolicySAC4DConfig) -> None:
        super().__init__()
        self.config = config
        # The observation is already the BC GRU recurrent feature.  A direct
        # affine mean permits an exact transplant of the BC translation and
        # gripper heads; adding a random trunk here would discard that flow.
        self.mean = nn.Linear(config.observation_embedding_dim, POLICY_ACTION_DIM_4D)
        self.log_standard_deviation = nn.Parameter(
            torch.full((POLICY_ACTION_DIM_4D,), -2.0)
        )
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def distribution_parameters(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        if embedding.ndim != 2 or embedding.shape[-1] != self.config.observation_embedding_dim:
            raise PolicySAC4DContractError("actor embedding must be [batch, embedding_dim]")
        if not bool(torch.isfinite(embedding).all()):
            raise PolicySAC4DContractError("actor embedding must be finite")
        mean = self.mean(embedding)
        raw = torch.tanh(self.log_standard_deviation).expand_as(mean)
        low = self.config.log_standard_deviation_minimum
        high = self.config.log_standard_deviation_maximum
        log_std = low + 0.5 * (high - low) * (raw + 1.0)
        return mean, log_std

    def transplant_bc_action_heads(self, policy: HighLevelBCPolicy) -> None:
        self.config.bc_handoff.assert_policy(policy)
        with torch.no_grad():
            self.mean.weight[:3].copy_(policy.translation_head.weight)
            self.mean.bias[:3].copy_(policy.translation_head.bias)
            self.mean.weight[3:4].copy_(policy.gripper_head.weight)
            self.mean.bias[3:4].copy_(policy.gripper_head.bias)

    @staticmethod
    def _transform(latent: Tensor) -> Tensor:
        xyz = torch.tanh(latent[..., :3])
        gripper = torch.sigmoid(latent[..., 3:4])
        return torch.cat((xyz, gripper), dim=-1)

    @staticmethod
    def _log_probability(latent: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        normal = torch.distributions.Normal(mean, log_std.exp())
        log_prob = normal.log_prob(latent)
        xyz_correction = torch.log(1.0 - torch.tanh(latent[..., :3]).pow(2) + 1.0e-6)
        # d sigmoid(z)/dz = sigmoid(z) * sigmoid(-z), evaluated stably.
        gripper_correction = F.logsigmoid(latent[..., 3:4]) + F.logsigmoid(
            -latent[..., 3:4]
        )
        correction = torch.cat((xyz_correction, gripper_correction), dim=-1)
        return (log_prob - correction).sum(dim=-1, keepdim=True)

    def sample(self, embedding: Tensor, *, deterministic: bool = False) -> PolicyActionSample4D:
        mean, log_std = self.distribution_parameters(embedding)
        latent = mean if deterministic else torch.distributions.Normal(mean, log_std.exp()).rsample()
        action = self._transform(latent)
        mean_action = self._transform(mean)
        log_probability = self._log_probability(latent, mean, log_std)
        return PolicyActionSample4D(action, log_probability, mean_action)


class PolicyTwinCritic4D(nn.Module):
    """Twin critics over the frozen observation embedding and exact 4-D action."""

    def __init__(self, config: PolicySAC4DConfig) -> None:
        super().__init__()
        width = config.observation_embedding_dim + POLICY_ACTION_DIM_4D
        self.q1 = _mlp(width, config.hidden_sizes, 1)
        self.q2 = _mlp(width, config.hidden_sizes, 1)

    def forward(self, embedding: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        if embedding.ndim != 2 or action.ndim != 2:
            raise PolicySAC4DContractError("critic inputs must be batched matrices")
        if action.shape != (embedding.shape[0], POLICY_ACTION_DIM_4D):
            raise PolicySAC4DContractError("critic requires embedding plus exact 4-D action")
        value = torch.cat((embedding, action), dim=-1)
        return self.q1(value), self.q2(value)


@dataclass(frozen=True)
class ReplayBatch4D:
    observation_embeddings: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observation_embeddings: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return asdict(self)


class PolicyReplayBuffer4D:
    """Exact-resume circular replay containing branch-local embeddings only."""

    def __init__(self, capacity: int, config: PolicySAC4DConfig) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise PolicySAC4DContractError("replay capacity must be positive")
        self.capacity = capacity
        self.config_fingerprint = config.fingerprint()
        self.observation_embedding_dim = config.observation_embedding_dim
        self.observation_embeddings = np.empty((capacity, self.observation_embedding_dim), np.float32)
        self.actions = np.empty((capacity, POLICY_ACTION_DIM_4D), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.next_observation_embeddings = np.empty((capacity, self.observation_embedding_dim), np.float32)
        self.terminated = np.empty((capacity, 1), np.float32)
        self.truncated = np.empty((capacity, 1), np.float32)
        self.position = 0
        self.size = 0
        self.total_inserted = 0
        self.rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation_embedding: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation_embedding: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        obs = np.asarray(observation_embedding, dtype=np.float32)
        nxt = np.asarray(next_observation_embedding, dtype=np.float32)
        act = np.asarray(action, dtype=np.float32)
        if obs.shape != (self.observation_embedding_dim,) or nxt.shape != obs.shape:
            raise PolicySAC4DContractError("replay embedding shape mismatch")
        if act.shape != (POLICY_ACTION_DIM_4D,):
            raise PolicySAC4DContractError("replay accepts exactly one 4-D policy action")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(nxt)) or not np.all(np.isfinite(act)) or not math.isfinite(float(reward)):
            raise PolicySAC4DContractError("replay transition must be finite")
        if np.any(np.abs(act[:3]) > 1.0) or not 0.0 <= float(act[3]) <= 1.0:
            raise PolicySAC4DContractError("replay action violates XYZ/gripper ranges")
        row = self.position
        self.observation_embeddings[row] = obs
        self.actions[row] = act
        self.rewards[row, 0] = float(reward)
        self.next_observation_embeddings[row] = nxt
        self.terminated[row, 0] = float(bool(terminated))
        self.truncated[row, 0] = float(bool(truncated))
        self.position = (row + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_inserted += 1

    def sample(self, batch_size: int) -> ReplayBatch4D:
        if type(batch_size) is not int or batch_size <= 0 or self.size < batch_size:
            raise PolicySAC4DContractError("replay does not contain the requested batch")
        rows = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch4D(
            self.observation_embeddings[rows].copy(),
            self.actions[rows].copy(),
            self.rewards[rows].copy(),
            self.next_observation_embeddings[rows].copy(),
            self.terminated[rows].copy(),
            self.truncated[rows].copy(),
        )

    def state_dict(self) -> dict[str, Any]:
        size = self.size
        return {
            "schema": POLICY_REPLAY_4D_SCHEMA,
            "config_fingerprint": self.config_fingerprint,
            "capacity": self.capacity,
            "observation_embedding_dim": self.observation_embedding_dim,
            "action_schema": POLICY_ACTION_SCHEMA_4D,
            "action_dim": POLICY_ACTION_DIM_4D,
            "position": self.position,
            "size": size,
            "total_inserted": self.total_inserted,
            "observation_embeddings": self.observation_embeddings[:size].copy(),
            "actions": self.actions[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "next_observation_embeddings": self.next_observation_embeddings[:size].copy(),
            "terminated": self.terminated[:size].copy(),
            "truncated": self.truncated[:size].copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "her": {"enabled": False, "reason": GoalRelabelingStatus.DISABLED_NO_AUTHORITATIVE_GOAL_CONTRACT.value},
            "her_force": {"enabled": False, "reason": GoalRelabelingStatus.DISABLED_NO_AUTHORITATIVE_GOAL_CONTRACT.value},
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        exact = {
            "schema": POLICY_REPLAY_4D_SCHEMA,
            "config_fingerprint": self.config_fingerprint,
            "capacity": self.capacity,
            "observation_embedding_dim": self.observation_embedding_dim,
            "action_schema": POLICY_ACTION_SCHEMA_4D,
            "action_dim": POLICY_ACTION_DIM_4D,
        }
        for name, expected in exact.items():
            if state.get(name) != expected:
                raise PolicySAC4DContractError(f"replay checkpoint mismatch for {name}")
        size, position = int(state["size"]), int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise PolicySAC4DContractError("invalid replay cursor")
        fields = {
            "observation_embeddings": (size, self.observation_embedding_dim),
            "actions": (size, POLICY_ACTION_DIM_4D),
            "rewards": (size, 1),
            "next_observation_embeddings": (size, self.observation_embedding_dim),
            "terminated": (size, 1),
            "truncated": (size, 1),
        }
        for name, shape in fields.items():
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise PolicySAC4DContractError(f"invalid replay field: {name}")
        action = np.asarray(state["actions"], dtype=np.float32)
        if action.size and (np.any(np.abs(action[:, :3]) > 1.0) or np.any(action[:, 3] < 0.0) or np.any(action[:, 3] > 1.0)):
            raise PolicySAC4DContractError("checkpoint replay action violates 4-D ranges")
        for feature in ("her", "her_force"):
            metadata = state.get(feature)
            if not isinstance(metadata, Mapping) or bool(metadata.get("enabled")):
                raise PolicySAC4DContractError(f"{feature} cannot be enabled without branch authority")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.validate_state_dict(state)
        size = int(state["size"])
        for name in (
            "observation_embeddings",
            "actions",
            "rewards",
            "next_observation_embeddings",
            "terminated",
            "truncated",
        ):
            getattr(self, name)[:size] = np.asarray(state[name], dtype=np.float32)
        self.size = size
        self.position = int(state["position"])
        self.total_inserted = int(state["total_inserted"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class PolicySACAgent4D:
    """Twin-Q SAC learner bound to one exact branch/config/asset fingerprint."""

    def __init__(self, config: PolicySAC4DConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise PolicySAC4DContractError("CUDA requested but unavailable")
        torch.manual_seed(config.seed)
        self.actor = PolicyActor4D(config).to(self.device)
        self.critics = PolicyTwinCritic4D(config).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.target_critics.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_learning_rate)
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_learning_rate)
        self.update_count = 0
        self._bc_handoff_applied = False
        self._bc_policy: HighLevelBCPolicy | None = None

    @property
    def bc_handoff_applied(self) -> bool:
        return self._bc_handoff_applied

    def apply_bc_handoff(self, policy: HighLevelBCPolicy) -> str:
        """Exactly transplant BC heads and reset actor optimizer moments."""

        self.actor.transplant_bc_action_heads(policy)
        policy.requires_grad_(False)
        policy.eval()
        self._bc_policy = policy
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self._bc_handoff_applied = True
        return self.config.bc_handoff.fingerprint()

    def _require_bc_handoff(self) -> None:
        if not self._bc_handoff_applied:
            raise PolicySAC4DContractError(
                "BC handoff must be applied before SAC action/update/checkpoint"
            )

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def select_action(self, embedding: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        self._require_bc_handoff()
        value = np.asarray(embedding, dtype=np.float32)
        if value.shape != (self.config.observation_embedding_dim,) or not np.all(np.isfinite(value)):
            raise PolicySAC4DContractError("invalid policy observation embedding")
        with torch.no_grad():
            sample = self.actor.sample(
                torch.from_numpy(value).to(self.device).unsqueeze(0),
                deterministic=deterministic,
            )
            selected = sample.mean_action if deterministic else sample.action
        return selected.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _tensor(self, batch: Mapping[str, np.ndarray], name: str) -> Tensor:
        if name not in batch:
            raise PolicySAC4DContractError(f"missing SAC batch field: {name}")
        return torch.as_tensor(batch[name], dtype=torch.float32, device=self.device)

    def update(self, batch: ReplayBatch4D | Mapping[str, np.ndarray]) -> dict[str, float]:
        self._require_bc_handoff()
        values = batch.as_mapping() if isinstance(batch, ReplayBatch4D) else batch
        obs = self._tensor(values, "observation_embeddings")
        action = self._tensor(values, "actions")
        reward = self._tensor(values, "rewards")
        nxt = self._tensor(values, "next_observation_embeddings")
        terminated = self._tensor(values, "terminated")
        truncated = self._tensor(values, "truncated")
        batch_size = obs.shape[0]
        if obs.shape != (batch_size, self.config.observation_embedding_dim) or nxt.shape != obs.shape:
            raise PolicySAC4DContractError("SAC embedding batch shape mismatch")
        if action.shape != (batch_size, POLICY_ACTION_DIM_4D):
            raise PolicySAC4DContractError("SAC action batch must be exactly 4-D")
        if reward.shape != (batch_size, 1) or terminated.shape != reward.shape or truncated.shape != reward.shape:
            raise PolicySAC4DContractError("SAC scalar batch fields must be [batch,1]")
        if not all(bool(torch.isfinite(value).all()) for value in (obs, action, reward, nxt, terminated, truncated)):
            raise PolicySAC4DContractError("SAC batch contains non-finite values")
        if bool((action[:, :3].abs() > 1.0).any()) or bool(
            (action[:, 3] < 0.0).any()
        ) or bool((action[:, 3] > 1.0).any()):
            raise PolicySAC4DContractError(
                "SAC data action violates XYZ/gripper 4-D ranges"
            )
        for name, value in (("terminated", terminated), ("truncated", truncated)):
            if bool(((value != 0.0) & (value != 1.0)).any()):
                raise PolicySAC4DContractError(f"{name} must be a binary replay flag")

        with torch.no_grad():
            next_sample = self.actor.sample(nxt)
            target_q1, target_q2 = self.target_critics(nxt, next_sample.action)
            target_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_sample.log_probability
            target = reward + self.config.gamma * (1.0 - terminated) * target_value

        q1, q2 = self.critics(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()

        self.critics.requires_grad_(False)
        sample = self.actor.sample(obs)
        actor_q1, actor_q2 = self.critics(obs, sample.action)
        actor_loss = (self.alpha.detach() * sample.log_probability - torch.minimum(actor_q1, actor_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (sample.log_probability.detach() + self.config.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_gradient = torch.nn.utils.clip_grad_norm_([self.log_alpha], self.config.gradient_clip)
        self.alpha_optimizer.step()

        with torch.no_grad():
            for source, target_parameter in zip(
                self.critics.parameters(), self.target_critics.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.config.tau).add_(source, alpha=self.config.tau)
        self.update_count += 1
        metrics = {
            "loss/critic": float(critic_loss.detach()),
            "loss/actor": float(actor_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "temperature/alpha": float(self.alpha.detach()),
            "policy/entropy": float(-sample.log_probability.detach().mean()),
            "q/data_mean": float(torch.cat((q1.detach(), q2.detach()), dim=1).mean()),
            "gradient/actor_norm": float(actor_gradient),
            "gradient/critic_norm": float(critic_gradient),
            "gradient/alpha_norm": float(alpha_gradient),
            "replay/truncated_fraction": float(truncated.mean()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite 4-D SAC metrics: {metrics}")
        return metrics

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != POLICY_SAC_4D_SCHEMA:
            raise PolicySAC4DContractError("SAC checkpoint schema mismatch")
        if state.get("config_fingerprint") != self.config.fingerprint():
            raise PolicySAC4DContractError("SAC checkpoint config/asset fingerprint mismatch")

    def state_dict(self) -> dict[str, Any]:
        self._require_bc_handoff()
        assert self._bc_policy is not None
        return {
            "schema": POLICY_SAC_4D_SCHEMA,
            "config": self.config.payload(),
            "config_fingerprint": self.config.fingerprint(),
            "bc_handoff_applied": True,
            "bc_handoff_fingerprint": self.config.bc_handoff.fingerprint(),
            "bc_policy_semantic_fingerprint": self._bc_policy.config.semantic_fingerprint(),
            "bc_policy_state": {
                name: value.detach().cpu().clone()
                for name, value in self._bc_policy.state_dict().items()
            },
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, bc_policy: HighLevelBCPolicy
    ) -> None:
        self.validate_state_dict(state)
        if state.get("bc_handoff_applied") is not True or state.get(
            "bc_handoff_fingerprint"
        ) != self.config.bc_handoff.fingerprint():
            raise PolicySAC4DContractError("checkpoint lacks exact BC handoff provenance")
        if not isinstance(bc_policy, HighLevelBCPolicy):
            raise PolicySAC4DContractError("checkpoint reload requires a BC policy instance")
        if state.get("bc_policy_semantic_fingerprint") != bc_policy.config.semantic_fingerprint():
            raise PolicySAC4DContractError("checkpoint BC architecture/semantics differ")
        bc_state = state.get("bc_policy_state")
        if not isinstance(bc_state, Mapping):
            raise PolicySAC4DContractError("checkpoint lacks frozen BC producer state")
        self.config.bc_handoff.assert_checkpoint_state(bc_state)
        bc_policy.load_state_dict(bc_state, strict=True)
        self.config.bc_handoff.assert_policy(bc_policy)
        bc_policy.requires_grad_(False)
        bc_policy.eval()
        self._bc_policy = bc_policy
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.target_critics.load_state_dict(state["target_critics"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.update_count = int(state["update_count"])
        self._bc_handoff_applied = True
        for optimizer in (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer):
            for optimizer_state in optimizer.state.values():
                for name, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[name] = value.to(self.device)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``load_full_checkpoint`` maps tensor state to the learner device.  On a
    # CUDA learner that also maps the serialized CPU RNG ByteTensor to CUDA,
    # while ``torch.set_rng_state`` strictly requires a CPU ByteTensor.  RNG
    # state is device-independent checkpoint metadata, so explicitly return
    # it to CPU before handing it to either generator API.
    torch.set_rng_state(state["torch_cpu"].detach().to(device="cpu"))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.detach().to(device="cpu") for value in state["torch_cuda"]]
        )


def build_full_checkpoint(
    agent: PolicySACAgent4D,
    replay: PolicyReplayBuffer4D,
    *,
    global_step: int,
    curriculum_state: Mapping[str, Any],
) -> dict[str, Any]:
    if type(global_step) is not int or global_step < 0:
        raise PolicySAC4DContractError("global step must be a non-negative integer")
    agent.config.asset.assert_exact()
    agent._require_bc_handoff()
    if replay.config_fingerprint != agent.config.fingerprint():
        raise PolicySAC4DContractError("agent and replay config fingerprints differ")
    # JSON validation rejects NaN/Inf and non-serializable curriculum state.
    curriculum = json.loads(json.dumps(dict(curriculum_state), allow_nan=False))
    return {
        "schema": POLICY_CHECKPOINT_4D_SCHEMA,
        "action_schema": POLICY_ACTION_SCHEMA_4D,
        "action_dim": POLICY_ACTION_DIM_4D,
        "config": agent.config.payload(),
        "config_fingerprint": agent.config.fingerprint(),
        "asset": agent.config.asset.payload(),
        "asset_fingerprint": agent.config.asset.fingerprint(),
        "agent": agent.state_dict(),
        "replay": replay.state_dict(),
        "curriculum_state": curriculum,
        "global_step": global_step,
        "rng_state": _capture_rng_state(),
    }


def save_full_checkpoint(
    path: Path | str,
    agent: PolicySACAgent4D,
    replay: PolicyReplayBuffer4D,
    *,
    global_step: int,
    curriculum_state: Mapping[str, Any],
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        build_full_checkpoint(
            agent, replay, global_step=global_step, curriculum_state=curriculum_state
        ),
        temporary,
    )
    os.replace(temporary, destination)
    return _file_sha256(destination)


def load_full_checkpoint(
    path: Path | str,
    agent: PolicySACAgent4D,
    replay: PolicyReplayBuffer4D,
    *,
    bc_policy: HighLevelBCPolicy,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if expected_sha256 is not None:
        if not _SHA256.fullmatch(expected_sha256) or _file_sha256(source) != expected_sha256:
            raise PolicySAC4DContractError("checkpoint file fingerprint mismatch")
    payload = torch.load(source, map_location=agent.device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != POLICY_CHECKPOINT_4D_SCHEMA:
        raise PolicySAC4DContractError("full checkpoint schema mismatch")
    fingerprint = agent.config.fingerprint()
    expected = {
        "action_schema": POLICY_ACTION_SCHEMA_4D,
        "action_dim": POLICY_ACTION_DIM_4D,
        "config_fingerprint": fingerprint,
        "asset_fingerprint": agent.config.asset.fingerprint(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise PolicySAC4DContractError(f"full checkpoint mismatch for {name}")
    agent.config.asset.assert_exact()
    agent.validate_state_dict(payload["agent"])
    bc_state = payload["agent"].get("bc_policy_state")
    if not isinstance(bc_state, Mapping):
        raise PolicySAC4DContractError("checkpoint lacks frozen BC producer state")
    agent.config.bc_handoff.assert_checkpoint_state(bc_state)
    replay.validate_state_dict(payload["replay"])
    if replay.config_fingerprint != fingerprint:
        raise PolicySAC4DContractError("checkpoint replay belongs to another training contract")
    global_step = int(payload["global_step"])
    if global_step < 0 or not isinstance(payload.get("curriculum_state"), Mapping):
        raise PolicySAC4DContractError("checkpoint counters/curriculum are invalid")
    # Validate every authority before mutating the target objects.
    agent.load_state_dict(payload["agent"], bc_policy=bc_policy)
    replay.load_state_dict(payload["replay"])
    _restore_rng_state(payload["rng_state"])
    return {
        "global_step": global_step,
        "curriculum_state": dict(payload["curriculum_state"]),
        "config_fingerprint": fingerprint,
        "asset_fingerprint": agent.config.asset.fingerprint(),
    }


__all__ = [
    "BCHandoffBinding",
    "GoalRelabelingBinding",
    "GoalRelabelingStatus",
    "M2_VALIDATED_ASSET_RELATIVE_PATH",
    "M2_VALIDATED_ASSET_SHA256",
    "POLICY_ACTION_DIM_4D",
    "POLICY_CHECKPOINT_4D_SCHEMA",
    "POLICY_REPLAY_4D_SCHEMA",
    "POLICY_SAC_4D_SCHEMA",
    "PolicyActor4D",
    "PolicyReplayBuffer4D",
    "PolicySAC4DConfig",
    "PolicySAC4DContractError",
    "PolicySACAgent4D",
    "PolicyTwinCritic4D",
    "ReplayBatch4D",
    "TrainingAssetBinding",
    "bc_action_heads_sha256",
    "bc_embedding_producer_sha256",
    "build_full_checkpoint",
    "load_full_checkpoint",
    "save_full_checkpoint",
]
