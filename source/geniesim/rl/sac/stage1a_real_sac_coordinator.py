# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Environment-agnostic Stage-1A real-row residual-SAC coordinator.

The coordinator deliberately does not launch Isaac or call ``env.step``.  A
runtime runner owns canonical action consumption and may commit a transition
here only after that consumption returned and the following real telemetry was
captured.  The frozen grasp GRU remains the sole gripper authority; SAC owns
only a three-dimensional XYZ residual.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .bc_residual_sac_contract import (
    ResidualActionComposition,
    ResidualAlphaCurriculum,
    compose_bc_and_residual_action,
)
from .human_grasp_gru_bc import (
    HysteresisCalibration,
    HumanGraspGRUBC,
    HumanGraspGRUConfig,
    HumanGraspGRUOutput,
    HumanGraspSequenceInputs,
    load_human_grasp_checkpoint,
    tensor_state_sha256,
    _validate_inputs,
    ROBOT_STATE_DIM,
)
from .keyboard_grasp_contract import canonical_keyboard_grasp_contract
from .residual_her_force import (
    ResidualHERForceConfig,
    prepare_residual_her_force_priorities,
)
from .residual_sac_runtime import (
    load_residual_sac_actor_checkpoint,
    residual_sac_actor_checkpoint_payload,
)
from .stage1a_grasp_reward import Stage1ARewardStep
from .stage1a_exception_safe_transition import ActionConsumptionState
from .stage1a_close_readiness_contract import (
    CLOSE_READINESS_TARGET_SCHEMA_VERSION,
)
from .stage1a_close_readiness_training_contract import (
    CLOSE_READINESS_STUDENT_INITIALIZATION_SCHEMA,
    normalize_close_readiness_feature,
    validate_close_readiness_training_contract,
)
from .stage2_sac import (
    ReplayBuffer,
    SACAgent,
    SACConfig,
    capture_rng_state,
    save_torch_checkpoint,
)


STAGE1A_REAL_COORDINATOR_SCHEMA = "g2_stage1a_real_row_sac_coordinator_v1"
STAGE1A_REAL_LEARNER_CHECKPOINT_SCHEMA = "g2_stage1a_real_row_sac_learner_v1"
STAGE1A_PERIODIC_CHECKPOINT_SCHEMA = "g2_stage1a_periodic_training_checkpoint_v1"
STAGE1A_SAC_GAMMA = 0.9993
CANDIDATE_A_DYNAMICS_AUTHORITY = "CANDIDATE_A"
REPLAY_SOURCE_BC_DEMO = "SOURCE_BC_DEMO"
REPLAY_SOURCE_EXISTING_CLOSE = "SOURCE_EXISTING_CLOSE"
REPLAY_SOURCE_ONLINE_SAC = "SOURCE_ONLINE_SAC"
REPLAY_SOURCES = frozenset(
    (REPLAY_SOURCE_BC_DEMO, REPLAY_SOURCE_EXISTING_CLOSE, REPLAY_SOURCE_ONLINE_SAC)
)
RUNTIME_HARDSTOP_FAILURE = "RUNTIME_HARDSTOP"
REPLAY_STRATEGY_SAC = "SAC"
REPLAY_STRATEGY_HER = "HER"
REPLAY_STRATEGY_HER_FORCE = "HER_FORCE"
REPLAY_STRATEGIES = frozenset(
    (REPLAY_STRATEGY_SAC, REPLAY_STRATEGY_HER, REPLAY_STRATEGY_HER_FORCE)
)
# This is deliberately an auxiliary student-only objective.  It is small
# enough that it cannot become reward authority or alter the three-dimensional
# residual action contract.  The teacher label is never concatenated into the
# actor observation.
CLOSE_READINESS_DISTILLATION_WEIGHT = 0.05
CLOSE_READINESS_BATCH_SIZE = 64


class Stage1ARealSACError(ValueError):
    """Raised before an invalid or non-real row can mutate replay/optimizer."""


class Stage1ACloseReadinessHead(nn.Module):
    """Small deployable-observation predictor for a geometry teacher label.

    The input is the existing frozen-GRU recurrent feature already consumed by
    the residual actor.  It contains only causal RGB-D/robot/previous-action
    information.  The head is intentionally separate from the SAC actor so a
    teacher-only bounded experiment cannot silently modify residual-action
    authority or the frozen GRU/BC path.
    """

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        if type(observation_dim) is not int or observation_dim <= 0:
            raise Stage1ARealSACError("close-readiness observation dimension is invalid")
        hidden = min(64, observation_dim)
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise Stage1ARealSACError("close-readiness observation must be [B,D]")
        return self.network(observation)


class Stage1ACloseReadinessLinearHead(nn.Module):
    """Calibratable linear logit head for the normalized teacher contract.

    This is deliberately separate from the frozen GRU/XYZ/residual actor.
    The smaller capacity matches the offline linear-probe authority and avoids
    teaching a 64-unit MLP to memorize a short episode-disjoint dataset.
    """

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        if type(observation_dim) is not int or observation_dim <= 0:
            raise Stage1ARealSACError("close-readiness observation dimension is invalid")
        self.linear = nn.Linear(observation_dim, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise Stage1ARealSACError("close-readiness observation must be [B,D]")
        return self.linear(observation)


def build_close_readiness_head(
    observation_dim: int, *, architecture: str = "MLP64_RELU_V1"
) -> nn.Module:
    """Build a versioned auxiliary head without altering legacy checkpoints."""

    if architecture == "MLP64_RELU_V1":
        return Stage1ACloseReadinessHead(observation_dim)
    if architecture == "LINEAR_LOGIT_V1":
        return Stage1ACloseReadinessLinearHead(observation_dim)
    raise Stage1ARealSACError("unknown close-readiness head architecture")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise Stage1ARealSACError(f"{name} must be finite [{size}]")
    return np.ascontiguousarray(result.copy())


def radial_metric_residual(normalized_xyz: Any) -> tuple[float, float, float]:
    """Use the strict runtime actor's radial-tanh 4.5-mm parameterization."""

    vector = _finite_vector("normalized SAC action", normalized_xyz, 3).astype(np.float64)
    radius = float(np.linalg.norm(vector))
    maximum = canonical_keyboard_grasp_contract().residual_raw_maximum_norm_m
    if radius <= 1.0e-12:
        metric = np.zeros(3, dtype=np.float64)
    else:
        metric = maximum * math.tanh(radius) * vector / radius
    if not np.isfinite(metric).all() or float(np.linalg.norm(metric)) >= maximum + 1.0e-12:
        raise Stage1ARealSACError("radial residual exceeded the 4.5-mm raw bound")
    return tuple(float(value) for value in metric)


@dataclass(frozen=True)
class Stage1AActionProposal:
    actor_observation: np.ndarray
    normalized_sac_action: np.ndarray
    raw_residual_metric_root_m: tuple[float, float, float]
    bc_action_4d_metric_root_m: tuple[float, float, float, float]
    composition: ResidualActionComposition
    close_probability: float
    feasibility_probability: float
    residual_active: bool

    def __post_init__(self) -> None:
        observation = np.ascontiguousarray(np.asarray(self.actor_observation, dtype=np.float32).copy())
        action = _finite_vector("normalized_sac_action", self.normalized_sac_action, 3)
        observation.flags.writeable = False
        action.flags.writeable = False
        object.__setattr__(self, "actor_observation", observation)
        object.__setattr__(self, "normalized_sac_action", action)
        for name in ("close_probability", "feasibility_probability"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise Stage1ARealSACError(f"{name} must be finite in [0,1]")
        if not isinstance(self.residual_active, bool):
            raise Stage1ARealSACError("residual_active must be bool")


@dataclass(frozen=True)
class Stage1AAcceptedRealRow:
    """A runner-attested post-``env.step`` transition.

    ``canonical_consumption_confirmed`` and ``real_telemetry`` must both be
    true.  Rejected, unknown-consumption, synthetic, or incomplete rows belong
    in diagnostic artifacts, never this API.
    """

    row_id: str
    proposal: Stage1AActionProposal
    next_actor_observation: np.ndarray
    reward_step: Stage1ARewardStep
    reward_env_index: int = 0
    canonical_consumption_confirmed: bool = True
    real_telemetry: bool = True
    safety_measurement_valid: bool = True
    safety_pass: bool = True
    lift: bool = False
    terminated: bool | None = None
    truncated: bool = False
    source: str = REPLAY_SOURCE_ONLINE_SAC
    episode_id: str = "ONLINE_EPISODE"
    phase: str | None = None
    success: bool | None = None
    failure_reason: str = "NONE"
    hardstop: bool = False
    contact: bool | None = None
    bilateral: bool | None = None
    stable: bool | None = None
    dynamics_authority: str = CANDIDATE_A_DYNAMICS_AUTHORITY
    transition_schema_complete: bool = True
    policy_observation_complete: bool = True
    mechanics_only_telemetry: bool = False
    action_consumption: str = ActionConsumptionState.FULLY_CONSUMED.value
    control_step_index: int = 0
    physics_step_start: int = 1
    physics_step_end: int = 10
    consumed_substeps: int = 10
    nominal_substeps: int = 10
    execution_fraction: float = 1.0
    terminal_observation_is_real: bool = True
    camera_timestamp: float = 0.0
    terminal_timestamp: float = 0.020
    camera_age: float = 0.020
    hardstop_joint: str | None = None
    idx83_min_margin: float | None = None
    # Teacher-side quantities are replay metadata only.  They are sampled by
    # the separate student auxiliary head and are never appended to
    # ``proposal.actor_observation`` or passed to the residual actor.
    pre_close_candidate: bool = False
    close_latched_before_supervision: bool = False
    privileged_close_ready_target: bool | None = None
    privileged_close_ready_score: float | None = None
    privileged_close_ready_negative_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage1ACheckpointReceipt:
    learner_path: str
    actor_path: str
    actor_sha256: str
    actor_reload_pass: bool
    learner_reload_pass: bool
    bc_sha256: str


@dataclass(frozen=True)
class Stage1APeriodicCheckpointReceipt:
    path: str
    sha256: str
    accepted_transitions: int
    replay_strategy: str
    reload_pass: bool


class Stage1ARealSACCoordinator:
    """Own frozen GRU inference, real replay and one residual SAC learner."""

    learning_starts = 64
    batch_size = 64
    updates_per_accepted_row = 1

    def __init__(
        self,
        *,
        frozen_bc: HumanGraspGRUBC,
        bc_checkpoint_path: str | Path,
        expected_bc_sha256: str,
        replay_capacity: int = 4096,
        device: str | torch.device = "cpu",
        seed: int = 42,
        close_calibration: HysteresisCalibration | None = None,
        replay_strategy: str = REPLAY_STRATEGY_HER_FORCE,
        residual_actor_checkpoint_path: str | Path | None = None,
        residual_actor_checkpoint_sha256: str | None = None,
        close_readiness_distillation: bool = False,
        close_readiness_training_contract: Mapping[str, Any] | None = None,
        close_readiness_initialization: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(frozen_bc, HumanGraspGRUBC):
            raise Stage1ARealSACError("HumanGraspGRUBC is required")
        path = Path(bc_checkpoint_path).expanduser().resolve()
        if not path.is_file() or _file_sha256(path) != expected_bc_sha256:
            raise Stage1ARealSACError("frozen BC checkpoint hash mismatch")
        frozen_bc.eval()
        for parameter in frozen_bc.parameters():
            parameter.requires_grad_(False)
        self.bc = frozen_bc
        self.bc_checkpoint_path = path
        self.bc_sha256 = expected_bc_sha256
        self.bc_state_sha256 = tensor_state_sha256(frozen_bc)
        self.device = torch.device(device)
        self.hidden: torch.Tensor | None = None
        self.close_calibration = close_calibration or HysteresisCalibration(
            open_threshold=0.30,
            close_threshold=0.70,
            validation_split_fingerprint="DIRECT_TEST_DEFAULT",
            close_precision=0.0,
            close_recall=0.0,
            close_f1=0.0,
        )
        self._closed = False
        observation_dim = frozen_bc.config.gru_hidden_dim
        self.agent = SACAgent(
            SACConfig(
                observation_dim=observation_dim,
                action_dim=3,
                action_mask=(1.0, 1.0, 1.0),
                gamma=STAGE1A_SAC_GAMMA,
                initial_alpha=0.01,
                pose_auxiliary_weight=0.1,
                seed=seed,
            ),
            device=self.device,
        )
        if not isinstance(close_readiness_distillation, bool):
            raise Stage1ARealSACError("close_readiness_distillation must be bool")
        self.close_readiness_distillation = close_readiness_distillation
        if close_readiness_training_contract is not None and not close_readiness_distillation:
            raise Stage1ARealSACError(
                "close-readiness training contract requires opt-in distillation"
            )
        self.close_readiness_training_contract = (
            validate_close_readiness_training_contract(
                close_readiness_training_contract,
                observation_dim=observation_dim,
            )
            if close_readiness_training_contract is not None
            else None
        )
        self.close_readiness_head_architecture = (
            str(self.close_readiness_training_contract["student_head_architecture"])
            if self.close_readiness_training_contract is not None
            else "MLP64_RELU_V1"
        )
        self.close_readiness_head = build_close_readiness_head(
            observation_dim, architecture=self.close_readiness_head_architecture
        ).to(self.device)
        self.close_readiness_optimizer = torch.optim.Adam(
            self.close_readiness_head.parameters(), lr=3.0e-4
        )
        self._close_readiness_rng = np.random.default_rng(seed + 1729)
        self._close_readiness_observations: list[np.ndarray] = []
        self._close_readiness_scores: list[float] = []
        self._close_readiness_targets: list[bool] = []
        self._student_close_readiness_scores: list[float] = []
        self._close_distillation_update_count = 0
        self._close_distillation_last_metrics: dict[str, float] = {
            "loss/close_distill": 0.0,
            "loss/close_distill_weighted": 0.0,
            "close_distill/student_mean": 0.0,
            "close_distill/teacher_mean": 0.0,
            "close_distill/active": 0.0,
        }
        self._close_readiness_initialization_sha256: str | None = None
        if close_readiness_initialization is not None:
            if self.close_readiness_training_contract is None:
                raise Stage1ARealSACError(
                    "close-readiness initialization requires a validated contract"
                )
            if (
                close_readiness_initialization.get("schema")
                != CLOSE_READINESS_STUDENT_INITIALIZATION_SCHEMA
                or close_readiness_initialization.get("contract_sha256")
                != self.close_readiness_training_contract["contract_sha256"]
                or int(close_readiness_initialization.get("observation_dim", -1))
                != observation_dim
                or close_readiness_initialization.get("head_architecture")
                != self.close_readiness_head_architecture
                or int(close_readiness_initialization.get("student_privileged_input_count", -1))
                != 0
            ):
                raise Stage1ARealSACError(
                    "close-readiness initialization contract mismatch"
                )
            state_dict = close_readiness_initialization.get("head")
            if not isinstance(state_dict, Mapping):
                raise Stage1ARealSACError("close-readiness initialization head missing")
            self.close_readiness_head.load_state_dict(state_dict, strict=True)
            self._close_readiness_initialization_sha256 = _module_sha256(
                self.close_readiness_head
            )
        if replay_strategy not in REPLAY_STRATEGIES:
            raise Stage1ARealSACError("unknown Stage-1A replay strategy")
        if replay_strategy == REPLAY_STRATEGY_HER:
            raise Stage1ARealSACError(
                "ordinary HER requires a validated goal-conditioned Stage-1A "
                "actor observation; GRU hidden features are not goal authority"
            )
        self.replay_strategy = replay_strategy
        if (residual_actor_checkpoint_path is None) != (
            residual_actor_checkpoint_sha256 is None
        ):
            raise Stage1ARealSACError(
                "residual actor checkpoint path/hash must be supplied together"
            )
        if residual_actor_checkpoint_path is None:
            self.agent.zero_initialize_residual_mean_head()
            self.actor_initialization = {
                "authority": "ZERO_INITIALIZED_MEAN_HEAD",
                "checkpoint_path": None,
                "checkpoint_sha256": None,
            }
        else:
            runtime = load_residual_sac_actor_checkpoint(
                residual_actor_checkpoint_path,
                expected_sha256=str(residual_actor_checkpoint_sha256),
                expected_human_grasp_checkpoint_sha256=expected_bc_sha256,
                expected_observation_dim=observation_dim,
                device=self.device,
            )
            self.agent.actor.load_state_dict(runtime.actor.state_dict(), strict=True)
            self.actor_initialization = {
                "authority": "VERIFIED_RESIDUAL_ACTOR_CHECKPOINT",
                "checkpoint_path": runtime.receipt.checkpoint_path,
                "checkpoint_sha256": runtime.receipt.checkpoint_sha256,
                "actor_state_sha256": runtime.receipt.actor_state_sha256,
            }
        self.replay = ReplayBuffer(
            replay_capacity, observation_dim, 3, seed=seed
        )
        self.replay.configure_prioritization(
            enabled=replay_strategy == REPLAY_STRATEGY_HER_FORCE
        )
        self.her_force = ResidualHERForceConfig()
        self.alpha = ResidualAlphaCurriculum()
        if not math.isclose(self.alpha.alpha, 0.10, abs_tol=1.0e-12):
            raise Stage1ARealSACError("Stage-1A residual alpha must be 0.10")
        self.accepted_transitions = 0
        self.sac_update_count = 0
        self._row_ids: set[str] = set()
        self._replay_provenance: list[dict[str, Any]] = []
        self._residual_norm_mm: list[float] = []
        self._reward_sums = {
            name: 0.0
            for name in (
                "progress", "lateral", "single", "bilateral", "stable", "success",
                "residual_penalty", "smoothness_penalty", "time_penalty", "safety_penalty",
                "cosine", "oscillation_penalty", "worsening_penalty",
                "milestone_20mm", "milestone_18mm", "milestone_16mm", "stable_hold",
                "hover_penalty", "single_contact_dwell_penalty",
            )
        }
        self._counts = {
            "hover": 0, "single_contact_dwell": 0, "bilateral": 0, "stable": 0,
            "success": 0, "contact_loss_termination": 0, "bound_violation": 0,
            "gripper_authority_violation": 0,
            "contact": 0, "v3_active": 0, "temporal_valid": 0,
            "progressing": 0, "static_hover": 0, "oscillatory_hover": 0,
            "worsening": 0, "ambiguous": 0, "cosine_sustained": 0,
            "cosine_with_progress": 0,
        }
        self._v3_active_nominal_residual_mm: list[float] = []
        self._v3_hover_durations_steps: list[int] = []
        self._current_v3_hover_steps = 0
        self._optimizer_metrics: dict[str, float] = {}
        self._original_transition_count = 0
        self._her_transition_count = 0
        self._her_force_transition_count = 0
        self._her_force_activated_count = 0
        self._max_residual_streak = 0
        self._current_max_residual_streak = 0
        self._episode_milestones: dict[str, int] = {}
        self.force_reward_farming = False
        self.recontact_farming = False

    @classmethod
    def from_bc_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        replay_capacity: int = 4096,
        device: str | torch.device = "cpu",
        seed: int = 42,
        replay_strategy: str = REPLAY_STRATEGY_HER_FORCE,
        residual_actor_checkpoint_path: str | Path | None = None,
        residual_actor_checkpoint_sha256: str | None = None,
        close_readiness_distillation: bool = False,
        close_readiness_training_contract: Mapping[str, Any] | None = None,
        close_readiness_initialization: Mapping[str, Any] | None = None,
    ) -> "Stage1ARealSACCoordinator":
        source = Path(path).expanduser().resolve()
        if not source.is_file() or _file_sha256(source) != expected_sha256:
            raise Stage1ARealSACError("frozen BC checkpoint hash mismatch")
        try:
            model, receipt = load_human_grasp_checkpoint(source, device=device)
        except Exception as strict_error:
            # The immutable 2026-09-22 production GRU predates the metadata-only
            # Candidate-A/25-Hz contract revision.  Its tensor architecture is
            # unchanged.  This compatibility path is intentionally local to
            # the bounded smoke, preserves the source file hash, requires the
            # exact v2 schema, strict tensor loading, and the recorded tensor
            # hash.  It does not rewrite or promote the checkpoint.
            try:
                payload = torch.load(source, map_location=device, weights_only=True)
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("schema") != "g2_human_grasp_wrist_gru_bc_checkpoint_v2"
                ):
                    raise Stage1ARealSACError(
                        "unsupported frozen BC schema"
                    ) from strict_error
                model = HumanGraspGRUBC(
                    HumanGraspGRUConfig(**dict(payload.get("config", {})))
                ).to(device)
                model.load_state_dict(payload["state_dict"], strict=True)
                if payload.get("state_sha256") != tensor_state_sha256(model):
                    raise Stage1ARealSACError(
                        "legacy frozen BC tensor hash mismatch"
                    )
                calibration = HysteresisCalibration(
                    **dict(payload.get("hysteresis_calibration", {}))
                )
                receipt = {
                    "file_sha256": _file_sha256(source),
                    "state_sha256": payload["state_sha256"],
                    "calibration": calibration,
                    "compatibility_mode": "IMMUTABLE_V2_TENSORS_METADATA_ONLY_V3_BRIDGE",
                }
            except Stage1ARealSACError:
                raise
            except Exception as error:
                raise Stage1ARealSACError(
                    "frozen BC compatibility load failed"
                ) from error
        if receipt["file_sha256"] != expected_sha256:
            raise Stage1ARealSACError("loaded BC receipt hash mismatch")
        return cls(
            frozen_bc=model,
            bc_checkpoint_path=source,
            expected_bc_sha256=expected_sha256,
            replay_capacity=replay_capacity,
            device=device,
            seed=seed,
            close_calibration=receipt["calibration"],
            replay_strategy=replay_strategy,
            residual_actor_checkpoint_path=residual_actor_checkpoint_path,
            residual_actor_checkpoint_sha256=residual_actor_checkpoint_sha256,
            close_readiness_distillation=close_readiness_distillation,
            close_readiness_training_contract=close_readiness_training_contract,
            close_readiness_initialization=close_readiness_initialization,
        )

    def reset_episode(self) -> None:
        if self._current_v3_hover_steps:
            self._v3_hover_durations_steps.append(self._current_v3_hover_steps)
            self._current_v3_hover_steps = 0
        self.hidden = None
        self._closed = False
        self._episode_milestones = {}

    def assert_bc_frozen(self) -> None:
        if _file_sha256(self.bc_checkpoint_path) != self.bc_sha256:
            raise Stage1ARealSACError("frozen BC file changed")
        if tensor_state_sha256(self.bc) != self.bc_state_sha256:
            raise Stage1ARealSACError("frozen BC weights changed")
        if self.bc.training or any(parameter.requires_grad for parameter in self.bc.parameters()):
            raise Stage1ARealSACError("frozen BC trainability changed")

    def _stream_forward(
        self, inputs: HumanGraspSequenceInputs
    ) -> HumanGraspGRUOutput:
        """Run one causal row without changing the frozen model source.

        The immutable model's public sequence validator requires every tensor
        passed as a standalone sequence to begin with a reset row.  A live
        one-row continuation instead carries an explicit hidden state.  We
        validate the row with an equivalent reset-mask copy, then execute the
        exact frozen modules while applying the real mask to the carried
        hidden state.  No parameter or input value is changed.
        """

        validation_inputs = replace(
            inputs,
            hidden_reset_mask=torch.ones_like(inputs.hidden_reset_mask),
        )
        batch, time = _validate_inputs(validation_inputs, self.bc.config)
        if (batch, time) != (1, 1):
            raise Stage1ARealSACError("streaming GRU requires exactly [1,1]")
        wrist_visual = self.bc.vision(
            inputs.right_wrist_rgb,
            inputs.right_wrist_depth_m,
            inputs.right_wrist_depth_valid,
        )
        state = torch.cat(
            (
                inputs.ee_pose_robot_root_m_xyzw,
                inputs.right_arm_joint_position_rad,
                inputs.right_arm_joint_velocity_rad_s,
                inputs.current_gripper_state,
                inputs.previous_policy_action_4d_metric_root_m,
            ),
            dim=-1,
        )
        if state.shape[-1] != ROBOT_STATE_DIM:
            raise Stage1ARealSACError("streaming robot-state width mismatch")
        fused = self.bc.fusion(
            torch.cat((wrist_visual, self.bc.state_encoder(state)), dim=-1)
        )
        hidden = self.hidden
        if hidden is None:
            hidden = torch.zeros(
                self.bc.config.gru_layers,
                batch,
                self.bc.config.gru_hidden_dim,
                device=fused.device,
                dtype=fused.dtype,
            )
        reset = inputs.hidden_reset_mask[:, 0]
        if bool(reset.any()):
            hidden = hidden * (~reset).to(hidden.dtype).view(1, batch, 1)
        features, final_hidden = self.bc.gru(fused, hidden)
        bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
        xyz_normalized = torch.tanh(self.bc.xyz_head(features))
        norm = torch.linalg.vector_norm(xyz_normalized, dim=-1, keepdim=True)
        projection_applied = norm[..., 0] > 1.0
        xyz = (
            xyz_normalized
            * torch.clamp(1.0 / norm.clamp_min(1.0e-12), max=1.0)
            * bound
        )
        close_logit = self.bc.close_head(features)
        close_probability = torch.sigmoid(close_logit)
        feasibility_logit = self.bc.feasibility_head(features)
        feasibility_probability = torch.sigmoid(feasibility_logit)
        return HumanGraspGRUOutput(
            xyz_action_robot_root_m=xyz,
            close_logit=close_logit,
            close_probability=close_probability,
            feasibility_logit=feasibility_logit,
            feasibility_probability=feasibility_probability,
            action_4d=torch.cat((xyz, close_probability), dim=-1),
            xyz_projection_applied=projection_applied,
            recurrent_features=features,
            final_hidden=final_hidden,
        )

    def propose(
        self,
        inputs: HumanGraspSequenceInputs,
        *,
        deterministic: bool = False,
        residual_active: bool = True,
        nominal_action_override_4d_metric_root_m: Any | None = None,
        gripper_authority_active: bool = True,
    ) -> Stage1AActionProposal:
        self.assert_bc_frozen()
        with torch.inference_mode():
            output = self._stream_forward(inputs)
        if output.recurrent_features.shape[:2] != (1, 1):
            raise Stage1ARealSACError("real coordinator accepts one env and one control row")
        self.hidden = output.final_hidden.detach()
        feature = output.recurrent_features[0, -1].detach().cpu().numpy().astype(np.float32)
        raw_action = tuple(
            float(value) for value in output.action_4d[0, -1].detach().cpu().tolist()
        )
        close_probability = raw_action[3]
        if not isinstance(gripper_authority_active, bool):
            raise Stage1ARealSACError("gripper_authority_active must be bool")
        if nominal_action_override_4d_metric_root_m is not None:
            override = _finite_vector(
                "nominal_action_override_4d_metric_root_m",
                nominal_action_override_4d_metric_root_m,
                4,
            )
            if gripper_authority_active:
                raise Stage1ARealSACError(
                    "nominal override cannot grant GRU gripper authority"
                )
            if float(np.linalg.norm(override[:3])) > 0.0045 + 1.0e-12:
                raise Stage1ARealSACError("nominal override exceeds 4.5 mm")
            if float(override[3]) != 0.0:
                raise Stage1ARealSACError(
                    "non-GRU nominal owner must preserve OPEN"
                )
            bc_action = tuple(float(value) for value in override)
        else:
            if not gripper_authority_active:
                # The simplified runtime CLOSE gate deliberately keeps the
                # frozen GRU as the Cartesian nominal owner while removing
                # p_close from actuation authority.  Preserve the causal GRU
                # forward/hidden-state update and its XYZ output, but force
                # the typed abstract gripper channel OPEN.  The caller may
                # later replace the complete proposal with the canonical
                # one-shot CLOSE receipt; this method never writes a finger
                # target or mutates the GRU CLOSE latch in logging-only mode.
                bc_action = (*raw_action[:3], 0.0)
            elif self.close_calibration.mode == "EDGE_SINGLE_EVENT":
                # A causal CLOSE-edge model emits one event.  Persistence belongs
                # to the runtime latch, never to repeated model positives, and an
                # episode reset is the sole OPEN transition for this mode.
                if not self._closed and close_probability >= self.close_calibration.close_threshold:
                    self._closed = True
            else:
                if close_probability >= self.close_calibration.close_threshold:
                    self._closed = True
                elif close_probability <= self.close_calibration.open_threshold:
                    self._closed = False
                bc_action = (*raw_action[:3], 1.0 if self._closed else 0.0)
            if gripper_authority_active and self.close_calibration.mode == "EDGE_SINGLE_EVENT":
                bc_action = (*raw_action[:3], 1.0 if self._closed else 0.0)
        if not isinstance(residual_active, bool):
            raise Stage1ARealSACError("residual_active must be bool")
        if residual_active:
            normalized = self.agent.select_action(feature, deterministic=deterministic)
            raw_metric = radial_metric_residual(normalized)
        else:
            # The applied action is the replay authority.  Outside the
            # 15--22 mm local grasp band SAC is inactive, so both the action
            # stored in replay and its metric contribution must be exactly
            # zero rather than an unconsumed actor sample.
            normalized = np.zeros(3, dtype=np.float32)
            raw_metric = (0.0, 0.0, 0.0)
        composition = compose_bc_and_residual_action(
            bc_action_metric_root_m=bc_action,
            raw_sac_residual_metric_root_m=raw_metric,
            alpha=self.alpha,
        )
        if composition.final_gripper_probability != bc_action[3]:
            raise Stage1ARealSACError("SAC changed BC gripper authority")
        return Stage1AActionProposal(
            feature,
            normalized,
            raw_metric,
            bc_action,
            composition,
            close_probability,
            float(output.feasibility_probability[0, -1, 0].detach().cpu().item()),
            residual_active,
        )

    def encode_terminal_observation(
        self, inputs: HumanGraspSequenceInputs
    ) -> np.ndarray:
        """Encode one real terminal observation without sampling an action.

        The recurrent state advances through the actual terminal sensor row,
        but CLOSE latch and SAC actor state are untouched.  Episode reset
        immediately follows a hard-stop commit.
        """

        self.assert_bc_frozen()
        with torch.inference_mode():
            output = self._stream_forward(inputs)
        if output.recurrent_features.shape[:2] != (1, 1):
            raise Stage1ARealSACError(
                "terminal observation requires exactly [1,1]"
            )
        self.hidden = output.final_hidden.detach()
        result = (
            output.recurrent_features[0, -1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if result.shape != (self.agent.config.observation_dim,) or not np.isfinite(result).all():
            raise Stage1ARealSACError("terminal actor observation is invalid")
        return np.ascontiguousarray(result)

    def predict_close_readiness(self, actor_observation: Any) -> float:
        """Predict student readiness from the existing deployable feature only."""

        observation = _finite_vector(
            "close-readiness actor observation",
            actor_observation,
            self.agent.config.observation_dim,
        )
        normalized = self._normalize_close_readiness_observation(observation)
        self.close_readiness_head.eval()
        with torch.inference_mode():
            logits = self.close_readiness_head(
                torch.from_numpy(normalized).to(self.device).unsqueeze(0)
            )
            score = float(torch.sigmoid(logits)[0, 0].detach().cpu().item())
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise Stage1ARealSACError("student close-readiness score is invalid")
        return score

    def _normalize_close_readiness_observation(
        self, observation: np.ndarray
    ) -> np.ndarray:
        """Apply the saved train-only normalizer to the auxiliary head only."""

        if self.close_readiness_training_contract is None:
            return observation
        try:
            return normalize_close_readiness_feature(
                observation,
                contract=self.close_readiness_training_contract,
                observation_dim=self.agent.config.observation_dim,
            )
        except ValueError as error:
            raise Stage1ARealSACError(
                "close-readiness feature normalization failed"
            ) from error

    def _record_close_readiness_supervision(
        self, row: Stage1AAcceptedRealRow
    ) -> None:
        """Record a teacher target without adding it to SAC/student inputs."""

        target = row.privileged_close_ready_target
        score = row.privileged_close_ready_score
        if target is None and score is None:
            return
        if not self.close_readiness_distillation:
            raise Stage1ARealSACError(
                "privileged close-readiness label requires opt-in distillation"
            )
        if type(target) is not bool:
            raise Stage1ARealSACError("close-readiness target must be bool")
        if not row.pre_close_candidate or row.close_latched_before_supervision:
            raise Stage1ARealSACError(
                "close-readiness supervision must be pre-CLOSE and unlatch state"
            )
        if not all(
            isinstance(reason, str) and reason
            for reason in row.privileged_close_ready_negative_reasons
        ):
            raise Stage1ARealSACError(
                "close-readiness negative reasons must be nonempty strings"
            )
        score_value = float(score)
        if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
            raise Stage1ARealSACError(
                "close-readiness teacher score must be finite in [0,1]"
            )
        if score_value != (1.0 if target else 0.0):
            raise Stage1ARealSACError(
                "close-readiness teacher score must be binary admission semantics"
            )
        observation = _finite_vector(
            "close-readiness student observation",
            row.proposal.actor_observation,
            self.agent.config.observation_dim,
        )
        # This prediction is captured before the current row's optimizer
        # update, so W&B agreement is an honest online student metric.
        student_score = self.predict_close_readiness(observation)
        self._close_readiness_observations.append(observation)
        self._close_readiness_scores.append(score_value)
        self._close_readiness_targets.append(target)
        self._student_close_readiness_scores.append(student_score)

    def _close_readiness_batch(self) -> Mapping[str, np.ndarray] | None:
        if (
            not self.close_readiness_distillation
            or len(self._close_readiness_observations) < CLOSE_READINESS_BATCH_SIZE
        ):
            return None
        if self.close_readiness_training_contract is None:
            indices = self._close_readiness_rng.integers(
                0,
                len(self._close_readiness_observations),
                size=CLOSE_READINESS_BATCH_SIZE,
            )
        else:
            targets = np.asarray(self._close_readiness_targets, dtype=np.bool_)
            positive = np.flatnonzero(targets)
            negative = np.flatnonzero(~targets)
            if positive.size == 0 or negative.size == 0:
                return None
            half = CLOSE_READINESS_BATCH_SIZE // 2
            indices = np.concatenate(
                (
                    self._close_readiness_rng.choice(positive, size=half, replace=True),
                    self._close_readiness_rng.choice(negative, size=half, replace=True),
                )
            )
            self._close_readiness_rng.shuffle(indices)
        observations = np.stack(
            [self._close_readiness_observations[int(index)] for index in indices],
            axis=0,
        ).astype(np.float32, copy=False)
        if self.close_readiness_training_contract is not None:
            observations = np.stack(
                [self._normalize_close_readiness_observation(row) for row in observations],
                axis=0,
            )
        scores = np.asarray(
            [self._close_readiness_scores[int(index)] for index in indices],
            dtype=np.float32,
        ).reshape(-1, 1)
        if (
            observations.shape
            != (CLOSE_READINESS_BATCH_SIZE, self.agent.config.observation_dim)
            or scores.shape != (CLOSE_READINESS_BATCH_SIZE, 1)
            or not np.isfinite(observations).all()
            or not np.isfinite(scores).all()
        ):
            raise Stage1ARealSACError("close-readiness distillation batch is invalid")
        return {"observations": observations, "teacher_scores": scores}

    def _update_close_readiness_student(self) -> Mapping[str, float]:
        batch = self._close_readiness_batch()
        if batch is None:
            return dict(self._close_distillation_last_metrics)
        observation = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        teacher_scores = torch.as_tensor(
            batch["teacher_scores"], dtype=torch.float32, device=self.device
        )
        self.close_readiness_head.train()
        logits = self.close_readiness_head(observation)
        if logits.shape != teacher_scores.shape:
            raise Stage1ARealSACError("close-readiness logits shape mismatch")
        if self.close_readiness_training_contract is None:
            loss = F.binary_cross_entropy_with_logits(logits, teacher_scores)
        else:
            loss_per_row = F.binary_cross_entropy_with_logits(
                logits, teacher_scores, reduction="none"
            )
            weights = torch.where(
                teacher_scores > 0.5,
                torch.full_like(
                    teacher_scores,
                    float(
                        self.close_readiness_training_contract[
                            "class_balanced_bce"
                        ]["positive_weight"]
                    ),
                ),
                torch.full_like(
                    teacher_scores,
                    float(
                        self.close_readiness_training_contract[
                            "class_balanced_bce"
                        ]["negative_weight"]
                    ),
                ),
            )
            loss = (loss_per_row * weights).sum() / weights.sum()
        weighted_loss = CLOSE_READINESS_DISTILLATION_WEIGHT * loss
        self.close_readiness_optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.close_readiness_head.parameters(), 5.0
        )
        self.close_readiness_optimizer.step()
        with torch.no_grad():
            student_scores = torch.sigmoid(logits)
        metrics = {
            "loss/close_distill": float(loss.detach()),
            "loss/close_distill_weighted": float(weighted_loss.detach()),
            "close_distill/student_mean": float(student_scores.mean()),
            "close_distill/teacher_mean": float(teacher_scores.mean()),
            "close_distill/gradient_norm": float(gradient_norm),
            "close_distill/active": 1.0,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise Stage1ARealSACError("close-readiness distillation became non-finite")
        self._close_distillation_update_count += 1
        self._close_distillation_last_metrics = metrics
        return dict(metrics)

    @staticmethod
    def _scalar(step: Stage1ARewardStep, name: str, index: int) -> float:
        value = getattr(step, name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not 0 <= index < value.shape[0]:
            raise Stage1ARealSACError(f"reward field {name} is not scalar-per-env")
        result = float(value[index].detach().cpu().item())
        if not math.isfinite(result):
            raise Stage1ARealSACError(f"reward field {name} is non-finite")
        return result

    @staticmethod
    def _optional_scalar(
        step: Stage1ARewardStep, name: str, index: int, default: float = 0.0
    ) -> float:
        value = getattr(step, name)
        if value is None:
            return default
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not 0 <= index < value.shape[0]:
            raise Stage1ARealSACError(f"reward field {name} is not scalar-per-env")
        result = float(value[index].detach().cpu().item())
        if not math.isfinite(result):
            raise Stage1ARealSACError(f"reward field {name} is non-finite")
        return result

    def accept_real_transition(self, row: Stage1AAcceptedRealRow) -> Mapping[str, float]:
        self.assert_bc_frozen()
        if not isinstance(row.reward_step, Stage1ARewardStep):
            raise Stage1ARealSACError("typed Stage1ARewardStep is required")
        if type(row.reward_env_index) is not int or row.reward_env_index < 0:
            raise Stage1ARealSACError("reward_env_index must be a nonnegative integer")
        if not isinstance(row.row_id, str) or not row.row_id or row.row_id in self._row_ids:
            raise Stage1ARealSACError("real replay row ID must be unique and nonempty")
        if not row.canonical_consumption_confirmed:
            raise Stage1ARealSACError("unknown/unconfirmed action consumption cannot enter replay")
        if not row.real_telemetry:
            raise Stage1ARealSACError("synthetic/incomplete telemetry cannot enter real replay")
        if row.source not in REPLAY_SOURCES:
            raise Stage1ARealSACError("unknown replay provenance source")
        if row.dynamics_authority != CANDIDATE_A_DYNAMICS_AUTHORITY:
            raise Stage1ARealSACError(
                "E1/non-Candidate-A transition cannot enter Candidate-A SAC replay"
            )
        if (
            not row.transition_schema_complete
            or not row.policy_observation_complete
            or row.mechanics_only_telemetry
        ):
            raise Stage1ARealSACError(
                "mechanics-only/incomplete transition cannot enter SAC replay"
            )
        if row.action_consumption not in {
            ActionConsumptionState.FULLY_CONSUMED.value,
            ActionConsumptionState.PARTIAL_TERMINATED.value,
        }:
            raise Stage1ARealSACError(
                "unknown/unfinished action consumption cannot enter replay"
            )
        if (
            type(row.control_step_index) is not int
            or type(row.physics_step_start) is not int
            or type(row.physics_step_end) is not int
            or type(row.consumed_substeps) is not int
            or type(row.nominal_substeps) is not int
            or row.physics_step_start != 1
            or row.nominal_substeps != 10
            or not 1 <= row.consumed_substeps <= row.nominal_substeps
            or row.physics_step_end != row.consumed_substeps
            or not math.isclose(
                float(row.execution_fraction),
                row.consumed_substeps / row.nominal_substeps,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise Stage1ARealSACError("invalid action-execution metadata")
        if not row.terminal_observation_is_real:
            raise Stage1ARealSACError("synthetic terminal observation cannot enter replay")
        for name in ("camera_timestamp", "terminal_timestamp", "camera_age"):
            value = float(getattr(row, name))
            if not math.isfinite(value) or value < 0.0:
                raise Stage1ARealSACError(f"{name} must be finite/nonnegative")
        if not math.isclose(
            row.terminal_timestamp - row.camera_timestamp,
            row.camera_age,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise Stage1ARealSACError("camera age/timestamp mismatch")
        if not isinstance(row.episode_id, str) or not row.episode_id:
            raise Stage1ARealSACError("episode_id must be nonempty")
        if (row.privileged_close_ready_target is None) != (
            row.privileged_close_ready_score is None
        ):
            raise Stage1ARealSACError(
                "close-readiness target and score must be present together"
            )
        if (
            row.privileged_close_ready_target is not None
            and not self.close_readiness_distillation
        ):
            raise Stage1ARealSACError(
                "privileged close-readiness label requires opt-in distillation"
            )
        if (
            row.privileged_close_ready_target is not None
            and (
                not row.pre_close_candidate
                or row.close_latched_before_supervision
            )
        ):
            raise Stage1ARealSACError(
                "post-CLOSE/non-candidate row cannot enter close-readiness supervision"
            )
        if row.privileged_close_ready_target is not None:
            if type(row.privileged_close_ready_target) is not bool:
                raise Stage1ARealSACError("close-readiness target must be bool")
            try:
                teacher_score = float(row.privileged_close_ready_score)
            except (TypeError, ValueError) as error:
                raise Stage1ARealSACError(
                    "close-readiness teacher score must be numeric"
                ) from error
            if not math.isfinite(teacher_score) or teacher_score != (
                1.0 if row.privileged_close_ready_target else 0.0
            ):
                raise Stage1ARealSACError(
                    "close-readiness teacher score must be binary admission semantics"
                )
            if not all(
                isinstance(reason, str) and reason
                for reason in row.privileged_close_ready_negative_reasons
            ):
                raise Stage1ARealSACError(
                    "close-readiness negative reasons must be nonempty strings"
                )
        next_observation = _finite_vector(
            "next_actor_observation", row.next_actor_observation, self.agent.config.observation_dim
        )
        if row.proposal.actor_observation.shape != (self.agent.config.observation_dim,):
            raise Stage1ARealSACError("proposal actor observation shape mismatch")
        if np.any(np.abs(row.proposal.normalized_sac_action) > 1.0 + 1.0e-6):
            raise Stage1ARealSACError("normalized SAC action is outside [-1,1]")
        final_norm = float(
            np.linalg.norm(row.proposal.composition.final_xyz_metric_root_m)
        )
        if final_norm > 0.0045 + 1.0e-12:
            raise Stage1ARealSACError("final action exceeds 4.5 mm")
        if (
            row.proposal.composition.final_gripper_probability
            != row.proposal.bc_action_4d_metric_root_m[3]
        ):
            raise Stage1ARealSACError("SAC changed BC gripper authority")
        index = row.reward_env_index
        reward = self._scalar(row.reward_step, "reward_total", index)
        success = bool(self._scalar(row.reward_step, "success", index))
        left = bool(self._scalar(row.reward_step, "left_contact_boolean", index))
        right = bool(self._scalar(row.reward_step, "right_contact_boolean", index))
        stable = bool(self._scalar(row.reward_step, "stable_grasp_boolean", index))
        bilateral = left and right
        if row.success is not None and bool(row.success) != success:
            raise Stage1ARealSACError("provenance success differs from reward authority")
        for name, expected, actual in (
            ("contact", row.contact, left or right),
            ("bilateral", row.bilateral, bilateral),
            ("stable", row.stable, stable),
        ):
            if expected is not None and bool(expected) != actual:
                raise Stage1ARealSACError(
                    f"provenance {name} differs from reward authority"
                )
        force = np.asarray(
            [[
                self._scalar(row.reward_step, "left_normal_force_n", index),
                self._scalar(row.reward_step, "right_normal_force_n", index),
            ]],
            dtype=np.float32,
        )
        action4 = np.asarray([row.proposal.composition.final_action_4d_metric_root_m], dtype=np.float32)
        observation = np.asarray([row.proposal.actor_observation], dtype=np.float32)
        protected_before = (
            reward,
            success,
            action4.copy(),
            observation.copy(),
        )
        # Safety authority is checked before HER_FORCE receives the row.  A
        # failed safety row is never allowed to reach a milestone-priority
        # implementation carrying a success bit.
        if not row.safety_pass and success:
            raise Stage1ARealSACError("safety failure cannot be success/HER success")
        priority_value = 1.0
        if self.replay_strategy == REPLAY_STRATEGY_HER_FORCE:
            priority = prepare_residual_her_force_priorities(
                row_ids=(row.row_id,), real_row=np.ones(1, dtype=np.bool_),
                left_contact=np.asarray([left]), right_contact=np.asarray([right]),
                stable=np.asarray([stable]), lift=np.asarray([bool(row.lift)]),
                contact_force_by_side_n=force,
                safety_measurement_valid=np.asarray([row.safety_measurement_valid]),
                safety_pass=np.asarray([row.safety_pass]),
                reward=np.asarray([reward], dtype=np.float32),
                success=np.asarray([success], dtype=np.bool_),
                action_4d_metric_root_m=action4,
                student_observation=observation,
                config=self.her_force,
            )
            priority_value = float(priority.selection_priority[0])
            self._her_force_transition_count += 1
            self._her_force_activated_count += int(priority_value > 1.0)
        if (
            protected_before[0] != reward or protected_before[1] != success
            or not np.array_equal(protected_before[2], action4)
            or not np.array_equal(protected_before[3], observation)
        ):
            raise Stage1ARealSACError("HER_FORCE mutated a learner tensor")
        phase = "LIFT" if row.lift else "STABLE_GRASP" if stable else "CONTACT" if (left or right) else "REACH"
        reward_done = bool(self._scalar(row.reward_step, "done", index))
        if row.terminated is not None and bool(row.terminated) != reward_done:
            raise Stage1ARealSACError("terminated flag differs from Stage-1A reward authority")
        terminated = reward_done
        if terminated and row.truncated:
            raise Stage1ARealSACError("transition cannot be terminated and truncated")
        safety_penalty = self._scalar(row.reward_step, "safety_penalty", index)
        if row.hardstop:
            if (
                row.failure_reason != RUNTIME_HARDSTOP_FAILURE
                or success
                or not terminated
                or row.safety_pass
                or safety_penalty >= 0.0
                or row.action_consumption
                != ActionConsumptionState.PARTIAL_TERMINATED.value
                or not row.hardstop_joint
                or row.idx83_min_margin is None
                or not math.isfinite(float(row.idx83_min_margin))
            ):
                raise Stage1ARealSACError(
                    "runtime hard-stop must be failed terminal with a negative safety penalty"
                )
        elif row.failure_reason == RUNTIME_HARDSTOP_FAILURE:
            raise Stage1ARealSACError("hard-stop failure reason requires hardstop=true")
        elif (
            row.action_consumption
            != ActionConsumptionState.FULLY_CONSUMED.value
            or row.consumed_substeps != row.nominal_substeps
        ):
            raise Stage1ARealSACError(
                "partial consumption is reserved for runtime hard-stop"
            )
        self.replay.add(
            row.proposal.actor_observation,
            row.proposal.normalized_sac_action,
            reward,
            next_observation,
            terminated,
            bool(row.truncated),
            priority=priority_value,
            phase_label=phase,
        )
        self._original_transition_count += 1
        self._row_ids.add(row.row_id)
        self._record_close_readiness_supervision(row)
        self._replay_provenance.append(
            {
                "row_id": row.row_id,
                "source": row.source,
                "episode_id": row.episode_id,
                "phase": row.phase or phase,
                "success": success,
                "failure_reason": row.failure_reason,
                "hardstop": bool(row.hardstop),
                "contact": left or right,
                "bilateral": bilateral,
                "stable": stable,
                "terminated": terminated,
                "truncated": bool(row.truncated),
                "dynamics_authority": row.dynamics_authority,
                "safety_pass": bool(row.safety_pass),
                "action_consumption": row.action_consumption,
                "control_step_index": row.control_step_index,
                "physics_step_start": row.physics_step_start,
                "physics_step_end": row.physics_step_end,
                "consumed_substeps": row.consumed_substeps,
                "nominal_substeps": row.nominal_substeps,
                "execution_fraction": row.execution_fraction,
                "terminal_observation_is_real": row.terminal_observation_is_real,
                "camera_timestamp": row.camera_timestamp,
                "terminal_timestamp": row.terminal_timestamp,
                "camera_age": row.camera_age,
                "hardstop_joint": row.hardstop_joint,
                "idx83_min_margin": row.idx83_min_margin,
                "pre_close_candidate": row.pre_close_candidate,
                "close_latched_before_supervision": (
                    row.close_latched_before_supervision
                ),
                "privileged_close_ready_target": row.privileged_close_ready_target,
                "privileged_close_ready_score": row.privileged_close_ready_score,
                "privileged_close_ready_negative_reasons": (
                    row.privileged_close_ready_negative_reasons
                ),
                "close_readiness_target_schema": (
                    CLOSE_READINESS_TARGET_SCHEMA_VERSION
                    if row.privileged_close_ready_target is not None
                    else None
                ),
                "student_privileged_input_count": 0,
            }
        )
        self.accepted_transitions += 1
        self._accumulate(row.reward_step, index, row.proposal)
        if len(self.replay) >= self.learning_starts:
            for _ in range(self.updates_per_accepted_row):
                self._optimizer_metrics = self.agent.update(self.replay.sample(self.batch_size))
                self._optimizer_metrics.update(self._update_close_readiness_student())
                self.sac_update_count += 1
        return dict(self._optimizer_metrics)

    def _accumulate(self, step: Stage1ARewardStep, index: int, proposal: Stage1AActionProposal) -> None:
        mapping = {
            "progress": "reward_progress", "lateral": "reward_lateral",
            "single": "reward_single_contact", "bilateral": "reward_bilateral",
            "stable": "reward_stable", "success": "reward_success",
            "residual_penalty": "residual_penalty", "smoothness_penalty": "smoothness_penalty",
            "time_penalty": "time_penalty", "safety_penalty": "safety_penalty",
            "hover_penalty": "penalty_hover",
            "single_contact_dwell_penalty": "penalty_single_contact_dwell",
        }
        for target, source in mapping.items():
            self._reward_sums[target] += self._scalar(step, source, index)
        optional_mapping = {
            "cosine": "reward_cosine",
            "oscillation_penalty": "penalty_oscillation",
            "worsening_penalty": "penalty_worsening",
            "milestone_20mm": "reward_milestone_20mm",
            "milestone_18mm": "reward_milestone_18mm",
            "milestone_16mm": "reward_milestone_16mm",
            "stable_hold": "reward_stable_hold",
        }
        for target, source in optional_mapping.items():
            self._reward_sums[target] += self._optional_scalar(step, source, index)
        effective_mm = 1000.0 * float(np.linalg.norm(proposal.composition.scaled_residual_contribution_m))
        self._residual_norm_mm.append(effective_mm)
        at_max = math.isclose(effective_mm, 0.45, rel_tol=0.0, abs_tol=1.0e-6)
        self._current_max_residual_streak = self._current_max_residual_streak + 1 if at_max else 0
        self._max_residual_streak = max(self._max_residual_streak, self._current_max_residual_streak)
        self._counts["hover"] += int(bool(self._scalar(step, "no_progress_hovering", index)))
        v3_active = bool(self._optional_scalar(step, "reward_v3_active", index))
        temporal_valid = bool(self._optional_scalar(step, "temporal_window_valid", index))
        self._counts["v3_active"] += int(v3_active)
        self._counts["temporal_valid"] += int(temporal_valid)
        for target, source in (
            ("progressing", "temporal_progressing"),
            ("static_hover", "temporal_static_hover"),
            ("oscillatory_hover", "temporal_oscillatory_hover"),
            ("worsening", "temporal_worsening"),
            ("ambiguous", "temporal_ambiguous"),
            ("cosine_sustained", "cosine_sustained"),
            ("cosine_with_progress", "cosine_with_progress"),
        ):
            self._counts[target] += int(bool(self._optional_scalar(step, source, index)))
        if v3_active:
            self._v3_active_nominal_residual_mm.append(
                self._optional_scalar(step, "nominal_residual_mm", index)
            )
        v3_hover = bool(
            self._optional_scalar(step, "temporal_static_hover", index)
            or self._optional_scalar(step, "temporal_oscillatory_hover", index)
        )
        if v3_hover:
            self._current_v3_hover_steps += 1
        elif self._current_v3_hover_steps:
            self._v3_hover_durations_steps.append(self._current_v3_hover_steps)
            self._current_v3_hover_steps = 0
        self._counts["single_contact_dwell"] += int(
            self._scalar(step, "penalty_single_contact_dwell", index) < 0.0
        )
        for target, source in (("bilateral", "reward_bilateral"), ("stable", "reward_stable"), ("success", "reward_success")):
            event = self._scalar(step, source, index) > 0.0
            self._counts[target] += int(event)
            if event:
                self._episode_milestones[target] = self._episode_milestones.get(target, 0) + 1
                if self._episode_milestones[target] > 1:
                    self.recontact_farming = True
        self._counts["contact"] += int(
            self._scalar(step, "reward_single_contact", index) > 0.0
        )
        self._counts["contact_loss_termination"] += int(
            bool(self._scalar(step, "contact_loss_fail", index))
        )
        final_norm = float(np.linalg.norm(proposal.composition.final_xyz_metric_root_m))
        self._counts["bound_violation"] += int(final_norm > 0.0045 + 1.0e-12)
        self._counts["gripper_authority_violation"] += int(
            proposal.composition.final_gripper_probability != proposal.bc_action_4d_metric_root_m[3]
        )

    def metrics(self) -> dict[str, float | int | bool]:
        values = np.asarray(self._residual_norm_mm, dtype=np.float64)
        teacher_scores = np.asarray(
            self._close_readiness_scores, dtype=np.float64
        )
        teacher_targets = np.asarray(
            self._close_readiness_targets, dtype=np.bool_
        )
        student_scores = np.asarray(
            self._student_close_readiness_scores, dtype=np.float64
        )
        if teacher_scores.shape != student_scores.shape or (
            teacher_scores.size != teacher_targets.size
        ):
            raise Stage1ARealSACError("close-readiness metric cardinality mismatch")
        predicted_ready = student_scores >= 0.5
        true_positive = int(np.sum(predicted_ready & teacher_targets))
        false_positive = int(np.sum(predicted_ready & ~teacher_targets))
        false_negative = int(np.sum(~predicted_ready & teacher_targets))
        true_negative = int(np.sum(~predicted_ready & ~teacher_targets))
        teacher_positive_count = int(np.sum(teacher_targets))
        teacher_negative_count = int(teacher_targets.size - teacher_positive_count)
        active_residual = np.asarray(
            self._v3_active_nominal_residual_mm, dtype=np.float64
        )
        hover_durations = np.asarray(
            self._v3_hover_durations_steps
            + ([self._current_v3_hover_steps] if self._current_v3_hover_steps else []),
            dtype=np.float64,
        )
        temporal_denominator = max(1, self._counts["temporal_valid"])
        result: dict[str, float | int | bool] = {
            "ACCEPTED_TRANSITIONS": self.accepted_transitions,
            "SAC_UPDATE_COUNT": self.sac_update_count,
            "RESIDUAL_NORM_MEAN_MM": float(values.mean()) if values.size else 0.0,
            "RESIDUAL_NORM_P95_MM": float(np.percentile(values, 95)) if values.size else 0.0,
            "RESIDUAL_NORM_MAX_MM": float(values.max()) if values.size else 0.0,
            "MAX_RESIDUAL_ALWAYS": bool(values.size and np.all(np.isclose(values, 0.45, atol=1e-6))),
            "MAX_RESIDUAL_STREAK": self._max_residual_streak,
            "HOVER_RATE": float(self._counts["hover"] / max(1, self.accepted_transitions)),
            "SINGLE_CONTACT_DWELL_COUNT": self._counts["single_contact_dwell"],
            "BILATERAL_COUNT": self._counts["bilateral"],
            "STABLE_COUNT": self._counts["stable"],
            "SUCCESS_COUNT": self._counts["success"],
            "CONTACT_LOSS_TERMINATION_COUNT": self._counts["contact_loss_termination"],
            "FINAL_ACTION_BOUND_VIOLATION": self._counts["bound_violation"],
            "GRIPPER_AUTHORITY_VIOLATION": self._counts["gripper_authority_violation"],
            "FORCE_REWARD_FARMING": self.force_reward_farming,
            "RECONTACT_FARMING": self.recontact_farming,
            "CONTACT_COUNT": self._counts["contact"],
            "TOTAL_SAC_ACTIVE_STEPS": self._counts["v3_active"],
            "MIN_NOMINAL_RESIDUAL_MM_IN_SAC_ACTIVE": float(active_residual.min()) if active_residual.size else None,
            "MEAN_NOMINAL_RESIDUAL_MM_IN_SAC_ACTIVE": float(active_residual.mean()) if active_residual.size else None,
            "TEMPORAL_WINDOW_VALID_STEPS": self._counts["temporal_valid"],
            "PROGRESSING_RATE": self._counts["progressing"] / temporal_denominator,
            "STATIC_HOVER_RATE": self._counts["static_hover"] / temporal_denominator,
            "OSCILLATORY_HOVER_RATE": self._counts["oscillatory_hover"] / temporal_denominator,
            "WORSENING_RATE": self._counts["worsening"] / temporal_denominator,
            "AMBIGUOUS_RATE": self._counts["ambiguous"] / temporal_denominator,
            "MEAN_CONSECUTIVE_HOVER_DURATION_MS": float(hover_durations.mean() * 20.0) if hover_durations.size else 0.0,
            "P95_CONSECUTIVE_HOVER_DURATION_MS": float(np.percentile(hover_durations, 95) * 20.0) if hover_durations.size else 0.0,
            "COSINE_ABOVE_THRESHOLD_RATE": self._counts["cosine_sustained"] / max(1, self._counts["v3_active"]),
            "COSINE_WITH_PROGRESS_RATE": self._counts["cosine_with_progress"] / max(1, self._counts["v3_active"]),
            "SAC_ACTIVE_STEPS": self._counts["v3_active"],
            "MIN_RESIDUAL_MM": float(active_residual.min()) if active_residual.size else None,
            "MEAN_RESIDUAL_MM_IN_ACTIVE_BAND": float(active_residual.mean()) if active_residual.size else None,
            "HOVER_RATE_TOTAL": float(self._counts["hover"] / max(1, self.accepted_transitions)),
            "0P3_PROGRESS_POSITIVE_RATE": self._counts["progressing"] / temporal_denominator,
            "COS_0P1_ABOVE_0P8_RATE": self._counts["cosine_sustained"] / max(1, self._counts["v3_active"]),
            "COS_0P1_ABOVE_0P8_WITH_PROGRESS_RATE": self._counts["cosine_with_progress"] / max(1, self._counts["v3_active"]),
            "BC_WEIGHTS_CHANGED": tensor_state_sha256(self.bc) != self.bc_state_sha256,
            "ACTOR_LOSS_FINITE": bool(
                self._optimizer_metrics
                and math.isfinite(self._optimizer_metrics.get("loss/actor", math.nan))
            ),
            "CRITIC_LOSS_FINITE": bool(
                self._optimizer_metrics
                and math.isfinite(self._optimizer_metrics.get("loss/critic", math.nan))
            ),
            "ALPHA_FINITE": bool(
                self._optimizer_metrics
                and math.isfinite(self._optimizer_metrics.get("loss/alpha", math.nan))
                and math.isfinite(self._optimizer_metrics.get("temperature/alpha", math.nan))
            ),
            "REPLAY_STRATEGY": self.replay_strategy,
            "ORIGINAL_TRANSITION_COUNT": self._original_transition_count,
            "HER_TRANSITION_COUNT": self._her_transition_count,
            "HER_SUCCESS_RATE": 0.0,
            "GOAL_RELABEL_RATIO": 0.0,
            "HER_FORCE_TRANSITION_COUNT": self._her_force_transition_count,
            "HER_FORCE_ACTIVATION_RATE": float(
                self._her_force_activated_count
                / max(1, self._her_force_transition_count)
            ),
            "HER_FORCE_SUCCESS_RATE": float(
                self._counts["success"] / max(1, self._her_force_transition_count)
            ),
            "CLOSE_DISTILLATION_ENABLED": self.close_readiness_distillation,
            "CLOSE_DISTILLATION_WEIGHT": CLOSE_READINESS_DISTILLATION_WEIGHT,
            "CLOSE_READINESS_NORMALIZATION_PARITY": bool(
                self.close_readiness_training_contract is not None
            ),
            "CLOSE_READINESS_CLASS_BALANCED_SAMPLING": bool(
                self.close_readiness_training_contract is not None
            ),
            "CLOSE_READINESS_CLASS_BALANCED_LOSS": bool(
                self.close_readiness_training_contract is not None
            ),
            "CLOSE_READINESS_TRAINING_CONTRACT_SHA256": (
                self.close_readiness_training_contract["contract_sha256"]
                if self.close_readiness_training_contract is not None
                else None
            ),
            "CLOSE_DISTILLATION_ROWS": int(teacher_scores.size),
            "CLOSE_DISTILLATION_UPDATE_COUNT": self._close_distillation_update_count,
            "PRIVILEGED_CLOSE_READY_RATE": float(
                teacher_positive_count / max(1, teacher_targets.size)
            ),
            "PRIVILEGED_CLOSE_READY_SCORE_MEAN": float(
                teacher_scores.mean()
            ) if teacher_scores.size else 0.0,
            "STUDENT_CLOSE_READY_MEAN": float(student_scores.mean()) if student_scores.size else 0.0,
            "TEACHER_STUDENT_AGREEMENT": float(
                (true_positive + true_negative) / max(1, teacher_targets.size)
            ),
            "STUDENT_CLOSE_READY_PRECISION": float(
                true_positive / max(1, true_positive + false_positive)
            ),
            "STUDENT_CLOSE_READY_RECALL": float(
                true_positive / max(1, true_positive + false_negative)
            ),
            "STUDENT_FALSE_REJECT_RATE": float(
                false_negative / max(1, teacher_positive_count)
            ),
            "STUDENT_FALSE_ACCEPT_RATE": float(
                false_positive / max(1, teacher_negative_count)
            ),
            "STUDENT_CLOSE_READY_TRUE_POSITIVE": true_positive,
            "STUDENT_CLOSE_READY_FALSE_POSITIVE": false_positive,
            "STUDENT_CLOSE_READY_FALSE_NEGATIVE": false_negative,
            "STUDENT_CLOSE_READY_TRUE_NEGATIVE": true_negative,
        }
        result.update({f"REWARD_SUM/{name}": value for name, value in self._reward_sums.items()})
        result.update(
            {
                "PROGRESS_REWARD_SUM": self._reward_sums["progress"],
                "LATERAL_REWARD_SUM": self._reward_sums["lateral"],
                "SINGLE_REWARD_SUM": self._reward_sums["single"],
                "BILATERAL_REWARD_SUM": self._reward_sums["bilateral"],
                "STABLE_REWARD_SUM": self._reward_sums["stable"],
                "SUCCESS_REWARD_SUM": self._reward_sums["success"],
                "RESIDUAL_PENALTY_SUM": self._reward_sums["residual_penalty"],
                "SMOOTHNESS_PENALTY_SUM": self._reward_sums["smoothness_penalty"],
                "TIME_PENALTY_SUM": self._reward_sums["time_penalty"],
                "SAFETY_PENALTY_SUM": self._reward_sums["safety_penalty"],
                "COSINE_REWARD_SUM": self._reward_sums["cosine"],
                "OSCILLATION_PENALTY_SUM": self._reward_sums["oscillation_penalty"],
                "WORSENING_PENALTY_SUM": self._reward_sums["worsening_penalty"],
                "MILESTONE_20MM_REWARD_SUM": self._reward_sums["milestone_20mm"],
                "MILESTONE_18MM_REWARD_SUM": self._reward_sums["milestone_18mm"],
                "MILESTONE_16MM_REWARD_SUM": self._reward_sums["milestone_16mm"],
                "STABLE_HOLD_REWARD_SUM": self._reward_sums["stable_hold"],
                "HOVER_PENALTY_SUM": self._reward_sums["hover_penalty"],
                "SINGLE_CONTACT_DWELL_PENALTY_SUM": self._reward_sums["single_contact_dwell_penalty"],
            }
        )
        result.update({f"optimizer/{name}": value for name, value in self._optimizer_metrics.items()})
        for source in sorted(REPLAY_SOURCES):
            result[f"REPLAY_SOURCE_COUNT/{source}"] = sum(
                item["source"] == source for item in self._replay_provenance
            )
        result["RUNTIME_HARDSTOP_COUNT"] = sum(
            bool(item["hardstop"]) for item in self._replay_provenance
        )
        result["FULL_50HZ_TRANSITIONS"] = sum(
            item["action_consumption"]
            == ActionConsumptionState.FULLY_CONSUMED.value
            for item in self._replay_provenance
        )
        result["PARTIAL_TERMINAL_TRANSITIONS"] = sum(
            item["action_consumption"]
            == ActionConsumptionState.PARTIAL_TERMINATED.value
            for item in self._replay_provenance
        )
        result["PARTIAL_TERMINAL_SUBSTEP_HISTOGRAM"] = {
            str(substep): sum(
                item["action_consumption"]
                == ActionConsumptionState.PARTIAL_TERMINATED.value
                and item["consumed_substeps"] == substep
                for item in self._replay_provenance
            )
            for substep in range(1, 11)
        }
        result["REPLAY_HARDSTOP_COUNT"] = result["RUNTIME_HARDSTOP_COUNT"]
        return result

    def _close_readiness_state_dict(self) -> dict[str, Any]:
        """Persist teacher-only auxiliary state without polluting actor input."""

        observations = (
            np.stack(self._close_readiness_observations, axis=0).astype(
                np.float32, copy=False
            )
            if self._close_readiness_observations
            else np.empty((0, self.agent.config.observation_dim), dtype=np.float32)
        )
        return {
            "enabled": self.close_readiness_distillation,
            "student_input_authority": "DEPLOYABLE_FROZEN_GRU_FEATURE_ONLY",
            "student_privileged_input_count": 0,
            "teacher_role": "SUPERVISION_ONLY_NOT_RUNTIME_CLOSE_GATE",
            "weight": CLOSE_READINESS_DISTILLATION_WEIGHT,
            "head_architecture": self.close_readiness_head_architecture,
            "normalization_parity": self.close_readiness_training_contract is not None,
            "class_balanced_sampling": self.close_readiness_training_contract is not None,
            "class_balanced_loss": self.close_readiness_training_contract is not None,
            "training_contract": self.close_readiness_training_contract,
            "training_contract_sha256": (
                self.close_readiness_training_contract["contract_sha256"]
                if self.close_readiness_training_contract is not None
                else None
            ),
            "initialization_head_sha256": self._close_readiness_initialization_sha256,
            "head": self.close_readiness_head.state_dict(),
            "head_sha256": _module_sha256(self.close_readiness_head),
            "optimizer": self.close_readiness_optimizer.state_dict(),
            "observations": observations,
            "teacher_scores": np.asarray(
                self._close_readiness_scores, dtype=np.float32
            ),
            "teacher_targets": np.asarray(
                self._close_readiness_targets, dtype=np.bool_
            ),
            "student_scores_before_update": np.asarray(
                self._student_close_readiness_scores, dtype=np.float32
            ),
            "update_count": self._close_distillation_update_count,
            "rng_state": self._close_readiness_rng.bit_generator.state,
        }

    @property
    def replay_provenance(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._replay_provenance)

    def export_checkpoints(self, output_dir: str | Path) -> Stage1ACheckpointReceipt:
        self.assert_bc_frozen()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        actor_path = output / "stage1a_residual_actor.pt"
        torch.save(
            residual_sac_actor_checkpoint_payload(
                self.agent.actor,
                human_grasp_checkpoint_sha256=self.bc_sha256,
            ),
            actor_path,
        )
        actor_sha = _file_sha256(actor_path)
        reloaded_actor = load_residual_sac_actor_checkpoint(
            actor_path,
            expected_sha256=actor_sha,
            expected_human_grasp_checkpoint_sha256=self.bc_sha256,
            expected_observation_dim=self.agent.config.observation_dim,
            device="cpu",
        )
        actor_reload = reloaded_actor.receipt.actor_state_sha256 == tensor_state_sha256(self.agent.actor)
        learner_path = output / "stage1a_smoke_learner.pt"
        payload = {
            "checkpoint_version": 1,
            "schema": STAGE1A_REAL_LEARNER_CHECKPOINT_SCHEMA,
            "bc_checkpoint_sha256": self.bc_sha256,
            "bc_state_sha256": self.bc_state_sha256,
            "agent": self.agent.state_dict(),
            "replay": self.replay.state_dict(),
            "accepted_transitions": self.accepted_transitions,
            "sac_update_count": self.sac_update_count,
            "row_ids": tuple(sorted(self._row_ids)),
            "replay_provenance": tuple(dict(item) for item in self._replay_provenance),
            "her_force": self.her_force.payload(),
            "replay_strategy": self.replay_strategy,
            "actor_initialization": dict(self.actor_initialization),
            "alpha": self.alpha.payload(),
            "close_readiness_distillation": self._close_readiness_state_dict(),
            "metrics": self.metrics(),
            "rng_state": capture_rng_state(),
        }
        save_torch_checkpoint(learner_path, payload)
        restored = torch.load(learner_path, map_location="cpu", weights_only=False)
        replay = ReplayBuffer(
            self.replay.capacity,
            self.replay.observation_dim,
            self.replay.action_dim,
            seed=0,
        )
        replay.load_state_dict(restored["replay"])
        agent = SACAgent(self.agent.config, device="cpu")
        agent.load_state_dict(restored["agent"])
        restored_close_head = build_close_readiness_head(
            self.agent.config.observation_dim,
            architecture=str(
                restored["close_readiness_distillation"].get(
                    "head_architecture", "MLP64_RELU_V1"
                )
            ),
        )
        restored_close_head.load_state_dict(
            restored["close_readiness_distillation"]["head"]
        )
        learner_reload = bool(
            restored.get("schema") == STAGE1A_REAL_LEARNER_CHECKPOINT_SCHEMA
            and len(replay) == len(self.replay)
            and agent.update_count == self.agent.update_count == self.sac_update_count
            and restored.get("bc_checkpoint_sha256") == self.bc_sha256
            and _module_sha256(restored_close_head)
            == restored["close_readiness_distillation"]["head_sha256"]
        )
        if not actor_reload or not learner_reload:
            raise Stage1ARealSACError("Stage-1A checkpoint reload verification failed")
        return Stage1ACheckpointReceipt(
            learner_path=str(learner_path), actor_path=str(actor_path),
            actor_sha256=actor_sha, actor_reload_pass=True, learner_reload_pass=True,
            bc_sha256=self.bc_sha256,
        )

    def export_periodic_training_checkpoint(
        self,
        path: str | Path,
        *,
        source_freeze: Mapping[str, Any],
        reward_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        allow_bounded_final_boundary: bool = False,
    ) -> Stage1APeriodicCheckpointReceipt:
        """Persist a fully resumable periodic or bounded-final checkpoint.

        The payload intentionally contains the replay and RNG state in addition
        to the actor.  A checkpoint is therefore a continuation point, not an
        actor-only deployment checkpoint.  The default remains locked to 3K
        boundaries; callers must explicitly attest a non-3K bounded final
        target (for example 7.5K).
        """

        self.assert_bc_frozen()
        step = int(self.accepted_transitions)
        if step <= 0 or (
            step % 3000 != 0 and not bool(allow_bounded_final_boundary)
        ):
            raise Stage1ARealSACError(
                "periodic checkpoint requires a positive 3000-transition boundary"
            )
        destination = Path(path).expanduser().resolve()
        if destination.exists():
            raise Stage1ARealSACError("periodic checkpoint refuses overwrite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": 1,
            "schema": STAGE1A_PERIODIC_CHECKPOINT_SCHEMA,
            "checkpoint_boundary_kind": (
                "BOUNDED_FINAL" if step % 3000 != 0 else "PERIODIC_3K"
            ),
            "accepted_transitions": step,
            "transition_count": step,
            "sac_update_count": int(self.sac_update_count),
            "replay_strategy": self.replay_strategy,
            "her_force": self.her_force.payload(),
            "agent": self.agent.state_dict(),
            "replay": self.replay.state_dict(),
            "close_readiness_distillation": self._close_readiness_state_dict(),
            "optimizer_state_included": True,
            "contents": {
                "actor": True,
                "critic": True,
                "alpha": True,
                "optimizer_state": True,
                "replay_buffer": True,
                "rng_state": True,
                **(
                    {"student_close_readiness_head": True}
                    if self.close_readiness_distillation
                    else {}
                ),
            },
            "bc_checkpoint_sha256": self.bc_sha256,
            "bc_state_sha256": self.bc_state_sha256,
            "actor_initialization": dict(self.actor_initialization),
            "alpha_curriculum": self.alpha.payload(),
            "config": {
                "sac": dict(self.agent.state_dict()["config"]),
                "runtime": dict(runtime_config),
            },
            "source_hash": dict(source_freeze),
            "reward_config": dict(reward_config),
            "row_ids": tuple(sorted(self._row_ids)),
            "replay_provenance": tuple(
                dict(item) for item in self._replay_provenance
            ),
            "metrics": self.metrics(),
            "rng_state": capture_rng_state(),
        }
        save_torch_checkpoint(destination, payload)
        restored = torch.load(destination, map_location="cpu", weights_only=False)
        replay = ReplayBuffer(
            self.replay.capacity,
            self.replay.observation_dim,
            self.replay.action_dim,
            seed=0,
        )
        replay.load_state_dict(restored["replay"])
        agent = SACAgent(self.agent.config, device="cpu")
        agent.load_state_dict(restored["agent"])
        restored_close_head = build_close_readiness_head(
            self.agent.config.observation_dim,
            architecture=str(
                restored["close_readiness_distillation"].get(
                    "head_architecture", "MLP64_RELU_V1"
                )
            ),
        )
        restored_close_head.load_state_dict(
            restored["close_readiness_distillation"]["head"]
        )
        reload_pass = bool(
            restored.get("schema") == STAGE1A_PERIODIC_CHECKPOINT_SCHEMA
            and int(restored.get("accepted_transitions", -1)) == step
            and restored.get("replay_strategy") == self.replay_strategy
            and len(replay) == len(self.replay)
            and agent.update_count == self.agent.update_count
            and restored.get("bc_checkpoint_sha256") == self.bc_sha256
            and all(bool(value) for value in restored.get("contents", {}).values())
            and _module_sha256(restored_close_head)
            == restored["close_readiness_distillation"]["head_sha256"]
        )
        if not reload_pass:
            raise Stage1ARealSACError(
                "periodic Stage-1A checkpoint reload verification failed"
            )
        return Stage1APeriodicCheckpointReceipt(
            path=str(destination),
            sha256=_file_sha256(destination),
            accepted_transitions=step,
            replay_strategy=self.replay_strategy,
            reload_pass=True,
        )


__all__ = [
    "STAGE1A_REAL_COORDINATOR_SCHEMA", "STAGE1A_REAL_LEARNER_CHECKPOINT_SCHEMA",
    "STAGE1A_PERIODIC_CHECKPOINT_SCHEMA", "Stage1AAcceptedRealRow",
    "Stage1AActionProposal", "Stage1ACheckpointReceipt",
    "Stage1APeriodicCheckpointReceipt",
    "Stage1ARealSACCoordinator", "Stage1ARealSACError", "radial_metric_residual",
    "REPLAY_STRATEGY_SAC", "REPLAY_STRATEGY_HER", "REPLAY_STRATEGY_HER_FORCE",
    "REPLAY_STRATEGIES",
]
