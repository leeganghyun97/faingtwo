# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Branch-local demonstration action schema and explicit legacy projection.

The historical G2 collection schema remains immutable.  This module creates a
new 4-D/7-D policy record only through a deliberate projection, so rotation or
gripper-sign semantics cannot be silently discarded while moving to the new
policy branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor

from .action_interface import (
    AbstractGripperIntent,
    GripperHysteresisConfig,
    POLICY_ACTION_SCHEMA_4D,
    POLICY_ACTION_SCHEMA_7D,
    PolicyActionMode,
)
from .observation import PolicyDataSemantics, PolicyObservation


HIGH_LEVEL_DEMONSTRATION_SCHEMA = "g2_high_level_policy_demonstration_v2"
LEGACY_CANONICAL_POLICY7_SCHEMA = "g2_se3_gripper_v1"
LEGACY_KEYBOARD_POLICY8_SCHEMA = "g2_se3_elbow_nullspace_gripper_v1"


class PolicyDatasetContractError(ValueError):
    """Raised when source demonstrations are not safe to project to this branch."""


class TaskPhase(str, Enum):
    """Evaluation/data metadata only; it is intentionally outside actor inputs."""

    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    CLOSE = "CLOSE"
    CONTACT = "CONTACT"
    STABLE_GRASP = "STABLE_GRASP"
    LIFT = "LIFT"
    MOVE = "MOVE"
    PLACE = "PLACE"


def policy_action_schema(mode: PolicyActionMode) -> str:
    return (
        POLICY_ACTION_SCHEMA_4D
        if mode is PolicyActionMode.TRANSLATION_GRIPPER_4D
        else POLICY_ACTION_SCHEMA_7D
    )


def _validate_legacy_policy7(action: Tensor) -> None:
    if action.ndim < 1 or action.shape[-1] != 7:
        raise PolicyDatasetContractError("legacy canonical action must end in width 7")
    if not torch.is_floating_point(action) or not bool(torch.isfinite(action).all()):
        raise PolicyDatasetContractError("legacy canonical action must be finite floating point")
    if bool((action[..., :6].abs() > 1.0).any()) or bool((action[..., 6].abs() > 1.0).any()):
        raise PolicyDatasetContractError("legacy canonical action must be normalised")


def legacy_gripper_sign_to_probability(
    legacy_sign: Tensor,
    *,
    binary_tolerance: float = 1.0e-6,
) -> Tensor:
    """Convert only the audited historical convention ``+1 OPEN / -1 CLOSE``."""

    if not torch.is_floating_point(legacy_sign) or not bool(torch.isfinite(legacy_sign).all()):
        raise PolicyDatasetContractError("legacy gripper signs must be finite")
    if bool((torch.abs(torch.abs(legacy_sign) - 1.0) > binary_tolerance).any()):
        raise PolicyDatasetContractError(
            "legacy gripper convention is not binary +/-1; explicit migration is required"
        )
    return (1.0 - legacy_sign) * 0.5


def project_legacy_policy7_to_high_level(
    legacy_action: Tensor,
    *,
    source_schema: str,
    mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    rotation_tolerance: float = 1.0e-6,
) -> Tensor:
    """Project legacy high-level EE data only when the requested semantics match.

    A 4-D policy cannot silently erase nonzero legacy rotations.  It must be
    trained from fixed-orientation rows, or the caller must select the explicit
    7-D action mode.
    """

    if source_schema != LEGACY_CANONICAL_POLICY7_SCHEMA:
        raise PolicyDatasetContractError(
            "legacy projection requires explicit g2_se3_gripper_v1 source schema"
        )
    _validate_legacy_policy7(legacy_action)
    if rotation_tolerance < 0.0:
        raise PolicyDatasetContractError("rotation_tolerance must be non-negative")
    close_probability = legacy_gripper_sign_to_probability(legacy_action[..., 6])
    if mode is PolicyActionMode.TRANSLATION_GRIPPER_4D:
        if bool((legacy_action[..., 3:6].abs() > rotation_tolerance).any()):
            raise PolicyDatasetContractError(
                "cannot project nonzero legacy rotations into fixed-orientation 4-D policy"
            )
        return torch.cat((legacy_action[..., :3], close_probability.unsqueeze(-1)), dim=-1)
    return torch.cat(
        (legacy_action[..., :6], close_probability.unsqueeze(-1)), dim=-1
    )


def project_legacy_policy8_to_high_level(
    legacy_action: Tensor,
    *,
    source_schema: str,
    mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    rotation_tolerance: float = 1.0e-6,
    elbow_tolerance: float = 1.0e-6,
) -> Tensor:
    """Project the recorded keyboard controller packet without learning fillers.

    Historical keyboard files store ``[xyz, rotvec, elbow, gripper_sign]``.
    Fixed orientation and fixed elbow are compatibility slots on the downstream
    controller surface, not policy targets.  The 4-D branch may therefore use a
    source episode only when those four slots are demonstrably zero.  They are
    then discarded, never emitted as zero-valued BC labels.
    """

    if source_schema != LEGACY_KEYBOARD_POLICY8_SCHEMA:
        raise PolicyDatasetContractError(
            "legacy keyboard projection requires explicit "
            "g2_se3_elbow_nullspace_gripper_v1 source schema"
        )
    if legacy_action.ndim < 1 or legacy_action.shape[-1] != 8:
        raise PolicyDatasetContractError("legacy keyboard action must end in width 8")
    if not torch.is_floating_point(legacy_action) or not bool(
        torch.isfinite(legacy_action).all()
    ):
        raise PolicyDatasetContractError(
            "legacy keyboard action must be finite floating point"
        )
    if bool((legacy_action[..., :7].abs() > 1.0).any()) or bool(
        (legacy_action[..., 7].abs() > 1.0).any()
    ):
        raise PolicyDatasetContractError("legacy keyboard action must be normalised")
    if rotation_tolerance < 0.0 or elbow_tolerance < 0.0:
        raise PolicyDatasetContractError("projection tolerances must be non-negative")
    if mode is PolicyActionMode.TRANSLATION_GRIPPER_4D:
        if bool((legacy_action[..., 3:6].abs() > rotation_tolerance).any()):
            raise PolicyDatasetContractError(
                "cannot project nonzero legacy rotations into fixed-orientation 4-D policy"
            )
        if bool((legacy_action[..., 6].abs() > elbow_tolerance).any()):
            raise PolicyDatasetContractError(
                "cannot project nonzero legacy elbow into fixed-elbow 4-D policy"
            )
    canonical7 = torch.cat(
        (legacy_action[..., :6], legacy_action[..., 7:8]), dim=-1
    )
    return project_legacy_policy7_to_high_level(
        canonical7,
        source_schema=LEGACY_CANONICAL_POLICY7_SCHEMA,
        mode=mode,
        rotation_tolerance=rotation_tolerance,
    )


def prepare_metric_depth_for_policy(
    raw_depth_m: Tensor,
    source_valid: Tensor,
    *,
    maximum_depth_m: float,
) -> tuple[Tensor, Tensor]:
    """Apply the live P0 depth validity rule while preserving metre units.

    ``RGBDEncoder`` owns the one and only ``depth / maximum_depth_m``
    normalization.  Dataset adapters must not pre-normalize valid depth or the
    BC path would divide by the range twice.  Far finite pixels are masked and
    zero-filled exactly like the current P0-C runtime observation adapter.
    """

    if not torch.is_floating_point(raw_depth_m):
        raise PolicyDatasetContractError("raw depth must be floating point metres")
    if source_valid.shape != raw_depth_m.shape:
        raise PolicyDatasetContractError("depth-valid mask must match raw depth")
    if source_valid.dtype is not torch.bool:
        raise PolicyDatasetContractError("depth-valid mask must be bool")
    if not math.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0:
        raise PolicyDatasetContractError("maximum_depth_m must be positive")
    corrupt = torch.isnan(raw_depth_m) | torch.isneginf(raw_depth_m) | (
        torch.isfinite(raw_depth_m) & (raw_depth_m < 0.0)
    )
    if bool(corrupt.any()):
        raise PolicyDatasetContractError("raw depth contains NaN, -Inf, or negative values")
    valid = (
        source_valid
        & torch.isfinite(raw_depth_m)
        & (raw_depth_m > 0.0)
        & (raw_depth_m <= maximum_depth_m)
    )
    metric_depth = torch.where(valid, raw_depth_m, torch.zeros_like(raw_depth_m))
    return metric_depth, valid


def project_policy_actions_to_controller_surface(
    actions: Tensor,
    *,
    mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    hysteresis: GripperHysteresisConfig = GripperHysteresisConfig(),
    initial_intent: AbstractGripperIntent = AbstractGripperIntent.OPEN,
    initial_closed: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Deterministically derive historical *controller-surface* requests.

    The result is still Cartesian action plus an abstract gripper sign.  It is
    intentionally not an articulation-target record.  The second tensor is a
    boolean CLOSED intent trace, which preserves the hysteresis interpretation
    in a dataset without exposing mechanism internals.
    """

    if actions.ndim != 3 or actions.shape[-1] != mode.dimension:
        raise PolicyDatasetContractError("policy action sequence must be [B,T,action_dim]")
    if not torch.is_floating_point(actions) or not bool(torch.isfinite(actions).all()):
        raise PolicyDatasetContractError("policy actions must be finite floating point")
    if bool((actions[..., :-1].abs() > 1.0).any()) or bool(
        (actions[..., -1] < 0.0).any() | (actions[..., -1] > 1.0).any()
    ):
        raise PolicyDatasetContractError("policy action values are outside their normalised range")
    batch, time, _ = actions.shape
    result = torch.zeros(
        batch, time, 7, dtype=actions.dtype, device=actions.device
    )
    result[..., :3] = actions[..., :3]
    if mode is PolicyActionMode.SE3_GRIPPER_7D:
        result[..., 3:6] = actions[..., 3:6]
    if initial_closed is None:
        closed = torch.full(
            (batch,),
            initial_intent is AbstractGripperIntent.CLOSE,
            dtype=torch.bool,
            device=actions.device,
        )
    else:
        if initial_closed.shape != (batch,) or initial_closed.dtype is not torch.bool:
            raise PolicyDatasetContractError("initial_closed must be bool [B]")
        closed = initial_closed.to(device=actions.device).clone()
    closed_trace = torch.empty(batch, time, dtype=torch.bool, device=actions.device)
    for step in range(time):
        probability = actions[:, step, -1]
        closed = torch.where(
            probability > hysteresis.close_threshold,
            torch.ones_like(closed),
            torch.where(
                probability < hysteresis.open_threshold,
                torch.zeros_like(closed),
                closed,
            ),
        )
        closed_trace[:, step] = closed
        result[:, step, 6] = torch.where(
            closed,
            torch.full_like(probability, -1.0),
            torch.full_like(probability, 1.0),
        )
    return result, closed_trace


@dataclass(frozen=True)
class HighLevelPolicyTrajectory:
    """One serialisable BC trajectory contract without individual joint actions.

    ``evaluation_phase_labels`` is useful for reporting Approach→Place
    progression, but it is not included in :class:`PolicyObservation` and may
    never be concatenated into the actor input.
    """

    observation: PolicyObservation
    policy_action: Tensor
    controller_surface_action_7d: Tensor
    gripper_closed_intent: Tensor
    evaluation_phase_labels: tuple[tuple[TaskPhase, ...], ...]
    data_semantics: PolicyDataSemantics
    data_semantic_fingerprint: str
    initial_gripper_closed: Tensor
    initial_previous_policy_action: Tensor
    schema: str = HIGH_LEVEL_DEMONSTRATION_SCHEMA

    def validate(self) -> None:
        if self.data_semantic_fingerprint != self.data_semantics.fingerprint():
            raise PolicyDatasetContractError(
                "trajectory data semantic fingerprint does not match its stored contract"
            )
        self.observation.assert_compatible_data_semantics(self.data_semantics)
        self.observation.validate(self.data_semantics.observation_config)
        expected = (
            *self.observation.right_wrist_rgb.shape[:2],
            self.data_semantics.action_mode.dimension,
        )
        if self.policy_action.shape != expected:
            raise PolicyDatasetContractError("policy_action shape does not match observation/action mode")
        batch, time = self.policy_action.shape[:2]
        if self.initial_previous_policy_action.shape != (
            batch,
            self.data_semantics.action_mode.dimension,
        ):
            raise PolicyDatasetContractError(
                "initial_previous_policy_action must be [batch, action_dim]"
            )
        if (
            not torch.is_floating_point(self.initial_previous_policy_action)
            or not bool(torch.isfinite(self.initial_previous_policy_action).all())
        ):
            raise PolicyDatasetContractError(
                "initial_previous_policy_action must be finite floating point"
            )
        initial = self.initial_previous_policy_action
        if bool((initial[..., :-1].abs() > 1.0).any()) or bool(
            (initial[..., -1] < 0.0).any() | (initial[..., -1] > 1.0).any()
        ):
            raise PolicyDatasetContractError(
                "initial_previous_policy_action is outside the policy action range"
            )
        if not torch.allclose(
            self.observation.previous_policy_action[:, 0], initial, rtol=0.0, atol=1.0e-6
        ):
            raise PolicyDatasetContractError(
                "initial previous action must agree with the first actor observation"
            )
        if time > 1 and not torch.allclose(
            self.observation.previous_policy_action[:, 1:],
            self.policy_action[:, :-1],
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise PolicyDatasetContractError(
                "previous_policy_action must equal the preceding branch-local policy action"
            )
        reset_mask = self.observation.hidden_reset_mask
        if reset_mask is None or not bool(reset_mask[:, 0].all()):
            raise PolicyDatasetContractError(
                "P1 trajectories require a GRU reset at their first row"
            )
        if time > 1 and bool(reset_mask[:, 1:].any()):
            raise PolicyDatasetContractError(
                "P1 trajectories cannot contain an unmodelled mid-sequence GRU reset"
            )
        if (
            self.initial_gripper_closed.shape != (batch,)
            or self.initial_gripper_closed.dtype is not torch.bool
        ):
            raise PolicyDatasetContractError(
                "initial_gripper_closed must be an explicit bool [batch] latch state"
            )
        observed_initial_closed = torch.isclose(
            self.observation.current_gripper_state[:, 0, 0],
            torch.ones(
                (),
                dtype=self.observation.current_gripper_state.dtype,
                device=self.observation.current_gripper_state.device,
            ),
        )
        if not torch.equal(self.initial_gripper_closed, observed_initial_closed):
            raise PolicyDatasetContractError(
                "initial_gripper_closed must agree with the first abstract gripper state"
            )
        projected, closed = project_policy_actions_to_controller_surface(
            self.policy_action,
            mode=self.data_semantics.action_mode,
            hysteresis=self.data_semantics.gripper_hysteresis,
            initial_closed=self.initial_gripper_closed,
        )
        if self.controller_surface_action_7d.shape != projected.shape or not torch.allclose(
            self.controller_surface_action_7d, projected
        ):
            raise PolicyDatasetContractError(
                "controller surface action must be the exact branch-local hysteresis expansion"
            )
        if self.gripper_closed_intent.shape != closed.shape or not torch.equal(
            self.gripper_closed_intent, closed
        ):
            raise PolicyDatasetContractError("gripper intent trace does not match hysteresis expansion")
        if len(self.evaluation_phase_labels) != batch or any(
            len(row) != time for row in self.evaluation_phase_labels
        ):
            raise PolicyDatasetContractError("phase label metadata must match [B,T]")
        if self.schema != HIGH_LEVEL_DEMONSTRATION_SCHEMA:
            raise PolicyDatasetContractError("unexpected high-level demonstration schema")


__all__ = [
    "HIGH_LEVEL_DEMONSTRATION_SCHEMA",
    "LEGACY_CANONICAL_POLICY7_SCHEMA",
    "LEGACY_KEYBOARD_POLICY8_SCHEMA",
    "HighLevelPolicyTrajectory",
    "PolicyDatasetContractError",
    "TaskPhase",
    "legacy_gripper_sign_to_probability",
    "policy_action_schema",
    "prepare_metric_depth_for_policy",
    "project_legacy_policy7_to_high_level",
    "project_legacy_policy8_to_high_level",
    "project_policy_actions_to_controller_surface",
]
