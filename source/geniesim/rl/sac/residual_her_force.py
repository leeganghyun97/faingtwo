# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Mandatory HER_FORCE replay priority for BC-regularized residual SAC.

This is not geometric HER.  It never relabels goals, rewards, success,
actions, or student observations.  It assigns an additional sampling weight
only to measured, safe *real* rows using boolean CONTACT/BILATERAL/STABLE/LIFT
milestones.  Raw force is diagnostic-only and never owns priority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .keyboard_grasp_contract import canonical_keyboard_grasp_contract
from .keyboard_grasp_her_force import (
    KeyboardHERForceConfig,
    compute_keyboard_her_force_priority,
)


RESIDUAL_HER_FORCE_SCHEMA = "g2_residual_sac_real_row_boolean_her_force_v2"


class ResidualHERForceError(ValueError):
    """Raised when replay priority could change learner semantics or provenance."""


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidualHERForceConfig:
    enabled: bool = True
    geometric_her_enabled: bool = False
    contact_ratio: float = 0.25
    stable_ratio: float = 0.15
    lift_ratio: float = 0.05
    minimum_priority: float = 1.0

    def __post_init__(self) -> None:
        common = canonical_keyboard_grasp_contract()
        if self.enabled is not common.her_force_enabled:
            raise ResidualHERForceError("HER_FORCE is mandatory and cannot be disabled")
        if self.geometric_her_enabled is not common.geometric_her_enabled:
            raise ResidualHERForceError("geometric HER must remain disabled")
        try:
            self.to_source_config()
        except ValueError as error:
            raise ResidualHERForceError(str(error)) from error

    def to_source_config(self) -> KeyboardHERForceConfig:
        return KeyboardHERForceConfig(
            enabled=self.enabled,
            contact_ratio=self.contact_ratio,
            stable_ratio=self.stable_ratio,
            lift_ratio=self.lift_ratio,
            minimum_priority=self.minimum_priority,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HER_FORCE_SCHEMA,
            "enabled": self.enabled,
            "geometric_her_enabled": self.geometric_her_enabled,
            "HER_FORCE_CONTACT_RATIO": self.contact_ratio,
            "HER_FORCE_STABLE_RATIO": self.stable_ratio,
            "HER_FORCE_LIFT_RATIO": self.lift_ratio,
            "minimum_priority": self.minimum_priority,
            "role": "boolean_milestone_sampling_priority_only",
            "real_rows_only": True,
            "primary_authority": "BOOLEAN_CONTACT",
            "force_priority_used": False,
            "force_unit": "N",
            "force_role": "raw_logging_diagnostic_excessive_contact_safety_only",
            "reward_unchanged": True,
            "success_unchanged": True,
            "action_unchanged": True,
            "student_observation_unchanged": True,
        }


@dataclass(frozen=True)
class ResidualHERForcePriorityBatch:
    row_ids: tuple[str, ...]
    selection_priority: np.ndarray
    sampling_probability: np.ndarray
    contact: np.ndarray
    bilateral_contact: np.ndarray
    stable: np.ndarray
    lift: np.ndarray
    immutable_input_sha256: Mapping[str, str]
    config: ResidualHERForceConfig

    def __post_init__(self) -> None:
        count = len(self.row_ids)
        for name in (
            "selection_priority",
            "sampling_probability",
            "contact",
            "bilateral_contact",
            "stable",
            "lift",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,):
                raise ResidualHERForceError(f"HER_FORCE {name} shape mismatch")
            frozen = np.ascontiguousarray(value.copy())
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)
        if count <= 0 or len(set(self.row_ids)) != count:
            raise ResidualHERForceError("HER_FORCE row IDs must be nonempty and unique")
        if not np.isclose(float(self.sampling_probability.sum()), 1.0):
            raise ResidualHERForceError("HER_FORCE sampling probabilities must sum to one")

    def sample_indices(self, count: int, *, seed: int) -> np.ndarray:
        """Sample only indices; replay tensors and labels are never rewritten."""

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ResidualHERForceError("HER_FORCE sample count must be positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ResidualHERForceError("HER_FORCE seed must be an integer")
        generator = np.random.default_rng(seed)
        return generator.choice(
            len(self.row_ids), size=count, replace=True, p=self.sampling_probability
        ).astype(np.int64)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HER_FORCE_SCHEMA,
            "config": self.config.payload(),
            "row_ids_sha256": hashlib.sha256(
                json.dumps(self.row_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "immutable_input_sha256": dict(self.immutable_input_sha256),
            "contact_rows": int(self.contact.sum()),
            "bilateral_rows": int(self.bilateral_contact.sum()),
            "stable_rows": int(self.stable.sum()),
            "lift_rows": int(self.lift.sum()),
            "row_count": len(self.row_ids),
            "reward_success_action_student_observation_unchanged": True,
        }


def prepare_residual_her_force_priorities(
    *,
    row_ids: Sequence[str],
    real_row: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    stable: np.ndarray,
    lift: np.ndarray,
    contact_force_by_side_n: np.ndarray,
    safety_measurement_valid: np.ndarray,
    safety_pass: np.ndarray,
    reward: np.ndarray,
    success: np.ndarray,
    action_4d_metric_root_m: np.ndarray,
    student_observation: np.ndarray,
    config: ResidualHERForceConfig = ResidualHERForceConfig(),
) -> ResidualHERForcePriorityBatch:
    """Build immutable priority evidence for real replay rows only."""

    if not isinstance(config, ResidualHERForceConfig) or not config.enabled:
        raise ResidualHERForceError("HER_FORCE enabled config is mandatory")
    ids = tuple(str(value) for value in row_ids)
    rows = len(ids)
    real = np.asarray(real_row)
    rewards = np.asarray(reward)
    successes = np.asarray(success)
    actions = np.asarray(action_4d_metric_root_m)
    observations = np.asarray(student_observation)
    if real.shape != (rows,) or real.dtype != np.bool_ or not bool(real.all()):
        raise ResidualHERForceError(
            "HER_FORCE replay surface accepts measured real rows only"
        )
    if rewards.shape != (rows,) or not bool(np.isfinite(rewards).all()):
        raise ResidualHERForceError("HER_FORCE reward audit must be finite [N]")
    if successes.shape != (rows,) or successes.dtype != np.bool_:
        raise ResidualHERForceError("HER_FORCE success audit must be bool [N]")
    if actions.shape != (rows, 4) or not bool(np.isfinite(actions).all()):
        raise ResidualHERForceError("HER_FORCE action audit must be finite [N,4]")
    if observations.ndim < 2 or observations.shape[0] != rows or not bool(
        np.isfinite(observations).all()
    ):
        raise ResidualHERForceError(
            "HER_FORCE student observation audit must be finite with leading N"
        )
    immutable = {
        "reward": _array_sha256("reward", rewards),
        "success": _array_sha256("success", successes),
        "action_4d_metric_root_m": _array_sha256("action", actions),
        "student_observation": _array_sha256("student_observation", observations),
    }
    try:
        priority = compute_keyboard_her_force_priority(
            left_contact=left_contact,
            right_contact=right_contact,
            stable=stable,
            lift=lift,
            contact_force_by_side_n=contact_force_by_side_n,
            safety_measurement_valid=safety_measurement_valid,
            safety_pass=safety_pass,
            config=config.to_source_config(),
        )
    except ValueError as error:
        raise ResidualHERForceError(str(error)) from error
    weights = np.asarray(priority.selection_priority, dtype=np.float64)
    if weights.shape != (rows,) or not bool(np.isfinite(weights).all()) or bool(
        (weights <= 0.0).any()
    ):
        raise ResidualHERForceError("HER_FORCE selection weights are invalid")
    probability = weights / float(weights.sum())
    # Recompute the protected hashes after priority calculation.  This guards
    # against an accidental in-place implementation in future revisions.
    after = {
        "reward": _array_sha256("reward", rewards),
        "success": _array_sha256("success", successes),
        "action_4d_metric_root_m": _array_sha256("action", actions),
        "student_observation": _array_sha256("student_observation", observations),
    }
    if after != immutable:
        raise ResidualHERForceError("HER_FORCE mutated a learner tensor or label")
    return ResidualHERForcePriorityBatch(
        row_ids=ids,
        selection_priority=priority.selection_priority,
        sampling_probability=probability.astype(np.float32),
        contact=priority.contact,
        bilateral_contact=priority.bilateral_contact,
        stable=priority.stable,
        lift=priority.lift,
        immutable_input_sha256=immutable,
        config=config,
    )


__all__ = [
    "RESIDUAL_HER_FORCE_SCHEMA",
    "ResidualHERForceConfig",
    "ResidualHERForceError",
    "ResidualHERForcePriorityBatch",
    "prepare_residual_her_force_priorities",
]
