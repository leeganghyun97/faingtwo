# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Boolean-milestone HER_FORCE sampling for simulated G2 grasp replay.

``HER_FORCE`` is retained as the project feature name, but force magnitude is
not the sampling authority.  Real boolean CONTACT/BILATERAL/STABLE/LIFT tags
own replay selection.  Raw force is accepted only for immutable diagnostics
and excessive-contact analysis; it never creates a milestone and never
enters the student observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KEYBOARD_GRASP_HER_FORCE_SCHEMA = (
    "g2_keyboard_grasp_her_force_boolean_milestone_priority_v2"
)


@dataclass(frozen=True)
class KeyboardHERForceConfig:
    enabled: bool = True
    contact_ratio: float = 0.25
    stable_ratio: float = 0.15
    lift_ratio: float = 0.05
    minimum_priority: float = 1.0
    contact_priority_increment: float = 1.0
    bilateral_priority_increment: float = 1.0
    stable_priority_increment: float = 2.0
    lift_priority_increment: float = 3.0

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("keyboard grasp HER_FORCE is mandatory and cannot be disabled")
        ratios = (self.contact_ratio, self.stable_ratio, self.lift_ratio)
        if any(not 0.0 <= float(value) <= 1.0 for value in ratios):
            raise ValueError("HER_FORCE milestone ratios must be in [0,1]")
        if sum(ratios) > 1.0 + 1.0e-12:
            raise ValueError("HER_FORCE milestone ratios must sum to <= 1")
        priorities = (
            self.minimum_priority,
            self.contact_priority_increment,
            self.bilateral_priority_increment,
            self.stable_priority_increment,
            self.lift_priority_increment,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in priorities):
            raise ValueError("HER_FORCE priorities must be finite/nonnegative")
        if self.minimum_priority <= 0.0:
            raise ValueError("HER_FORCE minimum priority must be positive")


def _bool_rows(name: str, value: np.ndarray, rows: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or (rows is not None and raw.shape != (rows,)):
        raise ValueError(f"{name} must be bool [N]")
    if raw.dtype != np.bool_ and not np.all(np.isin(raw, (0, 1))):
        raise ValueError(f"{name} must be binary")
    return raw.astype(np.bool_, copy=False)


@dataclass(frozen=True)
class KeyboardHERForceBatch:
    selection_priority: np.ndarray
    contact: np.ndarray
    bilateral_contact: np.ndarray
    stable: np.ndarray
    lift: np.ndarray
    safe_measurement: np.ndarray
    total_normal_force_n: np.ndarray

    def metrics(self) -> dict[str, float | int | str]:
        rows = int(self.contact.size)
        ratio = lambda value: float(np.count_nonzero(value) / rows) if rows else 0.0
        force = self.total_normal_force_n
        return {
            "schema": KEYBOARD_GRASP_HER_FORCE_SCHEMA,
            "her_force/enabled": 1,
            "her_force/role": "boolean_milestone_sampling_priority_only",
            "her_force/primary_authority": "BOOLEAN_CONTACT",
            "her_force/force_priority_used": 0,
            "her_force/row_count": rows,
            "her_force/safe_measurement_rows": int(self.safe_measurement.sum()),
            "replay/contact_ratio": ratio(self.contact),
            "replay/bilateral_ratio": ratio(self.bilateral_contact),
            "replay/stable_ratio": ratio(self.stable),
            "replay/lift_ratio": ratio(self.lift),
            "force/mean_n": float(force.mean()) if force.size else 0.0,
            "force/p95_n": float(np.percentile(force, 95)) if force.size else 0.0,
            "force/role": "raw_diagnostic_safety_auxiliary_only",
            "her_force/priority_mean": float(self.selection_priority.mean())
            if rows
            else 0.0,
            "her_force/priority_max": float(self.selection_priority.max())
            if rows
            else 0.0,
        }

    def sampling_probability(self) -> np.ndarray:
        weights = np.asarray(self.selection_priority, dtype=np.float64)
        if weights.size == 0 or not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("HER_FORCE selection priorities are invalid")
        return (weights / float(weights.sum())).astype(np.float32)

    def sample_indices(
        self,
        count: int,
        *,
        seed: int,
        config: KeyboardHERForceConfig = KeyboardHERForceConfig(),
    ) -> np.ndarray:
        """Sample real rows with requested milestone quotas when available.

        Missing classes are never fabricated.  Their unused quota is filled
        from the real-row population using boolean-derived priority weights.
        """

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("HER_FORCE sample count must be a positive integer")
        generator = np.random.default_rng(seed)
        selected: list[int] = []
        requests = (
            (self.lift, int(round(count * config.lift_ratio))),
            (self.stable & ~self.lift, int(round(count * config.stable_ratio))),
            (
                self.contact & ~self.stable,
                int(round(count * config.contact_ratio)),
            ),
        )
        for mask, quota in requests:
            candidates = np.flatnonzero(mask & self.safe_measurement)
            if quota > 0 and candidates.size:
                selected.extend(
                    int(value)
                    for value in generator.choice(candidates, size=quota, replace=True)
                )
        remaining = count - len(selected)
        if remaining > 0:
            selected.extend(
                int(value)
                for value in generator.choice(
                    self.contact.size,
                    size=remaining,
                    replace=True,
                    p=self.sampling_probability(),
                )
            )
        generator.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)


def compute_keyboard_her_force_priority(
    *,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    stable: np.ndarray,
    lift: np.ndarray,
    safety_measurement_valid: np.ndarray,
    safety_pass: np.ndarray,
    contact_force_by_side_n: np.ndarray | None = None,
    config: KeyboardHERForceConfig | None = None,
) -> KeyboardHERForceBatch:
    """Build priority from real boolean milestones; force is diagnostic only."""

    resolved = KeyboardHERForceConfig() if config is None else config
    if not resolved.enabled:
        raise ValueError("keyboard grasp HER_FORCE is mandatory and cannot be disabled")
    left = _bool_rows("left_contact", left_contact)
    rows = int(left.size)
    right = _bool_rows("right_contact", right_contact, rows)
    stable_rows = _bool_rows("stable", stable, rows)
    lift_rows = _bool_rows("lift", lift, rows)
    measurement_valid = _bool_rows(
        "safety_measurement_valid", safety_measurement_valid, rows
    )
    safe = _bool_rows("safety_pass", safety_pass, rows)
    contact = left | right
    bilateral = left & right
    if np.any(stable_rows & ~bilateral):
        raise ValueError("STABLE must be a subset of BILATERAL_CONTACT")
    if np.any(lift_rows & ~stable_rows):
        raise ValueError("LIFT must be a subset of STABLE")
    if contact_force_by_side_n is None:
        forces = np.zeros((rows, 2), dtype=np.float64)
    else:
        forces = np.asarray(contact_force_by_side_n, dtype=np.float64)
        if (
            forces.shape != (rows, 2)
            or not np.isfinite(forces).all()
            or np.any(forces < 0.0)
        ):
            raise ValueError("contact_force_by_side_n must be finite nonnegative [N,2]")
    safe_measurement = measurement_valid & safe
    priority = np.full(rows, resolved.minimum_priority, dtype=np.float64)
    priority += safe_measurement * contact * resolved.contact_priority_increment
    priority += (
        safe_measurement * bilateral * resolved.bilateral_priority_increment
    )
    priority += safe_measurement * stable_rows * resolved.stable_priority_increment
    priority += safe_measurement * lift_rows * resolved.lift_priority_increment
    return KeyboardHERForceBatch(
        selection_priority=priority.astype(np.float32),
        contact=contact,
        bilateral_contact=bilateral,
        stable=stable_rows,
        lift=lift_rows,
        safe_measurement=safe_measurement,
        total_normal_force_n=forces.sum(axis=1).astype(np.float32),
    )


def keyboard_her_force_contract() -> dict[str, object]:
    config = KeyboardHERForceConfig()
    return {
        "schema": KEYBOARD_GRASP_HER_FORCE_SCHEMA,
        "HER_FORCE_ENABLED": True,
        "HER_FORCE_CONTACT_RATIO": config.contact_ratio,
        "HER_FORCE_STABLE_RATIO": config.stable_ratio,
        "HER_FORCE_LIFT_RATIO": config.lift_ratio,
        "geometric_her_enabled": False,
        "role": "boolean_milestone_sampling_priority_only",
        "primary_authority": "BOOLEAN_CONTACT",
        "taxonomy": ["NO_CONTACT", "FIRST_CONTACT", "BILATERAL", "STABLE", "LIFT"],
        "force_role": "raw_logging_diagnostic_excessive_contact_safety_only",
        "force_priority_used": False,
        "real_rows_only": True,
        "fabricated_transition_allowed": False,
        "reward_unchanged": True,
        "success_unchanged": True,
        "student_input": False,
        "unsafe_or_unmeasured_boost": False,
    }


__all__ = [
    "KEYBOARD_GRASP_HER_FORCE_SCHEMA",
    "KeyboardHERForceBatch",
    "KeyboardHERForceConfig",
    "compute_keyboard_her_force_priority",
    "keyboard_her_force_contract",
]
