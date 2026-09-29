# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exception-safe terminal-transition contract for Stage-1A.

This module contains no Isaac imports.  The live 500-Hz detector creates the
receipt from the sample that triggered the immutable hard-stop predicate and
raises :class:`RuntimeHardstop`.  The canonical 50-Hz owner may then commit
that *real* partial interval as a terminal replay row without advancing
physics, guessing a successor state, or changing the safety predicate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


RUNTIME_HARDSTOP_REASON = "RUNTIME_HARDSTOP"
NOMINAL_PHYSICS_SUBSTEPS = 10
PHYSICS_DT_S = 0.002


class ActionConsumptionState(str, Enum):
    """Canonical action lifecycle exposed to replay and audit artifacts."""

    NOT_SUBMITTED = "NOT_SUBMITTED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_CONSUMED = "PARTIALLY_CONSUMED"
    FULLY_CONSUMED = "FULLY_CONSUMED"
    PARTIAL_TERMINATED = "PARTIAL_TERMINATED"


def _finite_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a nonempty finite sequence")
    return result


@dataclass(frozen=True)
class RuntimeHardstopReceipt:
    """Read-only evidence captured at the exact triggering physics sample."""

    physics_step_index: int
    physics_substep_index: int
    control_step_index: int
    consumed_substeps: int
    nominal_substeps: int
    terminal_physics_timestamp_s: float
    q_rad: tuple[float, ...]
    qd_rad_s: tuple[float, ...]
    qdd_rad_s2: tuple[float, ...]
    idx83_limit_margin_rad: float
    hardstop_joint: str
    hardstop_joint_index: int
    hardstop_reason: str
    contact: bool
    bilateral: bool
    stable: bool
    last_real_sensor_timestamps: Mapping[str, float]
    terminal_observation_is_real: bool = True

    def __post_init__(self) -> None:
        if type(self.physics_step_index) is not int or self.physics_step_index < 1:
            raise ValueError("physics_step_index must be a positive 1-based index")
        if (
            type(self.physics_substep_index) is not int
            or self.physics_substep_index != self.physics_step_index - 1
        ):
            raise ValueError("physics_substep_index must be the matching 0-based index")
        if type(self.control_step_index) is not int:
            raise ValueError("control_step_index must be int")
        if (
            type(self.nominal_substeps) is not int
            or self.nominal_substeps != NOMINAL_PHYSICS_SUBSTEPS
            or type(self.consumed_substeps) is not int
            or not 1 <= self.consumed_substeps <= self.nominal_substeps
            or self.consumed_substeps != self.physics_step_index
        ):
            raise ValueError("invalid partial physics interval")
        if (
            not math.isfinite(float(self.terminal_physics_timestamp_s))
            or float(self.terminal_physics_timestamp_s) < 0.0
        ):
            raise ValueError("terminal physics timestamp must be finite/nonnegative")
        q = _finite_tuple("q_rad", self.q_rad)
        qd = _finite_tuple("qd_rad_s", self.qd_rad_s)
        qdd = tuple(float(value) for value in self.qdd_rad_s2)
        if len(q) != len(qd) or len(q) != len(qdd):
            raise ValueError("q/qd/qdd widths differ")
        # The first interval sample may not have an FD estimate for every
        # joint, but the triggering joint itself must always be finite.
        if not 0 <= self.hardstop_joint_index < len(qdd) or not math.isfinite(
            qdd[self.hardstop_joint_index]
        ):
            raise ValueError("triggering hard-stop qdd is not finite")
        if not math.isfinite(float(self.idx83_limit_margin_rad)):
            raise ValueError("idx83 limit margin must be finite")
        if not isinstance(self.hardstop_joint, str) or not self.hardstop_joint:
            raise ValueError("hardstop_joint must be nonempty")
        if self.hardstop_reason != RUNTIME_HARDSTOP_REASON:
            raise ValueError("unsupported hard-stop reason")
        if any(type(value) is not bool for value in (self.contact, self.bilateral, self.stable)):
            raise ValueError("contact milestone values must be bool")
        if self.bilateral and not self.contact:
            raise ValueError("bilateral contact implies contact")
        if self.stable and not self.bilateral:
            raise ValueError("stable implies bilateral contact")
        timestamps = dict(self.last_real_sensor_timestamps)
        if any(
            not isinstance(name, str)
            or not name
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for name, value in timestamps.items()
        ):
            raise ValueError("sensor timestamps must be named finite values")
        if not self.terminal_observation_is_real:
            raise ValueError("synthetic terminal observation is forbidden")

    @property
    def execution_fraction(self) -> float:
        return self.consumed_substeps / self.nominal_substeps

    @property
    def elapsed_time_s(self) -> float:
        return self.consumed_substeps * PHYSICS_DT_S

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "action_consumption": ActionConsumptionState.PARTIAL_TERMINATED.value,
                "execution_fraction": self.execution_fraction,
                "elapsed_time_s": self.elapsed_time_s,
            }
        )
        return result


class RuntimeHardstop(RuntimeError):
    """Safety exception carrying the already-captured real terminal sample."""

    def __init__(self, receipt: RuntimeHardstopReceipt) -> None:
        if not isinstance(receipt, RuntimeHardstopReceipt):
            raise TypeError("RuntimeHardstop requires RuntimeHardstopReceipt")
        self.runtime_hardstop_receipt = receipt
        self.canonical_consumption_receipt: Any | None = None
        super().__init__(
            f"RUNTIME_HARDSTOP:{receipt.hardstop_joint}:"
            f"substep={receipt.consumed_substeps}:"
            f"qdd={receipt.qdd_rad_s2[receipt.hardstop_joint_index]}"
        )


def elapsed_discount(gamma_50hz: float, consumed_substeps: int) -> float:
    if not math.isfinite(float(gamma_50hz)) or not 0.0 < gamma_50hz <= 1.0:
        raise ValueError("gamma_50hz must be in (0,1]")
    if type(consumed_substeps) is not int or not 1 <= consumed_substeps <= 10:
        raise ValueError("consumed_substeps must be in [1,10]")
    return float(gamma_50hz) ** (consumed_substeps / 10.0)


__all__ = [
    "ActionConsumptionState",
    "NOMINAL_PHYSICS_SUBSTEPS",
    "PHYSICS_DT_S",
    "RUNTIME_HARDSTOP_REASON",
    "RuntimeHardstop",
    "RuntimeHardstopReceipt",
    "elapsed_discount",
]
