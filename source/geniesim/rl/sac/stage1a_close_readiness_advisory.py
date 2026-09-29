# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Frozen-student CLOSE-readiness advice with no command authority.

The deterministic distance/alignment FSM remains the *only* CLOSE authority.
This module labels a frozen student score for telemetry; it never changes the
canonical FSM CLOSE decision or the gripper command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class CloseReadinessAdvice(str, Enum):
    READY_ADVISORY = "READY_ADVISORY"
    DEFER_TO_FSM = "DEFER_TO_FSM"
    NOT_READY_ADVISORY = "NOT_READY_ADVISORY"


@dataclass(frozen=True)
class FrozenCloseReadinessAdvisory:
    """Validation-selected thresholds for an authority-free student output."""

    high_ready_threshold: float
    low_not_ready_threshold: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.high_ready_threshold < 1.0):
            raise ValueError("ADVISORY_HIGH_THRESHOLD_INVALID")
        if self.low_not_ready_threshold is not None and not (
            0.0 < self.low_not_ready_threshold < self.high_ready_threshold
        ):
            raise ValueError("ADVISORY_LOW_THRESHOLD_INVALID")

    def advise(self, student_score: float) -> CloseReadinessAdvice:
        if not math.isfinite(student_score) or not (0.0 <= student_score <= 1.0):
            raise ValueError("ADVISORY_STUDENT_SCORE_INVALID")
        if student_score >= self.high_ready_threshold:
            return CloseReadinessAdvice.READY_ADVISORY
        if self.low_not_ready_threshold is not None and student_score < self.low_not_ready_threshold:
            return CloseReadinessAdvice.NOT_READY_ADVISORY
        return CloseReadinessAdvice.DEFER_TO_FSM

    def preserve_fsm_close(self, *, fsm_close_triggered: bool) -> bool:
        """The one permitted action mapping: return the FSM decision unchanged."""
        if not isinstance(fsm_close_triggered, bool):
            raise TypeError("ADVISORY_FSM_DECISION_MUST_BE_BOOL")
        return fsm_close_triggered
