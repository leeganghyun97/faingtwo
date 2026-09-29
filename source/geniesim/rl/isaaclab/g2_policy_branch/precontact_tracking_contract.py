# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure waypoint tracking/catch-up contract for the contact-free branch.

This module does not own a controller, limiter, safety threshold, or physics
step.  It separates raw measured-feedback error from the bounded command and
keeps waypoint advancement subordinate to directional and magnitude checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Sequence

import numpy as np

from .precontact_contract import (
    CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
    PrecontactContractError,
)


class ExecutionSubstate(str, Enum):
    NORMAL_TRACKING = "NORMAL_TRACKING"
    TRACKING_CATCHUP = "TRACKING_CATCHUP"
    DIRECTIONAL_REEVALUATION = "DIRECTIONAL_REEVALUATION"
    FAIL_CLOSED = "FAIL_CLOSED"


def _xyz(name: str, value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not bool(np.isfinite(result).all()):
        raise PrecontactContractError(f"{name} must be finite xyz")
    return result


@dataclass(frozen=True)
class WaypointTrackingDecision:
    p_prev: tuple[float, float, float]
    p_target: tuple[float, float, float]
    p_measured: tuple[float, float, float]
    planner_direction: tuple[float, float, float]
    planner_increment_norm_m: float
    raw_residual: tuple[float, float, float]
    raw_residual_norm_m: float
    cosine_similarity: float | None
    bounded_command: tuple[float, float, float]
    bounded_command_norm_m: float
    remaining_residual: tuple[float, float, float]
    substate: ExecutionSubstate
    waypoint_advance_allowed: bool

    def payload(self) -> dict[str, object]:
        return {
            "p_prev": list(self.p_prev),
            "p_target": list(self.p_target),
            "p_measured": list(self.p_measured),
            "planner_direction": list(self.planner_direction),
            "planner_increment_norm_m": self.planner_increment_norm_m,
            "raw_residual": list(self.raw_residual),
            "raw_residual_norm_m": self.raw_residual_norm_m,
            "cosine_similarity": self.cosine_similarity,
            "bounded_command": list(self.bounded_command),
            "bounded_command_norm_m": self.bounded_command_norm_m,
            "remaining_residual": list(self.remaining_residual),
            "execution_substate": self.substate.value,
            "waypoint_advance_allowed": self.waypoint_advance_allowed,
        }


@dataclass(frozen=True)
class ResidualShapeDecision:
    parallel_residual: tuple[float, float, float]
    perpendicular_residual: tuple[float, float, float]
    parallel_norm_m: float
    perpendicular_norm_m: float
    lateral_ratio: float
    alpha: float
    beta: float
    shaped_command_pre_bound: tuple[float, float, float]
    shaped_command_pre_bound_norm_m: float
    final_bounded_command: tuple[float, float, float]
    final_bounded_command_norm_m: float
    remaining_residual: tuple[float, float, float]
    allowed: bool

    def payload(self) -> dict[str, object]:
        return {
            "r_parallel": list(self.parallel_residual),
            "r_parallel_norm_m": self.parallel_norm_m,
            "r_perp": list(self.perpendicular_residual),
            "r_perp_norm_m": self.perpendicular_norm_m,
            "lateral_ratio": self.lateral_ratio,
            "alpha": self.alpha,
            "beta": self.beta,
            "shaped_command_pre_bound": list(self.shaped_command_pre_bound),
            "shaped_command_pre_bound_norm_m": self.shaped_command_pre_bound_norm_m,
            "final_bounded_command": list(self.final_bounded_command),
            "final_bounded_command_norm_m": self.final_bounded_command_norm_m,
            "remaining_residual": list(self.remaining_residual),
            "allowed": self.allowed,
        }


def evaluate_waypoint_tracking(
    *,
    p_prev: Sequence[float],
    p_target: Sequence[float],
    p_measured: Sequence[float],
    maximum_metric_delta_m: float = CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
    normal_tracking_cosine_threshold: float | None = None,
    directional_reevaluation_cosine_threshold: float | None = None,
) -> WaypointTrackingDecision:
    """Classify one waypoint without consuming it.

    Thresholds are optional on purpose.  With no historical threshold supplied,
    only a negative cosine is considered directional failure; callers may pass
    evidence-derived candidates without changing the magnitude contract.
    """

    previous = _xyz("p_prev", p_prev)
    target = _xyz("p_target", p_target)
    measured = _xyz("p_measured", p_measured)
    maximum = float(maximum_metric_delta_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise PrecontactContractError("maximum_metric_delta_m must be positive")
    for name, threshold in (
        ("normal_tracking_cosine_threshold", normal_tracking_cosine_threshold),
        ("directional_reevaluation_cosine_threshold", directional_reevaluation_cosine_threshold),
    ):
        if threshold is not None and (not math.isfinite(float(threshold)) or not -1.0 <= float(threshold) <= 1.0):
            raise PrecontactContractError(f"{name} must be within [-1, 1]")

    direction = target - previous
    residual = target - measured
    direction_norm = float(np.linalg.norm(direction))
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1.0e-12:
        bounded = np.zeros(3, dtype=np.float64)
    else:
        bounded = residual * min(1.0, maximum / residual_norm)
    bounded_norm = float(np.linalg.norm(bounded))
    remaining = residual - bounded
    cosine: float | None
    if direction_norm <= 1.0e-12 or residual_norm <= 1.0e-12:
        cosine = None
    else:
        cosine = float(np.dot(residual, direction) / (residual_norm * direction_norm))
        cosine = max(-1.0, min(1.0, cosine))

    if cosine is not None and cosine < 0.0:
        substate = ExecutionSubstate.DIRECTIONAL_REEVALUATION
    elif (
        cosine is not None
        and directional_reevaluation_cosine_threshold is not None
        and cosine < directional_reevaluation_cosine_threshold
    ):
        substate = ExecutionSubstate.DIRECTIONAL_REEVALUATION
    elif residual_norm > maximum:
        substate = ExecutionSubstate.TRACKING_CATCHUP
    elif (
        cosine is not None
        and normal_tracking_cosine_threshold is not None
        and cosine < normal_tracking_cosine_threshold
    ):
        substate = ExecutionSubstate.DIRECTIONAL_REEVALUATION
    else:
        substate = ExecutionSubstate.NORMAL_TRACKING

    # A catch-up or directional decision never authorizes advancement.  A
    # normal residual is the only provisional acceptance state.
    return WaypointTrackingDecision(
        p_prev=tuple(float(v) for v in previous),
        p_target=tuple(float(v) for v in target),
        p_measured=tuple(float(v) for v in measured),
        planner_direction=tuple(float(v) for v in direction),
        planner_increment_norm_m=direction_norm,
        raw_residual=tuple(float(v) for v in residual),
        raw_residual_norm_m=residual_norm,
        cosine_similarity=cosine,
        bounded_command=tuple(float(v) for v in bounded),
        bounded_command_norm_m=bounded_norm,
        remaining_residual=tuple(float(v) for v in remaining),
        substate=substate,
        waypoint_advance_allowed=substate is ExecutionSubstate.NORMAL_TRACKING,
    )


def summarize_cosine_distribution(values: Iterable[float]) -> dict[str, object]:
    """Summarize historical directional evidence without selecting a threshold."""

    finite = np.asarray([float(v) for v in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "cosine_threshold_source": "HISTORICAL_DATA",
            "cosine_threshold_ready": False,
        }
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "negative_count": int(np.sum(finite < 0.0)),
        "cosine_threshold_source": "HISTORICAL_DATA",
        "cosine_threshold_ready": True,
        "normal_tracking_threshold_candidate": float(np.quantile(finite, 0.05)),
        "directional_reevaluation_threshold_candidate": float(np.quantile(finite, 0.10)),
        "negative_direction_threshold": 0.0,
    }


def cosine_weight_schedule(
    cosine_similarity: float | None,
    *,
    p05: float,
    p10: float,
) -> tuple[float, float, bool]:
    """Return evidence-derived ``(alpha, beta, allowed)`` schedule.

    P05/P10 are supplied by the caller from historical data; they are not
    production constants.  Parallel authority is retained (alpha=1).  The
    perpendicular weight is 1 at/above P10, linearly attenuated between P05
    and P10, and proportionally attenuated below P05.  Negative cosine never
    authorizes a normal command.
    """

    p05 = float(p05)
    p10 = float(p10)
    if not (math.isfinite(p05) and math.isfinite(p10) and 0.0 < p05 < p10 <= 1.0):
        raise PrecontactContractError("historical cosine quantiles must satisfy 0 < P05 < P10 <= 1")
    if cosine_similarity is None or not math.isfinite(float(cosine_similarity)):
        return 1.0, 0.0, False
    cosine = max(-1.0, min(1.0, float(cosine_similarity)))
    if cosine < 0.0:
        return 1.0, 0.0, False
    if cosine >= p10:
        beta = 1.0
    elif cosine >= p05:
        beta = (cosine - p05) / (p10 - p05)
    else:
        beta = cosine / p05
    return 1.0, max(0.0, min(1.0, beta)), True


def shape_waypoint_residual(
    *,
    p_prev: Sequence[float],
    p_target: Sequence[float],
    p_measured: Sequence[float],
    p05: float,
    p10: float,
    maximum_metric_delta_m: float = CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
) -> ResidualShapeDecision:
    """Decompose and shape a residual, then apply the hard metric bound."""

    previous = _xyz("p_prev", p_prev)
    target = _xyz("p_target", p_target)
    measured = _xyz("p_measured", p_measured)
    maximum = float(maximum_metric_delta_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise PrecontactContractError("maximum_metric_delta_m must be positive")
    direction = target - previous
    residual = target - measured
    direction_norm = float(np.linalg.norm(direction))
    residual_norm = float(np.linalg.norm(residual))
    if direction_norm <= 1.0e-12:
        parallel = np.zeros(3, dtype=np.float64)
        perpendicular = residual.copy()
        cosine = None
    else:
        unit = direction / direction_norm
        parallel = float(np.dot(residual, unit)) * unit
        perpendicular = residual - parallel
        cosine = None if residual_norm <= 1.0e-12 else float(
            np.dot(residual, direction) / (residual_norm * direction_norm)
        )
    alpha, beta, allowed = cosine_weight_schedule(cosine, p05=p05, p10=p10)
    shaped = alpha * parallel + beta * perpendicular if allowed else np.zeros(3, dtype=np.float64)
    shaped_norm = float(np.linalg.norm(shaped))
    bounded = shaped * min(1.0, maximum / shaped_norm) if shaped_norm > 1.0e-12 else np.zeros(3)
    bounded_norm = float(np.linalg.norm(bounded))
    lateral_ratio = float(np.linalg.norm(perpendicular) / residual_norm) if residual_norm > 1.0e-12 else 0.0
    return ResidualShapeDecision(
        parallel_residual=tuple(float(v) for v in parallel),
        perpendicular_residual=tuple(float(v) for v in perpendicular),
        parallel_norm_m=float(np.linalg.norm(parallel)),
        perpendicular_norm_m=float(np.linalg.norm(perpendicular)),
        lateral_ratio=lateral_ratio,
        alpha=alpha,
        beta=beta,
        shaped_command_pre_bound=tuple(float(v) for v in shaped),
        shaped_command_pre_bound_norm_m=shaped_norm,
        final_bounded_command=tuple(float(v) for v in bounded),
        final_bounded_command_norm_m=bounded_norm,
        remaining_residual=tuple(float(v) for v in residual - bounded),
        allowed=allowed,
    )


__all__ = [
    "ExecutionSubstate",
    "WaypointTrackingDecision",
    "ResidualShapeDecision",
    "evaluate_waypoint_tracking",
    "cosine_weight_schedule",
    "shape_waypoint_residual",
    "summarize_cosine_distribution",
]
