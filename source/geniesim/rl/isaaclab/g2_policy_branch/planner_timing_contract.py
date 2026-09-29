# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure timing and recurrent-mask contract for planner delivery.

This module has no Isaac, PhysX, controller, or cuRobo dependency.  It keeps
episode execution budget separate from the rows that may supervise the GRU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


PLANNER_TIMING_CONTRACT_SCHEMA = "g2_curobo_grasp_ready_timing_v1"
TASK_RELEVANT_PHASES = frozenset(
    {
        "WAYPOINT_TRACKING",
        "COARSE_REACH",
        "FINE_APPROACH",
        "MICRO_APPROACH",
        "FINAL_CONVERGENCE",
        "APPROACH",
        "PRE_GRASP",
        "CLOSE_ONSET",
        "CONTACT",
        "STABLE",
        "LIFT",
    }
)

PHASE_PROGRESS_CONTRACT = {
    "RESET": None,
    "OPEN": None,
    "COARSE_REACH": "EE_TO_PREGRASP_TARGET",
    "FINE_APPROACH": "EE_OR_PAD_TO_SAFE_HANDOFF_TARGET",
    "MICRO_APPROACH": "EE_OR_PAD_TO_DEMO_CLOSE_ONSET_REGION",
    "CLOSE": None,
    "CONTACT": "BILATERAL_CONTACT_ACQUISITION",
    "STABLE": "STABLE_CONTACT_COUNTER",
    "LIFT": "CUBE_Z_HEIGHT",
    "PLACE": "CUBE_TO_GOAL_DISTANCE",
}
NON_TRAINING_PHASES = frozenset(
    {
        "BASELINE",
        "OPEN_SETTLE",
        "FINAL_SETTLE",
        "TELEMETRY_FLUSH",
        "TIMEOUT_MARGIN",
        "ZERO_ACTION_IDLE",
        "POST_TASK_WAIT",
    }
)


@dataclass(frozen=True)
class PlannerExecutionBudget:
    waypoint_count: int
    ordinary_tracking_cap: int = 3
    critical_waypoint_count: int = 1
    critical_tracking_cap: int = 20
    open_settle_steps: int = 84
    final_convergence_cap: int = 80
    final_settle_cap: int = 20
    telemetry_flush_reserved_steps: int = 2
    margin_fraction: float = 0.25
    policy_dt_s: float = 0.02
    physics_substeps_per_policy_step: int = 10

    def __post_init__(self) -> None:
        integer_fields = (
            self.waypoint_count,
            self.ordinary_tracking_cap,
            self.critical_waypoint_count,
            self.critical_tracking_cap,
            self.open_settle_steps,
            self.final_convergence_cap,
            self.final_settle_cap,
            self.telemetry_flush_reserved_steps,
            self.physics_substeps_per_policy_step,
        )
        if any(isinstance(value, bool) or int(value) != value or value < 0 for value in integer_fields):
            raise ValueError("timing budget integer fields must be non-negative integers")
        if self.waypoint_count <= 0 or self.ordinary_tracking_cap <= 0:
            raise ValueError("waypoints and ordinary tracking cap must be positive")
        if not 0 <= self.critical_waypoint_count <= self.waypoint_count:
            raise ValueError("critical waypoint count is out of range")
        if self.critical_waypoint_count and self.critical_tracking_cap < self.ordinary_tracking_cap:
            raise ValueError("critical tracking cap cannot be smaller than ordinary cap")
        if not 0.2 <= self.margin_fraction <= 0.3:
            raise ValueError("episode margin must remain between 20 and 30 percent")
        if not math.isfinite(self.policy_dt_s) or self.policy_dt_s <= 0.0:
            raise ValueError("policy dt must be finite and positive")

    @property
    def maximum_waypoint_tracking_steps(self) -> int:
        ordinary = self.waypoint_count - self.critical_waypoint_count
        return (
            ordinary * self.ordinary_tracking_cap
            + self.critical_waypoint_count * self.critical_tracking_cap
        )

    @property
    def required_steps(self) -> int:
        return (
            self.open_settle_steps
            + self.maximum_waypoint_tracking_steps
            + self.final_convergence_cap
            + self.final_settle_cap
            + self.telemetry_flush_reserved_steps
        )

    @property
    def horizon_steps(self) -> int:
        return int(math.ceil(self.required_steps * (1.0 + self.margin_fraction)))

    def payload(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result.update(
            {
                "schema": PLANNER_TIMING_CONTRACT_SCHEMA,
                "maximum_waypoint_tracking_steps": self.maximum_waypoint_tracking_steps,
                "required_steps": self.required_steps,
                "required_duration_s": self.required_steps * self.policy_dt_s,
                "horizon_steps": self.horizon_steps,
                "horizon_duration_s": self.horizon_steps * self.policy_dt_s,
                "reserved_margin_steps": self.horizon_steps - self.required_steps,
                "maximum_required_physics_substeps": (
                    self.required_steps * self.physics_substeps_per_policy_step
                ),
            }
        )
        return result


@dataclass(frozen=True)
class AdaptiveWaypointConfig:
    """Non-authoritative quality criteria for coarse waypoint removal.

    Velocity and acceleration limits are existing hard gates.  Curvature,
    direction, and chord-deviation values only decide whether optimization is
    allowed; they never relax controller or collision safety.
    """

    joint_curvature_max_rad: float = 0.003
    ee_direction_change_max_rad: float = math.radians(5.0)
    maximum_path_deviation_m: float = 0.001
    translation_scale_m: float = 0.0225
    # Keep the planner target authority at or below the established active-joint
    # hard gate.  The saved trajectory peaks at 0.788486 rad/s, so 0.8 retains
    # the qualified trajectory without introducing a second, wider envelope.
    target_velocity_limit_rad_s: float = 0.8
    target_acceleration_limit_rad_s2: float = 10.0
    policy_dt_s: float = 0.02
    minimum_coarse_effective_fraction: float = 0.70

    def __post_init__(self) -> None:
        finite_positive = (
            self.joint_curvature_max_rad,
            self.ee_direction_change_max_rad,
            self.maximum_path_deviation_m,
            self.translation_scale_m,
            self.target_velocity_limit_rad_s,
            self.target_acceleration_limit_rad_s2,
            self.policy_dt_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("adaptive waypoint thresholds must be finite and positive")
        if not 0.0 < self.minimum_coarse_effective_fraction <= 1.0:
            raise ValueError("minimum coarse effective fraction is invalid")


def _maximum_fd(samples: np.ndarray, dt_s: float) -> tuple[float, float]:
    velocity = np.diff(samples, axis=0) / dt_s
    acceleration = np.diff(velocity, axis=0) / dt_s
    return (
        float(np.abs(velocity).max(initial=0.0)),
        float(np.abs(acceleration).max(initial=0.0)),
    )


def _direction_change(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(math.acos(cosine))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-18:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def build_adaptive_waypoint_schedule(
    q_rad: np.ndarray,
    ee_position_root_m: np.ndarray,
    *,
    coarse_waypoint_count: int,
    collision_free: Sequence[bool] | np.ndarray,
    certified_skip_edges: Mapping[tuple[int, int], bool] | None = None,
    config: AdaptiveWaypointConfig = AdaptiveWaypointConfig(),
) -> dict[str, object]:
    """Select coarse waypoints while preserving every fine/handoff waypoint.

    ``coarse_waypoint_count`` follows the saved cuRobo artifact and includes
    the duplicate coarse endpoint removed during coarse+fine composition.
    Consequently ``fine_start_index = coarse_waypoint_count - 1``.
    Collision authority is both an externally computed per-waypoint receipt
    and an explicit certificate for every newly connected skip edge.  Missing
    edge certificates conservatively preserve the original waypoint.
    """

    q = np.asarray(q_rad, dtype=np.float64)
    ee = np.asarray(ee_position_root_m, dtype=np.float64)
    collision = np.asarray(collision_free, dtype=np.bool_).reshape(-1)
    if q.ndim != 2 or ee.shape != (q.shape[0], 3) or collision.shape != (q.shape[0],):
        raise ValueError("adaptive waypoint input shape mismatch")
    if q.shape[0] < 3 or not np.isfinite(q).all() or not np.isfinite(ee).all():
        raise ValueError("adaptive waypoint input is invalid")
    fine_start = int(coarse_waypoint_count) - 1
    if not 2 <= fine_start < q.shape[0]:
        raise ValueError("coarse/fine boundary is invalid")
    if not bool(collision.all()):
        raise ValueError("adaptive skipping requires collision-free source receipts")

    selected = list(range(q.shape[0]))
    edge_certificates = dict(certified_skip_edges or {})
    skipped: list[int] = []
    skip_receipts: list[dict[str, object]] = []
    coarse_original = fine_start
    minimum_coarse = int(math.ceil(coarse_original * config.minimum_coarse_effective_fraction))
    for index in range(1, fine_start - 1):
        current_coarse = sum(candidate < fine_start for candidate in selected)
        if current_coarse <= minimum_coarse:
            break
        position = selected.index(index)
        previous_index = selected[position - 1]
        next_index = selected[position + 1]
        curvature = float(np.linalg.norm(q[index + 1] - 2.0 * q[index] + q[index - 1]))
        direction_change = _direction_change(
            ee[index] - ee[index - 1], ee[index + 1] - ee[index]
        )
        path_deviation = max(
            _point_segment_distance(ee[row], ee[previous_index], ee[next_index])
            for row in range(previous_index + 1, next_index)
        )
        normalized_delta_max = float(
            np.abs(ee[next_index] - ee[previous_index]).max() / config.translation_scale_m
        )
        candidate = selected.copy()
        candidate.pop(position)
        maximum_velocity, maximum_acceleration = _maximum_fd(
            q[candidate], config.policy_dt_s
        )
        checks = {
            "joint_curvature": curvature <= config.joint_curvature_max_rad,
            "ee_direction_change": direction_change
            <= config.ee_direction_change_max_rad,
            "collision_receipt": bool(
                collision[previous_index : next_index + 1].all()
                and edge_certificates.get((previous_index, next_index), False)
            ),
            "normalized_4d_range": normalized_delta_max <= 1.0,
            "target_velocity": maximum_velocity
            <= config.target_velocity_limit_rad_s + 1.0e-9,
            "target_acceleration": maximum_acceleration
            <= config.target_acceleration_limit_rad_s2 + 1.0e-9,
            "path_deviation": path_deviation <= config.maximum_path_deviation_m,
            "not_fine_or_handoff": index < fine_start - 1,
        }
        if all(checks.values()):
            selected = candidate
            skipped.append(index)
            skip_receipts.append(
                {
                    "skipped_index": index,
                    "previous_kept_index": previous_index,
                    "next_kept_index": next_index,
                    "joint_curvature_rad": curvature,
                    "ee_direction_change_rad": direction_change,
                    "path_deviation_m": path_deviation,
                    "normalized_4d_component_max": normalized_delta_max,
                    "candidate_maximum_velocity_rad_s": maximum_velocity,
                    "candidate_maximum_acceleration_rad_s2": maximum_acceleration,
                    "checks": checks,
                }
            )

    selected_array = np.asarray(selected, dtype=np.int64)
    maximum_velocity, maximum_acceleration = _maximum_fd(q[selected_array], config.policy_dt_s)
    coarse_effective = int(np.count_nonzero(selected_array < fine_start))
    fine_original = q.shape[0] - fine_start
    fine_effective = int(np.count_nonzero(selected_array >= fine_start))
    maximum_deviation = max(
        (float(receipt["path_deviation_m"]) for receipt in skip_receipts), default=0.0
    )
    return {
        "schema": "g2_adaptive_waypoint_schedule_v1",
        "config": asdict(config),
        "selected_indices": selected,
        "skipped_indices": skipped,
        "skip_receipts": skip_receipts,
        "original_waypoint_count": int(q.shape[0]),
        "effective_waypoint_count": len(selected),
        "skipped_waypoint_count": len(skipped),
        "fine_start_original_index": fine_start,
        "coarse_original_waypoint_count": coarse_original,
        "coarse_effective_waypoint_count": coarse_effective,
        "coarse_skip_ratio": 1.0 - coarse_effective / coarse_original,
        "coarse_effective_usage": coarse_effective / coarse_original,
        "fine_original_waypoint_count": fine_original,
        "fine_effective_waypoint_count": fine_effective,
        "fine_skip_ratio": 1.0 - fine_effective / fine_original,
        "maximum_path_deviation_m": maximum_deviation,
        "maximum_selected_target_velocity_rad_s": maximum_velocity,
        "maximum_selected_target_acceleration_rad_s2": maximum_acceleration,
        "collision_clearance_authority": (
            "EXPLICIT_PER_SKIP_EDGE_CERTIFICATE_REQUIRED"
        ),
        "certified_skip_edge_count": sum(bool(value) for value in edge_certificates.values()),
        "minimum_metric_collision_clearance_m": None,
        "fine_density_preserved": fine_effective == fine_original,
        "target_envelope_preserved": (
            maximum_velocity <= config.target_velocity_limit_rad_s + 1.0e-9
            and maximum_acceleration <= config.target_acceleration_limit_rad_s2 + 1.0e-9
        ),
    }


@dataclass(frozen=True)
class ProgressMetricConfig:
    progress_deadband_m: float = 0.00025
    stall_grace_steps: int = 8
    max_stall_steps: int = 40
    retreat_threshold_m: float = 0.001

    def __post_init__(self) -> None:
        if self.progress_deadband_m <= 0.0 or self.retreat_threshold_m <= 0.0:
            raise ValueError("progress distance thresholds must be positive")
        if self.stall_grace_steps < 0 or self.max_stall_steps <= self.stall_grace_steps:
            raise ValueError("progress stall budget is invalid")


class BestSoFarProgress:
    """Stateful metric-only best/stall/retreat classifier."""

    def __init__(self, config: ProgressMetricConfig = ProgressMetricConfig()) -> None:
        self.config = config
        self.best_distance_m: float | None = None
        self.previous_distance_m: float | None = None
        self.stall_counter = 0
        self.disabled = False

    def disable_after_contact(self) -> None:
        self.disabled = True

    def observe(self, distance_m: float) -> dict[str, object]:
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("progress distance must be finite and non-negative")
        if self.disabled:
            return {
                "progress_metric_active": False,
                "current_distance_cm": distance_m * 100.0,
                "best_distance_cm": None,
                "new_best_progress_mm": 0.0,
                "best_progress_mm": 0.0,
                "stall_counter": self.stall_counter,
                "progress_event": False,
                "stall_event": False,
                "max_stall_event": False,
                "retreat_event": False,
            }
        prior_best = self.best_distance_m
        prior_distance = self.previous_distance_m
        if prior_best is None:
            self.best_distance_m = distance_m
            new_best = 0.0
            progress_event = False
            self.stall_counter = 0
        else:
            candidate_best = min(prior_best, distance_m)
            new_best = prior_best - candidate_best
            progress_event = new_best > self.config.progress_deadband_m
            self.best_distance_m = candidate_best
            if progress_event:
                self.stall_counter = 0
            else:
                self.stall_counter += 1
        retreat_event = bool(
            prior_distance is not None
            and distance_m - prior_distance > self.config.retreat_threshold_m
        )
        self.previous_distance_m = distance_m
        return {
            "progress_metric_active": True,
            "current_distance_cm": distance_m * 100.0,
            "best_distance_cm": float(self.best_distance_m) * 100.0,
            "new_best_progress_mm": new_best * 1000.0,
            "best_progress_mm": new_best * 1000.0,
            "stall_counter": self.stall_counter,
            "progress_event": progress_event,
            "stall_event": self.stall_counter > self.config.stall_grace_steps,
            "max_stall_event": self.stall_counter >= self.config.max_stall_steps,
            "retreat_event": retreat_event,
        }


def training_valid_mask(
    phases: Sequence[str] | np.ndarray,
    *,
    zero_action_idle: Iterable[bool] | np.ndarray | None = None,
    close_onset: Iterable[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Mask execution-only waits while preserving task transitions.

    Close onset is an invariant: no requested or inferred idle classification
    may remove it from supervision.
    """

    phase = np.asarray(phases, dtype="U32").reshape(-1)
    unknown = sorted(set(phase.tolist()) - TASK_RELEVANT_PHASES - NON_TRAINING_PHASES)
    if unknown:
        raise ValueError(f"unknown execution phases: {unknown}")
    valid = np.isin(phase, tuple(TASK_RELEVANT_PHASES))
    onset = (
        np.zeros(phase.shape, dtype=np.bool_)
        if close_onset is None
        else np.asarray(tuple(close_onset), dtype=np.bool_).reshape(-1)
    )
    if onset.shape != phase.shape:
        raise ValueError("close-onset mask shape mismatch")
    idle = (
        np.zeros(phase.shape, dtype=np.bool_)
        if zero_action_idle is None
        else np.asarray(tuple(zero_action_idle), dtype=np.bool_).reshape(-1)
    )
    if idle.shape != phase.shape:
        raise ValueError("zero-action idle mask shape mismatch")
    valid &= ~idle
    valid |= onset
    if np.any(onset & ~valid):
        raise RuntimeError("close onset was masked")
    return valid


def training_mask_summary(
    mask: Iterable[bool] | np.ndarray,
    *,
    close_onset: Iterable[bool] | np.ndarray | None = None,
) -> dict[str, int | float | bool]:
    values = np.asarray(tuple(mask), dtype=np.bool_).reshape(-1)
    onset = (
        np.zeros(values.shape, dtype=np.bool_)
        if close_onset is None
        else np.asarray(tuple(close_onset), dtype=np.bool_).reshape(-1)
    )
    if onset.shape != values.shape:
        raise ValueError("close-onset mask shape mismatch")
    total = int(values.size)
    valid = int(np.count_nonzero(values))
    masked_onset = int(np.count_nonzero(onset & ~values))
    return {
        "total_episode_steps": total,
        "valid_training_steps": valid,
        "masked_idle_or_settle_steps": total - valid,
        "valid_fraction": valid / total if total else 0.0,
        "close_onset_steps": int(np.count_nonzero(onset)),
        "close_onset_steps_masked": masked_onset,
        "pass": masked_onset == 0,
    }


__all__ = [
    "AdaptiveWaypointConfig",
    "BestSoFarProgress",
    "NON_TRAINING_PHASES",
    "PHASE_PROGRESS_CONTRACT",
    "PLANNER_TIMING_CONTRACT_SCHEMA",
    "PlannerExecutionBudget",
    "ProgressMetricConfig",
    "TASK_RELEVANT_PHASES",
    "build_adaptive_waypoint_schedule",
    "training_mask_summary",
    "training_valid_mask",
]
