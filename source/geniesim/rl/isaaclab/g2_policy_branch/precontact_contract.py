# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Contact-free contract for the frozen Legacy Candidate-A pipeline.

This is deliberately a pure-Python policy/replay contract.  It has no Isaac,
PhysX, controller, cuRobo, or asset-loading dependency.  A caller must supply
both an observed pad-to-object distance and a planner segment-minimum-distance
receipt before it can forward an OPEN-only four-dimensional packet.

The contact guard is intentionally conservative: it prevents this branch from
entering the historical CLOSE-onset region.  It is not an assertion about the
unknown physical soft-pad contact distance, so a separate bounded live
contact-free attestation remains required before collection or training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

from .planner_timing_contract import BestSoFarProgress, ProgressMetricConfig


PRECONTACT_PIPELINE_SCHEMA = "g2_legacy_candidate_a_precontact_contract_v1"
PRECONTACT_ACTION_SCHEMA = "4D [dx,dy,dz,gripper]"
CONTACT_FREE_EXECUTION_MODE = "LEGACY_CANDIDATE_A_CONTACT_FREE_PRECONTACT"
CONTACT_GUARD_DISTANCE_M = 0.025051395199026397
# This is the already attested P0-A Cartesian micro-command magnitude
# (normalised root-X ``+0.20`` times the production 0.0225 m scale).  It is a
# deterministic re-timing bound for the contact-free smoke, not a new safety
# threshold or controller limit.
CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M = 0.0045
# A numerical classification tolerance for a target that was constructed from
# the same handoff sphere in ``pad_frame_open_handoff_ee_target``.  It is not
# a geometric guard expansion: the returned target is still cut at the
# analytic sphere intersection.  The tolerance only prevents cancellation in
# ``||pad + delta - cube||`` from turning an exact boundary target into a
# falsely "does not reach" route.
HANDOFF_BOUNDARY_NUMERICAL_TOLERANCE_M = 1.0e-12


class PrecontactContractError(ValueError):
    """Raised for a row or route that cannot remain contact-free."""


def validate_final_metric_action_delta(
    metric_delta_root_m: Sequence[float],
    *,
    maximum_translation_m: float = CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
) -> float:
    """Validate the final measured-feedback Cartesian command before ingress.

    Planner subdivision bounds target-to-target distance.  The canonical
    handoff instead computes ``planner_target - measured_ee``; tracking lag can
    therefore make that final command larger than a subdivision.  This pure
    guard is deliberately placed at the last metric representation boundary:
    it rejects the command before packet construction/``env.step`` and never
    clips or changes controller semantics.
    """

    delta = _finite_xyz("final_metric_action_delta_root_m", metric_delta_root_m)
    maximum = _finite_distance("maximum_translation_m", maximum_translation_m)
    if maximum <= 0.0:
        raise PrecontactContractError("maximum_translation_m must be positive")
    magnitude = float(np.linalg.norm(delta))
    if magnitude > maximum + 1.0e-12:
        raise PrecontactContractError(
            "FINAL_METRIC_ACTION_DELTA_EXCEEDS_PRECONTACT_CONTRACT"
        )
    return magnitude


def evaluate_contact_free_verdict(
    *,
    required_true: Mapping[str, bool],
    required_false: Mapping[str, bool],
    execution_mode: str,
) -> bool:
    """Evaluate the contact-free mode without inverting expected negatives.

    ``required_false`` contains *occurred* events (for example
    ``contact_occurred``), not the desired state.  This keeps the OPEN-only
    semantics separate from the CONTACT/M2 verdict vocabulary.
    """

    if execution_mode != CONTACT_FREE_EXECUTION_MODE:
        raise PrecontactContractError("CONTACT_FREE_EXECUTION_MODE_REQUIRED")
    return bool(all(bool(value) for value in required_true.values())) and not any(
        bool(value) for value in required_false.values()
    )


class PrecontactPhase(str, Enum):
    REACH = "REACH"
    PREGRASP = "PREGRASP"
    FINE_APPROACH = "FINE_APPROACH"
    OPEN_HANDOFF = "OPEN_HANDOFF"
    BC_MICRO_APPROACH = "BC_MICRO_APPROACH"
    CONTACT_BLOCKED = "CONTACT_BLOCKED"


PHASE_DISTANCE_METRIC: Mapping[PrecontactPhase, str | None] = {
    PrecontactPhase.REACH: "EE_GRASP_CENTER_TO_PREGRASP_TARGET",
    PrecontactPhase.PREGRASP: None,
    PrecontactPhase.FINE_APPROACH: "DISTAL_PAD_MIDPOINT_TO_GRASP_TARGET",
    PrecontactPhase.OPEN_HANDOFF: "DISTAL_PAD_MIDPOINT_TO_GRASP_TARGET",
    PrecontactPhase.BC_MICRO_APPROACH: "DISTAL_PAD_MIDPOINT_TO_GRASP_TARGET",
    PrecontactPhase.CONTACT_BLOCKED: None,
}


def _finite_distance(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise PrecontactContractError(f"{name} must be finite and non-negative metres")
    return value


def _finite_xyz(name: str, value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not bool(np.isfinite(result).all()):
        raise PrecontactContractError(f"{name} must be a finite xyz vector in metres")
    return result


def segment_minimum_pad_object_distance_m(
    *,
    start_pad_root_m: Sequence[float],
    end_pad_root_m: Sequence[float],
    object_root_m: Sequence[float],
) -> float:
    """Return the exact point-to-segment minimum in the root frame.

    The contact-free runtime uses this as the mandatory *planned segment*
    clearance receipt.  It intentionally models only the commanded OPEN
    Cartesian translation of the measured distal-pad midpoint; it makes no
    claim about soft-pad deformation or physical contact onset.
    """

    start = _finite_xyz("start_pad_root_m", start_pad_root_m)
    end = _finite_xyz("end_pad_root_m", end_pad_root_m)
    object_position = _finite_xyz("object_root_m", object_root_m)
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1.0e-18:
        return float(np.linalg.norm(object_position - start))
    interpolation = float(np.dot(object_position - start, direction) / denominator)
    interpolation = min(1.0, max(0.0, interpolation))
    closest = start + interpolation * direction
    return float(np.linalg.norm(object_position - closest))


def split_open_translation_to_handoff(
    *,
    start_ee_root_m: Sequence[float],
    start_pad_root_m: Sequence[float],
    object_root_m: Sequence[float],
    desired_ee_root_m: Sequence[float],
    handoff_distance_m: float,
    max_translation_per_segment_m: float = CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
) -> tuple[tuple[tuple[float, float, float], ...], bool]:
    """Re-time one existing planner segment without crossing OPEN handoff.

    With orientation and gripper state frozen, the predicted pad midpoint is
    translated by the same root-frame delta as the EE target.  If an existing
    planner segment would enter the handoff sphere, the returned path ends at
    its *first* intersection with that sphere.  Every returned segment is no
    longer than the already-attested 4.5 mm P0-A command.

    ``handoff_reached`` never means CLOSE/contact authority: it only marks the
    terminal OPEN handoff boundary.  Callers still must invoke
    :func:`gate_precontact_step` before forwarding each result.
    """

    start_ee = _finite_xyz("start_ee_root_m", start_ee_root_m)
    start_pad = _finite_xyz("start_pad_root_m", start_pad_root_m)
    object_position = _finite_xyz("object_root_m", object_root_m)
    desired_ee = _finite_xyz("desired_ee_root_m", desired_ee_root_m)
    handoff = _finite_distance("handoff_distance_m", handoff_distance_m)
    maximum = _finite_distance(
        "max_translation_per_segment_m", max_translation_per_segment_m
    )
    if maximum <= 0.0:
        raise PrecontactContractError("max_translation_per_segment_m must be positive")
    current_distance = float(np.linalg.norm(start_pad - object_position))
    if current_distance <= handoff:
        raise PrecontactContractError("start_pad_is_at_or_inside_open_handoff")

    full_delta = desired_ee - start_ee
    full_length = float(np.linalg.norm(full_delta))
    if full_length <= 1.0e-15:
        return (), False
    predicted_end_pad = start_pad + full_delta

    # Find the first t in (0, 1] at which the translated pad midpoint reaches
    # the handoff sphere.  A quadratic is used rather than sampled geometry so
    # the handoff cannot be skipped by the re-timing grid.
    relative_start = start_pad - object_position
    a = float(np.dot(full_delta, full_delta))
    b = 2.0 * float(np.dot(relative_start, full_delta))
    c = float(np.dot(relative_start, relative_start) - handoff * handoff)
    discriminant = b * b - 4.0 * a * c
    endpoint_distance = float(np.linalg.norm(predicted_end_pad - object_position))
    terminal_fraction = 1.0
    reached = False
    if (
        endpoint_distance <= handoff + HANDOFF_BOUNDARY_NUMERICAL_TOLERANCE_M
        and discriminant >= 0.0
    ):
        root = math.sqrt(max(0.0, discriminant))
        candidates = sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        valid = [item for item in candidates if 0.0 < item <= 1.0 + 1.0e-12]
        if not valid:
            raise PrecontactContractError("handoff_crossing_root_not_found")
        terminal_fraction = min(1.0, valid[0])
        reached = True

    terminal_delta = full_delta * terminal_fraction
    terminal_length = float(np.linalg.norm(terminal_delta))
    count = max(1, int(math.ceil(terminal_length / maximum)))
    targets = tuple(
        tuple((start_ee + terminal_delta * (index / count)).tolist())
        for index in range(1, count + 1)
    )
    return targets, reached


def pad_frame_open_handoff_ee_target(
    *,
    start_ee_root_m: Sequence[float],
    start_pad_root_m: Sequence[float],
    object_root_m: Sequence[float],
    handoff_distance_m: float,
) -> tuple[float, float, float]:
    """Return the fixed-orientation EE target for a pad-frame OPEN handoff.

    A cuRobo plan is normally expressed at the EE frame, while the OPEN-only
    contact boundary is defined at the distal-pad midpoint.  With orientation
    frozen, a Cartesian EE translation moves that midpoint by the exact same
    root-frame vector.  This helper therefore preserves the measured EE→pad
    offset and chooses the point on the *current* pad→object ray at the
    existing handoff distance.  It does not emit an action, run IK, alter a
    controller target, or authorize CLOSE/contact.
    """

    start_ee = _finite_xyz("start_ee_root_m", start_ee_root_m)
    start_pad = _finite_xyz("start_pad_root_m", start_pad_root_m)
    object_position = _finite_xyz("object_root_m", object_root_m)
    handoff = _finite_distance("handoff_distance_m", handoff_distance_m)
    if handoff <= 0.0:
        raise PrecontactContractError("handoff_distance_m must be positive")
    pad_to_object = object_position - start_pad
    current_distance = float(np.linalg.norm(pad_to_object))
    if current_distance <= handoff:
        raise PrecontactContractError("start_pad_is_at_or_inside_open_handoff")
    direction = pad_to_object / current_distance
    # ``nextafter`` keeps the terminal point on the OPEN side of the first
    # intersection despite floating-point evaluation of the same norm in the
    # segmentation routine.  It is numerical representation slack, not a new
    # distance threshold.
    terminal_distance = float(np.nextafter(handoff, 0.0))
    terminal_pad = object_position - direction * terminal_distance
    return tuple(float(value) for value in (start_ee + (terminal_pad - start_pad)))


@dataclass(frozen=True)
class PrecontactDistanceContract:
    """Evidence-labelled distance bands; none are a contact-model retune."""

    # Existing planner-only receipt.  The corresponding asset identity is
    # recorded by the static audit because it is not Candidate-A evidence.
    curobo_pregrasp_distance_m: float = 0.10480012292660178
    existing_pregrasp_tolerance_m: float = 0.003
    # Candidate handoff, deliberately a range rather than a fixed grasp point.
    open_handoff_candidate_m: float = 0.030
    open_handoff_min_m: float = CONTACT_GUARD_DISTANCE_M
    open_handoff_max_m: float = 0.040
    # Frozen empirical human-BC CLOSE-onset summary.  CLOSE itself remains
    # forbidden in this branch.
    close_onset_p05_m: float = 0.013301480421770694
    close_onset_median_m: float = 0.017615531970109708
    close_onset_p95_m: float = 0.025051395199026397

    def __post_init__(self) -> None:
        for name in (
            "curobo_pregrasp_distance_m",
            "existing_pregrasp_tolerance_m",
            "open_handoff_candidate_m",
            "open_handoff_min_m",
            "open_handoff_max_m",
            "close_onset_p05_m",
            "close_onset_median_m",
            "close_onset_p95_m",
        ):
            value = _finite_distance(name, getattr(self, name))
            if value <= 0.0:
                raise PrecontactContractError(f"{name} must be positive")
        if not self.close_onset_p05_m <= self.close_onset_median_m <= self.close_onset_p95_m:
            raise PrecontactContractError("CLOSE-onset quantiles are not ordered")
        if not self.open_handoff_min_m <= self.open_handoff_candidate_m <= self.open_handoff_max_m:
            raise PrecontactContractError("OPEN handoff candidate is outside its evidence range")
        # The OPEN-only branch must stop before it enters the historical
        # CLOSE-onset population.  Equality is allowed only within numerical
        # serialization tolerance; the runtime gate itself requires >.
        if self.open_handoff_min_m + 1.0e-12 < self.close_onset_p95_m:
            raise PrecontactContractError(
                "OPEN handoff lower bound enters the historical CLOSE-onset band"
            )
        if self.curobo_pregrasp_distance_m <= self.open_handoff_max_m:
            raise PrecontactContractError("pregrasp receipt must remain outside handoff")

    @property
    def contact_guard_distance_m(self) -> float:
        """Guard at the empirical upper CLOSE-onset edge, not contact onset."""

        return self.open_handoff_min_m

    def payload(self) -> dict[str, object]:
        return {
            "schema": PRECONTACT_PIPELINE_SCHEMA,
            **asdict(self),
            "contact_guard_distance_m": self.contact_guard_distance_m,
            "contact_guard_provenance": "EMPIRICAL_CLOSE_ONSET_P95_NOT_PHYSICAL_CONTACT_DISTANCE",
            "close_live_authorized": False,
            "contact_live_authorized": False,
        }


@dataclass(frozen=True)
class PrecontactGateReceipt:
    accepted: bool
    phase: PrecontactPhase
    termination_reason: str | None
    training_transition_accepted: bool
    observed_pad_object_distance_m: float
    planned_segment_minimum_pad_object_distance_m: float | None
    contact_guard_distance_m: float
    gripper_open_only: bool
    contact_detected: bool
    bilateral_contact_detected: bool


def validate_open_only_action(action_4d: Sequence[float]) -> tuple[float, float, float, float]:
    """Reject CLOSE rather than clipping or silently replacing it with OPEN."""

    if len(action_4d) != 4:
        raise PrecontactContractError("precontact action must have exact width 4")
    action = tuple(float(value) for value in action_4d)
    if not all(math.isfinite(value) for value in action):
        raise PrecontactContractError("precontact action must be finite")
    if any(abs(value) > 1.0 for value in action[:3]):
        raise PrecontactContractError("precontact translation action is outside normalized range")
    # The branch's abstract convention is 0=open, 1=close.  Exact zero avoids
    # applying a hysteresis-dependent interpretation at the safety boundary.
    if action[3] != 0.0:
        raise PrecontactContractError("PREMATURE_CLOSE_FAIL_CLOSED")
    return action


def gate_precontact_step(
    *,
    phase: PrecontactPhase,
    action_4d: Sequence[float],
    observed_pad_object_distance_m: float,
    planned_segment_minimum_pad_object_distance_m: float | None,
    contact_detected: bool,
    bilateral_contact_detected: bool,
    gripper_is_open: bool,
    distance: PrecontactDistanceContract = PrecontactDistanceContract(),
) -> PrecontactGateReceipt:
    """Fail closed before forwarding an unsafe precontact packet.

    The segment-minimum receipt is required because a current point distance
    alone cannot prove the next controller segment stays outside the guard.
    """

    validate_open_only_action(action_4d)
    observed = _finite_distance("observed_pad_object_distance_m", observed_pad_object_distance_m)
    minimum = (
        None
        if planned_segment_minimum_pad_object_distance_m is None
        else _finite_distance(
            "planned_segment_minimum_pad_object_distance_m",
            planned_segment_minimum_pad_object_distance_m,
        )
    )
    guard = distance.contact_guard_distance_m
    reason: str | None = None
    if phase is PrecontactPhase.CONTACT_BLOCKED:
        reason = "CONTACT_BLOCKED_PHASE"
    elif not gripper_is_open:
        reason = "GRIPPER_NOT_OPEN"
    elif contact_detected or bilateral_contact_detected:
        reason = "CONTACT_EVENT_FAIL_CLOSED"
    elif minimum is None:
        reason = "PLANNED_SEGMENT_DISTANCE_RECEIPT_MISSING"
    elif observed <= guard or minimum <= guard:
        reason = "CONTACT_GUARD_DISTANCE_CROSSED"
    return PrecontactGateReceipt(
        accepted=reason is None,
        phase=phase,
        termination_reason=reason,
        training_transition_accepted=reason is None,
        observed_pad_object_distance_m=observed,
        planned_segment_minimum_pad_object_distance_m=minimum,
        contact_guard_distance_m=guard,
        gripper_open_only=True,
        contact_detected=bool(contact_detected),
        bilateral_contact_detected=bool(bilateral_contact_detected),
    )


def next_precontact_phase(
    *,
    current: PrecontactPhase,
    pregrasp_converged: bool,
    pad_object_distance_m: float,
    contact_detected: bool,
    distance: PrecontactDistanceContract = PrecontactDistanceContract(),
) -> PrecontactPhase:
    """Metric-based phase progression without any CLOSE transition."""

    pad_distance = _finite_distance("pad_object_distance_m", pad_object_distance_m)
    if contact_detected or pad_distance <= distance.contact_guard_distance_m:
        return PrecontactPhase.CONTACT_BLOCKED
    if current is PrecontactPhase.CONTACT_BLOCKED:
        return current
    if current is PrecontactPhase.REACH and pregrasp_converged:
        return PrecontactPhase.PREGRASP
    if current is PrecontactPhase.PREGRASP:
        return PrecontactPhase.FINE_APPROACH
    if current is PrecontactPhase.FINE_APPROACH and pad_distance <= distance.open_handoff_candidate_m:
        return PrecontactPhase.OPEN_HANDOFF
    if current is PrecontactPhase.OPEN_HANDOFF:
        return PrecontactPhase.BC_MICRO_APPROACH
    return current


@dataclass(frozen=True)
class PrecontactReplayTransition:
    """Contact-free `(o_t, a_t, o_t+1)` row with mandatory provenance fields."""

    observation: Mapping[str, object]
    action_4d: tuple[float, float, float, float]
    next_observation: Mapping[str, object]
    phase: PrecontactPhase
    ee_object_distance_m: float
    contact_flag: bool
    planner_progress: float
    safety_reject: bool
    done: bool
    termination_reason: str | None
    planner_nominal_action: tuple[float, float, float, float]
    residual_action: tuple[float, float, float, float]
    reward_placeholder: float = 0.0

    def __post_init__(self) -> None:
        if not self.observation or not self.next_observation:
            raise PrecontactContractError("replay row requires observation and next_observation")
        validate_open_only_action(self.action_4d)
        validate_open_only_action(self.planner_nominal_action)
        validate_open_only_action(self.residual_action)
        _finite_distance("ee_object_distance_m", self.ee_object_distance_m)
        if self.contact_flag:
            raise PrecontactContractError("CONTACT transition cannot enter precontact replay")
        if self.phase is PrecontactPhase.CONTACT_BLOCKED:
            raise PrecontactContractError("CONTACT_BLOCKED is terminal, never training data")
        if not math.isfinite(float(self.planner_progress)):
            raise PrecontactContractError("planner progress must be finite")
        if float(self.reward_placeholder) != 0.0:
            raise PrecontactContractError("precontact replay uses a zero reward placeholder only")
        if self.safety_reject and not self.done:
            raise PrecontactContractError("a safety-rejected row must terminate")

    def payload(self) -> dict[str, object]:
        return {
            "schema": PRECONTACT_PIPELINE_SCHEMA,
            "transition": "(o_t,a_t,o_t_plus_1)",
            "phase": self.phase.value,
            "ee_object_distance_m": float(self.ee_object_distance_m),
            "contact_flag": False,
            "planner_progress": float(self.planner_progress),
            "safety_reject": bool(self.safety_reject),
            "done": bool(self.done),
            "termination_reason": self.termination_reason,
            "action_4d": list(self.action_4d),
            "planner_nominal_action": list(self.planner_nominal_action),
            "residual_action": list(self.residual_action),
            "reward_placeholder": 0.0,
        }


class PrecontactProgressMonitor:
    """Phase-specific anti-hover telemetry; deliberately never emits reward."""

    def __init__(self, config: ProgressMetricConfig = ProgressMetricConfig()) -> None:
        self._config = config
        self._trackers = {
            metric: BestSoFarProgress(config)
            for metric in set(PHASE_DISTANCE_METRIC.values())
            if metric is not None
        }
        self._contact_latched = False

    def observe(
        self,
        *,
        phase: PrecontactPhase,
        distance_m: float,
        contact_latched: bool = False,
    ) -> dict[str, object]:
        """Return progress/stall/retreat events with reward fixed at zero."""

        value = _finite_distance("distance_m", distance_m)
        self._contact_latched = self._contact_latched or bool(contact_latched) or (
            phase is PrecontactPhase.CONTACT_BLOCKED
        )
        if self._contact_latched:
            for tracker in self._trackers.values():
                tracker.disable_after_contact()
            return {
                "phase": phase.value,
                "distance_metric": PHASE_DISTANCE_METRIC[phase],
                "approach_reward": 0.0,
                "approach_reward_authority": "OFF_UNTIL_EPISODE_END",
                "approach_reward_latch_after_contact": True,
                "current_distance_cm": value * 100.0,
                "best_distance_cm": None,
                "new_best_progress_mm": 0.0,
                "stall_counter": 0,
                "stall_event": False,
                "retreat_event": False,
            }
        metric = PHASE_DISTANCE_METRIC[phase]
        if metric is None:
            return {
                "phase": phase.value,
                "distance_metric": None,
                "approach_reward": 0.0,
                "approach_reward_authority": "METRIC_NOT_APPLICABLE",
                "approach_reward_latch_after_contact": False,
                "current_distance_cm": value * 100.0,
                "best_distance_cm": None,
                "new_best_progress_mm": 0.0,
                "stall_counter": 0,
                "stall_event": False,
                "retreat_event": False,
            }
        telemetry = self._trackers[metric].observe(value)
        return {
            "phase": phase.value,
            "distance_metric": metric,
            "approach_reward": 0.0,
            "approach_reward_authority": "METRIC_ONLY_NO_REWARD",
            "approach_reward_latch_after_contact": False,
            **telemetry,
        }


__all__ = [
    "CONTACT_FREE_EXECUTION_MODE",
    "CONTACT_GUARD_DISTANCE_M",
    "CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M",
    "PHASE_DISTANCE_METRIC",
    "PRECONTACT_ACTION_SCHEMA",
    "PRECONTACT_PIPELINE_SCHEMA",
    "PrecontactContractError",
    "PrecontactDistanceContract",
    "PrecontactGateReceipt",
    "PrecontactPhase",
    "PrecontactProgressMonitor",
    "PrecontactReplayTransition",
    "gate_precontact_step",
    "next_precontact_phase",
    "segment_minimum_pad_object_distance_m",
    "split_open_translation_to_handoff",
    "validate_open_only_action",
]
