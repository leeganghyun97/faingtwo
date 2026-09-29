# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure mirror/telemetry contract for the synchronized joint endpoint limiter.

The production limiter remains authoritative.  This helper is an offline
precheck and evidence generator; it never clips, resets, or replaces the
production state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .precontact_contract import PrecontactContractError


@dataclass(frozen=True)
class LimiterPrecheckResult:
    feasible: bool
    reason: str | None
    previous_endpoint_q: tuple[float, ...]
    previous_endpoint_qdot: tuple[float, ...]
    ik_target_q: tuple[float, ...]
    predicted_endpoint_qdot: tuple[float, ...]
    predicted_endpoint_qdd: tuple[float, ...]
    endpoint_crossing: tuple[bool, ...]
    velocity_limit_margin: tuple[float, ...]
    acceleration_limit_margin: tuple[float, ...]

    def payload(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "previous_endpoint_q": list(self.previous_endpoint_q),
            "previous_endpoint_qdot": list(self.previous_endpoint_qdot),
            "ik_target_q": list(self.ik_target_q),
            "predicted_endpoint_qdot": list(self.predicted_endpoint_qdot),
            "predicted_endpoint_qdd": list(self.predicted_endpoint_qdd),
            "endpoint_crossing": list(self.endpoint_crossing),
            "velocity_limit_margin": list(self.velocity_limit_margin),
            "acceleration_limit_margin": list(self.acceleration_limit_margin),
        }


class PlannerEndpointAdmission(str, Enum):
    """Admission-only outcome before a new planner Cartesian endpoint.

    ``HOLD_EXISTING_CARTESIAN_ENDPOINT`` is deliberately not a joint-command
    operation.  The selected G2 action term retains ``ee_pos_des`` on a
    neutral packet, but it still recomputes the DLS joint target at every
    physics step.  Consequently a HOLD receipt is a request to avoid a *new*
    Cartesian endpoint; it is not proof that the runtime joint target is
    immutable or that a later PhysX step will be accepted.
    """

    SUBMIT_NEW_CARTESIAN_ENDPOINT = "SUBMIT_NEW_CARTESIAN_ENDPOINT"
    HOLD_EXISTING_CARTESIAN_ENDPOINT = "HOLD_EXISTING_CARTESIAN_ENDPOINT"
    REJECT_NO_FEASIBLE_CARTESIAN_HOLD = "REJECT_NO_FEASIBLE_CARTESIAN_HOLD"


@dataclass(frozen=True)
class PlannerEndpointAdmissionReceipt:
    """Pure scheduling decision; it never emits or mutates a joint target.

    ``candidate`` and ``held`` are both DLS *predictions* supplied by an
    external read-only provider.  This module neither runs DLS nor substitutes
    the measured state for the controller accumulator.  That separation keeps
    the production limiter authoritative and prevents a scheduler from
    disguising endpoint clipping as a valid planner action.
    """

    decision: PlannerEndpointAdmission
    candidate: LimiterPrecheckResult
    held: LimiterPrecheckResult | None
    requires_existing_cartesian_pose_hold: bool
    live_joint_endpoint_immutability_proven: bool = False

    def __post_init__(self) -> None:
        if self.decision is PlannerEndpointAdmission.SUBMIT_NEW_CARTESIAN_ENDPOINT:
            if not self.candidate.feasible or self.held is not None:
                raise PrecontactContractError("invalid submit admission receipt")
            if self.requires_existing_cartesian_pose_hold:
                raise PrecontactContractError("submit receipt cannot require hold")
        elif self.decision is PlannerEndpointAdmission.HOLD_EXISTING_CARTESIAN_ENDPOINT:
            if self.candidate.feasible or self.held is None or not self.held.feasible:
                raise PrecontactContractError("invalid hold admission receipt")
            if not self.requires_existing_cartesian_pose_hold:
                raise PrecontactContractError("hold receipt must require existing pose hold")
        elif self.decision is PlannerEndpointAdmission.REJECT_NO_FEASIBLE_CARTESIAN_HOLD:
            if self.candidate.feasible or self.held is None or self.held.feasible:
                raise PrecontactContractError("invalid reject admission receipt")
            if self.requires_existing_cartesian_pose_hold:
                raise PrecontactContractError("rejected receipt cannot request hold")
        else:  # pragma: no cover - protects future enum extension.
            raise PrecontactContractError("unknown planner admission decision")
        # The source action term retains a Cartesian pose goal, not a direct
        # seven-joint endpoint.  A pure mirror must never overstate that fact.
        if self.live_joint_endpoint_immutability_proven:
            raise PrecontactContractError(
                "offline scheduler cannot prove runtime joint endpoint immutability"
            )

    @property
    def blocks_new_endpoint(self) -> bool:
        return self.decision is not PlannerEndpointAdmission.SUBMIT_NEW_CARTESIAN_ENDPOINT

    def payload(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "candidate": self.candidate.payload(),
            "held": None if self.held is None else self.held.payload(),
            "requires_existing_cartesian_pose_hold": (
                self.requires_existing_cartesian_pose_hold
            ),
            "live_joint_endpoint_immutability_proven": False,
            "controller_authority": "PRODUCTION_SYNCHRONIZED_LIMITER_UNCHANGED",
            "scheduler_authority": "ADMISSION_ONLY_NO_CLIPPING_NO_JOINT_COMMAND",
        }


def _vector(name: str, value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not bool(np.isfinite(result).all()):
        raise PrecontactContractError(f"{name} must be a finite non-empty vector")
    return result


def precheck_synchronized_endpoint(
    *,
    previous_endpoint_q: Sequence[float],
    previous_endpoint_qdot: Sequence[float],
    ik_target_q: Sequence[float],
    dt_s: float,
    maximum_speed_rad_s: float | Sequence[float],
    maximum_acceleration_rad_s2: float,
) -> LimiterPrecheckResult:
    """Mirror the production endpoint predicate using CPU arrays only."""

    previous = _vector("previous_endpoint_q", previous_endpoint_q)
    velocity = _vector("previous_endpoint_qdot", previous_endpoint_qdot)
    target = _vector("ik_target_q", ik_target_q)
    if not previous.shape == velocity.shape == target.shape:
        raise PrecontactContractError("limiter vectors must have identical shape")
    dt = float(dt_s)
    acceleration = float(maximum_acceleration_rad_s2)
    if not math.isfinite(dt) or dt <= 0.0 or not math.isfinite(acceleration) or acceleration <= 0.0:
        raise PrecontactContractError("limiter dt and acceleration must be positive")
    speed = np.asarray(maximum_speed_rad_s, dtype=np.float64)
    if speed.ndim == 0:
        speed = np.full(previous.shape, float(speed))
    if speed.shape != previous.shape or not bool(np.isfinite(speed).all()) or bool((speed <= 0.0).any()):
        raise PrecontactContractError("limiter speed limits have invalid shape/value")

    delta = target - previous
    desired_velocity = delta / dt
    speed_scale = min(1.0, float(np.min(speed / np.maximum(np.abs(desired_velocity), 1.0e-12))))
    speed_limited = desired_velocity * speed_scale
    step_velocity = acceleration * dt
    braking_speed = np.sqrt(step_velocity**2 + 2.0 * acceleration * np.abs(delta)) - step_velocity
    moving = (delta * speed_limited) > 0.0
    per_joint_scale = np.where(
        moving & (np.abs(speed_limited) > 1.0e-12),
        np.minimum(1.0, braking_speed / np.maximum(np.abs(speed_limited), 1.0e-12)),
        1.0,
    )
    endpoint_scale = float(np.min(per_joint_scale))
    scheduled = speed_limited * endpoint_scale
    velocity_delta = scheduled - velocity
    acceleration_scale = min(1.0, float(np.min((acceleration * dt) / np.maximum(np.abs(velocity_delta), 1.0e-12))))
    predicted = velocity + velocity_delta * acceleration_scale
    proposed = predicted * dt
    crossing = (delta * proposed > 0.0) & (np.abs(proposed) > np.abs(delta))
    bounded_delta = np.clip(velocity_delta, -acceleration * dt, acceleration * dt)
    decelerated = velocity + bounded_delta
    predicted = np.where(crossing, decelerated, predicted)
    final_proposed = predicted * dt
    unresolved = (delta * final_proposed > 0.0) & (np.abs(final_proposed) > np.abs(delta))
    reason = "G2_SYNCHRONIZED_LIMITER_ENDPOINT_STATE_INFEASIBLE" if bool(unresolved.any()) else None
    return LimiterPrecheckResult(
        feasible=reason is None,
        reason=reason,
        previous_endpoint_q=tuple(float(v) for v in previous),
        previous_endpoint_qdot=tuple(float(v) for v in velocity),
        ik_target_q=tuple(float(v) for v in target),
        predicted_endpoint_qdot=tuple(float(v) for v in predicted),
        predicted_endpoint_qdd=tuple(float(v) for v in ((predicted - velocity) / dt)),
        endpoint_crossing=tuple(bool(v) for v in unresolved),
        velocity_limit_margin=tuple(float(v) for v in speed - np.abs(predicted)),
        acceleration_limit_margin=tuple(float(v) for v in acceleration - np.abs((predicted - velocity) / dt)),
    )


def admit_planner_endpoint_or_existing_pose_hold(
    *,
    previous_endpoint_q: Sequence[float],
    previous_endpoint_qdot: Sequence[float],
    candidate_ik_target_q: Sequence[float],
    held_cartesian_pose_ik_target_q: Sequence[float],
    dt_s: float,
    maximum_speed_rad_s: float | Sequence[float],
    maximum_acceleration_rad_s2: float,
) -> PlannerEndpointAdmissionReceipt:
    """Reject an unsafe new endpoint without changing the production limiter.

    The caller must obtain both joint-space targets from a read-only DLS
    prediction that has the same model/frame/index ordering as the selected
    runtime action term.  The first prediction corresponds to the proposed
    *new* Cartesian endpoint; the second corresponds to submitting a neutral
    packet, whose source-defined meaning is to retain the already accepted
    Cartesian pose target.  No vector is scaled, clipped, or rewritten here.

    A feasible hold only shows that the supplied mirror target is acceptable
    in the current accumulator state.  G2 recomputes DLS at each 2 ms physics
    step, so callers must record that runtime equivalence separately before
    treating a hold as a live-motion authorization.
    """

    shared = {
        "previous_endpoint_q": previous_endpoint_q,
        "previous_endpoint_qdot": previous_endpoint_qdot,
        "dt_s": dt_s,
        "maximum_speed_rad_s": maximum_speed_rad_s,
        "maximum_acceleration_rad_s2": maximum_acceleration_rad_s2,
    }
    candidate = precheck_synchronized_endpoint(
        **shared,
        ik_target_q=candidate_ik_target_q,
    )
    if candidate.feasible:
        return PlannerEndpointAdmissionReceipt(
            decision=PlannerEndpointAdmission.SUBMIT_NEW_CARTESIAN_ENDPOINT,
            candidate=candidate,
            held=None,
            requires_existing_cartesian_pose_hold=False,
        )
    held = precheck_synchronized_endpoint(
        **shared,
        ik_target_q=held_cartesian_pose_ik_target_q,
    )
    if held.feasible:
        return PlannerEndpointAdmissionReceipt(
            decision=PlannerEndpointAdmission.HOLD_EXISTING_CARTESIAN_ENDPOINT,
            candidate=candidate,
            held=held,
            requires_existing_cartesian_pose_hold=True,
        )
    return PlannerEndpointAdmissionReceipt(
        decision=PlannerEndpointAdmission.REJECT_NO_FEASIBLE_CARTESIAN_HOLD,
        candidate=candidate,
        held=held,
        requires_existing_cartesian_pose_hold=False,
    )


__all__ = [
    "LimiterPrecheckResult",
    "PlannerEndpointAdmission",
    "PlannerEndpointAdmissionReceipt",
    "admit_planner_endpoint_or_existing_pose_hold",
    "precheck_synchronized_endpoint",
]
