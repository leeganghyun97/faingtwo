# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure cuRobo waypoint to canonical G2 four-dimensional action adapter.

cuRobo is a high-level nominal planner in this branch.  It never owns a
joint command, an ActionManager, a controller, or a simulation step.  For one
planner waypoint this module computes the robot-root Cartesian residual

``p_plan - p_measured``

and represents it with the already-qualified policy action contract
``[dX, dY, dZ, OPEN]``.  The existing production route subsequently decodes
the normalized values to metres, obtains its safety receipt, converts the
accepted metric residual back to the selected production normalized port,
and delegates the one complete 8-D packet to ``env.step``.

There is deliberately no clipping here.  A planner waypoint farther than the
existing normalized action range is rejected and must be re-timed or
interpolated by the planner, never silently changed by this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

from .action_interface import (
    AbstractGripperIntent,
    CartesianControlFrame,
    HighLevelPolicyAction,
    PolicyActionMode,
)
from .production_metric_adapter import ProductionArmActionScale


CUROBO_CANONICAL_4D_HANDOFF_SCHEMA = "g2_curobo_canonical_4d_handoff_v1"
CUROBO_PLANNER_GRIPPER_PROBABILITY_OPEN = 0.0


class CuroboCanonical4DHandoffError(ValueError):
    """Raised when a planner waypoint cannot enter the frozen 4-D contract."""


def _finite_xyz(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise CuroboCanonical4DHandoffError(f"{name} must contain exactly xyz")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise CuroboCanonical4DHandoffError(f"{name} must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class CuroboCanonical4DHandoffReceipt:
    """Immutable evidence for one planner-to-policy representation change."""

    planner_target_root_m: tuple[float, float, float]
    measured_ee_root_m: tuple[float, float, float]
    metric_delta_root_m: tuple[float, float, float]
    normalized_action: tuple[float, float, float, float]
    reconstructed_metric_delta_root_m: tuple[float, float, float]
    frame: CartesianControlFrame
    gripper_intent: AbstractGripperIntent
    clipping_applied: bool
    forward_metric_scale_count_after_policy_ingress: int
    schema: str = CUROBO_CANONICAL_4D_HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CUROBO_CANONICAL_4D_HANDOFF_SCHEMA:
            raise CuroboCanonical4DHandoffError("unsupported handoff schema")
        if self.frame is not CartesianControlFrame.ROBOT_ROOT:
            raise CuroboCanonical4DHandoffError("planner handoff must use robot_root")
        if self.gripper_intent is not AbstractGripperIntent.OPEN:
            raise CuroboCanonical4DHandoffError("planner-only handoff must keep OPEN")
        if self.clipping_applied:
            raise CuroboCanonical4DHandoffError("planner handoff may not clip")
        if self.forward_metric_scale_count_after_policy_ingress != 1:
            raise CuroboCanonical4DHandoffError(
                "selected production forward metric scale count must be one"
            )

    @property
    def policy_action(self) -> HighLevelPolicyAction:
        return HighLevelPolicyAction(
            self.normalized_action,
            mode=PolicyActionMode.TRANSLATION_GRIPPER_4D,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "planner_target_root_m": list(self.planner_target_root_m),
            "measured_ee_root_m": list(self.measured_ee_root_m),
            "metric_delta_root_m": list(self.metric_delta_root_m),
            "normalized_action": list(self.normalized_action),
            "reconstructed_metric_delta_root_m": list(
                self.reconstructed_metric_delta_root_m
            ),
            "frame": self.frame.value,
            "gripper_intent": self.gripper_intent.value,
            "clipping_applied": self.clipping_applied,
            "forward_metric_scale_count_after_policy_ingress": (
                self.forward_metric_scale_count_after_policy_ingress
            ),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def planner_waypoint_to_canonical_4d(
    *,
    planner_target_root_m: Sequence[float],
    measured_ee_root_m: Sequence[float],
    scale: ProductionArmActionScale = ProductionArmActionScale(),
) -> CuroboCanonical4DHandoffReceipt:
    """Represent one measured-feedback planner residual without consuming it.

    This function has no safety or controller authority.  Its returned
    :class:`HighLevelPolicyAction` must still pass through the existing
    production-bound router and canonical ``env.step`` consumer.
    """

    if not isinstance(scale, ProductionArmActionScale):
        raise CuroboCanonical4DHandoffError("scale must be ProductionArmActionScale")
    if scale.frame is not CartesianControlFrame.ROBOT_ROOT:
        raise CuroboCanonical4DHandoffError("scale frame must be robot_root")
    target = _finite_xyz("planner_target_root_m", planner_target_root_m)
    measured = _finite_xyz("measured_ee_root_m", measured_ee_root_m)
    delta = tuple(target[index] - measured[index] for index in range(3))
    normalized_xyz = tuple(
        value / scale.translation_m_per_normalized for value in delta
    )
    if any(abs(value) > scale.normalized_limit for value in normalized_xyz):
        raise CuroboCanonical4DHandoffError(
            "PLANNER_DELTA_OUTSIDE_CANONICAL_NORMALIZED_RANGE_REJECTED_WITHOUT_CLIPPING"
        )
    reconstructed = tuple(
        value * scale.translation_m_per_normalized for value in normalized_xyz
    )
    normalized_action = (
        normalized_xyz[0],
        normalized_xyz[1],
        normalized_xyz[2],
        CUROBO_PLANNER_GRIPPER_PROBABILITY_OPEN,
    )
    # Construct once here so malformed range/action semantics fail before a
    # receipt can be returned.  The property reconstructs the same immutable
    # typed action for the canonical router.
    HighLevelPolicyAction(
        normalized_action,
        mode=PolicyActionMode.TRANSLATION_GRIPPER_4D,
    )
    return CuroboCanonical4DHandoffReceipt(
        planner_target_root_m=target,
        measured_ee_root_m=measured,
        metric_delta_root_m=delta,
        normalized_action=normalized_action,
        reconstructed_metric_delta_root_m=reconstructed,
        frame=CartesianControlFrame.ROBOT_ROOT,
        gripper_intent=AbstractGripperIntent.OPEN,
        clipping_applied=False,
        forward_metric_scale_count_after_policy_ingress=1,
    )


__all__ = [
    "CUROBO_CANONICAL_4D_HANDOFF_SCHEMA",
    "CUROBO_PLANNER_GRIPPER_PROBABILITY_OPEN",
    "CuroboCanonical4DHandoffError",
    "CuroboCanonical4DHandoffReceipt",
    "planner_waypoint_to_canonical_4d",
]
