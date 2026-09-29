# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""State-preserving planner-to-BC handoff contracts.

The nominal planner owns XYZ at the cuRobo/BC boundary.  The BC policy may
decide only the scalar gripper probability at this boundary.  Sending the
BC translation head directly would replace the planner endpoint every policy
step and can ask the synchronized joint-target limiter to stop beyond a newly
shortened endpoint.  This module instead re-encodes one fixed robot-root
planner endpoint from measured feedback on every step::

    delta_p[t] = planner_endpoint - measured_ee[t]

For Isaac Lab's relative differential-IK port this gives
``measured_ee[t] + delta_p[t] == planner_endpoint``.  No controller cache is
reset and the rate-limiter position/velocity state is therefore preserved.
The module has no controller, safety, simulation, or action-consumption
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from ..g2_teleop_dataset import synchronized_rate_limit_position_target
from .action_interface import HighLevelPolicyAction, PolicyActionMode
from .curobo_4d_handoff import (
    CuroboCanonical4DHandoffError,
    planner_waypoint_to_canonical_4d,
)
from .production_metric_adapter import ProductionArmActionScale


PLANNER_BC_STATE_BRIDGE_SCHEMA = "g2_planner_bc_state_preserving_bridge_v1"


class PlannerBCStateBridgeError(ValueError):
    """Raised when a bridge request cannot preserve the frozen 4-D contract."""


def _finite_probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise PlannerBCStateBridgeError("BC gripper probability must be finite in [0,1]")
    return value


@dataclass(frozen=True)
class PlannerBCStateBridgeReceipt:
    """One representation-only fixed-endpoint bridge decision."""

    planner_endpoint_root_m: tuple[float, float, float]
    measured_ee_root_m: tuple[float, float, float]
    action: HighLevelPolicyAction
    bc_translation_advisory_only: tuple[float, float, float] | None
    bc_gripper_probability: float
    force_open: bool
    translation_m_per_normalized: float
    schema: str = PLANNER_BC_STATE_BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANNER_BC_STATE_BRIDGE_SCHEMA:
            raise PlannerBCStateBridgeError("unsupported bridge schema")
        if self.action.mode is not PolicyActionMode.TRANSLATION_GRIPPER_4D:
            raise PlannerBCStateBridgeError("bridge must preserve exact 4-D action mode")
        if self.force_open and self.action.gripper_probability != 0.0:
            raise PlannerBCStateBridgeError("forced-open bridge must emit g=0 exactly")
        if (
            not math.isfinite(self.translation_m_per_normalized)
            or self.translation_m_per_normalized <= 0.0
        ):
            raise PlannerBCStateBridgeError("bridge translation scale must be positive")

    @property
    def decoded_endpoint_root_m(self) -> tuple[float, float, float]:
        """Endpoint reconstructed after the selected production XYZ scale."""

        return tuple(
            self.measured_ee_root_m[index]
            + self.action.translation_normalized[index]
            * self.translation_m_per_normalized
            for index in range(3)
        )

    @property
    def endpoint_error_m(self) -> float:
        return math.dist(self.decoded_endpoint_root_m, self.planner_endpoint_root_m)


def planner_owned_xyz_bc_gripper_action(
    *,
    planner_endpoint_root_m: Sequence[float],
    measured_ee_root_m: Sequence[float],
    bc_gripper_probability: float,
    bc_translation_advisory_only: Sequence[float] | None = None,
    force_open: bool = False,
    scale: ProductionArmActionScale = ProductionArmActionScale(),
) -> PlannerBCStateBridgeReceipt:
    """Build exact-4D action while retaining planner ownership of XYZ.

    ``bc_translation_advisory_only`` is accepted solely for an audit receipt;
    it never enters the emitted action.  During a bounded OPEN bridge,
    ``force_open=True`` emits exactly zero gripper probability.  Afterwards
    the same XYZ construction can carry the BC/GRU probability without giving
    BC authority over arm translation.
    """

    probability = _finite_probability(bc_gripper_probability)
    try:
        base = planner_waypoint_to_canonical_4d(
            planner_target_root_m=planner_endpoint_root_m,
            measured_ee_root_m=measured_ee_root_m,
            scale=scale,
        )
    except CuroboCanonical4DHandoffError as error:
        raise PlannerBCStateBridgeError(str(error)) from error
    advisory: tuple[float, float, float] | None = None
    if bc_translation_advisory_only is not None:
        if len(bc_translation_advisory_only) != 3:
            raise PlannerBCStateBridgeError("BC translation advisory must be xyz")
        advisory = tuple(float(value) for value in bc_translation_advisory_only)
        if not all(math.isfinite(value) and abs(value) <= 1.0 for value in advisory):
            raise PlannerBCStateBridgeError(
                "BC translation advisory must be finite and normalized"
            )
    gripper = 0.0 if force_open else probability
    action = HighLevelPolicyAction(
        (*base.normalized_action[:3], gripper),
        mode=PolicyActionMode.TRANSLATION_GRIPPER_4D,
    )
    receipt = PlannerBCStateBridgeReceipt(
        planner_endpoint_root_m=base.planner_target_root_m,
        measured_ee_root_m=base.measured_ee_root_m,
        action=action,
        bc_translation_advisory_only=advisory,
        bc_gripper_probability=probability,
        force_open=bool(force_open),
        translation_m_per_normalized=scale.translation_m_per_normalized,
    )
    if receipt.endpoint_error_m > 1.0e-12:
        raise PlannerBCStateBridgeError("bridge scale round-trip changed planner endpoint")
    return receipt


def minimum_velocity_drain_physics_steps(
    velocity_rad_s: Sequence[float],
    *,
    maximum_acceleration_rad_s2: float,
    physics_dt_s: float,
) -> int:
    """Lower bound to drain the captured limiter velocity to zero.

    Each limiter substep changes any velocity component by at most ``a*dt``.
    The synchronized endpoint schedule may require more steps, so promotion
    must use an actual fixed-endpoint replay rather than this bound alone.
    """

    if maximum_acceleration_rad_s2 <= 0.0 or physics_dt_s <= 0.0:
        raise PlannerBCStateBridgeError("acceleration and physics dt must be positive")
    values = tuple(float(value) for value in velocity_rad_s)
    if not values or not all(math.isfinite(value) for value in values):
        raise PlannerBCStateBridgeError("captured velocity must be finite and nonempty")
    return int(
        math.ceil(
            max(abs(value) for value in values)
            / (maximum_acceleration_rad_s2 * physics_dt_s)
        )
    )


def minimum_rest_to_rest_policy_steps_lower_bound(
    joint_delta_rad: Sequence[float],
    *,
    maximum_speed_rad_s: float,
    maximum_acceleration_rad_s2: float,
    physics_dt_s: float,
    physics_substeps_per_policy_step: int,
) -> int:
    """Bang-bang lower bound for a synchronized rest-to-rest joint move.

    For distance ``d`` the triangular duration is ``2*sqrt(d/a)`` when
    ``d <= v_max**2/a``; otherwise the trapezoidal duration is
    ``d/v_max + v_max/a``.  DLS changes and the common-row endpoint scale can
    only make the realized bridge longer.
    """

    if maximum_speed_rad_s <= 0.0 or maximum_acceleration_rad_s2 <= 0.0:
        raise PlannerBCStateBridgeError("speed and acceleration must be positive")
    if physics_dt_s <= 0.0 or physics_substeps_per_policy_step <= 0:
        raise PlannerBCStateBridgeError("time step and decimation must be positive")
    deltas = tuple(abs(float(value)) for value in joint_delta_rad)
    if not deltas or not all(math.isfinite(value) for value in deltas):
        raise PlannerBCStateBridgeError("joint delta must be finite and nonempty")
    transition = maximum_speed_rad_s**2 / maximum_acceleration_rad_s2
    durations = []
    for distance in deltas:
        if distance <= transition:
            durations.append(2.0 * math.sqrt(distance / maximum_acceleration_rad_s2))
        else:
            durations.append(
                distance / maximum_speed_rad_s
                + maximum_speed_rad_s / maximum_acceleration_rad_s2
            )
    control_dt = physics_dt_s * physics_substeps_per_policy_step
    return int(math.ceil(max(durations, default=0.0) / control_dt))


@dataclass(frozen=True)
class FixedEndpointLimiterReplay:
    settled: bool
    physics_steps: int
    policy_steps: int
    final_target_error_rad: float
    final_velocity_rad_s: float
    maximum_emitted_speed_rad_s: float
    maximum_emitted_acceleration_rad_s2: float


def replay_fixed_joint_endpoint(
    *,
    previous_target_rad: Sequence[float],
    desired_target_rad: Sequence[float],
    previous_velocity_rad_s: Sequence[float],
    maximum_speed_rad_s: float,
    maximum_acceleration_rad_s2: float,
    physics_dt_s: float,
    physics_substeps_per_policy_step: int,
    target_tolerance_rad: float = 1.0e-6,
    velocity_tolerance_rad_s: float = 1.0e-5,
    maximum_physics_steps: int = 10_000,
) -> FixedEndpointLimiterReplay:
    """CPU replay of the existing limiter against one immutable endpoint."""

    if target_tolerance_rad < 0.0 or velocity_tolerance_rad_s < 0.0:
        raise PlannerBCStateBridgeError("replay tolerances must be nonnegative")
    if physics_substeps_per_policy_step <= 0 or maximum_physics_steps <= 0:
        raise PlannerBCStateBridgeError("replay budgets must be positive")
    q = torch.tensor([tuple(previous_target_rad)], dtype=torch.float64)
    goal = torch.tensor([tuple(desired_target_rad)], dtype=torch.float64)
    velocity = torch.tensor([tuple(previous_velocity_rad_s)], dtype=torch.float64)
    if q.shape != goal.shape or q.shape != velocity.shape or q.shape[1] <= 0:
        raise PlannerBCStateBridgeError("replay joint state shapes must match")
    max_speed = 0.0
    max_acceleration = 0.0
    settled = False
    steps = 0
    for steps in range(1, maximum_physics_steps + 1):
        previous_q = q
        previous_velocity = velocity
        q, velocity = synchronized_rate_limit_position_target(
            previous_q,
            goal,
            previous_velocity,
            maximum_speed_rad_s=maximum_speed_rad_s,
            maximum_acceleration_rad_s2=maximum_acceleration_rad_s2,
            physics_dt_s=physics_dt_s,
        )
        max_speed = max(
            max_speed,
            float(torch.max(torch.abs(q - previous_q)).item() / physics_dt_s),
        )
        max_acceleration = max(
            max_acceleration,
            float(
                torch.max(torch.abs(velocity - previous_velocity)).item()
                / physics_dt_s
            ),
        )
        error = float(torch.max(torch.abs(goal - q)).item())
        speed = float(torch.max(torch.abs(velocity)).item())
        if error <= target_tolerance_rad and speed <= velocity_tolerance_rad_s:
            settled = True
            break
    return FixedEndpointLimiterReplay(
        settled=settled,
        physics_steps=steps,
        policy_steps=int(math.ceil(steps / physics_substeps_per_policy_step)),
        final_target_error_rad=float(torch.max(torch.abs(goal - q)).item()),
        final_velocity_rad_s=float(torch.max(torch.abs(velocity)).item()),
        maximum_emitted_speed_rad_s=max_speed,
        maximum_emitted_acceleration_rad_s2=max_acceleration,
    )


__all__ = [
    "FixedEndpointLimiterReplay",
    "PLANNER_BC_STATE_BRIDGE_SCHEMA",
    "PlannerBCStateBridgeError",
    "PlannerBCStateBridgeReceipt",
    "minimum_rest_to_rest_policy_steps_lower_bound",
    "minimum_velocity_drain_physics_steps",
    "planner_owned_xyz_bc_gripper_action",
    "replay_fixed_joint_endpoint",
]
