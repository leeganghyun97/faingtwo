# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded canonical-action recovery for OPEN gripper/table clearance.

This module is deliberately Isaac-free.  Exact collision-mesh clearance is
measured by the runtime oracle and supplied here.  The recovery never writes a
joint, torque, controller target, reward, or CLOSE predicate: it only chooses a
small positive-root-Z canonical Cartesian command while an episode is still
behind the measured reset barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


RESET_OPEN_TABLE_CLEARANCE_SCHEMA = (
    "g2_stage1a_reset_open_table_clearance_recovery_v1"
)
OPEN_TABLE_MINIMUM_CLEARANCE_M = 0.002
OPEN_TABLE_LIFT_STEP_M = 0.0005
OPEN_TABLE_MAXIMUM_LIFT_M = 0.020


class OpenTableClearanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenTableClearanceDecision:
    command_z_m: float
    current_minimum_clearance_m: float
    outer_link4_table_clearance_m: float
    inner_link4_table_clearance_m: float
    derived_safe_ee_z_m: float
    measured_lift_from_source_m: float
    clearance_ready: bool
    recovery_active: bool
    recovery_exhausted: bool

    def receipt(self) -> dict[str, object]:
        return {
            "schema": RESET_OPEN_TABLE_CLEARANCE_SCHEMA,
            "minimum_required_clearance_m": OPEN_TABLE_MINIMUM_CLEARANCE_M,
            "lift_step_m": OPEN_TABLE_LIFT_STEP_M,
            "maximum_lift_m": OPEN_TABLE_MAXIMUM_LIFT_M,
            "command_z_m": self.command_z_m,
            "current_minimum_clearance_m": self.current_minimum_clearance_m,
            "outer_link4_table_clearance_m": self.outer_link4_table_clearance_m,
            "inner_link4_table_clearance_m": self.inner_link4_table_clearance_m,
            "derived_safe_ee_z_m": self.derived_safe_ee_z_m,
            "measured_lift_from_source_m": self.measured_lift_from_source_m,
            "clearance_ready": self.clearance_ready,
            "recovery_active": self.recovery_active,
            "recovery_exhausted": self.recovery_exhausted,
            "authority": "LIVE_COLLISION_MESH_TO_TABLE_CLEARANCE",
            "canonical_action_only": True,
            "direct_joint_or_torque_write": False,
            "forbidden_collision_relaxed": False,
        }


def reset_open_clearance_decision(
    *,
    source_ee_z_m: float,
    current_ee_z_m: float,
    outer_link4_table_clearance_m: float,
    inner_link4_table_clearance_m: float,
) -> OpenTableClearanceDecision:
    """Choose a bounded upward reset command from exact current geometry."""

    values = (
        source_ee_z_m,
        current_ee_z_m,
        outer_link4_table_clearance_m,
        inner_link4_table_clearance_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise OpenTableClearanceError("OPEN_TABLE_CLEARANCE_NONFINITE")
    minimum = min(
        float(outer_link4_table_clearance_m),
        float(inner_link4_table_clearance_m),
    )
    measured_lift = max(0.0, float(current_ee_z_m) - float(source_ee_z_m))
    deficit = max(0.0, OPEN_TABLE_MINIMUM_CLEARANCE_M - minimum)
    derived_safe_ee_z = float(current_ee_z_m) + deficit
    ready = deficit <= 1.0e-9
    exhausted = bool(
        not ready
        and (
            measured_lift >= OPEN_TABLE_MAXIMUM_LIFT_M - 1.0e-9
            or derived_safe_ee_z
            > float(source_ee_z_m) + OPEN_TABLE_MAXIMUM_LIFT_M + 1.0e-9
        )
    )
    command = (
        0.0
        if ready or exhausted
        else min(
            OPEN_TABLE_LIFT_STEP_M,
            deficit,
            OPEN_TABLE_MAXIMUM_LIFT_M - measured_lift,
        )
    )
    return OpenTableClearanceDecision(
        command_z_m=float(command),
        current_minimum_clearance_m=minimum,
        outer_link4_table_clearance_m=float(outer_link4_table_clearance_m),
        inner_link4_table_clearance_m=float(inner_link4_table_clearance_m),
        derived_safe_ee_z_m=derived_safe_ee_z,
        measured_lift_from_source_m=measured_lift,
        clearance_ready=ready,
        recovery_active=bool(not ready and not exhausted),
        recovery_exhausted=exhausted,
    )


def project_open_precontact_action_for_table_clearance(
    action_4d_metric_root_m: Sequence[float],
    *,
    current_minimum_clearance_m: float,
) -> tuple[tuple[float, float, float, float], dict[str, object]]:
    """Prevent an OPEN pre-contact command from crossing the 2 mm plane.

    When current geometry is already safe this function can only reduce a
    downward Z component; X/Y and the gripper scalar are preserved.  If an
    unexpected physics drift has already crossed the margin, X/Y are held and
    the same bounded upward recovery command is emitted.
    """

    action = tuple(float(value) for value in action_4d_metric_root_m)
    if len(action) != 4 or not all(math.isfinite(value) for value in action):
        raise OpenTableClearanceError("OPEN_TABLE_ACTION_INVALID")
    clearance = float(current_minimum_clearance_m)
    if not math.isfinite(clearance):
        raise OpenTableClearanceError("OPEN_TABLE_CLEARANCE_NONFINITE")
    requested_z = action[2]
    if clearance < OPEN_TABLE_MINIMUM_CLEARANCE_M:
        applied = (
            0.0,
            0.0,
            min(
                OPEN_TABLE_LIFT_STEP_M,
                OPEN_TABLE_MINIMUM_CLEARANCE_M - clearance,
            ),
            action[3],
        )
        mode = "RECOVER_CURRENT_CLEARANCE"
    else:
        minimum_allowed_z = OPEN_TABLE_MINIMUM_CLEARANCE_M - clearance
        applied = (
            action[0],
            action[1],
            max(requested_z, minimum_allowed_z),
            action[3],
        )
        mode = "CLAMP_DOWNWARD_Z" if applied[2] != requested_z else "PASS_THROUGH"
    requested_clearance = clearance + requested_z
    applied_clearance = clearance + applied[2]
    return applied, {
        "schema": "g2_stage1a_open_precontact_table_clearance_projection_v1",
        "current_minimum_clearance_m": clearance,
        "minimum_required_clearance_m": OPEN_TABLE_MINIMUM_CLEARANCE_M,
        "requested_z_m": requested_z,
        "applied_z_m": applied[2],
        "requested_minimum_clearance_m": requested_clearance,
        "applied_minimum_clearance_m": applied_clearance,
        "intervention": applied != action,
        "mode": mode,
        "x_y_preserved": applied[:2] == action[:2],
        "gripper_scalar_preserved": applied[3] == action[3],
        "forbidden_collision_relaxed": False,
    }


__all__ = [
    "OPEN_TABLE_LIFT_STEP_M",
    "OPEN_TABLE_MAXIMUM_LIFT_M",
    "OPEN_TABLE_MINIMUM_CLEARANCE_M",
    "OpenTableClearanceDecision",
    "OpenTableClearanceError",
    "RESET_OPEN_TABLE_CLEARANCE_SCHEMA",
    "project_open_precontact_action_for_table_clearance",
    "reset_open_clearance_decision",
]
