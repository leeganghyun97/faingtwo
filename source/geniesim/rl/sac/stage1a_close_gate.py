# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure, offline-testable Stage-1A CLOSE admission contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CONTROL_HZ = 50
SIMPLIFIED_CLOSE_MIN_RESIDUAL_M = 0.015
SIMPLIFIED_CLOSE_MAX_RESIDUAL_M = 0.020
SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG = 15.0
SIMPLIFIED_CLOSE_PERSISTENCE_STEPS = 5
# V3.1 is deliberately a diagnostic candidate rather than a change to the
# historical V3 close authority.  It is derived from the immutable
# close-neighborhood artifact: all eight stable rows were <= 9.411 mm lateral
# error while the six unsafe rows began at 10.519 mm.  Ten millimetres leaves
# measured success margin without claiming a production-calibrated threshold.
V31_CLOSE_MAX_LATERAL_ERROR_M = 0.010


class SimplifiedCloseGateError(ValueError):
    pass


@dataclass
class PrivilegedGeometryClosePersistenceGate:
    """Diagnostic-only CLOSE gate backed by measured pad/cube geometry.

    This gate intentionally has no nominal-grasp-residual input.  It is used
    only by a bounded paired diagnostic to answer whether the deployed
    residual/orientation proxy is the reason that a canonical CLOSE does not
    yield pad contact.  It is *not* a production admission authority.

    The geometry fields originate from the live collision-enabled primary-pad
    meshes.  A positive result means the cube lies between the two pads and
    the current opening can geometrically contain its projected width.  The
    five-control-step persistence remains in place so this diagnostic does
    not turn an instantaneous mesh read into an unbounded CLOSE trigger.
    """

    geometry_ready_count: int = 0
    orientation_ready_count: int = 0
    persistence_ready_count: int = 0
    close_trigger_count: int = 0
    close_latched: bool = False

    def observe(
        self,
        *,
        cube_between_primary_pads: bool,
        aperture_geometrically_compatible: bool,
        inner_pad_cube_gap_mm: float,
        outer_pad_cube_gap_mm: float,
        gripper_aperture_mm: float,
        cube_effective_width_mm: float,
        orientation_error_deg: float,
        no_safety_violation: bool,
    ) -> dict[str, Any]:
        values = (
            float(inner_pad_cube_gap_mm),
            float(outer_pad_cube_gap_mm),
            float(gripper_aperture_mm),
            float(cube_effective_width_mm),
            float(orientation_error_deg),
        )
        if (
            not all(math.isfinite(value) for value in values)
            # A negative measured opening is a physically infeasible current
            # configuration (overlapping pad projections), not malformed
            # telemetry.  The oracle marks it geometry-incompatible below.
            or min(values[0], values[1], values[3]) < 0.0
            or not isinstance(cube_between_primary_pads, bool)
            or not isinstance(aperture_geometrically_compatible, bool)
            or not isinstance(no_safety_violation, bool)
        ):
            raise SimplifiedCloseGateError("PRIVILEGED_GEOMETRY_CLOSE_GATE_INPUT_INVALID")
        geometry_ready = bool(
            cube_between_primary_pads
            and aperture_geometrically_compatible
            and values[2] >= values[3]
        )
        orientation_ready = bool(
            orientation_error_deg <= SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG
        )
        self.geometry_ready_count = (
            self.geometry_ready_count + 1 if geometry_ready else 0
        )
        self.orientation_ready_count = (
            self.orientation_ready_count + 1 if orientation_ready else 0
        )
        combined_ready = bool(
            geometry_ready and orientation_ready and no_safety_violation
        )
        self.persistence_ready_count = (
            self.persistence_ready_count + 1 if combined_ready else 0
        )
        trigger = bool(
            not self.close_latched
            and self.persistence_ready_count >= SIMPLIFIED_CLOSE_PERSISTENCE_STEPS
        )
        if trigger:
            self.close_latched = True
            self.close_trigger_count += 1
        if self.close_trigger_count > 1:
            raise SimplifiedCloseGateError(
                "PRIVILEGED_GEOMETRY_CLOSE_DUPLICATE_TRIGGER"
            )
        return {
            "cube_between_primary_pads": cube_between_primary_pads,
            "aperture_geometrically_compatible": aperture_geometrically_compatible,
            "inner_pad_cube_gap_mm": values[0],
            "outer_pad_cube_gap_mm": values[1],
            "minimum_primary_pad_cube_gap_mm": min(values[0], values[1]),
            "gripper_aperture_mm": values[2],
            "cube_effective_width_mm": values[3],
            "geometry_ready": geometry_ready,
            "orientation_error_deg": values[4],
            "orientation_ready": orientation_ready,
            "no_safety_violation": no_safety_violation,
            "geometry_ready_count": self.geometry_ready_count,
            "orientation_ready_count": self.orientation_ready_count,
            "persistence_ready_count": self.persistence_ready_count,
            "persistence_required_steps": SIMPLIFIED_CLOSE_PERSISTENCE_STEPS,
            "persistence_duration_ms": (
                1000.0 * self.persistence_ready_count / CONTROL_HZ
            ),
            "close_trigger": trigger,
            "close_trigger_count": self.close_trigger_count,
            "close_latched": self.close_latched,
            "authority": (
                "DIAGNOSTIC_LIVE_PRIMARY_PAD_MESH_APERTURE_ORIENTATION_"
                "5STEP_SAFETY_GATE"
            ),
            "student_privileged_input_count": 0,
        }


@dataclass
class SimplifiedClosePersistenceGate:
    """Episode-local, exactly-once CLOSE gate with consecutive counters."""

    distance_ready_count: int = 0
    orientation_ready_count: int = 0
    persistence_ready_count: int = 0
    close_trigger_count: int = 0
    close_latched: bool = False

    def observe(
        self,
        *,
        nominal_grasp_residual_m: float,
        orientation_error_deg: float,
        no_safety_violation: bool,
    ) -> dict[str, Any]:
        residual_m = float(nominal_grasp_residual_m)
        orientation_deg = float(orientation_error_deg)
        if (
            not math.isfinite(residual_m)
            or not math.isfinite(orientation_deg)
            or not isinstance(no_safety_violation, bool)
        ):
            raise SimplifiedCloseGateError("SIMPLIFIED_CLOSE_GATE_INPUT_INVALID")
        distance_ready = bool(
            SIMPLIFIED_CLOSE_MIN_RESIDUAL_M
            <= residual_m
            <= SIMPLIFIED_CLOSE_MAX_RESIDUAL_M
        )
        orientation_ready = bool(
            orientation_deg <= SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG
        )
        self.distance_ready_count = (
            self.distance_ready_count + 1 if distance_ready else 0
        )
        self.orientation_ready_count = (
            self.orientation_ready_count + 1 if orientation_ready else 0
        )
        combined_ready = bool(
            distance_ready and orientation_ready and no_safety_violation
        )
        self.persistence_ready_count = (
            self.persistence_ready_count + 1 if combined_ready else 0
        )
        trigger = bool(
            not self.close_latched
            and self.persistence_ready_count >= SIMPLIFIED_CLOSE_PERSISTENCE_STEPS
        )
        if trigger:
            self.close_latched = True
            self.close_trigger_count += 1
        if self.close_trigger_count > 1:
            raise SimplifiedCloseGateError("SIMPLIFIED_CLOSE_DUPLICATE_TRIGGER")
        return {
            "nominal_grasp_residual_mm": residual_m * 1000.0,
            "orientation_error_deg": orientation_deg,
            "distance_ready": distance_ready,
            "orientation_ready": orientation_ready,
            "no_safety_violation": no_safety_violation,
            "distance_ready_count": self.distance_ready_count,
            "orientation_ready_count": self.orientation_ready_count,
            "persistence_ready_count": self.persistence_ready_count,
            "persistence_required_steps": SIMPLIFIED_CLOSE_PERSISTENCE_STEPS,
            "persistence_duration_ms": (
                1000.0 * self.persistence_ready_count / CONTROL_HZ
            ),
            "close_trigger": trigger,
            "close_trigger_count": self.close_trigger_count,
            "close_latched": self.close_latched,
            "authority": "SIMPLIFIED_15_20MM_15DEG_5STEP_SAFETY_GATE",
        }


@dataclass
class V31ClosePersistenceGate:
    """V3.1 diagnostic CLOSE gate with an explicit lateral-ready predicate.

    This is a separate type so historical Reward-V3 and its prior runs retain
    their exact distance/orientation-only implementation and provenance.
    """

    distance_ready_count: int = 0
    orientation_ready_count: int = 0
    lateral_ready_count: int = 0
    persistence_ready_count: int = 0
    close_trigger_count: int = 0
    close_latched: bool = False

    def observe(
        self,
        *,
        nominal_grasp_residual_m: float,
        orientation_error_deg: float,
        lateral_alignment_error_m: float,
        no_safety_violation: bool,
    ) -> dict[str, Any]:
        residual_m = float(nominal_grasp_residual_m)
        orientation_deg = float(orientation_error_deg)
        lateral_m = float(lateral_alignment_error_m)
        if (
            not math.isfinite(residual_m)
            or not math.isfinite(orientation_deg)
            or not math.isfinite(lateral_m)
            or lateral_m < 0.0
            or not isinstance(no_safety_violation, bool)
        ):
            raise SimplifiedCloseGateError("V31_CLOSE_GATE_INPUT_INVALID")
        distance_ready = bool(
            SIMPLIFIED_CLOSE_MIN_RESIDUAL_M
            <= residual_m
            <= SIMPLIFIED_CLOSE_MAX_RESIDUAL_M
        )
        orientation_ready = bool(
            orientation_deg <= SIMPLIFIED_CLOSE_MAX_ORIENTATION_ERROR_DEG
        )
        lateral_ready = bool(lateral_m <= V31_CLOSE_MAX_LATERAL_ERROR_M)
        self.distance_ready_count = (
            self.distance_ready_count + 1 if distance_ready else 0
        )
        self.orientation_ready_count = (
            self.orientation_ready_count + 1 if orientation_ready else 0
        )
        self.lateral_ready_count = (
            self.lateral_ready_count + 1 if lateral_ready else 0
        )
        combined_ready = bool(
            distance_ready
            and orientation_ready
            and lateral_ready
            and no_safety_violation
        )
        self.persistence_ready_count = (
            self.persistence_ready_count + 1 if combined_ready else 0
        )
        trigger = bool(
            not self.close_latched
            and self.persistence_ready_count >= SIMPLIFIED_CLOSE_PERSISTENCE_STEPS
        )
        if trigger:
            self.close_latched = True
            self.close_trigger_count += 1
        if self.close_trigger_count > 1:
            raise SimplifiedCloseGateError("V31_CLOSE_DUPLICATE_TRIGGER")
        return {
            "nominal_grasp_residual_mm": residual_m * 1000.0,
            "orientation_error_deg": orientation_deg,
            "lateral_alignment_error_mm": lateral_m * 1000.0,
            "distance_ready": distance_ready,
            "orientation_ready": orientation_ready,
            "lateral_ready": lateral_ready,
            "no_safety_violation": no_safety_violation,
            "distance_ready_count": self.distance_ready_count,
            "orientation_ready_count": self.orientation_ready_count,
            "lateral_ready_count": self.lateral_ready_count,
            "persistence_ready_count": self.persistence_ready_count,
            "persistence_required_steps": SIMPLIFIED_CLOSE_PERSISTENCE_STEPS,
            "persistence_duration_ms": (
                1000.0 * self.persistence_ready_count / CONTROL_HZ
            ),
            "close_trigger": trigger,
            "close_trigger_count": self.close_trigger_count,
            "close_latched": self.close_latched,
            "lateral_threshold_mm": V31_CLOSE_MAX_LATERAL_ERROR_M * 1000.0,
            "lateral_threshold_authority": (
                "CLOSE_NEIGHBORHOOD_STABLE_P95_9P411MM_UNSAFE_MIN_10P519MM"
            ),
            "authority": "V31_15_20MM_15DEG_10MM_LATERAL_5STEP_SAFETY_GATE",
        }
