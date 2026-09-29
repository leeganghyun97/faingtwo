# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Demonstration-derived grasp-ready geometry for the G2 4-D branch.

This module deliberately gives canonical gripper commands, rather than
delayed finger motion, authority over CLOSE onset.  The resulting region is
an offline planner/evaluation contract; it is not a new controller safety
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


GRASP_READY_GEOMETRY_SCHEMA = "g2_demo_close_onset_grasp_ready_geometry_v1"


def close_onset_mask(canonical_gripper_command: np.ndarray) -> np.ndarray:
    """Return rows whose canonical command changes from OPEN to CLOSE.

    The dataset contract uses +1 for OPEN and -1 for CLOSE.  A zero command
    is rejected because it does not have an authoritative binary meaning.
    """

    command = np.asarray(canonical_gripper_command, dtype=np.float64)
    if command.ndim != 1 or command.size == 0:
        raise ValueError("canonical gripper command must be a non-empty vector")
    if not np.isfinite(command).all():
        raise ValueError("canonical gripper command contains NaN/Inf")
    if np.any(command == 0.0):
        raise ValueError("canonical gripper command contains ambiguous zero")
    closed = command < 0.0
    previous_closed = np.concatenate((np.zeros(1, dtype=bool), closed[:-1]))
    return closed & ~previous_closed


def canonicalize_xyzw(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Normalize quaternion rows and select one deterministic hemisphere."""

    values = np.asarray(quaternions_xyzw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("quaternion array must have shape [rows,4]")
    if not np.isfinite(values).all():
        raise ValueError("quaternion array contains NaN/Inf")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("quaternion array contains a zero quaternion")
    normalized = values / norms[:, None]
    reference = normalized[0]
    signs = np.where(normalized @ reference < 0.0, -1.0, 1.0)
    return normalized * signs[:, None]


def median_quaternion_xyzw(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Return the normalized component median in a common hemisphere."""

    aligned = canonicalize_xyzw(quaternions_xyzw)
    candidate = np.median(aligned, axis=0)
    norm = float(np.linalg.norm(candidate))
    if norm <= 1.0e-12:
        raise ValueError("quaternion median is degenerate")
    return candidate / norm


def quaternion_error_rad(quaternions_xyzw: np.ndarray, target_xyzw: Sequence[float]) -> np.ndarray:
    """Return sign-invariant SO(3) geodesic errors."""

    values = canonicalize_xyzw(quaternions_xyzw)
    target = np.asarray(tuple(target_xyzw), dtype=np.float64)
    if target.shape != (4,) or not np.isfinite(target).all():
        raise ValueError("target quaternion must be finite xyzw")
    target_norm = float(np.linalg.norm(target))
    if target_norm <= 1.0e-12:
        raise ValueError("target quaternion is zero")
    target /= target_norm
    dot = np.clip(np.abs(values @ target), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


@dataclass(frozen=True)
class GraspReadyRegion:
    """Robust close-onset region derived from successful demonstrations."""

    relative_xyz_median_m: tuple[float, float, float]
    relative_xyz_p05_m: tuple[float, float, float]
    relative_xyz_p95_m: tuple[float, float, float]
    absolute_axis_error_p95_m: tuple[float, float, float]
    distance_p05_m: float
    distance_p50_m: float
    distance_p95_m: float
    orientation_xyzw: tuple[float, float, float, float]
    orientation_error_p95_rad: float

    def contains(
        self,
        *,
        cube_minus_ee_root_m: Sequence[float],
        ee_orientation_xyzw: Sequence[float],
    ) -> dict[str, object]:
        relative = np.asarray(tuple(cube_minus_ee_root_m), dtype=np.float64)
        if relative.shape != (3,) or not np.isfinite(relative).all():
            raise ValueError("cube_minus_ee_root_m must be finite xyz")
        center = np.asarray(self.relative_xyz_median_m, dtype=np.float64)
        axis_tolerance = np.asarray(self.absolute_axis_error_p95_m, dtype=np.float64)
        axis_error = np.abs(relative - center)
        distance = float(np.linalg.norm(relative))
        orientation_error = float(
            quaternion_error_rad(
                np.asarray([tuple(ee_orientation_xyzw)], dtype=np.float64),
                self.orientation_xyzw,
            )[0]
        )
        checks = {
            "axis_error_within_demo_p95": bool(np.all(axis_error <= axis_tolerance)),
            "distance_within_demo_p05_p95": bool(
                self.distance_p05_m <= distance <= self.distance_p95_m
            ),
            "orientation_within_demo_p95": bool(
                orientation_error <= self.orientation_error_p95_rad
            ),
        }
        return {
            "inside": all(checks.values()),
            "checks": checks,
            "cube_minus_ee_root_m": relative.tolist(),
            "axis_error_from_nominal_m": axis_error.tolist(),
            "distance_m": distance,
            "orientation_error_rad": orientation_error,
        }


def derive_grasp_ready_region(
    relative_xyz_m: np.ndarray,
    ee_orientation_xyzw: np.ndarray,
) -> GraspReadyRegion:
    """Derive a robust P05/P95 region without inventing a fixed distance."""

    relative = np.asarray(relative_xyz_m, dtype=np.float64)
    orientation = np.asarray(ee_orientation_xyzw, dtype=np.float64)
    if relative.ndim != 2 or relative.shape[1] != 3 or relative.shape[0] < 3:
        raise ValueError("at least three relative xyz samples are required")
    if orientation.shape != (relative.shape[0], 4):
        raise ValueError("orientation rows must match relative xyz rows")
    if not np.isfinite(relative).all():
        raise ValueError("relative xyz contains NaN/Inf")
    center = np.median(relative, axis=0)
    distance = np.linalg.norm(relative, axis=1)
    nominal_orientation = median_quaternion_xyzw(orientation)
    orientation_error = quaternion_error_rad(orientation, nominal_orientation)
    return GraspReadyRegion(
        relative_xyz_median_m=tuple(float(value) for value in center),
        relative_xyz_p05_m=tuple(
            float(value) for value in np.quantile(relative, 0.05, axis=0)
        ),
        relative_xyz_p95_m=tuple(
            float(value) for value in np.quantile(relative, 0.95, axis=0)
        ),
        absolute_axis_error_p95_m=tuple(
            float(value)
            for value in np.quantile(np.abs(relative - center), 0.95, axis=0)
        ),
        distance_p05_m=float(np.quantile(distance, 0.05)),
        distance_p50_m=float(np.quantile(distance, 0.50)),
        distance_p95_m=float(np.quantile(distance, 0.95)),
        orientation_xyzw=tuple(float(value) for value in nominal_orientation),
        orientation_error_p95_rad=float(np.quantile(orientation_error, 0.95)),
    )


def region_from_mapping(value: dict[str, object]) -> GraspReadyRegion:
    """Load the region portion of an audited JSON artifact."""

    return GraspReadyRegion(
        relative_xyz_median_m=tuple(value["relative_xyz_median_m"]),  # type: ignore[arg-type]
        relative_xyz_p05_m=tuple(value["relative_xyz_p05_m"]),  # type: ignore[arg-type]
        relative_xyz_p95_m=tuple(value["relative_xyz_p95_m"]),  # type: ignore[arg-type]
        absolute_axis_error_p95_m=tuple(value["absolute_axis_error_p95_m"]),  # type: ignore[arg-type]
        distance_p05_m=float(value["distance_p05_m"]),
        distance_p50_m=float(value["distance_p50_m"]),
        distance_p95_m=float(value["distance_p95_m"]),
        orientation_xyzw=tuple(value["orientation_xyzw"]),  # type: ignore[arg-type]
        orientation_error_p95_rad=float(value["orientation_error_p95_rad"]),
    )


__all__ = [
    "GRASP_READY_GEOMETRY_SCHEMA",
    "GraspReadyRegion",
    "canonicalize_xyzw",
    "close_onset_mask",
    "derive_grasp_ready_region",
    "median_quaternion_xyzw",
    "quaternion_error_rad",
    "region_from_mapping",
]
