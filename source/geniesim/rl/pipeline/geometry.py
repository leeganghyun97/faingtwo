# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Table-frame geometry and side-effect-free reset sampling.

All XY bounds are expressed in the explicitly calibrated table frame.  No
function assumes that world X/Y are the table axes.  Invalid candidates are
returned as ``None`` before a simulator step, reward, replay insertion, HER
relabel, or episode counter can occur.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config") / "planner_environment_v1.json"
TABLE_SURFACE_HEIGHT_M = 0.70
TABLE_SURFACE_HEIGHT_TOLERANCE_M = 0.02
# Keep the serialized boundary decimal-exact; deriving 0.70 - 0.02 in binary
# floating point produces 0.6799999999999999 in manifests.
TABLE_SURFACE_HEIGHT_RANGE_M = (0.68, 0.72)
# The venue/event-pregrasp scene is a second sealed geometry identity.  It is
# not a widening of the ordinary 0.70 +/- 0.02 m contract.
EVENT_TABLE_SURFACE_HEIGHT_M = 0.730
EVENT_TABLE_SURFACE_HEIGHT_RANGE_M = (0.727, 0.733)
SEALED_TABLE_SURFACE_HEIGHT_RANGES_M = (
    TABLE_SURFACE_HEIGHT_RANGE_M,
    EVENT_TABLE_SURFACE_HEIGHT_RANGE_M,
)
TABLE_ASSET_SURFACE_OFFSET_M = 0.4650046575220938
# The latest task-level override is a compact 40 cm x 20 cm tabletop.  These
# are physical/top-surface bounds, not a smaller hidden sampling window.
TABLE_TASK_SIZE_X_M = 0.40
TABLE_TASK_SIZE_Y_M = 0.20
# Keep the table at 0.60 m while spawning the Level-1 cube in the reachable
# front half for fixed-torso, right-arm-only control.
TABLE_CENTER_WORLD_X_M = 0.60
RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_WORLD_X_M = 0.50
RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_TABLE_X_M = (
    RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_WORLD_X_M - TABLE_CENTER_WORLD_X_M
)
CUBE_EDGE_M = 0.03
CUBE_MASS_PROFILE_ENV = "GENIESIM_CUBE_MASS_PROFILE"
_CUBE_MASS_PROFILES_KG = {
    "legacy_100g": 0.100,
    "heavy_1300g": 1.300,
}
_cube_mass_profile = os.environ.get(CUBE_MASS_PROFILE_ENV, "legacy_100g")
if _cube_mass_profile not in _CUBE_MASS_PROFILES_KG:
    raise RuntimeError(
        f"{CUBE_MASS_PROFILE_ENV} must be one of "
        f"{tuple(_CUBE_MASS_PROFILES_KG)}, got {_cube_mass_profile!r}"
    )
CUBE_MASS_PROFILE = _cube_mass_profile
CUBE_MASS_NOMINAL_KG = _CUBE_MASS_PROFILES_KG[CUBE_MASS_PROFILE]
CUBE_MASS_TOLERANCE_KG = 0.010
VIRTUAL_GOAL_EDGE_M = 0.035
RESET_XY_JITTER_M = 0.02
FRICTION_SCALE_RANGE = (0.99, 1.01)
NOMINAL_GOAL_OFFSET_M = 0.10


class GeometryContractError(ValueError):
    """Raised when a measured frame or geometry violates the frozen contract."""


def validate_sealed_table_surface_height_range_m(
    value: Sequence[float],
) -> tuple[float, float]:
    """Resolve one exact reviewed table-height identity.

    Downstream consumers must use the range carried by the reset/runtime
    contract instead of silently reapplying the legacy 0.68--0.72 m range.
    Only the two reviewed identities are accepted, so this does not relax a
    hard limit or permit an arbitrary caller-supplied range.
    """

    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise GeometryContractError(
            "table surface height range must be one reviewed finite pair"
        ) from error
    if len(bounds) != 2 or not all(math.isfinite(item) for item in bounds):
        raise GeometryContractError(
            "table surface height range must be one reviewed finite pair"
        )
    for reviewed in SEALED_TABLE_SURFACE_HEIGHT_RANGES_M:
        if all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(bounds, reviewed, strict=True)
        ):
            return tuple(float(item) for item in reviewed)
    raise GeometryContractError(
        "table surface height range is not a reviewed geometry identity"
    )


def validate_authored_cube_mass_kg(value: Real) -> float:
    """Require the authored cube mass to equal the selected sealed profile.

    The 0.010 kg allowance belongs only to a downstream *measured* validation
    boundary.  It is not a range from which an asset or reset may author mass.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GeometryContractError("authored cube mass must be one real scalar")
    mass = float(value)
    if not math.isfinite(mass) or not math.isclose(
        mass, CUBE_MASS_NOMINAL_KG, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise GeometryContractError(
            f"authored cube mass must be exactly {CUBE_MASS_NOMINAL_KG:.3f} kg"
        )
    return mass


def validate_measured_cube_mass_kg(value: Real) -> float:
    """Validate a measured mass against the explicit +/-0.010 kg boundary."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GeometryContractError("measured cube mass must be one real scalar")
    mass = float(value)
    if not math.isfinite(mass) or abs(mass - CUBE_MASS_NOMINAL_KG) > (
        CUBE_MASS_TOLERANCE_KG + 1.0e-12
    ):
        raise GeometryContractError(
            "measured cube mass must be within "
            f"{CUBE_MASS_NOMINAL_KG:.3f} +/-{CUBE_MASS_TOLERANCE_KG:.3f} kg"
        )
    return mass


def _vector2(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise GeometryContractError(f"{label} must be one finite XY vector")
    return vector


@dataclass(frozen=True)
class TableBounds2D:
    """Axis-aligned task bounds in the calibrated table frame."""

    lower_xy_m: tuple[float, float]
    upper_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        lower = _vector2(self.lower_xy_m, "lower_xy_m")
        upper = _vector2(self.upper_xy_m, "upper_xy_m")
        if np.any(lower >= upper):
            raise GeometryContractError("table lower bounds must be below upper bounds")

    @classmethod
    def centered_task_region(
        cls, center_xy_m: Sequence[float] = (0.0, 0.0)
    ) -> "TableBounds2D":
        center = _vector2(center_xy_m, "center_xy_m")
        half = np.asarray(
            [TABLE_TASK_SIZE_X_M / 2.0, TABLE_TASK_SIZE_Y_M / 2.0]
        )
        return cls(tuple(center - half), tuple(center + half))


@dataclass(frozen=True)
class ResetCandidate:
    cube_center_table_xy_m: tuple[float, float]
    virtual_goal_table_xy_m: tuple[float, float]
    friction_scale: float
    rejected_candidates_before_acceptance: int
    # Sampled once for a valid episode and then frozen for its full lifetime.
    # The default preserves compatibility with audited legacy candidates while
    # all Stage 2 samplers below explicitly draw from [0.68, 0.72] m.
    table_surface_height_m: float = TABLE_SURFACE_HEIGHT_M
    # A runtime may replace a sampled candidate with a separately sealed task
    # geometry.  Keeping the allowed range on the immutable candidate prevents
    # that opt-in from widening the legacy sampler's contract globally.
    table_surface_height_range_m: tuple[float, float] = (
        TABLE_SURFACE_HEIGHT_RANGE_M
    )

    def __post_init__(self) -> None:
        height = float(self.table_surface_height_m)
        bounds = np.asarray(self.table_surface_height_range_m, dtype=np.float64)
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
        ):
            raise GeometryContractError(
                "reset table surface height range must contain two ordered finite values"
            )
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if not math.isfinite(height) or height < lower or height > upper:
            raise GeometryContractError(
                "reset table surface height is outside its sealed candidate range"
            )


def table_reference_height_for_surface(
    surface_height_m: float = TABLE_SURFACE_HEIGHT_M,
) -> float:
    if not math.isfinite(surface_height_m):
        raise GeometryContractError("surface height must be finite")
    return float(surface_height_m) - TABLE_ASSET_SURFACE_OFFSET_M


def table_surface_height_from_reference(reference_height_m: float) -> float:
    if not math.isfinite(reference_height_m):
        raise GeometryContractError("table reference height must be finite")
    return float(reference_height_m) + TABLE_ASSET_SURFACE_OFFSET_M


def validate_table_surface_height(
    measured_height_m: float,
    *,
    target_height_m: float = TABLE_SURFACE_HEIGHT_M,
    tolerance_m: float = TABLE_SURFACE_HEIGHT_TOLERANCE_M,
) -> float:
    for value, label in (
        (measured_height_m, "measured_height_m"),
        (target_height_m, "target_height_m"),
        (tolerance_m, "tolerance_m"),
    ):
        if not math.isfinite(value):
            raise GeometryContractError(f"{label} must be finite")
    if tolerance_m < 0.0:
        raise GeometryContractError("tolerance_m cannot be negative")
    error = abs(float(measured_height_m) - float(target_height_m))
    if error > float(tolerance_m) and not math.isclose(
        error, float(tolerance_m), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise GeometryContractError(
            f"table surface height {measured_height_m:.6f} m is outside "
            f"{target_height_m:.6f} +/- {tolerance_m:.6f} m"
        )
    return error


def _footprint_valid(
    center: Sequence[float],
    half_extent: float | Sequence[float],
    table_bounds: TableBounds2D,
    safety_margin: float,
) -> bool:
    try:
        center_xy = _vector2(center, "center")
        if np.isscalar(half_extent):
            half = np.full(2, float(half_extent), dtype=np.float64)
        else:
            half = _vector2(half_extent, "half_extent")
    except (TypeError, ValueError, GeometryContractError):
        return False
    if (
        not np.all(np.isfinite(half))
        or np.any(half <= 0.0)
        or not math.isfinite(safety_margin)
        or safety_margin < 0.0
    ):
        return False
    lower = np.asarray(table_bounds.lower_xy_m, dtype=np.float64)
    upper = np.asarray(table_bounds.upper_xy_m, dtype=np.float64)
    return bool(
        np.all(center_xy - half - safety_margin >= lower)
        and np.all(center_xy + half + safety_margin <= upper)
    )


def is_cube_spawn_valid(
    cube_center: Sequence[float],
    cube_half_extent: float | Sequence[float],
    table_bounds: TableBounds2D,
    safety_margin: float = 0.0,
) -> bool:
    return _footprint_valid(
        cube_center, cube_half_extent, table_bounds, safety_margin
    )


def is_virtual_goal_valid(
    goal_center: Sequence[float],
    goal_half_extent: float | Sequence[float],
    table_bounds: TableBounds2D,
    safety_margin: float = 0.0,
) -> bool:
    return _footprint_valid(goal_center, goal_half_extent, table_bounds, safety_margin)


def sample_reset_candidate(
    rng: np.random.Generator,
    *,
    nominal_cube_table_xy_m: Sequence[float],
    table_bounds: TableBounds2D,
    safety_margin_m: float = 0.0,
    reference_path_valid: Callable[[np.ndarray, np.ndarray], bool],
    rejected_before: int = 0,
) -> ResetCandidate | None:
    """Sample exactly once and return ``None`` for a side-effect-free rejection."""

    nominal = _vector2(nominal_cube_table_xy_m, "nominal_cube_table_xy_m")
    cube = nominal + rng.uniform(-RESET_XY_JITTER_M, RESET_XY_JITTER_M, size=2)
    angle = float(rng.uniform(-math.pi, math.pi))
    goal = cube + NOMINAL_GOAL_OFFSET_M * np.asarray(
        [math.cos(angle), math.sin(angle)], dtype=np.float64
    )
    friction = float(rng.uniform(*FRICTION_SCALE_RANGE))
    table_surface_height = float(rng.uniform(*TABLE_SURFACE_HEIGHT_RANGE_M))
    if not is_cube_spawn_valid(
        cube, CUBE_EDGE_M / 2.0, table_bounds, safety_margin_m
    ):
        return None
    if not is_virtual_goal_valid(
        goal, VIRTUAL_GOAL_EDGE_M / 2.0, table_bounds, safety_margin_m
    ):
        return None
    if not bool(reference_path_valid(cube.copy(), goal.copy())):
        return None
    return ResetCandidate(
        cube_center_table_xy_m=(float(cube[0]), float(cube[1])),
        virtual_goal_table_xy_m=(float(goal[0]), float(goal[1])),
        friction_scale=friction,
        rejected_candidates_before_acceptance=int(rejected_before),
        table_surface_height_m=table_surface_height,
    )


def sample_valid_reset(
    rng: np.random.Generator,
    *,
    nominal_cube_table_xy_m: Sequence[float],
    table_bounds: TableBounds2D,
    safety_margin_m: float = 0.0,
    reference_path_valid: Callable[[np.ndarray, np.ndarray], bool],
    max_attempts: int = 10_000,
) -> ResetCandidate:
    """Resample invalid candidates without exposing them as episodes or steps."""

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise GeometryContractError("max_attempts must be a positive integer")
    for rejected in range(max_attempts):
        candidate = sample_reset_candidate(
            rng,
            nominal_cube_table_xy_m=nominal_cube_table_xy_m,
            table_bounds=table_bounds,
            safety_margin_m=safety_margin_m,
            reference_path_valid=reference_path_valid,
            rejected_before=rejected,
        )
        if candidate is not None:
            return candidate
    raise GeometryContractError(
        "no valid reset was found; configuration is infeasible, and no simulation "
        "step/reward/replay/episode side effect was performed"
    )


__all__ = [
    "CUBE_EDGE_M",
    "CUBE_MASS_NOMINAL_KG",
    "CUBE_MASS_PROFILE",
    "CUBE_MASS_PROFILE_ENV",
    "CUBE_MASS_TOLERANCE_KG",
    "DEFAULT_CONFIG_PATH",
    "FRICTION_SCALE_RANGE",
    "GeometryContractError",
    "ResetCandidate",
    "RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_TABLE_X_M",
    "RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_WORLD_X_M",
    "TABLE_SURFACE_HEIGHT_M",
    "TABLE_SURFACE_HEIGHT_RANGE_M",
    "TABLE_SURFACE_HEIGHT_TOLERANCE_M",
    "TABLE_CENTER_WORLD_X_M",
    "TABLE_TASK_SIZE_X_M",
    "TABLE_TASK_SIZE_Y_M",
    "TableBounds2D",
    "VIRTUAL_GOAL_EDGE_M",
    "is_cube_spawn_valid",
    "is_virtual_goal_valid",
    "sample_reset_candidate",
    "sample_valid_reset",
    "table_reference_height_for_surface",
    "table_surface_height_from_reference",
    "validate_authored_cube_mass_kg",
    "validate_measured_cube_mass_kg",
    "validate_table_surface_height",
]
