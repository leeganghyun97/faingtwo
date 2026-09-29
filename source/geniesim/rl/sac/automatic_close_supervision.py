# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure contracts for automatic Candidate-A CLOSE supervision.

This module is deliberately independent of Isaac.  Runtime code supplies the
measured PhysX contact points and the simulator-owned cube pose/extents; these
helpers only transform, classify, and aggregate those measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


CONTACT_REGION_WEIGHTS: Mapping[str, float] = {
    "FACE_CENTER": 1.00,
    "FACE_OFFCENTER": 0.85,
    "UPPER_EDGE": 0.65,
    "OTHER_EDGE": 0.55,
    "CORNER": 0.40,
}
CONTACT_REGION_WEIGHT_AUTHORITY = "INITIAL_CANDIDATE_NOT_PRODUCTION_LOCKED"

# Gripper-side surface quality is deliberately separate from cube-side contact
# geometry.  These values are offline ablation candidates only: they are not a
# runtime admission threshold and do not change the existing reward function.
GRIPPER_SURFACE_CLASSES = (
    "PAD_PRIMARY",
    "RUBBER_SECONDARY",
    "RIGID_NONPAD",
    "UNKNOWN",
)
GRIPPER_SURFACE_QUALITY_CANDIDATES: Mapping[str, float] = {
    "PAD_PRIMARY": 1.00,
    "PAD_PLUS_RUBBER": 0.80,
    "RUBBER_SECONDARY": 0.60,
    "RIGID_NONPAD": 0.00,
}
GRIPPER_SURFACE_QUALITY_AUTHORITY = (
    "INITIAL_OFFLINE_ABLATION_CANDIDATE_NOT_PRODUCTION_LOCKED"
)


class AutomaticCloseSupervisionError(ValueError):
    pass


def serialize_physx_vector(
    name: str, value: Sequence[float], *, size: int = 3
) -> tuple[list[float | None], list[dict[str, object]], bool]:
    """Serialize a fixed-width PhysX vector without emitting JSON NaN.

    PhysX contact buffers can contain allocated, non-contact slots whose
    geometry is NaN.  Those slots are evidence and must not be silently
    dropped, but they also are not valid contact points.  Finite components
    stay numeric; non-finite components become JSON ``null`` with an explicit
    index/token receipt.  Arbitrary reshaping is forbidden.
    """

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise AutomaticCloseSupervisionError(
            f"{name} must have shape ({size},); actual_shape={array.shape}"
        )
    invalid = [
        {"index": int(index), "token": str(float(component))}
        for index, component in enumerate(array)
        if not np.isfinite(component)
    ]
    serialized = [
        float(component) if np.isfinite(component) else None
        for component in array
    ]
    return serialized, invalid, not invalid


def _finite_vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    # Some PhysX/Warp readbacks preserve the singleton environment axis.
    # Accept exactly that documented single-env shape, never arbitrary
    # flattening that could merge contacts or environments silently.
    if array.shape == (1, size):
        array = array[0]
    if array.shape != (size,) or not np.isfinite(array).all():
        raise AutomaticCloseSupervisionError(
            f"{name} must be finite shape ({size},) or (1,{size}); "
            f"actual_shape={array.shape}; value={array.tolist()}"
        )
    return array


def _rotation_xyzw(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = _finite_vector("cube quaternion XYZW", quaternion_xyzw, 4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3):
        raise AutomaticCloseSupervisionError("cube quaternion must be unit XYZW")
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def root_point_to_cube_local(
    point_root_m: Sequence[float],
    cube_center_root_m: Sequence[float],
    cube_quat_root_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    point = _finite_vector("contact point root", point_root_m, 3)
    center = _finite_vector("cube center root", cube_center_root_m, 3)
    local = _rotation_xyzw(cube_quat_root_xyzw).T @ (point - center)
    return tuple(float(value) for value in local)


def cube_local_point_to_root(
    point_cube_local_m: Sequence[float],
    cube_center_root_m: Sequence[float],
    cube_quat_root_xyzw: Sequence[float],
) -> tuple[float, float, float]:
    point = _finite_vector("contact point cube local", point_cube_local_m, 3)
    center = _finite_vector("cube center root", cube_center_root_m, 3)
    root = _rotation_xyzw(cube_quat_root_xyzw) @ point + center
    return tuple(float(value) for value in root)


def root_normal_to_cube_local(
    normal_root: Sequence[float], cube_quat_root_xyzw: Sequence[float]
) -> tuple[float, float, float]:
    normal = _finite_vector("contact normal root", normal_root, 3)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise AutomaticCloseSupervisionError("contact normal has zero norm")
    local = _rotation_xyzw(cube_quat_root_xyzw).T @ (normal / norm)
    return tuple(float(value) for value in local)


def cube_local_normal_to_root(
    normal_cube_local: Sequence[float], cube_quat_root_xyzw: Sequence[float]
) -> tuple[float, float, float]:
    normal = _finite_vector("contact normal cube local", normal_cube_local, 3)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise AutomaticCloseSupervisionError("contact normal has zero norm")
    root = _rotation_xyzw(cube_quat_root_xyzw) @ (normal / norm)
    return tuple(float(value) for value in root)


@dataclass(frozen=True)
class ContactRegionReceipt:
    region: str
    quality_weight: float
    point_cube_local_m: tuple[float, float, float]
    normalized_abs_coordinate: tuple[float, float, float]
    surface_axes: tuple[int, ...]


def classify_cube_contact_region(
    point_cube_local_m: Sequence[float],
    cube_half_extents_m: Sequence[float],
    *,
    region_tolerance_m: float = 0.00025,
    face_center_fraction: float = 0.50,
) -> ContactRegionReceipt:
    """Classify one measured point against an oriented box.

    The two geometric fractions are collection-analysis candidates, not safety
    or production thresholds.  The closest box surface is always included so
    small PhysX penetration/separation does not turn a real contact into an
    unclassified point.
    """

    point = _finite_vector("contact point cube local", point_cube_local_m, 3)
    half = _finite_vector("cube half extents", cube_half_extents_m, 3)
    if np.any(half <= 0.0):
        raise AutomaticCloseSupervisionError("cube half extents must be positive")
    if not 0.0 < region_tolerance_m < float(np.min(half)):
        raise AutomaticCloseSupervisionError("region tolerance is invalid")
    if not 0.0 < face_center_fraction < 1.0:
        raise AutomaticCloseSupervisionError("face center fraction is invalid")
    normalized = np.abs(point) / half
    proximity = np.abs(normalized - 1.0)
    axes = tuple(
        int(index)
        for index in np.flatnonzero(np.abs(np.abs(point) - half) <= region_tolerance_m)
    )
    if not axes:
        axes = (int(np.argmin(proximity)),)
    if len(axes) >= 3:
        region = "CORNER"
    elif len(axes) == 2:
        region = "UPPER_EDGE" if 2 in axes and point[2] >= 0.0 else "OTHER_EDGE"
    else:
        face_axis = axes[0]
        tangential = [index for index in range(3) if index != face_axis]
        centered = all(normalized[index] <= face_center_fraction for index in tangential)
        region = "FACE_CENTER" if centered else "FACE_OFFCENTER"
    return ContactRegionReceipt(
        region=region,
        quality_weight=CONTACT_REGION_WEIGHTS[region],
        point_cube_local_m=tuple(float(value) for value in point),
        normalized_abs_coordinate=tuple(float(value) for value in normalized),
        surface_axes=axes,
    )


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise AutomaticCloseSupervisionError("quality values must be finite and non-empty")
    if np.any((array < 0.0) | (array > 1.0)):
        raise AutomaticCloseSupervisionError("quality values must be in [0,1]")
    if bool(np.any(array == 0.0)):
        return 0.0
    return float(np.exp(np.log(array).mean()))


def aggregate_contact_geometry(receipts: Sequence[ContactRegionReceipt]) -> float:
    if not receipts:
        return 0.0
    return geometric_mean(receipt.quality_weight for receipt in receipts)


def aggregate_gripper_surface_quality(surface_classes: Iterable[str]) -> float:
    """Return the offline candidate quality for an authoritative surface set.

    The caller must provide classifications established by collision/prim
    authority.  This helper intentionally does not infer a class from a body
    name or material-looking token.  UNKNOWN is rejected rather than silently
    treated as a compliant surface.
    """

    surfaces = frozenset(str(value) for value in surface_classes)
    if not surfaces:
        raise AutomaticCloseSupervisionError("gripper surface set is empty")
    unsupported = surfaces.difference(GRIPPER_SURFACE_CLASSES)
    if unsupported:
        raise AutomaticCloseSupervisionError(
            f"unsupported gripper surface classes: {sorted(unsupported)}"
        )
    if "UNKNOWN" in surfaces:
        raise AutomaticCloseSupervisionError("UNKNOWN gripper surface fails closed")
    if "RIGID_NONPAD" in surfaces:
        return GRIPPER_SURFACE_QUALITY_CANDIDATES["RIGID_NONPAD"]
    if surfaces == {"PAD_PRIMARY"}:
        return GRIPPER_SURFACE_QUALITY_CANDIDATES["PAD_PRIMARY"]
    if surfaces == {"RUBBER_SECONDARY"}:
        return GRIPPER_SURFACE_QUALITY_CANDIDATES["RUBBER_SECONDARY"]
    if surfaces == {"PAD_PRIMARY", "RUBBER_SECONDARY"}:
        return GRIPPER_SURFACE_QUALITY_CANDIDATES["PAD_PLUS_RUBBER"]
    raise AutomaticCloseSupervisionError(
        f"unhandled gripper surface combination: {sorted(surfaces)}"
    )


def updated_grasp_quality(
    *, q_antipodal: float, q_force_balance: float, q_slip: float,
    q_impact: float, q_omega: float, q_contact_geometry: float,
) -> float:
    return geometric_mean(
        (q_antipodal, q_force_balance, q_slip, q_impact, q_omega, q_contact_geometry)
    )


def offline_grasp_quality_with_surface(
    *, q_antipodal: float, q_force_balance: float, q_slip: float,
    q_impact: float, q_omega: float, q_contact_geometry: float,
    q_gripper_surface: float,
) -> float:
    """Seven-term offline ablation; the six-term runtime authority is intact."""

    return geometric_mean(
        (
            q_antipodal,
            q_force_balance,
            q_slip,
            q_impact,
            q_omega,
            q_contact_geometry,
            q_gripper_surface,
        )
    )
