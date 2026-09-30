# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Diagnostic-only, source-bound primary-pad/cube geometry measurements.

The functions in this module never form a policy observation.  They read the
collision-enabled USD meshes for the two Candidate-A primary pads and compare
them with the live cube oriented box.  The implementation deliberately keeps
all Isaac/USD imports inside the runtime entry point so the geometry kernels
remain unit-testable without starting Isaac.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


PRIMARY_PAD_BODIES = (
    "gripper_r_inner_link4",
    "gripper_r_outer_link4",
)


class PrivilegedGeometryOracleError(RuntimeError):
    pass


def _point_aabb_distance(point: np.ndarray, half_extents: np.ndarray) -> float:
    excess = np.maximum(np.abs(point) - half_extents, 0.0)
    return float(np.linalg.norm(excess))


def _closest_point_on_triangle(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Closest point from *point* to triangle ABC (Ericson region tests)."""

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = c - b
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * edge
    denominator = 1.0 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)


def _segment_segment_distance(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> float:
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    small = 1.0e-15
    s_numerator = denominator
    s_denominator = denominator
    t_numerator = denominator
    t_denominator = denominator
    if denominator < small:
        s_numerator = 0.0
        s_denominator = 1.0
        t_numerator = e
        t_denominator = c
    else:
        s_numerator = b * e - c * d
        t_numerator = a * e - b * d
        if s_numerator < 0.0:
            s_numerator = 0.0
            t_numerator = e
            t_denominator = c
        elif s_numerator > s_denominator:
            s_numerator = s_denominator
            t_numerator = e + b
            t_denominator = c
    if t_numerator < 0.0:
        t_numerator = 0.0
        if -d < 0.0:
            s_numerator = 0.0
        elif -d > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d
            s_denominator = a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if (-d + b) < 0.0:
            s_numerator = 0.0
        elif (-d + b) > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d + b
            s_denominator = a
    sc = 0.0 if abs(s_numerator) < small else s_numerator / s_denominator
    tc = 0.0 if abs(t_numerator) < small else t_numerator / t_denominator
    return float(np.linalg.norm(w + sc * u - tc * v))


def _segment_intersects_aabb(
    start: np.ndarray, end: np.ndarray, half_extents: np.ndarray
) -> bool:
    direction = end - start
    minimum = -half_extents
    maximum = half_extents
    t_low, t_high = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < 1.0e-15:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        inverse = 1.0 / float(direction[axis])
        first = (float(minimum[axis]) - float(start[axis])) * inverse
        second = (float(maximum[axis]) - float(start[axis])) * inverse
        if first > second:
            first, second = second, first
        t_low = max(t_low, first)
        t_high = min(t_high, second)
        if t_low > t_high:
            return False
    return True


def _segment_intersects_triangle(
    start: np.ndarray,
    end: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    direction = end - start
    edge1 = b - a
    edge2 = c - a
    cross = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, cross))
    if abs(determinant) < 1.0e-15:
        return False
    inverse = 1.0 / determinant
    offset = start - a
    u = inverse * float(np.dot(offset, cross))
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(offset, edge1)
    v = inverse * float(np.dot(direction, q))
    if v < 0.0 or u + v > 1.0:
        return False
    t = inverse * float(np.dot(edge2, q))
    return 0.0 <= t <= 1.0


def _box_vertices_and_edges(half_extents: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    vertices = np.asarray(
        [
            (sx * half_extents[0], sy * half_extents[1], sz * half_extents[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    edges: list[tuple[int, int]] = []
    for first in range(8):
        for second in range(first + 1, 8):
            if int(np.count_nonzero(vertices[first] != vertices[second])) == 1:
                edges.append((first, second))
    return vertices, tuple(edges)


def triangle_aabb_distance(
    triangle: Sequence[Sequence[float]], half_extents: Sequence[float]
) -> float:
    """Exact Euclidean distance between one triangle and an axis-aligned box."""

    tri = np.asarray(triangle, dtype=np.float64)
    half = np.asarray(half_extents, dtype=np.float64)
    if tri.shape != (3, 3) or half.shape != (3,):
        raise ValueError("TRIANGLE_AABB_SHAPE_INVALID")
    if not np.isfinite(tri).all() or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise ValueError("TRIANGLE_AABB_VALUE_INVALID")
    triangle_edges = ((0, 1), (1, 2), (2, 0))
    if any(_segment_intersects_aabb(tri[i], tri[j], half) for i, j in triangle_edges):
        return 0.0
    box_vertices, box_edges = _box_vertices_and_edges(half)
    if any(
        _segment_intersects_triangle(box_vertices[i], box_vertices[j], *tri)
        for i, j in box_edges
    ):
        return 0.0
    best = min(_point_aabb_distance(vertex, half) for vertex in tri)
    best = min(
        best,
        *(float(np.linalg.norm(vertex - _closest_point_on_triangle(vertex, *tri))) for vertex in box_vertices),
    )
    for tri_i, tri_j in triangle_edges:
        for box_i, box_j in box_edges:
            best = min(
                best,
                _segment_segment_distance(
                    tri[tri_i], tri[tri_j], box_vertices[box_i], box_vertices[box_j]
                ),
            )
    return float(best)


def mesh_to_cube_obb_gap(
    triangles_world_m: np.ndarray,
    *,
    cube_center_world_m: Sequence[float],
    cube_rotation_world: np.ndarray,
    cube_half_extents_m: Sequence[float],
) -> float:
    triangles = np.asarray(triangles_world_m, dtype=np.float64)
    center = np.asarray(cube_center_world_m, dtype=np.float64)
    rotation = np.asarray(cube_rotation_world, dtype=np.float64)
    half = np.asarray(cube_half_extents_m, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("MESH_TRIANGLES_SHAPE_INVALID")
    local = np.einsum("ij,ntj->nti", rotation.T, triangles - center)
    # Exact triangle/box evaluation is expensive for the 40k+ source meshes.
    # A triangle AABB distance is a conservative lower bound.  Evaluate exact
    # geometry in increasing lower-bound order and stop once no remaining
    # triangle can improve the current result; this preserves exactness while
    # avoiding a full Python traversal of distant triangles.
    tri_min = local.min(axis=1)
    tri_max = local.max(axis=1)
    lower_delta = np.maximum(np.maximum(-half - tri_max, tri_min - half), 0.0)
    lower_bound = np.linalg.norm(lower_delta, axis=1)
    vertex_delta = np.maximum(np.abs(local) - half, 0.0)
    best = float(np.linalg.norm(vertex_delta, axis=2).min())
    for index in np.argsort(lower_bound):
        if float(lower_bound[index]) >= best:
            break
        best = min(best, triangle_aabb_distance(local[index], half))
        if best <= 0.0:
            return 0.0
    return best


def _triangulate_faces(
    vertices: np.ndarray, counts: Iterable[int], indices: Iterable[int]
) -> np.ndarray:
    triangles: list[np.ndarray] = []
    cursor = 0
    flat = [int(value) for value in indices]
    for raw_count in counts:
        count = int(raw_count)
        face = flat[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            triangles.append(vertices[[face[0], face[offset], face[offset + 1]]])
    if cursor != len(flat) or not triangles:
        raise PrivilegedGeometryOracleError("PAD_COLLISION_MESH_TRIANGULATION_FAILED")
    return np.asarray(triangles, dtype=np.float64)


def _runtime_collision_mesh(
    stage: Any,
    body_name: str,
    *,
    body_position_world_m: Sequence[float],
    body_quat_world_xyzw: Sequence[float],
    environment_root_path: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from pxr import Gf, Usd, UsdGeom

    root = (
        str(environment_root_path).rstrip("/")
        if environment_root_path is not None
        else None
    )
    if root is not None and not stage.GetPrimAtPath(root).IsValid():
        raise PrivilegedGeometryOracleError(
            f"PAD_ENVIRONMENT_ROOT_MISSING:{root}"
        )
    body_candidates = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).endswith(f"/{body_name}")
        and (root is None or str(prim.GetPath()).startswith(root + "/"))
    ]
    if len(body_candidates) != 1:
        raise PrivilegedGeometryOracleError(
            f"PAD_RIGID_BODY_AUTHORITY_AMBIGUOUS:{body_name}:{len(body_candidates)}"
        )
    body_prim = body_candidates[0]
    authored_body_world = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    authored_world_to_body = authored_body_world.GetInverse()
    live_position = np.asarray(body_position_world_m, dtype=np.float64)
    live_quaternion = np.asarray(body_quat_world_xyzw, dtype=np.float64)
    if live_position.shape != (3,) or live_quaternion.shape != (4,):
        raise PrivilegedGeometryOracleError("PAD_LIVE_BODY_POSE_SHAPE_INVALID")
    live_quaternion /= np.linalg.norm(live_quaternion)
    x, y, z, w = live_quaternion
    live_rotation = np.asarray(
        (
            (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
        ),
        dtype=np.float64,
    )
    candidates: list[tuple[np.ndarray, str]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if root is not None and not path.startswith(root + "/"):
            continue
        if f"/{body_name}/" not in path or not prim.IsA(UsdGeom.Mesh):
            continue
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if not enabled.IsValid() or not bool(enabled.Get()):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        authored_world = np.asarray(
            [matrix.Transform(Gf.Vec3d(*point)) for point in points], dtype=np.float64
        )
        body_local = np.asarray(
            [
                authored_world_to_body.Transform(Gf.Vec3d(*point))
                for point in authored_world
            ],
            dtype=np.float64,
        )
        world = body_local @ live_rotation.T + live_position
        triangles = _triangulate_faces(
            world,
            mesh.GetFaceVertexCountsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),
        )
        candidates.append((triangles, path))
    if len(candidates) != 1:
        raise PrivilegedGeometryOracleError(
            f"PAD_COLLISION_MESH_AUTHORITY_AMBIGUOUS:{body_name}:{len(candidates)}"
        )
    triangles, path = candidates[0]
    return triangles, {
        "body": body_name,
        "collision_mesh_prim_path": path,
        "triangle_count": int(triangles.shape[0]),
        "geometry_source": "LIVE_USD_PHYSICS_COLLISION_ENABLED_MESH",
        "transform_source": "LIVE_PHYSX_BODY_POSE_PLUS_AUTHORED_MESH_TO_BODY",
        "environment_root_path": root,
    }


def _rotation_from_xyzw(quaternion_xyzw: Sequence[float], *, token: str) -> np.ndarray:
    """Return a finite SO(3) matrix without changing quaternion authority."""

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise PrivilegedGeometryOracleError(f"{token}_QUATERNION_INVALID")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise PrivilegedGeometryOracleError(f"{token}_QUATERNION_DEGENERATE")
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
        ),
        dtype=np.float64,
    )


def _measure_primary_pad_cube_geometry(
    *,
    meshes_world_m: dict[str, np.ndarray],
    authority: dict[str, Any],
    cube_center_world_m: Sequence[float],
    cube_quat_world_xyzw: Sequence[float],
    cube_half_extents_m: Sequence[float],
    table_surface_height_m: float | None = None,
) -> dict[str, Any]:
    """Measure the source-bound meshes against one live cube OBB."""

    rotation = _rotation_from_xyzw(cube_quat_world_xyzw, token="CUBE")
    half = np.asarray(cube_half_extents_m, dtype=np.float64)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise PrivilegedGeometryOracleError("CUBE_HALF_EXTENTS_INVALID")
    gaps = {
        body: mesh_to_cube_obb_gap(
            mesh,
            cube_center_world_m=cube_center_world_m,
            cube_rotation_world=rotation,
            cube_half_extents_m=half,
        )
        for body, mesh in meshes_world_m.items()
    }
    inner_vertices = meshes_world_m[PRIMARY_PAD_BODIES[0]].reshape(-1, 3)
    outer_vertices = meshes_world_m[PRIMARY_PAD_BODIES[1]].reshape(-1, 3)
    inner_center = inner_vertices.mean(axis=0)
    outer_center = outer_vertices.mean(axis=0)
    grasp_axis = outer_center - inner_center
    axis_norm = float(np.linalg.norm(grasp_axis))
    if axis_norm <= 1.0e-12:
        raise PrivilegedGeometryOracleError("PAD_GRASP_AXIS_DEGENERATE")
    grasp_axis /= axis_norm
    inner_toward = float(np.max(inner_vertices @ grasp_axis))
    outer_toward = float(np.min(outer_vertices @ grasp_axis))
    aperture = outer_toward - inner_toward
    cube_width = 2.0 * float(np.sum(np.abs(rotation.T @ grasp_axis) * half))
    cube_projection = float(np.dot(np.asarray(cube_center_world_m), grasp_axis))
    # These are the exact signed distances to the two containment planes used
    # by ``cube_between_primary_pads`` below.  They are deliberately distinct
    # from the non-negative mesh-to-OBB surface gaps above: only these values
    # retain a physical negative side when the cube leaves the pad interval.
    inner_containment_margin = cube_projection - inner_toward
    outer_containment_margin = outer_toward - cube_projection
    primary_pad_containment_margin = min(
        inner_containment_margin, outer_containment_margin
    )
    cube_between_pads = inner_toward <= cube_projection <= outer_toward
    result = {
        "authority": authority,
        "inner_pad_cube_gap_mm": gaps[PRIMARY_PAD_BODIES[0]] * 1000.0,
        "outer_pad_cube_gap_mm": gaps[PRIMARY_PAD_BODIES[1]] * 1000.0,
        "minimum_primary_pad_cube_gap_mm": min(gaps.values()) * 1000.0,
        "min_pad_surface_gap_mm": min(gaps.values()) * 1000.0,
        "grasp_axis_world": grasp_axis.tolist(),
        "inner_toward_world_m": inner_toward,
        "outer_toward_world_m": outer_toward,
        "cube_projection_world_m": cube_projection,
        "inner_containment_margin_mm": inner_containment_margin * 1000.0,
        "outer_containment_margin_mm": outer_containment_margin * 1000.0,
        "primary_pad_containment_margin_mm": (
            primary_pad_containment_margin * 1000.0
        ),
        "gripper_aperture_mm": aperture * 1000.0,
        "cube_effective_width_mm": cube_width * 1000.0,
        "required_close_stroke_mm": (aperture - cube_width) * 1000.0,
        "aperture_margin_mm": (aperture - cube_width) * 1000.0,
        "cube_between_primary_pads": bool(cube_between_pads),
        "aperture_geometrically_compatible": bool(
            cube_between_pads and aperture >= cube_width
        ),
        "student_observation_field_count": 0,
    }
    if table_surface_height_m is not None:
        table_height = float(table_surface_height_m)
        if not math.isfinite(table_height):
            raise PrivilegedGeometryOracleError("TABLE_SURFACE_HEIGHT_INVALID")
        table_clearance = {
            body: float(mesh.reshape(-1, 3)[:, 2].min() - table_height)
            for body, mesh in meshes_world_m.items()
        }
        result.update(
            {
                "inner_link4_table_clearance_mm": (
                    table_clearance[PRIMARY_PAD_BODIES[0]] * 1000.0
                ),
                "outer_link4_table_clearance_mm": (
                    table_clearance[PRIMARY_PAD_BODIES[1]] * 1000.0
                ),
                "minimum_primary_pad_table_clearance_mm": (
                    min(table_clearance.values()) * 1000.0
                ),
                "table_surface_height_m": table_height,
                "table_clearance_authority": (
                    "LIVE_USD_PHYSICS_COLLISION_ENABLED_MESH_MINIMUM_WORLD_Z"
                ),
            }
        )
    return result


class RuntimePrimaryPadCubeOracleCache:
    """Read USD pad meshes once and evaluate them at current PhysX poses.

    The cache is diagnostics-only and scoped to exactly one cloned Isaac
    environment.  Mesh topology and authored collision authority are static;
    pad transforms and cube pose are supplied for every measurement.  This
    makes a 25-Hz privileged comparison feasible without replacing the
    collision-enabled mesh authority with a link-origin proxy.
    """

    def __init__(
        self,
        *,
        initial_body_pose_world_m_xyzw_by_name: dict[str, Sequence[float]],
        environment_root_path: str | None = None,
    ) -> None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        local_meshes: dict[str, np.ndarray] = {}
        authority: dict[str, Any] = {}
        for body in PRIMARY_PAD_BODIES:
            pose = initial_body_pose_world_m_xyzw_by_name.get(body)
            if pose is None or len(pose) != 7:
                raise PrivilegedGeometryOracleError(
                    f"PAD_LIVE_BODY_POSE_MISSING:{body}"
                )
            position = np.asarray(pose[:3], dtype=np.float64)
            rotation = _rotation_from_xyzw(pose[3:], token=f"PAD_{body}")
            world_mesh, receipt = _runtime_collision_mesh(
                stage,
                body,
                body_position_world_m=position,
                body_quat_world_xyzw=pose[3:],
                environment_root_path=environment_root_path,
            )
            # ``world = local @ R.T + p``; retain the exact body-local source
            # mesh so every later query remains tied to the live body pose.
            local_meshes[body] = (world_mesh - position) @ rotation
            authority[body] = receipt
        self._local_meshes = local_meshes
        self._local_vertices = {
            body: np.unique(mesh.reshape(-1, 3), axis=0)
            for body, mesh in local_meshes.items()
        }
        self._authority = authority
        self.environment_root_path = environment_root_path

    def measure(
        self,
        *,
        cube_center_world_m: Sequence[float],
        cube_quat_world_xyzw: Sequence[float],
        cube_half_extents_m: Sequence[float],
        body_pose_world_m_xyzw_by_name: dict[str, Sequence[float]],
        table_surface_height_m: float | None = None,
    ) -> dict[str, Any]:
        meshes: dict[str, np.ndarray] = {}
        for body in PRIMARY_PAD_BODIES:
            pose = body_pose_world_m_xyzw_by_name.get(body)
            if pose is None or len(pose) != 7:
                raise PrivilegedGeometryOracleError(
                    f"PAD_LIVE_BODY_POSE_MISSING:{body}"
                )
            position = np.asarray(pose[:3], dtype=np.float64)
            rotation = _rotation_from_xyzw(pose[3:], token=f"PAD_{body}")
            meshes[body] = self._local_meshes[body] @ rotation.T + position
        result = _measure_primary_pad_cube_geometry(
            meshes_world_m=meshes,
            authority=self._authority,
            cube_center_world_m=cube_center_world_m,
            cube_quat_world_xyzw=cube_quat_world_xyzw,
            cube_half_extents_m=cube_half_extents_m,
            table_surface_height_m=table_surface_height_m,
        )
        result["geometry_cache"] = {
            "enabled": True,
            "environment_root_path": self.environment_root_path,
            "mesh_authority": "LIVE_USD_PHYSICS_COLLISION_ENABLED_MESH",
        }
        return result

    def measure_table_clearance(
        self,
        *,
        table_surface_height_m: float,
        body_pose_world_m_xyzw_by_name: dict[str, Sequence[float]],
    ) -> dict[str, Any]:
        """Measure exact link4 collision-mesh clearance without cube work."""

        table_height = float(table_surface_height_m)
        if not math.isfinite(table_height):
            raise PrivilegedGeometryOracleError("TABLE_SURFACE_HEIGHT_INVALID")
        clearances: dict[str, float] = {}
        for body in PRIMARY_PAD_BODIES:
            pose = body_pose_world_m_xyzw_by_name.get(body)
            if pose is None or len(pose) != 7:
                raise PrivilegedGeometryOracleError(
                    f"PAD_LIVE_BODY_POSE_MISSING:{body}"
                )
            position = np.asarray(pose[:3], dtype=np.float64)
            rotation = _rotation_from_xyzw(pose[3:], token=f"PAD_{body}")
            world_vertices = self._local_vertices[body] @ rotation.T + position
            clearances[body] = float(world_vertices[:, 2].min() - table_height)
        return {
            "inner_link4_table_clearance_m": clearances[PRIMARY_PAD_BODIES[0]],
            "outer_link4_table_clearance_m": clearances[PRIMARY_PAD_BODIES[1]],
            "minimum_primary_pad_table_clearance_m": min(clearances.values()),
            "table_surface_height_m": table_height,
            "authority": "LIVE_USD_PHYSICS_COLLISION_ENABLED_MESH_MINIMUM_WORLD_Z",
            "student_observation_field_count": 0,
        }


def runtime_primary_pad_cube_oracle(
    *,
    cube_center_world_m: Sequence[float],
    cube_quat_world_xyzw: Sequence[float],
    cube_half_extents_m: Sequence[float],
    body_pose_world_m_xyzw_by_name: dict[str, Sequence[float]],
    environment_root_path: str | None = None,
    table_surface_height_m: float | None = None,
) -> dict[str, Any]:
    """Return exact mesh/OBB separation and aperture from the live USD stage."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    meshes: dict[str, np.ndarray] = {}
    authority: dict[str, Any] = {}
    for body in PRIMARY_PAD_BODIES:
        if body not in body_pose_world_m_xyzw_by_name:
            raise PrivilegedGeometryOracleError(
                f"PAD_LIVE_BODY_POSE_MISSING:{body}"
            )
        body_pose = tuple(body_pose_world_m_xyzw_by_name[body])
        if len(body_pose) != 7:
            raise PrivilegedGeometryOracleError(
                f"PAD_LIVE_BODY_POSE_WIDTH:{body}:{len(body_pose)}"
            )
        triangles, receipt = _runtime_collision_mesh(
            stage,
            body,
            body_position_world_m=body_pose[:3],
            body_quat_world_xyzw=body_pose[3:],
            environment_root_path=environment_root_path,
        )
        meshes[body] = triangles
        authority[body] = receipt
    return _measure_primary_pad_cube_geometry(
        meshes_world_m=meshes,
        authority=authority,
        cube_center_world_m=cube_center_world_m,
        cube_quat_world_xyzw=cube_quat_world_xyzw,
        cube_half_extents_m=cube_half_extents_m,
        table_surface_height_m=table_surface_height_m,
    )


__all__ = [
    "PRIMARY_PAD_BODIES",
    "PrivilegedGeometryOracleError",
    "RuntimePrimaryPadCubeOracleCache",
    "mesh_to_cube_obb_gap",
    "runtime_primary_pad_cube_oracle",
    "triangle_aabb_distance",
]
