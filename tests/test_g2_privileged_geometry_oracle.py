# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import numpy as np

from geniesim.rl.sac.privileged_geometry_oracle import (
    mesh_to_cube_obb_gap,
    triangle_aabb_distance,
)


def test_triangle_aabb_distance_reports_separation_and_touch() -> None:
    half = (1.0, 1.0, 1.0)
    separated = ((2.0, -0.5, -0.5), (2.0, 0.5, -0.5), (2.0, 0.0, 0.5))
    touching = ((1.0, -0.5, -0.5), (1.0, 0.5, -0.5), (1.0, 0.0, 0.5))
    assert np.isclose(triangle_aabb_distance(separated, half), 1.0)
    assert np.isclose(triangle_aabb_distance(touching, half), 0.0)


def test_triangle_crossing_box_without_vertex_inside_is_zero() -> None:
    triangle = ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    assert np.isclose(triangle_aabb_distance(triangle, (0.5, 0.5, 0.5)), 0.0)


def test_mesh_cube_gap_respects_cube_rotation() -> None:
    triangles = np.asarray(
        [[[2.0, -0.5, -0.5], [2.0, 0.5, -0.5], [2.0, 0.0, 0.5]]],
        dtype=np.float64,
    )
    gap = mesh_to_cube_obb_gap(
        triangles,
        cube_center_world_m=(0.0, 0.0, 0.0),
        cube_rotation_world=np.eye(3),
        cube_half_extents_m=(1.0, 1.0, 1.0),
    )
    assert np.isclose(gap, 1.0)


def test_oracle_source_declares_no_student_observation_path() -> None:
    from pathlib import Path

    source = Path(
        "source/geniesim/rl/sac/privileged_geometry_oracle.py"
    ).read_text(encoding="utf-8")
    assert '"student_observation_field_count": 0' in source
    assert "physics:collisionEnabled" in source
    assert "gripper_r_inner_link4" in source
    assert "gripper_r_outer_link4" in source
