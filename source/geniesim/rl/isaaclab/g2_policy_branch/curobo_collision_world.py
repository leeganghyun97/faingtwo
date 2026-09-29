# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Source-owned cuRobo collision world for the G2 exact-4D task.

The task-authored table and cube are represented as exact oriented boxes in
the robot root frame.  Robot self collision remains owned by the checked-in
cuRobo sphere model referenced by :mod:`curobo_planner_authority`.  This
module is static: it does not create Isaac, move the robot, or alter either
the Production or M2-qualified USD.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, Mapping

from ..g2_keyboard_pose import (
    G2_KEYBOARD_CUBE_CENTER_WORLD_M,
    G2_KEYBOARD_CUBE_X_RANGE_M,
    G2_KEYBOARD_CUBE_Y_RANGE_M,
    G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M,
    G2_KEYBOARD_SIDE_PREGRASP_AXIAL_OFFSET_M,
    G2_KEYBOARD_TABLE_CENTER_WORLD_M,
)
from ..g2_lift_methodology import G2LiftSandboxContract
from .curobo_planner_authority import authority_manifest


CUROBO_COLLISION_WORLD_SCHEMA = "g2_curobo_collision_world_authority_v1"
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)
# Isaac's live root-state buffer is float32.  Historical contact-free runs
# read 0.7599999308586121 m for the source-authored 0.760 m cube Z, a
# 6.92e-8 m representation error.  This is a frame/dtype receipt tolerance,
# not a workspace expansion or a contact-clearance threshold.
SOURCE_RESET_FLOAT32_POSITION_TOLERANCE_M = 1.0e-6


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validated_cube_center_root_m(
    cube_center_root_m: Sequence[float],
) -> tuple[float, float, float]:
    """Validate a source-owned reset cube center for offline replanning only.

    The helper deliberately accepts only the task's configured XY reset range
    and its nominal table-supported Z.  It is not a new workspace authority
    and it must not be used to broaden the live task distribution.
    """

    if len(cube_center_root_m) != 3:
        raise ValueError("CUBE_CENTER_ROOT_M_MUST_HAVE_THREE_COMPONENTS")
    cube = tuple(float(component) for component in cube_center_root_m)
    if not all(math.isfinite(component) for component in cube):
        raise ValueError("CUBE_CENTER_ROOT_M_MUST_BE_FINITE")
    if not G2_KEYBOARD_CUBE_X_RANGE_M[0] <= cube[0] <= G2_KEYBOARD_CUBE_X_RANGE_M[1]:
        raise ValueError("CUBE_CENTER_ROOT_M_X_OUTSIDE_SOURCE_RESET_RANGE")
    if not G2_KEYBOARD_CUBE_Y_RANGE_M[0] <= cube[1] <= G2_KEYBOARD_CUBE_Y_RANGE_M[1]:
        raise ValueError("CUBE_CENTER_ROOT_M_Y_OUTSIDE_SOURCE_RESET_RANGE")
    nominal_z = float(G2_KEYBOARD_CUBE_CENTER_WORLD_M[2])
    if not math.isclose(
        cube[2],
        nominal_z,
        rel_tol=0.0,
        abs_tol=SOURCE_RESET_FLOAT32_POSITION_TOLERANCE_M,
    ):
        raise ValueError("CUBE_CENTER_ROOT_M_Z_MUST_EQUAL_SOURCE_NOMINAL")
    return cube


def pregrasp_position_root_m_for_cube(
    cube_center_root_m: Sequence[float],
) -> tuple[float, float, float]:
    """Return the source side-pinch pregrasp target for a validated cube pose."""

    cube = validated_cube_center_root_m(cube_center_root_m)
    return (
        cube[0] - G2_KEYBOARD_SIDE_PREGRASP_AXIAL_OFFSET_M,
        cube[1],
        cube[2] + G2_KEYBOARD_SIDE_GRASP_CENTER_HEIGHT_OFFSET_M,
    )


def canonical_pregrasp_position_root_m() -> tuple[float, float, float]:
    """Return the existing nominal side-pinch pregrasp target in robot root."""

    return pregrasp_position_root_m_for_cube(G2_KEYBOARD_CUBE_CENTER_WORLD_M)


def build_collision_world_dict(
    *,
    include_table: bool = True,
    include_cube: bool = True,
    cube_center_root_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a cuRobo primitive world from the canonical task geometry."""

    sandbox = G2LiftSandboxContract().validated()
    cube_center = (
        validated_cube_center_root_m(cube_center_root_m)
        if cube_center_root_m is not None
        else G2_KEYBOARD_CUBE_CENTER_WORLD_M
    )
    # cuRobo's serialized WorldConfig contract uses obstacle names as mapping
    # keys; the name is materialized by WorldConfig.from_dict.
    cuboids: dict[str, dict[str, Any]] = {}
    if include_table:
        cuboids["g2_task_table"] = {
            "pose": [*G2_KEYBOARD_TABLE_CENTER_WORLD_M, *IDENTITY_WXYZ],
            "dims": list(sandbox.table_size_m),
        }
    if include_cube:
        cuboids["g2_task_cube"] = {
            "pose": [*cube_center, *IDENTITY_WXYZ],
            "dims": list(sandbox.cube_size_m),
        }
    return {"cuboid": cuboids}


def collision_world_manifest(
    *, cube_center_root_m: Sequence[float] | None = None
) -> dict[str, Any]:
    sandbox = G2LiftSandboxContract().validated()
    planner = authority_manifest()
    cube_center = (
        validated_cube_center_root_m(cube_center_root_m)
        if cube_center_root_m is not None
        else G2_KEYBOARD_CUBE_CENTER_WORLD_M
    )
    world = build_collision_world_dict(cube_center_root_m=cube_center)
    pregrasp = pregrasp_position_root_m_for_cube(cube_center)
    result: dict[str, Any] = {
        "schema": CUROBO_COLLISION_WORLD_SCHEMA,
        "frame": "base_link_equals_robot_root_at_task_reset",
        "table": {
            "center_root_m": list(G2_KEYBOARD_TABLE_CENTER_WORLD_M),
            "size_m": list(sandbox.table_size_m),
            "representation": "EXACT_AXIS_ALIGNED_CUBOID",
        },
        "cube": {
            "center_root_m": list(cube_center),
            "size_m": list(sandbox.cube_size_m),
            "x_randomization_range_m": list(G2_KEYBOARD_CUBE_X_RANGE_M),
            "y_randomization_range_m": list(G2_KEYBOARD_CUBE_Y_RANGE_M),
            "representation": (
                "EXACT_AXIS_ALIGNED_CUBOID_AT_NOMINAL_POSE"
                if cube_center_root_m is None
                else "EXACT_AXIS_ALIGNED_CUBOID_AT_SOURCE_VALIDATED_RESET_POSE"
            ),
        },
        "canonical_pregrasp_position_root_m": list(pregrasp),
        "robot_self_collision": {
            "representation": "CHECKED_IN_CUROBO_LINK_SPHERES",
            "sphere_config": planner["source"]["base_collision_config"],
            "sphere_config_sha256": planner["source"][
                "base_collision_config_sha256"
            ],
            "self_collision_ignore_policy": "SAME_CHECKED_IN_YAML",
        },
        "included_static_geometry": ["table", "cube"],
        "intentionally_omitted": {
            "ground_plane": (
                "right-arm planning chain and all configured arm/gripper spheres "
                "remain above the fixed table task; live forbidden-collision "
                "authority remains downstream"
            ),
            "visual_only_exhibition_geometry": (
                "not present in the canonical policy task scene contract"
            ),
        },
        "world": world,
        "production_or_candidate_asset_modified": False,
    }
    result["world_canonical_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "CUROBO_COLLISION_WORLD_SCHEMA",
    "build_collision_world_dict",
    "canonical_pregrasp_position_root_m",
    "collision_world_manifest",
    "pregrasp_position_root_m_for_cube",
    "validated_cube_center_root_m",
]
