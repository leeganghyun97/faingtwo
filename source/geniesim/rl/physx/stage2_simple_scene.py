# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Frozen geometry contract for the planner-guided Stage 2 scene.

The learning scene is intentionally small: one fixed-base G2 with both
OmniPickers, one static 40 cm x 20 cm x 70 cm support, and one independent
3 cm dynamic cube.  This module contains no Kit imports, so launchers can
validate the requested scene before starting Isaac Sim.

All tabletop XY coordinates are expressed in the table frame whose origin is
the center of the top surface.  The default table center is 0.60 m in front of
the G2 base; changing that pose requires an explicit configuration rather than
silently reusing the old fixed-payload workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from geniesim.rl.pipeline.geometry import (
    CUBE_EDGE_M,
    CUBE_MASS_NOMINAL_KG,
    CUBE_MASS_TOLERANCE_KG,
    EVENT_TABLE_SURFACE_HEIGHT_M,
    EVENT_TABLE_SURFACE_HEIGHT_RANGE_M,
    FRICTION_SCALE_RANGE,
    RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_TABLE_X_M,
    ResetCandidate,
    TABLE_CENTER_WORLD_X_M,
    TABLE_SURFACE_HEIGHT_M,
    TABLE_SURFACE_HEIGHT_RANGE_M,
    TABLE_SURFACE_HEIGHT_TOLERANCE_M,
    TABLE_TASK_SIZE_X_M as TABLE_SIZE_X_M,
    TABLE_TASK_SIZE_Y_M as TABLE_SIZE_Y_M,
    TableBounds2D,
    VIRTUAL_GOAL_EDGE_M,
    is_cube_spawn_valid,
    is_virtual_goal_valid,
)


ENVIRONMENT_ID = "stage2_planner_guided_multicamera"
SCENE_SCHEMA = "geniesim_stage2_simple_table_scene_v1"

ROBOT_PRIM_PATH = "/World/Robot"
ROBOT_SOURCE_PRIM_PATH = "/genie"
TABLE_PRIM_PATH = "/World/Stage2Table"
CUBE_PRIM_PATH = "/World/Stage2Cube"
BASE_LINK_PRIM_PATH = f"{ROBOT_PRIM_PATH}/base_link"

CAMERA_PRIM_PATHS: Mapping[str, str] = {
    "head": f"{ROBOT_PRIM_PATH}/head_link3/head_front_Camera",
    "left_wrist": f"{ROBOT_PRIM_PATH}/gripper_l_base_link/Left_Camera",
    "right_wrist": f"{ROBOT_PRIM_PATH}/gripper_r_base_link/Right_Camera",
}
END_EFFECTOR_PRIM_PATHS: Mapping[str, str] = {
    "left": f"{ROBOT_PRIM_PATH}/gripper_l_center_link",
    "right": f"{ROBOT_PRIM_PATH}/gripper_r_center_link",
}
ARM_JOINT_NAMES: Mapping[str, tuple[str, ...]] = {
    "left": tuple(f"idx2{index}_arm_l_joint{index}" for index in range(1, 8)),
    "right": tuple(f"idx6{index}_arm_r_joint{index}" for index in range(1, 8)),
}
GRIPPER_MASTER_JOINT_NAMES: Mapping[str, str] = {
    "left": "idx41_gripper_l_outer_joint1",
    "right": "idx81_gripper_r_outer_joint1",
}
GRIPPER_MIMIC_JOINT_NAMES: Mapping[str, str] = {
    "left": "idx31_gripper_l_inner_joint1",
    "right": "idx71_gripper_r_inner_joint1",
}
GRIPPER_MIMIC_MULTIPLIER = -1.0
# The checked-in OmniPicker URDF uses the angular outer joint as its command
# authority: the lower limit is physically closed and the upper limit is
# physically open.  Keep this explicit in Stage 2 instead of inferring the
# meaning from numeric ordering.
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = math.pi / 4.0
GRIPPER_ACTION_OPEN = -1.0
GRIPPER_ACTION_CLOSED = 1.0


@dataclass(frozen=True)
class Stage2PrimPaths:
    """Environment-local prim paths used by scalar and cloned runtimes.

    The original task was authored directly below ``/World``.  A tensorized
    runtime places the same, already-audited hierarchy below
    ``/World/envs/env_N``.  Keeping every derived path in this value object
    prevents a cloned runtime from accidentally reading or commanding env 0.
    """

    robot: str = ROBOT_PRIM_PATH
    table: str = TABLE_PRIM_PATH
    cube: str = CUBE_PRIM_PATH
    material: str = "/World/Stage2CubePhysicsMaterial"

    @classmethod
    def for_environment(cls, index: int) -> "Stage2PrimPaths":
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise Stage2SimpleSceneError("environment index must be nonnegative")
        root = f"/World/envs/env_{index}"
        return cls(
            robot=f"{root}/Robot",
            table=f"{root}/Stage2Table",
            cube=f"{root}/Stage2Cube",
            material=f"{root}/Stage2CubePhysicsMaterial",
        )

    @property
    def environment_root(self) -> str:
        suffixes = ("/Robot", "/Stage2Table", "/Stage2Cube")
        for value, suffix in zip((self.robot, self.table, self.cube), suffixes):
            if not value.endswith(suffix):
                return "/World"
        roots = {
            self.robot[: -len(suffixes[0])],
            self.table[: -len(suffixes[1])],
            self.cube[: -len(suffixes[2])],
        }
        return roots.pop() if len(roots) == 1 else "/World"

    @property
    def base_link(self) -> str:
        return f"{self.robot}/base_link"

    @property
    def cameras(self) -> Mapping[str, str]:
        return {
            "head": f"{self.robot}/head_link3/head_front_Camera",
            "left_wrist": f"{self.robot}/gripper_l_base_link/Left_Camera",
            "right_wrist": f"{self.robot}/gripper_r_base_link/Right_Camera",
        }

    @property
    def end_effectors(self) -> Mapping[str, str]:
        return {
            "left": f"{self.robot}/gripper_l_center_link",
            "right": f"{self.robot}/gripper_r_center_link",
        }

    def rebase_robot_path(self, path: str) -> str:
        if path == ROBOT_PRIM_PATH:
            return self.robot
        prefix = f"{ROBOT_PRIM_PATH}/"
        if not path.startswith(prefix):
            raise Stage2SimpleSceneError(
                f"path is outside the canonical G2 hierarchy: {path}"
            )
        return f"{self.robot}/{path[len(prefix):]}"


class Stage2SimpleSceneError(ValueError):
    """Raised when a scene could differ from the requested learning scene."""


def _finite_vector(label: str, value: Sequence[float], size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise Stage2SimpleSceneError(f"{label} must be {size} finite values")
    return result


@dataclass(frozen=True)
class Stage2SimpleSceneConfig:
    """Geometry and runtime identity for one actual Stage 2 scene."""

    table_center_world_xy_m: tuple[float, float] = (TABLE_CENTER_WORLD_X_M, 0.0)
    table_size_xy_m: tuple[float, float] = (TABLE_SIZE_X_M, TABLE_SIZE_Y_M)
    table_surface_height_m: float = TABLE_SURFACE_HEIGHT_M
    table_surface_height_tolerance_m: float = TABLE_SURFACE_HEIGHT_TOLERANCE_M
    table_surface_height_range_m: tuple[float, float] = TABLE_SURFACE_HEIGHT_RANGE_M
    cube_edge_m: float = CUBE_EDGE_M
    # The cube center is exactly table_surface_z + cube_half_extent.  A hidden
    # 1 mm lift would violate the task's physical reset contract.
    cube_spawn_clearance_m: float = 0.0
    virtual_goal_edge_m: float = VIRTUAL_GOAL_EDGE_M
    footprint_safety_margin_m: float = 0.010
    nominal_cube_table_xy_m: tuple[float, float] = (
        RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_TABLE_X_M,
        0.0,
    )
    cube_reset_jitter_m: float = 0.020
    # Move 10 cm toward the robot in the table frame by default.  Reachability
    # must be revalidated whenever the world-space table centre changes.
    nominal_transport_offset_table_xy_m: tuple[float, float] = (-0.10, 0.0)
    friction_scale_range: tuple[float, float] = FRICTION_SCALE_RANGE
    robot_prim_path: str = ROBOT_PRIM_PATH
    table_prim_path: str = TABLE_PRIM_PATH
    cube_prim_path: str = CUBE_PRIM_PATH
    physics_device: str = "cuda:0"

    @property
    def prim_paths(self) -> Stage2PrimPaths:
        material_root = self.robot_prim_path.rsplit("/", maxsplit=1)[0]
        material = (
            "/World/Stage2CubePhysicsMaterial"
            if material_root == "/World"
            else f"{material_root}/Stage2CubePhysicsMaterial"
        )
        return Stage2PrimPaths(
            robot=self.robot_prim_path,
            table=self.table_prim_path,
            cube=self.cube_prim_path,
            material=material,
        )

    def validated(self) -> "Stage2SimpleSceneConfig":
        center = _finite_vector(
            "table_center_world_xy_m", self.table_center_world_xy_m, 2
        )
        size = _finite_vector("table_size_xy_m", self.table_size_xy_m, 2)
        offset = _finite_vector(
            "nominal_transport_offset_table_xy_m",
            self.nominal_transport_offset_table_xy_m,
            2,
        )
        nominal_cube = _finite_vector(
            "nominal_cube_table_xy_m", self.nominal_cube_table_xy_m, 2
        )
        if not math.isclose(
            float(center[0]),
            TABLE_CENTER_WORLD_X_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Stage2SimpleSceneError(
                "table center world X must remain exactly 0.60 m"
            )
        del center, offset
        if not np.allclose(
            nominal_cube,
            [RIGHT_ARM_FIXED_TORSO_CUBE_NOMINAL_TABLE_X_M, 0.0],
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise Stage2SimpleSceneError(
                "nominal cube table XY must remain exactly [-0.10, 0] for "
                "fixed-torso right-arm reachability"
            )
        if not math.isclose(
            self.cube_reset_jitter_m, 0.020, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise Stage2SimpleSceneError(
                "cube reset jitter must remain exactly +/-0.02 m"
            )
        friction_range = _finite_vector(
            "friction_scale_range", self.friction_scale_range, 2
        )
        height_range = _finite_vector(
            "table_surface_height_range_m", self.table_surface_height_range_m, 2
        )
        standard_table_geometry = bool(
            np.allclose(
                height_range,
                TABLE_SURFACE_HEIGHT_RANGE_M,
                atol=1.0e-12,
                rtol=0.0,
            )
            and abs(self.table_surface_height_m - TABLE_SURFACE_HEIGHT_M)
            <= TABLE_SURFACE_HEIGHT_TOLERANCE_M + 1.0e-12
        )
        event_table_geometry = bool(
            np.allclose(
                height_range,
                EVENT_TABLE_SURFACE_HEIGHT_RANGE_M,
                atol=1.0e-12,
                rtol=0.0,
            )
            and math.isclose(
                self.table_surface_height_m,
                EVENT_TABLE_SURFACE_HEIGHT_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        if not (standard_table_geometry or event_table_geometry):
            raise Stage2SimpleSceneError(
                "table geometry must be either the ordinary [0.68, 0.72] m "
                "contract or the exact event [0.727, 0.733] m / 0.730 m contract"
            )
        if (
            not np.allclose(
                friction_range, FRICTION_SCALE_RANGE, atol=1.0e-12, rtol=0.0
            )
            or friction_range[0] >= friction_range[1]
        ):
            raise Stage2SimpleSceneError(
                "cube friction scale range must remain exactly [0.99, 1.01]"
            )
        if not np.allclose(
            size, [TABLE_SIZE_X_M, TABLE_SIZE_Y_M], atol=1.0e-12, rtol=0.0
        ):
            raise Stage2SimpleSceneError(
                "Stage 2 tabletop must be exactly 0.40 m x 0.20 m"
            )
        numeric = (
            self.table_surface_height_m,
            self.table_surface_height_tolerance_m,
            self.cube_edge_m,
            self.cube_spawn_clearance_m,
            self.virtual_goal_edge_m,
            self.footprint_safety_margin_m,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise Stage2SimpleSceneError("scene dimensions must be finite")
        if not math.isclose(
            self.table_surface_height_tolerance_m,
            TABLE_SURFACE_HEIGHT_TOLERANCE_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Stage2SimpleSceneError(
                "table height tolerance must remain exactly 0.02 m"
            )
        if not math.isclose(
            self.cube_edge_m, CUBE_EDGE_M, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise Stage2SimpleSceneError("Stage 2 cube must be exactly 0.03 m")
        if not math.isclose(
            self.cube_spawn_clearance_m, 0.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise Stage2SimpleSceneError(
                "cube spawn center must be exactly table surface + 0.015 m"
            )
        if not math.isclose(
            self.virtual_goal_edge_m,
            VIRTUAL_GOAL_EDGE_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Stage2SimpleSceneError(
                "virtual goal must be exactly 0.035 m x 0.035 m"
            )
        if self.footprint_safety_margin_m < 0.0:
            raise Stage2SimpleSceneError("scene clearances cannot be negative")
        if self.physics_device != "cuda:0":
            raise Stage2SimpleSceneError(
                "actual Stage 2 PhysX must use cuda:0; CPU is not an acceptance runtime"
            )
        for label, value in (
            ("robot_prim_path", self.robot_prim_path),
            ("table_prim_path", self.table_prim_path),
            ("cube_prim_path", self.cube_prim_path),
        ):
            if not value.startswith("/World/"):
                raise Stage2SimpleSceneError(f"{label} must be an explicit /World path")
        if len({self.robot_prim_path, self.table_prim_path, self.cube_prim_path}) != 3:
            raise Stage2SimpleSceneError("robot, table and cube prim paths must differ")
        return self

    @property
    def table_bounds_table_xy_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        half = np.asarray(self.table_size_xy_m, dtype=np.float64) / 2.0
        return ((-float(half[0]), -float(half[1])), (float(half[0]), float(half[1])))

    @property
    def cube_center_world_z_m(self) -> float:
        return self.cube_center_world_z_for_surface(self.table_surface_height_m)

    def cube_center_world_z_for_surface(self, surface_height_m: float) -> float:
        if not math.isfinite(surface_height_m) or not (
            self.table_surface_height_range_m[0]
            <= surface_height_m
            <= self.table_surface_height_range_m[1]
        ):
            raise Stage2SimpleSceneError(
                "episode table surface height must be within [0.68, 0.72] m"
            )
        return (
            float(surface_height_m)
            + float(self.cube_edge_m) / 2.0
        )

    def table_to_world_position(self, table_xyz_m: Sequence[float]) -> np.ndarray:
        position = _finite_vector("table_xyz_m", table_xyz_m, 3).copy()
        position[:2] += np.asarray(self.table_center_world_xy_m, dtype=np.float64)
        position[2] += float(self.table_surface_height_m)
        return position

    def valid_cube_center(self, cube_table_xy_m: Sequence[float]) -> bool:
        lower, upper = self.table_bounds_table_xy_m
        bounds = TableBounds2D(lower_xy_m=lower, upper_xy_m=upper)
        return is_cube_spawn_valid(
            cube_table_xy_m,
            float(self.cube_edge_m) / 2.0,
            bounds,
            float(self.footprint_safety_margin_m),
        )

    def valid_virtual_goal_center(self, goal_table_xy_m: Sequence[float]) -> bool:
        """Require the complete 3.5 cm goal footprint to remain on-table."""

        lower, upper = self.table_bounds_table_xy_m
        bounds = TableBounds2D(lower_xy_m=lower, upper_xy_m=upper)
        return is_virtual_goal_valid(
            goal_table_xy_m,
            float(self.virtual_goal_edge_m) / 2.0,
            bounds,
            float(self.footprint_safety_margin_m),
        )

    def manifest(self, *, robot_asset: Path, curobo_config: Path) -> dict[str, Any]:
        self.validated()
        return {
            "schema": SCENE_SCHEMA,
            "environment": ENVIRONMENT_ID,
            "table": {
                "size_xyz_m": [TABLE_SIZE_X_M, TABLE_SIZE_Y_M, self.table_surface_height_m],
                "surface_height_m": self.table_surface_height_m,
                "surface_height_nominal_m": self.table_surface_height_m,
                "surface_height_range_m": list(self.table_surface_height_range_m),
                "height_sample_distribution": "uniform",
                "height_randomized_each_episode": True,
                "surface_height_tolerance_m": self.table_surface_height_tolerance_m,
                "bounds_table_xy_m": [
                    list(self.table_bounds_table_xy_m[0]),
                    list(self.table_bounds_table_xy_m[1]),
                ],
                "prim_path": self.table_prim_path,
            },
            "cube": {
                "edge_m": self.cube_edge_m,
                "mass_nominal_kg": CUBE_MASS_NOMINAL_KG,
                "mass_tolerance_kg": CUBE_MASS_TOLERANCE_KG,
                "mass_randomized_each_episode": False,
                "prim_path": self.cube_prim_path,
                "independent_dynamic_rigid_body": True,
                "fixed_payload": False,
            },
            "virtual_goal": {
                "size_xy_m": [self.virtual_goal_edge_m, self.virtual_goal_edge_m],
                "physics_object": False,
                "full_footprint_required": True,
            },
            "robot": {
                "name": "G2_omnipicker",
                "asset": str(robot_asset),
                "prim_path": self.robot_prim_path,
                "fixed_base": True,
                "controlled_arms": ["left", "right"],
                "independent_gripper_action": True,
            },
            "cameras": dict(self.prim_paths.cameras),
            "planner": {
                "backend": "curobo",
                "config": str(curobo_config),
                "planner_action_executed": False,
            },
            "action": {
                "dimension": 7,
                "fields": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
                "physics_authority": "sac_policy_only",
            },
            "physics_device": self.physics_device,
        }


def sample_scene_reset(
    config: Stage2SimpleSceneConfig, *, seed: int, attempt: int
) -> ResetCandidate | None:
    """Deterministically sample one side-effect-free cube/transport candidate."""

    config.validated()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise TypeError("attempt must be a nonnegative integer")
    # SeedSequence avoids repeating attempt zero after each rejected candidate.
    rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
    lower, upper = config.table_bounds_table_xy_m
    inset = config.cube_edge_m / 2.0 + config.footprint_safety_margin_m
    sampling_lower = np.asarray(lower, dtype=np.float64) + inset
    sampling_upper = np.asarray(upper, dtype=np.float64) - inset
    if np.any(sampling_lower >= sampling_upper):
        raise Stage2SimpleSceneError("cube footprint and margins make reset infeasible")
    nominal_cube = np.asarray(config.nominal_cube_table_xy_m, dtype=np.float64)
    cube = nominal_cube + rng.uniform(
        -config.cube_reset_jitter_m, config.cube_reset_jitter_m, size=2
    )
    nominal = np.asarray(config.nominal_transport_offset_table_xy_m, dtype=np.float64)
    # Mirror transport along X at either edge, then reject rather than clipping.
    signed_offset = nominal.copy()
    proposed_x = cube[0] + signed_offset[0]
    if proposed_x < sampling_lower[0] or proposed_x > sampling_upper[0]:
        signed_offset[0] *= -1.0
    transport = cube + signed_offset
    friction_scale = float(rng.uniform(*config.friction_scale_range))
    table_surface_height = float(
        rng.uniform(*config.table_surface_height_range_m)
    )
    if not config.valid_cube_center(cube) or not config.valid_virtual_goal_center(
        transport
    ):
        return None
    return ResetCandidate(
        cube_center_table_xy_m=(float(cube[0]), float(cube[1])),
        virtual_goal_table_xy_m=(float(transport[0]), float(transport[1])),
        friction_scale=friction_scale,
        rejected_candidates_before_acceptance=attempt,
        table_surface_height_m=table_surface_height,
        table_surface_height_range_m=tuple(
            float(value) for value in config.table_surface_height_range_m
        ),
    )


__all__ = [
    "ARM_JOINT_NAMES",
    "BASE_LINK_PRIM_PATH",
    "CAMERA_PRIM_PATHS",
    "CUBE_EDGE_M",
    "CUBE_MASS_NOMINAL_KG",
    "CUBE_MASS_TOLERANCE_KG",
    "CUBE_PRIM_PATH",
    "END_EFFECTOR_PRIM_PATHS",
    "ENVIRONMENT_ID",
    "FRICTION_SCALE_RANGE",
    "GRIPPER_MASTER_JOINT_NAMES",
    "GRIPPER_MIMIC_JOINT_NAMES",
    "GRIPPER_MIMIC_MULTIPLIER",
    "ROBOT_PRIM_PATH",
    "ROBOT_SOURCE_PRIM_PATH",
    "SCENE_SCHEMA",
    "Stage2SimpleSceneConfig",
    "Stage2SimpleSceneError",
    "TABLE_PRIM_PATH",
    "TABLE_SIZE_X_M",
    "TABLE_SIZE_Y_M",
    "TABLE_SURFACE_HEIGHT_M",
    "TABLE_SURFACE_HEIGHT_RANGE_M",
    "TABLE_SURFACE_HEIGHT_TOLERANCE_M",
    "sample_scene_reset",
]
