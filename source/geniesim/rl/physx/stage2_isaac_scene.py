# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Actual USD/PhysX composition of the simple planner-guided Stage 2 scene.

All Isaac/pxr imports are delayed until construction after ``SimulationApp``.
The composer does not create a legacy workspace reference: the table and cube
are authored directly with audited metric dimensions, so a fixed payload or
the old 25 cm x 10 cm task asset cannot leak into the new environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from geniesim.rl.pipeline.geometry import (
    GeometryContractError,
    validate_authored_cube_mass_kg,
)

from .stage2_simple_scene import (
    BASE_LINK_PRIM_PATH,
    CAMERA_PRIM_PATHS,
    CUBE_PRIM_PATH,
    CUBE_MASS_NOMINAL_KG,
    CUBE_MASS_TOLERANCE_KG,
    ENVIRONMENT_ID,
    ROBOT_PRIM_PATH,
    ROBOT_SOURCE_PRIM_PATH,
    TABLE_PRIM_PATH,
    TABLE_SIZE_X_M,
    TABLE_SIZE_Y_M,
    TABLE_SURFACE_HEIGHT_RANGE_M,
    Stage2SimpleSceneConfig,
)


class Stage2IsaacSceneError(RuntimeError):
    pass


TASK_HEAD_CAMERA_DOWNWARD_PITCH_RAD = math.radians(48.0)


@dataclass(frozen=True)
class G2CollisionManifestEntry:
    """One collision geometry approved by both the USD and checked-in URDF."""

    prim_path: str
    link_name: str
    collision_name: str
    usd_type_name: str
    urdf_geometry_kind: str
    urdf_mesh_uri: str | None = None


def _collision_entry(
    link_name: str,
    collision_name: str,
    usd_leaf_name: str,
    usd_type_name: str,
    urdf_geometry_kind: str,
    urdf_mesh_uri: str | None = None,
) -> G2CollisionManifestEntry:
    usd_collision_name = collision_name.replace(".", "_")
    return G2CollisionManifestEntry(
        prim_path=(
            f"{ROBOT_PRIM_PATH}/{link_name}/collisions/"
            f"{usd_collision_name}/{usd_leaf_name}"
        ),
        link_name=link_name,
        collision_name=collision_name,
        usd_type_name=usd_type_name,
        urdf_geometry_kind=urdf_geometry_kind,
        urdf_mesh_uri=urdf_mesh_uri,
    )


_BASE_COLLISION_MANIFEST = tuple(
    _collision_entry(
        "base_link", f"base_link_coarse_{index}", "box", "Cube", "box"
    )
    for index in range(3)
)
_BODY1_COLLISION_MANIFEST = (
    _collision_entry(
        "body_link1", "Cylinder", "cylinder", "Cylinder", "cylinder"
    ),
    _collision_entry(
        "body_link1", "Cylinder.001", "cylinder", "Cylinder", "cylinder"
    ),
    _collision_entry("body_link1", "Cube", "box", "Cube", "box"),
    _collision_entry("body_link1", "Cube.001", "box", "Cube", "box"),
    _collision_entry("body_link1", "Cube.002", "box", "Cube", "box"),
)
_BODY_MESH_COLLISION_MANIFEST = (
    _collision_entry(
        "body_link2",
        "body_link2_convex_0",
        "mesh",
        "Mesh",
        "mesh",
        "package://genie_robot_description/meshes/G2/T2/convex/body_link2.STL",
    ),
    _collision_entry(
        "body_link3",
        "body_link3_convex_0",
        "mesh",
        "Mesh",
        "mesh",
        "package://genie_robot_description/meshes/G2/T2/convex/body_link3.STL",
    ),
    *(
        _collision_entry(
            "body_link4",
            f"body_link4_convex_{index}",
            "mesh",
            "Mesh",
            "mesh",
            "package://genie_robot_description/meshes/G2/T2/convex/"
            f"body_link4_{index}.STL",
        )
        for index in range(1, 4)
    ),
    _collision_entry(
        "body_link5",
        "body_link5_convex_0",
        "mesh",
        "Mesh",
        "mesh",
        "package://genie_robot_description/meshes/G2/T2/convex/body_link5.STL",
    ),
)
_HEAD_COLLISION_MANIFEST = tuple(
    _collision_entry(
        f"head_link{index}",
        f"head_link{index}_convex_0",
        "mesh",
        "Mesh",
        "mesh",
        "package://genie_robot_description/meshes/G2/T2/convex/"
        f"head_link{index}.STL",
    )
    for index in range(1, 4)
)
_ARM_COLLISION_MANIFEST = tuple(
    _collision_entry(
        f"arm_{side}_link{index}",
        f"arm_{side}_link{index}_convex_0",
        "mesh",
        "Mesh",
        "mesh",
        "package://genie_robot_description/meshes/urdf/convex/"
        f"arm_{side}_link{index}_fine_0.STL",
    )
    for side in ("l", "r")
    for index in range(1, 8)
)

# This is deliberately an explicit, closed allowlist.  In particular, no
# visual mesh and no wheel/chassis collider is eligible for activation here.
G2_ROBOT_COLLISION_ALLOWLIST = (
    *_BASE_COLLISION_MANIFEST,
    *_BODY1_COLLISION_MANIFEST,
    *_BODY_MESH_COLLISION_MANIFEST,
    *_HEAD_COLLISION_MANIFEST,
    *_ARM_COLLISION_MANIFEST,
)

G2_GRIPPER_BASE_COLLISION_MESH_URI = (
    "package://genie_robot_description/meshes/omnipicker/convex/"
    "gripper_base_link.STL"
)
LEFT_GRIPPER_BASE_COLLISION_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/gripper_l_base_link/collisions/"
    "gripper_l_base_link_convex_0/mesh"
)
RIGHT_GRIPPER_BASE_COLLISION_PRIM_PATH = (
    f"{ROBOT_PRIM_PATH}/gripper_r_base_link/collisions/"
    "gripper_r_base_link_convex_0/mesh"
)
RIGHT_GRIPPER_BASE_COLLISION_XFORM_PATH = (
    f"{ROBOT_PRIM_PATH}/gripper_r_base_link/collisions/"
    "gripper_r_base_link_convex_0"
)
ARTICULATION_SELF_COLLISION_ATTRIBUTE = (
    "physxArticulation:enabledSelfCollisions"
)


def _finger_visual_collision_paths(side: str) -> tuple[str, ...]:
    return tuple(
        f"{ROBOT_PRIM_PATH}/gripper_{side}_{family}_link{index}/"
        f"visuals/{family}_link{index}/mesh"
        for family in ("inner", "outer")
        for index in range(1, 5)
    )


# These collision meshes are already enabled by the checked-in asset.  They
# are recorded so that the activation pass can prove it did not enable an
# arbitrary visual mesh.  The right base mesh is an instance proxy, so it must
# be addressed by its exact known path rather than discovered via Traverse().
G2_EXPECTED_PREEXISTING_ENABLED_COLLISION_PATHS = (
    LEFT_GRIPPER_BASE_COLLISION_PRIM_PATH,
    RIGHT_GRIPPER_BASE_COLLISION_PRIM_PATH,
    *_finger_visual_collision_paths("l"),
    *_finger_visual_collision_paths("r"),
)

G2_COLLISION_MESH_EQUALITY_ATTRIBUTE_NAMES = (
    "cornerIndices",
    "cornerSharpnesses",
    "creaseIndices",
    "creaseLengths",
    "creaseSharpnesses",
    "doubleSided",
    "extent",
    "faceVaryingLinearInterpolation",
    "faceVertexCounts",
    "faceVertexIndices",
    "holeIndices",
    "interpolateBoundary",
    "normals",
    "orientation",
    "physics:approximation",
    "points",
    "subdivisionScheme",
    "triangleSubdivisionRule",
)


def _required_collision_paths_by_body() -> Mapping[str, tuple[str, ...]]:
    paths: dict[str, list[str]] = {}
    for entry in G2_ROBOT_COLLISION_ALLOWLIST:
        paths.setdefault(entry.link_name, []).append(entry.prim_path)
    for side, base_path in (
        ("l", LEFT_GRIPPER_BASE_COLLISION_PRIM_PATH),
        ("r", RIGHT_GRIPPER_BASE_COLLISION_PRIM_PATH),
    ):
        paths[f"gripper_{side}_base_link"] = [base_path]
        for family in ("inner", "outer"):
            for index in range(1, 5):
                body = f"gripper_{side}_{family}_link{index}"
                paths[body] = [
                    f"{ROBOT_PRIM_PATH}/{body}/visuals/"
                    f"{family}_link{index}/mesh"
                ]
    return MappingProxyType(
        {body: tuple(body_paths) for body, body_paths in paths.items()}
    )


# Runtime consumers must query every path directly.  Prim traversal can omit
# the right gripper-base instance proxy and is therefore not authoritative.
G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY = (
    _required_collision_paths_by_body()
)

# The sealed source policy contains only adjacency that is proved by the
# pinned articulation sources: 36 direct active-collider URDF joints, four
# loop joints composed by USD, and two fixed arm-root attachments whose
# intermediate arm-base links have no collider.  Runtime overlap workarounds
# are deliberately kept out of this source authority and are identified as
# development-only below.
_G2_BODY_HEAD_STRUCTURAL_FILTER_PAIRS = frozenset(
    {
        tuple(sorted(("base_link", "body_link1"))),
        *(
            tuple(sorted((f"body_link{index}", f"body_link{index + 1}")))
            for index in range(1, 5)
        ),
        tuple(sorted(("body_link5", "head_link1"))),
        tuple(sorted(("head_link1", "head_link2"))),
        tuple(sorted(("head_link2", "head_link3"))),
    }
)
_G2_ARM_SERIAL_STRUCTURAL_FILTER_PAIRS = frozenset(
    {
        tuple(
            sorted(
                (f"arm_{side}_link{index}", f"arm_{side}_link{index + 1}")
            )
        )
        for side in ("l", "r")
        for index in range(1, 7)
    }
)
# The checked-in G2 SRDF classifies these two non-consecutive wrist links as
# ``Adjacent``.  The official drawing archive independently shows that link5
# and link7 share an overlapping wrist-housing envelope throughout the sealed
# right-arm path.  Import exactly these two entries; the SRDF's broad
# ``Default`` suppression set is intentionally not collision authority here.
G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS = frozenset(
    {
        tuple(sorted((f"arm_{side}_link5", f"arm_{side}_link7")))
        for side in ("l", "r")
    }
)
# The same pinned SRDF also identifies eight non-direct torso/head housing
# pairs.  Seven are ``Adjacent`` and body_link5/head_link3 is ``Always``.
# These are structural overlap exclusions, not runtime collision waivers.
G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIR_REASONS = MappingProxyType(
    {
        tuple(sorted(pair)): reason
        for pair, reason in (
            (("base_link", "body_link2"), "Adjacent"),
            (("body_link1", "body_link3"), "Adjacent"),
            (("body_link2", "body_link4"), "Adjacent"),
            (("body_link3", "body_link5"), "Adjacent"),
            (("body_link4", "head_link1"), "Adjacent"),
            (("body_link5", "head_link2"), "Adjacent"),
            (("head_link1", "head_link3"), "Adjacent"),
            (("body_link5", "head_link3"), "Always"),
        )
    }
)
G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIRS = frozenset(
    G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIR_REASONS
)
G2_SHARED_WRIST_STRUCTURAL_SRDF_RELATIVE_PATH = (
    "source/teleop/app/share/genie_robot_description/urdf/G2/G2.srdf"
)
G2_SHARED_WRIST_STRUCTURAL_SRDF_SHA256 = (
    "61db8bb29caaa7f74c221bcba664e2e8ced223a4faaa2ff6ce93099ac2a32cf8"
)
G2_OFFICIAL_DRAWING_ARCHIVE_RELATIVE_PATH = "G2 도면 파일.zip"
G2_OFFICIAL_DRAWING_ARCHIVE_SHA256 = (
    "f1ca8742c05846470e245d832ad30f913034b853da08a7ded76505e357c1388e"
)
G2_OFFICIAL_SHARED_WRIST_ENTRY_SHA256 = MappingProxyType(
    {
        "arm/crsB/arm_r_link5.STL": (
            "7894cd138953f66ed911af4a327ad63c6371c93e171dffbe1b026390b0288515"
        ),
        "arm/crsB/arm_r_link6.STL": (
            "dc96fe1ea9301816ba70bff8aa66c86ad92fb3a1b8ed62416b463f3ff5143f0a"
        ),
        "arm/crsB/arm_r_link7.STL": (
            "0673a201b908a1fd7374882981dc8bdb3a0c1915f6f4470c5e011b144a6fb032"
        ),
        "arm/crsB/convex/arm_r_link5.STL": (
            "1d0745a8a7fc6776461bab506a2aef7aa51ebfe901b5c490eddefc478235f150"
        ),
        "arm/crsB/convex/arm_r_link6.STL": (
            "6da22567f4b8d4a781e08b1a01dcaf7fb18684c611b87b68e90eb3f64a3dd2ba"
        ),
        "arm/crsB/convex/arm_r_link7.STL": (
            "4ecd772398304bda3367eb43ee0f9487d1e2fcc5a8fba329d6851cb108caf27f"
        ),
    }
)
G2_FIXED_CHAIN_STRUCTURAL_FILTER_PAIRS = frozenset(
    {
        tuple(sorted(("body_link5", f"arm_{side}_link1")))
        for side in ("l", "r")
    }
)
G2_USD_LOOP_STRUCTURAL_FILTER_PAIRS = frozenset(
    {
        (f"gripper_{side}_base_link", f"gripper_{side}_{family}_link2")
        for side in ("l", "r")
        for family in ("inner", "outer")
    }
)
_G2_OMNIPICKER_ARTICULATION_JOINT_FILTER_PAIRS = frozenset(
    {
        *((f"gripper_{side}_base_link", f"gripper_{side}_inner_link1")
          for side in ("l", "r")),
        *((f"gripper_{side}_base_link", f"gripper_{side}_inner_link2")
          for side in ("l", "r")),
        *((f"gripper_{side}_base_link", f"gripper_{side}_outer_link1")
          for side in ("l", "r")),
        *((f"gripper_{side}_base_link", f"gripper_{side}_outer_link2")
          for side in ("l", "r")),
        *((f"gripper_{side}_inner_link1", f"gripper_{side}_inner_link3")
          for side in ("l", "r")),
        *((f"gripper_{side}_inner_link2", f"gripper_{side}_inner_link4")
          for side in ("l", "r")),
        *((f"gripper_{side}_inner_link3", f"gripper_{side}_inner_link4")
          for side in ("l", "r")),
        *((f"gripper_{side}_outer_link1", f"gripper_{side}_outer_link3")
          for side in ("l", "r")),
        *((f"gripper_{side}_outer_link2", f"gripper_{side}_outer_link4")
          for side in ("l", "r")),
        *((f"gripper_{side}_outer_link3", f"gripper_{side}_outer_link4")
          for side in ("l", "r")),
    }
)
_G2_ACTIVE_COLLIDER_ARTICULATION_FILTER_PAIRS = frozenset(
    _G2_BODY_HEAD_STRUCTURAL_FILTER_PAIRS
    | _G2_ARM_SERIAL_STRUCTURAL_FILTER_PAIRS
    | G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS
    | G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIRS
    | _G2_OMNIPICKER_ARTICULATION_JOINT_FILTER_PAIRS
    | G2_FIXED_CHAIN_STRUCTURAL_FILTER_PAIRS
)
_G2_URDF_DIRECT_ACTIVE_COLLIDER_FILTER_PAIRS = frozenset(
    _G2_ACTIVE_COLLIDER_ARTICULATION_FILTER_PAIRS
    - G2_USD_LOOP_STRUCTURAL_FILTER_PAIRS
    - G2_FIXED_CHAIN_STRUCTURAL_FILTER_PAIRS
    - G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS
    - G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIRS
)
G2_STRUCTURAL_SELF_COLLISION_FILTER_PAIRS = tuple(
    sorted(_G2_ACTIVE_COLLIDER_ARTICULATION_FILTER_PAIRS)
)
G2_SELF_COLLISION_FILTER_PAIR_COUNT = 52
G2_SELF_COLLISION_FILTER_PAIRS_SHA256 = (
    "4872f03e9e9f7a9b8fcc1224e2e99f8d7ef64331113f419471eb7f886684df5c"
)
G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS: tuple[
    tuple[str, str], ...
] = ()
G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS = (
    ("gripper_l_inner_link1", "gripper_l_inner_link2"),
    ("gripper_l_inner_link2", "gripper_l_inner_link3"),
    ("gripper_l_inner_link4", "gripper_l_outer_link4"),
    ("gripper_l_outer_link1", "gripper_l_outer_link2"),
    ("gripper_l_outer_link2", "gripper_l_outer_link3"),
    ("gripper_r_inner_link1", "gripper_r_inner_link2"),
    ("gripper_r_inner_link2", "gripper_r_inner_link3"),
    ("gripper_r_inner_link4", "gripper_r_outer_link4"),
    ("gripper_r_outer_link1", "gripper_r_outer_link2"),
    ("gripper_r_outer_link2", "gripper_r_outer_link3"),
)
G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS_SHA256 = (
    "11fda4ad20df6220f3045c2400d48b4cff014b1f3476148bf2c61db884210094"
)
G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS = tuple(
    sorted(
        set(G2_STRUCTURAL_SELF_COLLISION_FILTER_PAIRS)
        | set(G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS)
        | set(G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS)
    )
)
G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIR_COUNT = 62
G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS_SHA256 = (
    "3ab94eb34c277b0f9ae46926fa2fa66b8c6f11bb5046be4180614eb8a300fe39"
)
G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_REASON = (
    "empty_after_SRDF_Adjacent_shared_wrist_promotion_and_exact_geometry_"
    "clearance_rejected_link5_gripper_exemptions"
)
G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_REASON = (
    "actual_Isaac_PhysX_v15_and_v2_smoke_reports_symmetric_inner1_inner2_"
    "and_outer1_outer2_plus_current_2000_step_train_reports_symmetric_"
    "inner2_inner3_and_outer2_outer3_plus_closed_reset_reports_symmetric_"
    "inner4_outer4_at_empty_full_close_closed_loop_mechanism_mesh_contacts"
)

_G2_COLLISION_SOURCE_SHA256 = {
    "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda": (
        "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
    ),
    "source/geniesim/assets/robot/G2_omnipicker/configuration/robot_base.usd": (
        "6abf3888859bc2385bd084bcdeec8fe464891110d61d3fa9ef1c7efd5a690488"
    ),
    "source/geniesim/assets/robot/G2_omnipicker/configuration/robot_physics.usd": (
        "1680e1dc7ea75a2585144eee33b1224284a80f5d25c9dacfd0caa0b9170040f2"
    ),
    "source/geniesim/assets/robot/G2_omnipicker/configuration/robot_robot.usd": (
        "a7f5238b029832980a6fd82a224a822781e7a121c15e4fcc29c8b71c32b4cae4"
    ),
    "source/geniesim/assets/robot/curobo_robot/assets/robot/G2/"
    "G2_omnipicker_fixed_dual.urdf": (
        # arm_[lr]_end_joint is aligned to the composed runtime USD and its
        # source URDF at 0.088 m.  Both gripper center joints are likewise
        # aligned to the runtime TCP authority at 0.14308 m; the former
        # planner-only 0.13008 m caused a 12.27 mm accumulated TCP error.
        "a5efa8eba603ee259ef502b2b371875764ecbf161776a46559fce654ada9af09"
    ),
    G2_SHARED_WRIST_STRUCTURAL_SRDF_RELATIVE_PATH: (
        G2_SHARED_WRIST_STRUCTURAL_SRDF_SHA256
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_collision_pair_sha256(
    pairs: Sequence[tuple[str, str]],
) -> str:
    payload = json.dumps(
        [list(pair) for pair in pairs],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _collision_side(link_name: str) -> str | None:
    if link_name.startswith(("arm_l_", "gripper_l_")):
        return "left"
    if link_name.startswith(("arm_r_", "gripper_r_")):
        return "right"
    return None


_G2_BODY_COLLISION_LINK_NAMES = frozenset(
    {
        "base_link",
        *(f"body_link{index}" for index in range(1, 6)),
        *(f"head_link{index}" for index in range(1, 4)),
    }
)
_G2_ARM7_GRIPPER_BASE_BRIDGE_PAIRS = frozenset(
    {
        tuple(sorted((f"arm_{side}_link7", f"gripper_{side}_base_link")))
        for side in ("l", "r")
    }
)


def _filtered_pair_safety_counts(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, int]:
    """Count filters that would hide a prompt-required physical contact."""

    structural_mount_arm_body = 0
    nonstructural_arm_body = 0
    structural_gripper_joint = 0
    development_gripper_mechanism = 0
    unapproved_intra_gripper = 0
    arm_gripper = 0
    gripper_body = 0
    cross_arm = 0
    cross_side_gripper = 0
    arm7_gripper_base = 0
    for first, second in pairs:
        canonical = tuple(sorted((first, second)))
        first_arm = first.startswith("arm_")
        second_arm = second.startswith("arm_")
        first_gripper = first.startswith("gripper_")
        second_gripper = second.startswith("gripper_")
        arm_body_pair = (
            (first_arm and second in _G2_BODY_COLLISION_LINK_NAMES)
            or (second_arm and first in _G2_BODY_COLLISION_LINK_NAMES)
        )
        if arm_body_pair:
            if canonical in G2_FIXED_CHAIN_STRUCTURAL_FILTER_PAIRS:
                structural_mount_arm_body += 1
            else:
                nonstructural_arm_body += 1
        same_side_gripper_pair = (
            first_gripper
            and second_gripper
            and _collision_side(first) == _collision_side(second)
        )
        if same_side_gripper_pair:
            if canonical in _G2_OMNIPICKER_ARTICULATION_JOINT_FILTER_PAIRS:
                structural_gripper_joint += 1
            elif canonical in set(
                G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS
            ):
                development_gripper_mechanism += 1
            else:
                unapproved_intra_gripper += 1
        if (first_arm and second_gripper) or (
            second_arm and first_gripper
        ):
            arm_gripper += 1
        if (
            first_gripper and second in _G2_BODY_COLLISION_LINK_NAMES
        ) or (
            second_gripper and first in _G2_BODY_COLLISION_LINK_NAMES
        ):
            gripper_body += 1
        sides = {_collision_side(first), _collision_side(second)}
        if sides == {"left", "right"}:
            cross_arm += 1
            if first_gripper and second_gripper:
                cross_side_gripper += 1
        if canonical in _G2_ARM7_GRIPPER_BASE_BRIDGE_PAIRS:
            arm7_gripper_base += 1
    return {
        "structural_mount_arm_body_pairs_filtered": (
            structural_mount_arm_body
        ),
        "nonstructural_arm_body_pairs_filtered": nonstructural_arm_body,
        "structural_gripper_joint_pairs_filtered": structural_gripper_joint,
        "development_gripper_mechanism_pairs_filtered": (
            development_gripper_mechanism
        ),
        "unapproved_intra_gripper_pairs_filtered": (
            unapproved_intra_gripper
        ),
        "arm_gripper_pairs_filtered": arm_gripper,
        "active_gripper_body_pairs_filtered": gripper_body,
        "cross_arm_pairs_filtered": cross_arm,
        "cross_side_gripper_pairs_filtered": cross_side_gripper,
        "arm7_gripper_base_pairs_filtered": arm7_gripper_base,
        "environment_pairs_filtered": 0,
    }


def _assert_filter_safety(
    safety: Mapping[str, int],
    *,
    expected_structural_mount_arm_body_pairs: int,
    expected_structural_gripper_joint_pairs: int,
    expected_development_gripper_mechanism_pairs: int,
    expected_development_arm_gripper_pairs: int = 0,
    message: str,
) -> None:
    expected = {
        "structural_mount_arm_body_pairs_filtered": (
            expected_structural_mount_arm_body_pairs
        ),
        "structural_gripper_joint_pairs_filtered": (
            expected_structural_gripper_joint_pairs
        ),
        "development_gripper_mechanism_pairs_filtered": (
            expected_development_gripper_mechanism_pairs
        ),
        "arm_gripper_pairs_filtered": expected_development_arm_gripper_pairs,
    }
    for name, value in safety.items():
        if name in expected:
            if value != expected[name]:
                raise Stage2IsaacSceneError(message)
        elif value != 0:
            raise Stage2IsaacSceneError(message)


def attest_g2_structural_filter_policy() -> dict[str, Any]:
    """Attest the exact mechanically structural collision-filter authority."""

    pairs = tuple(G2_STRUCTURAL_SELF_COLLISION_FILTER_PAIRS)
    actual_links = frozenset(G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY)
    if any(
        len(pair) != 2
        or pair[0] >= pair[1]
        or pair[0] not in actual_links
        or pair[1] not in actual_links
        for pair in pairs
    ) or len(pairs) != len(set(pairs)):
        raise Stage2IsaacSceneError(
            "G2 structural filtered pairs are not canonical actual-link pairs"
        )
    if (
        len(pairs) != G2_SELF_COLLISION_FILTER_PAIR_COUNT
        or _canonical_collision_pair_sha256(pairs)
        != G2_SELF_COLLISION_FILTER_PAIRS_SHA256
    ):
        raise Stage2IsaacSceneError(
            "G2 structural filtered pairs differ from the closed authority"
        )
    safety = _filtered_pair_safety_counts(pairs)
    _assert_filter_safety(
        safety,
        expected_structural_mount_arm_body_pairs=2,
        expected_structural_gripper_joint_pairs=20,
        expected_development_gripper_mechanism_pairs=0,
        message="G2 structural filters suppress prompt-required contact pairs",
    )
    return {
        "schema_version": 4,
        "source_authority": (
            "explicit_closed_active_collider_articulation_plus_pinned_SRDF_"
            "Adjacent_or_Always_structural_housing_pairs"
        ),
        "actual_usd_body_count": len(
            G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY
        ),
        "filtered_pair_count": len(pairs),
        "filtered_pair_sha256": _canonical_collision_pair_sha256(pairs),
        "filtered_link_pairs": [list(pair) for pair in pairs],
        "urdf_direct_active_collider_pair_count": len(
            _G2_URDF_DIRECT_ACTIVE_COLLIDER_FILTER_PAIRS
        ),
        "usd_loop_joint_pair_count": len(
            G2_USD_LOOP_STRUCTURAL_FILTER_PAIRS
        ),
        "fixed_chain_arm_root_attachment_pair_count": len(
            G2_FIXED_CHAIN_STRUCTURAL_FILTER_PAIRS
        ),
        "srdf_shared_wrist_housing_pair_count": len(
            G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS
        ),
        "srdf_shared_wrist_housing_pairs": [
            list(pair)
            for pair in sorted(
                G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS
            )
        ],
        "srdf_body_head_housing_pair_count": len(
            G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIRS
        ),
        "srdf_body_head_housing_pair_reasons": {
            "|".join(pair): reason
            for pair, reason in sorted(
                G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIR_REASONS.items()
            )
        },
        "srdf_shared_wrist_source_sha256": (
            G2_SHARED_WRIST_STRUCTURAL_SRDF_SHA256
        ),
        "official_drawing_corroboration": {
            "archive_relative_path": (
                G2_OFFICIAL_DRAWING_ARCHIVE_RELATIVE_PATH
            ),
            "archive_sha256": G2_OFFICIAL_DRAWING_ARCHIVE_SHA256,
            "required_entry_sha256": dict(
                G2_OFFICIAL_SHARED_WRIST_ENTRY_SHA256
            ),
            "role": (
                "offline_corroboration_of_shared_wrist_housing_adjacency"
            ),
            "runtime_dependency": False,
            "runtime_mesh_replacement_authorized": False,
        },
        "structural_gripper_joint_pair_count": len(
            _G2_OMNIPICKER_ARTICULATION_JOINT_FILTER_PAIRS
        ),
        **safety,
    }


def attest_g2_development_runtime_filter_policy() -> dict[str, Any]:
    """Layer ten exact measured PhysX exemptions over the source policy.

    Only the ten closed-loop mechanism pairs remain development-only
    workarounds for measured overlap in checked-in collision meshes.  The two
    shared wrist-housing pairs are source-backed structural adjacency and the
    two former link5/gripper-base exemptions were removed after exact geometry
    established more than 41 mm clearance.
    """

    source_policy = attest_g2_structural_filter_policy()
    source_pairs = tuple(G2_STRUCTURAL_SELF_COLLISION_FILTER_PAIRS)
    home_exemptions = tuple(G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS)
    mechanism_exemptions = tuple(
        G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS
    )
    runtime_pairs = tuple(
        G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS
    )
    actual_links = frozenset(G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY)
    if (
        len(home_exemptions) != 0
        or len(home_exemptions) != len(set(home_exemptions))
        or any(
            len(pair) != 2
            or pair[0] >= pair[1]
            or pair[0] not in actual_links
            or pair[1] not in actual_links
            for pair in home_exemptions
        )
        or set(home_exemptions) & set(source_pairs)
        or _canonical_collision_pair_sha256(home_exemptions)
        != G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS_SHA256
    ):
        raise Stage2IsaacSceneError(
            "G2 development home-overlap exemption set differs"
        )
    if (
        len(mechanism_exemptions) != 10
        or len(mechanism_exemptions) != len(set(mechanism_exemptions))
        or any(
            len(pair) != 2
            or pair[0] >= pair[1]
            or pair[0] not in actual_links
            or pair[1] not in actual_links
            for pair in mechanism_exemptions
        )
        or set(mechanism_exemptions) & set(source_pairs)
        or set(mechanism_exemptions) & set(home_exemptions)
        or _canonical_collision_pair_sha256(mechanism_exemptions)
        != G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS_SHA256
    ):
        raise Stage2IsaacSceneError(
            "G2 development gripper-mechanism exemption set differs"
        )
    if (
        set(runtime_pairs)
        != (
            set(source_pairs)
            | set(home_exemptions)
            | set(mechanism_exemptions)
        )
        or len(runtime_pairs)
        != G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIR_COUNT
        or _canonical_collision_pair_sha256(runtime_pairs)
        != G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS_SHA256
    ):
        raise Stage2IsaacSceneError(
            "G2 development runtime filtered pairs differ from source plus "
            "the exact development exemptions"
        )
    safety = _filtered_pair_safety_counts(runtime_pairs)
    _assert_filter_safety(
        safety,
        expected_structural_mount_arm_body_pairs=2,
        expected_structural_gripper_joint_pairs=20,
        expected_development_gripper_mechanism_pairs=10,
        expected_development_arm_gripper_pairs=0,
        message=(
            "G2 development runtime filters suppress prompt-required contacts"
        ),
    )
    home_exemption = {
        "schema_version": 2,
        "scope": "development_physx_asset_home_overlap_none",
        "exact_unordered_link_pairs": [
            list(pair) for pair in home_exemptions
        ],
        "pair_count": len(home_exemptions),
        "pair_sha256": _canonical_collision_pair_sha256(home_exemptions),
        "reason": G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_REASON,
        "removed_structural_pairs": [
            list(pair)
            for pair in sorted(
                G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS
            )
        ],
        "removed_nonstructural_pairs": [
            ["arm_l_link5", "gripper_l_base_link"],
            ["arm_r_link5", "gripper_r_base_link"],
        ],
        "minimum_measured_link5_gripper_base_clearance_m": (
            0.04137592439169478
        ),
        "development_non_acceptance": False,
        "acceptance_evidence_eligible": True,
    }
    mechanism_exemption = {
        "schema_version": 1,
        "scope": "development_physx_gripper_mechanism_overlap_only",
        "exact_unordered_link_pairs": [
            list(pair) for pair in mechanism_exemptions
        ],
        "pair_count": len(mechanism_exemptions),
        "pair_sha256": _canonical_collision_pair_sha256(
            mechanism_exemptions
        ),
        "reason": G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_REASON,
        "source_observation_outputs": [
            (
                "output/level1/"
                "smoke_collision_authority_seed260_8t_20260828_v15"
            ),
            (
                "output/level1/"
                "smoke_current_minimal_filters_seed442_64t_20260828_v2"
            ),
            (
                "output/level1/"
                "train_current_2000_seed42_20260828_v2"
            ),
        ],
        "development_non_acceptance": True,
        "acceptance_evidence_eligible": False,
    }
    return {
        "schema_version": 4,
        "source_policy": source_policy,
        "source_policy_pair_count": source_policy["filtered_pair_count"],
        "source_policy_pair_sha256": source_policy["filtered_pair_sha256"],
        "runtime_authoring": {
            "authority": (
                "source_policy_plus_ten_exact_development_PhysX_gripper_"
                "mechanism_overlap_exemptions"
            ),
            "pair_count": len(runtime_pairs),
            "pair_sha256": _canonical_collision_pair_sha256(runtime_pairs),
            "filtered_link_pairs": [list(pair) for pair in runtime_pairs],
        },
        # Compatibility key consumed by the USD authoring path.  Its runtime
        # role is explicit; the sealed source pair fields above remain 44.
        "filtered_link_pairs": [list(pair) for pair in runtime_pairs],
        "development_home_overlap_exemption": home_exemption,
        "development_gripper_mechanism_exemption": mechanism_exemption,
        "development_exemption_pair_count": (
            len(home_exemptions) + len(mechanism_exemptions)
        ),
        "development_non_acceptance": True,
        "acceptance_evidence_eligible": False,
        **safety,
    }


def _expected_filtered_pair_relationships(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for owner, target in pairs:
        result.setdefault(owner, []).append(target)
    return {
        owner: tuple(sorted(targets))
        for owner, targets in sorted(result.items())
    }


def attest_g2_filtered_pair_relationships(
    *,
    expected_pairs: Sequence[tuple[str, str]],
    measured_relationships: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Prove the USD relationship graph equals the canonical pair set."""

    expected = tuple(expected_pairs)
    actual_links = frozenset(G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY)
    if any(
        len(pair) != 2
        or pair[0] >= pair[1]
        or pair[0] not in actual_links
        or pair[1] not in actual_links
        for pair in expected
    ) or len(expected) != len(set(expected)):
        raise Stage2IsaacSceneError(
            "expected G2 filtered pairs are not canonical actual-link pairs"
        )
    canonical_expected = tuple(sorted(expected))
    expected_sha256 = _canonical_collision_pair_sha256(canonical_expected)
    source_authority = (
        len(canonical_expected) == G2_SELF_COLLISION_FILTER_PAIR_COUNT
        and expected_sha256 == G2_SELF_COLLISION_FILTER_PAIRS_SHA256
    )
    runtime_authority = (
        len(canonical_expected)
        == G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIR_COUNT
        and expected_sha256
        == G2_DEVELOPMENT_RUNTIME_SELF_COLLISION_FILTER_PAIRS_SHA256
    )
    if not source_authority and not runtime_authority:
        raise Stage2IsaacSceneError(
            "expected G2 filtered pairs differ from the audited authority"
        )
    if runtime_authority and (
        set(canonical_expected)
        - set(G2_STRUCTURAL_SELF_COLLISION_FILTER_PAIRS)
        != (
            set(G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS)
            | set(G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS)
        )
    ):
        raise Stage2IsaacSceneError(
            "G2 runtime filtered-pair delta differs from the exact development "
            "exemptions"
        )
    measured: dict[str, tuple[str, ...]] = {}
    for owner, targets in measured_relationships.items():
        if owner not in actual_links:
            raise Stage2IsaacSceneError(
                f"filtered-pair owner is not an actual G2 body: {owner}"
            )
        if (
            not isinstance(targets, Sequence)
            or isinstance(targets, (str, bytes))
            or any(not isinstance(target, str) for target in targets)
        ):
            raise Stage2IsaacSceneError(
                f"filtered-pair targets for {owner} must be a string list"
            )
        normalized_targets = tuple(sorted(targets))
        if len(normalized_targets) != len(set(normalized_targets)):
            raise Stage2IsaacSceneError(
                f"filtered-pair targets for {owner} contain duplicates"
            )
        if any(target not in actual_links for target in normalized_targets):
            raise Stage2IsaacSceneError(
                f"filtered-pair targets for {owner} leave the G2 body set"
            )
        measured[owner] = normalized_targets
    measured_pairs = tuple(
        sorted(
            {
                tuple(sorted((owner, target)))
                for owner, targets in measured.items()
                for target in targets
            }
        )
    )
    safety = _filtered_pair_safety_counts(measured_pairs)
    _assert_filter_safety(
        safety,
        expected_structural_mount_arm_body_pairs=2,
        expected_structural_gripper_joint_pairs=20,
        expected_development_gripper_mechanism_pairs=(
            10 if runtime_authority else 0
        ),
        expected_development_arm_gripper_pairs=0,
        message=(
            "authored G2 filters suppress prompt-required contact pairs"
        ),
    )
    expected_relationships = _expected_filtered_pair_relationships(
        canonical_expected
    )
    if measured != expected_relationships:
        raise Stage2IsaacSceneError(
            "authored G2 filtered-pair relationships differ from the exact "
            "audited set"
        )
    return {
        "status": "PASS",
        "authority_kind": (
            "structural_source_policy"
            if source_authority
            else "development_runtime_authoring"
        ),
        "relationship_owner_count": len(measured),
        "filtered_pair_count": len(canonical_expected),
        "filtered_pair_sha256": expected_sha256,
        "source_policy_pair_count": G2_SELF_COLLISION_FILTER_PAIR_COUNT,
        "source_policy_pair_sha256": G2_SELF_COLLISION_FILTER_PAIRS_SHA256,
        "runtime_authoring_pair_count": len(canonical_expected),
        "runtime_authoring_pair_sha256": expected_sha256,
        "development_home_overlap_exemption": (
            {
                "exact_unordered_link_pairs": [
                    list(pair)
                    for pair in G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS
                ],
                "pair_count": len(
                    G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS
                ),
                "pair_sha256": (
                    G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_PAIRS_SHA256
                ),
                "reason": G2_DEVELOPMENT_HOME_OVERLAP_EXEMPTION_REASON,
                "development_non_acceptance": False,
                "acceptance_evidence_eligible": True,
            }
            if runtime_authority
            else None
        ),
        "development_gripper_mechanism_exemption": (
            {
                "exact_unordered_link_pairs": [
                    list(pair)
                    for pair in (
                        G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS
                    )
                ],
                "pair_count": len(
                    G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS
                ),
                "pair_sha256": (
                    G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_PAIRS_SHA256
                ),
                "reason": (
                    G2_DEVELOPMENT_GRIPPER_MECHANISM_EXEMPTION_REASON
                ),
                "development_non_acceptance": True,
                "acceptance_evidence_eligible": False,
            }
            if runtime_authority
            else None
        ),
        "development_non_acceptance": runtime_authority,
        "acceptance_evidence_eligible": not runtime_authority,
        "exact_relationship_readback": True,
        "does_not_disable_articulation_self_collision": True,
        **safety,
    }


def _zero_urdf_origin(collision: ET.Element) -> tuple[str, str]:
    origin = collision.find("origin")
    xyz = "0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0")
    rpy = "0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0")
    try:
        values = tuple(float(value) for value in f"{xyz} {rpy}".split())
    except ValueError as exc:
        raise Stage2IsaacSceneError(
            "G2 gripper-base URDF collision origin is not numeric"
        ) from exc
    if len(values) != 6 or not all(
        math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        for value in values
    ):
        raise Stage2IsaacSceneError(
            "G2 gripper-base URDF collision origins must both be identity"
        )
    return xyz, rpy


def _attest_g2_collision_urdf(urdf_path: Path) -> dict[str, Any]:
    try:
        robot = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise Stage2IsaacSceneError(
            f"failed to parse checked-in G2 collision URDF: {urdf_path}"
        ) from exc
    if robot.tag != "robot" or robot.attrib.get("name") != "genie":
        raise Stage2IsaacSceneError(
            "checked-in G2 collision URDF must define robot name 'genie'"
        )
    links = {link.attrib.get("name"): link for link in robot.findall("link")}
    if len(links) != len(robot.findall("link")):
        raise Stage2IsaacSceneError(
            "checked-in G2 collision URDF contains duplicate link names"
        )
    active_links = frozenset(G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY)
    urdf_structural_pairs: set[tuple[str, str]] = set()
    for joint in robot.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        parent = (
            None
            if parent_element is None
            else parent_element.attrib.get("link")
        )
        child = (
            None
            if child_element is None
            else child_element.attrib.get("link")
        )
        if parent is None or child is None:
            raise Stage2IsaacSceneError(
                "checked-in G2 collision URDF joint lacks parent or child"
            )
        if parent in active_links and child in active_links:
            urdf_structural_pairs.add(tuple(sorted((parent, child))))
    expected_urdf_pairs = set(
        _G2_URDF_DIRECT_ACTIVE_COLLIDER_FILTER_PAIRS
    )
    if urdf_structural_pairs != expected_urdf_pairs:
        raise Stage2IsaacSceneError(
            "checked-in G2 URDF active-collider joint pairs differ from the "
            "closed structural filter authority"
        )
    for entry in G2_ROBOT_COLLISION_ALLOWLIST:
        link = links.get(entry.link_name)
        if link is None:
            raise Stage2IsaacSceneError(
                f"G2 collision URDF link is missing: {entry.link_name}"
            )
        collisions = [
            collision
            for collision in link.findall("collision")
            if collision.attrib.get("name") == entry.collision_name
        ]
        if len(collisions) != 1:
            raise Stage2IsaacSceneError(
                "G2 collision URDF must contain exactly one selected collision "
                f"{entry.link_name}/{entry.collision_name}"
            )
        geometry = collisions[0].find("geometry")
        children = [] if geometry is None else list(geometry)
        if len(children) != 1 or children[0].tag != entry.urdf_geometry_kind:
            raise Stage2IsaacSceneError(
                "G2 collision URDF geometry differs for "
                f"{entry.link_name}/{entry.collision_name}"
            )
        if entry.urdf_mesh_uri is not None and (
            children[0].attrib.get("filename") != entry.urdf_mesh_uri
        ):
            raise Stage2IsaacSceneError(
                "G2 collision URDF mesh URI differs for "
                f"{entry.link_name}/{entry.collision_name}"
            )

    gripper_sources: dict[str, dict[str, str]] = {}
    for side in ("l", "r"):
        link_name = f"gripper_{side}_base_link"
        collision_name = f"gripper_{side}_base_link_convex_0"
        link = links.get(link_name)
        collisions = (
            []
            if link is None
            else [
                collision
                for collision in link.findall("collision")
                if collision.attrib.get("name") == collision_name
            ]
        )
        if len(collisions) != 1:
            raise Stage2IsaacSceneError(
                f"G2 collision URDF lacks exact {collision_name}"
            )
        geometry = collisions[0].find("geometry")
        mesh = None if geometry is None else geometry.find("mesh")
        uri = None if mesh is None else mesh.attrib.get("filename")
        if uri != G2_GRIPPER_BASE_COLLISION_MESH_URI:
            raise Stage2IsaacSceneError(
                f"G2 {link_name} collision mesh URI is not the approved asset"
            )
        xyz, rpy = _zero_urdf_origin(collisions[0])
        gripper_sources[side] = {"mesh_uri": uri, "xyz": xyz, "rpy": rpy}
    if gripper_sources["l"] != gripper_sources["r"]:
        raise Stage2IsaacSceneError(
            "left/right G2 gripper-base URDF collision sources differ"
        )
    return {
        "robot_name": "genie",
        "allowlisted_collision_count": len(G2_ROBOT_COLLISION_ALLOWLIST),
        "allowlisted_collision_prim_paths": [
            entry.prim_path for entry in G2_ROBOT_COLLISION_ALLOWLIST
        ],
        "gripper_base_collision_source": gripper_sources["l"],
        "left_right_gripper_base_source_identical": True,
        "direct_active_collider_joint_pair_count": len(
            urdf_structural_pairs
        ),
        "direct_active_collider_joint_pairs": [
            list(pair) for pair in sorted(urdf_structural_pairs)
        ],
    }


def _attest_g2_shared_wrist_structural_srdf(
    srdf_path: Path,
) -> dict[str, Any]:
    """Attest the exact SRDF-backed wrist and torso/head housing pairs."""

    try:
        robot = ET.parse(srdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise Stage2IsaacSceneError(
            f"failed to parse checked-in G2 collision SRDF: {srdf_path}"
        ) from exc
    if robot.tag != "robot" or robot.attrib.get("name") != "genie":
        raise Stage2IsaacSceneError(
            "checked-in G2 collision SRDF must define robot name 'genie'"
        )
    entries: list[tuple[tuple[str, str], str]] = []
    for element in robot.findall("disable_collisions"):
        link1 = element.attrib.get("link1")
        link2 = element.attrib.get("link2")
        reason = element.attrib.get("reason")
        if not link1 or not link2 or not reason:
            raise Stage2IsaacSceneError(
                "checked-in G2 SRDF disable_collisions entry is incomplete"
            )
        entries.append((tuple(sorted((link1, link2))), reason))
    if len(entries) != len({(pair, reason) for pair, reason in entries}):
        raise Stage2IsaacSceneError(
            "checked-in G2 SRDF contains duplicate collision entries"
        )

    expected_wrist = set(G2_SRDF_SHARED_WRIST_HOUSING_STRUCTURAL_FILTER_PAIRS)
    expected_body_head = dict(
        G2_SRDF_BODY_HEAD_HOUSING_STRUCTURAL_FILTER_PAIR_REASONS
    )
    entry_reasons = dict(entries)
    adjacent = {pair for pair, reason in entries if reason == "Adjacent"}
    defaults = {pair for pair, reason in entries if reason == "Default"}
    if not expected_wrist <= adjacent:
        raise Stage2IsaacSceneError(
            "checked-in G2 SRDF no longer marks both shared wrist housings "
            "Adjacent"
        )
    if any(entry_reasons.get(pair) != reason for pair, reason in expected_body_head.items()):
        raise Stage2IsaacSceneError(
            "checked-in G2 SRDF torso/head structural reasons differ"
        )
    expected = expected_wrist | set(expected_body_head)
    structural = set(G2_STRUCTURAL_SELF_COLLISION_FILTER_PAIRS)
    if expected - structural:
        raise Stage2IsaacSceneError(
            "audited SRDF shared wrist pairs are absent from structural filters"
        )
    imported_from_srdf = structural & {
        pair for pair, reason in entries if reason in {"Adjacent", "Always"}
    }
    # No broad SRDF Default entry may enter the closed filter.
    if defaults & structural:
        raise Stage2IsaacSceneError(
            "a broad SRDF Default pair leaked into structural collision filters"
        )
    return {
        "schema_version": 2,
        "status": "PASS",
        "source_relative_path": (
            G2_SHARED_WRIST_STRUCTURAL_SRDF_RELATIVE_PATH
        ),
        "source_sha256": G2_SHARED_WRIST_STRUCTURAL_SRDF_SHA256,
        "authorized_reasons": ["Adjacent", "Always"],
        "authorized_pair_count": len(expected),
        "authorized_pairs": [list(pair) for pair in sorted(expected)],
        "authorized_pair_reasons": {
            "|".join(pair): entry_reasons[pair] for pair in sorted(expected)
        },
        "srdf_adjacent_entry_count": len(adjacent),
        "srdf_default_entry_count": len(defaults),
        "broad_default_pairs_imported": 0,
        "all_structural_pairs_also_present_in_srdf_count": len(
            imported_from_srdf
        ),
    }


def attest_checked_in_g2_collision_sources(
    build: "IsaacSceneBuildConfig",
) -> dict[str, Any]:
    """Pin and parse every checked-in file that composes collision geometry."""

    root = build.repository_root.resolve()
    expected_asset_relative = Path(
        "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
    )
    expected_urdf_relative = Path(
        "source/geniesim/assets/robot/curobo_robot/assets/robot/G2/"
        "G2_omnipicker_fixed_dual.urdf"
    )
    if build.robot_asset_relative != expected_asset_relative:
        raise Stage2IsaacSceneError(
            "G2 collision manifest only authorizes the checked-in robot_fix.usda"
        )
    if build.robot_urdf_relative != expected_urdf_relative:
        raise Stage2IsaacSceneError(
            "G2 collision manifest only authorizes the checked-in dual-arm URDF"
        )
    measured: dict[str, str] = {}
    for relative_text, expected_sha256 in _G2_COLLISION_SOURCE_SHA256.items():
        candidate = root / relative_text
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise Stage2IsaacSceneError(
                f"G2 collision source escapes repository root: {candidate}"
            ) from exc
        if candidate.is_symlink() or not resolved.is_file():
            raise Stage2IsaacSceneError(
                f"G2 collision source must be one regular file: {candidate}"
            )
        sha256 = _sha256_file(resolved)
        if sha256 != expected_sha256:
            raise Stage2IsaacSceneError(
                "G2 collision source hash differs from the audited manifest: "
                f"{relative_text}"
            )
        measured[relative_text] = sha256
    urdf = _attest_g2_collision_urdf(root / expected_urdf_relative)
    shared_wrist_srdf = _attest_g2_shared_wrist_structural_srdf(
        root / G2_SHARED_WRIST_STRUCTURAL_SRDF_RELATIVE_PATH
    )
    structural_filter = attest_g2_structural_filter_policy()
    development_runtime_filter = attest_g2_development_runtime_filter_policy()
    return {
        "schema_version": 4,
        "source_sha256": measured,
        "urdf": urdf,
        "shared_wrist_structural_srdf": shared_wrist_srdf,
        "structural_self_collision_filter": structural_filter,
        "development_runtime_self_collision_filter": (
            development_runtime_filter
        ),
        "arbitrary_visual_mesh_activation_allowed": False,
        "right_gripper_base_authority": (
            "exact_known_instance_proxy_same_urdf_mesh_uri_identity_origin"
        ),
    }


def attest_robot_collision_activation_delta(
    *,
    before: Mapping[str, bool],
    after: Mapping[str, bool],
    self_collision_before: bool,
    self_collision_after: bool,
) -> dict[str, Any]:
    """Validate the exact USD collision-state delta without importing Isaac."""

    allowlisted = {entry.prim_path for entry in G2_ROBOT_COLLISION_ALLOWLIST}
    preexisting = set(G2_EXPECTED_PREEXISTING_ENABLED_COLLISION_PATHS)
    if {path for path, enabled in before.items() if enabled} != preexisting:
        raise Stage2IsaacSceneError(
            "unexpected preexisting enabled G2 collision geometry"
        )
    missing_disabled = [path for path in allowlisted if before.get(path) is not False]
    if missing_disabled:
        raise Stage2IsaacSceneError(
            "G2 collision allowlist was not present and disabled before activation: "
            f"{sorted(missing_disabled)}"
        )
    expected_enabled = preexisting | allowlisted
    measured_enabled = {path for path, enabled in after.items() if enabled}
    if measured_enabled != expected_enabled:
        raise Stage2IsaacSceneError(
            "post-activation G2 collision geometry differs from the closed manifest"
        )
    changed = {
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    }
    expected_changed = allowlisted
    if changed != expected_changed:
        raise Stage2IsaacSceneError(
            "G2 collision activation authored an unexpected USD delta"
        )
    activated_visuals = sorted(path for path in changed if "/visuals/" in path)
    if activated_visuals:
        raise Stage2IsaacSceneError(
            f"G2 collision activation touched visual meshes: {activated_visuals}"
        )
    if self_collision_before or not self_collision_after:
        raise Stage2IsaacSceneError(
            "G2 articulation self-collision must change exactly false -> true"
        )
    return {
        "allowlisted_collision_prim_count": len(allowlisted),
        "activated_collision_prim_count": len(changed),
        "activated_collision_prim_paths": sorted(changed),
        "preexisting_enabled_collision_prim_count": len(preexisting),
        "post_enabled_collision_prim_count": len(expected_enabled),
        "visual_collision_prim_paths_activated": [],
        "right_gripper_base_instance_proxy_preexisting_and_unchanged": True,
        "articulation_self_collision_enabled": True,
    }


def attest_fixed_scene_identity_rotations(
    *,
    robot_base_world_rotation: Any,
    table_world_rotation: Any,
) -> dict[str, list[list[float]]]:
    """Fail fast unless the fixed base and tabletop use the audited frame.

    The current task's table-coordinate computations intentionally assume the
    fixed, axis-aligned scene.  Making that assumption executable prevents a
    silent yaw/tilt from corrupting every table-frame cube measurement.
    """

    result: dict[str, list[list[float]]] = {}
    for name, value in (
        ("robot_base_world_rotation", robot_base_world_rotation),
        ("table_world_rotation", table_world_rotation),
    ):
        rotation = np.asarray(value, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise Stage2IsaacSceneError(
                f"{name} must be one finite 3x3 matrix"
            )
        if not np.allclose(
            rotation, np.eye(3), atol=1.0e-6, rtol=0.0
        ):
            raise Stage2IsaacSceneError(
                f"{name} must remain identity in the fixed Stage-2 scene"
            )
        result[name] = rotation.tolist()
    return result


def task_head_camera_orientation_wxyz(
    authored_orientation_wxyz: Any,
) -> np.ndarray:
    """Pitch the simulated head sensor toward the compact tabletop.

    The repository camera mount looks only about 24 degrees below horizontal
    when the planner-locked head joints are zero.  The requested 40 x 20 cm
    task surface is roughly 48 degrees farther down, outside that camera's
    vertical field of view.  Rotate the *sensor calibration frame* in its
    parent frame instead of moving a planner-locked robot joint.
    """

    authored = np.asarray(authored_orientation_wxyz, dtype=np.float64)
    if authored.shape != (4,) or not np.all(np.isfinite(authored)):
        raise Stage2IsaacSceneError(
            "authored head-camera orientation must be one finite quaternion"
        )
    norm = float(np.linalg.norm(authored))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise Stage2IsaacSceneError(
            "authored head-camera orientation must be unit length"
        )
    half = TASK_HEAD_CAMERA_DOWNWARD_PITCH_RAD / 2.0
    pitch = np.asarray(
        [math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64
    )
    lw, lx, ly, lz = pitch
    rw, rx, ry, rz = authored / norm
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / float(np.linalg.norm(result))


@dataclass(frozen=True)
class IsaacSceneBuildConfig:
    repository_root: Path
    robot_asset_relative: Path = Path(
        "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
    )
    robot_urdf_relative: Path = Path(
        "source/geniesim/assets/robot/curobo_robot/assets/robot/G2/"
        "G2_omnipicker_fixed_dual.urdf"
    )
    headless: bool = True
    asset_load_timeout_s: float = 120.0
    cube_mass_kg: float = CUBE_MASS_NOMINAL_KG
    cube_color_rgb: tuple[float, float, float] = (0.82, 0.12, 0.08)

    def validated(self) -> "IsaacSceneBuildConfig":
        root = self.repository_root.resolve()
        asset = (root / self.robot_asset_relative).resolve()
        if not asset.is_file():
            raise Stage2IsaacSceneError(f"G2 OmniPicker asset is missing: {asset}")
        urdf = (root / self.robot_urdf_relative).resolve()
        if not urdf.is_file():
            raise Stage2IsaacSceneError(
                f"G2 OmniPicker collision URDF is missing: {urdf}"
            )
        if not math.isfinite(self.asset_load_timeout_s) or self.asset_load_timeout_s <= 0.0:
            raise Stage2IsaacSceneError("asset load timeout must be positive")
        try:
            validate_authored_cube_mass_kg(self.cube_mass_kg)
        except GeometryContractError as exc:
            raise Stage2IsaacSceneError(str(exc)) from exc
        color = np.asarray(self.cube_color_rgb, dtype=np.float64)
        if color.shape != (3,) or not np.all(np.isfinite(color)) or np.any(color < 0.0) or np.any(color > 1.0):
            raise Stage2IsaacSceneError("cube color must contain three values in [0, 1]")
        attest_checked_in_g2_collision_sources(self)
        return self

    @property
    def robot_asset(self) -> Path:
        return (self.repository_root.resolve() / self.robot_asset_relative).resolve()

    @property
    def robot_urdf(self) -> Path:
        return (self.repository_root.resolve() / self.robot_urdf_relative).resolve()


class IsaacSimpleStage2Scene:
    """Compose and validate the exact G2 + table + free-cube USD stage."""

    def __init__(
        self,
        simulation_app: Any,
        *,
        scene: Stage2SimpleSceneConfig,
        build: IsaacSceneBuildConfig,
    ) -> None:
        scene.validated()
        build.validated()
        if "isaacsim" not in sys.modules:
            raise Stage2IsaacSceneError(
                "IsaacSimpleStage2Scene must be created after SimulationApp starts"
            )
        self.simulation_app = simulation_app
        self.scene_config = scene
        self.build_config = build
        self._load_modules()
        self._compose()
        self.validation = self._validate_composed_scene()

    def _load_modules(self) -> None:
        self._stage_utils = importlib.import_module("isaacsim.core.utils.stage")
        self._viewport_utils = importlib.import_module("isaacsim.core.utils.viewports")
        self._semantics_utils = importlib.import_module("isaacsim.core.utils.semantics")
        self._Gf = importlib.import_module("pxr.Gf")
        self._Sdf = importlib.import_module("pxr.Sdf")
        self._Usd = importlib.import_module("pxr.Usd")
        self._UsdGeom = importlib.import_module("pxr.UsdGeom")
        self._UsdLux = importlib.import_module("pxr.UsdLux")
        self._UsdPhysics = importlib.import_module("pxr.UsdPhysics")
        try:
            self._PhysxSchema = importlib.import_module("pxr.PhysxSchema")
        except ImportError as exc:
            raise Stage2IsaacSceneError("PhysxSchema is unavailable in Isaac Sim") from exc

    def _robot_collision_states(self) -> dict[str, bool]:
        """Read collision shapes, including known instance proxies, by path."""

        candidates: dict[str, Any] = {}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(f"{ROBOT_PRIM_PATH}/") and prim.HasAPI(
                self._UsdPhysics.CollisionAPI
            ):
                candidates[path] = prim
        for paths in G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY.values():
            for path in paths:
                prim = self.stage.GetPrimAtPath(path)
                if prim.IsValid() and prim.HasAPI(self._UsdPhysics.CollisionAPI):
                    candidates[path] = prim
        result: dict[str, bool] = {}
        for path, prim in candidates.items():
            attribute = prim.GetAttribute("physics:collisionEnabled")
            if not attribute.IsValid() or attribute.Get() not in (True, False):
                raise Stage2IsaacSceneError(
                    f"G2 collisionEnabled state is invalid: {path}"
                )
            result[path] = bool(attribute.Get())
        return result

    def _gripper_base_collision_payload(self, path: str) -> dict[str, Any]:
        prim = self.stage.GetPrimAtPath(path)
        if (
            not prim.IsValid()
            or not prim.IsActive()
            or prim.GetTypeName() != "Mesh"
            or not prim.HasAPI(self._UsdPhysics.CollisionAPI)
            or not prim.HasAPI(self._UsdPhysics.MeshCollisionAPI)
        ):
            raise Stage2IsaacSceneError(
                f"G2 gripper-base collision mesh is invalid: {path}"
            )
        payload: dict[str, Any] = {}
        for name in G2_COLLISION_MESH_EQUALITY_ATTRIBUTE_NAMES:
            attribute = prim.GetAttribute(name)
            value = None if not attribute.IsValid() else attribute.Get()
            if not attribute.IsValid() or value is None:
                raise Stage2IsaacSceneError(
                    f"G2 gripper-base collision attribute is missing: {path}/{name}"
                )
            payload[name] = {
                "type_name": str(attribute.GetTypeName()),
                "value": repr(value),
            }
        normals = prim.GetAttribute("normals")
        payload["normals_interpolation"] = repr(
            normals.GetMetadata("interpolation")
        )
        return payload

    def _attest_gripper_base_collision_instance_proxy(self) -> dict[str, Any]:
        left = self.stage.GetPrimAtPath(LEFT_GRIPPER_BASE_COLLISION_PRIM_PATH)
        right = self.stage.GetPrimAtPath(RIGHT_GRIPPER_BASE_COLLISION_PRIM_PATH)
        if not left.IsValid() or left.IsInstanceProxy():
            raise Stage2IsaacSceneError(
                "left G2 gripper-base collision source must be a concrete prim"
            )
        if not right.IsValid() or not right.IsInstanceProxy():
            raise Stage2IsaacSceneError(
                "right G2 gripper-base collision must be the audited instance proxy"
            )
        left_payload = self._gripper_base_collision_payload(
            LEFT_GRIPPER_BASE_COLLISION_PRIM_PATH
        )
        right_payload = self._gripper_base_collision_payload(
            RIGHT_GRIPPER_BASE_COLLISION_PRIM_PATH
        )
        if left_payload != right_payload:
            raise Stage2IsaacSceneError(
                "left/right G2 gripper-base collision geometry differs"
            )
        digest = hashlib.sha256(repr(left_payload).encode("utf-8")).hexdigest()
        return {
            "left_prim_path": LEFT_GRIPPER_BASE_COLLISION_PRIM_PATH,
            "right_prim_path": RIGHT_GRIPPER_BASE_COLLISION_PRIM_PATH,
            "right_is_instance_proxy": True,
            "deinstanced": False,
            "cloned": False,
            "geometry_topology_subdivision_approximation_equal": True,
            "geometry_payload_sha256": digest,
            "physics_approximation": "convexHull",
            "collision_enabled": True,
        }

    def _validate_collision_manifest_prims(
        self, *, expected_enabled: bool
    ) -> None:
        for entry in G2_ROBOT_COLLISION_ALLOWLIST:
            if "/visuals/" in entry.prim_path:
                raise Stage2IsaacSceneError(
                    "G2 collision activation manifest contains a visual mesh"
                )
            prim = self.stage.GetPrimAtPath(entry.prim_path)
            if (
                not prim.IsValid()
                or not prim.IsActive()
                or prim.GetTypeName() != entry.usd_type_name
                or not prim.HasAPI(self._UsdPhysics.CollisionAPI)
            ):
                raise Stage2IsaacSceneError(
                    f"G2 collision allowlist prim differs: {entry.prim_path}"
                )
            enabled = prim.GetAttribute("physics:collisionEnabled").Get()
            if enabled is not expected_enabled:
                raise Stage2IsaacSceneError(
                    "G2 collision allowlist prim has an unexpected enabled state: "
                    f"{entry.prim_path}"
                )
            if entry.usd_type_name == "Mesh":
                if not prim.HasAPI(self._UsdPhysics.MeshCollisionAPI):
                    raise Stage2IsaacSceneError(
                        f"G2 collision mesh lacks MeshCollisionAPI: {entry.prim_path}"
                    )
                approximation = prim.GetAttribute("physics:approximation").Get()
                if str(approximation) != "convexHull":
                    raise Stage2IsaacSceneError(
                        "G2 collision mesh must retain convexHull approximation: "
                        f"{entry.prim_path}"
                    )

    @staticmethod
    def _robot_body_path(link_name: str) -> str:
        return f"{ROBOT_PRIM_PATH}/{link_name}"

    @staticmethod
    def _robot_body_name(path: str) -> str:
        prefix = f"{ROBOT_PRIM_PATH}/"
        if not path.startswith(prefix) or "/" in path[len(prefix) :]:
            raise Stage2IsaacSceneError(
                f"filtered-pair target is not one direct G2 body: {path}"
            )
        return path[len(prefix) :]

    def _read_g2_filtered_pair_relationships(
        self,
    ) -> dict[str, tuple[str, ...]]:
        """Read every actual-body relationship, rejecting hidden extras."""

        result: dict[str, tuple[str, ...]] = {}
        for link_name in sorted(G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY):
            path = self._robot_body_path(link_name)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive():
                raise Stage2IsaacSceneError(
                    f"G2 filtered-pair body is missing or inactive: {path}"
                )
            if prim.IsInstanceProxy():
                raise Stage2IsaacSceneError(
                    "G2 FilteredPairsAPI must be authored on a body prim, not "
                    f"an instance proxy: {path}"
                )
            has_api = prim.HasAPI(self._UsdPhysics.FilteredPairsAPI)
            relationship = prim.GetRelationship("physics:filteredPairs")
            relationship_valid = relationship.IsValid()
            if has_api != relationship_valid:
                raise Stage2IsaacSceneError(
                    "G2 FilteredPairsAPI/relationship schema state differs: "
                    f"{path}"
                )
            if not has_api:
                continue
            targets = tuple(
                self._robot_body_name(str(target))
                for target in relationship.GetTargets()
            )
            if len(targets) != len(set(targets)):
                raise Stage2IsaacSceneError(
                    f"G2 filtered-pair relationship has duplicate targets: {path}"
                )
            for target in targets:
                target_prim = self.stage.GetPrimAtPath(
                    self._robot_body_path(target)
                )
                if not target_prim.IsValid() or not target_prim.IsActive():
                    raise Stage2IsaacSceneError(
                        "G2 filtered-pair target body is missing or inactive: "
                        f"{target}"
                    )
            result[link_name] = tuple(sorted(targets))
        return result

    def _author_and_attest_g2_filtered_pairs(
        self,
        policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        pairs_value = policy.get("filtered_link_pairs")
        if not isinstance(pairs_value, Sequence):
            raise Stage2IsaacSceneError(
                "G2 self-collision policy lacks filtered_link_pairs"
            )
        pairs = tuple(
            tuple(str(link_name) for link_name in pair)
            for pair in pairs_value
        )
        before = self._read_g2_filtered_pair_relationships()
        if before:
            raise Stage2IsaacSceneError(
                "G2 asset contains unexpected pre-authored filtered pairs"
            )
        expected_relationships = _expected_filtered_pair_relationships(pairs)
        for owner, targets in expected_relationships.items():
            owner_path = self._robot_body_path(owner)
            prim = self.stage.GetPrimAtPath(owner_path)
            if prim.IsInstanceProxy():
                raise Stage2IsaacSceneError(
                    "cannot author G2 FilteredPairsAPI on instance proxy body: "
                    f"{owner_path}"
                )
            api = self._UsdPhysics.FilteredPairsAPI.Apply(prim)
            if not prim.HasAPI(self._UsdPhysics.FilteredPairsAPI):
                raise Stage2IsaacSceneError(
                    f"failed to apply G2 FilteredPairsAPI: {owner_path}"
                )
            relationship = api.CreateFilteredPairsRel()
            target_paths = [
                self._Sdf.Path(self._robot_body_path(target))
                for target in targets
            ]
            if not relationship.SetTargets(target_paths):
                raise Stage2IsaacSceneError(
                    f"failed to author G2 filtered-pair targets: {owner_path}"
                )
        self._stage_utils.update_stage()
        measured = self._read_g2_filtered_pair_relationships()
        result = attest_g2_filtered_pair_relationships(
            expected_pairs=pairs,
            measured_relationships=measured,
        )
        return {
            **result,
            "schema_version": 3,
            "authoring_phase": (
                "before_simulation_context_initialize_physics"
            ),
            "relationship_authority": (
                "UsdPhysics.FilteredPairsAPI_on_direct_articulation_body_prims"
            ),
            "source_policy_authority": policy["source_policy"][
                "source_authority"
            ],
            "source_policy_pair_sha256": policy[
                "source_policy_pair_sha256"
            ],
            "source_policy_pair_count": policy[
                "source_policy_pair_count"
            ],
            "runtime_authoring_pair_sha256": policy["runtime_authoring"][
                "pair_sha256"
            ],
            "runtime_authoring_pair_count": policy["runtime_authoring"][
                "pair_count"
            ],
            "development_home_overlap_exemption": policy[
                "development_home_overlap_exemption"
            ],
            "development_gripper_mechanism_exemption": policy[
                "development_gripper_mechanism_exemption"
            ],
            "development_non_acceptance": True,
            "acceptance_evidence_eligible": False,
        }

    def _validate_g2_filtered_pairs(
        self,
        policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        pairs_value = policy.get("filtered_link_pairs")
        if not isinstance(pairs_value, Sequence):
            raise Stage2IsaacSceneError(
                "G2 self-collision policy lacks filtered_link_pairs"
            )
        pairs = tuple(
            tuple(str(link_name) for link_name in pair)
            for pair in pairs_value
        )
        result = attest_g2_filtered_pair_relationships(
            expected_pairs=pairs,
            measured_relationships=(
                self._read_g2_filtered_pair_relationships()
            ),
        )
        return {
            **result,
            "source_policy_pair_sha256": policy[
                "source_policy_pair_sha256"
            ],
            "source_policy_pair_count": policy[
                "source_policy_pair_count"
            ],
            "runtime_authoring_pair_sha256": policy["runtime_authoring"][
                "pair_sha256"
            ],
            "runtime_authoring_pair_count": policy["runtime_authoring"][
                "pair_count"
            ],
            "development_home_overlap_exemption": policy[
                "development_home_overlap_exemption"
            ],
            "development_gripper_mechanism_exemption": policy[
                "development_gripper_mechanism_exemption"
            ],
            "development_non_acceptance": True,
            "acceptance_evidence_eligible": False,
        }

    def _enable_and_attest_robot_collisions(self) -> dict[str, Any]:
        """Enable only the closed G2 allowlist before PhysX imports the stage."""

        source_attestation = attest_checked_in_g2_collision_sources(
            self.build_config
        )
        self._validate_collision_manifest_prims(expected_enabled=False)
        before = self._robot_collision_states()
        root = self.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        api_schema_list_op = root.GetMetadata("apiSchemas")
        authored_api_schemas = (
            ()
            if api_schema_list_op is None
            else tuple(
                str(schema) for schema in api_schema_list_op.GetAppliedItems()
            )
        )
        if "PhysxArticulationAPI" not in authored_api_schemas:
            raise Stage2IsaacSceneError(
                "G2 root lacks PhysxArticulationAPI for self-collision"
            )
        self_collision = root.GetAttribute(ARTICULATION_SELF_COLLISION_ATTRIBUTE)
        if not self_collision.IsValid() or self_collision.Get() is not False:
            raise Stage2IsaacSceneError(
                "G2 articulation self-collision must be authored false before enable"
            )
        filtered_pairs = self._author_and_attest_g2_filtered_pairs(
            source_attestation[
                "development_runtime_self_collision_filter"
            ]
        )
        gripper_proxy = self._attest_gripper_base_collision_instance_proxy()
        for entry in G2_ROBOT_COLLISION_ALLOWLIST:
            attribute = self.stage.GetPrimAtPath(entry.prim_path).GetAttribute(
                "physics:collisionEnabled"
            )
            if not attribute.Set(True):
                raise Stage2IsaacSceneError(
                    f"failed to enable G2 collision prim: {entry.prim_path}"
                )
        if not self_collision.Set(True):
            raise Stage2IsaacSceneError(
                "failed to enable G2 articulation self-collision"
            )
        self._stage_utils.update_stage()
        self._validate_collision_manifest_prims(expected_enabled=True)
        after = self._robot_collision_states()
        delta = attest_robot_collision_activation_delta(
            before=before,
            after=after,
            self_collision_before=False,
            self_collision_after=bool(self_collision.Get()),
        )
        self.validate_robot_collision_activation()
        return {
            "schema_version": 3,
            "activation_phase": "before_simulation_context_initialize_physics",
            "source_attestation": source_attestation,
            "activation_delta": delta,
            "self_collision_filtered_pairs": filtered_pairs,
            "required_active_collision_paths_by_body": {
                body: list(paths)
                for body, paths in G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY.items()
            },
            "required_robot_body_count": len(
                G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY
            ),
            "required_robot_collision_shape_count": sum(
                len(paths)
                for paths in G2_REQUIRED_ACTIVE_COLLISION_PATHS_BY_BODY.values()
            ),
            "right_gripper_base_instance_proxy": gripper_proxy,
            "articulation_self_collision_attribute": (
                ARTICULATION_SELF_COLLISION_ATTRIBUTE
            ),
        }

    def validate_robot_collision_activation(self) -> dict[str, Any]:
        """Re-attest the exact active shapes without authoring another delta."""

        source_attestation = attest_checked_in_g2_collision_sources(
            self.build_config
        )
        filtered_pairs = self._validate_g2_filtered_pairs(
            source_attestation[
                "development_runtime_self_collision_filter"
            ]
        )
        self._validate_collision_manifest_prims(expected_enabled=True)
        measured = self._robot_collision_states()
        expected = {
            entry.prim_path for entry in G2_ROBOT_COLLISION_ALLOWLIST
        } | set(G2_EXPECTED_PREEXISTING_ENABLED_COLLISION_PATHS)
        enabled = {path for path, value in measured.items() if value}
        if enabled != expected:
            raise Stage2IsaacSceneError(
                "active G2 collision shapes differ from the required manifest"
            )
        root = self.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        self_collision = root.GetAttribute(ARTICULATION_SELF_COLLISION_ATTRIBUTE)
        if not self_collision.IsValid() or self_collision.Get() is not True:
            raise Stage2IsaacSceneError(
                "G2 articulation self-collision is no longer enabled"
            )
        proxy = self._attest_gripper_base_collision_instance_proxy()
        return {
            "active_collision_shape_count": len(enabled),
            "active_collision_shape_paths": sorted(enabled),
            "articulation_self_collision_enabled": True,
            "self_collision_filtered_pairs": filtered_pairs,
            "right_gripper_base_instance_proxy": proxy,
        }

    def _reference_robot(self) -> Any:
        robot = self.stage.DefinePrim(ROBOT_PRIM_PATH, "Xform")
        reference = self._Sdf.Reference(
            assetPath=str(self.build_config.robot_asset),
            primPath=self._Sdf.Path(ROBOT_SOURCE_PRIM_PATH),
        )
        if not robot.GetReferences().AddReference(reference):
            raise Stage2IsaacSceneError("failed to reference G2 OmniPicker asset")
        for variant_name, selection in (
            ("Physics", "PhysX"),
            ("Robot", "Robot"),
            ("Sensor", "Sensors"),
        ):
            variant = robot.GetVariantSet(variant_name)
            if not variant.IsValid() or selection not in variant.GetVariantNames():
                raise Stage2IsaacSceneError(
                    f"G2 asset lacks {variant_name}={selection}"
                )
            variant.SetVariantSelection(selection)
            if variant.GetVariantSelection() != selection:
                raise Stage2IsaacSceneError(
                    f"failed to select {variant_name}={selection}"
                )
        # The source asset authors /genie at world z=+0.04 m.  Planner and
        # policy use /base_link as their metric origin, so locally override the
        # referenced root to place the actual fixed articulation base at world
        # z=0 instead of applying a hidden planner-frame offset.
        translate = robot.GetAttribute("xformOp:translate")
        if not translate.IsValid():
            raise Stage2IsaacSceneError("G2 root xformOp:translate is missing")
        translate.Set(self._Gf.Vec3d(0.0, 0.0, 0.0))
        return robot

    def _define_table(self) -> Any:
        # A unit cube scaled to the requested full table dimensions.  It is a
        # static collider (CollisionAPI without RigidBodyAPI), with its bottom
        # at z=0 and its usable surface at the configured height.
        table = self._UsdGeom.Cube.Define(self.stage, TABLE_PRIM_PATH)
        table.CreateSizeAttr(1.0)
        table.CreateExtentAttr(
            [self._Gf.Vec3f(-0.5, -0.5, -0.5), self._Gf.Vec3f(0.5, 0.5, 0.5)]
        )
        xform = self._UsdGeom.Xformable(table.GetPrim())
        self._table_translate_op = xform.AddTranslateOp()
        self._table_translate_op.Set(
            self._Gf.Vec3d(
                float(self.scene_config.table_center_world_xy_m[0]),
                float(self.scene_config.table_center_world_xy_m[1]),
                float(self.scene_config.table_surface_height_m) / 2.0,
            )
        )
        self._table_scale_op = xform.AddScaleOp()
        self._table_scale_op.Set(
            self._Gf.Vec3d(
                TABLE_SIZE_X_M,
                TABLE_SIZE_Y_M,
                float(self.scene_config.table_surface_height_m),
            )
        )
        self._UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        display = table.CreateDisplayColorAttr()
        display.Set([self._Gf.Vec3f(0.33, 0.35, 0.38)])
        return table

    def set_table_surface_height(
        self,
        surface_height_m: float,
        *,
        allowed_height_range_m: tuple[float, float] = TABLE_SURFACE_HEIGHT_RANGE_M,
    ) -> float:
        """Update the actual static table collider for one episode reset."""

        bounds = np.asarray(allowed_height_range_m, dtype=np.float64)
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
        ):
            raise Stage2IsaacSceneError(
                "table surface height range must contain two ordered finite values"
            )
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if not math.isfinite(surface_height_m) or not lower <= surface_height_m <= upper:
            raise Stage2IsaacSceneError(
                "episode table surface height is outside its sealed runtime range"
            )
        height = float(surface_height_m)
        self._table_translate_op.Set(
            self._Gf.Vec3d(
                float(self.scene_config.table_center_world_xy_m[0]),
                float(self.scene_config.table_center_world_xy_m[1]),
                height / 2.0,
            )
        )
        self._table_scale_op.Set(
            self._Gf.Vec3d(TABLE_SIZE_X_M, TABLE_SIZE_Y_M, height)
        )
        self._stage_utils.update_stage()
        measured = self.table_surface_height_from_usd()
        if not math.isclose(measured, height, rel_tol=0.0, abs_tol=1.0e-6):
            raise Stage2IsaacSceneError(
                "authored static table collider height did not match reset sample"
            )
        return measured

    def table_surface_height_from_usd(self) -> float:
        translation = self._table_translate_op.Get()
        scale = self._table_scale_op.Get()
        measured = float(translation[2]) + float(scale[2]) / 2.0
        if not np.allclose(
            [float(scale[0]), float(scale[1])],
            [TABLE_SIZE_X_M, TABLE_SIZE_Y_M],
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise Stage2IsaacSceneError(
                "table collider XY size changed during reset: "
                f"measured={[float(scale[0]), float(scale[1])]}"
            )
        return measured

    def fixed_frame_rotation_attestation(
        self,
    ) -> dict[str, list[list[float]]]:
        """Re-read fixed base/table rotations from the current USD stage."""

        base_prim = self.stage.GetPrimAtPath(BASE_LINK_PRIM_PATH)
        table_prim = self.stage.GetPrimAtPath(TABLE_PRIM_PATH)
        if not base_prim.IsValid() or not table_prim.IsValid():
            raise Stage2IsaacSceneError(
                "fixed-frame rotation attestation prim is missing"
            )
        cache = self._UsdGeom.XformCache()
        base_world = cache.GetLocalToWorldTransform(base_prim)
        table_world = cache.GetLocalToWorldTransform(table_prim)
        return attest_fixed_scene_identity_rotations(
            robot_base_world_rotation=np.asarray(
                self._Gf.Matrix3d(base_world.ExtractRotation()),
                dtype=np.float64,
            ),
            table_world_rotation=np.asarray(
                self._Gf.Matrix3d(table_world.ExtractRotation()),
                dtype=np.float64,
            ),
        )

    def _define_cube(self) -> Any:
        cube = self._UsdGeom.Cube.Define(self.stage, CUBE_PRIM_PATH)
        cube.CreateSizeAttr(float(self.scene_config.cube_edge_m))
        half = float(self.scene_config.cube_edge_m) / 2.0
        cube.CreateExtentAttr(
            [
                self._Gf.Vec3f(-half, -half, -half),
                self._Gf.Vec3f(half, half, half),
            ]
        )
        xform = self._UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(
            self._Gf.Vec3d(
                float(self.scene_config.table_center_world_xy_m[0]),
                float(self.scene_config.table_center_world_xy_m[1]),
                float(self.scene_config.cube_center_world_z_m),
            )
        )
        self._UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self._UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        self._UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(
            float(self.build_config.cube_mass_kg)
        )
        rigid = self._PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
        rigid.CreateDisableGravityAttr(False)
        cube.CreateDisplayColorAttr().Set(
            [self._Gf.Vec3f(*self.build_config.cube_color_rgb)]
        )
        # Isaac 5 / Replicator reads the current UsdSemantics.LabelsAPI.  The
        # legacy SemanticsAPI helper is intentionally not used: it can leave
        # the rendered object UNLABELLED even though legacy USD attributes
        # exist.  This label supplies a measured segmentation mask only; the
        # cube pose is never substituted into the policy observation.
        self._semantics_utils.add_labels(
            cube.GetPrim(),
            labels=["stage2_cube"],
            instance_name="class",
            overwrite=True,
        )
        self.validate_cube_semantics()
        return cube

    def validate_cube_semantics(self) -> dict[str, list[str]]:
        """Attest the rendered cube's current LabelsAPI class assignment."""

        cube = self.stage.GetPrimAtPath(CUBE_PRIM_PATH)
        if not cube.IsValid():
            raise Stage2IsaacSceneError(
                "Stage 2 cube is missing during semantic attestation"
            )
        labels = self._semantics_utils.get_labels(cube)
        normalized = {
            str(instance): [str(label) for label in values]
            for instance, values in labels.items()
        }
        if normalized.get("class") != ["stage2_cube"]:
            raise Stage2IsaacSceneError(
                "Stage 2 cube must have current UsdSemantics class label "
                f"['stage2_cube']; measured={normalized}"
            )
        applied = tuple(str(value) for value in cube.GetAppliedSchemas())
        if "SemanticsLabelsAPI:class" not in applied:
            raise Stage2IsaacSceneError(
                "Stage 2 cube lacks SemanticsLabelsAPI:class"
            )
        return normalized

    def _configure_task_head_camera_view(self) -> None:
        """Author one measured task-camera extrinsic without moving the G2."""

        camera = self.stage.GetPrimAtPath(CAMERA_PRIM_PATHS["head"])
        if not camera.IsValid():
            raise Stage2IsaacSceneError("task head camera prim is missing")
        orient_ops = [
            op
            for op in self._UsdGeom.Xformable(camera).GetOrderedXformOps()
            if str(op.GetOpName()) == "xformOp:orient"
        ]
        if len(orient_ops) != 1:
            raise Stage2IsaacSceneError(
                "task head camera must have exactly one xformOp:orient"
            )
        authored_quaternion = orient_ops[0].Get()
        imaginary = authored_quaternion.GetImaginary()
        authored = np.asarray(
            [
                float(authored_quaternion.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            dtype=np.float64,
        )
        aimed = task_head_camera_orientation_wxyz(authored)
        orient_ops[0].Set(
            self._Gf.Quatd(
                float(aimed[0]),
                self._Gf.Vec3d(
                    float(aimed[1]), float(aimed[2]), float(aimed[3])
                ),
            )
        )
        self._task_head_camera_orientation_wxyz = aimed
        self._stage_utils.update_stage()

    def _define_gpu_physics_scene(self) -> None:
        physics_scene = self._UsdPhysics.Scene.Define(
            self.stage, "/World/Stage2PhysicsScene"
        )
        physics_scene.CreateGravityDirectionAttr(self._Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
        physx = self._PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        # Isaac Sim 5 does not support CCD with GPU dynamics.  Author and
        # read back the required disabled state before SimulationContext
        # reuses this scene.  Isaac 5.1 may still log its own GPU-selection
        # transition warning while constructing PhysicsContext, after which
        # the same disabled state remains active.
        ccd_attribute = physx.CreateEnableCCDAttr()
        if ccd_attribute.Set(False) is False or ccd_attribute.Get() is not False:
            raise Stage2IsaacSceneError(
                "failed to author GPU physics scene CCD=false"
            )
        physx.CreateEnableGPUDynamicsAttr(True)
        physx.CreateBroadphaseTypeAttr("GPU")
        physx.CreateSolverTypeAttr("TGS")

    def _compose(self) -> None:
        self._stage_utils.create_new_stage()
        self.stage = self._stage_utils.get_current_stage()
        self.stage.SetLoadRules(self._Usd.StageLoadRules.LoadNone())
        self._UsdGeom.SetStageUpAxis(self.stage, self._UsdGeom.Tokens.z)
        self._UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self._UsdGeom.Xform.Define(self.stage, "/World")
        self._reference_robot()
        self._define_table()
        self._define_cube()
        self._define_gpu_physics_scene()
        light = self._UsdLux.DistantLight.Define(
            self.stage, "/World/Stage2DistantLight"
        )
        light.CreateIntensityAttr(1200.0)
        self.stage.Load(ROBOT_PRIM_PATH)
        self._stage_utils.update_stage()
        deadline = time.monotonic() + self.build_config.asset_load_timeout_s
        while self._stage_utils.is_stage_loading():
            if time.monotonic() >= deadline:
                raise TimeoutError("Stage 2 G2 asset loading timed out")
            self.simulation_app.update()
        self._robot_collision_activation_attestation = (
            self._enable_and_attest_robot_collisions()
        )
        self._configure_task_head_camera_view()
        fixed_joint = self.stage.GetPrimAtPath(f"{BASE_LINK_PRIM_PATH}/FixedJoint")
        if not fixed_joint.IsValid():
            raise Stage2IsaacSceneError("G2 base fixed joint is missing")
        fixed_joint.SetActive(True)
        base_prim = self.stage.GetPrimAtPath(BASE_LINK_PRIM_PATH)
        base_world = self._UsdGeom.XformCache().GetLocalToWorldTransform(base_prim)
        base_translation = np.asarray(base_world.ExtractTranslation(), dtype=np.float64)
        if base_translation.shape != (3,) or not np.allclose(
            base_translation, np.zeros(3), atol=1.0e-6, rtol=0.0
        ):
            raise Stage2IsaacSceneError(
                "actual G2 base_link must be authored at world [0, 0, 0]"
            )
        self._robot_base_world_translation_m = base_translation
        self._fixed_scene_rotation_attestation = (
            self.fixed_frame_rotation_attestation()
        )
        bbox_cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(),
            [
                self._UsdGeom.Tokens.default_,
                self._UsdGeom.Tokens.render,
                self._UsdGeom.Tokens.proxy,
            ],
        )
        aligned_range = bbox_cache.ComputeWorldBound(base_prim.GetParent()).ComputeAlignedRange()
        bounds_min = np.asarray(aligned_range.GetMin(), dtype=np.float64)
        bounds_max = np.asarray(aligned_range.GetMax(), dtype=np.float64)
        if (
            bounds_min.shape != (3,)
            or bounds_max.shape != (3,)
            or not np.all(np.isfinite(np.concatenate((bounds_min, bounds_max))))
            or np.any(bounds_min >= bounds_max)
        ):
            raise Stage2IsaacSceneError("G2 visual world bounds are invalid")
        self._robot_visual_world_bounds_m = (bounds_min, bounds_max)
        if not self.build_config.headless:
            self._viewport_utils.set_camera_view(
                eye=np.asarray([1.35, -1.25, 1.50]),
                target=np.asarray(
                    [self.scene_config.table_center_world_xy_m[0], 0.0, 0.80]
                ),
                camera_prim_path="/OmniverseKit_Persp",
            )

    def _validate_composed_scene(self) -> dict[str, Any]:
        required = (
            ROBOT_PRIM_PATH,
            BASE_LINK_PRIM_PATH,
            TABLE_PRIM_PATH,
            CUBE_PRIM_PATH,
            *CAMERA_PRIM_PATHS.values(),
        )
        missing = [
            path
            for path in required
            if not self.stage.GetPrimAtPath(path).IsValid()
        ]
        if missing:
            raise Stage2IsaacSceneError(f"composed Stage 2 prims are missing: {missing}")
        cube = self.stage.GetPrimAtPath(CUBE_PRIM_PATH)
        if not cube.HasAPI(self._UsdPhysics.RigidBodyAPI):
            raise Stage2IsaacSceneError("Stage 2 cube is not a rigid body")
        if not cube.HasAPI(self._UsdPhysics.MassAPI):
            raise Stage2IsaacSceneError("Stage 2 cube has no explicit mass override")
        measured_mass = float(self._UsdPhysics.MassAPI(cube).GetMassAttr().Get())
        if abs(measured_mass - CUBE_MASS_NOMINAL_KG) > (
            CUBE_MASS_TOLERANCE_KG + 1.0e-12
        ):
            raise Stage2IsaacSceneError(
                "Stage 2 cube mass violates the selected mass profile"
            )
        semantic_labels = self.validate_cube_semantics()
        table = self.stage.GetPrimAtPath(TABLE_PRIM_PATH)
        if table.HasAPI(self._UsdPhysics.RigidBodyAPI):
            raise Stage2IsaacSceneError("Stage 2 table must be a static collider")
        return {
            "environment": ENVIRONMENT_ID,
            "actual_usd_composed": True,
            "robot_prim_path": ROBOT_PRIM_PATH,
            "robot_base_world_translation_m": [
                float(value) for value in self._robot_base_world_translation_m
            ],
            "robot_base_world_rotation_3x3": self._fixed_scene_rotation_attestation[
                "robot_base_world_rotation"
            ],
            "robot_fixed_base": True,
            "ground_plane_present": False,
            "robot_visual_world_bounds_m": {
                "min": [
                    float(value) for value in self._robot_visual_world_bounds_m[0]
                ],
                "max": [
                    float(value) for value in self._robot_visual_world_bounds_m[1]
                ],
            },
            "negative_visual_z_reason": (
                "source_asset_has_0.04m_staging_offset_removed_for_base_frame; "
                "no_ground_collider_in_fixed_base_table_task"
            ),
            "environment_static_collision_prims": [TABLE_PRIM_PATH],
            "robot_collision_activation": (
                self._robot_collision_activation_attestation
            ),
            "table_prim_path": TABLE_PRIM_PATH,
            "table_size_xy_m": [TABLE_SIZE_X_M, TABLE_SIZE_Y_M],
            "table_surface_height_m": self.scene_config.table_surface_height_m,
            "table_surface_height_nominal_m": self.scene_config.table_surface_height_m,
            "table_surface_height_range_m": list(
                self.scene_config.table_surface_height_range_m
            ),
            "table_height_sample_distribution": "uniform",
            "table_height_randomized_each_episode": True,
            "table_static_collider": True,
            "table_world_rotation_3x3": self._fixed_scene_rotation_attestation[
                "table_world_rotation"
            ],
            "fixed_scene_identity_rotation_attested": True,
            "cube_prim_path": CUBE_PRIM_PATH,
            "cube_edge_m": self.scene_config.cube_edge_m,
            "cube_mass_nominal_kg": CUBE_MASS_NOMINAL_KG,
            "cube_mass_tolerance_kg": CUBE_MASS_TOLERANCE_KG,
            "cube_mass_measured_kg": measured_mass,
            "cube_mass_randomized_each_episode": False,
            "cube_dynamic": True,
            "cube_fixed_payload": False,
            "cube_semantic_schema": "UsdSemantics.LabelsAPI",
            "cube_semantic_labels": semantic_labels,
            "camera_prim_paths": dict(CAMERA_PRIM_PATHS),
            "task_head_camera_downward_pitch_rad": (
                TASK_HEAD_CAMERA_DOWNWARD_PITCH_RAD
            ),
            "task_head_camera_orientation_wxyz": [
                float(value)
                for value in self._task_head_camera_orientation_wxyz
            ],
            "task_head_camera_alignment_authority": (
                "usd_sensor_extrinsic_override_planner_locked_head_unchanged"
            ),
            "gpu_dynamics_requested": True,
            "gpu_dynamics_ccd_enabled": False,
            "gpu_dynamics_ccd_scene_authoring_readback": False,
            "gpu_dynamics_collision_sampling": (
                "discrete_physx_at_500hz_default_runtime"
            ),
        }


__all__ = [
    "IsaacSceneBuildConfig",
    "IsaacSimpleStage2Scene",
    "Stage2IsaacSceneError",
    "TASK_HEAD_CAMERA_DOWNWARD_PITCH_RAD",
    "attest_fixed_scene_identity_rotations",
    "task_head_camera_orientation_wxyz",
]
