# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed cuRobo authority for the exact-4D G2 policy branch.

This module only constructs an offline planner model.  It never imports Kit,
creates an Isaac environment, sends a joint command, or changes the validated
4-D controller contract.  The existing bilateral cuRobo YAML remains the
collision-sphere source, but its 34-DoF cspace is deliberately replaced by the
seven active right-arm joints.  Every other movable coordinate relevant to
the fixed full-body collision model is locked at the existing task reset
authority.

The planner uses the exact frozen Legacy Candidate A USD for kinematics.  Its
complete dependency and mechanics provenance receipt is verified before a
configuration can be emitted.  The checked-in URDF remains an independent
name/order/limit authority; it cannot replace Candidate A as the FK asset.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import yaml

from ..g2_keyboard_pose import G2_KEYBOARD_PHOTO_RIGHT_ARM_Q
from ..g2_lift_methodology import (
    BASE_WHEEL_JOINTS,
    EXHIBITION_HEAD_Q,
    LEFT_ARM_HOME,
    RIGHT_ARM_JOINTS,
    stable_original_home,
)
from .contact_free_candidate_a_binding import (
    CANDIDATE_A_RELATIVE_PATH,
    CANDIDATE_A_SHA256,
    ContactFreeCandidateABindingError,
    resolve_contact_free_candidate_a,
)


CUROBO_RIGHT_ARM_AUTHORITY_SCHEMA = "g2_curobo_right_arm_7dof_candidate_a_authority_v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
BASE_COLLISION_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "source/data_collection/config/curobo/configs/robot/"
    "G2_omnipicker_fixed_dual.yml"
)
URDF_PATH = (
    REPOSITORY_ROOT
    / "source/geniesim/assets/robot/curobo_robot/assets/robot/G2/"
    "G2_omnipicker_fixed_dual.urdf"
)
CANDIDATE_A_USD_PATH = REPOSITORY_ROOT / CANDIDATE_A_RELATIVE_PATH
# Compatibility name for offline diagnostic imports.  It no longer resolves
# to E1: every production/expert-data planner path is Candidate-A-only.
M2_CANDIDATE_USD_PATH = CANDIDATE_A_USD_PATH

RIGHT_ARM_7DOF_JOINTS = tuple(RIGHT_ARM_JOINTS)
RIGHT_ARM_RETRACT_Q_RAD = tuple(float(v) for v in G2_KEYBOARD_PHOTO_RIGHT_ARM_Q)
BASE_FRAME = "base_link"
EE_FRAME = "gripper_r_center_link"
FIXED_WRIST_ORIENTATION_XYZW = (
    0.500459611415863,
    0.5004866719245911,
    0.49954208731651306,
    0.49951067566871643,
)

# The experimental cuRobo USD parser does not read
# ``physxJoint:maxJointVelocity`` and falls back to LinkParams' +/-2 rad/s.
# Constrain that parser value to the already-qualified project motion envelope
# instead of silently accepting the parser default.  These are planner bounds;
# the controller and post-physics safety gates remain unchanged and continue to
# own execution acceptance.
CUROBO_USD_PARSER_RAW_VELOCITY_LIMIT_RAD_S = 2.0
EXISTING_HARD_VELOCITY_LIMIT_RAD_S = 0.8
EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2 = 10.0
PLANNER_TIMING_DELTA_VELOCITY_LIMIT_RAD_S = 0.85
# cuRobo 0.7.7 applies ``velocity_scale`` once in
# CudaRobotGenerator._update_joint_limits and again in
# KinematicsTensorConfig.__post_init__.  Use the square root so the
# materialized limit, which is what MotionGen consumes, is exactly 0.8.
# The Stage-2 qualifier verifies that result and fails closed if a future
# cuRobo release changes this implementation detail.
PLANNER_VELOCITY_SCALE_APPLICATION_COUNT = 2
PLANNER_VELOCITY_SCALE = math.sqrt(
    EXISTING_HARD_VELOCITY_LIMIT_RAD_S
    / CUROBO_USD_PARSER_RAW_VELOCITY_LIMIT_RAD_S
)

# Existing collision-YAML open-hand geometry.  These are planner collision
# locks, not actuator commands and not a passive-mechanism authority.  The
# exact live settled hand remains owned by the Isaac reset/cache contract.
_RIGHT_OPEN_COLLISION_LOCKS = {
    "idx71_gripper_r_inner_joint1": -0.785,
    "idx72_gripper_r_inner_joint3": 0.0,
    "idx73_gripper_r_inner_joint4": 0.349,
    "idx81_gripper_r_outer_joint1": 0.785,
    "idx82_gripper_r_outer_joint3": 0.01,
    "idx83_gripper_r_outer_joint4": -0.35,
}


class CuroboPlannerAuthorityError(RuntimeError):
    """Raised when a source or exact right-arm contract has drifted."""


@dataclass(frozen=True)
class JointLimitAuthority:
    name: str
    lower_rad: float
    upper_rad: float
    velocity_rad_s: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_a_binding_receipt():
    """Resolve the complete Candidate A provenance or fail before planning."""

    try:
        receipt = resolve_contact_free_candidate_a(repo_root=REPOSITORY_ROOT)
    except ContactFreeCandidateABindingError as error:
        raise CuroboPlannerAuthorityError(
            f"CANDIDATE_A_AUTHORITY_UNRESOLVED:{error}"
        ) from error
    if (
        Path(receipt.candidate_asset_path).resolve() != CANDIDATE_A_USD_PATH.resolve()
        or receipt.candidate_asset_sha256 != CANDIDATE_A_SHA256
        or receipt.alternate_asset_fallback != "NONE"
    ):
        raise CuroboPlannerAuthorityError("CANDIDATE_A_AUTHORITY_RECEIPT_MISMATCH")
    return receipt


def assert_candidate_a_curobo_authority(config: Mapping[str, Any]) -> str:
    """Verify FK/collision configuration is bound to Candidate A only.

    The returned fingerprint covers the Candidate A asset/dependency/contract
    chain plus the cuRobo collision source and the exact planning frame/DoF
    semantics.  A diagnostic E1 path or any alternate USD is rejected.
    """

    receipt = _candidate_a_binding_receipt()
    try:
        kinematics = config["robot_cfg"]["kinematics"]
    except (KeyError, TypeError) as error:
        raise CuroboPlannerAuthorityError("CUROBO_KINEMATICS_CONFIG_MISSING") from error
    expected = CANDIDATE_A_USD_PATH.resolve()
    for field in ("isaac_usd_path", "usd_path"):
        try:
            observed = Path(kinematics[field]).expanduser().resolve()
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise CuroboPlannerAuthorityError(
                f"CUROBO_{field.upper()}_INVALID"
            ) from error
        if observed != expected:
            raise CuroboPlannerAuthorityError(
                f"DIAGNOSTIC_OR_ALTERNATE_ASSET_REJECTED:{field}:{observed}"
            )
    if not bool(kinematics.get("use_usd_kinematics")):
        raise CuroboPlannerAuthorityError("CANDIDATE_A_USD_KINEMATICS_REQUIRED")
    cspace = kinematics.get("cspace", {})
    if tuple(cspace.get("joint_names", ())) != RIGHT_ARM_7DOF_JOINTS:
        raise CuroboPlannerAuthorityError("CANDIDATE_A_PLANNING_DOF_MISMATCH")
    if kinematics.get("base_link") != BASE_FRAME or kinematics.get("ee_link") != EE_FRAME:
        raise CuroboPlannerAuthorityError("CANDIDATE_A_PLANNING_FRAME_MISMATCH")
    semantic_payload = {
        "candidate_asset_sha256": receipt.candidate_asset_sha256,
        "candidate_dependency_manifest_sha256": receipt.dependency_manifest_sha256,
        "candidate_manifest_sha256": receipt.candidate_manifest_sha256,
        "candidate_contract_sha256": receipt.candidate_contract_sha256,
        "mechanics_authority_sha256": receipt.mechanics_authority_sha256,
        "collision_model_sha256": _sha256(BASE_COLLISION_CONFIG_PATH),
        "joint_name_limit_authority_sha256": _sha256(URDF_PATH),
        "base_frame": BASE_FRAME,
        "ee_frame": EE_FRAME,
        "planning_joints": list(RIGHT_ARM_7DOF_JOINTS),
        "passive_or_mimic_planning_dof_count": 0,
    }
    return _canonical_sha256(semantic_payload)


def right_arm_joint_limits_from_urdf() -> tuple[JointLimitAuthority, ...]:
    """Read the seven active-joint limits from the checked-in URDF."""

    root = ET.parse(URDF_PATH).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    limits: list[JointLimitAuthority] = []
    for name in RIGHT_ARM_7DOF_JOINTS:
        joint = joints.get(name)
        if joint is None or joint.attrib.get("type") != "revolute":
            raise CuroboPlannerAuthorityError(f"RIGHT_ARM_JOINT_MISSING:{name}")
        limit = joint.find("limit")
        if limit is None:
            raise CuroboPlannerAuthorityError(f"RIGHT_ARM_LIMIT_MISSING:{name}")
        values = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
            float(limit.attrib["velocity"]),
        )
        if not all(math.isfinite(value) for value in values):
            raise CuroboPlannerAuthorityError(f"RIGHT_ARM_LIMIT_NONFINITE:{name}")
        if not values[0] < values[1] or values[2] <= 0.0:
            raise CuroboPlannerAuthorityError(f"RIGHT_ARM_LIMIT_INVALID:{name}")
        limits.append(JointLimitAuthority(name, *values))
    return tuple(limits)


def _fixed_collision_joint_locks() -> dict[str, float]:
    """Return non-planning coordinates for the task's fixed-body geometry."""

    task_pose = stable_original_home()
    locks: dict[str, float] = {
        name: float(value)
        for name, value in task_pose.items()
        if name not in RIGHT_ARM_7DOF_JOINTS
    }
    # The task keeps the camera-aware head pose and a closed, fixed left hand.
    locks.update(
        {
            f"idx1{i}_head_joint{i}": float(value)
            for i, value in enumerate(EXHIBITION_HEAD_Q, 1)
        }
    )
    locks.update(
        {
            f"idx2{i}_arm_l_joint{i}": float(value)
            for i, value in enumerate(LEFT_ARM_HOME, 1)
        }
    )
    # The existing cuRobo collision model does not include the link2 children,
    # so joint0 does not affect any configured collision sphere.  Remove those
    # four unused branch leaves instead of claiming an open-state authority.
    for name in tuple(locks):
        if name.endswith("_joint0"):
            locks.pop(name)
    # cuRobo's USD parser does not interpret PhysX mimic APIs.  Both inner and
    # outer joint1 therefore have to be fixed collision coordinates here.
    # They remain excluded from the planning cspace and are never commands.
    locks.update(_RIGHT_OPEN_COLLISION_LOCKS)
    # The cuRobo parser roots the planner at ``base_link`` and therefore does
    # not include the AMR wheel branches below that base even though their
    # XML elements remain in the whole-robot URDF.  They also have no spheres
    # in this planner collision model.  Do not present them as lock joints.
    for name in BASE_WHEEL_JOINTS:
        locks.pop(name, None)
    # Filter any other source pose entry that is not present in this URDF.
    urdf_joint_names = {
        joint.attrib["name"]
        for joint in ET.parse(URDF_PATH).getroot().findall("joint")
    }
    return {name: value for name, value in locks.items() if name in urdf_joint_names}


def _validated_planner_limits(
    *, velocity_limit_rad_s: float, acceleration_limit_rad_s2: float
) -> tuple[float, float]:
    """Validate an explicitly selected planner-only timing envelope.

    ``0.8`` remains the historical authority.  The only admitted delta is the
    bounded ``0.85`` planner timing experiment requested for the
    RESET->GRASP_READY requalification.  This does not modify the controller,
    asset, passive mechanism, or the 10-rad/s^2 acceleration gate.
    """

    velocity = float(velocity_limit_rad_s)
    acceleration = float(acceleration_limit_rad_s2)
    if velocity not in (
        EXISTING_HARD_VELOCITY_LIMIT_RAD_S,
        PLANNER_TIMING_DELTA_VELOCITY_LIMIT_RAD_S,
    ):
        raise CuroboPlannerAuthorityError(
            f"UNAUTHORIZED_PLANNER_VELOCITY_LIMIT:{velocity}"
        )
    if acceleration != EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2:
        raise CuroboPlannerAuthorityError(
            f"UNAUTHORIZED_PLANNER_ACCELERATION_LIMIT:{acceleration}"
        )
    return velocity, acceleration


def build_right_arm_7dof_robot_config(
    *,
    velocity_limit_rad_s: float = EXISTING_HARD_VELOCITY_LIMIT_RAD_S,
    acceleration_limit_rad_s2: float = EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2,
) -> dict[str, Any]:
    """Build a deterministic cuRobo dictionary with an exact 7-DoF cspace."""

    _candidate_a_binding_receipt()
    for path in (BASE_COLLISION_CONFIG_PATH, URDF_PATH, CANDIDATE_A_USD_PATH):
        if not path.is_file():
            raise CuroboPlannerAuthorityError(f"REQUIRED_SOURCE_MISSING:{path}")
    with BASE_COLLISION_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream)
    if not isinstance(source, dict) or "robot_cfg" not in source:
        raise CuroboPlannerAuthorityError("BASE_CUROBO_CONFIG_INVALID")
    velocity_limit_rad_s, acceleration_limit_rad_s2 = _validated_planner_limits(
        velocity_limit_rad_s=velocity_limit_rad_s,
        acceleration_limit_rad_s2=acceleration_limit_rad_s2,
    )
    result = deepcopy(source)
    kinematics = result["robot_cfg"]["kinematics"]
    if kinematics.get("base_link") != BASE_FRAME:
        raise CuroboPlannerAuthorityError("BASE_FRAME_DRIFT")
    if kinematics.get("ee_link") != EE_FRAME:
        raise CuroboPlannerAuthorityError("EE_FRAME_DRIFT")

    # Bind the exact, immutable Candidate A authority.  A previous URDF-mode
    # probe exposed a systematic 0.730-mm fixed-transform offset relative to
    # the composed USD; retaining URDF kinematics would therefore fail strict
    # model parity even though joint names and limits match.
    kinematics["use_usd_kinematics"] = True
    kinematics["urdf_path"] = str(URDF_PATH)
    kinematics["asset_root_path"] = str(URDF_PATH.parents[4])
    kinematics["isaac_usd_path"] = str(CANDIDATE_A_USD_PATH)
    kinematics["usd_path"] = str(CANDIDATE_A_USD_PATH)
    kinematics["usd_robot_root"] = "/genie"
    kinematics["lock_joints"] = _fixed_collision_joint_locks()
    kinematics["cspace"] = {
        "joint_names": list(RIGHT_ARM_7DOF_JOINTS),
        "retract_config": list(RIGHT_ARM_RETRACT_Q_RAD),
        "null_space_weight": [1.0] * 7,
        "cspace_distance_weight": [1.0] * 7,
        # cuRobo's experimental USD parser supplies +/-2 rad/s rather than
        # reading the USD velocity property.  Scale that parser value to the
        # unchanged project hard envelope and verify the materialized result
        # in the Stage-2 qualifier.  This does not replace the downstream
        # measured-motion safety gate.
        "velocity_scale": math.sqrt(
            velocity_limit_rad_s / CUROBO_USD_PARSER_RAW_VELOCITY_LIMIT_RAD_S
        ),
        "max_jerk": 500.0,
        "max_acceleration": acceleration_limit_rad_s2,
    }
    assert_candidate_a_curobo_authority(result)
    return result


def authority_manifest(
    *,
    velocity_limit_rad_s: float = EXISTING_HARD_VELOCITY_LIMIT_RAD_S,
    acceleration_limit_rad_s2: float = EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2,
) -> dict[str, Any]:
    """Return source-linked facts and a fingerprint for the built config."""

    velocity_limit_rad_s, acceleration_limit_rad_s2 = _validated_planner_limits(
        velocity_limit_rad_s=velocity_limit_rad_s,
        acceleration_limit_rad_s2=acceleration_limit_rad_s2,
    )
    config = build_right_arm_7dof_robot_config(
        velocity_limit_rad_s=velocity_limit_rad_s,
        acceleration_limit_rad_s2=acceleration_limit_rad_s2,
    )
    limits = right_arm_joint_limits_from_urdf()
    config_fingerprint = _canonical_sha256(config)
    candidate_binding = _candidate_a_binding_receipt()
    mechanical_semantic_fingerprint = assert_candidate_a_curobo_authority(config)
    return {
        "schema": CUROBO_RIGHT_ARM_AUTHORITY_SCHEMA,
        "planning_dof": 7,
        "planning_joints": list(RIGHT_ARM_7DOF_JOINTS),
        "excluded_from_planning": [
            "torso",
            "head",
            "left_arm",
            "base_wheels",
            "OmniPicker_master",
            "OmniPicker_mimic",
            "OmniPicker_passive_four_bar",
        ],
        "base_frame": BASE_FRAME,
        "ee_frame": EE_FRAME,
        "fixed_wrist_orientation_xyzw": list(FIXED_WRIST_ORIENTATION_XYZW),
        "retract_q_rad": list(RIGHT_ARM_RETRACT_Q_RAD),
        "joint_limits": [limit.__dict__ for limit in limits],
        "velocity_authority": {
            "urdf_model_limit_rad_s": [
                limit.velocity_rad_s for limit in limits
            ],
            "curobo_usd_parser_raw_limit_rad_s": (
                CUROBO_USD_PARSER_RAW_VELOCITY_LIMIT_RAD_S
            ),
            "planner_limit_rad_s": velocity_limit_rad_s,
            "planner_velocity_scale": math.sqrt(
                velocity_limit_rad_s
                / CUROBO_USD_PARSER_RAW_VELOCITY_LIMIT_RAD_S
            ),
            "curobo_velocity_scale_application_count": (
                PLANNER_VELOCITY_SCALE_APPLICATION_COUNT
            ),
            "ownership": (
                "PLANNER_CONSERVATIVE_BOUND_FROM_EXISTING_PROJECT_HARD_GATE;"
                "CONTROLLER_AND_POST_PHYSICS_GATE_UNCHANGED"
            ),
        },
        "acceleration_authority": {
            "planner_limit_rad_s2": acceleration_limit_rad_s2,
            "ownership": (
                "PLANNER_CONSERVATIVE_BOUND_FROM_EXISTING_PROJECT_HARD_GATE;"
                "CONTROLLER_AND_POST_PHYSICS_GATE_UNCHANGED"
            ),
        },
        "source": {
            "base_collision_config": str(BASE_COLLISION_CONFIG_PATH),
            "base_collision_config_sha256": _sha256(BASE_COLLISION_CONFIG_PATH),
            "urdf": str(URDF_PATH),
            "urdf_sha256": _sha256(URDF_PATH),
            "candidate_a_usd": str(CANDIDATE_A_USD_PATH),
            "candidate_a_usd_sha256": _sha256(CANDIDATE_A_USD_PATH),
            "candidate_a_dependency_manifest_sha256": (
                candidate_binding.dependency_manifest_sha256
            ),
            "candidate_a_contract_sha256": candidate_binding.candidate_contract_sha256,
        },
        "config_canonical_sha256": config_fingerprint,
        "mechanical_semantic_fingerprint": mechanical_semantic_fingerprint,
        "mechanical_authority": "LEGACY_CANDIDATE_A_CONTACT_FREE_ONLY",
        "diagnostic_asset_fallback": "REJECT",
        "passive_or_mimic_planning_dof_count": 0,
        "controller_or_asset_modified": False,
        "right_open_collision_lock_authority": (
            "EXISTING_CUROBO_COLLISION_YAML_APPROXIMATION_NOT_ACTUATOR_AUTHORITY"
        ),
    }


__all__ = [
    "BASE_FRAME",
    "CANDIDATE_A_USD_PATH",
    "CUROBO_RIGHT_ARM_AUTHORITY_SCHEMA",
    "CuroboPlannerAuthorityError",
    "CUROBO_USD_PARSER_RAW_VELOCITY_LIMIT_RAD_S",
    "EE_FRAME",
    "EXISTING_HARD_ACCELERATION_LIMIT_RAD_S2",
    "EXISTING_HARD_VELOCITY_LIMIT_RAD_S",
    "FIXED_WRIST_ORIENTATION_XYZW",
    "M2_CANDIDATE_USD_PATH",
    "PLANNER_TIMING_DELTA_VELOCITY_LIMIT_RAD_S",
    "PLANNER_VELOCITY_SCALE",
    "PLANNER_VELOCITY_SCALE_APPLICATION_COUNT",
    "RIGHT_ARM_7DOF_JOINTS",
    "RIGHT_ARM_RETRACT_Q_RAD",
    "URDF_PATH",
    "authority_manifest",
    "assert_candidate_a_curobo_authority",
    "build_right_arm_7dof_robot_config",
    "right_arm_joint_limits_from_urdf",
]
