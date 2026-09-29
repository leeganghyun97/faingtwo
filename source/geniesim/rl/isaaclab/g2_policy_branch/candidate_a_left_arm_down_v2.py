# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Common reset authority for the Candidate-A left-arm-down V2 branch.

The Candidate-A USD remains byte-for-byte unchanged.  This module binds one
versioned reset-state manifest to every Keyboard/BC/SAC/evaluation config that
opts into V2 and rejects missing or drifted provenance before Isaac starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..g2_lift_methodology import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from .contact_free_candidate_a_binding import CANDIDATE_A_SHA256, repository_root
from .omnipicker_product_contract import OMNIPICKER_PRODUCT_CONTRACT


BASELINE_NAME = "LEGACY_CANDIDATE_A_LEFT_ARM_DOWN_V2"
BASELINE_SCHEMA = "g2_candidate_a_left_arm_down_baseline_v2"
BASELINE_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g2_candidate_a_left_arm_down_v2/BASELINE_MANIFEST.json"
)
LEFT_ARM_POSTURE_ID = "G2_LEFT_ARM_VERTICAL_DOWN_STRAIGHT_V2"
LEFT_ARM_DOWN_Q_RAD = (
    1.620750745749757,
    -1.6085981733983385,
    -2.2180736694597014,
    0.0,
    0.0,
    0.0,
    0.0,
)
LEFT_ARM_JOINT_LIMITS_RAD = (
    (-3.1067, 3.1067),
    (-2.0944, 2.0944),
    (-3.1067, 3.1067),
    (-2.5307, 1.0472),
    (-3.1067, 3.1067),
    (-1.0472, 1.0472),
    (-1.5708, 1.5708),
)


class CandidateALeftArmDownV2Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value: Any, *, width: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise CandidateALeftArmDownV2Error(f"{label}_INVALID") from error
    if len(result) != width or not all(math.isfinite(component) for component in result):
        raise CandidateALeftArmDownV2Error(f"{label}_INVALID")
    return result


def load_baseline_manifest() -> tuple[Path, dict[str, Any], str]:
    path = repository_root() / BASELINE_MANIFEST_RELATIVE_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidateALeftArmDownV2Error("BASELINE_MANIFEST_INVALID") from error
    if not isinstance(value, dict) or value.get("schema") != BASELINE_SCHEMA:
        raise CandidateALeftArmDownV2Error("BASELINE_MANIFEST_SCHEMA_MISMATCH")
    if value.get("baseline_name") != BASELINE_NAME:
        raise CandidateALeftArmDownV2Error("BASELINE_NAME_MISMATCH")
    old = value.get("old_candidate_a_asset")
    if not isinstance(old, Mapping) or old.get("sha256") != CANDIDATE_A_SHA256:
        raise CandidateALeftArmDownV2Error("OLD_CANDIDATE_A_HASH_MISMATCH")
    if old.get("modified") is not False:
        raise CandidateALeftArmDownV2Error("OLD_CANDIDATE_A_NOT_IMMUTABLE")
    product = value.get("omnipicker_product_contract")
    expected_product = OMNIPICKER_PRODUCT_CONTRACT.as_dict()
    if not isinstance(product, Mapping):
        raise CandidateALeftArmDownV2Error("OMNIPICKER_PRODUCT_CONTRACT_MISSING")
    for key in (
        "model",
        "manual_url",
        "manual_hardware_version",
        "maximum_gripping_force_n",
        "active_dof",
        "physical_pad_touch_sensor_available",
        "protocol_force_feedback_semantics",
        "simulator_contact_role",
        "student_contact_input_count",
        "m2_gripping_force_metric",
        "sim_contact_to_oem_gripping_force_mapping",
        "m2_force_gate_status",
    ):
        if product.get(key) != expected_product.get(key):
            raise CandidateALeftArmDownV2Error(
                f"OMNIPICKER_PRODUCT_CONTRACT_MISMATCH:{key}"
            )
    left = value.get("left_arm")
    if not isinstance(left, Mapping):
        raise CandidateALeftArmDownV2Error("LEFT_ARM_CONTRACT_MISSING")
    if tuple(left.get("joint_names", ())) != LEFT_ARM_JOINTS:
        raise CandidateALeftArmDownV2Error("LEFT_ARM_JOINT_ORDER_MISMATCH")
    if _finite_vector(left.get("joint_position_rad"), width=7, label="LEFT_ARM_Q") != LEFT_ARM_DOWN_Q_RAD:
        raise CandidateALeftArmDownV2Error("LEFT_ARM_Q_MISMATCH")
    sample = value.get("pregrasp_initial_state")
    if not isinstance(sample, Mapping):
        raise CandidateALeftArmDownV2Error("PREGRASP_SAMPLE_MISSING")
    if tuple(sample.get("right_arm_joint_names", ())) != RIGHT_ARM_JOINTS:
        raise CandidateALeftArmDownV2Error("RIGHT_ARM_JOINT_ORDER_MISMATCH")
    _finite_vector(sample.get("right_arm_q_rad"), width=7, label="RIGHT_ARM_Q")
    _finite_vector(sample.get("right_arm_qd_rad_s"), width=7, label="RIGHT_ARM_QD")
    _finite_vector(sample.get("ee_pose_robot_root_m_xyzw"), width=7, label="EE_POSE")
    _finite_vector(sample.get("cube_pose_robot_root_m_xyzw"), width=7, label="CUBE_POSE")
    for key in ("hdf5_path", "report_path"):
        evidence_path = Path(str(sample.get(key, "")))
        expected = str(sample.get(key.replace("path", "sha256"), ""))
        if not evidence_path.is_file() or _sha256(evidence_path) != expected:
            raise CandidateALeftArmDownV2Error(f"PREGRASP_{key.upper()}_DRIFT")
    if any(int(sample.get(name, -1)) != 0 for name in (
        "contact_count", "close_count", "forbidden_collision_count", "safety_reject_count"
    )):
        raise CandidateALeftArmDownV2Error("PREGRASP_SAMPLE_NOT_SAFE_OPEN")
    if sample.get("camera_observation_valid") is not True:
        raise CandidateALeftArmDownV2Error("PREGRASP_CAMERA_INVALID")
    return path, value, _sha256(path)


def left_arm_limit_margin_min_rad() -> float:
    return min(
        min(value - lower, upper - value)
        for value, (lower, upper) in zip(
            LEFT_ARM_DOWN_Q_RAD, LEFT_ARM_JOINT_LIMITS_RAD, strict=True
        )
    )


def apply_left_arm_down_v2_to_cfg(cfg: Any) -> dict[str, Any]:
    """Apply only the seven left-arm reset positions to an existing config."""

    manifest_path, manifest, manifest_sha256 = load_baseline_manifest()
    current = dict(cfg.scene.robot.init_state.joint_pos)
    current.update(dict(zip(LEFT_ARM_JOINTS, LEFT_ARM_DOWN_Q_RAD, strict=True)))
    cfg.scene.robot.init_state.joint_pos = current
    return {
        "schema": BASELINE_SCHEMA,
        "baseline_name": BASELINE_NAME,
        "baseline_manifest_path": str(manifest_path),
        "baseline_manifest_sha256": manifest_sha256,
        "candidate_asset_sha256": CANDIDATE_A_SHA256,
        "left_arm_posture_id": LEFT_ARM_POSTURE_ID,
        "left_arm_joint_names": list(LEFT_ARM_JOINTS),
        "left_arm_joint_position_rad": list(LEFT_ARM_DOWN_Q_RAD),
        "left_arm_limit_margin_min_rad": left_arm_limit_margin_min_rad(),
        "scope": manifest["scope"],
        "old_baseline_unchanged": True,
    }


@dataclass(frozen=True)
class PregraspInitialState:
    sample_id: str
    hdf5_path: str
    hdf5_sha256: str
    report_path: str
    report_sha256: str
    hdf5_row_index: int
    source_control_step: int
    right_arm_q_rad: tuple[float, ...]
    right_arm_qd_rad_s: tuple[float, ...]
    ee_pose_robot_root_m_xyzw: tuple[float, ...]
    cube_pose_robot_root_m_xyzw: tuple[float, ...]
    pad_to_cube_distance_m: float
    ee_to_cube_distance_m: float
    gripper_state: str


def selected_pregrasp_initial_state() -> PregraspInitialState:
    _path, manifest, _hash = load_baseline_manifest()
    sample = manifest["pregrasp_initial_state"]
    return PregraspInitialState(
        sample_id=str(sample["sample_id"]),
        hdf5_path=str(sample["hdf5_path"]),
        hdf5_sha256=str(sample["hdf5_sha256"]),
        report_path=str(sample["report_path"]),
        report_sha256=str(sample["report_sha256"]),
        hdf5_row_index=int(sample["hdf5_row_index"]),
        source_control_step=int(sample["source_control_step"]),
        right_arm_q_rad=_finite_vector(sample["right_arm_q_rad"], width=7, label="RIGHT_ARM_Q"),
        right_arm_qd_rad_s=_finite_vector(sample["right_arm_qd_rad_s"], width=7, label="RIGHT_ARM_QD"),
        ee_pose_robot_root_m_xyzw=_finite_vector(sample["ee_pose_robot_root_m_xyzw"], width=7, label="EE_POSE"),
        cube_pose_robot_root_m_xyzw=_finite_vector(sample["cube_pose_robot_root_m_xyzw"], width=7, label="CUBE_POSE"),
        pad_to_cube_distance_m=float(sample["pad_to_cube_distance_m"]),
        ee_to_cube_distance_m=float(sample["ee_to_cube_distance_m"]),
        gripper_state=str(sample["gripper_state"]),
    )


__all__ = [
    "BASELINE_MANIFEST_RELATIVE_PATH",
    "BASELINE_NAME",
    "BASELINE_SCHEMA",
    "CandidateALeftArmDownV2Error",
    "LEFT_ARM_DOWN_Q_RAD",
    "LEFT_ARM_POSTURE_ID",
    "PregraspInitialState",
    "apply_left_arm_down_v2_to_cfg",
    "left_arm_limit_margin_min_rad",
    "load_baseline_manifest",
    "selected_pregrasp_initial_state",
]
