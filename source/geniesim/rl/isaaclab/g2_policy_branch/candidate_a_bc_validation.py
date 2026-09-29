# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validation-only Candidate-A dataset contract for P0.6.

This schema is deliberately separate from every training/replay dataset.  It
stores one row per 50-Hz control epoch while preserving the source-owned
25-Hz camera frame identity and acquisition timestamp.  Planner targets and
phase labels are validation targets/metadata and are never members of the
student observation inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "g2_candidate_a_native_bc_validation_v1"
CONTROL_HZ = 50
RGBD_HZ = 25
PHYSICS_HZ = 500
CONTROL_DT_S = 1.0 / CONTROL_HZ
RGBD_DT_S = 1.0 / RGBD_HZ
PHYSICS_DT_S = 1.0 / PHYSICS_HZ
ACTION_SCHEMA_VERSION = "g2_high_level_metric_4d_open_v1"
FRAME_CONTRACT_VERSION = "g2_robot_root_se3_xyzw_v1"
PHASES = ("far_reach", "pregrasp", "near_contact")
STUDENT_FIELDS = (
    "right_wrist_rgb",
    "right_wrist_depth_m",
    "right_wrist_depth_valid",
    "ee_pose_robot_root_m_xyzw",
    "right_arm_joint_position_rad",
    "right_arm_joint_velocity_rad_s",
    "right_arm_joint_acceleration_rad_s2",
    "gripper_state_open",
    "previous_policy_action_4d_metric_root_m",
)


class CandidateAValidationError(ValueError):
    pass


def _finite(name: str, value: Any, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise CandidateAValidationError(
            f"{name} must be finite {np.dtype(dtype)} with shape {shape}"
        )
    return array


def _digest(name: str, value: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CandidateAValidationError(f"{name} must be a full SHA-256")
    return text


def _text(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise CandidateAValidationError(f"{name} must be non-empty")
    return text


@dataclass(frozen=True)
class CandidateAValidationRow:
    episode_id: str
    row_id: int
    control_timestamp_s: float
    camera_timestamp_s: float
    camera_frame_id: int
    right_wrist_rgb: Any
    right_wrist_depth_m: Any
    right_wrist_depth_valid: Any
    ee_pose_robot_root_m_xyzw: Any
    right_arm_joint_position_rad: Any
    right_arm_joint_velocity_rad_s: Any
    right_arm_joint_acceleration_rad_s2: Any
    gripper_state_open: float
    previous_policy_action_4d_metric_root_m: Sequence[float]
    candidate_a_expert_action_4d_metric_root_m: Sequence[float]
    ee_target_robot_root_m: Sequence[float]
    phase: str
    asset_fingerprint: str
    planner_fingerprint: str
    git_commit: str
    frame_contract_version: str = FRAME_CONTRACT_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION

    def validate(self) -> None:
        _text("episode_id", self.episode_id)
        if type(self.row_id) is not int or self.row_id < 0:
            raise CandidateAValidationError("row_id must be a nonnegative int")
        expected_time = self.row_id * CONTROL_DT_S
        if not math.isfinite(float(self.control_timestamp_s)) or not math.isclose(
            float(self.control_timestamp_s), expected_time, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise CandidateAValidationError("control timestamp must be the 50-Hz epoch clock")
        if not math.isfinite(float(self.camera_timestamp_s)) or self.camera_timestamp_s < 0.0:
            raise CandidateAValidationError("camera timestamp must be a real acquisition time")
        if type(self.camera_frame_id) is not int or self.camera_frame_id < 0:
            raise CandidateAValidationError("camera_frame_id must be a nonnegative int")
        rgb = np.asarray(self.right_wrist_rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (192, 256, 3):
            raise CandidateAValidationError("RGB must be uint8[192,256,3]")
        depth = _finite("right_wrist_depth_m", self.right_wrist_depth_m, (192, 256, 1), np.float32)
        valid = np.asarray(self.right_wrist_depth_valid, dtype=np.bool_)
        if valid.shape != depth.shape or np.any((~valid) & (depth != 0.0)):
            raise CandidateAValidationError("invalid depth pixels must be zero-filled")
        pose = _finite("ee_pose_robot_root_m_xyzw", self.ee_pose_robot_root_m_xyzw, (7,), np.float32)
        if not math.isclose(float(np.linalg.norm(pose[3:])), 1.0, rel_tol=0.0, abs_tol=1.0e-4):
            raise CandidateAValidationError("EE quaternion must be unit XYZW")
        for name in (
            "right_arm_joint_position_rad",
            "right_arm_joint_velocity_rad_s",
            "right_arm_joint_acceleration_rad_s2",
        ):
            _finite(name, getattr(self, name), (7,), np.float32)
        if float(self.gripper_state_open) != 1.0:
            raise CandidateAValidationError("P0.6 validation requires measured OPEN")
        previous = _finite(
            "previous_policy_action_4d_metric_root_m",
            self.previous_policy_action_4d_metric_root_m,
            (4,),
            np.float64,
        )
        action = _finite(
            "candidate_a_expert_action_4d_metric_root_m",
            self.candidate_a_expert_action_4d_metric_root_m,
            (4,),
            np.float64,
        )
        if float(previous[3]) != 0.0 or float(action[3]) != 0.0:
            raise CandidateAValidationError("validation actions must retain HOLD_OPEN=0")
        if float(np.linalg.norm(action[:3])) > 0.0045 + 1.0e-12:
            raise CandidateAValidationError("expert action exceeds 4.5-mm bound")
        target = _finite("ee_target_robot_root_m", self.ee_target_robot_root_m, (3,), np.float64)
        if not np.allclose(target, pose[:3].astype(np.float64) + action[:3], rtol=0.0, atol=2.0e-6):
            raise CandidateAValidationError("EE target does not equal measured EE plus expert delta")
        if self.phase not in PHASES:
            raise CandidateAValidationError(f"unknown validation phase: {self.phase}")
        _digest("asset_fingerprint", self.asset_fingerprint)
        _digest("planner_fingerprint", self.planner_fingerprint)
        _text("git_commit", self.git_commit)
        if self.frame_contract_version != FRAME_CONTRACT_VERSION:
            raise CandidateAValidationError("frame contract mismatch")
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise CandidateAValidationError("action schema mismatch")


class CandidateAValidationWriter:
    """Atomic writer for a complete bounded validation-only capture."""

    def __init__(self, destination: str | Path, *, manifest: Mapping[str, Any]) -> None:
        self.target = Path(destination)
        if self.target.exists():
            raise FileExistsError(f"refusing to overwrite validation set: {self.target}")
        required = {
            "episode_id",
            "asset_fingerprint",
            "planner_fingerprint",
            "git_commit",
        }
        if not required <= set(manifest):
            raise CandidateAValidationError("validation manifest provenance missing")
        if manifest.get("student_fields") != list(STUDENT_FIELDS):
            raise CandidateAValidationError("student observation inventory mismatch")
        if manifest.get("control_hz") != CONTROL_HZ or manifest.get("rgbd_hz") != RGBD_HZ or manifest.get("physics_hz") != PHYSICS_HZ:
            raise CandidateAValidationError("50/25/500-Hz contract mismatch")
        self.manifest = dict(manifest)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.target.name}.", suffix=".tmp", dir=self.target.parent
        )
        os.close(fd)
        self._temporary = Path(temporary)
        self._handle = h5py.File(self._temporary, "w")
        self._handle.attrs["schema"] = SCHEMA
        self._handle.attrs["manifest_json"] = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._rows = self._handle.create_group("rows")
        self._create_datasets()
        self._count = 0
        self._last_frame: int | None = None
        self._last_camera_timestamp: float | None = None
        self._phases: set[str] = set()
        self._reuse_count = 0

    def _dataset(self, name: str, shape: tuple[int, ...], dtype: Any) -> None:
        self._rows.create_dataset(
            name,
            shape=(0, *shape),
            maxshape=(None, *shape),
            chunks=(1, *shape),
            dtype=dtype,
        )

    def _create_datasets(self) -> None:
        text = h5py.string_dtype(encoding="utf-8")
        for name in (
            "episode_id",
            "phase",
            "asset_fingerprint",
            "planner_fingerprint",
            "git_commit",
            "frame_contract_version",
            "action_schema_version",
        ):
            self._dataset(name, (), text)
        self._dataset("row_id", (), np.int64)
        self._dataset("control_timestamp_s", (), np.float64)
        self._dataset("camera_timestamp_s", (), np.float64)
        self._dataset("camera_frame_id", (), np.int64)
        self._dataset("right_wrist_rgb", (192, 256, 3), np.uint8)
        self._dataset("right_wrist_depth_m", (192, 256, 1), np.float32)
        self._dataset("right_wrist_depth_valid", (192, 256, 1), np.bool_)
        self._dataset("ee_pose_robot_root_m_xyzw", (7,), np.float32)
        self._dataset("right_arm_joint_position_rad", (7,), np.float32)
        self._dataset("right_arm_joint_velocity_rad_s", (7,), np.float32)
        self._dataset("right_arm_joint_acceleration_rad_s2", (7,), np.float32)
        self._dataset("gripper_state_open", (1,), np.float32)
        self._dataset("previous_policy_action_4d_metric_root_m", (4,), np.float32)
        self._dataset("candidate_a_expert_action_4d_metric_root_m", (4,), np.float32)
        self._dataset("ee_target_robot_root_m", (3,), np.float32)

    @property
    def row_count(self) -> int:
        return self._count

    def append(self, row: CandidateAValidationRow) -> None:
        row.validate()
        if row.row_id != self._count:
            raise CandidateAValidationError("validation row IDs must be contiguous")
        if row.episode_id != self.manifest["episode_id"]:
            raise CandidateAValidationError("episode ID differs from manifest")
        if row.asset_fingerprint != self.manifest["asset_fingerprint"] or row.planner_fingerprint != self.manifest["planner_fingerprint"]:
            raise CandidateAValidationError("row provenance differs from manifest")
        if self._last_frame is not None and self._last_camera_timestamp is not None:
            if row.camera_frame_id == self._last_frame:
                if row.camera_timestamp_s != self._last_camera_timestamp:
                    raise CandidateAValidationError("reused RGB-D frame changed timestamp")
                self._reuse_count += 1
            elif row.camera_timestamp_s <= self._last_camera_timestamp:
                raise CandidateAValidationError("new RGB-D frame did not advance timestamp")
        self._last_frame = row.camera_frame_id
        self._last_camera_timestamp = row.camera_timestamp_s
        self._phases.add(row.phase)
        for dataset in self._rows.values():
            dataset.resize((self._count + 1, *dataset.shape[1:]))
        for name in (
            "episode_id",
            "phase",
            "asset_fingerprint",
            "planner_fingerprint",
            "git_commit",
            "frame_contract_version",
            "action_schema_version",
            "row_id",
            "control_timestamp_s",
            "camera_timestamp_s",
            "camera_frame_id",
            "right_wrist_rgb",
            "right_wrist_depth_m",
            "right_wrist_depth_valid",
            "ee_pose_robot_root_m_xyzw",
            "right_arm_joint_position_rad",
            "right_arm_joint_velocity_rad_s",
            "right_arm_joint_acceleration_rad_s2",
            "previous_policy_action_4d_metric_root_m",
            "candidate_a_expert_action_4d_metric_root_m",
            "ee_target_robot_root_m",
        ):
            self._rows[name][self._count] = getattr(row, name)
        self._rows["gripper_state_open"][self._count] = (row.gripper_state_open,)
        self._handle.flush()
        self._count += 1

    def publish(self) -> Path:
        if self._count <= 0:
            raise CandidateAValidationError("validation set is empty")
        if set(PHASES) != self._phases:
            raise CandidateAValidationError(
                f"validation phases incomplete: {sorted(self._phases)}"
            )
        if self._reuse_count <= 0:
            raise CandidateAValidationError("25-Hz RGB-D reuse was not observed at 50-Hz control")
        self._handle.attrs["row_count"] = self._count
        self._handle.attrs["camera_reuse_count"] = self._reuse_count
        self._handle.flush()
        self._handle.close()
        self._handle = None
        if self.target.exists():
            raise FileExistsError(f"refusing to overwrite validation set: {self.target}")
        os.replace(self._temporary, self.target)
        return self.target

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._temporary.unlink(missing_ok=True)


def validation_manifest(
    *,
    episode_id: str,
    asset_fingerprint: str,
    planner_fingerprint: str,
    git_commit: str,
    runner_source_sha256: str,
    source_freeze_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "validation_only": True,
        "training_eligible": False,
        "episode_id": _text("episode_id", episode_id),
        "control_hz": CONTROL_HZ,
        "rgbd_hz": RGBD_HZ,
        "physics_hz": PHYSICS_HZ,
        "control_dt_s": CONTROL_DT_S,
        "rgbd_dt_s": RGBD_DT_S,
        "physics_dt_s": PHYSICS_DT_S,
        "rgbd_timestamp_source": "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE",
        "rgbd_alignment": "LATEST_VALID_ACQUISITION_REFERENCED_BY_FRAME_ID",
        "frame": "robot_root",
        "quaternion_order": "xyzw",
        "frame_contract_version": FRAME_CONTRACT_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action": "[dx,dy,dz,HOLD_OPEN=0] metric robot_root",
        "maximum_action_norm_m": 0.0045,
        "student_fields": list(STUDENT_FIELDS),
        "student_privileged_input_count": 0,
        "label_fields": [
            "candidate_a_expert_action_4d_metric_root_m",
            "ee_target_robot_root_m",
            "phase",
        ],
        "asset_fingerprint": _digest("asset_fingerprint", asset_fingerprint),
        "planner_fingerprint": _digest("planner_fingerprint", planner_fingerprint),
        "git_commit": _text("git_commit", git_commit),
        "runner_source_sha256": _digest("runner_source_sha256", runner_source_sha256),
        "source_freeze_fingerprint": _digest(
            "source_freeze_fingerprint", source_freeze_fingerprint
        ),
        "legacy_e1_rows_included": False,
        "hidden_action_path_count": 0,
        "legacy_8d_path_count": 0,
        "silent_clipping_count": 0,
    }


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "CandidateAValidationError",
    "CandidateAValidationRow",
    "CandidateAValidationWriter",
    "CONTROL_HZ",
    "FRAME_CONTRACT_VERSION",
    "PHASES",
    "PHYSICS_HZ",
    "RGBD_HZ",
    "SCHEMA",
    "STUDENT_FIELDS",
    "validation_manifest",
]
