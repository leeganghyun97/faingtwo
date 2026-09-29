# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Candidate-A pregrasp sample binding for bounded CLOSE outcome collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
    CANDIDATE_A_SHA256,
    PregraspInitialState,
)


SAMPLE_SCHEMA = "g2_candidate_a_close_outcome_pregrasp_sample_v1"


class CloseOutcomeCollectionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value: Any, width: int, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise CloseOutcomeCollectionError(f"{label} must be finite [{width}]")
    return tuple(float(item) for item in array)


def load_close_outcome_sample(path: Path) -> PregraspInitialState:
    """Load one immutable state pair and verify it against source artifacts."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SAMPLE_SCHEMA:
        raise CloseOutcomeCollectionError("sample schema mismatch")
    if payload.get("candidate_a_sha256") != CANDIDATE_A_SHA256:
        raise CloseOutcomeCollectionError("sample is not Candidate A")
    if payload.get("contact_count") != 0 or payload.get("close_count") != 0:
        raise CloseOutcomeCollectionError("sample is not contact-free OPEN")
    if payload.get("forbidden_collision_count") != 0 or payload.get("safety_reject_count") != 0:
        raise CloseOutcomeCollectionError("sample failed source safety checks")
    hdf5_path = Path(str(payload["hdf5_path"]))
    report_path = Path(str(payload["report_path"]))
    for source, expected in (
        (hdf5_path, payload["hdf5_sha256"]),
        (report_path, payload["report_sha256"]),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise CloseOutcomeCollectionError(f"source drift: {source}")
    row_index = int(payload["hdf5_row_index"])
    with h5py.File(hdf5_path, "r") as source:
        manifest = json.loads(str(source.attrs["manifest_json"]))
        if manifest["source_hashes"]["asset"] != CANDIDATE_A_SHA256:
            raise CloseOutcomeCollectionError("HDF5 asset authority mismatch")
        rows = source["rows"]
        if not 0 <= row_index < int(rows["control_step"].shape[0]):
            raise CloseOutcomeCollectionError("row index outside source HDF5")
        q = _vector(rows["right_arm_joint_position_rad"][row_index], 7, "right q")
        qd = _vector(rows["right_arm_joint_velocity_rad_s"][row_index], 7, "right qd")
        ee = _vector(rows["ee_pose_robot_root_m_xyzw"][row_index], 7, "EE pose")
        if float(rows["gripper_state_open"][row_index, 0]) != 1.0:
            raise CloseOutcomeCollectionError("source row gripper is not OPEN")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cube = _vector(
        report["runtime_replan_input_telemetry"]["cube_center_root_m"], 3, "cube"
    ) + (0.0, 0.0, 0.0, 1.0)
    for label, observed, expected in (
        ("right q", q, payload["right_arm_q_rad"]),
        ("right qd", qd, payload["right_arm_qd_rad_s"]),
        ("EE pose", ee, payload["ee_pose_robot_root_m_xyzw"]),
        ("cube pose", cube, payload["cube_pose_robot_root_m_xyzw"]),
    ):
        if not np.allclose(observed, expected, atol=1.0e-7, rtol=0.0):
            raise CloseOutcomeCollectionError(f"manifest/source mismatch: {label}")
    return PregraspInitialState(
        sample_id=str(payload["sample_id"]),
        hdf5_path=str(hdf5_path),
        hdf5_sha256=str(payload["hdf5_sha256"]),
        report_path=str(report_path),
        report_sha256=str(payload["report_sha256"]),
        hdf5_row_index=row_index,
        source_control_step=int(payload["source_control_step"]),
        right_arm_q_rad=q,
        right_arm_qd_rad_s=qd,
        ee_pose_robot_root_m_xyzw=ee,
        cube_pose_robot_root_m_xyzw=cube,
        pad_to_cube_distance_m=float(payload["pad_to_cube_distance_m"]),
        ee_to_cube_distance_m=float(payload["ee_to_cube_distance_m"]),
        gripper_state="OPEN",
    )


__all__ = [
    "CloseOutcomeCollectionError",
    "SAMPLE_SCHEMA",
    "load_close_outcome_sample",
]
