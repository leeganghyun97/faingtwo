# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Future keyboard-v2 grasp telemetry schema (offline contract only).

The existing keyboard contact-free exports remain unchanged.  This schema is
for a later collector that records real grasp geometry and contact telemetry.
Pad-surface labels are impossible to validate without the existing,
source-owned ``g2_pad_surface_calibration_v1`` verifier; link origins and
midpoints are never accepted as substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
    G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
)
from geniesim.rl.isaaclab.g2_rebuild.pad_surface_calibration import (
    CALIBRATION_SCHEMA,
    sha256_file,
    verify_calibration_artifact,
)

from .keyboard_grasp_contract import canonical_keyboard_grasp_contract


KEYBOARD_V2_GEOMETRY_SCHEMA = "g2_keyboard_v2_grasp_geometry_rows_v1"
KEYBOARD_V2_CALIBRATION_BINDING_SCHEMA = (
    "g2_keyboard_v2_verified_pad_surface_calibration_binding_v1"
)
KEYBOARD_V2_CANDIDATE_A_TRANSITIVE_BINDING_SCHEMA = (
    "g2_keyboard_v2_candidate_a_calibration_transitive_binding_v1"
)
CANDIDATE_A_SHA256 = (
    "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
)
TIMESTAMP_FLOAT32_REPRESENTATION_TOLERANCE_S = 1.0e-6
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PAD_POSITION_FIELDS = (
    "left_pad_surface_position_root_m",
    "right_pad_surface_position_root_m",
)
PAD_QUATERNION_FIELDS = (
    "left_pad_surface_quat_root_xyzw",
    "right_pad_surface_quat_root_xyzw",
)
DISTAL_LINK_POSE_FIELDS = (
    "left_distal_link4_pose_robot_root_m_xyzw",
    "right_distal_link4_pose_robot_root_m_xyzw",
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        *PAD_POSITION_FIELDS,
        *PAD_QUATERNION_FIELDS,
        *DISTAL_LINK_POSE_FIELDS,
        "cube_center_robot_root_m",
        "passive_gripper_joint_position_rad",
        "contact_force_by_side_n",
        "timestamp_s",
        "control_step",
    }
)


class KeyboardV2GeometrySchemaError(ValueError):
    """Raised when keyboard geometry lacks physical/calibration authority."""


@dataclass(frozen=True, init=False)
class VerifiedPadSurfaceCalibrationBinding:
    """Unforgeable-by-constructor receipt from the canonical verifier."""

    calibration_id: str
    artifact_path: str
    artifact_sha256: str
    inner_link_name: str
    outer_link_name: str
    calibration_rows: int
    calibration_schema: str
    schema: str

    @classmethod
    def from_artifact(
        cls,
        artifact: str | Path,
        *,
        expected_asset_path: str | Path | None = None,
    ) -> "VerifiedPadSurfaceCalibrationBinding":
        path = Path(artifact).expanduser().resolve()
        try:
            calibration = verify_calibration_artifact(
                path,
                expected_asset_path=(
                    None
                    if expected_asset_path is None
                    else Path(expected_asset_path).expanduser().resolve()
                ),
            )
        except Exception as error:
            raise KeyboardV2GeometrySchemaError(
                f"pad-surface calibration verification failed: {error}"
            ) from error
        result = object.__new__(cls)
        values = {
            "calibration_id": calibration.calibration_id,
            "artifact_path": str(path),
            "artifact_sha256": sha256_file(path),
            "inner_link_name": calibration.inner_link_name,
            "outer_link_name": calibration.outer_link_name,
            "calibration_rows": calibration.calibration_rows,
            "calibration_schema": CALIBRATION_SCHEMA,
            "schema": KEYBOARD_V2_CALIBRATION_BINDING_SCHEMA,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        result._validate()
        return result

    def _validate(self) -> None:
        if self.schema != KEYBOARD_V2_CALIBRATION_BINDING_SCHEMA:
            raise KeyboardV2GeometrySchemaError("pad calibration binding schema mismatch")
        if self.calibration_schema != "g2_pad_surface_calibration_v1":
            raise KeyboardV2GeometrySchemaError("pad calibration artifact schema mismatch")
        if not self.calibration_id.strip():
            raise KeyboardV2GeometrySchemaError("pad calibration id is empty")
        if len(self.artifact_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.artifact_sha256
        ):
            raise KeyboardV2GeometrySchemaError("pad calibration artifact hash is invalid")
        if self.calibration_rows <= 0:
            raise KeyboardV2GeometrySchemaError("pad calibration has no authority rows")

    def payload(self) -> dict[str, Any]:
        self._validate()
        return {
            "schema": self.schema,
            "calibration_schema": self.calibration_schema,
            "calibration_id": self.calibration_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "inner_link_name": self.inner_link_name,
            "outer_link_name": self.outer_link_name,
            "calibration_rows": self.calibration_rows,
            "verification": "verify_calibration_artifact",
            "link_origin_or_midpoint_fallback": False,
            "candidate_a_transitive_binding_proven": False,
        }


@dataclass(frozen=True, init=False)
class CandidateATransitiveCalibrationBinding:
    """Explicit evidence that a verified calibration transfers to Candidate A.

    The canonical verifier may attest a Production-bound calibration.  That
    does not prove its offsets/frames are valid for Candidate A.  A separate,
    immutable attestation is therefore required before keyboard-v2 geometry
    rows can be admitted.
    """

    candidate_a_asset_sha256: str
    calibration_artifact_sha256: str
    attestation_path: str
    attestation_sha256: str
    schema: str

    @classmethod
    def from_attestation(
        cls,
        attestation: str | Path,
        *,
        calibration: VerifiedPadSurfaceCalibrationBinding,
    ) -> "CandidateATransitiveCalibrationBinding":
        if not isinstance(calibration, VerifiedPadSurfaceCalibrationBinding):
            raise KeyboardV2GeometrySchemaError(
                "Candidate-A transitive binding requires verified calibration"
            )
        path = Path(attestation).expanduser().resolve()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise KeyboardV2GeometrySchemaError(
                f"Candidate-A calibration attestation is unreadable: {error}"
            ) from error
        expected = {
            "schema": KEYBOARD_V2_CANDIDATE_A_TRANSITIVE_BINDING_SCHEMA,
            "status": "VERIFIED",
            "candidate_a_asset_sha256": CANDIDATE_A_SHA256,
            "calibration_artifact_sha256": calibration.artifact_sha256,
            "pad_frame_semantics_preserved": True,
            "offsets_validated_on_candidate_a": True,
        }
        if not isinstance(document, Mapping) or any(
            document.get(name) != value for name, value in expected.items()
        ):
            raise KeyboardV2GeometrySchemaError(
                "Candidate-A calibration transitive authority is unresolved"
            )
        result = object.__new__(cls)
        values = {
            "candidate_a_asset_sha256": CANDIDATE_A_SHA256,
            "calibration_artifact_sha256": calibration.artifact_sha256,
            "attestation_path": str(path),
            "attestation_sha256": sha256_file(path),
            "schema": KEYBOARD_V2_CANDIDATE_A_TRANSITIVE_BINDING_SCHEMA,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        result._validate(calibration)
        return result

    def _validate(self, calibration: VerifiedPadSurfaceCalibrationBinding) -> None:
        if self.schema != KEYBOARD_V2_CANDIDATE_A_TRANSITIVE_BINDING_SCHEMA:
            raise KeyboardV2GeometrySchemaError(
                "Candidate-A transitive binding schema mismatch"
            )
        if self.candidate_a_asset_sha256 != CANDIDATE_A_SHA256:
            raise KeyboardV2GeometrySchemaError("Candidate-A asset hash mismatch")
        if self.calibration_artifact_sha256 != calibration.artifact_sha256:
            raise KeyboardV2GeometrySchemaError(
                "Candidate-A binding names another calibration artifact"
            )
        if not _SHA256.fullmatch(self.attestation_sha256):
            raise KeyboardV2GeometrySchemaError(
                "Candidate-A transitive attestation hash is invalid"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_a_asset_sha256": self.candidate_a_asset_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "attestation_path": self.attestation_path,
            "attestation_sha256": self.attestation_sha256,
            "transitive_binding_status": "VERIFIED",
        }


def keyboard_v2_geometry_schema_contract() -> dict[str, Any]:
    common = canonical_keyboard_grasp_contract()
    return {
        "schema": KEYBOARD_V2_GEOMETRY_SCHEMA,
        "row_fields": {
            "left_pad_surface_position_root_m": "[N,3],m",
            "left_pad_surface_quat_root_xyzw": "[N,4],unit_xyzw",
            "right_pad_surface_position_root_m": "[N,3],m",
            "right_pad_surface_quat_root_xyzw": "[N,4],unit_xyzw",
            "cube_center_robot_root_m": "[N,3],m",
            "left_distal_link4_pose_robot_root_m_xyzw": "[N,7],m+unit_xyzw",
            "right_distal_link4_pose_robot_root_m_xyzw": "[N,7],m+unit_xyzw",
            "passive_gripper_joint_position_rad": "[N,7],rad",
            "contact_force_by_side_n": "[N,2],N,[left,right]",
            "timestamp_s": "[N],s,50Hz-control-clock",
            "control_step": "[N],int64,strictly-contiguous",
        },
        "passive_gripper_joint_order": list(
            G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES
        ),
        "contact_force_side_order": ["left", "right"],
        "frame": common.geometry_frame,
        "position_unit": common.position_unit,
        "angle_unit": common.angle_unit,
        "force_unit": common.force_unit,
        "quaternion_order": "xyzw",
        "control_hz": common.control_hz,
        "policy_dt_s": common.policy_dt_s,
        "calibration": {
            "schema": "g2_pad_surface_calibration_v1",
            "row_scoped_metadata": ["calibration_id", "calibration_artifact_sha256"],
            "verified_binding_required": True,
            "candidate_a_transitive_binding": "REQUIRED_CURRENTLY_UNRESOLVED",
            "candidate_a_asset_sha256": CANDIDATE_A_SHA256,
            "production_bound_calibration_alone_sufficient": False,
            "pad_geometry_labels_without_binding": "FORBIDDEN",
            "distal_link_origin_as_pad_surface": "FORBIDDEN",
            "midpoint_proxy_as_pad_surface": "FORBIDDEN",
        },
        "actor_input": {
            "cube_ground_truth": False,
            "contact_force": False,
            "pad_surface_ground_truth": False,
            "telemetry_role": "label_metric_replay_priority_only",
        },
        "timestamp_tolerance": {
            "absolute_seconds": TIMESTAMP_FLOAT32_REPRESENTATION_TOLERANCE_S,
            "purpose": "float32 representation only; not physical cadence relaxation",
        },
    }


@dataclass(frozen=True)
class KeyboardV2GeometryValidationReceipt:
    row_count: int
    control_step_first: int
    control_step_last: int
    calibration_id: str
    calibration_artifact_sha256: str
    candidate_a_asset_sha256: str
    candidate_a_transitive_attestation_sha256: str
    passive_joint_order: tuple[str, ...]
    row_content_sha256: str
    schema: str = KEYBOARD_V2_GEOMETRY_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_count": self.row_count,
            "control_step_first": self.control_step_first,
            "control_step_last": self.control_step_last,
            "calibration_id": self.calibration_id,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "candidate_a_asset_sha256": self.candidate_a_asset_sha256,
            "candidate_a_transitive_attestation_sha256": (
                self.candidate_a_transitive_attestation_sha256
            ),
            "passive_joint_order": list(self.passive_joint_order),
            "row_content_sha256": self.row_content_sha256,
            "pad_geometry_authority": "VERIFIED_CALIBRATION_ONLY",
            "candidate_a_transitive_binding": "VERIFIED",
            "cube_center_actor_input": False,
        }


def _pose(name: str, value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows, 7) or not bool(np.isfinite(array).all()):
        raise KeyboardV2GeometrySchemaError(f"{name} must be finite [N,7]")
    norms = np.linalg.norm(array[:, 3:7], axis=1)
    if not bool(np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-4)):
        raise KeyboardV2GeometrySchemaError(f"{name} quaternion must be unit XYZW")
    return array


def _position(name: str, value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows, 3) or not bool(np.isfinite(array).all()):
        raise KeyboardV2GeometrySchemaError(f"{name} must be finite [N,3] metres")
    return array


def _quaternion(name: str, value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (rows, 4) or not bool(np.isfinite(array).all()):
        raise KeyboardV2GeometrySchemaError(f"{name} must be finite [N,4]")
    norms = np.linalg.norm(array, axis=1)
    if not bool(np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-4)):
        raise KeyboardV2GeometrySchemaError(f"{name} quaternion must be unit XYZW")
    return array


def validate_keyboard_v2_geometry_rows(
    fields: Mapping[str, Any],
    *,
    calibration: VerifiedPadSurfaceCalibrationBinding,
    candidate_a_transitive_binding: CandidateATransitiveCalibrationBinding,
    calibration_id_per_row: np.ndarray,
    calibration_artifact_sha256_per_row: np.ndarray,
) -> KeyboardV2GeometryValidationReceipt:
    """Validate future collector rows without starting Isaac or writing data."""

    if not isinstance(calibration, VerifiedPadSurfaceCalibrationBinding):
        raise KeyboardV2GeometrySchemaError(
            "pad geometry labels require a verified g2_pad_surface_calibration_v1 binding"
        )
    calibration._validate()
    if not isinstance(
        candidate_a_transitive_binding, CandidateATransitiveCalibrationBinding
    ):
        raise KeyboardV2GeometrySchemaError(
            "Candidate-A transitive calibration binding is unresolved/required"
        )
    candidate_a_transitive_binding._validate(calibration)
    if not isinstance(fields, Mapping) or set(fields) != REQUIRED_ROW_FIELDS:
        missing = sorted(REQUIRED_ROW_FIELDS.difference(fields)) if isinstance(fields, Mapping) else []
        extra = sorted(set(fields).difference(REQUIRED_ROW_FIELDS)) if isinstance(fields, Mapping) else []
        raise KeyboardV2GeometrySchemaError(
            f"keyboard-v2 geometry row fields mismatch: missing={missing},extra={extra}"
        )
    steps = np.asarray(fields["control_step"])
    timestamps = np.asarray(fields["timestamp_s"], dtype=np.float64)
    if steps.ndim != 1 or steps.size <= 0 or steps.dtype.kind not in "iu":
        raise KeyboardV2GeometrySchemaError("control_step must be nonempty integer [N]")
    rows = int(steps.size)
    if not np.array_equal(np.diff(steps.astype(np.int64)), np.ones(rows - 1, np.int64)):
        raise KeyboardV2GeometrySchemaError("control_step must be strictly contiguous at 50 Hz")
    if timestamps.shape != (rows,) or not bool(np.isfinite(timestamps).all()):
        raise KeyboardV2GeometrySchemaError("timestamp_s must be finite [N]")
    expected_elapsed = (steps - steps[0]).astype(np.float64) * (
        1.0 / canonical_keyboard_grasp_contract().control_hz
    )
    if not np.allclose(
        timestamps - timestamps[0],
        expected_elapsed,
        rtol=0.0,
        atol=TIMESTAMP_FLOAT32_REPRESENTATION_TOLERANCE_S,
    ):
        raise KeyboardV2GeometrySchemaError("timestamp/control-step 50-Hz cadence mismatch")
    arrays: dict[str, np.ndarray] = {}
    for name in PAD_POSITION_FIELDS:
        arrays[name] = _position(name, fields[name], rows)
    for name in PAD_QUATERNION_FIELDS:
        arrays[name] = _quaternion(name, fields[name], rows)
    for name in DISTAL_LINK_POSE_FIELDS:
        arrays[name] = _pose(name, fields[name], rows)
    cube = np.asarray(fields["cube_center_robot_root_m"], dtype=np.float64)
    passive = np.asarray(fields["passive_gripper_joint_position_rad"], dtype=np.float64)
    force = np.asarray(fields["contact_force_by_side_n"], dtype=np.float64)
    if cube.shape != (rows, 3) or not bool(np.isfinite(cube).all()):
        raise KeyboardV2GeometrySchemaError("cube center must be finite [N,3] metres")
    expected_passive = len(G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES)
    if passive.shape != (rows, expected_passive) or not bool(np.isfinite(passive).all()):
        raise KeyboardV2GeometrySchemaError(
            f"passive gripper q must be finite [N,{expected_passive}] radians"
        )
    if force.shape != (rows, 2) or not bool(np.isfinite(force).all()) or bool((force < 0.0).any()):
        raise KeyboardV2GeometrySchemaError("left/right contact force must be nonnegative [N,2] N")
    ids = np.asarray(calibration_id_per_row)
    hashes = np.asarray(calibration_artifact_sha256_per_row)
    if ids.shape != (rows,) or hashes.shape != (rows,):
        raise KeyboardV2GeometrySchemaError("calibration id/hash must be stored for every row")
    if not bool(np.all(ids.astype(str) == calibration.calibration_id)):
        raise KeyboardV2GeometrySchemaError("row calibration ID differs from verified binding")
    if not bool(np.all(hashes.astype(str) == calibration.artifact_sha256)):
        raise KeyboardV2GeometrySchemaError("row calibration hash differs from verified binding")
    digest = hashlib.sha256()
    for name in sorted(REQUIRED_ROW_FIELDS):
        value = np.ascontiguousarray(np.asarray(fields[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    digest.update(json.dumps(calibration.payload(), sort_keys=True).encode("utf-8"))
    digest.update(
        json.dumps(
            candidate_a_transitive_binding.payload(), sort_keys=True
        ).encode("utf-8")
    )
    return KeyboardV2GeometryValidationReceipt(
        row_count=rows,
        control_step_first=int(steps[0]),
        control_step_last=int(steps[-1]),
        calibration_id=calibration.calibration_id,
        calibration_artifact_sha256=calibration.artifact_sha256,
        candidate_a_asset_sha256=CANDIDATE_A_SHA256,
        candidate_a_transitive_attestation_sha256=(
            candidate_a_transitive_binding.attestation_sha256
        ),
        passive_joint_order=tuple(G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES),
        row_content_sha256=digest.hexdigest(),
    )


__all__ = [
    "CANDIDATE_A_SHA256",
    "CandidateATransitiveCalibrationBinding",
    "DISTAL_LINK_POSE_FIELDS",
    "KEYBOARD_V2_CALIBRATION_BINDING_SCHEMA",
    "KEYBOARD_V2_CANDIDATE_A_TRANSITIVE_BINDING_SCHEMA",
    "KEYBOARD_V2_GEOMETRY_SCHEMA",
    "KeyboardV2GeometrySchemaError",
    "KeyboardV2GeometryValidationReceipt",
    "PAD_POSITION_FIELDS",
    "PAD_QUATERNION_FIELDS",
    "REQUIRED_ROW_FIELDS",
    "TIMESTAMP_FLOAT32_REPRESENTATION_TOLERANCE_S",
    "VerifiedPadSurfaceCalibrationBinding",
    "keyboard_v2_geometry_schema_contract",
    "validate_keyboard_v2_geometry_rows",
]
