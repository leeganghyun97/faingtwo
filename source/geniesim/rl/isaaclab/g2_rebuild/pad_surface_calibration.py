# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed producer for the G2 distal pad-surface calibration.

The runtime G2 USD contains rigid-body frames for the two distal links, but it
does not contain a uniquely named physical pad-surface frame.  Consequently a
mesh centroid, inertial origin, distal-link origin, or successful grasp pose is
not a calibration authority.  This module accepts only one of two external
authorities:

* an explicit OEM/CAD link-frame authority; or
* paired, multi-pose, measured 3-D pad contact points.

The result is the existing ``g2_pad_surface_calibration_v1`` runtime artifact.
It is written with exclusive-create semantics and binds both the exact runtime
asset and the exact authority input by SHA-256.  This is a CPU-only producer;
it never starts Isaac Sim and it never invents offsets from meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .g2_asset_dependency_manifest import (
    G2AssetDependencyError,
    verify_g2_asset_dependency_manifest,
)
from .isaac_runtime_adapter import (
    RIGHT_INNER_DISTAL_LINK,
    RIGHT_OUTER_DISTAL_LINK,
    PadSurfaceCalibration,
)


CALIBRATION_SCHEMA = "g2_pad_surface_calibration_v1"
PRODUCER_SCHEMA = "g2_pad_surface_calibration_producer_v1"
MEASURED_CONTACT_SCHEMA = "g2_pad_surface_contact_measurements_v1"
CAD_AUTHORITY_SCHEMA = "g2_pad_surface_cad_authority_v1"
EXPECTED_G2_ASSET_SHA256 = (
    "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
)
LINK_NAMES = (RIGHT_INNER_DISTAL_LINK, RIGHT_OUTER_DISTAL_LINK)
PHYSX_CONTACT_AUTHORITY_KIND = "SIM601_PRODUCTION_VALIDATED"
RAW_PHYSX_EVIDENCE_SCHEMA = "g2_pad_surface_physx_raw_contact_evidence_v2"
RAW_PHYSX_VALIDATION_SCHEMA = "g2_pad_surface_physx_raw_contact_validation_v2"
TARGET_PHYSX_RUNTIME = "IsaacLab_v3.0.0-beta2-patch1+Isaac_Sim_6.0.1"
RAW_PHYSX_CONTACT_API = (
    "isaaclab.ContactSensor.track_contact_points/"
    "RigidContactView.get_contact_data+get_raw_contact_data+"
    "get_other_actor_paths_from_ids"
)


class PadSurfaceCalibrationError(RuntimeError):
    """Raised when an authority cannot identify the physical pad surfaces."""


@dataclass(frozen=True)
class ContactFitThresholds:
    """Recorded acceptance thresholds for multi-pose contact fitting."""

    minimum_rows_per_link: int = 6
    minimum_unique_poses: int = 3
    minimum_translation_span_m: float = 0.005
    minimum_rotation_span_rad: float = math.radians(5.0)
    maximum_rms_residual_m: float = 0.001
    maximum_residual_m: float = 0.003
    minimum_design_singular_value: float = 0.5
    maximum_design_condition_number: float = 10.0
    quaternion_norm_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.minimum_rows_per_link < 3:
            raise ValueError("minimum_rows_per_link must be at least 3")
        if self.minimum_unique_poses < 2:
            raise ValueError("minimum_unique_poses must be at least 2")
        positive = (
            self.minimum_translation_span_m,
            self.minimum_rotation_span_rad,
            self.maximum_rms_residual_m,
            self.maximum_residual_m,
            self.minimum_design_singular_value,
            self.maximum_design_condition_number,
            self.quaternion_norm_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("contact-fit thresholds must be finite and positive")
        if self.maximum_rms_residual_m > self.maximum_residual_m:
            raise ValueError("RMS residual limit cannot exceed maximum residual limit")

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_rows_per_link": self.minimum_rows_per_link,
            "minimum_unique_poses": self.minimum_unique_poses,
            "minimum_translation_span_m": self.minimum_translation_span_m,
            "minimum_rotation_span_rad": self.minimum_rotation_span_rad,
            "maximum_rms_residual_m": self.maximum_rms_residual_m,
            "maximum_residual_m": self.maximum_residual_m,
            "minimum_design_singular_value": self.minimum_design_singular_value,
            "maximum_design_condition_number": self.maximum_design_condition_number,
            "quaternion_norm_tolerance": self.quaternion_norm_tolerance,
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path: Path, *, label: str) -> tuple[Path, bytes, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PadSurfaceCalibrationError(f"{label}_missing:{resolved}")
    data = resolved.read_bytes()
    return resolved, data, sha256_bytes(data)


def _json_mapping(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PadSurfaceCalibrationError(f"{label}_invalid_json:{error}") from error
    if not isinstance(value, Mapping):
        raise PadSurfaceCalibrationError(f"{label}_must_be_json_object")
    return value


def _finite_vector(value: Any, *, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not bool(np.isfinite(array).all()):
        raise PadSurfaceCalibrationError(f"{label}_must_be_finite_vector_{size}")
    return array


def _quaternion_rotation_xyzw(
    value: Any, *, label: str, norm_tolerance: float
) -> np.ndarray:
    quaternion = _finite_vector(value, size=4, label=label)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12 or abs(norm - 1.0) > norm_tolerance:
        raise PadSurfaceCalibrationError(
            f"{label}_not_unit_xyzw:norm={norm:.12g}"
        )
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _maximum_pairwise_distance(values: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            maximum = max(maximum, float(np.linalg.norm(left - right)))
    return maximum


def _maximum_rotation_span(rotations: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for index, left in enumerate(rotations):
        for right in rotations[index + 1 :]:
            maximum = max(maximum, _rotation_distance(left, right))
    return maximum


@dataclass(frozen=True)
class _ContactRow:
    pose_id: str
    link_name: str
    position_world_m: np.ndarray
    rotation_world_from_link: np.ndarray
    measured_surface_point_world_m: np.ndarray


def _parse_contact_rows(
    authority: Mapping[str, Any], thresholds: ContactFitThresholds
) -> dict[str, list[_ContactRow]]:
    required_root = {
        "authority_id": str,
        "authority_kind": str,
        "coordinate_frame": str,
        "quaternion_order": str,
        "units": str,
        "rows": list,
    }
    if authority.get("schema") != MEASURED_CONTACT_SCHEMA:
        raise PadSurfaceCalibrationError("contact_authority_schema_invalid")
    for field, expected_type in required_root.items():
        if field not in authority or not isinstance(authority[field], expected_type):
            raise PadSurfaceCalibrationError(f"contact_authority_field_invalid:{field}")
    if not str(authority["authority_id"]).strip():
        raise PadSurfaceCalibrationError("contact_authority_id_empty")
    if authority["authority_kind"] not in {
        "CALIBRATED_CONTACT_PROBE",
        "COORDINATE_MEASURING_MACHINE",
        PHYSX_CONTACT_AUTHORITY_KIND,
    }:
        raise PadSurfaceCalibrationError("contact_authority_kind_not_approved")
    if authority["coordinate_frame"] != "world":
        raise PadSurfaceCalibrationError("contact_coordinate_frame_must_be_world")
    if authority["quaternion_order"] != "xyzw":
        raise PadSurfaceCalibrationError("contact_quaternion_order_must_be_xyzw")
    if authority["units"] != "m":
        raise PadSurfaceCalibrationError("contact_units_must_be_m")

    grouped = {name: [] for name in LINK_NAMES}
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(authority["rows"]):
        if not isinstance(raw, Mapping):
            raise PadSurfaceCalibrationError(f"contact_row_{index}_not_object")
        if raw.get("measurement_valid") is not True:
            raise PadSurfaceCalibrationError(f"contact_row_{index}_not_attested_valid")
        pose_id = str(raw.get("pose_id", "")).strip()
        link_name = str(raw.get("link_name", "")).strip()
        if not pose_id:
            raise PadSurfaceCalibrationError(f"contact_row_{index}_pose_id_empty")
        if link_name not in grouped:
            raise PadSurfaceCalibrationError(
                f"contact_row_{index}_unexpected_link:{link_name}"
            )
        key = (pose_id, link_name)
        if key in seen:
            raise PadSurfaceCalibrationError(
                f"contact_row_duplicate_pose_link:{pose_id}:{link_name}"
            )
        seen.add(key)
        grouped[link_name].append(
            _ContactRow(
                pose_id=pose_id,
                link_name=link_name,
                position_world_m=_finite_vector(
                    raw.get("link_position_world_m"),
                    size=3,
                    label=f"contact_row_{index}_link_position_world_m",
                ),
                rotation_world_from_link=_quaternion_rotation_xyzw(
                    raw.get("link_quaternion_world_xyzw"),
                    label=f"contact_row_{index}_link_quaternion_world_xyzw",
                    norm_tolerance=thresholds.quaternion_norm_tolerance,
                ),
                measured_surface_point_world_m=_finite_vector(
                    raw.get("measured_pad_surface_point_world_m"),
                    size=3,
                    label=f"contact_row_{index}_measured_pad_surface_point_world_m",
                ),
            )
        )

    pose_sets = {name: {row.pose_id for row in rows} for name, rows in grouped.items()}
    if pose_sets[LINK_NAMES[0]] != pose_sets[LINK_NAMES[1]]:
        raise PadSurfaceCalibrationError("contact_pose_pairs_incomplete_or_ambiguous")
    for link_name, rows in grouped.items():
        if len(rows) < thresholds.minimum_rows_per_link:
            raise PadSurfaceCalibrationError(
                f"contact_rows_insufficient:{link_name}:{len(rows)}"
            )
        if len({row.pose_id for row in rows}) < thresholds.minimum_unique_poses:
            raise PadSurfaceCalibrationError(
                f"contact_unique_poses_insufficient:{link_name}"
            )
    return grouped


def _fit_link_offset(
    rows: Sequence[_ContactRow], thresholds: ContactFitThresholds
) -> tuple[np.ndarray, dict[str, Any]]:
    positions = [row.position_world_m for row in rows]
    rotations = [row.rotation_world_from_link for row in rows]
    translation_span = _maximum_pairwise_distance(positions)
    rotation_span = _maximum_rotation_span(rotations)
    if translation_span < thresholds.minimum_translation_span_m:
        raise PadSurfaceCalibrationError(
            f"contact_pose_translation_span_insufficient:{translation_span:.12g}"
        )
    if rotation_span < thresholds.minimum_rotation_span_rad:
        raise PadSurfaceCalibrationError(
            f"contact_pose_rotation_span_insufficient:{rotation_span:.12g}"
        )

    design = np.concatenate(rotations, axis=0)
    observed = np.concatenate(
        [
            row.measured_surface_point_world_m - row.position_world_m
            for row in rows
        ],
        axis=0,
    )
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition_number = float(singular_values[0] / singular_values[-1])
    if rank != 3 or singular_values[-1] < thresholds.minimum_design_singular_value:
        raise PadSurfaceCalibrationError(
            "contact_fit_rank_deficient:"
            f"rank={rank}:minimum_singular={singular_values[-1]:.12g}"
        )
    if condition_number > thresholds.maximum_design_condition_number:
        raise PadSurfaceCalibrationError(
            f"contact_fit_ill_conditioned:{condition_number:.12g}"
        )
    offset, _, solved_rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if int(solved_rank) != 3 or not bool(np.isfinite(offset).all()):
        raise PadSurfaceCalibrationError("contact_fit_solution_invalid")

    residuals = np.asarray(
        [
            np.linalg.norm(
                row.position_world_m
                + row.rotation_world_from_link @ offset
                - row.measured_surface_point_world_m
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    maximum = float(residuals.max())
    if rms > thresholds.maximum_rms_residual_m:
        raise PadSurfaceCalibrationError(
            f"contact_fit_rms_residual_exceeded:{rms:.12g}"
        )
    if maximum > thresholds.maximum_residual_m:
        raise PadSurfaceCalibrationError(
            f"contact_fit_max_residual_exceeded:{maximum:.12g}"
        )
    if float(np.linalg.norm(offset)) <= 1.0e-6:
        raise PadSurfaceCalibrationError("contact_fit_zero_link_origin_offset_forbidden")
    return offset, {
        "rows": len(rows),
        "unique_pose_ids": len({row.pose_id for row in rows}),
        "design_rank": rank,
        "design_singular_values": singular_values.tolist(),
        "design_condition_number": condition_number,
        "translation_span_m": translation_span,
        "rotation_span_rad": rotation_span,
        "residual_rms_m": rms,
        "residual_p95_m": float(np.percentile(residuals, 95.0)),
        "residual_max_m": maximum,
    }


def _validate_asset_snapshot(
    asset: Path, expected_asset_sha256: str
) -> tuple[Path, bytes, str]:
    if len(expected_asset_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_asset_sha256.lower()
    ):
        raise PadSurfaceCalibrationError("expected_asset_sha256_invalid")
    resolved, data, digest = _snapshot(asset, label="source_asset")
    if digest != expected_asset_sha256.lower():
        raise PadSurfaceCalibrationError(
            f"source_asset_sha256_mismatch:expected={expected_asset_sha256}:observed={digest}"
        )
    return resolved, data, digest


def _validate_physx_raw_lineage(
    authority: Mapping[str, Any],
    *,
    asset_path: Path,
    asset_sha256: str,
) -> tuple[Path, bytes, str, Mapping[str, Any]]:
    """Re-open and independently reconstruct a PhysX contact authority.

    A hash string in an authority document is not sufficient lineage.  For the
    live PhysX authority kind, the producer re-opens the immutable six-buffer
    sidecar, re-runs the CPU validator, rebuilds the normalized authority rows,
    and requires exact equality before fitting an offset.  Imports are local to
    avoid the intentional dependency from the raw validator back to this
    module's error/value contracts.
    """

    provenance = authority.get("capture_provenance")
    if not isinstance(provenance, Mapping):
        raise PadSurfaceCalibrationError("physx_contact_provenance_missing")
    expected_fields = {
        "contact_point_api": RAW_PHYSX_CONTACT_API,
        "independent_validation_schema": RAW_PHYSX_VALIDATION_SCHEMA,
        "runtime": TARGET_PHYSX_RUNTIME,
        "production_runtime": TARGET_PHYSX_RUNTIME,
        "production_runtime_match": True,
        "physics_backend": "PhysX",
    }
    for field, expected in expected_fields.items():
        if provenance.get(field) != expected:
            raise PadSurfaceCalibrationError(
                f"physx_contact_provenance_invalid:{field}"
            )
    if provenance.get("pending_validations") != []:
        raise PadSurfaceCalibrationError("physx_contact_provenance_pending")
    try:
        verify_g2_asset_dependency_manifest(
            provenance.get("asset_dependency_manifest", {}),
            asset_path=asset_path,
        )
    except G2AssetDependencyError as error:
        raise PadSurfaceCalibrationError(
            f"physx_contact_asset_dependency_manifest_invalid:{error}"
        ) from error
    raw_path_value = provenance.get("raw_evidence_path")
    raw_sha_value = provenance.get("raw_evidence_sha256")
    if not isinstance(raw_path_value, str) or not raw_path_value.strip():
        raise PadSurfaceCalibrationError("physx_raw_evidence_path_invalid")
    if not isinstance(raw_sha_value, str) or len(raw_sha_value) != 64:
        raise PadSurfaceCalibrationError("physx_raw_evidence_sha256_invalid")
    raw_path, raw_bytes, raw_sha = _snapshot(
        Path(raw_path_value), label="physx_raw_evidence"
    )
    if raw_sha != raw_sha_value.lower():
        raise PadSurfaceCalibrationError("physx_raw_evidence_sha256_mismatch")
    raw = _json_mapping(raw_bytes, label="physx_raw_evidence")
    if raw.get("schema") != RAW_PHYSX_EVIDENCE_SCHEMA:
        raise PadSurfaceCalibrationError("physx_raw_evidence_schema_invalid")
    if raw.get("source_asset_sha256") != asset_sha256:
        raise PadSurfaceCalibrationError("physx_raw_evidence_asset_mismatch")
    if raw.get("contact_point_api") != RAW_PHYSX_CONTACT_API:
        raise PadSurfaceCalibrationError("physx_raw_evidence_api_mismatch")
    raw_provenance = raw.get("capture_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise PadSurfaceCalibrationError("physx_raw_capture_provenance_missing")
    for field in (
        "runtime",
        "production_runtime",
        "production_runtime_match",
        "physics_backend",
        "contact_point_api",
    ):
        if raw_provenance.get(field) != expected_fields[field]:
            raise PadSurfaceCalibrationError(
                f"physx_raw_capture_provenance_invalid:{field}"
            )
    if raw.get("acquisition_id") != authority.get("authority_id"):
        raise PadSurfaceCalibrationError("physx_raw_authority_identity_mismatch")

    from .pad_contact_acquisition import build_physx_contact_authority
    from .pad_contact_evidence import validate_raw_contact_evidence

    raw_rows, validation = validate_raw_contact_evidence(raw)
    if validation.get("status") != "PASS":
        raise PadSurfaceCalibrationError("physx_raw_independent_validation_failed")
    rebuilt = build_physx_contact_authority(
        authority_id=str(authority["authority_id"]),
        asset_path=asset_path,
        asset_sha256=asset_sha256,
        rows=raw_rows,
        capture_provenance=provenance,
        probe=raw.get("probe", {}),
        safety=raw.get("safety", {}),
    )
    for field in (
        "schema",
        "authority_id",
        "authority_kind",
        "coordinate_frame",
        "quaternion_order",
        "units",
        "source_asset_sha256",
        "capture_provenance",
        "probe",
        "safety",
        "rows",
    ):
        if authority.get(field) != rebuilt.get(field):
            raise PadSurfaceCalibrationError(
                f"physx_authority_raw_reconstruction_mismatch:{field}"
            )
    return raw_path, raw_bytes, raw_sha, validation


def _base_artifact(
    *,
    calibration_id: str,
    authority_path: Path,
    authority_sha256: str,
    authority_schema: str,
    asset_path: Path,
    asset_sha256: str,
    inner_offset: np.ndarray,
    outer_offset: np.ndarray,
    calibration_rows: int,
    method: str,
) -> dict[str, Any]:
    identifier = calibration_id.strip()
    if not identifier:
        raise PadSurfaceCalibrationError("calibration_id_empty")
    artifact = {
        "schema": CALIBRATION_SCHEMA,
        "producer_schema": PRODUCER_SCHEMA,
        "calibration_id": identifier,
        "method": method,
        "inner_link_name": RIGHT_INNER_DISTAL_LINK,
        "outer_link_name": RIGHT_OUTER_DISTAL_LINK,
        "inner_link_local_offset_m": inner_offset.tolist(),
        "outer_link_local_offset_m": outer_offset.tolist(),
        # The established runtime field binds the exact external authority.
        "source_artifact_sha256": authority_sha256,
        "calibration_rows": int(calibration_rows),
        "source_authority": {
            "path": str(authority_path),
            "sha256": authority_sha256,
            "schema": authority_schema,
        },
        "source_asset": {
            "path": str(asset_path),
            "sha256": asset_sha256,
        },
        "coordinate_contract": {
            "offset_units": "m",
            "offset_frame": "named_distal_link_local",
            "world_quaternion_order_for_measured_fit": "xyzw",
            "mesh_centroid_used": False,
            "link_origin_fallback_used": False,
            "inertial_origin_used": False,
        },
    }
    # Exercise the same core value contract used by live runtime loading.
    PadSurfaceCalibration(
        inner_link_local_offset_m=tuple(float(v) for v in inner_offset),
        outer_link_local_offset_m=tuple(float(v) for v in outer_offset),
        calibration_id=identifier,
        source_artifact_sha256=authority_sha256,
        calibration_rows=int(calibration_rows),
    )
    return artifact


def build_from_measured_contacts(
    *,
    authority_path: Path,
    asset_path: Path,
    calibration_id: str,
    expected_asset_sha256: str = EXPECTED_G2_ASSET_SHA256,
    thresholds: ContactFitThresholds | None = None,
) -> dict[str, Any]:
    """Fit two link-local pad centers from paired measured 3-D contacts."""

    limits = thresholds or ContactFitThresholds()
    authority_path, authority_bytes, authority_sha = _snapshot(
        authority_path, label="contact_authority"
    )
    authority = _json_mapping(authority_bytes, label="contact_authority")
    asset_path, asset_bytes, asset_sha = _validate_asset_snapshot(
        asset_path, expected_asset_sha256
    )
    declared_asset_sha = authority.get("source_asset_sha256")
    if declared_asset_sha != asset_sha:
        raise PadSurfaceCalibrationError(
            "contact_authority_source_asset_sha256_mismatch"
        )
    raw_lineage: tuple[Path, bytes, str, Mapping[str, Any]] | None = None
    if authority.get("authority_kind") == PHYSX_CONTACT_AUTHORITY_KIND:
        raw_lineage = _validate_physx_raw_lineage(
            authority,
            asset_path=asset_path,
            asset_sha256=asset_sha,
        )
    grouped = _parse_contact_rows(authority, limits)
    offsets: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for link_name in LINK_NAMES:
        offsets[link_name], metrics[link_name] = _fit_link_offset(
            grouped[link_name], limits
        )
    if authority_path.read_bytes() != authority_bytes:
        raise PadSurfaceCalibrationError("contact_authority_changed_during_fit")
    if asset_path.read_bytes() != asset_bytes:
        raise PadSurfaceCalibrationError("source_asset_changed_during_fit")
    if raw_lineage is not None:
        raw_path, raw_bytes, _, _ = raw_lineage
        if raw_path.read_bytes() != raw_bytes:
            raise PadSurfaceCalibrationError("physx_raw_evidence_changed_during_fit")

    artifact = _base_artifact(
        calibration_id=calibration_id,
        authority_path=authority_path,
        authority_sha256=authority_sha,
        authority_schema=MEASURED_CONTACT_SCHEMA,
        asset_path=asset_path,
        asset_sha256=asset_sha,
        inner_offset=offsets[RIGHT_INNER_DISTAL_LINK],
        outer_offset=offsets[RIGHT_OUTER_DISTAL_LINK],
        calibration_rows=sum(len(rows) for rows in grouped.values()),
        method="MULTI_POSE_MEASURED_CONTACT",
    )
    artifact["authority_id"] = authority["authority_id"]
    artifact["authority_kind"] = authority["authority_kind"]
    artifact["calibration_rows_by_link"] = {
        name: len(grouped[name]) for name in LINK_NAMES
    }
    artifact["fit_thresholds"] = limits.as_dict()
    artifact["fit_validation"] = {
        "applicable": True,
        "accepted": True,
        "links": metrics,
    }
    if raw_lineage is not None:
        raw_path, _, raw_sha, raw_validation = raw_lineage
        source_freeze_manifest = authority["capture_provenance"].get(
            "source_freeze_manifest"
        )
        if not isinstance(source_freeze_manifest, Mapping):
            raise PadSurfaceCalibrationError(
                "calibration_source_freeze_manifest_missing"
            )
        artifact["raw_physx_evidence"] = {
            "path": str(raw_path),
            "sha256": raw_sha,
            "schema": RAW_PHYSX_EVIDENCE_SCHEMA,
            "validation_schema": RAW_PHYSX_VALIDATION_SCHEMA,
            "runtime": TARGET_PHYSX_RUNTIME,
            "validation_status": raw_validation.get("status"),
            "asset_dependency_manifest_sha256": authority[
                "capture_provenance"
            ]["asset_dependency_manifest"]["manifest_sha256"],
            "source_freeze_manifest_file_sha256": source_freeze_manifest[
                "file_sha256"
            ],
            "source_freeze_manifest_document_sha256": source_freeze_manifest[
                "document_sha256"
            ],
        }
        artifact["source_freeze_manifest"] = dict(source_freeze_manifest)
    return artifact


def build_from_cad_authority(
    *,
    authority_path: Path,
    asset_path: Path,
    calibration_id: str,
    expected_asset_sha256: str = EXPECTED_G2_ASSET_SHA256,
) -> dict[str, Any]:
    """Build a calibration from an explicit OEM/CAD distal-link authority."""

    authority_path, authority_bytes, authority_sha = _snapshot(
        authority_path, label="cad_authority"
    )
    authority = _json_mapping(authority_bytes, label="cad_authority")
    asset_path, asset_bytes, asset_sha = _validate_asset_snapshot(
        asset_path, expected_asset_sha256
    )
    required = {
        "authority_id": str,
        "authority_kind": str,
        "issuer": str,
        "document_id": str,
        "revision": str,
        "coordinate_frame": str,
        "units": str,
        "source_asset_sha256": str,
        "links": dict,
    }
    if authority.get("schema") != CAD_AUTHORITY_SCHEMA:
        raise PadSurfaceCalibrationError("cad_authority_schema_invalid")
    for field, expected_type in required.items():
        if field not in authority or not isinstance(authority[field], expected_type):
            raise PadSurfaceCalibrationError(f"cad_authority_field_invalid:{field}")
    for field in ("authority_id", "issuer", "document_id", "revision"):
        if not str(authority[field]).strip():
            raise PadSurfaceCalibrationError(f"cad_authority_field_empty:{field}")
    if authority["authority_kind"] not in {"OEM_CAD", "OEM_DRAWING"}:
        raise PadSurfaceCalibrationError("cad_authority_kind_not_approved")
    if authority["coordinate_frame"] != "named_distal_link_local":
        raise PadSurfaceCalibrationError("cad_coordinate_frame_invalid")
    if authority["units"] != "m":
        raise PadSurfaceCalibrationError("cad_units_must_be_m")
    if authority["source_asset_sha256"] != asset_sha:
        raise PadSurfaceCalibrationError("cad_source_asset_sha256_mismatch")
    links = authority["links"]
    if set(links) != set(LINK_NAMES):
        raise PadSurfaceCalibrationError("cad_link_set_must_match_exact_runtime_links")
    offsets: dict[str, np.ndarray] = {}
    for link_name in LINK_NAMES:
        entry = links[link_name]
        if not isinstance(entry, Mapping):
            raise PadSurfaceCalibrationError(f"cad_link_entry_invalid:{link_name}")
        if entry.get("frame") != f"{link_name}:local":
            raise PadSurfaceCalibrationError(f"cad_link_frame_invalid:{link_name}")
        offsets[link_name] = _finite_vector(
            entry.get("pad_surface_center_local_m"),
            size=3,
            label=f"cad_pad_surface_center:{link_name}",
        )
        if float(np.linalg.norm(offsets[link_name])) <= 1.0e-6:
            raise PadSurfaceCalibrationError(
                f"cad_zero_link_origin_offset_forbidden:{link_name}"
            )
    if authority_path.read_bytes() != authority_bytes:
        raise PadSurfaceCalibrationError("cad_authority_changed_during_build")
    if asset_path.read_bytes() != asset_bytes:
        raise PadSurfaceCalibrationError("source_asset_changed_during_build")

    artifact = _base_artifact(
        calibration_id=calibration_id,
        authority_path=authority_path,
        authority_sha256=authority_sha,
        authority_schema=CAD_AUTHORITY_SCHEMA,
        asset_path=asset_path,
        asset_sha256=asset_sha,
        inner_offset=offsets[RIGHT_INNER_DISTAL_LINK],
        outer_offset=offsets[RIGHT_OUTER_DISTAL_LINK],
        calibration_rows=2,
        method="OEM_CAD_DIRECT_AUTHORITY",
    )
    artifact["authority_id"] = authority["authority_id"]
    artifact["authority_kind"] = authority["authority_kind"]
    artifact["authority_document"] = {
        "issuer": authority["issuer"],
        "document_id": authority["document_id"],
        "revision": authority["revision"],
    }
    artifact["calibration_rows_by_link"] = {name: 1 for name in LINK_NAMES}
    artifact["fit_validation"] = {
        "applicable": False,
        "accepted": None,
        "reason": "explicit_oem_cad_link_frame_authority_no_statistical_fit",
        "links": {
            name: {
                "rows": 1,
                "residual_rms_m": None,
                "residual_p95_m": None,
                "residual_max_m": None,
            }
            for name in LINK_NAMES
        },
    }
    return artifact


def verify_calibration_artifact(
    path: Path,
    *,
    expected_asset_path: Path | None = None,
    expected_asset_sha256: str = EXPECTED_G2_ASSET_SHA256,
) -> PadSurfaceCalibration:
    """Re-open and fully verify a runtime pad-surface calibration artifact.

    The small :class:`PadSurfaceCalibration` value object is intentionally not
    sufficient evidence by itself: offsets copied from another robot asset can
    have the right shape and still be unsafe.  This verifier binds the runtime
    values to the exact G2 asset, distal-link names, approved authority type,
    coordinate contract, and unchanged external authority bytes.  The
    calibration is rebuilt from that authority and the identified offsets are
    compared before a runtime value is returned.
    """

    artifact_path, artifact_bytes, _ = _snapshot(path, label="calibration_artifact")
    document = _json_mapping(artifact_bytes, label="calibration_artifact")
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise PadSurfaceCalibrationError("calibration_schema_invalid")
    if document.get("producer_schema") != PRODUCER_SCHEMA:
        raise PadSurfaceCalibrationError("calibration_producer_schema_invalid")

    method_to_authority_schema = {
        "MULTI_POSE_MEASURED_CONTACT": MEASURED_CONTACT_SCHEMA,
        "OEM_CAD_DIRECT_AUTHORITY": CAD_AUTHORITY_SCHEMA,
    }
    method = document.get("method")
    if method not in method_to_authority_schema:
        raise PadSurfaceCalibrationError("calibration_method_not_approved")
    if document.get("inner_link_name") != RIGHT_INNER_DISTAL_LINK:
        raise PadSurfaceCalibrationError("calibration_inner_link_mismatch")
    if document.get("outer_link_name") != RIGHT_OUTER_DISTAL_LINK:
        raise PadSurfaceCalibrationError("calibration_outer_link_mismatch")

    coordinate = document.get("coordinate_contract")
    expected_coordinate = {
        "offset_units": "m",
        "offset_frame": "named_distal_link_local",
        "world_quaternion_order_for_measured_fit": "xyzw",
        "mesh_centroid_used": False,
        "link_origin_fallback_used": False,
        "inertial_origin_used": False,
    }
    if coordinate != expected_coordinate:
        raise PadSurfaceCalibrationError("calibration_coordinate_contract_invalid")

    source_asset = document.get("source_asset")
    if not isinstance(source_asset, Mapping):
        raise PadSurfaceCalibrationError("calibration_source_asset_missing")
    declared_asset_path = source_asset.get("path")
    declared_asset_sha = source_asset.get("sha256")
    if not isinstance(declared_asset_path, str) or not declared_asset_path.strip():
        raise PadSurfaceCalibrationError("calibration_source_asset_path_invalid")
    if declared_asset_sha != expected_asset_sha256.lower():
        raise PadSurfaceCalibrationError("calibration_source_asset_sha256_mismatch")
    runtime_asset_path = (
        Path(expected_asset_path).expanduser().resolve()
        if expected_asset_path is not None
        else Path(declared_asset_path).expanduser().resolve()
    )
    _, _, runtime_asset_sha = _validate_asset_snapshot(
        runtime_asset_path, expected_asset_sha256
    )
    if runtime_asset_sha != declared_asset_sha:
        raise PadSurfaceCalibrationError("calibration_not_bound_to_runtime_asset")

    source_authority = document.get("source_authority")
    if not isinstance(source_authority, Mapping):
        raise PadSurfaceCalibrationError("calibration_source_authority_missing")
    authority_path_value = source_authority.get("path")
    authority_sha = source_authority.get("sha256")
    authority_schema = source_authority.get("schema")
    if not isinstance(authority_path_value, str) or not authority_path_value.strip():
        raise PadSurfaceCalibrationError("calibration_authority_path_invalid")
    authority_path, _, observed_authority_sha = _snapshot(
        Path(authority_path_value), label="calibration_authority"
    )
    if authority_sha != observed_authority_sha:
        raise PadSurfaceCalibrationError("calibration_authority_sha256_mismatch")
    if document.get("source_artifact_sha256") != observed_authority_sha:
        raise PadSurfaceCalibrationError("calibration_source_artifact_sha256_mismatch")
    if authority_schema != method_to_authority_schema[method]:
        raise PadSurfaceCalibrationError("calibration_authority_schema_mismatch")

    calibration_id = document.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise PadSurfaceCalibrationError("calibration_id_invalid")
    if method == "MULTI_POSE_MEASURED_CONTACT":
        raw_thresholds = document.get("fit_thresholds")
        if not isinstance(raw_thresholds, Mapping):
            raise PadSurfaceCalibrationError("calibration_fit_thresholds_missing")
        try:
            thresholds = ContactFitThresholds(**dict(raw_thresholds))
        except (TypeError, ValueError) as error:
            raise PadSurfaceCalibrationError(
                f"calibration_fit_thresholds_invalid:{error}"
            ) from error
        rebuilt = build_from_measured_contacts(
            authority_path=authority_path,
            asset_path=runtime_asset_path,
            calibration_id=calibration_id,
            expected_asset_sha256=expected_asset_sha256,
            thresholds=thresholds,
        )
    else:
        rebuilt = build_from_cad_authority(
            authority_path=authority_path,
            asset_path=runtime_asset_path,
            calibration_id=calibration_id,
            expected_asset_sha256=expected_asset_sha256,
        )

    for name in (
        "inner_link_local_offset_m",
        "outer_link_local_offset_m",
    ):
        observed = _finite_vector(document.get(name), size=3, label=name)
        expected = _finite_vector(rebuilt[name], size=3, label=f"rebuilt_{name}")
        if not bool(np.allclose(observed, expected, rtol=0.0, atol=1.0e-12)):
            raise PadSurfaceCalibrationError(f"calibration_rebuilt_offset_mismatch:{name}")
    if document.get("calibration_rows") != rebuilt["calibration_rows"]:
        raise PadSurfaceCalibrationError("calibration_row_count_mismatch")
    if document.get("fit_validation") != rebuilt["fit_validation"]:
        raise PadSurfaceCalibrationError("calibration_fit_validation_mismatch")
    if document.get("calibration_rows_by_link") != rebuilt.get(
        "calibration_rows_by_link"
    ):
        raise PadSurfaceCalibrationError("calibration_rows_by_link_mismatch")
    for name in ("authority_id", "authority_kind"):
        if document.get(name) != rebuilt.get(name):
            raise PadSurfaceCalibrationError(
                f"calibration_authority_identity_mismatch:{name}"
            )
    if rebuilt.get("authority_kind") == PHYSX_CONTACT_AUTHORITY_KIND:
        if document.get("raw_physx_evidence") != rebuilt.get("raw_physx_evidence"):
            raise PadSurfaceCalibrationError(
                "calibration_raw_physx_evidence_mismatch"
            )
    if method == "OEM_CAD_DIRECT_AUTHORITY":
        for name in ("authority_document",):
            if document.get(name) != rebuilt.get(name):
                raise PadSurfaceCalibrationError(
                    f"calibration_cad_authority_mismatch:{name}"
                )

    if artifact_path.read_bytes() != artifact_bytes:
        raise PadSurfaceCalibrationError("calibration_artifact_changed_during_verify")
    return PadSurfaceCalibration(
        inner_link_local_offset_m=tuple(
            float(value) for value in document["inner_link_local_offset_m"]
        ),
        outer_link_local_offset_m=tuple(
            float(value) for value in document["outer_link_local_offset_m"]
        ),
        calibration_id=calibration_id,
        source_artifact_sha256=observed_authority_sha,
        calibration_rows=int(document["calibration_rows"]),
        inner_link_name=str(document["inner_link_name"]),
        outer_link_name=str(document["outer_link_name"]),
    )


def write_calibration_immutable(output: Path, artifact: Mapping[str, Any]) -> str:
    """Durably create *output* without replacing any existing path."""

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(output) from None
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


__all__ = [
    "CAD_AUTHORITY_SCHEMA",
    "CALIBRATION_SCHEMA",
    "ContactFitThresholds",
    "EXPECTED_G2_ASSET_SHA256",
    "MEASURED_CONTACT_SCHEMA",
    "PHYSX_CONTACT_AUTHORITY_KIND",
    "PRODUCER_SCHEMA",
    "RAW_PHYSX_CONTACT_API",
    "RAW_PHYSX_EVIDENCE_SCHEMA",
    "RAW_PHYSX_VALIDATION_SCHEMA",
    "TARGET_PHYSX_RUNTIME",
    "PadSurfaceCalibrationError",
    "build_from_cad_authority",
    "build_from_measured_contacts",
    "sha256_file",
    "verify_calibration_artifact",
    "write_calibration_immutable",
]
