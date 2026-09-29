# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure helpers for measured G2 distal-pad contact acquisition.

The live capture is intentionally kept in a standalone Isaac process.  This
module contains only the fail-closed document builder and immutable writer so
its authority semantics can be tested without importing Kit or PhysX.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .g2_asset_dependency_manifest import (
    G2AssetDependencyError,
    verify_g2_asset_dependency_manifest,
)
from .pad_surface_calibration import (
    EXPECTED_G2_ASSET_SHA256,
    LINK_NAMES,
    MEASURED_CONTACT_SCHEMA,
    PadSurfaceCalibrationError,
    sha256_bytes,
)
from .pad_source_freeze_manifest import (
    PAD_INSTALLED_RUNTIME_SOURCE_PATHS,
    PAD_LOCAL_SOURCE_RELATIVE_PATHS,
    load_and_validate_pad_source_freeze_manifest,
)


ACQUISITION_SCHEMA = "g2_pad_surface_physx_contact_acquisition_v2"
AUTHORITY_KIND = "SIM601_PRODUCTION_VALIDATED"
CONTACT_POINT_API = (
    "isaaclab.ContactSensor.track_contact_points/"
    "RigidContactView.get_contact_data+get_raw_contact_data+"
    "get_other_actor_paths_from_ids"
)
TARGET_RUNTIME = "IsaacLab_v3.0.0-beta2-patch1+Isaac_Sim_6.0.1"
TARGET_RUNTIME_IDENTITY = {
    # Bind the native interpreter, not the convenience ``bin/python``
    # symlink.  Live capture derives this with ``Path(sys.executable).resolve``;
    # using the symlink here made command-freeze and runtime-identity checks
    # mutually impossible to satisfy in one process.
    "python_executable": "/data/fain-data/test/genie_sim_isaaclab3_sim601/bin/python3.12",
    "isaaclab_source_root": "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1",
    "isaaclab_git_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
    "isaaclab_git_describe": "v3.0.0-beta2.patch1",
    "isaaclab_git_dirty": False,
    "isaaclab_python_version": "6.1.14",
    "isaacsim_distribution_version": "6.0.1.0",
}
QUATERNION_PERSISTENCE_CONTRACT = (
    "xyzw end-to-end; no persisted quaternion component reordering"
)


def _finite_vector(value: Any, *, size: int, label: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise PadSurfaceCalibrationError(f"{label}_not_numeric") from error
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise PadSurfaceCalibrationError(f"{label}_must_be_finite_vector_{size}")
    return result


def normalize_xyzw(value: Sequence[float]) -> list[float]:
    """Validate and normalize one XYZW quaternion without component reordering."""

    quaternion = _finite_vector(value, size=4, label="link_quaternion_world_xyzw")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if abs(norm - 1.0) > 1.0e-3:
        raise PadSurfaceCalibrationError(
            f"link_quaternion_world_xyzw_not_unit:norm={norm:.12g}"
        )
    return [item / norm for item in quaternion]


def build_physx_contact_authority(
    *,
    authority_id: str,
    asset_path: Path,
    asset_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    capture_provenance: Mapping[str, Any],
    probe: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the producer input from actual filtered PhysX contact rows.

    Every pose must contain exactly one row for each named distal link.  A row
    is accepted only when it carries a finite positive pair force, a positive
    raw contact count, the exact filtered partner identity, and an explicit
    attestation that the world point came from the contact-point tensor API.
    Link origins are retained only as measured link poses in the fitting
    equation; they are never substituted for a contact point.
    """

    authority_id = str(authority_id).strip()
    if not authority_id:
        raise PadSurfaceCalibrationError("authority_id_empty")
    if asset_sha256 != EXPECTED_G2_ASSET_SHA256:
        raise PadSurfaceCalibrationError("source_asset_sha256_mismatch")
    if capture_provenance.get("contact_point_api") != CONTACT_POINT_API:
        raise PadSurfaceCalibrationError("contact_point_api_provenance_invalid")
    exact_runtime_contract = {
        "runtime": TARGET_RUNTIME,
        "production_runtime": TARGET_RUNTIME,
        "production_runtime_match": True,
        "physics_backend": "PhysX",
        "quaternion_native": "xyzw",
        "quaternion_persisted": QUATERNION_PERSISTENCE_CONTRACT,
        "pending_validations": [],
    }
    for field, expected in exact_runtime_contract.items():
        if capture_provenance.get(field) != expected:
            raise PadSurfaceCalibrationError(
                f"capture_runtime_provenance_invalid:{field}"
            )
    if capture_provenance.get("runtime_identity") != TARGET_RUNTIME_IDENTITY:
        raise PadSurfaceCalibrationError("capture_runtime_identity_invalid")
    manifest_reference = capture_provenance.get("source_freeze_manifest")
    if not isinstance(manifest_reference, Mapping):
        raise PadSurfaceCalibrationError("capture_source_freeze_manifest_missing")
    manifest_path = manifest_reference.get("path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise PadSurfaceCalibrationError(
            "capture_source_freeze_manifest_path_invalid"
        )
    capture_arguments = capture_provenance.get("capture_arguments")
    if not isinstance(capture_arguments, Mapping):
        raise PadSurfaceCalibrationError("capture_arguments_missing")
    repository_root = Path(__file__).resolve().parents[5]
    _, observed_manifest_reference = load_and_validate_pad_source_freeze_manifest(
        Path(manifest_path),
        expected_runtime_identity=TARGET_RUNTIME_IDENTITY,
        expected_seed=int(capture_provenance.get("seed", -1)),
        expected_capture_command=[
            TARGET_RUNTIME_IDENTITY["python_executable"],
            str(
                repository_root
                / "scripts/diagnostics/capture_g2_pad_surface_contact_authority.py"
            ),
        ],
        expected_capture_arguments=capture_arguments,
        required_local_source_paths=PAD_LOCAL_SOURCE_RELATIVE_PATHS,
        required_installed_source_paths=PAD_INSTALLED_RUNTIME_SOURCE_PATHS,
    )
    for field in ("path", "file_sha256", "document_sha256", "schema"):
        if manifest_reference.get(field) != observed_manifest_reference.get(field):
            raise PadSurfaceCalibrationError(
                f"capture_source_freeze_manifest_mismatch:{field}"
            )
    try:
        verify_g2_asset_dependency_manifest(
            capture_provenance.get("asset_dependency_manifest", {}),
            asset_path=asset_path,
        )
    except G2AssetDependencyError as error:
        raise PadSurfaceCalibrationError(
            f"capture_asset_dependency_manifest_invalid:{error}"
        ) from error
    if capture_provenance.get("link_origin_used_as_contact_point") is not False:
        raise PadSurfaceCalibrationError("link_origin_contact_substitution_not_forbidden")
    if capture_provenance.get("mesh_centroid_used_as_contact_point") is not False:
        raise PadSurfaceCalibrationError("mesh_centroid_contact_substitution_not_forbidden")
    if capture_provenance.get("legacy_grasp_center_used_as_contact_point") is not False:
        raise PadSurfaceCalibrationError("legacy_grasp_center_contact_substitution_not_forbidden")

    normalized_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    pose_links: dict[str, set[str]] = {}
    expected_partner_suffix = str(probe.get("prim_suffix", "")).strip()
    if not expected_partner_suffix.startswith("/"):
        raise PadSurfaceCalibrationError("probe_prim_suffix_invalid")
    for index, raw in enumerate(rows):
        pose_id = str(raw.get("pose_id", "")).strip()
        link_name = str(raw.get("link_name", "")).strip()
        if not pose_id or link_name not in LINK_NAMES:
            raise PadSurfaceCalibrationError(f"capture_row_identity_invalid:{index}")
        key = (pose_id, link_name)
        if key in seen:
            raise PadSurfaceCalibrationError(f"capture_row_duplicate:{pose_id}:{link_name}")
        seen.add(key)
        pose_links.setdefault(pose_id, set()).add(link_name)

        force_n = float(raw.get("filtered_pair_force_n", float("nan")))
        contact_count = int(raw.get("raw_contact_count", 0))
        sensor_path = str(raw.get("sensor_prim_path", ""))
        partner_path = str(raw.get("filter_partner_prim_path", ""))
        source = str(raw.get("contact_point_source", ""))
        if not math.isfinite(force_n) or force_n <= 0.0:
            raise PadSurfaceCalibrationError(f"capture_row_force_invalid:{index}")
        if contact_count <= 0:
            raise PadSurfaceCalibrationError(f"capture_row_contact_count_invalid:{index}")
        expected_sensor_suffix = f"/Robot/{link_name}"
        if not sensor_path.endswith(expected_sensor_suffix):
            raise PadSurfaceCalibrationError(f"capture_row_sensor_invalid:{index}")
        environment_prefix = sensor_path[: -len(expected_sensor_suffix)]
        if partner_path != f"{environment_prefix}{expected_partner_suffix}":
            raise PadSurfaceCalibrationError(f"capture_row_partner_invalid:{index}")
        if source != CONTACT_POINT_API:
            raise PadSurfaceCalibrationError(f"capture_row_point_source_invalid:{index}")
        selected_other_actor_path = str(raw.get("selected_other_actor_path", ""))
        try:
            selected_other_actor_id = int(raw.get("selected_other_actor_id", -1))
            selected_separation_m = float(raw.get("selected_separation_m", float("nan")))
            unexpected_force_residual_n = float(
                raw.get("unexpected_unfiltered_force_residual_n", float("nan"))
            )
        except (TypeError, ValueError) as error:
            raise PadSurfaceCalibrationError(
                f"capture_row_raw_identity_evidence_invalid:{index}"
            ) from error
        if selected_other_actor_path != partner_path or selected_other_actor_id < 0:
            raise PadSurfaceCalibrationError(
                f"capture_row_raw_partner_identity_invalid:{index}"
            )
        if not (
            math.isfinite(selected_separation_m)
            and math.isfinite(unexpected_force_residual_n)
            and unexpected_force_residual_n >= 0.0
        ):
            raise PadSurfaceCalibrationError(
                f"capture_row_raw_contact_metrics_invalid:{index}"
            )

        # The target Lab3/Sim6 APIs expose XYZW and the measured-contact
        # authority also persists XYZW.  Validate/normalize here, but never
        # reorder the components along the raw -> validated -> authority path.
        quaternion_xyzw = normalize_xyzw(raw.get("link_quaternion_world_xyzw"))
        normalized_rows.append(
            {
                "pose_id": pose_id,
                "link_name": link_name,
                "link_position_world_m": _finite_vector(
                    raw.get("link_position_world_m"), size=3, label=f"row_{index}_link_position"
                ),
                "link_quaternion_world_xyzw": quaternion_xyzw,
                "measured_pad_surface_point_world_m": _finite_vector(
                    raw.get("measured_pad_surface_point_world_m"),
                    size=3,
                    label=f"row_{index}_contact_point",
                ),
                "measurement_valid": raw.get("measurement_valid") is True,
                "filtered_pair_force_n": force_n,
                "raw_contact_count": contact_count,
                "sensor_prim_path": sensor_path,
                "filter_partner_prim_path": partner_path,
                "contact_point_source": source,
                "physics_step": int(raw.get("physics_step", -1)),
                "selected_other_actor_id": selected_other_actor_id,
                "selected_other_actor_path": selected_other_actor_path,
                "selected_separation_m": selected_separation_m,
                "unexpected_unfiltered_force_residual_n": unexpected_force_residual_n,
            }
        )
        if normalized_rows[-1]["measurement_valid"] is not True:
            raise PadSurfaceCalibrationError(f"capture_row_not_attested_valid:{index}")

    if len(pose_links) < 3:
        raise PadSurfaceCalibrationError("capture_unique_poses_insufficient")
    expected_links = set(LINK_NAMES)
    incomplete = sorted(pose_id for pose_id, links in pose_links.items() if links != expected_links)
    if incomplete:
        raise PadSurfaceCalibrationError(
            "capture_pose_pairs_incomplete:" + ",".join(incomplete)
        )
    if len(normalized_rows) < 2 * 6:
        raise PadSurfaceCalibrationError("capture_rows_insufficient")

    # A force-qualified pad/object point is not sufficient safety evidence on
    # its own.  The acquisition is promotable only when the existing full-body
    # collision authority was live for every step, including self and ground
    # contacts, and the first post-reset FD sample was included rather than
    # discarded as warm-up.
    required_true = (
        "pass",
        "collision_authority_live",
        "forbidden_collision_measured",
        "self_collision_measured",
        "ground_collision_measured",
        "initial_post_reset_fd_sample_included",
    )
    for field in required_true:
        if safety.get(field) is not True:
            raise PadSurfaceCalibrationError(f"capture_safety_attestation_invalid:{field}")
    if int(safety.get("forbidden_collision_count", -1)) != 0:
        raise PadSurfaceCalibrationError("capture_forbidden_collision_nonzero")
    bounded_values = (
        ("maximum_joint_speed_rad_s", "joint_speed_limit_rad_s"),
        ("maximum_joint_acceleration_rad_s2", "joint_acceleration_limit_rad_s2"),
        ("maximum_filtered_pair_force_n", "filtered_pair_force_limit_n"),
    )
    for measured_name, limit_name in bounded_values:
        try:
            measured = float(safety[measured_name])
            limit = float(safety[limit_name])
        except (KeyError, TypeError, ValueError) as error:
            raise PadSurfaceCalibrationError(
                f"capture_safety_bound_invalid:{measured_name}"
            ) from error
        if not (math.isfinite(measured) and math.isfinite(limit) and 0.0 <= measured <= limit):
            raise PadSurfaceCalibrationError(
                f"capture_safety_bound_exceeded:{measured_name}"
            )

    return {
        "schema": MEASURED_CONTACT_SCHEMA,
        "authority_id": authority_id,
        "authority_kind": AUTHORITY_KIND,
        "coordinate_frame": "world",
        "quaternion_order": "xyzw",
        "units": "m",
        "source_asset_sha256": asset_sha256,
        "source_asset_path": str(Path(asset_path).expanduser().resolve()),
        "acquisition_schema": ACQUISITION_SCHEMA,
        "capture_provenance": dict(capture_provenance),
        "probe": dict(probe),
        "safety": dict(safety),
        "rows": normalized_rows,
    }


def write_json_immutable(path: Path, document: Mapping[str, Any]) -> str:
    """Fsync and hard-link a JSON document without replacing any output."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


__all__ = [
    "ACQUISITION_SCHEMA",
    "AUTHORITY_KIND",
    "CONTACT_POINT_API",
    "QUATERNION_PERSISTENCE_CONTRACT",
    "TARGET_RUNTIME",
    "TARGET_RUNTIME_IDENTITY",
    "build_physx_contact_authority",
    "write_json_immutable",
    "normalize_xyzw",
]
