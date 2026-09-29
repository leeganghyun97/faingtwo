# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable source/runtime identity for G2 pad-contact calibration.

The manifest uses a canonical document hash that excludes only its own
``document_sha256`` field.  The immutable JSON file additionally has an
ordinary file SHA-256.  Keeping these identities separate avoids a circular
self-hash while still allowing every downstream artifact to bind both the
semantic document and the exact bytes read from disk.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .g2_asset_dependency_manifest import (
    build_g2_asset_dependency_manifest,
    verify_g2_asset_dependency_manifest,
)
from .pad_surface_calibration import PadSurfaceCalibrationError


SOURCE_FREEZE_MANIFEST_SCHEMA = "g2_pad_surface_source_freeze_manifest_v4"
PAD_LOCAL_SOURCE_RELATIVE_PATHS = (
    "source/geniesim/rl/isaaclab/g2_collision_authority.py",
    "source/geniesim/rl/isaaclab/g2_gripper_reset_contract.py",
    "source/geniesim/rl/isaaclab/g2_keyboard_pose.py",
    "source/geniesim/rl/isaaclab/g2_lift_methodology.py",
    "source/geniesim/rl/isaaclab/g2_process_lifecycle.py",
    "source/geniesim/rl/isaaclab/g2_quaternion.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/data_contract.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/g2_asset_dependency_manifest.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/isaac_runtime_adapter.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/observations.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/pad_contact_acquisition.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/pad_contact_evidence.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/pad_source_freeze_manifest.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/pad_surface_calibration.py",
    "source/geniesim/rl/isaaclab/g2_rebuild/sensor_packet.py",
    "scripts/build_g2_pad_source_freeze_manifest.py",
    "scripts/diagnostics/capture_g2_pad_surface_contact_authority.py",
    "scripts/diagnostics/g2_pad_keep_seed_anchor_overlay.py",
    "scripts/diagnostics/run_g2_pad_capture_attempt.py",
    "scripts/validate_g2_pad_surface_raw_contact_evidence.py",
    "scripts/produce_g2_pad_surface_calibration.py",
)
PAD_INSTALLED_RUNTIME_SOURCE_PATHS = (
    "/data/fain-data/test/genie_sim_isaaclab3_sim601/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.13+110.1.2.lx64.r.cp312.u7f4/omni/physics/tensors/api.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor_data.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab/isaaclab/scene/interactive_scene.py",
    "/data/fain-data/test/IsaacLab-v3.0.0-beta2-patch1/source/isaaclab/isaaclab/sim/simulation_context.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_payload(document: Mapping[str, Any]) -> bytes:
    payload = copy.deepcopy(dict(document))
    payload.pop("document_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_document_sha256(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(document)).hexdigest()


def _source_rows(paths: Sequence[Path], *, root: Path | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    resolved_root = root.expanduser().resolve() if root is not None else None
    for source in paths:
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_file():
            raise PadSurfaceCalibrationError(
                f"source_freeze_file_missing:{resolved}"
            )
        if resolved_root is not None:
            try:
                identity = resolved.relative_to(resolved_root).as_posix()
            except ValueError as error:
                raise PadSurfaceCalibrationError(
                    f"source_freeze_local_file_outside_repository:{resolved}"
                ) from error
        else:
            identity = str(resolved)
        if identity in seen:
            raise PadSurfaceCalibrationError(
                f"source_freeze_file_duplicate:{identity}"
            )
        seen.add(identity)
        rows.append(
            {
                "path": identity,
                "resolved_path": str(resolved),
                "sha256": sha256_file(resolved),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _input_artifacts(capture_arguments: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Freeze non-source inputs whose bytes select calibration samples."""

    rows: dict[str, dict[str, str]] = {}
    for key in ("pose_source_hdf5", "four_bar_anchor_overlay_manifest"):
        value = capture_arguments.get(key)
        if value is None:
            continue
        resolved = Path(str(value)).expanduser().resolve()
        if not resolved.is_file():
            raise PadSurfaceCalibrationError(
                f"source_freeze_input_artifact_missing:{resolved}"
            )
        row = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
        if key == "four_bar_anchor_overlay_manifest":
            try:
                overlay = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PadSurfaceCalibrationError(
                    "source_freeze_anchor_overlay_json_invalid"
                ) from error
            if not isinstance(overlay, dict):
                raise PadSurfaceCalibrationError(
                    "source_freeze_anchor_overlay_root_invalid"
                )
            semantic = copy.deepcopy(overlay)
            semantic.pop("document_sha256", None)
            document_sha256 = hashlib.sha256(
                json.dumps(
                    semantic,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if (
                overlay.get("schema")
                != "g2_omnipicker_keep_existing_seed_anchor_only_ab_manifest_v1"
                or overlay.get("candidate_id")
                != "KEEP_EXISTING_SEED_EXACT_POSITION_ANCHOR"
                or overlay.get("document_sha256") != document_sha256
                or "FRAME_ALIGNED" in json.dumps(overlay).upper()
            ):
                raise PadSurfaceCalibrationError(
                    "source_freeze_anchor_overlay_contract_invalid"
                )
            row.update(
                {
                    "schema": str(overlay["schema"]),
                    "candidate_id": str(overlay["candidate_id"]),
                    "document_sha256": document_sha256,
                }
            )
        rows[key] = row
    return rows


def build_pad_source_freeze_manifest(
    *,
    repository_root: Path,
    asset_path: Path,
    local_source_paths: Sequence[Path],
    installed_source_paths: Sequence[Path],
    runtime_identity: Mapping[str, Any],
    capture_command: Sequence[str],
    capture_arguments: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    repository_root = repository_root.expanduser().resolve()
    asset_path = asset_path.expanduser().resolve()
    if not repository_root.is_dir() or not asset_path.is_file():
        raise PadSurfaceCalibrationError("source_freeze_root_or_asset_missing")
    if not capture_command or not all(
        isinstance(item, str) and item for item in capture_command
    ):
        raise PadSurfaceCalibrationError("source_freeze_capture_command_invalid")
    if isinstance(seed, bool):
        raise PadSurfaceCalibrationError("source_freeze_seed_invalid")
    manifest: dict[str, Any] = {
        "schema": SOURCE_FREEZE_MANIFEST_SCHEMA,
        "repository_root": str(repository_root),
        "runtime_identity": copy.deepcopy(dict(runtime_identity)),
        "local_sources": _source_rows(local_source_paths, root=repository_root),
        "installed_runtime_sources": _source_rows(
            installed_source_paths, root=None
        ),
        "asset_path": str(asset_path),
        "asset_dependency_manifest": build_g2_asset_dependency_manifest(asset_path),
        "input_artifacts": _input_artifacts(capture_arguments),
        "capture_invocation": {
            "command": list(capture_command),
            "arguments": copy.deepcopy(dict(capture_arguments)),
            "seed": int(seed),
        },
        "hard_limits": {
            "maximum_joint_speed_rad_s": 0.8,
            "maximum_joint_acceleration_rad_s2": 10.0,
            "maximum_filtered_pair_force_n": 500.0,
            "limits_not_relaxed": True,
        },
    }
    manifest["document_sha256"] = canonical_document_sha256(manifest)
    return manifest


def validate_pad_source_freeze_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_runtime_identity: Mapping[str, Any] | None = None,
    expected_seed: int | None = None,
    expected_capture_command: Sequence[str] | None = None,
    expected_capture_arguments: Mapping[str, Any] | None = None,
    required_local_source_paths: Sequence[str] | None = None,
    required_installed_source_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    observed_schema = manifest.get("schema")
    if observed_schema != SOURCE_FREEZE_MANIFEST_SCHEMA:
        if isinstance(observed_schema, str) and observed_schema.startswith(
            "g2_pad_surface_source_freeze_manifest_v"
        ):
            raise PadSurfaceCalibrationError(
                "source_freeze_manifest_schema_stale:"
                f"{observed_schema}:expected:{SOURCE_FREEZE_MANIFEST_SCHEMA}"
            )
        raise PadSurfaceCalibrationError("source_freeze_manifest_schema_invalid")
    observed_document_hash = canonical_document_sha256(manifest)
    if manifest.get("document_sha256") != observed_document_hash:
        raise PadSurfaceCalibrationError("source_freeze_manifest_document_hash_invalid")
    root = Path(str(manifest.get("repository_root", ""))).expanduser().resolve()
    if not root.is_dir():
        raise PadSurfaceCalibrationError("source_freeze_repository_root_invalid")
    for group_name in ("local_sources", "installed_runtime_sources"):
        rows = manifest.get(group_name)
        if not isinstance(rows, list) or not rows:
            raise PadSurfaceCalibrationError(
                f"source_freeze_{group_name}_missing"
            )
        identities: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise PadSurfaceCalibrationError(
                    f"source_freeze_{group_name}_row_invalid"
                )
            identity = str(row.get("path", ""))
            resolved = Path(str(row.get("resolved_path", ""))).expanduser().resolve()
            if not identity or identity in identities:
                raise PadSurfaceCalibrationError(
                    f"source_freeze_{group_name}_identity_invalid"
                )
            identities.add(identity)
            if group_name == "local_sources":
                try:
                    if resolved.relative_to(root).as_posix() != identity:
                        raise PadSurfaceCalibrationError(
                            "source_freeze_local_source_identity_mismatch"
                        )
                except ValueError as error:
                    raise PadSurfaceCalibrationError(
                        "source_freeze_local_source_outside_repository"
                    ) from error
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                raise PadSurfaceCalibrationError(
                    f"source_freeze_source_hash_mismatch:{identity}"
                )
        required = (
            required_local_source_paths
            if group_name == "local_sources"
            else required_installed_source_paths
        )
        if required is not None and identities != set(required):
            raise PadSurfaceCalibrationError(
                f"source_freeze_{group_name}_closure_mismatch"
            )
    dependency_manifest = manifest.get("asset_dependency_manifest")
    if not isinstance(dependency_manifest, Mapping):
        raise PadSurfaceCalibrationError("source_freeze_asset_closure_missing")
    asset_path = Path(str(manifest.get("asset_path", ""))).expanduser().resolve()
    if not asset_path.is_file():
        raise PadSurfaceCalibrationError("source_freeze_asset_path_invalid")
    verify_g2_asset_dependency_manifest(dependency_manifest, asset_path=asset_path)
    if expected_runtime_identity is not None and manifest.get(
        "runtime_identity"
    ) != dict(expected_runtime_identity):
        raise PadSurfaceCalibrationError("source_freeze_runtime_identity_mismatch")
    invocation = manifest.get("capture_invocation")
    if not isinstance(invocation, Mapping):
        raise PadSurfaceCalibrationError("source_freeze_capture_invocation_missing")
    if expected_seed is not None and invocation.get("seed") != int(expected_seed):
        raise PadSurfaceCalibrationError("source_freeze_capture_seed_mismatch")
    if expected_capture_command is not None and invocation.get("command") != list(
        expected_capture_command
    ):
        raise PadSurfaceCalibrationError("source_freeze_capture_command_mismatch")
    if expected_capture_arguments is not None and invocation.get(
        "arguments"
    ) != dict(expected_capture_arguments):
        raise PadSurfaceCalibrationError("source_freeze_capture_arguments_mismatch")
    capture_arguments = invocation.get("arguments")
    if not isinstance(capture_arguments, Mapping):
        raise PadSurfaceCalibrationError("source_freeze_capture_arguments_invalid")
    expected_inputs = _input_artifacts(capture_arguments)
    if manifest.get("input_artifacts") != expected_inputs:
        raise PadSurfaceCalibrationError(
            "source_freeze_input_artifact_hash_mismatch"
        )
    hard_limits = manifest.get("hard_limits")
    if hard_limits != {
        "maximum_joint_speed_rad_s": 0.8,
        "maximum_joint_acceleration_rad_s2": 10.0,
        "maximum_filtered_pair_force_n": 500.0,
        "limits_not_relaxed": True,
    }:
        raise PadSurfaceCalibrationError("source_freeze_hard_limits_invalid")
    return {
        "schema": SOURCE_FREEZE_MANIFEST_SCHEMA,
        "document_sha256": observed_document_hash,
        "runtime_identity": copy.deepcopy(dict(manifest["runtime_identity"])),
        "asset_dependency_manifest_sha256": dependency_manifest[
            "manifest_sha256"
        ],
        "input_artifacts": copy.deepcopy(expected_inputs),
    }


def load_and_validate_pad_source_freeze_manifest(
    path: Path,
    **validation_arguments: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = resolved.read_bytes()
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PadSurfaceCalibrationError(
            "source_freeze_manifest_json_invalid"
        ) from error
    if not isinstance(manifest, dict):
        raise PadSurfaceCalibrationError("source_freeze_manifest_root_invalid")
    validation = validate_pad_source_freeze_manifest(
        manifest, **validation_arguments
    )
    validation.update(
        {
            "path": str(resolved),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return manifest, validation


__all__ = [
    "PAD_INSTALLED_RUNTIME_SOURCE_PATHS",
    "PAD_LOCAL_SOURCE_RELATIVE_PATHS",
    "SOURCE_FREEZE_MANIFEST_SCHEMA",
    "build_pad_source_freeze_manifest",
    "canonical_document_sha256",
    "load_and_validate_pad_source_freeze_manifest",
    "sha256_file",
    "validate_pad_source_freeze_manifest",
]
