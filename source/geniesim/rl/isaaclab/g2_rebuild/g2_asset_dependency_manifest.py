# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical dependency closure for the checked-in G2 runtime USD.

The root layer hash alone is not an asset identity: the composed robot also
depends on binary base/collision, physics, sensor, robot and camera layers.
This module intentionally stays CPU-only and hashes the exact, known layer
closure of ``robot_fix.usda``.  The root ASCII layer is also checked for every
composition arc, so a missing or renamed reference cannot be hidden by a
stale list in an artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ASSET_DEPENDENCY_MANIFEST_SCHEMA = "g2_usd_dependency_closure_manifest_v1"

# This is the layer set returned by UsdUtils.ComputeAllDependencies for the
# checked-in root layer.  ``OmniPBR.mdl`` is an Isaac runtime material module,
# not a repository layer and does not author robot physics/collision state.
EXPECTED_LAYER_ROLES = {
    "robot_fix.usda": "root_override_and_articulation",
    "configuration/Cam_L.usd": "left_head_camera_payload",
    "configuration/Cam_R.usd": "right_wrist_camera_payload",
    "configuration/robot_base.usd": "base_visual_collision_and_joint_layer",
    "configuration/robot_physics.usd": "physics_variant_layer",
    "configuration/robot_robot.usd": "robot_variant_layer",
    "configuration/robot_sensor.usd": "sensor_variant_layer",
}
EXPECTED_RUNTIME_ONLY_ASSETS = ("OmniPBR.mdl",)


class G2AssetDependencyError(RuntimeError):
    """Raised when the composed G2 asset closure differs from authority."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_payload(document: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in document.items() if key != "manifest_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_g2_asset_dependency_manifest(asset_path: Path) -> dict[str, Any]:
    root = Path(asset_path).expanduser().resolve()
    if root.name != "robot_fix.usda" or not root.is_file():
        raise G2AssetDependencyError(f"g2_asset_root_invalid:{root}")
    directory = root.parent
    root_text = root.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []
    for relative_path, role in EXPECTED_LAYER_ROLES.items():
        path = (directory / relative_path).resolve()
        if path.parent != directory and directory not in path.parents:
            raise G2AssetDependencyError(
                f"g2_asset_dependency_escaped_root:{relative_path}"
            )
        if not path.is_file():
            raise G2AssetDependencyError(
                f"g2_asset_dependency_missing:{relative_path}"
            )
        if relative_path != "robot_fix.usda":
            composition_token = f"@./{relative_path}@"
            if composition_token not in root_text:
                raise G2AssetDependencyError(
                    f"g2_asset_composition_arc_missing:{relative_path}"
                )
        entries.append(
            {
                "relative_path": relative_path,
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest: dict[str, Any] = {
        "schema": ASSET_DEPENDENCY_MANIFEST_SCHEMA,
        "root_file": "robot_fix.usda",
        "layer_count": len(entries),
        "layers": entries,
        "runtime_only_nonlayer_assets": list(EXPECTED_RUNTIME_ONLY_ASSETS),
        "pending_dependencies": [],
    }
    manifest["manifest_sha256"] = hashlib.sha256(_semantic_payload(manifest)).hexdigest()
    return manifest


def verify_g2_asset_dependency_manifest(
    manifest: Mapping[str, Any], *, asset_path: Path
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise G2AssetDependencyError("g2_asset_dependency_manifest_missing")
    if manifest.get("schema") != ASSET_DEPENDENCY_MANIFEST_SCHEMA:
        raise G2AssetDependencyError("g2_asset_dependency_manifest_schema_invalid")
    declared_hash = manifest.get("manifest_sha256")
    if not isinstance(declared_hash, str) or hashlib.sha256(
        _semantic_payload(manifest)
    ).hexdigest() != declared_hash:
        raise G2AssetDependencyError("g2_asset_dependency_manifest_hash_invalid")
    if manifest.get("pending_dependencies") != []:
        raise G2AssetDependencyError("g2_asset_dependency_manifest_pending")
    rebuilt = build_g2_asset_dependency_manifest(asset_path)
    if dict(manifest) != rebuilt:
        raise G2AssetDependencyError("g2_asset_dependency_manifest_rehash_mismatch")
    return rebuilt


__all__ = [
    "ASSET_DEPENDENCY_MANIFEST_SCHEMA",
    "EXPECTED_LAYER_ROLES",
    "EXPECTED_RUNTIME_ONLY_ASSETS",
    "G2AssetDependencyError",
    "build_g2_asset_dependency_manifest",
    "verify_g2_asset_dependency_manifest",
]
