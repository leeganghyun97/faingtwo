# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed asset binding for the qualified G2 policy simulation candidate.

This module is deliberately CPU-only.  It does not import Isaac Lab and it
does not create an environment.  A future BC/SAC runtime may call
``bind_m2_qualified_candidate`` after constructing an existing authoritative
G2 environment config and before constructing ``ManagerBasedRLEnv``.

The selected asset remains a separately versioned simulation correction
candidate.  It is neither the Production asset nor an OEM mechanical
authority.  Missing evidence, a changed byte, or a config that is not still
pointing at the frozen Production asset is an error; there is no fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ..g2_rebuild.g2_asset_dependency_manifest import (
    build_g2_asset_dependency_manifest,
)


RUNTIME_ASSET_BINDING_SCHEMA = "g2_policy_runtime_asset_binding_v1"
M2_QUALIFIED_CANDIDATE_BINDING_ID = (
    "M2_QUALIFIED_E1_WIDE_JOINT3_JOINT4_DIAGNOSTIC_CANDIDATE_V1"
)
M2_QUALIFIED_CANDIDATE_CLASSIFICATION = (
    "SEPARATE_VERSIONED_SIMULATION_MODEL_CORRECTION_CANDIDATE_NOT_OEM_AUTHORITY"
)

PRODUCTION_ASSET_RELATIVE_PATH = Path(
    "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
)
PRODUCTION_ASSET_SHA256 = (
    "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
)

# Use the immutable, source-backed simulation correction candidate generated
# for the current repeated-hard-stop requalification.  This is deliberately
# not the Production asset and is not an OEM range claim.
M2_QUALIFIED_CANDIDATE_RELATIVE_PATH = Path(
    "artifacts/g2_m2_combined_hardstop_requalification_20260920/"
    "model_correction_candidates/E1_wide_joint4_joint3/robot_fix.usda"
)
M2_QUALIFIED_CANDIDATE_SHA256 = (
    "d4505622f039da261df992ac7d8d1e1e24eae066058c9b52c29fc418f851a468"
)
M2_QUALIFIED_DEPENDENCY_MANIFEST_SHA256 = (
    "63c17c49bee87c65afc08385e59cb996006b9353d6afa1e137d5fef8718421c8"
)
M2_QUALIFICATION_REPORT_RELATIVE_PATH = Path(
    "artifacts/g2_m2_combined_hardstop_requalification_20260920/"
    "live/E1_m2_fresh_seed42/"
    "hierarchy-report.json"
)
M2_QUALIFICATION_REPORT_SHA256 = (
    "0d87e3258d55d972e106b32117e3ecfab2675b82471a362ac76ecc5cc947c8f3"
)
M2_REQUIRED_PHYSICS_VERDICT = "PASS_WITHIN_EXISTING_HARD_GATES"


class RuntimeAssetBindingError(RuntimeError):
    """Raised instead of silently selecting another robot asset."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_root() -> Path:
    """Return the repository root from this module's installed source path."""

    root = Path(__file__).resolve().parents[5]
    sentinel = root / PRODUCTION_ASSET_RELATIVE_PATH
    if not sentinel.is_file():
        raise RuntimeAssetBindingError(
            f"G2_POLICY_REPOSITORY_ROOT_UNRESOLVED:{root}"
        )
    return root


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_QUALIFICATION_REPORT_INVALID:{path}:{type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_QUALIFICATION_REPORT_NOT_OBJECT:{path}"
        )
    return value


@dataclass(frozen=True)
class M2QualifiedAssetBindingReceipt:
    schema: str
    binding_id: str
    classification: str
    repository_root: str
    production_asset_path: str
    production_asset_sha256: str
    candidate_asset_path: str
    candidate_asset_sha256: str
    dependency_manifest_sha256: str
    qualification_report_path: str
    qualification_report_sha256: str
    qualification_report_asset_path: str
    physics_verdict: str
    artifact_verdict: str
    alternate_asset_fallback: str
    production_asset_modified: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_m2_qualified_candidate(
    *, repo_root: Path | None = None
) -> M2QualifiedAssetBindingReceipt:
    """Verify and return the exact qualified candidate, or fail closed."""

    root = repository_root() if repo_root is None else Path(repo_root).resolve()
    production = (root / PRODUCTION_ASSET_RELATIVE_PATH).resolve()
    candidate = (root / M2_QUALIFIED_CANDIDATE_RELATIVE_PATH).resolve()
    qualification_report = (root / M2_QUALIFICATION_REPORT_RELATIVE_PATH).resolve()

    for label, path in (
        ("PRODUCTION_ASSET", production),
        ("M2_CANDIDATE_ASSET", candidate),
        ("M2_QUALIFICATION_REPORT", qualification_report),
    ):
        if not path.is_file():
            raise RuntimeAssetBindingError(f"G2_POLICY_{label}_MISSING:{path}")
    if production == candidate:
        raise RuntimeAssetBindingError("G2_POLICY_CANDIDATE_IS_PRODUCTION_ASSET")

    production_hash = _sha256(production)
    candidate_hash = _sha256(candidate)
    report_hash = _sha256(qualification_report)
    if production_hash != PRODUCTION_ASSET_SHA256:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_PRODUCTION_ASSET_HASH_MISMATCH:{production_hash}"
        )
    if candidate_hash != M2_QUALIFIED_CANDIDATE_SHA256:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_CANDIDATE_HASH_MISMATCH:{candidate_hash}"
        )
    if report_hash != M2_QUALIFICATION_REPORT_SHA256:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_QUALIFICATION_REPORT_HASH_MISMATCH:{report_hash}"
        )

    dependency_manifest = build_g2_asset_dependency_manifest(candidate)
    dependency_hash = str(dependency_manifest.get("manifest_sha256", ""))
    if dependency_hash != M2_QUALIFIED_DEPENDENCY_MANIFEST_SHA256:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_CANDIDATE_DEPENDENCY_MISMATCH:{dependency_hash}"
        )

    report = _strict_json(qualification_report)
    report_candidate = report.get("candidate")
    if not isinstance(report_candidate, dict):
        raise RuntimeAssetBindingError("G2_POLICY_M2_REPORT_CANDIDATE_MISSING")
    if report_candidate.get("asset_sha256") != candidate_hash:
        raise RuntimeAssetBindingError("G2_POLICY_M2_REPORT_CANDIDATE_HASH_MISMATCH")
    physics_verdict = str(report.get("physics_verdict", ""))
    artifact_verdict = str(report.get("artifact_verdict", ""))
    if physics_verdict != M2_REQUIRED_PHYSICS_VERDICT:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_PHYSICS_NOT_QUALIFIED:{physics_verdict}"
        )
    if artifact_verdict != "PASS":
        raise RuntimeAssetBindingError(
            f"G2_POLICY_M2_ARTIFACT_NOT_QUALIFIED:{artifact_verdict}"
        )

    return M2QualifiedAssetBindingReceipt(
        schema=RUNTIME_ASSET_BINDING_SCHEMA,
        binding_id=M2_QUALIFIED_CANDIDATE_BINDING_ID,
        classification=M2_QUALIFIED_CANDIDATE_CLASSIFICATION,
        repository_root=str(root),
        production_asset_path=str(production),
        production_asset_sha256=production_hash,
        candidate_asset_path=str(candidate),
        candidate_asset_sha256=candidate_hash,
        dependency_manifest_sha256=dependency_hash,
        qualification_report_path=str(qualification_report),
        qualification_report_sha256=report_hash,
        qualification_report_asset_path=str(report_candidate.get("asset_path", "")),
        physics_verdict=physics_verdict,
        artifact_verdict=artifact_verdict,
        alternate_asset_fallback="NONE",
        production_asset_modified=False,
    )


def bind_m2_qualified_candidate(
    cfg: Any, *, repo_root: Path | None = None
) -> M2QualifiedAssetBindingReceipt:
    """Bind the exact candidate to one existing G2 config instance.

    Only ``cfg.scene.robot.spawn.usd_path`` is changed.  Requiring the input
    config to still point at the frozen Production asset catches stale or
    chained overrides and prevents an implicit fallback path.
    """

    receipt = resolve_m2_qualified_candidate(repo_root=repo_root)
    try:
        spawn = cfg.scene.robot.spawn
        current = Path(spawn.usd_path).expanduser().resolve()
    except (AttributeError, TypeError) as error:
        raise RuntimeAssetBindingError(
            "G2_POLICY_CONFIG_ASSET_BINDING_SURFACE_MISSING"
        ) from error
    if str(current) != receipt.production_asset_path:
        raise RuntimeAssetBindingError(
            f"G2_POLICY_CONFIG_NOT_ON_FROZEN_PRODUCTION_ASSET:{current}"
        )
    spawn.usd_path = receipt.candidate_asset_path
    if Path(spawn.usd_path).resolve() != Path(receipt.candidate_asset_path):
        raise RuntimeAssetBindingError("G2_POLICY_CANDIDATE_BINDING_DID_NOT_STICK")
    return receipt


__all__ = [
    "M2_QUALIFIED_CANDIDATE_BINDING_ID",
    "M2_QUALIFIED_CANDIDATE_CLASSIFICATION",
    "M2_QUALIFIED_CANDIDATE_RELATIVE_PATH",
    "M2_QUALIFIED_CANDIDATE_SHA256",
    "M2_QUALIFIED_DEPENDENCY_MANIFEST_SHA256",
    "M2QualifiedAssetBindingReceipt",
    "PRODUCTION_ASSET_RELATIVE_PATH",
    "PRODUCTION_ASSET_SHA256",
    "RUNTIME_ASSET_BINDING_SCHEMA",
    "RuntimeAssetBindingError",
    "bind_m2_qualified_candidate",
    "repository_root",
    "resolve_m2_qualified_candidate",
]
