# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed binding for the Legacy Candidate A contact-free branch.

This authority is intentionally narrower than :mod:`runtime_asset_binding`.
Candidate A is the frozen OPEN/pre-contact simulation baseline; it is not an
OEM mechanical authority and it is not qualified for CLOSE, contact, or M2.
The resolver validates the complete retained provenance chain before changing
one environment-config USD path.  It never edits an asset and has no fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..g2_rebuild.g2_asset_dependency_manifest import (
    build_g2_asset_dependency_manifest,
)


CONTACT_FREE_CANDIDATE_A_BINDING_SCHEMA = (
    "g2_contact_free_candidate_a_runtime_binding_v1"
)
CONTACT_FREE_CANDIDATE_A_BINDING_ID = (
    "LEGACY_CANDIDATE_A_CONTACT_FREE_PRECONTACT_OPEN_ONLY_V1"
)
CONTACT_FREE_CANDIDATE_A_CLASSIFICATION = (
    "CONTACT_FREE_PRECONTACT_ONLY_NOT_OEM_NOT_M2_CONTACT_QUALIFIED"
)
CONTACT_FREE_CANDIDATE_A_SCOPE = "REACH_PREGRASP_CONTACT_FREE_MICRO_APPROACH"

PRODUCTION_ASSET_RELATIVE_PATH = Path(
    "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
)
PRODUCTION_ASSET_SHA256 = (
    "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
)
CANDIDATE_A_RELATIVE_PATH = Path(
    "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/"
    "A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
)
CANDIDATE_A_SHA256 = (
    "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
)
CANDIDATE_A_DEPENDENCY_MANIFEST_SHA256 = (
    "3f2120408d91778e47a22d88090db5f66970fad559d211eb08b88ff952e24bad"
)
CANDIDATE_A_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g2_bounded_passive_range_qualification_20260921/"
    "CANDIDATE_A_MANIFEST.json"
)
CANDIDATE_A_MANIFEST_SHA256 = (
    "ed4b238857384959059342a1e49abdeee731936da88efd216391cef6ec81a9c4"
)
CANDIDATE_A_CONTRACT_RELATIVE_PATH = Path(
    "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/"
    "A_source_min_q3_10deg_q4_11p25deg/bounded-range-contract.json"
)
CANDIDATE_A_CONTRACT_SHA256 = (
    "5e9a86d79bd4a4067ad25a77741f44bde28660e3e5b79d90fe95e3a6d76a6e16"
)
LEGACY_MECHANICS_AUTHORITY_RELATIVE_PATH = Path(
    "artifacts/g2_legacy_candidate_a_mechanics_authority_complete_static_rerun_20260921/"
    "LEGACY_CANDIDATE_A_MECHANICS_AUTHORITY.json"
)
LEGACY_MECHANICS_AUTHORITY_SHA256 = (
    "839f9e3a39e61cebacc394d306bb0178b0c57405573eeebd615d1323ac7930e6"
)
PRECONTACT_CONTRACT_RELATIVE_PATH = Path(
    "artifacts/g2_legacy_candidate_a_precontact_contract_20260921/"
    "PRECONTACT_PIPELINE_CONTRACT.json"
)
PRECONTACT_CONTRACT_SHA256 = (
    "dfca2c54cfae4db1385df91aa2f2a53cff2df2bc0fffda1393cb7735f40469c8"
)
LIMIT_AUTHORITY_RELATIVE_PATH = Path(
    "artifacts/g2_controlled_rebuild/20260918_omnipicker_limit_authority/"
    "authority-audit.json"
)
LIMIT_AUTHORITY_SHA256 = (
    "3ac17ce3b3d388c17cabe548eead9fac9264ba8ab644df981f89a15d753d85fa"
)
FROZEN_TRAJECTORY_RELATIVE_PATH = Path(
    "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
)
FROZEN_TRAJECTORY_SHA256 = (
    "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"
)


class ContactFreeCandidateABindingError(RuntimeError):
    """Raised before an unproven or differently-scoped asset can be bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / PRODUCTION_ASSET_RELATIVE_PATH).is_file():
        raise ContactFreeCandidateABindingError(
            f"G2_CONTACT_FREE_REPOSITORY_ROOT_UNRESOLVED:{root}"
        )
    return root


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ContactFreeCandidateABindingError(
            f"G2_CONTACT_FREE_{label}_INVALID:{path}:{type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        raise ContactFreeCandidateABindingError(
            f"G2_CONTACT_FREE_{label}_NOT_OBJECT:{path}"
        )
    return value


def _same_path(value: Any, expected: Path) -> bool:
    try:
        return Path(value).expanduser().resolve() == expected.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContactFreeCandidateABindingError(f"G2_CONTACT_FREE_{reason}")


@dataclass(frozen=True)
class ContactFreeCandidateABindingReceipt:
    schema: str
    binding_id: str
    classification: str
    scope: str
    repository_root: str
    production_asset_path: str
    production_asset_sha256: str
    candidate_asset_path: str
    candidate_asset_sha256: str
    dependency_manifest_sha256: str
    candidate_manifest_path: str
    candidate_manifest_sha256: str
    candidate_contract_path: str
    candidate_contract_sha256: str
    mechanics_authority_path: str
    mechanics_authority_sha256: str
    precontact_contract_path: str
    precontact_contract_sha256: str
    limit_authority_path: str
    limit_authority_sha256: str
    frozen_trajectory_path: str
    frozen_trajectory_sha256: str
    m2_contact_verdict: str
    close_authorized: bool
    contact_authorized: bool
    contact_training_authorized: bool
    hardware_authority_claim: bool
    production_promotion_claim: bool
    alternate_asset_fallback: str
    production_asset_modified: bool
    candidate_asset_modified: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_candidate_manifest(
    value: Mapping[str, Any], *, candidate: Path, contract: Path
) -> None:
    _require(
        value.get("schema") == "g2_bounded_passive_range_candidate_manifest_v1",
        "CANDIDATE_MANIFEST_SCHEMA_MISMATCH",
    )
    _require(value.get("candidate_count") == 1, "CANDIDATE_MANIFEST_COUNT_MISMATCH")
    _require(value.get("production_asset_immutable") is True, "PRODUCTION_NOT_IMMUTABLE")
    _require(value.get("existing_e1_asset_immutable") is True, "E1_NOT_IMMUTABLE")
    candidates = value.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == 1, "CANDIDATE_MANIFEST_ROWS_INVALID")
    row = candidates[0]
    _require(isinstance(row, Mapping), "CANDIDATE_MANIFEST_ROW_INVALID")
    _require(row.get("candidate_id") == "A_source_min_q3_10deg_q4_11p25deg", "CANDIDATE_ID_MISMATCH")
    _require(_same_path(row.get("asset_path"), candidate), "CANDIDATE_MANIFEST_ASSET_PATH_MISMATCH")
    _require(row.get("asset_sha256") == CANDIDATE_A_SHA256, "CANDIDATE_MANIFEST_ASSET_HASH_MISMATCH")
    _require(_same_path(row.get("contract_path"), contract), "CANDIDATE_MANIFEST_CONTRACT_PATH_MISMATCH")
    _require(row.get("contract_sha256") == CANDIDATE_A_CONTRACT_SHA256, "CANDIDATE_MANIFEST_CONTRACT_HASH_MISMATCH")
    _require(row.get("mechanical_diff_count") == 4, "CANDIDATE_MECHANICAL_DIFF_COUNT_MISMATCH")


def _validate_candidate_contract(
    value: Mapping[str, Any],
    *,
    candidate: Path,
    production: Path,
    trajectory: Path,
    limit_authority: Path,
) -> None:
    _require(value.get("schema") == "g2_bounded_passive_range_candidate_v1", "CANDIDATE_CONTRACT_SCHEMA_MISMATCH")
    _require(value.get("candidate_id") == "A_source_min_q3_10deg_q4_11p25deg", "CANDIDATE_CONTRACT_ID_MISMATCH")
    _require(_same_path(value.get("asset_path"), candidate), "CANDIDATE_CONTRACT_ASSET_PATH_MISMATCH")
    _require(value.get("asset_sha256") == CANDIDATE_A_SHA256, "CANDIDATE_CONTRACT_ASSET_HASH_MISMATCH")
    _require(_same_path(value.get("production_asset_path"), production), "CANDIDATE_CONTRACT_PRODUCTION_PATH_MISMATCH")
    _require(value.get("production_asset_sha256") == PRODUCTION_ASSET_SHA256, "CANDIDATE_CONTRACT_PRODUCTION_HASH_MISMATCH")
    _require(_same_path(value.get("frozen_trajectory_path"), trajectory), "CANDIDATE_CONTRACT_TRAJECTORY_PATH_MISMATCH")
    _require(value.get("frozen_trajectory_sha256") == FROZEN_TRAJECTORY_SHA256, "CANDIDATE_CONTRACT_TRAJECTORY_HASH_MISMATCH")
    _require(_same_path(value.get("authority_audit_path"), limit_authority), "CANDIDATE_CONTRACT_AUTHORITY_PATH_MISMATCH")
    _require(value.get("authority_audit_sha256") == LIMIT_AUTHORITY_SHA256, "CANDIDATE_CONTRACT_AUTHORITY_HASH_MISMATCH")
    _require(value.get("diagnostic_only") is True, "CANDIDATE_CONTRACT_NOT_DIAGNOSTIC_ONLY")
    _require(value.get("hardware_authority_claim") is False, "CANDIDATE_CONTRACT_HARDWARE_AUTHORITY_CLAIMED")
    _require(value.get("production_promotion_claim") is False, "CANDIDATE_CONTRACT_PRODUCTION_PROMOTION_CLAIMED")
    unchanged = value.get("unchanged_contract")
    _require(
        isinstance(unchanged, Mapping)
        and unchanged
        and all(item is True for item in unchanged.values()),
        "CANDIDATE_CONTRACT_UNCHANGED_SURFACE_MISMATCH",
    )
    properties = value.get("changed_properties")
    expected = {
        "/genie/joints/idx73_gripper_r_inner_joint4.physics:lowerLimit",
        "/genie/joints/idx73_gripper_r_inner_joint4.physics:upperLimit",
        "/genie/joints/idx83_gripper_r_outer_joint4.physics:lowerLimit",
        "/genie/joints/idx83_gripper_r_outer_joint4.physics:upperLimit",
    }
    _require(
        isinstance(properties, list)
        and {row.get("property") for row in properties if isinstance(row, Mapping)} == expected
        and len(properties) == len(expected),
        "CANDIDATE_CONTRACT_CHANGED_PROPERTIES_MISMATCH",
    )


def _validate_mechanics_authority(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == "g2_legacy_candidate_a_mechanics_authority_v1", "MECHANICS_AUTHORITY_SCHEMA_MISMATCH")
    _require(value.get("asset_mutated") is False, "MECHANICS_AUTHORITY_ASSET_MUTATED")
    _require(value.get("physics_executed") is False, "MECHANICS_AUTHORITY_NOT_STATIC")
    final = value.get("final")
    _require(isinstance(final, Mapping), "MECHANICS_AUTHORITY_FINAL_MISSING")
    _require(final.get("PRIMARY_BASELINE") == "LEGACY_CANDIDATE_A", "MECHANICS_AUTHORITY_BASELINE_MISMATCH")
    _require(final.get("M2") == "FAIL_CONTACT_MECHANICS", "MECHANICS_AUTHORITY_M2_MUST_REMAIN_FAILED")
    _require(final.get("MECHANICS_CHANGE_AUTHORIZED") == "NO", "MECHANICS_CHANGE_UNEXPECTEDLY_AUTHORIZED")
    _require(final.get("TRAINING_AUTHORIZED") == "NO", "MECHANICS_AUTHORITY_TRAINING_CHANGED")


def _validate_precontact_contract(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == "g2_legacy_candidate_a_precontact_readiness_v1", "PRECONTACT_CONTRACT_SCHEMA_MISMATCH")
    _require(value.get("source_freeze") == "PASS", "PRECONTACT_SOURCE_FREEZE_FAILED")
    _require(value.get("primary_baseline") == "LEGACY_CANDIDATE_A", "PRECONTACT_BASELINE_MISMATCH")
    _require(value.get("asset_mutated") is False, "PRECONTACT_ASSET_MUTATED")
    _require(value.get("physics_executed") is False, "PRECONTACT_CONTRACT_NOT_STATIC")
    qualification = value.get("qualification")
    readiness = value.get("readiness")
    _require(isinstance(qualification, Mapping), "PRECONTACT_QUALIFICATION_MISSING")
    _require(isinstance(readiness, Mapping), "PRECONTACT_READINESS_MISSING")
    _require(qualification.get("M2_CONTACT") == "FAIL_CONTACT_MECHANICS", "PRECONTACT_M2_MUST_REMAIN_FAILED")
    _require(readiness.get("BC_PRECONTACT_READY") == "YES_STATIC_OPEN_ONLY", "PRECONTACT_OPEN_ONLY_AUTHORITY_MISSING")
    _require(readiness.get("BC_CONTACT_READY") == "NO", "PRECONTACT_BC_CONTACT_UNEXPECTEDLY_AUTHORIZED")
    _require(readiness.get("CLOSE_LIVE_AUTHORIZED") == "NO", "PRECONTACT_CLOSE_UNEXPECTEDLY_AUTHORIZED")
    _require(readiness.get("CONTACT_LIVE_AUTHORIZED") == "NO", "PRECONTACT_CONTACT_UNEXPECTEDLY_AUTHORIZED")
    _require(readiness.get("CONTACT_TRAINING_AUTHORIZED") == "NO", "PRECONTACT_TRAINING_UNEXPECTEDLY_AUTHORIZED")


def resolve_contact_free_candidate_a(
    *, repo_root: Path | None = None
) -> ContactFreeCandidateABindingReceipt:
    """Resolve the exact OPEN-only Candidate A authority chain."""

    root = repository_root() if repo_root is None else Path(repo_root).resolve()
    paths = {
        "PRODUCTION_ASSET": root / PRODUCTION_ASSET_RELATIVE_PATH,
        "CANDIDATE_A_ASSET": root / CANDIDATE_A_RELATIVE_PATH,
        "CANDIDATE_A_MANIFEST": root / CANDIDATE_A_MANIFEST_RELATIVE_PATH,
        "CANDIDATE_A_CONTRACT": root / CANDIDATE_A_CONTRACT_RELATIVE_PATH,
        "MECHANICS_AUTHORITY": root / LEGACY_MECHANICS_AUTHORITY_RELATIVE_PATH,
        "PRECONTACT_CONTRACT": root / PRECONTACT_CONTRACT_RELATIVE_PATH,
        "LIMIT_AUTHORITY": root / LIMIT_AUTHORITY_RELATIVE_PATH,
        "FROZEN_TRAJECTORY": root / FROZEN_TRAJECTORY_RELATIVE_PATH,
    }
    paths = {name: path.resolve() for name, path in paths.items()}
    for label, path in paths.items():
        if not path.is_file():
            raise ContactFreeCandidateABindingError(
                f"G2_CONTACT_FREE_{label}_MISSING:{path}"
            )
    expected_hashes = {
        "PRODUCTION_ASSET": PRODUCTION_ASSET_SHA256,
        "CANDIDATE_A_ASSET": CANDIDATE_A_SHA256,
        "CANDIDATE_A_MANIFEST": CANDIDATE_A_MANIFEST_SHA256,
        "CANDIDATE_A_CONTRACT": CANDIDATE_A_CONTRACT_SHA256,
        "MECHANICS_AUTHORITY": LEGACY_MECHANICS_AUTHORITY_SHA256,
        "PRECONTACT_CONTRACT": PRECONTACT_CONTRACT_SHA256,
        "LIMIT_AUTHORITY": LIMIT_AUTHORITY_SHA256,
        "FROZEN_TRAJECTORY": FROZEN_TRAJECTORY_SHA256,
    }
    for label, expected in expected_hashes.items():
        observed = _sha256(paths[label])
        if observed != expected:
            raise ContactFreeCandidateABindingError(
                f"G2_CONTACT_FREE_{label}_HASH_MISMATCH:{observed}"
            )
    _require(paths["PRODUCTION_ASSET"] != paths["CANDIDATE_A_ASSET"], "CANDIDATE_IS_PRODUCTION_ASSET")
    dependency = build_g2_asset_dependency_manifest(paths["CANDIDATE_A_ASSET"])
    _require(
        dependency.get("manifest_sha256") == CANDIDATE_A_DEPENDENCY_MANIFEST_SHA256,
        "CANDIDATE_A_DEPENDENCY_MISMATCH",
    )
    _require(not dependency.get("pending_dependencies"), "CANDIDATE_A_DEPENDENCY_UNRESOLVED")

    manifest = _strict_json(paths["CANDIDATE_A_MANIFEST"], label="CANDIDATE_MANIFEST")
    contract = _strict_json(paths["CANDIDATE_A_CONTRACT"], label="CANDIDATE_CONTRACT")
    mechanics = _strict_json(paths["MECHANICS_AUTHORITY"], label="MECHANICS_AUTHORITY")
    precontact = _strict_json(paths["PRECONTACT_CONTRACT"], label="PRECONTACT_CONTRACT")
    _strict_json(paths["LIMIT_AUTHORITY"], label="LIMIT_AUTHORITY")
    _strict_json(paths["FROZEN_TRAJECTORY"], label="FROZEN_TRAJECTORY")
    _validate_candidate_manifest(
        manifest,
        candidate=paths["CANDIDATE_A_ASSET"],
        contract=paths["CANDIDATE_A_CONTRACT"],
    )
    _validate_candidate_contract(
        contract,
        candidate=paths["CANDIDATE_A_ASSET"],
        production=paths["PRODUCTION_ASSET"],
        trajectory=paths["FROZEN_TRAJECTORY"],
        limit_authority=paths["LIMIT_AUTHORITY"],
    )
    _validate_mechanics_authority(mechanics)
    _validate_precontact_contract(precontact)
    return ContactFreeCandidateABindingReceipt(
        schema=CONTACT_FREE_CANDIDATE_A_BINDING_SCHEMA,
        binding_id=CONTACT_FREE_CANDIDATE_A_BINDING_ID,
        classification=CONTACT_FREE_CANDIDATE_A_CLASSIFICATION,
        scope=CONTACT_FREE_CANDIDATE_A_SCOPE,
        repository_root=str(root),
        production_asset_path=str(paths["PRODUCTION_ASSET"]),
        production_asset_sha256=PRODUCTION_ASSET_SHA256,
        candidate_asset_path=str(paths["CANDIDATE_A_ASSET"]),
        candidate_asset_sha256=CANDIDATE_A_SHA256,
        dependency_manifest_sha256=CANDIDATE_A_DEPENDENCY_MANIFEST_SHA256,
        candidate_manifest_path=str(paths["CANDIDATE_A_MANIFEST"]),
        candidate_manifest_sha256=CANDIDATE_A_MANIFEST_SHA256,
        candidate_contract_path=str(paths["CANDIDATE_A_CONTRACT"]),
        candidate_contract_sha256=CANDIDATE_A_CONTRACT_SHA256,
        mechanics_authority_path=str(paths["MECHANICS_AUTHORITY"]),
        mechanics_authority_sha256=LEGACY_MECHANICS_AUTHORITY_SHA256,
        precontact_contract_path=str(paths["PRECONTACT_CONTRACT"]),
        precontact_contract_sha256=PRECONTACT_CONTRACT_SHA256,
        limit_authority_path=str(paths["LIMIT_AUTHORITY"]),
        limit_authority_sha256=LIMIT_AUTHORITY_SHA256,
        frozen_trajectory_path=str(paths["FROZEN_TRAJECTORY"]),
        frozen_trajectory_sha256=FROZEN_TRAJECTORY_SHA256,
        m2_contact_verdict="FAIL_CONTACT_MECHANICS",
        close_authorized=False,
        contact_authorized=False,
        contact_training_authorized=False,
        hardware_authority_claim=False,
        production_promotion_claim=False,
        alternate_asset_fallback="NONE",
        production_asset_modified=False,
        candidate_asset_modified=False,
    )


def bind_contact_free_candidate_a(
    cfg: Any, *, repo_root: Path | None = None
) -> ContactFreeCandidateABindingReceipt:
    """Bind Candidate A to a Production-based config before env creation."""

    receipt = resolve_contact_free_candidate_a(repo_root=repo_root)
    try:
        spawn = cfg.scene.robot.spawn
        current = Path(spawn.usd_path).expanduser().resolve()
    except (AttributeError, TypeError) as error:
        raise ContactFreeCandidateABindingError(
            "G2_CONTACT_FREE_CONFIG_ASSET_BINDING_SURFACE_MISSING"
        ) from error
    if str(current) != receipt.production_asset_path:
        raise ContactFreeCandidateABindingError(
            f"G2_CONTACT_FREE_CONFIG_NOT_ON_FROZEN_PRODUCTION_ASSET:{current}"
        )
    spawn.usd_path = receipt.candidate_asset_path
    if Path(spawn.usd_path).resolve() != Path(receipt.candidate_asset_path):
        raise ContactFreeCandidateABindingError(
            "G2_CONTACT_FREE_CANDIDATE_A_BINDING_DID_NOT_STICK"
        )
    return receipt


__all__ = [
    "CANDIDATE_A_CONTRACT_SHA256",
    "CANDIDATE_A_DEPENDENCY_MANIFEST_SHA256",
    "CANDIDATE_A_MANIFEST_SHA256",
    "CANDIDATE_A_RELATIVE_PATH",
    "CANDIDATE_A_SHA256",
    "CONTACT_FREE_CANDIDATE_A_BINDING_ID",
    "CONTACT_FREE_CANDIDATE_A_BINDING_SCHEMA",
    "CONTACT_FREE_CANDIDATE_A_CLASSIFICATION",
    "CONTACT_FREE_CANDIDATE_A_SCOPE",
    "ContactFreeCandidateABindingError",
    "ContactFreeCandidateABindingReceipt",
    "FROZEN_TRAJECTORY_SHA256",
    "LEGACY_MECHANICS_AUTHORITY_SHA256",
    "LIMIT_AUTHORITY_SHA256",
    "PRECONTACT_CONTRACT_SHA256",
    "PRODUCTION_ASSET_SHA256",
    "bind_contact_free_candidate_a",
    "repository_root",
    "resolve_contact_free_candidate_a",
]
