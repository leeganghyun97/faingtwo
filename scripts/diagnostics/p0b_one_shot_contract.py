# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure, fail-closed contract helpers for the current-generation P0-B child.

Nothing in this module launches Isaac, creates an environment, starts a
process, or submits an action.  It exists so the P0-B child and its parent can
share one typed interpretation of the frozen P0-A waiver, the legacy P0-B
semantic intent, and the multi-packet single-consumption ledger.

The legacy P0-B implementation is evidence for *what* is tested (approach,
close, retreat, open).  It is not an implementation dependency: it has a
legacy schema, a provider-less ``EESafetyValidator`` path, and a parent that
can promote directly to P1.  This module deliberately does not import it as a
Python namespace package.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "g2_p0_b_one_shot_contract_v1"
P0_A_PREREQUISITE_SCHEMA = "g2_p0_b_p0_a_waiver_prerequisite_v1"
P0_B_PACKET_LEDGER_SCHEMA = "g2_p0_b_packet_ledger_v1"
P0_B_WAIVER_CONSUMPTION_SCHEMA = "g2_p0_b_packet_ledger_consumption_v1"
P0_B_LAUNCH_AUTHORIZATION_SCHEMA = "g2_p0_b_one_shot_launch_authorization_v1"
P0_B_IDENTITY_RETRY_AUTHORIZATION_SCHEMA = "g2_p0_b_identity_normalization_retry_authorization_v1"
P0_B_IDENTITY_RETRY_AUTHORIZATION_ID = "P0_B_ONE_SHOT_RETRY_AFTER_IDENTITY_NORMALIZATION"
P0_B_BOOTSTRAP_RETRY_AUTHORIZATION_SCHEMA = "g2_p0_b_pre_action_bootstrap_retry_authorization_v1"
P0_B_MANIFEST_RETRY_AUTHORIZATION_ID = "P0_B_BOOTSTRAP_RETRY_AFTER_MANIFEST_SCHEMA_BINDING"
P0_B_AXIS_REBASE_FRESH_AUTHORIZATION_SCHEMA = "g2_p0_b_axis_rebase_fresh_authorization_v1"
P0_B_AXIS_REBASE_FRESH_AUTHORIZATION_ID = "P0_B_FRESH_ONE_SHOT_AFTER_AXIS_REBASE_FIX"
PHYSX_OBSERVED_TUPLE_PREFIX = (110, 1, 13, "")


def repository_root_from_contract(path: Path | None = None) -> Path:
    """Resolve the repository root from this source file, never from CWD."""

    contract = (path or Path(__file__)).expanduser().resolve()
    try:
        root = contract.parents[2]
    except IndexError as error:
        raise RuntimeError(f"P0_B_REPOSITORY_ROOT_DERIVATION_FAILED:{contract}") from error
    expected = root / "scripts/diagnostics/p0b_one_shot_contract.py"
    if contract != expected.resolve():
        raise RuntimeError(
            "P0_B_REPOSITORY_ROOT_CONTRACT_PATH_MISMATCH:"
            + repr({"actual": str(contract), "expected": str(expected.resolve())})
        )
    if not (root / "source").is_dir():
        raise RuntimeError(f"P0_B_REPOSITORY_ROOT_SOURCE_DIRECTORY_MISSING:{root}")
    return root


ROOT = repository_root_from_contract()
LEGACY_ATTESTATION_SOURCE = ROOT / "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py"
P0_A_ONE_SHOT_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0a_one_shot_integration.py"
P0_B_CHILD_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0b_one_shot_integration.py"
P0_B_SUPERVISOR_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0b_one_shot_integration_supervisor.py"
WAIVER_SOURCE = ROOT / "scripts/diagnostics/p0_known_isaac_finalization_waiver.py"
ACTION_INTERFACE_SOURCE = ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/action_interface.py"
P0_SCOPE_AUDIT = ROOT / "output/policy_branch_p0/p0_scope_1_provenance_20260920_final/P0_SCOPE_1_PROVENANCE.md"
EE_SAFETY_VALIDATOR_SOURCE = ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/ee_safety_validator.py"
NEW_POLICY_SAFETY_BRIDGE_SOURCE = ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/new_policy_safety_bridge.py"
PRODUCTION_USD = ROOT / "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
FROZEN_TRAJECTORY = ROOT / "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
EXPECTED_PRODUCTION_USD_SHA256 = "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
EXPECTED_FROZEN_TRAJECTORY_SHA256 = "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"
DEFAULT_P0_A_WAIVER_RECEIPT = (
    ROOT
    / "output/policy_branch_p0/p0_known_isaac_finalization_waiver_20260920_v2/"
    "P0_KNOWN_ISAAC_FINALIZATION_WAIVER.json"
)
P0_B_STATIC_TEST_SOURCES = (
    ROOT / "tests/test_g2_p0_known_isaac_finalization_waiver.py",
    ROOT / "tests/test_g2_p0b_one_shot_integration.py",
    ROOT / "tests/test_g2_p0a_one_shot_integration.py",
    ROOT / "tests/test_g2_policy_branch_p0_attestation_static.py",
)

# These are the P0-A source authorities recorded by the frozen functional
# artifact, not a hand-maintained list inferred from the current tree.  They
# must still be byte-identical before P0-B may reuse P0-A's factory/lifecycle.
P0_A_FROZEN_SOURCE_REQUIRED_KEYS = (
    "configs/diagnostics/m2_arm_move_target_seed42_v1.json",
    "scripts/diagnostics/run_g2_p0a_one_shot_integration.py",
    "scripts/diagnostics/run_g2_p0a_one_shot_integration_supervisor.py",
    "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py",
    "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda",
    "source/geniesim/rl/isaaclab/g2_policy_branch/action_interface.py",
    "source/geniesim/rl/isaaclab/g2_policy_branch/p0_read_only_observation.py",
    "source/geniesim/rl/isaaclab/g2_policy_branch/production_metric_adapter.py",
    "source/geniesim/rl/isaaclab/g2_redundancy_action.py",
    "source/geniesim/rl/isaaclab/g2_redundancy_teleop_env_cfg.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a finite JSON value deterministically for evidence hashing."""

    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def strict_json_load(path: Path) -> dict[str, Any]:
    """Read one finite JSON object; malformed/non-finite input is fail-closed."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"P0_B_NONFINITE_JSON_CONSTANT:{value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"P0_B_STRICT_JSON_LOAD_FAILED:{path}:{type(error).__name__}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"P0_B_STRICT_JSON_NOT_OBJECT:{path}")
    # ``allow_nan=False`` catches Python float values too, not only JSON's
    # textual NaN/Infinity constants.
    canonical_json_bytes(payload)
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalize_physx_runtime_identity(raw: Any, *, frozen_authority: str) -> dict[str, Any]:
    """Normalize only the two proven PhysX version representations.

    The frozen authority is the canonical extension build string recorded by
    P0-A.  The current Kit API may return that exact string directly or the
    observed five-field tuple ``(110, 1, 13, '', build_string)``.  No
    substring, prefix, or fuzzy comparison is permitted.
    """

    raw_type = type(raw).__name__
    raw_evidence: Any = list(raw) if isinstance(raw, (tuple, list)) else raw
    normalized: str | None = None
    representation: str | None = None
    error: str | None = None
    if not isinstance(frozen_authority, str) or not frozen_authority:
        error = "INVALID_FROZEN_AUTHORITY"
    elif isinstance(raw, str):
        if raw == frozen_authority:
            normalized = raw
            representation = "CANONICAL_BUILD_STRING"
        else:
            error = "CANONICAL_BUILD_MISMATCH"
    elif isinstance(raw, (tuple, list)):
        valid_shape = bool(
            len(raw) == 5
            and all(isinstance(value, int) and not isinstance(value, bool) for value in raw[:3])
            and isinstance(raw[3], str)
            and isinstance(raw[4], str)
        )
        if not valid_shape:
            error = "MALFORMED_VERSION_TUPLE"
        elif tuple(raw[:4]) != PHYSX_OBSERVED_TUPLE_PREFIX:
            error = "UNKNOWN_VERSION_TUPLE_REPRESENTATION"
        elif raw[4] != frozen_authority:
            error = "VERSION_TUPLE_BUILD_MISMATCH"
        else:
            normalized = raw[4]
            representation = "KIT_EXTENSION_VERSION_TUPLE_V1"
    else:
        error = "UNKNOWN_RUNTIME_IDENTITY_REPRESENTATION"
    return {
        "status": "PASS" if error is None else "FAIL_CLOSED",
        "PHYSX_RUNTIME_IDENTITY_RAW": raw_evidence,
        "PHYSX_RUNTIME_IDENTITY_RAW_TYPE": raw_type,
        "PHYSX_RUNTIME_IDENTITY_NORMALIZED": normalized,
        "PHYSX_FROZEN_AUTHORITY": frozen_authority,
        "representation": representation,
        "error": error,
    }


def _load_waiver_module() -> Any:
    """Load the one waiver evaluator by absolute file path, without fallback."""

    source = WAIVER_SOURCE.resolve()
    if not source.is_file():
        raise RuntimeError(f"P0_B_WAIVER_SOURCE_MISSING:{source}")
    source_hash = sha256(source)
    name = f"_g2_p0_b_waiver_{source_hash[:16]}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("P0_B_WAIVER_SPEC_CREATION_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    if Path(getattr(module, "__file__", "")).resolve() != source:
        raise RuntimeError("P0_B_WAIVER_MODULE_PROVENANCE_MISMATCH")
    if sha256(source) != source_hash:
        raise RuntimeError("P0_B_WAIVER_SOURCE_CHANGED_DURING_LOAD")
    if not callable(getattr(module, "p0_b_launch_authorized", None)):
        raise RuntimeError("P0_B_WAIVER_LAUNCH_AUTHORIZER_MISSING")
    if not callable(getattr(module, "evaluate_waiver", None)):
        raise RuntimeError("P0_B_WAIVER_EVALUATOR_MISSING")
    return module


def _ast_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_ast_literal(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_ast_literal(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_literal(node.operand)
        if isinstance(value, (int, float)):
            return -value
    raise ValueError(f"P0_B_UNSUPPORTED_LEGACY_AST_LITERAL:{ast.dump(node, include_attributes=False)}")


def _legacy_sequence_node() -> ast.Tuple:
    tree = ast.parse(LEGACY_ATTESTATION_SOURCE.read_text(encoding="utf-8"), filename=str(LEGACY_ATTESTATION_SOURCE))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_p0_b"
        ),
        None,
    )
    if function is None:
        raise RuntimeError("P0_B_LEGACY_FUNCTION_MISSING")
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "sequence"
            and isinstance(node.value, ast.Tuple)
        ):
            return node.value
    raise RuntimeError("P0_B_LEGACY_SEQUENCE_MISSING")


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attribute_name(node.value)}.{node.attr}"
    raise ValueError(f"P0_B_UNSUPPORTED_LEGACY_ATTRIBUTE:{ast.dump(node, include_attributes=False)}")


def extract_legacy_p0_b_semantic_intent() -> list[dict[str, Any]]:
    """Extract legacy P0-B command facts from source, rather than hand-copying.

    The returned ``legacy_hold_steps`` are provenance.  The current child uses
    the same bounded observation budget only through its separate current
    contract, not through the legacy router, schema, or supervisor.
    """

    rows: list[dict[str, Any]] = []
    for item in _legacy_sequence_node().elts:
        if not isinstance(item, ast.Dict):
            raise RuntimeError("P0_B_LEGACY_SEQUENCE_ROW_NOT_DICT")
        values: dict[str, Any] = {}
        for key_node, value_node in zip(item.keys, item.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise RuntimeError("P0_B_LEGACY_SEQUENCE_KEY_INVALID")
            key = key_node.value
            if key == "expected_intent":
                values[key] = _attribute_name(value_node).rsplit(".", 1)[-1]
            else:
                values[key] = _ast_literal(value_node)
        rows.append(
            {
                "label": values.get("label"),
                "high_level_4d": list(values.get("values", ())),
                "legacy_hold_steps": values.get("holds"),
                "expected_gripper_intent": values.get("expected_intent"),
                "expected_transition": values.get("expected_transition"),
                "motion_axis": values.get("motion_axis"),
            }
        )
    return rows


@dataclass(frozen=True)
class P0BSemanticCommand:
    sequence: int
    label: str
    high_level_4d: tuple[float, float, float, float]
    expected_gripper_intent: str
    expected_transition: bool
    motion_axis: int | None
    legacy_hold_steps: int

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "label": self.label,
            "high_level_4d": list(self.high_level_4d),
            "expected_gripper_intent": self.expected_gripper_intent,
            "expected_transition": self.expected_transition,
            "motion_axis": self.motion_axis,
            "legacy_hold_steps": self.legacy_hold_steps,
            "expected_metric_delta_m": [
                self.high_level_4d[0] * 0.0225,
                self.high_level_4d[1] * 0.0225,
                self.high_level_4d[2] * 0.0225,
            ],
        }


P0_B_SEMANTIC_COMMANDS: tuple[P0BSemanticCommand, ...] = (
    P0BSemanticCommand(1, "APPROACH_X", (0.10, 0.0, 0.0, 0.50), "OPEN", False, 0, 20),
    P0BSemanticCommand(2, "CLOSE", (0.0, 0.0, 0.0, 0.71), "CLOSE", True, None, 20),
    P0BSemanticCommand(3, "RETREAT_Z", (0.0, 0.0, 0.10, 0.50), "CLOSE", False, 2, 20),
    P0BSemanticCommand(4, "OPEN", (0.0, 0.0, 0.0, 0.29), "OPEN", True, None, 4),
)


def semantic_intent_report() -> dict[str, Any]:
    """Return documented legacy purpose and the current non-legacy mapping."""

    legacy = extract_legacy_p0_b_semantic_intent()
    current = [item.payload() for item in P0_B_SEMANTIC_COMMANDS]
    legacy_comparable = [
        {
            "label": item["label"],
            "high_level_4d": item["high_level_4d"],
            "legacy_hold_steps": item["legacy_hold_steps"],
            "expected_gripper_intent": item["expected_gripper_intent"],
            "expected_transition": item["expected_transition"],
            "motion_axis": item["motion_axis"],
        }
        for item in legacy
    ]
    current_comparable = [
        {
            key: item[key]
            for key in (
                "label",
                "high_level_4d",
                "legacy_hold_steps",
                "expected_gripper_intent",
                "expected_transition",
                "motion_axis",
            )
        }
        for item in current
    ]
    return {
        "schema": "g2_p0_b_semantic_intent_v1",
        "legacy_provenance": {
            "source": str(LEGACY_ATTESTATION_SOURCE),
            "source_sha256": sha256(LEGACY_ATTESTATION_SOURCE),
            "callable": "_run_p0_b",
            "semantic_sequence_source_lines": "2243-2276",
            "validity_source_lines": "2403-2448",
        },
        "P0_B_PURPOSE": "COMBINED_EE_GRIPPER_CONTROLLER_INTEGRATION_SEQUENCE_NOT_A_GRASP_TEST",
        "P0_B_REQUIRED_COMMANDS": current,
        "P0_B_COMMAND_ORDER": [command.label for command in P0_B_SEMANTIC_COMMANDS],
        "P0_B_EXPECTED_OBSERVATIONS": {
            "EE": "APPROACH_X and RETREAT_Z must preserve source-defined positive target/measured response semantics",
            "gripper": "OPEN -> CLOSE -> CLOSE -> OPEN via existing abstract controller state",
            "orientation_and_elbow": "every full-8D packet has zero RPY and zero elbow",
            "termination": "no unexpected termination or forbidden collision",
        },
        "P0_B_GRIPPER_BEHAVIOR": "HYSTERETIC_ABSTRACT_OPEN_CLOSE_ONLY_NO_INDIVIDUAL_OMNIPICKER_JOINT_COMMAND",
        "P0_B_PASS_CRITERIA": [
            "all bounded semantic frames retain exact-one canonical consumption",
            "command order and full-8D expansion match",
            "observed abstract gripper intent matches each phase",
            "required X/Z controller-target and measured-direction evidence passes source-defined checks",
            "no unexpected termination or forbidden collision",
        ],
        "P0_B_FAIL_CRITERIA": [
            "unknown or duplicate packet consumption",
            "noncanonical/partial action packet",
            "unexpected gripper state",
            "forbidden collision or unexpected termination",
            "missing pre-close artifact or lifecycle marker",
        ],
        "P0_B_RESET_REQUIREMENTS": "CURRENT_P0_A_SOURCE_DEFINED_RESET_OPEN_SETTLE_THEN_HISTORICAL_ZERO_BASELINE",
        "P0_B_STEP_BUDGET": {
            "semantic_direct_packets": 4,
            "legacy_observation_hold_packets": sum(command.legacy_hold_steps for command in P0_B_SEMANTIC_COMMANDS),
            "semantic_plus_hold_packets": 4 + sum(command.legacy_hold_steps for command in P0_B_SEMANTIC_COMMANDS),
            "current_execution_rule": "LEGACY_HOLDS_ARE_REEXPRESSED_AS_CURRENT_CANONICAL_FULL_8D_HOLD_PACKETS; LEGACY_ROUTER_SCHEMA_AND_SUPERVISOR_ARE_NOT_REUSED",
        },
        "legacy_semantic_match": legacy_comparable == current_comparable,
        "legacy_execution_assumptions_not_reused": [
            "legacy phase/p0_semantic_verdict prerequisite schema",
            "legacy PolicyCommandRouter plus provider-less EESafetyValidator",
            "legacy artifact-after-env-close ordering",
            "legacy scripts.diagnostics namespace import",
            "legacy direct P1 promotion",
        ],
    }


def _receipt_source_freeze_pass(receipt: Mapping[str, Any]) -> bool:
    evidence = _mapping(receipt.get("EVIDENCE"))
    return evidence.get("source_freeze") == "PASS" or receipt.get("SOURCE_FREEZE") == "PASS"


def _p0_a_functional_source_freeze(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Read and authenticate the frozen P0-A functional source map.

    P0-B cannot simply reuse a current P0-A helper because the helper's source
    might have drifted after P0-A was qualified.  The current waiver receipt
    already names the immutable pre-close artifact and its file SHA-256.  This
    helper verifies that link, then extracts the P0-A-owned local source map.
    It never rewrites P0-A history or accepts a fallback artifact.
    """

    evidence = _mapping(receipt.get("EVIDENCE"))
    descriptor = _mapping(evidence.get("functional_artifact"))
    raw_path = descriptor.get("path")
    expected_sha256 = descriptor.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("P0_B_P0_A_FUNCTIONAL_ARTIFACT_REFERENCE_INVALID")
    artifact_path = Path(raw_path).expanduser().resolve()
    if not artifact_path.is_file():
        raise RuntimeError(f"P0_B_P0_A_FUNCTIONAL_ARTIFACT_MISSING:{artifact_path}")
    actual_sha256 = sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("P0_B_P0_A_FUNCTIONAL_ARTIFACT_SHA256_MISMATCH")
    artifact = strict_json_load(artifact_path)
    source_freeze = _mapping(artifact.get("source_freeze"))
    local = _mapping(source_freeze.get("local_source_sha256"))
    missing = [key for key in P0_A_FROZEN_SOURCE_REQUIRED_KEYS if not isinstance(local.get(key), str)]
    if source_freeze.get("SOURCE_FREEZE") != "PASS" or missing:
        raise RuntimeError("P0_B_P0_A_FUNCTIONAL_SOURCE_FREEZE_INCOMPLETE:" + repr(missing))
    return {
        "functional_artifact_path": str(artifact_path),
        "functional_artifact_sha256": actual_sha256,
        "functional_artifact_payload_sha256": artifact.get("artifact_payload_sha256"),
        "p0_a_local_source_sha256": {key: str(local[key]) for key in P0_A_FROZEN_SOURCE_REQUIRED_KEYS},
        "p0_a_source_freeze_manifest_path": source_freeze.get("manifest_path"),
        "p0_a_source_freeze_manifest_sha256": source_freeze.get("manifest_sha256"),
    }


def adapt_current_p0_a_prerequisite(receipt: Mapping[str, Any], *, receipt_sha256: str | None = None) -> dict[str, Any]:
    """Convert a current P0-A waiver receipt into a typed P0-B prerequisite.

    This is a read-only adapter.  It never writes legacy ``phase`` or
    ``p0_semantic_verdict`` fields and will reject a record that only resembles
    the historical P0-A schema.
    """

    waiver = _load_waiver_module()
    authorization = _mapping(receipt.get("CAN_AUTHORIZE_PHASE"))
    try:
        frozen_p0_a_sources = _p0_a_functional_source_freeze(receipt)
        frozen_p0_a_sources_ok = True
    except BaseException as error:
        frozen_p0_a_sources = {"error": f"{type(error).__name__}:{error}"}
        frozen_p0_a_sources_ok = False
    checks = {
        "waiver_schema": receipt.get("schema") == waiver.WAIVER_SCHEMA,
        "waiver_id": receipt.get("WAIVER_ID") == waiver.WAIVER_ID,
        "waiver_enabled": receipt.get("WAIVER_ENABLED") == "YES",
        "source_phase": receipt.get("SOURCE_PHASE") == "P0_A",
        "p0_a_functional": receipt.get("P0_A_FUNCTIONAL") == "PASS",
        "p0_a_process_is_known_failure": receipt.get("P0_A_PROCESS") == "FAIL_KNOWN_ISAAC_FINALIZATION",
        "p0_a_promotion": receipt.get("P0_A_PROMOTION_STATUS") == "PASS_WITH_KNOWN_INFRASTRUCTURE_WAIVER",
        "p0_b_authorized": receipt.get("P0_B_AUTHORIZED_NEXT") == "YES_WITH_INFRASTRUCTURE_WAIVER",
        "p0_b_only_authority": authorization.get("P0_B") is True,
        "no_downstream_authority": (
            authorization.get("P0_C") is False
            and authorization.get("M1") is False
            and authorization.get("M2") is False
            and authorization.get("TRAINING") is False
        ),
        "source_freeze": _receipt_source_freeze_pass(receipt),
        "p0_a_functional_source_freeze": frozen_p0_a_sources_ok,
        "waiver_helper": waiver.p0_b_launch_authorized(receipt) is True,
        "legacy_schema_not_required": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL_CLOSED"
    evidence = _mapping(receipt.get("EVIDENCE"))
    runtime = _mapping(evidence.get("runtime"))
    return {
        "schema": P0_A_PREREQUISITE_SCHEMA,
        "status": status,
        "checks": checks,
        "source_receipt_sha256": receipt_sha256,
        "source_waiver_schema": receipt.get("schema"),
        "source_waiver_id": receipt.get("WAIVER_ID"),
        "source_phase": receipt.get("SOURCE_PHASE"),
        "source_p0_a_functional": receipt.get("P0_A_FUNCTIONAL"),
        "source_p0_a_promotion_status": receipt.get("P0_A_PROMOTION_STATUS"),
        "source_p0_b_authorized_next": receipt.get("P0_B_AUTHORIZED_NEXT"),
        "source_runtime": dict(runtime),
        "p0_a_frozen_source_authority": frozen_p0_a_sources,
        "legacy_fields_fabricated": False,
        "legacy_phase_field_required": False,
        "legacy_p0_semantic_verdict_required": False,
        "P0_B_PREREQUISITE": "PASS" if status == "PASS" else "FAIL_CLOSED",
    }


def load_and_adapt_p0_a_prerequisite(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    receipt = strict_json_load(resolved)
    adapter = adapt_current_p0_a_prerequisite(receipt, receipt_sha256=sha256(resolved))
    return {**adapter, "source_receipt_path": str(resolved)}


def current_ee_safety_binding() -> dict[str, Any]:
    """State the P0-scope-specific conclusion about the legacy validator.

    The result is deliberately narrow: it does *not* prove pre-controller IK,
    workspace, collision, or physical safety.  It only establishes that those
    claims were not part of the original P0 integration contract and that the
    legacy validator cannot be used as a current acceptance authority.
    """

    required = (
        P0_SCOPE_AUDIT,
        LEGACY_ATTESTATION_SOURCE,
        EE_SAFETY_VALIDATOR_SOURCE,
        NEW_POLICY_SAFETY_BRIDGE_SOURCE,
        ACTION_INTERFACE_SOURCE,
        P0_A_ONE_SHOT_SOURCE,
        P0_B_CHILD_SOURCE,
    )
    present = {str(path): path.is_file() for path in required}
    legacy_text = LEGACY_ATTESTATION_SOURCE.read_text(encoding="utf-8") if LEGACY_ATTESTATION_SOURCE.is_file() else ""
    validator_text = EE_SAFETY_VALIDATOR_SOURCE.read_text(encoding="utf-8") if EE_SAFETY_VALIDATOR_SOURCE.is_file() else ""
    bridge_text = NEW_POLICY_SAFETY_BRIDGE_SOURCE.read_text(encoding="utf-8") if NEW_POLICY_SAFETY_BRIDGE_SOURCE.is_file() else ""
    scope_text = P0_SCOPE_AUDIT.read_text(encoding="utf-8") if P0_SCOPE_AUDIT.is_file() else ""
    action_text = ACTION_INTERFACE_SOURCE.read_text(encoding="utf-8") if ACTION_INTERFACE_SOURCE.is_file() else ""
    p0a_text = P0_A_ONE_SHOT_SOURCE.read_text(encoding="utf-8") if P0_A_ONE_SHOT_SOURCE.is_file() else ""
    child_text = P0_B_CHILD_SOURCE.read_text(encoding="utf-8") if P0_B_CHILD_SOURCE.is_file() else ""
    checks = {
        "p0_scope_integration_gate": "P0_SCOPE: INTEGRATION_GATE" in scope_text,
        "legacy_p0_b_constructs_providerless_validator": "EESafetyValidator()" in legacy_text,
        "legacy_p0_b_uses_legacy_router_receipt": "all_routes_accepted" in legacy_text,
        "legacy_default_provider_is_unresolved": "UnresolvedDryRunProvider" in validator_text,
        "current_snapshot_provider_is_protocol": "class ReadOnlyProductionIKSnapshotProvider(Protocol)" in bridge_text,
        "current_full8d_action_contract": all(
            token in action_text
            for token in (
                "class HighLevelPolicyAction",
                "class GripperHysteresisLatch",
                "def expand_to_existing_controller_8d",
            )
        ),
        "frozen_p0_a_builder_is_current_route": all(
            token in p0a_text
            for token in (
                "def _build_authoritative_packet",
                "expand_to_existing_controller_8d",
                "NormalizedProductionActionManagerPacket",
            )
        ),
        "current_p0_b_does_not_construct_legacy_or_new_safety_authority": all(
            token not in child_text
            for token in (
                "EESafetyValidator(",
                "PolicyCommandRouter(",
                "NewPolicySafetyPreControllerBridge(",
            )
        ),
        "current_p0_b_binds_frozen_full8d_env_step_route": all(
            token in child_text
            for token in (
                "_build_authoritative_packet",
                "DeferredFull8DActionPacketPort",
                "self._env.step(ingress_tensor)",
            )
        ),
        "no_new_authority_constructed": True,
        "all_evidence_sources_present": all(present.values()),
    }
    return {
        "P0_B_EE_SAFETY_BINDING": "LEGACY_ONLY_NOT_SEMANTIC" if all(checks.values()) else "UNRESOLVED",
        "CURRENT_P0_B_INTEGRATION_BINDING": (
            "BOUND_EXISTING_CURRENT_FULL8D_ENVS_STEP_INTERFACE" if all(checks.values()) else "UNRESOLVED"
        ),
        "PRE_CONTROLLER_ACCEPTANCE_AUTHORITY": "ABSENT",
        "NEW_AUTHORITY_REQUIRED_IF_PRE_CONTROLLER_PROOF_IS_DEMANDED": True,
        "checks": checks,
        "evidence_sources": {
            "p0_scope_audit": f"{P0_SCOPE_AUDIT}:5-31",
            "legacy_providerless_validator": f"{LEGACY_ATTESTATION_SOURCE}:363-399",
            "validator_default_provider": f"{EE_SAFETY_VALIDATOR_SOURCE}:202-219,337-353,517-545",
            "current_snapshot_protocol": f"{NEW_POLICY_SAFETY_BRIDGE_SOURCE}:33-45",
            "current_full_8d_interface": f"{ACTION_INTERFACE_SOURCE}:92-126,199-255,272-286",
            "frozen_p0_a_packet_builder": f"{P0_A_ONE_SHOT_SOURCE}:904-935",
            "current_p0_b_canonical_ingress": f"{P0_B_CHILD_SOURCE}:254-474",
        },
    }


def full_8d_authority_report() -> dict[str, Any]:
    """Prove the current full-8D route without treating a test oracle as authority.

    ``full_8d_for_command`` below is intentionally only a deterministic replay
    oracle.  This report binds the real source-owned sequence used by the
    future child: high-level action -> hysteresis latch -> existing 8-D
    expansion -> normalized ActionManager packet -> deferred canonical
    ``env.step`` ingress.
    """

    required = (ACTION_INTERFACE_SOURCE, P0_A_ONE_SHOT_SOURCE, P0_B_CHILD_SOURCE)
    texts = {path: path.read_text(encoding="utf-8") if path.is_file() else "" for path in required}
    action_text = texts[ACTION_INTERFACE_SOURCE]
    p0a_text = texts[P0_A_ONE_SHOT_SOURCE]
    child_text = texts[P0_B_CHILD_SOURCE]
    checks = {
        "action_interface_source_present": ACTION_INTERFACE_SOURCE.is_file(),
        "p0_a_builder_source_present": P0_A_ONE_SHOT_SOURCE.is_file(),
        "p0_b_child_source_present": P0_B_CHILD_SOURCE.is_file(),
        "high_level_and_hysteresis": all(
            token in action_text
            for token in ("HighLevelPolicyAction", "GripperHysteresisLatch", "AbstractGripperIntent")
        ),
        "existing_8d_expansion": "def expand_to_existing_controller_8d" in action_text,
        "normalized_packet": "NormalizedProductionActionManagerPacket" in p0a_text,
        "p0_a_authoritative_builder": "def _build_authoritative_packet" in p0a_text,
        "deferred_full8d_port": "DeferredFull8DActionPacketPort" in child_text,
        "canonical_env_step_only": "self._env.step(ingress_tensor)" in child_text,
        "no_direct_action_manager_call": "action_manager.process_action(" not in child_text,
        "no_direct_physics_step": ".sim.step(" not in child_text,
    }
    return {
        "FULL_8D_SCHEMA": "PASS_STATIC_SOURCE_BOUND" if all(checks.values()) else "FAIL_CLOSED",
        "authority": "HighLevelPolicyAction->GripperHysteresisLatch->expand_to_existing_controller_8d->NormalizedProductionActionManagerPacket->DeferredFull8DActionPacketPort->ManagerBasedRLEnv.step",
        "checks": checks,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) if path.is_file() else None for path in required
        },
        "reference_oracle_only": "full_8d_for_command",
        "direct_physics_bypass": "NONE",
    }


def full_8d_for_command(command: P0BSemanticCommand) -> tuple[float, float, float, float, float, float, float, float]:
    """Pure reference expansion of the frozen high-level mapping.

    Live execution must call the existing source-owned expansion function too;
    this reference value is a static ordering/replay regression, not a second
    controller implementation.
    """

    gripper_sign = 1.0 if command.expected_gripper_intent == "OPEN" else -1.0
    return (*command.high_level_4d[:3], 0.0, 0.0, 0.0, 0.0, gripper_sign)


def _finite_8d(packet: Sequence[Any]) -> bool:
    return len(packet) == 8 and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in packet)


_DEFERRED_FULL_8D_SCHEMA = "g2_r0g_deferred_full_8d_packet_v1"
_PRODUCTION_FULL_8D_FIELDS = (
    "root_translation_x_normalized",
    "root_translation_y_normalized",
    "root_translation_z_normalized",
    "root_rotation_vector_x_normalized",
    "root_rotation_vector_y_normalized",
    "root_rotation_vector_z_normalized",
    "elbow_nullspace_normalized",
    "abstract_gripper_sign",
)


def _flatten_finite_numbers(value: Any) -> list[float] | None:
    """Flatten a JSON tensor capture without accepting booleans/NaN/Inf."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else None
    if not isinstance(value, (list, tuple)):
        return None
    flattened: list[float] = []
    for item in value:
        nested = _flatten_finite_numbers(item)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    """Return one finite numeric vector without accepting nested shapes."""

    if not isinstance(value, list) or len(value) != length:
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def _capture_matches_packet(capture: Mapping[str, Any], *, packet: Sequence[Any], context: Any) -> bool:
    """Verify a real P0-A passive capture against one full-8D packet.

    Captures are serialized from the actual builder/deferred/env.step/
    ActionManager tensors.  Values may be IEEE float32 while the semantic
    packet is a Python float, so we use a deliberately tight numeric comparison
    and require the three runtime capture SHA-256 values to agree separately.
    """

    shape = capture.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in shape)
        or shape[0] <= 0
        or shape[1] != 8
        or capture.get("context") != context
        or not isinstance(capture.get("sha256"), str)
        or len(str(capture.get("sha256"))) != 64
    ):
        return False
    values = _flatten_finite_numbers(capture.get("values"))
    if values is None or len(values) != shape[0] * 8:
        return False
    expected = [float(component) for _ in range(shape[0]) for component in packet]
    # ``_LifecycleCounter._packet_record`` hashes the contiguous CPU float32
    # tensor bytes.  Recompute that digest instead of trusting a child-provided
    # hash as an identity assertion.
    capture_bytes = b"".join(struct.pack("<f", float(value)) for value in values)
    capture_hash = hashlib.sha256(capture_bytes).hexdigest()
    return bool(
        capture.get("sha256") == capture_hash
        and all(
            math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-6)
            for actual, wanted in zip(values, expected)
        )
    )


def _deferred_packet_receipt_matches(
    record: Mapping[str, Any], *, packet: Sequence[Any], sequence: int, capture: Mapping[str, Any]
) -> bool:
    """Recompute the deferred envelope fingerprint from stored immutable data."""

    envelope = _mapping(record.get("deferred_envelope"))
    shape = capture.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        return False
    payload = {
        "schema": envelope.get("schema"),
        "binding_id": envelope.get("binding_id"),
        "batch_shape": envelope.get("batch_shape"),
        "device": envelope.get("device"),
        "dtype": envelope.get("dtype"),
        "semantic_field_order": envelope.get("semantic_field_order"),
        "values_float_hex": envelope.get("values_float_hex"),
    }
    expected_fingerprint = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    fingerprint = record.get("deferred_content_fingerprint")
    binding_id = envelope.get("binding_id")
    acknowledgement = _mapping(record.get("deferred_acknowledgement"))
    return bool(
        envelope.get("schema") == _DEFERRED_FULL_8D_SCHEMA
        and isinstance(binding_id, str)
        and binding_id
        and envelope.get("sequence_id") == sequence
        and envelope.get("batch_shape") == shape
        and isinstance(envelope.get("device"), str)
        and envelope.get("device")
        and envelope.get("dtype") == "torch.float32"
        and envelope.get("semantic_field_order") == list(_PRODUCTION_FULL_8D_FIELDS)
        and envelope.get("values_float_hex") == [float(value).hex() for value in packet]
        and fingerprint == expected_fingerprint
        and record.get("deferred_packet_id") == f"{binding_id}:{sequence:016d}:{expected_fingerprint[:16]}"
        and acknowledgement.get("packet_id") == record.get("deferred_packet_id")
        and acknowledgement.get("content_fingerprint") == expected_fingerprint
        and acknowledgement.get("binding_id") == binding_id
        and acknowledgement.get("state") == "CONSUMED"
    )


def packet_fingerprint(*, sequence: int, role: str, semantic_command: str, full_8d_packet: Sequence[Any]) -> str:
    """Hash immutable identity, including sequence so equal hold packets differ."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "g2_p0_b_packet_fingerprint_v1",
                "packet_sequence": sequence,
                "packet_role": role,
                "semantic_command": semantic_command,
                "full_8d_packet": [float(value) for value in full_8d_packet],
            }
        )
    ).hexdigest()


def consumption_state(
    *,
    submission_started: bool,
    env_step_returned: bool,
    router_process_action_count: int | None,
    deferred_packet_count: int | None,
    env_step_calls: int | None,
    process_action_count: int | None,
) -> str:
    if not submission_started:
        return "NOT_SUBMITTED"
    counts = (router_process_action_count, deferred_packet_count, env_step_calls, process_action_count)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        return "CONSUMPTION_UNKNOWN"
    if not env_step_returned:
        return "CONSUMPTION_UNKNOWN"
    if (
        router_process_action_count == 0
        and deferred_packet_count == 1
        and env_step_calls == 1
        and process_action_count == 1
    ):
        return "KNOWN_SINGLE_CONSUMPTION"
    if any(value > 1 for value in (router_process_action_count, deferred_packet_count, env_step_calls, process_action_count)):
        return "DUPLICATE_CONSUMPTION"
    return "CONSUMPTION_UNKNOWN"


def validate_packet_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation for P0-B's immutable multi-packet evidence."""

    records = ledger.get("records")
    errors: list[str] = []
    if ledger.get("schema") != P0_B_PACKET_LEDGER_SCHEMA:
        errors.append("SCHEMA")
    if not isinstance(records, list) or not records:
        errors.append("RECORDS")
        records = []
    seen: set[str] = set()
    semantic: list[Mapping[str, Any]] = []
    for expected_sequence, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            errors.append(f"ROW_{expected_sequence}_NOT_MAPPING")
            continue
        if record.get("packet_sequence") != expected_sequence:
            errors.append(f"ROW_{expected_sequence}_SEQUENCE")
        packet = record.get("full_8d_packet")
        if not isinstance(packet, list) or not _finite_8d(packet):
            errors.append(f"ROW_{expected_sequence}_FULL_8D")
        fingerprint = record.get("packet_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or fingerprint in seen:
            errors.append(f"ROW_{expected_sequence}_FINGERPRINT")
        else:
            seen.add(fingerprint)
            if isinstance(packet, list) and _finite_8d(packet):
                expected = packet_fingerprint(
                    sequence=expected_sequence,
                    role=str(record.get("packet_role")),
                    semantic_command=str(record.get("semantic_command")),
                    full_8d_packet=packet,
                )
                if fingerprint != expected:
                    errors.append(f"ROW_{expected_sequence}_FINGERPRINT_CONTENT")
        values = {
            "submission_started": record.get("submission_started"),
            "env_step_returned": record.get("env_step_returned"),
            "router_process_action_count": record.get("router_process_action_count"),
            "deferred_packet_count": record.get("deferred_packet_count"),
            "env_step_calls": record.get("env_step_calls"),
            "process_action_count": record.get("process_action_count"),
        }
        expected_state = consumption_state(**values)
        if record.get("consumption_state") != expected_state or expected_state != "KNOWN_SINGLE_CONSUMPTION":
            errors.append(f"ROW_{expected_sequence}_CONSUMPTION")
        if (
            not isinstance(record.get("deferred_packet_id"), str)
            or not record.get("deferred_packet_id")
            or not isinstance(record.get("deferred_content_fingerprint"), str)
            or len(str(record.get("deferred_content_fingerprint"))) != 64
            or record.get("deferred_claim_count") != 1
            or record.get("deferred_acknowledgement_count") != 1
        ):
            errors.append(f"ROW_{expected_sequence}_DEFERRED_RECEIPT")
        identity = _mapping(record.get("canonical_packet_identity"))
        builder_capture = _mapping(identity.get("authoritative_p0_a_builder"))
        deferred_capture = _mapping(identity.get("deferred_env_step_tensor"))
        env_step_packets = identity.get("env_step_packets")
        process_action_packets = identity.get("process_action_packets")
        runtime_captures = (
            builder_capture,
            deferred_capture,
            _mapping(env_step_packets[0]) if isinstance(env_step_packets, list) and len(env_step_packets) == 1 else {},
            _mapping(process_action_packets[0])
            if isinstance(process_action_packets, list) and len(process_action_packets) == 1
            else {},
        )
        capture_values_valid = bool(
            isinstance(packet, list)
            and _finite_8d(packet)
            and all(_capture_matches_packet(capture, packet=packet, context=record.get("label")) for capture in runtime_captures)
        )
        capture_hashes = [capture.get("sha256") for capture in runtime_captures]
        capture_hashes_match = bool(
            len(capture_hashes) == 4
            and all(isinstance(value, str) and len(value) == 64 for value in capture_hashes)
            and len(set(capture_hashes)) == 1
        )
        if not (
            capture_values_valid
            and capture_hashes_match
            and identity.get("builder_matches_deferred") is True
            and identity.get("env_step_matches_deferred") is True
            and identity.get("process_action_matches_deferred") is True
            and identity.get("same_immutable_packet_consumed_once") is True
        ):
            errors.append(f"ROW_{expected_sequence}_PACKET_IDENTITY")
        if not (
            isinstance(packet, list)
            and _finite_8d(packet)
            and _deferred_packet_receipt_matches(
                record, packet=packet, sequence=expected_sequence, capture=deferred_capture
            )
        ):
            errors.append(f"ROW_{expected_sequence}_DEFERRED_ENVELOPE")
        if record.get("packet_role") == "semantic":
            semantic.append(record)
    semantic_labels = [str(row.get("semantic_command")) for row in semantic]
    expected_labels = [command.label for command in P0_B_SEMANTIC_COMMANDS]
    if semantic_labels != expected_labels:
        errors.append("SEMANTIC_COMMAND_ORDER")
    if len(semantic) != 4:
        errors.append("SEMANTIC_COMMAND_COUNT")

    # P0-B is deliberately multi-packet.  A four-row ledger that happens to
    # contain the semantic labels is not evidence that its bounded command and
    # observation sequence actually ran.  Validate the current full-8D
    # re-expression of the legacy schedule:
    #
    #   [0..80 reset-open packets] -> five zero baseline packets
    #   -> direct semantic packet -> exact source-derived holds (each phase)
    #
    # Reset-open duration is runtime-state dependent, so only its bounded
    # source-defined range is variable.  Everything after it is exact.  This
    # is a receipt validator, not a second controller implementation.
    cursor = 0
    reset_open_packets = 0
    zero_open = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    while cursor < len(records):
        record = records[cursor]
        if not isinstance(record, Mapping):
            break
        if record.get("packet_role") != "setup" or record.get("semantic_command") != "RESET_OPEN_SETTLE":
            break
        if record.get("full_8d_packet") != zero_open or record.get("expected_gripper_intent") != "OPEN":
            errors.append(f"RESET_OPEN_PACKET_{cursor + 1}")
        reset_open_packets += 1
        cursor += 1
    if reset_open_packets > 80:
        errors.append("RESET_OPEN_SOURCE_CAP")
    for baseline_index in range(5):
        if cursor >= len(records) or not isinstance(records[cursor], Mapping):
            errors.append("ZERO_BASELINE_MISSING")
            break
        record = records[cursor]
        if not (
            record.get("packet_role") == "setup"
            and record.get("semantic_command") == "ZERO_BASELINE"
            and record.get("full_8d_packet") == zero_open
            and record.get("expected_gripper_intent") == "OPEN"
        ):
            errors.append(f"ZERO_BASELINE_PACKET_{baseline_index + 1}")
        cursor += 1
    for command in P0_B_SEMANTIC_COMMANDS:
        if cursor >= len(records) or not isinstance(records[cursor], Mapping):
            errors.append(f"SEMANTIC_PACKET_MISSING_{command.label}")
            break
        direct = records[cursor]
        expected_direct = list(full_8d_for_command(command))
        if not (
            direct.get("packet_role") == "semantic"
            and direct.get("semantic_command") == command.label
            and direct.get("full_8d_packet") == expected_direct
            and direct.get("expected_gripper_intent") == command.expected_gripper_intent
        ):
            errors.append(f"SEMANTIC_PACKET_CONTRACT_{command.label}")
        cursor += 1
        expected_hold = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, expected_direct[-1]]
        for hold_index in range(command.legacy_hold_steps):
            if cursor >= len(records) or not isinstance(records[cursor], Mapping):
                errors.append(f"HOLD_PACKET_MISSING_{command.label}_{hold_index + 1}")
                break
            hold = records[cursor]
            if not (
                hold.get("packet_role") == "hold"
                and hold.get("semantic_command") == command.label
                and hold.get("full_8d_packet") == expected_hold
                and hold.get("expected_gripper_intent") == command.expected_gripper_intent
            ):
                errors.append(f"HOLD_PACKET_CONTRACT_{command.label}_{hold_index + 1}")
            cursor += 1
    if cursor != len(records):
        errors.append("UNEXPECTED_PACKET_AFTER_BOUNDED_SEQUENCE")
    expected_packet_count = ledger.get("packet_count")
    if expected_packet_count != len(records):
        errors.append("PACKET_COUNT")
    totals = ledger.get("totals")
    if not isinstance(totals, Mapping):
        errors.append("TOTALS")
    else:
        expected_total = len(records)
        if (
            totals.get("router_process_action_calls") != 0
            or totals.get("env_step_calls") != expected_total
            or totals.get("action_manager_process_action_calls") != expected_total
            or totals.get("logical_packet_consumption_count") != expected_total
        ):
            errors.append("TOTALS_COUNTS")
    port_receipt = _mapping(ledger.get("deferred_port_receipt"))
    expected_total = len(records)
    if not (
        port_receipt.get("stage_count") == expected_total
        and port_receipt.get("claim_count") == expected_total
        and port_receipt.get("acknowledgement_count") == expected_total
        and port_receipt.get("outstanding_packet") is False
    ):
        errors.append("DEFERRED_PORT_TOTALS")
    termination = _mapping(ledger.get("termination_receipt"))
    if not (
        termination.get("provider") == "EXISTING_TERMINATION_MANAGER_POST_ENV_STEP"
        and termination.get("packet_count") == expected_total
        and termination.get("all_env_step_returned") is True
        and termination.get("terminated_packet_count") == 0
        and termination.get("truncated_packet_count") == 0
        and termination.get("forbidden_collision_observed") is False
        and termination.get("fixed_torso_drift_observed") is False
        and termination.get("complete") is True
    ):
        errors.append("TERMINATION_RECEIPT")
    return {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "errors": errors,
        "packet_count": len(records),
        "semantic_packet_count": len(semantic),
        "all_packet_consumption_known": not errors,
        "reset_open_settle_packet_count": reset_open_packets,
        "bounded_semantic_plus_hold_packet_count": 68,
    }


def waiver_consumption_payload(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a validated ledger to the existing waiver's strict P0-B mode."""

    validation = validate_packet_ledger(ledger)
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    totals = _mapping(ledger.get("totals"))
    return {
        "schema": P0_B_WAIVER_CONSUMPTION_SCHEMA,
        "known": validation["status"] == "PASS",
        "consumption_unknown": validation["status"] != "PASS",
        "packet_count": len(records),
        "semantic_packet_count": validation["semantic_packet_count"],
        "records": records,
        "env_step_calls": totals.get("env_step_calls"),
        "router_process_action_calls": totals.get("router_process_action_calls"),
        "action_manager_process_action_calls": totals.get("action_manager_process_action_calls"),
        "ledger_validation": validation,
    }


def validate_p0_b_functional_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_p0_a_receipt_sha256: str | None = None,
    expected_launch_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the child-owned pre-close artifact before waiver evaluation."""

    errors: list[str] = []
    if artifact.get("schema") != "g2_p0_b_one_shot_integration_v1":
        errors.append("SCHEMA")
    if artifact.get("phase") != "P0_B":
        errors.append("PHASE")
    source = _mapping(artifact.get("source_freeze"))
    if source.get("SOURCE_FREEZE") != "PASS":
        errors.append("SOURCE_FREEZE")
    if source.get("production_usd_sha256") != EXPECTED_PRODUCTION_USD_SHA256:
        errors.append("PRODUCTION_USD_HASH")
    if source.get("frozen_trajectory_sha256") != EXPECTED_FROZEN_TRAJECTORY_SHA256:
        errors.append("FROZEN_TRAJECTORY_HASH")
    p0b_sources = _mapping(source.get("p0_b_implementation_source_sha256"))
    for relative in (
        "scripts/diagnostics/p0b_one_shot_contract.py",
        "scripts/diagnostics/run_g2_p0b_one_shot_integration.py",
        "scripts/diagnostics/run_g2_p0b_one_shot_integration_supervisor.py",
        *(str(path.relative_to(ROOT)) for path in P0_B_STATIC_TEST_SOURCES),
    ):
        candidate = (ROOT / relative).resolve()
        if not candidate.is_file() or p0b_sources.get(relative) != sha256(candidate):
            errors.append(f"P0_B_SOURCE_FREEZE:{relative}")
    p0_a_comparison = _mapping(source.get("p0_a_frozen_source_comparison"))
    if not all(
        _mapping(p0_a_comparison.get(relative)).get("match") is True
        for relative in P0_A_FROZEN_SOURCE_REQUIRED_KEYS
    ):
        errors.append("P0_A_SOURCE_FREEZE_COMPARISON")
    factory = _mapping(artifact.get("factory_identity"))
    if not (
        factory.get("factory_source") == str(LEGACY_ATTESTATION_SOURCE.resolve())
        and factory.get("factory_callable") == "_make_env"
        and factory.get("requested_phase") == "P0_B"
        and factory.get("namespace_import_dependency") == "NONE"
        and factory.get("alternate_factory_fallback") == "NONE"
        and factory.get("factory_implementation_copied") is False
    ):
        errors.append("FACTORY_BINDING")
    invocation = _mapping(artifact.get("factory_invocation_receipt"))
    if not (
        invocation.get("factory_environment_invoked") is True
        and invocation.get("factory_invocation_count") == 1
        and invocation.get("requested_phase") == "P0_B"
        and invocation.get("factory_source") == str(LEGACY_ATTESTATION_SOURCE.resolve())
        and invocation.get("factory_callable") == "_make_env"
        and invocation.get("source_sha256_before") == sha256(LEGACY_ATTESTATION_SOURCE)
        and invocation.get("source_sha256_after") == sha256(LEGACY_ATTESTATION_SOURCE)
        and invocation.get("callable_source") == str(LEGACY_ATTESTATION_SOURCE.resolve())
        and invocation.get("callable_code_filename") == str(LEGACY_ATTESTATION_SOURCE.resolve())
    ):
        errors.append("FACTORY_INVOCATION_RECEIPT")
    prerequisite = _mapping(artifact.get("p0_a_prerequisite_receipt"))
    if prerequisite.get("schema") != P0_A_PREREQUISITE_SCHEMA or prerequisite.get("status") != "PASS":
        errors.append("P0_A_PREREQUISITE")
    if expected_p0_a_receipt_sha256 is not None and prerequisite.get("source_receipt_sha256") != expected_p0_a_receipt_sha256:
        errors.append("P0_A_PREREQUISITE_RECEIPT_HASH")
    contract_identity = _mapping(artifact.get("p0_b_contract_identity"))
    semantic_report = semantic_intent_report()
    if not (
        contract_identity.get("schema") == SCHEMA
        and contract_identity.get("semantic_intent_schema") == semantic_report.get("schema")
        and contract_identity.get("semantic_intent_source_sha256")
        == _mapping(semantic_report.get("legacy_provenance")).get("source_sha256")
    ):
        errors.append("P0_B_CONTRACT_IDENTITY")
    ledger = _mapping(artifact.get("packet_ledger"))
    ledger_validation = validate_packet_ledger(ledger)
    if ledger_validation.get("status") != "PASS":
        errors.append("PACKET_LEDGER")
    ledger_records_value = ledger.get("records")
    ledger_records = ledger_records_value if isinstance(ledger_records_value, list) else []
    ledger_by_fingerprint = {
        str(row.get("packet_fingerprint")): row
        for row in ledger_records
        if isinstance(row, Mapping) and isinstance(row.get("packet_fingerprint"), str)
    }

    semantic = artifact.get("semantic_command_sequence")
    validated_responses: list[Mapping[str, Any]] = []
    if not isinstance(semantic, list) or len(semantic) != len(P0_B_SEMANTIC_COMMANDS):
        errors.append("SEMANTIC_SEQUENCE")
    else:
        for index, (row, command) in enumerate(zip(semantic, P0_B_SEMANTIC_COMMANDS), start=1):
            if not isinstance(row, Mapping):
                errors.append(f"SEMANTIC_ROW_{index}")
                continue
            if not (
                row.get("label") == command.label
                and row.get("high_level_4d") == list(command.high_level_4d)
                and row.get("expected_gripper_intent") == command.expected_gripper_intent
                and row.get("expected_transition") == command.expected_transition
                and row.get("motion_axis") == command.motion_axis
                and row.get("legacy_hold_steps") == command.legacy_hold_steps
                and row.get("hold_packet_count") == command.legacy_hold_steps
            ):
                errors.append(f"SEMANTIC_ROW_CONTRACT_{command.label}")

            expected_direct = list(full_8d_for_command(command))
            direct = _mapping(row.get("direct_packet"))
            direct_fingerprint = direct.get("packet_fingerprint")
            direct_in_ledger = ledger_by_fingerprint.get(str(direct_fingerprint))
            if not (
                direct.get("packet_role") == "semantic"
                and direct.get("semantic_command") == command.label
                and direct.get("full_8d_packet") == expected_direct
                and direct.get("expected_gripper_intent") == command.expected_gripper_intent
                and direct.get("observed_gripper_intent") == command.expected_gripper_intent
                and isinstance(direct_in_ledger, Mapping)
                and canonical_json_bytes(direct) == canonical_json_bytes(direct_in_ledger)
            ):
                errors.append(f"SEMANTIC_DIRECT_PACKET_{command.label}")

            holds_value = row.get("hold_packets")
            holds = holds_value if isinstance(holds_value, list) else []
            expected_hold = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, expected_direct[-1]]
            if len(holds) != command.legacy_hold_steps:
                errors.append(f"SEMANTIC_HOLD_COUNT_{command.label}")
            for hold_index, hold_value in enumerate(holds, start=1):
                hold = _mapping(hold_value)
                hold_fingerprint = hold.get("packet_fingerprint")
                hold_in_ledger = ledger_by_fingerprint.get(str(hold_fingerprint))
                if not (
                    hold.get("packet_role") == "hold"
                    and hold.get("semantic_command") == command.label
                    and hold.get("full_8d_packet") == expected_hold
                    and hold.get("expected_gripper_intent") == command.expected_gripper_intent
                    and hold.get("observed_gripper_intent") == command.expected_gripper_intent
                    and isinstance(hold_in_ledger, Mapping)
                    and canonical_json_bytes(hold) == canonical_json_bytes(hold_in_ledger)
                ):
                    errors.append(f"SEMANTIC_HOLD_PACKET_{command.label}_{hold_index}")

            response = _mapping(row.get("response"))
            scale = _finite_vector(response.get("arm_action_scale_xyz_m_per_normalized"), length=3)
            expected_delta = _finite_vector(response.get("expected_metric_delta_root_m"), length=3)
            target_delta = _finite_vector(response.get("controller_target_delta_root_m_after_direct"), length=3)
            formula_origin = _finite_vector(
                response.get("controller_formula_origin_root_m"), length=3
            )
            measured_delta = _finite_vector(response.get("measured_ee_delta_root_m_after_holds"), length=3)
            expected_from_scale = (
                [float(command.high_level_4d[axis]) * scale[axis] for axis in range(3)]
                if scale is not None
                else None
            )
            target_error = response.get("controller_target_error_m")
            measured_axis_delta = response.get("measured_axis_delta_m")
            orientation_drift = response.get("orientation_drift_rad")
            orientation_limit = response.get("orientation_source_limit_rad")
            scalars_finite = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (target_error, measured_axis_delta, orientation_drift, orientation_limit)
            )
            after_direct = _mapping(row.get("after_direct"))
            actual_target = _finite_vector(
                _mapping(
                    _mapping(_mapping(after_direct.get("sample")).get("target_cache")).get(
                        "desired_ee_pose_root"
                    )
                ).get("position_m"),
                length=3,
            )
            recomputed_target_error = (
                max(
                    abs(actual - (origin + delta))
                    for actual, origin, delta in zip(
                        actual_target, formula_origin, expected_delta
                    )
                )
                if command.motion_axis is not None
                and actual_target is not None
                and formula_origin is not None
                and expected_delta is not None
                else 0.0 if command.motion_axis is None else None
            )
            motion_axis_ok = response.get("motion_axis") == command.motion_axis
            motion_sign_ok = True
            if command.motion_axis is not None:
                motion_sign_ok = bool(
                    measured_delta is not None
                    and measured_delta[command.motion_axis] > 0.0
                    and isinstance(measured_axis_delta, (int, float))
                    and not isinstance(measured_axis_delta, bool)
                    and math.isclose(
                        float(measured_axis_delta),
                        measured_delta[command.motion_axis],
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                )
            if not (
                scale is not None
                and all(value > 0.0 for value in scale)
                and expected_delta is not None
                and expected_from_scale is not None
                and all(
                    math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-12)
                    for actual, wanted in zip(expected_delta, expected_from_scale)
                )
                and target_delta is not None
                and (
                    command.motion_axis is None
                    or (formula_origin is not None and actual_target is not None)
                )
                and measured_delta is not None
                and scalars_finite
                and recomputed_target_error is not None
                and float(target_error) <= 1.0e-6
                and recomputed_target_error <= 1.0e-6
                and motion_axis_ok
                and motion_sign_ok
                and float(orientation_limit) > 0.0
                and 0.0 <= float(orientation_drift) <= float(orientation_limit)
                and response.get("motion_pass") is True
                and response.get("orientation_pass") is True
                and row.get("gripper_pass") is True
                and bool(_mapping(row.get("before")))
                and bool(_mapping(row.get("after_direct")))
                and bool(_mapping(row.get("after_holds")))
            ):
                errors.append(f"SEMANTIC_RESPONSE_{command.label}")
            validated_responses.append(response)
    if artifact.get("P0_B_FUNCTIONAL") != "PASS":
        errors.append("FUNCTIONAL_VERDICT")
    initial_observation = _mapping(artifact.get("initial_observation"))
    final_observation = _mapping(artifact.get("final_observation"))
    ee_response = artifact.get("ee_response")
    gripper = _mapping(artifact.get("gripper_state"))
    if not initial_observation or not final_observation or not isinstance(ee_response, list) or len(ee_response) != 4:
        errors.append("OBSERVATION_OR_EE_RESPONSE")
    elif canonical_json_bytes(ee_response) != canonical_json_bytes(validated_responses):
        errors.append("EE_RESPONSE_SEMANTIC_MISMATCH")
    if not (
        gripper.get("initial") == "OPEN"
        and gripper.get("final") == "OPEN"
        and gripper.get("expected_final") == "OPEN"
    ):
        errors.append("GRIPPER_STATE")
    collision = _mapping(artifact.get("collision_state"))
    forbidden_collision = _mapping(artifact.get("forbidden_collision_state"))
    termination = _mapping(artifact.get("termination_truncation"))
    if not (
        collision.get("authority") == "EXISTING_TERMINATION_MANAGER_POST_ENV_STEP_RECEIPT"
        and collision.get("all_packet_active_terminations_empty") is True
        and forbidden_collision.get("authority") == "EXISTING_TERMINATION_MANAGER.forbidden_collision"
        and forbidden_collision.get("observed") is False
        and forbidden_collision.get("receipt_complete") is True
        and termination.get("terminated_packet_count") == 0
        and termination.get("truncated_packet_count") == 0
        and termination.get("fixed_torso_drift_observed") is False
        and termination.get("all_packets_clear") is True
    ):
        errors.append("COLLISION_TERMINATION_RECEIPT")
    launch_authorization = _mapping(artifact.get("launch_authorization"))
    launch_validation = _mapping(artifact.get("launch_authorization_validation"))
    launch_authorization_payload = {
        key: value
        for key, value in launch_authorization.items()
        if key not in {"authorization_file_path", "authorization_file_sha256"}
    }
    if not (
        launch_authorization.get("schema") == P0_B_LAUNCH_AUTHORIZATION_SCHEMA
        and launch_validation.get("status") == "PASS"
        and isinstance(launch_authorization.get("authorization_payload_sha256"), str)
        and launch_authorization.get("authorization_payload_sha256")
        == _payload_sha256(launch_authorization_payload, field="authorization_payload_sha256")
        and isinstance(launch_authorization.get("authorization_file_path"), str)
        and isinstance(launch_authorization.get("authorization_file_sha256"), str)
    ):
        errors.append("LAUNCH_AUTHORIZATION")
    if (
        expected_launch_authorization_sha256 is not None
        and launch_authorization.get("authorization_file_sha256") != expected_launch_authorization_sha256
    ):
        errors.append("LAUNCH_AUTHORIZATION_SHA256")
    runtime = _mapping(artifact.get("runtime_identity"))
    if not (
        runtime.get("provenance")
        == "LIVE_RUNTIME_READ_ONLY_COMPARED_TO_P0_A_FROZEN_WAIVER_FINGERPRINT"
        and runtime.get("runtime_identity_match_frozen_p0_a") is True
        and runtime.get("same_child_interpreter_as_stock_t0") is True
        and runtime.get("isaacsim_runtime_version") == "6.0.1"
        and runtime.get("kit_version") == "110.1.2+production.326809.f9bf0dda.gl"
        and runtime.get("python_major_minor") == "3.12"
        and runtime.get("physx_version") == "110.1.2.lx64.r.cp312.u7f4"
        and runtime.get("PHYSX_RUNTIME_IDENTITY_RAW")
        == [110, 1, 13, "", "110.1.2.lx64.r.cp312.u7f4"]
        and runtime.get("PHYSX_RUNTIME_IDENTITY_RAW_TYPE") == "tuple"
        and runtime.get("PHYSX_RUNTIME_IDENTITY_NORMALIZED") == "110.1.2.lx64.r.cp312.u7f4"
        and runtime.get("PHYSX_FROZEN_AUTHORITY") == "110.1.2.lx64.r.cp312.u7f4"
        and _mapping(runtime.get("physx_identity_normalization")).get("status") == "PASS"
        and runtime.get("identity_errors") == []
    ):
        errors.append("LIVE_RUNTIME_IDENTITY")
    checksum = artifact.get("artifact_payload_sha256")
    without_checksum = {key: value for key, value in artifact.items() if key != "artifact_payload_sha256"}
    expected_checksum = hashlib.sha256(canonical_json_bytes(without_checksum)).hexdigest()
    if not isinstance(checksum, str) or checksum != expected_checksum:
        errors.append("ARTIFACT_CHECKSUM")
    if artifact.get("functional_artifact_persisted_before_shutdown") is not True:
        errors.append("PRE_CLOSE_PERSISTENCE")
    preclose = _mapping(artifact.get("preclose_receipt"))
    required_preclose_false = (
        "initialization_failure",
        "physics_stepping_failure",
        "env_step_failure",
        "action_submission_failure",
        "unexpected_collision",
        "unexpected_termination",
        "new_preclose_stack_signature",
        "instrumentation_restore_failure",
    )
    if not all(preclose.get(key) is False for key in required_preclose_false) or preclose.get(
        "termination_receipt_complete"
    ) is not True:
        errors.append("PRECLOSE_RECEIPT")
    if artifact.get("P0_C") != "BLOCKED_NOT_EXECUTED":
        errors.append("P0_C_EXECUTION_BOUNDARY")
    if artifact.get("P1") != "BLOCKED" or artifact.get("M1") != "NOT_AUTHORIZED" or artifact.get("M2") != "NOT_AUTHORIZED" or artifact.get("TRAINING") != "NOT_AUTHORIZED":
        errors.append("DOWNSTREAM_PROMOTION_BOUNDARY")
    return {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "errors": errors,
        "ledger_validation": ledger_validation,
        "artifact_functional": artifact.get("P0_B_FUNCTIONAL"),
    }


def source_freeze_report(prerequisite: Mapping[str, Any]) -> dict[str, Any]:
    p0_a_authority = _mapping(prerequisite.get("p0_a_frozen_source_authority"))
    p0_a_manifest_path_raw = p0_a_authority.get("p0_a_source_freeze_manifest_path")
    p0_a_manifest_sha256 = p0_a_authority.get("p0_a_source_freeze_manifest_sha256")
    p0_a_manifest_path = (
        Path(p0_a_manifest_path_raw).expanduser().resolve()
        if isinstance(p0_a_manifest_path_raw, str) and p0_a_manifest_path_raw
        else None
    )
    p0_a_manifest_binding_match = bool(
        p0_a_manifest_path is not None
        and p0_a_manifest_path.is_file()
        and isinstance(p0_a_manifest_sha256, str)
        and sha256(p0_a_manifest_path) == p0_a_manifest_sha256
    )
    frozen_p0_a_hashes = _mapping(p0_a_authority.get("p0_a_local_source_sha256"))
    paths = (
        Path(__file__).resolve(),
        P0_B_CHILD_SOURCE,
        P0_B_SUPERVISOR_SOURCE,
        *P0_B_STATIC_TEST_SOURCES,
        P0_B_CHILD_SOURCE,
        P0_B_SUPERVISOR_SOURCE,
        LEGACY_ATTESTATION_SOURCE,
        P0_A_ONE_SHOT_SOURCE,
        WAIVER_SOURCE,
        ACTION_INTERFACE_SOURCE,
        EE_SAFETY_VALIDATOR_SOURCE,
        NEW_POLICY_SAFETY_BRIDGE_SOURCE,
        PRODUCTION_USD,
        FROZEN_TRAJECTORY,
    )
    # ``dict.fromkeys`` avoids a changed source appearing twice while keeping
    # deterministic provenance ordering in the report.
    unique_paths = tuple(dict.fromkeys(paths))
    present = {str(path.relative_to(ROOT)): path.is_file() for path in unique_paths if ROOT in path.parents}
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in unique_paths if path.is_file() and ROOT in path.parents}
    p0_a_comparison: dict[str, dict[str, Any]] = {}
    for relative, frozen_hash in frozen_p0_a_hashes.items():
        if not isinstance(relative, str) or not isinstance(frozen_hash, str):
            continue
        current_path = (ROOT / relative).resolve()
        current_hash = sha256(current_path) if current_path.is_file() else None
        p0_a_comparison[relative] = {
            "frozen_sha256": frozen_hash,
            "current_sha256": current_hash,
            "match": current_hash == frozen_hash,
        }
    required_p0_a_keys_present = all(key in p0_a_comparison for key in P0_A_FROZEN_SOURCE_REQUIRED_KEYS)
    p0_a_hashes_match = required_p0_a_keys_present and all(
        comparison["match"] is True for comparison in p0_a_comparison.values()
    )
    p0_b_implementation_paths = (
        "scripts/diagnostics/p0b_one_shot_contract.py",
        "scripts/diagnostics/run_g2_p0b_one_shot_integration.py",
        "scripts/diagnostics/run_g2_p0b_one_shot_integration_supervisor.py",
        *(str(path.relative_to(ROOT)) for path in P0_B_STATIC_TEST_SOURCES),
    )
    checks = {
        "all_sources_present": all(present.values()),
        "production_usd_hash": sha256(PRODUCTION_USD) == EXPECTED_PRODUCTION_USD_SHA256,
        "frozen_trajectory_hash": sha256(FROZEN_TRAJECTORY) == EXPECTED_FROZEN_TRAJECTORY_SHA256,
        "p0_a_prerequisite": prerequisite.get("status") == "PASS",
        "p0_a_frozen_source_keys_present": required_p0_a_keys_present,
        "p0_a_frozen_source_hashes_match": p0_a_hashes_match,
        "p0_a_source_freeze_manifest_binding": p0_a_manifest_binding_match,
        "current_p0_b_implementation_hashes_captured": all(path in hashes for path in p0_b_implementation_paths),
    }
    return {
        "SOURCE_FREEZE": "PASS" if all(checks.values()) else "FAIL",
        # P0-A's read-only observation binding consumes this exact immutable
        # manifest fingerprint.  Re-expose the already authenticated P0-A
        # authority; do not synthesize a second manifest or a replacement
        # fingerprint for P0-B.
        "manifest_path": str(p0_a_manifest_path) if p0_a_manifest_path is not None else None,
        "manifest_sha256": p0_a_manifest_sha256,
        "checks": checks,
        "source_sha256": hashes,
        "production_usd_sha256": hashes.get(str(PRODUCTION_USD.relative_to(ROOT))),
        "frozen_trajectory_sha256": hashes.get(str(FROZEN_TRAJECTORY.relative_to(ROOT))),
        "p0_a_frozen_source_authority": dict(p0_a_authority),
        "p0_a_frozen_source_comparison": p0_a_comparison,
        "p0_b_implementation_source_sha256": {path: hashes.get(path) for path in p0_b_implementation_paths},
    }


def _payload_sha256(payload: Mapping[str, Any], *, field: str) -> str:
    return hashlib.sha256(canonical_json_bytes({key: value for key, value in payload.items() if key != field})).hexdigest()


def build_p0_b_bootstrap_retry_authorization(*, previous_supervisor_report: Path) -> dict[str, Any]:
    """Authorize one replacement trial for a proven pre-action bootstrap bug.

    This does not erase or replenish the original budget.  It binds an
    additional, separately named one-child authorization to the immutable
    failed supervisor report and child artifact that prove no action was
    submitted before the representation-only failure.
    """

    report_path = previous_supervisor_report.expanduser().resolve()
    report = strict_json_load(report_path)
    if report.get("schema") == "g2_p0_b_rebase_semantics_offline_adjudication_v1":
        return build_p0_b_axis_rebase_fresh_authorization(adjudication_receipt=report_path)
    artifact_path = report_path.parent / "child" / "P0_B_FUNCTIONAL_PRE_CLOSE.json"
    artifact = strict_json_load(artifact_path)
    cleanup_path = report_path.parent / "child" / "P0_B_CHILD_CLEANUP_PRE_APP_CLOSE.json"
    cleanup = strict_json_load(cleanup_path) if cleanup_path.is_file() else {}
    errors: list[str] = []
    if report.get("P0_B_CHILD_LAUNCH_COUNT") != 1:
        errors.append("PREVIOUS_CHILD_LAUNCH_COUNT")
    if report.get("P0_B_ACTION_SUBMISSION_COUNT") != 0:
        errors.append("PREVIOUS_ACTION_SUBMISSION_COUNT")
    if report.get("P0_B_PROMOTION_STATUS") != "FAIL" or report.get("P0_C_AUTHORIZED_NEXT") != "NO":
        errors.append("PREVIOUS_PROMOTION_BOUNDARY")
    if artifact.get("P0_B_FUNCTIONAL") != "FAIL":
        errors.append("PREVIOUS_FUNCTIONAL_VERDICT")
    exception = str(artifact.get("error", artifact.get("exception", "")))
    error_type = str(artifact.get("error_type", ""))
    if "P0_B_RUNTIME_IDENTITY_MISMATCH_FROZEN_P0_A" in exception:
        authorization_id = P0_B_IDENTITY_RETRY_AUTHORIZATION_ID
        reason = "RUNTIME_IDENTITY_NORMALIZATION_BUG"
        failure = "P0_B_RUNTIME_IDENTITY_MISMATCH_FROZEN_P0_A"
    elif exception.strip("'\"") == "manifest_sha256" and error_type == "KeyError":
        authorization_id = P0_B_MANIFEST_RETRY_AUTHORIZATION_ID
        reason = "SOURCE_FREEZE_MANIFEST_SCHEMA_BINDING_BUG"
        failure = "KeyError:manifest_sha256"
    else:
        authorization_id = "UNAUTHORIZED"
        reason = "UNAUTHORIZED"
        failure = exception
        errors.append("PREVIOUS_FAILURE_NOT_APPROVED_PRE_ACTION_BOOTSTRAP_CLASS")
    ledger = _mapping(artifact.get("packet_ledger"))
    totals = _mapping(ledger.get("totals"))
    if totals and totals.get("submission_count") not in {None, 0}:
        errors.append("PREVIOUS_LEDGER_NONZERO")
    if errors:
        raise RuntimeError("P0_B_BOOTSTRAP_RETRY_NOT_AUTHORIZED:" + ",".join(errors))
    payload: dict[str, Any] = {
        "schema": P0_B_BOOTSTRAP_RETRY_AUTHORIZATION_SCHEMA,
        "authorization_id": authorization_id,
        "scope": "ONE_ADDITIONAL_P0_B_CHILD_AFTER_APPROVED_PRE_ACTION_BOOTSTRAP_FIX_ONLY",
        "reason": reason,
        "previous_attempt_preserved": True,
        "previous_supervisor_report": {"path": str(report_path), "sha256": sha256(report_path)},
        "previous_child_artifact": {"path": str(artifact_path), "sha256": sha256(artifact_path)},
        "previous_attempt": {
            "child_launch_count": 1,
            "action_submission_count": 0,
            "environment_created": cleanup.get("env_created") is True,
            "failure": failure,
            "budget": report.get("P0_B_ONE_SHOT_BUDGET"),
        },
        "retry_contract": {
            "additional_child_launch_count_limit": 1,
            "additional_action_submission_budget_before_launch": "NOT_CONSUMED",
            "adaptive_retry": "PROHIBITED",
            "only_potential_next_phase": "P0_C",
        },
    }
    payload["authorization_payload_sha256"] = _payload_sha256(payload, field="authorization_payload_sha256")
    return payload


def build_p0_b_axis_rebase_fresh_authorization(*, adjudication_receipt: Path) -> dict[str, Any]:
    """Bind the explicitly approved fresh trial to immutable rebase evidence."""

    receipt_path = adjudication_receipt.expanduser().resolve()
    receipt = strict_json_load(receipt_path)
    errors: list[str] = []
    if receipt.get("schema") != "g2_p0_b_rebase_semantics_offline_adjudication_v1":
        errors.append("ADJUDICATION_SCHEMA")
    if receipt.get("P0_B_FUNCTIONAL") != "PASS":
        errors.append("ADJUDICATED_FUNCTIONAL")
    if receipt.get("P0_B_PROMOTION_STATUS") != "PASS_WITH_KNOWN_INFRASTRUCTURE_WAIVER":
        errors.append("ADJUDICATED_PROMOTION")
    if receipt.get("artifact_integrity") is not True or receipt.get("single_consumption") is not True:
        errors.append("ADJUDICATED_EVIDENCE_INTEGRITY")
    if receipt.get("preclose_safe") is not True or receipt.get("runtime_safe") is not True:
        errors.append("ADJUDICATED_RUNTIME_SAFETY")
    phases = receipt.get("phase_adjudication")
    phase_rows = phases if isinstance(phases, list) else []
    retreat_rows = [row for row in phase_rows if isinstance(row, Mapping) and row.get("label") == "RETREAT_Z"]
    if not (
        len(phase_rows) == 4
        and all(
            isinstance(row, Mapping)
            and row.get("motion_pass") is True
            and row.get("orientation_pass") is True
            and row.get("gripper_pass") is True
            for row in phase_rows
        )
        and len(retreat_rows) == 1
        and retreat_rows[0].get("old_validator_motion_pass") is False
        and retreat_rows[0].get("same_axis_repeat") is False
    ):
        errors.append("AXIS_REBASE_ADJUDICATION")
    if errors:
        raise RuntimeError("P0_B_AXIS_REBASE_FRESH_NOT_AUTHORIZED:" + ",".join(errors))
    payload: dict[str, Any] = {
        "schema": P0_B_AXIS_REBASE_FRESH_AUTHORIZATION_SCHEMA,
        "authorization_id": P0_B_AXIS_REBASE_FRESH_AUTHORIZATION_ID,
        "scope": "EXACTLY_ONE_FRESH_P0_B_CHILD_USING_CORRECTED_AXIS_REBASE_VALIDATOR",
        "reason": "STALE_GLOBAL_AXIS_ORIGIN_VALIDATOR_FIXED_TO_MEASURED_AXIS_TRANSITION_REBASE",
        "previous_attempt_preserved": True,
        "previous_adjudication_receipt": {
            "path": str(receipt_path),
            "sha256": sha256(receipt_path),
        },
        "previous_attempt": {
            "child_launch_count": 1,
            "action_submission_count": 133,
            "environment_created": True,
            "failure": "VALIDATOR_FALSE_NEGATIVE_RETREAT_Z_AXIS_REBASE",
            "budget": "CONSUMED_AND_PRESERVED_NEW_ROOT_AUTHORIZATION_REQUIRED",
        },
        "retry_contract": {
            "additional_child_launch_count_limit": 1,
            "additional_action_submission_budget_before_launch": "NEW_EXPLICIT_AUTHORIZATION_NOT_CONSUMED",
            "adaptive_retry": "PROHIBITED",
            "only_potential_next_phase": "P0_C",
        },
    }
    payload["authorization_payload_sha256"] = _payload_sha256(
        payload, field="authorization_payload_sha256"
    )
    return payload


def build_p0_b_identity_retry_authorization(*, previous_supervisor_report: Path) -> dict[str, Any]:
    """Backward-compatible alias for callers created before bootstrap generalization."""

    return build_p0_b_bootstrap_retry_authorization(previous_supervisor_report=previous_supervisor_report)


def validate_p0_b_bootstrap_retry_authorization(authorization: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if authorization.get("schema") not in {
        P0_B_IDENTITY_RETRY_AUTHORIZATION_SCHEMA,
        P0_B_BOOTSTRAP_RETRY_AUTHORIZATION_SCHEMA,
        P0_B_AXIS_REBASE_FRESH_AUTHORIZATION_SCHEMA,
    }:
        errors.append("SCHEMA")
    authorization_id = authorization.get("authorization_id")
    reason = authorization.get("reason")
    valid_identity = bool(
        authorization_id == P0_B_IDENTITY_RETRY_AUTHORIZATION_ID
        and reason == "RUNTIME_IDENTITY_NORMALIZATION_BUG"
    )
    valid_manifest = bool(
        authorization_id == P0_B_MANIFEST_RETRY_AUTHORIZATION_ID
        and reason == "SOURCE_FREEZE_MANIFEST_SCHEMA_BINDING_BUG"
    )
    valid_axis_rebase = bool(
        authorization_id == P0_B_AXIS_REBASE_FRESH_AUTHORIZATION_ID
        and reason == "STALE_GLOBAL_AXIS_ORIGIN_VALIDATOR_FIXED_TO_MEASURED_AXIS_TRANSITION_REBASE"
    )
    if not (valid_identity or valid_manifest or valid_axis_rebase):
        errors.append("AUTHORIZATION_ID")
    if authorization.get("authorization_payload_sha256") != _payload_sha256(
        authorization, field="authorization_payload_sha256"
    ):
        errors.append("CHECKSUM")
    binding_keys = (
        ("previous_adjudication_receipt",)
        if valid_axis_rebase
        else ("previous_supervisor_report", "previous_child_artifact")
    )
    for key in binding_keys:
        binding = _mapping(authorization.get(key))
        try:
            path = Path(str(binding.get("path"))).expanduser().resolve()
        except (TypeError, ValueError):
            errors.append(key.upper())
            continue
        if not path.is_file() or binding.get("sha256") != sha256(path):
            errors.append(key.upper())
    previous = _mapping(authorization.get("previous_attempt"))
    valid_previous_preaction = bool(
        previous.get("child_launch_count") == 1
        and previous.get("action_submission_count") == 0
        and isinstance(previous.get("environment_created"), bool)
        and previous.get("failure")
        in {"P0_B_RUNTIME_IDENTITY_MISMATCH_FROZEN_P0_A", "KeyError:manifest_sha256"}
    )
    valid_previous_rebase = bool(
        valid_axis_rebase
        and previous.get("child_launch_count") == 1
        and previous.get("action_submission_count") == 133
        and previous.get("environment_created") is True
        and previous.get("failure") == "VALIDATOR_FALSE_NEGATIVE_RETREAT_Z_AXIS_REBASE"
    )
    if not (
        authorization.get("previous_attempt_preserved") is True
        and (valid_previous_preaction or valid_previous_rebase)
    ):
        errors.append("PREVIOUS_ATTEMPT")
    retry = _mapping(authorization.get("retry_contract"))
    valid_retry_scope = bool(
        retry.get("additional_child_launch_count_limit") == 1
        and retry.get("additional_action_submission_budget_before_launch")
        in {"NOT_CONSUMED", "NEW_EXPLICIT_AUTHORIZATION_NOT_CONSUMED"}
        and retry.get("adaptive_retry") == "PROHIBITED"
        and retry.get("only_potential_next_phase") == "P0_C"
    )
    if not valid_retry_scope:
        errors.append("RETRY_SCOPE")
    return {"status": "PASS" if not errors else "FAIL_CLOSED", "errors": errors}


def validate_p0_b_identity_retry_authorization(authorization: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for the generalized pre-action validator."""

    return validate_p0_b_bootstrap_retry_authorization(authorization)


def build_p0_b_launch_authorization(
    *,
    preflight: Mapping[str, Any],
    p0_a_waiver_receipt: Path,
    child_output: Path,
    child_interpreter: Path,
    identity_retry_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the parent-issued, one-child launch manifest for a future run.

    It is intentionally a small, run-scoped binding—not a global process
    manager or a reusable credential.  The current static milestone never
    calls this to launch a child.  A future parent writes it atomically and
    the child revalidates every bound path/hash before importing Isaac.
    """

    if preflight.get("status") != "PASS":
        raise RuntimeError("P0_B_LAUNCH_AUTHORIZATION_PREFLIGHT_NOT_PASS")
    source_freeze = _mapping(preflight.get("source_freeze"))
    if source_freeze.get("SOURCE_FREEZE") != "PASS":
        raise RuntimeError("P0_B_LAUNCH_AUTHORIZATION_SOURCE_FREEZE_NOT_PASS")
    receipt = p0_a_waiver_receipt.expanduser().resolve()
    output = child_output.expanduser().resolve()
    interpreter = child_interpreter.expanduser().resolve()
    if not receipt.is_file() or not interpreter.is_file():
        raise RuntimeError("P0_B_LAUNCH_AUTHORIZATION_INPUT_MISSING")
    retry = dict(identity_retry_authorization or {})
    retry_validation = validate_p0_b_bootstrap_retry_authorization(retry)
    if retry_validation.get("status") != "PASS":
        raise RuntimeError("P0_B_LAUNCH_AUTHORIZATION_IDENTITY_RETRY_NOT_PASS")
    payload: dict[str, Any] = {
        "schema": P0_B_LAUNCH_AUTHORIZATION_SCHEMA,
        "issuer": "ROOT_P0_B_ONE_SHOT_SUPERVISOR",
        "scope": "ONE_CHILD_ONLY_CURRENT_P0_B_NO_P0_C_EXECUTION_NO_P1_NO_M1_NO_M2_NO_TRAINING",
        "p0_a_waiver_receipt": {"path": str(receipt), "sha256": sha256(receipt)},
        "child_output": str(output),
        "child_interpreter": str(interpreter),
        "child_source": {"path": str(P0_B_CHILD_SOURCE.resolve()), "sha256": sha256(P0_B_CHILD_SOURCE)},
        "supervisor_source": {"path": str(P0_B_SUPERVISOR_SOURCE.resolve()), "sha256": sha256(P0_B_SUPERVISOR_SOURCE)},
        "contract_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "source_freeze": dict(source_freeze),
        "identity_normalization_retry_authorization": retry,
        "one_shot_contract": {
            "child_launch_count_limit": 1,
            "action_submission_budget_before_launch": "NOT_CONSUMED",
            "child_action_submission_count_before_launch": 0,
            "canonical_phase": "P0_B",
            "only_potential_next_phase": "P0_C",
            "direct_p1_promotion": "PROHIBITED",
        },
    }
    payload["authorization_payload_sha256"] = _payload_sha256(payload, field="authorization_payload_sha256")
    return payload


def validate_p0_b_launch_authorization(
    authorization: Mapping[str, Any],
    *,
    expected_p0_a_waiver_receipt: Path,
    expected_child_output: Path,
    expected_child_interpreter: Path,
    expected_source_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a child manifest still binds this exact frozen run."""

    errors: list[str] = []
    receipt = expected_p0_a_waiver_receipt.expanduser().resolve()
    output = expected_child_output.expanduser().resolve()
    interpreter = expected_child_interpreter.expanduser().resolve()
    if authorization.get("schema") != P0_B_LAUNCH_AUTHORIZATION_SCHEMA:
        errors.append("SCHEMA")
    if authorization.get("issuer") != "ROOT_P0_B_ONE_SHOT_SUPERVISOR":
        errors.append("ISSUER")
    if authorization.get("authorization_payload_sha256") != _payload_sha256(
        authorization, field="authorization_payload_sha256"
    ):
        errors.append("CHECKSUM")
    receipt_binding = _mapping(authorization.get("p0_a_waiver_receipt"))
    if not (
        receipt_binding.get("path") == str(receipt)
        and receipt.is_file()
        and receipt_binding.get("sha256") == sha256(receipt)
    ):
        errors.append("P0_A_RECEIPT")
    if authorization.get("child_output") != str(output):
        errors.append("CHILD_OUTPUT")
    if authorization.get("child_interpreter") != str(interpreter):
        errors.append("CHILD_INTERPRETER")
    for key, source in (
        ("CHILD_SOURCE", P0_B_CHILD_SOURCE),
        ("SUPERVISOR_SOURCE", P0_B_SUPERVISOR_SOURCE),
        ("CONTRACT_SOURCE", Path(__file__).resolve()),
    ):
        binding = _mapping(authorization.get(key.lower()))
        if not (
            binding.get("path") == str(source.resolve())
            and source.is_file()
            and binding.get("sha256") == sha256(source)
        ):
            errors.append(key)
    frozen = _mapping(authorization.get("source_freeze"))
    if canonical_json_bytes(frozen) != canonical_json_bytes(expected_source_freeze):
        errors.append("SOURCE_FREEZE_MANIFEST")
    source_hashes = _mapping(frozen.get("source_sha256"))
    for relative, expected_hash in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            errors.append("SOURCE_FREEZE_SOURCE_HASH_FORMAT")
            continue
        candidate = (ROOT / relative).resolve()
        if ROOT not in candidate.parents or not candidate.is_file() or sha256(candidate) != expected_hash:
            errors.append(f"SOURCE_FREEZE_DRIFT:{relative}")
    one_shot = _mapping(authorization.get("one_shot_contract"))
    if not (
        one_shot.get("child_launch_count_limit") == 1
        and one_shot.get("action_submission_budget_before_launch") == "NOT_CONSUMED"
        and one_shot.get("child_action_submission_count_before_launch") == 0
        and one_shot.get("canonical_phase") == "P0_B"
        and one_shot.get("only_potential_next_phase") == "P0_C"
        and one_shot.get("direct_p1_promotion") == "PROHIBITED"
    ):
        errors.append("ONE_SHOT_SCOPE")
    retry_validation = validate_p0_b_bootstrap_retry_authorization(
        _mapping(authorization.get("identity_normalization_retry_authorization"))
    )
    if retry_validation.get("status") != "PASS":
        errors.append("IDENTITY_NORMALIZATION_RETRY_AUTHORIZATION")
    return {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "errors": errors,
        "schema": authorization.get("schema"),
        "issuer": authorization.get("issuer"),
        "authorization_payload_sha256": authorization.get("authorization_payload_sha256"),
        "P0_B_CHILD_LAUNCH_COUNT_BEFORE": 0,
        "P0_B_ACTION_SUBMISSION_COUNT_BEFORE": 0,
        "P0_B_ONE_SHOT_BUDGET_BEFORE": "NOT_CONSUMED",
        "identity_normalization_retry_authorization": retry_validation,
    }
