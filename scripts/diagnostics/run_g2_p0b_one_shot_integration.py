#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Current-generation, bounded P0-B EE/gripper integration child.

This child is intentionally separate from the legacy P0-B implementation.  It
keeps only the documented semantic sequence (approach, close, retreat, open),
then drives it through the frozen full-8D ``env.step`` ingress.  It never calls
``ActionManager.process_action`` directly, never uses raw physics, and never
promotes P1/M1/M2/training.

This file is safe to import for static regression: Isaac imports happen only in
``main`` after a static source-freeze/prerequisite gate has passed.
"""

from __future__ import annotations

import argparse
import atexit
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping


def _repository_root_from_harness(path: Path | None = None) -> Path:
    harness = (path or Path(__file__)).expanduser().resolve()
    try:
        root = harness.parents[2]
    except IndexError as error:
        raise RuntimeError(f"P0_B_REPOSITORY_ROOT_DERIVATION_FAILED:{harness}") from error
    expected = root / "scripts/diagnostics/run_g2_p0b_one_shot_integration.py"
    if harness != expected.resolve():
        raise RuntimeError(
            "P0_B_REPOSITORY_ROOT_HARNESS_PATH_MISMATCH:"
            + repr({"actual": str(harness), "expected": str(expected.resolve())})
        )
    if not (root / "source").is_dir():
        raise RuntimeError(f"P0_B_REPOSITORY_ROOT_SOURCE_DIRECTORY_MISSING:{root}")
    return root


ROOT = _repository_root_from_harness()
SOURCE = ROOT / "source"
CONTRACT_SOURCE = ROOT / "scripts/diagnostics/p0b_one_shot_contract.py"
P0_A_HARNESS_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0a_one_shot_integration.py"
SCHEMA = "g2_p0_b_one_shot_integration_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_exact_module(source: Path, *, module_prefix: str) -> Any:
    """Load a prescribed source file by absolute path with provenance checks."""

    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"P0_B_MODULE_SOURCE_MISSING:{resolved}")
    before = _sha256(resolved)
    name = f"_{module_prefix}_{before[:16]}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"P0_B_MODULE_SPEC_CREATION_FAILED:{resolved}")
    origin = Path(spec.origin or "").expanduser().resolve()
    if origin != resolved:
        raise RuntimeError(f"P0_B_MODULE_SPEC_ORIGIN_MISMATCH:{origin}!={resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    if _sha256(resolved) != before:
        raise RuntimeError(f"P0_B_MODULE_SOURCE_CHANGED_DURING_LOAD:{resolved}")
    if Path(getattr(module, "__file__", "")).expanduser().resolve() != resolved:
        raise RuntimeError(f"P0_B_MODULE_PROVENANCE_MISMATCH:{resolved}")
    return module


CONTRACT = _load_exact_module(CONTRACT_SOURCE, module_prefix="g2_p0_b_contract")


def _load_current_p0_a_harness() -> tuple[Any, dict[str, Any]]:
    """Load frozen P0-A helpers by file; no namespace or factory fallback."""

    module = _load_exact_module(P0_A_HARNESS_SOURCE, module_prefix="g2_p0_b_current_p0_a")
    required = (
        "_load_existing_p0_factory",
        "_LifecycleCounter",
        "_assert_existing_p0_runtime_surface",
        "_runtime_observation_provider",
        "_read_only_initialization_baseline",
        "_gripper_command_telemetry",
        "_active_terminations",
        "_build_authoritative_packet",
        "_snapshot_payload",
        "_orientation_distance_rad",
        "_tensor",
        "_plain",
        "_historical_lifecycle_contract",
    )
    missing = [name for name in required if not callable(getattr(module, name, None)) and name != "_LifecycleCounter"]
    if missing or not isinstance(getattr(module, "_LifecycleCounter", None), type):
        raise RuntimeError("P0_B_CURRENT_P0_A_HELPER_SURFACE_MISSING:" + repr(missing))
    return module, {
        "source": str(P0_A_HARNESS_SOURCE.resolve()),
        "source_sha256": _sha256(P0_A_HARNESS_SOURCE),
        "loader": "importlib.util.spec_from_file_location",
        "namespace_import_dependency": "NONE",
        "alternate_factory_fallback": "NONE",
        "required_helper_names": list(required),
    }


def _load_authoritative_p0_b_factory(p0a: Any) -> tuple[Any, dict[str, Any]]:
    """Bind *the same* authoritative factory callable to phase ``P0_B``."""

    factory, provenance = p0a._load_existing_p0_factory()
    try:
        inspect.signature(factory).bind("P0_B")
    except TypeError as error:
        raise RuntimeError("P0_B_FACTORY_PHASE_SIGNATURE_INCOMPATIBLE") from error
    if provenance.get("factory_source") != str((ROOT / "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py").resolve()):
        raise RuntimeError("P0_B_FACTORY_SOURCE_NOT_AUTHORITATIVE")
    if provenance.get("factory_callable") != "_make_env":
        raise RuntimeError("P0_B_FACTORY_CALLABLE_NOT_AUTHORITATIVE")
    return factory, {
        **provenance,
        "requested_phase": "P0_B",
        "phase_signature_bound": True,
        "factory_implementation_copied": False,
        "factory_environment_invoked": False,
        "factory_invocation_count": 0,
    }


def _invoke_authoritative_p0_b_factory(factory: Any, binding: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Invoke the already provenance-checked factory exactly once for P0-B.

    This helper is called only after the child has authenticated its
    parent-issued launch authorization and created the simulation application.
    It does not construct a second factory or provide a fallback path.
    """

    source = Path(str(binding.get("factory_source", ""))).expanduser().resolve()
    if source != (ROOT / "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py").resolve():
        raise RuntimeError("P0_B_FACTORY_INVOCATION_SOURCE_MISMATCH")
    before = _sha256(source)
    if before != binding.get("source_sha256_before") or before != binding.get("source_sha256_after"):
        raise RuntimeError("P0_B_FACTORY_SOURCE_CHANGED_BEFORE_INVOCATION")
    env, disabled_task_terms = factory("P0_B")
    after = _sha256(source)
    if after != before:
        raise RuntimeError("P0_B_FACTORY_SOURCE_CHANGED_DURING_INVOCATION")
    receipt = {
        **dict(binding),
        "factory_environment_invoked": True,
        "factory_invocation_count": 1,
        "requested_phase": "P0_B",
        "source_sha256_before": before,
        "source_sha256_after": after,
        "disabled_task_terms": list(disabled_task_terms) if isinstance(disabled_task_terms, (list, tuple)) else disabled_task_terms,
    }
    return env, receipt


def _direct_lifecycle_marker_support() -> dict[str, Any]:
    """Statically bind the direct P0-B marker producer to the waiver tuple."""

    waiver = CONTRACT._load_waiver_module()
    required = tuple(waiver.P0_B_EXACT_LIFECYCLE_MARKERS)
    source_text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(Path(__file__).resolve()))
    marker_lines: dict[str, list[int]] = {marker: [] for marker in required}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in marker_lines
        ):
            continue
        marker_lines[node.args[0].value].append(node.lineno)
    marker_prints = {marker: len(lines) for marker, lines in marker_lines.items()}
    atexit_register = source_text.find("atexit.register(emit_atexit_markers)")
    app_launcher = source_text.find("launcher = AppLauncher(")
    checks = {
        "same_waiver_marker_tuple": required == tuple(waiver.P0_B_EXACT_LIFECYCLE_MARKERS),
        "all_direct_marker_emissions_present_once": all(count == 1 for count in marker_prints.values()),
        "atexit_registered_before_app_launcher": atexit_register >= 0 and app_launcher >= 0 and atexit_register < app_launcher,
        "artifact_marker_precedes_close_marker": source_text.find('print("P0_B_FUNCTIONAL_ARTIFACT_PERSISTED"')
        < source_text.find('print("BEFORE_APP_CLOSE"'),
    }
    return {
        "DIRECT_LIFECYCLE_MARKER_SUPPORT": "PASS_STATIC_SOURCE_BOUND" if all(checks.values()) else "FAIL_CLOSED",
        "required_markers": list(required),
        "marker_print_occurrences": marker_prints,
        "marker_emission_lines": marker_lines,
        "checks": checks,
    }


def static_pre_live_contract(
    p0_a_waiver_receipt: Path | None = None,
) -> dict[str, Any]:
    """Close current P0-B's static contract without importing Isaac."""

    receipt_path = (p0_a_waiver_receipt or CONTRACT.DEFAULT_P0_A_WAIVER_RECEIPT).expanduser().resolve()
    prerequisite: dict[str, Any]
    try:
        prerequisite = CONTRACT.load_and_adapt_p0_a_prerequisite(receipt_path)
    except BaseException as error:
        prerequisite = {
            "schema": CONTRACT.P0_A_PREREQUISITE_SCHEMA,
            "status": "FAIL_CLOSED",
            "error": f"{type(error).__name__}:{error}",
            "source_receipt_path": str(receipt_path),
            "legacy_fields_fabricated": False,
        }
    semantic = CONTRACT.semantic_intent_report()
    safety = CONTRACT.current_ee_safety_binding()
    p0a: Any | None = None
    p0a_provenance: dict[str, Any]
    factory_provenance: dict[str, Any]
    lifecycle: dict[str, Any]
    try:
        p0a, p0a_provenance = _load_current_p0_a_harness()
        _, factory_provenance = _load_authoritative_p0_b_factory(p0a)
        lifecycle = p0a._historical_lifecycle_contract()
    except BaseException as error:
        p0a_provenance = {"status": "FAIL", "error": f"{type(error).__name__}:{error}"}
        factory_provenance = {"status": "FAIL", "error": f"{type(error).__name__}:{error}"}
        lifecycle = {"verified_exact": False, "error": f"{type(error).__name__}:{error}"}
    source_freeze = CONTRACT.source_freeze_report(prerequisite)
    full_8d_authority = CONTRACT.full_8d_authority_report()
    marker_support = _direct_lifecycle_marker_support()
    full_8d = [
        {"label": command.label, "full_8d_packet": list(CONTRACT.full_8d_for_command(command))}
        for command in CONTRACT.P0_B_SEMANTIC_COMMANDS
    ]
    p0_a_parity = bool(
        p0a is not None
        and tuple(getattr(p0a, "P0_FULL_8D_ACTION", ())) == (0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        and tuple(getattr(p0a, "P0_ZERO_FULL_8D_ACTION", ())) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    )
    lifecycle_parity = bool(
        lifecycle.get("verified_exact") is True
        and lifecycle.get("reset_open_settle_cap_policy_steps") == 80
        and lifecycle.get("historical_zero_baseline_total_policy_steps") == 5
    )
    checks = {
        "source_freeze": source_freeze.get("SOURCE_FREEZE") == "PASS",
        "semantic_intent": semantic.get("legacy_semantic_match") is True,
        "p0_a_prerequisite": prerequisite.get("status") == "PASS",
        "ee_safety_scope": safety.get("P0_B_EE_SAFETY_BINDING") == "LEGACY_ONLY_NOT_SEMANTIC",
        "current_p0_a_helpers": p0a is not None,
        "factory_binding": factory_provenance.get("phase_signature_bound") is True
        and factory_provenance.get("namespace_import_dependency") == "NONE"
        and factory_provenance.get("alternate_factory_fallback") == "NONE"
        and factory_provenance.get("newly_loaded_isaac_or_omni_modules") == [],
        "p0_a_schema_parity": p0_a_parity,
        "p0_a_lifecycle_parity": lifecycle_parity,
        "full_8d_order": all(packet["full_8d_packet"][3:7] == [0.0, 0.0, 0.0, 0.0] for packet in full_8d),
        "full_8d_authority": full_8d_authority.get("FULL_8D_SCHEMA") == "PASS_STATIC_SOURCE_BOUND",
        "direct_lifecycle_markers": marker_support.get("DIRECT_LIFECYCLE_MARKER_SUPPORT") == "PASS_STATIC_SOURCE_BOUND",
        "no_direct_action_manager_path": full_8d_authority.get("checks", {}).get("no_direct_action_manager_call") is True,
    }
    return {
        "schema": "g2_p0_b_one_shot_static_pre_live_contract_v1",
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "SOURCE_FREEZE": source_freeze.get("SOURCE_FREEZE"),
        "checks": checks,
        "source_freeze": source_freeze,
        "p0_a_prerequisite": prerequisite,
        "semantic_intent": semantic,
        "p0_b_ee_safety_binding": safety,
        "current_p0_a_harness": p0a_provenance,
        "factory_binding": factory_provenance,
        "p0_a_schema_parity": "PASS" if p0_a_parity else "FAIL",
        "p0_a_lifecycle_parity": "PASS" if lifecycle_parity else "FAIL",
        "current_full_8d_commands": full_8d,
        "full_8d_authority": full_8d_authority,
        "direct_lifecycle_marker_support": marker_support,
        "canonical_ingress": "ManagerBasedRLEnv.step(full_8d_packet)",
        "router_direct_process_action_calls": 0,
        "direct_physics_bypass": "NONE",
        "scope": "STATIC_ONLY_NO_ISAAC_NO_ENV_CREATION_NO_ACTION_SUBMISSION_NO_PHYSICS",
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist finite evidence and fsync it before native teardown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class _PacketLedger:
    """Append-only receipts for one-use deferred packets and canonical ingress.

    The source-owned ``DeferredFull8DActionPacketPort`` supplies the immutable
    stage/claim/acknowledgement lifecycle.  This class only observes that
    lifecycle and compares its materialized tensor to the existing P0-A
    full-8D builder plus the canonical ``env.step`` and ActionManager captures.
    It has neither an ActionManager/process-action capability nor a physics
    API.
    """

    def __init__(self, *, env: Any, p0a: Any, counter: Any, deferred_port: Any) -> None:
        self._env = env
        self._p0a = p0a
        self._counter = counter
        self._deferred_port = deferred_port
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _same_capture(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return bool(
            left.get("shape") == right.get("shape")
            and left.get("values") == right.get("values")
            and left.get("sha256") == right.get("sha256")
        )

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def submit(
        self,
        *,
        packet: Any,
        authoritative_packet_tensor: Any,
        label: str,
        packet_role: str,
        semantic_command: str,
        expected_gripper_intent: str,
    ) -> dict[str, Any]:
        """Stage, claim, submit, and acknowledge one immutable full-8D packet."""

        sequence = len(self._records) + 1
        full = [float(value) for value in packet.values]
        record: dict[str, Any] = {
            "packet_sequence": sequence,
            "packet_role": packet_role,
            "semantic_command": semantic_command,
            "label": label,
            "packet_fingerprint": CONTRACT.packet_fingerprint(
                sequence=sequence,
                role=packet_role,
                semantic_command=semantic_command,
                full_8d_packet=full,
            ),
            "full_8d_packet": full,
            "expected_gripper_intent": expected_gripper_intent,
            "submission_started": True,
            "env_step_returned": False,
            # There is no PolicyCommandRouter object in the current P0
            # integration path.  The source-owned deferred port is the only
            # staging object and cannot call ActionManager.process_action.
            "router_process_action_count": 0,
            "router_process_action_evidence": "NO_ROUTER_INSTANCE_DEFERRED_PORT_HAS_NO_ACTION_MANAGER_CAPABILITY",
            "deferred_packet_count": 0,
            "env_step_calls": None,
            "process_action_count": None,
            "consumption_state": "CONSUMPTION_UNKNOWN",
        }
        self._records.append(record)
        stage_before = int(self._deferred_port.stage_count)
        claim_before = int(self._deferred_port.claim_count)
        acknowledgement_before = int(self._deferred_port.acknowledgement_count)
        claimed = False
        self._counter.context = label
        before = self._counter.checkpoint()
        record["before_counter"] = dict(before)
        try:
            deferred = self._deferred_port.defer_normalized_action_manager_packet(packet)
            record["deferred_packet_id"] = deferred.packet_id
            record["deferred_content_fingerprint"] = deferred.content_fingerprint
            record["deferred_packet_schema"] = deferred.schema
            # Persist the exact immutable envelope whose fingerprint was
            # issued by the existing deferred port.  The parent-side contract
            # recomputes this SHA-256; it does not trust a child boolean.
            from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import (
                PRODUCTION_FULL_8D_ACTION_COMPONENTS,
            )

            record["deferred_envelope"] = {
                "schema": deferred.schema,
                "binding_id": deferred.binding_id,
                "sequence_id": deferred.sequence_id,
                "batch_shape": list(deferred.shape),
                "device": deferred.device,
                "dtype": str(deferred.dtype),
                "semantic_field_order": list(PRODUCTION_FULL_8D_ACTION_COMPONENTS),
                "values_float_hex": [float(value).hex() for value in deferred.values],
            }
            if tuple(float(value) for value in deferred.values) != tuple(full):
                raise RuntimeError("P0_B_DEFERRED_PACKET_VALUES_MISMATCH")
            claimed_packet = self._deferred_port.claim_for_env_step(deferred.packet_id)
            claimed = True
            if (
                claimed_packet.packet_id != deferred.packet_id
                or claimed_packet.content_fingerprint != deferred.content_fingerprint
            ):
                raise RuntimeError("P0_B_DEFERRED_PACKET_CLAIM_IDENTITY_MISMATCH")
            ingress_tensor = claimed_packet.as_env_step_tensor()
            builder_capture = self._p0a._LifecycleCounter._packet_record(authoritative_packet_tensor, label)
            ingress_capture = self._p0a._LifecycleCounter._packet_record(ingress_tensor, label)
            record["canonical_packet_identity"] = {
                "authoritative_p0_a_builder": builder_capture,
                "deferred_env_step_tensor": ingress_capture,
                "builder_matches_deferred": self._same_capture(builder_capture, ingress_capture),
            }
            if record["canonical_packet_identity"]["builder_matches_deferred"] is not True:
                raise RuntimeError("P0_B_AUTHORITATIVE_BUILDER_DEFERRED_TENSOR_MISMATCH")
            observation, reward, terminated, truncated, info = self._env.step(ingress_tensor)
            del observation, reward, info
            receipt = self._deferred_port.acknowledge_env_step_success(deferred.packet_id)
            record["deferred_acknowledgement"] = {
                "packet_id": receipt.packet_id,
                "content_fingerprint": receipt.content_fingerprint,
                "binding_id": receipt.binding_id,
                "state": receipt.state.value,
            }
            if (
                receipt.packet_id != deferred.packet_id
                or receipt.content_fingerprint != deferred.content_fingerprint
                or receipt.state.value != "CONSUMED"
            ):
                raise RuntimeError("P0_B_DEFERRED_PACKET_ACKNOWLEDGEMENT_MISMATCH")
            record["env_step_returned"] = True
            record["terminated"] = bool(self._p0a._tensor(terminated)[0].item())
            record["truncated"] = bool(self._p0a._tensor(truncated)[0].item())
            record["active_terminations"] = list(self._p0a._active_terminations(self._env))
            record["gripper_telemetry"] = self._p0a._gripper_command_telemetry(
                self._env.action_manager.get_term("gripper_action")
            )
        except BaseException as error:
            record["env_step_exception"] = f"{type(error).__name__}:{error}"
            if claimed:
                try:
                    self._deferred_port.mark_env_step_consumption_unknown(record.get("deferred_packet_id", ""))
                    record["deferred_marked_consumption_unknown"] = True
                except BaseException as mark_error:
                    record["deferred_mark_unknown_exception"] = f"{type(mark_error).__name__}:{mark_error}"
            raise
        finally:
            interval = self._counter.interval(before)
            record["after_counter"] = self._counter.checkpoint()
            record["deferred_packet_count"] = int(self._deferred_port.stage_count) - stage_before
            record["deferred_claim_count"] = int(self._deferred_port.claim_count) - claim_before
            record["deferred_acknowledgement_count"] = int(self._deferred_port.acknowledgement_count) - acknowledgement_before
            record["env_step_calls"] = interval["env_step_calls"]
            record["process_action_count"] = interval["action_manager_process_action_calls"]
            env_packets = interval["env_step_packets"]
            process_packets = interval["process_action_packets"]
            identity = record.setdefault("canonical_packet_identity", {})
            identity["env_step_packets"] = env_packets
            identity["process_action_packets"] = process_packets
            deferred_tensor = identity.get("deferred_env_step_tensor")
            identity["env_step_matches_deferred"] = bool(
                isinstance(deferred_tensor, Mapping)
                and len(env_packets) == 1
                and self._same_capture(deferred_tensor, env_packets[0])
            )
            identity["process_action_matches_deferred"] = bool(
                isinstance(deferred_tensor, Mapping)
                and len(process_packets) == 1
                and self._same_capture(deferred_tensor, process_packets[0])
            )
            identity["same_immutable_packet_consumed_once"] = bool(
                identity.get("builder_matches_deferred") is True
                and identity.get("env_step_matches_deferred") is True
                and identity.get("process_action_matches_deferred") is True
            )
            record["consumption_state"] = CONTRACT.consumption_state(
                submission_started=bool(record["submission_started"]),
                env_step_returned=bool(record["env_step_returned"]),
                router_process_action_count=record["router_process_action_count"],
                deferred_packet_count=record["deferred_packet_count"],
                env_step_calls=record["env_step_calls"],
                process_action_count=record["process_action_count"],
            )
        if record["consumption_state"] != "KNOWN_SINGLE_CONSUMPTION" or not record[
            "canonical_packet_identity"
        ].get("same_immutable_packet_consumed_once"):
            raise RuntimeError(f"P0_B_PACKET_CONSUMPTION_{record['consumption_state']}:{label}")
        if record.get("terminated") or record.get("truncated") or record.get("active_terminations"):
            raise RuntimeError(f"P0_B_UNEXPECTED_TERMINATION:{label}")
        observed_intent = "CLOSE" if record["gripper_telemetry"]["close_command_active"] else "OPEN"
        record["observed_gripper_intent"] = observed_intent
        if observed_intent != expected_gripper_intent:
            raise RuntimeError(
                "P0_B_GRIPPER_INTENT_MISMATCH:"
                + repr({"label": label, "expected": expected_gripper_intent, "observed": observed_intent})
            )
        return record

    def payload(self) -> dict[str, Any]:
        totals = self._counter.summary()
        records = self.records
        active_terms = sorted(
            {term for record in records for term in record.get("active_terminations", []) if isinstance(term, str)}
        )
        termination_receipt = {
            "provider": "EXISTING_TERMINATION_MANAGER_POST_ENV_STEP",
            "packet_count": len(records),
            "all_env_step_returned": all(record.get("env_step_returned") is True for record in records),
            "terminated_packet_count": sum(bool(record.get("terminated")) for record in records),
            "truncated_packet_count": sum(bool(record.get("truncated")) for record in records),
            "active_terms": active_terms,
            "forbidden_collision_observed": "forbidden_collision" in active_terms,
            "fixed_torso_drift_observed": "fixed_torso_drift" in active_terms,
            "complete": bool(records) and all(record.get("env_step_returned") is True for record in records),
        }
        result = {
            "schema": CONTRACT.P0_B_PACKET_LEDGER_SCHEMA,
            "records": records,
            "packet_count": len(records),
            "totals": {
                "router_process_action_calls": 0,
                "env_step_calls": totals["global_env_step_calls"],
                "action_manager_process_action_calls": totals["global_action_manager_process_action_calls"],
                "logical_packet_consumption_count": len(records),
            },
            "termination_receipt": termination_receipt,
            "deferred_port_receipt": {
                "stage_count": int(self._deferred_port.stage_count),
                "claim_count": int(self._deferred_port.claim_count),
                "acknowledgement_count": int(self._deferred_port.acknowledgement_count),
                "outstanding_packet": self._deferred_port.outstanding_packet is not None,
            },
        }
        result["validation"] = CONTRACT.validate_packet_ledger(result)
        return result


def _source_defined_setup_with_ledger(
    *,
    env: Any,
    p0a: Any,
    ledger: _PacketLedger,
    zero_packet: Any,
    zero_tensor: Any,
) -> dict[str, Any]:
    """Replay the frozen P0-A reset/open/zero lifecycle through the ledger."""

    lifecycle = p0a._historical_lifecycle_contract()
    gripper = env.action_manager.get_term("gripper_action")
    initial_remaining = int(p0a._tensor(gripper._g2_open_hold_remaining)[0].item())
    reset_steps = 0
    while bool(p0a._tensor(gripper.reset_open_hold_active)[0].item()):
        ledger.submit(
            packet=zero_packet,
            authoritative_packet_tensor=zero_tensor,
            label=f"CURRENT_P0_A_RESET_OPEN_SETTLE_{reset_steps}",
            packet_role="setup",
            semantic_command="RESET_OPEN_SETTLE",
            expected_gripper_intent="OPEN",
        )
        reset_steps += 1
        if reset_steps > lifecycle["reset_open_settle_cap_policy_steps"]:
            raise RuntimeError("P0_B_RESET_OPEN_SETTLE_TIMEOUT_SOURCE_DEFINED_CAP")
    if bool(p0a._tensor(gripper.close_command_active)[0].item()):
        raise RuntimeError("P0_B_RESET_GRIPPER_NOT_OPEN")
    for index in range(lifecycle["historical_zero_baseline_total_policy_steps"]):
        ledger.submit(
            packet=zero_packet,
            authoritative_packet_tensor=zero_tensor,
            label=f"CURRENT_P0_A_HISTORICAL_ZERO_BASELINE_{index}",
            packet_role="setup",
            semantic_command="ZERO_BASELINE",
            expected_gripper_intent="OPEN",
        )
    return {
        "initial_open_hold_remaining_physics_steps": initial_remaining,
        "reset_open_settle_policy_steps": reset_steps,
        "reset_open_settle_source_cap_policy_steps": lifecycle["reset_open_settle_cap_policy_steps"],
        "historical_zero_baseline_policy_steps": lifecycle["historical_zero_baseline_total_policy_steps"],
        "source_contract": lifecycle,
    }


def _phase_response(
    *,
    p0a: Any,
    before: Any,
    after_direct: Any,
    after_holds: Any,
    command: Any,
    arm_scale: tuple[float, ...],
) -> dict[str, Any]:
    """Evaluate legacy semantic observations using frozen P0-A tolerances."""

    import torch

    expected = tuple(command.high_level_4d[index] * arm_scale[index] for index in range(3))
    before_target = torch.tensor(before.sample.target_cache.desired_ee_pose_root.position_m, dtype=torch.float32)
    direct_target = torch.tensor(after_direct.sample.target_cache.desired_ee_pose_root.position_m, dtype=torch.float32)
    before_measured = torch.tensor(before.sample.ee_pose_root.position_m, dtype=torch.float32)
    final_measured = torch.tensor(after_holds.sample.ee_pose_root.position_m, dtype=torch.float32)
    # The existing controller rebases a newly selected translation axis from
    # the measured EE pose.  Only a repeat on the same active axis inherits
    # that axis from the cached target.  This is the P0-A source-defined
    # contract generalized from +X to each P0-B motion axis; comparing a new
    # axis against the stale prior-axis cache creates a false cross-axis error.
    formula_origin = before_measured.clone()
    same_axis_repeat = False
    if command.motion_axis is not None:
        axis = int(command.motion_axis)
        same_axis_repeat = bool(
            before.sample.target_cache.pose_target_active
            and before.sample.target_cache.active_translation_axis == axis
        )
        if same_axis_repeat:
            formula_origin[axis] = before_target[axis]
    target_delta = direct_target - formula_origin
    measured_delta = final_measured - before_measured
    first_quaternion = torch.tensor(before.sample.ee_pose_root.quaternion_xyzw, dtype=torch.float32).view(1, 4)
    final_quaternion = torch.tensor(after_holds.sample.ee_pose_root.quaternion_xyzw, dtype=torch.float32).view(1, 4)
    orientation_drift = float(p0a._orientation_distance_rad(first_quaternion, final_quaternion))
    if command.motion_axis is None:
        motion_pass = True
        axis = None
        target_error = 0.0
        measured_axis_delta = 0.0
    else:
        axis = int(command.motion_axis)
        expected_target = formula_origin + torch.tensor(expected)
        target_error = float(torch.max(torch.abs(direct_target - expected_target)).item())
        measured_axis_delta = float(measured_delta[axis].item())
        motion_pass = bool(
            target_error <= p0a.P0_TARGET_FORMULA_ERROR_MAX_M
            and measured_axis_delta > p0a.P0_OBSERVED_POSITIVE_AXIS_FLOOR_M
        )
    return {
        "arm_action_scale_xyz_m_per_normalized": [float(value) for value in arm_scale[:3]],
        "expected_metric_delta_root_m": list(expected),
        "controller_target_delta_root_m_after_direct": [float(value) for value in target_delta.tolist()],
        "controller_formula_origin_root_m": [float(value) for value in formula_origin.tolist()],
        "controller_same_axis_repeat": same_axis_repeat,
        "controller_new_axis_rebased_to_measured_ee": bool(
            command.motion_axis is not None and not same_axis_repeat
        ),
        "measured_ee_delta_root_m_after_holds": [float(value) for value in measured_delta.tolist()],
        "motion_axis": axis,
        "controller_target_error_m": target_error,
        "measured_axis_delta_m": measured_axis_delta,
        "orientation_drift_rad": orientation_drift,
        "orientation_source_limit_rad": p0a.P0_ORIENTATION_DRIFT_MAX_RAD,
        "motion_pass": motion_pass,
        "orientation_pass": orientation_drift <= p0a.P0_ORIENTATION_DRIFT_MAX_RAD,
    }


def _runtime_identity(app: Any, prerequisite: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the loaded child runtime while Kit is alive and compare it.

    The frozen P0-A receipt is the comparison authority; it is not copied as
    if it were a measurement from this child.
    """

    source_runtime = prerequisite.get("source_runtime")
    source_runtime = source_runtime if isinstance(source_runtime, Mapping) else {}
    source_interpreter = source_runtime.get("child_interpreter")
    same_interpreter = False
    if isinstance(source_interpreter, str):
        try:
            same_interpreter = Path(sys.executable).resolve() == Path(source_interpreter).resolve()
        except OSError:
            same_interpreter = False
    identity: dict[str, Any] = {
        "child_interpreter": str(Path(sys.executable).resolve()),
        "same_child_interpreter_as_stock_t0": same_interpreter,
        "isaacsim_runtime_version": None,
        "kit_version": None,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "physx_version": None,
        "PHYSX_RUNTIME_IDENTITY_RAW": None,
        "PHYSX_RUNTIME_IDENTITY_RAW_TYPE": None,
        "PHYSX_RUNTIME_IDENTITY_NORMALIZED": None,
        "PHYSX_FROZEN_AUTHORITY": source_runtime.get("physx_version"),
        "physx_identity_normalization": {"status": "NOT_OBSERVED"},
        "kit_experience": str(app.config.get("experience", "")),
        "identity_errors": [],
        "provenance": "LIVE_RUNTIME_READ_ONLY_COMPARED_TO_P0_A_FROZEN_WAIVER_FINGERPRINT",
    }
    try:
        from isaacsim.core.version import get_version

        values = tuple(str(value) for value in get_version())
        identity["isaacsim_runtime_version"] = values[0] if values else None
        identity["isaacsim_runtime_version_tuple"] = list(values)
    except BaseException as error:
        identity["identity_errors"].append(f"isaacsim.core.version:{type(error).__name__}:{error}")

    kit_app = None
    extension_manager = None
    try:
        import omni.kit.app

        kit_app = omni.kit.app.get_app()
        identity["kit_version"] = kit_app.get_kit_version()
        extension_manager = kit_app.get_extension_manager()
        enabled = tuple(
            entry
            for entry in extension_manager.get_extensions()
            if isinstance(entry, dict) and entry.get("enabled") is True
        )
        physx = next(
            (
                entry
                for entry in enabled
                if str(entry.get("name")) == "omni.physx"
                or str(entry.get("id", "")).startswith("omni.physx-")
            ),
            None,
        )
        if isinstance(physx, dict):
            normalization = CONTRACT.normalize_physx_runtime_identity(
                physx.get("version"), frozen_authority=str(source_runtime.get("physx_version", ""))
            )
            identity["PHYSX_RUNTIME_IDENTITY_RAW"] = normalization["PHYSX_RUNTIME_IDENTITY_RAW"]
            identity["PHYSX_RUNTIME_IDENTITY_RAW_TYPE"] = normalization["PHYSX_RUNTIME_IDENTITY_RAW_TYPE"]
            identity["PHYSX_RUNTIME_IDENTITY_NORMALIZED"] = normalization[
                "PHYSX_RUNTIME_IDENTITY_NORMALIZED"
            ]
            identity["PHYSX_FROZEN_AUTHORITY"] = normalization["PHYSX_FROZEN_AUTHORITY"]
            identity["physx_identity_normalization"] = normalization
            identity["physx_version"] = normalization["PHYSX_RUNTIME_IDENTITY_NORMALIZED"]
            if normalization["status"] != "PASS":
                identity["identity_errors"].append(
                    "physx_identity_normalization:" + str(normalization.get("error"))
                )
        else:
            identity["identity_errors"].append("kit_runtime:omni.physx_enabled_extension_missing")
    except BaseException as error:
        identity["identity_errors"].append(f"kit_runtime:{type(error).__name__}:{error}")
    finally:
        extension_manager = None
        kit_app = None

    identity["runtime_identity_match_frozen_p0_a"] = bool(
        identity["same_child_interpreter_as_stock_t0"] is True
        and identity["isaacsim_runtime_version"] == source_runtime.get("isaacsim_runtime_version")
        and identity["kit_version"] == source_runtime.get("kit_version")
        and identity["python_major_minor"] == source_runtime.get("python_major_minor")
        and identity["physx_version"] == source_runtime.get("physx_version")
        and not identity["identity_errors"]
    )
    return identity


def _artifact_checksum(payload: Mapping[str, Any]) -> str:
    without_checksum = {key: value for key, value in payload.items() if key != "artifact_payload_sha256"}
    return hashlib.sha256(CONTRACT.canonical_json_bytes(without_checksum)).hexdigest()


def _finalize_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    payload["artifact_payload_sha256"] = _artifact_checksum(payload)
    return payload


def _same_source_freeze(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Reject source drift between static preflight and environment creation.

    The second check is intentionally performed only after the AppLauncher
    exists and before the authoritative factory is invoked.  It never creates
    an environment or submits a packet; it merely prevents a source file from
    changing in the narrow interval between the parent-visible static gate and
    current child factory binding.
    """

    return bool(
        left.get("SOURCE_FREEZE") == "PASS"
        and right.get("SOURCE_FREEZE") == "PASS"
        and left.get("production_usd_sha256") == right.get("production_usd_sha256")
        and left.get("frozen_trajectory_sha256") == right.get("frozen_trajectory_sha256")
        and left.get("source_sha256") == right.get("source_sha256")
    )


def _functional_report(
    *,
    env: Any,
    p0a: Any,
    provider: Any,
    observation_source: Any,
    counter: Any,
    factory_binding: Mapping[str, Any],
    factory_invocation_receipt: Mapping[str, Any],
    pre_live: Mapping[str, Any],
    launch_authorization: Mapping[str, Any],
    launch_authorization_validation: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    ledger_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the current full-8D P0-B sequence when future live use is allowed."""

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        GripperHysteresisLatch,
        HighLevelPolicyAction,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import (
        DeferredFull8DActionPacketPort,
    )

    p0a._assert_existing_p0_runtime_surface(env)
    observation, _ = env.reset(seed=42)
    del observation
    if counter.checkpoint() != {"env_step_count": 0, "process_action_count": 0}:
        raise RuntimeError("P0_B_ENV_RESET_UNEXPECTED_ACTION_CONSUMPTION")
    initialization_baseline = p0a._read_only_initialization_baseline(env, counter=counter)
    latch = GripperHysteresisLatch(initial_intent=AbstractGripperIntent.OPEN)
    zero_action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.50))
    zero_packet, zero_tensor, zero_derivation = p0a._build_authoritative_packet(
        high_level=zero_action, batch_size=env.num_envs, device=env.device, latch=latch
    )
    if tuple(zero_packet.values) != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0):
        raise RuntimeError("P0_B_ZERO_PACKET_SEMANTIC_MISMATCH")
    deferred_port = DeferredFull8DActionPacketPort(
        batch_size=int(env.num_envs),
        device=env.device,
        binding_id="g2_p0_b_current_full8d_deferred_env_step_v1",
    )
    ledger = _PacketLedger(env=env, p0a=p0a, counter=counter, deferred_port=deferred_port)
    if ledger_holder is not None:
        ledger_holder["packet_ledger"] = ledger
    setup = _source_defined_setup_with_ledger(
        env=env,
        p0a=p0a,
        ledger=ledger,
        zero_packet=zero_packet,
        zero_tensor=zero_tensor,
    )
    initial = provider.capture_snapshot()
    arm = env.action_manager.get_term("arm_action")
    arm_scale = tuple(float(value) for value in arm.cfg.scale)
    if len(arm_scale) != 7:
        raise RuntimeError("P0_B_ARM_SCALE_DIMENSION_MISMATCH")
    phase_evidence: list[dict[str, Any]] = []
    for command in CONTRACT.P0_B_SEMANTIC_COMMANDS:
        before = provider.capture_snapshot()
        action = HighLevelPolicyAction.from_sequence(command.high_level_4d)
        packet, packet_tensor, derivation = p0a._build_authoritative_packet(
            high_level=action, batch_size=env.num_envs, device=env.device, latch=latch
        )
        expected_full = CONTRACT.full_8d_for_command(command)
        if tuple(float(value) for value in packet.values) != expected_full:
            raise RuntimeError(
                "P0_B_FULL_8D_SEMANTIC_EXPANSION_MISMATCH:"
                + repr({"label": command.label, "actual": list(packet.values), "expected": list(expected_full)})
            )
        direct = ledger.submit(
            packet=packet,
            authoritative_packet_tensor=packet_tensor,
            label=f"P0_B_{command.label}",
            packet_role="semantic",
            semantic_command=command.label,
            expected_gripper_intent=command.expected_gripper_intent,
        )
        latch.commit(packet.gripper_intent)
        after_direct = provider.capture_snapshot()
        hold_rows: list[dict[str, Any]] = []
        for index in range(command.legacy_hold_steps):
            hold_action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.50))
            hold_packet, hold_tensor, _ = p0a._build_authoritative_packet(
                high_level=hold_action, batch_size=env.num_envs, device=env.device, latch=latch
            )
            hold_rows.append(
                ledger.submit(
                    packet=hold_packet,
                    authoritative_packet_tensor=hold_tensor,
                    label=f"P0_B_{command.label}_CURRENT_FULL8D_HOLD_{index}",
                    packet_role="hold",
                    semantic_command=command.label,
                    expected_gripper_intent=command.expected_gripper_intent,
                )
            )
            latch.commit(hold_packet.gripper_intent)
        after_holds = provider.capture_snapshot()
        response = _phase_response(
            p0a=p0a,
            before=before,
            after_direct=after_direct,
            after_holds=after_holds,
            command=command,
            arm_scale=arm_scale,
        )
        phase_evidence.append(
            {
                **command.payload(),
                "direct_packet": direct,
                "hold_packet_count": len(hold_rows),
                "hold_packets": hold_rows,
                "derivation": derivation,
                "before": p0a._snapshot_payload(before),
                "after_direct": p0a._snapshot_payload(after_direct),
                "after_holds": p0a._snapshot_payload(after_holds),
                "response": response,
                "gripper_pass": direct.get("observed_gripper_intent") == command.expected_gripper_intent
                and all(row.get("observed_gripper_intent") == command.expected_gripper_intent for row in hold_rows),
            }
        )
    final = provider.capture_snapshot()
    packet_ledger = ledger.payload()
    termination_receipt = packet_ledger["termination_receipt"]
    existing_runtime_receipt_pass = bool(
        termination_receipt.get("complete") is True
        and termination_receipt.get("terminated_packet_count") == 0
        and termination_receipt.get("truncated_packet_count") == 0
        and termination_receipt.get("forbidden_collision_observed") is False
        and termination_receipt.get("fixed_torso_drift_observed") is False
    )
    response_pass = all(
        bool(row["response"]["motion_pass"])
        and bool(row["response"]["orientation_pass"])
        and bool(row["gripper_pass"])
        for row in phase_evidence
    )
    semantic_order_pass = [row["label"] for row in phase_evidence] == [
        command.label for command in CONTRACT.P0_B_SEMANTIC_COMMANDS
    ]
    full_8d_zero_orientation_elbow = all(
        row["direct_packet"]["full_8d_packet"][3:7] == [0.0, 0.0, 0.0, 0.0]
        for row in phase_evidence
    )
    functional = "PASS" if (
        packet_ledger["validation"]["status"] == "PASS"
        and response_pass
        and semantic_order_pass
        and full_8d_zero_orientation_elbow
        and existing_runtime_receipt_pass
    ) else "FAIL"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "P0_B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CURRENT_GENERATION_BOUNDED_P0_B_EE_GRIPPER_INTEGRATION_NO_P0_C_NO_P1_NO_M1_NO_M2_NO_TRAINING",
        "source_freeze": pre_live["source_freeze"],
        "factory_identity": dict(factory_binding),
        "factory_invocation_receipt": dict(factory_invocation_receipt),
        "launch_authorization": dict(launch_authorization),
        "launch_authorization_validation": dict(launch_authorization_validation),
        "p0_b_contract_identity": {
            "schema": CONTRACT.SCHEMA,
            "semantic_intent_schema": pre_live["semantic_intent"]["schema"],
            "semantic_intent_source_sha256": pre_live["semantic_intent"]["legacy_provenance"]["source_sha256"],
        },
        "p0_a_prerequisite_receipt": pre_live["p0_a_prerequisite"],
        "runtime_identity": dict(runtime_identity),
        "initialization": {
            "read_only_before_any_packet": initialization_baseline,
            "current_p0_a_reset_open_zero_lifecycle": setup,
            "zero_packet_derivation": zero_derivation,
        },
        "semantic_command_sequence": phase_evidence,
        "packet_ledger": packet_ledger,
        "initial_observation": p0a._snapshot_payload(initial),
        "final_observation": p0a._snapshot_payload(final),
        "ee_response": [row["response"] for row in phase_evidence],
        "gripper_state": {
            "initial": initial.sample.abstract_gripper_state.value,
            "final": final.sample.abstract_gripper_state.value,
            "expected_final": "OPEN",
        },
        "collision_state": {
            "authority": "EXISTING_TERMINATION_MANAGER_POST_ENV_STEP_RECEIPT",
            "nontermination_collision_query": "NOT_EXPOSED_BY_FROZEN_P0_RUNTIME",
            "active_termination_terms": termination_receipt["active_terms"],
            "all_packet_active_terminations_empty": not termination_receipt["active_terms"],
        },
        "forbidden_collision_state": {
            "authority": "EXISTING_TERMINATION_MANAGER.forbidden_collision",
            "observed": termination_receipt["forbidden_collision_observed"],
            "receipt_complete": termination_receipt["complete"],
        },
        "termination_truncation": {
            "terminated_packet_count": termination_receipt["terminated_packet_count"],
            "truncated_packet_count": termination_receipt["truncated_packet_count"],
            "fixed_torso_drift_observed": termination_receipt["fixed_torso_drift_observed"],
            "all_packets_clear": existing_runtime_receipt_pass,
        },
        "p0_b_ee_safety_binding": pre_live["p0_b_ee_safety_binding"],
        "preclose_receipt": {
            "initialization_failure": False,
            "physics_stepping_failure": False,
            "env_step_failure": False,
            "action_submission_failure": False,
            "unexpected_collision": bool(termination_receipt["forbidden_collision_observed"]),
            "unexpected_termination": not existing_runtime_receipt_pass,
            "new_preclose_stack_signature": False,
            "instrumentation_restore_failure": False,
            "termination_receipt_complete": True,
        },
        "P0_B_FUNCTIONAL": functional,
        "P0_B_PROCESS": "UNCLASSIFIED_UNTIL_PARENT_OBSERVES_EXIT",
        "P0_B_PROMOTION_STATUS": "PENDING_PARENT_PROCESS_ADJUDICATION",
        "P0_C_AUTHORIZED_NEXT": "PENDING_PARENT_PROCESS_ADJUDICATION",
        "P0_C": "BLOCKED_NOT_EXECUTED",
        "P1": "BLOCKED",
        "M1": "NOT_AUTHORIZED",
        "M2": "NOT_AUTHORIZED",
        "TRAINING": "NOT_AUTHORIZED",
        "functional_artifact_persisted_before_shutdown": True,
        "read_only_observation_capture_calls": observation_source.captures,
    }
    return _finalize_artifact(result)


def _failure_report(
    *,
    pre_live: Mapping[str, Any],
    error: BaseException,
    ledger: Mapping[str, Any] | None = None,
    launch_authorization: Mapping[str, Any] | None = None,
    launch_authorization_validation: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "P0_B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CURRENT_GENERATION_BOUNDED_P0_B_FAILURE_PRE_CLOSE",
        "source_freeze": pre_live.get("source_freeze"),
        "p0_a_prerequisite_receipt": pre_live.get("p0_a_prerequisite"),
        "launch_authorization": dict(launch_authorization or {}),
        "launch_authorization_validation": dict(launch_authorization_validation or {}),
        "runtime_identity": dict(runtime_identity or {}),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "packet_ledger": dict(ledger or {}),
        "preclose_receipt": {
            "initialization_failure": True,
            "physics_stepping_failure": False,
            "env_step_failure": False,
            "action_submission_failure": False,
            "unexpected_collision": False,
            "unexpected_termination": False,
            "new_preclose_stack_signature": False,
            "instrumentation_restore_failure": False,
            "termination_receipt_complete": False,
        },
        "P0_B_FUNCTIONAL": "FAIL",
        "P0_B_PROCESS": "UNCLASSIFIED_UNTIL_PARENT_OBSERVES_EXIT",
        "P0_B_PROMOTION_STATUS": "NO",
        "P0_C_AUTHORIZED_NEXT": "NO",
        "P0_C": "BLOCKED",
        "P1": "BLOCKED",
        "M1": "NOT_AUTHORIZED",
        "M2": "NOT_AUTHORIZED",
        "TRAINING": "NOT_AUTHORIZED",
        "functional_artifact_persisted_before_shutdown": True,
    }
    return _finalize_artifact(payload)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# Current-generation P0-B one-shot child", "", "```text"]
    for key in (
        "P0_B_FUNCTIONAL",
        "P0_B_PROCESS",
        "P0_B_PROMOTION_STATUS",
        "P0_C_AUTHORIZED_NEXT",
        "P0_C",
        "P1",
        "M1",
        "M2",
        "TRAINING",
    ):
        lines.append(f"{key}: {payload.get(key)}")
    lines.extend(["```", "", "The parent independently records actual child exit status and waiver eligibility."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-a-waiver-receipt", type=Path, default=CONTRACT.DEFAULT_P0_A_WAIVER_RECEIPT)
    parser.add_argument(
        "--launch-authorization",
        type=Path,
        required=True,
        help="Parent-issued immutable P0-B launch manifest; direct child launch is rejected.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("--output must name a new immutable directory")
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = args.p0_a_waiver_receipt.expanduser().resolve()
    authorization_path = args.launch_authorization.expanduser().resolve()
    pre_live = static_pre_live_contract(receipt_path)
    _atomic_json(output / "P0_B_STATIC_PRE_LIVE_CONTRACT.json", pre_live)
    launch_authorization: dict[str, Any] = {}
    launch_authorization_validation: dict[str, Any] = {"status": "FAIL_CLOSED", "errors": ["NOT_LOADED"]}
    try:
        launch_authorization = CONTRACT.strict_json_load(authorization_path)
        launch_authorization_validation = CONTRACT.validate_p0_b_launch_authorization(
            launch_authorization,
            expected_p0_a_waiver_receipt=receipt_path,
            expected_child_output=output,
            expected_child_interpreter=Path(sys.executable),
            expected_source_freeze=_mapping(pre_live.get("source_freeze")),
        )
        launch_authorization = {
            **launch_authorization,
            "authorization_file_path": str(authorization_path),
            "authorization_file_sha256": _sha256(authorization_path),
        }
    except BaseException as error:
        launch_authorization_validation = {
            "status": "FAIL_CLOSED",
            "errors": [f"AUTHORIZATION_LOAD:{type(error).__name__}:{error}"],
        }
    if pre_live["status"] != "PASS" or launch_authorization_validation.get("status") != "PASS":
        failure = _failure_report(
            pre_live=pre_live,
            error=RuntimeError("P0_B_STATIC_PRE_LIVE_OR_LAUNCH_AUTHORIZATION_FAIL_CLOSED"),
            launch_authorization=launch_authorization,
            launch_authorization_validation=launch_authorization_validation,
        )
        _atomic_json(output / "P0_B_FUNCTIONAL_PRE_CLOSE.json", failure)
        _atomic_text(output / "P0_B_FUNCTIONAL_PRE_CLOSE.md", _markdown(failure))
        print("P0_B_STATIC_PRE_LIVE_CONTRACT_FAIL", flush=True)
        return 2

    # Register after static gating but *before* AppLauncher.  Python atexit is
    # LIFO; this preserves the intended order in which Kit/App callbacks that
    # AppLauncher registers later run before the child's final marker.  If
    # initialization fails, the missing app/runtime markers still fail closed.
    def emit_atexit_markers() -> None:
        print("ATEXIT_ENTER", flush=True)
        print("ATEXIT_RETURN", flush=True)

    atexit.register(emit_atexit_markers)
    print("SOURCE_FREEZE_OK", flush=True)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False, fast_shutdown=False)
    app = launcher.app
    print("APP_CREATED", flush=True)
    env = None
    counter = None
    runtime_identity: dict[str, Any] = {}
    result: dict[str, Any]
    cleanup: dict[str, Any] = {"env_created": False, "env_close": "NOT_APPLICABLE"}
    ledger_holder: dict[str, Any] = {}
    try:
        print("RUNTIME_BEGIN", flush=True)
        # Re-evaluate the same static source contract after app creation but
        # before factory/environment construction.  A changed asset/source is
        # a pre-action fail-closed condition, never a reason to use a fallback.
        post_app_pre_live = static_pre_live_contract(receipt_path)
        if post_app_pre_live.get("status") != "PASS" or not _same_source_freeze(
            pre_live["source_freeze"], post_app_pre_live["source_freeze"]
        ):
            raise RuntimeError("P0_B_SOURCE_FREEZE_CHANGED_BEFORE_FACTORY_INVOCATION")
        post_app_authorization = CONTRACT.validate_p0_b_launch_authorization(
            {key: value for key, value in launch_authorization.items() if key not in {"authorization_file_path", "authorization_file_sha256"}},
            expected_p0_a_waiver_receipt=receipt_path,
            expected_child_output=output,
            expected_child_interpreter=Path(sys.executable),
            expected_source_freeze=_mapping(post_app_pre_live.get("source_freeze")),
        )
        if post_app_authorization.get("status") != "PASS":
            raise RuntimeError("P0_B_LAUNCH_AUTHORIZATION_CHANGED_BEFORE_FACTORY_INVOCATION")
        runtime_identity = _runtime_identity(app, pre_live["p0_a_prerequisite"])
        if runtime_identity.get("runtime_identity_match_frozen_p0_a") is not True:
            raise RuntimeError("P0_B_RUNTIME_IDENTITY_MISMATCH_FROZEN_P0_A")
        p0a, _ = _load_current_p0_a_harness()
        factory, factory_binding = _load_authoritative_p0_b_factory(p0a)
        env, factory_invocation_receipt = _invoke_authoritative_p0_b_factory(factory, factory_binding)
        cleanup["env_created"] = True
        p0a._assert_existing_p0_runtime_surface(env)
        counter = p0a._LifecycleCounter(env)
        counter.install()
        provider, observation_source, _ = p0a._runtime_observation_provider(env, pre_live["source_freeze"])
        result = _functional_report(
            env=env,
            p0a=p0a,
            provider=provider,
            observation_source=observation_source,
            counter=counter,
            factory_binding=factory_binding,
            factory_invocation_receipt=factory_invocation_receipt,
            pre_live=pre_live,
            launch_authorization=launch_authorization,
            launch_authorization_validation=post_app_authorization,
            runtime_identity=runtime_identity,
            ledger_holder=ledger_holder,
        )
    except BaseException as error:
        partial: dict[str, Any] | None = None
        if counter is not None:
            partial = {"counter_summary": counter.summary()}
        packet_ledger = ledger_holder.get("packet_ledger")
        if isinstance(packet_ledger, _PacketLedger):
            partial = dict(partial or {})
            partial["packet_ledger"] = packet_ledger.payload()
        result = _failure_report(
            pre_live=pre_live,
            error=error,
            ledger=partial,
            launch_authorization=launch_authorization,
            launch_authorization_validation=launch_authorization_validation,
            runtime_identity=runtime_identity,
        )
    finally:
        # One direct marker after either the success or exception branch.  It
        # precedes artifact persistence and is not inferred by the parent.
        print("RUNTIME_END", flush=True)
        if counter is not None:
            try:
                counter.restore()
            except BaseException as error:
                # The functional artifact must still be persisted.  A failed
                # observer restore is a pre-close lifecycle defect, so it
                # invalidates functional promotion rather than being hidden by
                # native finalization later.
                preclose = result.setdefault("preclose_receipt", {})
                if isinstance(preclose, dict):
                    preclose["instrumentation_restore_failure"] = True
                    preclose["instrumentation_restore_exception"] = f"{type(error).__name__}:{error}"
                result["P0_B_FUNCTIONAL"] = "FAIL"
                result.pop("artifact_payload_sha256", None)
                _finalize_artifact(result)
        # This is the functional authority. It is flushed before *any*
        # env/application close, so a post-main native finalization crash can
        # never erase it.
        _atomic_json(output / "P0_B_FUNCTIONAL_PRE_CLOSE.json", result)
        _atomic_text(output / "P0_B_FUNCTIONAL_PRE_CLOSE.md", _markdown(result))
        print("REPORT_SAVED", flush=True)
        print("P0_B_FUNCTIONAL_ARTIFACT_PERSISTED", flush=True)
        if env is not None:
            try:
                env.close()
                cleanup["env_close"] = "PASS"
            except BaseException as error:
                cleanup["env_close"] = "FAIL_EXCEPTION"
                cleanup["env_close_exception"] = f"{type(error).__name__}:{error}"
        counter = None
        env = None
        cleanup["user_owned_runtime_references_released_before_app_close"] = True
        cleanup["gc_collect_while_application_alive"] = int(gc.collect())
        _atomic_json(output / "P0_B_CHILD_CLEANUP_PRE_APP_CLOSE.json", cleanup)
        print("BEFORE_APP_CLOSE", flush=True)
        app.close()
        print("APP_CLOSE_RETURNED", flush=True)
        print("BEFORE_MAIN_RETURN", flush=True)
        print("PYTHON_MAIN_RETURNED", flush=True)
    return 0 if result.get("P0_B_FUNCTIONAL") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
