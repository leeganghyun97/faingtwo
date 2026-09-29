#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""One bounded P0-A integration smoke using the restored source contract.

The only non-zero high-level policy command in this diagnostic is the frozen
P0 root-frame ``+X=+0.20`` action.  The source-defined reset-open settle,
historical zero baseline, and post-pulse zero holds are retained as explicit
existing-controller lifecycle traffic.  They are reported separately and are
never relabelled as the authoritative P0 pulse.

No new pre-controller admission, workspace, IK, collision, velocity,
acceleration, or gripper-mechanism rule is introduced here.  The script uses
the existing P0 environment factory and sends every lifecycle packet through
``ManagerBasedRLEnv.step``.  Its child result is pre-close functional evidence;
the parent supervisor alone classifies process exit and P0-B promotion.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping


def _repository_root_from_harness(harness_path: Path | None = None) -> Path:
    """Derive the repository root from this harness path, never from CWD.

    The P0 child inherits a ROS-heavy Python environment in which ``scripts``
    is already a regular package.  This helper intentionally does *not* rely
    on a namespace import or on the working directory to locate the existing
    P0 factory source.
    """

    harness = (harness_path or Path(__file__)).expanduser().resolve()
    try:
        root = harness.parents[2]
    except IndexError as error:
        raise RuntimeError(f"P0_REPOSITORY_ROOT_DERIVATION_FAILED:{harness}") from error
    expected_harness = root / "scripts/diagnostics/run_g2_p0a_one_shot_integration.py"
    if harness != expected_harness.resolve():
        raise RuntimeError(
            "P0_REPOSITORY_ROOT_HARNESS_PATH_MISMATCH:"
            + repr({"actual": str(harness), "expected": str(expected_harness.resolve())})
        )
    if not (root / "source").is_dir():
        raise RuntimeError(f"P0_REPOSITORY_ROOT_SOURCE_DIRECTORY_MISSING:{root}")
    return root


ROOT = _repository_root_from_harness()
SOURCE = ROOT / "source"
for _path in (str(ROOT), str(SOURCE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


SCHEMA = "g2_p0_a_one_shot_integration_v2"
PRODUCTION_USD = SOURCE / "geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
FROZEN_TRAJECTORY = ROOT / "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
EXPECTED_PRODUCTION_USD_SHA256 = "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
EXPECTED_FROZEN_TRAJECTORY_SHA256 = "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"

P0_ATTESTATION = ROOT / "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py"
P0_FACTORY_SOURCE = P0_ATTESTATION
P0_FACTORY_CALLABLE = "_make_env"
P0_PREFLIGHT = ROOT / "scripts/diagnostics/audit_g2_p0_integration_preflight.py"
P0_ONE_SHOT_SUPERVISOR = ROOT / "scripts/diagnostics/run_g2_p0a_one_shot_integration_supervisor.py"
P0_OBSERVATION = SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/p0_read_only_observation.py"
P0_ACTION_INTERFACE = SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/action_interface.py"
P0_METRIC_ADAPTER = SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/production_metric_adapter.py"
P0_REDUNDANCY_ACTION = SOURCE / "geniesim/rl/isaaclab/g2_redundancy_action.py"
P0_TELEOP_ENV_CFG = SOURCE / "geniesim/rl/isaaclab/g2_redundancy_teleop_env_cfg.py"

# These values are restored from the historical P0 X-axis evidence and are
# checked against the immutable parent preflight manifest before Isaac starts.
P0_NORMALIZED_ACTION = (0.20, 0.0, 0.0, 0.50)
P0_ZERO_ACTION = (0.0, 0.0, 0.0, 0.50)
P0_FULL_8D_ACTION = (0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
P0_ZERO_FULL_8D_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
P0_METRIC_DELTA_M = (0.0045, 0.0, 0.0)
P0_OBSERVED_POSITIVE_AXIS_FLOOR_M = 5.0e-4
P0_ROUTER_SCALE_ERROR_MAX_M = 1.0e-9
P0_TARGET_FORMULA_ERROR_MAX_M = 1.0e-6
P0_ORIENTATION_DRIFT_MAX_RAD = math.pi / 180.0
P0_RESET_OPEN_SETTLE_MAX_POLICY_STEPS = 80
P0_HISTORICAL_ZERO_BASELINE_POLICY_STEPS = 5
P0_POST_PULSE_HOLD_POLICY_STEPS = 40

P0_LOCAL_STATIC_INPUTS = (
    Path(__file__),
    P0_ATTESTATION,
    P0_PREFLIGHT,
    P0_ONE_SHOT_SUPERVISOR,
    P0_OBSERVATION,
    P0_ACTION_INTERFACE,
    P0_METRIC_ADAPTER,
    P0_REDUNDANCY_ACTION,
    P0_TELEOP_ENV_CFG,
    PRODUCTION_USD,
    FROZEN_TRAJECTORY,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_existing_p0_factory(
    *,
    factory_source: Path | None = None,
    expected_callable: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the one authoritative existing P0 factory by absolute file path.

    This intentionally avoids the repository diagnostics namespace.  The
    repository's ``scripts`` directory is an implicit namespace package, while the Isaac
    child inherits a ROS regular package named ``scripts``.  Importing by
    namespace can therefore resolve a different package before the local
    diagnostics directory is considered.  There is exactly one accepted
    source file and exactly one expected callable; no alternate factory,
    namespace fallback, or copied environment-construction implementation is
    permitted.

    The function imports the factory *definition* only.  It never calls the
    factory, creates Isaac objects, or starts a simulation application.
    """

    authoritative = P0_FACTORY_SOURCE.expanduser().resolve()
    source = (factory_source or authoritative).expanduser().resolve()
    callable_name = expected_callable or P0_FACTORY_CALLABLE
    if not source.is_file():
        raise RuntimeError(f"P0_FACTORY_SOURCE_MISSING:{source}")
    if source != authoritative:
        raise RuntimeError(
            "P0_FACTORY_SOURCE_NOT_AUTHORITATIVE:"
            + repr({"requested": str(source), "authoritative": str(authoritative)})
        )
    if not callable_name or not isinstance(callable_name, str):
        raise RuntimeError("P0_FACTORY_CALLABLE_NAME_INVALID")

    source_hash_before = _sha256(source)
    module_name = f"_g2_p0_authoritative_factory_{source_hash_before[:16]}"
    scripts_before = {
        name for name in sys.modules if name == "scripts" or name.startswith("scripts.")
    }
    isaac_before = {
        name
        for name in sys.modules
        if name == "isaaclab" or name.startswith("isaaclab.") or name == "omni" or name.startswith("omni.")
    }

    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"P0_FACTORY_SPEC_CREATION_FAILED:{source}")
    origin = Path(specification.origin or "").expanduser().resolve()
    if origin != source:
        raise RuntimeError(
            "P0_FACTORY_SPEC_ORIGIN_MISMATCH:"
            + repr({"origin": str(origin), "source": str(source)})
        )

    # A private, hash-derived module name avoids both the ROS ``scripts``
    # package and reuse of any prior namespace-loaded candidate.  Replacing a
    # previous private module is deliberate: it is not a fallback and forces
    # this call to execute the exact source file again.
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise

    source_hash_after = _sha256(source)
    if source_hash_after != source_hash_before:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise RuntimeError("P0_FACTORY_SOURCE_CHANGED_DURING_LOAD")

    factory = getattr(module, callable_name, None)
    if not callable(factory):
        raise RuntimeError(f"P0_FACTORY_CALLABLE_MISSING:{callable_name}")
    source_file = Path(inspect.getsourcefile(factory) or "").expanduser().resolve()
    code_file = Path(getattr(getattr(factory, "__code__", None), "co_filename", "")).expanduser().resolve()
    module_file = Path(getattr(module, "__file__", "")).expanduser().resolve()
    if source_file != source:
        raise RuntimeError(
            "P0_FACTORY_SOURCE_PROVENANCE_MISMATCH:"
            + repr({"callable_source": str(source_file), "source": str(source)})
        )
    if code_file != source:
        raise RuntimeError(
            "P0_FACTORY_CODE_PROVENANCE_MISMATCH:"
            + repr({"callable_code": str(code_file), "source": str(source)})
        )
    if module_file != source:
        raise RuntimeError(
            "P0_FACTORY_MODULE_PROVENANCE_MISMATCH:"
            + repr({"module_file": str(module_file), "source": str(source)})
        )
    if factory.__module__ != module_name or factory.__qualname__ != callable_name:
        raise RuntimeError(
            "P0_FACTORY_IDENTITY_MISMATCH:"
            + repr(
                {
                    "module": factory.__module__,
                    "qualname": factory.__qualname__,
                    "expected_module": module_name,
                    "expected_qualname": callable_name,
                }
            )
        )
    try:
        inspect.signature(factory).bind("P0_A")
    except TypeError as error:
        raise RuntimeError("P0_FACTORY_CALLABLE_SIGNATURE_INCOMPATIBLE") from error

    scripts_after = {
        name for name in sys.modules if name == "scripts" or name.startswith("scripts.")
    }
    isaac_after = {
        name
        for name in sys.modules
        if name == "isaaclab" or name.startswith("isaaclab.") or name == "omni" or name.startswith("omni.")
    }
    return factory, {
        "repository_root": str(ROOT),
        "factory_source": str(source),
        "factory_callable": callable_name,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "loader": "importlib.util.spec_from_file_location",
        "module_name": module_name,
        "module_file": str(module_file),
        "spec_origin": str(origin),
        "callable_source": str(source_file),
        "callable_code_filename": str(code_file),
        "callable_qualname": factory.__qualname__,
        "factory_environment_invoked": False,
        "namespace_import_dependency": "NONE",
        "alternate_factory_fallback": "NONE",
        "newly_loaded_scripts_modules": sorted(scripts_after - scripts_before),
        "newly_loaded_isaac_or_omni_modules": sorted(isaac_after - isaac_before),
    }


def _historical_lifecycle_contract() -> dict[str, Any]:
    """Extract, rather than merely name-check, the old P0 lifecycle facts.

    The restored integration gate is allowed to retain only lifecycle traffic
    that was already part of the historical P0 implementation.  Parsing the
    frozen source makes a changed ``ZERO_HOLD`` count, settle duration, or
    reset comparison fail preflight before Isaac starts.
    """

    tree = ast.parse(P0_ATTESTATION.read_text(encoding="utf-8"), filename=str(P0_ATTESTATION))

    def function(name: str) -> ast.FunctionDef:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise RuntimeError(f"P0_HISTORICAL_FUNCTION_MISSING:{name}")

    def keyword_constant(call: ast.Call, name: str) -> Any | None:
        for keyword in call.keywords:
            if keyword.arg == name and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
        return None

    legacy = function("_legacy_run_p0_a_motion_attestation_disabled")
    reset = function("_settle_open")
    settle_steps: int | None = None
    for node in ast.walk(legacy):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "settle_steps"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            settle_steps = node.value.value

    zero_execute = [
        node
        for node in ast.walk(legacy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_execute"
        and keyword_constant(node, "label") == "ZERO_HOLD"
    ]
    zero_hold_counts = [
        keyword_constant(node, "count")
        for node in ast.walk(legacy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_hold"
        and keyword_constant(node, "label") == "ZERO_HOLD"
    ]
    reset_caps = [
        node.comparators[0].value
        for node in ast.walk(reset)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
        and isinstance(node.left, ast.Name)
        and node.left.id == "iterations"
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and isinstance(node.comparators[0].value, int)
    ]
    exact = (
        len(zero_execute) == 1
        and zero_hold_counts == [4]
        and settle_steps == 40
        and reset_caps == [80]
    )
    if not exact:
        raise RuntimeError(
            "P0_HISTORICAL_LIFECYCLE_CONTRACT_DRIFT:"
            + repr(
                {
                    "zero_execute_count": len(zero_execute),
                    "zero_hold_counts": zero_hold_counts,
                    "post_pulse_hold_steps": settle_steps,
                    "reset_caps": reset_caps,
                }
            )
        )
    return {
        "historical_source_path": str(P0_ATTESTATION.relative_to(ROOT)),
        "historical_source_sha256": _sha256(P0_ATTESTATION),
        "reset_open_settle_comparison": "iterations > cap_after_existing_canonical_step",
        "reset_open_settle_cap_policy_steps": reset_caps[0],
        "historical_zero_baseline_execute_steps": len(zero_execute),
        "historical_zero_baseline_hold_steps": zero_hold_counts[0],
        "historical_zero_baseline_total_policy_steps": len(zero_execute) + zero_hold_counts[0],
        "post_pulse_zero_hold_policy_steps": settle_steps,
        "verified_exact": True,
    }


def _strict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"STRICT_JSON_NONFINITE:{value!r}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach"):
        return _plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_strict(_plain(payload)), indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _strict_json(path: Path) -> dict[str, Any]:
    def _reject(value: str) -> None:
        raise ValueError(f"STRICT_JSON_NONFINITE_CONSTANT:{value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    if not isinstance(payload, dict):
        raise ValueError("P0_PREFLIGHT_MANIFEST_NOT_OBJECT")
    return payload


def _tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    try:
        return torch.from_dlpack(value)
    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
        raise TypeError(f"P0_UNSUPPORTED_RUNTIME_TENSOR:{type(value)!r}") from error


def _finite_tuple(value: Any, *, size: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in _tensor(value).detach().to("cpu").reshape(-1).tolist())
    if len(values) != size or not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"P0_INVALID_{name.upper()}")
    return values


def _rotate_xyzw(quaternion, vector):
    import torch

    xyz = quaternion[..., :3]
    w = quaternion[..., 3:]
    cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * w * cross + 2.0 * torch.linalg.cross(xyz, cross, dim=-1)


def _quaternion_multiply_xyzw(left, right):
    import torch

    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _orientation_distance_rad(first, second) -> float:
    import torch

    dot = torch.clamp(torch.abs(torch.sum(first * second, dim=-1)), max=1.0)
    return float((2.0 * torch.acos(dot)).amax().item())


def _world_ee_pose_to_root(robot: Any, ee_frame: Any):
    import torch

    from geniesim.rl.isaaclab.g2_quaternion import (
        isaaclab_native_quaternion_order,
        quaternion_native_to_xyzw,
    )

    native_order = isaaclab_native_quaternion_order()
    root_position_w = _tensor(robot.data.root_pos_w).to(torch.float32)
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(robot.data.root_quat_w).to(torch.float32), native_order=native_order
    )
    ee_position_w = _tensor(ee_frame.data.target_pos_w)[:, 0].to(torch.float32)
    ee_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(ee_frame.data.target_quat_w)[:, 0].to(torch.float32), native_order=native_order
    )
    inverse_root = root_quaternion_xyzw.clone()
    inverse_root[:, :3] *= -1.0
    position_root = _rotate_xyzw(inverse_root, ee_position_w - root_position_w)
    quaternion_root = _quaternion_multiply_xyzw(inverse_root, ee_quaternion_xyzw)
    quaternion_root = quaternion_root / torch.linalg.vector_norm(
        quaternion_root, dim=-1, keepdim=True
    )
    quaternion_root = torch.where(quaternion_root[:, 3:4] < 0.0, -quaternion_root, quaternion_root)
    return position_root, quaternion_root


def _manifest_hash_match(manifest: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    freeze = manifest.get("source_freeze")
    if not isinstance(freeze, Mapping):
        return False, {"reason": "P0_PREFLIGHT_SOURCE_FREEZE_MISSING"}
    expected = freeze.get("input_sha256")
    if not isinstance(expected, Mapping):
        return False, {"reason": "P0_PREFLIGHT_SOURCE_HASH_MAP_MISSING"}
    current: dict[str, str | None] = {}
    mismatch: dict[str, dict[str, str | None]] = {}
    for stored_path, expected_hash in expected.items():
        path_text = str(stored_path)
        path = Path(path_text) if path_text.startswith("/") else ROOT / path_text
        current_hash = _sha256(path) if path.is_file() else None
        current[path_text] = current_hash
        if not isinstance(expected_hash, str) or current_hash != expected_hash:
            mismatch[path_text] = {"expected": str(expected_hash), "actual": current_hash}
    return not mismatch, {
        "expected_input_sha256": dict(expected),
        "current_input_sha256": current,
        "mismatch": mismatch,
    }


def _manifest_contract_match(manifest: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    verdict = manifest.get("verdict")
    contract = manifest.get("p0_contract")
    integration = manifest.get("integration_contract")
    if not isinstance(verdict, Mapping) or not isinstance(contract, Mapping) or not isinstance(integration, Mapping):
        return False, {"reason": "P0_PREFLIGHT_REQUIRED_SECTIONS_MISSING"}
    current = contract.get("authoritative_current_p0_a")
    full = integration.get("full_8d_packet")
    expected = {
        "SOURCE_FREEZE": "PASS",
        "P0_SCOPE": "INTEGRATION_GATE",
        "P0_CONTRACT_RESTORED": "PASS_STATIC",
        "P0_A_INTEGRATION_PREFLIGHT": "PASS_STATIC",
        "P0_A_LIVE_AUTHORIZED_NEXT": "YES",
    }
    checks = {
        f"verdict_{key}": verdict.get(key) == value for key, value in expected.items()
    }
    checks.update(
        {
            "normalized_action": isinstance(current, Mapping)
            and current.get("normalized_command") == list(P0_NORMALIZED_ACTION),
            "metric_action": isinstance(current, Mapping)
            and current.get("metric_command_m") == list(P0_METRIC_DELTA_M),
            "root_frame": isinstance(current, Mapping) and current.get("frame") == "robot_root",
            "full_8d": isinstance(full, Mapping) and full.get("values") == list(P0_FULL_8D_ACTION),
            "one_shot_harness": verdict.get("ONE_SHOT_HARNESS") == "PASS_STATIC",
            "canonical_ingress": str(verdict.get("CANONICAL_INGRESS", "")).startswith("PASS"),
            "single_consumption": verdict.get("SINGLE_CONSUMPTION") == "PASS_STATIC",
        }
    )
    return all(checks.values()), {"checks": checks}


def static_pre_live_contract(preflight_manifest: Path | None = None) -> dict[str, Any]:
    """Check source-defined P0 wiring before importing Isaac or creating envs."""

    lifecycle = _historical_lifecycle_contract()
    factory_bootstrap: dict[str, Any]
    try:
        _, factory_bootstrap = _load_existing_p0_factory()
        factory_bootstrap = {
            **factory_bootstrap,
            "status": "PASS",
        }
    except BaseException as error:
        factory_bootstrap = {
            "status": "FAIL",
            "error": f"{type(error).__name__}:{error}",
            "factory_environment_invoked": False,
            "namespace_import_dependency": "NONE",
            "alternate_factory_fallback": "NONE",
        }
    missing = [str(path) for path in P0_LOCAL_STATIC_INPUTS if not path.is_file()]
    local_hashes = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in P0_LOCAL_STATIC_INPUTS
        if path.is_file() and ROOT in path.parents
    }
    local_checks = {
        "all_local_inputs_present": not missing,
        # Compatibility names consumed by the static preflight; both refer to
        # the same source-only fact and neither is a runtime safety authority.
        "all_inputs_present": not missing,
        "production_usd_hash": _sha256(PRODUCTION_USD) == EXPECTED_PRODUCTION_USD_SHA256,
        "frozen_trajectory_hash": _sha256(FROZEN_TRAJECTORY) == EXPECTED_FROZEN_TRAJECTORY_SHA256,
        "authoritative_normalized_plus_x": P0_NORMALIZED_ACTION == (0.20, 0.0, 0.0, 0.50),
        "metric_scale_once": P0_METRIC_DELTA_M == (0.0045, 0.0, 0.0),
        "full_8d_packet": P0_FULL_8D_ACTION == (0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "zero_8d_packet": P0_ZERO_FULL_8D_ACTION == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "orientation_zero": P0_FULL_8D_ACTION[3:6] == (0.0, 0.0, 0.0),
        "elbow_zero": P0_FULL_8D_ACTION[6] == 0.0,
        "gripper_open_hold_sign": P0_FULL_8D_ACTION[7] == 1.0,
        "legacy_zero_baseline_source_present": lifecycle["historical_zero_baseline_total_policy_steps"]
        == P0_HISTORICAL_ZERO_BASELINE_POLICY_STEPS,
        "source_defined_reset_cap": lifecycle["reset_open_settle_cap_policy_steps"]
        == P0_RESET_OPEN_SETTLE_MAX_POLICY_STEPS,
        "source_defined_post_hold": lifecycle["post_pulse_zero_hold_policy_steps"]
        == P0_POST_PULSE_HOLD_POLICY_STEPS,
        "deterministic_factory_source_file": factory_bootstrap.get("status") == "PASS"
        and factory_bootstrap.get("factory_source") == str(P0_FACTORY_SOURCE.resolve()),
        "deterministic_factory_callable": factory_bootstrap.get("status") == "PASS"
        and factory_bootstrap.get("factory_callable") == P0_FACTORY_CALLABLE
        and factory_bootstrap.get("callable_source") == str(P0_FACTORY_SOURCE.resolve())
        and factory_bootstrap.get("callable_code_filename") == str(P0_FACTORY_SOURCE.resolve()),
        "factory_namespace_import_not_required": factory_bootstrap.get("namespace_import_dependency")
        == "NONE"
        and factory_bootstrap.get("newly_loaded_scripts_modules", []) == [],
        "factory_loader_did_not_import_isaac": factory_bootstrap.get(
            "newly_loaded_isaac_or_omni_modules", []
        )
        == [],
        "factory_alternate_fallback_none": factory_bootstrap.get("alternate_factory_fallback")
        == "NONE",
        "no_new_precontroller_safety_authority": True,
    }
    manifest_payload: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "hash_match": "NOT_REQUESTED",
        "contract_match": "NOT_REQUESTED",
        "detail": {},
    }
    # Static preflight loads this function before it has written the immutable
    # manifest it is about to freeze.  The child always supplies that manifest
    # and upgrades this deferred fact to an actual byte-for-byte comparison
    # before AppLauncher.  This avoids a circular static-build dependency.
    manifest_ok = True
    if preflight_manifest is not None:
        resolved = preflight_manifest.expanduser().resolve()
        manifest_payload["path"] = str(resolved)
        if resolved.is_file():
            manifest_payload["sha256"] = _sha256(resolved)
            manifest = _strict_json(resolved)
            hash_ok, hash_detail = _manifest_hash_match(manifest)
            contract_ok, contract_detail = _manifest_contract_match(manifest)
            manifest_payload.update(
                {
                    "hash_match": "PASS" if hash_ok else "FAIL",
                    "contract_match": "PASS" if contract_ok else "FAIL",
                    "detail": {"hash": hash_detail, "contract": contract_detail},
                }
            )
            manifest_ok = hash_ok and contract_ok
        else:
            manifest_payload["detail"] = {"reason": "P0_PREFLIGHT_MANIFEST_MISSING"}
            manifest_ok = False
    checks = {**local_checks, "immutable_preflight_manifest_match": manifest_ok}
    return {
        "schema": "g2_p0_a_one_shot_static_pre_live_contract_v3",
        "SOURCE_FREEZE": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "local_source_sha256": local_hashes,
        "production_usd_sha256": _sha256(PRODUCTION_USD) if PRODUCTION_USD.is_file() else None,
        "frozen_trajectory_sha256": _sha256(FROZEN_TRAJECTORY) if FROZEN_TRAJECTORY.is_file() else None,
        "immutable_preflight_manifest": manifest_payload,
        "factory_bootstrap": factory_bootstrap,
        "authoritative_command": {
            "high_level_normalized": list(P0_NORMALIZED_ACTION),
            "frame": "robot_root",
            "metric_delta_m": list(P0_METRIC_DELTA_M),
            "full_8d_packet": list(P0_FULL_8D_ACTION),
            "later_0_20_mm_candidate_used": False,
        },
        "canonical_runtime_factory": {
            "p0_environment_factory": "absolute_file_loader(run_g2_policy_branch_p0_attestation.py)._make_env(phase='P0_A')",
            "factory_source": str(P0_FACTORY_SOURCE.resolve()),
            "factory_callable": P0_FACTORY_CALLABLE,
            "factory_loader": "importlib.util.spec_from_file_location",
            "namespace_import_dependency": "NONE",
            "alternate_factory_fallback": "NONE",
            "packet_factory": "HighLevelPolicyAction -> GripperHysteresisLatch -> expand_to_existing_controller_8d -> NormalizedProductionActionManagerPacket",
            "canonical_ingress": "ManagerBasedRLEnv.step(full_8d_packet)",
            "action_manager_owner": "IsaacLab ManagerBasedRLEnv.step",
            "router_direct_process_action_calls": 0,
            "legacy_direct_action_manager_port": "NOT_USED_TEST_ONLY",
        },
        "read_only_observation": {
            "provider": "P0IntegrationObservationProvider",
            "runtime_source": "RuntimeP0AReadOnlyObservationSource",
            "authority": "NO_ACCEPTANCE_AUTHORITY",
            "runtime_binding": "BOUND_ONLY_AFTER_EXISTING_P0_ENV_INITIALIZATION",
        },
        "lifecycle_provenance": {
            "historical_source_contract": lifecycle,
            "reset_open_settle": "EXISTING_CANONICAL_ZERO_OPEN_ENV_STEP_THEN_SOURCE_DEFINED_ITERATIONS_GREATER_THAN_80_FAIL",
            "reset_open_settle_max_policy_steps": lifecycle["reset_open_settle_cap_policy_steps"],
            "historical_zero_baseline": "EXECUTE_EXACTLY_SOURCE_DEFINED_ONE_ZERO_HOLD_PLUS_FOUR_ZERO_HOLDS",
            "historical_zero_baseline_policy_steps": lifecycle[
                "historical_zero_baseline_total_policy_steps"
            ],
            "post_pulse_hold": "EXECUTE_EXACTLY_SOURCE_DEFINED_FORTY_ZERO_HOLDS",
            "post_pulse_hold_policy_steps": lifecycle["post_pulse_zero_hold_policy_steps"],
            "authoritative_nonzero_policy_packet_count": 1,
            "no_raw_apply_action_or_raw_sim_step": True,
        },
        "scope": "STATIC_ONLY_NO_ISAAC_NO_ENV_STEP_NO_ACTION_SUBMISSION_NO_PHYSICS",
    }


class _LifecycleCounter:
    """Passive instrumentation of existing env/manager calls; no packet mutation."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.manager = env.action_manager
        self._original_process: Any | None = None
        self._original_env_step: Any | None = None
        self.context = "UNLABELLED"
        self.process_records: list[dict[str, Any]] = []
        self.env_step_records: list[dict[str, Any]] = []

    @staticmethod
    def _packet_record(action: Any, context: str) -> dict[str, Any]:
        tensor = _tensor(action).detach().to("cpu").contiguous()
        values = _plain(tensor)
        encoded = tensor.numpy().tobytes()
        return {
            "context": context,
            "shape": [int(value) for value in tensor.shape],
            "values": values,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def install(self) -> None:
        if self._original_process is not None or self._original_env_step is not None:
            raise RuntimeError("P0_LIFECYCLE_COUNTER_ALREADY_INSTALLED")
        process = getattr(self.manager, "process_action", None)
        env_step = getattr(self.env, "step", None)
        if not callable(process) or not callable(env_step):
            raise RuntimeError("P0_CANONICAL_RUNTIME_SURFACE_MISSING")

        def counted_process(action: Any) -> Any:
            self.process_records.append(self._packet_record(action, self.context))
            return process(action)

        def counted_env_step(action: Any) -> Any:
            self.env_step_records.append(self._packet_record(action, self.context))
            return env_step(action)

        self._original_process = process
        self._original_env_step = env_step
        self.manager.process_action = counted_process
        self.env.step = counted_env_step

    def restore(self) -> None:
        if self._original_process is not None:
            self.manager.process_action = self._original_process
            self._original_process = None
        if self._original_env_step is not None:
            self.env.step = self._original_env_step
            self._original_env_step = None

    def checkpoint(self) -> dict[str, int]:
        return {
            "env_step_count": len(self.env_step_records),
            "process_action_count": len(self.process_records),
        }

    def interval(self, before: Mapping[str, int]) -> dict[str, Any]:
        env_start = int(before["env_step_count"])
        process_start = int(before["process_action_count"])
        return {
            "env_step_calls": len(self.env_step_records) - env_start,
            "action_manager_process_action_calls": len(self.process_records) - process_start,
            "env_step_packets": self.env_step_records[env_start:],
            "process_action_packets": self.process_records[process_start:],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "global_env_step_calls": len(self.env_step_records),
            "global_action_manager_process_action_calls": len(self.process_records),
            "env_step_packets": self.env_step_records,
            "process_action_packets": self.process_records,
        }


@dataclass
class RuntimeP0AReadOnlyObservationSource:
    """Copies existing state/cache surfaces without controller mutation."""

    env: Any
    binding: Any
    captures: int = 0

    def capture_integration_observation(self):
        from geniesim.rl.isaaclab.g2_policy_branch.p0_read_only_observation import (
            AbstractGripperCommandState,
            P0IntegrationObservationSample,
            P0IntegrationPose,
            P0IntegrationTargetCache,
            P0_ROOT_FRAME,
            P0_WORLD_FRAME,
        )

        import torch

        robot = self.env.scene["robot"]
        ee_frame = self.env.scene["ee_frame"]
        arm_term = self.env.action_manager.get_term("arm_action")
        gripper_term = self.env.action_manager.get_term("gripper_action")
        joint_ids = tuple(int(item) for item in arm_term._joint_ids)
        if len(joint_ids) != 7:
            raise RuntimeError("P0_RIGHT_ARM_JOINT_COUNT_MISMATCH")
        root_position, root_quaternion = _world_ee_pose_to_root(robot, ee_frame)
        from geniesim.rl.isaaclab.g2_quaternion import (
            isaaclab_native_quaternion_order,
            quaternion_native_to_xyzw,
        )

        native = isaaclab_native_quaternion_order()
        world_root_quaternion = quaternion_native_to_xyzw(
            _tensor(robot.data.root_quat_w).to(torch.float32), native_order=native
        )
        desired_position = _tensor(arm_term.ee_desired_position)[0]
        desired_wxyz = _tensor(arm_term.ee_desired_orientation)[0]
        desired_xyzw = torch.cat((desired_wxyz[1:], desired_wxyz[:1]))
        self.captures += 1
        return P0IntegrationObservationSample(
            control_epoch=int(self.env.common_step_counter),
            capture_time_monotonic_s=time.monotonic(),
            joint_position_rad=_finite_tuple(
                _tensor(robot.data.joint_pos)[0, list(joint_ids)], size=7, name="joint_position"
            ),
            joint_velocity_rad_s=_finite_tuple(
                _tensor(robot.data.joint_vel)[0, list(joint_ids)], size=7, name="joint_velocity"
            ),
            root_pose_world=P0IntegrationPose(
                position_m=_finite_tuple(_tensor(robot.data.root_pos_w)[0], size=3, name="root_position"),
                quaternion_xyzw=_finite_tuple(world_root_quaternion[0], size=4, name="root_quaternion"),
                frame=P0_WORLD_FRAME,
            ),
            ee_pose_root=P0IntegrationPose(
                position_m=_finite_tuple(root_position[0], size=3, name="ee_root_position"),
                quaternion_xyzw=_finite_tuple(root_quaternion[0], size=4, name="ee_root_quaternion"),
                frame=P0_ROOT_FRAME,
            ),
            target_cache=P0IntegrationTargetCache(
                desired_ee_pose_root=P0IntegrationPose(
                    position_m=_finite_tuple(desired_position, size=3, name="target_position"),
                    quaternion_xyzw=_finite_tuple(desired_xyzw, size=4, name="target_quaternion"),
                    frame=P0_ROOT_FRAME,
                ),
                pose_target_active=bool(_tensor(arm_term.ee_pose_target_active)[0].item()),
                active_translation_axis=int(_tensor(arm_term.active_translation_axis)[0].item()),
                maximum_outstanding_translation_axis_error_m=float(
                    arm_term.cfg.maximum_outstanding_translation_axis_error_m
                ),
                elbow_normalized=float(_tensor(arm_term.raw_actions)[0, 6].item()),
            ),
            abstract_gripper_state=(
                AbstractGripperCommandState.CLOSED
                if bool(_tensor(gripper_term.close_command_active)[0].item())
                else AbstractGripperCommandState.OPEN
            ),
        )


def _runtime_observation_provider(env: Any, source_freeze: Mapping[str, Any]):
    from geniesim.rl.isaaclab.g2_policy_branch.p0_read_only_observation import (
        P0IntegrationObservationBinding,
        P0IntegrationObservationProvider,
    )

    robot = env.scene["robot"]
    arm_term = env.action_manager.get_term("arm_action")
    joint_ids = tuple(int(item) for item in arm_term._joint_ids)
    payload = {
        "total_action_dim": int(env.action_manager.total_action_dim),
        "active_terms": list(env.action_manager.active_terms),
        "arm_scale": [float(value) for value in arm_term.cfg.scale],
        "arm_cfg_type": type(arm_term.cfg).__name__,
    }
    binding = P0IntegrationObservationBinding(
        binding_id="g2-p0-a-one-shot-existing-control-env-v2",
        ordered_right_arm_joint_names=tuple(str(robot.joint_names[index]) for index in joint_ids),
        root_body_name="base_link",
        ee_body_name="gripper_r_center_link",
        source_fingerprint=str(source_freeze["manifest_sha256"]),
        controller_config_fingerprint=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        production_usd_fingerprint=str(source_freeze["production_usd_sha256"]),
    )
    source = RuntimeP0AReadOnlyObservationSource(env=env, binding=binding)
    return P0IntegrationObservationProvider(binding=binding, source=source), source, payload


def _active_terminations(env: Any) -> list[str]:
    result: list[str] = []
    for name in env.termination_manager.active_terms:
        if bool(_tensor(env.termination_manager.get_term(name))[0].item()):
            result.append(str(name))
    return result


def _make_existing_p0_env():
    """Reuse the authoritative existing P0-A configuration factory verbatim."""

    factory, _ = _load_existing_p0_factory()
    return factory("P0_A")


def _assert_existing_p0_runtime_surface(env: Any) -> None:
    from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import (
        assert_selected_production_action_manager_surface,
    )

    assert_selected_production_action_manager_surface(env.action_manager)
    if set(env.termination_manager.active_terms) != {"forbidden_collision", "fixed_torso_drift"}:
        raise RuntimeError(
            "P0_EXISTING_TERMINATION_SURFACE_MISMATCH:" + repr(tuple(env.termination_manager.active_terms))
        )


def _build_authoritative_packet(*, high_level: Any, batch_size: int, device: Any, latch: Any | None = None):
    """Derive the exact full-8D packet from high-level semantics and latch."""

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        GripperHysteresisLatch,
        expand_to_existing_controller_8d,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import (
        NormalizedProductionActionManagerPacket,
        NormalizedProductionArmAction,
    )

    if latch is None:
        latch = GripperHysteresisLatch(initial_intent=AbstractGripperIntent.OPEN)
    proposed = latch.proposed_intent(high_level.gripper_probability)
    expanded = tuple(expand_to_existing_controller_8d(high_level, gripper_intent=proposed))
    packet = NormalizedProductionActionManagerPacket(
        NormalizedProductionArmAction(tuple(expanded[:7])), proposed
    )
    if packet.values != expanded:
        raise RuntimeError("P0_PACKET_EXPANSION_ADAPTER_MISMATCH")
    tensor = packet.as_tensor(batch_size=batch_size, device=device)
    if tuple(tensor.shape) != (batch_size, 8):
        raise RuntimeError("P0_AUTHORITATIVE_PACKET_SHAPE_MISMATCH")
    return packet, tensor, {
        "high_level": list(high_level.values),
        "latch_initial": AbstractGripperIntent.OPEN.value,
        "latch_proposed": proposed.value,
        "full_8d_expanded": list(expanded),
        "packet_values": list(packet.values),
    }


def _submit_existing_packet(
    env: Any,
    *,
    counter: _LifecycleCounter,
    packet_tensor: Any,
    label: str,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Use the canonical env.step path and retain event evidence pre-exit."""

    counter.context = label
    before = counter.checkpoint()
    ledger["last_submission"] = {
        "label": label,
        "before": dict(before),
        "packet": _LifecycleCounter._packet_record(packet_tensor, label),
    }
    observation, reward, terminated, truncated, info = env.step(packet_tensor)
    del observation, reward, info
    active_after = _active_terminations(env)
    terminated_flag = bool(_tensor(terminated)[0].item())
    truncated_flag = bool(_tensor(truncated)[0].item())
    event = {
        "label": label,
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "active_terminations_after_env_step": active_after,
        "termination_cause_observability": (
            "DIRECT_EXISTING_TERM_READ" if active_after else "UNKNOWN_IF_AUTO_RESET_OCCURRED"
        ),
        "counter_interval": counter.interval(before),
    }
    ledger.setdefault("submission_events", []).append(event)
    ledger["last_submission"] = {**ledger["last_submission"], "after": counter.checkpoint(), "event": event}
    if terminated_flag or truncated_flag or active_after:
        raise RuntimeError("P0_EXISTING_RUNTIME_TERMINATED:" + label)
    gripper = env.action_manager.get_term("gripper_action")
    if bool(_tensor(gripper.close_command_active)[0].item()):
        raise RuntimeError("P0_UNEXPECTED_GRIPPER_CLOSE:" + label)
    return event


def _target_formula(before_snapshot: Any):
    import torch

    measured = torch.tensor(before_snapshot.sample.ee_pose_root.position_m, dtype=torch.float32)
    target = torch.tensor(before_snapshot.sample.target_cache.desired_ee_pose_root.position_m, dtype=torch.float32)
    same_axis = bool(
        before_snapshot.sample.target_cache.pose_target_active
        and before_snapshot.sample.target_cache.active_translation_axis == 0
    )
    origin = measured.clone()
    if same_axis:
        origin[0] = target[0]
    return origin, origin + torch.tensor(P0_METRIC_DELTA_M, dtype=torch.float32), same_axis


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    return _plain(snapshot.payload())


def _read_only_initialization_baseline(env: Any, *, counter: _LifecycleCounter) -> dict[str, Any]:
    """Capture the required pre-command state without initializing a target.

    ``G2RedundancyDifferentialIKAction`` initializes its desired Cartesian
    endpoint during the first canonical action packet.  The typed P0 snapshot
    deliberately rejects an all-zero quaternion, so it cannot truthfully
    represent that not-yet-initialized target cache.  This raw, read-only
    baseline therefore records the live articulation/frame state and reports
    target-cache availability explicitly rather than submitting a synthetic
    zero packet just to make the cache valid.
    """

    import torch

    if counter.checkpoint() != {"env_step_count": 0, "process_action_count": 0}:
        raise RuntimeError("P0_INITIALIZATION_BASELINE_ACTION_ALREADY_CONSUMED")
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    arm_term = env.action_manager.get_term("arm_action")
    gripper_term = env.action_manager.get_term("gripper_action")
    joint_ids = tuple(int(item) for item in arm_term._joint_ids)
    if len(joint_ids) != 7:
        raise RuntimeError("P0_RIGHT_ARM_JOINT_COUNT_MISMATCH")
    ee_position_root, ee_quaternion_root = _world_ee_pose_to_root(robot, ee_frame)
    target_cache: dict[str, Any]
    try:
        desired_position = _finite_tuple(
            _tensor(arm_term.ee_desired_position)[0], size=3, name="initial_target_position"
        )
        desired_wxyz = _finite_tuple(
            _tensor(arm_term.ee_desired_orientation)[0], size=4, name="initial_target_orientation"
        )
        desired_xyzw = (*desired_wxyz[1:], desired_wxyz[0])
        target_cache = {
            "status": "AVAILABLE" if math.sqrt(sum(value * value for value in desired_xyzw)) > 0.0 else "UNINITIALIZED",
            "desired_ee_position_root_m": list(desired_position),
            "desired_ee_quaternion_xyzw": list(desired_xyzw),
            "pose_target_active": bool(_tensor(arm_term.ee_pose_target_active)[0].item()),
            "active_translation_axis": int(_tensor(arm_term.active_translation_axis)[0].item()),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        target_cache = {
            "status": "UNAVAILABLE",
            "reason": f"{type(error).__name__}:{error}",
        }
    return {
        "capture_kind": "READ_ONLY_INITIALIZATION_BEFORE_ANY_POLICY_PACKET",
        "ACTION_SUBMISSION_COUNT": 0,
        "ROUTER_PROCESS_ACTION_COUNT": 0,
        "ACTION_MANAGER_PROCESS_ACTION_COUNT": 0,
        "control_epoch": int(env.common_step_counter),
        "simulation_timestamp": {
            "physics_dt_s": float(env.physics_dt),
            "environment_step_dt_s": float(env.step_dt),
        },
        "root_pose_world": {
            "position_m": list(_finite_tuple(_tensor(robot.data.root_pos_w)[0], size=3, name="initial_root_position")),
            "quaternion_native": list(_finite_tuple(_tensor(robot.data.root_quat_w)[0], size=4, name="initial_root_quaternion")),
        },
        "ee_pose_root": {
            "position_m": list(_finite_tuple(ee_position_root[0], size=3, name="initial_ee_root_position")),
            "quaternion_xyzw": list(_finite_tuple(ee_quaternion_root[0], size=4, name="initial_ee_root_quaternion")),
        },
        "right_arm_joint_position_rad": list(
            _finite_tuple(_tensor(robot.data.joint_pos)[0, list(joint_ids)], size=7, name="initial_joint_position")
        ),
        "right_arm_joint_velocity_rad_s": list(
            _finite_tuple(_tensor(robot.data.joint_vel)[0, list(joint_ids)], size=7, name="initial_joint_velocity")
        ),
        "gripper_semantic_state": "CLOSED"
        if bool(_tensor(gripper_term.close_command_active)[0].item())
        else "OPEN",
        "target_cache": target_cache,
    }


def _gripper_command_telemetry(gripper_term: Any) -> dict[str, Any]:
    """Copy existing abstract-command telemetry; never infer mechanics."""

    result: dict[str, Any] = {
        "close_command_active": bool(_tensor(gripper_term.close_command_active)[0].item()),
        "exact_open_target_emitted": bool(_tensor(gripper_term.exact_open_target_emitted)[0].item()),
        "exact_close_target_emitted": bool(_tensor(gripper_term.exact_close_target_emitted)[0].item()),
        "reset_open_hold_active": bool(_tensor(gripper_term.reset_open_hold_active)[0].item()),
    }
    for name in ("_raw_actions", "_processed_actions", "_last_rate_limited_target"):
        value = getattr(gripper_term, name, None)
        if value is not None:
            result[name.lstrip("_")] = _plain(_tensor(value)[0])
    return result


def _source_defined_setup(
    env: Any,
    *,
    counter: _LifecycleCounter,
    provider: Any,
    zero_tensor: Any,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Replay only the explicit historical reset/open and zero lifecycle."""

    lifecycle = _historical_lifecycle_contract()
    gripper = env.action_manager.get_term("gripper_action")
    initial_remaining = int(_tensor(gripper._g2_open_hold_remaining)[0].item())
    reset_steps = 0
    while bool(_tensor(gripper.reset_open_hold_active)[0].item()):
        _submit_existing_packet(
            env,
            counter=counter,
            packet_tensor=zero_tensor,
            label="EXISTING_RESET_OPEN_SETTLE_ZERO_OPEN",
            ledger=ledger,
        )
        reset_steps += 1
        # Match the source's post-step ``if iterations > 80`` comparison
        # exactly.  Do not add a pre-step cap or inspect private physics-time
        # counters as an extra authority.
        if reset_steps > lifecycle["reset_open_settle_cap_policy_steps"]:
            raise RuntimeError("P0_RESET_OPEN_SETTLE_TIMEOUT_SOURCE_DEFINED_CAP")
    if bool(_tensor(gripper.close_command_active)[0].item()):
        raise RuntimeError("P0_RESET_GRIPPER_NOT_OPEN")

    baseline_before = provider.capture_snapshot()
    baseline_steps = 0
    for index in range(lifecycle["historical_zero_baseline_total_policy_steps"]):
        _submit_existing_packet(
            env,
            counter=counter,
            packet_tensor=zero_tensor,
            label=f"HISTORICAL_ZERO_BASELINE_{index}",
            ledger=ledger,
        )
        baseline_steps += 1
    baseline_after = provider.capture_snapshot()
    return {
        "after_reset_open_settle_before_historical_zero_baseline": _snapshot_payload(baseline_before),
        "reset_open_hold_initial_physics_steps": initial_remaining,
        "reset_open_settle_policy_steps": reset_steps,
        "reset_open_settle_source_cap_policy_steps": lifecycle["reset_open_settle_cap_policy_steps"],
        "historical_zero_baseline_policy_steps": baseline_steps,
        "historical_zero_baseline_source_expected_steps": lifecycle[
            "historical_zero_baseline_total_policy_steps"
        ],
        "historical_lifecycle_source_contract": lifecycle,
        "gripper_open_after_setup": not bool(_tensor(gripper.close_command_active)[0].item()),
        "post_setup_snapshot": _snapshot_payload(baseline_after),
    }


def _functional_report(
    *,
    env: Any,
    provider: Any,
    observation_source: RuntimeP0AReadOnlyObservationSource,
    counter: _LifecycleCounter,
    disabled_task_terms: Mapping[str, Any],
    pre_live: Mapping[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Run source-defined lifecycle, then exactly one non-zero P0 pulse."""

    import torch

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        GripperHysteresisLatch,
        HighLevelPolicyAction,
    )

    _assert_existing_p0_runtime_surface(env)
    observation, _ = env.reset(seed=42)
    del observation
    if counter.checkpoint() != {"env_step_count": 0, "process_action_count": 0}:
        raise RuntimeError("P0_ENV_RESET_UNEXPECTED_ACTION_CONSUMPTION")
    initialization_baseline = _read_only_initialization_baseline(env, counter=counter)

    p0_gripper_latch = GripperHysteresisLatch(initial_intent=AbstractGripperIntent.OPEN)
    zero_action = HighLevelPolicyAction.from_sequence(P0_ZERO_ACTION)
    zero_packet, zero_tensor, zero_derivation = _build_authoritative_packet(
        high_level=zero_action, batch_size=env.num_envs, device=env.device, latch=p0_gripper_latch
    )
    if zero_packet.values != P0_ZERO_FULL_8D_ACTION:
        raise RuntimeError("P0_ZERO_BASELINE_PACKET_SEMANTIC_MISMATCH")
    setup = _source_defined_setup(
        env,
        counter=counter,
        provider=provider,
        zero_tensor=zero_tensor,
        ledger=ledger,
    )

    initial = provider.capture_snapshot()
    pulse_before = counter.checkpoint()
    formula_origin, expected_target, same_axis_repeat = _target_formula(initial)
    gripper_before_pulse = _gripper_command_telemetry(env.action_manager.get_term("gripper_action"))
    high_level = HighLevelPolicyAction.from_sequence(P0_NORMALIZED_ACTION)
    packet, packet_tensor, derivation = _build_authoritative_packet(
        high_level=high_level, batch_size=env.num_envs, device=env.device, latch=p0_gripper_latch
    )
    if packet.values != P0_FULL_8D_ACTION:
        raise RuntimeError("P0_AUTHORITATIVE_FULL_PACKET_SEMANTIC_MISMATCH")

    # This is the sole authoritative non-zero P0 action.  Do not call a router
    # or ActionManager directly: env.step owns the single process_action call.
    counter.context = "P0_AUTHORITATIVE_PLUS_X_SINGLE_SUBMISSION"
    submission_before = counter.checkpoint()
    ledger["authoritative_pulse"] = {
        "packet": _LifecycleCounter._packet_record(packet_tensor, counter.context),
        "before": dict(submission_before),
        "submitted": True,
    }
    observation, reward, terminated, truncated, info = env.step(packet_tensor)
    del observation, reward, info
    active_after_pulse = _active_terminations(env)
    pulse_terminated = bool(_tensor(terminated)[0].item())
    pulse_truncated = bool(_tensor(truncated)[0].item())
    pulse_interval = counter.interval(submission_before)
    ledger["authoritative_pulse"].update(
        {
            "after": counter.checkpoint(),
            "terminated": pulse_terminated,
            "truncated": pulse_truncated,
            "active_terminations_after_env_step": active_after_pulse,
            "counter_interval": pulse_interval,
        }
    )
    after_ingress = provider.capture_snapshot()
    gripper_after_ingress = _gripper_command_telemetry(env.action_manager.get_term("gripper_action"))
    if pulse_terminated or pulse_truncated or active_after_pulse:
        raise RuntimeError("P0_EXISTING_RUNTIME_TERMINATED:P0_AUTHORITATIVE_PLUS_X_SINGLE_SUBMISSION")
    if gripper_after_ingress["close_command_active"]:
        raise RuntimeError("P0_UNEXPECTED_GRIPPER_CLOSE:P0_AUTHORITATIVE_PLUS_X_SINGLE_SUBMISSION")
    p0_gripper_latch.commit(packet.gripper_intent)

    # Historical P0 evaluates the X semantic slice after 40 canonical zero
    # heartbeat holds.  They keep the accepted endpoint alive; each is logged
    # separately, and none is another non-zero P0 policy command.
    post_hold_steps = 0
    historical_lifecycle = setup["historical_lifecycle_source_contract"]
    for index in range(historical_lifecycle["post_pulse_zero_hold_policy_steps"]):
        _submit_existing_packet(
            env,
            counter=counter,
            packet_tensor=zero_tensor,
            label=f"HISTORICAL_PLUS_X_ZERO_HOLD_{index}",
            ledger=ledger,
        )
        post_hold_steps += 1
    final = provider.capture_snapshot()
    gripper_final = _gripper_command_telemetry(env.action_manager.get_term("gripper_action"))

    arm = env.action_manager.get_term("arm_action")
    scale = tuple(float(value) for value in arm.cfg.scale)
    if len(scale) != 7:
        raise RuntimeError("P0_ARM_SCALE_DIMENSION_MISMATCH")
    resolved_metric_delta = high_level.translation_normalized[0] * scale[0]
    scale_error = abs(resolved_metric_delta - P0_METRIC_DELTA_M[0])
    target_after = torch.tensor(after_ingress.sample.target_cache.desired_ee_pose_root.position_m, dtype=torch.float32)
    target_final = torch.tensor(final.sample.target_cache.desired_ee_pose_root.position_m, dtype=torch.float32)
    target_formula_error = float(torch.max(torch.abs(target_after - expected_target)).item())
    target_hold_delta = tuple(float(value) for value in (target_final - target_after).tolist())
    target_delta = tuple(
        float(after_ingress.sample.target_cache.desired_ee_pose_root.position_m[index]
              - initial.sample.target_cache.desired_ee_pose_root.position_m[index])
        for index in range(3)
    )
    measured_delta = tuple(
        float(final.sample.ee_pose_root.position_m[index] - initial.sample.ee_pose_root.position_m[index])
        for index in range(3)
    )
    initial_quat = torch.tensor(initial.sample.ee_pose_root.quaternion_xyzw, dtype=torch.float32).view(1, 4)
    final_quat = torch.tensor(final.sample.ee_pose_root.quaternion_xyzw, dtype=torch.float32).view(1, 4)
    orientation_drift = _orientation_distance_rad(initial_quat, final_quat)
    gripper_states = {
        "initial": initial.sample.abstract_gripper_state.value,
        "after_ingress": after_ingress.sample.abstract_gripper_state.value,
        "final": final.sample.abstract_gripper_state.value,
        "expected": "OPEN",
        "before_authoritative_pulse": gripper_before_pulse,
        "after_authoritative_ingress": gripper_after_ingress,
        "after_source_defined_holds": gripper_final,
    }
    gripper_states["unchanged_open"] = all(value == "OPEN" for key, value in gripper_states.items() if key in {"initial", "after_ingress", "final"})
    gripper_states["controller_transition"] = (
        "NONE_OPEN_TO_OPEN"
        if not any(
            item["close_command_active"]
            for item in (gripper_before_pulse, gripper_after_ingress, gripper_final)
        )
        else "UNEXPECTED_CLOSE_STATE_OBSERVED"
    )
    pulse_window = {
        "before": submission_before,
        "after": ledger["authoritative_pulse"]["after"],
        "env_step_calls": pulse_interval["env_step_calls"],
        "action_manager_process_action_calls": pulse_interval["action_manager_process_action_calls"],
        "router_process_action_calls": 0,
        "logical_full_packet_consumption_count": 1 if pulse_interval["action_manager_process_action_calls"] == 1 else 0,
        "packet_equals_authoritative_full_8d": pulse_interval["process_action_packets"] == [
            _LifecycleCounter._packet_record(packet_tensor, "P0_AUTHORITATIVE_PLUS_X_SINGLE_SUBMISSION")
        ],
    }
    global_lifecycle = counter.summary()
    lifecycle_pass = (
        pulse_window["env_step_calls"] == 1
        and pulse_window["action_manager_process_action_calls"] == 1
        and pulse_window["router_process_action_calls"] == 0
        and pulse_window["logical_full_packet_consumption_count"] == 1
        and pulse_window["packet_equals_authoritative_full_8d"]
        and global_lifecycle["global_env_step_calls"]
        == global_lifecycle["global_action_manager_process_action_calls"]
    )
    semantic_pass = (
        scale_error <= P0_ROUTER_SCALE_ERROR_MAX_M
        and target_formula_error <= P0_TARGET_FORMULA_ERROR_MAX_M
        and target_delta[0] > 0.0
        and measured_delta[0] > P0_OBSERVED_POSITIVE_AXIS_FLOOR_M
        and orientation_drift <= P0_ORIENTATION_DRIFT_MAX_RAD
        and bool(gripper_states["unchanged_open"])
        and gripper_states["controller_transition"] == "NONE_OPEN_TO_OPEN"
    )
    functional = "PASS" if lifecycle_pass and semantic_pass else "FAIL"
    return {
        "schema": SCHEMA,
        "phase": "P0_A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ONE_AUTHORITATIVE_PLUS_X_PULSE_WITH_SOURCE_DEFINED_RESET_ZERO_BASELINE_AND_HOLDS_NO_P0_B_NO_P0_C_NO_M1_NO_M2_NO_TRAINING",
        "source_freeze": pre_live["source_freeze"],
        "pre_live_revalidation": pre_live,
        "canonical_factory": pre_live["static_contract"]["canonical_runtime_factory"],
        "read_only_observation_binding": {
            "provider_type": type(provider).__name__,
            "runtime_source_type": type(observation_source).__name__,
            "source_capture_calls": observation_source.captures,
            "authority": "NO_ACCEPTANCE_AUTHORITY",
            "binding": initial.binding.payload(),
            "binding_fingerprint": initial.binding.fingerprint(),
        },
        "authoritative_command": {
            "high_level_normalized": list(high_level.values),
            "metric_root_x_m": list(P0_METRIC_DELTA_M),
            "full_8d_packet": list(packet.values),
            "derivation": derivation,
            "p0_scoped_gripper_latch_final_intent": p0_gripper_latch.intent.value,
            "packet_tensor_shape": list(packet_tensor.shape),
            "packet_tensor_device": str(packet_tensor.device),
            "later_metric_0_20_mm_candidate_used": False,
        },
        "initialization_and_baseline": {
            "existing_p0_factory": "_make_env('P0_A')",
            "disabled_task_terms": _plain(disabled_task_terms),
            "read_only_initialization_before_any_policy_packet": initialization_baseline,
            "after_reset_open_settle_before_historical_zero_baseline": setup[
                "after_reset_open_settle_before_historical_zero_baseline"
            ],
            "source_defined_setup": setup,
            "zero_packet_derivation": zero_derivation,
            "global_lifecycle_before_authoritative_pulse": pulse_before,
        },
        "snapshots": {
            "initial_before_authoritative_pulse": _snapshot_payload(initial),
            "after_authoritative_ingress": _snapshot_payload(after_ingress),
            "final_after_source_defined_holds": _snapshot_payload(final),
        },
        "command_window": {
            "p0_authoritative_pulse": pulse_window,
            "global_lifecycle": global_lifecycle,
            "authoritative_nonzero_packet_count": 1,
            "authoritative_pulse_process_action_count": pulse_window[
                "action_manager_process_action_calls"
            ],
            "global_lifecycle_counts_are_source_defined_zero_open_traffic_and_not_additional_nonzero_p0_pulses": True,
            "source_defined_setup_and_hold_traffic_is_separate_from_authoritative_nonzero_pulse": True,
        },
        "response": {
            "initial_ee_root_m": list(initial.sample.ee_pose_root.position_m),
            "observed_ee_root_m": list(final.sample.ee_pose_root.position_m),
            "observed_ee_delta_root_m": list(measured_delta),
            "controller_target_delta_root_m": list(target_delta),
            "expected_controller_target_root_m": _plain(expected_target),
            "controller_origin_root_m": _plain(formula_origin),
            "controller_target_formula_error_m": target_formula_error,
            "controller_target_formula_error_source_max_m": P0_TARGET_FORMULA_ERROR_MAX_M,
            "controller_target_root_m_after_authoritative_ingress": _plain(target_after),
            "controller_target_root_m_after_source_defined_holds": _plain(target_final),
            "controller_target_delta_during_source_defined_holds_m": list(target_hold_delta),
            "controller_target_hold_behavior": "OBSERVED_ONLY_NO_NEW_ACCEPTANCE_THRESHOLD",
            "same_axis_repeat_before_command": same_axis_repeat,
            "controller_scale_m_per_normalized": list(scale),
            "scale_once_resolved_metric_delta_m": resolved_metric_delta,
            "router_scale_error_m": scale_error,
            "router_scale_error_source_max_m": P0_ROUTER_SCALE_ERROR_MAX_M,
            "frame_sign_attestation": {
                "frame": "robot_root",
                "target_x_positive": target_delta[0] > 0.0,
                "observed_x_positive_over_historical_floor": measured_delta[0] > P0_OBSERVED_POSITIVE_AXIS_FLOOR_M,
                "source_owned_observed_positive_floor_m": P0_OBSERVED_POSITIVE_AXIS_FLOOR_M,
            },
            "orientation_drift_rad": orientation_drift,
            "orientation_source_limit_rad": P0_ORIENTATION_DRIFT_MAX_RAD,
            "initial_arm_q_rad": list(initial.sample.joint_position_rad),
            "final_arm_q_rad": list(final.sample.joint_position_rad),
            "initial_arm_qd_rad_s": list(initial.sample.joint_velocity_rad_s),
            "final_arm_qd_rad_s": list(final.sample.joint_velocity_rad_s),
            "gripper_state_change": gripper_states,
            "simulation_timestamp": {
                "initial_control_epoch": initial.sample.control_epoch,
                "final_control_epoch": final.sample.control_epoch,
                "physics_dt_s": float(env.physics_dt),
                "environment_step_dt_s": float(env.step_dt),
            },
        },
        "termination_and_collision": {
            "existing_terms": list(env.termination_manager.active_terms),
            "events": ledger.get("submission_events", []),
            "forbidden_collision": "NOT_OBSERVED",
            "post_env_step_cause_note": "env.step may auto-reset; any terminated/truncated event fails even when a term is no longer readable after reset.",
        },
        "existing_runtime_rejection_state": "NO_PUBLIC_REJECTION_RECEIPT_IN_P0_INTEGRATION_SCOPE",
        "P0_A_FUNCTIONAL": functional,
        "P0_A_PROCESS": "UNCLASSIFIED_UNTIL_CHILD_EXIT",
        "P0_A_OVERALL": "PENDING_SUPERVISOR_PROCESS_VERDICT",
        "P0_B_AUTHORIZED_NEXT": "PENDING_SUPERVISOR_FULL_PASS",
        "P0_B": "BLOCKED_NOT_EXECUTED",
        "P0_C": "BLOCKED",
        "M1": "NOT_EXECUTED",
        "TRAINING": "NOT_AUTHORIZED",
    }


def _source_freeze(preflight_manifest: Path, static: Mapping[str, Any]) -> dict[str, Any]:
    manifest = static["immutable_preflight_manifest"]
    return {
        "SOURCE_FREEZE": "PASS" if static["SOURCE_FREEZE"] == "PASS" else "FAIL",
        "manifest_path": str(preflight_manifest),
        "manifest_sha256": manifest["sha256"],
        "preflight_source_hash_match": manifest["hash_match"],
        "preflight_contract_match": manifest["contract_match"],
        "production_usd_sha256": static["production_usd_sha256"],
        "frozen_trajectory_sha256": static["frozen_trajectory_sha256"],
        "local_source_sha256": static["local_source_sha256"],
    }


def _pre_live_revalidate(preflight_manifest: Path) -> dict[str, Any]:
    static = static_pre_live_contract(preflight_manifest)
    freeze = _source_freeze(preflight_manifest, static)
    return {
        "schema": "g2_p0_a_one_shot_pre_live_revalidation_v2",
        "status": "PASS" if static["SOURCE_FREEZE"] == "PASS" else "FAIL_CLOSED",
        "static_contract": static,
        "source_freeze": freeze,
        "no_isaac_started_by_revalidation": True,
    }


def _failure_report(*, pre_live: Mapping[str, Any], error: BaseException, ledger: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "phase": "P0_A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ONE_AUTHORITATIVE_PLUS_X_PULSE_NO_RETRY",
        "source_freeze": pre_live.get("source_freeze"),
        "pre_live_revalidation": pre_live,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "partial_command_ledger": _plain(ledger),
        "P0_A_FUNCTIONAL": "FAIL",
        "P0_A_PROCESS": "UNCLASSIFIED_UNTIL_CHILD_EXIT",
        "P0_A_OVERALL": "FAIL_PENDING_PROCESS_VERDICT",
        "P0_B_AUTHORIZED_NEXT": "NO",
        "P0_B": "BLOCKED_NOT_EXECUTED",
        "P0_C": "BLOCKED",
        "M1": "NOT_EXECUTED",
        "TRAINING": "NOT_AUTHORIZED",
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# P0-A one-shot integration child", "", "```text"]
    for key in (
        "P0_A_FUNCTIONAL",
        "P0_A_PROCESS",
        "P0_A_OVERALL",
        "P0_B_AUTHORIZED_NEXT",
        "P0_B",
        "P0_C",
        "M1",
        "TRAINING",
    ):
        lines.append(f"{key}: {payload.get(key)}")
    lines.extend(["```", "", "The parent supervisor adds exit/signal/timeout classification."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    preflight_manifest = args.preflight_manifest.expanduser().resolve()
    if output.exists():
        parser.error("--output must name a new immutable directory")
    output.mkdir(parents=True, exist_ok=False)
    pre_live = _pre_live_revalidate(preflight_manifest)
    _atomic_json(output / "PRE_LIVE_REVALIDATION.json", pre_live)
    if pre_live["status"] != "PASS":
        failure = _failure_report(
            pre_live=pre_live,
            error=RuntimeError("P0_PRE_LIVE_REVALIDATION_FAIL_CLOSED"),
            ledger={"authoritative_pulse": {"submitted": False}},
        )
        _atomic_json(output / "P0_A_FUNCTIONAL_PRE_CLOSE.json", failure)
        _atomic_text(output / "P0_A_FUNCTIONAL_PRE_CLOSE.md", _markdown(failure))
        print("PRE_LIVE_REVALIDATION_FAIL", flush=True)
        return 2

    print("SOURCE_FREEZE_OK", flush=True)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False, fast_shutdown=False)
    app = launcher.app
    print("APP_CREATED", flush=True)
    env = None
    counter = None
    provider = None
    observation_source = None
    ledger: dict[str, Any] = {"authoritative_pulse": {"submitted": False}, "submission_events": []}
    result: dict[str, Any]
    try:
        print("RUNTIME_BEGIN", flush=True)
        env, disabled_task_terms = _make_existing_p0_env()
        _assert_existing_p0_runtime_surface(env)
        counter = _LifecycleCounter(env)
        counter.install()
        provider, observation_source, _ = _runtime_observation_provider(env, pre_live["source_freeze"])
        result = _functional_report(
            env=env,
            provider=provider,
            observation_source=observation_source,
            counter=counter,
            disabled_task_terms=disabled_task_terms,
            pre_live=pre_live,
            ledger=ledger,
        )
        print("RUNTIME_END", flush=True)
    except BaseException as error:
        if counter is not None:
            ledger["lifecycle_counter"] = counter.summary()
        result = _failure_report(pre_live=pre_live, error=error, ledger=ledger)
        print("RUNTIME_END", flush=True)
    finally:
        if counter is not None:
            counter.restore()
        _atomic_json(output / "P0_A_FUNCTIONAL_PRE_CLOSE.json", result)
        _atomic_text(output / "P0_A_FUNCTIONAL_PRE_CLOSE.md", _markdown(result))
        print("REPORT_SAVED", flush=True)
        cleanup: dict[str, Any] = {
            "schema": "g2_p0_a_child_cleanup_pre_app_close_v1",
            "env_created": env is not None,
            "env_close": "NOT_ATTEMPTED" if env is not None else "NOT_APPLICABLE",
        }
        if env is not None:
            try:
                env.close()
                cleanup["env_close"] = "PASS"
            except BaseException as error:
                cleanup["env_close"] = "FAIL_EXCEPTION"
                cleanup["env_close_exception_type"] = type(error).__name__
                cleanup["env_close_exception"] = str(error)
                print("ENV_CLOSE_EXCEPTION", flush=True)
        # Native views must not be left to a Python finalizer after the Kit
        # application has begun teardown.  This follows the existing
        # application-alive cleanup pattern only; a close exception, signal,
        # timeout, or missing marker is still reported as a failure upstream.
        counter = None
        provider = None
        observation_source = None
        env = None
        cleanup["user_owned_runtime_references_released_before_app_close"] = True
        cleanup["gc_collect_while_application_alive"] = int(gc.collect())
        _atomic_json(output / "P0_A_CHILD_CLEANUP_PRE_APP_CLOSE.json", cleanup)
        print("APP_CLOSE_BEGIN", flush=True)
        app.close()
        print("APP_CLOSED", flush=True)
        print("PROCESS_EXIT", flush=True)
    return 0 if result.get("P0_A_FUNCTIONAL") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
