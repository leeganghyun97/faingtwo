#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded cuRobo delivery, with an optional BC/GRU handoff continuation.

The child consumes saved planner EE waypoints, not planner joint targets.  It
keeps the gripper OPEN and submits every exact-4D nominal action through the
existing complete 8-D ``ManagerBasedRLEnv.step`` ingress.  Functional
evidence is persisted before app shutdown; the known Isaac finalization
failure remains a parent-process verdict.  Without ``--bc-checkpoint`` the
historical planner-only behavior is unchanged.  With a checkpoint, the BC GRU
is rolled through the real planner history but cannot command the robot until
the planner has completed its final settle/telemetry-flush contract.

The historical Candidate-A path consumes a frozen seed-42 replay.  A separate,
explicitly opt-in runtime-replan mode is also available for a future reviewed
contact-free collection run.  It obtains the cube pose and arm state only
*after* ``env.reset()``, calls the planner, and then reuses the same canonical
4-D/complete-8-D action route.  It is not a policy input path and it does not
make CLOSE, contact, BC, or replay collection permissible by itself.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Callable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
for _path in (str(ROOT), str(SOURCE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from geniesim.rl.isaaclab.g2_quaternion import (  # noqa: E402
    isaaclab_native_quaternion_order,
    quaternion_native_to_xyzw,
)
from geniesim.rl.isaaclab.g2_camera_timing import (  # noqa: E402
    camera_capture_time_and_age,
)
from geniesim.rl.isaaclab.g2_rebuild.sensor_packet import (  # noqa: E402
    world_quaternions_to_root,
    world_points_to_root_from_native_quaternion,
)
from geniesim.rl.isaaclab.g2_policy_branch.rgbd_logging_contract import (  # noqa: E402
    CAMERA_TIMESTAMP_SOURCE,
    DEPTH_RAW_UNIT,
    DEPTH_TO_METER_SCALE,
    RGBD_CAMERA_NAMES,
    RGBD_EVIDENCE_SCHEMA,
    flatten_calibration_metadata,
)
from geniesim.rl.sac.close_mechanics_telemetry import (  # noqa: E402
    CloseMechanicsRingBuffer500Hz,
)
from geniesim.rl.sac.stage1a_exception_safe_transition import (  # noqa: E402
    RUNTIME_HARDSTOP_REASON,
    RuntimeHardstop,
    RuntimeHardstopReceipt,
)

BASELINE = ROOT / "artifacts/g2_curobo_delta_qualification_20260920/BASELINE_FREEZE.json"
P0_DELTA = ROOT / "artifacts/g2_curobo_delta_qualification_20260920/P0_DELTA_VALIDATION.json"
HANDOFF = ROOT / "artifacts/g2_curobo_delta_qualification_20260920/CANONICAL_4D_HANDOFF.json"
PREFLIGHT_SOURCE = ROOT / "scripts/diagnostics/run_g2_policy_4d_training_runtime_preflight.py"
P0C_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0c_current_integration.py"
PREGRASP_TOLERANCE_M = 0.003  # Existing MotionGen planning tolerance from Stage 5.
MAXIMUM_SETTLE_STEPS = 40
MAXIMUM_TRACKING_STEPS_PER_WAYPOINT = 20  # Diagnostic budget, not a safety limit.
MEASURED_ACTIVE_VELOCITY_HARD_LIMIT_RAD_S = 0.8
AB_ASSET_VARIANTS = ("candidate", "production", "custom")
HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD = 1.0e-5
HARD_STOP_ACCELERATION_LIMIT_RAD_S2 = 10.0
CANDIDATE_A_ASSET = (
    ROOT
    / "artifacts/g2_bounded_passive_range_qualification_20260921"
    / "candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
)
EXPECTED_CANDIDATE_A_ASSET_SHA256 = (
    "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
)
STAGE1A_FROZEN_GRU_CHECKPOINT = (
    Path(os.environ["GENIESIM_GRU_CHECKPOINT"])
    if os.environ.get("GENIESIM_GRU_CHECKPOINT")
    else ROOT
    / "artifacts/g2_keyboard_v3_recovery_20260924_v1"
    / "GRU_CAUSAL_CLOSE_EDGE_GPU_3K_WANDB_V1"
    / "BEST_HUMAN_GRASP_GRU_BC.pt"
)
STAGE1A_FROZEN_GRU_SHA256 = (
    "fdec68aef317e7cd1758bf4b48f4fe20207353ebd51d8d20e4378a0d2324d510"
)
STAGE1A_FAR_REACH_BC_CHECKPOINT = Path(
    os.environ.get(
        "GENIESIM_FAR_REACH_BC_CHECKPOINT",
        ROOT / "artifacts/external/far_reach_bc_best.pt",
    )
)
STAGE1A_FAR_REACH_BC_SHA256 = (
    "bcb1b7ed0455453c38762d9da00f9db30b5e811ddff44fe1fdbc0957588a14bc"
)
STAGE1A_RESIDUAL_ACTOR_CHECKPOINT = (
    Path(os.environ["GENIESIM_RESIDUAL_ACTOR_CHECKPOINT"])
    if os.environ.get("GENIESIM_RESIDUAL_ACTOR_CHECKPOINT")
    else ROOT
    / "artifacts/g2_candidate_a_close_sac_exception_safe_qualification_20260925_v5"
    / "checkpoints/stage1a_residual_actor.pt"
)
STAGE1A_RESIDUAL_ACTOR_SHA256 = (
    "bbc32ed96704e80058b5d9791194dca31d5862019a96edcb9beef48d2f85959d"
)
STAGE1A_ACCEPTED_TRANSITION_TARGET = 1000
STAGE1A_TRAINING_TRANSITION_TARGET = 15000
STAGE1A_REWARD_V3_TRANSITION_TARGET = 3000
STAGE1A_REPLAY_STRATEGIES = ("SAC", "HER", "HER_FORCE")


def _canonical_camera_acquisition_timing(
    *, physics_dt_s: float, capture_rate_hz: int
) -> tuple[float, int]:
    """Resolve sensor acquisition cadence without legacy render-step aliases."""

    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise RuntimeError("CANONICAL_COLLECTION_PHYSICS_DT_INVALID")
    if isinstance(capture_rate_hz, bool) or capture_rate_hz <= 0:
        raise RuntimeError("CANONICAL_COLLECTION_CAPTURE_RATE_INVALID")
    period_s = 1.0 / float(capture_rate_hz)
    physics_steps_float = period_s / float(physics_dt_s)
    physics_steps = int(round(physics_steps_float))
    if physics_steps <= 0 or not math.isclose(
        physics_steps_float, float(physics_steps), rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError("CANONICAL_COLLECTION_CAPTURE_PERIOD_NOT_PHYSICS_ALIGNED")
    return period_s, physics_steps

# The Phase-6 baseline remains immutable.  Keyboard-v3 is a later, narrower
# Candidate-A branch whose only baseline-source delta is the dedicated
# Candidate-A factory entry point.  Do not silently bless arbitrary changes:
# every new source participating in the one-shot route is pinned here.
KEYBOARD_V3_BRANCH_SOURCE_HASHES = {
    "source/geniesim/rl/isaaclab/g2_policy_branch/production_metric_adapter.py":
        "447f63f9fbbe053e0bea8f3147b0bba14b6a2433e6190a6224d7964c4b48b7f7",
    "configs/g2_policy_branch/keyboard_collection_v3.json":
        "9bed6e1114341cf450359330c67fa2ec56c16aa2ad7fb4103419f7670de73afe",
    "scripts/prepare_g2_keyboard_v3_collection.py":
        "ea30a0fb09f49401695a2d33f74015a2c074e044f4cdbeeb40c2d2b6df790329",
    "scripts/run_g2_keyboard_v3_terminal_collection.sh":
        "ea477a559ce4fd3183ff755e499da0d528c316d38bd407af7a05d140ed6bf66b",
    "scripts/train_g2_canonical_contact_free_bc.py":
        "b7f8f88981581ef12cda7089b5d6a969694af58e18b64656b0ce83682e579bfb",
    "scripts/train_g2_independent_contact_free_bc.py":
        "e4f1dd9b317a93eaa9fee88db64584ff703e8999a3e40c9224f28b2be42e9b36",
    "scripts/run_g2_independent_contact_free_collection.py":
        "7bd71d0e4bf6d07f2b1f7a36b1766f762ee687555ebab09e9b49a531ced8d0bb",
    "scripts/run_g2_independent_contact_free_bc_after_collection.py":
        "5b66d28e421d2e5133e8235358dcde0ace897551d9d221b6f093661f2ee2135f",
    "source/geniesim/rl/isaaclab/g2_policy_branch/training_env_factory.py":
        "cc8c83a6aa7c9c66aa9bfecacea95947f35036b568400ee3ebff85939493194a",
    "source/geniesim/rl/isaaclab/g2_policy_branch/contact_free_candidate_a_binding.py":
        "d18a1160e464d5ecfa8b9c3f6271a766bef9426a51e32194a843c0b1f5103e2f",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_collection_contract.py":
        "ba30940935a3b9bec1ea6fc5d3e2bdff6964b33feaff81aa2719ae3ec598c39d",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_operator_ui.py":
        "45033991810e12fdce9fdc01d1c4c784fe02b342329b30c2dce91d532741050b",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_runtime.py":
        "a36605f8e497cf87e3fb014da88521488e77e8373932e6d790ee70a19ff3a125",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_curobo_planner.py":
        "853c348ce191f365d3baa2edbd5f5bf8f06823f71877012ad7e774a297340757",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_live_driver.py":
        "a7a27b54827872ab2e91ba6636061afc78eaa9c5ee382bf5c1becc6fdc2bd418",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_operator_views.py":
        "517ce5f0b070c824be76fd60db13909c2085170fbedbc8e3701445b2b094d5d7",
    "source/geniesim/rl/sac/keyboard_v3_dataset.py":
        "db91a51e2987bafed31bf8cef4f911e0badffe6b32a9a34d41522be4f911e7c6",
    "source/geniesim/rl/isaaclab/g2_policy_branch/rgbd_logging_contract.py":
        "347cb462c8650c5cfd33b5c01f6b453916c304742670d01ed004750268c5590f",
    "source/geniesim/rl/isaaclab/g2_policy_branch/canonical_contact_free_collection_v2.py":
        "85f2d03337fd3d5541c41e5b94ad5d51eb40c77c6be7c66eac7f8bf8dc9f5d6f",
    "source/geniesim/rl/isaaclab/g2_policy_branch/candidate_a_left_arm_down_v2.py":
        "c3b137d2b7504fb2e092c1e7e3607dde858e6c6a38ebf0366ba993be96dbb7d8",
    "source/geniesim/rl/isaaclab/g2_policy_branch/omnipicker_product_contract.py":
        "b6a89d07224fd98c32a5edc553188f592e27eb438c2d97db5e43b0f078ee727f",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_direct_pregrasp_init.py":
        "fe8df84effffe058a1161770e03dd351776a1055a6050b70aecc654057160737",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_terminal_input.py":
        "ebaeb2b8efdbad717c1b35731cf3892477d62f147fd3f4a617e11416f2cc4d4c",
    "artifacts/g2_candidate_a_left_arm_down_v2/BASELINE_MANIFEST.json":
        "cfd930feebb152c0a583b3bc418a5f4982111959814dea491c0720e57e2dfea3",
    "source/geniesim/rl/sac/keyboard_grasp_contract.py":
        "8a0c2e227cd3d3649ca15970596fd7cb57c492761db61b29f398f035cf710718",
    "source/geniesim/rl/sac/human_grasp_gru_bc.py":
        "2e73c4183166cdd1d2b3d287d9c0dc7166bd21aa72257d87e39ec89d3bc4934a",
    "source/geniesim/rl/sac/bc_residual_sac_contract.py":
        "1e7724791be1fe837e6f08d2d8513722f9c7c38becadb7e5dbdd23bc6b12a81d",
    "source/geniesim/rl/sac/keyboard_grasp_her_force.py":
        "c1e2ba18c894ca77651016a6d7af5b13b365f63e3bb775ae8b2adaf3303345c9",
    "source/geniesim/rl/sac/residual_her_force.py":
        "269aee67374e54144311ad39b5026204998579399be4d6482e0767129c580780",
    "source/geniesim/rl/isaaclab/g2_policy_branch/curobo_planner_authority.py":
        "00e3fdf006facfa45a1d69cf683e0d3b832d3c1ae21f0ad7e5757dda4d985576",
    "source/geniesim/rl/isaaclab/g2_rebuild/sensor_packet.py":
        "425f474043fab61308639f977ff1a356d31d899b05e9b72e1ca5f8eac8032985",
    "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_action_adapter.py":
        "c5420d53ebea78fc86b8f8b081e50a047933bbe9da4d1c179e26f404051f38e5",
    "source/geniesim/rl/isaaclab/g2_lift_rgbd_env_cfg.py":
        "40aebe470779f75d097abc4e3e7edbabaf91f23f147068a61d4dbb7114f2e027",
    "source/geniesim/rl/isaaclab/g2_lift_task_mdp.py":
        "0a5e577eb8f5efe50a500bda4c55c1db5a3ce7f65e3a1a82640b7a604599dd9b",
    "source/geniesim/rl/isaaclab/g2_asset_camera_pose.py":
        "75e60897d4b62ce34adb58991898380c26bd12a5c103a765b12dbf2c43314dab",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
    # Some bounded branches persist their first post-reset receipt before the
    # common report writer has created the run directory.  Artifact plumbing
    # must never be the reason a valid planner handoff is skipped.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _durable_jsonl(stream: Any, payload: Mapping[str, Any]) -> None:
    """Append one fsync'd lifecycle receipt to an already-open JSONL stream."""

    stream.write(
        json.dumps(_json_ready(payload), sort_keys=True, allow_nan=False) + "\n"
    )
    stream.flush()
    os.fsync(stream.fileno())


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _config_leaf_differences(
    before: Any, after: Any, *, prefix: str = ""
) -> list[dict[str, Any]]:
    """Return deterministic leaf differences for a config-only A/B assertion."""

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                differences.append(
                    {
                        "path": path,
                        "before": _json_ready(before.get(key)),
                        "after": _json_ready(after.get(key)),
                    }
                )
                continue
            differences.extend(
                _config_leaf_differences(before[key], after[key], prefix=path)
            )
        return differences
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        differences = []
        for index in range(max(len(before), len(after))):
            path = f"{prefix}[{index}]"
            if index >= len(before) or index >= len(after):
                differences.append(
                    {
                        "path": path,
                        "before": _json_ready(before[index]) if index < len(before) else None,
                        "after": _json_ready(after[index]) if index < len(after) else None,
                    }
                )
                continue
            differences.extend(
                _config_leaf_differences(before[index], after[index], prefix=path)
            )
        return differences
    if before == after:
        return []
    return [{"path": prefix, "before": _json_ready(before), "after": _json_ready(after)}]


def _load_exact_module(path: Path, name: str) -> Any:
    resolved = path.resolve()
    before = _sha256(resolved)
    spec = importlib.util.spec_from_file_location(f"{name}_{before[:16]}", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_FAILED:{resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != resolved or _sha256(resolved) != before:
        raise RuntimeError(f"MODULE_PROVENANCE_CHANGED:{resolved}")
    return module


def _load_contact_free_visual_bc_checkpoint(
    checkpoint: Path, *, expected_sha256: str, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Compatibility wrapper over the one shared cuRobo-BC loader."""

    from geniesim.rl.isaaclab.g2_policy_branch.contact_free_visual_bc_runtime import (
        load_contact_free_visual_bc_checkpoint,
    )

    try:
        return load_contact_free_visual_bc_checkpoint(
            checkpoint,
            expected_sha256=expected_sha256,
            device=device,
        )
    except ValueError as error:
        raise RuntimeError(f"CONTACT_FREE_BC_PLAYBACK_LOAD_FAILED:{error}") from error


def _source_freeze(*, keyboard_v3_branch: bool = False) -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for record in baseline["immutable_inputs"].values():
        expected[str(record["path"])] = str(record["sha256"])
    expected.update(baseline["source_hashes_from_authoritative_phase6_receipt"])
    phase6_expected = dict(expected)
    branch_delta: dict[str, dict[str, Any]] = {}
    if keyboard_v3_branch:
        authorized_replacements = {
            (
                "source/geniesim/rl/isaaclab/g2_policy_branch/"
                "production_metric_adapter.py"
            ): "EXCEPTION_SAFE_PARTIAL_TERMINATED_LIFECYCLE_ADDITION",
            (
                "source/geniesim/rl/isaaclab/g2_policy_branch/"
                "training_env_factory.py"
            ): "DEDICATED_CANDIDATE_A_FACTORY_ADDITION",
            "source/geniesim/rl/isaaclab/g2_lift_task_mdp.py": (
                "SIM_BOOLEAN_CONTACT_TELEMETRY_ADDITION_NO_THRESHOLD_CHANGE"
            ),
        }
        for relative, reason in authorized_replacements.items():
            historical = expected.get(relative)
            replacement = KEYBOARD_V3_BRANCH_SOURCE_HASHES[relative]
            if historical is None or historical == replacement:
                raise RuntimeError("KEYBOARD_V3_BRANCH_DELTA_AUTHORITY_INVALID")
            expected[relative] = replacement
            branch_delta[relative] = {
                "phase6_expected": historical,
                "keyboard_v3_expected": replacement,
                "reason": reason,
            }
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in expected.items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        observed[relative] = {
            "expected": expected_hash,
            "actual": actual,
            "match": actual == expected_hash,
        }
    additional = (
        P0_DELTA,
        HANDOFF,
        PREFLIGHT_SOURCE,
        P0C_SOURCE,
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/curobo_4d_handoff.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/planner_timing_contract.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/planner_bc_state_bridge.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/precontact_contract.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/canonical_contact_free_collection_v2.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/dls_read_only_preview_api.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/dls_read_only_preview_capture.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/contact_free_runtime_replan.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/contact_free_visual_bc_runtime.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/hybrid_grasp_runtime.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/curobo_collision_world.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_lift_env_cfg.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_lift_rgbd_env_cfg.py",
        ROOT / "source/geniesim/rl/sac/stage1a_isaac_short_smoke.py",
        ROOT / "source/geniesim/rl/sac/stage1a_exception_safe_transition.py",
        ROOT / "source/geniesim/rl/sac/stage1a_grasp_reward.py",
        ROOT / "source/geniesim/rl/sac/stage1a_isaac_telemetry_adapter.py",
        ROOT / "source/geniesim/rl/sac/stage1a_real_sac_coordinator.py",
        ROOT / "source/geniesim/rl/sac/residual_sac_runtime.py",
        ROOT / "source/geniesim/rl/sac/close_mechanics_telemetry.py",
        ROOT / "source/geniesim/rl/sac/privileged_geometry_oracle.py",
        Path(__file__).resolve(),
    )
    additional_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in additional}
    keyboard_v3_observed: dict[str, dict[str, Any]] = {}
    if keyboard_v3_branch:
        for relative, expected_hash in KEYBOARD_V3_BRANCH_SOURCE_HASHES.items():
            path = ROOT / relative
            actual = _sha256(path) if path.is_file() else None
            keyboard_v3_observed[relative] = {
                "expected": expected_hash,
                "actual": actual,
                "match": actual == expected_hash,
            }
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "frozen": observed,
                "additional": additional_hashes,
                "keyboard_v3": keyboard_v3_observed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    passed = all(bool(record["match"]) for record in observed.values())
    passed &= all(
        bool(record["match"]) for record in keyboard_v3_observed.values()
    )
    p0_delta = json.loads(P0_DELTA.read_text(encoding="utf-8"))
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    passed &= p0_delta["verdict"]["PLANNER_ONLY_LIVE_SMOKE_AUTHORIZED_NEXT"] == "YES"
    passed &= handoff["verdict"]["CANONICAL_4D_HANDOFF"] == "PASS"
    return {
        "SOURCE_FREEZE": "PASS" if passed else "FAIL",
        "SOURCE_FREEZE_COMPLETE": bool(passed),
        "manifest_sha256": fingerprint,
        "production_usd_sha256": baseline["immutable_inputs"]["production_usd"]["sha256"],
        "m2_candidate_usd_sha256": baseline["immutable_inputs"]["m2_candidate_usd"]["sha256"],
        "frozen_trajectory_sha256": baseline["immutable_inputs"]["frozen_trajectory"]["sha256"],
        "frozen": observed,
        "additional": additional_hashes,
        "freeze_profile": (
            "KEYBOARD_V3_CANDIDATE_A_ONE_SHOT"
            if keyboard_v3_branch
            else "PHASE6_HISTORICAL"
        ),
        "historical_phase6_expected": phase6_expected,
        "authorized_branch_delta": branch_delta,
        "keyboard_v3_sources": keyboard_v3_observed,
        "freeze_scope": {
            "asset": {
                "candidate_a_composed_usd_sha256": EXPECTED_CANDIDATE_A_ASSET_SHA256,
                "production_usd_sha256": baseline["immutable_inputs"]["production_usd"]["sha256"],
                "used_layer_hashes": "NOT_EXPOSED_BY_USD_SINGLE_FILE; COMPOSED_ASSET_HASH_BOUND",
            },
            "mechanics": [
                "joint_limits",
                "master_mimic_topology",
                "spherical_loops",
                "mass_inertia",
                "drive_properties",
                "materials",
                "collision_geometry",
                "solver_articulation_configuration",
            ],
            "control": [
                "exact_4d_high_level_action",
                "metric_to_normalized_adapter",
                "single_consumption_env_step_path",
                "open_only_gripper_route",
                "direct_follower_writes_forbidden",
            ],
            "planner": {
                "trajectory_hash": baseline["immutable_inputs"]["frozen_trajectory"]["sha256"],
                "maximum_segment_translation_m": 0.0045,
            },
            "safety": {
                "clearance_guard_m": 0.025051395199026397,
                "fail_closed": True,
                "existing_safety_thresholds_unchanged": True,
            },
            "policy": {
                "bc_loaded": False,
                "rl_loaded": False,
                "residual_action": "ZERO_DISABLED",
            },
        },
    }


def _tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    return torch.from_dlpack(value)


def _camera_pose_root_m_xyzw(env: Any, camera_name: str) -> np.ndarray:
    camera = env.scene[f"{camera_name}_camera"]
    robot = env.scene["robot"]
    position_root = world_points_to_root_from_native_quaternion(
        _tensor(camera.data.pos_w),
        _tensor(robot.data.root_pos_w),
        _tensor(robot.data.root_quat_w),
        native_order=isaaclab_native_quaternion_order(),
    )
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(robot.data.root_quat_w), isaaclab_native_quaternion_order()
    )
    camera_quaternion_root_xyzw = world_quaternions_to_root(
        _tensor(camera.data.quat_w_world), root_quaternion_xyzw
    )
    import torch

    return (
        torch.cat((position_root, camera_quaternion_root_xyzw), dim=-1)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float32, copy=True)
    )


def _camera_calibration_snapshot(env: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for camera_name in RGBD_CAMERA_NAMES:
        camera = env.scene[f"{camera_name}_camera"]
        intrinsics = (
            _tensor(camera.data.intrinsic_matrices)[0]
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        )
        result[camera_name] = {
            "camera_name": camera_name,
            "source_scene_key": f"{camera_name}_camera",
            "source_frame_id": str(camera.cfg.prim_path),
            "intrinsics_3x3": intrinsics.tolist(),
            "image_width": int(camera.cfg.width),
            "image_height": int(camera.cfg.height),
            "extrinsic_reference_frame": "robot_root",
            "pose_convention": "position_m+quaternion_xyzw",
            "timestamp_source": CAMERA_TIMESTAMP_SOURCE,
        }
    return result


def _ee_root_position(env: Any) -> np.ndarray:
    p0a = env._g2_curobo_p0a_module
    position, _quaternion = p0a._world_ee_pose_to_root(
        env.scene["robot"], env.scene["ee_frame"]
    )[:2]
    return (
        _tensor(position)[0].detach().to("cpu").numpy().astype(np.float64)
    )


def _ee_root_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    p0a = env._g2_curobo_p0a_module
    position, quaternion = p0a._world_ee_pose_to_root(
        env.scene["robot"], env.scene["ee_frame"]
    )[:2]
    return (
        _tensor(position)[0].detach().to("cpu").numpy().astype(np.float64),
        _tensor(quaternion)[0].detach().to("cpu").numpy().astype(np.float64),
    )


def _cube_root_position(env: Any) -> np.ndarray:
    robot_data = env.scene["robot"].data
    root_position_world_m = _tensor(robot_data.root_pos_w)
    cube_position_root_m = world_points_to_root_from_native_quaternion(
        _tensor(env.scene["object"].data.root_pos_w),
        root_position_world_m,
        _tensor(robot_data.root_quat_w),
        native_order=isaaclab_native_quaternion_order(),
    )
    return (
        cube_position_root_m[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )


def _capture_keyboard_v3_handoff_state(env: Any) -> dict[str, Any]:
    """Capture the already-qualified cuRobo handoff for in-process reuse."""

    robot = env.scene["robot"]
    cube = env.scene["object"]
    return {
        "robot_joint_pos": _tensor(robot.data.joint_pos).clone(),
        "robot_joint_vel": _tensor(robot.data.joint_vel).clone(),
        "cube_root_pose": _tensor(cube.data.root_state_w)[:, :7].clone(),
        "cube_root_velocity": _tensor(cube.data.root_state_w)[:, 7:13].clone(),
        "ee_position_root_m": _ee_root_position(env).copy(),
    }


def _restore_keyboard_v3_handoff_state(
    *,
    env: Any,
    preflight: Any,
    counter: Any,
    deferred_port: Any,
    open_packet: Any,
    latch: Any,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Reset managers, restore the handoff pair, and rebase controller caches."""

    import torch

    env.reset()
    robot = env.scene["robot"]
    cube = env.scene["object"]
    q = snapshot["robot_joint_pos"].clone().to(env.device)
    qd = snapshot["robot_joint_vel"].clone().to(env.device)
    robot.write_joint_state_to_sim(q, qd)
    arm_term = env.action_manager.get_term("arm_action")
    arm_ids_long = torch.as_tensor(
        [int(value) for value in arm_term._joint_ids],
        device=env.device,
        dtype=torch.long,
    )
    arm_ids_warp = arm_ids_long.to(dtype=torch.int32)
    robot.set_joint_position_target(
        q.index_select(1, arm_ids_long), joint_ids=arm_ids_warp
    )
    cube.write_root_pose_to_sim(snapshot["cube_root_pose"].clone().to(env.device))
    cube.write_root_velocity_to_sim(
        snapshot["cube_root_velocity"].clone().to(env.device)
    )
    env_ids = torch.zeros((1,), device=env.device, dtype=torch.long)
    synchronize = getattr(arm_term, "synchronize_reset_to_measured", None)
    if not callable(synchronize):
        raise RuntimeError("KEYBOARD_SESSION_ARM_CACHE_REBASE_API_MISSING")
    synchronize(env_ids)
    previous = getattr(env, "_g2_previous_accepted_policy_action", None)
    if previous is not None and callable(getattr(previous, "reset", None)):
        previous.reset()
    latch.reset()
    receipts: list[dict[str, Any]] = []
    for refresh_index in range(2):
        outputs, receipt = preflight._consume_once(
            env=env,
            counter=counter,
            deferred_port=deferred_port,
            packet=open_packet,
            label=f"KEYBOARD_SESSION_RESET_REFRESH_{refresh_index + 1}",
        )
        if not bool(receipt["single_consumption"]):
            raise RuntimeError("KEYBOARD_SESSION_RESET_NOT_SINGLE_CONSUMPTION")
        latch.commit(open_packet.gripper_intent)
        _, _, terminated, truncated, _ = outputs
        active = preflight._active_termination_names(env, terminated, truncated)
        if bool(terminated.reshape(-1)[0].item()) or bool(
            truncated.reshape(-1)[0].item()
        ) or active:
            raise RuntimeError("KEYBOARD_SESSION_RESET_TERMINATED:" + ",".join(active))
        receipts.append(dict(receipt))
    synchronize(env_ids)
    restored_ee = _ee_root_position(env)
    ee_error_m = float(
        np.linalg.norm(restored_ee - np.asarray(snapshot["ee_position_root_m"]))
    )
    emitted = _tensor(arm_term.last_emitted_joint_position_target)
    measured = _tensor(robot.data.joint_pos).index_select(1, arm_ids_long)
    target_error_rad = float(torch.max(torch.abs(emitted - measured)).item())
    if ee_error_m > 0.003 or target_error_rad > 1.0e-6:
        raise RuntimeError(
            f"KEYBOARD_SESSION_HANDOFF_RESTORE_MISMATCH:{ee_error_m}:{target_error_rad}"
        )
    return {
        "schema": "g2_keyboard_v3_in_process_handoff_restore_v1",
        "startup_curobo_planning_count": 0,
        "refresh_action_count": len(receipts),
        "restored_ee_error_m": ee_error_m,
        "target_to_measured_max_rad": target_error_rad,
        "single_consumption": all(bool(row["single_consumption"]) for row in receipts),
    }


def _numbered_episode_id(initial_episode_id: str, offset: int) -> str:
    prefix, separator, suffix = initial_episode_id.rpartition("-")
    if not separator or not suffix.isdigit():
        raise RuntimeError("CONTINUOUS_SESSION_REQUIRES_NUMBERED_EPISODE_ID")
    return f"{prefix}-{int(suffix) + int(offset):0{len(suffix)}d}"


def _run_keyboard_v3_session(
    *,
    collection_root: Path,
    initial_episode_id: str,
    initial_report_path: Path,
    continuous_session: bool,
    maximum_episodes: int,
    startup_curobo_plan_count: int,
    initial_restore_receipt: Mapping[str, Any] | None,
    run_episode: Callable[[str, Path, Mapping[str, Any] | None], int],
    restore_for_next_episode: Callable[[], Mapping[str, Any]],
) -> int:
    """Run one or more bounded episodes without restarting Isaac.

    Episode execution remains one-shot and fail-closed: an unsafe episode is
    never promoted.  In a continuous operator session, however, a persisted
    episode failure does not tear down Isaac.  The supervisor restores the
    validated handoff pair and starts a fresh independent episode.  Only an
    explicit operator quit, a configured positive episode bound, or a failure
    before a durable report exists ends the session.
    """

    session_progress_path = collection_root / (
        f"KEYBOARD_V3_SESSION_{initial_episode_id}.json"
    )
    session_rows: list[dict[str, Any]] = []
    episode_offset = 0
    latest_restore_receipt = initial_restore_receipt
    while True:
        current_episode_id = _numbered_episode_id(
            initial_episode_id, episode_offset
        )
        current_report = (
            initial_report_path
            if episode_offset == 0
            else collection_root
            / f"KEYBOARD_V3_LIVE_REPORT_{current_episode_id}.json"
        )
        episode_started_s = time.monotonic()
        final_status = run_episode(
            current_episode_id, current_report, latest_restore_receipt
        )
        episode_elapsed_s = time.monotonic() - episode_started_s
        episode_report = json.loads(current_report.read_text(encoding="utf-8"))
        session_rows.append(
            {
                "episode_id": current_episode_id,
                "report_path": str(current_report),
                "functional": bool(
                    episode_report.get("keyboard_v3_live_driver_qualified")
                ),
                "save_class": episode_report.get("save_class"),
                "saved_path": episode_report.get("saved_path"),
                "session_stop_requested": bool(
                    episode_report.get("session_stop_requested", False)
                ),
                "runtime_error": episode_report.get("runtime_error"),
                "wall_duration_s": float(episode_elapsed_s),
            }
        )
        _atomic_json(
            session_progress_path,
            {
                "schema": "g2_keyboard_v3_continuous_session_v2",
                "initial_episode_id": initial_episode_id,
                "continuous_session": bool(continuous_session),
                "startup_curobo_plan_count": int(startup_curobo_plan_count),
                "completed_episode_count": len(session_rows),
                "failed_episode_count": sum(
                    int(row["runtime_error"] is not None) for row in session_rows
                ),
                "process_lifetime_authority": (
                    "EXPLICIT_OPERATOR_QUIT_OR_CONFIGURED_EPISODE_BOUND"
                ),
                "episodes": session_rows,
            },
        )
        print(
            "[KEYBOARD_V3_SESSION_PROGRESS] "
            f"completed={len(session_rows)} "
            f"canonical={sum(row['save_class'] == 'CANONICAL' for row in session_rows)} "
            f"rejected={sum(row['save_class'] == 'REJECTED' for row in session_rows)} "
            f"discarded={sum(row['save_class'] == 'DISCARD' for row in session_rows)} "
            f"episode_wall_s={episode_elapsed_s:.3f}",
            flush=True,
        )
        stop_requested = bool(
            episode_report.get("session_stop_requested", False)
        )
        bounded_complete = bool(
            maximum_episodes > 0 and len(session_rows) >= maximum_episodes
        )
        if (
            not continuous_session
            or stop_requested
            or bounded_complete
        ):
            return int(final_status)
        if final_status != 0:
            print(
                "KEYBOARD_V3_EPISODE_FAILED_SESSION_REMAINS_ALIVE "
                f"episode={current_episode_id} status={final_status} "
                f"runtime_error={episode_report.get('runtime_error')}",
                flush=True,
            )
        episode_offset += 1
        restore_started_s = time.monotonic()
        latest_restore_receipt = dict(restore_for_next_episode())
        restore_elapsed_s = time.monotonic() - restore_started_s
        print(
            "KEYBOARD_V3_NEXT_EPISODE_READY "
            f"episode={_numbered_episode_id(initial_episode_id, episode_offset)} "
            f"restore_error_m={float(latest_restore_receipt['restored_ee_error_m']):.9f} "
            f"restore_wall_s={restore_elapsed_s:.3f}",
            flush=True,
        )


def _ab_settled_state_snapshot(env: Any, robot: Any) -> dict[str, Any]:
    """Read the common reset/OPEN state without changing controller authority."""

    joint_names = tuple(robot.joint_names)
    requested = (
        "idx79_gripper_r_inner_joint0",
        "idx71_gripper_r_inner_joint1",
        "idx72_gripper_r_inner_joint3",
        "idx73_gripper_r_inner_joint4",
        "idx89_gripper_r_outer_joint0",
        "idx81_gripper_r_outer_joint1",
        "idx82_gripper_r_outer_joint3",
        "idx83_gripper_r_outer_joint4",
    )
    missing = [name for name in requested if name not in joint_names]
    if missing:
        raise RuntimeError(f"CUROBO_AB_SETTLED_JOINTS_MISSING:{missing}")
    q = _tensor(robot.data.joint_pos)[0].detach().to("cpu").numpy().astype(np.float64)
    qdot = _tensor(robot.data.joint_vel)[0].detach().to("cpu").numpy().astype(np.float64)
    limits = (
        _tensor(robot.data.joint_pos_limits)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    positions = (
        _tensor(robot.data.body_pos_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    quaternions = (
        _tensor(robot.data.body_quat_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    body_names = tuple(robot.body_names)
    pad_names = ("gripper_r_inner_link4", "gripper_r_outer_link4")
    if any(name not in body_names for name in pad_names):
        raise RuntimeError("CUROBO_AB_PAD_LINKS_MISSING")
    pad_indices = [body_names.index(name) for name in pad_names]
    pad_positions = positions[pad_indices]
    pad_quaternions = quaternions[pad_indices]
    root = _tensor(robot.data.root_pos_w)[0].detach().to("cpu").numpy().astype(np.float64)
    cube_root = _cube_root_position(env)
    ee_root = _ee_root_position(env)
    joint_state: dict[str, Any] = {}
    for name in requested:
        index = joint_names.index(name)
        margin = min(q[index] - limits[index, 0], limits[index, 1] - q[index])
        joint_state[name] = {
            "q_rad": float(q[index]),
            "q_deg": float(np.degrees(q[index])),
            "qdot_rad_s": float(qdot[index]),
            "lower_limit_rad": float(limits[index, 0]),
            "upper_limit_rad": float(limits[index, 1]),
            "limit_margin_rad": float(margin),
        }
    midpoint_world = np.mean(pad_positions, axis=0)
    return {
        "master_joint": "idx81_gripper_r_outer_joint1",
        "master_open_target_rad": float(joint_state["idx81_gripper_r_outer_joint1"]["q_rad"]),
        "joints": joint_state,
        "pad_links": list(pad_names),
        "pad_positions_world_m": pad_positions.tolist(),
        "pad_orientations_world_wxyz": pad_quaternions.tolist(),
        "pad_midpoint_world_m": midpoint_world.tolist(),
        "pad_midpoint_root_m": (midpoint_world - root).tolist(),
        "pad_to_pad_distance_m": float(np.linalg.norm(pad_positions[1] - pad_positions[0])),
        "ee_root_m": ee_root.tolist(),
        "cube_root_m": cube_root.tolist(),
        "cube_minus_ee_root_m": (cube_root - ee_root).tolist(),
        "ee_cube_distance_m": float(np.linalg.norm(cube_root - ee_root)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return None
    return float(np.dot(left, right) / denominator)


def _fd_metrics(samples: np.ndarray, dt_s: float, indices: np.ndarray) -> dict[str, Any]:
    selected = samples[:, indices]
    velocity = np.diff(selected, axis=0) / dt_s
    acceleration = np.diff(velocity, axis=0) / dt_s
    return {
        "sample_count": int(samples.shape[0]),
        "maximum_fd_velocity_rad_s": float(np.abs(velocity).max(initial=0.0)),
        "maximum_fd_acceleration_rad_s2": float(np.abs(acceleration).max(initial=0.0)),
    }


def _right_distal_pad_midpoint_root_m(env: Any, robot: Any) -> np.ndarray:
    """Read the live right-distal-link midpoint in robot-root coordinates.

    This is a live link-frame clearance measurement, not a soft-pad
    calibration or contact-model authority.  Candidate A differs from
    Production only in passive limits, and this helper introduces no mechanics
    delta.
    """

    body_names = tuple(robot.body_names)
    names = ("gripper_r_inner_link4", "gripper_r_outer_link4")
    missing = [name for name in names if name not in body_names]
    if missing:
        raise RuntimeError(f"PRECONTACT_PAD_LINKS_MISSING:{missing}")
    positions = (
        _tensor(robot.data.body_pos_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    root = (
        _tensor(robot.data.root_pos_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    return np.mean(
        positions[[body_names.index(name) for name in names]], axis=0
    ) - root


def _plan_contact_free_after_reset(
    *,
    env: Any,
    seed: int,
    input_telemetry: dict[str, Any] | None = None,
    input_telemetry_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Bind a planner-only replan to the current post-reset simulator state.

    This function is deliberately called only after the canonical reset and
    OPEN settle lifecycle.  It reads the cube root pose for the *planner
    request* and the current active arm positions for MotionGen.  The return
    value has no actor observation, replay row, gripper target, or action
    packet; the existing Candidate-A executor remains the only code that can
    form and submit a canonical full-8-D packet through ``env.step``.
    """

    from geniesim.rl.isaaclab.g2_policy_branch.contact_free_runtime_replan import (
        ContactFreeRuntimeReplanRequest,
        plan_contact_free_reach,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.curobo_collision_world import (
        SOURCE_RESET_FLOAT32_POSITION_TOLERANCE_M,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.curobo_planner_authority import (
        RIGHT_ARM_7DOF_JOINTS,
    )

    robot = env.scene["robot"]
    joint_names = tuple(str(name) for name in robot.joint_names)
    missing = [name for name in RIGHT_ARM_7DOF_JOINTS if name not in joint_names]
    if missing:
        raise RuntimeError(f"RUNTIME_REPLAN_ACTIVE_ARM_JOINTS_MISSING:{missing}")
    joint_ids = [joint_names.index(name) for name in RIGHT_ARM_7DOF_JOINTS]
    joint_pos = (
        _tensor(robot.data.joint_pos)[0, joint_ids]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    robot_root_world_m = (
        _tensor(robot.data.root_pos_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    robot_root_world_quaternion_wxyz = (
        _tensor(robot.data.root_quat_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    cube_center_world_m = (
        _tensor(env.scene["object"].data.root_pos_w)[0]
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float64)
    )
    cube_root_m = cube_center_world_m - robot_root_world_m
    # Persist the raw post-reset planning receipt before validation.  This is
    # diagnostic/planner provenance only: it is never placed in the actor
    # input, canonical collection, or learning replay.  Recording it before
    # ``validated()`` means a future fail-closed geometry rejection remains
    # attributable (not silently inferred from a historical float32 value).
    if input_telemetry is not None:
        input_telemetry.update(
            {
                "schema": "g2_post_reset_runtime_replan_input_v2",
                "stage": "POST_RESET_PRE_VALIDATION",
                "pre_validation_stage": "POST_RESET_PRE_VALIDATION",
                "planner_validation_completed": False,
                "planner_frame": "robot_root",
                "cube_root_definition": (
                    "cube_center_world_m_minus_robot_root_world_position_m"
                ),
                "robot_root_world_position_m": [
                    float(value) for value in robot_root_world_m
                ],
                "robot_root_world_quaternion_wxyz": [
                    float(value) for value in robot_root_world_quaternion_wxyz
                ],
                "cube_center_world_m": [
                    float(value) for value in cube_center_world_m
                ],
                "cube_center_root_m": [float(value) for value in cube_root_m],
                "right_arm_q_rad": [float(value) for value in joint_pos],
                "seed": int(seed),
                "source_reset_float32_position_tolerance_m": float(
                    SOURCE_RESET_FLOAT32_POSITION_TOLERANCE_M
                ),
                "units": {
                    "robot_root_world_position_m": "m",
                    "cube_center_world_m": "m",
                    "cube_center_root_m": "m",
                    "right_arm_q_rad": "rad",
                    "source_reset_float32_position_tolerance_m": "m",
                },
                "actor_input_contains_cube_gt": False,
                "replay_row_contains_planner_cube_gt": False,
                "action_packet_generated": False,
                "gripper_command_generated": False,
            }
        )
        # This receipt is written before validation and before cuRobo is
        # created.  It preserves the actual float32-origin reset input for a
        # fail-closed planner rejection; it is provenance only, never actor
        # input, replay, controller input, or an action packet.
        if input_telemetry_path is not None:
            _atomic_json(input_telemetry_path, input_telemetry)
    request = ContactFreeRuntimeReplanRequest(
        cube_center_root_m=tuple(float(value) for value in cube_root_m),
        right_arm_q_rad=tuple(float(value) for value in joint_pos),
        seed=int(seed),
    ).validated()
    plan = plan_contact_free_reach(request).validated()
    receipt = plan.planner_only_receipt()
    if not np.allclose(
        np.asarray(receipt["planner_input_cube_center_root_m"], dtype=np.float64),
        cube_root_m,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("RUNTIME_REPLAN_POST_RESET_CUBE_BINDING_MISMATCH")
    if receipt["actor_input_contains_cube_gt"] or receipt[
        "replay_row_contains_planner_cube_gt"
    ]:
        raise RuntimeError("RUNTIME_REPLAN_PRIVILEGED_DATA_LEAK")
    if receipt["gripper_command_generated"] or receipt["action_packet_generated"]:
        raise RuntimeError("RUNTIME_REPLAN_EXECUTION_OWNERSHIP_VIOLATION")
    if input_telemetry is not None:
        input_telemetry.update(
            {
                "stage": "POST_RESET_VALIDATED_PLANNER_RECEIPT",
                "planner_validation_completed": True,
                "planner_receipt_cube_center_root_m": list(
                    receipt["planner_input_cube_center_root_m"]
                ),
            }
        )
    return plan, receipt


def _run_candidate_a_contact_free_smoke(
    *,
    output: Path,
    replay: Path | None,
    runtime_replan_receipt: Mapping[str, Any] | None,
    runtime_replan_input_telemetry: Mapping[str, Any] | None,
    runtime_replan_input_telemetry_path: Path | None,
    seed: int,
    env: Any,
    p0a: Any,
    preflight: Any,
    g2_lift_task_mdp: Any,
    counter: Any,
    deferred_port: Any,
    latch: Any,
    settle: Mapping[str, Any],
    freeze_before: Mapping[str, Any],
    asset_receipt: Any,
    diagnostic_asset_selection: Mapping[str, Any],
    task_contract: Any,
    task_geometry_binding: Mapping[str, Any],
    task_geometry_readback: Mapping[str, Any],
    p0a_provenance: Mapping[str, Any],
    selected_asset: Path,
    selected_hash: str,
    planned_ee: np.ndarray,
    fine_start_index: int,
    ordinary_tracking_cap: int,
    critical_tracking_cap: int,
    cosine_p05: float | None,
    cosine_p10: float | None,
    target_telemetry: Any | None,
    canonical_collection_output: Path | None,
    canonical_capture_rate_hz: int,
    runtime_replan_plan_sidecar_sha256: str | None,
    independent_collection_episode_id: str | None,
    contact_free_bc_playback_checkpoint: Path | None,
    contact_free_bc_playback_checkpoint_sha256: str | None,
    candidate_a_bc_validation_output: Path | None,
    candidate_a_bc_validation_episode_id: str | None,
) -> int:
    """Run exactly the Candidate-A OPEN-only contact-free planner boundary."""

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.curobo_4d_handoff import (
        planner_waypoint_to_canonical_4d,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.precontact_contract import (
        CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
        CONTACT_FREE_EXECUTION_MODE,
        PrecontactContractError,
        PrecontactDistanceContract,
        PrecontactPhase,
        PrecontactProgressMonitor,
        PrecontactReplayTransition,
        gate_precontact_step,
        pad_frame_open_handoff_ee_target,
        segment_minimum_pad_object_distance_m,
        split_open_translation_to_handoff,
        validate_final_metric_action_delta,
        evaluate_contact_free_verdict,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.precontact_tracking_contract import (
        ExecutionSubstate,
        evaluate_waypoint_tracking,
        shape_waypoint_residual,
        summarize_cosine_distribution,
    )
    from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
        G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES,
    )
    import torch
    from geniesim.rl.isaaclab.g2_policy_branch.contact_free_visual_bc_runtime import (
        prepare_runtime_input,
        validate_metric_output,
    )

    playback_enabled = contact_free_bc_playback_checkpoint is not None
    p06_validation_requested = candidate_a_bc_validation_output is not None
    playback_model = None
    playback_checkpoint_receipt = None
    if playback_enabled:
        if contact_free_bc_playback_checkpoint_sha256 is None:
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_CHECKPOINT_SHA256_REQUIRED")
        playback_model, playback_checkpoint_receipt = (
            _load_contact_free_visual_bc_checkpoint(
                contact_free_bc_playback_checkpoint,
                expected_sha256=contact_free_bc_playback_checkpoint_sha256,
                device=env.device,
            )
        )
        if canonical_collection_output is not None or p06_validation_requested:
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_REJECTS_COLLECTION_OUTPUT")
        print("CONTACT_FREE_BC_PLAYBACK_CHECKPOINT_LOADED", flush=True)

    if (replay is None) != (runtime_replan_receipt is not None):
        raise RuntimeError("CANDIDATE_A_CONTACT_FREE_PLAN_PROVENANCE_AMBIGUOUS")
    if runtime_replan_receipt is not None:
        required_runtime_fields = {
            "schema",
            "planner_input_cube_center_root_m",
            "planner_input_right_arm_q_rad",
            "actor_input_contains_cube_gt",
            "replay_row_contains_planner_cube_gt",
            "gripper_command_generated",
            "action_packet_generated",
        }
        if not required_runtime_fields <= set(runtime_replan_receipt):
            raise RuntimeError("RUNTIME_REPLAN_RECEIPT_FIELDS_MISSING")
        if runtime_replan_receipt["schema"] != "g2_contact_free_runtime_replan_v1":
            raise RuntimeError("RUNTIME_REPLAN_RECEIPT_SCHEMA_INVALID")

    runtime_replan_collection = (
        canonical_collection_output is not None and runtime_replan_receipt is not None
    )
    if p06_validation_requested:
        if (
            runtime_replan_receipt is None
            or runtime_replan_plan_sidecar_sha256 is None
            or candidate_a_bc_validation_episode_id is None
        ):
            raise RuntimeError("P06_VALIDATION_REQUIRES_RUNTIME_REPLAN_PROVENANCE")
        if canonical_collection_output is not None:
            raise RuntimeError("P06_VALIDATION_REJECTS_TRAINING_COLLECTION_OUTPUT")
    if runtime_replan_collection:
        if (
            runtime_replan_plan_sidecar_sha256 is None
            or independent_collection_episode_id is None
        ):
            raise RuntimeError("RUNTIME_REPLAN_CANONICAL_COLLECTION_PROVENANCE_MISSING")
    elif not p06_validation_requested and (
        runtime_replan_plan_sidecar_sha256 is not None
        or independent_collection_episode_id is not None
    ):
        raise RuntimeError("RUNTIME_REPLAN_CANONICAL_COLLECTION_PROVENANCE_UNEXPECTED")

    if canonical_collection_output is not None:
        if replay is None and not runtime_replan_collection:
            raise RuntimeError("CANONICAL_COLLECTION_PLAN_PROVENANCE_UNSUPPORTED")
        from geniesim.rl.isaaclab.g2_policy_branch.canonical_contact_free_collection_v2 import (
            ACTOR_FIELDS,
            CanonicalContactFreeRow,
            CanonicalContactFreeStreamingWriter,
            PacketEpochReceipt,
            _hash_packet,
            collection_manifest_v2,
        )
        if runtime_replan_collection:
            from geniesim.rl.isaaclab.g2_policy_branch.independent_contact_free_collection import (
                IndependentCollectionEpisode,
                runtime_replan_collection_manifest_v2,
            )

    if canonical_collection_output is not None or p06_validation_requested:
        from geniesim.rl.isaaclab.g2_policy_branch.dls_read_only_preview_capture import (
            G2DLSReadOnlyPreviewCaptureRequest,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.observation import (
            PolicyDataSemantics,
        )
        from geniesim.rl.isaaclab.g2_lift_env_cfg import SANDBOX

    if p06_validation_requested:
        from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_bc_validation import (
            CandidateAValidationRow,
            CandidateAValidationWriter,
            STUDENT_FIELDS as P06_STUDENT_FIELDS,
            validation_manifest as p06_validation_manifest,
        )

    if canonical_capture_rate_hz != 25:
        raise RuntimeError("CANONICAL_COLLECTION_CAPTURE_RATE_MUST_BE_25_HZ")
    canonical_capture_control_stride = 50 // canonical_capture_rate_hz
    canonical_camera_period_s: float | None = None
    canonical_camera_physics_steps: int | None = None
    if canonical_collection_output is not None or p06_validation_requested:
        canonical_camera_period_s, canonical_camera_physics_steps = (
            _canonical_camera_acquisition_timing(
                physics_dt_s=float(SANDBOX.physics_dt_s),
                capture_rate_hz=canonical_capture_rate_hz,
            )
        )

    if (
        diagnostic_asset_selection.get("variant") != "custom"
        or selected_asset.resolve() != CANDIDATE_A_ASSET.resolve()
        or selected_hash != EXPECTED_CANDIDATE_A_ASSET_SHA256
        or (seed != 42 and not runtime_replan_collection and not p06_validation_requested)
    ):
        raise RuntimeError("CANDIDATE_A_CONTACT_FREE_FROZEN_BINDING_MISMATCH")

    robot = env.scene["robot"]
    arm_term = env.action_manager.get_term("arm_action")
    distance = PrecontactDistanceContract()
    monitor = PrecontactProgressMonitor()
    records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    phase_sequence: list[str] = []
    failure_classification: str | None = None
    failure_detail: str | None = None
    handoff_reached = False
    all_single_consumption = True
    all_open = True
    all_no_contact = True
    all_no_collision = True
    all_exact_4d = True
    all_zero_orientation = True
    follower_direct_target_write_count = 0
    safety_reject_count = 0
    maximum_segment_translation_m = 0.0
    maximum_nominal_planner_increment_m = 0.0
    previous_planner_target: np.ndarray | None = None
    last_tracking_decision: Any | None = None
    catchup_step_count = 0
    total_catchup_step_count = 0
    maximum_catchup_steps_observed = 0
    # The child supervisor already binds the coarse and fine tracking budgets
    # in its immutable command line.  The Candidate-A branch must consume the
    # same authority instead of silently substituting an unrelated fixed cap.
    # A fine/open-handoff segment can require a brief bounded deceleration
    # when the frozen EE-frame plan transitions to the measured pad-frame
    # extension; it still emits at most one 4.5-mm packet per policy step and
    # cannot advance the waypoint until the measured residual is normal.
    if ordinary_tracking_cap <= 0 or critical_tracking_cap < ordinary_tracking_cap:
        raise RuntimeError("CANDIDATE_A_CONTACT_FREE_TRACKING_BUDGET_INVALID")
    minimum_pre_submit_clearance_m = float("inf")
    previous_qdot: np.ndarray | None = None
    previous_actor_arm_qd: torch.Tensor | None = None
    playback_inference_count = 0
    playback_action_norms_m: list[float] = []
    maximum_active_qdot_rad_s = 0.0
    maximum_passive_qdot_rad_s = 0.0
    maximum_active_qdd_rad_s2 = 0.0
    maximum_passive_qdd_rad_s2 = 0.0
    limiter_failure_snapshot: list[dict[str, Any]] = []
    pad_frame_handoff_extension_used = False
    pad_frame_handoff_extension_segment_count = 0
    canonical_writer: Any | None = None
    canonical_row_count = 0
    previous_canonical_action = (0.0, 0.0, 0.0, 0.0)
    canonical_capture_enabled = False
    canonical_capture_disable_error: str | None = None
    p06_writer: Any | None = None
    p06_row_count = 0
    p06_phase_counts = {"far_reach": 0, "pregrasp": 0, "near_contact": 0}
    # PREGRASP is a single source-owned boundary waypoint in the current
    # planner; do not manufacture repeated labels merely to balance counts.
    # Near-contact receives the larger minimum because it is the authority-
    # sensitive region this validation is intended to measure.
    p06_minimum_phase_rows = {"far_reach": 32, "pregrasp": 1, "near_contact": 32}
    p06_bounded_capture_complete = False
    p06_cached_rgbd: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        float,
        float,
        dict[str, float | int],
    ] | None = None
    p06_publication: dict[str, Any] = {
        "requested": p06_validation_requested,
        "state": "NOT_REQUESTED",
        "path": (
            str(candidate_a_bc_validation_output.resolve())
            if candidate_a_bc_validation_output is not None
            else None
        ),
        "row_count": 0,
        "error": None,
    }
    canonical_collection_receipt: dict[str, Any] = {
        "requested": canonical_collection_output is not None,
        "provenance": (
            "POST_RESET_RUNTIME_PLAN_SIDECAR"
            if runtime_replan_collection
            else "FROZEN_REPLAY"
            if canonical_collection_output is not None
            else None
        ),
        "independent_episode_id": (
            independent_collection_episode_id if runtime_replan_collection else None
        ),
        "runtime_plan_sidecar_sha256": (
            runtime_replan_plan_sidecar_sha256 if runtime_replan_collection else None
        ),
        "destination": (
            str(canonical_collection_output.resolve())
            if canonical_collection_output is not None
            else None
        ),
        "state": "NOT_REQUESTED",
        "row_count": 0,
        "camera_frame_ids": [],
        "camera_frame_ids_by_camera": {
            camera_name: [] for camera_name in RGBD_CAMERA_NAMES
        },
        "camera_timestamps_s_by_camera": {
            camera_name: [] for camera_name in RGBD_CAMERA_NAMES
        },
        "camera_update_period_s": None,
        "capture_rate_hz": canonical_capture_rate_hz if canonical_collection_output is not None else None,
        "control_rate_hz": 50 if canonical_collection_output is not None else None,
        "capture_control_step_stride": canonical_capture_control_stride if canonical_collection_output is not None else None,
        "error": None,
        "partial_safe_destination": (
            str(canonical_collection_output.with_name("PARTIAL_SAFE_ROWS.hdf5").resolve())
            if canonical_collection_output is not None
            else None
        ),
    }
    if canonical_collection_output is not None:
        if canonical_collection_output.exists():
            raise RuntimeError("CANONICAL_COLLECTION_REFUSES_EXISTING_DESTINATION")
        if env.num_envs != 1:
            raise RuntimeError("CANONICAL_COLLECTION_REQUIRES_ONE_ENVIRONMENT")
        if runtime_replan_collection:
            canonical_manifest = runtime_replan_collection_manifest_v2(
                source_sha256=_sha256(Path(__file__).resolve()),
                asset_sha256=selected_hash,
                runtime_plan_sidecar_sha256=str(runtime_replan_plan_sidecar_sha256),
                episode=IndependentCollectionEpisode(
                    episode_id=str(independent_collection_episode_id), seed=int(seed)
                ),
            )
        else:
            canonical_manifest = collection_manifest_v2(
                source_sha256=_sha256(Path(__file__).resolve()),
                asset_sha256=selected_hash,
                trajectory_sha256=_sha256(replay),
                snapshot_hook_enabled=True,
            )
        camera_calibrations = _camera_calibration_snapshot(env)
        canonical_manifest["camera_evidence_contract"] = {
            "active": True,
            "schema": RGBD_EVIDENCE_SCHEMA,
            "camera_names": list(RGBD_CAMERA_NAMES),
            "depth_raw_unit": DEPTH_RAW_UNIT,
            "depth_to_meter_scale": DEPTH_TO_METER_SCALE,
            "raw_depth_semantics": (
                "UNCLAMPED_METRIC_WITH_POSITIVE_INFINITY_PRESERVED_AND_SEPARATE_VALID_MASK"
            ),
            "current_actor_inputs_changed": False,
            "calibration": flatten_calibration_metadata(camera_calibrations),
        }
        canonical_manifest["head_and_right_wrist_camera_runtime_contract"] = {
            "resolution_px": [256, 192],
            "update_period_s": canonical_camera_period_s,
            "capture_interval_physics_steps": canonical_camera_physics_steps,
            "physics_dt_s": float(SANDBOX.physics_dt_s),
            "policy_sampling": (
                "latest source-owned RGB-D frame associated to 50 Hz control; "
                "dataset evidence captured at 25 Hz"
            ),
        }
        canonical_manifest["dataset_capture_contract"] = {
            "capture_rate_hz": int(canonical_capture_rate_hz),
            "control_rate_hz": 50,
            "capture_control_step_stride": int(canonical_capture_control_stride),
            "row_semantics": "one actor row per captured control epoch; control remains 50 Hz",
            "timestamp_semantics": "original 50 Hz control epoch seconds",
        }
        canonical_writer = CanonicalContactFreeStreamingWriter(
            canonical_collection_output, manifest=canonical_manifest
        )
        arm_term.set_read_only_dls_preview_capture_enabled(True)
        canonical_capture_enabled = True
        canonical_collection_receipt["state"] = "CAPTURING"
    elif p06_validation_requested:
        if candidate_a_bc_validation_output is None:
            raise RuntimeError("P06_VALIDATION_DESTINATION_MISSING")
        if candidate_a_bc_validation_output.exists():
            raise RuntimeError("P06_VALIDATION_REFUSES_EXISTING_DESTINATION")
        if env.num_envs != 1:
            raise RuntimeError("P06_VALIDATION_REQUIRES_ONE_ENVIRONMENT")
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("P06_GIT_COMMIT_UNAVAILABLE") from error
        p06_manifest = p06_validation_manifest(
            episode_id=str(candidate_a_bc_validation_episode_id),
            asset_fingerprint=selected_hash,
            planner_fingerprint=str(runtime_replan_plan_sidecar_sha256),
            git_commit=git_commit,
            runner_source_sha256=_sha256(Path(__file__).resolve()),
            source_freeze_fingerprint=str(freeze_before["manifest_sha256"]),
        )
        if p06_manifest["student_fields"] != list(P06_STUDENT_FIELDS):
            raise RuntimeError("P06_STUDENT_INVENTORY_MISMATCH")
        p06_writer = CandidateAValidationWriter(
            candidate_a_bc_validation_output,
            manifest=p06_manifest,
        )
        arm_term.set_read_only_dls_preview_capture_enabled(True)
        canonical_capture_enabled = True
        p06_publication["state"] = "CAPTURING"
    arm_indices = np.asarray(
        env.action_manager.get_term("arm_action")._joint_ids, dtype=np.int64
    )
    passive_indices = np.asarray(
        [
            robot.joint_names.index(name)
            for name in G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES
            if name in robot.joint_names
        ],
        dtype=np.int64,
    )

    def _state() -> dict[str, Any]:
        pad = _right_distal_pad_midpoint_root_m(env, robot)
        cube = _cube_root_position(env)
        ee = _ee_root_position(env)
        gripper = p0a._gripper_command_telemetry(
            env.action_manager.get_term("gripper_action")
        )
        task = preflight._task_metrics(env, g2_lift_task_mdp)
        evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
        sensor_peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
        forbidden_peak = float(
            max((max(values) for values in sensor_peaks.values()), default=0.0)
        )
        physical_contact = bool(
            float(task["inner_contact_force_n"]) > 0.0
            or float(task["outer_contact_force_n"]) > 0.0
            or bool(task["bilateral_contact"])
            or bool(task["ever_bilateral_contact"])
        )
        return {
            "pad_midpoint_root_m": pad,
            "cube_root_m": cube,
            "ee_root_m": ee,
            "pad_object_distance_m": float(np.linalg.norm(cube - pad)),
            "ee_object_distance_m": float(np.linalg.norm(cube - ee)),
            "gripper": gripper,
            "gripper_is_open": not bool(gripper["close_command_active"]),
            "task": task,
            "physical_contact": physical_contact,
            "forbidden_contact_sensor_peak_n": sensor_peaks,
            "forbidden_contact_peak_n": forbidden_peak,
            "joint_qdot_rad_s": _tensor(robot.data.joint_vel)[0]
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64),
        }

    def _capture_canonical_wrist_rgbd() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, int, float, float, dict[str, float | int]
    ]:
        """Clone the source-owned wrist stream before the sole env.step.

        This intentionally has no GT, task phase, cube, or future-state
        output.  The collection row is permitted to carry only actor-facing
        RGB-D plus the DLS receipt that will bind the *same* packet after
        normal ActionManager processing.
        """

        if canonical_collection_output is None and not p06_validation_requested:
            raise RuntimeError("CANONICAL_COLLECTION_CAPTURE_NOT_REQUESTED")
        camera = env.scene["right_wrist_camera"]
        output = camera.data.output
        if "rgb" not in output or "distance_to_image_plane" not in output:
            raise RuntimeError("CANONICAL_COLLECTION_WRIST_RGBD_STREAM_MISSING")
        rgb = _tensor(output["rgb"])[..., :3]
        raw_depth = _tensor(output["distance_to_image_plane"])
        if raw_depth.ndim == 3:
            raw_depth = raw_depth.unsqueeze(-1)
        if tuple(rgb.shape) != (1, 192, 256, 3) or rgb.dtype != torch.uint8:
            raise RuntimeError("CANONICAL_COLLECTION_RGB_RUNTIME_CONTRACT_MISMATCH")
        if tuple(raw_depth.shape) != (1, 192, 256, 1):
            raise RuntimeError("CANONICAL_COLLECTION_DEPTH_RUNTIME_SHAPE_MISMATCH")
        update_period_s = float(camera.cfg.update_period)
        if canonical_camera_period_s is None:
            raise RuntimeError("CANONICAL_COLLECTION_CAMERA_PERIOD_NOT_BOUND")
        if not math.isclose(
            update_period_s,
            canonical_camera_period_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("CANONICAL_COLLECTION_CAMERA_SOURCE_CADENCE_MISMATCH")
        raw_depth = raw_depth.to(dtype=torch.float32)
        maximum_depth_m = float(PolicyDataSemantics().maximum_depth_m)
        # The policy dataset adapter owns the source-defined far-depth rule:
        # finite pixels beyond 2 m are *invalid*, not a simulator corruption.
        # Treating the rendered far background as a hard error would make the
        # collection contract disagree with the actual BC input pipeline.
        corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (
            torch.isfinite(raw_depth) & (raw_depth < 0.0)
        )
        if bool(corrupt.any().item()):
            raise RuntimeError("CANONICAL_COLLECTION_DEPTH_CORRUPT_OR_OUT_OF_RANGE")
        source_valid = torch.isfinite(raw_depth) & (raw_depth >= 0.0)
        valid = source_valid & (raw_depth <= maximum_depth_m)
        depth_m = torch.where(valid, raw_depth, torch.zeros_like(raw_depth))
        frame = _tensor(camera.frame).reshape(-1)
        if frame.numel() != 1:
            raise RuntimeError("CANONICAL_COLLECTION_CAMERA_FRAME_CARDINALITY_MISMATCH")
        from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age

        captured, _age = camera_capture_time_and_age(camera)
        captured_tensor = _tensor(captured).reshape(-1)
        if captured_tensor.numel() != 1:
            raise RuntimeError("CANONICAL_COLLECTION_CAMERA_TIMESTAMP_CARDINALITY_MISMATCH")
        camera_timestamp_s = float(captured_tensor[0].item())
        if not math.isfinite(camera_timestamp_s) or camera_timestamp_s < 0.0:
            raise RuntimeError("CANONICAL_COLLECTION_CAMERA_TIMESTAMP_INVALID")
        return (
            rgb[0].detach().to("cpu").numpy().astype(np.uint8, copy=True),
            depth_m[0].detach().to("cpu").numpy().astype(np.float32, copy=True),
            valid[0].detach().to("cpu").numpy().astype(np.bool_, copy=True),
            int(frame[0].item()),
            update_period_s,
            camera_timestamp_s,
            {
                "maximum_depth_m": maximum_depth_m,
                "source_positive_fraction": float(
                    source_valid.to(torch.float32).mean().item()
                ),
                "far_finite_masked_fraction": float(
                    (source_valid & ~valid).to(torch.float32).mean().item()
                ),
                "policy_valid_fraction": float(
                    valid.to(torch.float32).mean().item()
                ),
                "raw_finite_max_m": float(
                    raw_depth[torch.isfinite(raw_depth)].max().item()
                ),
            },
        )

    def _capture_canonical_dual_rgbd_evidence() -> tuple[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            float,
            float,
            dict[str, float | int],
        ],
        dict[str, Any],
    ]:
        """Capture Head+Wrist raw evidence without changing actor inputs."""

        legacy_wrist = _capture_canonical_wrist_rgbd()
        camera_rows: dict[str, Any] = {}
        for camera_name in RGBD_CAMERA_NAMES:
            camera = env.scene[f"{camera_name}_camera"]
            output = camera.data.output
            if "rgb" not in output or "distance_to_image_plane" not in output:
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_RGBD_STREAM_MISSING"
                )
            rgb = _tensor(output["rgb"])[..., :3]
            raw_depth = _tensor(output["distance_to_image_plane"])
            if raw_depth.ndim == 3:
                raw_depth = raw_depth.unsqueeze(-1)
            if tuple(rgb.shape) != (1, 192, 256, 3) or rgb.dtype != torch.uint8:
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_RGB_CONTRACT_MISMATCH"
                )
            if tuple(raw_depth.shape) != (1, 192, 256, 1):
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_DEPTH_SHAPE_MISMATCH"
                )
            raw_depth = raw_depth.to(dtype=torch.float32)
            corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (
                torch.isfinite(raw_depth) & (raw_depth < 0.0)
            )
            if bool(corrupt.any().item()):
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_RAW_DEPTH_CORRUPT"
                )
            source_valid = torch.isfinite(raw_depth) & (raw_depth >= 0.0)
            frame_tensor = _tensor(camera.frame).reshape(-1)
            if frame_tensor.numel() != 1:
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_FRAME_CARDINALITY_MISMATCH"
                )
            from geniesim.rl.isaaclab.g2_camera_timing import (
                camera_capture_time_and_age,
            )

            captured, _age = camera_capture_time_and_age(camera)
            captured_tensor = _tensor(captured).reshape(-1)
            if captured_tensor.numel() != 1:
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_TIMESTAMP_CARDINALITY_MISMATCH"
                )
            timestamp_s = float(captured_tensor[0].item())
            frame_id = int(frame_tensor[0].item())
            if not math.isfinite(timestamp_s) or timestamp_s < 0.0 or frame_id < 0:
                raise RuntimeError(
                    f"CANONICAL_COLLECTION_{camera_name.upper()}_FRAME_TIME_INVALID"
                )
            camera_rows[camera_name] = {
                "rgb": rgb[0].detach().to("cpu").numpy().astype(np.uint8, copy=True),
                "depth_raw_m": raw_depth[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float32, copy=True),
                "depth_source_valid": source_valid[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.bool_, copy=True),
                "timestamp_s": timestamp_s,
                "frame_id": frame_id,
                "associated_frame_index": frame_id,
                "pose_root_m_xyzw": _camera_pose_root_m_xyzw(
                    env, camera_name
                ),
            }
        if (
            camera_rows["right_wrist"]["frame_id"] != legacy_wrist[3]
            or camera_rows["right_wrist"]["timestamp_s"] != legacy_wrist[5]
            or not np.array_equal(
                camera_rows["right_wrist"]["rgb"], legacy_wrist[0]
            )
        ):
            raise RuntimeError("CANONICAL_COLLECTION_WRIST_EVIDENCE_NOT_ATOMIC")
        evidence = {
            "schema": RGBD_EVIDENCE_SCHEMA,
            "depth_raw_unit": DEPTH_RAW_UNIT,
            "depth_to_meter_scale": DEPTH_TO_METER_SCALE,
            **camera_rows,
        }
        return legacy_wrist, evidence

    def _contact_free_bc_metric_action() -> np.ndarray:
        nonlocal previous_actor_arm_qd, playback_inference_count
        if playback_model is None:
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_MODEL_NOT_LOADED")
        camera = env.scene["right_wrist_camera"]
        output = camera.data.output
        if "rgb" not in output or "distance_to_image_plane" not in output:
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_WRIST_RGBD_MISSING")
        rgb = _tensor(output["rgb"])[..., :3]
        raw_depth = _tensor(output["distance_to_image_plane"])
        if raw_depth.ndim == 3:
            raw_depth = raw_depth.unsqueeze(-1)
        if tuple(rgb.shape) != (1, 192, 256, 3) or rgb.dtype != torch.uint8:
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_RGB_CONTRACT_MISMATCH")
        if tuple(raw_depth.shape) != (1, 192, 256, 1):
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_DEPTH_SHAPE_MISMATCH")
        raw_depth = raw_depth.to(dtype=torch.float32)
        corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (
            torch.isfinite(raw_depth) & (raw_depth < 0.0)
        )
        if bool(corrupt.any().item()):
            raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_DEPTH_CORRUPT")
        depth_valid = (
            torch.isfinite(raw_depth) & (raw_depth > 0.0) & (raw_depth <= 2.0)
        )
        depth_m = torch.where(depth_valid, raw_depth, torch.zeros_like(raw_depth))
        ee_position, ee_quaternion = _ee_root_pose(env)
        ee_pose = torch.tensor(
            [[*ee_position.tolist(), *ee_quaternion.tolist()]],
            device=env.device,
            dtype=torch.float32,
        )
        arm_index_tensor = torch.as_tensor(
            arm_indices, device=env.device, dtype=torch.long
        )
        q = _tensor(robot.data.joint_pos).index_select(1, arm_index_tensor).to(torch.float32)
        qd = _tensor(robot.data.joint_vel).index_select(1, arm_index_tensor).to(torch.float32)
        qdd = (
            torch.zeros_like(qd)
            if previous_actor_arm_qd is None
            else (qd - previous_actor_arm_qd) / float(env.step_dt)
        )
        previous_actor_arm_qd = qd.detach().clone()
        image, proprio = prepare_runtime_input(
            right_wrist_rgb=rgb,
            right_wrist_depth_m=depth_m,
            right_wrist_depth_valid=depth_valid,
            ee_pose_robot_root_m_xyzw=ee_pose,
            right_arm_joint_position_rad=q,
            right_arm_joint_velocity_rad_s=qd,
            right_arm_joint_acceleration_rad_s2=qdd,
            gripper_state_open=torch.ones((1, 1), device=env.device),
            previous_policy_action_4d_metric_root_m=torch.tensor(
                [previous_canonical_action], device=env.device, dtype=torch.float32
            ),
        )
        with torch.inference_mode():
            action = playback_model(image, proprio)
        validate_metric_output(action)
        playback_inference_count += 1
        metric = action[0, :3].detach().to("cpu").numpy().astype(np.float64)
        playback_action_norms_m.append(float(np.linalg.norm(metric)))
        return metric

    def _append_phase(phase: PrecontactPhase) -> None:
        if not phase_sequence or phase_sequence[-1] != phase.value:
            phase_sequence.append(phase.value)

    def _failure_for_receipt(receipt: Any) -> str:
        if receipt.termination_reason == "CONTACT_EVENT_FAIL_CLOSED":
            return "UNEXPECTED_CONTACT"
        if receipt.termination_reason in {
            "CONTACT_GUARD_DISTANCE_CROSSED",
            "PLANNED_SEGMENT_DISTANCE_RECEIPT_MISSING",
        }:
            return "CLEARANCE_GUARD_FAILURE"
        if receipt.termination_reason == "GRIPPER_NOT_OPEN":
            return "ACTION_CONTRACT_FAILURE"
        return "PHASE_TRANSITION_FAILURE"

    def _forward(
        *,
        target: np.ndarray,
        phase: PrecontactPhase,
        label: str,
        force_zero_playback: bool = False,
    ) -> bool:
        nonlocal failure_classification, failure_detail, all_single_consumption
        nonlocal all_open, all_no_contact, all_no_collision, all_exact_4d
        nonlocal all_zero_orientation
        nonlocal safety_reject_count, maximum_segment_translation_m
        nonlocal maximum_nominal_planner_increment_m, previous_planner_target
        nonlocal last_tracking_decision, catchup_step_count
        nonlocal total_catchup_step_count, maximum_catchup_steps_observed
        nonlocal minimum_pre_submit_clearance_m, previous_qdot
        nonlocal maximum_active_qdot_rad_s, maximum_passive_qdot_rad_s
        nonlocal maximum_active_qdd_rad_s2, maximum_passive_qdd_rad_s2
        nonlocal limiter_failure_snapshot
        nonlocal previous_canonical_action
        nonlocal canonical_row_count
        nonlocal p06_row_count
        nonlocal p06_cached_rgbd
        nonlocal p06_bounded_capture_complete
        before = _state()
        tracking_anchor = (
            before["ee_root_m"]
            if previous_planner_target is None
            else previous_planner_target
        )
        tracking_decision = evaluate_waypoint_tracking(
            p_prev=tracking_anchor,
            p_target=target,
            p_measured=before["ee_root_m"],
        )
        last_tracking_decision = tracking_decision
        shape_decision = None
        shaped_command = np.asarray(tracking_decision.bounded_command, dtype=np.float64)
        if playback_enabled:
            shaped_command = (
                np.zeros(3, dtype=np.float64)
                if force_zero_playback
                else _contact_free_bc_metric_action()
            )
        if cosine_p05 is not None or cosine_p10 is not None:
            if playback_enabled:
                raise RuntimeError("CONTACT_FREE_BC_PLAYBACK_REJECTS_COSINE_PLANNER_SHAPING")
            if cosine_p05 is None or cosine_p10 is None:
                raise RuntimeError("COSINE_SHAPING_REQUIRES_P05_AND_P10")
            if tracking_decision.planner_increment_norm_m <= 1.0e-12:
                # The initial retained waypoint can equal the current planner
                # anchor exactly.  It has no direction to shape, so consume it
                # with an explicit neutral packet.  General zero-direction
                # shaping remains fail-closed in the pure contract.
                shaped_command = np.zeros(3, dtype=np.float64)
            else:
                shape_decision = shape_waypoint_residual(
                    p_prev=tracking_anchor,
                    p_target=target,
                    p_measured=before["ee_root_m"],
                    p05=cosine_p05,
                    p10=cosine_p10,
                )
                if not shape_decision.allowed:
                    failure_classification = "ACTION_CONTRACT_FAILURE"
                    failure_detail = "COSINE_DIRECTIONAL_REEVALUATION"
                    safety_reject_count += 1
                    return False
                shaped_command = np.asarray(shape_decision.final_bounded_command, dtype=np.float64)
        if tracking_decision.substate is ExecutionSubstate.TRACKING_CATCHUP:
            catchup_step_count += 1
            total_catchup_step_count += 1
            maximum_catchup_steps_observed = max(
                maximum_catchup_steps_observed, catchup_step_count
            )
        elif tracking_decision.substate is ExecutionSubstate.DIRECTIONAL_REEVALUATION:
            catchup_step_count = 0
        else:
            catchup_step_count = 0
        maximum_catchup_steps_per_waypoint = (
            critical_tracking_cap
            if phase is PrecontactPhase.FINE_APPROACH
            else ordinary_tracking_cap
        )
        if catchup_step_count > maximum_catchup_steps_per_waypoint:
            failure_classification = "ACTION_CONTRACT_FAILURE"
            failure_detail = "TRACKING_CATCHUP_BUDGET_EXCEEDED"
            safety_reject_count += 1
            return False
        bounded_target = before["ee_root_m"] + np.asarray(
            shaped_command, dtype=np.float64
        )
        predicted_end_pad = before["pad_midpoint_root_m"] + (
            bounded_target - before["ee_root_m"]
        )
        planned_minimum = segment_minimum_pad_object_distance_m(
            start_pad_root_m=before["pad_midpoint_root_m"],
            end_pad_root_m=predicted_end_pad,
            object_root_m=before["cube_root_m"],
        )
        minimum_pre_submit_clearance_m = min(
            minimum_pre_submit_clearance_m, planned_minimum
        )
        # Validate the metric residual before the handoff performs inverse
        # normalization.  This is the authoritative contact-free boundary;
        # the post-handoff check below is an immutable-receipt consistency
        # assertion.
        try:
            validate_final_metric_action_delta(bounded_target - before["ee_root_m"])
        except PrecontactContractError as error:
            failure_classification = "ACTION_CONTRACT_FAILURE"
            failure_detail = f"FINAL_METRIC_CONTRACT_REJECT:{error}"
            safety_reject_count += 1
            return False
        handoff = planner_waypoint_to_canonical_4d(
            planner_target_root_m=bounded_target,
            measured_ee_root_m=before["ee_root_m"],
        )
        # The planner grid bounds target-to-target distance, while this
        # measured-feedback residual is the actual metric command that will be
        # normalized and submitted.  Tracking lag must never silently enlarge
        # the contact-free 4.5 mm contract.
        try:
            final_metric_action_m = validate_final_metric_action_delta(
                handoff.metric_delta_root_m
            )
        except PrecontactContractError as error:
            failure_classification = "ACTION_CONTRACT_FAILURE"
            failure_detail = f"FINAL_METRIC_RECEIPT_CONTRACT_REJECT:{error}"
            safety_reject_count += 1
            return False
        maximum_segment_translation_m = max(
            maximum_segment_translation_m, final_metric_action_m
        )
        nominal_increment_m = (
            None
            if previous_planner_target is None
            else float(np.linalg.norm(target - previous_planner_target))
        )
        if nominal_increment_m is not None:
            maximum_nominal_planner_increment_m = max(
                maximum_nominal_planner_increment_m, nominal_increment_m
            )
        action_4d = tuple(float(value) for value in handoff.normalized_action)
        pre_gate = gate_precontact_step(
            phase=phase,
            action_4d=action_4d,
            observed_pad_object_distance_m=float(before["pad_object_distance_m"]),
            planned_segment_minimum_pad_object_distance_m=planned_minimum,
            contact_detected=bool(before["physical_contact"]),
            bilateral_contact_detected=bool(before["task"]["bilateral_contact"]),
            gripper_is_open=bool(before["gripper_is_open"]),
            distance=distance,
        )
        if not pre_gate.accepted:
            safety_reject_count += 1
            failure_classification = _failure_for_receipt(pre_gate)
            failure_detail = str(pre_gate.termination_reason)
            return False
        packet, _, derivation = p0a._build_authoritative_packet(
            high_level=handoff.policy_action,
            batch_size=env.num_envs,
            device=env.device,
            latch=latch,
        )
        if (
            len(action_4d) != 4
            or action_4d[3] != 0.0
            or tuple(float(value) for value in packet.values[3:7])
            != (0.0, 0.0, 0.0, 0.0)
            or packet.gripper_intent is not AbstractGripperIntent.OPEN
        ):
            failure_classification = "ACTION_CONTRACT_FAILURE"
            failure_detail = "OPEN_4D_OR_ZERO_ORIENTATION_CONTRACT_FAILED"
            return False
        collection_rgbd: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            float,
            float,
            dict[str, float | int],
        ] | None = None
        collection_camera_evidence: dict[str, Any] | None = None
        collection_bundle: Any | None = None
        collection_request: Any | None = None
        control_epoch = len(records)
        capture_this_step = (
            p06_validation_requested
            or (
                canonical_collection_output is not None
                and control_epoch % canonical_capture_control_stride == 0
            )
        )
        # The read-only controller receipt must still be staged/consumed for
        # every 50 Hz control epoch.  Dataset capture is independently
        # decimated to 25 Hz; decimation must never create a second action
        # path or leave the hook without an epoch request.
        canonical_hook_this_step = (
            canonical_collection_output is not None or p06_validation_requested
        )
        collection_epoch = control_epoch
        if canonical_hook_this_step:
            if capture_this_step:
                if p06_validation_requested:
                    if collection_epoch % 2 == 0:
                        p06_cached_rgbd = _capture_canonical_wrist_rgbd()
                    if p06_cached_rgbd is None:
                        raise RuntimeError("P06_25HZ_ACQUISITION_LATCH_EMPTY")
                    collection_rgbd = p06_cached_rgbd
                else:
                    (
                        collection_rgbd,
                        collection_camera_evidence,
                    ) = _capture_canonical_dual_rgbd_evidence()
            metric_tensor = torch.tensor(
                [[*tuple(float(value) for value in handoff.metric_delta_root_m), 0.0]],
                dtype=torch.float32,
                device=env.device,
            )
            full8_tensor = torch.tensor(
                [tuple(float(value) for value in packet.values)],
                dtype=torch.float32,
                device=env.device,
            )
            collection_request = G2DLSReadOnlyPreviewCaptureRequest.from_tensors(
                control_epoch=collection_epoch,
                metric_action_4d_root_m=metric_tensor,
                full_action_packet_8d=full8_tensor,
            )
            arm_term.stage_read_only_dls_preview_capture(collection_request)
        try:
            outputs, consumption = preflight._consume_once(
                env=env,
                counter=counter,
                deferred_port=deferred_port,
                packet=packet,
                label=label,
            )
            if canonical_hook_this_step:
                collection_bundle = arm_term.take_read_only_dls_preview_capture_bundle(
                    control_epoch=collection_epoch
                )
                following = collection_bundle.following_controller_telemetry
                capture = collection_bundle.receipt
                canonical_collection_receipt["last_hook_diagnostic"] = {
                    "control_epoch": capture.control_epoch,
                    "selected_packet_hash": capture.selected_packet_hash,
                    "request_packet_hash": (
                        collection_request.selected_packet_hash
                        if collection_request is not None
                        else None
                    ),
                    "process_capture_count": collection_bundle.process_capture_count,
                    "normal_apply_count": collection_bundle.normal_apply_count,
                    "normal_target_matches_preview_exact_float32": (
                        following.source_defined_exact_float32_correspondence
                    ),
                    "normal_target_unit": "rad",
                    "normal_target_dtype": "float32",
                }
                if (
                    collection_request is None
                    or capture.control_epoch != collection_epoch
                    or capture.selected_packet_hash
                    != collection_request.selected_packet_hash
                    or following.selected_packet_hash != capture.selected_packet_hash
                    or collection_bundle.process_capture_count != 1
                    or collection_bundle.normal_apply_count < 1
                ):
                    raise RuntimeError("CANONICAL_COLLECTION_READ_ONLY_RECEIPT_INVALID")
        except BaseException as error:
            if target_telemetry is not None:
                limiter_failure_snapshot = target_telemetry.failure_snapshot()
            failure_classification = "ACTION_CONTRACT_FAILURE"
            failure_detail = (
                f"CONTROLLER_OR_CANONICAL_INGRESS_REJECT:{type(error).__name__}:{error}"
            )
            return False
        _, _reward, terminated, truncated, _info = outputs
        latch.commit(packet.gripper_intent)
        after = _state()
        qdot_after = after["joint_qdot_rad_s"]
        maximum_active_qdot_rad_s = max(
            maximum_active_qdot_rad_s,
            float(np.max(np.abs(qdot_after[arm_indices]), initial=0.0)),
        )
        maximum_passive_qdot_rad_s = max(
            maximum_passive_qdot_rad_s,
            float(np.max(np.abs(qdot_after[passive_indices]), initial=0.0))
            if passive_indices.size
            else 0.0,
        )
        if previous_qdot is not None:
            qdd_after = (qdot_after - previous_qdot) / float(env.step_dt)
            maximum_active_qdd_rad_s2 = max(
                maximum_active_qdd_rad_s2,
                float(np.max(np.abs(qdd_after[arm_indices]), initial=0.0)),
            )
            maximum_passive_qdd_rad_s2 = max(
                maximum_passive_qdd_rad_s2,
                float(np.max(np.abs(qdd_after[passive_indices]), initial=0.0))
                if passive_indices.size
                else 0.0,
            )
        previous_qdot = qdot_after.copy()
        active_terminations = preflight._active_termination_names(
            env, terminated, truncated
        )
        post_gate = gate_precontact_step(
            phase=phase,
            action_4d=action_4d,
            observed_pad_object_distance_m=float(after["pad_object_distance_m"]),
            planned_segment_minimum_pad_object_distance_m=float(
                after["pad_object_distance_m"]
            ),
            contact_detected=bool(after["physical_contact"]),
            bilateral_contact_detected=bool(after["task"]["bilateral_contact"]),
            gripper_is_open=bool(after["gripper_is_open"]),
            distance=distance,
        )
        all_single_consumption &= bool(consumption["single_consumption"])
        all_open &= bool(after["gripper_is_open"])
        all_no_contact &= not bool(after["physical_contact"])
        all_no_collision &= float(after["forbidden_contact_peak_n"]) == 0.0
        all_exact_4d &= len(action_4d) == 4 and action_4d[3] == 0.0
        all_zero_orientation &= tuple(float(value) for value in packet.values[3:7]) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        progress = monitor.observe(
            phase=phase,
            distance_m=float(after["pad_object_distance_m"]),
            contact_latched=bool(after["physical_contact"]),
        )
        record = {
            "step": len(records),
            "action_source": (
                "CONTACT_FREE_VISUAL_BC_METRIC_4D"
                if playback_enabled and not force_zero_playback
                else "FORCED_ZERO_HANDOFF_HEARTBEAT"
                if playback_enabled
                else "CUROBO_PLANNER_NOMINAL_4D"
            ),
            "label": label,
            "phase": phase.value,
            "planner_target_root_m": target.astype(float).tolist(),
            "bounded_target_root_m": bounded_target.astype(float).tolist(),
            "nominal_planner_increment_m": nominal_increment_m,
            "raw_residual_m": list(tracking_decision.raw_residual),
            "raw_residual_norm_m": tracking_decision.raw_residual_norm_m,
            "planner_direction_m": list(tracking_decision.planner_direction),
            "planner_increment_norm_m": tracking_decision.planner_increment_norm_m,
            "cosine_similarity": tracking_decision.cosine_similarity,
            "bounded_command_m": list(tracking_decision.bounded_command),
            "bounded_command_norm_m": tracking_decision.bounded_command_norm_m,
            "r_parallel": (
                list(shape_decision.parallel_residual)
                if shape_decision is not None else None
            ),
            "r_parallel_norm_m": (
                shape_decision.parallel_norm_m if shape_decision is not None else None
            ),
            "r_perp": (
                list(shape_decision.perpendicular_residual)
                if shape_decision is not None else None
            ),
            "r_perp_norm_m": (
                shape_decision.perpendicular_norm_m if shape_decision is not None else None
            ),
            "lateral_ratio": (
                shape_decision.lateral_ratio if shape_decision is not None else None
            ),
            "alpha": shape_decision.alpha if shape_decision is not None else None,
            "beta": shape_decision.beta if shape_decision is not None else None,
            "shaped_command_pre_bound": (
                list(shape_decision.shaped_command_pre_bound)
                if shape_decision is not None else None
            ),
            "shaped_command_pre_bound_norm_m": (
                shape_decision.shaped_command_pre_bound_norm_m
                if shape_decision is not None else None
            ),
            "final_bounded_command_m": shaped_command.tolist(),
            "final_bounded_command_norm_m": float(np.linalg.norm(shaped_command)),
            "shaped_remaining_residual_m": (
                list(shape_decision.remaining_residual)
                if shape_decision is not None else None
            ),
            "remaining_residual_m": list(tracking_decision.remaining_residual),
            "execution_substate": tracking_decision.substate.value,
            "waypoint_consumed": tracking_decision.waypoint_advance_allowed,
            "catchup_count": catchup_step_count,
            "catchup_cap": maximum_catchup_steps_per_waypoint,
            "observed_before": {
                "ee_root_m": before["ee_root_m"].astype(float).tolist(),
                "pad_midpoint_root_m": before["pad_midpoint_root_m"].astype(float).tolist(),
                "cube_root_m": before["cube_root_m"].astype(float).tolist(),
                "pad_object_distance_m": before["pad_object_distance_m"],
            },
            "observed_after": {
                "ee_root_m": after["ee_root_m"].astype(float).tolist(),
                "pad_midpoint_root_m": after["pad_midpoint_root_m"].astype(float).tolist(),
                "cube_root_m": after["cube_root_m"].astype(float).tolist(),
                "pad_object_distance_m": after["pad_object_distance_m"],
            },
            "planned_segment_minimum_pad_object_distance_m": planned_minimum,
            "pre_gate": asdict(pre_gate),
            "post_gate": asdict(post_gate),
            "normalized_4d_action": list(action_4d),
            "full_8d_packet": [float(value) for value in packet.values],
            "packet_derivation": derivation,
            "single_consumption": consumption,
            "gripper_intent": packet.gripper_intent.value,
            "task": after["task"],
            "physical_contact": after["physical_contact"],
            "forbidden_contact_sensor_peak_n": after["forbidden_contact_sensor_peak_n"],
            "active_termination_names": active_terminations,
            "phase_progress": progress,
        }
        records.append(record)
        if playback_enabled:
            previous_canonical_action = (
                *tuple(float(value) for value in handoff.metric_delta_root_m),
                0.0,
            )
        if tracking_decision.waypoint_advance_allowed:
            previous_planner_target = target.copy()
        if (
            bool(terminated.reshape(-1)[0].item())
            or bool(truncated.reshape(-1)[0].item())
            or active_terminations
        ):
            failure_classification = "PHASE_TRANSITION_FAILURE"
            failure_detail = f"UNEXPECTED_TERMINATION:{active_terminations}"
            return False
        if not post_gate.accepted:
            failure_classification = _failure_for_receipt(post_gate)
            failure_detail = str(post_gate.termination_reason)
            return False
        # Runtime replanning is allowed to publish only the canonical
        # deployable row and its privileged sidecar.  The historical
        # PrecontactReplayTransition contains cube-root GT and is therefore
        # diagnostic-only; appending it here would make a valid runtime
        # collection fail its own ``replay_rows == 0`` contract.  Frozen
        # replay diagnostics retain the legacy rows for post-hoc analysis.
        if runtime_replan_receipt is None:
            try:
                replay_transition = PrecontactReplayTransition(
                    observation={
                        "ee_root_m": record["observed_before"]["ee_root_m"],
                        "pad_midpoint_root_m": record["observed_before"]["pad_midpoint_root_m"],
                        "cube_root_m": record["observed_before"]["cube_root_m"],
                        "gripper_open": bool(before["gripper_is_open"]),
                    },
                    action_4d=action_4d,
                    next_observation={
                        "ee_root_m": record["observed_after"]["ee_root_m"],
                        "pad_midpoint_root_m": record["observed_after"]["pad_midpoint_root_m"],
                        "cube_root_m": record["observed_after"]["cube_root_m"],
                        "gripper_open": bool(after["gripper_is_open"]),
                    },
                    phase=phase,
                    ee_object_distance_m=float(after["ee_object_distance_m"]),
                    contact_flag=False,
                    planner_progress=float(
                        before["pad_object_distance_m"] - after["pad_object_distance_m"]
                    ),
                    safety_reject=False,
                    done=False,
                    termination_reason=None,
                    planner_nominal_action=action_4d,
                    residual_action=(0.0, 0.0, 0.0, 0.0),
                )
                replay_rows.append(replay_transition.payload())
            except Exception as error:
                failure_classification = "REPLAY_CONTRACT_FAILURE"
                failure_detail = f"{type(error).__name__}:{error}"
                return False
        if capture_this_step:
            try:
                if collection_rgbd is None or collection_bundle is None:
                    raise RuntimeError("CANONICAL_COLLECTION_RECEIPT_MISSING")
                capture_receipt = collection_bundle.receipt
                snapshot = capture_receipt.snapshot
                (
                    rgb,
                    depth_m,
                    depth_valid,
                    camera_frame,
                    camera_update_period_s,
                    camera_timestamp_s,
                    depth_stats,
                ) = collection_rgbd
                if p06_validation_requested:
                    phase_name = (
                        "far_reach"
                        if phase is PrecontactPhase.REACH
                        else "pregrasp"
                        if phase is PrecontactPhase.PREGRASP
                        else "near_contact"
                    )
                    p06_row = CandidateAValidationRow(
                        episode_id=str(candidate_a_bc_validation_episode_id),
                        row_id=collection_epoch,
                        control_timestamp_s=float(collection_epoch) * 0.020,
                        camera_timestamp_s=camera_timestamp_s,
                        camera_frame_id=camera_frame,
                        right_wrist_rgb=rgb,
                        right_wrist_depth_m=depth_m,
                        right_wrist_depth_valid=depth_valid,
                        ee_pose_robot_root_m_xyzw=(
                            *snapshot.measured_ee_position_root_m,
                            *snapshot.measured_ee_quaternion_root_xyzw,
                        ),
                        right_arm_joint_position_rad=snapshot.measured_joint_position_rad,
                        right_arm_joint_velocity_rad_s=snapshot.measured_joint_velocity_rad_s,
                        right_arm_joint_acceleration_rad_s2=(
                            snapshot.measured_joint_acceleration_rad_s2
                        ),
                        gripper_state_open=1.0 if before["gripper_is_open"] else 0.0,
                        previous_policy_action_4d_metric_root_m=previous_canonical_action,
                        candidate_a_expert_action_4d_metric_root_m=(
                            *tuple(float(value) for value in handoff.metric_delta_root_m),
                            0.0,
                        ),
                        ee_target_robot_root_m=(
                            np.asarray(
                                snapshot.measured_ee_position_root_m,
                                dtype=np.float64,
                            )
                            + np.asarray(
                                handoff.metric_delta_root_m, dtype=np.float64
                            )
                        ),
                        phase=phase_name,
                        asset_fingerprint=selected_hash,
                        planner_fingerprint=str(runtime_replan_plan_sidecar_sha256),
                        git_commit=str(p06_manifest["git_commit"]),
                    )
                    if p06_writer is None:
                        raise RuntimeError("P06_VALIDATION_WRITER_MISSING")
                    p06_writer.append(p06_row)
                    p06_row_count += 1
                    p06_phase_counts[phase_name] += 1
                    p06_bounded_capture_complete = all(
                        p06_phase_counts[name] >= minimum
                        for name, minimum in p06_minimum_phase_rows.items()
                    )
                    previous_canonical_action = tuple(
                        float(value)
                        for value in p06_row.candidate_a_expert_action_4d_metric_root_m
                    )
                    p06_publication.setdefault("camera_frame_ids", []).append(
                        camera_frame
                    )
                    p06_publication.setdefault("camera_timestamps_s", []).append(
                        camera_timestamp_s
                    )
                    p06_publication.setdefault("depth_runtime_statistics", []).append(
                        depth_stats
                    )
                else:
                    full8 = tuple(float(value) for value in packet.values)
                    epoch_id = f"candidate-a-contact-free:{collection_epoch:06d}"
                    receipt = PacketEpochReceipt(
                        epoch_id=epoch_id,
                        full8_packet=full8,
                        packet_hash=_hash_packet(epoch_id, full8),
                        env_step_count=int(consumption["env_step_count"]),
                        process_action_count=int(
                            consumption["action_manager_process_action_count"]
                        ),
                        controller_consumption_count=1,
                        snapshot_hook_enabled=True,
                        snapshot_hook_receipt=(
                            f"dls-capture:{capture_receipt.control_epoch}:"
                            f"{capture_receipt.selected_packet_hash}"
                        ),
                        source_sha256=_sha256(Path(__file__).resolve()),
                        asset_sha256=selected_hash,
                        trajectory_sha256=(
                            str(runtime_replan_plan_sidecar_sha256)
                            if runtime_replan_collection
                            else _sha256(replay)
                        ),
                    )
                    row = CanonicalContactFreeRow(
                        timestamp_s=float(collection_epoch) * 0.020,
                        control_step=collection_epoch,
                        actor_fields=ACTOR_FIELDS,
                        right_wrist_rgb=rgb,
                        right_wrist_depth_m=depth_m,
                        right_wrist_depth_valid=depth_valid,
                        ee_pose_robot_root_m_xyzw=(
                            *snapshot.measured_ee_position_root_m,
                            *snapshot.measured_ee_quaternion_root_xyzw,
                        ),
                        right_arm_joint_position_rad=(
                            snapshot.measured_joint_position_rad
                        ),
                        right_arm_joint_velocity_rad_s=(
                            snapshot.measured_joint_velocity_rad_s
                        ),
                        right_arm_joint_acceleration_rad_s2=(
                            snapshot.measured_joint_acceleration_rad_s2
                        ),
                        gripper_state_open=(
                            1.0 if before["gripper_is_open"] else 0.0
                        ),
                        previous_policy_action_4d_metric_root_m=(
                            previous_canonical_action
                        ),
                        policy_action_4d_metric_root_m=(
                            *tuple(
                                float(value)
                                for value in handoff.metric_delta_root_m
                            ),
                            0.0,
                        ),
                        packet_receipt=receipt,
                        jacobian_6x7=snapshot.frame_jacobian_root,
                        tensor_dtype=snapshot.runtime_dtype,
                        runtime_device=snapshot.runtime_device,
                        camera_evidence=collection_camera_evidence,
                    )
                    row.validate()
                    if canonical_writer is None:
                        raise RuntimeError("CANONICAL_COLLECTION_WRITER_MISSING")
                    canonical_writer.append(row)
                    canonical_row_count += 1
                    previous_canonical_action = row.policy_action_4d_metric_root_m
                    canonical_collection_receipt["camera_frame_ids"].append(camera_frame)
                    canonical_collection_receipt["camera_update_period_s"] = (
                        camera_update_period_s
                    )
                    canonical_collection_receipt.setdefault("camera_timestamps_s", []).append(
                        camera_timestamp_s
                    )
                    canonical_collection_receipt.setdefault("depth_runtime_statistics", []).append(
                        depth_stats
                    )
                    if collection_camera_evidence is None:
                        raise RuntimeError("CANONICAL_COLLECTION_DUAL_CAMERA_EVIDENCE_MISSING")
                    for camera_name in RGBD_CAMERA_NAMES:
                        camera_evidence = collection_camera_evidence[camera_name]
                        canonical_collection_receipt[
                            "camera_frame_ids_by_camera"
                        ][camera_name].append(int(camera_evidence["frame_id"]))
                        canonical_collection_receipt[
                            "camera_timestamps_s_by_camera"
                        ][camera_name].append(float(camera_evidence["timestamp_s"]))
            except BaseException as error:
                failure_classification = "CANONICAL_COLLECTION_CONTRACT_FAILURE"
                failure_detail = f"{type(error).__name__}:{error}"
                return False
        _append_phase(phase)
        return True

    try:
        initial = _state()
        if initial["pad_object_distance_m"] <= distance.open_handoff_candidate_m:
            raise RuntimeError("CANDIDATE_A_CONTACT_FREE_INITIAL_STATE_INSIDE_HANDOFF")
        for waypoint_index, waypoint in enumerate(planned_ee):
            phase = (
                PrecontactPhase.REACH
                if waypoint_index < fine_start_index
                else PrecontactPhase.PREGRASP
                if waypoint_index == fine_start_index
                else PrecontactPhase.FINE_APPROACH
            )
            current = _state()
            targets, reaches_handoff = split_open_translation_to_handoff(
                start_ee_root_m=current["ee_root_m"],
                start_pad_root_m=current["pad_midpoint_root_m"],
                object_root_m=current["cube_root_m"],
                desired_ee_root_m=waypoint,
                handoff_distance_m=distance.open_handoff_candidate_m,
            )
            # ``targets`` are re-timed from this *measured* state.  Reset the
            # directional anchor at the same boundary; retaining the prior
            # nominal waypoint would compare the new feedback target to a
            # stale reference and can manufacture a negative cosine during
            # normal tracking lag.
            previous_planner_target = current["ee_root_m"].copy()
            for segment_index, raw_target in enumerate(targets):
                repeat_index = 0
                while True:
                    if not _forward(
                        target=np.asarray(raw_target, dtype=np.float64),
                        phase=phase,
                        label=(
                            f"CANDIDATE_A_CONTACT_FREE_{phase.value}_"
                            f"WP{waypoint_index:03d}_SEG{segment_index:03d}_"
                            f"HOLD{repeat_index:02d}"
                        ),
                    ):
                        raise RuntimeError("CANDIDATE_A_CONTACT_FREE_STEP_REJECTED")
                    if p06_bounded_capture_complete:
                        break
                    if (
                        last_tracking_decision is None
                        or last_tracking_decision.waypoint_advance_allowed
                    ):
                        break
                    repeat_index += 1
                    if last_tracking_decision.substate is ExecutionSubstate.DIRECTIONAL_REEVALUATION:
                        failure_classification = "ACTION_CONTRACT_FAILURE"
                        failure_detail = "DIRECTIONAL_REEVALUATION_REQUIRES_REPLANNING"
                        raise RuntimeError("CANDIDATE_A_DIRECTIONAL_REEVALUATION_REQUIRED")
                if p06_bounded_capture_complete:
                    break
            if p06_bounded_capture_complete:
                break
            if reaches_handoff:
                handoff_reached = True
                if not _forward(
                    target=_ee_root_position(env),
                    phase=PrecontactPhase.OPEN_HANDOFF,
                    label="CANDIDATE_A_OPEN_HANDOFF_BOUNDARY",
                    force_zero_playback=True,
                ):
                    raise RuntimeError("CANDIDATE_A_OPEN_HANDOFF_REJECTED")
                # BC is deliberately not loaded or called.  This OPEN
                # heartbeat proves only the boundary at which a later,
                # separately-authorized micro-approach could begin.
                if not _forward(
                    target=_ee_root_position(env),
                    phase=PrecontactPhase.BC_MICRO_APPROACH,
                    label="CANDIDATE_A_BC_MICRO_APPROACH_PRE_ENTRY_OPEN_ONLY",
                    force_zero_playback=True,
                ):
                    raise RuntimeError("CANDIDATE_A_BC_MICRO_PRE_ENTRY_REJECTED")
                break
        if (
            not p06_bounded_capture_complete
            and not handoff_reached
            and failure_classification is None
        ):
            # The frozen cuRobo trajectory is EE-frame based.  Candidate-A
            # telemetry proves that its terminal EE point can still leave the
            # distal-pad midpoint outside the 3 cm OPEN handoff sphere.  Keep
            # the frozen path untouched, then append only this measured,
            # fixed-orientation pad-frame translation.  It remains OPEN-only,
            # is re-timed by the existing 4.5 mm contract, and stops at the
            # first handoff intersection before the empirical CLOSE band.
            current = _state()
            extension_target = np.asarray(
                pad_frame_open_handoff_ee_target(
                    start_ee_root_m=current["ee_root_m"],
                    start_pad_root_m=current["pad_midpoint_root_m"],
                    object_root_m=current["cube_root_m"],
                    handoff_distance_m=distance.open_handoff_candidate_m,
                ),
                dtype=np.float64,
            )
            targets, reaches_handoff = split_open_translation_to_handoff(
                start_ee_root_m=current["ee_root_m"],
                start_pad_root_m=current["pad_midpoint_root_m"],
                object_root_m=current["cube_root_m"],
                desired_ee_root_m=extension_target,
                handoff_distance_m=distance.open_handoff_candidate_m,
            )
            if not reaches_handoff or not targets:
                raise RuntimeError("PAD_FRAME_HANDOFF_EXTENSION_DID_NOT_REACH_OPEN_BOUNDARY")
            previous_planner_target = current["ee_root_m"].copy()
            for segment_index, raw_target in enumerate(targets):
                repeat_index = 0
                while True:
                    if not _forward(
                        target=np.asarray(raw_target, dtype=np.float64),
                        phase=PrecontactPhase.FINE_APPROACH,
                        label=(
                            "CANDIDATE_A_CONTACT_FREE_FINE_APPROACH_"
                            f"PAD_FRAME_HANDOFF_SEG{segment_index:03d}_"
                            f"HOLD{repeat_index:02d}"
                        ),
                    ):
                        raise RuntimeError("PAD_FRAME_HANDOFF_EXTENSION_STEP_REJECTED")
                    if p06_bounded_capture_complete:
                        break
                    if (
                        last_tracking_decision is None
                        or last_tracking_decision.waypoint_advance_allowed
                    ):
                        break
                    repeat_index += 1
                    if last_tracking_decision.substate is ExecutionSubstate.DIRECTIONAL_REEVALUATION:
                        failure_classification = "ACTION_CONTRACT_FAILURE"
                        failure_detail = "PAD_FRAME_HANDOFF_DIRECTIONAL_REEVALUATION_REQUIRES_REPLANNING"
                        raise RuntimeError("PAD_FRAME_HANDOFF_DIRECTIONAL_REEVALUATION_REQUIRED")
                if p06_bounded_capture_complete:
                    break
            pad_frame_handoff_extension_used = True
            pad_frame_handoff_extension_segment_count = len(targets)
            handoff_reached = True
            if not _forward(
                target=_ee_root_position(env),
                phase=PrecontactPhase.OPEN_HANDOFF,
                label="CANDIDATE_A_OPEN_HANDOFF_BOUNDARY",
                force_zero_playback=True,
            ):
                raise RuntimeError("CANDIDATE_A_OPEN_HANDOFF_REJECTED")
            if not _forward(
                target=_ee_root_position(env),
                phase=PrecontactPhase.BC_MICRO_APPROACH,
                label="CANDIDATE_A_BC_MICRO_APPROACH_PRE_ENTRY_OPEN_ONLY",
                force_zero_playback=True,
            ):
                raise RuntimeError("CANDIDATE_A_BC_MICRO_PRE_ENTRY_REJECTED")
        if not handoff_reached and failure_classification is None:
            failure_classification = "PLANNER_HANDOFF_FAILURE"
            failure_detail = "PLANNER_PATH_DID_NOT_REACH_3CM_OPEN_HANDOFF"
    except BaseException as error:
        if failure_classification is None:
            failure_classification = "PHASE_TRANSITION_FAILURE"
            failure_detail = f"{type(error).__name__}:{error}"

    final_state = _state()
    if canonical_capture_enabled:
        try:
            arm_term.set_read_only_dls_preview_capture_enabled(False)
            canonical_capture_enabled = False
        except BaseException as error:
            canonical_capture_disable_error = f"{type(error).__name__}:{error}"
    freeze_after = _source_freeze(
        keyboard_v3_branch=(
            p06_validation_requested or runtime_replan_collection
        )
    )
    expected_phase_sequence = [
        PrecontactPhase.REACH.value,
        PrecontactPhase.PREGRASP.value,
        PrecontactPhase.FINE_APPROACH.value,
        PrecontactPhase.OPEN_HANDOFF.value,
        PrecontactPhase.BC_MICRO_APPROACH.value,
    ]
    checks = {
        "candidate_a_binding": (
            selected_asset.resolve() == CANDIDATE_A_ASSET.resolve()
            and selected_hash == EXPECTED_CANDIDATE_A_ASSET_SHA256
        ),
        "source_freeze_stable": freeze_before == freeze_after
        and freeze_after.get("SOURCE_FREEZE") == "PASS",
        "gripper_remained_open": all_open and bool(final_state["gripper_is_open"]),
        "minimum_pad_object_clearance_strictly_above_guard": bool(records)
        and min(
            float(record["observed_after"]["pad_object_distance_m"])
            for record in records
        )
        > distance.contact_guard_distance_m,
        "all_planned_segment_receipts_strictly_above_guard": bool(records)
        and all(
            float(record["planned_segment_minimum_pad_object_distance_m"])
            > distance.contact_guard_distance_m
            for record in records
        ),
        "no_physical_contact": all_no_contact and not bool(final_state["physical_contact"]),
        "no_forbidden_collision": all_no_collision
        and float(final_state["forbidden_contact_peak_n"]) == 0.0,
        "no_close_command": all(
            record["gripper_intent"] == "OPEN" for record in records
        ),
        "exact_4d_open_action": all_exact_4d,
        "no_orientation_or_elbow_policy_action": all_zero_orientation,
        "no_follower_direct_commands": follower_direct_target_write_count == 0,
        "single_consumption": all_single_consumption,
        "phase_sequence": phase_sequence == expected_phase_sequence,
        "open_handoff_reached": handoff_reached,
        "final_open_handoff_distance_in_valid_range": bool(
            distance.contact_guard_distance_m
            < float(final_state["pad_object_distance_m"])
            <= distance.open_handoff_max_m
        ),
        "precontact_replay_rows_only": (
            len(replay_rows) == len(records)
            and all(row["contact_flag"] is False for row in replay_rows)
            if runtime_replan_receipt is None
            else len(replay_rows) == 0
        ),
        "runtime_replan_learning_replay_disabled": (
            runtime_replan_receipt is None or len(replay_rows) == 0
        ),
        # These are required false predicates for this execution mode, not
        # required true predicates.  A contact-free run must not load BC,
        # issue CLOSE, or observe CONTACT.
        "bc_not_executed": not playback_enabled,
        "contact_free_bc_playback_contract": (
            playback_inference_count > 0
            if playback_enabled
            else playback_inference_count == 0
        ),
        "close_not_executed": True,
        "contact_not_executed": True,
    }
    required_true = {
        key: value
        for key, value in checks.items()
        if key not in {"bc_not_executed", "close_not_executed", "contact_not_executed"}
    }
    required_false_occurred = {
        "bc_executed": (
            False if playback_enabled else not checks["bc_not_executed"]
        ),
        "close_executed": not checks["close_not_executed"],
        "contact_executed": not checks["contact_not_executed"],
        "physical_contact": not all_no_contact,
        "bilateral_contact": bool(final_state["task"]["bilateral_contact"]),
    }
    functional = (
        failure_classification is None
        and evaluate_contact_free_verdict(
            required_true=required_true,
            required_false=required_false_occurred,
            execution_mode=CONTACT_FREE_EXECUTION_MODE,
        )
    )
    if p06_validation_requested:
        try:
            if canonical_capture_disable_error is not None:
                raise RuntimeError(
                    "P06_VALIDATION_CAPTURE_DISABLE_FAILED:"
                    + canonical_capture_disable_error
                )
            p06_safe_capture = bool(
                p06_bounded_capture_complete
                and failure_classification is None
                and checks["candidate_a_binding"]
                and checks["source_freeze_stable"]
                and checks["gripper_remained_open"]
                and checks["minimum_pad_object_clearance_strictly_above_guard"]
                and checks["all_planned_segment_receipts_strictly_above_guard"]
                and checks["no_physical_contact"]
                and checks["no_forbidden_collision"]
                and checks["no_close_command"]
                and checks["exact_4d_open_action"]
                and checks["no_orientation_or_elbow_policy_action"]
                and checks["no_follower_direct_commands"]
                and checks["single_consumption"]
            )
            if not p06_safe_capture:
                raise RuntimeError("P06_BOUNDED_VALIDATION_CAPTURE_CONTRACT_FAILED")
            if p06_writer is None or p06_row_count != len(records) or p06_row_count <= 0:
                raise RuntimeError("P06_VALIDATION_ROW_CARDINALITY_MISMATCH")
            published = p06_writer.publish()
            import h5py

            with h5py.File(published, "r") as handle:
                rows = handle["rows"]
                phases = {
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in rows["phase"][:]
                }
                frames = np.asarray(rows["camera_frame_id"][:], dtype=np.int64)
                camera_times = np.asarray(
                    rows["camera_timestamp_s"][:], dtype=np.float64
                )
                control_times = np.asarray(
                    rows["control_timestamp_s"][:], dtype=np.float64
                )
                if (
                    phases != {"far_reach", "pregrasp", "near_contact"}
                    or not np.array_equal(
                        rows["row_id"][:], np.arange(p06_row_count, dtype=np.int64)
                    )
                    or not np.allclose(
                        control_times,
                        np.arange(p06_row_count, dtype=np.float64) * 0.020,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or not np.any(frames[1:] == frames[:-1])
                    or not np.all(
                        camera_times[1:][frames[1:] == frames[:-1]]
                        == camera_times[:-1][frames[1:] == frames[:-1]]
                    )
                ):
                    raise RuntimeError("P06_DURABLE_DATASET_AUDIT_FAILED")
            p06_publication.update(
                {
                    "state": "PUBLISHED",
                    "capture_scope": "BOUNDED_VALIDATION_ONLY_BEFORE_CONTACT",
                    "base_contact_free_smoke_completed": bool(functional),
                    "path": str(published.resolve()),
                    "sha256": _sha256(published),
                    "row_count": p06_row_count,
                    "phase_counts": dict(p06_phase_counts),
                    "minimum_phase_rows": dict(p06_minimum_phase_rows),
                    "camera_reuse_count": int(np.sum(frames[1:] == frames[:-1])),
                    "control_hz": 50,
                    "rgbd_hz": 25,
                    "physics_hz": 500,
                    "training_eligible": False,
                }
            )
        except BaseException as error:
            if p06_writer is not None:
                p06_writer.abort()
            p06_publication.update(
                {
                    "state": "FAILED",
                    "row_count": p06_row_count,
                    "error": f"{type(error).__name__}:{error}",
                }
            )
            if failure_classification is None:
                failure_classification = "P06_VALIDATION_CONTRACT_FAILURE"
                failure_detail = f"{type(error).__name__}:{error}"
        # P0.6 is a bounded validation-capture contract, not a request to
        # complete the contact-free handoff smoke.  Its functional verdict is
        # therefore the durable authoritative dataset publication itself.
        functional = bool(p06_publication["state"] == "PUBLISHED")
    if canonical_collection_output is not None:
        try:
            if canonical_capture_disable_error is not None:
                raise RuntimeError(
                    "CANONICAL_COLLECTION_CAPTURE_DISABLE_FAILED:"
                    + canonical_capture_disable_error
                )
            if canonical_writer is None:
                raise RuntimeError("CANONICAL_COLLECTION_WRITER_MISSING")
            expected_capture_steps = list(
                range(0, len(records), canonical_capture_control_stride)
            )
            if canonical_row_count != len(expected_capture_steps) or canonical_row_count <= 0:
                raise RuntimeError("CANONICAL_COLLECTION_ROW_RECEIPT_CARDINALITY_MISMATCH")
            partial_safe_failure = bool(
                not functional
                and failure_classification == "ACTION_CONTRACT_FAILURE"
                and failure_detail == "TRACKING_CATCHUP_BUDGET_EXCEEDED"
                and all_single_consumption
                and all_open
                and all_no_contact
                and all_no_collision
                and all_exact_4d
                and all_zero_orientation
                and freeze_before == freeze_after
                and freeze_after.get("SOURCE_FREEZE") == "PASS"
            )
            if not functional and not partial_safe_failure:
                raise RuntimeError("CANONICAL_COLLECTION_BASE_PRECONTACT_CONTRACT_FAILED")
            if functional:
                published = canonical_writer.publish()
            else:
                published = canonical_writer.publish_partial_safe(
                    canonical_collection_output.with_name("PARTIAL_SAFE_ROWS.hdf5"),
                    failure_classification=str(failure_classification),
                    failure_detail=str(failure_detail),
                    # The catch-up overflow is rejected before packet creation
                    # and before RGB-D append, so no unsafe stored row needs
                    # trimming.  The rejected trigger is recorded separately.
                    removed_failure_tail_rows=0,
                    removed_safety_rows=0,
                    removed_contract_rows=0,
                )
            # Verify the durable artifact by reopening it.  This is a schema
            # receipt only; it never participates in policy/control routing.
            import h5py

            with h5py.File(published, "r") as handle:
                stored_manifest = json.loads(str(handle.attrs["manifest_json"]))
                group = handle["rows"]
                if (
                    stored_manifest.get("actor_fields") != list(ACTOR_FIELDS)
                    or not bool(
                        stored_manifest.get("camera_evidence_contract", {}).get(
                            "active", False
                        )
                    )
                    or int(group["policy_action_4d_metric_root_m"].shape[0])
                    != canonical_row_count
                    or tuple(group["right_wrist_rgb"].shape[1:]) != (192, 256, 3)
                    or tuple(group["right_wrist_depth_m"].shape[1:]) != (192, 256, 1)
                    or not np.array_equal(
                        group["control_step"][:],
                        np.asarray(expected_capture_steps, dtype=np.int64),
                    )
                    or not np.allclose(
                        group["timestamp_s"][:],
                        np.asarray(expected_capture_steps, dtype=np.float64) * 0.020,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    raise RuntimeError("CANONICAL_COLLECTION_DURABLE_SCHEMA_AUDIT_FAILED")
                for camera_name in RGBD_CAMERA_NAMES:
                    raw = group[f"{camera_name}_depth_raw_m"][:]
                    source_valid = group[f"{camera_name}_depth_source_valid"][:]
                    frames = group[f"{camera_name}_camera_frame_id"][:]
                    associated = group[f"associated_{camera_name}_frame_index"][:]
                    camera_times = group[f"{camera_name}_camera_timestamp_s"][:]
                    poses = group[f"{camera_name}_camera_pose_root_m_xyzw"][:]
                    if (
                        raw.shape != (canonical_row_count, 192, 256, 1)
                        or source_valid.shape != raw.shape
                        or not np.array_equal(frames, associated)
                        or np.any(np.diff(frames) < 0)
                        or np.any(np.diff(camera_times) < 0.0)
                        or poses.shape != (canonical_row_count, 7)
                        or not np.isfinite(poses).all()
                        or np.any(np.isnan(raw))
                        or np.any(np.isneginf(raw))
                        or np.any(np.isfinite(raw) & (raw < 0.0))
                    ):
                        raise RuntimeError(
                            f"CANONICAL_COLLECTION_{camera_name.upper()}_EVIDENCE_AUDIT_FAILED"
                        )
                if tuple(group["head_rgb"].shape[1:]) != (192, 256, 3):
                    raise RuntimeError("CANONICAL_COLLECTION_HEAD_RGB_AUDIT_FAILED")
            publication = {
                "state": (
                    "PUBLISHED" if functional else "PARTIAL_SAFE_PUBLISHED"
                ),
                "path": str(published.resolve()),
                "sha256": _sha256(published),
                "row_count": canonical_row_count,
                "policy_rate_hz": 50,
                "capture_rate_hz": int(canonical_capture_rate_hz),
                "capture_control_step_stride": int(canonical_capture_control_stride),
                "source_camera_rate_hz": (
                    1.0
                    / float(canonical_collection_receipt["camera_update_period_s"])
                ),
                "physics_dt_s": 0.002,
                "action_contract": "[dx,dy,dz,HOLD_OPEN] robot_root meters",
                "actor_gt_fields": "NONE",
                "runtime_plan_sidecar_only": bool(runtime_replan_collection),
            }
            if not functional:
                publication["partial_safe"] = {
                    "eligible": True,
                    "failure_trigger_rejected_before_action_submission": True,
                    "failure_trigger_rejected_before_rgbd_append": True,
                    "total_raw_rows": canonical_row_count,
                    "partial_safe_rows": canonical_row_count,
                    "removed_failure_tail_rows": 0,
                    "removed_safety_rows": 0,
                    "removed_contract_rows": 0,
                }
            canonical_collection_receipt.update(publication)
        except BaseException as error:
            if failure_classification is None:
                failure_classification = "CANONICAL_COLLECTION_CONTRACT_FAILURE"
                failure_detail = f"{type(error).__name__}:{error}"
            if canonical_writer is not None:
                canonical_writer.abort()
            canonical_collection_receipt.update(
                {
                    "state": "FAILED",
                    "row_count": canonical_row_count,
                    "error": f"{type(error).__name__}:{error}",
                }
            )
        functional = bool(functional and canonical_collection_receipt["state"] == "PUBLISHED")
    cosine_summary = summarize_cosine_distribution(
        record["cosine_similarity"]
        for record in records
        if record.get("cosine_similarity") is not None
    )
    checks["canonical_collection"] = (
        canonical_collection_output is None
        or canonical_collection_receipt["state"] == "PUBLISHED"
    )
    checks["p06_candidate_a_validation"] = (
        not p06_validation_requested or p06_publication["state"] == "PUBLISHED"
    )
    # ``make_g2_policy_4d_training_env_cfg`` supplies a historical default
    # E1 receipt.  Candidate-A contact-free runs must not publish that stale
    # provenance as their selected asset.  Keep only common binding fields and
    # replace the candidate identity with the hash-checked selection above.
    selected_asset_binding = {
        key: value
        for key, value in asset_receipt.as_dict().items()
        if key
        not in {
            "binding_id",
            "candidate_asset_path",
            "candidate_asset_sha256",
            "qualification_report_asset_path",
            "qualification_report_path",
            "qualification_report_sha256",
            "dependency_manifest_sha256",
            "physics_verdict",
        }
    }
    selected_asset_binding.update(
        {
            "binding_id": "LEGACY_CANDIDATE_A_CONTACT_FREE_DIAGNOSTIC_V1",
            "candidate_asset_path": str(selected_asset.resolve()),
            "candidate_asset_sha256": selected_hash,
            "classification": (
                "LEGACY_CANDIDATE_A_CONTACT_FREE_DIAGNOSTIC_ONLY_NOT_M2_AUTHORITY"
            ),
            "physics_verdict": "NOT_EXECUTED_IN_CONTACT_FREE_SCOPE",
        }
    )
    report = {
        "schema": (
            "g2_contact_free_visual_bc_playback_v1"
            if playback_enabled
            else "g2_legacy_candidate_a_contact_free_smoke_v2"
        ),
        "execution_mode": CONTACT_FREE_EXECUTION_MODE,
        "scope": (
            "BOUNDED_CANDIDATE_A_VISUAL_BC_PLAYBACK_OPEN_ONLY_NO_CLOSE_NO_CONTACT"
            if playback_enabled
            else "ONE_BOUNDED_CANDIDATE_A_OPEN_ONLY_PRECONTACT_SMOKE_NO_BC_NO_CLOSE_NO_CONTACT"
        ),
        "source_freeze_before": freeze_before,
        "source_freeze_after": freeze_after,
        "asset_binding": selected_asset_binding,
        "diagnostic_asset_selection": diagnostic_asset_selection,
        "task_contract": task_contract,
        "task_geometry_binding": task_geometry_binding,
        "task_geometry_readback": task_geometry_readback,
        "p0a_provenance": p0a_provenance,
        "planner_provenance": (
            "POST_RESET_RUNTIME_REPLAN"
            if runtime_replan_receipt is not None
            else "FROZEN_REPLAY"
        ),
        # Historical Candidate-A evidence is tied to a frozen replay.  The
        # separately reviewed runtime-replan route instead records a
        # planner-only receipt which is explicitly not an actor/replay field.
        "trajectory_replay": str(replay.resolve()) if replay is not None else None,
        "trajectory_replay_sha256": _sha256(replay) if replay is not None else None,
        "runtime_replan_receipt": (
            dict(runtime_replan_receipt)
            if runtime_replan_receipt is not None
            else None
        ),
        "runtime_replan_input_telemetry": (
            dict(runtime_replan_input_telemetry)
            if runtime_replan_input_telemetry is not None
            else None
        ),
        "runtime_replan_input_telemetry_path": (
            str(runtime_replan_input_telemetry_path)
            if runtime_replan_input_telemetry_path is not None
            else None
        ),
        "independent_collection": (
            {
                "episode_id": independent_collection_episode_id,
                "reset_seed": int(seed),
                "planner_provenance": "POST_RESET_RUNTIME_REPLAN",
                "runtime_plan_sidecar_sha256": runtime_replan_plan_sidecar_sha256,
                "actor_input_contains_cube_gt": False,
                "replay_row_contains_planner_cube_gt": False,
            }
            if runtime_replan_collection
            else None
        ),
        "open_settle": dict(settle),
        "contact_free_bc_playback": {
            "enabled": playback_enabled,
            "checkpoint": playback_checkpoint_receipt,
            "inference_count": playback_inference_count,
            "action_norm_m": {
                "maximum": max(playback_action_norms_m, default=0.0),
                "mean": float(np.mean(playback_action_norms_m))
                if playback_action_norms_m
                else 0.0,
            },
            "actor_gt_input": "NONE",
            "gripper": "HOLD_OPEN",
            "contact_guard": "FAIL_CLOSED_BEFORE_ENV_STEP",
        },
        "contact_free_contract": {
            **distance.payload(),
            "segment_retiming_authority": "P0_A_ATTESTED_4P5_MM_CARTESIAN_MICRO_COMMAND",
            "maximum_translation_per_segment_m": CONTACT_FREE_SEGMENT_MAX_TRANSLATION_M,
            "pad_measurement": "LIVE_RIGHT_DISTAL_LINK4_ORIGIN_MIDPOINT_IN_ROBOT_ROOT",
            "pad_measurement_is_soft_pad_calibration": False,
        },
        "phase_sequence": phase_sequence,
        "records": records,
        "replay_rows": replay_rows,
        "canonical_contact_free_collection": canonical_collection_receipt,
        "p06_candidate_a_bc_validation": p06_publication,
        "final_state": {
            "pad_object_distance_m": final_state["pad_object_distance_m"],
            "ee_object_distance_m": final_state["ee_object_distance_m"],
            "gripper_is_open": final_state["gripper_is_open"],
            "physical_contact": final_state["physical_contact"],
            "task": final_state["task"],
            "forbidden_contact_sensor_peak_n": final_state[
                "forbidden_contact_sensor_peak_n"
            ],
        },
        "minimum_observed_pad_object_distance_m": (
            min(
                float(record["observed_after"]["pad_object_distance_m"])
                for record in records
            )
            if records
            else None
        ),
        "minimum_planned_segment_pad_object_distance_m": (
            min(
                float(record["planned_segment_minimum_pad_object_distance_m"])
                for record in records
            )
            if records
            else None
        ),
        "maximum_segment_translation_m": maximum_segment_translation_m,
        "maximum_final_metric_action_m": maximum_segment_translation_m,
        "maximum_nominal_planner_increment_m": maximum_nominal_planner_increment_m,
        "tracking_budget": {
            "ordinary_tracking_cap": ordinary_tracking_cap,
            "critical_tracking_cap": critical_tracking_cap,
            "fine_approach_uses_critical_cap": True,
            "policy_rate_hz": 50,
            "physics_dt_s": 0.002,
        },
        "cosine_direction_summary": cosine_summary,
        "catchup_step_count": total_catchup_step_count,
        "maximum_catchup_steps_per_waypoint_observed": maximum_catchup_steps_observed,
        "minimum_pre_submit_clearance_m": (
            minimum_pre_submit_clearance_m
            if math.isfinite(minimum_pre_submit_clearance_m)
            else None
        ),
        "pad_frame_handoff_extension": {
            "used": pad_frame_handoff_extension_used,
            "segment_count": pad_frame_handoff_extension_segment_count,
            "authority": "MEASURED_EE_TO_DISTAL_PAD_OFFSET_FIXED_ORIENTATION_OPEN_ONLY",
            "frozen_curobo_trajectory_modified": False,
        },
        "safety_reject_count": safety_reject_count,
        "accepted_replay_row_count": len(replay_rows),
        "rejected_replay_row_count": 0,
        "replay_contact_row_count": 0,
        "replay_close_row_count": 0,
        "replay_bc_row_count": 0,
        "maximum_active_qdot_rad_s": maximum_active_qdot_rad_s,
        "maximum_active_qdd_rad_s2": maximum_active_qdd_rad_s2,
        "maximum_passive_qdot_rad_s": maximum_passive_qdot_rad_s,
        "maximum_passive_qdd_rad_s2": maximum_passive_qdd_rad_s2,
        "final_phase": (
            phase_sequence[-1] if phase_sequence else None
        ),
        "follower_direct_target_write_count": follower_direct_target_write_count,
        "failure_classification": failure_classification,
        "failure_detail": failure_detail,
        "limiter_failure_snapshot": limiter_failure_snapshot,
        "checks": checks,
        "required_true": required_true,
        "required_false_occurred": required_false_occurred,
        "source_freeze_complete": bool(
            freeze_before.get("SOURCE_FREEZE") == "PASS"
            and freeze_after.get("SOURCE_FREEZE") == "PASS"
            and freeze_before == freeze_after
            and selected_hash == EXPECTED_CANDIDATE_A_ASSET_SHA256
        ),
        "unexpected_mutation": not bool(freeze_before == freeze_after),
        "lifecycle_counts": counter.summary(),
        "verdict": {
            "EXECUTION_MODE": CONTACT_FREE_EXECUTION_MODE,
            "SOURCE_FREEZE_COMPLETE": bool(
                freeze_before.get("SOURCE_FREEZE") == "PASS"
                and freeze_after.get("SOURCE_FREEZE") == "PASS"
                and freeze_before == freeze_after
                and selected_hash == EXPECTED_CANDIDATE_A_ASSET_SHA256
            ),
            "CANDIDATE_A_CONTACT_FREE_SMOKE": (
                "NOT_COMPLETED_VALIDATION_ONLY"
                if p06_validation_requested and functional
                else "PASS" if functional else "FAIL"
            ),
            "PRECONTACT_TRAINING_READINESS": "NO_P06_DOES_NOT_AUTHORIZE_TRAINING",
            "CANONICAL_CONTACT_FREE_COLLECTION": (
                "PASS"
                if canonical_collection_output is not None
                and canonical_collection_receipt["state"] == "PUBLISHED"
                else "NOT_REQUESTED"
                if canonical_collection_output is None
                else "FAIL"
            ),
            "P0_6_CANDIDATE_A_VALIDATION_DATASET": (
                "PASS"
                if p06_validation_requested
                and p06_publication["state"] == "PUBLISHED"
                else "NOT_REQUESTED"
                if not p06_validation_requested
                else "FAIL"
            ),
            "M2_CONTACT": "FAIL_CONTACT_MECHANICS_UNCHANGED",
            "TRAINING_AUTHORIZED": "NO",
            "FUNCTIONAL_VERDICT": "PASS" if functional else "FAIL",
            "PROCESS_VERDICT": "PENDING_PARENT",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, report)
    print("RUNTIME_END", flush=True)
    print("REPORT_SAVED", flush=True)
    return 0 if functional else 2


def _load_frozen_contact_free_plan(
    *,
    replay: Path,
    adaptive_waypoints: bool,
    adaptive_collision_report: Path | None,
    planner_active_velocity_limit_rad_s: float,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any], dict[str, Any] | None]:
    """Load the historical immutable replay without involving a reset state.

    This remains the only loader for the legacy seed-42 path.  Runtime replans
    must bypass this function entirely and are constructed after reset by
    :func:`_plan_contact_free_after_reset`.
    """

    from geniesim.rl.isaaclab.g2_policy_branch.planner_timing_contract import (
        AdaptiveWaypointConfig,
        build_adaptive_waypoint_schedule,
    )

    replay_data = np.load(replay)
    planned_ee = np.asarray(replay_data["ee_position_root_m"], dtype=np.float64)
    planned_q = np.asarray(replay_data["q_rad"], dtype=np.float64)
    if planned_ee.ndim != 2 or planned_ee.shape[1] != 3 or planned_ee.shape[0] < 2:
        raise RuntimeError("CUROBO_LIVE_REPLAY_SHAPE_INVALID")
    if planned_q.shape != (planned_ee.shape[0], 7):
        raise RuntimeError("CUROBO_LIVE_REPLAY_JOINT_SHAPE_INVALID")
    if not np.isfinite(planned_ee).all() or not np.isfinite(planned_q).all():
        raise RuntimeError("CUROBO_LIVE_REPLAY_NONFINITE")
    coarse_count = int(replay_data["coarse_waypoint_count"][0]) if (
        "coarse_waypoint_count" in replay_data.files
    ) else 58
    fine_start_index = coarse_count - 1
    collision_receipt: dict[str, Any] | None = None
    if adaptive_waypoints:
        if adaptive_collision_report is None or not adaptive_collision_report.is_file():
            raise RuntimeError("CUROBO_ADAPTIVE_COLLISION_RECEIPT_MISSING")
        collision_receipt = json.loads(adaptive_collision_report.read_text(encoding="utf-8"))
        collision_checks = collision_receipt.get("checks", {})
        motiongen = collision_receipt.get("motiongen", {})
        collision_source_pass = bool(
            collision_checks.get("path_table_and_self_collision_free") is True
            and collision_checks.get("unintended_non_pad_cube_collision_free") is True
            and int(motiongen.get("blocking_table_or_self_collision_waypoint_count", -1))
            == 0
            and float(motiongen.get("maximum_table_collision_cost", float("inf")))
            == 0.0
            and float(motiongen.get("maximum_self_collision_cost", float("inf")))
            == 0.0
            and int(motiongen.get("combined_waypoint_count", -1))
            == planned_q.shape[0]
        )
        if not collision_source_pass:
            raise RuntimeError("CUROBO_ADAPTIVE_COLLISION_RECEIPT_FAILED")
        adaptive_schedule = build_adaptive_waypoint_schedule(
            planned_q,
            planned_ee,
            coarse_waypoint_count=coarse_count,
            collision_free=np.ones(planned_q.shape[0], dtype=np.bool_),
            config=AdaptiveWaypointConfig(
                target_velocity_limit_rad_s=planner_active_velocity_limit_rad_s,
                target_acceleration_limit_rad_s2=10.0,
                policy_dt_s=0.02,
            ),
        )
    else:
        adaptive_schedule = {
            "schema": "g2_adaptive_waypoint_schedule_v1",
            "selected_indices": list(range(planned_q.shape[0])),
            "skipped_indices": [],
            "original_waypoint_count": int(planned_q.shape[0]),
            "effective_waypoint_count": int(planned_q.shape[0]),
            "skipped_waypoint_count": 0,
            "fine_start_original_index": fine_start_index,
            "coarse_original_waypoint_count": fine_start_index,
            "coarse_effective_waypoint_count": fine_start_index,
            "coarse_skip_ratio": 0.0,
            "coarse_effective_usage": 1.0,
            "fine_original_waypoint_count": int(planned_q.shape[0] - fine_start_index),
            "fine_effective_waypoint_count": int(planned_q.shape[0] - fine_start_index),
            "fine_skip_ratio": 0.0,
            "maximum_path_deviation_m": 0.0,
            "fine_density_preserved": True,
            "target_envelope_preserved": True,
        }
    if any(index >= fine_start_index for index in adaptive_schedule["skipped_indices"]):
        raise RuntimeError("CUROBO_ADAPTIVE_FINE_WAYPOINT_REMOVED")
    if not bool(adaptive_schedule["fine_density_preserved"]):
        raise RuntimeError("CUROBO_ADAPTIVE_FINE_DENSITY_NOT_PRESERVED")
    if not bool(adaptive_schedule["target_envelope_preserved"]):
        raise RuntimeError("CUROBO_ADAPTIVE_TARGET_ENVELOPE_FAILED")
    return planned_ee, planned_q, fine_start_index, adaptive_schedule, collision_receipt


class _TargetLimiterTelemetry:
    """Read-only interception of the existing target limiter.

    The wrapper calls the original bound method exactly once and never changes
    its arguments or result.  GPU tensors are cloned while the controller is
    running and copied to CPU only after the policy step has returned.
    """

    def __init__(self, arm_term: Any) -> None:
        self.arm_term = arm_term
        self.original_limiter = arm_term._synchronized_rate_limited_target
        self.original_apply = arm_term.apply_actions
        self.original_reset = arm_term._reset_target_rate_limit
        self.original_synchronize = arm_term.synchronize_target_to_measured
        self.current_substeps: list[dict[str, Any]] = []
        self.reset_generation = 0
        self.synchronize_generation = 0
        self._previous_limiter_velocity_for_qdd: np.ndarray | None = None
        self.installed = False

    @staticmethod
    def _cpu(tensor: Any) -> np.ndarray:
        return tensor.detach().to("cpu").numpy().astype(np.float64)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("TARGET_LIMITER_TELEMETRY_ALREADY_INSTALLED")

        def wrapped_limiter(_term, desired, measured):
            from geniesim.rl.isaaclab.g2_policy_branch.precontact_limiter_contract import (
                precheck_synchronized_endpoint,
            )

            before_target = _term._g2_previous_target.clone()
            before_velocity = _term._g2_previous_target_velocity.clone()
            before_initialized = _term._g2_target_initialized.clone()
            desired_clone = desired.clone()
            measured_clone = measured.clone()
            measured_q = self._cpu(measured_clone)[0]
            measured_qdot = None
            try:
                measured_qdot = self._cpu(
                    _term._asset.data.joint_vel[:, _term._joint_ids]
                )[0]
            except Exception:
                measured_qdot = None
            previous_qdd = None
            if self._previous_limiter_velocity_for_qdd is not None:
                previous_qdd = (
                    self._cpu(before_velocity)[0]
                    - self._previous_limiter_velocity_for_qdd
                ) / float(_term._g2_physics_dt_s)
            speed_limit_raw = self._cpu(
                _term._g2_environment_speed_limit_rad_s
            )[0]
            speed_limit = (
                float(speed_limit_raw.reshape(-1)[0])
                if speed_limit_raw.size == 1
                else speed_limit_raw
            )
            acceleration_limit = float(
                _term.cfg.maximum_joint_target_acceleration_rad_s2
            )
            try:
                precheck_payload = precheck_synchronized_endpoint(
                    previous_endpoint_q=self._cpu(before_target)[0],
                    previous_endpoint_qdot=self._cpu(before_velocity)[0],
                    ik_target_q=self._cpu(desired_clone)[0],
                    dt_s=float(_term._g2_physics_dt_s),
                    maximum_speed_rad_s=speed_limit,
                    maximum_acceleration_rad_s2=acceleration_limit,
                ).payload()
            except Exception as error:
                # Instrumentation must never become a second controller
                # authority or alter the original limiter's behavior.
                precheck_payload = {
                    "available": False,
                    "error": f"{type(error).__name__}:{error}",
                }
            common_input = {
                "measured_q": measured_q,
                "measured_qdot": measured_qdot,
                "ik_target_q": self._cpu(desired_clone)[0],
                "previous_endpoint_q": self._cpu(before_target)[0],
                "previous_endpoint_qdot": self._cpu(before_velocity)[0],
                "previous_endpoint_qdd": previous_qdd,
                "dt_s": float(_term._g2_physics_dt_s),
                "velocity_limit_rad_s": speed_limit,
                "acceleration_limit_rad_s2": acceleration_limit,
                "requested_synchronized_duration_s": None,
                "synchronized_duration_authority": "NOT_EXPLICIT_IN_PRODUCTION_LIMITER",
                "limiter_precheck": precheck_payload,
            }
            reset_generation = self.reset_generation
            synchronize_generation = self.synchronize_generation
            try:
                limited = self.original_limiter(desired, measured)
            except Exception as error:
                # Evidence-only failure receipt.  Preserve the exact inputs and
                # limiter dynamic state that produced the rejection, without
                # retrying, mutating, or replacing the production limiter.
                self.current_substeps.append(
                    {
                        **common_input,
                        "pre_rate_limit_target": desired_clone,
                        "measured_input": measured_clone,
                        "previous_post_limit_target": before_target,
                        "previous_accumulator_velocity": before_velocity,
                        "target_initialized_before": before_initialized,
                        "limiter_exception": f"{type(error).__name__}:{error}",
                        "limiter_accept": False,
                        "reject_reason": f"{type(error).__name__}:{error}",
                        "reset_generation": reset_generation,
                        "synchronize_generation": synchronize_generation,
                    }
                )
                self._previous_limiter_velocity_for_qdd = self._cpu(before_velocity)[0]
                raise
            after_target = _term._g2_previous_target.clone()
            after_velocity = _term._g2_previous_target_velocity.clone()
            if not bool((limited == after_target).all()):
                raise RuntimeError("TARGET_LIMITER_TELEMETRY_OUTPUT_STATE_MISMATCH")
            self.current_substeps.append(
                {
                    **common_input,
                    "pre_rate_limit_target": desired_clone,
                    "measured_input": measured_clone,
                    "previous_post_limit_target": before_target,
                    "previous_accumulator_velocity": before_velocity,
                    "target_initialized_before": before_initialized,
                    "post_limit_target": limited.clone(),
                    "accumulator_velocity_after": after_velocity,
                    "reset_generation": reset_generation,
                    "synchronize_generation": synchronize_generation,
                    "synchronized_speed_scale": (
                        _term.last_synchronized_speed_scale.clone()
                    ),
                    "synchronized_common_endpoint_scale": (
                        _term.last_synchronized_endpoint_scale.clone()
                    ),
                    "synchronized_per_joint_endpoint_scale": (
                        _term.last_synchronized_per_joint_endpoint_scale.clone()
                    ),
                    "synchronized_acceleration_scale": (
                        _term.last_synchronized_acceleration_scale.clone()
                    ),
                    "synchronized_endpoint_deceleration_mask": (
                        _term.last_synchronized_endpoint_deceleration_mask.clone()
                    ),
                    "synchronized_unavoidable_endpoint_overshoot_mask": (
                        _term.last_synchronized_unavoidable_overshoot_mask.clone()
                    ),
                    "limiter_accept": True,
                    "reject_reason": None,
                }
            )
            self._previous_limiter_velocity_for_qdd = self._cpu(after_velocity)[0]
            return limited

        def wrapped_apply(_term):
            record_count_before = len(self.current_substeps)
            result = self.original_apply()
            new_records = self.current_substeps[record_count_before:]
            if len(new_records) != 1:
                raise RuntimeError(
                    "TARGET_LIMITER_TELEMETRY_APPLY_CALL_COUNT_MISMATCH:"
                    f"{len(new_records)}"
                )
            record = new_records[0]
            record["final_emitted_target"] = _term._g2_previous_target.clone()
            record["final_accumulator_velocity"] = (
                _term._g2_previous_target_velocity.clone()
            )
            record["post_limiter_joint_clamp_applied"] = bool(
                (record["post_limit_target"] != record["final_emitted_target"]).any()
            )
            return result

        def wrapped_reset(_term, env_ids):
            self.reset_generation += 1
            self._previous_limiter_velocity_for_qdd = None
            return self.original_reset(env_ids)

        def wrapped_synchronize(_term, env_ids):
            self.synchronize_generation += 1
            self._previous_limiter_velocity_for_qdd = None
            return self.original_synchronize(env_ids)

        self.arm_term._synchronized_rate_limited_target = types.MethodType(
            wrapped_limiter, self.arm_term
        )
        self.arm_term.apply_actions = types.MethodType(wrapped_apply, self.arm_term)
        self.arm_term._reset_target_rate_limit = types.MethodType(
            wrapped_reset, self.arm_term
        )
        self.arm_term.synchronize_target_to_measured = types.MethodType(
            wrapped_synchronize, self.arm_term
        )
        self.installed = True

    def begin_policy_step(self) -> tuple[int, int]:
        if self.current_substeps:
            raise RuntimeError("TARGET_LIMITER_TELEMETRY_UNCONSUMED_SUBSTEPS")
        return self.reset_generation, self.synchronize_generation

    def end_policy_step(self, generations_before: tuple[int, int]) -> dict[str, Any]:
        if not self.current_substeps:
            raise RuntimeError("TARGET_LIMITER_TELEMETRY_NO_PHYSICS_SUBSTEP")
        substeps = self.current_substeps
        self.current_substeps = []
        result: dict[str, Any] = {
            "physics_substep_count": len(substeps),
            "reset_during_policy_step": self.reset_generation != generations_before[0],
            "synchronize_during_policy_step": (
                self.synchronize_generation != generations_before[1]
            ),
            "substeps": [],
        }
        for index, record in enumerate(substeps):
            converted = {
                key: self._cpu(value)[0]
                if hasattr(value, "detach") and value.ndim == 2
                else self._cpu(value)
                if hasattr(value, "detach")
                else value
                for key, value in record.items()
            }
            converted["physics_substep_index"] = index
            result["substeps"].append(converted)
        return result

    def restore(self) -> None:
        if not self.installed:
            return
        self.arm_term._synchronized_rate_limited_target = self.original_limiter
        self.arm_term.apply_actions = self.original_apply
        self.arm_term._reset_target_rate_limit = self.original_reset
        self.arm_term.synchronize_target_to_measured = self.original_synchronize
        self.installed = False

    def failure_snapshot(self) -> list[dict[str, Any]]:
        """Return unconsumed limiter evidence after a rejected policy step."""

        result: list[dict[str, Any]] = []
        for record in self.current_substeps:
            converted: dict[str, Any] = {}
            for key, value in record.items():
                if hasattr(value, "detach"):
                    array = self._cpu(value)
                    converted[key] = array[0] if array.ndim == 2 else array
                else:
                    converted[key] = value
            result.append(converted)
        return result


class _PassiveContactPhysicsTelemetry:
    """Read-only full-articulation samples after every physics scene update.

    The wrapper delegates to the original ``scene.update`` exactly once and
    only then copies public readback buffers.  It does not step simulation,
    write a target, modify contact state, or participate in a safety verdict.
    """

    CONTACT_CAPACITY = 32
    LOOP_JOINT_NAMES = (
        "idx93_gripper_r_outer_joint2",
        "idx94_gripper_r_inner_joint2",
    )
    SURFACE_SENSOR_KEYS = (
        (
            "inner",
            "right_inner_finger_contact",
            "gripper_r_inner_link4",
            "PAD_PRIMARY",
        ),
        (
            "outer",
            "right_outer_finger_contact",
            "gripper_r_outer_link4",
            "PAD_PRIMARY",
        ),
        (
            "outer_link2",
            "diagnostic_right_outer_link2_contact",
            "gripper_r_outer_link2",
            "OUTER_LINK2_CANDIDATE",
        ),
    )

    def __init__(
        self,
        env: Any,
        robot: Any,
        task_mdp: Any,
        *,
        torch_module: Any,
        warp_module: Any,
        stage: Any,
        stop_on_hard_stop: bool = False,
        defer_qdd_to_close_mechanics: bool = False,
    ) -> None:
        self.env = env
        self.robot = robot
        self.task_mdp = task_mdp
        self.torch = torch_module
        self.wp = warp_module
        self.stage = stage
        self.original_update = env.scene.update
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.installed = False
        self._substep_in_policy = 0
        self._first_any_contact_sample: int | None = None
        self._previous_qdot: np.ndarray | None = None
        self._hard_stop_event: dict[str, Any] | None = None
        self._hard_stop_receipt: RuntimeHardstopReceipt | None = None
        self._stop_on_hard_stop = bool(stop_on_hard_stop)
        self._physics_elapsed_s = 0.0
        self._defer_qdd_to_close_mechanics = bool(
            defer_qdd_to_close_mechanics
        )
        self._body_index = {
            name: index for index, name in enumerate(self.robot.body_names)
        }
        self._joint_index = {
            name: index for index, name in enumerate(self.robot.joint_names)
        }
        self._contact_sources = self._bind_contact_sources()
        self._loop_descriptors = self._bind_loop_descriptors()
        self._joint_limits = self._cpu_row(self.robot.data.joint_pos_limits)
        self._body_mass_kg = self._cpu_row(self.robot.root_view.get_masses())
        self._body_com_local_pose_xyzw = self._cpu_row(
            self.robot.root_view.get_coms()
        )
        inertia_flat = self._cpu_row(self.robot.root_view.get_inertias())
        self._body_inertia_link_kg_m2 = np.stack(
            [
                np.asarray(values, dtype=np.float64).reshape(3, 3, order="F")
                for values in inertia_flat
            ]
        )
        self._body_principal_moments_kg_m2 = np.empty(
            (len(self.robot.body_names), 3), dtype=np.float64
        )
        self._body_principal_axes_link = np.empty(
            (len(self.robot.body_names), 3, 3), dtype=np.float64
        )
        for body_index, inertia in enumerate(self._body_inertia_link_kg_m2):
            moments, axes = np.linalg.eigh(0.5 * (inertia + inertia.T))
            self._body_principal_moments_kg_m2[body_index] = moments
            self._body_principal_axes_link[body_index] = axes

    @staticmethod
    def _contact_view(sensor: Any) -> Any:
        view = getattr(sensor, "contact_view", None)
        if view is None:
            view = getattr(sensor, "contact_physx_view", None)
        if view is None:
            raise RuntimeError("PASSIVE_CONTACT_RAW_VIEW_UNAVAILABLE")
        return view

    def _bind_contact_sources(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for side, sensor_key, link_name, surface_identity in self.SURFACE_SENSOR_KEYS:
            sensor = self.env.scene[sensor_key]
            view = self._contact_view(sensor)
            if not callable(getattr(view, "get_contact_data", None)):
                raise RuntimeError(
                    f"PASSIVE_CONTACT_RAW_DATA_API_UNAVAILABLE:{sensor_key}"
                )
            if not callable(getattr(view, "get_friction_data", None)):
                raise RuntimeError(
                    f"PASSIVE_CONTACT_FRICTION_DATA_API_UNAVAILABLE:{sensor_key}"
                )
            if link_name not in self._body_index:
                raise RuntimeError(f"PASSIVE_CONTACT_PAD_BODY_MISSING:{link_name}")
            result[side] = {
                "sensor": sensor,
                "view": view,
                "link_name": link_name,
                "body_index": self._body_index[link_name],
                "surface_identity": surface_identity,
                # All bound sensors place Object at filtered pair index zero.
                # Some legacy pad sensors expose one pair while the collision
                # authority's outer-link2 sensor exposes Object and Table.
                "object_filter_pair_index": 0,
                # The filtered sensor contract is exact Object-only.  Raw
                # actor IDs are not needed to infer this name and are not
                # relabelled as collision-shape identity.
                "partner_actor": "/World/envs/env_0/Object",
            }
        return result

    def _bind_loop_descriptors(self) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        robot_root = "/World/envs/env_0/Robot"
        for joint_name in self.LOOP_JOINT_NAMES:
            prim = self.stage.GetPrimAtPath(
                f"{robot_root}/loop_joints/{joint_name}"
            )
            if not prim.IsValid():
                raise RuntimeError(
                    f"PASSIVE_CONTACT_LOOP_PRIM_MISSING:{joint_name}"
                )
            body0_targets = prim.GetRelationship("physics:body0").GetTargets()
            body1_targets = prim.GetRelationship("physics:body1").GetTargets()
            local0 = prim.GetAttribute("physics:localPos0").Get()
            local1 = prim.GetAttribute("physics:localPos1").Get()
            if (
                len(body0_targets) != 1
                or len(body1_targets) != 1
                or local0 is None
                or local1 is None
            ):
                raise RuntimeError(
                    f"PASSIVE_CONTACT_LOOP_AUTHORITY_INVALID:{joint_name}"
                )
            body0 = str(body0_targets[0]).rsplit("/", 1)[-1]
            body1 = str(body1_targets[0]).rsplit("/", 1)[-1]
            if body0 not in self._body_index or body1 not in self._body_index:
                raise RuntimeError(
                    f"PASSIVE_CONTACT_LOOP_BODY_MAPPING_MISSING:{joint_name}"
                )
            descriptors.append(
                {
                    "joint_name": joint_name,
                    "body0": body0,
                    "body1": body1,
                    "body0_index": self._body_index[body0],
                    "body1_index": self._body_index[body1],
                    "local_pos0_m": np.asarray(local0, dtype=np.float64),
                    "local_pos1_m": np.asarray(local1, dtype=np.float64),
                }
            )
        return descriptors

    @staticmethod
    def _cpu_row(value: Any) -> np.ndarray:
        array = _tensor(value).detach().to("cpu").numpy().astype(np.float64)
        return array[0] if array.ndim >= 2 and array.shape[0] == 1 else array

    @staticmethod
    def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        xyz = quaternion[:3]
        scalar = quaternion[3]
        return vector + 2.0 * (
            scalar * np.cross(xyz, vector)
            + np.cross(xyz, np.cross(xyz, vector))
        )

    @classmethod
    def _quat_matrix_xyzw(cls, quaternion: np.ndarray) -> np.ndarray:
        basis = np.eye(3, dtype=np.float64)
        return np.stack(
            [cls._quat_rotate_xyzw(quaternion, axis) for axis in basis],
            axis=1,
        )

    def _loop_anchor_residual(self) -> np.ndarray:
        # ``body_link_pose_w`` is the same direct PhysX buffer and XYZW
        # convention used by the previously validated passive-chain hierarchy
        # probe.  Do not combine the public WXYZ quaternion cache with these
        # authored local anchors.
        body_pose = self._cpu_row(self.robot.data.body_link_pose_w)
        body_position = body_pose[:, :3]
        body_quaternion = body_pose[:, 3:7]
        residuals: list[np.ndarray] = []
        for item in self._loop_descriptors:
            index0 = int(item["body0_index"])
            index1 = int(item["body1_index"])
            anchor0 = body_position[index0] + self._quat_rotate_xyzw(
                body_quaternion[index0], item["local_pos0_m"]
            )
            anchor1 = body_position[index1] + self._quat_rotate_xyzw(
                body_quaternion[index1], item["local_pos1_m"]
            )
            residuals.append(anchor1 - anchor0)
        return np.stack(residuals)

    def _mimic_residual(self, q: np.ndarray, qdot: np.ndarray) -> tuple[float, float]:
        # Production/E1 authority uses the legacy PhysX mimic relation
        # q_inner + q_outer = 0 for joint1.  This is a measured diagnostic,
        # never an independently commanded passive target.
        inner = self._joint_index["idx71_gripper_r_inner_joint1"]
        outer = self._joint_index["idx81_gripper_r_outer_joint1"]
        return float(q[inner] + q[outer]), float(qdot[inner] + qdot[outer])

    def _pad_body_state(self, side: str) -> dict[str, Any]:
        source = self._contact_sources[side]
        body_id = int(source["body_index"])
        link_pose = self._cpu_row(self.robot.data.body_link_pose_w)[body_id]
        com_pose = self._cpu_row(self.robot.data.body_com_pose_w)[body_id]
        link_velocity = self._cpu_row(self.robot.data.body_link_vel_w)[body_id]
        com_velocity = self._cpu_row(self.robot.data.body_com_vel_w)[body_id]
        link_rotation_world = self._quat_matrix_xyzw(link_pose[3:7])
        inertia_world = (
            link_rotation_world
            @ self._body_inertia_link_kg_m2[body_id]
            @ link_rotation_world.T
        )
        return {
            "body_index": body_id,
            "link_pose_world_xyzw": link_pose,
            "com_local_pose_xyzw": self._body_com_local_pose_xyzw[body_id],
            "com_world_pose_xyzw": com_pose,
            "link_linear_velocity_world_m_s": link_velocity[:3],
            "link_angular_velocity_world_rad_s": link_velocity[3:6],
            "com_linear_velocity_world_m_s": com_velocity[:3],
            "com_angular_velocity_world_rad_s": com_velocity[3:6],
            "mass_kg": float(self._body_mass_kg[body_id]),
            "inertia_link_kg_m2": self._body_inertia_link_kg_m2[body_id],
            "inertia_world_kg_m2": inertia_world,
            "principal_moments_kg_m2": self._body_principal_moments_kg_m2[
                body_id
            ],
            "principal_axes_link": self._body_principal_axes_link[body_id],
        }

    def _raw_pad_contact(
        self, side: str, dt_s: float, body_state: Mapping[str, Any]
    ) -> dict[str, Any]:
        source = self._contact_sources[side]
        values = tuple(source["view"].get_contact_data(dt=dt_s))
        if len(values) != 6:
            raise RuntimeError(
                f"PASSIVE_CONTACT_RAW_BUFFER_COUNT_INVALID:{side}:{len(values)}"
            )
        forces, points, normals, separations, counts, starts = values
        normal_force = (
            _tensor(forces)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1)
        )
        point = _tensor(points).detach().to("cpu").numpy().astype(np.float64).reshape(-1, 3)
        normal = _tensor(normals).detach().to("cpu").numpy().astype(np.float64).reshape(-1, 3)
        separation = _tensor(separations).detach().to("cpu").numpy().astype(np.float64).reshape(-1)
        count_values = _tensor(counts).detach().to("cpu").numpy().astype(np.int64).reshape(-1)
        start_values = _tensor(starts).detach().to("cpu").numpy().astype(np.int64).reshape(-1)
        pair_index = int(source["object_filter_pair_index"])
        if (
            count_values.size != start_values.size
            or count_values.size <= pair_index
        ):
            raise RuntimeError(
                f"PASSIVE_CONTACT_OBJECT_FILTER_PAIR_MISSING:{side}:"
                f"{count_values.size}:{start_values.size}"
            )
        count = int(count_values[pair_index])
        start = int(start_values[pair_index])
        if count < 0 or count > self.CONTACT_CAPACITY:
            raise RuntimeError(
                f"PASSIVE_CONTACT_CAPACITY_EXCEEDED:{side}:{count}"
            )
        if start < 0 or start + count > normal_force.shape[0]:
            raise RuntimeError(f"PASSIVE_CONTACT_RAW_SLICE_INVALID:{side}")
        selected = slice(start, start + count)
        normal_force = normal_force[selected]
        point = point[selected]
        normal = normal[selected]
        separation = separation[selected]

        # PhysX documents ``dt`` as the divisor used to convert its solver
        # impulses into forces.  Reading the same public buffer with dt=1.0
        # therefore preserves the native impulse value numerically.  This is
        # kept separate from the force*actual_dt diagnostic below.
        native_values = tuple(source["view"].get_contact_data(dt=1.0))
        if len(native_values) != 6:
            raise RuntimeError(
                f"PASSIVE_CONTACT_NATIVE_BUFFER_COUNT_INVALID:"
                f"{side}:{len(native_values)}"
            )
        (
            native_normal_raw,
            native_point_raw,
            native_normal_raw_vector,
            _native_separation_raw,
            native_counts,
            native_starts,
        ) = native_values
        native_count_values = (
            _tensor(native_counts)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )
        native_start_values = (
            _tensor(native_starts)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )
        if (
            native_count_values.size != native_start_values.size
            or native_count_values.size <= pair_index
            or int(native_count_values[pair_index]) != count
            or int(native_start_values[pair_index]) != start
        ):
            raise RuntimeError(f"PASSIVE_CONTACT_NATIVE_PAIR_MISMATCH:{side}")
        native_selected = slice(start, start + count)
        native_normal_impulse = (
            _tensor(native_normal_raw)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1)[native_selected]
        )
        native_points = (
            _tensor(native_point_raw)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)[native_selected]
        )
        native_normals = (
            _tensor(native_normal_raw_vector)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)[native_selected]
        )
        if count and (
            not np.allclose(native_points, point, atol=1.0e-9, rtol=0.0)
            or not np.allclose(native_normals, normal, atol=1.0e-9, rtol=0.0)
        ):
            raise RuntimeError(f"PASSIVE_CONTACT_NATIVE_POINT_DRIFT:{side}")
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        unit_normal = np.divide(
            normal,
            normal_norm,
            out=np.zeros_like(normal),
            where=normal_norm > 1.0e-12,
        )
        normal_force_vector = normal_force[:, None] * unit_normal
        friction_values = tuple(source["view"].get_friction_data(dt=dt_s))
        if len(friction_values) != 4:
            raise RuntimeError(
                f"PASSIVE_CONTACT_FRICTION_BUFFER_COUNT_INVALID:"
                f"{side}:{len(friction_values)}"
            )
        friction_force_raw, friction_point_raw, friction_counts, friction_starts = (
            friction_values
        )
        friction_count_values = (
            _tensor(friction_counts)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )
        friction_start_values = (
            _tensor(friction_starts)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )
        if (
            friction_count_values.size != friction_start_values.size
            or friction_count_values.size <= pair_index
        ):
            raise RuntimeError(f"PASSIVE_CONTACT_FRICTION_PAIR_MISMATCH:{side}")
        friction_count = int(friction_count_values[pair_index])
        if friction_count < 0 or friction_count > self.CONTACT_CAPACITY:
            raise RuntimeError(
                f"PASSIVE_CONTACT_FRICTION_CAPACITY_EXCEEDED:"
                f"{side}:{friction_count}"
            )
        friction_start = int(friction_start_values[pair_index])
        friction_force_all = (
            _tensor(friction_force_raw)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)
        )
        friction_point_all = (
            _tensor(friction_point_raw)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)
        )
        if (
            friction_start < 0
            or friction_start + friction_count > friction_force_all.shape[0]
        ):
            raise RuntimeError(f"PASSIVE_CONTACT_FRICTION_SLICE_INVALID:{side}")
        friction_selected = slice(
            friction_start, friction_start + friction_count
        )
        tangential_force = friction_force_all[friction_selected]
        friction_point = friction_point_all[friction_selected]

        native_friction_values = tuple(
            source["view"].get_friction_data(dt=1.0)
        )
        if len(native_friction_values) != 4:
            raise RuntimeError(
                f"PASSIVE_CONTACT_NATIVE_FRICTION_BUFFER_COUNT_INVALID:"
                f"{side}:{len(native_friction_values)}"
            )
        (
            native_tangential_raw,
            native_friction_point_raw,
            native_friction_counts,
            native_friction_starts,
        ) = native_friction_values
        native_friction_count_values = (
            _tensor(native_friction_counts)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )
        native_friction_start_values = (
            _tensor(native_friction_starts)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )
        if (
            native_friction_count_values.size
            != native_friction_start_values.size
            or native_friction_count_values.size <= pair_index
            or int(native_friction_count_values[pair_index]) != friction_count
            or int(native_friction_start_values[pair_index]) != friction_start
        ):
            raise RuntimeError(
                f"PASSIVE_CONTACT_NATIVE_FRICTION_PAIR_MISMATCH:{side}"
            )
        native_friction_selected = slice(
            friction_start, friction_start + friction_count
        )
        native_tangential_impulse = (
            _tensor(native_tangential_raw)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)[native_friction_selected]
        )
        native_friction_points = (
            _tensor(native_friction_point_raw)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)[native_friction_selected]
        )
        if friction_count and not np.allclose(
            native_friction_points, friction_point, atol=1.0e-9, rtol=0.0
        ):
            raise RuntimeError(
                f"PASSIVE_CONTACT_NATIVE_FRICTION_POINT_DRIFT:{side}"
            )

        raw_native_available = False
        raw_native_count = 0
        raw_native_other_actor_ids = np.zeros(
            self.CONTACT_CAPACITY, dtype=np.uint64
        )
        get_raw_contact_data = getattr(
            source["view"], "get_raw_contact_data", None
        )
        if callable(get_raw_contact_data):
            try:
                raw_native_values = tuple(get_raw_contact_data(dt=1.0))
                if len(raw_native_values) == 7:
                    raw_counts = (
                        _tensor(raw_native_values[4])
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.int64)
                        .reshape(-1)
                    )
                    raw_starts = (
                        _tensor(raw_native_values[5])
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.int64)
                        .reshape(-1)
                    )
                    if (
                        raw_counts.size == raw_starts.size
                        and raw_counts.size > pair_index
                    ):
                        raw_start = int(raw_starts[pair_index])
                        raw_native_count = min(
                            int(raw_counts[pair_index]), self.CONTACT_CAPACITY
                        )
                        raw_ids = (
                            _tensor(raw_native_values[6])
                            .detach()
                            .to("cpu")
                            .numpy()
                            .astype(np.uint64)
                            .reshape(-1)
                        )
                        raw_native_other_actor_ids[:raw_native_count] = raw_ids[
                            raw_start : raw_start + raw_native_count
                        ]
                        raw_native_available = True
            except Exception:
                # Actor IDs are useful provenance but not required for the
                # filtered Object-only impulse authority.  Do not reinterpret
                # an unavailable public raw-ID path as a zero contact.
                raw_native_available = False

        body_id = int(source["body_index"])
        body_position = np.asarray(body_state["link_pose_world_xyzw"][:3])
        body_linear = np.asarray(
            body_state["link_linear_velocity_world_m_s"]
        )
        body_angular = np.asarray(
            body_state["link_angular_velocity_world_rad_s"]
        )
        cube = self.env.scene["object"]
        cube_position = self._cpu_row(cube.data.root_pos_w)
        cube_linear = self._cpu_row(cube.data.root_lin_vel_w)
        cube_angular = self._cpu_row(cube.data.root_ang_vel_w)
        if count:
            pad_point_velocity = body_linear + np.cross(
                np.broadcast_to(body_angular, (count, 3)),
                point - body_position,
            )
            cube_point_velocity = cube_linear + np.cross(
                np.broadcast_to(cube_angular, (count, 3)),
                point - cube_position,
            )
            relative_velocity = cube_point_velocity - pad_point_velocity
            relative_normal_velocity = np.sum(
                relative_velocity * unit_normal, axis=1
            )
            tangential_velocity = (
                relative_velocity
                - relative_normal_velocity[:, None] * unit_normal
            )
            slip_speed = np.linalg.norm(tangential_velocity, axis=1)
        else:
            relative_normal_velocity = np.empty((0,), dtype=np.float64)
            tangential_velocity = np.empty((0, 3), dtype=np.float64)
            slip_speed = np.empty((0,), dtype=np.float64)

        com_world = np.asarray(body_state["com_world_pose_xyzw"][:3])
        lever_arm = point - com_world
        lever_arm_norm = np.linalg.norm(lever_arm, axis=1)
        native_normal_impulse_vector = native_normal_impulse[:, None] * unit_normal
        normal_angular_impulse = np.cross(
            lever_arm, native_normal_impulse_vector
        )
        friction_lever_arm = native_friction_points - com_world
        tangential_angular_impulse = np.cross(
            friction_lever_arm, native_tangential_impulse
        )
        total_angular_impulse = np.sum(normal_angular_impulse, axis=0)
        if friction_count:
            total_angular_impulse += np.sum(tangential_angular_impulse, axis=0)
        inertia_world = np.asarray(body_state["inertia_world_kg_m2"])
        expected_isolated_delta_omega = np.linalg.pinv(
            0.5 * (inertia_world + inertia_world.T), rcond=1.0e-12
        ) @ total_angular_impulse

        def padded(values: np.ndarray, width: int | None = None) -> np.ndarray:
            shape = (
                (self.CONTACT_CAPACITY,)
                if width is None
                else (self.CONTACT_CAPACITY, width)
            )
            result = np.full(shape, np.nan, dtype=np.float64)
            result[:count] = values
            return result

        def padded_friction(
            values: np.ndarray, width: int | None = None
        ) -> np.ndarray:
            shape = (
                (self.CONTACT_CAPACITY,)
                if width is None
                else (self.CONTACT_CAPACITY, width)
            )
            result = np.full(shape, np.nan, dtype=np.float64)
            result[:friction_count] = values
            return result

        return {
            "count": count,
            "friction_count": friction_count,
            "point_world_m": padded(point, 3),
            "normal_world": padded(unit_normal, 3),
            "separation_m": padded(separation),
            "penetration_depth_m": padded(np.maximum(-separation, 0.0)),
            "normal_force_vector_n": padded(normal_force_vector, 3),
            "normal_force_n": padded(normal_force),
            "native_normal_impulse_vector_ns": padded(
                native_normal_impulse_vector, 3
            ),
            "native_normal_impulse_ns": padded(native_normal_impulse),
            "tangential_force_vector_n": padded_friction(
                tangential_force, 3
            ),
            "native_tangential_impulse_vector_ns": padded_friction(
                native_tangential_impulse, 3
            ),
            "friction_point_world_m": padded_friction(friction_point, 3),
            # The installed API exposes a force buffer after dt scaling.  The
            # following are explicit derived impulse proxies, not a native
            # per-contact solver impulse readback.
            "derived_normal_impulse_ns": padded(normal_force * dt_s),
            "derived_tangential_impulse_vector_ns": padded_friction(
                tangential_force * dt_s, 3
            ),
            "native_minus_derived_normal_impulse_ns": padded(
                native_normal_impulse - normal_force * dt_s
            ),
            "lever_arm_world_m": padded(lever_arm, 3),
            "lever_arm_norm_m": padded(lever_arm_norm),
            "normal_angular_impulse_tendency_world_nms": padded(
                normal_angular_impulse, 3
            ),
            "friction_lever_arm_world_m": padded_friction(
                friction_lever_arm, 3
            ),
            "tangential_angular_impulse_tendency_world_nms": padded_friction(
                tangential_angular_impulse, 3
            ),
            "total_angular_impulse_tendency_world_nms": total_angular_impulse,
            "expected_isolated_delta_omega_world_rad_s": (
                expected_isolated_delta_omega
            ),
            "relative_normal_velocity_m_s": padded(relative_normal_velocity),
            "tangential_relative_velocity_m_s": padded(tangential_velocity, 3),
            "slip_speed_m_s": padded(slip_speed),
            "raw_native_actor_ids_available": raw_native_available,
            "raw_native_contact_count": raw_native_count,
            "raw_native_other_actor_ids": raw_native_other_actor_ids,
            "body_state": dict(body_state),
        }

    @property
    def post_first_contact_duration_s(self) -> float:
        if self._first_any_contact_sample is None:
            return 0.0
        return max(
            0.0,
            (len(self.records) - 1 - self._first_any_contact_sample)
            * float(self.records[-1]["dt_s"]),
        )

    def set_context(
        self,
        *,
        policy_step: int,
        bc_step: int | None,
        phase: str,
        gripper_intent: str,
        close_onset: bool,
        clipping: bool,
        last_real_sensor_timestamps: Mapping[str, float] | None = None,
        physics_clock_anchor_s: float | None = None,
    ) -> None:
        sensor_timestamps = dict(last_real_sensor_timestamps or {})
        if sensor_timestamps:
            newest_sensor_s = max(float(value) for value in sensor_timestamps.values())
            if not math.isfinite(newest_sensor_s) or newest_sensor_s < 0.0:
                raise RuntimeError("NONFINITE_LAST_REAL_SENSOR_TIMESTAMP")
        if physics_clock_anchor_s is not None:
            anchor_s = float(physics_clock_anchor_s)
            if not math.isfinite(anchor_s) or anchor_s < 0.0:
                raise RuntimeError("NONFINITE_PHYSICS_CLOCK_ANCHOR")
            if sensor_timestamps and anchor_s + 1.0e-6 < newest_sensor_s:
                raise RuntimeError("PHYSICS_CLOCK_ANCHOR_BEFORE_SENSOR_CAPTURE")
            # The capture timestamp identifies the actual reused RGB-D frame;
            # the sensor's current timestamp (capture + measured age) is the
            # clock-origin authority for the following physics interval.  Do
            # not turn a stale 25-Hz frame into a synthetic fresh timestamp.
            self._physics_elapsed_s = max(self._physics_elapsed_s, anchor_s)
        elif sensor_timestamps:
            # Existing non-Stage-1A diagnostics do not expose a sensor-current
            # clock.  Preserve their historical capture-time alignment.
            self._physics_elapsed_s = max(self._physics_elapsed_s, newest_sensor_s)
        self.context = {
            "policy_step": int(policy_step),
            "bc_step": -1 if bc_step is None else int(bc_step),
            "phase": str(phase),
            "gripper_intent": str(gripper_intent),
            "close_onset": bool(close_onset),
            "clipping": bool(clipping),
            "last_real_sensor_timestamps": sensor_timestamps,
        }
        self._substep_in_policy = 0

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("PASSIVE_CONTACT_TELEMETRY_ALREADY_INSTALLED")

        def wrapped_update(dt: float):
            result = self.original_update(dt)
            if self.context is not None:
                self._capture(float(dt))
            return result

        self.env.scene.update = wrapped_update
        self.installed = True

    def reset_episode_buffers(self) -> None:
        """Clear read-only episode history without touching PhysX mechanics.

        The environment reset and direct-state initializer own physical state.
        This method only prevents contact/qdot evidence from one supervision
        trial leaking into the next trial in a persistent Isaac session.
        """

        self.context = None
        self.records.clear()
        self._substep_in_policy = 0
        self._first_any_contact_sample = None
        self._previous_qdot = None
        self._hard_stop_event = None
        self._hard_stop_receipt = None
        self._physics_elapsed_s = 0.0

    def _optional_joint_row(self, attribute: str) -> tuple[np.ndarray, bool]:
        value = getattr(self.robot.data, attribute, None)
        if value is None:
            return np.full(len(self.robot.joint_names), np.nan), False
        try:
            result = self._cpu_row(value)
        except Exception:
            return np.full(len(self.robot.joint_names), np.nan), False
        if result.shape != (len(self.robot.joint_names),):
            return np.full(len(self.robot.joint_names), np.nan), False
        return result, bool(np.isfinite(result).all())

    def _incoming_joint_wrench(self) -> tuple[np.ndarray, bool]:
        expected = (len(self.robot.body_names), 6)
        getter = getattr(
            self.robot.root_view, "get_link_incoming_joint_force", None
        )
        if not callable(getter):
            return np.full(expected, np.nan, dtype=np.float64), False
        try:
            result = self._cpu_row(getter())
        except Exception:
            return np.full(expected, np.nan, dtype=np.float64), False
        if result.shape != expected:
            return np.full(expected, np.nan, dtype=np.float64), False
        return result, bool(np.isfinite(result).all())

    def _physics_source_clock_s(self) -> float:
        """Read Isaac's current sensor clock on the physics sample epoch."""

        camera = self.env.scene["right_wrist_camera"]
        captured, age = camera_capture_time_and_age(camera)
        captured_array = np.asarray(self._cpu_row(captured), dtype=np.float64).reshape(-1)
        age_array = np.asarray(self._cpu_row(age), dtype=np.float64).reshape(-1)
        if captured_array.size != 1 or age_array.size != 1:
            raise RuntimeError("PHYSICS_SOURCE_CLOCK_CARDINALITY_MISMATCH")
        captured_s = float(captured_array[0])
        age_s = float(age_array[0])
        current_s = captured_s + age_s
        if (
            not math.isfinite(captured_s)
            or not math.isfinite(age_s)
            or age_s < 0.0
            or not math.isfinite(current_s)
            or current_s < 0.0
        ):
            raise RuntimeError("PHYSICS_SOURCE_CLOCK_INVALID")
        return current_s

    def _capture(self, dt_s: float) -> None:
        assert self.context is not None
        # Use the source-owned Isaac sensor current clock at this exact scene
        # update.  Accumulating Python float dt alongside Isaac's float32
        # sensor clock creates microsecond drift and can falsely place a real
        # 25-Hz acquisition after its enclosing 500-Hz physics sample.
        source_clock_s = self._physics_source_clock_s()
        if source_clock_s + 1.0e-6 < self._physics_elapsed_s:
            raise RuntimeError("PHYSICS_SOURCE_CLOCK_REGRESSED")
        self._physics_elapsed_s = source_clock_s
        q = self._cpu_row(self.robot.data.joint_pos)
        qdot = self._cpu_row(self.robot.data.joint_vel)
        root_position_world_m = self._cpu_row(self.robot.data.root_pos_w)
        root_quat_world_xyzw = self._cpu_row(
            quaternion_native_to_xyzw(
                _tensor(self.robot.data.root_quat_w),
                isaaclab_native_quaternion_order(),
            )
        )
        root_linear_velocity_world_m_s = self._cpu_row(
            self.robot.data.root_lin_vel_w
        )
        root_angular_velocity_world_rad_s = self._cpu_row(
            self.robot.data.root_ang_vel_w
        )
        projected_joint_effort = self._cpu_row(
            self.robot.root_view.get_dof_projected_joint_forces()
        )
        actuation_force = self._cpu_row(
            self.robot.root_view.get_dof_actuation_forces()
        )
        target_q, target_q_available = self._optional_joint_row("joint_pos_target")
        target_qdot, target_qdot_available = self._optional_joint_row(
            "joint_vel_target"
        )
        inner, outer, bilateral, _slip, stable = (
            self.task_mdp.contact_grasp_telemetry(self.env)
        )
        evaluator = getattr(self.env, "_g2_forbidden_collision_evaluator", None)
        sensor_peaks = (
            evaluator.sensor_peak_forces_n() if evaluator is not None else {}
        )
        forbidden_peak = (
            max(max(values) for values in sensor_peaks.values())
            if sensor_peaks
            else 0.0
        )
        body_inner = self._pad_body_state("inner")
        body_outer = self._pad_body_state("outer")
        raw_inner = self._raw_pad_contact("inner", dt_s, body_inner)
        raw_outer = self._raw_pad_contact("outer", dt_s, body_outer)
        body_outer_link2 = self._pad_body_state("outer_link2")
        raw_outer_link2 = self._raw_pad_contact(
            "outer_link2", dt_s, body_outer_link2
        )
        incoming_joint_wrench, incoming_joint_wrench_available = (
            self._incoming_joint_wrench()
        )
        any_contact = (
            raw_inner["count"] > 0
            or raw_outer["count"] > 0
            or raw_outer_link2["count"] > 0
        )
        if any_contact and self._first_any_contact_sample is None:
            self._first_any_contact_sample = len(self.records)
        loop_residual = self._loop_anchor_residual()
        mimic_position_residual, mimic_velocity_residual = self._mimic_residual(
            q, qdot
        )
        limit_margin = np.minimum(
            q - self._joint_limits[:, 0], self._joint_limits[:, 1] - q
        )
        qdd = np.full_like(qdot, np.nan)
        if not self._defer_qdd_to_close_mechanics:
            if self._previous_qdot is not None:
                qdd = (qdot - self._previous_qdot) / dt_s
            self._previous_qdot = qdot.copy()
        hard_stop_mask = (
            np.isfinite(qdd)
            & (limit_margin <= HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD)
            & (np.abs(qdd) > HARD_STOP_ACCELERATION_LIMIT_RAD_S2)
        )
        self.records.append(
            {
                **self.context,
                "global_physics_sample": len(self.records),
                "physics_substep": self._substep_in_policy,
                "dt_s": float(dt_s),
                "physics_timestamp_s": float(self._physics_elapsed_s),
                "q_rad": q,
                "raw_qdot_rad_s": qdot,
                "root_position_world_m": root_position_world_m,
                "root_quat_world_xyzw": root_quat_world_xyzw,
                "root_linear_velocity_world_m_s": root_linear_velocity_world_m_s,
                "root_angular_velocity_world_rad_s": root_angular_velocity_world_rad_s,
                "projected_joint_effort": projected_joint_effort,
                "actuation_force": actuation_force,
                "target_q_rad": target_q,
                "target_q_available": target_q_available,
                "target_qdot_rad_s": target_qdot,
                "target_qdot_available": target_qdot_available,
                "inner_contact_force_n": float(inner.reshape(-1)[0].item()),
                "outer_contact_force_n": float(outer.reshape(-1)[0].item()),
                "bilateral_contact": bool(bilateral.reshape(-1)[0].item()),
                "stable_contact": bool(stable.reshape(-1)[0].item()),
                "forbidden_contact_force_n": float(forbidden_peak),
                "forbidden_collision": bool(forbidden_peak > 0.0),
                "controller_reject": False,
                "raw_inner": raw_inner,
                "raw_outer": raw_outer,
                "raw_outer_link2": raw_outer_link2,
                "incoming_joint_wrench_child_frame": incoming_joint_wrench,
                "incoming_joint_wrench_available": (
                    incoming_joint_wrench_available
                ),
                "loop_anchor_residual_vector_m": loop_residual,
                "mimic_position_residual_rad": mimic_position_residual,
                "mimic_velocity_residual_rad_s": mimic_velocity_residual,
                "joint_limit_margin_rad": limit_margin,
                "fd_qdd_rad_s2": qdd,
            }
        )
        self._substep_in_policy += 1
        if self._hard_stop_event is None and bool(np.any(hard_stop_mask)):
            joint_index = int(np.argmax(np.where(hard_stop_mask, np.abs(qdd), -1.0)))
            self._hard_stop_event = {
                "legacy_failure_family": "CUROBO_AB_HARD_STOP_EVENT",
                "global_physics_sample": len(self.records) - 1,
                "policy_step": int(self.context["policy_step"]),
                "bc_step": int(self.context["bc_step"]),
                "physics_substep": self._substep_in_policy - 1,
                "phase": str(self.context["phase"]),
                "joint_index": joint_index,
                "joint_name": self.robot.joint_names[joint_index],
                "q_rad": float(q[joint_index]),
                "qdot_rad_s": float(qdot[joint_index]),
                "fd_qdd_rad_s2": float(qdd[joint_index]),
                "joint_limit_margin_rad": float(limit_margin[joint_index]),
                "limit_numerical_tolerance_rad": HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD,
                "acceleration_hard_limit_rad_s2": HARD_STOP_ACCELERATION_LIMIT_RAD_S2,
            }
            idx83_index = self._joint_index["idx83_gripper_r_outer_joint4"]
            sensor_timestamps = dict(
                self.context.get("last_real_sensor_timestamps", {})
            )
            self._hard_stop_receipt = RuntimeHardstopReceipt(
                physics_step_index=self._substep_in_policy,
                physics_substep_index=self._substep_in_policy - 1,
                control_step_index=int(self.context["policy_step"]),
                consumed_substeps=self._substep_in_policy,
                nominal_substeps=10,
                terminal_physics_timestamp_s=float(self._physics_elapsed_s),
                q_rad=tuple(float(value) for value in q),
                qd_rad_s=tuple(float(value) for value in qdot),
                qdd_rad_s2=tuple(float(value) for value in qdd),
                idx83_limit_margin_rad=float(limit_margin[idx83_index]),
                hardstop_joint=str(self.robot.joint_names[joint_index]),
                hardstop_joint_index=joint_index,
                hardstop_reason=RUNTIME_HARDSTOP_REASON,
                contact=bool(
                    raw_inner["count"] > 0
                    or raw_outer["count"] > 0
                    or raw_outer_link2["count"] > 0
                ),
                bilateral=bool(bilateral.reshape(-1)[0].item()),
                stable=bool(stable.reshape(-1)[0].item()),
                last_real_sensor_timestamps=sensor_timestamps,
            )
            if self._stop_on_hard_stop:
                raise RuntimeHardstop(self._hard_stop_receipt)

    @property
    def hard_stop_event(self) -> Mapping[str, Any] | None:
        return None if self._hard_stop_event is None else dict(self._hard_stop_event)

    @property
    def hard_stop_receipt(self) -> RuntimeHardstopReceipt | None:
        return self._hard_stop_receipt

    def save(self, output: Path) -> dict[str, Any]:
        if not self.records:
            raise RuntimeError("PASSIVE_CONTACT_TELEMETRY_EMPTY")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}.npz")
        scalar_fields = {
            "global_physics_sample": np.int64,
            "policy_step": np.int64,
            "bc_step": np.int64,
            "physics_substep": np.int64,
            "dt_s": np.float64,
            "physics_timestamp_s": np.float64,
            "phase": "U32",
            "gripper_intent": "U8",
            "close_onset": np.bool_,
            "clipping": np.bool_,
            "target_q_available": np.bool_,
            "target_qdot_available": np.bool_,
            "inner_contact_force_n": np.float64,
            "outer_contact_force_n": np.float64,
            "bilateral_contact": np.bool_,
            "stable_contact": np.bool_,
            "forbidden_contact_force_n": np.float64,
            "forbidden_collision": np.bool_,
            "controller_reject": np.bool_,
            "incoming_joint_wrench_available": np.bool_,
        }
        arrays: dict[str, Any] = {
            name: np.asarray([row[name] for row in self.records], dtype=dtype)
            for name, dtype in scalar_fields.items()
        }
        for name in (
            "q_rad",
            "raw_qdot_rad_s",
            "projected_joint_effort",
            "actuation_force",
            "target_q_rad",
            "target_qdot_rad_s",
            "fd_qdd_rad_s2",
            "incoming_joint_wrench_child_frame",
        ):
            arrays[name] = np.stack([row[name] for row in self.records])
        arrays["joint_names"] = np.asarray(self.robot.joint_names, dtype="U64")
        arrays["body_names"] = np.asarray(self.robot.body_names, dtype="U64")
        arrays["joint_limit_margin_rad"] = np.stack(
            [row["joint_limit_margin_rad"] for row in self.records]
        )
        arrays["loop_joint_names"] = np.asarray(
            [item["joint_name"] for item in self._loop_descriptors], dtype="U64"
        )
        arrays["loop_anchor_residual_vector_m"] = np.stack(
            [row["loop_anchor_residual_vector_m"] for row in self.records]
        )
        arrays["mimic_position_residual_rad"] = np.asarray(
            [row["mimic_position_residual_rad"] for row in self.records],
            dtype=np.float64,
        )
        arrays["mimic_velocity_residual_rad_s"] = np.asarray(
            [row["mimic_velocity_residual_rad_s"] for row in self.records],
            dtype=np.float64,
        )
        for side in ("inner", "outer", "outer_link2"):
            key = f"raw_{side}"
            arrays[f"{side}_contact_point_count"] = np.asarray(
                [row[key]["count"] for row in self.records], dtype=np.int64
            )
            arrays[f"{side}_friction_point_count"] = np.asarray(
                [row[key]["friction_count"] for row in self.records],
                dtype=np.int64,
            )
            for field in (
                "point_world_m",
                "normal_world",
                "separation_m",
                "penetration_depth_m",
                "normal_force_vector_n",
                "normal_force_n",
                "native_normal_impulse_vector_ns",
                "native_normal_impulse_ns",
                "tangential_force_vector_n",
                "native_tangential_impulse_vector_ns",
                "friction_point_world_m",
                "derived_normal_impulse_ns",
                "derived_tangential_impulse_vector_ns",
                "native_minus_derived_normal_impulse_ns",
                "lever_arm_world_m",
                "lever_arm_norm_m",
                "normal_angular_impulse_tendency_world_nms",
                "friction_lever_arm_world_m",
                "tangential_angular_impulse_tendency_world_nms",
                "relative_normal_velocity_m_s",
                "tangential_relative_velocity_m_s",
                "slip_speed_m_s",
            ):
                arrays[f"{side}_{field}"] = np.stack(
                    [row[key][field] for row in self.records]
                )
            for field in (
                "total_angular_impulse_tendency_world_nms",
                "expected_isolated_delta_omega_world_rad_s",
            ):
                arrays[f"{side}_{field}"] = np.stack(
                    [row[key][field] for row in self.records]
                )
            arrays[f"{side}_raw_native_actor_ids_available"] = np.asarray(
                [
                    row[key]["raw_native_actor_ids_available"]
                    for row in self.records
                ],
                dtype=np.bool_,
            )
            arrays[f"{side}_raw_native_contact_count"] = np.asarray(
                [row[key]["raw_native_contact_count"] for row in self.records],
                dtype=np.int64,
            )
            arrays[f"{side}_raw_native_other_actor_ids"] = np.stack(
                [row[key]["raw_native_other_actor_ids"] for row in self.records]
            )
            body_state_fields = (
                "link_pose_world_xyzw",
                "com_local_pose_xyzw",
                "com_world_pose_xyzw",
                "link_linear_velocity_world_m_s",
                "link_angular_velocity_world_rad_s",
                "com_linear_velocity_world_m_s",
                "com_angular_velocity_world_rad_s",
                "inertia_link_kg_m2",
                "inertia_world_kg_m2",
                "principal_moments_kg_m2",
                "principal_axes_link",
            )
            for field in body_state_fields:
                arrays[f"{side}_pad_{field}"] = np.stack(
                    [row[key]["body_state"][field] for row in self.records]
                )
            arrays[f"{side}_pad_mass_kg"] = np.asarray(
                [row[key]["body_state"]["mass_kg"] for row in self.records],
                dtype=np.float64,
            )
            arrays[f"{side}_pad_body_index"] = np.asarray(
                [row[key]["body_state"]["body_index"] for row in self.records],
                dtype=np.int64,
            )
        arrays["contacting_pad_link_names"] = np.asarray(
            [self._contact_sources[side]["link_name"] for side in ("inner", "outer")],
            dtype="U64",
        )
        arrays["contact_partner_actor_paths"] = np.asarray(
            [self._contact_sources[side]["partner_actor"] for side in ("inner", "outer")],
            dtype="U128",
        )
        arrays["penetration_available"] = np.asarray([True], dtype=np.bool_)
        arrays["native_contact_impulse_available"] = np.asarray(
            [True], dtype=np.bool_
        )
        arrays["native_contact_impulse_authority"] = np.asarray(
            ["PHYSX_CONTACT_BUFFER_DT_ONE"], dtype="U64"
        )
        arrays["native_constraint_impulse_available"] = np.asarray(
            [False], dtype=np.bool_
        )
        arrays["native_loop_constraint_reaction_available"] = np.asarray(
            [False], dtype=np.bool_
        )
        arrays["joint_reaction_authority"] = np.asarray(
            ["PHYSX_LINK_INCOMING_JOINT_6D_WRENCH_CHILD_FRAME"], dtype="U96"
        )
        arrays["contact_force_times_dt_impulse_proxy"] = np.asarray(
            [True], dtype=np.bool_
        )
        arrays["first_any_contact_physics_sample"] = np.asarray(
            [
                -1
                if self._first_any_contact_sample is None
                else self._first_any_contact_sample
            ],
            dtype=np.int64,
        )
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
        return {
            "path": str(output),
            "sha256": _sha256(output),
            "physics_sample_count": len(self.records),
            "joint_count": len(self.robot.joint_names),
            "readback_stage": "AFTER_ORIGINAL_SCENE_UPDATE",
            "control_mutation": False,
            "penetration_available": True,
            "native_contact_impulse_available": True,
            "native_contact_impulse_authority": "PHYSX_CONTACT_BUFFER_DT_ONE",
            "native_constraint_impulse_available": False,
            "native_loop_constraint_reaction_available": False,
            "joint_reaction_available": bool(
                any(
                    bool(row["incoming_joint_wrench_available"])
                    for row in self.records
                )
            ),
            "joint_reaction_authority": (
                "PHYSX_LINK_INCOMING_JOINT_6D_WRENCH_CHILD_FRAME"
            ),
            "derived_force_times_dt_impulse_proxy": True,
            "first_any_contact_physics_sample": self._first_any_contact_sample,
            "post_first_contact_duration_s": self.post_first_contact_duration_s,
            "loop_joint_count": len(self._loop_descriptors),
            "contact_capacity_per_pad": self.CONTACT_CAPACITY,
            "hard_stop_event": self._hard_stop_event,
            "stop_on_hard_stop": self._stop_on_hard_stop,
            "qdd_callback_computation": (
                "DEFERRED_TO_CLOSE_MECHANICS_POST_STEP"
                if self._defer_qdd_to_close_mechanics
                else "LEGACY_2POINT_IN_CALLBACK"
            ),
        }

    def restore(self) -> None:
        if not self.installed:
            return
        self.env.scene.update = self.original_update
        self.installed = False


def _fd_series(samples: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.full_like(samples, np.nan, dtype=np.float64)
    acceleration = np.full_like(samples, np.nan, dtype=np.float64)
    velocity[1:] = np.diff(samples, axis=0) / dt_s
    acceleration[2:] = np.diff(velocity[1:], axis=0) / dt_s
    return velocity, acceleration


def _consecutive_true_run(mask: np.ndarray, index: int) -> tuple[int, int]:
    start = index
    while start > 0 and bool(mask[start - 1]):
        start -= 1
    end = index
    while end + 1 < mask.shape[0] and bool(mask[end + 1]):
        end += 1
    return start, end


def _save_target_attribution(
    *,
    output: Path,
    samples: list[dict[str, Any]],
    substep_rows: list[dict[str, Any]],
    joint_names: list[str],
    control_dt_s: float,
    physics_dt_s: float,
    hard_acceleration_limit_rad_s2: float,
) -> dict[str, Any]:
    if len(samples) < 3:
        raise RuntimeError("TARGET_ATTRIBUTION_INSUFFICIENT_POLICY_SAMPLES")
    policy_step = np.asarray([row["policy_step"] for row in samples], dtype=np.int64)
    timestamp_s = np.asarray([row["timestamp_s"] for row in samples], dtype=np.float64)
    waypoint_index = np.asarray([row["waypoint_index"] for row in samples], dtype=np.int64)
    boundary = np.asarray([row["waypoint_boundary"] for row in samples], dtype=np.bool_)
    repeat_index = np.asarray([row["waypoint_repeat_index"] for row in samples], dtype=np.int64)
    repeat_transition = np.asarray(
        [row["waypoint_repeat_transition"] for row in samples], dtype=np.bool_
    )
    phase = np.asarray([row["phase"] for row in samples], dtype="U24")
    training_valid = np.isin(
        phase,
        np.asarray(
            (
                "WAYPOINT_TRACKING",
                "COARSE_REACH",
                "FINE_APPROACH",
                "FINAL_CONVERGENCE",
            ),
            dtype="U24",
        ),
    )
    reset = np.asarray([row["limiter_reset"] for row in samples], dtype=np.bool_)
    synchronize = np.asarray(
        [row["limiter_synchronize"] for row in samples], dtype=np.bool_
    )
    controller_reject = np.asarray(
        [row["controller_reject"] for row in samples], dtype=np.bool_
    )
    clipping = np.asarray([row["clipping"] for row in samples], dtype=np.bool_)

    def stack(key: str) -> np.ndarray:
        return np.stack([np.asarray(row[key], dtype=np.float64) for row in samples])

    raw_q = stack("raw_planner_q")
    raw_prev_q = stack("previous_raw_planner_q")
    raw_next_q = stack("next_raw_planner_q")
    pre_q = stack("pre_rate_limit_target_q")
    post_q = stack("post_rate_limit_target_q")
    limiter_velocity = stack("limiter_accumulator_velocity")
    measured_q = stack("measured_q")
    measured_qdot = stack("measured_qdot")
    ee_position = stack("ee_position_root_m")
    ee_quaternion = stack("ee_quaternion_root_wxyz")
    planner_ee = stack("planner_ee_target_root_m")
    tracking_error = np.asarray(
        [row["ee_tracking_error_m"] for row in samples], dtype=np.float64
    )
    forbidden_force = np.asarray(
        [row["maximum_forbidden_contact_force_n"] for row in samples],
        dtype=np.float64,
    )

    post_velocity, post_acceleration = _fd_series(post_q, control_dt_s)
    pre_velocity, pre_acceleration = _fd_series(pre_q, control_dt_s)
    measured_fd_velocity, measured_fd_acceleration = _fd_series(
        measured_q, control_dt_s
    )
    previous_post_q = np.vstack((np.full((1, post_q.shape[1]), np.nan), post_q[:-1]))
    next_post_q = np.vstack((post_q[1:], np.full((1, post_q.shape[1]), np.nan)))

    finite_target_acceleration = np.nan_to_num(
        np.abs(post_acceleration), nan=-np.inf
    )
    peak_flat = int(np.argmax(finite_target_acceleration))
    peak_sample_raw, peak_joint_raw = np.unravel_index(
        peak_flat, finite_target_acceleration.shape
    )
    peak_sample = int(peak_sample_raw)
    peak_joint = int(peak_joint_raw)
    peak_value = float(post_acceleration[peak_sample, peak_joint])
    above = np.abs(post_acceleration[:, peak_joint]) > hard_acceleration_limit_rad_s2
    run_start, run_end = _consecutive_true_run(above, peak_sample)
    window_start = max(0, peak_sample - 5)
    window_end = min(len(samples) - 1, peak_sample + 5)

    substep_policy = np.asarray(
        [row["policy_step"] for row in substep_rows], dtype=np.int64
    )
    substep_index = np.asarray(
        [row["physics_substep_index"] for row in substep_rows], dtype=np.int64
    )
    substep_pre = np.stack(
        [np.asarray(row["pre_rate_limit_target"], dtype=np.float64) for row in substep_rows]
    )
    substep_post = np.stack(
        [np.asarray(row["post_limit_target"], dtype=np.float64) for row in substep_rows]
    )
    substep_final = np.stack(
        [np.asarray(row["final_emitted_target"], dtype=np.float64) for row in substep_rows]
    )
    substep_joint_clamp = np.asarray(
        [row["post_limiter_joint_clamp_applied"] for row in substep_rows],
        dtype=np.bool_,
    )
    substep_previous_velocity = np.stack(
        [
            np.asarray(row["previous_accumulator_velocity"], dtype=np.float64)
            for row in substep_rows
        ]
    )
    substep_velocity = np.stack(
        [
            np.asarray(row["accumulator_velocity_after"], dtype=np.float64)
            for row in substep_rows
        ]
    )
    substep_accumulator_qdd = (
        substep_velocity - substep_previous_velocity
    ) / physics_dt_s
    substep_final_fd_qdot, substep_final_fd_qdd = _fd_series(
        substep_final, physics_dt_s
    )
    substep_reset_generation = np.asarray(
        [row["reset_generation"] for row in substep_rows], dtype=np.int64
    )
    substep_synchronize_generation = np.asarray(
        [row["synchronize_generation"] for row in substep_rows], dtype=np.int64
    )
    substep_speed_scale = np.asarray(
        [float(np.asarray(row["synchronized_speed_scale"]).reshape(-1)[0]) for row in substep_rows],
        dtype=np.float64,
    )
    substep_common_endpoint_scale = np.asarray(
        [
            float(
                np.asarray(row["synchronized_common_endpoint_scale"]).reshape(-1)[0]
            )
            for row in substep_rows
        ],
        dtype=np.float64,
    )
    substep_per_joint_endpoint_scale = np.stack(
        [
            np.asarray(row["synchronized_per_joint_endpoint_scale"], dtype=np.float64)
            for row in substep_rows
        ]
    )
    substep_acceleration_scale = np.asarray(
        [
            float(
                np.asarray(row["synchronized_acceleration_scale"]).reshape(-1)[0]
            )
            for row in substep_rows
        ],
        dtype=np.float64,
    )
    substep_endpoint_deceleration_mask = np.stack(
        [
            np.asarray(
                row["synchronized_endpoint_deceleration_mask"], dtype=np.bool_
            )
            for row in substep_rows
        ]
    )

    event_substep_mask = (
        (substep_policy >= int(policy_step[max(0, peak_sample - 2)]))
        & (substep_policy <= int(policy_step[peak_sample]))
    )
    event_accumulator_qdd = np.abs(
        substep_accumulator_qdd[event_substep_mask, peak_joint]
    )
    event_final_fd_qdd = np.abs(
        substep_final_fd_qdd[event_substep_mask, peak_joint]
    )
    maximum_accumulator_qdd = float(event_accumulator_qdd.max(initial=0.0))
    finite_event_final_qdd = event_final_fd_qdd[np.isfinite(event_final_fd_qdd)]
    maximum_final_substep_qdd = float(
        finite_event_final_qdd.max(initial=0.0)
    )
    event_substep_indices = np.flatnonzero(event_substep_mask)
    causal_local_index = int(
        np.nanargmax(
            np.abs(substep_accumulator_qdd[event_substep_mask, peak_joint])
        )
    )
    causal_substep_index = int(event_substep_indices[causal_local_index])
    reset_near_event = bool(reset[max(0, peak_sample - 2) : peak_sample + 1].any())
    synchronize_near_event = bool(
        synchronize[max(0, peak_sample - 2) : peak_sample + 1].any()
    )
    joint_clamp_near_event = bool(substep_joint_clamp[event_substep_mask].any())
    boundary_near_event = bool(
        boundary[max(0, peak_sample - 2) : peak_sample + 1].any()
    )
    repeat_near_event = bool(
        repeat_transition[max(0, peak_sample - 2) : peak_sample + 1].any()
        or np.any(repeat_index[max(0, peak_sample - 2) : peak_sample + 1] > 1)
    )
    pre_spike = bool(
        np.isfinite(pre_acceleration[peak_sample, peak_joint])
        and abs(float(pre_acceleration[peak_sample, peak_joint]))
        > hard_acceleration_limit_rad_s2
    )
    measured_at_peak = float(measured_fd_acceleration[peak_sample, peak_joint])
    multi_joint_count = int(
        np.count_nonzero(
            np.abs(post_acceleration[peak_sample]) > hard_acceleration_limit_rad_s2
        )
    )
    collision_or_reject = bool(
        forbidden_force[peak_sample] > 1.0e-6 or controller_reject[peak_sample]
    )

    above_limit_policy_event_count = int(
        np.count_nonzero(
            np.any(
                np.abs(post_acceleration) > hard_acceleration_limit_rad_s2,
                axis=1,
            )
        )
    )
    above_limit_joint_sample_pair_count = int(
        np.count_nonzero(
            np.abs(post_acceleration) > hard_acceleration_limit_rad_s2
        )
    )
    limiter_state_discontinuity = bool(
        reset_near_event
        or synchronize_near_event
        or joint_clamp_near_event
        or maximum_accumulator_qdd > hard_acceleration_limit_rad_s2
    )
    if above_limit_policy_event_count == 0:
        root_cause = "NO_ABOVE_LIMIT_EVENT"
    elif limiter_state_discontinuity:
        root_cause = "LIMITER_STATE_DISCONTINUITY"
    elif repeat_near_event:
        root_cause = "RESAMPLING_OR_REPEAT_TRANSITION_SPIKE"
    elif (
        boundary_near_event
        and (run_end - run_start + 1) == 1
        and abs(measured_at_peak) <= hard_acceleration_limit_rad_s2
        and not collision_or_reject
        and multi_joint_count == 1
    ):
        root_cause = "ONE_STEP_WAYPOINT_BOUNDARY_SPIKE"
    elif (
        maximum_accumulator_qdd <= hard_acceleration_limit_rad_s2
        and maximum_final_substep_qdd <= hard_acceleration_limit_rad_s2
    ):
        root_cause = "FD_ESTIMATOR_ARTIFACT"
    elif (
        (run_end - run_start + 1) > 1
        or maximum_accumulator_qdd > hard_acceleration_limit_rad_s2
        or maximum_final_substep_qdd > hard_acceleration_limit_rad_s2
    ):
        root_cause = "REAL_TARGET_COMMAND_OVERSHOOT"
    else:
        root_cause = "UNRESOLVED"

    npz_payload = {
        "joint_names": np.asarray(joint_names, dtype="U32"),
        "policy_step": policy_step,
        "timestamp_s": timestamp_s,
        "control_dt_s": np.asarray([control_dt_s], dtype=np.float64),
        "physics_dt_s": np.asarray([physics_dt_s], dtype=np.float64),
        "planner_waypoint_index": waypoint_index,
        "waypoint_boundary_flag": boundary,
        "waypoint_repeat_index": repeat_index,
        "waypoint_repeat_transition_flag": repeat_transition,
        "phase": phase,
        "training_valid_mask": training_valid,
        "raw_planner_joint_target_q_rad": raw_q,
        "previous_raw_planner_target_q_rad": raw_prev_q,
        "next_raw_planner_target_q_rad": raw_next_q,
        "pre_rate_limit_target_q_rad": pre_q,
        "post_rate_limit_target_q_rad": post_q,
        "previous_post_rate_limit_target_q_rad": previous_post_q,
        "next_post_rate_limit_target_q_rad": next_post_q,
        "target_fd_qdot_rad_s": post_velocity,
        "target_fd_qdd_rad_s2": post_acceleration,
        "pre_limit_fd_qdot_rad_s": pre_velocity,
        "pre_limit_fd_qdd_rad_s2": pre_acceleration,
        "limiter_accumulator_velocity_rad_s": limiter_velocity,
        "limiter_reset_flag": reset,
        "limiter_synchronize_flag": synchronize,
        "commanded_qdot_rad_s": limiter_velocity,
        "measured_q_rad": measured_q,
        "measured_qdot_rad_s": measured_qdot,
        "measured_fd_qdot_rad_s": measured_fd_velocity,
        "measured_fd_qdd_rad_s2": measured_fd_acceleration,
        "controller_reject_flag": controller_reject,
        "clipping_flag": clipping,
        "ee_position_root_m": ee_position,
        "ee_quaternion_root_wxyz": ee_quaternion,
        "planner_ee_target_root_m": planner_ee,
        "ee_tracking_error_m": tracking_error,
        "maximum_forbidden_contact_force_n": forbidden_force,
        "physics_substep_policy_step": substep_policy,
        "physics_substep_index": substep_index,
        "physics_substep_pre_rate_limit_target_q_rad": substep_pre,
        "physics_substep_post_rate_limit_target_q_rad": substep_post,
        "physics_substep_final_emitted_target_q_rad": substep_final,
        "physics_substep_post_limiter_joint_clamp_flag": substep_joint_clamp,
        "physics_substep_previous_accumulator_velocity_rad_s": substep_previous_velocity,
        "physics_substep_accumulator_velocity_rad_s": substep_velocity,
        "physics_substep_accumulator_qdd_rad_s2": substep_accumulator_qdd,
        "physics_substep_final_target_fd_qdot_rad_s": substep_final_fd_qdot,
        "physics_substep_final_target_fd_qdd_rad_s2": substep_final_fd_qdd,
        "physics_substep_reset_generation": substep_reset_generation,
        "physics_substep_synchronize_generation": substep_synchronize_generation,
        "physics_substep_speed_scale": substep_speed_scale,
        "physics_substep_common_endpoint_scale": substep_common_endpoint_scale,
        "physics_substep_per_joint_endpoint_scale": substep_per_joint_endpoint_scale,
        "physics_substep_acceleration_scale": substep_acceleration_scale,
        "physics_substep_endpoint_deceleration_mask": (
            substep_endpoint_deceleration_mask
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **npz_payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)

    window: list[dict[str, Any]] = []
    for sample_index in range(window_start, window_end + 1):
        window.append(
            {
                "sample_index": sample_index,
                "policy_step": int(policy_step[sample_index]),
                "timestamp_s": float(timestamp_s[sample_index]),
                "waypoint_index": int(waypoint_index[sample_index]),
                "waypoint_boundary": bool(boundary[sample_index]),
                "waypoint_repeat_index": int(repeat_index[sample_index]),
                "post_limit_target_rad": float(post_q[sample_index, peak_joint]),
                "target_fd_qdot_rad_s": (
                    float(post_velocity[sample_index, peak_joint])
                    if np.isfinite(post_velocity[sample_index, peak_joint])
                    else None
                ),
                "target_fd_qdd_rad_s2": (
                    float(post_acceleration[sample_index, peak_joint])
                    if np.isfinite(post_acceleration[sample_index, peak_joint])
                    else None
                ),
                "pre_limit_target_rad": float(pre_q[sample_index, peak_joint]),
                "pre_limit_fd_qdd_rad_s2": (
                    float(pre_acceleration[sample_index, peak_joint])
                    if np.isfinite(pre_acceleration[sample_index, peak_joint])
                    else None
                ),
                "limiter_accumulator_velocity_rad_s": float(
                    limiter_velocity[sample_index, peak_joint]
                ),
                "limiter_reset": bool(reset[sample_index]),
                "limiter_synchronize": bool(synchronize[sample_index]),
                "measured_q_rad": float(measured_q[sample_index, peak_joint]),
                "measured_qdot_rad_s": float(measured_qdot[sample_index, peak_joint]),
                "measured_fd_qdd_rad_s2": (
                    float(measured_fd_acceleration[sample_index, peak_joint])
                    if np.isfinite(measured_fd_acceleration[sample_index, peak_joint])
                    else None
                ),
            }
        )

    return {
        "schema": "g2_curobo_target_acceleration_aligned_telemetry_v1",
        "telemetry_npz": str(output),
        "telemetry_npz_sha256": _sha256(output),
        "policy_sample_count": len(samples),
        "physics_substep_sample_count": len(substep_rows),
        "joint_names": joint_names,
        "event": {
            "offending_joint_name": joint_names[peak_joint],
            "offending_joint_index_in_right_arm": peak_joint,
            "policy_sample_index": peak_sample,
            "policy_step": int(policy_step[peak_sample]),
            "timestamp_s": float(timestamp_s[peak_sample]),
            "target_fd_qdd_rad_s2": peak_value,
            "target_fd_qdd_abs_rad_s2": abs(peak_value),
            "measured_fd_qdd_same_step_rad_s2": measured_at_peak,
            "joint_group": "RIGHT_ARM_ACTIVE",
            "duration_above_10_steps": run_end - run_start + 1,
            "duration_above_10_seconds": (run_end - run_start + 1) * control_dt_s,
            "duration_class": (
                "ONE_POLICY_STEP"
                if (run_end - run_start + 1) == 1
                else "SUSTAINED_MULTI_POLICY_STEP"
            ),
            "above_limit_run_start_policy_step": int(policy_step[run_start]),
            "above_limit_run_end_policy_step": int(policy_step[run_end]),
            "waypoint_index": int(waypoint_index[peak_sample]),
            "waypoint_boundary": bool(boundary[peak_sample]),
            "waypoint_boundary_in_fd_window": boundary_near_event,
            "waypoint_repeat_index": int(repeat_index[peak_sample]),
            "repeat_transition_in_fd_window": repeat_near_event,
            "limiter_reset_in_fd_window": reset_near_event,
            "limiter_synchronize_in_fd_window": synchronize_near_event,
            "post_limiter_joint_clamp_in_fd_window": joint_clamp_near_event,
            "pre_limiter_spike_same_sample": pre_spike,
            "post_limiter_spike": abs(peak_value) > hard_acceleration_limit_rad_s2,
            "maximum_accumulator_qdd_near_event_rad_s2": maximum_accumulator_qdd,
            "maximum_final_substep_target_fd_qdd_near_event_rad_s2": (
                maximum_final_substep_qdd
            ),
            "multi_joint_above_limit_count_same_sample": multi_joint_count,
            "above_limit_policy_event_count": above_limit_policy_event_count,
            "above_limit_joint_sample_pair_count": (
                above_limit_joint_sample_pair_count
            ),
            "collision_or_controller_reject": collision_or_reject,
            "limiter_state_discontinuity": limiter_state_discontinuity,
            "causal_limiter_substep": {
                "policy_step": int(substep_policy[causal_substep_index]),
                "physics_substep_index": int(substep_index[causal_substep_index]),
                "accumulator_qdd_rad_s2": float(
                    substep_accumulator_qdd[causal_substep_index, peak_joint]
                ),
                "final_target_fd_qdd_rad_s2": float(
                    substep_final_fd_qdd[causal_substep_index, peak_joint]
                ),
                "previous_accumulator_velocity_rad_s": float(
                    substep_previous_velocity[causal_substep_index, peak_joint]
                ),
                "accumulator_velocity_after_rad_s": float(
                    substep_velocity[causal_substep_index, peak_joint]
                ),
                "pre_rate_limit_target_rad": float(
                    substep_pre[causal_substep_index, peak_joint]
                ),
                "post_rate_limit_target_rad": float(
                    substep_post[causal_substep_index, peak_joint]
                ),
                "final_emitted_target_rad": float(
                    substep_final[causal_substep_index, peak_joint]
                ),
            },
            "root_cause": root_cause,
            "window_minus_plus_5": window,
        },
        "summary": {
            "maximum_target_fd_velocity_rad_s": float(np.nanmax(np.abs(post_velocity))),
            "maximum_target_fd_acceleration_rad_s2": float(
                np.nanmax(np.abs(post_acceleration))
            ),
            "maximum_measured_fd_acceleration_rad_s2": float(
                np.nanmax(np.abs(measured_fd_acceleration))
            ),
            "maximum_accumulator_qdd_rad_s2": float(
                np.max(np.abs(substep_accumulator_qdd))
            ),
            "maximum_final_substep_target_fd_qdd_rad_s2": float(
                np.nanmax(np.abs(substep_final_fd_qdd))
            ),
            "maximum_forbidden_contact_force_n": float(forbidden_force.max(initial=0.0)),
            "controller_reject_count": int(controller_reject.sum()),
            "clipping_count": int(clipping.sum()),
            "post_limiter_joint_clamp_substep_count": int(
                substep_joint_clamp.sum()
            ),
            "minimum_common_endpoint_scale": float(
                substep_common_endpoint_scale.min(initial=1.0)
            ),
            "minimum_per_joint_endpoint_scale": float(
                substep_per_joint_endpoint_scale.min(initial=1.0)
            ),
            "endpoint_specific_deceleration_count": int(
                substep_endpoint_deceleration_mask.sum()
            ),
        },
        "verdict": {
            "TARGET_ACCEL_ATTRIBUTION_PROBE": "PASS",
            "TARGET_ACCEL_EVENT_ROOT_CAUSE": root_cause,
            "RETIMING": "NOT_IMPLEMENTED_PROHIBITED_IN_THIS_PROBE",
            "BC": "OFF",
            "SAC": "OFF",
        },
    }


def _run_live(
    *,
    output: Path,
    replay: Path | None,
    seed: int,
    attribution_output: Path | None = None,
    timing_optimized: bool = False,
    episode_horizon_steps: int = 640,
    ordinary_tracking_cap: int = MAXIMUM_TRACKING_STEPS_PER_WAYPOINT,
    critical_tracking_cap: int = MAXIMUM_TRACKING_STEPS_PER_WAYPOINT,
    final_convergence_cap: int = 0,
    final_settle_cap: int = MAXIMUM_SETTLE_STEPS,
    planner_active_velocity_limit_rad_s: float = 0.8,
    adaptive_waypoints: bool = False,
    adaptive_collision_report: Path | None = None,
    telemetry_flush_steps: int = 0,
    progress_deadband_m: float = 0.00025,
    stall_grace_steps: int = 8,
    max_stall_steps: int = 40,
    retreat_threshold_m: float = 0.001,
    bc_checkpoint: Path | None = None,
    bc_checkpoint_sha256: str | None = None,
    bc_closed_loop_steps: int = 0,
    grasp_ready_geometry: Path | None = None,
    state_preserving_bridge: bool = False,
    passive_contact_telemetry_output: Path | None = None,
    minimum_post_contact_observation_s: float = 0.0,
    diagnostic_asset_variant: str = "candidate",
    diagnostic_asset_path: Path | None = None,
    diagnostic_asset_sha256: str | None = None,
    open_only: bool = False,
    stop_on_hard_stop: bool = False,
    candidate_a_contact_last_ab: bool = False,
    candidate_a_contact_free: bool = False,
    candidate_a_contact_free_runtime_replan: bool = False,
    cosine_p05: float | None = None,
    cosine_p10: float | None = None,
    canonical_contact_free_collection_output: Path | None = None,
    runtime_replan_canonical_contact_free_collection_output: Path | None = None,
    canonical_capture_rate_hz: int = 50,
    independent_collection_episode_id: str | None = None,
    contact_free_bc_playback_checkpoint: Path | None = None,
    contact_free_bc_playback_checkpoint_sha256: str | None = None,
    gui: bool = False,
    keyboard_v3_collection_root: Path | None = None,
    keyboard_v3_episode_id: str | None = None,
    keyboard_v3_backoff_m: float = 0.030,
    keyboard_v3_maximum_steps: int = 1600,
    keyboard_v3_translation_step_m: float = 0.001,
    keyboard_v3_motion_min_interval_s: float = 0.050,
    keyboard_v3_terminal_smoothing_steps: int = 1,
    keyboard_v3_continuous_session: bool = False,
    keyboard_v3_session_max_episodes: int = 0,
    keyboard_v3_direct_pregrasp_v2: bool = False,
    keyboard_v3_v2_smoke_only: bool = False,
    candidate_a_bc_validation_output: Path | None = None,
    candidate_a_bc_validation_episode_id: str | None = None,
    stage1a_isaac_accepted_transitions: int | None = None,
    stage1a_training_15k: bool = False,
    stage1a_reward_v3_smoke: bool = False,
    stage1a_stable_only_reward_v3: bool = False,
    stage1a_replay_strategy: str = "HER_FORCE",
    stage1a_training_seed: int = 42,
    stage1a_hybrid_activation_smoke: bool = False,
    stage1a_hybrid_activation_max_steps: int = 256,
    stage1a_playback_actor_checkpoint: Path | None = None,
    stage1a_playback_actor_checkpoint_sha256: str | None = None,
    stage1a_stable_playback: bool = False,
    stage1a_playback_video_path: Path | None = None,
    stage1a_close_admission_diagnostic: bool = False,
    stage1a_close_residual_sweep_mm: int | None = None,
    stage1a_close_lateral_offset_mm: float = 0.0,
    stage1a_close_height_offset_mm: float = 0.0,
    stage1a_close_approach_yaw_deg: float = 0.0,
    stage1a_extended_hard_stop_telemetry: bool = False,
    stage1a_geometry_forced_close_diagnostic: bool = False,
    stage1a_relaxed_close_hold_bilateral: bool = False,
    stage1a_relaxed_close_hold_event: str = "BILATERAL",
    stage1a_simplified_close_persistence_gate: bool = False,
    stage1a_close_speed_scale: float = 1.0,
    stage1a_two_stage_close: bool = False,
    stage1a_privileged_close_motion_interlock: bool = False,
    stage1a_close_calibration_sample: Path | None = None,
    stage1a_close_persistent_plan: Path | None = None,
    stage1a_bc_checkpoint: Path | None = None,
    stage1a_bc_checkpoint_sha256: str | None = None,
    stage1a_output_dir: Path | None = None,
    stage1a_wandb: bool = False,
    stage1a_wandb_mode: str = "offline",
    stage1a_wandb_project: str = "geniesim-g2-stage1a-residual-sac",
    stage1a_wandb_entity: str | None = None,
    stage1a_wandb_run_name: str | None = None,
    stage1a_wandb_group: str | None = None,
) -> int:
    from isaaclab.app import AppLauncher

    # Isaac Lab 3 may still select the offscreen input device when only
    # ``headless=False`` is supplied.  Human keyboard-v3 collection requires
    # the same explicit Kit visualizer intent as the established teacher
    # collector; otherwise no app-window keyboard events can be acquired.
    launcher = AppLauncher(
        headless=not gui,
        enable_cameras=True,
        fast_shutdown=False,
        visualizer=["kit"] if gui else ["none"],
        visualizer_explicit=True,
    )
    app = launcher.app
    print("APP_CREATED", flush=True)
    env = None
    counter = None
    target_telemetry = None
    passive_contact_telemetry = None
    passive_contact_receipt = None
    report: dict[str, Any] = {}
    path_records: list[dict[str, Any]] = []
    bc_records: list[dict[str, Any]] = []
    bc_history_records: list[dict[str, Any]] = []
    telemetry_samples: list[dict[str, Any]] = []
    telemetry_substeps: list[dict[str, Any]] = []
    pending_bc_attempt: dict[str, Any] | None = None
    settled_initial_state: dict[str, Any] | None = None
    diagnostic_asset_selection: dict[str, Any] | None = None
    physics_delta_assertion: dict[str, Any] | None = None
    runtime_replan_input_telemetry: dict[str, Any] | None = None
    runtime_replan_input_telemetry_path: Path | None = None
    partial_path = output.with_name("PARTIAL_TELEMETRY.json")
    progress: dict[str, Any] = {
        "open_settle_steps": None,
        "waypoint_tracking_steps": 0,
        "waypoints_submitted": 0,
        "final_convergence_steps": 0,
        "final_settle_steps": 0,
        "telemetry_flush_steps": 0,
        "episode_horizon_steps": int(episode_horizon_steps),
        "timeout_episode_step": None,
        "bc_closed_loop_steps": 0,
        "minimum_post_contact_observation_s": float(
            minimum_post_contact_observation_s
        ),
        "post_contact_observation_s": 0.0,
    }
    cleanup: dict[str, Any] = {"env_created": False, "env_close": "NOT_APPLICABLE"}
    freeze_before = _source_freeze(
        keyboard_v3_branch=(
            keyboard_v3_collection_root is not None
            or keyboard_v3_direct_pregrasp_v2
            or keyboard_v3_v2_smoke_only
            or candidate_a_bc_validation_output is not None
            or runtime_replan_canonical_contact_free_collection_output is not None
        )
    )
    try:
        print("RUNTIME_BEGIN", flush=True)
        if freeze_before["SOURCE_FREEZE"] != "PASS":
            raise RuntimeError("CUROBO_LIVE_SOURCE_FREEZE_FAILED")
        preflight = _load_exact_module(PREFLIGHT_SOURCE, "g2_curobo_preflight")
        p0c = _load_exact_module(P0C_SOURCE, "g2_curobo_p0c")
        p0a, p0a_provenance = p0c.P0B._load_current_p0_a_harness()

        import omni.usd
        import torch
        import warp as wp
        from isaaclab.envs import ManagerBasedRLEnv
        from geniesim.rl.isaaclab import g2_lift_task_mdp
        from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
            FullArticulationFDSafetyAudit,
            G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES,
        )
        from geniesim.rl.isaaclab.g2_keyboard_pose import (
            G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
            G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
            AbstractGripperIntent,
            GripperHysteresisLatch,
            HighLevelPolicyAction,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.curobo_4d_handoff import (
            planner_waypoint_to_canonical_4d,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import (
            DeferredFull8DActionPacketPort,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
            PreviousAcceptedPolicyAction,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.grasp_ready_geometry import (
            region_from_mapping,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.training_env_factory import (
            attest_g2_policy_4d_training_task_geometry_runtime,
            bind_g2_policy_4d_training_task_geometry_runtime,
            make_g2_contact_free_candidate_a_training_env_cfg,
            make_g2_candidate_a_left_arm_down_v2_training_env_cfg,
            make_g2_policy_4d_training_env_cfg,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.planner_timing_contract import (
            AdaptiveWaypointConfig,
            BestSoFarProgress,
            PHASE_PROGRESS_CONTRACT,
            ProgressMetricConfig,
            build_adaptive_waypoint_schedule,
            training_mask_summary,
            training_valid_mask,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.planner_bc_state_bridge import (
            minimum_velocity_drain_physics_steps,
            planner_owned_xyz_bc_gripper_action,
        )

        if (
            candidate_a_contact_free_runtime_replan
            or keyboard_v3_direct_pregrasp_v2
            or keyboard_v3_v2_smoke_only
            or stage1a_isaac_accepted_transitions is not None
        ):
            # The new path must not even read a stale replay before reset.
            # The values are replaced by a planner-only post-reset receipt
            # below, immediately before the existing canonical executor.
            if replay is not None:
                raise RuntimeError("RUNTIME_REPLAN_REJECTS_FROZEN_REPLAY")
            if adaptive_waypoints or adaptive_collision_report is not None:
                raise RuntimeError("RUNTIME_REPLAN_ADAPTIVE_SCHEDULE_NOT_YET_AUTHORIZED")
            planned_ee = np.empty((0, 3), dtype=np.float64)
            planned_q = np.empty((0, 7), dtype=np.float64)
            fine_start_index = 0
            adaptive_schedule: dict[str, Any] = {}
            collision_receipt: dict[str, Any] | None = None
            selected_waypoint_indices: list[int] = []
        else:
            if replay is None:
                raise RuntimeError("CUROBO_LIVE_FROZEN_REPLAY_REQUIRED")
            (
                planned_ee,
                planned_q,
                fine_start_index,
                adaptive_schedule,
                collision_receipt,
            ) = _load_frozen_contact_free_plan(
                replay=replay,
                adaptive_waypoints=adaptive_waypoints,
                adaptive_collision_report=adaptive_collision_report,
                planner_active_velocity_limit_rad_s=planner_active_velocity_limit_rad_s,
            )
            selected_waypoint_indices = [
                int(value) for value in adaptive_schedule["selected_indices"]
            ]

        if (
            keyboard_v3_collection_root is not None
            or candidate_a_contact_free_runtime_replan
            or candidate_a_contact_free
            or contact_free_bc_playback_checkpoint is not None
            or keyboard_v3_v2_smoke_only
            or stage1a_isaac_accepted_transitions is not None
        ):
            cfg, asset_receipt, task_contract = (
                make_g2_candidate_a_left_arm_down_v2_training_env_cfg(num_envs=1)
            )
        else:
            cfg, asset_receipt, task_contract = make_g2_policy_4d_training_env_cfg(
                num_envs=1
            )
        if diagnostic_asset_variant not in AB_ASSET_VARIANTS:
            raise RuntimeError(
                f"CUROBO_AB_ASSET_VARIANT_INVALID:{diagnostic_asset_variant}"
            )
        selected_asset = Path(asset_receipt.candidate_asset_path).resolve()
        selected_hash = asset_receipt.candidate_asset_sha256
        if diagnostic_asset_variant == "production":
            selected_asset = Path(asset_receipt.production_asset_path).resolve()
            selected_hash = asset_receipt.production_asset_sha256
            cfg.scene.robot.spawn.usd_path = str(selected_asset)
        elif diagnostic_asset_variant == "custom":
            if diagnostic_asset_path is None or diagnostic_asset_sha256 is None:
                raise RuntimeError("CUROBO_CUSTOM_ASSET_BINDING_INCOMPLETE")
            selected_asset = diagnostic_asset_path.resolve()
            selected_hash = diagnostic_asset_sha256
            cfg.scene.robot.spawn.usd_path = str(selected_asset)
        if not selected_asset.is_file() or _sha256(selected_asset) != selected_hash:
            raise RuntimeError("CUROBO_AB_SELECTED_ASSET_HASH_MISMATCH")
        if Path(cfg.scene.robot.spawn.usd_path).resolve() != selected_asset:
            raise RuntimeError("CUROBO_AB_SELECTED_ASSET_BINDING_FAILED")
        diagnostic_asset_selection = {
            "variant": diagnostic_asset_variant,
            "path": str(selected_asset),
            "sha256": selected_hash,
            "production_asset_modified": False,
            "candidate_asset_modified": False,
            "selection_scope": "DIAGNOSTIC_AB_ONLY",
            "factory_default_candidate_binding_preserved": True,
            "factory_authority": (
                "CANDIDATE_A_LEFT_ARM_DOWN_V2_COMMON_BASELINE"
                if (
                    keyboard_v3_collection_root is not None
                    or candidate_a_contact_free_runtime_replan
                    or candidate_a_contact_free
                    or contact_free_bc_playback_checkpoint is not None
                    or keyboard_v3_v2_smoke_only
                    or stage1a_isaac_accepted_transitions is not None
                )
                else "HISTORICAL_M2_CANDIDATE"
            ),
        }
        from isaaclab_physx.physics import PhysxCfg

        # SimulationContext turns ``SimulationCfg.physics is None`` into a
        # default PhysxCfg.  Materialize that exact default here so the A/B
        # changes one *effective* PhysX leaf instead of relying on an API name
        # from an older Isaac Lab generation.
        effective_baseline_physics = (
            PhysxCfg() if cfg.sim.physics is None else cfg.sim.physics.copy()
        )
        baseline_physx_cfg = effective_baseline_physics.to_dict()
        if candidate_a_contact_last_ab:
            if (
                diagnostic_asset_variant != "custom"
                or selected_asset != CANDIDATE_A_ASSET.resolve()
                or selected_hash != EXPECTED_CANDIDATE_A_ASSET_SHA256
                or seed != 42
                or bc_checkpoint is None
                or passive_contact_telemetry_output is None
                or not stop_on_hard_stop
            ):
                raise RuntimeError("CONTACT_LAST_AB_FROZEN_CONDITION_MISMATCH")
            if bool(effective_baseline_physics.solve_articulation_contact_last):
                raise RuntimeError("CONTACT_LAST_AB_BASELINE_NOT_FALSE")
            delta_physics = effective_baseline_physics.copy()
            delta_physics.solve_articulation_contact_last = True
            cfg.sim.physics = delta_physics
        delta_physx_cfg = (
            effective_baseline_physics.to_dict()
            if cfg.sim.physics is None
            else cfg.sim.physics.to_dict()
        )
        physics_differences = _config_leaf_differences(
            baseline_physx_cfg, delta_physx_cfg
        )
        if candidate_a_contact_last_ab:
            expected_difference = [
                {
                    "path": "solve_articulation_contact_last",
                    "before": False,
                    "after": True,
                }
            ]
            if physics_differences != expected_difference:
                raise RuntimeError(
                    "CONTACT_LAST_AB_UNEXPECTED_PHYSICS_DIFF:"
                    + json.dumps(_json_ready(physics_differences), sort_keys=True)
                )
            physics_delta_assertion = {
                "scope": "CANDIDATE_A_CONTACT_LAST_ONE_SHOT_AB",
                "CHANGED_PROPERTY_COUNT": 1,
                "CHANGED_PROPERTY": "solve_articulation_contact_last",
                "BASELINE": False,
                "DELTA": True,
                "UNEXPECTED_ASSET_DIFF": 0,
                "UNEXPECTED_CONTROLLER_DIFF": 0,
                "UNEXPECTED_PHYSICS_PARAMETER_DIFF": 0,
                "physics_config_leaf_differences": physics_differences,
                "selected_asset_path": str(selected_asset),
                "selected_asset_sha256": selected_hash,
                "runtime_readback": "PENDING_ENV_CREATION",
            }
        elif physics_differences:
            raise RuntimeError("UNREQUESTED_PHYSICS_CONFIG_MUTATION")
        if (
            passive_contact_telemetry_output is not None
            or stage1a_isaac_accepted_transitions is not None
        ):
            # Telemetry-only contact-point allocation.  This does not alter
            # the collision geometry, materials, solver, controller or task
            # action path.  It exposes the existing filtered pad/Object
            # contact buffers at physics-substep resolution.
            for sensor_cfg in (
                cfg.scene.right_inner_finger_contact,
                cfg.scene.right_outer_finger_contact,
                cfg.scene.diagnostic_right_outer_link2_contact,
            ):
                sensor_cfg.track_contact_points = True
                sensor_cfg.max_contact_data_count_per_prim = (
                    _PassiveContactPhysicsTelemetry.CONTACT_CAPACITY
                )
        if episode_horizon_steps <= 0:
            raise RuntimeError("CUROBO_LIVE_EPISODE_HORIZON_INVALID")
        policy_dt_s = float(cfg.sim.dt) * int(cfg.decimation)
        cfg.episode_length_s = int(episode_horizon_steps) * policy_dt_s
        goal_hold_s = cfg.episode_length_s + policy_dt_s
        cfg.commands.object_pose.resampling_time_range = (goal_hold_s, goal_hold_s)
        cfg.observations.policy.enable_corruption = False
        cfg.commands.object_pose.debug_vis = False
        env = ManagerBasedRLEnv(cfg=cfg)
        env._g2_curobo_p0a_module = p0a
        cleanup["env_created"] = True
        if physics_delta_assertion is not None:
            import omni.usd

            physics_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
                str(cfg.sim.physics_prim_path)
            )
            contact_last_attribute = physics_prim.GetAttribute(
                "physxScene:solveArticulationContactLast"
            )
            runtime_contact_last = (
                bool(contact_last_attribute.Get())
                if contact_last_attribute and contact_last_attribute.IsValid()
                else None
            )
            physics_delta_assertion["runtime_readback"] = runtime_contact_last
            physics_delta_assertion["runtime_prim_path"] = str(
                cfg.sim.physics_prim_path
            )
            physics_delta_assertion["runtime_attribute"] = (
                "physxScene:solveArticulationContactLast"
            )
            if runtime_contact_last is not True:
                raise RuntimeError("CONTACT_LAST_AB_RUNTIME_READBACK_FAILED")
        task_geometry_binding = bind_g2_policy_4d_training_task_geometry_runtime(env)
        g2_lift_task_mdp.configure_grasp_reward_target(
            env,
            G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M,
            calibration_rows=G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
        )
        _, reset_info = env.reset(seed=seed)
        del reset_info
        task_geometry_readback = attest_g2_policy_4d_training_task_geometry_runtime(env)
        print("ENV_READY", flush=True)

        arm_term = env.action_manager.get_term("arm_action")
        counter = p0a._LifecycleCounter(env)
        counter.install()
        latch = GripperHysteresisLatch(initial_intent=AbstractGripperIntent.OPEN)
        open_action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.0))
        open_packet, _, _ = p0a._build_authoritative_packet(
            high_level=open_action,
            batch_size=env.num_envs,
            device=env.device,
            latch=latch,
        )
        deferred_port = DeferredFull8DActionPacketPort(
            batch_size=1,
            device=env.device,
            binding_id="g2_curobo_planner_live_smoke_v1",
        )
        try:
            settle = preflight._settle_open_before_policy(
                env=env,
                p0a=p0a,
                counter=counter,
                deferred_port=deferred_port,
                open_packet=open_packet,
            )
        except BaseException:
            settled_initial_state = _ab_settled_state_snapshot(
                env, env.scene["robot"]
            )
            settled_initial_state["open_settle_status"] = "FAILED_TARGET_NOT_REACHED"
            raise
        progress["open_settle_steps"] = int(settle["actual_policy_steps"])
        print("OPEN_SETTLE_DONE", flush=True)
        latch.reset()

        runtime_replan_receipt: dict[str, Any] | None = None
        keyboard_v3_plan: Any | None = None
        direct_init_receipt: dict[str, Any] | None = None
        hybrid_activation_initial_receipt: dict[str, Any] | None = None
        hybrid_activation_nominal_pose: tuple[float, ...] | None = None
        persistent_close_plan: dict[str, Any] | None = None
        runtime_replan_plan_sidecar_sha256: str | None = None
        canonical_collection_output = (
            runtime_replan_canonical_contact_free_collection_output
            if runtime_replan_canonical_contact_free_collection_output is not None
            else canonical_contact_free_collection_output
        )
        if (
            keyboard_v3_direct_pregrasp_v2
            or keyboard_v3_v2_smoke_only
            or stage1a_isaac_accepted_transitions is not None
        ):
            from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
                BASELINE_NAME,
                LEFT_ARM_POSTURE_ID,
                load_baseline_manifest,
                selected_pregrasp_initial_state,
            )
            from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_collection_contract import (
                NearGraspCondition,
            )
            from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_direct_pregrasp_init import (
                apply_direct_pregrasp_initial_state,
                direct_pregrasp_plan_receipt,
            )
            from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_live_driver import (
                _capture_camera_rgbd,
                _ee_root_pose,
            )

            from geniesim.rl.sac.close_outcome_collection import (
                load_close_outcome_sample,
            )

            sample = selected_pregrasp_initial_state()
            if stage1a_hybrid_activation_smoke or stage1a_training_15k or stage1a_reward_v3_smoke:
                from geniesim.rl.sac.stage1a_isaac_short_smoke import (
                    source_authoritative_hybrid_activation_initial_state,
                )

                (
                    sample,
                    hybrid_activation_nominal_pose,
                    hybrid_activation_initial_receipt,
                ) = source_authoritative_hybrid_activation_initial_state()
            elif stage1a_close_persistent_plan is not None:
                persistent_close_plan = json.loads(
                    stage1a_close_persistent_plan.read_text(encoding="utf-8")
                )
                jobs = persistent_close_plan.get("jobs")
                if (
                    persistent_close_plan.get("schema")
                    != "g2_candidate_a_persistent_close_plan_v1"
                    or not isinstance(jobs, list)
                    or not jobs
                    or persistent_close_plan.get("workers") != 1
                ):
                    raise RuntimeError("PERSISTENT_CLOSE_PLAN_SCHEMA_MISMATCH")
                first_sample_path = Path(str(jobs[0]["sample_path"])).resolve()
                sample = load_close_outcome_sample(first_sample_path)
            elif stage1a_close_calibration_sample is not None:
                sample = load_close_outcome_sample(
                    stage1a_close_calibration_sample
                )
            manifest_path, _baseline_manifest, baseline_manifest_sha256 = (
                load_baseline_manifest()
            )
            camera_before: dict[str, tuple[int, float]] = {}
            for camera_name in ("head", "right_wrist"):
                capture = _capture_camera_rgbd(
                    env,
                    camera_name=camera_name,
                    prior_frame=None,
                    prior_sensor_time_s=0.0,
                )
                camera_before[camera_name] = (int(capture[3]), float(capture[4]))
            direct_receipt_object = apply_direct_pregrasp_initial_state(
                env, sample=sample
            )
            direct_init_receipt = direct_receipt_object.as_dict()
            condition = NearGraspCondition(
                condition_id="dataset_pregrasp_30mm_contract",
                backoff_m=float(keyboard_v3_backoff_m),
                start_offset_xyz_m=(0.0, 0.0, 0.0),
            )
            keyboard_v3_plan = direct_pregrasp_plan_receipt(
                condition, sample=sample
            )
            runtime_replan_receipt = keyboard_v3_plan.planner_only_receipt()
            runtime_plan = keyboard_v3_plan
            direct_refresh_receipts: list[dict[str, Any]] = []
            for refresh_index in range(2):
                outputs, refresh_receipt = preflight._consume_once(
                    env=env,
                    counter=counter,
                    deferred_port=deferred_port,
                    packet=open_packet,
                    label=f"DIRECT_PREGRASP_CAMERA_REFRESH_{refresh_index + 1}",
                )
                if not bool(refresh_receipt["single_consumption"]):
                    raise RuntimeError("DIRECT_INIT_REFRESH_NOT_SINGLE_CONSUMPTION")
                latch.commit(open_packet.gripper_intent)
                _, _, terminated, truncated, _ = outputs
                active = preflight._active_termination_names(env, terminated, truncated)
                if bool(terminated.reshape(-1)[0].item()) or bool(
                    truncated.reshape(-1)[0].item()
                ) or active:
                    raise RuntimeError(
                        "DIRECT_INIT_REFRESH_TERMINATED:" + ",".join(active)
                    )
                direct_refresh_receipts.append(dict(refresh_receipt))
            camera_after: dict[str, tuple[int, float]] = {}
            for camera_name in ("head", "right_wrist"):
                prior_frame, prior_time = camera_before[camera_name]
                capture = _capture_camera_rgbd(
                    env,
                    camera_name=camera_name,
                    prior_frame=prior_frame,
                    prior_sensor_time_s=prior_time,
                )
                camera_after[camera_name] = (int(capture[3]), float(capture[4]))
                if int(capture[3]) <= prior_frame or float(capture[4]) <= prior_time:
                    raise RuntimeError(f"{camera_name.upper()}_FRAME_STALE_AFTER_RESET")
            arm_term.synchronize_reset_to_measured(
                torch.zeros((1,), device=env.device, dtype=torch.long)
            )
            arm_ids = torch.as_tensor(
                [int(value) for value in arm_term._joint_ids],
                device=env.device,
                dtype=torch.long,
            )
            measured_arm = _tensor(env.scene["robot"].data.joint_pos).index_select(
                1, arm_ids
            )
            emitted_arm = _tensor(arm_term.last_emitted_joint_position_target)
            post_refresh_target_error = float(
                torch.max(torch.abs(measured_arm - emitted_arm)).item()
            )
            if post_refresh_target_error > 1.0e-6:
                raise RuntimeError(
                    f"DIRECT_INIT_POST_REFRESH_TARGET_DISCONTINUITY:{post_refresh_target_error}"
                )
            evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
            peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
            forbidden_peak = float(
                max((max(values) for values in peaks.values()), default=0.0)
            )
            task_metrics = preflight._task_metrics(env, g2_lift_task_mdp)
            if (
                forbidden_peak > 1.0e-6
                or float(task_metrics["inner_contact_force_n"]) > 0.0
                or float(task_metrics["outer_contact_force_n"]) > 0.0
            ):
                raise RuntimeError("DIRECT_INIT_CONTACT_OR_FORBIDDEN_COLLISION")
            ee_position, _ee_quaternion = _ee_root_pose(env, p0a)
            direct_ee_error_m = float(
                np.linalg.norm(
                    ee_position - np.asarray(sample.ee_pose_robot_root_m_xyzw[:3])
                )
            )
            cube_position = _cube_root_position(env)
            cube_position_error_m = float(
                np.linalg.norm(
                    cube_position
                    - np.asarray(sample.cube_pose_robot_root_m_xyzw[:3])
                )
            )
            left_ids = torch.as_tensor(
                [
                    env.scene["robot"].joint_names.index(name)
                    for name in direct_receipt_object.left_arm_joint_names
                ],
                device=env.device,
                dtype=torch.long,
            )
            left_measured = (
                _tensor(env.scene["robot"].data.joint_pos)
                .index_select(1, left_ids)[0]
                .detach()
                .to("cpu")
                .numpy()
            )
            left_pose_error_rad = float(
                np.max(
                    np.abs(
                        left_measured
                        - np.asarray(direct_receipt_object.left_arm_q_rad)
                    )
                )
            )
            if left_pose_error_rad > 1.0e-3:
                raise RuntimeError(
                    f"LEFT_ARM_DOWN_POSE_DRIFT:{left_pose_error_rad}"
                )
            direct_init_receipt.update(
                {
                    "baseline_manifest_path": str(manifest_path),
                    "baseline_manifest_sha256": baseline_manifest_sha256,
                    "camera_refresh_action_submission_count": 2,
                    "camera_refresh_process_action_count": int(
                        sum(
                            int(item["action_manager_process_action_count"])
                            for item in direct_refresh_receipts
                        )
                    ),
                    "camera_frame_before": {
                        name: value[0] for name, value in camera_before.items()
                    },
                    "camera_frame_after": {
                        name: value[0] for name, value in camera_after.items()
                    },
                    "first_head_frame_after_reset": "CURRENT_STATE",
                    "first_wrist_frame_after_reset": "CURRENT_STATE",
                    "post_refresh_target_to_measured_max_rad": post_refresh_target_error,
                    "measured_ee_to_selected_state_error_m": direct_ee_error_m,
                    "post_refresh_cube_settle_displacement_m": cube_position_error_m,
                    "left_arm_pose_max_error_rad": left_pose_error_rad,
                    "direct_follower_target_write_count": 0,
                    "forbidden_collision_count": 0,
                    "contact_count": 0,
                    "old_startup_curobo_time_s": float(sample.source_control_step) / 50.0,
                    "new_direct_init_time_s": 2.0 / 50.0,
                    "old_startup_control_steps": int(sample.source_control_step),
                    "new_startup_control_steps": 2,
                    "startup_compute_reduction_percent": 100.0
                    * (1.0 - 2.0 / float(sample.source_control_step)),
                }
            )
            path_records = []
            maximum_forbidden_contact_force_n = forbidden_peak
            final_target = np.asarray(
                sample.ee_pose_robot_root_m_xyzw[:3], dtype=np.float64
            )
            print("DIRECT_PREGRASP_V2_READY", flush=True)

            if stage1a_isaac_accepted_transitions is not None:
                # Dedicated bounded Stage-1A path.  This branch is deliberately
                # downstream of the exact Candidate-A V2 direct-init/cache and
                # camera refresh assertions above.  It owns neither an alternate
                # environment factory nor a direct ActionManager call.
                from geniesim.rl.sac.stage1a_isaac_short_smoke import (
                    _reset_direct_pregrasp,
                    run_stage1a_close_admission_diagnostic,
                    run_stage1a_close_residual_sweep_episode,
                    run_stage1a_hybrid_activation_smoke,
                    run_stage1a_isaac_short_smoke,
                )

                if (
                    stage1a_bc_checkpoint is None
                    or stage1a_bc_checkpoint_sha256 is None
                    or stage1a_output_dir is None
                ):
                    raise RuntimeError("STAGE1A_ISAAC_SMOKE_BINDING_INCOMPLETE")
                close_mechanics_telemetry = None
                if (
                    persistent_close_plan is not None
                    or stage1a_close_residual_sweep_mm is not None
                ):
                    stage1a_robot = env.scene["robot"]
                    stage1a_arm_term = env.action_manager.get_term("arm_action")
                    close_mechanics_telemetry = CloseMechanicsRingBuffer500Hz(
                        env=env,
                        robot=stage1a_robot,
                        torch_module=torch,
                        arm_joint_names=(
                            stage1a_robot.joint_names[int(index)]
                            for index in stage1a_arm_term._joint_ids
                        ),
                        master_joint_name="idx71_gripper_r_inner_joint1",
                        passive_joint_names=G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES,
                        post_close_window_ms=(
                            int(
                                persistent_close_plan.get(
                                    "close_mechanics_post_window_ms", 1500
                                )
                            )
                            if persistent_close_plan is not None
                            else 1500
                        ),
                    )
                    # Install the raw-only wrapper first.  The existing contact
                    # telemetry/hard-stop wrapper is installed around it, so a
                    # hard-stop exception cannot hide the triggering q/qd
                    # substep from the CLOSE mechanics artifact.
                    close_mechanics_telemetry.install()
                passive_contact_telemetry = _PassiveContactPhysicsTelemetry(
                    env=env,
                    task_mdp=g2_lift_task_mdp,
                    robot=env.scene["robot"],
                    torch_module=torch,
                    warp_module=wp,
                    stage=omni.usd.get_context().get_stage(),
                    stop_on_hard_stop=(
                        not stage1a_extended_hard_stop_telemetry
                        and not stage1a_stable_only_reward_v3
                        and not stage1a_stable_playback
                    ),
                    defer_qdd_to_close_mechanics=(
                        stage1a_extended_hard_stop_telemetry
                    ),
                )
                passive_contact_telemetry.install()
                if persistent_close_plan is not None:
                    from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
                        nominal_grasp_pose_root_m_xyzw_for_cube,
                    )

                    jobs = list(persistent_close_plan["jobs"])
                    quota = {
                        int(key): int(value)
                        for key, value in persistent_close_plan["valid_target_per_distance"].items()
                    }
                    valid_counts = {
                        int(key): int(value)
                        for key, value in persistent_close_plan["initial_valid_by_distance"].items()
                    }
                    progress_path = Path(
                        str(persistent_close_plan["progress_jsonl"])
                    ).resolve()
                    if progress_path.exists():
                        raise RuntimeError("PERSISTENT_CLOSE_PROGRESS_REFUSES_OVERWRITE")
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    persistent_results: list[dict[str, Any]] = []
                    reset_count = 0
                    with progress_path.open("x", encoding="utf-8") as progress_stream:
                        for job_index, job in enumerate(jobs):
                            target_mm = int(job["target_residual_mm"])
                            if target_mm not in quota:
                                raise RuntimeError(
                                    f"PERSISTENT_CLOSE_TARGET_NOT_CONTRACTED:{target_mm}"
                                )
                            if valid_counts[target_mm] >= quota[target_mm]:
                                continue
                            job_sample_path = Path(str(job["sample_path"])).resolve()
                            job_sample = load_close_outcome_sample(job_sample_path)
                            if job_index > 0:
                                _reset_direct_pregrasp(
                                    env=env,
                                    p0a=p0a,
                                    preflight=preflight,
                                    counter=counter,
                                    deferred_port=deferred_port,
                                    latch=latch,
                                    physics_telemetry=passive_contact_telemetry,
                                    episode_index=job_index,
                                    sample=job_sample,
                                    runtime_seed=int(job["runtime_seed"]),
                                )
                                assert close_mechanics_telemetry is not None
                                close_mechanics_telemetry.reset_episode()
                                reset_count += 1
                            elif job_sample.sample_id != sample.sample_id:
                                raise RuntimeError(
                                    "PERSISTENT_CLOSE_FIRST_SAMPLE_BINDING_MISMATCH"
                                )
                            else:
                                passive_contact_telemetry.reset_episode_buffers()
                                assert close_mechanics_telemetry is not None
                                close_mechanics_telemetry.reset_episode()

                            report_path = Path(str(job["report_path"])).resolve()
                            job_output_dir = Path(str(job["output_dir"])).resolve()
                            if report_path.exists() or job_output_dir.exists():
                                raise RuntimeError(
                                    f"PERSISTENT_CLOSE_JOB_REFUSES_OVERWRITE:{job['run_id']}"
                                )
                            nominal_pose = nominal_grasp_pose_root_m_xyzw_for_cube(
                                job_sample.cube_pose_robot_root_m_xyzw[:3]
                            )
                            started_job = time.monotonic()
                            error_text = None
                            try:
                                return_code = run_stage1a_close_residual_sweep_episode(
                                    env=env,
                                    p0a=p0a,
                                    preflight=preflight,
                                    task_mdp=g2_lift_task_mdp,
                                    counter=counter,
                                    deferred_port=deferred_port,
                                    latch=latch,
                                    physics_telemetry=passive_contact_telemetry,
                                    close_mechanics_telemetry=close_mechanics_telemetry,
                                    acceleration_limit_rad_s2=HARD_STOP_ACCELERATION_LIMIT_RAD_S2,
                                    source_freeze_before=freeze_before,
                                    source_freeze_provider=lambda: _source_freeze(
                                        keyboard_v3_branch=True
                                    ),
                                    selected_asset_path=selected_asset,
                                    selected_asset_sha256=selected_hash,
                                    nominal_grasp_pose_root_m_xyzw=nominal_pose,
                                    bc_checkpoint=stage1a_bc_checkpoint,
                                    bc_checkpoint_sha256=stage1a_bc_checkpoint_sha256,
                                    output_dir=job_output_dir,
                                    report_path=report_path,
                                    calibration_episode_id=(
                                        f"{job_sample_path.stem}-{target_mm}mm-"
                                        f"seed-{job['runtime_seed']}"
                                    ),
                                    pregrasp_sample_id=job_sample.sample_id,
                                    target_residual_mm=target_mm,
                                    runtime_seed=int(job["runtime_seed"]),
                                    lateral_offset_mm=float(job["lateral_offset_mm"]),
                                    height_offset_mm=float(job["height_offset_mm"]),
                                    approach_yaw_deg=float(job["approach_yaw_deg"]),
                                    maximum_post_close_submissions=int(
                                        job.get(
                                            "maximum_post_close_submissions", 80
                                        )
                                    ),
                                    hard_stop_limit_numerical_tolerance_rad=(
                                        HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD
                                    ),
                                    continue_after_hard_stop_for_telemetry=(
                                        stage1a_extended_hard_stop_telemetry
                                    ),
                                    geometry_forced_close_diagnostic=(
                                        stage1a_geometry_forced_close_diagnostic
                                    ),
                                    simplified_close_persistence_gate=(
                                        stage1a_simplified_close_persistence_gate
                                    ),
                                    close_speed_scale=float(
                                        job.get("close_speed_scale", 1.0)
                                    ),
                                    two_stage_close=bool(
                                        job.get("two_stage_close", False)
                                    ),
                                )
                            except Exception as error:
                                return_code = 2
                                error_text = f"{type(error).__name__}:{error}"
                                if not report_path.exists():
                                    _atomic_json(
                                        report_path,
                                        {
                                            "schema": "g2_persistent_close_invalid_v1",
                                            "outcome": "INVALID",
                                            "outcome_reason": error_text,
                                            "target_residual_mm": target_mm,
                                            "runtime_seed": int(job["runtime_seed"]),
                                            "pregrasp_sample_id": job_sample.sample_id,
                                            "source_freeze_match": (
                                                freeze_before
                                                == _source_freeze(keyboard_v3_branch=True)
                                            ),
                                            "asset": {
                                                "path": str(selected_asset),
                                                "sha256": selected_hash,
                                            },
                                            "training": "OFF",
                                            "residual_sac": "OFF",
                                        },
                                    )
                            job_report = json.loads(
                                report_path.read_text(encoding="utf-8")
                            )
                            outcome = str(job_report.get("outcome", "INVALID"))
                            if outcome in ("SAFE_CLOSE", "UNSAFE_CLOSE"):
                                valid_counts[target_mm] += 1
                            event = {
                                "schema": "g2_candidate_a_persistent_close_event_v1",
                                "job_index": job_index,
                                "run_id": str(job["run_id"]),
                                "target_residual_mm": target_mm,
                                "runtime_seed": int(job["runtime_seed"]),
                                "sample_path": str(job_sample_path),
                                "sample_id": job_sample.sample_id,
                                "lateral_offset_mm": float(job["lateral_offset_mm"]),
                                "height_offset_mm": float(job["height_offset_mm"]),
                                "approach_yaw_deg": float(job["approach_yaw_deg"]),
                                "close_speed_scale": float(
                                    job.get("close_speed_scale", 1.0)
                                ),
                                "two_stage_close": bool(
                                    job.get("two_stage_close", False)
                                ),
                                "maximum_post_close_submissions": int(
                                    job.get("maximum_post_close_submissions", 80)
                                ),
                                "report_path": str(report_path),
                                "output_dir": str(job_output_dir),
                                "outcome": outcome,
                                "outcome_reason": job_report.get("outcome_reason"),
                                "return_code": return_code,
                                "error": error_text,
                                "duration_s": time.monotonic() - started_job,
                                "valid_by_distance": {
                                    str(key): value
                                    for key, value in sorted(valid_counts.items())
                                },
                                "in_process_reset_count": reset_count,
                                "app_recreation_count": 0,
                                "training_started": False,
                                "residual_sac_started": False,
                            }
                            _durable_jsonl(progress_stream, event)
                            persistent_results.append(event)
                            print(
                                "[PERSISTENT_CLOSE_PROGRESS] "
                                f"completed={len(persistent_results)} "
                                f"target={target_mm} outcome={outcome} "
                                f"valid={sum(valid_counts.values())}/"
                                f"{sum(quota.values())}",
                                flush=True,
                            )
                            if all(
                                valid_counts[key] >= quota[key]
                                for key in quota
                            ):
                                break
                    persistent_pass = all(
                        valid_counts[key] >= quota[key] for key in quota
                    )
                    persistent_report = {
                        "schema": "g2_candidate_a_persistent_close_session_v1",
                        "execution_mode": "ONE_ISAAC_APP_MULTIPLE_IN_PROCESS_EPISODE_RESETS",
                        "source_plan": str(stage1a_close_persistent_plan),
                        "source_plan_sha256": _sha256(stage1a_close_persistent_plan),
                        "result_count": len(persistent_results),
                        "valid_by_distance": {
                            str(key): value for key, value in sorted(valid_counts.items())
                        },
                        "valid_target_per_distance": {
                            str(key): value for key, value in sorted(quota.items())
                        },
                        "in_process_reset_count": reset_count,
                        "app_creation_count": 1,
                        "asset_load_count": 1,
                        "process_restart_per_episode": False,
                        "mechanics_contract_changed": False,
                        "control_contract_changed": False,
                        "close_contract_changed": False,
                        "training_started": False,
                        "residual_sac_started": False,
                        "PERSISTENT_COLLECTION": "PASS" if persistent_pass else "INCOMPLETE",
                        "progress_jsonl": str(progress_path),
                    }
                    _atomic_json(output, persistent_report)
                    print("RUNTIME_END", flush=True)
                    print("REPORT_SAVED", flush=True)
                    return 0 if persistent_pass else 2
                if stage1a_close_residual_sweep_mm is not None:
                    from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
                        nominal_grasp_pose_root_m_xyzw_for_cube,
                    )

                    sweep_nominal_pose = nominal_grasp_pose_root_m_xyzw_for_cube(
                        sample.cube_pose_robot_root_m_xyzw[:3]
                    )
                    stage1a_result = run_stage1a_close_residual_sweep_episode(
                        env=env,
                        p0a=p0a,
                        preflight=preflight,
                        task_mdp=g2_lift_task_mdp,
                        counter=counter,
                        deferred_port=deferred_port,
                        latch=latch,
                        physics_telemetry=passive_contact_telemetry,
                        close_mechanics_telemetry=close_mechanics_telemetry,
                        acceleration_limit_rad_s2=HARD_STOP_ACCELERATION_LIMIT_RAD_S2,
                        source_freeze_before=freeze_before,
                        source_freeze_provider=lambda: _source_freeze(
                            keyboard_v3_branch=True
                        ),
                        selected_asset_path=selected_asset,
                        selected_asset_sha256=selected_hash,
                        nominal_grasp_pose_root_m_xyzw=sweep_nominal_pose,
                        bc_checkpoint=stage1a_bc_checkpoint,
                        bc_checkpoint_sha256=stage1a_bc_checkpoint_sha256,
                        output_dir=stage1a_output_dir,
                        report_path=output,
                        calibration_episode_id=(
                            f"{stage1a_close_calibration_sample.stem}-"
                            f"{stage1a_close_residual_sweep_mm}mm"
                        ),
                        pregrasp_sample_id=sample.sample_id,
                        target_residual_mm=stage1a_close_residual_sweep_mm,
                        runtime_seed=int(seed),
                        lateral_offset_mm=float(stage1a_close_lateral_offset_mm),
                        height_offset_mm=float(stage1a_close_height_offset_mm),
                        approach_yaw_deg=float(stage1a_close_approach_yaw_deg),
                        hard_stop_limit_numerical_tolerance_rad=(
                            HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD
                        ),
                        continue_after_hard_stop_for_telemetry=(
                            stage1a_extended_hard_stop_telemetry
                        ),
                        geometry_forced_close_diagnostic=(
                            stage1a_geometry_forced_close_diagnostic
                        ),
                        relaxed_close_hold_bilateral=(
                            stage1a_relaxed_close_hold_bilateral
                        ),
                        relaxed_close_hold_event=(
                            stage1a_relaxed_close_hold_event
                        ),
                        simplified_close_persistence_gate=(
                            stage1a_simplified_close_persistence_gate
                        ),
                        close_speed_scale=stage1a_close_speed_scale,
                        two_stage_close=stage1a_two_stage_close,
                    )
                    if passive_contact_telemetry_output is not None:
                        passive_contact_receipt = passive_contact_telemetry.save(
                            passive_contact_telemetry_output
                        )
                    return stage1a_result
                if stage1a_hybrid_activation_smoke:
                    if (
                        hybrid_activation_nominal_pose is None
                        or hybrid_activation_initial_receipt is None
                    ):
                        raise RuntimeError(
                            "HYBRID_ACTIVATION_SOURCE_INITIAL_STATE_MISSING"
                        )
                    return run_stage1a_hybrid_activation_smoke(
                        env=env,
                        p0a=p0a,
                        preflight=preflight,
                        counter=counter,
                        deferred_port=deferred_port,
                        latch=latch,
                        physics_telemetry=passive_contact_telemetry,
                        nominal_grasp_pose_root_m_xyzw=(
                            hybrid_activation_nominal_pose
                        ),
                        source_initial_state_receipt=(
                            hybrid_activation_initial_receipt
                        ),
                        source_freeze_before=freeze_before,
                        source_freeze_provider=lambda: _source_freeze(
                            keyboard_v3_branch=True
                        ),
                        selected_asset_path=selected_asset,
                        selected_asset_sha256=selected_hash,
                        far_reach_checkpoint=STAGE1A_FAR_REACH_BC_CHECKPOINT,
                        far_reach_checkpoint_sha256=STAGE1A_FAR_REACH_BC_SHA256,
                        bc_checkpoint=stage1a_bc_checkpoint,
                        bc_checkpoint_sha256=stage1a_bc_checkpoint_sha256,
                        residual_actor_checkpoint=(
                            stage1a_playback_actor_checkpoint
                            or STAGE1A_RESIDUAL_ACTOR_CHECKPOINT
                        ),
                        residual_actor_checkpoint_sha256=(
                            stage1a_playback_actor_checkpoint_sha256
                            or STAGE1A_RESIDUAL_ACTOR_SHA256
                        ),
                        output_dir=stage1a_output_dir,
                        report_path=output,
                        maximum_control_steps=(
                            stage1a_hybrid_activation_max_steps
                        ),
                        wandb_enabled=stage1a_wandb,
                        wandb_mode=stage1a_wandb_mode,
                        wandb_project=stage1a_wandb_project,
                        wandb_entity=stage1a_wandb_entity,
                        wandb_run_name=stage1a_wandb_run_name,
                        stable_playback=stage1a_stable_playback,
                        record_video_path=stage1a_playback_video_path,
                    )
                if stage1a_close_admission_diagnostic:
                    # The dataset-direct Keyboard-v3 receipt intentionally
                    # preserves the 30 mm planner handoff contract.  Stage-1A
                    # CLOSE calibration, however, is defined only inside the
                    # 15--22 mm local grasp band.  Derive a calibration-only
                    # target in robot-root coordinates from the measured EE
                    # toward the matching sample cube.  This target is used by
                    # reward/telemetry only: it is not an actor input and does
                    # not change the frozen GRU CLOSE latch or command path.
                    calibration_ee_root, _ = _ee_root_pose(env, p0a)
                    calibration_cube_root = np.asarray(
                        sample.cube_pose_robot_root_m_xyzw[:3], dtype=np.float64
                    )
                    calibration_direction = (
                        calibration_cube_root
                        - np.asarray(calibration_ee_root, dtype=np.float64)
                    )
                    calibration_direction_norm = float(
                        np.linalg.norm(calibration_direction)
                    )
                    if calibration_direction_norm <= 1.0e-9:
                        raise RuntimeError(
                            "CLOSE_CALIBRATION_TARGET_DIRECTION_DEGENERATE"
                        )
                    calibration_direction /= calibration_direction_norm
                    calibration_nominal_position = np.asarray(
                        calibration_ee_root, dtype=np.float64
                    ) + calibration_direction * 0.0185
                    calibration_nominal_pose = (
                        *calibration_nominal_position.tolist(),
                        *keyboard_v3_plan.nominal_grasp_pose_root_m_xyzw[3:],
                    )
                    return run_stage1a_close_admission_diagnostic(
                        env=env,
                        p0a=p0a,
                        preflight=preflight,
                        task_mdp=g2_lift_task_mdp,
                        counter=counter,
                        deferred_port=deferred_port,
                        latch=latch,
                        physics_telemetry=passive_contact_telemetry,
                        source_freeze_before=freeze_before,
                        source_freeze_provider=lambda: _source_freeze(
                            keyboard_v3_branch=True
                        ),
                        selected_asset_path=selected_asset,
                        selected_asset_sha256=selected_hash,
                        nominal_grasp_pose_root_m_xyzw=calibration_nominal_pose,
                        bc_checkpoint=stage1a_bc_checkpoint,
                        bc_checkpoint_sha256=stage1a_bc_checkpoint_sha256,
                        output_dir=stage1a_output_dir,
                        report_path=output,
                        calibration_episode_id=(
                            stage1a_close_calibration_sample.stem
                            if stage1a_close_calibration_sample is not None
                            else "candidate-a-common-baseline"
                        ),
                        pregrasp_sample_id=sample.sample_id,
                    )
                return run_stage1a_isaac_short_smoke(
                    app=app,
                    env=env,
                    p0a=p0a,
                    preflight=preflight,
                    task_mdp=g2_lift_task_mdp,
                    counter=counter,
                    deferred_port=deferred_port,
                    latch=latch,
                    physics_telemetry=passive_contact_telemetry,
                    selected_pregrasp_sample=sample,
                    direct_init_receipt=direct_init_receipt,
                    source_freeze_before=freeze_before,
                    source_freeze_provider=lambda: _source_freeze(
                        keyboard_v3_branch=True
                    ),
                    selected_asset_path=selected_asset,
                    selected_asset_sha256=selected_hash,
                    accepted_transition_target=stage1a_isaac_accepted_transitions,
                    nominal_grasp_pose_root_m_xyzw=(
                        hybrid_activation_nominal_pose
                        if (stage1a_training_15k or stage1a_reward_v3_smoke)
                        else keyboard_v3_plan.nominal_grasp_pose_root_m_xyzw
                    ),
                    bc_checkpoint=stage1a_bc_checkpoint,
                    bc_checkpoint_sha256=stage1a_bc_checkpoint_sha256,
                    output_dir=stage1a_output_dir,
                    report_path=output,
                    wandb_enabled=stage1a_wandb,
                    wandb_mode=stage1a_wandb_mode,
                    wandb_project=stage1a_wandb_project,
                    wandb_entity=stage1a_wandb_entity,
                    wandb_run_name=stage1a_wandb_run_name,
                    wandb_group=stage1a_wandb_group,
                    replay_strategy=stage1a_replay_strategy,
                    far_reach_checkpoint=(
                        STAGE1A_FAR_REACH_BC_CHECKPOINT
                        if (stage1a_training_15k or stage1a_reward_v3_smoke) else None
                    ),
                    far_reach_checkpoint_sha256=(
                        STAGE1A_FAR_REACH_BC_SHA256
                        if (stage1a_training_15k or stage1a_reward_v3_smoke) else None
                    ),
                    residual_actor_checkpoint=(
                        STAGE1A_RESIDUAL_ACTOR_CHECKPOINT
                        if (stage1a_training_15k or stage1a_reward_v3_smoke) else None
                    ),
                    residual_actor_checkpoint_sha256=(
                        STAGE1A_RESIDUAL_ACTOR_SHA256
                        if (stage1a_training_15k or stage1a_reward_v3_smoke) else None
                    ),
                    reward_v3=(
                        stage1a_reward_v3_smoke
                        or stage1a_stable_only_reward_v3
                    ),
                    stable_only=stage1a_stable_only_reward_v3,
                    privileged_close_motion_interlock=(
                        stage1a_privileged_close_motion_interlock
                    ),
                    training_seed=stage1a_training_seed,
                    acceleration_limit_rad_s2=(
                        HARD_STOP_ACCELERATION_LIMIT_RAD_S2
                    ),
                )

            if keyboard_v3_v2_smoke_only:
                from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_operator_views import (
                    KeyboardV3OperatorViews,
                )

                views = KeyboardV3OperatorViews()
                views.create()
                view_receipt = views.receipt()
                before_move, _ = _ee_root_pose(env, p0a)
                small_move = HighLevelPolicyAction.from_sequence((0.02, 0.0, 0.0, 0.0))
                move_packet, _, _ = p0a._build_authoritative_packet(
                    high_level=small_move,
                    batch_size=env.num_envs,
                    device=env.device,
                    latch=latch,
                )
                outputs, move_receipt = preflight._consume_once(
                    env=env,
                    counter=counter,
                    deferred_port=deferred_port,
                    packet=move_packet,
                    label="DIRECT_PREGRASP_V2_CONTACT_FREE_X_0P45MM",
                )
                _, _, terminated, truncated, _ = outputs
                active = preflight._active_termination_names(env, terminated, truncated)
                after_move, _ = _ee_root_pose(env, p0a)
                task_metrics = preflight._task_metrics(env, g2_lift_task_mdp)
                peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
                forbidden_peak = float(
                    max((max(values) for values in peaks.values()), default=0.0)
                )
                smoke_pass = bool(
                    move_receipt["single_consumption"]
                    and not bool(terminated.reshape(-1)[0].item())
                    and not bool(truncated.reshape(-1)[0].item())
                    and not active
                    and forbidden_peak <= 1.0e-6
                    and float(task_metrics["inner_contact_force_n"]) <= 0.0
                    and float(task_metrics["outer_contact_force_n"]) <= 0.0
                    and view_receipt["head_rgb_visible"]
                    and view_receipt["wrist_rgb_visible"]
                )
                views.close()
                _atomic_json(
                    output,
                    {
                        "schema": "g2_candidate_a_left_arm_down_v2_runtime_smoke_v1",
                        "baseline_variant": BASELINE_NAME,
                        "left_arm_posture_id": LEFT_ARM_POSTURE_ID,
                        "source_freeze": freeze_before,
                        "direct_init_receipt": direct_init_receipt,
                        "operator_view_receipt": view_receipt,
                        "terminal_keyboard_input": "STATIC_PTY_REGRESSION_ONLY",
                        "small_move_requested_m": [0.00045, 0.0, 0.0],
                        "small_move_observed_m": (after_move - before_move).tolist(),
                        "single_consumption": bool(move_receipt["single_consumption"]),
                        "contact_count": 0,
                        "forbidden_collision_count": int(forbidden_peak > 1.0e-6),
                        "startup_curobo_planning_count": 0,
                        "training_started": False,
                        "functional_pass": smoke_pass,
                    },
                )
                return 0 if smoke_pass else 2

            if keyboard_v3_collection_root is not None:
                if keyboard_v3_episode_id is None:
                    raise RuntimeError("KEYBOARD_V3_EPISODE_ID_MISSING")
                from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_live_driver import (
                    run_keyboard_v3_one_shot,
                )
                from geniesim.rl.isaaclab.g2_policy_branch.omnipicker_product_contract import (
                    OMNIPICKER_MANUAL_URL,
                    OMNIPICKER_PRODUCT_CONTRACT,
                )

                baseline_metadata = {
                    "baseline_variant": BASELINE_NAME,
                    "left_arm_posture_id": LEFT_ARM_POSTURE_ID,
                    "pregrasp_init_source": "CUROBO_DATASET",
                    "pregrasp_sample_id": sample.sample_id,
                    "cube_sample_id": sample.sample_id,
                    "head_camera": "ENABLED",
                    "wrist_camera": "ENABLED",
                    "control_source": "TERMINAL_KEYBOARD",
                    "omnipicker_product_model": "OmniPicker",
                    "omnipicker_manual_authority_url": OMNIPICKER_MANUAL_URL,
                    "omnipicker_manual_hardware_version": "1.2",
                    "omnipicker_pcba_version": "UNRESOLVED_20_OR_30",
                    "omnipicker_firmware_version": "UNRESOLVED_RUNTIME_DEVICE",
                    "omnipicker_maximum_gripping_force_n": (
                        OMNIPICKER_PRODUCT_CONTRACT.maximum_gripping_force_n
                    ),
                    "omnipicker_physical_pad_touch_sensor_available": False,
                    "omnipicker_protocol_force_feedback_semantics": (
                        OMNIPICKER_PRODUCT_CONTRACT.protocol_force_feedback_semantics
                    ),
                    "sim_contact_telemetry_role": (
                        OMNIPICKER_PRODUCT_CONTRACT.simulator_contact_role
                    ),
                }
                handoff_snapshot = _capture_keyboard_v3_handoff_state(env)

                def _run_direct_episode(
                    episode_id: str,
                    report_path: Path,
                    restore_receipt: Mapping[str, Any] | None,
                ) -> int:
                    return run_keyboard_v3_one_shot(
                        app=app,
                        env=env,
                        p0a=p0a,
                        preflight=preflight,
                        task_mdp=g2_lift_task_mdp,
                        counter=counter,
                        deferred_port=deferred_port,
                        latch=latch,
                        plan=keyboard_v3_plan,
                        collection_root=keyboard_v3_collection_root,
                        episode_id=episode_id,
                        report_path=report_path,
                        maximum_operator_steps=keyboard_v3_maximum_steps,
                        source_freeze_before=freeze_before,
                        source_freeze_provider=lambda: _source_freeze(
                            keyboard_v3_branch=True
                        ),
                        selected_asset_path=selected_asset,
                        selected_asset_sha256=selected_hash,
                        planner_path_records=[],
                        planner_maximum_forbidden_contact_force_n=forbidden_peak,
                        planner_final_error_m=direct_ee_error_m,
                        planner_settled=post_refresh_target_error <= 1.0e-6,
                        terminal_keyboard=True,
                        terminal_translation_step_m=keyboard_v3_translation_step_m,
                        terminal_motion_min_interval_s=(
                            keyboard_v3_motion_min_interval_s
                        ),
                        terminal_smoothing_steps=(
                            keyboard_v3_terminal_smoothing_steps
                        ),
                        baseline_metadata=baseline_metadata,
                        direct_init_receipt=(
                            dict(restore_receipt)
                            if restore_receipt is not None
                            else direct_init_receipt
                        ),
                    )

                def _restore_direct_episode() -> Mapping[str, Any]:
                    return _restore_keyboard_v3_handoff_state(
                        env=env,
                        preflight=preflight,
                        counter=counter,
                        deferred_port=deferred_port,
                        open_packet=open_packet,
                        latch=latch,
                        snapshot=handoff_snapshot,
                    )

                return _run_keyboard_v3_session(
                    collection_root=keyboard_v3_collection_root,
                    initial_episode_id=keyboard_v3_episode_id,
                    initial_report_path=output,
                    continuous_session=keyboard_v3_continuous_session,
                    maximum_episodes=keyboard_v3_session_max_episodes,
                    startup_curobo_plan_count=0,
                    initial_restore_receipt=direct_init_receipt,
                    run_episode=_run_direct_episode,
                    restore_for_next_episode=_restore_direct_episode,
                )

        if candidate_a_contact_free_runtime_replan:
            runtime_replan_input_telemetry = {}
            runtime_replan_input_telemetry_path = output.with_name(
                "RUNTIME_REPLAN_POST_RESET_INPUT.json"
            )
            if keyboard_v3_collection_root is not None:
                from geniesim.rl.isaaclab.g2_policy_branch.contact_free_runtime_replan import (
                    ContactFreeRuntimeReplanRequest,
                )
                from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_collection_contract import (
                    NearGraspCondition,
                )
                from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
                    plan_keyboard_v3_near_grasp,
                )

                planner_robot = env.scene["robot"]
                planner_arm_term = env.action_manager.get_term("arm_action")
                planner_q = (
                    _tensor(planner_robot.data.joint_pos)[0, planner_arm_term._joint_ids]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )
                planner_cube = _cube_root_position(env)
                request = ContactFreeRuntimeReplanRequest(
                    cube_center_root_m=tuple(float(value) for value in planner_cube),
                    right_arm_q_rad=tuple(float(value) for value in planner_q),
                    seed=int(seed),
                ).validated()
                condition = NearGraspCondition(
                    condition_id=f"backoff_{int(round(keyboard_v3_backoff_m * 1000)):02d}mm_center",
                    backoff_m=float(keyboard_v3_backoff_m),
                    start_offset_xyz_m=(0.0, 0.0, 0.0),
                )
                keyboard_v3_plan = plan_keyboard_v3_near_grasp(
                    request, condition
                ).validated()
                runtime_plan = keyboard_v3_plan
                runtime_replan_receipt = keyboard_v3_plan.planner_only_receipt()
                runtime_replan_input_telemetry.update(
                    {
                        "stage": "POST_RESET_KEYBOARD_V3_NEAR_GRASP_PLAN",
                        "planner_validation_completed": True,
                        "planner_input_cube_center_root_m": planner_cube.tolist(),
                        "planner_input_right_arm_q_rad": planner_q.tolist(),
                        "keyboard_action_submission_count": 0,
                    }
                )
                _atomic_json(
                    runtime_replan_input_telemetry_path,
                    runtime_replan_input_telemetry,
                )
            else:
                runtime_plan, runtime_replan_receipt = _plan_contact_free_after_reset(
                    env=env,
                    seed=seed,
                    input_telemetry=runtime_replan_input_telemetry,
                    input_telemetry_path=runtime_replan_input_telemetry_path,
                )
            if (
                runtime_replan_canonical_contact_free_collection_output is not None
                or candidate_a_bc_validation_output is not None
            ):
                from geniesim.rl.isaaclab.g2_policy_branch.independent_contact_free_collection import (
                    IndependentCollectionEpisode,
                    runtime_plan_sidecar,
                )

                episode = IndependentCollectionEpisode(
                    episode_id=str(
                        independent_collection_episode_id
                        if runtime_replan_canonical_contact_free_collection_output
                        is not None
                        else candidate_a_bc_validation_episode_id
                    ),
                    seed=int(seed),
                ).validated()
                sidecar = runtime_plan_sidecar(
                    episode=episode,
                    runtime_replan_receipt=runtime_replan_receipt,
                    post_reset_input=runtime_replan_input_telemetry,
                    runner_source_sha256=_sha256(Path(__file__).resolve()),
                    candidate_asset_sha256=selected_hash,
                )
                sidecar_path = output.with_name("RUNTIME_REPLAN_PLAN_SIDECAR.json")
                if sidecar_path.exists():
                    raise RuntimeError("RUNTIME_REPLAN_PLAN_SIDECAR_REFUSES_OVERWRITE")
                _atomic_json(sidecar_path, sidecar)
                runtime_replan_plan_sidecar_sha256 = _sha256(sidecar_path)
            planned_ee = np.asarray(
                runtime_plan.ee_position_root_m, dtype=np.float64
            )
            planned_q = np.asarray(runtime_plan.q_rad, dtype=np.float64)
            if planned_ee.shape[0] < 2 or planned_q.shape != (
                planned_ee.shape[0],
                7,
            ):
                raise RuntimeError("RUNTIME_REPLAN_EXECUTION_PATH_SHAPE_INVALID")
            # The runtime plan ends at PREGRASP.  The existing executor owns
            # the contact-free pad-frame extension to the 3-cm OPEN handoff;
            # hence every MotionGen waypoint is coarse and no fine waypoint
            # may be skipped or substituted.
            fine_start_index = int(planned_ee.shape[0] - 1)
            if keyboard_v3_collection_root is not None:
                selected_waypoint_indices = list(range(int(planned_ee.shape[0])))
            print("RUNTIME_REPLAN_AFTER_RESET_DONE", flush=True)

        candidate_contact_free_mode: str | None = None
        if candidate_a_contact_free:
            # Preserve the historical frozen-replay selector as an explicit
            # branch.  The runtime-replan route is intentionally a separate
            # selector and cannot silently inherit a replay.
            candidate_contact_free_mode = "FROZEN_REPLAY"
        elif candidate_a_contact_free_runtime_replan:
            candidate_contact_free_mode = "POST_RESET_RUNTIME_REPLAN"
        if (
            candidate_contact_free_mode is not None
            and keyboard_v3_collection_root is None
        ):
            target_telemetry = _TargetLimiterTelemetry(arm_term)
            target_telemetry.install()
            return _run_candidate_a_contact_free_smoke(
                output=output,
                replay=replay,
                runtime_replan_receipt=runtime_replan_receipt,
                runtime_replan_input_telemetry=runtime_replan_input_telemetry,
                runtime_replan_input_telemetry_path=runtime_replan_input_telemetry_path,
                seed=seed,
                env=env,
                p0a=p0a,
                preflight=preflight,
                g2_lift_task_mdp=g2_lift_task_mdp,
                counter=counter,
                deferred_port=deferred_port,
                latch=latch,
                settle=settle,
                freeze_before=freeze_before,
                asset_receipt=asset_receipt,
                diagnostic_asset_selection=diagnostic_asset_selection,
                task_contract=task_contract,
                task_geometry_binding=task_geometry_binding,
                task_geometry_readback=task_geometry_readback,
                p0a_provenance=p0a_provenance,
                selected_asset=selected_asset,
                selected_hash=selected_hash,
                planned_ee=planned_ee,
                fine_start_index=fine_start_index,
                ordinary_tracking_cap=ordinary_tracking_cap,
                critical_tracking_cap=critical_tracking_cap,
                cosine_p05=cosine_p05,
                cosine_p10=cosine_p10,
                target_telemetry=target_telemetry,
                canonical_collection_output=canonical_collection_output,
                canonical_capture_rate_hz=canonical_capture_rate_hz,
                runtime_replan_plan_sidecar_sha256=(
                    runtime_replan_plan_sidecar_sha256
                ),
                independent_collection_episode_id=independent_collection_episode_id,
                contact_free_bc_playback_checkpoint=(
                    contact_free_bc_playback_checkpoint
                ),
                contact_free_bc_playback_checkpoint_sha256=(
                    contact_free_bc_playback_checkpoint_sha256
                ),
                candidate_a_bc_validation_output=(
                    candidate_a_bc_validation_output
                ),
                candidate_a_bc_validation_episode_id=(
                    candidate_a_bc_validation_episode_id
                ),
            )

        if open_only:
            robot = env.scene["robot"]
            settled_initial_state = _ab_settled_state_snapshot(env, robot)
            settled_initial_state["open_settle_status"] = "PASS"
            freeze_after = _source_freeze(
                keyboard_v3_branch=keyboard_v3_collection_root is not None
            )
            selected_asset_hash_after = _sha256(selected_asset)
            if selected_asset_hash_after != selected_hash:
                raise RuntimeError("CUROBO_CUSTOM_ASSET_CHANGED_DURING_OPEN_PROBE")
            focused_margins = [
                float(row["limit_margin_rad"])
                for name, row in settled_initial_state["joints"].items()
                if name.endswith("_joint3") or name.endswith("_joint4")
            ]
            if not focused_margins or not all(math.isfinite(value) for value in focused_margins):
                raise RuntimeError("CUROBO_OPEN_PROBE_LIMIT_MARGIN_INVALID")
            open_report = {
                "schema": "g2_curobo_open_feasibility_probe_v1",
                "scope": "RESET_AND_CANONICAL_OPEN_ONLY_NO_PLANNER_NO_BC_NO_CONTACT",
                "source_freeze_before": freeze_before,
                "source_freeze_after": freeze_after,
                "asset_binding": asset_receipt.as_dict(),
                "diagnostic_asset_selection": diagnostic_asset_selection,
                "settled_initial_state": settled_initial_state,
                "task_contract": task_contract,
                "task_geometry_binding": task_geometry_binding,
                "task_geometry_readback": task_geometry_readback,
                "p0a_provenance": p0a_provenance,
                "open_settle": settle,
                "minimum_joint3_joint4_limit_margin_rad": min(focused_margins),
                "lifecycle_counts": counter.summary(),
                "checks": {
                    "canonical_open_target_reached": True,
                    "full_46dof_fd_safety_pass": bool(
                        settle["full_46dof_fd_safety"]["pass"]
                    ),
                    "passive_joint_not_at_endpoint": min(focused_margins)
                    > HARD_STOP_LIMIT_NUMERICAL_TOLERANCE_RAD,
                    "single_consumption": bool(
                        settle["all_setup_packets_single_consumption"]
                    ),
                    "no_nan": True,
                    "no_termination_during_settle": True,
                    "planner_executed": False,
                    "bc_executed": False,
                    "contact_executed": False,
                    "selected_asset_hash_stable": True,
                },
                "verdict": {
                    "OPEN_FEASIBILITY": "PASS",
                    "PLANNER_ONLY_LIVE_SMOKE": "NOT_EXECUTED_OPEN_ONLY",
                    "CUROBO_BC_GRU_CLOSED_LOOP": "NOT_EXECUTED_OPEN_ONLY",
                    "FUNCTIONAL_VERDICT": "PASS",
                    "PROCESS_VERDICT": "PENDING_PARENT",
                    "SAC": "OFF",
                    "TRAINING": "NOT_EXECUTED",
                },
                "cleanup": cleanup,
            }
            if not all(
                bool(value)
                for key, value in open_report["checks"].items()
                if key not in {"planner_executed", "bc_executed", "contact_executed"}
            ):
                open_report["verdict"]["OPEN_FEASIBILITY"] = "FAIL"
                open_report["verdict"]["FUNCTIONAL_VERDICT"] = "FAIL"
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(output, open_report)
            print("RUNTIME_END", flush=True)
            print("REPORT_SAVED", flush=True)
            return 0 if open_report["verdict"]["FUNCTIONAL_VERDICT"] == "PASS" else 2

        bc_enabled = bc_checkpoint is not None
        bc_model = None
        bc_checkpoint_receipt = None
        bc_provider = None
        bc_observation_source = None
        bc_controller_binding = None
        bc_observation_binding = None
        bc_previous = PreviousAcceptedPolicyAction()
        bc_previous.reset()
        bc_hidden = None
        bc_history_started = False
        grasp_ready_region = None
        if bc_enabled:
            if (
                bc_checkpoint_sha256 is None
                or len(bc_checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in bc_checkpoint_sha256)
            ):
                raise RuntimeError("CUROBO_BC_CHECKPOINT_SHA256_REQUIRED")
            if bc_closed_loop_steps <= 0:
                raise RuntimeError("CUROBO_BC_CLOSED_LOOP_STEPS_REQUIRED")
            if grasp_ready_geometry is None or not grasp_ready_geometry.is_file():
                raise RuntimeError("CUROBO_BC_GRASP_READY_GEOMETRY_REQUIRED")
            geometry_payload = json.loads(
                grasp_ready_geometry.read_text(encoding="utf-8")
            )
            grasp_ready_region = region_from_mapping(
                geometry_payload["grasp_ready_region"]
            )
            (
                bc_provider,
                bc_observation_source,
                bc_controller_binding,
            ) = p0a._runtime_observation_provider(env, freeze_before)
            bc_observation_binding = p0c._runtime_binding(env)
            bc_model, bc_checkpoint_receipt = preflight._load_bc_checkpoint_strict(
                bc_checkpoint,
                expected_file_sha256=bc_checkpoint_sha256,
                device=env.device,
            )
            bc_model.config.data_semantics.assert_compatible(
                bc_observation_binding.data_semantics
            )
            print("BC_CHECKPOINT_LOADED", flush=True)

        robot = env.scene["robot"]
        arm_term = env.action_manager.get_term("arm_action")
        arm_indices = np.asarray([int(value) for value in arm_term._joint_ids], dtype=np.int64)
        arm_joint_names = [robot.joint_names[index] for index in arm_indices]
        passive_indices = np.asarray(
            [robot.joint_names.index(name) for name in G2_FULL_PASSIVE_OR_MIMIC_JOINT_NAMES],
            dtype=np.int64,
        )
        settled_initial_state = _ab_settled_state_snapshot(env, robot)
        settled_initial_state["open_settle_status"] = "PASS"
        initial_ee = _ee_root_position(env)
        initial_cube = _cube_root_position(env)
        initial_distance = float(np.linalg.norm(initial_cube - initial_ee))
        final_target = planned_ee[-1]
        coarse_progress_target = planned_ee[fine_start_index]
        progress_metric_config = ProgressMetricConfig(
            progress_deadband_m=progress_deadband_m,
            stall_grace_steps=stall_grace_steps,
            max_stall_steps=max_stall_steps,
            retreat_threshold_m=retreat_threshold_m,
        )
        coarse_progress_tracker = BestSoFarProgress(progress_metric_config)
        fine_progress_tracker = BestSoFarProgress(progress_metric_config)
        q_samples = [
            _tensor(robot.data.joint_pos)[0].detach().to("cpu").numpy().astype(np.float64)
        ]
        target_samples = [
            _tensor(arm_term.last_emitted_joint_position_target)[0]
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        ]
        audit = FullArticulationFDSafetyAudit.start(
            robot.data.joint_pos,
            robot.joint_names,
            dt_s=float(env.step_dt),
        )
        maximum_forbidden_contact_force_n = 0.0
        controller_reject_count = 0
        terminated_early = False
        all_single_consumption = True
        previous_waypoint_index: int | None = None

        if passive_contact_telemetry_output is not None:
            passive_contact_telemetry = _PassiveContactPhysicsTelemetry(
                env,
                robot,
                g2_lift_task_mdp,
                torch_module=torch,
                warp_module=wp,
                stage=omni.usd.get_context().get_stage(),
                stop_on_hard_stop=stop_on_hard_stop,
            )
            passive_contact_telemetry.install()

        if attribution_output is not None:
            target_telemetry = _TargetLimiterTelemetry(arm_term)
            target_telemetry.install()
            baseline_ee, baseline_quaternion = _ee_root_pose(env)
            baseline_measured_q = (
                _tensor(robot.data.joint_pos)[0, arm_term._joint_ids]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            baseline_measured_qdot = (
                _tensor(robot.data.joint_vel)[0, arm_term._joint_ids]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            baseline_target = (
                _tensor(arm_term.last_emitted_joint_position_target)[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            baseline_limiter_velocity = (
                _tensor(arm_term.current_target_velocity_rad_s)[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            telemetry_samples.append(
                {
                    "policy_step": 0,
                    "timestamp_s": 0.0,
                    "waypoint_index": -1,
                    "waypoint_boundary": False,
                    "waypoint_repeat_index": 0,
                    "waypoint_repeat_transition": False,
                    "phase": "BASELINE",
                    "raw_planner_q": planned_q[0],
                    "previous_raw_planner_q": planned_q[0],
                    "next_raw_planner_q": planned_q[0],
                    "pre_rate_limit_target_q": baseline_target,
                    "post_rate_limit_target_q": baseline_target,
                    "limiter_accumulator_velocity": baseline_limiter_velocity,
                    "limiter_reset": False,
                    "limiter_synchronize": False,
                    "controller_reject": False,
                    "clipping": False,
                    "measured_q": baseline_measured_q,
                    "measured_qdot": baseline_measured_qdot,
                    "ee_position_root_m": baseline_ee,
                    "ee_quaternion_root_wxyz": baseline_quaternion,
                    "planner_ee_target_root_m": planned_ee[0],
                    "ee_tracking_error_m": float(
                        np.linalg.norm(planned_ee[0] - baseline_ee)
                    ),
                    "maximum_forbidden_contact_force_n": 0.0,
                }
            )

        def advance_bc_history(*, phase: str) -> dict[str, Any] | None:
            """Roll the frozen BC GRU through the action history without authority."""

            nonlocal bc_hidden, bc_history_started
            if not bc_enabled:
                return None
            assert bc_model is not None
            assert bc_provider is not None
            assert bc_observation_binding is not None
            observation, capture = p0c._capture_observation(
                env=env,
                p0a=p0a,
                provider=bc_provider,
                previous=bc_previous,
                hidden_reset=not bc_history_started,
                binding=bc_observation_binding,
            )
            with torch.inference_mode():
                network_output = bc_model(observation, initial_hidden=bc_hidden)
            bc_hidden = network_output.final_hidden.detach()
            bc_history_started = True
            probability = float(
                network_output.gripper_probability[0, -1, 0].item()
            )
            clipped_probability = min(max(probability, 1.0e-8), 1.0 - 1.0e-8)
            monitor = {
                "phase": phase,
                "history_index": len(bc_history_records),
                "sequence_id": capture.get("sequence_id"),
                "hidden_reset": bool(capture["hidden_reset"]),
                "previous_policy_action": capture["previous_policy_action"],
                "p_close_advisory_only": probability,
                "gripper_logit_advisory_only": float(
                    math.log(clipped_probability / (1.0 - clipped_probability))
                ),
                "hidden_l2_norm": float(torch.linalg.vector_norm(bc_hidden).item()),
            }
            bc_history_records.append(monitor)
            return monitor

        def execute_target(
            target: np.ndarray,
            label: str,
            waypoint_index: int | None,
            waypoint_repeat_index: int,
            phase: str,
        ) -> None:
            nonlocal maximum_forbidden_contact_force_n, controller_reject_count
            nonlocal terminated_early, all_single_consumption
            nonlocal previous_waypoint_index
            bc_monitor = advance_bc_history(phase=phase)
            before = _ee_root_position(env)
            handoff = planner_waypoint_to_canonical_4d(
                planner_target_root_m=target,
                measured_ee_root_m=before,
            )
            packet, _, derivation = p0a._build_authoritative_packet(
                high_level=handoff.policy_action,
                batch_size=env.num_envs,
                device=env.device,
                latch=latch,
            )
            if tuple(float(value) for value in packet.values[3:7]) != (0.0, 0.0, 0.0, 0.0):
                raise RuntimeError("CUROBO_LIVE_ROTATION_OR_ELBOW_NONZERO")
            if packet.gripper_intent is not AbstractGripperIntent.OPEN:
                raise RuntimeError("CUROBO_LIVE_GRIPPER_NOT_OPEN")
            generations_before = (
                target_telemetry.begin_policy_step()
                if target_telemetry is not None
                else None
            )
            if passive_contact_telemetry is not None:
                passive_contact_telemetry.set_context(
                    policy_step=len(path_records),
                    bc_step=None,
                    phase=phase,
                    gripper_intent=packet.gripper_intent.value,
                    close_onset=False,
                    clipping=bool(handoff.clipping_applied),
                )
            try:
                outputs, receipt = preflight._consume_once(
                    env=env,
                    counter=counter,
                    deferred_port=deferred_port,
                    packet=packet,
                    label=label,
                )
            except Exception:
                controller_reject_count += 1
                raise
            limiter_step = (
                target_telemetry.end_policy_step(generations_before)
                if target_telemetry is not None and generations_before is not None
                else None
            )
            _, reward, terminated, truncated, _ = outputs
            latch.commit(packet.gripper_intent)
            if bc_enabled:
                bc_previous.record_accepted(handoff.policy_action)
            audit.observe(robot.data.joint_pos, step=len(path_records) + 1)
            q_now = _tensor(robot.data.joint_pos)[0].detach().to("cpu").numpy().astype(np.float64)
            q_samples.append(q_now)
            target_samples.append(
                _tensor(arm_term.last_emitted_joint_position_target)[0]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            after = _ee_root_position(env)
            cube = _cube_root_position(env)
            if phase == "COARSE_REACH":
                progress_metric = coarse_progress_tracker.observe(
                    float(np.linalg.norm(coarse_progress_target - after))
                )
                progress_metric["phase_target"] = "PREGRASP"
            elif phase in ("FINE_APPROACH", "FINAL_CONVERGENCE"):
                progress_metric = fine_progress_tracker.observe(
                    float(np.linalg.norm(final_target - after))
                )
                progress_metric["phase_target"] = "SAFE_HANDOFF"
            else:
                progress_metric = {
                    "progress_metric_active": False,
                    "current_distance_cm": None,
                    "best_distance_cm": None,
                    "new_best_progress_mm": 0.0,
                    "best_progress_mm": 0.0,
                    "stall_counter": 0,
                    "progress_event": False,
                    "stall_event": False,
                    "max_stall_event": False,
                    "retreat_event": False,
                    "phase_target": None,
                }
            active_termination_names = preflight._active_termination_names(
                env, terminated, truncated
            )
            evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
            sensor_peaks = evaluator.sensor_peak_forces_n() if evaluator is not None else {}
            if sensor_peaks:
                maximum_forbidden_contact_force_n = max(
                    maximum_forbidden_contact_force_n,
                    max(max(values) for values in sensor_peaks.values()),
                )
            command_delta = np.asarray(handoff.metric_delta_root_m, dtype=np.float64)
            measured_delta = after - before
            record = {
                "step": len(path_records),
                "label": label,
                "waypoint_index": waypoint_index,
                "planner_target_root_m": target.astype(float).tolist(),
                "measured_before_root_m": before.astype(float).tolist(),
                "measured_after_root_m": after.astype(float).tolist(),
                "metric_delta_root_m": command_delta.astype(float).tolist(),
                "normalized_4d_action": list(handoff.normalized_action),
                "full_8d_packet": list(packet.values),
                "packet_derivation": derivation,
                "single_consumption": receipt,
                "tracking_error_after_m": float(np.linalg.norm(target - after)),
                "ee_cube_distance_m": float(np.linalg.norm(cube - after)),
                "action_goal_alignment_cosine": _cosine(command_delta, measured_delta),
                "reward": float(reward.reshape(-1)[0].item()),
                "terminated": bool(terminated.reshape(-1)[0].item()),
                "truncated": bool(truncated.reshape(-1)[0].item()),
                "active_termination_names": active_termination_names,
                "forbidden_contact_sensor_peak_n": sensor_peaks,
                "gripper_intent": packet.gripper_intent.value,
                "clipping_applied": handoff.clipping_applied,
                "phase": phase,
                "phase_progress": progress_metric,
                "bc_gru_history_monitor": bc_monitor,
            }
            path_records.append(record)
            if limiter_step is not None:
                policy_step = len(path_records)
                for substep in limiter_step["substeps"]:
                    telemetry_substeps.append(
                        {
                            **substep,
                            "policy_step": policy_step,
                            "waypoint_index": (
                                -1 if waypoint_index is None else waypoint_index
                            ),
                            "waypoint_repeat_index": waypoint_repeat_index,
                            "phase": phase,
                        }
                    )
                resolved_index = (
                    planned_q.shape[0] - 1
                    if waypoint_index is None
                    else waypoint_index
                )
                prior_index = max(0, resolved_index - 1)
                next_index = min(planned_q.shape[0] - 1, resolved_index + 1)
                last_substep = limiter_step["substeps"][-1]
                ee_position, ee_quaternion = _ee_root_pose(env)
                measured_q = (
                    _tensor(robot.data.joint_pos)[0, arm_term._joint_ids]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )
                measured_qdot = (
                    _tensor(robot.data.joint_vel)[0, arm_term._joint_ids]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )
                final_post_target = (
                    _tensor(arm_term.last_emitted_joint_position_target)[0]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )
                telemetry_samples.append(
                    {
                        "policy_step": policy_step,
                        "timestamp_s": policy_step * float(env.step_dt),
                        "waypoint_index": (
                            -1 if waypoint_index is None else waypoint_index
                        ),
                        "waypoint_boundary": bool(
                            waypoint_index is not None
                            and waypoint_index != previous_waypoint_index
                        ),
                        "waypoint_repeat_index": waypoint_repeat_index,
                        "waypoint_repeat_transition": bool(
                            waypoint_index is not None and waypoint_repeat_index > 1
                        ),
                        "phase": phase,
                        "raw_planner_q": planned_q[resolved_index],
                        "previous_raw_planner_q": planned_q[prior_index],
                        "next_raw_planner_q": planned_q[next_index],
                        "pre_rate_limit_target_q": last_substep[
                            "pre_rate_limit_target"
                        ],
                        "post_rate_limit_target_q": final_post_target,
                        "limiter_accumulator_velocity": last_substep[
                            "final_accumulator_velocity"
                        ],
                        "limiter_reset": bool(
                            limiter_step["reset_during_policy_step"]
                        ),
                        "limiter_synchronize": bool(
                            limiter_step["synchronize_during_policy_step"]
                        ),
                        "controller_reject": False,
                        "clipping": bool(handoff.clipping_applied),
                        "measured_q": measured_q,
                        "measured_qdot": measured_qdot,
                        "ee_position_root_m": ee_position,
                        "ee_quaternion_root_wxyz": ee_quaternion,
                        "planner_ee_target_root_m": target,
                        "ee_tracking_error_m": float(np.linalg.norm(target - ee_position)),
                        "maximum_forbidden_contact_force_n": float(
                            max(
                                (max(values) for values in sensor_peaks.values()),
                                default=0.0,
                            )
                        ),
                    }
                )
                previous_waypoint_index = waypoint_index
            _atomic_json(
                partial_path,
                {
                    "schema": "g2_curobo_live_partial_telemetry_v2",
                    "path_record_count": len(path_records),
                    "path_record_ring": path_records[-8:],
                    "limiter_policy_sample_ring": telemetry_samples[-4:],
                    "limiter_substep_ring": telemetry_substeps[-8:],
                    "progress": progress,
                    "latest_measured_joint_q": q_samples[-1],
                    "latest_emitted_target_q": target_samples[-1],
                    "source_freeze_manifest_sha256": freeze_before[
                        "manifest_sha256"
                    ],
                },
            )
            all_single_consumption &= bool(receipt["single_consumption"])
            if record["terminated"] or record["truncated"] or active_termination_names:
                terminated_early = True
                progress["timeout_episode_step"] = int(
                    _tensor(env.episode_length_buf).reshape(-1)[0].item()
                )
                raise RuntimeError(f"CUROBO_LIVE_UNEXPECTED_TERMINATION:{active_termination_names}")

        waypoint_tracking_steps: list[int] = []
        coarse_tracking_steps = 0
        fine_tracking_steps = 0
        early_advance_count = 0
        critical_handoff_repeat_count = 0
        critical_waypoint_index = fine_start_index
        for effective_index, original_index in enumerate(selected_waypoint_indices):
            target = planned_ee[original_index]
            execution_phase = (
                "COARSE_REACH"
                if original_index < fine_start_index
                else "FINE_APPROACH"
            )
            reached = False
            tracking_cap = (
                critical_tracking_cap
                if timing_optimized and original_index == critical_waypoint_index
                else ordinary_tracking_cap
            )
            for tracking_step in range(1, tracking_cap + 1):
                execute_target(
                    target,
                    f"CUROBO_WAYPOINT_{original_index:03d}_TRACK_{tracking_step:02d}",
                    original_index,
                    tracking_step,
                    execution_phase,
                )
                progress["waypoint_tracking_steps"] = int(
                    progress["waypoint_tracking_steps"]
                ) + 1
                if execution_phase == "COARSE_REACH":
                    coarse_tracking_steps += 1
                else:
                    fine_tracking_steps += 1
                if original_index == critical_waypoint_index:
                    critical_handoff_repeat_count += 1
                if float(path_records[-1]["tracking_error_after_m"]) <= PREGRASP_TOLERANCE_M:
                    reached = True
                    waypoint_tracking_steps.append(tracking_step)
                    if tracking_step < tracking_cap:
                        early_advance_count += 1
                    break
            progress["waypoints_submitted"] = effective_index + 1
            if not reached and (
                not timing_optimized or original_index == critical_waypoint_index
            ):
                raise RuntimeError(
                    f"CUROBO_WAYPOINT_TRACKING_TIMEOUT:index={original_index}:"
                    f"error_m={path_records[-1]['tracking_error_after_m']}"
                )
            if not reached:
                waypoint_tracking_steps.append(tracking_cap)

        final_convergence_steps = 0
        if timing_optimized:
            for final_convergence_steps in range(1, final_convergence_cap + 1):
                execute_target(
                    final_target,
                    f"CUROBO_FINAL_CONVERGENCE_{final_convergence_steps:03d}",
                    planned_ee.shape[0] - 1,
                    final_convergence_steps,
                    "FINAL_CONVERGENCE",
                )
                progress["final_convergence_steps"] = final_convergence_steps
                if float(np.linalg.norm(final_target - _ee_root_position(env))) <= PREGRASP_TOLERANCE_M:
                    break
            if float(np.linalg.norm(final_target - _ee_root_position(env))) > PREGRASP_TOLERANCE_M:
                raise RuntimeError("CUROBO_FINAL_CONVERGENCE_TIMEOUT")

        settle_steps = 0
        for settle_steps in range(1, final_settle_cap + 1):
            # A neutral heartbeat retains the controller endpoint without
            # adding the remaining measured error to it a second time.
            settle_origin = _ee_root_position(env)
            execute_target(
                settle_origin,
                f"CUROBO_FINAL_SETTLE_{settle_steps:03d}",
                None,
                settle_steps,
                "FINAL_SETTLE",
            )
            progress["final_settle_steps"] = settle_steps
            final_error = float(np.linalg.norm(final_target - _ee_root_position(env)))
            if final_error <= PREGRASP_TOLERANCE_M and audit.settled_for_cache:
                break

        for flush_step in range(1, telemetry_flush_steps + 1):
            flush_origin = _ee_root_position(env)
            execute_target(
                flush_origin,
                f"CUROBO_TELEMETRY_FLUSH_{flush_step:03d}",
                None,
                flush_step,
                "TELEMETRY_FLUSH",
            )
            progress["telemetry_flush_steps"] = flush_step

        if keyboard_v3_collection_root is not None:
            if keyboard_v3_plan is None or keyboard_v3_episode_id is None:
                raise RuntimeError("KEYBOARD_V3_RUNTIME_PLAN_OR_EPISODE_MISSING")
            from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
                BASELINE_NAME,
                LEFT_ARM_POSTURE_ID,
            )
            from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_live_driver import (
                run_keyboard_v3_one_shot,
            )
            from geniesim.rl.isaaclab.g2_policy_branch.omnipicker_product_contract import (
                OMNIPICKER_MANUAL_URL,
                OMNIPICKER_PRODUCT_CONTRACT,
            )

            planner_final_error_m = float(
                np.linalg.norm(final_target - _ee_root_position(env))
            )
            runtime_sample_id = (
                str(direct_init_receipt["pregrasp_sample_id"])
                if direct_init_receipt is not None
                else (
                    f"runtime_curobo_seed_{seed}_target_"
                    f"{int(round(keyboard_v3_backoff_m * 1000.0))}mm"
                )
            )
            baseline_metadata = {
                "baseline_variant": BASELINE_NAME,
                "left_arm_posture_id": LEFT_ARM_POSTURE_ID,
                "pregrasp_init_source": "CUROBO_DATASET",
                "pregrasp_sample_id": runtime_sample_id,
                "cube_sample_id": runtime_sample_id,
                "head_camera": "ENABLED",
                "wrist_camera": "ENABLED",
                "control_source": "TERMINAL_KEYBOARD",
                "omnipicker_product_model": "OmniPicker",
                "omnipicker_manual_authority_url": OMNIPICKER_MANUAL_URL,
                "omnipicker_manual_hardware_version": "1.2",
                "omnipicker_pcba_version": "UNRESOLVED_20_OR_30",
                "omnipicker_firmware_version": "UNRESOLVED_RUNTIME_DEVICE",
                "omnipicker_maximum_gripping_force_n": (
                    OMNIPICKER_PRODUCT_CONTRACT.maximum_gripping_force_n
                ),
                "omnipicker_physical_pad_touch_sensor_available": False,
                "omnipicker_protocol_force_feedback_semantics": (
                    OMNIPICKER_PRODUCT_CONTRACT.protocol_force_feedback_semantics
                ),
                "sim_contact_telemetry_role": (
                    OMNIPICKER_PRODUCT_CONTRACT.simulator_contact_role
                ),
            }
            handoff_snapshot = _capture_keyboard_v3_handoff_state(env)

            def _run_runtime_replan_episode(
                episode_id: str,
                report_path: Path,
                restore_receipt: Mapping[str, Any] | None,
            ) -> int:
                return run_keyboard_v3_one_shot(
                    app=app,
                    env=env,
                    p0a=p0a,
                    preflight=preflight,
                    task_mdp=g2_lift_task_mdp,
                    counter=counter,
                    deferred_port=deferred_port,
                    latch=latch,
                    plan=keyboard_v3_plan,
                    collection_root=keyboard_v3_collection_root,
                    episode_id=episode_id,
                    report_path=report_path,
                    maximum_operator_steps=keyboard_v3_maximum_steps,
                    source_freeze_before=freeze_before,
                    source_freeze_provider=lambda: _source_freeze(
                        keyboard_v3_branch=True
                    ),
                    selected_asset_path=selected_asset,
                    selected_asset_sha256=selected_hash,
                    planner_path_records=path_records,
                    planner_maximum_forbidden_contact_force_n=(
                        maximum_forbidden_contact_force_n
                    ),
                    planner_final_error_m=planner_final_error_m,
                    planner_settled=bool(audit.settled_for_cache),
                    terminal_keyboard=True,
                    terminal_translation_step_m=keyboard_v3_translation_step_m,
                    terminal_motion_min_interval_s=(
                        keyboard_v3_motion_min_interval_s
                    ),
                    terminal_smoothing_steps=(
                        keyboard_v3_terminal_smoothing_steps
                    ),
                    # Dataset metadata has an exact, frozen key set. Session
                    # orchestration provenance belongs in the separate
                    # KEYBOARD_V3_SESSION report, never in actor data.
                    baseline_metadata=baseline_metadata,
                    direct_init_receipt=(
                        dict(restore_receipt)
                        if restore_receipt is not None
                        else direct_init_receipt
                    ),
                )

            def _restore_runtime_replan_episode() -> Mapping[str, Any]:
                return _restore_keyboard_v3_handoff_state(
                    env=env,
                    preflight=preflight,
                    counter=counter,
                    deferred_port=deferred_port,
                    open_packet=open_packet,
                    latch=latch,
                    snapshot=handoff_snapshot,
                )

            return _run_keyboard_v3_session(
                collection_root=keyboard_v3_collection_root,
                initial_episode_id=keyboard_v3_episode_id,
                initial_report_path=output,
                continuous_session=keyboard_v3_continuous_session,
                maximum_episodes=keyboard_v3_session_max_episodes,
                startup_curobo_plan_count=(
                    0 if keyboard_v3_direct_pregrasp_v2 else 1
                ),
                initial_restore_receipt=direct_init_receipt,
                run_episode=_run_runtime_replan_episode,
                restore_for_next_episode=_restore_runtime_replan_episode,
            )

        handoff_region_evaluation = None
        handoff_pregrasp_error_m = None
        first_close_step = None
        first_close_region_evaluation = None
        premature_close_count = 0
        bilateral_contact_observed = False
        stable_contact_observed = False
        first_bilateral_contact_step = None
        stable_hold_steps = 0
        bridge_steps_required = 0
        bridge_steps_completed = 0
        if bc_enabled:
            assert bc_model is not None
            assert bc_provider is not None
            assert bc_observation_binding is not None
            assert grasp_ready_region is not None
            handoff_position, handoff_orientation = _ee_root_pose(env)
            handoff_cube = _cube_root_position(env)
            handoff_pregrasp_error_m = float(
                np.linalg.norm(final_target - handoff_position)
            )
            handoff_region_evaluation = grasp_ready_region.contains(
                cube_minus_ee_root_m=handoff_cube - handoff_position,
                ee_orientation_xyzw=handoff_orientation,
            )
            if state_preserving_bridge:
                drain_physics_steps = minimum_velocity_drain_physics_steps(
                    _tensor(arm_term.current_target_velocity_rad_s)[0]
                    .detach()
                    .to("cpu")
                    .tolist(),
                    maximum_acceleration_rad_s2=float(
                        arm_term.cfg.maximum_joint_target_acceleration_rad_s2
                    ),
                    physics_dt_s=float(env.physics_dt),
                )
                bridge_steps_required = max(
                    1,
                    int(math.ceil(drain_physics_steps / int(cfg.decimation))),
                )
                if bridge_steps_required > 10:
                    raise RuntimeError(
                        "CUROBO_BC_BRIDGE_REQUIRED_STEPS_EXCEED_BOUND:"
                        f"{bridge_steps_required}"
                    )
            print("BC_HANDOFF_BEGIN", flush=True)
            for bc_step in range(bc_closed_loop_steps):
                observation, capture = p0c._capture_observation(
                    env=env,
                    p0a=p0a,
                    provider=bc_provider,
                    previous=bc_previous,
                    hidden_reset=not bc_history_started,
                    binding=bc_observation_binding,
                )
                with torch.inference_mode():
                    network_output = bc_model(observation, initial_hidden=bc_hidden)
                bc_hidden = network_output.final_hidden.detach()
                bc_history_started = True
                output_tensor = network_output.policy_action[:, -1]
                if tuple(output_tensor.shape) != (1, 4) or not bool(
                    torch.isfinite(output_tensor).all()
                ):
                    raise RuntimeError("CUROBO_BC_OUTPUT_INVALID")
                action_values = tuple(
                    float(value) for value in output_tensor[0].tolist()
                )
                if not all(abs(value) <= 1.0 for value in action_values[:3]) or not (
                    0.0 <= action_values[3] <= 1.0
                ):
                    raise RuntimeError("CUROBO_BC_OUTPUT_OUTSIDE_NORMALIZED_RANGE")
                probability = float(action_values[3])
                clipped_probability = min(
                    max(probability, 1.0e-8), 1.0 - 1.0e-8
                )
                position_before, orientation_before = _ee_root_pose(env)
                cube_before = _cube_root_position(env)
                close_region = grasp_ready_region.contains(
                    cube_minus_ee_root_m=cube_before - position_before,
                    ee_orientation_xyzw=orientation_before,
                )
                bridge_force_open = bool(
                    state_preserving_bridge
                    and bridge_steps_completed < bridge_steps_required
                )
                bridge_receipt = None
                if state_preserving_bridge:
                    bridge_receipt = planner_owned_xyz_bc_gripper_action(
                        planner_endpoint_root_m=final_target,
                        measured_ee_root_m=position_before,
                        bc_gripper_probability=probability,
                        bc_translation_advisory_only=action_values[:3],
                        force_open=bridge_force_open,
                    )
                    action = bridge_receipt.action
                else:
                    action = HighLevelPolicyAction.from_sequence(action_values)
                previous_intent = latch.intent
                packet, _, derivation = p0a._build_authoritative_packet(
                    high_level=action,
                    batch_size=env.num_envs,
                    device=env.device,
                    latch=latch,
                )
                if tuple(float(value) for value in packet.values[3:7]) != (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ):
                    raise RuntimeError("CUROBO_BC_ROTATION_OR_ELBOW_NONZERO")
                accepted_first_close = bool(
                    first_close_step is None
                    and previous_intent is AbstractGripperIntent.OPEN
                    and packet.gripper_intent is AbstractGripperIntent.CLOSE
                )
                if accepted_first_close:
                    first_close_step = bc_step
                    first_close_region_evaluation = close_region
                    if not bool(close_region["inside"]):
                        premature_close_count += 1
                generations_before = (
                    target_telemetry.begin_policy_step()
                    if target_telemetry is not None
                    else None
                )
                pending_bc_attempt = {
                    "step": bc_step,
                    "model_output_4d": list(action_values),
                    "full_8d_packet": list(packet.values),
                    "previous_policy_action": capture["previous_policy_action"],
                    "previous_gripper_intent": previous_intent.value,
                    "requested_gripper_intent": packet.gripper_intent.value,
                    "state_preserving_bridge": bool(state_preserving_bridge),
                    "bridge_force_open": bridge_force_open,
                    "bridge_steps_required": bridge_steps_required,
                    "bridge_steps_completed": bridge_steps_completed,
                    "executed_policy_action_4d": list(action.values),
                    "fixed_planner_endpoint_root_m": (
                        list(final_target) if state_preserving_bridge else None
                    ),
                    "decoded_bridge_endpoint_root_m": (
                        list(bridge_receipt.decoded_endpoint_root_m)
                        if bridge_receipt is not None
                        else None
                    ),
                    "bridge_endpoint_error_m": (
                        bridge_receipt.endpoint_error_m
                        if bridge_receipt is not None
                        else None
                    ),
                    "p_close": probability,
                    "gripper_logit": float(
                        math.log(clipped_probability / (1.0 - clipped_probability))
                    ),
                    "close_region_before_action": close_region,
                    "measured_ee_root_m": position_before,
                    "cube_root_m": cube_before,
                    "limiter_previous_post_target_q": (
                        _tensor(arm_term.last_emitted_joint_position_target)[0]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float64)
                    ),
                    "limiter_previous_accumulator_velocity_rad_s": (
                        _tensor(arm_term.current_target_velocity_rad_s)[0]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float64)
                    ),
                    "controller_endpoint_root_m_before_process_action": (
                        _tensor(arm_term.ee_desired_position)[0]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float64)
                    ),
                }
                if passive_contact_telemetry is not None:
                    passive_contact_telemetry.set_context(
                        policy_step=len(path_records) + bc_step,
                        bc_step=bc_step,
                        phase="MICRO_APPROACH",
                        gripper_intent=packet.gripper_intent.value,
                        close_onset=accepted_first_close,
                        clipping=False,
                    )
                try:
                    outputs, receipt = preflight._consume_once(
                        env=env,
                        counter=counter,
                        deferred_port=deferred_port,
                        packet=packet,
                        label=f"BC_MICRO_APPROACH_{bc_step:03d}",
                    )
                except Exception:
                    controller_reject_count += 1
                    raise
                limiter_step = (
                    target_telemetry.end_policy_step(generations_before)
                    if target_telemetry is not None
                    and generations_before is not None
                    else None
                )
                _, reward, terminated, truncated, _ = outputs
                latch.commit(packet.gripper_intent)
                bc_previous.record_accepted(action)
                if bridge_force_open:
                    bridge_steps_completed += 1
                audit.observe(robot.data.joint_pos, step=len(path_records) + len(bc_records) + 1)
                q_now = (
                    _tensor(robot.data.joint_pos)[0]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )
                q_samples.append(q_now)
                target_samples.append(
                    _tensor(arm_term.last_emitted_joint_position_target)[0]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )
                position_after = _ee_root_position(env)
                cube_after = _cube_root_position(env)
                task_metrics = preflight._task_metrics(env, g2_lift_task_mdp)
                bilateral_contact_observed |= bool(
                    task_metrics["ever_bilateral_contact"]
                )
                if (
                    first_bilateral_contact_step is None
                    and bool(task_metrics["bilateral_contact"])
                ):
                    first_bilateral_contact_step = bc_step
                stable_contact_observed |= bool(task_metrics["ever_stable_grasp"])
                stable_hold_steps = (
                    stable_hold_steps + 1 if bool(task_metrics["stable_now"]) else 0
                )
                active_termination_names = preflight._active_termination_names(
                    env, terminated, truncated
                )
                evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
                sensor_peaks = (
                    evaluator.sensor_peak_forces_n() if evaluator is not None else {}
                )
                if sensor_peaks:
                    maximum_forbidden_contact_force_n = max(
                        maximum_forbidden_contact_force_n,
                        max(max(values) for values in sensor_peaks.values()),
                    )
                record = {
                    "step": bc_step,
                    "episode_step": len(path_records) + bc_step,
                    "phase": "MICRO_APPROACH",
                    "observation": capture,
                    "model_output_4d": list(action_values),
                    "executed_policy_action_4d": list(action.values),
                    "bc_translation_advisory_only": (
                        list(action_values[:3]) if state_preserving_bridge else None
                    ),
                    "state_preserving_bridge": bool(state_preserving_bridge),
                    "bridge_force_open": bridge_force_open,
                    "bridge_steps_required": bridge_steps_required,
                    "bridge_steps_completed": bridge_steps_completed,
                    "bridge_endpoint_error_m": (
                        bridge_receipt.endpoint_error_m
                        if bridge_receipt is not None
                        else None
                    ),
                    "p_close": probability,
                    "gripper_logit": float(
                        math.log(clipped_probability / (1.0 - clipped_probability))
                    ),
                    "hidden_l2_norm": float(
                        torch.linalg.vector_norm(bc_hidden).item()
                    ),
                    "accepted_first_close": accepted_first_close,
                    "close_region_before_action": close_region,
                    "measured_before_root_m": position_before,
                    "measured_after_root_m": position_after,
                    "cube_root_m": cube_after,
                    "ee_cube_distance_m": float(
                        np.linalg.norm(cube_after - position_after)
                    ),
                    "grasp_target_distance_m": float(
                        np.linalg.norm(final_target - position_after)
                    ),
                    "full_8d_packet": list(packet.values),
                    "packet_derivation": derivation,
                    "single_consumption": receipt,
                    "gripper_intent": packet.gripper_intent.value,
                    "task": task_metrics,
                    "reward": float(reward.reshape(-1)[0].item()),
                    "terminated": bool(terminated.reshape(-1)[0].item()),
                    "truncated": bool(truncated.reshape(-1)[0].item()),
                    "active_termination_names": active_termination_names,
                    "forbidden_contact_sensor_peak_n": sensor_peaks,
                }
                bc_records.append(record)
                pending_bc_attempt = None
                progress["bc_closed_loop_steps"] = len(bc_records)
                all_single_consumption &= bool(receipt["single_consumption"])
                if limiter_step is not None:
                    policy_step = len(path_records) + len(bc_records)
                    for substep in limiter_step["substeps"]:
                        telemetry_substeps.append(
                            {
                                **substep,
                                "policy_step": policy_step,
                                "waypoint_index": -2,
                                "waypoint_repeat_index": bc_step + 1,
                                "phase": "MICRO_APPROACH",
                            }
                        )
                    last_substep = limiter_step["substeps"][-1]
                    ee_position, ee_quaternion = _ee_root_pose(env)
                    measured_q = (
                        _tensor(robot.data.joint_pos)[0, arm_term._joint_ids]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float64)
                    )
                    measured_qdot = (
                        _tensor(robot.data.joint_vel)[0, arm_term._joint_ids]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float64)
                    )
                    final_post_target = (
                        _tensor(arm_term.last_emitted_joint_position_target)[0]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(np.float64)
                    )
                    telemetry_samples.append(
                        {
                            "policy_step": policy_step,
                            "timestamp_s": policy_step * float(env.step_dt),
                            "waypoint_index": -2,
                            "waypoint_boundary": bc_step == 0,
                            "waypoint_repeat_index": bc_step + 1,
                            "waypoint_repeat_transition": bc_step > 0,
                            "phase": "MICRO_APPROACH",
                            "raw_planner_q": planned_q[-1],
                            "previous_raw_planner_q": planned_q[-1],
                            "next_raw_planner_q": planned_q[-1],
                            "pre_rate_limit_target_q": last_substep[
                                "pre_rate_limit_target"
                            ],
                            "post_rate_limit_target_q": final_post_target,
                            "limiter_accumulator_velocity": last_substep[
                                "final_accumulator_velocity"
                            ],
                            "limiter_reset": bool(
                                limiter_step["reset_during_policy_step"]
                            ),
                            "limiter_synchronize": bool(
                                limiter_step["synchronize_during_policy_step"]
                            ),
                            "controller_reject": False,
                            "clipping": False,
                            "measured_q": measured_q,
                            "measured_qdot": measured_qdot,
                            "ee_position_root_m": ee_position,
                            "ee_quaternion_root_wxyz": ee_quaternion,
                            "planner_ee_target_root_m": final_target,
                            "ee_tracking_error_m": float(
                                np.linalg.norm(final_target - ee_position)
                            ),
                            "maximum_forbidden_contact_force_n": float(
                                max(
                                    (
                                        max(values)
                                        for values in sensor_peaks.values()
                                    ),
                                    default=0.0,
                                )
                            ),
                        }
                    )
                _atomic_json(
                    partial_path,
                    {
                        "schema": "g2_curobo_bc_live_partial_telemetry_v1",
                        "path_record_count": len(path_records),
                        "bc_record_count": len(bc_records),
                        "path_record_ring": path_records[-4:],
                        "bc_record_ring": bc_records[-8:],
                        "progress": progress,
                        "latest_measured_joint_q": q_samples[-1],
                        "latest_emitted_target_q": target_samples[-1],
                        "source_freeze_manifest_sha256": freeze_before[
                            "manifest_sha256"
                        ],
                    },
                )
                if record["terminated"] or record["truncated"]:
                    terminated_early = True
                    progress["timeout_episode_step"] = int(
                        _tensor(env.episode_length_buf).reshape(-1)[0].item()
                    )
                    break
                if passive_contact_telemetry is not None:
                    progress["post_contact_observation_s"] = (
                        passive_contact_telemetry.post_first_contact_duration_s
                    )
                if (
                    stable_hold_steps >= 5
                    and (
                        passive_contact_telemetry is None
                        or passive_contact_telemetry.post_first_contact_duration_s
                        >= minimum_post_contact_observation_s
                    )
                ):
                    break
            print("BC_HANDOFF_END", flush=True)

        final_ee = _ee_root_position(env)
        final_cube = _cube_root_position(env)
        final_distance = float(np.linalg.norm(final_cube - final_ee))
        final_error = float(np.linalg.norm(final_target - final_ee))
        q_array = np.stack(q_samples, axis=0)
        target_array = np.stack(target_samples, axis=0)
        active_fd = _fd_metrics(q_array, float(env.step_dt), arm_indices)
        passive_fd = _fd_metrics(q_array, float(env.step_dt), passive_indices)
        command_fd = _fd_metrics(
            target_array,
            float(env.step_dt),
            np.arange(target_array.shape[1], dtype=np.int64),
        )
        planner_phases = [record["phase"] for record in path_records]
        bc_mask_phases = [
            "ZERO_ACTION_IDLE" if record["bridge_force_open"] else "MICRO_APPROACH"
            for record in bc_records
        ]
        phases = np.asarray(planner_phases + bc_mask_phases, dtype="U32")
        planner_zero_action_idle = [
            bool(
                np.linalg.norm(
                    np.asarray(record["metric_delta_root_m"], dtype=np.float64)
                )
                <= 1.0e-12
            )
            for record in path_records
        ]
        zero_action_idle = np.asarray(
            planner_zero_action_idle
            + [bool(record["bridge_force_open"]) for record in bc_records],
            dtype=np.bool_,
        )
        close_onset_mask = np.asarray(
            [
                False for _record in path_records
            ],
            dtype=np.bool_,
        )
        close_onset_mask = np.concatenate(
            (
                close_onset_mask,
                np.asarray(
                    [bool(record["accepted_first_close"]) for record in bc_records],
                    dtype=np.bool_,
                ),
            )
        )
        training_mask = training_valid_mask(
            phases,
            zero_action_idle=zero_action_idle,
            close_onset=close_onset_mask,
        )
        training_mask_receipt = training_mask_summary(
            training_mask, close_onset=close_onset_mask
        )
        training_mask_receipt["sequence_length_unchanged"] = 16
        training_mask_receipt["burn_in_steps_unchanged"] = 4
        training_mask_receipt["combined_planner_bc_row_count"] = (
            len(path_records) + len(bc_records)
        )
        training_mask_receipt["bridge_rows_masked"] = int(
            sum(bool(record["bridge_force_open"]) for record in bc_records)
        )
        training_mask_receipt["hard_negative_steps_masked"] = 0
        attribution_result: dict[str, Any] | None = None
        if attribution_output is not None:
            attribution_result = _save_target_attribution(
                output=attribution_output,
                samples=telemetry_samples,
                substep_rows=telemetry_substeps,
                joint_names=arm_joint_names,
                control_dt_s=float(env.step_dt),
                physics_dt_s=float(env.physics_dt),
                hard_acceleration_limit_rad_s2=10.0,
            )
            attribution_json = attribution_output.with_name(
                "TARGET_ACCEL_EVENT_ATTRIBUTION.json"
            )
            event = attribution_result["event"]
            summary = attribution_result["summary"]
            root_cause = str(event["root_cause"])
            eligible_for_review = bool(
                root_cause == "ONE_STEP_WAYPOINT_BOUNDARY_SPIKE"
                and event["duration_above_10_steps"] == 1
                and abs(event["measured_fd_qdd_same_step_rad_s2"]) <= 10.0
                and not event["collision_or_controller_reject"]
                and event["multi_joint_above_limit_count_same_sample"] == 1
                and event["above_limit_policy_event_count"] == 1
                and summary["clipping_count"] == 0
                and summary["post_limiter_joint_clamp_substep_count"] == 0
            )
            limiter_delta_pass = bool(
                root_cause == "NO_ABOVE_LIMIT_EVENT"
                and summary["maximum_target_fd_velocity_rad_s"]
                <= planner_active_velocity_limit_rad_s
                and summary["maximum_target_fd_acceleration_rad_s2"] <= 10.0
                and summary["maximum_accumulator_qdd_rad_s2"] <= 2.00001
                and summary["clipping_count"] == 0
                and summary["post_limiter_joint_clamp_substep_count"] == 0
                and summary["controller_reject_count"] == 0
                and summary["maximum_forbidden_contact_force_n"] <= 1.0e-6
            )
            attribution_result["gate_semantics_recommendation"] = {
                "M1_INHERITANCE": (
                    "PASS_WITH_LIMITER_SEMANTICS_DELTA"
                    if limiter_delta_pass
                    else (
                        "ELIGIBLE_FOR_GATE_SEMANTICS_REVIEW"
                        if eligible_for_review
                        else "KEEP_REOPEN_REQUIRED"
                    )
                ),
                "M2_INHERITANCE": "REOPEN_REQUIRED",
                "BC": "BLOCKED",
                "SAC": "BLOCKED",
                "gate_changed_by_probe": False,
            }
            _atomic_json(attribution_json, attribution_result)
        full_audit = audit.result()
        nonzero_alignments = [
            float(record["action_goal_alignment_cosine"])
            for record in path_records
            if record["action_goal_alignment_cosine"] is not None
        ]
        freeze_after = _source_freeze(
            keyboard_v3_branch=(
                keyboard_v3_collection_root is not None
                or candidate_a_bc_validation_output is not None
                or runtime_replan_canonical_contact_free_collection_output is not None
            )
        )
        checks = {
            "source_freeze_stable": freeze_before == freeze_after
            and freeze_after["SOURCE_FREEZE"] == "PASS",
            "selected_asset_exact": bool(
                diagnostic_asset_selection is not None
                and diagnostic_asset_selection["sha256"]
                == (
                    freeze_before["production_usd_sha256"]
                    if diagnostic_asset_variant == "production"
                    else diagnostic_asset_sha256
                    if diagnostic_asset_variant == "custom"
                    else freeze_before["m2_candidate_usd_sha256"]
                )
            ),
            "planner_waypoints_consumed": len(
                {
                    int(record["waypoint_index"])
                    for record in path_records
                    if record["waypoint_index"] is not None
                }
            )
            == len(selected_waypoint_indices),
            "adaptive_skips_certified": (
                not adaptive_waypoints
                or (
                    bool(adaptive_schedule["fine_density_preserved"])
                    and bool(adaptive_schedule["target_envelope_preserved"])
                    and collision_receipt is not None
                )
            ),
            "single_consumption_all_packets": all_single_consumption,
            "router_direct_process_action_count_zero": True,
            "controller_reject_count_zero": controller_reject_count == 0,
            "no_clipping": all(not bool(record["clipping_applied"]) for record in path_records),
            "gripper_open_fixed": all(record["gripper_intent"] == "OPEN" for record in path_records),
            "no_unexpected_termination": not terminated_early,
            "no_forbidden_collision": maximum_forbidden_contact_force_n <= 1.0e-6,
            "final_pregrasp_within_existing_planning_tolerance": (
                final_error if handoff_pregrasp_error_m is None else handoff_pregrasp_error_m
            )
            <= PREGRASP_TOLERANCE_M,
            "active_velocity_within_selected_planner_delta_limit": active_fd[
                "maximum_fd_velocity_rad_s"
            ]
            <= MEASURED_ACTIVE_VELOCITY_HARD_LIMIT_RAD_S,
            "active_acceleration_within_existing_hard_limit": active_fd[
                "maximum_fd_acceleration_rad_s2"
            ]
            <= 10.0,
            "passive_velocity_within_existing_hard_limit": passive_fd[
                "maximum_fd_velocity_rad_s"
            ]
            <= 0.8,
            "passive_acceleration_within_existing_hard_limit": passive_fd[
                "maximum_fd_acceleration_rad_s2"
            ]
            <= 10.0,
            "full_articulation_fd_audit_pass": bool(full_audit["pass"]),
            "telemetry_flush_completed": int(progress["telemetry_flush_steps"])
            == telemetry_flush_steps,
            "all_metrics_finite": all(
                math.isfinite(value)
                for value in (
                    initial_distance,
                    final_distance,
                    final_error,
                    maximum_forbidden_contact_force_n,
                    active_fd["maximum_fd_velocity_rad_s"],
                    active_fd["maximum_fd_acceleration_rad_s2"],
                    passive_fd["maximum_fd_velocity_rad_s"],
                    passive_fd["maximum_fd_acceleration_rad_s2"],
                )
            ),
        }
        if attribution_result is not None:
            checks.update(
                {
                    "target_attribution_policy_samples_aligned": len(
                        telemetry_samples
                    )
                    == target_array.shape[0],
                    "target_attribution_peak_matches_functional_summary": math.isclose(
                        attribution_result["summary"][
                            "maximum_target_fd_acceleration_rad_s2"
                        ],
                        command_fd["maximum_fd_acceleration_rad_s2"],
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    ),
                    "target_attribution_has_physics_substeps": len(
                        telemetry_substeps
                    )
                    >= len(path_records) + len(bc_records),
                    "target_attribution_probe_pass": attribution_result["verdict"][
                        "TARGET_ACCEL_ATTRIBUTION_PROBE"
                    ]
                    == "PASS",
                    "target_velocity_within_existing_hard_limit": (
                        attribution_result["summary"][
                            "maximum_target_fd_velocity_rad_s"
                        ]
                        <= planner_active_velocity_limit_rad_s
                    ),
                    "target_acceleration_within_existing_hard_limit": (
                        attribution_result["summary"][
                            "maximum_target_fd_acceleration_rad_s2"
                        ]
                        <= 10.0
                    ),
                    "limiter_accumulator_acceleration_within_configured_limit": (
                        attribution_result["summary"][
                            "maximum_accumulator_qdd_rad_s2"
                        ]
                        <= 2.00001
                    ),
                    "target_above_limit_event_count_zero": (
                        attribution_result["event"][
                            "above_limit_policy_event_count"
                        ]
                        == 0
                    ),
                }
            )
        planner_functional = all(checks.values())
        stage1_checks: dict[str, bool] = {}
        stage1_failure_classification = None
        if bc_enabled:
            stage1_checks = {
                "safe_handoff_reached": bool(
                    handoff_region_evaluation is not None
                    and handoff_region_evaluation["inside"]
                ),
                "state_preserving_bridge_completed": bool(
                    not state_preserving_bridge
                    or bridge_steps_completed >= bridge_steps_required
                ),
                "bc_history_rolled_from_episode_start": len(bc_history_records)
                == len(path_records),
                "close_observed": first_close_step is not None,
                "first_close_inside_demo_region": bool(
                    first_close_region_evaluation is not None
                    and first_close_region_evaluation["inside"]
                ),
                "premature_close_count_zero": premature_close_count == 0,
                "bilateral_contact_observed": bilateral_contact_observed,
                "bilateral_contact_after_first_close": bool(
                    first_close_step is not None
                    and first_bilateral_contact_step is not None
                    and first_bilateral_contact_step >= first_close_step
                ),
                "minimum_post_contact_observation_completed": bool(
                    passive_contact_telemetry is None
                    or passive_contact_telemetry.post_first_contact_duration_s
                    >= minimum_post_contact_observation_s
                ),
                "controller_reject_count_zero": controller_reject_count == 0,
                "single_consumption_all_packets": all_single_consumption,
                "forbidden_collision_count_zero": maximum_forbidden_contact_force_n
                <= 1.0e-6,
                "active_velocity_within_hard_limit": active_fd[
                    "maximum_fd_velocity_rad_s"
                ]
                <= MEASURED_ACTIVE_VELOCITY_HARD_LIMIT_RAD_S,
                "active_acceleration_within_hard_limit": active_fd[
                    "maximum_fd_acceleration_rad_s2"
                ]
                <= 10.0,
                "passive_velocity_within_hard_limit": passive_fd[
                    "maximum_fd_velocity_rad_s"
                ]
                <= 0.8,
                "passive_acceleration_within_hard_limit": passive_fd[
                    "maximum_fd_acceleration_rad_s2"
                ]
                <= 10.0,
            }
            if not all(
                stage1_checks[name]
                for name in (
                    "controller_reject_count_zero",
                    "single_consumption_all_packets",
                    "forbidden_collision_count_zero",
                    "active_velocity_within_hard_limit",
                    "active_acceleration_within_hard_limit",
                    "passive_velocity_within_hard_limit",
                    "passive_acceleration_within_hard_limit",
                )
            ):
                stage1_failure_classification = "SAFETY_FAILURE"
            elif first_close_step is None:
                stage1_failure_classification = "BC_CLOSE_TIMING_FAILURE"
            elif not stage1_checks["first_close_inside_demo_region"]:
                stage1_failure_classification = "BC_CLOSE_TIMING_FAILURE"
            elif not bilateral_contact_observed:
                stage1_failure_classification = "CONTACT_ACQUISITION_FAILURE"
            elif not stage1_checks["bc_history_rolled_from_episode_start"]:
                stage1_failure_classification = "PLANNER_TO_BC_HANDOFF_FAILURE"
            elif not stage1_checks["safe_handoff_reached"]:
                stage1_failure_classification = "PLANNER_TO_BC_HANDOFF_FAILURE"
            elif not stage1_checks["state_preserving_bridge_completed"]:
                stage1_failure_classification = "PLANNER_TO_BC_HANDOFF_FAILURE"
        functional = bool(
            planner_functional
            and (not bc_enabled or all(stage1_checks.values()))
        )
        actual_tracking_steps = int(sum(waypoint_tracking_steps))
        bounded_tracking_steps = (
            (len(selected_waypoint_indices) - 1) * int(ordinary_tracking_cap)
            + int(critical_tracking_cap)
        )
        historical_repeat_overhead = 555 - int(planned_ee.shape[0])
        original_density_capped_overhead = (
            (int(planned_ee.shape[0]) - 1) * (int(ordinary_tracking_cap) - 1)
            + (int(critical_tracking_cap) - 1)
        )
        budget_savings = {
            "scope": "SEPARATE_COUNTERFACTUAL_COMPONENTS_NOT_SUMMABLE",
            "waypoint_skipping_mandatory_submission_steps": int(
                adaptive_schedule["skipped_waypoint_count"]
            ),
            "repeat_cap_overhead_steps_vs_historical_at_original_density": max(
                0, historical_repeat_overhead - original_density_capped_overhead
            ),
            "early_advance_steps_within_selected_bounded_schedule": max(
                0, bounded_tracking_steps - actual_tracking_steps
            ),
            "configured_dwell_reduction_steps": max(
                0, MAXIMUM_SETTLE_STEPS - int(final_settle_cap)
            ),
            "historical_tracking_steps": 555,
            "actual_tracking_steps": actual_tracking_steps,
        }
        _atomic_json(
            partial_path,
            {
                "schema": (
                    "g2_curobo_bc_live_partial_telemetry_v1"
                    if bc_enabled
                    else "g2_curobo_live_partial_telemetry_v1"
                ),
                "path_record_count": len(path_records),
                "bc_record_count": len(bc_records),
                "latest_record": path_records[-1],
                "latest_bc_record": bc_records[-1] if bc_records else None,
                "progress": progress,
                "final_flush_complete": True,
                "source_freeze_manifest_sha256": freeze_before["manifest_sha256"],
            },
        )
        if passive_contact_telemetry is not None:
            assert passive_contact_telemetry_output is not None
            passive_contact_receipt = passive_contact_telemetry.save(
                passive_contact_telemetry_output
            )

        report = {
            "schema": (
                "g2_curobo_bc_gru_closed_loop_v1"
                if bc_enabled
                else "g2_curobo_planner_only_live_smoke_v2"
            ),
            "scope": (
                "CUROBO_RESET_TO_SAFE_HANDOFF_THEN_BC_GRU_MICRO_APPROACH_CLOSE_CONTACT_NO_SAC"
                if bc_enabled
                else "ONE_BOUNDED_PLANNER_ONLY_RESET_TO_PREGRASP_NO_BC_NO_SAC_GRIPPER_OPEN"
            ),
            "source_freeze_before": freeze_before,
            "source_freeze_after": freeze_after,
            "asset_binding": asset_receipt.as_dict(),
            "diagnostic_asset_selection": diagnostic_asset_selection,
            "physics_delta_assertion": physics_delta_assertion,
            "runtime_replan_input_telemetry": runtime_replan_input_telemetry,
            "runtime_replan_input_telemetry_path": (
                str(runtime_replan_input_telemetry_path)
                if runtime_replan_input_telemetry_path is not None
                else None
            ),
            "settled_initial_state": settled_initial_state,
            "task_contract": task_contract,
            "task_geometry_binding": task_geometry_binding,
            "task_geometry_readback": task_geometry_readback,
            "p0a_provenance": p0a_provenance,
            "trajectory_replay": str(replay.resolve()),
            "trajectory_replay_sha256": _sha256(replay),
            "prerequisite_authority": (
                "USER_ASSUMED_PREREQUISITE_PASS"
                if bc_enabled
                else "EXISTING_PLANNER_ONLY_AUTHORITY"
            ),
            "adaptive_waypoint_execution": {
                **adaptive_schedule,
                "enabled": bool(adaptive_waypoints),
                "collision_receipt_path": (
                    str(adaptive_collision_report.resolve())
                    if adaptive_collision_report is not None
                    else None
                ),
                "collision_receipt_sha256": (
                    _sha256(adaptive_collision_report)
                    if adaptive_collision_report is not None
                    else None
                ),
                "collision_clearance": {
                    "source_waypoint_collision_cost_max": (
                        float(collision_receipt["motiongen"]["maximum_table_collision_cost"])
                        if collision_receipt is not None
                        else None
                    ),
                    "source_self_collision_cost_max": (
                        float(collision_receipt["motiongen"]["maximum_self_collision_cost"])
                        if collision_receipt is not None
                        else None
                    ),
                    "metric_clearance_m": None,
                    "interpretation": (
                        "SOURCE_CUROBO_COLLISION_FREE_RECEIPT; METRIC_CLEARANCE_NOT_EXPOSED"
                    ),
                },
            },
            "phase_progress_contract": PHASE_PROGRESS_CONTRACT,
            "progress_metric_config": asdict(progress_metric_config),
            "budget_savings": budget_savings,
            "open_settle": settle,
            "timing_contract": {
                "optimized": bool(timing_optimized),
                "episode_horizon_steps": int(episode_horizon_steps),
                "environment_max_episode_length": int(env.max_episode_length),
                "ordinary_tracking_cap": int(ordinary_tracking_cap),
                "critical_tracking_cap": int(critical_tracking_cap),
                "critical_waypoint_index": int(critical_waypoint_index),
                "final_convergence_cap": int(final_convergence_cap),
                "final_settle_cap": int(final_settle_cap),
                "telemetry_flush_steps": int(telemetry_flush_steps),
                "completion_episode_step": int(
                    _tensor(env.episode_length_buf).reshape(-1)[0].item()
                ),
                "remaining_horizon_steps": int(env.max_episode_length)
                - int(_tensor(env.episode_length_buf).reshape(-1)[0].item()),
                "total_physics_substeps": int(
                    _tensor(env.episode_length_buf).reshape(-1)[0].item()
                )
                * int(cfg.decimation),
            },
            "training_valid_mask": training_mask.astype(bool).tolist(),
            "training_mask_receipt": training_mask_receipt,
            "initial_ee_root_m": initial_ee.astype(float).tolist(),
            "initial_cube_root_m": initial_cube.astype(float).tolist(),
            "initial_ee_cube_distance_m": initial_distance,
            "final_target_root_m": final_target.astype(float).tolist(),
            "final_ee_root_m": final_ee.astype(float).tolist(),
            "final_cube_root_m": final_cube.astype(float).tolist(),
            "final_ee_cube_distance_m": final_distance,
            "minimum_ee_cube_distance_m": min(
                float(record["ee_cube_distance_m"]) for record in path_records
            ),
            "final_pregrasp_error_m": (
                final_error
                if handoff_pregrasp_error_m is None
                else handoff_pregrasp_error_m
            ),
            "handoff_pregrasp_error_m": handoff_pregrasp_error_m,
            "final_post_bc_grasp_target_error_m": (
                final_error if bc_enabled else None
            ),
            "existing_pregrasp_tolerance_m": PREGRASP_TOLERANCE_M,
            "planner_progress_waypoints": len(selected_waypoint_indices),
            "planner_original_waypoints": int(planned_ee.shape[0]),
            "planner_effective_waypoints": len(selected_waypoint_indices),
            "planner_skipped_waypoints": int(adaptive_schedule["skipped_waypoint_count"]),
            "waypoint_tracking_steps": waypoint_tracking_steps,
            "maximum_tracking_steps_for_one_waypoint": max(waypoint_tracking_steps),
            "total_waypoint_tracking_steps": sum(waypoint_tracking_steps),
            "coarse_planner_tracking_steps": coarse_tracking_steps,
            "fine_planner_tracking_steps": fine_tracking_steps,
            "effective_repeat_count": actual_tracking_steps,
            "average_repeat_per_effective_waypoint": (
                actual_tracking_steps / len(selected_waypoint_indices)
            ),
            "early_advance_count": early_advance_count,
            "critical_handoff_repeat_count": critical_handoff_repeat_count,
            "tracking_budget_per_waypoint": int(ordinary_tracking_cap),
            "critical_tracking_budget": int(critical_tracking_cap),
            "planner_final_convergence_steps": int(final_convergence_steps),
            "planner_settle_steps": settle_steps,
            "telemetry_flush_steps": int(progress["telemetry_flush_steps"]),
            "path_tracking_error_max_m": max(
                float(record["tracking_error_after_m"]) for record in path_records
            ),
            "path_tracking_error_mean_m": float(
                np.mean([record["tracking_error_after_m"] for record in path_records])
            ),
            "action_goal_alignment_cosine_min": min(nonzero_alignments),
            "action_goal_alignment_cosine_mean": float(np.mean(nonzero_alignments)),
            "controller_reject_count": controller_reject_count,
            "maximum_forbidden_contact_force_n": maximum_forbidden_contact_force_n,
            "first_collision_link": None,
            "active_fd": active_fd,
            "passive_fd": passive_fd,
            "command_target_fd": command_fd,
            "target_acceleration_attribution": attribution_result,
            "passive_contact_physics_telemetry": passive_contact_receipt,
            "full_articulation_fd_audit": full_audit,
            "lifecycle_counts": counter.summary(),
            "checks": checks,
            "path_records": path_records,
            "bc_handoff": {
                "enabled": bc_enabled,
                "checkpoint": bc_checkpoint_receipt,
                "geometry_path": (
                    str(grasp_ready_geometry.resolve())
                    if grasp_ready_geometry is not None
                    else None
                ),
                "geometry_sha256": (
                    _sha256(grasp_ready_geometry)
                    if grasp_ready_geometry is not None
                    else None
                ),
                "controller_binding": bc_controller_binding,
                "observation_binding": (
                    bc_observation_binding.payload()
                    if bc_observation_binding is not None
                    else None
                ),
                "observation_capture_count": (
                    bc_observation_source.captures
                    if bc_observation_source is not None
                    else 0
                ),
                "planner_history_rollin_steps": len(bc_history_records),
                "handoff_region_evaluation": handoff_region_evaluation,
                "requested_closed_loop_steps": int(bc_closed_loop_steps),
                "completed_closed_loop_steps": len(bc_records),
                "state_preserving_bridge_enabled": bool(state_preserving_bridge),
                "bridge_steps_required": bridge_steps_required,
                "bridge_steps_completed": bridge_steps_completed,
                "endpoint_jump_mm": (
                    1000.0
                    * float(
                        np.linalg.norm(
                            np.asarray(
                                bc_records[0]["measured_before_root_m"],
                                dtype=np.float64,
                            )
                            + np.asarray(
                                bc_records[0]["executed_policy_action_4d"][:3],
                                dtype=np.float64,
                            )
                            * 0.0225
                            - final_target
                        )
                    )
                    if bc_records
                    else None
                ),
                "first_post_limiter_q_target_jump_l2_rad": (
                    float(
                        np.linalg.norm(
                            target_array[len(path_records) + 1]
                            - target_array[len(path_records)]
                        )
                    )
                    if bc_records
                    else None
                ),
                "first_post_limiter_q_target_jump_max_rad": (
                    float(
                        np.max(
                            np.abs(
                                target_array[len(path_records) + 1]
                                - target_array[len(path_records)]
                            )
                        )
                    )
                    if bc_records
                    else None
                ),
                "controller_reject_count": controller_reject_count,
                "first_close_step": first_close_step,
                "first_close_region_evaluation": first_close_region_evaluation,
                "first_close_distance_cm": (
                    100.0 * float(first_close_region_evaluation["distance_m"])
                    if first_close_region_evaluation is not None
                    else None
                ),
                "premature_close_count": premature_close_count,
                "bilateral_contact_observed": bilateral_contact_observed,
                "first_bilateral_contact_step": first_bilateral_contact_step,
                "stable_contact_observed": stable_contact_observed,
                "stable_hold_steps": stable_hold_steps,
                "p_close_min": (
                    min(record["p_close"] for record in bc_records)
                    if bc_records
                    else None
                ),
                "p_close_max": (
                    max(record["p_close"] for record in bc_records)
                    if bc_records
                    else None
                ),
                "minimum_ee_cube_distance_cm": (
                    100.0
                    * min(record["ee_cube_distance_m"] for record in bc_records)
                    if bc_records
                    else None
                ),
                "stage1_checks": stage1_checks,
                "failure_classification": stage1_failure_classification,
                "records": bc_records,
            },
            "verdict": {
                "PLANNER_ONLY_LIVE_SMOKE": (
                    "PASS" if planner_functional else "FAIL"
                ),
                "CUROBO_BC_GRU_CLOSED_LOOP": (
                    "PASS"
                    if bc_enabled and functional
                    else "FAIL"
                    if bc_enabled
                    else "NOT_REQUESTED"
                ),
                "FUNCTIONAL_VERDICT": "PASS" if functional else "FAIL",
                "PROCESS_VERDICT": "PENDING_PARENT",
                "GRIPPER": "BC_GRU_AUTHORITY_AFTER_HANDOFF" if bc_enabled else "OPEN_FIXED",
                "TARGET_ACCEL_ATTRIBUTION_PROBE": (
                    "PASS" if attribution_result is not None else "NOT_REQUESTED"
                ),
                "M1_INHERITANCE": (
                    attribution_result["gate_semantics_recommendation"][
                        "M1_INHERITANCE"
                    ]
                    if attribution_result is not None
                    else "UNCHANGED"
                ),
                "M2_INHERITANCE": "REOPEN_REQUIRED",
                "BC": (
                    "PASS" if bc_enabled and functional else "FAIL"
                    if bc_enabled
                    else "BLOCKED" if attribution_result is not None else "OFF"
                ),
                "SAC": "BLOCKED" if attribution_result is not None else "OFF",
            },
            "cleanup": cleanup,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output, report)
        print("RUNTIME_END", flush=True)
        print("REPORT_SAVED", flush=True)
        return 0 if functional else 2
    except BaseException as error:
        if (
            passive_contact_telemetry is not None
            and passive_contact_telemetry_output is not None
            and passive_contact_receipt is None
        ):
            try:
                passive_contact_receipt = passive_contact_telemetry.save(
                    passive_contact_telemetry_output
                )
            except Exception as telemetry_error:
                passive_contact_receipt = {
                    "path": str(passive_contact_telemetry_output),
                    "save_error": (
                        f"{type(telemetry_error).__name__}:{telemetry_error}"
                    ),
                }
        if env is not None:
            try:
                progress["timeout_episode_step"] = int(
                    _tensor(env.episode_length_buf).reshape(-1)[0].item()
                )
            except Exception:
                pass
        report = {
            "schema": (
                "g2_curobo_bc_gru_closed_loop_v1"
                if bc_checkpoint is not None
                else "g2_curobo_planner_only_live_smoke_v2"
            ),
            "scope": (
                "CUROBO_RESET_TO_SAFE_HANDOFF_THEN_BC_GRU_MICRO_APPROACH_CLOSE_CONTACT_NO_SAC"
                if bc_checkpoint is not None
                else "ONE_BOUNDED_PLANNER_ONLY_RESET_TO_PREGRASP_NO_BC_NO_SAC_GRIPPER_OPEN"
            ),
            "source_freeze_before": freeze_before,
            "diagnostic_asset_selection": diagnostic_asset_selection,
            "physics_delta_assertion": physics_delta_assertion,
            "runtime_replan_input_telemetry": runtime_replan_input_telemetry,
            "runtime_replan_input_telemetry_path": (
                str(runtime_replan_input_telemetry_path)
                if runtime_replan_input_telemetry_path is not None
                else None
            ),
            "settled_initial_state": settled_initial_state,
            "error": f"{type(error).__name__}:{error}",
            "traceback": traceback.format_exc(),
            "progress": progress,
            "path_record_count": len(path_records),
            "path_record_ring": path_records[-16:],
            "bc_record_count": len(bc_records),
            "bc_record_ring": bc_records[-16:],
            "bc_history_record_ring": bc_history_records[-16:],
            "pending_bc_attempt": pending_bc_attempt,
            "limiter_failure_snapshot": (
                target_telemetry.failure_snapshot()
                if target_telemetry is not None
                else []
            ),
            "limiter_policy_sample_ring": telemetry_samples[-24:],
            "limiter_substep_ring": telemetry_substeps[-32:],
            "durable_partial_telemetry": str(partial_path),
            "passive_contact_physics_telemetry": passive_contact_receipt,
            "verdict": {
                "PLANNER_ONLY_LIVE_SMOKE": "FAIL",
                "CUROBO_BC_GRU_CLOSED_LOOP": (
                    "FAIL" if bc_checkpoint is not None else "NOT_REQUESTED"
                ),
                "FUNCTIONAL_VERDICT": "FAIL",
                "PROCESS_VERDICT": "PENDING_PARENT",
            },
            "cleanup": cleanup,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output, report)
        print("RUNTIME_END", flush=True)
        print("REPORT_SAVED", flush=True)
        return 2
    finally:
        if passive_contact_telemetry is not None:
            try:
                passive_contact_telemetry.restore()
            except Exception:
                pass
        if target_telemetry is not None:
            try:
                target_telemetry.restore()
            except Exception:
                pass
        if counter is not None:
            try:
                counter.restore()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
                cleanup["env_close"] = "PASS"
            except Exception as error:
                cleanup["env_close"] = f"FAIL:{type(error).__name__}:{error}"
        print("APP_CLOSE_BEGIN", flush=True)
        try:
            app.close()
            print("APP_CLOSED", flush=True)
        except Exception as error:
            print(f"APP_CLOSE_ERROR:{type(error).__name__}:{error}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-replay", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attribution-output", type=Path)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--timing-optimized", action="store_true")
    parser.add_argument("--episode-horizon-steps", type=int, default=640)
    parser.add_argument(
        "--ordinary-tracking-cap", type=int, default=MAXIMUM_TRACKING_STEPS_PER_WAYPOINT
    )
    parser.add_argument(
        "--critical-tracking-cap", type=int, default=MAXIMUM_TRACKING_STEPS_PER_WAYPOINT
    )
    parser.add_argument("--final-convergence-cap", type=int, default=0)
    parser.add_argument("--final-settle-cap", type=int, default=MAXIMUM_SETTLE_STEPS)
    parser.add_argument(
        "--planner-active-velocity-limit-rad-s", type=float, default=0.8
    )
    parser.add_argument("--adaptive-waypoints", action="store_true")
    parser.add_argument("--adaptive-collision-report", type=Path)
    parser.add_argument("--telemetry-flush-steps", type=int, default=0)
    parser.add_argument("--progress-deadband-m", type=float, default=0.00025)
    parser.add_argument("--stall-grace-steps", type=int, default=8)
    parser.add_argument("--max-stall-steps", type=int, default=40)
    parser.add_argument("--retreat-threshold-m", type=float, default=0.001)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--bc-checkpoint-sha256")
    parser.add_argument("--contact-free-bc-playback-checkpoint", type=Path)
    parser.add_argument("--contact-free-bc-playback-checkpoint-sha256")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="render the bounded contact-free playback in the Isaac Sim GUI",
    )
    parser.add_argument("--keyboard-v3-collection-root", type=Path)
    parser.add_argument("--keyboard-v3-episode-id")
    parser.add_argument(
        "--keyboard-v3-backoff-m",
        type=float,
        choices=(
            0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022,
            0.025, 0.030, 0.035,
        ),
        default=0.030,
    )
    parser.add_argument("--keyboard-v3-maximum-steps", type=int, default=1600)
    parser.add_argument(
        "--keyboard-v3-translation-step-m",
        type=float,
        default=0.001,
        help="terminal XYZ pulse in metres; default 1 mm, hard maximum 4.5 mm",
    )
    parser.add_argument(
        "--keyboard-v3-motion-min-interval-s",
        type=float,
        default=0.050,
        help="minimum accepted interval between terminal XYZ pulses",
    )
    parser.add_argument(
        "--keyboard-v3-terminal-smoothing-steps",
        type=int,
        choices=range(1, 11),
        default=1,
        help="raised-cosine steps per accepted terminal XYZ pulse",
    )
    parser.add_argument(
        "--keyboard-v3-continuous-session",
        action="store_true",
        help="save, reset to the qualified handoff, and collect the next episode",
    )
    parser.add_argument(
        "--keyboard-v3-session-max-episodes",
        type=int,
        default=0,
        help="0 keeps collecting until operator quit; otherwise bounded episode count",
    )
    parser.add_argument(
        "--keyboard-v3-direct-pregrasp-v2",
        action="store_true",
        help="load the immutable Candidate-A V2 pregrasp robot/cube pair; do not plan",
    )
    parser.add_argument(
        "--keyboard-v3-v2-smoke-only",
        action="store_true",
        help="bounded contact-free direct-init/camera/+0.45-mm smoke; no operator episode",
    )
    parser.add_argument("--bc-closed-loop-steps", type=int, default=0)
    parser.add_argument("--grasp-ready-geometry", type=Path)
    parser.add_argument("--state-preserving-bridge", action="store_true")
    parser.add_argument("--passive-contact-telemetry-output", type=Path)
    parser.add_argument(
        "--diagnostic-asset-variant",
        choices=AB_ASSET_VARIANTS,
        default="candidate",
    )
    parser.add_argument("--diagnostic-asset-path", type=Path)
    parser.add_argument("--diagnostic-asset-sha256")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument("--stop-on-hard-stop", action="store_true")
    parser.add_argument("--candidate-a-contact-last-ab", action="store_true")
    parser.add_argument("--candidate-a-contact-free", action="store_true")
    parser.add_argument(
        "--candidate-a-contact-free-runtime-replan",
        action="store_true",
        help=(
            "separately reviewed contact-free path: plan only after reset; "
            "never consumes a frozen replay"
        ),
    )
    parser.add_argument("--canonical-contact-free-collection-output", type=Path)
    parser.add_argument(
        "--runtime-replan-canonical-contact-free-collection-output", type=Path,
        help=(
            "one independently seeded post-reset runtime-plan collection; "
            "writes only no-GT actor rows and a separate planner sidecar"
        ),
    )
    parser.add_argument(
        "--canonical-capture-rate-hz",
        type=int,
        choices=(25,),
        default=50,
        help="dataset capture rate while control remains fixed at 50 Hz",
    )
    parser.add_argument("--independent-collection-episode-id")
    parser.add_argument(
        "--candidate-a-bc-validation-output",
        type=Path,
        help="new P0.6 validation-only 50-Hz rows with true 25-Hz camera receipts",
    )
    parser.add_argument("--candidate-a-bc-validation-episode-id")
    parser.add_argument("--cosine-p05", type=float)
    parser.add_argument("--cosine-p10", type=float)
    parser.add_argument(
        "--minimum-post-contact-observation-ms", type=float, default=0.0
    )
    parser.add_argument(
        "--stage1a-isaac-short-smoke",
        action="store_true",
        help="exactly 1000 accepted Candidate-A V2 transitions; no long training",
    )
    parser.add_argument(
        "--stage1a-training-15k",
        action="store_true",
        help=(
            "exactly 15000 accepted Candidate-A hybrid transitions using one "
            "explicit SAC/HER/HER_FORCE replay strategy"
        ),
    )
    parser.add_argument(
        "--stage1a-reward-v3-smoke",
        action="store_true",
        help=(
            "exactly 3000 accepted Candidate-A hybrid transitions with opt-in "
            "temporal Reward V3 and SAC-only replay"
        ),
    )
    parser.add_argument(
        "--stage1a-stable-only-reward-v3",
        action="store_true",
        help=(
            "opt-in 15K Stable-Bilateral-only Reward-V3 contract; disables "
            "Lift/Place and requires the deterministic canonical CLOSE gate"
        ),
    )
    parser.add_argument(
        "--stage1a-replay-strategy",
        choices=STAGE1A_REPLAY_STRATEGIES,
        default="HER_FORCE",
    )
    parser.add_argument("--stage1a-training-seed", type=int, default=42)
    parser.add_argument(
        "--stage1a-hybrid-activation-smoke",
        action="store_true",
        help=(
            "bounded inference-only canonical router/actor/controller activation proof; "
            "no replay learning or optimizer update"
        ),
    )
    parser.add_argument(
        "--stage1a-hybrid-activation-max-steps", type=int, default=256
    )
    parser.add_argument("--stage1a-playback-actor-checkpoint", type=Path)
    parser.add_argument("--stage1a-playback-actor-checkpoint-sha256")
    parser.add_argument(
        "--stage1a-stable-playback",
        action="store_true",
        help=(
            "evaluation-only deterministic CLOSE and Stable-Bilateral-10-step "
            "playback; no replay or optimizer mutation"
        ),
    )
    parser.add_argument("--stage1a-playback-video-path", type=Path)
    parser.add_argument(
        "--stage1a-close-admission-diagnostic",
        action="store_true",
        help=(
            "one bounded 80-action frozen-GRU CLOSE diagnostic; residual SAC "
            "is disabled and no replay/optimizer update is permitted"
        ),
    )
    parser.add_argument(
        "--stage1a-close-calibration-sample",
        type=Path,
        help=(
            "immutable Candidate-A contact-free pregrasp sample receipt for "
            "bounded CLOSE outcome collection"
        ),
    )
    parser.add_argument(
        "--stage1a-close-residual-sweep-mm",
        type=int,
        choices=(15, 16, 17, 18, 19, 20, 21, 22, 23),
        help=(
            "deterministic one-CLOSE diagnostic at a cuRobo nominal-grasp "
            "translation residual; frozen GRU probabilities are logging-only"
        ),
    )
    parser.add_argument(
        "--stage1a-close-persistent-plan",
        type=Path,
        help=(
            "single-worker multi-episode CLOSE supervision plan; keeps one "
            "Isaac/App/asset session and resets only episode state"
        ),
    )
    parser.add_argument("--stage1a-close-lateral-offset-mm", type=float, default=0.0)
    parser.add_argument("--stage1a-close-height-offset-mm", type=float, default=0.0)
    parser.add_argument("--stage1a-close-approach-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--stage1a-extended-hard-stop-telemetry",
        action="store_true",
        help=(
            "qualification-only: latch the unchanged hard-stop authority "
            "post-step and finish the bounded 1500-ms raw observation window"
        ),
    )
    parser.add_argument(
        "--stage1a-geometry-forced-close-diagnostic",
        action="store_true",
        help=(
            "diagnostic-only privileged primary-pad/cube geometry gate before "
            "the existing deterministic one-shot abstract CLOSE path"
        ),
    )
    parser.add_argument(
        "--stage1a-relaxed-close-hold-bilateral",
        action="store_true",
        help=(
            "diagnostic-only: after privileged bilateral pad contact, hold the "
            "last canonical rate-limited gripper target instead of continuing "
            "to the binary closed endpoint"
        ),
    )
    parser.add_argument(
        "--stage1a-relaxed-close-hold-event",
        choices=("FIRST_CONTACT", "BILATERAL", "STABLE"),
        default="BILATERAL",
    )
    parser.add_argument(
        "--stage1a-simplified-close-persistence-gate",
        action="store_true",
        help=(
            "bounded diagnostic: at 15-20 mm and <=15 deg with five "
            "consecutive safety-clean 50-Hz observations, submit exactly "
            "one canonical CLOSE; frozen GRU CLOSE remains logging-only"
        ),
    )
    parser.add_argument(
        "--stage1a-close-speed-scale",
        type=float,
        choices=(0.25, 0.50, 0.75, 1.00),
        default=1.00,
        help=(
            "bounded mechanics diagnostic only: reduce the existing canonical "
            "per-environment gripper target-speed cap; never expands authority"
        ),
    )
    parser.add_argument(
        "--stage1a-two-stage-close",
        action="store_true",
        help=(
            "bounded mechanics diagnostic only: canonical normal CLOSE until "
            "first valid pad contact, then 25-percent speed until stable, then HOLD"
        ),
    )
    parser.add_argument(
        "--stage1a-privileged-close-motion-interlock",
        action="store_true",
        help=(
            "opt-in Reward-V3 3K diagnostic: use the simplified 15-20 mm, "
            "15-deg, five-step safety gate for one canonical CLOSE and "
            "command zero XYZ throughout closure; GRU CLOSE is logging-only"
        ),
    )
    parser.add_argument(
        "--stage1a-accepted-transitions",
        type=int,
        default=STAGE1A_ACCEPTED_TRANSITION_TARGET,
    )
    parser.add_argument(
        "--stage1a-bc-checkpoint",
        type=Path,
        default=STAGE1A_FROZEN_GRU_CHECKPOINT,
    )
    parser.add_argument(
        "--stage1a-bc-checkpoint-sha256",
        default=STAGE1A_FROZEN_GRU_SHA256,
    )
    parser.add_argument("--stage1a-output-dir", type=Path)
    parser.add_argument("--stage1a-wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stage1a-wandb-mode", choices=("online", "offline"), default="offline"
    )
    parser.add_argument(
        "--stage1a-wandb-project", default="geniesim-g2-stage1a-residual-sac"
    )
    parser.add_argument("--stage1a-wandb-entity")
    parser.add_argument("--stage1a-wandb-run-name")
    parser.add_argument("--stage1a-wandb-group")
    args = parser.parse_args()
    keyboard_v3_requested = args.keyboard_v3_collection_root is not None
    if sum(
        (
            bool(args.stage1a_isaac_short_smoke),
            bool(args.stage1a_training_15k),
            bool(args.stage1a_reward_v3_smoke),
            bool(args.stage1a_hybrid_activation_smoke),
            bool(args.stage1a_close_admission_diagnostic),
            args.stage1a_close_residual_sweep_mm is not None,
            args.stage1a_close_persistent_plan is not None,
        )
    ) > 1:
        raise SystemExit("STAGE1A_MODES_ARE_MUTUALLY_EXCLUSIVE")
    stage1a_requested = bool(
        args.stage1a_isaac_short_smoke
        or args.stage1a_training_15k
        or args.stage1a_reward_v3_smoke
        or args.stage1a_hybrid_activation_smoke
        or args.stage1a_close_admission_diagnostic
        or args.stage1a_close_residual_sweep_mm is not None
        or args.stage1a_close_persistent_plan is not None
    )
    playback_actor = (
        args.stage1a_playback_actor_checkpoint.resolve()
        if args.stage1a_playback_actor_checkpoint is not None
        else STAGE1A_RESIDUAL_ACTOR_CHECKPOINT
    )
    playback_actor_sha256 = (
        args.stage1a_playback_actor_checkpoint_sha256
        if args.stage1a_playback_actor_checkpoint is not None
        else STAGE1A_RESIDUAL_ACTOR_SHA256
    )
    if (args.stage1a_playback_actor_checkpoint is None) != (
        args.stage1a_playback_actor_checkpoint_sha256 is None
    ):
        raise SystemExit("STAGE1A_PLAYBACK_ACTOR_PATH_HASH_PAIR_REQUIRED")
    if (
        (args.stage1a_training_15k or args.stage1a_reward_v3_smoke)
        and args.stage1a_replay_strategy == "HER"
    ):
        raise SystemExit(
            "STAGE1A_HER_REQUIRES_VALIDATED_GOAL_CONDITIONED_REPLAY_AUTHORITY"
        )
    if stage1a_requested:
        if (
            not args.execute_live
            or (args.gui and not args.stage1a_stable_playback)
            or not args.keyboard_v3_direct_pregrasp_v2
            or args.keyboard_v3_v2_smoke_only
            or keyboard_v3_requested
            or (
                args.seed != 42
                and args.stage1a_close_residual_sweep_mm is None
                and args.stage1a_close_persistent_plan is None
                and not args.stage1a_stable_playback
            )
            or args.stage1a_accepted_transitions
            != (
                STAGE1A_REWARD_V3_TRANSITION_TARGET
                if args.stage1a_reward_v3_smoke
                else STAGE1A_TRAINING_TRANSITION_TARGET
                if args.stage1a_training_15k
                else STAGE1A_ACCEPTED_TRANSITION_TARGET
            )
            or (
                args.stage1a_hybrid_activation_smoke
                and not 1 <= args.stage1a_hybrid_activation_max_steps <= 512
            )
            or (
                args.stage1a_stable_playback
                and not args.stage1a_hybrid_activation_smoke
            )
            or (
                args.stage1a_playback_video_path is not None
                and not args.stage1a_stable_playback
            )
            or (
                (args.stage1a_hybrid_activation_smoke or args.stage1a_training_15k or args.stage1a_reward_v3_smoke)
                and (
                    not STAGE1A_FAR_REACH_BC_CHECKPOINT.is_file()
                    or _sha256(STAGE1A_FAR_REACH_BC_CHECKPOINT)
                    != STAGE1A_FAR_REACH_BC_SHA256
                    or not playback_actor.is_file()
                    or _sha256(playback_actor) != playback_actor_sha256
                )
            )
            or args.stage1a_output_dir is None
            or (
                (args.stage1a_training_15k or args.stage1a_reward_v3_smoke)
                and (not args.stage1a_wandb or args.stage1a_wandb_mode != "online")
            )
            or (
                args.stage1a_reward_v3_smoke
                and args.stage1a_replay_strategy != "SAC"
            )
            or (
                args.stage1a_stable_only_reward_v3
                and (
                    not args.stage1a_training_15k
                    or args.stage1a_replay_strategy != "HER_FORCE"
                    or not args.stage1a_privileged_close_motion_interlock
                )
            )
            or (
                args.stage1a_stable_only_reward_v3
                and args.stage1a_training_seed < 0
            )
            or args.stage1a_bc_checkpoint.resolve()
            != STAGE1A_FROZEN_GRU_CHECKPOINT.resolve()
            or args.stage1a_bc_checkpoint_sha256 != STAGE1A_FROZEN_GRU_SHA256
            or not args.stage1a_bc_checkpoint.is_file()
            or _sha256(args.stage1a_bc_checkpoint.resolve())
            != STAGE1A_FROZEN_GRU_SHA256
            or args.diagnostic_asset_variant != "custom"
            or args.diagnostic_asset_path is None
            or args.diagnostic_asset_path.resolve() != CANDIDATE_A_ASSET.resolve()
            or args.diagnostic_asset_sha256
            != EXPECTED_CANDIDATE_A_ASSET_SHA256
            or args.trajectory_replay is not None
            or args.candidate_a_contact_free
            or args.candidate_a_contact_free_runtime_replan
            or args.candidate_a_contact_last_ab
            or args.open_only
            or args.bc_checkpoint is not None
            or args.contact_free_bc_playback_checkpoint is not None
            or (
                args.passive_contact_telemetry_output is not None
                and not (
                    args.stage1a_close_residual_sweep_mm is not None
                    and args.stage1a_extended_hard_stop_telemetry
                )
            )
            or args.minimum_post_contact_observation_ms != 0.0
            or (
                (
                    args.stage1a_close_admission_diagnostic
                    or args.stage1a_close_residual_sweep_mm is not None
                )
                and args.stage1a_wandb
            )
            or (
                args.stage1a_close_calibration_sample is not None
                and not (
                    args.stage1a_close_admission_diagnostic
                    or args.stage1a_close_residual_sweep_mm is not None
                )
            )
            or (
                args.stage1a_close_residual_sweep_mm is not None
                and args.stage1a_close_calibration_sample is None
            )
            or (
                args.stage1a_extended_hard_stop_telemetry
                and args.stage1a_close_residual_sweep_mm is None
                and args.stage1a_close_persistent_plan is None
            )
            or (
                args.stage1a_geometry_forced_close_diagnostic
                and args.stage1a_close_residual_sweep_mm is None
                and args.stage1a_close_persistent_plan is None
            )
            or (
                args.stage1a_relaxed_close_hold_bilateral
                and not (
                    (
                        args.stage1a_close_residual_sweep_mm is not None
                        or args.stage1a_close_persistent_plan is not None
                    )
                    and args.stage1a_geometry_forced_close_diagnostic
                )
            )
            or (
                not args.stage1a_relaxed_close_hold_bilateral
                and args.stage1a_relaxed_close_hold_event != "BILATERAL"
            )
            or (
                args.stage1a_simplified_close_persistence_gate
                and not (
                    (
                        args.stage1a_close_residual_sweep_mm is not None
                        or args.stage1a_close_persistent_plan is not None
                    )
                    and args.stage1a_geometry_forced_close_diagnostic
                )
            )
            or (
                args.stage1a_close_speed_scale != 1.0
                and not (
                    args.stage1a_close_residual_sweep_mm is not None
                    and args.stage1a_geometry_forced_close_diagnostic
                    and args.stage1a_simplified_close_persistence_gate
                )
            )
            or (
                args.stage1a_two_stage_close
                and not (
                    args.stage1a_close_residual_sweep_mm is not None
                    and args.stage1a_geometry_forced_close_diagnostic
                    and args.stage1a_simplified_close_persistence_gate
                    and args.stage1a_close_speed_scale == 1.0
                )
            )
            or (
                args.stage1a_privileged_close_motion_interlock
                and not (
                    args.stage1a_reward_v3_smoke
                    or args.stage1a_stable_only_reward_v3
                )
            )
            or (
                args.stage1a_close_persistent_plan is not None
                and not args.stage1a_close_persistent_plan.is_file()
            )
            or (
                args.stage1a_close_residual_sweep_mm is None
                and any(
                    value != 0.0
                    for value in (
                        args.stage1a_close_lateral_offset_mm,
                        args.stage1a_close_height_offset_mm,
                        args.stage1a_close_approach_yaw_deg,
                    )
                )
            )
            or (
                args.stage1a_close_calibration_sample is not None
                and not args.stage1a_close_calibration_sample.is_file()
            )
        ):
            raise SystemExit("STAGE1A_ISAAC_SHORT_SMOKE_SCOPE_MISMATCH")
        if args.stage1a_output_dir.exists() or args.output.exists():
            raise SystemExit("STAGE1A_ISAAC_SHORT_SMOKE_REFUSES_OVERWRITE")
    elif (
        args.stage1a_output_dir is not None
        or args.stage1a_wandb
        or args.stage1a_wandb_entity is not None
        or args.stage1a_wandb_run_name is not None
        or args.stage1a_wandb_group is not None
        or args.stage1a_replay_strategy != "HER_FORCE"
        or args.stage1a_stable_only_reward_v3
        or args.stage1a_training_seed != 42
        or args.stage1a_playback_actor_checkpoint is not None
        or args.stage1a_stable_playback
        or args.stage1a_playback_video_path is not None
        or args.stage1a_accepted_transitions
        != STAGE1A_ACCEPTED_TRANSITION_TARGET
        or args.stage1a_hybrid_activation_max_steps != 256
        or args.stage1a_bc_checkpoint.resolve()
        != STAGE1A_FROZEN_GRU_CHECKPOINT.resolve()
        or args.stage1a_bc_checkpoint_sha256 != STAGE1A_FROZEN_GRU_SHA256
        or args.stage1a_close_calibration_sample is not None
        or args.stage1a_extended_hard_stop_telemetry
        or args.stage1a_geometry_forced_close_diagnostic
        or args.stage1a_relaxed_close_hold_bilateral
        or args.stage1a_relaxed_close_hold_event != "BILATERAL"
        or args.stage1a_simplified_close_persistence_gate
        or args.stage1a_close_speed_scale != 1.0
        or args.stage1a_two_stage_close
        or args.stage1a_privileged_close_motion_interlock
    ):
        raise SystemExit("STAGE1A_OPTIONS_REQUIRE_STAGE1A_ISAAC_SHORT_SMOKE")
    if keyboard_v3_requested:
        if (
            not args.execute_live
            or not args.gui
            or not args.keyboard_v3_direct_pregrasp_v2
            or args.candidate_a_contact_free_runtime_replan
            or args.keyboard_v3_v2_smoke_only
            or args.diagnostic_asset_variant != "custom"
            or args.diagnostic_asset_path is None
            or args.diagnostic_asset_path.resolve() != CANDIDATE_A_ASSET.resolve()
            or args.diagnostic_asset_sha256
            != EXPECTED_CANDIDATE_A_ASSET_SHA256
            or args.keyboard_v3_episode_id is None
            or not args.keyboard_v3_episode_id.strip()
            or args.keyboard_v3_backoff_m
            not in (0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022)
            or args.keyboard_v3_maximum_steps <= 0
            or args.keyboard_v3_maximum_steps > 1600
            or not 0.0 < args.keyboard_v3_translation_step_m <= 0.0045
            or not 0.0 <= args.keyboard_v3_motion_min_interval_s <= 1.0
            or not 1 <= args.keyboard_v3_terminal_smoothing_steps <= 10
            or args.keyboard_v3_session_max_episodes < 0
            or (
                args.keyboard_v3_session_max_episodes > 0
                and not args.keyboard_v3_continuous_session
            )
            or args.bc_checkpoint is not None
            or args.contact_free_bc_playback_checkpoint is not None
            or args.passive_contact_telemetry_output is not None
            or args.attribution_output is not None
            or args.candidate_a_contact_free
            or args.candidate_a_contact_last_ab
            or args.open_only
            or args.stop_on_hard_stop
            or args.canonical_contact_free_collection_output is not None
            or args.runtime_replan_canonical_contact_free_collection_output
            is not None
            or args.adaptive_waypoints
            or args.adaptive_collision_report is not None
        ):
            raise SystemExit("KEYBOARD_V3_ONE_SHOT_SCOPE_MISMATCH")
        collection_root = args.keyboard_v3_collection_root.resolve()
        if not collection_root.is_dir():
            raise SystemExit("KEYBOARD_V3_COLLECTION_ROOT_MISSING")
        required_collection_entries = (
            collection_root / "COLLECTION_MANIFEST.json",
            collection_root / "episodes",
            collection_root / "rejected_episodes",
            collection_root / "result_receipts",
        )
        if not all(path.exists() for path in required_collection_entries):
            raise SystemExit("KEYBOARD_V3_COLLECTION_ROOT_INCOMPLETE")
        episode_name = f"{args.keyboard_v3_episode_id}.hdf5"
        receipt_name = f"{args.keyboard_v3_episode_id}.result.json"
        if any(
            path.exists()
            for path in (
                collection_root / "episodes" / episode_name,
                collection_root / "rejected_episodes" / episode_name,
                collection_root / "result_receipts" / receipt_name,
                args.output.resolve(),
            )
        ):
            raise SystemExit("KEYBOARD_V3_ONE_SHOT_REFUSES_OVERWRITE")
    elif args.keyboard_v3_episode_id is not None:
        raise SystemExit("KEYBOARD_V3_EPISODE_ID_WITHOUT_COLLECTION_ROOT")
    elif args.keyboard_v3_continuous_session or args.keyboard_v3_session_max_episodes:
        raise SystemExit("KEYBOARD_V3_SESSION_OPTIONS_REQUIRE_COLLECTION_ROOT")
    if args.keyboard_v3_direct_pregrasp_v2 and not (
        keyboard_v3_requested or args.keyboard_v3_v2_smoke_only or stage1a_requested
    ):
        raise SystemExit("KEYBOARD_V3_DIRECT_INIT_REQUIRES_KEYBOARD_OR_SMOKE")
    if args.keyboard_v3_v2_smoke_only and (
        keyboard_v3_requested
        or not args.keyboard_v3_direct_pregrasp_v2
        or not args.execute_live
        or not args.gui
        or args.seed != 42
        or args.diagnostic_asset_variant != "custom"
        or args.diagnostic_asset_path is None
        or args.diagnostic_asset_path.resolve() != CANDIDATE_A_ASSET.resolve()
        or args.diagnostic_asset_sha256 != EXPECTED_CANDIDATE_A_ASSET_SHA256
        or args.candidate_a_contact_free_runtime_replan
        or args.candidate_a_contact_free
        or args.candidate_a_contact_last_ab
        or args.bc_checkpoint is not None
        or args.contact_free_bc_playback_checkpoint is not None
        or args.passive_contact_telemetry_output is not None
    ):
        raise SystemExit("KEYBOARD_V3_V2_SMOKE_SCOPE_MISMATCH")
    runtime_replan_collection_requested = (
        args.runtime_replan_canonical_contact_free_collection_output is not None
    )
    p06_validation_requested = args.candidate_a_bc_validation_output is not None
    if p06_validation_requested != (
        args.candidate_a_bc_validation_episode_id is not None
    ):
        raise SystemExit("P06_VALIDATION_OUTPUT_AND_EPISODE_ID_REQUIRED_TOGETHER")
    if p06_validation_requested and (
        not args.candidate_a_contact_free_runtime_replan
        or runtime_replan_collection_requested
        or args.canonical_contact_free_collection_output is not None
        or args.contact_free_bc_playback_checkpoint is not None
        or args.bc_checkpoint is not None
        or args.canonical_capture_rate_hz != 25
    ):
        raise SystemExit("P06_VALIDATION_SCOPE_MISMATCH")
    if p06_validation_requested and args.candidate_a_bc_validation_output.exists():
        raise SystemExit("P06_VALIDATION_OUTPUT_MUST_NOT_EXIST")
    if (
        args.canonical_contact_free_collection_output is not None
        and runtime_replan_collection_requested
    ):
        raise SystemExit("CANONICAL_COLLECTION_PROVENANCE_FLAGS_CONFLICT")
    if runtime_replan_collection_requested and not args.candidate_a_contact_free_runtime_replan:
        raise SystemExit("RUNTIME_REPLAN_COLLECTION_REQUIRES_RUNTIME_REPLAN")
    if runtime_replan_collection_requested and args.independent_collection_episode_id is None:
        raise SystemExit("RUNTIME_REPLAN_COLLECTION_EPISODE_ID_REQUIRED")
    if (
        args.canonical_capture_rate_hz != 50
        and not runtime_replan_collection_requested
        and not p06_validation_requested
    ):
        raise SystemExit("CANONICAL_CAPTURE_RATE_REQUIRES_RUNTIME_REPLAN_COLLECTION")
    if (
        not runtime_replan_collection_requested
        and args.independent_collection_episode_id is not None
    ):
        raise SystemExit("INDEPENDENT_COLLECTION_EPISODE_ID_UNEXPECTED")
    if args.candidate_a_contact_free_runtime_replan and args.trajectory_replay is not None:
        raise SystemExit("RUNTIME_REPLAN_REJECTS_FROZEN_REPLAY")
    if (
        not args.candidate_a_contact_free_runtime_replan
        and not args.keyboard_v3_direct_pregrasp_v2
        and not args.keyboard_v3_v2_smoke_only
        and args.trajectory_replay is None
    ):
        raise SystemExit("CUROBO_LIVE_FROZEN_REPLAY_REQUIRED")
    if args.planner_active_velocity_limit_rad_s not in (0.8, 0.85):
        raise SystemExit("CUROBO_LIVE_UNAUTHORIZED_ACTIVE_VELOCITY_LIMIT")
    custom_asset_arguments = (
        args.diagnostic_asset_path,
        args.diagnostic_asset_sha256,
    )
    if args.diagnostic_asset_variant == "custom":
        if not all(value is not None for value in custom_asset_arguments):
            raise SystemExit("CUROBO_CUSTOM_ASSET_ARGUMENTS_INCOMPLETE")
        if (
            len(str(args.diagnostic_asset_sha256)) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(args.diagnostic_asset_sha256)
            )
        ):
            raise SystemExit("CUROBO_CUSTOM_ASSET_SHA256_INVALID")
    elif any(value is not None for value in custom_asset_arguments):
        raise SystemExit("CUROBO_CUSTOM_ASSET_ARGUMENTS_WITH_NONCUSTOM_VARIANT")
    if args.open_only and any(
        value is not None
        for value in (
            args.bc_checkpoint,
            args.contact_free_bc_playback_checkpoint,
            args.passive_contact_telemetry_output,
        )
    ):
        raise SystemExit("CUROBO_OPEN_ONLY_REJECTS_BC_OR_CONTACT")
    if args.episode_horizon_steps <= 0:
        raise SystemExit("CUROBO_LIVE_INVALID_EPISODE_HORIZON")
    if (args.cosine_p05 is None) != (args.cosine_p10 is None):
        raise SystemExit("CUROBO_COSINE_SHAPING_REQUIRES_P05_AND_P10")
    for name in (
        "ordinary_tracking_cap",
        "critical_tracking_cap",
        "final_settle_cap",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"CUROBO_LIVE_INVALID_{name.upper()}")
    if args.timing_optimized and args.final_convergence_cap <= 0:
        raise SystemExit("CUROBO_LIVE_TIMING_OPTIMIZED_REQUIRES_FINAL_CONVERGENCE")
    if args.adaptive_waypoints and args.adaptive_collision_report is None:
        raise SystemExit("CUROBO_LIVE_ADAPTIVE_COLLISION_REPORT_REQUIRED")
    if args.telemetry_flush_steps < 0 or args.telemetry_flush_steps > 2:
        raise SystemExit("CUROBO_LIVE_TELEMETRY_FLUSH_STEPS_INVALID")
    bc_arguments = (
        args.bc_checkpoint,
        args.bc_checkpoint_sha256,
        args.grasp_ready_geometry,
    )
    if any(value is not None for value in bc_arguments):
        if not all(value is not None for value in bc_arguments):
            raise SystemExit("CUROBO_BC_HANDOFF_ARGUMENTS_INCOMPLETE")
        if not 1 <= args.bc_closed_loop_steps <= 200:
            raise SystemExit("CUROBO_BC_CLOSED_LOOP_STEPS_OUT_OF_RANGE")
    elif args.bc_closed_loop_steps != 0:
        raise SystemExit("CUROBO_BC_STEPS_WITHOUT_CHECKPOINT")
    playback_arguments = (
        args.contact_free_bc_playback_checkpoint,
        args.contact_free_bc_playback_checkpoint_sha256,
    )
    if any(value is not None for value in playback_arguments):
        if not all(value is not None for value in playback_arguments):
            raise SystemExit("CONTACT_FREE_BC_PLAYBACK_ARGUMENTS_INCOMPLETE")
        playback_hash = str(args.contact_free_bc_playback_checkpoint_sha256)
        if len(playback_hash) != 64 or any(
            character not in "0123456789abcdef" for character in playback_hash
        ):
            raise SystemExit("CONTACT_FREE_BC_PLAYBACK_SHA256_INVALID")
        if (
            not args.execute_live
            or not args.candidate_a_contact_free
            or args.bc_checkpoint is not None
            or args.candidate_a_contact_free_runtime_replan
            or args.canonical_contact_free_collection_output is not None
            or args.runtime_replan_canonical_contact_free_collection_output is not None
            or args.cosine_p05 is not None
            or args.cosine_p10 is not None
        ):
            raise SystemExit("CONTACT_FREE_BC_PLAYBACK_SCOPE_MISMATCH")
    if args.state_preserving_bridge and args.bc_checkpoint is None:
        raise SystemExit("CUROBO_BC_BRIDGE_WITHOUT_CHECKPOINT")
    if args.candidate_a_contact_last_ab and not (
        args.execute_live
        and args.seed == 42
        and args.diagnostic_asset_variant == "custom"
        and args.diagnostic_asset_path is not None
        and args.diagnostic_asset_path.resolve() == CANDIDATE_A_ASSET.resolve()
        and args.diagnostic_asset_sha256 == EXPECTED_CANDIDATE_A_ASSET_SHA256
        and args.bc_checkpoint is not None
        and args.passive_contact_telemetry_output is not None
        and args.stop_on_hard_stop
    ):
        raise SystemExit("CONTACT_LAST_AB_FROZEN_CONDITION_MISMATCH")
    if args.candidate_a_contact_free and not (
        args.execute_live
        and args.seed == 42
        and args.diagnostic_asset_variant == "custom"
        and args.diagnostic_asset_path is not None
        and args.diagnostic_asset_path.resolve() == CANDIDATE_A_ASSET.resolve()
        and args.diagnostic_asset_sha256 == EXPECTED_CANDIDATE_A_ASSET_SHA256
        and args.bc_checkpoint is None
        and args.passive_contact_telemetry_output is None
        and args.attribution_output is None
        and not args.open_only
        and not args.stop_on_hard_stop
        and not args.candidate_a_contact_last_ab
    ):
        raise SystemExit("CANDIDATE_A_CONTACT_FREE_FROZEN_CONDITION_MISMATCH")
    if args.candidate_a_contact_free_runtime_replan and not (
        args.execute_live
        and (
            args.seed == 42
            or runtime_replan_collection_requested
            or p06_validation_requested
            or keyboard_v3_requested
        )
        and args.diagnostic_asset_variant == "custom"
        and args.diagnostic_asset_path is not None
        and args.diagnostic_asset_path.resolve() == CANDIDATE_A_ASSET.resolve()
        and args.diagnostic_asset_sha256 == EXPECTED_CANDIDATE_A_ASSET_SHA256
        and args.bc_checkpoint is None
        and args.passive_contact_telemetry_output is None
        and args.attribution_output is None
        and not args.open_only
        and not args.stop_on_hard_stop
        and not args.candidate_a_contact_last_ab
        and not args.candidate_a_contact_free
        and args.canonical_contact_free_collection_output is None
        and (
            (not runtime_replan_collection_requested and not p06_validation_requested)
            or args.runtime_replan_canonical_contact_free_collection_output is not None
            or args.candidate_a_bc_validation_output is not None
        )
        and not args.adaptive_waypoints
        and args.adaptive_collision_report is None
    ):
        raise SystemExit("CANDIDATE_A_CONTACT_FREE_RUNTIME_REPLAN_FROZEN_CONDITION_MISMATCH")
    if args.canonical_contact_free_collection_output is not None:
        if not args.candidate_a_contact_free:
            raise SystemExit("CANONICAL_COLLECTION_REQUIRES_CANDIDATE_A_CONTACT_FREE")
        if args.canonical_contact_free_collection_output.exists():
            raise SystemExit("CANONICAL_COLLECTION_OUTPUT_MUST_NOT_EXIST")
    if runtime_replan_collection_requested:
        from geniesim.rl.isaaclab.g2_policy_branch.independent_contact_free_collection import (
            IndependentCollectionEpisode,
        )

        try:
            IndependentCollectionEpisode(
                episode_id=str(args.independent_collection_episode_id), seed=int(args.seed)
            ).validated()
        except ValueError as error:
            raise SystemExit(f"RUNTIME_REPLAN_COLLECTION_EPISODE_INVALID:{error}") from error
        if args.runtime_replan_canonical_contact_free_collection_output.exists():
            raise SystemExit("RUNTIME_REPLAN_COLLECTION_OUTPUT_MUST_NOT_EXIST")
    # Physics-substep telemetry is also valid for planner-only M1/M1.5
    # qualification.  Post-contact dwell still requires BC because the
    # planner-only path keeps the gripper OPEN and never enters CONTACT.
    if not 0.0 <= args.minimum_post_contact_observation_ms <= 1000.0:
        raise SystemExit("CUROBO_POST_CONTACT_OBSERVATION_OUT_OF_RANGE")
    if (
        args.minimum_post_contact_observation_ms > 0.0
        and args.passive_contact_telemetry_output is None
    ):
        raise SystemExit("CUROBO_POST_CONTACT_OBSERVATION_WITHOUT_TELEMETRY")
    atexit.register(lambda: print("ATEXIT_RETURN", flush=True))
    freeze = _source_freeze(
        keyboard_v3_branch=(
            keyboard_v3_requested
            or args.keyboard_v3_direct_pregrasp_v2
            or args.keyboard_v3_v2_smoke_only
            or p06_validation_requested
            or runtime_replan_collection_requested
            or stage1a_requested
        )
    )
    if freeze["SOURCE_FREEZE"] != "PASS":
        raise SystemExit("CUROBO_LIVE_SOURCE_FREEZE_FAILED")
    print("SOURCE_FREEZE_OK", flush=True)
    if not args.execute_live:
        print(json.dumps(freeze, sort_keys=True))
        return 0
    result = _run_live(
        output=args.output,
        replay=(
            args.trajectory_replay.resolve()
            if args.trajectory_replay is not None
            else None
        ),
        seed=args.seed,
        attribution_output=(
            args.attribution_output.resolve()
            if args.attribution_output is not None
            else None
        ),
        timing_optimized=args.timing_optimized,
        episode_horizon_steps=args.episode_horizon_steps,
        ordinary_tracking_cap=args.ordinary_tracking_cap,
        critical_tracking_cap=args.critical_tracking_cap,
        final_convergence_cap=args.final_convergence_cap,
        final_settle_cap=args.final_settle_cap,
        planner_active_velocity_limit_rad_s=(
            args.planner_active_velocity_limit_rad_s
        ),
        adaptive_waypoints=args.adaptive_waypoints,
        adaptive_collision_report=(
            args.adaptive_collision_report.resolve()
            if args.adaptive_collision_report is not None
            else None
        ),
        telemetry_flush_steps=args.telemetry_flush_steps,
        progress_deadband_m=args.progress_deadband_m,
        stall_grace_steps=args.stall_grace_steps,
        max_stall_steps=args.max_stall_steps,
        retreat_threshold_m=args.retreat_threshold_m,
        bc_checkpoint=(
            args.bc_checkpoint.resolve()
            if args.bc_checkpoint is not None
            else None
        ),
        bc_checkpoint_sha256=args.bc_checkpoint_sha256,
        bc_closed_loop_steps=args.bc_closed_loop_steps,
        grasp_ready_geometry=(
            args.grasp_ready_geometry.resolve()
            if args.grasp_ready_geometry is not None
            else None
        ),
        state_preserving_bridge=args.state_preserving_bridge,
        passive_contact_telemetry_output=(
            args.passive_contact_telemetry_output.resolve()
            if args.passive_contact_telemetry_output is not None
            else None
        ),
        minimum_post_contact_observation_s=(
            args.minimum_post_contact_observation_ms / 1000.0
        ),
        diagnostic_asset_variant=args.diagnostic_asset_variant,
        diagnostic_asset_path=(
            args.diagnostic_asset_path.resolve()
            if args.diagnostic_asset_path is not None
            else None
        ),
        diagnostic_asset_sha256=args.diagnostic_asset_sha256,
        open_only=args.open_only,
        stop_on_hard_stop=args.stop_on_hard_stop,
        candidate_a_contact_last_ab=args.candidate_a_contact_last_ab,
        candidate_a_contact_free=args.candidate_a_contact_free,
        candidate_a_contact_free_runtime_replan=(
            args.candidate_a_contact_free_runtime_replan
        ),
        cosine_p05=args.cosine_p05,
        cosine_p10=args.cosine_p10,
        canonical_contact_free_collection_output=(
            args.canonical_contact_free_collection_output.resolve()
            if args.canonical_contact_free_collection_output is not None
            else None
        ),
        runtime_replan_canonical_contact_free_collection_output=(
            args.runtime_replan_canonical_contact_free_collection_output.resolve()
            if args.runtime_replan_canonical_contact_free_collection_output is not None
            else None
        ),
        canonical_capture_rate_hz=args.canonical_capture_rate_hz,
        independent_collection_episode_id=args.independent_collection_episode_id,
        contact_free_bc_playback_checkpoint=(
            args.contact_free_bc_playback_checkpoint.resolve()
            if args.contact_free_bc_playback_checkpoint is not None
            else None
        ),
        contact_free_bc_playback_checkpoint_sha256=(
            args.contact_free_bc_playback_checkpoint_sha256
        ),
        gui=args.gui,
        keyboard_v3_collection_root=(
            args.keyboard_v3_collection_root.resolve()
            if args.keyboard_v3_collection_root is not None
            else None
        ),
        keyboard_v3_episode_id=args.keyboard_v3_episode_id,
        keyboard_v3_backoff_m=args.keyboard_v3_backoff_m,
        keyboard_v3_maximum_steps=args.keyboard_v3_maximum_steps,
        keyboard_v3_translation_step_m=args.keyboard_v3_translation_step_m,
        keyboard_v3_motion_min_interval_s=(
            args.keyboard_v3_motion_min_interval_s
        ),
        keyboard_v3_terminal_smoothing_steps=(
            args.keyboard_v3_terminal_smoothing_steps
        ),
        keyboard_v3_continuous_session=args.keyboard_v3_continuous_session,
        keyboard_v3_session_max_episodes=(
            args.keyboard_v3_session_max_episodes
        ),
        keyboard_v3_direct_pregrasp_v2=args.keyboard_v3_direct_pregrasp_v2,
        keyboard_v3_v2_smoke_only=args.keyboard_v3_v2_smoke_only,
        candidate_a_bc_validation_output=(
            args.candidate_a_bc_validation_output.resolve()
            if args.candidate_a_bc_validation_output is not None
            else None
        ),
        candidate_a_bc_validation_episode_id=(
            args.candidate_a_bc_validation_episode_id
        ),
        stage1a_isaac_accepted_transitions=(
            args.stage1a_accepted_transitions if stage1a_requested else None
        ),
        stage1a_training_15k=args.stage1a_training_15k,
        stage1a_reward_v3_smoke=args.stage1a_reward_v3_smoke,
        stage1a_stable_only_reward_v3=(
            args.stage1a_stable_only_reward_v3
        ),
        stage1a_replay_strategy=args.stage1a_replay_strategy,
        stage1a_training_seed=args.stage1a_training_seed,
        stage1a_hybrid_activation_smoke=(
            args.stage1a_hybrid_activation_smoke
        ),
        stage1a_hybrid_activation_max_steps=(
            args.stage1a_hybrid_activation_max_steps
        ),
        stage1a_playback_actor_checkpoint=(
            playback_actor
            if args.stage1a_playback_actor_checkpoint is not None
            else None
        ),
        stage1a_playback_actor_checkpoint_sha256=(
            playback_actor_sha256
            if args.stage1a_playback_actor_checkpoint is not None
            else None
        ),
        stage1a_stable_playback=args.stage1a_stable_playback,
        stage1a_playback_video_path=(
            args.stage1a_playback_video_path.resolve()
            if args.stage1a_playback_video_path is not None
            else None
        ),
        stage1a_close_admission_diagnostic=(
            args.stage1a_close_admission_diagnostic
        ),
        stage1a_close_residual_sweep_mm=(
            args.stage1a_close_residual_sweep_mm
        ),
        stage1a_close_lateral_offset_mm=args.stage1a_close_lateral_offset_mm,
        stage1a_close_height_offset_mm=args.stage1a_close_height_offset_mm,
        stage1a_close_approach_yaw_deg=args.stage1a_close_approach_yaw_deg,
        stage1a_extended_hard_stop_telemetry=(
            args.stage1a_extended_hard_stop_telemetry
        ),
        stage1a_geometry_forced_close_diagnostic=(
            args.stage1a_geometry_forced_close_diagnostic
        ),
        stage1a_relaxed_close_hold_bilateral=(
            args.stage1a_relaxed_close_hold_bilateral
        ),
        stage1a_relaxed_close_hold_event=args.stage1a_relaxed_close_hold_event,
        stage1a_simplified_close_persistence_gate=(
            args.stage1a_simplified_close_persistence_gate
        ),
        stage1a_close_speed_scale=args.stage1a_close_speed_scale,
        stage1a_two_stage_close=args.stage1a_two_stage_close,
        stage1a_privileged_close_motion_interlock=(
            args.stage1a_privileged_close_motion_interlock
        ),
        stage1a_close_calibration_sample=(
            args.stage1a_close_calibration_sample.resolve()
            if args.stage1a_close_calibration_sample is not None
            else None
        ),
        stage1a_close_persistent_plan=(
            args.stage1a_close_persistent_plan.resolve()
            if args.stage1a_close_persistent_plan is not None
            else None
        ),
        stage1a_bc_checkpoint=(
            args.stage1a_bc_checkpoint.resolve() if stage1a_requested else None
        ),
        stage1a_bc_checkpoint_sha256=(
            args.stage1a_bc_checkpoint_sha256 if stage1a_requested else None
        ),
        stage1a_output_dir=(
            args.stage1a_output_dir.resolve() if stage1a_requested else None
        ),
        stage1a_wandb=args.stage1a_wandb,
        stage1a_wandb_mode=args.stage1a_wandb_mode,
        stage1a_wandb_project=args.stage1a_wandb_project,
        stage1a_wandb_entity=args.stage1a_wandb_entity,
        stage1a_wandb_run_name=args.stage1a_wandb_run_name,
        stage1a_wandb_group=args.stage1a_wandb_group,
    )
    print("PROCESS_EXIT", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
