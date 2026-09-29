#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Current P0-C RGB-D/action integration child.

The legacy P0-C router instantiated a provider-less ``EESafetyValidator`` and
therefore rejected every otherwise valid EE request.  This child deliberately
does not use that obsolete validation plumbing.  It reuses the current P0-B
full-8D deferred packet/``env.step`` ingress and adds only the source-owned
wrist RGB-D, previous-action, and reset contracts required by P0-C.

The functional report is persisted before environment/application shutdown.
Isaac/Kit process finalization is classified by the parent process.
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
PRODUCTION_USD = SOURCE / "geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
FROZEN_TRAJECTORY = ROOT / "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
FROZEN_RUNTIME_AUTHORITY = (
    ROOT
    / "output/policy_branch_p0/p0_b_axis_rebase_fresh_live_20260920_v1/child/P0_B_STATIC_PRE_LIVE_CONTRACT.json"
)
EXPECTED_PRODUCTION_USD_SHA256 = "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
EXPECTED_FROZEN_TRAJECTORY_SHA256 = "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"
SCHEMA = "g2_p0_c_current_integration_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_exact_module(source: Path, name: str) -> Any:
    resolved = source.resolve()
    before = _sha256(resolved)
    spec = importlib.util.spec_from_file_location(f"_{name}_{before[:16]}", resolved)
    if spec is None or spec.loader is None or Path(spec.origin or "").resolve() != resolved:
        raise RuntimeError(f"P0_C_MODULE_SPEC_FAIL:{resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != resolved or _sha256(resolved) != before:
        raise RuntimeError(f"P0_C_MODULE_PROVENANCE_FAIL:{resolved}")
    return module


P0B_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0b_one_shot_integration.py"
P0B = _load_exact_module(P0B_SOURCE, "g2_p0c_current_p0b")


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"NONFINITE_JSON:{value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def _source_manifest(prerequisite_audit: Path, p1_contract: Path) -> dict[str, Any]:
    sources = (
        Path(__file__).resolve(),
        P0B_SOURCE,
        ROOT / "scripts/diagnostics/run_g2_p0a_one_shot_integration.py",
        ROOT / "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py",
        ROOT / "scripts/diagnostics/p0b_one_shot_contract.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/action_interface.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/production_metric_adapter.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/p0_runtime_contract.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/observation.py",
        SOURCE / "geniesim/rl/isaaclab/g2_lift_rgbd_env_cfg.py",
        PRODUCTION_USD,
        FROZEN_TRAJECTORY,
        prerequisite_audit.resolve(),
        p1_contract.resolve(),
        FROZEN_RUNTIME_AUTHORITY,
    )
    missing = [str(path) for path in sources if not path.is_file()]
    hashes = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): _sha256(path) for path in sources if path.is_file()}
    checks = {
        "all_sources_present": not missing,
        "production_usd_hash": _sha256(PRODUCTION_USD) == EXPECTED_PRODUCTION_USD_SHA256,
        "frozen_trajectory_hash": _sha256(FROZEN_TRAJECTORY) == EXPECTED_FROZEN_TRAJECTORY_SHA256,
    }
    manifest_fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "g2_p0_c_current_source_freeze_v1",
        "SOURCE_FREEZE": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing": missing,
        "source_sha256": hashes,
        "manifest_sha256": manifest_fingerprint,
        "production_usd_sha256": hashes.get(str(PRODUCTION_USD.relative_to(ROOT))),
        "frozen_trajectory_sha256": hashes.get(str(FROZEN_TRAJECTORY.relative_to(ROOT))),
    }


def _same_manifest(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(left == right and left.get("SOURCE_FREEZE") == "PASS")


def _validate_prerequisites(audit_path: Path, p1_path: Path) -> dict[str, Any]:
    audit = _strict_json(audit_path)
    p1 = _strict_json(p1_path)
    p0_b = audit.get("p0_b") if isinstance(audit.get("p0_b"), Mapping) else {}
    checks = {
        "audit_status": audit.get("status") == "PASS",
        "audit_no_action_submission": audit.get("no_action_submission") is True,
        "p0_a_compatibility_pass": (audit.get("p0_a") or {}).get("p0_semantic_verdict") == "PASS",
        "p0_b_compatibility_pass": p0_b.get("p0_semantic_verdict") == "PASS",
        "p0_b_authority_is_fresh_adjudication": "p0_b_axis_rebase_fresh_adjudication_20260920_v1" in str(p0_b.get("authoritative_receipt", "")),
        "p1_payload_matches_audit": audit.get("p1_contract") == p1,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "checks": checks,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "p1_contract_path": str(p1_path.resolve()),
        "p1_contract_sha256": _sha256(p1_path),
    }


def _tensor_bytes_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _runtime_binding(env: Any) -> Any:
    from geniesim.rl.isaaclab.g2_lift_env_cfg import SANDBOX
    from geniesim.rl.isaaclab.g2_lift_rgbd_env_cfg import CAMERA_CAPTURE_INTERVAL_STEPS, CAMERA_RESOLUTION
    from geniesim.rl.isaaclab.g2_policy_branch.observation import PolicyDataSemantics
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import P0RuntimeObservationBinding, WristCameraRuntimeContract

    camera = env.scene["right_wrist_camera"]
    resolution = (int(CAMERA_RESOLUTION[0]), int(CAMERA_RESOLUTION[1]))
    if (int(camera.cfg.width), int(camera.cfg.height)) != resolution:
        raise RuntimeError("P0_C_CAMERA_RESOLUTION_MISMATCH")
    update_period = float(SANDBOX.physics_dt_s) * int(CAMERA_CAPTURE_INTERVAL_STEPS)
    if not math.isclose(float(camera.cfg.update_period), update_period, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("P0_C_CAMERA_CADENCE_MISMATCH")
    return P0RuntimeObservationBinding(
        data_semantics=PolicyDataSemantics(),
        wrist_camera=WristCameraRuntimeContract(
            width_px=resolution[0],
            height_px=resolution[1],
            capture_interval_physics_steps=int(CAMERA_CAPTURE_INTERVAL_STEPS),
            maximum_frame_age_s=update_period + 1.0e-6,
        ),
    )


def _capture_observation(*, env: Any, p0a: Any, provider: Any, previous: Any, hidden_reset: bool, binding: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
    from geniesim.rl.isaaclab.g2_lift_methodology import RIGHT_ARM_JOINTS
    from geniesim.rl.isaaclab.g2_policy_branch.observation import PolicyObservation

    camera = env.scene["right_wrist_camera"]
    output = camera.data.output
    if "rgb" not in output or "distance_to_image_plane" not in output:
        raise RuntimeError("P0_C_WRIST_RGBD_STREAM_MISSING")
    rgb = p0a._tensor(output["rgb"])[..., :3]
    raw_depth = p0a._tensor(output["distance_to_image_plane"]).to(torch.float32)
    if raw_depth.ndim == 3:
        raw_depth = raw_depth.unsqueeze(-1)
    if rgb.dtype != torch.uint8 or raw_depth.shape != (*rgb.shape[:3], 1):
        raise RuntimeError("P0_C_RGBD_SHAPE_OR_DTYPE_MISMATCH")
    corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (torch.isfinite(raw_depth) & (raw_depth < 0.0))
    if bool(corrupt.any().item()):
        raise RuntimeError("P0_C_DEPTH_CORRUPT")
    maximum_depth = float(binding.data_semantics.maximum_depth_m)
    valid = torch.isfinite(raw_depth) & (raw_depth > 0.0) & (raw_depth <= maximum_depth)
    depth = torch.where(valid, raw_depth, torch.zeros_like(raw_depth))

    snapshot = provider.capture_snapshot()
    ee_pose = torch.tensor(
        (*snapshot.sample.ee_pose_root.position_m, *snapshot.sample.ee_pose_root.quaternion_xyzw),
        device=env.device,
        dtype=torch.float32,
    ).view(1, 7)
    robot = env.scene["robot"]
    joint_ids, names = robot.find_joints(list(RIGHT_ARM_JOINTS))
    if tuple(names) != RIGHT_ARM_JOINTS:
        raise RuntimeError("P0_C_RIGHT_ARM_ORDER_MISMATCH")
    indices = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    q = p0a._tensor(robot.data.joint_pos).index_select(1, indices).to(torch.float32)
    qd = p0a._tensor(robot.data.joint_vel).index_select(1, indices).to(torch.float32)
    gripper = p0a._gripper_command_telemetry(env.action_manager.get_term("gripper_action"))
    closed = 1.0 if gripper["close_command_active"] else 0.0
    episode_time = p0a._tensor(env.episode_length_buf).to(torch.float32) * float(env.step_dt)
    capture_time, frame_age = camera_capture_time_and_age(camera, fallback_episode_time_s=episode_time)
    sequence = p0a._tensor(camera.frame).to(device=env.device, dtype=torch.int64).reshape(-1)
    observation = PolicyObservation(
        right_wrist_rgb=rgb.unsqueeze(1).clone(),
        right_wrist_depth_m=depth.unsqueeze(1).clone(),
        right_wrist_depth_valid=valid.unsqueeze(1).clone(),
        ee_pose_robot_root_xyzw=ee_pose.unsqueeze(1).clone(),
        arm_joint_position_rad=q.unsqueeze(1).clone(),
        arm_joint_velocity_rad_s=qd.unsqueeze(1).clone(),
        current_gripper_state=torch.full((1, 1, 1), closed, device=env.device, dtype=torch.float32),
        previous_policy_action=torch.tensor(previous.value.values, device=env.device, dtype=torch.float32).view(1, 1, -1),
        data_semantics=binding.data_semantics,
        data_semantic_fingerprint=binding.data_semantics.fingerprint(),
        right_wrist_frame_age_s=frame_age.to(torch.float32).view(1, 1, 1),
        hidden_reset_mask=torch.tensor([[hidden_reset]], device=env.device, dtype=torch.bool),
    )
    binding.validate_observation(observation)
    metadata = {
        "sequence_id": int(sequence[0].item()),
        "capture_timestamp_s": float(capture_time.reshape(-1)[0].item()),
        "control_timestamp_s": float(episode_time.reshape(-1)[0].item()),
        "frame_age_s": float(frame_age.reshape(-1)[0].item()),
        "rgb_shape": list(observation.right_wrist_rgb.shape),
        "rgb_dtype": str(observation.right_wrist_rgb.dtype),
        "rgb_sha256": _tensor_bytes_sha256(observation.right_wrist_rgb),
        "depth_shape": list(observation.right_wrist_depth_m.shape),
        "depth_dtype": str(observation.right_wrist_depth_m.dtype),
        "depth_sha256": _tensor_bytes_sha256(observation.right_wrist_depth_m),
        "depth_valid_shape": list(observation.right_wrist_depth_valid.shape),
        "depth_valid_dtype": str(observation.right_wrist_depth_valid.dtype),
        "depth_valid_fraction": float(observation.right_wrist_depth_valid.to(torch.float32).mean().item()),
        "ee_pose_root_xyzw": [float(value) for value in observation.ee_pose_robot_root_xyzw[0, 0].tolist()],
        "right_arm_names": list(names),
        "previous_policy_action": [float(value) for value in observation.previous_policy_action[0, 0].tolist()],
        "current_gripper_state": closed,
        "hidden_reset": bool(hidden_reset),
    }
    return observation, metadata


def _fresh_capture(*, env: Any, p0a: Any, provider: Any, ledger: Any, latch: Any, previous: Any, binding: Any, hidden_reset: bool, label: str, minimum_hold_steps: int = 0) -> tuple[Any, dict[str, Any]]:
    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import HighLevelPolicyAction

    camera = env.scene["right_wrist_camera"]
    _ = camera.data.output
    before = int(p0a._tensor(camera.frame).reshape(-1)[0].item())
    if minimum_hold_steps < 0:
        raise RuntimeError("P0_C_NEGATIVE_MINIMUM_HOLD_STEPS")
    maximum = max(
        4,
        int(binding.wrist_camera.capture_interval_physics_steps) + 2,
        int(minimum_hold_steps),
    )
    for index in range(maximum):
        action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.5))
        packet, tensor, _ = p0a._build_authoritative_packet(
            high_level=action, batch_size=env.num_envs, device=env.device, latch=latch
        )
        ledger.submit(
            packet=packet,
            authoritative_packet_tensor=tensor,
            label=f"P0_C_{label}_CAMERA_HOLD_{index}",
            packet_role="camera_hold",
            semantic_command="CAMERA_HOLD",
            expected_gripper_intent="OPEN",
        )
        latch.commit(packet.gripper_intent)
        _ = camera.data.output
        after = int(p0a._tensor(camera.frame).reshape(-1)[0].item())
        if after > before and index + 1 >= minimum_hold_steps:
            return _capture_observation(
                env=env,
                p0a=p0a,
                provider=provider,
                previous=previous,
                hidden_reset=hidden_reset,
                binding=binding,
            )
    raise RuntimeError(f"P0_C_{label}_CAMERA_FRESHNESS_TIMEOUT")


def _packet_validation(ledger_payload: Mapping[str, Any]) -> dict[str, Any]:
    records = ledger_payload.get("records")
    errors: list[str] = []
    if not isinstance(records, list) or not records:
        return {"status": "FAIL_CLOSED", "errors": ["NO_PACKET_RECORDS"]}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"ROW_{index}_NOT_OBJECT")
            continue
        if record.get("router_process_action_count") != 0:
            errors.append(f"ROW_{index}_ROUTER_PROCESS_ACTION")
        if record.get("deferred_packet_count") != 1 or record.get("env_step_calls") != 1 or record.get("process_action_count") != 1:
            errors.append(f"ROW_{index}_COUNT")
        if record.get("consumption_state") != "KNOWN_SINGLE_CONSUMPTION":
            errors.append(f"ROW_{index}_CONSUMPTION")
        if (record.get("canonical_packet_identity") or {}).get("same_immutable_packet_consumed_once") is not True:
            errors.append(f"ROW_{index}_IDENTITY")
        if record.get("terminated") or record.get("truncated") or record.get("active_terminations"):
            errors.append(f"ROW_{index}_TERMINATION")
    semantic = [row for row in records if isinstance(row, Mapping) and row.get("packet_role") == "semantic"]
    return {
        "status": "PASS" if not errors and len(semantic) == 1 else "FAIL_CLOSED",
        "errors": errors + ([] if len(semantic) == 1 else ["SEMANTIC_PACKET_COUNT"]),
        "packet_count": len(records),
        "semantic_packet_count": len(semantic),
        "all_packet_single_consumption": not errors,
    }


def _run_functional(*, env: Any, p0a: Any, provider: Any, counter: Any, prerequisite: Mapping[str, Any], p1_path: Path) -> dict[str, Any]:
    import torch
    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import AbstractGripperIntent, GripperHysteresisLatch, HighLevelPolicyAction
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import PreviousAcceptedPolicyAction, load_p1_collection_runtime_semantic_contract
    from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import DeferredFull8DActionPacketPort

    p0a._assert_existing_p0_runtime_surface(env)
    observation, _ = env.reset(seed=42)
    del observation
    if counter.checkpoint() != {"env_step_count": 0, "process_action_count": 0}:
        raise RuntimeError("P0_C_RESET_CONSUMED_ACTION")
    baseline = p0a._read_only_initialization_baseline(env, counter=counter)
    latch = GripperHysteresisLatch(initial_intent=AbstractGripperIntent.OPEN)
    previous = PreviousAcceptedPolicyAction()
    previous.reset()
    zero_action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.5))
    zero_packet, zero_tensor, zero_derivation = p0a._build_authoritative_packet(
        high_level=zero_action, batch_size=env.num_envs, device=env.device, latch=latch
    )
    deferred = DeferredFull8DActionPacketPort(
        batch_size=int(env.num_envs), device=env.device, binding_id="g2_p0_c_current_full8d_deferred_env_step_v1"
    )
    ledger = P0B._PacketLedger(env=env, p0a=p0a, counter=counter, deferred_port=deferred)
    setup = P0B._source_defined_setup_with_ledger(
        env=env, p0a=p0a, ledger=ledger, zero_packet=zero_packet, zero_tensor=zero_tensor
    )
    binding = _runtime_binding(env)
    p1_receipt = load_p1_collection_runtime_semantic_contract(
        p1_path.read_text(encoding="utf-8"),
        data_semantics=binding.data_semantics,
        p0_runtime_observation_binding=binding,
    )
    initial_observation, initial_capture = _fresh_capture(
        env=env, p0a=p0a, provider=provider, ledger=ledger, latch=latch,
        previous=previous, binding=binding, hidden_reset=True, label="INITIAL",
    )

    before = provider.capture_snapshot()
    accepted_action = HighLevelPolicyAction.from_sequence((0.05, 0.0, 0.0, 0.5))
    packet, packet_tensor, derivation = p0a._build_authoritative_packet(
        high_level=accepted_action, batch_size=env.num_envs, device=env.device, latch=latch
    )
    semantic_record = ledger.submit(
        packet=packet,
        authoritative_packet_tensor=packet_tensor,
        label="P0_C_ACCEPTED_PREVIOUS_ACTION_PROBE",
        packet_role="semantic",
        semantic_command="ACCEPTED_PLUS_X",
        expected_gripper_intent="OPEN",
    )
    latch.commit(packet.gripper_intent)
    previous.record_accepted(accepted_action)
    after_direct = provider.capture_snapshot()
    accepted_observation, accepted_capture = _fresh_capture(
        env=env, p0a=p0a, provider=provider, ledger=ledger, latch=latch,
        previous=previous, binding=binding, hidden_reset=False, label="ACCEPTED",
        # Preserve the established P0-B response window.  This changes only
        # when the already-submitted target is observed; it does not submit a
        # second semantic command or alter the command magnitude/tolerance.
        minimum_hold_steps=20,
    )
    after_holds = provider.capture_snapshot()
    arm_scale = tuple(float(value) for value in env.action_manager.get_term("arm_action").cfg.scale)
    response = P0B._phase_response(
        p0a=p0a,
        before=before,
        after_direct=after_direct,
        after_holds=after_holds,
        command=SimpleNamespace(high_level_4d=accepted_action.values, motion_axis=0),
        arm_scale=arm_scale,
    )

    # Exercise the real reset boundary and source-defined reset-open lifecycle.
    env.reset(seed=42)
    previous.reset()
    latch.reset()
    post_reset_setup = P0B._source_defined_setup_with_ledger(
        env=env, p0a=p0a, ledger=ledger, zero_packet=zero_packet, zero_tensor=zero_tensor
    )
    reset_observation, reset_capture = _fresh_capture(
        env=env, p0a=p0a, provider=provider, ledger=ledger, latch=latch,
        previous=previous, binding=binding, hidden_reset=True, label="POST_RESET",
    )
    del initial_observation, accepted_observation
    ledger_payload = ledger.payload()
    packet_validation = _packet_validation(ledger_payload)
    termination = ledger_payload["termination_receipt"]
    runtime_safe = bool(
        termination.get("complete") is True
        and termination.get("terminated_packet_count") == 0
        and termination.get("truncated_packet_count") == 0
        and termination.get("forbidden_collision_observed") is False
        and termination.get("fixed_torso_drift_observed") is False
    )
    previous_pass = all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
        for actual, expected in zip(accepted_capture["previous_policy_action"], accepted_action.values)
    )
    reset_pass = bool(
        reset_capture["hidden_reset"] is True
        and reset_capture["previous_policy_action"] == [0.0, 0.0, 0.0, 0.0]
        and reset_capture["current_gripper_state"] == 0.0
    )
    camera_pass = bool(
        initial_capture["sequence_id"] < accepted_capture["sequence_id"]
        and all(row["frame_age_s"] <= binding.wrist_camera.maximum_frame_age_s for row in (initial_capture, accepted_capture, reset_capture))
        and all(row["depth_valid_fraction"] > 0.0 for row in (initial_capture, accepted_capture, reset_capture))
    )
    functional_pass = bool(
        packet_validation["status"] == "PASS"
        and response["motion_pass"]
        and response["orientation_pass"]
        and semantic_record.get("observed_gripper_intent") == "OPEN"
        and previous_pass
        and reset_pass
        and camera_pass
        and runtime_safe
    )
    return {
        "schema": SCHEMA,
        "phase": "P0_C",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CURRENT_FULL8D_RGBD_PREVIOUS_ACTION_RESET_INTEGRATION_NO_P1_COLLECTION_NO_M1_NO_M2_NO_TRAINING",
        "prerequisite": dict(prerequisite),
        "initialization": {"read_only_baseline": baseline, "setup": setup, "zero_derivation": zero_derivation},
        "post_reset_setup": post_reset_setup,
        "runtime_binding": binding.payload(),
        "runtime_binding_fingerprint": binding.fingerprint(),
        "p1_semantic_receipt": p1_receipt.payload(),
        "actor_input_fields": [
            "right_wrist_rgb", "right_wrist_depth_m", "right_wrist_depth_valid",
            "ee_pose_robot_root_xyzw", "arm_joint_position_rad", "arm_joint_velocity_rad_s",
            "current_gripper_state", "previous_policy_action",
        ],
        "privileged_or_gt_actor_input": False,
        "learned_policy_loaded": False,
        "captures": {"initial": initial_capture, "accepted": accepted_capture, "post_reset": reset_capture},
        "semantic_action": {
            "high_level_4d": list(accepted_action.values),
            "full_8d_packet": list(packet.values),
            "derivation": derivation,
            "ledger_record": semantic_record,
            "response": response,
            "previous_action_lag1_pass": previous_pass,
        },
        "packet_ledger": ledger_payload,
        "current_p0_c_packet_validation": packet_validation,
        "camera_contract_pass": camera_pass,
        "reset_contract_pass": reset_pass,
        "runtime_safety_receipt_pass": runtime_safe,
        "preclose_receipt": {
            "initialization_failure": False,
            "physics_stepping_failure": False,
            "env_step_failure": False,
            "action_submission_failure": False,
            "unexpected_collision": bool(termination.get("forbidden_collision_observed")),
            "unexpected_termination": not runtime_safe,
            "new_preclose_stack_signature": False,
            "instrumentation_restore_failure": False,
            "termination_receipt_complete": True,
        },
        "P0_C_FUNCTIONAL": "PASS" if functional_pass else "FAIL",
        "P0_C_PROCESS": "UNCLASSIFIED_UNTIL_PARENT_OBSERVES_EXIT",
        "P0_FUNCTIONAL": "PASS_AND_FROZEN" if functional_pass else "FAIL_CLOSED",
        "M1": "NOT_EXECUTED",
        "M1_5": "NOT_EXECUTED",
        "M2": "NOT_EXECUTED",
        "TRAINING": "NOT_AUTHORIZED",
        "functional_artifact_persisted_before_shutdown": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prerequisite-audit", type=Path, required=True)
    parser.add_argument("--p1-contract", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new immutable directory")
    output.mkdir(parents=True, exist_ok=False)
    audit = args.prerequisite_audit.resolve()
    p1_contract = args.p1_contract.resolve()
    prerequisite = _validate_prerequisites(audit, p1_contract)
    freeze = _source_manifest(audit, p1_contract)
    _atomic_json(output / "P0_C_PREREQUISITE_VALIDATION.json", prerequisite)
    _atomic_json(output / "P0_C_SOURCE_FREEZE.json", freeze)
    if prerequisite["status"] != "PASS" or freeze["SOURCE_FREEZE"] != "PASS":
        return 2

    def emit_atexit() -> None:
        print("ATEXIT_ENTER", flush=True)
        print("ATEXIT_RETURN", flush=True)

    atexit.register(emit_atexit)
    print("SOURCE_FREEZE_OK", flush=True)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True, fast_shutdown=False)
    app = launcher.app
    print("APP_CREATED", flush=True)
    env = None
    counter = None
    result: dict[str, Any]
    cleanup: dict[str, Any] = {"env_created": False, "env_close": "NOT_APPLICABLE"}
    try:
        print("RUNTIME_BEGIN", flush=True)
        if not _same_manifest(freeze, _source_manifest(audit, p1_contract)):
            raise RuntimeError("P0_C_SOURCE_FREEZE_CHANGED_AFTER_APP_CREATE")
        p0a, _ = P0B._load_current_p0_a_harness()
        runtime_authority_document = _strict_json(FROZEN_RUNTIME_AUTHORITY)
        runtime_identity = P0B._runtime_identity(
            app, runtime_authority_document["p0_a_prerequisite"]
        )
        if runtime_identity.get("runtime_identity_match_frozen_p0_a") is not True:
            raise RuntimeError("P0_C_RUNTIME_IDENTITY_MISMATCH")
        factory, factory_provenance = p0a._load_existing_p0_factory()
        expected_factory = (ROOT / "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py").resolve()
        if factory_provenance.get("factory_source") != str(expected_factory) or factory_provenance.get("factory_callable") != "_make_env":
            raise RuntimeError("P0_C_FACTORY_PROVENANCE_MISMATCH")
        env, disabled = factory("P0_C")
        cleanup["env_created"] = True
        p0a._assert_existing_p0_runtime_surface(env)
        counter = p0a._LifecycleCounter(env)
        counter.install()
        provider, observation_source, _ = p0a._runtime_observation_provider(env, freeze)
        result = _run_functional(
            env=env, p0a=p0a, provider=provider, counter=counter,
            prerequisite=prerequisite, p1_path=p1_contract,
        )
        result["source_freeze"] = freeze
        result["runtime_identity"] = runtime_identity
        result["factory_identity"] = factory_provenance
        result["disabled_task_terms"] = disabled
        result["observation_capture_calls"] = observation_source.captures
    except BaseException as error:
        result = {
            "schema": SCHEMA,
            "phase": "P0_C",
            "source_freeze": freeze,
            "prerequisite": prerequisite,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "preclose_receipt": {
                "initialization_failure": True,
                "physics_stepping_failure": False,
                "env_step_failure": False,
                "action_submission_failure": False,
                "unexpected_collision": False,
                "unexpected_termination": False,
                "new_preclose_stack_signature": True,
                "instrumentation_restore_failure": False,
                "termination_receipt_complete": False,
            },
            "P0_C_FUNCTIONAL": "FAIL",
            "P0_FUNCTIONAL": "FAIL_CLOSED",
            "TRAINING": "NOT_AUTHORIZED",
            "functional_artifact_persisted_before_shutdown": True,
        }
    finally:
        print("RUNTIME_END", flush=True)
        if counter is not None:
            try:
                counter.restore()
            except BaseException as error:
                result["P0_C_FUNCTIONAL"] = "FAIL"
                result["P0_FUNCTIONAL"] = "FAIL_CLOSED"
                result["preclose_receipt"]["instrumentation_restore_failure"] = True
                result["preclose_receipt"]["instrumentation_restore_exception"] = f"{type(error).__name__}:{error}"
        _atomic_json(output / "P0_C_FUNCTIONAL_PRE_CLOSE.json", result)
        print("REPORT_SAVED", flush=True)
        if env is not None:
            try:
                env.close()
                cleanup["env_close"] = "PASS"
            except BaseException as error:
                cleanup["env_close"] = f"FAIL:{type(error).__name__}:{error}"
        env = None
        counter = None
        cleanup["gc_collect_while_app_alive"] = int(gc.collect())
        _atomic_json(output / "P0_C_CLEANUP_PRE_APP_CLOSE.json", cleanup)
        print("APP_CLOSE_BEGIN", flush=True)
        app.close()
        print("APP_CLOSED", flush=True)
        print("BEFORE_MAIN_RETURN", flush=True)
        print("PYTHON_MAIN_RETURNED", flush=True)
    return 0 if result.get("P0_C_FUNCTIONAL") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
