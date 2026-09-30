#!/usr/bin/env python3
"""Bounded BC runtime preflight for the qualified 4-D G2 policy task.

The default invocation is a CPU-only dry run.  ``--execute-live`` is an
explicitly separate operation for a later supervisor: it creates one
``ManagerBasedRLEnv``, settles the right OmniPicker OPEN through the ordinary
8-D ``env.step`` ingress, resets policy-local temporal state, strictly loads
one authoritative 4-D BC checkpoint, then runs a bounded closed loop.  Every
policy packet is consumed by exactly one ``env.step``.  Setup/evaluation rows
are never exposed to a replay buffer and no learner is constructed here.

The functional artifact is fsync'd before environment/application shutdown.
Isaac/Kit's known post-finalization SIGSEGV remains a parent-process verdict;
this child never maps it to success.
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
PRODUCTION_USD = SOURCE / "geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
FROZEN_TRAJECTORY = ROOT / "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
M2_CANDIDATE = (
    ROOT
    / "artifacts/g2_m2_combined_hardstop_requalification_20260920/"
    "model_correction_candidates/E1_wide_joint4_joint3/robot_fix.usda"
)
M2_REPORT = (
    ROOT
    / "artifacts/g2_m2_combined_hardstop_requalification_20260920/"
    "live/E1_m2_fresh_seed42/hierarchy-report.json"
)
P0A_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0a_one_shot_integration.py"
P0B_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0b_one_shot_integration.py"
P0C_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0c_current_integration.py"

EXPECTED_PRODUCTION_SHA256 = (
    "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
)
EXPECTED_TRAJECTORY_SHA256 = (
    "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"
)
EXPECTED_CANDIDATE_SHA256 = (
    "d4505622f039da261df992ac7d8d1e1e24eae066058c9b52c29fc418f851a468"
)
EXPECTED_M2_REPORT_SHA256 = (
    "0d87e3258d55d972e106b32117e3ecfab2675b82471a362ac76ecc5cc947c8f3"
)

SCHEMA = "g2_policy_4d_training_runtime_preflight_v3"
SETUP_OPEN_POLICY_ACTION_4D = (0.0, 0.0, 0.0, 0.0)
EXPECTED_FULL_8D_OPEN_PACKET = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
DEFAULT_BOUNDED_CLOSED_LOOP_STEPS = 16
MAXIMUM_BOUNDED_CLOSED_LOOP_STEPS = 64
MAXIMUM_FULL_EPISODE_EVALUATION_STEPS = 640


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


def _load_exact_module(source: Path, *, prefix: str) -> Any:
    """Load one prescribed file and reject provenance drift or fallback."""

    resolved = source.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"G2_4D_PREFLIGHT_MODULE_MISSING:{resolved}")
    before = _sha256(resolved)
    name = f"_{prefix}_{before[:16]}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"G2_4D_PREFLIGHT_MODULE_SPEC_FAILED:{resolved}")
    if Path(spec.origin or "").resolve() != resolved:
        raise RuntimeError(f"G2_4D_PREFLIGHT_MODULE_ORIGIN_MISMATCH:{resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if Path(module.__file__).resolve() != resolved or _sha256(resolved) != before:
        raise RuntimeError(f"G2_4D_PREFLIGHT_MODULE_PROVENANCE_CHANGED:{resolved}")
    return module


def _source_manifest() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        P0A_SOURCE,
        P0B_SOURCE,
        P0C_SOURCE,
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/training_env_factory.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/runtime_asset_binding.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/bc_policy.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/bc_training.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/keyboard_bc_dataset.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/dataset.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/action_interface.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/production_metric_adapter.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/observation.py",
        SOURCE / "geniesim/rl/isaaclab/g2_policy_branch/p0_runtime_contract.py",
        SOURCE / "geniesim/rl/isaaclab/g2_gripper_reset_contract.py",
        SOURCE / "geniesim/rl/isaaclab/g2_keyboard_pose.py",
        SOURCE / "geniesim/rl/isaaclab/g2_lift_task_mdp.py",
        PRODUCTION_USD,
        FROZEN_TRAJECTORY,
        M2_CANDIDATE,
        M2_REPORT,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    hashes = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
        if path.is_file()
    }
    checks = {
        "all_sources_present": not missing,
        "production_usd_exact": hashes.get(str(PRODUCTION_USD.relative_to(ROOT)))
        == EXPECTED_PRODUCTION_SHA256,
        "frozen_trajectory_exact": hashes.get(str(FROZEN_TRAJECTORY.relative_to(ROOT)))
        == EXPECTED_TRAJECTORY_SHA256,
        "m2_candidate_exact": hashes.get(str(M2_CANDIDATE.relative_to(ROOT)))
        == EXPECTED_CANDIDATE_SHA256,
        "m2_report_exact": hashes.get(str(M2_REPORT.relative_to(ROOT)))
        == EXPECTED_M2_REPORT_SHA256,
    }
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "g2_policy_4d_training_preflight_source_freeze_v2",
        "SOURCE_FREEZE": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing": missing,
        "source_sha256": hashes,
        "manifest_sha256": fingerprint,
        "production_usd_sha256": hashes.get(str(PRODUCTION_USD.relative_to(ROOT))),
        "frozen_trajectory_sha256": hashes.get(str(FROZEN_TRAJECTORY.relative_to(ROOT))),
        "m2_candidate_sha256": hashes.get(str(M2_CANDIDATE.relative_to(ROOT))),
        "m2_report_sha256": hashes.get(str(M2_REPORT.relative_to(ROOT))),
    }


def _same_manifest(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(left == right and left.get("SOURCE_FREEZE") == "PASS")


def _packet_values_are_exact(values: Any) -> bool:
    try:
        actual = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return actual == EXPECTED_FULL_8D_OPEN_PACKET


def _tensor_plain(tensor: Any) -> Any:
    value = tensor.detach().to("cpu")
    return value.item() if value.numel() == 1 else value.tolist()


def _consume_once(
    *,
    env: Any,
    counter: Any,
    deferred_port: Any,
    packet: Any,
    label: str,
) -> tuple[tuple[Any, Any, Any, Any, Any], dict[str, Any]]:
    """Consume one immutable packet through the sole canonical ingress."""

    deferred = deferred_port.defer_normalized_action_manager_packet(packet)
    claimed = deferred_port.claim_for_env_step(deferred.packet_id)
    before = counter.checkpoint()
    counter.context = label
    try:
        result = env.step(claimed.as_env_step_tensor())
    except BaseException as error:
        hardstop = getattr(error, "runtime_hardstop_receipt", None)
        if hardstop is not None:
            # A RuntimeHardstop is the sole structured exception whose
            # consumption is knowable: the detector captured the real sample
            # after process_action and k scene updates, then stopped physics.
            interval = counter.interval(before)
            if (
                interval["env_step_calls"] != 1
                or interval["action_manager_process_action_calls"] != 1
            ):
                deferred_port.mark_env_step_consumption_unknown(deferred.packet_id)
                raise RuntimeError(
                    f"G2_4D_PARTIAL_NOT_SINGLE_CONSUMPTION:{label}:{interval}"
                ) from error
            partial = deferred_port.acknowledge_partial_termination(
                deferred.packet_id,
                consumed_substeps=int(hardstop.consumed_substeps),
                nominal_substeps=int(hardstop.nominal_substeps),
                termination_reason=str(hardstop.hardstop_reason),
            )
            error.canonical_consumption_receipt = {
                "label": label,
                "packet_id": deferred.packet_id,
                "content_fingerprint": deferred.content_fingerprint,
                "packet_values": [float(value) for value in packet.values],
                "router_process_action_count": 0,
                "env_step_count": interval["env_step_calls"],
                "action_manager_process_action_count": interval[
                    "action_manager_process_action_calls"
                ],
                "acknowledgement_state": partial.state.value,
                "action_consumption": partial.action_consumption,
                "consumed_substeps": partial.consumed_substeps,
                "nominal_substeps": partial.nominal_substeps,
                "termination_reason": partial.termination_reason,
                "single_consumption": True,
            }
            raise
        deferred_port.mark_env_step_consumption_unknown(deferred.packet_id)
        raise
    acknowledgement = deferred_port.acknowledge_env_step_success(deferred.packet_id)
    interval = counter.interval(before)
    if (
        interval["env_step_calls"] != 1
        or interval["action_manager_process_action_calls"] != 1
    ):
        raise RuntimeError(f"G2_4D_PREFLIGHT_NOT_SINGLE_CONSUMPTION:{label}:{interval}")
    env_capture = interval["env_step_packets"][0]
    process_capture = interval["process_action_packets"][0]
    if (
        env_capture["sha256"] != process_capture["sha256"]
        or env_capture["values"] != process_capture["values"]
    ):
        raise RuntimeError(f"G2_4D_PREFLIGHT_PACKET_IDENTITY_MISMATCH:{label}")
    receipt = {
        "label": label,
        "packet_id": deferred.packet_id,
        "content_fingerprint": deferred.content_fingerprint,
        "packet_values": [float(value) for value in packet.values],
        "packet_sha256": env_capture["sha256"],
        "router_process_action_count": 0,
        "env_step_count": interval["env_step_calls"],
        "action_manager_process_action_count": interval[
            "action_manager_process_action_calls"
        ],
        "acknowledgement_state": acknowledgement.state.value,
        "action_consumption": acknowledgement.action_consumption,
        "consumed_substeps": acknowledgement.consumed_substeps,
        "nominal_substeps": acknowledgement.nominal_substeps,
        "termination_reason": acknowledgement.termination_reason,
        "single_consumption": True,
    }
    return result, receipt


def _active_termination_names(env: Any, terminated: Any, truncated: Any) -> list[str]:
    if not bool(terminated.reshape(-1)[0].item()) and not bool(
        truncated.reshape(-1)[0].item()
    ):
        return []
    names: list[str] = []
    for name in env.termination_manager.active_terms:
        if bool(env.termination_manager.get_term(name).reshape(-1)[0].item()):
            names.append(str(name))
    return names


def _reward_metrics(env: Any, reward: Any) -> dict[str, Any]:
    rates = {
        str(name): float(values[0])
        for name, values in env.reward_manager.get_active_iterable_terms(0)
    }
    contributions = {
        name: value * float(env.step_dt) for name, value in rates.items()
    }
    return {
        "total": float(reward.reshape(-1)[0].item()),
        "term_rates": rates,
        "term_step_contributions": contributions,
        "term_contribution_sum": float(sum(contributions.values())),
        "step_dt_s": float(env.step_dt),
    }


def _task_metrics(env: Any, task_mdp: Any) -> dict[str, Any]:
    import torch

    inner, outer, bilateral, slip, stable = task_mdp.contact_grasp_telemetry(env)
    inner_contact, outer_contact, contact, boolean_bilateral = (
        task_mdp.contact_boolean_telemetry(env)
    )
    if not torch.equal(bilateral, boolean_bilateral):
        raise RuntimeError("G2_BOOLEAN_CONTACT_TELEMETRY_MISMATCH")
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    cube = env.scene["object"].data.root_pos_w
    distance = torch.linalg.vector_norm(cube - ee, dim=-1)
    return {
        "inner_contact_force_n": float(inner.reshape(-1)[0].item()),
        "outer_contact_force_n": float(outer.reshape(-1)[0].item()),
        "inner_contact": bool(inner_contact.reshape(-1)[0].item()),
        "outer_contact": bool(outer_contact.reshape(-1)[0].item()),
        "contact": bool(contact.reshape(-1)[0].item()),
        "bilateral_contact": bool(bilateral.reshape(-1)[0].item()),
        "slip_speed_m_s": float(slip.reshape(-1)[0].item()),
        "stable_now": bool(stable.reshape(-1)[0].item()),
        "ever_bilateral_contact": bool(
            env._g2_task_ever_bilateral_contact.reshape(-1)[0].item()
        ),
        "ever_stable_grasp": bool(
            env._g2_task_ever_stable_grasp.reshape(-1)[0].item()
        ),
        "ever_lifted_while_stable": bool(
            env._g2_task_ever_lifted_while_stable.reshape(-1)[0].item()
        ),
        "ee_cube_distance_m": float(distance.reshape(-1)[0].item()),
    }


def _load_bc_checkpoint_strict(
    checkpoint: Path,
    *,
    expected_file_sha256: str,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load only the source-owned Wrist-only 4-D BC checkpoint schema."""

    import torch
    from geniesim.rl.isaaclab.g2_policy_branch.bc_policy import (
        HighLevelBCPolicy,
        PolicyBCConfig,
        PolicyCheckpointMetadata,
        tensor_state_sha256,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.bc_training import (
        BC_TRAINING_CHECKPOINT_SCHEMA,
        policy_bc_config_payload,
    )

    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"G2_4D_PREFLIGHT_BC_CHECKPOINT_MISSING:{path}")
    observed_file_sha256 = _sha256(path)
    if observed_file_sha256 != expected_file_sha256:
        raise RuntimeError(
            "G2_4D_PREFLIGHT_BC_CHECKPOINT_FILE_HASH_MISMATCH:"
            f"{observed_file_sha256}!={expected_file_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_CHECKPOINT_NOT_MAPPING")
    required = {
        "schema",
        "metadata",
        "model_config",
        "model_semantic_fingerprint",
        "policy_data_semantic_fingerprint",
        "dataset_fingerprint",
        "dataset_audit",
        "episode_split",
        "selected_gripper_balance",
        "gripper_onset_repair",
        "training_config",
        "training_asset_path",
        "training_asset_sha256",
        "m2_validated_asset_sha256",
        "vision_checkpoint",
        "frozen_wrist_encoder_sha256",
        "model_state_dict",
        "optimizer_state_dict",
        "warmup_optimizer_state_dict",
        "history",
        "completed_epochs",
    }
    if set(payload) != required:
        raise RuntimeError(
            "G2_4D_PREFLIGHT_BC_CHECKPOINT_FIELDS_MISMATCH:"
            + repr(sorted(set(payload) ^ required))
        )
    if payload["schema"] != BC_TRAINING_CHECKPOINT_SCHEMA:
        raise RuntimeError("G2_4D_PREFLIGHT_BC_CHECKPOINT_SCHEMA_MISMATCH")
    config = PolicyBCConfig()
    if payload["model_config"] != policy_bc_config_payload(config):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_MODEL_CONFIG_NOT_CANONICAL_4D")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_METADATA_NOT_MAPPING")
    PolicyCheckpointMetadata(**metadata).assert_compatible(config)
    if (
        payload["model_semantic_fingerprint"] != config.semantic_fingerprint()
        or payload["policy_data_semantic_fingerprint"]
        != config.data_semantics.fingerprint()
    ):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_SEMANTIC_FINGERPRINT_MISMATCH")
    if (
        payload["training_asset_sha256"] != EXPECTED_CANDIDATE_SHA256
        or payload["m2_validated_asset_sha256"] != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_M2_ASSET_BINDING_MISMATCH")
    if int(payload["completed_epochs"]) <= 0:
        raise RuntimeError("G2_4D_PREFLIGHT_BC_CHECKPOINT_HAS_NO_COMPLETED_EPOCH")
    onset_repair = payload["gripper_onset_repair"]
    if not isinstance(onset_repair, Mapping):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_ONSET_REPAIR_NOT_MAPPING")
    onset_required = {
        "enabled": True,
        "loss": "MACRO_BCE_OPEN_CLOSE_ONSET_HOLD_CLOSED",
        "sampler": "FIXED_STRIDE_PLUS_ONE_ALIGNED_WINDOW_PER_OBSERVED_ONSET",
        "onset_label_authority": "SOURCE_PREVIOUS_CANONICAL_GRIPPER_COMMAND",
        "onset_window_hidden_initialization": (
            "EPISODE_START_PREFIX_ROLL_IN_NO_GRAD_THEN_BOUNDED_BPTT"
        ),
        "decision_threshold": 0.5,
        "runtime_close_threshold": config.gripper_hysteresis.close_threshold,
    }
    if any(onset_repair.get(key) != value for key, value in onset_required.items()):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_ONSET_REPAIR_AUTHORITY_MISMATCH")
    if int(onset_repair.get("heldout_onset_window_count", 0)) <= 0:
        raise RuntimeError("G2_4D_PREFLIGHT_BC_ONSET_REPAIR_HAS_NO_HELDOUT_ONSET")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping):
        raise RuntimeError("G2_4D_PREFLIGHT_BC_MODEL_STATE_NOT_MAPPING")
    model = HighLevelBCPolicy(config)
    model.load_state_dict(state, strict=True)
    model_hash = tensor_state_sha256(model)
    encoder_hash = tensor_state_sha256(model.wrist_encoder)
    if encoder_hash != payload["frozen_wrist_encoder_sha256"]:
        raise RuntimeError("G2_4D_PREFLIGHT_BC_WRIST_ENCODER_HASH_MISMATCH")
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"G2_4D_PREFLIGHT_BC_NONFINITE_PARAMETER:{name}")
    model.to(device)
    model.eval()
    receipt = {
        "path": str(path),
        "file_sha256": observed_file_sha256,
        "schema": payload["schema"],
        "metadata": dict(metadata),
        "model_config": dict(payload["model_config"]),
        "model_semantic_fingerprint": payload["model_semantic_fingerprint"],
        "policy_data_semantic_fingerprint": payload[
            "policy_data_semantic_fingerprint"
        ],
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "training_asset_sha256": payload["training_asset_sha256"],
        "model_state_sha256": model_hash,
        "wrist_encoder_state_sha256": encoder_hash,
        "completed_epochs": int(payload["completed_epochs"]),
        "gripper_onset_repair": dict(onset_repair),
        "strict_state_dict_load": True,
        "legacy_7d_checkpoint_rejected_by_metadata": True,
    }
    return model, receipt


def _settle_open_before_policy(
    *,
    env: Any,
    p0a: Any,
    counter: Any,
    deferred_port: Any,
    open_packet: Any,
) -> dict[str, Any]:
    """Reproduce keyboard OPEN settling without producing replay rows."""

    from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
        FullArticulationFDSafetyAudit,
        G2_RIGHT_GRIPPER_MASTER,
        G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS,
        G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS,
        G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD,
        G2_RIGHT_MASTER_OPEN_TARGET_RAD,
        capture_measured_right_hand_cache,
        install_measured_right_hand_cache,
    )

    robot = env.scene["robot"]
    if int(env.num_envs) != 1:
        raise RuntimeError("G2_4D_PREFLIGHT_OPEN_SETTLE_REQUIRES_ONE_ENV")
    master_index = robot.joint_names.index(G2_RIGHT_GRIPPER_MASTER)
    audit = FullArticulationFDSafetyAudit.start(
        robot.data.joint_pos,
        robot.joint_names,
        dt_s=float(env.step_dt),
    )
    receipts: list[dict[str, Any]] = []
    measured = float("nan")
    settled_steps = 0
    for settled_steps in range(
        1, G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS + 1
    ):
        outputs, receipt = _consume_once(
            env=env,
            counter=counter,
            deferred_port=deferred_port,
            packet=open_packet,
            label=f"TRAINING_RESET_OPEN_SETTLE_{settled_steps}",
        )
        _, _, terminated, truncated, _ = outputs
        active = _active_termination_names(env, terminated, truncated)
        if bool(terminated.reshape(-1)[0].item()) or bool(
            truncated.reshape(-1)[0].item()
        ):
            raise RuntimeError(
                f"G2_4D_PREFLIGHT_OPEN_SETTLE_TERMINATED:{settled_steps}:{active}"
            )
        audit.observe(robot.data.joint_pos, step=settled_steps)
        measured = float(robot.data.joint_pos[0, master_index].item())
        receipts.append(receipt)
        if (
            settled_steps >= G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
            and abs(measured - G2_RIGHT_MASTER_OPEN_TARGET_RAD)
            <= G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD
            and audit.settled_for_cache
        ):
            break
    error = abs(measured - G2_RIGHT_MASTER_OPEN_TARGET_RAD)
    audit_result = audit.result()
    if (
        settled_steps < G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
        or error > G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD
        or not bool(audit_result["pass"])
    ):
        raise RuntimeError(
            "G2_4D_PREFLIGHT_OPEN_SETTLE_FAILED:"
            + json.dumps(
                {
                    "steps": settled_steps,
                    "measured_rad": measured,
                    "error_rad": error,
                    "audit": audit_result,
                },
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    measured_cache = capture_measured_right_hand_cache(
        robot.data.joint_pos,
        robot.joint_names,
        safety_attestation=audit_result,
    )
    install_measured_right_hand_cache(
        robot.data.default_joint_pos,
        measured_cache,
        robot.joint_names,
    )
    gripper = p0a._gripper_command_telemetry(
        env.action_manager.get_term("gripper_action")
    )
    if gripper["close_command_active"]:
        raise RuntimeError("G2_4D_PREFLIGHT_OPEN_SETTLE_ENDED_CLOSED")
    return {
        "minimum_policy_steps": G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS,
        "maximum_policy_steps": G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS,
        "actual_policy_steps": settled_steps,
        "target_rad": G2_RIGHT_MASTER_OPEN_TARGET_RAD,
        "measured_rad": measured,
        "absolute_error_rad": error,
        "tolerance_rad": G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD,
        "full_46dof_fd_safety": audit_result,
        "measured_full_hand_cache_installed_for_future_resets": True,
        "setup_transition_count": len(receipts),
        "setup_replay_insertion_count": 0,
        "all_setup_packets_single_consumption": all(
            receipt["single_consumption"] for receipt in receipts
        ),
        "first_setup_packet_sha256": receipts[0]["packet_sha256"],
        "last_setup_packet_sha256": receipts[-1]["packet_sha256"],
        "gripper_telemetry": gripper,
    }


def _run_live(
    *,
    output: Path,
    seed: int,
    freeze: Mapping[str, Any],
    bc_checkpoint: Path,
    bc_checkpoint_sha256: str,
    closed_loop_steps: int,
    full_episode_evaluation: bool = False,
) -> int:
    """Run the bounded preflight.  A supervisor must opt in explicitly."""

    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
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
        if not _same_manifest(freeze, _source_manifest()):
            raise RuntimeError("G2_4D_PREFLIGHT_SOURCE_FREEZE_CHANGED_AFTER_APP_CREATE")

        from isaaclab.envs import ManagerBasedRLEnv
        from geniesim.rl.isaaclab import g2_lift_task_mdp
        from geniesim.rl.isaaclab.g2_keyboard_pose import (
            G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
            G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
            AbstractGripperIntent,
            GripperHysteresisLatch,
            HighLevelPolicyAction,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
            PreviousAcceptedPolicyAction,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.production_metric_adapter import (
            DeferredFull8DActionPacketPort,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.training_env_factory import (
            attest_g2_policy_4d_training_task_geometry_runtime,
            bind_g2_policy_4d_training_task_geometry_runtime,
            make_g2_policy_4d_training_env_cfg,
        )

        p0c = _load_exact_module(P0C_SOURCE, prefix="g2_4d_preflight_p0c")
        p0a, p0a_provenance = p0c.P0B._load_current_p0_a_harness()
        cfg, asset_receipt, task_contract = make_g2_policy_4d_training_env_cfg(
            num_envs=1
        )
        cfg.observations.policy.enable_corruption = False
        cfg.commands.object_pose.debug_vis = False
        env = ManagerBasedRLEnv(cfg=cfg)
        cleanup["env_created"] = True
        task_geometry_binding = bind_g2_policy_4d_training_task_geometry_runtime(env)
        print("ENV_READY", flush=True)

        # RewardManager executes during OPEN settling.  Calibration therefore
        # precedes reset and every explicit env.step, exactly as in keyboard
        # collection.  GT is not added to PolicyObservation.
        g2_lift_task_mdp.configure_grasp_reward_target(
            env,
            G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M,
            calibration_rows=G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
        )
        print("REWARD_CALIBRATION_READY", flush=True)
        _, reset_info = env.reset(seed=seed)
        del reset_info
        task_geometry_readback = attest_g2_policy_4d_training_task_geometry_runtime(
            env
        )
        print("TASK_GEOMETRY_READY", flush=True)

        counter = p0a._LifecycleCounter(env)
        counter.install()
        provider, observation_source, controller_binding = (
            p0a._runtime_observation_provider(env, freeze)
        )
        observation_binding = p0c._runtime_binding(env)
        latch = GripperHysteresisLatch(
            initial_intent=AbstractGripperIntent.OPEN
        )
        previous = PreviousAcceptedPolicyAction()
        previous.reset()
        open_action = HighLevelPolicyAction.from_sequence(
            SETUP_OPEN_POLICY_ACTION_4D
        )
        open_packet, _, open_derivation = p0a._build_authoritative_packet(
            high_level=open_action,
            batch_size=env.num_envs,
            device=env.device,
            latch=latch,
        )
        if not _packet_values_are_exact(open_packet.values):
            raise RuntimeError(
                f"G2_4D_PREFLIGHT_OPEN_PACKET_MISMATCH:{open_packet.values}"
            )
        deferred_port = DeferredFull8DActionPacketPort(
            batch_size=1,
            device=env.device,
            binding_id="g2_policy_4d_training_runtime_preflight_v2",
        )
        print("OPEN_SETTLE_BEGIN", flush=True)
        settle = _settle_open_before_policy(
            env=env,
            p0a=p0a,
            counter=counter,
            deferred_port=deferred_port,
            open_packet=open_packet,
        )
        print("OPEN_SETTLE_DONE", flush=True)

        # Setup is outside replay.  Reset all policy-local temporal authority
        # and reward-history state before the first evaluated transition.
        latch.reset()
        previous.reset()
        all_env_ids = __import__("torch").arange(
            env.num_envs, device=env.device, dtype=__import__("torch").long
        )
        g2_lift_task_mdp.reset_task_progress_state(env, all_env_ids)
        model, checkpoint_receipt = _load_bc_checkpoint_strict(
            bc_checkpoint,
            expected_file_sha256=bc_checkpoint_sha256,
            device=env.device,
        )
        model.config.data_semantics.assert_compatible(
            observation_binding.data_semantics
        )
        print("BC_CHECKPOINT_LOADED", flush=True)

        import torch
        from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
            FullArticulationFDSafetyAudit,
        )

        hidden = None
        policy_safety_audit = FullArticulationFDSafetyAudit.start(
            env.scene["robot"].data.joint_pos,
            env.scene["robot"].joint_names,
            dt_s=float(env.step_dt),
        )
        terminal_post_reset_sample_excluded = False
        policy_counter_before = counter.checkpoint()
        transition_records: list[dict[str, Any]] = []
        policy_failed = False
        first_capture: dict[str, Any] | None = None
        last_capture: dict[str, Any] | None = None
        previous_output_tensor = None
        for step in range(closed_loop_steps):
            observation, capture = p0c._capture_observation(
                env=env,
                p0a=p0a,
                provider=provider,
                previous=previous,
                hidden_reset=step == 0,
                binding=observation_binding,
            )
            if first_capture is None:
                first_capture = capture
                print("POLICY_OBSERVATION_READY", flush=True)
            last_capture = capture
            with torch.inference_mode():
                network_output = model(observation, initial_hidden=hidden)
            hidden = network_output.final_hidden.detach()
            output_tensor = network_output.policy_action[:, -1]
            if tuple(output_tensor.shape) != (1, 4) or not bool(
                torch.isfinite(output_tensor).all()
            ):
                raise RuntimeError("G2_4D_PREFLIGHT_BC_OUTPUT_INVALID")
            action_values = tuple(float(value) for value in output_tensor[0].tolist())
            action = HighLevelPolicyAction.from_sequence(action_values)
            action_l2 = float(torch.linalg.vector_norm(output_tensor[0]).item())
            action_step_delta_l2 = (
                0.0
                if previous_output_tensor is None
                else float(
                    torch.linalg.vector_norm(
                        output_tensor[0] - previous_output_tensor
                    ).item()
                )
            )
            action_step_delta_max_abs = (
                0.0
                if previous_output_tensor is None
                else float(
                    torch.max(
                        torch.abs(output_tensor[0] - previous_output_tensor)
                    ).item()
                )
            )
            previous_output_tensor = output_tensor[0].detach().clone()
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
                raise RuntimeError("G2_4D_PREFLIGHT_RPY_OR_ELBOW_NOT_ZERO")
            outputs, consumption = _consume_once(
                env=env,
                counter=counter,
                deferred_port=deferred_port,
                packet=packet,
                label=f"BC_CLOSED_LOOP_STEP_{step}",
            )
            _, reward, terminated, truncated, info = outputs
            del info
            latch.commit(packet.gripper_intent)
            previous.record_accepted(action)
            termination_names = _active_termination_names(
                env, terminated, truncated
            )
            reward_metrics = _reward_metrics(env, reward)
            task_metrics = _task_metrics(env, g2_lift_task_mdp)
            gripper = p0a._gripper_command_telemetry(
                env.action_manager.get_term("gripper_action")
            )
            terminal = bool(
                terminated.reshape(-1)[0].item()
                or truncated.reshape(-1)[0].item()
            )
            successful_terminal = bool(
                terminal
                and termination_names
                and set(termination_names) == {"object_reached_goal"}
            )
            # ManagerBasedRLEnv auto-resets before returning a terminal step.
            # Do not join that discontinuity to the preceding FD interval.
            if terminal:
                terminal_post_reset_sample_excluded = True
            else:
                policy_safety_audit.observe(
                    env.scene["robot"].data.joint_pos, step=step + 1
                )
            unexpected_terminal = terminal and not (
                full_episode_evaluation and successful_terminal
            )
            row_failed = bool(
                not consumption["single_consumption"]
                or unexpected_terminal
                or not math.isfinite(reward_metrics["total"])
                or not math.isfinite(task_metrics["ee_cube_distance_m"])
            )
            transition_records.append(
                {
                    "step": step,
                    "observation": capture,
                    "model_output_4d": list(action.values),
                    "model_output_finite": all(
                        math.isfinite(value) for value in action.values
                    ),
                    "normalized_bounds_pass": bool(
                        all(abs(value) <= 1.0 for value in action.values[:3])
                        and 0.0 <= action.values[3] <= 1.0
                    ),
                    "action_l2": action_l2,
                    "action_step_delta_l2": action_step_delta_l2,
                    "action_step_delta_max_abs_component": action_step_delta_max_abs,
                    "hidden_l2_norm": float(torch.linalg.vector_norm(hidden).item()),
                    "full_8d_packet": list(packet.values),
                    "zero_rpy_and_elbow": True,
                    "derivation": derivation,
                    "consumption": consumption,
                    "reward": reward_metrics,
                    "termination": {
                        "terminated": bool(terminated.reshape(-1)[0].item()),
                        "truncated": bool(truncated.reshape(-1)[0].item()),
                        "active_terms": termination_names,
                        "successful_terminal": successful_terminal,
                    },
                    "task": task_metrics,
                    "gripper": gripper,
                    "row_failed": row_failed,
                }
            )
            if row_failed:
                policy_failed = True
                break
            if terminal:
                break
        policy_interval = counter.interval(policy_counter_before)
        completed_steps = len(transition_records)
        one_step_per_packet = bool(
            policy_interval["env_step_calls"] == completed_steps
            and policy_interval["action_manager_process_action_calls"]
            == completed_steps
            and all(
                row["consumption"]["single_consumption"]
                for row in transition_records
            )
        )
        rewards = [row["reward"]["total"] for row in transition_records]
        distances = [row["task"]["ee_cube_distance_m"] for row in transition_records]
        action_l2_values = [row["action_l2"] for row in transition_records]
        action_step_delta_l2_values = [
            row["action_step_delta_l2"] for row in transition_records[1:]
        ]
        action_step_delta_max_values = [
            row["action_step_delta_max_abs_component"]
            for row in transition_records[1:]
        ]
        normalized_ee_components = [
            abs(value)
            for row in transition_records
            for value in row["model_output_4d"][:3]
        ]
        gripper_probabilities = [
            row["model_output_4d"][3] for row in transition_records
        ]
        closed_loop_metrics = {
            "requested_steps": closed_loop_steps,
            "completed_steps": completed_steps,
            "one_env_step_per_packet": one_step_per_packet,
            "reward_sum": float(sum(rewards)),
            "reward_mean": float(sum(rewards) / len(rewards)),
            "model_output_finite_all": all(
                row["model_output_finite"] for row in transition_records
            ),
            "normalized_bounds_pass_all": all(
                row["normalized_bounds_pass"] for row in transition_records
            ),
            "action_l2_mean": float(sum(action_l2_values) / len(action_l2_values)),
            "action_l2_max": float(max(action_l2_values)),
            "maximum_action_step_delta_l2": float(
                max(action_step_delta_l2_values, default=0.0)
            ),
            "maximum_action_step_delta_abs_component": float(
                max(action_step_delta_max_values, default=0.0)
            ),
            "maximum_abs_normalized_ee_component": float(
                max(normalized_ee_components)
            ),
            "gripper_probability_minimum": float(min(gripper_probabilities)),
            "gripper_probability_maximum": float(max(gripper_probabilities)),
            "initial_ee_cube_distance_m": float(distances[0]),
            "minimum_ee_cube_distance_m": float(min(distances)),
            "final_ee_cube_distance_m": float(distances[-1]),
            "ee_cube_distance_progress_m": float(distances[0] - distances[-1]),
            "contact_observed": any(
                row["task"]["ever_bilateral_contact"] for row in transition_records
            ),
            "stable_observed": any(
                row["task"]["ever_stable_grasp"] for row in transition_records
            ),
            "lift_observed": any(
                row["task"]["ever_lifted_while_stable"]
                for row in transition_records
            ),
            "gripper_close_observed": any(
                row["gripper"]["close_command_active"]
                for row in transition_records
            ),
            "termination_observed": any(
                row["termination"]["terminated"]
                or row["termination"]["truncated"]
                for row in transition_records
            ),
            "forbidden_collision_observed": any(
                "forbidden_collision" in row["termination"]["active_terms"]
                for row in transition_records
            ),
        }
        policy_safety_result = policy_safety_audit.result()
        successful_terminal_observed = any(
            row["termination"]["successful_terminal"]
            for row in transition_records
        )
        base_functional_pass = bool(
            (completed_steps == closed_loop_steps or successful_terminal_observed)
            and one_step_per_packet
            and not policy_failed
            and first_capture is not None
            and first_capture["hidden_reset"] is True
            and first_capture["previous_policy_action"]
            == [0.0, 0.0, 0.0, 0.0]
            and closed_loop_metrics["model_output_finite_all"]
            and closed_loop_metrics["normalized_bounds_pass_all"]
            and not closed_loop_metrics["forbidden_collision_observed"]
            and bool(policy_safety_result["pass"])
        )
        outcome_gate_pass = bool(
            closed_loop_metrics["gripper_close_observed"]
            and closed_loop_metrics["contact_observed"]
            and closed_loop_metrics["stable_observed"]
            and closed_loop_metrics["lift_observed"]
        )
        functional_pass = bool(
            base_functional_pass
            and (not full_episode_evaluation or outcome_gate_pass)
        )
        print("BOUNDED_CLOSED_LOOP_DONE", flush=True)
        result = {
            "schema": SCHEMA,
            "task": "AGENT_4D_TRAINING_RUNTIME_PREFLIGHT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "ONE_ENV_OPEN_SETTLE_THEN_FULL_EPISODE_BC_OUTCOME_EVALUATION_NO_REPLAY_NO_UPDATES"
                if full_episode_evaluation
                else "ONE_ENV_OPEN_SETTLE_THEN_STRICT_BC_BOUNDED_CLOSED_LOOP_NO_REPLAY_NO_UPDATES"
            ),
            "status": "PASS" if functional_pass else "FAIL",
            "source_freeze": freeze,
            "asset_binding": asset_receipt.as_dict(),
            "task_contract": task_contract,
            "task_geometry_binding": task_geometry_binding,
            "task_geometry_readback_after_reset_before_first_step": (
                task_geometry_readback
            ),
            "p0_helper_provenance": p0a_provenance,
            "controller_binding": controller_binding,
            "reward_calibration": {
                "cube_minus_ee_root_m": list(
                    G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M
                ),
                "calibration_rows": G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
                "configured_before_first_env_step": True,
            },
            "open_settle": settle,
            "policy_temporal_reset_after_settle": {
                "gripper_latch_initial": "OPEN",
                "gripper_latch_final": latch.intent.value,
                "previous_action_before_transition": first_capture[
                    "previous_policy_action"
                ],
                "recurrent_hidden_reset_mask": first_capture["hidden_reset"],
                "task_reward_progress_state_reset": True,
            },
            "bc_checkpoint": checkpoint_receipt,
            "observation_contract": {
                "binding": observation_binding.payload(),
                "binding_fingerprint": observation_binding.fingerprint(),
                "first": first_capture,
                "last": last_capture,
                "capture_calls": observation_source.captures,
                "privileged_or_gt_actor_input": False,
            },
            "action_contract": {
                "public_action_schema": "g2_ee_xyz_gripper_v1",
                "internal_action_schema": [
                    "dx", "dy", "dz", "rx", "ry", "rz", "elbow", "gripper"
                ],
                "all_packets_zero_rpy_and_elbow": all(
                    row["zero_rpy_and_elbow"] for row in transition_records
                ),
                "open_settle_derivation": open_derivation,
                "policy_interval": policy_interval,
            },
            "closed_loop_metrics": closed_loop_metrics,
            "policy_full_46dof_fd_safety": policy_safety_result,
            "terminal_post_auto_reset_fd_sample_excluded": (
                terminal_post_reset_sample_excluded
            ),
            "outcome_gate": {
                "required": full_episode_evaluation,
                "requires": ["gripper_close", "bilateral_contact", "stable", "lift"],
                "pass": outcome_gate_pass,
            },
            "transition_records": transition_records,
            "replay": {
                "setup_rows_inserted": 0,
                "evaluated_rows_inserted": 0,
                "replay_object_constructed": False,
            },
            "learner": {
                "bc_policy_loaded_for_inference": True,
                "optimizer_loaded": False,
                "bc_updates": 0,
                "sac_updates": 0,
                "legacy_7d_sac_used": False,
            },
            "TRAINING_RUNTIME_PREFLIGHT": (
                "PASS" if functional_pass else "FAIL_CLOSED"
            ),
            "BC_CLOSED_LOOP_EVALUATION": (
                "PASS" if full_episode_evaluation and functional_pass
                else "FAIL_CLOSED" if full_episode_evaluation
                else "NOT_REQUESTED"
            ),
            "TRAINING_EXECUTED": False,
            "PROCESS_VERDICT": "UNCLASSIFIED_UNTIL_PARENT_OBSERVES_EXIT",
            "KNOWN_ISAAC_FINALIZATION_DEFECT": "OPEN_PARENT_MUST_RETAIN_ACTUAL_EXIT_-11_OR_139",
            "functional_artifact_persisted_before_shutdown": True,
        }
    except BaseException as error:
        result = {
            "schema": SCHEMA,
            "task": "AGENT_4D_TRAINING_RUNTIME_PREFLIGHT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "source_freeze": dict(freeze),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "TRAINING_RUNTIME_PREFLIGHT": "FAIL_CLOSED",
            "TRAINING_EXECUTED": False,
            "PROCESS_VERDICT": "UNCLASSIFIED_UNTIL_PARENT_OBSERVES_EXIT",
            "KNOWN_ISAAC_FINALIZATION_DEFECT": "OPEN_PARENT_MUST_RETAIN_ACTUAL_EXIT_-11_OR_139",
            "functional_artifact_persisted_before_shutdown": True,
        }
    finally:
        print("RUNTIME_END", flush=True)
        if counter is not None:
            try:
                counter.restore()
            except BaseException as error:
                result["status"] = "FAIL"
                result["TRAINING_RUNTIME_PREFLIGHT"] = "FAIL_CLOSED"
                result["counter_restore_error"] = f"{type(error).__name__}:{error}"
        if not _same_manifest(freeze, _source_manifest()):
            result["status"] = "FAIL"
            result["TRAINING_RUNTIME_PREFLIGHT"] = "FAIL_CLOSED"
            result["source_freeze_changed_before_report"] = True
        _atomic_json(output / "FUNCTIONAL_RESULT_PRE_CLOSE.json", result)
        print("REPORT_SAVED", flush=True)
        if env is not None:
            try:
                env.close()
                cleanup["env_close"] = "PASS"
            except BaseException as error:
                cleanup["env_close"] = f"FAIL:{type(error).__name__}:{error}"
        _atomic_json(output / "CLEANUP_PRE_APP_CLOSE.json", cleanup)
        print("APP_CLOSE_BEGIN", flush=True)
        app.close()
        print("APP_CLOSED", flush=True)
        print("BEFORE_MAIN_RETURN", flush=True)
        print("PYTHON_MAIN_RETURNED", flush=True)
    return 0 if result.get("TRAINING_RUNTIME_PREFLIGHT") == "PASS" else 2


def _dry_run_payload(
    freeze: Mapping[str, Any],
    seed: int,
    *,
    bc_checkpoint: Path | None,
    bc_checkpoint_sha256: str | None,
    closed_loop_steps: int,
    full_episode_evaluation: bool = False,
) -> dict[str, Any]:
    checkpoint_identity: dict[str, Any]
    if bc_checkpoint is None:
        checkpoint_identity = {
            "provided": False,
            "live_execution_ready": False,
            "reason": "BC_CHECKPOINT_REQUIRED_FOR_LIVE",
        }
    else:
        resolved = bc_checkpoint.expanduser().resolve()
        observed = _sha256(resolved) if resolved.is_file() else None
        checkpoint_identity = {
            "provided": True,
            "path": str(resolved),
            "exists": resolved.is_file(),
            "observed_sha256": observed,
            "expected_sha256": bc_checkpoint_sha256,
            "file_hash_match": bool(
                observed is not None
                and bc_checkpoint_sha256 is not None
                and observed == bc_checkpoint_sha256
            ),
        }
        checkpoint_identity["live_execution_ready"] = checkpoint_identity[
            "file_hash_match"
        ]
    return {
        "schema": SCHEMA,
        "task": "AGENT_4D_TRAINING_RUNTIME_PREFLIGHT",
        "mode": "DRY_RUN_NO_ISAAC_NO_ENV_NO_STEP",
        "status": "PASS" if freeze.get("SOURCE_FREEZE") == "PASS" else "FAIL",
        "seed": seed,
        "source_freeze": dict(freeze),
        "bc_checkpoint_identity": checkpoint_identity,
        "plan": {
            "num_envs": 1,
            "setup_open_action_4d": list(SETUP_OPEN_POLICY_ACTION_4D),
            "setup_open_internal_action_8d": list(EXPECTED_FULL_8D_OPEN_PACKET),
            "evaluated_action_owner": "STRICTLY_LOADED_HIGH_LEVEL_BC_POLICY",
            "reward_calibration_before_first_step": True,
            "open_settle_minimum_policy_steps": 60,
            "open_settle_maximum_policy_steps": 120,
            "setup_replay_insertion_count": 0,
            "policy_temporal_state_reset_after_settle": True,
            "bounded_closed_loop_steps": closed_loop_steps,
            "full_episode_evaluation": full_episode_evaluation,
            "outcome_gate_requires": (
                ["gripper_close", "bilateral_contact", "stable", "lift"]
                if full_episode_evaluation
                else []
            ),
            "legacy_7d_sac": "NOT_USED",
            "training": "NOT_EXECUTED",
        },
        "TRAINING_RUNTIME_PREFLIGHT": "NOT_EXECUTED_DRY_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--bc-checkpoint-sha256")
    parser.add_argument(
        "--closed-loop-steps",
        type=int,
        default=DEFAULT_BOUNDED_CLOSED_LOOP_STEPS,
    )
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="explicitly run the one-transition Isaac preflight; default is dry",
    )
    parser.add_argument(
        "--full-episode-evaluation",
        action="store_true",
        help=(
            "run a BC-only outcome gate up to 640 steps; requires close, "
            "bilateral contact, stable grasp and lift"
        ),
    )
    args = parser.parse_args()
    maximum_steps = (
        MAXIMUM_FULL_EPISODE_EVALUATION_STEPS
        if args.full_episode_evaluation
        else MAXIMUM_BOUNDED_CLOSED_LOOP_STEPS
    )
    if not 1 <= args.closed_loop_steps <= maximum_steps:
        parser.error(
            f"--closed-loop-steps must be in [1,{maximum_steps}]"
        )
    if args.bc_checkpoint_sha256 is not None and (
        len(args.bc_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.bc_checkpoint_sha256)
    ):
        parser.error("--bc-checkpoint-sha256 must be 64 lowercase hex characters")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new immutable directory")
    output.mkdir(parents=True, exist_ok=False)
    freeze = _source_manifest()
    _atomic_json(output / "SOURCE_FREEZE.json", freeze)
    if freeze["SOURCE_FREEZE"] != "PASS":
        return 2
    print("SOURCE_FREEZE_OK", flush=True)
    if not args.execute_live:
        _atomic_json(
            output / "DRY_RUN_RESULT.json",
            _dry_run_payload(
                freeze,
                args.seed,
                bc_checkpoint=args.bc_checkpoint,
                bc_checkpoint_sha256=args.bc_checkpoint_sha256,
                closed_loop_steps=args.closed_loop_steps,
                full_episode_evaluation=args.full_episode_evaluation,
            ),
        )
        print("DRY_RUN_COMPLETE_NO_ISAAC", flush=True)
        return 0
    if args.bc_checkpoint is None or args.bc_checkpoint_sha256 is None:
        _atomic_json(
            output / "LIVE_PREFLIGHT_REJECTED.json",
            {
                "schema": SCHEMA,
                "status": "FAIL_CLOSED",
                "reason": "EXACT_BC_CHECKPOINT_PATH_AND_SHA256_REQUIRED",
                "TRAINING_RUNTIME_PREFLIGHT": "NOT_EXECUTED",
            },
        )
        return 2
    checkpoint = args.bc_checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != args.bc_checkpoint_sha256:
        _atomic_json(
            output / "LIVE_PREFLIGHT_REJECTED.json",
            {
                "schema": SCHEMA,
                "status": "FAIL_CLOSED",
                "reason": "BC_CHECKPOINT_MISSING_OR_HASH_MISMATCH",
                "checkpoint": str(checkpoint),
                "expected_sha256": args.bc_checkpoint_sha256,
                "observed_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
                "TRAINING_RUNTIME_PREFLIGHT": "NOT_EXECUTED",
            },
        )
        return 2

    def emit_atexit() -> None:
        print("ATEXIT_ENTER", flush=True)
        print("ATEXIT_RETURN", flush=True)

    atexit.register(emit_atexit)
    return _run_live(
        output=output,
        seed=args.seed,
        freeze=freeze,
        bc_checkpoint=checkpoint,
        bc_checkpoint_sha256=args.bc_checkpoint_sha256,
        closed_loop_steps=args.closed_loop_steps,
        full_episode_evaluation=args.full_episode_evaluation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
