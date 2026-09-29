#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Launch the independent Stage-1A 1/10-env vector runtime.

This is a separate launcher on purpose: the historical diagnostic runner and
its scalar direct-pregrasp initialization remain untouched.  A 10-env launch
therefore cannot accidentally resume the 6,910-transition scalar run or use
the scalar deferred packet's batch-repeat behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
P0C_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0c_current_integration.py"
P0A_SOURCE = ROOT / "scripts/diagnostics/run_g2_p0a_one_shot_integration.py"
DIAGNOSTIC_SOURCE = ROOT / "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py"
CANDIDATE_A_ASSET = ROOT / "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
CANDIDATE_A_SHA256 = "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
GRU_CHECKPOINT = ROOT / "artifacts/g2_keyboard_v3_recovery_20260924_v1/GRU_CAUSAL_CLOSE_EDGE_GPU_3K_WANDB_V1/BEST_HUMAN_GRASP_GRU_BC.pt"
GRU_SHA256 = ""  # resolved from the diagnostic authority below
FAR_BC_CHECKPOINT = ROOT / "artifacts/g2_curobo_grasp_bc/CONTACT_FREE_BC.pt"
RESIDUAL_ACTOR_CHECKPOINT = ROOT / "artifacts/g2_stage1a_residual_actor/RESIDUAL_ACTOR.pt"
GOVERNOR_FIX_RELATIVE_PATH = "source/geniesim/rl/isaaclab/g2_gripper_reset_contract.py"
GOVERNOR_FIX_PREVIOUS_SHA256 = "0edb05f7fbaadbf2519cb6b8deff42ba49094e220cb467a08dd55399127400ae"
# Active reset-lifecycle authority pins two evidence-scoped corrections: the
# 1600-step OPEN observation budget for a legitimate post-CLOSE four-bar
# recovery, and the 4-µrad numeric landing allowance required by r5's
# stationary 0.66--3.90-µrad PhysX solver residual.  The existing 0.0002-rad
# stop band, 0.005-rad/s settle proof, 0.8-rad/s cap, joint limits, controller,
# and mechanics remain unchanged.  This is not blanket acceptance of arbitrary
# historical drift.
GOVERNOR_FIX_CURRENT_SHA256 = "18fbd24cac18f3327ed8863f21ab9828d4806f5bb12090e217c9584b0a4b21f7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_key(path: Path) -> str:
    """Return a stable manifest key for workspace and immutable external input.

    Boundary-collection plans live beneath the user-selected output root so
    they cannot consume the nearly-full workspace filesystem.  They are still
    source-freeze authorities and must be hashed, but are not children of
    ``ROOT``.  Record that distinction explicitly rather than dropping the
    plan or requiring an unsafe copy into the repository.
    """

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"EXTERNAL:{resolved}"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_LOAD_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vector_source_freeze(
    diagnostic: Any,
    *,
    collection_source_catalog: Path | None = None,
    boundary_paired_plan: Path | None = None,
    frozen_student_advisory_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Freeze the new vector authority without rewriting the historical freeze.

    The historical Keyboard-V3 manifest predates the independent vector
    runtime and consequently cannot attest files added for this task.  Keep
    that historical manifest read-only, require its immutable Phase-6 inputs
    and promotion contracts to remain valid, then snapshot every source file
    that is authoritative for this new path.  The exact snapshot is compared
    again after the smoke/training run.
    """

    historical = diagnostic._source_freeze(keyboard_v3_branch=True)
    historical_mismatches = {
        relative: dict(record)
        for relative, record in historical["frozen"].items()
        if not bool(record["match"])
    }
    controlled_governor_fix_delta = (
        set(historical_mismatches) == {GOVERNOR_FIX_RELATIVE_PATH}
        and historical_mismatches[GOVERNOR_FIX_RELATIVE_PATH]["expected"]
        == GOVERNOR_FIX_PREVIOUS_SHA256
        and historical_mismatches[GOVERNOR_FIX_RELATIVE_PATH]["actual"]
        == GOVERNOR_FIX_CURRENT_SHA256
    )
    historical_inputs_ok = (
        not historical_mismatches or controlled_governor_fix_delta
    )
    p0_delta = __import__("json").loads(diagnostic.P0_DELTA.read_text(encoding="utf-8"))
    handoff = __import__("json").loads(diagnostic.HANDOFF.read_text(encoding="utf-8"))
    historical_contract_ok = (
        p0_delta["verdict"]["PLANNER_ONLY_LIVE_SMOKE_AUTHORIZED_NEXT"] == "YES"
        and handoff["verdict"]["CANONICAL_4D_HANDOFF"] == "PASS"
    )
    authority_paths = (
        Path(__file__).resolve(),
        ROOT / "source/geniesim/rl/sac/stage1a_vector_contract.py",
        ROOT / "source/geniesim/rl/sac/stage1a_vector_initial_states.py",
        ROOT / "source/geniesim/rl/sac/stage1a_vector_runtime.py",
        ROOT / "source/geniesim/rl/sac/stage1a_vector_telemetry.py",
        ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py",
        ROOT / "source/geniesim/rl/sac/stage1a_reset_open_restore.py",
        ROOT / "source/geniesim/rl/sac/stage1a_grasp_evaluation_contract.py",
        ROOT / "source/geniesim/rl/sac/stage1a_close_readiness_contract.py",
        ROOT / "source/geniesim/rl/sac/stage1a_boundary_paired_collection.py",
        ROOT / "source/geniesim/rl/sac/stage1a_signed_readiness_margin.py",
        ROOT / "source/geniesim/rl/sac/stage1a_fsm_sequence_dataset.py",
        ROOT / "source/geniesim/rl/sac/stage1a_close_readiness_advisory.py",
        ROOT / "source/geniesim/rl/sac/stage1a_frozen_spatial_student.py",
        ROOT / "source/geniesim/rl/sac/stage1a_close_readiness_training_contract.py",
        ROOT / "source/geniesim/rl/sac/stage1a_close_gate.py",
        ROOT / "source/geniesim/rl/sac/bc_residual_sac_contract.py",
        ROOT / "source/geniesim/rl/sac/stage1a_grasp_reward.py",
        ROOT / "source/geniesim/rl/sac/stage1a_isaac_telemetry_adapter.py",
        ROOT / "source/geniesim/rl/sac/privileged_geometry_oracle.py",
        ROOT / "source/geniesim/rl/sac/stage1a_real_sac_coordinator.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_lift_task_mdp.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_keyboard_pose.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_redundancy_action.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_lift_env_cfg.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_lift_rgbd_env_cfg.py",
        ROOT / GOVERNOR_FIX_RELATIVE_PATH,
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/training_env_factory.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/hybrid_grasp_runtime.py",
        ROOT / "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_direct_pregrasp_init.py",
        ROOT / "scripts/audit_g2_stage1a_fsm_sequence_dataset.py",
        DIAGNOSTIC_SOURCE,
        P0C_SOURCE,
        P0A_SOURCE,
    )
    if collection_source_catalog is not None:
        authority_paths = (*authority_paths, collection_source_catalog.resolve())
    if boundary_paired_plan is not None:
        authority_paths = (*authority_paths, boundary_paired_plan.resolve())
    if frozen_student_advisory_checkpoint is not None:
        authority_paths = (*authority_paths, frozen_student_advisory_checkpoint.resolve())
    missing = [str(path) for path in authority_paths if not path.is_file()]
    hashes = {
        _freeze_key(path): _sha256(path)
        for path in authority_paths
        if path.is_file()
    }
    manifest_sha256 = hashlib.sha256(
        repr(sorted(hashes.items())).encode("utf-8")
    ).hexdigest()
    return {
        "SOURCE_FREEZE": "PASS" if historical_inputs_ok and historical_contract_ok and not missing else "FAIL",
        "SOURCE_FREEZE_COMPLETE": historical_inputs_ok and historical_contract_ok and not missing,
        "freeze_profile": "STAGE1A_VECTOR_RUNTIME_ACTIVE_MASK_FIX_DELTA_V1",
        "manifest_sha256": manifest_sha256,
        "authority_hashes": hashes,
        "historical_immutable_inputs_match": historical_inputs_ok,
        "historical_frozen_mismatches": historical_mismatches,
        "controlled_governor_fix_delta": {
            "authorized": controlled_governor_fix_delta,
            "relative_path": GOVERNOR_FIX_RELATIVE_PATH,
            "previous_sha256": GOVERNOR_FIX_PREVIOUS_SHA256,
            "current_sha256": GOVERNOR_FIX_CURRENT_SHA256,
        },
        "historical_promotion_contracts_match": historical_contract_ok,
        "historical_keyboard_v3_branch_manifest_status": historical["SOURCE_FREEZE"],
        "historical_keyboard_v3_delta_not_rewritten": True,
        "missing_authority_paths": missing,
    }


def _write_launch_marker(path: Path, stage: str, **details: Any) -> None:
    """Persist a marker outside the run directory for startup attribution."""

    payload = {"schema": "g2_stage1a_vector_launcher_v1", "stage": stage, **details}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preflight_receipt_path(report: Path, *, kind: str) -> Path:
    """Return a startup receipt path without creating the run directory.

    ``run_stage1a_isaac_vector_smoke`` intentionally refuses an existing
    output directory.  The launcher must therefore keep its startup and
    physics-smoke receipts adjacent to the future run directory, never inside
    it.  Otherwise a successful pretraining smoke itself creates the path
    that the runtime correctly rejects as an overwrite attempt.
    """

    if kind not in {"launch", "physics_smoke"}:
        raise ValueError("VECTOR_PREFLIGHT_RECEIPT_KIND_INVALID")
    run_directory = report.parent
    return run_directory.parent / f"{run_directory.name}_{kind}.json"


def _tensor_tree_is_finite(value: Any, *, torch_module: Any) -> bool:
    """Validate a nested Isaac observation without coercing its schema."""

    if isinstance(value, torch_module.Tensor):
        return bool(torch_module.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(
            _tensor_tree_is_finite(item, torch_module=torch_module)
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return all(
            _tensor_tree_is_finite(item, torch_module=torch_module)
            for item in value
        )
    return True


def _tensor_tree_has_batch_size(value: Any, *, batch_size: int, torch_module: Any) -> bool:
    """Require every tensor observation leaf to remain clone-batched."""

    if isinstance(value, torch_module.Tensor):
        return bool(value.ndim >= 1 and int(value.shape[0]) == batch_size)
    if isinstance(value, Mapping):
        return all(
            _tensor_tree_has_batch_size(
                item, batch_size=batch_size, torch_module=torch_module
            )
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return all(
            _tensor_tree_has_batch_size(
                item, batch_size=batch_size, torch_module=torch_module
            )
            for item in value
        )
    return True


def _run_pretraining_physics_smoke(
    *, env: Any, report_path: Path, seed: int
) -> dict[str, Any]:
    """Exercise one canonical OPEN packet before the SAC coordinator exists.

    This is intentionally not a rollout and never constructs a replay row or
    learner update.  The following direct-pregrasp initialization in the
    vector runtime resets the selected robot/cube states again, so this first
    physics step cannot become training data or affect the 3K distribution.
    """

    import torch

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
    )
    from geniesim.rl.sac.stage1a_vector_contract import (
        ImmutablePerEnvActionPacket,
        PerEnvCanonicalAction,
    )

    rows = tuple(
        PerEnvCanonicalAction(
            env_id=env_id,
            final_action_4d_metric_root_m=(0.0, 0.0, 0.0, 0.0),
            gripper_intent=AbstractGripperIntent.OPEN,
        )
        for env_id in range(int(env.num_envs))
    )
    packet = ImmutablePerEnvActionPacket(
        rows=rows,
        binding_id="stage1a-vector-pretraining-physics-smoke",
        # Packet ports reserve positive sequence identifiers; this isolated
        # preflight has no port but must still satisfy the immutable packet
        # contract before it reaches the canonical ActionManager path.
        sequence_id=1,
        device=env.device,
    )
    action = packet.as_env_step_tensor()
    if tuple(action.shape) != (int(env.num_envs), 8):
        raise RuntimeError("PRETRAINING_SMOKE_CANONICAL_ACTION_SHAPE_INVALID")
    if not bool(torch.isfinite(action).all().item()):
        raise RuntimeError("PRETRAINING_SMOKE_ACTION_NONFINITE")
    try:
        outputs = env.step(action)
        if not isinstance(outputs, tuple) or len(outputs) != 5:
            raise RuntimeError("PRETRAINING_SMOKE_ENV_STEP_OUTPUT_INVALID")
        observation, reward, terminated, truncated, _extras = outputs
        finite_observation = _tensor_tree_is_finite(
            observation, torch_module=torch
        )
        observation_batched = _tensor_tree_has_batch_size(
            observation, batch_size=int(env.num_envs), torch_module=torch
        )
        finite_reward = bool(torch.isfinite(reward).all().item())
        finite_terminated = bool(torch.isfinite(terminated.to(torch.float32)).all().item())
        finite_truncated = bool(torch.isfinite(truncated.to(torch.float32)).all().item())
        if not all(
            (
                finite_observation,
                observation_batched,
                finite_reward,
                finite_terminated,
                finite_truncated,
            )
        ):
            raise RuntimeError("PRETRAINING_SMOKE_NONFINITE_OR_UNBATCHED_RUNTIME")
        # Isolate the one physical smoke transition completely from the
        # subsequent direct-pregrasp 3K initialization.
        env.reset(seed=int(seed))
        payload = {
            "schema": "g2_stage1a_vector_pretraining_physics_smoke_v1",
            "PHYSICS_SMOKE": "PASS",
            "NUM_ENVS": int(env.num_envs),
            "CONTROL_HZ": 50,
            "PHYSICS_HZ": 500,
            "FIRST_PHYSICS_STEP_REACHED": True,
            "CANONICAL_OPEN_PACKET": True,
            "ACTION_FINITE": True,
            "OBSERVATION_FINITE": finite_observation,
            "OBSERVATION_BATCHED": observation_batched,
            "REWARD_FINITE": finite_reward,
            "TERMINATION_FINITE": finite_terminated and finite_truncated,
            "REPLAY_ROWS_WRITTEN": 0,
            "SAC_UPDATES": 0,
            "POST_SMOKE_RESET": "PASS",
        }
    except BaseException as error:
        payload = {
            "schema": "g2_stage1a_vector_pretraining_physics_smoke_v1",
            "PHYSICS_SMOKE": "FAIL",
            "NUM_ENVS": int(env.num_envs),
            "REPLAY_ROWS_WRITTEN": 0,
            "SAC_UPDATES": 0,
            "exception_type": type(error).__name__,
            "exception": str(error),
        }
        _write_launch_marker(report_path, "PHYSICS_SMOKE_FAIL", **payload)
        raise
    _write_launch_marker(report_path, "PHYSICS_SMOKE_PASS", **payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--num-envs", type=int, choices=(1, 10, 25), required=True)
    parser.add_argument(
        "--accepted-transitions",
        type=int,
        choices=(100, 3000, 6000, 7500, 15000, 30000),
        required=True,
    )
    parser.add_argument(
        "--runtime-variant",
        choices=(
            "V3_BASELINE",
            "V3_1_POST_CLOSE_MICRO",
            "V3_1_LATERAL_GATE_OFF_PAIRED",
            "V3_1_LATERAL_OFF_HER_FORCE_15K",
            "V3_1_LATERAL_OFF_HER_FORCE_30K",
            "V3_CURRENT_HER_FORCE_FAIR_6K",
            "V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
            "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_15K",
            "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_FAIR_6K",
            "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K",
            "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K_25ENV",
            "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV",
            "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV",
            "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K",
            "V3_2_RELAX_LATERAL_STABILIZE",
            "V3_PRIVILEGED_GEOMETRY_TEACHER_HER_FORCE_3K",
            "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K",
        ),
        default="V3_BASELINE",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="offline")
    parser.add_argument("--wandb-project", default="geniesim-g2-stage1a-residual-sac")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="bounded 25-env/100-transition reset/runtime smoke with zero SAC updates",
    )
    parser.add_argument(
        "--close-readiness-training-contract",
        type=Path,
        help=(
            "immutable train-only normalizer / class-balanced student contract; "
            "required by the corrected privileged-distillation variant"
        ),
    )
    parser.add_argument(
        "--close-readiness-initialization",
        type=Path,
        help=(
            "offline-validated normalized student-head initialization paired "
            "with --close-readiness-training-contract"
        ),
    )
    parser.add_argument(
        "--frozen-student-advisory-checkpoint",
        type=Path,
        help=(
            "hash-pinned Conditional-CORAL Wrist RGB-D student used only for "
            "HIGH_CONFIDENCE_ADVISORY_FSM_DEFER telemetry"
        ),
    )
    parser.add_argument(
        "--frozen-student-advisory-sha256",
        help="required full SHA-256 for --frozen-student-advisory-checkpoint",
    )
    parser.add_argument(
        "--preclose-collection-only",
        action="store_true",
        help=(
            "collect canonical pre-CLOSE teacher rows without SAC or student-head "
            "optimizer updates; requires the bounded 10-env/3K distillation variant"
        ),
    )
    parser.add_argument(
        "--preclose-source-catalog",
        type=Path,
        help=(
            "optional hash-frozen Candidate-A V2 multi-source catalog; valid only "
            "for --preclose-collection-only and never used by training/evaluation"
        ),
    )
    parser.add_argument(
        "--boundary-paired-plan",
        type=Path,
        help=(
            "hash-attested label-agnostic boundary probe plan; valid only for "
            "update-free pre-CLOSE collection with a frozen source catalog"
        ),
    )
    parser.add_argument(
        "--v6-wrist-diagnostic",
        action="store_true",
        help=(
            "enable diagnostic-only semantic/instance Wrist AOVs; requires a "
            "V6 boundary plan and remains excluded from student inputs"
        ),
    )
    args = parser.parse_args()
    if not args.execute_live:
        raise SystemExit("VECTOR_RUNTIME_REQUIRES_EXPLICIT_EXECUTE_LIVE")
    if args.accepted_transitions in (15000, 30000) and args.num_envs not in (10, 25):
        raise SystemExit("VECTOR_LONG_RUN_REQUIRES_NUM_ENVS_10_OR_25")
    if args.accepted_transitions in (15000, 30000) and (not args.wandb or args.wandb_mode != "online"):
        raise SystemExit("VECTOR_LONG_RUN_REQUIRES_ONLINE_WANDB")
    fair_6k_variants = {
        "V3_CURRENT_HER_FORCE_FAIR_6K",
        "V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K",
        "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_FAIR_6K",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K",
    }
    reset_fixed_7p5k_variants = {
        "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV",
    }
    reset_fixed_6k_variants = {
        "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV",
    }
    if args.preflight_only and args.runtime_variant not in reset_fixed_7p5k_variants:
        raise SystemExit("PREFLIGHT_ONLY_RESERVED_FOR_RESET_FIXED_25ENV_METHODS")
    if args.runtime_variant in fair_6k_variants and (
        args.num_envs != 10
        or args.accepted_transitions != 6000
        or not args.wandb
        or args.wandb_mode != "online"
    ):
        raise SystemExit(
            "FAIR_6K_REQUIRES_NUM_ENVS_10_TARGET_6000_AND_ONLINE_WANDB"
        )
    if args.runtime_variant in reset_fixed_7p5k_variants and (
        args.num_envs != 25
        or not (
            (
                not args.preflight_only
                and args.accepted_transitions == 7500
                and args.wandb
                and args.wandb_mode == "online"
            )
            or (
                args.preflight_only
                and args.accepted_transitions == 100
                and not args.wandb
            )
        )
    ):
        raise SystemExit(
            "RESET_FIXED_FAIR_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB"
        )
    if args.runtime_variant in reset_fixed_6k_variants and (
        args.num_envs != 25
        or args.accepted_transitions != 6000
        or not args.wandb
        or args.wandb_mode != "online"
    ):
        raise SystemExit(
            "RESET_FIXED_FAIR_6K_REQUIRES_NUM_ENVS_25_TARGET_6000_AND_ONLINE_WANDB"
        )
    # Legacy advisory-specific receipt retained for artifact/test readers:
    # V31_LATERAL_OFF_FSM_ADVISORY_RESET_FIXED_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB
    if args.runtime_variant == "V3_1_POST_CLOSE_MICRO" and (
        args.num_envs != 10 or args.accepted_transitions not in (3000, 30000)
    ):
        raise SystemExit("V31_REQUIRES_NUM_ENVS_10_AND_TARGET_3000_OR_30000")
    if args.runtime_variant == "V3_1_LATERAL_GATE_OFF_PAIRED" and (
        args.num_envs != 10 or args.accepted_transitions != 3000
    ):
        raise SystemExit("V31_LATERAL_OFF_REQUIRES_NUM_ENVS_10_AND_TARGET_3000")
    if args.runtime_variant == "V3_1_LATERAL_OFF_HER_FORCE_15K" and (
        args.num_envs != 10 or args.accepted_transitions != 15000
    ):
        raise SystemExit(
            "V31_LATERAL_OFF_15K_REQUIRES_NUM_ENVS_10_AND_TARGET_15000"
        )
    if args.runtime_variant == "V3_1_LATERAL_OFF_HER_FORCE_30K" and (
        args.num_envs != 10 or args.accepted_transitions != 30000
    ):
        raise SystemExit(
            "V31_LATERAL_OFF_30K_REQUIRES_NUM_ENVS_10_AND_TARGET_30000"
        )
    if args.runtime_variant == "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_15K" and (
        args.num_envs != 25 or args.accepted_transitions != 15000
    ):
        raise SystemExit(
            "V31_LATERAL_OFF_FSM_SEQUENCE_REQUIRES_NUM_ENVS_25_AND_TARGET_15000"
        )
    advisory_variant = args.runtime_variant in (
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K_25ENV",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV",
        "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K",
    )
    if args.runtime_variant == "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K" and (
        args.num_envs != 10 or args.accepted_transitions != 15000
    ):
        raise SystemExit(
            "V31_LATERAL_OFF_FSM_ADVISORY_REQUIRES_NUM_ENVS_10_AND_TARGET_15000"
        )
    if args.runtime_variant == "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K_25ENV" and (
        args.num_envs != 25 or args.accepted_transitions != 15000
    ):
        raise SystemExit(
            "V31_LATERAL_OFF_FSM_ADVISORY_25ENV_REQUIRES_NUM_ENVS_25_AND_TARGET_15000"
        )
    if args.runtime_variant == "V3_2_RELAX_LATERAL_STABILIZE" and (
        args.num_envs != 10 or args.accepted_transitions != 3000
    ):
        raise SystemExit("V32_REQUIRES_NUM_ENVS_10_AND_TARGET_3000")
    if args.runtime_variant == "V3_PRIVILEGED_GEOMETRY_TEACHER_HER_FORCE_3K" and (
        args.num_envs != 10 or args.accepted_transitions != 3000
    ):
        raise SystemExit(
            "PRIVILEGED_GEOMETRY_TEACHER_REQUIRES_NUM_ENVS_10_AND_TARGET_3000"
        )
    if args.runtime_variant == "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K" and (
        args.num_envs != 10 or args.accepted_transitions != 3000
    ):
        raise SystemExit(
            "PRIVILEGED_DISTILLATION_REQUIRES_NUM_ENVS_10_AND_TARGET_3000"
        )
    distillation_variant = (
        args.runtime_variant
        == "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K"
    )
    if advisory_variant:
        if (
            args.frozen_student_advisory_checkpoint is None
            or not args.frozen_student_advisory_checkpoint.is_file()
            or not isinstance(args.frozen_student_advisory_sha256, str)
            or len(args.frozen_student_advisory_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.frozen_student_advisory_sha256.lower()
            )
        ):
            raise SystemExit(
                "FROZEN_STUDENT_ADVISORY_REQUIRES_EXISTING_CHECKPOINT_AND_SHA256"
            )
        if _sha256(args.frozen_student_advisory_checkpoint) != (
            args.frozen_student_advisory_sha256.lower()
        ):
            raise SystemExit("FROZEN_STUDENT_ADVISORY_CHECKPOINT_HASH_MISMATCH")
    elif (
        args.frozen_student_advisory_checkpoint is not None
        or args.frozen_student_advisory_sha256 is not None
    ):
        raise SystemExit("FROZEN_STUDENT_ADVISORY_ARGS_RESERVED_FOR_ADVISORY_VARIANT")
    if distillation_variant and not args.preclose_collection_only and (
        args.close_readiness_training_contract is None
        or args.close_readiness_initialization is None
        or not args.close_readiness_training_contract.is_file()
        or not args.close_readiness_initialization.is_file()
    ):
        raise SystemExit(
            "PRIVILEGED_DISTILLATION_REQUIRES_NORMALIZED_BALANCED_CONTRACT_AND_INIT"
        )
    if not distillation_variant and (
        args.close_readiness_training_contract is not None
        or args.close_readiness_initialization is not None
    ):
        raise SystemExit(
            "CLOSE_READINESS_CONTRACT_ARGS_RESERVED_FOR_DISTILLATION_VARIANT"
        )
    if args.preclose_collection_only and (
        args.close_readiness_training_contract is not None
        or args.close_readiness_initialization is not None
    ):
        raise SystemExit(
            "PRECLOSE_COLLECTION_FORBIDS_TRAINING_CONTRACT_AND_INITIALIZATION"
        )
    if args.preclose_source_catalog is not None and (
        not args.preclose_collection_only or not args.preclose_source_catalog.is_file()
    ):
        raise SystemExit(
            "PRECLOSE_SOURCE_CATALOG_REQUIRES_COLLECTION_ONLY_EXISTING_FILE"
        )
    if args.boundary_paired_plan is not None and (
        not args.preclose_collection_only
        or args.preclose_source_catalog is None
        or not args.boundary_paired_plan.is_file()
    ):
        raise SystemExit(
            "BOUNDARY_PAIRED_PLAN_REQUIRES_CATALOGGED_UPDATE_FREE_COLLECTION"
        )
    if distillation_variant and not args.preclose_collection_only:
        try:
            preflight_contract = json.loads(
                args.close_readiness_training_contract.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit("CLOSE_READINESS_CONTRACT_UNREADABLE") from error
        if not bool(
            preflight_contract.get("qualification", {}).get(
                "offline_contract_pass", False)
        ):
            raise SystemExit(
                "PRIVILEGED_DISTILLATION_OFFLINE_CONTRACT_NOT_QUALIFIED"
            )
    if args.preclose_collection_only and (
        args.runtime_variant != "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K"
        or args.num_envs != 10
        or args.accepted_transitions != 3000
        or args.wandb
    ):
        raise SystemExit(
            "PRECLOSE_COLLECTION_REQUIRES_10ENV_3K_TEACHER_ONLY_AND_NO_WANDB"
        )
    if args.v6_wrist_diagnostic and (
        not args.preclose_collection_only
        or args.boundary_paired_plan is None
        or args.preclose_source_catalog is None
    ):
        raise SystemExit("V6_DIAGNOSTIC_REQUIRES_UPDATE_FREE_BOUNDARY_COLLECTION")
    if args.output_dir.exists() or args.report.exists():
        raise SystemExit("VECTOR_OUTPUT_REFUSES_OVERWRITE")
    if not CANDIDATE_A_ASSET.is_file() or _sha256(CANDIDATE_A_ASSET) != CANDIDATE_A_SHA256:
        raise SystemExit("VECTOR_CANDIDATE_A_HASH_MISMATCH")

    launch_marker = _preflight_receipt_path(args.report, kind="launch")
    _write_launch_marker(
        launch_marker,
        "ARGS_VALIDATED",
        num_envs=args.num_envs,
        accepted_transitions=args.accepted_transitions,
        runtime_variant=args.runtime_variant,
        preclose_collection_only=bool(args.preclose_collection_only),
        preclose_source_catalog=(
            str(args.preclose_source_catalog.resolve())
            if args.preclose_source_catalog is not None
            else None
        ),
        boundary_paired_plan=(
            str(args.boundary_paired_plan.resolve())
            if args.boundary_paired_plan is not None
            else None
        ),
        v6_wrist_diagnostic=bool(args.v6_wrist_diagnostic),
        close_readiness_training_contract=(
            str(args.close_readiness_training_contract)
            if args.close_readiness_training_contract is not None
            else None
        ),
        close_readiness_initialization=(
            str(args.close_readiness_initialization)
            if args.close_readiness_initialization is not None
            else None
        ),
        frozen_student_advisory_checkpoint=(
            str(args.frozen_student_advisory_checkpoint.resolve())
            if args.frozen_student_advisory_checkpoint is not None
            else None
        ),
        frozen_student_advisory_sha256=args.frozen_student_advisory_sha256,
    )

    # Construct SimulationApp before importing any project module that can pull
    # USD/PXR symbols into the process.  Isaac Kit aborts if those native
    # modules are loaded first; this is particularly easy to trigger through
    # an environment config's asset imports.
    from isaaclab.app import AppLauncher
    _write_launch_marker(launch_marker, "PRE_APP_LAUNCH")
    launcher = AppLauncher(headless=True, enable_cameras=True, fast_shutdown=False)
    app = launcher.app
    _write_launch_marker(launch_marker, "POST_APP_LAUNCH")
    env = None
    counter = None
    try:
        _write_launch_marker(launch_marker, "PRE_ENV_IMPORT")
        from isaaclab.envs import ManagerBasedRLEnv
        from geniesim.rl.isaaclab.g2_policy_branch.training_env_factory import (
            bind_g2_policy_4d_training_task_geometry_runtime,
            make_g2_candidate_a_left_arm_down_v2_training_env_cfg,
        )
        from geniesim.rl.isaaclab import g2_lift_task_mdp
        from geniesim.rl.isaaclab.g2_keyboard_pose import (
            G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
            G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M,
        )
        from geniesim.rl.sac.stage1a_isaac_vector_smoke import run_stage1a_isaac_vector_smoke
        from geniesim.rl.sac.stage1a_boundary_paired_collection import (
            load_boundary_paired_plan,
        )

        close_readiness_training_contract = None
        close_readiness_initialization = None
        boundary_paired_plan = None
        if args.boundary_paired_plan is not None:
            boundary_paired_plan = load_boundary_paired_plan(
                args.boundary_paired_plan,
                source_catalog=args.preclose_source_catalog,
            )
            if args.v6_wrist_diagnostic != bool(
                boundary_paired_plan.get("v6_diagnostic_receipt", False)
            ):
                raise RuntimeError("V6_DIAGNOSTIC_PLAN_FLAG_MISMATCH")
        if distillation_variant and not args.preclose_collection_only:
            close_readiness_training_contract = json.loads(
                args.close_readiness_training_contract.read_text(encoding="utf-8")
            )
            import torch

            close_readiness_initialization = torch.load(
                args.close_readiness_initialization,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(close_readiness_initialization, Mapping):
                raise RuntimeError("CLOSE_READINESS_INITIALIZATION_PAYLOAD_INVALID")

        diagnostic = _load(DIAGNOSTIC_SOURCE, "g2_vector_diagnostic_authority")
        p0c = _load(P0C_SOURCE, "g2_vector_p0c_authority")
        p0a_module = _load(P0A_SOURCE, "g2_vector_p0a_counter")
        p0a, _p0a_provenance = p0c.P0B._load_current_p0_a_harness()
        cfg, asset_receipt, _task_contract = make_g2_candidate_a_left_arm_down_v2_training_env_cfg(num_envs=args.num_envs)
        if args.v6_wrist_diagnostic:
            # V6 consumes the existing right-wrist Camera prim and leaves its
            # mount/update period untouched.  The extra AOVs and the ephemeral
            # ``class:cube`` tag are diagnostic renderer metadata only; they
            # do not alter USD files, physics, mechanics, reward, or actor
            # observations.  The collection runtime fail-closes if either
            # renderer label cannot be resolved.
            wrist_cfg = cfg.scene.right_wrist_camera
            wrist_cfg.data_types = [
                "rgb",
                "distance_to_image_plane",
                "semantic_segmentation",
                "instance_segmentation_fast",
            ]
            wrist_cfg.renderer_cfg.colorize_semantic_segmentation = False
            wrist_cfg.renderer_cfg.colorize_instance_segmentation = False
            cfg.scene.object.spawn.semantic_tags = [("class", "cube")]
        if Path(cfg.scene.robot.spawn.usd_path).resolve() != CANDIDATE_A_ASSET.resolve():
            raise RuntimeError("VECTOR_FACTORY_CANDIDATE_A_BINDING_MISMATCH")
        _write_launch_marker(launch_marker, "PRE_ENV_CONSTRUCT")
        env = ManagerBasedRLEnv(cfg)
        _write_launch_marker(launch_marker, "POST_ENV_CONSTRUCT")
        # This is the existing source-owned task configuration needed by the
        # untouched environment reward manager.  It precedes every env.step
        # and is identical for one and ten clones.
        bind_g2_policy_4d_training_task_geometry_runtime(env)
        g2_lift_task_mdp.configure_grasp_reward_target(
            env,
            G2_KEYBOARD_REWARD_GRASP_CUBE_MINUS_EE_ROOT_M,
            calibration_rows=G2_KEYBOARD_REWARD_GRASP_CALIBRATION_ROWS,
        )
        env.reset(seed=int(args.seed))
        _write_launch_marker(launch_marker, "POST_ENV_RESET")
        # A successful constructor is insufficient evidence for a training
        # retry.  Require one canonical, replay-free physics transition with
        # finite 10-env tensors before the coordinator and W&B training run
        # are even constructed.
        physics_smoke_path = _preflight_receipt_path(args.report, kind="physics_smoke")
        _write_launch_marker(launch_marker, "PRE_PHYSICS_SMOKE")
        physics_smoke = _run_pretraining_physics_smoke(
            env=env,
            report_path=physics_smoke_path,
            seed=int(args.seed),
        )
        if physics_smoke.get("PHYSICS_SMOKE") != "PASS":
            raise RuntimeError("PRETRAINING_PHYSICS_SMOKE_FAILED")
        _write_launch_marker(
            launch_marker,
            "POST_PHYSICS_SMOKE",
            physics_smoke_report=str(physics_smoke_path),
        )
        counter = p0a_module._LifecycleCounter(env)
        counter.install()
        freeze = _vector_source_freeze(
            diagnostic,
            collection_source_catalog=args.preclose_source_catalog,
            boundary_paired_plan=args.boundary_paired_plan,
            frozen_student_advisory_checkpoint=(
                args.frozen_student_advisory_checkpoint
                if advisory_variant
                else None
            ),
        )
        if freeze.get("SOURCE_FREEZE") != "PASS":
            raise RuntimeError("VECTOR_SOURCE_FREEZE_FAILED")
        _write_launch_marker(launch_marker, "PRE_VECTOR_RUNTIME")
        result = run_stage1a_isaac_vector_smoke(
            env=env, p0a=p0a, counter=counter, task_mdp=g2_lift_task_mdp,
            source_freeze_before=freeze,
            source_freeze_provider=lambda: _vector_source_freeze(
                diagnostic,
                collection_source_catalog=args.preclose_source_catalog,
                boundary_paired_plan=args.boundary_paired_plan,
                frozen_student_advisory_checkpoint=(
                    args.frozen_student_advisory_checkpoint
                    if advisory_variant
                    else None
                ),
            ),
            selected_asset_path=CANDIDATE_A_ASSET, selected_asset_sha256=CANDIDATE_A_SHA256,
            accepted_transition_target=args.accepted_transitions,
            bc_checkpoint=diagnostic.STAGE1A_FROZEN_GRU_CHECKPOINT,
            bc_checkpoint_sha256=diagnostic.STAGE1A_FROZEN_GRU_SHA256,
            far_reach_checkpoint=diagnostic.STAGE1A_FAR_REACH_BC_CHECKPOINT,
            far_reach_checkpoint_sha256=diagnostic.STAGE1A_FAR_REACH_BC_SHA256,
            residual_actor_checkpoint=diagnostic.STAGE1A_RESIDUAL_ACTOR_CHECKPOINT,
            residual_actor_checkpoint_sha256=diagnostic.STAGE1A_RESIDUAL_ACTOR_SHA256,
            output_dir=args.output_dir.resolve(), report_path=args.report.resolve(),
            training_seed=int(args.seed), wandb_enabled=bool(args.wandb), wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name, wandb_group=args.wandb_group,
            runtime_variant=args.runtime_variant,
            preclose_collection_only=bool(args.preclose_collection_only),
            collection_source_catalog=args.preclose_source_catalog,
            boundary_paired_plan=boundary_paired_plan,
            close_readiness_training_contract=close_readiness_training_contract,
            close_readiness_initialization=close_readiness_initialization,
            frozen_student_advisory_checkpoint=(
                args.frozen_student_advisory_checkpoint
                if advisory_variant
                else None
            ),
            frozen_student_advisory_checkpoint_sha256=(
                args.frozen_student_advisory_sha256
                if advisory_variant
                else None
            ),
        )
        _write_launch_marker(launch_marker, "VECTOR_RUNTIME_RETURNED", exit_code=result)
        return result
    except BaseException as error:
        _write_launch_marker(
            launch_marker,
            "PYTHON_EXCEPTION",
            exception_type=type(error).__name__,
            exception=str(error),
        )
        raise
    finally:
        if counter is not None:
            counter.restore()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
