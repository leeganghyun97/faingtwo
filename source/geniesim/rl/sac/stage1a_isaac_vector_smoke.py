# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded, independently-stateful Stage-1A vector runtime.

This module is intentionally separate from the frozen one-environment runner.
It never changes its reward/CLOSE/SAC semantics; it only supplies the missing
per-clone ownership around the existing frozen coordinator and reward code.
"""

from __future__ import annotations

import csv
import json
import math
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from geniesim.rl.isaaclab.g2_policy_branch.action_interface import AbstractGripperIntent
from geniesim.rl.isaaclab.g2_policy_branch.contact_free_visual_bc_runtime import (
    load_contact_free_visual_bc_checkpoint,
    prepare_runtime_input,
    validate_metric_output,
)
from geniesim.rl.isaaclab.g2_policy_branch.hybrid_grasp_runtime import (
    HybridGraspPhaseRouter,
    HybridGraspRuntimeError,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
    nominal_grasp_pose_root_m_xyzw_for_cube,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_direct_pregrasp_init import (
    apply_vector_direct_pregrasp_initial_states,
)
from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
    G2_RIGHT_GRIPPER_MASTER,
    G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
    G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS,
    G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS,
)
from geniesim.rl.isaaclab.g2_asset_camera_pose import G2AssetCameraPoseResolver
from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
from geniesim.rl.sac.keyboard_grasp_contract import (
    canonical_keyboard_grasp_contract,
)
from geniesim.rl.sac.bc_residual_sac_contract import (
    compose_bc_and_residual_action,
)
from geniesim.rl.sac.stage1a_close_gate import (
    PrivilegedGeometryClosePersistenceGate,
    SimplifiedClosePersistenceGate,
    V31ClosePersistenceGate,
)
from geniesim.rl.sac.stage1a_close_readiness_contract import (
    build_pre_close_teacher_target,
    is_pre_close_candidate,
    target_receipt_dict,
)
from geniesim.rl.sac.stage1a_boundary_paired_collection import (
    BOUNDARY_PAIRED_ROW_SCHEMA,
    BoundaryProbe,
    apply_probe_to_nominal_pose,
    boundary_pair_id,
    source_family_id_from_sample_id,
)
from geniesim.rl.sac.stage1a_grasp_reward import (
    Stage1AGraspReward,
    Stage1ARewardV3Config,
    Stage1ARewardV32Config,
    stage1a_reward_contract,
)
from geniesim.rl.sac.stage1a_grasp_evaluation_contract import (
    canonical_grasp_evaluation_summary,
    canonical_grasp_evaluation_wandb_metrics,
    episode_contact_timeline_flags,
)
from geniesim.rl.sac.stage1a_reset_open_restore import (
    RESET_OPEN_RESTORE_SCHEMA,
    ResetOpenRestoreProgress,
)
from geniesim.rl.sac.stage1a_isaac_short_smoke import (
    CONTROL_HZ,
    MAX_FINAL_ACTION_M,
    _orientation_error_deg,
    _zero_xyz_close_execution_proposal,
)
from geniesim.rl.sac.stage1a_isaac_telemetry_adapter import (
    Stage1ARuntimeTransitionState,
    build_stage1a_reward_inputs,
)
from geniesim.rl.sac.stage1a_real_sac_coordinator import (
    REPLAY_STRATEGY_HER_FORCE,
    Stage1AAcceptedRealRow,
    Stage1ARealSACCoordinator,
)
from geniesim.rl.sac.stage1a_vector_contract import (
    PerEnvCameraCache,
    PerEnvCanonicalAction,
    PerEnvCoordinatorStateAdapter,
    PerEnvPacketPort,
    PerEnvReplayIdentity,
    PerEnvStateRegistry,
)
from geniesim.rl.sac.stage1a_vector_initial_states import (
    MEASURED_OPEN_RESTORE_MIN_EE_CUBE_CENTER_DISTANCE_M,
    select_catalog_collection_sample_for_source_family,
    select_stage1a_vector_initial_states,
)
from geniesim.rl.sac.stage1a_vector_runtime import (
    capture_vector_rgbd_sensor_frames,
    capture_vector_wrist_gru_inputs,
    consume_per_env_packet_once,
    stack_single_env_reward_inputs,
    vector_ee_and_cube_state,
)
from geniesim.rl.sac.stage1a_vector_telemetry import VectorPassiveContactPhysicsTelemetry
from geniesim.rl.sac.stage1a_fsm_sequence_dataset import (
    FSM_SEQUENCE_SCHEMA,
    FsmSequenceDatasetWriter,
)
from geniesim.rl.sac.stage1a_frozen_spatial_student import (
    DEFAULT_HIGH_READY_THRESHOLD,
    FrozenSpatialMissingnessStudent,
)
from geniesim.rl.sac.stage1a_current_gru_privileged import (
    CurrentGruPrivilegedAuxiliary,
)
from geniesim.rl.sac.stage1a_signed_readiness_margin import (
    build_signed_readiness_margin,
    receipt_dict as signed_margin_receipt_dict,
)
from geniesim.rl.sac.stage1a_v6_wrist_diagnostic import (
    V6_WRIST_DIAGNOSTIC_SCHEMA,
    capture_v6_wrist_diagnostic_frames,
    pooled_wrist_rgbd_feature_v6,
)


VECTOR_RUNTIME_SCHEMA = "g2_stage1a_independent_vector_runtime_v1"
V31_RUNTIME_VARIANT = "V3_1_POST_CLOSE_MICRO"
V31_LATERAL_OFF_RUNTIME_VARIANT = "V3_1_LATERAL_GATE_OFF_PAIRED"
# The paired 3K ablation is immutable evidence.  The 15K main run has a
# distinct identity so it cannot be mistaken for a resume or an extension of
# that bounded causal experiment.  Its executable policy is intentionally
# identical to the paired lateral-off route: V3.1 reward, the simplified
# 15--20 mm / 15 degree / five-step one-shot CLOSE gate, and lateral recorded
# only as telemetry.
V31_LATERAL_OFF_15K_RUNTIME_VARIANT = "V3_1_LATERAL_OFF_HER_FORCE_15K"
# The 30K run deliberately reuses the frozen V3.1 lateral-off behavior.  It
# has a distinct identity from the 3K paired ablation and prior 15K run, so
# its checkpoints and W&B records cannot be mistaken for either one.
V31_LATERAL_OFF_30K_RUNTIME_VARIANT = "V3_1_LATERAL_OFF_HER_FORCE_30K"
# The reset-fixed method comparison deliberately has fresh 6K identities.
# They must not be conflated with the historical 3K paired ablation or either
# 15K/30K continuation, all of which predate the canonical OPEN-restore
# contract.  Every FAIR_6K route is exactly 10 environments x 6000 accepted
# transitions and shares the same seed supplied by the launcher.
V3_CURRENT_FAIR_6K_RUNTIME_VARIANT = "V3_CURRENT_HER_FORCE_FAIR_6K"
V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_HER_FORCE_FAIR_6K"
)
V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_FAIR_6K"
)
V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_FAIR_6K"
)
# This route has a new immutable identity because it changes no reward, SAC,
# controller, or CLOSE authority, but it does add a durable *sidecar*
# sequence dataset.  Keeping it separate prevents a 25-env collection run
# from being confused with either frozen 10-env evidence run.
V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_15K"
)
# This is a fresh 10-env execution identity.  It preserves the deterministic
# V3.1 lateral-off FSM as the sole CLOSE authority and adds only frozen
# student telemetry plus a durable causal sequence receipt.  It must never be
# conflated with the earlier 25-env sequence-only collection.
V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K"
)
# A separate 25-env identity keeps the restarted advisory evidence distinct
# from the stopped 10-env run.  It has identical authority: the deterministic
# FSM remains the only runtime CLOSE owner.
V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K_25ENV"
)
# The reset-fixed 25-env re-test is deliberately not a continuation of the
# historical 15K route.  It preserves the same deterministic FSM CLOSE owner
# and frozen advisory sidecar, but is bounded to the 300-accepted-row-per-env
# exposure requested for reset-parity revalidation (25 * 300 = 7,500).
# Keeping a separate immutable identity prevents any report/checkpoint from
# being mistaken for the reset-confounded 15K evidence.
V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
# Fresh fair-comparison identities.  These aliases change no method behavior;
# they only let every already-canonical Stage-1A method run under the same
# measured-state reset gate, 25-env vector shape, and 7.5K exposure.  Keeping
# them distinct prevents reset-fixed evidence from being conflated with the
# historical reset-confounded 3K artifacts.
V3_CURRENT_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_CURRENT_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_1_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_2_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT = (
    "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_7P5K_25ENV"
)
# Fresh 6K identities requested after the G OPEN-restore smoke.  They reuse
# the exact same method implementations and measured-state reset contract as
# the 7.5K routes; only the accepted-transition budget differs.
V3_CURRENT_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV"
)
V31_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_1_HER_FORCE_RESET_FIXED_6K_25ENV"
)
V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV"
)
V32_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_2_HER_FORCE_RESET_FIXED_6K_25ENV"
)
PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV"
)
V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV"
)
CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT = (
    "V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV"
)
V32_RUNTIME_VARIANT = "V3_2_RELAX_LATERAL_STABILIZE"
V3_RUNTIME_VARIANT = "V3_BASELINE"
# This bounded run changes only CLOSE-readiness authority: the live
# primary-pad geometry oracle is a simulator teacher that chooses the
# canonical one-shot CLOSE.  RGB-D/robot state remain the only student
# observation, and the frozen GRU/BC still owns nominal XYZ.
PRIVILEGED_GEOMETRY_TEACHER_RUNTIME_VARIANT = (
    "V3_PRIVILEGED_GEOMETRY_TEACHER_HER_FORCE_3K"
)
# This distinct identity keeps the teacher-only student-distillation run
# separate from the earlier diagnostic hard geometry gate.  Its runtime CLOSE
# authority is the unchanged V3.1 lateral-off CURRENT gate.
PRIVILEGED_DISTILLATION_RUNTIME_VARIANT = (
    "V3_1_LATERAL_OFF_PRIVILEGED_DISTILLATION_HER_FORCE_3K"
)
# Geometry is acquired at 25 Hz; its last valid result is intentionally reused
# by the adjacent 50-Hz control step.  The timestamp is never synthesized.
PRIVILEGED_GEOMETRY_POLL_CONTROL_STEPS = 2
# The paired V3.1 ablation retains the exact V3.1 post-CLOSE micro controller.
# Its only behavioral change is dispatching the pre-CLOSE gate without lateral
# alignment as a hard admission predicate.
POST_CLOSE_MICRO_RUNTIME_VARIANTS = (
    V31_RUNTIME_VARIANT,
    V31_LATERAL_OFF_RUNTIME_VARIANT,
    V31_LATERAL_OFF_15K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_30K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    V31_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    V32_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
    V32_RUNTIME_VARIANT,
    PRIVILEGED_GEOMETRY_TEACHER_RUNTIME_VARIANT,
    PRIVILEGED_DISTILLATION_RUNTIME_VARIANT,
)
POST_CLOSE_SINGLE_CONTACT_MICRO_CAP_M = 0.00015
POST_CLOSE_BILATERAL_MICRO_CAP_M = 0.000075


class Stage1AVectorIsaacSmokeError(RuntimeError):
    pass


def _uses_privileged_geometry_teacher(runtime_variant: str) -> bool:
    """Return whether only CLOSE readiness is owned by the geometry teacher."""

    return runtime_variant in (
        PRIVILEGED_GEOMETRY_TEACHER_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _is_v31_variant(runtime_variant: str) -> bool:
    """Whether the route preserves the canonical V3.1 hard-lateral method."""

    return runtime_variant in (
        V31_RUNTIME_VARIANT,
        V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _is_v32_variant(runtime_variant: str) -> bool:
    """Whether the route preserves the canonical V3.2 reward/hover package."""

    return runtime_variant in (
        V32_RUNTIME_VARIANT,
        V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V32_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _uses_privileged_geometry_distillation(runtime_variant: str) -> bool:
    """Return whether geometry is teacher-only for the student head."""

    return runtime_variant == PRIVILEGED_DISTILLATION_RUNTIME_VARIANT


def _uses_privileged_geometry_receipt(runtime_variant: str) -> bool:
    """Return whether exact 25-Hz primary-pad telemetry is collected."""

    return (
        _uses_privileged_geometry_teacher(runtime_variant)
        or _uses_privileged_geometry_distillation(runtime_variant)
        or _uses_fsm_sequence_dataset(runtime_variant)
    )


def _uses_current_gru_privileged(runtime_variant: str) -> bool:
    """True for the signed-margin teacher / causal-GRU auxiliary route."""

    return runtime_variant in (
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _uses_frozen_student_advisory(runtime_variant: str) -> bool:
    """True only for telemetry-only frozen-student advisory routes."""

    return runtime_variant in (
        V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _is_v31_lateral_off_variant(runtime_variant: str) -> bool:
    """Return whether a route is the frozen V3.1 lateral-telemetry path."""

    return runtime_variant in (
        V31_LATERAL_OFF_RUNTIME_VARIANT,
        V31_LATERAL_OFF_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_30K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
        PRIVILEGED_DISTILLATION_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _is_v31_lateral_off_long_run(runtime_variant: str) -> bool:
    """Return whether the route has long-run diagnostic-only receipts."""

    return runtime_variant in (
        V31_LATERAL_OFF_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_30K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _uses_fsm_sequence_dataset(runtime_variant: str) -> bool:
    """Whether this run writes dual-RGB-D GRU sidecar evidence only."""

    return runtime_variant in (
        V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    )


def _is_fair_6k_variant(runtime_variant: str) -> bool:
    """Whether a route is an immutable reset-fixed 10-env fair comparison."""

    return runtime_variant in (
        V3_CURRENT_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
    )


RESET_FIXED_25ENV_7P5K_RUNTIME_VARIANTS = frozenset(
    {
        V3_CURRENT_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
    }
)

RESET_FIXED_25ENV_6K_RUNTIME_VARIANTS = frozenset(
    {
        V3_CURRENT_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V32_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
    }
)


def _is_reset_fixed_25env_7p5k_variant(runtime_variant: str) -> bool:
    return runtime_variant in RESET_FIXED_25ENV_7P5K_RUNTIME_VARIANTS


def _is_reset_fixed_25env_6k_variant(runtime_variant: str) -> bool:
    return runtime_variant in RESET_FIXED_25ENV_6K_RUNTIME_VARIANTS


def _is_reset_fixed_25env_variant(runtime_variant: str) -> bool:
    return _is_reset_fixed_25env_7p5k_variant(
        runtime_variant
    ) or _is_reset_fixed_25env_6k_variant(runtime_variant)


def _reset_fixed_7p5k_method(runtime_variant: str) -> str | None:
    return {
        V3_CURRENT_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: "V3 CURRENT",
        V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: "V3.1",
        V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: "V3.1 lateral-off",
        V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: "V3.2",
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: (
            "Privileged geometry hard-gate"
        ),
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: (
            "FSM advisory"
        ),
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT: (
            "CURRENT GRU + Privileged"
        ),
    }.get(runtime_variant)


def _reset_fixed_6k_method(runtime_variant: str) -> str | None:
    return {
        V3_CURRENT_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: "V3 CURRENT",
        V31_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: "V3.1",
        V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: "V3.1 lateral-off",
        V32_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: "V3.2",
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: (
            "Privileged geometry hard-gate"
        ),
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: (
            "FSM advisory"
        ),
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT: (
            "CURRENT GRU + Privileged"
        ),
    }.get(runtime_variant)


def _fair_6k_method(runtime_variant: str) -> str | None:
    """Return the comparison label without influencing runtime authority."""

    return {
        V3_CURRENT_FAIR_6K_RUNTIME_VARIANT: "V3_CURRENT",
        V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT: "V3_1_LATERAL_OFF",
        V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT: "DETERMINISTIC_FSM",
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT: (
            "FSM_CONDITIONAL_CORAL_ADVISORY_ONLY"
        ),
    }.get(runtime_variant)


def _replay_teacher_fields(
    *, runtime_variant: str, gate: Mapping[str, Any]
) -> tuple[bool | None, float | None, tuple[str, ...]]:
    """Return teacher values only for the explicit distillation route.

    Exact sequence labels are deliberately collected by the FSM sidecar, but
    they must never enter a :class:`Stage1AAcceptedRealRow`: that object is
    the SAC/HER_FORCE replay authority.  Keeping this guard at the replay
    boundary prevents an innocuous logging addition from changing RL data.
    """

    if not _uses_privileged_geometry_distillation(runtime_variant):
        return None, None, ()
    target = gate.get("privileged_close_ready_target")
    return (
        None if target is None else bool(target),
        None if target is None else float(gate["privileged_close_ready_score"]),
        tuple(
            str(reason)
            for reason in gate.get("privileged_close_ready_negative_reasons", ())
        ),
    )


def _completed_episode_summary(
    *,
    metrics_rows: list[Mapping[str, Any]],
    completed_episode_end_transition: Mapping[tuple[int, int], int],
    upper_transition: int,
    lower_exclusive_transition: int,
    slip_reference_m_s: float,
) -> dict[str, float | int | list[float]]:
    """Summarize only episodes that reached a canonical terminal row.

    Training transitions are intentionally not the denominator for the
    contact/stable rates.  A clone still in flight at a 3K boundary is not an
    evaluation episode, so it is excluded from both the cumulative and recent
    window summaries.  This helper is diagnostic-only: it never contributes
    to reward, replay, the CLOSE gate, or the student observation.
    """

    keys = {
        key
        for key, end_transition in completed_episode_end_transition.items()
        if lower_exclusive_transition < int(end_transition) <= upper_transition
    }
    by_episode: dict[tuple[int, int], list[Mapping[str, Any]]] = {
        key: [] for key in keys
    }
    for row in metrics_rows:
        key = (int(row["env_id"]), int(row["episode_id"]))
        if key in by_episode:
            by_episode[key].append(row)

    returns: list[float] = []
    residual_norms_mm: list[float] = []
    contact_count = bilateral_count = stable_count = 0
    premature_pre_close_contact_count = post_close_contact_count = 0
    valid_close_ready_contact_count = 0
    close_count = close_contact_count = close_bilateral_count = close_stable_count = 0
    close_no_contact_count = 0
    contact_loss_after_bilateral_count = slip_episode_count = hover_episode_count = 0
    for rows in by_episode.values():
        if not rows:
            continue
        timeline = episode_contact_timeline_flags(rows)
        contact = bool(timeline["any_contact"])
        bilateral = bool(timeline["any_bilateral"])
        stable = bool(timeline["any_stable"])
        close = bool(timeline["close"])
        premature_pre_close_contact = bool(
            timeline["premature_pre_close_contact"]
        )
        post_close_contact = bool(timeline["post_close_contact"])
        valid_close_ready_contact = bool(timeline["valid_close_ready_contact"])
        contact_loss_after_bilateral = bool(
            timeline["contact_loss_after_bilateral"]
        )
        post_close_bilateral = bool(timeline["post_close_bilateral"])
        post_close_stable = bool(timeline["post_close_stable"])
        slip = any(
            float(row["tangential_slip_m_s"]) > float(slip_reference_m_s)
            for row in rows
        )
        hover = any(bool(row["hover"]) for row in rows)
        returns.append(sum(float(row["reward_total"]) for row in rows))
        residual_norms_mm.extend(float(row["residual_norm_mm"]) for row in rows)
        contact_count += int(contact)
        bilateral_count += int(bilateral)
        stable_count += int(stable)
        premature_pre_close_contact_count += int(premature_pre_close_contact)
        post_close_contact_count += int(post_close_contact)
        valid_close_ready_contact_count += int(valid_close_ready_contact)
        close_count += int(close)
        # Conversion metrics are deliberately tied to events occurring after
        # the canonical CLOSE onset.  Any earlier contact remains visible in
        # ``premature_pre_close_contact_rate`` instead of being counted as a
        # successful CLOSE conversion.
        close_contact_count += int(close and post_close_contact)
        close_bilateral_count += int(close and post_close_bilateral)
        close_stable_count += int(close and post_close_stable)
        close_no_contact_count += int(close and not post_close_contact)
        contact_loss_after_bilateral_count += int(contact_loss_after_bilateral)
        slip_episode_count += int(slip)
        hover_episode_count += int(hover)

    episode_count = len(returns)

    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "episode_count": episode_count,
        "contact_rate": rate(contact_count, episode_count),
        "any_contact_rate": rate(contact_count, episode_count),
        "premature_pre_close_contact_rate": rate(
            premature_pre_close_contact_count, episode_count
        ),
        "post_close_contact_rate": rate(post_close_contact_count, episode_count),
        "valid_close_ready_contact_rate": rate(
            valid_close_ready_contact_count, episode_count
        ),
        "bilateral_rate": rate(bilateral_count, episode_count),
        "stable_rate": rate(stable_count, episode_count),
        "contact_to_bilateral": rate(bilateral_count, contact_count),
        "bilateral_to_stable": rate(stable_count, bilateral_count),
        "close_to_contact": rate(close_contact_count, close_count),
        "close_to_bilateral": rate(close_bilateral_count, close_count),
        "close_to_stable": rate(close_stable_count, close_count),
        "close_triggered_no_contact_rate": rate(close_no_contact_count, close_count),
        "contact_loss_after_bilateral_rate": rate(
            contact_loss_after_bilateral_count, bilateral_count
        ),
        "slip_rate": rate(slip_episode_count, episode_count),
        "hover_rate": rate(hover_episode_count, episode_count),
        "episode_return_mean": float(np.mean(returns)) if returns else 0.0,
        "residual_mean_mm": float(np.mean(residual_norms_mm))
        if residual_norms_mm
        else 0.0,
        "residual_p95_mm": _percentile_or_zero(residual_norms_mm, 95.0),
        "residual_max_mm": max(residual_norms_mm, default=0.0),
        "contact_episode_count": contact_count,
        "any_contact_episode_count": contact_count,
        "premature_pre_close_contact_episode_count": (
            premature_pre_close_contact_count
        ),
        "post_close_contact_episode_count": post_close_contact_count,
        "valid_close_ready_contact_episode_count": (
            valid_close_ready_contact_count
        ),
        "bilateral_episode_count": bilateral_count,
        "stable_episode_count": stable_count,
        "close_triggered_episode_count": close_count,
        "close_to_contact_count": close_contact_count,
        "close_to_bilateral_count": close_bilateral_count,
        "close_to_stable_count": close_stable_count,
        "close_triggered_no_contact_count": close_no_contact_count,
        "contact_loss_after_bilateral_count": contact_loss_after_bilateral_count,
        "slip_episode_count": slip_episode_count,
        "hover_episode_count": hover_episode_count,
    }


def _assert_finite_optimizer_metrics(metrics: Mapping[str, Any]) -> None:
    """Fail closed on an optimizer NaN/Inf instead of continuing a long run."""

    for name, value in metrics.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise Stage1AVectorIsaacSmokeError(
                    f"VECTOR_OPTIMIZER_NONFINITE:{name}"
                )


def _normalized_action_for_exact_raw_residual(raw_residual_m: np.ndarray) -> np.ndarray:
    """Invert the learner's radial-tanh parameterization without clipping.

    A V3.1 post-CLOSE action is an explicit controller-side cap.  Replay must
    contain the exact cap-respecting action that was actually submitted, not
    the uncapped actor proposal.  This inverse maps that known metric residual
    back to its normalized SAC representation and rejects rather than clips.
    """

    vector = np.asarray(raw_residual_m, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise Stage1AVectorIsaacSmokeError("V31_RAW_RESIDUAL_INVALID")
    maximum = float(canonical_keyboard_grasp_contract().residual_raw_maximum_norm_m)
    norm = float(np.linalg.norm(vector))
    if norm > maximum + 1.0e-12:
        raise Stage1AVectorIsaacSmokeError("V31_RAW_RESIDUAL_EXCEEDS_CONTRACT")
    if norm <= 1.0e-15:
        return np.zeros(3, dtype=np.float32)
    # ``norm == maximum`` is not representable by tanh, and therefore is
    # deliberately rejected instead of silently rounding inward.
    if norm >= maximum:
        raise Stage1AVectorIsaacSmokeError("V31_RAW_RESIDUAL_TANH_BOUNDARY")
    normalized_norm = math.atanh(norm / maximum)
    return np.asarray(normalized_norm * vector / norm, dtype=np.float32)


def _post_close_micro_execution_proposal(
    proposal: Any,
    *,
    alpha: Any,
    cap_m: float,
    control_phase: str,
) -> tuple[Any, dict[str, Any]]:
    """Create the exact, bounded post-CLOSE action and its audit receipt.

    The nominal BC XYZ action is zero after CLOSE.  Only a capped residual can
    remain, and its component on the configured nominal approach axis is
    removed.  Thus this diagnostic permits lateral stabilization only; it
    cannot turn into a forward normal push.
    """

    cap = float(cap_m)
    if cap < 0.0 or cap > POST_CLOSE_SINGLE_CONTACT_MICRO_CAP_M + 1.0e-12:
        raise Stage1AVectorIsaacSmokeError("V31_MICRO_CAP_OUT_OF_RANGE")
    source = np.asarray(
        proposal.composition.scaled_residual_contribution_m, dtype=np.float64
    )
    if source.shape != (3,) or not np.isfinite(source).all():
        raise Stage1AVectorIsaacSmokeError("V31_SOURCE_CONTRIBUTION_INVALID")
    # The frozen reward contract names +X as the nominal approach axis.  A
    # physical contact normal is intentionally *not* inferred from student
    # input.  Removing the entire approach-axis component is conservative:
    # the permitted micro correction lies only in root-Y/Z lateral space.
    lateral = source.copy()
    approach_axis_component_m = float(lateral[0])
    lateral[0] = 0.0
    lateral_norm = float(np.linalg.norm(lateral))
    cap_hit = bool(lateral_norm > cap + 1.0e-15)
    if cap <= 0.0 or lateral_norm <= 1.0e-15:
        contribution = np.zeros(3, dtype=np.float64)
    elif cap_hit:
        contribution = lateral * (cap / lateral_norm)
    else:
        contribution = lateral
    if float(alpha.alpha) <= 0.0:
        raise Stage1AVectorIsaacSmokeError("V31_ALPHA_INVALID")
    raw = contribution / float(alpha.alpha)
    normalized = _normalized_action_for_exact_raw_residual(raw)
    composition = compose_bc_and_residual_action(
        bc_action_metric_root_m=(0.0, 0.0, 0.0, 1.0),
        raw_sac_residual_metric_root_m=raw,
        alpha=alpha,
    )
    applied = replace(
        proposal,
        normalized_sac_action=normalized,
        raw_residual_metric_root_m=tuple(float(value) for value in raw),
        bc_action_4d_metric_root_m=(0.0, 0.0, 0.0, 1.0),
        composition=composition,
        residual_active=bool(float(np.linalg.norm(contribution)) > 0.0),
    )
    return applied, {
        "post_close_control_phase": control_phase,
        "post_close_micro_correction_allowed": bool(cap > 0.0),
        "post_close_micro_correction_cap_mm": 1000.0 * cap,
        "post_close_micro_correction_norm_mm": 1000.0
        * float(np.linalg.norm(contribution)),
        "post_close_micro_correction_cap_hit": cap_hit,
        "approach_axis_component_removed_mm": 1000.0
        * approach_axis_component_m,
        "forward_normal_push_allowed": False,
        "micro_correction_frame": "robot_root",
        "micro_correction_unit": "m",
    }


def _v31_observed_post_close_phase(
    *, gate: Mapping[str, Any], reward_step: Any, env_id: int
) -> str:
    """Return a telemetry-only phase from the completed transition."""

    if not bool(gate.get("close_latched", False)):
        return "PRE_CLOSE"
    if bool(reward_step.stable_grasp_boolean[env_id].item()):
        return "STABLE"
    if bool(reward_step.bilateral_contact_boolean[env_id].item()):
        return "BILATERAL"
    if bool(
        reward_step.left_contact_boolean[env_id].item()
        or reward_step.right_contact_boolean[env_id].item()
    ):
        return "SINGLE_CONTACT"
    return "CLOSE_TO_FIRST_CONTACT"


def _v31_failure_taxonomy(
    *, gate: Mapping[str, Any], reward_step: Any, env_id: int, reward: Stage1AGraspReward
) -> str:
    """Diagnose terminal V3.1 outcomes without feeding labels to the actor."""

    contact = bool(
        reward_step.left_contact_boolean[env_id].item()
        or reward_step.right_contact_boolean[env_id].item()
    )
    bilateral = bool(reward_step.bilateral_contact_boolean[env_id].item())
    stable = bool(reward_step.stable_grasp_boolean[env_id].item())
    if not contact:
        if not bool(gate.get("close_latched", False)):
            if gate.get("lateral_ready") is False:
                return "BAD_LATERAL_ALIGNMENT"
            if gate.get("orientation_ready") is False:
                return "BAD_ORIENTATION_ALIGNMENT"
            return "CLOSE_NOT_TRIGGERED"
        return "CLOSE_TRIGGERED_NO_PAD_CONTACT"
    if bilateral and not stable:
        if bool(reward_step.contact_loss_fail[env_id].item()):
            return "CONTACT_LOSS"
        if (
            float(reward_step.tangential_slip_m_s[env_id].item())
            > float(reward.config.slip_reference_m_s)
        ):
            return "HIGH_SLIP"
        if (
            float(reward_step.contact_normal_velocity_m_s[env_id].item())
            > float(reward.config.impact_reference_m_s)
        ):
            return "HIGH_RELATIVE_VELOCITY"
        return "UNKNOWN"
    return "NONE"


def _lateral_off_failure_taxonomy(
    *, gate: Mapping[str, Any], reward_step: Any, env_id: int, reward: Stage1AGraspReward
) -> str:
    """Classify a completed lateral-off episode without changing its policy.

    This is a post-transition diagnostic only.  In particular, the contact,
    slip, relative-motion, and cube-omega values below are never fed into the
    residual actor, GRU, CLOSE gate, or reward.  It gives the 15K main run an
    outcome taxonomy that does not collapse every non-success into ``NONE``.
    """

    contact = bool(
        reward_step.left_contact_boolean[env_id].item()
        or reward_step.right_contact_boolean[env_id].item()
    )
    bilateral = bool(reward_step.bilateral_contact_boolean[env_id].item())
    stable = bool(reward_step.stable_grasp_boolean[env_id].item())
    if stable:
        return "SUCCESS:STABLE_10_STEPS"
    if not contact:
        if not bool(gate.get("close_latched", False)):
            return "NO_CONTACT:CLOSE_NOT_TRIGGERED"
        return "NO_CONTACT:CLOSE_TRIGGERED_NO_CONTACT"
    if not bilateral:
        if bool(reward_step.contact_loss_fail[env_id].item()):
            return "SINGLE_CONTACT_ONLY:CONTACT_LOSS"
        return "SINGLE_CONTACT_ONLY:DWELL_TIMEOUT"
    if bool(reward_step.contact_loss_fail[env_id].item()):
        return "BILATERAL_NOT_STABLE:CONTACT_LOSS"
    if (
        float(reward_step.tangential_slip_m_s[env_id].item())
        > float(reward.config.slip_reference_m_s)
    ):
        return "BILATERAL_NOT_STABLE:HIGH_SLIP"
    if (
        float(reward_step.contact_normal_velocity_m_s[env_id].item())
        > float(reward.config.impact_reference_m_s)
    ):
        return "BILATERAL_NOT_STABLE:HIGH_RELATIVE_VELOCITY"
    if (
        float(reward_step.cube_angular_velocity_rad_s[env_id].item())
        > float(reward.config.cube_omega_reference_rad_s)
    ):
        return "BILATERAL_NOT_STABLE:HIGH_CUBE_ANGULAR_VELOCITY"
    return "BILATERAL_NOT_STABLE:UNKNOWN"


def _v32_failure_taxonomy(
    *, gate: Mapping[str, Any], reward_step: Any, env_id: int, reward: Stage1AGraspReward
) -> str:
    """Classify terminal outcomes without reintroducing a lateral hard gate.

    Every field used here is post-transition diagnostics only.  The result is
    persisted in replay provenance and never reaches the actor, the close
    gate, or any student observation.
    """

    contact = bool(
        reward_step.left_contact_boolean[env_id].item()
        or reward_step.right_contact_boolean[env_id].item()
    )
    bilateral = bool(reward_step.bilateral_contact_boolean[env_id].item())
    stable = bool(reward_step.stable_grasp_boolean[env_id].item())
    hardstop = bool(
        reward_step.runtime_hardstop[env_id].item()
        if reward_step.runtime_hardstop is not None
        else False
    )
    if stable:
        return "SUCCESS:STABLE_10_STEPS"
    if not contact:
        if not bool(gate.get("close_latched", False)):
            if gate.get("orientation_ready") is False:
                return "NO_CONTACT:BAD_ORIENTATION"
            return "NO_CONTACT:CLOSE_NOT_TRIGGERED"
        return "NO_CONTACT:CLOSE_TRIGGERED_NO_CONTACT"
    if not bilateral:
        if bool(reward_step.contact_loss_fail[env_id].item()):
            return "SINGLE_CONTACT_ONLY:CONTACT_LOSS"
        if int(reward_step.single_contact_counter[env_id].item()) > 10:
            return "SINGLE_CONTACT_ONLY:DWELL_TIMEOUT"
        return "SINGLE_CONTACT_ONLY:ALIGNMENT_FAILURE"
    if hardstop:
        return "BILATERAL_NOT_STABLE:PASSIVE_DYNAMICS"
    if bool(reward_step.contact_loss_fail[env_id].item()):
        return "BILATERAL_NOT_STABLE:CONTACT_LOSS"
    if (
        float(reward_step.tangential_slip_m_s[env_id].item())
        > float(reward.config.slip_reference_m_s)
    ):
        return "BILATERAL_NOT_STABLE:HIGH_SLIP"
    if (
        float(reward_step.contact_normal_velocity_m_s[env_id].item())
        > float(reward.config.impact_reference_m_s)
    ):
        return "BILATERAL_NOT_STABLE:HIGH_RELATIVE_VELOCITY"
    if (
        float(reward_step.cube_angular_velocity_rad_s[env_id].item())
        > float(reward.config.cube_omega_reference_rad_s)
    ):
        return "BILATERAL_NOT_STABLE:HIGH_CUBE_ANGULAR_VELOCITY"
    return "BILATERAL_NOT_STABLE:UNKNOWN"


def _percentile_or_zero(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _diagnostic_float(
    diagnostic: Mapping[str, Any], key: str, default: float = float("nan")
) -> float:
    """Serialize optional diagnostic values without inventing a measurement."""

    value = diagnostic.get(key, default)
    return float(default) if value is None else float(value)


def _runtime_metric_row(
    *,
    accepted_transitions: int,
    vector_step: int,
    env_id: int,
    episode_id: int,
    reward_step: Any,
    proposal: Any,
    gate: Mapping[str, Any],
    execution: Mapping[str, Any],
    nominal_distance_mm: float,
    phase: str,
    owner: str,
    camera_frame_id: int,
    camera_timestamp_s: float,
    packet_unique_4d_rows: int,
    packet_unique_8d_rows: int,
    forbidden_collision: bool,
    failure_reason: str = "NONE",
    close_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit scale- and authority-explicit V3/V3.1 analysis telemetry."""

    contact = bool(
        reward_step.left_contact_boolean[env_id].item()
        or reward_step.right_contact_boolean[env_id].item()
    )
    lateral_value = gate.get("lateral_alignment_error_mm")
    lateral_mm = float(lateral_value) if lateral_value is not None else float("nan")
    close_diagnostic = close_diagnostic or {}
    return {
        "accepted_transitions": accepted_transitions,
        "vector_step": vector_step,
        "env_id": env_id,
        "episode_id": episode_id,
        "reward_total": float(reward_step.reward_total[env_id]),
        "nominal_grasp_remaining_mm": nominal_distance_mm,
        "residual_norm_mm": 1000.0
        * float(np.linalg.norm(proposal.composition.scaled_residual_contribution_m)),
        "contact": int(contact),
        "bilateral": int(bool(reward_step.bilateral_contact_boolean[env_id].item())),
        "stable": int(bool(reward_step.stable_grasp_boolean[env_id].item())),
        "success": int(bool(reward_step.success[env_id].item())),
        "phase": phase,
        "owner": owner,
        "camera_frame_id": camera_frame_id,
        "camera_timestamp_s": camera_timestamp_s,
        "packet_unique_4d_rows": packet_unique_4d_rows,
        "packet_unique_8d_rows": packet_unique_8d_rows,
        "post_handoff_retreat_terminal": 0,
        "close_latched": int(bool(gate.get("close_latched", False))),
        "close_trigger": int(bool(gate.get("close_trigger", False))),
        "geometry_ready": int(bool(gate.get("geometry_ready", False))),
        "geometry_frame_id": int(gate.get("geometry_frame_id", -1) or 0),
        "geometry_poll_count": int(gate.get("geometry_poll_count", 0)),
        # A geometry timestamp exists only after a real 25-Hz oracle poll.
        # ``None`` on FAR_REACH / initial local rows means "not acquired",
        # not a control-derived sensor time; preserve that distinction as NaN
        # in the non-training metrics stream.
        "teacher_geometry_timestamp_s": _diagnostic_float(
            gate, "teacher_geometry_timestamp_s"
        ),
        "teacher_geometry_timestamp_source": str(
            gate.get("teacher_geometry_timestamp_source", "") or ""
        ),
        "student_feature_timestamp_s": _diagnostic_float(
            gate, "student_feature_timestamp_s"
        ),
        "gru_reset_generation": int(gate.get("gru_reset_generation", -1)),
        "privileged_teacher_available": int(
            bool(gate.get("privileged_teacher_available", False))
        ),
        "privileged_close_ready_target": (
            -1
            if gate.get("privileged_close_ready_target") is None
            else int(bool(gate["privileged_close_ready_target"]))
        ),
        # A live geometry receipt can exist before the row enters the
        # pre-CLOSE supervision domain.  In that case the teacher contract
        # deliberately represents the target/score as ``None`` rather than
        # fabricating a negative label.  Metrics must preserve that absence as
        # NaN; converting it with ``float(None)`` used to abort the whole
        # 25-env sequence collection once the first such local-grasp row was
        # observed.
        "privileged_close_ready_score": _diagnostic_float(
            gate, "privileged_close_ready_score"
        ),
        "student_close_ready_score": _diagnostic_float(
            gate, "student_close_ready_score"
        ),
        # This frozen score is telemetry-only.  The canonical action below
        # still consumes the unmodified deterministic FSM ``close_trigger``.
        "student_advisory_score": _diagnostic_float(gate, "student_advisory_score"),
        "student_advice": str(gate.get("student_advice", "NOT_CONFIGURED")),
        "student_ready_advisory": int(bool(gate.get("student_ready_advisory", False))),
        "student_defer_to_fsm": int(bool(gate.get("student_defer_to_fsm", False))),
        "fsm_close_triggered_receipt": int(
            bool(gate.get("fsm_close_triggered_receipt", gate.get("close_trigger", False)))
        ),
        "fsm_student_agreement": int(bool(gate.get("fsm_student_agreement", False))),
        "ready_advisory_fsm_not_close": int(
            bool(gate.get("ready_advisory_fsm_not_close", False))
        ),
        "fsm_close_student_defer": int(bool(gate.get("fsm_close_student_defer", False))),
        "student_advisory_teacher_target_known": int(
            gate.get("privileged_close_ready_target") is not None
        ),
        "ready_advisory_teacher_false_accept": int(
            bool(gate.get("ready_advisory_teacher_false_accept", False))
        ),
        "student_advisory_authority": str(
            gate.get("student_advisory_authority", "NOT_CONFIGURED")
        ),
        "student_privileged_input_count": int(
            gate.get("student_privileged_input_count", 0)
        ),
        "current_gru_privileged_score": _diagnostic_float(
            gate, "current_gru_privileged_score"
        ),
        "current_gru_privileged_margin": _diagnostic_float(
            gate, "current_gru_privileged_margin"
        ),
        "current_gru_privileged_trained": int(
            bool(gate.get("current_gru_privileged_trained", False))
        ),
        "current_gru_binary_loss": float(
            gate.get("current_gru_binary_loss", 0.0)
        ),
        "current_gru_signed_margin_loss": float(
            gate.get("current_gru_signed_margin_loss", 0.0)
        ),
        "current_gru_loss": float(gate.get("current_gru_loss", 0.0)),
        "current_gru_sequence_steps": int(
            gate.get("current_gru_sequence_steps", 0)
        ),
        "current_gru_privileged_input_count": int(
            gate.get("current_gru_privileged_input_count", 0)
        ),
        "pre_close_candidate": int(bool(gate.get("pre_close_candidate", False))),
        "close_latched_before_supervision": int(
            bool(gate.get("close_latched_before_supervision", False))
        ),
        "privileged_close_ready_negative_reasons": ";".join(
            str(reason)
            for reason in gate.get("privileged_close_ready_negative_reasons", ())
        ),
        "privileged_close_ready_excluded_reason": str(
            gate.get("privileged_close_ready_excluded_reason", "") or ""
        ),
        "close_readiness_target_schema": str(
            gate.get("close_readiness_target_schema", "") or ""
        ),
        "privileged_teacher_pad_geometry_ready": (
            -1
            if gate.get("privileged_teacher_pad_geometry_ready") is None
            else int(bool(gate["privileged_teacher_pad_geometry_ready"]))
        ),
        "privileged_teacher_aperture_ready": (
            -1
            if gate.get("privileged_teacher_aperture_ready") is None
            else int(bool(gate["privileged_teacher_aperture_ready"]))
        ),
        "privileged_teacher_orientation_ready": (
            -1
            if gate.get("privileged_teacher_orientation_ready") is None
            else int(bool(gate["privileged_teacher_orientation_ready"]))
        ),
        "privileged_teacher_safety_ready": (
            -1
            if gate.get("privileged_teacher_safety_ready") is None
            else int(bool(gate["privileged_teacher_safety_ready"]))
        ),
        "distance_ready": int(bool(gate.get("distance_ready", False))),
        "orientation_ready": int(bool(gate.get("orientation_ready", False))),
        "orientation_error_deg": _diagnostic_float(gate, "orientation_error_deg"),
        "lateral_ready": (
            -1 if gate.get("lateral_ready") is None else int(bool(gate["lateral_ready"]))
        ),
        "lateral_alignment_error_mm": lateral_mm,
        "persistence_ready_count": int(gate.get("persistence_ready_count", 0)),
        "post_close_control_phase": str(
            execution.get("post_close_control_phase", "PRE_CLOSE")
        ),
        "post_close_micro_correction_allowed": int(
            bool(execution.get("post_close_micro_correction_allowed", False))
        ),
        "post_close_micro_correction_cap_mm": float(
            execution.get("post_close_micro_correction_cap_mm", 0.0)
        ),
        "post_close_micro_correction_norm_mm": float(
            execution.get("post_close_micro_correction_norm_mm", 0.0)
        ),
        "post_close_micro_correction_cap_hit": int(
            bool(execution.get("post_close_micro_correction_cap_hit", False))
        ),
        "approach_axis_component_removed_mm": float(
            execution.get("approach_axis_component_removed_mm", 0.0)
        ),
        "forward_normal_push_allowed": int(
            bool(execution.get("forward_normal_push_allowed", False))
        ),
        "micro_slip_before_m_s": float(execution.get("micro_slip_before_m_s", float("nan"))),
        "micro_slip_after_m_s": float(execution.get("micro_slip_after_m_s", float("nan"))),
        "micro_relative_velocity_before_m_s": float(execution.get("micro_relative_velocity_before_m_s", float("nan"))),
        "micro_relative_velocity_after_m_s": float(execution.get("micro_relative_velocity_after_m_s", float("nan"))),
        "micro_cube_omega_before_rad_s": float(execution.get("micro_cube_omega_before_rad_s", float("nan"))),
        "micro_cube_omega_after_rad_s": float(execution.get("micro_cube_omega_after_rad_s", float("nan"))),
        "micro_contact_retained": int(bool(execution.get("micro_contact_retained", False))),
        "micro_correction_quality": str(execution.get("micro_correction_quality", "NOT_APPLIED")),
        "reward_bilateral_stability_shaping": float(
            reward_step.reward_bilateral_stability_shaping[env_id].item()
            if reward_step.reward_bilateral_stability_shaping is not None else 0.0
        ),
        "penalty_bilateral_instability": float(
            reward_step.penalty_bilateral_instability[env_id].item()
            if reward_step.penalty_bilateral_instability is not None else 0.0
        ),
        "hover": int(bool(reward_step.no_progress_hovering[env_id].item())),
        "single_contact_dwell_counter": int(
            reward_step.single_contact_counter[env_id].item()
        ),
        "tangential_slip_m_s": float(reward_step.tangential_slip_m_s[env_id].item()),
        "contact_normal_velocity_m_s": float(
            reward_step.contact_normal_velocity_m_s[env_id].item()
        ),
        "cube_angular_velocity_rad_s": float(
            reward_step.cube_angular_velocity_rad_s[env_id].item()
        ),
        "contact_loss_fail": int(bool(reward_step.contact_loss_fail[env_id].item())),
        "runtime_hardstop": int(
            bool(reward_step.runtime_hardstop[env_id].item())
            if reward_step.runtime_hardstop is not None else False
        ),
        "forbidden_collision": int(bool(forbidden_collision)),
        # CLOSE geometry is privileged diagnostic evidence only.  It is saved
        # in the metrics/replay-side audit stream, never in a student input.
        "inner_pad_cube_gap_mm": _diagnostic_float(
            close_diagnostic, "inner_pad_cube_gap_mm"
        ),
        "outer_pad_cube_gap_mm": _diagnostic_float(
            close_diagnostic, "outer_pad_cube_gap_mm"
        ),
        "minimum_primary_pad_cube_gap_mm": _diagnostic_float(
            close_diagnostic, "minimum_primary_pad_cube_gap_mm"
        ),
        "gripper_aperture_mm": _diagnostic_float(
            close_diagnostic, "gripper_aperture_mm"
        ),
        "cube_between_primary_pads": int(
            bool(close_diagnostic.get("cube_between_primary_pads", False))
        ),
        "aperture_geometrically_compatible": int(
            bool(close_diagnostic.get("aperture_geometrically_compatible", False))
        ),
        "close_control_timestamp_s": _diagnostic_float(
            close_diagnostic, "close_control_timestamp_s"
        ),
        "first_contact_after_close_ms": _diagnostic_float(
            close_diagnostic, "first_contact_after_close_ms"
        ),
        "failure_reason": failure_reason,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _task_bool(task_mdp: Any, env: Any, name: str) -> np.ndarray:
    value = task_mdp.forbidden_collision(env) if name == "forbidden" else None
    if value is None:
        raise Stage1AVectorIsaacSmokeError(f"VECTOR_TASK_VALUE_UNKNOWN:{name}")
    result = value.detach().to("cpu").numpy().astype(bool, copy=False).reshape(-1)
    if result.shape != (int(env.num_envs),):
        raise Stage1AVectorIsaacSmokeError(f"VECTOR_TASK_VALUE_SHAPE_INVALID:{name}")
    return result


def _primary_pad_close_geometry(
    *, env: Any, task_mdp: Any, state: Mapping[str, torch.Tensor], env_id: int
) -> dict[str, Any]:
    """Read Tier-A pad/cube geometry for the 25-Hz teacher receipt.

    The exact collision-enabled pad meshes remain the authority; no
    approximate pad center or student observation is substituted for the
    geometry oracle.  A caller may cache this live receipt for the immediately
    following 50-Hz control step, but it must never enter an actor input.
    """

    from geniesim.rl.isaaclab.g2_quaternion import (
        isaaclab_native_quaternion_order,
        quaternion_native_to_xyzw,
    )
    from geniesim.rl.sac.privileged_geometry_oracle import (
        runtime_primary_pad_cube_oracle,
    )

    robot = env.scene["robot"]
    body_names = tuple(str(name) for name in robot.body_names)
    required = ("gripper_r_inner_link4", "gripper_r_outer_link4")
    missing = [name for name in required if name not in body_names]
    if missing:
        raise Stage1AVectorIsaacSmokeError(
            f"VECTOR_CLOSE_GEOMETRY_BODY_MISSING:{missing}"
        )
    body_pos = torch.as_tensor(robot.data.body_pos_w, device=env.device)
    body_quat_xyzw = quaternion_native_to_xyzw(
        torch.as_tensor(robot.data.body_quat_w, device=env.device),
        isaaclab_native_quaternion_order(),
    )
    pose_by_name = {
        name: tuple(
            float(value)
            for value in torch.cat(
                (
                    body_pos[env_id, body_names.index(name)],
                    body_quat_xyzw[env_id, body_names.index(name)],
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        for name in required
    }
    return runtime_primary_pad_cube_oracle(
        cube_center_world_m=state["cube_position_world_m"][env_id]
        .detach()
        .cpu()
        .tolist(),
        cube_quat_world_xyzw=state["cube_quaternion_world_xyzw"][env_id]
        .detach()
        .cpu()
        .tolist(),
        cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
        body_pose_world_m_xyzw_by_name=pose_by_name,
        # ManagerBasedRLEnv materializes clone i under this verified IsaacLab
        # vector namespace.  Supplying it prevents the scalar oracle from
        # conflating identically named pads in the other nine environments.
        environment_root_path=f"/World/envs/env_{env_id}",
    )


def _privileged_close_readiness_teacher_receipt(
    *,
    geometry: Mapping[str, Any],
    orientation_error_deg: float,
    no_safety_violation: bool,
    pre_close_candidate: bool,
    close_latched_before_supervision: bool,
    geometry_frame_id: int | None,
    geometry_poll_count: int,
) -> dict[str, Any]:
    """Generate a teacher target without becoming a runtime admission gate.

    The target uses exact cube/pad geometry and the existing 15-degree
    orientation contract.  It deliberately has no persistence counter and is
    not read by the canonical CURRENT CLOSE state machine.  The target is
    binary admission semantics (ready=1.0, not-ready=0.0), never a predicate
    fraction that could make a negative example look admission-ready.
    """

    required = (
        "cube_between_primary_pads",
        "aperture_geometrically_compatible",
        "inner_pad_cube_gap_mm",
        "outer_pad_cube_gap_mm",
        "gripper_aperture_mm",
        "cube_effective_width_mm",
    )
    if any(name not in geometry for name in required):
        raise Stage1AVectorIsaacSmokeError(
            "VECTOR_PRIVILEGED_DISTILLATION_GEOMETRY_RECEIPT_INCOMPLETE"
        )
    inner_gap_mm = float(geometry["inner_pad_cube_gap_mm"])
    outer_gap_mm = float(geometry["outer_pad_cube_gap_mm"])
    aperture_mm = float(geometry["gripper_aperture_mm"])
    cube_width_mm = float(geometry["cube_effective_width_mm"])
    orientation_deg = float(orientation_error_deg)
    if (
        not all(
            math.isfinite(value)
            for value in (
                inner_gap_mm,
                outer_gap_mm,
                aperture_mm,
                cube_width_mm,
                orientation_deg,
            )
        )
        or cube_width_mm < 0.0
        or not isinstance(no_safety_violation, bool)
    ):
        raise Stage1AVectorIsaacSmokeError(
            "VECTOR_PRIVILEGED_DISTILLATION_TEACHER_INPUT_INVALID"
        )
    pad_geometry_ready = bool(
        geometry["cube_between_primary_pads"]
        and min(inner_gap_mm, outer_gap_mm) >= 0.0
    )
    aperture_ready = bool(
        geometry["aperture_geometrically_compatible"]
        and aperture_mm >= cube_width_mm
    )
    orientation_ready = bool(orientation_deg <= 15.0)
    target = build_pre_close_teacher_target(
        pre_close_candidate=pre_close_candidate,
        close_latched_before_supervision=close_latched_before_supervision,
        pad_geometry_ready=pad_geometry_ready,
        aperture_ready=aperture_ready,
        orientation_ready=orientation_ready,
        owner_valid=bool(geometry.get("owner_valid", True)),
        geometry_valid=bool(geometry.get("geometry_valid", True)),
        no_safety_violation=no_safety_violation,
    )
    return {
        **target_receipt_dict(target),
        "inner_pad_cube_gap_mm": inner_gap_mm,
        "outer_pad_cube_gap_mm": outer_gap_mm,
        "minimum_primary_pad_cube_gap_mm": min(inner_gap_mm, outer_gap_mm),
        # V2 signed-margin collection retains the exact geometry intermediates
        # when the live oracle provides them.  Older binary-only calls remain
        # valid; the V2 persistence contract rejects a missing value rather
        # than converting the Boolean containment flag into a fake margin.
        "min_pad_surface_gap_mm": geometry.get(
            "min_pad_surface_gap_mm", min(inner_gap_mm, outer_gap_mm)
        ),
        "inner_containment_margin_mm": geometry.get(
            "inner_containment_margin_mm"
        ),
        "outer_containment_margin_mm": geometry.get(
            "outer_containment_margin_mm"
        ),
        "primary_pad_containment_margin_mm": geometry.get(
            "primary_pad_containment_margin_mm"
        ),
        "inner_toward_world_m": geometry.get("inner_toward_world_m"),
        "outer_toward_world_m": geometry.get("outer_toward_world_m"),
        "cube_projection_world_m": geometry.get("cube_projection_world_m"),
        "gripper_aperture_mm": aperture_mm,
        "cube_effective_width_mm": cube_width_mm,
        "aperture_margin_mm": geometry.get(
            "aperture_margin_mm", aperture_mm - cube_width_mm
        ),
        "cube_between_primary_pads": bool(geometry["cube_between_primary_pads"]),
        "aperture_geometrically_compatible": bool(
            geometry["aperture_geometrically_compatible"]
        ),
        "orientation_error_deg": orientation_deg,
        "orientation_margin_deg": 15.0 - orientation_deg,
        "owner_valid": bool(geometry.get("owner_valid", True)),
        "geometry_valid": bool(geometry.get("geometry_valid", True)),
        "privileged_teacher_pad_geometry_ready": pad_geometry_ready,
        "privileged_teacher_aperture_ready": aperture_ready,
        "privileged_teacher_orientation_ready": orientation_ready,
        "privileged_teacher_safety_ready": no_safety_violation,
        "safety_valid": no_safety_violation,
        "privileged_teacher_geometry_frame_id": geometry_frame_id,
        "privileged_teacher_geometry_poll_count": int(geometry_poll_count),
        "privileged_teacher_authority": (
            "LIVE_USD_PHYSICS_PRIMARY_PAD_GEOMETRY_SUPERVISION_ONLY"
        ),
        "student_privileged_input_count": 0,
    }


def _fsm_sequence_teacher_receipt(
    *,
    geometry: Mapping[str, Any],
    orientation_error_deg: float,
    no_safety_violation: bool,
    pre_close_candidate: bool,
    close_latched_before_supervision: bool,
    geometry_frame_id: int | None,
    geometry_poll_count: int,
) -> dict[str, Any]:
    """Return exact sidecar labels without becoming runtime CLOSE authority.

    The binary target is intentionally restricted to causal pre-CLOSE rows.
    The independently useful signed physical margin can remain an exact
    diagnostic on post-CLOSE rows, but never enters an actor/replay input.
    """

    receipt = _privileged_close_readiness_teacher_receipt(
        geometry=geometry,
        orientation_error_deg=orientation_error_deg,
        no_safety_violation=no_safety_violation,
        pre_close_candidate=pre_close_candidate,
        close_latched_before_supervision=close_latched_before_supervision,
        geometry_frame_id=geometry_frame_id,
        geometry_poll_count=geometry_poll_count,
    )
    margin = build_signed_readiness_margin(
        pad_containment_margin_mm=receipt.get("primary_pad_containment_margin_mm"),
        minimum_primary_pad_cube_gap_mm=float(
            receipt["minimum_primary_pad_cube_gap_mm"]
        ),
        gripper_aperture_mm=float(receipt["gripper_aperture_mm"]),
        cube_effective_width_mm=float(receipt["cube_effective_width_mm"]),
        orientation_error_deg=float(receipt["orientation_error_deg"]),
        owner_valid=bool(receipt["owner_valid"]),
        geometry_valid=bool(receipt["geometry_valid"]),
        no_safety_violation=bool(no_safety_violation),
    )
    receipt.update(signed_margin_receipt_dict(margin))
    receipt["exact_signed_margin_available"] = bool(margin.recordable)
    receipt["signed_readiness_margin"] = margin.aggregate_margin
    receipt["pad_surface_gap_mm"] = float(
        receipt["minimum_primary_pad_cube_gap_mm"]
    )
    return receipt


def run_stage1a_isaac_vector_smoke(
    *,
    env: Any,
    p0a: Any,
    counter: Any,
    task_mdp: Any,
    source_freeze_before: Mapping[str, Any],
    source_freeze_provider: Callable[[], Mapping[str, Any]],
    selected_asset_path: Path,
    selected_asset_sha256: str,
    accepted_transition_target: int,
    bc_checkpoint: Path,
    bc_checkpoint_sha256: str,
    far_reach_checkpoint: Path,
    far_reach_checkpoint_sha256: str,
    residual_actor_checkpoint: Path,
    residual_actor_checkpoint_sha256: str,
    output_dir: Path,
    report_path: Path,
    training_seed: int = 42,
    wandb_enabled: bool = False,
    wandb_mode: str = "offline",
    wandb_project: str = "geniesim-g2-stage1a-residual-sac",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    runtime_variant: str = V3_RUNTIME_VARIANT,
    preclose_collection_only: bool = False,
    collection_source_catalog: Path | None = None,
    boundary_paired_plan: Mapping[str, Any] | None = None,
    close_readiness_training_contract: Mapping[str, Any] | None = None,
    close_readiness_initialization: Mapping[str, Any] | None = None,
    frozen_student_advisory_checkpoint: Path | None = None,
    frozen_student_advisory_checkpoint_sha256: str | None = None,
) -> int:
    """Run a 1-env parity/10-env smoke or a fresh 10-env HER_FORCE run.

    ``accepted_transition_target`` counts replay rows, never vector steps.
    The caller is responsible for constructing the environment with exactly
    the chosen vector count.  Existing scalar runs are never read or resumed.
    """

    num_envs = int(env.num_envs)
    if num_envs not in (1, 10, 25):
        raise Stage1AVectorIsaacSmokeError("VECTOR_RUNTIME_REQUIRES_NUM_ENVS_1_10_OR_25")
    if accepted_transition_target not in (100, 3000, 6000, 7500, 15000, 30000):
        raise Stage1AVectorIsaacSmokeError(
            "VECTOR_TARGET_MUST_BE_100_3000_6000_7500_15000_OR_30000"
        )
    if runtime_variant not in (
        V3_RUNTIME_VARIANT,
        V31_RUNTIME_VARIANT,
        V31_LATERAL_OFF_RUNTIME_VARIANT,
        V31_LATERAL_OFF_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_30K_RUNTIME_VARIANT,
        V3_CURRENT_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_SEQUENCE_FAIR_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_FAIR_6K_RUNTIME_VARIANT,
        V3_CURRENT_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT,
        V3_CURRENT_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V32_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V31_LATERAL_OFF_FSM_ADVISORY_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        CURRENT_GRU_PRIVILEGED_25ENV_RESET_FIXED_6K_RUNTIME_VARIANT,
        V32_RUNTIME_VARIANT,
        PRIVILEGED_GEOMETRY_TEACHER_RUNTIME_VARIANT,
        PRIVILEGED_DISTILLATION_RUNTIME_VARIANT,
    ):
        raise Stage1AVectorIsaacSmokeError("VECTOR_RUNTIME_VARIANT_UNKNOWN")
    if _uses_frozen_student_advisory(runtime_variant):
        if (
            frozen_student_advisory_checkpoint is None
            or frozen_student_advisory_checkpoint_sha256 is None
        ):
            raise Stage1AVectorIsaacSmokeError(
                "VECTOR_FROZEN_STUDENT_ADVISORY_CHECKPOINT_REQUIRED"
            )
    elif (
        frozen_student_advisory_checkpoint is not None
        or frozen_student_advisory_checkpoint_sha256 is not None
    ):
        raise Stage1AVectorIsaacSmokeError(
            "VECTOR_FROZEN_STUDENT_ADVISORY_ARGS_RESERVED"
        )
    if accepted_transition_target in (15000, 30000) and num_envs not in (10, 25):
        raise Stage1AVectorIsaacSmokeError("VECTOR_LONG_RUN_REQUIRES_NUM_ENVS_10_OR_25")
    if _is_fair_6k_variant(runtime_variant) and (
        num_envs != 10
        or accepted_transition_target != 6000
        or not wandb_enabled
        or wandb_mode != "online"
    ):
        raise Stage1AVectorIsaacSmokeError(
            "FAIR_6K_REQUIRES_NUM_ENVS_10_TARGET_6000_AND_ONLINE_WANDB"
        )
    if _is_reset_fixed_25env_7p5k_variant(runtime_variant) and (
        num_envs != 25
        or not (
            (
                accepted_transition_target == 7500
                and wandb_enabled
                and wandb_mode == "online"
            )
            or (accepted_transition_target == 100 and not wandb_enabled)
        )
    ):
        raise Stage1AVectorIsaacSmokeError(
            "RESET_FIXED_FAIR_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB"
        )
    if _is_reset_fixed_25env_6k_variant(runtime_variant) and (
        num_envs != 25
        or accepted_transition_target != 6000
        or not wandb_enabled
        or wandb_mode != "online"
    ):
        raise Stage1AVectorIsaacSmokeError(
            "RESET_FIXED_FAIR_6K_REQUIRES_NUM_ENVS_25_TARGET_6000_AND_ONLINE_WANDB"
        )
    # Legacy advisory-specific receipt retained for artifact/test readers:
    # V31_LATERAL_OFF_FSM_ADVISORY_RESET_FIXED_7P5K_REQUIRES_NUM_ENVS_25_TARGET_7500_AND_ONLINE_WANDB
    if runtime_variant == V31_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target not in (3000, 30000)
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_REQUIRES_NUM_ENVS_10_AND_TARGET_3000_OR_30000"
        )
    if runtime_variant == V31_LATERAL_OFF_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target != 3000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_LATERAL_OFF_REQUIRES_NUM_ENVS_10_AND_TARGET_3000"
        )
    if runtime_variant == V31_LATERAL_OFF_15K_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target != 15000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_LATERAL_OFF_15K_REQUIRES_NUM_ENVS_10_AND_TARGET_15000"
        )
    if runtime_variant == V31_LATERAL_OFF_30K_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target != 30000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_LATERAL_OFF_30K_REQUIRES_NUM_ENVS_10_AND_TARGET_30000"
        )
    if runtime_variant == V31_LATERAL_OFF_FSM_SEQUENCE_15K_RUNTIME_VARIANT and (
        num_envs != 25 or accepted_transition_target != 15000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_LATERAL_OFF_FSM_SEQUENCE_REQUIRES_NUM_ENVS_25_AND_TARGET_15000"
        )
    if runtime_variant == V31_LATERAL_OFF_FSM_ADVISORY_15K_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target != 15000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_LATERAL_OFF_FSM_ADVISORY_REQUIRES_NUM_ENVS_10_AND_TARGET_15000"
        )
    if runtime_variant == V31_LATERAL_OFF_FSM_ADVISORY_25ENV_15K_RUNTIME_VARIANT and (
        num_envs != 25 or accepted_transition_target != 15000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V31_LATERAL_OFF_FSM_ADVISORY_25ENV_REQUIRES_NUM_ENVS_25_AND_TARGET_15000"
        )
    if runtime_variant == V32_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target != 3000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "V32_REQUIRES_NUM_ENVS_10_AND_TARGET_3000"
        )
    if runtime_variant == PRIVILEGED_GEOMETRY_TEACHER_RUNTIME_VARIANT and (
        num_envs != 10 or accepted_transition_target != 3000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "PRIVILEGED_GEOMETRY_TEACHER_REQUIRES_NUM_ENVS_10_AND_TARGET_3000"
        )
    if _uses_privileged_geometry_distillation(runtime_variant) and (
        num_envs != 10 or accepted_transition_target != 3000
    ):
        raise Stage1AVectorIsaacSmokeError(
            "PRIVILEGED_DISTILLATION_REQUIRES_NUM_ENVS_10_AND_TARGET_3000"
        )
    if _uses_privileged_geometry_distillation(runtime_variant) and not preclose_collection_only and (
        close_readiness_training_contract is None
        or close_readiness_initialization is None
    ):
        raise Stage1AVectorIsaacSmokeError(
            "PRIVILEGED_DISTILLATION_REQUIRES_NORMALIZED_BALANCED_STUDENT_CONTRACT"
        )
    if _uses_privileged_geometry_distillation(runtime_variant) and not preclose_collection_only and not bool(
        close_readiness_training_contract.get("qualification", {}).get(
            "offline_contract_pass", False
        )
    ):
        raise Stage1AVectorIsaacSmokeError(
            "PRIVILEGED_DISTILLATION_OFFLINE_CONTRACT_NOT_QUALIFIED"
        )
    if not _uses_privileged_geometry_distillation(runtime_variant) and (
        close_readiness_training_contract is not None
        or close_readiness_initialization is not None
    ):
        raise Stage1AVectorIsaacSmokeError(
            "CLOSE_READINESS_STUDENT_CONTRACT_RESERVED_FOR_DISTILLATION_VARIANT"
        )
    if preclose_collection_only and (
        not _uses_privileged_geometry_distillation(runtime_variant)
        or num_envs != 10
        or accepted_transition_target != 3000
        or wandb_enabled
        or close_readiness_training_contract is not None
        or close_readiness_initialization is not None
    ):
        raise Stage1AVectorIsaacSmokeError(
            "PRECLOSE_COLLECTION_REQUIRES_10ENV_3K_TEACHER_ONLY_AND_NO_WANDB"
        )
    if collection_source_catalog is not None and not preclose_collection_only:
        raise Stage1AVectorIsaacSmokeError(
            "COLLECTION_SOURCE_CATALOG_RESERVED_FOR_COLLECTION_ONLY"
        )
    if boundary_paired_plan is not None and (
        not preclose_collection_only
        or collection_source_catalog is None
        or bool(wandb_enabled)
    ):
        raise Stage1AVectorIsaacSmokeError(
            "BOUNDARY_PAIRED_COLLECTION_REQUIRES_CATALOGGED_UPDATE_FREE_COLLECTION"
        )
    boundary_probes: tuple[BoundaryProbe, ...] = ()
    if boundary_paired_plan is not None:
        candidate_probes = boundary_paired_plan.get("probes")
        if (
            not isinstance(candidate_probes, tuple)
            or len(candidate_probes) < 2
            or not all(isinstance(item, BoundaryProbe) for item in candidate_probes)
            or boundary_paired_plan.get("student_privileged_input_count") != 0
            or boundary_paired_plan.get("sac_update") != 0
            or boundary_paired_plan.get("student_optimizer_update") != 0
            or boundary_paired_plan.get("privileged_hard_gate") is not False
        ):
            raise Stage1AVectorIsaacSmokeError("BOUNDARY_PAIRED_PLAN_RUNTIME_CONTRACT_INVALID")
        boundary_probes = candidate_probes
        if bool(boundary_paired_plan.get("v6_diagnostic_receipt", False)) and (
            boundary_paired_plan.get("wrist_rgbd_receipt") is not True
            or boundary_paired_plan.get("causal_student_state_receipt") is not True
            or boundary_paired_plan.get("v6_diagnostic_student_input") is not False
        ):
            raise Stage1AVectorIsaacSmokeError(
                "V6_DIAGNOSTIC_PLAN_RUNTIME_CONTRACT_INVALID"
            )
    if not math.isclose(float(env.step_dt), 0.02, rel_tol=0.0, abs_tol=1.0e-12):
        raise Stage1AVectorIsaacSmokeError("VECTOR_CONTROL_HZ_MUST_BE_50")
    if output_dir.exists() or report_path.exists():
        raise Stage1AVectorIsaacSmokeError("VECTOR_OUTPUT_REFUSES_OVERWRITE")
    if not selected_asset_path.is_file():
        raise Stage1AVectorIsaacSmokeError("VECTOR_CANDIDATE_ASSET_MISSING")

    # Leave a durable, monotonic stage marker before each native boundary.
    # This does not alter the learning path; it makes a crash before the first
    # replay transition attributable instead of appearing as a silent empty
    # run directory.
    output_dir.mkdir(parents=True, exist_ok=False)

    def mark(stage: str, **details: Any) -> None:
        _atomic_json(
            output_dir / "EXECUTION_STAGE.json",
            {
                "schema": VECTOR_RUNTIME_SCHEMA,
                "stage": stage,
                "num_envs": int(env.num_envs),
                "accepted_transitions": int(getattr(locals().get("coordinator", None), "accepted_transitions", 0)),
                **details,
            },
        )

    mark("PRE_DIRECT_INIT")

    # The established runtime/evaluation selector is immutable.  A bounded
    # teacher-only collection can request a different *already validated*
    # contact-free source subset per seed, preventing repeated batches from
    # inheriting the same episode-level class confound.
    selection = select_stage1a_vector_initial_states(
        num_envs=num_envs,
        selection_seed=int(training_seed) if preclose_collection_only else None,
        collection_source_catalog=collection_source_catalog,
        collection_source_family_ids=(
            tuple(str(item) for item in boundary_paired_plan["source_family_ids"])
            if boundary_paired_plan is not None
            and boundary_paired_plan.get("signed_margin_telemetry") is True
            else None
        ),
        collection_allow_duplicate_source_samples=bool(
            boundary_paired_plan is not None
            and boundary_paired_plan.get("paired_clone_allocation") is True
        ),
        collection_prefer_nearest_handoff_source=bool(
            boundary_paired_plan is not None
            and boundary_paired_plan.get("source_row_selection")
            == "NEAREST_STRICTLY_OUTSIDE_30MM_HANDOFF_FROM_FROZEN_CATALOG"
        ),
        # The reset-fixed 25-env route restores arm/cube state before its
        # controller-owned canonical OPEN can physically clear the gripper.
        # Use only source-attested OPEN rows that are outside the observed
        # closed-jaw source-clearance envelope.  This does not alter any
        # controller, CLOSE, reward, safety, or teacher predicate.
        measured_open_restore_min_ee_cube_center_distance_m=(
            MEASURED_OPEN_RESTORE_MIN_EE_CUBE_CENTER_DISTANCE_M
            if _is_reset_fixed_25env_variant(runtime_variant)
            else None
        ),
    )
    # Normal runtime/evaluation retains the immutable initial selection for
    # every reset.  Collection-only runs rotate an environment through new
    # rows in *its same approved source stratum* at episode boundaries.  This
    # creates natural pre-CLOSE state coverage without changing any CLOSE
    # predicate or using teacher labels/outcomes to select an initial state.
    active_initial_samples = list(selection.samples)
    collection_reset_source_receipts: list[dict[str, Any]] = []
    # A boundary pair always reuses one immutable source sample across every
    # label-agnostic probe.  Only after a complete probe cycle do we advance
    # to another pre-attested HDF5 row in that *same* source family.  Neither
    # operation consults the teacher target/outcome.
    boundary_probe_indices = np.asarray(
        (
            boundary_paired_plan.get("initial_probe_index_by_env")
            if boundary_paired_plan is not None
            and boundary_paired_plan.get("paired_clone_allocation") is True
            else [0] * num_envs
        ),
        dtype=np.int64,
    )
    if boundary_probe_indices.shape != (num_envs,) or np.any(
        boundary_probe_indices < 0
    ) or np.any(boundary_probe_indices >= max(1, len(boundary_probes))):
        raise Stage1AVectorIsaacSmokeError(
            "BOUNDARY_PAIRED_INITIAL_PROBE_ALLOCATION_INVALID"
        )
    boundary_pair_generations = np.zeros(num_envs, dtype=np.int64)
    boundary_seen_sample_ids: list[set[str]] = [
        {sample.sample_id} for sample in active_initial_samples
    ]
    selection_receipt = selection.receipt()
    paired_clone_allocation = bool(
        boundary_paired_plan is not None
        and boundary_paired_plan.get("paired_clone_allocation") is True
    )
    if (
        not selection_receipt["gripper_open_all"]
        or (
            not paired_clone_allocation
            and not selection_receipt["distinct_source_sample_ids"]
        )
    ):
        raise Stage1AVectorIsaacSmokeError(
            "VECTOR_INITIAL_SOURCE_SELECTION_VALIDATION_FAILED"
        )
    if paired_clone_allocation and any(
        active_initial_samples[env_id].sample_id
        != active_initial_samples[env_id + 1].sample_id
        for env_id in range(0, num_envs, 2)
    ):
        raise Stage1AVectorIsaacSmokeError(
            "BOUNDARY_PAIRED_CLONE_SOURCE_PARITY_INVALID"
        )
    direct_receipts = apply_vector_direct_pregrasp_initial_states(
        env,
        samples=selection.samples,
        allow_explicit_paired_clone_duplicates=paired_clone_allocation,
    )
    mark("POST_DIRECT_INIT", selected_sample_ids=[sample.sample_id for sample in selection.samples])
    if len(direct_receipts) != num_envs:
        raise Stage1AVectorIsaacSmokeError("VECTOR_DIRECT_INIT_RECEIPT_CARDINALITY")
    # Direct source restore intentionally writes only the arm/cube source
    # state.  Its authority correctly excludes the passive four-bar, but it
    # also cannot clear the gripper action term's *Python-side* limiter,
    # external-hold, close-arm, or passive-governor cache.  Reset that normal
    # action term before the first canonical OPEN packet.  This changes no
    # articulation coordinate, drive, speed cap, or threshold; the measured
    # OPEN restoration below remains the sole permission to activate policy.
    gripper_action_term = env.action_manager.get_term("gripper_action")
    gripper_action_reset = getattr(gripper_action_term, "reset", None)
    if not callable(gripper_action_reset):
        raise Stage1AVectorIsaacSmokeError("VECTOR_GRIPPER_ACTION_RESET_API_MISSING")
    gripper_action_reset(tuple(range(num_envs)))

    coordinator = Stage1ARealSACCoordinator.from_bc_checkpoint(
        bc_checkpoint,
        expected_sha256=bc_checkpoint_sha256,
        replay_capacity=accepted_transition_target + num_envs + 64,
        device=env.device,
        seed=int(training_seed),
        replay_strategy=REPLAY_STRATEGY_HER_FORCE,
        residual_actor_checkpoint_path=residual_actor_checkpoint,
        residual_actor_checkpoint_sha256=residual_actor_checkpoint_sha256,
        close_readiness_distillation=_uses_privileged_geometry_distillation(
            runtime_variant
        ),
        close_readiness_training_contract=close_readiness_training_contract,
        close_readiness_initialization=close_readiness_initialization,
    )
    reset_fixed_preflight = bool(
        _is_reset_fixed_25env_variant(runtime_variant)
        and accepted_transition_target == 100
    )
    if preclose_collection_only or reset_fixed_preflight:
        # Rows remain durable replay/telemetry evidence, but the optimizer is
        # unreachable.  This collection mode is not a short training run.
        coordinator.learning_starts = accepted_transition_target + 1
    far_model, _far_receipt = load_contact_free_visual_bc_checkpoint(
        far_reach_checkpoint,
        expected_sha256=far_reach_checkpoint_sha256,
        device=env.device,
    )
    state_adapter = PerEnvCoordinatorStateAdapter(coordinator, num_envs=num_envs)
    states = PerEnvStateRegistry(num_envs)
    cameras = PerEnvCameraCache(num_envs)
    # The frozen SAC/BC actor consumes only the wrist cache above.  The head
    # cache is a collection sidecar and is never passed to ``propose``.
    head_cameras = PerEnvCameraCache(num_envs)
    # The direct initial source restore preceded construction of the Python
    # caches.  Start an explicit reset generation anyway: a wrist sensor can
    # restart its local clock on the first subsequent physics packet, and
    # that transition is admissible only while OPEN parity is still pending.
    for env_id in range(num_envs):
        cameras.reset(env_id)
        head_cameras.reset(env_id)
    reward = Stage1AGraspReward(
        num_envs,
        device=env.device,
        v3_config=Stage1ARewardV3Config(),
        # Historical canonical expression, extended by the alias-aware helper:
        # v32_config=(Stage1ARewardV32Config() if runtime_variant == V32_RUNTIME_VARIANT else None)
        v32_config=(Stage1ARewardV32Config() if _is_v32_variant(runtime_variant) else None),
    )
    phase_routers = [HybridGraspPhaseRouter() for _ in range(num_envs)]
    def new_close_gate() -> (
        PrivilegedGeometryClosePersistenceGate
        | V31ClosePersistenceGate
        | SimplifiedClosePersistenceGate
    ):
        """Construct one episode-local gate without cross-episode state."""

        if _uses_privileged_geometry_teacher(runtime_variant):
            return PrivilegedGeometryClosePersistenceGate()
        if runtime_variant == V31_RUNTIME_VARIANT:
            return V31ClosePersistenceGate()
        if runtime_variant == V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT:
            return V31ClosePersistenceGate()
        return SimplifiedClosePersistenceGate()

    close_gates = [new_close_gate() for _ in range(num_envs)]
    frozen_student_advisory = (
        FrozenSpatialMissingnessStudent(
            checkpoint_path=Path(frozen_student_advisory_checkpoint),
            expected_sha256=str(frozen_student_advisory_checkpoint_sha256),
            high_ready_threshold=DEFAULT_HIGH_READY_THRESHOLD,
            device=env.device,
        )
        if _uses_frozen_student_advisory(runtime_variant)
        else None
    )
    if _uses_frozen_student_advisory(runtime_variant) and frozen_student_advisory is None:
        raise Stage1AVectorIsaacSmokeError("VECTOR_FROZEN_STUDENT_ADVISORY_INIT_FAILED")
    current_gru_privileged = (
        CurrentGruPrivilegedAuxiliary(num_envs=num_envs, device=env.device)
        if _uses_current_gru_privileged(runtime_variant)
        else None
    )
    if _uses_current_gru_privileged(runtime_variant) and current_gru_privileged is None:
        raise Stage1AVectorIsaacSmokeError("VECTOR_CURRENT_GRU_PRIVILEGED_INIT_FAILED")
    # These values are explicitly outside student observations and replay
    # actor inputs.  They are the 25-Hz teacher receipt for CLOSE readiness
    # only; the following 50-Hz step is permitted to reuse the same frame.
    privileged_geometry_cache: list[dict[str, Any] | None] = [
        None for _ in range(num_envs)
    ]
    privileged_geometry_frame_ids: list[int | None] = [
        None for _ in range(num_envs)
    ]
    privileged_geometry_poll_counts = [0 for _ in range(num_envs)]
    # Timestamp is attached only when the 25-Hz oracle is evaluated.  The
    # adjacent 50-Hz control row preserves that same receipt when it reuses
    # cached geometry; it is never synthesized from a control index.
    privileged_geometry_timestamp_s: list[float | None] = [
        None for _ in range(num_envs)
    ]
    # This is the post-action contact phase observed on the previous 50-Hz
    # transition.  It decides which V3.1 micro-cap applies to the next action
    # and never enters the student observation.
    post_close_control_phase = ["PRE_CLOSE" for _ in range(num_envs)]
    # The action at t is assessed against the already-observed physical state
    # at t-1.  These are diagnostic-only values: they are never presented to
    # the GRU/BC or residual actor.
    last_contact_quality: list[dict[str, Any] | None] = [None for _ in range(num_envs)]
    # Per-clone, replay-external failure chain.  It records privileged CLOSE
    # geometry and post-CLOSE dynamics for diagnosis only; no value stored
    # here can enter the frozen student observation, reward, or CLOSE gate.
    episode_failure_diagnostics: list[dict[str, Any] | None] = [
        None for _ in range(num_envs)
    ]
    finalized_failure_events: list[dict[str, Any]] = []
    packet_port = PerEnvPacketPort(batch_size=num_envs, device=env.device, binding_id="stage1a-vector")
    telemetry = VectorPassiveContactPhysicsTelemetry(env=env, task_mdp=task_mdp)
    telemetry.install()
    # A direct-state restore deliberately preserves the *measured* passive
    # four-bar.  Resolve the named measured coordinates once, but never write
    # any follower/mimic target.  The ordinary canonical OPEN action remains
    # the sole command authority during reset restoration.
    robot = env.scene["robot"]
    robot_joint_names = tuple(str(name) for name in robot.joint_names)
    if G2_RIGHT_GRIPPER_MASTER not in robot_joint_names:
        raise Stage1AVectorIsaacSmokeError("VECTOR_RESET_MASTER_JOINT_MISSING")
    missing_passive_names = tuple(
        name
        for name in G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES
        if name not in robot_joint_names
    )
    if missing_passive_names:
        raise Stage1AVectorIsaacSmokeError(
            f"VECTOR_RESET_PASSIVE_JOINT_MISSING:{list(missing_passive_names)}"
        )
    right_master_joint_index = robot_joint_names.index(G2_RIGHT_GRIPPER_MASTER)
    right_passive_joint_indices = tuple(
        robot_joint_names.index(name)
        for name in G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES
    )
    # A direct-state reset is a teleport, not a camera acquisition.  This
    # first packet only advances the sensor/physics epoch.  Policy/replay stay
    # blocked below until the same canonical OPEN action has demonstrated the
    # measured 60--120-step OPEN-restoration contract.
    initial_packet = packet_port.stage(
        [
            PerEnvCanonicalAction(
                env_id=env_id,
                final_action_4d_metric_root_m=(0.0, 0.0, 0.0, 0.0),
                gripper_intent=AbstractGripperIntent.OPEN,
            )
            for env_id in range(num_envs)
        ]
    )
    _initial_outputs, initial_consumption = consume_per_env_packet_once(
        env=env,
        counter=counter,
        port=packet_port,
        packet=initial_packet,
        label="VECTOR_DIRECT_INIT_CAMERA_REFRESH",
    )
    if not initial_consumption.single_batched_consumption or initial_consumption.action_broadcast_detected:
        raise Stage1AVectorIsaacSmokeError("VECTOR_DIRECT_INIT_REFRESH_CONSUMPTION_FAILED")
    mark("POST_DIRECT_INIT_CAMERA_REFRESH")
    previous_action = np.zeros((num_envs, 4), dtype=np.float32)
    previous_residual = np.zeros((num_envs, 3), dtype=np.float64)
    pending_reset = np.ones(num_envs, dtype=bool)
    episode_ids = [0 for _ in range(num_envs)]
    reset_open_restore_progress: list[ResetOpenRestoreProgress] = [
        ResetOpenRestoreProgress(
            env_id=env_id,
            episode_id=episode_ids[env_id],
            source_sample_id=active_initial_samples[env_id].sample_id,
            reset_vector_step=0,
        )
        for env_id in range(num_envs)
    ]
    # A Python cache reset is not proof that Isaac rendered a new image.  Keep
    # the raw camera identity observed immediately before every direct restore
    # and require a *different* raw identity before geometry can unlock
    # policy.  A camera can legitimately restart its local epoch on reset, so
    # numeric ordering across episodes is not an admissible freshness test.
    reset_camera_watermarks: list[tuple[int, float] | None] = [
        None for _ in range(num_envs)
    ]
    reset_open_restore_receipts: list[dict[str, Any]] = []
    reset_probe_completed = False
    # A V3 signed-margin collection starts matched clone pairs in the same
    # immutable source state.  It still has to demonstrate that one clone can
    # be reset without mutating the other nine environments.  Keep this probe
    # source/probe-preserving: advancing a boundary variant here would turn a
    # vector-contract check into an unrecorded collection intervention.
    reset_probe_source_preserving = False
    reset_probe_receipt: dict[str, Any] = {
        "executed": False,
        "reset_env_id": None,
        "other_env_state_unchanged": None,
        "source_probe_preserved": None,
    }
    nominal = np.asarray(
        [nominal_grasp_pose_root_m_xyzw_for_cube(sample.cube_pose_robot_root_m_xyzw[:3]) for sample in selection.samples],
        dtype=np.float64,
    )
    if nominal.shape != (num_envs, 7):
        raise Stage1AVectorIsaacSmokeError("VECTOR_NOMINAL_POSE_SHAPE_INVALID")
    if boundary_probes:
        for env_id in range(num_envs):
            nominal[env_id] = np.asarray(
                apply_probe_to_nominal_pose(
                    nominal[env_id], boundary_probes[int(boundary_probe_indices[env_id])]
                ),
                dtype=np.float64,
            )
    grasp_band_near_m, grasp_band_far_m = (
        canonical_keyboard_grasp_contract().local_grasp_band_m
    )
    run = None
    if wandb_enabled:
        import wandb
        run = wandb.init(
            project=wandb_project, entity=wandb_entity, name=wandb_run_name,
            group=wandb_group, mode=wandb_mode,
            config={
                "schema": VECTOR_RUNTIME_SCHEMA, "num_envs": num_envs,
                "accepted_transition_target": accepted_transition_target,
                "training_seed": int(training_seed),
                "control_hz": 50, "rgbd_hz": 25, "physics_hz": 500,
                "replay_strategy": "HER_FORCE",
                "reward_version": "V3.2" if _is_v32_variant(runtime_variant) else "V3",
                "runtime_variant": runtime_variant,
                "close_readiness_authority": (
                    "PRIVILEGED_GEOMETRY_TEACHER"
                    if _uses_privileged_geometry_teacher(runtime_variant)
                    else "CURRENT_DISTANCE_ALIGNMENT_FSM"
                    if _uses_fsm_sequence_dataset(runtime_variant)
                    else "CURRENT_RULE"
                ),
                "frozen_student_advisory": _uses_frozen_student_advisory(
                    runtime_variant
                ),
                "frozen_student_advisory_threshold": (
                    DEFAULT_HIGH_READY_THRESHOLD
                    if _uses_frozen_student_advisory(runtime_variant)
                    else None
                ),
                "frozen_student_advisory_checkpoint_sha256": (
                    frozen_student_advisory.checkpoint_sha256
                    if frozen_student_advisory is not None
                    else None
                ),
                "privileged_geometry_role": (
                    "RUNTIME_HARD_CLOSE_GATE"
                    if _uses_privileged_geometry_teacher(runtime_variant)
                    else "STUDENT_CLOSE_READINESS_DISTILLATION_TEACHER_ONLY"
                    if _uses_privileged_geometry_distillation(runtime_variant)
                    else "EXACT_SIGNED_MARGIN_SEQUENCE_TELEMETRY_ONLY"
                    if _uses_fsm_sequence_dataset(runtime_variant)
                    else "NOT_USED"
                ),
                "fsm_sequence_dataset_enabled": _uses_fsm_sequence_dataset(
                    runtime_variant
                ),
                "fsm_sequence_head_rgbd": _uses_fsm_sequence_dataset(
                    runtime_variant
                ),
                "fsm_sequence_right_wrist_rgbd": _uses_fsm_sequence_dataset(
                    runtime_variant
                ),
                "fair_comparison_id": (
                    "stage1a_reset_fixed_10env_6k"
                    if _is_fair_6k_variant(runtime_variant)
                    else None
                ),
                "fair_comparison_method": _fair_6k_method(runtime_variant),
                "gru_runtime_authority": False,
                "privileged_runtime_authority": False,
                "privileged_geometry_hz": (
                    25 if _uses_privileged_geometry_receipt(runtime_variant) else None
                ),
                "student_privileged_input_count": 0,
                "close_distillation_enabled": _uses_privileged_geometry_distillation(
                    runtime_variant
                ),
                "close_distillation_weight": 0.05
                if _uses_privileged_geometry_distillation(runtime_variant)
                else None,
                "close_readiness_training_contract_sha256": (
                    coordinator.close_readiness_training_contract["contract_sha256"]
                    if coordinator.close_readiness_training_contract is not None
                    else None
                ),
                "close_readiness_normalization_parity": bool(
                    coordinator.close_readiness_training_contract is not None
                ),
                "close_readiness_class_balanced_sampling": bool(
                    coordinator.close_readiness_training_contract is not None
                ),
                "close_readiness_class_balanced_loss": bool(
                    coordinator.close_readiness_training_contract is not None
                ),
                "bc_frozen": True, "residual_alpha": 0.10,
                "max_effective_residual_mm": 0.45,
                "v31_lateral_gate_mm": 10.0
                if _is_v31_variant(runtime_variant) else None,
                "v31_lateral_gate_mode": (
                    "HARD_10MM"
                    if _is_v31_variant(runtime_variant)
                    else "TELEMETRY_ONLY"
                    if _is_v31_lateral_off_variant(runtime_variant)
                    else "NOT_USED_BY_PRIVILEGED_GEOMETRY_TEACHER"
                    if _uses_privileged_geometry_teacher(runtime_variant)
                    else None
                ),
                "v31_single_contact_micro_cap_mm": 0.15
                if runtime_variant in POST_CLOSE_MICRO_RUNTIME_VARIANTS
                else None,
                "v31_bilateral_micro_cap_mm": 0.075
                if runtime_variant in POST_CLOSE_MICRO_RUNTIME_VARIANTS
                else None,
                "fair_comparison_id": (
                    "stage1a_reset_fixed_10env_6k"
                    if _is_fair_6k_variant(runtime_variant)
                    else None
                ),
                "fair_comparison_method": _fair_6k_method(runtime_variant),
                "paired_against_runtime_variant": (
                    V31_RUNTIME_VARIANT
                    if runtime_variant == V31_LATERAL_OFF_RUNTIME_VARIANT
                    else None
                ),
                "frozen_from_runtime_variant": (
                    V31_LATERAL_OFF_RUNTIME_VARIANT
                    if _is_v31_lateral_off_long_run(runtime_variant)
                    else None
                ),
                "v32_lateral_close_gate": "TELEMETRY_ONLY"
                if _is_v32_variant(runtime_variant) else None,
                "v32_phase_aware_hover": _is_v32_variant(runtime_variant),
                "v32_bilateral_stability_reward_per_step": 0.001
                if _is_v32_variant(runtime_variant) else None,
                "v32_bilateral_instability_penalty_per_step": -0.002
                if _is_v32_variant(runtime_variant) else None,
                "v32_single_contact_dwell_penalty": -0.006
                if _is_v32_variant(runtime_variant) else None,
                "v32_single_contact_micro_cap_mm": 0.15
                if _is_v32_variant(runtime_variant) else None,
                "v32_bilateral_micro_cap_mm": 0.075
                if _is_v32_variant(runtime_variant) else None,
            },
        )

    def propose(env_id: int, inputs: Any, distance_m: float) -> tuple[Any | None, Any | None, str, bool]:
        try:
            decision = phase_routers[env_id].decide(distance_m)
        except HybridGraspRuntimeError as error:
            # Keep the immutable 30-mm handoff rule.  In a vector rollout a
            # post-handoff retreat terminates only this clone's episode.
            if str(error) == "POST_HANDOFF_RETREAT_ABOVE_30MM":
                return None, None, "POST_HANDOFF_RETREAT_FAIL_CLOSED", True
            raise
        if decision.human_grasp_nominal_active:
            proposal = state_adapter.call(
                env_id, "propose", inputs, deterministic=False,
                residual_active=decision.residual_sac_active,
                gripper_authority_active=False,
            )
            return proposal, decision, "HUMAN_GRASP_GRU_BC", False
        robot = env.scene["robot"]
        arm_ids = torch.as_tensor(env.action_manager.get_term("arm_action")._joint_ids, dtype=torch.long, device=env.device)
        qdd = torch.as_tensor(robot.data.joint_acc, device=env.device).index_select(1, arm_ids)[env_id:env_id + 1]
        image, proprio = prepare_runtime_input(
            right_wrist_rgb=inputs.right_wrist_rgb[:, 0], right_wrist_depth_m=inputs.right_wrist_depth_m[:, 0],
            right_wrist_depth_valid=inputs.right_wrist_depth_valid[:, 0],
            ee_pose_robot_root_m_xyzw=inputs.ee_pose_robot_root_m_xyzw[:, 0],
            right_arm_joint_position_rad=inputs.right_arm_joint_position_rad[:, 0],
            right_arm_joint_velocity_rad_s=inputs.right_arm_joint_velocity_rad_s[:, 0],
            right_arm_joint_acceleration_rad_s2=qdd,
            gripper_state_open=torch.ones((1, 1), device=env.device),
            previous_policy_action_4d_metric_root_m=torch.cat((inputs.previous_policy_action_4d_metric_root_m[:, 0, :3], torch.zeros((1, 1), device=env.device)), dim=-1),
        )
        with torch.inference_mode():
            action = far_model(image, proprio)
        validate_metric_output(action)
        proposal = state_adapter.call(
            env_id, "propose", inputs, deterministic=False, residual_active=False,
            nominal_action_override_4d_metric_root_m=tuple(float(v) for v in action[0].detach().cpu().tolist()),
            gripper_authority_active=False,
        )
        return proposal, decision, "CUROBO_CONTACT_FREE_BC", False

    metrics_rows: list[dict[str, Any]] = []
    # Keep immutable per-packet evidence rather than inferring vectorization
    # from a final replay-row count.  Equal actions can be valid for two
    # clones, but every row must still have been built and consumed separately.
    action_packet_evidence: list[dict[str, Any]] = []
    tail_env_steps_not_accepted = 0
    post_handoff_retreat_discarded_env_steps = 0
    # An Isaac task termination that is not also a Stage-1A reward terminal
    # has no canonical reward-authoritative replay target.  Keep it outside
    # replay and reset that clone; never coerce it into ``terminated=True``.
    runtime_unattributed_termination_discarded_env_steps = 0
    transitions_path: Path | None = None
    close_readiness_rows_path: Path | None = None
    sequence_writer: FsmSequenceDatasetWriter | None = None
    # V5 has its own, pre-CLOSE-only frame store.  It is deliberately not the
    # FSM sequence store: the latter is a 50-Hz full-control authority while
    # this store exists solely to give each teacher row an exact existing
    # 25-Hz right-wrist RGB-D reference.
    wrist_rgbd_writer: FsmSequenceDatasetWriter | None = None
    v6_camera_pose_resolver: G2AssetCameraPoseResolver | None = None
    try:
        mark("PRE_POLICY_INITIALIZATION")
        transitions_path = output_dir / "REPLAY_TRANSITIONS.jsonl"
        close_readiness_rows_path = output_dir / "CLOSE_READINESS_PRE_CLOSE_ROWS.jsonl"
        if _uses_fsm_sequence_dataset(runtime_variant):
            sequence_writer = FsmSequenceDatasetWriter(
                output_dir=output_dir, num_envs=num_envs
            )
        if (
            boundary_paired_plan is not None
            and boundary_paired_plan.get("wrist_rgbd_receipt") is True
        ):
            wrist_rgbd_writer = FsmSequenceDatasetWriter(
                output_dir=output_dir,
                num_envs=num_envs,
                frames_filename=(
                    "V6_WRIST_RGBD_DIAGNOSTIC_FRAMES.h5"
                    if boundary_paired_plan.get("v6_diagnostic_receipt") is True
                    else "V5_WRIST_RGBD_FRAMES.h5"
                ),
                rows_filename=None,
                v6_wrist_diagnostic=bool(
                    boundary_paired_plan.get("v6_diagnostic_receipt") is True
                ),
            )
        if (
            boundary_paired_plan is not None
            and boundary_paired_plan.get("v6_diagnostic_receipt") is True
        ):
            # The resolver composes the live right-wrist parent body pose
            # with the checked-in USD optical extrinsic.  It does not write a
            # camera prim and is never used by the actor/controller.
            v6_camera_pose_resolver = G2AssetCameraPoseResolver(env)
        inputs, initial_frames = capture_vector_wrist_gru_inputs(
            env=env, p0a=p0a, previous_actions_4d_metric_root_m=previous_action,
            hidden_reset_mask=[True] * num_envs, camera_cache=cameras,
        )
        # ``current_frames`` belongs to the policy action about to be
        # executed.  It is deliberately distinct from ``next_frames``
        # captured after the physics step: a terminal transition must never
        # inherit a camera timestamp from the reset/new episode that Isaac
        # may have created while returning that step's termination flag.
        current_frames: list[Any | None] = list(initial_frames)
        head_current_frames: list[Any | None] = list(
            capture_vector_rgbd_sensor_frames(
                env=env,
                camera_scene_key="head_camera",
                camera_cache=head_cameras,
            )
        ) if sequence_writer is not None else [None for _ in range(num_envs)]
        state = vector_ee_and_cube_state(env=env, p0a=p0a)
        # Do not let even an inference-only proposal consume a GRU/reset
        # generation before measured OPEN restoration completes.  A pending
        # clone has no policy owner and emits only the canonical OPEN packet.
        proposals: list[Any | None] = []
        decisions: list[Any | None] = []
        owners: list[str] = []
        for env_id in range(num_envs):
            if pending_reset[env_id]:
                proposals.append(None)
                decisions.append(None)
                owners.append("RESET_OPEN_RESTORE")
                continue
            distance = float(np.linalg.norm(nominal[env_id, :3] - state["ee_position_root_m"][env_id].detach().cpu().numpy()))
            proposal, decision, owner, retreat = propose(env_id, inputs[env_id], distance)
            if retreat or proposal is None or decision is None:
                raise Stage1AVectorIsaacSmokeError(
                    "VECTOR_INITIAL_PREGRASP_PHASE_INVALID"
                )
            proposals.append(proposal); decisions.append(decision); owners.append(owner)
        mark("POST_POLICY_INITIALIZATION")

        periodic = (
            set(range(3000, accepted_transition_target + 1, 3000))
            if accepted_transition_target >= 3000 and not preclose_collection_only
            else set()
        )
        # Bounded targets that are not a multiple of 3K still need a durable
        # final checkpoint/evaluation.  Existing 3K-multiple routes are
        # unchanged because ``set.add`` is idempotent.
        if periodic:
            periodic.add(int(accepted_transition_target))
        saved: list[int] = []
        completed_episode_end_transition: dict[tuple[int, int], int] = {}
        checkpoint_evaluations: list[dict[str, Any]] = []
        last_checkpoint_boundary = 0

        def checkpoint_and_evaluate() -> None:
            """Save a full learner checkpoint and episode-level 3K receipt."""

            nonlocal last_checkpoint_boundary
            step = int(coordinator.accepted_transitions)
            if step not in periodic or step in saved:
                return
            receipt = coordinator.export_periodic_training_checkpoint(
                output_dir / "checkpoints" / f"checkpoint_{step}.pt",
                source_freeze=source_freeze_before,
                reward_config=stage1a_reward_contract(
                    reward_v3=True, reward_v32=_is_v32_variant(runtime_variant)
                ),
                runtime_config={
                    "num_envs": num_envs,
                    "training_seed": int(training_seed),
                    "control_hz": 50,
                    "physics_hz": 500,
                    "rgbd_hz": 25,
                    "replay_strategy": "HER_FORCE",
                    "runtime_variant": runtime_variant,
                    "close_readiness_authority": (
                        "PRIVILEGED_GEOMETRY_TEACHER"
                        if _uses_privileged_geometry_teacher(runtime_variant)
                        else "CURRENT_DISTANCE_ALIGNMENT_FSM"
                        if _uses_fsm_sequence_dataset(runtime_variant)
                        else "CURRENT_RULE"
                    ),
                    "fair_comparison_id": (
                        "stage1a_reset_fixed_10env_6k"
                        if _is_fair_6k_variant(runtime_variant)
                        else None
                    ),
                    "fair_comparison_method": _fair_6k_method(runtime_variant),
                    "student_privileged_input_count": 0,
                    "frozen_student_advisory": _uses_frozen_student_advisory(
                        runtime_variant
                    ),
                    "frozen_student_advisory_threshold": (
                        DEFAULT_HIGH_READY_THRESHOLD
                        if _uses_frozen_student_advisory(runtime_variant)
                        else None
                    ),
                    "frozen_student_advisory_checkpoint_sha256": (
                        frozen_student_advisory.checkpoint_sha256
                        if frozen_student_advisory is not None
                        else None
                    ),
                    "current_gru_privileged_auxiliary": bool(
                        current_gru_privileged is not None
                    ),
                    "privileged_teacher_used_in_sac_replay": False,
                },
                allow_bounded_final_boundary=bool(
                    step == int(accepted_transition_target)
                    and step % 3000 != 0
                ),
            )
            if not receipt.reload_pass:
                raise Stage1AVectorIsaacSmokeError(
                    "VECTOR_CHECKPOINT_RELOAD_FAILED"
                )
            if current_gru_privileged is not None:
                current_gru_privileged.save(
                    output_dir
                    / "checkpoints"
                    / f"current_gru_privileged_auxiliary_{step}.pt",
                    runtime_variant=runtime_variant,
                )
            learner_summary_at_boundary = coordinator.metrics()
            if not (
                bool(learner_summary_at_boundary["ACTOR_LOSS_FINITE"])
                and bool(learner_summary_at_boundary["CRITIC_LOSS_FINITE"])
                and bool(learner_summary_at_boundary["ALPHA_FINITE"])
            ):
                _atomic_json(
                    output_dir / "FAIL_CLOSED.json",
                    {
                        "schema": VECTOR_RUNTIME_SCHEMA,
                        "reason": "NONFINITE_SAC_METRICS",
                        "accepted_transitions": step,
                    },
                )
                raise Stage1AVectorIsaacSmokeError("VECTOR_SAC_METRICS_NONFINITE")
            cumulative = _completed_episode_summary(
                metrics_rows=metrics_rows,
                completed_episode_end_transition=completed_episode_end_transition,
                upper_transition=step,
                lower_exclusive_transition=0,
                slip_reference_m_s=float(reward.config.slip_reference_m_s),
            )
            recent = _completed_episode_summary(
                metrics_rows=metrics_rows,
                completed_episode_end_transition=completed_episode_end_transition,
                upper_transition=step,
                lower_exclusive_transition=last_checkpoint_boundary,
                slip_reference_m_s=float(reward.config.slip_reference_m_s),
            )
            cumulative_grasp_evaluation = canonical_grasp_evaluation_summary(
                cumulative
            )
            recent_grasp_evaluation = canonical_grasp_evaluation_summary(recent)
            safety = {
                "runtime_hardstop": sum(
                    int(row["runtime_hardstop"]) for row in metrics_rows
                ),
                "forbidden_collision": sum(
                    int(row["forbidden_collision"]) for row in metrics_rows
                ),
                "action_bound_violation": int(
                    learner_summary_at_boundary["FINAL_ACTION_BOUND_VIOLATION"]
                ),
                "gripper_authority_violation": int(
                    learner_summary_at_boundary["GRIPPER_AUTHORITY_VIOLATION"]
                ),
            }
            if any(safety.values()):
                _atomic_json(
                    output_dir / "FAIL_CLOSED.json",
                    {
                        "schema": VECTOR_RUNTIME_SCHEMA,
                        "reason": "AUTHORITATIVE_SAFETY_COUNTER_NONZERO",
                        "accepted_transitions": step,
                        "safety": safety,
                    },
                )
                raise Stage1AVectorIsaacSmokeError(
                    "VECTOR_AUTHORITATIVE_SAFETY_COUNTER_NONZERO"
                )
            payload = {
                "schema": "g2_stage1a_checkpoint_evaluation_v2",
                "checkpoint_step": step,
                "checkpoint_path": receipt.path,
                "checkpoint_sha256": receipt.sha256,
                "checkpoint_reload_pass": receipt.reload_pass,
                "metric_denominator": "COMPLETED_EPISODES_ONLY",
                "cumulative": cumulative,
                "recent_window": recent,
                "grasp_evaluation": {
                    "cumulative": cumulative_grasp_evaluation,
                    "recent_window": recent_grasp_evaluation,
                },
                "safety": safety,
                "sac": {
                    "alpha": learner_summary_at_boundary.get(
                        "optimizer/temperature/alpha"
                    ),
                    "actor_loss": learner_summary_at_boundary.get(
                        "optimizer/loss/actor"
                    ),
                    "critic_loss": learner_summary_at_boundary.get(
                        "optimizer/loss/critic"
                    ),
                },
                "current_gru_privileged": (
                    current_gru_privileged.metrics()
                    if current_gru_privileged is not None
                    else None
                ),
            }
            evaluation_dir = output_dir / "evaluation"
            _atomic_json(evaluation_dir / f"checkpoint_{step}.json", payload)
            checkpoint_evaluations.append(payload)
            if run is not None:
                cumulative_wandb_grasp_metrics = (
                    canonical_grasp_evaluation_wandb_metrics(cumulative)
                )
                recent_wandb_grasp_metrics = {
                    metric_key.replace("eval/", "eval_window/", 1): value
                    for metric_key, value in canonical_grasp_evaluation_wandb_metrics(
                        recent
                    ).items()
                }
                run.log(
                    {
                        "eval/episode_count": cumulative["episode_count"],
                        **cumulative_wandb_grasp_metrics,
                        "eval/close_to_contact": cumulative["close_to_contact"],
                        "eval/close_triggered_no_contact_rate": cumulative[
                            "close_triggered_no_contact_rate"
                        ],
                        "eval/contact_loss_after_bilateral_rate": cumulative[
                            "contact_loss_after_bilateral_rate"
                        ],
                        "eval/slip_rate": cumulative["slip_rate"],
                        "eval/hover_rate": cumulative["hover_rate"],
                        "eval/residual_mean_mm": cumulative["residual_mean_mm"],
                        "eval/residual_p95_mm": cumulative["residual_p95_mm"],
                        "eval/residual_max_mm": cumulative["residual_max_mm"],
                        "eval_window/episode_count": recent["episode_count"],
                        **recent_wandb_grasp_metrics,
                        "eval_window/close_to_contact": recent["close_to_contact"],
                        "eval_window/close_triggered_no_contact_rate": recent[
                            "close_triggered_no_contact_rate"
                        ],
                        "eval_window/contact_loss_after_bilateral_rate": recent[
                            "contact_loss_after_bilateral_rate"
                        ],
                        "eval_window/slip_rate": recent["slip_rate"],
                        "eval_window/hover_rate": recent["hover_rate"],
                        "train/episode_return": cumulative["episode_return_mean"],
                        "train/recent_episode_return": recent["episode_return_mean"],
                        "sac/alpha": payload["sac"]["alpha"],
                        "sac/actor_loss": payload["sac"]["actor_loss"],
                        "sac/critic_loss": payload["sac"]["critic_loss"],
                        "safety/runtime_hardstop": safety["runtime_hardstop"],
                        "safety/forbidden_collision": safety[
                            "forbidden_collision"
                        ],
                        "safety/action_bound_violation": safety[
                            "action_bound_violation"
                        ],
                        "safety/gripper_authority_violation": safety[
                            "gripper_authority_violation"
                        ],
                        **(
                            frozen_student_advisory_wandb_metrics()
                            if _uses_frozen_student_advisory(runtime_variant)
                            else {}
                        ),
                        **(
                            {
                                "student/teacher_agreement": current_gru_privileged.metrics()[
                                    "teacher_student_agreement"
                                ],
                                "student/false_accept": current_gru_privileged.metrics()[
                                    "false_accept_rate"
                                ],
                                "student/false_reject": current_gru_privileged.metrics()[
                                    "false_reject_rate"
                                ],
                                "student/signed_margin_loss": current_gru_privileged.metrics()[
                                    "signed_margin_loss"
                                ],
                                "student/gru_loss": current_gru_privileged.metrics()[
                                    "gru_loss"
                                ],
                                "student/privileged_input_count": 0,
                            }
                            if current_gru_privileged is not None
                            else {}
                        ),
                    },
                    step=step,
                )
            saved.append(step)
            last_checkpoint_boundary = step

        def close_distillation_wandb_metrics() -> dict[str, float | int | bool]:
            summary = coordinator.metrics()
            return {
                "distillation/privileged_close_ready_rate": summary[
                    "PRIVILEGED_CLOSE_READY_RATE"
                ],
                "distillation/student_close_ready_mean": summary[
                    "STUDENT_CLOSE_READY_MEAN"
                ],
                "distillation/teacher_student_agreement": summary[
                    "TEACHER_STUDENT_AGREEMENT"
                ],
                "distillation/student_precision": summary[
                    "STUDENT_CLOSE_READY_PRECISION"
                ],
                "distillation/student_recall": summary[
                    "STUDENT_CLOSE_READY_RECALL"
                ],
                "distillation/false_reject_rate": summary[
                    "STUDENT_FALSE_REJECT_RATE"
                ],
                "distillation/false_accept_rate": summary[
                    "STUDENT_FALSE_ACCEPT_RATE"
                ],
                "distillation/teacher_rows": summary["CLOSE_DISTILLATION_ROWS"],
                "distillation/update_count": summary[
                    "CLOSE_DISTILLATION_UPDATE_COUNT"
                ],
            }

        def current_gru_privileged_wandb_metrics() -> dict[str, float | int]:
            if current_gru_privileged is None:
                return {}
            summary = current_gru_privileged.metrics()
            return {
                "student/teacher_agreement": summary["teacher_student_agreement"],
                "student/false_accept": summary["false_accept_rate"],
                "student/false_reject": summary["false_reject_rate"],
                "student/binary_loss": summary["binary_loss"],
                "student/signed_margin_loss": summary["signed_margin_loss"],
                "student/gru_loss": summary["gru_loss"],
                "student/teacher_rows": summary["teacher_rows"],
                "student/update_count": summary["update_count"],
                "student/privileged_input_count": 0,
            }

        def frozen_student_advisory_wandb_metrics() -> dict[str, float | int]:
            """Aggregate telemetry-only advisor receipts without action access."""

            rows = [
                row
                for row in metrics_rows
                if row.get("student_advisory_authority")
                == "FROZEN_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY"
            ]
            scores = [
                float(row["student_advisory_score"])
                for row in rows
                if math.isfinite(float(row["student_advisory_score"]))
            ]
            known_teacher = [
                row
                for row in rows
                if bool(row.get("student_advisory_teacher_target_known", False))
            ]
            fsm_rows = [
                row
                for row in rows
                if bool(row.get("fsm_close_triggered_receipt", False))
                or bool(row.get("student_ready_advisory", False))
                or bool(row.get("student_defer_to_fsm", False))
            ]
            rate = lambda numerator, denominator: (
                float(numerator / denominator) if denominator else 0.0
            )
            return {
                "advisory/ready_count": sum(
                    int(row["student_ready_advisory"]) for row in rows
                ),
                "advisory/defer_count": sum(
                    int(row["student_defer_to_fsm"]) for row in rows
                ),
                "advisory/coverage": rate(
                    sum(int(row["student_ready_advisory"]) for row in rows),
                    len(rows),
                ),
                "advisory/defer_rate": rate(
                    sum(int(row["student_defer_to_fsm"]) for row in rows),
                    len(rows),
                ),
                "advisory/score_mean": float(np.mean(scores)) if scores else 0.0,
                "advisory/score_p95": _percentile_or_zero(scores, 95.0),
                "advisory/fsm_student_agreement": rate(
                    sum(int(row["fsm_student_agreement"]) for row in fsm_rows),
                    len(fsm_rows),
                ),
                "advisory/ready_before_fsm_close_count": sum(
                    int(row["ready_advisory_fsm_not_close"]) for row in rows
                ),
                "advisory/fsm_close_student_defer_count": sum(
                    int(row["fsm_close_student_defer"]) for row in rows
                ),
                "advisory/ready_teacher_false_accept_rate": rate(
                    sum(
                        int(row["ready_advisory_teacher_false_accept"])
                        for row in known_teacher
                    ),
                    sum(int(row["student_ready_advisory"]) for row in known_teacher),
                ),
                "fsm/close_trigger_count": sum(
                    int(row["fsm_close_triggered_receipt"]) for row in rows
                ),
            }

        vector_step = 0
        with ExitStack() as streams:
            metric_stream = streams.enter_context(
                (output_dir / "metrics.csv").open("x", newline="", encoding="utf-8")
            )
            replay_stream = streams.enter_context(
                transitions_path.open("x", encoding="utf-8")
            )
            failure_stream = streams.enter_context(
                (output_dir / "FAILURE_EVENTS.jsonl").open("x", encoding="utf-8")
            )
            reset_open_restore_stream = streams.enter_context(
                (output_dir / "RESET_OPEN_RESTORE_RECEIPTS.jsonl").open(
                    "x", encoding="utf-8"
                )
            )
            close_readiness_stream = (
                streams.enter_context(close_readiness_rows_path.open("x", encoding="utf-8"))
                if _uses_privileged_geometry_distillation(runtime_variant)
                else None
            )
            fieldnames = [
                "accepted_transitions", "vector_step", "env_id", "episode_id",
                "reward_total", "nominal_grasp_remaining_mm", "residual_norm_mm",
                "contact", "bilateral", "stable", "success", "phase", "owner",
                "camera_frame_id", "camera_timestamp_s", "packet_unique_4d_rows",
                "packet_unique_8d_rows", "post_handoff_retreat_terminal",
                "close_latched", "close_trigger", "geometry_ready",
                "geometry_frame_id", "geometry_poll_count",
                "teacher_geometry_timestamp_s", "teacher_geometry_timestamp_source",
                "student_feature_timestamp_s", "gru_reset_generation",
                "privileged_teacher_available", "privileged_close_ready_target",
                "privileged_close_ready_score", "student_close_ready_score",
                "student_advisory_score", "student_advice",
                "student_ready_advisory", "student_defer_to_fsm",
                "fsm_close_triggered_receipt", "fsm_student_agreement",
                "ready_advisory_fsm_not_close", "fsm_close_student_defer",
                "student_advisory_teacher_target_known",
                "ready_advisory_teacher_false_accept",
                "student_advisory_authority",
                "student_privileged_input_count", "distance_ready",
                "current_gru_privileged_score", "current_gru_privileged_margin",
                "current_gru_privileged_trained", "current_gru_binary_loss",
                "current_gru_signed_margin_loss", "current_gru_loss",
                "current_gru_sequence_steps",
                "current_gru_privileged_input_count",
                "pre_close_candidate", "close_latched_before_supervision",
                "privileged_close_ready_negative_reasons",
                "privileged_close_ready_excluded_reason",
                "close_readiness_target_schema",
                "privileged_teacher_pad_geometry_ready",
                "privileged_teacher_aperture_ready",
                "privileged_teacher_orientation_ready",
                "privileged_teacher_safety_ready",
                "orientation_ready", "orientation_error_deg", "lateral_ready",
                "lateral_alignment_error_mm",
                "persistence_ready_count", "post_close_control_phase",
                "post_close_micro_correction_allowed",
                "post_close_micro_correction_cap_mm",
                "post_close_micro_correction_norm_mm",
                "post_close_micro_correction_cap_hit",
                "approach_axis_component_removed_mm", "forward_normal_push_allowed",
                "micro_slip_before_m_s", "micro_slip_after_m_s",
                "micro_relative_velocity_before_m_s", "micro_relative_velocity_after_m_s",
                "micro_cube_omega_before_rad_s", "micro_cube_omega_after_rad_s",
                "micro_contact_retained", "micro_correction_quality",
                "reward_bilateral_stability_shaping", "penalty_bilateral_instability",
                "hover", "single_contact_dwell_counter", "tangential_slip_m_s",
                "contact_normal_velocity_m_s", "cube_angular_velocity_rad_s",
                "contact_loss_fail", "runtime_hardstop", "forbidden_collision",
                "inner_pad_cube_gap_mm", "outer_pad_cube_gap_mm",
                "minimum_primary_pad_cube_gap_mm", "gripper_aperture_mm",
                "cube_between_primary_pads", "aperture_geometrically_compatible",
                "close_control_timestamp_s", "first_contact_after_close_ms",
                "failure_reason",
            ]
            writer = csv.DictWriter(metric_stream, fieldnames=fieldnames); writer.writeheader()

            def persist_pre_close_teacher_row(
                *,
                identity: PerEnvReplayIdentity,
                proposal: Any,
                gate: Mapping[str, Any],
                student_feature_timestamp_s: float,
                gru_reset_generation: int,
                forbidden_collision_before_supervision: bool,
            ) -> None:
                """Persist deployable feature/teacher rows for offline-only validation."""

                if close_readiness_stream is None:
                    return
                target = gate.get("privileged_close_ready_target")
                if target is None:
                    return
                if (
                    not bool(gate.get("pre_close_candidate", False))
                    or bool(gate.get("close_latched_before_supervision", False))
                ):
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_CLOSE_READINESS_POSTCLOSE_ROW_FORBIDDEN"
                    )
                score = float(gate["privileged_close_ready_score"])
                if score != (1.0 if bool(target) else 0.0):
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_CLOSE_READINESS_NONBINARY_TARGET_FORBIDDEN"
                    )
                feature = np.asarray(proposal.actor_observation, dtype=np.float32)
                if feature.ndim != 1 or not np.isfinite(feature).all():
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_CLOSE_READINESS_STUDENT_FEATURE_INVALID"
                    )
                teacher_timestamp = gate.get("teacher_geometry_timestamp_s")
                raw_geometry_fields = (
                    "inner_pad_cube_gap_mm",
                    "outer_pad_cube_gap_mm",
                    "minimum_primary_pad_cube_gap_mm",
                    "gripper_aperture_mm",
                    "cube_effective_width_mm",
                    "orientation_error_deg",
                )
                signed_margin_collection = bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("signed_margin_telemetry") is True
                )
                # V4 is a recording-only representation experiment.  Its
                # state receipt is sampled from the same causal input packet
                # that produced ``proposal.actor_observation``.  Cube pose,
                # relative pose, and every privileged geometry field remain
                # teacher-only and are deliberately absent from this input.
                causal_state_receipt = bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("causal_student_state_receipt")
                    is True
                )
                wrist_rgbd_receipt = bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("wrist_rgbd_receipt") is True
                )
                v6_diagnostic_receipt = bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("v6_diagnostic_receipt") is True
                )
                if v6_diagnostic_receipt and not math.isclose(
                    float(teacher_timestamp),
                    float(student_feature_timestamp_s),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ):
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_V6_TEACHER_STUDENT_TIMESTAMP_MISMATCH"
                    )
                signed_margin_fields = (
                    "inner_containment_margin_mm",
                    "outer_containment_margin_mm",
                    "primary_pad_containment_margin_mm",
                    "min_pad_surface_gap_mm",
                    "aperture_margin_mm",
                    "orientation_margin_deg",
                    "nominal_grasp_residual_mm",
                    "inner_toward_world_m",
                    "outer_toward_world_m",
                    "cube_projection_world_m",
                )
                if (
                    not isinstance(teacher_timestamp, (int, float))
                    or not math.isfinite(float(teacher_timestamp))
                    or float(teacher_timestamp) < 0.0
                    or not math.isfinite(float(student_feature_timestamp_s))
                    or float(student_feature_timestamp_s) < 0.0
                    or type(gru_reset_generation) is not int
                    or gru_reset_generation < 0
                    or not all(
                        isinstance(gate.get(name), (int, float))
                        and math.isfinite(float(gate[name]))
                        for name in raw_geometry_fields
                    )
                    or (
                        signed_margin_collection
                        and not all(
                            isinstance(gate.get(name), (int, float))
                            and math.isfinite(float(gate[name]))
                            for name in signed_margin_fields
                        )
                    )
                    or not all(
                        isinstance(gate.get(name), bool)
                        for name in (
                            "cube_between_primary_pads",
                            "aperture_geometrically_compatible",
                            "owner_valid",
                            "geometry_valid",
                        )
                    )
                ):
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_CLOSE_READINESS_TEMPORAL_RECEIPT_INVALID"
                    )
                sample = active_initial_samples[identity.env_id]
                relative_pose_root_m = (
                    state["cube_position_root_m"][identity.env_id]
                    - state["ee_position_root_m"][identity.env_id]
                ).detach().cpu().numpy().astype(np.float64, copy=False)
                if relative_pose_root_m.shape != (3,) or not np.isfinite(relative_pose_root_m).all():
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_CLOSE_READINESS_RELATIVE_POSE_INVALID"
                    )
                negative_reason = (
                    "NONE"
                    if bool(target)
                    else ";".join(
                        str(reason)
                        for reason in gate.get(
                            "privileged_close_ready_negative_reasons", ()
                        )
                    )
                )
                payload: dict[str, Any] = {
                    "schema": "g2_stage1a_preclose_teacher_row_v1",
                    "row_id": identity.row_id,
                    "episode_key": f"env-{identity.env_id:02d}:{identity.episode_id}",
                    "env_id": identity.env_id,
                    "episode_id": identity.episode_id,
                    "pregrasp_source_sample_id": sample.sample_id,
                    "pregrasp_source_hdf5_row": int(sample.hdf5_row_index),
                    "pre_close_candidate": True,
                    "close_latched_before_supervision": False,
                    "teacher_geometry_timestamp_s": float(teacher_timestamp),
                    "teacher_geometry_timestamp_source": str(
                        gate.get("teacher_geometry_timestamp_source", "") or ""
                    ),
                    "student_feature_timestamp_s": float(student_feature_timestamp_s),
                    "gru_reset_generation": gru_reset_generation,
                    "privileged_close_ready_target": bool(target),
                    "privileged_close_ready_score": score,
                    "privileged_close_ready_negative_reasons": list(
                        gate.get("privileged_close_ready_negative_reasons", ())
                    ),
                    "negative_reason": negative_reason,
                    "inner_pad_cube_gap_mm": float(gate["inner_pad_cube_gap_mm"]),
                    "outer_pad_cube_gap_mm": float(gate["outer_pad_cube_gap_mm"]),
                    "minimum_primary_pad_cube_gap_mm": float(
                        gate["minimum_primary_pad_cube_gap_mm"]
                    ),
                    "gripper_aperture_mm": float(gate["gripper_aperture_mm"]),
                    "cube_effective_width_mm": float(gate["cube_effective_width_mm"]),
                    "orientation_error_deg": float(gate["orientation_error_deg"]),
                    "cube_between_primary_pads": bool(gate["cube_between_primary_pads"]),
                    "aperture_geometrically_compatible": bool(
                        gate["aperture_geometrically_compatible"]
                    ),
                    "owner_valid": bool(gate["owner_valid"]),
                    "geometry_valid": bool(gate["geometry_valid"]),
                    "student_privileged_input_count": 0,
                    "actor_observation": feature.tolist(),
                }
                if causal_state_receipt:
                    input_row = inputs[identity.env_id]
                    ee_pose = (
                        input_row.ee_pose_robot_root_m_xyzw[0, 0]
                        .detach().cpu().numpy().astype(np.float64, copy=False)
                    )
                    right_q = (
                        input_row.right_arm_joint_position_rad[0, 0]
                        .detach().cpu().numpy().astype(np.float64, copy=False)
                    )
                    right_qd = (
                        input_row.right_arm_joint_velocity_rad_s[0, 0]
                        .detach().cpu().numpy().astype(np.float64, copy=False)
                    )
                    gripper_state_open = float(
                        input_row.current_gripper_state[0, 0, 0].item()
                    )
                    previous_policy_action = np.asarray(
                        previous_action[identity.env_id], dtype=np.float64
                    )
                    if (
                        ee_pose.shape != (7,)
                        or right_q.shape != (7,)
                        or right_qd.shape != (7,)
                        or previous_policy_action.shape != (4,)
                        or not math.isfinite(gripper_state_open)
                        or not all(
                            np.isfinite(value).all()
                            for value in (
                                ee_pose,
                                right_q,
                                right_qd,
                                previous_policy_action,
                            )
                        )
                    ):
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_CLOSE_READINESS_CAUSAL_STATE_INVALID"
                        )
                    # This equality is intentional: both values describe the
                    # captured causal packet, not an old pregrasp snapshot.
                    robot_state_timestamp_s = float(student_feature_timestamp_s)
                    payload.update({
                        "student_causal_robot_state_schema": (
                            "g2_deployable_causal_robot_state_v1"
                        ),
                        "student_causal_state_fields": [
                            "ee_pose_robot_root_m_xyzw",
                            "right_arm_joint_position_rad",
                            "right_arm_joint_velocity_rad_s",
                            "gripper_state_open",
                            "measured_aperture_mm",
                            "previous_policy_action_4d_metric_root_m",
                        ],
                        "student_causal_state_forbidden_fields": [
                            "cube_gt",
                            "relative_pose_root_m",
                            "privileged_geometry",
                        ],
                        "robot_state_timestamp_s": robot_state_timestamp_s,
                        "state_timestamp_parity_abs_s": abs(
                            robot_state_timestamp_s
                            - float(student_feature_timestamp_s)
                        ),
                        "ee_pose_robot_root_m_xyzw": ee_pose.tolist(),
                        "right_arm_joint_position_rad": right_q.tolist(),
                        "right_arm_joint_velocity_rad_s": right_qd.tolist(),
                        "gripper_state_open": gripper_state_open,
                        "measured_aperture_mm": float(
                            gate["gripper_aperture_mm"]
                        ),
                        "previous_policy_action_4d_metric_root_m": (
                            previous_policy_action.tolist()
                        ),
                        "ee_linear_velocity_m_s": None,
                        "ee_angular_velocity_rad_s": None,
                        "ee_velocity_receipt": "NOT_AVAILABLE_IN_CAUSAL_INPUT_PACKET",
                        "student_input_privileged_field_count": 0,
                    })
                if wrist_rgbd_receipt:
                    # The policy input and this receipt must refer to the
                    # same source-owned 25-Hz wrist frame.  A new capture or
                    # an interpolated timestamp here would defeat V5's sole
                    # purpose, so fail closed instead.
                    if wrist_rgbd_writer is None:
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V5_WRIST_RGBD_WRITER_MISSING"
                        )
                    wrist_frame = current_frames[identity.env_id]
                    if wrist_frame is None:
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V5_WRIST_RGBD_FRAME_MISSING"
                        )
                    if not math.isclose(
                        float(wrist_frame.timestamp_s),
                        float(student_feature_timestamp_s),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    ):
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V5_WRIST_RGBD_TIMESTAMP_MISMATCH"
                        )
                    depth_valid_fraction = float(
                        np.mean(wrist_frame.depth_valid.astype(np.float32))
                    )
                    if (
                        wrist_frame.rgb.shape != (192, 256, 3)
                        or wrist_frame.depth_m.shape != (192, 256, 1)
                        or not np.isfinite(wrist_frame.depth_m).all()
                        or not math.isfinite(depth_valid_fraction)
                        or depth_valid_fraction <= 0.0
                    ):
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V5_WRIST_RGBD_FRAME_INVALID"
                        )
                    intrinsic_tensor = getattr(
                        env.scene["right_wrist_camera"].data,
                        "intrinsic_matrices",
                        None,
                    )
                    if intrinsic_tensor is None:
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V5_WRIST_CAMERA_INTRINSICS_MISSING"
                        )
                    intrinsics = np.asarray(
                        intrinsic_tensor[identity.env_id]
                        .detach().to("cpu").numpy(),
                        dtype=np.float64,
                    )
                    if (
                        intrinsics.shape != (3, 3)
                        or not np.isfinite(intrinsics).all()
                        or float(intrinsics[0, 0]) <= 0.0
                        or float(intrinsics[1, 1]) <= 0.0
                    ):
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V5_WRIST_CAMERA_INTRINSICS_INVALID"
                        )
                    frame_store_index = wrist_rgbd_writer.append_frame(
                        sensor="right_wrist",
                        episode_id=int(episode_ids[identity.env_id]),
                        frame=wrist_frame,
                    )
                    payload.update({
                        "wrist_rgbd_receipt_schema": (
                            "g2_stage1a_timestamp_aligned_wrist_rgbd_v5"
                        ),
                        "wrist_camera_binding": "right_wrist_camera",
                        "wrist_rgb_frame_ref": int(frame_store_index),
                        "wrist_depth_frame_ref": int(frame_store_index),
                        "right_wrist_frame_store_index": int(frame_store_index),
                        "wrist_rgbd_hdf5_path": str(
                            wrist_rgbd_writer.frames_path.resolve()
                        ),
                        "camera_frame_id": int(wrist_frame.frame_id),
                        "camera_timestamp_s": float(wrist_frame.timestamp_s),
                        "camera_age_ms": float(
                            1000.0 * abs(
                                float(student_feature_timestamp_s)
                                - float(wrist_frame.timestamp_s)
                            )
                        ),
                        "wrist_rgb_valid": True,
                        "wrist_depth_valid": True,
                        "wrist_depth_valid_fraction": depth_valid_fraction,
                        "wrist_rgb_shape": [192, 256, 3],
                        "wrist_depth_shape": [192, 256, 1],
                        "wrist_depth_unit": "m",
                        "wrist_camera_intrinsics_3x3": intrinsics.tolist(),
                        "wrist_camera_metadata_authority": (
                            "ISAAC_RIGHT_WRIST_CAMERA_DATA_INTRINSIC_MATRICES"
                        ),
                        "wrist_rgbd_timestamp_parity_abs_s": abs(
                            float(student_feature_timestamp_s)
                            - float(wrist_frame.timestamp_s)
                        ),
                        "student_input_privileged_field_count": 0,
                    })
                    if v6_diagnostic_receipt:
                        v6 = v6_pre_action_frames.get(identity.env_id)
                        if v6 is None:
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_V6_PREACTION_CAMERA_RECEIPT_MISSING"
                            )
                        v6_frame_store_index = wrist_rgbd_writer.append_v6_wrist_diagnostic(
                            episode_id=int(episode_ids[identity.env_id]),
                            frame=wrist_frame,
                            diagnostic={
                                "camera_world_pose_m_xyzw": v6.camera_world_pose_m_xyzw,
                                "cube_pose_camera_optical_m_xyzw": v6.cube_pose_camera_optical_m_xyzw,
                                "intrinsics_3x3": v6.intrinsics_3x3,
                                "cube_semantic_mask": v6.cube_semantic_mask,
                                "cube_instance_mask": v6.cube_instance_mask,
                                "cube_expected_silhouette_mask": v6.cube_expected_silhouette_mask,
                                "gt_cube_visible_pixel_count": v6.gt_cube_visible_pixel_count,
                                "gt_cube_projected_area_px": v6.gt_cube_projected_area_px,
                                "gt_occlusion_ratio": v6.gt_occlusion_ratio,
                                "cube_mask_depth_valid_ratio": v6.cube_mask_depth_valid_ratio,
                                "cube_mask_depth_valid_defined": v6.cube_mask_depth_valid_defined,
                            },
                        )
                        if v6_frame_store_index != frame_store_index:
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_V6_FRAME_REFERENCE_MISMATCH"
                            )
                        def _probability_logit(value: float) -> float:
                            if not 0.0 < value < 1.0 or not math.isfinite(value):
                                raise Stage1AVectorIsaacSmokeError(
                                    "VECTOR_V6_FROZEN_STUDENT_SCORE_INVALID"
                                )
                            return float(math.log(value / (1.0 - value)))
                        close_score = float(proposal.close_probability)
                        feasibility_score = float(proposal.feasibility_probability)
                        wrist_feature = pooled_wrist_rgbd_feature_v6(wrist_frame)
                        payload.update({
                            "schema": "g2_stage1a_signed_margin_teacher_state_wrist_diagnostic_row_v6",
                            "wrist_rgbd_receipt_schema": (
                                "g2_stage1a_timestamp_aligned_wrist_rgbd_v6"
                            ),
                            "v6_diagnostic_schema": V6_WRIST_DIAGNOSTIC_SCHEMA,
                            "v6_wrist_diagnostic_hdf5_path": str(
                                wrist_rgbd_writer.frames_path.resolve()
                            ),
                            "v6_wrist_diagnostic_frame_ref": int(v6_frame_store_index),
                            "right_wrist_camera_world_pose_m_xyzw": v6.camera_world_pose_m_xyzw.tolist(),
                            "right_wrist_camera_pose_authority": (
                                "URDF_LINK_PLUS_CHECKED_IN_G2_USD_OPTICAL_EXTRINSIC"
                            ),
                            "cube_pose_right_wrist_camera_optical_m_xyzw": (
                                v6.cube_pose_camera_optical_m_xyzw.tolist()
                            ),
                            "camera_optical_convention": v6.camera_optical_convention,
                            "v6_cube_semantic_mask_ref": int(v6_frame_store_index),
                            "v6_cube_instance_mask_ref": int(v6_frame_store_index),
                            "v6_cube_expected_silhouette_mask_ref": int(v6_frame_store_index),
                            "gt_cube_visible_pixel_count": int(v6.gt_cube_visible_pixel_count),
                            "gt_cube_projected_area_px": int(v6.gt_cube_projected_area_px),
                            "gt_occlusion_ratio": float(v6.gt_occlusion_ratio),
                            "cube_mask_depth_valid_ratio": float(v6.cube_mask_depth_valid_ratio),
                            "cube_mask_depth_valid_defined": bool(v6.cube_mask_depth_valid_defined),
                            "semantic_cube_label_ids": list(v6.semantic_label_ids),
                            "instance_cube_label_ids": list(v6.instance_label_ids),
                            "semantic_renderer_info_sha256": v6.semantic_info_sha256,
                            "instance_renderer_info_sha256": v6.instance_info_sha256,
                            "wrist_feature_schema": "RIGHT_WRIST_RGBD_AVGPOOL_16X16_TO_12X16_V6",
                            "wrist_feature_960d": wrist_feature.tolist(),
                            # These are frozen GRU deployable heads, not a
                            # newly trained V6 classifier.  Recording both
                            # prevents a later audit from conflating raw
                            # visual feature shift with current score scale.
                            "student_score_semantics": "FROZEN_GRU_FEASIBILITY_PROBABILITY",
                            "student_score": feasibility_score,
                            "student_logit": _probability_logit(feasibility_score),
                            "frozen_gru_close_score": close_score,
                            "frozen_gru_close_logit": _probability_logit(close_score),
                            "student_privileged_input_count": 0,
                        })
                if boundary_probes:
                    # Boundary canonical rows require a physically clean
                    # pre-CLOSE receipt.  OWNER/GEOMETRY invalid states remain
                    # diagnostics and cannot be silently repurposed as a
                    # geometry-not-ready student negative.
                    if (
                        forbidden_collision_before_supervision
                        or not bool(gate["owner_valid"])
                        or not bool(gate["geometry_valid"])
                    ):
                        return
                    probe = boundary_probes[int(boundary_probe_indices[identity.env_id])]
                    payload.update(
                        {
                            "schema": (
                                str(boundary_paired_plan["row_schema"])
                                if signed_margin_collection
                                else BOUNDARY_PAIRED_ROW_SCHEMA
                            ),
                            "source_family_id": source_family_id_from_sample_id(
                                sample.sample_id
                            ),
                            "source_sample_id": sample.sample_id,
                            "collection_batch_id": output_dir.name,
                            "source_provenance": {
                                "hdf5_path": sample.hdf5_path,
                                "hdf5_sha256": sample.hdf5_sha256,
                                "planner_sidecar_sha256": sample.report_sha256,
                                "catalog_sha256": str(
                                    boundary_paired_plan["source_catalog_sha256"]
                                ),
                            },
                            "boundary_pair_id": boundary_pair_id(
                                source_sample_id=sample.sample_id,
                                generation=int(
                                    boundary_pair_generations[identity.env_id]
                                ),
                            ),
                            "boundary_variant_id": probe.variant_id,
                            "boundary_nominal_offset_root_m": list(
                                probe.nominal_offset_root_m
                            ),
                            "perturbation_type": probe.perturbation_type,
                            "perturbation_value": {
                                "nominal_offset_root_m": list(
                                    probe.nominal_offset_root_m
                                ),
                                "orientation_root_z_deg": probe.orientation_root_z_deg,
                            },
                            "contact_free_at_direct_init": True,
                            "canonical_open_at_direct_init": True,
                            "forbidden_collision_before_supervision": False,
                            "relative_pose_root_m": relative_pose_root_m.tolist(),
                            "frozen_student_feature_128d": feature.tolist(),
                        }
                    )
                    if signed_margin_collection:
                        payload.update({
                            "signed_margin_telemetry_schema": "g2_close_readiness_signed_margin_v1",
                            "inner_containment_margin_mm": float(gate["inner_containment_margin_mm"]),
                            "outer_containment_margin_mm": float(gate["outer_containment_margin_mm"]),
                            "primary_pad_containment_margin_mm": float(gate["primary_pad_containment_margin_mm"]),
                            "min_pad_surface_gap_mm": float(gate["min_pad_surface_gap_mm"]),
                            "aperture_margin_mm": float(gate["aperture_margin_mm"]),
                            "orientation_margin_deg": float(gate["orientation_margin_deg"]),
                            "nominal_residual_mm": float(gate["nominal_grasp_residual_mm"]),
                            "inner_toward_world_m": float(gate["inner_toward_world_m"]),
                            "outer_toward_world_m": float(gate["outer_toward_world_m"]),
                            "cube_projection_world_m": float(gate["cube_projection_world_m"]),
                            "safety_valid": bool(gate["safety_valid"]),
                        })
                close_readiness_stream.write(
                    json.dumps(payload, sort_keys=True)
                    + "\n"
                )

            def _finite_or_none(value: Any) -> float | None:
                if not isinstance(value, (int, float)):
                    return None
                result = float(value)
                return result if math.isfinite(result) else None

            def persist_fsm_sequence_row(
                *,
                identity: PerEnvReplayIdentity,
                proposal: Any,
                executed: Any,
                gate: Mapping[str, Any],
                row: PerEnvCanonicalAction,
                reward_step: Any,
                records: list[Mapping[str, Any]],
                failure_reason: str,
            ) -> None:
                """Append exactly one causal 50-Hz sidecar row per replay row.

                This runs after the 500-Hz physical window has produced the
                outcome labels, but it stores the *pre-step* camera/robot
                observation and previous action.  Hence later GRU training
                can consume a causal history without looking at post-CLOSE
                contact geometry as an input.
                """

                if sequence_writer is None:
                    return
                wrist_frame = current_frames[identity.env_id]
                head_frame = head_current_frames[identity.env_id]
                if wrist_frame is None or head_frame is None:
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_FSM_SEQUENCE_CAMERA_FRAME_MISSING"
                    )
                wrist_index = sequence_writer.append_frame(
                    sensor="right_wrist",
                    episode_id=int(episode_ids[identity.env_id]),
                    frame=wrist_frame,
                )
                head_index = sequence_writer.append_frame(
                    sensor="head",
                    episode_id=int(episode_ids[identity.env_id]),
                    frame=head_frame,
                )
                actor_observation = np.asarray(
                    proposal.actor_observation, dtype=np.float32
                )
                if actor_observation.ndim != 1 or not np.isfinite(actor_observation).all():
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_FSM_SEQUENCE_ACTOR_OBSERVATION_INVALID"
                    )
                input_row = inputs[identity.env_id]
                right_q = (
                    input_row.right_arm_joint_position_rad[0, 0]
                    .detach().cpu().numpy().astype(np.float32, copy=False)
                )
                right_qd = (
                    input_row.right_arm_joint_velocity_rad_s[0, 0]
                    .detach().cpu().numpy().astype(np.float32, copy=False)
                )
                ee_pose = (
                    input_row.ee_pose_robot_root_m_xyzw[0, 0]
                    .detach().cpu().numpy().astype(np.float32, copy=False)
                )
                previous_action_row = np.asarray(
                    previous_action[identity.env_id], dtype=np.float32
                )
                final_action = np.asarray(
                    row.final_action_4d_metric_root_m, dtype=np.float32
                )
                residual_action = np.asarray(
                    executed.composition.scaled_residual_contribution_m,
                    dtype=np.float32,
                )
                if (
                    right_q.shape != (7,)
                    or right_qd.shape != (7,)
                    or ee_pose.shape != (7,)
                    or previous_action_row.shape != (4,)
                    or final_action.shape != (4,)
                    or residual_action.shape != (3,)
                    or not all(
                        np.isfinite(value).all()
                        for value in (
                            right_q, right_qd, ee_pose, previous_action_row,
                            final_action, residual_action,
                        )
                    )
                ):
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_FSM_SEQUENCE_DEPLOYABLE_STATE_INVALID"
                    )
                cube_delta_root_m = (
                    state["cube_position_root_m"][identity.env_id]
                    - state["ee_position_root_m"][identity.env_id]
                ).detach().cpu().numpy().astype(np.float64, copy=False)
                if cube_delta_root_m.shape != (3,) or not np.isfinite(cube_delta_root_m).all():
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_FSM_SEQUENCE_EE_CUBE_DELTA_INVALID"
                    )
                nominal_residual_mm = _finite_or_none(
                    gate.get("nominal_grasp_residual_mm")
                )
                orientation_error_deg = _finite_or_none(
                    gate.get("orientation_error_deg")
                )
                lateral_error_mm = float(abs(float(
                    nominal[identity.env_id, 1]
                    - state["ee_position_root_m"][identity.env_id, 1].item()
                )) * 1000.0)
                height_error_mm = float(abs(float(
                    nominal[identity.env_id, 2]
                    - state["ee_position_root_m"][identity.env_id, 2].item()
                )) * 1000.0)
                teacher = {
                    "authority": "LIVE_USD_PHYSICS_PRIMARY_PAD_GEOMETRY_TEACHER_ONLY",
                    "geometry_frame_id": gate.get("geometry_frame_id"),
                    "geometry_timestamp_s": _finite_or_none(
                        gate.get("teacher_geometry_timestamp_s")
                    ),
                    "geometry_timestamp_source": str(
                        gate.get("teacher_geometry_timestamp_source", "") or ""
                    ),
                    "owner_valid": gate.get("owner_valid"),
                    "geometry_valid": gate.get("geometry_valid"),
                    "primary_pad_containment_margin_mm": _finite_or_none(
                        gate.get("primary_pad_containment_margin_mm")
                    ),
                    "inner_containment_margin_mm": _finite_or_none(
                        gate.get("inner_containment_margin_mm")
                    ),
                    "outer_containment_margin_mm": _finite_or_none(
                        gate.get("outer_containment_margin_mm")
                    ),
                    "aperture_margin_mm": _finite_or_none(
                        gate.get("aperture_margin_mm")
                    ),
                    "gripper_aperture_mm": _finite_or_none(
                        gate.get("gripper_aperture_mm")
                    ),
                    "cube_effective_width_mm": _finite_or_none(
                        gate.get("cube_effective_width_mm")
                    ),
                    "pad_surface_gap_mm": _finite_or_none(
                        gate.get("pad_surface_gap_mm")
                    ),
                    "orientation_margin_deg": _finite_or_none(
                        gate.get("orientation_margin_deg")
                    ),
                    "signed_readiness_margin": _finite_or_none(
                        gate.get("signed_readiness_margin")
                    ),
                    "most_restrictive_predicate": gate.get(
                        "signed_readiness_most_restrictive_predicate"
                    ),
                    "binary_close_ready_target": gate.get(
                        "privileged_close_ready_target"
                    ),
                    "binary_close_ready_negative_reasons": list(
                        gate.get("privileged_close_ready_negative_reasons", ())
                    ),
                    "exact_signed_margin_available": bool(
                        gate.get("exact_signed_margin_available", False)
                    ),
                }
                control_timestamp_s = float(records[-1]["physics_timestamp_s"])
                if not math.isfinite(control_timestamp_s):
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_FSM_SEQUENCE_CONTROL_TIMESTAMP_INVALID"
                    )
                sample = active_initial_samples[identity.env_id]
                close_diagnostic = episode_failure_diagnostics[identity.env_id] or {}
                sequence_writer.append_control_row(
                    {
                        "schema": FSM_SEQUENCE_SCHEMA,
                        "episode_id": int(episode_ids[identity.env_id]),
                        "env_id": identity.env_id,
                        "source_family_id": source_family_id_from_sample_id(
                            sample.sample_id
                        ),
                        "source_sample_id": sample.sample_id,
                        "timestep": identity.step_in_episode,
                        "control_step": identity.step_in_episode,
                        "control_timestamp_s": control_timestamp_s,
                        "head_frame_store_index": head_index,
                        "right_wrist_frame_store_index": wrist_index,
                        "head_camera_frame_id": head_frame.frame_id,
                        "right_wrist_camera_frame_id": wrist_frame.frame_id,
                        "head_camera_timestamp_s": head_frame.timestamp_s,
                        "right_wrist_camera_timestamp_s": wrist_frame.timestamp_s,
                        "teacher_geometry_timestamp_s": teacher[
                            "geometry_timestamp_s"
                        ],
                        "gru_reset_generation": int(
                            states.state(identity.env_id).episode_index
                        ),
                        "student_privileged_input_count": 0,
                        "gru_runtime_authority": False,
                        "privileged_runtime_authority": False,
                        "student_advisory": {
                            "authority": str(
                                gate.get(
                                    "student_advisory_authority",
                                    "NOT_CONFIGURED",
                                )
                            ),
                            "score": _finite_or_none(
                                gate.get("student_advisory_score")
                            ),
                            "high_ready_threshold": _finite_or_none(
                                gate.get("student_advisory_threshold")
                            ),
                            "advice": str(
                                gate.get("student_advice", "NOT_CONFIGURED")
                            ),
                            "ready_advisory": bool(
                                gate.get("student_ready_advisory", False)
                            ),
                            "defer_to_fsm": bool(
                                gate.get("student_defer_to_fsm", False)
                            ),
                            "fsm_close_triggered": bool(
                                gate.get(
                                    "fsm_close_triggered_receipt",
                                    gate.get("close_trigger", False),
                                )
                            ),
                            "fsm_student_agreement": bool(
                                gate.get("fsm_student_agreement", False)
                            ),
                            "teacher_target_known": gate.get(
                                "privileged_close_ready_target"
                            )
                            is not None,
                            "ready_advisory_teacher_false_accept": bool(
                                gate.get(
                                    "ready_advisory_teacher_false_accept", False
                                )
                            ),
                        },
                        "student_observation": {
                            "actor_observation": actor_observation.tolist(),
                            "right_arm_q_rad": right_q.tolist(),
                            "right_arm_qd_rad_s": right_qd.tolist(),
                            "ee_pose_robot_root_m_xyzw": ee_pose.tolist(),
                            "causal_previous_gripper_state": float(
                                input_row.current_gripper_state[0, 0, 0].item()
                            ),
                        },
                        "previous_action_4d_metric_root_m": previous_action_row.tolist(),
                        "executed_action_4d_metric_root_m": final_action.tolist(),
                        "sac_residual_xyz_action_m": residual_action.tolist(),
                        "fsm": {
                            "authority": "CURRENT_DETERMINISTIC_DISTANCE_ALIGNMENT_FSM",
                            "nominal_grasp_residual_mm": nominal_residual_mm,
                            "ee_to_cube_center_mm": float(
                                np.linalg.norm(cube_delta_root_m) * 1000.0
                            ),
                            "lateral_error_mm": lateral_error_mm,
                            "height_error_mm": height_error_mm,
                            "orientation_error_deg": orientation_error_deg,
                            "distance_ready": bool(gate.get("distance_ready", False)),
                            "orientation_ready": bool(gate.get("orientation_ready", False)),
                            "safety_ready": bool(gate.get("safety_ready", True)),
                            "persistence_counter": int(
                                gate.get("persistence_ready_count", 0)
                            ),
                            "open_close_state": (
                                "CLOSE" if bool(gate.get("close_latched", False))
                                else "OPEN"
                            ),
                            "close_trigger": bool(gate.get("close_trigger", False)),
                        },
                        "teacher": teacher,
                        "outcome": {
                            "contact": bool(
                                reward_step.left_contact_boolean[identity.env_id].item()
                                or reward_step.right_contact_boolean[identity.env_id].item()
                            ),
                            "bilateral": bool(
                                reward_step.bilateral_contact_boolean[
                                    identity.env_id
                                ].item()
                            ),
                            "stable": bool(
                                reward_step.stable_grasp_boolean[
                                    identity.env_id
                                ].item()
                            ),
                            "contact_loss": bool(
                                reward_step.contact_loss_fail[identity.env_id].item()
                            ),
                            "slip_m_s": float(
                                reward_step.tangential_slip_m_s[identity.env_id].item()
                            ),
                            "high_slip": bool(
                                float(
                                    reward_step.tangential_slip_m_s[
                                        identity.env_id
                                    ].item()
                                )
                                > float(reward.config.slip_reference_m_s)
                            ),
                            "success": bool(
                                reward_step.success[identity.env_id].item()
                            ),
                            "failure_reason": str(failure_reason),
                            "time_to_first_contact_ms": close_diagnostic.get(
                                "first_contact_after_close_ms"
                            ),
                            "time_to_bilateral_ms": close_diagnostic.get(
                                "first_bilateral_after_close_ms"
                            ),
                            "time_to_stable_ms": close_diagnostic.get(
                                "first_stable_after_close_ms"
                            ),
                        },
                    }
                )

            def finalize_failure_event(
                *,
                env_id: int,
                reason: str,
                terminal_kind: str,
                terminal_timestamp_s: float,
                replay_eligible: bool,
            ) -> Mapping[str, Any] | None:
                """Persist one clone-local outcome without affecting replay.

                A task-level termination that is outside the reward authority
                remains excluded from replay, but its diagnostic event is
                still durable.  Thus failure attribution cannot be lost just
                because canonical replay correctly rejects that physical row.
                """

                if not _is_v31_lateral_off_long_run(runtime_variant):
                    return None
                diagnostic = episode_failure_diagnostics[env_id]
                if diagnostic is None:
                    diagnostic = {
                        "schema": "g2_stage1a_close_failure_event_v1",
                        "env_id": env_id,
                        "episode_id": episode_ids[env_id],
                        "close_triggered": False,
                        "first_contact_occurred": False,
                        "bilateral_occurred": False,
                        "stable_occurred": False,
                        "contact_loss": False,
                        "bilateral_dwell_steps": 0,
                        "micro_correction_count": 0,
                        "micro_correction_events": [],
                    }
                diagnostic.update(
                    {
                        "failure_reason": reason,
                        "terminal_kind": terminal_kind,
                        "terminal_timestamp_s": float(terminal_timestamp_s),
                        "replay_eligible": bool(replay_eligible),
                    }
                )
                failure_stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                finalized_failure_events.append(diagnostic)
                return diagnostic

            while coordinator.accepted_transitions < accepted_transition_target:
                mark("VECTOR_CONTROL_LOOP", vector_step=vector_step)
                state = vector_ee_and_cube_state(env=env, p0a=p0a)
                forbidden_before = _task_bool(task_mdp, env, "forbidden")
                rows: list[PerEnvCanonicalAction] = []
                close_onset: list[bool] = []
                executed_proposals: list[Any | None] = [None for _ in range(num_envs)]
                gate_receipts: list[dict[str, Any]] = [
                    {"close_latched": False, "close_trigger": False}
                    for _ in range(num_envs)
                ]
                execution_receipts: list[dict[str, Any]] = [
                    {
                        "post_close_control_phase": "RESET_REFRESH",
                        "post_close_micro_correction_allowed": False,
                        "post_close_micro_correction_cap_mm": 0.0,
                        "post_close_micro_correction_norm_mm": 0.0,
                        "post_close_micro_correction_cap_hit": False,
                        "approach_axis_component_removed_mm": 0.0,
                        "forward_normal_push_allowed": False,
                    }
                    for _ in range(num_envs)
                ]
                # V6 must capture the renderer AOV while ``current_frames``
                # is still the policy's actual 25-Hz input.  Persisting a
                # row after ``env.step`` is intentionally too late: Isaac may
                # already have advanced the live camera buffer to
                # ``next_frames``.  This short-lived cache is evidence only;
                # it is never read by the actor, reward, replay, or gate.
                v6_pre_action_frames: dict[int, Any] = {}
                for env_id in range(num_envs):
                    if pending_reset[env_id]:
                        final = (0.0, 0.0, 0.0, 0.0); intent = AbstractGripperIntent.OPEN; close_onset.append(False)
                    else:
                        current_ee_root_m = (
                            state["ee_position_root_m"][env_id]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        residual_error_root_m = nominal[env_id, :3] - current_ee_root_m
                        residual_m = float(np.linalg.norm(residual_error_root_m))
                        lateral_m = float(np.linalg.norm(residual_error_root_m[1:]))
                        orientation_deg = _orientation_error_deg(nominal[env_id, 3:], state["ee_quaternion_root_xyzw"][env_id].detach().cpu().numpy())
                        if _uses_privileged_geometry_teacher(runtime_variant):
                            geometry = privileged_geometry_cache[env_id]
                            # A geometry poll is a real primary-pad mesh read
                            # at 25 Hz.  Its cached receipt is reused at the
                            # intervening 50-Hz control step; no camera or
                            # synthetic geometry timestamp reaches the actor.
                            if (
                                decisions[env_id].human_grasp_nominal_active
                                and vector_step
                                % PRIVILEGED_GEOMETRY_POLL_CONTROL_STEPS == 0
                            ):
                                geometry = _primary_pad_close_geometry(
                                    env=env,
                                    task_mdp=task_mdp,
                                    state=state,
                                    env_id=env_id,
                                )
                                privileged_geometry_cache[env_id] = geometry
                                privileged_geometry_frame_ids[env_id] = (
                                    0
                                    if privileged_geometry_frame_ids[env_id]
                                    is None
                                    else privileged_geometry_frame_ids[env_id]
                                    + 1
                                )
                                privileged_geometry_poll_counts[env_id] += 1
                                if current_frames[env_id] is None:
                                    raise Stage1AVectorIsaacSmokeError(
                                        "VECTOR_TEACHER_GEOMETRY_CAMERA_REFERENCE_MISSING"
                                    )
                                privileged_geometry_timestamp_s[env_id] = float(
                                    current_frames[env_id].timestamp_s
                                )
                            if (
                                not decisions[env_id].human_grasp_nominal_active
                                or geometry is None
                            ):
                                gate = {
                                    "close_latched": False,
                                    "close_trigger": False,
                                    "persistence_ready_count": 0,
                                    "authority": "PRIVILEGED_GEOMETRY_TEACHER_WAITING",
                                    "student_privileged_input_count": 0,
                                }
                            else:
                                gate = close_gates[env_id].observe(
                                    cube_between_primary_pads=bool(
                                        geometry["cube_between_primary_pads"]
                                    ),
                                    aperture_geometrically_compatible=bool(
                                        geometry[
                                            "aperture_geometrically_compatible"
                                        ]
                                    ),
                                    inner_pad_cube_gap_mm=float(
                                        geometry["inner_pad_cube_gap_mm"]
                                    ),
                                    outer_pad_cube_gap_mm=float(
                                        geometry["outer_pad_cube_gap_mm"]
                                    ),
                                    gripper_aperture_mm=float(
                                        geometry["gripper_aperture_mm"]
                                    ),
                                    cube_effective_width_mm=float(
                                        geometry["cube_effective_width_mm"]
                                    ),
                                    orientation_error_deg=orientation_deg,
                                    no_safety_violation=not bool(
                                        forbidden_before[env_id]
                                    ),
                                )
                                gate = dict(gate) | {
                                    "nominal_grasp_residual_mm": residual_m * 1000.0,
                                    "lateral_alignment_error_mm": lateral_m * 1000.0,
                                    "lateral_ready": None,
                                    "geometry_frame_id": privileged_geometry_frame_ids[
                                        env_id
                                    ],
                                    "geometry_poll_count": privileged_geometry_poll_counts[
                                        env_id
                                    ],
                                    "geometry_timestamp_source": (
                                        "LIVE_25HZ_PRIMARY_PAD_MESH__PAIRED_CAMERA_CAPTURE"
                                    ),
                                    "teacher_geometry_timestamp_s": (
                                        privileged_geometry_timestamp_s[env_id]
                                    ),
                                }
                        elif _uses_privileged_geometry_distillation(runtime_variant):
                            # The exact pad/cube oracle is polled at 25 Hz and
                            # cached across the adjacent control step, but its
                            # output remains *teacher metadata*.  The runtime
                            # gate below is the unchanged CURRENT 15--20 mm /
                            # 15-degree / five-sample state machine.
                            close_latched_before_supervision = bool(
                                close_gates[env_id].close_latched
                            )
                            pre_close_candidate = is_pre_close_candidate(
                                phase=decisions[env_id].phase.value,
                                close_latched_before_supervision=(
                                    close_latched_before_supervision
                                ),
                            )
                            geometry = privileged_geometry_cache[env_id]
                            if (
                                pre_close_candidate
                                and (
                                    vector_step
                                    % PRIVILEGED_GEOMETRY_POLL_CONTROL_STEPS == 0
                                    # V6 is a recording-only timestamp audit.
                                    # Refresh its teacher geometry whenever a
                                    # new source-owned camera frame arrives,
                                    # so a cached older teacher state cannot
                                    # be labelled as same-time evidence.
                                    or (
                                        boundary_paired_plan is not None
                                        and boundary_paired_plan.get(
                                            "v6_diagnostic_receipt"
                                        ) is True
                                        and current_frames[env_id] is not None
                                        and (
                                            privileged_geometry_timestamp_s[env_id]
                                            is None
                                            or not math.isclose(
                                                float(
                                                    privileged_geometry_timestamp_s[
                                                        env_id
                                                    ]
                                                ),
                                                float(
                                                    current_frames[env_id].timestamp_s
                                                ),
                                                rel_tol=0.0,
                                                abs_tol=1.0e-6,
                                            )
                                        )
                                    )
                                )
                            ):
                                geometry = _primary_pad_close_geometry(
                                    env=env,
                                    task_mdp=task_mdp,
                                    state=state,
                                    env_id=env_id,
                                )
                                privileged_geometry_cache[env_id] = geometry
                                privileged_geometry_frame_ids[env_id] = (
                                    0
                                    if privileged_geometry_frame_ids[env_id]
                                    is None
                                    else privileged_geometry_frame_ids[env_id]
                                    + 1
                                )
                                privileged_geometry_poll_counts[env_id] += 1
                                if current_frames[env_id] is None:
                                    raise Stage1AVectorIsaacSmokeError(
                                        "VECTOR_TEACHER_GEOMETRY_CAMERA_REFERENCE_MISSING"
                                    )
                                privileged_geometry_timestamp_s[env_id] = float(
                                    current_frames[env_id].timestamp_s
                                )
                            gate = close_gates[env_id].observe(
                                nominal_grasp_residual_m=residual_m,
                                orientation_error_deg=orientation_deg,
                                no_safety_violation=not bool(forbidden_before[env_id]),
                            )
                            gate = dict(gate) | {
                                "nominal_grasp_residual_mm": residual_m * 1000.0,
                                "lateral_alignment_error_mm": lateral_m * 1000.0,
                                "lateral_ready": None,
                                "geometry_frame_id": privileged_geometry_frame_ids[
                                    env_id
                                ],
                                "geometry_poll_count": privileged_geometry_poll_counts[
                                    env_id
                                ],
                                "teacher_geometry_timestamp_s": (
                                    privileged_geometry_timestamp_s[env_id]
                                ),
                                "teacher_geometry_timestamp_source": (
                                    "LIVE_25HZ_PRIMARY_PAD_MESH__PAIRED_CAMERA_CAPTURE"
                                ),
                                "student_feature_timestamp_s": float(
                                    current_frames[env_id].timestamp_s
                                )
                                if current_frames[env_id] is not None
                                else float("nan"),
                                "gru_reset_generation": int(
                                    states.state(env_id).episode_index
                                ),
                                "student_privileged_input_count": 0,
                                "pre_close_candidate": pre_close_candidate,
                                "close_latched_before_supervision": (
                                    close_latched_before_supervision
                                ),
                            }
                            if (
                                pre_close_candidate
                                and geometry is not None
                            ):
                                teacher = _privileged_close_readiness_teacher_receipt(
                                    geometry=geometry,
                                    orientation_error_deg=orientation_deg,
                                    no_safety_violation=not bool(
                                        forbidden_before[env_id]
                                    ),
                                    pre_close_candidate=pre_close_candidate,
                                    close_latched_before_supervision=(
                                        close_latched_before_supervision
                                    ),
                                    geometry_frame_id=privileged_geometry_frame_ids[
                                        env_id
                                    ],
                                    geometry_poll_count=privileged_geometry_poll_counts[
                                        env_id
                                    ],
                                )
                                # This call reads only the existing frozen-GRU
                                # recurrent student feature.  The returned
                                # score is telemetry and never affects
                                # ``gate['close_trigger']`` or gripper intent.
                                teacher["student_close_ready_score"] = (
                                    coordinator.predict_close_readiness(
                                        proposals[env_id].actor_observation
                                    )
                                    if teacher["privileged_teacher_available"]
                                    else float("nan")
                                )
                                gate.update(teacher)
                            else:
                                gate.update(
                                    {
                                        "privileged_teacher_available": False,
                                        "privileged_close_ready_target": None,
                                        "privileged_close_ready_score": float("nan"),
                                        "student_close_ready_score": float("nan"),
                                        "privileged_close_ready_negative_reasons": (),
                                        "privileged_close_ready_excluded_reason": (
                                            "WAITING_FOR_PRE_CLOSE_GEOMETRY"
                                        ),
                                        "privileged_teacher_authority": "WAITING_FOR_LOCAL_GRASP_GEOMETRY",
                                    }
                                )
                        elif _uses_fsm_sequence_dataset(runtime_variant):
                            # Exact geometry is acquired strictly as a 25-Hz
                            # *sidecar label*.  The simplified CURRENT FSM
                            # below remains the sole OPEN/CLOSE authority.
                            close_latched_before_supervision = bool(
                                close_gates[env_id].close_latched
                            )
                            pre_close_candidate = is_pre_close_candidate(
                                phase=decisions[env_id].phase.value,
                                close_latched_before_supervision=(
                                    close_latched_before_supervision
                                ),
                            )
                            geometry = privileged_geometry_cache[env_id]
                            if (
                                (
                                    _uses_frozen_student_advisory(runtime_variant)
                                    or decisions[env_id].human_grasp_nominal_active
                                )
                                and vector_step
                                % PRIVILEGED_GEOMETRY_POLL_CONTROL_STEPS == 0
                            ):
                                geometry = _primary_pad_close_geometry(
                                    env=env,
                                    task_mdp=task_mdp,
                                    state=state,
                                    env_id=env_id,
                                )
                                privileged_geometry_cache[env_id] = geometry
                                privileged_geometry_frame_ids[env_id] = (
                                    0
                                    if privileged_geometry_frame_ids[env_id]
                                    is None
                                    else privileged_geometry_frame_ids[env_id]
                                    + 1
                                )
                                privileged_geometry_poll_counts[env_id] += 1
                                if current_frames[env_id] is None:
                                    raise Stage1AVectorIsaacSmokeError(
                                        "VECTOR_FSM_SEQUENCE_GEOMETRY_CAMERA_REFERENCE_MISSING"
                                    )
                                privileged_geometry_timestamp_s[env_id] = float(
                                    current_frames[env_id].timestamp_s
                                )
                            # The deterministic distance/orientation FSM is
                            # deliberately evaluated before and independently
                            # from the privileged receipt.
                            gate = close_gates[env_id].observe(
                                nominal_grasp_residual_m=residual_m,
                                orientation_error_deg=orientation_deg,
                                no_safety_violation=not bool(forbidden_before[env_id]),
                            )
                            gate = dict(gate) | {
                                "nominal_grasp_residual_mm": residual_m * 1000.0,
                                "lateral_alignment_error_mm": lateral_m * 1000.0,
                                "height_alignment_error_mm": float(
                                    residual_error_root_m[2] * 1000.0
                                ),
                                "lateral_ready": None,
                                "geometry_frame_id": privileged_geometry_frame_ids[
                                    env_id
                                ],
                                "geometry_poll_count": privileged_geometry_poll_counts[
                                    env_id
                                ],
                                "teacher_geometry_timestamp_s": (
                                    privileged_geometry_timestamp_s[env_id]
                                ),
                                "teacher_geometry_timestamp_source": (
                                    "LIVE_25HZ_PRIMARY_PAD_MESH__PAIRED_WRIST_CAPTURE"
                                ),
                                "student_feature_timestamp_s": float(
                                    current_frames[env_id].timestamp_s
                                )
                                if current_frames[env_id] is not None
                                else float("nan"),
                                "gru_reset_generation": int(
                                    states.state(env_id).episode_index
                                ),
                                "student_privileged_input_count": 0,
                                "pre_close_candidate": pre_close_candidate,
                                "close_latched_before_supervision": (
                                    close_latched_before_supervision
                                ),
                                "authority": "CURRENT_DISTANCE_ALIGNMENT_FSM",
                            }
                            if geometry is not None:
                                gate.update(
                                    _fsm_sequence_teacher_receipt(
                                        geometry=geometry,
                                        orientation_error_deg=orientation_deg,
                                        no_safety_violation=not bool(
                                            forbidden_before[env_id]
                                        ),
                                        pre_close_candidate=pre_close_candidate,
                                        close_latched_before_supervision=(
                                            close_latched_before_supervision
                                        ),
                                        geometry_frame_id=(
                                            privileged_geometry_frame_ids[env_id]
                                        ),
                                        geometry_poll_count=(
                                            privileged_geometry_poll_counts[env_id]
                                        ),
                                    )
                                )
                            else:
                                gate.update(
                                    {
                                        "privileged_teacher_available": False,
                                        "privileged_close_ready_target": None,
                                        "privileged_close_ready_score": float("nan"),
                                        "privileged_close_ready_negative_reasons": (),
                                        "privileged_close_ready_excluded_reason": (
                                            "WAITING_FOR_LOCAL_GRASP_GEOMETRY"
                                        ),
                                        "exact_signed_margin_available": False,
                                        "signed_readiness_margin": None,
                                    }
                                )
                            if _uses_frozen_student_advisory(runtime_variant):
                                # The advisory reads the same current 25-Hz
                                # Wrist frame that belongs to this policy
                                # action.  It receives no teacher field, FSM
                                # field, family id, history, or future frame.
                                wrist_frame = current_frames[env_id]
                                if wrist_frame is None or frozen_student_advisory is None:
                                    raise Stage1AVectorIsaacSmokeError(
                                        "VECTOR_FROZEN_STUDENT_ADVISORY_FRAME_MISSING"
                                    )
                                student_score, advice = (
                                    frozen_student_advisory.score_and_advise(
                                        rgb=wrist_frame.rgb,
                                        depth_m=wrist_frame.depth_m[..., 0],
                                        depth_valid=wrist_frame.depth_valid[..., 0],
                                    )
                                )
                                fsm_close = bool(gate["close_trigger"])
                                # This assertion is the action-authority
                                # contract: advisory cannot veto, trigger, or
                                # otherwise alter canonical CLOSE.
                                final_close = frozen_student_advisory.advisory.preserve_fsm_close(
                                    fsm_close_triggered=fsm_close
                                )
                                if final_close is not fsm_close:
                                    raise Stage1AVectorIsaacSmokeError(
                                        "VECTOR_ADVISORY_ALTERED_FSM_CLOSE"
                                    )
                                ready = advice.value == "READY_ADVISORY"
                                defer = advice.value == "DEFER_TO_FSM"
                                gate.update(
                                    {
                                        "student_advisory_score": student_score,
                                        "student_advice": advice.value,
                                        "student_ready_advisory": ready,
                                        "student_defer_to_fsm": defer,
                                        "fsm_close_triggered_receipt": fsm_close,
                                        "fsm_student_agreement": (
                                            (fsm_close and ready)
                                            or (not fsm_close and defer)
                                        ),
                                        "ready_advisory_fsm_not_close": ready
                                        and not fsm_close,
                                        "fsm_close_student_defer": fsm_close and defer,
                                        "ready_advisory_teacher_false_accept": (
                                            ready
                                            and gate.get(
                                                "privileged_close_ready_target"
                                            )
                                            is False
                                        ),
                                        "student_advisory_authority": (
                                            "FROZEN_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY"
                                        ),
                                        "student_advisory_threshold": (
                                            DEFAULT_HIGH_READY_THRESHOLD
                                        ),
                                    }
                                )
                                if current_gru_privileged is not None:
                                    target = gate.get(
                                        "privileged_close_ready_target"
                                    )
                                    margin = gate.get("signed_readiness_margin")
                                    supervision_eligible = bool(
                                        not reset_fixed_preflight
                                        and
                                        pre_close_candidate
                                        and not close_latched_before_supervision
                                        and gate.get(
                                            "privileged_teacher_available", False
                                        )
                                        and gate.get(
                                            "exact_signed_margin_available", False
                                        )
                                        and target is not None
                                        and margin is not None
                                    )
                                    gru_receipt = current_gru_privileged.observe(
                                        env_id=env_id,
                                        actor_observation=proposals[
                                            env_id
                                        ].actor_observation,
                                        frozen_student_score=student_score,
                                        teacher_target=(
                                            bool(target)
                                            if supervision_eligible
                                            else None
                                        ),
                                        signed_margin=(
                                            float(margin)
                                            if supervision_eligible
                                            else None
                                        ),
                                        supervision_eligible=supervision_eligible,
                                    )
                                    gate.update(
                                        {
                                            "current_gru_privileged_score": gru_receipt.score,
                                            "current_gru_privileged_margin": gru_receipt.predicted_margin,
                                            "current_gru_privileged_trained": gru_receipt.trained,
                                            "current_gru_binary_loss": gru_receipt.binary_loss,
                                            "current_gru_signed_margin_loss": gru_receipt.signed_margin_loss,
                                            "current_gru_loss": gru_receipt.gru_loss,
                                            "current_gru_sequence_steps": gru_receipt.sequence_steps,
                                            "current_gru_runtime_authority": False,
                                            "current_gru_teacher_only": True,
                                            "current_gru_privileged_input_count": 0,
                                        }
                                    )
                        elif runtime_variant == V31_RUNTIME_VARIANT or runtime_variant == V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT:
                            gate = close_gates[env_id].observe(
                                nominal_grasp_residual_m=residual_m,
                                orientation_error_deg=orientation_deg,
                                lateral_alignment_error_m=lateral_m,
                                no_safety_violation=not bool(forbidden_before[env_id]),
                            )
                        else:
                            gate = close_gates[env_id].observe(
                                nominal_grasp_residual_m=residual_m,
                                orientation_error_deg=orientation_deg,
                                no_safety_violation=not bool(forbidden_before[env_id]),
                            )
                            gate = dict(gate) | {
                                "lateral_alignment_error_mm": lateral_m * 1000.0,
                                "lateral_ready": None,
                            }
                        gate_receipts[env_id] = gate
                        if bool(gate["close_trigger"]) and (
                            _is_v31_lateral_off_long_run(runtime_variant)
                            or _uses_privileged_geometry_teacher(runtime_variant)
                        ):
                            geometry = (
                                dict(privileged_geometry_cache[env_id] or {})
                                if _uses_privileged_geometry_teacher(runtime_variant)
                                else _primary_pad_close_geometry(
                                    env=env,
                                    task_mdp=task_mdp,
                                    state=state,
                                    env_id=env_id,
                                )
                            )
                            if not geometry:
                                raise Stage1AVectorIsaacSmokeError(
                                    "VECTOR_PRIVILEGED_CLOSE_GEOMETRY_RECEIPT_MISSING"
                                )
                            # Remove the verbose mesh-path receipt from the
                            # per-episode event while retaining the exact
                            # oracle-derived numerical result and authority
                            # type.  It is diagnostics-only, never an input.
                            geometry_authority = geometry.pop("authority")
                            episode_failure_diagnostics[env_id] = {
                                "schema": "g2_stage1a_close_failure_event_v1",
                                "env_id": env_id,
                                "episode_id": episode_ids[env_id],
                                "close_triggered": True,
                                "close_vector_step": vector_step,
                                "close_control_step": states.state(env_id).control_step,
                                "close_control_timestamp_s": None,
                                "nominal_grasp_residual_mm": float(
                                    gate["nominal_grasp_residual_mm"]
                                ),
                                "orientation_error_deg": float(
                                    gate["orientation_error_deg"]
                                ),
                                "lateral_alignment_error_mm": float(
                                    gate["lateral_alignment_error_mm"]
                                ),
                                "distance_ready_count": int(
                                    gate.get("distance_ready_count", 0)
                                ),
                                "geometry_ready_count": int(
                                    gate.get("geometry_ready_count", 0)
                                ),
                                "orientation_ready_count": int(
                                    gate["orientation_ready_count"]
                                ),
                                "persistence_ready_count": int(
                                    gate["persistence_ready_count"]
                                ),
                                "close_readiness_authority": str(
                                    gate.get("authority", "CURRENT_RULE")
                                ),
                                "geometry_frame_id": gate.get(
                                    "geometry_frame_id"
                                ),
                                "geometry_poll_count": int(
                                    gate.get("geometry_poll_count", 0)
                                ),
                                "student_privileged_input_count": int(
                                    gate.get(
                                        "student_privileged_input_count", 0
                                    )
                                ),
                                "primary_pad_geometry_authority": (
                                    "LIVE_USD_PHYSICS_COLLISION_ENABLED_MESH"
                                ),
                                "primary_pad_geometry_receipt": geometry_authority,
                                **geometry,
                                "first_contact_occurred": False,
                                "first_contact_after_close_ms": None,
                                "bilateral_occurred": False,
                                "first_bilateral_after_close_ms": None,
                                "stable_occurred": False,
                                "first_stable_after_close_ms": None,
                                "contact_loss": False,
                                "bilateral_dwell_steps": 0,
                                "max_slip_m_s": 0.0,
                                "max_relative_velocity_m_s": 0.0,
                                "max_cube_angular_velocity_rad_s": 0.0,
                                "micro_correction_count": 0,
                                "micro_correction_events": [],
                            }
                        close_onset.append(bool(gate["close_trigger"]))
                        if not gate["close_latched"]:
                            applied = proposals[env_id]
                            execution_receipts[env_id] = {
                                "post_close_control_phase": "PRE_CLOSE",
                                "post_close_micro_correction_allowed": False,
                                "post_close_micro_correction_cap_mm": 0.0,
                                "post_close_micro_correction_norm_mm": 0.0,
                                "post_close_micro_correction_cap_hit": False,
                                "approach_axis_component_removed_mm": 0.0,
                                "forward_normal_push_allowed": False,
                            }
                        elif runtime_variant not in POST_CLOSE_MICRO_RUNTIME_VARIANTS:
                            applied = _zero_xyz_close_execution_proposal(
                                proposals[env_id], alpha=coordinator.alpha
                            )
                            execution_receipts[env_id] = {
                                "post_close_control_phase": "V3_FULL_FREEZE",
                                "post_close_micro_correction_allowed": False,
                                "post_close_micro_correction_cap_mm": 0.0,
                                "post_close_micro_correction_norm_mm": 0.0,
                                "post_close_micro_correction_cap_hit": False,
                                "approach_axis_component_removed_mm": 0.0,
                                "forward_normal_push_allowed": False,
                            }
                        else:
                            previous_phase = post_close_control_phase[env_id]
                            cap_m = (
                                POST_CLOSE_SINGLE_CONTACT_MICRO_CAP_M
                                if previous_phase == "SINGLE_CONTACT"
                                else POST_CLOSE_BILATERAL_MICRO_CAP_M
                                if previous_phase == "BILATERAL"
                                else 0.0
                            )
                            applied, execution_receipts[env_id] = (
                                _post_close_micro_execution_proposal(
                                    proposals[env_id],
                                    alpha=coordinator.alpha,
                                    cap_m=cap_m,
                                    control_phase=(
                                        previous_phase
                                        if previous_phase != "PRE_CLOSE"
                                        else "CLOSE_TO_FIRST_CONTACT"
                                    ),
                                )
                            )
                        executed_proposals[env_id] = applied
                        final = tuple(float(v) for v in applied.composition.final_action_4d_metric_root_m)
                        intent = AbstractGripperIntent.CLOSE if gate["close_latched"] else AbstractGripperIntent.OPEN
                        if float(np.linalg.norm(final[:3])) > MAX_FINAL_ACTION_M + 1e-12:
                            raise Stage1AVectorIsaacSmokeError("VECTOR_FINAL_ACTION_BOUND_VIOLATION")
                    rows.append(PerEnvCanonicalAction(env_id=env_id, final_action_4d_metric_root_m=final, gripper_intent=intent))
                if (
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("v6_diagnostic_receipt") is True
                ):
                    if v6_camera_pose_resolver is None:
                        raise Stage1AVectorIsaacSmokeError(
                            "VECTOR_V6_CAMERA_POSE_RESOLVER_MISSING"
                        )
                    v6_env_ids = tuple(
                        env_id
                        for env_id in range(num_envs)
                        if bool(gate_receipts[env_id].get("pre_close_candidate", False))
                        and not bool(
                            gate_receipts[env_id].get(
                                "close_latched_before_supervision", False
                            )
                        )
                        and gate_receipts[env_id].get(
                            "privileged_close_ready_target"
                        ) is not None
                        and current_frames[env_id] is not None
                    )
                    if v6_env_ids:
                        captured_v6 = capture_v6_wrist_diagnostic_frames(
                            env=env,
                            frames={
                                env_id: current_frames[env_id]
                                for env_id in v6_env_ids
                            },
                            world_from_right_wrist=(
                                v6_camera_pose_resolver.world_transform("right_wrist")
                            ),
                            cube_position_world_m=state["cube_position_world_m"],
                            cube_quaternion_world_xyzw=state[
                                "cube_quaternion_world_xyzw"
                            ],
                        )
                        if len(captured_v6) != len(v6_env_ids):
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_V6_DIAGNOSTIC_CARDINALITY_INVALID"
                            )
                        v6_pre_action_frames = dict(zip(v6_env_ids, captured_v6, strict=True))
                packet = packet_port.stage(rows)
                starts = telemetry.begin_packet(
                    policy_step=vector_step,
                    phase_by_env=tuple(
                        "RESET_OPEN_RESTORE"
                        if pending_reset[env_id] or decisions[env_id] is None
                        else decisions[env_id].phase.value
                        for env_id in range(num_envs)
                    ),
                    gripper_intent_by_env=tuple(
                        row.gripper_intent.value for row in rows
                    ),
                    close_onset_by_env=tuple(close_onset),
                )
                outputs, consumption = consume_per_env_packet_once(env=env, counter=counter, port=packet_port, packet=packet, label=f"VECTOR_STEP_{vector_step:07d}")
                records_by_env = telemetry.end_packet(starts)
                if not consumption.single_batched_consumption or consumption.action_broadcast_detected:
                    raise Stage1AVectorIsaacSmokeError("VECTOR_CANONICAL_CONSUMPTION_FAILED")
                independently_owned = tuple(row.env_id for row in rows) == tuple(range(num_envs))
                if not independently_owned:
                    raise Stage1AVectorIsaacSmokeError("VECTOR_PER_ENV_ACTION_OWNERSHIP_INVALID")
                action_packet_evidence.append(
                    {
                        "vector_step": vector_step,
                        "unique_4d_action_rows": consumption.unique_4d_action_rows,
                        "unique_8d_packet_rows": consumption.unique_8d_packet_rows,
                        "independently_owned": independently_owned,
                        "single_batched_consumption": consumption.single_batched_consumption,
                    }
                )
                terminated = outputs[2].detach().to("cpu").numpy().astype(bool).reshape(-1)
                truncated = outputs[3].detach().to("cpu").numpy().astype(bool).reshape(-1)
                if terminated.shape != (num_envs,) or truncated.shape != (num_envs,):
                    raise Stage1AVectorIsaacSmokeError("VECTOR_TERMINATION_SHAPE_INVALID")
                next_state = vector_ee_and_cube_state(env=env, p0a=p0a)
                forbidden_after = _task_bool(task_mdp, env, "forbidden")
                scalar_inputs = []
                for env_id in range(num_envs):
                    final_xyz = np.asarray(rows[env_id].final_action_4d_metric_root_m[:3], dtype=np.float64)
                    executed = executed_proposals[env_id]
                    effective = (
                        np.asarray(
                            executed.composition.scaled_residual_contribution_m,
                            dtype=np.float64,
                        )
                        if executed is not None
                        else np.zeros(3, dtype=np.float64)
                    )
                    current_residual_m = float(
                        np.linalg.norm(
                            nominal[env_id, :3]
                            - state["ee_position_root_m"][env_id]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    # Reward phase authority is the actual current state of
                    # this 50-Hz transition, not the prior action proposal's
                    # cached routing decision.  It uses the same canonical
                    # 15--22 mm contract without changing any threshold.
                    reset_restore_active = bool(pending_reset[env_id])
                    reward_grasp_decision_phase = bool(
                        not reset_restore_active
                        and grasp_band_near_m - 1.0e-6
                        <= current_residual_m
                        <= grasp_band_far_m + 1.0e-6
                    )
                    runtime_state = Stage1ARuntimeTransitionState(
                        current_ee_position_root_m=state["ee_position_root_m"][env_id].detach().cpu().numpy(), next_ee_position_root_m=next_state["ee_position_root_m"][env_id].detach().cpu().numpy(),
                        nominal_grasp_position_root_m=nominal[env_id, :3], nominal_approach_axis_root=(1.0, 0.0, 0.0),
                        # OPEN restore is deliberately outside the episode
                        # reward/milestone clock.  Resetting these reward
                        # fields for every held OPEN packet prevents a
                        # partially closed inherited four-bar from creating a
                        # contact milestone before policy activation.
                        stage1a_active=not reset_restore_active, grasp_decision_phase=reward_grasp_decision_phase, phase_reset=reset_restore_active or states.state(env_id).control_step == 0, episode_reset=reset_restore_active or states.state(env_id).control_step == 0,
                        root_position_world_m_by_substep=np.stack([r["root_position_world_m"] for r in records_by_env[env_id]]), root_quat_world_xyzw_by_substep=np.stack([r["root_quat_world_xyzw"] for r in records_by_env[env_id]]),
                        root_linear_velocity_world_m_s_by_substep=np.stack([r["root_linear_velocity_world_m_s"] for r in records_by_env[env_id]]), root_angular_velocity_world_rad_s_by_substep=np.stack([r["root_angular_velocity_world_rad_s"] for r in records_by_env[env_id]]),
                        cube_center_world_m=next_state["cube_position_world_m"][env_id].detach().cpu().numpy(), cube_quat_world_xyzw=next_state["cube_quaternion_world_xyzw"][env_id].detach().cpu().numpy(), cube_linear_velocity_world_m_s=next_state["cube_linear_velocity_world_m_s"][env_id].detach().cpu().numpy(), cube_angular_velocity_world_rad_s=next_state["cube_angular_velocity_world_rad_s"][env_id].detach().cpu().numpy(), cube_half_extents_m=task_mdp.TASK.cube_half_extents_m,
                        effective_residual_root_m=effective, previous_effective_residual_root_m=previous_residual[env_id], final_action_xyz_root_m=final_xyz,
                        contract_grasp_state_valid=True, safety_violation=bool(forbidden_after[env_id]), authoritative_safety_penalty=0.0,
                        non_pad_gripper_cube_contact=bool(forbidden_after[env_id]), close_command=rows[env_id].gripper_intent is AbstractGripperIntent.CLOSE,
                    )
                    scalar_inputs.append(build_stage1a_reward_inputs(records_by_env[env_id], runtime_state, contact_partner_actor_paths=telemetry.contact_partner_paths(env_id), device=env.device))
                reward_step = reward.step(stack_single_env_reward_inputs(scalar_inputs))
                # Long-run safety is fail-closed.  The diagnostic-only qdd
                # threshold is deliberately absent here; only the existing
                # authoritative runtime hard-stop and forbidden-collision
                # signals may halt training.
                for name, value in vars(reward_step).items():
                    if isinstance(value, torch.Tensor) and not bool(
                        torch.isfinite(value.to(dtype=torch.float32)).all().item()
                    ):
                        _atomic_json(
                            output_dir / "FAIL_CLOSED.json",
                            {
                                "schema": VECTOR_RUNTIME_SCHEMA,
                                "reason": "NONFINITE_REWARD_TELEMETRY",
                                "field": name,
                                "accepted_transitions": coordinator.accepted_transitions,
                            },
                        )
                        raise Stage1AVectorIsaacSmokeError(
                            f"VECTOR_NONFINITE_REWARD_TELEMETRY:{name}"
                        )
                hardstop_envs = [
                    env_id
                    for env_id in range(num_envs)
                    if reward_step.runtime_hardstop is not None
                    and bool(reward_step.runtime_hardstop[env_id].item())
                ]
                forbidden_envs = [
                    env_id
                    for env_id in range(num_envs)
                    if bool(forbidden_after[env_id])
                ]
                if hardstop_envs or forbidden_envs:
                    _atomic_json(
                        output_dir / "FAIL_CLOSED.json",
                        {
                            "schema": VECTOR_RUNTIME_SCHEMA,
                            "reason": (
                                "RUNTIME_HARDSTOP"
                                if hardstop_envs else "FORBIDDEN_COLLISION"
                            ),
                            "hardstop_env_ids": hardstop_envs,
                            "forbidden_collision_env_ids": forbidden_envs,
                            "accepted_transitions": coordinator.accepted_transitions,
                            "diagnostic_only_qdd_gate_used": False,
                        },
                    )
                    raise Stage1AVectorIsaacSmokeError(
                        "VECTOR_AUTHORITATIVE_SAFETY_FAIL_CLOSED"
                    )
                for env_id in range(num_envs):
                    before = last_contact_quality[env_id]
                    contact_after = bool(
                        reward_step.left_contact_boolean[env_id].item()
                        or reward_step.right_contact_boolean[env_id].item()
                    )
                    after = {
                        "contact": contact_after,
                        "slip_m_s": float(reward_step.tangential_slip_m_s[env_id].item()),
                        "relative_velocity_m_s": float(
                            reward_step.contact_normal_velocity_m_s[env_id].item()
                        ),
                        "cube_omega_rad_s": float(
                            reward_step.cube_angular_velocity_rad_s[env_id].item()
                        ),
                    }
                    receipt = execution_receipts[env_id]
                    micro_applied = float(
                        receipt.get("post_close_micro_correction_norm_mm", 0.0)
                    ) > 0.0
                    if micro_applied and before is not None:
                        contact_retained = bool(before["contact"] and after["contact"])
                        improved = bool(
                            after["slip_m_s"] < before["slip_m_s"] - 1.0e-9
                            or after["relative_velocity_m_s"]
                            < before["relative_velocity_m_s"] - 1.0e-9
                            or after["cube_omega_rad_s"]
                            < before["cube_omega_rad_s"] - 1.0e-9
                        )
                        adverse = bool(
                            not contact_retained
                            or (
                                after["slip_m_s"] > before["slip_m_s"] + 1.0e-9
                                and after["relative_velocity_m_s"]
                                > before["relative_velocity_m_s"] + 1.0e-9
                                and after["cube_omega_rad_s"]
                                > before["cube_omega_rad_s"] + 1.0e-9
                            )
                        )
                        receipt.update(
                            {
                                "micro_slip_before_m_s": before["slip_m_s"],
                                "micro_slip_after_m_s": after["slip_m_s"],
                                "micro_relative_velocity_before_m_s": before["relative_velocity_m_s"],
                                "micro_relative_velocity_after_m_s": after["relative_velocity_m_s"],
                                "micro_cube_omega_before_rad_s": before["cube_omega_rad_s"],
                                "micro_cube_omega_after_rad_s": after["cube_omega_rad_s"],
                                "micro_contact_retained": contact_retained,
                                "micro_correction_quality": (
                                    "HELPFUL" if contact_retained and improved
                                    else "ADVERSE" if adverse else "NEUTRAL"
                                ),
                            }
                        )
                    else:
                        receipt["micro_correction_quality"] = (
                            "NOT_APPLIED" if not micro_applied else "INSUFFICIENT_HISTORY"
                        )
                    diagnostic = episode_failure_diagnostics[env_id]
                    if diagnostic is not None:
                        # The trigger occurs before the packet; bind its
                        # timestamp to the first completed 500-Hz window.
                        # This remains a logged physical outcome, not an
                        # input to the policy or CLOSE admission condition.
                        if diagnostic["close_control_timestamp_s"] is None:
                            diagnostic["close_control_timestamp_s"] = float(
                                records_by_env[env_id][0]["physics_timestamp_s"]
                            )
                        if contact_after and not diagnostic["first_contact_occurred"]:
                            diagnostic["first_contact_occurred"] = True
                            diagnostic["first_contact_after_close_ms"] = float(
                                1000.0
                                * (
                                    float(records_by_env[env_id][-1]["physics_timestamp_s"])
                                    - float(diagnostic["close_control_timestamp_s"])
                                )
                            )
                        bilateral_after = bool(
                            reward_step.bilateral_contact_boolean[env_id].item()
                        )
                        diagnostic["bilateral_occurred"] = bool(
                            diagnostic["bilateral_occurred"] or bilateral_after
                        )
                        if (
                            bilateral_after
                            and diagnostic["first_bilateral_after_close_ms"] is None
                        ):
                            diagnostic["first_bilateral_after_close_ms"] = float(
                                1000.0
                                * (
                                    float(records_by_env[env_id][-1]["physics_timestamp_s"])
                                    - float(diagnostic["close_control_timestamp_s"])
                                )
                            )
                        diagnostic["stable_occurred"] = bool(
                            diagnostic["stable_occurred"]
                            or bool(reward_step.stable_grasp_boolean[env_id].item())
                        )
                        if (
                            bool(reward_step.stable_grasp_boolean[env_id].item())
                            and diagnostic["first_stable_after_close_ms"] is None
                        ):
                            diagnostic["first_stable_after_close_ms"] = float(
                                1000.0
                                * (
                                    float(records_by_env[env_id][-1]["physics_timestamp_s"])
                                    - float(diagnostic["close_control_timestamp_s"])
                                )
                            )
                        diagnostic["contact_loss"] = bool(
                            diagnostic["contact_loss"]
                            or bool(reward_step.contact_loss_fail[env_id].item())
                        )
                        diagnostic["bilateral_dwell_steps"] = int(
                            diagnostic["bilateral_dwell_steps"]
                            + int(bilateral_after)
                        )
                        diagnostic["max_slip_m_s"] = max(
                            float(diagnostic["max_slip_m_s"]), after["slip_m_s"]
                        )
                        diagnostic["max_relative_velocity_m_s"] = max(
                            float(diagnostic["max_relative_velocity_m_s"]),
                            after["relative_velocity_m_s"],
                        )
                        diagnostic["max_cube_angular_velocity_rad_s"] = max(
                            float(diagnostic["max_cube_angular_velocity_rad_s"]),
                            after["cube_omega_rad_s"],
                        )
                        if micro_applied:
                            diagnostic["micro_correction_count"] = int(
                                diagnostic["micro_correction_count"] + 1
                            )
                            diagnostic["micro_correction_events"].append(
                                {
                                    "norm_mm": float(
                                        receipt["post_close_micro_correction_norm_mm"]
                                    ),
                                    "quality": str(
                                        receipt["micro_correction_quality"]
                                    ),
                                    "slip_before_m_s": receipt.get(
                                        "micro_slip_before_m_s"
                                    ),
                                    "slip_after_m_s": receipt.get(
                                        "micro_slip_after_m_s"
                                    ),
                                    "relative_velocity_before_m_s": receipt.get(
                                        "micro_relative_velocity_before_m_s"
                                    ),
                                    "relative_velocity_after_m_s": receipt.get(
                                        "micro_relative_velocity_after_m_s"
                                    ),
                                }
                            )
                    last_contact_quality[env_id] = after
                if runtime_variant in POST_CLOSE_MICRO_RUNTIME_VARIANTS:
                    # Record only the completed physical state.  These labels
                    # guide the *next* capped command but never enter either
                    # student observation or GRU/BC input tensors.
                    for env_id in range(num_envs):
                        post_close_control_phase[env_id] = (
                            _v31_observed_post_close_phase(
                                gate=gate_receipts[env_id],
                                reward_step=reward_step,
                                env_id=env_id,
                            )
                        )
                # Isaac may restart a camera's frame/timestamp epoch when an
                # individual clone terminates.  Its next acquired 25-Hz frame
                # must start a new per-episode cache; comparing it with the
                # old episode's timestamp is neither a temporal-contract
                # violation nor a reason to affect the other nine clones.
                terminal_camera_resets = tuple(
                    env_id
                    for env_id in range(num_envs)
                    if bool(reward_step.done[env_id].item())
                    or bool(terminated[env_id])
                    or bool(truncated[env_id])
                )
                for env_id in terminal_camera_resets:
                    cameras.reset(env_id)
                    head_cameras.reset(env_id)
                next_inputs, next_frames = capture_vector_wrist_gru_inputs(
                    env=env,
                    p0a=p0a,
                    previous_actions_4d_metric_root_m=np.asarray(
                        [row.final_action_4d_metric_root_m for row in rows],
                        dtype=np.float32,
                    ),
                    # The first input handed to a just-restored clone carries
                    # the same reset generation used to clear its coordinator
                    # state below.  This is causal reset metadata, never a
                    # privileged geometry input.
                    hidden_reset_mask=[
                        bool(pending_reset[env_id]) for env_id in range(num_envs)
                    ],
                    camera_cache=cameras,
                )
                next_head_frames: list[Any | None] = list(
                    capture_vector_rgbd_sensor_frames(
                        env=env,
                        camera_scene_key="head_camera",
                        camera_cache=head_cameras,
                    )
                ) if sequence_writer is not None else [None for _ in range(num_envs)]
                reset_ids: list[int] = []
                for env_id in range(num_envs):
                    # A physical vector step is always a full [N, 8] packet,
                    # but the requested training budget counts accepted replay
                    # rows exactly.  When a clone-local reset makes the target
                    # non-divisible by N, explicitly leave the harmless tail
                    # observations out of replay rather than silently
                    # overshooting the approved transition budget.
                    if coordinator.accepted_transitions >= accepted_transition_target:
                        tail_env_steps_not_accepted += 1
                        continue
                    reward_done = bool(reward_step.done[env_id].item())
                    runtime_terminated = bool(terminated[env_id])
                    runtime_truncated = bool(truncated[env_id])
                    terminal_event = bool(reward_done or runtime_terminated or runtime_truncated)
                    if pending_reset[env_id]:
                        # A measured OPEN restoration is not a policy/replay
                        # transition.  Do not silently retry a terminal reset:
                        # an inherited closed four-bar that contacts before it
                        # can open is precisely the condition this contract is
                        # meant to expose.
                        if terminal_event:
                            failure_receipt = {
                                "schema": RESET_OPEN_RESTORE_SCHEMA,
                                "env_id": int(env_id),
                                "episode_id": int(episode_ids[env_id]),
                                "source_sample_id": active_initial_samples[
                                    env_id
                                ].sample_id,
                                "open_restore_pass": False,
                                "failure_reason": "TERMINATED_DURING_OPEN_RESTORE",
                                "reward_done": reward_done,
                                "runtime_terminated": runtime_terminated,
                                "runtime_truncated": runtime_truncated,
                            }
                            reset_open_restore_receipts.append(failure_receipt)
                            reset_open_restore_stream.write(
                                json.dumps(failure_receipt, sort_keys=True) + "\n"
                            )
                            _atomic_json(
                                output_dir / "FAIL_CLOSED.json",
                                {
                                    "schema": VECTOR_RUNTIME_SCHEMA,
                                    "reason": "RESET_OPEN_RESTORE_TERMINATED",
                                    "accepted_transitions": coordinator.accepted_transitions,
                                    "receipt": failure_receipt,
                                },
                            )
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_RESET_OPEN_RESTORE_TERMINATED"
                            )
                        frame = next_frames[env_id]
                        geometry: dict[str, Any] | None = None
                        aperture_mm: float | None = None
                        owner_valid = False
                        geometry_valid = False
                        geometry_frame_id: int | None = None
                        geometry_timestamp_s: float | None = None
                        geometry_age_ms: float | None = None
                        camera_watermark = reset_camera_watermarks[env_id]
                        camera_frame_changed_since_restore = False
                        reset_geometry_receipt: dict[str, Any] = {}
                        if frame is not None:
                            geometry = _primary_pad_close_geometry(
                                env=env,
                                task_mdp=task_mdp,
                                state=next_state,
                                env_id=env_id,
                            )
                            raw_aperture_mm = geometry.get("gripper_aperture_mm")
                            aperture_mm = (
                                float(raw_aperture_mm)
                                if raw_aperture_mm is not None
                                else None
                            )
                            owner_valid = bool(geometry.get("owner_valid", True))
                            geometry_required = (
                                "inner_pad_cube_gap_mm",
                                "outer_pad_cube_gap_mm",
                                "gripper_aperture_mm",
                                "cube_effective_width_mm",
                            )
                            geometry_complete = all(
                                name in geometry
                                and geometry[name] is not None
                                and math.isfinite(float(geometry[name]))
                                for name in geometry_required
                            )
                            geometry_valid = bool(
                                geometry.get("geometry_valid", True)
                            ) and geometry_complete
                            geometry_frame_id = int(frame.frame_id)
                            geometry_timestamp_s = float(frame.timestamp_s)
                            # Camera and physics clocks may have different
                            # origins.  Use the camera's own measured age
                            # rather than subtracting unlike clocks.
                            _capture_times, camera_ages_s = (
                                camera_capture_time_and_age(
                                    env.scene["right_wrist_camera"]
                                )
                            )
                            geometry_age_ms = 1000.0 * float(
                                camera_ages_s[env_id].item()
                            )
                            camera_frame_changed_since_restore = bool(
                                camera_watermark is None
                                or (
                                    int(frame.frame_id),
                                    float(frame.timestamp_s),
                                )
                                != (
                                    int(camera_watermark[0]),
                                    float(camera_watermark[1]),
                                )
                            )
                            if geometry_valid:
                                reset_orientation_deg = _orientation_error_deg(
                                    nominal[env_id, 3:],
                                    next_state["ee_quaternion_root_xyzw"][env_id]
                                    .detach()
                                    .cpu()
                                    .numpy(),
                                )
                                reset_margin = build_signed_readiness_margin(
                                    pad_containment_margin_mm=geometry.get(
                                        "primary_pad_containment_margin_mm"
                                    ),
                                    minimum_primary_pad_cube_gap_mm=min(
                                        float(geometry["inner_pad_cube_gap_mm"]),
                                        float(geometry["outer_pad_cube_gap_mm"]),
                                    ),
                                    gripper_aperture_mm=float(
                                        geometry["gripper_aperture_mm"]
                                    ),
                                    cube_effective_width_mm=float(
                                        geometry["cube_effective_width_mm"]
                                    ),
                                    orientation_error_deg=float(reset_orientation_deg),
                                    owner_valid=owner_valid,
                                    geometry_valid=geometry_valid,
                                    no_safety_violation=not bool(
                                        forbidden_after[env_id]
                                    ),
                                )
                                # A geometry receipt without its exact signed
                                # containment authority is incomplete for the
                                # reset parity contract.  This does *not*
                                # require CLOSE readiness: a negative signed
                                # margin remains a valid fresh receipt.
                                geometry_valid = bool(
                                    geometry_valid and reset_margin.recordable
                                )
                                reset_geometry_receipt = {
                                    "inner_pad_cube_gap_mm": float(
                                        geometry["inner_pad_cube_gap_mm"]
                                    ),
                                    "outer_pad_cube_gap_mm": float(
                                        geometry["outer_pad_cube_gap_mm"]
                                    ),
                                    "minimum_primary_pad_cube_gap_mm": min(
                                        float(geometry["inner_pad_cube_gap_mm"]),
                                        float(geometry["outer_pad_cube_gap_mm"]),
                                    ),
                                    "cube_effective_width_mm": float(
                                        geometry["cube_effective_width_mm"]
                                    ),
                                    "orientation_error_deg": float(
                                        reset_orientation_deg
                                    ),
                                    "signed_readiness_margin": (
                                        reset_margin.aggregate_margin
                                    ),
                                    "exact_signed_margin_available": bool(
                                        reset_margin.recordable
                                    ),
                                }
                        measured_q = torch.as_tensor(
                            robot.data.joint_pos, device=env.device
                        )
                        measured_qd = torch.as_tensor(
                            robot.data.joint_vel, device=env.device
                        )
                        passive_q = {
                            name: float(measured_q[env_id, joint_index].item())
                            for name, joint_index in zip(
                                G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
                                right_passive_joint_indices,
                                strict=True,
                            )
                        }
                        passive_qd = {
                            name: float(measured_qd[env_id, joint_index].item())
                            for name, joint_index in zip(
                                G2_RIGHT_HAND_PASSIVE_OR_MIMIC_JOINT_NAMES,
                                right_passive_joint_indices,
                                strict=True,
                            )
                        }
                        progress = reset_open_restore_progress[env_id]
                        open_receipt = progress.observe(
                            master_q_rad=float(
                                measured_q[env_id, right_master_joint_index].item()
                            ),
                            master_qd_rad_s=float(
                                measured_qd[env_id, right_master_joint_index].item()
                            ),
                            passive_q_rad_by_name=passive_q,
                            passive_qd_rad_s_by_name=passive_qd,
                            aperture_mm=aperture_mm,
                            geometry_valid=geometry_valid,
                            owner_valid=owner_valid,
                            geometry_frame_id=geometry_frame_id,
                            geometry_timestamp_s=geometry_timestamp_s,
                            geometry_age_ms=geometry_age_ms,
                            geometry_cache_fresh=camera_frame_changed_since_restore,
                        )
                        open_receipt["camera_watermark_before_source_restore"] = (
                            None
                            if camera_watermark is None
                            else {
                                "frame_id": int(camera_watermark[0]),
                                "timestamp_s": float(camera_watermark[1]),
                            }
                        )
                        open_receipt[
                            "camera_frame_changed_since_source_restore"
                        ] = bool(camera_frame_changed_since_restore)
                        open_receipt[
                            "camera_raw_identity_changed_since_source_restore"
                        ] = bool(camera_frame_changed_since_restore)
                        open_receipt[
                            "camera_cache_generation_after_source_restore"
                        ] = cameras.reset_generation(env_id)
                        open_receipt["camera_epoch_transition_count"] = (
                            cameras.epoch_transition_count(env_id)
                        )
                        open_receipt["camera_epoch_transition_allowed"] = (
                            cameras.reset_epoch_transition_allowed(env_id)
                        )
                        open_receipt[
                            "camera_cache_cleared_after_source_restore"
                        ] = True
                        open_receipt["previous_episode_geometry_used"] = not bool(
                            camera_frame_changed_since_restore
                        )
                        open_receipt.update(reset_geometry_receipt)
                        if progress.completed:
                            # The reset-only camera epoch allowance closes
                            # before this receipt is persisted and before the
                            # clone is permitted to enter policy ACTIVE.
                            cameras.seal_reset_epoch(env_id)
                            head_cameras.seal_reset_epoch(env_id)
                            open_receipt[
                                "camera_epoch_transition_allowed"
                            ] = False
                        reset_open_contact = bool(
                            any(
                                int(record["raw_inner"]["count"]) > 0
                                or int(record["raw_outer"]["count"]) > 0
                                for record in records_by_env[env_id]
                            )
                        )
                        open_receipt["primary_pad_contact_during_open_restore"] = (
                            reset_open_contact
                        )
                        open_receipt["first_contact_nominal_residual_mm"] = (
                            float(
                                np.linalg.norm(
                                    nominal[env_id, :3]
                                    - next_state["ee_position_root_m"][env_id]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            )
                            * 1000.0
                            if reset_open_contact
                            else None
                        )
                        reset_open_restore_receipts.append(open_receipt)
                        reset_open_restore_stream.write(
                            json.dumps(open_receipt, sort_keys=True) + "\n"
                        )
                        if reset_open_contact:
                            _atomic_json(
                                output_dir / "FAIL_CLOSED.json",
                                {
                                    "schema": VECTOR_RUNTIME_SCHEMA,
                                    "reason": "RESET_OPEN_RESTORE_PREMATURE_CONTACT",
                                    "accepted_transitions": coordinator.accepted_transitions,
                                    "receipt": open_receipt,
                                },
                            )
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_RESET_OPEN_RESTORE_PREMATURE_CONTACT"
                            )
                        if progress.expired:
                            _atomic_json(
                                output_dir / "FAIL_CLOSED.json",
                                {
                                    "schema": VECTOR_RUNTIME_SCHEMA,
                                    "reason": "RESET_OPEN_RESTORE_TIMEOUT",
                                    "accepted_transitions": coordinator.accepted_transitions,
                                    "receipt": open_receipt,
                                },
                            )
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_RESET_OPEN_RESTORE_TIMEOUT"
                            )
                        if not progress.completed:
                            continue
                        # Re-enter the policy only after measured parity.  All
                        # episode-local recurrent/cache/latch state is reset
                        # *after* the OPEN hold so no reset-era observation can
                        # affect the first policy action or teacher row.
                        activation_mask = torch.zeros(
                            num_envs, dtype=torch.bool, device=env.device
                        )
                        activation_mask[env_id] = True
                        reward.reset(activation_mask)
                        telemetry.reset_envs((env_id,))
                        state_adapter.reset(env_id)
                        states.reset(env_id)
                        if current_gru_privileged is not None:
                            current_gru_privileged.reset(env_id)
                        phase_routers[env_id].reset()
                        close_gates[env_id] = new_close_gate()
                        privileged_geometry_cache[env_id] = None
                        privileged_geometry_frame_ids[env_id] = None
                        privileged_geometry_poll_counts[env_id] = 0
                        privileged_geometry_timestamp_s[env_id] = None
                        post_close_control_phase[env_id] = "PRE_CLOSE"
                        last_contact_quality[env_id] = None
                        episode_failure_diagnostics[env_id] = None
                        previous_action[env_id] = 0.0
                        previous_residual[env_id] = 0.0
                        # The immediately preceding OPEN packet has now
                        # supplied both joint and fresh geometry evidence.
                        # Re-enter this one environment's recurrent policy
                        # without consuming a replay transition, leaving all
                        # other clones unchanged.
                        next_distance = float(np.linalg.norm(nominal[env_id, :3] - next_state["ee_position_root_m"][env_id].detach().cpu().numpy()))
                        first_active_inputs = replace(
                            next_inputs[env_id],
                            hidden_reset_mask=torch.ones_like(
                                next_inputs[env_id].hidden_reset_mask
                            ),
                        )
                        next_proposal, next_decision, next_owner, retreat = propose(
                            env_id, first_active_inputs, next_distance
                        )
                        if retreat or next_proposal is None or next_decision is None:
                            reset_ids.append(env_id)
                            continue
                        proposals[env_id], decisions[env_id], owners[env_id] = next_proposal, next_decision, next_owner
                        pending_reset[env_id] = False
                        continue
                    if runtime_terminated and not reward_done:
                        # The coordinator intentionally owns ``terminated``
                        # through Stage-1A reward.done.  An environment-only
                        # stop (for example a task-level reset condition) is
                        # neither a success nor a valid Stage-1A failure row,
                        # so including it would silently corrupt replay
                        # terminal semantics.  This is equivalent in spirit
                        # to the existing post-handoff-retreat discard: reset
                        # only this clone and preserve the other nine.
                        if _is_v31_lateral_off_long_run(runtime_variant):
                            finalize_failure_event(
                                env_id=env_id,
                                reason=_lateral_off_failure_taxonomy(
                                    gate=gate_receipts[env_id],
                                    reward_step=reward_step,
                                    env_id=env_id,
                                    reward=reward,
                                ),
                                terminal_kind="RUNTIME_UNATTRIBUTED_TERMINATION",
                                terminal_timestamp_s=float(
                                    records_by_env[env_id][-1]["physics_timestamp_s"]
                                ),
                                replay_eligible=False,
                            )
                        runtime_unattributed_termination_discarded_env_steps += 1
                        reset_ids.append(env_id)
                        continue
                    # The environment can already have reset its camera
                    # stream by the time ``next_frames`` is acquired.  The
                    # terminal row remains part of the episode that produced
                    # ``rows[env_id]`` and must therefore retain this action's
                    # pre-step frame.  The bootstrap observation is ignored
                    # by the terminal replay row, so reuse its causal policy
                    # observation rather than crossing an episode boundary.
                    if terminal_event:
                        terminal_frame = current_frames[env_id]
                        if terminal_frame is None:
                            raise Stage1AVectorIsaacSmokeError(
                                "VECTOR_TERMINAL_CAMERA_FRAME_MISSING"
                            )
                        terminal_distance = float(
                            np.linalg.norm(
                                nominal[env_id, :3]
                                - state["ee_position_root_m"][env_id]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        )
                        identity = PerEnvReplayIdentity(env_id=env_id, episode_id=f"episode-{episode_ids[env_id]:05d}", step_in_episode=states.state(env_id).control_step)
                        terminal_timestamp = float(records_by_env[env_id][-1]["physics_timestamp_s"])
                        camera_age = terminal_timestamp - float(terminal_frame.timestamp_s)
                        if camera_age < -1e-6:
                            raise Stage1AVectorIsaacSmokeError("VECTOR_CAMERA_AFTER_PHYSICS")
                        executed = executed_proposals[env_id]
                        if executed is None:
                            raise Stage1AVectorIsaacSmokeError("VECTOR_EXECUTED_PROPOSAL_MISSING")
                        failure_reason = (
                            _lateral_off_failure_taxonomy(
                                gate=gate_receipts[env_id], reward_step=reward_step,
                                env_id=env_id, reward=reward,
                            )
                            if _is_v31_lateral_off_variant(runtime_variant)
                            else
                            _v31_failure_taxonomy(
                                gate=gate_receipts[env_id], reward_step=reward_step,
                                env_id=env_id, reward=reward,
                            )
                            if runtime_variant
                            in (V31_RUNTIME_VARIANT, V31_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT)
                            else _v32_failure_taxonomy(
                                gate=gate_receipts[env_id], reward_step=reward_step,
                                env_id=env_id, reward=reward,
                            )
                            if _is_v32_variant(runtime_variant)
                            else "NONE"
                        )
                        close_diagnostic = finalize_failure_event(
                            env_id=env_id,
                            reason=failure_reason,
                            terminal_kind="STAGE1A_REWARD_TERMINAL",
                            terminal_timestamp_s=terminal_timestamp,
                            replay_eligible=True,
                        )
                        replay_teacher_target, replay_teacher_score, replay_teacher_reasons = _replay_teacher_fields(runtime_variant=runtime_variant, gate=gate_receipts[env_id])
                        learner_metrics = state_adapter.call(env_id, "accept_real_transition", Stage1AAcceptedRealRow(row_id=identity.row_id, proposal=executed, next_actor_observation=executed.actor_observation, reward_step=reward_step, reward_env_index=env_id, safety_pass=not bool(forbidden_after[env_id]), lift=False, terminated=reward_done, truncated=bool(runtime_truncated and not reward_done), episode_id=identity.episode_id, phase=decisions[env_id].phase.value, failure_reason=failure_reason, contact=bool(reward_step.left_contact_boolean[env_id] or reward_step.right_contact_boolean[env_id]), bilateral=bool(reward_step.bilateral_contact_boolean[env_id]), stable=bool(reward_step.stable_grasp_boolean[env_id]), control_step_index=states.state(env_id).control_step, camera_timestamp=float(terminal_frame.timestamp_s), terminal_timestamp=terminal_timestamp, camera_age=max(0.0, camera_age), pre_close_candidate=bool(gate_receipts[env_id].get("pre_close_candidate", False)), close_latched_before_supervision=bool(gate_receipts[env_id].get("close_latched_before_supervision", False)), privileged_close_ready_target=replay_teacher_target, privileged_close_ready_score=replay_teacher_score, privileged_close_ready_negative_reasons=replay_teacher_reasons))
                        _assert_finite_optimizer_metrics(learner_metrics)
                        metrics = _runtime_metric_row(accepted_transitions=coordinator.accepted_transitions, vector_step=vector_step, env_id=env_id, episode_id=episode_ids[env_id], reward_step=reward_step, proposal=executed, gate=gate_receipts[env_id], execution=execution_receipts[env_id], nominal_distance_mm=terminal_distance * 1000.0, phase=decisions[env_id].phase.value, owner=owners[env_id], camera_frame_id=terminal_frame.frame_id, camera_timestamp_s=float(terminal_frame.timestamp_s), packet_unique_4d_rows=consumption.unique_4d_action_rows, packet_unique_8d_rows=consumption.unique_8d_packet_rows, forbidden_collision=bool(forbidden_after[env_id]), failure_reason=failure_reason, close_diagnostic=close_diagnostic)
                        writer.writerow(metrics); metrics_rows.append(metrics)
                        completed_episode_end_transition[
                            (int(env_id), int(episode_ids[env_id]))
                        ] = int(coordinator.accepted_transitions)
                        replay_stream.write(json.dumps({**dict(coordinator.replay_provenance[-1]), "env_id": env_id, "episode_id": identity.episode_id, "step_in_episode": identity.step_in_episode, "camera_frame_id": terminal_frame.frame_id, "post_handoff_retreat_terminal": False}) + "\n")
                        persist_pre_close_teacher_row(
                            identity=identity,
                            proposal=proposals[env_id],
                            gate=gate_receipts[env_id],
                            student_feature_timestamp_s=float(
                                current_frames[env_id].timestamp_s
                            ),
                            gru_reset_generation=int(
                                states.state(env_id).episode_index
                            ),
                            forbidden_collision_before_supervision=bool(
                                forbidden_before[env_id]
                            ),
                        )
                        persist_fsm_sequence_row(
                            identity=identity,
                            proposal=proposals[env_id],
                            executed=executed,
                            gate=gate_receipts[env_id],
                            row=rows[env_id],
                            reward_step=reward_step,
                            records=records_by_env[env_id],
                            failure_reason=failure_reason,
                        )
                        previous_action[env_id] = np.asarray(rows[env_id].final_action_4d_metric_root_m, dtype=np.float32); previous_residual[env_id] = np.asarray(executed.composition.scaled_residual_contribution_m, dtype=np.float64); states.state(env_id).control_step += 1
                        reset_ids.append(env_id)
                        if run is not None: run.log({**learner_metrics, **close_distillation_wandb_metrics(), **current_gru_privileged_wandb_metrics(), **{f"vector/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}}, step=coordinator.accepted_transitions)
                        checkpoint_and_evaluate()
                        continue

                    next_distance = float(np.linalg.norm(nominal[env_id, :3] - next_state["ee_position_root_m"][env_id].detach().cpu().numpy()))
                    next_proposal, next_decision, next_owner, local_retreat = propose(
                        env_id, next_inputs[env_id], next_distance
                    )
                    if local_retreat:
                        # A post-handoff retreat is intentionally not mapped
                        # onto the Stage-1A reward terminal predicate.  The
                        # coordinator rejects mismatched terminal authority,
                        # so discard this clone-local physical tail from
                        # canonical replay and reset only this clone.
                        post_handoff_retreat_discarded_env_steps += 1
                        reset_ids.append(env_id)
                        continue
                    identity = PerEnvReplayIdentity(env_id=env_id, episode_id=f"episode-{episode_ids[env_id]:05d}", step_in_episode=states.state(env_id).control_step)
                    camera_age = float(records_by_env[env_id][-1]["physics_timestamp_s"] - next_frames[env_id].timestamp_s)
                    if camera_age < -1e-6:
                        raise Stage1AVectorIsaacSmokeError("VECTOR_CAMERA_AFTER_PHYSICS")
                    row_terminated = False
                    executed = executed_proposals[env_id]
                    if executed is None:
                        raise Stage1AVectorIsaacSmokeError("VECTOR_EXECUTED_PROPOSAL_MISSING")
                    replay_teacher_target, replay_teacher_score, replay_teacher_reasons = _replay_teacher_fields(runtime_variant=runtime_variant, gate=gate_receipts[env_id])
                    learner_metrics = state_adapter.call(env_id, "accept_real_transition", Stage1AAcceptedRealRow(row_id=identity.row_id, proposal=executed, next_actor_observation=next_proposal.actor_observation, reward_step=reward_step, reward_env_index=env_id, safety_pass=not bool(forbidden_after[env_id]), lift=False, terminated=row_terminated, truncated=runtime_truncated, episode_id=identity.episode_id, phase=decisions[env_id].phase.value, contact=bool(reward_step.left_contact_boolean[env_id] or reward_step.right_contact_boolean[env_id]), bilateral=bool(reward_step.bilateral_contact_boolean[env_id]), stable=bool(reward_step.stable_grasp_boolean[env_id]), control_step_index=states.state(env_id).control_step, camera_timestamp=float(next_frames[env_id].timestamp_s), terminal_timestamp=float(records_by_env[env_id][-1]["physics_timestamp_s"]), camera_age=max(0.0, camera_age), pre_close_candidate=bool(gate_receipts[env_id].get("pre_close_candidate", False)), close_latched_before_supervision=bool(gate_receipts[env_id].get("close_latched_before_supervision", False)), privileged_close_ready_target=replay_teacher_target, privileged_close_ready_score=replay_teacher_score, privileged_close_ready_negative_reasons=replay_teacher_reasons))
                    _assert_finite_optimizer_metrics(learner_metrics)
                    metrics = _runtime_metric_row(accepted_transitions=coordinator.accepted_transitions, vector_step=vector_step, env_id=env_id, episode_id=episode_ids[env_id], reward_step=reward_step, proposal=executed, gate=gate_receipts[env_id], execution=execution_receipts[env_id], nominal_distance_mm=next_distance * 1000.0, phase=decisions[env_id].phase.value, owner=owners[env_id], camera_frame_id=next_frames[env_id].frame_id, camera_timestamp_s=float(next_frames[env_id].timestamp_s), packet_unique_4d_rows=consumption.unique_4d_action_rows, packet_unique_8d_rows=consumption.unique_8d_packet_rows, forbidden_collision=bool(forbidden_after[env_id]), close_diagnostic=episode_failure_diagnostics[env_id])
                    writer.writerow(metrics); metrics_rows.append(metrics)
                    replay_stream.write(json.dumps({**dict(coordinator.replay_provenance[-1]), "env_id": env_id, "episode_id": identity.episode_id, "step_in_episode": identity.step_in_episode, "camera_frame_id": next_frames[env_id].frame_id, "post_handoff_retreat_terminal": False}) + "\n")
                    persist_pre_close_teacher_row(
                        identity=identity,
                        proposal=proposals[env_id],
                        gate=gate_receipts[env_id],
                        student_feature_timestamp_s=float(
                            current_frames[env_id].timestamp_s
                        ),
                        gru_reset_generation=int(states.state(env_id).episode_index),
                        forbidden_collision_before_supervision=bool(
                            forbidden_before[env_id]
                        ),
                    )
                    persist_fsm_sequence_row(
                        identity=identity,
                        proposal=proposals[env_id],
                        executed=executed,
                        gate=gate_receipts[env_id],
                        row=rows[env_id],
                        reward_step=reward_step,
                        records=records_by_env[env_id],
                        failure_reason="NONE",
                    )
                    previous_action[env_id] = np.asarray(rows[env_id].final_action_4d_metric_root_m, dtype=np.float32); previous_residual[env_id] = effective; states.state(env_id).control_step += 1
                    proposals[env_id], decisions[env_id], owners[env_id] = next_proposal, next_decision, next_owner
                    if run is not None: run.log({**learner_metrics, **close_distillation_wandb_metrics(), **current_gru_privileged_wandb_metrics(), **{f"vector/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}}, step=coordinator.accepted_transitions)
                    checkpoint_and_evaluate()
                # Deliberately exercise one clone-local reset in every vector
                # smoke.  It proves reset isolation without changing the
                # reward, CLOSE, residual, or safety contracts.
                if (
                    num_envs > 1
                    and not reset_probe_completed
                    and vector_step >= 1
                ):
                    reset_ids.append(0)
                    reset_probe_completed = True
                    reset_probe_source_preserving = bool(paired_clone_allocation)
                    reset_probe_receipt["executed"] = True
                    reset_probe_receipt["reset_env_id"] = 0
                    reset_probe_receipt["source_probe_preserved"] = bool(
                        paired_clone_allocation
                    )
                if reset_ids:
                    unique = tuple(sorted(set(reset_ids)))
                    other_before = {
                        env_id: (
                            states.state(env_id).control_step,
                            cameras.get(env_id).frame_id if cameras.get(env_id) is not None else None,
                            episode_ids[env_id],
                        )
                        for env_id in range(num_envs)
                        if env_id not in unique
                    }
                    reset_samples = []
                    for env_id in unique:
                        if preclose_collection_only:
                            # Boundary collection keeps every probe in one
                            # pair on the exact same source sample.  The next
                            # pair advances to another frozen HDF5 row from
                            # the same family, strictly by seed/generation.
                            # The teacher target and outcome are never read
                            # during this lifecycle decision.
                            preserve_this_probe = bool(
                                reset_probe_source_preserving
                                and env_id == reset_probe_receipt["reset_env_id"]
                            )
                            if boundary_probes and not preserve_this_probe:
                                next_probe = int(boundary_probe_indices[env_id]) + 1
                                if next_probe >= len(boundary_probes):
                                    boundary_probe_indices[env_id] = 0
                                    boundary_pair_generations[env_id] += 1
                                    family_id = source_family_id_from_sample_id(
                                        active_initial_samples[env_id].sample_id
                                    )
                                    active_initial_samples[env_id] = (
                                        select_catalog_collection_sample_for_source_family(
                                            catalog_path=collection_source_catalog,
                                            source_family_id=family_id,
                                            selection_seed=(
                                                int(training_seed)
                                                + 1_000_003
                                                * int(boundary_pair_generations[env_id])
                                                + int(env_id)
                                            ),
                                            exclude_sample_ids=tuple(
                                                boundary_seen_sample_ids[env_id]
                                            ),
                                        )
                                    )
                                    boundary_seen_sample_ids[env_id].add(
                                        active_initial_samples[env_id].sample_id
                                    )
                                else:
                                    boundary_probe_indices[env_id] = next_probe
                                base_nominal = nominal_grasp_pose_root_m_xyzw_for_cube(
                                    active_initial_samples[
                                        env_id
                                    ].cube_pose_robot_root_m_xyzw[:3]
                                )
                                nominal[env_id] = np.asarray(
                                    apply_probe_to_nominal_pose(
                                        base_nominal,
                                        boundary_probes[
                                            int(boundary_probe_indices[env_id])
                                        ],
                                    ),
                                    dtype=np.float64,
                                )
                                collection_reset_source_receipts.append(
                                    {
                                        "env_id": int(env_id),
                                        "boundary_pair_generation": int(
                                            boundary_pair_generations[env_id]
                                        ),
                                        "boundary_variant_id": boundary_probes[
                                            int(boundary_probe_indices[env_id])
                                        ].variant_id,
                                        "source_family_id": source_family_id_from_sample_id(
                                            active_initial_samples[env_id].sample_id
                                        ),
                                        "sample_id": active_initial_samples[env_id].sample_id,
                                        "hdf5_row_index": int(
                                            active_initial_samples[env_id].hdf5_row_index
                                        ),
                                        "teacher_or_outcome_lookup_used": False,
                                    }
                                )
                            elif boundary_probes:
                                # The dedicated vector-isolation probe above
                                # restores exactly the active immutable
                                # source/probe.  It is not a data-generation
                                # transition and therefore must not advance a
                                # paired-clone generation.
                                collection_reset_source_receipts.append(
                                    {
                                        "env_id": int(env_id),
                                        "boundary_pair_generation": int(
                                            boundary_pair_generations[env_id]
                                        ),
                                        "boundary_variant_id": boundary_probes[
                                            int(boundary_probe_indices[env_id])
                                        ].variant_id,
                                        "source_family_id": source_family_id_from_sample_id(
                                            active_initial_samples[env_id].sample_id
                                        ),
                                        "sample_id": active_initial_samples[env_id].sample_id,
                                        "hdf5_row_index": int(
                                            active_initial_samples[env_id].hdf5_row_index
                                        ),
                                        "reset_probe_source_preserving": True,
                                        "teacher_or_outcome_lookup_used": False,
                                    }
                                )
                            else:
                                # Natural collection remains available as a
                                # legacy diagnostic mode.  Boundary-paired
                                # collection never uses this random reset
                                # path because it would break pair identity.
                                reset_generation = int(episode_ids[env_id]) + 1
                                reset_selection = select_stage1a_vector_initial_states(
                                    num_envs=num_envs,
                                    selection_seed=(
                                        int(training_seed)
                                        + 1_000_003 * reset_generation
                                        + int(env_id)
                                    ),
                                    collection_source_catalog=collection_source_catalog,
                                )
                                active_initial_samples[env_id] = reset_selection.samples[
                                    env_id
                                ]
                                collection_reset_source_receipts.append(
                                    {
                                        "env_id": int(env_id),
                                        "episode_generation": reset_generation,
                                        "sample_id": active_initial_samples[env_id].sample_id,
                                        "hdf5_row_index": int(
                                            active_initial_samples[env_id].hdf5_row_index
                                        ),
                                        "selection_rule": reset_selection.selection_rule,
                                        "teacher_or_outcome_lookup_used": False,
                                    }
                                )
                                nominal[env_id] = np.asarray(
                                    nominal_grasp_pose_root_m_xyzw_for_cube(
                                        active_initial_samples[
                                            env_id
                                        ].cube_pose_robot_root_m_xyzw[:3]
                                    ),
                                    dtype=np.float64,
                                )
                        reset_samples.append(active_initial_samples[env_id])
                    # Snapshot raw sensor identity *before* the direct source
                    # restore.  Clearing the Python cache alone is not a
                    # camera update; policy may only see geometry once the
                    # sensor has emitted a different raw frame/timestamp.
                    for env_id in unique:
                        watermark_frame = next_frames[env_id]
                        if watermark_frame is None:
                            watermark_frame = current_frames[env_id]
                        reset_camera_watermarks[env_id] = (
                            (
                                int(watermark_frame.frame_id),
                                float(watermark_frame.timestamp_s),
                            )
                            if watermark_frame is not None
                            else None
                        )
                    apply_vector_direct_pregrasp_initial_states(
                        env,
                        samples=tuple(reset_samples),
                        env_ids=unique,
                        allow_explicit_paired_clone_duplicates=paired_clone_allocation,
                    )
                    # See the initial direct-restore reset above.  Keep the
                    # action/governor cache generation aligned with the new
                    # source state before canonical OPEN begins; no follower
                    # or torque command is written here.
                    gripper_action_reset(tuple(unique))
                    for env_id in unique:
                        # Clear source-dependent caches immediately after the
                        # direct restore.  Policy/recurrent state is reset
                        # exactly once *at activation* below, after measured
                        # OPEN and fresh geometry pass; that avoids consuming
                        # two GRU-reset generations per physical episode.
                        cameras.reset(env_id)
                        head_cameras.reset(env_id)
                        privileged_geometry_cache[env_id] = None
                        privileged_geometry_frame_ids[env_id] = None
                        privileged_geometry_poll_counts[env_id] = 0
                        privileged_geometry_timestamp_s[env_id] = None
                        episode_ids[env_id] += 1
                        pending_reset[env_id] = True
                        reset_open_restore_progress[env_id] = ResetOpenRestoreProgress(
                            env_id=env_id,
                            episode_id=episode_ids[env_id],
                            source_sample_id=active_initial_samples[env_id].sample_id,
                            reset_vector_step=vector_step + 1,
                        )
                    if reset_probe_receipt["executed"] and reset_probe_receipt["other_env_state_unchanged"] is None:
                        other_after = {
                            env_id: (
                                states.state(env_id).control_step,
                                cameras.get(env_id).frame_id if cameras.get(env_id) is not None else None,
                                episode_ids[env_id],
                            )
                            for env_id in other_before
                        }
                        reset_probe_receipt["other_env_state_unchanged"] = other_before == other_after
                    reset_probe_source_preserving = False
                # The next action for every still-active clone consumes the
                # 25-Hz frame acquired after this vector step.  Reset clones
                # have no valid current frame until their fresh state has
                # completed its open refresh on the following step.
                current_frames = list(next_frames)
                head_current_frames = list(next_head_frames)
                for env_id in set(reset_ids):
                    current_frames[env_id] = None
                    head_current_frames[env_id] = None
                metric_stream.flush()
                replay_stream.flush()
                failure_stream.flush()
                reset_open_restore_stream.flush()
                if close_readiness_stream is not None:
                    close_readiness_stream.flush()
                if sequence_writer is not None:
                    sequence_writer.flush()
                if wrist_rgbd_writer is not None:
                    wrist_rgbd_writer.flush()
                vector_step += 1
        if sequence_writer is not None:
            sequence_writer.flush()
            sequence_summary: Mapping[str, Any] | None = sequence_writer.summary()
            if int(sequence_summary["sequence_row_count"]) != int(
                coordinator.accepted_transitions
            ):
                raise Stage1AVectorIsaacSmokeError(
                    "VECTOR_FSM_SEQUENCE_REPLAY_CARDINALITY_MISMATCH"
                )
        else:
            sequence_summary = None
        if wrist_rgbd_writer is not None:
            wrist_rgbd_writer.flush()
            wrist_rgbd_summary: Mapping[str, Any] | None = wrist_rgbd_writer.summary()
            # A V5 pre-CLOSE row carries exactly one reference to a frame in
            # this store; deduplication makes the frame count lower than the
            # row count by design, so only nonzero storage is required here.
            if int(wrist_rgbd_summary["right_wrist_frame_count"]) <= 0:
                raise Stage1AVectorIsaacSmokeError(
                    "VECTOR_V5_WRIST_RGBD_STORE_EMPTY"
                )
            if (
                boundary_paired_plan is not None
                and boundary_paired_plan.get("v6_diagnostic_receipt") is True
                and int(wrist_rgbd_summary["v6_wrist_diagnostic_frame_count"])
                <= 0
            ):
                raise Stage1AVectorIsaacSmokeError(
                    "VECTOR_V6_WRIST_DIAGNOSTIC_STORE_EMPTY"
                )
        else:
            wrist_rgbd_summary = None
        freeze_after = source_freeze_provider()
        # ``None`` is the only valid cache state immediately after an
        # individually reset clone: it prevents an old-episode RGB-D frame
        # from being reused.  Every populated cache must still be owned by
        # its matching environment.
        camera_cache_identity = all(
            cameras.get(env_id) is None or cameras.get(env_id).env_id == env_id
            for env_id in range(num_envs)
        ) and all(
            head_cameras.get(env_id) is None
            or head_cameras.get(env_id).env_id == env_id
            for env_id in range(num_envs)
        )
        camera_timestamp_violations: list[dict[str, Any]] = []
        last_camera_timestamp: dict[tuple[int, int], float] = {}
        for item in metrics_rows:
            key = (int(item["env_id"]), int(item["episode_id"]))
            timestamp = float(item["camera_timestamp_s"])
            previous = last_camera_timestamp.get(key)
            if previous is not None and timestamp + 1.0e-9 < previous:
                camera_timestamp_violations.append(
                    {"env_id": key[0], "episode_id": key[1], "previous_timestamp_s": previous, "timestamp_s": timestamp}
                )
            last_camera_timestamp[key] = timestamp
        episode_scoped_camera_timestamp_monotonic = not camera_timestamp_violations
        action_diversity_observed = any(
            int(item["unique_4d_action_rows"]) > 1
            for item in action_packet_evidence
        )
        vector_contract_pass = (
            bool(action_packet_evidence)
            and all(bool(item["independently_owned"]) and bool(item["single_batched_consumption"]) for item in action_packet_evidence)
            and camera_cache_identity
            and episode_scoped_camera_timestamp_monotonic
            and (num_envs == 1 or action_diversity_observed)
            and (num_envs == 1 or reset_probe_receipt["other_env_state_unchanged"] is True)
        )
        episode_summary: dict[tuple[int, int], dict[str, bool]] = {}
        for item in metrics_rows:
            key = (int(item["env_id"]), int(item["episode_id"]))
            flags = episode_summary.setdefault(
                key, {"contact": False, "bilateral": False, "stable": False}
            )
            for name in flags:
                flags[name] = flags[name] or bool(item[name])
        episode_count = len(episode_summary)
        contact_episode_count = sum(item["contact"] for item in episode_summary.values())
        bilateral_episode_count = sum(item["bilateral"] for item in episode_summary.values())
        stable_episode_count = sum(item["stable"] for item in episode_summary.values())
        close_triggered_episode_keys = {
            (int(item["env_id"]), int(item["episode_id"]))
            for item in metrics_rows
            if bool(item["close_trigger"])
        }
        close_triggered_episode_count = len(close_triggered_episode_keys)
        close_to_contact_count = sum(
            bool(episode_summary[key]["contact"])
            for key in close_triggered_episode_keys
            if key in episode_summary
        )
        close_to_bilateral_count = sum(
            bool(episode_summary[key]["bilateral"])
            for key in close_triggered_episode_keys
            if key in episode_summary
        )
        close_to_stable_count = sum(
            bool(episode_summary[key]["stable"])
            for key in close_triggered_episode_keys
            if key in episode_summary
        )
        phase_rows = {
            name: [item for item in metrics_rows if item["post_close_control_phase"] == name]
            for name in ("PRE_CLOSE", "CLOSE_TO_FIRST_CONTACT", "SINGLE_CONTACT", "BILATERAL", "STABLE")
        }
        def rate(count: int, denominator: int) -> float:
            return float(count / denominator) if denominator else 0.0
        micro_rows = [
            item for item in metrics_rows
            if float(item["post_close_micro_correction_norm_mm"]) > 0.0
        ]
        residual_by_phase = {
            phase: _percentile_or_zero(
                [float(item["residual_norm_mm"]) for item in items], 95.0
            )
            for phase, items in phase_rows.items()
        }
        cap_hits_by_phase = {
            phase: rate(
                sum(int(item["post_close_micro_correction_cap_hit"]) for item in items),
                len(items),
            )
            for phase, items in phase_rows.items()
        }
        residual_authority_cap_hits_by_phase = {
            phase: rate(
                sum(float(item["residual_norm_mm"]) >= 0.45 - 1.0e-6 for item in items),
                len(items),
            )
            for phase, items in phase_rows.items()
        }
        # The 15K main path counts one final reason per ended clone-local
        # episode.  This avoids falsely treating every 50-Hz row as a new
        # failure while preserving a durable event for non-replayable runtime
        # terminations.  Legacy variants retain their historical row count.
        failure_taxonomy: dict[str, int] = {}
        failure_source = (
            finalized_failure_events
            if _is_v31_lateral_off_long_run(runtime_variant)
            else metrics_rows
        )
        for item in failure_source:
            name = str(item.get("failure_reason", "NONE"))
            if name != "NONE":
                failure_taxonomy[name] = failure_taxonomy.get(name, 0) + 1
        reward_contract = stage1a_reward_contract(
            reward_v3=True, reward_v32=_is_v32_variant(runtime_variant)
        )
        micro_quality_rows = [
            item for item in micro_rows
            if item["micro_correction_quality"] not in ("INSUFFICIENT_HISTORY", "NOT_APPLIED")
        ]
        v32_contact_rate = rate(contact_episode_count, episode_count)
        v32_bilateral_rate = rate(bilateral_episode_count, episode_count)
        v32_stable_rate = rate(stable_episode_count, episode_count)
        residual_norms_mm = [float(item["residual_norm_mm"]) for item in metrics_rows]
        close_trigger_count = sum(
            int(item["close_trigger"])
            for item in metrics_rows
        )
        close_triggered_no_contact_count = sum(
            name == "NO_CONTACT:CLOSE_TRIGGERED_NO_CONTACT"
            for name in (str(item.get("failure_reason", "NONE")) for item in failure_source)
        )
        high_slip_count = sum(
            name == "BILATERAL_NOT_STABLE:HIGH_SLIP"
            for name in (str(item.get("failure_reason", "NONE")) for item in failure_source)
        )
        contact_loss_count = sum(
            "CONTACT_LOSS" in str(item.get("failure_reason", ""))
            for item in failure_source
        )
        micro_contact_retained_rate = rate(
            sum(int(item["micro_contact_retained"]) for item in micro_rows),
            len(micro_rows),
        )
        learner_summary = coordinator.metrics()
        completed_final = _completed_episode_summary(
            metrics_rows=metrics_rows,
            completed_episode_end_transition=completed_episode_end_transition,
            upper_transition=int(coordinator.accepted_transitions),
            lower_exclusive_transition=0,
            slip_reference_m_s=float(reward.config.slip_reference_m_s),
        )
        final_grasp_evaluation = canonical_grasp_evaluation_summary(completed_final)
        advisory_rows = [
            item
            for item in metrics_rows
            if item.get("student_advisory_authority")
            == "FROZEN_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY"
        ]
        advisory_scores = [
            float(item["student_advisory_score"])
            for item in advisory_rows
            if math.isfinite(float(item["student_advisory_score"]))
        ]
        advisory_teacher_rows = [
            item
            for item in advisory_rows
            if bool(item.get("student_advisory_teacher_target_known", False))
        ]
        fsm_advisory_comparison_rows = [
            item
            for item in advisory_rows
            if bool(item["fsm_close_triggered_receipt"])
            or bool(item["student_ready_advisory"])
            or bool(item["student_defer_to_fsm"])
        ]
        advisory_ready_count = sum(
            int(item["student_ready_advisory"]) for item in advisory_rows
        )
        advisory_defer_count = sum(
            int(item["student_defer_to_fsm"]) for item in advisory_rows
        )
        advisory_teacher_ready_count = sum(
            int(item["student_ready_advisory"]) for item in advisory_teacher_rows
        )
        advisory_summary = {
            "enabled": _uses_frozen_student_advisory(runtime_variant),
            "checkpoint_path": (
                str(frozen_student_advisory.checkpoint_path)
                if frozen_student_advisory is not None
                else None
            ),
            "checkpoint_sha256": (
                frozen_student_advisory.checkpoint_sha256
                if frozen_student_advisory is not None
                else None
            ),
            "high_ready_threshold": (
                DEFAULT_HIGH_READY_THRESHOLD
                if frozen_student_advisory is not None
                else None
            ),
            "runtime_authority": (
                "TELEMETRY_ONLY__FSM_CLOSE_UNCHANGED"
                if frozen_student_advisory is not None
                else "NOT_USED"
            ),
            "receipt_count": len(advisory_rows),
            "ready_advisory_count": advisory_ready_count,
            "defer_count": advisory_defer_count,
            "advisory_coverage": rate(advisory_ready_count, len(advisory_rows)),
            "defer_rate": rate(advisory_defer_count, len(advisory_rows)),
            "score_mean": float(np.mean(advisory_scores)) if advisory_scores else 0.0,
            "score_p50": _percentile_or_zero(advisory_scores, 50.0),
            "score_p95": _percentile_or_zero(advisory_scores, 95.0),
            "fsm_student_agreement": rate(
                sum(int(item["fsm_student_agreement"]) for item in fsm_advisory_comparison_rows),
                len(fsm_advisory_comparison_rows),
            ),
            "ready_advisory_fsm_not_close_count": sum(
                int(item["ready_advisory_fsm_not_close"]) for item in advisory_rows
            ),
            "fsm_close_student_defer_count": sum(
                int(item["fsm_close_student_defer"]) for item in advisory_rows
            ),
            "ready_advisory_false_accept": rate(
                sum(
                    int(item["ready_advisory_teacher_false_accept"])
                    for item in advisory_teacher_rows
                ),
                advisory_teacher_ready_count,
            ),
            "teacher_comparison_row_count": len(advisory_teacher_rows),
            "student_privileged_input_count": 0,
            "fsm_runtime_authority": True,
            "student_hard_gate": False,
        }
        reset_attempt_last_receipt: dict[tuple[int, int], dict[str, Any]] = {}
        for item in reset_open_restore_receipts:
            if "env_id" not in item or "episode_id" not in item:
                continue
            reset_attempt_last_receipt[(int(item["env_id"]), int(item["episode_id"]))] = item
        reset_attempts = list(reset_attempt_last_receipt.values())
        reset_open_restore_summary = {
            "schema": RESET_OPEN_RESTORE_SCHEMA,
            "attempt_count": len(reset_attempts),
            "pass_count": sum(
                bool(item.get("open_restore_pass", False))
                for item in reset_attempts
            ),
            "failure_count": sum(
                not bool(item.get("open_restore_pass", False))
                for item in reset_attempts
            ),
            "open_restore_pass": bool(reset_attempts) and all(
                bool(item.get("open_restore_pass", False))
                for item in reset_attempts
            ),
            "reset_parity_pass": bool(reset_attempts) and all(
                bool(item.get("master_open_ok", False))
                and bool(item.get("velocity_settled", False))
                and bool(item.get("aperture_open_ok", False))
                for item in reset_attempts
            ),
            "teacher_receipt_parity_pass": bool(reset_attempts) and all(
                bool(item.get("fresh_geometry_receipt", False))
                for item in reset_attempts
            ),
            "previous_episode_geometry_used": sum(
                not bool(item.get("geometry_cache_fresh", False))
                for item in reset_attempts
            ),
            "minimum_open_command_steps": (
                G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
            ),
            "maximum_open_command_steps": (
                G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS
            ),
            "receipt_path": str(output_dir / "RESET_OPEN_RESTORE_RECEIPTS.jsonl"),
            "attempts": reset_attempts,
        }
        if preclose_collection_only and (
            coordinator.sac_update_count != 0
            or int(learner_summary.get("CLOSE_DISTILLATION_UPDATE_COUNT", -1)) != 0
        ):
            raise Stage1AVectorIsaacSmokeError(
                "PRECLOSE_COLLECTION_OPTIMIZER_UPDATE_FORBIDDEN"
            )
        report = {
            "schema": VECTOR_RUNTIME_SCHEMA,
            "runtime_variant": runtime_variant,
            "training_seed": int(training_seed),
            "fair_comparison_id": (
                "stage1a_reset_fixed_25env_7p5k_preflight"
                if _is_reset_fixed_25env_7p5k_variant(runtime_variant)
                and accepted_transition_target == 100
                else "stage1a_reset_fixed_25env_7p5k"
                if _is_reset_fixed_25env_7p5k_variant(runtime_variant)
                else "stage1a_reset_fixed_25env_6k"
                if _is_reset_fixed_25env_6k_variant(runtime_variant)
                else "stage1a_reset_fixed_10env_6k"
                if _is_fair_6k_variant(runtime_variant)
                else None
            ),
            "fair_comparison_method": (
                _reset_fixed_7p5k_method(runtime_variant)
                if _is_reset_fixed_25env_7p5k_variant(runtime_variant)
                else _reset_fixed_6k_method(runtime_variant)
                if _is_reset_fixed_25env_6k_variant(runtime_variant)
                else _fair_6k_method(runtime_variant)
            ),
            "fair_comparison_contract": (
                {
                    "num_envs": 25,
                    "accepted_transitions": int(accepted_transition_target),
                    "accepted_transitions_per_env": (
                        accepted_transition_target // num_envs
                    ),
                    "preflight_only": accepted_transition_target == 100,
                    "replay_strategy": "HER_FORCE",
                    "wandb_mode": "online",
                    "same_seed_required": True,
                    "reset_open_restore_required": True,
                }
                if _is_reset_fixed_25env_variant(runtime_variant)
                else
                {
                    "num_envs": 10,
                    "accepted_transitions": 6000,
                    "replay_strategy": "HER_FORCE",
                    "wandb_mode": "online",
                    "same_seed_required": True,
                    "reset_open_restore_required": True,
                }
                if _is_fair_6k_variant(runtime_variant)
                else None
            ),
            "num_envs": num_envs,
            "preflight_only": reset_fixed_preflight,
            "accepted_transitions": coordinator.accepted_transitions,
            "expected_vector_steps": int(math.ceil(accepted_transition_target / num_envs)),
            "actual_vector_steps": vector_step,
            "vector_clone_count": num_envs,
            "source_initial_states": selection.receipt(),
            "sequence_dataset": sequence_summary,
            "wrist_rgbd_dataset": wrist_rgbd_summary,
            "frozen_student_advisory": advisory_summary,
            "current_gru_privileged": (
                current_gru_privileged.metrics()
                if current_gru_privileged is not None
                else {
                    "enabled": False,
                    "student_privileged_input_count": 0,
                }
            ),
            "collection_source_catalog": (
                str(collection_source_catalog.resolve())
                if collection_source_catalog is not None
                else None
            ),
            "boundary_paired_collection": {
                "enabled": bool(boundary_probes),
                "schema": (
                    str(boundary_paired_plan.get("schema"))
                    if boundary_paired_plan is not None
                    else None
                ),
                "row_schema": (
                    str(boundary_paired_plan.get("row_schema"))
                    if boundary_paired_plan is not None
                    else None
                ),
                "signed_margin_telemetry": bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("signed_margin_telemetry") is True
                ),
                "causal_student_state_receipt": bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("causal_student_state_receipt") is True
                ),
                "wrist_rgbd_receipt": bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("wrist_rgbd_receipt") is True
                ),
                "v6_diagnostic_receipt": bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("v6_diagnostic_receipt") is True
                ),
                "paired_clone_allocation": bool(
                    boundary_paired_plan is not None
                    and boundary_paired_plan.get("paired_clone_allocation") is True
                ),
                "source_row_selection": (
                    str(boundary_paired_plan.get("source_row_selection"))
                    if boundary_paired_plan is not None
                    else None
                ),
                "source_family_ids": (
                    list(boundary_paired_plan.get("source_family_ids", ()))
                    if boundary_paired_plan is not None
                    else []
                ),
                "plan_path": (
                    str(boundary_paired_plan.get("plan_path"))
                    if boundary_paired_plan is not None
                    else None
                ),
                "plan_sha256": (
                    str(boundary_paired_plan.get("plan_sha256"))
                    if boundary_paired_plan is not None
                    else None
                ),
                "probe_ids": [probe.variant_id for probe in boundary_probes],
                "label_assigned_by_plan": False,
                "sac_update": 0 if boundary_probes else None,
                "student_optimizer_update": 0 if boundary_probes else None,
                "privileged_hard_gate": False if boundary_probes else None,
                "student_privileged_input_count": 0,
            },
            "collection_reset_source_receipts": collection_reset_source_receipts,
            "reset_open_restore": reset_open_restore_summary,
            "direct_init": [receipt.as_dict() for receipt in direct_receipts],
            "direct_init_camera_refresh": True,
            "action_batch_shape": [num_envs, 8],
            "action_broadcast_detected": False,
            "single_batched_action_consumption": True,
            "vector_action_packet_count": len(action_packet_evidence),
            "max_unique_4d_action_rows": max((int(item["unique_4d_action_rows"]) for item in action_packet_evidence), default=0),
            "max_unique_8d_packet_rows": max((int(item["unique_8d_packet_rows"]) for item in action_packet_evidence), default=0),
            "action_diversity_observed": action_diversity_observed,
            "per_env_camera_cache_integrity": camera_cache_identity,
            "episode_scoped_camera_timestamp_monotonic": episode_scoped_camera_timestamp_monotonic,
            "camera_timestamp_violation_count": len(camera_timestamp_violations),
            "camera_timestamp_violations": camera_timestamp_violations[:32],
            "replay_env_id_present": True,
            "tail_env_steps_not_accepted": tail_env_steps_not_accepted,
            "post_handoff_retreat_discarded_env_steps": post_handoff_retreat_discarded_env_steps,
            "runtime_unattributed_termination_discarded_env_steps": (
                runtime_unattributed_termination_discarded_env_steps
            ),
            "reset_isolation": "PER_ENV",
            "reset_probe": reset_probe_receipt,
            "control_hz": 50,
            "rgbd_hz": 25,
            "physics_hz": 500,
            "sac_update_count": coordinator.sac_update_count,
            "student_optimizer_update_count": learner_summary.get(
                "CLOSE_DISTILLATION_UPDATE_COUNT", 0
            ) + (
                current_gru_privileged.metrics()["update_count"]
                if current_gru_privileged is not None
                else 0
            ),
            "current_gru_privileged_update_count": (
                current_gru_privileged.metrics()["update_count"]
                if current_gru_privileged is not None
                else 0
            ),
            "updates_per_accepted_transition": (
                coordinator.sac_update_count / max(
                    1, coordinator.accepted_transitions - coordinator.learning_starts
                )
            ),
            "checkpoints_saved": saved,
            "checkpoint_evaluations": checkpoint_evaluations,
            "FINAL_METRIC_DENOMINATOR": "COMPLETED_EPISODES_ONLY",
            "PRECLOSE_COLLECTION_ONLY": preclose_collection_only,
            "TRAINING_STARTED": False if preclose_collection_only else True,
            "COLLECTION_COMPLETED": (
                coordinator.accepted_transitions == accepted_transition_target
                if preclose_collection_only
                else False
            ),
            "source_freeze_match": dict(source_freeze_before) == dict(freeze_after),
            "vector_contract_pass": vector_contract_pass,
            "wandb": ({"id": run.id, "url": run.url} if run is not None else None),
            "TRAINING_COMPLETED": (
                False
                if preclose_collection_only
                else coordinator.accepted_transitions == accepted_transition_target
            ),
            # V3.2 keeps every V3 milestone and scale.  Its only reward
            # addition is the separately accounted, capped bilateral phase
            # term documented in ``reward_contract``.
            "REWARD_V3_BASE_STRUCTURE_CHANGED": False,
            "BILATERAL_STABILIZATION_SHAPING": _is_v32_variant(runtime_variant),
            "BILATERAL_STABILITY_SHAPING_REWARD_PER_STEP": (
                0.001 if _is_v32_variant(runtime_variant) else 0.0
            ),
            "BILATERAL_INSTABILITY_PENALTY_PER_STEP": (
                -0.002 if _is_v32_variant(runtime_variant) else 0.0
            ),
            "BILATERAL_STABILITY_SHAPING_EPISODE_CAP": (
                0.020 if _is_v32_variant(runtime_variant) else 0.0
            ),
            "HOVER_PENALTY_PHASE_AWARE": _is_v32_variant(runtime_variant),
            "SINGLE_CONTACT_DWELL_STRENGTHENED": _is_v32_variant(runtime_variant),
            "LATERAL_CLOSE_GATE_ADDED": _is_v31_variant(runtime_variant),
            "LATERAL_CLOSE_GATE_THRESHOLD_MM": 10.0 if _is_v31_variant(runtime_variant) else None,
            "LATERAL_CLOSE_GATE": (
                "TELEMETRY_ONLY"
                if runtime_variant == V32_RUNTIME_VARIANT
                or runtime_variant == V32_25ENV_RESET_FIXED_7P5K_RUNTIME_VARIANT
                or _is_v31_lateral_off_variant(runtime_variant)
                else "NOT_USED_BY_PRIVILEGED_GEOMETRY_TEACHER"
                if _uses_privileged_geometry_teacher(runtime_variant)
                else "HARD_10MM" if _is_v31_variant(runtime_variant)
                else "NOT_PRESENT"
            ),
            "PAIRED_V31_LATERAL_GATE_OFF": (
                runtime_variant == V31_LATERAL_OFF_RUNTIME_VARIANT
            ),
            "FROZEN_V31_LATERAL_OFF_MAIN_RUN": (
                _is_v31_lateral_off_long_run(runtime_variant)
            ),
            "ONLY_FUNCTIONAL_CHANGE": (
                "ADD_FROZEN_STUDENT_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY"
                if _uses_frozen_student_advisory(runtime_variant)
                else "REMOVE_LATERAL_10MM_HARD_CLOSE_GATE"
                if _is_v31_lateral_off_variant(runtime_variant)
                else "PRIVILEGED_GEOMETRY_TEACHER_CLOSE_READINESS"
                if _uses_privileged_geometry_teacher(runtime_variant)
                else "ADD_PRIVILEGED_CLOSE_READINESS_STUDENT_DISTILLATION"
                if _uses_privileged_geometry_distillation(runtime_variant)
                else None
            ),
            "PAIRED_AGAINST_RUNTIME_VARIANT": (
                V31_RUNTIME_VARIANT
                if runtime_variant == V31_LATERAL_OFF_RUNTIME_VARIANT
                else None
            ),
            "FROZEN_FROM_RUNTIME_VARIANT": (
                V31_LATERAL_OFF_RUNTIME_VARIANT
                if _is_v31_lateral_off_long_run(runtime_variant)
                else None
            ),
            "POST_CLOSE_FULL_XYZ_FREEZE_REMOVED": runtime_variant in POST_CLOSE_MICRO_RUNTIME_VARIANTS,
            "CONTACT_TO_BILATERAL_MICRO_CORRECTION_MAX_MM": 0.15 if runtime_variant in POST_CLOSE_MICRO_RUNTIME_VARIANTS else 0.0,
            "BILATERAL_TO_STABLE_MICRO_CORRECTION_MAX_MM": 0.075 if runtime_variant in POST_CLOSE_MICRO_RUNTIME_VARIANTS else 0.0,
            "FORWARD_NORMAL_PUSH_ALLOWED": False,
            "POST_CLOSE_PRIVILEGED_STUDENT_INPUT_COUNT": 0,
            "CLOSE_READINESS_AUTHORITY": (
                "PRIVILEGED_GEOMETRY_TEACHER"
                if _uses_privileged_geometry_teacher(runtime_variant)
                else "CURRENT_DISTANCE_ALIGNMENT_FSM"
                if _uses_fsm_sequence_dataset(runtime_variant)
                else "CURRENT_RULE"
            ),
            "FSM_RUNTIME_AUTHORITY": (
                "YES" if _uses_fsm_sequence_dataset(runtime_variant) else "NO"
            ),
            "STUDENT_HARD_GATE": "NO",
            "FROZEN_STUDENT_ADVISORY_RULE": (
                "HIGH_CONFIDENCE_ADVISORY_FSM_DEFER"
                if _uses_frozen_student_advisory(runtime_variant)
                else "NOT_USED"
            ),
            "PRIVILEGED_GEOMETRY_HZ": (
                25 if _uses_privileged_geometry_receipt(runtime_variant) else None
            ),
            "PRIVILEGED_GEOMETRY_ROLE": (
                "RUNTIME_HARD_CLOSE_GATE"
                if _uses_privileged_geometry_teacher(runtime_variant)
                else "STUDENT_CLOSE_READINESS_DISTILLATION_TEACHER_ONLY"
                if _uses_privileged_geometry_distillation(runtime_variant)
                else "EXACT_SIGNED_MARGIN_SEQUENCE_TELEMETRY_ONLY"
                if _uses_fsm_sequence_dataset(runtime_variant)
                else "NOT_USED"
            ),
            "PRIVILEGED_USED_AS_HARD_GATE": _uses_privileged_geometry_teacher(
                runtime_variant
            ),
            "PRIVILEGED_USED_AS_TEACHER": _uses_privileged_geometry_receipt(
                runtime_variant
            ),
            "PRIVILEGED_GEOMETRY_POLL_COUNT": sum(
                privileged_geometry_poll_counts
            ),
            "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
            "GRU_RUNTIME_AUTHORITY": "NO",
            "GRU_NOMINAL_XYZ_AUTHORITY": "YES",
            "CURRENT_GRU_PRIVILEGED_AUXILIARY": (
                "SIGNED_MARGIN_AND_BINARY_TEACHER_SUPERVISION_ONLY"
                if current_gru_privileged is not None
                else "NOT_USED"
            ),
            "CURRENT_GRU_PRIVILEGED_USED_IN_SAC_REPLAY": "NO",
            "PRIVILEGED_RUNTIME_AUTHORITY": "NO",
            "SAC_REPLAY_SOURCE": "CURRENT_RUNTIME_ROLLOUT_ONLY",
            "OLD_TEACHER_DATA_USED_IN_SAC": "NO",
            "BATCH10_USED": "NO",
            "CLOSE_READINESS_PRE_CLOSE_ROWS_PATH": (
                str(close_readiness_rows_path)
                if _uses_privileged_geometry_distillation(runtime_variant)
                and close_readiness_rows_path is not None
                else None
            ),
            "CLOSE_READINESS_TARGET_SEMANTICS": (
                "BINARY_PRE_CLOSE_ADMISSION"
                if _uses_privileged_geometry_distillation(runtime_variant)
                else None
            ),
            "NORMALIZATION_PARITY": learner_summary.get(
                "CLOSE_READINESS_NORMALIZATION_PARITY", False
            ),
            "CLASS_BALANCED_SAMPLING": learner_summary.get(
                "CLOSE_READINESS_CLASS_BALANCED_SAMPLING", False
            ),
            "CLASS_BALANCED_LOSS": learner_summary.get(
                "CLOSE_READINESS_CLASS_BALANCED_LOSS", False
            ),
            "CLOSE_READINESS_TRAINING_CONTRACT_SHA256": learner_summary.get(
                "CLOSE_READINESS_TRAINING_CONTRACT_SHA256"
            ),
            "TIMESTAMP_PARITY_LOGGED": bool(
                _uses_privileged_geometry_distillation(runtime_variant)
                or _uses_fsm_sequence_dataset(runtime_variant)
            ),
            "GRU_RESET_GENERATION_LOGGED": bool(
                _uses_privileged_geometry_distillation(runtime_variant)
                or _uses_fsm_sequence_dataset(runtime_variant)
            ),
            "DISTILLATION_LOSS_FINAL": learner_summary.get(
                "optimizer/loss/close_distill", 0.0
            ),
            "PRIVILEGED_CLOSE_READY_RATE": learner_summary.get(
                "PRIVILEGED_CLOSE_READY_RATE", 0.0
            ),
            "STUDENT_CLOSE_READY_MEAN": learner_summary.get(
                "STUDENT_CLOSE_READY_MEAN", 0.0
            ),
            "TEACHER_STUDENT_AGREEMENT": learner_summary.get(
                "TEACHER_STUDENT_AGREEMENT", 0.0
            ),
            "STUDENT_PRECISION": learner_summary.get(
                "STUDENT_CLOSE_READY_PRECISION", 0.0
            ),
            "STUDENT_RECALL": learner_summary.get(
                "STUDENT_CLOSE_READY_RECALL", 0.0
            ),
            "FALSE_REJECT_RATE": learner_summary.get(
                "STUDENT_FALSE_REJECT_RATE", 0.0
            ),
            "FALSE_ACCEPT_RATE": learner_summary.get(
                "STUDENT_FALSE_ACCEPT_RATE", 0.0
            ),
            "episode_count_observed": completed_final["episode_count"],
            "contact_episode_count": completed_final["contact_episode_count"],
            "premature_pre_close_contact_episode_count": completed_final[
                "premature_pre_close_contact_episode_count"
            ],
            "post_close_contact_episode_count": completed_final[
                "post_close_contact_episode_count"
            ],
            "valid_close_ready_contact_episode_count": completed_final[
                "valid_close_ready_contact_episode_count"
            ],
            "bilateral_episode_count": completed_final["bilateral_episode_count"],
            "stable_episode_count": completed_final["stable_episode_count"],
            "GRASP_EVALUATION": final_grasp_evaluation,
            "ANY_CONTACT_RATE": final_grasp_evaluation["ANY_CONTACT_RATE"],
            "CONTACT_RATE": final_grasp_evaluation["CONTACT_RATE"],
            "PREMATURE_PRE_CLOSE_CONTACT_RATE": final_grasp_evaluation[
                "PREMATURE_PRE_CLOSE_CONTACT_RATE"
            ],
            "POST_CLOSE_CONTACT_RATE": final_grasp_evaluation[
                "POST_CLOSE_CONTACT_RATE"
            ],
            "VALID_CLOSE_READY_CONTACT_RATE": final_grasp_evaluation[
                "VALID_CLOSE_READY_CONTACT_RATE"
            ],
            "BILATERAL_CONTACT_CANDIDATE_RATE": final_grasp_evaluation[
                "BILATERAL_CONTACT_CANDIDATE_RATE"
            ],
            "GRASP_SUCCESS_STABLE_RATE": final_grasp_evaluation[
                "GRASP_SUCCESS_STABLE_RATE"
            ],
            "CONTACT_TO_BILATERAL": final_grasp_evaluation[
                "CONTACT_TO_BILATERAL"
            ],
            "BILATERAL_TO_STABLE": final_grasp_evaluation[
                "BILATERAL_TO_STABLE"
            ],
            "POST_BILATERAL_CONTACT_LOSS_RATE": final_grasp_evaluation[
                "POST_BILATERAL_CONTACT_LOSS_RATE"
            ],
            "POST_BILATERAL_CONTACT_LOSS": final_grasp_evaluation[
                "POST_BILATERAL_CONTACT_LOSS_RATE"
            ],
            "CLOSE_TRIGGER_COUNT": close_trigger_count,
            "FSM_CLOSE_TRIGGER_COUNT": close_trigger_count,
            "CLOSE_TRIGGER_RATE": rate(close_trigger_count, len(metrics_rows)),
            "CLOSE_TRIGGERED_EPISODE_COUNT": completed_final[
                "close_triggered_episode_count"
            ],
            "CLOSE_TO_CONTACT_COUNT": completed_final["close_to_contact_count"],
            "CLOSE_TO_BILATERAL_COUNT": completed_final[
                "close_to_bilateral_count"
            ],
            "CLOSE_TO_STABLE_COUNT": completed_final["close_to_stable_count"],
            "CLOSE_TO_CONTACT": completed_final["close_to_contact"],
            "CLOSE_TO_BILATERAL": completed_final["close_to_bilateral"],
            "CLOSE_TO_STABLE": completed_final["close_to_stable"],
            "ADVISORY_COVERAGE": advisory_summary["advisory_coverage"],
            "DEFER_RATE": advisory_summary["defer_rate"],
            "FSM_STUDENT_AGREEMENT": advisory_summary["fsm_student_agreement"],
            "READY_ADVISORY_FALSE_ACCEPT": advisory_summary[
                "ready_advisory_false_accept"
            ],
            "READY_ADVISORY_FSM_NOT_CLOSE_COUNT": advisory_summary[
                "ready_advisory_fsm_not_close_count"
            ],
            "FSM_CLOSE_STUDENT_DEFER_COUNT": advisory_summary[
                "fsm_close_student_defer_count"
            ],
            "GRU_SEQUENCE_ROWS": (
                int(sequence_summary["sequence_row_count"])
                if sequence_summary is not None
                else 0
            ),
            "VALID_SEQUENCE_EPISODES": episode_count,
            "SIGNED_MARGIN_TELEMETRY_COMPLETE": bool(
                sequence_summary is not None
                and int(sequence_summary["exact_signed_margin_row_count"])
                == int(sequence_summary["sequence_row_count"])
                and int(sequence_summary["sequence_row_count"]) > 0
            ),
            "CLOSE_TRIGGERED_NO_CONTACT_COUNT": completed_final[
                "close_triggered_no_contact_count"
            ],
            "CLOSE_TRIGGERED_NO_CONTACT_RATE": completed_final[
                "close_triggered_no_contact_rate"
            ],
            "HOVER_PRE_CLOSE_RATE": rate(sum(int(item["hover"]) for item in phase_rows["PRE_CLOSE"]), len(phase_rows["PRE_CLOSE"])),
            "HOVER_SINGLE_CONTACT_RATE": rate(sum(int(item["hover"]) for item in phase_rows["SINGLE_CONTACT"]), len(phase_rows["SINGLE_CONTACT"])),
            "HOVER_BILATERAL_RATE": rate(sum(int(item["hover"]) for item in phase_rows["BILATERAL"]), len(phase_rows["BILATERAL"])),
            "HOVER_POST_STABLE_RATE": rate(sum(int(item["hover"]) for item in phase_rows["STABLE"]), len(phase_rows["STABLE"])),
            "SINGLE_CONTACT_DWELL_COUNT": sum(1 for item in metrics_rows if int(item["single_contact_dwell_counter"]) > 10),
            "SLIP_RATE": rate(sum(1 for item in metrics_rows if float(item["tangential_slip_m_s"]) > float(reward.config.slip_reference_m_s)), len(metrics_rows)),
            "HIGH_SLIP_COUNT": high_slip_count,
            "CONTACT_LOSS_RATE": rate(sum(int(item["contact_loss_fail"]) for item in metrics_rows), len(metrics_rows)),
            "CONTACT_LOSS_COUNT": contact_loss_count,
            "POST_CLOSE_MICRO_CORRECTION_RATE": rate(len(micro_rows), len(metrics_rows)),
            "POST_CLOSE_MICRO_CORRECTION_MEAN_MM": (float(np.mean([float(item["post_close_micro_correction_norm_mm"]) for item in micro_rows])) if micro_rows else 0.0),
            "POST_CLOSE_MICRO_CORRECTION_P95_MM": _percentile_or_zero([float(item["post_close_micro_correction_norm_mm"]) for item in micro_rows], 95.0),
            "POST_CLOSE_MICRO_CORRECTION_MAX_MM": max((float(item["post_close_micro_correction_norm_mm"]) for item in micro_rows), default=0.0),
            "MICRO_CORRECTION_HELPFUL_RATE": rate(
                sum(item["micro_correction_quality"] == "HELPFUL" for item in micro_quality_rows),
                len(micro_quality_rows),
            ),
            "MICRO_CORRECTION_ADVERSE_RATE": rate(
                sum(item["micro_correction_quality"] == "ADVERSE" for item in micro_quality_rows),
                len(micro_quality_rows),
            ),
            "MICRO_CORRECTION_CONTACT_RETAINED_RATE": micro_contact_retained_rate,
            "MICRO_CORRECTION_QUALITY_COUNTS": {
                name: sum(item["micro_correction_quality"] == name for item in micro_rows)
                for name in ("HELPFUL", "ADVERSE", "NEUTRAL", "INSUFFICIENT_HISTORY")
            },
            "PRE_CLOSE_RESIDUAL_P95_MM": residual_by_phase["PRE_CLOSE"],
            "SINGLE_CONTACT_RESIDUAL_P95_MM": residual_by_phase["SINGLE_CONTACT"],
            "BILATERAL_RESIDUAL_P95_MM": residual_by_phase["BILATERAL"],
            "RESIDUAL_CAP_HIT_RATE_BY_PHASE": cap_hits_by_phase,
            "RESIDUAL_AUTHORITY_CAP_HIT_RATE_BY_PHASE": residual_authority_cap_hits_by_phase,
            "RESIDUAL_MEAN_P95_MAX_MM": [
                float(np.mean(residual_norms_mm)) if residual_norms_mm else 0.0,
                _percentile_or_zero(residual_norms_mm, 95.0),
                max(residual_norms_mm, default=0.0),
            ],
            "RESIDUAL_CAP_HIT_RATE": rate(
                sum(value >= 0.45 - 1.0e-6 for value in residual_norms_mm),
                len(residual_norms_mm),
            ),
            "failure_taxonomy": failure_taxonomy,
            "failure_event_count": len(finalized_failure_events),
            "failure_event_stream": "FAILURE_EVENTS.jsonl",
            "ACTION_BOUND_VIOLATION": 0,
            "GRIPPER_AUTHORITY_VIOLATION": 0,
            "RUNTIME_HARDSTOP": sum(int(item["runtime_hardstop"]) for item in metrics_rows),
            "FORBIDDEN_COLLISION": sum(int(item["forbidden_collision"]) for item in metrics_rows),
            "REWARD_FARMING": "NO",
            "reward_contract": reward_contract,
            "coordinator_metrics": learner_summary,
            "ACTOR_CRITIC_ALPHA": (
                "FINITE"
                if bool(learner_summary["ACTOR_LOSS_FINITE"])
                and bool(learner_summary["CRITIC_LOSS_FINITE"])
                and bool(learner_summary["ALPHA_FINITE"])
                else "FAIL"
            ),
        }
        _atomic_json(report_path, report); _atomic_json(output_dir / "STAGE1A_VECTOR_REPORT.json", report)
        if run is not None: run.summary.update(report); run.finish()
        return 0 if report["TRAINING_COMPLETED"] and report["source_freeze_match"] and report["vector_contract_pass"] else 2
    except BaseException:
        if run is not None: run.finish(exit_code=2)
        raise
    finally:
        if sequence_writer is not None:
            sequence_writer.close()
        if wrist_rgbd_writer is not None:
            wrist_rgbd_writer.close()
        telemetry.restore()


__all__ = ["Stage1AVectorIsaacSmokeError", "VECTOR_RUNTIME_SCHEMA", "run_stage1a_isaac_vector_smoke"]
