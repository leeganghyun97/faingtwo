# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Task-relevant RGB-D models for the G2 teacher/student pipeline.

RGB and depth are encoded separately, and head/wrist weight sharing is an
explicit experiment option.  The deployment student consumes only cameras,
robot state, previous action and camera age.  Simulator state is restricted
to critic and auxiliary *targets* during training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from geniesim.rl.sac.stage2_sac import polyak_update, squashed_gaussian_log_prob
from .g2_quaternion import canonicalize_quaternion_xyzw
from .g2_teacher_sac import G2TeacherObservationContract


G2_VISUAL_SAC_SCHEMA = "g2_temporal_safe_close_asymmetric_sac_v15"
G2_VISUAL_REPLAY_SCHEMA = "g2_temporal_safe_close_episode_sequence_replay_v10"
G2_FUTURE_SAFE_CLOSE_READY_LABEL_SCHEMA = (
    "g2_future_safe_close_ready_timing_contact_time_label_v3"
)
G2_VISUAL_CAMERA_SHAPE = (2, 6, 48, 64)
G2_VISUAL_PROPRIO_DIM = 45
G2_VISUAL_PRIVILEGED_DIM = 59
G2_VISUAL_ACTION_DIM = 7
G2_VISUAL_ARM_ACTION_DIM = 6
G2_VISUAL_GRIPPER_ACTION_INDEX = 6
G2_ACTOR_CUBE_POSITION_INPUT_MODES = (
    "absolute_head_xyz",
    "relative_grasp_xyz",
)
G2_REPLAY_BEHAVIOR_AUTHORITY = {
    "POLICY": 0,
    "REFERENCE": 1,
    "ASSISTED": 2,
}
G2_ROLLOUT_SAMPLING_CONTRACT = (
    "g2_recurrent_single_encoder_mixed_deterministic_mask_rng_v1"
)
G2_PURE_POLICY_CONTACT_SAFETY_GATE_SCHEMA = (
    "g2_pure_policy_contact_markov_projection_v1"
)
G2_PRECONTACT_TASK_ALIGNMENT_SAFETY_PROJECTION_SCHEMA = (
    "g2_pure_policy_precontact_task_alignment_projection_v3"
)
G2_PRECONTACT_PAD_TABLE_CLEARANCE_PROJECTION_SCHEMA = (
    "g2_pure_policy_precontact_pad_table_clearance_projection_v1"
)
G2_PURE_POLICY_GRASP_READY_INTERLOCK_SCHEMA = (
    "g2_pure_policy_grasp_ready_interlock_v4"
)
G2_ACTOR_UPDATE_SCOPES = ("full", "gripper_only", "action_heads_only")
G2_ACTOR_UPDATE_MUTABLE_PREFIXES = {
    # Preserve the v13 bounded diagnostic exactly.
    "gripper_only": (
        "gripper_logit.",
        "policy.gripper_head.",
    ),
    # Bounded reach repair: update only the Teacher/Student action heads.  In
    # particular, Teacher log_std and every visual, recurrent and auxiliary
    # component remain protected.
    "action_heads_only": (
        "mean.",
        "gripper_logit.",
        "policy.arm_action_head.",
        "policy.gripper_head.",
    ),
}


def _metric_leq_with_dtype_roundoff(
    value: torch.Tensor, limit: float
) -> torch.Tensor:
    """Compare a measured metric with a physical limit plus dtype roundoff.

    The slack is bounded to eight machine epsilons at unit scale.  For the
    live float32 grasp-distance tensor this is below one micrometre, so it
    only absorbs arithmetic/FK roundoff and does not relax the millimetre-
    scale physical close shell.
    """

    if not torch.is_floating_point(value):
        raise ValueError("metric tolerance comparison requires floating point")
    if not math.isfinite(float(limit)):
        raise ValueError("metric tolerance limit must be finite")
    roundoff = 8.0 * torch.finfo(value.dtype).eps * max(1.0, abs(float(limit)))
    return value <= float(limit) + roundoff


def accumulate_sparse_validation_metrics(
    totals: dict[str, float],
    counts: dict[str, float],
    metrics: Mapping[str, float],
) -> None:
    """Accumulate sparse vision errors with their true selected-row counts.

    A near-contact metric is absent when a mini-batch has no matching row.
    When present, its companion ``vision/*_samples/<bin>`` value is the only
    valid denominator; averaging mini-batch means would over-weight batches
    containing a single rare row.
    """

    for name, value in metrics.items():
        sample_key = None
        if name.startswith("vision/"):
            if "_xyz_error_mm/" in name:
                sample_key = name.replace("_xyz_error_mm/", "_samples/", 1)
            elif name.endswith("_xyz_error_mm"):
                sample_key = name.removesuffix("_xyz_error_mm") + "_samples"
            else:
                for axis in ("x", "y", "z"):
                    token = f"_error_{axis}_mm"
                    if f"{token}/" in name:
                        sample_key = name.replace(f"{token}/", "_samples/", 1)
                        break
                    if name.endswith(token):
                        sample_key = name.removesuffix(token) + "_samples"
                        break
        sample_count = (
            float(metrics[sample_key])
            if sample_key is not None and sample_key in metrics
            else None
        )
        if sample_count is not None and sample_count > 0.0:
            totals[name] = totals.get(name, 0.0) + float(value) * sample_count
            counts[name] = counts.get(name, 0.0) + sample_count
        else:
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0.0) + 1.0


def mean_sparse_validation_metrics(
    totals: Mapping[str, float], counts: Mapping[str, float]
) -> dict[str, float]:
    """Finalize metrics accumulated by :func:`accumulate_sparse_validation_metrics`."""

    return {
        name: value / float(counts[name])
        for name, value in totals.items()
        if counts.get(name, 0.0) > 0.0
    }


def deployable_close_frame_geometry(
    *,
    predicted_cube_xyz_root_m: torch.Tensor,
    grasp_center_position_root_m: torch.Tensor,
    reference_grasp_offset_root_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deployable close-frame relative geometry and scalar errors.

    This helper deliberately accepts a *predicted* cube position and a live-FK
    distal-pad midpoint.  Supplying simulator cube state is neither required
    nor supported by the actor/runtime call site.  The reference offset is a
    calibration constant, not privileged episode state.
    """

    if predicted_cube_xyz_root_m.shape != grasp_center_position_root_m.shape:
        raise ValueError("predicted cube and grasp-center tensors must match")
    if predicted_cube_xyz_root_m.shape[-1] != 3:
        raise ValueError("predicted cube and grasp-center tensors must end in xyz")
    if reference_grasp_offset_root_m.numel() != 3 and (
        reference_grasp_offset_root_m.shape != predicted_cube_xyz_root_m.shape
    ):
        raise ValueError("reference grasp offset must be xyz or match the batch")
    if not bool(torch.isfinite(predicted_cube_xyz_root_m).all()):
        raise ValueError("predicted cube position contains non-finite values")
    if not bool(torch.isfinite(grasp_center_position_root_m).all()):
        raise ValueError("grasp-center position contains non-finite values")
    relative_xyz = predicted_cube_xyz_root_m - grasp_center_position_root_m
    reference = reference_grasp_offset_root_m.to(
        device=relative_xyz.device, dtype=relative_xyz.dtype
    )
    if reference.numel() == 3:
        reference = reference.reshape(*([1] * (relative_xyz.ndim - 1)), 3)
    first_close_distance_m = torch.linalg.vector_norm(relative_xyz, dim=-1)
    close_frame_relative_error_m = torch.linalg.vector_norm(
        relative_xyz - reference, dim=-1
    )
    return relative_xyz, first_close_distance_m, close_frame_relative_error_m


def grasp_ready_supervision_target(
    *,
    cube_position_root_m: torch.Tensor,
    grasp_center_position_root_m: torch.Tensor,
    reference_grasp_offset_root_m: torch.Tensor,
    head_visible: torch.Tensor,
    depth_valid: torch.Tensor,
    safe_for_contact: torch.Tensor,
    gripper_open: torch.Tensor,
    grasp_ready_tolerance_m: float,
    alignment_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a learner-only grasp-ready label from simulator evidence.

    The returned label may supervise the deployable visual head, but none of
    its privileged inputs are part of the actor observation.  Runtime uses the
    head's probability plus predicted relative geometry instead.
    """

    relative, _raw_distance, grasp_error = deployable_close_frame_geometry(
        predicted_cube_xyz_root_m=cube_position_root_m,
        grasp_center_position_root_m=grasp_center_position_root_m,
        reference_grasp_offset_root_m=reference_grasp_offset_root_m,
    )
    del relative, _raw_distance
    if grasp_ready_tolerance_m <= 0.0:
        raise ValueError("grasp-ready tolerance must be positive")
    expected = grasp_error.shape
    masks = {
        "head_visible": head_visible,
        "depth_valid": depth_valid,
        "safe_for_contact": safe_for_contact,
        "gripper_open": gripper_open,
    }
    if alignment_valid is not None:
        masks["alignment_valid"] = alignment_valid
    normalized: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        if mask.shape == (*expected, 1):
            mask = mask.squeeze(-1)
        if mask.shape != expected:
            raise ValueError(f"{name} must match grasp-ready batch dimensions")
        normalized[name] = mask.to(device=grasp_error.device, dtype=torch.bool)
    target = (
        normalized["head_visible"]
        & normalized["depth_valid"]
        & normalized["safe_for_contact"]
        & normalized["gripper_open"]
        & _metric_leq_with_dtype_roundoff(
            grasp_error, grasp_ready_tolerance_m
        )
    )
    if alignment_valid is not None:
        target &= normalized["alignment_valid"]
    return target


@dataclass(frozen=True)
class G2VisionSafeCloseGateResult:
    action: torch.Tensor
    eligible_first_close_mask: torch.Tensor
    accepted_first_close_mask: torch.Tensor
    rejected_first_close_mask: torch.Tensor
    intervention_mask: torch.Tensor


def apply_g2_vision_safe_close_gate(
    action: torch.Tensor,
    *,
    policy_mask: torch.Tensor,
    controller_authority_mask: torch.Tensor,
    current_gripper_latch: torch.Tensor,
    grasp_ready_probability: torch.Tensor,
    grasp_ready_persistent: torch.Tensor,
    close_timing_probability: torch.Tensor,
    predicted_contact_steps: torch.Tensor,
    contact_time_gate_enabled: bool,
    minimum_predicted_contact_steps: float,
    maximum_predicted_contact_steps: float,
    head_valid: torch.Tensor,
    relative_geometry_valid: torch.Tensor,
    relative_grasp_error_m: torch.Tensor,
    ee_speed_m_s: torch.Tensor,
    probability_threshold: float,
    timing_probability_threshold: float,
    maximum_ee_speed_m_s: float,
) -> G2VisionSafeCloseGateResult:
    """Accept only a deployable, motion-safe first close request.

    This gate never generates CLOSE.  It only preserves or rejects the
    actor's first open-to-close request.  Once the external actuator latch is
    closed, subsequent close maintenance passes through unchanged.
    """

    if action.ndim != 2 or action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("safe-close action must be [N,7]")
    batch = action.shape[0]
    for name, value in {
        "policy_mask": policy_mask,
        "controller_authority_mask": controller_authority_mask,
        "grasp_ready_persistent": grasp_ready_persistent,
        "head_valid": head_valid,
        "relative_geometry_valid": relative_geometry_valid,
    }.items():
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean [N]")
    for name, value in {
        "current_gripper_latch": current_gripper_latch,
        "grasp_ready_probability": grasp_ready_probability,
        "close_timing_probability": close_timing_probability,
        "predicted_contact_steps": predicted_contact_steps,
        "relative_grasp_error_m": relative_grasp_error_m,
        "ee_speed_m_s": ee_speed_m_s,
    }.items():
        if value.shape != (batch,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite [N]")
    if bool((policy_mask & controller_authority_mask).any()):
        raise ValueError("policy and controller authority masks overlap")
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("grasp-ready probability threshold must be in (0,1)")
    if not 0.0 < timing_probability_threshold < 1.0:
        raise ValueError("close-timing probability threshold must be in (0,1)")
    if maximum_ee_speed_m_s <= 0.0:
        raise ValueError("safe-close physical thresholds must be positive")
    if not 0.0 <= minimum_predicted_contact_steps < maximum_predicted_contact_steps:
        raise ValueError("predicted contact-step bounds are invalid")

    requested_close = action[:, G2_VISUAL_GRIPPER_ACTION_INDEX] < 0.0
    currently_open = current_gripper_latch >= 0.0
    eligible_first_close = policy_mask & currently_open & requested_close
    safe = (
        head_valid
        & grasp_ready_persistent
        & (grasp_ready_probability >= float(probability_threshold))
        & (close_timing_probability >= float(timing_probability_threshold))
        # The historical isotropic 16 mm CONTACT shell is telemetry only. It
        # is not a CLOSE_NOW authority. Geometry validity here means the
        # deployable Head prediction and live pad-FK transform are available
        # and finite; temporal heads own the close timing decision.
        & relative_geometry_valid
        & (
            (~torch.full_like(head_valid, bool(contact_time_gate_enabled)))
            | (
                (predicted_contact_steps >= float(minimum_predicted_contact_steps))
                & (predicted_contact_steps <= float(maximum_predicted_contact_steps))
            )
        )
        & _metric_leq_with_dtype_roundoff(ee_speed_m_s, maximum_ee_speed_m_s)
    )
    accepted = eligible_first_close & safe
    rejected = eligible_first_close & ~safe
    projected = action.clone()
    projected[rejected, G2_VISUAL_GRIPPER_ACTION_INDEX] = 1.0
    intervention = (projected != action).any(dim=-1)
    if not torch.equal(
        projected[controller_authority_mask], action[controller_authority_mask]
    ):
        raise RuntimeError("safe-close gate touched controller-owned row")
    return G2VisionSafeCloseGateResult(
        action=projected,
        eligible_first_close_mask=eligible_first_close,
        accepted_first_close_mask=accepted,
        rejected_first_close_mask=rejected,
        intervention_mask=intervention,
    )


G2_RECURRENT_STUDENT_SCHEMA = "g2_temporal_safe_close_gru_student_v15"
G2_RECURRENT_VISUAL_POLICY_SCHEMA = "g2_temporal_safe_close_gru_policy_v15"
G2_HEAD_OBSERVATION_MODES = (
    "cube_detector_depth_xyz",
    # Retained as accepted CLI aliases so old launch scripts fail on the
    # checkpoint schema rather than at argument parsing.  All aliases use the
    # detector-only Head path below; raw Head tokens are never exposed to SAC.
    "full_rgbd",
    "rgb_mask_depth_median_centroid",
)
# Single source of truth for the tensors materialized by the fresh-agent
# demonstration BC path.  Keep the privileged Teacher state here even though
# it is never exposed to the deployable actor: it supplies detector/auxiliary
# labels and the recorded grasp-center/phase targets used during BC.
G2_DEMONSTRATION_BC_BATCH_TENSOR_FIELDS = (
    "rgbd_u8",
    "deployable_proprioception",
    "grasp_center_position_root_m",
    "expert_action_target",
    "teacher_state_target",
    "expert_confidence",
    "episode_id",
    "sequence_step",
    "padding_mask",
    "hidden_reset_mask",
    "sequence_lengths",
    "relative_pose_target",
    "contact_target",
    "stable_grasp_target",
    "depth_validity_target",
    "future_safe_close_target",
    "future_safe_close_onset_target",
    "future_safe_close_ready_target",
    "future_safe_close_ready_valid",
    "close_timing_target",
    "close_timing_valid",
    "contact_lead_steps_target",
    "contact_lead_steps_valid",
)
G2_DEPLOYABLE_PHASES = ("REACH", "PRE_GRASP", "CONTACT", "STABLE_GRASP", "LIFT")
G2_DEPLOYABLE_PHASE_DIM = len(G2_DEPLOYABLE_PHASES)
G2_HEAD_DEPTH_PROFILES = (
    "legacy",
    "hrnet_fpn_spatial_softmax",
    "coord_gn_silu",
    "coord_gn_silu_unet_aux",
    "modified_resnet18_spatial_softmax",
    "lite_unet_fpn_spatial_softmax",
)
G2_CAMERA_ENCODER_PROFILES = {
    "baseline_4layer": (32, 64, 96, 128),
    "cnn5_160": (32, 64, 96, 128, 160),
    "cnn7_160": (32, 64, 96, 128, 160, 160, 160),
    "cnn9_160": (32, 64, 96, 128, 160, 160, 160, 160, 160),
    "cnn11_160": (32, 64, 96, 128, 160, 160, 160, 160, 160, 160, 160),
    "cnn13_160": (
        32, 64, 96, 128, 160, 160, 160, 160, 160, 160, 160, 160, 160
    ),
    # Unique signatures select residual encoders in `_camera_encoder` rather
    # than plain CNN stacks. ResNet-13 has six basic blocks (1+2+2+1).
    "resnet13": (48, 48, 96, 160, 256),
    # A unique channel signature selects the actual ResNet-18 implementation
    # in `_camera_encoder`; it is not interpreted as a five-layer plain CNN.
    "resnet18": (64, 64, 128, 256, 512),
}


def corrected_gripper_open_targets(
    *,
    expert_action: torch.Tensor,
    teacher_state: torch.Tensor,
    predicted_cube_position_root_m: torch.Tensor | None,
    grasp_center_position_root_m: torch.Tensor,
    stable_grasp: torch.Tensor | None,
    contact: torch.Tensor | None,
    reference_grasp_offset_root_m: torch.Tensor,
    grasp_ready_tolerance_m: float,
    future_safe_close: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return BC open/close labels with unsafe early closes relabeled open.

    Cube GT is forbidden even for this BC classifier label.  A recorded close
    remains positive only in physical contact/stable state or when bounded
    future physical evidence proves that starting the close was safe.  The
    calibrated static geometry shell is retained as telemetry only: CONTACT
    geometry is not a valid authority for the earlier CLOSE_READY decision.
    """

    if expert_action.shape[:-1] != teacher_state.shape[:-1]:
        raise ValueError("expert action and Teacher state leading dimensions differ")
    if expert_action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("expert action must use the canonical 7-D contract")
    if teacher_state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("Teacher state must use the canonical privileged contract")
    if grasp_center_position_root_m.shape != (*teacher_state.shape[:-1], 3):
        raise ValueError("grasp center must match Teacher sequence dimensions")
    if reference_grasp_offset_root_m.numel() != 3:
        raise ValueError("reference grasp offset must contain xyz")
    if grasp_ready_tolerance_m <= 0.0:
        raise ValueError("gripper supervision tolerances must be positive")
    teacher_slices = G2TeacherObservationContract().slices
    if predicted_cube_position_root_m is None:
        cube_minus_grasp = torch.zeros_like(grasp_center_position_root_m)
        raw_distance = torch.full_like(
            grasp_center_position_root_m[..., :1], float("inf")
        )
        alignment_error = torch.full_like(
            grasp_center_position_root_m[..., :1], float("inf")
        )
    else:
        if predicted_cube_position_root_m.shape != grasp_center_position_root_m.shape:
            raise ValueError("predicted cube and grasp center shapes differ")
        if not bool(torch.isfinite(predicted_cube_position_root_m).all()):
            raise ValueError("predicted cube position contains non-finite values")
        cube_minus_grasp = (
            predicted_cube_position_root_m.detach()
            - grasp_center_position_root_m
        )
        raw_distance = torch.linalg.vector_norm(
            cube_minus_grasp, dim=-1, keepdim=True
        )
    reference = reference_grasp_offset_root_m.to(
        device=teacher_state.device, dtype=teacher_state.dtype
    ).reshape(*([1] * (teacher_state.ndim - 1)), 3)
    if predicted_cube_position_root_m is not None:
        alignment_error = torch.linalg.vector_norm(
            cube_minus_grasp - reference, dim=-1, keepdim=True
        )
    contact_mask = teacher_state[
        ..., teacher_slices["bilateral_contact_features"]
    ].amin(dim=-1, keepdim=True) > 0.5
    if isinstance(contact, torch.Tensor):
        if contact.shape[:-1] != teacher_state.shape[:-1]:
            raise ValueError("contact target sequence dimensions differ")
        contact_mask = contact_mask | (contact > 0.5).all(dim=-1, keepdim=True)
    stable_mask = torch.zeros_like(contact_mask)
    if isinstance(stable_grasp, torch.Tensor):
        if stable_grasp.shape[:-1] != teacher_state.shape[:-1]:
            raise ValueError("stable target sequence dimensions differ")
        stable_mask = stable_grasp > 0.5
    # The calibrated reference offset is generally non-zero.  A raw
    # cube-to-pad distance cap would reject valid poses whenever the offset
    # itself is larger than that cap.  Grasp readiness is therefore defined
    # solely in the calibrated error frame.
    grasp_ready = alignment_error <= float(grasp_ready_tolerance_m)
    recorded_close = expert_action[..., 6:7] < 0.0
    future_safe_mask = torch.zeros_like(contact_mask)
    if isinstance(future_safe_close, torch.Tensor):
        if future_safe_close.shape != contact_mask.shape:
            raise ValueError("future-safe close target must match Teacher sequence dimensions")
        future_safe_mask = future_safe_close.to(
            device=teacher_state.device, dtype=torch.bool
        )
    # A position-controlled gripper starts closing before measured contact.
    # The bounded future-safe label is computed offline from the *recorded*
    # expert edge, collision authority and later physical contact.  It is a
    # supervision target only; no future state or cube GT enters the actor.
    allowed_close = contact_mask | stable_mask | future_safe_mask
    corrected_close = recorded_close & allowed_close
    target_open = (~corrected_close).to(torch.float32)
    return target_open, {
        "recorded_close": recorded_close,
        "corrected_close": corrected_close,
        "premature_close_relabel": recorded_close & (~allowed_close),
        "grasp_ready": grasp_ready,
        "contact_or_stable": contact_mask | stable_mask,
        "future_safe_close": future_safe_mask,
        # Raw distance is retained as telemetry only.  It has no authority in
        # the grasp-ready or corrected-close decision above.
        "distance_m": raw_distance,
        "alignment_error_m": alignment_error,
    }


def bounded_future_safe_close_ready_rows(
    *,
    current_gripper_open: torch.Tensor,
    expert_first_close: torch.Tensor,
    padding_mask: torch.Tensor,
    geometry_valid: torch.Tensor,
    visibility_valid: torch.Tensor,
    depth_valid: torch.Tensor,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    safe_for_contact: torch.Tensor,
    episode_id: torch.Tensor,
    sequence_step: torch.Tensor,
    source_dataset_index: torch.Tensor,
    first_close_proximity_steps: int,
    contact_horizon_steps: int,
    stable_horizon_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Label OPEN rows whose bounded observed future reaches safe grasp evidence.

    This is an offline supervision authority.  It canonicalizes overlapping
    recurrent windows by physical row identity and looks forward only inside
    the same dataset/episode.  A positive row must be OPEN now and every row
    through ordered bilateral CONTACT then STABLE_GRASP evidence must be
    present and collision-safe.  A candidate is positive only when an expert
    first-close edge occurs within ``K`` rows.  This prevents the long physical
    actuation horizon from labeling arbitrarily early approach rows positive.
    Future state is returned only as a learner target and is never an
    actor/runtime input.
    """

    if current_gripper_open.ndim != 3 or current_gripper_open.shape[-1] != 1:
        raise ValueError("future-safe ready OPEN mask must be [W,T,1]")
    windows, steps = current_gripper_open.shape[:2]
    row_shape = (windows, steps, 1)
    for name, value in (
        ("expert_first_close", expert_first_close),
        ("geometry_valid", geometry_valid),
        ("visibility_valid", visibility_valid),
        ("depth_valid", depth_valid),
        ("stable_grasp", stable_grasp),
        ("safe_for_contact", safe_for_contact),
    ):
        if value.shape != row_shape:
            raise ValueError(f"{name} must be [W,T,1]")
    if bilateral_contact.shape[:2] != (windows, steps):
        raise ValueError("bilateral contact must share [W,T]")
    if padding_mask.shape != (windows, steps):
        raise ValueError("padding mask must be [W,T]")
    if episode_id.shape != (windows, steps) or sequence_step.shape != (windows, steps):
        raise ValueError("episode id and sequence step must be [W,T]")
    if source_dataset_index.shape != (windows,):
        raise ValueError("source dataset index must be [W]")
    if first_close_proximity_steps < 0:
        raise ValueError("first-close proximity steps cannot be negative")
    if contact_horizon_steps <= 0 or stable_horizon_steps <= 0:
        raise ValueError("contact and stable horizons must be positive")

    device = current_gripper_open.device
    opened = current_gripper_open.detach().cpu().numpy()[..., 0].astype(bool)
    first_close = expert_first_close.detach().cpu().numpy()[..., 0].astype(bool)
    padding = padding_mask.detach().cpu().numpy().astype(bool)
    geometry = geometry_valid.detach().cpu().numpy()[..., 0].astype(bool)
    visible = visibility_valid.detach().cpu().numpy()[..., 0].astype(bool)
    depth = depth_valid.detach().cpu().numpy()[..., 0].astype(bool)
    safe = safe_for_contact.detach().cpu().numpy()[..., 0].astype(bool)
    bilateral = bilateral_contact.detach().cpu().numpy()
    contact = (
        bilateral > 0.5
        if bilateral.ndim == 2
        else np.all(bilateral > 0.5, axis=-1)
    )
    stable = stable_grasp.detach().cpu().numpy()[..., 0] > 0.5
    episode = episode_id.detach().cpu().numpy()
    step_index = sequence_step.detach().cpu().numpy()
    source = source_dataset_index.detach().cpu().numpy()

    rows: dict[tuple[int, int, int], dict[str, bool]] = {}
    for window in range(windows):
        for offset in range(steps):
            if not padding[window, offset]:
                continue
            key = (
                int(source[window]),
                int(episode[window, offset]),
                int(step_index[window, offset]),
            )
            values = {
                "open": bool(opened[window, offset]),
                "first_close": bool(first_close[window, offset]),
                "geometry": bool(geometry[window, offset]),
                "visible": bool(visible[window, offset]),
                "depth": bool(depth[window, offset]),
                "safe": bool(safe[window, offset]),
                "contact": bool(contact[window, offset]),
                "stable": bool(stable[window, offset]),
            }
            prior = rows.get(key)
            if prior is None:
                rows[key] = values
            elif prior != values:
                raise ValueError("overlapping future-safe ready labels disagree")

    valid_keys: set[tuple[int, int, int]] = set()
    positive_keys: dict[tuple[int, int, int], int] = {}
    missing_future_rows = 0
    unsafe_future_rows = 0
    contact_only_rows = 0
    no_near_first_close_rows = 0
    for key, row in sorted(rows.items()):
        if not (
            row["open"]
            and row["geometry"]
            and row["visible"]
            and row["depth"]
            and row["safe"]
        ):
            continue
        dataset, episode_value, start_step = key
        proximity = [
            rows.get((dataset, episode_value, step))
            for step in range(
                start_step, start_step + first_close_proximity_steps + 1
            )
        ]
        if any(item is None for item in proximity):
            missing_future_rows += 1
            continue
        onset_offset = next(
            (
                offset
                for offset, item in enumerate(proximity)
                if item is not None and item["first_close"]
            ),
            None,
        )
        if onset_offset is None:
            no_near_first_close_rows += 1
            valid_keys.add(key)
            continue
        onset_step = start_step + int(onset_offset)
        maximum_required_step = onset_step + max(
            contact_horizon_steps, stable_horizon_steps
        )
        observed = {
            step: rows.get((dataset, episode_value, step))
            for step in range(start_step, maximum_required_step + 1)
        }
        contact_step = next(
            (
                step
                for step in range(onset_step, onset_step + contact_horizon_steps + 1)
                if observed.get(step) is not None and observed[step]["contact"]
            ),
            None,
        )
        stable_step = next(
            (
                step
                for step in range(onset_step, onset_step + stable_horizon_steps + 1)
                if observed.get(step) is not None and observed[step]["stable"]
            ),
            None,
        )
        if contact_step is not None and stable_step is not None and stable_step >= contact_step:
            path = [
                rows.get((dataset, episode_value, step))
                for step in range(start_step, stable_step + 1)
            ]
            if all(item is not None and item["safe"] for item in path):
                valid_keys.add(key)
                positive_keys[key] = int(onset_offset)
                continue
        unsafe_step = next(
            (
                step
                for step, item in observed.items()
                if item is not None and not item["safe"]
            ),
            None,
        )
        if unsafe_step is not None:
            unsafe_future_rows += 1
            valid_keys.add(key)
            continue
        contact_complete = rows.get(
            (dataset, episode_value, onset_step + contact_horizon_steps)
        ) is not None
        stable_complete = rows.get(
            (dataset, episode_value, onset_step + stable_horizon_steps)
        ) is not None
        if contact_complete and stable_complete:
            valid_keys.add(key)
            contact_only_rows += int(contact_step is not None and stable_step is None)
        else:
            missing_future_rows += 1

    target = torch.zeros(row_shape, dtype=torch.bool, device=device)
    valid = torch.zeros_like(target)
    for window in range(windows):
        for offset in range(steps):
            if not padding[window, offset]:
                continue
            key = (
                int(source[window]),
                int(episode[window, offset]),
                int(step_index[window, offset]),
            )
            valid[window, offset, 0] = key in valid_keys
            if key in positive_keys:
                target[window, offset, 0] = True
    unique_leads = list(positive_keys.values())
    metrics = {
        "future_safe_ready/valid_unique_rows": float(len(valid_keys)),
        "future_safe_ready/positive_unique_rows": float(len(positive_keys)),
        "future_safe_ready/negative_unique_rows": float(
            len(valid_keys) - len(positive_keys)
        ),
        "future_safe_ready/positive_fraction": float(
            len(positive_keys) / max(len(valid_keys), 1)
        ),
        "future_safe_ready/missing_future_rows": float(missing_future_rows),
        "future_safe_ready/unsafe_future_rows": float(unsafe_future_rows),
        "future_safe_ready/contact_only_negative_rows": float(contact_only_rows),
        "future_safe_ready/no_near_first_close_negative_rows": float(
            no_near_first_close_rows
        ),
        "future_safe_ready/right_censored_unique_rows": float(
            sum(
                row["open"] and row["geometry"] and row["visible"]
                and row["depth"] and row["safe"]
                for row in rows.values()
            )
            - len(valid_keys)
        ),
        "future_safe_ready/lead_steps_mean": float(
            np.mean(unique_leads) if unique_leads else 0.0
        ),
        "future_safe_ready/lead_steps_median": float(
            np.median(unique_leads) if unique_leads else 0.0
        ),
        "future_safe_ready/first_close_proximity_steps": float(
            first_close_proximity_steps
        ),
        "future_safe_ready/contact_horizon_steps": float(contact_horizon_steps),
        "future_safe_ready/stable_horizon_steps": float(stable_horizon_steps),
    }
    return target, valid, metrics


def temporal_close_auxiliary_targets(
    *,
    expert_first_close: torch.Tensor,
    current_gripper_open: torch.Tensor,
    padding_mask: torch.Tensor,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    safe_for_contact: torch.Tensor,
    episode_id: torch.Tensor,
    sequence_step: torch.Tensor,
    source_dataset_index: torch.Tensor,
    timing_support_steps: int = 6,
    timing_temperature_steps: float = 2.5,
    contact_horizon_steps: int = 64,
    stable_horizon_steps: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build soft close-now and event-specific contact-time supervision.

    Overlapping recurrent windows are canonicalized by physical row identity.
    Timing targets exist only on OPEN/pre-onset rows and are Gaussian in the
    distance to the expert first-close edge.  A near-onset target is valid only
    when that close is followed by ordered CONTACT then STABLE without an
    unsafe intermediate row.  Rows whose required future is unavailable are
    right-censored rather than converted into negatives.  Contact lead is
    supervised once, at the accepted expert close edge, from the episode's
    actual measured bilateral-contact event.
    """

    if expert_first_close.ndim != 3 or expert_first_close.shape[-1] != 1:
        raise ValueError("expert first-close must be [W,T,1]")
    windows, steps = expert_first_close.shape[:2]
    if padding_mask.shape != (windows, steps):
        raise ValueError("padding mask must be [W,T]")
    if safe_for_contact.shape != (windows, steps, 1):
        raise ValueError("safe-for-contact must be [W,T,1]")
    if current_gripper_open.shape != (windows, steps, 1):
        raise ValueError("current gripper-open state must be [W,T,1]")
    if stable_grasp.shape != (windows, steps, 1):
        raise ValueError("stable grasp must be [W,T,1]")
    if bilateral_contact.shape[:2] != (windows, steps):
        raise ValueError("bilateral contact must share [W,T]")
    if timing_support_steps < 0 or timing_temperature_steps <= 0.0:
        raise ValueError("timing support/temperature are invalid")
    if contact_horizon_steps <= 0 or stable_horizon_steps <= 0:
        raise ValueError("contact/stable horizons must be positive")

    first = expert_first_close.detach().cpu().numpy()[..., 0].astype(bool)
    gripper_open = current_gripper_open.detach().cpu().numpy()[..., 0].astype(bool)
    pad = padding_mask.detach().cpu().numpy().astype(bool)
    safe = safe_for_contact.detach().cpu().numpy()[..., 0].astype(bool)
    stable = stable_grasp.detach().cpu().numpy()[..., 0].astype(bool)
    bilateral = bilateral_contact.detach().cpu().numpy()
    contact = bilateral > 0.5 if bilateral.ndim == 2 else np.all(bilateral > 0.5, axis=-1)
    episode = episode_id.detach().cpu().numpy()
    sequence = sequence_step.detach().cpu().numpy()
    source = source_dataset_index.detach().cpu().numpy()
    rows: dict[tuple[int, int, int], dict[str, bool]] = {}
    for window in range(windows):
        for offset in range(steps):
            if not pad[window, offset]:
                continue
            key = (int(source[window]), int(episode[window, offset]), int(sequence[window, offset]))
            value = {
                "first": bool(first[window, offset]),
                "open": bool(gripper_open[window, offset]),
                "safe": bool(safe[window, offset]),
                "contact": bool(contact[window, offset]),
                "stable": bool(stable[window, offset]),
            }
            prior = rows.get(key)
            if prior is not None and prior != value:
                raise ValueError("overlapping temporal close labels disagree")
            rows[key] = value

    timing_values: dict[tuple[int, int, int], float] = {}
    timing_valid_keys: set[tuple[int, int, int]] = set()
    contact_values: dict[tuple[int, int, int], float] = {}
    contact_valid_keys: set[tuple[int, int, int]] = set()
    contact_to_stable: list[int] = []
    right_censored = 0
    episodes = sorted({(key[0], key[1]) for key in rows})
    for dataset_value, episode_value in episodes:
        episode_rows = {key[2]: value for key, value in rows.items() if key[:2] == (dataset_value, episode_value)}
        if not episode_rows:
            continue
        first_steps = sorted(step for step, value in episode_rows.items() if value["first"])
        for start_step, row in sorted(episode_rows.items()):
            # Timing is a close-now decision defined only while the gripper is
            # open.  HOLD_CLOSED and post-onset rows are masked, not converted
            # into easy negative examples.
            if not row["open"]:
                continue
            if row["first"]:
                # The edge itself is the maximum soft target and remains an
                # OPEN->CLOSE decision row, not a maintain-closed row.
                onset_step = start_step
            else:
                onset_step = next((step for step in first_steps if 0 <= step - start_step <= timing_support_steps), None)
            if onset_step is None:
                support_end = start_step + timing_support_steps
                if all(step in episode_rows for step in range(start_step, support_end + 1)):
                    timing_valid_keys.add((dataset_value, episode_value, start_step))
                    timing_values[(dataset_value, episode_value, start_step)] = 0.0
                else:
                    right_censored += 1
                continue
            contact_step = next(
                (step for step in range(onset_step, onset_step + contact_horizon_steps + 1)
                 if episode_rows.get(step, {}).get("contact", False)),
                None,
            )
            stable_step = next(
                (step for step in range(onset_step, onset_step + stable_horizon_steps + 1)
                 if episode_rows.get(step, {}).get("stable", False)),
                None,
            )
            complete = (
                onset_step + contact_horizon_steps in episode_rows
                and onset_step + stable_horizon_steps in episode_rows
            )
            ordered_success = contact_step is not None and stable_step is not None and stable_step >= contact_step
            unsafe = False
            if ordered_success:
                unsafe = any(
                    step not in episode_rows or not episode_rows[step]["safe"]
                    for step in range(start_step, stable_step + 1)
                )
            if ordered_success and not unsafe:
                key = (dataset_value, episode_value, start_step)
                timing_valid_keys.add(key)
                delta = float(start_step - onset_step)
                timing_values[key] = math.exp(
                    -(delta * delta) / (2.0 * timing_temperature_steps * timing_temperature_steps)
                )
                if start_step == onset_step:
                    contact_valid_keys.add(key)
                    contact_values[key] = float(contact_step - onset_step)
                    contact_to_stable.append(int(stable_step - contact_step))
            elif unsafe or complete:
                key = (dataset_value, episode_value, start_step)
                timing_valid_keys.add(key)
                timing_values[key] = 0.0
            else:
                right_censored += 1

    shape = (windows, steps, 1)
    device = expert_first_close.device
    timing_target = torch.zeros(shape, dtype=torch.float32, device=device)
    timing_valid = torch.zeros(shape, dtype=torch.bool, device=device)
    contact_target = torch.zeros(shape, dtype=torch.float32, device=device)
    contact_valid = torch.zeros(shape, dtype=torch.bool, device=device)
    for window in range(windows):
        for offset in range(steps):
            if not pad[window, offset]:
                continue
            key = (int(source[window]), int(episode[window, offset]), int(sequence[window, offset]))
            if key in timing_valid_keys:
                timing_valid[window, offset, 0] = True
                timing_target[window, offset, 0] = float(timing_values[key])
            if key in contact_valid_keys:
                contact_valid[window, offset, 0] = True
                contact_target[window, offset, 0] = float(contact_values[key])
    lead_values = list(contact_values.values())
    metrics = {
        "close_timing/valid_unique_rows": float(len(timing_valid_keys)),
        "close_timing/soft_positive_unique_rows": float(sum(value > 0.0 for value in timing_values.values())),
        "close_timing/right_censored_unique_rows": float(right_censored),
        "close_timing/support_steps": float(timing_support_steps),
        "close_timing/temperature_steps": float(timing_temperature_steps),
        "contact_time/event_count": float(len(contact_values)),
        "contact_time/lead_steps_mean": float(np.mean(lead_values) if lead_values else 0.0),
        "contact_time/lead_steps_median": float(np.median(lead_values) if lead_values else 0.0),
        "contact_time/contact_to_stable_steps_mean": float(np.mean(contact_to_stable) if contact_to_stable else 0.0),
        "contact_time/contact_to_stable_steps_median": float(np.median(contact_to_stable) if contact_to_stable else 0.0),
    }
    return timing_target, timing_valid, contact_target, contact_valid, metrics


def close_ready_geometry_audit_metrics(
    *,
    relative_grasp_xyz_m: torch.Tensor,
    first_close_onset: torch.Tensor,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    padding_mask: torch.Tensor,
    episode_id: torch.Tensor,
    sequence_step: torch.Tensor,
    source_dataset_index: torch.Tensor,
) -> dict[str, float]:
    """Return deduplicated first-close/contact/stable geometry telemetry."""

    if relative_grasp_xyz_m.ndim != 3 or relative_grasp_xyz_m.shape[-1] != 3:
        raise ValueError("relative grasp geometry must be [W,T,3]")
    windows, steps = relative_grasp_xyz_m.shape[:2]
    if first_close_onset.shape != (windows, steps, 1):
        raise ValueError("first-close onset must be [W,T,1]")
    if stable_grasp.shape != (windows, steps, 1):
        raise ValueError("stable grasp must be [W,T,1]")
    if bilateral_contact.shape[:2] != (windows, steps):
        raise ValueError("bilateral contact must share [W,T]")
    if padding_mask.shape != (windows, steps):
        raise ValueError("padding mask must be [W,T]")

    relative = relative_grasp_xyz_m.detach().cpu().numpy()
    onset = first_close_onset.detach().cpu().numpy()[..., 0].astype(bool)
    bilateral = bilateral_contact.detach().cpu().numpy()
    contact = (
        bilateral > 0.5
        if bilateral.ndim == 2
        else np.all(bilateral > 0.5, axis=-1)
    )
    stable = stable_grasp.detach().cpu().numpy()[..., 0] > 0.5
    padding = padding_mask.detach().cpu().numpy().astype(bool)
    episode = episode_id.detach().cpu().numpy()
    sequence = sequence_step.detach().cpu().numpy()
    source = source_dataset_index.detach().cpu().numpy()
    rows: dict[tuple[int, int, int], tuple[np.ndarray, bool, bool, bool]] = {}
    for window in range(windows):
        for offset in range(steps):
            if not padding[window, offset]:
                continue
            key = (
                int(source[window]),
                int(episode[window, offset]),
                int(sequence[window, offset]),
            )
            value = (
                relative[window, offset],
                bool(onset[window, offset]),
                bool(contact[window, offset]),
                bool(stable[window, offset]),
            )
            prior = rows.get(key)
            if prior is not None and (
                not np.allclose(prior[0], value[0], atol=1.0e-7, rtol=0.0)
                or prior[1:] != value[1:]
            ):
                raise ValueError("overlapping close geometry rows disagree")
            rows[key] = value

    metrics: dict[str, float] = {}
    for name, selector in (
        ("first_close", 1),
        ("contact", 2),
        ("stable", 3),
    ):
        selected = np.asarray(
            [value[0] for value in rows.values() if value[selector]],
            dtype=np.float64,
        ).reshape(-1, 3)
        prefix = f"grasp/{name}_geometry"
        metrics[f"{prefix}_count"] = float(len(selected))
        if not len(selected):
            continue
        selected_mm = selected * 1000.0
        norm_mm = np.linalg.norm(selected_mm, axis=-1)
        for axis, axis_name in enumerate(("x", "y", "z")):
            axis_values = selected_mm[:, axis]
            metrics[f"{prefix}_{axis_name}_mm_mean"] = float(axis_values.mean())
            metrics[f"{prefix}_{axis_name}_mm_median"] = float(
                np.median(axis_values)
            )
            metrics[f"{prefix}_{axis_name}_mm_std"] = float(axis_values.std())
            for quantile in (5, 25, 50, 75, 95):
                metrics[f"{prefix}_{axis_name}_mm_p{quantile:02d}"] = float(
                    np.percentile(axis_values, quantile)
                )
        metrics[f"{prefix}_norm_mm_mean"] = float(norm_mm.mean())
        metrics[f"{prefix}_norm_mm_median"] = float(np.median(norm_mm))
    return metrics


def bounded_future_safe_expert_close_rows(
    *,
    expert_action: torch.Tensor,
    previous_gripper_open: torch.Tensor,
    episode_outcome_success: torch.Tensor,
    padding_mask: torch.Tensor,
    expert_confidence: torch.Tensor,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    safe_for_contact: torch.Tensor,
    episode_id: torch.Tensor,
    sequence_step: torch.Tensor,
    source_dataset_index: torch.Tensor,
    maximum_lead_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build bounded close labels from later measured physical evidence.

    Overlapping recurrent windows duplicate physical rows.  This function
    first canonicalizes them by ``(dataset, episode, step)`` and then accepts
    an expert open->close edge only when bilateral contact or stable grasp is
    observed within ``maximum_lead_steps``, every intervening row has a valid,
    collision-free authority, and the recorded close command remains asserted
    continuously through that evidence row.  Re-opening starts a different
    physical attempt and must never let its later contact supervise the earlier
    edge.  The returned latched mask preserves the expert close command from
    the accepted edge through the evidence row.

    This is offline supervision, not an actor feature.  In particular, cube
    ground truth is neither accepted nor consulted here.
    """

    if expert_action.ndim != 3 or expert_action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("future-safe close expects expert actions [W,T,7]")
    windows, steps = expert_action.shape[:2]
    row_shape = (windows, steps, 1)
    for name, value in (
        ("previous_gripper_open", previous_gripper_open),
        ("expert_confidence", expert_confidence),
        ("stable_grasp", stable_grasp),
        ("safe_for_contact", safe_for_contact),
    ):
        if value.shape != row_shape:
            raise ValueError(f"{name} must be [W,T,1]")
    if bilateral_contact.shape[:2] != (windows, steps):
        raise ValueError("bilateral contact must share [W,T]")
    if padding_mask.shape != (windows, steps):
        raise ValueError("padding mask must be [W,T]")
    if episode_id.shape != (windows, steps) or sequence_step.shape != (windows, steps):
        raise ValueError("episode id and sequence step must be [W,T]")
    if episode_outcome_success.shape != (windows,):
        raise ValueError("episode outcome success must be [W]")
    if source_dataset_index.shape != (windows,):
        raise ValueError("source dataset index must be [W]")
    if maximum_lead_steps <= 0:
        raise ValueError("maximum close lead steps must be positive")

    device = expert_action.device
    recorded_close = (expert_action[..., 6:7] < 0.0).detach().cpu().numpy()
    previous_open = previous_gripper_open.detach().cpu().numpy().astype(bool)
    success = episode_outcome_success.detach().cpu().numpy().astype(bool)
    padding = padding_mask.detach().cpu().numpy().astype(bool)
    confidence = expert_confidence.detach().cpu().numpy() > 0.0
    bilateral = bilateral_contact.detach().cpu().numpy()
    if bilateral.ndim == 2:
        physical_contact = bilateral > 0.5
    else:
        physical_contact = np.all(bilateral > 0.5, axis=-1)
    stable = stable_grasp.detach().cpu().numpy()[..., 0] > 0.5
    safe = safe_for_contact.detach().cpu().numpy()[..., 0].astype(bool)
    episode = episode_id.detach().cpu().numpy()
    step_index = sequence_step.detach().cpu().numpy()
    source = source_dataset_index.detach().cpu().numpy()

    # Canonical row values must agree across overlapping windows.  Safety is
    # combined conservatively (all copies safe), while contact is positive if
    # any identical copy reports the measured event.
    rows: dict[tuple[int, int, int], dict[str, bool]] = {}
    onset_keys: set[tuple[int, int, int]] = set()
    for window in range(windows):
        for offset in range(steps):
            if not padding[window, offset]:
                continue
            key = (
                int(source[window]),
                int(episode[window, offset]),
                int(step_index[window, offset]),
            )
            item = rows.setdefault(
                key,
                {
                    "contact": False,
                    "bilateral": False,
                    "stable": False,
                    "safe": True,
                    "recorded_close": bool(recorded_close[window, offset, 0]),
                },
            )
            item["bilateral"] = bool(
                item["bilateral"] or physical_contact[window, offset]
            )
            item["stable"] = bool(item["stable"] or stable[window, offset])
            item["contact"] = bool(
                item["contact"] or item["bilateral"] or item["stable"]
            )
            item["safe"] = bool(item["safe"] and safe[window, offset])
            if item["recorded_close"] != bool(recorded_close[window, offset, 0]):
                raise ValueError("overlapping expert close labels disagree")
            if (
                success[window]
                and confidence[window, offset, 0]
                and recorded_close[window, offset, 0]
                and previous_open[window, offset, 0]
            ):
                onset_keys.add(key)

    accepted: dict[tuple[int, int, int], int] = {}
    rejected_no_contact = 0
    rejected_unsafe = 0
    rejected_reopened_before_contact = 0
    for key in sorted(onset_keys):
        dataset, episode_value, onset_step = key
        evidence_step = None
        unsafe = False
        reopened = False
        for candidate_step in range(onset_step, onset_step + maximum_lead_steps + 1):
            candidate = rows.get((dataset, episode_value, candidate_step))
            if candidate is None or not candidate["safe"]:
                unsafe = True
                break
            if not candidate["recorded_close"]:
                reopened = True
                break
            if candidate["contact"]:
                evidence_step = candidate_step
                break
        if evidence_step is None:
            if unsafe:
                rejected_unsafe += 1
            elif reopened:
                rejected_reopened_before_contact += 1
            else:
                rejected_no_contact += 1
            continue
        accepted[key] = evidence_step

    accepted_onset_keys = set(accepted)
    latched_keys: set[tuple[int, int, int]] = set()
    latencies: list[int] = []
    contact_latencies: list[int] = []
    stable_latencies: list[int] = []
    for key, evidence_step in accepted.items():
        dataset, episode_value, onset_step = key
        latencies.append(evidence_step - onset_step)
        for candidate_step in range(onset_step, evidence_step + 1):
            candidate_key = (dataset, episode_value, candidate_step)
            candidate = rows[candidate_key]
            if candidate["recorded_close"]:
                latched_keys.add(candidate_key)
        for candidate_step in range(onset_step, onset_step + maximum_lead_steps + 1):
            candidate = rows.get((dataset, episode_value, candidate_step))
            if candidate is None or not candidate["safe"] or not candidate["recorded_close"]:
                break
            if candidate["bilateral"]:
                contact_latencies.append(candidate_step - onset_step)
                break
        for candidate_step in range(onset_step, onset_step + maximum_lead_steps + 1):
            candidate = rows.get((dataset, episode_value, candidate_step))
            if candidate is None or not candidate["safe"] or not candidate["recorded_close"]:
                break
            if candidate["stable"]:
                stable_latencies.append(candidate_step - onset_step)
                break

    onset_mask = torch.zeros(row_shape, dtype=torch.bool, device=device)
    close_mask = torch.zeros_like(onset_mask)
    for window in range(windows):
        for offset in range(steps):
            if not padding[window, offset]:
                continue
            key = (
                int(source[window]),
                int(episode[window, offset]),
                int(step_index[window, offset]),
            )
            onset_mask[window, offset, 0] = key in accepted_onset_keys
            close_mask[window, offset, 0] = key in latched_keys
    def latency_metrics(prefix: str, values: list[int]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_count": float(array.size),
            f"{prefix}_mean": float(array.mean() if array.size else 0.0),
            f"{prefix}_median": float(np.median(array) if array.size else 0.0),
            f"{prefix}_p05": float(np.percentile(array, 5) if array.size else 0.0),
            f"{prefix}_p50": float(np.percentile(array, 50) if array.size else 0.0),
            f"{prefix}_p95": float(np.percentile(array, 95) if array.size else 0.0),
        }

    metrics = {
        "future_safe_close/raw_unique_onsets": float(len(onset_keys)),
        "future_safe_close/accepted_unique_onsets": float(len(accepted)),
        "future_safe_close/rejected_no_contact": float(rejected_no_contact),
        "future_safe_close/rejected_unsafe": float(rejected_unsafe),
        "future_safe_close/rejected_reopened_before_contact": float(
            rejected_reopened_before_contact
        ),
        "future_safe_close/acceptance_rate": float(
            len(accepted) / max(len(onset_keys), 1)
        ),
        "future_safe_close/mean_lead_steps": float(
            np.mean(latencies) if latencies else 0.0
        ),
        "future_safe_close/maximum_lead_steps_observed": float(
            max(latencies) if latencies else 0.0
        ),
        "future_safe_close/configured_maximum_lead_steps": float(maximum_lead_steps),
        "grasp/close_to_contact_success_rate": float(
            len(contact_latencies) / max(len(onset_keys), 1)
        ),
        "grasp/close_to_stable_success_rate": float(
            len(stable_latencies) / max(len(onset_keys), 1)
        ),
        **latency_metrics("grasp/close_to_contact_steps", contact_latencies),
        **latency_metrics("grasp/close_to_stable_steps", stable_latencies),
    }
    return close_mask, onset_mask, metrics


def expand_safe_close_onset_support(
    *,
    corrected_close: torch.Tensor,
    previous_gripper_open: torch.Tensor,
    grasp_ready: torch.Tensor,
    preceding_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a one-row close event into a short, physically safe onset band.

    Human teleoperation records only the first discrete close command.  With a
    recurrent window this makes the causal decision one row among many
    maintain-open/maintain-closed rows.  The deployment policy then learns a
    good approach but can remain infinitesimally on the open side of its 0.5
    threshold forever.  We add only *preceding* rows that already satisfy the
    audited grasp-ready geometry and whose recorded previous command is open.
    No contact, success, or simulator pose enters the actor; these quantities
    are used only to construct the offline BC label.
    """

    if corrected_close.shape != previous_gripper_open.shape or (
        corrected_close.shape != grasp_ready.shape
    ):
        raise ValueError("close-onset support tensors must have identical shapes")
    if corrected_close.ndim != 3 or corrected_close.shape[-1] != 1:
        raise ValueError("close-onset support expects [B,T,1] tensors")
    if preceding_steps < 0:
        raise ValueError("close-onset support steps cannot be negative")
    raw_onset = corrected_close.to(torch.bool) & previous_gripper_open.to(torch.bool)
    support = torch.zeros_like(raw_onset)
    for offset in range(1, int(preceding_steps) + 1):
        if offset >= corrected_close.shape[1]:
            break
        support[:, :-offset] |= raw_onset[:, offset:]
    support &= previous_gripper_open.to(torch.bool)
    support &= grasp_ready.to(torch.bool)
    support &= ~corrected_close.to(torch.bool)
    return corrected_close.to(torch.bool) | support, support


def onset_open_row_quota_weights(
    *,
    base_weight: torch.Tensor,
    close_onset: torch.Tensor,
    safe_open: torch.Tensor,
    close_onset_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build an OPEN-vs-CLOSE_ONSET loss authority with no HOLD_CLOSED rows.

    The gripper head predicts the causal open-to-close edge.  Rows whose
    previous command is already closed are state-maintenance observations, not
    another close decision, and therefore receive exactly zero classifier
    weight.  When a row quota is configured, all valid rows remain in the GRU
    sequence while the aggregate OPEN/ONSET classifier mass is normalized to
    the requested fractions.  This avoids randomly deleting temporal context.
    """

    if base_weight.shape != close_onset.shape or base_weight.shape != safe_open.shape:
        raise ValueError("gripper row-quota tensors must have identical shapes")
    if not 0.0 <= close_onset_fraction < 1.0:
        raise ValueError("close-onset row fraction must be in [0,1)")
    if bool((close_onset & safe_open).any()):
        raise ValueError("OPEN and CLOSE_ONSET rows must be disjoint")

    onset_weight = base_weight * close_onset.to(base_weight.dtype)
    open_weight = base_weight * safe_open.to(base_weight.dtype)
    raw_onset_sum = onset_weight.sum()
    raw_open_sum = open_weight.sum()
    onset_scale = base_weight.new_ones(())
    open_scale = base_weight.new_ones(())
    if close_onset_fraction > 0.0 and bool(
        (raw_onset_sum > 0.0) & (raw_open_sum > 0.0)
    ):
        # Normalize both classes to a common unit mass before assigning the
        # requested row-level quota.  The later onset loss multiplier remains
        # a separate, auditable authority.
        onset_scale = close_onset_fraction / raw_onset_sum
        open_scale = (1.0 - close_onset_fraction) / raw_open_sum
    classifier_weight = onset_weight * onset_scale + open_weight * open_scale
    return classifier_weight, {
        "raw_onset_weight": raw_onset_sum,
        "raw_open_weight": raw_open_sum,
        "onset_scale": onset_scale,
        "open_scale": open_scale,
        "hold_closed_rows": (
            (base_weight > 0.0) & (~close_onset) & (~safe_open)
        ).to(torch.float32).sum(),
    }


def successful_expert_first_close_rows(
    *,
    expert_action: torch.Tensor,
    previous_gripper_open: torch.Tensor,
    episode_outcome_success: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Select causal close initiations backed by a later physical success.

    Contact on this hand occurs roughly 45--47 policy rows after the operator
    starts closing, so requiring contact/stable on the *same recurrent window*
    removes every valid onset.  This selector uses only the recorded expert
    command transition plus the episode's eventual physical success label.  It
    does not use cube GT or manufacture the BC target: the later BC forward
    pass still accepts/rejects closing from Head-predicted cube XYZ minus the
    recorded distal-pad midpoint FK.
    """

    if expert_action.ndim != 3 or expert_action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("expert close selector expects actions [B,T,7]")
    expected = (*expert_action.shape[:2], 1)
    if previous_gripper_open.shape != expected:
        raise ValueError("previous gripper-open mask must be [B,T,1]")
    # Canonical sequence loading stores eventual physical success once per
    # recurrent window as ``[B]``.  Some online callers retain the repeated
    # transition form ``[B,T]``/``[B,T,1]``.  Accept both representations but
    # expand only the episode/window label; never infer success from a close
    # transition or from privileged cube geometry.
    if episode_outcome_success.shape == (expert_action.shape[0],):
        episode_outcome_success = episode_outcome_success[:, None, None].expand(
            expected
        )
    elif episode_outcome_success.shape == (expert_action.shape[0], 1):
        episode_outcome_success = episode_outcome_success[:, :, None].expand(
            expected
        )
    elif episode_outcome_success.shape == expert_action.shape[:2]:
        episode_outcome_success = episode_outcome_success.unsqueeze(-1)
    if episode_outcome_success.shape != expected:
        raise ValueError(
            "episode outcome success must be [B], [B,1], [B,T] or [B,T,1]"
        )
    if padding_mask.shape != expert_action.shape[:2]:
        raise ValueError("close selector padding mask must be [B,T]")
    commanded_close = (
        expert_action[..., G2_VISUAL_GRIPPER_ACTION_INDEX : G2_VISUAL_GRIPPER_ACTION_INDEX + 1]
        < 0.0
    )
    return (
        commanded_close
        & previous_gripper_open.to(torch.bool)
        & episode_outcome_success.to(torch.bool)
        & padding_mask.unsqueeze(-1).to(torch.bool)
    )


def policy_origin_assisted_close_sequence_keys(
    batch: Mapping[str, np.ndarray],
    *,
    policy_origin_trigger_episode_lineage: set[tuple[int, int]],
) -> set[tuple[int, int, int]]:
    """Return close-sequence endpoints attributable to a policy-origin handoff.

    This is deliberately stricter than the generic corrective sampler.  A
    qualifying sequence must end with the calibrated close correction under
    ``ASSISTED`` behavior authority, stay within one env/episode with
    contiguous sequence steps, and belong to an episode in which a
    policy-origin handoff actually triggered.  Reference-transfer episodes
    are therefore useful training data but cannot attest learned-policy
    competence or satisfy a promotion gate.
    """

    required = {
        "expert_action_target",
        "env_id",
        "episode_id",
        "sequence_step",
        "behavior_authority_code",
    }
    missing = required.difference(batch)
    if missing:
        raise ValueError(
            "corrective sequence provenance fields are missing: "
            + ", ".join(sorted(missing))
        )
    action = np.asarray(batch["expert_action_target"])
    env_id = np.asarray(batch["env_id"])
    episode_id = np.asarray(batch["episode_id"])
    sequence_step = np.asarray(batch["sequence_step"])
    authority = np.asarray(batch["behavior_authority_code"])
    if action.ndim != 3 or action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("corrective expert actions must be [B,T,7]")
    expected = action.shape[:2]
    if any(
        value.shape != expected
        for value in (env_id, episode_id, sequence_step, authority)
    ):
        raise ValueError("corrective provenance must share [B,T] shape")

    assisted_code = G2_REPLAY_BEHAVIOR_AUTHORITY["ASSISTED"]
    keys: set[tuple[int, int, int]] = set()
    for row in range(action.shape[0]):
        if not np.all(env_id[row] == env_id[row, 0]):
            continue
        if not np.all(episode_id[row] == episode_id[row, 0]):
            continue
        if not np.all(np.diff(sequence_step[row]) == 1):
            continue
        lineage = (int(env_id[row, -1]), int(episode_id[row, -1]))
        if lineage not in policy_origin_trigger_episode_lineage:
            continue
        if float(action[row, -1, G2_VISUAL_GRIPPER_ACTION_INDEX]) >= 0.0:
            continue
        assisted_rows = authority[row] == assisted_code
        if not bool(assisted_rows[-1]):
            continue
        first_assisted = int(np.flatnonzero(assisted_rows)[0])
        if not bool(assisted_rows[first_assisted:].all()):
            continue
        keys.add((lineage[0], lineage[1], int(sequence_step[row, -1])))
    return keys


def g2_tensor_state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes and bytes for load attestation."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def g2_camera_encoder_profile_name(channels: tuple[int, ...]) -> str:
    normalized = tuple(int(value) for value in channels)
    for name, registered in G2_CAMERA_ENCODER_PROFILES.items():
        if normalized == registered:
            return name
    raise ValueError("unknown camera encoder channel contract")
G2_STUDENT_FAILURE_CLASSES = (
    "miss", "contact", "stable_grasp", "slip", "collision", "lift", "place"
)
G2_DEMONSTRATION_PHASES = (
    "REACH", "PRE_GRASP", "CONTACT", "STABLE_GRASP", "LIFT"
)
G2_HEAD_DISTANCE_DEMONSTRATION_PHASES = (
    "PRE_GRASP", "TRANSITION", "CONTACT", "LIFT"
)
G2_ONLINE_REPLAY_PHASES = ("REACH", "CONTACT", "STABLE_GRASP", "LIFT")


def preprocess_demonstration_expert_actions(
    expert_action: torch.Tensor, *, arm_action_scale: float
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Create a BC-only speed-retimed view of canonical expert actions.

    Only the six continuous arm labels are scaled.  The recorded gripper
    command (binary in current live collections, potentially continuous in
    canonical legacy/test fixtures) and every state/reward/next-state remain
    bit-identical, so this function must not be used to rewrite an RL
    transition action.
    """

    if expert_action.ndim < 2 or expert_action.shape[-1] != G2_VISUAL_ACTION_DIM:
        raise ValueError("expert action must end in the canonical 7-D action")
    if not 0.0 < arm_action_scale <= 1.0:
        raise ValueError("expert arm action scale must be in (0,1]")
    if not bool(torch.isfinite(expert_action).all()):
        raise ValueError("expert action contains non-finite values")
    gripper = expert_action[..., G2_VISUAL_GRIPPER_ACTION_INDEX]
    processed = expert_action.clone()
    before = processed[..., :G2_VISUAL_ARM_ACTION_DIM].abs()
    processed[..., :G2_VISUAL_ARM_ACTION_DIM] *= float(arm_action_scale)
    after = processed[..., :G2_VISUAL_ARM_ACTION_DIM].abs()
    if not torch.equal(processed[..., G2_VISUAL_GRIPPER_ACTION_INDEX], gripper):
        raise RuntimeError("expert gripper changed during arm speed preprocessing")
    return processed, {
        "schema": "g2_demonstration_bc_arm_speed_retime_v1",
        "arm_action_scale": float(arm_action_scale),
        "arm_absolute_max_before": float(before.max()) if before.numel() else 0.0,
        "arm_absolute_max_after": float(after.max()) if after.numel() else 0.0,
        "gripper_contract": "RECORDED_VALUE_EXACTLY_UNCHANGED",
        "usage": "BEHAVIOR_CLONING_LABEL_ONLY_NOT_RL_TRANSITION_REWRITE",
    }


def classify_demonstration_sequence_phases(
    sequences: Mapping[str, torch.Tensor | int],
) -> torch.Tensor:
    """Return one highest-achieved phase code for every expert window.

    The labels are derived from the canonical Teacher state and physical
    contact/stable targets.  No simulator-only value is added to the actor
    input: this code is used solely by the offline batch sampler.
    """

    required = (
        "teacher_state_target",
        "expert_action_target",
        "bilateral_contact_target",
        "stable_grasp_target",
        "padding_mask",
    )
    if any(not isinstance(sequences.get(name), torch.Tensor) for name in required):
        raise ValueError("demonstration sequences lack canonical phase tensors")
    state = sequences["teacher_state_target"]
    action = sequences["expert_action_target"]
    bilateral = sequences["bilateral_contact_target"]
    stable = sequences["stable_grasp_target"]
    padding = sequences["padding_mask"].to(torch.bool)
    assert isinstance(state, torch.Tensor)
    assert isinstance(action, torch.Tensor)
    assert isinstance(bilateral, torch.Tensor)
    assert isinstance(stable, torch.Tensor)
    assert isinstance(padding, torch.Tensor)
    if state.ndim != 3 or state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("teacher_state_target must be [B,T,59]")
    if action.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
        raise ValueError("expert_action_target must be [B,T,7]")
    burn_in = int(sequences.get("burn_in_steps", 0))
    learning = padding.clone()
    learning[:, :burn_in] = False
    phase_slice = G2TeacherObservationContract().slices[
        "curriculum_phase_features"
    ]
    phase = state[..., phase_slice]
    valid = learning.unsqueeze(-1)
    contact = ((phase[..., 1:2] > 0.5) | (bilateral > 0.5)) & valid
    stable_mask = (stable > 0.5) & valid
    lift = ((phase[..., 2:3] > 0.5) | (phase[..., 3:4] > 0.5)) & valid
    pregrasp = (action[..., 6:7] < 0.0) & ~contact & valid
    code = torch.zeros(padding.shape[0], dtype=torch.long, device=state.device)
    code[pregrasp.any(dim=(1, 2))] = 1
    code[contact.any(dim=(1, 2))] = 2
    code[stable_mask.any(dim=(1, 2))] = 3
    code[lift.any(dim=(1, 2))] = 4
    return code


def classify_head_visible_distance_sequence_phases(
    sequences: Mapping[str, torch.Tensor | int],
    *,
    transition_distance_m: float,
    minimum_cube_pixels: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stratify demonstrations by head-visible gripper--cube distance.

    The compact head RGB-D frame provides the visibility authority.  The
    metric distance is taken from the canonical EE-relative cube-pose label,
    rather than from the policy's current pose prediction, so sampling cannot
    create a self-reinforcing feedback loop while that prediction is still
    being learned.  This label is sampler-only metadata and is never appended
    to the deployable actor observation.  Real contact/lift state overrides a
    visual distance bin, including after the object becomes occluded by the
    hand.

    Windows with neither a head-visible cube nor physical contact/lift receive
    code ``-1`` and are excluded instead of being assigned a fictitious range.
    """

    required = (
        "rgbd_u8",
        "relative_pose_target",
        "teacher_state_target",
        "bilateral_contact_target",
        "stable_grasp_target",
        "padding_mask",
    )
    if any(not isinstance(sequences.get(name), torch.Tensor) for name in required):
        raise ValueError("demonstration sequences lack head-distance phase tensors")
    if transition_distance_m <= 0.0:
        raise ValueError("transition distance must be positive")
    if minimum_cube_pixels <= 0:
        raise ValueError("minimum cube pixels must be positive")
    rgbd = sequences["rgbd_u8"]
    relative = sequences["relative_pose_target"]
    state = sequences["teacher_state_target"]
    bilateral = sequences["bilateral_contact_target"]
    stable = sequences["stable_grasp_target"]
    padding = sequences["padding_mask"].to(torch.bool)
    assert isinstance(rgbd, torch.Tensor)
    assert isinstance(relative, torch.Tensor)
    assert isinstance(state, torch.Tensor)
    assert isinstance(bilateral, torch.Tensor)
    assert isinstance(stable, torch.Tensor)
    assert isinstance(padding, torch.Tensor)
    if rgbd.ndim != 6 or tuple(rgbd.shape[2:4]) != (2, 6):
        raise ValueError("rgbd_u8 must be [B,T,2,6,H,W]")
    if relative.shape != (*padding.shape, 7):
        raise ValueError("relative_pose_target must be [B,T,7]")
    burn_in = int(sequences.get("burn_in_steps", 0))
    learning = padding.clone()
    learning[:, :burn_in] = False

    head = rgbd[:, :, 0]
    red = head[:, :, 0].to(torch.int16)
    green = head[:, :, 1].to(torch.int16)
    blue = head[:, :, 2].to(torch.int16)
    depth_valid = head[:, :, 5] > 127
    # Same red-dominance contract as the deployable head-centroid path, in
    # exact uint8 units (0.30*255 and 0.10*255).
    cube_mask = (
        depth_valid
        & (red > 77)
        & (red - torch.maximum(green, blue) > 26)
    )
    visible = cube_mask.flatten(-2).sum(dim=-1) >= int(minimum_cube_pixels)
    visual_valid = learning & visible
    distance = torch.linalg.vector_norm(relative[..., :3].to(torch.float32), dim=-1)
    finite_distance = torch.isfinite(distance)
    visual_valid &= finite_distance
    masked_distance = torch.where(
        visual_valid, distance, torch.full_like(distance, float("inf"))
    )
    minimum_distance = masked_distance.min(dim=1).values

    phase_slice = G2TeacherObservationContract().slices[
        "curriculum_phase_features"
    ]
    phase = state[..., phase_slice]
    valid = learning.unsqueeze(-1)
    physical_contact = (
        (bilateral > 0.5) | (stable > 0.5) | (phase[..., 1:2] > 0.5)
    ) & valid
    physical_lift = (
        (phase[..., 2:3] > 0.5) | (phase[..., 3:4] > 0.5)
    ) & valid

    code = torch.full(
        (padding.shape[0],), -1, dtype=torch.long, device=padding.device
    )
    head_visible_window = visual_valid.any(dim=1)
    code[head_visible_window] = 0
    code[head_visible_window & (minimum_distance <= transition_distance_m)] = 1
    code[physical_contact.any(dim=(1, 2))] = 2
    code[physical_lift.any(dim=(1, 2))] = 3
    eligible = code >= 0
    return code, {
        "head_distance/window_count": float(code.numel()),
        "head_distance/eligible_window_count": float(eligible.sum()),
        "head_distance/unobserved_window_count": float((~eligible).sum()),
        "head_distance/head_visible_window_fraction": float(
            head_visible_window.to(torch.float32).mean()
        ),
        "head_distance/transition_distance_m": float(transition_distance_m),
        "head_distance/minimum_cube_pixels": float(minimum_cube_pixels),
    }


def sample_phase_stratified_demonstration_indices(
    phase_codes: torch.Tensor,
    success: torch.Tensor,
    *,
    batch_size: int,
    phase_probabilities: tuple[float, ...],
    success_priority_weight: float,
    rng: np.random.Generator,
    force_priorities: torch.Tensor | None = None,
    force_priority_mixture: float = 0.25,
    outcome_success_fraction: float | None = None,
    phase_names: tuple[str, ...] = G2_DEMONSTRATION_PHASES,
    preserve_phase_marginal: bool = False,
    close_onset_windows: torch.Tensor | None = None,
    close_onset_fraction: float = 0.0,
    source_dataset_indices: torch.Tensor | None = None,
    balance_source_datasets: bool = False,
    near_contact_windows: torch.Tensor | None = None,
    near_contact_fraction: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample expert windows by phase, outcome, source and rare task strata.

    Force priority is mixed with the phase-local base distribution instead of
    multiplying it without a bound.  The latter made the 12% of windows with
    force evidence occupy roughly half of every expert batch in a live run,
    repeatedly training on a very small set of trajectories and degrading the
    reference/contact curriculum.  A mixture keeps real force evidence active
    while preserving phase-local demonstration diversity.
    """

    if phase_codes.ndim != 1 or success.shape != phase_codes.shape:
        raise ValueError("phase_codes and success must both be [windows]")
    if batch_size <= 0 or phase_codes.numel() == 0:
        raise ValueError("expert sampling requires a positive non-empty batch")
    probability = np.asarray(phase_probabilities, dtype=np.float64)
    if not phase_names or len(set(phase_names)) != len(phase_names):
        raise ValueError("demonstration phase names must be unique and non-empty")
    if probability.shape != (len(phase_names),) or np.any(probability < 0):
        raise ValueError("invalid demonstration phase probabilities")
    if not np.isclose(probability.sum(), 1.0):
        raise ValueError("demonstration phase probabilities must sum to one")
    if success_priority_weight < 1.0:
        raise ValueError("success priority weight must be at least one")
    if outcome_success_fraction is not None and not 0.0 < outcome_success_fraction < 1.0:
        raise ValueError("outcome success fraction must be in (0,1)")
    if not 0.0 <= force_priority_mixture <= 1.0:
        raise ValueError("force_priority_mixture must be in [0,1]")
    if not 0.0 <= close_onset_fraction < 1.0:
        raise ValueError("close_onset_fraction must be in [0,1)")
    if not 0.0 <= near_contact_fraction < 1.0:
        raise ValueError("near_contact_fraction must be in [0,1)")
    if close_onset_windows is None:
        if close_onset_fraction != 0.0:
            raise ValueError(
                "close_onset_fraction requires close_onset_windows"
            )
        onset_numpy = np.zeros(phase_codes.numel(), dtype=np.bool_)
    else:
        if close_onset_windows.shape != phase_codes.shape:
            raise ValueError("close_onset_windows must be [windows]")
        onset_numpy = close_onset_windows.detach().cpu().numpy().astype(bool)
        if close_onset_fraction > 0.0 and not bool(onset_numpy.any()):
            raise ValueError("close-onset sampling requires an onset window")
        if close_onset_fraction > 0.0 and not bool((~onset_numpy).any()):
            raise ValueError("close-onset sampling requires a non-onset window")
    if near_contact_windows is None:
        if near_contact_fraction != 0.0:
            raise ValueError(
                "near_contact_fraction requires near_contact_windows"
            )
        near_numpy = np.zeros(phase_codes.numel(), dtype=np.bool_)
    else:
        if near_contact_windows.shape != phase_codes.shape:
            raise ValueError("near_contact_windows must be [windows]")
        near_numpy = near_contact_windows.detach().cpu().numpy().astype(bool)
        if near_contact_fraction > 0.0 and not bool(near_numpy.any()):
            raise ValueError("near-contact sampling requires a near window")
        if near_contact_fraction > 0.0 and not bool((~near_numpy).any()):
            raise ValueError("near-contact sampling requires a non-near window")
    if source_dataset_indices is None:
        if balance_source_datasets:
            raise ValueError(
                "balance_source_datasets requires source_dataset_indices"
            )
        source_numpy = np.zeros(phase_codes.numel(), dtype=np.int64)
    else:
        if source_dataset_indices.shape != phase_codes.shape:
            raise ValueError("source_dataset_indices must be [windows]")
        source_numpy = (
            source_dataset_indices.detach().cpu().numpy().astype(np.int64)
        )
    if force_priorities is not None:
        if force_priorities.shape != phase_codes.shape:
            raise ValueError("force_priorities must be [windows]")
        if not bool(torch.isfinite(force_priorities).all()) or bool(
            (force_priorities < 1.0).any()
        ):
            raise ValueError("force_priorities must be finite and at least one")
    phase_numpy = phase_codes.detach().cpu().numpy()
    success_numpy = success.detach().cpu().numpy().astype(bool)
    force_numpy = (
        np.ones_like(phase_numpy, dtype=np.float64)
        if force_priorities is None
        else force_priorities.detach().cpu().numpy().astype(np.float64)
    )
    all_indices = np.arange(len(phase_numpy), dtype=np.int64)
    if outcome_success_fraction is not None:
        if not bool(success_numpy.any()) or not bool((~success_numpy).any()):
            raise ValueError("explicit outcome sampling requires both outcomes")
        requested_outcomes = rng.random(batch_size) < outcome_success_fraction
    else:
        requested_outcomes = np.full(batch_size, False, dtype=np.bool_)
    # Build an exact-size, shuffled quota rather than independent Bernoulli
    # draws.  Close initiation is a rare decision (unlike maintaining an
    # already closed hand), so a noisy per-batch count can erase it from many
    # consecutive optimizer steps even when its expected fraction is nonzero.
    requested_onset = np.zeros(batch_size, dtype=np.bool_)
    onset_quota = int(round(batch_size * close_onset_fraction))
    if onset_quota:
        requested_onset[:onset_quota] = True
        rng.shuffle(requested_onset)
    requested_near = np.zeros(batch_size, dtype=np.bool_)
    near_quota = int(round(batch_size * near_contact_fraction))
    if near_quota:
        requested_near[:near_quota] = True
        rng.shuffle(requested_near)
    selected: list[int] = []
    counts = {name: 0 for name in phase_names}
    source_counts = {
        int(source): 0 for source in np.unique(source_numpy[phase_numpy >= 0])
    }
    for sample_index in range(batch_size):
        onset_mask = (
            onset_numpy
            if requested_onset[sample_index]
            else (
                ~onset_numpy
                if close_onset_fraction > 0.0
                else np.ones_like(onset_numpy, dtype=np.bool_)
            )
        )
        near_mask = (
            near_numpy
            if requested_near[sample_index]
            else (
                ~near_numpy
                if near_contact_fraction > 0.0
                else np.ones_like(near_numpy, dtype=np.bool_)
            )
        )
        stratum_mask = onset_mask & near_mask
        if preserve_phase_marginal:
            available = np.asarray(
                [
                    bool(np.any((phase_numpy == phase) & stratum_mask))
                    for phase in range(len(probability))
                ],
                dtype=np.bool_,
            )
            effective_probability = probability * available
            if effective_probability.sum() <= 0.0:
                raise ValueError("demonstration sampler has no available phase")
            effective_probability /= effective_probability.sum()
            phase_code = int(
                rng.choice(len(effective_probability), p=effective_probability)
            )
            phase_mask = (phase_numpy == phase_code) & stratum_mask
            requested_outcome_mask = success_numpy == requested_outcomes[sample_index]
            outcome_mask = (
                requested_outcome_mask
                if outcome_success_fraction is not None
                and bool(np.any(phase_mask & requested_outcome_mask))
                else np.ones_like(success_numpy, dtype=np.bool_)
            )
        else:
            requested_outcome_mask = (
                success_numpy == requested_outcomes[sample_index]
            )
            outcome_mask = (
                requested_outcome_mask
                if outcome_success_fraction is not None
                and bool(np.any(requested_outcome_mask & stratum_mask))
                else np.ones_like(success_numpy, dtype=np.bool_)
            )
            available = np.asarray(
                [
                    bool(
                        np.any(
                            (phase_numpy == phase)
                            & outcome_mask
                            & stratum_mask
                        )
                    )
                    for phase in range(len(probability))
                ],
                dtype=np.bool_,
            )
            effective_probability = probability * available
            if effective_probability.sum() <= 0.0:
                raise ValueError("demonstration outcome has no available phase")
            effective_probability /= effective_probability.sum()
            phase_code = int(
                rng.choice(len(effective_probability), p=effective_probability)
            )
        candidates = all_indices[
            (phase_numpy == phase_code) & outcome_mask & stratum_mask
        ]
        if candidates.size == 0:
            raise RuntimeError("sampler selected an unavailable demonstration phase")
        if balance_source_datasets:
            available_sources = np.unique(source_numpy[candidates])
            minimum_count = min(source_counts[int(source)] for source in available_sources)
            least_sampled_sources = np.asarray(
                [
                    source
                    for source in available_sources
                    if source_counts[int(source)] == minimum_count
                ],
                dtype=np.int64,
            )
            chosen_source = int(rng.choice(least_sampled_sources))
            candidates = candidates[source_numpy[candidates] == chosen_source]
        base_weights = (
            np.ones(candidates.size, dtype=np.float64)
            if outcome_success_fraction is not None
            else np.where(success_numpy[candidates], success_priority_weight, 1.0).astype(np.float64)
        )
        base_probability = base_weights / base_weights.sum()
        force_weights = base_weights * force_numpy[candidates]
        force_probability = force_weights / force_weights.sum()
        weights = (
            (1.0 - force_priority_mixture) * base_probability
            + force_priority_mixture * force_probability
        )
        chosen = int(rng.choice(candidates, p=weights))
        selected.append(chosen)
        source_counts[int(source_numpy[chosen])] += 1
        counts[phase_names[int(phase_numpy[chosen])]] += 1
    return np.asarray(selected, dtype=np.int64), counts


def demonstration_policy_mastery(
    evaluation_rates: tuple[float, float, float],
    *,
    evaluation_episodes: int,
    minimum_evaluation_episodes: int = 10,
    thresholds: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> float:
    """Measure Q-filter trust earned by held-out policy task outcomes."""

    if evaluation_episodes < minimum_evaluation_episodes:
        return 0.0
    rates = np.asarray(evaluation_rates, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if rates.shape != (3,) or limits.shape != (3,) or np.any(limits <= 0.0):
        raise ValueError("demonstration policy mastery expects three positive gates")
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("evaluation rates must be finite and non-negative")
    return float(np.clip(np.min(rates / limits), 0.0, 1.0))


def demonstration_bc_schedule(
    *,
    actor_updates: int,
    decay_updates: int,
    initial_coefficient: float,
    minimum_coefficient: float,
    policy_mastery: float,
) -> tuple[float, float]:
    """Return performance-gated BC coefficient and Q-filter reliability."""

    if actor_updates < 0 or decay_updates <= 0:
        raise ValueError(
            "demonstration BC actor updates must be non-negative and decay positive"
        )
    if not 0.0 <= minimum_coefficient <= initial_coefficient:
        raise ValueError("invalid demonstration BC coefficient range")
    if not 0.0 <= policy_mastery <= 1.0:
        raise ValueError("policy mastery must be in [0,1]")
    # Reference-owned transitions may advance while the actor receives no
    # optimizer step.  Decay on the only meaningful learning clock.
    time_progress = np.clip(int(actor_updates) / float(decay_updates), 0.0, 1.0)
    trusted_progress = float(time_progress) * float(policy_mastery)
    coefficient = initial_coefficient + (
        minimum_coefficient - initial_coefficient
    ) * trusted_progress
    return float(coefficient), float(policy_mastery)


def demonstration_force_energy_priorities(
    sequences: Mapping[str, torch.Tensor | int],
    *,
    temperature_j: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build bounded window priorities from recorded physical contact work.

    The authority is the original two-finger force and consecutive cube pose
    displacement.  Contact labels, safe-for-force attestation and recurrent
    padding all have to agree.  This does not synthesize contact and does not
    turn demonstrations into HER transitions; it only prioritizes real force
    evidence in the actor's expert batch.
    """

    if temperature_j <= 0.0:
        raise ValueError("demonstration force temperature must be positive")
    required = (
        "teacher_state_target",
        "contact_force_target_n",
        "bilateral_contact_target",
        "safe_for_her_force",
        "stable_grasp_target",
        "padding_mask",
    )
    if any(not isinstance(sequences.get(name), torch.Tensor) for name in required):
        raise ValueError("demonstration sequences lack force-priority tensors")
    state = sequences["teacher_state_target"]
    force = sequences["contact_force_target_n"]
    bilateral = sequences["bilateral_contact_target"]
    safe = sequences["safe_for_her_force"]
    stable = sequences["stable_grasp_target"]
    padding = sequences["padding_mask"]
    assert isinstance(state, torch.Tensor)
    assert isinstance(force, torch.Tensor)
    assert isinstance(bilateral, torch.Tensor)
    assert isinstance(safe, torch.Tensor)
    assert isinstance(stable, torch.Tensor)
    assert isinstance(padding, torch.Tensor)
    if force.shape != (*padding.shape, 2):
        raise ValueError("contact_force_target_n must be [B,T,2]")
    cube_slice = G2TeacherObservationContract().slices["cube_pose_root_xyzw"]
    cube_position = state[..., cube_slice][..., :3]
    displacement = torch.zeros_like(padding, dtype=torch.float32)
    displacement[:, 1:] = torch.linalg.vector_norm(
        cube_position[:, 1:] - cube_position[:, :-1], dim=-1
    )
    burn_in = int(sequences.get("burn_in_steps", 0))
    # ``Tensor.to`` may return the original object when it is already bool.
    # Never mutate the canonical recurrent padding mask while excluding the
    # burn-in prefix for force-priority estimation.
    valid = padding.to(torch.bool).clone()
    valid[:, :burn_in] = False
    valid &= safe.to(torch.bool).squeeze(-1)
    valid &= bilateral.to(torch.bool).squeeze(-1)
    total_force_n = force.to(torch.float32).clamp_min(0.0).sum(dim=-1)
    energy_j = torch.where(valid, total_force_n * displacement, torch.zeros_like(displacement))
    row_priority = 1.0 + torch.clamp(energy_j / temperature_j, 0.0, 4.0)
    stable_valid = stable.to(torch.bool).squeeze(-1) & valid
    row_priority = torch.where(
        stable_valid,
        torch.maximum(row_priority, torch.full_like(row_priority, 3.0)),
        row_priority,
    )
    window_priority = row_priority.max(dim=1).values
    return window_priority, {
        "demonstration_force/window_priority_mean": float(window_priority.mean()),
        "demonstration_force/window_priority_max": float(window_priority.max()),
        "demonstration_force/prioritized_window_fraction": float(
            (window_priority > 1.0).to(torch.float32).mean()
        ),
        "demonstration_force/contact_energy_mean_j": float(
            energy_j[valid].mean() if bool(valid.any()) else 0.0
        ),
        "demonstration_force/contact_energy_max_j": float(energy_j.max()),
        "demonstration_force/temperature_j": float(temperature_j),
    }


def g2_reference_controller_action(
    teacher_state: torch.Tensor,
    *,
    end_effector_to_cube_slice: slice,
    cube_to_goal_slice: slice,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    maximum_arm_action_magnitude: float,
    translation_scale_m: float = 0.0225,
    target_cube_minus_ee_m: tuple[float, float, float] | torch.Tensor = (0.0, 0.0, -0.020),
    gripper_close_position_tolerance_m: float = 0.010,
) -> torch.Tensor:
    """Build the bounded privileged reference used only for data collection.

    Before contact the hand approaches a calibrated side-pinch relative pose.
    ``target_cube_minus_ee_m`` is expressed in the robot-root frame and may be
    estimated only from physically successful, collision-free demonstrations.
    A stable grasp changes authority to the episode's object goal in all three
    translation axes, so independently randomized cube/goal XY positions do
    not make the canonical 5 mm success condition unreachable.
    """

    if teacher_state.ndim != 2 or teacher_state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("reference controller requires [N,59] teacher state")
    batch = teacher_state.shape[0]
    for name, value in (
        ("bilateral_contact", bilateral_contact),
        ("stable_grasp", stable_grasp),
    ):
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [N] tensor")
    if not 0.0 < maximum_arm_action_magnitude <= 1.0:
        raise ValueError("maximum arm action magnitude must be in (0,1]")
    if translation_scale_m <= 0.0:
        raise ValueError("translation scale must be positive")
    if gripper_close_position_tolerance_m <= 0.0:
        raise ValueError("reference gripper close tolerance must be positive")

    result = torch.zeros(
        (batch, G2_VISUAL_ACTION_DIM),
        dtype=teacher_state.dtype,
        device=teacher_state.device,
    )
    target_relative = torch.as_tensor(
        target_cube_minus_ee_m,
        dtype=teacher_state.dtype,
        device=teacher_state.device,
    )
    if target_relative.shape == (3,):
        target_relative = target_relative.unsqueeze(0).expand(batch, -1)
    if target_relative.shape != (batch, 3) or not bool(
        torch.isfinite(target_relative).all()
    ):
        raise ValueError(
            "reference grasp offset must be finite [3] or [batch,3]"
        )
    # Moving the EE by +delta changes (cube - EE) by -delta.  Therefore the
    # Cartesian command that drives the measured relative pose to the
    # calibrated target is current_relative - target_relative.
    approach = (
        teacher_state[:, end_effector_to_cube_slice] - target_relative
    )
    result[:, :3] = torch.clamp(
        approach / translation_scale_m,
        -maximum_arm_action_magnitude,
        maximum_arm_action_magnitude,
    )
    result[bilateral_contact, :G2_VISUAL_ARM_ACTION_DIM] = 0.0
    goal_delta = teacher_state[:, cube_to_goal_slice]
    result[stable_grasp, :3] = torch.clamp(
        goal_delta[stable_grasp] / translation_scale_m,
        -maximum_arm_action_magnitude,
        maximum_arm_action_magnitude,
    )
    # Approaching with a closed parallel gripper made the outer pad strike and
    # push the cube while the inner pad never contacted it.  Preserve the
    # audited open pose until the calibrated side-pinch center is reached;
    # only then close.  This is reference collection logic, not actor input.
    position_error = torch.linalg.vector_norm(approach, dim=-1)
    close_gripper = _metric_leq_with_dtype_roundoff(
        position_error, gripper_close_position_tolerance_m
    ) | bilateral_contact
    result[:, G2_VISUAL_GRIPPER_ACTION_INDEX] = torch.where(
        close_gripper,
        -torch.ones_like(position_error),
        torch.ones_like(position_error),
    )
    return result


@dataclass(frozen=True)
class G2PreContactTaskAlignmentSafetyProjectionResult:
    """Result of the component-wise PRE_CONTACT task-alignment projection."""

    action: torch.Tensor
    eligible_mask: torch.Tensor
    translation_eligible_mask: torch.Tensor
    rotation_eligible_mask: torch.Tensor
    wrong_direction_axis_mask: torch.Tensor
    overshoot_axis_mask: torch.Tensor
    translation_intervention_mask: torch.Tensor
    rotation_intervention_mask: torch.Tensor
    intervention_mask: torch.Tensor


def _quat_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def _quat_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _axis_angle_to_quaternion_xyzw(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half) / torch.clamp_min(angle, 1.0e-8),
        0.5 - angle.square() / 48.0,
    )
    quaternion = torch.cat((axis_angle * scale, torch.cos(half)), dim=-1)
    return quaternion / torch.clamp_min(
        torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), 1.0e-8
    )


def _quaternion_error_axis_angle_root_xyzw(
    current: torch.Tensor, desired: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    current = canonicalize_quaternion_xyzw(current)
    desired = canonicalize_quaternion_xyzw(desired)
    error_quaternion = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(desired, _quat_conjugate_xyzw(current))
    )
    xyz = error_quaternion[..., :3]
    xyz_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, error_quaternion[..., 3:4].clamp_min(0.0))
    axis = xyz / torch.clamp_min(xyz_norm, 1.0e-8)
    axis_angle = torch.where(xyz_norm > 1.0e-8, axis * angle, torch.zeros_like(xyz))
    return axis_angle, angle.squeeze(-1)


def apply_g2_precontact_task_alignment_safety_projection(
    applied_controller_action: torch.Tensor,
    *,
    pure_policy_mask: torch.Tensor,
    controller_authority_mask: torch.Tensor,
    contact_approach_mask: torch.Tensor,
    orientation_hold_mask: torch.Tensor | None = None,
    any_finger_contact: torch.Tensor,
    cube_minus_ee_root_m: torch.Tensor,
    target_cube_minus_ee_root_m: tuple[float, float, float],
    translation_scale_m: float,
    current_ee_orientation_root_xyzw: torch.Tensor | None = None,
    desired_ee_orientation_root_xyzw: torch.Tensor | None = None,
    rotation_scale_rad: float | None = None,
) -> G2PreContactTaskAlignmentSafetyProjectionResult:
    """Keep a pure-policy PRE_CONTACT request on the calibrated approach.

    The caller owns the existing definition of ``contact_approach_mask``; this
    helper intentionally introduces no distance threshold.  The first three
    action components are normalized root-frame EE translation deltas.  If
    ``r = cube - EE`` and one action produces ``delta_EE = scale * action``,
    then ``r_next = r - delta_EE``.  Each eligible translation component is
    therefore projected onto the closed interval from zero to the remaining
    component-wise error ``(r - target) / scale``.  This rejects motion away
    from the target and caps motion toward it before an overshoot can occur.

    Translation eligibility remains the caller-supplied contact-approach
    region.  Rotation may use a wider caller-supplied hold region so a
    near-reset cannot accumulate orientation drift before entering the final
    translation corridor.  When omitted, the rotation mask is exactly the
    translation mask for backwards compatibility.  The operation is
    non-amplifying, never changes the gripper, and leaves controller-owned and
    all other ineligible rows byte-identical.
    """

    if (
        applied_controller_action.ndim != 2
        or applied_controller_action.shape[-1] != G2_VISUAL_ACTION_DIM
    ):
        raise ValueError("applied controller action must be [N,7]")
    if not torch.is_floating_point(applied_controller_action):
        raise ValueError("applied controller action must be floating point")
    if not bool(torch.isfinite(applied_controller_action).all()):
        raise ValueError("applied controller action contains non-finite values")
    batch = applied_controller_action.shape[0]
    if orientation_hold_mask is None:
        orientation_hold_mask = contact_approach_mask
    masks = {
        "pure_policy_mask": pure_policy_mask,
        "controller_authority_mask": controller_authority_mask,
        "contact_approach_mask": contact_approach_mask,
        "orientation_hold_mask": orientation_hold_mask,
        "any_finger_contact": any_finger_contact,
    }
    for name, value in masks.items():
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [N] tensor")
        if value.device != applied_controller_action.device:
            raise ValueError(f"{name} must share the action device")
    if bool((pure_policy_mask & controller_authority_mask).any()):
        raise ValueError("pure-policy and controller authority masks overlap")
    if cube_minus_ee_root_m.shape != (batch, 3):
        raise ValueError("cube-minus-EE position must be [N,3]")
    if cube_minus_ee_root_m.device != applied_controller_action.device:
        raise ValueError("cube-minus-EE position must share the action device")
    if cube_minus_ee_root_m.dtype != applied_controller_action.dtype:
        raise ValueError("cube-minus-EE position must share the action dtype")
    if not bool(torch.isfinite(cube_minus_ee_root_m).all()):
        raise ValueError("cube-minus-EE position contains non-finite values")
    if (
        len(target_cube_minus_ee_root_m) != 3
        or not all(
            math.isfinite(float(value))
            for value in target_cube_minus_ee_root_m
        )
    ):
        raise ValueError(
            "calibrated cube-minus-EE target must contain three finite values"
        )
    if (
        not math.isfinite(float(translation_scale_m))
        or translation_scale_m <= 0.0
    ):
        raise ValueError("translation scale must be finite and positive")

    translation_eligible = (
        pure_policy_mask & contact_approach_mask & ~any_finger_contact
    )
    rotation_eligible = (
        pure_policy_mask & orientation_hold_mask & ~any_finger_contact
    )
    eligible = translation_eligible | rotation_eligible
    eligible_axes = translation_eligible.unsqueeze(-1)
    target = torch.as_tensor(
        target_cube_minus_ee_root_m,
        dtype=applied_controller_action.dtype,
        device=applied_controller_action.device,
    )
    remaining_error_m = cube_minus_ee_root_m - target
    direction = torch.sign(remaining_error_m)
    requested_xyz = applied_controller_action[:, :3]
    signed_requested = requested_xyz * direction
    maximum_toward_magnitude = remaining_error_m.abs() / float(
        translation_scale_m
    )
    projected_xyz = direction * torch.minimum(
        torch.clamp_min(signed_requested, 0.0),
        maximum_toward_magnitude,
    )

    wrong_direction_axis = (
        eligible_axes
        & (requested_xyz != 0.0)
        & (signed_requested <= 0.0)
    )
    overshoot_axis = eligible_axes & (
        signed_requested > maximum_toward_magnitude
    )
    projected = applied_controller_action.clone()
    projected[:, :3] = torch.where(
        eligible_axes, projected_xyz, requested_xyz
    )
    orientation_inputs = (
        current_ee_orientation_root_xyzw,
        desired_ee_orientation_root_xyzw,
        rotation_scale_rad,
    )
    if any(value is not None for value in orientation_inputs):
        if any(value is None for value in orientation_inputs):
            raise ValueError("orientation projection inputs must be supplied together")
        assert current_ee_orientation_root_xyzw is not None
        assert desired_ee_orientation_root_xyzw is not None
        assert rotation_scale_rad is not None
        if current_ee_orientation_root_xyzw.shape != (batch, 4):
            raise ValueError("current EE orientation must be [N,4] xyzw")
        if desired_ee_orientation_root_xyzw.shape not in ((4,), (batch, 4)):
            raise ValueError("desired EE orientation must be [4] or [N,4] xyzw")
        if not math.isfinite(float(rotation_scale_rad)) or rotation_scale_rad <= 0.0:
            raise ValueError("rotation scale must be finite and positive")
        desired_orientation = desired_ee_orientation_root_xyzw
        if desired_orientation.ndim == 1:
            desired_orientation = desired_orientation.unsqueeze(0).expand(batch, -1)
        if (
            desired_orientation.device != applied_controller_action.device
            or current_ee_orientation_root_xyzw.device != applied_controller_action.device
        ):
            raise ValueError("orientation tensors must share the action device")
        if (
            desired_orientation.dtype != applied_controller_action.dtype
            or current_ee_orientation_root_xyzw.dtype != applied_controller_action.dtype
        ):
            raise ValueError("orientation tensors must share the action dtype")
        error_axis_angle, current_error = _quaternion_error_axis_angle_root_xyzw(
            current_ee_orientation_root_xyzw, desired_orientation
        )
        requested_rotation = (
            applied_controller_action[:, 3:G2_VISUAL_ARM_ACTION_DIM]
            * float(rotation_scale_rad)
        )
        error_direction = torch.sign(error_axis_angle)
        signed_requested_rotation = requested_rotation * error_direction
        projected_rotation = error_direction * torch.minimum(
            torch.clamp_min(signed_requested_rotation, 0.0),
            error_axis_angle.abs(),
        )
        candidate_orientation = _quat_multiply_xyzw(
            _axis_angle_to_quaternion_xyzw(projected_rotation),
            canonicalize_quaternion_xyzw(current_ee_orientation_root_xyzw),
        )
        _, candidate_error = _quaternion_error_axis_angle_root_xyzw(
            candidate_orientation, desired_orientation
        )
        improves_orientation = candidate_error <= current_error + 1.0e-7
        projected_rotation = torch.where(
            improves_orientation.unsqueeze(-1),
            projected_rotation,
            torch.zeros_like(projected_rotation),
        )
        projected_rotation_action = projected_rotation / float(rotation_scale_rad)
        projected[:, 3:G2_VISUAL_ARM_ACTION_DIM] = torch.where(
            rotation_eligible.unsqueeze(-1),
            projected_rotation_action,
            applied_controller_action[:, 3:G2_VISUAL_ARM_ACTION_DIM],
        )
    else:
        # Backwards-compatible conservative behavior for callers that do not
        # possess a live orientation authority. Runtime supplies all three
        # inputs and therefore uses the v3 non-worsening projection.
        projected[rotation_eligible, 3:G2_VISUAL_ARM_ACTION_DIM] = 0.0

    translation_intervention = eligible & (
        projected[:, :3] != applied_controller_action[:, :3]
    ).any(dim=-1)
    rotation_intervention = rotation_eligible & (~torch.isclose(
        projected[:, 3:G2_VISUAL_ARM_ACTION_DIM],
        applied_controller_action[:, 3:G2_VISUAL_ARM_ACTION_DIM],
        rtol=1.0e-6,
        atol=1.0e-7,
    )).any(dim=-1)
    intervention = translation_intervention | rotation_intervention
    if not torch.equal(
        projected[:, G2_VISUAL_GRIPPER_ACTION_INDEX],
        applied_controller_action[:, G2_VISUAL_GRIPPER_ACTION_INDEX],
    ):
        raise RuntimeError(
            "PRE_CONTACT task-alignment projection modified the gripper"
        )
    if bool(
        (
            projected[:, :G2_VISUAL_ARM_ACTION_DIM].abs()
            > applied_controller_action[:, :G2_VISUAL_ARM_ACTION_DIM].abs()
        ).any()
    ):
        raise RuntimeError(
            "PRE_CONTACT task-alignment projection amplified an arm action"
        )
    if not torch.equal(
        projected[controller_authority_mask],
        applied_controller_action[controller_authority_mask],
    ):
        raise RuntimeError(
            "PRE_CONTACT task-alignment projection touched controller-owned row"
        )
    if not torch.equal(projected[~eligible], applied_controller_action[~eligible]):
        raise RuntimeError(
            "PRE_CONTACT task-alignment projection touched ineligible row"
        )

    return G2PreContactTaskAlignmentSafetyProjectionResult(
        action=projected,
        eligible_mask=eligible,
        translation_eligible_mask=translation_eligible,
        rotation_eligible_mask=rotation_eligible,
        wrong_direction_axis_mask=wrong_direction_axis,
        overshoot_axis_mask=overshoot_axis,
        translation_intervention_mask=translation_intervention,
        rotation_intervention_mask=rotation_intervention,
        intervention_mask=intervention,
    )


@dataclass(frozen=True)
class G2PreContactPadTableClearanceProjectionResult:
    action: torch.Tensor
    eligible_mask: torch.Tensor
    intervention_mask: torch.Tensor
    current_minimum_clearance_m: torch.Tensor
    requested_minimum_clearance_m: torch.Tensor
    applied_minimum_clearance_m: torch.Tensor
    applied_scale: torch.Tensor


def apply_g2_precontact_pad_table_clearance_projection(
    applied_controller_action: torch.Tensor,
    *,
    pure_policy_mask: torch.Tensor,
    controller_authority_mask: torch.Tensor,
    precontact_mask: torch.Tensor,
    any_finger_contact: torch.Tensor,
    current_ee_position_root_m: torch.Tensor,
    current_pad_positions_root_m: torch.Tensor,
    table_surface_height_m: float,
    minimum_clearance_m: float,
    translation_scale_m: float,
    rotation_scale_rad: float,
    bisection_iterations: int = 12,
) -> G2PreContactPadTableClearanceProjectionResult:
    """Scale a pure-policy pre-contact SE(3) request to preserve pad clearance.

    The two distal pads are treated as rigid points relative to the live EE
    frame.  Isaac Lab applies the relative rotation in the root frame, so the
    same left-multiplied angle-axis transform is used here.  No motion is
    generated: the largest sampled scalar in ``[0,1]`` that preserves the
    existing 2 mm collision margin is retained.
    """

    if applied_controller_action.ndim != 2 or applied_controller_action.shape[-1] != 7:
        raise ValueError("applied controller action must be [N,7]")
    batch = applied_controller_action.shape[0]
    for name, mask in {
        "pure_policy_mask": pure_policy_mask,
        "controller_authority_mask": controller_authority_mask,
        "precontact_mask": precontact_mask,
        "any_finger_contact": any_finger_contact,
    }.items():
        if mask.shape != (batch,) or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean [N]")
        if mask.device != applied_controller_action.device:
            raise ValueError(f"{name} must share the action device")
    if bool((pure_policy_mask & controller_authority_mask).any()):
        raise ValueError("pure-policy and controller authority masks overlap")
    if current_ee_position_root_m.shape != (batch, 3):
        raise ValueError("current EE position must be [N,3]")
    if current_pad_positions_root_m.shape != (batch, 2, 3):
        raise ValueError("current pad positions must be [N,2,3]")
    if not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (minimum_clearance_m, translation_scale_m, rotation_scale_rad)
    ):
        raise ValueError("clearance and action scales must be finite and positive")
    if not math.isfinite(float(table_surface_height_m)):
        raise ValueError("table surface height must be finite")
    if bisection_iterations <= 0:
        raise ValueError("bisection iterations must be positive")
    if not bool(torch.isfinite(current_pad_positions_root_m).all()):
        raise ValueError("pad positions contain non-finite values")

    eligible = pure_policy_mask & precontact_mask & ~any_finger_contact
    current_clearance = (
        current_pad_positions_root_m[..., 2].amin(dim=-1)
        - float(table_surface_height_m)
    )
    pad_offsets = current_pad_positions_root_m - current_ee_position_root_m.unsqueeze(1)

    def candidate_clearance(scale: torch.Tensor) -> torch.Tensor:
        translation = (
            applied_controller_action[:, :3]
            * float(translation_scale_m)
            * scale.unsqueeze(-1)
        )
        axis_angle = (
            applied_controller_action[:, 3:6]
            * float(rotation_scale_rad)
            * scale.unsqueeze(-1)
        )
        quaternion = _axis_angle_to_quaternion_xyzw(axis_angle)
        vector_quaternion = torch.cat(
            (pad_offsets, torch.zeros_like(pad_offsets[..., :1])), dim=-1
        )
        expanded_quaternion = quaternion.unsqueeze(1).expand(-1, 2, -1)
        rotated_offsets = _quat_multiply_xyzw(
            _quat_multiply_xyzw(expanded_quaternion, vector_quaternion),
            _quat_conjugate_xyzw(expanded_quaternion),
        )[..., :3]
        candidate_pad = (
            current_ee_position_root_m.unsqueeze(1)
            + translation.unsqueeze(1)
            + rotated_offsets
        )
        return candidate_pad[..., 2].amin(dim=-1) - float(table_surface_height_m)

    ones = torch.ones(batch, dtype=applied_controller_action.dtype, device=applied_controller_action.device)
    requested_clearance = candidate_clearance(ones)
    safe_at_full = requested_clearance >= float(minimum_clearance_m)
    safe_at_zero = current_clearance >= float(minimum_clearance_m)
    low = torch.zeros_like(ones)
    high = torch.ones_like(ones)
    for _ in range(bisection_iterations):
        middle = 0.5 * (low + high)
        safe = candidate_clearance(middle) >= float(minimum_clearance_m)
        low = torch.where(safe, middle, low)
        high = torch.where(safe, high, middle)
    scale = torch.where(safe_at_full, ones, low)
    scale = torch.where(safe_at_zero, scale, torch.zeros_like(scale))
    scale = torch.where(eligible, scale, ones)
    projected = applied_controller_action.clone()
    projected[:, :6] = projected[:, :6] * scale.unsqueeze(-1)
    intervention = eligible & (scale < 1.0 - 1.0e-7)
    if not torch.equal(projected[:, 6], applied_controller_action[:, 6]):
        raise RuntimeError("pad/table clearance projection modified the gripper")
    if not torch.equal(projected[controller_authority_mask], applied_controller_action[controller_authority_mask]):
        raise RuntimeError("pad/table clearance projection touched controller-owned row")
    return G2PreContactPadTableClearanceProjectionResult(
        action=projected,
        eligible_mask=eligible,
        intervention_mask=intervention,
        current_minimum_clearance_m=current_clearance,
        requested_minimum_clearance_m=requested_clearance,
        applied_minimum_clearance_m=candidate_clearance(scale),
        applied_scale=scale,
    )


@dataclass(frozen=True)
class G2PurePolicyGraspReadyInterlockResult:
    """Post-transform result for the demo-calibrated close interlock."""

    action: torch.Tensor
    precontact_policy_mask: torch.Tensor
    arm_hold_mask: torch.Tensor
    grasp_ready_arm_hold_mask: torch.Tensor
    latched_closed_recovery_arm_hold_mask: torch.Tensor
    position_ready_orientation_blocked_mask: torch.Tensor
    orientation_close_rejected_mask: torch.Tensor
    outside_close_rejected_mask: torch.Tensor
    intervention_mask: torch.Tensor


def apply_g2_pure_policy_grasp_ready_interlock(
    post_transform_action: torch.Tensor,
    *,
    pure_policy_mask: torch.Tensor,
    controller_authority_mask: torch.Tensor,
    any_finger_contact: torch.Tensor,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
    calibrated_grasp_offset_error_m: torch.Tensor,
    calibrated_grasp_orientation_error_rad: torch.Tensor,
    close_position_tolerance_m: float,
    close_orientation_tolerance_rad: float,
    current_gripper_latch: torch.Tensor,
) -> G2PurePolicyGraspReadyInterlockResult:
    """Coordinate policy approach and closure without generating an action.

    The reference controller stops Cartesian progression at the existing
    demonstration-derived close boundary before closing.  A pure-policy row
    previously lacked the same mechanical interlock: it could close outside
    that boundary or keep advancing through it while the fingers moved.  This
    helper reuses that exact boundary.  Before contact it rejects only an
    unsafe early close (substituting the already-audited open command).  A
    close is preserved only when both the demonstrated position shell and the
    existing pre-grasp orientation tolerance are satisfied.  Position-ready
    rows with bad orientation are translation-held while rotational
    correction remains possible.  If the minimum-dwell latch remains closed
    after either geometry condition is lost, the full arm remains held until
    the latch can reopen or contact takes over.  This prevents the passive
    outer-link mechanism from closing sideways into the object.  It never
    auto-closes the gripper, never touches controller-owned rows, and changes
    no force, motion, reward, or termination threshold.
    """

    if (
        post_transform_action.ndim != 2
        or post_transform_action.shape[-1] != G2_VISUAL_ACTION_DIM
    ):
        raise ValueError("post-transform action must be [N,7]")
    if not torch.is_floating_point(post_transform_action):
        raise ValueError("post-transform action must be floating point")
    if not bool(torch.isfinite(post_transform_action).all()):
        raise ValueError("post-transform action contains non-finite values")
    batch = post_transform_action.shape[0]
    masks = {
        "pure_policy_mask": pure_policy_mask,
        "controller_authority_mask": controller_authority_mask,
        "any_finger_contact": any_finger_contact,
        "bilateral_contact": bilateral_contact,
        "stable_grasp": stable_grasp,
    }
    for name, value in masks.items():
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [N] tensor")
        if value.device != post_transform_action.device:
            raise ValueError(f"{name} must share the action device")
    if bool((pure_policy_mask & controller_authority_mask).any()):
        raise ValueError("pure-policy and controller authority masks overlap")
    if calibrated_grasp_offset_error_m.shape != (batch,):
        raise ValueError("calibrated grasp-offset error must be [N]")
    if calibrated_grasp_offset_error_m.device != post_transform_action.device:
        raise ValueError("grasp-offset error must share the action device")
    if not bool(torch.isfinite(calibrated_grasp_offset_error_m).all()):
        raise ValueError("grasp-offset error contains non-finite values")
    if calibrated_grasp_orientation_error_rad.shape != (batch,):
        raise ValueError("grasp-orientation error must be [N]")
    if (
        calibrated_grasp_orientation_error_rad.device
        != post_transform_action.device
    ):
        raise ValueError("grasp-orientation error must share the action device")
    if not bool(torch.isfinite(calibrated_grasp_orientation_error_rad).all()):
        raise ValueError("grasp-orientation error contains non-finite values")
    if bool((calibrated_grasp_orientation_error_rad < 0.0).any()):
        raise ValueError("grasp-orientation error must be non-negative")
    if (
        not math.isfinite(float(close_position_tolerance_m))
        or close_position_tolerance_m <= 0.0
    ):
        raise ValueError("close position tolerance must be finite and positive")
    if (
        not math.isfinite(float(close_orientation_tolerance_rad))
        or not 0.0 < close_orientation_tolerance_rad <= math.pi
    ):
        raise ValueError(
            "close orientation tolerance must be in the finite interval (0,pi]"
        )
    if current_gripper_latch.shape != (batch,):
        raise ValueError("current gripper latch must be [N]")
    if (
        current_gripper_latch.device != post_transform_action.device
        or current_gripper_latch.dtype != post_transform_action.dtype
    ):
        raise ValueError("current gripper latch must share action device and dtype")
    if not bool(torch.isfinite(current_gripper_latch).all()):
        raise ValueError("current gripper latch contains non-finite values")

    precontact_policy = (
        pure_policy_mask
        & ~any_finger_contact
        & ~bilateral_contact
        & ~stable_grasp
    )
    position_ready = _metric_leq_with_dtype_roundoff(
        calibrated_grasp_offset_error_m,
        close_position_tolerance_m,
    )
    orientation_ready = calibrated_grasp_orientation_error_rad <= float(
        close_orientation_tolerance_rad
    )
    grasp_ready = position_ready & orientation_ready
    position_ready_orientation_blocked = (
        precontact_policy & position_ready & ~orientation_ready
    )
    latched_closed_recovery_hold = (
        precontact_policy & ~grasp_ready & (current_gripper_latch < 0.0)
    )
    grasp_ready_arm_hold = precontact_policy & grasp_ready
    full_arm_hold = grasp_ready_arm_hold | latched_closed_recovery_hold
    translation_only_hold = (
        position_ready_orientation_blocked & ~latched_closed_recovery_hold
    )
    arm_hold = full_arm_hold | translation_only_hold
    requested_close = (
        post_transform_action[:, G2_VISUAL_GRIPPER_ACTION_INDEX] < 0.0
    )
    reject_close = precontact_policy & ~grasp_ready & requested_close
    orientation_reject_close = reject_close & ~orientation_ready

    projected = post_transform_action.clone()
    projected[full_arm_hold, :G2_VISUAL_ARM_ACTION_DIM] = 0.0
    projected[translation_only_hold, :3] = 0.0
    projected[reject_close, G2_VISUAL_GRIPPER_ACTION_INDEX] = 1.0
    # Count an intervention only when the emitted action actually changed.
    # ``arm_hold`` remains a separate semantic/eligibility counter: a row
    # whose actor already requested zero arm motion is held safely but is not
    # misreported as a projection delta.
    intervention = (projected != post_transform_action).any(dim=-1)

    if not torch.equal(
        projected[controller_authority_mask],
        post_transform_action[controller_authority_mask],
    ):
        raise RuntimeError("grasp-ready interlock touched controller-owned row")
    if not torch.equal(
        projected[~precontact_policy],
        post_transform_action[~precontact_policy],
    ):
        raise RuntimeError("grasp-ready interlock touched an ineligible row")
    if bool(
        (
            projected[:, :G2_VISUAL_ARM_ACTION_DIM].abs()
            > post_transform_action[:, :G2_VISUAL_ARM_ACTION_DIM].abs()
        ).any()
    ):
        raise RuntimeError("grasp-ready interlock amplified an arm request")
    return G2PurePolicyGraspReadyInterlockResult(
        action=projected,
        precontact_policy_mask=precontact_policy,
        arm_hold_mask=arm_hold,
        grasp_ready_arm_hold_mask=grasp_ready_arm_hold,
        latched_closed_recovery_arm_hold_mask=latched_closed_recovery_hold,
        position_ready_orientation_blocked_mask=(
            position_ready_orientation_blocked
        ),
        orientation_close_rejected_mask=orientation_reject_close,
        outside_close_rejected_mask=reject_close,
        intervention_mask=intervention,
    )


@dataclass(frozen=True)
class G2PurePolicyContactSafetyGateResult:
    """Result of the fixed, non-generative pure-policy contact projection."""

    action: torch.Tensor
    contact_settle_mask: torch.Tensor
    stable_lift_projection_mask: torch.Tensor
    intervention_mask: torch.Tensor


def apply_g2_pure_policy_contact_safety_gate(
    applied_controller_action: torch.Tensor,
    *,
    pure_policy_mask: torch.Tensor,
    controller_authority_mask: torch.Tensor,
    any_finger_contact: torch.Tensor,
    bilateral_contact: torch.Tensor,
    stable_grasp: torch.Tensor,
) -> G2PurePolicyContactSafetyGateResult:
    """Project only pure-policy contact actions onto the fixed safe subspace.

    The projection is deliberately applied *after* the stateful slew transform.
    Setting only the normalized request to zero can otherwise leave several
    steps of residual Cartesian motion while the gripper is making contact.

    This is not a reference action and it never adds a non-zero component:

    * before stable grasp, any measured finger contact holds all six arm axes;
    * after stable grasp, root-frame X/Y and all rotation deltas are held while
      only the actor's upward (+Z) component is retained.

    The gripper channel is unchanged.  Reference and assisted rows are
    explicitly disjoint and therefore remain owned by their audited recovery
    controllers.
    """

    if (
        applied_controller_action.ndim != 2
        or applied_controller_action.shape[-1] != G2_VISUAL_ACTION_DIM
    ):
        raise ValueError("applied controller action must be [N,7]")
    if not bool(torch.isfinite(applied_controller_action).all()):
        raise ValueError("applied controller action contains non-finite values")
    batch = applied_controller_action.shape[0]
    masks = {
        "pure_policy_mask": pure_policy_mask,
        "controller_authority_mask": controller_authority_mask,
        "any_finger_contact": any_finger_contact,
        "bilateral_contact": bilateral_contact,
        "stable_grasp": stable_grasp,
    }
    for name, value in masks.items():
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [N] tensor")
        if value.device != applied_controller_action.device:
            raise ValueError(f"{name} must share the action device")
    if bool((pure_policy_mask & controller_authority_mask).any()):
        raise ValueError("pure-policy and controller authority masks overlap")

    gated = applied_controller_action.clone()
    contact_settle = (
        pure_policy_mask
        & (any_finger_contact | bilateral_contact)
        & ~stable_grasp
    )
    stable_lift = pure_policy_mask & stable_grasp
    gated[contact_settle, :G2_VISUAL_ARM_ACTION_DIM] = 0.0
    if bool(stable_lift.any()):
        # Differential-IK translation deltas are expressed in the robot-root
        # frame.  Preserve only the actor-selected upward lift component.
        gated[stable_lift, 0] = 0.0
        gated[stable_lift, 1] = 0.0
        gated[stable_lift, 2] = torch.clamp_min(gated[stable_lift, 2], 0.0)
        gated[stable_lift, 3:G2_VISUAL_ARM_ACTION_DIM] = 0.0
    if not torch.equal(
        gated[:, G2_VISUAL_GRIPPER_ACTION_INDEX],
        applied_controller_action[:, G2_VISUAL_GRIPPER_ACTION_INDEX],
    ):
        raise RuntimeError("pure-policy safety gate modified the gripper")
    if bool(
        (
            gated[:, :G2_VISUAL_ARM_ACTION_DIM].abs()
            > applied_controller_action[:, :G2_VISUAL_ARM_ACTION_DIM].abs()
        ).any()
    ):
        raise RuntimeError("pure-policy safety gate amplified an arm action")
    intervention = (
        gated[:, :G2_VISUAL_ARM_ACTION_DIM]
        != applied_controller_action[:, :G2_VISUAL_ARM_ACTION_DIM]
    ).any(dim=1)
    if bool((intervention & controller_authority_mask).any()):
        raise RuntimeError("pure-policy safety gate touched controller-owned row")
    return G2PurePolicyContactSafetyGateResult(
        action=gated,
        contact_settle_mask=contact_settle,
        stable_lift_projection_mask=stable_lift,
        intervention_mask=intervention,
    )


@dataclass(frozen=True)
class G2RecurrentVisualPolicyContract:
    """One sealed input/backbone contract for keyboard, teacher and student."""

    schema: str = G2_RECURRENT_VISUAL_POLICY_SCHEMA
    camera_shape: tuple[int, int, int, int] = G2_VISUAL_CAMERA_SHAPE
    proprioception_dim: int = G2_VISUAL_PROPRIO_DIM
    action_dim: int = G2_VISUAL_ACTION_DIM
    hidden_dim: int = 256
    gru_num_layers: int = 1
    sequence_length: int = 16
    burn_in_steps: int = 4
    sequence_stride: int = 12
    camera_encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
        "baseline_4layer"
    ]
    head_observation_mode: str = "cube_detector_depth_xyz"
    head_depth_profile: str = "legacy"

    def validated(self) -> "G2RecurrentVisualPolicyContract":
        if self.camera_shape != (2, 6, 48, 64):
            raise ValueError("visual policy requires head/wrist packed RGB-D")
        if self.proprioception_dim != 45 or self.action_dim != 7:
            raise ValueError("visual policy input/output dimension drift")
        if self.hidden_dim <= 0:
            raise ValueError("visual policy hidden dimension must be positive")
        if self.gru_num_layers != 1:
            raise ValueError("visual policy contract requires exactly one GRU layer")
        if not 0 <= self.burn_in_steps < self.sequence_length:
            raise ValueError("visual policy burn-in must be inside the sequence")
        if not 0 < self.sequence_stride <= self.sequence_length:
            raise ValueError("visual policy sequence stride must be in [1, sequence_length]")
        if tuple(self.camera_encoder_channels) not in G2_CAMERA_ENCODER_PROFILES.values():
            raise ValueError("unknown camera encoder channel contract")
        if self.head_observation_mode not in G2_HEAD_OBSERVATION_MODES:
            raise ValueError("unknown head observation mode")
        if self.head_depth_profile not in G2_HEAD_DEPTH_PROFILES:
            raise ValueError("unknown head-depth profile")
        return self

    def serializable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "camera_order": ["head", "right_wrist"],
            "camera_channels": ["rgb_r", "rgb_g", "rgb_b", "depth_hi", "depth_lo", "depth_valid"],
            "camera_shape": list(self.camera_shape),
            "proprioception_dim": self.proprioception_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "sequence_length": self.sequence_length,
            "burn_in_steps": self.burn_in_steps,
            "sequence_stride": self.sequence_stride,
            "backbone": "HEAD_CUBE_DETECTOR_XYZ_PLUS_WRIST_SPATIAL_SOFTMAX_GRU",
            "head_observation_mode": "cube_detector_depth_xyz",
            "head_depth_profile": self.head_depth_profile,
            "head_actor_input": "CUBE_XYZ_CONFIDENCE_VALIDITY_ONLY_NO_RAW_TOKEN",
            "head_detector_supervision": "SIMULATOR_GT_LABEL_TRAIN_AND_EVAL_ONLY",
            "grasp_distance_source": "DISTAL_PAD_MIDPOINT_TO_ESTIMATED_CUBE_XYZ",
            "deployable_phase_order": list(G2_DEPLOYABLE_PHASES),
            "camera_encoder_profile": g2_camera_encoder_profile_name(
                self.camera_encoder_channels
            ),
            "camera_encoder_channels": list(self.camera_encoder_channels),
            "camera_fusion": "DEPLOYABLE_DISTANCE_VISIBILITY_GATED_HEAD_XYZ_WRIST_FEATURE_NO_ATTENTION",
            "cross_camera_attention": False,
            "cross_camera_contrastive": "OPTIONAL_LEARNER_ONLY_CUBE_ROI_INFONCE",
            "actor_ground_truth_cube_position": False,
            "torso_policy_output_enabled": False,
        }


def _quat_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _quat_apply_inverse_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
    pure = torch.cat((vector, torch.zeros_like(vector[..., :1])), dim=-1)
    return _quat_multiply_xyzw(_quat_multiply_xyzw(conjugate, pure), quaternion)[..., :3]


def _axis_angle_quaternion_xyzw(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in axis))
    scale = math.sin(angle_rad / 2.0) / norm
    return (
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(angle_rad / 2.0),
    )


# The 24 proper rotations of a cube, represented in the cube-local frame.
# A visually and physically symmetric cube orientation is therefore q*S.
_CUBE_ROTATIONAL_SYMMETRIES_XYZW = (
    ((0.0, 0.0, 0.0, 1.0),)
    + tuple(
        _axis_angle_quaternion_xyzw(axis, angle)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for angle in (math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    )
    + tuple(
        _axis_angle_quaternion_xyzw(axis, angle)
        for axis in (
            (1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
        )
        for angle in (2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    )
    + tuple(
        _axis_angle_quaternion_xyzw(axis, math.pi)
        for axis in (
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, -1.0),
        )
    )
)


def relative_pose_target_from_teacher_state(state: torch.Tensor) -> torch.Tensor:
    """Return cube pose in the EE frame from the privileged XYZW contract."""

    if state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("teacher state must have 59 fields")
    slices = G2TeacherObservationContract().slices
    ee = state[..., slices["end_effector_pose_root_xyzw"]]
    cube = state[..., slices["cube_pose_root_xyzw"]]
    ee_quaternion = canonicalize_quaternion_xyzw(ee[..., 3:7])
    cube_quaternion = canonicalize_quaternion_xyzw(cube[..., 3:7])
    relative_position = _quat_apply_inverse_xyzw(
        ee_quaternion, cube[..., :3] - ee[..., :3]
    )
    ee_conjugate = torch.cat((-ee_quaternion[..., :3], ee_quaternion[..., 3:4]), dim=-1)
    relative_quaternion = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(ee_conjugate, cube_quaternion)
    )
    return torch.cat((relative_position, relative_quaternion), dim=-1)


def deployable_phase_one_hot_from_recorded_state(
    state: torch.Tensor,
    *,
    stable_grasp: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map recorded physical events to the deployable five-phase contract.

    This helper is for replay/demonstration reconstruction.  Online rollout
    builds the same phase from live contact/gripper state; simulator cube pose
    is deliberately not inspected here.
    """

    if state.shape[-1] != G2_VISUAL_PRIVILEGED_DIM:
        raise ValueError("recorded Teacher state must end in 59 fields")
    contact = state[..., G2TeacherObservationContract().slices[
        "bilateral_contact_features"
    ]].amin(dim=-1) > 0.5
    if stable_grasp is None:
        stable = torch.zeros_like(contact)
    else:
        stable = stable_grasp.to(device=state.device).reshape(contact.shape) > 0.5
    pregrasp = torch.zeros_like(contact)
    lift = torch.zeros_like(contact)
    if previous_action is not None:
        if (
            previous_action.shape[:-1] != state.shape[:-1]
            or previous_action.shape[-1] != G2_VISUAL_ACTION_DIM
        ):
            raise ValueError(
                "recorded previous action and state leading dimensions differ"
            )
        # Match the live rollout contract exactly: phase is observable state,
        # never the action being supervised at this timestep.  Conditioning
        # this feature on the current expert close label leaks the answer into
        # BC and creates a deadlock at deployment (PRE_GRASP is only observed
        # after the policy has already emitted its first close command).
        pregrasp = (
            previous_action[..., G2_VISUAL_GRIPPER_ACTION_INDEX] < 0.0
        )
        # LIFT is deployable only when a physically stable grasp is followed
        # by an upward arm command.  Never expose the simulator cube-height
        # milestone through the actor phase input.
        lift = stable & (previous_action[..., 2] > 0.0)
    index = torch.zeros_like(contact, dtype=torch.long)
    index = torch.where(pregrasp, torch.ones_like(index), index)
    index = torch.where(contact, torch.full_like(index, 2), index)
    index = torch.where(stable, torch.full_like(index, 3), index)
    index = torch.where(lift, torch.full_like(index, 4), index)
    return F.one_hot(index, num_classes=G2_DEPLOYABLE_PHASE_DIM).to(state.dtype)


def relative_pose_auxiliary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    cube_symmetry_invariant: bool = True,
) -> torch.Tensor:
    """Position plus sign- and cube-symmetry-invariant orientation loss."""

    if prediction.shape != target.shape or prediction.shape[-1] != 7:
        raise ValueError("relative pose prediction/target must have matching [...,7] shape")
    position = F.smooth_l1_loss(prediction[..., :3], target[..., :3], reduction="none").mean(-1)
    predicted_quaternion = canonicalize_quaternion_xyzw(prediction[..., 3:7])
    target_quaternion = canonicalize_quaternion_xyzw(target[..., 3:7])
    if cube_symmetry_invariant:
        symmetries = target_quaternion.new_tensor(_CUBE_ROTATIONAL_SYMMETRIES_XYZW)
        equivalent_targets = _quat_multiply_xyzw(
            target_quaternion.unsqueeze(-2), symmetries
        )
        difference = (
            predicted_quaternion.unsqueeze(-2) - equivalent_targets
        ).square().mean(-1)
        opposite_difference = (
            predicted_quaternion.unsqueeze(-2) + equivalent_targets
        ).square().mean(-1)
        orientation = torch.minimum(difference, opposite_difference).amin(-1)
    else:
        orientation = torch.minimum(
            (predicted_quaternion - target_quaternion).square().mean(-1),
            (predicted_quaternion + target_quaternion).square().mean(-1),
        )
    value = position + orientation
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    raise ValueError("relative pose loss reduction must be 'none' or 'mean'")


def relative_pose_consistency_loss(
    original_prediction: torch.Tensor,
    shifted_prediction: torch.Tensor,
    *,
    rotation_weight: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pose invariance loss for original and aligned RGB-D shift views.

    Position uses the requested L1 metric.  Orientation uses the sign-invariant
    quaternion geodesic distance in radians.  No privileged target is consumed:
    both arguments are predictions from the deployable visual/GRU path.
    """

    if (
        original_prediction.shape != shifted_prediction.shape
        or original_prediction.shape[-1] != 7
    ):
        raise ValueError("pose consistency predictions must have matching [...,7] shape")
    if rotation_weight < 0.0:
        raise ValueError("pose consistency rotation weight cannot be negative")
    position = (original_prediction[..., :3] - shifted_prediction[..., :3]).abs().sum(-1)
    original_quaternion = canonicalize_quaternion_xyzw(
        original_prediction[..., 3:7]
    )
    shifted_quaternion = canonicalize_quaternion_xyzw(
        shifted_prediction[..., 3:7]
    )
    cosine_half_angle = (original_quaternion * shifted_quaternion).sum(-1).abs()
    # Avoid the undefined sqrt/atan2 gradient at exactly identical unit
    # quaternions; this introduces only a sub-milliradian numerical floor.
    cosine_half_angle = cosine_half_angle.clamp(0.0, 1.0 - 1.0e-7)
    sine_half_angle = torch.sqrt((1.0 - cosine_half_angle.square()).clamp_min(0.0))
    rotation = 2.0 * torch.atan2(sine_half_angle, cosine_half_angle.clamp_min(1.0e-8))
    value = position + float(rotation_weight) * rotation
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    raise ValueError("pose consistency reduction must be 'none' or 'mean'")


def cross_camera_contrastive_loss(
    projected_camera_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    relative_pose_target: torch.Tensor | None = None,
    temperature: float = 0.1,
    false_negative_position_threshold_m: float = 0.01,
) -> torch.Tensor:
    """Symmetric head/wrist InfoNCE over camera-specific projection tokens.

    The positive pair is the head and right-wrist observation from the same
    environment and timestep.  Other samples whose privileged relative
    positions are nearly identical are removed from the denominator rather
    than treated as false negatives.  Privileged pose is used only to build
    this learner-side mask and never enters the deployable actor input.
    """

    if projected_camera_tokens.ndim != 4 or projected_camera_tokens.shape[2] != 2:
        raise ValueError("projected camera tokens must be [B,T,2,D]")
    if tuple(valid_mask.shape) != tuple(projected_camera_tokens.shape[:2]):
        raise ValueError("contrastive valid mask must be [B,T]")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    if false_negative_position_threshold_m < 0.0:
        raise ValueError("false-negative position threshold cannot be negative")
    selected = projected_camera_tokens[valid_mask.to(torch.bool)]
    if selected.shape[0] < 2:
        return projected_camera_tokens.sum() * 0.0
    head = F.normalize(selected[:, 0], dim=-1)
    wrist = F.normalize(selected[:, 1], dim=-1)
    logits = head @ wrist.transpose(0, 1) / float(temperature)
    allowed = torch.ones_like(logits, dtype=torch.bool)
    if relative_pose_target is not None:
        if tuple(relative_pose_target.shape[:2]) != tuple(valid_mask.shape) or relative_pose_target.shape[-1] != 7:
            raise ValueError("contrastive relative-pose target must be [B,T,7]")
        positions = relative_pose_target[valid_mask.to(torch.bool), :3]
        separation = torch.cdist(positions, positions)
        allowed &= separation > float(false_negative_position_threshold_m)
    allowed.fill_diagonal_(True)
    floor = torch.finfo(logits.dtype).min
    row_logits = logits.masked_fill(~allowed, floor)
    column_logits = logits.transpose(0, 1).masked_fill(~allowed.transpose(0, 1), floor)
    diagonal = torch.arange(logits.shape[0], device=logits.device)
    row_loss = -row_logits[diagonal, diagonal] + torch.logsumexp(row_logits, dim=1)
    column_loss = -column_logits[diagonal, diagonal] + torch.logsumexp(column_logits, dim=1)
    # An untrained InfoNCE objective is approximately log(N).  Normalize by
    # that value so its configured weight has the same meaning for offline BC
    # and online SAC batches of different sizes.
    normalizer = max(math.log(float(logits.shape[0])), 1.0)
    return 0.5 * (row_loss.mean() + column_loss.mean()) / normalizer


def temporal_pose_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    rotation_weight: float = 0.1,
) -> torch.Tensor:
    """Match consecutive object-in-EE pose changes without crossing resets."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 7:
        raise ValueError("temporal pose prediction/target must be matching [B,T,7]")
    if tuple(valid_mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("temporal pose valid mask must be [B,T]")
    if rotation_weight < 0.0:
        raise ValueError("temporal pose rotation weight cannot be negative")
    if prediction.shape[1] < 2:
        return prediction.sum() * 0.0
    pair_valid = valid_mask[:, 1:].to(torch.bool) & valid_mask[:, :-1].to(torch.bool)
    if not bool(pair_valid.any()):
        return prediction.sum() * 0.0
    predicted_position_delta = prediction[:, 1:, :3] - prediction[:, :-1, :3]
    target_position_delta = target[:, 1:, :3] - target[:, :-1, :3]
    position = F.smooth_l1_loss(
        predicted_position_delta, target_position_delta, reduction="none"
    ).mean(-1)

    predicted_previous = canonicalize_quaternion_xyzw(prediction[:, :-1, 3:7])
    predicted_next = canonicalize_quaternion_xyzw(prediction[:, 1:, 3:7])
    target_previous = canonicalize_quaternion_xyzw(target[:, :-1, 3:7])
    target_next = canonicalize_quaternion_xyzw(target[:, 1:, 3:7])
    predicted_delta = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(
            torch.cat((-predicted_previous[..., :3], predicted_previous[..., 3:4]), dim=-1),
            predicted_next,
        )
    )
    target_delta = canonicalize_quaternion_xyzw(
        _quat_multiply_xyzw(
            torch.cat((-target_previous[..., :3], target_previous[..., 3:4]), dim=-1),
            target_next,
        )
    )
    cosine_half_angle = (predicted_delta * target_delta).sum(-1).abs().clamp(0.0, 1.0 - 1.0e-7)
    sine_half_angle = torch.sqrt((1.0 - cosine_half_angle.square()).clamp_min(0.0))
    rotation = 2.0 * torch.atan2(sine_half_angle, cosine_half_angle.clamp_min(1.0e-8))
    value = position + float(rotation_weight) * rotation
    weight = pair_valid.to(value.dtype)
    return (weight * value).sum() / weight.sum().clamp_min(1.0)


@dataclass(frozen=True)
class G2StudentObservationContract:
    """Ordered deployment input excluding all privileged task oracles."""

    arm_hand_joint_count: int = 8
    torso_joint_count: int = 5
    previous_action_dim: int = G2_VISUAL_ACTION_DIM

    @property
    def fields(self) -> tuple[tuple[str, int], ...]:
        return (
            ("arm_hand_joint_position_relative_rad", self.arm_hand_joint_count),
            ("arm_hand_joint_velocity_rad_s", self.arm_hand_joint_count),
            ("torso_joint_position_relative_rad", self.torso_joint_count),
            ("torso_joint_velocity_rad_s", self.torso_joint_count),
            ("end_effector_pose_root_xyzw", 7),
            ("goal_position_root_m", 3),
            ("previous_action", self.previous_action_dim),
            ("camera_frame_age_s", 2),
        )

    @property
    def observation_dim(self) -> int:
        return sum(width for _, width in self.fields)

    @property
    def slices(self) -> Mapping[str, slice]:
        result: dict[str, slice] = {}
        cursor = 0
        for name, width in self.fields:
            result[name] = slice(cursor, cursor + width)
            cursor += width
        return result

    def build(self, **values: torch.Tensor) -> torch.Tensor:
        expected_names = tuple(name for name, _ in self.fields)
        if tuple(values) != expected_names:
            raise ValueError(f"student observation fields/order differ: {tuple(values)}")
        batch = next(iter(values.values())).shape[0]
        pieces = []
        for name, width in self.fields:
            value = values[name]
            if tuple(value.shape) != (batch, width):
                raise ValueError(f"{name} must be {(batch, width)}, got {tuple(value.shape)}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
            if name == "end_effector_pose_root_xyzw":
                value = torch.cat(
                    (value[:, :3], canonicalize_quaternion_xyzw(value[:, 3:7])),
                    dim=-1,
                )
            pieces.append(value)
        observation = torch.cat(pieces, dim=-1)
        if observation.shape[-1] != G2_VISUAL_PROPRIO_DIM:
            raise RuntimeError("student observation dimension contract mismatch")
        return observation


def pack_rgbd(
    head_rgb: torch.Tensor,
    head_depth_m: torch.Tensor,
    wrist_rgb: torch.Tensor,
    wrist_depth_m: torch.Tensor,
    *,
    output_hw: tuple[int, int] = (48, 64),
    maximum_depth_m: float = 2.0,
) -> torch.Tensor:
    """Return compact uint8 ``[N,2,6,H,W]`` RGB/depth-hi/depth-lo/valid.

    Metric depth is clipped then encoded as an unsigned 16-bit integer split
    across two uint8 channels.  At the default 2 m range the quantization is
    about 0.031 mm, rather than the former 7.84 mm single-byte step.
    """

    if maximum_depth_m <= 0.0:
        raise ValueError("maximum_depth_m must be positive")

    def one(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise ValueError("RGB must have shape [N,H,W,C>=3]")
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3 or depth.shape[:3] != rgb.shape[:3]:
            raise ValueError("depth must match RGB batch/height/width")
        rgb_chw = rgb[..., :3].permute(0, 3, 1, 2).float()
        if rgb.dtype != torch.uint8:
            # Isaac camera products are floating point but, depending on the
            # renderer/backend, use either [0, 1] or [0, 255].  Treating every
            # floating product as normalized saturated the live rollout to an
            # all-white image while keyboard demonstrations remained uint8.
            # Keep the choice on-device so the camera path does not introduce
            # a per-frame CPU synchronization.
            normalized_scale = torch.where(
                rgb_chw.detach().amax() <= 1.0 + 1.0e-6,
                rgb_chw.new_tensor(255.0),
                rgb_chw.new_tensor(1.0),
            )
            rgb_chw = rgb_chw.clamp(0.0, 255.0) * normalized_scale
        valid = torch.isfinite(depth) & (depth > 0)
        valid_depth = torch.where(valid, depth, 0.0)
        depth_chw = valid_depth.clamp_max(maximum_depth_m).unsqueeze(1)
        valid_chw = valid.to(depth_chw.dtype).unsqueeze(1)
        rgb_small = F.interpolate(rgb_chw, output_hw, mode="area")
        depth_small = F.interpolate(depth_chw, output_hw, mode="nearest")
        valid_small = F.interpolate(valid_chw, output_hw, mode="nearest")
        depth_u16 = depth_small.mul(65535.0 / maximum_depth_m).round().clamp(0, 65535)
        depth_high = torch.floor(depth_u16 / 256.0)
        depth_low = depth_u16 - depth_high * 256.0
        return torch.cat(
            (rgb_small, depth_high, depth_low, valid_small.mul(255.0)), dim=1
        ).round().clamp(0, 255).to(torch.uint8)

    return torch.stack((one(head_rgb, head_depth_m), one(wrist_rgb, wrist_depth_m)), dim=1)


def merge_corrective_demonstration_batch(
    static_batch: Mapping[str, torch.Tensor] | None,
    corrective_numpy: Mapping[str, np.ndarray],
    *,
    hidden_dim: int,
    burn_in_steps: int,
    confidence_scale: float,
) -> dict[str, torch.Tensor | int]:
    """Merge canonical and online-corrective recurrent BC sequences.

    Replay exposes provenance and deployable-geometry telemetry in addition
    to tensors consumed by BC.  The old generic dictionary merge treated
    those extra fields as mandatory canonical fields and failed exactly when
    joint training first sampled a corrective sequence.  This contract merges
    only learner inputs, while retaining the live distal-pad grasp center for
    corrective rows.  Canonical and corrective rows must both provide the
    explicit pad midpoint; older EE-origin-only HDF5 files fail closed.
    """

    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be non-negative")
    if not 0.0 <= confidence_scale <= 1.0:
        raise ValueError("confidence_scale must be in [0,1]")

    required = (
        "rgbd_u8",
        "deployable_proprioception",
        "grasp_center_position_root_m",
        "expert_action_target",
        "teacher_state_target",
        "expert_confidence",
        "padding_mask",
        "hidden_reset_mask",
        "episode_id",
        "sequence_step",
        "sequence_lengths",
    )
    corrective: dict[str, torch.Tensor] = {}
    for name in (*required, "initial_recurrent_hidden"):
        value = corrective_numpy.get(name)
        if value is not None:
            corrective[name] = torch.from_numpy(np.asarray(value))
    missing = [name for name in required if name not in corrective]
    if missing:
        raise ValueError(f"corrective BC batch is missing fields: {missing}")
    corrective["expert_confidence"] = (
        corrective["expert_confidence"] * float(confidence_scale)
    )

    corrective_count = int(corrective["rgbd_u8"].shape[0])
    if "initial_recurrent_hidden" not in corrective:
        corrective["initial_recurrent_hidden"] = torch.zeros(
            (corrective_count, hidden_dim), dtype=torch.float32
        )
    if static_batch is None:
        result: dict[str, torch.Tensor | int] = dict(corrective)
    else:
        result = dict(static_batch)
        static_count = int(static_batch["rgbd_u8"].shape[0])
        if "initial_recurrent_hidden" not in result:
            result["initial_recurrent_hidden"] = torch.zeros(
                (static_count, hidden_dim), dtype=torch.float32
            )
        for name in (
            *required,
            "initial_recurrent_hidden",
        ):
            existing = result.get(name)
            incoming = corrective.get(name)
            if not isinstance(existing, torch.Tensor) or not isinstance(
                incoming, torch.Tensor
            ):
                raise ValueError(f"cannot merge corrective BC field {name}")
            if tuple(existing.shape[1:]) != tuple(incoming.shape[1:]):
                raise ValueError(
                    f"corrective BC field shape mismatch for {name}: "
                    f"{tuple(existing.shape)} vs {tuple(incoming.shape)}"
                )
            result[name] = torch.cat((existing, incoming), dim=0)
        # Offline physical supervision has no synthetic equivalent for a live
        # corrective replay row.  Preserve tensor alignment with explicit
        # false masks; the live teacher state remains available to infer
        # measured contact for corrective rows.
        for name in (
            "contact_target",
            "stable_grasp_target",
            "future_safe_close_target",
            "future_safe_close_onset_target",
        ):
            existing = result.get(name)
            if not isinstance(existing, torch.Tensor):
                continue
            incoming = torch.zeros(
                (corrective_count, *existing.shape[1:]), dtype=existing.dtype
            )
            result[name] = torch.cat((existing, incoming), dim=0)
    result["burn_in_steps"] = int(burn_in_steps)
    return result


def random_shift_rgbd_sequences(
    rgbd_u8: torch.Tensor,
    *,
    pad: int = 4,
    return_principal_point_shift: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply one aligned integer random shift per sequence and camera.

    RGB, the two-byte metric depth encoding, and the depth-valid mask always
    receive the same crop.  The crop is also held constant across time so the
    augmentation cannot manufacture apparent camera or object motion for the
    GRU.  Augmentation is a learner-only operation; rollout and deployment
    observations are unchanged.
    """

    if rgbd_u8.ndim != 6 or tuple(rgbd_u8.shape[2:]) != G2_VISUAL_CAMERA_SHAPE:
        raise ValueError("RGB-D sequence must be [B,T,2,6,48,64]")
    if rgbd_u8.dtype != torch.uint8:
        raise ValueError("RGB-D random shift requires packed uint8 input")
    if pad < 0:
        raise ValueError("random-shift padding cannot be negative")
    batch, steps, cameras, channels, height, width = rgbd_u8.shape
    if pad == 0:
        principal_point_shift = torch.zeros(
            (batch, steps, cameras, 2),
            dtype=torch.float32,
            device=rgbd_u8.device,
        )
        return (
            (rgbd_u8, principal_point_shift)
            if return_principal_point_shift
            else rgbd_u8
        )
    # Keep the time dimension next to channels while applying a single crop
    # offset to each (batch, camera) pair.
    packed = rgbd_u8.permute(0, 2, 1, 3, 4, 5).reshape(
        batch * cameras, steps * channels, height, width
    )
    padded = F.pad(packed, (pad, pad, pad, pad), mode="replicate")
    windows = padded.unfold(2, height, 1).unfold(3, width, 1)
    offsets = torch.randint(
        0, 2 * pad + 1, (batch * cameras, 2), device=rgbd_u8.device
    )
    selected = windows[
        torch.arange(batch * cameras, device=rgbd_u8.device),
        :,
        offsets[:, 0],
        offsets[:, 1],
    ]
    shifted = selected.reshape(
        batch, cameras, steps, channels, height, width
    ).permute(0, 2, 1, 3, 4, 5).contiguous()
    # Padding followed by a crop at (offset_v, offset_u) moves the nominal
    # principal point by (pad-offset_u, pad-offset_v).  Focal length is
    # unchanged, so this is the exact K -> K' update for this augmentation.
    principal_point_shift = torch.stack(
        (
            float(pad) - offsets[:, 1].to(torch.float32),
            float(pad) - offsets[:, 0].to(torch.float32),
        ),
        dim=-1,
    ).reshape(batch, cameras, 2)
    principal_point_shift = principal_point_shift[:, None].expand(
        batch, steps, cameras, 2
    )
    return (
        (shifted, principal_point_shift)
        if return_principal_point_shift
        else shifted
    )


class _CameraRefinementStage(nn.Module):
    """One convolutional stage with a shape-safe residual connection."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.activation = nn.SiLU()
        self.residual = input_channels == output_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        refined = self.activation(self.convolution(value))
        return value + refined if self.residual else refined


class _ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.normalization1 = nn.BatchNorm2d(output_channels)
        self.convolution2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.normalization2 = nn.BatchNorm2d(output_channels)
        self.projection = (
            nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
            if stride != 1 or input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.projection(value)
        value = F.relu(self.normalization1(self.convolution1(value)), inplace=True)
        value = self.normalization2(self.convolution2(value))
        return F.relu(value + residual, inplace=True)


class _ResNet18CameraEncoder(nn.Module):
    """ResNet-18 image encoder with modality-specific input channels.

    The RGB branch receives three channels and the depth branch receives
    metric depth plus its validity mask.  Adaptive pooling preserves the
    existing latent interface for the 48x64 policy image contract.
    """

    def __init__(self, input_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._stage(64, 64, blocks=2, stride=1)
        self.stage2 = self._stage(64, 128, blocks=2, stride=2)
        self.stage3 = self._stage(128, 256, blocks=2, stride=2)
        self.stage4 = self._stage(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim)
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(
        input_channels: int, output_channels: int, *, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            _ResNetBasicBlock(input_channels, output_channels, stride=stride)
        ]
        layers.extend(
            _ResNetBasicBlock(output_channels, output_channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.stage3(value)
        value = self.stage4(value)
        return self.projection(self.pool(value))


class _ResNet13CameraEncoder(nn.Module):
    """Memory-conscious 13-layer residual RGB/depth encoder.

    It preserves the four independent camera/modality branches and their
    64-D fusion interface while using fewer blocks and narrower feature maps
    than ResNet-18. Shortcut projections are not counted in the layer name.
    """

    def __init__(self, input_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = _ResNet18CameraEncoder._stage(48, 48, blocks=1, stride=1)
        self.stage2 = _ResNet18CameraEncoder._stage(48, 96, blocks=2, stride=2)
        self.stage3 = _ResNet18CameraEncoder._stage(96, 160, blocks=2, stride=2)
        self.stage4 = _ResNet18CameraEncoder._stage(160, 256, blocks=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim)
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.stage3(value)
        value = self.stage4(value)
        return self.projection(self.pool(value))


def _camera_encoder(
    input_channels: int,
    latent_dim: int,
    encoder_channels: tuple[int, ...],
) -> nn.Module:
    channels = tuple(int(value) for value in encoder_channels)
    if channels not in G2_CAMERA_ENCODER_PROFILES.values():
        raise ValueError("camera encoder channels must use a registered profile")
    if channels == G2_CAMERA_ENCODER_PROFILES["resnet13"]:
        return _ResNet13CameraEncoder(input_channels, latent_dim)
    if channels == G2_CAMERA_ENCODER_PROFILES["resnet18"]:
        return _ResNet18CameraEncoder(input_channels, latent_dim)
    layers: list[nn.Module] = []
    previous = int(input_channels)
    for index, output in enumerate(channels):
        # Four stride-2 stages preserve the established 48x64 -> 3x4 feature
        # geometry.  Optional later stages enrich capacity without changing
        # the visual embedding or GRU interface.
        if index >= 4:
            layers.append(_CameraRefinementStage(previous, output))
            previous = output
            continue
        kernel = 5 if index == 0 else 3
        padding = 2 if index == 0 else 1
        layers.extend((nn.Conv2d(previous, output, kernel, 2, padding), nn.SiLU()))
        previous = output
    layers.extend(
        (
            nn.Flatten(),
            nn.Linear(channels[-1] * 3 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )
    )
    return nn.Sequential(*layers)


def _group_count(channels: int) -> int:
    """Return the largest small GroupNorm divisor for a channel count."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _GNSiLUConv(nn.Sequential):
    """Convolution used by the spatial head-depth ablations.

    GroupNorm is independent of vector-environment batch statistics.  The
    block intentionally has no MaxPool so image geometry is preserved.
    """

    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
        )


class _SpatialSoftmax2d(nn.Module):
    """Encode each feature channel by its differentiable expected (x, y)."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("spatial softmax input must be [N,C,H,W]")
        batch, channels, height, width = value.shape
        probability = torch.softmax(value.reshape(batch, channels, -1), dim=-1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=value.device, dtype=value.dtype),
            torch.linspace(-1.0, 1.0, width, device=value.device, dtype=value.dtype),
            indexing="ij",
        )
        expected_x = (probability * xx.flatten()).sum(dim=-1)
        expected_y = (probability * yy.flatten()).sum(dim=-1)
        return torch.cat((expected_x, expected_y), dim=-1)


class _GNResidualBlock(nn.Module):
    """ResNet basic block without BatchNorm or spatial pooling."""

    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _GNSiLUConv(input_channels, output_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(value)) + self.skip(value))


class _ModifiedHeadDepthResNet18(nn.Module):
    """Localization-preserving ResNet-18 for head depth.

    The standard 7x7/stride-2 stem, MaxPool, BatchNorm and global average
    pooling are intentionally absent.  Only stages 2 and 3 downsample, so a
    48x64 observation never becomes smaller than 12x16.
    """

    def __init__(self, input_channels: int, latent_dim: int) -> None:
        super().__init__()
        widths = (24, 32, 48, 64)
        self.stem = _GNSiLUConv(input_channels, widths[0])
        self.stages = nn.ModuleList(
            (
                self._stage(widths[0], widths[0], stride=1),
                self._stage(widths[0], widths[1], stride=2),
                self._stage(widths[1], widths[2], stride=2),
                self._stage(widths[2], widths[3], stride=1),
            )
        )
        self.spatial_softmax = _SpatialSoftmax2d()
        self.projection = nn.Sequential(
            nn.Linear(2 * widths[-1], latent_dim),
            nn.SiLU(),
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _stage(input_channels: int, output_channels: int, *, stride: int) -> nn.Sequential:
        return nn.Sequential(
            _GNResidualBlock(input_channels, output_channels, stride=stride),
            _GNResidualBlock(output_channels, output_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        for stage in self.stages:
            value = stage(value)
        return self.projection(self.spatial_softmax(value))


class _LiteDepthwiseGNSiLU(nn.Sequential):
    """Depthwise-separable spatial block used by the Lite U-Net/FPN."""

    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                input_channels,
                3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(input_channels), input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
        )


class _LiteHeadDepthUNetFPN(nn.Module):
    """Lightweight U-Net/FPN encoder with multi-scale spatial tokens."""

    def __init__(self, input_channels: int, latent_dim: int) -> None:
        super().__init__()
        high_channels, middle_channels, low_channels = 24, 40, 64
        self.pyramid_channels = (high_channels, middle_channels, low_channels)
        self.high = nn.Sequential(
            _GNSiLUConv(input_channels, high_channels),
            _LiteDepthwiseGNSiLU(high_channels, high_channels),
        )
        self.middle = nn.Sequential(
            _LiteDepthwiseGNSiLU(high_channels, middle_channels, stride=2),
            _LiteDepthwiseGNSiLU(middle_channels, middle_channels),
        )
        self.low = nn.Sequential(
            _LiteDepthwiseGNSiLU(middle_channels, low_channels, stride=2),
            _LiteDepthwiseGNSiLU(low_channels, low_channels),
        )
        self.low_to_middle = nn.Conv2d(low_channels, middle_channels, 1, bias=False)
        self.middle_fuse = _LiteDepthwiseGNSiLU(middle_channels, middle_channels)
        self.middle_to_high = nn.Conv2d(middle_channels, high_channels, 1, bias=False)
        self.high_fuse = _LiteDepthwiseGNSiLU(high_channels, high_channels)
        self.edge_head = nn.Conv2d(high_channels, 1, 1)
        self.spatial_softmax = _SpatialSoftmax2d()
        self.projection = nn.Sequential(
            nn.Linear(2 * sum(self.pyramid_channels), latent_dim),
            nn.SiLU(),
            nn.LayerNorm(latent_dim),
        )

    def _fused_features(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high = self.high(value)
        middle = self.middle(high)
        low = self.low(middle)
        fused_middle = self.middle_fuse(
            middle
            + F.interpolate(
                self.low_to_middle(low),
                size=middle.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        fused_high = self.high_fuse(
            high
            + F.interpolate(
                self.middle_to_high(fused_middle),
                size=high.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        return fused_high, fused_middle, low

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        fused_high, fused_middle, low = self._fused_features(value)
        spatial = torch.cat(
            (
                self.spatial_softmax(fused_high),
                self.spatial_softmax(fused_middle),
                self.spatial_softmax(low),
            ),
            dim=-1,
        )
        return self.projection(spatial)

    def predict_edges(self, value: torch.Tensor) -> torch.Tensor:
        fused_high, _, _ = self._fused_features(value)
        return self.edge_head(fused_high)


class _HeadDepthSpatialEncoder(nn.Module):
    """Head-depth encoder with at most 4x spatial downsampling.

    ``hrnet_fpn_spatial_softmax`` keeps a high-resolution branch and fuses a
    two-level pyramid before spatial softmax.  The coordinate profiles use
    ``[D_normalized, M_valid, X_pixel, Y_pixel]`` and GroupNorm/SiLU.  The
    ``*_unet_aux`` profile additionally owns a train-only depth decoder.
    """

    def __init__(self, profile: str, latent_dim: int) -> None:
        super().__init__()
        if profile not in G2_HEAD_DEPTH_PROFILES or profile == "legacy":
            raise ValueError("specialized head-depth encoder requires a non-legacy profile")
        self.profile = profile
        self.uses_coordinate_input = profile != "hrnet_fpn_spatial_softmax"
        input_channels = 4 if self.uses_coordinate_input else 2
        self.specialized_backend = None
        if profile == "modified_resnet18_spatial_softmax":
            self.specialized_backend = _ModifiedHeadDepthResNet18(
                input_channels, latent_dim
            )
        elif profile == "lite_unet_fpn_spatial_softmax":
            self.specialized_backend = _LiteHeadDepthUNetFPN(
                input_channels, latent_dim
            )
        if self.specialized_backend is not None:
            self.pyramid_channels = self.specialized_backend.pyramid_channels if hasattr(
                self.specialized_backend, "pyramid_channels"
            ) else (24, 32, 48, 64)
            self.decoder = None
            return
        # Phase-1 keeps the full 48x64 spatial grid but narrows channel width.
        # This preserves small-object localization while reducing the multiple
        # retained online/demo autograd graphs that previously exhausted VRAM.
        if profile == "hrnet_fpn_spatial_softmax":
            high_channels, middle_channels, low_channels = 24, 48, 72
        else:
            high_channels, middle_channels, low_channels = 32, 64, 96
        self.pyramid_channels = (high_channels, middle_channels, low_channels)
        self.high = nn.Sequential(
            _GNSiLUConv(input_channels, high_channels),
            _GNSiLUConv(high_channels, high_channels),
        )
        self.middle = nn.Sequential(
            _GNSiLUConv(high_channels, middle_channels, stride=2),
            _GNSiLUConv(middle_channels, middle_channels),
        )
        self.low = nn.Sequential(
            _GNSiLUConv(middle_channels, low_channels, stride=2),
            _GNSiLUConv(low_channels, low_channels),
        )
        if profile == "hrnet_fpn_spatial_softmax":
            self.low_to_middle = nn.Conv2d(
                low_channels, middle_channels, 1, bias=False
            )
            self.middle_to_high = nn.Conv2d(
                middle_channels, high_channels, 1, bias=False
            )
            spatial_channels = high_channels
        else:
            self.coordinate_refine = _GNSiLUConv(low_channels, low_channels)
            spatial_channels = low_channels
        self.spatial_softmax = _SpatialSoftmax2d()
        self.projection = nn.Sequential(
            nn.Linear(2 * spatial_channels, latent_dim),
            nn.SiLU(),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = None
        if profile == "coord_gn_silu_unet_aux":
            self.decoder = nn.Sequential(
                _GNSiLUConv(low_channels + middle_channels, middle_channels),
                _GNSiLUConv(middle_channels, middle_channels),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                _GNSiLUConv(middle_channels + high_channels, high_channels),
                _GNSiLUConv(high_channels, high_channels),
                nn.Conv2d(high_channels, 1, 1),
                nn.Sigmoid(),
            )

    @staticmethod
    def _coordinate_channels(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = depth.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=depth.device, dtype=depth.dtype),
            torch.linspace(-1.0, 1.0, width, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        return (
            xx.expand(depth.shape[0], 1, height, width),
            yy.expand(depth.shape[0], 1, height, width),
        )

    def _input(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Invalid samples are zero-filled only in conjunction with an explicit
        # validity channel; the network can never confuse a hole with a valid
        # near-zero range measurement.
        masked_depth = depth * valid
        if not self.uses_coordinate_input:
            return torch.cat((masked_depth, valid), dim=1)
        xx, yy = self._coordinate_channels(depth)
        return torch.cat((masked_depth, valid, xx, yy), dim=1)

    def encode(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.specialized_backend is not None:
            packed = self._input(depth, valid)
            # Recurrent SAC flattens batch x sequence (64 x 13 learning
            # frames in the production profile).  U-Net/FPN skip features
            # are spatially large even though the model has few parameters.
            # Checkpointed chunks preserve the exact batch-independent
            # GroupNorm computation while avoiding simultaneous retention of
            # every high-resolution activation for both shifted and original
            # observation graphs.
            if self.training and torch.is_grad_enabled() and packed.shape[0] > 128:
                return torch.cat(
                    tuple(
                        checkpoint(
                            self.specialized_backend,
                            packed[start : start + 128],
                            use_reentrant=False,
                        )
                        for start in range(0, packed.shape[0], 128)
                    ),
                    dim=0,
                )
            if not torch.is_grad_enabled() and packed.shape[0] > 128:
                # Critic-only warm-up needs inference features for the target
                # action but no actor gradients.  Chunking is mathematically
                # invariant for this GroupNorm-only backend and avoids a
                # single 64 x 13-frame high-resolution activation peak.
                return torch.cat(
                    tuple(
                        self.specialized_backend(packed[start : start + 128])
                        for start in range(0, packed.shape[0], 128)
                    ),
                    dim=0,
                )
            return self.specialized_backend(packed)
        high = self.high(self._input(depth, valid))
        middle = self.middle(high)
        low = self.low(middle)
        if self.profile == "hrnet_fpn_spatial_softmax":
            fused_middle = middle + F.interpolate(
                self.low_to_middle(low), size=middle.shape[-2:], mode="bilinear", align_corners=False
            )
            spatial = high + F.interpolate(
                self.middle_to_high(fused_middle), size=high.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            spatial = self.coordinate_refine(low)
        return self.projection(self.spatial_softmax(spatial))

    def reconstruct(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("head-depth reconstruction is available only for the U-Net auxiliary profile")
        high = self.high(self._input(depth, valid))
        middle = self.middle(high)
        low = self.low(middle)
        value = F.interpolate(low, size=middle.shape[-2:], mode="bilinear", align_corners=False)
        # The sequential decoder contains concatenation boundaries, so execute
        # its blocks explicitly while retaining a compact registered module.
        value = self.decoder[0](torch.cat((value, middle), dim=1))
        value = self.decoder[1](value)
        value = self.decoder[2](value)
        value = self.decoder[3](torch.cat((value, high), dim=1))
        value = self.decoder[4](value)
        value = self.decoder[5](value)
        return self.decoder[6](value)

    def predict_edges(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.profile != "lite_unet_fpn_spatial_softmax":
            raise RuntimeError("head-depth edge prediction requires the Lite U-Net/FPN profile")
        packed = self._input(depth, valid)
        backend = self.specialized_backend
        assert backend is not None
        if self.training and torch.is_grad_enabled() and packed.shape[0] > 128:
            return torch.cat(
                tuple(
                    checkpoint(
                        backend.predict_edges,
                        packed[start : start + 128],
                        use_reentrant=False,
                    )
                    for start in range(0, packed.shape[0], 128)
                ),
                dim=0,
            )
        return backend.predict_edges(packed)


def masked_head_depth_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Validity-normalized L1 depth loss; invalid holes have zero authority."""

    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("depth prediction, target and validity mask must have identical shapes")
    weight = valid.to(dtype=prediction.dtype)
    return (weight * (prediction - target).abs()).sum() / (weight.sum() + float(epsilon))


def masked_head_depth_edge_loss(
    logits: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    edge_scale_normalized: float = 0.01,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise metric-depth discontinuities without turning holes into edges.

    Sobel magnitude is converted to a bounded soft edge target.  A pixel has
    authority only when its complete 3x3 neighborhood contains valid depth,
    preventing sensor holes and shifted-image padding from becoming false
    object boundaries.
    """

    if logits.shape != depth.shape or depth.shape != valid.shape:
        raise ValueError("edge logits, depth and validity mask must have identical shapes")
    if edge_scale_normalized <= 0.0:
        raise ValueError("normalized depth edge scale must be positive")
    kernel_x = depth.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    masked = depth * valid
    gradient_x = F.conv2d(masked, kernel_x, padding=1)
    gradient_y = F.conv2d(masked, kernel_y, padding=1)
    # Depth is an observation target (no gradient required), so retain an
    # exact zero for flat regions instead of adding epsilon that would label
    # every valid pixel as a weak edge.
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
    target = 1.0 - torch.exp(-magnitude / float(edge_scale_normalized))
    neighborhood = F.conv2d(valid, torch.ones_like(kernel_x), padding=1)
    authority = (neighborhood >= 9.0 - 1.0e-6).to(dtype=logits.dtype)
    error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (authority * error).sum() / (authority.sum() + float(epsilon))
    target_mean = (authority * target).sum() / (authority.sum() + float(epsilon))
    return loss, target_mean


class _HeadCubeDetector(nn.Module):
    """Small-object heatmap detector used only by the fixed Head camera.

    The detector consumes RGB, but its dense feature map is never exposed to
    the actor.  Only geometry decoded from the heatmap and metric depth may
    cross the deployment boundary.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _GNSiLUConv(3, 24),
            _GNSiLUConv(24, 32),
            _GNSiLUConv(32, 32),
        )
        self.heatmap = nn.Conv2d(32, 1, 1)
        # A fresh untrained detector must fail closed instead of declaring the
        # whole image to be the cube mask.
        nn.init.zeros_(self.heatmap.weight)
        nn.init.constant_(self.heatmap.bias, -4.0)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.heatmap(self.backbone(rgb))


class _WristRGBDSpatialEncoder(nn.Module):
    """Eye-in-hand RGB-D encoder whose output preserves image location."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        # RGB, normalized metric depth, and an explicit validity bit.  The
        # Lite FPN contains SpatialSoftmax at all three scales and never uses
        # global average pooling.
        self.backend = _LiteHeadDepthUNetFPN(5, latent_dim)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        return self.backend(torch.cat((rgb, depth * valid, valid), dim=1))


class G2TaskRelevantVisualEncoder(nn.Module):
    """Detector-only Head geometry plus an eye-in-hand spatial feature.

    Raw Head RGB/depth features never reach SAC.  Head RGB predicts a cube
    heatmap; the heatmap and metric depth are back-projected into the robot
    root frame.  The Wrist RGB-D feature is distance/phase gated and both
    cameras fail closed on stale, invalid, or non-visible frames.
    """

    def __init__(
        self,
        *,
        latent_per_modality: int = 64,
        output_dim: int = 70,
        share_camera_encoder_weights: bool = False,
        encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
            "baseline_4layer"
        ],
        head_observation_mode: str = "cube_detector_depth_xyz",
        actor_cube_position_input_mode: str = "relative_grasp_xyz",
        head_depth_profile: str = "legacy",
        head_intrinsic_3x3: tuple[float, ...] | None = None,
        root_from_head_optical_4x4: tuple[float, ...] | None = None,
        wrist_fusion_enabled: bool = True,
        wrist_fusion_near_distance_m: float = 0.05,
        wrist_fusion_far_distance_m: float = 0.12,
        wrist_fusion_max_weight: float = 1.0,
        wrist_fusion_phase_gate_enabled: bool = True,
        cube_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06),
    ) -> None:
        super().__init__()
        self.latent_per_modality = int(latent_per_modality)
        self.output_dim = int(output_dim)
        self.share_camera_encoder_weights = bool(share_camera_encoder_weights)
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        if self.encoder_channels not in G2_CAMERA_ENCODER_PROFILES.values():
            raise ValueError("unknown camera encoder profile")
        self.encoder_profile = g2_camera_encoder_profile_name(self.encoder_channels)
        if head_observation_mode not in G2_HEAD_OBSERVATION_MODES:
            raise ValueError("unknown head observation mode")
        self.head_observation_mode = head_observation_mode
        if actor_cube_position_input_mode not in G2_ACTOR_CUBE_POSITION_INPUT_MODES:
            raise ValueError("unknown actor cube-position input mode")
        self.actor_cube_position_input_mode = actor_cube_position_input_mode
        if head_depth_profile not in G2_HEAD_DEPTH_PROFILES:
            raise ValueError("unknown head-depth profile")
        self.head_depth_profile = head_depth_profile
        if head_intrinsic_3x3 is not None and len(head_intrinsic_3x3) != 9:
            raise ValueError("head calibration intrinsic must contain 9 values")
        if root_from_head_optical_4x4 is not None and len(root_from_head_optical_4x4) != 16:
            raise ValueError("head root-from-optical transform must contain 16 values")
        intrinsic = (
            torch.tensor(
                ((50.0, 0.0, 31.5), (0.0, 50.0, 23.5), (0.0, 0.0, 1.0)),
                dtype=torch.float32,
            )
            if head_intrinsic_3x3 is None
            else torch.tensor(head_intrinsic_3x3, dtype=torch.float32).reshape(3, 3)
        )
        root_from_optical = torch.eye(4) if root_from_head_optical_4x4 is None else torch.tensor(
            root_from_head_optical_4x4, dtype=torch.float32
        ).reshape(4, 4)
        self.register_buffer("head_intrinsic_3x3", intrinsic)
        self.register_buffer("root_from_head_optical_4x4", root_from_optical)
        self.head_cube_detector = _HeadCubeDetector()
        self.head_geometry_encoder = nn.Sequential(
            nn.Linear(5, self.latent_per_modality),
            nn.SiLU(),
            nn.LayerNorm(self.latent_per_modality),
        )
        # Legacy centroid diagnostics remain callable, but this 8-D encoder is
        # not connected to the actor.  The canonical actor path above accepts
        # only the 5-D detected cube geometry.
        self.head_centroid_encoder = nn.Sequential(
            nn.Linear(8, self.latent_per_modality),
            nn.SiLU(),
            nn.LayerNorm(self.latent_per_modality),
        )
        self.head_depth_encoder = (
            None
            if self.head_depth_profile == "legacy"
            else _HeadDepthSpatialEncoder(self.head_depth_profile, self.latent_per_modality)
        )
        self.wrist_encoder = _WristRGBDSpatialEncoder(self.latent_per_modality)
        self.wrist_cube_offset_head = nn.Sequential(
            nn.Linear(self.latent_per_modality, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )
        # Learner-only cube-centric tokens. They are never concatenated into
        # the SAC actor input, so global Head localization and eye-in-hand
        # Wrist geometry remain camera-specific.
        self.head_crossview_projection = nn.Sequential(
            nn.Linear(self.latent_per_modality + 5, self.latent_per_modality),
            nn.SiLU(),
            nn.LayerNorm(self.latent_per_modality),
        )
        self.wrist_crossview_projection = nn.Sequential(
            nn.Linear(self.latent_per_modality + 5, self.latent_per_modality),
            nn.SiLU(),
            nn.LayerNorm(self.latent_per_modality),
        )
        # Deployable visual grasp-readiness estimate.  Its supervision may use
        # simulator geometry, but its inference input is strictly the same
        # Head-relative geometry and gated Wrist feature exposed to the actor.
        # The resulting continuous probability is appended to the actor
        # feature, so it reaches the GRU rather than acting as an external
        # binary training gate.
        self.grasp_ready_head = nn.Sequential(
            nn.Linear(5 + self.latent_per_modality, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.maximum_camera_frame_age_s = 0.12
        self.minimum_depth_valid_ratio = 0.02
        self.minimum_cube_pixels = 2
        self.wrist_fusion_enabled = bool(wrist_fusion_enabled)
        self.near_gate_distance_m = float(wrist_fusion_near_distance_m)
        self.far_gate_distance_m = float(wrist_fusion_far_distance_m)
        self.wrist_fusion_max_weight = float(wrist_fusion_max_weight)
        self.wrist_fusion_phase_gate_enabled = bool(wrist_fusion_phase_gate_enabled)
        self.cube_size_m = tuple(float(value) for value in cube_size_m)
        if not 0.0 < self.near_gate_distance_m < self.far_gate_distance_m:
            raise ValueError("wrist fusion distances must satisfy 0 < near < far")
        if not 0.0 <= self.wrist_fusion_max_weight <= 1.0:
            raise ValueError("wrist fusion maximum weight must be in [0,1]")
        if len(self.cube_size_m) != 3 or any(value <= 0.0 for value in self.cube_size_m):
            raise ValueError("cube size must contain three positive dimensions")
        self._last_perception: dict[str, torch.Tensor] = {}

    def encode_with_camera_tokens(
        self,
        rgbd_u8: torch.Tensor,
        principal_point_shift_px: torch.Tensor | None = None,
        goal_position_root_m: torch.Tensor | None = None,
        deployable_proprioception: torch.Tensor | None = None,
        grasp_center_position_root_m: torch.Tensor | None = None,
        phase_one_hot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(rgbd_u8.shape[1:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError(f"RGB-D must be [N,{G2_VISUAL_CAMERA_SHAPE}], got {tuple(rgbd_u8.shape)}")
        if principal_point_shift_px is None:
            principal_point_shift_px = torch.zeros(
                (rgbd_u8.shape[0], 2, 2),
                dtype=torch.float32,
                device=rgbd_u8.device,
            )
        if tuple(principal_point_shift_px.shape) != (rgbd_u8.shape[0], 2, 2):
            raise ValueError("principal-point shift must be [N,2,2] in (du,dv) pixels")
        count = int(rgbd_u8.shape[0])
        if deployable_proprioception is None:
            deployable_proprioception = torch.zeros(
                (count, G2_VISUAL_PROPRIO_DIM), device=rgbd_u8.device
            )
        if tuple(deployable_proprioception.shape) != (count, G2_VISUAL_PROPRIO_DIM):
            raise ValueError(f"deployable proprioception must be [N,{G2_VISUAL_PROPRIO_DIM}]")
        observation_slices = G2StudentObservationContract().slices
        if grasp_center_position_root_m is None:
            raise ValueError(
                "actor input requires live distal-pad midpoint FK; "
                "EE-origin and GT-cube fallbacks are forbidden"
            )
        if tuple(grasp_center_position_root_m.shape) != (count, 3):
            raise ValueError("distal-pad grasp center must be [N,3]")
        if phase_one_hot is None:
            phase_one_hot = torch.zeros(
                (count, G2_DEPLOYABLE_PHASE_DIM), device=rgbd_u8.device
            )
            phase_one_hot[:, 0] = 1.0
        if tuple(phase_one_hot.shape) != (count, G2_DEPLOYABLE_PHASE_DIM):
            raise ValueError(f"deployable phase must be [N,{G2_DEPLOYABLE_PHASE_DIM}]")
        if not bool(torch.isfinite(phase_one_hot).all()):
            raise ValueError("deployable phase contains non-finite values")
        frame_age = deployable_proprioception[
            :, observation_slices["camera_frame_age_s"]
        ]

        head = self._unpack_camera(rgbd_u8[:, 0])
        wrist = self._unpack_camera(rgbd_u8[:, 1])
        head_logits = self.head_cube_detector(head[0])
        head_geometry = self._head_geometry_from_heatmap(
            head_logits,
            head[1],
            head[2],
            principal_point_shift_px[:, 0],
            frame_age[:, 0],
        )
        wrist_feature = self.wrist_encoder(*wrist)
        wrist_object_descriptor = self._cube_roi_descriptor(*wrist)
        wrist_red_pixels = self._red_cube_pixel_count(wrist[0], wrist[2])
        wrist_depth_ratio = wrist[2].mean(dim=(1, 2, 3))
        wrist_valid = (
            (frame_age[:, 1] <= self.maximum_camera_frame_age_s)
            & (wrist_depth_ratio >= self.minimum_depth_valid_ratio)
            & (wrist_red_pixels >= self.minimum_cube_pixels)
        )
        wrist_feature = torch.where(
            wrist_valid[:, None], wrist_feature, torch.zeros_like(wrist_feature)
        )
        wrist_offset = 0.20 * self.wrist_cube_offset_head(wrist_feature)
        wrist_cube_root = grasp_center_position_root_m + wrist_offset

        head_xyz = head_geometry[:, :3]
        head_valid = head_geometry[:, 4] > 0.5
        estimated_for_distance = torch.where(
            head_valid[:, None], head_xyz, wrist_cube_root
        )
        grasp_distance = torch.linalg.vector_norm(
            estimated_for_distance - grasp_center_position_root_m, dim=-1
        )
        distance_coordinate = (
            (self.far_gate_distance_m - grasp_distance)
            / (self.far_gate_distance_m - self.near_gate_distance_m)
        ).clamp(0.0, 1.0)
        distance_gate = distance_coordinate.square() * (
            3.0 - 2.0 * distance_coordinate
        )
        phase_weight = torch.sum(
            phase_one_hot.to(distance_gate.dtype)
            * distance_gate.new_tensor((0.0, 0.65, 1.0, 1.0, 1.0)),
            dim=-1,
        ).clamp(0.0, 1.0)
        wrist_gate = (
            torch.maximum(distance_gate, phase_weight)
            if self.wrist_fusion_phase_gate_enabled
            else distance_gate
        )
        wrist_gate = wrist_gate.clamp_max(self.wrist_fusion_max_weight)
        wrist_gate = torch.where(wrist_valid, wrist_gate, torch.zeros_like(wrist_gate))
        if self.wrist_fusion_enabled:
            wrist_gate = torch.where(
                (~head_valid) & wrist_valid,
                torch.full_like(wrist_gate, self.wrist_fusion_max_weight),
                wrist_gate,
            )
        else:
            wrist_gate = torch.zeros_like(wrist_gate)
        # The production mode exposes the task-relative prior directly:
        # [Head predicted cube XYZ - live distal-pad midpoint FK,
        #  confidence, validity].  The legacy absolute Head XYZ mode remains
        # an explicit ablation.  Neither branch uses simulator cube GT.
        head_relative_geometry = torch.cat(
            (
                head_xyz - grasp_center_position_root_m,
                head_geometry[:, 3:5],
            ),
            dim=-1,
        )
        head_absolute_geometry = torch.cat(
            (head_xyz, head_geometry[:, 3:5]), dim=-1
        )
        actor_head_geometry = (
            head_relative_geometry
            if self.actor_cube_position_input_mode == "relative_grasp_xyz"
            else head_absolute_geometry
        )
        masked_head_geometry = torch.where(
            head_valid[:, None],
            actor_head_geometry,
            torch.zeros_like(actor_head_geometry),
        )
        actor_feature_without_ready = torch.cat(
            (masked_head_geometry, wrist_feature * wrist_gate[:, None]), dim=-1
        )
        grasp_ready_feature = torch.cat(
            (
                torch.where(
                    head_valid[:, None],
                    head_relative_geometry,
                    torch.zeros_like(head_relative_geometry),
                ),
                wrist_feature * wrist_gate[:, None],
            ),
            dim=-1,
        )
        grasp_ready_logit = self.grasp_ready_head(grasp_ready_feature)
        grasp_ready_probability = torch.sigmoid(grasp_ready_logit)
        # Visibility/depth validity is a hard runtime authority.  Preserve the
        # raw logit for supervised learning, while the actor receives zero
        # readiness when Head geometry is not deployable.
        deployable_grasp_ready_probability = torch.where(
            head_valid[:, None],
            grasp_ready_probability,
            torch.zeros_like(grasp_ready_probability),
        )
        actor_feature = torch.cat(
            (actor_feature_without_ready, deployable_grasp_ready_probability),
            dim=-1,
        )
        if actor_feature.shape[-1] != self.output_dim:
            raise RuntimeError("head-geometry/wrist feature dimension drift")
        if self.wrist_fusion_enabled:
            fused_cube_root = torch.where(
                head_valid[:, None] & wrist_valid[:, None],
                (1.0 - wrist_gate[:, None]) * head_xyz
                + wrist_gate[:, None] * wrist_cube_root,
                torch.where(head_valid[:, None], head_xyz, wrist_cube_root),
            )
            fused_valid = head_valid | wrist_valid
        else:
            # `--no-wrist-fusion` is a full production contract, not merely a
            # zero blend weight when both cameras happen to be valid.  The old
            # fallback still substituted Wrist on Head-invalid rows, so the
            # validation population for "fused" differed from Head and could
            # fail Fused<=Head despite fusion being disabled.
            fused_cube_root = head_xyz
            fused_valid = head_valid
        fused_cube_root = torch.where(
            fused_valid[:, None], fused_cube_root, torch.zeros_like(fused_cube_root)
        )
        head_token = self.head_geometry_encoder(masked_head_geometry)
        head_object_token = self.head_crossview_projection(
            torch.cat((head_token, masked_head_geometry), dim=-1)
        )
        wrist_object_token = self.wrist_crossview_projection(
            torch.cat((wrist_feature, wrist_object_descriptor), dim=-1)
        )
        head_object_token = torch.where(
            head_valid[:, None], head_object_token, torch.zeros_like(head_object_token)
        )
        wrist_object_token = torch.where(
            wrist_valid[:, None], wrist_object_token, torch.zeros_like(wrist_object_token)
        )
        camera_tokens = torch.stack((head_object_token, wrist_object_token), dim=1)
        self._last_perception = {
            "head_heatmap_logits": head_logits,
            "head_cube_mask": (
                (torch.sigmoid(head_logits[:, 0]) >= 0.5) & (head[2][:, 0] > 0.5)
            ),
            "head_cube_xyz_root_m": head_xyz,
            "head_cube_minus_grasp_center_m": head_xyz
            - grasp_center_position_root_m,
            "actor_head_geometry": masked_head_geometry,
            "actor_head_relative_geometry": torch.where(
                head_valid[:, None],
                head_relative_geometry,
                torch.zeros_like(head_relative_geometry),
            ),
            "actor_head_absolute_geometry": torch.where(
                head_valid[:, None],
                head_absolute_geometry,
                torch.zeros_like(head_absolute_geometry),
            ),
            "head_confidence": head_geometry[:, 3],
            "head_valid": head_valid,
            "head_cube_pixels": self._red_cube_pixel_count(head[0], head[2]),
            "head_heatmap_pixels": (torch.sigmoid(head_logits[:, 0]) >= 0.5).sum(dim=(1, 2)),
            "head_depth_valid_ratio": head[2].mean(dim=(1, 2, 3)),
            "wrist_cube_xyz_root_m": wrist_cube_root,
            "wrist_valid": wrist_valid,
            "wrist_cube_pixels": wrist_red_pixels,
            "wrist_depth_valid_ratio": wrist_depth_ratio,
            "wrist_gate_weight": wrist_gate,
            "grasp_center_cube_distance_m": grasp_distance,
            "fused_cube_xyz_root_m": fused_cube_root,
            "fused_valid": fused_valid,
            "camera_frame_age_s": frame_age,
            "grasp_center_position_root_m": grasp_center_position_root_m,
            "grasp_ready_logit": grasp_ready_logit[:, 0],
            "grasp_ready_probability": deployable_grasp_ready_probability[:, 0],
            "head_principal_point_shift_px": principal_point_shift_px[:, 0],
            "crossview_object_tokens": camera_tokens,
            "wrist_object_roi_descriptor": wrist_object_descriptor,
        }
        return actor_feature, camera_tokens

    @staticmethod
    def _unpack_camera(
        packed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = packed[:, :3].float().div(255.0)
        depth = (
            packed[:, 3].float() * 256.0 + packed[:, 4].float()
        ).div(65535.0).mul(2.0).unsqueeze(1)
        valid = packed[:, 5:6].float().div(255.0)
        return rgb, depth, valid

    @staticmethod
    def _red_cube_pixel_count(rgb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        red, green, blue = rgb.unbind(dim=1)
        mask = (
            (valid[:, 0] > 0.5)
            & (red > 0.30)
            & (red - torch.maximum(green, blue) > 0.10)
        )
        return mask.sum(dim=(1, 2))

    @staticmethod
    def _cube_roi_descriptor(
        rgb: torch.Tensor, depth_m: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Summarize the deployable red-cube ROI for learner-side InfoNCE."""

        red, green, blue = rgb.unbind(dim=1)
        weights = torch.relu(red - torch.maximum(green, blue)) * valid[:, 0]
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0e-6)
        height, width = weights.shape[-2:]
        vv, uu = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype),
            torch.linspace(-1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype),
            indexing="ij",
        )
        u = (weights * uu).sum(dim=(1, 2)) / denominator
        v = (weights * vv).sum(dim=(1, 2)) / denominator
        depth = (weights * depth_m[:, 0]).sum(dim=(1, 2)) / denominator
        pixel_fraction = (weights > 0.10).to(rgb.dtype).mean(dim=(1, 2))
        valid_ratio = valid.mean(dim=(1, 2, 3))
        return torch.stack((u, v, depth, pixel_fraction, valid_ratio), dim=-1)

    def _head_geometry_from_heatmap(
        self,
        logits: torch.Tensor,
        depth_m: torch.Tensor,
        depth_valid: torch.Tensor,
        principal_point_shift_px: torch.Tensor,
        frame_age_s: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logits[:, 0])
        valid = depth_valid[:, 0] > 0.5
        hard_mask = (probability >= 0.5) & valid
        pixel_count = hard_mask.sum(dim=(1, 2))
        depth_ratio = depth_valid.mean(dim=(1, 2, 3))
        visible = (
            (pixel_count >= self.minimum_cube_pixels)
            & (frame_age_s <= self.maximum_camera_frame_age_s)
            & (depth_ratio >= self.minimum_depth_valid_ratio)
        )
        weights = probability * valid.to(probability.dtype)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0e-6)
        height, width = probability.shape[-2:]
        vv, uu = torch.meshgrid(
            torch.arange(height, device=probability.device, dtype=probability.dtype),
            torch.arange(width, device=probability.device, dtype=probability.dtype),
            indexing="ij",
        )
        u = (weights * uu).sum(dim=(1, 2)) / denominator
        v = (weights * vv).sum(dim=(1, 2)) / denominator
        depth = (weights * depth_m[:, 0]).sum(dim=(1, 2)) / denominator
        intrinsic = self.head_intrinsic_3x3.to(device=depth.device, dtype=depth.dtype)
        cx = intrinsic[0, 2] + principal_point_shift_px[:, 0].to(depth.dtype)
        cy = intrinsic[1, 2] + principal_point_shift_px[:, 1].to(depth.dtype)
        optical = torch.stack(
            (
                (u - cx) * depth / intrinsic[0, 0],
                -(v - cy) * depth / intrinsic[1, 1],
                -depth,
                torch.ones_like(depth),
            ),
            dim=-1,
        )
        root = torch.matmul(
            optical,
            self.root_from_head_optical_4x4.to(
                device=depth.device, dtype=depth.dtype
            ).transpose(0, 1),
        )[:, :3]
        confidence = (
            probability.amax(dim=(1, 2))
            * (pixel_count.to(probability.dtype) / 8.0).clamp(0.0, 1.0)
            * depth_ratio.clamp(0.0, 1.0)
        )
        confidence = torch.where(visible, confidence, torch.zeros_like(confidence))
        return torch.cat(
            (root, confidence[:, None], visible.to(root.dtype)[:, None]), dim=-1
        )

    @property
    def last_perception(self) -> Mapping[str, torch.Tensor]:
        return self._last_perception

    def cube_localization_supervision(
        self,
        cube_position_root_m: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        *,
        perception: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Train/evaluate detector geometry without exposing GT to the actor.

        Ground-truth cube position is consumed only here, after deployable
        features have already been produced.  It cannot influence the
        distance/phase gate or the action path.
        """

        perception = self._last_perception if perception is None else perception
        required = (
            "head_heatmap_logits",
            "head_cube_xyz_root_m",
            "wrist_cube_xyz_root_m",
            "fused_cube_xyz_root_m",
            "wrist_valid",
            "fused_valid",
            "head_principal_point_shift_px",
        )
        if any(name not in perception for name in required):
            raise RuntimeError("cube localization supervision requires a preceding visual encode")
        logits = perception["head_heatmap_logits"]
        count = int(logits.shape[0])
        if tuple(cube_position_root_m.shape) != (count, 3):
            raise ValueError("cube supervision position must be [N,3]")
        if sample_weight is None:
            sample_weight = torch.ones(count, device=logits.device, dtype=logits.dtype)
        if tuple(sample_weight.shape) != (count,):
            raise ValueError("cube supervision weight must be [N]")
        sample_weight = sample_weight.to(device=logits.device, dtype=logits.dtype)

        root_from_optical = self.root_from_head_optical_4x4.to(
            device=logits.device, dtype=logits.dtype
        )
        optical_from_root = torch.linalg.inv(root_from_optical)
        homogeneous = torch.cat(
            (
                cube_position_root_m.to(logits.dtype),
                torch.ones((count, 1), device=logits.device, dtype=logits.dtype),
            ),
            dim=-1,
        )
        optical = homogeneous @ optical_from_root.transpose(0, 1)
        depth = -optical[:, 2]
        intrinsic = self.head_intrinsic_3x3.to(
            device=logits.device, dtype=logits.dtype
        )
        shift = perception["head_principal_point_shift_px"].to(logits.dtype)
        u = intrinsic[0, 0] * optical[:, 0] / depth.clamp_min(1.0e-6) + intrinsic[0, 2] + shift[:, 0]
        v = -intrinsic[1, 1] * optical[:, 1] / depth.clamp_min(1.0e-6) + intrinsic[1, 2] + shift[:, 1]
        height, width = logits.shape[-2:]
        target_visible = (
            (depth > 0.0)
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
        )
        yy, xx = torch.meshgrid(
            torch.arange(height, device=logits.device, dtype=logits.dtype),
            torch.arange(width, device=logits.device, dtype=logits.dtype),
            indexing="ij",
        )
        # The actual task cube occupies only a few pixels in the overview image. The
        # physical half-width projected through K sets a bounded Gaussian
        # radius instead of an arbitrary full-image segmentation target.
        sigma = (
            intrinsic[0, 0] * (0.5 * self.cube_size_m[0]) / depth.clamp_min(1.0e-3)
        ).clamp(1.0, 4.0)
        target = torch.exp(
            -(
                (xx[None] - u[:, None, None]).square()
                + (yy[None] - v[:, None, None]).square()
            )
            / (2.0 * sigma[:, None, None].square())
        )
        target = target * target_visible[:, None, None].to(target.dtype)
        probability = torch.sigmoid(logits[:, 0])
        positive_weight = 1.0 + 7.0 * target
        heatmap_error = F.binary_cross_entropy_with_logits(
            logits[:, 0], target, reduction="none"
        ) * positive_weight
        weighted_visible = sample_weight * target_visible.to(sample_weight.dtype)
        heatmap_loss = (
            heatmap_error.mean(dim=(1, 2)) * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1.0)
        head_error = torch.linalg.vector_norm(
            perception["head_cube_xyz_root_m"] - cube_position_root_m, dim=-1
        )
        head_position_loss = (
            F.smooth_l1_loss(
                perception["head_cube_xyz_root_m"],
                cube_position_root_m,
                reduction="none",
            ).mean(dim=-1)
            * weighted_visible
        ).sum() / weighted_visible.sum().clamp_min(1.0)
        wrist_valid = perception["wrist_valid"].to(sample_weight.dtype)
        wrist_weight = sample_weight * wrist_valid
        wrist_position_loss = (
            F.smooth_l1_loss(
                perception["wrist_cube_xyz_root_m"],
                cube_position_root_m,
                reduction="none",
            ).mean(dim=-1)
            * wrist_weight
        ).sum() / wrist_weight.sum().clamp_min(1.0)
        fused_error = torch.linalg.vector_norm(
            perception["fused_cube_xyz_root_m"] - cube_position_root_m, dim=-1
        )
        return heatmap_loss + head_position_loss, wrist_position_loss, {
            # Keep differentiable components available to the dedicated
            # pre-SAC vision learner.  Callers that only log diagnostics use
            # the detached public metrics below.
            "loss_head_heatmap": heatmap_loss,
            "loss_head_position": head_position_loss,
            "loss_wrist_position": wrist_position_loss,
            "head_position_error_m": head_error.detach(),
            "wrist_position_error_m": torch.linalg.vector_norm(
                perception["wrist_cube_xyz_root_m"] - cube_position_root_m, dim=-1
            ).detach(),
            "fused_position_error_m": fused_error.detach(),
            "target_visible": target_visible.detach(),
            "heatmap_peak_probability": probability.amax(dim=(1, 2)).detach(),
        }

    def _head_centroid_token(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        depth_valid: torch.Tensor,
        principal_point_shift_px: torch.Tensor,
        goal_position_root_m: torch.Tensor,
    ) -> torch.Tensor:
        """Compress the fixed overview camera to a robust cube/goal token.

        The exhibition cube is the only red task object.  RGB supplies a
        color-dominance mask; the authoritative range is the median metric
        depth inside that mask.  The goal is not a rendered object in this
        Lift scene, so its deployable task-command coordinate is appended
        explicitly instead of pretending that it was visually detected.
        """

        descriptor = self.head_centroid_observation(
            rgb,
            depth_m,
            depth_valid,
            principal_point_shift_px,
            goal_position_root_m,
        )
        return self.head_centroid_encoder(descriptor)

    def head_depth_reconstruction_loss(
        self,
        rgbd_u8: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        maximum_frames_per_chunk: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked normalized-depth L1 and metric MAE for profile 3.

        The decoder is called only by learner updates.  Actor rollout and
        deployment inference call :meth:`encode_with_camera_tokens` and do
        not execute this branch.
        """

        zero = next(self.parameters()).new_zeros(())
        if self.head_depth_encoder is None or self.head_depth_encoder.decoder is None:
            return zero, zero
        if tuple(rgbd_u8.shape[1:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError("RGB-D must be [N,2,6,48,64]")
        if maximum_frames_per_chunk <= 0:
            raise ValueError("maximum head-depth reconstruction chunk must be positive")
        if sample_mask is not None and tuple(sample_mask.shape) != (rgbd_u8.shape[0],):
            raise ValueError("head-depth sample mask must be [N]")

        # SAC flattens batch x sequence (64 x 16 = 1024 frames).  Running the
        # train-only U-Net on that full tensor retains a second high-resolution
        # graph beside the actor graph and can exceed 16 GiB VRAM.  Chunked
        # activation checkpointing preserves the exact global valid-pixel L1
        # objective while bounding live decoder activations.  It changes only
        # the compute/memory schedule, not sampling, model width or loss scale.
        numerator = zero
        denominator = zero
        for start in range(0, int(rgbd_u8.shape[0]), maximum_frames_per_chunk):
            stop = min(start + maximum_frames_per_chunk, int(rgbd_u8.shape[0]))
            packed = rgbd_u8[start:stop, 0]
            depth = (
                packed[:, 3].float() * 256.0 + packed[:, 4].float()
            ).div(65535.0).unsqueeze(1)
            valid = packed[:, 5:6].float().div(255.0)
            if sample_mask is not None:
                valid = valid * sample_mask[start:stop].to(
                    device=valid.device, dtype=valid.dtype
                )[:, None, None, None]
            prediction = checkpoint(
                self.head_depth_encoder.reconstruct,
                depth,
                valid,
                use_reentrant=False,
            )
            weight = valid.to(dtype=prediction.dtype)
            numerator = numerator + (weight * (prediction - depth).abs()).sum()
            denominator = denominator + weight.sum()
        loss = numerator / (denominator + 1.0e-6)
        return loss, loss.detach() * 2.0

    def head_depth_edge_loss(
        self,
        rgbd_u8: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked Sobel-edge supervision for the Phase-5 head depth."""

        zero = next(self.parameters()).new_zeros(())
        if (
            self.head_depth_encoder is None
            or self.head_depth_profile != "lite_unet_fpn_spatial_softmax"
        ):
            return zero, zero
        if tuple(rgbd_u8.shape[1:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError("RGB-D must be [N,2,6,48,64]")
        if sample_mask is not None and tuple(sample_mask.shape) != (rgbd_u8.shape[0],):
            raise ValueError("head-depth sample mask must be [N]")
        packed = rgbd_u8[:, 0]
        depth = (packed[:, 3].float() * 256.0 + packed[:, 4].float()).div(
            65535.0
        ).unsqueeze(1)
        valid = packed[:, 5:6].float().div(255.0)
        if sample_mask is not None:
            valid = valid * sample_mask.to(
                device=valid.device, dtype=valid.dtype
            ).view(-1, 1, 1, 1)
        logits = self.head_depth_encoder.predict_edges(depth, valid)
        return masked_head_depth_edge_loss(logits, depth, valid)

    def head_centroid_observation(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        depth_valid: torch.Tensor,
        principal_point_shift_px: torch.Tensor,
        goal_position_root_m: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[cube_root_xyz, goal_root_xyz, confidence, validity]``."""

        red, green, blue = rgb.unbind(dim=1)
        mask = (
            (depth_valid > 0.5)
            & (red > 0.30)
            & (red - torch.maximum(green, blue) > 0.10)
        )
        flat_mask = mask.flatten(1)
        count = flat_mask.sum(dim=1)
        valid_detection = count >= 2
        height, width = depth_m.shape[-2:]
        vv, uu = torch.meshgrid(
            torch.arange(height, device=depth_m.device, dtype=depth_m.dtype),
            torch.arange(width, device=depth_m.device, dtype=depth_m.dtype),
            indexing="ij",
        )
        weights = mask.to(depth_m.dtype)
        denominator = count.clamp_min(1).to(depth_m.dtype)
        u = (weights * uu).flatten(1).sum(dim=1) / denominator
        v = (weights * vv).flatten(1).sum(dim=1) / denominator
        masked_depth = torch.where(mask, depth_m, torch.full_like(depth_m, float("inf")))
        ordered_depth = masked_depth.flatten(1).sort(dim=1).values
        median_index = ((count.clamp_min(1) - 1) // 2).unsqueeze(1)
        depth = ordered_depth.gather(1, median_index).squeeze(1)
        depth = torch.where(valid_detection, depth, torch.zeros_like(depth))
        intrinsic = self.head_intrinsic_3x3.to(device=depth.device, dtype=depth.dtype)
        cx = intrinsic[0, 2] + principal_point_shift_px[:, 0].to(depth.dtype)
        cy = intrinsic[1, 2] + principal_point_shift_px[:, 1].to(depth.dtype)
        optical = torch.stack(
            (
                (u - cx) * depth / intrinsic[0, 0],
                -(v - cy) * depth / intrinsic[1, 1],
                -depth,
                torch.ones_like(depth),
            ),
            dim=-1,
        )
        cube_root = torch.matmul(
            optical,
            self.root_from_head_optical_4x4.to(
                device=depth.device, dtype=depth.dtype
            ).transpose(0, 1),
        )[:, :3]
        cube_root = torch.where(valid_detection[:, None], cube_root, torch.zeros_like(cube_root))
        confidence = (count.to(depth.dtype) / 16.0).clamp(0.0, 1.0).unsqueeze(1)
        return torch.cat(
            (
                cube_root,
                goal_position_root_m.to(dtype=depth.dtype),
                confidence,
                valid_detection.to(depth.dtype).unsqueeze(1),
            ),
            dim=-1,
        )

    def encode(
        self,
        rgbd_u8: torch.Tensor,
        principal_point_shift_px: torch.Tensor | None = None,
        deployable_proprioception: torch.Tensor | None = None,
        grasp_center_position_root_m: torch.Tensor | None = None,
        phase_one_hot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``encode`` is a vision-only diagnostic/auxiliary entry point.  It is
        # intentionally not used by the recurrent actor, whose stricter
        # ``encode_recurrent_features`` contract requires a real live/recorded
        # distal-pad midpoint.  Supplying an explicit zero here preserves
        # detector-only probes without creating an actor fallback.
        if grasp_center_position_root_m is None:
            grasp_center_position_root_m = torch.zeros(
                (rgbd_u8.shape[0], 3),
                dtype=torch.float32,
                device=rgbd_u8.device,
            )
        fused, _ = self.encode_with_camera_tokens(
            rgbd_u8,
            principal_point_shift_px,
            deployable_proprioception=deployable_proprioception,
            grasp_center_position_root_m=grasp_center_position_root_m,
            phase_one_hot=phase_one_hot,
        )
        return fused


class _VisualActor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(feature_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        if action_dim != G2_VISUAL_ACTION_DIM:
            raise ValueError("G2 visual actor requires the sealed 6+1 action contract")
        self.mean = nn.Linear(256, G2_VISUAL_ARM_ACTION_DIM)
        self.log_std = nn.Linear(256, G2_VISUAL_ARM_ACTION_DIM)
        self.gripper_logit = nn.Linear(256, 1)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, -2.0)
        nn.init.zeros_(self.gripper_logit.bias)

    def distribution(self, features: torch.Tensor):
        hidden = self.trunk(features)
        mean = self.mean(hidden)
        log_std = -5.0 + 3.5 * (torch.tanh(self.log_std(hidden)) + 1.0)
        return mean, log_std, self.gripper_logit(hidden)

    def sample_arm(self, features: torch.Tensor, deterministic: bool = False):
        mean, log_std, gripper_logit = self.distribution(features)
        pre_tanh = (
            mean
            if deterministic
            else torch.distributions.Normal(mean, log_std.exp()).rsample()
        )
        arm_action = torch.tanh(pre_tanh)
        arm_log_prob = squashed_gaussian_log_prob(pre_tanh, mean, log_std)
        return arm_action, arm_log_prob, torch.tanh(mean), log_std, gripper_logit

    def sample(self, features: torch.Tensor, deterministic: bool = False):
        arm, arm_log_prob, arm_mean, log_std, gripper_logit = self.sample_arm(
            features, deterministic
        )
        probability_open = torch.sigmoid(gripper_logit)
        if deterministic:
            open_sample = probability_open >= 0.5
        else:
            open_sample = torch.bernoulli(probability_open).to(torch.bool)
        gripper = torch.where(
            open_sample,
            torch.ones_like(probability_open),
            -torch.ones_like(probability_open),
        )
        selected_probability = torch.where(
            open_sample, probability_open, 1.0 - probability_open
        )
        log_probability = arm_log_prob + torch.log(
            selected_probability.clamp_min(1.0e-8)
        )
        mean_action = torch.cat(
            (
                arm_mean,
                torch.where(
                    probability_open >= 0.5,
                    torch.ones_like(probability_open),
                    -torch.ones_like(probability_open),
                ),
            ),
            dim=-1,
        )
        return torch.cat((arm, gripper), dim=-1), log_probability, mean_action, log_std


class _Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G2_VISUAL_PRIVILEGED_DIM + G2_VISUAL_ACTION_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, action), dim=-1))


@dataclass(frozen=True)
class G2VisualSACConfig:
    gamma: float = 0.9993
    tau: float = 0.005
    learning_rate: float = 3.0e-4
    initial_alpha: float = 0.1
    target_entropy: float | None = None
    relative_pose_weight: float = 0.2
    pose_consistency_weight: float = 0.1
    pose_consistency_rotation_weight: float = 0.1
    contact_weight: float = 0.1
    depth_validity_weight: float = 0.05
    share_camera_encoder_weights: bool = False
    camera_encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
        "baseline_4layer"
    ]
    gradient_clip: float = 5.0

    @property
    def resolved_target_entropy(self) -> float:
        return (
            -(float(G2_VISUAL_ARM_ACTION_DIM) + math.log(2.0))
            if self.target_entropy is None
            else float(self.target_entropy)
        )


class G2VisualAsymmetricSAC:
    """Feed-forward visual actor and privileged twin-Q critics.

    A recurrent policy is deliberately not used: camera capture is faster than
    the 50 Hz policy boundary, and single-transition replay remains valid.  If
    later occlusion metrics show temporal aliasing, a sequence replay contract
    must be introduced before GRU/LSTM is enabled.
    """

    def __init__(self, config: G2VisualSACConfig | None = None, *, device="cpu") -> None:
        self.config = config or G2VisualSACConfig()
        self.device = torch.device(device)
        self.visual = G2TaskRelevantVisualEncoder(
            share_camera_encoder_weights=self.config.share_camera_encoder_weights,
            encoder_channels=self.config.camera_encoder_channels,
        ).to(self.device)
        visual_dim = self.visual.output_dim
        self.actor = _VisualActor(visual_dim + G2_VISUAL_PROPRIO_DIM, G2_VISUAL_ACTION_DIM).to(self.device)
        self.relative_pose_head = nn.Sequential(
            nn.Linear(visual_dim, 64), nn.SiLU(), nn.Linear(64, 7)
        ).to(self.device)
        self.contact_head = nn.Sequential(
            nn.Linear(visual_dim, 32), nn.SiLU(), nn.Linear(32, 3)
        ).to(self.device)
        self.depth_validity_head = nn.Sequential(
            nn.Linear(visual_dim, 32), nn.SiLU(), nn.Linear(32, 2)
        ).to(self.device)
        self.q1, self.q2 = _Critic().to(self.device), _Critic().to(self.device)
        self.tq1, self.tq2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.tq1.requires_grad_(False); self.tq2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            list(self.visual.parameters())
            + list(self.actor.parameters())
            + list(self.relative_pose_head.parameters())
            + list(self.contact_head.parameters())
            + list(self.depth_validity_head.parameters()),
            lr=self.config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.config.learning_rate
        )
        self.log_alpha = torch.tensor(math.log(self.config.initial_alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _features(self, rgbd, proprio, grasp_center_position_root_m):
        return torch.cat(
            (
                self.visual.encode(
                    rgbd,
                    deployable_proprioception=proprio,
                    grasp_center_position_root_m=grasp_center_position_root_m,
                ),
                proprio,
            ),
            dim=-1,
        )

    @staticmethod
    def _hybrid_expectation(
        arm_action: torch.Tensor,
        arm_log_probability: torch.Tensor,
        gripper_logit: torch.Tensor,
        q1,
        q2,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact binary-gripper expectation for Q and log pi."""

        probability_open = torch.sigmoid(gripper_logit)
        closed_action = torch.cat(
            (arm_action, -torch.ones_like(probability_open)), dim=-1
        )
        open_action = torch.cat(
            (arm_action, torch.ones_like(probability_open)), dim=-1
        )
        closed_q = torch.minimum(q1(state, closed_action), q2(state, closed_action))
        open_q = torch.minimum(q1(state, open_action), q2(state, open_action))
        expected_q = (1.0 - probability_open) * closed_q + probability_open * open_q
        binary_expected_log_probability = (
            (1.0 - probability_open)
            * torch.log((1.0 - probability_open).clamp_min(1.0e-8))
            + probability_open * torch.log(probability_open.clamp_min(1.0e-8))
        )
        return (
            expected_q,
            arm_log_probability + binary_expected_log_probability,
            probability_open,
        )

    @torch.no_grad()
    def select_actions(
        self,
        rgbd_u8,
        proprio,
        grasp_center_position_root_m,
        *,
        deterministic=False,
    ) -> np.ndarray:
        rgbd = torch.as_tensor(rgbd_u8, dtype=torch.uint8, device=self.device)
        prop = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        grasp_center = torch.as_tensor(
            grasp_center_position_root_m, dtype=torch.float32, device=self.device
        )
        action, _, mean, _ = self.actor.sample(
            self._features(rgbd, prop, grasp_center), deterministic
        )
        return (mean if deterministic else action).cpu().numpy()

    def update(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        t = lambda name, dtype=torch.float32: torch.as_tensor(batch[name], dtype=dtype, device=self.device)
        rgbd, next_rgbd = t("rgbd", torch.uint8), t("next_rgbd", torch.uint8)
        prop, next_prop = t("proprio"), t("next_proprio")
        grasp_center = t("grasp_center_position_root_m")
        next_grasp_center = t("next_grasp_center_position_root_m")
        state, next_state = t("privileged"), t("next_privileged")
        action, reward, terminated = t("actions"), t("rewards"), t("terminated")
        with torch.no_grad():
            next_features = self._features(
                next_rgbd, next_prop, next_grasp_center
            )
            next_arm, next_arm_logp, _, _, next_gripper_logit = (
                self.actor.sample_arm(next_features)
            )
            target_q, next_logp, _ = self._hybrid_expectation(
                next_arm,
                next_arm_logp,
                next_gripper_logit,
                self.tq1,
                self.tq2,
                next_state,
            )
            target = reward + self.config.gamma * (1.0 - terminated) * (
                target_q - self.alpha.detach() * next_logp
            )
        q1, q2 = self.q1(state, action), self.q2(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.config.gradient_clip)
        self.critic_optimizer.step()

        features = self._features(rgbd, prop, grasp_center)
        sampled_arm, arm_logp, _, log_std, gripper_logit = self.actor.sample_arm(
            features
        )
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(False)
        expected_q, logp, probability_open = self._hybrid_expectation(
            sampled_arm, arm_logp, gripper_logit, self.q1, self.q2, state
        )
        actor_loss = (self.alpha.detach() * logp - expected_q).mean()
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(True)
        visual_latent = features[:, : self.visual.output_dim]
        # Privileged values supervise task-relevant heads only.  They are not
        # concatenated into the visual actor's deployment input.
        relative_target = relative_pose_target_from_teacher_state(state)
        relative_prediction = self.relative_pose_head(visual_latent)
        relative_pose_loss = relative_pose_auxiliary_loss(relative_prediction, relative_target)
        contact_slice = G2TeacherObservationContract().slices["bilateral_contact_features"]
        contact_target = state[:, contact_slice].clamp(0.0, 1.0)
        contact_logits = self.contact_head(visual_latent)
        contact_loss = F.binary_cross_entropy_with_logits(contact_logits, contact_target)
        depth_validity_target = rgbd[:, :, 5].float().div(255.0).mean(dim=(-1, -2))
        depth_validity_logits = self.depth_validity_head(visual_latent)
        depth_validity_loss = F.binary_cross_entropy_with_logits(
            depth_validity_logits, depth_validity_target
        )
        visual_loss = (
            self.config.relative_pose_weight * relative_pose_loss
            + self.config.contact_weight * contact_loss
            + self.config.depth_validity_weight * depth_validity_loss
        )
        total_actor_loss = actor_loss + visual_loss
        self.actor_optimizer.zero_grad(set_to_none=True); total_actor_loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(
            list(self.visual.parameters())
            + list(self.actor.parameters())
            + list(self.relative_pose_head.parameters())
            + list(self.contact_head.parameters())
            + list(self.depth_validity_head.parameters()),
            self.config.gradient_clip,
        )
        self.actor_optimizer.step()

        # Match the repository SAC temperature contract exactly.  The default
        # target entropy is -action_dim; using +action_dim here would reverse
        # the configured entropy target and make visual/teacher SAC diverge.
        alpha_loss = -(
            self.log_alpha
            * (logp.detach() + self.config.resolved_target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_optimizer.step()
        polyak_update(self.q1, self.tq1, self.config.tau); polyak_update(self.q2, self.tq2, self.config.tau)
        self.update_count += 1
        return {
            "loss/actor": float(actor_loss.detach()), "loss/critic": float(critic_loss.detach()),
            "loss/visual_relative_pose": float(relative_pose_loss.detach()),
            "loss/visual_contact": float(contact_loss.detach()),
            "loss/visual_depth_validity": float(depth_validity_loss.detach()),
            "loss/actor_total": float(total_actor_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "visual/relative_position_error_m": float(
                torch.linalg.vector_norm(
                    relative_prediction.detach()[:, :3] - relative_target[:, :3], dim=-1
                ).mean()
            ),
            "visual/contact_probability_mean": float(torch.sigmoid(contact_logits.detach()).mean()),
            "q/q1_mean": float(q1.detach().mean()), "q/q2_mean": float(q2.detach().mean()),
            "q/disagreement_mean": float(torch.abs(q1.detach() - q2.detach()).mean()),
            "entropy/alpha": float(self.alpha.detach()), "entropy/log_std_mean": float(log_std.detach().mean()),
            "entropy/gripper_open_probability_mean": float(
                probability_open.detach().mean()
            ),
            "entropy/target": float(self.config.resolved_target_entropy),
            "gradient/actor_visual_norm": float(actor_norm), "gradient/critic_norm": float(critic_norm),
        }

    def state_dict(self) -> dict:
        return {
            "schema": G2_VISUAL_SAC_SCHEMA,
            "config": self.config.__dict__,
            "visual": self.visual.state_dict(), "actor": self.actor.state_dict(),
            "relative_pose_head": self.relative_pose_head.state_dict(),
            "contact_head": self.contact_head.state_dict(),
            "depth_validity_head": self.depth_validity_head.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "tq1": self.tq1.state_dict(), "tq2": self.tq2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }


class G2RecurrentVisualStudent(nn.Module):
    """Deployment-safe RGB-D GRU used for BC/distillation and DAgger.

    Sequences must never cross an episode boundary.  Simulator cube/contact/
    Privileged simulator phase state is never used.  A deployable phase
    one-hot derived from controller/contact observations is an explicit GRU
    input and resets with the episode.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        gru_num_layers: int = 1,
        value_loss_weight: float = 0.2,
        auxiliary_loss_weight: float = 0.2,
        pose_consistency_loss_weight: float = 0.1,
        pose_consistency_rotation_weight: float = 0.1,
        cross_camera_contrastive_loss_weight: float = 0.0,
        cross_camera_contrastive_temperature: float = 0.1,
        cross_camera_false_negative_position_threshold_m: float = 0.01,
        temporal_pose_residual_loss_weight: float = 0.05,
        temporal_pose_residual_rotation_weight: float = 0.1,
        contact_loss_weight: float = 0.1,
        stable_grasp_loss_weight: float = 0.1,
        slip_speed_loss_weight: float = 0.1,
        depth_validity_loss_weight: float = 0.05,
        failure_loss_weight: float = 0.1,
        head_cube_detector_loss_weight: float = 0.2,
        wrist_cube_position_loss_weight: float = 0.1,
        share_camera_encoder_weights: bool = False,
        camera_encoder_channels: tuple[int, ...] = G2_CAMERA_ENCODER_PROFILES[
            "baseline_4layer"
        ],
        head_observation_mode: str = "cube_detector_depth_xyz",
        actor_cube_position_input_mode: str = "relative_grasp_xyz",
        head_depth_profile: str = "legacy",
        head_intrinsic_3x3: tuple[float, ...] | None = None,
        root_from_head_optical_4x4: tuple[float, ...] | None = None,
        wrist_fusion_enabled: bool = True,
        wrist_fusion_near_distance_m: float = 0.05,
        wrist_fusion_far_distance_m: float = 0.12,
        wrist_fusion_max_weight: float = 1.0,
        wrist_fusion_phase_gate_enabled: bool = True,
        cube_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06),
        torso_control_enabled: bool = False,
        rotation_action_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.visual = G2TaskRelevantVisualEncoder(
            share_camera_encoder_weights=share_camera_encoder_weights,
            encoder_channels=camera_encoder_channels,
            head_observation_mode=head_observation_mode,
            actor_cube_position_input_mode=actor_cube_position_input_mode,
            head_depth_profile=head_depth_profile,
            head_intrinsic_3x3=head_intrinsic_3x3,
            root_from_head_optical_4x4=root_from_head_optical_4x4,
            wrist_fusion_enabled=wrist_fusion_enabled,
            wrist_fusion_near_distance_m=wrist_fusion_near_distance_m,
            wrist_fusion_far_distance_m=wrist_fusion_far_distance_m,
            wrist_fusion_max_weight=wrist_fusion_max_weight,
            wrist_fusion_phase_gate_enabled=wrist_fusion_phase_gate_enabled,
            cube_size_m=cube_size_m,
        )
        # No actor-side cross-camera attention is used. The returned 64-D
        # cube-ROI tokens are consumed only by the optional learner objective.
        self.cross_camera_projection = nn.Identity()
        self.state_encoder = nn.Sequential(
            nn.Linear(G2_VISUAL_PROPRIO_DIM, 64), nn.SiLU(), nn.LayerNorm(64)
        )
        self.hidden_dim = int(hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        if self.gru_num_layers != 1:
            raise ValueError("G2 recurrent visual policy requires exactly one GRU layer")
        self.value_loss_weight = float(value_loss_weight)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.pose_consistency_loss_weight = float(pose_consistency_loss_weight)
        self.pose_consistency_rotation_weight = float(pose_consistency_rotation_weight)
        self.cross_camera_contrastive_loss_weight = float(
            cross_camera_contrastive_loss_weight
        )
        self.cross_camera_contrastive_temperature = float(
            cross_camera_contrastive_temperature
        )
        self.cross_camera_false_negative_position_threshold_m = float(
            cross_camera_false_negative_position_threshold_m
        )
        self.temporal_pose_residual_loss_weight = float(
            temporal_pose_residual_loss_weight
        )
        self.temporal_pose_residual_rotation_weight = float(
            temporal_pose_residual_rotation_weight
        )
        self.contact_loss_weight = float(contact_loss_weight)
        self.stable_grasp_loss_weight = float(stable_grasp_loss_weight)
        self.slip_speed_loss_weight = float(slip_speed_loss_weight)
        self.depth_validity_loss_weight = float(depth_validity_loss_weight)
        self.failure_loss_weight = float(failure_loss_weight)
        self.head_cube_detector_loss_weight = float(
            head_cube_detector_loss_weight
        )
        self.wrist_cube_position_loss_weight = float(
            wrist_cube_position_loss_weight
        )
        if self.cross_camera_contrastive_loss_weight < 0.0:
            raise ValueError("cross-camera contrastive weight cannot be negative")
        if self.cross_camera_contrastive_temperature <= 0.0:
            raise ValueError("cross-camera contrastive temperature must be positive")
        if self.cross_camera_false_negative_position_threshold_m < 0.0:
            raise ValueError("cross-camera false-negative threshold cannot be negative")
        if self.temporal_pose_residual_loss_weight < 0.0:
            raise ValueError("temporal pose-residual weight cannot be negative")
        if self.temporal_pose_residual_rotation_weight < 0.0:
            raise ValueError("temporal pose-residual rotation weight cannot be negative")
        if self.head_cube_detector_loss_weight < 0.0:
            raise ValueError("head cube-detector loss weight cannot be negative")
        if self.wrist_cube_position_loss_weight < 0.0:
            raise ValueError("wrist cube-position loss weight cannot be negative")
        self.torso_control_enabled = bool(torso_control_enabled)
        self.rotation_action_enabled = bool(rotation_action_enabled)
        self.gru = nn.GRU(
            # The deployable phase is an explicit recurrent input as well as
            # an authority signal for the wrist-distance gate.  It is derived
            # from observable controller/contact state, never cube GT.
            input_size=self.visual.output_dim + 64 + G2_DEPLOYABLE_PHASE_DIM,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
        )
        self.arm_action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128), nn.SiLU(), nn.Linear(128, 6), nn.Tanh()
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Tanh()
        )
        # This temporal head is the runtime CLOSE_READY authority.  The visual
        # encoder also emits a frame-local advisory logit, but only this GRU
        # head incorporates recent approach dynamics.
        self.future_safe_close_ready_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.close_timing_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.contact_time_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self._last_future_safe_close_ready_logit: torch.Tensor | None = None
        self._last_close_timing_logit: torch.Tensor | None = None
        self._last_predicted_contact_steps: torch.Tensor | None = None
        self.torso_action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 5), nn.Tanh()
        )
        self.torso_gate_head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        if not self.torso_control_enabled:
            self.torso_action_head.requires_grad_(False)
            self.torso_gate_head.requires_grad_(False)
        self.value_head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.relative_pose_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 7)
        )
        self.contact_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(), nn.Linear(64, 3)
        )
        self.stable_grasp_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.slip_speed_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.depth_validity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), nn.SiLU(), nn.Linear(32, 2)
        )
        self.failure_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64), nn.SiLU(),
            nn.Linear(64, len(G2_STUDENT_FAILURE_CLASSES)),
        )

    def initial_hidden(self, batch: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            (self.gru_num_layers, int(batch), self.hidden_dim),
            device=parameter.device if device is None else device,
            dtype=dtype,
        )

    @staticmethod
    def validate_sequence_boundaries(
        episode_id: torch.Tensor,
        sequence_step: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if episode_id.ndim != 2 or sequence_step.shape != episode_id.shape:
            raise ValueError("episode_id and sequence_step must both be [B,T]")
        if padding_mask is None:
            padding_mask = torch.ones_like(episode_id, dtype=torch.bool)
        if padding_mask.shape != episode_id.shape:
            raise ValueError("padding_mask must be [B,T]")
        if sequence_lengths is not None:
            if tuple(sequence_lengths.shape) != (episode_id.shape[0],):
                raise ValueError("sequence_lengths must be [B]")
            expected = torch.arange(episode_id.shape[1], device=episode_id.device)[None, :]
            expected = expected < sequence_lengths[:, None]
            if not torch.equal(expected, padding_mask.to(torch.bool)):
                raise ValueError("sequence_lengths and padding_mask disagree")
        adjacent = padding_mask[:, 1:] & padding_mask[:, :-1]
        if bool(((episode_id[:, 1:] != episode_id[:, :-1]) & adjacent).any()):
            raise ValueError("student sequence crosses an episode boundary")
        if bool(((sequence_step[:, 1:] != sequence_step[:, :-1] + 1) & adjacent).any()):
            raise ValueError("student sequence is not temporally contiguous")
        return padding_mask.to(torch.bool)

    def forward_sequence(
        self,
        rgbd_u8: torch.Tensor,
        deployable_proprioception: torch.Tensor,
        hidden: torch.Tensor | None = None,
        hidden_reset_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        camera_principal_point_shift_px: torch.Tensor | None = None,
        grasp_center_position_root_m: torch.Tensor | None = None,
        phase_one_hot: torch.Tensor | None = None,
        *,
        return_camera_tokens: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoded = self.encode_recurrent_features(
            rgbd_u8,
            deployable_proprioception,
            hidden,
            hidden_reset_mask,
            padding_mask,
            camera_principal_point_shift_px,
            grasp_center_position_root_m,
            phase_one_hot,
            return_camera_tokens=return_camera_tokens,
        )
        if return_camera_tokens:
            recurrent, next_hidden, projected_camera_tokens = encoded
        else:
            recurrent, next_hidden = encoded
            projected_camera_tokens = None
        arm_action = self.arm_action_head(recurrent)
        if not self.rotation_action_enabled:
            # Preserve the canonical 7-D action/storage contract while making
            # the three rotational residual slots exact, non-trainable zeros.
            # Orientation may still be estimated by auxiliary heads for
            # perception/safety; it is never emitted as a control request.
            arm_action = torch.cat(
                (arm_action[..., :3], torch.zeros_like(arm_action[..., 3:6])),
                dim=-1,
            )
        gripper_action = self.gripper_head(recurrent)
        torso_proposal = self.torso_action_head(recurrent)
        torso_gate_probability = self.torso_gate_head(recurrent)
        if self.torso_control_enabled:
            torso_action = torso_proposal * torso_gate_probability
        else:
            torso_action = torch.zeros_like(torso_proposal)
            torso_gate_probability = torch.zeros_like(torso_gate_probability)
        result = {
            "arm_action": arm_action,
            "gripper_action": gripper_action,
            "action": torch.cat((arm_action, gripper_action), dim=-1),
            "torso_action": torso_action,
            "torso_action_proposal": torso_proposal,
            "torso_gate_probability": torso_gate_probability,
            "value": self.value_head(recurrent),
            "relative_pose_aux": self.relative_pose_head(recurrent),
            "contact_logits": self.contact_head(recurrent),
            "stable_grasp_logits": self.stable_grasp_head(recurrent),
            # Softplus keeps the physical regression non-negative.  The loss
            # below normalizes by the sealed 0.06 m/s stable-grasp threshold.
            "slip_speed_m_s": F.softplus(self.slip_speed_head(recurrent)) * 0.06,
            "depth_validity_logits": self.depth_validity_head(recurrent),
            "failure_logits": self.failure_head(recurrent),
        }
        temporal_ready_logit = self.future_safe_close_ready_head(recurrent)
        close_timing_logit = self.close_timing_head(recurrent)
        predicted_contact_steps = F.softplus(self.contact_time_head(recurrent))
        self._last_future_safe_close_ready_logit = temporal_ready_logit.reshape(-1)
        self._last_close_timing_logit = close_timing_logit.reshape(-1)
        self._last_predicted_contact_steps = predicted_contact_steps.reshape(-1)
        result["future_safe_close_ready_logit"] = temporal_ready_logit
        result["grasp_ready_probability"] = torch.sigmoid(temporal_ready_logit)
        result["close_timing_logit"] = close_timing_logit
        result["close_timing_probability"] = torch.sigmoid(close_timing_logit)
        result["predicted_contact_steps"] = predicted_contact_steps
        perception = self.visual.last_perception
        if perception:
            batch, steps = rgbd_u8.shape[:2]
            for public_name, internal_name in (
                ("head_cube_heatmap_logits", "head_heatmap_logits"),
                ("head_cube_mask", "head_cube_mask"),
                ("head_cube_xyz_root_m", "head_cube_xyz_root_m"),
                ("head_cube_confidence", "head_confidence"),
                ("head_cube_visibility", "head_valid"),
                ("head_cube_pixel_count", "head_heatmap_pixels"),
                ("wrist_camera_valid", "wrist_valid"),
                ("wrist_gate_weight", "wrist_gate_weight"),
                ("fused_cube_xyz_root_m", "fused_cube_xyz_root_m"),
            ):
                value = perception[internal_name]
                result[public_name] = value.reshape(
                    batch, steps, *value.shape[1:]
                )
            result["head_cube_mask_probability"] = torch.sigmoid(
                result["head_cube_heatmap_logits"]
            )
        if projected_camera_tokens is not None:
            result["projected_camera_tokens"] = projected_camera_tokens
        return result, next_hidden

    @property
    def last_future_safe_close_ready_probability(self) -> torch.Tensor:
        if self._last_future_safe_close_ready_logit is None:
            raise RuntimeError("future-safe close-ready head has not run")
        return torch.sigmoid(self._last_future_safe_close_ready_logit)

    @property
    def last_close_timing_probability(self) -> torch.Tensor:
        if self._last_close_timing_logit is None:
            raise RuntimeError("close-timing head has not run")
        return torch.sigmoid(self._last_close_timing_logit)

    @property
    def last_predicted_contact_steps(self) -> torch.Tensor:
        if self._last_predicted_contact_steps is None:
            raise RuntimeError("contact-time head has not run")
        return self._last_predicted_contact_steps

    def encode_recurrent_features(
        self,
        rgbd_u8: torch.Tensor,
        deployable_proprioception: torch.Tensor,
        hidden: torch.Tensor | None = None,
        hidden_reset_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        camera_principal_point_shift_px: torch.Tensor | None = None,
        grasp_center_position_root_m: torch.Tensor | None = None,
        phase_one_hot: torch.Tensor | None = None,
        *,
        return_camera_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the canonical keyboard/teacher input with the shared GRU.

        Both online visual SAC and offline keyboard distillation call this
        exact method.  Keeping the recurrent backbone in one class prevents
        camera channel, proprioception ordering, and hidden-reset drift.
        """

        if rgbd_u8.ndim != 6 or tuple(rgbd_u8.shape[2:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError("student RGB-D must be [B,T,2,6,48,64]")
        batch, steps = rgbd_u8.shape[:2]
        if tuple(deployable_proprioception.shape) != (batch, steps, G2_VISUAL_PROPRIO_DIM):
            raise ValueError(f"student proprioception must be [B,T,{G2_VISUAL_PROPRIO_DIM}]")
        if camera_principal_point_shift_px is None:
            camera_principal_point_shift_px = torch.zeros(
                (batch, steps, 2, 2), dtype=torch.float32, device=rgbd_u8.device
            )
        if tuple(camera_principal_point_shift_px.shape) != (batch, steps, 2, 2):
            raise ValueError("camera principal-point shift must be [B,T,2,2]")
        if grasp_center_position_root_m is None:
            raise ValueError(
                "recurrent actor input requires distal-pad midpoint FK [B,T,3]"
            )
        if tuple(grasp_center_position_root_m.shape) != (batch, steps, 3):
            raise ValueError("distal-pad grasp center sequence must be [B,T,3]")
        if phase_one_hot is None:
            phase_one_hot = torch.zeros(
                (batch, steps, G2_DEPLOYABLE_PHASE_DIM),
                dtype=torch.float32,
                device=rgbd_u8.device,
            )
            phase_one_hot[..., 0] = 1.0
        if tuple(phase_one_hot.shape) != (batch, steps, G2_DEPLOYABLE_PHASE_DIM):
            raise ValueError(
                f"deployable phase sequence must be [B,T,{G2_DEPLOYABLE_PHASE_DIM}]"
            )
        if not bool(torch.isfinite(phase_one_hot).all()) or not bool(
            torch.allclose(
                phase_one_hot.sum(dim=-1),
                torch.ones((batch, steps), device=phase_one_hot.device),
                atol=1.0e-5,
                rtol=0.0,
            )
        ):
            raise ValueError("deployable phase must be finite one-hot rows")
        visual, camera_tokens = self.visual.encode_with_camera_tokens(
            rgbd_u8.reshape(batch * steps, *G2_VISUAL_CAMERA_SHAPE),
            camera_principal_point_shift_px.reshape(batch * steps, 2, 2),
            deployable_proprioception=deployable_proprioception.reshape(
                batch * steps, G2_VISUAL_PROPRIO_DIM
            ),
            grasp_center_position_root_m=grasp_center_position_root_m.reshape(
                batch * steps, 3
            ),
            phase_one_hot=phase_one_hot.reshape(
                batch * steps, G2_DEPLOYABLE_PHASE_DIM
            ),
        )
        projected_camera_tokens = self.cross_camera_projection(camera_tokens).reshape(
            batch, steps, 2, -1
        )
        state = self.state_encoder(deployable_proprioception.reshape(batch * steps, -1))
        feature = torch.cat(
            (
                visual,
                state,
                phase_one_hot.reshape(batch * steps, G2_DEPLOYABLE_PHASE_DIM).to(
                    device=visual.device, dtype=visual.dtype
                ),
            ),
            dim=-1,
        ).reshape(batch, steps, -1)
        if hidden is None:
            hidden = self.initial_hidden(batch, device=feature.device, dtype=feature.dtype)
        if hidden_reset_mask is None:
            hidden_reset_mask = torch.zeros((batch, steps), dtype=torch.bool, device=feature.device)
            hidden_reset_mask[:, 0] = True
        if tuple(hidden_reset_mask.shape) != (batch, steps):
            raise ValueError("hidden_reset_mask must be [B,T]")
        if padding_mask is None:
            padding_mask = torch.ones((batch, steps), dtype=torch.bool, device=feature.device)
        if tuple(padding_mask.shape) != (batch, steps):
            raise ValueError("padding_mask must be [B,T]")
        padding_mask = padding_mask.to(device=feature.device, dtype=torch.bool)
        if bool((padding_mask[:, 1:] & ~padding_mask[:, :-1]).any()):
            raise ValueError("padding_mask must be a left-aligned valid prefix")
        if bool((hidden_reset_mask.to(torch.bool) & ~padding_mask).any()):
            raise ValueError("hidden reset cannot occur on a padded timestep")
        outputs = []
        next_hidden = hidden
        for step in range(steps):
            reset = hidden_reset_mask[:, step].to(device=feature.device, dtype=torch.bool)
            next_hidden = torch.where(
                reset.view(1, batch, 1), torch.zeros_like(next_hidden), next_hidden
            )
            proposed_output, proposed_hidden = self.gru(
                feature[:, step : step + 1], next_hidden
            )
            active = padding_mask[:, step]
            next_hidden = torch.where(
                active.view(1, batch, 1), proposed_hidden, next_hidden
            )
            outputs.append(
                torch.where(
                    active.view(batch, 1, 1),
                    proposed_output,
                    torch.zeros_like(proposed_output),
                )
            )
        recurrent = torch.cat(outputs, dim=1)
        if return_camera_tokens:
            return recurrent, next_hidden, projected_camera_tokens
        return recurrent, next_hidden

    def distillation_loss(
        self,
        *,
        rgbd_u8: torch.Tensor,
        pose_consistency_rgbd_u8: torch.Tensor | None = None,
        camera_principal_point_shift_px: torch.Tensor | None = None,
        deployable_proprioception: torch.Tensor,
        expert_action: torch.Tensor,
        expert_confidence: torch.Tensor,
        episode_id: torch.Tensor,
        sequence_step: torch.Tensor,
        teacher_value: torch.Tensor | None = None,
        teacher_value_valid: torch.Tensor | None = None,
        relative_pose_target: torch.Tensor | None = None,
        contact_target: torch.Tensor | None = None,
        stable_grasp_target: torch.Tensor | None = None,
        slip_speed_target_m_s: torch.Tensor | None = None,
        depth_validity_target: torch.Tensor | None = None,
        failure_target: torch.Tensor | None = None,
        failure_target_valid: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        hidden_reset_mask: torch.Tensor | None = None,
        burn_in_steps: int = 0,
        hidden: torch.Tensor | None = None,
        cube_position_root_m: torch.Tensor | None = None,
        grasp_center_position_root_m: torch.Tensor | None = None,
        phase_one_hot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if burn_in_steps < 0 or burn_in_steps >= rgbd_u8.shape[1]:
            raise ValueError("burn_in_steps must be in [0,T)")
        padding_mask = self.validate_sequence_boundaries(
            episode_id, sequence_step, padding_mask, sequence_lengths
        )
        output, next_hidden = self.forward_sequence(
            rgbd_u8,
            deployable_proprioception,
            hidden,
            hidden_reset_mask,
            padding_mask,
            camera_principal_point_shift_px,
            grasp_center_position_root_m,
            phase_one_hot,
            return_camera_tokens=True,
        )
        head_cube_detector = output["value"].new_zeros(())
        wrist_cube_position = output["value"].new_zeros(())
        if cube_position_root_m is not None:
            if tuple(cube_position_root_m.shape) != (*rgbd_u8.shape[:2], 3):
                raise ValueError("cube position label must be [B,T,3]")
            head_cube_detector, wrist_cube_position, _ = (
                self.visual.cube_localization_supervision(
                    cube_position_root_m.reshape(-1, 3),
                    padding_mask.reshape(-1).to(torch.float32),
                )
            )
        if expert_action.shape != output["action"].shape:
            raise ValueError("expert action target must be [B,T,7]")
        if expert_confidence.shape != output["value"].shape:
            raise ValueError("expert confidence must be [B,T,1]")
        loss_mask = padding_mask.unsqueeze(-1).to(expert_confidence.dtype)
        loss_mask[:, :burn_in_steps] = 0.0
        confidence = torch.clamp(expert_confidence, 0.0, 1.0) * loss_mask
        denominator = torch.clamp(confidence.sum(), min=1.0)
        effective_expert_action = expert_action
        if not self.rotation_action_enabled:
            effective_expert_action = torch.cat(
                (
                    expert_action[..., :3],
                    torch.zeros_like(expert_action[..., 3:6]),
                    expert_action[..., 6:7],
                ),
                dim=-1,
            )
            action_error = torch.cat(
                (
                    output["action"][..., :3]
                    - effective_expert_action[..., :3],
                    output["action"][..., 6:7]
                    - effective_expert_action[..., 6:7],
                ),
                dim=-1,
            )
        else:
            action_error = output["action"] - effective_expert_action
        imitation = (
            confidence * action_error.square().mean(dim=-1, keepdim=True)
        ).sum() / denominator
        value = output["value"].new_zeros(())
        if teacher_value is not None:
            if teacher_value.shape != output["value"].shape:
                raise ValueError("teacher value target must be [B,T,1]")
            if teacher_value_valid is None:
                teacher_value_valid = torch.ones_like(teacher_value, dtype=torch.bool)
            if teacher_value_valid.shape != teacher_value.shape:
                raise ValueError("teacher value validity must be [B,T,1]")
            value_weight = confidence * teacher_value_valid.to(confidence.dtype)
            value_denominator = torch.clamp(value_weight.sum(), min=1.0)
            value = (
                value_weight * (output["value"] - teacher_value).square()
            ).sum() / value_denominator
        relative_pose = output["value"].new_zeros(())
        if relative_pose_target is not None:
            if relative_pose_target.shape != output["relative_pose_aux"].shape:
                raise ValueError("relative pose target must be [B,T,7]")
            relative_error = relative_pose_auxiliary_loss(
                output["relative_pose_aux"], relative_pose_target, reduction="none"
            ).unsqueeze(-1)
            relative_pose = (loss_mask * relative_error).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        pose_consistency = output["value"].new_zeros(())
        if pose_consistency_rgbd_u8 is not None:
            if pose_consistency_rgbd_u8.shape != rgbd_u8.shape:
                raise ValueError("pose consistency RGB-D view must match RGB-D input")
            consistency_output, _ = self.forward_sequence(
                pose_consistency_rgbd_u8,
                deployable_proprioception,
                hidden,
                hidden_reset_mask,
                padding_mask,
                None,
                grasp_center_position_root_m,
                phase_one_hot,
            )
            consistency_error = relative_pose_consistency_loss(
                consistency_output["relative_pose_aux"],
                output["relative_pose_aux"],
                rotation_weight=self.pose_consistency_rotation_weight,
                reduction="none",
            ).unsqueeze(-1)
            pose_consistency = (
                loss_mask * consistency_error
            ).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        cross_camera_contrastive = output["value"].new_zeros(())
        temporal_pose_residual = output["value"].new_zeros(())
        if relative_pose_target is not None:
            if self.cross_camera_contrastive_loss_weight > 0.0:
                perception = self.visual.last_perception
                both_visible = (
                    perception["head_valid"] & perception["wrist_valid"]
                ).reshape_as(loss_mask[..., 0])
                cross_camera_contrastive = cross_camera_contrastive_loss(
                    output["projected_camera_tokens"],
                    loss_mask[..., 0].to(torch.bool) & both_visible,
                    relative_pose_target=relative_pose_target,
                    temperature=self.cross_camera_contrastive_temperature,
                    false_negative_position_threshold_m=(
                        self.cross_camera_false_negative_position_threshold_m
                    ),
                )
            temporal_pose_residual = temporal_pose_residual_loss(
                output["relative_pose_aux"],
                relative_pose_target,
                loss_mask[..., 0].to(torch.bool),
                rotation_weight=self.temporal_pose_residual_rotation_weight,
            )
        contact = output["value"].new_zeros(())
        if contact_target is not None:
            if contact_target.shape != output["contact_logits"].shape:
                raise ValueError("contact target must be [B,T,3]")
            contact_error = F.binary_cross_entropy_with_logits(
                output["contact_logits"], contact_target, reduction="none"
            ).mean(dim=-1, keepdim=True)
            contact = (loss_mask * contact_error).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        stable_grasp = output["value"].new_zeros(())
        if stable_grasp_target is not None:
            if stable_grasp_target.shape != output["stable_grasp_logits"].shape:
                raise ValueError("stable-grasp target must be [B,T,1]")
            stable_error = F.binary_cross_entropy_with_logits(
                output["stable_grasp_logits"], stable_grasp_target, reduction="none"
            )
            stable_grasp = (loss_mask * stable_error).sum() / torch.clamp(
                loss_mask.sum(), min=1.0
            )
        slip_speed = output["value"].new_zeros(())
        if slip_speed_target_m_s is not None:
            if slip_speed_target_m_s.shape != output["slip_speed_m_s"].shape:
                raise ValueError("slip-speed target must be [B,T,1]")
            if bool((slip_speed_target_m_s < 0.0).any()):
                raise ValueError("slip-speed target cannot be negative")
            slip_error = F.smooth_l1_loss(
                output["slip_speed_m_s"] / 0.06,
                slip_speed_target_m_s / 0.06,
                reduction="none",
            )
            slip_speed = (loss_mask * slip_error).sum() / torch.clamp(
                loss_mask.sum(), min=1.0
            )
        depth_validity = output["value"].new_zeros(())
        if depth_validity_target is not None:
            if depth_validity_target.shape != output["depth_validity_logits"].shape:
                raise ValueError("depth-validity target must be [B,T,2]")
            depth_error = F.binary_cross_entropy_with_logits(
                output["depth_validity_logits"], depth_validity_target, reduction="none"
            ).mean(dim=-1, keepdim=True)
            depth_validity = (loss_mask * depth_error).sum() / torch.clamp(loss_mask.sum(), min=1.0)
        failure = output["value"].new_zeros(())
        if failure_target is not None:
            if failure_target.shape != episode_id.shape:
                raise ValueError("failure target must be [B,T]")
            if failure_target_valid is None:
                failure_target_valid = torch.ones_like(
                    failure_target, dtype=torch.bool
                )
            if failure_target_valid.shape != episode_id.shape:
                raise ValueError("failure target validity must be [B,T]")
            failure_error = F.cross_entropy(
                output["failure_logits"].reshape(-1, len(G2_STUDENT_FAILURE_CLASSES)),
                failure_target.reshape(-1),
                reduction="none",
            ).reshape(*episode_id.shape, 1)
            failure_weight = loss_mask * failure_target_valid.unsqueeze(-1).to(
                loss_mask.dtype
            )
            failure = (failure_weight * failure_error).sum() / torch.clamp(
                failure_weight.sum(), min=1.0
            )
        auxiliary = (
            relative_pose
            + self.pose_consistency_loss_weight * pose_consistency
            + self.cross_camera_contrastive_loss_weight
            * cross_camera_contrastive
            + self.temporal_pose_residual_loss_weight * temporal_pose_residual
        )
        total = (
            imitation
            + self.value_loss_weight * value
            + self.auxiliary_loss_weight * relative_pose
            + self.pose_consistency_loss_weight * pose_consistency
            + self.cross_camera_contrastive_loss_weight
            * cross_camera_contrastive
            + self.temporal_pose_residual_loss_weight * temporal_pose_residual
            + self.contact_loss_weight * contact
            + self.stable_grasp_loss_weight * stable_grasp
            + self.slip_speed_loss_weight * slip_speed
            + self.depth_validity_loss_weight * depth_validity
            + self.failure_loss_weight * failure
            + self.head_cube_detector_loss_weight * head_cube_detector
            + self.wrist_cube_position_loss_weight * wrist_cube_position
        )
        return total, {
            "loss/student_imitation": imitation,
            "loss/student_value": value,
            "loss/student_auxiliary": auxiliary,
            "loss/student_relative_pose": relative_pose,
            "loss/student_pose_consistency": pose_consistency,
            "loss/student_cross_camera_contrastive": cross_camera_contrastive,
            "loss/student_temporal_pose_residual": temporal_pose_residual,
            "loss/student_contact": contact,
            "loss/student_stable_grasp": stable_grasp,
            "loss/student_slip_speed": slip_speed,
            "loss/student_depth_validity": depth_validity,
            "loss/student_failure": failure,
            "loss/student_head_cube_detector": head_cube_detector,
            "loss/student_wrist_cube_position": wrist_cube_position,
            "loss/student_total": total,
        }, next_hidden

@dataclass(frozen=True)
class G2RecurrentVisualSACConfig(G2VisualSACConfig):
    """SAC settings for the shared keyboard/teacher recurrent backbone."""

    hidden_dim: int = 256
    gru_num_layers: int = 1
    sequence_length: int = 16
    burn_in_steps: int = 4
    sequence_stride: int | None = None
    gradient_clip: float = 5.0
    random_shift_pad: int = 4
    demonstration_q_filter_temperature: float = 0.1
    student_head_distillation_weight: float = 0.1
    # Actor requests live in normalized [-1, 1] coordinates, while the
    # deployable proprioception carries the previously *applied* Cartesian
    # action.  Penalize only the portion of a new request that the audited
    # slew limiter cannot realize on the next policy step.  This keeps SAC's
    # action/replay coordinates unchanged and does not relax a safety limit.
    action_slew_excess_weight: float = 0.0
    maximum_arm_action_magnitude: float = 0.10
    arm_action_slew_per_policy_step: float = 0.02
    near_contact_auxiliary_distance_m: float = 0.10
    near_contact_relative_pose_multiplier: float = 4.0
    cross_camera_contrastive_weight: float = 0.0
    cross_camera_contrastive_temperature: float = 0.1
    cross_camera_false_negative_position_threshold_m: float = 0.01
    temporal_pose_residual_weight: float = 0.05
    temporal_pose_residual_rotation_weight: float = 0.1
    head_observation_mode: str = "cube_detector_depth_xyz"
    actor_cube_position_input_mode: str = "relative_grasp_xyz"
    head_depth_profile: str = "legacy"
    head_depth_reconstruction_weight: float = 0.0
    head_depth_edge_weight: float = 0.0
    head_cube_detector_weight: float = 0.2
    wrist_cube_position_weight: float = 0.1
    wrist_fusion_enabled: bool = True
    wrist_fusion_near_distance_m: float = 0.05
    wrist_fusion_far_distance_m: float = 0.12
    wrist_fusion_max_weight: float = 1.0
    wrist_fusion_phase_gate_enabled: bool = True
    cube_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06)
    vision_xyz_loss_normalization_m: float = 1.0
    vision_loss_gradient_diagnostic_interval: int = 50
    grasp_ready_auxiliary_weight: float = 0.2
    grasp_ready_probability_threshold: float = 0.5
    close_timing_auxiliary_weight: float = 0.1
    close_timing_probability_threshold: float = 0.5
    close_timing_support_steps: int = 6
    close_timing_temperature_steps: float = 2.5
    contact_time_auxiliary_weight: float = 0.05
    contact_time_normalization_steps: float = 64.0
    require_frozen_visual_for_behavior_clone: bool = False
    demonstration_arm_loss_weight: float = 1.0
    demonstration_gripper_loss_weight: float = 1.0
    # A close state spans many frames, while the causal open->close decision
    # normally occupies one row.  Weight that decision separately so BC does
    # not learn only to maintain a close that was initiated by the expert.
    demonstration_close_onset_loss_weight: float = 1.0
    # Window-level onset sampling does not imply row-level balance: a 16-step
    # recurrent window normally contains only one causal open->close row.  Use
    # this as a *minimum aggregate loss fraction* for those causal rows so the
    # surrounding maintain-open rows cannot erase the decision.  Zero keeps
    # the historical fixed-weight behavior.
    demonstration_close_onset_target_loss_fraction: float = 0.0
    # Aggregate row-level quota applied after sequence/window sampling.  This
    # is deliberately separate from both the close-window sampling fraction
    # and the per-onset BCE multiplier.
    demonstration_close_onset_row_fraction: float = 0.0
    deterministic_gripper_close_probability_threshold: float = 0.5
    demonstration_close_onset_support_steps: int = 0
    # Calibrated error-shell authority.  This is deliberately not a raw
    # cube-to-pad distance cap: the non-zero reference grasp offset is part of
    # the intended grasp geometry.
    grasp_ready_tolerance_m: float = 0.016
    head_intrinsic_3x3: tuple[float, ...] | None = None
    root_from_head_optical_4x4: tuple[float, ...] | None = None
    # Keep the on-disk/on-wire action shape [dx,dy,dz,rx,ry,rz,gripper], but
    # when disabled make rx/ry/rz exact zeros and exclude them from SAC
    # entropy and imitation losses.
    rotation_action_enabled: bool = False

    @property
    def resolved_target_entropy(self) -> float:
        continuous_dimensions = 6 if self.rotation_action_enabled else 3
        return (
            -(float(continuous_dimensions) + math.log(2.0))
            if self.target_entropy is None
            else float(self.target_entropy)
        )

    @property
    def resolved_sequence_stride(self) -> int:
        """Use sequence-minus-burn-in for the selected production or test contract."""

        return (
            self.sequence_length - self.burn_in_steps
            if self.sequence_stride is None
            else int(self.sequence_stride)
        )


class _RecurrentVisualActor(nn.Module):
    """Stochastic SAC head over the exact deployable recurrent model."""

    def __init__(self, config: G2RecurrentVisualSACConfig) -> None:
        super().__init__()
        self.policy = G2RecurrentVisualStudent(
            hidden_dim=config.hidden_dim,
            gru_num_layers=config.gru_num_layers,
            pose_consistency_loss_weight=config.pose_consistency_weight,
            pose_consistency_rotation_weight=config.pose_consistency_rotation_weight,
            cross_camera_contrastive_loss_weight=(
                config.cross_camera_contrastive_weight
            ),
            cross_camera_contrastive_temperature=(
                config.cross_camera_contrastive_temperature
            ),
            cross_camera_false_negative_position_threshold_m=(
                config.cross_camera_false_negative_position_threshold_m
            ),
            temporal_pose_residual_loss_weight=(
                config.temporal_pose_residual_weight
            ),
            temporal_pose_residual_rotation_weight=(
                config.temporal_pose_residual_rotation_weight
            ),
            share_camera_encoder_weights=config.share_camera_encoder_weights,
            camera_encoder_channels=config.camera_encoder_channels,
            head_observation_mode=config.head_observation_mode,
            actor_cube_position_input_mode=config.actor_cube_position_input_mode,
            head_depth_profile=config.head_depth_profile,
            head_intrinsic_3x3=config.head_intrinsic_3x3,
            root_from_head_optical_4x4=config.root_from_head_optical_4x4,
            wrist_fusion_enabled=config.wrist_fusion_enabled,
            wrist_fusion_near_distance_m=config.wrist_fusion_near_distance_m,
            wrist_fusion_far_distance_m=config.wrist_fusion_far_distance_m,
            wrist_fusion_max_weight=config.wrist_fusion_max_weight,
            wrist_fusion_phase_gate_enabled=config.wrist_fusion_phase_gate_enabled,
            cube_size_m=config.cube_size_m,
            torso_control_enabled=False,
            rotation_action_enabled=config.rotation_action_enabled,
        )
        self.mean = nn.Linear(config.hidden_dim, G2_VISUAL_ARM_ACTION_DIM)
        self.log_std = nn.Linear(config.hidden_dim, G2_VISUAL_ARM_ACTION_DIM)
        self.gripper_logit = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, -2.0)
        nn.init.zeros_(self.gripper_logit.bias)

    def initial_hidden(self, batch: int, *, device=None) -> torch.Tensor:
        return self.policy.initial_hidden(batch, device=device)

    def recurrent_features(
        self,
        rgbd_u8: torch.Tensor,
        proprioception: torch.Tensor,
        hidden: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        grasp_center_position_root_m: torch.Tensor | None = None,
        phase_one_hot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(rgbd_u8.shape[0])
        if grasp_center_position_root_m is None:
            raise ValueError(
                "actor recurrent features require live distal-pad midpoint FK"
            )
        if reset_mask is None:
            reset_mask = torch.zeros(batch, dtype=torch.bool, device=rgbd_u8.device)
        features, next_hidden = self.policy.encode_recurrent_features(
            rgbd_u8.unsqueeze(1),
            proprioception.unsqueeze(1),
            hidden,
            reset_mask.to(torch.bool).unsqueeze(1),
            torch.ones((batch, 1), dtype=torch.bool, device=rgbd_u8.device),
            grasp_center_position_root_m=grasp_center_position_root_m.unsqueeze(1),
            phase_one_hot=(
                None if phase_one_hot is None else phase_one_hot.unsqueeze(1)
            ),
        )
        self.policy._last_future_safe_close_ready_logit = (
            self.policy.future_safe_close_ready_head(features).reshape(-1)
        )
        self.policy._last_close_timing_logit = (
            self.policy.close_timing_head(features).reshape(-1)
        )
        self.policy._last_predicted_contact_steps = F.softplus(
            self.policy.contact_time_head(features)
        ).reshape(-1)
        return features[:, 0], next_hidden

    def sample_from_features(
        self,
        features: torch.Tensor,
        *,
        deterministic: bool = False,
        deterministic_mask: torch.Tensor | np.ndarray | None = None,
    ):
        """Sample one mixed rollout batch without wasting RNG on fixed rows.

        ``deterministic=True`` retains the legacy all-row evaluation API.
        ``deterministic_mask`` is the rollout-only mixed-mode authority: rows
        marked true use the arm mean and binary gripper threshold, while the
        Normal and Bernoulli samplers see only the remaining ordered rows.
        This distinction is important for vector environments because drawing
        a stochastic action and subsequently overwriting it still consumes RNG
        and previously required a second recurrent encoder pass.
        """

        if deterministic and deterministic_mask is not None:
            raise ValueError(
                "deterministic and deterministic_mask are mutually exclusive"
            )
        batch = int(features.shape[0])
        if deterministic_mask is None:
            row_is_deterministic = torch.full(
                (batch,),
                bool(deterministic),
                dtype=torch.bool,
                device=features.device,
            )
        else:
            row_is_deterministic = torch.as_tensor(
                deterministic_mask, device=features.device
            )
            if row_is_deterministic.dtype != torch.bool:
                raise TypeError("deterministic_mask must have boolean dtype")
            if tuple(row_is_deterministic.shape) != (batch,):
                raise ValueError("deterministic_mask must have shape [batch]")

        mean = self.mean(features)
        log_std = -5.0 + 3.5 * (torch.tanh(self.log_std(features)) + 1.0)
        active_dimensions = 6 if self.policy.rotation_action_enabled else 3
        active_mean = mean[..., :active_dimensions]
        active_log_std = log_std[..., :active_dimensions]
        stochastic_rows = (~row_is_deterministic).nonzero(as_tuple=True)[0]
        if stochastic_rows.numel() == 0:
            # Do not instantiate or sample either distribution: an all-
            # deterministic rollout must leave the RNG stream untouched.
            active_pre_tanh = active_mean
        elif stochastic_rows.numel() == batch:
            # Preserve the exact legacy all-stochastic sampling path.
            active_pre_tanh = torch.distributions.Normal(
                active_mean, active_log_std.exp()
            ).rsample()
        else:
            sampled_pre_tanh = torch.distributions.Normal(
                active_mean.index_select(0, stochastic_rows),
                active_log_std.exp().index_select(0, stochastic_rows),
            ).rsample()
            active_pre_tanh = active_mean.index_copy(
                0, stochastic_rows, sampled_pre_tanh
            )
        active_arm = torch.tanh(active_pre_tanh)
        if active_dimensions == G2_VISUAL_ARM_ACTION_DIM:
            arm = active_arm
            mean_action = torch.tanh(mean)
        else:
            rotation_zeros = torch.zeros_like(mean[..., 3:6])
            arm = torch.cat((active_arm, rotation_zeros), dim=-1)
            mean_action = torch.cat(
                (torch.tanh(active_mean), rotation_zeros), dim=-1
            )
        arm_log_probability = squashed_gaussian_log_prob(
            active_pre_tanh, active_mean, active_log_std
        )
        gripper_logit = self.gripper_logit(features)
        probability_open = torch.sigmoid(gripper_logit)
        threshold_open = (1.0 - probability_open) <= float(
            self.deterministic_gripper_close_probability_threshold
        )
        if stochastic_rows.numel() == 0:
            open_sample = threshold_open
        elif stochastic_rows.numel() == batch:
            open_sample = torch.bernoulli(probability_open).to(torch.bool)
        else:
            sampled_open = torch.bernoulli(
                probability_open.index_select(0, stochastic_rows)
            ).to(torch.bool)
            open_sample = threshold_open.index_copy(
                0, stochastic_rows, sampled_open
            )
        gripper = torch.where(
            open_sample,
            torch.ones_like(probability_open),
            -torch.ones_like(probability_open),
        )
        selected_probability = torch.where(
            open_sample, probability_open, 1.0 - probability_open
        )
        log_probability = arm_log_probability + torch.log(
            selected_probability.clamp_min(1.0e-8)
        )
        return (
            torch.cat((arm, gripper), dim=-1),
            log_probability,
            mean_action,
            log_std,
            gripper_logit,
        )


class G2RecurrentVisualAsymmetricSAC:
    """RGB-D encoder + proprioception fusion + GRU asymmetric SAC.

    Replay samples contiguous episode sequences.  The stored rollout state is
    only the starting point: burn-in observations refresh it with the current
    encoder/GRU before losses are applied, avoiding a single-transition GRU
    update with a stale history summary.  Keyboard BC and this online teacher
    share :class:`G2RecurrentVisualStudent` exactly.
    """

    def __init__(
        self,
        config: G2RecurrentVisualSACConfig | None = None,
        *,
        device="cpu",
    ) -> None:
        self.config = config or G2RecurrentVisualSACConfig()
        if self.config.sequence_length <= 0:
            raise ValueError("recurrent sequence length must be positive")
        if self.config.gru_num_layers != 1:
            raise ValueError("recurrent SAC requires exactly one GRU layer")
        if not 0 <= self.config.burn_in_steps < self.config.sequence_length:
            raise ValueError("recurrent burn-in must be inside the sequence")
        if not 0 < self.config.resolved_sequence_stride <= self.config.sequence_length:
            raise ValueError("recurrent sequence stride must be in [1, sequence_length]")
        if self.config.random_shift_pad < 0:
            raise ValueError("random-shift padding cannot be negative")
        if self.config.pose_consistency_weight < 0.0:
            raise ValueError("pose-consistency weight cannot be negative")
        if self.config.pose_consistency_rotation_weight < 0.0:
            raise ValueError("pose-consistency rotation weight cannot be negative")
        if self.config.demonstration_q_filter_temperature <= 0.0:
            raise ValueError("demonstration Q-filter temperature must be positive")
        if self.config.student_head_distillation_weight < 0.0:
            raise ValueError("student head distillation weight cannot be negative")
        if self.config.action_slew_excess_weight < 0.0:
            raise ValueError("action slew-excess weight cannot be negative")
        if self.config.maximum_arm_action_magnitude <= 0.0:
            raise ValueError("maximum arm action magnitude must be positive")
        if not (
            0.0
            < self.config.arm_action_slew_per_policy_step
            <= self.config.maximum_arm_action_magnitude
        ):
            raise ValueError(
                "arm action slew per policy step must be in (0, maximum magnitude]"
            )
        if self.config.near_contact_auxiliary_distance_m <= 0.0:
            raise ValueError("near-contact auxiliary distance must be positive")
        if self.config.near_contact_relative_pose_multiplier < 1.0:
            raise ValueError("near-contact relative-pose multiplier must be at least one")
        if self.config.cross_camera_contrastive_weight < 0.0:
            raise ValueError("cross-camera contrastive weight cannot be negative")
        if self.config.cross_camera_contrastive_temperature <= 0.0:
            raise ValueError("cross-camera contrastive temperature must be positive")
        if self.config.cross_camera_false_negative_position_threshold_m < 0.0:
            raise ValueError("cross-camera false-negative threshold cannot be negative")
        if self.config.temporal_pose_residual_weight < 0.0:
            raise ValueError("temporal pose-residual weight cannot be negative")
        if self.config.temporal_pose_residual_rotation_weight < 0.0:
            raise ValueError("temporal pose-residual rotation weight cannot be negative")
        if self.config.head_depth_profile not in G2_HEAD_DEPTH_PROFILES:
            raise ValueError("unknown head-depth profile")
        if self.config.head_depth_reconstruction_weight < 0.0:
            raise ValueError("head-depth reconstruction weight cannot be negative")
        if self.config.head_depth_edge_weight < 0.0:
            raise ValueError("head-depth edge weight cannot be negative")
        if self.config.head_cube_detector_weight < 0.0:
            raise ValueError("head cube-detector loss cannot be negative")
        if self.config.wrist_cube_position_weight < 0.0:
            raise ValueError("wrist cube-position loss cannot be negative")
        if self.config.vision_xyz_loss_normalization_m <= 0.0:
            raise ValueError("vision XYZ loss normalization must be positive")
        if self.config.vision_loss_gradient_diagnostic_interval <= 0:
            raise ValueError("vision gradient diagnostic interval must be positive")
        if self.config.grasp_ready_auxiliary_weight < 0.0:
            raise ValueError("grasp-ready auxiliary weight cannot be negative")
        if not 0.0 < self.config.grasp_ready_probability_threshold < 1.0:
            raise ValueError("grasp-ready probability threshold must be in (0,1)")
        if self.config.close_timing_auxiliary_weight < 0.0:
            raise ValueError("close-timing auxiliary weight cannot be negative")
        if not 0.0 < self.config.close_timing_probability_threshold < 1.0:
            raise ValueError("close-timing probability threshold must be in (0,1)")
        if self.config.close_timing_support_steps < 0:
            raise ValueError("close-timing support cannot be negative")
        if self.config.close_timing_temperature_steps <= 0.0:
            raise ValueError("close-timing temperature must be positive")
        if self.config.contact_time_auxiliary_weight < 0.0:
            raise ValueError("contact-time auxiliary weight cannot be negative")
        if self.config.contact_time_normalization_steps <= 0.0:
            raise ValueError("contact-time normalization must be positive")
        if self.config.demonstration_arm_loss_weight < 0.0:
            raise ValueError("demonstration arm loss weight cannot be negative")
        if self.config.demonstration_gripper_loss_weight < 0.0:
            raise ValueError("demonstration gripper loss weight cannot be negative")
        if self.config.demonstration_close_onset_loss_weight < 1.0:
            raise ValueError(
                "demonstration close-onset loss weight must be at least one"
            )
        if not 0.0 <= self.config.demonstration_close_onset_target_loss_fraction < 1.0:
            raise ValueError(
                "demonstration close-onset target loss fraction must be in [0,1)"
            )
        if not 0.0 <= self.config.demonstration_close_onset_row_fraction < 1.0:
            raise ValueError(
                "demonstration close-onset row fraction must be in [0,1)"
            )
        if not 0.0 < self.config.deterministic_gripper_close_probability_threshold < 1.0:
            raise ValueError(
                "deterministic gripper close probability threshold must be in (0,1)"
            )
        if self.config.demonstration_close_onset_support_steps < 0:
            raise ValueError(
                "demonstration close-onset support steps cannot be negative"
            )
        if self.config.grasp_ready_tolerance_m <= 0.0:
            raise ValueError("grasp-ready tolerance must be positive")
        if (
            self.config.head_depth_reconstruction_weight > 0.0
            and self.config.head_depth_profile != "coord_gn_silu_unet_aux"
        ):
            raise ValueError("head-depth reconstruction loss requires the U-Net auxiliary profile")
        if (
            self.config.head_depth_edge_weight > 0.0
            and self.config.head_depth_profile != "lite_unet_fpn_spatial_softmax"
        ):
            raise ValueError("head-depth edge loss requires the Lite U-Net/FPN profile")
        self.device = torch.device(device)
        self.actor = _RecurrentVisualActor(self.config).to(self.device)
        self.q1, self.q2 = _Critic().to(self.device), _Critic().to(self.device)
        self.tq1, self.tq2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.tq1.requires_grad_(False)
        self.tq2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        vision_parameters = (
            list(self.actor.policy.visual.parameters())
            + list(self.actor.policy.state_encoder.parameters())
            + list(self.actor.policy.gru.parameters())
            + list(self.actor.policy.future_safe_close_ready_head.parameters())
            + list(self.actor.policy.close_timing_head.parameters())
            + list(self.actor.policy.contact_time_head.parameters())
            + list(self.actor.policy.relative_pose_head.parameters())
            + list(self.actor.policy.depth_validity_head.parameters())
        )
        self.vision_optimizer = torch.optim.Adam(
            vision_parameters, lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.learning_rate,
        )
        self.log_alpha = torch.tensor(
            math.log(self.config.initial_alpha),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.learning_rate
        )
        self.update_count = 0
        self._vision_update_count = 0
        # Calibrated on excluded demonstration rows before policy-only
        # rollout.  It affects deterministic rollout only; stochastic SAC
        # sampling and entropy remain unchanged.
        self.actor.deterministic_gripper_close_probability_threshold = float(
            self.config.deterministic_gripper_close_probability_threshold
        )

    def reset_vision_optimizer(self) -> None:
        """Reset only vision optimizer moments for an independent candidate."""

        vision_parameters = (
            list(self.actor.policy.visual.parameters())
            + list(self.actor.policy.state_encoder.parameters())
            + list(self.actor.policy.gru.parameters())
            + list(self.actor.policy.future_safe_close_ready_head.parameters())
            + list(self.actor.policy.close_timing_head.parameters())
            + list(self.actor.policy.contact_time_head.parameters())
            + list(self.actor.policy.relative_pose_head.parameters())
            + list(self.actor.policy.depth_validity_head.parameters())
        )
        self.vision_optimizer = torch.optim.Adam(
            vision_parameters, lr=self.config.learning_rate
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def initial_hidden(self, batch: int) -> torch.Tensor:
        return self.actor.initial_hidden(batch, device=self.device)

    def set_pre_sac_encoder_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze representation modules at curriculum boundaries."""

        policy = self.actor.policy
        for module in (
            policy.visual,
            policy.state_encoder,
            policy.gru,
            policy.future_safe_close_ready_head,
            policy.close_timing_head,
            policy.contact_time_head,
            policy.relative_pose_head,
            policy.depth_validity_head,
        ):
            module.requires_grad_(not frozen)
        if frozen:
            policy.eval()
        else:
            policy.train()

    def set_visual_encoder_frozen(self, frozen: bool) -> None:
        """Freeze only camera representation weights, not the fresh controller.

        This is the staged-vision/fresh-controller boundary.  The GRU and
        action heads must remain trainable during phase-balanced BC and SAC.
        """

        self.actor.policy.visual.requires_grad_(not frozen)
        if frozen:
            self.actor.policy.visual.eval()
        else:
            self.actor.policy.visual.train()

    def set_controller_training_mode(self) -> None:
        """Put the trainable controller path in training mode.

        Final vision validation deliberately leaves the complete policy in
        evaluation mode.  Phase-balanced BC immediately follows that event,
        and cuDNN refuses to backpropagate through an RNN whose forward pass
        ran in evaluation mode.  Own the module-mode boundary here instead of
        relying on the preceding curriculum event.  A frozen visual encoder
        remains in evaluation mode so BatchNorm/running-state buffers cannot
        drift while the fresh GRU and action heads learn.
        """

        self.actor.train()
        if not any(
            parameter.requires_grad
            for parameter in self.actor.policy.visual.parameters()
        ):
            self.actor.policy.visual.eval()

    @staticmethod
    def _reset_module_parameters(module: nn.Module) -> None:
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()

    def reset_recurrent_and_action_heads_after_vision(self) -> dict[str, object]:
        """Discard every old controller weight while preserving visual weights."""

        visual_hash_before = g2_tensor_state_dict_sha256(
            self.actor.policy.visual.state_dict()
        )
        audited_modules = {
            "gru": self.actor.policy.gru,
            "arm_head": self.actor.policy.arm_action_head,
            "gripper_head": self.actor.policy.gripper_head,
            "future_safe_head": self.actor.policy.future_safe_close_ready_head,
            "timing_head": self.actor.policy.close_timing_head,
            "contact_time_head": self.actor.policy.contact_time_head,
        }
        component_hashes_before = {
            name: g2_tensor_state_dict_sha256(module.state_dict())
            for name, module in audited_modules.items()
        }
        controller_modules = (
            *audited_modules.values(),
            self.actor.policy.torso_action_head,
            self.actor.policy.torso_gate_head,
            self.actor.mean,
            self.actor.log_std,
            self.actor.gripper_logit,
        )
        controller_hash_before = g2_tensor_state_dict_sha256(
            {
                name: value
                for name, value in self.actor.state_dict().items()
                if name.startswith((
                    "policy.gru.", "policy.arm_action_head.",
                    "policy.gripper_head.", "policy.torso_action_head.",
                    "policy.future_safe_close_ready_head.",
                    "policy.close_timing_head.", "policy.contact_time_head.",
                    "policy.torso_gate_head.", "mean.", "log_std.",
                    "gripper_logit.",
                ))
            }
        )
        for module in controller_modules:
            self._reset_module_parameters(module)
        for module in (
            self.actor.policy.state_encoder,
            self.actor.policy.gru,
            self.actor.policy.arm_action_head,
            self.actor.policy.gripper_head,
            self.actor.policy.future_safe_close_ready_head,
            self.actor.policy.close_timing_head,
            self.actor.policy.contact_time_head,
            self.actor.mean,
            self.actor.log_std,
            self.actor.gripper_logit,
        ):
            module.requires_grad_(True)
        nn.init.zeros_(self.actor.mean.bias)
        nn.init.constant_(self.actor.log_std.bias, -2.0)
        nn.init.zeros_(self.actor.gripper_logit.bias)
        self.actor.policy.torso_action_head.requires_grad_(False)
        self.actor.policy.torso_gate_head.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            (parameter for parameter in self.actor.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
        )
        self.set_visual_encoder_frozen(True)
        self.set_controller_training_mode()
        visual_hash_after = g2_tensor_state_dict_sha256(
            self.actor.policy.visual.state_dict()
        )
        controller_hash_after = g2_tensor_state_dict_sha256(
            {
                name: value
                for name, value in self.actor.state_dict().items()
                if name.startswith((
                    "policy.gru.", "policy.arm_action_head.",
                    "policy.gripper_head.", "policy.torso_action_head.",
                    "policy.future_safe_close_ready_head.",
                    "policy.close_timing_head.", "policy.contact_time_head.",
                    "policy.torso_gate_head.", "mean.", "log_std.",
                    "gripper_logit.",
                ))
            }
        )
        component_hashes_after = {
            name: g2_tensor_state_dict_sha256(module.state_dict())
            for name, module in audited_modules.items()
        }
        if visual_hash_before != visual_hash_after:
            raise RuntimeError("G2_FRESH_CONTROLLER_RESET_CHANGED_VISUAL_ENCODER")
        if controller_hash_before == controller_hash_after:
            raise RuntimeError("G2_FRESH_CONTROLLER_RESET_DID_NOT_CHANGE_CONTROLLER")
        unchanged = sorted(
            name for name in audited_modules
            if component_hashes_before[name] == component_hashes_after[name]
        )
        if unchanged:
            raise RuntimeError(
                "G2_FRESH_CONTROLLER_COMPONENT_NOT_RESET:" + ",".join(unchanged)
            )
        return {
            "visual_sha256": visual_hash_after,
            "controller_sha256_before": controller_hash_before,
            "controller_sha256_after": controller_hash_after,
            "component_sha256_before": component_hashes_before,
            "component_sha256_after": component_hashes_after,
        }

    def vision_pretrain_update(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        apply_update: bool = True,
        cross_camera_contrastive_weight: float | None = None,
    ) -> dict[str, float]:
        """Update only representation modules from the separate disk dataset.

        Simulator cube geometry is consumed strictly below as an auxiliary
        label.  It is never concatenated into the actor observation.  Camera
        visibility gates every pose objective for which the cube is absent;
        contact/stable-grasp labels are intentionally not part of this loss.
        """

        required = {
            "rgbd_u8",
            "proprio",
            "cube_position_root_m",
            "camera_visible",
            "camera_frame_age_s",
            "grasp_center_position_root_m",
            "phase_one_hot",
            "env_id",
            "episode_id",
            "sequence_step",
            "safe_for_contact",
            "alignment_valid",
            "reference_grasp_offset_root_m",
            "future_safe_close_ready",
            "future_safe_close_ready_valid",
            "close_timing_target",
            "close_timing_valid",
            "contact_lead_steps_target",
            "contact_lead_steps_valid",
        }
        metadata_fields = {
            # Dataset provenance is used only for source-balanced sampling
            # and per-source validation telemetry.  It must never enter the
            # visual encoder, recurrent actor, or auxiliary supervision.
            "source_dataset_index",
            "collection_mode",
            "policy_like_active",
            "perturbation_xyz_m",
            "ee_cube_distance_m",
            "distance_bin",
            "ee_pose_root_xyzw",
            "curriculum_stage",
            "recovery_success",
            "recovery_completed",
            "recovery_timeout",
            "recovery_steps",
            "reference_progress_frozen",
            "recovery_anchor_target_cube_minus_ee_m",
            "recovery_anchor_phase",
            "gripper_state",
            "gripper_open_command",
            "expert_first_close",
            "bilateral_contact",
            "stable_grasp",
            "future_safe_close_ready_right_censored",
            "future_safe_close_to_onset_steps",
            "future_safe_close_to_contact_steps",
            "future_safe_close_to_stable_steps",
            # Historical keyboard demonstrations predate per-frame camera
            # transform storage.  They are still valid for RGB-D/XYZ
            # supervision because the camera calibration is fixed and the
            # GT label is already expressed in robot-root coordinates.  The
            # transform is optional only for geometry diagnostics; live data
            # always supplies it.
            "camera_world_transform",
        }
        missing = required - set(batch)
        unexpected = set(batch) - required - metadata_fields
        if missing or unexpected:
            raise ValueError(
                f"vision learner fields differ: missing={missing}, unexpected={unexpected}"
            )
        rgbd = torch.as_tensor(batch["rgbd_u8"], device=self.device)
        proprio = torch.as_tensor(batch["proprio"], dtype=torch.float32, device=self.device)
        cube = torch.as_tensor(
            batch["cube_position_root_m"], dtype=torch.float32, device=self.device
        )
        visible = torch.as_tensor(
            batch["camera_visible"], dtype=torch.bool, device=self.device
        )
        safe_for_contact = torch.as_tensor(
            batch["safe_for_contact"], dtype=torch.bool, device=self.device
        )
        alignment_valid = torch.as_tensor(
            batch["alignment_valid"], dtype=torch.bool, device=self.device
        )
        reference_grasp_offset = torch.as_tensor(
            batch["reference_grasp_offset_root_m"],
            dtype=torch.float32,
            device=self.device,
        )
        grasp_center = torch.as_tensor(
            batch["grasp_center_position_root_m"], dtype=torch.float32, device=self.device
        )
        phase = torch.as_tensor(
            batch["phase_one_hot"], dtype=torch.float32, device=self.device
        )
        future_safe_ready = torch.as_tensor(
            batch["future_safe_close_ready"], dtype=torch.bool, device=self.device
        )
        future_safe_ready_valid = torch.as_tensor(
            batch["future_safe_close_ready_valid"],
            dtype=torch.bool,
            device=self.device,
        )
        close_timing_target = torch.as_tensor(
            batch["close_timing_target"], dtype=torch.float32, device=self.device
        )
        close_timing_valid = torch.as_tensor(
            batch["close_timing_valid"], dtype=torch.bool, device=self.device
        )
        contact_lead_target = torch.as_tensor(
            batch["contact_lead_steps_target"], dtype=torch.float32, device=self.device
        )
        contact_lead_valid = torch.as_tensor(
            batch["contact_lead_steps_valid"], dtype=torch.bool, device=self.device
        )
        if rgbd.ndim != 6 or tuple(rgbd.shape[2:]) != G2_VISUAL_CAMERA_SHAPE:
            raise ValueError("vision learner RGB-D must be [B,T,2,6,48,64]")
        batch_size, steps = rgbd.shape[:2]
        expected_sequence = (batch_size, steps)
        if tuple(proprio.shape[:2]) != expected_sequence or tuple(cube.shape) != (*expected_sequence, 3):
            raise ValueError("vision learner sequence fields disagree")
        reset = torch.zeros(expected_sequence, dtype=torch.bool, device=self.device)
        reset[:, 0] = True
        padding = torch.ones(expected_sequence, dtype=torch.bool, device=self.device)
        if apply_update:
            self.set_pre_sac_encoder_frozen(False)
        features, _, projected_camera_tokens = self.actor.policy.encode_recurrent_features(
            rgbd,
            proprio,
            self.actor.initial_hidden(batch_size, device=self.device),
            reset,
            padding,
            grasp_center_position_root_m=grasp_center,
            phase_one_hot=phase,
            return_camera_tokens=True,
        )
        perception = self.actor.policy.visual.last_perception
        flat_cube = cube.reshape(-1, 3)
        flat_visible = visible.reshape(-1, 2)
        head_detector, wrist_position, localization = (
            self.actor.policy.visual.cube_localization_supervision(
                flat_cube,
                perception=perception,
            )
        )
        # Recompute the two position reductions with the *rendered* visibility
        # masks from the dataset.  Geometric in-frustum status alone is not a
        # valid supervision authority under occlusion.
        head_mask = flat_visible[:, 0].to(torch.float32)
        wrist_mask = flat_visible[:, 1].to(torch.float32)
        xyz_scale = float(self.config.vision_xyz_loss_normalization_m)
        head_delta = perception["head_cube_xyz_root_m"] - flat_cube
        wrist_delta = perception["wrist_cube_xyz_root_m"] - flat_cube
        fused_delta = perception["fused_cube_xyz_root_m"] - flat_cube
        head_error = head_delta.abs().mean(dim=-1) / xyz_scale
        wrist_error = wrist_delta.abs().mean(dim=-1) / xyz_scale
        head_position = (head_error * head_mask).sum() / head_mask.sum().clamp_min(1.0)
        wrist_position = (wrist_error * wrist_mask).sum() / wrist_mask.sum().clamp_min(1.0)
        head_logits = perception["head_heatmap_logits"].flatten(1)
        head_visibility_logit = torch.logsumexp(head_logits, dim=-1) - math.log(
            float(head_logits.shape[-1])
        )
        visibility_loss = F.binary_cross_entropy_with_logits(
            head_visibility_logit, head_mask
        )
        previous_action = proprio[
            ..., G2StudentObservationContract().slices["previous_action"]
        ]
        gripper_open = (
            previous_action[..., G2_VISUAL_GRIPPER_ACTION_INDEX] >= 0.0
        ).reshape(-1)
        head_depth_valid = (
            rgbd[:, :, 0, 5].reshape(-1, *rgbd.shape[-2:]) > 127
        ).flatten(1).any(dim=-1)
        static_geometry_target = grasp_ready_supervision_target(
            cube_position_root_m=flat_cube,
            grasp_center_position_root_m=grasp_center.reshape(-1, 3),
            reference_grasp_offset_root_m=reference_grasp_offset.reshape(-1, 3),
            head_visible=flat_visible[:, 0],
            depth_valid=head_depth_valid,
            safe_for_contact=safe_for_contact.reshape(-1),
            gripper_open=gripper_open,
            grasp_ready_tolerance_m=self.config.grasp_ready_tolerance_m,
            alignment_valid=alignment_valid.reshape(-1),
        )
        grasp_ready_target = future_safe_ready.reshape(-1)
        grasp_ready_valid = (
            future_safe_ready_valid.reshape(-1)
            & flat_visible[:, 0]
            & head_depth_valid
            & gripper_open
        )
        temporal_grasp_ready_logit = self.actor.policy.future_safe_close_ready_head(
            features
        ).reshape(-1)
        frame_grasp_ready_logit = perception["grasp_ready_logit"]
        grasp_ready_logit = temporal_grasp_ready_logit
        valid_logits = grasp_ready_logit[grasp_ready_valid]
        valid_targets = grasp_ready_target[grasp_ready_valid]
        positive_loss = F.binary_cross_entropy_with_logits(
            valid_logits[valid_targets],
            torch.ones_like(valid_logits[valid_targets]),
        ) if bool(valid_targets.any()) else grasp_ready_logit.sum() * 0.0
        negative_loss = F.binary_cross_entropy_with_logits(
            valid_logits[~valid_targets],
            torch.zeros_like(valid_logits[~valid_targets]),
        ) if bool((~valid_targets).any()) else grasp_ready_logit.sum() * 0.0
        temporal_grasp_ready_loss = 0.5 * (positive_loss + negative_loss)
        frame_valid_logits = frame_grasp_ready_logit[grasp_ready_valid]
        frame_positive_loss = F.binary_cross_entropy_with_logits(
            frame_valid_logits[valid_targets],
            torch.ones_like(frame_valid_logits[valid_targets]),
        ) if bool(valid_targets.any()) else frame_grasp_ready_logit.sum() * 0.0
        frame_negative_loss = F.binary_cross_entropy_with_logits(
            frame_valid_logits[~valid_targets],
            torch.zeros_like(frame_valid_logits[~valid_targets]),
        ) if bool((~valid_targets).any()) else frame_grasp_ready_logit.sum() * 0.0
        frame_grasp_ready_loss = 0.5 * (frame_positive_loss + frame_negative_loss)
        grasp_ready_loss = (
            0.75 * temporal_grasp_ready_loss + 0.25 * frame_grasp_ready_loss
        )
        close_timing_logit = self.actor.policy.close_timing_head(features)
        timing_mask = close_timing_valid.reshape(-1)
        close_timing_loss = (
            F.binary_cross_entropy_with_logits(
                close_timing_logit.reshape(-1)[timing_mask],
                close_timing_target.reshape(-1)[timing_mask],
            )
            if bool(timing_mask.any())
            else close_timing_logit.sum() * 0.0
        )
        close_timing_prediction = (
            torch.sigmoid(close_timing_logit.reshape(-1))
            >= float(self.config.close_timing_probability_threshold)
        )
        timing_binary_target = close_timing_target.reshape(-1) >= 0.5
        timing_tp = (
            close_timing_prediction & timing_binary_target & timing_mask
        ).sum().to(torch.float32)
        timing_precision = timing_tp / (close_timing_prediction & timing_mask).sum().clamp_min(1)
        timing_recall = timing_tp / (timing_binary_target & timing_mask).sum().clamp_min(1)
        timing_f1 = (
            2.0 * timing_precision * timing_recall
            / (timing_precision + timing_recall).clamp_min(1.0e-8)
        )
        contact_time_prediction = F.softplus(
            self.actor.policy.contact_time_head(features)
        ).reshape(-1)
        contact_time_mask = contact_lead_valid.reshape(-1)
        contact_scale = float(self.config.contact_time_normalization_steps)
        contact_time_loss = (
            F.smooth_l1_loss(
                contact_time_prediction[contact_time_mask] / contact_scale,
                contact_lead_target.reshape(-1)[contact_time_mask] / contact_scale,
            )
            if bool(contact_time_mask.any())
            else contact_time_prediction.sum() * 0.0
        )
        contact_time_mae = (
            (contact_time_prediction[contact_time_mask] - contact_lead_target.reshape(-1)[contact_time_mask]).abs().mean()
            if bool(contact_time_mask.any())
            else contact_time_prediction.new_zeros(())
        )
        grasp_ready_prediction = (
            torch.sigmoid(grasp_ready_logit)
            >= float(self.config.grasp_ready_probability_threshold)
        )
        valid_prediction = grasp_ready_prediction & grasp_ready_valid
        valid_positive = grasp_ready_target & grasp_ready_valid
        grasp_ready_tp = (valid_prediction & valid_positive).sum()
        grasp_ready_precision = grasp_ready_tp.to(torch.float32) / valid_prediction.sum().clamp_min(1)
        grasp_ready_recall = grasp_ready_tp.to(torch.float32) / valid_positive.sum().clamp_min(1)
        grasp_ready_f1 = (
            2.0 * grasp_ready_precision * grasp_ready_recall
            / (grasp_ready_precision + grasp_ready_recall).clamp_min(1.0e-8)
        )
        flat_phase = phase.reshape(-1, G2_DEPLOYABLE_PHASE_DIM)
        flat_episode = torch.as_tensor(
            batch["episode_id"], dtype=torch.long, device=self.device
        ).reshape(-1)
        raw_distance = torch.linalg.vector_norm(
            flat_cube - grasp_center.reshape(-1, 3), dim=-1
        )
        positive_rows = int(valid_positive.sum().item())
        valid_rows = int(grasp_ready_valid.sum().item())
        negative_rows = int(valid_rows - positive_rows)
        positive_episode_count = int(
            torch.unique(flat_episode[valid_positive]).numel()
        )
        static_on_valid = static_geometry_target & grasp_ready_valid
        static_tp = (static_on_valid & valid_positive).sum().to(torch.float32)
        static_precision = static_tp / static_on_valid.sum().clamp_min(1)
        static_recall = static_tp / valid_positive.sum().clamp_min(1)
        static_f1 = (
            2.0 * static_precision * static_recall
            / (static_precision + static_recall).clamp_min(1.0e-8)
        )
        depth_target = rgbd[:, :, :, 5].float().div(255.0).mean(dim=(-1, -2))
        depth_loss = F.binary_cross_entropy_with_logits(
            self.actor.policy.depth_validity_head(features), depth_target
        )
        relative_prediction = self.actor.policy.relative_pose_head(features)[..., :3]
        relative_target = cube - grasp_center
        temporal_mask = (
            visible[:, 1:, :].any(dim=-1)
            & visible[:, :-1, :].any(dim=-1)
        ).to(torch.float32)
        predicted_delta = relative_prediction[:, 1:] - relative_prediction[:, :-1]
        target_delta = relative_target[:, 1:] - relative_target[:, :-1]
        temporal_error = (predicted_delta - target_delta).abs().mean(dim=-1)
        temporal_loss = (
            (temporal_error * temporal_mask).sum()
            / temporal_mask.sum().clamp_min(1.0)
        )
        contrastive_valid = visible[..., 0] & visible[..., 1]
        relative_pose_for_pairing = torch.cat(
            (
                cube - grasp_center,
                torch.zeros((*cube.shape[:2], 3), device=self.device),
                torch.ones((*cube.shape[:2], 1), device=self.device),
            ),
            dim=-1,
        )
        effective_crossview_weight = (
            float(self.config.cross_camera_contrastive_weight)
            if cross_camera_contrastive_weight is None
            else float(cross_camera_contrastive_weight)
        )
        if not 0.0 <= effective_crossview_weight <= 1.0:
            raise ValueError("cross-camera contrastive weight override must be in [0,1]")
        if effective_crossview_weight > 0.0:
            crossview_loss = cross_camera_contrastive_loss(
                projected_camera_tokens,
                contrastive_valid,
                relative_pose_target=relative_pose_for_pairing,
                temperature=self.config.cross_camera_contrastive_temperature,
                false_negative_position_threshold_m=(
                    self.config.cross_camera_false_negative_position_threshold_m
                ),
            )
        else:
            crossview_loss = projected_camera_tokens.sum() * 0.0
        selected_tokens = projected_camera_tokens[contrastive_valid]
        if selected_tokens.shape[0]:
            normalized_head = F.normalize(selected_tokens[:, 0], dim=-1)
            normalized_wrist = F.normalize(selected_tokens[:, 1], dim=-1)
            positive_similarity = (normalized_head * normalized_wrist).sum(dim=-1).mean()
            if selected_tokens.shape[0] > 1:
                similarity_matrix = normalized_head @ normalized_wrist.transpose(0, 1)
                negative_similarity = (
                    similarity_matrix.sum() - similarity_matrix.diagonal().sum()
                ) / float(selected_tokens.shape[0] * (selected_tokens.shape[0] - 1))
            else:
                negative_similarity = positive_similarity.new_zeros(())
        else:
            positive_similarity = loss_zero = features.sum() * 0.0
            negative_similarity = loss_zero
        heatmap_loss = localization["loss_head_heatmap"]
        weighted_terms = {
            "heatmap": heatmap_loss,
            "head_xyz": 0.2 * head_position,
            "wrist_xyz": 0.2 * wrist_position,
            "depth": 0.05 * depth_loss,
            "visibility": 0.05 * visibility_loss,
            "temporal": 0.05 * temporal_loss,
            # InfoNCE is minimized, hence the positive sign. A minus sign is
            # correct only when the objective is defined as similarity.
            "crossview": effective_crossview_weight * crossview_loss,
            "grasp_ready": self.config.grasp_ready_auxiliary_weight * grasp_ready_loss,
            "close_timing": self.config.close_timing_auxiliary_weight * close_timing_loss,
            "contact_time": self.config.contact_time_auxiliary_weight * contact_time_loss,
        }
        loss = sum(weighted_terms.values())
        gradient_contributions: dict[str, float] = {}
        if apply_update:
            self.vision_optimizer.zero_grad(set_to_none=True)
            self._vision_update_count += 1
            if self._vision_update_count % self.config.vision_loss_gradient_diagnostic_interval == 0:
                parameters = [
                    parameter
                    for group in self.vision_optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.requires_grad
                ]
                for name, term in weighted_terms.items():
                    gradients = torch.autograd.grad(
                        term, parameters, retain_graph=True, allow_unused=True
                    )
                    squared = sum(
                        gradient.detach().square().sum()
                        for gradient in gradients
                        if gradient is not None
                    )
                    gradient_contributions[name] = float(torch.sqrt(squared)) if not isinstance(squared, int) else 0.0
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in self.vision_optimizer.param_groups
                    for parameter in group["params"]
                ],
                self.config.gradient_clip,
            )
            self.vision_optimizer.step()
        else:
            gradient_norm = loss.new_zeros(())
        metric = {
            "vision_pretrain/loss_total": float(loss.detach()),
            "vision_pretrain/loss_heatmap": float(heatmap_loss.detach()),
            "vision_pretrain/loss_head_xyz": float(head_position.detach()),
            "vision_pretrain/loss_wrist_xyz": float(wrist_position.detach()),
            "vision_pretrain/loss_depth": float(depth_loss.detach()),
            "vision_pretrain/loss_visibility": float(visibility_loss.detach()),
            "vision_pretrain/loss_temporal": float(temporal_loss.detach()),
            "vision_pretrain/loss_crossview": float(crossview_loss.detach()),
            "vision_pretrain/loss_grasp_ready": float(grasp_ready_loss.detach()),
            "vision_pretrain/loss_close_timing": float(close_timing_loss.detach()),
            "vision_pretrain/loss_contact_time": float(contact_time_loss.detach()),
            "vision_pretrain/loss_future_safe_temporal": float(
                temporal_grasp_ready_loss.detach()
            ),
            "vision_pretrain/loss_future_safe_frame_advisory": float(
                frame_grasp_ready_loss.detach()
            ),
            "vision_pretrain/effective_crossview_weight": effective_crossview_weight,
            "vision_pretrain/crossview_positive_cosine": float(positive_similarity.detach()),
            "vision_pretrain/crossview_negative_cosine": float(negative_similarity.detach()),
            "vision_pretrain/crossview_pair_fraction": float(contrastive_valid.float().mean()),
            "vision_pretrain/xyz_normalization_m": xyz_scale,
            "vision_pretrain/head_visible_fraction": float(head_mask.mean()),
            "vision_pretrain/wrist_visible_fraction": float(wrist_mask.mean()),
            "vision_pretrain/gradient_norm": float(gradient_norm),
            "vision_pretrain/update_applied": float(apply_update),
            "vision/grasp_ready_precision": float(grasp_ready_precision.detach()),
            "vision/grasp_ready_recall": float(grasp_ready_recall.detach()),
            "vision/grasp_ready_f1": float(grasp_ready_f1.detach()),
            "vision/future_safe_grasp_ready_precision": float(grasp_ready_precision.detach()),
            "vision/future_safe_grasp_ready_recall": float(grasp_ready_recall.detach()),
            "vision/future_safe_grasp_ready_f1": float(grasp_ready_f1.detach()),
            "vision/future_safe_precision": float(grasp_ready_precision.detach()),
            "vision/future_safe_recall": float(grasp_ready_recall.detach()),
            "vision/future_safe_f1": float(grasp_ready_f1.detach()),
            "vision/close_timing_precision": float(timing_precision.detach()),
            "vision/close_timing_recall": float(timing_recall.detach()),
            "vision/close_timing_f1": float(timing_f1.detach()),
            "dynamics/predicted_contact_steps": float(
                contact_time_prediction[contact_time_mask].mean().detach()
                if bool(contact_time_mask.any()) else 0.0
            ),
            "dynamics/actual_contact_steps": float(
                contact_lead_target.reshape(-1)[contact_time_mask].mean().detach()
                if bool(contact_time_mask.any()) else 0.0
            ),
            "dynamics/contact_step_prediction_error": float(contact_time_mae.detach()),
            "vision/static_geometry_vs_future_precision": float(static_precision.detach()),
            "vision/static_geometry_vs_future_recall": float(static_recall.detach()),
            "vision/static_geometry_vs_future_f1": float(static_f1.detach()),
            "vision/grasp_ready_positive_fraction": float(
                positive_rows / max(valid_rows, 1)
            ),
            "vision/grasp_ready_audit_total_rows": float(valid_rows),
            "vision/grasp_ready_audit_positive_rows": float(positive_rows),
            "vision/grasp_ready_audit_negative_rows": float(negative_rows),
            "vision/grasp_ready_audit_positive_episode_count": float(
                positive_episode_count
            ),
        }
        for phase_index, phase_name in enumerate(G2_DEPLOYABLE_PHASES[:3]):
            phase_mask = flat_phase[:, phase_index] > 0.5
            phase_positive = valid_positive & phase_mask
            metric[
                f"vision/grasp_ready_audit_positive_{phase_name.lower()}_rows"
            ] = float(phase_positive.sum().item())
            metric[
                f"vision/grasp_ready_audit_{phase_name.lower()}_positive_fraction"
            ] = float(
                phase_positive.sum().item()
                / max(1, int((phase_mask & grasp_ready_valid).sum().item()))
            )
        for distance_name, distance_mask in {
            "gt8cm": raw_distance > 0.08,
            "5to8cm": (raw_distance >= 0.05) & (raw_distance <= 0.08),
            "lt5cm": raw_distance < 0.05,
        }.items():
            metric[
                f"vision/grasp_ready_audit_positive_{distance_name}_rows"
            ] = float((valid_positive & distance_mask).sum().item())
        for name, term in weighted_terms.items():
            metric[f"vision_pretrain/weighted_{name}"] = float(term.detach())
            metric[f"vision_pretrain/gradient_{name}_norm"] = gradient_contributions.get(name, 0.0)

        def masked_axis_and_norm(prefix: str, delta: torch.Tensor, mask: torch.Tensor) -> None:
            selected = delta[mask.to(torch.bool)]
            if selected.numel() == 0:
                return
            metric[f"vision/{prefix}_samples"] = float(selected.shape[0])
            metric[f"vision/{prefix}_xyz_error_mm"] = float(
                torch.linalg.vector_norm(selected.detach(), dim=-1).mean() * 1000.0
            )
            for axis, axis_name in enumerate(("x", "y", "z")):
                metric[f"vision/{prefix}_error_{axis_name}_mm"] = float(
                    selected[:, axis].detach().mean() * 1000.0
                )

        masked_axis_and_norm("head", head_delta, flat_visible[:, 0])
        masked_axis_and_norm("wrist", wrist_delta, flat_visible[:, 1])
        masked_axis_and_norm("fused", fused_delta, flat_visible.any(dim=-1))

        # Validation-only ablation: no sweep value enters actor features.
        both = flat_visible[:, 0] & flat_visible[:, 1]
        if bool(both.any()):
            head_xyz = perception["head_cube_xyz_root_m"]
            wrist_xyz = perception["wrist_cube_xyz_root_m"]
            for gate in (0.0, 0.25, 0.5, 0.75, 1.0):
                estimate = (1.0 - gate) * head_xyz + gate * wrist_xyz
                error_mm = torch.linalg.vector_norm(estimate[both] - flat_cube[both], dim=-1).mean() * 1000.0
                metric[f"vision/fusion_sweep_g{gate:.2f}_xyz_error_mm"] = float(error_mm.detach())
            # Deployment-safe candidate sweep: its distance is computed from
            # Head prediction plus distal-pad FK, never simulator cube GT.
            predicted_distance = torch.linalg.vector_norm(
                head_xyz - grasp_center.reshape(-1, 3), dim=-1
            )
            coordinate = ((0.08 - predicted_distance) / 0.03).clamp(0.0, 1.0)
            smooth_gate = coordinate.square() * (3.0 - 2.0 * coordinate)
            for maximum in (0.25, 0.50, 0.75, 1.0):
                gate = maximum * smooth_gate
                estimate = (1.0 - gate[:, None]) * head_xyz + gate[:, None] * wrist_xyz
                estimate_error = torch.linalg.vector_norm(
                    estimate - flat_cube, dim=-1
                ) * 1000.0
                error_mm = estimate_error[both].mean()
                metric[
                    f"vision/distance_gate_5to8cm_max{maximum:.2f}_xyz_error_mm"
                ] = float(error_mm.detach())
                gate_bins = {
                    "5to8cm": both
                    & (predicted_distance >= 0.05)
                    & (predicted_distance <= 0.08),
                    "lt5cm": both & (predicted_distance < 0.05),
                }
                for gate_bin_name, gate_bin_mask in gate_bins.items():
                    if bool(gate_bin_mask.any()):
                        metric[
                            "vision/distance_gate_"
                            f"max{maximum:.2f}_xyz_error_mm/{gate_bin_name}"
                        ] = float(estimate_error[gate_bin_mask].mean().detach())

        gt_distance = torch.linalg.vector_norm(cube - grasp_center, dim=-1).reshape(-1)
        distance_masks = {
            "gt20cm": gt_distance > 0.20,
            "10to20cm": (gt_distance > 0.10) & (gt_distance <= 0.20),
            "5to10cm": (gt_distance > 0.05) & (gt_distance <= 0.10),
            "lt5cm": gt_distance <= 0.05,
            "near_contact": (
                (phase.reshape(-1, G2_DEPLOYABLE_PHASE_DIM)[:, 1:3].sum(dim=-1) > 0.5)
                & flat_visible.all(dim=-1)
            ),
        }
        # v29 policy-like collection bins are task-facing and intentionally
        # narrower than the historical diagnostic bins above.
        distance_masks.update(
            {
                "gt8cm": gt_distance > 0.08,
                "5to8cm": (gt_distance >= 0.05) & (gt_distance <= 0.08),
                "lt5cm": gt_distance < 0.05,
            }
        )
        for bin_name, bin_mask in distance_masks.items():
            for source_name, delta, source_mask in (
                ("head", head_delta, flat_visible[:, 0]),
                ("wrist", wrist_delta, flat_visible[:, 1]),
                ("fused", fused_delta, flat_visible.any(dim=-1)),
            ):
                selected = bin_mask & source_mask
                if bool(selected.any()):
                    metric[f"vision/{source_name}_xyz_error_mm/{bin_name}"] = float(
                        torch.linalg.vector_norm(delta[selected].detach(), dim=-1).mean() * 1000.0
                    )
                    metric[f"vision/{source_name}_samples/{bin_name}"] = float(selected.sum())

        source_dataset_item = batch.get("source_dataset_index")
        if source_dataset_item is not None:
            source_dataset = torch.as_tensor(
                source_dataset_item, dtype=torch.long, device=self.device
            )
            if source_dataset.ndim == 1:
                source_dataset = source_dataset[:, None].expand(
                    -1, phase.shape[1]
                )
            source_dataset = source_dataset.reshape(-1)
            if source_dataset.shape[0] != flat_cube.shape[0]:
                raise ValueError(
                    "source_dataset_index must be [batch] or [batch,time]"
                )
            near_mask = distance_masks["near_contact"]
            for source_id in torch.unique(source_dataset).tolist():
                source_id = int(source_id)
                source_row_mask = source_dataset == source_id
                for source_name, delta, visibility_mask in (
                    ("head", head_delta, flat_visible[:, 0]),
                    ("wrist", wrist_delta, flat_visible[:, 1]),
                    ("fused", fused_delta, flat_visible.any(dim=-1)),
                ):
                    selected = source_row_mask & near_mask & visibility_mask
                    if bool(selected.any()):
                        metric[
                            f"vision/{source_name}_xyz_error_mm/"
                            f"source_{source_id}/near_contact"
                        ] = float(
                            torch.linalg.vector_norm(
                                delta[selected].detach(), dim=-1
                            ).mean()
                            * 1000.0
                        )
                        metric[
                            f"vision/{source_name}_samples/"
                            f"source_{source_id}/near_contact"
                        ] = float(selected.sum())
                        for axis, axis_name in enumerate(("x", "y", "z")):
                            metric[
                                f"vision/{source_name}_error_{axis_name}_mm/"
                                f"source_{source_id}/near_contact"
                            ] = float(delta[selected, axis].detach().mean() * 1000.0)

        collection_mode_item = batch.get("collection_mode")
        policy_like_item = batch.get("policy_like_active")
        if collection_mode_item is not None:
            collection_mode = torch.as_tensor(
                collection_mode_item, dtype=torch.long, device=self.device
            ).reshape(-1)
            collection_names = (
                "nominal",
                "ee_offset",
                "near_hold",
                "bad_approach",
                "occlusion",
                "depth_noise",
                "closed_gripper",
                "recovery",
            )
            for mode_index, mode_name in enumerate(collection_names):
                mode_mask = collection_mode == mode_index
                metric[f"vision/collection_fraction/{mode_name}"] = float(
                    mode_mask.to(torch.float32).mean()
                )
                for source_name, delta, source_mask in (
                    ("head", head_delta, flat_visible[:, 0]),
                    ("wrist", wrist_delta, flat_visible[:, 1]),
                    ("fused", fused_delta, flat_visible.any(dim=-1)),
                ):
                    selected = mode_mask & source_mask
                    if bool(selected.any()):
                        metric[
                            f"vision/{source_name}_xyz_error_mm/mode_{mode_name}"
                        ] = float(
                            torch.linalg.vector_norm(
                                delta[selected].detach(), dim=-1
                            ).mean()
                            * 1000.0
                        )
                        metric[
                            f"vision/{source_name}_samples/mode_{mode_name}"
                        ] = float(selected.sum())
        if policy_like_item is not None:
            metric["vision/policy_like_sample_fraction"] = float(
                torch.as_tensor(
                    policy_like_item, dtype=torch.float32, device=self.device
                ).mean()
            )
        recovery_completed_item = batch.get("recovery_completed")
        recovery_success_item = batch.get("recovery_success")
        recovery_timeout_item = batch.get("recovery_timeout")
        if recovery_completed_item is not None and recovery_success_item is not None:
            completed = torch.as_tensor(
                recovery_completed_item, dtype=torch.bool, device=self.device
            ).reshape(-1)
            succeeded = torch.as_tensor(
                recovery_success_item, dtype=torch.bool, device=self.device
            ).reshape(-1)
            metric["vision/recovery_completion_fraction"] = float(
                completed.to(torch.float32).mean()
            )
            if bool(completed.any()):
                metric["vision/recovery_success_rate"] = float(
                    succeeded[completed].to(torch.float32).mean()
                )
            if recovery_timeout_item is not None:
                timed_out = torch.as_tensor(
                    recovery_timeout_item, dtype=torch.bool, device=self.device
                ).reshape(-1)
                metric["vision/recovery_timeout_rate"] = float(
                    timed_out[completed].to(torch.float32).mean()
                    if bool(completed.any())
                    else 0.0
                )

        frame_age = torch.as_tensor(
            batch["camera_frame_age_s"], dtype=torch.float32, device=self.device
        )
        metric["vision/head_frame_age_ms"] = float(frame_age[..., 0].mean() * 1000.0)
        metric["vision/wrist_frame_age_ms"] = float(frame_age[..., 1].mean() * 1000.0)
        metric["vision/head_wrist_timestamp_delta_ms"] = float(
            (frame_age[..., 0] - frame_age[..., 1]).abs().mean() * 1000.0
        )
        camera_world_transform = batch.get("camera_world_transform")
        if camera_world_transform is not None:
            metric.update(
                self._vision_geometry_sanity_metrics(
                    cube_position_root_m=cube,
                    camera_world_transform=torch.as_tensor(
                        camera_world_transform,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    head_prediction_root_m=perception[
                        "head_cube_xyz_root_m"
                    ].reshape(batch_size, steps, 3),
                    wrist_prediction_root_m=perception[
                        "wrist_cube_xyz_root_m"
                    ].reshape(batch_size, steps, 3),
                    visible=visible,
                )
            )
            metric["vision/camera_transform_diagnostic_available"] = 1.0
        else:
            metric["vision/camera_transform_diagnostic_available"] = 0.0
        return metric

    def _vision_geometry_sanity_metrics(
        self,
        *,
        cube_position_root_m: torch.Tensor,
        camera_world_transform: torch.Tensor,
        head_prediction_root_m: torch.Tensor,
        wrist_prediction_root_m: torch.Tensor,
        visible: torch.Tensor,
    ) -> dict[str, float]:
        """Audit transform ordering and center-vs-visible-surface bias.

        Camera transforms and privileged cube geometry are consumed only by
        this learner/evaluation diagnostic. No returned value enters actor
        observations, distance gating, actions, or replay state.
        """

        if camera_world_transform.shape[-3:] != (2, 4, 4):
            raise ValueError("camera world transforms must end in [2,4,4]")
        shape = cube_position_root_m.shape[:2]
        root_from_head = self.actor.policy.visual.root_from_head_optical_4x4.to(
            device=self.device, dtype=torch.float32
        )
        world_from_head = camera_world_transform[..., 0, :, :]
        world_from_wrist = camera_world_transform[..., 1, :, :]
        world_from_root = world_from_head @ torch.linalg.inv(root_from_head)

        def transform(matrix: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
            homogeneous = torch.cat(
                (point, torch.ones((*point.shape[:-1], 1), device=point.device)), dim=-1
            )
            return (matrix @ homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]

        cube_world = transform(world_from_root, cube_position_root_m)
        cube_wrist = transform(torch.linalg.inv(world_from_wrist), cube_world)
        cube_world_roundtrip = transform(world_from_wrist, cube_wrist)
        cube_root_roundtrip = transform(torch.linalg.inv(world_from_root), cube_world_roundtrip)
        transform_residual = torch.linalg.vector_norm(
            cube_root_roundtrip - cube_position_root_m, dim=-1
        )
        result = {
            "vision/wrist_to_base_gt_transform_residual_mean_mm": float(
                transform_residual.mean() * 1000.0
            ),
            "vision/wrist_to_base_gt_transform_residual_max_mm": float(
                transform_residual.max() * 1000.0
            ),
        }

        half_extent = torch.tensor(
            self.config.cube_size_m, device=self.device, dtype=torch.float32
        ) * 0.5
        for camera_index, (name, prediction) in enumerate(
            (("head", head_prediction_root_m), ("wrist", wrist_prediction_root_m))
        ):
            camera_from_root = torch.linalg.inv(camera_world_transform[..., camera_index, :, :]) @ world_from_root
            cube_camera = transform(camera_from_root, cube_position_root_m)
            prediction_camera = transform(camera_from_root, prediction)
            # OpenGL optical frame uses -Z forward; metric depth is therefore -z.
            signed_depth_bias = (-prediction_camera[..., 2]) - (-cube_camera[..., 2])
            rotation = camera_from_root[..., :3, :3]
            projected_half_extent = (rotation[..., 2, :].abs() * half_extent).sum(dim=-1)
            mask = visible[..., camera_index]
            if bool(mask.any()):
                selected_bias = signed_depth_bias[mask]
                selected_extent = projected_half_extent[mask]
                result[f"vision/{name}_optical_depth_bias_mm"] = float(
                    selected_bias.detach().mean() * 1000.0
                )
                result[f"vision/{name}_center_hypothesis_abs_error_mm"] = float(
                    selected_bias.detach().abs().mean() * 1000.0
                )
                result[f"vision/{name}_surface_hypothesis_abs_error_mm"] = float(
                    (selected_bias + selected_extent).detach().abs().mean() * 1000.0
                )
                result[f"vision/{name}_projected_half_extent_mm"] = float(
                    selected_extent.detach().mean() * 1000.0
                )
        return result

    @property
    def last_rollout_visual_diagnostics(self) -> Mapping[str, torch.Tensor]:
        """CPU rollout diagnostics captured before any learner forward pass."""

        return getattr(self, "_last_rollout_visual_diagnostics", {})

    @torch.no_grad()
    def select_actions(
        self,
        rgbd_u8,
        proprioception,
        hidden: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        deterministic_mask: torch.Tensor | np.ndarray | None = None,
        grasp_center_position_root_m: torch.Tensor | np.ndarray | None = None,
        phase_one_hot: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        # Environment collection/evaluation is inference, even while the
        # learner module otherwise remains in train mode.  In particular,
        # held-out or reference rollouts must not silently update ResNet
        # BatchNorm buffers and thereby mutate the actor outside an optimizer
        # step.  Restore the learner's mode before returning.
        actor_training_before_selection = self.actor.training
        self.actor.eval()
        try:
            rgbd = torch.as_tensor(rgbd_u8, dtype=torch.uint8, device=self.device)
            proprio = torch.as_tensor(
                proprioception, dtype=torch.float32, device=self.device
            )
            if grasp_center_position_root_m is None:
                raise ValueError(
                    "online actor requires live distal-pad midpoint FK; no fallback"
                )
            grasp_center = torch.as_tensor(
                grasp_center_position_root_m,
                dtype=torch.float32,
                device=self.device,
            )
            phase = (
                None
                if phase_one_hot is None
                else torch.as_tensor(
                    phase_one_hot, dtype=torch.float32, device=self.device
                )
            )
            features, next_hidden = self.actor.recurrent_features(
                rgbd,
                proprio,
                hidden.to(self.device),
                reset_mask,
                grasp_center,
                phase,
            )
            action, _, _, _, _ = self.actor.sample_from_features(
                features,
                deterministic=deterministic,
                deterministic_mask=deterministic_mask,
            )
            self._last_rollout_visual_diagnostics = {
                name: value.detach().cpu()
                for name, value in self.actor.policy.visual.last_perception.items()
                if name not in {"head_heatmap_logits", "head_cube_mask"}
            }
            return action.cpu().numpy(), next_hidden.detach()
        finally:
            self.actor.train(actor_training_before_selection)

    def behavior_clone(self, batch: Mapping[str, torch.Tensor | int]) -> dict[str, float]:
        """Warm-start the online Teacher actor from canonical keyboard data.

        This updates only the deployable RGB-D/proprioception/GRU actor path.
        Critics, target critics, alpha and online replay remain fresh.  The
        caller must supply episode-safe sequence windows produced by
        :class:`G2StudentSequenceContract`.
        """

        if self.config.require_frozen_visual_for_behavior_clone and any(
            parameter.requires_grad
            for parameter in self.actor.policy.visual.parameters()
        ):
            raise RuntimeError("G2_BC_STARTED_BEFORE_VISUAL_ENCODER_FREEZE")
        # BC owns its training-mode boundary.  In particular, this must run
        # after a validation event without inheriting that event's eval mode.
        self.set_controller_training_mode()
        diagnostic_first_call = not getattr(self, "_bc_first_call_diagnosed", False)

        def mark(stage: str) -> None:
            if diagnostic_first_call:
                print(f"G2_BC_FIRST_CALL_STAGE {stage}", flush=True)

        mark("batch_transfer_begin")

        def value(name: str, dtype=None) -> torch.Tensor:
            item = batch.get(name)
            if not isinstance(item, torch.Tensor):
                raise ValueError(f"teacher BC batch is missing tensor {name}")
            return item.to(device=self.device, dtype=dtype)

        original_rgbd = value("rgbd_u8", torch.uint8)
        mark("batch_transfer_rgbd_done")
        rgbd, principal_point_shift_px = random_shift_rgbd_sequences(
            original_rgbd,
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        mark("random_shift_done")
        proprio = value("deployable_proprioception", torch.float32)
        target = value("expert_action_target", torch.float32)
        teacher_state = value("teacher_state_target", torch.float32)
        confidence = value("expert_confidence", torch.float32)
        padding = value("padding_mask", torch.bool)
        reset = value("hidden_reset_mask", torch.bool)
        episode_id = value("episode_id", torch.long)
        sequence_step = value("sequence_step", torch.long)
        lengths = value("sequence_lengths", torch.long)
        G2RecurrentVisualStudent.validate_sequence_boundaries(
            episode_id, sequence_step, padding, lengths
        )
        mark("batch_validation_done")
        if target.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
            raise ValueError("teacher BC action target must be [B,T,7]")
        if confidence.shape != (*padding.shape, 1):
            raise ValueError("teacher BC confidence must be [B,T,1]")
        if teacher_state.shape != (*padding.shape, G2_VISUAL_PRIVILEGED_DIM):
            raise ValueError("teacher BC state target must be [B,T,59]")
        teacher_slices = G2TeacherObservationContract().slices
        burn_in = int(batch.get("burn_in_steps", self.config.burn_in_steps))
        if not 0 <= burn_in < padding.shape[1]:
            raise ValueError("teacher BC burn-in must be in [0,T)")

        recorded_grasp_center_item = batch.get("grasp_center_position_root_m")
        if not isinstance(recorded_grasp_center_item, torch.Tensor):
            raise ValueError(
                "teacher BC batch is missing distal-pad midpoint FK; "
                "EE-origin fallback is forbidden"
            )
        recorded_grasp_center = recorded_grasp_center_item.to(
            device=self.device, dtype=torch.float32
        )
        if recorded_grasp_center.shape != (*padding.shape, 3):
            raise ValueError("online BC grasp center must be [B,T,3]")
        stable_target_for_phase = batch.get("stable_grasp_target")
        stable_phase = (
            None
            if not isinstance(stable_target_for_phase, torch.Tensor)
            else stable_target_for_phase.to(device=self.device, dtype=torch.float32)
        )
        deployable_phase = deployable_phase_one_hot_from_recorded_state(
            teacher_state,
            stable_grasp=stable_phase,
            previous_action=proprio[
                ..., G2StudentObservationContract().slices["previous_action"]
            ],
        )
        mark("shifted_encoder_begin")
        features, _, projected_camera_tokens = self.actor.policy.encode_recurrent_features(
            rgbd, proprio, self.actor.initial_hidden(rgbd.shape[0], device=self.device),
            reset, padding, principal_point_shift_px,
            recorded_grasp_center,
            deployable_phase,
            return_camera_tokens=True,
        )
        mark("shifted_encoder_returned")
        mark("original_encoder_begin")
        original_features, _ = self.actor.policy.encode_recurrent_features(
            original_rgbd,
            proprio,
            self.actor.initial_hidden(rgbd.shape[0], device=self.device),
            reset,
            padding,
            grasp_center_position_root_m=recorded_grasp_center,
            phase_one_hot=deployable_phase,
        )
        mark("original_encoder_returned")
        predicted_cube_for_close = self.actor.policy.visual.last_perception[
            "head_cube_xyz_root_m"
        ].reshape(*padding.shape, 3)
        mask = padding.unsqueeze(-1).to(torch.float32)
        mask[:, :burn_in] = 0.0
        weight = mask * confidence.clamp(0.0, 1.0)
        denominator = weight.sum().clamp_min(1.0)
        arm_prediction = torch.tanh(self.actor.mean(features))
        active_arm_dimensions = (
            G2_VISUAL_ARM_ACTION_DIM
            if self.config.rotation_action_enabled
            else 3
        )
        if not self.config.rotation_action_enabled:
            arm_prediction = torch.cat(
                (arm_prediction[..., :3], torch.zeros_like(arm_prediction[..., 3:6])),
                dim=-1,
            )
        arm_loss = (
            weight
            * (
                arm_prediction[..., :active_arm_dimensions]
                - target[..., :active_arm_dimensions]
            )
            .square()
            .mean(dim=-1, keepdim=True)
        ).sum() / denominator
        reference_offset_item = batch.get("reference_grasp_offset_root_m")
        if not isinstance(reference_offset_item, torch.Tensor):
            reference_offset_item = teacher_state.new_tensor((0.0, 0.0, 0.0))
        contact_target = value("contact_target", torch.float32)
        future_safe_item = batch.get("future_safe_close_target")
        future_safe_close = (
            None
            if not isinstance(future_safe_item, torch.Tensor)
            else future_safe_item.to(device=self.device, dtype=torch.bool)
        )
        future_ready_item = batch.get("future_safe_close_ready_target")
        future_ready_valid_item = batch.get("future_safe_close_ready_valid")
        future_ready_authority = (
            torch.zeros_like(target[..., 6:7], dtype=torch.bool)
            if not isinstance(future_ready_item, torch.Tensor)
            or not isinstance(future_ready_valid_item, torch.Tensor)
            else (
                future_ready_item.to(device=self.device, dtype=torch.bool)
                & future_ready_valid_item.to(device=self.device, dtype=torch.bool)
            )
        )
        gripper_target_open, gripper_supervision = corrected_gripper_open_targets(
            expert_action=target,
            teacher_state=teacher_state,
            predicted_cube_position_root_m=predicted_cube_for_close,
            grasp_center_position_root_m=recorded_grasp_center,
            stable_grasp=stable_phase,
            contact=contact_target,
            reference_grasp_offset_root_m=reference_offset_item,
            grasp_ready_tolerance_m=self.config.grasp_ready_tolerance_m,
            future_safe_close=future_safe_close,
        )
        previous_gripper_open = (
            proprio[
                ...,
                G2StudentObservationContract().slices["previous_action"],
            ][..., G2_VISUAL_GRIPPER_ACTION_INDEX : G2_VISUAL_GRIPPER_ACTION_INDEX + 1]
            >= 0.0
        )
        supported_close, close_onset_support = expand_safe_close_onset_support(
            corrected_close=gripper_supervision["corrected_close"],
            previous_gripper_open=previous_gripper_open,
            grasp_ready=future_ready_authority,
            preceding_steps=self.config.demonstration_close_onset_support_steps,
        )
        close_onset = supported_close & previous_gripper_open
        # OPEN-vs-CLOSE_ONSET is a decision made only while the hand was
        # previously open.  HOLD_CLOSED rows are temporal context for the
        # GRU/latch, not negative examples for the onset classifier.
        safe_open = previous_gripper_open & ~supported_close
        gripper_target_open = safe_open.to(torch.float32)
        gripper_row_weight, row_quota = onset_open_row_quota_weights(
            base_weight=weight,
            close_onset=close_onset,
            safe_open=safe_open,
            close_onset_fraction=(
                self.config.demonstration_close_onset_row_fraction
            ),
        )
        quota_weight_sum = gripper_row_weight.sum().clamp_min(1.0e-12)
        quota_onset_fraction = (
            gripper_row_weight[close_onset].sum() / quota_weight_sum
        )
        quota_open_fraction = (
            gripper_row_weight[safe_open].sum() / quota_weight_sum
        )
        gripper_row_weight = gripper_row_weight * torch.where(
            close_onset,
            torch.full_like(
                gripper_row_weight,
                self.config.demonstration_close_onset_loss_weight,
            ),
            torch.ones_like(gripper_row_weight),
        )
        # Sampling reserves close-onset *windows*, whereas BCE is evaluated on
        # rows.  With T=16, one causal close row can still contribute less than
        # two percent of the batch and a fixed 8x multiplier is insufficient.
        # Raise (never lower) its aggregate weight to the configured fraction;
        # safe-open rows retain the complementary loss mass.
        target_close_fraction = float(
            self.config.demonstration_close_onset_target_loss_fraction
        )
        effective_close_scale = gripper_row_weight.new_ones(())
        if target_close_fraction > 0.0:
            close_weight_sum = gripper_row_weight[close_onset].sum()
            open_weight_sum = gripper_row_weight[~close_onset].sum()
            if bool((close_weight_sum > 0.0) & (open_weight_sum > 0.0)):
                required_close_weight = (
                    open_weight_sum
                    * target_close_fraction
                    / (1.0 - target_close_fraction)
                )
                effective_close_scale = torch.maximum(
                    gripper_row_weight.new_ones(()),
                    required_close_weight / close_weight_sum.clamp_min(1.0e-12),
                )
                gripper_row_weight = gripper_row_weight * torch.where(
                    close_onset,
                    effective_close_scale.expand_as(gripper_row_weight),
                    torch.ones_like(gripper_row_weight),
                )
        effective_close_loss_fraction = (
            gripper_row_weight[close_onset].sum()
            / gripper_row_weight.sum().clamp_min(1.0e-12)
        )
        gripper_loss = (
            gripper_row_weight
            * F.binary_cross_entropy_with_logits(
                self.actor.gripper_logit(features),
                gripper_target_open,
                reduction="none",
            )
        ).sum() / gripper_row_weight.sum().clamp_min(1.0)
        with torch.no_grad():
            predicted_open = self.actor.gripper_logit(features) >= 0.0
            onset_mask = mask.to(torch.bool) & close_onset
            safe_open_mask = mask.to(torch.bool) & (~supported_close)
            close_onset_recall = (
                (~predicted_open[onset_mask]).to(torch.float32).mean()
                if bool(onset_mask.any())
                else predicted_open.new_zeros((), dtype=torch.float32)
            )
            safe_open_recall = (
                predicted_open[safe_open_mask].to(torch.float32).mean()
                if bool(safe_open_mask.any())
                else predicted_open.new_zeros((), dtype=torch.float32)
            )

        relative_target = value("relative_pose_target", torch.float32)
        relative_prediction = self.actor.policy.relative_pose_head(features)
        relative_error = relative_pose_auxiliary_loss(
            relative_prediction,
            relative_target,
            reduction="none",
        ).unsqueeze(-1)
        relative_loss = (mask * relative_error).sum() / mask.sum().clamp_min(1.0)
        consistency_error = relative_pose_consistency_loss(
            self.actor.policy.relative_pose_head(original_features),
            self.actor.policy.relative_pose_head(features),
            rotation_weight=self.config.pose_consistency_rotation_weight,
            reduction="none",
        ).unsqueeze(-1)
        consistency_loss = (
            mask * consistency_error
        ).sum() / mask.sum().clamp_min(1.0)
        contact_error = F.binary_cross_entropy_with_logits(
            self.actor.policy.contact_head(features), contact_target, reduction="none"
        ).mean(dim=-1, keepdim=True)
        contact_loss = (mask * contact_error).sum() / mask.sum().clamp_min(1.0)
        depth_target = value("depth_validity_target", torch.float32)
        depth_error = F.binary_cross_entropy_with_logits(
            self.actor.policy.depth_validity_head(features),
            depth_target,
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        depth_loss = (mask * depth_error).sum() / mask.sum().clamp_min(1.0)
        future_ready_target = value("future_safe_close_ready_target", torch.float32)
        future_ready_valid = value("future_safe_close_ready_valid", torch.bool)
        future_ready_mask = mask.to(torch.bool) & future_ready_valid
        future_ready_logits = self.actor.policy.future_safe_close_ready_head(features)
        future_ready_loss = (
            F.binary_cross_entropy_with_logits(
                future_ready_logits[future_ready_mask],
                future_ready_target[future_ready_mask],
            )
            if bool(future_ready_mask.any())
            else future_ready_logits.sum() * 0.0
        )
        close_timing_target = value("close_timing_target", torch.float32)
        close_timing_valid = value("close_timing_valid", torch.bool)
        close_timing_mask = mask.to(torch.bool) & close_timing_valid
        close_timing_logits = self.actor.policy.close_timing_head(features)
        close_timing_loss = (
            F.binary_cross_entropy_with_logits(
                close_timing_logits[close_timing_mask],
                close_timing_target[close_timing_mask],
            )
            if bool(close_timing_mask.any())
            else close_timing_logits.sum() * 0.0
        )
        contact_lead_target = value("contact_lead_steps_target", torch.float32)
        contact_lead_valid = value("contact_lead_steps_valid", torch.bool)
        contact_lead_mask = mask.to(torch.bool) & contact_lead_valid
        predicted_contact_steps = F.softplus(self.actor.policy.contact_time_head(features))
        contact_time_loss = (
            F.smooth_l1_loss(
                predicted_contact_steps[contact_lead_mask]
                / float(self.config.contact_time_normalization_steps),
                contact_lead_target[contact_lead_mask]
                / float(self.config.contact_time_normalization_steps),
            )
            if bool(contact_lead_mask.any())
            else predicted_contact_steps.sum() * 0.0
        )
        flat_contrastive_rgbd = rgbd.reshape(-1, *G2_VISUAL_CAMERA_SHAPE)
        head_rgb, _, head_depth_valid = self.actor.policy.visual._unpack_camera(
            flat_contrastive_rgbd[:, 0]
        )
        wrist_rgb, _, wrist_depth_valid = self.actor.policy.visual._unpack_camera(
            flat_contrastive_rgbd[:, 1]
        )
        head_cube_visible = self.actor.policy.visual._red_cube_pixel_count(
            head_rgb, head_depth_valid
        ) >= self.actor.policy.visual.minimum_cube_pixels
        wrist_cube_visible = self.actor.policy.visual._red_cube_pixel_count(
            wrist_rgb, wrist_depth_valid
        ) >= self.actor.policy.visual.minimum_cube_pixels
        contrastive_valid = mask[..., 0].to(torch.bool) & (
            head_cube_visible & wrist_cube_visible
        ).reshape_as(mask[..., 0])
        # A zero-weight auxiliary objective has exactly zero contribution and
        # zero gradient. Avoid constructing its otherwise unused pairwise or
        # temporal graph while retaining an explicit zero metric.
        if self.config.cross_camera_contrastive_weight > 0.0:
            cross_camera_contrastive = cross_camera_contrastive_loss(
                projected_camera_tokens,
                contrastive_valid,
                relative_pose_target=relative_target,
                temperature=self.config.cross_camera_contrastive_temperature,
                false_negative_position_threshold_m=(
                    self.config.cross_camera_false_negative_position_threshold_m
                ),
            )
        else:
            cross_camera_contrastive = relative_prediction.new_zeros(())
        if self.config.temporal_pose_residual_weight > 0.0:
            temporal_pose_residual = temporal_pose_residual_loss(
                relative_prediction,
                relative_target,
                mask[..., 0].to(torch.bool),
                rotation_weight=self.config.temporal_pose_residual_rotation_weight,
            )
        else:
            temporal_pose_residual = relative_prediction.new_zeros(())
        if self.config.head_depth_reconstruction_weight > 0.0:
            head_depth_reconstruction, head_depth_mae_m = (
                self.actor.policy.visual.head_depth_reconstruction_loss(
                    rgbd.reshape(-1, *G2_VISUAL_CAMERA_SHAPE),
                    mask[..., 0].reshape(-1),
                )
            )
        else:
            head_depth_reconstruction = relative_prediction.new_zeros(())
            head_depth_mae_m = relative_prediction.new_zeros(())
        if self.config.head_depth_edge_weight > 0.0:
            head_depth_edge, head_depth_edge_target_mean = (
                self.actor.policy.visual.head_depth_edge_loss(
                    rgbd.reshape(-1, *G2_VISUAL_CAMERA_SHAPE),
                    mask[..., 0].reshape(-1),
                )
            )
        else:
            head_depth_edge = relative_prediction.new_zeros(())
            head_depth_edge_target_mean = relative_prediction.new_zeros(())
        cube_position = teacher_state[
            ..., teacher_slices["cube_pose_root_xyzw"]
        ][..., :3].reshape(-1, 3)
        head_detector_loss, wrist_position_loss, localization_metrics = (
            self.actor.policy.visual.cube_localization_supervision(
                cube_position,
                mask[..., 0].reshape(-1),
            )
        )
        bc_perception = self.actor.policy.visual.last_perception
        flat_bc_learning = mask[..., 0].reshape(-1).to(torch.bool)
        head_metric_mask = flat_bc_learning & bc_perception["head_valid"].reshape(-1)
        wrist_metric_mask = flat_bc_learning & bc_perception["wrist_valid"].reshape(-1)
        fused_metric_mask = flat_bc_learning & bc_perception["fused_valid"].reshape(-1)

        def valid_localization_mean(name: str, valid_mask: torch.Tensor) -> torch.Tensor:
            values = localization_metrics[name].reshape(-1)
            if bool(valid_mask.any()):
                return values[valid_mask].mean()
            return values.new_zeros(())

        head_position_error = valid_localization_mean(
            "head_position_error_m", head_metric_mask
        )
        wrist_position_error = valid_localization_mean(
            "wrist_position_error_m", wrist_metric_mask
        )
        fused_position_error = valid_localization_mean(
            "fused_position_error_m", fused_metric_mask
        )
        loss = (
            self.config.demonstration_arm_loss_weight * arm_loss
            + self.config.demonstration_gripper_loss_weight * gripper_loss
            + self.config.relative_pose_weight * relative_loss
            + self.config.pose_consistency_weight * consistency_loss
            + self.config.cross_camera_contrastive_weight
            * cross_camera_contrastive
            + self.config.temporal_pose_residual_weight * temporal_pose_residual
            + self.config.contact_weight * contact_loss
            + self.config.depth_validity_weight * depth_loss
            + self.config.head_depth_reconstruction_weight
            * head_depth_reconstruction
            + self.config.head_depth_edge_weight * head_depth_edge
            + self.config.head_cube_detector_weight * head_detector_loss
            + self.config.wrist_cube_position_weight * wrist_position_loss
            + self.config.grasp_ready_auxiliary_weight * future_ready_loss
            + self.config.close_timing_auxiliary_weight * close_timing_loss
            + self.config.contact_time_auxiliary_weight * contact_time_loss
        )
        weighted_bc_terms = {
            "arm": self.config.demonstration_arm_loss_weight * arm_loss,
            "gripper": self.config.demonstration_gripper_loss_weight * gripper_loss,
            "future_safe": self.config.grasp_ready_auxiliary_weight * future_ready_loss,
            "close_timing": self.config.close_timing_auxiliary_weight * close_timing_loss,
            "contact_time": self.config.contact_time_auxiliary_weight * contact_time_loss,
        }
        audited_auxiliary_heads = {
            "future_safe": self.actor.policy.future_safe_close_ready_head,
            "close_timing": self.actor.policy.close_timing_head,
            "contact_time": self.actor.policy.contact_time_head,
        }
        audit_parameter_update = not getattr(self, "_bc_first_call_diagnosed", False)
        auxiliary_hashes_before = (
            {
                name: g2_tensor_state_dict_sha256(module.state_dict())
                for name, module in audited_auxiliary_heads.items()
            }
            if audit_parameter_update else {}
        )
        mark("loss_assembled")
        self.actor_optimizer.zero_grad(set_to_none=True)
        mark("backward_begin")
        loss.backward()
        mark("backward_returned")

        def module_gradient_norm(module: nn.Module) -> float:
            squared = sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            return math.sqrt(squared)

        auxiliary_gradient_norms = {
            name: module_gradient_norm(module)
            for name, module in audited_auxiliary_heads.items()
        }
        gradient_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.gradient_clip
        )
        mark("optimizer_step_begin")
        self.actor_optimizer.step()
        mark("optimizer_step_returned")
        auxiliary_parameter_changed = (
            {
                name: float(
                    auxiliary_hashes_before[name]
                    != g2_tensor_state_dict_sha256(module.state_dict())
                )
                for name, module in audited_auxiliary_heads.items()
            }
            if audit_parameter_update else {name: -1.0 for name in audited_auxiliary_heads}
        )
        self._bc_first_call_diagnosed = True
        return {
            "demonstration_bc/loss_total": float(loss.detach()),
            "demonstration_bc/loss_arm": float(arm_loss.detach()),
            "demonstration_bc/loss_gripper": float(gripper_loss.detach()),
            "demonstration_bc/arm_loss_weight": float(
                self.config.demonstration_arm_loss_weight
            ),
            "demonstration_bc/gripper_loss_weight": float(
                self.config.demonstration_gripper_loss_weight
            ),
            "demonstration_bc/close_onset_loss_weight": float(
                self.config.demonstration_close_onset_loss_weight
            ),
            "demonstration_bc/close_onset_target_loss_fraction": float(
                self.config.demonstration_close_onset_target_loss_fraction
            ),
            "demonstration_bc/close_onset_row_quota_fraction": float(
                self.config.demonstration_close_onset_row_fraction
            ),
            "demonstration_bc/hold_closed_excluded_rows": float(
                row_quota["hold_closed_rows"].detach()
            ),
            "demonstration_bc/hold_closed_excluded_row_fraction": float(
                row_quota["hold_closed_rows"].detach()
                / mask.sum().clamp_min(1.0)
            ),
            "demonstration_bc/quota_open_weight_fraction": float(
                quota_open_fraction.detach()
            ),
            "demonstration_bc/quota_close_onset_weight_fraction": float(
                quota_onset_fraction.detach()
            ),
            "demonstration_bc/close_onset_effective_loss_fraction": float(
                effective_close_loss_fraction.detach()
            ),
            "demonstration_bc/close_onset_dynamic_scale": float(
                effective_close_scale.detach()
            ),
            "demonstration_bc/close_onset_support_steps": float(
                self.config.demonstration_close_onset_support_steps
            ),
            "demonstration_bc/close_onset_row_fraction": float(
                (mask * close_onset.to(mask.dtype)).sum()
                / mask.sum().clamp_min(1.0)
            ),
            "demonstration_bc/close_onset_support_row_fraction": float(
                (mask * close_onset_support.to(mask.dtype)).sum()
                / mask.sum().clamp_min(1.0)
            ),
            "demonstration_bc/close_onset_recall": float(close_onset_recall),
            "demonstration_bc/safe_open_recall": float(safe_open_recall),
            "demonstration_bc/premature_close_relabel_fraction": float(
                (
                    mask
                    * gripper_supervision["premature_close_relabel"].to(mask.dtype)
                ).sum()
                / mask.sum().clamp_min(1.0)
            ),
            "demonstration_bc/positive_close_fraction": float(
                (mask * gripper_supervision["corrected_close"].to(mask.dtype)).sum()
                / mask.sum().clamp_min(1.0)
            ),
            "demonstration_bc/loss_relative_pose": float(relative_loss.detach()),
            "demonstration_bc/loss_pose_consistency": float(consistency_loss.detach()),
            "demonstration_bc/loss_cross_camera_contrastive": float(
                cross_camera_contrastive.detach()
            ),
            "demonstration_bc/loss_temporal_pose_residual": float(
                temporal_pose_residual.detach()
            ),
            "demonstration_bc/loss_contact": float(contact_loss.detach()),
            "demonstration_bc/loss_depth_validity": float(depth_loss.detach()),
            "demonstration_bc/loss_future_safe": float(future_ready_loss.detach()),
            "demonstration_bc/loss_close_timing": float(close_timing_loss.detach()),
            "demonstration_bc/loss_contact_time": float(contact_time_loss.detach()),
            "demonstration_bc/weighted_arm": float(weighted_bc_terms["arm"].detach()),
            "demonstration_bc/weighted_gripper": float(weighted_bc_terms["gripper"].detach()),
            "demonstration_bc/weighted_future_safe": float(weighted_bc_terms["future_safe"].detach()),
            "demonstration_bc/weighted_close_timing": float(weighted_bc_terms["close_timing"].detach()),
            "demonstration_bc/weighted_contact_time": float(weighted_bc_terms["contact_time"].detach()),
            "demonstration_bc/gradient_future_safe_head": auxiliary_gradient_norms["future_safe"],
            "demonstration_bc/gradient_close_timing_head": auxiliary_gradient_norms["close_timing"],
            "demonstration_bc/gradient_contact_time_head": auxiliary_gradient_norms["contact_time"],
            "demonstration_bc/parameter_changed_future_safe_head": auxiliary_parameter_changed["future_safe"],
            "demonstration_bc/parameter_changed_close_timing_head": auxiliary_parameter_changed["close_timing"],
            "demonstration_bc/parameter_changed_contact_time_head": auxiliary_parameter_changed["contact_time"],
            "demonstration_bc/future_safe_valid_rows": float(future_ready_mask.sum().detach()),
            "demonstration_bc/close_timing_valid_rows": float(close_timing_mask.sum().detach()),
            "demonstration_bc/contact_time_valid_rows": float(contact_lead_mask.sum().detach()),
            "dynamics/predicted_contact_steps": float(
                predicted_contact_steps[contact_lead_mask].mean().detach()
                if bool(contact_lead_mask.any()) else 0.0
            ),
            "dynamics/actual_contact_steps": float(
                contact_lead_target[contact_lead_mask].mean().detach()
                if bool(contact_lead_mask.any()) else 0.0
            ),
            "dynamics/contact_step_prediction_error": float(
                (predicted_contact_steps[contact_lead_mask] - contact_lead_target[contact_lead_mask]).abs().mean().detach()
                if bool(contact_lead_mask.any()) else 0.0
            ),
            "demonstration_bc/loss_head_depth_reconstruction": float(
                head_depth_reconstruction.detach()
            ),
            "demonstration_bc/head_depth_reconstruction_mae_m": float(
                head_depth_mae_m.detach()
            ),
            "demonstration_bc/loss_head_depth_edge": float(
                head_depth_edge.detach()
            ),
            "demonstration_bc/head_depth_edge_target_mean": float(
                head_depth_edge_target_mean.detach()
            ),
            "demonstration_bc/loss_head_cube_detector": float(
                head_detector_loss.detach()
            ),
            "demonstration_bc/loss_wrist_cube_position": float(
                wrist_position_loss.detach()
            ),
            "demonstration_bc/head_cube_position_error_m": float(
                head_position_error
            ),
            "demonstration_bc/wrist_cube_position_error_m": float(
                wrist_position_error
            ),
            "demonstration_bc/fused_cube_position_error_m": float(
                fused_position_error
            ),
            "demonstration_bc/head_localization_valid_rows": float(
                head_metric_mask.sum()
            ),
            "demonstration_bc/wrist_localization_valid_rows": float(
                wrist_metric_mask.sum()
            ),
            "demonstration_bc/gradient_norm": float(gradient_norm),
            "demonstration_bc/supervised_rows": float(weight.sum().detach()),
        }

    @torch.no_grad()
    def evaluate_demonstration_close_gate(
        self,
        batch: Mapping[str, torch.Tensor | int],
        *,
        close_probability_threshold: float = 0.5,
        close_probability_thresholds: tuple[float, ...] | None = None,
    ) -> dict[str, float]:
        """Evaluate first-close and safe-open classification without updates.

        The post-vision BC loop previously reported these values only from its
        last training mini-batch.  This mirrors the deployable BC path on an
        excluded holdout so representation transfer and the rare first-close
        decision are attested before policy rollout or SAC.
        """

        if not 0.0 < close_probability_threshold < 1.0:
            raise ValueError("close probability threshold must be in (0,1)")
        if close_probability_thresholds is not None and any(
            not 0.0 < value < 1.0 for value in close_probability_thresholds
        ):
            raise ValueError("all close probability thresholds must be in (0,1)")

        def value(name: str, dtype: torch.dtype) -> torch.Tensor:
            item = batch.get(name)
            if not isinstance(item, torch.Tensor):
                raise ValueError(f"close gate batch is missing tensor {name}")
            return item.to(device=self.device, dtype=dtype)

        rgbd = value("rgbd_u8", torch.uint8)
        proprio = value("deployable_proprioception", torch.float32)
        target = value("expert_action_target", torch.float32)
        teacher_state = value("teacher_state_target", torch.float32)
        confidence = value("expert_confidence", torch.float32)
        padding = value("padding_mask", torch.bool)
        reset = value("hidden_reset_mask", torch.bool)
        if target.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
            raise ValueError("close gate action target must be [B,T,7]")
        burn_in = int(batch.get("burn_in_steps", self.config.burn_in_steps))
        if not 0 <= burn_in < padding.shape[1]:
            raise ValueError("close gate burn-in must be in [0,T)")

        recorded_grasp_center_item = batch.get("grasp_center_position_root_m")
        if not isinstance(recorded_grasp_center_item, torch.Tensor):
            raise ValueError(
                "close-gate batch is missing distal-pad midpoint FK; no fallback"
            )
        recorded_grasp_center = recorded_grasp_center_item.to(
            device=self.device, dtype=torch.float32
        )
        if recorded_grasp_center.shape != (*padding.shape, 3):
            raise ValueError("close-gate grasp center must be [B,T,3]")
        stable_item = batch.get("stable_grasp_target")
        stable = (
            None
            if not isinstance(stable_item, torch.Tensor)
            else stable_item.to(device=self.device, dtype=torch.float32)
        )
        previous_action = proprio[
            ..., G2StudentObservationContract().slices["previous_action"]
        ]
        phase = deployable_phase_one_hot_from_recorded_state(
            teacher_state,
            stable_grasp=stable,
            previous_action=previous_action,
        )

        actor_training = self.actor.training
        self.actor.eval()
        try:
            features, _ = self.actor.policy.encode_recurrent_features(
                rgbd,
                proprio,
                self.actor.initial_hidden(rgbd.shape[0], device=self.device),
                reset,
                padding,
                grasp_center_position_root_m=recorded_grasp_center,
                phase_one_hot=phase,
            )
            predicted_cube_for_close = self.actor.policy.visual.last_perception[
                "head_cube_xyz_root_m"
            ].reshape(*padding.shape, 3)
            reference_offset = batch.get("reference_grasp_offset_root_m")
            if not isinstance(reference_offset, torch.Tensor):
                reference_offset = teacher_state.new_tensor((0.0, 0.0, 0.0))
            contact = value("contact_target", torch.float32)
            future_safe_item = batch.get("future_safe_close_target")
            future_safe_close = (
                None
                if not isinstance(future_safe_item, torch.Tensor)
                else future_safe_item.to(device=self.device, dtype=torch.bool)
            )
            future_ready_item = batch.get("future_safe_close_ready_target")
            future_ready_valid_item = batch.get("future_safe_close_ready_valid")
            future_ready_authority = (
                torch.zeros_like(target[..., 6:7], dtype=torch.bool)
                if not isinstance(future_ready_item, torch.Tensor)
                or not isinstance(future_ready_valid_item, torch.Tensor)
                else (
                    future_ready_item.to(device=self.device, dtype=torch.bool)
                    & future_ready_valid_item.to(
                        device=self.device, dtype=torch.bool
                    )
                )
            )
            _, supervision = corrected_gripper_open_targets(
                expert_action=target,
                teacher_state=teacher_state,
                predicted_cube_position_root_m=predicted_cube_for_close,
                grasp_center_position_root_m=recorded_grasp_center,
                stable_grasp=stable,
                contact=contact,
                reference_grasp_offset_root_m=reference_offset,
                grasp_ready_tolerance_m=self.config.grasp_ready_tolerance_m,
                future_safe_close=future_safe_close,
            )
            previous_open = previous_action[
                ...,
                G2_VISUAL_GRIPPER_ACTION_INDEX : G2_VISUAL_GRIPPER_ACTION_INDEX + 1,
            ] >= 0.0
            supported_close, support = expand_safe_close_onset_support(
                corrected_close=supervision["corrected_close"],
                previous_gripper_open=previous_open,
                grasp_ready=future_ready_authority,
                preceding_steps=self.config.demonstration_close_onset_support_steps,
            )
            learning = padding.unsqueeze(-1) & (confidence > 0.0)
            learning[:, :burn_in] = False
            support_band = learning & supported_close & previous_open
            future_onset_item = batch.get("future_safe_close_onset_target")
            if isinstance(future_onset_item, torch.Tensor):
                exact_onset = learning & future_onset_item.to(
                    device=self.device, dtype=torch.bool
                )
            else:
                exact_onset = (
                    learning
                    & supervision["corrected_close"]
                    & previous_open
                )
            onset = exact_onset
            safe_open = learning & previous_open & (~supported_close)
            close_probability = torch.sigmoid(-self.actor.gripper_logit(features))
            predicted_open = close_probability <= float(
                close_probability_threshold
            )
            # Precision is a property of the open-vs-first-close decision.
            # Rows that correctly maintain an already closed hand are neither
            # onset positives nor safe-open negatives and must not enter its
            # denominator.  Counting them made a valid maintained grasp look
            # like thousands of false-positive close initiations.
            decision_rows = onset | safe_open
            predicted_close = (~predicted_open) & decision_rows
            close_true_positive = predicted_close & onset
            close_recall = (
                predicted_close[onset].to(torch.float32).mean()
                if bool(onset.any())
                else predicted_open.new_zeros((), dtype=torch.float32)
            )
            close_precision = (
                close_true_positive.to(torch.float32).sum()
                / predicted_close.to(torch.float32).sum().clamp_min(1.0)
            )
            open_recall = (
                predicted_open[safe_open].to(torch.float32).mean()
                if bool(safe_open.any())
                else predicted_open.new_zeros((), dtype=torch.float32)
            )
            support_only = support_band & (~exact_onset)
            support_band_recall = (
                (~predicted_open[support_band]).to(torch.float32).mean()
                if bool(support_band.any())
                else predicted_open.new_zeros((), dtype=torch.float32)
            )
            result = {
                "bc_heldout/close_onset_recall": float(close_recall),
                "bc_heldout/close_onset_precision": float(close_precision),
                "bc_heldout/safe_open_recall": float(open_recall),
                "bc_heldout/close_onset_rows": float(onset.sum()),
                "bc_heldout/safe_open_rows": float(safe_open.sum()),
                "bc_heldout/close_onset_correct_rows": float(
                    close_true_positive.sum()
                ),
                "bc_heldout/predicted_close_rows": float(predicted_close.sum()),
                "bc_heldout/safe_open_correct_rows": float(
                    (predicted_open & safe_open).sum()
                ),
                "bc_heldout/support_rows": float((learning & support).sum()),
                "bc_heldout/support_only_rows": float(support_only.sum()),
                "bc_heldout/support_band_rows": float(support_band.sum()),
                "bc_heldout/support_band_close_recall": float(
                    support_band_recall
                ),
                "bc_heldout/close_probability_threshold": float(
                    close_probability_threshold
                ),
            }
            if close_probability_thresholds is not None:
                for index, threshold in enumerate(close_probability_thresholds):
                    sweep_open = close_probability <= float(threshold)
                    sweep_close = (~sweep_open) & decision_rows
                    sweep_true_positive = sweep_close & onset
                    prefix = f"bc_heldout/sweep_{index}"
                    result[f"{prefix}/threshold"] = float(threshold)
                    result[f"{prefix}/true_positive_rows"] = float(
                        sweep_true_positive.sum()
                    )
                    result[f"{prefix}/predicted_close_rows"] = float(
                        sweep_close.sum()
                    )
                    result[f"{prefix}/safe_open_correct_rows"] = float(
                        (sweep_open & safe_open).sum()
                    )
            return result
        finally:
            self.actor.train(actor_training)

    @staticmethod
    def _hybrid_expectation(
        arm_action,
        arm_log_probability,
        gripper_logit,
        q1,
        q2,
        state,
    ):
        return G2VisualAsymmetricSAC._hybrid_expectation(
            arm_action,
            arm_log_probability,
            gripper_logit,
            q1,
            q2,
            state,
        )

    def _demonstration_regularization_loss(
        self,
        batch: Mapping[str, torch.Tensor | int],
        *,
        q_filter_reliability: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute actor-only, soft-Q-filtered BC on canonical sequences.

        The privileged Teacher state is consumed only by the frozen critics
        that calculate the Q filter.  Actor features use exactly the same
        RGB-D, 45-D deployable proprioception and GRU path as the Student.
        """

        if not 0.0 <= q_filter_reliability <= 1.0:
            raise ValueError("Q-filter reliability must be in [0,1]")

        def value(name: str, dtype=None) -> torch.Tensor:
            item = batch.get(name)
            if not isinstance(item, torch.Tensor):
                raise ValueError(f"online BC batch is missing tensor {name}")
            return item.to(device=self.device, dtype=dtype)

        rgbd, principal_point_shift_px = random_shift_rgbd_sequences(
            value("rgbd_u8", torch.uint8),
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        proprio = value("deployable_proprioception", torch.float32)
        expert_action = value("expert_action_target", torch.float32)
        confidence = value("expert_confidence", torch.float32)
        teacher_state = value("teacher_state_target", torch.float32)
        padding = value("padding_mask", torch.bool)
        reset = value("hidden_reset_mask", torch.bool)
        episode_id = value("episode_id", torch.long)
        sequence_step = value("sequence_step", torch.long)
        lengths = value("sequence_lengths", torch.long)
        G2RecurrentVisualStudent.validate_sequence_boundaries(
            episode_id, sequence_step, padding, lengths
        )
        if teacher_state.shape != (*padding.shape, G2_VISUAL_PRIVILEGED_DIM):
            raise ValueError("online BC teacher state must be [B,T,59]")
        if expert_action.shape != (*padding.shape, G2_VISUAL_ACTION_DIM):
            raise ValueError("online BC expert action must be [B,T,7]")
        teacher_slices = G2TeacherObservationContract().slices
        burn_in = int(batch.get("burn_in_steps", self.config.burn_in_steps))
        if not 0 <= burn_in < padding.shape[1]:
            raise ValueError("online BC burn-in must be in [0,T)")
        recorded_grasp_center_item = batch.get("grasp_center_position_root_m")
        if not isinstance(recorded_grasp_center_item, torch.Tensor):
            raise ValueError(
                "online demonstration batch is missing distal-pad midpoint FK; "
                "EE-origin fallback is forbidden"
            )
        recorded_grasp_center = recorded_grasp_center_item.to(
            device=self.device, dtype=torch.float32
        )
        if recorded_grasp_center.shape != (*padding.shape, 3):
            raise ValueError("teacher BC grasp center must be [B,T,3]")
        stable_item = batch.get("stable_grasp_target")
        stable_tensor = (
            None
            if not isinstance(stable_item, torch.Tensor)
            else stable_item.to(device=self.device, dtype=torch.float32)
        )
        deployable_phase = deployable_phase_one_hot_from_recorded_state(
            teacher_state,
            stable_grasp=stable_tensor,
            previous_action=proprio[
                ..., G2StudentObservationContract().slices["previous_action"]
            ],
        )
        reference_offset_item = batch.get("reference_grasp_offset_root_m")
        if not isinstance(reference_offset_item, torch.Tensor):
            reference_offset_item = teacher_state.new_tensor((0.0, 0.0, 0.0))
        contact_item = batch.get("contact_target")
        contact_tensor = (
            None
            if not isinstance(contact_item, torch.Tensor)
            else contact_item.to(device=self.device, dtype=torch.float32)
        )
        future_safe_item = batch.get("future_safe_close_target")
        future_safe_tensor = (
            None
            if not isinstance(future_safe_item, torch.Tensor)
            else future_safe_item.to(device=self.device, dtype=torch.bool)
        )
        future_ready_item = batch.get("future_safe_close_ready_target")
        future_ready_valid_item = batch.get("future_safe_close_ready_valid")
        future_ready_authority = (
            torch.zeros_like(padding.unsqueeze(-1), dtype=torch.bool)
            if not isinstance(future_ready_item, torch.Tensor)
            or not isinstance(future_ready_valid_item, torch.Tensor)
            else (
                future_ready_item.to(device=self.device, dtype=torch.bool)
                & future_ready_valid_item.to(device=self.device, dtype=torch.bool)
            )
        )
        previous_gripper_open = (
            proprio[
                ...,
                G2StudentObservationContract().slices["previous_action"],
            ][..., G2_VISUAL_GRIPPER_ACTION_INDEX : G2_VISUAL_GRIPPER_ACTION_INDEX + 1]
            >= 0.0
        )
        initial_hidden_item = batch.get("initial_recurrent_hidden")
        if initial_hidden_item is None:
            initial_hidden = self.actor.initial_hidden(
                rgbd.shape[0], device=self.device
            )
        elif isinstance(initial_hidden_item, torch.Tensor):
            initial_hidden_value = initial_hidden_item.to(
                device=self.device, dtype=torch.float32
            )
            if initial_hidden_value.shape != (rgbd.shape[0], self.config.hidden_dim):
                raise ValueError("online BC initial recurrent hidden must be [B,H]")
            initial_hidden = initial_hidden_value.unsqueeze(0)
        else:
            raise ValueError("online BC initial recurrent hidden must be a tensor")
        features, _ = self.actor.policy.encode_recurrent_features(
            rgbd,
            proprio,
            initial_hidden,
            reset,
            padding,
            principal_point_shift_px,
            recorded_grasp_center,
            deployable_phase,
        )
        predicted_cube_for_close = self.actor.policy.visual.last_perception[
            "head_cube_xyz_root_m"
        ].reshape(*padding.shape, 3)
        corrected_target_open, gripper_supervision = corrected_gripper_open_targets(
            expert_action=expert_action,
            teacher_state=teacher_state,
            predicted_cube_position_root_m=predicted_cube_for_close,
            grasp_center_position_root_m=recorded_grasp_center,
            stable_grasp=stable_tensor,
            contact=contact_tensor,
            reference_grasp_offset_root_m=reference_offset_item,
            grasp_ready_tolerance_m=self.config.grasp_ready_tolerance_m,
            future_safe_close=future_safe_tensor,
        )
        supported_close, close_onset_support = expand_safe_close_onset_support(
            corrected_close=gripper_supervision["corrected_close"],
            previous_gripper_open=previous_gripper_open,
            grasp_ready=future_ready_authority,
            preceding_steps=self.config.demonstration_close_onset_support_steps,
        )
        corrected_target_open = (~supported_close).to(torch.float32)
        corrected_expert_action = expert_action.clone()
        corrected_expert_action[..., 6:7] = torch.where(
            corrected_target_open > 0.5,
            torch.ones_like(corrected_target_open),
            -torch.ones_like(corrected_target_open),
        )
        learning = padding.clone()
        learning[:, :burn_in] = False
        flat_learning = learning.reshape(-1)
        flat_features = features.reshape(-1, self.config.hidden_dim)[flat_learning]
        flat_state = teacher_state.reshape(-1, G2_VISUAL_PRIVILEGED_DIM)[
            flat_learning
        ]
        flat_expert = corrected_expert_action.reshape(-1, G2_VISUAL_ACTION_DIM)[
            flat_learning
        ]
        flat_phase = deployable_phase.reshape(-1, G2_DEPLOYABLE_PHASE_DIM)[
            flat_learning
        ]
        flat_premature_relabel = gripper_supervision[
            "premature_close_relabel"
        ].reshape(-1, 1)[flat_learning]
        flat_close_onset = (
            supported_close & previous_gripper_open
        ).reshape(-1, 1)[flat_learning]
        flat_safe_open = (
            previous_gripper_open & (~supported_close)
        ).reshape(-1, 1)[flat_learning]
        flat_close_onset_support = close_onset_support.reshape(-1, 1)[flat_learning]
        active_arm_dimensions = (
            G2_VISUAL_ARM_ACTION_DIM
            if self.config.rotation_action_enabled
            else 3
        )
        if not self.config.rotation_action_enabled:
            flat_expert = torch.cat(
                (
                    flat_expert[:, :3],
                    torch.zeros_like(flat_expert[:, 3:6]),
                    flat_expert[:, 6:7],
                ),
                dim=-1,
            )
        flat_confidence = confidence.reshape(-1, 1)[flat_learning].clamp(0.0, 1.0)
        if flat_features.shape[0] == 0:
            raise ValueError("online BC batch has no post-burn-in rows")

        arm_prediction = torch.tanh(self.actor.mean(flat_features))
        if not self.config.rotation_action_enabled:
            arm_prediction = torch.cat(
                (arm_prediction[:, :3], torch.zeros_like(arm_prediction[:, 3:6])),
                dim=-1,
            )
        gripper_logit = self.actor.gripper_logit(flat_features)
        gripper_prediction = torch.where(
            gripper_logit >= 0.0,
            torch.ones_like(gripper_logit),
            -torch.ones_like(gripper_logit),
        )
        policy_action = torch.cat((arm_prediction, gripper_prediction), dim=-1)
        # Q is an eligibility/strength filter only.  Detaching prevents the
        # demonstration path from updating critics or exploiting Q gradients.
        with torch.no_grad():
            expert_q = torch.minimum(
                self.q1(flat_state, flat_expert), self.q2(flat_state, flat_expert)
            )
            policy_q = torch.minimum(
                self.q1(flat_state, policy_action.detach()),
                self.q2(flat_state, policy_action.detach()),
            )
            advantage = expert_q - policy_q
            raw_q_weight = torch.sigmoid(
                advantage / self.config.demonstration_q_filter_temperature
            )
            # Online-only critics are not allowed to veto expert labels until
            # held-out policy rollouts have demonstrated real task mastery.
            q_weight = (
                (1.0 - float(q_filter_reliability))
                + float(q_filter_reliability) * raw_q_weight
            )
        weight = flat_confidence * q_weight
        denominator = weight.sum().clamp_min(1.0)
        arm_loss = (
            weight
            * (
                arm_prediction[:, :active_arm_dimensions]
                - flat_expert[:, :active_arm_dimensions]
            )
            .square()
            .mean(dim=-1, keepdim=True)
        ).sum() / denominator
        gripper_target_open = flat_safe_open.to(torch.float32)
        gripper_weight, online_row_quota = onset_open_row_quota_weights(
            base_weight=weight,
            close_onset=flat_close_onset,
            safe_open=flat_safe_open,
            close_onset_fraction=(
                self.config.demonstration_close_onset_row_fraction
            ),
        )
        online_quota_weight_sum = gripper_weight.sum().clamp_min(1.0e-12)
        online_quota_onset_fraction = (
            gripper_weight[flat_close_onset].sum() / online_quota_weight_sum
        )
        online_quota_open_fraction = (
            gripper_weight[flat_safe_open].sum() / online_quota_weight_sum
        )
        gripper_weight = gripper_weight * torch.where(
            flat_close_onset,
            torch.full_like(
                gripper_weight,
                self.config.demonstration_close_onset_loss_weight,
            ),
            torch.ones_like(gripper_weight),
        )
        online_close_scale = gripper_weight.new_ones(())
        target_close_fraction = float(
            self.config.demonstration_close_onset_target_loss_fraction
        )
        if target_close_fraction > 0.0:
            close_weight_sum = gripper_weight[flat_close_onset].sum()
            open_weight_sum = gripper_weight[~flat_close_onset].sum()
            if bool((close_weight_sum > 0.0) & (open_weight_sum > 0.0)):
                required_close_weight = (
                    open_weight_sum
                    * target_close_fraction
                    / (1.0 - target_close_fraction)
                )
                online_close_scale = torch.maximum(
                    gripper_weight.new_ones(()),
                    required_close_weight / close_weight_sum.clamp_min(1.0e-12),
                )
                gripper_weight = gripper_weight * torch.where(
                    flat_close_onset,
                    online_close_scale.expand_as(gripper_weight),
                    torch.ones_like(gripper_weight),
                )
        online_effective_close_fraction = (
            gripper_weight[flat_close_onset].sum()
            / gripper_weight.sum().clamp_min(1.0e-12)
        )
        gripper_denominator = gripper_weight.sum().clamp_min(1.0)
        gripper_loss = (
            gripper_weight
            * F.binary_cross_entropy_with_logits(
                gripper_logit, gripper_target_open, reduction="none"
            )
        ).sum() / gripper_denominator
        # The stochastic Teacher uses distribution heads around the shared
        # recurrent backbone, whereas deployment calls the deterministic
        # Student heads on that same backbone.  Train both against the exact
        # canonical 7-D label so a Teacher checkpoint contains a usable
        # Student action surface instead of only compatible feature weights.
        student_arm = self.actor.policy.arm_action_head(flat_features)
        if not self.config.rotation_action_enabled:
            student_arm = torch.cat(
                (student_arm[:, :3], torch.zeros_like(student_arm[:, 3:6])), dim=-1
            )
        student_gripper = self.actor.policy.gripper_head(flat_features)
        student_arm_loss = (
            weight
            * (
                student_arm[:, :active_arm_dimensions]
                - flat_expert[:, :active_arm_dimensions]
            )
            .square()
            .mean(dim=-1, keepdim=True)
        ).sum() / denominator
        student_gripper_loss = (
            gripper_weight * (student_gripper - flat_expert[:, 6:7]).square()
        ).sum() / gripper_denominator
        student_compatibility_loss = student_arm_loss + student_gripper_loss
        head_detector_loss, wrist_position_loss, _ = (
            self.actor.policy.visual.cube_localization_supervision(
                teacher_state[..., teacher_slices["cube_pose_root_xyzw"]][..., :3].reshape(-1, 3),
                padding.reshape(-1).to(torch.float32),
            )
        )
        loss = (
            self.config.demonstration_arm_loss_weight * arm_loss
            + self.config.demonstration_gripper_loss_weight * gripper_loss
            + self.config.student_head_distillation_weight
            * student_compatibility_loss
            + self.config.head_cube_detector_weight * head_detector_loss
            + self.config.wrist_cube_position_weight * wrist_position_loss
        )
        with torch.no_grad():
            counterfactual = flat_expert.clone()
            counterfactual[:, 6:7] = -counterfactual[:, 6:7]
            counterfactual_q = torch.minimum(
                self.q1(flat_state, counterfactual),
                self.q2(flat_state, counterfactual),
            )
            ranking_delta = expert_q - counterfactual_q
            ranking_metrics: dict[str, float] = {
                "critic/q_expert_minus_premature": float(
                    ranking_delta[flat_premature_relabel[:, 0]].mean()
                )
                if bool(flat_premature_relabel.any())
                else 0.0,
                "critic/q_ranking_accuracy": float(
                    (ranking_delta > 0.0).to(torch.float32).mean()
                ),
            }
            for phase_index, phase_name in enumerate(G2_DEPLOYABLE_PHASES):
                phase_mask = flat_phase[:, phase_index] > 0.5
                ranking_metrics[
                    f"critic/q_expert_minus_counterfactual_{phase_name.lower()}"
                ] = (
                    float(ranking_delta[phase_mask].mean())
                    if bool(phase_mask.any())
                    else 0.0
                )
                ranking_metrics[
                    f"critic/q_ranking_accuracy_{phase_name.lower()}"
                ] = (
                    float((ranking_delta[phase_mask] > 0.0).to(torch.float32).mean())
                    if bool(phase_mask.any())
                    else 0.0
                )
        return loss, {
            "demonstration_online_bc/loss": float(loss.detach()),
            "demonstration_online_bc/loss_arm": float(arm_loss.detach()),
            "demonstration_online_bc/loss_gripper": float(gripper_loss.detach()),
            "demonstration_online_bc/arm_loss_weight": float(
                self.config.demonstration_arm_loss_weight
            ),
            "demonstration_online_bc/gripper_loss_weight": float(
                self.config.demonstration_gripper_loss_weight
            ),
            "demonstration_online_bc/close_onset_loss_weight": float(
                self.config.demonstration_close_onset_loss_weight
            ),
            "demonstration_online_bc/close_onset_support_steps": float(
                self.config.demonstration_close_onset_support_steps
            ),
            "demonstration_online_bc/close_onset_target_loss_fraction": float(
                self.config.demonstration_close_onset_target_loss_fraction
            ),
            "demonstration_online_bc/close_onset_row_quota_fraction": float(
                self.config.demonstration_close_onset_row_fraction
            ),
            "demonstration_online_bc/hold_closed_excluded_rows": float(
                online_row_quota["hold_closed_rows"].detach()
            ),
            "demonstration_online_bc/quota_open_weight_fraction": float(
                online_quota_open_fraction.detach()
            ),
            "demonstration_online_bc/quota_close_onset_weight_fraction": float(
                online_quota_onset_fraction.detach()
            ),
            "demonstration_online_bc/close_onset_effective_loss_fraction": float(
                online_effective_close_fraction.detach()
            ),
            "demonstration_online_bc/close_onset_dynamic_scale": float(
                online_close_scale.detach()
            ),
            "demonstration_online_bc/close_onset_row_fraction": float(
                flat_close_onset.to(torch.float32).mean()
            ),
            "demonstration_online_bc/close_onset_support_row_fraction": float(
                flat_close_onset_support.to(torch.float32).mean()
            ),
            "demonstration_online_bc/premature_close_relabel_fraction": float(
                flat_premature_relabel.to(torch.float32).mean()
            ),
            "demonstration_online_bc/loss_student_compatibility": float(
                student_compatibility_loss.detach()
            ),
            "demonstration_online_bc/loss_head_cube_detector": float(
                head_detector_loss.detach()
            ),
            "demonstration_online_bc/loss_wrist_cube_position": float(
                wrist_position_loss.detach()
            ),
            "demonstration_online_bc/q_advantage_mean": float(advantage.mean()),
            "demonstration_online_bc/q_filter_weight_mean": float(q_weight.mean()),
            "demonstration_online_bc/q_filter_raw_weight_mean": float(
                raw_q_weight.mean()
            ),
            "demonstration_online_bc/q_filter_reliability": float(
                q_filter_reliability
            ),
            "demonstration_online_bc/q_filter_expert_better_fraction": float(
                (advantage > 0.0).float().mean()
            ),
            "demonstration_online_bc/supervised_rows": float(
                (flat_confidence > 0.0).sum()
            ),
            **ranking_metrics,
        }

    def update(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        demonstration_batch: Mapping[str, torch.Tensor | int] | None = None,
        demonstration_bc_coefficient: float = 0.0,
        demonstration_q_filter_reliability: float = 1.0,
        update_actor: bool = True,
        update_alpha: bool = True,
        actor_update_scope: str = "full",
    ) -> dict[str, float]:
        if actor_update_scope not in G2_ACTOR_UPDATE_SCOPES:
            raise ValueError(
                f"actor update scope must be one of {G2_ACTOR_UPDATE_SCOPES}"
            )
        if update_alpha and not update_actor:
            raise ValueError("alpha update cannot precede actor update")
        gripper_only_update = bool(
            update_actor and actor_update_scope == "gripper_only"
        )
        action_heads_only_update = bool(
            update_actor and actor_update_scope == "action_heads_only"
        )
        bounded_actor_update = gripper_only_update or action_heads_only_update
        if bounded_actor_update and update_alpha:
            raise ValueError(
                f"{actor_update_scope.replace('_', '-')} actor update requires "
                "byte-frozen alpha"
            )
        # ``requires_grad=False``/skipping ``optimizer.step`` is not enough to
        # freeze an actor containing BatchNorm (the supported resnet13/18
        # profiles): train-mode forward passes still mutate running_mean and
        # running_var.  Critic-only deterministic-first warm-up promises a
        # byte-for-byte protected actor, so use inference-mode module
        # semantics for every actor forward in that update and restore the
        # caller's mode before returning.
        actor_training_before_update = self.actor.training
        actor_requires_grad_before_update = {
            name: parameter.requires_grad
            for name, parameter in self.actor.named_parameters()
        }
        protected_actor_state_before: dict[str, torch.Tensor] = {}
        alpha_before_update = self.log_alpha.detach().clone()
        if bounded_actor_update:
            # Bounded JOINT_TRAINING scopes expose an explicit parameter
            # allowlist.  eval() also makes train-mode buffers byte-stable for
            # every supported backbone; requires_grad alone would not protect
            # BatchNorm running statistics.
            mutable_prefixes = G2_ACTOR_UPDATE_MUTABLE_PREFIXES[
                actor_update_scope
            ]
            self.actor.eval()
            for name, parameter in self.actor.named_parameters():
                parameter.requires_grad_(name.startswith(mutable_prefixes))
            protected_actor_state_before = {
                name: value.detach().clone()
                for name, value in self.actor.state_dict().items()
                if not name.startswith(mutable_prefixes)
            }
        elif not update_actor:
            self.actor.eval()
            # Critic-only deterministic-first warm-up must not retain stale
            # actor/temperature gradients or construct an actor autograd graph.
            # ``eval()`` protects BatchNorm buffers, but it does not disable
            # autograd by itself.
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.alpha_optimizer.zero_grad(set_to_none=True)
        if demonstration_bc_coefficient < 0.0:
            raise ValueError("demonstration BC coefficient cannot be negative")
        if demonstration_bc_coefficient > 0.0 and demonstration_batch is None:
            raise ValueError("positive demonstration BC coefficient requires a batch")
        tensor = lambda name, dtype=torch.float32: torch.as_tensor(
            batch[name], dtype=dtype, device=self.device
        )
        rgbd, next_rgbd = tensor("rgbd", torch.uint8), tensor(
            "next_rgbd", torch.uint8
        )
        proprio, next_proprio = tensor("proprio"), tensor("next_proprio")
        state, next_state = tensor("privileged"), tensor("next_privileged")
        action, reward, terminated = (
            tensor("actions"),
            tensor("rewards"),
            tensor("terminated"),
        )
        # Preserve the legacy one-transition update for small unit diagnostics,
        # but production replay supplies [B,T,...] sequences.
        if rgbd.ndim == 5:
            rgbd, next_rgbd = rgbd.unsqueeze(1), next_rgbd.unsqueeze(1)
            proprio, next_proprio = proprio.unsqueeze(1), next_proprio.unsqueeze(1)
            state, next_state = state.unsqueeze(1), next_state.unsqueeze(1)
            action, reward, terminated = (
                action.unsqueeze(1), reward.unsqueeze(1), terminated.unsqueeze(1)
            )
        if rgbd.ndim != 6 or rgbd.shape != next_rgbd.shape:
            raise ValueError("recurrent SAC RGB-D replay must be [B,T,2,6,48,64]")
        batch_size, sequence_steps = rgbd.shape[:2]
        if proprio.shape[:2] != (batch_size, sequence_steps):
            raise ValueError("recurrent SAC proprioception sequence shape mismatch")
        burn_in = min(
            self.config.burn_in_steps if sequence_steps > 1 else 0,
            sequence_steps - 1,
        )
        hidden_value = tensor("recurrent_hidden")
        # Sequence replay returns only the state before its first observation;
        # legacy replay returns one state per sampled transition.
        if hidden_value.ndim == 3:
            hidden_value = hidden_value[:, 0]
        hidden = hidden_value.unsqueeze(0)
        if tuple(hidden.shape) != (1, batch_size, self.config.hidden_dim):
            raise ValueError("recurrent replay initial hidden-state shape mismatch")

        if "episode_id" in batch and "sequence_step" in batch:
            episode_id = tensor("episode_id", torch.long)
            sequence_step = tensor("sequence_step", torch.long)
            G2RecurrentVisualStudent.validate_sequence_boundaries(
                episode_id, sequence_step
            )

        # Build one recurrent chain: [s_0, s_1, ..., s_T].  Contiguous replay
        # guarantees next(s_t)==s_(t+1), while the final next state supplies
        # the bootstrap observation.  This gives the target action the exact
        # history implied by the sampled sequence.
        original_observation_chain = torch.cat((rgbd[:, :1], next_rgbd), dim=1)
        proprio_chain = torch.cat((proprio[:, :1], next_proprio), dim=1)
        if "grasp_center_position_root_m" not in batch or (
            "next_grasp_center_position_root_m" not in batch
        ):
            raise ValueError(
                "recurrent replay batch requires current/next distal-pad midpoint FK"
            )
        grasp_center = tensor("grasp_center_position_root_m")
        next_grasp_center = tensor("next_grasp_center_position_root_m")
        if grasp_center.ndim == 2:
            grasp_center = grasp_center.unsqueeze(1)
            next_grasp_center = next_grasp_center.unsqueeze(1)
        grasp_center_chain = torch.cat(
            (grasp_center[:, :1], next_grasp_center), dim=1
        )
        if "deployable_phase_one_hot" in batch:
            deployable_phase = tensor("deployable_phase_one_hot")
            next_deployable_phase = tensor("next_deployable_phase_one_hot")
        else:
            deployable_phase = deployable_phase_one_hot_from_recorded_state(state)
            next_deployable_phase = deployable_phase_one_hot_from_recorded_state(
                next_state
            )
        if deployable_phase.ndim == 2:
            deployable_phase = deployable_phase.unsqueeze(1)
            next_deployable_phase = next_deployable_phase.unsqueeze(1)
        deployable_phase_chain = torch.cat(
            (deployable_phase[:, :1], next_deployable_phase), dim=1
        )
        observation_chain, principal_point_shift_px = random_shift_rgbd_sequences(
            original_observation_chain,
            pad=self.config.random_shift_pad,
            return_principal_point_shift=True,
        )
        original_hidden = hidden.detach().clone() if update_actor else None
        if burn_in:
            with torch.no_grad():
                _, hidden = self.actor.policy.encode_recurrent_features(
                    observation_chain[:, :burn_in],
                    proprio_chain[:, :burn_in],
                    hidden,
                    torch.zeros(
                        (batch_size, burn_in), dtype=torch.bool, device=self.device
                    ),
                    torch.ones(
                        (batch_size, burn_in), dtype=torch.bool, device=self.device
                    ),
                    principal_point_shift_px[:, :burn_in],
                    grasp_center_chain[:, :burn_in],
                    deployable_phase_chain[:, :burn_in],
                )
            hidden = hidden.detach()
        learning_rgbd = observation_chain[:, burn_in:]
        learning_proprio = proprio_chain[:, burn_in:]
        recurrent_done = torch.zeros(
            learning_rgbd.shape[:2], dtype=torch.bool, device=self.device
        )
        recurrent_valid = torch.ones(
            learning_rgbd.shape[:2], dtype=torch.bool, device=self.device
        )
        if update_actor:
            all_features, _, projected_camera_tokens = (
                self.actor.policy.encode_recurrent_features(
                    learning_rgbd,
                    learning_proprio,
                    hidden,
                    recurrent_done,
                    recurrent_valid,
                    principal_point_shift_px[:, burn_in:],
                    grasp_center_chain[:, burn_in:],
                    deployable_phase_chain[:, burn_in:],
                    return_camera_tokens=True,
                )
            )
            shifted_perception = {
                name: value
                for name, value in self.actor.policy.visual.last_perception.items()
            }
        else:
            # The target policy still needs the recurrent next-state feature,
            # but critic warm-up must not build the visual/GRU actor graph or
            # materialize camera tokens used only by actor auxiliaries.
            with torch.no_grad():
                all_features, _ = self.actor.policy.encode_recurrent_features(
                    learning_rgbd,
                    learning_proprio,
                    hidden,
                    recurrent_done,
                    recurrent_valid,
                    principal_point_shift_px[:, burn_in:],
                    grasp_center_chain[:, burn_in:],
                    deployable_phase_chain[:, burn_in:],
                    return_camera_tokens=False,
                )
            projected_camera_tokens = None
        learning_features = all_features[:, :-1]
        features = learning_features.reshape(-1, self.config.hidden_dim)
        next_features = all_features[:, 1:].reshape(-1, self.config.hidden_dim)
        learning_state = state[:, burn_in:]
        if update_actor:
            if burn_in:
                with torch.no_grad():
                    _, original_hidden = self.actor.policy.encode_recurrent_features(
                        original_observation_chain[:, :burn_in],
                        proprio_chain[:, :burn_in],
                        original_hidden,
                        torch.zeros(
                            (batch_size, burn_in),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        torch.ones(
                            (batch_size, burn_in),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        grasp_center_position_root_m=grasp_center_chain[:, :burn_in],
                        phase_one_hot=deployable_phase_chain[:, :burn_in],
                    )
                original_hidden = original_hidden.detach()
            original_learning_rgbd = original_observation_chain[:, burn_in:]
            original_all_features, _ = self.actor.policy.encode_recurrent_features(
                original_learning_rgbd,
                proprio_chain[:, burn_in:],
                original_hidden,
                torch.zeros(
                    original_learning_rgbd.shape[:2],
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.ones(
                    original_learning_rgbd.shape[:2],
                    dtype=torch.bool,
                    device=self.device,
                ),
                grasp_center_position_root_m=grasp_center_chain[:, burn_in:],
                phase_one_hot=deployable_phase_chain[:, burn_in:],
            )
            original_features = original_all_features[:, :-1].reshape(
                -1, self.config.hidden_dim
            )
            relative_target_sequence = relative_pose_target_from_teacher_state(
                learning_state
            )
            relative_prediction_sequence = self.actor.policy.relative_pose_head(
                learning_features
            )
            temporal_valid = torch.ones(
                relative_target_sequence.shape[:2],
                dtype=torch.bool,
                device=self.device,
            )
            if terminated.shape[1] > burn_in + 1:
                temporal_valid[:, 1:] &= ~terminated[:, burn_in:-1, 0].to(
                    torch.bool
                )
            if self.config.temporal_pose_residual_weight > 0.0:
                temporal_pose_residual = temporal_pose_residual_loss(
                    relative_prediction_sequence,
                    relative_target_sequence,
                    temporal_valid,
                    rotation_weight=(
                        self.config.temporal_pose_residual_rotation_weight
                    ),
                )
            else:
                temporal_pose_residual = relative_prediction_sequence.new_zeros(())
            if self.config.cross_camera_contrastive_weight > 0.0:
                contrastive_valid = (
                    shifted_perception["head_valid"]
                    & shifted_perception["wrist_valid"]
                ).reshape(learning_rgbd.shape[0], learning_rgbd.shape[1])[:, :-1]
                cross_camera_contrastive = cross_camera_contrastive_loss(
                    projected_camera_tokens[:, :-1],
                    contrastive_valid,
                    relative_pose_target=relative_target_sequence,
                    temperature=self.config.cross_camera_contrastive_temperature,
                    false_negative_position_threshold_m=(
                        self.config.cross_camera_false_negative_position_threshold_m
                    ),
                )
            else:
                cross_camera_contrastive = relative_prediction_sequence.new_zeros(())
        state = learning_state.reshape(-1, G2_VISUAL_PRIVILEGED_DIM)
        next_state = next_state[:, burn_in:].reshape(-1, G2_VISUAL_PRIVILEGED_DIM)
        action = action[:, burn_in:].reshape(-1, G2_VISUAL_ACTION_DIM)
        if not self.config.rotation_action_enabled:
            # Replayed legacy demonstrations/checkpoints may contain RPY
            # residuals.  They are outside the translation+gripper MDP and
            # must not enter either critic when this authority is disabled.
            action = torch.cat(
                (action[:, :3], torch.zeros_like(action[:, 3:6]), action[:, 6:7]),
                dim=-1,
            )
        reward = reward[:, burn_in:].reshape(-1, 1)
        terminated = terminated[:, burn_in:].reshape(-1, 1)
        if update_actor:
            auxiliary_rgbd = learning_rgbd[:, :-1].reshape(
                -1, *G2_VISUAL_CAMERA_SHAPE
            )

        with torch.no_grad():
            next_features_target = next_features.detach()
            next_mean = self.actor.mean(next_features_target)
            next_log_std = -5.0 + 3.5 * (
                torch.tanh(self.actor.log_std(next_features_target)) + 1.0
            )
            active_arm_dimensions = (
                G2_VISUAL_ARM_ACTION_DIM
                if self.config.rotation_action_enabled
                else 3
            )
            next_active_mean = next_mean[:, :active_arm_dimensions]
            next_active_log_std = next_log_std[:, :active_arm_dimensions]
            # The bounded action-head repair is a deterministic update
            # contract, including its critic bootstrap.  It must not advance
            # Normal/Bernoulli RNG streams and must use exactly the Teacher
            # mean action plus the analytic binary-gripper expectation.
            if action_heads_only_update:
                next_active_pre_tanh = next_active_mean
            else:
                next_active_pre_tanh = torch.distributions.Normal(
                    next_active_mean, next_active_log_std.exp()
                ).rsample()
            next_active_arm = torch.tanh(next_active_pre_tanh)
            next_arm = (
                next_active_arm
                if self.config.rotation_action_enabled
                else torch.cat(
                    (next_active_arm, torch.zeros_like(next_mean[:, 3:6])), dim=-1
                )
            )
            next_arm_logp = squashed_gaussian_log_prob(
                next_active_pre_tanh, next_active_mean, next_active_log_std
            )
            next_gripper_logit = self.actor.gripper_logit(next_features_target)
            target_q, next_logp, _ = self._hybrid_expectation(
                next_arm,
                next_arm_logp,
                next_gripper_logit,
                self.tq1,
                self.tq2,
                next_state,
            )
            target = reward + self.config.gamma * (1.0 - terminated) * (
                target_q - self.alpha.detach() * next_logp
            )

        q1, q2 = self.q1(state, action), self.q2(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not update_actor:
            nonfinite_inputs = tuple(
                name
                for name, value in (
                    ("target", target),
                    ("q1", q1),
                    ("q2", q2),
                    ("critic_loss", critic_loss),
                )
                if not bool(torch.isfinite(value).all())
            )
            if nonfinite_inputs:
                self.actor.train(actor_training_before_update)
                raise FloatingPointError(
                    "G2_CRITIC_WARMUP_NONFINITE_FORWARD:"
                    + ",".join(nonfinite_inputs)
                )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        try:
            critic_norm = nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.config.gradient_clip,
                error_if_nonfinite=not update_actor,
            )
        except RuntimeError as error:
            if update_actor:
                raise
            self.actor.train(actor_training_before_update)
            raise FloatingPointError(
                "G2_CRITIC_WARMUP_NONFINITE_GRADIENT"
            ) from error
        self.critic_optimizer.step()

        if not update_actor:
            if not all(
                bool(torch.isfinite(parameter).all())
                for module in (self.q1, self.q2)
                for parameter in module.parameters()
            ):
                self.actor.train(actor_training_before_update)
                raise FloatingPointError(
                    "G2_CRITIC_WARMUP_NONFINITE_CRITIC_AFTER_STEP"
                )
            # Nothing below this branch contributes to a critic-only update.
            # Avoid constructing current-policy, auxiliary, reconstruction,
            # distillation, demonstration, and temperature graphs.
            polyak_update(self.q1, self.tq1, self.config.tau)
            polyak_update(self.q2, self.tq2, self.config.tau)
            if not all(
                bool(torch.isfinite(parameter).all())
                for module in (self.tq1, self.tq2)
                for parameter in module.parameters()
            ):
                self.actor.train(actor_training_before_update)
                raise FloatingPointError(
                    "G2_CRITIC_WARMUP_NONFINITE_TARGET_AFTER_UPDATE"
                )
            self.update_count += 1
            q_data = torch.cat((q1.detach(), q2.detach()), dim=-1)
            td_error = torch.cat(
                (target.detach() - q1.detach(), target.detach() - q2.detach()),
                dim=-1,
            )
            metrics = {
                "loss/critic": float(critic_loss.detach()),
                "gradient/critic_norm": float(critic_norm),
                "entropy/alpha": float(self.alpha.detach()),
                "q/data_min": float(q_data.min()),
                "q/data_mean": float(q_data.mean()),
                "q/data_max": float(q_data.max()),
                "q/q1_mean": float(q1.detach().mean()),
                "q/q2_mean": float(q2.detach().mean()),
                "q/target_mean": float(target.detach().mean()),
                "td_error/max_abs": float(td_error.abs().max()),
                "replay/sequence_length": float(sequence_steps),
                "replay/burn_in_steps": float(burn_in),
                "replay/learning_steps": float(sequence_steps - burn_in),
                "replay/augmentation_random_shift_pad": float(
                    self.config.random_shift_pad
                ),
                "update/actor_applied": 0.0,
                "update/alpha_applied": 0.0,
                "update/critic_applied": 1.0,
                "update/target_applied": 1.0,
                "update/actor_graph_evaluated": 0.0,
                "deterministic_first/critic_only_fast_path": 1.0,
                "demonstration_online_bc/loss": 0.0,
                "demonstration_online_bc/coefficient": 0.0,
                "demonstration_online_bc/weighted_loss": 0.0,
            }
            self.actor.train(actor_training_before_update)
            return metrics

        mutable_head_features = (
            features.detach() if action_heads_only_update else features
        )
        mean = self.actor.mean(mutable_head_features)
        log_std = -5.0 + 3.5 * (
            torch.tanh(self.actor.log_std(mutable_head_features)) + 1.0
        )
        active_arm_dimensions = (
            G2_VISUAL_ARM_ACTION_DIM
            if self.config.rotation_action_enabled
            else 3
        )
        active_mean = mean[:, :active_arm_dimensions]
        active_log_std = log_std[:, :active_arm_dimensions]
        if action_heads_only_update:
            active_pre_tanh = active_mean
        else:
            active_pre_tanh = torch.distributions.Normal(
                active_mean, active_log_std.exp()
            ).rsample()
        active_arm = torch.tanh(active_pre_tanh)
        arm = (
            active_arm
            if self.config.rotation_action_enabled
            else torch.cat((active_arm, torch.zeros_like(mean[:, 3:6])), dim=-1)
        )
        arm_logp = squashed_gaussian_log_prob(
            active_pre_tanh, active_mean, active_log_std
        )
        gripper_logit = self.actor.gripper_logit(mutable_head_features)
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(False)
        expected_q, logp, probability_open = self._hybrid_expectation(
            arm, arm_logp, gripper_logit, self.q1, self.q2, state
        )
        actor_loss = (self.alpha.detach() * logp - expected_q).mean()
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(True)

        policy = self.actor.policy
        previous_action_slice = G2StudentObservationContract().slices[
            "previous_action"
        ]
        previous_applied_arm = proprio[:, burn_in:].reshape(
            -1, G2_VISUAL_PROPRIO_DIM
        )[:, previous_action_slice][:, :G2_VISUAL_ARM_ACTION_DIM]
        deterministic_requested_arm = torch.tanh(mean)
        if not self.config.rotation_action_enabled:
            deterministic_requested_arm = torch.cat(
                (
                    deterministic_requested_arm[:, :3],
                    torch.zeros_like(deterministic_requested_arm[:, 3:6]),
                ),
                dim=-1,
            )
        deterministic_controller_target = (
            deterministic_requested_arm
            * float(self.config.maximum_arm_action_magnitude)
        )
        slew_excess = torch.relu(
            (deterministic_controller_target - previous_applied_arm).abs()
            - float(self.config.arm_action_slew_per_policy_step)
        )
        action_slew_excess_loss = slew_excess[:, :active_arm_dimensions].square().mean() / (
            float(self.config.maximum_arm_action_magnitude) ** 2
        )
        action_slew_excess_fraction = (
            slew_excess.detach().amax(dim=-1) > 0.0
        ).to(torch.float32).mean()
        relative_target = relative_target_sequence.reshape(
            -1, relative_target_sequence.shape[-1]
        )
        relative_prediction = relative_prediction_sequence.reshape(
            -1, relative_prediction_sequence.shape[-1]
        )
        original_relative_prediction = policy.relative_pose_head(original_features)
        relative_error = relative_pose_auxiliary_loss(
            relative_prediction, relative_target, reduction="none"
        )
        ee_cube_slice = G2TeacherObservationContract().slices[
            "end_effector_to_cube_m"
        ]
        near_contact = (
            torch.linalg.vector_norm(state[:, ee_cube_slice], dim=-1)
            <= self.config.near_contact_auxiliary_distance_m
        )
        relative_weight = torch.where(
            near_contact,
            torch.full_like(
                relative_error,
                self.config.near_contact_relative_pose_multiplier,
            ),
            torch.ones_like(relative_error),
        )
        relative_loss = (relative_error * relative_weight).sum() / relative_weight.sum()
        pose_consistency_loss = relative_pose_consistency_loss(
            original_relative_prediction,
            relative_prediction,
            rotation_weight=self.config.pose_consistency_rotation_weight,
        )
        contact_slice = G2TeacherObservationContract().slices[
            "bilateral_contact_features"
        ]
        contact_target = state[:, contact_slice].clamp(0.0, 1.0)
        contact_logits = policy.contact_head(features)
        contact_loss = F.binary_cross_entropy_with_logits(
            contact_logits, contact_target
        )
        depth_target = auxiliary_rgbd[:, :, 5].float().div(255.0).mean(dim=(-1, -2))
        depth_logits = policy.depth_validity_head(features)
        depth_loss = F.binary_cross_entropy_with_logits(depth_logits, depth_target)
        if self.config.head_depth_reconstruction_weight > 0.0:
            head_depth_reconstruction, head_depth_mae_m = (
                policy.visual.head_depth_reconstruction_loss(auxiliary_rgbd)
            )
        else:
            head_depth_reconstruction = depth_loss.new_zeros(())
            head_depth_mae_m = depth_loss.new_zeros(())
        if self.config.head_depth_edge_weight > 0.0:
            head_depth_edge, head_depth_edge_target_mean = (
                policy.visual.head_depth_edge_loss(auxiliary_rgbd)
            )
        else:
            head_depth_edge = depth_loss.new_zeros(())
            head_depth_edge_target_mean = depth_loss.new_zeros(())
        current_perception = {
            name: value.reshape(
                batch_size,
                learning_rgbd.shape[1],
                *value.shape[1:],
            )[:, :-1].reshape(-1, *value.shape[1:])
            for name, value in shifted_perception.items()
        }
        cube_slice = G2TeacherObservationContract().slices["cube_pose_root_xyzw"]
        head_detector_loss, wrist_position_loss, localization_metrics = (
            policy.visual.cube_localization_supervision(
                state[:, cube_slice][:, :3],
                perception=current_perception,
            )
        )
        auxiliary_loss = (
            self.config.relative_pose_weight * relative_loss
            + self.config.pose_consistency_weight * pose_consistency_loss
            + self.config.cross_camera_contrastive_weight
            * cross_camera_contrastive
            + self.config.temporal_pose_residual_weight * temporal_pose_residual
            + self.config.contact_weight * contact_loss
            + self.config.depth_validity_weight * depth_loss
            + self.config.head_depth_reconstruction_weight
            * head_depth_reconstruction
            + self.config.head_depth_edge_weight * head_depth_edge
            + self.config.head_cube_detector_weight * head_detector_loss
            + self.config.wrist_cube_position_weight * wrist_position_loss
        )
        # Continuously distil the current deterministic Teacher action into
        # the deployment Student heads.  Features and Teacher targets are
        # detached so this term cannot distort SAC's shared encoder/GRU; it
        # updates only the Student action heads carried in actor.policy.
        detached_features = features.detach()
        teacher_action_target = torch.cat(
            (deterministic_requested_arm.detach(), torch.tanh(gripper_logit.detach())),
            dim=-1,
        )
        student_arm_prediction = policy.arm_action_head(detached_features)
        if not self.config.rotation_action_enabled:
            student_arm_prediction = torch.cat(
                (
                    student_arm_prediction[:, :3],
                    torch.zeros_like(student_arm_prediction[:, 3:6]),
                ),
                dim=-1,
            )
            student_distillation_prediction = torch.cat(
                (student_arm_prediction[:, :3], policy.gripper_head(detached_features)),
                dim=-1,
            )
            student_distillation_target = torch.cat(
                (teacher_action_target[:, :3], teacher_action_target[:, 6:7]), dim=-1
            )
        else:
            student_distillation_prediction = torch.cat(
                (student_arm_prediction, policy.gripper_head(detached_features)), dim=-1
            )
            student_distillation_target = teacher_action_target
        student_head_distillation_loss = F.smooth_l1_loss(
            student_distillation_prediction, student_distillation_target
        )
        demonstration_loss = actor_loss.new_zeros(())
        demonstration_metrics: dict[str, float] = {
            "demonstration_online_bc/loss": 0.0,
            "demonstration_online_bc/coefficient": float(
                demonstration_bc_coefficient
            ),
        }
        if demonstration_batch is not None and demonstration_bc_coefficient > 0.0:
            demonstration_loss, computed = self._demonstration_regularization_loss(
                demonstration_batch,
                q_filter_reliability=demonstration_q_filter_reliability,
            )
            demonstration_metrics.update(computed)
        total_actor_loss = (
            actor_loss
            + auxiliary_loss
            + self.config.action_slew_excess_weight * action_slew_excess_loss
            + self.config.student_head_distillation_weight
            * student_head_distillation_loss
            + float(demonstration_bc_coefficient) * demonstration_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        if update_actor:
            total_actor_loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.gradient_clip
            )
            self.actor_optimizer.step()
        else:
            actor_norm = total_actor_loss.new_zeros(())

        alpha_loss = -(
            self.log_alpha
            * (logp.detach() + self.config.resolved_target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        if update_alpha:
            alpha_loss.backward()
            self.alpha_optimizer.step()
        protected_actor_tensor_match = True
        if bounded_actor_update:
            actor_state_after = self.actor.state_dict()
            protected_actor_tensor_match = all(
                torch.equal(value, actor_state_after[name])
                for name, value in protected_actor_state_before.items()
            )
            if not protected_actor_tensor_match:
                raise RuntimeError(
                    "G2_BOUNDED_ACTOR_UPDATE_MODIFIED_PROTECTED_ACTOR_TENSOR:"
                    f"{actor_update_scope}"
                )
            if not torch.equal(alpha_before_update, self.log_alpha.detach()):
                raise RuntimeError(
                    "G2_BOUNDED_ACTOR_UPDATE_MODIFIED_ALPHA:"
                    f"{actor_update_scope}"
                )
        polyak_update(self.q1, self.tq1, self.config.tau)
        polyak_update(self.q2, self.tq2, self.config.tau)
        self.update_count += 1
        q_data = torch.cat((q1.detach(), q2.detach()), dim=-1)
        td_error = torch.cat(
            (target.detach() - q1.detach(), target.detach() - q2.detach()),
            dim=-1,
        )
        metrics = {
            "loss/actor": float(actor_loss.detach()),
            "loss/actor_total": float(total_actor_loss.detach()),
            "loss/action_slew_excess": float(action_slew_excess_loss.detach()),
            "loss/action_slew_excess_weighted": float(
                (
                    self.config.action_slew_excess_weight
                    * action_slew_excess_loss
                ).detach()
            ),
            "action/predicted_slew_excess_row_fraction": float(
                action_slew_excess_fraction
            ),
            "loss/critic": float(critic_loss.detach()),
            "loss/visual_relative_pose": float(relative_loss.detach()),
            "loss/visual_pose_consistency": float(pose_consistency_loss.detach()),
            "loss/visual_cross_camera_contrastive": float(
                cross_camera_contrastive.detach()
            ),
            "loss/visual_temporal_pose_residual": float(
                temporal_pose_residual.detach()
            ),
            "loss/visual_contact": float(contact_loss.detach()),
            "loss/visual_depth_validity": float(depth_loss.detach()),
            "loss/visual_head_depth_reconstruction": float(
                head_depth_reconstruction.detach()
            ),
            "visual/head_depth_reconstruction_mae_m": float(
                head_depth_mae_m.detach()
            ),
            "loss/visual_head_depth_edge": float(head_depth_edge.detach()),
            "visual/head_depth_edge_target_mean": float(
                head_depth_edge_target_mean.detach()
            ),
            "loss/visual_head_cube_detector": float(head_detector_loss.detach()),
            "loss/visual_wrist_cube_position": float(wrist_position_loss.detach()),
            "visual/head_cube_position_error_m": float(
                localization_metrics["head_position_error_m"].detach().mean()
            ),
            "visual/wrist_cube_position_error_m": float(
                localization_metrics["wrist_position_error_m"].detach().mean()
            ),
            "visual/fused_cube_position_error_m": float(
                localization_metrics["fused_position_error_m"].detach().mean()
            ),
            "visual/head_validity_rate": float(
                current_perception["head_valid"].detach().to(torch.float32).mean()
            ),
            "visual/wrist_validity_rate": float(
                current_perception["wrist_valid"].detach().to(torch.float32).mean()
            ),
            "visual/wrist_gate_weight_mean": float(
                current_perception["wrist_gate_weight"].detach().mean()
            ),
            "visual/head_cube_pixels_mean": float(
                current_perception["head_cube_pixels"].detach().to(torch.float32).mean()
            ),
            "visual/head_frame_age_s_mean": float(
                current_perception["camera_frame_age_s"][:, 0].detach().mean()
            ),
            "visual/wrist_frame_age_s_mean": float(
                current_perception["camera_frame_age_s"][:, 1].detach().mean()
            ),
            "loss/teacher_student_action_head_distillation": float(
                student_head_distillation_loss.detach()
            ),
            "loss/alpha": float(alpha_loss.detach()),
            "entropy/alpha": float(self.alpha.detach()),
            "entropy/log_std_mean": float(log_std.detach().mean()),
            "entropy/gripper_open_probability_mean": float(
                probability_open.detach().mean()
            ),
            "gradient/actor_visual_gru_norm": float(actor_norm),
            "gradient/critic_norm": float(critic_norm),
            "update/actor_applied": float(update_actor),
            "update/actor_full_applied": float(
                update_actor and actor_update_scope == "full"
            ),
            "update/actor_gripper_only_applied": float(gripper_only_update),
            "update/actor_action_heads_only_applied": float(
                action_heads_only_update
            ),
            "update/actor_action_heads_only_deterministic": float(
                action_heads_only_update
            ),
            "update/actor_protected_tensor_match": float(
                protected_actor_tensor_match
            ),
            "update/alpha_applied": float(update_alpha),
            "q/data_min": float(q_data.min()),
            "q/data_mean": float(q_data.mean()),
            "q/data_max": float(q_data.max()),
            "q/q1_mean": float(q1.detach().mean()),
            "q/q2_mean": float(q2.detach().mean()),
            "q/policy_mean": float(expected_q.detach().mean()),
            "q/target_mean": float(target.detach().mean()),
            "td_error/max_abs": float(td_error.abs().max()),
            "replay/sequence_length": float(sequence_steps),
            "replay/burn_in_steps": float(burn_in),
            "replay/learning_steps": float(sequence_steps - burn_in),
            "replay/augmentation_random_shift_pad": float(
                self.config.random_shift_pad
            ),
            "visual/relative_position_error_m": float(
                torch.linalg.vector_norm(
                    relative_prediction.detach()[:, :3] - relative_target[:, :3],
                    dim=-1,
                ).mean()
            ),
            "visual/near_contact_fraction": float(near_contact.float().mean()),
            "visual/near_contact_relative_position_error_m": float(
                torch.linalg.vector_norm(
                    relative_prediction.detach()[:, :3] - relative_target[:, :3],
                    dim=-1,
                )[near_contact].mean()
                if bool(near_contact.any())
                else 0.0
            ),
        }
        metrics["demonstration_online_bc/weighted_loss"] = float(
            (
                float(demonstration_bc_coefficient) * demonstration_loss
            ).detach()
        )
        metrics.update(demonstration_metrics)
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(actor_requires_grad_before_update[name])
        self.actor.train(actor_training_before_update)
        return metrics

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "g2_recurrent_rgbd_sequence_burnin_asymmetric_sac_v6",
            "actor_cube_position_input_mode": (
                self.config.actor_cube_position_input_mode
            ),
            "shared_policy_contract": G2RecurrentVisualPolicyContract(
                hidden_dim=self.config.hidden_dim,
                gru_num_layers=self.config.gru_num_layers,
                sequence_length=self.config.sequence_length,
                burn_in_steps=self.config.burn_in_steps,
                sequence_stride=self.config.resolved_sequence_stride,
                camera_encoder_channels=self.config.camera_encoder_channels,
                head_observation_mode=self.config.head_observation_mode,
                head_depth_profile=self.config.head_depth_profile,
            ).validated().serializable(),
            "config": self.config.__dict__,
            "deterministic_gripper_close_probability_threshold": float(
                self.actor.deterministic_gripper_close_probability_threshold
            ),
            "actor": self.actor.state_dict(),
            "deployment_student": self.actor.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "tq1": self.tq1.state_dict(),
            "tq2": self.tq2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        *,
        load_optimizers: bool = True,
    ) -> None:
        """Restore a contract-compatible recurrent SAC agent.

        Replay and environment counters intentionally remain runtime-owned.
        This makes checkpoint initialization explicit instead of silently
        treating a stale replay directory as part of a new experiment.
        """

        self._validate_checkpoint_state_contract(state)
        self.actor.load_state_dict(state["actor"], strict=True)
        calibrated_threshold = float(
            state.get(
                "deterministic_gripper_close_probability_threshold",
                self.config.deterministic_gripper_close_probability_threshold,
            )
        )
        self.actor.deterministic_gripper_close_probability_threshold = (
            calibrated_threshold
        )
        self.q1.load_state_dict(state["q1"], strict=True)
        self.q2.load_state_dict(state["q2"], strict=True)
        self.tq1.load_state_dict(state["tq1"], strict=True)
        self.tq2.load_state_dict(state["tq2"], strict=True)
        self.tq1.requires_grad_(False)
        self.tq2.requires_grad_(False)
        with torch.no_grad():
            self.log_alpha.copy_(
                torch.as_tensor(state["log_alpha"], device=self.device)
            )
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.update_count = int(state.get("update_count", 0))

    def _validate_checkpoint_state_contract(
        self, state: Mapping[str, object]
    ) -> None:
        expected_schema = "g2_recurrent_rgbd_sequence_burnin_asymmetric_sac_v6"
        if state.get("schema") != expected_schema:
            raise ValueError("recurrent SAC checkpoint schema differs from runtime")
        if state.get(
            "actor_cube_position_input_mode", "relative_grasp_xyz"
        ) != self.config.actor_cube_position_input_mode:
            raise ValueError("actor cube-position input mode differs from checkpoint")
        saved_contract = state.get("shared_policy_contract")
        runtime_contract = G2RecurrentVisualPolicyContract(
            hidden_dim=self.config.hidden_dim,
            gru_num_layers=self.config.gru_num_layers,
            sequence_length=self.config.sequence_length,
            burn_in_steps=self.config.burn_in_steps,
            sequence_stride=self.config.resolved_sequence_stride,
            camera_encoder_channels=self.config.camera_encoder_channels,
            head_observation_mode=self.config.head_observation_mode,
            head_depth_profile=self.config.head_depth_profile,
        ).validated().serializable()
        if saved_contract != runtime_contract:
            raise ValueError("recurrent policy contract differs from checkpoint")

    def load_actor_only_state_dict(self, state: Mapping[str, object]) -> None:
        """Strictly initialize only the complete visual/recurrent actor.

        This intentionally leaves twin critics, target critics, entropy,
        every optimizer and ``update_count`` exactly as constructed by the
        current runtime.  It is the checkpoint-compatible initialization
        authority used by deterministic-first training: visual encoders,
        fusion, GRU, auxiliary/student heads and stochastic action heads all
        live below ``self.actor`` and therefore load together, strictly.
        """

        self._validate_checkpoint_state_contract(state)
        self.actor.load_state_dict(state["actor"], strict=True)
        calibrated_threshold = float(
            state.get(
                "deterministic_gripper_close_probability_threshold",
                self.config.deterministic_gripper_close_probability_threshold,
            )
        )
        self.actor.deterministic_gripper_close_probability_threshold = (
            calibrated_threshold
        )


class G2ReverseCurriculum:
    """Evaluation-success-gated GRASP_READY→PRE_GRASP→REACH mixture.

    Training outcomes are tracked separately for diagnostics.  Promotion must
    be driven by deterministic evaluation episodes; using noisy replay-
    collection episodes creates an optimistic, policy-dependent environment
    shift and does not match the COHER evaluation-rollout contract.
    """

    def __init__(self, *, initial_probability=.60, minimum_probability=.20, decrement=.05, window=100):
        self.beta = float(initial_probability); self.minimum = float(minimum_probability)
        self.decrement = float(decrement); self.window = int(window)
        self.history = deque(maxlen=self.window)
        self.training_history = deque(maxlen=self.window)
        self.promotion_count = 0

    def sample_near(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.random(count) < self.beta

    def record_episode(self, *, contact: bool, stable: bool, lift: bool) -> bool:
        return self.record_episodes(
            contact=[contact], stable=[stable], lift=[lift]
        )

    def record_training_episodes(self, *, contact, stable, lift) -> None:
        for values in zip(contact, stable, lift, strict=True):
            self.training_history.append(tuple(bool(value) for value in values))

    def record_evaluation_episodes(self, *, contact, stable, lift) -> bool:
        return self.record_episodes(contact=contact, stable=stable, lift=lift)

    @staticmethod
    def _rates(history: deque) -> tuple[float, float, float]:
        if not history:
            return (0.0, 0.0, 0.0)
        values = np.asarray(history, dtype=np.float32).mean(0)
        return tuple(float(value) for value in values)

    @property
    def evaluation_rates(self) -> tuple[float, float, float]:
        return self._rates(self.history)

    @property
    def training_rates(self) -> tuple[float, float, float]:
        return self._rates(self.training_history)

    def record_episodes(self, *, contact, stable, lift) -> bool:
        """Record one vector completion batch and promote at most once.

        Without this batch boundary, thousands of simultaneous completions
        can repeatedly clear/refill a short deque and collapse beta through
        every curriculum level in one environment step.
        """

        values = list(zip(contact, stable, lift, strict=True))
        for contact_value, stable_value, lift_value in values:
            self.history.append(
                (bool(contact_value), bool(stable_value), bool(lift_value))
            )
        if len(self.history) < self.window:
            return False
        rates = np.asarray(self.history, dtype=np.float32).mean(0)
        if rates[0] >= .50 and rates[1] >= .30 and rates[2] >= .20 and self.beta > self.minimum:
            self.beta = max(self.minimum, self.beta - self.decrement)
            self.history.clear()
            self.promotion_count += 1
            return True
        return False


class G2EpisodeReferenceBehavior:
    """Latch reference-vs-policy authority for complete episodes.

    Resampling authority at every transition lets a privileged reference
    rescue arbitrary fragments of a policy rollout.  The resulting contact
    counts cannot attest that either controller completed a coherent episode.
    This scheduler samples only at reset boundaries and keeps held-out
    evaluation environments policy-only.
    """

    schema = "g2_episode_latched_reference_behavior_v1"

    def __init__(
        self,
        training_mask,
        *,
        seed: int,
    ) -> None:
        mask = np.asarray(training_mask, dtype=np.bool_)
        if mask.ndim != 1 or mask.size == 0 or not bool(mask.any()):
            raise ValueError("reference behavior needs a non-empty training mask")
        self.training_mask = mask.copy()
        self.reference_mask = mask.copy()
        self.rng = np.random.default_rng(int(seed))

    def current_mask(self) -> np.ndarray:
        return self.reference_mask.copy()

    def reset_episodes(
        self,
        env_ids,
        *,
        learning_started: bool,
        reference_probability: float,
        force_policy_mask=None,
    ) -> None:
        probability = float(reference_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("reference probability must be in [0,1]")
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        forced = (
            np.zeros(ids.shape, dtype=np.bool_)
            if force_policy_mask is None
            else np.asarray(force_policy_mask, dtype=np.bool_).reshape(-1)
        )
        if forced.shape != ids.shape:
            raise ValueError("force-policy mask must align with reset environment ids")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.training_mask.size):
            raise IndexError("reference behavior environment index is out of range")
        for offset, env_id in enumerate(ids):
            if not bool(self.training_mask[env_id]):
                self.reference_mask[env_id] = False
            elif bool(forced[offset]) and learning_started:
                self.reference_mask[env_id] = False
            elif not learning_started:
                self.reference_mask[env_id] = True
            else:
                self.reference_mask[env_id] = bool(self.rng.random() < probability)

    def serializable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_sampling_boundary": "EPISODE_RESET_ONLY",
            "training_mask": self.training_mask.tolist(),
            "reference_mask": self.reference_mask.tolist(),
            "held_out_evaluation_policy_only": True,
        }


class G2VisualReplayBuffer:
    """RAM-bounded replay that keeps compact RGB-D as uint8."""

    @classmethod
    def required_storage_bytes(cls, capacity: int) -> int:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        per_row = (
            2 * int(np.prod(G2_VISUAL_CAMERA_SHAPE)) * np.dtype(np.uint8).itemsize
            + 2 * G2_VISUAL_PROPRIO_DIM * np.dtype(np.float32).itemsize
            + 2 * 3 * np.dtype(np.float32).itemsize
            + 2 * G2_VISUAL_PRIVILEGED_DIM * np.dtype(np.float32).itemsize
            + G2_VISUAL_ACTION_DIM * np.dtype(np.float32).itemsize
            + 3 * np.dtype(np.float32).itemsize
        )
        return int(capacity) * int(per_row)

    def __init__(self, capacity: int, *, seed: int = 42, storage_dir=None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.size, self.position = int(capacity), 0, 0
        if storage_dir is not None:
            from pathlib import Path
            root = Path(storage_dir); root.mkdir(parents=True, exist_ok=True)
            def allocate(name, shape, dtype):
                return np.memmap(root / f"{name}.mmap", mode="w+", shape=shape, dtype=dtype)
        else:
            allocate = lambda _name, shape, dtype: np.empty(shape, dtype=dtype)
        shape = (capacity, *G2_VISUAL_CAMERA_SHAPE)
        self.rgbd = allocate("rgbd", shape, np.uint8); self.next_rgbd = allocate("next_rgbd", shape, np.uint8)
        self.proprio = allocate("proprio", (capacity, G2_VISUAL_PROPRIO_DIM), np.float32)
        self.next_proprio = allocate("next_proprio", self.proprio.shape, np.float32)
        self.grasp_center_position_root_m = allocate(
            "grasp_center_position_root_m", (capacity, 3), np.float32
        )
        self.next_grasp_center_position_root_m = allocate(
            "next_grasp_center_position_root_m", (capacity, 3), np.float32
        )
        self.privileged = allocate("privileged", (capacity, G2_VISUAL_PRIVILEGED_DIM), np.float32)
        self.next_privileged = allocate("next_privileged", self.privileged.shape, np.float32)
        self.actions = allocate("actions", (capacity, G2_VISUAL_ACTION_DIM), np.float32)
        self.rewards = allocate("rewards", (capacity, 1), np.float32)
        self.terminated = allocate("terminated", (capacity, 1), np.float32)
        self.truncated = allocate("truncated", (capacity, 1), np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add_batch(self, **batch) -> None:
        count = len(batch["rewards"])
        required = {
            "rgbd", "next_rgbd", "proprio", "next_proprio",
            "grasp_center_position_root_m",
            "next_grasp_center_position_root_m",
            "privileged", "next_privileged", "actions", "rewards",
            "terminated", "truncated",
        }
        if set(batch) != required:
            raise ValueError(f"visual replay fields differ: {set(batch) ^ required}")
        if not 0 < count <= self.capacity:
            raise ValueError("visual replay batch must fit in replay capacity")
        actions = np.asarray(batch["actions"])
        if not np.all(np.isfinite(actions)):
            raise ValueError("visual replay actions contain non-finite values")
        if not np.all(np.isin(actions[:, G2_VISUAL_GRIPPER_ACTION_INDEX], (-1.0, 1.0))):
            raise ValueError("visual replay gripper actions must be exactly -1 or +1")
        indices = (np.arange(count, dtype=np.int64) + self.position) % self.capacity
        for name in required:
            source = np.asarray(batch[name])
            destination = getattr(self, name)
            if source.shape != (count, *destination.shape[1:]):
                raise ValueError(
                    f"visual replay {name} shape mismatch: {source.shape} != "
                    f"{(count, *destination.shape[1:])}"
                )
            destination[indices] = source
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not 0 < batch_size <= self.size:
            raise ValueError("invalid batch size")
        indices = self.rng.integers(0, self.size, batch_size)
        return {name: getattr(self, name)[indices] for name in (
            "rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
            "next_privileged", "grasp_center_position_root_m",
            "next_grasp_center_position_root_m", "actions", "rewards",
            "terminated", "truncated",
        )}

    def flush(self) -> None:
        for name in ("rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
                     "next_privileged", "grasp_center_position_root_m",
                     "next_grasp_center_position_root_m", "actions", "rewards",
                     "terminated", "truncated"):
            value = getattr(self, name)
            if isinstance(value, np.memmap):
                value.flush()


class G2RecurrentVisualReplayBuffer(G2VisualReplayBuffer):
    """Episode-safe recurrent replay with stored start states and burn-in."""

    @classmethod
    def required_storage_bytes(cls, capacity: int, hidden_dim: int = 128) -> int:
        if capacity <= 0 or hidden_dim <= 0:
            raise ValueError("recurrent replay capacity and hidden dimension must be positive")
        recurrent_per_row = (
            2 * int(hidden_dim) * np.dtype(np.float32).itemsize
            + 3 * np.dtype(np.int64).itemsize
            + 2 * G2_DEPLOYABLE_PHASE_DIM * np.dtype(np.float32).itemsize
            + np.dtype(np.int8).itemsize
            + np.dtype(np.float32).itemsize
            + np.dtype(np.bool_).itemsize
            + G2_VISUAL_ACTION_DIM * np.dtype(np.float32).itemsize
            + np.dtype(np.bool_).itemsize
            + np.dtype(np.int8).itemsize
        )
        return super().required_storage_bytes(capacity) + int(capacity) * int(
            recurrent_per_row
        )

    def __init__(
        self,
        capacity: int,
        *,
        hidden_dim: int = 128,
        seed: int = 42,
        storage_dir=None,
    ) -> None:
        super().__init__(capacity, seed=seed, storage_dir=storage_dir)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("recurrent replay hidden dimension must be positive")
        shape = (capacity, self.hidden_dim)
        if storage_dir is None:
            self.recurrent_hidden = np.empty(shape, dtype=np.float32)
            self.next_recurrent_hidden = np.empty(shape, dtype=np.float32)
            self.env_id = np.full((capacity,), -1, dtype=np.int64)
            self.episode_id = np.full((capacity,), -1, dtype=np.int64)
            self.sequence_step = np.full((capacity,), -1, dtype=np.int64)
            self.deployable_phase_one_hot = np.zeros(
                (capacity, G2_DEPLOYABLE_PHASE_DIM), dtype=np.float32
            )
            self.next_deployable_phase_one_hot = np.zeros(
                (capacity, G2_DEPLOYABLE_PHASE_DIM), dtype=np.float32
            )
        else:
            from pathlib import Path

            root = Path(storage_dir)
            self.recurrent_hidden = np.memmap(
                root / "recurrent_hidden.mmap",
                mode="w+",
                shape=shape,
                dtype=np.float32,
            )
            self.next_recurrent_hidden = np.memmap(
                root / "next_recurrent_hidden.mmap",
                mode="w+",
                shape=shape,
                dtype=np.float32,
            )
            self.env_id = np.memmap(
                root / "env_id.mmap", mode="w+", shape=(capacity,), dtype=np.int64
            )
            self.episode_id = np.memmap(
                root / "episode_id.mmap", mode="w+", shape=(capacity,), dtype=np.int64
            )
            self.sequence_step = np.memmap(
                root / "sequence_step.mmap", mode="w+", shape=(capacity,), dtype=np.int64
            )
            self.deployable_phase_one_hot = np.memmap(
                root / "deployable_phase_one_hot.mmap",
                mode="w+", shape=(capacity, G2_DEPLOYABLE_PHASE_DIM), dtype=np.float32,
            )
            self.next_deployable_phase_one_hot = np.memmap(
                root / "next_deployable_phase_one_hot.mmap",
                mode="w+", shape=(capacity, G2_DEPLOYABLE_PHASE_DIM), dtype=np.float32,
            )
            self.env_id[:] = self.episode_id[:] = self.sequence_step[:] = -1
        if storage_dir is None:
            self.phase_code = np.zeros((capacity,), dtype=np.int8)
            self.force_priority = np.ones((capacity,), dtype=np.float32)
            self.hindsight_relabel = np.zeros((capacity,), dtype=np.bool_)
            self.corrective_action = np.zeros(
                (capacity, G2_VISUAL_ACTION_DIM), dtype=np.float32
            )
            self.corrective_valid = np.zeros((capacity,), dtype=np.bool_)
            self.behavior_authority_code = np.zeros(
                (capacity,), dtype=np.int8
            )
        else:
            self.phase_code = np.memmap(
                root / "phase_code.mmap", mode="w+", shape=(capacity,), dtype=np.int8
            )
            self.force_priority = np.memmap(
                root / "force_priority.mmap", mode="w+", shape=(capacity,), dtype=np.float32
            )
            self.hindsight_relabel = np.memmap(
                root / "hindsight_relabel.mmap", mode="w+", shape=(capacity,), dtype=np.bool_
            )
            self.corrective_action = np.memmap(
                root / "corrective_action.mmap",
                mode="w+",
                shape=(capacity, G2_VISUAL_ACTION_DIM),
                dtype=np.float32,
            )
            self.corrective_valid = np.memmap(
                root / "corrective_valid.mmap",
                mode="w+",
                shape=(capacity,),
                dtype=np.bool_,
            )
            self.behavior_authority_code = np.memmap(
                root / "behavior_authority_code.mmap",
                mode="w+",
                shape=(capacity,),
                dtype=np.int8,
            )
            self.phase_code[:] = 0
            self.force_priority[:] = 1.0
            self.hindsight_relabel[:] = False
            self.corrective_action[:] = 0.0
            self.corrective_valid[:] = False
            self.behavior_authority_code[:] = G2_REPLAY_BEHAVIOR_AUTHORITY[
                "POLICY"
            ]
        self._sequence_index: dict[tuple[int, int, int], int] = {}
        self._phase_slots: tuple[set[int], ...] = tuple(
            set() for _ in G2_ONLINE_REPLAY_PHASES
        )

    def add_batch(self, **batch) -> None:
        count = len(batch["rewards"])
        phase_code = np.asarray(
            batch.pop("phase_code", np.zeros(count, dtype=np.int8)), dtype=np.int8
        )
        if "grasp_center_position_root_m" not in batch:
            raise ValueError(
                "online replay insert is missing current distal-pad midpoint FK"
            )
        if "next_grasp_center_position_root_m" not in batch:
            raise ValueError(
                "online replay insert is missing next distal-pad midpoint FK"
            )
        grasp_center = np.asarray(
            batch.pop("grasp_center_position_root_m"),
            dtype=np.float32,
        )
        next_grasp_center = np.asarray(
            batch.pop("next_grasp_center_position_root_m"),
            dtype=np.float32,
        )
        default_phase_index = np.asarray((0, 2, 3, 4), dtype=np.int64)[phase_code]
        default_phase = np.eye(G2_DEPLOYABLE_PHASE_DIM, dtype=np.float32)[
            default_phase_index
        ]
        deployable_phase = np.asarray(
            batch.pop("deployable_phase_one_hot", default_phase), dtype=np.float32
        )
        next_deployable_phase = np.asarray(
            batch.pop("next_deployable_phase_one_hot", deployable_phase),
            dtype=np.float32,
        )
        force_priority = np.asarray(
            batch.pop("force_priority", np.ones(count, dtype=np.float32)),
            dtype=np.float32,
        )
        hindsight_relabel = np.asarray(
            batch.pop("hindsight_relabel", np.zeros(count, dtype=np.bool_)),
            dtype=np.bool_,
        )
        corrective_action = np.asarray(
            batch.pop(
                "corrective_action",
                np.zeros((count, G2_VISUAL_ACTION_DIM), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        corrective_valid = np.asarray(
            batch.pop("corrective_valid", np.zeros(count, dtype=np.bool_)),
            dtype=np.bool_,
        )
        behavior_authority_code = np.asarray(
            batch.pop(
                "behavior_authority_code",
                np.full(
                    count,
                    G2_REPLAY_BEHAVIOR_AUTHORITY["POLICY"],
                    dtype=np.int8,
                ),
            ),
            dtype=np.int8,
        )
        recurrent = np.asarray(batch.pop("recurrent_hidden"), dtype=np.float32)
        next_recurrent = np.asarray(
            batch.pop("next_recurrent_hidden"), dtype=np.float32
        )
        env_id = np.asarray(
            batch.pop("env_id", np.arange(count, dtype=np.int64)), dtype=np.int64
        )
        episode_id = np.asarray(
            batch.pop("episode_id", np.zeros(count, dtype=np.int64)), dtype=np.int64
        )
        sequence_step = np.asarray(
            batch.pop("sequence_step", np.zeros(count, dtype=np.int64)), dtype=np.int64
        )
        expected = (count, self.hidden_dim)
        if recurrent.shape != expected or next_recurrent.shape != expected:
            raise ValueError(
                "recurrent replay hidden state must have shape " + str(expected)
            )
        if not np.isfinite(recurrent).all() or not np.isfinite(next_recurrent).all():
            raise ValueError("recurrent replay hidden state contains non-finite values")
        for name, value in (
            ("env_id", env_id),
            ("episode_id", episode_id),
            ("sequence_step", sequence_step),
        ):
            if value.shape != (count,) or np.any(value < 0):
                raise ValueError(f"recurrent replay {name} must be non-negative [N]")
        if phase_code.shape != (count,) or np.any(phase_code < 0) or np.any(
            phase_code >= len(G2_ONLINE_REPLAY_PHASES)
        ):
            raise ValueError("recurrent replay phase_code must be a valid [N] phase")
        if grasp_center.shape != (count, 3) or next_grasp_center.shape != (count, 3):
            raise ValueError("recurrent replay grasp center must be [N,3]")
        if deployable_phase.shape != (count, G2_DEPLOYABLE_PHASE_DIM) or next_deployable_phase.shape != (
            count, G2_DEPLOYABLE_PHASE_DIM
        ):
            raise ValueError(
                f"recurrent replay deployable phase must be [N,{G2_DEPLOYABLE_PHASE_DIM}]"
            )
        if not all(
            np.isfinite(value).all()
            for value in (
                grasp_center,
                next_grasp_center,
                deployable_phase,
                next_deployable_phase,
            )
        ):
            raise ValueError("recurrent replay deployable geometry is non-finite")
        for name, value in (
            ("deployable_phase_one_hot", deployable_phase),
            ("next_deployable_phase_one_hot", next_deployable_phase),
        ):
            binary = np.logical_or(np.isclose(value, 0.0), np.isclose(value, 1.0))
            if not np.all(binary) or not np.allclose(value.sum(axis=1), 1.0):
                raise ValueError(f"recurrent replay {name} must contain one-hot rows")
        if force_priority.shape != (count,) or not np.all(np.isfinite(force_priority)) or np.any(
            force_priority < 1.0
        ):
            raise ValueError("recurrent replay force_priority must be finite [N] >= 1")
        if hindsight_relabel.shape != (count,):
            raise ValueError("recurrent replay hindsight_relabel must be boolean [N]")
        if corrective_action.shape != (count, G2_VISUAL_ACTION_DIM):
            raise ValueError("recurrent replay corrective_action must be [N,7]")
        if corrective_valid.shape != (count,):
            raise ValueError("recurrent replay corrective_valid must be boolean [N]")
        if behavior_authority_code.shape != (count,) or not np.all(
            np.isin(
                behavior_authority_code,
                tuple(G2_REPLAY_BEHAVIOR_AUTHORITY.values()),
            )
        ):
            raise ValueError(
                "recurrent replay behavior_authority_code must be POLICY/REFERENCE/ASSISTED [N]"
            )
        if not np.isfinite(corrective_action).all():
            raise ValueError("recurrent replay corrective_action is non-finite")
        if np.any(np.abs(corrective_action[:, :G2_VISUAL_ARM_ACTION_DIM]) > 1.0):
            raise ValueError("corrective arm action must remain normalized in [-1,1]")
        valid_gripper = corrective_action[corrective_valid, G2_VISUAL_GRIPPER_ACTION_INDEX]
        if valid_gripper.size and not np.all(np.isin(valid_gripper, (-1.0, 1.0))):
            raise ValueError("valid corrective gripper action must be exactly -1 or +1")
        indices = (np.arange(count, dtype=np.int64) + self.position) % self.capacity
        # The parent advances ``position``; write the recurrent fields at the
        # same pre-advance indices first.
        self.recurrent_hidden[indices] = recurrent
        self.next_recurrent_hidden[indices] = next_recurrent
        for index in indices:
            for slots in self._phase_slots:
                slots.discard(int(index))
            old_key = (
                int(self.env_id[index]),
                int(self.episode_id[index]),
                int(self.sequence_step[index]),
            )
            if old_key[0] >= 0 and self._sequence_index.get(old_key) == int(index):
                del self._sequence_index[old_key]
        self.env_id[indices] = env_id
        self.episode_id[indices] = episode_id
        self.sequence_step[indices] = sequence_step
        self.phase_code[indices] = phase_code
        self.grasp_center_position_root_m[indices] = grasp_center
        self.next_grasp_center_position_root_m[indices] = next_grasp_center
        self.deployable_phase_one_hot[indices] = deployable_phase
        self.next_deployable_phase_one_hot[indices] = next_deployable_phase
        self.force_priority[indices] = force_priority
        self.hindsight_relabel[indices] = hindsight_relabel
        self.corrective_action[indices] = corrective_action
        self.corrective_valid[indices] = corrective_valid
        self.behavior_authority_code[indices] = behavior_authority_code
        for offset, index in enumerate(indices):
            key = (int(env_id[offset]), int(episode_id[offset]), int(sequence_step[offset]))
            if key in self._sequence_index:
                raise ValueError(f"duplicate recurrent replay sequence key: {key}")
            self._sequence_index[key] = int(index)
            self._phase_slots[int(phase_code[offset])].add(int(index))
        batch["grasp_center_position_root_m"] = grasp_center
        batch["next_grasp_center_position_root_m"] = next_grasp_center
        super().add_batch(**batch)

    def _sequence_for_end(self, end_index: int, length: int) -> list[int] | None:
        env = int(self.env_id[end_index])
        episode = int(self.episode_id[end_index])
        end_step = int(self.sequence_step[end_index])
        if env < 0 or episode < 0 or end_step < length - 1:
            return None
        indices = [
            self._sequence_index.get((env, episode, step))
            for step in range(end_step - length + 1, end_step + 1)
        ]
        if any(index is None for index in indices):
            return None
        resolved = [int(index) for index in indices]
        # A terminal transition may end a sampled sequence, never precede a
        # later transition in that same training sequence.
        if length > 1 and bool(np.asarray(self.terminated[resolved[:-1]]).any()):
            return None
        return resolved

    def has_sequences(self, length: int) -> bool:
        if length <= 0:
            raise ValueError("sequence length must be positive")
        return any(
            self._sequence_for_end(index, length) is not None
            for index in self._sequence_index.values()
        )

    def complete_sequence_endpoint_counts(
        self,
        length: int,
        *,
        phases: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Count unique, complete recurrent endpoints in selected phases.

        Stratified replay is allowed to sample with replacement and to fall
        back to another stratum.  Those conveniences must not be mistaken for
        physical coverage when deciding whether a protected actor may receive
        SAC gradients.  This method therefore reports actual unique endpoint
        evidence without materializing sequence payloads.
        """

        if length <= 0:
            raise ValueError("sequence length must be positive")
        selected = G2_ONLINE_REPLAY_PHASES if phases is None else tuple(phases)
        unknown = set(selected) - set(G2_ONLINE_REPLAY_PHASES)
        if unknown:
            raise ValueError(
                "unknown recurrent replay phases: " + ",".join(sorted(unknown))
            )
        result: dict[str, int] = {}
        for phase in selected:
            phase_index = G2_ONLINE_REPLAY_PHASES.index(phase)
            result[phase] = sum(
                self._sequence_for_end(index, length) is not None
                for index in self._phase_slots[phase_index]
            )
        return result

    def complete_sequence_episode_counts(
        self,
        length: int,
        *,
        phases: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Count distinct physical episodes contributing phase endpoints."""

        if length <= 0:
            raise ValueError("sequence length must be positive")
        selected = G2_ONLINE_REPLAY_PHASES if phases is None else tuple(phases)
        unknown = set(selected) - set(G2_ONLINE_REPLAY_PHASES)
        if unknown:
            raise ValueError(
                "unknown recurrent replay phases: " + ",".join(sorted(unknown))
            )
        result: dict[str, int] = {}
        for phase in selected:
            phase_index = G2_ONLINE_REPLAY_PHASES.index(phase)
            episodes = {
                (int(self.env_id[index]), int(self.episode_id[index]))
                for index in self._phase_slots[phase_index]
                if self._sequence_for_end(index, length) is not None
            }
            result[phase] = len(episodes)
        return result

    def sample_sequences(self, batch_size: int, length: int) -> dict[str, np.ndarray]:
        if batch_size <= 0 or length <= 0:
            raise ValueError("sequence batch size and length must be positive")
        # Do not rebuild an O(replay_capacity) candidate list for every SAC
        # update.  Physical replay slots are sampled with replacement and are
        # accepted only when their env/episode keys form a complete sequence.
        # Once the first ``length`` vector steps have been collected, the
        # acceptance rate is high while memory and CPU cost stay O(batch).
        slot_count = self.capacity if self.size == self.capacity else self.size
        sequences: list[list[int]] = []
        attempts = 0
        maximum_attempts = max(batch_size * 128, min(slot_count * 2, 1_000_000))
        while len(sequences) < batch_size and attempts < maximum_attempts:
            end_index = int(self.rng.integers(0, slot_count))
            sequence = self._sequence_for_end(end_index, length)
            if sequence is not None:
                sequences.append(sequence)
            attempts += 1
        if len(sequences) < batch_size:
            # Sparse early buffers need a deterministic bounded fallback.  It
            # exits as soon as enough sequences are found and never retains a
            # capacity-sized Python object between updates.
            for end_index in self._sequence_index.values():
                sequence = self._sequence_for_end(end_index, length)
                if sequence is not None:
                    sequences.append(sequence)
                    if len(sequences) >= batch_size:
                        break
        if not sequences:
            raise ValueError("recurrent replay has no complete episode-safe sequence")
        while len(sequences) < batch_size:
            sequences.append(sequences[int(self.rng.integers(0, len(sequences)))])
        indices = np.asarray(sequences[:batch_size])
        sequence_names = (
            "rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
            "next_privileged", "actions", "rewards", "terminated", "truncated",
            "next_recurrent_hidden", "env_id", "episode_id", "sequence_step",
            "behavior_authority_code",
            "grasp_center_position_root_m", "next_grasp_center_position_root_m",
            "deployable_phase_one_hot", "next_deployable_phase_one_hot",
        )
        result = {name: np.asarray(getattr(self, name)[indices]) for name in sequence_names}
        result["recurrent_hidden"] = np.asarray(self.recurrent_hidden[indices[:, 0]])
        return result

    def sample_sequences_stratified(
        self,
        batch_size: int,
        length: int,
        *,
        phase_probabilities: tuple[float, float, float, float],
        force_priority_mixture: float = 0.25,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Sample complete sequences by final physical phase.

        Contact/stable/lift endpoints may carry a bounded force-energy weight.
        A missing early-training stratum falls back to any complete sequence;
        realized rather than requested counts are returned for audit.
        """

        probability = np.asarray(phase_probabilities, dtype=np.float64)
        if (
            probability.shape != (len(G2_ONLINE_REPLAY_PHASES),)
            or np.any(probability < 0.0)
            or not np.isclose(probability.sum(), 1.0)
        ):
            raise ValueError(
                "online replay phase probabilities must be four non-negative values summing to one"
            )
        if batch_size <= 0 or length <= 0:
            raise ValueError("sequence batch size and length must be positive")
        if not 0.0 <= force_priority_mixture <= 1.0:
            raise ValueError("force priority mixture must be in [0,1]")
        requested = self.rng.choice(len(probability), size=batch_size, p=probability)
        sequences: list[list[int]] = []
        counts = {name: 0 for name in G2_ONLINE_REPLAY_PHASES}
        all_valid: list[list[int]] | None = None
        for requested_phase in requested:
            candidates = [
                index
                for index in self._phase_slots[int(requested_phase)]
                if self._sequence_for_end(index, length) is not None
            ]
            if candidates:
                force_weights = np.asarray(
                    [self.force_priority[index] for index in candidates],
                    dtype=np.float64,
                )
                force_probability = force_weights / force_weights.sum()
                uniform_probability = np.full(
                    len(candidates), 1.0 / len(candidates), dtype=np.float64
                )
                # Normal HER/phase replay remains the primary distribution.
                # Physical contact energy is a bounded local refinement, not
                # the sole authority inside CONTACT/STABLE/LIFT strata.
                weights = (
                    (1.0 - force_priority_mixture) * uniform_probability
                    + force_priority_mixture * force_probability
                )
                end_index = int(self.rng.choice(candidates, p=weights))
                sequence = self._sequence_for_end(end_index, length)
                assert sequence is not None
            else:
                if all_valid is None:
                    all_valid = [
                        sequence
                        for index in self._sequence_index.values()
                        if (sequence := self._sequence_for_end(index, length)) is not None
                    ]
                if not all_valid:
                    raise ValueError("recurrent replay has no complete episode-safe sequence")
                sequence = all_valid[int(self.rng.integers(0, len(all_valid)))]
            sequences.append(sequence)
            realized_phase = int(self.phase_code[sequence[-1]])
            counts[G2_ONLINE_REPLAY_PHASES[realized_phase]] += 1
        indices = np.asarray(sequences, dtype=np.int64)
        sequence_names = (
            "rgbd", "next_rgbd", "proprio", "next_proprio", "privileged",
            "next_privileged", "actions", "rewards", "terminated", "truncated",
            "next_recurrent_hidden", "env_id", "episode_id", "sequence_step",
            "phase_code", "force_priority", "hindsight_relabel",
            "behavior_authority_code",
            "grasp_center_position_root_m", "next_grasp_center_position_root_m",
            "deployable_phase_one_hot", "next_deployable_phase_one_hot",
        )
        result = {name: np.asarray(getattr(self, name)[indices]) for name in sequence_names}
        result["recurrent_hidden"] = np.asarray(self.recurrent_hidden[indices[:, 0]])
        return result, counts

    def sample_corrective_sequences(
        self,
        batch_size: int,
        length: int,
        *,
        require_close_gate_endpoint: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return complete sequences carrying audited online corrections.

        A label can be either a policy-visited privileged correction or a
        reference-executed action retained for the later joint-training
        stage.  Recurrent episode boundaries are preserved, but behavior
        authority may legitimately change from POLICY to ASSISTED within one
        episode.  Row-level replay provenance, rather than the sequence as a
        whole, remains the authority.
        """

        if batch_size <= 0 or length <= 0:
            raise ValueError("corrective sequence batch size and length must be positive")
        slot_count = self.capacity if self.size == self.capacity else self.size
        candidates: list[list[int]] = []
        attempts = 0
        maximum_attempts = max(batch_size * 256, min(slot_count * 2, 200_000))
        while len(candidates) < batch_size and attempts < maximum_attempts:
            end_index = int(self.rng.integers(0, slot_count))
            close_gate_endpoint = (
                not require_close_gate_endpoint
                or float(
                    self.corrective_action[
                        end_index, G2_VISUAL_GRIPPER_ACTION_INDEX
                    ]
                )
                < 0.0
            )
            if bool(self.corrective_valid[end_index]) and close_gate_endpoint:
                sequence = self._sequence_for_end(end_index, length)
                if sequence is not None and bool(self.corrective_valid[sequence].all()):
                    candidates.append(sequence)
            attempts += 1
        if not candidates:
            qualifier = (
                " close-gate"
                if require_close_gate_endpoint
                else ""
            )
            raise ValueError(
                f"recurrent replay has no complete{qualifier} corrective sequence"
            )
        while len(candidates) < batch_size:
            candidates.append(candidates[int(self.rng.integers(0, len(candidates)))])
        indices = np.asarray(candidates[:batch_size], dtype=np.int64)
        episode_id = np.asarray(self.episode_id[indices])
        sequence_step = np.asarray(self.sequence_step[indices])
        hidden_reset = np.zeros((batch_size, length), dtype=np.bool_)
        hidden_reset[:, 0] = sequence_step[:, 0] == 0
        result = {
            "rgbd_u8": np.asarray(self.rgbd[indices]),
            "deployable_proprioception": np.asarray(self.proprio[indices]),
            "expert_action_target": np.asarray(self.corrective_action[indices]),
            "expert_confidence": np.ones((batch_size, length, 1), dtype=np.float32),
            "teacher_state_target": np.asarray(self.privileged[indices]),
            "padding_mask": np.ones((batch_size, length), dtype=np.bool_),
            "hidden_reset_mask": hidden_reset,
            "env_id": np.asarray(self.env_id[indices]),
            "episode_id": episode_id,
            "sequence_step": sequence_step,
            "behavior_authority_code": np.asarray(
                self.behavior_authority_code[indices]
            ),
            "sequence_lengths": np.full((batch_size,), length, dtype=np.int64),
            "initial_recurrent_hidden": np.asarray(
                self.recurrent_hidden[indices[:, 0]]
            ),
            "grasp_center_position_root_m": np.asarray(
                self.grasp_center_position_root_m[indices]
            ),
            "deployable_phase_one_hot": np.asarray(
                self.deployable_phase_one_hot[indices]
            ),
        }
        return result

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not 0 < batch_size <= self.size:
            raise ValueError("invalid batch size")
        indices = self.rng.integers(0, self.size, batch_size)
        names = (
            "rgbd",
            "next_rgbd",
            "proprio",
            "next_proprio",
            "privileged",
            "next_privileged",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "recurrent_hidden",
            "next_recurrent_hidden",
            "grasp_center_position_root_m",
            "next_grasp_center_position_root_m",
            "deployable_phase_one_hot",
            "next_deployable_phase_one_hot",
        )
        return {name: getattr(self, name)[indices] for name in names}

    def flush(self) -> None:
        super().flush()
        for value in (
            self.recurrent_hidden,
            self.next_recurrent_hidden,
            self.env_id,
            self.episode_id,
            self.sequence_step,
            self.phase_code,
            self.force_priority,
            self.hindsight_relabel,
            self.corrective_action,
            self.corrective_valid,
            self.behavior_authority_code,
            self.grasp_center_position_root_m,
            self.next_grasp_center_position_root_m,
            self.deployable_phase_one_hot,
            self.next_deployable_phase_one_hot,
        ):
            if isinstance(value, np.memmap):
                value.flush()


__all__ = [name for name in globals() if name.startswith("G2_") or name in {"pack_rgbd"}]
