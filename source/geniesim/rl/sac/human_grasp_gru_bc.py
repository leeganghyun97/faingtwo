# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Human-only Wrist RGB-D GRU behavioral cloning for G2 grasp timing.

This module is deliberately simulator-free.  It consumes migrated human
demonstration sequences and never reads cube truth, pad geometry, contact
geometry, or planner state as an actor input.  The shared
``keyboard_grasp_contract`` owns time, frame, unit, action, and bound
semantics; this module only owns the offline recurrent learner.

The two supervision paths are intentionally independent:

* XYZ imitation uses only rows explicitly classified ``VALID_DIRECT`` by the
  migration pipeline.
* The close head uses its own valid temporal-label mask.  Neither path may
  include a corrupt row or padding.

Offline BC has no Isaac vector environments: ``num_envs == 0``.  Batch size
is an ordinary optimizer setting (default 256), not an environment count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .keyboard_grasp_contract import canonical_keyboard_grasp_contract
from .legacy_keyboard_dataset import EXPECTED_CONTROLLED_JOINT_ORDER


HUMAN_GRASP_GRU_BC_SCHEMA = (
    "g2_human_grasp_wrist_gru_bc_candidate_a_left_arm_down_v2_v3"
)
HUMAN_GRASP_GRU_CHECKPOINT_SCHEMA = (
    "g2_human_grasp_wrist_gru_bc_candidate_a_left_arm_down_v2_checkpoint_v3"
)
VALID_DIRECT_XYZ_MASK_AUTHORITY = "VALID_DIRECT"
CLOSE_STATE_TARGET = "HUMAN_CLOSE_STATE"
CLOSE_EDGE_TARGET = "HUMAN_CLOSE_EDGE"
DEFAULT_OFFLINE_BATCH_SIZE = 256
OFFLINE_NUM_ENVS = 0
RIGHT_ARM_DOF = 7
EE_POSE_DIM = 7
PREVIOUS_ACTION_DIM = 4
GRIPPER_STATE_DIM = 1
ROBOT_STATE_DIM = (
    EE_POSE_DIM
    + RIGHT_ARM_DOF
    + RIGHT_ARM_DOF
    + GRIPPER_STATE_DIM
    + PREVIOUS_ACTION_DIM
)

ACTOR_INPUT_FIELDS = (
    "right_wrist_rgb",
    "right_wrist_depth_m",
    "right_wrist_depth_valid",
    "ee_pose_robot_root_m_xyzw",
    "right_arm_joint_position_rad",
    "right_arm_joint_velocity_rad_s",
    "current_gripper_state",
    "previous_policy_action_4d_metric_root_m",
)
PROHIBITED_ACTOR_INPUT_TOKENS = (
    "touch",
    "contact",
    "force",
    "cube_gt",
    "cube_pose",
    "cube_center",
    "pad_pose",
    "pad_distance",
    "pad_surface",
    "pad_midpoint",
    "relative_grasp",
    "stable",
    "grasp_success",
    "privileged",
    "planner_target",
)
MIGRATED_INPUT_SOURCE_FIELDS = (
    "right_wrist_rgb",
    "right_wrist_depth",
    "right_wrist_depth_valid",
)


class HumanGraspBCContractError(ValueError):
    """Raised before incompatible data can reach the learner."""


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise HumanGraspBCContractError(f"{name} contains NaN/Inf")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class HumanGraspGRUConfig:
    """Architecture plus explicit offline-loss scale configuration."""

    vision_feature_dim: int = 64
    state_feature_dim: int = 64
    fusion_dim: int = 128
    gru_hidden_dim: int = 128
    gru_layers: int = 1
    encoder_image_height: int = 48
    encoder_image_width: int = 64
    maximum_depth_m: float = 2.0
    lambda_xyz: float = 1.0
    lambda_close: float = 1.0
    lambda_privileged_feasibility: float = 1.0
    batch_size: int = DEFAULT_OFFLINE_BATCH_SIZE
    num_envs: int = OFFLINE_NUM_ENVS

    def __post_init__(self) -> None:
        for name in (
            "vision_feature_dim",
            "state_feature_dim",
            "fusion_dim",
            "gru_hidden_dim",
            "gru_layers",
            "encoder_image_height",
            "encoder_image_width",
            "batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise HumanGraspBCContractError(f"{name} must be a positive integer")
        for name in (
            "maximum_depth_m",
            "lambda_xyz",
            "lambda_close",
            "lambda_privileged_feasibility",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise HumanGraspBCContractError(f"{name} must be finite and positive")
        if self.num_envs != OFFLINE_NUM_ENVS:
            raise HumanGraspBCContractError(
                "offline BC num_envs is 0/NOT_APPLICABLE; batch size is not num_envs"
            )
        if (self.encoder_image_height, self.encoder_image_width) != (48, 64):
            raise HumanGraspBCContractError(
                "Wrist encoder resolution must match the legacy 48x64 policy contract"
            )
        if self.maximum_depth_m != canonical_keyboard_grasp_contract().maximum_policy_depth_m:
            raise HumanGraspBCContractError(
                "maximum_depth_m must match the frozen deployable policy depth contract"
            )

    def contract_payload(self) -> dict[str, Any]:
        contract = canonical_keyboard_grasp_contract()
        return {
            "schema": HUMAN_GRASP_GRU_BC_SCHEMA,
            "shared_contract": contract.as_dict(),
            "actor_input_fields": list(ACTOR_INPUT_FIELDS),
            "prohibited_actor_input_tokens": list(PROHIBITED_ACTOR_INPUT_TOKENS),
            "baseline_variant": "LEGACY_CANDIDATE_A_LEFT_ARM_DOWN_V2",
            "recorded_camera_names": ["head", "right_wrist"],
            "grasp_actor_camera_names": ["right_wrist"],
            "recorded_non_actor_camera_names": ["head"],
            "wrist_rgbd_only": True,
            "head_camera": False,
            "robot_state_dim": ROBOT_STATE_DIM,
            "visual_input_channels": 5,
            "encoder_image_hw": [
                self.encoder_image_height,
                self.encoder_image_width,
            ],
            "rgb_resize": "bilinear_align_corners_false",
            "depth_and_valid_resize": "nearest",
            "gru_input_dim": self.fusion_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
            "xyz_output_dim": 3,
            "close_output_dim": 1,
            "feasibility_output_dim": 1,
            "action_output": "[dx,dy,dz,p_close]",
            "auxiliary_output": "p_feasible",
            "xyz_frame": contract.geometry_frame,
            "xyz_unit": contract.position_unit,
            "depth_raw_unit": contract.depth_raw_unit,
            "depth_to_meter_scale": contract.depth_to_meter_scale,
            "joint_angle_unit": contract.angle_unit,
            "gripper_state_encoding": "OPEN=0,CLOSED=1",
            "control_hz": contract.control_hz,
            "policy_dt_s": contract.policy_dt_s,
            "rgbd_acquisition_hz": contract.rgbd_acquisition_hz,
            "rgbd_frame_reuse": "TWO_50HZ_CONTROL_ROWS_MAY_REFERENCE_ONE_25HZ_FRAME",
            "fixed_orientation": contract.fixed_orientation,
            "elbow_is_policy_action": contract.elbow_is_policy_action,
            "maximum_xyz_norm_m": contract.maximum_final_xyz_norm_m,
            "curobo_handoff_band_m": list(contract.curobo_handoff_band_m),
            "curobo_default_handoff_m": contract.curobo_default_handoff_m,
            "local_grasp_band_m": list(contract.local_grasp_band_m),
            "local_grasp_distance_reference": (
                contract.local_grasp_distance_reference
            ),
            "local_grasp_distance_is_actor_input": False,
            "local_policy_role": "VISUAL_MICRO_APPROACH_AND_CLOSE_TIMING",
            "offline_num_envs": self.num_envs,
            "optimizer_batch_size": self.batch_size,
            "loss": (
                "lambda_xyz*MSE(xyz/0.0045m) + "
                "lambda_close*train_split_weighted_BCE(human_close) + "
                "lambda_privileged_feasibility*BCE(geometry_feasible)"
            ),
            "xyz_raw_m2_logged": True,
            "xyz_output_bound_parameterization": (
                "tanh_plus_radial_projection_inside_model; "
                "not downstream execution clipping"
            ),
            "silent_execution_clipping": False,
            "lambda_xyz": self.lambda_xyz,
            "lambda_close": self.lambda_close,
            "lambda_privileged_feasibility": self.lambda_privileged_feasibility,
            "privileged_teacher_quantities_are_actor_inputs": False,
            "student_privileged_input_count": 0,
            "privileged_loss_requires_verified_geometry_labels": True,
            "human_close_and_privileged_feasible_labels_are_distinct": True,
            "boolean_contact_is_actor_input": False,
            "raw_force_is_actor_input": False,
            "xyz_mask_authority": VALID_DIRECT_XYZ_MASK_AUTHORITY,
        }


@dataclass(frozen=True)
class HumanGraspSequenceInputs:
    """Deployable actor inputs; every tensor is ``[B,T,...]``."""

    right_wrist_rgb: Tensor
    right_wrist_depth_m: Tensor
    right_wrist_depth_valid: Tensor
    ee_pose_robot_root_m_xyzw: Tensor
    right_arm_joint_position_rad: Tensor
    right_arm_joint_velocity_rad_s: Tensor
    current_gripper_state: Tensor
    previous_policy_action_4d_metric_root_m: Tensor
    hidden_reset_mask: Tensor


@dataclass(frozen=True)
class HumanGraspSequenceTargets:
    """Migration-owned labels and masks, never actor inputs."""

    xyz_action_robot_root_m: Tensor
    close_target: Tensor
    padding_valid: Tensor
    row_valid: Tensor
    xyz_bc_eligible_mask: Tensor
    close_temporal_valid_mask: Tensor
    early_region_mask: Tensor
    late_region_mask: Tensor
    far_region_mask: Tensor
    xyz_mask_authority: str = VALID_DIRECT_XYZ_MASK_AUTHORITY
    close_target_semantics: str = CLOSE_STATE_TARGET


@dataclass(frozen=True)
class PrivilegedFeasibilityTargets:
    """Verified geometry labels; never part of ``HumanGraspSequenceInputs``."""

    feasible_target: Tensor
    valid_mask: Tensor
    geometry_receipt_sha256: str
    threshold_receipt_sha256: str
    source_schema: str = "g2_keyboard_v2_grasp_geometry_rows_v1"
    calibration_schema: str = "g2_pad_surface_calibration_v1"
    candidate_a_transitive_binding: str = "VERIFIED"
    student_privileged_input_count: int = 0

    def __post_init__(self) -> None:
        if self.feasible_target.ndim != 3 or self.feasible_target.shape[-1] != 1:
            raise HumanGraspBCContractError(
                "privileged feasibility target must be [B,T,1]"
            )
        if self.valid_mask.shape != self.feasible_target.shape[:2] or self.valid_mask.dtype is not torch.bool:
            raise HumanGraspBCContractError(
                "privileged feasibility valid_mask must be bool [B,T]"
            )
        _finite("privileged feasibility target", self.feasible_target)
        if not bool(
            ((self.feasible_target == 0.0) | (self.feasible_target == 1.0)).all()
        ):
            raise HumanGraspBCContractError(
                "privileged feasibility target must be exact binary 0/1"
            )
        for name, value in (
            ("geometry_receipt_sha256", self.geometry_receipt_sha256),
            ("threshold_receipt_sha256", self.threshold_receipt_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise HumanGraspBCContractError(f"{name} is not a lowercase SHA-256")
        if (
            self.source_schema
            not in {
                "g2_keyboard_v2_grasp_geometry_rows_v1",
                "g2_keyboard_v3_hdf5_v6_wrist_actor_boolean_contact_privileged_left_arm_down_v2",
            }
            or self.calibration_schema != "g2_pad_surface_calibration_v1"
            or self.candidate_a_transitive_binding != "VERIFIED"
            or self.student_privileged_input_count != 0
        ):
            raise HumanGraspBCContractError(
                "privileged feasibility authority is not verified/frozen"
            )


def targets_from_migrated_legacy_window(
    window: Mapping[str, Any],
    *,
    padding_valid: Any | None = None,
    far_region_mask: Any | None = None,
) -> HumanGraspSequenceTargets:
    """Adapt one strict ``LegacyKeyboardDataset.load_window`` result.

    XYZ is reconstructed exclusively from the migration loader's compact
    ``xyz_bc_eligible_*`` pair.  No physical action from a
    ``REQUIRES_MIGRATION`` row can enter XYZ BC.  Close labels use their
    separate temporal mask and the frozen legacy sign (+1 OPEN, -1 CLOSE).

    Legacy v26 has no authoritative far-distance label.  A validation-only
    ``far_region_mask`` may be supplied by a separately audited metric layer;
    otherwise the mask is empty and the reported far-region row count is zero.
    """

    required = {
        "row_valid_mask",
        "xyz_bc_eligible_mask",
        "close_timing_eligible_mask",
        "persistent_gripper_label",
        "context_class",
        "xyz_bc_eligible_local_indices",
        "xyz_bc_target_action_4d_metric_root_m",
    }
    missing = sorted(required - set(window))
    if missing:
        raise HumanGraspBCContractError(
            f"migrated legacy window missing fields: {missing}"
        )
    row_valid = torch.as_tensor(window["row_valid_mask"])
    xyz_mask = torch.as_tensor(window["xyz_bc_eligible_mask"])
    close_mask = torch.as_tensor(window["close_timing_eligible_mask"])
    if row_valid.ndim != 1 or row_valid.dtype is not torch.bool:
        raise HumanGraspBCContractError("migration row_valid_mask must be bool [T]")
    rows = int(row_valid.shape[0])
    for name, value in (
        ("xyz_bc_eligible_mask", xyz_mask),
        ("close_timing_eligible_mask", close_mask),
    ):
        if value.shape != (rows,) or value.dtype is not torch.bool:
            raise HumanGraspBCContractError(f"migration {name} must be bool [T]")
    local = torch.as_tensor(
        window["xyz_bc_eligible_local_indices"], dtype=torch.int64
    )
    compact = torch.as_tensor(
        window["xyz_bc_target_action_4d_metric_root_m"], dtype=torch.float32
    )
    expected_local = torch.nonzero(xyz_mask, as_tuple=False).squeeze(1)
    if not torch.equal(local, expected_local):
        raise HumanGraspBCContractError(
            "migration compact XYZ indices do not match xyz_bc_eligible_mask"
        )
    if compact.shape != (local.numel(), 4):
        raise HumanGraspBCContractError("migration compact 4-D target shape mismatch")
    _finite("migration compact target", compact)
    gripper = torch.as_tensor(
        window["persistent_gripper_label"], dtype=torch.float32
    )
    if gripper.shape != (rows,) or not bool(
        ((gripper == 1.0) | (gripper == -1.0)).all()
    ):
        raise HumanGraspBCContractError(
            "persistent_gripper_label must be exact +1 OPEN/-1 CLOSE"
        )
    xyz = torch.zeros((rows, 3), dtype=torch.float32)
    if local.numel():
        xyz[local] = compact[:, :3]
        compact_close = compact[:, 3]
        if not bool(((compact_close == 1.0) | (compact_close == -1.0)).all()):
            raise HumanGraspBCContractError("compact gripper target sign is invalid")
        if not torch.equal(compact_close, gripper[local]):
            raise HumanGraspBCContractError(
                "compact target gripper sign disagrees with persistent label"
            )
    context = tuple(str(value) for value in window["context_class"])
    if len(context) != rows or any(
        value not in ("PRE_CLOSE", "CLOSE_TRANSITION", "POST_CLOSE_CONTEXT")
        for value in context
    ):
        raise HumanGraspBCContractError("migration context_class is invalid")
    early = torch.tensor(
        [value == "PRE_CLOSE" for value in context], dtype=torch.bool
    ) & row_valid
    late = torch.tensor(
        [value == "POST_CLOSE_CONTEXT" for value in context], dtype=torch.bool
    ) & row_valid
    if far_region_mask is None:
        far = torch.zeros((rows,), dtype=torch.bool)
    else:
        far = torch.as_tensor(far_region_mask)
        if far.shape != (rows,) or far.dtype is not torch.bool:
            raise HumanGraspBCContractError("far_region_mask must be bool [T]")
        if bool((far & ~row_valid).any()):
            raise HumanGraspBCContractError("far_region_mask includes rejected rows")
    if padding_valid is None:
        padding = torch.ones((rows,), dtype=torch.bool)
    else:
        padding = torch.as_tensor(padding_valid)
        if padding.shape != (rows,) or padding.dtype is not torch.bool:
            raise HumanGraspBCContractError("padding_valid must be bool [T]")
    return HumanGraspSequenceTargets(
        xyz_action_robot_root_m=xyz.unsqueeze(0),
        close_target=(gripper < 0.0).float().view(1, rows, 1),
        padding_valid=padding.unsqueeze(0),
        row_valid=row_valid.unsqueeze(0),
        xyz_bc_eligible_mask=xyz_mask.unsqueeze(0),
        close_temporal_valid_mask=close_mask.unsqueeze(0),
        early_region_mask=early.unsqueeze(0),
        late_region_mask=late.unsqueeze(0),
        far_region_mask=far.unsqueeze(0),
        xyz_mask_authority=VALID_DIRECT_XYZ_MASK_AUTHORITY,
    )


def inputs_from_migrated_legacy_window(
    window: Mapping[str, Any],
    *,
    config: HumanGraspGRUConfig = HumanGraspGRUConfig(),
) -> tuple[HumanGraspSequenceInputs, Tensor]:
    """Build deployable inputs from one authoritative migrated window.

    The loader, not this adapter, resolves relative stored q into absolute
    right-arm q using the validated reset pose.  This function deliberately
    refuses ``controlled_joint_position_relative_rad`` as a substitute.
    Causal previous 4-D action is also loader-owned: no legacy 8-D action is
    sliced here.

    Returns ``(inputs, input_valid_mask[B,T])``.  A predecessor classified
    ``REQUIRES_MIGRATION`` or ``REJECT`` makes the current row ineligible and
    resets recurrent history.  Call :func:`mask_targets_for_input_validity`
    before loss computation.
    """

    required = {
        "controlled_joint_order",
        "right_arm_joint_position_rad",
        "right_arm_joint_velocity_rad_s",
        "ee_pose_robot_root_m_xyzw",
        "previous_policy_action_4d_metric_root_m",
        "previous_policy_action_valid_mask",
        "previous_policy_action_hidden_reset_mask",
        "row_valid_mask",
        "depth_unit",
        "depth_to_meter_scale",
        "source_fields",
    }
    missing = sorted(required - set(window))
    if missing:
        raise HumanGraspBCContractError(
            f"migrated legacy input window missing fields: {missing}"
        )
    if tuple(window["controlled_joint_order"]) != EXPECTED_CONTROLLED_JOINT_ORDER:
        raise HumanGraspBCContractError("controlled joint order authority mismatch")
    contract = canonical_keyboard_grasp_contract()
    if (
        window["depth_unit"] != contract.depth_raw_unit
        or float(window["depth_to_meter_scale"]) != contract.depth_to_meter_scale
    ):
        raise HumanGraspBCContractError("Wrist depth metric authority mismatch")
    source_fields = window["source_fields"]
    if not isinstance(source_fields, Mapping):
        raise HumanGraspBCContractError("migration source_fields must be a mapping")
    source_required = set(MIGRATED_INPUT_SOURCE_FIELDS)
    missing_source = sorted(source_required - set(source_fields))
    if missing_source:
        raise HumanGraspBCContractError(
            f"migrated Wrist RGB-D fields missing: {missing_source}"
        )
    def _camera(name: str) -> tuple[Tensor, Tensor, Tensor]:
        rgb_value = torch.as_tensor(source_fields[f"{name}_rgb"])
        depth_value = torch.as_tensor(source_fields[f"{name}_depth"], dtype=torch.float32)
        valid_value = torch.as_tensor(source_fields[f"{name}_depth_valid"])
        if valid_value.dtype is not torch.bool:
            if not bool(((valid_value == 0) | (valid_value == 1)).all()):
                raise HumanGraspBCContractError(f"{name} depth validity is not binary")
            valid_value = valid_value.to(torch.bool)
        if rgb_value.ndim != 4 or rgb_value.shape[-1] != 3:
            raise HumanGraspBCContractError(f"migrated {name} RGB must be [T,H,W,3]")
        camera_rows, camera_height, camera_width, _ = rgb_value.shape
        if depth_value.shape == (camera_rows, camera_height, camera_width):
            depth_value = depth_value.unsqueeze(-1)
        if valid_value.shape == (camera_rows, camera_height, camera_width):
            valid_value = valid_value.unsqueeze(-1)
        if depth_value.shape != (camera_rows, camera_height, camera_width, 1) or valid_value.shape != depth_value.shape:
            raise HumanGraspBCContractError(f"migrated {name} depth shape mismatch")
        valid_value = (
            valid_value & torch.isfinite(depth_value) & (depth_value >= 0.0)
            & (depth_value <= contract.maximum_policy_depth_m)
        )
        return rgb_value, torch.where(valid_value, depth_value, torch.zeros_like(depth_value)), valid_value

    rgb, depth, depth_valid = _camera("right_wrist")
    rows, height, width, _ = rgb.shape
    # Match the existing deployable PolicyDataSemantics: the stored sensor
    # bit is necessary but not sufficient.  Non-finite, negative, and
    # beyond-2m samples become invalid before the encoder, then only invalid
    # pixels map to its finite-zero representation.  This is validity
    # filtering, not a metric-depth clip.
    q = torch.as_tensor(window["right_arm_joint_position_rad"], dtype=torch.float32)
    qd = torch.as_tensor(window["right_arm_joint_velocity_rad_s"], dtype=torch.float32)
    ee = torch.as_tensor(window["ee_pose_robot_root_m_xyzw"], dtype=torch.float32)
    previous = torch.as_tensor(
        window["previous_policy_action_4d_metric_root_m"], dtype=torch.float32
    )
    previous_valid = torch.as_tensor(window["previous_policy_action_valid_mask"])
    previous_reset = torch.as_tensor(
        window["previous_policy_action_hidden_reset_mask"]
    )
    row_valid = torch.as_tensor(window["row_valid_mask"])
    if q.shape != (rows, 7) or qd.shape != (rows, 7) or ee.shape != (rows, 7):
        raise HumanGraspBCContractError("absolute q/qd/EE migration shape mismatch")
    if previous.shape != (rows, 4):
        raise HumanGraspBCContractError("causal previous 4-D action shape mismatch")
    for name, mask in (
        ("previous_policy_action_valid_mask", previous_valid),
        ("previous_policy_action_hidden_reset_mask", previous_reset),
        ("row_valid_mask", row_valid),
    ):
        if mask.shape != (rows,) or mask.dtype is not torch.bool:
            raise HumanGraspBCContractError(f"{name} must be bool [T]")
    expected_previous_reset = ~previous_valid
    expected_previous_reset = expected_previous_reset.clone()
    expected_previous_reset[0] = True
    if not torch.equal(previous_reset, expected_previous_reset):
        raise HumanGraspBCContractError("previous-action reset mask authority mismatch")
    input_valid = row_valid & previous_valid
    hidden_reset = previous_reset.clone()
    hidden_reset[0] = True
    if rows > 1:
        hidden_reset[1:] |= ~input_valid[:-1]
    inputs = HumanGraspSequenceInputs(
        right_wrist_rgb=rgb.unsqueeze(0),
        right_wrist_depth_m=depth.unsqueeze(0),
        right_wrist_depth_valid=depth_valid.unsqueeze(0),
        ee_pose_robot_root_m_xyzw=ee.unsqueeze(0),
        right_arm_joint_position_rad=q.unsqueeze(0),
        right_arm_joint_velocity_rad_s=qd.unsqueeze(0),
        current_gripper_state=previous[:, 3:4].unsqueeze(0),
        previous_policy_action_4d_metric_root_m=previous.unsqueeze(0),
        hidden_reset_mask=hidden_reset.unsqueeze(0),
    )
    # Run the complete input contract now, rather than deferring errors until
    # a training step.
    _validate_inputs(inputs, config)
    return inputs, input_valid.unsqueeze(0)


def mask_targets_for_input_validity(
    targets: HumanGraspSequenceTargets, input_valid_mask: Tensor
) -> HumanGraspSequenceTargets:
    """Exclude rows whose causal deployable input could not be reconstructed."""

    mask = torch.as_tensor(input_valid_mask)
    if mask.shape != targets.padding_valid.shape or mask.dtype is not torch.bool:
        raise HumanGraspBCContractError("input_valid_mask must be bool [B,T]")
    padding = targets.padding_valid & mask
    return replace(
        targets,
        padding_valid=padding,
        row_valid=targets.row_valid & mask,
        xyz_bc_eligible_mask=targets.xyz_bc_eligible_mask & mask,
        close_temporal_valid_mask=targets.close_temporal_valid_mask & mask,
        early_region_mask=targets.early_region_mask & mask,
        late_region_mask=targets.late_region_mask & mask,
        far_region_mask=targets.far_region_mask & mask,
    )


@dataclass(frozen=True)
class HumanGraspGRUOutput:
    xyz_action_robot_root_m: Tensor
    close_logit: Tensor
    close_probability: Tensor
    feasibility_logit: Tensor
    feasibility_probability: Tensor
    action_4d: Tensor
    xyz_projection_applied: Tensor
    recurrent_features: Tensor
    final_hidden: Tensor


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        groups = tuple(set(value) for value in (self.train, self.validation, self.test))
        if any(not group for group in groups):
            raise HumanGraspBCContractError("train/validation/test episode splits must be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise HumanGraspBCContractError("episode leakage across train/validation/test")
        if any(not isinstance(item, str) or not item for group in groups for item in group):
            raise HumanGraspBCContractError("episode IDs must be non-empty strings")

    def as_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed": self.seed,
            "split_unit": "whole_episode",
            "adjacent_row_leakage": False,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.as_dict())


def split_episode_ids(
    episode_ids: Sequence[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> EpisodeSplit:
    """Split whole episodes only; rows are never independently shuffled."""

    ids = tuple(episode_ids)
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise HumanGraspBCContractError("at least three unique episode IDs are required")
    if not all(isinstance(value, str) and value for value in ids):
        raise HumanGraspBCContractError("episode IDs must be non-empty strings")
    if not (
        math.isfinite(train_fraction)
        and math.isfinite(validation_fraction)
        and 0.0 < train_fraction < 1.0
        and 0.0 < validation_fraction < 1.0
        and train_fraction + validation_fraction < 1.0
    ):
        raise HumanGraspBCContractError("invalid episode split fractions")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = max(1, min(total - 2, int(math.floor(total * train_fraction))))
    validation_count = max(
        1,
        min(total - train_count - 1, int(math.floor(total * validation_fraction))),
    )
    return EpisodeSplit(
        train=tuple(shuffled[:train_count]),
        validation=tuple(shuffled[train_count : train_count + validation_count]),
        test=tuple(shuffled[train_count + validation_count :]),
        seed=seed,
    )


def assert_batch_partition(
    episode_ids: Sequence[str], split: EpisodeSplit, partition: str
) -> None:
    allowed = {
        "train": set(split.train),
        "validation": set(split.validation),
        "test": set(split.test),
    }.get(partition)
    if allowed is None:
        raise HumanGraspBCContractError(f"unknown episode partition: {partition}")
    observed = tuple(episode_ids)
    if not observed or any(value not in allowed for value in observed):
        raise HumanGraspBCContractError(
            f"batch crosses the whole-episode {partition} partition"
        )


def _validate_inputs(
    inputs: HumanGraspSequenceInputs, config: HumanGraspGRUConfig
) -> tuple[int, int]:
    rgb = inputs.right_wrist_rgb
    depth = inputs.right_wrist_depth_m
    valid = inputs.right_wrist_depth_valid
    if rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise HumanGraspBCContractError("right_wrist_rgb must be [B,T,H,W,3]")
    batch, time, height, width, _ = rgb.shape
    if depth.shape != (batch, time, height, width, 1):
        raise HumanGraspBCContractError("right_wrist_depth_m shape mismatch")
    if valid.shape != depth.shape or valid.dtype is not torch.bool:
        raise HumanGraspBCContractError("right_wrist_depth_valid must be bool depth-shaped")
    if rgb.dtype != torch.uint8 and not torch.is_floating_point(rgb):
        raise HumanGraspBCContractError("right_wrist_rgb must be uint8 or floating point")
    if torch.is_floating_point(rgb):
        _finite("right_wrist_rgb", rgb)
        if bool(((rgb < 0.0) | (rgb > 1.0)).any()):
            raise HumanGraspBCContractError("floating RGB must be normalized to [0,1]")
    _finite("right_wrist_depth_m", depth)
    if bool((depth < 0.0).any()) or bool((depth[valid] > config.maximum_depth_m).any()):
        raise HumanGraspBCContractError("valid Wrist depth is outside the metric range")
    if bool((depth[~valid] != 0.0).any()):
        raise HumanGraspBCContractError("invalid Wrist depth pixels must be finite zero")
    shapes = {
        "ee_pose_robot_root_m_xyzw": (batch, time, 7),
        "right_arm_joint_position_rad": (batch, time, 7),
        "right_arm_joint_velocity_rad_s": (batch, time, 7),
        "current_gripper_state": (batch, time, 1),
        "previous_policy_action_4d_metric_root_m": (batch, time, 4),
    }
    for name, shape in shapes.items():
        value = getattr(inputs, name)
        if value.shape != shape:
            raise HumanGraspBCContractError(f"{name} must have shape {shape}")
        _finite(name, value)
    if inputs.hidden_reset_mask.shape != (batch, time) or inputs.hidden_reset_mask.dtype is not torch.bool:
        raise HumanGraspBCContractError("hidden_reset_mask must be bool [B,T]")
    if not bool(inputs.hidden_reset_mask[:, 0].all()):
        raise HumanGraspBCContractError("every episode sequence must reset GRU state at t=0")
    quaternion = inputs.ee_pose_robot_root_m_xyzw[..., 3:7]
    if bool((torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1.0) > 1.0e-3).any()):
        raise HumanGraspBCContractError("EE quaternion must be unit XYZW")
    if not bool(
        ((inputs.current_gripper_state == 0.0) | (inputs.current_gripper_state == 1.0)).all()
    ):
        raise HumanGraspBCContractError(
            "current_gripper_state must use exact OPEN=0/CLOSED=1 encoding"
        )
    previous = inputs.previous_policy_action_4d_metric_root_m
    bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
    if bool((torch.linalg.vector_norm(previous[..., :3], dim=-1) > bound + 1.0e-8).any()):
        raise HumanGraspBCContractError("previous XYZ action exceeds the 4.5-mm bound")
    if bool(((previous[..., 3] < 0.0) | (previous[..., 3] > 1.0)).any()):
        raise HumanGraspBCContractError("previous gripper action must be in [0,1]")
    return batch, time


def _validate_targets(
    targets: HumanGraspSequenceTargets, *, batch: int, time: int
) -> tuple[Tensor, Tensor]:
    if targets.xyz_mask_authority != VALID_DIRECT_XYZ_MASK_AUTHORITY:
        raise HumanGraspBCContractError("XYZ supervision authority must be VALID_DIRECT")
    if targets.close_target_semantics not in (CLOSE_STATE_TARGET, CLOSE_EDGE_TARGET):
        raise HumanGraspBCContractError("unknown CLOSE target semantics")
    if targets.xyz_action_robot_root_m.shape != (batch, time, 3):
        raise HumanGraspBCContractError("XYZ target must be [B,T,3] robot-root metres")
    if targets.close_target.shape != (batch, time, 1):
        raise HumanGraspBCContractError("close_target must be [B,T,1]")
    _finite("xyz_action_robot_root_m", targets.xyz_action_robot_root_m)
    _finite("close_target", targets.close_target)
    if not bool(((targets.close_target == 0.0) | (targets.close_target == 1.0)).all()):
        raise HumanGraspBCContractError("close_target must be exact binary 0/1")
    masks = (
        "padding_valid",
        "row_valid",
        "xyz_bc_eligible_mask",
        "close_temporal_valid_mask",
        "early_region_mask",
        "late_region_mask",
        "far_region_mask",
    )
    for name in masks:
        mask = getattr(targets, name)
        if mask.shape != (batch, time) or mask.dtype is not torch.bool:
            raise HumanGraspBCContractError(f"{name} must be bool [B,T]")
    base = targets.padding_valid & targets.row_valid
    if bool((targets.xyz_bc_eligible_mask & ~base).any()):
        raise HumanGraspBCContractError("VALID_DIRECT XYZ mask includes invalid/corrupt rows")
    if bool((targets.close_temporal_valid_mask & ~base).any()):
        raise HumanGraspBCContractError("close temporal mask includes invalid/corrupt rows")
    for name in ("early_region_mask", "late_region_mask", "far_region_mask"):
        if bool((getattr(targets, name) & ~base).any()):
            raise HumanGraspBCContractError(f"{name} includes invalid/corrupt rows")
    xyz_mask = base & targets.xyz_bc_eligible_mask
    close_mask = base & targets.close_temporal_valid_mask
    if targets.close_target_semantics == CLOSE_EDGE_TARGET:
        edge = targets.close_target[..., 0].to(torch.bool)
        if bool((edge.sum(dim=1) > 1).any()):
            raise HumanGraspBCContractError("CLOSE_EDGE target permits at most one event per episode")
        if bool((edge & ~close_mask).any()):
            raise HumanGraspBCContractError("CLOSE_EDGE onset must be included in close loss")
        for batch_index in range(batch):
            edge_index = torch.nonzero(edge[batch_index], as_tuple=False).reshape(-1)
            if edge_index.numel():
                onset = int(edge_index[0].item())
                if bool(close_mask[batch_index, onset + 1 :].any()):
                    raise HumanGraspBCContractError(
                        "post-CLOSE persistence rows must be excluded from close loss"
                    )
    bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
    if bool((torch.linalg.vector_norm(targets.xyz_action_robot_root_m[xyz_mask], dim=-1) > bound + 1.0e-8).any()):
        raise HumanGraspBCContractError("VALID_DIRECT XYZ target exceeds the 4.5-mm bound")
    return xyz_mask, close_mask


class WristRGBDEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        maximum_depth_m: float,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.maximum_depth_m = float(maximum_depth_m)
        self.image_hw = (int(image_height), int(image_width))
        self.network = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, output_dim)

    def forward(self, rgb: Tensor, depth_m: Tensor, depth_valid: Tensor) -> Tensor:
        rgb_float = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
        depth = torch.where(
            depth_valid,
            depth_m.float() / self.maximum_depth_m,
            torch.zeros_like(depth_m, dtype=torch.float32),
        )
        batch, time, height, width = rgb_float.shape[:4]
        rgb_nchw = rgb_float.permute(0, 1, 4, 2, 3).reshape(
            batch * time, 3, height, width
        )
        depth_nchw = depth.permute(0, 1, 4, 2, 3).reshape(
            batch * time, 1, height, width
        )
        valid_nchw = depth_valid.float().permute(0, 1, 4, 2, 3).reshape(
            batch * time, 1, height, width
        )
        if (height, width) != self.image_hw:
            rgb_nchw = F.interpolate(
                rgb_nchw,
                size=self.image_hw,
                mode="bilinear",
                align_corners=False,
            )
            depth_nchw = F.interpolate(
                depth_nchw, size=self.image_hw, mode="nearest"
            )
            valid_nchw = F.interpolate(
                valid_nchw, size=self.image_hw, mode="nearest"
            )
        nchw = torch.cat((rgb_nchw, depth_nchw, valid_nchw), dim=1)
        encoded = self.network(nchw).flatten(1)
        return self.projection(encoded).reshape(batch, time, -1)


class HumanGraspGRUBC(nn.Module):
    """Wrist-centered recurrent actor with metric XYZ and close heads."""

    def __init__(self, config: HumanGraspGRUConfig = HumanGraspGRUConfig()) -> None:
        super().__init__()
        self.config = config
        self.vision = WristRGBDEncoder(
            config.vision_feature_dim,
            maximum_depth_m=config.maximum_depth_m,
            image_height=config.encoder_image_height,
            image_width=config.encoder_image_width,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(ROBOT_STATE_DIM, config.state_feature_dim),
            nn.LayerNorm(config.state_feature_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.vision_feature_dim + config.state_feature_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            config.fusion_dim,
            config.gru_hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
        )
        self.xyz_head = nn.Linear(config.gru_hidden_dim, 3)
        self.close_head = nn.Linear(config.gru_hidden_dim, 1)
        self.feasibility_head = nn.Linear(config.gru_hidden_dim, 1)
        # Optional and absent in historical checkpoints.  It is attached only
        # by an explicit CLOSE-edge experiment and consumes detached shared
        # fusion features, so it cannot send gradients into the XYZ path.
        self.close_temporal: nn.GRU | None = None
        self.close_temporal_head: nn.Linear | None = None
        self.close_temporal_branch_config: dict[str, int] | None = None

    def enable_close_temporal_branch(
        self,
        *,
        hidden_dim: int = 32,
        layers: int = 1,
    ) -> None:
        if self.close_temporal is not None or self.close_temporal_head is not None:
            raise HumanGraspBCContractError("CLOSE temporal branch already enabled")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise HumanGraspBCContractError("CLOSE temporal hidden_dim must be positive")
        if type(layers) is not int or layers <= 0:
            raise HumanGraspBCContractError("CLOSE temporal layers must be positive")
        reference = next(self.parameters())
        self.close_temporal = nn.GRU(
            self.config.fusion_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
        ).to(device=reference.device, dtype=reference.dtype)
        self.close_temporal_head = nn.Linear(hidden_dim, 1).to(
            device=reference.device, dtype=reference.dtype
        )
        self.close_temporal_branch_config = {
            "hidden_dim": hidden_dim,
            "layers": layers,
        }

    def forward(
        self, inputs: HumanGraspSequenceInputs, hidden: Tensor | None = None
    ) -> HumanGraspGRUOutput:
        batch, time = _validate_inputs(inputs, self.config)
        wrist_visual = self.vision(
            inputs.right_wrist_rgb,
            inputs.right_wrist_depth_m,
            inputs.right_wrist_depth_valid,
        )
        state = torch.cat(
            (
                inputs.ee_pose_robot_root_m_xyzw,
                inputs.right_arm_joint_position_rad,
                inputs.right_arm_joint_velocity_rad_s,
                inputs.current_gripper_state,
                inputs.previous_policy_action_4d_metric_root_m,
            ),
            dim=-1,
        )
        if state.shape[-1] != ROBOT_STATE_DIM:
            raise RuntimeError("robot state width drifted from the 26-D contract")
        fused = self.fusion(
            torch.cat((wrist_visual, self.state_encoder(state)), dim=-1)
        )
        if hidden is None:
            hidden = torch.zeros(
                self.config.gru_layers,
                batch,
                self.config.gru_hidden_dim,
                device=fused.device,
                dtype=fused.dtype,
            )
        expected_hidden = (
            self.config.gru_layers,
            batch,
            self.config.gru_hidden_dim,
        )
        if hidden.shape != expected_hidden:
            raise HumanGraspBCContractError(f"hidden must have shape {expected_hidden}")
        recurrent: list[Tensor] = []
        for index in range(time):
            reset = inputs.hidden_reset_mask[:, index]
            if bool(reset.any()):
                hidden = hidden * (~reset).to(hidden.dtype).view(1, batch, 1)
            step, hidden = self.gru(fused[:, index : index + 1], hidden)
            recurrent.append(step)
        features = torch.cat(recurrent, dim=1)
        bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
        xyz_normalized = torch.tanh(self.xyz_head(features))
        norm = torch.linalg.vector_norm(xyz_normalized, dim=-1, keepdim=True)
        projection_applied = norm[..., 0] > 1.0
        xyz = (
            xyz_normalized
            * torch.clamp(1.0 / norm.clamp_min(1.0e-12), max=1.0)
            * bound
        )
        if self.close_temporal is None:
            logit = self.close_head(features)
        else:
            if self.close_temporal_head is None or self.close_temporal_branch_config is None:
                raise RuntimeError("incomplete CLOSE temporal branch")
            close_hidden = torch.zeros(
                self.close_temporal_branch_config["layers"],
                batch,
                self.close_temporal_branch_config["hidden_dim"],
                device=fused.device,
                dtype=fused.dtype,
            )
            close_recurrent: list[Tensor] = []
            # The explicit detach is the gradient firewall protecting vision,
            # fusion, the existing XYZ GRU, and XYZ head.
            close_input = fused.detach()
            for index in range(time):
                reset = inputs.hidden_reset_mask[:, index]
                if bool(reset.any()):
                    close_hidden = close_hidden * (~reset).to(
                        close_hidden.dtype
                    ).view(1, batch, 1)
                close_step, close_hidden = self.close_temporal(
                    close_input[:, index : index + 1], close_hidden
                )
                close_recurrent.append(close_step)
            logit = self.close_temporal_head(torch.cat(close_recurrent, dim=1))
        probability = torch.sigmoid(logit)
        feasibility_logit = self.feasibility_head(features)
        feasibility_probability = torch.sigmoid(feasibility_logit)
        action = torch.cat((xyz, probability), dim=-1)
        return HumanGraspGRUOutput(
            xyz_action_robot_root_m=xyz,
            close_logit=logit,
            close_probability=probability,
            feasibility_logit=feasibility_logit,
            feasibility_probability=feasibility_probability,
            action_4d=action,
            xyz_projection_applied=projection_applied,
            recurrent_features=features,
            final_hidden=hidden,
        )


@dataclass(frozen=True)
class ClosePositiveWeightReceipt:
    split_fingerprint: str
    positive_rows: int
    negative_rows: int
    positive_weight: float
    source_partition: str = "train"

    def __post_init__(self) -> None:
        if self.source_partition != "train":
            raise HumanGraspBCContractError("close positive weight must be train-derived")
        if self.positive_rows <= 0 or self.negative_rows <= 0:
            raise HumanGraspBCContractError("train close labels need positive and negative rows")
        expected = self.negative_rows / self.positive_rows
        if not math.isfinite(self.positive_weight) or abs(self.positive_weight - expected) > 1.0e-12:
            raise HumanGraspBCContractError("close positive weight is not train-derived N/P")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_train_close_positive_weight(
    targets: HumanGraspSequenceTargets,
    *,
    batch_episode_ids: Sequence[str],
    split: EpisodeSplit,
) -> ClosePositiveWeightReceipt:
    assert_batch_partition(batch_episode_ids, split, "train")
    if set(batch_episode_ids) != set(split.train):
        raise HumanGraspBCContractError(
            "close positive weight must cover every train-split episode"
        )
    batch, time = targets.close_target.shape[:2]
    if len(batch_episode_ids) != batch:
        raise HumanGraspBCContractError("one episode ID is required per sequence")
    _, close_mask = _validate_targets(targets, batch=batch, time=time)
    labels = targets.close_target[..., 0][close_mask]
    positive = int((labels == 1.0).sum().item())
    negative = int((labels == 0.0).sum().item())
    return ClosePositiveWeightReceipt(
        split_fingerprint=split.fingerprint,
        positive_rows=positive,
        negative_rows=negative,
        positive_weight=negative / positive if positive else float("inf"),
    )


@dataclass(frozen=True)
class HumanGraspBCLoss:
    total: Tensor
    xyz_raw_mse_m2: Tensor
    xyz_normalized_mse: Tensor
    close_raw_bce: Tensor
    privileged_feasibility_raw_bce: Tensor
    xyz_weighted: Tensor
    close_weighted: Tensor
    privileged_feasibility_weighted: Tensor
    xyz_valid_rows: int
    close_valid_rows: int
    privileged_feasibility_valid_rows: int
    close_positive_weight: float

    def metrics(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach()),
            "loss/xyz_raw_mse_m2": float(self.xyz_raw_mse_m2.detach()),
            "loss/xyz_normalized_mse": float(self.xyz_normalized_mse.detach()),
            "loss/close_raw_bce": float(self.close_raw_bce.detach()),
            "loss/privileged_feasibility_raw_bce": float(
                self.privileged_feasibility_raw_bce.detach()
            ),
            "loss/xyz_weighted_contribution": float(self.xyz_weighted.detach()),
            "loss/close_weighted_contribution": float(self.close_weighted.detach()),
            "loss/privileged_feasibility_weighted_contribution": float(
                self.privileged_feasibility_weighted.detach()
            ),
            "mask/xyz_valid_direct_rows": float(self.xyz_valid_rows),
            "mask/close_temporal_valid_rows": float(self.close_valid_rows),
            "mask/privileged_feasibility_valid_rows": float(
                self.privileged_feasibility_valid_rows
            ),
            "scale/close_positive_weight_train_only": self.close_positive_weight,
        }


def compute_human_grasp_bc_loss(
    output: HumanGraspGRUOutput,
    targets: HumanGraspSequenceTargets,
    *,
    config: HumanGraspGRUConfig,
    positive_weight: ClosePositiveWeightReceipt,
    split: EpisodeSplit,
    privileged_targets: PrivilegedFeasibilityTargets | None = None,
) -> HumanGraspBCLoss:
    if positive_weight.split_fingerprint != split.fingerprint:
        raise HumanGraspBCContractError("close weight/split receipt mismatch")
    batch, time = output.xyz_action_robot_root_m.shape[:2]
    xyz_mask, close_mask = _validate_targets(targets, batch=batch, time=time)
    xyz_count = int(xyz_mask.sum().item())
    close_count = int(close_mask.sum().item())
    if xyz_count <= 0 and close_count <= 0:
        raise HumanGraspBCContractError("batch has no XYZ or close supervision")
    if xyz_count:
        xyz_raw_loss = F.mse_loss(
            output.xyz_action_robot_root_m[xyz_mask],
            targets.xyz_action_robot_root_m[xyz_mask],
        )
        xyz_loss = F.mse_loss(
            output.xyz_action_robot_root_m[xyz_mask]
            / canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m,
            targets.xyz_action_robot_root_m[xyz_mask]
            / canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m,
        )
    else:
        xyz_raw_loss = output.xyz_action_robot_root_m.sum() * 0.0
        xyz_loss = xyz_raw_loss
    pos_weight = torch.tensor(
        positive_weight.positive_weight,
        device=output.close_logit.device,
        dtype=output.close_logit.dtype,
    )
    if close_count:
        close_loss = F.binary_cross_entropy_with_logits(
            output.close_logit[..., 0][close_mask],
            targets.close_target[..., 0][close_mask],
            pos_weight=pos_weight,
        )
    else:
        close_loss = output.close_logit.sum() * 0.0
    xyz_weighted = config.lambda_xyz * xyz_loss
    close_weighted = config.lambda_close * close_loss
    if privileged_targets is None:
        feasibility_loss = output.feasibility_logit.sum() * 0.0
        feasibility_count = 0
    else:
        if privileged_targets.feasible_target.shape[:2] != (batch, time):
            raise HumanGraspBCContractError(
                "privileged feasibility target/output sequence mismatch"
            )
        feasibility_mask = (
            privileged_targets.valid_mask
            & targets.padding_valid
            & targets.row_valid
        )
        feasibility_count = int(feasibility_mask.sum().item())
        if feasibility_count <= 0:
            raise HumanGraspBCContractError(
                "verified privileged supervision has no valid rows"
            )
        feasibility_loss = F.binary_cross_entropy_with_logits(
            output.feasibility_logit[..., 0][feasibility_mask],
            privileged_targets.feasible_target[..., 0][feasibility_mask],
        )
    feasibility_weighted = (
        config.lambda_privileged_feasibility * feasibility_loss
    )
    total = xyz_weighted + close_weighted + feasibility_weighted
    for name, value in (
        ("raw XYZ loss", xyz_raw_loss),
        ("normalized XYZ loss", xyz_loss),
        ("close loss", close_loss),
        ("privileged feasibility loss", feasibility_loss),
        ("total loss", total),
    ):
        _finite(name, value)
    return HumanGraspBCLoss(
        total=total,
        xyz_raw_mse_m2=xyz_raw_loss,
        xyz_normalized_mse=xyz_loss,
        close_raw_bce=close_loss,
        privileged_feasibility_raw_bce=feasibility_loss,
        xyz_weighted=xyz_weighted,
        close_weighted=close_weighted,
        privileged_feasibility_weighted=feasibility_weighted,
        xyz_valid_rows=xyz_count,
        close_valid_rows=close_count,
        privileged_feasibility_valid_rows=feasibility_count,
        close_positive_weight=positive_weight.positive_weight,
    )


@dataclass(frozen=True)
class HysteresisCalibration:
    open_threshold: float
    close_threshold: float
    validation_split_fingerprint: str
    close_precision: float
    close_recall: float
    close_f1: float
    mode: str = "STATE_HYSTERESIS"

    def __post_init__(self) -> None:
        if self.mode == "STATE_HYSTERESIS":
            valid_thresholds = 0.0 <= self.open_threshold < self.close_threshold <= 1.0
        elif self.mode == "EDGE_SINGLE_EVENT":
            valid_thresholds = (
                0.0 <= self.close_threshold <= 1.0
                and self.open_threshold == self.close_threshold
            )
        else:
            raise HumanGraspBCContractError("unknown CLOSE calibration mode")
        if not valid_thresholds:
            raise HumanGraspBCContractError("invalid OPEN/CLOSE thresholds")
        for name in ("close_precision", "close_recall", "close_f1"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise HumanGraspBCContractError(
                    f"hysteresis calibration {name} must be finite in [0,1]"
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeasibilityCalibration:
    threshold: float
    validation_split_fingerprint: str
    precision: float
    recall: float
    f1: float
    geometry_receipt_sha256: str
    threshold_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("threshold", "precision", "recall", "f1"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise HumanGraspBCContractError(
                    f"feasibility calibration {name} must be in [0,1]"
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classification_counts(predicted: Tensor, target: Tensor) -> tuple[int, int, int]:
    truth = target == 1.0
    return (
        int((predicted & truth).sum().item()),
        int((predicted & ~truth).sum().item()),
        int((~predicted & truth).sum().item()),
    )


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def calibrate_feasibility_on_validation(
    feasibility_probability: Tensor,
    targets: PrivilegedFeasibilityTargets,
    *,
    batch_episode_ids: Sequence[str],
    split: EpisodeSplit,
) -> FeasibilityCalibration:
    """Select the feasibility threshold from verified validation labels only."""

    assert_batch_partition(batch_episode_ids, split, "validation")
    if set(batch_episode_ids) != set(split.validation):
        raise HumanGraspBCContractError(
            "feasibility threshold calibration must cover every validation episode"
        )
    if feasibility_probability.shape != targets.feasible_target.shape:
        raise HumanGraspBCContractError(
            "feasibility probability/target shape mismatch"
        )
    _finite("feasibility_probability", feasibility_probability)
    if bool(
        ((feasibility_probability < 0.0) | (feasibility_probability > 1.0)).any()
    ):
        raise HumanGraspBCContractError("feasibility probability must be in [0,1]")
    valid = targets.valid_mask
    values = sorted(
        float(value)
        for value in torch.unique(
            feasibility_probability[..., 0][valid].detach().cpu()
        ).tolist()
    )
    if not values:
        raise HumanGraspBCContractError("no verified feasibility validation rows")
    label = targets.feasible_target[..., 0]
    best: tuple[tuple[float, float, float, float], float, tuple[float, float, float]] | None = None
    for threshold in values:
        predicted = feasibility_probability[..., 0] >= threshold
        tp, fp, fn = _classification_counts(predicted[valid], label[valid])
        precision, recall, f1 = _prf(tp, fp, fn)
        score = (f1, precision, recall, threshold)
        if best is None or score > best[0]:
            best = (score, threshold, (precision, recall, f1))
    assert best is not None
    return FeasibilityCalibration(
        threshold=best[1],
        validation_split_fingerprint=split.fingerprint,
        precision=best[2][0],
        recall=best[2][1],
        f1=best[2][2],
        geometry_receipt_sha256=targets.geometry_receipt_sha256,
        threshold_receipt_sha256=targets.threshold_receipt_sha256,
    )


def apply_close_feasibility_gate(
    close_probability: Tensor,
    feasibility_probability: Tensor,
    previous_closed: Tensor,
    *,
    close_calibration: HysteresisCalibration,
    feasibility_calibration: FeasibilityCalibration,
) -> Tensor:
    """Combine validation-calibrated heads without granting SAC gripper authority."""

    if close_calibration.validation_split_fingerprint != feasibility_calibration.validation_split_fingerprint:
        raise HumanGraspBCContractError("close/feasibility calibration split mismatch")
    if close_probability.shape != feasibility_probability.shape or close_probability.shape != previous_closed.shape:
        raise HumanGraspBCContractError("combined close gate tensor shape mismatch")
    if previous_closed.dtype is not torch.bool:
        raise HumanGraspBCContractError("previous_closed must be boolean")
    close_now = (
        (close_probability >= close_calibration.close_threshold)
        & (feasibility_probability >= feasibility_calibration.threshold)
    )
    if close_calibration.mode == "EDGE_SINGLE_EVENT":
        return previous_closed | close_now
    open_now = close_probability <= close_calibration.open_threshold
    return torch.where(close_now, True, torch.where(open_now, False, previous_closed))


def _hysteresis_predictions(probability: Tensor, valid: Tensor, opening: float, closing: float) -> Tensor:
    batch, time = probability.shape
    predicted = torch.zeros((batch, time), dtype=torch.bool, device=probability.device)
    closed = torch.zeros((batch,), dtype=torch.bool, device=probability.device)
    for index in range(time):
        row_valid = valid[:, index]
        closed = torch.where(row_valid & ~closed & (probability[:, index] >= closing), True, closed)
        closed = torch.where(row_valid & closed & (probability[:, index] <= opening), False, closed)
        predicted[:, index] = closed
    return predicted


def calibrate_hysteresis_on_validation(
    close_probability: Tensor,
    targets: HumanGraspSequenceTargets,
    *,
    batch_episode_ids: Sequence[str],
    split: EpisodeSplit,
) -> HysteresisCalibration:
    """Select both hysteresis thresholds on validation episodes only."""

    assert_batch_partition(batch_episode_ids, split, "validation")
    if set(batch_episode_ids) != set(split.validation):
        raise HumanGraspBCContractError(
            "threshold calibration must cover every validation episode"
        )
    if close_probability.ndim != 3 or close_probability.shape[-1] != 1:
        raise HumanGraspBCContractError("close_probability must be [B,T,1]")
    _finite("close_probability", close_probability)
    if bool(((close_probability < 0.0) | (close_probability > 1.0)).any()):
        raise HumanGraspBCContractError("close_probability must be in [0,1]")
    batch, time = close_probability.shape[:2]
    _, valid = _validate_targets(targets, batch=batch, time=time)
    values = torch.unique(close_probability[..., 0][valid].detach().cpu()).tolist()
    if len(values) < 2:
        raise HumanGraspBCContractError("validation probabilities cannot calibrate hysteresis")
    # Candidate thresholds are data-derived.  If there are many unique values,
    # evenly ranked values retain bounded O(K^2) work without inventing 0.5.
    values = sorted(float(value) for value in values)
    if len(values) > 64:
        indices = [round(index * (len(values) - 1) / 63) for index in range(64)]
        values = sorted({values[index] for index in indices})
    probability = close_probability[..., 0]
    label = targets.close_target[..., 0]
    if targets.close_target_semantics == CLOSE_EDGE_TARGET:
        best_edge: tuple[
            tuple[float, float, float, float], float, tuple[float, float, float]
        ] | None = None
        for threshold in values:
            predicted = probability >= threshold
            tp, fp, fn = _classification_counts(predicted[valid], label[valid])
            precision, recall, f1 = _prf(tp, fp, fn)
            # F1 is primary; precision wins the tie because a premature CLOSE
            # is more costly than rejecting one candidate onset.
            score = (f1, precision, recall, threshold)
            if best_edge is None or score > best_edge[0]:
                best_edge = (score, threshold, (precision, recall, f1))
        assert best_edge is not None
        return HysteresisCalibration(
            open_threshold=best_edge[1],
            close_threshold=best_edge[1],
            validation_split_fingerprint=split.fingerprint,
            close_precision=best_edge[2][0],
            close_recall=best_edge[2][1],
            close_f1=best_edge[2][2],
            mode="EDGE_SINGLE_EVENT",
        )
    best: tuple[tuple[float, float, float, float], float, float, tuple[float, float, float]] | None = None
    for open_threshold in values:
        for close_threshold in values:
            if open_threshold >= close_threshold:
                continue
            predicted = _hysteresis_predictions(
                probability, valid, open_threshold, close_threshold
            )
            tp, fp, fn = _classification_counts(predicted[valid], label[valid])
            precision, recall, f1 = _prf(tp, fp, fn)
            score = (f1, precision, recall, close_threshold - open_threshold)
            if best is None or score > best[0]:
                best = (score, open_threshold, close_threshold, (precision, recall, f1))
    if best is None:
        raise HumanGraspBCContractError("no ordered validation hysteresis pair exists")
    return HysteresisCalibration(
        open_threshold=best[1],
        close_threshold=best[2],
        validation_split_fingerprint=split.fingerprint,
        close_precision=best[3][0],
        close_recall=best[3][1],
        close_f1=best[3][2],
        mode="STATE_HYSTERESIS",
    )


def _binary_average_precision(probability: Tensor, label: Tensor) -> float:
    """Exact non-interpolated PR-AUC/AP for a binary row set."""

    positives = int((label == 1.0).sum().item())
    if positives <= 0:
        return 0.0
    order = torch.argsort(probability, descending=True, stable=True)
    truth = (label[order] == 1.0).to(torch.float64)
    precision = torch.cumsum(truth, dim=0) / torch.arange(
        1, truth.numel() + 1, device=truth.device, dtype=torch.float64
    )
    return float((precision * truth).sum().item() / positives)


def evaluate_human_grasp_bc(
    output: HumanGraspGRUOutput,
    targets: HumanGraspSequenceTargets,
    *,
    batch_episode_ids: Sequence[str],
    split: EpisodeSplit,
    partition: str,
    calibration: HysteresisCalibration,
) -> dict[str, float]:
    assert_batch_partition(batch_episode_ids, split, partition)
    if calibration.validation_split_fingerprint != split.fingerprint:
        raise HumanGraspBCContractError("threshold calibration/split mismatch")
    batch, time = output.xyz_action_robot_root_m.shape[:2]
    xyz_mask, close_mask = _validate_targets(targets, batch=batch, time=time)
    prediction_xyz = output.xyz_action_robot_root_m
    xyz_error = prediction_xyz[xyz_mask] - targets.xyz_action_robot_root_m[xyz_mask]
    rmse_mm = (
        float((torch.sqrt(torch.mean(xyz_error.square())) * 1000.0).detach())
        if bool(xyz_mask.any())
        else 0.0
    )
    target_xyz = targets.xyz_action_robot_root_m[xyz_mask]
    predicted_valid_xyz = prediction_xyz[xyz_mask]
    directional = torch.linalg.vector_norm(target_xyz, dim=-1) > 1.0e-8
    directional_rows = int(directional.sum().item())
    if bool(directional.any()):
        cosine = F.cosine_similarity(
            predicted_valid_xyz[directional], target_xyz[directional], dim=-1, eps=1.0e-8
        ).clamp(-1.0, 1.0)
        direction_deg = float(torch.rad2deg(torch.acos(cosine)).mean().detach())
    else:
        direction_deg = 0.0
    edge_mode = targets.close_target_semantics == CLOSE_EDGE_TARGET
    if edge_mode != (calibration.mode == "EDGE_SINGLE_EVENT"):
        raise HumanGraspBCContractError("CLOSE target/calibration mode mismatch")
    if edge_mode:
        predicted_close = (
            output.close_probability[..., 0] >= calibration.close_threshold
        ) & close_mask
    else:
        predicted_close = _hysteresis_predictions(
            output.close_probability[..., 0],
            close_mask,
            calibration.open_threshold,
            calibration.close_threshold,
        )
    label = targets.close_target[..., 0]
    tp, fp, fn = _classification_counts(predicted_close[close_mask], label[close_mask])
    precision, recall, f1 = _prf(tp, fp, fn)
    edge_pr_auc = _binary_average_precision(
        output.close_probability[..., 0][close_mask], label[close_mask]
    ) if edge_mode else 0.0

    onset_errors_steps: list[int] = []
    missing_predicted_onsets = 0
    missing_target_onsets = 0
    early_onset_count = 0
    late_onset_count = 0
    within_one_step_count = 0
    within_two_step_count = 0
    for batch_index in range(batch):
        row_mask = close_mask[batch_index]
        predicted_rows = predicted_close[batch_index][row_mask]
        target_rows = label[batch_index][row_mask].to(torch.bool)
        if edge_mode:
            predicted_edges = predicted_rows
            target_edges = target_rows
        else:
            predicted_edges = predicted_rows & ~torch.cat(
                (
                    torch.zeros(1, dtype=torch.bool, device=predicted_rows.device),
                    predicted_rows[:-1],
                )
            )
            target_edges = target_rows & ~torch.cat(
                (
                    torch.zeros(1, dtype=torch.bool, device=target_rows.device),
                    target_rows[:-1],
                )
            )
        predicted_index = torch.nonzero(predicted_edges, as_tuple=False).reshape(-1)
        target_index = torch.nonzero(target_edges, as_tuple=False).reshape(-1)
        if not bool(target_index.numel()):
            missing_target_onsets += 1
            continue
        if not bool(predicted_index.numel()):
            missing_predicted_onsets += 1
            late_onset_count += 1
            continue
        error = int(predicted_index[0].item()) - int(target_index[0].item())
        onset_errors_steps.append(error)
        early_onset_count += int(error < 0)
        late_onset_count += int(error > 0)
        within_one_step_count += int(abs(error) <= 1)
        within_two_step_count += int(abs(error) <= 2)
    onset_abs = [abs(value) for value in onset_errors_steps]
    onset_sorted = sorted(onset_abs)
    onset_p95 = (
        float(onset_sorted[math.ceil(0.95 * len(onset_sorted)) - 1])
        if onset_sorted else 0.0
    )
    onset_mean = (
        float(sum(onset_errors_steps) / len(onset_errors_steps))
        if onset_errors_steps else 0.0
    )
    onset_mean_abs = (
        float(sum(onset_abs) / len(onset_abs)) if onset_abs else 0.0
    )
    target_onset_episode_count = batch - missing_target_onsets

    def region_rate(event: Tensor, region: Tensor) -> float:
        selected = close_mask & region
        if not bool(selected.any()):
            return 0.0
        return float((event & selected).sum().item() / selected.sum().item())

    bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
    bound_violations = int(
        (
            torch.linalg.vector_norm(prediction_xyz, dim=-1)
            > bound + 1.0e-8
        ).sum().item()
    )
    metrics = {
        "xyz/rmse_mm": rmse_mm,
        "xyz/direction_error_deg": direction_deg,
        "xyz/directional_target_rows": float(directional_rows),
        "xyz/direction_metric_available": float(directional_rows > 0),
        "xyz/valid_direct_rows": float(xyz_mask.sum().item()),
        "close/precision": precision,
        "close/recall": recall,
        "close/f1": f1,
        "close/edge_precision": precision if edge_mode else 0.0,
        "close/edge_recall": recall if edge_mode else 0.0,
        "close/edge_f1": f1 if edge_mode else 0.0,
        "close/edge_pr_auc": edge_pr_auc,
        "close/first_onset_error_mean_steps": onset_mean,
        "close/first_onset_error_mean_ms": onset_mean * 20.0,
        "close/first_onset_error_mean_abs_steps": onset_mean_abs,
        "close/first_onset_error_mean_abs_ms": onset_mean_abs * 20.0,
        "close/first_onset_error_p95_abs_steps": onset_p95,
        "close/first_onset_error_p95_abs_ms": onset_p95 * 20.0,
        "close/first_onset_error_max_abs_steps": float(max(onset_abs, default=0)),
        "close/first_onset_error_max_abs_ms": float(max(onset_abs, default=0)) * 20.0,
        "close/first_onset_matched_episode_count": float(len(onset_errors_steps)),
        "close/first_onset_missing_prediction_count": float(missing_predicted_onsets),
        "close/first_onset_missing_target_count": float(missing_target_onsets),
        "close/onset_within_1_step_rate": (
            within_one_step_count / target_onset_episode_count
            if target_onset_episode_count else 0.0
        ),
        "close/onset_within_2_step_rate": (
            within_two_step_count / target_onset_episode_count
            if target_onset_episode_count else 0.0
        ),
        "close/early_close_rate": (
            early_onset_count / target_onset_episode_count
            if edge_mode and target_onset_episode_count
            else region_rate(predicted_close, targets.early_region_mask)
        ),
        # A late close is a temporal false negative: the label says CLOSE but
        # the calibrated hysteresis output is still OPEN in the late region.
        "close/late_close_rate": (
            late_onset_count / target_onset_episode_count
            if edge_mode and target_onset_episode_count
            else region_rate(~predicted_close & (label == 1.0), targets.late_region_mask)
        ),
        "close/false_close_far_rate": region_rate(
            predicted_close & (label == 0.0), targets.far_region_mask
        ),
        "close/early_region_rows": float((close_mask & targets.early_region_mask).sum().item()),
        "close/late_region_rows": float((close_mask & targets.late_region_mask).sum().item()),
        "close/far_region_rows": float((close_mask & targets.far_region_mask).sum().item()),
        "close/late_metric_available": float(
            bool((close_mask & targets.late_region_mask).any())
        ),
        "close/far_metric_available": float(
            bool((close_mask & targets.far_region_mask).any())
        ),
        "close/valid_temporal_rows": float(close_mask.sum().item()),
        "action/bound_violation_count": float(bound_violations),
        "action/radial_projection_count": float(
            output.xyz_projection_applied.sum().item()
        ),
        "integrity/nonfinite_count": 0.0,
        "scale/open_threshold_validation_selected": calibration.open_threshold,
        "scale/close_threshold_validation_selected": calibration.close_threshold,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"non-finite human grasp BC metric: {metrics}")
    return metrics


class HumanGraspBCTrainer:
    """Small offline trainer with explicit split and weight receipts."""

    def __init__(
        self,
        model: HumanGraspGRUBC,
        *,
        split: EpisodeSplit,
        positive_weight: ClosePositiveWeightReceipt,
        learning_rate: float = 3.0e-4,
        objective: str = "FULL_BC",
        optimizer_parameter_groups: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if positive_weight.split_fingerprint != split.fingerprint:
            raise HumanGraspBCContractError("trainer close weight/split mismatch")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise HumanGraspBCContractError("learning_rate must be finite and positive")
        if objective not in ("FULL_BC", "CLOSE_EDGE_ONLY"):
            raise HumanGraspBCContractError("unknown BC optimization objective")
        self.model = model
        self.split = split
        self.positive_weight = positive_weight
        self.objective = objective
        if optimizer_parameter_groups is None:
            trainable = tuple(
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            )
            if not trainable:
                raise HumanGraspBCContractError("optimizer has no trainable parameters")
            self.optimizer = torch.optim.Adam(trainable, lr=learning_rate)
        else:
            groups = tuple(optimizer_parameter_groups)
            if not groups:
                raise HumanGraspBCContractError("optimizer parameter groups are empty")
            self.optimizer = torch.optim.Adam(groups, lr=learning_rate)
        self.optimizer_step = 0

    def train_batch(
        self,
        inputs: HumanGraspSequenceInputs,
        targets: HumanGraspSequenceTargets,
        *,
        batch_episode_ids: Sequence[str],
        privileged_targets: PrivilegedFeasibilityTargets | None = None,
    ) -> dict[str, float]:
        assert_batch_partition(batch_episode_ids, self.split, "train")
        self.model.train()
        output = self.model(inputs)
        losses = compute_human_grasp_bc_loss(
            output,
            targets,
            config=self.model.config,
            positive_weight=self.positive_weight,
            split=self.split,
            privileged_targets=privileged_targets,
        )
        self.optimizer.zero_grad(set_to_none=True)
        objective_loss = (
            losses.close_weighted
            if self.objective == "CLOSE_EDGE_ONLY"
            else losses.total
        )
        objective_loss.backward()
        trainable = tuple(
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0.0:
            raise FloatingPointError("BC gradient is zero or non-finite")
        self.optimizer.step()
        self.optimizer_step += 1
        return {
            "trainer/optimizer_step": float(self.optimizer_step),
            "loss/optimization_objective": float(objective_loss.detach()),
            "gradient/model_norm": float(gradient_norm),
            "scale/lambda_xyz": self.model.config.lambda_xyz,
            "scale/lambda_close": self.model.config.lambda_close,
            "scale/lambda_privileged_feasibility": (
                self.model.config.lambda_privileged_feasibility
            ),
            "runtime/offline_num_envs": 0.0,
            "runtime/batch_size": float(inputs.right_wrist_rgb.shape[0]),
            "action/radial_projection_count": float(
                output.xyz_projection_applied.sum().item()
            ),
            **losses.metrics(),
        }


def save_human_grasp_checkpoint(
    path: str | Path,
    *,
    trainer: HumanGraspBCTrainer,
    calibration: HysteresisCalibration,
    evaluation_metrics: Mapping[str, float],
) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    if calibration.validation_split_fingerprint != trainer.split.fingerprint:
        raise HumanGraspBCContractError("checkpoint calibration/split mismatch")
    if any(
        not bool(torch.isfinite(value.detach()).all())
        for value in trainer.model.state_dict().values()
    ):
        raise HumanGraspBCContractError("checkpoint model contains NaN/Inf")
    metrics = {str(name): float(value) for name, value in evaluation_metrics.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise HumanGraspBCContractError("checkpoint metrics contain NaN/Inf")
    payload = {
        "schema": HUMAN_GRASP_GRU_CHECKPOINT_SCHEMA,
        "config": asdict(trainer.model.config),
        "contract": trainer.model.config.contract_payload(),
        "state_dict": trainer.model.state_dict(),
        "state_sha256": tensor_state_sha256(trainer.model),
        "optimizer_step": trainer.optimizer_step,
        "episode_split": trainer.split.as_dict(),
        "episode_split_fingerprint": trainer.split.fingerprint,
        "close_positive_weight": trainer.positive_weight.as_dict(),
        "hysteresis_calibration": calibration.as_dict(),
        "evaluation_metrics": metrics,
        "close_temporal_branch": trainer.model.close_temporal_branch_config,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest


def load_human_grasp_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HumanGraspGRUBC, dict[str, Any]]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location=device, weights_only=True)
    except Exception as error:
        raise HumanGraspBCContractError("human grasp BC checkpoint is unreadable") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != HUMAN_GRASP_GRU_CHECKPOINT_SCHEMA:
        raise HumanGraspBCContractError("human grasp BC checkpoint schema mismatch")
    config = HumanGraspGRUConfig(**dict(payload.get("config", {})))
    if payload.get("contract") != config.contract_payload():
        raise HumanGraspBCContractError("human grasp BC checkpoint contract mismatch")
    split_payload = payload.get("episode_split")
    if not isinstance(split_payload, Mapping):
        raise HumanGraspBCContractError("checkpoint episode split missing")
    split = EpisodeSplit(
        train=tuple(split_payload.get("train", ())),
        validation=tuple(split_payload.get("validation", ())),
        test=tuple(split_payload.get("test", ())),
        seed=int(split_payload.get("seed", -1)),
    )
    if payload.get("episode_split_fingerprint") != split.fingerprint:
        raise HumanGraspBCContractError("checkpoint episode split fingerprint mismatch")
    weight = ClosePositiveWeightReceipt(**dict(payload.get("close_positive_weight", {})))
    if weight.split_fingerprint != split.fingerprint:
        raise HumanGraspBCContractError("checkpoint positive weight split mismatch")
    calibration = HysteresisCalibration(**dict(payload.get("hysteresis_calibration", {})))
    if calibration.validation_split_fingerprint != split.fingerprint:
        raise HumanGraspBCContractError("checkpoint hysteresis split mismatch")
    model = HumanGraspGRUBC(config).to(device)
    close_temporal_branch = payload.get("close_temporal_branch")
    if close_temporal_branch is not None:
        if not isinstance(close_temporal_branch, Mapping):
            raise HumanGraspBCContractError(
                "checkpoint CLOSE temporal branch metadata is invalid"
            )
        try:
            model.enable_close_temporal_branch(
                hidden_dim=int(close_temporal_branch["hidden_dim"]),
                layers=int(close_temporal_branch["layers"]),
            )
        except Exception as error:
            raise HumanGraspBCContractError(
                "checkpoint CLOSE temporal branch construction failed"
            ) from error
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as error:
        raise HumanGraspBCContractError("human grasp BC state_dict mismatch") from error
    if payload.get("state_sha256") != tensor_state_sha256(model):
        raise HumanGraspBCContractError("human grasp BC tensor hash mismatch")
    if any(
        not bool(torch.isfinite(value.detach()).all())
        for value in model.state_dict().values()
    ):
        raise HumanGraspBCContractError("human grasp BC checkpoint contains NaN/Inf")
    evaluation_metrics = payload.get("evaluation_metrics")
    if not isinstance(evaluation_metrics, Mapping) or not all(
        isinstance(name, str)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for name, value in evaluation_metrics.items()
    ):
        raise HumanGraspBCContractError("checkpoint evaluation metrics are invalid")
    return model, {
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "state_sha256": payload["state_sha256"],
        "optimizer_step": int(payload.get("optimizer_step", -1)),
        "split": split,
        "positive_weight": weight,
        "calibration": calibration,
        "evaluation_metrics": dict(evaluation_metrics),
        "close_temporal_branch": close_temporal_branch,
    }


__all__ = [
    "ACTOR_INPUT_FIELDS",
    "DEFAULT_OFFLINE_BATCH_SIZE",
    "EpisodeSplit",
    "FeasibilityCalibration",
    "HUMAN_GRASP_GRU_BC_SCHEMA",
    "HUMAN_GRASP_GRU_CHECKPOINT_SCHEMA",
    "HumanGraspBCContractError",
    "HumanGraspBCLoss",
    "HumanGraspBCTrainer",
    "HumanGraspGRUBC",
    "HumanGraspGRUConfig",
    "HumanGraspGRUOutput",
    "HumanGraspSequenceInputs",
    "HumanGraspSequenceTargets",
    "HysteresisCalibration",
    "MIGRATED_INPUT_SOURCE_FIELDS",
    "OFFLINE_NUM_ENVS",
    "PROHIBITED_ACTOR_INPUT_TOKENS",
    "PrivilegedFeasibilityTargets",
    "ROBOT_STATE_DIM",
    "VALID_DIRECT_XYZ_MASK_AUTHORITY",
    "assert_batch_partition",
    "apply_close_feasibility_gate",
    "calibrate_feasibility_on_validation",
    "calibrate_hysteresis_on_validation",
    "compute_human_grasp_bc_loss",
    "derive_train_close_positive_weight",
    "evaluate_human_grasp_bc",
    "inputs_from_migrated_legacy_window",
    "load_human_grasp_checkpoint",
    "mask_targets_for_input_validity",
    "save_human_grasp_checkpoint",
    "split_episode_ids",
    "tensor_state_sha256",
    "targets_from_migrated_legacy_window",
]
