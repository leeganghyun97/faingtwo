# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Runtime contract for the cuRobo-only contact-free visual BC student."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contact_free_training_contract import CONTACT_FREE_MAX_DELTA_M


CONTACT_FREE_VISUAL_BC_RUNTIME_SCHEMA = "g2_contact_free_visual_bc_runtime_v1"
IMAGE_HW = (48, 64)
PROPRIO_DIM = 33
_FLOAT32_RADIAL_GUARD_ULPS = 8
_model_output_bound = np.float32(CONTACT_FREE_MAX_DELTA_M)
for _ in range(_FLOAT32_RADIAL_GUARD_ULPS):
    _model_output_bound = np.nextafter(
        _model_output_bound, np.float32(0.0), dtype=np.float32
    )
MODEL_OUTPUT_MAX_DELTA_M = float(_model_output_bound)


class ContactFreeVisualBC(nn.Module):
    """Exact runtime twin of the canonical contact-free BC architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(96 + PROPRIO_DIM),
            nn.Linear(96 + PROPRIO_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, image: Tensor, proprio: Tensor) -> Tensor:
        xyz = torch.tanh(self.head(torch.cat((self.vision(image), proprio), dim=-1)))
        norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
        xyz = xyz * torch.clamp(
            MODEL_OUTPUT_MAX_DELTA_M / norm.clamp_min(1.0e-12), max=1.0
        )
        return torch.cat((xyz, torch.zeros_like(xyz[:, :1])), dim=-1)


def load_contact_free_visual_bc_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[ContactFreeVisualBC, dict[str, Any]]:
    """Load the existing cuRobo-only BC checkpoint without runner coupling.

    The live planner runner and the unified grasp router intentionally share
    this one loader.  This prevents a second checkpoint interpretation from
    becoming an accidental parallel policy pipeline.
    """

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise ValueError("contact-free BC checkpoint is missing")
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("contact-free BC checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("contact-free BC checkpoint must be a mapping")
    if payload.get("schema") not in {
        "g2_curobo_contact_free_bc_15k_v2",
        "g2_curobo_contact_free_bc_15k_kfold_v1",
    }:
        raise ValueError("contact-free BC checkpoint schema mismatch")
    optimizer_step = payload.get("optimizer_step")
    valid_optimizer_step = bool(
        isinstance(optimizer_step, int)
        and 500 <= optimizer_step <= 15_000
        and optimizer_step % 500 == 0
        and (
            payload.get("schema") == "g2_curobo_contact_free_bc_15k_kfold_v1"
            or optimizer_step == 15_000
        )
    )
    if (
        not valid_optimizer_step
        or payload.get("action_dim") != 4
        or payload.get("action_contract")
        != "[dx,dy,dz,HOLD_OPEN=0],robot_root,m,max_norm=0.0045"
        or payload.get("policy_dt_s") != 0.020
        or payload.get("capture_rate_hz") != 25
        or payload.get("source_domain") != "CUROBO_ONLY"
        or payload.get("contact") is not False
        or payload.get("keyboard") is not False
        or payload.get("episode_envs") != 25
    ):
        raise ValueError("contact-free BC checkpoint contract mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("contact-free BC state_dict is missing")
    model = ContactFreeVisualBC()
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("contact-free BC checkpoint contains NaN/Inf")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(path),
        "sha256": expected_sha256,
        "schema": payload["schema"],
        "optimizer_step": int(optimizer_step),
        "action_dim": 4,
        "action_frame": "robot_root",
        "action_unit": "m",
        "gripper": "HOLD_OPEN",
        "episode_envs": int(payload["episode_envs"]),
        "fold_index": payload.get("fold_index"),
        "strict_state_dict_load": True,
    }


def prepare_runtime_input(
    *,
    right_wrist_rgb: Tensor,
    right_wrist_depth_m: Tensor,
    right_wrist_depth_valid: Tensor,
    ee_pose_robot_root_m_xyzw: Tensor,
    right_arm_joint_position_rad: Tensor,
    right_arm_joint_velocity_rad_s: Tensor,
    right_arm_joint_acceleration_rad_s2: Tensor,
    gripper_state_open: Tensor,
    previous_policy_action_4d_metric_root_m: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build exactly the image/proprio tensors serialized by the BC dataset."""

    rgb = right_wrist_rgb
    depth = right_wrist_depth_m
    valid = right_wrist_depth_valid
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("runtime RGB must be [B,H,W,3]")
    if depth.shape != (*rgb.shape[:3], 1) or valid.shape != depth.shape:
        raise ValueError("runtime depth/valid shapes must match RGB")
    batch = int(rgb.shape[0])
    vector_shapes = {
        "ee_pose": (batch, 7),
        "joint_position": (batch, 7),
        "joint_velocity": (batch, 7),
        "joint_acceleration": (batch, 7),
        "gripper_open": (batch, 1),
        "previous_action": (batch, 4),
    }
    values = {
        "ee_pose": ee_pose_robot_root_m_xyzw,
        "joint_position": right_arm_joint_position_rad,
        "joint_velocity": right_arm_joint_velocity_rad_s,
        "joint_acceleration": right_arm_joint_acceleration_rad_s2,
        "gripper_open": gripper_state_open,
        "previous_action": previous_policy_action_4d_metric_root_m,
    }
    for name, expected in vector_shapes.items():
        value = values[name]
        if tuple(value.shape) != expected or not bool(torch.isfinite(value).all()):
            raise ValueError(f"runtime {name} contract mismatch")
    if not bool(((gripper_state_open == 0.0) | (gripper_state_open == 1.0)).all()):
        raise ValueError("runtime gripper-open state must be binary")
    if not bool((previous_policy_action_4d_metric_root_m[:, 3] == 0.0).all()):
        raise ValueError("runtime previous action must retain HOLD_OPEN")
    if bool(
        (
            torch.linalg.vector_norm(
                previous_policy_action_4d_metric_root_m[:, :3], dim=1
            )
            > CONTACT_FREE_MAX_DELTA_M + 1.0e-8
        ).any()
    ):
        raise ValueError("runtime previous metric action exceeds 4.5 mm")
    image = torch.cat(
        (
            rgb.permute(0, 3, 1, 2).to(torch.float32).div(255.0),
            depth.permute(0, 3, 1, 2).to(torch.float32).clamp(0.0, 2.0).div(2.0),
            valid.permute(0, 3, 1, 2).to(torch.float32).clamp(0.0, 1.0),
        ),
        dim=1,
    )
    image = F.interpolate(image, size=IMAGE_HW, mode="bilinear", align_corners=False)
    proprio = torch.cat(tuple(values.values()), dim=1).to(torch.float32)
    if tuple(proprio.shape) != (batch, PROPRIO_DIM):
        raise ValueError("runtime proprio width mismatch")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("runtime image contains NaN/Inf")
    return image, proprio


def validate_metric_output(action: Tensor) -> None:
    if tuple(action.shape) != (1, 4) or not bool(torch.isfinite(action).all()):
        raise ValueError("BC runtime output must be finite [1,4]")
    if float(action[0, 3].item()) != 0.0:
        raise ValueError("BC runtime output must retain HOLD_OPEN")
    norm = float(torch.linalg.vector_norm(action[0, :3]).item())
    if not math.isfinite(norm) or norm > CONTACT_FREE_MAX_DELTA_M + 1.0e-9:
        raise ValueError("BC runtime XYZ output exceeds 4.5 mm")


__all__ = [
    "CONTACT_FREE_VISUAL_BC_RUNTIME_SCHEMA",
    "ContactFreeVisualBC",
    "IMAGE_HW",
    "MODEL_OUTPUT_MAX_DELTA_M",
    "PROPRIO_DIM",
    "load_contact_free_visual_bc_checkpoint",
    "prepare_runtime_input",
    "validate_metric_output",
]
