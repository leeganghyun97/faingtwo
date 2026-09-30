# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Measured OPEN-restoration contract for Stage-1A vector episodes.

The direct pregrasp source restore deliberately preserves the observed
four-bar state.  That is correct: passive/follower joints are not independent
actuators and must never be written as a reset shortcut.  It also means that a
source restore cannot be treated as an OPEN gripper state merely because a
single abstract OPEN packet was sent.

This small, Isaac-free state machine records the required evidence before a
clone may return to policy/replay:

``SOURCE RESTORE -> canonical OPEN hold -> measured settle -> fresh geometry``.

It deliberately does not decide CLOSE readiness, alter a threshold, or change
the controller.  The live runner supplies measured joint/geometry values after
each ordinary canonical OPEN packet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch

from geniesim.rl.isaaclab.g2_gripper_reset_contract import (
    G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES,
    G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S,
    G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS,
    G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS,
    G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD,
    G2_RIGHT_MASTER_OPEN_TARGET_RAD,
)


RESET_OPEN_RESTORE_SCHEMA = "g2_stage1a_measured_open_restore_v1"
RESET_OPEN_RESTORE_EPISODE_CLOCK_SCHEMA = (
    "g2_stage1a_open_restore_episode_clock_hold_v1"
)


def hold_episode_clocks_for_open_restore(
    env: Any, *, env_ids: Sequence[int]
) -> dict[str, object]:
    """Keep reset-only physics outside the policy episode time budget.

    ``ManagerBasedRLEnv.step`` advances ``episode_length_buf`` even when a
    clone is deliberately held behind the measured OPEN barrier.  An OPEN
    restore may validly need more control packets than the task episode
    limit, so allowing that clock to run can make Isaac auto-reset a clone
    before the gripper reaches OPEN parity.  Only the public episode clock of
    the explicitly pending clones is cleared here; articulation state,
    controller state, termination thresholds, and active clones are left
    untouched.

    The caller invokes this immediately before every reset-only ``env.step``
    and once more at activation.  Consequently the first policy step starts
    from episode time zero while non-timeout termination and safety signals
    remain authoritative.
    """

    selected = tuple(int(value) for value in env_ids)
    num_envs = int(getattr(env, "num_envs", 0))
    if (
        num_envs <= 0
        or not selected
        or len(set(selected)) != len(selected)
        or any(value < 0 or value >= num_envs for value in selected)
    ):
        raise ValueError("RESET_OPEN_RESTORE_EPISODE_CLOCK_ENV_IDS_INVALID")
    raw_lengths = getattr(env, "episode_length_buf", None)
    lengths = (
        raw_lengths
        if isinstance(raw_lengths, torch.Tensor)
        else getattr(raw_lengths, "torch", None)
    )
    if not isinstance(lengths, torch.Tensor) or tuple(lengths.shape) != (num_envs,):
        raise ValueError("RESET_OPEN_RESTORE_EPISODE_CLOCK_BUFFER_INVALID")
    if lengths.dtype == torch.bool or torch.is_floating_point(lengths):
        raise ValueError("RESET_OPEN_RESTORE_EPISODE_CLOCK_DTYPE_INVALID")
    maximum = getattr(env, "max_episode_length", None)
    if maximum is None or int(maximum) <= 1:
        raise ValueError("RESET_OPEN_RESTORE_MAX_EPISODE_LENGTH_INVALID")

    indices = torch.as_tensor(selected, dtype=torch.long, device=lengths.device)
    before = lengths.index_select(0, indices).detach().to("cpu").tolist()
    lengths.index_fill_(0, indices, 0)
    after = lengths.index_select(0, indices).detach().to("cpu").tolist()
    if any(int(value) != 0 for value in after):
        raise ValueError("RESET_OPEN_RESTORE_EPISODE_CLOCK_HOLD_FAILED")
    return {
        "schema": RESET_OPEN_RESTORE_EPISODE_CLOCK_SCHEMA,
        "env_ids": list(selected),
        "episode_length_before_hold_by_env": {
            str(env_id): int(value) for env_id, value in zip(selected, before, strict=True)
        },
        "episode_length_after_hold_by_env": {
            str(env_id): int(value) for env_id, value in zip(selected, after, strict=True)
        },
        "max_episode_length_control_steps": int(maximum),
        "episode_clock_held": True,
        "physical_state_changed": False,
        "controller_or_threshold_changed": False,
    }


@dataclass
class ResetOpenRestoreProgress:
    """Per-environment evidence for one direct-source restore generation."""

    env_id: int
    episode_id: int
    source_sample_id: str
    reset_vector_step: int
    open_command_steps: int = 0
    consecutive_settled_samples: int = 0
    completed: bool = False
    expired: bool = False
    last_receipt: dict[str, object] = field(default_factory=dict)

    def observe(
        self,
        *,
        master_q_rad: float,
        master_qd_rad_s: float,
        passive_q_rad_by_name: Mapping[str, float],
        passive_qd_rad_s_by_name: Mapping[str, float],
        aperture_mm: float | None,
        geometry_valid: bool,
        owner_valid: bool,
        geometry_frame_id: int | None,
        geometry_timestamp_s: float | None,
        geometry_age_ms: float | None,
        geometry_cache_fresh: bool,
        table_clearance_ready: bool = True,
        minimum_primary_pad_table_clearance_m: float | None = None,
    ) -> dict[str, object]:
        """Consume one completed canonical OPEN packet and return its receipt.

        Geometry can be absent until the camera has supplied a frame after the
        direct restore.  Joint settling remains observable in that interval,
        but policy activation waits for a finite, source-fresh geometry
        receipt as well.
        """

        if self.completed or self.expired:
            raise RuntimeError("RESET_OPEN_RESTORE_ALREADY_TERMINAL")
        self.open_command_steps += 1
        master_error_rad = abs(float(master_q_rad) - G2_RIGHT_MASTER_OPEN_TARGET_RAD)
        passive_q_values = {str(name): float(value) for name, value in passive_q_rad_by_name.items()}
        passive_qd_values = {
            str(name): float(value) for name, value in passive_qd_rad_s_by_name.items()
        }
        finite_joints = all(
            math.isfinite(value)
            for value in (
                float(master_q_rad),
                float(master_qd_rad_s),
                *passive_q_values.values(),
                *passive_qd_values.values(),
            )
        )
        master_open_ok = bool(
            finite_joints
            and master_error_rad <= G2_RIGHT_MASTER_OPEN_SETTLE_TOLERANCE_RAD
        )
        velocity_values = (abs(float(master_qd_rad_s)), *map(abs, passive_qd_values.values()))
        max_abs_qd_rad_s = max(velocity_values, default=float("inf"))
        velocity_settled = bool(
            finite_joints
            and max_abs_qd_rad_s <= G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S
        )
        aperture_open_ok = bool(
            aperture_mm is not None
            and math.isfinite(float(aperture_mm))
            and float(aperture_mm) > 0.0
        )
        settled_sample = bool(master_open_ok and velocity_settled and aperture_open_ok)
        self.consecutive_settled_samples = (
            self.consecutive_settled_samples + 1 if settled_sample else 0
        )
        dynamics_ready = bool(
            self.open_command_steps >= G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
            and self.consecutive_settled_samples
            >= G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES
        )
        geometry_fresh = bool(
            geometry_cache_fresh
            and geometry_frame_id is not None
            and geometry_timestamp_s is not None
            and math.isfinite(float(geometry_timestamp_s))
            and geometry_age_ms is not None
            and math.isfinite(float(geometry_age_ms))
            and float(geometry_age_ms) >= -1.0e-3
            and float(geometry_age_ms) <= 40.0 + 1.0e-3
            and bool(geometry_valid)
            and bool(owner_valid)
        )
        table_clearance_finite = bool(
            minimum_primary_pad_table_clearance_m is None
            or math.isfinite(float(minimum_primary_pad_table_clearance_m))
        )
        clearance_ready = bool(table_clearance_finite and table_clearance_ready)
        self.completed = bool(dynamics_ready and geometry_fresh and clearance_ready)
        self.expired = bool(
            not self.completed
            and self.open_command_steps >= G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS
        )
        if self.expired:
            if not master_open_ok:
                failure_reason = "MASTER_OPEN_NOT_SETTLED"
            elif not velocity_settled:
                failure_reason = "PASSIVE_OR_MASTER_VELOCITY_NOT_SETTLED"
            elif not aperture_open_ok:
                failure_reason = "APERTURE_OPEN_PARITY_FAILED"
            elif not geometry_fresh:
                failure_reason = "FRESH_GEOMETRY_RECEIPT_FAILED"
            elif not clearance_ready:
                failure_reason = "OPEN_TABLE_CLEARANCE_NOT_READY"
            else:
                failure_reason = "UNRESOLVED_OPEN_RESTORE_FAILURE"
        else:
            failure_reason = None
        receipt: dict[str, object] = {
            "schema": RESET_OPEN_RESTORE_SCHEMA,
            "env_id": int(self.env_id),
            "episode_id": int(self.episode_id),
            "source_sample_id": self.source_sample_id,
            "reset_vector_step": int(self.reset_vector_step),
            "open_command_steps": int(self.open_command_steps),
            "master_open_target_rad": G2_RIGHT_MASTER_OPEN_TARGET_RAD,
            "master_q_rad": float(master_q_rad),
            "master_qd_rad_s": float(master_qd_rad_s),
            "master_target_error_rad": float(master_error_rad),
            "master_open_ok": master_open_ok,
            "passive_q_rad_by_name": passive_q_values,
            "passive_qd_rad_s_by_name": passive_qd_values,
            "passive_max_abs_qd_rad_s": float(max_abs_qd_rad_s),
            "velocity_settled": velocity_settled,
            "settled_velocity_threshold_rad_s": (
                G2_RESET_SETTLED_MAXIMUM_FD_VELOCITY_RAD_S
            ),
            "aperture_mm": None if aperture_mm is None else float(aperture_mm),
            "aperture_open_ok": aperture_open_ok,
            "settled_sample": settled_sample,
            "consecutive_settled_samples": int(self.consecutive_settled_samples),
            "minimum_consecutive_settled_samples": (
                G2_RESET_MINIMUM_CONSECUTIVE_SETTLED_SAMPLES
            ),
            "minimum_open_command_steps": (
                G2_RIGHT_MASTER_OPEN_SETTLE_MINIMUM_POLICY_STEPS
            ),
            "maximum_open_command_steps": (
                G2_RIGHT_MASTER_OPEN_SETTLE_MAXIMUM_POLICY_STEPS
            ),
            "dynamics_ready": dynamics_ready,
            "geometry_valid": bool(geometry_valid),
            "owner_valid": bool(owner_valid),
            "geometry_frame_id": geometry_frame_id,
            "geometry_timestamp_s": geometry_timestamp_s,
            "geometry_age_ms": geometry_age_ms,
            "geometry_cache_fresh": bool(geometry_cache_fresh),
            "fresh_geometry_receipt": geometry_fresh,
            "minimum_primary_pad_table_clearance_m": (
                None
                if minimum_primary_pad_table_clearance_m is None
                else float(minimum_primary_pad_table_clearance_m)
            ),
            "table_clearance_ready": clearance_ready,
            "open_restore_pass": bool(self.completed),
            "open_restore_expired": bool(self.expired),
            "failure_reason": failure_reason,
        }
        self.last_receipt = receipt
        return receipt
