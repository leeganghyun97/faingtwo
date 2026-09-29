# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure NumPy oracle for the Stage 1 legacy ``place_workpiece`` task.

This module deliberately has no simulator, Isaac, ROS, Torch, or RLinf imports.
It preserves the reward and termination order in
``RLinf/rlinf/envs/geniesim/tasks/place_workpiece.py`` so the future direct
PhysX environment can be checked against a small deterministic reference.

The legacy task has an important outward-reward inconsistency.  It accumulates
the computed reward (including the one-shot ``+10`` success bonus) into the
episode return, then replaces the terminal success reward exposed to the agent
with ``5``.  :class:`RewardOutputMode` makes that behavior explicit:

``COMPATIBILITY``
    Reproduce the existing replay reward and episode-return mismatch.

``CORRECTED``
    Expose the computed reward unchanged, so replay reward and episode return
    agree.  The component formulas and all success/failure gates stay legacy
    compatible.

Pose convention is ``xyz + quaternion(w, x, y, z)``.  Workpiece and workspace
positions are interpreted in the same world frame; workspace orientation is
intentionally ignored because that is what the legacy task does.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class RewardOutputMode(str, Enum):
    """How a successful terminal transition is exposed to an RL learner."""

    COMPATIBILITY = "compatibility"
    CORRECTED = "corrected"


@dataclass(frozen=True)
class LegacyPlaceTaskConfig:
    """Constants copied from the active legacy RLinf place task.

    Defaults are the checked-out task values.  Fields remain configurable so
    boundary behavior can be unit-tested without hundreds of repeated steps.
    """

    target_relative_position: tuple[float, float, float] = (-0.073, 0.007, 1.185)
    target_workpiece_quaternion_wxyz: tuple[float, float, float, float] = (
        0.1807,
        0.6802,
        0.6847,
        0.1896,
    )
    rivet_height: float = 0.01
    control_hz: float = 30.0
    still_speed_threshold: float = 0.02
    still_steps_required: int = 5
    xy_tolerance: float = 0.02
    z_tolerance: float = 0.01
    orientation_tolerance: float = 0.15
    termination_xy_distance: float = 0.12
    termination_z_drop: float = -0.04
    termination_z_high: float = 0.15
    termination_orientation_difference: float = 2.50
    termination_ee_speed: float = 5.0
    idle_distance_threshold: float = 0.002
    idle_steps_limit: int = 100
    failure_warmup_steps: int = 150
    accumulate_idle_during_failure_warmup: bool = True
    max_episode_steps: int = 300
    alive_scale: float = 5.0
    position_decay: float = 10.0
    orientation_decay: float = 5.0
    below_margin: float = 0.01
    below_scale: float = 20.0
    success_bonus: float = 10.0
    compatibility_success_reward: float = 5.0

    def __post_init__(self) -> None:
        for config_field in fields(self):
            value = np.asarray(getattr(self, config_field.name), dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{config_field.name} must contain finite values")
        if self.still_steps_required <= 0:
            raise ValueError("still_steps_required must be positive")
        if self.idle_steps_limit <= 0:
            raise ValueError("idle_steps_limit must be positive")
        if self.failure_warmup_steps < 0:
            raise ValueError("failure_warmup_steps cannot be negative")
        if not isinstance(self.accumulate_idle_during_failure_warmup, bool):
            raise TypeError(
                "accumulate_idle_during_failure_warmup must be a bool"
            )
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")


@dataclass(frozen=True)
class Stage1PlaceStep:
    """Outputs and diagnostics for one vectorized oracle step.

    All arrays have leading shape ``(num_envs,)``.  ``computed_reward`` is the
    exact component sum.  ``reward`` is what the selected output mode exposes,
    and ``return_increment`` is what the legacy/custom episode-return tracker
    accumulates.
    """

    reward: np.ndarray
    computed_reward: np.ndarray
    return_increment: np.ndarray
    episode_return: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    success: np.ndarray
    failure: np.ndarray
    warmup: np.ndarray
    step_count: np.ndarray
    stable_count: np.ndarray
    idle_count: np.ndarray
    reward_detail: Mapping[str, np.ndarray]
    termination_reasons: tuple[tuple[str, ...], ...]


def _quaternion_angle_difference_wxyz(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Match the legacy sign-invariant quaternion angular distance."""

    first_normalized = first / (
        np.linalg.norm(first, axis=-1, keepdims=True) + np.float32(1e-8)
    )
    second_normalized = second / (
        np.linalg.norm(second, axis=-1, keepdims=True) + np.float32(1e-8)
    )
    dot = np.abs(np.sum(first_normalized * second_normalized, axis=-1))
    dot = np.minimum(dot, np.float32(1.0))
    return np.float32(2.0) * np.arccos(dot)


class Stage1PlaceTaskOracle:
    """Stateful, vectorized reference for legacy reward and episode semantics.

    ``step`` follows the active wrapper's ordering:

    1. increment the step counter;
    2. compute reward and advance the stable counter;
    3. accumulate the *computed* reward;
    4. evaluate failures, idle, warmup, success, and timeout;
    5. optionally replace the outward success reward in compatibility mode;
    6. clear task counters for completed environments.

    ``auto_reset=True`` additionally applies the legacy partial-reset behavior:
    previous positions become invalid for one transition and episode return is
    cleared after the returned result has been snapshotted.
    """

    def __init__(
        self,
        num_envs: int = 1,
        *,
        config: LegacyPlaceTaskConfig | None = None,
        reward_output_mode: RewardOutputMode | str = RewardOutputMode.COMPATIBILITY,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.config = config or LegacyPlaceTaskConfig()
        self.reward_output_mode = RewardOutputMode(reward_output_mode)

        self._target_relative_position = np.asarray(
            self.config.target_relative_position, dtype=np.float32
        )
        if self._target_relative_position.shape != (3,):
            raise ValueError("target_relative_position must contain 3 values")

        self._target_workpiece_quaternion = np.asarray(
            self.config.target_workpiece_quaternion_wxyz, dtype=np.float32
        )
        if self._target_workpiece_quaternion.shape != (4,):
            raise ValueError(
                "target_workpiece_quaternion_wxyz must contain 4 values"
            )
        target_norm = np.linalg.norm(self._target_workpiece_quaternion)
        if not np.isfinite(target_norm) or target_norm <= 0.0:
            raise ValueError("target_workpiece_quaternion_wxyz must be non-zero")
        self._target_workpiece_quaternion /= target_norm

        self._step_counter = np.zeros(self.num_envs, dtype=np.int32)
        self._stable_counter = np.zeros(self.num_envs, dtype=np.int32)
        self._idle_counter = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self._previous_workpiece_position = np.zeros(
            (self.num_envs, 3), dtype=np.float32
        )
        self._previous_ee_position = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._has_previous_workpiece_position = False
        self._has_previous_ee_position = False

    @property
    def target_relative_position(self) -> np.ndarray:
        return self._target_relative_position.copy()

    @property
    def target_workpiece_quaternion_wxyz(self) -> np.ndarray:
        return self._target_workpiece_quaternion.copy()

    @property
    def step_count(self) -> np.ndarray:
        return self._step_counter.copy()

    @property
    def stable_count(self) -> np.ndarray:
        return self._stable_counter.copy()

    @property
    def idle_count(self) -> np.ndarray:
        return self._idle_counter.copy()

    @property
    def episode_return(self) -> np.ndarray:
        return self._episode_return.copy()

    def reset(self, env_ids: Sequence[int] | np.ndarray | None = None) -> None:
        """Reset all environments, or apply legacy partial reset to ``env_ids``."""

        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.intp)
            full_reset = True
        else:
            ids = np.asarray(env_ids, dtype=np.intp)
            if ids.ndim == 0:
                ids = ids.reshape(1)
            if ids.ndim != 1:
                raise ValueError("env_ids must be a one-dimensional sequence")
            if np.any(ids < 0) or np.any(ids >= self.num_envs):
                raise IndexError("env_ids contains an out-of-range environment")
            full_reset = False

        self._step_counter[ids] = 0
        self._stable_counter[ids] = 0
        self._idle_counter[ids] = 0
        self._episode_return[ids] = np.float32(0.0)

        if full_reset:
            self._has_previous_workpiece_position = False
            self._has_previous_ee_position = False
        else:
            if self._has_previous_workpiece_position:
                self._previous_workpiece_position[ids] = np.nan
            if self._has_previous_ee_position:
                self._previous_ee_position[ids] = np.nan

    def step(
        self,
        *,
        workpiece_pose_wxyz: np.ndarray | Sequence[float],
        workspace_pose_wxyz: np.ndarray | Sequence[float],
        ee_position: np.ndarray | Sequence[float],
        ee_linear_velocity: np.ndarray | Sequence[float],
        collecting: bool = False,
        base_terminated: np.ndarray | Sequence[bool] | bool = False,
        base_truncated: np.ndarray | Sequence[bool] | bool = False,
        auto_reset: bool = True,
    ) -> Stage1PlaceStep:
        """Evaluate one transition using legacy reward and termination order.

        Args:
            workpiece_pose_wxyz: World pose(s), shape ``(N, 7)``.
            workspace_pose_wxyz: World pose(s), shape ``(N, 7)``.  Only xyz is
                used, matching the source task.
            ee_position: Right-EE position(s), shape ``(N, 3)``.
            ee_linear_velocity: Right-EE linear velocity, shape ``(N, 3)``.
            collecting: Match the legacy data-collection switch, which disables
                success termination but not failure termination.
            base_terminated: Simulator termination mask to OR into the result.
            base_truncated: Simulator truncation mask to OR with the 300-step
                timeout.  It never passes through failure warmup.
            auto_reset: Apply partial reset after snapshotting completed results.
        """

        wp_pose = self._as_batch("workpiece_pose_wxyz", workpiece_pose_wxyz, 7)
        ws_pose = self._as_batch("workspace_pose_wxyz", workspace_pose_wxyz, 7)
        ee_pos = self._as_batch("ee_position", ee_position, 3)
        ee_velocity = self._as_batch(
            "ee_linear_velocity", ee_linear_velocity, 3
        )
        base_term = self._as_mask("base_terminated", base_terminated)
        base_trunc = self._as_mask("base_truncated", base_truncated)

        self._step_counter += 1

        wp_position = wp_pose[:, :3]
        wp_quaternion = wp_pose[:, 3:7]
        workspace_position = ws_pose[:, :3]
        relative_position = wp_position - workspace_position

        xy_delta = (
            relative_position[:, :2] - self._target_relative_position[None, :2]
        )
        distance_xy = np.linalg.norm(xy_delta, axis=-1)
        difference_z = (
            relative_position[:, 2] - self._target_relative_position[None, 2]
        )
        distance_3d = np.sqrt(distance_xy**2 + difference_z**2)
        orientation_difference = _quaternion_angle_difference_wxyz(
            wp_quaternion,
            self._target_workpiece_quaternion[None, :],
        )

        cfg = self.config
        reward_alive = (
            np.float32(cfg.alive_scale)
            * np.exp(-np.float32(cfg.position_decay) * distance_3d)
            * np.exp(-np.float32(cfg.orientation_decay) * orientation_difference)
        ).astype(np.float32, copy=False)
        overshoot = np.maximum(
            -difference_z - np.float32(cfg.below_margin), np.float32(0.0)
        )
        reward_below = (-np.float32(cfg.below_scale) * overshoot).astype(
            np.float32, copy=False
        )

        if self._has_previous_workpiece_position:
            workpiece_speed = (
                np.linalg.norm(
                    wp_position - self._previous_workpiece_position, axis=-1
                )
                * np.float32(cfg.control_hz)
            )
        else:
            workpiece_speed = np.zeros(self.num_envs, dtype=np.float32)
        self._previous_workpiece_position[:] = wp_position
        self._has_previous_workpiece_position = True

        xy_aligned = distance_xy < np.float32(cfg.xy_tolerance)
        orientation_ok = orientation_difference < np.float32(
            cfg.orientation_tolerance
        )
        near_target = (
            xy_aligned
            & (np.abs(difference_z) < np.float32(cfg.z_tolerance))
            & orientation_ok
        )
        still_ok = workpiece_speed < np.float32(cfg.still_speed_threshold)
        stable_now = near_target & still_ok
        self._stable_counter[stable_now] += 1
        self._stable_counter[~stable_now] = 0

        success = self._stable_counter >= cfg.still_steps_required
        just_succeeded = self._stable_counter == cfg.still_steps_required
        reward_success = np.where(
            just_succeeded,
            np.float32(cfg.success_bonus),
            np.float32(0.0),
        ).astype(np.float32, copy=False)
        computed_reward = (reward_alive + reward_below + reward_success).astype(
            np.float32, copy=False
        )

        # The legacy wrapper updates its custom return before replacing the
        # externally visible reward on successful terminal transitions.
        return_increment = computed_reward.copy()
        self._episode_return += return_increment

        drop = difference_z < np.float32(cfg.termination_z_drop)
        z_high = difference_z > np.float32(cfg.termination_z_high)
        xy_far = distance_xy > np.float32(cfg.termination_xy_distance)
        orientation_bad = orientation_difference > np.float32(
            cfg.termination_orientation_difference
        )
        ee_speed = np.linalg.norm(ee_velocity, axis=-1)
        ee_too_fast = ee_speed > np.float32(cfg.termination_ee_speed)

        warmup = self._step_counter < cfg.failure_warmup_steps
        idle_tracking_enabled = (
            np.ones(self.num_envs, dtype=bool)
            if cfg.accumulate_idle_during_failure_warmup
            else ~warmup
        )
        if self._has_previous_ee_position:
            ee_displacement = np.linalg.norm(
                ee_pos - self._previous_ee_position, axis=-1
            )
            not_near_target = (distance_xy > np.float32(cfg.xy_tolerance)) | (
                np.abs(difference_z) > np.float32(cfg.z_tolerance)
            )
            idle_increment = (
                ee_displacement < np.float32(cfg.idle_distance_threshold)
            ) & not_near_target & idle_tracking_enabled
            self._idle_counter[idle_increment] += 1
            self._idle_counter[~idle_increment] = 0
        else:
            ee_displacement = np.full(self.num_envs, np.nan, dtype=np.float32)
            idle_increment = np.zeros(self.num_envs, dtype=bool)
        self._previous_ee_position[:] = ee_pos
        self._has_previous_ee_position = True

        idle = self._idle_counter >= cfg.idle_steps_limit
        failure_candidate = drop | z_high | xy_far | orientation_bad | ee_too_fast | idle
        task_failure = failure_candidate & ~warmup
        task_terminated = task_failure.copy()
        if not collecting:
            task_terminated |= success

        terminated = base_term | task_terminated
        timeout = self._step_counter >= cfg.max_episode_steps
        truncated = base_trunc | timeout
        done = terminated | truncated

        outward_reward = computed_reward.copy()
        compatibility_override = success & task_terminated
        if self.reward_output_mode is RewardOutputMode.COMPATIBILITY:
            outward_reward[compatibility_override] = np.float32(
                cfg.compatibility_success_reward
            )

        # This mirrors the task's unused ``fail_term = term & ~success_mask``.
        failure = task_terminated & ~success
        details = {
            "r_alive": reward_alive.copy(),
            "r_below": reward_below.copy(),
            "r_success": reward_success.copy(),
            "dist_3d": distance_3d.copy(),
            "dist_xy": distance_xy.copy(),
            "diff_z": difference_z.copy(),
            "orient_diff": orientation_difference.copy(),
            "workpiece_speed": workpiece_speed.copy(),
            "ee_speed": ee_speed.copy(),
            "near_target": near_target.copy(),
            "still_ok": still_ok.copy(),
            "failure_candidate": failure_candidate.copy(),
            "idle_tracking_enabled": idle_tracking_enabled.copy(),
            "task_failure": task_failure.copy(),
            "timeout": timeout.copy(),
        }
        reasons = self._termination_reasons(
            success=success,
            drop=drop,
            z_high=z_high,
            xy_far=xy_far,
            orientation_bad=orientation_bad,
            ee_too_fast=ee_too_fast,
            idle=idle,
            warmup=warmup,
            task_terminated=task_terminated,
            base_terminated=base_term,
            truncated=truncated,
            timeout=timeout,
        )

        result = Stage1PlaceStep(
            reward=outward_reward.copy(),
            computed_reward=computed_reward.copy(),
            return_increment=return_increment.copy(),
            episode_return=self._episode_return.copy(),
            terminated=terminated.copy(),
            truncated=truncated.copy(),
            done=done.copy(),
            success=success.copy(),
            failure=failure.copy(),
            warmup=warmup.copy(),
            step_count=self._step_counter.copy(),
            stable_count=self._stable_counter.copy(),
            idle_count=self._idle_counter.copy(),
            reward_detail=details,
            termination_reasons=reasons,
        )

        # The source wrapper clears these task counters on every done, whether
        # or not automatic simulator reset is requested.
        self._step_counter[done] = 0
        self._stable_counter[done] = 0
        self._idle_counter[done] = 0
        if auto_reset and np.any(done):
            self.reset(np.flatnonzero(done))

        return result

    def _as_batch(
        self,
        name: str,
        values: np.ndarray | Sequence[float],
        width: int,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1 and self.num_envs == 1:
            array = array.reshape(1, -1)
        expected = (self.num_envs, width)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _as_mask(
        self,
        name: str,
        values: np.ndarray | Sequence[bool] | bool,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=bool)
        if array.ndim == 0:
            return np.full(self.num_envs, bool(array), dtype=bool)
        if array.shape != (self.num_envs,):
            raise ValueError(
                f"{name} must be scalar or shape {(self.num_envs,)}, "
                f"got {array.shape}"
            )
        return array

    def _termination_reasons(
        self,
        *,
        success: np.ndarray,
        drop: np.ndarray,
        z_high: np.ndarray,
        xy_far: np.ndarray,
        orientation_bad: np.ndarray,
        ee_too_fast: np.ndarray,
        idle: np.ndarray,
        warmup: np.ndarray,
        task_terminated: np.ndarray,
        base_terminated: np.ndarray,
        truncated: np.ndarray,
        timeout: np.ndarray,
    ) -> tuple[tuple[str, ...], ...]:
        all_reasons: list[tuple[str, ...]] = []
        for index in range(self.num_envs):
            reasons: list[str] = []
            if task_terminated[index]:
                if success[index]:
                    reasons.append("success")
                if drop[index]:
                    reasons.append("drop")
                if z_high[index]:
                    reasons.append("z_high")
                if xy_far[index]:
                    reasons.append("xy_far")
                if orientation_bad[index]:
                    reasons.append("orientation_bad")
                if ee_too_fast[index]:
                    reasons.append("ee_speed")
                if idle[index]:
                    reasons.append("idle")
            elif warmup[index] and (
                drop[index]
                or z_high[index]
                or xy_far[index]
                or orientation_bad[index]
                or ee_too_fast[index]
                or idle[index]
            ):
                reasons.append("failure_suppressed_by_warmup")
            if base_terminated[index]:
                reasons.append("base_terminated")
            if truncated[index]:
                reasons.append("timeout" if timeout[index] else "base_truncated")
            all_reasons.append(tuple(reasons))
        return tuple(all_reasons)


__all__ = [
    "LegacyPlaceTaskConfig",
    "RewardOutputMode",
    "Stage1PlaceStep",
    "Stage1PlaceTaskOracle",
]
