# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Raw CLOSE-relative mechanics evidence for Candidate-A supervision.

This module is deliberately separate from the runtime hard-stop detector.
The callback path only clones raw joint position/velocity tensors into a
bounded in-memory ring.  Serialization and all acceleration estimators run
after ``env.step`` has returned (or after the bounded episode has stopped).
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PHYSICS_HZ = 500
PRE_CLOSE_WINDOW_MS = 100
POST_CLOSE_WINDOW_MS = 1500
PRE_CLOSE_SAMPLE_COUNT = PHYSICS_HZ * PRE_CLOSE_WINDOW_MS // 1000
POST_CLOSE_SAMPLE_COUNT = PHYSICS_HZ * POST_CLOSE_WINDOW_MS // 1000
RING_BUFFER_SAMPLE_TARGET = PRE_CLOSE_SAMPLE_COUNT + POST_CLOSE_SAMPLE_COUNT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_derivative(sample_count: int, joint_count: int) -> np.ndarray:
    return np.full((sample_count, joint_count), np.nan, dtype=np.float64)


def acceleration_estimators(
    timestamps_s: np.ndarray,
    qd_rad_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Differentiate qd with three independent, offline estimators.

    The five-point estimate is the standard centered fourth-order stencil.
    The regression estimate is a centered five-sample local linear fit.  Both
    are intentionally unavailable at the window edges instead of silently
    falling back to the two-point estimate.
    """

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    qd = np.asarray(qd_rad_s, dtype=np.float64)
    if timestamps.ndim != 1 or qd.ndim != 2 or qd.shape[0] != timestamps.size:
        raise ValueError("CLOSE_MECHANICS_SERIES_SHAPE_INVALID")
    if timestamps.size and (
        not np.isfinite(timestamps).all()
        or not np.isfinite(qd).all()
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("CLOSE_MECHANICS_SERIES_NONFINITE_OR_NONMONOTONIC")

    sample_count, joint_count = qd.shape
    two_point = _empty_derivative(sample_count, joint_count)
    if sample_count >= 2:
        dt = np.diff(timestamps)
        two_point[1:] = np.diff(qd, axis=0) / dt[:, None]

    five_point = _empty_derivative(sample_count, joint_count)
    regression = _empty_derivative(sample_count, joint_count)
    if sample_count >= 5:
        dt = np.diff(timestamps)
        nominal_dt = float(np.median(dt))
        if not np.allclose(dt, nominal_dt, rtol=0.0, atol=1.0e-9):
            raise ValueError("CLOSE_MECHANICS_5POINT_REQUIRES_UNIFORM_TIMING")
        five_point[2:-2] = (
            qd[:-4] - 8.0 * qd[1:-3] + 8.0 * qd[3:-1] - qd[4:]
        ) / (12.0 * nominal_dt)
        for index in range(2, sample_count - 2):
            x = timestamps[index - 2 : index + 3] - timestamps[index]
            denominator = float(np.dot(x, x))
            if denominator <= 0.0:
                continue
            regression[index] = (x[:, None] * qd[index - 2 : index + 3]).sum(
                axis=0
            ) / denominator
    return {
        "qdd_2point_rad_s2": two_point,
        "qdd_5point_rad_s2": five_point,
        "qdd_regression_rad_s2": regression,
    }


def _phase_summary(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    phase_mask: np.ndarray,
    joint_names: Sequence[str],
    close_timestamp_s: float,
    acceleration_limit_rad_s2: float,
) -> dict[str, Any]:
    eligible = phase_mask[:, None] & np.isfinite(values)
    if not bool(eligible.any()):
        return {
            "available": False,
            "max_abs_qdd_rad_s2": None,
            "joint_index": None,
            "joint_name": None,
            "timestamp_s": None,
            "close_relative_latency_ms": None,
            "threshold_violation": None,
            "first_threshold_crossing": None,
        }
    magnitude = np.where(eligible, np.abs(values), -np.inf)
    flat_index = int(np.argmax(magnitude))
    sample_index, joint_index = np.unravel_index(flat_index, magnitude.shape)
    crossing = eligible & (np.abs(values) > acceleration_limit_rad_s2)
    first = None
    crossing_samples = np.flatnonzero(crossing.any(axis=1))
    if crossing_samples.size:
        first_sample = int(crossing_samples[0])
        first_joint = int(
            np.argmax(np.where(crossing[first_sample], np.abs(values[first_sample]), -1.0))
        )
        first = {
            "sample_index": first_sample,
            "joint_index": first_joint,
            "joint_name": str(joint_names[first_joint]),
            "qdd_rad_s2": float(values[first_sample, first_joint]),
            "timestamp_s": float(timestamps_s[first_sample]),
            "close_relative_latency_ms": float(
                (timestamps_s[first_sample] - close_timestamp_s) * 1000.0
            ),
        }
    return {
        "available": True,
        "max_abs_qdd_rad_s2": float(magnitude[sample_index, joint_index]),
        "joint_index": int(joint_index),
        "joint_name": str(joint_names[joint_index]),
        "timestamp_s": float(timestamps_s[sample_index]),
        "close_relative_latency_ms": float(
            (timestamps_s[sample_index] - close_timestamp_s) * 1000.0
        ),
        "threshold_violation": bool(crossing.any()),
        "first_threshold_crossing": first,
    }


def analyze_close_mechanics(
    *,
    timestamps_s: np.ndarray,
    qd_rad_s: np.ndarray,
    joint_names: Sequence[str],
    close_timestamp_s: float,
    acceleration_limit_rad_s2: float,
    arm_joint_indices: Sequence[int] | None = None,
    passive_joint_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build CLOSE-relative multi-estimator evidence without changing labels."""

    if not math.isfinite(close_timestamp_s):
        raise ValueError("CLOSE_SUBMIT_TIMESTAMP_INVALID")
    if not math.isfinite(acceleration_limit_rad_s2) or acceleration_limit_rad_s2 <= 0:
        raise ValueError("ACCELERATION_AUTHORITY_INVALID")
    estimates = acceleration_estimators(timestamps_s, qd_rad_s)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    # Centered estimators at exactly CLOSE contain post-CLOSE support and must
    # not be attributed to the causal pre-CLOSE phase.
    pre = timestamps < close_timestamp_s
    post = timestamps > close_timestamp_s
    summaries: dict[str, Any] = {}
    joint_count = len(joint_names)

    def validated_indices(
        values: Sequence[int] | None, *, name: str
    ) -> tuple[int, ...]:
        if values is None:
            return tuple()
        result = tuple(int(value) for value in values)
        if len(set(result)) != len(result) or any(
            value < 0 or value >= joint_count for value in result
        ):
            raise ValueError(f"CLOSE_MECHANICS_{name}_JOINT_INDICES_INVALID")
        return result

    arm_indices = validated_indices(arm_joint_indices, name="ARM")
    passive_indices = validated_indices(passive_joint_indices, name="PASSIVE")

    def group_summary(
        values: np.ndarray, phase_mask: np.ndarray, indices: tuple[int, ...]
    ) -> dict[str, Any]:
        if not indices:
            return {
                "available": False,
                "max_abs_qdd_rad_s2": None,
                "joint_index": None,
                "joint_name": None,
                "timestamp_s": None,
                "close_relative_latency_ms": None,
                "threshold_violation": None,
                "first_threshold_crossing": None,
            }
        selected = np.full_like(values, np.nan)
        selected[:, np.asarray(indices, dtype=np.int64)] = values[
            :, np.asarray(indices, dtype=np.int64)
        ]
        return _phase_summary(
            selected,
            timestamps,
            phase_mask,
            joint_names,
            close_timestamp_s,
            acceleration_limit_rad_s2,
        )

    for name, values in estimates.items():
        summaries[name] = {
            "pre_close": _phase_summary(
                values, timestamps, pre, joint_names, close_timestamp_s,
                acceleration_limit_rad_s2,
            ),
            "post_close": _phase_summary(
                values, timestamps, post, joint_names, close_timestamp_s,
                acceleration_limit_rad_s2,
            ),
            "groups": {
                "arm": {
                    "pre_close": group_summary(values, pre, arm_indices),
                    "post_close": group_summary(values, post, arm_indices),
                },
                "passive": {
                    "pre_close": group_summary(values, pre, passive_indices),
                    "post_close": group_summary(values, post, passive_indices),
                },
            },
        }
    post_flags = [
        summaries[name]["post_close"]["threshold_violation"]
        for name in estimates
    ]
    all_available = all(flag is not None for flag in post_flags)
    unanimous_crossing = bool(
        all_available and len(set(bool(flag) for flag in post_flags)) == 1
    )
    first_events = [
        summaries[name]["post_close"]["first_threshold_crossing"]
        for name in estimates
    ]
    if unanimous_crossing and not any(bool(flag) for flag in post_flags):
        event_agreement = "PASS"
        same_joint = True
        latency_span_ms = 0.0
    elif unanimous_crossing and all(event is not None for event in first_events):
        event_joints = {str(event["joint_name"]) for event in first_events}
        latencies = [float(event["close_relative_latency_ms"]) for event in first_events]
        same_joint = len(event_joints) == 1
        latency_span_ms = max(latencies) - min(latencies)
        # Five 500-Hz samples span 8 ms.  This is estimator support width,
        # not a new mechanics or safety threshold.
        event_agreement = (
            "PASS" if same_joint and latency_span_ms <= 8.0 + 1.0e-9 else "FAIL"
        )
    else:
        event_agreement = "FAIL"
        same_joint = False
        latency_span_ms = None
    agreement = "PASS" if unanimous_crossing else "FAIL"
    two = summaries["qdd_2point_rad_s2"]
    first_post = two["post_close"]["first_threshold_crossing"]
    return {
        "schema": "g2_close_mechanics_multi_estimator_v1",
        "acceleration_limit_rad_s2": float(acceleration_limit_rad_s2),
        "threshold_authority": "EXISTING_RUNTIME_HARD_STOP_AUTHORITY_UNCHANGED",
        "estimators": summaries,
        "estimator_agreement": agreement,
        "estimator_agreement_rule": "POST_CLOSE_THRESHOLD_CROSSING_BOOLEAN_UNANIMOUS",
        "estimator_event_agreement": event_agreement,
        "estimator_event_agreement_rule": (
            "SAME_FIRST_CROSSING_JOINT_AND_LATENCY_SPAN_WITHIN_5_SAMPLE_"
            "500HZ_SUPPORT_WIDTH_8MS"
        ),
        "estimator_event_same_joint": same_joint,
        "estimator_event_latency_span_ms": latency_span_ms,
        "pre_close_accel_violation": any(
            summary["pre_close"]["threshold_violation"] is True
            for summary in summaries.values()
        ),
        "post_close_accel_violation": any(
            summary["post_close"]["threshold_violation"] is True
            for summary in summaries.values()
        ),
        "first_accel_violation_after_close_ms": (
            None if first_post is None else first_post["close_relative_latency_ms"]
        ),
        "first_accel_violation_authority": "QDD_2POINT_EXISTING_RUNTIME_ESTIMATOR",
    }


class CloseMechanicsRingBuffer500Hz:
    """Raw-only callback wrapper for one Candidate-A CLOSE trial."""

    def __init__(
        self,
        *,
        env: Any,
        robot: Any,
        torch_module: Any,
        arm_joint_names: Iterable[str],
        master_joint_name: str,
        passive_joint_names: Iterable[str],
        post_close_window_ms: int = POST_CLOSE_WINDOW_MS,
    ) -> None:
        self.env = env
        self.robot = robot
        self.torch = torch_module
        self.original_update = env.scene.update
        self.joint_names = tuple(str(name) for name in robot.joint_names)
        lookup = {name: index for index, name in enumerate(self.joint_names)}
        self.arm_joint_indices = tuple(lookup[name] for name in arm_joint_names)
        self.master_joint_index = lookup[master_joint_name]
        self.passive_joint_indices = tuple(
            lookup[name] for name in passive_joint_names if name in lookup
        )
        limits = robot.data.joint_pos_limits
        if not isinstance(limits, self.torch.Tensor):
            limits = self.torch.from_dlpack(limits)
        if limits.ndim != 3 or limits.shape[0] != 1 or limits.shape[2] != 2:
            raise RuntimeError("CLOSE_MECHANICS_JOINT_LIMIT_SHAPE_INVALID")
        self._joint_limits_device = limits[0].detach().clone()
        if (
            isinstance(post_close_window_ms, bool)
            or not isinstance(post_close_window_ms, int)
            or post_close_window_ms < POST_CLOSE_WINDOW_MS
            or post_close_window_ms > 10_000
            or (post_close_window_ms * PHYSICS_HZ) % 1000 != 0
        ):
            raise ValueError("CLOSE_MECHANICS_POST_WINDOW_MS_INVALID")
        self._post_close_window_ms = int(post_close_window_ms)
        self._post_close_sample_count = (
            PHYSICS_HZ * self._post_close_window_ms // 1000
        )
        self.installed = False
        self.reset_episode()

    def reset_episode(self) -> None:
        self._pre_samples: deque[dict[str, Any]] = deque(
            maxlen=PRE_CLOSE_SAMPLE_COUNT
        )
        self._post_samples: list[dict[str, Any]] = []
        self._frozen_pre_samples: list[dict[str, Any]] = []
        self._elapsed_s = 0.0
        self._physics_step = 0
        self._control_step = -1
        self.close_submit_step: int | None = None
        self.close_submit_physics_step: int | None = None
        self.close_submit_timestamp_s: float | None = None
        self._unpolled_samples: list[dict[str, Any]] = []
        self._detector_previous_qd: np.ndarray | None = None
        self._hard_stop_event: dict[str, Any] | None = None

    def set_control_step(self, control_step: int) -> None:
        self._control_step = int(control_step)

    def mark_close_submission(self, close_submit_step: int) -> None:
        if self.close_submit_step is not None:
            raise RuntimeError("CLOSE_MECHANICS_DUPLICATE_CLOSE_SUBMISSION")
        self.close_submit_step = int(close_submit_step)
        self.close_submit_physics_step = int(self._physics_step)
        self.close_submit_timestamp_s = float(self._elapsed_s)
        self._frozen_pre_samples = list(self._pre_samples)

    @property
    def post_window_complete(self) -> bool:
        return len(self._post_samples) >= self._post_close_sample_count

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CLOSE_MECHANICS_RING_ALREADY_INSTALLED")

        def wrapped_update(dt: float):
            result = self.original_update(dt)
            self._capture_raw(float(dt))
            return result

        self.env.scene.update = wrapped_update
        self.installed = True

    def _capture_raw(self, dt_s: float) -> None:
        # Callback contract: clone raw q/qd only.  No CPU conversion, qdd,
        # serialization, contact processing, or file I/O is allowed here.
        self._elapsed_s += dt_s
        sample = {
            "timestamp_s": float(self._elapsed_s),
            "physics_step": int(self._physics_step),
            "control_step": int(self._control_step),
            "q_rad": self._clone_device_row(self.robot.data.joint_pos),
            "qd_rad_s": self._clone_device_row(self.robot.data.joint_vel),
        }
        self._physics_step += 1
        if self.close_submit_step is None:
            self._pre_samples.append(sample)
        elif len(self._post_samples) < self._post_close_sample_count:
            self._post_samples.append(sample)
        self._unpolled_samples.append(sample)

    def _clone_device_row(self, value: Any) -> Any:
        tensor = (
            value
            if isinstance(value, self.torch.Tensor)
            else getattr(value, "torch", None)
        )
        if not isinstance(tensor, self.torch.Tensor):
            tensor = self.torch.from_dlpack(value)
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise RuntimeError("CLOSE_MECHANICS_EXPECTED_SINGLE_ENV_JOINT_ROW")
        return tensor[0].detach().clone()

    @property
    def hard_stop_event(self) -> dict[str, Any] | None:
        return None if self._hard_stop_event is None else dict(self._hard_stop_event)

    def poll_hard_stop_post_step(
        self,
        *,
        acceleration_limit_rad_s2: float,
        limit_numerical_tolerance_rad: float,
    ) -> dict[str, Any] | None:
        """Apply the unchanged 2-point authority outside the update callback.

        The callback only appends device-side q/qd clones.  This method is
        called after ``env.step`` returns, synchronizes the at-most-one-control-
        step batch, and latches the first hard-stop event.  It never changes a
        target or advances physics.
        """

        if self._hard_stop_event is not None:
            self._unpolled_samples.clear()
            return self.hard_stop_event
        if (
            not math.isfinite(acceleration_limit_rad_s2)
            or acceleration_limit_rad_s2 <= 0.0
            or not math.isfinite(limit_numerical_tolerance_rad)
            or limit_numerical_tolerance_rad < 0.0
        ):
            raise ValueError("CLOSE_MECHANICS_HARD_STOP_AUTHORITY_INVALID")
        pending = self._unpolled_samples
        self._unpolled_samples = []
        if not pending:
            return None
        q = self.torch.stack([row["q_rad"] for row in pending]).to("cpu").numpy()
        qd = self.torch.stack([row["qd_rad_s"] for row in pending]).to("cpu").numpy()
        limits = self._joint_limits_device.to("cpu").numpy()
        timestamps = np.asarray([row["timestamp_s"] for row in pending], dtype=np.float64)
        for local_index, row in enumerate(pending):
            previous = self._detector_previous_qd
            self._detector_previous_qd = qd[local_index].copy()
            if previous is None:
                continue
            dt_s = (
                float(timestamps[local_index] - timestamps[local_index - 1])
                if local_index > 0
                else 1.0 / float(PHYSICS_HZ)
            )
            qdd = (qd[local_index] - previous) / dt_s
            distance_to_lower = q[local_index] - limits[:, 0]
            distance_to_upper = limits[:, 1] - q[local_index]
            limit_margin = np.minimum(distance_to_lower, distance_to_upper)
            hard_stop_mask = (
                np.isfinite(qdd)
                & (limit_margin <= limit_numerical_tolerance_rad)
                & (np.abs(qdd) > acceleration_limit_rad_s2)
            )
            if not bool(np.any(hard_stop_mask)):
                continue
            joint_index = int(
                np.argmax(np.where(hard_stop_mask, np.abs(qdd), -1.0))
            )
            close_latency_ms = (
                None
                if self.close_submit_timestamp_s is None
                else 1000.0
                * (float(row["timestamp_s"]) - self.close_submit_timestamp_s)
            )
            self._hard_stop_event = {
                "physics_step": int(row["physics_step"]),
                "control_step": int(row["control_step"]),
                "timestamp_s": float(row["timestamp_s"]),
                "relative_to_close_ms": close_latency_ms,
                "joint_index": joint_index,
                "joint_name": self.joint_names[joint_index],
                "q_rad": float(q[local_index, joint_index]),
                "qd_rad_s": float(qd[local_index, joint_index]),
                "qdd_2point_rad_s2": float(qdd[joint_index]),
                "distance_to_lower_limit_rad": float(
                    distance_to_lower[joint_index]
                ),
                "distance_to_upper_limit_rad": float(
                    distance_to_upper[joint_index]
                ),
                "limit_numerical_tolerance_rad": float(
                    limit_numerical_tolerance_rad
                ),
                "acceleration_hard_limit_rad_s2": float(
                    acceleration_limit_rad_s2
                ),
                "detector_stage": "POST_ENV_STEP_FROM_RAW_500HZ_Q_QD",
            }
            break
        return self.hard_stop_event

    def save_and_analyze(
        self,
        output: Path,
        *,
        acceleration_limit_rad_s2: float,
    ) -> dict[str, Any]:
        """Synchronize once, persist raw arrays, then run offline estimators."""

        if self.close_submit_timestamp_s is None:
            return {
                "available": False,
                "reason": "CLOSE_NOT_SUBMITTED",
                "ring_buffer_samples": len(self._pre_samples),
                "pre_close_samples": len(self._pre_samples),
                "post_close_samples": 0,
                "estimator_agreement": "FAIL",
            }
        samples = self._frozen_pre_samples + self._post_samples
        if not samples:
            raise RuntimeError("CLOSE_MECHANICS_RING_EMPTY")
        q = self.torch.stack([row["q_rad"] for row in samples]).to("cpu").numpy()
        qd = self.torch.stack([row["qd_rad_s"] for row in samples]).to("cpu").numpy()
        timestamps = np.asarray([row["timestamp_s"] for row in samples], dtype=np.float64)
        physics_steps = np.asarray([row["physics_step"] for row in samples], dtype=np.int64)
        control_steps = np.asarray([row["control_step"] for row in samples], dtype=np.int64)
        evidence = analyze_close_mechanics(
            timestamps_s=timestamps,
            qd_rad_s=qd,
            joint_names=self.joint_names,
            close_timestamp_s=float(self.close_submit_timestamp_s),
            acceleration_limit_rad_s2=acceleration_limit_rad_s2,
            arm_joint_indices=self.arm_joint_indices,
            passive_joint_indices=self.passive_joint_indices,
        )
        estimates = acceleration_estimators(timestamps, qd)
        joint_limits = self._joint_limits_device.to("cpu").numpy()
        distance_to_lower = q - joint_limits[:, 0][None, :]
        distance_to_upper = joint_limits[:, 1][None, :] - q
        pre_mask = timestamps <= float(self.close_submit_timestamp_s)
        post_mask = timestamps > float(self.close_submit_timestamp_s)

        def enrich_peak_state(summary: dict[str, Any]) -> None:
            """Attach raw state at an estimator peak without changing its verdict."""

            joint_index = summary.get("joint_index")
            timestamp_s = summary.get("timestamp_s")
            if joint_index is None or timestamp_s is None:
                return
            sample_index = int(np.argmin(np.abs(timestamps - float(timestamp_s))))
            joint_index = int(joint_index)
            summary.update(
                {
                    "sample_index": sample_index,
                    "q_rad_at_peak": float(q[sample_index, joint_index]),
                    "qd_rad_s_at_peak": float(qd[sample_index, joint_index]),
                    "qd_rad_s_before_peak": (
                        None
                        if sample_index == 0
                        else float(qd[sample_index - 1, joint_index])
                    ),
                    "qd_rad_s_after_peak": (
                        None
                        if sample_index + 1 >= qd.shape[0]
                        else float(qd[sample_index + 1, joint_index])
                    ),
                    "lower_joint_limit_rad": float(joint_limits[joint_index, 0]),
                    "upper_joint_limit_rad": float(joint_limits[joint_index, 1]),
                    "distance_to_lower_limit_rad": float(
                        distance_to_lower[sample_index, joint_index]
                    ),
                    "distance_to_upper_limit_rad": float(
                        distance_to_upper[sample_index, joint_index]
                    ),
                    "distance_to_nearest_limit_rad": float(
                        min(
                            distance_to_lower[sample_index, joint_index],
                            distance_to_upper[sample_index, joint_index],
                        )
                    ),
                }
            )

        for estimator in evidence["estimators"].values():
            for phase in ("pre_close", "post_close"):
                enrich_peak_state(estimator[phase])
                for group in estimator["groups"].values():
                    enrich_peak_state(group[phase])

        runtime_hard_stop_event = self.hard_stop_event
        if runtime_hard_stop_event is not None:
            event_physics_step = int(runtime_hard_stop_event["physics_step"])
            matching = np.flatnonzero(physics_steps == event_physics_step)
            if matching.size == 1:
                sample_index = int(matching[0])
                joint_index = int(runtime_hard_stop_event["joint_index"])
                runtime_hard_stop_event.update(
                    {
                        "sample_index_in_ring": sample_index,
                        "qd_rad_s_before": (
                            None
                            if sample_index == 0
                            else float(qd[sample_index - 1, joint_index])
                        ),
                        "qd_rad_s_after": (
                            None
                            if sample_index + 1 >= qd.shape[0]
                            else float(qd[sample_index + 1, joint_index])
                        ),
                        "qdd_5point_rad_s2": (
                            None
                            if not np.isfinite(
                                estimates["qdd_5point_rad_s2"][
                                    sample_index, joint_index
                                ]
                            )
                            else float(
                                estimates["qdd_5point_rad_s2"][
                                    sample_index, joint_index
                                ]
                            )
                        ),
                        "qdd_regression_rad_s2": (
                            None
                            if not np.isfinite(
                                estimates["qdd_regression_rad_s2"][
                                    sample_index, joint_index
                                ]
                            )
                            else float(
                                estimates["qdd_regression_rad_s2"][
                                    sample_index, joint_index
                                ]
                            )
                        ),
                        "lower_joint_limit_rad": float(
                            joint_limits[joint_index, 0]
                        ),
                        "upper_joint_limit_rad": float(
                            joint_limits[joint_index, 1]
                        ),
                        "distance_to_nearest_limit_rad": float(
                            min(
                                distance_to_lower[sample_index, joint_index],
                                distance_to_upper[sample_index, joint_index],
                            )
                        ),
                    }
                )

        def max_passive_qd(mask: np.ndarray) -> float | None:
            if not bool(mask.any()) or not self.passive_joint_indices:
                return None
            values = np.abs(
                qd[mask][:, np.asarray(self.passive_joint_indices, dtype=np.int64)]
            )
            return float(np.max(values)) if values.size else None

        evidence["pre_close_max_passive_qd_rad_s"] = max_passive_qd(pre_mask)
        evidence["post_close_max_passive_qd_rad_s"] = max_passive_qd(post_mask)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                timestamp_s=timestamps,
                close_relative_timestamp_s=timestamps - self.close_submit_timestamp_s,
                physics_step=physics_steps,
                control_step=control_steps,
                q_rad=q,
                qd_rad_s=qd,
                joint_limits_rad=joint_limits,
                distance_to_lower_limit_rad=distance_to_lower,
                distance_to_upper_limit_rad=distance_to_upper,
                joint_names=np.asarray(self.joint_names, dtype="U64"),
                arm_joint_indices=np.asarray(self.arm_joint_indices, dtype=np.int64),
                gripper_master_joint_index=np.asarray([self.master_joint_index], dtype=np.int64),
                passive_joint_indices=np.asarray(self.passive_joint_indices, dtype=np.int64),
                close_submit_step=np.asarray([self.close_submit_step], dtype=np.int64),
                close_submit_physics_step=np.asarray(
                    [self.close_submit_physics_step], dtype=np.int64
                ),
                close_submit_timestamp_s=np.asarray(
                    [self.close_submit_timestamp_s], dtype=np.float64
                ),
                **estimates,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        evidence.update(
            {
                "available": True,
                "raw_artifact_path": str(output),
                "raw_artifact_sha256": _sha256(output),
                "ring_buffer_samples": len(samples),
                "pre_close_samples": len(self._frozen_pre_samples),
                "post_close_samples": len(self._post_samples),
                "pre_close_window_ms": PRE_CLOSE_WINDOW_MS,
                "post_close_window_ms": self._post_close_window_ms,
                "post_close_window_complete": self.post_window_complete,
                "runtime_hard_stop_event": runtime_hard_stop_event,
                "physics_hz": PHYSICS_HZ,
                "close_submit_step": self.close_submit_step,
                "close_submit_physics_step": self.close_submit_physics_step,
                "close_submit_timestamp_s": self.close_submit_timestamp_s,
                "joint_names": list(self.joint_names),
                "arm_joint_indices": list(self.arm_joint_indices),
                "gripper_master_joint_index": self.master_joint_index,
                "passive_joint_indices": list(self.passive_joint_indices),
                "callback_computation": "RAW_DEVICE_Q_QD_CLONE_ONLY",
                "post_step_computation": "NPZ_SERIALIZATION_AND_3_QDD_ESTIMATORS",
            }
        )
        return evidence

    def restore(self) -> None:
        if not self.installed:
            return
        self.env.scene.update = self.original_update
        self.installed = False
