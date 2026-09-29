# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Independent validator for raw PhysX distal-pad contact evidence.

The live Isaac process only records immutable buffers.  It does not decide
that an averaged ContactSensor point is a pad calibration authority.  This
CPU-only module reconstructs every sensor/filter slice from the six filtered
``RigidContactView.get_contact_data`` tensors, cross-identifies each selected
point against the seven unfiltered ``get_raw_contact_data`` buffers and
CPU-uint64 actor-path resolver, validates a deterministic
opposing-surface sphere fixture, checks temporal/trial repeatability and true
SO(3) pose excitation, and only then emits rows suitable for the measured
contact calibration producer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .g2_asset_dependency_manifest import (
    G2AssetDependencyError,
    verify_g2_asset_dependency_manifest,
)
from .pad_contact_acquisition import (
    QUATERNION_PERSISTENCE_CONTRACT,
    TARGET_RUNTIME_IDENTITY,
)
from .pad_surface_calibration import (
    EXPECTED_G2_ASSET_SHA256,
    LINK_NAMES,
    RAW_PHYSX_CONTACT_API,
    TARGET_PHYSX_RUNTIME,
    PadSurfaceCalibrationError,
)
from .pad_source_freeze_manifest import (
    PAD_INSTALLED_RUNTIME_SOURCE_PATHS,
    PAD_LOCAL_SOURCE_RELATIVE_PATHS,
    load_and_validate_pad_source_freeze_manifest,
)


RAW_EVIDENCE_SCHEMA = "g2_pad_surface_physx_raw_contact_evidence_v2"
RAW_VALIDATION_SCHEMA = "g2_pad_surface_physx_raw_contact_validation_v2"
RESOLVED_PATH_QUERY_AUTHORITY = "RigidContactView.sensor_paths/filter_paths"


@dataclass(frozen=True)
class RawContactValidationThresholds:
    minimum_pose_count: int = 12
    minimum_trials_per_pose: int = 3
    minimum_samples_per_trial: int = 20
    minimum_translation_span_m: float = 0.005
    minimum_rotation_span_rad: float = math.radians(10.0)
    minimum_second_rotation_axis_singular_value_rad: float = math.radians(1.0)
    sphere_surface_residual_m: float = 0.001
    sphere_normal_residual_m: float = 0.0015
    opposing_midpoint_residual_m: float = 0.0015
    opposing_normal_cosine_maximum: float = -0.95
    manifold_spread_m: float = 0.002
    maximum_absolute_separation_m: float = 0.002
    temporal_repeatability_m: float = 0.0015
    inter_trial_repeatability_m: float = 0.002
    link_pose_hold_translation_m: float = 0.0005
    link_pose_hold_rotation_rad: float = math.radians(0.2)
    force_matrix_absolute_tolerance_n: float = 0.05
    force_matrix_relative_tolerance: float = 0.05
    maximum_unfiltered_force_residual_n: float = 0.05
    heldout_rms_residual_m: float = 0.001
    heldout_maximum_residual_m: float = 0.003
    maximum_open_master_error_rad: float = 0.01
    maximum_contact_free_force_n: float = 0.1
    maximum_master_mimic_residual_rad: float = 0.02


def _validate_resolved_pair_path_attestation(
    sensor: Mapping[str, Any],
    *,
    link_name: str,
    environment_count: int,
    label: str,
) -> None:
    """Require an exact sensor-to-filter path bijection for every clone.

    Sim 6 exposes resolved contact filters grouped per sensor.  The live
    capture flattens the single-filter groups only after validating that API
    shape.  This independent validator consequently accepts only the exact
    JSON list emitted by that normalization: one unique sensor and one unique
    same-environment object path, in environment order, with an explicit
    attestation of which live API supplied the paths.
    """

    if sensor.get("resolved_paths_queried_from") != RESOLVED_PATH_QUERY_AUTHORITY:
        raise PadSurfaceCalibrationError(f"{label}_resolved_path_query_authority_invalid")
    sensor_paths = sensor.get("sensor_prim_paths")
    partner_paths = sensor.get("filter_partner_prim_paths")
    for value, name in (
        (sensor_paths, "sensor_prim_paths"),
        (partner_paths, "filter_partner_prim_paths"),
    ):
        if not isinstance(value, list):
            raise PadSurfaceCalibrationError(f"{label}_{name}_type_invalid")
        if len(value) != environment_count:
            raise PadSurfaceCalibrationError(f"{label}_{name}_length_invalid")
        if any(not isinstance(path, str) or not path for path in value):
            raise PadSurfaceCalibrationError(f"{label}_{name}_entry_invalid")
        if len(set(value)) != environment_count:
            raise PadSurfaceCalibrationError(f"{label}_{name}_not_unique")

    expected_sensor_paths = [
        f"/World/envs/env_{env_index}/Robot/{link_name}"
        for env_index in range(environment_count)
    ]
    expected_partner_paths = [
        f"/World/envs/env_{env_index}/Object"
        for env_index in range(environment_count)
    ]
    if sensor_paths != expected_sensor_paths or partner_paths != expected_partner_paths:
        raise PadSurfaceCalibrationError(f"{label}_resolved_path_bijection_invalid")


def _validated_count_start_layout(
    counts: np.ndarray,
    starts: np.ndarray,
    *,
    capacity: int,
    label: str,
) -> list[tuple[int, int]]:
    """Validate a complete PhysX count/start layout, not only one selected row.

    A per-environment slice can look valid while overlapping another slice or
    pointing into an occupied entry owned by another environment.  Such a
    layout makes actor identity ambiguous, so the whole layout is part of the
    evidence contract.
    """

    flat_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    flat_starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if flat_counts.shape != flat_starts.shape or flat_counts.size == 0:
        raise PadSurfaceCalibrationError(f"{label}_count_start_shape_invalid")
    if bool((flat_counts < 0).any()) or bool((flat_starts < 0).any()):
        raise PadSurfaceCalibrationError(f"{label}_count_start_negative")
    slices: list[tuple[int, int]] = []
    occupied: set[int] = set()
    for env_index, (count_value, start_value) in enumerate(
        zip(flat_counts, flat_starts, strict=True)
    ):
        count = int(count_value)
        start = int(start_value)
        if start > capacity or start + count > capacity:
            raise PadSurfaceCalibrationError(
                f"{label}_slice_invalid:{env_index}:{start}:{count}:{capacity}"
            )
        current = set(range(start, start + count))
        if occupied.intersection(current):
            raise PadSurfaceCalibrationError(
                f"{label}_slice_overlap:{env_index}:{start}:{count}"
            )
        occupied.update(current)
        slices.append((start, count))
    if len(occupied) >= capacity:
        raise PadSurfaceCalibrationError(
            f"{label}_raw_buffer_saturated:{len(occupied)}:{capacity}"
        )
    if len(occupied) != int(flat_counts.sum()):
        raise PadSurfaceCalibrationError(f"{label}_occupied_count_mismatch")
    return slices


def _array(value: Any, *, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < len(shape_tail) or result.shape[-len(shape_tail) :] != shape_tail:
        raise PadSurfaceCalibrationError(f"{label}_shape_invalid:{result.shape}")
    if not bool(np.isfinite(result).all()):
        raise PadSurfaceCalibrationError(f"{label}_nonfinite")
    return result


def _quaternion_xyzw(value: Any, *, label: str) -> np.ndarray:
    quaternion = _array(value, shape_tail=(4,), label=label).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-3:
        raise PadSurfaceCalibrationError(f"{label}_not_unit:{norm:.12g}")
    return quaternion / norm


def _rotation_xyzw(value: Any, *, label: str) -> np.ndarray:
    x, y, z, w = _quaternion_xyzw(value, label=label)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _maximum_pairwise_distance(values: Sequence[np.ndarray]) -> float:
    return max(
        (float(np.linalg.norm(left - right)) for i, left in enumerate(values) for right in values[i + 1 :]),
        default=0.0,
    )


def _maximum_rotation_span(values: Sequence[np.ndarray]) -> float:
    return max(
        (_rotation_distance(left, right) for i, left in enumerate(values) for right in values[i + 1 :]),
        default=0.0,
    )


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    angle = float(math.acos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    skew = np.asarray(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    return (angle / (2.0 * math.sin(angle))) * skew


def _raw_pair_slice(
    sensor: Mapping[str, Any], *, env_index: int, label: str
) -> dict[str, np.ndarray | int | bool]:
    buffers = sensor.get("raw_contact_data")
    if not isinstance(buffers, Mapping):
        raise PadSurfaceCalibrationError(f"{label}_raw_buffers_missing")
    forces = _array(buffers.get("force_buffer_n"), shape_tail=(1,), label=f"{label}_forces").reshape(-1)
    points = _array(buffers.get("point_buffer_world_m"), shape_tail=(3,), label=f"{label}_points").reshape(-1, 3)
    normals = _array(buffers.get("normal_buffer_world"), shape_tail=(3,), label=f"{label}_normals").reshape(-1, 3)
    separations = _array(buffers.get("separation_buffer_m"), shape_tail=(1,), label=f"{label}_separations").reshape(-1)
    counts = np.asarray(buffers.get("contact_count_buffer"), dtype=np.int64)
    starts = np.asarray(buffers.get("start_indices_buffer"), dtype=np.int64)
    if counts.ndim != 2 or starts.shape != counts.shape or counts.shape[1] != 1:
        raise PadSurfaceCalibrationError(f"{label}_count_start_shape_invalid")
    if not 0 <= env_index < counts.shape[0]:
        raise PadSurfaceCalibrationError(f"{label}_environment_index_invalid")
    capacity = forces.shape[0]
    if not (points.shape[0] == normals.shape[0] == separations.shape[0] == capacity):
        raise PadSurfaceCalibrationError(f"{label}_raw_capacity_mismatch")
    slices = _validated_count_start_layout(
        counts, starts, capacity=capacity, label=f"{label}_filtered"
    )
    start, count = slices[env_index]
    if count <= 0:
        raise PadSurfaceCalibrationError(f"{label}_contact_slice_invalid:{start}:{count}:{capacity}")
    selection = slice(start, start + count)
    return {
        "forces": forces[selection],
        "points": points[selection],
        "normals": normals[selection],
        "separations": separations[selection],
        "count": count,
        "start": start,
        "capacity": capacity,
        "saturated": False,
    }


def _unfiltered_raw_slice(
    sensor: Mapping[str, Any],
    *,
    env_index: int,
    label: str,
    maximum_absolute_separation_m: float,
    maximum_unexpected_force_n: float,
) -> dict[str, np.ndarray | int]:
    buffers = sensor.get("unfiltered_raw_contact_data")
    if not isinstance(buffers, Mapping):
        raise PadSurfaceCalibrationError(f"{label}_unfiltered_raw_buffers_missing")
    forces = _array(
        buffers.get("force_buffer_n"), shape_tail=(1,), label=f"{label}_unfiltered_forces"
    ).reshape(-1)
    points = _array(
        buffers.get("point_buffer_world_m"), shape_tail=(3,), label=f"{label}_unfiltered_points"
    ).reshape(-1, 3)
    normals = _array(
        buffers.get("normal_buffer_world"), shape_tail=(3,), label=f"{label}_unfiltered_normals"
    ).reshape(-1, 3)
    separations = _array(
        buffers.get("separation_buffer_m"), shape_tail=(1,), label=f"{label}_unfiltered_separations"
    ).reshape(-1)
    counts = np.asarray(buffers.get("contact_count_buffer"), dtype=np.int64).reshape(-1)
    starts = np.asarray(buffers.get("start_indices_buffer"), dtype=np.int64).reshape(-1)
    actor_ids = np.asarray(buffers.get("other_actor_id_buffer"), dtype=np.uint64).reshape(-1)
    actor_paths = buffers.get("other_actor_path_buffer")
    if buffers.get("identity_resolution") != (
        "RigidContactView.get_other_actor_paths_from_ids on CPU wp.uint64 occupied IDs"
    ):
        raise PadSurfaceCalibrationError(
            f"{label}_other_actor_identity_resolution_invalid"
        )
    capacity = forces.shape[0]
    if not isinstance(actor_paths, list) or len(actor_paths) != capacity:
        raise PadSurfaceCalibrationError(f"{label}_other_actor_path_buffer_invalid")
    if not (
        points.shape[0]
        == normals.shape[0]
        == separations.shape[0]
        == actor_ids.shape[0]
        == capacity
    ):
        raise PadSurfaceCalibrationError(f"{label}_unfiltered_raw_capacity_mismatch")
    if counts.shape != starts.shape or not 0 <= env_index < counts.shape[0]:
        raise PadSurfaceCalibrationError(f"{label}_unfiltered_count_start_shape_invalid")
    slices = _validated_count_start_layout(
        counts, starts, capacity=capacity, label=f"{label}_unfiltered"
    )
    start, count = slices[env_index]
    if count <= 0:
        raise PadSurfaceCalibrationError(
            f"{label}_unfiltered_contact_slice_invalid:{start}:{count}:{capacity}"
        )
    # Validate *every occupied entry* in the view.  Looking only at the
    # selected environment would let a malformed /Table contact, zero normal,
    # negative force, or impossible separation hide in a neighbouring slice.
    actor_path_by_id: dict[int, str] = {}
    actor_ids_by_env: dict[int, set[int]] = {}
    for owner_env_index, (owner_start, owner_count) in enumerate(slices):
        expected_partner = f"/World/envs/env_{owner_env_index}/Object"
        unexpected_force_magnitude_sum = 0.0
        for raw_index in range(owner_start, owner_start + owner_count):
            force = float(forces[raw_index])
            normal_norm = float(np.linalg.norm(normals[raw_index]))
            separation = float(separations[raw_index])
            actor_path = actor_paths[raw_index]
            actor_id = int(actor_ids[raw_index])
            if not math.isfinite(force) or force <= 0.0:
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_force_invalid:{owner_env_index}:{raw_index}"
                )
            if not math.isfinite(normal_norm) or abs(normal_norm - 1.0) > 1.0e-2:
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_normal_invalid:{owner_env_index}:{raw_index}"
                )
            if (
                not math.isfinite(separation)
                or abs(separation) > maximum_absolute_separation_m
            ):
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_separation_invalid:{owner_env_index}:{raw_index}"
                )
            if not isinstance(actor_path, str) or not actor_path:
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_partner_identity_missing:{owner_env_index}:{raw_index}"
                )
            if actor_id <= 0:
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_actor_id_invalid:{owner_env_index}:{raw_index}"
                )
            previous_path = actor_path_by_id.setdefault(actor_id, actor_path)
            if previous_path != actor_path:
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_actor_id_path_inconsistent:{actor_id}"
                )
            actor_ids_by_env.setdefault(owner_env_index, set()).add(actor_id)
            if actor_path != expected_partner:
                unexpected_force_magnitude_sum += abs(force)
        if unexpected_force_magnitude_sum > maximum_unexpected_force_n:
            raise PadSurfaceCalibrationError(
                f"{label}_unexpected_unfiltered_force_magnitude_sum:"
                f"{owner_env_index}:{unexpected_force_magnitude_sum:.9g}"
            )
        if any(
            actor_paths[raw_index] != expected_partner
            for raw_index in range(owner_start, owner_start + owner_count)
        ):
            raise PadSurfaceCalibrationError(
                f"{label}_unfiltered_partner_invalid:{owner_env_index}"
            )
    populated_actor_sets = [
        ids for ids in actor_ids_by_env.values() if ids
    ]
    for left_index, left_ids in enumerate(populated_actor_sets):
        for right_ids in populated_actor_sets[left_index + 1 :]:
            if left_ids.intersection(right_ids):
                raise PadSurfaceCalibrationError(
                    f"{label}_unfiltered_actor_id_cross_env_alias"
                )
    selection = slice(start, start + count)
    return {
        "forces": forces[selection],
        "points": points[selection],
        "normals": normals[selection],
        "separations": separations[selection],
        "actor_ids": actor_ids[selection],
        "actor_paths": np.asarray(actor_paths[selection], dtype=object),
        "count": count,
        "start": start,
        "capacity": capacity,
    }


def _fit_offset(samples: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    return np.mean(
        [sample["rotation"].T @ (sample["point"] - sample["position"]) for sample in samples],
        axis=0,
    )


def _residuals(samples: Sequence[Mapping[str, np.ndarray]], offset: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.norm(sample["position"] + sample["rotation"] @ offset - sample["point"])
            for sample in samples
        ],
        dtype=np.float64,
    )


def _validate_trial_lifecycle(
    evidence: Mapping[str, Any],
    *,
    snapshots: Sequence[Mapping[str, Any]],
    thresholds: RawContactValidationThresholds,
    logical_pose_count: int,
    runtime_environment_count: int,
    sequential_single_environment: bool,
) -> list[Mapping[str, Any]]:
    """Validate three truly independent detach/reapproach capture trials."""

    trial_events = evidence.get("trial_events")
    if not isinstance(trial_events, list):
        raise PadSurfaceCalibrationError("raw_contact_trial_events_missing")
    expected_trials = list(range(thresholds.minimum_trials_per_pose))
    logical_pose_indices: list[int | None] = (
        list(range(logical_pose_count)) if sequential_single_environment else [None]
    )
    event_by_key: dict[tuple[int | None, int], Mapping[str, Any]] = {}
    for event in trial_events:
        if not isinstance(event, Mapping):
            raise PadSurfaceCalibrationError("raw_contact_trial_event_invalid")
        trial = int(event.get("trial_index", -1))
        pose_index: int | None = None
        if sequential_single_environment:
            pose_index = int(event.get("pose_index", -1))
            if pose_index not in range(logical_pose_count):
                raise PadSurfaceCalibrationError(
                    f"raw_contact_trial_event_pose_invalid:{pose_index}"
                )
        key = (pose_index, trial)
        if key in event_by_key:
            raise PadSurfaceCalibrationError(
                f"raw_contact_trial_event_duplicate:{pose_index}:{trial}"
            )
        event_by_key[key] = event
    expected_event_keys = {
        (pose_index, trial)
        for pose_index in logical_pose_indices
        for trial in expected_trials
    }
    if set(event_by_key) != expected_event_keys:
        raise PadSurfaceCalibrationError(
            "raw_contact_trial_event_indices_invalid:"
            + ",".join(
                f"{pose_index}:{trial}"
                for pose_index, trial in sorted(
                    event_by_key,
                    key=lambda value: (
                        -1 if value[0] is None else int(value[0]), value[1]
                    ),
                )
            )
        )

    snapshot_keys: set[tuple[int | None, int, int]] = set()
    physics_steps: list[int] = []
    by_key: dict[tuple[int | None, int], list[Mapping[str, Any]]] = {
        (pose_index, trial): []
        for pose_index in logical_pose_indices
        for trial in expected_trials
    }
    for snapshot in snapshots:
        trial = int(snapshot.get("trial_index", -1))
        temporal = int(snapshot.get("temporal_index", -1))
        step = int(snapshot.get("physics_step", -1))
        pose_index = (
            int(snapshot.get("pose_index", -1))
            if sequential_single_environment
            else None
        )
        identity = (pose_index, trial, temporal)
        if identity in snapshot_keys:
            raise PadSurfaceCalibrationError(
                "raw_contact_snapshot_identity_duplicate:"
                f"{pose_index}:{trial}:{temporal}"
            )
        snapshot_keys.add(identity)
        physics_steps.append(step)
        key = (pose_index, trial)
        if key not in by_key:
            raise PadSurfaceCalibrationError(
                f"raw_contact_snapshot_trial_invalid:{pose_index}:{trial}"
            )
        by_key[key].append(snapshot)
    if any(step < 0 for step in physics_steps):
        raise PadSurfaceCalibrationError("raw_contact_snapshot_physics_step_invalid")
    if len(set(physics_steps)) != len(physics_steps):
        raise PadSurfaceCalibrationError("raw_contact_snapshot_physics_step_duplicate")
    if any(right <= left for left, right in zip(physics_steps, physics_steps[1:])):
        raise PadSurfaceCalibrationError(
            "raw_contact_snapshot_physics_step_not_strictly_increasing"
        )

    required_temporal = list(range(thresholds.minimum_samples_per_trial))
    previous_capture_step = -1
    for pose_index, trial in (
        (pose_index, trial)
        for pose_index in logical_pose_indices
        for trial in expected_trials
    ):
        samples = by_key[(pose_index, trial)]
        temporal_values = [int(item.get("temporal_index", -1)) for item in samples]
        if temporal_values != required_temporal:
            raise PadSurfaceCalibrationError(
                "raw_contact_temporal_indices_invalid:"
                f"{pose_index}:{trial}:{temporal_values}"
            )
        sample_steps = [int(item["physics_step"]) for item in samples]
        event = event_by_key[(pose_index, trial)]
        try:
            detach_begin = int(event["detach_begin_step"])
            contact_free_begin = int(event["contact_free_begin_step"])
            contact_free_confirmed = int(event["contact_free_confirmed_step"])
            reapproach_begin = int(event["reapproach_begin_step"])
            onset = int(event["fresh_bilateral_onset_step"])
            persistence = int(event["persistence_satisfied_step"])
            contact_free_samples = int(event["contact_free_sample_count"])
            required_persistence = int(event["required_contact_persistence_steps"])
            maximum_contact_free_force = float(event["maximum_contact_free_force_n"])
            maximum_open_error = float(event["maximum_open_master_error_rad"])
        except (KeyError, TypeError, ValueError) as error:
            raise PadSurfaceCalibrationError(
                f"raw_contact_trial_event_fields_invalid:{trial}"
            ) from error
        if not (
            0 <= detach_begin <= contact_free_begin <= contact_free_confirmed
            < reapproach_begin <= onset <= persistence < sample_steps[0]
        ):
            raise PadSurfaceCalibrationError(
                f"raw_contact_trial_lifecycle_order_invalid:{trial}"
            )
        if sample_steps[-1] <= previous_capture_step:
            raise PadSurfaceCalibrationError(
                f"raw_contact_trial_capture_order_invalid:{trial}"
            )
        previous_capture_step = sample_steps[-1]
        if contact_free_samples < 5 or contact_free_confirmed - contact_free_begin + 1 < 5:
            raise PadSurfaceCalibrationError(
                f"raw_contact_contact_free_evidence_insufficient:{trial}"
            )
        if not math.isfinite(maximum_contact_free_force) or (
            maximum_contact_free_force > thresholds.maximum_contact_free_force_n
        ):
            raise PadSurfaceCalibrationError(
                f"raw_contact_contact_free_force_invalid:{trial}"
            )
        if not math.isfinite(maximum_open_error) or (
            maximum_open_error > thresholds.maximum_open_master_error_rad
        ):
            raise PadSurfaceCalibrationError(
                f"raw_contact_open_master_error_invalid:{trial}"
            )
        if required_persistence <= 0 or persistence - onset + 1 < required_persistence:
            raise PadSurfaceCalibrationError(
                f"raw_contact_fresh_persistence_invalid:{trial}"
            )
        onset_by_env = event.get("fresh_bilateral_onset_step_by_env")
        persistence_by_env = event.get("persistence_satisfied_step_by_env")
        if (
            not isinstance(onset_by_env, list)
            or not isinstance(persistence_by_env, list)
            or len(onset_by_env) != runtime_environment_count
            or len(persistence_by_env) != runtime_environment_count
        ):
            raise PadSurfaceCalibrationError(
                f"raw_contact_per_env_lifecycle_invalid:{trial}"
            )
        for env_index, (env_onset_value, env_persistence_value) in enumerate(
            zip(onset_by_env, persistence_by_env, strict=True)
        ):
            env_onset = int(env_onset_value)
            env_persistence = int(env_persistence_value)
            if not (
                reapproach_begin <= env_onset <= env_persistence
                and env_persistence - env_onset + 1 >= required_persistence
                and env_persistence < sample_steps[0]
            ):
                raise PadSurfaceCalibrationError(
                    f"raw_contact_per_env_persistence_invalid:{trial}:{env_index}"
                )
        if onset != min(int(value) for value in onset_by_env) or persistence != max(
            int(value) for value in persistence_by_env
        ):
            raise PadSurfaceCalibrationError(
                f"raw_contact_aggregate_lifecycle_invalid:{trial}"
            )
        if event.get("fresh_contact_started_after_contact_free") is not True:
            raise PadSurfaceCalibrationError(
                f"raw_contact_fresh_onset_attestation_invalid:{trial}"
            )
        capture_sequence = event.get("capture_sequence")
        expected_sequence = [
            {"temporal_index": index, "physics_step": step}
            for index, step in enumerate(sample_steps)
        ]
        if capture_sequence != expected_sequence:
            raise PadSurfaceCalibrationError(
                f"raw_contact_capture_sequence_invalid:{trial}"
            )

    attestation = evidence.get("passive_mimic_drive_attestation")
    if not isinstance(attestation, Mapping):
        raise PadSurfaceCalibrationError("raw_contact_passive_mimic_attestation_missing")
    required_true = (
        "pass",
        "right_master_independently_driven",
        "passive_mimic_excluded_from_target_writes",
        "runtime_positions_measured",
    )
    for field in required_true:
        if attestation.get(field) is not True:
            raise PadSurfaceCalibrationError(
                f"raw_contact_passive_mimic_attestation_invalid:{field}"
            )
    written = attestation.get("independent_target_write_joint_names")
    passive = attestation.get("passive_or_mimic_joint_names")
    if not isinstance(written, list) or not isinstance(passive, list):
        raise PadSurfaceCalibrationError(
            "raw_contact_passive_mimic_joint_sets_invalid"
        )
    if set(written).intersection(passive):
        raise PadSurfaceCalibrationError(
            "raw_contact_passive_mimic_independently_driven"
        )
    try:
        mimic_residual = float(
            attestation["maximum_abs_master_plus_inner_mimic_residual_rad"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PadSurfaceCalibrationError(
            "raw_contact_passive_mimic_residual_invalid"
        ) from error
    if not math.isfinite(mimic_residual) or (
        mimic_residual > thresholds.maximum_master_mimic_residual_rad
    ):
        raise PadSurfaceCalibrationError(
            "raw_contact_passive_mimic_residual_exceeded"
        )
    return [
        event_by_key[(pose_index, trial)]
        for pose_index in logical_pose_indices
        for trial in expected_trials
    ]


def validate_raw_contact_evidence(
    evidence: Mapping[str, Any],
    *,
    thresholds: RawContactValidationThresholds = RawContactValidationThresholds(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate immutable raw evidence and return direct raw calibration rows."""

    if evidence.get("schema") != RAW_EVIDENCE_SCHEMA:
        raise PadSurfaceCalibrationError("raw_contact_evidence_schema_invalid")
    if evidence.get("source_asset_sha256") != EXPECTED_G2_ASSET_SHA256:
        raise PadSurfaceCalibrationError("raw_contact_evidence_asset_hash_invalid")
    if evidence.get("contact_point_api") != RAW_PHYSX_CONTACT_API:
        raise PadSurfaceCalibrationError("raw_contact_evidence_api_invalid")
    source_asset_path = evidence.get("source_asset_path")
    if not isinstance(source_asset_path, str) or not source_asset_path.strip():
        raise PadSurfaceCalibrationError("raw_contact_evidence_asset_path_invalid")
    provenance = evidence.get("capture_provenance")
    if not isinstance(provenance, Mapping):
        raise PadSurfaceCalibrationError("raw_contact_evidence_provenance_missing")
    exact_provenance = {
        "runtime": TARGET_PHYSX_RUNTIME,
        "production_runtime": TARGET_PHYSX_RUNTIME,
        "production_runtime_match": True,
        "physics_backend": "PhysX",
        "pending_validations": [],
        "contact_point_api": RAW_PHYSX_CONTACT_API,
        "quaternion_native": "xyzw",
        "quaternion_persisted": QUATERNION_PERSISTENCE_CONTRACT,
    }
    for field, expected in exact_provenance.items():
        if provenance.get(field) != expected:
            raise PadSurfaceCalibrationError(
                f"raw_contact_evidence_provenance_invalid:{field}"
            )
    if provenance.get("runtime_identity") != TARGET_RUNTIME_IDENTITY:
        raise PadSurfaceCalibrationError(
            "raw_contact_evidence_runtime_identity_invalid"
        )
    manifest_reference = provenance.get("source_freeze_manifest")
    if not isinstance(manifest_reference, Mapping):
        raise PadSurfaceCalibrationError(
            "raw_contact_source_freeze_manifest_missing"
        )
    manifest_path = manifest_reference.get("path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise PadSurfaceCalibrationError(
            "raw_contact_source_freeze_manifest_path_invalid"
        )
    capture_arguments = provenance.get("capture_arguments")
    if not isinstance(capture_arguments, Mapping):
        raise PadSurfaceCalibrationError("raw_contact_capture_arguments_missing")
    repository_root = Path(__file__).resolve().parents[5]
    _, observed_manifest_reference = load_and_validate_pad_source_freeze_manifest(
        Path(manifest_path),
        expected_runtime_identity=TARGET_RUNTIME_IDENTITY,
        expected_seed=int(provenance.get("seed", -1)),
        expected_capture_command=[
            TARGET_RUNTIME_IDENTITY["python_executable"],
            str(
                repository_root
                / "scripts/diagnostics/capture_g2_pad_surface_contact_authority.py"
            ),
        ],
        expected_capture_arguments=capture_arguments,
        required_local_source_paths=PAD_LOCAL_SOURCE_RELATIVE_PATHS,
        required_installed_source_paths=PAD_INSTALLED_RUNTIME_SOURCE_PATHS,
    )
    for field in ("path", "file_sha256", "document_sha256", "schema"):
        if manifest_reference.get(field) != observed_manifest_reference.get(field):
            raise PadSurfaceCalibrationError(
                f"raw_contact_source_freeze_manifest_mismatch:{field}"
            )
    try:
        verify_g2_asset_dependency_manifest(
            provenance.get("asset_dependency_manifest", {}),
            asset_path=Path(source_asset_path),
        )
    except G2AssetDependencyError as error:
        raise PadSurfaceCalibrationError(
            f"raw_contact_evidence_dependency_manifest_invalid:{error}"
        ) from error
    probe = evidence.get("probe")
    if not isinstance(probe, Mapping) or probe.get("kind") != "analytic_kinematic_sphere":
        raise PadSurfaceCalibrationError("raw_contact_probe_not_analytic_sphere")
    radius = float(probe.get("radius_m", float("nan")))
    if not math.isfinite(radius) or radius <= 0.0:
        raise PadSurfaceCalibrationError("raw_contact_probe_radius_invalid")
    poses = evidence.get("poses")
    snapshots = evidence.get("snapshots")
    if not isinstance(poses, list) or len(poses) < thresholds.minimum_pose_count:
        raise PadSurfaceCalibrationError("raw_contact_pose_count_insufficient")
    if not isinstance(snapshots, list):
        raise PadSurfaceCalibrationError("raw_contact_snapshots_missing")
    capture_schedule = provenance.get("capture_schedule")
    sequential_single_environment = False
    runtime_environment_count = len(poses)
    if capture_schedule is not None:
        if not isinstance(capture_schedule, Mapping):
            raise PadSurfaceCalibrationError("raw_contact_capture_schedule_invalid")
        mode = capture_schedule.get("mode")
        if mode != "SEQUENTIAL_SINGLE_ENVIRONMENT":
            raise PadSurfaceCalibrationError(
                f"raw_contact_capture_schedule_mode_invalid:{mode}"
            )
        sequential_single_environment = True
        if (
            int(capture_schedule.get("runtime_environment_count", -1)) != 1
            or int(capture_schedule.get("logical_pose_count", -1)) != len(poses)
            or int(capture_schedule.get("trials_per_pose", -1))
            != thresholds.minimum_trials_per_pose
            or int(capture_schedule.get("temporal_samples_per_trial", -1))
            != thresholds.minimum_samples_per_trial
            or capture_schedule.get("uninterrupted_safety_stream") is not True
            or capture_schedule.get("safety_stream_reset_between_poses") is not False
        ):
            raise PadSurfaceCalibrationError(
                "raw_contact_capture_schedule_contract_invalid"
            )
        runtime_environment_count = 1
        if len(poses) != thresholds.minimum_pose_count:
            raise PadSurfaceCalibrationError(
                "raw_contact_sequential_logical_pose_count_invalid"
            )
    pose_by_env: dict[int, Mapping[str, Any]] = {}
    runtime_env_by_pose: dict[int, int] = {}
    for pose in poses:
        logical_pose_index = int(
            pose.get("pose_index", -1)
            if sequential_single_environment
            else pose.get("env_index", -1)
        )
        runtime_env_index = int(
            pose.get("runtime_env_index", -1)
            if sequential_single_environment
            else pose.get("env_index", -1)
        )
        if logical_pose_index in pose_by_env or logical_pose_index < 0:
            raise PadSurfaceCalibrationError("raw_contact_pose_environment_invalid")
        if runtime_env_index not in range(runtime_environment_count):
            raise PadSurfaceCalibrationError(
                "raw_contact_pose_runtime_environment_invalid"
            )
        pose_by_env[logical_pose_index] = pose
        runtime_env_by_pose[logical_pose_index] = runtime_env_index
    if sorted(pose_by_env) != list(range(len(pose_by_env))):
        raise PadSurfaceCalibrationError("raw_contact_pose_environment_order_invalid")
    if sequential_single_environment and set(runtime_env_by_pose.values()) != {0}:
        raise PadSurfaceCalibrationError(
            "raw_contact_sequential_runtime_environment_invalid"
        )
    if sequential_single_environment:
        for collection_name, collection in (
            ("trial_event", evidence.get("trial_events", [])),
            ("snapshot", snapshots),
        ):
            for item in collection:
                pose_index = int(item.get("pose_index", -1))
                if pose_index not in pose_by_env:
                    raise PadSurfaceCalibrationError(
                        f"raw_contact_{collection_name}_logical_pose_invalid"
                    )
                if (
                    str(item.get("pose_id", ""))
                    != str(pose_by_env[pose_index].get("pose_id", ""))
                    or int(item.get("runtime_env_index", -1))
                    != runtime_env_by_pose[pose_index]
                ):
                    raise PadSurfaceCalibrationError(
                        f"raw_contact_{collection_name}_pose_binding_invalid"
                    )
    trial_events = _validate_trial_lifecycle(
        evidence,
        snapshots=snapshots,
        thresholds=thresholds,
        logical_pose_count=len(pose_by_env),
        runtime_environment_count=runtime_environment_count,
        sequential_single_environment=sequential_single_environment,
    )

    grouped: dict[tuple[int, int, str], list[dict[str, np.ndarray | float | int]]] = {}
    for snapshot_index, snapshot in enumerate(snapshots):
        trial = int(snapshot.get("trial_index", -1))
        sensors = snapshot.get("sensors")
        sphere_centers = _array(
            snapshot.get("probe_center_world_m"),
            shape_tail=(3,),
            label=f"snapshot_{snapshot_index}_probe_centers",
        )
        if (
            not isinstance(sensors, Mapping)
            or sphere_centers.shape[0] != runtime_environment_count
        ):
            raise PadSurfaceCalibrationError(f"snapshot_{snapshot_index}_shape_invalid")
        snapshot_pose_indices = (
            [int(snapshot.get("pose_index", -1))]
            if sequential_single_environment
            else sorted(pose_by_env)
        )
        if any(pose_index not in pose_by_env for pose_index in snapshot_pose_indices):
            raise PadSurfaceCalibrationError(
                f"snapshot_{snapshot_index}_logical_pose_invalid"
            )
        for link_name in LINK_NAMES:
            sensor = sensors.get(link_name)
            if not isinstance(sensor, Mapping):
                raise PadSurfaceCalibrationError(f"snapshot_{snapshot_index}_{link_name}_missing")
            _validate_resolved_pair_path_attestation(
                sensor,
                link_name=link_name,
                environment_count=runtime_environment_count,
                label=f"snapshot_{snapshot_index}_{link_name}",
            )
            sensor_paths = sensor.get("sensor_prim_paths")
            partner_paths = sensor.get("filter_partner_prim_paths")
            link_positions = _array(
                sensor.get("link_position_world_m"), shape_tail=(3,), label=f"snapshot_{snapshot_index}_link_pos"
            )
            link_quaternions = _array(
                sensor.get("link_quaternion_world"), shape_tail=(4,), label=f"snapshot_{snapshot_index}_link_quat"
            )
            articulation_positions = _array(
                sensor.get("articulation_link_position_world_m"),
                shape_tail=(3,),
                label=f"snapshot_{snapshot_index}_articulation_pos",
            )
            articulation_quaternions = _array(
                sensor.get("articulation_link_quaternion_world"),
                shape_tail=(4,),
                label=f"snapshot_{snapshot_index}_articulation_quat",
            )
            sensor_quaternion_order = str(sensor.get("link_quaternion_order", ""))
            articulation_quaternion_order = str(
                sensor.get("articulation_link_quaternion_order", "")
            )
            # This evidence contract targets Lab3/Sim6 PhysX Warp tensors.
            # Both sources are XYZW.  Missing or legacy WXYZ declarations are
            # rejected instead of guessed and reordered twice.
            if sensor_quaternion_order != "xyzw" or articulation_quaternion_order != "xyzw":
                raise PadSurfaceCalibrationError(
                    f"snapshot_{snapshot_index}_{link_name}_target_runtime_quaternion_order_invalid"
                )
            pair_forces = _array(
                sensor.get("pair_force_matrix_vector_n"),
                shape_tail=(3,),
                label=f"snapshot_{snapshot_index}_pair_force",
            )
            net_forces = _array(
                sensor.get("net_force_vector_n"),
                shape_tail=(3,),
                label=f"snapshot_{snapshot_index}_net_force",
            )
            for logical_pose_index in snapshot_pose_indices:
                runtime_env_index = runtime_env_by_pose[logical_pose_index]
                label = (
                    f"snapshot_{snapshot_index}_pose_{logical_pose_index}_"
                    f"env_{runtime_env_index}_{link_name}"
                )
                expected_sensor = (
                    f"/World/envs/env_{runtime_env_index}/Robot/{link_name}"
                )
                expected_partner = f"/World/envs/env_{runtime_env_index}/Object"
                if (
                    sensor_paths[runtime_env_index] != expected_sensor
                    or partner_paths[runtime_env_index] != expected_partner
                ):
                    raise PadSurfaceCalibrationError(f"{label}_pair_identity_invalid")
                raw = _raw_pair_slice(
                    sensor, env_index=runtime_env_index, label=label
                )
                unfiltered = _unfiltered_raw_slice(
                    sensor,
                    env_index=runtime_env_index,
                    label=label,
                    maximum_absolute_separation_m=(
                        thresholds.maximum_absolute_separation_m
                    ),
                    maximum_unexpected_force_n=(
                        thresholds.maximum_unfiltered_force_residual_n
                    ),
                )
                forces = raw["forces"]
                points = raw["points"]
                normals = raw["normals"]
                separations = raw["separations"]
                normal_norms = np.linalg.norm(normals, axis=1)
                if bool((forces <= 0.0).any()) or float(np.max(np.abs(normal_norms - 1.0))) > 1.0e-2:
                    raise PadSurfaceCalibrationError(f"{label}_force_or_normal_invalid")
                reconstructed = np.sum(forces[:, None] * normals, axis=0)
                pair_force = pair_forces[runtime_env_index]
                force_error = min(
                    float(np.linalg.norm(reconstructed - pair_force)),
                    float(np.linalg.norm(reconstructed + pair_force)),
                )
                force_limit = thresholds.force_matrix_absolute_tolerance_n + (
                    thresholds.force_matrix_relative_tolerance * float(np.linalg.norm(pair_force))
                )
                if force_error > force_limit:
                    raise PadSurfaceCalibrationError(f"{label}_raw_force_matrix_mismatch:{force_error:.9g}")
                center = sphere_centers[runtime_env_index]
                radii = np.linalg.norm(points - center, axis=1)
                sphere_residual = float(np.max(np.abs(radii - radius)))
                signed_normal_residual = np.minimum(
                    np.linalg.norm(points - (center + radius * normals), axis=1),
                    np.linalg.norm(points - (center - radius * normals), axis=1),
                )
                if sphere_residual > thresholds.sphere_surface_residual_m:
                    raise PadSurfaceCalibrationError(f"{label}_sphere_surface_residual:{sphere_residual:.9g}")
                if float(np.max(signed_normal_residual)) > thresholds.sphere_normal_residual_m:
                    raise PadSurfaceCalibrationError(f"{label}_sphere_normal_residual")
                if float(np.max(np.abs(separations))) > thresholds.maximum_absolute_separation_m:
                    raise PadSurfaceCalibrationError(f"{label}_separation_invalid")
                if _maximum_pairwise_distance(list(points)) > thresholds.manifold_spread_m:
                    raise PadSurfaceCalibrationError(f"{label}_contact_manifold_spread")
                raw_partner_paths = unfiltered["actor_paths"]
                raw_partner_points = unfiltered["points"]
                raw_partner_forces = unfiltered["forces"]
                raw_partner_normals = unfiltered["normals"]
                raw_partner_separations = unfiltered["separations"]
                raw_normal_norms = np.linalg.norm(raw_partner_normals, axis=1)
                if bool((raw_partner_forces <= 0.0).any()):
                    raise PadSurfaceCalibrationError(
                        f"{label}_unfiltered_force_invalid"
                    )
                if float(np.max(np.abs(raw_normal_norms - 1.0))) > 1.0e-2:
                    raise PadSurfaceCalibrationError(
                        f"{label}_unfiltered_normal_invalid"
                    )
                if (
                    float(np.max(np.abs(raw_partner_separations)))
                    > thresholds.maximum_absolute_separation_m
                ):
                    raise PadSurfaceCalibrationError(
                        f"{label}_unfiltered_separation_invalid"
                    )
                unexpected_force_magnitude_sum = float(
                    np.sum(
                        np.abs(raw_partner_forces)[
                            np.asarray(raw_partner_paths != expected_partner, dtype=bool)
                        ]
                    )
                )
                if unexpected_force_magnitude_sum > thresholds.maximum_unfiltered_force_residual_n:
                    raise PadSurfaceCalibrationError(
                        f"{label}_unexpected_unfiltered_force_magnitude_sum:"
                        f"{unexpected_force_magnitude_sum:.9g}"
                    )
                if bool(np.any(raw_partner_paths != expected_partner)):
                    raise PadSurfaceCalibrationError(
                        f"{label}_unfiltered_partner_invalid"
                    )
                filtered_to_unfiltered_matches: list[int] = []
                for filtered_index, filtered_point in enumerate(points):
                    candidates = [
                        raw_index
                        for raw_index, actor_path in enumerate(raw_partner_paths)
                        if actor_path == expected_partner
                        and float(
                            np.linalg.norm(raw_partner_points[raw_index] - filtered_point)
                        )
                        <= 1.0e-6
                        and abs(
                            float(raw_partner_forces[raw_index])
                            - float(forces[filtered_index])
                        )
                        <= thresholds.force_matrix_absolute_tolerance_n
                        and float(
                            np.dot(
                                raw_partner_normals[raw_index],
                                normals[filtered_index],
                            )
                        )
                        >= 0.999
                        and abs(
                            float(raw_partner_separations[raw_index])
                            - float(separations[filtered_index])
                        )
                        <= 1.0e-7
                    ]
                    if len(candidates) != 1:
                        raise PadSurfaceCalibrationError(
                            f"{label}_filtered_raw_partner_identity_mismatch:"
                            f"{filtered_index}:{len(candidates)}"
                        )
                    filtered_to_unfiltered_matches.append(candidates[0])
                if (
                    len(set(filtered_to_unfiltered_matches)) != len(points)
                    or len(raw_partner_points) != len(points)
                    or set(filtered_to_unfiltered_matches)
                    != set(range(len(raw_partner_points)))
                ):
                    raise PadSurfaceCalibrationError(
                        f"{label}_filtered_raw_bijection_invalid"
                    )
                unfiltered_reconstructed = np.sum(
                    raw_partner_forces[:, None] * raw_partner_normals, axis=0
                )
                net_force = net_forces[runtime_env_index]
                unfiltered_residual = min(
                    float(np.linalg.norm(unfiltered_reconstructed - net_force)),
                    float(np.linalg.norm(unfiltered_reconstructed + net_force)),
                )
                if unfiltered_residual > thresholds.maximum_unfiltered_force_residual_n:
                    raise PadSurfaceCalibrationError(
                        f"{label}_unfiltered_raw_net_force_mismatch:{unfiltered_residual:.9g}"
                    )
                selected_index = int(np.argmax(forces))
                selected_unfiltered_index = filtered_to_unfiltered_matches[selected_index]
                sensor_rotation = _rotation_xyzw(
                    link_quaternions[runtime_env_index],
                    label=f"{label}_sensor_quaternion",
                )
                articulation_rotation = _rotation_xyzw(
                    articulation_quaternions[runtime_env_index],
                    label=f"{label}_articulation_quaternion",
                )
                if float(
                    np.linalg.norm(
                        link_positions[runtime_env_index]
                        - articulation_positions[runtime_env_index]
                    )
                ) > 0.001:
                    raise PadSurfaceCalibrationError(f"{label}_sensor_articulation_position_mismatch")
                if _rotation_distance(sensor_rotation, articulation_rotation) > math.radians(0.1):
                    raise PadSurfaceCalibrationError(f"{label}_sensor_articulation_rotation_mismatch")
                grouped.setdefault(
                    (logical_pose_index, trial, link_name), []
                ).append(
                    {
                        "point": points[selected_index],
                        "normal": normals[selected_index],
                        "separation": float(separations[selected_index]),
                        "force": float(forces[selected_index]),
                        "position": link_positions[runtime_env_index],
                        "rotation": sensor_rotation,
                        "quaternion_xyzw": _quaternion_xyzw(
                            link_quaternions[runtime_env_index],
                            label=f"{label}_selected_quaternion",
                        ),
                        "physics_step": int(snapshot.get("physics_step", -1)),
                        "sphere_center": center,
                        "raw_count": int(raw["count"]),
                        "unfiltered_force_residual_n": unfiltered_residual,
                        "other_actor_id": int(
                            unfiltered["actor_ids"][selected_unfiltered_index]
                        ),
                        "other_actor_path": str(
                            raw_partner_paths[selected_unfiltered_index]
                        ),
                        "runtime_env_index": runtime_env_index,
                    }
                )

    trial_indices = sorted({key[1] for key in grouped})
    if len(trial_indices) < thresholds.minimum_trials_per_pose:
        raise PadSurfaceCalibrationError("raw_contact_trial_count_insufficient")
    representatives: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for key, samples in grouped.items():
        if len(samples) < thresholds.minimum_samples_per_trial:
            raise PadSurfaceCalibrationError(f"raw_contact_temporal_samples_insufficient:{key}")
        points = [sample["point"] for sample in samples]
        positions = [sample["position"] for sample in samples]
        rotations = [sample["rotation"] for sample in samples]
        if _maximum_pairwise_distance(points) > thresholds.temporal_repeatability_m:
            raise PadSurfaceCalibrationError(f"raw_contact_temporal_point_drift:{key}")
        if _maximum_pairwise_distance(positions) > thresholds.link_pose_hold_translation_m:
            raise PadSurfaceCalibrationError(f"raw_contact_link_translation_drift:{key}")
        if _maximum_rotation_span(rotations) > thresholds.link_pose_hold_rotation_rad:
            raise PadSurfaceCalibrationError(f"raw_contact_link_rotation_drift:{key}")
        median = np.median(np.stack(points), axis=0)
        representatives[key] = min(samples, key=lambda sample: float(np.linalg.norm(sample["point"] - median)))

    # Each pose/trial is a bidirectional registration: the two chosen raw pad
    # points must lie on opposite sides of the same analytic sphere, their
    # midpoint must recover its center, and radial normals must oppose.
    bidirectional_metrics: list[dict[str, float | int]] = []
    for env_index in sorted(pose_by_env):
        for trial in trial_indices:
            inner = representatives[(env_index, trial, LINK_NAMES[0])]
            outer = representatives[(env_index, trial, LINK_NAMES[1])]
            center = inner["sphere_center"]
            midpoint_residual = float(np.linalg.norm(0.5 * (inner["point"] + outer["point"]) - center))
            radial_inner = (inner["point"] - center) / np.linalg.norm(inner["point"] - center)
            radial_outer = (outer["point"] - center) / np.linalg.norm(outer["point"] - center)
            opposing_cosine = float(np.dot(radial_inner, radial_outer))
            if midpoint_residual > thresholds.opposing_midpoint_residual_m:
                raise PadSurfaceCalibrationError("raw_contact_opposing_midpoint_residual")
            if opposing_cosine > thresholds.opposing_normal_cosine_maximum:
                raise PadSurfaceCalibrationError("raw_contact_opposing_normals_invalid")
            bidirectional_metrics.append(
                {
                    "pose_index": env_index,
                    "runtime_env_index": runtime_env_by_pose[env_index],
                    "trial_index": trial,
                    "midpoint_residual_m": midpoint_residual,
                    "radial_cosine": opposing_cosine,
                }
            )

    selected: dict[tuple[int, str], Mapping[str, Any]] = {}
    for env_index in sorted(pose_by_env):
        for link_name in LINK_NAMES:
            values = [representatives[(env_index, trial, link_name)] for trial in trial_indices]
            local_points = [value["rotation"].T @ (value["point"] - value["position"]) for value in values]
            if _maximum_pairwise_distance(local_points) > thresholds.inter_trial_repeatability_m:
                raise PadSurfaceCalibrationError(f"raw_contact_inter_trial_repeatability:{env_index}:{link_name}")
            local_median = np.median(np.stack(local_points), axis=0)
            selected[(env_index, link_name)] = min(
                values,
                key=lambda sample: float(
                    np.linalg.norm(sample["rotation"].T @ (sample["point"] - sample["position"]) - local_median)
                ),
            )

    rows: list[dict[str, Any]] = []
    heldout_metrics: dict[str, dict[str, float]] = {}
    diversity: dict[str, dict[str, float]] = {}
    for link_name in LINK_NAMES:
        samples = [selected[(env_index, link_name)] for env_index in sorted(pose_by_env)]
        translations = [sample["position"] for sample in samples]
        rotations = [sample["rotation"] for sample in samples]
        translation_span = _maximum_pairwise_distance(translations)
        rotation_span = _maximum_rotation_span(rotations)
        rotation_logs = np.stack([_rotation_log(rotations[0].T @ value) for value in rotations])
        centered_logs = rotation_logs - np.mean(rotation_logs, axis=0, keepdims=True)
        rotation_axis_singular_values = np.linalg.svd(centered_logs, compute_uv=False)
        if translation_span < thresholds.minimum_translation_span_m:
            raise PadSurfaceCalibrationError(f"raw_contact_translation_span_insufficient:{link_name}")
        if rotation_span < thresholds.minimum_rotation_span_rad:
            raise PadSurfaceCalibrationError(f"raw_contact_rotation_span_insufficient:{link_name}")
        if (
            rotation_axis_singular_values.shape[0] < 2
            or float(rotation_axis_singular_values[1])
            < thresholds.minimum_second_rotation_axis_singular_value_rad
        ):
            raise PadSurfaceCalibrationError(f"raw_contact_rotation_axis_rank_insufficient:{link_name}")
        diversity[link_name] = {
            "translation_span_m": translation_span,
            "rotation_span_rad": rotation_span,
            "rotation_log_covariance_rad2": np.cov(rotation_logs, rowvar=False).tolist(),
            "rotation_log_singular_values_rad": rotation_axis_singular_values.tolist(),
            "excited_rotation_axes": int(
                np.count_nonzero(
                    rotation_axis_singular_values
                    >= thresholds.minimum_second_rotation_axis_singular_value_rad
                )
            ),
        }
        train = [sample for index, sample in enumerate(samples) if index % 3 != 2]
        heldout = [sample for index, sample in enumerate(samples) if index % 3 == 2]
        offset = _fit_offset(train)
        residual = _residuals(heldout, offset)
        rms = float(np.sqrt(np.mean(residual**2)))
        maximum = float(np.max(residual))
        if rms > thresholds.heldout_rms_residual_m or maximum > thresholds.heldout_maximum_residual_m:
            raise PadSurfaceCalibrationError(f"raw_contact_heldout_fit_rejected:{link_name}:{rms:.9g}:{maximum:.9g}")
        heldout_metrics[link_name] = {"rms_residual_m": rms, "maximum_residual_m": maximum, "train_rows": len(train), "heldout_rows": len(heldout)}
        for env_index, sample in zip(sorted(pose_by_env), samples, strict=True):
            xyzw = sample.get("quaternion_xyzw")
            if xyzw is None:
                raise PadSurfaceCalibrationError(
                    f"selected_same_step_quaternion_missing:{env_index}:{link_name}"
                )
            rows.append(
                {
                    "pose_id": str(pose_by_env[env_index]["pose_id"]),
                    "link_name": link_name,
                    "link_position_world_m": sample["position"].tolist(),
                    "link_quaternion_world_xyzw": np.asarray(xyzw).tolist(),
                    "measured_pad_surface_point_world_m": sample["point"].tolist(),
                    "measurement_valid": True,
                    "filtered_pair_force_n": float(sample["force"]),
                    "raw_contact_count": int(sample["raw_count"]),
                    "selected_separation_m": float(sample["separation"]),
                    "unexpected_unfiltered_force_residual_n": float(
                        sample["unfiltered_force_residual_n"]
                    ),
                    "selected_other_actor_id": int(sample["other_actor_id"]),
                    "selected_other_actor_path": str(sample["other_actor_path"]),
                    "logical_pose_index": env_index,
                    "runtime_env_index": int(sample["runtime_env_index"]),
                    "sensor_prim_path": (
                        f"/World/envs/env_{int(sample['runtime_env_index'])}/"
                        f"Robot/{link_name}"
                    ),
                    "filter_partner_prim_path": (
                        f"/World/envs/env_{int(sample['runtime_env_index'])}/Object"
                    ),
                    "contact_point_source": evidence["contact_point_api"],
                    "physics_step": int(sample["physics_step"]),
                }
            )

    report = {
        "schema": RAW_VALIDATION_SCHEMA,
        "status": "PASS",
        "empirical_surface_reference": (
            "maximum-normal-force raw PhysX point on each opposing distal pad, "
            "registered by an analytic sphere diameter through the fixture center"
        ),
        "pose_count": len(pose_by_env),
        "capture_schedule": (
            "SEQUENTIAL_SINGLE_ENVIRONMENT"
            if sequential_single_environment
            else "VECTOR_ENV_PER_POSE"
        ),
        "runtime_environment_count": runtime_environment_count,
        "trial_count": len(trial_indices),
        "row_count": len(rows),
        "diversity": diversity,
        "heldout_fit": heldout_metrics,
        "bidirectional_fixture": bidirectional_metrics,
        "trial_lifecycle": [dict(event) for event in trial_events],
        "passive_mimic_drive_attestation": dict(
            evidence["passive_mimic_drive_attestation"]
        ),
        "resolved_sensor_filter_path_bijection_verified": True,
        "resolved_paths_queried_from": RESOLVED_PATH_QUERY_AUTHORITY,
        "selected_raw_partner_identity_verified": True,
        "all_occupied_unfiltered_entries_validated": True,
        "filtered_unfiltered_one_to_one_bijection_verified": True,
        "count_start_slices_nonoverlapping": True,
        "thresholds": thresholds.__dict__,
    }
    return rows, report


__all__ = [
    "RAW_EVIDENCE_SCHEMA",
    "RAW_VALIDATION_SCHEMA",
    "RawContactValidationThresholds",
    "validate_raw_contact_evidence",
]
