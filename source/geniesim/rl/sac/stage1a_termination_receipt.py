# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only receipts for Stage-1A task-level terminations.

The manager has already evaluated every termination term when this module is
called.  Receipts therefore read only cached term masks and cached camera
visibility evidence; they never invoke a predicate a second time.  This is
important for stateful terms such as the pre-contact visibility grace timer
and the persistent table-bounds timer.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
from geniesim.rl.isaaclab.g2_redundancy_teleop import tensor_value


TERMINATION_RECEIPT_SCHEMA = "g2_stage1a_task_termination_receipt_v1"


def _array(value: Any) -> np.ndarray:
    resolved = tensor_value(value)
    if isinstance(resolved, torch.Tensor) or hasattr(resolved, "detach"):
        return resolved.detach().to("cpu").numpy()
    return np.asarray(resolved)


def _env_scalar(value: Any, env_id: int) -> Any:
    array = _array(value).reshape(-1)
    if env_id < 0 or env_id >= array.size:
        raise IndexError(f"env index {env_id} outside value with {array.size} rows")
    scalar = array[env_id].item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, (float, np.floating)):
        numeric = float(scalar)
        return numeric if math.isfinite(numeric) else None
    return str(scalar)


def _cached_term_masks(env: Any, env_id: int) -> tuple[list[dict[str, Any]], list[str]]:
    manager = env.termination_manager
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in tuple(manager.active_terms):
        try:
            raw = _env_scalar(manager.get_term(name), env_id)
            rows.append(
                {
                    "predicate_name": str(name),
                    "predicate_boolean": bool(raw),
                    # Manager terms in this task expose their evaluated mask.
                    # The exact numeric antecedent is added below when the
                    # task keeps a public diagnostic buffer.
                    "raw_predicate_value": raw,
                    "value_authority": "TERMINATION_MANAGER_CACHED_TERM",
                }
            )
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{name}:{type(exc).__name__}:{exc}")
    return rows, errors


def _cached_visibility(env: Any, env_id: int) -> dict[str, Any]:
    result = getattr(env, "_g2_precontact_camera_visibility_last_result", None)
    camera_rows: dict[str, Any] = {}
    for visibility_name, scene_name in (
        ("head", "head_camera"),
        ("right_wrist", "right_wrist_camera"),
    ):
        sensor = env.scene[scene_name]
        timestamp_s: float | None = None
        age_ms: float | None = None
        timing_error: str | None = None
        try:
            captured, age = camera_capture_time_and_age(sensor)
            timestamp_s = float(_env_scalar(captured, env_id))
            age_ms = float(_env_scalar(age, env_id)) * 1000.0
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            timing_error = f"{type(exc).__name__}:{exc}"

        output = getattr(sensor.data, "output", {})
        rgb = output.get("rgb")
        depth = output.get("distance_to_image_plane")
        rgb_valid = False
        depth_valid = False
        depth_valid_ratio: float | None = None
        if rgb is not None:
            rgb_env = torch.as_tensor(rgb)[env_id]
            rgb_valid = bool(rgb_env.numel() and torch.isfinite(rgb_env).all().item())
        if depth is not None:
            depth_env = torch.as_tensor(depth)[env_id]
            valid = torch.isfinite(depth_env) & (depth_env > 0.0)
            depth_valid = bool(valid.any().item())
            depth_valid_ratio = float(valid.to(torch.float32).mean().item())

        cached = None if result is None else result.cameras.get(visibility_name)
        camera_rows[visibility_name] = {
            "scene_name": scene_name,
            "frame_id": _env_scalar(sensor.frame, env_id),
            "timestamp_s": timestamp_s,
            "age_ms": age_ms,
            "timing_error": timing_error,
            "rgb_valid": rgb_valid,
            "depth_valid": depth_valid,
            "depth_valid_ratio": depth_valid_ratio,
            "cube_visible_pixel_count": (
                None
                if cached is None or cached.rendered_cube_visible_pixel_count is None
                else _env_scalar(cached.rendered_cube_visible_pixel_count, env_id)
            ),
            "rendered_cube_visible": (
                None
                if cached is None or cached.rendered_cube_visible is None
                else _env_scalar(cached.rendered_cube_visible, env_id)
            ),
            "any_cube_part_in_frustum": (
                None
                if cached is None
                else _env_scalar(cached.any_cube_part_in_frustum, env_id)
            ),
            "rendered_depth_consistent": (
                None
                if cached is None or cached.rendered_depth_consistent is None
                else _env_scalar(cached.rendered_depth_consistent, env_id)
            ),
        }

    loss_age = getattr(env, "_g2_precontact_camera_loss_age_s", None)
    loss_now = getattr(env, "_g2_precontact_camera_visibility_last_loss_now", None)
    step_dt = float(env.step_dt)
    loss_age_s = None if loss_age is None else float(_env_scalar(loss_age, env_id))
    return {
        "cached_predicate_result_available": result is not None,
        "visibility_valid": bool(
            result is not None
            and all(
                row["rgb_valid"] and row["depth_valid"]
                for row in camera_rows.values()
            )
        ),
        "visible_camera_count": (
            None if result is None else _env_scalar(result.visible_camera_count, env_id)
        ),
        "dual_camera_cube_evidence": (
            None if result is None else _env_scalar(result.precontact_pass, env_id)
        ),
        "visibility_loss_now": (
            None if loss_now is None else _env_scalar(loss_now, env_id)
        ),
        "visibility_loss_age_s": loss_age_s,
        "visibility_loss_age_ms": (
            None if loss_age_s is None else loss_age_s * 1000.0
        ),
        "visibility_consecutive_lost_control_steps": (
            None
            if loss_age_s is None or step_dt <= 0.0
            else int(round(loss_age_s / step_dt))
        ),
        "cameras": camera_rows,
    }


def _env_from_serialized(value: Any, env_id: int) -> Any:
    """Select one env from evaluator diagnostic values serialized on CPU."""

    if isinstance(value, Mapping):
        return {str(key): _env_from_serialized(item, env_id) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > env_id and all(
            isinstance(item, (bool, int, float, list)) for item in value
        ):
            return value[env_id]
        return value
    return value


def _cached_collision_authority(env: Any, env_id: int) -> dict[str, Any]:
    """Read the collision evaluator's pre-auto-reset failure snapshot.

    This accessor never calls the collision predicate.  All values come from
    the evaluator invocation already performed by the termination manager.
    """

    evaluator = getattr(env, "_g2_forbidden_collision_evaluator", None)
    if evaluator is None:
        return {
            "available": False,
            "capture_error": "COLLISION_EVALUATOR_UNAVAILABLE",
        }
    try:
        peaks_all = evaluator.sensor_peak_forces_n()
        body_peaks_all = evaluator.sensor_body_peak_forces_n()
        body_vectors_all = evaluator.sensor_body_force_vectors_n()
        pad_all = evaluator.filtered_pad_force_components_n()
        link_all = evaluator.diagnostic_contact_force_components_n()
        contact_details_payload = evaluator.filtered_contact_details()
        contact_details_all = contact_details_payload.get("rows_by_sensor", {})
        peaks = {
            name: values[env_id] for name, values in peaks_all.items()
        }
        body_peaks = {
            sensor_name: {
                body_name: values[env_id]
                for body_name, values in bodies.items()
            }
            for sensor_name, bodies in body_peaks_all.items()
        }
        body_vectors = {
            sensor_name: {
                body_name: values[env_id]
                for body_name, values in bodies.items()
            }
            for sensor_name, bodies in body_vectors_all.items()
        }
        pad = {
            name: _env_from_serialized(value, env_id)
            for name, value in pad_all.items()
        }
        link = {
            name: _env_from_serialized(value, env_id)
            for name, value in link_all.items()
        }
        contact_details = {
            name: rows[env_id]
            for name, rows in contact_details_all.items()
            if len(rows) > env_id
        }
        threshold = float(getattr(evaluator, "force_threshold_n", 1.0e-6))
        triggered_sensors = sorted(
            name
            for name, value in peaks.items()
            if value is None or not math.isfinite(float(value)) or float(value) > threshold
        )
        dominant_sensor = max(
            peaks,
            key=lambda name: -math.inf
            if peaks[name] is None
            else float(peaks[name]),
            default=None,
        )
        dominant_body: str | None = None
        dominant_body_force_n = 0.0
        if dominant_sensor in body_peaks:
            for body_name, value in body_peaks[dominant_sensor].items():
                if float(value) > dominant_body_force_n:
                    dominant_body = body_name
                    dominant_body_force_n = float(value)
        return {
            "available": True,
            "value_authority": "CACHED_TERMINATION_FRAME_COLLISION_EVALUATOR",
            "force_threshold_n": threshold,
            "sensor_peak_forces_n": peaks,
            "triggered_sensors": triggered_sensors,
            "sensor_body_peak_forces_n": body_peaks,
            "sensor_body_force_vectors_n": body_vectors,
            "filtered_pad_force_components_n": pad,
            "filtered_outer_link2_force_components_n": link,
            "filtered_contact_details": contact_details,
            "filtered_contact_detail_capture_errors": contact_details_payload.get(
                "capture_errors_by_sensor", {}
            ),
            "dominant_sensor": dominant_sensor,
            "dominant_robot_link": dominant_body,
            "dominant_robot_link_force_n": dominant_body_force_n,
            "capture_error": None,
        }
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "capture_error": f"{type(exc).__name__}:{exc}",
        }


def capture_task_termination_receipt(
    *,
    env: Any,
    env_id: int,
    episode_id: int,
    source_sample_id: str,
    control_step: int,
    vector_step: int,
    runtime_terminated: bool,
    runtime_truncated: bool,
    reward_done: bool,
    gate_receipt: Mapping[str, Any],
    cube_position_world_m: Any | None,
    robot_root_position_world_m: Any | None,
    maximum_episode_control_steps: int,
    runtime_state_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the already-evaluated terminal packet before clone reset."""

    predicates, errors = _cached_term_masks(env, env_id)
    predicate_by_name = {
        str(row["predicate_name"]): bool(row["predicate_boolean"])
        for row in predicates
    }
    outside_age = getattr(env, "_g2_task_outside_table_time_s", None)
    cube_position = (
        None
        if cube_position_world_m is None
        else _array(cube_position_world_m).reshape(-1).astype(float).tolist()
    )
    robot_root_position = (
        None
        if robot_root_position_world_m is None
        else _array(robot_root_position_world_m).reshape(-1).astype(float).tolist()
    )
    contact_event_diagnostic = getattr(
        env, "_g2_stage1a_forbidden_contact_event_diagnostic", None
    )
    contact_event_window = None
    if contact_event_diagnostic is not None:
        try:
            contact_event_window = contact_event_diagnostic.terminal_window(
                env_id=env_id,
                episode_id=episode_id,
                source_sample_id=source_sample_id,
                termination_vector_step=vector_step,
            )
        except Exception as exc:
            contact_event_window = {
                "capture_error": f"{type(exc).__name__}:{exc}",
                "behavior_changed": False,
            }
    return {
        "schema": TERMINATION_RECEIPT_SCHEMA,
        "env_id": int(env_id),
        "episode_id": int(episode_id),
        "source_sample_id": str(source_sample_id),
        "control_step": int(control_step),
        "vector_step": int(vector_step),
        "runtime_terminated": bool(runtime_terminated),
        "runtime_truncated": bool(runtime_truncated),
        "reward_done": bool(reward_done),
        "termination_flag_sources": [
            name
            for name, active in (
                ("RUNTIME_TERMINATED", runtime_terminated),
                ("RUNTIME_TRUNCATED", runtime_truncated),
                ("STAGE1A_REWARD_DONE", reward_done),
            )
            if active
        ],
        "reset_scheduler_request": bool(
            runtime_terminated or runtime_truncated or reward_done
        ),
        "termination_predicates": predicates,
        "triggered_predicate_names": [
            name for name, triggered in predicate_by_name.items() if triggered
        ],
        "termination_predicate_capture_errors": errors,
        "precontact_visibility_triggered": bool(
            predicate_by_name.get("precontact_camera_visibility", False)
        ),
        "cube_bounds_triggered": bool(
            predicate_by_name.get("cube_left_table_persistently", False)
            or predicate_by_name.get("object_dropping", False)
        ),
        "fixed_torso_drift_triggered": bool(
            predicate_by_name.get("fixed_torso_drift", False)
        ),
        "safety_triggered": bool(
            predicate_by_name.get("forbidden_collision", False)
        ),
        "timeout_triggered": bool(predicate_by_name.get("time_out", False)),
        "runtime_state_receipt": (
            None if runtime_state_receipt is None else dict(runtime_state_receipt)
        ),
        "forbidden_collision_receipt": _cached_collision_authority(env, env_id),
        "forbidden_contact_event_window": contact_event_window,
        "visibility": _cached_visibility(env, env_id),
        "geometry": {
            "geometry_valid": gate_receipt.get("geometry_valid"),
            "owner_valid": gate_receipt.get("owner_valid"),
            "geometry_frame_id": gate_receipt.get("geometry_frame_id"),
            "geometry_timestamp_s": gate_receipt.get(
                "teacher_geometry_timestamp_s",
                gate_receipt.get("geometry_timestamp_s"),
            ),
            "geometry_age_ms": gate_receipt.get("geometry_age_ms"),
            "pad_gap_ready": gate_receipt.get("pad_gap_ready"),
            "aperture_ready": gate_receipt.get("aperture_ready"),
            "orientation_ready": gate_receipt.get("orientation_ready"),
        },
        "cube_position_world_m": cube_position,
        "robot_root_position_world_m": robot_root_position,
        "cube_outside_table_age_s": (
            None if outside_age is None else _env_scalar(outside_age, env_id)
        ),
        "maximum_episode_control_steps": int(maximum_episode_control_steps),
        "timeout_remaining_s": max(
            0.0,
            (int(maximum_episode_control_steps) - int(control_step))
            * float(env.step_dt),
        ),
        "behavior_changed": False,
    }


__all__ = [
    "TERMINATION_RECEIPT_SCHEMA",
    "capture_task_termination_receipt",
]
