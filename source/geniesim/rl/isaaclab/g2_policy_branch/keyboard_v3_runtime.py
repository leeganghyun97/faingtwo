# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Collection-side state machine for one canonical keyboard-v3 episode.

This module is the narrow adapter between an existing Isaac environment and
the v3 HDF writer.  It owns buffering/validation only: cuRobo, camera capture,
``env.step`` and controller consumption remain with the established runtime.
Consequently this code cannot create a second action-consumption path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from geniesim.rl.sac.keyboard_v3_dataset import (
    KEYBOARD_V3_CAMERA_NAMES,
    KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES,
    KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES,
    KEYBOARD_V3_OPERATOR_VIEW_NAMES,
    KeyboardV3DatasetError,
    KeyboardV3Episode,
    canonical_action_4d,
    close_edge_from_state,
    preprocess_depth_m,
    write_keyboard_v3_episode,
)
from geniesim.rl.sac.keyboard_grasp_contract import (
    LOCAL_GRASP_PHASE_CODE,
    canonical_keyboard_grasp_contract,
    classify_local_grasp_phase,
    nominal_grasp_translation_residual_m,
)

from .keyboard_v3_collection_contract import (
    COLLECTION_TARGET_REFERENCE,
    COLLECTION_TARGETS_M,
    NearGraspCondition,
    preclose_geometry_metrics,
)
from .omnipicker_product_contract import OMNIPICKER_PRODUCT_CONTRACT
from .rgbd_logging_contract import (
    DEPTH_RAW_UNIT,
    DEPTH_TO_METER_SCALE,
    RGBD_CAMERA_NAMES,
    flatten_calibration_metadata,
    raw_depth_and_validity,
)


KEYBOARD_V3_RUNTIME_SCHEMA = "g2_keyboard_v3_runtime_recorder_v1"
RGBD_TIMESTAMP_SOURCE = "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE"


class KeyboardV3RuntimeError(RuntimeError):
    pass


class CollectionPhase(str, Enum):
    RESET = "RESET"
    CUROBO_NEAR_GRASP = "CUROBO_NEAR_GRASP"
    RECORDING = "RECORDING"
    POST_CLOSE = "POST_CLOSE"
    FINALIZED = "FINALIZED"


@dataclass(frozen=True)
class PlannerStartReceipt:
    nominal_grasp_pose_root_m_xyzw: Sequence[float]
    near_grasp_pose_root_m_xyzw: Sequence[float]
    approach_axis_root: Sequence[float]
    condition: NearGraspCondition
    nominal_grasp_q_rad: Sequence[float] | None = None
    near_grasp_q_rad: Sequence[float] | None = None

    def validated(self) -> "PlannerStartReceipt":
        for name in ("nominal_grasp_pose_root_m_xyzw", "near_grasp_pose_root_m_xyzw"):
            pose = np.asarray(getattr(self, name), dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise KeyboardV3RuntimeError(f"{name} must be seven finite values")
            if abs(float(np.linalg.norm(pose[3:])) - 1.0) > 1.0e-3:
                raise KeyboardV3RuntimeError(f"{name} quaternion must be unit XYZW")
        for name in ("nominal_grasp_q_rad", "near_grasp_q_rad"):
            value = getattr(self, name)
            if value is not None:
                q = np.asarray(value, dtype=np.float64)
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise KeyboardV3RuntimeError(f"{name} must be seven finite radians")
        axis = np.asarray(self.approach_axis_root, dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(float(np.linalg.norm(axis)) - 1.0) > 1.0e-6:
            raise KeyboardV3RuntimeError("approach_axis_root must be a unit robot-root vector")
        nominal = np.asarray(self.nominal_grasp_pose_root_m_xyzw, dtype=np.float64)
        near = np.asarray(self.near_grasp_pose_root_m_xyzw, dtype=np.float64)
        expected = nominal[:3] - axis * self.condition.backoff_m + np.asarray(
            self.condition.start_offset_xyz_m, dtype=np.float64
        )
        if not np.allclose(near[:3], expected, rtol=0.0, atol=1.0e-6):
            raise KeyboardV3RuntimeError("near-grasp pose does not match backoff/offset")
        if not np.allclose(near[3:], nominal[3:], rtol=0.0, atol=1.0e-6):
            raise KeyboardV3RuntimeError("near-grasp orientation must remain fixed")
        return self


@dataclass(frozen=True)
class KeyboardV3Row:
    head_rgb: Any
    head_depth_source_m: Any
    head_depth_source_valid: Any
    head_rgb_timestamp_s: float
    head_depth_timestamp_s: float
    head_frame_id: int
    right_wrist_rgb: Any
    right_wrist_depth_source_m: Any
    right_wrist_depth_source_valid: Any
    right_wrist_rgb_timestamp_s: float
    right_wrist_depth_timestamp_s: float
    right_wrist_frame_id: int
    ee_position_root_m: Sequence[float]
    ee_quat_root_xyzw: Sequence[float]
    arm_q_rad: Sequence[float]
    arm_qd_rad_s: Sequence[float]
    gripper_state: float
    action_4d: Sequence[float]
    cube_center_root_m: Sequence[float]
    timestamp_s: float
    control_step: int
    privileged_optional: Mapping[str, Any] = field(default_factory=dict)
    head_camera_pose_root_m_xyzw: Any | None = None
    right_wrist_camera_pose_root_m_xyzw: Any | None = None


class KeyboardV3EpisodeRecorder:
    """Append rows after cuRobo handoff and publish one validated episode."""

    def __init__(
        self,
        *,
        episode_id: str,
        planner: PlannerStartReceipt,
        rgbd_hz: int,
        pad_surface_valid: bool = False,
        pad_calibration_verified: bool = False,
        baseline_metadata: Mapping[str, Any] | None = None,
        camera_calibration_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not episode_id:
            raise KeyboardV3RuntimeError("episode_id must be non-empty")
        contract = canonical_keyboard_grasp_contract()
        if rgbd_hz != contract.rgbd_acquisition_hz:
            raise KeyboardV3RuntimeError("keyboard-v3 RGB-D rate must be exactly 25 Hz")
        if pad_surface_valid and not pad_calibration_verified:
            raise KeyboardV3RuntimeError("pad surfaces require approved calibration")
        self.episode_id = episode_id
        self.planner = planner.validated()
        self.rgbd_hz = rgbd_hz
        self.pad_surface_valid = pad_surface_valid
        self.pad_calibration_verified = pad_calibration_verified
        self.baseline_metadata = dict(baseline_metadata or {})
        self.camera_calibration_metadata = (
            None
            if camera_calibration_metadata is None
            else flatten_calibration_metadata(camera_calibration_metadata)
        )
        required_baseline_keys = {
            "baseline_variant",
            "left_arm_posture_id",
            "pregrasp_init_source",
            "pregrasp_sample_id",
            "cube_sample_id",
            "head_camera",
            "wrist_camera",
            "control_source",
            "omnipicker_product_model",
            "omnipicker_manual_authority_url",
            "omnipicker_manual_hardware_version",
            "omnipicker_pcba_version",
            "omnipicker_firmware_version",
            "omnipicker_maximum_gripping_force_n",
            "omnipicker_physical_pad_touch_sensor_available",
            "omnipicker_protocol_force_feedback_semantics",
            "sim_contact_telemetry_role",
        }
        if self.baseline_metadata and set(self.baseline_metadata) != required_baseline_keys:
            raise KeyboardV3RuntimeError("keyboard-v3 baseline metadata keys mismatch")
        self.phase = CollectionPhase.CUROBO_NEAR_GRASP
        self._rows: list[KeyboardV3Row] = []
        self._previous_action = np.zeros(4, dtype=np.float32)
        self._close_seen = False
        self._last_camera_frame: dict[str, int | None] = {
            camera_name: None for camera_name in KEYBOARD_V3_CAMERA_NAMES
        }
        self._last_camera_timestamp_s: dict[str, float | None] = {
            camera_name: None for camera_name in KEYBOARD_V3_CAMERA_NAMES
        }

    def start_recording(self) -> None:
        if self.phase is not CollectionPhase.CUROBO_NEAR_GRASP:
            raise KeyboardV3RuntimeError("recording can start only after near-grasp handoff")
        self.phase = CollectionPhase.RECORDING

    def append(self, row: KeyboardV3Row) -> None:
        if self.phase not in (CollectionPhase.RECORDING, CollectionPhase.POST_CLOSE):
            raise KeyboardV3RuntimeError("row appended outside recording phase")
        action = canonical_action_4d(row.action_4d)
        if row.control_step != len(self._rows):
            raise KeyboardV3RuntimeError("control_step must be contiguous from zero")
        contract = canonical_keyboard_grasp_contract()
        expected_time = row.control_step * contract.policy_dt_s
        if abs(float(row.timestamp_s) - expected_time) > 1.0e-6:
            raise KeyboardV3RuntimeError("timestamp must share the 50-Hz episode clock")
        for camera_name in KEYBOARD_V3_CAMERA_NAMES:
            rgb_timestamp = float(getattr(row, f"{camera_name}_rgb_timestamp_s"))
            depth_timestamp = float(
                getattr(row, f"{camera_name}_depth_timestamp_s")
            )
            if not (np.isfinite(rgb_timestamp) and np.isfinite(depth_timestamp)):
                raise KeyboardV3RuntimeError(
                    f"{camera_name} RGB-D acquisition timestamp must be finite"
                )
            if rgb_timestamp != depth_timestamp:
                raise KeyboardV3RuntimeError(
                    f"{camera_name} RGB/depth acquisition timestamps differ"
                )
            frame = int(getattr(row, f"{camera_name}_frame_id"))
            prior_frame = self._last_camera_frame[camera_name]
            prior_timestamp = self._last_camera_timestamp_s[camera_name]
            if prior_frame is not None and prior_timestamp is not None:
                if frame == prior_frame and rgb_timestamp != prior_timestamp:
                    raise KeyboardV3RuntimeError(
                        f"{camera_name} reused frame changed acquisition timestamp"
                    )
                if frame != prior_frame and rgb_timestamp <= prior_timestamp:
                    raise KeyboardV3RuntimeError(
                        f"{camera_name} new frame did not advance acquisition timestamp"
                    )
            self._last_camera_frame[camera_name] = frame
            self._last_camera_timestamp_s[camera_name] = rgb_timestamp
            pose = getattr(row, f"{camera_name}_camera_pose_root_m_xyzw")
            if self.camera_calibration_metadata is not None:
                pose_array = np.asarray(pose, dtype=np.float64)
                if (
                    pose_array.shape != (7,)
                    or not np.isfinite(pose_array).all()
                    or abs(float(np.linalg.norm(pose_array[3:])) - 1.0) > 1.0e-3
                ):
                    raise KeyboardV3RuntimeError(
                        f"{camera_name} camera pose must be finite robot-root XYZW"
                    )
        if action[3] == 0.0 and self._close_seen:
            # Explicit re-open is legal in the command contract, but it ends
            # the one-close demonstration context; publish before a new trial.
            raise KeyboardV3RuntimeError("explicit re-open requires a new episode")
        if action[3] == 1.0 and not self._close_seen:
            self._close_seen = True
            self.phase = CollectionPhase.POST_CLOSE
        self._rows.append(row)
        self._previous_action = action

    def build_episode(self, *, success: bool, failure_type: str) -> KeyboardV3Episode:
        """Freeze buffered rows into one v3 episode without publishing it."""

        contract = canonical_keyboard_grasp_contract()
        if self.phase not in (CollectionPhase.RECORDING, CollectionPhase.POST_CLOSE):
            raise KeyboardV3RuntimeError("episode cannot finalize from current phase")
        if not self._rows:
            raise KeyboardV3RuntimeError("empty keyboard-v3 episode")
        rows = self._rows
        action = np.stack([canonical_action_4d(row.action_4d) for row in rows])
        previous = np.zeros_like(action)
        previous[1:] = action[:-1]
        close_state = action[:, 3].astype(bool)
        close_indices = np.flatnonzero(close_edge_from_state(close_state))
        first_close = int(close_indices[0]) if close_indices.size else None
        future_horizon_steps = 64
        post_close_observed_steps = (
            len(rows) - first_close - 1 if first_close is not None else 0
        )
        future_outcome_right_censored = bool(
            first_close is not None
            and post_close_observed_steps < future_horizon_steps
        )
        observations = {
            "ee_position_root_m": np.asarray([row.ee_position_root_m for row in rows], dtype=np.float32),
            "ee_quat_root_xyzw": np.asarray([row.ee_quat_root_xyzw for row in rows], dtype=np.float32),
            "arm_q_rad": np.asarray([row.arm_q_rad for row in rows], dtype=np.float32),
            "arm_qd_rad_s": np.asarray([row.arm_qd_rad_s for row in rows], dtype=np.float32),
            "gripper_state": np.asarray([[row.gripper_state] for row in rows], dtype=np.float32),
        }
        for camera_name in KEYBOARD_V3_CAMERA_NAMES:
            raw_rows: list[np.ndarray] = []
            source_valid_rows: list[np.ndarray] = []
            for row in rows:
                raw, source_valid = raw_depth_and_validity(
                    getattr(row, f"{camera_name}_depth_source_m"),
                    getattr(row, f"{camera_name}_depth_source_valid"),
                )
                raw_rows.append(raw)
                source_valid_rows.append(source_valid)
            depth_rows, valid_rows = zip(
                *(
                    preprocess_depth_m(
                        getattr(row, f"{camera_name}_depth_source_m"),
                        getattr(row, f"{camera_name}_depth_source_valid"),
                    )
                    for row in rows
                ),
                strict=True,
            )
            observations[f"{camera_name}_rgb"] = np.stack(
                [
                    np.asarray(getattr(row, f"{camera_name}_rgb"), dtype=np.uint8)
                    for row in rows
                ]
            )
            observations[f"{camera_name}_depth_m"] = np.stack(depth_rows)
            observations[f"{camera_name}_depth_valid"] = np.stack(valid_rows)
            observations[f"{camera_name}_depth_raw_m"] = np.stack(raw_rows)
            observations[f"{camera_name}_depth_source_valid"] = np.stack(
                source_valid_rows
            )
            observations[f"{camera_name}_rgb_timestamp_s"] = np.asarray(
                [getattr(row, f"{camera_name}_rgb_timestamp_s") for row in rows],
                dtype=np.float64,
            )
            observations[f"{camera_name}_depth_timestamp_s"] = np.asarray(
                [getattr(row, f"{camera_name}_depth_timestamp_s") for row in rows],
                dtype=np.float64,
            )
            observations[f"{camera_name}_frame_id"] = np.asarray(
                [getattr(row, f"{camera_name}_frame_id") for row in rows],
                dtype=np.int64,
            )
            observations[f"associated_{camera_name}_frame_index"] = observations[
                f"{camera_name}_frame_id"
            ].copy()
            if self.camera_calibration_metadata is not None:
                observations[f"{camera_name}_camera_pose_root_m_xyzw"] = np.asarray(
                    [
                        getattr(row, f"{camera_name}_camera_pose_root_m_xyzw")
                        for row in rows
                    ],
                    dtype=np.float32,
                )
        privileged: dict[str, Any] = {
            "cube_center_root_m": np.asarray([row.cube_center_root_m for row in rows], dtype=np.float32)
        }
        optional_names = set().union(*(row.privileged_optional.keys() for row in rows))
        for name in optional_names:
            if not all(name in row.privileged_optional for row in rows):
                raise KeyboardV3RuntimeError(f"optional privileged field is sparse: {name}")
            privileged[name] = np.asarray([row.privileged_optional[name] for row in rows])
        planner = {
            "nominal_grasp_pose_root_m_xyzw": np.asarray(self.planner.nominal_grasp_pose_root_m_xyzw, dtype=np.float32),
            "near_grasp_pose_root_m_xyzw": np.asarray(self.planner.near_grasp_pose_root_m_xyzw, dtype=np.float32),
            "approach_axis_root": np.asarray(self.planner.approach_axis_root, dtype=np.float32),
        }
        metric_rows: list[dict[str, float]] = []
        for index, row in enumerate(rows):
            metric_rows.append(
                preclose_geometry_metrics(
                    ee_position_root_m=row.ee_position_root_m,
                    previous_ee_position_root_m=(
                        None if index == 0 else rows[index - 1].ee_position_root_m
                    ),
                    cube_center_root_m=row.cube_center_root_m,
                    nominal_grasp_pose_root_m_xyzw=(
                        self.planner.nominal_grasp_pose_root_m_xyzw
                    ),
                    approach_axis_root=self.planner.approach_axis_root,
                    action_xyz_root_m=action[index, :3],
                    control_dt_s=contract.policy_dt_s,
                )
            )
        nominal_residual_m = np.asarray(
            [item["current_nominal_grasp_residual_m"] for item in metric_rows],
            dtype=np.float32,
        )
        phase_names = [
            classify_local_grasp_phase(
                float(residual), gripper_closed=bool(closed)
            )
            for residual, closed in zip(
                nominal_residual_m, close_state, strict=True
            )
        ]
        planner["current_nominal_grasp_residual_m"] = nominal_residual_m
        planner["target_residual_m"] = np.full(
            len(rows), self.planner.condition.backoff_m, dtype=np.float32
        )
        planner["lateral_error_m"] = np.asarray(
            [item["lateral_error_m"] for item in metric_rows], dtype=np.float32
        )
        planner["approach_cosine"] = np.asarray(
            [item["approach_cosine"] for item in metric_rows], dtype=np.float32
        )
        planner["ee_speed_m_s"] = np.asarray(
            [item["ee_speed_m_s"] for item in metric_rows], dtype=np.float32
        )
        phase_code = np.asarray(
            [LOCAL_GRASP_PHASE_CODE[name] for name in phase_names], dtype=np.int8
        )
        planner["local_grasp_phase_code"] = phase_code
        planner["phase_code"] = phase_code.copy()
        if self.planner.nominal_grasp_q_rad is not None:
            planner["nominal_grasp_q_rad"] = np.asarray(self.planner.nominal_grasp_q_rad, dtype=np.float32)
        if self.planner.near_grasp_q_rad is not None:
            planner["near_grasp_q_rad"] = np.asarray(self.planner.near_grasp_q_rad, dtype=np.float32)
        return KeyboardV3Episode(
            episode_id=self.episode_id,
            metadata={
                "schema_version": "keyboard_v3",
                "action_dim": 4,
                "action_frame": "robot_root",
                "action_unit": "m",
                "joint_position_unit": "rad",
                "joint_velocity_unit": "rad/s",
                "depth_unit": "m",
                "depth_raw_unit": DEPTH_RAW_UNIT,
                "depth_to_meter_scale": DEPTH_TO_METER_SCALE,
                "raw_depth_storage_semantics": (
                    "UNCLAMPED_METRIC_WITH_POSITIVE_INFINITY_PRESERVED_AND_SEPARATE_VALID_MASK"
                ),
                "control_hz": contract.control_hz,
                "control_dt_s": contract.policy_dt_s,
                "dataset_row_hz": contract.dataset_row_hz,
                "physics_hz": contract.physics_hz,
                "physics_dt_s": contract.physics_dt_s,
                "ee_frame": "gripper_r_center_link",
                "orientation_policy": "FIXED",
                "elbow_policy": "NON_POLICY",
                "student_privileged_input_count": 0,
                "gru_actor_camera_names": list(KEYBOARD_V3_GRASP_ACTOR_CAMERA_NAMES),
                "recorded_non_actor_camera_names": list(KEYBOARD_V3_NON_ACTOR_CAMERA_NAMES),
                "operator_display_only_view_names": list(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
                "outcome_alignment": "POST_ACTION_TRANSITION_T_PLUS_1",
                "outcome_label_authority": "RAW_TELEMETRY_OFFLINE_DERIVATION_ONLY",
                "human_close_label_authority": "KEYBOARD_K_CLOSE_EDGE",
                "privileged_feasible_label_authority": "VERIFIED_SIM_GEOMETRY_ONLY",
                "human_and_privileged_labels_merged": False,
                "primary_privileged_learning_signal": "BOOLEAN_CONTACT",
                "force_learning_role": "RAW_DIAGNOSTIC_SAFETY_AUXILIARY_ONLY",
                "contact_channel_names": ["right_inner", "right_outer"],
                "contact_channel_source": "ISAAC_SIM_CONTACT_SENSOR_PRIVILEGED_ONLY",
                "contact_boolean_extraction": "FILTERED_FORCE_MATRIX_W_GREATER_THAN_1N_TO_BOOLEAN",
                "contact_boolean_threshold_n": 1.0,
                "hardware_pad_touch_sensor_available": False,
                "oem_maximum_gripping_force_n": (
                    OMNIPICKER_PRODUCT_CONTRACT.maximum_gripping_force_n
                ),
                "oem_gripping_force_metric": (
                    OMNIPICKER_PRODUCT_CONTRACT.m2_gripping_force_metric
                ),
                "sim_contact_to_oem_force_mapping": (
                    OMNIPICKER_PRODUCT_CONTRACT.sim_contact_to_oem_gripping_force_mapping
                ),
                "protocol_force_feedback_semantics": (
                    OMNIPICKER_PRODUCT_CONTRACT.protocol_force_feedback_semantics
                ),
                "student_contact_input_count": 0,
                "pad_pose_authority": "DISTAL_LINK_FRAME_RAW_NOT_CALIBRATED_PAD_SURFACE",
                "future_outcome_horizon_steps": future_horizon_steps,
                "post_close_observed_steps": post_close_observed_steps,
                "future_outcome_right_censored": future_outcome_right_censored,
                "success": bool(success),
                "failure_type": failure_type,
                "curobo_backoff_m": self.planner.condition.backoff_m,
                "collection_target_residual_m": self.planner.condition.backoff_m,
                "collection_target_residual_mm": int(
                    round(self.planner.condition.backoff_m * 1000.0)
                ),
                "collection_target_reference": COLLECTION_TARGET_REFERENCE,
                "collection_target_authority_active": bool(
                    self.planner.condition.backoff_m in COLLECTION_TARGETS_M
                ),
                "handoff_distance_m": self.planner.condition.backoff_m,
                "handoff_band_m": list(contract.curobo_handoff_band_m),
                "default_handoff_m": contract.curobo_default_handoff_m,
                "local_grasp_band_m": list(contract.local_grasp_band_m),
                "local_grasp_distance_reference": (
                    contract.local_grasp_distance_reference
                ),
                "local_grasp_distance_is_student_input": False,
                "local_grasp_phase_names": list(LOCAL_GRASP_PHASE_CODE),
                "start_offset_xyz_m": self.planner.condition.start_offset_xyz_m,
                "condition_id": self.planner.condition.condition_id,
                "pad_surface_valid": self.pad_surface_valid,
                "pad_calibration_verified": self.pad_calibration_verified,
                "rgbd_hz": self.rgbd_hz,
                "rgbd_dt_s": contract.rgbd_dt_s,
                "rgbd_timestamp_source": RGBD_TIMESTAMP_SOURCE,
                "rgbd_observation_alignment": (
                    "LATEST_VALID_ACQUISITION_REFERENCED_BY_FRAME_ID"
                ),
                "recorded_camera_count": len(KEYBOARD_V3_CAMERA_NAMES),
                "recorded_camera_names": list(KEYBOARD_V3_CAMERA_NAMES),
                "operator_view_count": len(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
                "operator_view_names": list(KEYBOARD_V3_OPERATOR_VIEW_NAMES),
                "camera_capture_contract": (
                    "ACTUAL_ACQUISITION_TIMESTAMP_HELD_ACROSS_TWO_CONTROL_ROWS"
                ),
                "camera_evidence_extension_active": bool(
                    self.camera_calibration_metadata is not None
                ),
                **(self.camera_calibration_metadata or {}),
                **self.baseline_metadata,
            },
            observations=observations,
            actions={"previous_action_4d": previous, "action_4d": action},
            events={"close_edge": close_edge_from_state(close_state), "close_state": close_state},
            privileged=privileged,
            planner=planner,
            time={
                "timestamp_s": np.asarray([row.timestamp_s for row in rows], dtype=np.float64),
                "control_step": np.asarray([row.control_step for row in rows], dtype=np.int64),
            },
        )
    def finalize(
        self,
        destination: str | Path,
        *,
        success: bool,
        failure_type: str,
    ) -> dict[str, Any]:
        episode = self.build_episode(success=success, failure_type=failure_type)
        validation = write_keyboard_v3_episode(destination, episode)
        self.phase = CollectionPhase.FINALIZED
        return validation.payload()

    def mark_discarded(self) -> None:
        if self.phase not in (CollectionPhase.RECORDING, CollectionPhase.POST_CLOSE):
            raise KeyboardV3RuntimeError("episode cannot be discarded from current phase")
        self.phase = CollectionPhase.FINALIZED


__all__ = [
    "CollectionPhase",
    "KeyboardV3EpisodeRecorder",
    "KeyboardV3Row",
    "KeyboardV3RuntimeError",
    "PlannerStartReceipt",
]
