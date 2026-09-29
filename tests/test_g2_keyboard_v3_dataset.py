# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import h5py

from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_collection_contract import (
    COLLECTION_TARGETS_MM,
    LOCAL_GRASP_PHASE_CODE,
    PersistentGripperCommand,
    classify_local_grasp_phase,
    collection_target_conditions,
    format_terminal_status,
    keyboard_action,
    near_grasp_pose_root_m_xyzw,
    nominal_grasp_translation_residual_m,
    pilot_conditions,
    preclose_geometry_metrics,
    select_next_target_mm,
    target_distribution_counts,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_runtime import (
    KeyboardV3EpisodeRecorder,
    KeyboardV3Row,
    PlannerStartReceipt,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_action_adapter import (
    KeyboardV3ActionAdapterError,
    adapt_legacy_keyboard_physical_8d,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
    nominal_grasp_pose_root_m_xyzw_for_cube,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_operator_ui import (
    KeyboardV3OperatorUI,
    OperatorUIError,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_operator_views import (
    KEYBOARD_V3_OPERATOR_VIEW_NAMES,
    PERSPECTIVE_OPERATOR_EYE_M,
)
from geniesim.rl.sac.keyboard_v3_dataset import (
    KeyboardV3DatasetError,
    KeyboardV3Episode,
    OfflineGraspLabelThresholds,
    canonical_action_4d,
    close_edge_from_state,
    derive_offline_grasp_labels,
    episode_to_gru,
    preprocess_depth_m,
    read_keyboard_v3_episode,
    training_readiness,
    validate_episode,
    write_keyboard_v3_episode,
)
from geniesim.rl.sac.human_grasp_gru_bc import (
    ClosePositiveWeightReceipt,
    EpisodeSplit,
    HUMAN_GRASP_GRU_CHECKPOINT_SCHEMA,
    HysteresisCalibration,
    HumanGraspGRUBC,
    HumanGraspGRUConfig,
    tensor_state_sha256,
)
from geniesim.rl.sac.keyboard_v3_offline_replay import (
    KeyboardV3OfflineReplay,
    KeyboardV3OfflineReplayError,
)


def _episode(episode_id: str = "episode_000001", *, close_index: int = 6) -> KeyboardV3Episode:
    rows, height, width = 12, 8, 10
    action = np.zeros((rows, 4), dtype=np.float32)
    action[:close_index, 0] = 0.001
    action[close_index:, 3] = 1.0
    previous = np.zeros_like(action)
    previous[1:] = action[:-1]
    close_state = action[:, 3].astype(bool)
    depth = np.ones((rows, height, width, 1), dtype=np.float32) * 0.4
    valid = np.ones_like(depth, dtype=bool)
    timestamps = np.arange(rows, dtype=np.float64) * 0.02
    sensor_indices = np.arange(rows, dtype=np.int64) // 2
    sensor_timestamps = sensor_indices.astype(np.float64) * 0.04
    camera_observations = {}
    for camera_index, camera_name in enumerate(
        ("head", "right_wrist")
    ):
        camera_observations.update(
            {
                f"{camera_name}_rgb": np.full(
                    (rows, height, width, 3), camera_index, dtype=np.uint8
                ),
                f"{camera_name}_depth_m": depth.copy(),
                f"{camera_name}_depth_valid": valid.copy(),
                f"{camera_name}_rgb_timestamp_s": sensor_timestamps.copy(),
                f"{camera_name}_depth_timestamp_s": sensor_timestamps.copy(),
                f"{camera_name}_frame_id": sensor_indices.copy(),
            }
        )
    return KeyboardV3Episode(
        episode_id=episode_id,
        metadata={
            "schema_version": "keyboard_v3",
            "action_dim": 4,
            "action_frame": "robot_root",
            "action_unit": "m",
            "joint_position_unit": "rad",
            "joint_velocity_unit": "rad/s",
            "depth_unit": "m",
            "control_hz": 50,
            "control_dt_s": 0.02,
            "dataset_row_hz": 50,
            "physics_hz": 500,
            "physics_dt_s": 0.002,
            "ee_frame": "gripper_r_center_link",
            "orientation_policy": "FIXED",
            "elbow_policy": "NON_POLICY",
            "student_privileged_input_count": 0,
            "success": False,
            "failure_type": "FAIL_LATE_CLOSE",
            "curobo_backoff_m": 0.030,
            "start_offset_xyz_m": [0.0, 0.003, 0.0],
            "pad_surface_valid": False,
            "pad_calibration_verified": False,
            "rgbd_hz": 25,
            "rgbd_dt_s": 0.04,
            "rgbd_timestamp_source": "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE",
            "rgbd_observation_alignment": "LATEST_VALID_ACQUISITION_REFERENCED_BY_FRAME_ID",
            "recorded_camera_count": 2,
            "recorded_camera_names": ["head", "right_wrist"],
            "operator_view_count": 3,
            "operator_view_names": ["perspective", "head", "right_wrist"],
            "camera_capture_contract": "ACTUAL_ACQUISITION_TIMESTAMP_HELD_ACROSS_TWO_CONTROL_ROWS",
            "gru_actor_camera_names": ["right_wrist"],
            "recorded_non_actor_camera_names": ["head"],
            "operator_display_only_view_names": ["perspective", "head"],
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
            "oem_maximum_gripping_force_n": 30.0,
            "oem_gripping_force_metric": "whole_gripper_gripping_force_estimate_n",
            "sim_contact_to_oem_force_mapping": "UNRESOLVED",
            "protocol_force_feedback_semantics": "CURRENT_MOTOR_TORQUE_NORMALIZED_0_TO_FF",
            "student_contact_input_count": 0,
            "pad_pose_authority": "DISTAL_LINK_FRAME_RAW_NOT_CALIBRATED_PAD_SURFACE",
            "future_outcome_horizon_steps": 64,
            "post_close_observed_steps": rows - close_index - 1,
            "future_outcome_right_censored": True,
        },
        observations={
            **camera_observations,
            "ee_position_root_m": np.zeros((rows, 3), dtype=np.float32),
            "ee_quat_root_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (rows, 1)).astype(np.float32),
            "arm_q_rad": np.zeros((rows, 7), dtype=np.float32),
            "arm_qd_rad_s": np.zeros((rows, 7), dtype=np.float32),
            "gripper_state": close_state.astype(np.float32).reshape(rows, 1),
        },
        actions={"previous_action_4d": previous, "action_4d": action},
        events={
            "close_edge": close_edge_from_state(close_state),
            "close_state": close_state,
        },
        privileged={
            "cube_center_root_m": np.tile([0.4, -0.1, 0.8], (rows, 1)).astype(np.float32),
            "contact_force_left_n": np.zeros(rows, dtype=np.float32),
            "contact_force_right_n": np.zeros(rows, dtype=np.float32),
            "total_normal_force_n": np.zeros(rows, dtype=np.float32),
            "contact_force_valid": np.ones((rows, 2), dtype=bool),
            "contact_measurement_valid": np.ones(rows, dtype=bool),
            "contact_timestamp_s": timestamps + 0.02,
            "contact_control_step": np.arange(1, rows + 1, dtype=np.int64),
            "left_contact": np.zeros(rows, dtype=bool),
            "right_contact": np.zeros(rows, dtype=bool),
            "contact": np.zeros(rows, dtype=bool),
            "exact_inner_contact": np.zeros(rows, dtype=bool),
            "exact_outer_contact": np.zeros(rows, dtype=bool),
            "bilateral_contact": np.zeros(rows, dtype=bool),
            "stable_grasp": np.zeros(rows, dtype=bool),
            "physical_lift": np.zeros(rows, dtype=bool),
            "slip_speed_m_s": np.zeros(rows, dtype=np.float32),
            "forbidden_collision": np.zeros(rows, dtype=bool),
            "forbidden_collision_valid": np.ones(rows, dtype=bool),
            "safety_violation": np.zeros(rows, dtype=bool),
            "safety_measurement_valid": np.ones(rows, dtype=bool),
            "cube_linear_velocity_root_m_s": np.zeros((rows, 3), dtype=np.float32),
            "cube_pose_root_m_xyzw": np.tile([0.4, -0.1, 0.8, 0, 0, 0, 1], (rows, 1)).astype(np.float32),
            "pad_center_relative_to_cube_velocity_root_m_s": np.zeros((rows, 3), dtype=np.float32),
            "gripper_command": action[:, 3].copy(),
            "gripper_master_position_rad": np.zeros(rows, dtype=np.float32),
            "gripper_master_velocity_rad_s": np.zeros(rows, dtype=np.float32),
            "right_inner_distal_link_pose_root_m_xyzw": np.tile([0.4, -0.11, 0.8, 0, 0, 0, 1], (rows, 1)).astype(np.float32),
            "right_outer_distal_link_pose_root_m_xyzw": np.tile([0.4, -0.09, 0.8, 0, 0, 0, 1], (rows, 1)).astype(np.float32),
        },
        planner={
            "nominal_grasp_pose_root_m_xyzw": np.asarray([0.4, -0.1, 0.8, 0, 0, 0, 1], dtype=np.float32),
            "near_grasp_pose_root_m_xyzw": np.asarray([0.370, -0.097, 0.8, 0, 0, 0, 1], dtype=np.float32),
            "approach_axis_root": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "nominal_grasp_q_rad": np.zeros(7, dtype=np.float32),
            "near_grasp_q_rad": np.zeros(7, dtype=np.float32),
        },
        time={"timestamp_s": timestamps, "control_step": np.arange(rows, dtype=np.int64)},
    )


def _recorded_camera_row_fields(step: int, *, rgbd_hz: int) -> dict[str, object]:
    assert rgbd_hz == 25
    sensor_step = step // 2
    sensor_time_s = sensor_step / float(rgbd_hz)
    values: dict[str, object] = {}
    for camera_index, camera_name in enumerate(
        ("head", "right_wrist")
    ):
        values.update(
            {
                f"{camera_name}_rgb": np.full(
                    (8, 10, 3), camera_index, dtype=np.uint8
                ),
                f"{camera_name}_depth_source_m": np.ones(
                    (8, 10, 1), dtype=np.float32
                )
                * 0.4,
                f"{camera_name}_depth_source_valid": np.ones(
                    (8, 10, 1), dtype=bool
                ),
                f"{camera_name}_rgb_timestamp_s": sensor_time_s,
                f"{camera_name}_depth_timestamp_s": sensor_time_s,
                f"{camera_name}_frame_id": sensor_step,
            }
        )
    return values


def _outcome_fields(step: int, *, contact_after: int | None = None) -> dict[str, object]:
    contact = contact_after is not None and step >= contact_after
    pose_inner = [0.4, -0.01, 0.8, 0.0, 0.0, 0.0, 1.0]
    pose_outer = [0.4, 0.01, 0.8, 0.0, 0.0, 0.0, 1.0]
    return {
        "contact_force_left_n": 1.0 if contact else 0.0,
        "contact_force_right_n": 1.0 if contact else 0.0,
        "total_normal_force_n": 2.0 if contact else 0.0,
        "contact_force_valid": [True, True],
        "contact_measurement_valid": True,
        "contact_timestamp_s": (step + 1) * 0.02,
        "contact_control_step": step + 1,
        "left_contact": contact,
        "right_contact": contact,
        "contact": contact,
        "exact_inner_contact": contact,
        "exact_outer_contact": contact,
        "bilateral_contact": contact,
        "stable_grasp": contact,
        "physical_lift": False,
        "slip_speed_m_s": 0.0,
        "forbidden_collision": False,
        "forbidden_collision_valid": True,
        "safety_violation": False,
        "safety_measurement_valid": True,
        "cube_linear_velocity_root_m_s": [0.0, 0.0, 0.0],
        "cube_pose_root_m_xyzw": [0.4, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        "pad_center_relative_to_cube_velocity_root_m_s": [0.0, 0.0, 0.0],
        "gripper_command": float(step >= 3),
        "gripper_master_position_rad": 0.0,
        "gripper_master_velocity_rad_s": 0.0,
        "right_inner_distal_link_pose_root_m_xyzw": pose_inner,
        "right_outer_distal_link_pose_root_m_xyzw": pose_outer,
    }


def test_depth_preprocess_invalid_is_finite_zero_without_clipping():
    raw = np.asarray([[[[0.4], [np.nan], [2.1], [-0.1]]]], dtype=np.float32)
    depth, valid = preprocess_depth_m(raw, np.ones_like(raw, dtype=bool))
    assert valid.reshape(-1).tolist() == [True, False, False, False]
    assert depth.reshape(-1).tolist() == pytest.approx([0.4, 0.0, 0.0, 0.0])


def test_action_rejects_over_bound_and_legacy_width():
    assert canonical_action_4d([0.0045, 0, 0, 0]).shape == (4,)
    with pytest.raises(KeyboardV3DatasetError):
        canonical_action_4d([0.0046, 0, 0, 0])
    with pytest.raises(KeyboardV3DatasetError):
        canonical_action_4d([0, 0, 0, 0, 0, 0, 0, 1])


def test_legacy_8d_requires_explicit_typed_adapter_without_slicing_or_clipping():
    receipt = adapt_legacy_keyboard_physical_8d(
        [0.001, -0.002, 0.003, 0.0, 0.0, 0.0, 0.0, -1.0]
    )
    assert receipt.action_4d.tolist() == pytest.approx([0.001, -0.002, 0.003, 1.0])
    assert receipt.hidden_component_count == 0
    assert receipt.silent_clipping_count == 0
    with pytest.raises(KeyboardV3ActionAdapterError, match="rotation/elbow"):
        adapt_legacy_keyboard_physical_8d(
            [0.001, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]
        )
    with pytest.raises(KeyboardV3ActionAdapterError, match="gripper sign"):
        adapt_legacy_keyboard_physical_8d(
            [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
        )
    with pytest.raises(KeyboardV3DatasetError, match="exceeds"):
        adapt_legacy_keyboard_physical_8d(
            [0.0046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )


def test_time_contract_rejects_50hz_rgbd_and_control_derived_sensor_clock():
    episode = _episode()
    metadata = dict(episode.metadata)
    metadata["rgbd_hz"] = 50
    assert "RGBD_HZ_MUST_BE_25" in validate_episode(
        replace(episode, metadata=metadata)
    ).errors

    observations = dict(episode.observations)
    synthetic = np.arange(12, dtype=np.float64) * 0.02
    for camera_name in ("head", "right_wrist"):
        observations[f"{camera_name}_rgb_timestamp_s"] = synthetic.copy()
        observations[f"{camera_name}_depth_timestamp_s"] = synthetic.copy()
    errors = validate_episode(replace(episode, observations=observations)).errors
    assert any("NOT_25HZ_ACQUISITION_CLOCK" in error for error in errors)


def test_time_contract_accepts_float32_sensor_clock_without_accepting_60ms_gap():
    episode = _episode()
    observations = dict(episode.observations)
    frame_id = np.asarray(observations["head_frame_id"], dtype=np.int64)
    # Reproduce the frozen Isaac camera's measured float32 increment.  Its
    # accumulated distance from an ideal float64 origin exceeds 1 us even
    # though every acquisition remains a valid 25-Hz event.
    acquisition = (
        np.float32(0.02)
        + np.arange(int(frame_id.max()) + 1, dtype=np.float32)
        * np.float32(0.03999813)
    ).astype(np.float64)
    sensor_time = acquisition[frame_id]
    for camera_name in ("head", "right_wrist"):
        observations[f"{camera_name}_rgb_timestamp_s"] = sensor_time.copy()
        observations[f"{camera_name}_depth_timestamp_s"] = sensor_time.copy()
    assert validate_episode(replace(episode, observations=observations)).passed

    delayed = dict(observations)
    delayed_time = sensor_time.copy()
    delayed_time[8:] += 0.02
    for camera_name in ("head", "right_wrist"):
        delayed[f"{camera_name}_rgb_timestamp_s"] = delayed_time.copy()
        delayed[f"{camera_name}_depth_timestamp_s"] = delayed_time.copy()
    errors = validate_episode(replace(episode, observations=delayed)).errors
    assert any("NOT_25HZ_ACQUISITION_CLOCK" in error for error in errors)


def test_persistent_close_edge_and_explicit_reopen():
    latch = PersistentGripperCommand()
    assert latch.apply() == (0.0, False)
    assert latch.apply(close=True) == (1.0, True)
    assert latch.apply() == (1.0, False)
    assert keyboard_action([0.001, 0, 0], latch).tolist() == pytest.approx([0.001, 0, 0, 1])
    assert latch.apply(open_=True) == (0.0, False)


def test_pilot_has_21_bounded_conditions_and_pose_backoff():
    conditions = pilot_conditions()
    assert len(conditions) == 21
    assert {item.backoff_m for item in conditions} == {0.025, 0.030, 0.035}
    center = next(item for item in conditions if item.condition_id == "backoff_30mm_center")
    pose = near_grasp_pose_root_m_xyzw([0.4, 0, 0.8, 0, 0, 0, 1], [1, 0, 0], center)
    assert pose == pytest.approx((0.370, 0, 0.8, 0, 0, 0, 1))


def test_keyboard_v3_operator_views_match_v1_without_recording_perspective():
    assert KEYBOARD_V3_OPERATOR_VIEW_NAMES == (
        "perspective",
        "head",
        "right_wrist",
    )
    assert PERSPECTIVE_OPERATOR_EYE_M == pytest.approx((0.975, -0.84, 1.235))


def test_source_nominal_and_center_30mm_near_grasp_contract():
    nominal = nominal_grasp_pose_root_m_xyzw_for_cube((0.49, -0.24, 0.76))
    center = next(
        item for item in pilot_conditions()
        if item.condition_id == "backoff_30mm_center"
    )
    near = near_grasp_pose_root_m_xyzw(nominal, (1.0, 0.0, 0.0), center)
    assert nominal[:3] == pytest.approx([0.49, -0.24, 0.78])
    assert near[:3] == pytest.approx([0.46, -0.24, 0.78])
    assert near[3:] == pytest.approx(nominal[3:])


def test_episode_roundtrip_and_direct_gru_collation(tmp_path: Path):
    episode = _episode()
    validation = validate_episode(episode)
    assert validation.passed, validation.errors
    assert validation.student_privileged_input_count == 0
    path = tmp_path / "episode.hdf5"
    write_keyboard_v3_episode(path, episode)
    loaded = read_keyboard_v3_episode(path)
    inputs, targets = episode_to_gru(loaded)
    assert not hasattr(inputs, "head_rgb")
    assert "head_rgb" in loaded.observations
    assert inputs.previous_policy_action_4d_metric_root_m.shape == (1, 12, 4)
    assert torch.all(inputs.right_wrist_rgb == 1)
    assert targets.xyz_action_robot_root_m.shape == (1, 12, 3)
    assert targets.close_target[0, 6, 0].item() == 1.0
    assert targets.close_target_semantics == "HUMAN_CLOSE_EDGE"
    assert int(targets.close_target.sum().item()) == 1
    assert bool(targets.close_temporal_valid_mask[0, :7].all())
    assert not bool(targets.close_temporal_valid_mask[0, 7:].any())
    assert int(
        targets.close_temporal_valid_mask[0, 7:].sum().item()
    ) == 0
    # CLOSE onset is not visible through same-row gripper state.  The input is
    # exactly the previously issued action and episode start is OPEN/zero.
    assert inputs.current_gripper_state[0, 0, 0].item() == 0.0
    assert torch.equal(
        inputs.current_gripper_state,
        inputs.previous_policy_action_4d_metric_root_m[..., 3:4],
    )
    assert inputs.current_gripper_state[0, 6, 0].item() == 0.0
    assert targets.close_target[0, 6, 0].item() == 1.0
    assert inputs.current_gripper_state[0, 7, 0].item() == 1.0
    window_inputs, _ = episode_to_gru(loaded, close_window=(-10, 10))
    assert window_inputs.right_wrist_rgb.shape[1] == 12
    with pytest.raises(KeyboardV3DatasetError):
        write_keyboard_v3_episode(path, episode)


def test_v1_parity_head_and_right_wrist_recordings_are_mandatory(tmp_path: Path):
    episode = _episode()
    missing = dict(episode.observations)
    del missing["head_depth_m"]
    validation = validate_episode(replace(episode, observations=missing))
    assert "OBSERVATION_HEAD_DEPTH_M_MISSING" in validation.errors

    mismatch = dict(episode.observations)
    mismatch["head_rgb"] = mismatch["head_rgb"][:, :, :-1]
    validation = validate_episode(replace(episode, observations=mismatch))
    assert "RECORDED_CAMERA_RESOLUTION_MISMATCH" in validation.errors

    path = tmp_path / "v1_parity_views.hdf5"
    write_keyboard_v3_episode(path, episode)
    loaded = read_keyboard_v3_episode(path)
    for index, camera_name in enumerate(("head", "right_wrist")):
        assert loaded.observations[f"{camera_name}_rgb"].shape == (12, 8, 10, 3)
        assert np.all(loaded.observations[f"{camera_name}_rgb"] == index)


def _strict_checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    model = HumanGraspGRUBC(HumanGraspGRUConfig())
    split = EpisodeSplit(
        train=("episode_train",),
        validation=("episode_validation",),
        test=("episode_test",),
        seed=42,
    )
    calibration = HysteresisCalibration(
        open_threshold=0.3,
        close_threshold=0.7,
        validation_split_fingerprint=split.fingerprint,
        close_precision=0.8,
        close_recall=0.8,
        close_f1=0.8,
    )
    weight = ClosePositiveWeightReceipt(
        split_fingerprint=split.fingerprint,
        positive_rows=2,
        negative_rows=4,
        positive_weight=2.0,
    )
    torch.save(
        {
            "schema": HUMAN_GRASP_GRU_CHECKPOINT_SCHEMA,
            "config": dict(model.config.__dict__),
            "contract": model.config.contract_payload(),
            "state_dict": model.state_dict(),
            "state_sha256": tensor_state_sha256(model),
            "optimizer_step": 3,
            "episode_split": split.as_dict(),
            "episode_split_fingerprint": split.fingerprint,
            "close_positive_weight": weight.as_dict(),
            "hysteresis_calibration": calibration.as_dict(),
            "evaluation_metrics": {"validation/loss": 1.0},
        },
        path,
    )
    return path


def test_offline_multiview_causal_replay_uses_strict_wrist_checkpoint(tmp_path: Path):
    dataset = tmp_path / "episode.hdf5"
    checkpoint = tmp_path / "model.pt"
    write_keyboard_v3_episode(dataset, _episode())
    _strict_checkpoint(checkpoint)
    replay = KeyboardV3OfflineReplay(dataset, checkpoint)
    report = replay.report()
    assert report["recorded_camera_names"] == ["head", "right_wrist"]
    assert report["operator_view_names"] == ["perspective", "head", "right_wrist"]
    assert report["grasp_actor_camera_names"] == ["right_wrist"]
    assert report["head_camera_policy_input"] is False
    assert report["dataset_extra_views_are_checkpoint_inputs"] is False
    assert report["isaac_started"] is False
    assert report["action_submissions"] == 0
    frame = replay.frame(7)
    full_xyz = replay.full_output.xyz_action_robot_root_m[0, 7].detach().cpu().numpy()
    assert np.asarray(frame.predicted_xyz_m) == pytest.approx(full_xyz)
    for index, camera_name in enumerate(("head", "right_wrist")):
        rgb, depth = replay.camera_frame(camera_name, 4)
        assert np.all(rgb == index)
        assert depth.shape == (8, 10, 1)
    with pytest.raises(
        KeyboardV3OfflineReplayError,
        match="RESIDUAL_SAC_CHECKPOINT_LOADER_NOT_IMPLEMENTED",
    ):
        KeyboardV3OfflineReplay(
            dataset, checkpoint, residual_checkpoint_path=tmp_path / "unsupported.pt"
        )


def test_direct_v3_batch_runs_human_only_gru_without_privileged_input():
    inputs, targets = episode_to_gru(_episode())
    config = HumanGraspGRUConfig()
    model = HumanGraspGRUBC(config)
    output = model(inputs)
    assert output.action_4d.shape == (1, 12, 4)
    assert output.feasibility_probability.shape == (1, 12, 1)
    assert targets.xyz_mask_authority == "VALID_DIRECT"


def test_gru_prefix_is_invariant_to_every_future_frame():
    inputs, _targets = episode_to_gru(_episode())
    model = HumanGraspGRUBC(HumanGraspGRUConfig()).eval()
    cutoff = 5
    changed = replace(
        inputs,
        right_wrist_rgb=inputs.right_wrist_rgb.clone(),
        right_wrist_depth_m=inputs.right_wrist_depth_m.clone(),
        ee_pose_robot_root_m_xyzw=inputs.ee_pose_robot_root_m_xyzw.clone(),
    )
    changed.right_wrist_rgb[:, cutoff + 1 :] = 255
    changed.right_wrist_depth_m[:, cutoff + 1 :] = 1.5
    changed.ee_pose_robot_root_m_xyzw[:, cutoff + 1 :, :3] = 9.0
    with torch.inference_mode():
        baseline = model(inputs).action_4d[:, : cutoff + 1]
        observed = model(changed).action_4d[:, : cutoff + 1]
    assert torch.equal(baseline, observed)


def test_previous_action_causality_and_pad_proxy_fail_closed():
    episode = _episode()
    bad_previous = dict(episode.actions)
    bad_previous["previous_action_4d"] = np.asarray(bad_previous["previous_action_4d"]).copy()
    bad_previous["previous_action_4d"][4, 0] = 0.002
    validation = validate_episode(replace(episode, actions=bad_previous))
    assert "PREVIOUS_ACTION_NOT_CAUSAL" in validation.errors
    privileged = dict(episode.privileged)
    privileged["pad_midpoint_root_m"] = np.zeros((12, 3), dtype=np.float32)
    validation = validate_episode(replace(episode, privileged=privileged))
    assert "PAD_MIDPOINT_PROXY_FORBIDDEN" in validation.errors


def test_training_readiness_requires_three_episode_split(tmp_path: Path):
    paths = []
    for index in range(3):
        path = tmp_path / f"episode_{index}.hdf5"
        write_keyboard_v3_episode(path, _episode(f"episode_{index:06d}"))
        paths.append(path)
    receipt = training_readiness(paths)
    assert receipt["gru_bc_training_ready"] is True
    assert receipt["legacy_migration_required"] is False
    assert receipt["student_privileged_input_count"] == 0
    assert receipt["train_val_test_episode_split"] == "PASS"


def test_no_close_failure_is_valid_open_hard_negative_and_collates() -> None:
    episode = _episode()
    rows = episode.actions["action_4d"].shape[0]
    action = np.asarray(episode.actions["action_4d"]).copy()
    action[:, 3] = 0.0
    previous = np.zeros_like(action)
    previous[1:] = action[:-1]
    metadata = dict(episode.metadata)
    metadata.update(
        {
            "post_close_observed_steps": 0,
            "future_outcome_right_censored": False,
            "success": False,
            "failure_type": "FAIL_OPERATOR_ABORT",
        }
    )
    no_close = replace(
        episode,
        metadata=metadata,
        actions={"action_4d": action, "previous_action_4d": previous},
        events={
            "close_state": np.zeros(rows, dtype=bool),
            "close_edge": np.zeros(rows, dtype=bool),
        },
        privileged={
            **episode.privileged,
            "gripper_command": np.zeros(rows, dtype=np.float32),
        },
    )
    validation = validate_episode(no_close)
    assert validation.passed, validation.errors
    inputs, targets = episode_to_gru(no_close)
    assert inputs.right_wrist_rgb.shape[1] == rows
    assert not bool(targets.close_target.any())


def test_missing_future_label_telemetry_is_rejected() -> None:
    episode = _episode()
    privileged = dict(episode.privileged)
    del privileged["stable_grasp"]
    validation = validate_episode(replace(episode, privileged=privileged))
    assert "PRIVILEGED_STABLE_GRASP_MISSING" in validation.errors


def test_raw_outcomes_are_relabelled_offline_without_production_threshold() -> None:
    episode = _episode()
    privileged = dict(episode.privileged)
    privileged["contact_force_left_n"][7:] = 2.0
    privileged["contact_force_right_n"][7:] = 2.0
    privileged["total_normal_force_n"][7:] = 4.0
    privileged["left_contact"][7:] = True
    privileged["right_contact"][7:] = True
    privileged["contact"][7:] = True
    privileged["exact_inner_contact"][7:] = True
    privileged["exact_outer_contact"][7:] = True
    privileged["bilateral_contact"][7:] = True
    labels = derive_offline_grasp_labels(
        replace(episode, privileged=privileged),
        thresholds=OfflineGraspLabelThresholds(
            minimum_total_force_n=2.0,
            maximum_total_force_n=10.0,
            maximum_relative_speed_m_s=0.01,
            stable_dwell_s=0.06,
            distribution_receipt_sha256="a" * 64,
        ),
        cube_retained=np.ones(12, dtype=bool),
        cube_retained_valid=np.ones(12, dtype=bool),
    )
    assert labels["authority"] == "ANALYSIS_ONLY_NOT_PRODUCTION_LOCKED"
    assert labels["bilateral_contact"].tolist() == [False] * 7 + [True] * 5
    assert labels["stable"].tolist() == [False] * 9 + [True] * 3
    assert labels["grasp_success"] is True


def test_runtime_recorder_publishes_direct_v3(tmp_path: Path):
    condition = next(item for item in pilot_conditions() if item.condition_id == "backoff_30mm_center")
    recorder = KeyboardV3EpisodeRecorder(
        episode_id="episode_runtime",
        planner=PlannerStartReceipt(
            nominal_grasp_pose_root_m_xyzw=[0.4, 0, 0.8, 0, 0, 0, 1],
            near_grasp_pose_root_m_xyzw=[0.370, 0, 0.8, 0, 0, 0, 1],
            approach_axis_root=[1, 0, 0],
            nominal_grasp_q_rad=[0] * 7,
            near_grasp_q_rad=[0] * 7,
            condition=condition,
        ),
        rgbd_hz=25,
    )
    recorder.start_recording()
    for step in range(5):
        recorder.append(
            KeyboardV3Row(
                **_recorded_camera_row_fields(step, rgbd_hz=25),
                ee_position_root_m=[0.3, 0, 0.8],
                ee_quat_root_xyzw=[0, 0, 0, 1],
                arm_q_rad=[0] * 7,
                arm_qd_rad_s=[0] * 7,
                gripper_state=float(step >= 3),
                action_4d=[0.001 if step < 3 else 0.0, 0, 0, float(step >= 3)],
                cube_center_root_m=[0.4, 0, 0.8],
                timestamp_s=step * 0.02,
                control_step=step,
                privileged_optional=_outcome_fields(step),
            )
        )
    path = tmp_path / "runtime.hdf5"
    receipt = recorder.finalize(path, success=False, failure_type="FAIL_LATE_CLOSE")
    assert receipt["passed"] is True
    loaded = read_keyboard_v3_episode(path)
    assert loaded.actions["action_4d"].shape == (5, 4)
    assert loaded.metadata["recorded_camera_names"] == ["head", "right_wrist"]
    assert loaded.metadata["operator_view_names"] == ["perspective", "head", "right_wrist"]
    assert loaded.observations["head_frame_id"].tolist() == [0, 0, 1, 1, 2]
    assert loaded.observations["right_wrist_frame_id"].tolist() == [0, 0, 1, 1, 2]
    assert loaded.metadata["handoff_distance_m"] == pytest.approx(0.030)
    assert loaded.metadata["local_grasp_band_m"].tolist() == pytest.approx(
        [0.015, 0.022]
    )
    assert bool(loaded.metadata["local_grasp_distance_is_student_input"]) is False
    np.testing.assert_allclose(
        loaded.planner["current_nominal_grasp_residual_m"],
        np.full(5, 0.1, dtype=np.float32),
    )
    assert loaded.planner["local_grasp_phase_code"].tolist() == [0, 0, 0, 4, 4]
    assert nominal_grasp_translation_residual_m(
        [0.378, 0.0, 0.8], [0.4, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
    ) == pytest.approx(0.022)
    assert classify_local_grasp_phase(0.022, gripper_closed=False) == (
        "GRASP_DECISION_BAND"
    )


def test_runtime_recorder_preserves_dual_raw_depth_calibration_and_gru_inputs(tmp_path: Path):
    condition = next(item for item in pilot_conditions() if item.condition_id == "backoff_30mm_center")
    calibration = {
        name: {
            "camera_name": name,
            "source_scene_key": f"{name}_camera",
            "source_frame_id": f"/World/envs/env_.*/Robot/{name}",
            "intrinsics_3x3": [[100.0, 0.0, 128.0], [0.0, 100.0, 96.0], [0.0, 0.0, 1.0]],
            "image_width": 256,
            "image_height": 192,
            "extrinsic_reference_frame": "robot_root",
            "pose_convention": "position_m+quaternion_xyzw",
            "timestamp_source": "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE",
        }
        for name in ("head", "right_wrist")
    }
    recorder = KeyboardV3EpisodeRecorder(
        episode_id="episode_rgbd_evidence",
        planner=PlannerStartReceipt(
            nominal_grasp_pose_root_m_xyzw=[0.4, 0, 0.8, 0, 0, 0, 1],
            near_grasp_pose_root_m_xyzw=[0.370, 0, 0.8, 0, 0, 0, 1],
            approach_axis_root=[1, 0, 0],
            condition=condition,
        ),
        rgbd_hz=25,
        camera_calibration_metadata=calibration,
    )
    recorder.start_recording()
    for step in range(5):
        camera = _recorded_camera_row_fields(step, rgbd_hz=25)
        for name in ("head", "right_wrist"):
            camera[f"{name}_depth_source_m"][0, 0, 0] = np.inf
            camera[f"{name}_depth_source_valid"][0, 0, 0] = False
            camera[f"{name}_camera_pose_root_m_xyzw"] = [0, 0, 0, 0, 0, 0, 1]
        recorder.append(
            KeyboardV3Row(
                **camera,
                ee_position_root_m=[0.3, 0, 0.8],
                ee_quat_root_xyzw=[0, 0, 0, 1],
                arm_q_rad=[0] * 7,
                arm_qd_rad_s=[0] * 7,
                gripper_state=float(step >= 3),
                action_4d=[0.001 if step < 3 else 0.0, 0, 0, float(step >= 3)],
                cube_center_root_m=[0.4, 0, 0.8],
                timestamp_s=step * 0.02,
                control_step=step,
                privileged_optional=_outcome_fields(step),
            )
        )
    path = tmp_path / "rgbd_evidence.hdf5"
    recorder.finalize(path, success=False, failure_type="FAIL_LATE_CLOSE")
    loaded = read_keyboard_v3_episode(path)
    assert loaded.metadata["camera_evidence_extension_active"]
    assert loaded.metadata["depth_raw_unit"] == "meter"
    assert loaded.metadata["depth_to_meter_scale"] == 1.0
    assert np.isposinf(loaded.observations["head_depth_raw_m"][0, 0, 0, 0])
    assert loaded.observations["head_depth_m"][0, 0, 0, 0] == 0.0
    assert np.array_equal(
        loaded.observations["associated_right_wrist_frame_index"],
        loaded.observations["right_wrist_frame_id"],
    )
    inputs, _targets = episode_to_gru(loaded)
    assert inputs.right_wrist_rgb.shape[1] == 5
    assert "head_rgb" not in inputs.__dict__
    assert LOCAL_GRASP_PHASE_CODE["GRASP_DECISION_BAND"] == 2


def _ui_recorder(episode_id: str, *, invalid_pad: bool = False) -> KeyboardV3EpisodeRecorder:
    condition = next(
        item for item in collection_target_conditions() if item.backoff_m == 0.018
    )
    recorder = KeyboardV3EpisodeRecorder(
        episode_id=episode_id,
        planner=PlannerStartReceipt(
            nominal_grasp_pose_root_m_xyzw=[0.4, 0, 0.8, 0, 0, 0, 1],
            near_grasp_pose_root_m_xyzw=[0.382, 0, 0.8, 0, 0, 0, 1],
            approach_axis_root=[1, 0, 0],
            condition=condition,
        ),
        rgbd_hz=25,
    )
    recorder.start_recording()
    for step in range(68):
        optional = _outcome_fields(step, contact_after=5)
        if invalid_pad:
            optional["left_pad_surface_position_root_m"] = [0.0, 0.0, 0.0]
        recorder.append(
            KeyboardV3Row(
                **_recorded_camera_row_fields(step, rgbd_hz=25),
                ee_position_root_m=[0.3, 0, 0.8],
                ee_quat_root_xyzw=[0, 0, 0, 1],
                arm_q_rad=[0] * 7,
                arm_qd_rad_s=[0] * 7,
                gripper_state=float(step >= 3),
                action_4d=[0.001 if step < 3 else 0.0, 0, 0, float(step >= 3)],
                cube_center_root_m=[0.4, 0, 0.8],
                timestamp_s=step * 0.02,
                control_step=step,
                privileged_optional=optional,
            )
        )
    return recorder


def test_enter_only_opens_result_selection_then_success_is_explicit(tmp_path: Path, capsys):
    controller = KeyboardV3OperatorUI(
        _ui_recorder("episode_ui_success"),
        canonical_episode_directory=tmp_path / "episodes",
        rejected_episode_directory=tmp_path / "rejected",
        receipt_directory=tmp_path / "receipts",
    )
    assert not controller.next_episode_allowed
    with pytest.raises(OperatorUIError):
        controller.choose("1")
    controller.press_enter()
    assert controller.result_selection_active
    assert not (tmp_path / "episodes").exists()
    publication = controller.choose("1")
    assert publication.success is True
    assert publication.failure_type == "NONE"
    assert publication.canonical_training_registered is True
    assert controller.next_episode_allowed
    assert (tmp_path / "episodes/episode_ui_success.hdf5").is_file()
    aggregate_path = tmp_path / "g2_keyboard_v3_rgbd.hdf5"
    assert publication.aggregate_save_path == str(aggregate_path)
    assert publication.aggregate_demo_id == "demo_0"
    with h5py.File(aggregate_path, "r") as aggregate:
        assert aggregate.attrs["format_version"] == 1
        assert aggregate.attrs["layout_compatibility"] == "V1_DATA_DEMO_N"
        assert not bool(aggregate.attrs["legacy_v1_8d_action_compatible"])
        assert aggregate["data"].attrs["total"] == 68
        assert aggregate["data"].attrs["num_demos"] == 1
        demo = aggregate["data/demo_0"]
        assert demo.attrs["episode_id"] == "episode_ui_success"
        assert demo.attrs["action_dim"] == 4
        assert demo["actions"].shape == (68, 4)
        assert demo["head_rgb"].shape[0] == 68
        assert demo["right_wrist_rgb"].shape[0] == 68
        assert "left_contact" in demo
        assert "right_contact" in demo
        assert "contact_force_left_n" in demo
        assert "contact_force_right_n" in demo
    output = capsys.readouterr().out
    for field in (
        "EPISODE_ID:", "ROW_COUNT:", "CLOSE_EVENT_COUNT:",
        "NONZERO_XYZ_ROW_COUNT:", "SUCCESS:", "FAILURE_TYPE:",
        "SAVE_PATH:", "VALIDATOR_RESULT:",
    ):
        assert field in output


def test_failure_is_saved_and_discard_is_not_saved(tmp_path: Path):
    failure = KeyboardV3OperatorUI(
        _ui_recorder("episode_ui_failure"),
        canonical_episode_directory=tmp_path / "episodes",
        rejected_episode_directory=tmp_path / "rejected",
        receipt_directory=tmp_path / "receipts",
    )
    failure.press_enter()
    publication = failure.choose("4")
    assert publication.success is False
    assert publication.failure_type == "FAIL_LATERAL_MISALIGNMENT"
    assert publication.canonical_training_registered is True

    discard = KeyboardV3OperatorUI(
        _ui_recorder("episode_ui_discard"),
        canonical_episode_directory=tmp_path / "episodes",
        rejected_episode_directory=tmp_path / "rejected",
        receipt_directory=tmp_path / "receipts",
    )
    discard.press_enter()
    publication = discard.choose("0")
    assert publication.save_path is None
    assert publication.aggregate_save_path is None
    assert publication.canonical_training_registered is False
    assert not (tmp_path / "episodes/episode_ui_discard.hdf5").exists()
    with h5py.File(tmp_path / "g2_keyboard_v3_rgbd.hdf5", "r") as aggregate:
        assert list(aggregate["data"].keys()) == ["demo_0"]
        assert aggregate["data/demo_0"].attrs["success"] == np.False_


def test_validator_failure_routes_to_rejected_episode(tmp_path: Path):
    controller = KeyboardV3OperatorUI(
        _ui_recorder("episode_ui_rejected", invalid_pad=True),
        canonical_episode_directory=tmp_path / "episodes",
        rejected_episode_directory=tmp_path / "rejected",
        receipt_directory=tmp_path / "receipts",
    )
    controller.press_enter()
    publication = controller.choose("2")
    assert publication.canonical_training_registered is False
    assert publication.validator_result.startswith("FAIL_REJECTED:")
    assert not (tmp_path / "episodes/episode_ui_rejected.hdf5").exists()
    rejected = tmp_path / "rejected/episode_ui_rejected.hdf5"
    assert rejected.is_file()
    with pytest.raises(KeyboardV3DatasetError):
        read_keyboard_v3_episode(rejected)


def test_final_collection_target_scheduler_terminal_and_schema(tmp_path: Path) -> None:
    assert list(COLLECTION_TARGETS_MM) == [16, 17, 18, 19, 20, 21, 22]
    receipts = [
        {"target_residual_mm": target}
        for target in COLLECTION_TARGETS_MM
        for _ in range(2)
    ]
    assert target_distribution_counts(receipts) == {
        target: 2 for target in COLLECTION_TARGETS_MM
    }
    assert select_next_target_mm(receipts) == 16
    metrics = preclose_geometry_metrics(
        ee_position_root_m=[0.382, 0.0, 0.8],
        previous_ee_position_root_m=[0.383, 0.0, 0.8],
        cube_center_root_m=[0.42, 0.0, 0.8],
        nominal_grasp_pose_root_m_xyzw=[0.4, 0.0, 0.8, 0, 0, 0, 1],
        approach_axis_root=[1, 0, 0],
        action_xyz_root_m=[0.001, 0.0, 0.0],
    )
    display = format_terminal_status(
        episode_id="dry-run",
        control_step=7,
        target_residual_mm=18,
        metrics=metrics,
        action_xyz_root_m=[0.001, 0.0, 0.0],
        gripper_state="OPEN",
        close_edge=False,
        close_state=False,
        phase="GRASP_DECISION_BAND",
    )
    for label in (
        "Target nominal residual", "Current nominal residual",
        "EE->Cube center distance", "Lateral error", "Approach cosine",
        "XYZ command robot_root", "Pad-surface distance",
    ):
        assert label in display
    recorder = _ui_recorder("episode_final_readiness")
    episode = recorder.build_episode(success=False, failure_type="FAIL_OPERATOR_ABORT")
    validation = validate_episode(episode)
    assert validation.passed, validation.errors
    for field in (
        "target_residual_m", "current_nominal_grasp_residual_m",
        "lateral_error_m", "approach_cosine", "ee_speed_m_s", "phase_code",
    ):
        assert field in episode.planner
    inputs, targets = episode_to_gru(episode)
    assert inputs.right_wrist_rgb.shape[1] == 68
    assert targets.xyz_action_robot_root_m.shape[-1] == 3
    collection_target_conditions,
    format_terminal_status,
