# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

from dataclasses import replace
import os
import pty
from pathlib import Path
import termios
from types import SimpleNamespace

import numpy as np

from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
    BASELINE_NAME,
    LEFT_ARM_DOWN_Q_RAD,
    LEFT_ARM_POSTURE_ID,
    apply_left_arm_down_v2_to_cfg,
    left_arm_limit_margin_min_rad,
    load_baseline_manifest,
    selected_pregrasp_initial_state,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_collection_contract import (
    NearGraspCondition,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_direct_pregrasp_init import (
    direct_pregrasp_plan_receipt,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_terminal_input import (
    KeyboardV3TerminalInput,
)
from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
    AbstractGripperIntent,
    HighLevelPolicyAction,
    expand_to_existing_controller_8d,
)
from geniesim.rl.isaaclab.g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from geniesim.rl.sac.keyboard_v3_dataset import validate_episode

from test_g2_keyboard_v3_dataset import _episode


ROOT = Path(__file__).resolve().parents[1]
OLD_ASSET = ROOT / (
    "artifacts/g2_bounded_passive_range_qualification_20260921/"
    "candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
)
OLD_HASH = "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"


def test_manifest_preserves_candidate_and_selected_robot_cube_pair() -> None:
    path, manifest, manifest_hash = load_baseline_manifest()
    sample = selected_pregrasp_initial_state()
    assert path.is_file() and len(manifest_hash) == 64
    assert manifest["baseline_name"] == BASELINE_NAME
    assert manifest["old_candidate_a_asset"]["modified"] is False
    assert manifest["omnipicker_product_contract"]["maximum_gripping_force_n"] == 30.0
    assert manifest["omnipicker_product_contract"]["physical_pad_touch_sensor_available"] is False
    assert manifest["omnipicker_product_contract"]["student_contact_input_count"] == 0
    import hashlib

    assert hashlib.sha256(OLD_ASSET.read_bytes()).hexdigest() == OLD_HASH
    assert sample.sample_id
    assert sample.gripper_state == "OPEN"
    assert sample.pad_to_cube_distance_m > sample.ee_to_cube_distance_m > 0.0
    assert left_arm_limit_margin_min_rad() > 0.48


def test_common_reset_authority_changes_only_left_arm_defaults() -> None:
    initial = {"other_joint": 1.25, **{f"idx2{i}_arm_l_joint{i}": -9.0 for i in range(1, 8)}}
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(init_state=SimpleNamespace(joint_pos=dict(initial)))
        )
    )
    receipt = apply_left_arm_down_v2_to_cfg(cfg)
    assert cfg.scene.robot.init_state.joint_pos["other_joint"] == 1.25
    assert tuple(
        cfg.scene.robot.init_state.joint_pos[f"idx2{i}_arm_l_joint{i}"]
        for i in range(1, 8)
    ) == LEFT_ARM_DOWN_Q_RAD
    assert receipt["baseline_name"] == BASELINE_NAME
    assert receipt["left_arm_posture_id"] == LEFT_ARM_POSTURE_ID


def test_direct_pregrasp_receipt_contains_no_startup_motion() -> None:
    plan = direct_pregrasp_plan_receipt(
        NearGraspCondition(
            condition_id="dataset_pregrasp_15mm_contract",
            backoff_m=0.030,
            start_offset_xyz_m=(0.0, 0.0, 0.0),
        )
    )
    assert np.array_equal(plan.q_rad[0], plan.q_rad[1])
    assert np.array_equal(plan.ee_position_root_m[0], plan.ee_position_root_m[1])
    receipt = plan.planner_only_receipt()
    assert receipt["startup_curobo_planning_count"] == 0
    assert receipt["startup_curobo_execution_count"] == 0


def test_terminal_keyboard_is_typed_4d_nonblocking_and_restores_tty() -> None:
    master_fd, slave_fd = pty.openpty()
    original = termios.tcgetattr(slave_fd)
    stream = os.fdopen(os.dup(slave_fd), "r", buffering=1)
    try:
        terminal = KeyboardV3TerminalInput(stream=stream, translation_step_m=0.0045)
        os.write(master_fd, b"w")
        result = terminal.poll()
        assert result.action_4d_metric_root_m == (0.0045, 0.0, 0.0, 0.0)
        os.write(master_fd, b"k")
        result = terminal.poll()
        assert result.event == "GRIPPER_CLOSE"
        assert result.action_4d_metric_root_m == (0.0, 0.0, 0.0, 1.0)
        os.write(master_fd, b"k")
        repeated = terminal.poll()
        assert repeated.event == "GRIPPER_CLOSE_HELD"
        assert repeated.action_4d_metric_root_m == (0.0, 0.0, 0.0, 1.0)
        terminal.close()
        assert terminal.closed
        assert termios.tcgetattr(slave_fd) == original
    finally:
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_terminal_keyboard_default_speed_is_throttled_without_clipping() -> None:
    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(os.dup(slave_fd), "r", buffering=1)
    now = [10.0]
    try:
        terminal = KeyboardV3TerminalInput(stream=stream, clock=lambda: now[0])
        os.write(master_fd, b"w")
        first = terminal.poll()
        assert first.action_4d_metric_root_m == (0.001, 0.0, 0.0, 0.0)
        os.write(master_fd, b"w")
        suppressed = terminal.poll()
        assert suppressed.action_4d_metric_root_m == (0.0, 0.0, 0.0, 0.0)
        assert terminal.suppressed_motion_key_count == 1
        now[0] += 0.050
        os.write(master_fd, b"w")
        accepted = terminal.poll()
        assert accepted.action_4d_metric_root_m == (0.001, 0.0, 0.0, 0.0)
        terminal.close()
    finally:
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_terminal_keyboard_cosine_smoothing_preserves_distance_and_bound() -> None:
    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(os.dup(slave_fd), "r", buffering=1)
    try:
        terminal = KeyboardV3TerminalInput(
            stream=stream,
            translation_step_m=0.001,
            smoothing_steps=5,
        )
        os.write(master_fd, b"w")
        samples = [terminal.poll().action_4d_metric_root_m[0]]
        samples.extend(
            terminal.poll().action_4d_metric_root_m[0] for _ in range(4)
        )
        assert all(0.0 < value < 0.001 for value in samples)
        assert np.isclose(sum(samples), 0.001, atol=1.0e-12)
        assert np.allclose(samples, list(reversed(samples)), atol=1.0e-15)
        terminal.close()
    finally:
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_terminal_keyboard_all_six_isolated_keys_preserve_root_axis_sign_and_scale() -> None:
    expected = {
        "w": (1.0, 0.0, 0.0),
        "s": (-1.0, 0.0, 0.0),
        "a": (0.0, 1.0, 0.0),
        "d": (0.0, -1.0, 0.0),
        "q": (0.0, 0.0, 1.0),
        "e": (0.0, 0.0, -1.0),
    }
    for key, direction in expected.items():
        master_fd, slave_fd = pty.openpty()
        stream = os.fdopen(os.dup(slave_fd), "r", buffering=1)
        try:
            terminal = KeyboardV3TerminalInput(
                stream=stream,
                translation_step_m=0.001,
                motion_min_interval_s=0.0,
                smoothing_steps=5,
            )
            os.write(master_fd, key.encode("ascii"))
            rows = [terminal.poll().action_4d_metric_root_m]
            rows.extend(terminal.poll().action_4d_metric_root_m for _ in range(4))
            total = np.sum(np.asarray(rows, dtype=np.float64)[:, :3], axis=0)
            assert np.allclose(total, np.asarray(direction) * 0.001, atol=1.0e-12)
            assert all(
                np.count_nonzero(np.abs(np.asarray(row[:3])) > 1.0e-15) == 1
                for row in rows
            )

            first_xyz = np.asarray(rows[0][:3], dtype=np.float64)
            normalized = first_xyz / G2_TRANSLATION_ACTION_SCALE_M
            packet = expand_to_existing_controller_8d(
                HighLevelPolicyAction((*normalized.tolist(), 0.0)),
                gripper_intent=AbstractGripperIntent.OPEN,
            )
            assert np.allclose(
                np.asarray(packet[:3]) * G2_TRANSLATION_ACTION_SCALE_M,
                first_xyz,
                atol=1.0e-12,
            )
            assert packet[3:7] == (0.0, 0.0, 0.0, 0.0)
            assert packet[7] == 1.0
            terminal.close()
        finally:
            stream.close()
            os.close(master_fd)
            os.close(slave_fd)


def test_terminal_close_cancels_pending_cosine_translation_tail() -> None:
    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(os.dup(slave_fd), "r", buffering=1)
    try:
        terminal = KeyboardV3TerminalInput(
            stream=stream,
            translation_step_m=0.001,
            smoothing_steps=5,
        )
        os.write(master_fd, b"w")
        first = terminal.poll()
        assert first.action_4d_metric_root_m[0] > 0.0
        os.write(master_fd, b"k")
        close = terminal.poll()
        assert close.action_4d_metric_root_m == (0.0, 0.0, 0.0, 1.0)
        after = terminal.poll()
        assert after.action_4d_metric_root_m == (0.0, 0.0, 0.0, 1.0)
        terminal.close()
    finally:
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_terminal_axis_switch_cancels_old_tail_and_e_is_only_negative_root_z() -> None:
    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(os.dup(slave_fd), "r", buffering=1)
    try:
        terminal = KeyboardV3TerminalInput(
            stream=stream,
            translation_step_m=0.001,
            motion_min_interval_s=0.0,
            smoothing_steps=5,
        )
        os.write(master_fd, b"w")
        assert terminal.poll().action_4d_metric_root_m[0] > 0.0
        assert terminal.poll().action_4d_metric_root_m[0] > 0.0
        os.write(master_fd, b"e")
        switched = terminal.poll()
        assert switched.action_4d_metric_root_m[0] == 0.0
        assert switched.action_4d_metric_root_m[1] == 0.0
        assert np.isclose(
            switched.action_4d_metric_root_m[2],
            -1.0 / 12_000.0,
            atol=1.0e-15,
        )
        assert terminal.cross_axis_tail_cancel_count == 1
        for _ in range(4):
            row = terminal.poll().action_4d_metric_root_m
            assert row[0] == 0.0
            assert row[1] == 0.0
            assert row[2] < 0.0
        terminal.close()
    finally:
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_separate_terminal_launcher_preserves_diagnostics_after_child_exit() -> None:
    launcher = ROOT / "scripts/run_g2_keyboard_v3_terminal_collection.sh"
    source = launcher.read_text(encoding="utf-8")
    assert "gnome-terminal" in source
    assert "exec bash --noprofile --norc" in source
    assert "PROCESS_EXIT_CODE" in source
    assert "--keyboard-v3-translation-step-m" in source
    assert "--keyboard-v3-motion-min-interval-s" in source
    assert "--keyboard-v3-terminal-smoothing-steps" in source
    assert "--keyboard-v3-continuous-session" in source
    assert "--keyboard-v3-session-max-episodes" in source


def test_v2_dataset_requires_complete_pair_provenance() -> None:
    episode = _episode()
    sample = selected_pregrasp_initial_state()
    metadata = {
        **episode.metadata,
        "baseline_variant": BASELINE_NAME,
        "left_arm_posture_id": LEFT_ARM_POSTURE_ID,
        "pregrasp_init_source": "CUROBO_DATASET",
        "pregrasp_sample_id": sample.sample_id,
        "cube_sample_id": sample.sample_id,
        "head_camera": "ENABLED",
        "wrist_camera": "ENABLED",
        "control_source": "TERMINAL_KEYBOARD",
        "omnipicker_product_model": "OmniPicker",
        "omnipicker_manual_authority_url": "https://www.agibot.com/filepage/265.html",
        "omnipicker_manual_hardware_version": "1.2",
        "omnipicker_pcba_version": "UNRESOLVED_20_OR_30",
        "omnipicker_firmware_version": "UNRESOLVED_RUNTIME_DEVICE",
        "omnipicker_maximum_gripping_force_n": 30.0,
        "omnipicker_physical_pad_touch_sensor_available": False,
        "omnipicker_protocol_force_feedback_semantics": "CURRENT_MOTOR_TORQUE_NORMALIZED_0_TO_FF",
        "sim_contact_telemetry_role": "PRIVILEGED_TEACHER_LABEL_AND_M2_DIAGNOSTIC_ONLY",
    }
    assert validate_episode(replace(episode, metadata=metadata)).passed
    metadata["cube_sample_id"] = "different-row"
    assert "METADATA_ROBOT_CUBE_SAMPLE_PAIR_MISMATCH" in validate_episode(
        replace(episode, metadata=metadata)
    ).errors


def test_future_pipeline_entrypoints_bind_v2_factory() -> None:
    runner = (ROOT / "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py").read_text()
    sac = (ROOT / "scripts/run_g2_contact_free_bc_regularized_sac_25env.py").read_text()
    prepare = (ROOT / "scripts/prepare_g2_keyboard_v3_collection.py").read_text()
    symbol = "make_g2_candidate_a_left_arm_down_v2_training_env_cfg"
    assert symbol in runner
    assert symbol in sac
    assert "g2_keyboard_v3_collection_manifest_v6_boolean_contact_primary" in prepare
