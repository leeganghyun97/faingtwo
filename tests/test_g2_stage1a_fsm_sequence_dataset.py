# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Regression coverage for the deterministic-FSM GRU sequence sidecar."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from geniesim.rl.sac.stage1a_fsm_sequence_dataset import (
    FSM_SEQUENCE_SCHEMA,
    FsmSequenceDatasetWriter,
    V6_WRIST_DIAGNOSTIC_FRAME_STORE_SCHEMA,
)
from geniesim.rl.sac.stage1a_v6_wrist_diagnostic import (
    _mask_from_renderer_tokens,
    _renderer_label_tokens,
)
from geniesim.rl.sac.stage1a_vector_contract import PerEnvCameraFrame


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts/audit_g2_stage1a_fsm_sequence_dataset.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("fsm_sequence_audit", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(*, frame_id: int, timestamp_s: float) -> PerEnvCameraFrame:
    return PerEnvCameraFrame(
        env_id=0,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        rgb=np.zeros((192, 256, 3), dtype=np.uint8),
        depth_m=np.ones((192, 256, 1), dtype=np.float32),
        depth_valid=np.ones((192, 256, 1), dtype=np.bool_),
    )


def test_fsm_sequence_audit_accepts_dual_rgbd_causal_rows(tmp_path: Path) -> None:
    writer = FsmSequenceDatasetWriter(output_dir=tmp_path, num_envs=25)
    for step in range(25):
        frame_id = step // 2
        timestamp = 0.04 * frame_id
        head_index = writer.append_frame(
            sensor="head", episode_id=0, frame=_frame(frame_id=frame_id, timestamp_s=timestamp)
        )
        wrist_index = writer.append_frame(
            sensor="right_wrist", episode_id=0, frame=_frame(frame_id=frame_id, timestamp_s=timestamp)
        )
        writer.append_control_row(
            {
                "schema": FSM_SEQUENCE_SCHEMA,
                "episode_id": 0,
                "env_id": 0,
                "source_family_id": "episode-00005",
                "source_sample_id": "episode-00005/row-000091",
                "timestep": step,
                "control_step": step,
                "control_timestamp_s": 0.02 * step,
                "head_frame_store_index": head_index,
                "right_wrist_frame_store_index": wrist_index,
                "head_camera_frame_id": frame_id,
                "right_wrist_camera_frame_id": frame_id,
                "head_camera_timestamp_s": timestamp,
                "right_wrist_camera_timestamp_s": timestamp,
                "teacher_geometry_timestamp_s": timestamp,
                "gru_reset_generation": 0,
                "student_privileged_input_count": 0,
                "gru_runtime_authority": False,
                "privileged_runtime_authority": False,
                "student_observation": {"actor_observation": [0.0] * 128},
                "previous_action_4d_metric_root_m": [0.0] * 4,
                "executed_action_4d_metric_root_m": [0.0] * 4,
                "sac_residual_xyz_action_m": [0.0] * 3,
                "fsm": {"open_close_state": "OPEN"},
                "teacher": {
                    "exact_signed_margin_available": True,
                    "primary_pad_containment_margin_mm": 1.0,
                    "inner_containment_margin_mm": 1.0,
                    "outer_containment_margin_mm": 1.0,
                    "aperture_margin_mm": 1.0,
                    "pad_surface_gap_mm": 1.0,
                    "orientation_margin_deg": 1.0,
                    "signed_readiness_margin": 0.1,
                },
                "outcome": {"contact": False, "bilateral": False, "stable": False},
            }
        )
    summary = writer.summary()
    writer.close()
    (tmp_path / "STAGE1A_VECTOR_REPORT.json").write_text(
        json.dumps({"accepted_transitions": 25, "sequence_dataset": summary}),
        encoding="utf-8",
    )

    result = _audit_module().audit(run_dir=tmp_path)

    assert result["SEQUENCE_DATASET_VALID"] == "YES"
    assert result["NEW_SEQUENCE_ROWS"] == 25
    assert result["NEW_EXACT_SIGNED_MARGIN_ROWS"] == 25
    assert result["sequence_continuity"]["camera_reuse_max_control_rows"] == {
        "head": 2,
        "right_wrist": 2,
    }


def test_v6_frame_store_keeps_gt_masks_on_the_same_wrist_frame_key(tmp_path: Path) -> None:
    writer = FsmSequenceDatasetWriter(
        output_dir=tmp_path,
        num_envs=10,
        frames_filename="V6.h5",
        rows_filename=None,
        v6_wrist_diagnostic=True,
    )
    frame = _frame(frame_id=7, timestamp_s=0.28)
    index = writer.append_v6_wrist_diagnostic(
        episode_id=3,
        frame=frame,
        diagnostic={
            "camera_world_pose_m_xyzw": np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=np.float64),
            "cube_pose_camera_optical_m_xyzw": np.asarray((0, 0, -0.4, 0, 0, 0, 1), dtype=np.float64),
            "intrinsics_3x3": np.eye(3, dtype=np.float64),
            "cube_semantic_mask": np.ones((192, 256), dtype=np.uint8),
            "cube_instance_mask": np.ones((192, 256), dtype=np.uint8),
            "cube_expected_silhouette_mask": np.ones((192, 256), dtype=np.uint8),
            "gt_cube_visible_pixel_count": 192 * 256,
            "gt_cube_projected_area_px": 192 * 256,
            "gt_occlusion_ratio": 0.0,
            "cube_mask_depth_valid_ratio": 1.0,
            "cube_mask_depth_valid_defined": True,
        },
    )
    assert index == 0
    assert writer.append_v6_wrist_diagnostic(
        episode_id=3,
        frame=frame,
        diagnostic={
            "camera_world_pose_m_xyzw": np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=np.float64),
            "cube_pose_camera_optical_m_xyzw": np.asarray((0, 0, -0.4, 0, 0, 0, 1), dtype=np.float64),
            "intrinsics_3x3": np.eye(3, dtype=np.float64),
            "cube_semantic_mask": np.ones((192, 256), dtype=np.uint8),
            "cube_instance_mask": np.ones((192, 256), dtype=np.uint8),
            "cube_expected_silhouette_mask": np.ones((192, 256), dtype=np.uint8),
            "gt_cube_visible_pixel_count": 192 * 256,
            "gt_cube_projected_area_px": 192 * 256,
            "gt_occlusion_ratio": 0.0,
            "cube_mask_depth_valid_ratio": 1.0,
            "cube_mask_depth_valid_defined": True,
        },
    ) == 0
    summary = writer.summary()
    writer.close()
    assert summary["v6_wrist_diagnostic"] is True
    assert summary["v6_wrist_diagnostic_frame_count"] == 1
    import h5py
    with h5py.File(tmp_path / "V6.h5", "r") as store:
        assert store.attrs["schema"] == V6_WRIST_DIAGNOSTIC_FRAME_STORE_SCHEMA
        assert int(store["right_wrist_v6_diagnostic/frame_store_index"][0]) == 0


def test_v6_uses_renderer_metadata_rgba_tokens_for_cube_masks() -> None:
    """RTX may expose AOV labels as RGBA tokens rather than scalar IDs."""

    info = {
        "idToLabels": {
            "(0, 0, 0, 0)": {"class": "BACKGROUND"},
            "(33, 243, 3, 255)": {"class": "cube"},
        }
    }
    tokens = _renderer_label_tokens(info, required_text="cube", source="SEMANTIC")
    assert tokens == ((33, 243, 3, 255),)
    value = torch.tensor(
        [
            [[0, 0, 0, 0], [33, 243, 3, 255]],
            [[33, 243, 3, 255], [0, 0, 0, 0]],
        ],
        dtype=torch.uint8,
    )
    mask = _mask_from_renderer_tokens(value, tokens=tokens, source="SEMANTIC")
    assert mask.tolist() == [[False, True], [True, False]]
    rgb_only = value[..., :3].clone()
    assert _mask_from_renderer_tokens(
        rgb_only, tokens=tokens, source="SEMANTIC"
    ).tolist() == [[False, True], [True, False]]
    packed = int.from_bytes(bytes((33, 243, 3, 255)), byteorder="little", signed=False)
    packed_value = torch.tensor(
        [[[0], [packed - (1 << 32)]], [[packed - (1 << 32)], [0]]],
        dtype=torch.int32,
    )
    assert _mask_from_renderer_tokens(
        packed_value, tokens=tokens, source="SEMANTIC"
    ).tolist() == [[False, True], [True, False]]


def test_25env_fsm_sequence_route_is_separate_from_frozen_10env_long_run() -> None:
    runtime = (ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts/run_g2_stage1a_vector_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "V3_1_LATERAL_OFF_FSM_SEQUENCE_HER_FORCE_15K" in runtime
    assert "num_envs != 25 or accepted_transition_target != 15000" in runtime
    assert "SAC_REPLAY_SOURCE\": \"CURRENT_RUNTIME_ROLLOUT_ONLY\"" in runtime
    assert "--num-envs\", type=int, choices=(1, 10, 25)" in runner


def test_sequence_teacher_receipt_is_blocked_at_sac_replay_boundary() -> None:
    runtime = (ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "def _replay_teacher_fields" in runtime
    assert "if not _uses_privileged_geometry_distillation(runtime_variant):" in runtime
    assert "privileged_close_ready_target=replay_teacher_target" in runtime


def test_sequence_metrics_preserve_missing_non_candidate_teacher_score() -> None:
    """A geometry receipt outside pre-CLOSE must not abort vector collection.

    The exact teacher contract intentionally uses ``None`` for its score when
    a row is not in the pre-CLOSE supervision domain.  The runtime metrics
    stream must serialize that absence as diagnostic NaN rather than applying
    ``float(None)`` and aborting a 25-env run.
    """

    runtime = (ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py").read_text(
        encoding="utf-8"
    )
    assert '"privileged_close_ready_score": _diagnostic_float(' in runtime
    assert 'gate, "privileged_close_ready_score"' in runtime
    assert '"student_close_ready_score": _diagnostic_float(' in runtime
