# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Durable, causal sequence capture for the deterministic Stage-1A FSM.

The residual-SAC replay remains deliberately small and transition-shaped.  A
future GRU, however, needs the complete 50-Hz causal history and both RGB-D
views.  This writer keeps those two data authorities physically separate:

* ``REPLAY_TRANSITIONS.jsonl`` is untouched and remains the *only* SAC replay
  source;
* this module writes an HDF5 RGB-D frame store plus a control-row JSONL file
  for later, offline-only GRU work.

Frames are deduplicated only for the same (sensor, env, episode, real camera
frame id, sensor timestamp).  Therefore a 25-Hz image may be referenced by
two adjacent 50-Hz control rows, but it can never be substituted across an
environment or an episode boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .stage1a_vector_contract import PerEnvCameraFrame


FSM_SEQUENCE_SCHEMA = "g2_stage1a_deterministic_fsm_gru_sequence_v2"
FSM_FRAME_STORE_SCHEMA = "g2_stage1a_deterministic_fsm_rgbd_frame_store_v1"
V6_WRIST_DIAGNOSTIC_FRAME_STORE_SCHEMA = (
    "g2_stage1a_timestamp_aligned_wrist_rgbd_camera_mask_store_v6"
)


class Stage1AFsmSequenceDatasetError(ValueError):
    """Raised before an invalid causal sequence row reaches durable storage."""


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalars/arrays without silently changing semantic values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


class FsmSequenceDatasetWriter:
    """Append-only RGB-D frame and 50-Hz control-row writer.

    ``h5py`` is deliberately imported only at construction time.  This module
    is imported by static tests too, whereas the Isaac distribution is the
    authority that ships the HDF5 dependency used by the live writer.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        num_envs: int,
        frames_filename: str = "FSM_GRU_SEQUENCE_RGBD.h5",
        rows_filename: str | None = "FSM_GRU_SEQUENCE_ROWS.jsonl",
        v6_wrist_diagnostic: bool = False,
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_NUM_ENVS_INVALID")
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - live dependency check
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_H5PY_UNAVAILABLE") from error
        self._h5py = h5py
        self.output_dir = Path(output_dir)
        self.num_envs = num_envs
        if not isinstance(v6_wrist_diagnostic, bool):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_V6_FLAG_INVALID")
        self.v6_wrist_diagnostic = v6_wrist_diagnostic
        if not isinstance(frames_filename, str) or not frames_filename:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_FRAMES_FILENAME_INVALID")
        if rows_filename is not None and (not isinstance(rows_filename, str) or not rows_filename):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_ROWS_FILENAME_INVALID")
        self.frames_path = self.output_dir / frames_filename
        self.rows_path = self.output_dir / rows_filename if rows_filename is not None else None
        if self.frames_path.exists() or (
            self.rows_path is not None and self.rows_path.exists()
        ):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_OUTPUT_REFUSES_OVERWRITE")
        self._file = h5py.File(self.frames_path, "x")
        self._file.attrs["schema"] = (
            V6_WRIST_DIAGNOSTIC_FRAME_STORE_SCHEMA
            if self.v6_wrist_diagnostic
            else FSM_FRAME_STORE_SCHEMA
        )
        self._file.attrs["control_hz"] = 50
        self._file.attrs["rgbd_hz"] = 25
        self._file.attrs["num_envs"] = num_envs
        self._datasets: dict[str, Mapping[str, Any]] = {}
        self._frame_index: dict[tuple[str, int, int, int, float], int] = {}
        self._rows = (
            self.rows_path.open("x", encoding="utf-8")
            if self.rows_path is not None else None
        )
        self.sequence_row_count = 0
        self.exact_signed_margin_row_count = 0
        self._frame_count = {"head": 0, "right_wrist": 0}
        self._v6_datasets: Mapping[str, Any] | None = None

    def _ensure_sensor(self, sensor: str, frame: PerEnvCameraFrame) -> Mapping[str, Any]:
        if sensor not in ("head", "right_wrist"):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_SENSOR_UNKNOWN")
        existing = self._datasets.get(sensor)
        if existing is not None:
            return existing
        rgb_shape = tuple(frame.rgb.shape)
        depth_shape = tuple(frame.depth_m.shape)
        if rgb_shape != (192, 256, 3) or depth_shape != (192, 256, 1):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_RGBD_SHAPE_INVALID")
        group = self._file.create_group(sensor)
        group.attrs["sensor_binding"] = sensor
        group.attrs["rgb_dtype"] = "uint8"
        group.attrs["depth_unit"] = "m"
        group.attrs["depth_dtype"] = "float32"
        datasets: dict[str, Any] = {
            "rgb": group.create_dataset(
                "rgb", shape=(0, *rgb_shape), maxshape=(None, *rgb_shape),
                dtype=np.uint8, chunks=(1, *rgb_shape), compression="gzip", compression_opts=1,
            ),
            "depth_m": group.create_dataset(
                "depth_m", shape=(0, *depth_shape), maxshape=(None, *depth_shape),
                dtype=np.float32, chunks=(1, *depth_shape), compression="gzip", compression_opts=1,
            ),
            "depth_valid": group.create_dataset(
                "depth_valid", shape=(0, *depth_shape), maxshape=(None, *depth_shape),
                dtype=np.uint8, chunks=(1, *depth_shape), compression="gzip", compression_opts=1,
            ),
            "env_id": group.create_dataset("env_id", shape=(0,), maxshape=(None,), dtype=np.int32),
            "episode_id": group.create_dataset("episode_id", shape=(0,), maxshape=(None,), dtype=np.int32),
            "sensor_frame_id": group.create_dataset("sensor_frame_id", shape=(0,), maxshape=(None,), dtype=np.int64),
            "sensor_timestamp_s": group.create_dataset("sensor_timestamp_s", shape=(0,), maxshape=(None,), dtype=np.float64),
        }
        self._datasets[sensor] = datasets
        return datasets

    def append_frame(
        self,
        *,
        sensor: str,
        episode_id: int,
        frame: PerEnvCameraFrame,
    ) -> int:
        if type(episode_id) is not int or episode_id < 0:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_EPISODE_ID_INVALID")
        if frame.env_id >= self.num_envs:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_FRAME_ENV_OUTSIDE_BATCH")
        if not np.isfinite(frame.depth_m).all() or not np.isfinite(frame.rgb).all():
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_FRAME_NONFINITE")
        key = (sensor, frame.env_id, episode_id, frame.frame_id, float(frame.timestamp_s))
        prior = self._frame_index.get(key)
        if prior is not None:
            return prior
        datasets = self._ensure_sensor(sensor, frame)
        index = int(datasets["rgb"].shape[0])
        for dataset in datasets.values():
            dataset.resize((index + 1, *dataset.shape[1:]))
        datasets["rgb"][index] = frame.rgb
        datasets["depth_m"][index] = frame.depth_m
        datasets["depth_valid"][index] = frame.depth_valid.astype(np.uint8, copy=False)
        datasets["env_id"][index] = frame.env_id
        datasets["episode_id"][index] = episode_id
        datasets["sensor_frame_id"][index] = frame.frame_id
        datasets["sensor_timestamp_s"][index] = frame.timestamp_s
        self._frame_index[key] = index
        self._frame_count[sensor] += 1
        return index

    def append_control_row(self, payload: Mapping[str, Any]) -> None:
        if self._rows is None:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_CONTROL_ROW_WRITER_DISABLED")
        required = (
            "schema", "episode_id", "env_id", "control_step", "control_timestamp_s",
            "head_frame_store_index", "right_wrist_frame_store_index",
            "head_camera_timestamp_s", "right_wrist_camera_timestamp_s",
            "student_observation", "previous_action_4d_metric_root_m",
            "sac_residual_xyz_action_m", "fsm", "outcome", "teacher",
            "student_privileged_input_count", "gru_runtime_authority",
            "privileged_runtime_authority",
        )
        if any(name not in payload for name in required):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_ROW_INCOMPLETE")
        if payload["schema"] != FSM_SEQUENCE_SCHEMA:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_ROW_SCHEMA_INVALID")
        if payload["student_privileged_input_count"] != 0:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_PRIVILEGED_STUDENT_INPUT_FORBIDDEN")
        if payload["gru_runtime_authority"] is not False or payload["privileged_runtime_authority"] is not False:
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_RUNTIME_AUTHORITY_INVALID")
        advisory = payload.get("student_advisory")
        if advisory is not None:
            if not isinstance(advisory, Mapping):
                raise Stage1AFsmSequenceDatasetError("SEQUENCE_ADVISORY_RECEIPT_INVALID")
            if advisory.get("authority") == "FROZEN_HIGH_CONFIDENCE_ADVISORY_FSM_DEFER_TELEMETRY_ONLY":
                score = advisory.get("score")
                threshold = advisory.get("high_ready_threshold")
                if not (
                    isinstance(score, (int, float))
                    and math.isfinite(float(score))
                    and 0.0 <= float(score) <= 1.0
                    and isinstance(threshold, (int, float))
                    and math.isfinite(float(threshold))
                    and 0.0 < float(threshold) < 1.0
                    and advisory.get("advice") in ("READY_ADVISORY", "DEFER_TO_FSM")
                ):
                    raise Stage1AFsmSequenceDatasetError(
                        "SEQUENCE_ADVISORY_FROZEN_RECEIPT_INVALID"
                    )
        if not math.isfinite(float(payload["control_timestamp_s"])):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_CONTROL_TIMESTAMP_INVALID")
        teacher = payload["teacher"]
        if not isinstance(teacher, Mapping):
            raise Stage1AFsmSequenceDatasetError("SEQUENCE_TEACHER_RECEIPT_INVALID")
        exact = bool(teacher.get("exact_signed_margin_available", False))
        if exact:
            required_margin = (
                "primary_pad_containment_margin_mm", "inner_containment_margin_mm",
                "outer_containment_margin_mm", "aperture_margin_mm",
                "pad_surface_gap_mm", "orientation_margin_deg", "signed_readiness_margin",
            )
            if not all(
                isinstance(teacher.get(name), (int, float))
                and math.isfinite(float(teacher[name]))
                for name in required_margin
            ):
                raise Stage1AFsmSequenceDatasetError("SEQUENCE_SIGNED_MARGIN_INCOMPLETE")
            self.exact_signed_margin_row_count += 1
        self._rows.write(json.dumps(_json_ready(dict(payload)), sort_keys=True) + "\n")
        self.sequence_row_count += 1

    def _ensure_v6_wrist_diagnostic(self) -> Mapping[str, Any]:
        """Create V6 teacher-only datasets beside the immutable RGB-D frames."""

        if not self.v6_wrist_diagnostic:
            raise Stage1AFsmSequenceDatasetError("V6_DIAGNOSTIC_WRITER_DISABLED")
        if self._v6_datasets is not None:
            return self._v6_datasets
        group = self._file.create_group("right_wrist_v6_diagnostic")
        group.attrs["schema"] = V6_WRIST_DIAGNOSTIC_FRAME_STORE_SCHEMA
        group.attrs["camera_binding"] = "right_wrist_camera"
        group.attrs["camera_pose_authority"] = (
            "URDF_LINK_PLUS_CHECKED_IN_G2_USD_OPTICAL_EXTRINSIC"
        )
        group.attrs["camera_optical_convention"] = (
            "USD_OPENGL_OPTICAL_NEGATIVE_Z_FORWARD"
        )
        group.attrs["student_privileged_input_count"] = 0
        datasets: dict[str, Any] = {
            "frame_store_index": group.create_dataset(
                "frame_store_index", shape=(0,), maxshape=(None,), dtype=np.int64
            ),
            "camera_world_pose_m_xyzw": group.create_dataset(
                "camera_world_pose_m_xyzw", shape=(0, 7), maxshape=(None, 7), dtype=np.float64
            ),
            "cube_pose_camera_optical_m_xyzw": group.create_dataset(
                "cube_pose_camera_optical_m_xyzw", shape=(0, 7), maxshape=(None, 7), dtype=np.float64
            ),
            "intrinsics_3x3": group.create_dataset(
                "intrinsics_3x3", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype=np.float64
            ),
            "cube_semantic_mask": group.create_dataset(
                "cube_semantic_mask", shape=(0, 192, 256), maxshape=(None, 192, 256),
                dtype=np.uint8, chunks=(1, 192, 256), compression="gzip", compression_opts=1,
            ),
            "cube_instance_mask": group.create_dataset(
                "cube_instance_mask", shape=(0, 192, 256), maxshape=(None, 192, 256),
                dtype=np.uint8, chunks=(1, 192, 256), compression="gzip", compression_opts=1,
            ),
            "cube_expected_silhouette_mask": group.create_dataset(
                "cube_expected_silhouette_mask", shape=(0, 192, 256), maxshape=(None, 192, 256),
                dtype=np.uint8, chunks=(1, 192, 256), compression="gzip", compression_opts=1,
            ),
            "gt_cube_visible_pixel_count": group.create_dataset(
                "gt_cube_visible_pixel_count", shape=(0,), maxshape=(None,), dtype=np.int32
            ),
            "gt_cube_projected_area_px": group.create_dataset(
                "gt_cube_projected_area_px", shape=(0,), maxshape=(None,), dtype=np.int32
            ),
            "gt_occlusion_ratio": group.create_dataset(
                "gt_occlusion_ratio", shape=(0,), maxshape=(None,), dtype=np.float64
            ),
            "cube_mask_depth_valid_ratio": group.create_dataset(
                "cube_mask_depth_valid_ratio", shape=(0,), maxshape=(None,), dtype=np.float64
            ),
            "cube_mask_depth_valid_defined": group.create_dataset(
                "cube_mask_depth_valid_defined", shape=(0,), maxshape=(None,), dtype=np.uint8
            ),
        }
        self._v6_datasets = datasets
        return datasets

    def append_v6_wrist_diagnostic(
        self,
        *,
        episode_id: int,
        frame: PerEnvCameraFrame,
        diagnostic: Mapping[str, Any],
    ) -> int:
        """Append one exact V6 diagnostic alongside its deduplicated frame.

        The raw RGB, metric depth, and depth-valid mask remain in the standard
        ``right_wrist`` group.  This group stores only derived GT visibility
        evidence, with the same immutable frame index as its foreign key.
        """

        required = (
            "camera_world_pose_m_xyzw", "cube_pose_camera_optical_m_xyzw",
            "intrinsics_3x3", "cube_semantic_mask", "cube_instance_mask",
            "cube_expected_silhouette_mask", "gt_cube_visible_pixel_count",
            "gt_cube_projected_area_px", "gt_occlusion_ratio",
            "cube_mask_depth_valid_ratio", "cube_mask_depth_valid_defined",
        )
        if any(name not in diagnostic for name in required):
            raise Stage1AFsmSequenceDatasetError("V6_DIAGNOSTIC_RECEIPT_INCOMPLETE")
        index = self.append_frame(sensor="right_wrist", episode_id=episode_id, frame=frame)
        datasets = self._ensure_v6_wrist_diagnostic()
        existing = int(datasets["frame_store_index"].shape[0])
        if existing > index:
            # One camera frame may be referenced by two 50-Hz rows.  The V6
            # receipt is attached exactly once at its actual 25-Hz capture.
            if int(datasets["frame_store_index"][index]) != index:
                raise Stage1AFsmSequenceDatasetError("V6_FRAME_STORE_INDEX_MISMATCH")
            return index
        if existing != index:
            raise Stage1AFsmSequenceDatasetError("V6_FRAME_DIAGNOSTIC_DENSE_INDEX_REQUIRED")

        expected_shapes = {
            "camera_world_pose_m_xyzw": (7,),
            "cube_pose_camera_optical_m_xyzw": (7,),
            "intrinsics_3x3": (3, 3),
            "cube_semantic_mask": (192, 256),
            "cube_instance_mask": (192, 256),
            "cube_expected_silhouette_mask": (192, 256),
        }
        converted: dict[str, np.ndarray] = {}
        for name, shape in expected_shapes.items():
            array = np.asarray(diagnostic[name])
            if array.shape != shape or not np.isfinite(array).all():
                raise Stage1AFsmSequenceDatasetError(f"V6_DIAGNOSTIC_{name.upper()}_INVALID")
            converted[name] = array
        for dataset in datasets.values():
            dataset.resize((index + 1, *dataset.shape[1:]))
        datasets["frame_store_index"][index] = index
        datasets["camera_world_pose_m_xyzw"][index] = converted["camera_world_pose_m_xyzw"]
        datasets["cube_pose_camera_optical_m_xyzw"][index] = converted["cube_pose_camera_optical_m_xyzw"]
        datasets["intrinsics_3x3"][index] = converted["intrinsics_3x3"]
        for name in (
            "cube_semantic_mask", "cube_instance_mask", "cube_expected_silhouette_mask"
        ):
            datasets[name][index] = converted[name].astype(np.uint8, copy=False)
        visible = int(diagnostic["gt_cube_visible_pixel_count"])
        area = int(diagnostic["gt_cube_projected_area_px"])
        occlusion = float(diagnostic["gt_occlusion_ratio"])
        depth_ratio = float(diagnostic["cube_mask_depth_valid_ratio"])
        if (
            visible < 0 or area <= 0 or visible > 192 * 256
            or not math.isfinite(occlusion) or not 0.0 <= occlusion <= 1.0
            or not math.isfinite(depth_ratio) or not 0.0 <= depth_ratio <= 1.0
        ):
            raise Stage1AFsmSequenceDatasetError("V6_DIAGNOSTIC_SCALAR_INVALID")
        datasets["gt_cube_visible_pixel_count"][index] = visible
        datasets["gt_cube_projected_area_px"][index] = area
        datasets["gt_occlusion_ratio"][index] = occlusion
        datasets["cube_mask_depth_valid_ratio"][index] = depth_ratio
        datasets["cube_mask_depth_valid_defined"][index] = int(
            bool(diagnostic["cube_mask_depth_valid_defined"])
        )
        return index

    def flush(self) -> None:
        if self._rows is not None:
            self._rows.flush()
        self._file.flush()

    def close(self) -> None:
        if self._rows is not None and not self._rows.closed:
            self.flush()
            self._rows.close()
        if getattr(self._file, "id", None) is not None and self._file.id.valid:
            self._file.close()

    def summary(self) -> dict[str, Any]:
        return {
            "schema": FSM_SEQUENCE_SCHEMA,
            "rows_path": str(self.rows_path) if self.rows_path is not None else None,
            "frames_path": str(self.frames_path),
            "sequence_row_count": int(self.sequence_row_count),
            "exact_signed_margin_row_count": int(self.exact_signed_margin_row_count),
            "head_frame_count": int(self._frame_count["head"]),
            "right_wrist_frame_count": int(self._frame_count["right_wrist"]),
            "control_hz": 50,
            "rgbd_hz": 25,
            "student_privileged_input_count": 0,
            "gru_runtime_authority": False,
            "privileged_runtime_authority": False,
            "v6_wrist_diagnostic": self.v6_wrist_diagnostic,
            "v6_wrist_diagnostic_frame_count": (
                int(self._v6_datasets["frame_store_index"].shape[0])
                if self._v6_datasets is not None else 0
            ),
        }


__all__ = [
    "FSM_FRAME_STORE_SCHEMA",
    "FSM_SEQUENCE_SCHEMA",
    "V6_WRIST_DIAGNOSTIC_FRAME_STORE_SCHEMA",
    "FsmSequenceDatasetWriter",
    "Stage1AFsmSequenceDatasetError",
]
