# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared, model-neutral RGB-D evidence contract for G2 collection paths.

The contract deliberately separates raw metric renderer evidence from the
existing deployable 0--2 m preprocessing.  Nothing in this module is an actor
input definition and it has no Isaac/PhysX dependency.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


RGBD_EVIDENCE_SCHEMA = "g2_head_wrist_rgbd_evidence_v1"
RGBD_CAMERA_NAMES = ("head", "right_wrist")
DEPTH_RAW_UNIT = "meter"
DEPTH_TO_METER_SCALE = 1.0
EXTRINSIC_REFERENCE_FRAME = "robot_root"
CAMERA_TIMESTAMP_SOURCE = "ISAAC_CAMERA_TIMESTAMP_LAST_UPDATE"


class RGBDLoggingContractError(ValueError):
    pass


def raw_depth_and_validity(
    source_depth_m: Any, source_valid: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return un-clamped float32 metric depth and its source validity mask.

    Positive infinity is retained as raw renderer evidence and marked invalid.
    NaN, negative infinity, and finite negative depth indicate corruption.
    """

    depth = np.asarray(source_depth_m, dtype=np.float32)
    if depth.ndim != 3 or depth.shape[-1] != 1 or min(depth.shape[:2]) <= 0:
        raise RGBDLoggingContractError("raw depth must be float32[H,W,1]")
    corrupt = np.isnan(depth) | np.isneginf(depth) | (np.isfinite(depth) & (depth < 0.0))
    if np.any(corrupt):
        raise RGBDLoggingContractError("raw depth contains NaN/-inf/negative values")
    derived = np.isfinite(depth) & (depth >= 0.0)
    if source_valid is not None:
        supplied = np.asarray(source_valid)
        if supplied.shape != depth.shape:
            raise RGBDLoggingContractError("raw depth/source validity shapes differ")
        if supplied.dtype != np.bool_ and not np.all(np.isin(supplied, (0, 1))):
            raise RGBDLoggingContractError("source validity must be boolean/binary")
        derived &= supplied.astype(np.bool_, copy=False)
    return depth.copy(), derived.astype(np.bool_, copy=False)


def validate_camera_calibration_metadata(
    metadata: Mapping[str, Any], *, camera_name: str
) -> dict[str, Any]:
    if camera_name not in RGBD_CAMERA_NAMES:
        raise RGBDLoggingContractError(f"unsupported camera: {camera_name}")
    required = {
        "camera_name",
        "source_scene_key",
        "source_frame_id",
        "intrinsics_3x3",
        "image_width",
        "image_height",
        "extrinsic_reference_frame",
        "pose_convention",
        "timestamp_source",
    }
    if set(metadata) != required:
        raise RGBDLoggingContractError(
            f"{camera_name} calibration keys mismatch: {sorted(set(metadata) ^ required)}"
        )
    if metadata["camera_name"] != camera_name:
        raise RGBDLoggingContractError("camera calibration identity mismatch")
    if metadata["source_scene_key"] != f"{camera_name}_camera":
        raise RGBDLoggingContractError("camera scene key is not source-owned")
    if not str(metadata["source_frame_id"]):
        raise RGBDLoggingContractError("camera source frame ID is missing")
    intrinsics = np.asarray(metadata["intrinsics_3x3"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise RGBDLoggingContractError("camera intrinsics must be finite 3x3")
    if int(metadata["image_width"]) != 256 or int(metadata["image_height"]) != 192:
        raise RGBDLoggingContractError("camera image size must be 256x192")
    if metadata["extrinsic_reference_frame"] != EXTRINSIC_REFERENCE_FRAME:
        raise RGBDLoggingContractError("camera extrinsic reference must be robot_root")
    if metadata["pose_convention"] != "position_m+quaternion_xyzw":
        raise RGBDLoggingContractError("camera pose convention mismatch")
    if metadata["timestamp_source"] != CAMERA_TIMESTAMP_SOURCE:
        raise RGBDLoggingContractError("camera timestamp source mismatch")
    result = dict(metadata)
    result["intrinsics_3x3"] = intrinsics.tolist()
    result["image_width"] = 256
    result["image_height"] = 192
    return result


def validate_camera_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one dual-camera observation without changing actor inputs."""

    expected_top = {"schema", "depth_raw_unit", "depth_to_meter_scale", *RGBD_CAMERA_NAMES}
    if set(evidence) != expected_top:
        raise RGBDLoggingContractError("dual-camera evidence keys mismatch")
    if evidence["schema"] != RGBD_EVIDENCE_SCHEMA:
        raise RGBDLoggingContractError("RGB-D evidence schema mismatch")
    if evidence["depth_raw_unit"] != DEPTH_RAW_UNIT:
        raise RGBDLoggingContractError("raw depth unit must be meter")
    if float(evidence["depth_to_meter_scale"]) != DEPTH_TO_METER_SCALE:
        raise RGBDLoggingContractError("depth-to-meter scale must be exactly one")
    result: dict[str, Any] = {
        "schema": RGBD_EVIDENCE_SCHEMA,
        "depth_raw_unit": DEPTH_RAW_UNIT,
        "depth_to_meter_scale": DEPTH_TO_METER_SCALE,
    }
    for camera_name in RGBD_CAMERA_NAMES:
        camera = dict(evidence[camera_name])
        required = {
            "rgb",
            "depth_raw_m",
            "depth_source_valid",
            "timestamp_s",
            "frame_id",
            "associated_frame_index",
            "pose_root_m_xyzw",
        }
        if set(camera) != required:
            raise RGBDLoggingContractError(f"{camera_name} evidence keys mismatch")
        rgb = np.asarray(camera["rgb"], dtype=np.uint8)
        if rgb.shape != (192, 256, 3):
            raise RGBDLoggingContractError(f"{camera_name} RGB shape mismatch")
        raw, valid = raw_depth_and_validity(
            camera["depth_raw_m"], camera["depth_source_valid"]
        )
        timestamp_s = float(camera["timestamp_s"])
        frame_id = int(camera["frame_id"])
        associated = int(camera["associated_frame_index"])
        if not np.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise RGBDLoggingContractError(f"{camera_name} timestamp invalid")
        if frame_id < 0 or associated != frame_id:
            raise RGBDLoggingContractError(f"{camera_name} frame association invalid")
        pose = np.asarray(camera["pose_root_m_xyzw"], dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise RGBDLoggingContractError(f"{camera_name} pose invalid")
        if not np.isclose(np.linalg.norm(pose[3:]), 1.0, rtol=0.0, atol=1.0e-3):
            raise RGBDLoggingContractError(f"{camera_name} quaternion is not unit XYZW")
        result[camera_name] = {
            "rgb": rgb.copy(),
            "depth_raw_m": raw,
            "depth_source_valid": valid,
            "timestamp_s": timestamp_s,
            "frame_id": frame_id,
            "associated_frame_index": associated,
            "pose_root_m_xyzw": pose,
        }
    return result


def flatten_calibration_metadata(
    calibrations: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(calibrations) != set(RGBD_CAMERA_NAMES):
        raise RGBDLoggingContractError("both Head and Right Wrist calibration are required")
    flattened: dict[str, Any] = {
        "rgbd_evidence_schema": RGBD_EVIDENCE_SCHEMA,
        "depth_raw_unit": DEPTH_RAW_UNIT,
        "depth_to_meter_scale": DEPTH_TO_METER_SCALE,
    }
    for camera_name in RGBD_CAMERA_NAMES:
        item = validate_camera_calibration_metadata(
            calibrations[camera_name], camera_name=camera_name
        )
        prefix = f"{camera_name}_camera_"
        for key, value in item.items():
            flattened[prefix + key] = value
    return flattened


__all__ = [
    "CAMERA_TIMESTAMP_SOURCE",
    "DEPTH_RAW_UNIT",
    "DEPTH_TO_METER_SCALE",
    "EXTRINSIC_REFERENCE_FRAME",
    "RGBD_CAMERA_NAMES",
    "RGBD_EVIDENCE_SCHEMA",
    "RGBDLoggingContractError",
    "flatten_calibration_metadata",
    "raw_depth_and_validity",
    "validate_camera_calibration_metadata",
    "validate_camera_evidence",
]
