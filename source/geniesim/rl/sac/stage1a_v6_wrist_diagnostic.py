# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""V6 teacher-side Wrist RGB-D visibility diagnostics.

The V5 representation audit could prove a family-specific depth-topology
shift, but it could not say whether the shift originated in the moving wrist
mount, the RTX depth/mask path, or real cube occlusion.  This module records
those facts from the *existing* ``right_wrist_camera`` binding only.  Nothing
here contributes to a SAC observation, a GRU input, a reward, or a CLOSE
decision.

Camera coordinates follow the checked-in USD/OpenGL optical convention used by
``g2_camera_visibility``: the forward direction is ``-Z`` and metric depth is
therefore ``-z``.  The convention is explicit in every receipt rather than
being inferred by an offline consumer.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from geniesim.rl.sac.stage1a_vector_contract import PerEnvCameraFrame


V6_WRIST_DIAGNOSTIC_SCHEMA = "g2_stage1a_wrist_rgbd_camera_mask_diagnostic_v6"
V6_CAMERA_OPTICAL_CONVENTION = "USD_OPENGL_OPTICAL_NEGATIVE_Z_FORWARD"


class Stage1AV6WristDiagnosticError(ValueError):
    """Raised instead of fabricating an unavailable V6 GT receipt."""


@dataclass(frozen=True)
class V6WristDiagnosticFrame:
    """One exact 25-Hz right-wrist frame's diagnostic-only evidence."""

    camera_world_pose_m_xyzw: np.ndarray
    cube_pose_camera_optical_m_xyzw: np.ndarray
    intrinsics_3x3: np.ndarray
    cube_semantic_mask: np.ndarray
    cube_instance_mask: np.ndarray
    cube_expected_silhouette_mask: np.ndarray
    gt_cube_visible_pixel_count: int
    gt_cube_projected_area_px: int
    gt_occlusion_ratio: float
    cube_mask_depth_valid_ratio: float
    cube_mask_depth_valid_defined: bool
    # ``idToLabels`` may use scalar IDs or renderer-native RGBA tokens.
    semantic_label_ids: tuple[int | tuple[int, ...], ...]
    instance_label_ids: tuple[int | tuple[int, ...], ...]
    semantic_info_sha256: str
    instance_info_sha256: str
    camera_optical_convention: str = V6_CAMERA_OPTICAL_CONVENTION


def _tensor(value: Any) -> torch.Tensor:
    tensor = getattr(value, "torch", value)
    if not isinstance(tensor, torch.Tensor):
        raise Stage1AV6WristDiagnosticError("V6_RUNTIME_TENSOR_UNAVAILABLE")
    return tensor


def _normalized_quaternion_xyzw(value: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise Stage1AV6WristDiagnosticError(f"V6_{label}_QUATERNION_INVALID")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise Stage1AV6WristDiagnosticError(f"V6_{label}_QUATERNION_DEGENERATE")
    result = result / norm
    # A sign-normalized durable receipt makes exact metadata comparisons
    # independent of the equivalent q/-q representation.
    return -result if result[3] < 0.0 else result


def _rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalized_quaternion_xyzw(quaternion, label="CUBE")
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _quaternion_from_rotation_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise Stage1AV6WristDiagnosticError("V6_ROTATION_INVALID")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=2.0e-5):
        raise Stage1AV6WristDiagnosticError("V6_ROTATION_NOT_ORTHONORMAL")
    if float(np.linalg.det(matrix)) <= 0.0:
        raise Stage1AV6WristDiagnosticError("V6_ROTATION_NOT_PROPER")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        value = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            ((matrix[2, 1] - matrix[1, 2]) / value,
             (matrix[0, 2] - matrix[2, 0]) / value,
             (matrix[1, 0] - matrix[0, 1]) / value,
             0.25 * value),
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            value = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (0.25 * value,
                 (matrix[0, 1] + matrix[1, 0]) / value,
                 (matrix[0, 2] + matrix[2, 0]) / value,
                 (matrix[2, 1] - matrix[1, 2]) / value),
                dtype=np.float64,
            )
        elif index == 1:
            value = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                ((matrix[0, 1] + matrix[1, 0]) / value,
                 0.25 * value,
                 (matrix[1, 2] + matrix[2, 1]) / value,
                 (matrix[0, 2] - matrix[2, 0]) / value),
                dtype=np.float64,
            )
        else:
            value = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                ((matrix[0, 2] + matrix[2, 0]) / value,
                 (matrix[1, 2] + matrix[2, 1]) / value,
                 0.25 * value,
                 (matrix[1, 0] - matrix[0, 1]) / value),
                dtype=np.float64,
            )
    return _normalized_quaternion_xyzw(quaternion, label="ROTATION")


def _finite_pose(value: np.ndarray, *, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise Stage1AV6WristDiagnosticError(f"V6_{label}_POSE_INVALID")
    pose = pose.copy()
    pose[3:] = _normalized_quaternion_xyzw(pose[3:], label=label)
    return pose


def _json_sha256(value: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _renderer_label_tokens(
    info: Any, *, required_text: str, source: str
) -> tuple[int | tuple[int, ...], ...]:
    """Resolve actual renderer label tokens from metadata, never heuristics.

    The object matching text comes from the explicitly bound scene prim
    (``Object``) or the diagnostic-only ``class:cube`` semantic tag.  We use
    metadata returned by Replicator, rather than guessing IDs from raw masks.

    Isaac's renderer can emit either integer IDs or non-colorized-looking
    RGBA label tokens (the latter are still the renderer's authoritative
    semantic IDs, as documented by its ``idToLabels`` mapping).  The token is
    matched exactly against the corresponding AOV value; it is never inferred
    from the RGB image or cube GT.
    """

    if not isinstance(info, Mapping):
        raise Stage1AV6WristDiagnosticError(f"V6_{source}_INFO_MISSING")
    mapping = info.get("idToLabels")
    if not isinstance(mapping, Mapping):
        raise Stage1AV6WristDiagnosticError(f"V6_{source}_ID_TO_LABELS_MISSING")
    text = required_text.lower()
    tokens: list[int | tuple[int, ...]] = []
    for raw_id, labels in mapping.items():
        label_text = json.dumps(labels, sort_keys=True, default=str).lower()
        if text not in label_text:
            continue
        try:
            token: int | tuple[int, ...] = int(raw_id)
        except (TypeError, ValueError):
            try:
                decoded = ast.literal_eval(str(raw_id))
            except (SyntaxError, ValueError):
                continue
            if (
                not isinstance(decoded, tuple)
                or not decoded
                or any(type(component) is not int or component < 0 or component > 255 for component in decoded)
            ):
                continue
            token = tuple(decoded)
        tokens.append(token)
    result = tuple(sorted(set(tokens), key=lambda token: (isinstance(token, tuple), repr(token))))
    if not result:
        # This receipt is intentionally metadata-only.  It captures the
        # renderer's own label mapping at the point of failure, so a missing
        # semantic tag cannot be silently converted into a guessed cube mask.
        # Keep it bounded because this error is persisted in the launcher
        # receipt as well as stderr.
        receipt = {
            str(raw_id): labels
            for raw_id, labels in sorted(mapping.items(), key=lambda item: str(item[0]))[:64]
        }
        encoded = json.dumps(receipt, sort_keys=True, default=str, separators=(",", ":"))
        raise Stage1AV6WristDiagnosticError(
            f"V6_{source}_CUBE_LABEL_UNRESOLVED:ID_TO_LABELS={encoded[:4096]}"
        )
    return result


def _mask_from_renderer_tokens(
    value: torch.Tensor,
    *,
    tokens: Sequence[int | tuple[int, ...]],
    source: str,
) -> np.ndarray:
    tensor = _tensor(value)
    if tensor.ndim != 3 or tensor.shape[-1] not in (1, 3, 4):
        raise Stage1AV6WristDiagnosticError(f"V6_{source}_MASK_SHAPE_INVALID")
    array = tensor.detach().to("cpu").numpy()
    if not np.issubdtype(array.dtype, np.integer):
        raise Stage1AV6WristDiagnosticError(f"V6_{source}_MASK_DTYPE_INVALID")
    integer_tokens = tuple(token for token in tokens if isinstance(token, int))
    vector_tokens = tuple(token for token in tokens if isinstance(token, tuple))
    result = np.zeros(array.shape[:2], dtype=bool)
    if integer_tokens and array.shape[-1] == 1:
        result |= np.isin(
            array[..., 0].astype(np.int64, copy=False),
            np.asarray(integer_tokens, dtype=np.int64),
        )
    if vector_tokens:
        for token in vector_tokens:
            comparable = token
            if len(token) == 4 and array.shape[-1] == 3:
                # RTX records semantic keys as RGBA but this Camera AOV
                # exposes the corresponding RGB channels.  Dropping only the
                # documented alpha component retains an exact renderer-label
                # comparison; it is not an RGB-image colour heuristic.
                comparable = token[:3]
            if len(comparable) != array.shape[-1]:
                continue
            result |= np.all(
                array.astype(np.int64, copy=False) == np.asarray(comparable, dtype=np.int64),
                axis=-1,
            )
    if vector_tokens and array.shape[-1] == 1:
        # A different Isaac RTX AOV transport encodes the same ``idToLabels``
        # RGBA token into one uint32/int32 channel.  Compare both byte orders
        # emitted by Kit transport, including their signed int32 views.  This
        # is still an exact token equality from renderer metadata, not a
        # colour-based image segmentation fallback.
        scalar = array[..., 0].astype(np.int64, copy=False)
        for token in vector_tokens:
            if len(token) != 4:
                continue
            raw = bytes(token)
            values: set[int] = set()
            for byteorder in ("little", "big"):
                unsigned = int.from_bytes(raw, byteorder=byteorder, signed=False)
                values.add(unsigned)
                values.add(unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned)
            for packed in values:
                result |= scalar == packed
    if not result.any() and not (
        (integer_tokens and array.shape[-1] == 1)
        or (vector_tokens and array.shape[-1] == 1)
        or any(
            len(token) == array.shape[-1]
            or (len(token) == 4 and array.shape[-1] == 3)
            for token in vector_tokens
        )
    ):
        raise Stage1AV6WristDiagnosticError(f"V6_{source}_TOKEN_CHANNEL_PARITY_INVALID")
    return result


def _expected_cube_silhouette_mask(
    *,
    cube_pose_camera_optical_m_xyzw: np.ndarray,
    cube_size_m: Sequence[float],
    intrinsics_3x3: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize the unoccluded cuboid silhouette using camera rays.

    This is a geometric GT projection, not an image-colour proxy.  A ray-box
    intersection makes the denominator robust to cube orientation and is the
    reference for the instance-mask-derived occlusion ratio.
    """

    pose = _finite_pose(cube_pose_camera_optical_m_xyzw, label="CUBE_CAMERA")
    half = np.asarray(cube_size_m, dtype=np.float64) * 0.5
    intrinsic = np.asarray(intrinsics_3x3, dtype=np.float64)
    if (
        half.shape != (3,)
        or not np.isfinite(half).all()
        or np.any(half <= 0.0)
        or intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or min(height, width) <= 1
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise Stage1AV6WristDiagnosticError("V6_GT_PROJECTION_CONTRACT_INVALID")
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    direction_camera = np.stack(
        (
            (columns - intrinsic[0, 2]) / intrinsic[0, 0],
            -(rows - intrinsic[1, 2]) / intrinsic[1, 1],
            -np.ones_like(columns),
        ),
        axis=-1,
    )
    rotation_camera_from_cube = _rotation_from_quaternion_xyzw(pose[3:])
    # Row-vector form: d_cube = R_camera_from_cube.T @ d_camera.
    direction_cube = direction_camera @ rotation_camera_from_cube
    origin_cube = -rotation_camera_from_cube.T @ pose[:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        low = (-half - origin_cube) / direction_cube
        high = (half - origin_cube) / direction_cube
    near = np.max(np.minimum(low, high), axis=-1)
    far = np.min(np.maximum(low, high), axis=-1)
    return np.isfinite(near) & np.isfinite(far) & (far >= np.maximum(near, 0.0))


def _camera_cube_pose(
    *,
    world_from_camera: np.ndarray,
    cube_position_world_m: np.ndarray,
    cube_quaternion_world_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(world_from_camera, dtype=np.float64)
    position = np.asarray(cube_position_world_m, dtype=np.float64)
    quaternion = _normalized_quaternion_xyzw(cube_quaternion_world_xyzw, label="CUBE_WORLD")
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or position.shape != (3,)
        or not np.isfinite(position).all()
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-5, rtol=0.0)
    ):
        raise Stage1AV6WristDiagnosticError("V6_WORLD_FROM_CAMERA_INVALID")
    rotation_world_from_camera = transform[:3, :3]
    if not np.allclose(
        rotation_world_from_camera.T @ rotation_world_from_camera,
        np.eye(3), atol=2.0e-5, rtol=0.0,
    ) or float(np.linalg.det(rotation_world_from_camera)) <= 0.0:
        raise Stage1AV6WristDiagnosticError("V6_WORLD_FROM_CAMERA_ROTATION_INVALID")
    rotation_camera_from_world = rotation_world_from_camera.T
    cube_position_camera = rotation_camera_from_world @ (position - transform[:3, 3])
    rotation_camera_from_cube = rotation_camera_from_world @ _rotation_from_quaternion_xyzw(quaternion)
    pose = np.concatenate(
        (cube_position_camera, _quaternion_from_rotation_xyzw(rotation_camera_from_cube))
    )
    camera_pose = np.concatenate(
        (transform[:3, 3], _quaternion_from_rotation_xyzw(rotation_world_from_camera))
    )
    return _finite_pose(camera_pose, label="CAMERA_WORLD"), _finite_pose(pose, label="CUBE_CAMERA")


def pooled_wrist_rgbd_feature_v6(frame: PerEnvCameraFrame) -> np.ndarray:
    """Return V6's explicit deployable RGB-D descriptor (12*16*5 = 960)."""

    if (
        frame.rgb.shape != (192, 256, 3)
        or frame.depth_m.shape != (192, 256, 1)
        or frame.depth_valid.shape != (192, 256, 1)
        or not np.isfinite(frame.depth_m).all()
    ):
        raise Stage1AV6WristDiagnosticError("V6_WRIST_RGBD_FEATURE_FRAME_INVALID")
    rgb = frame.rgb.astype(np.float32, copy=False) / 255.0
    valid = frame.depth_valid.astype(np.float32, copy=False)
    depth = np.clip(frame.depth_m.astype(np.float32, copy=False), 0.0, 2.0) / 2.0
    channels = np.concatenate((rgb, depth * valid, valid), axis=-1)
    feature = channels.reshape(12, 16, 16, 16, 5).mean(axis=(1, 3)).reshape(-1)
    if feature.shape != (960,) or not np.isfinite(feature).all():
        raise Stage1AV6WristDiagnosticError("V6_WRIST_RGBD_FEATURE_INVALID")
    return feature.astype(np.float32, copy=False)


def capture_v6_wrist_diagnostic_frames(
    *,
    env: Any,
    frames: Mapping[int, PerEnvCameraFrame],
    world_from_right_wrist: torch.Tensor,
    cube_position_world_m: torch.Tensor,
    cube_quaternion_world_xyzw: torch.Tensor,
) -> tuple[V6WristDiagnosticFrame, ...]:
    """Read V6 masks/poses from the current 25-Hz wrist camera frame.

    The caller must invoke this only for the policy-owned camera frame being
    persisted.  The function rejects any absent renderer AOV instead of
    replacing it with red-pixel heuristics or cube GT in a student field.
    """

    num_envs = int(env.num_envs)
    if not frames or any(
        type(env_id) is not int or env_id < 0 or env_id >= num_envs
        or frame.env_id != env_id
        for env_id, frame in frames.items()
    ):
        raise Stage1AV6WristDiagnosticError("V6_FRAME_ENV_MAPPING_INVALID")
    camera = env.scene["right_wrist_camera"]
    output = camera.data.output
    if (
        not isinstance(output, Mapping)
        or "semantic_segmentation" not in output
        or "instance_segmentation_fast" not in output
    ):
        raise Stage1AV6WristDiagnosticError("V6_SEGMENTATION_OUTPUT_MISSING")
    semantic_all = _tensor(output["semantic_segmentation"])
    instance_all = _tensor(output["instance_segmentation_fast"])
    if (
        semantic_all.ndim != 4
        or semantic_all.shape[:3] != (num_envs, 192, 256)
        or semantic_all.shape[-1] not in (1, 3, 4)
        or instance_all.ndim != 4
        or instance_all.shape[:3] != (num_envs, 192, 256)
        or instance_all.shape[-1] not in (1, 3, 4)
    ):
        raise Stage1AV6WristDiagnosticError("V6_SEGMENTATION_OUTPUT_SHAPE_INVALID")
    info = getattr(camera.data, "info", None)
    if not isinstance(info, Mapping):
        raise Stage1AV6WristDiagnosticError("V6_SEGMENTATION_INFO_MISSING")
    semantic_info = info.get("semantic_segmentation")
    instance_info = info.get("instance_segmentation_fast")
    semantic_tokens = _renderer_label_tokens(
        semantic_info, required_text="cube", source="SEMANTIC"
    )
    # Instance output maps IDs to actual prim paths.  The exact scene object
    # binding (not a visual colour or a name-like heuristic) is the authority.
    instance_tokens = _renderer_label_tokens(
        instance_info, required_text="/object", source="INSTANCE"
    )
    intrinsics_all = _tensor(getattr(camera.data, "intrinsic_matrices", None))
    frame_ids = _tensor(camera.frame).reshape(-1)
    transforms = _tensor(world_from_right_wrist)
    cube_position = _tensor(cube_position_world_m)
    cube_quaternion = _tensor(cube_quaternion_world_xyzw)
    if (
        intrinsics_all.shape != (num_envs, 3, 3)
        or frame_ids.shape != (num_envs,)
        or transforms.shape != (num_envs, 4, 4)
        or cube_position.shape != (num_envs, 3)
        or cube_quaternion.shape != (num_envs, 4)
    ):
        raise Stage1AV6WristDiagnosticError("V6_POSE_OR_INTRINSICS_SHAPE_INVALID")
    try:
        cube_size_m = tuple(float(value) for value in env.cfg.scene.object.spawn.size)
    except (AttributeError, TypeError, ValueError) as error:
        raise Stage1AV6WristDiagnosticError("V6_CUBE_SIZE_AUTHORITY_MISSING") from error
    if len(cube_size_m) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in cube_size_m):
        raise Stage1AV6WristDiagnosticError("V6_CUBE_SIZE_INVALID")
    result: list[V6WristDiagnosticFrame] = []
    for env_id, frame in sorted(frames.items()):
        if int(frame_ids[env_id].item()) != int(frame.frame_id):
            raise Stage1AV6WristDiagnosticError("V6_CAMERA_FRAME_ID_PARITY_FAILED")
        intrinsics = np.asarray(
            intrinsics_all[env_id].detach().to("cpu").numpy(), dtype=np.float64
        )
        if (
            intrinsics.shape != (3, 3)
            or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0.0
            or intrinsics[1, 1] <= 0.0
        ):
            raise Stage1AV6WristDiagnosticError("V6_INTRINSICS_INVALID")
        semantic_mask = _mask_from_renderer_tokens(
            semantic_all[env_id], tokens=semantic_tokens, source="SEMANTIC"
        )
        instance_mask = _mask_from_renderer_tokens(
            instance_all[env_id], tokens=instance_tokens, source="INSTANCE"
        )
        camera_pose, cube_pose = _camera_cube_pose(
            world_from_camera=np.asarray(
                transforms[env_id].detach().to("cpu").numpy(), dtype=np.float64
            ),
            cube_position_world_m=np.asarray(
                cube_position[env_id].detach().to("cpu").numpy(), dtype=np.float64
            ),
            cube_quaternion_world_xyzw=np.asarray(
                cube_quaternion[env_id].detach().to("cpu").numpy(), dtype=np.float64
            ),
        )
        expected_mask = _expected_cube_silhouette_mask(
            cube_pose_camera_optical_m_xyzw=cube_pose,
            cube_size_m=cube_size_m,
            intrinsics_3x3=intrinsics,
            height=192,
            width=256,
        )
        projected_area = int(expected_mask.sum())
        visible = int(instance_mask.sum())
        if projected_area <= 0:
            raise Stage1AV6WristDiagnosticError("V6_CUBE_GT_PROJECTED_AREA_ZERO")
        # Semantic and instance masks should be the same object surface once
        # the V6 diagnostic-only class:cube tag is present.  A mismatch is an
        # instrumented renderer fact, retained as two masks for audit rather
        # than coerced to a single inferred label.
        valid = np.asarray(frame.depth_valid[..., 0], dtype=bool)
        defined = visible > 0
        depth_ratio = float(valid[instance_mask].mean()) if defined else 0.0
        if not math.isfinite(depth_ratio) or not 0.0 <= depth_ratio <= 1.0:
            raise Stage1AV6WristDiagnosticError("V6_CUBE_DEPTH_VALID_RATIO_INVALID")
        result.append(
            V6WristDiagnosticFrame(
                camera_world_pose_m_xyzw=camera_pose,
                cube_pose_camera_optical_m_xyzw=cube_pose,
                intrinsics_3x3=intrinsics,
                cube_semantic_mask=semantic_mask.astype(np.uint8, copy=False),
                cube_instance_mask=instance_mask.astype(np.uint8, copy=False),
                cube_expected_silhouette_mask=expected_mask.astype(np.uint8, copy=False),
                gt_cube_visible_pixel_count=visible,
                gt_cube_projected_area_px=projected_area,
                gt_occlusion_ratio=float(max(0.0, 1.0 - visible / projected_area)),
                cube_mask_depth_valid_ratio=depth_ratio,
                cube_mask_depth_valid_defined=defined,
                semantic_label_ids=semantic_tokens,
                instance_label_ids=instance_tokens,
                semantic_info_sha256=_json_sha256(semantic_info),
                instance_info_sha256=_json_sha256(instance_info),
            )
        )
    return tuple(result)


__all__ = [
    "V6_CAMERA_OPTICAL_CONVENTION",
    "V6_WRIST_DIAGNOSTIC_SCHEMA",
    "Stage1AV6WristDiagnosticError",
    "V6WristDiagnosticFrame",
    "capture_v6_wrist_diagnostic_frames",
    "pooled_wrist_rgbd_feature_v6",
]
