# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only, same-state preview of the contact-free G2 arm command path.

This is an *additive diagnostic API*.  It deliberately has no dependency on
Isaac/Kit or an articulation writer.  The runtime action term is responsible
for collecting a clone-only :class:`G2DLSReadOnlySnapshot`; this module then
replays the source DLS, null-space, soft-limit, and synchronized-limiter math
on fresh local tensors only.

The API is intentionally narrow: it accepts only the frozen contact-free 4-D
route, represented at the ActionManager boundary by a full 8-D packet
``[arm7, +1 OPEN]``.  It cannot authorize a command and cannot establish
runtime equivalence until a separately authorized following-action comparison
has been performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import struct
from typing import Mapping, Sequence

import torch

from ..g2_lift_methodology import RIGHT_ARM_JOINTS
from ..g2_quaternion import quaternion_native_to_xyzw
from ..g2_redundancy_teleop import damped_nullspace_project
from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from ..g2_teleop_dataset import synchronized_rate_limit_position_target


G2_DLS_READ_ONLY_PREVIEW_SCHEMA = "g2_dls_read_only_preview_v1"
G2_DLS_READ_ONLY_PREVIEW_FRAME = "robot_root"
G2_DLS_READ_ONLY_PREVIEW_QUATERNION = "xyzw"
G2_DLS_READ_ONLY_PREVIEW_FULL8_OPEN = 1.0
G2_DLS_READ_ONLY_PREVIEW_HOLD_OPEN = 0.0
G2_DLS_READ_ONLY_PACKET_HASH_SCHEMA = "full8_packet_only_sha256_v1"
# The controller and preview packet are float32.  A direction whose ideal
# float64 Euclidean norm is exactly 4.5 mm can round a few 1e-10 m above the
# decimal value when three float32 components are squared in float64.  This is
# a representation receipt tolerance only: it neither clips the packet nor
# expands the controller's normalized [-1, 1] command surface.
G2_DLS_READ_ONLY_FLOAT32_METRIC_ROUNDTRIP_TOLERANCE_M = 1.0e-9


class G2DLSReadOnlyPreviewError(ValueError):
    """Malformed or unsafe-to-preview state; callers must fail closed."""


class G2DLSReadOnlyPreviewVerdict(str, Enum):
    AVAILABLE = "AVAILABLE_STATIC_PREVIEW"
    REJECTED = "REJECTED_CONTRACT_OR_NUMERICAL_STATE"


class G2DLSRuntimeEquivalence(str, Enum):
    UNQUALIFIED = "UNQUALIFIED_REQUIRES_FOLLOWING_ACTION_COMPARISON"


def _finite_tuple(name: str, values: Sequence[float], size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise G2DLSReadOnlyPreviewError(f"{name} must contain {size} finite values")
    return result


def _finite_matrix(name: str, values: Sequence[Sequence[float]], rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    result = tuple(_finite_tuple(f"{name}[{index}]", value, columns) for index, value in enumerate(values))
    if len(result) != rows:
        raise G2DLSReadOnlyPreviewError(f"{name} must have shape ({rows}, {columns})")
    return result


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise G2DLSReadOnlyPreviewError(f"{name} must be finite and positive")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise G2DLSReadOnlyPreviewError(f"{name} must be finite and non-negative")
    return value


def _hash_action(values: Sequence[float]) -> str:
    canonical = ",".join(format(float(value), ".9g") for value in values)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def source_sha256(path: str | Path) -> str:
    """Return a content hash without treating a filename as an authority."""

    source = Path(path).resolve()
    if not source.is_file():
        raise G2DLSReadOnlyPreviewError(f"source file unavailable: {source}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


@dataclass(frozen=True)
class G2DLSReadOnlySnapshot:
    """Immutable, one-environment clone of all numerical DLS preview inputs.

    All vector/matrix fields are plain tuples rather than tensors.  This keeps
    the receipt durable and proves that the preview cannot alias a live action
    term buffer.  ``runtime_dtype`` and ``runtime_device`` attest the original
    tensors before this diagnostic representation is materialized.
    """

    control_epoch: int
    environment_index: int
    metric_action_4d_root_m: tuple[float, float, float, float]
    full_action_packet_8d: tuple[float, ...]
    packet_hash: str
    packet_hash_schema: str
    ordered_joint_names: tuple[str, ...]
    frame: str
    quaternion_order: str
    native_dls_quaternion_order: str
    native_to_canonical_quaternion_converter: str
    translation_m_per_normalized: float
    runtime_dtype: str
    runtime_device: str
    measured_ee_position_root_m: tuple[float, ...]
    measured_ee_quaternion_root_xyzw: tuple[float, ...]
    desired_ee_position_root_m: tuple[float, ...]
    desired_ee_quaternion_root_xyzw: tuple[float, ...]
    measured_ee_quaternion_native: tuple[float, ...]
    desired_ee_quaternion_native: tuple[float, ...]
    measured_joint_position_rad: tuple[float, ...]
    measured_joint_velocity_rad_s: tuple[float, ...]
    measured_joint_acceleration_rad_s2: tuple[float, ...]
    frame_jacobian_root: tuple[tuple[float, ...], ...]
    soft_joint_limits_rad: tuple[tuple[float, float], ...]
    limiter_previous_target_rad: tuple[float, ...]
    limiter_previous_velocity_rad_s: tuple[float, ...]
    limiter_target_initialized: bool
    limiter_speed_rad_s: float
    limiter_acceleration_rad_s2: float
    physics_dt_s: float
    policy_dt_s: float
    dls_lambda: float
    nullspace_seed_joint_index: int
    nullspace_damping: float
    maximum_nullspace_joint_delta_rad_per_physics_step: float
    joint_limit_margin_rad: float
    effective_elbow_command: float
    rotation_only_request: bool
    source_hashes: tuple[tuple[str, str], ...]
    schema: str = G2_DLS_READ_ONLY_PREVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G2_DLS_READ_ONLY_PREVIEW_SCHEMA:
            raise G2DLSReadOnlyPreviewError("unsupported preview schema")
        if not isinstance(self.control_epoch, int) or self.control_epoch < 0:
            raise G2DLSReadOnlyPreviewError("control_epoch must be a non-negative integer")
        if not isinstance(self.environment_index, int) or self.environment_index < 0:
            raise G2DLSReadOnlyPreviewError("environment_index must be a non-negative integer")
        metric = _finite_tuple("metric_action_4d_root_m", self.metric_action_4d_root_m, 4)
        if math.sqrt(sum(value * value for value in metric[:3])) > 0.0045 + G2_DLS_READ_ONLY_FLOAT32_METRIC_ROUNDTRIP_TOLERANCE_M or metric[3] != G2_DLS_READ_ONLY_PREVIEW_HOLD_OPEN:
            raise G2DLSReadOnlyPreviewError("preview requires <=4.5 mm metric xyz plus HOLD_OPEN == 0")
        object.__setattr__(self, "metric_action_4d_root_m", metric)
        packet = _finite_tuple("full_action_packet_8d", self.full_action_packet_8d, 8)
        normalized = tuple(value / G2_TRANSLATION_ACTION_SCALE_M for value in metric[:3])
        if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7) for actual, expected in zip(packet[:3], normalized, strict=True)) or packet[3:7] != (0.0, 0.0, 0.0, 0.0):
            raise G2DLSReadOnlyPreviewError("full 8-D arm packet does not match fixed 4-D route")
        if packet[7] != G2_DLS_READ_ONLY_PREVIEW_FULL8_OPEN:
            raise G2DLSReadOnlyPreviewError("contact-free preview requires full8 gripper OPEN == +1")
        object.__setattr__(self, "full_action_packet_8d", packet)
        if self.packet_hash_schema != G2_DLS_READ_ONLY_PACKET_HASH_SCHEMA:
            raise G2DLSReadOnlyPreviewError("unsupported low-level packet hash schema")
        if self.packet_hash != _hash_action(packet):
            raise G2DLSReadOnlyPreviewError("packet hash does not bind the full 8-D packet")
        if tuple(self.ordered_joint_names) != RIGHT_ARM_JOINTS:
            raise G2DLSReadOnlyPreviewError("right arm joint order mismatch")
        if self.frame != G2_DLS_READ_ONLY_PREVIEW_FRAME:
            raise G2DLSReadOnlyPreviewError("preview frame must be robot_root")
        if self.quaternion_order != G2_DLS_READ_ONLY_PREVIEW_QUATERNION:
            raise G2DLSReadOnlyPreviewError("quaternion order must be xyzw")
        if self.native_dls_quaternion_order not in {"xyzw", "wxyz"}:
            raise G2DLSReadOnlyPreviewError("native DLS quaternion order must be explicit xyzw/wxyz")
        if self.native_to_canonical_quaternion_converter != "g2_quaternion.quaternion_native_to_xyzw":
            raise G2DLSReadOnlyPreviewError("native-to-canonical quaternion converter provenance mismatch")
        scale = _finite_positive("translation_m_per_normalized", self.translation_m_per_normalized)
        # The normal controller stores this source-defined scale in a
        # float32 tensor.  Comparing that tensor value after conversion to a
        # Python float against the decimal literal would reject its canonical
        # representation (0.022500000894...).  Preserve strictness in the
        # actual runtime representation instead: both values must encode to
        # exactly the same IEEE-754 float32 bytes.  This is not a tolerance.
        if struct.pack("<f", scale) != struct.pack("<f", G2_TRANSLATION_ACTION_SCALE_M):
            raise G2DLSReadOnlyPreviewError("translation scale must be exactly 0.0225 m/normalized")
        object.__setattr__(self, "translation_m_per_normalized", scale)
        if self.runtime_dtype != "float32":
            raise G2DLSReadOnlyPreviewError("runtime dtype must be float32")
        if not isinstance(self.runtime_device, str) or not self.runtime_device:
            raise G2DLSReadOnlyPreviewError("runtime device must be explicit")
        for name, values, size in (
            ("measured_ee_position_root_m", self.measured_ee_position_root_m, 3),
            ("measured_ee_quaternion_root_xyzw", self.measured_ee_quaternion_root_xyzw, 4),
            ("desired_ee_position_root_m", self.desired_ee_position_root_m, 3),
            ("desired_ee_quaternion_root_xyzw", self.desired_ee_quaternion_root_xyzw, 4),
            ("measured_ee_quaternion_native", self.measured_ee_quaternion_native, 4),
            ("desired_ee_quaternion_native", self.desired_ee_quaternion_native, 4),
            ("measured_joint_position_rad", self.measured_joint_position_rad, 7),
            ("measured_joint_velocity_rad_s", self.measured_joint_velocity_rad_s, 7),
            ("measured_joint_acceleration_rad_s2", self.measured_joint_acceleration_rad_s2, 7),
            ("limiter_previous_target_rad", self.limiter_previous_target_rad, 7),
            ("limiter_previous_velocity_rad_s", self.limiter_previous_velocity_rad_s, 7),
        ):
            object.__setattr__(self, name, _finite_tuple(name, values, size))
        quat_norm = math.sqrt(sum(value * value for value in self.measured_ee_quaternion_root_xyzw))
        desired_quat_norm = math.sqrt(sum(value * value for value in self.desired_ee_quaternion_root_xyzw))
        if abs(quat_norm - 1.0) > 1.0e-4 or abs(desired_quat_norm - 1.0) > 1.0e-4:
            raise G2DLSReadOnlyPreviewError("EE quaternions must be unit xyzw values")
        native_measured = torch.tensor((self.measured_ee_quaternion_native,), dtype=torch.float32)
        native_desired = torch.tensor((self.desired_ee_quaternion_native,), dtype=torch.float32)
        converted_measured = quaternion_native_to_xyzw(native_measured, self.native_dls_quaternion_order)[0]
        converted_desired = quaternion_native_to_xyzw(native_desired, self.native_dls_quaternion_order)[0]
        for name, converted, canonical in (
            ("measured native quaternion", converted_measured, self.measured_ee_quaternion_root_xyzw),
            ("desired native quaternion", converted_desired, self.desired_ee_quaternion_root_xyzw),
        ):
            # Canonicalization permits q/-q, but no unlabeled order change.
            canonical_tensor = torch.tensor(canonical, dtype=torch.float32)
            if not bool(torch.allclose(converted, canonical_tensor, rtol=0.0, atol=1.0e-5)):
                raise G2DLSReadOnlyPreviewError(f"{name} does not match canonical XYZW conversion")
        object.__setattr__(self, "frame_jacobian_root", _finite_matrix("frame_jacobian_root", self.frame_jacobian_root, 6, 7))
        limits = tuple(_finite_tuple(f"soft_joint_limits_rad[{index}]", value, 2) for index, value in enumerate(self.soft_joint_limits_rad))
        if len(limits) != 7 or any(lower >= upper for lower, upper in limits):
            raise G2DLSReadOnlyPreviewError("soft joint limits must be seven ordered lower/upper pairs")
        object.__setattr__(self, "soft_joint_limits_rad", limits)
        if not isinstance(self.limiter_target_initialized, bool):
            raise G2DLSReadOnlyPreviewError("limiter initialization state must be bool")
        for name in ("limiter_speed_rad_s", "limiter_acceleration_rad_s2", "physics_dt_s", "policy_dt_s", "dls_lambda", "maximum_nullspace_joint_delta_rad_per_physics_step", "joint_limit_margin_rad"):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        if self.physics_dt_s != 0.002 or self.policy_dt_s != 0.020:
            raise G2DLSReadOnlyPreviewError("preview requires frozen 2 ms physics and 20 ms policy intervals")
        if not 0 <= self.nullspace_seed_joint_index < 7:
            raise G2DLSReadOnlyPreviewError("nullspace seed joint index must be in [0, 6]")
        object.__setattr__(self, "nullspace_damping", _finite_nonnegative("nullspace_damping", self.nullspace_damping))
        object.__setattr__(self, "effective_elbow_command", _finite_tuple("effective_elbow_command", (self.effective_elbow_command,), 1)[0])
        if self.effective_elbow_command != 0.0 or self.rotation_only_request:
            raise G2DLSReadOnlyPreviewError("fixed 4-D preview rejects rotation-only or elbow state")
        hashes = tuple((str(name), str(digest)) for name, digest in self.source_hashes)
        if not hashes or any(len(digest) != 64 for _, digest in hashes):
            raise G2DLSReadOnlyPreviewError("source hashes must contain SHA-256 provenance")
        object.__setattr__(self, "source_hashes", hashes)

    @property
    def requested_metric_delta_root_m(self) -> tuple[float, float, float]:
        return tuple(self.metric_action_4d_root_m[:3])


@dataclass(frozen=True)
class G2DLSReadOnlyPreviewReceipt:
    """Immutable result; it is never controller/admission authority."""

    control_epoch: int
    environment_index: int
    packet_hash: str
    packet_hash_schema: str
    verdict: G2DLSReadOnlyPreviewVerdict
    runtime_equivalence: G2DLSRuntimeEquivalence
    command_authorized: bool
    reason: str | None
    requested_metric_delta_root_m: tuple[float, float, float]
    post_dls_target_rad: tuple[float, ...] | None
    post_nullspace_target_rad: tuple[float, ...] | None
    post_soft_limit_target_rad: tuple[float, ...] | None
    predicted_emitted_target_rad: tuple[float, ...] | None
    predicted_target_velocity_rad_s: tuple[float, ...] | None
    predicted_target_acceleration_rad_s2: tuple[float, ...] | None
    limiter_diagnostics: Mapping[str, tuple[float, ...] | tuple[bool, ...]]
    source_hashes: tuple[tuple[str, str], ...]
    no_side_effects: bool = True
    schema: str = G2_DLS_READ_ONLY_PREVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G2_DLS_READ_ONLY_PREVIEW_SCHEMA:
            raise G2DLSReadOnlyPreviewError("unsupported preview receipt schema")
        if self.packet_hash_schema != G2_DLS_READ_ONLY_PACKET_HASH_SCHEMA:
            raise G2DLSReadOnlyPreviewError("unsupported low-level packet hash schema")
        if not isinstance(self.packet_hash, str) or len(self.packet_hash) != 64:
            raise G2DLSReadOnlyPreviewError("packet hash must be a SHA-256 digest")
        if self.command_authorized or not self.no_side_effects:
            raise G2DLSReadOnlyPreviewError("preview cannot authorize commands or mutate runtime state")
        if self.runtime_equivalence is not G2DLSRuntimeEquivalence.UNQUALIFIED:
            raise G2DLSReadOnlyPreviewError("runtime equivalence cannot be promoted by preview")


def _local_tensor(values: Sequence[float] | Sequence[Sequence[float]], snapshot: G2DLSReadOnlySnapshot) -> torch.Tensor:
    # Local CPU tensors are intentional: all runtime tensors were copied to
    # immutable values first.  dtype is preserved; device provenance remains in
    # the receipt because a static preview must not create a CUDA context.
    return torch.tensor(values, dtype=torch.float32, device="cpu").clone()


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = first.unbind(dim=-1)
    x2, y2, z2, w2 = second.unbind(dim=-1)
    return torch.stack((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ), dim=-1)


def _axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    # Exact source convention in Isaac Lab utils.math.axis_angle_from_quat.
    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0))
    magnitude = torch.linalg.norm(quat[..., :3], dim=-1)
    half_angle = torch.atan2(magnitude, quat[..., 3])
    angle = 2.0 * half_angle
    sin_half_over_angle = torch.where(
        angle.abs() > 1.0e-6,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48.0,
    )
    return quat[..., :3] / sin_half_over_angle.unsqueeze(-1)


def _source_pose_error(current_position: torch.Tensor, current_quaternion: torch.Tensor, desired_position: torch.Tensor, desired_quaternion: torch.Tensor) -> torch.Tensor:
    source_norm = _quat_mul(current_quaternion, _quat_conjugate(current_quaternion))[:, 3]
    source_inverse = _quat_conjugate(current_quaternion) / source_norm.unsqueeze(-1)
    quat_error = _quat_mul(desired_quaternion, source_inverse)
    return torch.cat((desired_position - current_position, _axis_angle_from_quat(quat_error)), dim=-1)


def _tuples(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())


def _diagnostics_to_tuples(diagnostics: Mapping[str, torch.Tensor]) -> dict[str, tuple[float, ...] | tuple[bool, ...]]:
    converted: dict[str, tuple[float, ...] | tuple[bool, ...]] = {}
    for name, value in diagnostics.items():
        flattened = value.detach().cpu().reshape(-1)
        converted[name] = tuple(bool(item) for item in flattened.tolist()) if value.dtype == torch.bool else tuple(float(item) for item in flattened.tolist())
    return converted


def preview_g2_dls_read_only(snapshot: G2DLSReadOnlySnapshot) -> G2DLSReadOnlyPreviewReceipt:
    """Run source-equivalent DLS/limiter math using only local cloned tensors."""

    if not isinstance(snapshot, G2DLSReadOnlySnapshot):
        raise G2DLSReadOnlyPreviewError("snapshot must be G2DLSReadOnlySnapshot")
    base = dict(
        control_epoch=snapshot.control_epoch,
        environment_index=snapshot.environment_index,
        packet_hash=snapshot.packet_hash,
        packet_hash_schema=snapshot.packet_hash_schema,
        runtime_equivalence=G2DLSRuntimeEquivalence.UNQUALIFIED,
        command_authorized=False,
        requested_metric_delta_root_m=snapshot.requested_metric_delta_root_m,
        source_hashes=snapshot.source_hashes,
    )
    try:
        current_position = _local_tensor((snapshot.measured_ee_position_root_m,), snapshot)
        current_quaternion = _local_tensor((snapshot.measured_ee_quaternion_root_xyzw,), snapshot)
        desired_position = _local_tensor((snapshot.desired_ee_position_root_m,), snapshot)
        desired_quaternion = _local_tensor((snapshot.desired_ee_quaternion_root_xyzw,), snapshot)
        joint_position = _local_tensor((snapshot.measured_joint_position_rad,), snapshot)
        jacobian = _local_tensor((snapshot.frame_jacobian_root,), snapshot)
        pose_error = _source_pose_error(current_position, current_quaternion, desired_position, desired_quaternion)
        jacobian_transpose = jacobian.transpose(-1, -2)
        identity = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
        delta_joint = jacobian_transpose @ torch.linalg.inv(
            jacobian @ jacobian_transpose + snapshot.dls_lambda**2 * identity
        ) @ pose_error.unsqueeze(-1)
        post_dls = joint_position + delta_joint.squeeze(-1)
        projection = damped_nullspace_project(
            jacobian,
            torch.zeros((1,), dtype=jacobian.dtype, device=jacobian.device),
            seed_joint_index=snapshot.nullspace_seed_joint_index,
            damping=snapshot.nullspace_damping,
            maximum_joint_delta_rad=snapshot.maximum_nullspace_joint_delta_rad_per_physics_step,
        )
        post_nullspace = post_dls + projection.joint_delta
        limits = _local_tensor((snapshot.soft_joint_limits_rad,), snapshot)
        post_soft_limit = torch.clamp(
            post_nullspace,
            limits[..., 0] + snapshot.joint_limit_margin_rad,
            limits[..., 1] - snapshot.joint_limit_margin_rad,
        )
        if snapshot.limiter_target_initialized:
            previous_target = _local_tensor((snapshot.limiter_previous_target_rad,), snapshot)
            previous_velocity = _local_tensor((snapshot.limiter_previous_velocity_rad_s,), snapshot)
        else:
            previous_target = joint_position.clone()
            previous_velocity = torch.zeros_like(joint_position)
        diagnostics: dict[str, torch.Tensor] = {}
        emitted, target_velocity = synchronized_rate_limit_position_target(
            previous_target.clone(),
            post_soft_limit.clone(),
            previous_velocity.clone(),
            maximum_speed_rad_s=snapshot.limiter_speed_rad_s,
            maximum_acceleration_rad_s2=snapshot.limiter_acceleration_rad_s2,
            physics_dt_s=snapshot.physics_dt_s,
            diagnostics=diagnostics,
        )
        target_acceleration = (target_velocity - previous_velocity) / snapshot.physics_dt_s
    except (RuntimeError, ValueError, torch.linalg.LinAlgError) as error:
        return G2DLSReadOnlyPreviewReceipt(
            verdict=G2DLSReadOnlyPreviewVerdict.REJECTED,
            reason=f"PREVIEW_NUMERICAL_OR_LIMITER_REJECTED:{error}",
            post_dls_target_rad=None,
            post_nullspace_target_rad=None,
            post_soft_limit_target_rad=None,
            predicted_emitted_target_rad=None,
            predicted_target_velocity_rad_s=None,
            predicted_target_acceleration_rad_s2=None,
            limiter_diagnostics={},
            **base,
        )
    return G2DLSReadOnlyPreviewReceipt(
        verdict=G2DLSReadOnlyPreviewVerdict.AVAILABLE,
        reason=None,
        post_dls_target_rad=_tuples(post_dls[0]),
        post_nullspace_target_rad=_tuples(post_nullspace[0]),
        post_soft_limit_target_rad=_tuples(post_soft_limit[0]),
        predicted_emitted_target_rad=_tuples(emitted[0]),
        predicted_target_velocity_rad_s=_tuples(target_velocity[0]),
        predicted_target_acceleration_rad_s2=_tuples(target_acceleration[0]),
        limiter_diagnostics=_diagnostics_to_tuples(diagnostics),
        **base,
    )


__all__ = [
    "G2_DLS_READ_ONLY_PREVIEW_FRAME",
    "G2_DLS_READ_ONLY_PREVIEW_FULL8_OPEN",
    "G2_DLS_READ_ONLY_PREVIEW_HOLD_OPEN",
    "G2_DLS_READ_ONLY_PACKET_HASH_SCHEMA",
    "G2_DLS_READ_ONLY_PREVIEW_QUATERNION",
    "G2_DLS_READ_ONLY_PREVIEW_SCHEMA",
    "G2DLSReadOnlyPreviewError",
    "G2DLSReadOnlyPreviewReceipt",
    "G2DLSReadOnlyPreviewVerdict",
    "G2DLSReadOnlySnapshot",
    "G2DLSRuntimeEquivalence",
    "preview_g2_dls_read_only",
    "source_sha256",
]
