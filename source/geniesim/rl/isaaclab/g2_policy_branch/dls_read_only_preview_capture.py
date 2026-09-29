# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Disabled-by-default, source-owned receipt queue for DLS preview capture.

The queue has no environment, ActionManager, controller, or articulation
writer dependency.  The arm action term calls it *after* its ordinary
``process_actions`` work has completed.  It therefore cannot consume a second
packet or become a command path.  A caller must stage one immutable full-8D
packet for one epoch before canonical ``env.step`` processing; missing,
duplicate, late, or mismatched packets fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Sequence

import torch

from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from .dls_read_only_preview_api import (
    G2_DLS_READ_ONLY_FLOAT32_METRIC_ROUNDTRIP_TOLERANCE_M,
    G2_DLS_READ_ONLY_PACKET_HASH_SCHEMA,
    G2DLSReadOnlyPreviewError,
    G2DLSReadOnlyPreviewReceipt,
    G2DLSReadOnlySnapshot,
    preview_g2_dls_read_only,
)


G2_DLS_READ_ONLY_CAPTURE_SCHEMA = "g2_dls_read_only_capture_v1"
G2_DLS_READ_ONLY_BATCH_HASH_SCHEMA = "full8_batch_sha256_v1"


class G2DLSReadOnlyCaptureError(ValueError):
    """Capture staging/lifecycle contract violated; fail closed."""


def _finite_row(name: str, values: Sequence[float], width: int) -> tuple[float, ...]:
    row = tuple(float(value) for value in values)
    if len(row) != width or not all(math.isfinite(value) for value in row):
        raise G2DLSReadOnlyCaptureError(f"{name} must be {width} finite values")
    return row


def _packet_hash(row: Sequence[float]) -> str:
    return hashlib.sha256(
        ",".join(format(float(value), ".9g") for value in row).encode("ascii")
    ).hexdigest()


def _batch_hash(rows: Sequence[Sequence[float]]) -> str:
    return hashlib.sha256(
        ";".join(
            ",".join(format(float(value), ".9g") for value in row)
            for row in rows
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class G2DLSReadOnlyPreviewCaptureRequest:
    """One immutable, contact-free full packet staged before canonical ingress."""

    control_epoch: int
    metric_action_4d_root_m: tuple[tuple[float, ...], ...]
    full_action_packet_8d: tuple[tuple[float, ...], ...]
    environment_index: int
    full8_batch_hash: str
    full8_batch_hash_schema: str = G2_DLS_READ_ONLY_BATCH_HASH_SCHEMA
    schema: str = G2_DLS_READ_ONLY_CAPTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G2_DLS_READ_ONLY_CAPTURE_SCHEMA:
            raise G2DLSReadOnlyCaptureError("unsupported capture schema")
        if type(self.control_epoch) is not int or self.control_epoch < 0:
            raise G2DLSReadOnlyCaptureError("control_epoch must be a non-negative integer")
        metric = tuple(
            _finite_row(f"metric_action_4d_root_m[{index}]", row, 4)
            for index, row in enumerate(self.metric_action_4d_root_m)
        )
        packet = tuple(
            _finite_row(f"full_action_packet_8d[{index}]", row, 8)
            for index, row in enumerate(self.full_action_packet_8d)
        )
        if not metric or len(metric) != len(packet):
            raise G2DLSReadOnlyCaptureError("metric/full8 batch must be non-empty and equal")
        if not 0 <= self.environment_index < len(packet):
            raise G2DLSReadOnlyCaptureError("environment_index is outside staged packet batch")
        for index, (metric_row, packet_row) in enumerate(zip(metric, packet, strict=True)):
            if math.sqrt(sum(value * value for value in metric_row[:3])) > 0.0045 + G2_DLS_READ_ONLY_FLOAT32_METRIC_ROUNDTRIP_TOLERANCE_M:
                raise G2DLSReadOnlyCaptureError(f"metric action row {index} exceeds 4.5 mm")
            if metric_row[3] != 0.0:
                raise G2DLSReadOnlyCaptureError("contact-free capture requires HOLD_OPEN == 0")
            normalized = tuple(value / G2_TRANSLATION_ACTION_SCALE_M for value in metric_row[:3])
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-7)
                for actual, expected in zip(packet_row[:3], normalized, strict=True)
            ) or packet_row[3:7] != (0.0, 0.0, 0.0, 0.0) or packet_row[7] != 1.0:
                raise G2DLSReadOnlyCaptureError(
                    "full8 must be xyz/0.0225, zero rpy/elbow, and OPEN +1"
                )
        if self.full8_batch_hash_schema != G2_DLS_READ_ONLY_BATCH_HASH_SCHEMA:
            raise G2DLSReadOnlyCaptureError("unsupported batch hash schema")
        if self.full8_batch_hash != _batch_hash(packet):
            raise G2DLSReadOnlyCaptureError("full8 batch hash mismatch")
        object.__setattr__(self, "metric_action_4d_root_m", metric)
        object.__setattr__(self, "full_action_packet_8d", packet)

    @classmethod
    def from_tensors(
        cls,
        *,
        control_epoch: int,
        metric_action_4d_root_m: torch.Tensor,
        full_action_packet_8d: torch.Tensor,
        environment_index: int = 0,
    ) -> "G2DLSReadOnlyPreviewCaptureRequest":
        if (
            metric_action_4d_root_m.ndim != 2
            or full_action_packet_8d.ndim != 2
            or metric_action_4d_root_m.shape[1] != 4
            or full_action_packet_8d.shape[1] != 8
            or metric_action_4d_root_m.shape[0] != full_action_packet_8d.shape[0]
            or metric_action_4d_root_m.dtype != torch.float32
            or full_action_packet_8d.dtype != torch.float32
            or metric_action_4d_root_m.device != full_action_packet_8d.device
        ):
            raise G2DLSReadOnlyCaptureError(
                "capture request requires same-device float32 [N,4] and [N,8] tensors"
            )
        metric = tuple(tuple(float(value) for value in row) for row in metric_action_4d_root_m.detach().cpu().tolist())
        packet = tuple(tuple(float(value) for value in row) for row in full_action_packet_8d.detach().cpu().tolist())
        return cls(
            control_epoch=control_epoch,
            metric_action_4d_root_m=metric,
            full_action_packet_8d=packet,
            environment_index=environment_index,
            full8_batch_hash=_batch_hash(packet),
        )

    @property
    def selected_packet_hash(self) -> str:
        return _packet_hash(self.full_action_packet_8d[self.environment_index])

    @property
    def batch_size(self) -> int:
        return len(self.full_action_packet_8d)

    def arm7_tensor(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            tuple(row[:7] for row in self.full_action_packet_8d),
            dtype=torch.float32,
            device=device,
        )

    def metric_tensor(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.metric_action_4d_root_m, dtype=torch.float32, device=device)

    def full8_tensor(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.full_action_packet_8d, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class G2DLSReadOnlyPreviewCaptureReceipt:
    """Immutable post-process receipt; it cannot authorize or apply a command."""

    control_epoch: int
    full8_batch_hash: str
    full8_batch_hash_schema: str
    selected_packet_hash: str
    snapshot: G2DLSReadOnlySnapshot
    preview_receipt: G2DLSReadOnlyPreviewReceipt
    command_authorized: bool = False
    schema: str = G2_DLS_READ_ONLY_CAPTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G2_DLS_READ_ONLY_CAPTURE_SCHEMA:
            raise G2DLSReadOnlyCaptureError("unsupported capture receipt schema")
        if self.command_authorized:
            raise G2DLSReadOnlyCaptureError("capture receipt can never authorize a command")
        if self.full8_batch_hash_schema != G2_DLS_READ_ONLY_BATCH_HASH_SCHEMA:
            raise G2DLSReadOnlyCaptureError("unsupported capture receipt batch hash schema")
        if self.snapshot.control_epoch != self.control_epoch:
            raise G2DLSReadOnlyCaptureError("snapshot epoch mismatch")
        if self.snapshot.packet_hash_schema != G2_DLS_READ_ONLY_PACKET_HASH_SCHEMA:
            raise G2DLSReadOnlyCaptureError("snapshot packet hash schema mismatch")
        if self.snapshot.packet_hash != self.selected_packet_hash:
            raise G2DLSReadOnlyCaptureError("snapshot does not bind selected staged full8 packet")
        if self.preview_receipt.packet_hash != self.selected_packet_hash:
            raise G2DLSReadOnlyCaptureError("preview does not bind selected staged full8 packet")
        if self.preview_receipt.predicted_emitted_target_rad is None:
            raise G2DLSReadOnlyCaptureError("preview did not produce an emitted target")


@dataclass(frozen=True)
class G2DLSReadOnlyFollowingControllerTelemetry:
    """Clone-only normal controller target captured in the same action epoch."""

    control_epoch: int
    selected_packet_hash: str
    ordered_joint_names: tuple[str, ...]
    normal_controller_target_rad: tuple[float, ...]
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_acceleration_rad_s2: tuple[float, ...]
    source_defined_exact_float32_correspondence: bool
    comparison_domain: str = "FLOAT32_TENSOR_BYTES"
    capture_phase: str = "FOLLOWING_NORMAL_CONTROLLER_TARGET"
    command_authorized: bool = False

    def __post_init__(self) -> None:
        if self.command_authorized or self.capture_phase != "FOLLOWING_NORMAL_CONTROLLER_TARGET":
            raise G2DLSReadOnlyCaptureError("following telemetry cannot be command authority")
        if self.comparison_domain != "FLOAT32_TENSOR_BYTES":
            raise G2DLSReadOnlyCaptureError("following target comparison domain mismatch")
        for name, value in (
            ("normal_controller_target_rad", self.normal_controller_target_rad),
            ("joint_position_rad", self.joint_position_rad),
            ("joint_velocity_rad_s", self.joint_velocity_rad_s),
            ("joint_acceleration_rad_s2", self.joint_acceleration_rad_s2),
        ):
            _finite_row(name, value, 7)


@dataclass(frozen=True)
class G2DLSReadOnlyPreviewCaptureBundle:
    """The pre-apply snapshot and following normal controller target together."""

    receipt: G2DLSReadOnlyPreviewCaptureReceipt
    following_controller_telemetry: G2DLSReadOnlyFollowingControllerTelemetry
    process_capture_count: int
    normal_apply_count: int

    def __post_init__(self) -> None:
        if self.receipt.control_epoch != self.following_controller_telemetry.control_epoch:
            raise G2DLSReadOnlyCaptureError("bundle epoch mismatch")
        if self.receipt.selected_packet_hash != self.following_controller_telemetry.selected_packet_hash:
            raise G2DLSReadOnlyCaptureError("bundle packet hash mismatch")
        if self.process_capture_count != 1 or self.normal_apply_count < 1:
            raise G2DLSReadOnlyCaptureError("bundle dynamic count mismatch")


class G2DLSReadOnlyPreviewCaptureQueue:
    """One-request/one-receipt bounded queue used only inside arm processing."""

    def __init__(self) -> None:
        self._enabled = False
        self._pending: G2DLSReadOnlyPreviewCaptureRequest | None = None
        self._receipt: G2DLSReadOnlyPreviewCaptureReceipt | None = None
        self._following: G2DLSReadOnlyFollowingControllerTelemetry | None = None
        self._process_capture_count = 0
        self._normal_apply_count = 0
        self._last_completed_epoch: int | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def pending_epoch(self) -> int | None:
        return None if self._pending is None else self._pending.control_epoch

    @property
    def receipt_epoch(self) -> int | None:
        return None if self._receipt is None else self._receipt.control_epoch

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if not enabled and (self._pending is not None or self._receipt is not None):
            raise G2DLSReadOnlyCaptureError("cannot disable with pending capture evidence")
        self._enabled = enabled

    def stage(self, request: G2DLSReadOnlyPreviewCaptureRequest) -> None:
        if not self._enabled:
            raise G2DLSReadOnlyCaptureError("capture instrumentation is disabled")
        if not isinstance(request, G2DLSReadOnlyPreviewCaptureRequest):
            raise G2DLSReadOnlyCaptureError("request must be immutable capture request")
        if self._pending is not None:
            raise G2DLSReadOnlyCaptureError("duplicate or unconsumed capture request")
        if self._receipt is not None:
            raise G2DLSReadOnlyCaptureError("previous capture receipt must be taken before next epoch")
        if self._last_completed_epoch is not None and request.control_epoch <= self._last_completed_epoch:
            raise G2DLSReadOnlyCaptureError("late or duplicate control epoch")
        self._pending = request

    def capture_after_normal_process(
        self,
        *,
        processed_arm7: torch.Tensor,
        snapshot_factory: Callable[[G2DLSReadOnlyPreviewCaptureRequest], G2DLSReadOnlySnapshot],
    ) -> G2DLSReadOnlyPreviewCaptureReceipt | None:
        """Capture once after normal arm processing; never calls a writer itself."""

        if not self._enabled:
            return None
        request = self._pending
        if request is None:
            raise G2DLSReadOnlyCaptureError("missing staged full8 capture request for processed epoch")
        if (
            processed_arm7.dtype != torch.float32
            or processed_arm7.ndim != 2
            or tuple(processed_arm7.shape) != (request.batch_size, 7)
            or not torch.equal(processed_arm7, request.arm7_tensor(device=processed_arm7.device))
        ):
            raise G2DLSReadOnlyCaptureError("processed arm7 mismatches staged full8 packet")
        snapshot = snapshot_factory(request)
        if not isinstance(snapshot, G2DLSReadOnlySnapshot):
            raise G2DLSReadOnlyCaptureError("snapshot factory returned wrong type")
        preview = preview_g2_dls_read_only(snapshot)
        if preview.predicted_emitted_target_rad is None:
            raise G2DLSReadOnlyCaptureError("clone-only preview lacks emitted target")
        receipt = G2DLSReadOnlyPreviewCaptureReceipt(
            control_epoch=request.control_epoch,
            full8_batch_hash=request.full8_batch_hash,
            full8_batch_hash_schema=request.full8_batch_hash_schema,
            selected_packet_hash=request.selected_packet_hash,
            snapshot=snapshot,
            preview_receipt=preview,
        )
        self._pending = None
        self._receipt = receipt
        self._process_capture_count += 1
        self._last_completed_epoch = request.control_epoch
        return receipt

    def record_following_normal_controller_target(
        self,
        *,
        emitted_target_rad: torch.Tensor,
        joint_position_rad: torch.Tensor,
        joint_velocity_rad_s: torch.Tensor,
        joint_acceleration_rad_s2: torch.Tensor,
        ordered_joint_names: Sequence[str],
    ) -> G2DLSReadOnlyFollowingControllerTelemetry | None:
        """Clone actual normal target after limiter and before the writer call."""

        if not self._enabled:
            return None
        if self._receipt is None:
            raise G2DLSReadOnlyCaptureError("following target missing receipt")
        tensors = (emitted_target_rad, joint_position_rad, joint_velocity_rad_s, joint_acceleration_rad_s2)
        if any(tensor.dtype != torch.float32 or tensor.ndim != 2 or tensor.shape != (1, 7) for tensor in tensors):
            raise G2DLSReadOnlyCaptureError("following target requires one float32 7-DoF row")
        self._normal_apply_count += 1
        if self._following is not None:
            return self._following
        predicted = torch.tensor(
            (self._receipt.preview_receipt.predicted_emitted_target_rad,),
            dtype=torch.float32,
            device=emitted_target_rad.device,
        )
        target = emitted_target_rad.detach().clone()
        telemetry = G2DLSReadOnlyFollowingControllerTelemetry(
            control_epoch=self._receipt.control_epoch,
            selected_packet_hash=self._receipt.selected_packet_hash,
            ordered_joint_names=tuple(str(name) for name in ordered_joint_names),
            normal_controller_target_rad=tuple(float(value) for value in target[0].cpu().tolist()),
            joint_position_rad=tuple(float(value) for value in joint_position_rad.detach().clone()[0].cpu().tolist()),
            joint_velocity_rad_s=tuple(float(value) for value in joint_velocity_rad_s.detach().clone()[0].cpu().tolist()),
            joint_acceleration_rad_s2=tuple(float(value) for value in joint_acceleration_rad_s2.detach().clone()[0].cpu().tolist()),
            source_defined_exact_float32_correspondence=bool(torch.equal(target, predicted)),
        )
        self._following = telemetry
        return telemetry

    def assert_no_pending_before_apply(self) -> None:
        if self._enabled and self._pending is not None:
            raise G2DLSReadOnlyCaptureError("capture request remained pending at apply boundary")

    def take_receipt(self, *, control_epoch: int) -> G2DLSReadOnlyPreviewCaptureReceipt:
        if self._receipt is None:
            raise G2DLSReadOnlyCaptureError("no capture receipt is available")
        if control_epoch != self._receipt.control_epoch:
            raise G2DLSReadOnlyCaptureError("late or mismatched capture receipt epoch")
        receipt = self._receipt
        self._receipt = None
        return receipt

    def take_bundle(self, *, control_epoch: int) -> G2DLSReadOnlyPreviewCaptureBundle:
        if self._receipt is None or self._following is None:
            raise G2DLSReadOnlyCaptureError("same-epoch receipt lacks following controller telemetry")
        if control_epoch != self._receipt.control_epoch:
            raise G2DLSReadOnlyCaptureError("late or mismatched capture bundle epoch")
        bundle = G2DLSReadOnlyPreviewCaptureBundle(
            receipt=self._receipt,
            following_controller_telemetry=self._following,
            process_capture_count=self._process_capture_count,
            normal_apply_count=self._normal_apply_count,
        )
        self._receipt = None
        self._following = None
        self._process_capture_count = 0
        self._normal_apply_count = 0
        return bundle


__all__ = [
    "G2_DLS_READ_ONLY_BATCH_HASH_SCHEMA",
    "G2_DLS_READ_ONLY_CAPTURE_SCHEMA",
    "G2DLSReadOnlyCaptureError",
    "G2DLSReadOnlyPreviewCaptureQueue",
    "G2DLSReadOnlyPreviewCaptureBundle",
    "G2DLSReadOnlyPreviewCaptureReceipt",
    "G2DLSReadOnlyPreviewCaptureRequest",
    "G2DLSReadOnlyFollowingControllerTelemetry",
]
