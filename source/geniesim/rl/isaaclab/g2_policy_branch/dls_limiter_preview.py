# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable diagnostic preview for the G2 DLS-to-limiter boundary.

This module is deliberately *not* a controller, an admission authority, or a
replacement for :class:`G2RedundancyDifferentialIKAction`.  Its narrow job is
to apply the production ``synchronized_rate_limit_position_target`` function
to an immutable copy of a same-epoch, **already resolved** DLS target.

The selected action term computes that target only in ``apply_actions`` after
using mutable controller state.  Until a separately qualified runtime adapter
can materialize the post-DLS/post-nullspace/post-soft-limit target without
calling ``process_actions`` or ``apply_actions``, this preview must fail
closed.  In particular, a feasible preview never authorizes ``env.step``.

Keeping this boundary explicit prevents an offline NumPy mirror or a planner
joint target from being mistaken for the live seven-joint DLS target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import torch

from ..g2_lift_methodology import RIGHT_ARM_JOINTS
from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from ..g2_teleop_dataset import synchronized_rate_limit_position_target


DLS_LIMITER_PREVIEW_SCHEMA = "g2_dls_limiter_read_only_preview_v1"
DLS_LIMITER_PREVIEW_FRAME = "robot_root"
DLS_TARGET_PROVENANCE = (
    "G2RedundancyDifferentialIKAction.apply_actions:"
    "post_dls_nullspace_soft_limit_clamp"
)
PRODUCTION_LIMITER_SOURCE = (
    "geniesim.rl.isaaclab.g2_teleop_dataset."
    "synchronized_rate_limit_position_target"
)


class DLSLimiterPreviewError(ValueError):
    """Raised for a malformed immutable preview snapshot.

    Callers must treat this as fail-closed.  A moving DLS endpoint that cannot
    be reached exactly in one bounded acceleration step is *not* malformed:
    the source limiter emits its continuous bounded deceleration and records
    the condition in diagnostics.
    """


class PreviewAvailability(str, Enum):
    """Whether all source-bound inputs needed for the limiter are present."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE_REQUIRED_SOURCE_STATE"


class PreviewEquivalence(str, Enum):
    """Scope of the diagnostic receipt.

    This remains unqualified even when every numeric input is present: the
    future runtime binder must compare a preview to a following action-term
    execution at the same epoch before it can claim runtime equivalence.
    """

    UNQUALIFIED_STATIC_PREVIEW = "UNQUALIFIED_REQUIRES_RUNTIME_EQUIVALENCE"


def _finite_tuple(name: str, values: Sequence[float], size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise DLSLimiterPreviewError(f"{name} must contain {size} finite values")
    return result


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DLSLimiterPreviewError(f"{name} must be finite and positive")
    return result


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise DLSLimiterPreviewError("tensor_dtype must be float32 or float64")


@dataclass(frozen=True)
class DLSLimiterPreviewSnapshot:
    """An immutable, epoch-bound input to the source limiter.

    ``post_dls_soft_limit_target_rad`` is optional on purpose.  Existing G2
    runtime surfaces expose the limiter accumulator but not a side-effect-free
    DLS target query.  Omitting it yields an explicit unavailable receipt;
    supplying a planner joint target, a NumPy reproduction, or a target from a
    different epoch is prohibited by this contract.
    """

    control_epoch: int
    ordered_joint_names: tuple[str, ...]
    normalized_translation_xyz: tuple[float, float, float]
    translation_m_per_normalized: float
    frame: str
    measured_joint_position_rad: tuple[float, ...]
    previous_emitted_target_rad: tuple[float, ...]
    previous_target_velocity_rad_s: tuple[float, ...]
    target_initialized: bool
    post_dls_soft_limit_target_rad: tuple[float, ...] | None
    post_dls_target_provenance: str | None
    maximum_speed_rad_s: float
    maximum_acceleration_rad_s2: float
    physics_dt_s: float
    physics_substeps_per_policy_step: int
    tensor_dtype: str
    runtime_device: str
    schema: str = DLS_LIMITER_PREVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DLS_LIMITER_PREVIEW_SCHEMA:
            raise DLSLimiterPreviewError("unsupported preview schema")
        if not isinstance(self.control_epoch, int) or self.control_epoch < 0:
            raise DLSLimiterPreviewError("control_epoch must be a non-negative integer")
        if tuple(self.ordered_joint_names) != RIGHT_ARM_JOINTS:
            raise DLSLimiterPreviewError("right-arm joint ordering mismatch")
        if self.frame != DLS_LIMITER_PREVIEW_FRAME:
            raise DLSLimiterPreviewError("preview frame must be robot_root")
        translation = _finite_tuple(
            "normalized_translation_xyz", self.normalized_translation_xyz, 3
        )
        if any(abs(value) > 1.0 for value in translation):
            raise DLSLimiterPreviewError("normalized translation must be in [-1, 1]")
        object.__setattr__(self, "normalized_translation_xyz", translation)
        scale = _positive("translation_m_per_normalized", self.translation_m_per_normalized)
        if not math.isclose(scale, G2_TRANSLATION_ACTION_SCALE_M, rel_tol=0.0, abs_tol=0.0):
            raise DLSLimiterPreviewError("translation scale does not match selected G2 port")
        object.__setattr__(self, "translation_m_per_normalized", scale)
        for name in (
            "measured_joint_position_rad",
            "previous_emitted_target_rad",
            "previous_target_velocity_rad_s",
        ):
            object.__setattr__(self, name, _finite_tuple(name, getattr(self, name), 7))
        if not isinstance(self.target_initialized, bool):
            raise DLSLimiterPreviewError("target_initialized must be bool")
        if self.post_dls_soft_limit_target_rad is not None:
            object.__setattr__(
                self,
                "post_dls_soft_limit_target_rad",
                _finite_tuple(
                    "post_dls_soft_limit_target_rad",
                    self.post_dls_soft_limit_target_rad,
                    7,
                ),
            )
            if self.post_dls_target_provenance != DLS_TARGET_PROVENANCE:
                raise DLSLimiterPreviewError("post-DLS target provenance mismatch")
        elif self.post_dls_target_provenance is not None:
            raise DLSLimiterPreviewError("DLS provenance requires a DLS target")
        object.__setattr__(
            self, "maximum_speed_rad_s", _positive("maximum_speed_rad_s", self.maximum_speed_rad_s)
        )
        object.__setattr__(
            self,
            "maximum_acceleration_rad_s2",
            _positive("maximum_acceleration_rad_s2", self.maximum_acceleration_rad_s2),
        )
        object.__setattr__(self, "physics_dt_s", _positive("physics_dt_s", self.physics_dt_s))
        if self.physics_dt_s != 0.002:
            raise DLSLimiterPreviewError("preview must use the frozen 2 ms physics step")
        if self.physics_substeps_per_policy_step != 10:
            raise DLSLimiterPreviewError("preview must use ten physics substeps per policy step")
        _dtype(self.tensor_dtype)
        if not isinstance(self.runtime_device, str) or not self.runtime_device:
            raise DLSLimiterPreviewError("runtime_device must be a non-empty string")

    @property
    def requested_metric_delta_root_m(self) -> tuple[float, float, float]:
        return tuple(
            value * self.translation_m_per_normalized
            for value in self.normalized_translation_xyz
        )

    @property
    def availability(self) -> PreviewAvailability:
        return (
            PreviewAvailability.AVAILABLE
            if self.post_dls_soft_limit_target_rad is not None
            else PreviewAvailability.UNAVAILABLE
        )


@dataclass(frozen=True)
class DLSLimiterPreviewReceipt:
    """Result of one non-mutating source-limiter preview."""

    control_epoch: int
    availability: PreviewAvailability
    equivalence: PreviewEquivalence
    limiter_feasible: bool
    production_command_authorized: bool
    reason: str | None
    requested_metric_delta_root_m: tuple[float, float, float]
    effective_previous_target_rad: tuple[float, ...]
    effective_previous_velocity_rad_s: tuple[float, ...]
    post_dls_soft_limit_target_rad: tuple[float, ...] | None
    predicted_emitted_target_rad: tuple[float, ...] | None
    predicted_target_velocity_rad_s: tuple[float, ...] | None
    limiter_diagnostics: dict[str, tuple[float, ...] | tuple[bool, ...]]
    source_limiter: str = PRODUCTION_LIMITER_SOURCE
    schema: str = DLS_LIMITER_PREVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DLS_LIMITER_PREVIEW_SCHEMA:
            raise DLSLimiterPreviewError("unsupported preview receipt schema")
        if self.production_command_authorized:
            raise DLSLimiterPreviewError("diagnostic preview may not authorize a command")
        if self.availability is PreviewAvailability.UNAVAILABLE:
            if self.limiter_feasible or self.post_dls_soft_limit_target_rad is not None:
                raise DLSLimiterPreviewError("unavailable preview cannot claim limiter feasibility")
        if self.equivalence is not PreviewEquivalence.UNQUALIFIED_STATIC_PREVIEW:
            raise DLSLimiterPreviewError("preview equivalence scope cannot be promoted here")


def _tuple_tensor(value: tuple[float, ...], *, dtype: torch.dtype) -> torch.Tensor:
    """Materialize a local CPU copy; caller-owned state is never aliased."""

    return torch.tensor((value,), dtype=dtype, device="cpu").clone()


def _diagnostic_tuples(
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, tuple[float, ...] | tuple[bool, ...]]:
    result: dict[str, tuple[float, ...] | tuple[bool, ...]] = {}
    for name, value in diagnostics.items():
        flattened = value.detach().cpu().reshape(-1)
        if value.dtype == torch.bool:
            result[name] = tuple(bool(item) for item in flattened.tolist())
        else:
            result[name] = tuple(float(item) for item in flattened.tolist())
    return result


def preview_same_state_dls_limiter(
    snapshot: DLSLimiterPreviewSnapshot,
) -> DLSLimiterPreviewReceipt:
    """Preview exactly one limiter substep without mutating any live object.

    The DLS part of the requested boundary is represented by the immutable
    post-DLS/post-nullspace/post-soft-limit candidate in ``snapshot``.  The
    helper intentionally will not synthesize that candidate from a planner q,
    run ``process_actions``, call a controller setter, or call an articulation
    target writer.  Missing candidate state is therefore an explicit
    fail-closed receipt rather than an estimate.
    """

    if not isinstance(snapshot, DLSLimiterPreviewSnapshot):
        raise DLSLimiterPreviewError("snapshot must be DLSLimiterPreviewSnapshot")
    dtype = _dtype(snapshot.tensor_dtype)
    # This exactly matches the source action term's first-use initialization
    # branch at g2_redundancy_action.py:_synchronized_rate_limited_target.
    if snapshot.target_initialized:
        previous_target = snapshot.previous_emitted_target_rad
        previous_velocity = snapshot.previous_target_velocity_rad_s
    else:
        previous_target = snapshot.measured_joint_position_rad
        previous_velocity = (0.0,) * 7
    base = {
        "control_epoch": snapshot.control_epoch,
        "equivalence": PreviewEquivalence.UNQUALIFIED_STATIC_PREVIEW,
        "production_command_authorized": False,
        "requested_metric_delta_root_m": snapshot.requested_metric_delta_root_m,
        "effective_previous_target_rad": tuple(previous_target),
        "effective_previous_velocity_rad_s": tuple(previous_velocity),
    }
    if snapshot.availability is PreviewAvailability.UNAVAILABLE:
        return DLSLimiterPreviewReceipt(
            availability=PreviewAvailability.UNAVAILABLE,
            limiter_feasible=False,
            reason="REQUIRED_POST_DLS_SOURCE_STATE_UNAVAILABLE",
            post_dls_soft_limit_target_rad=None,
            predicted_emitted_target_rad=None,
            predicted_target_velocity_rad_s=None,
            limiter_diagnostics={},
            **base,
        )

    # The production limiter itself is pure on supplied tensors.  Use clones
    # even though it currently performs no in-place writes, so a future source
    # change cannot silently mutate caller-owned runtime state through this
    # diagnostic path.
    previous = _tuple_tensor(tuple(previous_target), dtype=dtype)
    velocity = _tuple_tensor(tuple(previous_velocity), dtype=dtype)
    desired = _tuple_tensor(snapshot.post_dls_soft_limit_target_rad, dtype=dtype)
    diagnostics: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        emitted, next_velocity = synchronized_rate_limit_position_target(
            previous.clone(),
            desired.clone(),
            velocity.clone(),
            maximum_speed_rad_s=float(snapshot.maximum_speed_rad_s),
            maximum_acceleration_rad_s2=float(snapshot.maximum_acceleration_rad_s2),
            physics_dt_s=float(snapshot.physics_dt_s),
            diagnostics=diagnostics,
        )
    return DLSLimiterPreviewReceipt(
        availability=PreviewAvailability.AVAILABLE,
        limiter_feasible=True,
        reason=None,
        post_dls_soft_limit_target_rad=snapshot.post_dls_soft_limit_target_rad,
        predicted_emitted_target_rad=tuple(float(item) for item in emitted[0].tolist()),
        predicted_target_velocity_rad_s=tuple(
            float(item) for item in next_velocity[0].tolist()
        ),
        limiter_diagnostics=_diagnostic_tuples(diagnostics),
        **base,
    )


__all__ = [
    "DLS_LIMITER_PREVIEW_FRAME",
    "DLS_LIMITER_PREVIEW_SCHEMA",
    "DLS_TARGET_PROVENANCE",
    "DLSLimiterPreviewError",
    "DLSLimiterPreviewReceipt",
    "DLSLimiterPreviewSnapshot",
    "PreviewAvailability",
    "PreviewEquivalence",
    "preview_same_state_dls_limiter",
]
