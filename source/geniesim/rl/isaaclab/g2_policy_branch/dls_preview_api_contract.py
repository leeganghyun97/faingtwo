# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Versioned, side-effect-free API contract for a G2 DLS/limiter preview.

This module is the *public compatibility boundary* around
``dls_limiter_preview``.  It deliberately does not make a controller target,
call an Isaac API, or grant a command admission.  A future runtime binder may
only use it after it has cloned a same-epoch source snapshot.  Until that
binder is separately qualified, every receipt remains an unqualified static
preview.

The contract makes the otherwise easy-to-confuse representations explicit:

* policy action: ``[dx, dy, dz, HOLD_OPEN]`` in robot-root metres;
* production arm input: normalized ``[xyz, rotvec, elbow]`` (7-D);
* full ActionManager packet: arm7 followed by existing simulation ``OPEN=+1``;
* EE pose: metres plus a unit ``XYZW`` quaternion;
* joints/Jacobian: rad, rad/s, rad/s² and ``[6, 7]`` respectively.

It is intentionally fail-closed about source state, actor-input leakage,
scale ownership, frames, dtype/device identity, and side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from typing import Mapping, Sequence

from ..g2_lift_methodology import RIGHT_ARM_JOINTS
from ..g2_teleop_dataset import G2_TRANSLATION_ACTION_SCALE_M
from .curobo_contact_free_collection import assert_deployable_actor_input_inventory
from .dls_limiter_preview import (
    DLSLimiterPreviewSnapshot,
    PreviewAvailability,
)


DLS_PREVIEW_API_SCHEMA = "g2_dls_limiter_preview_api_v1"
DLS_PREVIEW_API_FRAME = "robot_root"
DLS_PREVIEW_API_QUATERNION_ORDER = "xyzw"
DLS_PREVIEW_API_NATIVE_QUATERNION_ORDERS = ("xyzw", "wxyz")
DLS_PREVIEW_API_NATIVE_QUATERNION_CONVERSION_SOURCE = (
    "geniesim.rl.isaaclab.g2_quaternion.quaternion_xyzw_to_native"
)
DLS_PREVIEW_API_DTYPE = "float32"
DLS_PREVIEW_API_ACTION_DIM = 4
DLS_PREVIEW_API_ARM_DIM = 7
DLS_PREVIEW_API_PACKET_DIM = 8
DLS_PREVIEW_API_MAX_DELTA_M = 0.0045
DLS_PREVIEW_API_TRANSLATION_SCALE_M = G2_TRANSLATION_ACTION_SCALE_M
DLS_PREVIEW_API_PHYSICS_DT_S = 0.002
DLS_PREVIEW_API_SUBSTEPS = 10
DLS_PREVIEW_API_CONTROL_PERIOD_S = 0.020
DLS_PREVIEW_API_CONTROL_RATE_HZ = 50.0
REQUIRED_SOURCE_HASH_KEYS = (
    "g2_redundancy_action_source",
    "g2_teleop_dataset_source",
    "right_arm_joint_order_source",
    "dls_preview_api_contract_source",
)


class DLSPreviewAPIContractError(ValueError):
    """Raised for a malformed, ambiguous, or unauthorized preview request."""


class PreviewRuntimeEquivalence(str, Enum):
    """The static API cannot assert equality to a future controller apply."""

    UNQUALIFIED = "UNQUALIFIED_REQUIRES_SAME_EPOCH_RUNTIME_COMPARISON"


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DLSPreviewAPIContractError(f"{name} must be a real number, not an implicit conversion")
    result = float(value)
    if not math.isfinite(result):
        raise DLSPreviewAPIContractError(f"{name} must be finite")
    return result


def _finite_tuple(name: str, values: Sequence[object], width: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise DLSPreviewAPIContractError(f"{name} must be an explicit numeric sequence")
    result = tuple(_finite_real(f"{name}[{index}]", value) for index, value in enumerate(values))
    if len(result) != width:
        raise DLSPreviewAPIContractError(f"{name} must have exact width {width}")
    return result


def _sha256_pairs(value: Mapping[str, str] | Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    else:
        pairs = tuple(value)
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise DLSPreviewAPIContractError("source_hashes must contain (name, sha256) pairs")
        name, digest = item
        if not isinstance(name, str) or not name or name in names:
            raise DLSPreviewAPIContractError("source hash names must be unique non-empty strings")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise DLSPreviewAPIContractError(f"source hash {name} must be a full SHA-256 digest")
        names.add(name)
        result.append((name, digest.lower()))
    missing = sorted(set(REQUIRED_SOURCE_HASH_KEYS).difference(names))
    if missing:
        raise DLSPreviewAPIContractError(f"source hash receipt is missing required keys: {missing}")
    return tuple(sorted(result))


def _action_hash(
    *,
    epoch_id: str,
    control_epoch: int,
    metric_action_4d: tuple[float, float, float, float],
    source_hashes: tuple[tuple[str, str], ...],
) -> str:
    """Hash only canonical representation fields with strict JSON encoding."""

    payload = {
        "control_epoch": control_epoch,
        "epoch_id": epoch_id,
        "frame": DLS_PREVIEW_API_FRAME,
        "metric_action_4d_robot_root_m": list(metric_action_4d),
        "schema": DLS_PREVIEW_API_SCHEMA,
        "source_hashes": list(source_hashes),
        "translation_m_per_normalized": DLS_PREVIEW_API_TRANSLATION_SCALE_M,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _canonical_xyzw(name: str, quaternion: Sequence[object]) -> tuple[float, float, float, float]:
    """Canonicalize the persisted XYZW boundary without guessing native order."""

    raw = _finite_tuple(name, quaternion, 4)
    norm = math.sqrt(sum(value * value for value in raw))
    if norm < 1.0e-8:
        raise DLSPreviewAPIContractError(f"{name} has a zero quaternion norm")
    values = tuple(value / norm for value in raw)
    scalar = values[3]
    dominant = values[max(range(3), key=lambda index: abs(values[index]))]
    if (abs(scalar) > 1.0e-8 and scalar < 0.0) or (abs(scalar) <= 1.0e-8 and dominant < 0.0):
        values = tuple(-value for value in values)
    return values  # type: ignore[return-value]


def canonical_xyzw_to_native(
    quaternion_xyzw: Sequence[object], *, native_order: str
) -> tuple[float, float, float, float]:
    """Pure tuple mirror of the source-backed XYZW-to-native boundary reorder."""

    canonical = _canonical_xyzw("quaternion_xyzw", quaternion_xyzw)
    if native_order == "xyzw":
        return canonical
    if native_order == "wxyz":
        return canonical[3], canonical[0], canonical[1], canonical[2]
    raise DLSPreviewAPIContractError("native DLS quaternion order is unresolved or unsupported")


def native_to_canonical_xyzw(
    quaternion_native: Sequence[object], *, native_order: str
) -> tuple[float, float, float, float]:
    native = _finite_tuple("quaternion_native", quaternion_native, 4)
    if native_order == "xyzw":
        return _canonical_xyzw("quaternion_native_xyzw", native)
    if native_order == "wxyz":
        return _canonical_xyzw("quaternion_native_wxyz", (native[1], native[2], native[3], native[0]))
    raise DLSPreviewAPIContractError("native DLS quaternion order is unresolved or unsupported")


@dataclass(frozen=True)
class DLSPreviewTensorIdentity:
    """Exact dtype/device identity that the eventual runtime binder must clone."""

    dtype: str
    device: str

    def __post_init__(self) -> None:
        if self.dtype != DLS_PREVIEW_API_DTYPE:
            raise DLSPreviewAPIContractError(
                "preview API only accepts exact runtime float32; float64 mirrors are not authority"
            )
        if not isinstance(self.device, str) or not self.device or self.device != self.device.strip():
            raise DLSPreviewAPIContractError("tensor device must be an explicit non-empty runtime descriptor")


@dataclass(frozen=True)
class DLSPreviewLimiterConfiguration:
    """Source-owned limiter parameters copied into an immutable request.

    The API does not manufacture speed, acceleration, damping, or soft-limit
    settings.  The caller must provide their source/config provenance and a
    full configuration hash from the selected action-term instance.
    """

    maximum_speed_rad_s: float
    maximum_acceleration_rad_s2: float
    soft_limit_margin_rad: float
    dls_damping: float
    source_provenance: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "maximum_speed_rad_s",
            "maximum_acceleration_rad_s2",
            "soft_limit_margin_rad",
            "dls_damping",
        ):
            value = _finite_real(name, getattr(self, name))
            if value <= 0.0:
                raise DLSPreviewAPIContractError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.source_provenance, str) or not self.source_provenance:
            raise DLSPreviewAPIContractError("limiter source provenance is required")
        _sha256_pairs((("limiter_configuration", self.configuration_sha256), *(
            (name, "0" * 64) for name in REQUIRED_SOURCE_HASH_KEYS
        )))


@dataclass(frozen=True)
class ContactFreeMetricOpenAction:
    """The only high-level action accepted by this contact-free preview API."""

    metric_delta_root_m: tuple[float, float, float]
    hold_open: float
    frame: str = DLS_PREVIEW_API_FRAME

    def __post_init__(self) -> None:
        metric = _finite_tuple("metric_delta_root_m", self.metric_delta_root_m, 3)
        if self.frame != DLS_PREVIEW_API_FRAME:
            raise DLSPreviewAPIContractError("metric policy action frame must be robot_root")
        if math.sqrt(sum(value * value for value in metric)) > DLS_PREVIEW_API_MAX_DELTA_M + 1.0e-12:
            raise DLSPreviewAPIContractError("metric policy action exceeds the 4.5 mm Euclidean bound")
        if _finite_real("hold_open", self.hold_open) != 0.0:
            raise DLSPreviewAPIContractError("contact-free API accepts only exact HOLD_OPEN=0")
        object.__setattr__(self, "metric_delta_root_m", metric)
        object.__setattr__(self, "hold_open", 0.0)

    @property
    def values_4d(self) -> tuple[float, float, float, float]:
        return (*self.metric_delta_root_m, self.hold_open)

    @property
    def normalized_arm7(self) -> tuple[float, float, float, float, float, float, float]:
        normalized = tuple(value / DLS_PREVIEW_API_TRANSLATION_SCALE_M for value in self.metric_delta_root_m)
        if any(abs(value) > 1.0 for value in normalized):  # defensive: bound above is stronger.
            raise DLSPreviewAPIContractError("metric action cannot be represented by the selected normalized arm port")
        return (*normalized, 0.0, 0.0, 0.0, 0.0)

    @property
    def full_8d_open_packet(self) -> tuple[float, float, float, float, float, float, float, float]:
        # The high-level HOLD_OPEN=0 semantic maps once to the existing
        # simulation ActionManager OPEN sign (+1), not to a joint target.
        return (*self.normalized_arm7, 1.0)


@dataclass(frozen=True)
class DLSPreviewAPIRequest:
    """One same-epoch, read-only DLS/limiter API request.

    ``limiter_snapshot`` contains the only accepted post-DLS target authority:
    a source-provenanced result from the selected action term.  If that target
    is absent, the request is valid but produces an explicit unavailable
    receipt; a planner joint target or offline DLS imitation is rejected by
    the underlying preview contract.
    """

    epoch_id: str
    control_epoch: int
    policy_action: ContactFreeMetricOpenAction
    actor_input_fields: tuple[str, ...]
    measured_ee_pose_robot_root_m_xyzw: tuple[float, float, float, float, float, float, float]
    native_dls_quaternion_order: str
    native_dls_ee_quaternion: tuple[float, float, float, float]
    ordered_right_arm_joint_names: tuple[str, ...]
    joint_position_rad: tuple[float, float, float, float, float, float, float]
    joint_velocity_rad_s: tuple[float, float, float, float, float, float, float]
    joint_acceleration_rad_s2: tuple[float, float, float, float, float, float, float]
    frame_jacobian_6x7: tuple[tuple[float, ...], ...]
    tensor_identity: DLSPreviewTensorIdentity
    limiter_configuration: DLSPreviewLimiterConfiguration
    limiter_snapshot: DLSLimiterPreviewSnapshot
    source_hashes: Mapping[str, str] | tuple[tuple[str, str], ...]
    physics_dt_s: float = DLS_PREVIEW_API_PHYSICS_DT_S
    physics_substeps_per_policy_step: int = DLS_PREVIEW_API_SUBSTEPS
    control_period_s: float = DLS_PREVIEW_API_CONTROL_PERIOD_S
    control_rate_hz: float = DLS_PREVIEW_API_CONTROL_RATE_HZ
    schema: str = DLS_PREVIEW_API_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DLS_PREVIEW_API_SCHEMA:
            raise DLSPreviewAPIContractError("unsupported DLS preview API schema")
        if not isinstance(self.epoch_id, str) or not self.epoch_id:
            raise DLSPreviewAPIContractError("epoch_id must be a non-empty string")
        if type(self.control_epoch) is not int or self.control_epoch < 0:
            raise DLSPreviewAPIContractError("control_epoch must be a non-negative integer")
        if not isinstance(self.policy_action, ContactFreeMetricOpenAction):
            raise DLSPreviewAPIContractError("policy_action must be ContactFreeMetricOpenAction")
        # This rejects cube GT, contact/phase labels, planner state, and any
        # future/outcome field through the central deployable inventory.
        assert_deployable_actor_input_inventory(self.actor_input_fields)
        pose = _finite_tuple("measured_ee_pose_robot_root_m_xyzw", self.measured_ee_pose_robot_root_m_xyzw, 7)
        canonical_quaternion = _canonical_xyzw("measured_ee_pose_robot_root_m_xyzw[3:7]", pose[3:])
        object.__setattr__(self, "measured_ee_pose_robot_root_m_xyzw", (*pose[:3], *canonical_quaternion))
        if self.native_dls_quaternion_order not in DLS_PREVIEW_API_NATIVE_QUATERNION_ORDERS:
            raise DLSPreviewAPIContractError("native DLS quaternion order must be source-attested xyzw or wxyz")
        native = _finite_tuple("native_dls_ee_quaternion", self.native_dls_ee_quaternion, 4)
        observed_canonical = native_to_canonical_xyzw(native, native_order=self.native_dls_quaternion_order)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6)
            for actual, expected in zip(observed_canonical, canonical_quaternion, strict=True)
        ):
            raise DLSPreviewAPIContractError(
                "persisted XYZW and native DifferentialIK quaternion disagree after source-backed reorder"
            )
        object.__setattr__(self, "native_dls_ee_quaternion", native)
        if tuple(self.ordered_right_arm_joint_names) != RIGHT_ARM_JOINTS:
            raise DLSPreviewAPIContractError("right-arm joint ordering differs from the selected runtime authority")
        for name in ("joint_position_rad", "joint_velocity_rad_s", "joint_acceleration_rad_s2"):
            object.__setattr__(self, name, _finite_tuple(name, getattr(self, name), 7))
        rows = tuple(_finite_tuple(f"frame_jacobian_6x7[{index}]", row, 7) for index, row in enumerate(self.frame_jacobian_6x7))
        if len(rows) != 6:
            raise DLSPreviewAPIContractError("frame Jacobian must have exact shape [6,7]")
        object.__setattr__(self, "frame_jacobian_6x7", rows)
        if not isinstance(self.tensor_identity, DLSPreviewTensorIdentity):
            raise DLSPreviewAPIContractError("tensor_identity is required")
        if not isinstance(self.limiter_configuration, DLSPreviewLimiterConfiguration):
            raise DLSPreviewAPIContractError("limiter_configuration is required")
        if not isinstance(self.limiter_snapshot, DLSLimiterPreviewSnapshot):
            raise DLSPreviewAPIContractError("limiter_snapshot must be the source preview snapshot")
        _validate_snapshot_compatibility(self)
        hashes = _sha256_pairs(self.source_hashes)
        object.__setattr__(self, "source_hashes", hashes)
        physics_dt_s = _finite_real("physics_dt_s", self.physics_dt_s)
        if physics_dt_s != DLS_PREVIEW_API_PHYSICS_DT_S:
            raise DLSPreviewAPIContractError("physics dt must be exactly 0.002 s")
        if self.physics_substeps_per_policy_step != DLS_PREVIEW_API_SUBSTEPS:
            raise DLSPreviewAPIContractError("policy step must contain exactly 10 physics substeps")
        control_period_s = _finite_real("control_period_s", self.control_period_s)
        control_rate_hz = _finite_real("control_rate_hz", self.control_rate_hz)
        if not math.isclose(control_period_s, physics_dt_s * self.physics_substeps_per_policy_step, rel_tol=0.0, abs_tol=1.0e-12):
            raise DLSPreviewAPIContractError("control period does not equal physics dt times substeps")
        if not math.isclose(control_rate_hz, 1.0 / control_period_s, rel_tol=0.0, abs_tol=1.0e-12):
            raise DLSPreviewAPIContractError("control rate does not equal inverse control period")
        if control_rate_hz != DLS_PREVIEW_API_CONTROL_RATE_HZ:
            raise DLSPreviewAPIContractError("control rate must be exactly 50 Hz")
        object.__setattr__(self, "physics_dt_s", physics_dt_s)
        object.__setattr__(self, "control_period_s", control_period_s)
        object.__setattr__(self, "control_rate_hz", control_rate_hz)

    @property
    def action_hash(self) -> str:
        return _action_hash(
            epoch_id=self.epoch_id,
            control_epoch=self.control_epoch,
            metric_action_4d=self.policy_action.values_4d,
            source_hashes=self.source_hashes,
        )


def _validate_snapshot_compatibility(request: DLSPreviewAPIRequest) -> None:
    snapshot = request.limiter_snapshot
    if snapshot.control_epoch != request.control_epoch:
        raise DLSPreviewAPIContractError("limiter snapshot is from a different control epoch")
    if snapshot.frame != DLS_PREVIEW_API_FRAME:
        raise DLSPreviewAPIContractError("limiter snapshot frame mismatch")
    if snapshot.tensor_dtype != request.tensor_identity.dtype or snapshot.runtime_device != request.tensor_identity.device:
        raise DLSPreviewAPIContractError("limiter snapshot dtype/device identity mismatch")
    if snapshot.ordered_joint_names != RIGHT_ARM_JOINTS:
        raise DLSPreviewAPIContractError("limiter snapshot joint order mismatch")
    expected_normalized = request.policy_action.normalized_arm7[:3]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(snapshot.normalized_translation_xyz, expected_normalized, strict=True)
    ):
        raise DLSPreviewAPIContractError("metric-to-normalized scale is absent, duplicated, or inconsistent")
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(snapshot.measured_joint_position_rad, request.joint_position_rad, strict=True)
    ):
        raise DLSPreviewAPIContractError("limiter snapshot joint position does not match request")
    if snapshot.physics_dt_s != request.physics_dt_s or snapshot.physics_substeps_per_policy_step != request.physics_substeps_per_policy_step:
        raise DLSPreviewAPIContractError("limiter snapshot timing mismatch")
    config = request.limiter_configuration
    if (
        snapshot.maximum_speed_rad_s != config.maximum_speed_rad_s
        or snapshot.maximum_acceleration_rad_s2 != config.maximum_acceleration_rad_s2
    ):
        raise DLSPreviewAPIContractError("limiter snapshot differs from source-owned configuration")


@dataclass(frozen=True)
class DLSPreviewAPIReceipt:
    """Immutable evidence returned by :func:`preview_dls_limiter_api`.

    ``command_authorized`` is deliberately fixed ``False``.  This receipt is
    a static numerical comparison artifact, never an actuator permission.
    """

    epoch_id: str
    control_epoch: int
    action_hash: str
    source_hashes: tuple[tuple[str, str], ...]
    metric_action_4d_robot_root_m: tuple[float, float, float, float]
    normalized_arm7: tuple[float, float, float, float, float, float, float]
    full_8d_packet: tuple[float, float, float, float, float, float, float, float]
    ee_pose_robot_root_m_xyzw: tuple[float, float, float, float, float, float, float]
    native_dls_quaternion_order: str
    native_dls_ee_quaternion: tuple[float, float, float, float]
    ordered_right_arm_joint_names: tuple[str, ...]
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_acceleration_rad_s2: tuple[float, ...]
    frame_jacobian_6x7: tuple[tuple[float, ...], ...]
    tensor_dtype: str
    runtime_device: str
    physics_dt_s: float
    control_period_s: float
    control_rate_hz: float
    preview_availability: PreviewAvailability
    limiter_feasible: bool
    infeasible_reason: str | None
    effective_previous_target_rad: tuple[float, ...]
    effective_previous_velocity_rad_s: tuple[float, ...]
    post_dls_soft_limit_target_rad: tuple[float, ...] | None
    predicted_emitted_target_rad: tuple[float, ...] | None
    predicted_target_velocity_rad_s: tuple[float, ...] | None
    predicted_target_acceleration_rad_s2: tuple[float, ...] | None
    limiter_diagnostics: tuple[tuple[str, tuple[float | bool, ...]], ...]
    command_authorized: bool = False
    no_side_effects_attested: bool = True
    runtime_equivalence: PreviewRuntimeEquivalence = PreviewRuntimeEquivalence.UNQUALIFIED
    schema: str = DLS_PREVIEW_API_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DLS_PREVIEW_API_SCHEMA:
            raise DLSPreviewAPIContractError("unsupported preview receipt schema")
        if self.command_authorized or not self.no_side_effects_attested:
            raise DLSPreviewAPIContractError("read-only preview cannot authorize or mutate a command")
        if self.runtime_equivalence is not PreviewRuntimeEquivalence.UNQUALIFIED:
            raise DLSPreviewAPIContractError("static preview cannot claim runtime equivalence")
        if self.preview_availability is PreviewAvailability.UNAVAILABLE and self.limiter_feasible:
            raise DLSPreviewAPIContractError("unavailable preview cannot claim feasibility")


@dataclass(frozen=True)
class DLSPreviewAPIManifest:
    """Machine-readable, versioned API contract used by all specialist modules."""

    schema: str = DLS_PREVIEW_API_SCHEMA
    policy_action: str = "[dx_m,dy_m,dz_m,HOLD_OPEN=0]"
    arm_action: str = "[dx_norm,dy_norm,dz_norm,rx=0,ry=0,rz=0,elbow=0]"
    full_packet: str = "[arm7,existing_simulation_open_sign=+1]"
    frame: str = DLS_PREVIEW_API_FRAME
    quaternion_order: str = DLS_PREVIEW_API_QUATERNION_ORDER
    joint_position_unit: str = "rad"
    joint_velocity_unit: str = "rad/s"
    joint_acceleration_unit: str = "rad/s^2"
    jacobian_shape: str = "[6,7]"
    dtype: str = DLS_PREVIEW_API_DTYPE
    translation_m_per_normalized: float = DLS_PREVIEW_API_TRANSLATION_SCALE_M
    max_metric_delta_m: float = DLS_PREVIEW_API_MAX_DELTA_M
    physics_dt_s: float = DLS_PREVIEW_API_PHYSICS_DT_S
    physics_substeps: int = DLS_PREVIEW_API_SUBSTEPS
    control_rate_hz: float = DLS_PREVIEW_API_CONTROL_RATE_HZ
    runtime_equivalence: str = PreviewRuntimeEquivalence.UNQUALIFIED.value

    def as_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "right_arm_joint_order": list(RIGHT_ARM_JOINTS),
            "required_source_hash_keys": list(REQUIRED_SOURCE_HASH_KEYS),
            "native_dls_quaternion_orders": list(DLS_PREVIEW_API_NATIVE_QUATERNION_ORDERS),
            "native_dls_quaternion_conversion_source": DLS_PREVIEW_API_NATIVE_QUATERNION_CONVERSION_SOURCE,
            "actor_inputs_must_be": "exact deployable inventory; GT/planner/future/contact/phase rejected",
            "command_authorized": False,
        }


def preview_api_manifest() -> DLSPreviewAPIManifest:
    """Return a fresh immutable declaration; no runtime/controller lookup occurs."""

    return DLSPreviewAPIManifest()


def _freeze_diagnostics(
    diagnostics: Mapping[str, tuple[float, ...] | tuple[bool, ...]],
) -> tuple[tuple[str, tuple[float | bool, ...]], ...]:
    frozen: list[tuple[str, tuple[float | bool, ...]]] = []
    for name, values in sorted(diagnostics.items()):
        if not isinstance(name, str) or not name:
            raise DLSPreviewAPIContractError("limiter diagnostic names must be non-empty strings")
        frozen_values: list[float | bool] = []
        for index, value in enumerate(values):
            if isinstance(value, bool):
                frozen_values.append(value)
            else:
                frozen_values.append(_finite_real(f"limiter_diagnostics[{name}][{index}]", value))
        frozen.append((name, tuple(frozen_values)))
    return tuple(frozen)


def _predicted_acceleration(
    *,
    previous_velocity: tuple[float, ...],
    next_velocity: tuple[float, ...] | None,
    physics_dt_s: float,
) -> tuple[float, ...] | None:
    if next_velocity is None:
        return None
    return tuple((next_value - previous_value) / physics_dt_s for next_value, previous_value in zip(next_velocity, previous_velocity, strict=True))


def validate_preview_request_contract(request: DLSPreviewAPIRequest) -> str:
    """Validate only the public request boundary and return its stable hash.

    The numerical execution API is intentionally owned by
    :mod:`dls_read_only_preview_api`.  This module is a reusable schema and
    compatibility validator; it never replays DLS/limiter math.
    """

    if not isinstance(request, DLSPreviewAPIRequest):
        raise DLSPreviewAPIContractError("request must be DLSPreviewAPIRequest")
    return request.action_hash


def assert_preview_receipt_matches_request_contract(
    request: DLSPreviewAPIRequest,
    receipt: DLSPreviewAPIReceipt,
) -> None:
    """Validate receipt identity without becoming a second numerical API."""

    if receipt.action_hash != request.action_hash or receipt.control_epoch != request.control_epoch:
        raise DLSPreviewAPIContractError("receipt does not belong to the requested action epoch")
    if receipt.source_hashes != request.source_hashes:
        raise DLSPreviewAPIContractError("receipt source hashes differ from request source hashes")
    if receipt.metric_action_4d_robot_root_m != request.policy_action.values_4d:
        raise DLSPreviewAPIContractError("receipt changed metric policy action representation")
    if receipt.normalized_arm7 != request.policy_action.normalized_arm7 or receipt.full_8d_packet != request.policy_action.full_8d_open_packet:
        raise DLSPreviewAPIContractError("receipt changed scale, arm7, or full8 OPEN mapping")
    if receipt.native_dls_quaternion_order != request.native_dls_quaternion_order or receipt.native_dls_ee_quaternion != request.native_dls_ee_quaternion:
        raise DLSPreviewAPIContractError("receipt native quaternion boundary mismatch")
    if receipt.command_authorized or not receipt.no_side_effects_attested:
        raise DLSPreviewAPIContractError("receipt incorrectly claims command authority or side effects")


__all__ = [
    "ContactFreeMetricOpenAction",
    "DLS_PREVIEW_API_ACTION_DIM",
    "DLS_PREVIEW_API_ARM_DIM",
    "DLS_PREVIEW_API_CONTROL_PERIOD_S",
    "DLS_PREVIEW_API_CONTROL_RATE_HZ",
    "DLS_PREVIEW_API_DTYPE",
    "DLS_PREVIEW_API_FRAME",
    "DLS_PREVIEW_API_MAX_DELTA_M",
    "DLS_PREVIEW_API_NATIVE_QUATERNION_CONVERSION_SOURCE",
    "DLS_PREVIEW_API_NATIVE_QUATERNION_ORDERS",
    "DLS_PREVIEW_API_PACKET_DIM",
    "DLS_PREVIEW_API_PHYSICS_DT_S",
    "DLS_PREVIEW_API_QUATERNION_ORDER",
    "DLS_PREVIEW_API_SCHEMA",
    "DLS_PREVIEW_API_SUBSTEPS",
    "DLS_PREVIEW_API_TRANSLATION_SCALE_M",
    "DLSPreviewAPIContractError",
    "DLSPreviewAPIManifest",
    "DLSPreviewAPIReceipt",
    "DLSPreviewAPIRequest",
    "DLSPreviewLimiterConfiguration",
    "DLSPreviewTensorIdentity",
    "PreviewRuntimeEquivalence",
    "REQUIRED_SOURCE_HASH_KEYS",
    "assert_preview_receipt_matches_request_contract",
    "canonical_xyzw_to_native",
    "native_to_canonical_xyzw",
    "preview_api_manifest",
    "validate_preview_request_contract",
]
