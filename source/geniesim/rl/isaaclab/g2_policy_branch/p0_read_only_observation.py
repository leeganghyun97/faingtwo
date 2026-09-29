# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable read-only observation contract for the G2 P0 integration path.

This module deliberately does *not* implement a safety/admission authority.
It only gives a future, already-bound runtime adapter a narrow way to capture
the current high-level controller cache and articulation observations without
calling a controller, changing an action buffer, or advancing physics.

The snapshot is useful for integration evidence (for example, proving that a
policy observation refers to one control epoch).  It must never be treated as
an ACCEPT/REJECT receipt, an IK feasibility result, a collision query, or a
prediction of active/passive motion.

The module imports neither Isaac Sim nor Torch.  A runtime-specific adapter
has to materialize ordinary Python scalars/tuples from its public read-only
surfaces before handing them to :class:`P0IntegrationObservationProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Protocol, Sequence, runtime_checkable


P0_READ_ONLY_INTEGRATION_OBSERVATION_SCHEMA = "g2_p0_read_only_integration_observation_v1"
P0_INTEGRATION_OBSERVATION_AUTHORITY = "NO_ACCEPTANCE_AUTHORITY"
P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS = (
    "existing_high_level_controller_command_state_open0_closed1_v1"
)
P0_ROOT_FRAME = "ROBOT_ROOT"
P0_WORLD_FRAME = "WORLD"


class P0IntegrationObservationError(ValueError):
    """Raised when a purported read-only integration observation is invalid."""


class AbstractGripperCommandState(str, Enum):
    """Existing high-level gripper command state, not mechanical feedback."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


def _finite_scalar(name: str, value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise P0IntegrationObservationError(f"{name} must be numeric") from error
    if not math.isfinite(numeric):
        raise P0IntegrationObservationError(f"{name} must be finite")
    return numeric


def _float_tuple(name: str, value: Sequence[object], size: int) -> tuple[float, ...]:
    try:
        values = tuple(_finite_scalar(name, item) for item in value)
    except TypeError as error:
        raise P0IntegrationObservationError(f"{name} must be a sequence") from error
    if len(values) != size:
        raise P0IntegrationObservationError(f"{name} must have length {size}")
    return values


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise P0IntegrationObservationError(f"{name} must be a lowercase SHA-256 value")
    return value


@dataclass(frozen=True)
class P0IntegrationPose:
    """A finite pose copied from one existing runtime frame.

    ``frame`` is explicit so the snapshot cannot silently equate root-relative
    and world-frame data.  This is an observation only; it is not an FK query.
    """

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _float_tuple("position_m", self.position_m, 3))
        quaternion = _float_tuple("quaternion_xyzw", self.quaternion_xyzw, 4)
        if math.sqrt(sum(component * component for component in quaternion)) <= 0.0:
            raise P0IntegrationObservationError("quaternion_xyzw must be non-zero")
        if self.frame not in (P0_ROOT_FRAME, P0_WORLD_FRAME):
            raise P0IntegrationObservationError("pose frame must be ROBOT_ROOT or WORLD")
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    def payload(self) -> dict[str, object]:
        return {
            "position_m": list(self.position_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
            "frame": self.frame,
        }


@dataclass(frozen=True)
class P0IntegrationTargetCache:
    """Read-only copy of existing G2 Cartesian endpoint bookkeeping.

    The fields correspond to public inspection properties of
    ``G2RedundancyDifferentialIKAction``.  Capturing them does not make the
    existing endpoint queue clamp an admission rule.
    """

    desired_ee_pose_root: P0IntegrationPose
    pose_target_active: bool
    active_translation_axis: int
    maximum_outstanding_translation_axis_error_m: float
    elbow_normalized: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.desired_ee_pose_root, P0IntegrationPose):
            raise P0IntegrationObservationError("desired_ee_pose_root must be P0IntegrationPose")
        if self.desired_ee_pose_root.frame != P0_ROOT_FRAME:
            raise P0IntegrationObservationError("desired_ee_pose_root must use ROBOT_ROOT")
        if not isinstance(self.pose_target_active, bool):
            raise P0IntegrationObservationError("pose_target_active must be bool")
        # ``-2`` is the source-defined sentinel for a simultaneous multi-axis
        # policy translation.  ``-1`` is neutral/no active translation and
        # ``0..2`` are the keyboard-style single XYZ axes.  This field is
        # descriptive read-only state, not an admission or safety rule.
        if not isinstance(self.active_translation_axis, int) or not -2 <= self.active_translation_axis <= 2:
            raise P0IntegrationObservationError(
                "active_translation_axis must be one of -2, -1, 0, 1, 2"
            )
        maximum = _finite_scalar(
            "maximum_outstanding_translation_axis_error_m",
            self.maximum_outstanding_translation_axis_error_m,
        )
        if maximum <= 0.0:
            raise P0IntegrationObservationError(
                "maximum_outstanding_translation_axis_error_m must be positive"
            )
        object.__setattr__(self, "maximum_outstanding_translation_axis_error_m", maximum)
        object.__setattr__(self, "elbow_normalized", _finite_scalar("elbow_normalized", self.elbow_normalized))

    def payload(self) -> dict[str, object]:
        return {
            "desired_ee_pose_root": self.desired_ee_pose_root.payload(),
            "pose_target_active": self.pose_target_active,
            "active_translation_axis": self.active_translation_axis,
            "maximum_outstanding_translation_axis_error_m": self.maximum_outstanding_translation_axis_error_m,
            "elbow_normalized": self.elbow_normalized,
        }


@dataclass(frozen=True)
class P0IntegrationObservationBinding:
    """Static identity facts that bind an observation to the selected port.

    Fingerprints document source/config/asset provenance.  They are not a
    safety endorsement and they never cause a command to be accepted.
    """

    binding_id: str
    ordered_right_arm_joint_names: tuple[str, ...]
    root_body_name: str
    ee_body_name: str
    source_fingerprint: str
    controller_config_fingerprint: str
    production_usd_fingerprint: str
    gripper_command_state_semantics: str = P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise P0IntegrationObservationError("binding_id must be non-empty")
        names = tuple(self.ordered_right_arm_joint_names)
        if len(names) != 7 or len(set(names)) != 7 or any(not isinstance(name, str) or not name for name in names):
            raise P0IntegrationObservationError("ordered_right_arm_joint_names must be seven unique names")
        if not isinstance(self.root_body_name, str) or not self.root_body_name:
            raise P0IntegrationObservationError("root_body_name must be non-empty")
        if not isinstance(self.ee_body_name, str) or not self.ee_body_name:
            raise P0IntegrationObservationError("ee_body_name must be non-empty")
        _sha256("source_fingerprint", self.source_fingerprint)
        _sha256("controller_config_fingerprint", self.controller_config_fingerprint)
        _sha256("production_usd_fingerprint", self.production_usd_fingerprint)
        if self.gripper_command_state_semantics != P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS:
            raise P0IntegrationObservationError("unsupported abstract gripper command-state semantics")
        object.__setattr__(self, "ordered_right_arm_joint_names", names)

    def payload(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "ordered_right_arm_joint_names": list(self.ordered_right_arm_joint_names),
            "root_body_name": self.root_body_name,
            "ee_body_name": self.ee_body_name,
            "source_fingerprint": self.source_fingerprint,
            "controller_config_fingerprint": self.controller_config_fingerprint,
            "production_usd_fingerprint": self.production_usd_fingerprint,
            "gripper_command_state_semantics": self.gripper_command_state_semantics,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class P0IntegrationObservationSample:
    """Fresh values copied by a runtime-specific read-only source.

    The runtime source is responsible for using public observation surfaces
    and must not call a target setter, action processing, controller function,
    or a physics-step API while producing this value.
    """

    control_epoch: int
    capture_time_monotonic_s: float
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    root_pose_world: P0IntegrationPose
    ee_pose_root: P0IntegrationPose
    target_cache: P0IntegrationTargetCache
    abstract_gripper_state: AbstractGripperCommandState

    def __post_init__(self) -> None:
        if not isinstance(self.control_epoch, int) or self.control_epoch < 0:
            raise P0IntegrationObservationError("control_epoch must be a non-negative int")
        object.__setattr__(self, "capture_time_monotonic_s", _finite_scalar("capture_time_monotonic_s", self.capture_time_monotonic_s))
        if self.capture_time_monotonic_s < 0.0:
            raise P0IntegrationObservationError("capture_time_monotonic_s must be non-negative")
        object.__setattr__(self, "joint_position_rad", _float_tuple("joint_position_rad", self.joint_position_rad, 7))
        object.__setattr__(self, "joint_velocity_rad_s", _float_tuple("joint_velocity_rad_s", self.joint_velocity_rad_s, 7))
        if not isinstance(self.root_pose_world, P0IntegrationPose) or self.root_pose_world.frame != P0_WORLD_FRAME:
            raise P0IntegrationObservationError("root_pose_world must be a WORLD P0IntegrationPose")
        if not isinstance(self.ee_pose_root, P0IntegrationPose) or self.ee_pose_root.frame != P0_ROOT_FRAME:
            raise P0IntegrationObservationError("ee_pose_root must be a ROBOT_ROOT P0IntegrationPose")
        if not isinstance(self.target_cache, P0IntegrationTargetCache):
            raise P0IntegrationObservationError("target_cache must be P0IntegrationTargetCache")
        if not isinstance(self.abstract_gripper_state, AbstractGripperCommandState):
            raise P0IntegrationObservationError("abstract_gripper_state must be AbstractGripperCommandState")

    def payload(self) -> dict[str, object]:
        return {
            "control_epoch": self.control_epoch,
            "capture_time_monotonic_s": self.capture_time_monotonic_s,
            "joint_position_rad": list(self.joint_position_rad),
            "joint_velocity_rad_s": list(self.joint_velocity_rad_s),
            "root_pose_world": self.root_pose_world.payload(),
            "ee_pose_root": self.ee_pose_root.payload(),
            "target_cache": self.target_cache.payload(),
            "abstract_gripper_state": self.abstract_gripper_state.value,
        }


@dataclass(frozen=True)
class P0IntegrationObservationSnapshot:
    """One immutable integration observation with explicit non-authority.

    ``authority`` remains fixed to ``NO_ACCEPTANCE_AUTHORITY``.  Consumers
    must obtain any safety/acceptance decision from its separately approved
    owner, not infer it from this state observation.
    """

    capture_id: str
    binding: P0IntegrationObservationBinding
    sample: P0IntegrationObservationSample
    schema: str = P0_READ_ONLY_INTEGRATION_OBSERVATION_SCHEMA
    authority: str = P0_INTEGRATION_OBSERVATION_AUTHORITY

    def __post_init__(self) -> None:
        _sha256("capture_id", self.capture_id)
        if not isinstance(self.binding, P0IntegrationObservationBinding):
            raise P0IntegrationObservationError("binding must be P0IntegrationObservationBinding")
        if not isinstance(self.sample, P0IntegrationObservationSample):
            raise P0IntegrationObservationError("sample must be P0IntegrationObservationSample")
        if self.schema != P0_READ_ONLY_INTEGRATION_OBSERVATION_SCHEMA:
            raise P0IntegrationObservationError("unsupported P0 read-only observation schema")
        if self.authority != P0_INTEGRATION_OBSERVATION_AUTHORITY:
            raise P0IntegrationObservationError("observation may not claim an acceptance authority")

    @property
    def control_epoch(self) -> int:
        return self.sample.control_epoch

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority": self.authority,
            "capture_id": self.capture_id,
            "binding": self.binding.payload(),
            "binding_fingerprint": self.binding.fingerprint(),
            "sample": self.sample.payload(),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@runtime_checkable
class ReadOnlyP0IntegrationObservationSource(Protocol):
    """Runtime-specific source contract for one copied, mutation-free sample."""

    def capture_integration_observation(self) -> P0IntegrationObservationSample:
        """Return current copied values without controller/action/physics mutation."""


@runtime_checkable
class ReadOnlyP0IntegrationObservationProvider(Protocol):
    """Consumer-facing immutable snapshot provider protocol."""

    def capture_snapshot(self) -> P0IntegrationObservationSnapshot:
        """Capture one immutable observation only; never decide admission."""


class P0IntegrationObservationProvider:
    """Materialize an immutable observation from an injected read-only source.

    This class intentionally has no environment, ActionManager, controller,
    articulation, timeline, or actuator reference.  The only dynamic call it
    makes is ``source.capture_integration_observation()`` exactly once.
    """

    def __init__(
        self,
        *,
        binding: P0IntegrationObservationBinding,
        source: ReadOnlyP0IntegrationObservationSource,
    ) -> None:
        if not isinstance(binding, P0IntegrationObservationBinding):
            raise P0IntegrationObservationError("binding must be P0IntegrationObservationBinding")
        if not isinstance(source, ReadOnlyP0IntegrationObservationSource):
            raise P0IntegrationObservationError(
                "source must implement ReadOnlyP0IntegrationObservationSource"
            )
        self._binding = binding
        self._source = source

    @property
    def binding(self) -> P0IntegrationObservationBinding:
        return self._binding

    def capture_snapshot(self) -> P0IntegrationObservationSnapshot:
        sample = self._source.capture_integration_observation()
        if not isinstance(sample, P0IntegrationObservationSample):
            raise P0IntegrationObservationError(
                "read-only source must return P0IntegrationObservationSample"
            )
        identity_payload = {
            "schema": P0_READ_ONLY_INTEGRATION_OBSERVATION_SCHEMA,
            "binding_fingerprint": self._binding.fingerprint(),
            "sample": sample.payload(),
        }
        encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        capture_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return P0IntegrationObservationSnapshot(
            capture_id=capture_id,
            binding=self._binding,
            sample=sample,
        )


def static_contract() -> dict[str, object]:
    """Return the intentionally narrow, source-inspectable contract.

    This is metadata for static P0 preflight evidence only.  It performs no
    capture and deliberately makes no claim that a concrete production source
    has already been runtime-bound.
    """

    return {
        "schema": P0_READ_ONLY_INTEGRATION_OBSERVATION_SCHEMA,
        "authority": P0_INTEGRATION_OBSERVATION_AUTHORITY,
        "provider": "P0IntegrationObservationProvider",
        "source_protocol": "ReadOnlyP0IntegrationObservationSource",
        "runtime_binding": "NOT_BOUND_BY_PURE_INTERFACE_MODULE",
        "observed_fields": [
            "right_arm_joint_position_rad",
            "right_arm_joint_velocity_rad_s",
            "root_pose_world",
            "ee_pose_root",
            "existing_target_cache",
            "capture_time_monotonic_s",
            "control_epoch",
            "abstract_gripper_command_state",
        ],
        "guarantees": [
            "NO_ACCEPTANCE_AUTHORITY",
            "NO_CONTROLLER_MUTATION",
            "NO_PHYSICS_STEPPING",
            "NO_HISTORY_ADVANCEMENT",
            "ONE_READ_ONLY_SOURCE_CAPTURE_PER_SNAPSHOT",
            "IMMUTABLE_VALUE_COPY",
        ],
        "not_a_claim": [
            "IK_FEASIBILITY",
            "WORKSPACE_ACCEPTANCE",
            "ACTIVE_OR_PASSIVE_VELOCITY_PREDICTION",
            "ACTIVE_OR_PASSIVE_ACCELERATION_PREDICTION",
            "COLLISION_PREDICTION",
            "RUNTIME_PRODUCTION_BINDING",
        ],
    }


__all__ = [
    "AbstractGripperCommandState",
    "P0_ABSTRACT_GRIPPER_COMMAND_STATE_SEMANTICS",
    "P0_INTEGRATION_OBSERVATION_AUTHORITY",
    "P0_READ_ONLY_INTEGRATION_OBSERVATION_SCHEMA",
    "P0_ROOT_FRAME",
    "P0_WORLD_FRAME",
    "P0IntegrationObservationBinding",
    "P0IntegrationObservationError",
    "P0IntegrationObservationProvider",
    "P0IntegrationObservationSample",
    "P0IntegrationObservationSnapshot",
    "P0IntegrationPose",
    "P0IntegrationTargetCache",
    "ReadOnlyP0IntegrationObservationProvider",
    "ReadOnlyP0IntegrationObservationSource",
    "static_contract",
]
