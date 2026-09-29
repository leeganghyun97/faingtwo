# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Production-bound metric-to-normalized arm-action adapter for G2.

This module is the deliberately narrow bridge between the policy branch's
post-safety :class:`~.action_interface.EEResidualCommand` (metres and
axis-angle radians) and the *selected existing* G2 production arm action
term.  It does not implement IK, a workspace rule, a joint target, a gripper
target, or an actuator authority.

The selected production arm port is
``G2RedundancyDifferentialIKAction.process_actions``.  Its seven components
are, in order::

    [root dX, root dY, root dZ, rotvec X, rotvec Y, rotvec Z,
     elbow-nullspace request]

``process_actions`` multiplies the first six components by the configured
production scales before it invokes the existing DLS controller.  The gripper
is a *separate* binary action term and is intentionally absent here.

The adapter only performs inverse normalization.  Consequently the sole
forward metric scale between this adapter's output and the production
Cartesian request is the pre-existing production action term's scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Protocol, runtime_checkable

import torch

from ..g2_teleop_dataset import (
    G2_ROTATION_ACTION_SCALE_RAD,
    G2_TRANSLATION_ACTION_SCALE_M,
)
from .action_interface import (
    AbstractGripperIntent,
    CartesianControlFrame,
    CartesianResidualScale,
    CartesianSafetyBoundary,
    EEResidualCommand,
    ExistingAbstractGripperController,
    ExistingEEController,
    GripperHysteresisConfig,
    GripperHysteresisLatch,
    GripperSubmission,
    HighLevelPolicyAction,
    PolicyActionMode,
    PolicyActionContractError,
    PolicyRouteResult,
    SafetyProjection,
    decode_ee_residual,
    gripper_intent_to_existing_simulation_sign,
)


PRODUCTION_METRIC_TO_NORMALIZED_ADAPTER_SCHEMA = (
    "g2_production_metric_to_normalized_arm7_adapter_v1"
)
PRODUCTION_NORMALIZED_ARM_ACTION_SCHEMA = "g2_redundancy_arm_normalized_7d_v1"
PRODUCTION_DEFERRED_FULL_8D_PACKET_SCHEMA = "g2_r0g_deferred_full_8d_packet_v1"
PRODUCTION_DEFERRED_PACKET_PORT_SCHEMA = "g2_r0g_deferred_packet_port_v1"
PRODUCTION_ARM_ACTION_DIM = 7
PRODUCTION_ARM_ACTION_COMPONENTS = (
    "root_translation_x_normalized",
    "root_translation_y_normalized",
    "root_translation_z_normalized",
    "root_rotation_vector_x_normalized",
    "root_rotation_vector_y_normalized",
    "root_rotation_vector_z_normalized",
    "elbow_nullspace_normalized",
)
PRODUCTION_FULL_8D_ACTION_COMPONENTS = (
    *PRODUCTION_ARM_ACTION_COMPONENTS,
    "abstract_gripper_sign",
)
LEGACY_DIRECT_ACTION_MANAGER_PORT_STATUS = "TEST_ONLY"


class ProductionMetricAdapterError(ValueError):
    """Raised for a malformed selected-production action-port conversion."""


@dataclass(frozen=True)
class ProductionArmActionScale:
    """Read-only scale contract of the selected production arm action term.

    The defaults are imported from the same G2 production configuration used
    by the selected production action term.  ``normalized_limit`` is a
    representation contract enforced by this bridge: values beyond it are
    rejected rather than silently clipped.  The selected arm action term does
    not own a raw-action clamp that this policy bridge could reuse.  This is
    not a workspace, joint, force, velocity, or acceleration authority.
    """

    translation_m_per_normalized: float = G2_TRANSLATION_ACTION_SCALE_M
    rotation_rad_per_normalized: float = G2_ROTATION_ACTION_SCALE_RAD
    elbow_per_normalized: float = 1.0
    normalized_limit: float = 1.0
    frame: CartesianControlFrame = CartesianControlFrame.ROBOT_ROOT

    def __post_init__(self) -> None:
        for name, value in (
            ("translation_m_per_normalized", self.translation_m_per_normalized),
            ("rotation_rad_per_normalized", self.rotation_rad_per_normalized),
            ("elbow_per_normalized", self.elbow_per_normalized),
            ("normalized_limit", self.normalized_limit),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ProductionMetricAdapterError(f"{name} must be finite and positive")
        if self.frame is not CartesianControlFrame.ROBOT_ROOT:
            raise ProductionMetricAdapterError(
                "selected production arm action expects robot-root residuals"
            )


@dataclass(frozen=True)
class NormalizedProductionArmAction:
    """A normalized command for the existing seven-dimensional *arm* port.

    This object intentionally has no gripper field.  Its final component is
    the existing elbow-nullspace request; policy branch translation/SE(3)
    actions have no elbow authority, so the adapter always emits exactly
    zero for it.
    """

    values: tuple[float, float, float, float, float, float, float]
    schema: str = PRODUCTION_NORMALIZED_ARM_ACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRODUCTION_NORMALIZED_ARM_ACTION_SCHEMA:
            raise ProductionMetricAdapterError("unsupported normalized arm action schema")
        if len(self.values) != PRODUCTION_ARM_ACTION_DIM:
            raise ProductionMetricAdapterError("normalized production arm action must be 7-D")
        if not all(math.isfinite(float(value)) for value in self.values):
            raise ProductionMetricAdapterError("normalized production arm action must be finite")
        if any(abs(float(value)) > 1.0 for value in self.values):
            raise ProductionMetricAdapterError(
                "normalized production arm action violates the selected [-1, 1] representation contract"
            )
        if float(self.values[6]) != 0.0:
            raise ProductionMetricAdapterError(
                "policy branch has no elbow-nullspace authority; arm action[6] must be exactly zero"
            )

    @property
    def translation_normalized(self) -> tuple[float, float, float]:
        return self.values[:3]

    @property
    def rotation_normalized(self) -> tuple[float, float, float]:
        return self.values[3:6]

    @property
    def elbow_normalized(self) -> float:
        return self.values[6]

    def as_tensor(self, *, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialize the existing action-term input shape ``[N, 7]``.

        Materialization alone does not invoke IK, controller apply, physics,
        or an actuator target.  The actual existing action term decides when
        to process this action in its ordinary environment lifecycle.
        """

        if type(batch_size) is not int or batch_size <= 0:
            raise ProductionMetricAdapterError("batch_size must be a positive int")
        return torch.tensor(self.values, device=device, dtype=dtype).view(1, 7).expand(batch_size, -1).clone()


@dataclass(frozen=True)
class MetricToNormalizedConversionReceipt:
    """Auditable representation-only conversion evidence for one command."""

    input_translation_m: tuple[float, float, float]
    input_rotation_rad: tuple[float, float, float]
    pre_saturation_normalized_arm_action: tuple[float, float, float, float, float, float, float]
    output: NormalizedProductionArmAction
    saturated_components: tuple[bool, bool, bool, bool, bool, bool, bool]
    reconstructed_translation_m: tuple[float, float, float]
    reconstructed_rotation_rad: tuple[float, float, float]
    scale_application_count_after_adapter: int
    schema: str = PRODUCTION_METRIC_TO_NORMALIZED_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRODUCTION_METRIC_TO_NORMALIZED_ADAPTER_SCHEMA:
            raise ProductionMetricAdapterError("unsupported conversion receipt schema")
        if self.scale_application_count_after_adapter != 1:
            raise ProductionMetricAdapterError(
                "selected production forward XYZ scale count must be exactly one"
            )

    @property
    def any_saturation(self) -> bool:
        return any(self.saturated_components)


@runtime_checkable
class ExistingProductionNormalizedArmPort(Protocol):
    """Buffer-only port between the safety receipt and an 8-D frame commit.

    This deliberately does *not* mean that an arm term may be processed on
    its own.  The selected Isaac Lab runtime dispatches the arm and gripper
    through one ActionManager packet.  A conforming implementation therefore
    stores the arm action until the same policy route has resolved the
    abstract gripper intent and commits the complete 8-D packet.
    """

    def submit_normalized_arm_action(self, action: NormalizedProductionArmAction) -> None:
        """Buffer one normalized arm7 action; do not invoke a controller term."""


@dataclass(frozen=True)
class NormalizedProductionActionManagerPacket:
    """One full existing ActionManager packet: arm7 followed by gripper1.

    ``ActionManager.process_action`` dispatches terms in configured order.
    This packet fixes that order to the selected production redundancy runtime:
    the seven-dimensional arm action is followed by the existing binary
    gripper action.  The abstract gripper intent is converted only through
    the existing simulation sign mapper; it is not a mechanism/joint target.
    """

    arm: NormalizedProductionArmAction
    gripper_intent: AbstractGripperIntent

    def __post_init__(self) -> None:
        if not isinstance(self.arm, NormalizedProductionArmAction):
            raise ProductionMetricAdapterError("packet arm must be normalized production arm action")
        if not isinstance(self.gripper_intent, AbstractGripperIntent):
            raise ProductionMetricAdapterError("packet gripper must be an abstract intent")

    @property
    def values(self) -> tuple[float, float, float, float, float, float, float, float]:
        return (*self.arm.values, gripper_intent_to_existing_simulation_sign(self.gripper_intent))

    def as_tensor(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ProductionMetricAdapterError("batch_size must be a positive int")
        return (
            torch.tensor(self.values, device=device, dtype=dtype)
            .view(1, 8)
            .expand(batch_size, -1)
            .clone()
        )


@dataclass(frozen=True)
class DeferredFull8DActionPacket:
    """Immutable full-8D packet staged before canonical ``env.step`` ingress.

    The packet deliberately holds scalar semantic values only.  It does not
    retain a mutable action tensor or an ActionManager reference.  A fresh
    ``[N, 8]`` tensor is materialized *only* by the canonical environment-step
    consumer.  ``packet_id`` distinguishes two valid equal-valued policy
    frames, while ``content_fingerprint`` proves the frozen semantic payload
    has not changed before ingress.
    """

    packet: NormalizedProductionActionManagerPacket
    binding_id: str
    sequence_id: int
    batch_size: int
    device: str
    dtype: torch.dtype
    schema: str = PRODUCTION_DEFERRED_FULL_8D_PACKET_SCHEMA
    packet_id: str = field(init=False)
    content_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.packet, NormalizedProductionActionManagerPacket):
            raise ProductionMetricAdapterError("deferred packet requires normalized full 8-D payload")
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise ProductionMetricAdapterError("deferred packet binding_id must be non-empty")
        if type(self.sequence_id) is not int or self.sequence_id <= 0:
            raise ProductionMetricAdapterError("deferred packet sequence_id must be a positive int")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ProductionMetricAdapterError("deferred packet batch_size must be a positive int")
        if self.schema != PRODUCTION_DEFERRED_FULL_8D_PACKET_SCHEMA:
            raise ProductionMetricAdapterError("unsupported deferred full-8D packet schema")
        device = str(torch.device(self.device))
        if not isinstance(self.dtype, torch.dtype):
            raise ProductionMetricAdapterError("deferred packet dtype must be torch.dtype")
        payload = {
            "schema": self.schema,
            "binding_id": self.binding_id,
            "batch_shape": [self.batch_size, 8],
            "device": device,
            "dtype": str(self.dtype),
            "semantic_field_order": list(PRODUCTION_FULL_8D_ACTION_COMPONENTS),
            # Hex keeps the exact frozen Python-float representation in the
            # fingerprint rather than silently rounding it for JSON display.
            "values_float_hex": [float(value).hex() for value in self.packet.values],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        packet_id = f"{self.binding_id}:{self.sequence_id:016d}:{fingerprint[:16]}"
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "content_fingerprint", fingerprint)
        object.__setattr__(self, "packet_id", packet_id)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.batch_size, 8)

    @property
    def values(self) -> tuple[float, float, float, float, float, float, float, float]:
        """The frozen production-order values, independent of tensor identity."""

        return self.packet.values

    def as_env_step_tensor(self) -> torch.Tensor:
        """Materialize a fresh tensor solely for ``ManagerBasedRLEnv.step``.

        Materialization itself performs no manager dispatch, controller target
        write, physics step, or actuator mutation.
        """

        return self.packet.as_tensor(
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
        )


class DeferredPacketLifecycleState(str, Enum):
    """One-way state of a deferred policy packet.

    A failed ``env.step`` may have called ``process_action`` before failing at
    a later stage.  It is therefore intentionally ``CONSUMPTION_UNKNOWN``
    rather than silently made reusable.
    """

    PENDING = "PENDING"
    CLAIMED_FOR_ENV_STEP = "CLAIMED_FOR_ENV_STEP"
    CONSUMED = "CONSUMED"
    PARTIAL_TERMINATED = "PARTIAL_TERMINATED"
    CONSUMPTION_UNKNOWN = "CONSUMPTION_UNKNOWN"


@dataclass(frozen=True)
class DeferredPacketConsumptionReceipt:
    """Evidence that canonical ``env.step`` returned for one packet."""

    packet_id: str
    content_fingerprint: str
    binding_id: str
    state: DeferredPacketLifecycleState = DeferredPacketLifecycleState.CONSUMED
    action_consumption: str = "FULLY_CONSUMED"
    consumed_substeps: int = 10
    nominal_substeps: int = 10
    termination_reason: str = "NONE"


@runtime_checkable
class DeferredFull8DActionPacketPortProtocol(Protocol):
    """Single-use staging port with no ActionManager/controller capability."""

    def defer_normalized_action_manager_packet(
        self, packet: NormalizedProductionActionManagerPacket
    ) -> DeferredFull8DActionPacket:
        """Stage one immutable full 8-D packet for later canonical ingress."""

    def claim_for_env_step(self, packet_id: str) -> DeferredFull8DActionPacket:
        """Claim the exact pending packet before calling ``env.step`` once."""

    def acknowledge_env_step_success(self, packet_id: str) -> DeferredPacketConsumptionReceipt:
        """Mark a claimed packet consumed after ``env.step`` returns."""

    def mark_env_step_consumption_unknown(self, packet_id: str) -> None:
        """Fail closed when an ``env.step`` exception leaves consumption unknown."""

    def acknowledge_partial_termination(
        self,
        packet_id: str,
        *,
        consumed_substeps: int,
        nominal_substeps: int,
        termination_reason: str,
    ) -> DeferredPacketConsumptionReceipt:
        """Close a structured, source-attested partial terminal consume."""

    @property
    def outstanding_packet(self) -> DeferredFull8DActionPacket | None:
        """The one unresolved packet, if any."""


class DeferredFull8DActionPacketPort:
    """In-memory deferred packet issuer; it deliberately cannot process actions.

    This is an action-lifecycle ownership object, not a safety/controller
    authority.  It never receives an ActionManager, environment, controller,
    articulation, or target setter.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        binding_id: str = "g2_r0g_deferred_full8d_v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ProductionMetricAdapterError("batch_size must be a positive int")
        if not isinstance(binding_id, str) or not binding_id:
            raise ProductionMetricAdapterError("binding_id must be non-empty")
        if not isinstance(dtype, torch.dtype):
            raise ProductionMetricAdapterError("dtype must be torch.dtype")
        self._batch_size = batch_size
        self._device = str(torch.device(device))
        self._dtype = dtype
        self._binding_id = binding_id
        self._next_sequence_id = 1
        self._packet: DeferredFull8DActionPacket | None = None
        self._state: DeferredPacketLifecycleState | None = None
        self._consumed_packet_ids: set[str] = set()
        self._stage_count = 0
        self._claim_count = 0
        self._acknowledgement_count = 0

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @property
    def pending_packet(self) -> DeferredFull8DActionPacket | None:
        return self._packet if self._state is DeferredPacketLifecycleState.PENDING else None

    @property
    def outstanding_packet(self) -> DeferredFull8DActionPacket | None:
        return self._packet

    @property
    def state(self) -> DeferredPacketLifecycleState | None:
        return self._state

    @property
    def stage_count(self) -> int:
        return self._stage_count

    @property
    def claim_count(self) -> int:
        return self._claim_count

    @property
    def acknowledgement_count(self) -> int:
        return self._acknowledgement_count

    def defer_normalized_action_manager_packet(
        self, packet: NormalizedProductionActionManagerPacket
    ) -> DeferredFull8DActionPacket:
        if not isinstance(packet, NormalizedProductionActionManagerPacket):
            raise ProductionMetricAdapterError("deferred port only accepts combined normalized 8-D packets")
        if self._packet is not None:
            raise ProductionMetricAdapterError(
                f"DEFERRED_PACKET_OUTSTANDING:{self._state.value if self._state else 'UNKNOWN'}"
            )
        envelope = DeferredFull8DActionPacket(
            packet=packet,
            binding_id=self._binding_id,
            sequence_id=self._next_sequence_id,
            batch_size=self._batch_size,
            device=self._device,
            dtype=self._dtype,
        )
        self._next_sequence_id += 1
        self._packet = envelope
        self._state = DeferredPacketLifecycleState.PENDING
        self._stage_count += 1
        return envelope

    def _require_packet(self, packet_id: str, state: DeferredPacketLifecycleState) -> DeferredFull8DActionPacket:
        if not isinstance(packet_id, str) or not packet_id:
            raise ProductionMetricAdapterError("deferred packet_id must be non-empty")
        if packet_id in self._consumed_packet_ids:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_REPLAY_DETECTED")
        if self._packet is None or self._packet.packet_id != packet_id:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_IDENTITY_MISMATCH")
        if self._state is not state:
            state_text = self._state.value if self._state is not None else "NONE"
            raise ProductionMetricAdapterError(
                f"DEFERRED_PACKET_STATE_MISMATCH:expected={state.value}:actual={state_text}"
            )
        return self._packet

    def claim_for_env_step(self, packet_id: str) -> DeferredFull8DActionPacket:
        envelope = self._require_packet(packet_id, DeferredPacketLifecycleState.PENDING)
        self._state = DeferredPacketLifecycleState.CLAIMED_FOR_ENV_STEP
        self._claim_count += 1
        return envelope

    def acknowledge_env_step_success(self, packet_id: str) -> DeferredPacketConsumptionReceipt:
        envelope = self._require_packet(packet_id, DeferredPacketLifecycleState.CLAIMED_FOR_ENV_STEP)
        self._state = DeferredPacketLifecycleState.CONSUMED
        self._consumed_packet_ids.add(packet_id)
        self._acknowledgement_count += 1
        receipt = DeferredPacketConsumptionReceipt(
            packet_id=envelope.packet_id,
            content_fingerprint=envelope.content_fingerprint,
            binding_id=envelope.binding_id,
        )
        # Retain no replayable packet after a successful canonical consume.
        self._packet = None
        self._state = None
        return receipt

    def mark_env_step_consumption_unknown(self, packet_id: str) -> None:
        self._require_packet(packet_id, DeferredPacketLifecycleState.CLAIMED_FOR_ENV_STEP)
        self._state = DeferredPacketLifecycleState.CONSUMPTION_UNKNOWN

    def acknowledge_partial_termination(
        self,
        packet_id: str,
        *,
        consumed_substeps: int,
        nominal_substeps: int,
        termination_reason: str,
    ) -> DeferredPacketConsumptionReceipt:
        envelope = self._require_packet(
            packet_id, DeferredPacketLifecycleState.CLAIMED_FOR_ENV_STEP
        )
        if (
            type(consumed_substeps) is not int
            or type(nominal_substeps) is not int
            or nominal_substeps != 10
            or not 1 <= consumed_substeps <= nominal_substeps
            or termination_reason != "RUNTIME_HARDSTOP"
        ):
            raise ProductionMetricAdapterError(
                "INVALID_PARTIAL_TERMINATION_RECEIPT"
            )
        self._state = DeferredPacketLifecycleState.PARTIAL_TERMINATED
        self._consumed_packet_ids.add(packet_id)
        self._acknowledgement_count += 1
        receipt = DeferredPacketConsumptionReceipt(
            packet_id=envelope.packet_id,
            content_fingerprint=envelope.content_fingerprint,
            binding_id=envelope.binding_id,
            state=DeferredPacketLifecycleState.PARTIAL_TERMINATED,
            action_consumption="PARTIAL_TERMINATED",
            consumed_substeps=consumed_substeps,
            nominal_substeps=nominal_substeps,
            termination_reason=termination_reason,
        )
        # A partially terminated packet is terminal and never replayable.
        self._packet = None
        self._state = None
        return receipt


@dataclass(frozen=True)
class ProductionDeferredRouteResult(PolicyRouteResult):
    """Compatibility-preserving policy receipt plus a one-use full-8D packet."""

    deferred_packet: DeferredFull8DActionPacket | None = None


@runtime_checkable
class ExistingProductionActionManagerPort(Protocol):
    """Legacy direct combined-packet surface, retained for inert diagnostics.

    It is deliberately not part of the deferred canonical R0g factory.
    """

    def submit_normalized_action_manager_packet(
        self, packet: NormalizedProductionActionManagerPacket
    ) -> None:
        """Submit exactly one combined ``[arm7, gripper1]`` packet."""


def assert_selected_production_action_manager_surface(action_manager: object) -> None:
    """Read-only structural check for the audited existing 8-D manager surface.

    This function performs no packet processing.  It is shared by the legacy
    inert-test port and the deferred canonical env-step binder so the latter
    need not own or invoke the ActionManager directly.
    """

    expected_arm_term_name = "arm_action"
    expected_gripper_term_name = "gripper_action"
    expected_arm_term_class = "G2RedundancyDifferentialIKAction"
    expected_gripper_term_class = "G2RateLimitedBinaryJointPositionAction"
    manager = action_manager
    if not callable(getattr(manager, "process_action", None)):
        raise ProductionMetricAdapterError("selected action manager must expose process_action")
    if getattr(manager, "total_action_dim", None) != 8:
        raise ProductionMetricAdapterError(
            "selected production action manager must expose total_action_dim == 8"
        )
    if tuple(getattr(manager, "active_terms", ())) != (
        expected_arm_term_name,
        expected_gripper_term_name,
    ):
        raise ProductionMetricAdapterError(
            "selected production action-manager term order must be arm_action, gripper_action"
        )
    get_term = getattr(manager, "get_term", None)
    if not callable(get_term):
        raise ProductionMetricAdapterError("selected action manager must expose get_term")
    arm = get_term(expected_arm_term_name)
    gripper = get_term(expected_gripper_term_name)
    if type(arm).__name__ != expected_arm_term_class or getattr(arm, "action_dim", None) != 7:
        raise ProductionMetricAdapterError("selected arm term is not the audited 7-D redundancy action")
    if type(gripper).__name__ != expected_gripper_term_class or getattr(gripper, "action_dim", None) != 1:
        raise ProductionMetricAdapterError("selected gripper term is not the audited binary action")
    arm_cfg = getattr(arm, "cfg", None)
    scale = tuple(float(value) for value in getattr(arm_cfg, "scale", ()))
    expected_scale = (
        G2_TRANSLATION_ACTION_SCALE_M,
        G2_TRANSLATION_ACTION_SCALE_M,
        G2_TRANSLATION_ACTION_SCALE_M,
        G2_ROTATION_ACTION_SCALE_RAD,
        G2_ROTATION_ACTION_SCALE_RAD,
        G2_ROTATION_ACTION_SCALE_RAD,
        1.0,
    )
    if scale != expected_scale:
        raise ProductionMetricAdapterError("selected arm term scale differs from audited production contract")
    controller = getattr(arm_cfg, "controller", None)
    if (
        getattr(controller, "command_type", None) != "pose"
        or getattr(controller, "use_relative_mode", None) is not True
        or getattr(controller, "ik_method", None) != "dls"
    ):
        raise ProductionMetricAdapterError("selected arm controller is not audited relative pose DLS")


class BufferedNormalizedProductionArmPort(ExistingProductionNormalizedArmPort):
    """One-use arm7 buffer for a complete ActionManager transaction.

    ``NewPolicySafetyPreControllerBridge`` can use this as its downstream
    ``ExistingEEController`` endpoint through
    :class:`ProductionMetricToNormalizedArmAdapter`.  The buffer creates no
    joint target and does not call the ActionManager.  It exists specifically
    so that a later gripper rejection cannot leave a processed arm action.
    """

    def __init__(self) -> None:
        self._pending: NormalizedProductionArmAction | None = None
        self.last_buffered_action: NormalizedProductionArmAction | None = None

    @property
    def pending(self) -> NormalizedProductionArmAction | None:
        return self._pending

    def submit_normalized_arm_action(self, action: NormalizedProductionArmAction) -> None:
        if not isinstance(action, NormalizedProductionArmAction):
            raise ProductionMetricAdapterError("arm buffer only accepts NormalizedProductionArmAction")
        if self._pending is not None:
            raise ProductionMetricAdapterError("a production arm action is already pending")
        self._pending = action
        self.last_buffered_action = action

    def discard(self) -> None:
        """Discard a non-committed arm action after any route rejection."""

        self._pending = None

    def take_packet(self, intent: AbstractGripperIntent) -> NormalizedProductionActionManagerPacket:
        """Consume the pending arm action as one arm7+gripper1 packet."""

        if self._pending is None:
            raise ProductionMetricAdapterError("no buffered arm action is available for frame commit")
        packet = NormalizedProductionActionManagerPacket(self._pending, intent)
        self._pending = None
        return packet


class G2RedundancyActionManagerPort:
    """Legacy/test-only direct binding to ``ActionManager.process_action([N,8])``.

    ``LEGACY_DIRECT_ACTION_MANAGER_PORT_STATUS`` is ``TEST_ONLY``.  The
    canonical deferred R0g router/factory never accepts or constructs this
    class.  It remains only to preserve historical, explicitly inert
    diagnostic compatibility while those artifacts stay read-only evidence.
    """

    def __init__(
        self,
        *,
        action_manager: object,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ProductionMetricAdapterError("batch_size must be a positive int")
        self._action_manager = action_manager
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._dtype = dtype
        self._assert_selected_production_surface()
        self.last_submitted_packet: NormalizedProductionActionManagerPacket | None = None
        self.last_submitted_tensor: torch.Tensor | None = None

    def _assert_selected_production_surface(self) -> None:
        assert_selected_production_action_manager_surface(self._action_manager)

    def submit_normalized_action_manager_packet(
        self, packet: NormalizedProductionActionManagerPacket
    ) -> None:
        if not isinstance(packet, NormalizedProductionActionManagerPacket):
            raise ProductionMetricAdapterError("action-manager port only accepts a combined normalized packet")
        self._assert_selected_production_surface()
        tensor = packet.as_tensor(
            batch_size=self._batch_size, device=self._device, dtype=self._dtype
        )
        self._action_manager.process_action(tensor)
        self.last_submitted_packet = packet
        self.last_submitted_tensor = tensor


@runtime_checkable
class ExistingAbstractGripperPreflight(Protocol):
    """Existing gripper-state preflight, without an independent write.

    The ActionManager packet is the only write in this binding.  A concrete
    implementation may read existing reset-open/contact-hold state and return
    the already-defined abstract result, but must not emit a separate finger
    command.  This protocol intentionally exposes neither a joint nor a
    mechanism target.
    """

    def preflight_gripper_intent(self, intent: AbstractGripperIntent) -> GripperSubmission:
        """Return existing-controller acceptance for a prospective intent."""


class StaticAbstractGripperPreflight:
    """Minimal default for bindings with no separate existing rejection API.

    The selected binary action term still owns reset-open, rate limiting and
    physical application when the full packet is later processed.  This
    default is *not* a claim that the gripper has reached an endpoint; it only
    carries the requested abstract OPEN/CLOSE state into that existing term.
    Runtime deployments may supply an existing read-only preflight adapter
    when such feedback is available.
    """

    def preflight_gripper_intent(self, intent: AbstractGripperIntent) -> GripperSubmission:
        if not isinstance(intent, AbstractGripperIntent):
            raise ProductionMetricAdapterError("gripper intent must be abstract OPEN/CLOSE")
        return GripperSubmission(True, intent)


@dataclass(frozen=True)
class _ReceiptBoundAdapterPathEvidence:
    """Unforgeable-by-configuration identity proof made only by the factory.

    The generic safety bridge exposes no public downstream-controller getter.
    Rather than inspect its private field, the single construction factory
    creates this object at the exact point where it supplies ``adapter`` as
    the bridge's downstream controller.  The router accepts a receipt-bound
    path only when all object identities match this evidence.  Direct adapter
    wiring remains valid for a pure static/unit-test route.
    """

    bridge: object
    adapter: "ProductionMetricToNormalizedArmAdapter"
    arm_buffer: BufferedNormalizedProductionArmPort


class ProductionBoundPolicyCommandRouter:
    """Production-bound high-level route that only stages a full 8-D packet.

    This replaces neither the existing safety authority nor the selected
    differential-IK controller.  It performs only:

    ``policy -> metric residual -> safety receipt -> inverse normalization
    -> buffered arm7 + existing abstract gripper -> deferred full 8-D packet``.

    ``ManagerBasedRLEnv.step(packet)`` remains the sole canonical owner of
    ``ActionManager.process_action(packet)``.  The router intentionally has
    no ActionManager, environment, controller-target, actuator, or physics
    reference.  Its hysteresis latch commits only after the canonical
    env-step consumer acknowledges that ``env.step`` returned.
    """

    def __init__(
        self,
        *,
        safety_boundary: CartesianSafetyBoundary,
        ee_submission_controller: ExistingEEController,
        adapter: "ProductionMetricToNormalizedArmAdapter",
        arm_buffer: BufferedNormalizedProductionArmPort,
        deferred_packet_port: DeferredFull8DActionPacketPortProtocol,
        gripper_preflight: ExistingAbstractGripperPreflight | None = None,
        receipt_bound_path: _ReceiptBoundAdapterPathEvidence | None = None,
        action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
        scale: ProductionArmActionScale = ProductionArmActionScale(),
        hysteresis: GripperHysteresisConfig = GripperHysteresisConfig(),
    ) -> None:
        if not isinstance(safety_boundary, CartesianSafetyBoundary):
            raise ProductionMetricAdapterError("safety_boundary must implement CartesianSafetyBoundary")
        if not isinstance(ee_submission_controller, ExistingEEController):
            raise ProductionMetricAdapterError("ee_submission_controller must implement ExistingEEController")
        if not isinstance(adapter, ProductionMetricToNormalizedArmAdapter):
            raise ProductionMetricAdapterError("adapter must be ProductionMetricToNormalizedArmAdapter")
        if not isinstance(arm_buffer, BufferedNormalizedProductionArmPort):
            raise ProductionMetricAdapterError("arm_buffer must be BufferedNormalizedProductionArmPort")
        if not isinstance(deferred_packet_port, DeferredFull8DActionPacketPortProtocol):
            raise ProductionMetricAdapterError(
                "deferred_packet_port must expose the deferred full-8D packet protocol"
            )
        if not isinstance(action_mode, PolicyActionMode):
            raise ProductionMetricAdapterError("action_mode must be PolicyActionMode")
        if not isinstance(scale, ProductionArmActionScale):
            raise ProductionMetricAdapterError("scale must be ProductionArmActionScale")
        if not isinstance(hysteresis, GripperHysteresisConfig):
            raise ProductionMetricAdapterError("hysteresis must be GripperHysteresisConfig")
        if adapter.downstream_port is not arm_buffer:
            raise ProductionMetricAdapterError(
                "metric adapter must feed the same buffer committed by this runtime router"
            )
        direct_adapter_path = ee_submission_controller is adapter
        receipt_bound_adapter_path = bool(
            isinstance(receipt_bound_path, _ReceiptBoundAdapterPathEvidence)
            and receipt_bound_path.bridge is safety_boundary
            and ee_submission_controller is receipt_bound_path.bridge
            and receipt_bound_path.adapter is adapter
            and receipt_bound_path.arm_buffer is arm_buffer
        )
        # A controller merely satisfying the same protocol is not enough.  It
        # could submit a joint/arm action immediately before gripper preflight
        # rejects.  Only a direct buffer adapter (for static tests) or the
        # factory-proven receipt bridge path may enter this router.
        if not (direct_adapter_path or receipt_bound_adapter_path):
            raise ProductionMetricAdapterError(
                "EE submission controller is not proven to route through the selected metric adapter buffer"
            )
        # Receipt-bound safety bridges must retain ownership of the accepted
        # command submission; a direct buffer adapter would bypass its
        # epoch/fingerprint check.
        if bool(getattr(safety_boundary, "requires_atomic_accepted_submission", False)) and not receipt_bound_adapter_path:
            raise ProductionMetricAdapterError(
                "receipt-bound safety bridge requires factory-proven adapter-path evidence"
            )
        self._safety_boundary = safety_boundary
        self._ee_submission_controller = ee_submission_controller
        self._adapter = adapter
        self._arm_buffer = arm_buffer
        self._deferred_packet_port = deferred_packet_port
        self._gripper_preflight = (
            StaticAbstractGripperPreflight()
            if gripper_preflight is None
            else gripper_preflight
        )
        if not isinstance(self._gripper_preflight, ExistingAbstractGripperPreflight):
            raise ProductionMetricAdapterError("gripper_preflight must expose abstract preflight")
        self._action_mode = action_mode
        self._scale = scale
        self._latch = GripperHysteresisLatch(hysteresis)
        self.last_packet: NormalizedProductionActionManagerPacket | None = None
        self.last_deferred_packet: DeferredFull8DActionPacket | None = None
        self._pending_effective_intent: AbstractGripperIntent | None = None
        self.last_route_error: str = ""

    @property
    def gripper_intent(self) -> AbstractGripperIntent:
        return self._latch.intent

    @property
    def deferred_packet_port(self) -> DeferredFull8DActionPacketPortProtocol:
        """Read-only lifecycle port; it has no ActionManager capability."""

        return self._deferred_packet_port

    @property
    def pending_deferred_packet(self) -> DeferredFull8DActionPacket | None:
        return self._deferred_packet_port.outstanding_packet

    def reset(self) -> None:
        """Reset only when no staged packet has an unresolved consumption state."""

        if self._deferred_packet_port.outstanding_packet is not None:
            raise ProductionMetricAdapterError(
                "DEFERRED_PACKET_OUTSTANDING_RESET_FORBIDDEN"
            )

        self._arm_buffer.discard()
        self._latch.reset()
        self.last_packet = None
        self.last_deferred_packet = None
        self._pending_effective_intent = None
        self.last_route_error = ""

    def _rejected(
        self,
        *,
        reason: str,
        requested_intent: AbstractGripperIntent | None = None,
        submission: GripperSubmission | None = None,
    ) -> ProductionDeferredRouteResult:
        self._arm_buffer.discard()
        self.last_packet = None
        self.last_deferred_packet = None
        self.last_route_error = reason
        current = self._latch.intent
        return ProductionDeferredRouteResult(
            accepted=False,
            reason=reason,
            ee_command=None,
            gripper_intent=current,
            requested_gripper_intent=current if requested_intent is None else requested_intent,
            gripper_command_emitted=False,
            gripper_submission=submission,
            deferred_packet=None,
        )

    def _pending_rejected(self) -> ProductionDeferredRouteResult:
        """Reject a second policy frame without disturbing the first staged frame."""

        current = self._latch.intent
        self.last_route_error = "DEFERRED_PACKET_PENDING_UNCONSUMED"
        return ProductionDeferredRouteResult(
            accepted=False,
            reason=self.last_route_error,
            ee_command=None,
            gripper_intent=current,
            requested_gripper_intent=current,
            gripper_command_emitted=False,
            gripper_submission=None,
            deferred_packet=None,
        )

    def route(self, action: HighLevelPolicyAction) -> ProductionDeferredRouteResult:
        """Validate and stage one immutable full packet; never consume it directly."""

        if not isinstance(action, HighLevelPolicyAction):
            raise ProductionMetricAdapterError("route requires HighLevelPolicyAction")
        if action.mode is not self._action_mode:
            raise ProductionMetricAdapterError(
                f"router is locked to {self._action_mode.value}, got {action.mode.value}"
            )
        if self._pending_effective_intent is not None or self.pending_deferred_packet is not None:
            return self._pending_rejected()
        self._arm_buffer.discard()
        requested = decode_ee_residual(
            action,
            scale=CartesianResidualScale(
                translation_m_per_normalized=self._scale.translation_m_per_normalized,
                rotation_rad_per_normalized=self._scale.rotation_rad_per_normalized,
                control_frame=self._scale.frame,
            ),
        )
        projection = self._safety_boundary.project(requested)
        if not isinstance(projection, SafetyProjection):
            raise ProductionMetricAdapterError("safety boundary must return SafetyProjection")
        if not projection.accepted:
            return self._rejected(reason=projection.reason or "EE_COMMAND_REJECTED_BY_SAFETY")
        if projection.command is None or projection.command.frame is not requested.frame:
            return self._rejected(reason="SAFETY_PROJECTION_FRAME_MISMATCH")
        requested_intent = self._latch.proposed_intent(action.gripper_probability)
        try:
            # For a receipt-bound bridge this performs its existing one-shot
            # freshness check and then calls the adapter.  The adapter only
            # buffers a normalized arm7 action at this point.
            self._ee_submission_controller.submit_ee_residual(projection.command)
        except Exception as error:
            return self._rejected(reason=f"EE_ADAPTER_SUBMISSION_REJECTED:{type(error).__name__}")
        try:
            submission = self._gripper_preflight.preflight_gripper_intent(requested_intent)
        except Exception as error:
            return self._rejected(
                reason=f"GRIPPER_PREFLIGHT_ERROR:{type(error).__name__}",
                requested_intent=requested_intent,
            )
        if not isinstance(submission, GripperSubmission):
            return self._rejected(
                reason="GRIPPER_PREFLIGHT_INVALID_RESULT",
                requested_intent=requested_intent,
            )
        if not submission.accepted:
            return self._rejected(
                reason=submission.reason or "GRIPPER_INTENT_REJECTED_BY_EXISTING_CONTROLLER",
                requested_intent=requested_intent,
                submission=submission,
            )
        try:
            packet = self._arm_buffer.take_packet(submission.effective_intent)
            deferred_packet = self._deferred_packet_port.defer_normalized_action_manager_packet(packet)
        except Exception as error:
            self._arm_buffer.discard()
            return self._rejected(
                reason=f"DEFERRED_PACKET_STAGE_FAILED:{type(error).__name__}",
                requested_intent=requested_intent,
                submission=submission,
            )
        self.last_packet = packet
        self.last_deferred_packet = deferred_packet
        self._pending_effective_intent = submission.effective_intent
        self.last_route_error = ""
        # Do not commit the latch yet: an env-step failure can occur after a
        # manager process call, so stage and consumption must remain distinct.
        return ProductionDeferredRouteResult(
            accepted=True,
            reason="",
            ee_command=projection.command,
            gripper_intent=self._latch.intent,
            requested_gripper_intent=requested_intent,
            gripper_command_emitted=False,
            gripper_submission=submission,
            deferred_packet=deferred_packet,
        )

    def assert_pending_deferred_packet(
        self, packet: DeferredFull8DActionPacket
    ) -> None:
        """Verify immutable identity before the canonical consumer claims it."""

        if not isinstance(packet, DeferredFull8DActionPacket):
            raise ProductionMetricAdapterError("canonical consumer requires DeferredFull8DActionPacket")
        pending = self.pending_deferred_packet
        if pending is None or self._pending_effective_intent is None:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_NOT_PENDING")
        if pending.packet_id != packet.packet_id:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_IDENTITY_MISMATCH")
        if (
            pending.content_fingerprint != packet.content_fingerprint
            or pending.values != packet.values
            or pending.shape != packet.shape
            or pending.device != packet.device
            or pending.dtype != packet.dtype
        ):
            raise ProductionMetricAdapterError("DEFERRED_PACKET_MUTATION_DETECTED")

    def claim_deferred_packet_for_env_step(
        self, packet: DeferredFull8DActionPacket
    ) -> DeferredFull8DActionPacket:
        self.assert_pending_deferred_packet(packet)
        claimed = self._deferred_packet_port.claim_for_env_step(packet.packet_id)
        if claimed.packet_id != packet.packet_id or claimed.content_fingerprint != packet.content_fingerprint:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_CLAIM_MISMATCH")
        return claimed

    def acknowledge_env_step_success(
        self, packet_id: str
    ) -> DeferredPacketConsumptionReceipt:
        """Commit the staged gripper intent only after ``env.step`` returned."""

        if self._pending_effective_intent is None:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_ACK_WITHOUT_PENDING_ROUTE")
        receipt = self._deferred_packet_port.acknowledge_env_step_success(packet_id)
        self._latch.commit(self._pending_effective_intent)
        self._pending_effective_intent = None
        self.last_route_error = ""
        return receipt

    def mark_env_step_consumption_unknown(self, packet_id: str) -> None:
        """Lock this route after an env-step exception; never retry blindly."""

        self._deferred_packet_port.mark_env_step_consumption_unknown(packet_id)
        self.last_route_error = "DEFERRED_PACKET_CONSUMPTION_UNKNOWN"

    def acknowledge_partial_termination(
        self,
        packet_id: str,
        *,
        consumed_substeps: int,
        nominal_substeps: int,
        termination_reason: str,
    ) -> DeferredPacketConsumptionReceipt:
        """Commit a source-attested terminal consume without retrying it."""

        if self._pending_effective_intent is None:
            raise ProductionMetricAdapterError(
                "DEFERRED_PACKET_PARTIAL_ACK_WITHOUT_PENDING_ROUTE"
            )
        receipt = self._deferred_packet_port.acknowledge_partial_termination(
            packet_id,
            consumed_substeps=consumed_substeps,
            nominal_substeps=nominal_substeps,
            termination_reason=termination_reason,
        )
        # The command was physically consumed for at least one substep.  Its
        # effective gripper state is therefore committed before the episode is
        # closed, but the packet can never be retried.
        self._latch.commit(self._pending_effective_intent)
        self._pending_effective_intent = None
        self.last_route_error = "RUNTIME_HARDSTOP_PARTIAL_TERMINATION"
        return receipt


@dataclass
class ProductionMetricAdapterRuntimeBinding:
    """The explicit construction record for the selected policy runtime path.

    The record keeps the receipt bridge, the metric adapter, its one-use arm
    buffer, and the deferred packet issuer together.  It is intentionally not
    a controller object and its construction neither starts a simulation nor
    submits an action.  It deliberately contains no ActionManager reference.
    """

    safety_bridge: Any
    arm_buffer: BufferedNormalizedProductionArmPort
    metric_adapter: "ProductionMetricToNormalizedArmAdapter"
    deferred_packet_port: DeferredFull8DActionPacketPortProtocol
    router: ProductionBoundPolicyCommandRouter


def build_receipt_bound_production_policy_runtime(
    *,
    authority: Any,
    snapshot_provider: Any,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    binding_id: str = "g2_r0g_deferred_full8d_v1",
    gripper_preflight: ExistingAbstractGripperPreflight | None = None,
    action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    scale: ProductionArmActionScale = ProductionArmActionScale(),
    hysteresis: GripperHysteresisConfig = GripperHysteresisConfig(),
) -> ProductionMetricAdapterRuntimeBinding:
    """Build the deferred-only receipt-bound Production adapter composition.

    ``NewPolicySafetyPreControllerBridge`` remains the owner of the existing
    receipt/freshness authority.  The bridge forwards an accepted metric
    residual to a buffer-only adapter; the router returns a one-use deferred
    8-D packet.  Only ``ManagerBasedRLEnv.step`` may later process it.  This
    function creates no Isaac application, action-manager binding,
    articulation, target, or PhysX step.  If the authority is unresolved, the
    resulting bridge continues to reject fail-closed.
    """

    # Keep this import local: the policy adapter package can be inspected in
    # static tooling without eagerly importing the larger authority module.
    from .new_policy_safety_bridge import NewPolicySafetyPreControllerBridge

    arm_buffer = BufferedNormalizedProductionArmPort()
    adapter = ProductionMetricToNormalizedArmAdapter(
        downstream_port=arm_buffer,
        scale=scale,
    )
    bridge = NewPolicySafetyPreControllerBridge(
        authority=authority,
        snapshot_provider=snapshot_provider,
        downstream_controller=adapter,
    )
    evidence = _ReceiptBoundAdapterPathEvidence(
        bridge=bridge,
        adapter=adapter,
        arm_buffer=arm_buffer,
    )
    port = DeferredFull8DActionPacketPort(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        binding_id=binding_id,
    )
    router = ProductionBoundPolicyCommandRouter(
        safety_boundary=bridge,
        ee_submission_controller=bridge,
        adapter=adapter,
        arm_buffer=arm_buffer,
        deferred_packet_port=port,
        gripper_preflight=gripper_preflight,
        receipt_bound_path=evidence,
        action_mode=action_mode,
        scale=scale,
        hysteresis=hysteresis,
    )
    return ProductionMetricAdapterRuntimeBinding(
        safety_bridge=bridge,
        arm_buffer=arm_buffer,
        metric_adapter=adapter,
        deferred_packet_port=port,
        router=router,
    )


class CanonicalManagerBasedRLEnvDeferredPacketConsumer:
    """Thin canonical ingress binder: one staged packet delegates to ``env.step``.

    The object receives the environment rather than an independently supplied
    ActionManager.  It read-only validates the environment's selected manager
    surface during construction, then delegates exactly one immutable packet
    to ``env.step``.  It never calls ``process_action`` directly and has no
    controller target, physics, or actuator API.
    """

    def __init__(self, *, env: object, router: ProductionBoundPolicyCommandRouter) -> None:
        if not isinstance(router, ProductionBoundPolicyCommandRouter):
            raise ProductionMetricAdapterError("router must be ProductionBoundPolicyCommandRouter")
        if not callable(getattr(env, "step", None)):
            raise ProductionMetricAdapterError("canonical environment must expose env.step")
        action_manager = getattr(env, "action_manager", None)
        assert_selected_production_action_manager_surface(action_manager)
        self._env = env
        self._router = router
        self._submitted_packet_ids: set[str] = set()
        self.env_step_submission_count = 0
        self.last_ingress_packet_id: str | None = None
        self.last_ingress_fingerprint: str | None = None

    def consume(self, packet: DeferredFull8DActionPacket) -> DeferredPacketConsumptionReceipt:
        """Delegate the exact pending packet once, then acknowledge its result."""

        self._router.assert_pending_deferred_packet(packet)
        if packet.packet_id in self._submitted_packet_ids:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_REPLAY_DETECTED")
        tensor = packet.as_env_step_tensor()
        if tuple(tensor.shape) != packet.shape:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_TENSOR_SHAPE_MISMATCH")
        if tensor.dtype != packet.dtype or str(tensor.device) != packet.device:
            raise ProductionMetricAdapterError("DEFERRED_PACKET_TENSOR_METADATA_MISMATCH")
        claimed = self._router.claim_deferred_packet_for_env_step(packet)
        self._submitted_packet_ids.add(packet.packet_id)
        self.env_step_submission_count += 1
        self.last_ingress_packet_id = claimed.packet_id
        self.last_ingress_fingerprint = claimed.content_fingerprint
        try:
            # This is the one and only action-consumption entry point in this
            # policy binding.  Isaac Lab owns its downstream process_action.
            self._env.step(tensor)
        except Exception:
            self._router.mark_env_step_consumption_unknown(packet.packet_id)
            raise
        return self._router.acknowledge_env_step_success(packet.packet_id)


@dataclass
class CanonicalProductionPolicyRuntimeBinding:
    """Deferred policy composition plus its canonical environment-step binder."""

    deferred_runtime: ProductionMetricAdapterRuntimeBinding
    env_step_consumer: CanonicalManagerBasedRLEnvDeferredPacketConsumer


def build_canonical_env_step_production_policy_runtime(
    *,
    env: object,
    authority: Any,
    snapshot_provider: Any,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    binding_id: str = "g2_r0g_deferred_full8d_v1",
    gripper_preflight: ExistingAbstractGripperPreflight | None = None,
    action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    scale: ProductionArmActionScale = ProductionArmActionScale(),
    hysteresis: GripperHysteresisConfig = GripperHysteresisConfig(),
) -> CanonicalProductionPolicyRuntimeBinding:
    """Bind a deferred route to existing canonical ``ManagerBasedRLEnv.step``.

    Constructing this object performs only source-surface identity validation;
    it does not send a packet, step physics, or mutate a controller.
    """

    runtime = build_receipt_bound_production_policy_runtime(
        authority=authority,
        snapshot_provider=snapshot_provider,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        binding_id=binding_id,
        gripper_preflight=gripper_preflight,
        action_mode=action_mode,
        scale=scale,
        hysteresis=hysteresis,
    )
    return CanonicalProductionPolicyRuntimeBinding(
        deferred_runtime=runtime,
        env_step_consumer=CanonicalManagerBasedRLEnvDeferredPacketConsumer(
            env=env,
            router=runtime.router,
        ),
    )


class ProductionMetricToNormalizedArmAdapter(ExistingEEController):
    """Adapt a post-safety metric residual to the selected arm action port.

    This is intentionally an ``ExistingEEController`` implementation so it
    can be connected directly after ``PolicyCommandRouter`` (or its
    receipt-bound safety bridge).  The adapter is not an IK/safety layer:
    valid metric residuals already passed through the upstream safety
    boundary.  It rejects values outside the selected production port's
    audited normalized representation rather than clipping them.
    """

    def __init__(
        self,
        *,
        downstream_port: ExistingProductionNormalizedArmPort,
        scale: ProductionArmActionScale = ProductionArmActionScale(),
    ) -> None:
        if not isinstance(downstream_port, ExistingProductionNormalizedArmPort):
            raise ProductionMetricAdapterError(
                "downstream_port must expose the selected normalized arm-action port"
            )
        if not isinstance(scale, ProductionArmActionScale):
            raise ProductionMetricAdapterError("scale must be ProductionArmActionScale")
        self._downstream_port = downstream_port
        self._scale = scale
        self.last_conversion: MetricToNormalizedConversionReceipt | None = None

    @property
    def scale(self) -> ProductionArmActionScale:
        return self._scale

    @property
    def downstream_port(self) -> ExistingProductionNormalizedArmPort:
        """Read-only identity needed to prove the one-frame binding."""

        return self._downstream_port

    def convert(self, command: EEResidualCommand) -> MetricToNormalizedConversionReceipt:
        """Convert metric root-frame residuals without calling a controller."""

        if not isinstance(command, EEResidualCommand):
            raise ProductionMetricAdapterError("adapter requires EEResidualCommand[m/rad]")
        if command.frame is not self._scale.frame:
            raise ProductionMetricAdapterError(
                "metric residual frame does not match selected production root frame"
            )
        pre = (
            *(float(value) / self._scale.translation_m_per_normalized for value in command.translation_m),
            *(float(value) / self._scale.rotation_rad_per_normalized for value in command.rotation_rad),
            0.0,
        )
        if not all(math.isfinite(value) for value in pre):
            raise ProductionMetricAdapterError("metric-to-normalized conversion is non-finite")
        if any(abs(value) > self._scale.normalized_limit for value in pre):
            raise ProductionMetricAdapterError(
                "metric residual is outside the selected production normalized "
                "arm-action representation; rejected without clipping"
            )
        action = NormalizedProductionArmAction(pre)
        return MetricToNormalizedConversionReceipt(
            input_translation_m=tuple(float(value) for value in command.translation_m),
            input_rotation_rad=tuple(float(value) for value in command.rotation_rad),
            pre_saturation_normalized_arm_action=pre,
            output=action,
            saturated_components=(False, False, False, False, False, False, False),
            reconstructed_translation_m=tuple(
                value * self._scale.translation_m_per_normalized
                for value in action.translation_normalized
            ),
            reconstructed_rotation_rad=tuple(
                value * self._scale.rotation_rad_per_normalized
                for value in action.rotation_normalized
            ),
            # The adapter divides; the selected action term remains the only
            # forward metric scale after this boundary.
            scale_application_count_after_adapter=1,
        )

    def submit_ee_residual(self, command: EEResidualCommand) -> None:
        """Convert then forward exactly one normalized arm action."""

        receipt = self.convert(command)
        self._downstream_port.submit_normalized_arm_action(receipt.output)
        self.last_conversion = receipt


__all__ = [
    "BufferedNormalizedProductionArmPort",
    "CanonicalManagerBasedRLEnvDeferredPacketConsumer",
    "CanonicalProductionPolicyRuntimeBinding",
    "DeferredFull8DActionPacket",
    "DeferredFull8DActionPacketPort",
    "DeferredFull8DActionPacketPortProtocol",
    "DeferredPacketConsumptionReceipt",
    "DeferredPacketLifecycleState",
    "ExistingAbstractGripperPreflight",
    "ExistingProductionActionManagerPort",
    "ExistingProductionNormalizedArmPort",
    "G2RedundancyActionManagerPort",
    "LEGACY_DIRECT_ACTION_MANAGER_PORT_STATUS",
    "MetricToNormalizedConversionReceipt",
    "NormalizedProductionActionManagerPacket",
    "NormalizedProductionArmAction",
    "PRODUCTION_ARM_ACTION_COMPONENTS",
    "PRODUCTION_ARM_ACTION_DIM",
    "PRODUCTION_DEFERRED_FULL_8D_PACKET_SCHEMA",
    "PRODUCTION_DEFERRED_PACKET_PORT_SCHEMA",
    "PRODUCTION_FULL_8D_ACTION_COMPONENTS",
    "PRODUCTION_METRIC_TO_NORMALIZED_ADAPTER_SCHEMA",
    "PRODUCTION_NORMALIZED_ARM_ACTION_SCHEMA",
    "ProductionArmActionScale",
    "ProductionBoundPolicyCommandRouter",
    "ProductionDeferredRouteResult",
    "ProductionMetricAdapterRuntimeBinding",
    "ProductionMetricAdapterError",
    "ProductionMetricToNormalizedArmAdapter",
    "StaticAbstractGripperPreflight",
    "assert_selected_production_action_manager_surface",
    "build_canonical_env_step_production_policy_runtime",
    "build_receipt_bound_production_policy_runtime",
]
