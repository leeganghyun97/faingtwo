# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Offline closure of the G2 P1 final-action path.

This module joins already-existing authorities without replacing any of them::

    25-Hz wrist RGB-D + 50-Hz robot state
      -> frozen framewise visual BC
      -> existing nominal feasibility receipt
      -> deterministic 3-D residual composition
      -> existing synchronized-limiter receipt
      -> production metric-to-normalized adapter
      -> immutable deferred full-8-D packet

The final packet is deliberately *not* consumed here.  Only the existing
``CanonicalManagerBasedRLEnvDeferredPacketConsumer`` may later pass it to
``env.step``.  Consequently importing or executing this module cannot start
Isaac, invoke ``ActionManager.process_action``, submit a controller action, or
step physics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Protocol, runtime_checkable

import torch

from geniesim.rl.sac.bc_residual_sac_contract import (
    ResidualActionComposition,
    ResidualAlphaCurriculum,
    ResidualLimiterBindingResult,
    bind_existing_limiter_receipt,
    compose_bc_and_residual_action,
)

from .action_interface import (
    AbstractGripperIntent,
    CartesianControlFrame,
    CartesianSafetyBoundary,
    EEResidualCommand,
    SafetyProjection,
)
from .candidate_a_bc_validation import (
    ACTION_SCHEMA_VERSION,
    CONTROL_DT_S,
    CONTROL_HZ,
    FRAME_CONTRACT_VERSION,
    PHYSICS_HZ,
    RGBD_DT_S,
    RGBD_HZ,
    STUDENT_FIELDS,
)
from .contact_free_visual_bc_runtime import (
    ContactFreeVisualBC,
    prepare_runtime_input,
    validate_metric_output,
)
from .new_policy_safety_authority import FeasibilityReceipt
from .precontact_limiter_contract import PlannerEndpointAdmissionReceipt
from .production_metric_adapter import (
    BufferedNormalizedProductionArmPort,
    DeferredFull8DActionPacket,
    DeferredFull8DActionPacketPort,
    ProductionArmActionScale,
    ProductionMetricToNormalizedArmAdapter,
)


P1_FINAL_ACTION_PATH_SCHEMA = "g2_p1_final_action_path_v1"
P1_CONTROLLER_BOUNDARY = "DEFERRED_FULL_8D_PACKET_BEFORE_CANONICAL_ENV_STEP"


class P1FinalActionPathError(ValueError):
    """Raised when an input attempts to cross a frozen P1 contract."""


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise P1FinalActionPathError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class P1ObservationMetadata:
    """Clock/frame/schema metadata paired with one framewise observation."""

    control_step: int
    control_timestamp_s: float
    camera_timestamp_s: float
    camera_frame_id: int
    control_hz: int = CONTROL_HZ
    rgbd_hz: int = RGBD_HZ
    physics_hz: int = PHYSICS_HZ
    frame_contract_version: str = FRAME_CONTRACT_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION
    frame: str = "robot_root"
    position_unit: str = "m"
    camera_timestamp_source: str = "ACTUAL_ACQUISITION"
    student_fields: tuple[str, ...] = STUDENT_FIELDS

    def validate(self) -> None:
        if type(self.control_step) is not int or self.control_step < 0:
            raise P1FinalActionPathError("control_step must be a nonnegative int")
        if (self.control_hz, self.rgbd_hz, self.physics_hz) != (
            CONTROL_HZ,
            RGBD_HZ,
            PHYSICS_HZ,
        ):
            raise P1FinalActionPathError("P1 requires the 50/25/500-Hz contract")
        control_time = _finite("control_timestamp_s", self.control_timestamp_s)
        camera_time = _finite("camera_timestamp_s", self.camera_timestamp_s)
        if not math.isclose(
            control_time,
            self.control_step * CONTROL_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise P1FinalActionPathError("control timestamp is not the 50-Hz epoch clock")
        if camera_time < 0.0:
            raise P1FinalActionPathError("camera acquisition timestamp cannot be negative")
        if type(self.camera_frame_id) is not int or self.camera_frame_id < 0:
            raise P1FinalActionPathError("camera_frame_id must be a nonnegative int")
        if self.camera_timestamp_source != "ACTUAL_ACQUISITION":
            raise P1FinalActionPathError("synthetic control-derived camera timestamp is forbidden")
        if self.frame_contract_version != FRAME_CONTRACT_VERSION or self.frame != "robot_root":
            raise P1FinalActionPathError("P1 observation frame contract mismatch")
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise P1FinalActionPathError("P1 action schema mismatch")
        if self.position_unit != "m":
            raise P1FinalActionPathError("P1 Cartesian unit must be metres")
        if tuple(self.student_fields) != tuple(STUDENT_FIELDS):
            raise P1FinalActionPathError("student observation inventory mismatch")


class P1ObservationSequenceValidator:
    """Fail-closed 50-Hz control / 25-Hz acquisition sequence validator."""

    def __init__(self) -> None:
        self._previous: P1ObservationMetadata | None = None
        self.camera_reuse_count = 0

    def observe(self, metadata: P1ObservationMetadata) -> None:
        if not isinstance(metadata, P1ObservationMetadata):
            raise P1FinalActionPathError("typed P1 observation metadata is required")
        metadata.validate()
        previous = self._previous
        if previous is not None:
            if metadata.control_step != previous.control_step + 1:
                raise P1FinalActionPathError("control sequence is discontinuous")
            if not math.isclose(
                metadata.control_timestamp_s - previous.control_timestamp_s,
                CONTROL_DT_S,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise P1FinalActionPathError("control cadence is not 50 Hz")
            if metadata.camera_frame_id == previous.camera_frame_id:
                if metadata.camera_timestamp_s != previous.camera_timestamp_s:
                    raise P1FinalActionPathError("reused camera frame changed acquisition timestamp")
                self.camera_reuse_count += 1
            else:
                # The native sensor sequence is authoritative and may expose
                # every other renderer frame at a 25-Hz acquisition cadence;
                # require strict advance, not a synthetic +1 renumbering.
                if metadata.camera_frame_id <= previous.camera_frame_id:
                    raise P1FinalActionPathError("camera acquisition frame ID did not advance")
                if not math.isclose(
                    metadata.camera_timestamp_s - previous.camera_timestamp_s,
                    RGBD_DT_S,
                    rel_tol=0.0,
                    abs_tol=1.0e-5,
                ):
                    raise P1FinalActionPathError("new camera frame cadence is not 25 Hz")
        self._previous = metadata


@dataclass(frozen=True)
class P1StudentObservation:
    """Only the frozen visual-BC student fields; no planner/object GT exists."""

    metadata: P1ObservationMetadata
    right_wrist_rgb: torch.Tensor
    right_wrist_depth_m: torch.Tensor
    right_wrist_depth_valid: torch.Tensor
    ee_pose_robot_root_m_xyzw: torch.Tensor
    right_arm_joint_position_rad: torch.Tensor
    right_arm_joint_velocity_rad_s: torch.Tensor
    right_arm_joint_acceleration_rad_s2: torch.Tensor
    gripper_state_open: torch.Tensor
    previous_policy_action_4d_metric_root_m: torch.Tensor

    def prepare(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.metadata.validate()
        return prepare_runtime_input(
            right_wrist_rgb=self.right_wrist_rgb,
            right_wrist_depth_m=self.right_wrist_depth_m,
            right_wrist_depth_valid=self.right_wrist_depth_valid,
            ee_pose_robot_root_m_xyzw=self.ee_pose_robot_root_m_xyzw,
            right_arm_joint_position_rad=self.right_arm_joint_position_rad,
            right_arm_joint_velocity_rad_s=self.right_arm_joint_velocity_rad_s,
            right_arm_joint_acceleration_rad_s2=self.right_arm_joint_acceleration_rad_s2,
            gripper_state_open=self.gripper_state_open,
            previous_policy_action_4d_metric_root_m=(
                self.previous_policy_action_4d_metric_root_m
            ),
        )


@runtime_checkable
class P1NominalFeasibilityGate(Protocol):
    """Adapter over the existing ``NewPolicySafetyAuthority`` evaluation."""

    def evaluate(self, command: EEResidualCommand) -> FeasibilityReceipt:
        """Return an existing-authority receipt without controller submission."""


@runtime_checkable
class P1FinalLimiterGate(Protocol):
    """Read-only provider of the existing synchronized-limiter receipt."""

    def admit(self, composition: ResidualActionComposition) -> PlannerEndpointAdmissionReceipt:
        """Evaluate the exact post-residual candidate without clipping it."""


class _ExactAcceptedFinalBoundary(CartesianSafetyBoundary):
    """One-command boundary proving the limiter-approved command is unchanged."""

    def __init__(self, expected: EEResidualCommand) -> None:
        self._expected = expected
        self.project_count = 0

    def project(self, command: EEResidualCommand) -> SafetyProjection:
        self.project_count += 1
        if command != self._expected:
            return SafetyProjection(False, None, "P1_FINAL_LIMITER_COMMAND_IDENTITY_MISMATCH")
        return SafetyProjection(True, command)


@dataclass(frozen=True)
class P1ActionPathResult:
    accepted: bool
    reason: str
    bc_nominal_action_4d_metric_root_m: tuple[float, float, float, float]
    nominal_feasibility_accepted: bool
    nominal_feasibility_reason: str
    final_feasibility_accepted: bool | None
    final_feasibility_reason: str | None
    composition: ResidualActionComposition | None
    limiter_binding: ResidualLimiterBindingResult | None
    controller_packet: DeferredFull8DActionPacket | None
    adapter_scale_count: int
    final_boundary_project_count: int
    env_step_call_count: int = 0
    action_manager_process_action_call_count: int = 0
    controller_consumption_count: int = 0
    schema: str = P1_FINAL_ACTION_PATH_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "reason": self.reason,
            "bc_nominal_action_4d_metric_root_m": list(
                self.bc_nominal_action_4d_metric_root_m
            ),
            "nominal_feasibility_accepted": self.nominal_feasibility_accepted,
            "nominal_feasibility_reason": self.nominal_feasibility_reason,
            "final_feasibility_accepted": self.final_feasibility_accepted,
            "final_feasibility_reason": self.final_feasibility_reason,
            "composition": None if self.composition is None else self.composition.payload(),
            "limiter_binding": (
                None if self.limiter_binding is None else asdict(self.limiter_binding)
            ),
            "controller_boundary": P1_CONTROLLER_BOUNDARY,
            "controller_packet": (
                None
                if self.controller_packet is None
                else {
                    "packet_id": self.controller_packet.packet_id,
                    "content_fingerprint": self.controller_packet.content_fingerprint,
                    "values": list(self.controller_packet.values),
                    "shape": list(self.controller_packet.shape),
                    "state": "PENDING_NOT_CONSUMED",
                }
            ),
            "adapter_scale_count": self.adapter_scale_count,
            "final_boundary_project_count": self.final_boundary_project_count,
            "env_step_call_count": self.env_step_call_count,
            "action_manager_process_action_call_count": (
                self.action_manager_process_action_call_count
            ),
            "controller_consumption_count": self.controller_consumption_count,
            "silent_clipping_applied": False,
        }


class P1FinalActionPath:
    """Execute one complete offline P1 graph and stop before ``env.step``."""

    def __init__(
        self,
        *,
        frozen_bc: ContactFreeVisualBC,
        nominal_feasibility_gate: P1NominalFeasibilityGate,
        final_limiter_gate: P1FinalLimiterGate,
        alpha: ResidualAlphaCurriculum = ResidualAlphaCurriculum(),
    ) -> None:
        if not isinstance(frozen_bc, ContactFreeVisualBC):
            raise P1FinalActionPathError("P1 requires the frozen framewise ContactFreeVisualBC")
        if frozen_bc.training:
            raise P1FinalActionPathError("frozen BC must be in eval mode")
        if any(parameter.requires_grad for parameter in frozen_bc.parameters()):
            raise P1FinalActionPathError("frozen BC parameters must have requires_grad=False")
        if not isinstance(nominal_feasibility_gate, P1NominalFeasibilityGate):
            raise P1FinalActionPathError("typed nominal feasibility gate is required")
        if not isinstance(final_limiter_gate, P1FinalLimiterGate):
            raise P1FinalActionPathError("typed final limiter gate is required")
        if not isinstance(alpha, ResidualAlphaCurriculum):
            raise P1FinalActionPathError("explicit residual alpha contract is required")
        self._model = frozen_bc
        self._nominal_gate = nominal_feasibility_gate
        self._final_limiter = final_limiter_gate
        self._alpha = alpha

    def run(
        self,
        *,
        observation: P1StudentObservation,
        raw_residual_metric_root_m: tuple[float, float, float],
    ) -> P1ActionPathResult:
        if not isinstance(observation, P1StudentObservation):
            raise P1FinalActionPathError("typed student observation is required")
        image, proprio = observation.prepare()
        if int(image.shape[0]) != 1:
            raise P1FinalActionPathError("P1 offline path accepts exactly one policy row")
        with torch.inference_mode():
            action_tensor = self._model(image, proprio)
        validate_metric_output(action_tensor)
        action = tuple(float(value) for value in action_tensor[0].detach().cpu().tolist())
        command = EEResidualCommand(
            translation_m=action[:3],
            rotation_rad=(0.0, 0.0, 0.0),
            frame=CartesianControlFrame.ROBOT_ROOT,
        )
        receipt = self._nominal_gate.evaluate(command)
        if not isinstance(receipt, FeasibilityReceipt):
            raise P1FinalActionPathError("nominal gate must return FeasibilityReceipt")
        reason = receipt.reason.value
        if not receipt.accepted:
            return P1ActionPathResult(
                accepted=False,
                reason=f"NOMINAL_FEASIBILITY_REJECT:{reason}",
                bc_nominal_action_4d_metric_root_m=action,
                nominal_feasibility_accepted=False,
                nominal_feasibility_reason=reason,
                final_feasibility_accepted=None,
                final_feasibility_reason=None,
                composition=None,
                limiter_binding=None,
                controller_packet=None,
                adapter_scale_count=0,
                final_boundary_project_count=0,
            )
        if (
            receipt.requested_target is None
            or receipt.validated_target is None
            or receipt.requested_target.source_residual != command
            or receipt.validated_target.source_residual != command
        ):
            raise P1FinalActionPathError("nominal feasibility receipt is not bound to exact BC action")

        composition = compose_bc_and_residual_action(
            bc_action_metric_root_m=action,
            raw_sac_residual_metric_root_m=raw_residual_metric_root_m,
            alpha=self._alpha,
        )
        final_command = EEResidualCommand(
            translation_m=composition.final_xyz_metric_root_m,
            rotation_rad=(0.0, 0.0, 0.0),
            frame=CartesianControlFrame.ROBOT_ROOT,
        )
        # A residual may not inherit the nominal receipt.  Re-evaluate the
        # exact post-composition command before the synchronized limiter.
        final_receipt = self._nominal_gate.evaluate(final_command)
        if not isinstance(final_receipt, FeasibilityReceipt):
            raise P1FinalActionPathError("final feasibility gate must return FeasibilityReceipt")
        final_reason = final_receipt.reason.value
        if not final_receipt.accepted:
            return P1ActionPathResult(
                accepted=False,
                reason=f"FINAL_FEASIBILITY_REJECT:{final_reason}",
                bc_nominal_action_4d_metric_root_m=action,
                nominal_feasibility_accepted=True,
                nominal_feasibility_reason=reason,
                final_feasibility_accepted=False,
                final_feasibility_reason=final_reason,
                composition=composition,
                limiter_binding=None,
                controller_packet=None,
                adapter_scale_count=0,
                final_boundary_project_count=0,
            )
        if (
            final_receipt.requested_target is None
            or final_receipt.validated_target is None
            or final_receipt.requested_target.source_residual != final_command
            or final_receipt.validated_target.source_residual != final_command
        ):
            raise P1FinalActionPathError("final feasibility receipt is not bound to composed action")
        limiter_receipt = self._final_limiter.admit(composition)
        limiter = bind_existing_limiter_receipt(composition, limiter_receipt)
        if not limiter.accepted:
            return P1ActionPathResult(
                accepted=False,
                reason=f"FINAL_LIMITER_REJECT:{limiter.rejection_reason}",
                bc_nominal_action_4d_metric_root_m=action,
                nominal_feasibility_accepted=True,
                nominal_feasibility_reason=reason,
                final_feasibility_accepted=True,
                final_feasibility_reason=final_reason,
                composition=composition,
                limiter_binding=limiter,
                controller_packet=None,
                adapter_scale_count=0,
                final_boundary_project_count=0,
            )

        boundary = _ExactAcceptedFinalBoundary(final_command)
        arm_buffer = BufferedNormalizedProductionArmPort()
        adapter = ProductionMetricToNormalizedArmAdapter(
            downstream_port=arm_buffer,
            scale=ProductionArmActionScale(),
        )
        deferred_port = DeferredFull8DActionPacketPort(
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            binding_id="g2_p1_offline_final_action_path_v1",
        )
        projection = boundary.project(final_command)
        if not projection.accepted or projection.command != final_command:
            raise P1FinalActionPathError("final limiter boundary changed or rejected its bound command")
        # This direct adapter use is the source-documented static/offline path:
        # it buffers arm7 only.  The complete packet is then staged at the
        # same deferred boundary used by the canonical env.step runtime.  No
        # ActionManager/controller method is callable from this sequence.
        adapter.submit_ee_residual(final_command)
        packet = arm_buffer.take_packet(AbstractGripperIntent.OPEN)
        deferred_packet = deferred_port.defer_normalized_action_manager_packet(packet)
        conversion = adapter.last_conversion
        if conversion is None or conversion.any_saturation:
            raise P1FinalActionPathError("controller adapter failed exact no-clipping conversion")
        if conversion.input_translation_m != composition.final_xyz_metric_root_m:
            raise P1FinalActionPathError("controller adapter received a different final action")
        if tuple(deferred_packet.values[3:7]) != (0.0, 0.0, 0.0, 0.0):
            raise P1FinalActionPathError("orientation/elbow authority must remain exactly zero")
        return P1ActionPathResult(
            accepted=True,
            reason="ACCEPTED_DEFERRED_NOT_CONSUMED",
            bc_nominal_action_4d_metric_root_m=action,
            nominal_feasibility_accepted=True,
            nominal_feasibility_reason=reason,
            final_feasibility_accepted=True,
            final_feasibility_reason=final_reason,
            composition=composition,
            limiter_binding=limiter,
            controller_packet=deferred_packet,
            adapter_scale_count=conversion.scale_application_count_after_adapter,
            final_boundary_project_count=boundary.project_count,
        )


__all__ = [
    "P1ActionPathResult",
    "P1FinalActionPath",
    "P1FinalActionPathError",
    "P1FinalLimiterGate",
    "P1NominalFeasibilityGate",
    "P1ObservationMetadata",
    "P1ObservationSequenceValidator",
    "P1StudentObservation",
    "P1_CONTROLLER_BOUNDARY",
    "P1_FINAL_ACTION_PATH_SCHEMA",
]
