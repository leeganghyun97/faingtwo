# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""High-level EE plus abstract-gripper policy boundary for G2.

This module is deliberately independent of Isaac Sim, ROS, and the current
OmniPicker mechanical-authority investigation.  It defines the only command
surface a learned policy may use in this branch:

``[normalised dX, normalised dY, normalised dZ, close_probability]``

or, in an explicitly enabled later mode, the same command with a normalised
rotation vector.  It never contains a joint name, finger target, motor
current, torque, or passive-link command.  A runtime binding may adapt its
``AbstractGripperIntent`` to an *already validated* controller API, but that
binding is outside this pure contract and is not hardware authority evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol, Sequence, runtime_checkable


POLICY_ACTION_SCHEMA_4D = "g2_ee_xyz_gripper_v1"
POLICY_ACTION_SCHEMA_7D = "g2_ee_se3_gripper_v1"
POLICY_CONTROLLER_SURFACE_SCHEMA = "g2_existing_ee_controller_surface_v1"
SIMULATION_EXISTING_MAPPER_ONLY = "SIMULATION_EXISTING_MAPPER_ONLY"


class PolicyActionContractError(ValueError):
    """Raised when a policy command crosses the high-level command boundary."""


class PolicyActionMode(str, Enum):
    """Supported policy action layouts.

    ``TRANSLATION_GRIPPER_4D`` is the default first experiment.  Keeping the
    orientation residual exactly zero removes an unnecessary source of
    variation while cube approach and binary close timing are qualified.
    """

    TRANSLATION_GRIPPER_4D = POLICY_ACTION_SCHEMA_4D
    SE3_GRIPPER_7D = POLICY_ACTION_SCHEMA_7D

    @property
    def dimension(self) -> int:
        return 4 if self is PolicyActionMode.TRANSLATION_GRIPPER_4D else 7


class CartesianControlFrame(str, Enum):
    """Frame used by every translational/rotational residual in this branch."""

    ROBOT_ROOT = "robot_root"


class AbstractGripperIntent(str, Enum):
    """The only gripper semantic exposed to a policy-facing runtime."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"


@dataclass(frozen=True)
class CartesianResidualScale:
    """Controller-normalisation scale, not a safety-limit authority.

    These defaults match the existing high-level G2 controller surface.  The
    downstream safety/IK/controller layer remains solely responsible for
    workspace, joint, velocity, acceleration, collision, and stabilization
    checks.
    """

    translation_m_per_normalized: float = 0.0225
    rotation_rad_per_normalized: float = 0.28125
    control_frame: CartesianControlFrame = CartesianControlFrame.ROBOT_ROOT

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.translation_m_per_normalized)
            or self.translation_m_per_normalized <= 0.0
        ):
            raise PolicyActionContractError("translation scale must be finite and positive")
        if (
            not math.isfinite(self.rotation_rad_per_normalized)
            or self.rotation_rad_per_normalized <= 0.0
        ):
            raise PolicyActionContractError("rotation scale must be finite and positive")
        if not isinstance(self.control_frame, CartesianControlFrame):
            raise PolicyActionContractError("control_frame must be a CartesianControlFrame")


@dataclass(frozen=True)
class HighLevelPolicyAction:
    """A finite, normalised high-level policy request.

    The final component is a continuous close probability in ``[0, 1]``.  It
    is intentionally not a finger position.  Hysteresis converts it into an
    abstract OPEN/CLOSE intent only after the Cartesian request has passed
    through the safety boundary.
    """

    values: tuple[float, ...]
    mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D

    def __post_init__(self) -> None:
        if len(self.values) != self.mode.dimension:
            raise PolicyActionContractError(
                f"{self.mode.value} requires {self.mode.dimension} values"
            )
        if not all(math.isfinite(float(value)) for value in self.values):
            raise PolicyActionContractError("policy action must be finite")
        if any(abs(float(value)) > 1.0 for value in self.values[:-1]):
            raise PolicyActionContractError(
                "normalised EE residual components must be in [-1, 1]"
            )
        if not 0.0 <= self.gripper_probability <= 1.0:
            raise PolicyActionContractError("gripper probability must be in [0, 1]")

    @classmethod
    def from_sequence(
        cls,
        values: Sequence[float],
        *,
        mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
    ) -> "HighLevelPolicyAction":
        return cls(tuple(float(value) for value in values), mode=mode)

    @property
    def translation_normalized(self) -> tuple[float, float, float]:
        return self.values[0], self.values[1], self.values[2]

    @property
    def rotation_normalized(self) -> tuple[float, float, float]:
        if self.mode is PolicyActionMode.TRANSLATION_GRIPPER_4D:
            return (0.0, 0.0, 0.0)
        return self.values[3], self.values[4], self.values[5]

    @property
    def gripper_probability(self) -> float:
        return self.values[-1]


@dataclass(frozen=True)
class EEResidualCommand:
    """Physical-unit Cartesian residual passed to the safety/controller layer."""

    translation_m: tuple[float, float, float]
    rotation_rad: tuple[float, float, float]
    frame: CartesianControlFrame = CartesianControlFrame.ROBOT_ROOT

    def __post_init__(self) -> None:
        components = (*self.translation_m, *self.rotation_rad)
        if len(self.translation_m) != 3 or len(self.rotation_rad) != 3:
            raise PolicyActionContractError("EE residual must contain xyz and rotation xyz")
        if not all(math.isfinite(float(value)) for value in components):
            raise PolicyActionContractError("EE residual must be finite")
        if not isinstance(self.frame, CartesianControlFrame):
            raise PolicyActionContractError("EE residual frame must be explicit")


def decode_ee_residual(
    action: HighLevelPolicyAction,
    *,
    scale: CartesianResidualScale = CartesianResidualScale(),
) -> EEResidualCommand:
    """Decode a policy request without applying any safety policy."""

    return EEResidualCommand(
        translation_m=tuple(
            value * scale.translation_m_per_normalized
            for value in action.translation_normalized
        ),
        rotation_rad=tuple(
            value * scale.rotation_rad_per_normalized
            for value in action.rotation_normalized
        ),
        frame=scale.control_frame,
    )


@dataclass(frozen=True)
class GripperHysteresisConfig:
    """Stable OPEN/CLOSE conversion for the continuous policy output."""

    open_threshold: float = 0.30
    close_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.open_threshold)
            and math.isfinite(self.close_threshold)
            and 0.0 <= self.open_threshold < self.close_threshold <= 1.0
        ):
            raise PolicyActionContractError(
                "hysteresis thresholds must satisfy 0 <= open < close <= 1"
            )


class GripperHysteresisLatch:
    """Maps scalar ``g`` to a stateful abstract gripper intent.

    Boundary values deliberately remain in the hysteresis band: only
    ``g > close_threshold`` closes and only ``g < open_threshold`` opens.
    Reset always returns OPEN; this does not replace the downstream
    controller's own reset-open and contact-hold protections.
    """

    def __init__(
        self,
        config: GripperHysteresisConfig = GripperHysteresisConfig(),
        *,
        initial_intent: AbstractGripperIntent = AbstractGripperIntent.OPEN,
    ) -> None:
        self.config = config
        self._intent = initial_intent

    @property
    def intent(self) -> AbstractGripperIntent:
        return self._intent

    def reset(self) -> AbstractGripperIntent:
        self._intent = AbstractGripperIntent.OPEN
        return self._intent

    def proposed_intent(self, probability: float) -> AbstractGripperIntent:
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise PolicyActionContractError("gripper probability must be finite in [0, 1]")
        if value > self.config.close_threshold:
            return AbstractGripperIntent.CLOSE
        if value < self.config.open_threshold:
            return AbstractGripperIntent.OPEN
        return self._intent

    def commit(self, intent: AbstractGripperIntent) -> AbstractGripperIntent:
        if not isinstance(intent, AbstractGripperIntent):
            raise PolicyActionContractError("gripper intent must be AbstractGripperIntent")
        self._intent = intent
        return self._intent

    def update(self, probability: float) -> AbstractGripperIntent:
        """Pure-contract convenience; runtime routes should commit after feedback."""

        return self.commit(self.proposed_intent(probability))


def gripper_intent_to_existing_simulation_sign(intent: AbstractGripperIntent) -> float:
    """Adapt an abstract intent to the existing *simulation* controller sign.

    This is an integration adapter only.  It does not name or target any
    OmniPicker mechanism, and must not be interpreted as a hardware
    motor-to-pivot authority while H4.7 remains unresolved.
    """

    return 1.0 if intent is AbstractGripperIntent.OPEN else -1.0


def expand_to_existing_controller_7d(
    action: HighLevelPolicyAction,
    *,
    gripper_intent: AbstractGripperIntent,
) -> tuple[float, float, float, float, float, float, float]:
    """Create the pre-existing EE controller surface, never a joint command."""

    return (
        *action.translation_normalized,
        *action.rotation_normalized,
        gripper_intent_to_existing_simulation_sign(gripper_intent),
    )


def expand_to_existing_controller_8d(
    action: HighLevelPolicyAction,
    *,
    gripper_intent: AbstractGripperIntent,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compatibility expansion for an old controller surface with elbow slot.

    The inserted elbow request is exactly zero.  This branch does not learn
    redundancy or individual mechanism motion.
    """

    command_7d = expand_to_existing_controller_7d(
        action, gripper_intent=gripper_intent
    )
    return (*command_7d[:6], 0.0, command_7d[6])


@dataclass(frozen=True)
class SafetyProjection:
    """Outcome of the existing Cartesian safety boundary."""

    accepted: bool
    command: EEResidualCommand | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.accepted and self.command is None:
            raise PolicyActionContractError("accepted safety projection requires a command")
        if not self.accepted and self.command is not None:
            raise PolicyActionContractError("rejected safety projection cannot expose a command")


@runtime_checkable
class CartesianSafetyBoundary(Protocol):
    """Port owned by workspace/IK/limit/collision safety implementation."""

    def project(self, command: EEResidualCommand) -> SafetyProjection:
        """Clamp or reject an EE request before it reaches a controller."""


@runtime_checkable
class ExistingEEController(Protocol):
    """High-level EE controller port; no articulation targets are exposed."""

    def submit_ee_residual(self, command: EEResidualCommand) -> None:
        """Submit a safety-projected Cartesian residual."""


@runtime_checkable
class ExistingAbstractGripperController(Protocol):
    """Validated OPEN/CLOSE interface owned below the policy boundary."""

    def submit_gripper_intent(self, intent: AbstractGripperIntent) -> "GripperSubmission":
        """Request an intent and return its controller-side acceptance/state."""


@dataclass(frozen=True)
class GripperSubmission:
    """High-level feedback from the existing safe gripper controller.

    It deliberately exposes only the abstract effective state.  A binding may
    use its own reset-open, contact-hold, and safe-close checks internally,
    but it cannot surface low-level actuator targets through this type.
    """

    accepted: bool
    effective_intent: AbstractGripperIntent
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.effective_intent, AbstractGripperIntent):
            raise PolicyActionContractError("effective gripper state must be abstract")
        if self.accepted and self.reason:
            raise PolicyActionContractError("accepted gripper submission cannot carry a rejection reason")


@dataclass(frozen=True)
class PolicyRouteResult:
    """Audit record for one policy request routed through the safe boundary."""

    accepted: bool
    reason: str
    ee_command: EEResidualCommand | None
    gripper_intent: AbstractGripperIntent
    requested_gripper_intent: AbstractGripperIntent
    gripper_command_emitted: bool
    gripper_submission: GripperSubmission | None


class PolicyCommandRouter:
    """Routes a policy action without permitting low-level command bypasses."""

    def __init__(
        self,
        *,
        safety_boundary: CartesianSafetyBoundary,
        ee_controller: ExistingEEController,
        gripper_controller: ExistingAbstractGripperController,
        action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D,
        scale: CartesianResidualScale = CartesianResidualScale(),
        hysteresis: GripperHysteresisConfig = GripperHysteresisConfig(),
    ) -> None:
        if not isinstance(action_mode, PolicyActionMode):
            raise PolicyActionContractError("router action_mode must be PolicyActionMode")
        if not isinstance(scale, CartesianResidualScale):
            raise PolicyActionContractError("router scale must be CartesianResidualScale")
        if not isinstance(hysteresis, GripperHysteresisConfig):
            raise PolicyActionContractError("router hysteresis must be GripperHysteresisConfig")
        # Some safety boundaries (the new feasibility bridge) issue a
        # single-use receipt which must be consumed by that same object.  A
        # separately wired raw controller would bypass its epoch/fingerprint
        # check, so reject that wiring at construction rather than hoping the
        # caller keeps two ports paired correctly.
        if (
            bool(getattr(safety_boundary, "requires_atomic_accepted_submission", False))
            and safety_boundary is not ee_controller
        ):
            raise PolicyActionContractError(
                "receipt-bound safety boundary must also be the EE submission controller"
            )
        self._safety_boundary = safety_boundary
        self._ee_controller = ee_controller
        self._gripper_controller = gripper_controller
        self._action_mode = action_mode
        self._scale = scale
        self._latch = GripperHysteresisLatch(hysteresis)

    @property
    def gripper_intent(self) -> AbstractGripperIntent:
        return self._latch.intent

    @property
    def action_mode(self) -> PolicyActionMode:
        return self._action_mode

    def assert_compatible_policy_config(self, policy_config: object) -> None:
        """Fail before the first command if checkpoint and router semantics drift.

        ``PolicyBCConfig`` owns these three fields, but this module deliberately
        avoids importing the BC model to keep the controller boundary free of
        Torch/model dependencies.  Structural checking here prevents a loaded
        checkpoint from being paired with a different action layout, frame/
        scale, or gripper hysteresis at deployment.
        """

        try:
            action_mode = getattr(policy_config, "action_mode")
            residual_scale = getattr(policy_config, "residual_scale")
            gripper_hysteresis = getattr(policy_config, "gripper_hysteresis")
        except AttributeError as error:
            raise PolicyActionContractError(
                "policy config must expose action_mode, residual_scale, and gripper_hysteresis"
            ) from error
        if (
            action_mode is not self._action_mode
            or residual_scale != self._scale
            or gripper_hysteresis != self._latch.config
        ):
            raise PolicyActionContractError(
                "policy checkpoint config and controller router semantics differ"
            )

    def _submit_gripper_intent(
        self, intent: AbstractGripperIntent
    ) -> GripperSubmission:
        submission = self._gripper_controller.submit_gripper_intent(intent)
        if not isinstance(submission, GripperSubmission):
            raise PolicyActionContractError(
                "existing gripper controller must return GripperSubmission"
            )
        return submission

    def synchronize_gripper_intent(self, observed_intent: AbstractGripperIntent) -> None:
        """Synchronize from trusted controller-level feedback, never joint state."""

        self._latch.commit(observed_intent)

    def reset(self) -> GripperSubmission:
        """Request OPEN and commit it only after controller acknowledgment."""

        submission = self._submit_gripper_intent(AbstractGripperIntent.OPEN)
        self._latch.commit(submission.effective_intent)
        return submission

    def route(self, action: HighLevelPolicyAction) -> PolicyRouteResult:
        if action.mode is not self._action_mode:
            raise PolicyActionContractError(
                f"router is locked to {self._action_mode.value}, got {action.mode.value}"
            )
        requested = decode_ee_residual(action, scale=self._scale)
        projection = self._safety_boundary.project(requested)
        if not projection.accepted:
            return PolicyRouteResult(
                accepted=False,
                reason=projection.reason or "EE_COMMAND_REJECTED_BY_SAFETY",
                ee_command=None,
                gripper_intent=self._latch.intent,
                requested_gripper_intent=self._latch.intent,
                gripper_command_emitted=False,
                gripper_submission=None,
            )

        assert projection.command is not None
        if projection.command.frame is not requested.frame:
            return PolicyRouteResult(
                accepted=False,
                reason="SAFETY_PROJECTION_FRAME_MISMATCH",
                ee_command=None,
                gripper_intent=self._latch.intent,
                requested_gripper_intent=self._latch.intent,
                gripper_command_emitted=False,
                gripper_submission=None,
            )
        self._ee_controller.submit_ee_residual(projection.command)
        previous_intent = self._latch.intent
        requested_intent = self._latch.proposed_intent(action.gripper_probability)
        emitted = requested_intent is not previous_intent
        submission: GripperSubmission | None = None
        reason = ""
        if emitted:
            submission = self._submit_gripper_intent(requested_intent)
            self._latch.commit(submission.effective_intent)
            if not submission.accepted:
                reason = submission.reason or "GRIPPER_INTENT_REJECTED_BY_CONTROLLER"
        return PolicyRouteResult(
            accepted=submission is None or submission.accepted,
            reason=reason,
            ee_command=projection.command,
            gripper_intent=self._latch.intent,
            requested_gripper_intent=requested_intent,
            gripper_command_emitted=emitted,
            gripper_submission=submission,
        )


__all__ = [
    "AbstractGripperIntent",
    "CartesianResidualScale",
    "CartesianControlFrame",
    "CartesianSafetyBoundary",
    "EEResidualCommand",
    "ExistingAbstractGripperController",
    "ExistingEEController",
    "GripperHysteresisConfig",
    "GripperHysteresisLatch",
    "GripperSubmission",
    "HighLevelPolicyAction",
    "POLICY_ACTION_SCHEMA_4D",
    "POLICY_ACTION_SCHEMA_7D",
    "POLICY_CONTROLLER_SURFACE_SCHEMA",
    "PolicyActionContractError",
    "PolicyActionMode",
    "PolicyCommandRouter",
    "PolicyRouteResult",
    "SIMULATION_EXISTING_MAPPER_ONLY",
    "SafetyProjection",
    "decode_ee_residual",
    "expand_to_existing_controller_7d",
    "expand_to_existing_controller_8d",
    "gripper_intent_to_existing_simulation_sign",
]
