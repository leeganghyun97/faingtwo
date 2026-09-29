# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Deterministic policy-pre-controller EE safety boundary.

This module deliberately owns a *small* contract.  It does not define a
Cartesian workspace, modify a target, solve IK, tune a controller, or expose
joint/actuator commands.  It checks that a high-level policy request obeys
the already-published 4-D branch contract and delegates kinematic feasibility
to an explicitly supplied, side-effect-free dry-run authority.

The dry-run provider is intentionally required for acceptance.  An absent,
unresolved, malformed, or exception-raising provider therefore rejects the
request with an immutable receipt.  This prevents this helper from silently
becoming a new, unapproved workspace or IK authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Protocol, Sequence, runtime_checkable

from .action_interface import (
    CartesianControlFrame,
    CartesianResidualScale,
    EEResidualCommand,
    HighLevelPolicyAction,
    PolicyActionContractError,
    PolicyActionMode,
    SafetyProjection,
    decode_ee_residual,
)


EE_SAFETY_VALIDATOR_SCHEMA = "g2_policy_branch_ee_safety_validator_v1"
EE_SAFETY_VALIDATOR_SCOPE = "POLICY_PRE_CONTROLLER_ONLY"


class P0SafetyRejectReason(str, Enum):
    """Stable policy-observable outcomes for the pre-controller boundary."""

    ACCEPT = "ACCEPT"
    INVALID_INPUT = "INVALID_INPUT"
    NONFINITE_TARGET = "NONFINITE_TARGET"
    ACTION_BOUND_VIOLATION = "ACTION_BOUND_VIOLATION"
    ACTION_SCHEMA_VIOLATION = "ACTION_SCHEMA_VIOLATION"
    ORIENTATION_CONTRACT_VIOLATION = "ORIENTATION_CONTRACT_VIOLATION"
    FRAME_CONTRACT_VIOLATION = "FRAME_CONTRACT_VIOLATION"
    IK_NO_SOLUTION = "IK_NO_SOLUTION"
    IK_NONFINITE = "IK_NONFINITE"
    IK_RESIDUAL_EXCEEDED = "IK_RESIDUAL_EXCEEDED"
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
    INTERNAL_VALIDATION_FAILURE = "INTERNAL_VALIDATION_FAILURE"


@dataclass(frozen=True)
class EESafetyValidatorConfig:
    """Immutable policy-contract facts; deliberately no workspace box exists.

    ``residual_scale`` is the existing controller-normalisation contract, not
    a controller queue bound or a robot-workspace authority.  It gives the
    validator the exact policy maximum of 22.5 mm per Cartesian axis.
    """

    residual_scale: CartesianResidualScale = CartesianResidualScale()
    action_mode: PolicyActionMode = PolicyActionMode.TRANSLATION_GRIPPER_4D
    schema: str = EE_SAFETY_VALIDATOR_SCHEMA
    scope: str = EE_SAFETY_VALIDATOR_SCOPE
    orientation_must_be_exact_zero: bool = True
    target_mutation_allowed: bool = False
    arbitrary_workspace_rule_added: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.residual_scale, CartesianResidualScale):
            raise ValueError("residual_scale must be CartesianResidualScale")
        if self.action_mode is not PolicyActionMode.TRANSLATION_GRIPPER_4D:
            raise ValueError("P0 EE safety validator is fixed to 4-D policy actions")
        if self.schema != EE_SAFETY_VALIDATOR_SCHEMA:
            raise ValueError("unsupported EE safety validator schema")
        if self.scope != EE_SAFETY_VALIDATOR_SCOPE:
            raise ValueError("EE safety validator scope must be policy-pre-controller only")
        if not self.orientation_must_be_exact_zero:
            raise ValueError("P0 requires exact-zero orientation")
        if self.target_mutation_allowed:
            raise ValueError("P0 EE safety validator must not mutate targets")
        if self.arbitrary_workspace_rule_added:
            raise ValueError("P0 EE safety validator must not define a workspace rule")

    @property
    def maximum_translation_delta_m_per_axis(self) -> float:
        return self.residual_scale.translation_m_per_normalized

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "scope": self.scope,
            "policy_action_schema": self.action_mode.value,
            "translation_scale_m_per_normalized": (
                self.residual_scale.translation_m_per_normalized
            ),
            "maximum_translation_delta_m_per_axis": (
                self.maximum_translation_delta_m_per_axis
            ),
            "control_frame": self.residual_scale.control_frame.value,
            "orientation_must_be_exact_zero": self.orientation_must_be_exact_zero,
            "target_mutation_allowed": self.target_mutation_allowed,
            "arbitrary_workspace_rule_added": self.arbitrary_workspace_rule_added,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DryRunIKResult:
    """A side-effect-free kinematic-feasibility result supplied by its owner.

    The validator does not choose a residual threshold.  A provider may mark
    a result feasible only when its *own documented/source-owned* acceptance
    threshold has been applied.  The provider must identify that authority.
    Unit-test fakes can exercise the branches but are never live authority
    evidence.
    """

    feasible: bool
    reason: P0SafetyRejectReason
    authority_source_path: str
    authority_symbol: str
    authority_id: str
    solution_joint_position_rad: tuple[float, ...] | None = None
    residual: float | None = None
    residual_threshold_authority: str | None = None
    joint_limits_satisfied: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, P0SafetyRejectReason):
            raise ValueError("dry-run reason must be P0SafetyRejectReason")
        for label, value in (
            ("authority_source_path", self.authority_source_path),
            ("authority_symbol", self.authority_symbol),
            ("authority_id", self.authority_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"dry-run {label} must be a non-empty string")
        if self.feasible:
            if self.reason is not P0SafetyRejectReason.ACCEPT:
                raise ValueError("feasible dry-run result must use ACCEPT")
            if not self.solution_joint_position_rad or not all(
                math.isfinite(float(value)) for value in self.solution_joint_position_rad
            ):
                raise ValueError("feasible dry-run result requires finite joint solution")
            if self.residual is None or not math.isfinite(float(self.residual)):
                raise ValueError("feasible dry-run result requires finite residual")
            if not self.residual_threshold_authority:
                raise ValueError(
                    "feasible dry-run result requires residual threshold authority"
                )
            if self.joint_limits_satisfied is not True:
                raise ValueError(
                    "feasible dry-run result requires explicit joint-limit satisfaction"
                )
        elif self.reason is P0SafetyRejectReason.ACCEPT:
            raise ValueError("infeasible dry-run result cannot use ACCEPT")

    def public_payload(self) -> dict[str, object]:
        """Return evidence without leaking an IK joint target downstream."""

        return {
            "feasible": self.feasible,
            "reason": self.reason.value,
            "authority_source_path": self.authority_source_path,
            "authority_symbol": self.authority_symbol,
            "authority_id": self.authority_id,
            "solution_dimension": (
                None
                if self.solution_joint_position_rad is None
                else len(self.solution_joint_position_rad)
            ),
            "solution_finite": (
                None
                if self.solution_joint_position_rad is None
                else all(math.isfinite(float(value)) for value in self.solution_joint_position_rad)
            ),
            "residual": self.residual,
            "residual_threshold_authority": self.residual_threshold_authority,
            "joint_limits_satisfied": self.joint_limits_satisfied,
        }


@runtime_checkable
class DryRunIKFeasibility(Protocol):
    """Side-effect-free production-kinematics feasibility injection port."""

    def evaluate_dry_run(self, target: EEResidualCommand) -> DryRunIKResult:
        """Return feasibility; must never submit an execution command."""


class UnresolvedDryRunProvider:
    """Fail-closed default when no source-authoritative dry-run query exists."""

    authority_source_path = "UNRESOLVED"
    authority_symbol = "UNRESOLVED"
    authority_id = "NO_SOURCE_AUTHORITATIVE_DRY_RUN_IK"

    def evaluate_dry_run(self, target: EEResidualCommand) -> DryRunIKResult:
        del target
        return DryRunIKResult(
            feasible=False,
            reason=P0SafetyRejectReason.INTERNAL_VALIDATION_FAILURE,
            authority_source_path=self.authority_source_path,
            authority_symbol=self.authority_symbol,
            authority_id=self.authority_id,
            residual_threshold_authority="UNRESOLVED",
            joint_limits_satisfied=None,
        )


def _target_payload(target: EEResidualCommand | None) -> dict[str, object] | None:
    if target is None:
        return None
    return {
        "translation_m": list(target.translation_m),
        "rotation_rad": list(target.rotation_rad),
        "frame": target.frame.value,
    }


def _safe_action_payload(values: tuple[float, ...] | None) -> list[object] | None:
    if values is None:
        return None
    result: list[object] = []
    for value in values:
        if math.isnan(value):
            result.append("NaN")
        elif math.isinf(value):
            result.append("Infinity" if value > 0.0 else "-Infinity")
        else:
            result.append(value)
    return result


@dataclass(frozen=True)
class SafetyValidationReceipt:
    """Immutable, policy-observable decision made before any controller call."""

    accepted: bool
    reason: P0SafetyRejectReason
    requested_action: tuple[float, ...] | None
    requested_target: EEResidualCommand | None
    validated_target: EEResidualCommand | None
    target_mutated: bool
    downstream_submission_performed: bool
    config_fingerprint: str
    dry_run: DryRunIKResult | None
    schema: str = EE_SAFETY_VALIDATOR_SCHEMA
    scope: str = EE_SAFETY_VALIDATOR_SCOPE

    def __post_init__(self) -> None:
        if self.schema != EE_SAFETY_VALIDATOR_SCHEMA:
            raise ValueError("unsupported safety receipt schema")
        if self.scope != EE_SAFETY_VALIDATOR_SCOPE:
            raise ValueError("safety receipt scope mismatch")
        if not isinstance(self.reason, P0SafetyRejectReason):
            raise ValueError("safety receipt reason must be P0SafetyRejectReason")
        if len(self.config_fingerprint) != 64:
            raise ValueError("safety receipt requires a SHA-256 config fingerprint")
        if self.target_mutated:
            raise ValueError("P0 safety receipts cannot represent target mutation")
        if self.accepted:
            if self.reason is not P0SafetyRejectReason.ACCEPT:
                raise ValueError("accepted safety receipt must use ACCEPT")
            if self.requested_target is None or self.validated_target is None:
                raise ValueError("accepted safety receipt requires requested/validated target")
            if self.requested_target != self.validated_target:
                raise ValueError("accepted safety receipt must preserve target exactly")
        else:
            if self.reason is P0SafetyRejectReason.ACCEPT:
                raise ValueError("rejected safety receipt cannot use ACCEPT")
            if self.validated_target is not None:
                raise ValueError("rejected safety receipt cannot expose controller target")
            if self.downstream_submission_performed:
                raise ValueError("rejected safety receipt cannot reach downstream")

    @property
    def authority_source_path(self) -> str | None:
        return None if self.dry_run is None else self.dry_run.authority_source_path

    @property
    def authority_symbol(self) -> str | None:
        return None if self.dry_run is None else self.dry_run.authority_symbol

    @property
    def authority_id(self) -> str | None:
        return None if self.dry_run is None else self.dry_run.authority_id

    def with_downstream_submission(self) -> "SafetyValidationReceipt":
        """Create the post-submit receipt only after a real controller call."""

        if not self.accepted:
            raise ValueError("cannot mark rejected safety receipt as submitted")
        return replace(self, downstream_submission_performed=True)

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "scope": self.scope,
            "accepted": self.accepted,
            "reason": self.reason.value,
            "requested_action": _safe_action_payload(self.requested_action),
            "requested_target": _target_payload(self.requested_target),
            "validated_target": _target_payload(self.validated_target),
            "target_mutated": self.target_mutated,
            "downstream_submission_performed": self.downstream_submission_performed,
            "config_fingerprint": self.config_fingerprint,
            "dry_run": None if self.dry_run is None else self.dry_run.public_payload(),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class AcceptedActionRecorder(Protocol):
    """Minimal port used to preserve existing lag-one previous-action semantics."""

    def record_accepted(self, action: HighLevelPolicyAction) -> HighLevelPolicyAction:
        """Record a fully accepted branch action after downstream submission."""


class EESafetyValidator:
    """Validate-only boundary that never clamps, re-scales, or executes a target."""

    def __init__(
        self,
        *,
        config: EESafetyValidatorConfig = EESafetyValidatorConfig(),
        dry_run_provider: DryRunIKFeasibility | None = None,
    ) -> None:
        if not isinstance(config, EESafetyValidatorConfig):
            raise ValueError("config must be EESafetyValidatorConfig")
        self.config = config
        self._dry_run_provider: DryRunIKFeasibility = (
            UnresolvedDryRunProvider()
            if dry_run_provider is None
            else dry_run_provider
        )
        self._last_receipt: SafetyValidationReceipt | None = None

    @property
    def last_receipt(self) -> SafetyValidationReceipt | None:
        """Most recent immutable decision, including every rejected command."""

        return self._last_receipt

    def _receipt(
        self,
        *,
        accepted: bool,
        reason: P0SafetyRejectReason,
        requested_action: tuple[float, ...] | None,
        requested_target: EEResidualCommand | None,
        dry_run: DryRunIKResult | None = None,
    ) -> SafetyValidationReceipt:
        receipt = SafetyValidationReceipt(
            accepted=accepted,
            reason=reason,
            requested_action=requested_action,
            requested_target=requested_target,
            validated_target=requested_target if accepted else None,
            target_mutated=False,
            downstream_submission_performed=False,
            config_fingerprint=self.config.fingerprint(),
            dry_run=dry_run,
        )
        self._last_receipt = receipt
        return receipt

    @staticmethod
    def _extract_raw_action(raw_action: object) -> tuple[float, ...] | None:
        if isinstance(raw_action, HighLevelPolicyAction):
            return raw_action.values
        if isinstance(raw_action, Sequence) and not isinstance(
            raw_action, (str, bytes, bytearray)
        ):
            try:
                return tuple(float(value) for value in raw_action)
            except (TypeError, ValueError, OverflowError):
                return None
        return None

    def validate_policy_action(self, raw_action: object) -> SafetyValidationReceipt:
        """Decode and validate one raw branch action without a controller call."""

        values = self._extract_raw_action(raw_action)
        if values is None:
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.INVALID_INPUT,
                requested_action=None,
                requested_target=None,
            )
        # A diagnostic/non-policy caller can inject the old 7-D controller
        # shape.  Make a non-zero orientation failure explicit rather than
        # hiding it behind a generic dimension error.  A zero-rotation 7-D
        # request still violates the frozen 4-D policy schema below.
        if len(values) == PolicyActionMode.SE3_GRIPPER_7D.dimension:
            if not all(math.isfinite(value) for value in values):
                return self._receipt(
                    accepted=False,
                    reason=P0SafetyRejectReason.INVALID_INPUT,
                    requested_action=values,
                    requested_target=None,
                )
            if tuple(values[3:6]) != (0.0, 0.0, 0.0):
                return self._receipt(
                    accepted=False,
                    reason=P0SafetyRejectReason.ORIENTATION_CONTRACT_VIOLATION,
                    requested_action=values,
                    requested_target=None,
                )
        if len(values) != self.config.action_mode.dimension:
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.ACTION_SCHEMA_VIOLATION,
                requested_action=values,
                requested_target=None,
            )
        if not all(math.isfinite(value) for value in values):
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.INVALID_INPUT,
                requested_action=values,
                requested_target=None,
            )
        if any(abs(value) > 1.0 for value in values[:3]) or not 0.0 <= values[3] <= 1.0:
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.ACTION_BOUND_VIOLATION,
                requested_action=values,
                requested_target=None,
            )
        try:
            action = HighLevelPolicyAction.from_sequence(
                values, mode=self.config.action_mode
            )
            target = decode_ee_residual(action, scale=self.config.residual_scale)
        except (PolicyActionContractError, TypeError, ValueError):
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.INVALID_INPUT,
                requested_action=values,
                requested_target=None,
            )
        return self._validate_target(target, requested_action=action.values)

    def validate(self, target: object) -> SafetyValidationReceipt:
        """Validate a decoded EE target using the same no-mutation contract.

        This method implements the existing ``CartesianSafetyBoundary``-style
        target boundary expected by the router.  It is intentionally distinct
        from :meth:`validate_policy_action`, which also gives malformed raw
        policy inputs an explicit receipt.
        """

        return self._validate_target(target, requested_action=None)

    def _validate_target(
        self,
        target: object,
        *,
        requested_action: tuple[float, ...] | None,
    ) -> SafetyValidationReceipt:
        if not isinstance(target, EEResidualCommand):
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.INVALID_INPUT,
                requested_action=requested_action,
                requested_target=None,
            )
        components = (*target.translation_m, *target.rotation_rad)
        if not all(math.isfinite(float(value)) for value in components):
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.NONFINITE_TARGET,
                requested_action=requested_action,
                requested_target=target,
            )
        if target.frame is not self.config.residual_scale.control_frame:
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.FRAME_CONTRACT_VIOLATION,
                requested_action=requested_action,
                requested_target=target,
            )
        if self.config.orientation_must_be_exact_zero and target.rotation_rad != (0.0, 0.0, 0.0):
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.ORIENTATION_CONTRACT_VIOLATION,
                requested_action=requested_action,
                requested_target=target,
            )
        maximum = self.config.maximum_translation_delta_m_per_axis
        if any(abs(float(value)) > maximum for value in target.translation_m):
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.ACTION_BOUND_VIOLATION,
                requested_action=requested_action,
                requested_target=target,
            )
        try:
            dry_run = self._dry_run_provider.evaluate_dry_run(target)
            if not isinstance(dry_run, DryRunIKResult):
                raise TypeError("dry-run provider must return DryRunIKResult")
        except Exception:
            return self._receipt(
                accepted=False,
                reason=P0SafetyRejectReason.INTERNAL_VALIDATION_FAILURE,
                requested_action=requested_action,
                requested_target=target,
                dry_run=None,
            )
        if not dry_run.feasible:
            return self._receipt(
                accepted=False,
                reason=dry_run.reason,
                requested_action=requested_action,
                requested_target=target,
                dry_run=dry_run,
            )
        # ``DryRunIKResult.__post_init__`` proves the finite solution,
        # source-owned residual acceptance, and production-limit satisfaction.
        return self._receipt(
            accepted=True,
            reason=P0SafetyRejectReason.ACCEPT,
            requested_action=requested_action,
            requested_target=target,
            dry_run=dry_run,
        )

    def project(self, command: EEResidualCommand) -> SafetyProjection:
        """Router-compatible non-mutating adapter with receipt observability."""

        receipt = self.validate(command)
        return SafetyProjection(
            accepted=receipt.accepted,
            command=receipt.validated_target,
            reason=receipt.reason.value,
        )

    @staticmethod
    def record_previous_action_after_submission(
        *,
        receipt: SafetyValidationReceipt,
        action: HighLevelPolicyAction,
        recorder: AcceptedActionRecorder,
    ) -> bool:
        """Preserve lag-one semantics: rejects can never update prior action.

        The caller must mark the immutable receipt with
        :meth:`SafetyValidationReceipt.with_downstream_submission` only after
        the existing controller target setter has actually been called.
        """

        if not receipt.accepted or not receipt.downstream_submission_performed:
            return False
        recorder.record_accepted(action)
        return True


__all__ = [
    "AcceptedActionRecorder",
    "DryRunIKFeasibility",
    "DryRunIKResult",
    "EESafetyValidator",
    "EESafetyValidatorConfig",
    "EE_SAFETY_VALIDATOR_SCHEMA",
    "EE_SAFETY_VALIDATOR_SCOPE",
    "P0SafetyRejectReason",
    "SafetyValidationReceipt",
    "UnresolvedDryRunProvider",
]
