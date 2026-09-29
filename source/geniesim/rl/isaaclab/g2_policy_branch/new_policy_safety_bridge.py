# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Narrow static/live adapter for the new policy feasibility authority.

The bridge owns only receipt-to-controller binding.  It does not implement
IK, FK, joint commands, workspace rules, or gripper control.  In particular,
it never turns a rejected receipt into a controller call and it verifies the
same snapshot epoch immediately before an accepted residual is submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol, runtime_checkable

from .action_interface import (
    EEResidualCommand,
    ExistingEEController,
    PolicyActionContractError,
    SafetyProjection,
)
from .new_policy_safety_authority import (
    FeasibilityReceipt,
    FeasibilityRejectReason,
    NewPolicySafetyAuthority,
    ProductionIKSnapshot,
)


NEW_POLICY_SAFETY_BRIDGE_SCHEMA = "g2_new_policy_safety_bridge_v1"


@runtime_checkable
class ReadOnlyProductionIKSnapshotProvider(Protocol):
    """A future live adapter must provide a cloned, read-only snapshot.

    A live adapter must serialize ``capture_snapshot`` and the immediately
    following controller submission within one control-cycle critical section.
    This static bridge can reject a changed snapshot, but cannot itself lock a
    foreign controller or timeline.
    """

    def capture_snapshot(self) -> ProductionIKSnapshot:
        """Capture no controller/action/queue/timeline mutation."""


def _command_fingerprint(command: EEResidualCommand) -> str:
    payload = {
        "translation_m": list(command.translation_m),
        "rotation_rad": list(command.rotation_rad),
        "frame": command.frame.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreControllerSubmissionEvidence:
    """Proof that one accepted receipt reached exactly one EE controller call."""

    snapshot_id: str
    control_epoch: int
    receipt_fingerprint: str
    validated_command_fingerprint: str
    submitted_command_fingerprint: str
    exact_command_match: bool
    downstream_submission_performed: bool
    schema: str = NEW_POLICY_SAFETY_BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NEW_POLICY_SAFETY_BRIDGE_SCHEMA:
            raise ValueError("unsupported pre-controller bridge evidence schema")
        if not self.exact_command_match or not self.downstream_submission_performed:
            raise ValueError("submission evidence requires an exact actual downstream submission")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "snapshot_id": self.snapshot_id,
            "control_epoch": self.control_epoch,
            "receipt_fingerprint": self.receipt_fingerprint,
            "validated_command_fingerprint": self.validated_command_fingerprint,
            "submitted_command_fingerprint": self.submitted_command_fingerprint,
            "exact_command_match": self.exact_command_match,
            "downstream_submission_performed": self.downstream_submission_performed,
        }


@dataclass(frozen=True)
class PreControllerSubmissionFailureEvidence:
    """Observable failed consumption of a previously accepted receipt."""

    snapshot_id: str | None
    control_epoch: int | None
    receipt_fingerprint: str | None
    failure_stage: str
    downstream_submission_attempted: bool
    schema: str = NEW_POLICY_SAFETY_BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NEW_POLICY_SAFETY_BRIDGE_SCHEMA:
            raise ValueError("unsupported pre-controller bridge failure schema")
        if not self.failure_stage:
            raise ValueError("submission failure needs an explicit stage")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "snapshot_id": self.snapshot_id,
            "control_epoch": self.control_epoch,
            "receipt_fingerprint": self.receipt_fingerprint,
            "failure_stage": self.failure_stage,
            "downstream_submission_attempted": self.downstream_submission_attempted,
        }


class NewPolicySafetyPreControllerBridge:
    """Bind a new-authority receipt to an existing EE-controller surface."""

    # Consumed by ``PolicyCommandRouter`` to prevent a receipt projection from
    # being paired with a different, raw controller port.
    requires_atomic_accepted_submission = True

    def __init__(
        self,
        *,
        authority: NewPolicySafetyAuthority,
        snapshot_provider: ReadOnlyProductionIKSnapshotProvider,
        downstream_controller: ExistingEEController,
    ) -> None:
        if not isinstance(authority, NewPolicySafetyAuthority):
            raise ValueError("authority must be NewPolicySafetyAuthority")
        if not isinstance(snapshot_provider, ReadOnlyProductionIKSnapshotProvider):
            raise ValueError("snapshot_provider must implement read-only capture")
        if not isinstance(downstream_controller, ExistingEEController):
            raise ValueError("downstream_controller must implement ExistingEEController")
        self._authority = authority
        self._snapshot_provider = snapshot_provider
        self._downstream_controller = downstream_controller
        self._last_receipt: FeasibilityReceipt | None = None
        self._last_submission_evidence: PreControllerSubmissionEvidence | None = None
        self._last_submission_failure: PreControllerSubmissionFailureEvidence | None = None
        self._submission_failure_history: list[PreControllerSubmissionFailureEvidence] = []
        self._pending: tuple[str, str, int, str, str] | None = None

    @property
    def last_receipt(self) -> FeasibilityReceipt | None:
        return self._last_receipt

    @property
    def last_submission_evidence(self) -> PreControllerSubmissionEvidence | None:
        return self._last_submission_evidence

    @property
    def last_submission_failure(self) -> PreControllerSubmissionFailureEvidence | None:
        return self._last_submission_failure

    @property
    def submission_failure_history(self) -> tuple[PreControllerSubmissionFailureEvidence, ...]:
        return tuple(self._submission_failure_history)

    def _clear_pending(self) -> None:
        self._pending = None

    def _record_submission_failure(
        self,
        *,
        pending: tuple[str, str, int, str, str] | None,
        stage: str,
        downstream_submission_attempted: bool,
    ) -> None:
        self._last_submission_evidence = None
        self._last_submission_failure = PreControllerSubmissionFailureEvidence(
            snapshot_id=None if pending is None else pending[3],
            control_epoch=None if pending is None else pending[2],
            receipt_fingerprint=None if pending is None else pending[1],
            failure_stage=stage,
            downstream_submission_attempted=downstream_submission_attempted,
        )
        self._submission_failure_history.append(self._last_submission_failure)

    def project(self, command: EEResidualCommand) -> SafetyProjection:
        """Evaluate only; this method never calls the downstream controller."""

        self._clear_pending()
        self._last_submission_evidence = None
        self._last_submission_failure = None
        self._submission_failure_history.clear()
        try:
            snapshot = self._snapshot_provider.capture_snapshot()
        except Exception as error:
            self._last_receipt = self._authority.reject_without_snapshot(
                reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                error_detail=f"READ_ONLY_SNAPSHOT_CAPTURE_FAILURE:{type(error).__name__}",
            )
            return SafetyProjection(False, None, "P0_NEW_AUTHORITY_SNAPSHOT_CAPTURE_FAILURE")
        receipt = self._authority.evaluate(snapshot, command)
        self._last_receipt = receipt
        if not receipt.accepted:
            return SafetyProjection(False, None, receipt.reason.value)
        if receipt.requested_target is None or receipt.validated_target is None:
            return SafetyProjection(False, None, "P0_NEW_AUTHORITY_ACCEPT_RECEIPT_INCOMPLETE")
        command_fingerprint = _command_fingerprint(command)
        if receipt.requested_target.source_residual != command or receipt.validated_target.source_residual != command:
            return SafetyProjection(False, None, "P0_NEW_AUTHORITY_TARGET_BINDING_MISMATCH")
        self._pending = (
            command_fingerprint,
            receipt.fingerprint(),
            snapshot.control_epoch,
            snapshot.snapshot_id,
            snapshot.fingerprint(),
        )
        return SafetyProjection(True, command)

    def submit_ee_residual(self, command: EEResidualCommand) -> None:
        """Submit exactly a pending accepted command after epoch freshness check."""

        pending = self._pending
        receipt = self._last_receipt
        submitted_fingerprint = _command_fingerprint(command)
        if (
            pending is None
            or receipt is None
            or not receipt.accepted
            or pending[0] != submitted_fingerprint
            or pending[1] != receipt.fingerprint()
            or receipt.requested_target is None
            or receipt.validated_target is None
            or receipt.requested_target.source_residual != command
            or receipt.validated_target.source_residual != command
        ):
            self._clear_pending()
            self._record_submission_failure(
                pending=pending,
                stage="RECEIPT_BINDING_MISMATCH",
                downstream_submission_attempted=False,
            )
            raise PolicyActionContractError("P0_NEW_AUTHORITY_RECEIPT_BINDING_MISMATCH")
        try:
            latest = self._snapshot_provider.capture_snapshot()
        except Exception as error:
            self._clear_pending()
            self._record_submission_failure(
                pending=pending,
                stage="SNAPSHOT_RECAPTURE_FAILURE",
                downstream_submission_attempted=False,
            )
            raise PolicyActionContractError("P0_NEW_AUTHORITY_SNAPSHOT_CAPTURE_FAILURE") from error
        if (
            latest.control_epoch != pending[2]
            or latest.snapshot_id != pending[3]
            or latest.fingerprint() != pending[4]
        ):
            self._clear_pending()
            self._record_submission_failure(
                pending=pending,
                stage="STALE_SNAPSHOT_REJECT",
                downstream_submission_attempted=False,
            )
            raise PolicyActionContractError("P0_NEW_AUTHORITY_STALE_SNAPSHOT_REJECT")
        # This is the only existing-controller call in this bridge.  It has no
        # access to a joint target and receives the original residual exactly.
        # Consume first: an exception after a partial downstream submission
        # must never leave a reusable accepted receipt behind for retry.
        self._clear_pending()
        try:
            self._downstream_controller.submit_ee_residual(command)
        except Exception:
            self._record_submission_failure(
                pending=pending,
                stage="DOWNSTREAM_SUBMISSION_FAILURE",
                downstream_submission_attempted=True,
            )
            raise
        self._last_submission_evidence = PreControllerSubmissionEvidence(
            snapshot_id=pending[3],
            control_epoch=pending[2],
            receipt_fingerprint=pending[1],
            validated_command_fingerprint=pending[0],
            submitted_command_fingerprint=submitted_fingerprint,
            exact_command_match=True,
            downstream_submission_performed=True,
        )
        self._last_submission_failure = None


__all__ = [
    "NEW_POLICY_SAFETY_BRIDGE_SCHEMA",
    "NewPolicySafetyPreControllerBridge",
    "PreControllerSubmissionEvidence",
    "PreControllerSubmissionFailureEvidence",
    "ReadOnlyProductionIKSnapshotProvider",
]
