# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Checkpoint-compatible contact recovery state for the G2 reference path.

This module deliberately owns no reward, termination, force, velocity, or
acceleration threshold.  It converts already-audited runtime predicates into
vectorized controller-authority masks.  Consequently it can be added around
an existing actor/reference checkpoint without changing any learned tensor or
the action contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch


class G2ReferenceRecoveryPhase(IntEnum):
    """Phases of the deterministic reference close/recovery hand-off."""

    APPROACH_OPEN = 0
    PRECONTACT_CLOSE = 1
    CONTACT_SETTLE = 2
    STABLE_LIFT = 3
    RECOVERY_OPEN = 4


def exact_emitted_endpoint_mask(
    emitted_target: torch.Tensor, configured_endpoint: torch.Tensor
) -> torch.Tensor:
    """Return vector rows exactly equal to an existing actuator endpoint."""

    if emitted_target.ndim != 2:
        raise ValueError("emitted target must be a two-dimensional tensor")
    if configured_endpoint.shape != (emitted_target.shape[1],):
        raise ValueError("configured endpoint width does not match emitted target")
    if configured_endpoint.device != emitted_target.device:
        raise ValueError("configured endpoint and emitted target must share a device")
    if configured_endpoint.dtype != emitted_target.dtype:
        raise ValueError("configured endpoint and emitted target must share a dtype")
    return torch.eq(
        emitted_target,
        configured_endpoint.unsqueeze(0).expand_as(emitted_target),
    ).all(dim=-1)


@dataclass(frozen=True)
class G2ReferenceRecoveryOutput:
    """Authority masks for one vectorized control boundary."""

    phase: torch.Tensor
    arm_hold: torch.Tensor
    command_gripper_close: torch.Tensor
    command_gripper_open: torch.Tensor
    hold_gripper_target: torch.Tensor
    lift_authority: torch.Tensor
    recovery_started: torch.Tensor
    recovery_completed: torch.Tensor
    ungraspable_elapsed_s: torch.Tensor


@dataclass(frozen=True)
class G2TrainingHandoffMasks:
    """Pure masks for the opt-in deterministic-first training handoff."""

    trigger: torch.Tensor
    policy_origin_trigger: torch.Tensor
    reference_transfer: torch.Tensor
    latched: torch.Tensor
    controller_authority: torch.Tensor
    heldout_authority_violation: torch.Tensor


def deterministic_first_training_handoff_masks(
    *,
    enabled: bool,
    deterministic_first_phase: str,
    training_mask: torch.Tensor,
    heldout_mask: torch.Tensor,
    reference_behavior_mask: torch.Tensor,
    previously_latched: torch.Tensor,
    calibrated_close_gate: torch.Tensor,
) -> G2TrainingHandoffMasks:
    """Latch controller authority without defining a new task threshold.

    The caller supplies the existing calibrated close predicate.  Any
    training row in ``JOINT_TRAINING`` can trigger at that exact gate.  A row
    already owned by reference behavior is explicitly transferred to the
    assisted authority at the boundary; this avoids requiring an untrained
    policy to rediscover grasp-ready before deterministic-first assistance
    can ever activate.  Held-out rows remain ineligible.  A stale latch on
    held-out evaluation is returned as an explicit violation rather than
    silently cleared, allowing the runtime to fail closed.
    """

    masks = (
        training_mask,
        heldout_mask,
        reference_behavior_mask,
        previously_latched,
        calibrated_close_gate,
    )
    if any(value.dtype != torch.bool or value.ndim != 1 for value in masks):
        raise ValueError("training handoff inputs must be boolean [N] tensors")
    if any(value.shape != training_mask.shape for value in masks[1:]):
        raise ValueError("training handoff masks must have identical shape")
    if any(value.device != training_mask.device for value in masks[1:]):
        raise ValueError("training handoff masks must share one device")
    if bool((training_mask & heldout_mask).any()):
        raise ValueError("training and held-out masks must be disjoint")

    trigger = torch.zeros_like(training_mask)
    if enabled and deterministic_first_phase == "JOINT_TRAINING":
        trigger = (
            training_mask
            & ~previously_latched
            & calibrated_close_gate
        )
    policy_origin_trigger = trigger & ~reference_behavior_mask
    reference_transfer = trigger & reference_behavior_mask
    latched = previously_latched | trigger
    heldout_violation = latched & heldout_mask
    return G2TrainingHandoffMasks(
        trigger=trigger,
        policy_origin_trigger=policy_origin_trigger,
        reference_transfer=reference_transfer,
        latched=latched,
        controller_authority=reference_behavior_mask | latched,
        heldout_authority_violation=heldout_violation,
    )


class G2ReferenceRecoveryStateMachine:
    """Vectorized deterministic close, settle, lift, and bounded recovery.

    ``close_gate``, ``close_target_emitted``, and ``open_target_emitted`` are
    supplied by the existing calibrated controller/actuator contract.  This
    helper intentionally does not recreate their position thresholds.  In
    particular, the no-contact timer does not run while the bounded gripper is
    still travelling from its open target to its closed target.  The existing
    final-contact cap can make that motion take several seconds.

    ``ungraspable_persistence_s`` is likewise injected from
    :class:`G2LiftSandboxContract`, whose current value is 0.20 seconds.

    A close attempt that cannot form bilateral contact for the persistence
    interval after the exact close endpoint is emitted, or a stable grasp that
    remains lost for that interval, opens the gripper before re-entering the
    approach.  This includes persistent unilateral contact; it must not
    deadlock CONTACT_SETTLE forever.  Episode completion and rows not owned by
    reference behavior reset immediately, preventing controller latches from
    leaking into policy rows or the next episode.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        ungraspable_persistence_s: float,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("reference recovery requires a positive env count")
        if not math.isfinite(float(ungraspable_persistence_s)) or not (
            ungraspable_persistence_s > 0.0
        ):
            raise ValueError("ungraspable persistence must be finite and positive")
        self.num_envs = int(num_envs)
        self.ungraspable_persistence_s = float(ungraspable_persistence_s)
        self.phase = torch.full(
            (self.num_envs,),
            int(G2ReferenceRecoveryPhase.APPROACH_OPEN),
            dtype=torch.long,
            device=device,
        )
        self.ungraspable_elapsed_s = torch.zeros(
            # Time, rather than a derived step count, is the authority.  Keep
            # the accumulator in float64 so ten 20-ms intervals meet the
            # existing 0.20-s contract without a float32 boundary miss.
            self.num_envs, dtype=torch.float64, device=device
        )

    def reset(self, env_mask: torch.Tensor) -> None:
        """Reset selected rows without touching any environment/task state."""

        self._validate_bool("env_mask", env_mask)
        self.phase[env_mask] = int(G2ReferenceRecoveryPhase.APPROACH_OPEN)
        self.ungraspable_elapsed_s[env_mask] = 0.0

    def step(
        self,
        *,
        reference_mask: torch.Tensor,
        close_gate: torch.Tensor,
        close_target_emitted: torch.Tensor,
        open_target_emitted: torch.Tensor,
        any_finger_contact: torch.Tensor,
        bilateral_contact: torch.Tensor,
        stable_grasp: torch.Tensor,
        done: torch.Tensor,
        control_dt_s: float,
        freeze_mask: torch.Tensor | None = None,
    ) -> G2ReferenceRecoveryOutput:
        """Advance controller state and return masks for the current command.

        Every physical predicate must describe the most recently completed
        simulation step.  The caller remains responsible for applying its
        established speed/acceleration/collision limits after these masks.
        """

        for name, value in (
            ("reference_mask", reference_mask),
            ("close_gate", close_gate),
            ("close_target_emitted", close_target_emitted),
            ("open_target_emitted", open_target_emitted),
            ("any_finger_contact", any_finger_contact),
            ("bilateral_contact", bilateral_contact),
            ("stable_grasp", stable_grasp),
            ("done", done),
        ):
            self._validate_bool(name, value)
        if not math.isfinite(float(control_dt_s)) or not (control_dt_s > 0.0):
            raise ValueError("control dt must be finite and positive")
        if freeze_mask is None:
            freeze_mask = torch.zeros_like(reference_mask)
        self._validate_bool("freeze_mask", freeze_mask)
        if bool((freeze_mask & ~reference_mask).any()):
            raise ValueError("reference freeze rows must be reference-owned")
        if bool((bilateral_contact & ~any_finger_contact).any()):
            raise ValueError("bilateral contact requires finger-contact evidence")
        if bool((stable_grasp & ~bilateral_contact).any()):
            raise ValueError("stable grasp requires bilateral contact")

        active = reference_mask & ~done
        advancing = active & ~freeze_mask
        # A policy row or completed episode must never inherit reference
        # controller state from the preceding vectorized control boundary.
        self.reset(~active)

        recovery_started = torch.zeros_like(active)
        recovery_completed = torch.zeros_like(active)

        approach = advancing & (
            self.phase == int(G2ReferenceRecoveryPhase.APPROACH_OPEN)
        )
        enter_close = approach & close_gate
        self.phase[enter_close] = int(G2ReferenceRecoveryPhase.PRECONTACT_CLOSE)
        self.ungraspable_elapsed_s[enter_close] = 0.0

        precontact = advancing & (
            self.phase == int(G2ReferenceRecoveryPhase.PRECONTACT_CLOSE)
        )
        precontact_contact = precontact & any_finger_contact
        self.phase[precontact_contact] = int(
            G2ReferenceRecoveryPhase.CONTACT_SETTLE
        )
        self.ungraspable_elapsed_s[precontact_contact] = 0.0

        contact_settle = advancing & (
            self.phase == int(G2ReferenceRecoveryPhase.CONTACT_SETTLE)
        )
        become_stable = (precontact | contact_settle) & stable_grasp
        self.phase[become_stable] = int(G2ReferenceRecoveryPhase.STABLE_LIFT)
        self.ungraspable_elapsed_s[become_stable] = 0.0

        # Bilateral evidence clears the bounded failure timer.  Before stable,
        # the arm remains held while the existing stable timer accumulates.
        # Unilateral contact is not enough once the exact close endpoint has
        # been emitted.
        contact_settle = active & (
            self.phase == int(G2ReferenceRecoveryPhase.CONTACT_SETTLE)
        )
        self.ungraspable_elapsed_s[contact_settle & bilateral_contact] = 0.0

        stable_lift = advancing & (
            self.phase == int(G2ReferenceRecoveryPhase.STABLE_LIFT)
        )
        lost_stable = stable_lift & ~stable_grasp
        self.phase[lost_stable] = int(G2ReferenceRecoveryPhase.CONTACT_SETTLE)
        self.ungraspable_elapsed_s[lost_stable] = torch.where(
            any_finger_contact[lost_stable],
            torch.zeros_like(self.ungraspable_elapsed_s[lost_stable]),
            torch.full_like(
                self.ungraspable_elapsed_s[lost_stable], float(control_dt_s)
            ),
        )

        precontact = advancing & (
            self.phase == int(G2ReferenceRecoveryPhase.PRECONTACT_CLOSE)
        )
        contact_settle = advancing & (
            self.phase == int(G2ReferenceRecoveryPhase.CONTACT_SETTLE)
        )
        # A reference close may take seconds at the established 0.10-rad/s
        # final-contact cap.  Declaring it ungraspable 0.20 seconds after the
        # close command would therefore create an open/close oscillator.  A
        # never-contacted or unilateral close only starts the persistence
        # clock after the existing actuator contract reports its exact emitted
        # endpoint.  After a stable grasp is lost, CONTACT_SETTLE releases the
        # external hold; the close target can then finish before timing failed
        # reacquisition.
        bilateral_missing_at_endpoint = (
            (precontact | contact_settle)
            & close_target_emitted
            & ~bilateral_contact
        )
        travelling_without_bilateral = (
            (precontact | contact_settle)
            & ~close_target_emitted
            & ~bilateral_contact
        )
        self.ungraspable_elapsed_s[travelling_without_bilateral] = 0.0
        # ``lost_stable`` already received the first interval above.
        increment = bilateral_missing_at_endpoint & ~lost_stable
        self.ungraspable_elapsed_s[increment] += float(control_dt_s)
        timed_out = bilateral_missing_at_endpoint & (
            self.ungraspable_elapsed_s
            >= self.ungraspable_persistence_s - 1.0e-9
        )
        recovery_started |= timed_out
        self.phase[timed_out] = int(G2ReferenceRecoveryPhase.RECOVERY_OPEN)
        self.ungraspable_elapsed_s[timed_out] = 0.0

        recovery = active & (
            self.phase == int(G2ReferenceRecoveryPhase.RECOVERY_OPEN)
        )
        recovered = recovery & open_target_emitted
        recovery_completed |= recovered
        self.phase[recovered] = int(G2ReferenceRecoveryPhase.APPROACH_OPEN)
        self.ungraspable_elapsed_s[recovered] = 0.0

        # Re-apply completion/reset precedence after all state transitions.
        self.reset(done | ~reference_mask)

        phase = self.phase.clone()
        precontact = active & (
            phase == int(G2ReferenceRecoveryPhase.PRECONTACT_CLOSE)
        )
        contact_settle = active & (
            phase == int(G2ReferenceRecoveryPhase.CONTACT_SETTLE)
        )
        stable_lift = active & (
            phase == int(G2ReferenceRecoveryPhase.STABLE_LIFT)
        )
        recovery = active & (
            phase == int(G2ReferenceRecoveryPhase.RECOVERY_OPEN)
        )
        return G2ReferenceRecoveryOutput(
            phase=phase,
            arm_hold=precontact | contact_settle | recovery,
            command_gripper_close=precontact | contact_settle | stable_lift,
            command_gripper_open=(
                active
                & (
                    (phase == int(G2ReferenceRecoveryPhase.APPROACH_OPEN))
                    | recovery
                )
            ),
            hold_gripper_target=(
                stable_lift | (contact_settle & bilateral_contact)
            ),
            lift_authority=stable_lift,
            recovery_started=recovery_started,
            recovery_completed=recovery_completed,
            ungraspable_elapsed_s=self.ungraspable_elapsed_s.clone(),
        )

    def _validate_bool(self, name: str, value: torch.Tensor) -> None:
        if value.shape != (self.num_envs,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean [N] tensor")
        if value.device != self.phase.device:
            raise ValueError(f"{name} must be on the state-machine device")


__all__ = [
    "G2ReferenceRecoveryOutput",
    "G2ReferenceRecoveryPhase",
    "G2ReferenceRecoveryStateMachine",
    "G2TrainingHandoffMasks",
    "deterministic_first_training_handoff_masks",
    "exact_emitted_endpoint_mask",
]
