# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""One runtime path for cuRobo-BC, grasp GRU, and XYZ-only residual SAC.

The module is simulator independent and never consumes an action itself.  It
selects exactly one phase owner, reuses the existing model/composition/limiter
implementations, and stages one immutable packet on the existing production
router.  The already-existing canonical env-step consumer remains the sole
owner of ``env.step -> ActionManager.process_action -> controller``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Any

import torch

from geniesim.rl.sac.bc_residual_sac_contract import (
    ResidualActionComposition,
    ResidualAlphaCurriculum,
    ResidualLimiterBindingResult,
    bind_existing_limiter_receipt,
    compose_bc_and_residual_action,
)
from geniesim.rl.sac.human_grasp_gru_bc import (
    HysteresisCalibration,
    HumanGraspGRUBC,
    HumanGraspGRUOutput,
    HumanGraspSequenceInputs,
    ROBOT_STATE_DIM,
    _validate_inputs,
    load_human_grasp_checkpoint,
)
from geniesim.rl.sac.keyboard_grasp_contract import canonical_keyboard_grasp_contract
from geniesim.rl.sac.residual_sac_runtime import (
    ResidualSACRuntime,
    load_residual_sac_actor_checkpoint,
)

from .action_interface import HighLevelPolicyAction
from .contact_free_visual_bc_runtime import (
    ContactFreeVisualBC,
    load_contact_free_visual_bc_checkpoint,
    validate_metric_output,
)
from .p1_final_action_path import P1FinalLimiterGate, P1StudentObservation
from .production_metric_adapter import (
    CanonicalManagerBasedRLEnvDeferredPacketConsumer,
    DeferredFull8DActionPacket,
    DeferredPacketConsumptionReceipt,
    ProductionBoundPolicyCommandRouter,
    ProductionDeferredRouteResult,
)


HYBRID_GRASP_RUNTIME_SCHEMA = "g2_curobo_gru_residual_sac_runtime_v1"


class HybridGraspRuntimeError(ValueError):
    """Fail-closed phase, representation, or authority violation."""


class HybridGraspPhase(str, Enum):
    FAR_REACH = "FAR_REACH"
    LOCAL_GRASP = "LOCAL_GRASP"
    # Retained only for artifact/schema compatibility.  The canonical router
    # keeps the 15--22 mm decision window inside LOCAL_GRASP.
    GRASP_DECISION = "GRASP_DECISION"


@dataclass(frozen=True)
class HybridPhaseDecision:
    phase: HybridGraspPhase
    nominal_grasp_residual_m: float
    handoff_latched: bool
    human_grasp_nominal_active: bool
    residual_sac_active: bool


class HybridGraspPhaseRouter:
    """30-mm phase handoff, 22-mm GRU ownership, and 15--22-mm residual gate."""

    def __init__(self) -> None:
        self._handoff_latched = False

    @property
    def handoff_latched(self) -> bool:
        return self._handoff_latched

    def reset(self) -> None:
        self._handoff_latched = False

    def decide(self, nominal_grasp_residual_m: float) -> HybridPhaseDecision:
        distance = float(nominal_grasp_residual_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise HybridGraspRuntimeError("nominal grasp residual must be finite and nonnegative")
        contract = canonical_keyboard_grasp_contract()
        lower, upper = contract.local_grasp_band_m
        handoff = contract.curobo_default_handoff_m
        if not self._handoff_latched and distance > handoff:
            return HybridPhaseDecision(
                phase=HybridGraspPhase.FAR_REACH,
                nominal_grasp_residual_m=distance,
                handoff_latched=False,
                human_grasp_nominal_active=False,
                residual_sac_active=False,
            )
        self._handoff_latched = True
        if distance > handoff:
            raise HybridGraspRuntimeError("POST_HANDOFF_RETREAT_ABOVE_30MM")
        human_grasp_nominal_active = distance <= upper
        residual_sac_active = lower <= distance <= upper
        return HybridPhaseDecision(
            phase=HybridGraspPhase.LOCAL_GRASP,
            nominal_grasp_residual_m=distance,
            handoff_latched=True,
            human_grasp_nominal_active=human_grasp_nominal_active,
            residual_sac_active=residual_sac_active,
        )


@dataclass(frozen=True)
class HybridGraspRuntimeResult:
    phase: HybridGraspPhase
    nominal_grasp_residual_m: float
    source_policy: str
    residual_sac_active: bool
    bc_action_4d_metric_root_m: tuple[float, float, float, float]
    raw_residual_sac_xyz_metric_root_m: tuple[float, float, float]
    composition: ResidualActionComposition
    limiter_binding: ResidualLimiterBindingResult
    final_action_4d_metric_root_m: tuple[float, float, float, float]
    downstream_route: ProductionDeferredRouteResult
    deferred_packet: DeferredFull8DActionPacket
    gru_hidden: torch.Tensor | None
    schema: str = HYBRID_GRASP_RUNTIME_SCHEMA


@dataclass(frozen=True)
class HybridGraspInferenceResult:
    """Canonical phase/model/composition result before the production gate.

    This is the live-runner boundary: it contains no packet and cannot consume
    an action.  The existing production packet builder and sole ``env.step``
    consumer remain downstream authorities.
    """

    decision: HybridPhaseDecision
    source_policy: str
    bc_action_4d_metric_root_m: tuple[float, float, float, float]
    close_probability: float
    feasibility_probability: float
    actor_forward_called: bool
    actor_raw_normalized_xyz: tuple[float, float, float]
    raw_residual_sac_xyz_metric_root_m: tuple[float, float, float]
    post_phase_gate_residual_metric_root_m: tuple[float, float, float]
    composition: ResidualActionComposition
    gru_hidden: torch.Tensor | None
    schema: str = HYBRID_GRASP_RUNTIME_SCHEMA


class UnifiedHybridGraspRuntime:
    """The sole high-level policy router before canonical ``env.step``."""

    def __init__(
        self,
        *,
        far_reach_bc: ContactFreeVisualBC,
        human_grasp_gru: HumanGraspGRUBC,
        residual_sac: ResidualSACRuntime,
        final_limiter_gate: P1FinalLimiterGate | None,
        production_router: ProductionBoundPolicyCommandRouter | None,
        alpha: ResidualAlphaCurriculum = ResidualAlphaCurriculum(),
        close_calibration: HysteresisCalibration | None = None,
    ) -> None:
        if not isinstance(far_reach_bc, ContactFreeVisualBC):
            raise HybridGraspRuntimeError("existing ContactFreeVisualBC is required")
        if not isinstance(human_grasp_gru, HumanGraspGRUBC):
            raise HybridGraspRuntimeError("existing HumanGraspGRUBC is required")
        if not isinstance(residual_sac, ResidualSACRuntime):
            raise HybridGraspRuntimeError("strict ResidualSACRuntime is required")
        if final_limiter_gate is not None and not isinstance(final_limiter_gate, P1FinalLimiterGate):
            raise HybridGraspRuntimeError("existing synchronized limiter gate is required")
        if production_router is not None and not isinstance(production_router, ProductionBoundPolicyCommandRouter):
            raise HybridGraspRuntimeError("existing production command router is required")
        if residual_sac.observation_dim != human_grasp_gru.config.gru_hidden_dim:
            raise HybridGraspRuntimeError("residual SAC input must be the grasp-GRU feature width")
        if far_reach_bc.training or human_grasp_gru.training or residual_sac.actor.training:
            raise HybridGraspRuntimeError("all runtime policies must be in eval mode")
        for model in (far_reach_bc, human_grasp_gru, residual_sac.actor):
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise HybridGraspRuntimeError("runtime policies must be frozen")
        self._far_reach_bc = far_reach_bc
        self._human_grasp_gru = human_grasp_gru
        self._residual_sac = residual_sac
        self._limiter = final_limiter_gate
        self._production_router = production_router
        self._alpha = alpha
        self._phase = HybridGraspPhaseRouter()
        self._gru_hidden: torch.Tensor | None = None
        self._closed = False
        self._close_calibration = close_calibration

    @classmethod
    def from_checkpoints(
        cls,
        *,
        far_reach_checkpoint: str | Path,
        far_reach_checkpoint_sha256: str,
        human_grasp_checkpoint: str | Path,
        residual_sac_checkpoint: str | Path,
        residual_sac_checkpoint_sha256: str,
        final_limiter_gate: P1FinalLimiterGate,
        production_router: ProductionBoundPolicyCommandRouter,
        alpha: ResidualAlphaCurriculum = ResidualAlphaCurriculum(),
        device: str | torch.device = "cpu",
    ) -> "UnifiedHybridGraspRuntime":
        """Load all three existing policies into the one canonical router."""

        far, _far_receipt = load_contact_free_visual_bc_checkpoint(
            far_reach_checkpoint,
            expected_sha256=far_reach_checkpoint_sha256,
            device=device,
        )
        grasp, grasp_receipt = load_human_grasp_checkpoint(
            human_grasp_checkpoint, device=device
        )
        grasp.eval()
        for parameter in grasp.parameters():
            parameter.requires_grad_(False)
        residual = load_residual_sac_actor_checkpoint(
            residual_sac_checkpoint,
            expected_sha256=residual_sac_checkpoint_sha256,
            expected_human_grasp_checkpoint_sha256=str(grasp_receipt["file_sha256"]),
            expected_observation_dim=grasp.config.gru_hidden_dim,
            device=device,
        )
        return cls(
            far_reach_bc=far,
            human_grasp_gru=grasp,
            residual_sac=residual,
            final_limiter_gate=final_limiter_gate,
            production_router=production_router,
            alpha=alpha,
            close_calibration=grasp_receipt["calibration"],
        )

    @property
    def phase_router(self) -> HybridGraspPhaseRouter:
        return self._phase

    def reset(self) -> None:
        self._phase.reset()
        self._gru_hidden = None
        self._closed = False
        if self._production_router is not None:
            self._production_router.reset()

    def _far_action(self, observation: P1StudentObservation) -> tuple[float, float, float, float]:
        image, proprio = observation.prepare()
        with torch.inference_mode():
            output = self._far_reach_bc(image, proprio)
        validate_metric_output(output)
        return tuple(float(value) for value in output[0].detach().cpu().tolist())

    def _grasp_action(
        self, inputs: HumanGraspSequenceInputs
    ) -> tuple[tuple[float, float, float, float], HumanGraspGRUOutput]:
        with torch.inference_mode():
            if self._gru_hidden is None or bool(inputs.hidden_reset_mask[:, 0].all()):
                output = self._human_grasp_gru(inputs, hidden=self._gru_hidden)
            else:
                # The immutable training model requires stored sequences to
                # carry a reset at t=0.  A live one-row continuation instead
                # carries the previous hidden state.  Validate an equivalent
                # reset-marked view, then execute the original mask and exact
                # frozen modules without changing the checkpoint source.
                validation = replace(
                    inputs,
                    hidden_reset_mask=torch.ones_like(inputs.hidden_reset_mask),
                )
                batch, time = _validate_inputs(
                    validation, self._human_grasp_gru.config
                )
                if (batch, time) != (1, 1):
                    raise HybridGraspRuntimeError(
                        "live grasp continuation requires [1,1]"
                    )
                model = self._human_grasp_gru
                visual = model.vision(
                    inputs.right_wrist_rgb,
                    inputs.right_wrist_depth_m,
                    inputs.right_wrist_depth_valid,
                )
                state = torch.cat(
                    (
                        inputs.ee_pose_robot_root_m_xyzw,
                        inputs.right_arm_joint_position_rad,
                        inputs.right_arm_joint_velocity_rad_s,
                        inputs.current_gripper_state,
                        inputs.previous_policy_action_4d_metric_root_m,
                    ),
                    dim=-1,
                )
                if state.shape[-1] != ROBOT_STATE_DIM:
                    raise HybridGraspRuntimeError("grasp state width mismatch")
                fused = model.fusion(
                    torch.cat((visual, model.state_encoder(state)), dim=-1)
                )
                features, final_hidden = model.gru(fused, self._gru_hidden)
                bound = canonical_keyboard_grasp_contract().maximum_final_xyz_norm_m
                xyz_normalized = torch.tanh(model.xyz_head(features))
                norm = torch.linalg.vector_norm(
                    xyz_normalized, dim=-1, keepdim=True
                )
                projection_applied = norm[..., 0] > 1.0
                xyz = (
                    xyz_normalized
                    * torch.clamp(1.0 / norm.clamp_min(1.0e-12), max=1.0)
                    * bound
                )
                close_logit = model.close_head(features)
                close_probability = torch.sigmoid(close_logit)
                feasibility_logit = model.feasibility_head(features)
                feasibility_probability = torch.sigmoid(feasibility_logit)
                output = HumanGraspGRUOutput(
                    xyz_action_robot_root_m=xyz,
                    close_logit=close_logit,
                    close_probability=close_probability,
                    feasibility_logit=feasibility_logit,
                    feasibility_probability=feasibility_probability,
                    action_4d=torch.cat((xyz, close_probability), dim=-1),
                    xyz_projection_applied=projection_applied,
                    recurrent_features=features,
                    final_hidden=final_hidden,
                )
        self._gru_hidden = output.final_hidden.detach()
        action = output.action_4d[:, -1]
        if tuple(action.shape) != (1, 4) or not bool(torch.isfinite(action).all()):
            raise HybridGraspRuntimeError("HumanGraspGRUBC output must be finite [1,4]")
        raw = tuple(float(value) for value in action[0].detach().cpu().tolist())
        probability = raw[3]
        if self._close_calibration is None:
            return raw, output
        if self._close_calibration.mode == "EDGE_SINGLE_EVENT":
            if not self._closed and probability >= self._close_calibration.close_threshold:
                self._closed = True
        else:
            if probability >= self._close_calibration.close_threshold:
                self._closed = True
            elif probability <= self._close_calibration.open_threshold:
                self._closed = False
        return (*raw[:3], 1.0 if self._closed else 0.0), output

    def infer(
        self,
        *,
        nominal_grasp_residual_m: float,
        far_observation: P1StudentObservation | None = None,
        grasp_inputs: HumanGraspSequenceInputs | None = None,
    ) -> HybridGraspInferenceResult:
        """Run the sole phase/model/composition path without routing a packet."""

        decision = self._phase.decide(nominal_grasp_residual_m)
        gru_output: HumanGraspGRUOutput | None = None
        close_probability = 0.0
        feasibility_probability = 0.0
        # The 30-mm handoff changes the canonical phase, but the qualified
        # cuRobo-BC remains the nominal approach owner until the existing
        # 22-mm upper edge of the grasp-decision band.  This prevents the
        # sub-millimetre local GRU output from stalling the 30->22-mm travel
        # and introduces no new distance threshold.
        if not decision.human_grasp_nominal_active:
            if far_observation is None:
                raise HybridGraspRuntimeError("approach phases require the cuRobo-BC observation")
            bc_action = self._far_action(far_observation)
            source_policy = "CUROBO_CONTACT_FREE_BC"
        else:
            if grasp_inputs is None:
                raise HybridGraspRuntimeError("local phases require HumanGraspGRUBC inputs")
            bc_action, gru_output = self._grasp_action(grasp_inputs)
            close_probability = float(gru_output.close_probability[0, -1, 0].item())
            feasibility_probability = float(
                gru_output.feasibility_probability[0, -1, 0].item()
            )
            source_policy = "HUMAN_GRASP_GRU_BC"

        actor_forward_called = False
        actor_normalized = (0.0, 0.0, 0.0)
        raw_residual = (0.0, 0.0, 0.0)
        if decision.residual_sac_active:
            assert gru_output is not None
            recurrent = gru_output.recurrent_features[:, -1]
            actor = self._residual_sac.infer(recurrent)
            actor_forward_called = True
            actor_normalized = actor.normalized_action
            raw_residual = actor.metric_residual_root_m

        composition = compose_bc_and_residual_action(
            bc_action_metric_root_m=bc_action,
            raw_sac_residual_metric_root_m=raw_residual,
            alpha=self._alpha,
        )
        return HybridGraspInferenceResult(
            decision=decision,
            source_policy=source_policy,
            bc_action_4d_metric_root_m=bc_action,
            close_probability=close_probability,
            feasibility_probability=feasibility_probability,
            actor_forward_called=actor_forward_called,
            actor_raw_normalized_xyz=actor_normalized,
            raw_residual_sac_xyz_metric_root_m=raw_residual,
            post_phase_gate_residual_metric_root_m=(
                composition.scaled_residual_contribution_m
            ),
            composition=composition,
            gru_hidden=(None if self._gru_hidden is None else self._gru_hidden.detach()),
        )

    def step(
        self,
        *,
        nominal_grasp_residual_m: float,
        far_observation: P1StudentObservation | None = None,
        grasp_inputs: HumanGraspSequenceInputs | None = None,
    ) -> HybridGraspRuntimeResult:
        """Stage exactly one policy packet; never call ``env.step`` here."""

        inference = self.infer(
            nominal_grasp_residual_m=nominal_grasp_residual_m,
            far_observation=far_observation,
            grasp_inputs=grasp_inputs,
        )
        if self._limiter is None or self._production_router is None:
            raise HybridGraspRuntimeError("packet routing authorities are not bound")
        decision = inference.decision
        bc_action = inference.bc_action_4d_metric_root_m
        composition = inference.composition
        limiter_receipt = self._limiter.admit(composition)
        limiter = bind_existing_limiter_receipt(composition, limiter_receipt)
        if not limiter.accepted:
            raise HybridGraspRuntimeError(
                f"FINAL_LIMITER_REJECT:{limiter.rejection_reason}"
            )
        final_action = composition.final_action_4d_metric_root_m
        if final_action[3] != bc_action[3]:
            raise HybridGraspRuntimeError("residual SAC changed BC gripper authority")
        normalized = (*composition.normalized_xyz_for_existing_port, final_action[3])
        route = self._production_router.route(HighLevelPolicyAction.from_sequence(normalized))
        if not route.accepted or route.deferred_packet is None or route.ee_command is None:
            raise HybridGraspRuntimeError(f"PRODUCTION_ROUTE_REJECT:{route.reason}")
        if any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for observed, expected in zip(
                route.ee_command.translation_m,
                composition.final_xyz_metric_root_m,
                strict=True,
            )
        ):
            raise HybridGraspRuntimeError("production route changed the composed XYZ action")
        return HybridGraspRuntimeResult(
            phase=decision.phase,
            nominal_grasp_residual_m=decision.nominal_grasp_residual_m,
            source_policy=inference.source_policy,
            residual_sac_active=decision.residual_sac_active,
            bc_action_4d_metric_root_m=bc_action,
            raw_residual_sac_xyz_metric_root_m=(
                inference.raw_residual_sac_xyz_metric_root_m
            ),
            composition=composition,
            limiter_binding=limiter,
            final_action_4d_metric_root_m=final_action,
            downstream_route=route,
            deferred_packet=route.deferred_packet,
            gru_hidden=(None if self._gru_hidden is None else self._gru_hidden.detach()),
        )

    def consume_with_canonical_env_step(
        self,
        result: HybridGraspRuntimeResult,
        *,
        consumer: CanonicalManagerBasedRLEnvDeferredPacketConsumer,
    ) -> DeferredPacketConsumptionReceipt:
        """Use the existing sole ``env.step`` owner; never call ActionManager directly."""

        if not isinstance(result, HybridGraspRuntimeResult):
            raise HybridGraspRuntimeError("typed hybrid runtime result is required")
        if not isinstance(consumer, CanonicalManagerBasedRLEnvDeferredPacketConsumer):
            raise HybridGraspRuntimeError("canonical env-step consumer is required")
        return consumer.consume(result.deferred_packet)


__all__ = [
    "HYBRID_GRASP_RUNTIME_SCHEMA",
    "HybridGraspPhase",
    "HybridGraspPhaseRouter",
    "HybridGraspRuntimeError",
    "HybridGraspInferenceResult",
    "HybridGraspRuntimeResult",
    "HybridPhaseDecision",
    "UnifiedHybridGraspRuntime",
]
