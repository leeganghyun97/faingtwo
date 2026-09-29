# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure BC + residual-SAC composition contract for the G2 grasp branch.

The module has no Isaac or training side effects.  It fixes the only learned
SAC authority to a three-dimensional, robot-root, metric residual.  The BC
policy remains the sole owner of the gripper channel and the existing
production DLS/synchronised limiter remains the sole endpoint admission
authority.

There is deliberately no clipping path here.  Invalid raw residuals or final
actions are rejected, while a valid candidate is emitted only after an
existing :class:`PlannerEndpointAdmissionReceipt` approves the new endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from geniesim.rl.isaaclab.g2_policy_branch.contact_free_training_contract import (
    CONTACT_FREE_MAX_DELTA_M,
    CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M,
    metric_xyz_to_normalized,
)
from geniesim.rl.isaaclab.g2_policy_branch.precontact_limiter_contract import (
    PlannerEndpointAdmission,
    PlannerEndpointAdmissionReceipt,
)

from .keyboard_grasp_contract import (
    KEYBOARD_GRASP_PIPELINE_SCHEMA,
    KeyboardGraspPipelineContract,
    canonical_keyboard_grasp_contract,
)


BC_RESIDUAL_SAC_SCHEMA = "g2_bc_residual_sac_interface_v1"
BC_RESIDUAL_ALPHA_SCHEMA = "g2_bc_residual_alpha_curriculum_v1"
BC_RESIDUAL_CHECKPOINT_SCHEMA = "g2_bc_residual_sac_checkpoint_contract_v1"
BC_RESIDUAL_LIMITER_BINDING_SCHEMA = "g2_bc_residual_limiter_binding_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BCResidualSACContractError(ValueError):
    """Raised when a residual action crosses the frozen authority boundary."""


class ResidualAlphaStage(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResidualAlphaCurriculum:
    """Explicit, checkpointed residual authority; never a time schedule."""

    stage: ResidualAlphaStage = ResidualAlphaStage.SMALL
    promotion_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ResidualAlphaStage):
            raise BCResidualSACContractError("residual alpha stage must be explicit")
        if self.stage is ResidualAlphaStage.SMALL:
            if self.promotion_receipt_sha256 is not None:
                raise BCResidualSACContractError(
                    "SMALL stage cannot claim a promotion receipt"
                )
        elif not isinstance(self.promotion_receipt_sha256, str) or not _SHA256.fullmatch(
            self.promotion_receipt_sha256
        ):
            raise BCResidualSACContractError(
                "MEDIUM stage requires an explicit lowercase SHA-256 promotion receipt"
            )

    @property
    def alpha(self) -> float:
        contract = canonical_keyboard_grasp_contract()
        if self.stage is ResidualAlphaStage.SMALL:
            return contract.residual_alpha_small
        return contract.residual_alpha_medium

    @property
    def maximum_contribution_norm_m(self) -> float:
        contract = canonical_keyboard_grasp_contract()
        return self.alpha * contract.residual_raw_maximum_norm_m

    def promote(self, *, promotion_receipt_sha256: str) -> "ResidualAlphaCurriculum":
        """Return MEDIUM only after an explicit external promotion receipt."""

        if self.stage is not ResidualAlphaStage.SMALL:
            raise BCResidualSACContractError("residual alpha can be promoted only once")
        return ResidualAlphaCurriculum(
            stage=ResidualAlphaStage.MEDIUM,
            promotion_receipt_sha256=promotion_receipt_sha256,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": BC_RESIDUAL_ALPHA_SCHEMA,
            "stage": self.stage.value,
            "alpha_dimensionless": self.alpha,
            "promotion_receipt_sha256": self.promotion_receipt_sha256,
            "automatic_schedule": False,
            "raw_residual_maximum_norm_m": (
                canonical_keyboard_grasp_contract().residual_raw_maximum_norm_m
            ),
            "maximum_contribution_norm_m": self.maximum_contribution_norm_m,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualAlphaCurriculum":
        if not isinstance(payload, Mapping) or payload.get("schema") != BC_RESIDUAL_ALPHA_SCHEMA:
            raise BCResidualSACContractError("residual alpha checkpoint schema mismatch")
        try:
            stage = ResidualAlphaStage(str(payload.get("stage")))
        except ValueError as error:
            raise BCResidualSACContractError("unknown residual alpha checkpoint stage") from error
        result = cls(
            stage=stage,
            promotion_receipt_sha256=payload.get("promotion_receipt_sha256"),
        )
        if dict(payload) != result.payload():
            raise BCResidualSACContractError("residual alpha checkpoint payload mismatch")
        return result


@dataclass(frozen=True)
class BCResidualSACStaticConfig:
    """Frozen SAC-side contract stored with every future learner checkpoint."""

    alpha: ResidualAlphaCurriculum = ResidualAlphaCurriculum()
    num_envs: int = 25
    control_hz: int = 50
    policy_dt_s: float = 0.02
    physics_dt_s: float = 0.002
    frame: str = "robot_root"
    position_unit: str = "m"
    bc_action_dim: int = 4
    sac_action_dim: int = 3
    geometric_her_enabled: bool = False
    her_force_enabled: bool = True

    def __post_init__(self) -> None:
        common = canonical_keyboard_grasp_contract()
        expected = {
            "num_envs": common.future_sac_num_envs,
            "control_hz": common.control_hz,
            "policy_dt_s": common.policy_dt_s,
            "physics_dt_s": common.physics_dt_s,
            "frame": common.geometry_frame,
            "position_unit": common.position_unit,
            "bc_action_dim": common.bc_action_dim,
            "sac_action_dim": common.sac_action_dim,
            "geometric_her_enabled": common.geometric_her_enabled,
            "her_force_enabled": common.her_force_enabled,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise BCResidualSACContractError(f"frozen residual SAC mismatch: {name}")
        if not isinstance(self.alpha, ResidualAlphaCurriculum):
            raise BCResidualSACContractError("residual SAC alpha contract is missing")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": BC_RESIDUAL_CHECKPOINT_SCHEMA,
            "common_contract_schema": KEYBOARD_GRASP_PIPELINE_SCHEMA,
            "common_contract_sha256": _canonical_sha256(
                canonical_keyboard_grasp_contract().as_dict()
            ),
            **{
                name: value
                for name, value in asdict(self).items()
                if name != "alpha"
            },
            "alpha": self.alpha.payload(),
            "composition": "final_xyz=bc_xyz+alpha*delta_sac_metric",
            "final_gripper": "g_bc",
            "orientation_authority": False,
            "elbow_authority": False,
            "finger_or_torque_authority": False,
            "silent_clipping": False,
            "limiter_authority": "EXISTING_PRODUCTION_DLS_SYNCHRONIZED_LIMITER",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BCResidualSACStaticConfig":
        if not isinstance(payload, Mapping) or payload.get("schema") != BC_RESIDUAL_CHECKPOINT_SCHEMA:
            raise BCResidualSACContractError("residual SAC checkpoint contract schema mismatch")
        alpha = ResidualAlphaCurriculum.from_payload(payload.get("alpha", {}))
        result = cls(
            alpha=alpha,
            num_envs=payload.get("num_envs"),
            control_hz=payload.get("control_hz"),
            policy_dt_s=payload.get("policy_dt_s"),
            physics_dt_s=payload.get("physics_dt_s"),
            frame=payload.get("frame"),
            position_unit=payload.get("position_unit"),
            bc_action_dim=payload.get("bc_action_dim"),
            sac_action_dim=payload.get("sac_action_dim"),
            geometric_her_enabled=payload.get("geometric_her_enabled"),
            her_force_enabled=payload.get("her_force_enabled"),
        )
        if dict(payload) != result.payload():
            raise BCResidualSACContractError("residual SAC checkpoint contract changed")
        return result


def _finite_vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not bool(np.isfinite(result).all()):
        raise BCResidualSACContractError(f"{name} must be finite shape [{size}]")
    return result


@dataclass(frozen=True)
class ResidualActionComposition:
    bc_action_metric_root_m: tuple[float, float, float, float]
    raw_sac_residual_metric_root_m: tuple[float, float, float]
    alpha_stage: str
    alpha_dimensionless: float
    scaled_residual_contribution_m: tuple[float, float, float]
    final_xyz_metric_root_m: tuple[float, float, float]
    final_gripper_probability: float
    normalized_xyz_for_existing_port: tuple[float, float, float]
    candidate_sha256: str
    frame: str = "robot_root"
    unit: str = "m"
    metric_to_normalized_scale_count: int = 1
    silent_clipping_applied: bool = False
    limiter_admission_required: bool = True

    @property
    def final_action_4d_metric_root_m(self) -> tuple[float, float, float, float]:
        return (*self.final_xyz_metric_root_m, self.final_gripper_probability)

    def payload(self) -> dict[str, Any]:
        return asdict(self) | {
            "schema": BC_RESIDUAL_SAC_SCHEMA,
            "final_action_4d_metric_root_m": list(self.final_action_4d_metric_root_m),
            "gripper_authority": "BC_ONLY",
            "sac_authority": "XYZ_RESIDUAL_ONLY",
            "orientation_elbow_finger_torque_authority": False,
        }

    def metrics(self) -> dict[str, float]:
        """Return scale-explicit logging values for future SAC runs."""

        return {
            "residual_sac/raw_residual_norm_mm": 1000.0
            * float(np.linalg.norm(self.raw_sac_residual_metric_root_m)),
            "residual_sac/scaled_contribution_norm_mm": 1000.0
            * float(np.linalg.norm(self.scaled_residual_contribution_m)),
            "residual_sac/bc_xyz_norm_mm": 1000.0
            * float(np.linalg.norm(self.bc_action_metric_root_m[:3])),
            "residual_sac/final_xyz_norm_mm": 1000.0
            * float(np.linalg.norm(self.final_xyz_metric_root_m)),
            "residual_sac/alpha": self.alpha_dimensionless,
        }


def compose_bc_and_residual_action(
    *,
    bc_action_metric_root_m: Sequence[float],
    raw_sac_residual_metric_root_m: Sequence[float],
    alpha: ResidualAlphaCurriculum,
) -> ResidualActionComposition:
    """Compose one candidate action without clipping or controller mutation."""

    common = canonical_keyboard_grasp_contract()
    if not isinstance(alpha, ResidualAlphaCurriculum):
        raise BCResidualSACContractError("explicit alpha curriculum is required")
    bc = _finite_vector("bc_action_metric_root_m", bc_action_metric_root_m, 4)
    residual = _finite_vector(
        "raw_sac_residual_metric_root_m", raw_sac_residual_metric_root_m, 3
    )
    if not 0.0 <= float(bc[3]) <= 1.0:
        raise BCResidualSACContractError("BC gripper probability must be within [0,1]")
    if float(np.linalg.norm(bc[:3])) > common.maximum_final_xyz_norm_m + 1.0e-12:
        raise BCResidualSACContractError("BC XYZ exceeds the frozen 4.5-mm norm")
    raw_norm = float(np.linalg.norm(residual))
    if raw_norm > common.residual_raw_maximum_norm_m + 1.0e-12:
        raise BCResidualSACContractError("raw SAC residual exceeds the 4.5-mm norm")
    contribution = alpha.alpha * residual
    contribution_norm = float(np.linalg.norm(contribution))
    if contribution_norm > alpha.maximum_contribution_norm_m + 1.0e-12:
        raise BCResidualSACContractError("scaled SAC residual exceeds its alpha-stage bound")
    final_xyz = bc[:3] + contribution
    if float(np.linalg.norm(final_xyz)) > common.maximum_final_xyz_norm_m + 1.0e-12:
        raise BCResidualSACContractError(
            "composed XYZ exceeds the 4.5-mm bound; candidate rejected without clipping"
        )
    normalized = metric_xyz_to_normalized(final_xyz)
    payload = {
        "bc_action_metric_root_m": bc.tolist(),
        "raw_sac_residual_metric_root_m": residual.tolist(),
        "alpha_stage": alpha.stage.value,
        "alpha_dimensionless": alpha.alpha,
        "scaled_residual_contribution_m": contribution.tolist(),
        "final_xyz_metric_root_m": final_xyz.tolist(),
        "final_gripper_probability": float(bc[3]),
        "frame": common.geometry_frame,
        "unit": common.position_unit,
        "normalized_xyz_for_existing_port": normalized.tolist(),
        "normalization_scale_m": CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M,
        "metric_to_normalized_scale_count": 1,
    }
    return ResidualActionComposition(
        bc_action_metric_root_m=tuple(float(value) for value in bc),
        raw_sac_residual_metric_root_m=tuple(float(value) for value in residual),
        alpha_stage=alpha.stage.value,
        alpha_dimensionless=alpha.alpha,
        scaled_residual_contribution_m=tuple(float(value) for value in contribution),
        final_xyz_metric_root_m=tuple(float(value) for value in final_xyz),
        final_gripper_probability=float(bc[3]),
        normalized_xyz_for_existing_port=tuple(float(value) for value in normalized),
        candidate_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True)
class ResidualLimiterBindingResult:
    accepted: bool
    candidate_sha256: str
    limiter_decision: str
    emitted_action_4d_metric_root_m: tuple[float, float, float, float] | None
    rejection_reason: str | None
    schema: str = BC_RESIDUAL_LIMITER_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.accepted != (self.emitted_action_4d_metric_root_m is not None):
            raise BCResidualSACContractError("limiter result/action presence mismatch")
        if self.accepted and self.rejection_reason is not None:
            raise BCResidualSACContractError("accepted limiter result cannot carry rejection")
        if not self.accepted and not self.rejection_reason:
            raise BCResidualSACContractError("rejected limiter result requires a reason")


def bind_existing_limiter_receipt(
    composition: ResidualActionComposition,
    limiter_receipt: PlannerEndpointAdmissionReceipt,
) -> ResidualLimiterBindingResult:
    """Expose the exact candidate only when the existing limiter admits it."""

    if not isinstance(composition, ResidualActionComposition):
        raise BCResidualSACContractError("residual composition receipt is required")
    if not isinstance(limiter_receipt, PlannerEndpointAdmissionReceipt):
        raise BCResidualSACContractError("existing limiter admission receipt is required")
    decision = limiter_receipt.decision
    if decision is PlannerEndpointAdmission.SUBMIT_NEW_CARTESIAN_ENDPOINT:
        return ResidualLimiterBindingResult(
            accepted=True,
            candidate_sha256=composition.candidate_sha256,
            limiter_decision=decision.value,
            emitted_action_4d_metric_root_m=composition.final_action_4d_metric_root_m,
            rejection_reason=None,
        )
    return ResidualLimiterBindingResult(
        accepted=False,
        candidate_sha256=composition.candidate_sha256,
        limiter_decision=decision.value,
        emitted_action_4d_metric_root_m=None,
        rejection_reason=(
            "EXISTING_ENDPOINT_HELD"
            if decision is PlannerEndpointAdmission.HOLD_EXISTING_CARTESIAN_ENDPOINT
            else "NO_FEASIBLE_ENDPOINT_OR_HOLD"
        ),
    )


def residual_interface_provenance() -> dict[str, Any]:
    common: KeyboardGraspPipelineContract = canonical_keyboard_grasp_contract()
    return {
        "schema": BC_RESIDUAL_SAC_SCHEMA,
        "common_contract_sha256": _canonical_sha256(common.as_dict()),
        "bc_action": "[dx,dy,dz,g]",
        "sac_action": "[delta_x,delta_y,delta_z]",
        "bc_action_dim": common.bc_action_dim,
        "sac_action_dim": common.sac_action_dim,
        "frame": common.geometry_frame,
        "position_unit": common.position_unit,
        "control_hz": common.control_hz,
        "policy_dt_s": common.policy_dt_s,
        "physics_dt_s": common.physics_dt_s,
        "future_sac_num_envs": common.future_sac_num_envs,
        "raw_residual_maximum_norm_m": common.residual_raw_maximum_norm_m,
        "alpha_small": common.residual_alpha_small,
        "alpha_medium": common.residual_alpha_medium,
        "small_contribution_maximum_m": common.residual_contribution_bound_m("small"),
        "medium_contribution_maximum_m": common.residual_contribution_bound_m("medium"),
        "maximum_final_xyz_norm_m": CONTACT_FREE_MAX_DELTA_M,
        "curobo_handoff_band_m": list(common.curobo_handoff_band_m),
        "curobo_default_handoff_m": common.curobo_default_handoff_m,
        "local_grasp_band_m": list(common.local_grasp_band_m),
        "local_grasp_distance_reference": common.local_grasp_distance_reference,
        "local_grasp_distance_is_actor_input": False,
        "metric_to_normalized_scale_m": CONTACT_FREE_PRODUCTION_PORT_TRANSLATION_SCALE_M,
        "metric_to_normalized_scale_count": 1,
        "final_gripper": "g_bc",
        "geometric_her_enabled": False,
        "her_force_enabled": True,
        "automatic_alpha_schedule": False,
        "silent_clipping": False,
        "existing_limiter_receipt_required": True,
    }


__all__ = [
    "BC_RESIDUAL_ALPHA_SCHEMA",
    "BC_RESIDUAL_CHECKPOINT_SCHEMA",
    "BC_RESIDUAL_LIMITER_BINDING_SCHEMA",
    "BC_RESIDUAL_SAC_SCHEMA",
    "BCResidualSACContractError",
    "BCResidualSACStaticConfig",
    "ResidualActionComposition",
    "ResidualAlphaCurriculum",
    "ResidualAlphaStage",
    "ResidualLimiterBindingResult",
    "bind_existing_limiter_receipt",
    "compose_bc_and_residual_action",
    "residual_interface_provenance",
]
