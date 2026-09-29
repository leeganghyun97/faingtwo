# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure pre-controller feasibility authority for the high-level G2 policy.

This module deliberately creates a *new* authority.  The selected production
controller has a DLS execution path, but it does not expose an ACCEPT/REJECT
result, a raw-joint feasibility check, or an FK residual check.  Nothing in
this file reinterprets the controller's soft clamp or rate limiter as such an
authority.

The implementation is intentionally independent from Isaac Sim, Torch,
articulations, USD, and the existing mutable controller.  A caller provides
an immutable read-only snapshot and a pure FK object.  The only result is an
immutable receipt; this module never submits a command.

``KinematicModelBinding.UNRESOLVED`` is a valid state.  It allows callers to
record why a production deployment is blocked without substituting a legacy
URDF/Pinocchio model for the selected production DLS path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .action_interface import CartesianControlFrame, EEResidualCommand


NEW_POLICY_SAFETY_AUTHORITY_SCHEMA = "g2_new_policy_safety_authority_v1"
NEW_POLICY_SAFETY_AUTHORITY_VERSION = "v1"
AUTHORITY_ORIGIN = "NEW_POLICY_SAFETY_AUTHORITY"
EXISTING_PRODUCTION_ACCEPTANCE_AUTHORITY = "ABSENT"
QUATERNION_CONVENTION = "xyzw"
PRODUCTION_DLS_LAMBDA = 0.01
PRODUCTION_NULLSPACE_DAMPING = 0.05
PRODUCTION_NULLSPACE_SEED_JOINT_INDEX = 2
PRODUCTION_NULLSPACE_MAX_DELTA_RAD = 0.0002
PRODUCTION_TARGET_QUEUE_EPSILON = 1.0e-8


class FeasibilityRejectReason(str, Enum):
    """Explicit outcomes emitted by the new, fail-closed authority."""

    ACCEPT = "ACCEPT"
    INVALID_INPUT = "INVALID_INPUT"
    IK_SOLVE_FAILURE = "IK_SOLVE_FAILURE"
    IK_NONFINITE = "IK_NONFINITE"
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
    FK_NONFINITE = "FK_NONFINITE"
    POSITION_ERROR_EXCEEDED = "POSITION_ERROR_EXCEEDED"
    ORIENTATION_ERROR_EXCEEDED = "ORIENTATION_ERROR_EXCEEDED"
    TARGET_TRANSPARENCY_VIOLATION = "TARGET_TRANSPARENCY_VIOLATION"
    INTERNAL_VALIDATION_FAILURE = "INTERNAL_VALIDATION_FAILURE"


class ThresholdSource(str, Enum):
    """Provenance classes mandated for a new acceptance threshold."""

    EXISTING_PROJECT_REQUIREMENT = "EXISTING_PROJECT_REQUIREMENT"
    MEASURED_CONTROL_TOLERANCE = "MEASURED_CONTROL_TOLERANCE"
    ENGINEERING_DESIGN_CHOICE = "ENGINEERING_DESIGN_CHOICE"
    UNRESOLVED = "UNRESOLVED"


class KinematicModelBinding(str, Enum):
    """Whether a pure FK model is established to match production mechanics."""

    PRODUCTION_EQUIVALENT = "PRODUCTION_EQUIVALENT"
    TEST_FIXTURE_ONLY = "TEST_FIXTURE_ONLY"
    UNRESOLVED = "UNRESOLVED"


class AuthorityEvaluationScope(str, Enum):
    """Explicitly separate a deployable query from a synthetic test fixture."""

    PRODUCTION_DEPLOYMENT = "PRODUCTION_DEPLOYMENT"
    STATIC_TEST_FIXTURE = "STATIC_TEST_FIXTURE"


def _is_finite_vector(values: Sequence[float], *, dimension: int) -> bool:
    try:
        return len(values) == dimension and all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError, OverflowError):
        return False


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_scalar(value: object | None) -> float | str | None:
    """Keep JSON reports strict when a negative case intentionally has NaN."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "NON_NUMERIC"
    if math.isnan(numeric):
        return "NaN"
    if math.isinf(numeric):
        return "Infinity" if numeric > 0.0 else "-Infinity"
    return numeric


def _safe_vector(values: Sequence[object] | None) -> list[float | str] | None:
    if values is None:
        return None
    return [_safe_scalar(value) for value in values]


@dataclass(frozen=True)
class EEPose:
    """A root-frame pose in Isaac Lab's explicit ``xyzw`` convention."""

    position_root_m: tuple[float, float, float]
    quaternion_root_xyzw: tuple[float, float, float, float]

    def finite(self) -> bool:
        return _is_finite_vector(self.position_root_m, dimension=3) and _is_finite_vector(
            self.quaternion_root_xyzw, dimension=4
        )

    def quaternion_nonzero(self) -> bool:
        if not self.finite():
            return False
        return float(np.linalg.norm(np.asarray(self.quaternion_root_xyzw, dtype=np.float64))) > 0.0

    def payload(self) -> dict[str, object]:
        return {
            "position_root_m": _safe_vector(self.position_root_m),
            "quaternion_root_xyzw": _safe_vector(self.quaternion_root_xyzw),
            "quaternion_convention": QUATERNION_CONVENTION,
        }


@dataclass(frozen=True)
class NewSafetyThreshold:
    """A threshold with provenance; no implicit numerical default is allowed."""

    name: str
    value: float | None
    unit: str
    reason: str
    source: ThresholdSource
    scope: str
    sensitivity: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.reason or not self.scope or not self.sensitivity:
            raise ValueError("threshold metadata must be non-empty")
        if not isinstance(self.source, ThresholdSource):
            raise ValueError("threshold source must be ThresholdSource")
        if self.value is not None and (
            not math.isfinite(float(self.value)) or float(self.value) < 0.0
        ):
            raise ValueError("threshold value must be finite and non-negative when set")
        if self.source is ThresholdSource.UNRESOLVED and self.value is not None:
            raise ValueError("an unresolved threshold must not silently carry a value")
        if self.source is not ThresholdSource.UNRESOLVED and self.value is None:
            raise ValueError("a resolved threshold requires a finite value")

    @property
    def resolved(self) -> bool:
        return self.source is not ThresholdSource.UNRESOLVED and self.value is not None

    def payload(self) -> dict[str, object]:
        return {
            "classification": "NEW_SAFETY_THRESHOLD",
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "reason": self.reason,
            "source": self.source.value,
            "scope": self.scope,
            "sensitivity": self.sensitivity,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class FeasibilityThresholds:
    """Threshold set used only by the new authority's final FK checks."""

    position_error: NewSafetyThreshold
    orientation_error: NewSafetyThreshold
    dls_equivalence: NewSafetyThreshold
    fk_runtime_parity: NewSafetyThreshold

    @property
    def acceptance_resolved(self) -> bool:
        return self.position_error.resolved and self.orientation_error.resolved

    @property
    def all_resolved(self) -> bool:
        return all(
            threshold.resolved
            for threshold in (
                self.position_error,
                self.orientation_error,
                self.dls_equivalence,
                self.fk_runtime_parity,
            )
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "g2_new_policy_safety_thresholds_v1",
            "acceptance_thresholds_resolved": self.acceptance_resolved,
            "all_thresholds_resolved": self.all_resolved,
            "thresholds": [
                self.position_error.payload(),
                self.orientation_error.payload(),
                self.dls_equivalence.payload(),
                self.fk_runtime_parity.payload(),
            ],
        }


def unresolved_production_thresholds() -> FeasibilityThresholds:
    """Return the intentionally fail-closed production threshold contract.

    The repository audit found no semantically matching production target-FK
    acceptance threshold.  Keeping these unresolved is safer than borrowing a
    post-physics tracking, grasp, or offline-planner tolerance.
    """

    common = "NEW_POLICY_SAFETY_AUTHORITY"
    return FeasibilityThresholds(
        position_error=NewSafetyThreshold(
            name="fk_position_error_max_m",
            value=None,
            unit="m",
            reason="new pre-controller requested-target versus FK acceptance",
            source=ThresholdSource.UNRESOLVED,
            scope=common,
            sensitivity="REQUIRED_BEFORE_LIVE",
        ),
        orientation_error=NewSafetyThreshold(
            name="fk_orientation_error_max_rad",
            value=None,
            unit="rad",
            reason="new pre-controller requested-target orientation acceptance",
            source=ThresholdSource.UNRESOLVED,
            scope=common,
            sensitivity="REQUIRED_BEFORE_LIVE",
        ),
        dls_equivalence=NewSafetyThreshold(
            name="dls_numerical_equivalence_max_rad",
            value=None,
            unit="rad",
            reason="pure helper versus source DLS numeric comparator",
            source=ThresholdSource.UNRESOLVED,
            scope=common,
            sensitivity="REQUIRED_BEFORE_LIVE",
        ),
        fk_runtime_parity=NewSafetyThreshold(
            name="fk_runtime_parity_max_m",
            value=None,
            unit="m",
            reason="pure production-FK model versus read-only runtime current pose",
            source=ThresholdSource.UNRESOLVED,
            scope=common,
            sensitivity="REQUIRED_BEFORE_LIVE",
        ),
    )


@runtime_checkable
class PureForwardKinematics(Protocol):
    """Read-only FK query port; implementations may not mutate live state."""

    model_id: str

    def forward(self, joint_position_rad: tuple[float, ...]) -> EEPose:
        """Return the EE pose for an arbitrary candidate joint vector."""


@dataclass(frozen=True)
class TargetResolutionState:
    """Read-only controller state needed to predict the selected target logic.

    This mirrors the source-defined translation-origin selection and existing
    30 mm target-queue clamp only for transparency checking.  The clamp is
    never used as a feasibility pass criterion.
    """

    previous_desired_ee_pose: EEPose
    pose_target_active: bool
    active_translation_axis: int
    maximum_outstanding_translation_axis_error_m: float

    def structurally_valid(self) -> bool:
        return (
            self.previous_desired_ee_pose.finite()
            and self.previous_desired_ee_pose.quaternion_nonzero()
            and isinstance(self.pose_target_active, bool)
            and isinstance(self.active_translation_axis, int)
            and -2 <= self.active_translation_axis <= 2
            and math.isfinite(float(self.maximum_outstanding_translation_axis_error_m))
            and float(self.maximum_outstanding_translation_axis_error_m) > 0.0
        )


@dataclass(frozen=True)
class ProductionIKSnapshot:
    """Immutable input required for a complete side-effect-free query."""

    snapshot_id: str
    control_epoch: int
    ordered_joint_names: tuple[str, ...]
    joint_position_rad: tuple[float, ...]
    hard_joint_lower_rad: tuple[float, ...]
    hard_joint_upper_rad: tuple[float, ...]
    measured_ee_pose: EEPose
    jacobian_root: tuple[tuple[float, ...], ...]
    target_resolution_state: TargetResolutionState
    production_model_fingerprint: str
    production_joint_limits_fingerprint: str
    controller_config_fingerprint: str
    production_usd_fingerprint: str
    resolved_joint_ids: tuple[int, ...]
    ee_body_name: str
    ee_frame_name: str
    body_offset_enabled: bool
    fk_model: PureForwardKinematics
    kinematic_model_binding: KinematicModelBinding
    elbow_command: float = 0.0
    dls_lambda: float = PRODUCTION_DLS_LAMBDA
    nullspace_damping: float = PRODUCTION_NULLSPACE_DAMPING
    nullspace_seed_joint_index: int = PRODUCTION_NULLSPACE_SEED_JOINT_INDEX
    maximum_nullspace_joint_delta_rad: float = PRODUCTION_NULLSPACE_MAX_DELTA_RAD

    def structurally_valid(self) -> bool:
        count = len(self.ordered_joint_names)
        try:
            jacobian_shape_ok = len(self.jacobian_root) == 6 and all(
                len(row) == count for row in self.jacobian_root
            )
            jacobian_finite = jacobian_shape_ok and all(
                math.isfinite(float(value)) for row in self.jacobian_root for value in row
            )
            vectors_finite = all(
                _is_finite_vector(vector, dimension=count)
                for vector in (
                    self.joint_position_rad,
                    self.hard_joint_lower_rad,
                    self.hard_joint_upper_rad,
                )
            )
            limits_ordered = all(
                float(low) <= float(high)
                for low, high in zip(self.hard_joint_lower_rad, self.hard_joint_upper_rad)
            )
            scalar_valid = (
                math.isfinite(float(self.elbow_command))
                and math.isfinite(float(self.dls_lambda))
                and float(self.dls_lambda) > 0.0
                and math.isfinite(float(self.nullspace_damping))
                and float(self.nullspace_damping) >= 0.0
                and math.isfinite(float(self.maximum_nullspace_joint_delta_rad))
                and float(self.maximum_nullspace_joint_delta_rad) > 0.0
                and 0 <= int(self.nullspace_seed_joint_index) < count
            )
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            bool(self.snapshot_id)
            and isinstance(self.control_epoch, int)
            and count == 7
            and all(isinstance(name, str) and name for name in self.ordered_joint_names)
            and vectors_finite
            and limits_ordered
            and self.measured_ee_pose.finite()
            and self.measured_ee_pose.quaternion_nonzero()
            and jacobian_finite
            and self.target_resolution_state.structurally_valid()
            and _is_sha256(self.production_model_fingerprint)
            and _is_sha256(self.production_joint_limits_fingerprint)
            and _is_sha256(self.controller_config_fingerprint)
            and _is_sha256(self.production_usd_fingerprint)
            and self.resolved_joint_ids == tuple(sorted(set(self.resolved_joint_ids)))
            and len(self.resolved_joint_ids) == count
            and all(isinstance(joint_id, int) and joint_id >= 0 for joint_id in self.resolved_joint_ids)
            and bool(self.ee_body_name)
            and bool(self.ee_frame_name)
            and isinstance(self.body_offset_enabled, bool)
            and isinstance(self.kinematic_model_binding, KinematicModelBinding)
            and isinstance(self.fk_model, PureForwardKinematics)
            and scalar_valid
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "control_epoch": self.control_epoch,
            "ordered_joint_names": list(self.ordered_joint_names),
            "joint_position_rad": _safe_vector(self.joint_position_rad),
            "hard_joint_lower_rad": _safe_vector(self.hard_joint_lower_rad),
            "hard_joint_upper_rad": _safe_vector(self.hard_joint_upper_rad),
            "measured_ee_pose": self.measured_ee_pose.payload(),
            "jacobian_shape": [len(self.jacobian_root), len(self.ordered_joint_names)],
            "target_resolution_state": {
                "previous_desired_ee_pose": self.target_resolution_state.previous_desired_ee_pose.payload(),
                "pose_target_active": self.target_resolution_state.pose_target_active,
                "active_translation_axis": self.target_resolution_state.active_translation_axis,
                "maximum_outstanding_translation_axis_error_m": (
                    self.target_resolution_state.maximum_outstanding_translation_axis_error_m
                ),
            },
            "production_model_fingerprint": self.production_model_fingerprint,
            "production_joint_limits_fingerprint": self.production_joint_limits_fingerprint,
            "controller_config_fingerprint": self.controller_config_fingerprint,
            "production_usd_fingerprint": self.production_usd_fingerprint,
            "resolved_joint_ids": list(self.resolved_joint_ids),
            "ee_body_name": self.ee_body_name,
            "ee_frame_name": self.ee_frame_name,
            "body_offset_enabled": self.body_offset_enabled,
            "fk_model_id": getattr(self.fk_model, "model_id", "UNKNOWN"),
            "kinematic_model_binding": self.kinematic_model_binding.value,
            "quaternion_convention": QUATERNION_CONVENTION,
            "elbow_command": self.elbow_command,
            "dls_lambda": self.dls_lambda,
            "nullspace_damping": self.nullspace_damping,
            "nullspace_seed_joint_index": self.nullspace_seed_joint_index,
            "maximum_nullspace_joint_delta_rad": self.maximum_nullspace_joint_delta_rad,
        }

    def immutable_payload(self) -> dict[str, object]:
        """Return every immutable field that affects a feasibility decision.

        ``public_payload`` intentionally omits the Jacobian values because it
        is convenient for concise reports.  A bridge freshness check needs
        the full read-only numerical snapshot instead: keeping only a
        snapshot ID and epoch would allow a changed state to reuse a receipt.
        """

        payload = self.public_payload()
        payload["jacobian_root"] = [list(row) for row in self.jacobian_root]
        return payload

    def fingerprint(self) -> str:
        """Stable fingerprint for bridge freshness, never an authority claim."""

        encoded = json.dumps(
            self.immutable_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedEETarget:
    """Absolute root-frame target derived without calling controller APIs."""

    pose: EEPose
    source_residual: EEResidualCommand
    resolution_semantics: str

    def payload(self) -> dict[str, object]:
        return {
            "pose": self.pose.payload(),
            "source_residual": {
                "translation_m": _safe_vector(self.source_residual.translation_m),
                "rotation_rad": _safe_vector(self.source_residual.rotation_rad),
                "frame": self.source_residual.frame.value,
            },
            "resolution_semantics": self.resolution_semantics,
        }


@dataclass(frozen=True)
class TargetResolution:
    """Requested endpoint and a non-authoritative prediction of controller output."""

    requested_target: ResolvedEETarget
    predicted_controller_target: ResolvedEETarget
    transparent: bool


@dataclass(frozen=True)
class DLSComputation:
    """Pre-execution numerical result, deliberately before clamp/rate limit."""

    q_dls_rad: tuple[float, ...]
    nullspace_delta_rad: tuple[float, ...]
    q_raw_ik_rad: tuple[float, ...]
    pose_error: tuple[float, float, float, float, float, float]


@runtime_checkable
class PureDLSSolver(Protocol):
    """Query-only DLS solver port used by the new authority."""

    def solve(
        self, snapshot: ProductionIKSnapshot, target: ResolvedEETarget
    ) -> DLSComputation:
        """Compute a raw candidate without calling a mutable controller."""


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), dtype=np.float64)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.array(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dtype=np.float64,
    )


def _axis_angle_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Exact scalar translation of Isaac Lab ``axis_angle_from_quat``.

    Do not normalize here.  ``compute_pose_error`` first forms
    ``q_target * inverse(q_source)`` and Isaac Lab then applies this routine
    directly, with the same sign canonicalization and ``1e-6`` Taylor branch.
    A normalization here would silently change non-unit snapshot semantics.
    """

    q = np.asarray(quaternion, dtype=np.float64).copy()
    q *= 1.0 - 2.0 * float(q[3] < 0.0)
    magnitude = float(np.linalg.norm(q[:3]))
    half_angle = math.atan2(magnitude, float(q[3]))
    angle = 2.0 * half_angle
    if abs(angle) > 1.0e-6:
        denominator = math.sin(half_angle) / angle
    else:
        denominator = 0.5 - angle * angle / 48.0
    if not math.isfinite(denominator) or denominator == 0.0:
        return np.full(3, np.nan, dtype=np.float64)
    return q[:3] / denominator


def pose_error_axis_angle(source: EEPose, target: EEPose) -> tuple[float, ...]:
    """Return production-convention [xyz, axis-angle] error in root frame."""

    source_position = np.asarray(source.position_root_m, dtype=np.float64)
    target_position = np.asarray(target.position_root_m, dtype=np.float64)
    source_quat = np.asarray(source.quaternion_root_xyzw, dtype=np.float64)
    target_quat = np.asarray(target.quaternion_root_xyzw, dtype=np.float64)
    source_norm_sq = float(np.dot(source_quat, source_quat))
    if not math.isfinite(source_norm_sq) or source_norm_sq == 0.0:
        return (math.nan,) * 6
    source_inverse = _quat_conjugate(source_quat) / source_norm_sq
    error_quat = _quat_multiply(target_quat, source_inverse)
    rotation = _axis_angle_from_quaternion(error_quat)
    return tuple(float(value) for value in np.concatenate((target_position - source_position, rotation)))


def _translation_axis(translation: np.ndarray) -> tuple[bool, int | None]:
    requested = np.abs(translation) > PRODUCTION_TARGET_QUEUE_EPSILON
    count = int(np.count_nonzero(requested))
    if count == 1:
        return True, int(np.argmax(requested))
    return False, None


def resolve_production_target(
    snapshot: ProductionIKSnapshot, command: EEResidualCommand
) -> TargetResolution:
    """Purely reproduce P0 translation-origin and queue-clamp semantics.

    The returned predicted endpoint is used only to reject an opaque execution
    mutation.  It is not used for DLS feasibility or as a replacement target.
    """

    if command.frame is not CartesianControlFrame.ROBOT_ROOT:
        raise ValueError("only root-frame EE residuals are supported")
    if command.rotation_rad != (0.0, 0.0, 0.0):
        raise ValueError("P0 translation-only authority requires exact-zero orientation")
    if not snapshot.target_resolution_state.structurally_valid():
        raise ValueError("malformed production target-resolution snapshot")
    if not snapshot.measured_ee_pose.finite() or not snapshot.measured_ee_pose.quaternion_nonzero():
        raise ValueError("non-finite measured EE pose")

    measured_position = np.asarray(snapshot.measured_ee_pose.position_root_m, dtype=np.float64)
    measured_quaternion = snapshot.measured_ee_pose.quaternion_root_xyzw
    previous = snapshot.target_resolution_state.previous_desired_ee_pose
    translation = np.asarray(command.translation_m, dtype=np.float64)
    single_axis, axis = _translation_axis(translation)
    any_translation = bool(np.any(np.abs(translation) > PRODUCTION_TARGET_QUEUE_EPSILON))
    state = snapshot.target_resolution_state

    origin = np.asarray(
        previous.position_root_m if state.pose_target_active else measured_position,
        dtype=np.float64,
    ).copy()
    if any_translation:
        origin = measured_position.copy()
        if (
            single_axis
            and axis is not None
            and state.pose_target_active
            and state.active_translation_axis == axis
        ):
            origin[axis] = float(previous.position_root_m[axis])

    requested_position = origin + translation
    requested_quaternion = (
        previous.quaternion_root_xyzw
        if state.pose_target_active and not any_translation
        else measured_quaternion
    )
    requested = ResolvedEETarget(
        pose=EEPose(
            position_root_m=tuple(float(value) for value in requested_position),
            quaternion_root_xyzw=tuple(float(value) for value in requested_quaternion),
        ),
        source_residual=command,
        resolution_semantics="G2_REDUNDANCY_ACTION_AXIS_ISOLATED_ORIGIN_V1",
    )
    queue_maximum = float(state.maximum_outstanding_translation_axis_error_m)
    predicted_position = measured_position + np.clip(
        requested_position - measured_position,
        -queue_maximum,
        queue_maximum,
    )
    predicted = ResolvedEETarget(
        pose=EEPose(
            position_root_m=tuple(float(value) for value in predicted_position),
            quaternion_root_xyzw=tuple(float(value) for value in requested_quaternion),
        ),
        source_residual=command,
        resolution_semantics="G2_REDUNDANCY_ACTION_QUEUE_CLAMP_PREDICTION_V1",
    )
    return TargetResolution(
        requested_target=requested,
        predicted_controller_target=predicted,
        transparent=requested.pose == predicted.pose,
    )


class ProductionDLSMath:
    """Pure numeric reproduction of the selected production DLS plus nullspace.

    It intentionally returns pre-clamp/pre-rate-limit values only.  It must
    never call ``DifferentialIKController.set_command`` or an action term.
    """

    def solve(
        self, snapshot: ProductionIKSnapshot, target: ResolvedEETarget
    ) -> DLSComputation:
        pose_error = np.asarray(pose_error_axis_angle(snapshot.measured_ee_pose, target.pose), dtype=np.float64)
        joint_position = np.asarray(snapshot.joint_position_rad, dtype=np.float64)
        jacobian = np.asarray(snapshot.jacobian_root, dtype=np.float64)
        if not np.isfinite(pose_error).all() or not np.isfinite(joint_position).all() or not np.isfinite(jacobian).all():
            return DLSComputation(
                q_dls_rad=tuple(float("nan") for _ in joint_position),
                nullspace_delta_rad=tuple(float("nan") for _ in joint_position),
                q_raw_ik_rad=tuple(float("nan") for _ in joint_position),
                pose_error=tuple(float(value) for value in pose_error),
            )
        identity6 = np.eye(6, dtype=np.float64)
        lambda_squared = float(snapshot.dls_lambda) ** 2
        # This is the same J^T inverse(J J^T + lambda^2 I) expression used by
        # the selected Isaac Lab DLS branch, not a different iterative solver.
        pseudoinverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + lambda_squared * identity6)
        q_dls = joint_position + pseudoinverse @ pose_error

        nullspace_identity6 = np.eye(6, dtype=np.float64)
        nullspace_pseudoinverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + float(snapshot.nullspace_damping) ** 2 * nullspace_identity6,
            nullspace_identity6,
        )
        projector = np.eye(jacobian.shape[1], dtype=np.float64) - nullspace_pseudoinverse @ jacobian
        seed = np.zeros(jacobian.shape[1], dtype=np.float64)
        seed[int(snapshot.nullspace_seed_joint_index)] = float(
            np.clip(snapshot.elbow_command, -1.0, 1.0)
        )
        unbounded = projector @ seed * float(snapshot.maximum_nullspace_joint_delta_rad)
        nullspace_delta = np.clip(
            unbounded,
            -float(snapshot.maximum_nullspace_joint_delta_rad),
            float(snapshot.maximum_nullspace_joint_delta_rad),
        )
        raw = q_dls + nullspace_delta
        return DLSComputation(
            q_dls_rad=tuple(float(value) for value in q_dls),
            nullspace_delta_rad=tuple(float(value) for value in nullspace_delta),
            q_raw_ik_rad=tuple(float(value) for value in raw),
            pose_error=tuple(float(value) for value in pose_error),
        )


@dataclass(frozen=True)
class FeasibilityReceipt:
    """Full explicit outcome of an authority query; never a controller command."""

    evaluated: bool
    accepted: bool
    reason: FeasibilityRejectReason
    requested_target: ResolvedEETarget | None
    validated_target: ResolvedEETarget | None
    predicted_controller_target: ResolvedEETarget | None
    q_candidate_rad: tuple[float, ...] | None
    achieved_ee_pose: EEPose | None
    position_error_m: float | None
    orientation_error_rad: float | None
    snapshot_id: str | None
    control_epoch: int | None
    q_dls_rad: tuple[float, ...] | None = None
    q_raw_ik_rad: tuple[float, ...] | None = None
    q_after_soft_limit_clamp_rad: tuple[float, ...] | None = None
    q_after_rate_limit_rad: tuple[float, ...] | None = None
    hard_joint_limits_satisfied: bool | None = None
    target_transparent: bool | None = None
    thresholds_resolved: bool = False
    evaluation_scope: AuthorityEvaluationScope = AuthorityEvaluationScope.PRODUCTION_DEPLOYMENT
    error_detail: str | None = None
    authority_origin: str = AUTHORITY_ORIGIN
    authority_version: str = NEW_POLICY_SAFETY_AUTHORITY_VERSION
    schema: str = NEW_POLICY_SAFETY_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if self.authority_origin != AUTHORITY_ORIGIN:
            raise ValueError("receipt authority origin must identify the new authority")
        if self.authority_version != NEW_POLICY_SAFETY_AUTHORITY_VERSION:
            raise ValueError("unsupported new authority version")
        if self.schema != NEW_POLICY_SAFETY_AUTHORITY_SCHEMA:
            raise ValueError("unsupported new authority receipt schema")
        if not isinstance(self.reason, FeasibilityRejectReason):
            raise ValueError("receipt reason must be FeasibilityRejectReason")
        if not isinstance(self.evaluation_scope, AuthorityEvaluationScope):
            raise ValueError("receipt evaluation scope must be AuthorityEvaluationScope")
        if self.accepted:
            if self.reason is not FeasibilityRejectReason.ACCEPT:
                raise ValueError("accepted receipt must use ACCEPT")
            if self.requested_target is None or self.validated_target is None:
                raise ValueError("accepted receipt requires requested and validated target")
            if self.requested_target != self.validated_target:
                raise ValueError("accepted target may not be changed")
            if self.target_transparent is not True or self.hard_joint_limits_satisfied is not True:
                raise ValueError("accepted receipt needs transparent hard-limit-valid target")
        elif self.reason is FeasibilityRejectReason.ACCEPT:
            raise ValueError("rejected receipt cannot use ACCEPT")

    @property
    def downstream_submission_performed(self) -> bool:
        """Always false: submission belongs to a later, separately bound bridge."""

        return False

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_origin": self.authority_origin,
            "authority_version": self.authority_version,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "reason": self.reason.value,
            "requested_target": None if self.requested_target is None else self.requested_target.payload(),
            "validated_target": None if self.validated_target is None else self.validated_target.payload(),
            "predicted_controller_target": (
                None
                if self.predicted_controller_target is None
                else self.predicted_controller_target.payload()
            ),
            "q_candidate_rad": _safe_vector(self.q_candidate_rad),
            "q_dls_rad": _safe_vector(self.q_dls_rad),
            "q_raw_ik_rad": _safe_vector(self.q_raw_ik_rad),
            "q_after_soft_limit_clamp_rad": _safe_vector(self.q_after_soft_limit_clamp_rad),
            "q_after_rate_limit_rad": _safe_vector(self.q_after_rate_limit_rad),
            "execution_target_mutations": "NOT_EVALUATED_BY_PURE_FEASIBILITY_AUTHORITY",
            "achieved_ee_pose": None if self.achieved_ee_pose is None else self.achieved_ee_pose.payload(),
            "position_error_m": _safe_scalar(self.position_error_m),
            "orientation_error_rad": _safe_scalar(self.orientation_error_rad),
            "hard_joint_limits_satisfied": self.hard_joint_limits_satisfied,
            "target_transparent": self.target_transparent,
            "thresholds_resolved": self.thresholds_resolved,
            "evaluation_scope": self.evaluation_scope.value,
            "snapshot_id": self.snapshot_id,
            "control_epoch": self.control_epoch,
            "downstream_submission_performed": False,
            "error_detail": self.error_detail,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class NewPolicySafetyAuthority:
    """Evaluate an immutable snapshot without issuing any execution command."""

    def __init__(
        self,
        *,
        thresholds: FeasibilityThresholds | None = None,
        dls_solver: PureDLSSolver | None = None,
        target_resolver: Callable[[ProductionIKSnapshot, EEResidualCommand], TargetResolution]
        | None = None,
        evaluation_scope: AuthorityEvaluationScope = AuthorityEvaluationScope.PRODUCTION_DEPLOYMENT,
    ) -> None:
        if not isinstance(evaluation_scope, AuthorityEvaluationScope):
            raise ValueError("evaluation_scope must be AuthorityEvaluationScope")
        # Constructor injection is useful for static fixtures, but it is not
        # an authorized way to install a production IK/FK/threshold authority.
        # Until a separately attested immutable production manifest exists,
        # every such override is rejected in deployment scope.
        self._production_constructor_override_requested = (
            evaluation_scope is AuthorityEvaluationScope.PRODUCTION_DEPLOYMENT
            and (thresholds is not None or dls_solver is not None or target_resolver is not None)
        )
        self._thresholds = unresolved_production_thresholds() if thresholds is None else thresholds
        if not isinstance(self._thresholds, FeasibilityThresholds):
            raise ValueError("thresholds must be FeasibilityThresholds")
        self._dls_solver: PureDLSSolver = ProductionDLSMath() if dls_solver is None else dls_solver
        self._evaluation_scope = evaluation_scope
        # Injection is test/adapter plumbing only.  The default is the pure
        # source-derived resolver above; no mutable action term is callable.
        self._target_resolver = (
            resolve_production_target if target_resolver is None else target_resolver
        )

    @property
    def thresholds(self) -> FeasibilityThresholds:
        return self._thresholds

    @property
    def evaluation_scope(self) -> AuthorityEvaluationScope:
        return self._evaluation_scope

    def _receipt(
        self,
        *,
        accepted: bool,
        reason: FeasibilityRejectReason,
        requested_target: ResolvedEETarget | None = None,
        validated_target: ResolvedEETarget | None = None,
        predicted_controller_target: ResolvedEETarget | None = None,
        q_candidate_rad: tuple[float, ...] | None = None,
        achieved_ee_pose: EEPose | None = None,
        position_error_m: float | None = None,
        orientation_error_rad: float | None = None,
        snapshot: ProductionIKSnapshot | None = None,
        q_dls_rad: tuple[float, ...] | None = None,
        q_raw_ik_rad: tuple[float, ...] | None = None,
        hard_joint_limits_satisfied: bool | None = None,
        target_transparent: bool | None = None,
        error_detail: str | None = None,
    ) -> FeasibilityReceipt:
        return FeasibilityReceipt(
            evaluated=True,
            accepted=accepted,
            reason=reason,
            requested_target=requested_target,
            validated_target=validated_target,
            predicted_controller_target=predicted_controller_target,
            q_candidate_rad=q_candidate_rad,
            achieved_ee_pose=achieved_ee_pose,
            position_error_m=position_error_m,
            orientation_error_rad=orientation_error_rad,
            snapshot_id=None if snapshot is None else snapshot.snapshot_id,
            control_epoch=None if snapshot is None else snapshot.control_epoch,
            q_dls_rad=q_dls_rad,
            q_raw_ik_rad=q_raw_ik_rad,
            hard_joint_limits_satisfied=hard_joint_limits_satisfied,
            target_transparent=target_transparent,
            thresholds_resolved=self._thresholds.all_resolved,
            evaluation_scope=self._evaluation_scope,
            error_detail=error_detail,
        )

    def reject_without_snapshot(
        self,
        *,
        reason: FeasibilityRejectReason = FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
        error_detail: str,
    ) -> FeasibilityReceipt:
        """Create a strict, explicit reject receipt when capture itself fails.

        The bridge uses this when no immutable snapshot exists.  It is not an
        acceptance path and does not expose a command or controller state.
        """

        if reason is FeasibilityRejectReason.ACCEPT:
            raise ValueError("snapshotless receipt cannot be accepted")
        return self._receipt(accepted=False, reason=reason, error_detail=error_detail)

    def evaluate(
        self, snapshot: object, candidate_command: object
    ) -> FeasibilityReceipt:
        """Run the explicit hierarchy with no mutable controller interaction."""

        if not isinstance(snapshot, ProductionIKSnapshot) or not snapshot.structurally_valid():
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.INVALID_INPUT,
                error_detail="MALFORMED_OR_NONFINITE_SNAPSHOT",
            )
        if not isinstance(candidate_command, EEResidualCommand):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.INVALID_INPUT,
                snapshot=snapshot,
                error_detail="MALFORMED_CANDIDATE_TARGET",
            )
        if (
            candidate_command.frame is not CartesianControlFrame.ROBOT_ROOT
            or candidate_command.rotation_rad != (0.0, 0.0, 0.0)
            or not _is_finite_vector(candidate_command.translation_m, dimension=3)
        ):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.INVALID_INPUT,
                snapshot=snapshot,
                error_detail="P0_FRAME_OR_ORIENTATION_OR_FINITE_CONTRACT_FAILURE",
            )
        if self._evaluation_scope is AuthorityEvaluationScope.PRODUCTION_DEPLOYMENT:
            if snapshot.kinematic_model_binding is not KinematicModelBinding.PRODUCTION_EQUIVALENT:
                return self._receipt(
                    accepted=False,
                    reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                    snapshot=snapshot,
                    error_detail="PRODUCTION_FK_MODEL_BINDING_UNRESOLVED",
                )
            if self._production_constructor_override_requested:
                return self._receipt(
                    accepted=False,
                    reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                    snapshot=snapshot,
                    error_detail="PRODUCTION_CONSTRUCTOR_OVERRIDE_FORBIDDEN_UNTIL_MANIFEST_BINDING",
                )
            if float(snapshot.elbow_command) != 0.0:
                return self._receipt(
                    accepted=False,
                    reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                    snapshot=snapshot,
                    error_detail="P0_4D_ELBOW_COMMAND_MUST_BE_EXACT_ZERO",
                )
            if (
                float(snapshot.dls_lambda) != PRODUCTION_DLS_LAMBDA
                or float(snapshot.nullspace_damping) != PRODUCTION_NULLSPACE_DAMPING
                or int(snapshot.nullspace_seed_joint_index) != PRODUCTION_NULLSPACE_SEED_JOINT_INDEX
                or float(snapshot.maximum_nullspace_joint_delta_rad)
                != PRODUCTION_NULLSPACE_MAX_DELTA_RAD
            ):
                return self._receipt(
                    accepted=False,
                    reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                    snapshot=snapshot,
                    error_detail="PRODUCTION_DLS_NULLSPACE_PARAMETER_BINDING_MISMATCH",
                )
        current_limits_satisfied = all(
            float(lower) <= float(value) <= float(upper)
            for value, lower, upper in zip(
                snapshot.joint_position_rad,
                snapshot.hard_joint_lower_rad,
                snapshot.hard_joint_upper_rad,
            )
        )
        if not current_limits_satisfied:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.JOINT_LIMIT_VIOLATION,
                q_candidate_rad=snapshot.joint_position_rad,
                snapshot=snapshot,
                hard_joint_limits_satisfied=False,
                error_detail="CURRENT_STATE_OUTSIDE_PRODUCTION_HARD_JOINT_LIMITS",
            )
        try:
            resolution = self._target_resolver(snapshot, candidate_command)
        except Exception as error:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                snapshot=snapshot,
                error_detail=f"TARGET_RESOLUTION_FAILURE:{type(error).__name__}",
            )
        if (
            not isinstance(resolution, TargetResolution)
            or not isinstance(resolution.requested_target, ResolvedEETarget)
            or not isinstance(resolution.predicted_controller_target, ResolvedEETarget)
            or not isinstance(resolution.transparent, bool)
            or not resolution.requested_target.pose.finite()
            or not resolution.requested_target.pose.quaternion_nonzero()
            or not resolution.predicted_controller_target.pose.finite()
            or not resolution.predicted_controller_target.pose.quaternion_nonzero()
            or resolution.requested_target.source_residual != candidate_command
            or resolution.predicted_controller_target.source_residual != candidate_command
        ):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                snapshot=snapshot,
                error_detail="MALFORMED_TARGET_RESOLUTION_RESULT",
            )
        if not resolution.transparent or resolution.requested_target.pose != resolution.predicted_controller_target.pose:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.TARGET_TRANSPARENCY_VIOLATION,
                requested_target=resolution.requested_target,
                predicted_controller_target=resolution.predicted_controller_target,
                snapshot=snapshot,
                target_transparent=False,
                error_detail="EXISTING_TARGET_QUEUE_WOULD_MUTATE_REQUESTED_ENDPOINT",
            )
        try:
            computation = self._dls_solver.solve(snapshot, resolution.requested_target)
        except Exception as error:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.IK_SOLVE_FAILURE,
                requested_target=resolution.requested_target,
                snapshot=snapshot,
                target_transparent=True,
                error_detail=f"PURE_DLS_FAILURE:{type(error).__name__}",
            )
        if not isinstance(computation, DLSComputation):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.IK_SOLVE_FAILURE,
                requested_target=resolution.requested_target,
                snapshot=snapshot,
                target_transparent=True,
                error_detail="MALFORMED_PURE_DLS_RESULT",
            )
        raw = computation.q_raw_ik_rad
        if not _is_finite_vector(raw, dimension=len(snapshot.ordered_joint_names)):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.IK_NONFINITE,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                target_transparent=True,
                error_detail="NONFINITE_Q_RAW_IK",
            )
        limits_satisfied = all(
            float(lower) <= float(value) <= float(upper)
            for value, lower, upper in zip(
                raw, snapshot.hard_joint_lower_rad, snapshot.hard_joint_upper_rad
            )
        )
        if not limits_satisfied:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.JOINT_LIMIT_VIOLATION,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=False,
                target_transparent=True,
                error_detail="RAW_IK_OUTSIDE_PRODUCTION_HARD_JOINT_LIMITS",
            )
        try:
            achieved = snapshot.fk_model.forward(tuple(float(value) for value in raw))
        except Exception as error:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.FK_NONFINITE,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=True,
                target_transparent=True,
                error_detail=f"PURE_FK_FAILURE:{type(error).__name__}",
            )
        if not isinstance(achieved, EEPose) or not achieved.finite() or not achieved.quaternion_nonzero():
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.FK_NONFINITE,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                achieved_ee_pose=achieved if isinstance(achieved, EEPose) else None,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=True,
                target_transparent=True,
                error_detail="NONFINITE_OR_MALFORMED_FK_OUTPUT",
            )
        error = pose_error_axis_angle(achieved, resolution.requested_target.pose)
        if not _is_finite_vector(error, dimension=6):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.FK_NONFINITE,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                achieved_ee_pose=achieved,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=True,
                target_transparent=True,
                error_detail="NONFINITE_FK_TARGET_ERROR",
            )
        position_error = float(np.linalg.norm(np.asarray(error[:3], dtype=np.float64)))
        orientation_error = float(np.linalg.norm(np.asarray(error[3:], dtype=np.float64)))
        threshold_resolved = (
            self._thresholds.all_resolved
            if self._evaluation_scope is AuthorityEvaluationScope.PRODUCTION_DEPLOYMENT
            else self._thresholds.acceptance_resolved
        )
        if not threshold_resolved:
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.INTERNAL_VALIDATION_FAILURE,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                achieved_ee_pose=achieved,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=True,
                target_transparent=True,
                error_detail="NEW_SAFETY_THRESHOLD_AUTHORITY_UNRESOLVED",
            )
        assert self._thresholds.position_error.value is not None
        assert self._thresholds.orientation_error.value is not None
        if position_error > float(self._thresholds.position_error.value):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.POSITION_ERROR_EXCEEDED,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                achieved_ee_pose=achieved,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=True,
                target_transparent=True,
            )
        if orientation_error > float(self._thresholds.orientation_error.value):
            return self._receipt(
                accepted=False,
                reason=FeasibilityRejectReason.ORIENTATION_ERROR_EXCEEDED,
                requested_target=resolution.requested_target,
                q_candidate_rad=raw,
                achieved_ee_pose=achieved,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                snapshot=snapshot,
                q_dls_rad=computation.q_dls_rad,
                q_raw_ik_rad=raw,
                hard_joint_limits_satisfied=True,
                target_transparent=True,
            )
        return self._receipt(
            accepted=True,
            reason=FeasibilityRejectReason.ACCEPT,
            requested_target=resolution.requested_target,
            validated_target=resolution.requested_target,
            predicted_controller_target=resolution.predicted_controller_target,
            q_candidate_rad=raw,
            achieved_ee_pose=achieved,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            snapshot=snapshot,
            q_dls_rad=computation.q_dls_rad,
            q_raw_ik_rad=raw,
            hard_joint_limits_satisfied=True,
            target_transparent=True,
        )


def static_thresholds_for_test_fixture() -> FeasibilityThresholds:
    """Explicit test-only thresholds, never a production deployment preset."""

    source = ThresholdSource.ENGINEERING_DESIGN_CHOICE
    scope = "STATIC_TEST_FIXTURE_ONLY_NOT_PRODUCTION_DEPLOYMENT"
    return FeasibilityThresholds(
        position_error=NewSafetyThreshold(
            name="fk_position_error_max_m",
            value=1.0e-6,
            unit="m",
            reason="linear fixture permits a deterministic DLS/FK regression check",
            source=source,
            scope=scope,
            sensitivity="0.5x,0.75x,1.0x,1.25x,1.5x static only",
        ),
        orientation_error=NewSafetyThreshold(
            name="fk_orientation_error_max_rad",
            value=1.0e-6,
            unit="rad",
            reason="translation-only fixture holds orientation exactly constant",
            source=source,
            scope=scope,
            sensitivity="0.5x,0.75x,1.0x,1.25x,1.5x static only",
        ),
        dls_equivalence=NewSafetyThreshold(
            name="dls_numerical_equivalence_max_rad",
            value=1.0e-8,
            unit="rad",
            reason="float32 source-controller versus float64 pure-math fixture comparison",
            source=source,
            scope=scope,
            sensitivity="fixed precommitted comparator for static source formula test",
        ),
        fk_runtime_parity=NewSafetyThreshold(
            name="fk_runtime_parity_max_m",
            value=1.0e-6,
            unit="m",
            reason="fixture-only current-pose parity; no production parity claim",
            source=source,
            scope=scope,
            sensitivity="not a deployment threshold",
        ),
    )


__all__ = [
    "AUTHORITY_ORIGIN",
    "AuthorityEvaluationScope",
    "EXISTING_PRODUCTION_ACCEPTANCE_AUTHORITY",
    "NEW_POLICY_SAFETY_AUTHORITY_SCHEMA",
    "NEW_POLICY_SAFETY_AUTHORITY_VERSION",
    "QUATERNION_CONVENTION",
    "DLSComputation",
    "EEPose",
    "FeasibilityReceipt",
    "FeasibilityRejectReason",
    "FeasibilityThresholds",
    "KinematicModelBinding",
    "NewPolicySafetyAuthority",
    "NewSafetyThreshold",
    "ProductionDLSMath",
    "ProductionIKSnapshot",
    "PureDLSSolver",
    "PureForwardKinematics",
    "ResolvedEETarget",
    "TargetResolution",
    "TargetResolutionState",
    "ThresholdSource",
    "pose_error_axis_angle",
    "resolve_production_target",
    "static_thresholds_for_test_fixture",
    "unresolved_production_thresholds",
]
