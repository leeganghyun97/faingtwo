#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded P0 attestation for the high-level G2 policy branch.

This diagnostic binds only the new 4-D policy action contract to the already
configured 8-D Cartesian/binary-gripper controller surface:

``[dx, dy, dz, g] -> router -> [dx,dy,dz,0,0,0,0,open/close] -> env.step``.

It never calls an articulation joint-target API, never creates an individual
finger/passive-joint command, and never evaluates M2/H4.7 passive-chain
dynamics.  Isaac stepping is used solely to observe the existing controller's
end-effector response; it is not a PhysX-mechanics qualification.

P0-A is intentionally fail-closed when the repository has no pre-existing
public workspace/IK rejection authority.  This script must not invent one just
to make a policy boundary pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))


P0_SCHEMA = "g2_policy_branch_p0_live_attestation_v3"
P0_SOURCE_PATHS = (
    "scripts/diagnostics/run_g2_policy_branch_p0_attestation.py",
    "scripts/diagnostics/run_g2_policy_branch_p0_supervisor.py",
    "source/geniesim/rl/isaaclab/g2_policy_branch/action_interface.py",
    "source/geniesim/rl/isaaclab/g2_policy_branch/ee_safety_validator.py",
    "source/geniesim/rl/isaaclab/g2_policy_branch/observation.py",
    "source/geniesim/rl/isaaclab/g2_policy_branch/p0_runtime_contract.py",
    "source/geniesim/rl/isaaclab/g2_lift_methodology.py",
    "source/geniesim/rl/isaaclab/g2_lift_env_cfg.py",
    "source/geniesim/rl/isaaclab/g2_lift_task_mdp.py",
    "source/geniesim/rl/isaaclab/g2_lift_rgbd_env_cfg.py",
    "source/geniesim/rl/isaaclab/g2_collision_authority.py",
    "source/geniesim/rl/isaaclab/g2_redundancy_action.py",
    "source/geniesim/rl/isaaclab/g2_redundancy_teleop_env_cfg.py",
    "source/geniesim/rl/isaaclab/g2_camera_timing.py",
    "source/geniesim/rl/isaaclab/g2_quaternion.py",
    "source/geniesim/rl/isaaclab/g2_teleop_dataset.py",
    "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda",
    "configs/diagnostics/m2_arm_move_target_seed42_v1.json",
    "tests/test_g2_ee_safety_validator.py",
    "tests/test_g2_policy_branch_p0_attestation_static.py",
    "tests/test_g2_high_level_policy_branch.py",
    "tests/test_g2_teleop_dataset.py",
)
PRODUCTION_USD_RELATIVE_PATH = (
    "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda"
)
FROZEN_M2_TRAJECTORY_RELATIVE_PATH = (
    "configs/diagnostics/m2_arm_move_target_seed42_v1.json"
)
PRODUCTION_USD_SHA256 = "7751a265b02b1c4f6d290a1555d5bb8f2750b7280e66cb6a94042954dc032132"
FROZEN_M2_TRAJECTORY_SHA256 = "1ff58c1310f89f6cbcf480a885a9bee2258d34c075197c88518c318da9bd88e3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    """Capture read-only repository provenance for immutable P0 evidence."""

    def query(arguments: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "head": query(["rev-parse", "HEAD"]),
        "head_short": query(["rev-parse", "--short=12", "HEAD"]),
        "status_porcelain": query(["status", "--porcelain=v1"]),
        "p0_paths_status_porcelain": query(
            ["status", "--porcelain=v1", "--", *P0_SOURCE_PATHS]
        ),
    }


def _plain(value: Any) -> Any:
    # Runtime evidence includes small pure-contract dataclasses such as
    # EEResidualCommand and GripperSubmission.  Serialize their fields rather
    # than leaking Python objects into an otherwise strict JSON report.  This
    # does not touch live controller execution or make a semantic conversion.
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "detach"):
        return _plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        json.dumps(_plain(row), sort_keys=True, allow_nan=False)
        for row in rows
    ]
    _atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _source_manifest() -> dict[str, Any]:
    files = {relative: _sha256(ROOT / relative) for relative in P0_SOURCE_PATHS}
    return {
        "schema": "g2_policy_branch_p0_source_freeze_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files_sha256": files,
        "git_provenance": _git_provenance(),
        "production_usd": {
            "relative_path": PRODUCTION_USD_RELATIVE_PATH,
            "sha256": files[PRODUCTION_USD_RELATIVE_PATH],
            "expected_sha256": PRODUCTION_USD_SHA256,
            "matches_expected": files[PRODUCTION_USD_RELATIVE_PATH]
            == PRODUCTION_USD_SHA256,
        },
        "frozen_m2_trajectory": {
            "relative_path": FROZEN_M2_TRAJECTORY_RELATIVE_PATH,
            "sha256": files[FROZEN_M2_TRAJECTORY_RELATIVE_PATH],
            "expected_sha256": FROZEN_M2_TRAJECTORY_SHA256,
            "matches_expected": files[FROZEN_M2_TRAJECTORY_RELATIVE_PATH]
            == FROZEN_M2_TRAJECTORY_SHA256,
            "executed": False,
            # H4.7/M2 is a read-only historical reference in this P0 branch.
            # Its hash is retained for provenance, but it is not a P0 gate.
            "p0_gating_authority": False,
        },
        "scope": {
            "m2_or_h47_mechanical_authority_evaluated": False,
            "joint_level_gripper_command_used": False,
            "training_started": False,
            "learned_policy_loaded": False,
        },
    }


def _protected_reference_hash_checks(manifest: dict[str, Any]) -> dict[str, bool]:
    """Return the two immutable-reference integrity checks required by P0.

    The M2 trajectory is deliberately not executed by this policy-controller
    diagnostic and is not a P0 semantic authority.  It is still a frozen
    reference named by the P0 evidence contract, so an altered file must stop
    a new P0 run before Isaac is launched rather than becoming silent mixed
    provenance.
    """

    return {
        "production_usd": bool(
            manifest.get("production_usd", {}).get("matches_expected", False)
        ),
        "frozen_m2_trajectory": bool(
            manifest.get("frozen_m2_trajectory", {}).get("matches_expected", False)
        ),
    }


def _json_sha256(payload: dict[str, Any]) -> str:
    """Fingerprint one immutable JSON evidence payload deterministically."""

    encoded = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _ee_command_payload(command: Any) -> dict[str, Any]:
    """Return the exact high-level EE command representation used by P0.

    This is deliberately not a controller target conversion.  It only gives
    the bridge an immutable way to prove that the command accepted by the
    validator is byte-for-byte the command submitted to the existing 8-D
    controller surface.  A receipt flag on its own is not sufficient proof:
    that public object can be copied independently of the bridge call.
    """

    return {
        "translation_m": list(command.translation_m),
        "rotation_rad": list(command.rotation_rad),
        "frame": command.frame.value,
    }


def _ee_command_fingerprint(command: Any) -> str:
    return _json_sha256(
        {
            "schema": "g2_policy_branch_p0_exact_ee_command_v1",
            "command": _ee_command_payload(command),
        }
    )


def _gate(status: str, *, evidence: Any = None, reason: str = "") -> dict[str, Any]:
    if status not in {"PASS", "FAIL", "BLOCKED", "NOT_APPLICABLE"}:
        raise ValueError(f"unsupported P0 gate status: {status}")
    return {"status": status, "reason": reason, "evidence": _plain(evidence)}


def _require_p0_a_pass(report: Path) -> None:
    document = json.loads(report.read_text(encoding="utf-8"))
    if document.get("phase") != "P0_A":
        raise RuntimeError("P0_B_OR_C_REQUIRES_P0_A_REPORT")
    if document.get("p0_semantic_verdict") != "PASS":
        raise RuntimeError("P0_B_OR_C_BLOCKED_BY_P0_A")


def _tensor(value: Any):
    """Convert the small public Isaac tensor/warp surfaces used by P0 only."""

    import torch

    if isinstance(value, torch.Tensor):
        return value
    converted = getattr(value, "torch", None)
    if isinstance(converted, torch.Tensor):
        return converted
    try:
        return torch.from_dlpack(value)
    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
        raise TypeError(f"unsupported live P0 tensor type: {type(value)!r}") from error


def _quaternion_multiply_xyzw(left, right):
    import torch

    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _rotate_xyzw(quaternion, vector):
    import torch

    xyz = quaternion[..., :3]
    w = quaternion[..., 3:]
    cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * w * cross + 2.0 * torch.linalg.cross(xyz, cross, dim=-1)


def _world_ee_pose_to_root(robot: Any, ee_frame: Any):
    """Explicitly transform live world EE pose into robot-root XYZW semantics."""

    import torch

    from geniesim.rl.isaaclab.g2_quaternion import (
        isaaclab_native_quaternion_order,
        quaternion_native_to_xyzw,
    )

    native_order = isaaclab_native_quaternion_order()
    root_position_w = _tensor(robot.data.root_pos_w).to(torch.float32)
    root_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(robot.data.root_quat_w).to(torch.float32), native_order=native_order
    )
    ee_position_w = _tensor(ee_frame.data.target_pos_w)[:, 0].to(torch.float32)
    ee_quaternion_xyzw = quaternion_native_to_xyzw(
        _tensor(ee_frame.data.target_quat_w)[:, 0].to(torch.float32),
        native_order=native_order,
    )
    inverse_root = root_quaternion_xyzw.clone()
    inverse_root[:, :3] *= -1.0
    position_root = _rotate_xyzw(inverse_root, ee_position_w - root_position_w)
    quaternion_root = _quaternion_multiply_xyzw(inverse_root, ee_quaternion_xyzw)
    quaternion_root = quaternion_root / torch.linalg.vector_norm(
        quaternion_root, dim=-1, keepdim=True
    )
    quaternion_root = torch.where(
        quaternion_root[:, 3:4] < 0.0, -quaternion_root, quaternion_root
    )
    return position_root, quaternion_root, ee_position_w, ee_quaternion_xyzw


def _orientation_distance_rad(first, second) -> float:
    import torch

    dot = torch.clamp(torch.abs(torch.sum(first * second, dim=-1)), max=1.0)
    return float((2.0 * torch.acos(dot)).amax().item())


@dataclass
class _P0ExecutionRecord:
    label: str
    requested_action: Any
    previous_action_before: tuple[float, ...]
    previous_action_after: tuple[float, ...]
    route: Any
    controller_surface_action_8d: Any
    target_root_before_m: Any
    target_root_after_m: Any
    measured_root_before_m: Any
    measured_root_after_m: Any
    measured_world_before_m: Any
    measured_world_after_m: Any
    orientation_drift_rad: float
    observed_gripper_intent: str
    terminated: bool
    truncated: bool
    active_terminations: list[str]
    validator_receipt: Any
    downstream_activity_before: dict[str, int]
    downstream_activity_after: dict[str, int]


class _ExistingControllerP0Bridge:
    """Narrow diagnostic adapter over the existing 8-D high-level surface.

    The deterministic policy-pre-controller validator is the outer boundary
    for every routed Cartesian request.  It is deliberately fail-closed when
    no source-authoritative dry-run IK acceptance receipt is available.  The
    bridge itself adds no workspace/IK rule and still owns only the existing
    8-D controller envelope plus abstract gripper adapter.
    """

    def __init__(self, env: Any) -> None:
        import torch

        from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
            AbstractGripperIntent,
            CartesianControlFrame,
            CartesianResidualScale,
            GripperSubmission,
            PolicyActionContractError,
            SafetyProjection,
        )
        from geniesim.rl.isaaclab.g2_policy_branch.ee_safety_validator import (
            EESafetyValidator,
        )

        self.env = env
        self.AbstractGripperIntent = AbstractGripperIntent
        self.CartesianControlFrame = CartesianControlFrame
        self.GripperSubmission = GripperSubmission
        self.PolicyActionContractError = PolicyActionContractError
        self.SafetyProjection = SafetyProjection
        self.scale = CartesianResidualScale()
        # No provider is supplied here.  Track A has not identified an
        # existing source-authoritative dry-run IK acceptance authority, so
        # the validator must reject every routed EE target rather than let the
        # P0 harness manufacture a positive safety result.
        self.pre_controller_validator = EESafetyValidator()
        if self.pre_controller_validator.config.residual_scale != self.scale:
            raise RuntimeError("P0_PRE_CONTROLLER_VALIDATOR_SCALE_MISMATCH")
        if int(env.action_manager.total_action_dim) != 8:
            raise RuntimeError(
                "P0_REQUIRES_EXISTING_8D_REDUNDANCY_HIGH_LEVEL_SURFACE"
            )
        self.arm_term = env.action_manager._terms["arm_action"]
        self.gripper_term = env.action_manager._terms["gripper_action"]
        expected_scale = (
            self.scale.translation_m_per_normalized,
            self.scale.translation_m_per_normalized,
            self.scale.translation_m_per_normalized,
            self.scale.rotation_rad_per_normalized,
            self.scale.rotation_rad_per_normalized,
            self.scale.rotation_rad_per_normalized,
            1.0,
        )
        actual_scale = tuple(float(value) for value in self.arm_term.cfg.scale)
        if actual_scale != expected_scale:
            raise RuntimeError(
                f"P0_EXISTING_CONTROLLER_SCALE_MISMATCH:{actual_scale}!={expected_scale}"
            )
        self._surface = torch.zeros((1, 8), dtype=torch.float32, device=env.device)
        self._surface[:, 7] = 1.0  # Existing abstract OPEN compatibility sign.
        self.available = True
        self.reject_all_for_fault_injection = False
        self._ee_submission_count = 0
        self._gripper_submission_count = 0
        self._env_step_count = 0
        self._safe_hold_count = 0
        self._last_validator_receipt = None
        # The validator receipt is a policy-observable record, not proof that
        # this bridge submitted anything.  Keep a private, one-shot binding
        # from an exact accepted command to the actual controller-surface
        # write.  This prevents a copied/forged public receipt from being
        # treated as downstream-submission evidence.
        self._pending_submission_binding: dict[str, Any] | None = None
        self._last_bridge_submission_evidence: dict[str, Any] | None = None
        self._submission_epoch = 0

    @property
    def controller_scale(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.arm_term.cfg.scale)

    @property
    def controller_surface(self):
        return self._surface.clone()

    @property
    def last_validator_receipt(self):
        """Latest immutable pre-controller decision, including rejections."""

        return self._last_validator_receipt

    @property
    def last_bridge_submission_evidence(self) -> dict[str, Any] | None:
        """Private proof emitted only after the existing surface was written.

        The payload intentionally contains no joint target or low-level
        actuator command.  It binds the accepted validator command to the
        bridge's existing controller-surface setter call.
        """

        if self._last_bridge_submission_evidence is None:
            return None
        return dict(self._last_bridge_submission_evidence)

    def _clear_pending_submission_binding(self) -> None:
        self._pending_submission_binding = None

    def _bind_accepted_validator_receipt(self, command: Any) -> bool:
        """Prepare a one-shot private submission binding for ``command``.

        A valid accepted result must preserve the exact command, use the
        current immutable validator configuration, and have no pre-existing
        downstream flag.  This helper does not mutate the controller surface.
        """

        receipt = self._last_validator_receipt
        if receipt is None or not receipt.accepted:
            return False
        if (
            receipt.requested_target != command
            or receipt.validated_target != command
            or receipt.target_mutated
            or receipt.downstream_submission_performed
            or receipt.config_fingerprint
            != self.pre_controller_validator.config.fingerprint()
        ):
            return False
        command_fingerprint = _ee_command_fingerprint(command)
        receipt_fingerprint = _json_sha256(receipt.payload())
        self._pending_submission_binding = {
            "schema": "g2_policy_branch_p0_bridge_private_submission_binding_v1",
            "command_fingerprint": command_fingerprint,
            "validator_pre_submit_receipt_fingerprint": receipt_fingerprint,
            "validator_config_fingerprint": receipt.config_fingerprint,
            "submission_epoch_expected": self._submission_epoch + 1,
        }
        return True

    def has_private_submission_evidence_for(self, command: Any) -> bool:
        """True only if a bridge-owned write proves this exact command.

        The public ``downstream_submission_performed`` receipt flag is
        intentionally not used as the authority for this check.
        """

        evidence = self._last_bridge_submission_evidence
        return bool(
            evidence
            and evidence.get("bridge_private_evidence") is True
            and evidence.get("exact_validated_command_match") is True
            and evidence.get("submitted_command_fingerprint")
            == _ee_command_fingerprint(command)
            and evidence.get("controller_target_setter_calls_after")
            == self._ee_submission_count
        )

    def downstream_activity_snapshot(self) -> dict[str, int]:
        """Counters proving whether a candidate crossed the controller boundary."""

        return {
            # These names are evidence-contract terms, not new controller
            # hooks.  The bridge is the only writer to the existing 8-D
            # controller surface, and only ``step`` enters the existing IK
            # execution path in this diagnostic.
            "controller_target_setter_calls": self._ee_submission_count,
            "ik_execution_command_calls": self._env_step_count,
            "physical_command_calls": 0,
            "env_step_calls": self._env_step_count,
            "gripper_submission_calls": self._gripper_submission_count,
            "safe_hold_calls": self._safe_hold_count,
        }

    def p0_a_configuration_receipt(self) -> dict[str, Any]:
        """P0-A-only controller/validator configuration evidence.

        This is intentionally distinct from the P0-C RGB-D observation
        binding fingerprint.  It records only the current P0 action/router/
        existing-controller surface and the fail-closed validator contract.
        """

        payload = {
            "schema": "g2_policy_branch_p0_a_controller_configuration_v1",
            "phase": "P0_A",
            "policy_action_schema": "g2_ee_xyz_gripper_v1",
            "controller_surface_schema": "g2_existing_ee_controller_surface_v1",
            "controller_surface_dimension": int(self.env.action_manager.total_action_dim),
            "existing_controller_scale": list(self.controller_scale),
            "control_frame": self.scale.control_frame.value,
            "orientation_residual_exact_zero": True,
            "redundancy_elbow_exact_zero": True,
            "pre_controller_validator_config": self.pre_controller_validator.config.payload(),
            "pre_controller_validator_config_fingerprint": (
                self.pre_controller_validator.config.fingerprint()
            ),
            "p0_c_runtime_observation_binding_fingerprint": "NOT_APPLICABLE_P0_A",
        }
        return {
            "payload": payload,
            "fingerprint": _json_sha256(payload),
        }

    def observed_abstract_intent(self):
        if bool(self.gripper_term.close_command_active[0].item()):
            return self.AbstractGripperIntent.CLOSE
        return self.AbstractGripperIntent.OPEN

    def set_safe_hold_surface(self) -> None:
        self._safe_hold_count += 1
        self._surface[:, :7] = 0.0
        self._surface[:, 7] = (
            -1.0
            if self.observed_abstract_intent() is self.AbstractGripperIntent.CLOSE
            else 1.0
        )

    # Implements CartesianSafetyBoundary.  The validator executes first and
    # never mutates a target.  No local workspace/IK rule is introduced here.
    def project(self, command):
        self._clear_pending_submission_binding()
        self._last_bridge_submission_evidence = None
        projection = self.pre_controller_validator.project(command)
        self._last_validator_receipt = self.pre_controller_validator.last_receipt
        if not projection.accepted:
            return projection
        if not self._bind_accepted_validator_receipt(command):
            return self.SafetyProjection(
                False,
                None,
                "P0_VALIDATOR_RECEIPT_BINDING_MISMATCH",
            )
        if not self.available:
            self._clear_pending_submission_binding()
            return self.SafetyProjection(False, None, "EXISTING_CONTROLLER_UNAVAILABLE")
        if self.reject_all_for_fault_injection:
            self._clear_pending_submission_binding()
            return self.SafetyProjection(False, None, "P0_INJECTED_SAFETY_REJECTION")
        if command.frame is not self.CartesianControlFrame.ROBOT_ROOT:
            self._clear_pending_submission_binding()
            return self.SafetyProjection(False, None, "P0_FRAME_NOT_ROBOT_ROOT")
        if command.rotation_rad != (0.0, 0.0, 0.0):
            self._clear_pending_submission_binding()
            return self.SafetyProjection(False, None, "P0_ORIENTATION_NONZERO")
        normalized = tuple(
            value / self.scale.translation_m_per_normalized
            for value in command.translation_m
        )
        if any(not math.isfinite(value) or abs(value) > 1.0 + 1.0e-8 for value in normalized):
            self._clear_pending_submission_binding()
            return self.SafetyProjection(False, None, "P0_CONTROLLER_SURFACE_RANGE_REJECTED")
        return self.SafetyProjection(True, command)

    # Implements ExistingEEController.  It writes only the pre-existing
    # Cartesian action-manager envelope, never an articulation target.
    def submit_ee_residual(self, command) -> None:
        if command.frame is not self.CartesianControlFrame.ROBOT_ROOT:
            raise self.PolicyActionContractError("P0 controller frame mismatch")
        if command.rotation_rad != (0.0, 0.0, 0.0):
            raise self.PolicyActionContractError("P0 controller orientation leak")
        pending = self._pending_submission_binding
        receipt = self._last_validator_receipt
        command_fingerprint = _ee_command_fingerprint(command)
        if (
            pending is None
            or pending.get("command_fingerprint") != command_fingerprint
            or pending.get("submission_epoch_expected") != self._submission_epoch + 1
            or receipt is None
            or not receipt.accepted
            or receipt.requested_target != command
            or receipt.validated_target != command
            or receipt.target_mutated
            or receipt.downstream_submission_performed
            or pending.get("validator_pre_submit_receipt_fingerprint")
            != _json_sha256(receipt.payload())
        ):
            self._clear_pending_submission_binding()
            raise self.PolicyActionContractError(
                "P0_VALIDATOR_RECEIPT_SUBMISSION_BINDING_MISMATCH"
            )
        values = tuple(
            value / self.scale.translation_m_per_normalized
            for value in command.translation_m
        )
        self._surface[:, :3] = self._surface.new_tensor(values)
        self._surface[:, 3:7] = 0.0  # exact RPY and redundancy-elbow invariant
        self._ee_submission_count += 1
        self._submission_epoch += 1
        self._last_bridge_submission_evidence = {
            "schema": "g2_policy_branch_p0_bridge_private_submission_evidence_v1",
            "bridge_private_evidence": True,
            "submission_epoch": self._submission_epoch,
            "validator_pre_submit_receipt_fingerprint": pending[
                "validator_pre_submit_receipt_fingerprint"
            ],
            "validated_command_fingerprint": pending["command_fingerprint"],
            "submitted_command_fingerprint": command_fingerprint,
            "exact_validated_command_match": True,
            "controller_target_setter_calls_after": self._ee_submission_count,
        }
        # This public receipt is updated only after the bridge-owned evidence
        # exists.  Downstream success is still never inferred from the public
        # flag alone; see ``has_private_submission_evidence_for`` above.
        self._last_validator_receipt = receipt.with_downstream_submission()
        self._clear_pending_submission_binding()

    # Implements ExistingAbstractGripperController.  The feedback derives
    # solely from the existing binary action term's abstract command state.
    def submit_gripper_intent(self, intent):
        self._gripper_submission_count += 1
        if not self.available:
            return self.GripperSubmission(
                False,
                self.observed_abstract_intent(),
                "EXISTING_CONTROLLER_UNAVAILABLE",
            )
        if (
            intent is self.AbstractGripperIntent.CLOSE
            and bool(self.gripper_term.reset_open_hold_active[0].item())
        ):
            self._surface[:, 7] = 1.0
            return self.GripperSubmission(
                False,
                self.AbstractGripperIntent.OPEN,
                "EXISTING_RESET_OPEN_HOLD_ACTIVE",
            )
        self._surface[:, 7] = (
            1.0 if intent is self.AbstractGripperIntent.OPEN else -1.0
        )
        return self.GripperSubmission(True, intent)

    def step(self) -> tuple[Any, Any, Any, Any, list[str]]:
        self._env_step_count += 1
        observation, reward, terminated, truncated, info = self.env.step(self._surface)
        del reward, info
        active = [
            name
            for name in self.env.termination_manager.active_terms
            if bool(self.env.termination_manager.get_term(name)[0].item())
        ]
        return observation, terminated, truncated, self.observed_abstract_intent(), active


def _disable_p0_task_reward_and_curriculum_terms(cfg: Any) -> dict[str, Any]:
    """Remove task scoring/curricula that are outside P0's controller scope.

    The existing action manager, collision/fixed-torso terminations, scene, and
    controller configuration remain untouched.  This prevents a lift-task
    reward dependency from running during a controller-semantic attestation,
    without creating a new safety/controller authority.
    """

    disabled: dict[str, Any] = {}
    for manager_name in ("rewards", "curriculum"):
        manager_cfg = getattr(cfg, manager_name, None)
        if manager_cfg is None:
            disabled[manager_name] = []
            continue
        if not is_dataclass(manager_cfg) and not hasattr(manager_cfg, "__dict__"):
            raise RuntimeError(f"P0_UNREADABLE_{manager_name.upper()}_CONFIG")
        # G2 adds several lift terms dynamically in ``__post_init__``.  A
        # dataclass-field-only traversal misses those terms, so use the union
        # of declared fields and live config attributes.
        candidate_names = {
            config_field.name for config_field in fields(manager_cfg)
        } if is_dataclass(manager_cfg) else set()
        candidate_names.update(
            name for name in vars(manager_cfg) if not name.startswith("_")
        )
        names: list[str] = []
        for name in sorted(candidate_names):
            if getattr(manager_cfg, name) is not None:
                names.append(name)
        # Use an empty mapping rather than ``None``.  ManagerBase registers a
        # physics-ready callback even for a falsey config; that callback can
        # safely iterate an empty mapping but cannot inspect ``None.__dict__``.
        # Replacing the whole manager config prevents any lift-task term from
        # resolving during P0 without re-enabling a zero-weight reward path.
        setattr(cfg, manager_name, {})
        disabled[manager_name] = names
    return disabled


def _make_env(phase: str):
    from isaaclab.envs import ManagerBasedRLEnv

    collision_termination_restored = False
    if phase == "P0_C":
        from geniesim.rl.isaaclab.g2_redundancy_teleop_env_cfg import (
            G2RedundancyTeleopEnvCfg,
        )

        cfg = G2RedundancyTeleopEnvCfg()
        # The keyboard collection config deliberately delegates collision-row
        # ownership to its recorder and therefore disables this term.  P0-C is
        # not a recorder: it must keep the exact existing sandbox collision
        # termination instead of silently observing through an unsafe reset.
        if cfg.terminations.forbidden_collision is None:
            from isaaclab.managers import TerminationTermCfg as DoneTerm

            from geniesim.rl.isaaclab import g2_lift_task_mdp

            cfg.terminations.forbidden_collision = DoneTerm(
                func=g2_lift_task_mdp.forbidden_collision,
                params={"force_threshold_n": 1.0e-6},
            )
            collision_termination_restored = True
    else:
        from geniesim.rl.isaaclab.g2_redundancy_teleop_env_cfg import (
            G2RedundancyControlEnvCfg,
        )

        cfg = G2RedundancyControlEnvCfg()
    # P0 observes the post-command controller result.  These task outcomes
    # would auto-reset before it can be recorded and are not P0 criteria.
    cfg.terminations.time_out = None
    cfg.terminations.object_dropping = None
    cfg.terminations.object_reached_goal = None
    cfg.terminations.cube_left_table_persistently = None
    disabled_task_terms = _disable_p0_task_reward_and_curriculum_terms(cfg)
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    cfg.commands.object_pose.debug_vis = False
    env = ManagerBasedRLEnv(cfg=cfg)
    active_rewards = tuple(env.reward_manager.active_terms)
    active_curriculum = tuple(env.curriculum_manager.active_terms)
    if active_rewards or active_curriculum:
        raise RuntimeError(
            "P0_TASK_MANAGER_TERMS_REMAIN_ACTIVE:"
            + repr({"rewards": active_rewards, "curriculum": active_curriculum})
        )
    active_terminations = set(env.termination_manager.active_terms)
    required_terminations = {"forbidden_collision", "fixed_torso_drift"}
    if active_terminations != required_terminations:
        raise RuntimeError(
            "P0_EXISTING_SAFETY_TERMINATION_ALLOWLIST_MISMATCH:"
            + repr(
                {
                    "expected": sorted(required_terminations),
                    "actual": sorted(active_terminations),
                }
            )
        )
    disabled_task_terms["active_reward_terms_after_build"] = list(active_rewards)
    disabled_task_terms["active_curriculum_terms_after_build"] = list(active_curriculum)
    disabled_task_terms["active_termination_terms_after_build"] = sorted(active_terminations)
    disabled_task_terms["collision_termination_restored_from_existing_sandbox_cfg"] = [
        str(collision_termination_restored)
    ]
    return env, disabled_task_terms


def _pose_snapshot(env: Any) -> dict[str, Any]:
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    position_root, quaternion_root, position_world, quaternion_world = (
        _world_ee_pose_to_root(robot, ee_frame)
    )
    arm_term = env.action_manager._terms["arm_action"]
    if not hasattr(arm_term, "active_translation_axis") or not hasattr(
        arm_term, "ee_pose_target_active"
    ):
        raise RuntimeError("P0_EXISTING_REDUNDANCY_TARGET_ORIGIN_TELEMETRY_MISSING")
    arm_target_root = _tensor(arm_term.ee_desired_position)
    return {
        "measured_root_position_m": position_root.clone(),
        "measured_root_quaternion_xyzw": quaternion_root.clone(),
        "measured_world_position_m": position_world.clone(),
        "measured_world_quaternion_xyzw": quaternion_world.clone(),
        "controller_target_root_position_m": arm_target_root.clone(),
        "controller_active_translation_axis": _tensor(
            arm_term.active_translation_axis
        ).clone(),
        "controller_pose_target_active": _tensor(
            arm_term.ee_pose_target_active
        ).clone(),
    }


def _existing_controller_target_formula(
    before: dict[str, Any],
    route: Any,
):
    """Predict the published controller endpoint for a single-axis P0 pulse.

    The approved redundancy controller deliberately re-bases an explicit new
    axis request on its *measured* EE state, while a same-axis repeat retains
    the prior desired endpoint.  Consequently, ``target_after-target_before``
    can include a small pre-existing tracking residual on uncommanded axes.
    That reconciliation is controller-state behavior, not an extra 4-D policy
    command.  P0 therefore compares the target to this source-defined origin
    formula and reports the raw target delta separately.
    """

    import torch

    if route.ee_command is None:
        raise RuntimeError("P0_ACCEPTED_ROUTE_MISSING_EE_COMMAND")
    command = before["measured_root_position_m"].new_tensor(
        route.ee_command.translation_m
    ).view(1, 3)
    requested = torch.abs(command) > 1.0e-12
    if int(requested.sum(dim=-1).item()) != 1:
        raise RuntimeError("P0_TARGET_FORMULA_REQUIRES_SINGLE_AXIS_PULSE")
    requested_axis = int(torch.argmax(requested.to(torch.int64), dim=-1).item())
    origin = before["measured_root_position_m"].clone()
    prior_target_active = bool(before["controller_pose_target_active"][0].item())
    prior_axis = int(before["controller_active_translation_axis"][0].item())
    same_axis_repeat = prior_target_active and prior_axis == requested_axis
    if same_axis_repeat:
        origin[:, requested_axis] = before["controller_target_root_position_m"][
            :, requested_axis
        ]
    return {
        "requested_axis": requested_axis,
        "controller_origin_root_m": origin,
        "expected_controller_target_root_m": origin + command,
        "router_limiter_delta_root_m": command,
        "same_axis_repeat": same_axis_repeat,
        "target_reconciliation_residual_root_m": (
            before["measured_root_position_m"]
            - before["controller_target_root_position_m"]
        ),
    }


def _settle_open(env: Any, bridge: _ExistingControllerP0Bridge, router: Any) -> dict[str, Any]:
    """Use the existing reset-open settle, not a policy-issued close/open path."""

    submission = router.reset()
    before = int(bridge.gripper_term._g2_open_hold_remaining[0].item())
    iterations = 0
    while bool(bridge.gripper_term.reset_open_hold_active[0].item()):
        _, terminated, truncated, observed, active = bridge.step()
        iterations += 1
        router.synchronize_gripper_intent(observed)
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            raise RuntimeError("P0_RESET_OPEN_SETTLE_TERMINATED:" + ",".join(active))
        if iterations > 80:
            raise RuntimeError("P0_RESET_OPEN_SETTLE_TIMEOUT")
    return {
        "router_reset_submission": submission,
        "existing_open_hold_initial_physics_steps": before,
        "existing_open_hold_policy_steps": iterations,
        "observed_after_settle": bridge.observed_abstract_intent(),
    }


def _execute(
    *,
    env: Any,
    bridge: _ExistingControllerP0Bridge,
    router: Any,
    previous_action: Any,
    label: str,
    action: Any,
) -> _P0ExecutionRecord:
    before = _pose_snapshot(env)
    previous_before = previous_action.value.values
    downstream_before = bridge.downstream_activity_snapshot()
    route = router.route(action)
    validator_receipt = bridge.last_validator_receipt
    # A safety-rejected candidate must remain entirely upstream of the
    # existing controller.  In particular, do not substitute a safe-hold
    # command and do not advance physics just to make rejection look benign:
    # that would defeat the evidence that no setter/IK/controller command was
    # reached for the rejected candidate.
    if route.ee_command is None:
        terminated = truncated = None
        observed = bridge.observed_abstract_intent()
        active: list[str] = []
    else:
        _, terminated, truncated, observed, active = bridge.step()
        router.synchronize_gripper_intent(observed)
    if route.accepted:
        previous_action.record_accepted(action)
    previous_after = previous_action.value.values
    after = _pose_snapshot(env)
    downstream_after = bridge.downstream_activity_snapshot()
    orientation_drift = _orientation_distance_rad(
        before["measured_root_quaternion_xyzw"],
        after["measured_root_quaternion_xyzw"],
    )
    return _P0ExecutionRecord(
        label=label,
        requested_action=action,
        previous_action_before=previous_before,
        previous_action_after=previous_after,
        route=route,
        controller_surface_action_8d=bridge.controller_surface,
        target_root_before_m=before["controller_target_root_position_m"],
        target_root_after_m=after["controller_target_root_position_m"],
        measured_root_before_m=before["measured_root_position_m"],
        measured_root_after_m=after["measured_root_position_m"],
        measured_world_before_m=before["measured_world_position_m"],
        measured_world_after_m=after["measured_world_position_m"],
        orientation_drift_rad=orientation_drift,
        observed_gripper_intent=observed.value,
        terminated=False if terminated is None else bool(terminated[0].item()),
        truncated=False if truncated is None else bool(truncated[0].item()),
        active_terminations=active,
        validator_receipt=validator_receipt,
        downstream_activity_before=downstream_before,
        downstream_activity_after=downstream_after,
    )


def _hold(
    *,
    env: Any,
    bridge: _ExistingControllerP0Bridge,
    router: Any,
    previous_action: Any,
    count: int,
    label: str,
) -> list[_P0ExecutionRecord]:
    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import HighLevelPolicyAction

    records = []
    # 0.5 is deliberately inside the hysteresis band and preserves the
    # current abstract intent without creating a new OPEN/CLOSE transition.
    action = HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.5))
    for index in range(count):
        record = _execute(
            env=env,
            bridge=bridge,
            router=router,
            previous_action=previous_action,
            label=f"{label}_{index}",
            action=action,
        )
        records.append(record)
        if record.terminated or record.truncated:
            break
    return records


def _record_payload(record: _P0ExecutionRecord) -> dict[str, Any]:
    route = record.route
    return {
        "label": record.label,
        "requested_policy_action": record.requested_action.values,
        "previous_policy_action_before": record.previous_action_before,
        "previous_policy_action_after": record.previous_action_after,
        "requested_orientation_residual_rad": (0.0, 0.0, 0.0),
        "route_accepted": route.accepted,
        "route_reason": route.reason,
        "router_ee_command": route.ee_command,
        "router_gripper_intent": route.gripper_intent.value,
        "router_requested_gripper_intent": route.requested_gripper_intent.value,
        "gripper_command_emitted": route.gripper_command_emitted,
        "gripper_submission": route.gripper_submission,
        "controller_surface_action_8d": record.controller_surface_action_8d,
        "controller_target_root_before_m": record.target_root_before_m,
        "controller_target_root_after_m": record.target_root_after_m,
        "measured_root_before_m": record.measured_root_before_m,
        "measured_root_after_m": record.measured_root_after_m,
        "measured_world_before_m": record.measured_world_before_m,
        "measured_world_after_m": record.measured_world_after_m,
        "orientation_drift_rad": record.orientation_drift_rad,
        "observed_controller_abstract_gripper_intent": record.observed_gripper_intent,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "active_terminations": record.active_terminations,
        "pre_controller_validator_receipt": record.validator_receipt,
        "downstream_activity_before": record.downstream_activity_before,
        "downstream_activity_after": record.downstream_activity_after,
    }


def _run_negative_tests(
    bridge: _ExistingControllerP0Bridge, previous_action: Any
) -> dict[str, Any]:
    """Test rejection without allowing malformed data to reach env.step()."""

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        HighLevelPolicyAction,
        PolicyCommandRouter,
    )

    invalid_cases: dict[str, bool] = {}
    for name, values in {
        "wrong_dimension": (0.0, 0.0, 0.0),
        "nan": (0.0, 0.0, 0.0, float("nan")),
        "inf": (0.0, 0.0, 0.0, float("inf")),
        "out_of_range": (1.01, 0.0, 0.0, 0.5),
    }.items():
        try:
            HighLevelPolicyAction.from_sequence(values)
        except Exception:
            invalid_cases[name] = True
        else:
            invalid_cases[name] = False

    prior_surface = bridge.controller_surface
    prior_previous_action = previous_action.value.values
    bridge.reject_all_for_fault_injection = True
    fault_router = PolicyCommandRouter(
        safety_boundary=bridge,
        ee_controller=bridge,
        gripper_controller=bridge,
    )
    rejected = fault_router.route(
        HighLevelPolicyAction.from_sequence((0.1, 0.0, 0.0, 0.9))
    )
    bridge.reject_all_for_fault_injection = False
    injected_rejection_pass = (
        not rejected.accepted
        and rejected.ee_command is None
        and not rejected.gripper_command_emitted
        and torch_equal(prior_surface, bridge.controller_surface)
    )

    bridge.available = False
    unavailable_router = PolicyCommandRouter(
        safety_boundary=bridge,
        ee_controller=bridge,
        gripper_controller=bridge,
    )
    unavailable = unavailable_router.route(
        HighLevelPolicyAction.from_sequence((0.1, 0.0, 0.0, 0.9))
    )
    bridge.available = True
    unavailable_pass = (
        not unavailable.accepted
        and unavailable.ee_command is None
        and unavailable.gripper_intent is AbstractGripperIntent.OPEN
    )
    rejected_action_ring_pass = (
        previous_action.value.values == prior_previous_action
    )
    return {
        "invalid_action_constructor_rejected": invalid_cases,
        "injected_router_safety_rejection": {
            "pass": injected_rejection_pass,
            "route_reason": rejected.reason,
            "env_step_called": False,
        },
        "controller_unavailable_fault_injection": {
            "pass": unavailable_pass,
            "route_reason": unavailable.reason,
            "env_step_called": False,
        },
        "rejected_action_does_not_advance_previous_action": {
            "pass": rejected_action_ring_pass,
            "previous_before": prior_previous_action,
            "previous_after": previous_action.value.values,
        },
        "impossible_workspace_or_ik_request": {
            "status": "NOT_APPLICABLE_NO_EXISTING_PUBLIC_REJECTION_AUTHORITY",
            "reason": (
                "Existing DLS has target/joint limiting but exposes no approved public "
                "workspace/IK rejection result for PolicyCommandRouter."
            ),
        },
        "stale_observation": {
            "status": "DEFERRED_TO_P0_C_RUNTIME_SENSOR_CONTRACT",
        },
    }


def torch_equal(first: Any, second: Any) -> bool:
    import torch

    return bool(torch.equal(first, second))


def _legacy_run_p0_a_motion_attestation_disabled() -> dict[str, Any]:
    import torch

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        HighLevelPolicyAction,
        PolicyCommandRouter,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
        PreviousAcceptedPolicyAction,
    )

    env, disabled_task_terms = _make_env("P0_A")
    try:
        observation, _ = env.reset(seed=42)
        del observation
        bridge = _ExistingControllerP0Bridge(env)
        router = PolicyCommandRouter(
            safety_boundary=bridge,
            ee_controller=bridge,
            gripper_controller=bridge,
        )
        previous = PreviousAcceptedPolicyAction()
        reset_evidence = _settle_open(env, bridge, router)
        previous.reset()
        initial_observed = bridge.observed_abstract_intent()
        initial_latch_pass = (
            reset_evidence["router_reset_submission"].accepted
            and initial_observed is AbstractGripperIntent.OPEN
            and router.gripper_intent is AbstractGripperIntent.OPEN
            and previous.value.values == (0.0, 0.0, 0.0, 0.0)
        )

        records: list[_P0ExecutionRecord] = []
        records.append(
            _execute(
                env=env,
                bridge=bridge,
                router=router,
                previous_action=previous,
                label="ZERO_HOLD",
                action=HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, 0.5)),
            )
        )
        records.extend(
            _hold(
                env=env,
                bridge=bridge,
                router=router,
                previous_action=previous,
                count=4,
                label="ZERO_HOLD",
            )
        )

        axis_results: list[dict[str, Any]] = []
        pulse_normalized = 0.20
        settle_steps = 40
        for axis, axis_name in enumerate(("X", "Y", "Z")):
            requested = [0.0, 0.0, 0.0, 0.5]
            requested[axis] = pulse_normalized
            before = _pose_snapshot(env)
            pulse = _execute(
                env=env,
                bridge=bridge,
                router=router,
                previous_action=previous,
                label=f"PLUS_{axis_name}_PULSE",
                action=HighLevelPolicyAction.from_sequence(requested),
            )
            records.append(pulse)
            records.extend(
                _hold(
                    env=env,
                    bridge=bridge,
                    router=router,
                    previous_action=previous,
                    count=settle_steps,
                    label=f"PLUS_{axis_name}_HOLD",
                )
            )
            after = _pose_snapshot(env)
            target_formula = _existing_controller_target_formula(before, pulse.route)
            target_delta = after["controller_target_root_position_m"] - before[
                "controller_target_root_position_m"
            ]
            measured_delta = after["measured_root_position_m"] - before[
                "measured_root_position_m"
            ]
            expected = pulse_normalized * bridge.scale.translation_m_per_normalized
            router_delta = target_formula["router_limiter_delta_root_m"]
            router_axis_error = float(abs(router_delta[0, axis].item() - expected))
            controller_target_formula_error = float(
                torch.max(
                    torch.abs(
                        after["controller_target_root_position_m"]
                        - target_formula["expected_controller_target_root_m"]
                    )
                ).item()
            )
            raw_target_increment_axis_error = float(
                abs(target_delta[0, axis].item() - expected)
            )
            orthogonal_target = float(
                torch.max(torch.abs(torch.cat((target_delta[:, :axis], target_delta[:, axis + 1 :]), dim=-1))).item()
                if axis < 2
                else torch.max(torch.abs(target_delta[:, :2])).item()
            )
            # A root-frame target is the semantic authority.  Measured sign is
            # assessed after settling with a small nonzero floor, independent
            # of a task/grasp success criterion.
            measured_axis = float(measured_delta[0, axis].item())
            measured_orthogonal = float(
                torch.linalg.vector_norm(
                    torch.cat((measured_delta[:, :axis], measured_delta[:, axis + 1 :]), dim=-1),
                    dim=-1,
                )[0].item()
            )
            orientation_drift = _orientation_distance_rad(
                before["measured_root_quaternion_xyzw"],
                after["measured_root_quaternion_xyzw"],
            )
            axis_results.append(
                {
                    "axis": axis_name,
                    "requested_normalized": pulse_normalized,
                    "requested_delta_m": expected,
                    "resolved_frame": "robot_root",
                    "limiter_after_router_delta_m": router_delta,
                    "controller_target_delta_m": target_delta,
                    "controller_target_origin_root_m": target_formula[
                        "controller_origin_root_m"
                    ],
                    "expected_controller_target_root_m": target_formula[
                        "expected_controller_target_root_m"
                    ],
                    "controller_target_formula_error_m": controller_target_formula_error,
                    "controller_target_same_axis_repeat": target_formula[
                        "same_axis_repeat"
                    ],
                    "target_reconciliation_residual_root_m": target_formula[
                        "target_reconciliation_residual_root_m"
                    ],
                    "measured_root_delta_m": measured_delta,
                    "router_scale_error_m": router_axis_error,
                    "raw_target_increment_axis_error_m": raw_target_increment_axis_error,
                    "raw_target_increment_orthogonal_delta_m": orthogonal_target,
                    "measured_axis_delta_m": measured_axis,
                    "measured_orthogonal_leakage_m": measured_orthogonal,
                    "orientation_drift_rad": orientation_drift,
                    "target_sign_pass": float(router_delta[0, axis].item()) > 0.0,
                    "measured_sign_pass": measured_axis > 0.0005,
                    "router_scale_pass": router_axis_error <= 1.0e-9,
                    "controller_target_formula_pass": controller_target_formula_error <= 1.0e-6,
                    "scale_pass": (
                        router_axis_error <= 1.0e-9
                        and controller_target_formula_error <= 1.0e-6
                    ),
                    "orientation_pass": orientation_drift <= math.radians(1.0),
                }
            )
            returning = [0.0, 0.0, 0.0, 0.5]
            returning[axis] = -pulse_normalized
            records.append(
                _execute(
                    env=env,
                    bridge=bridge,
                    router=router,
                    previous_action=previous,
                    label=f"RETURN_{axis_name}_PULSE",
                    action=HighLevelPolicyAction.from_sequence(returning),
                )
            )
            records.extend(
                _hold(
                    env=env,
                    bridge=bridge,
                    router=router,
                    previous_action=previous,
                    count=settle_steps,
                    label=f"RETURN_{axis_name}_HOLD",
                )
            )

        # Actual controller-backed hysteresis path.  The g values in the
        # middle band must preserve state and emit no new gripper transition.
        hysteresis_sequence = (
            ("MIDDLE_OPEN_049", 0.49, "OPEN"),
            ("MIDDLE_OPEN_051", 0.51, "OPEN"),
            ("CLOSE_071", 0.71, "CLOSE"),
            ("MIDDLE_CLOSE_049", 0.49, "CLOSE"),
            ("MIDDLE_CLOSE_051", 0.51, "CLOSE"),
            ("MIDDLE_CLOSE_048", 0.48, "CLOSE"),
            ("OPEN_029", 0.29, "OPEN"),
        )
        hysteresis_records: list[dict[str, Any]] = []
        for label, probability, expected_intent in hysteresis_sequence:
            record = _execute(
                env=env,
                bridge=bridge,
                router=router,
                previous_action=previous,
                label=label,
                action=HighLevelPolicyAction.from_sequence((0.0, 0.0, 0.0, probability)),
            )
            records.append(record)
            hysteresis_records.append(
                {
                    "label": label,
                    "g": probability,
                    "expected_intent": expected_intent,
                    "route_intent": record.route.gripper_intent.value,
                    "observed_controller_intent": record.observed_gripper_intent,
                    "gripper_command_emitted": record.route.gripper_command_emitted,
                    "route_accepted": record.route.accepted,
                }
            )
        hysteresis_pass = all(
            item["route_accepted"]
            and item["route_intent"] == item["expected_intent"]
            and item["observed_controller_intent"] == item["expected_intent"]
            for item in hysteresis_records
        )
        # The two middle-band pairs must not emit an OPEN/CLOSE command.
        hysteresis_pass = hysteresis_pass and not any(
            item["gripper_command_emitted"]
            for item in hysteresis_records
            if item["label"].startswith("MIDDLE")
        )

        negative = _run_negative_tests(bridge, previous)
        no_termination = not any(item.terminated or item.truncated for item in records)
        all_orientation_zero = all(
            tuple(float(value) for value in item.controller_surface_action_8d[0, 3:7].tolist())
            == (0.0, 0.0, 0.0, 0.0)
            for item in records
        )
        previous_action_trace_pass = all(
            (
                record.previous_action_after == record.requested_action.values
                if record.route.accepted
                else record.previous_action_after == record.previous_action_before
            )
            for record in records
        )
        previous_action_pass = (
            previous.value.values == (0.0, 0.0, 0.0, 0.29)
            and previous_action_trace_pass
            and bool(negative["rejected_action_does_not_advance_previous_action"]["pass"])
        )
        # This is deliberately a FAIL rather than a newly invented workspace
        # clamp.  The existing DLS limiter remains downstream and is recorded,
        # but it has no public rejection result required by the P0 contract.
        gates = {
            "EE_FRAME_SEMANTICS": _gate(
                "PASS" if all(item["resolved_frame"] == "robot_root" for item in axis_results) else "FAIL",
                evidence={
                    "controller_frame": "robot_root",
                    "measurement_frame": "world_to_robot_root_explicit_transform",
                    "axis_results": axis_results,
                },
            ),
            "EE_SIGN_SEMANTICS": _gate(
                "PASS" if all(item["target_sign_pass"] and item["measured_sign_pass"] for item in axis_results) else "FAIL",
                evidence=axis_results,
            ),
            "EE_SCALE_SEMANTICS": _gate(
                "PASS" if all(item["scale_pass"] for item in axis_results) else "FAIL",
                evidence={
                    "router_translation_scale_m_per_normalized": bridge.scale.translation_m_per_normalized,
                    "existing_controller_scale": bridge.controller_scale,
                    "axis_results": axis_results,
                },
            ),
            "ORIENTATION_ZERO_INVARIANT": _gate(
                "PASS" if all_orientation_zero and all(item["orientation_pass"] for item in axis_results) else "FAIL",
                evidence={"axis_results": axis_results, "surface_rpy_elbow_exact_zero": all_orientation_zero},
            ),
            "GRIPPER_ABSTRACTION": _gate(
                "PASS",
                evidence={
                    "policy_gripper_surface": "g_in_0_1_only",
                    "router_intent": "abstract_OPEN_CLOSE_only",
                    "controller_surface": "existing_8d_cartesian_plus_binary_gripper",
                    "individual_joint_target_api_called": False,
                    "torque_current_can_exposed": False,
                },
            ),
            "HYSTERESIS": _gate("PASS" if hysteresis_pass else "FAIL", evidence=hysteresis_records),
            "INITIAL_LATCH": _gate("PASS" if initial_latch_pass else "FAIL", evidence=reset_evidence),
            "PREVIOUS_ACTION_SEMANTICS": _gate(
                "PASS" if previous_action_pass else "FAIL",
                evidence={
                    "authority": "last_accepted_branch_policy_action_lag1_v1",
                    "initial_previous_action": (0.0, 0.0, 0.0, 0.0),
                    "final_previous_action": previous.value.values,
                    "accepted_action_trace_pass": previous_action_trace_pass,
                    "rejected_action_does_not_advance": negative[
                        "rejected_action_does_not_advance_previous_action"
                    ],
                },
            ),
            "RESET_SEMANTICS": _gate("PASS" if initial_latch_pass else "FAIL", evidence=reset_evidence),
            "SAFETY_BOUNDARY": _gate(
                "FAIL",
                reason="NO_EXISTING_PUBLIC_WORKSPACE_IK_REJECTION_AUTHORITY",
                evidence={
                    "existing_downstream_dls_limiter": "present",
                    "existing_public_router_projection": "absent_before_P0",
                    "new_workspace_or_ik_authority_created": False,
                    "impossible_request_test": negative["impossible_workspace_or_ik_request"],
                },
            ),
            "NEGATIVE_INPUT_REJECTION": _gate(
                "PASS"
                if all(negative["invalid_action_constructor_rejected"].values())
                and negative["injected_router_safety_rejection"]["pass"]
                and negative["controller_unavailable_fault_injection"]["pass"]
                else "FAIL",
                evidence=negative,
            ),
            "RUNTIME_TERMINATION_FREE": _gate(
                "PASS" if no_termination else "FAIL",
                evidence=[_record_payload(item) for item in records if item.terminated or item.truncated],
            ),
        }
        required = (
            "EE_FRAME_SEMANTICS",
            "EE_SIGN_SEMANTICS",
            "EE_SCALE_SEMANTICS",
            "ORIENTATION_ZERO_INVARIANT",
            "GRIPPER_ABSTRACTION",
            "HYSTERESIS",
            "INITIAL_LATCH",
            "PREVIOUS_ACTION_SEMANTICS",
            "RESET_SEMANTICS",
            "SAFETY_BOUNDARY",
            "NEGATIVE_INPUT_REJECTION",
        )
        semantic_pass = all(gates[name]["status"] == "PASS" for name in required)
        return {
            "phase": "P0_A",
            "p0_semantic_verdict": "PASS" if semantic_pass else "FAIL",
            "gates": gates,
            "axis_results": axis_results,
            "hysteresis_records": hysteresis_records,
            "runtime_records": [_record_payload(record) for record in records],
            "execution_scope": {
                "learned_policy_loaded": False,
                "training_started": False,
                "m2_h47_passive_chain_metrics_collected": False,
                "joint_level_gripper_control": False,
                "raw_low_level_actuator_command": False,
                "controller_surface": "existing_8d_cartesian_binary_gripper_only",
                "disabled_task_reward_and_curriculum_terms": disabled_task_terms,
                "retained_termination_terms": (
                    "forbidden_collision",
                    "fixed_torso_drift",
                ),
            },
        }
    finally:
        env.close()


# The original P0-A motion/hysteresis implementation above is retained only
# as historical source evidence.  It is deliberately not called: the new
# pre-controller validator is unbound until Track A identifies an existing,
# source-authoritative dry-run IK rejection API.  Running accepted motion by
# falling back to a diagnostic-local rule would falsely promote P0-A.
P0_A_TRACK_A_AUDIT_RELATIVE_PATH = (
    "output/policy_branch_p0/track_a_static_audit/"
    "TRACK_A_EXISTING_SAFETY_AUTHORITY_AUDIT.json"
)
P0_A_NO_DOWNSTREAM_COUNTERS = (
    "controller_target_setter_calls",
    "ik_execution_command_calls",
    "physical_command_calls",
    "env_step_calls",
    "gripper_submission_calls",
    "safe_hold_calls",
)


def _counter_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    """Compute a strict activity delta for one P0 safety probe."""

    required = set(P0_A_NO_DOWNSTREAM_COUNTERS)
    missing = (required - set(before)) | (required - set(after))
    if missing:
        raise RuntimeError("P0_ACTIVITY_COUNTER_MISSING:" + ",".join(sorted(missing)))
    return {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in P0_A_NO_DOWNSTREAM_COUNTERS
    }


def _no_downstream_activity(delta: dict[str, int]) -> bool:
    return all(int(delta.get(name, 0)) == 0 for name in P0_A_NO_DOWNSTREAM_COUNTERS)


def _receipt_payload(receipt: Any) -> dict[str, Any] | None:
    return None if receipt is None else _plain(receipt.payload())


def _static_semantic_regression_contract() -> dict[str, Any]:
    """Frozen semantics kept separate from live safety authorization."""

    return {
        "normalized_action": (0.20, 0.0, 0.0, 0.50),
        "translation_scale_m_per_normalized": 0.0225,
        "expected_delta_m": 0.0045,
        "frame": "robot_root",
        "orientation_residual_rad": (0.0, 0.0, 0.0),
        "redundancy_elbow_residual": 0.0,
        "rebase_authority": (
            "G2RedundancyDifferentialIKAction measured-EE origin on a new axis"
        ),
        "static_test": "tests/test_g2_teleop_dataset.py",
        "policy_contract_test": "tests/test_g2_high_level_policy_branch.py",
    }


def _p0_a_safety_authority_contract() -> dict[str, Any]:
    """Serialize the validator's deliberately narrow non-authority contract."""

    from geniesim.rl.isaaclab.g2_policy_branch.ee_safety_validator import (
        EESafetyValidatorConfig,
        P0SafetyRejectReason,
    )

    config = EESafetyValidatorConfig()
    return {
        "schema": "g2_policy_branch_p0_a_safety_authority_contract_v2",
        "status": "FAIL_CLOSED_UNTIL_EXISTING_PUBLIC_DRY_RUN_AUTHORITY_IS_BOUND",
        "implementation_module": (
            "source/geniesim/rl/isaaclab/g2_policy_branch/ee_safety_validator.py"
        ),
        "validator_schema": config.schema,
        "validator_scope": config.scope,
        "validator_config": config.payload(),
        "validator_config_fingerprint": config.fingerprint(),
        "policy_action_schema": "g2_ee_xyz_gripper_v1",
        "control_frame": "robot_root",
        "target_mutation_allowed": False,
        "arbitrary_workspace_rule_added": False,
        "default_dry_run_provider": "UnresolvedDryRunProvider",
        "default_valid_action_behavior": "REJECT_INTERNAL_VALIDATION_FAILURE",
        "requires_existing_public_rejection_receipt": True,
        "track_a_authority_audit": P0_A_TRACK_A_AUDIT_RELATIVE_PATH,
        "rejection_reasons": [
            reason.value
            for reason in P0SafetyRejectReason
            if reason is not P0SafetyRejectReason.ACCEPT
        ],
        "bridge_submission_proof": {
            "public_receipt_flag_sufficient": False,
            "requires_bridge_private_exact_command_evidence": True,
            "requires_requested_equals_validated_equals_submitted": True,
        },
    }


def _p0_a_static_contract_self_check() -> dict[str, Any]:
    """Run pure-Python validator contract checks without SimulationApp.

    This is intentionally a compact evidence check, not a substitute for the
    repository pytest suite.  Its output says exactly what ran so it cannot
    be mistaken for a live P0 or a source-authoritative IK result.
    """

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        CartesianControlFrame,
        EEResidualCommand,
        GripperSubmission,
        HighLevelPolicyAction,
        PolicyCommandRouter,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.ee_safety_validator import (
        DryRunIKResult,
        EESafetyValidator,
        P0SafetyRejectReason,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
        PreviousAcceptedPolicyAction,
    )

    class _StaticRejectedProvider:
        def __init__(self, reason: Any) -> None:
            self.reason = reason

        def evaluate_dry_run(self, target: Any) -> Any:
            del target
            return DryRunIKResult(
                feasible=False,
                reason=self.reason,
                authority_source_path="STATIC_SELF_CHECK_ONLY",
                authority_symbol="_StaticRejectedProvider",
                authority_id="STATIC_SELF_CHECK_ONLY",
                residual_threshold_authority="STATIC_SELF_CHECK_ONLY",
                joint_limits_satisfied=False,
            )

    class _StaticRaisingProvider:
        def evaluate_dry_run(self, target: Any) -> Any:
            del target
            raise RuntimeError("STATIC_SELF_CHECK_EXCEPTION")

    class _StaticFeasibleProvider:
        def evaluate_dry_run(self, target: Any) -> Any:
            del target
            return DryRunIKResult(
                feasible=True,
                reason=P0SafetyRejectReason.ACCEPT,
                authority_source_path="STATIC_SELF_CHECK_ONLY",
                authority_symbol="_StaticFeasibleProvider",
                authority_id="STATIC_SELF_CHECK_ONLY",
                solution_joint_position_rad=(0.0,),
                residual=0.0,
                residual_threshold_authority="STATIC_SELF_CHECK_ONLY",
                joint_limits_satisfied=True,
            )

    class _StaticEEController:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        def submit_ee_residual(self, command: Any) -> None:
            self.commands.append(command)

    class _StaticGripperController:
        def __init__(self) -> None:
            self.intents: list[Any] = []

        def submit_gripper_intent(self, intent: Any) -> Any:
            self.intents.append(intent)
            return GripperSubmission(True, intent)

    validator = EESafetyValidator()
    valid = validator.validate_policy_action((0.20, 0.0, 0.0, 0.50))
    nonfinite = validator.validate_policy_action((float("nan"), 0.0, 0.0, 0.50))
    infinite = validator.validate_policy_action((float("inf"), 0.0, 0.0, 0.50))
    out_of_range = validator.validate_policy_action((1.01, 0.0, 0.0, 0.50))
    orientation = validator.validate(
        EEResidualCommand(
            translation_m=(0.0, 0.0, 0.0),
            rotation_rad=(0.0, 1.0e-6, 0.0),
            frame=CartesianControlFrame.ROBOT_ROOT,
        )
    )
    ik_no_solution = EESafetyValidator(
        dry_run_provider=_StaticRejectedProvider(P0SafetyRejectReason.IK_NO_SOLUTION)
    ).validate_policy_action((0.20, 0.0, 0.0, 0.50))
    joint_limit = EESafetyValidator(
        dry_run_provider=_StaticRejectedProvider(
            P0SafetyRejectReason.JOINT_LIMIT_VIOLATION
        )
    ).validate_policy_action((0.20, 0.0, 0.0, 0.50))
    provider_exception = EESafetyValidator(
        dry_run_provider=_StaticRaisingProvider()
    ).validate_policy_action((0.20, 0.0, 0.0, 0.50))
    rejected_ee = _StaticEEController()
    rejected_gripper = _StaticGripperController()
    rejected_router = PolicyCommandRouter(
        safety_boundary=EESafetyValidator(
            dry_run_provider=_StaticRejectedProvider(P0SafetyRejectReason.IK_NO_SOLUTION)
        ),
        ee_controller=rejected_ee,
        gripper_controller=rejected_gripper,
    )
    suppressed = rejected_router.route(
        HighLevelPolicyAction.from_sequence((0.20, 0.0, 0.0, 0.90))
    )
    accepted_ee = _StaticEEController()
    accepted_gripper = _StaticGripperController()
    accepted_validator = EESafetyValidator(
        dry_run_provider=_StaticFeasibleProvider()
    )
    accepted_router = PolicyCommandRouter(
        safety_boundary=accepted_validator,
        ee_controller=accepted_ee,
        gripper_controller=accepted_gripper,
    )
    accepted_action = HighLevelPolicyAction.from_sequence((0.20, 0.0, 0.0, 0.90))
    reset_submission = accepted_router.reset()
    accepted_route = accepted_router.route(accepted_action)
    accepted_receipt = accepted_validator.last_receipt
    ring = PreviousAcceptedPolicyAction()
    accepted_previous = bool(
        accepted_receipt is not None
        and accepted_validator.record_previous_action_after_submission(
            receipt=accepted_receipt.with_downstream_submission(),
            action=accepted_action,
            recorder=ring,
        )
    )
    accepted_router.reset()
    ring.reset()
    checks = {
        "unbound_valid_action_rejects": (
            not valid.accepted
            and valid.reason is P0SafetyRejectReason.INTERNAL_VALIDATION_FAILURE
        ),
        "nan_rejects": (
            not nonfinite.accepted
            and nonfinite.reason is P0SafetyRejectReason.INVALID_INPUT
        ),
        "inf_rejects": (
            not infinite.accepted
            and infinite.reason is P0SafetyRejectReason.INVALID_INPUT
        ),
        "bound_rejects": (
            not out_of_range.accepted
            and out_of_range.reason is P0SafetyRejectReason.ACTION_BOUND_VIOLATION
        ),
        "nonzero_orientation_rejects": (
            not orientation.accepted
            and orientation.reason
            is P0SafetyRejectReason.ORIENTATION_CONTRACT_VIOLATION
        ),
        "no_workspace_rule": not validator.config.arbitrary_workspace_rule_added,
        "no_target_mutation": not validator.config.target_mutation_allowed,
        # These two fakes are intentionally static-label coverage only.  They
        # do not bind a production IK authority or make N5/N6 live-executed.
        "static_ik_no_solution_receipt": (
            not ik_no_solution.accepted
            and ik_no_solution.reason is P0SafetyRejectReason.IK_NO_SOLUTION
        ),
        "static_joint_limit_receipt": (
            not joint_limit.accepted
            and joint_limit.reason is P0SafetyRejectReason.JOINT_LIMIT_VIOLATION
        ),
        "provider_exception_fails_closed": (
            not provider_exception.accepted
            and provider_exception.reason
            is P0SafetyRejectReason.INTERNAL_VALIDATION_FAILURE
        ),
        "rejected_router_suppresses_ee_and_gripper": (
            not suppressed.accepted
            and rejected_ee.commands == []
            and rejected_gripper.intents == []
        ),
        "static_target_identity": (
            accepted_route.accepted
            and accepted_route.ee_command is not None
            and len(accepted_ee.commands) == 1
            and accepted_ee.commands[0] == accepted_route.ee_command
            and accepted_receipt is not None
            and accepted_receipt.requested_target == accepted_receipt.validated_target
            and accepted_receipt.validated_target == accepted_route.ee_command
            and not accepted_receipt.target_mutated
        ),
        "static_reset_previous_action": (
            reset_submission.accepted
            and reset_submission.effective_intent is AbstractGripperIntent.OPEN
            and accepted_previous
            and ring.value.values == (0.0, 0.0, 0.0, 0.0)
            and accepted_router.gripper_intent is AbstractGripperIntent.OPEN
        ),
    }
    return {
        "schema": "g2_policy_branch_p0_a_static_contract_checks_v1",
        "execution": "PURE_PYTHON_NO_SIMULATION_APP",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "external_pytest_command": (
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            "${GENIESIM_PYTHON:-python3} -m pytest -q "
            "tests/test_g2_ee_safety_validator.py "
            "tests/test_g2_policy_branch_p0_attestation_static.py "
            "tests/test_g2_high_level_policy_branch.py "
            "tests/test_g2_teleop_dataset.py"
        ),
        "external_pytest_status": "NOT_EXECUTED_BY_LIVE_CHILD",
        "full_suite_coverage_contract": {
            "valid_nan_inf_bound_orientation": "tests/test_g2_ee_safety_validator.py",
            "ik_joint_limit_exception_suppression": "tests/test_g2_ee_safety_validator.py",
            "accepted_exact_target_bridge_binding": (
                "tests/test_g2_policy_branch_p0_attestation_static.py::"
                "test_accepted_submission_requires_private_exact_command_binding"
            ),
            "reset_previous_action_hysteresis": "tests/test_g2_high_level_policy_branch.py",
            "scale_frame_rebase": "tests/test_g2_teleop_dataset.py",
        },
        "self_check_is_not_live_p0_evidence": True,
    }


def _p0_a_safety_design_markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# P0-A safety authority design",
            "",
            "This artifact is intentionally fail-closed.  It adds no Cartesian",
            "workspace box, IK residual threshold, joint-limit rule, target clamp,",
            "or collision policy.  A valid normalized action is accepted only if an",
            "existing, public, source-authoritative, side-effect-free dry-run IK",
            "authority returns an explicit feasible receipt.  No such authority is",
            "currently bound.",
            "",
            "A public receipt's `downstream_submission_performed` field is not proof",
            "of a controller write.  A bridge-private one-shot record must bind the",
            "exact requested, validated, and submitted EE command before the P0",
            "bridge treats a submission as evidenced.",
            "",
            "```json",
            json.dumps(contract, indent=2, sort_keys=True),
            "```",
            "",
        )
    )


def _write_p0_a_evidence_artifacts(output: Path, result: dict[str, Any]) -> None:
    """Write P0-A evidence before ``app.close`` using atomic replacements."""

    contract = _p0_a_safety_authority_contract()
    configuration = result.get("p0_a_configuration")
    _atomic_json(output / "p0_a_safety_authority_contract.json", contract)
    _atomic_json(output / "p0_a_safety_static_tests.json", _p0_a_static_contract_self_check())
    if isinstance(configuration, dict):
        _atomic_json(
            output / "config_fingerprint.json",
            {
                "schema": "g2_policy_branch_p0_a_configuration_fingerprint_v1",
                "configuration": configuration.get("payload"),
                "fingerprint": configuration.get("fingerprint"),
            },
        )
    _atomic_json(output / "p0_a_safety_live_result.json", result)
    _atomic_jsonl(
        output / "p0_a_safety_receipts.jsonl",
        [
            item
            for item in result.get("safety_receipts", [])
            if isinstance(item, dict)
        ],
    )
    _atomic_text(
        output / "P0_A_SAFETY_AUTHORITY_DESIGN.md",
        _p0_a_safety_design_markdown(contract),
    )


def _rejection_probe_payload(
    *,
    label: str,
    input_kind: str,
    raw_or_target: Any,
    expected_reason: str | None,
    receipt: Any,
    activity_before: dict[str, int],
    activity_after: dict[str, int],
    surface_unchanged: bool,
    previous_action_unchanged: bool,
    status: str = "EXECUTED",
    reason: str = "",
) -> dict[str, Any]:
    delta = _counter_delta(activity_before, activity_after)
    receipt_data = _receipt_payload(receipt)
    receipt_reason = None if receipt_data is None else receipt_data["reason"]
    receipt_rejected = bool(receipt_data is not None and not receipt_data["accepted"])
    expected_reason_match = (
        expected_reason is None or receipt_reason == expected_reason
    )
    return {
        "label": label,
        "status": status,
        "input_kind": input_kind,
        "raw_or_target": _plain(raw_or_target),
        "expected_reason": expected_reason,
        "reason": reason,
        "validator_receipt": receipt_data,
        "receipt_rejected": receipt_rejected,
        "expected_reason_match": expected_reason_match,
        "downstream_activity_before": activity_before,
        "downstream_activity_after": activity_after,
        "downstream_activity_delta": delta,
        "no_downstream_activity": _no_downstream_activity(delta),
        "controller_surface_unchanged": surface_unchanged,
        "previous_action_unchanged": previous_action_unchanged,
        "bridge_private_submission_evidence": None,
    }


def _run_p0_a() -> dict[str, Any]:
    """Run only P0-A's fail-closed pre-controller rejection attestation.

    N0--N4 are deliberately evaluated before the existing controller surface
    and must leave all downstream counters unchanged.  N5/N6 are not
    fabricated with a test IK provider: Track A has established that the
    repository exposes no production public authority capable of producing
    those live receipts.  The setup reset/open sequence is completed before
    the baseline counters are captured, so its legitimate existing calls are
    never attributed to a rejected safety candidate.
    """

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        CartesianControlFrame,
        EEResidualCommand,
        HighLevelPolicyAction,
        PolicyCommandRouter,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.ee_safety_validator import (
        P0SafetyRejectReason,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
        PreviousAcceptedPolicyAction,
    )

    env, disabled_task_terms = _make_env("P0_A")
    try:
        observation, _ = env.reset(seed=42)
        del observation
        bridge = _ExistingControllerP0Bridge(env)
        router = PolicyCommandRouter(
            safety_boundary=bridge,
            ee_controller=bridge,
            gripper_controller=bridge,
        )
        previous = PreviousAcceptedPolicyAction()
        reset_evidence = _settle_open(env, bridge, router)
        previous.reset()
        initial_observed = bridge.observed_abstract_intent()
        initial_latch_pass = (
            reset_evidence["router_reset_submission"].accepted
            and initial_observed is AbstractGripperIntent.OPEN
            and router.gripper_intent is AbstractGripperIntent.OPEN
            and previous.value.values == (0.0, 0.0, 0.0, 0.0)
        )

        # The reset/open-settle may itself take existing physics steps.  This
        # baseline is the start of the N0--N6 evidence window.
        baseline_activity = bridge.downstream_activity_snapshot()
        baseline_surface = bridge.controller_surface
        baseline_previous = previous.value.values
        probes: list[dict[str, Any]] = []

        # N0: syntactically valid +X command.  It must be rejected because
        # the default provider is deliberately unresolved, not because a new
        # workspace/IK authority has been created in this branch.
        n0_raw = (0.20, 0.0, 0.0, 0.50)
        before = bridge.downstream_activity_snapshot()
        surface_before = bridge.controller_surface
        previous_before = previous.value.values
        n0_route = router.route(HighLevelPolicyAction.from_sequence(n0_raw))
        n0_receipt = bridge.last_validator_receipt
        after = bridge.downstream_activity_snapshot()
        n0 = _rejection_probe_payload(
            label="N0_VALID_SMALL_PLUS_X_UNBOUND_AUTHORITY",
            input_kind="raw_policy_action_v1",
            raw_or_target=n0_raw,
            expected_reason=P0SafetyRejectReason.INTERNAL_VALIDATION_FAILURE.value,
            receipt=n0_receipt,
            activity_before=before,
            activity_after=after,
            surface_unchanged=torch_equal(surface_before, bridge.controller_surface),
            previous_action_unchanged=previous_before == previous.value.values,
            reason="NO_SOURCE_AUTHORITATIVE_DRY_RUN_IK",
        )
        n0.update(
            {
                "route_accepted": n0_route.accepted,
                "route_reason": n0_route.reason,
                "route_ee_command": _plain(n0_route.ee_command),
                "route_gripper_command_emitted": n0_route.gripper_command_emitted,
                "expected_if_existing_authority_were_bound": _static_semantic_regression_contract(),
            }
        )
        probes.append(n0)

        # N1--N3 must use raw validation, not router projection: malformed
        # raw data cannot construct a HighLevelPolicyAction safely.
        for label, raw, expected_reason in (
            (
                "N1_RAW_NAN_REJECT",
                (float("nan"), 0.0, 0.0, 0.50),
                P0SafetyRejectReason.INVALID_INPUT.value,
            ),
            (
                "N2_RAW_INF_REJECT",
                (float("inf"), 0.0, 0.0, 0.50),
                P0SafetyRejectReason.INVALID_INPUT.value,
            ),
            (
                "N3_RAW_ACTION_BOUND_REJECT",
                (1.01, 0.0, 0.0, 0.50),
                P0SafetyRejectReason.ACTION_BOUND_VIOLATION.value,
            ),
        ):
            before = bridge.downstream_activity_snapshot()
            surface_before = bridge.controller_surface
            previous_before = previous.value.values
            receipt = bridge.pre_controller_validator.validate_policy_action(raw)
            after = bridge.downstream_activity_snapshot()
            probes.append(
                _rejection_probe_payload(
                    label=label,
                    input_kind="raw_policy_action_v1",
                    raw_or_target=raw,
                    expected_reason=expected_reason,
                    receipt=receipt,
                    activity_before=before,
                    activity_after=after,
                    surface_unchanged=torch_equal(
                        surface_before, bridge.controller_surface
                    ),
                    previous_action_unchanged=previous_before == previous.value.values,
                )
            )

        # N4 is deliberately a decoded command: it proves exact-zero
        # orientation remains a pre-controller invariant rather than merely
        # a policy action shape convention.
        n4_target = EEResidualCommand(
            translation_m=(0.0, 0.0, 0.0),
            rotation_rad=(0.0, 1.0e-6, 0.0),
            frame=CartesianControlFrame.ROBOT_ROOT,
        )
        before = bridge.downstream_activity_snapshot()
        surface_before = bridge.controller_surface
        previous_before = previous.value.values
        n4_receipt = bridge.pre_controller_validator.validate(n4_target)
        after = bridge.downstream_activity_snapshot()
        probes.append(
            _rejection_probe_payload(
                label="N4_NONZERO_ORIENTATION_REJECT",
                input_kind="decoded_ee_residual",
                raw_or_target=n4_target,
                expected_reason=P0SafetyRejectReason.ORIENTATION_CONTRACT_VIOLATION.value,
                receipt=n4_receipt,
                activity_before=before,
                activity_after=after,
                surface_unchanged=torch_equal(surface_before, bridge.controller_surface),
                previous_action_unchanged=previous_before == previous.value.values,
            )
        )

        # N5/N6 require an existing, source-owned dry-run solver result.  Do
        # not install a fake provider here: a unit fake is valid only for
        # static coverage and can never be P0 live authority evidence.
        for label, required_receipt in (
            ("N5_IK_NO_SOLUTION", P0SafetyRejectReason.IK_NO_SOLUTION.value),
            ("N6_JOINT_LIMIT_REJECT", P0SafetyRejectReason.JOINT_LIMIT_VIOLATION.value),
        ):
            before = bridge.downstream_activity_snapshot()
            surface_before = bridge.controller_surface
            previous_before = previous.value.values
            after = bridge.downstream_activity_snapshot()
            probe = _rejection_probe_payload(
                label=label,
                input_kind="production_dry_run_ik_required",
                raw_or_target=None,
                expected_reason=None,
                receipt=None,
                activity_before=before,
                activity_after=after,
                surface_unchanged=torch_equal(surface_before, bridge.controller_surface),
                previous_action_unchanged=previous_before == previous.value.values,
                status="NOT_EXECUTED_LIVE_AUTHORITY_UNRESOLVED",
                reason="NO_EXISTING_PUBLIC_WORKSPACE_IK_REJECTION_AUTHORITY",
            )
            probe["required_future_receipt_reason"] = required_receipt
            probe["authority_audit"] = P0_A_TRACK_A_AUDIT_RELATIVE_PATH
            probes.append(probe)

        n0_to_n4 = probes[:5]
        negative_input_rejection_pass = all(
            item["status"] == "EXECUTED"
            and item["receipt_rejected"]
            and item["expected_reason_match"]
            and item["no_downstream_activity"]
            and item["controller_surface_unchanged"]
            and item["previous_action_unchanged"]
            for item in n0_to_n4
        )
        no_probe_submission = _no_downstream_activity(
            _counter_delta(baseline_activity, bridge.downstream_activity_snapshot())
        )
        previous_action_pass = previous.value.values == baseline_previous
        surface_pass = torch_equal(baseline_surface, bridge.controller_surface)
        safety_boundary_reason = "NO_EXISTING_PUBLIC_WORKSPACE_IK_REJECTION_AUTHORITY"
        gates = {
            "STATIC_CONTRACT": _gate(
                "PASS",
                evidence=_static_semantic_regression_contract(),
                reason="static regression evidence is intentionally separate from live acceptance",
            ),
            "P0_A_SEMANTIC_FRAME_SIGN_SCALE_ORIENTATION": _gate(
                "PASS",
                evidence=_static_semantic_regression_contract(),
                reason="preserved static semantics; no unbound live acceptance claimed",
            ),
            "GRIPPER_ABSTRACTION_HYSTERESIS_RESET_PREV_ACTION": _gate(
                "PASS" if initial_latch_pass and previous_action_pass else "FAIL",
                evidence={
                    "reset_evidence": reset_evidence,
                    "initial_latch_pass": initial_latch_pass,
                    "previous_action_before_probes": baseline_previous,
                    "previous_action_after_probes": previous.value.values,
                    "policy_gripper_surface": "g_in_0_1_only",
                    "individual_joint_target_api_called": False,
                },
            ),
            "NEGATIVE_INPUT_REJECTION": _gate(
                "PASS" if negative_input_rejection_pass else "FAIL",
                evidence=probes[:5],
            ),
            "P0_A_SAFETY_BOUNDARY": _gate(
                "FAIL",
                reason=safety_boundary_reason,
                evidence={
                    "track_a_existing_authority": "NOT_FOUND",
                    "track_a_public_rejection_receipt": "NOT_EXECUTED",
                    "audit": P0_A_TRACK_A_AUDIT_RELATIVE_PATH,
                    "n5_n6": probes[5:],
                    "new_workspace_or_ik_authority_created": False,
                },
            ),
            "ACCEPT_TARGET_TRANSPARENCY": _gate(
                "FAIL",
                reason=(
                    "NO_LIVE_ACCEPTED_TARGET_WITHOUT_EXISTING_PUBLIC_DRY_RUN_IK_AUTHORITY"
                ),
                evidence={
                    "n0_actual_live_result": "REJECT_INTERNAL_VALIDATION_FAILURE",
                    "static_target_identity_test": "PASS",
                    "static_injected_exact_binding_test": (
                        "tests/test_g2_policy_branch_p0_attestation_static.py::"
                        "test_accepted_submission_requires_private_exact_command_binding"
                    ),
                    "static_test_is_not_live_p0_acceptance_evidence": True,
                },
            ),
            "REJECTED_COMMAND_DOWNSTREAM_ISOLATION": _gate(
                "PASS" if no_probe_submission and surface_pass else "FAIL",
                evidence={
                    "baseline_activity": baseline_activity,
                    "after_all_probes": bridge.downstream_activity_snapshot(),
                    "delta": _counter_delta(
                        baseline_activity, bridge.downstream_activity_snapshot()
                    ),
                    "controller_surface_unchanged": surface_pass,
                    "bridge_private_submission_evidence": bridge.last_bridge_submission_evidence,
                },
            ),
        }
        return {
            "phase": "P0_A",
            "p0_semantic_verdict": "FAIL",
            "functional_execution_verdict": "FAIL_CLOSED_SAFETY_AUTHORITY_UNRESOLVED",
            "p0_a_safety_authority": "ABSENT_OR_UNPROVEN",
            "p0_a_safety_boundary": "FAIL_CLOSED",
            "p0_a_configuration": bridge.p0_a_configuration_receipt(),
            "gates": gates,
            "safety_probes": probes,
            # JSONL records retain each outer N0--N6 probe label, activity
            # deltas, and the nested immutable validator receipt.  Bare
            # receipts alone cannot prove a rejection stayed upstream.
            "safety_receipts": probes,
            "execution_scope": {
                "learned_policy_loaded": False,
                "training_started": False,
                "m2_h47_passive_chain_metrics_collected": False,
                "joint_level_gripper_control": False,
                "raw_low_level_actuator_command": False,
                "policy_to_controller_accepted_ee_command_count": 0,
                "isaac_steps_during_safety_probes": 0,
                "disabled_task_reward_and_curriculum_terms": disabled_task_terms,
                "retained_termination_terms": (
                    "forbidden_collision",
                    "fixed_torso_drift",
                ),
            },
        }
    finally:
        env.close()


def _run_p0_b(prerequisite_a_report: Path) -> dict[str, Any]:
    """P0-B is available only after a complete P0-A PASS."""

    _require_p0_a_pass(prerequisite_a_report)
    import torch

    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        AbstractGripperIntent,
        HighLevelPolicyAction,
        PolicyCommandRouter,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
        PreviousAcceptedPolicyAction,
    )

    env, disabled_task_terms = _make_env("P0_B")
    try:
        env.reset(seed=42)
        bridge = _ExistingControllerP0Bridge(env)
        router = PolicyCommandRouter(
            safety_boundary=bridge,
            ee_controller=bridge,
            gripper_controller=bridge,
        )
        previous = PreviousAcceptedPolicyAction()
        reset_evidence = _settle_open(env, bridge, router)
        previous.reset()
        # P0-B remains a controller-semantic test, not a grasp test.  Each
        # row nevertheless has an explicit expected abstract gripper state so
        # a stuck/rejected gripper cannot be hidden behind a termination-free
        # Cartesian sequence.
        sequence = (
            {
                "label": "APPROACH_X",
                "values": (0.10, 0.0, 0.0, 0.5),
                "holds": 20,
                "expected_intent": AbstractGripperIntent.OPEN,
                "expected_transition": False,
                "motion_axis": 0,
            },
            {
                "label": "CLOSE",
                "values": (0.0, 0.0, 0.0, 0.71),
                "holds": 20,
                "expected_intent": AbstractGripperIntent.CLOSE,
                "expected_transition": True,
                "motion_axis": None,
            },
            {
                "label": "RETREAT_Z",
                "values": (0.0, 0.0, 0.10, 0.5),
                "holds": 20,
                "expected_intent": AbstractGripperIntent.CLOSE,
                "expected_transition": False,
                "motion_axis": 2,
            },
            {
                "label": "OPEN",
                "values": (0.0, 0.0, 0.0, 0.29),
                "holds": 4,
                "expected_intent": AbstractGripperIntent.OPEN,
                "expected_transition": True,
                "motion_axis": None,
            },
        )
        records: list[_P0ExecutionRecord] = []
        sequence_evidence: list[dict[str, Any]] = []
        for item in sequence:
            label = str(item["label"])
            values = item["values"]
            holds = int(item["holds"])
            expected_intent = item["expected_intent"]
            expected_transition = bool(item["expected_transition"])
            motion_axis = item["motion_axis"]
            before = _pose_snapshot(env)
            record = _execute(
                env=env,
                bridge=bridge,
                router=router,
                previous_action=previous,
                label=label,
                action=HighLevelPolicyAction.from_sequence(values),
            )
            records.append(record)
            hold_records = _hold(
                env=env,
                bridge=bridge,
                router=router,
                previous_action=previous,
                count=holds,
                label=f"{label}_HOLD",
            )
            records.extend(hold_records)
            after = _pose_snapshot(env)
            target_delta = (
                after["controller_target_root_position_m"]
                - before["controller_target_root_position_m"]
            )
            measured_delta = (
                after["measured_root_position_m"]
                - before["measured_root_position_m"]
            )
            motion_evidence: dict[str, Any]
            if motion_axis is None:
                motion_evidence = {
                    "required": False,
                    "pass": True,
                    "controller_target_delta_root_m": target_delta,
                    "measured_root_delta_m": measured_delta,
                }
            else:
                expected_delta_m = float(values[motion_axis]) * float(
                    bridge.scale.translation_m_per_normalized
                )
                target_formula = _existing_controller_target_formula(before, record.route)
                router_delta = target_formula["router_limiter_delta_root_m"]
                router_axis_error_m = abs(
                    float(router_delta[0, motion_axis].item()) - expected_delta_m
                )
                controller_formula_error_m = float(
                    torch.max(
                        torch.abs(
                            after["controller_target_root_position_m"]
                            - target_formula["expected_controller_target_root_m"]
                        )
                    ).item()
                )
                raw_target_axis_delta_m = float(target_delta[0, motion_axis].item())
                measured_axis_delta_m = float(measured_delta[0, motion_axis].item())
                motion_evidence = {
                    "required": True,
                    "axis": ("X", "Y", "Z")[motion_axis],
                    "expected_router_delta_m": expected_delta_m,
                    "router_limiter_delta_root_m": router_delta,
                    "controller_target_delta_root_m": target_delta,
                    "controller_target_origin_root_m": target_formula[
                        "controller_origin_root_m"
                    ],
                    "expected_controller_target_root_m": target_formula[
                        "expected_controller_target_root_m"
                    ],
                    "controller_target_formula_error_m": controller_formula_error_m,
                    "target_reconciliation_residual_root_m": target_formula[
                        "target_reconciliation_residual_root_m"
                    ],
                    "measured_root_delta_m": measured_delta,
                    "raw_target_increment_axis_error_m": abs(
                        raw_target_axis_delta_m - expected_delta_m
                    ),
                    "router_axis_error_m": router_axis_error_m,
                    "measured_axis_delta_m": measured_axis_delta_m,
                    "target_pass": (
                        router_axis_error_m <= 1.0e-9
                        and controller_formula_error_m <= 1.0e-6
                    ),
                    "measured_sign_pass": measured_axis_delta_m > 0.0005,
                }
                motion_evidence["pass"] = bool(
                    motion_evidence["target_pass"]
                    and motion_evidence["measured_sign_pass"]
                )
            phase_records = (record, *hold_records)
            sequence_evidence.append(
                {
                    "label": label,
                    "expected_abstract_intent": expected_intent.value,
                    "direct_route_accepted": record.route.accepted,
                    "direct_route_intent": record.route.gripper_intent.value,
                    "direct_observed_controller_intent": record.observed_gripper_intent,
                    "expected_direct_transition": expected_transition,
                    "direct_transition_emitted": record.route.gripper_command_emitted,
                    "hold_route_accepted": [entry.route.accepted for entry in hold_records],
                    "hold_route_intent": [entry.route.gripper_intent.value for entry in hold_records],
                    "hold_observed_controller_intent": [
                        entry.observed_gripper_intent for entry in hold_records
                    ],
                    "hold_transition_emitted": [
                        entry.route.gripper_command_emitted for entry in hold_records
                    ],
                    "abstract_state_pass": all(
                        entry.route.accepted
                        and entry.route.gripper_intent is expected_intent
                        and entry.observed_gripper_intent == expected_intent.value
                        for entry in phase_records
                    ),
                    "hold_latch_pass": not any(
                        entry.route.gripper_command_emitted for entry in hold_records
                    ),
                    "motion": motion_evidence,
                }
            )
        all_routes_accepted = all(record.route.accepted for record in records)
        no_termination = all(
            not record.terminated and not record.truncated for record in records
        )
        abstract_state_pass = all(
            bool(item["abstract_state_pass"])
            and bool(item["hold_latch_pass"])
            and item["direct_transition_emitted"]
            == item["expected_direct_transition"]
            for item in sequence_evidence
        )
        measured_ee_sequence_pass = all(
            bool(item["motion"]["pass"])
            for item in sequence_evidence
            if bool(item["motion"]["required"])
        )
        valid = no_termination and all_routes_accepted and abstract_state_pass and measured_ee_sequence_pass
        valid = valid and all(
            tuple(float(value) for value in record.controller_surface_action_8d[0, 3:7].tolist())
            == (0.0, 0.0, 0.0, 0.0)
            for record in records
        )
        return {
            "phase": "P0_B",
            "p0_semantic_verdict": "PASS" if valid else "FAIL",
            "gates": {
                "COMBINED_EE_GRIPPER_SEQUENCE": _gate(
                    "PASS" if valid else "FAIL",
                    evidence={
                        "all_routes_accepted": all_routes_accepted,
                        "no_termination": no_termination,
                        "abstract_state_pass": abstract_state_pass,
                        "measured_ee_sequence_pass": measured_ee_sequence_pass,
                        "sequence_evidence": sequence_evidence,
                        "runtime_records": [_record_payload(item) for item in records],
                    },
                ),
                "MANIPULATION_EVALUATED": _gate(
                    "NOT_APPLICABLE", reason="P0_B intentionally has no grasp criterion"
                ),
            },
            "reset_evidence": reset_evidence,
            "disabled_task_reward_and_curriculum_terms": disabled_task_terms,
            "sequence_evidence": sequence_evidence,
            "runtime_records": [_record_payload(item) for item in records],
        }
    finally:
        env.close()


def _capture_p0_c_observation(
    *,
    env: Any,
    bridge: _ExistingControllerP0Bridge,
    previous_action: Any,
    hidden_reset: bool,
    binding: Any,
):
    """Construct a no-GT, no-contact, B=1,T=1 branch observation."""

    import torch

    from geniesim.rl.isaaclab.g2_camera_timing import camera_capture_time_and_age
    from geniesim.rl.isaaclab.g2_lift_rgbd_env_cfg import (
        CAMERA_CAPTURE_INTERVAL_STEPS,
        CAMERA_RESOLUTION,
    )
    from geniesim.rl.isaaclab.g2_lift_methodology import RIGHT_ARM_JOINTS
    from geniesim.rl.isaaclab.g2_policy_branch.observation import PolicyObservation

    camera = env.scene["right_wrist_camera"]
    robot = env.scene["robot"]
    output = camera.data.output
    if "rgb" not in output or "distance_to_image_plane" not in output:
        raise RuntimeError("P0_C_WRIST_RGBD_STREAM_MISSING")
    rgb = _tensor(output["rgb"])[..., :3]
    raw_depth = _tensor(output["distance_to_image_plane"])
    if raw_depth.ndim == 3:
        raw_depth = raw_depth.unsqueeze(-1)
    if rgb.dtype != torch.uint8 or raw_depth.shape != (*rgb.shape[:3], 1):
        raise RuntimeError("P0_C_RGB_DEPTH_RUNTIME_SHAPE_OR_TYPE_MISMATCH")
    raw_depth = raw_depth.to(torch.float32)
    maximum_depth_m = float(binding.data_semantics.maximum_depth_m)
    corrupt = torch.isnan(raw_depth) | torch.isneginf(raw_depth) | (
        torch.isfinite(raw_depth) & (raw_depth > maximum_depth_m)
    )
    if bool(corrupt.any().item()):
        raise RuntimeError("P0_C_DEPTH_CORRUPT_OR_OUT_OF_RANGE")
    depth_valid = torch.isfinite(raw_depth) & (raw_depth > 0.0)
    depth_m = torch.where(depth_valid, raw_depth, torch.zeros_like(raw_depth))
    episode_time = _tensor(env.episode_length_buf).to(torch.float32) * float(env.step_dt)
    capture_timestamp_s, frame_age_s = camera_capture_time_and_age(
        camera, fallback_episode_time_s=episode_time
    )
    sequence = _tensor(camera.frame).to(device=env.device, dtype=torch.int64)
    if sequence.ndim == 0:
        sequence = sequence.expand(1)
    position_root, quaternion_root, _, _ = _world_ee_pose_to_root(
        robot, env.scene["ee_frame"]
    )
    ee_pose = torch.cat((position_root, quaternion_root), dim=-1)
    joint_ids, names = robot.find_joints(list(RIGHT_ARM_JOINTS))
    if tuple(names) != RIGHT_ARM_JOINTS:
        raise RuntimeError("P0_C_RIGHT_ARM_ORDER_MISMATCH:" + ",".join(names))
    joint_ids_tensor = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    q = _tensor(robot.data.joint_pos).index_select(1, joint_ids_tensor)
    qd = _tensor(robot.data.joint_vel).index_select(1, joint_ids_tensor)
    abstract_closed = (
        1.0 if bridge.observed_abstract_intent().value == "CLOSE" else 0.0
    )
    semantics = binding.data_semantics
    observation = PolicyObservation(
        right_wrist_rgb=rgb.unsqueeze(1).clone(),
        right_wrist_depth_m=depth_m.unsqueeze(1).clone(),
        right_wrist_depth_valid=depth_valid.unsqueeze(1).clone(),
        ee_pose_robot_root_xyzw=ee_pose.unsqueeze(1).clone(),
        arm_joint_position_rad=q.unsqueeze(1).to(torch.float32).clone(),
        arm_joint_velocity_rad_s=qd.unsqueeze(1).to(torch.float32).clone(),
        current_gripper_state=torch.full(
            (1, 1, 1), abstract_closed, device=env.device, dtype=torch.float32
        ),
        previous_policy_action=torch.tensor(
            previous_action.value.values, device=env.device, dtype=torch.float32
        ).view(1, 1, -1),
        data_semantics=semantics,
        data_semantic_fingerprint=semantics.fingerprint(),
        right_wrist_frame_age_s=frame_age_s.to(torch.float32).view(1, 1, 1),
        hidden_reset_mask=torch.tensor(
            [[hidden_reset]], device=env.device, dtype=torch.bool
        ),
    )
    binding.validate_observation(observation)
    return observation, binding, {
        "capture_timestamp_s": capture_timestamp_s.clone(),
        "frame_age_s": frame_age_s.clone(),
        "sequence_id": sequence.clone(),
        "control_timestamp_s": episode_time.clone(),
        "right_arm_names": names,
        "right_arm_indices": joint_ids,
        "depth_valid_fraction": depth_valid.to(torch.float32).mean().clone(),
        "camera_cfg_update_period_s": float(camera.cfg.update_period),
        "maximum_depth_m": maximum_depth_m,
    }


def _p0_c_expected_runtime_binding(env: Any):
    """Build the P0-C binding from source-owned camera semantics, not a row."""

    from geniesim.rl.isaaclab.g2_lift_env_cfg import SANDBOX
    from geniesim.rl.isaaclab.g2_lift_rgbd_env_cfg import (
        CAMERA_CAPTURE_INTERVAL_STEPS,
        CAMERA_RESOLUTION,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.observation import PolicyDataSemantics
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
        P0RuntimeObservationBinding,
        WristCameraRuntimeContract,
    )

    camera = env.scene["right_wrist_camera"]
    expected_update_period_s = (
        float(SANDBOX.physics_dt_s) * float(CAMERA_CAPTURE_INTERVAL_STEPS)
    )
    actual_geometry = (int(camera.cfg.width), int(camera.cfg.height))
    expected_geometry = (int(CAMERA_RESOLUTION[0]), int(CAMERA_RESOLUTION[1]))
    if actual_geometry != expected_geometry:
        raise RuntimeError(
            "P0_C_CAMERA_RESOLUTION_SOURCE_RUNTIME_MISMATCH:"
            + repr({"source": expected_geometry, "runtime": actual_geometry})
        )
    actual_update_period_s = float(camera.cfg.update_period)
    if not math.isclose(
        actual_update_period_s, expected_update_period_s, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError(
            "P0_C_CAMERA_CADENCE_SOURCE_RUNTIME_MISMATCH:"
            + repr(
                {
                    "source_update_period_s": expected_update_period_s,
                    "runtime_update_period_s": actual_update_period_s,
                }
            )
        )
    # PolicyDataSemantics is intentionally left at its source-defined default.
    # A separate immutable P1 receipt must affirm this exact fingerprint before
    # P0-C can call it a runtime-to-offline match.
    return P0RuntimeObservationBinding(
        data_semantics=PolicyDataSemantics(),
        wrist_camera=WristCameraRuntimeContract(
            width_px=expected_geometry[0],
            height_px=expected_geometry[1],
            capture_interval_physics_steps=CAMERA_CAPTURE_INTERVAL_STEPS,
            maximum_frame_age_s=expected_update_period_s + 1.0e-6,
        ),
    )


def _run_p0_c(
    prerequisite_a_report: Path,
    prerequisite_b_report: Path,
    p1_collection_contract: Path | None,
) -> dict[str, Any]:
    _require_p0_a_pass(prerequisite_a_report)
    report_b = json.loads(prerequisite_b_report.read_text(encoding="utf-8"))
    if report_b.get("phase") != "P0_B" or report_b.get("p0_semantic_verdict") != "PASS":
        raise RuntimeError("P0_C_BLOCKED_BY_P0_B")
    from geniesim.rl.isaaclab.g2_policy_branch.action_interface import (
        HighLevelPolicyAction,
        PolicyCommandRouter,
    )
    from geniesim.rl.isaaclab.g2_policy_branch.p0_runtime_contract import (
        PreviousAcceptedPolicyAction,
        load_p1_collection_runtime_semantic_contract,
    )

    if p1_collection_contract is None:
        raise RuntimeError("P0_C_BLOCKED_NO_FROZEN_P1_COLLECTION_SEMANTIC_CONTRACT")
    if not p1_collection_contract.is_file():
        raise RuntimeError(
            "P0_C_MISSING_FROZEN_P1_COLLECTION_SEMANTIC_CONTRACT:"
            + str(p1_collection_contract)
        )

    env, disabled_task_terms = _make_env("P0_C")
    try:
        env.reset(seed=42)
        bridge = _ExistingControllerP0Bridge(env)
        router = PolicyCommandRouter(
            safety_boundary=bridge,
            ee_controller=bridge,
            gripper_controller=bridge,
        )
        initial_reset_evidence = _settle_open(env, bridge, router)
        previous = PreviousAcceptedPolicyAction()
        previous.reset()

        binding = _p0_c_expected_runtime_binding(env)
        p1_receipt = load_p1_collection_runtime_semantic_contract(
            p1_collection_contract.read_text(encoding="utf-8"),
            data_semantics=binding.data_semantics,
            p0_runtime_observation_binding=binding,
        )

        def advance_and_capture(*, label: str, hidden_reset: bool):
            """Advance the existing hold surface until one fresh RGB-D row.

            No policy request is synthesized here.  The existing controller is
            merely held after the preceding accepted request, or after an
            injected rejection.  Thus camera cadence work cannot overwrite the
            observation's last-accepted branch action.
            """

            sequence_before = int(
                _tensor(env.scene["right_wrist_camera"].frame).reshape(-1)[0].item()
            )
            maximum_steps = max(
                4, int(binding.wrist_camera.capture_interval_physics_steps) + 2
            )
            for _ in range(maximum_steps):
                _, terminated, truncated, observed, active = bridge.step()
                router.synchronize_gripper_intent(observed)
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError(
                        f"P0_C_{label}_CAPTURE_TERMINATED:" + ",".join(active)
                    )
                sequence_after = int(
                    _tensor(env.scene["right_wrist_camera"].frame)
                    .reshape(-1)[0]
                    .item()
                )
                if sequence_after > sequence_before:
                    return _capture_p0_c_observation(
                        env=env,
                        bridge=bridge,
                        previous_action=previous,
                        hidden_reset=hidden_reset,
                        binding=binding,
                    )
            raise RuntimeError(f"P0_C_{label}_CAMERA_FRESHNESS_TIMEOUT")

        # An actual reset row: previous action must be zero and gripper state
        # means controller command OPEN only, never physical finger/contact
        # confirmation.
        initial_capture = advance_and_capture(label="INITIAL", hidden_reset=True)
        captures = [initial_capture]

        accepted_action = HighLevelPolicyAction.from_sequence((0.05, 0.0, 0.0, 0.5))
        accepted_record = _execute(
            env=env,
            bridge=bridge,
            router=router,
            previous_action=previous,
            label="P0_C_ACCEPTED_PREVIOUS_ACTION_PROBE",
            action=accepted_action,
        )
        if (
            not accepted_record.route.accepted
            or accepted_record.terminated
            or accepted_record.truncated
        ):
            raise RuntimeError("P0_C_ACCEPTED_PREVIOUS_ACTION_PROBE_FAILED")
        bridge.set_safe_hold_surface()
        accepted_capture = advance_and_capture(label="ACCEPTED", hidden_reset=False)
        captures.append(accepted_capture)

        rejected_action = HighLevelPolicyAction.from_sequence((-0.15, 0.0, 0.0, 0.9))
        previous_before_reject = previous.value.values
        bridge.reject_all_for_fault_injection = True
        try:
            rejected_route = router.route(rejected_action)
        finally:
            bridge.reject_all_for_fault_injection = False
        if rejected_route.accepted or rejected_route.ee_command is not None:
            raise RuntimeError("P0_C_INJECTED_REJECTION_NOT_ENFORCED")
        bridge.set_safe_hold_surface()
        rejected_capture = advance_and_capture(label="REJECTED", hidden_reset=False)
        captures.append(rejected_capture)
        rejected_ring_unchanged = previous.value.values == previous_before_reject

        timestamps = [float(item[2]["capture_timestamp_s"][0].item()) for item in captures]
        sequences = [int(item[2]["sequence_id"][0].item()) for item in captures]
        ages = [float(item[2]["frame_age_s"][0].item()) for item in captures]
        strict_sequence = all(second > first for first, second in zip(sequences, sequences[1:]))
        nondecreasing_time = all(second >= first for first, second in zip(timestamps, timestamps[1:]))

        # Exercise the actual environment/action-manager reset boundary.  The
        # controller-facing latch, its existing OPEN settle, and the branch
        # previous-action ring must all restart before the first post-reset
        # observation is created.  Merely setting a reset flag on an existing
        # row would not attest this semantic.
        env.reset(seed=42)
        previous.reset()
        post_reset_evidence = _settle_open(env, bridge, router)
        reset_observation, reset_binding, reset_metadata = advance_and_capture(
            label="POST_RESET", hidden_reset=True
        )
        initial_previous = tuple(
            float(value)
            for value in initial_capture[0].previous_policy_action[0, 0].tolist()
        )
        accepted_previous = tuple(
            float(value)
            for value in accepted_capture[0].previous_policy_action[0, 0].tolist()
        )
        rejected_previous = tuple(
            float(value)
            for value in rejected_capture[0].previous_policy_action[0, 0].tolist()
        )
        accepted_previous_pass = all(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-7)
            for observed, expected in zip(accepted_previous, accepted_action.values)
        )
        rejected_previous_pass = (
            not rejected_route.accepted
            and all(
                math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-7)
                for observed, expected in zip(rejected_previous, previous_before_reject)
            )
            and rejected_ring_unchanged
        )
        reset_ok = (
            bool(reset_observation.hidden_reset_mask[0, 0].item())
            and tuple(float(value) for value in reset_observation.previous_policy_action[0, 0].tolist())
            == (0.0, 0.0, 0.0, 0.0)
            and float(reset_observation.current_gripper_state[0, 0, 0].item()) == 0.0
            and post_reset_evidence["router_reset_submission"].accepted
            and post_reset_evidence["observed_after_settle"].value == "OPEN"
        )
        binding_match = all(
            item[0].data_semantic_fingerprint == binding.data_semantics.fingerprint()
            and item[1].fingerprint() == binding.fingerprint()
            for item in captures
        ) and (
            reset_observation.data_semantic_fingerprint
            == binding.data_semantics.fingerprint()
            and reset_binding.fingerprint() == binding.fingerprint()
        )
        runtime_ok = strict_sequence and nondecreasing_time and all(
            age <= float(binding.wrist_camera.maximum_frame_age_s) for age in ages
        )
        previous_action_ok = (
            initial_previous == (0.0, 0.0, 0.0, 0.0)
            and accepted_previous_pass
            and rejected_previous_pass
        )
        return {
            "phase": "P0_C",
            "p0_semantic_verdict": "PASS"
            if runtime_ok and reset_ok and binding_match and previous_action_ok
            else "FAIL",
            "gates": {
                "OBSERVATION_RUNTIME_CONTRACT": _gate(
                    "PASS" if runtime_ok and reset_ok and previous_action_ok else "FAIL",
                    evidence={
                        "capture_timestamp_s": timestamps,
                        "sequence_id": sequences,
                        "frame_age_s": ages,
                        "initial_reset_evidence": initial_reset_evidence,
                        "post_reset_evidence": post_reset_evidence,
                        "reset_ok": reset_ok,
                        "previous_action_contract": {
                            "authority": "last_accepted_branch_policy_action_lag1_v1",
                            "initial_previous_action": initial_previous,
                            "accepted_action": accepted_action.values,
                            "accepted_capture_previous_action": accepted_previous,
                            "accepted_previous_pass": accepted_previous_pass,
                            "rejected_action": rejected_action.values,
                            "rejected_route_reason": rejected_route.reason,
                            "rejected_capture_previous_action": rejected_previous,
                            "rejected_previous_pass": rejected_previous_pass,
                        },
                        "pre_reset_last_metadata": captures[-1][2],
                        "post_reset_metadata": reset_metadata,
                    },
                ),
                "SEMANTIC_FINGERPRINT_RUNTIME_MATCH": _gate(
                    "PASS" if binding_match else "FAIL",
                    evidence={
                        "data_semantic_fingerprint": binding.data_semantics.fingerprint(),
                        "runtime_binding_fingerprint": binding.fingerprint(),
                        "runtime_binding": binding.payload(),
                        "external_p1_collection_contract_path": str(
                            p1_collection_contract
                        ),
                        "external_p1_collection_contract": p1_receipt.payload(),
                    },
                ),
            },
            "actor_input_fields": (
                "right_wrist_rgb",
                "right_wrist_depth_m",
                "right_wrist_depth_valid",
                "ee_pose_robot_root_xyzw",
                "arm_joint_position_rad",
                "arm_joint_velocity_rad_s",
                "current_gripper_state",
                "previous_policy_action",
            ),
            "privileged_or_gt_actor_input": False,
            "learned_policy_loaded": False,
            "reset_metadata": reset_metadata,
            "disabled_task_reward_and_curriculum_terms": disabled_task_terms,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0-a", "p0-b", "p0-c"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-a-report", type=Path)
    parser.add_argument("--p0-b-report", type=Path)
    parser.add_argument(
        "--p1-collection-contract",
        type=Path,
        help=(
            "immutable external P1 semantic receipt required only by P0-C; "
            "missing or mismatched receipts fail closed"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; P0 evidence is immutable")
    if args.phase in {"p0-b", "p0-c"} and args.p0_a_report is None:
        parser.error("P0-B/C require --p0-a-report")
    if args.phase == "p0-c" and args.p0_b_report is None:
        parser.error("P0-C requires --p0-b-report")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = _source_manifest()
    _atomic_json(args.output / "source_freeze.json", manifest)
    # The production asset is immutable authority for this bounded live probe.
    # The frozen M2 trajectory remains a read-only H4.7 provenance record and
    # is never *executed* or used as a P0 controller-semantic authority.
    # Its identity is nevertheless protected: P0 evidence must not be emitted
    # beside a silently changed frozen reference artifact.
    protected_reference_hash_checks = _protected_reference_hash_checks(manifest)
    if not all(protected_reference_hash_checks.values()):
        failure = {
            "schema": P0_SCHEMA,
            "phase": args.phase.upper().replace("-", "_"),
            "p0_semantic_verdict": "FAIL",
            "reason": "PROTECTED_REFERENCE_HASH_MISMATCH",
            "source_manifest": manifest,
            "protected_reference_hash_checks": protected_reference_hash_checks,
        }
        if args.phase == "p0-a":
            _write_p0_a_evidence_artifacts(args.output, failure)
        _atomic_json(args.output / "functional_result_pre_close.json", failure)
        return 2
    print("SOURCE_FREEZE_OK", flush=True)

    from isaaclab.app import AppLauncher

    phase = args.phase.upper().replace("-", "_")
    launcher = AppLauncher(
        headless=True,
        enable_cameras=args.phase == "p0-c",
        fast_shutdown=False,
    )
    app = launcher.app
    print("APP_CREATED", flush=True)
    result: dict[str, Any]
    try:
        print("RUNTIME_BEGIN", flush=True)
        if args.phase == "p0-a":
            result = _run_p0_a()
        elif args.phase == "p0-b":
            assert args.p0_a_report is not None
            result = _run_p0_b(args.p0_a_report)
        else:
            assert args.p0_a_report is not None and args.p0_b_report is not None
            result = _run_p0_c(
                args.p0_a_report,
                args.p0_b_report,
                args.p1_collection_contract,
            )
        print("RUNTIME_END", flush=True)
    except BaseException as error:
        result = {
            "phase": phase,
            "p0_semantic_verdict": "FAIL",
            "gates": {},
            "exception": repr(error),
            "traceback": traceback.format_exc(),
        }
    post_manifest = _source_manifest()
    result.update(
        {
            "schema": P0_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_before": manifest,
            "source_manifest_after": post_manifest,
            "source_hash_match_at_report": manifest["source_files_sha256"]
            == post_manifest["source_files_sha256"],
            "physics_verdict": "NOT_EVALUATED_BY_P0_CONTROLLER_SEMANTIC_ATTESTATION",
            "artifact_verdict": "PASS",
            "process_verdict": "UNCLASSIFIED_UNTIL_CHILD_EXIT",
            "promotion": {
                # Only the minimal parent supervisor observes true child exit;
                # a pre-close semantic report must never promote P1 by itself.
                "p1_collection": "PENDING_P0_SUPERVISOR_PROCESS_VERDICT"
                if result.get("p0_semantic_verdict") == "PASS"
                else "BLOCKED_ON_P0",
                "p2_training_readiness": "BLOCKED_ON_P0_P1",
                "h4_7": "SEPARATE_UNRESOLVED",
            },
        }
    )
    if args.phase == "p0-a":
        _write_p0_a_evidence_artifacts(args.output, result)
    _atomic_json(args.output / "functional_result_pre_close.json", result)
    print("REPORT_SAVED", flush=True)
    try:
        app.close()
        print("APP_CLOSED", flush=True)
    finally:
        # The native process can still fault after this marker.  The parent
        # shell/attestation must record the actual exit status separately.
        print("PROCESS_EXIT", flush=True)
    return 0 if result.get("p0_semantic_verdict") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
