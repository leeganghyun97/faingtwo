# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Authoritative terminal-transition capture for ManagerBasedRLEnv.

Isaac Lab auto-resets completed environments inside ``env.step`` and returns
the *reset* observation.  That returned value must never be joined to the
terminal action/reward in replay.  This module provides a source-owned,
exact-4D receipt captured by the official ``RecorderTerm.record_post_step``
hook, which executes after reward/termination computation and before reset.

The module does not step physics, submit an action, alter a controller, or
change termination/reward semantics.  Installing the recorder only creates a
read-only observation receipt.  Existing nonterminal smoke paths do not need
to install it and remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .action_interface import HighLevelPolicyAction, PolicyActionMode
from .observation import PolicyObservation


TERMINAL_TRANSITION_SCHEMA = "g2_policy_4d_terminal_transition_v1"
TERMINAL_CAPTURE_BINDING_SCHEMA = "g2_policy_4d_terminal_capture_binding_v1"
MANAGER_AUTO_RESET_AUDIT_SCHEMA = "g2_manager_based_auto_reset_audit_v1"
TERMINAL_RECORDER_TERM_NAME = "g2_policy_4d_terminal_next_state"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TerminalTransitionContractError(RuntimeError):
    """Raised before an ambiguous transition can enter replay."""


class NextObservationSource(str, Enum):
    """Provenance of a replay next-observation."""

    RETURNED_NONTERMINAL = "RETURNED_NONTERMINAL"
    RECORDER_POST_STEP_PRE_RESET = "RECORDER_POST_STEP_PRE_RESET"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_bool(value: Any, *, name: str) -> bool:
    try:
        flattened = value.reshape(-1)
        if int(flattened.shape[0]) != 1:
            raise ValueError
        scalar = flattened[0].item()
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TerminalTransitionContractError(
            f"{name} must be a one-environment scalar tensor"
        ) from error
    if scalar not in (False, True, 0, 1):
        raise TerminalTransitionContractError(f"{name} is not boolean")
    return bool(scalar)


def _scalar_float(value: Any, *, name: str) -> float:
    try:
        flattened = value.reshape(-1)
        if int(flattened.shape[0]) != 1:
            raise ValueError
        scalar = float(flattened[0].item())
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TerminalTransitionContractError(
            f"{name} must be a one-environment scalar tensor"
        ) from error
    if not math.isfinite(scalar):
        raise TerminalTransitionContractError(f"{name} must be finite")
    return scalar


@dataclass(frozen=True)
class ManagerAutoResetAudit:
    """Static proof of the installed ManagerBasedRLEnv step ordering."""

    source_path: str
    source_sha256: str
    anchor_offsets: Mapping[str, int]
    post_step_recorder_precedes_reset: bool
    returned_observation_is_post_reset_for_done_env: bool
    schema: str = MANAGER_AUTO_RESET_AUDIT_SCHEMA

    @property
    def passed(self) -> bool:
        return bool(
            self.schema == MANAGER_AUTO_RESET_AUDIT_SCHEMA
            and _SHA256.fullmatch(self.source_sha256)
            and self.post_step_recorder_precedes_reset
            and self.returned_observation_is_post_reset_for_done_env
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "anchor_offsets": dict(self.anchor_offsets),
            "post_step_recorder_precedes_reset": self.post_step_recorder_precedes_reset,
            "returned_observation_is_post_reset_for_done_env": (
                self.returned_observation_is_post_reset_for_done_env
            ),
            "terminal_next_observation_authority": (
                NextObservationSource.RECORDER_POST_STEP_PRE_RESET.value
                if self.passed
                else "UNRESOLVED"
            ),
        }


def audit_manager_based_rl_env_step(source_path: str | Path) -> ManagerAutoResetAudit:
    """Fail closed unless RecorderTerm is provably before reset and return after it."""

    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise TerminalTransitionContractError(
            f"ManagerBasedRLEnv source is missing: {path}"
        )
    source = path.read_text(encoding="utf-8")
    anchors = {
        "termination_compute": "self.reset_buf = self.termination_manager.compute()",
        "reward_compute": "self.reward_buf = self.reward_manager.compute(dt=self.step_dt)",
        "record_post_step": "self.recorder_manager.record_post_step()",
        "record_pre_reset": "self.recorder_manager.record_pre_reset(reset_env_ids)",
        "reset": "self._reset_idx(reset_env_ids)",
        "post_reset_observation": (
            "self.obs_buf = self.observation_manager.compute(update_history=True)"
        ),
        "return": (
            "return self.obs_buf, self.reward_buf, self.reset_terminated, "
            "self.reset_time_outs, self.extras"
        ),
    }
    offsets = {name: source.find(anchor) for name, anchor in anchors.items()}
    present = all(offset >= 0 for offset in offsets.values())
    ordered = bool(
        present
        and offsets["termination_compute"] < offsets["reward_compute"]
        < offsets["record_post_step"] < offsets["record_pre_reset"]
        < offsets["reset"] < offsets["post_reset_observation"]
        < offsets["return"]
    )
    audit = ManagerAutoResetAudit(
        source_path=str(path),
        source_sha256=_file_sha256(path),
        anchor_offsets=offsets,
        post_step_recorder_precedes_reset=ordered,
        returned_observation_is_post_reset_for_done_env=ordered,
    )
    if not audit.passed:
        raise TerminalTransitionContractError(
            "ManagerBasedRLEnv auto-reset/recorder ordering is unresolved"
        )
    return audit


@dataclass(frozen=True)
class TerminalObservationCaptureBinding:
    """Immutable provenance for the callable that builds PolicyObservation."""

    binding_id: str
    source_path: str
    source_sha256: str
    observation_semantic_fingerprint: str
    schema: str = TERMINAL_CAPTURE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TERMINAL_CAPTURE_BINDING_SCHEMA or not self.binding_id:
            raise TerminalTransitionContractError("invalid terminal capture binding")
        if not _SHA256.fullmatch(self.source_sha256):
            raise TerminalTransitionContractError("capture source SHA-256 is invalid")
        if not _SHA256.fullmatch(self.observation_semantic_fingerprint):
            raise TerminalTransitionContractError(
                "observation semantic fingerprint is invalid"
            )
        path = Path(self.source_path).expanduser().resolve()
        if not path.is_file() or _file_sha256(path) != self.source_sha256:
            raise TerminalTransitionContractError(
                "terminal observation capture source changed or is missing"
            )

    def payload(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "binding_id": self.binding_id,
            "source_path": str(Path(self.source_path).expanduser().resolve()),
            "source_sha256": self.source_sha256,
            "observation_semantic_fingerprint": self.observation_semantic_fingerprint,
        }

    def assert_current(self) -> None:
        """Fail closed if the source-owned capture adapter changed after binding."""

        path = Path(self.source_path).expanduser().resolve()
        if not path.is_file() or _file_sha256(path) != self.source_sha256:
            raise TerminalTransitionContractError(
                "terminal observation capture source changed after binding"
            )


@dataclass(frozen=True)
class StagedPolicyPacket4D:
    """One exact-4D policy packet awaiting one canonical env.step."""

    step: int
    packet_id: str
    packet_sha256: str
    action: HighLevelPolicyAction

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise TerminalTransitionContractError("staged step must be non-negative")
        if not self.packet_id or not _SHA256.fullmatch(self.packet_sha256):
            raise TerminalTransitionContractError("staged packet identity is invalid")
        if (
            not isinstance(self.action, HighLevelPolicyAction)
            or self.action.mode is not PolicyActionMode.TRANSLATION_GRIPPER_4D
            or len(self.action.values) != 4
        ):
            raise TerminalTransitionContractError(
                "terminal transition authority accepts exact 4-D actions only"
            )


@dataclass(frozen=True)
class TerminalTransitionReceipt:
    """Pre-reset physical next-observation for one terminal transition."""

    staged: StagedPolicyPacket4D
    observation: PolicyObservation
    reward: float
    terminated: bool
    truncated: bool
    active_termination_terms: tuple[str, ...]
    capture_index: int
    capture_binding: TerminalObservationCaptureBinding
    capture_metadata: Mapping[str, Any]
    schema: str = TERMINAL_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TERMINAL_TRANSITION_SCHEMA:
            raise TerminalTransitionContractError("terminal receipt schema mismatch")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TerminalTransitionContractError("done flags must be bool")
        if self.terminated == self.truncated:
            raise TerminalTransitionContractError(
                "terminal receipt requires exactly one terminated/truncated flag"
            )
        if not math.isfinite(self.reward):
            raise TerminalTransitionContractError("terminal reward must be finite")
        if type(self.capture_index) is not int or self.capture_index <= 0:
            raise TerminalTransitionContractError("capture index must be positive")
        if not isinstance(self.observation, PolicyObservation):
            raise TerminalTransitionContractError(
                "terminal next observation must be PolicyObservation"
            )
        self.observation.validate(self.observation.data_semantics.observation_config)
        if (
            self.observation.data_semantic_fingerprint
            != self.capture_binding.observation_semantic_fingerprint
        ):
            raise TerminalTransitionContractError(
                "terminal observation semantic fingerprint mismatch"
            )
        try:
            json.dumps(dict(self.capture_metadata), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TerminalTransitionContractError(
                "terminal capture metadata must be finite JSON"
            ) from error

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "step": self.staged.step,
            "packet_id": self.staged.packet_id,
            "packet_sha256": self.staged.packet_sha256,
            "action_schema": self.staged.action.mode.value,
            "action_4d": list(self.staged.action.values),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "active_termination_terms": list(self.active_termination_terms),
            "capture_index": self.capture_index,
            "capture_binding": self.capture_binding.payload(),
            "capture_metadata": dict(self.capture_metadata),
            "next_observation_source": (
                NextObservationSource.RECORDER_POST_STEP_PRE_RESET.value
            ),
        }


@dataclass(frozen=True)
class ReplayNextObservationSelection:
    """Resolved replay source after env.step returns."""

    next_observation: PolicyObservation
    source: NextObservationSource
    terminated: bool
    truncated: bool
    terminal_receipt: TerminalTransitionReceipt | None


CaptureCallable = Callable[
    [Any, HighLevelPolicyAction], tuple[PolicyObservation, Mapping[str, Any]]
]


class TerminalTransitionCaptureContext:
    """Single-consumption lifecycle shared with one official RecorderTerm."""

    def __init__(
        self,
        *,
        capture: CaptureCallable,
        binding: TerminalObservationCaptureBinding,
        auto_reset_audit: ManagerAutoResetAudit,
    ) -> None:
        if not callable(capture):
            raise TerminalTransitionContractError("capture callback must be callable")
        if not isinstance(binding, TerminalObservationCaptureBinding):
            raise TerminalTransitionContractError("capture binding is required")
        if not isinstance(auto_reset_audit, ManagerAutoResetAudit) or not auto_reset_audit.passed:
            raise TerminalTransitionContractError("passing auto-reset audit is required")
        self.capture = capture
        self.binding = binding
        self.auto_reset_audit = auto_reset_audit
        self._staged: StagedPolicyPacket4D | None = None
        self._callback_count = 0
        self._stage_callback_start = 0
        self._callback_done: tuple[bool, bool] | None = None
        self._terminal_receipt: TerminalTransitionReceipt | None = None
        self._capture_index = 0

    @property
    def callback_count(self) -> int:
        return self._callback_count

    def stage(self, packet: StagedPolicyPacket4D) -> None:
        if self._staged is not None:
            raise TerminalTransitionContractError(
                "previous policy packet has not been resolved"
            )
        if not isinstance(packet, StagedPolicyPacket4D):
            raise TerminalTransitionContractError("stage requires exact packet receipt")
        self.binding.assert_current()
        self._staged = packet
        self._stage_callback_start = self._callback_count
        self._callback_done = None
        self._terminal_receipt = None

    def record_post_step(self, env: Any) -> None:
        """Capture inside RecorderTerm, after reward/done and before auto-reset."""

        staged = self._staged
        if staged is None:
            raise TerminalTransitionContractError(
                "RecorderTerm callback occurred without a staged packet"
            )
        self.binding.assert_current()
        if self._callback_count != self._stage_callback_start:
            raise TerminalTransitionContractError(
                "duplicate RecorderTerm callback for one policy packet"
            )
        terminated = _scalar_bool(env.reset_terminated, name="reset_terminated")
        truncated = _scalar_bool(env.reset_time_outs, name="reset_time_outs")
        if terminated and truncated:
            raise TerminalTransitionContractError(
                "transition cannot be both terminated and truncated"
            )
        self._callback_count += 1
        self._callback_done = (terminated, truncated)
        if not (terminated or truncated):
            return

        observation, metadata = self.capture(env, staged.action)
        capture_metadata = dict(metadata)
        try:
            json.dumps(capture_metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TerminalTransitionContractError(
                "terminal observation metadata must be finite JSON"
            ) from error
        terms: list[str] = []
        manager = env.termination_manager
        for name in manager.active_terms:
            if _scalar_bool(manager.get_term(name), name=f"termination:{name}"):
                terms.append(str(name))
        self._capture_index += 1
        self._terminal_receipt = TerminalTransitionReceipt(
            staged=staged,
            observation=observation,
            reward=_scalar_float(env.reward_buf, name="reward_buf"),
            terminated=terminated,
            truncated=truncated,
            active_termination_terms=tuple(terms),
            capture_index=self._capture_index,
            capture_binding=self.binding,
            capture_metadata=capture_metadata,
        )

    def resolve_after_env_step(
        self,
        *,
        returned_observation: PolicyObservation,
        returned_terminated: Any,
        returned_truncated: Any,
        expected_step: int,
        expected_packet_id: str,
        expected_packet_sha256: str,
    ) -> ReplayNextObservationSelection:
        """Select replay next-state without ever joining reset state to done."""

        staged = self._staged
        if staged is None:
            raise TerminalTransitionContractError("no staged packet to resolve")
        if (
            staged.step != expected_step
            or staged.packet_id != expected_packet_id
            or staged.packet_sha256 != expected_packet_sha256
        ):
            raise TerminalTransitionContractError(
                "returned step/packet identity does not match staged action"
            )
        if self._callback_count != self._stage_callback_start + 1 or self._callback_done is None:
            raise TerminalTransitionContractError(
                "exactly one pre-reset RecorderTerm callback is required"
            )
        terminated = _scalar_bool(returned_terminated, name="returned_terminated")
        truncated = _scalar_bool(returned_truncated, name="returned_truncated")
        if terminated and truncated:
            raise TerminalTransitionContractError(
                "returned transition cannot be both terminated and truncated"
            )
        if self._callback_done != (terminated, truncated):
            raise TerminalTransitionContractError(
                "pre-reset and returned termination flags differ"
            )

        receipt = self._terminal_receipt
        if terminated or truncated:
            if receipt is None:
                raise TerminalTransitionContractError(
                    "terminal transition lacks pre-reset next observation"
                )
            if (
                receipt.staged.packet_id != expected_packet_id
                or receipt.staged.packet_sha256 != expected_packet_sha256
                or receipt.staged.step != expected_step
                or receipt.terminated != terminated
                or receipt.truncated != truncated
            ):
                raise TerminalTransitionContractError(
                    "terminal receipt identity/done flags mismatch"
                )
            selection = ReplayNextObservationSelection(
                next_observation=receipt.observation,
                source=NextObservationSource.RECORDER_POST_STEP_PRE_RESET,
                terminated=terminated,
                truncated=truncated,
                terminal_receipt=receipt,
            )
        else:
            if receipt is not None:
                raise TerminalTransitionContractError(
                    "nonterminal transition unexpectedly has terminal receipt"
                )
            if not isinstance(returned_observation, PolicyObservation):
                raise TerminalTransitionContractError(
                    "nonterminal next observation must be PolicyObservation"
                )
            returned_observation.validate(
                returned_observation.data_semantics.observation_config
            )
            selection = ReplayNextObservationSelection(
                next_observation=returned_observation,
                source=NextObservationSource.RETURNED_NONTERMINAL,
                terminated=False,
                truncated=False,
                terminal_receipt=None,
            )

        self._staged = None
        self._callback_done = None
        self._terminal_receipt = None
        return selection


def build_terminal_recorder_term_class(
    recorder_term_base: type,
    context: TerminalTransitionCaptureContext,
) -> type:
    """Build the official RecorderTerm adapter without importing Isaac at module load."""

    if not isinstance(context, TerminalTransitionCaptureContext):
        raise TerminalTransitionContractError("terminal capture context is required")

    class G2Policy4DTerminalRecorder(recorder_term_base):
        def record_post_step(self):
            context.record_post_step(self._env)
            return None, None

    G2Policy4DTerminalRecorder.__name__ = "G2Policy4DTerminalRecorder"
    return G2Policy4DTerminalRecorder


def install_terminal_transition_recorder(
    cfg: Any,
    *,
    context: TerminalTransitionCaptureContext,
) -> None:
    """Configure one read-only recorder before ManagerBasedRLEnv construction.

    No controller, reward, termination, action, or physics setting is changed.
    Existing recorder/export settings are intentionally preserved.  The only
    configuration change is adding this read-only pre-reset receipt term.
    """

    from isaaclab.managers import RecorderTermCfg
    from isaaclab.managers.recorder_manager import RecorderTerm

    recorders = getattr(cfg, "recorders", None)
    if recorders is None:
        raise TerminalTransitionContractError("environment has no recorder config")
    if getattr(recorders, TERMINAL_RECORDER_TERM_NAME, None) is not None:
        raise TerminalTransitionContractError("terminal recorder already configured")
    setattr(
        recorders,
        TERMINAL_RECORDER_TERM_NAME,
        RecorderTermCfg(
            class_type=build_terminal_recorder_term_class(RecorderTerm, context)
        ),
    )


__all__ = [
    "MANAGER_AUTO_RESET_AUDIT_SCHEMA",
    "ManagerAutoResetAudit",
    "NextObservationSource",
    "ReplayNextObservationSelection",
    "StagedPolicyPacket4D",
    "TERMINAL_CAPTURE_BINDING_SCHEMA",
    "TERMINAL_RECORDER_TERM_NAME",
    "TERMINAL_TRANSITION_SCHEMA",
    "TerminalObservationCaptureBinding",
    "TerminalTransitionCaptureContext",
    "TerminalTransitionContractError",
    "TerminalTransitionReceipt",
    "audit_manager_based_rl_env_step",
    "build_terminal_recorder_term_class",
    "install_terminal_transition_recorder",
]
