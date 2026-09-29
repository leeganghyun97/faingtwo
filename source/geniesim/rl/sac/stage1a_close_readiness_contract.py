# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed contract for teacher-only CLOSE-readiness supervision.

The simulator geometry oracle may create labels, but a student may only learn
from *pre-CLOSE* deployable observations.  In particular, a post-latch/contact
row is not a negative CLOSE-admission example: it describes a different state
of the gripper state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CLOSE_READINESS_TARGET_SCHEMA_VERSION = "g2_close_readiness_preclose_binary_v1"


class CloseReadinessContractError(ValueError):
    """Raised when a teacher label would violate the causal contract."""


@dataclass(frozen=True)
class PreCloseTeacherTarget:
    """Binary, pre-CLOSE-only teacher target plus explicit failure reasons."""

    pre_close_candidate: bool
    close_latched_before_supervision: bool
    recordable: bool
    target: bool | None
    score: float | None
    negative_reasons: tuple[str, ...]
    excluded_reason: str | None


def is_pre_close_candidate(*, phase: str, close_latched_before_supervision: bool) -> bool:
    """Return the sole phase/state predicate allowed to enter this dataset."""

    if not isinstance(phase, str) or not phase:
        raise CloseReadinessContractError("phase must be a nonempty string")
    if not isinstance(close_latched_before_supervision, bool):
        raise CloseReadinessContractError("close_latched_before_supervision must be bool")
    return bool(phase == "LOCAL_GRASP" and not close_latched_before_supervision)


def build_pre_close_teacher_target(
    *,
    pre_close_candidate: bool,
    close_latched_before_supervision: bool,
    pad_geometry_ready: bool,
    aperture_ready: bool,
    orientation_ready: bool,
    owner_valid: bool,
    geometry_valid: bool,
    no_safety_violation: bool,
) -> PreCloseTeacherTarget:
    """Build an admission-semantic target without folding safety into geometry.

    Safety-invalid rows are excluded rather than relabelled as geometry
    negatives.  A recordable target is strictly binary: score 1.0 means ready
    and score 0.0 means not-ready.  Predicate fractions are deliberately
    forbidden because a score such as 0.75 has no valid CLOSE-admission
    probability meaning for a binary not-ready target.
    """

    flags = {
        "pre_close_candidate": pre_close_candidate,
        "close_latched_before_supervision": close_latched_before_supervision,
        "pad_geometry_ready": pad_geometry_ready,
        "aperture_ready": aperture_ready,
        "orientation_ready": orientation_ready,
        "owner_valid": owner_valid,
        "geometry_valid": geometry_valid,
        "no_safety_violation": no_safety_violation,
    }
    if not all(isinstance(value, bool) for value in flags.values()):
        raise CloseReadinessContractError("all CLOSE-readiness predicates must be bool")
    if not pre_close_candidate:
        return PreCloseTeacherTarget(
            pre_close_candidate=False,
            close_latched_before_supervision=close_latched_before_supervision,
            recordable=False,
            target=None,
            score=None,
            negative_reasons=(),
            excluded_reason="NOT_PRE_CLOSE_CANDIDATE",
        )
    if close_latched_before_supervision:
        return PreCloseTeacherTarget(
            pre_close_candidate=True,
            close_latched_before_supervision=True,
            recordable=False,
            target=None,
            score=None,
            negative_reasons=(),
            excluded_reason="POST_CLOSE_LATCHED",
        )
    if not no_safety_violation:
        return PreCloseTeacherTarget(
            pre_close_candidate=True,
            close_latched_before_supervision=False,
            recordable=False,
            target=None,
            score=None,
            negative_reasons=(),
            excluded_reason="SAFETY_EXCLUDED",
        )
    reasons: list[str] = []
    if not owner_valid:
        reasons.append("OWNER_INVALID")
    if not geometry_valid:
        reasons.append("GEOMETRY_INVALID")
    if not pad_geometry_ready:
        reasons.append("PAD_GEOMETRY_NOT_READY")
    if not aperture_ready:
        reasons.append("APERTURE_INCOMPATIBLE")
    if not orientation_ready:
        reasons.append("ORIENTATION_NOT_READY")
    target = not reasons
    return PreCloseTeacherTarget(
        pre_close_candidate=True,
        close_latched_before_supervision=False,
        recordable=True,
        target=target,
        score=1.0 if target else 0.0,
        negative_reasons=tuple(reasons),
        excluded_reason=None,
    )


def target_receipt_dict(target: PreCloseTeacherTarget) -> Mapping[str, object]:
    """Return stable telemetry keys for CSV/JSONL provenance."""

    return {
        "pre_close_candidate": target.pre_close_candidate,
        "close_latched_before_supervision": target.close_latched_before_supervision,
        "privileged_teacher_available": target.recordable,
        "privileged_close_ready_target": target.target,
        "privileged_close_ready_score": target.score,
        "privileged_close_ready_negative_reasons": target.negative_reasons,
        "privileged_close_ready_excluded_reason": target.excluded_reason,
        "close_readiness_target_schema": CLOSE_READINESS_TARGET_SCHEMA_VERSION,
    }
