# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

from geniesim.rl.sac.stage1a_close_readiness_contract import (
    CLOSE_READINESS_TARGET_SCHEMA_VERSION,
    build_pre_close_teacher_target,
    is_pre_close_candidate,
    target_receipt_dict,
)


def test_preclose_teacher_target_is_binary_and_keeps_geometry_reasons() -> None:
    target = build_pre_close_teacher_target(
        pre_close_candidate=True,
        close_latched_before_supervision=False,
        pad_geometry_ready=False,
        aperture_ready=False,
        orientation_ready=True,
        owner_valid=True,
        geometry_valid=True,
        no_safety_violation=True,
    )
    assert target.recordable is True
    assert target.target is False
    assert target.score == 0.0
    assert target.negative_reasons == (
        "PAD_GEOMETRY_NOT_READY",
        "APERTURE_INCOMPATIBLE",
    )
    receipt = target_receipt_dict(target)
    assert receipt["close_readiness_target_schema"] == CLOSE_READINESS_TARGET_SCHEMA_VERSION
    assert receipt["privileged_close_ready_score"] == 0.0


def test_safety_and_postclose_are_excluded_not_geometry_negatives() -> None:
    safety = build_pre_close_teacher_target(
        pre_close_candidate=True,
        close_latched_before_supervision=False,
        pad_geometry_ready=False,
        aperture_ready=False,
        orientation_ready=False,
        owner_valid=False,
        geometry_valid=False,
        no_safety_violation=False,
    )
    assert safety.recordable is False
    assert safety.excluded_reason == "SAFETY_EXCLUDED"
    assert safety.target is None
    assert safety.negative_reasons == ()

    postclose = build_pre_close_teacher_target(
        pre_close_candidate=True,
        close_latched_before_supervision=True,
        pad_geometry_ready=True,
        aperture_ready=True,
        orientation_ready=True,
        owner_valid=True,
        geometry_valid=True,
        no_safety_violation=True,
    )
    assert postclose.recordable is False
    assert postclose.excluded_reason == "POST_CLOSE_LATCHED"
    assert is_pre_close_candidate(
        phase="LOCAL_GRASP", close_latched_before_supervision=False
    ) is True
    assert is_pre_close_candidate(
        phase="LOCAL_GRASP", close_latched_before_supervision=True
    ) is False
