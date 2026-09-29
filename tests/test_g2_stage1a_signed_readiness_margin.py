# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.rl.sac.stage1a_signed_readiness_margin import (
    SIGNED_MARGIN_TARGET_SCHEMA,
    build_signed_readiness_margin,
    receipt_dict,
)


def _target(**overrides):
    values = {
        "pad_containment_margin_mm": 5.0,
        "minimum_primary_pad_cube_gap_mm": 2.0,
        "gripper_aperture_mm": 52.0,
        "cube_effective_width_mm": 50.0,
        "orientation_error_deg": 10.0,
        "owner_valid": True,
        "geometry_valid": True,
        "no_safety_violation": True,
    }
    values.update(overrides)
    return build_signed_readiness_margin(**values)


def test_signed_margin_uses_restrictive_physical_predicate_and_zero_boundary():
    target = _target()
    # Values normalized by actual physical authority: containment 0.10,
    # surface gap 0.04, aperture 0.04, orientation 1/3.  The tied physical
    # minimum is still a real predicate boundary, never a fraction score.
    assert target.recordable is True
    assert target.aggregate_margin == 0.04
    assert target.most_restrictive_predicate in {
        "PAD_SURFACE_GAP", "APERTURE_COMPATIBILITY"
    }

    at_boundary = _target(gripper_aperture_mm=50.0)
    assert at_boundary.aggregate_margin == 0.0
    assert at_boundary.most_restrictive_predicate == "APERTURE_COMPATIBILITY"

    not_ready = _target(gripper_aperture_mm=45.0)
    assert not_ready.aggregate_margin == -0.1
    assert not_ready.most_restrictive_predicate == "APERTURE_COMPATIBILITY"


def test_signed_margin_refuses_to_invent_containment_distance_from_boolean():
    target = _target(pad_containment_margin_mm=None)
    assert target.recordable is False
    assert target.exclusion_reason == "PAD_CONTAINMENT_SIGNED_MARGIN_MISSING"
    receipt = receipt_dict(target)
    assert receipt["signed_readiness_margin_schema"] == SIGNED_MARGIN_TARGET_SCHEMA
    assert receipt["privileged_signed_readiness_margin"] is None


def test_signed_margin_excludes_boolean_validity_guards_instead_of_fabricating_scores():
    target = _target(owner_valid=False)
    assert target.recordable is False
    assert target.exclusion_reason == "OWNER_INVALID"
