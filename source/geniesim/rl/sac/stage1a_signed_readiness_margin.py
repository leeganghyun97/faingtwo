# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure contract for a privileged signed CLOSE-readiness target.

This module intentionally has no Isaac, policy, or optimizer dependency.  It
defines the only continuous teacher target that may later supervise a
deployable student: the normalized signed distance to the *existing*
privileged CLOSE-admission predicates.  A positive target is on the ready
side, a negative target is on the not-ready side, and zero is the exact
predicate boundary.

Boolean predicate values are not substituted with arbitrary signed numbers.
In particular, ``cube_between_primary_pads`` requires the signed containment
distance along the live primary-pad grasp axis.  If that receipt was not
captured, a full signed-margin target is unavailable rather than guessed from
the non-negative mesh-to-OBB gap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


ORIENTATION_LIMIT_DEG = 15.0
SIGNED_MARGIN_TARGET_SCHEMA = "g2_close_readiness_signed_margin_v1"


class SignedReadinessMarginError(ValueError):
    """Raised when a requested signed target has no physical authority."""


@dataclass(frozen=True)
class SignedReadinessMargin:
    """A teacher receipt whose aggregate is the restrictive predicate margin.

    ``pad_containment_margin`` is the signed centre-containment clearance in
    millimetres:

    ``min(cube_projection - inner_pad_plane, outer_pad_plane - cube_projection)``.

    The scalar is positive only while the cube centre is between the primary
    pads, zero at either inward pad plane, and negative outside.  It is not a
    surface gap.  All millimetre values are normalized by the simultaneously
    measured cube effective width; orientation is normalized by the existing
    15 degree admission limit.  Therefore the aggregate's zero retains the
    physical admission-boundary meaning across cube sizes.
    """

    schema: str
    recordable: bool
    exclusion_reason: str | None
    pad_containment_margin: float | None
    pad_surface_gap_margin: float | None
    pad_geometry_margin: float | None
    aperture_margin: float | None
    orientation_margin: float | None
    aggregate_margin: float | None
    most_restrictive_predicate: str | None


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SignedReadinessMarginError(f"{name}_NONFINITE")
    return value


def build_signed_readiness_margin(
    *,
    pad_containment_margin_mm: float | None,
    minimum_primary_pad_cube_gap_mm: float,
    gripper_aperture_mm: float,
    cube_effective_width_mm: float,
    orientation_error_deg: float,
    owner_valid: bool,
    geometry_valid: bool,
    no_safety_violation: bool,
) -> SignedReadinessMargin:
    """Build a conservative, physically normalized signed teacher margin.

    Owner/geometry/safety are hard validity predicates, not fabricated
    continuous scores.  Invalid rows are excluded, exactly as in the binary
    teacher contract.  ``pad_containment_margin_mm`` is mandatory because the
    existing Boolean containment flag has no magnitude and cannot be used to
    create a meaningful negative target.
    """

    if not all(isinstance(value, bool) for value in (
        owner_valid, geometry_valid, no_safety_violation,
    )):
        raise SignedReadinessMarginError("HARD_VALIDITY_PREDICATE_NOT_BOOL")
    if not owner_valid:
        return SignedReadinessMargin(
            schema=SIGNED_MARGIN_TARGET_SCHEMA, recordable=False,
            exclusion_reason="OWNER_INVALID", pad_containment_margin=None,
            pad_surface_gap_margin=None, pad_geometry_margin=None,
            aperture_margin=None, orientation_margin=None,
            aggregate_margin=None, most_restrictive_predicate=None,
        )
    if not geometry_valid:
        return SignedReadinessMargin(
            schema=SIGNED_MARGIN_TARGET_SCHEMA, recordable=False,
            exclusion_reason="GEOMETRY_INVALID", pad_containment_margin=None,
            pad_surface_gap_margin=None, pad_geometry_margin=None,
            aperture_margin=None, orientation_margin=None,
            aggregate_margin=None, most_restrictive_predicate=None,
        )
    if not no_safety_violation:
        return SignedReadinessMargin(
            schema=SIGNED_MARGIN_TARGET_SCHEMA, recordable=False,
            exclusion_reason="SAFETY_EXCLUDED", pad_containment_margin=None,
            pad_surface_gap_margin=None, pad_geometry_margin=None,
            aperture_margin=None, orientation_margin=None,
            aggregate_margin=None, most_restrictive_predicate=None,
        )
    if pad_containment_margin_mm is None:
        return SignedReadinessMargin(
            schema=SIGNED_MARGIN_TARGET_SCHEMA, recordable=False,
            exclusion_reason="PAD_CONTAINMENT_SIGNED_MARGIN_MISSING",
            pad_containment_margin=None, pad_surface_gap_margin=None,
            pad_geometry_margin=None, aperture_margin=None,
            orientation_margin=None, aggregate_margin=None,
            most_restrictive_predicate=None,
        )

    width = _finite("CUBE_EFFECTIVE_WIDTH_MM", cube_effective_width_mm)
    if width <= 0.0:
        raise SignedReadinessMarginError("CUBE_EFFECTIVE_WIDTH_MM_NONPOSITIVE")
    containment = _finite("PAD_CONTAINMENT_MARGIN_MM", pad_containment_margin_mm) / width
    # ``mesh_to_cube_obb_gap`` is non-negative under the current oracle.  It
    # remains an explicit term so a future signed mesh-gap authority can be
    # added without altering the aggregate target definition.
    surface_gap = _finite(
        "MINIMUM_PRIMARY_PAD_CUBE_GAP_MM", minimum_primary_pad_cube_gap_mm
    ) / width
    aperture = (
        _finite("GRIPPER_APERTURE_MM", gripper_aperture_mm) - width
    ) / width
    orientation = (
        ORIENTATION_LIMIT_DEG - _finite("ORIENTATION_ERROR_DEG", orientation_error_deg)
    ) / ORIENTATION_LIMIT_DEG
    components = {
        "PAD_CONTAINMENT": containment,
        "PAD_SURFACE_GAP": surface_gap,
        "APERTURE_COMPATIBILITY": aperture,
        "ORIENTATION": orientation,
    }
    predicate, aggregate = min(components.items(), key=lambda item: item[1])
    return SignedReadinessMargin(
        schema=SIGNED_MARGIN_TARGET_SCHEMA,
        recordable=True,
        exclusion_reason=None,
        pad_containment_margin=containment,
        pad_surface_gap_margin=surface_gap,
        pad_geometry_margin=min(containment, surface_gap),
        aperture_margin=aperture,
        orientation_margin=orientation,
        aggregate_margin=aggregate,
        most_restrictive_predicate=predicate,
    )


def receipt_dict(target: SignedReadinessMargin) -> Mapping[str, object]:
    """Produce durable, explicit field names for a future teacher receipt."""

    return {
        "signed_readiness_margin_schema": target.schema,
        "signed_readiness_margin_available": target.recordable,
        "signed_readiness_margin_exclusion_reason": target.exclusion_reason,
        "pad_containment_margin_normalized": target.pad_containment_margin,
        "pad_surface_gap_margin_normalized": target.pad_surface_gap_margin,
        "pad_geometry_margin_normalized": target.pad_geometry_margin,
        "aperture_compatibility_margin_normalized": target.aperture_margin,
        "orientation_margin_normalized": target.orientation_margin,
        "privileged_signed_readiness_margin": target.aggregate_margin,
        "signed_readiness_most_restrictive_predicate": target.most_restrictive_predicate,
    }


__all__ = (
    "ORIENTATION_LIMIT_DEG",
    "SIGNED_MARGIN_TARGET_SCHEMA",
    "SignedReadinessMargin",
    "SignedReadinessMarginError",
    "build_signed_readiness_margin",
    "receipt_dict",
)
