# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Contracts for provenance-balanced privileged CLOSE-readiness collection.

This module deliberately plans *geometry probes*, not labels.  A probe is a
small, reversible nominal-grasp offset applied to a contact-free, canonical
OPEN Candidate-A source state.  The unchanged live geometry oracle remains
the only authority allowed to assign ``teacher_target``.  In particular, an
offset name such as ``LATERAL_PLUS_2_MM`` must never be interpreted as a
pre-assigned positive or negative label.

The contract gives the collector enough immutable provenance to keep a pair
of boundary trials tied to the same source sample/family while keeping every
student input deployable (the frozen 128-D feature only).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BOUNDARY_PAIRED_PLAN_SCHEMA = "g2_stage1a_boundary_paired_plan_v1"
BOUNDARY_PAIRED_ROW_SCHEMA = "g2_stage1a_boundary_paired_teacher_row_v1"
BOUNDARY_PAIRED_CONTRACT_VERSION = "candidate_a_v2_preclose_boundary_paired_v1"
SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA = "g2_stage1a_signed_margin_boundary_plan_v2"
SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA = "g2_stage1a_signed_margin_teacher_row_v2"
SIGNED_MARGIN_BOUNDARY_CONTRACT_VERSION = "candidate_a_v2_preclose_signed_margin_v2"
SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA = "g2_stage1a_signed_margin_paired_clone_plan_v3"
SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA = "g2_stage1a_signed_margin_teacher_row_v3"
SIGNED_MARGIN_PAIRED_CLONE_CONTRACT_VERSION = "candidate_a_v2_preclose_signed_margin_paired_clone_v3"
# V4 is intentionally a new provenance domain.  It keeps V3's exact live
# privileged teacher unchanged, but makes the current deployable robot-state
# receipt durable at the *same sampled timestamp* as the frozen 128-D feature.
# Existing V3 evidence must never be relabelled or silently merged into it.
SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA = "g2_stage1a_signed_margin_paired_clone_state_plan_v4"
SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA = "g2_stage1a_signed_margin_teacher_state_row_v4"
SIGNED_MARGIN_PAIRED_CLONE_STATE_CONTRACT_VERSION = "candidate_a_v2_preclose_signed_margin_paired_clone_state_v4"
# V5 retains V4's causal state receipt and adds only an HDF5 wrist RGB-D
# reference captured from the exact same 25-Hz frame as the student packet.
SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA = "g2_stage1a_signed_margin_paired_clone_state_wrist_plan_v5"
SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA = "g2_stage1a_signed_margin_teacher_state_wrist_row_v5"
SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_CONTRACT_VERSION = "candidate_a_v2_preclose_signed_margin_paired_clone_state_wrist_v5"
# V6 is a diagnostic-only descendant of V5.  It preserves the exact V5
# deployable RGB-D receipt and adds *teacher-side* camera/segmentation
# evidence.  It is deliberately a new schema so none of the V5 family split
# or its canonical rows can be silently changed or merged.
SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA = "g2_stage1a_signed_margin_paired_clone_state_wrist_diagnostic_plan_v6"
SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA = "g2_stage1a_signed_margin_teacher_state_wrist_diagnostic_row_v6"
SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_CONTRACT_VERSION = "candidate_a_v2_preclose_signed_margin_paired_clone_state_wrist_diagnostic_v6"


class BoundaryPairedCollectionError(ValueError):
    """Raised before an ambiguous or unsafe paired probe can be used."""


@dataclass(frozen=True)
class BoundaryProbe:
    """One label-agnostic nominal target perturbation in robot-root metres."""

    variant_id: str
    nominal_offset_root_m: tuple[float, float, float]
    perturbation_type: str = "NOMINAL_ROOT_TRANSLATION"
    orientation_root_z_deg: float = 0.0

    def receipt(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "nominal_offset_root_m": list(self.nominal_offset_root_m),
            "perturbation_type": self.perturbation_type,
            "orientation_root_z_deg": self.orientation_root_z_deg,
            "label_assigned_by_plan": False,
        }


# These are the existing bounded diagnostic geometry ranges expressed in the
# robot-root frame.  They are probe amplitudes, not teacher thresholds:
# lateral/height remain within the established +/-4 mm and +/-2 mm envelope.
# The approach-axis perturbation is intentionally smaller (1 mm) and only
# changes the nominal EE target; it does not alter an asset, controller or
# gripper endpoint.
DEFAULT_BOUNDARY_PROBES: tuple[BoundaryProbe, ...] = (
    BoundaryProbe("CENTER", (0.0, 0.0, 0.0)),
    BoundaryProbe("LATERAL_PLUS_2MM", (0.0, 0.002, 0.0)),
    BoundaryProbe("LATERAL_MINUS_2MM", (0.0, -0.002, 0.0)),
    BoundaryProbe("HEIGHT_PLUS_1MM", (0.0, 0.0, 0.001)),
    BoundaryProbe("HEIGHT_MINUS_1MM", (0.0, 0.0, -0.001)),
    BoundaryProbe("APPROACH_PLUS_1MM", (0.001, 0.0, 0.0)),
    BoundaryProbe("APPROACH_MINUS_1MM", (-0.001, 0.0, 0.0)),
)

# V2 retains every original translation probe and adds only a bounded
# root-Z orientation target variation.  These are exploration inputs, never
# labels; the unchanged live pad/cube oracle determines every outcome.
SIGNED_MARGIN_BOUNDARY_PROBES: tuple[BoundaryProbe, ...] = (
    BoundaryProbe("CENTER", (0.0, 0.0, 0.0), "CENTER"),
    BoundaryProbe("CONTAINMENT_ROOT_Y_PLUS_2MM", (0.0, 0.002, 0.0), "GRIPPER_CUBE_LATERAL_ROOT_Y"),
    BoundaryProbe("CONTAINMENT_ROOT_Y_MINUS_2MM", (0.0, -0.002, 0.0), "GRIPPER_CUBE_LATERAL_ROOT_Y"),
    BoundaryProbe("HEIGHT_PLUS_1MM", (0.0, 0.0, 0.001), "EE_HEIGHT_ROOT_Z"),
    BoundaryProbe("HEIGHT_MINUS_1MM", (0.0, 0.0, -0.001), "EE_HEIGHT_ROOT_Z"),
    BoundaryProbe("RESIDUAL_APPROACH_PLUS_1MM", (0.001, 0.0, 0.0), "EE_GRASP_RESIDUAL_ROOT_X"),
    BoundaryProbe("RESIDUAL_APPROACH_MINUS_1MM", (-0.001, 0.0, 0.0), "EE_GRASP_RESIDUAL_ROOT_X"),
    BoundaryProbe("ORIENTATION_ROOT_Z_PLUS_5DEG", (0.0, 0.0, 0.0), "EE_ORIENTATION_ROOT_Z", 5.0),
    BoundaryProbe("ORIENTATION_ROOT_Z_MINUS_5DEG", (0.0, 0.0, 0.0), "EE_ORIENTATION_ROOT_Z", -5.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_family_id_from_sample_id(sample_id: str) -> str:
    """Return the catalog source identity without the HDF5-row suffix."""

    if not isinstance(sample_id, str) or "/row-" not in sample_id:
        raise BoundaryPairedCollectionError("SOURCE_SAMPLE_ID_NOT_CATALOG_ROW")
    family, row = sample_id.rsplit("/row-", 1)
    if not family or not row.isdecimal():
        raise BoundaryPairedCollectionError("SOURCE_SAMPLE_ID_FORMAT_INVALID")
    return family


def _finite_offset(value: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise BoundaryPairedCollectionError(f"{label}_SHAPE_INVALID")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise BoundaryPairedCollectionError(f"{label}_NONFINITE")
    # Preserve the already-attested bounded geometry diagnostic scope.  The
    # plan cannot be used to generate a large or unsafe target excursion.
    if abs(result[0]) > 0.001 + 1.0e-12:
        raise BoundaryPairedCollectionError("APPROACH_OFFSET_OUT_OF_SCOPE")
    if abs(result[1]) > 0.004 + 1.0e-12:
        raise BoundaryPairedCollectionError("LATERAL_OFFSET_OUT_OF_SCOPE")
    if abs(result[2]) > 0.002 + 1.0e-12:
        raise BoundaryPairedCollectionError("HEIGHT_OFFSET_OUT_OF_SCOPE")
    return result


def _probe_from_raw(raw: Mapping[str, Any], *, signed_margin_plan: bool) -> BoundaryProbe:
    if not isinstance(raw, Mapping):
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PROBE_NOT_MAPPING")
    variant = raw.get("variant_id")
    if not isinstance(variant, str) or not variant:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_VARIANT_INVALID")
    if "teacher_target" in raw or "intended_target" in raw:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PLAN_LABEL_FORBIDDEN")
    perturbation_type = raw.get("perturbation_type", "NOMINAL_ROOT_TRANSLATION")
    orientation_root_z_deg = raw.get("orientation_root_z_deg", 0.0)
    if not isinstance(perturbation_type, str) or not perturbation_type:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PERTURBATION_TYPE_INVALID")
    try:
        orientation_root_z_deg = float(orientation_root_z_deg)
    except (TypeError, ValueError) as error:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_ORIENTATION_OFFSET_INVALID") from error
    if not math.isfinite(orientation_root_z_deg) or abs(orientation_root_z_deg) > 5.0 + 1.0e-12:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_ORIENTATION_OFFSET_OUT_OF_SCOPE")
    if not signed_margin_plan and orientation_root_z_deg != 0.0:
        raise BoundaryPairedCollectionError("V1_BOUNDARY_PLAN_ORIENTATION_OFFSET_FORBIDDEN")
    return BoundaryProbe(
        variant,
        _finite_offset(raw.get("nominal_offset_root_m"), label=variant),
        perturbation_type,
        orientation_root_z_deg,
    )


def load_boundary_paired_plan(
    path: Path,
    *,
    source_catalog: Path | None = None,
) -> dict[str, Any]:
    """Load a hash-attested, label-agnostic paired collection plan."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PLAN_MISSING")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PLAN_UNREADABLE") from error
    if not isinstance(payload, Mapping):
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PLAN_NOT_MAPPING")
    schema = payload.get("schema")
    signed_margin_plan = schema in {
        SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
    }
    paired_clone_plan = schema in {
        SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
    }
    causal_state_receipt = schema in {
        SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
    }
    wrist_rgbd_receipt = schema in {
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
    }
    v6_diagnostic_receipt = schema == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA
    if (
        schema not in {
            BOUNDARY_PAIRED_PLAN_SCHEMA,
            SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA,
            SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA,
            SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA,
            SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
            SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
        }
        or payload.get("contract_version") != (
            (
                (
                    (
                        SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_CONTRACT_VERSION
                        if v6_diagnostic_receipt
                        else SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_CONTRACT_VERSION
                        if wrist_rgbd_receipt
                        else SIGNED_MARGIN_PAIRED_CLONE_STATE_CONTRACT_VERSION
                    ) if causal_state_receipt
                    else SIGNED_MARGIN_PAIRED_CLONE_CONTRACT_VERSION
                ) if paired_clone_plan else SIGNED_MARGIN_BOUNDARY_CONTRACT_VERSION
            ) if signed_margin_plan else BOUNDARY_PAIRED_CONTRACT_VERSION
        )
        or payload.get("teacher_label_authority")
        != "UNCHANGED_LIVE_PRIVILEGED_GEOMETRY_PREDICATE"
        or payload.get("student_privileged_input_count") != 0
        or payload.get("sac_update") != 0
        or payload.get("student_optimizer_update") != 0
        or payload.get("privileged_hard_gate") is not False
        or payload.get("wandb_training_run") is not False
        or (
            causal_state_receipt
            and payload.get("causal_student_state_receipt") is not True
        )
        or (
            wrist_rgbd_receipt
            and payload.get("wrist_rgbd_receipt") is not True
        )
        or (
            v6_diagnostic_receipt
            and (
                payload.get("v6_diagnostic_receipt") is not True
                or payload.get("v6_diagnostic_student_input") is not False
            )
        )
    ):
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PLAN_CONTRACT_INVALID")
    if source_catalog is not None:
        catalog = Path(source_catalog).resolve()
        if not catalog.is_file() or payload.get("source_catalog_sha256") != sha256(catalog):
            raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_SOURCE_CATALOG_HASH_MISMATCH")
        if payload.get("source_catalog_path") != str(catalog):
            raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_SOURCE_CATALOG_PATH_MISMATCH")
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list) or len(raw_probes) < 2:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_PROBES_INSUFFICIENT")
    source_family_ids = payload.get("source_family_ids")
    if signed_margin_plan:
        allocation_valid = (
            isinstance(source_family_ids, list)
            and len(source_family_ids) == 10
            and all(isinstance(item, str) and item for item in source_family_ids)
        )
        if paired_clone_plan:
            allocation_valid = bool(
                allocation_valid
                and len(set(source_family_ids)) == 5
                and all(source_family_ids.count(item) == 2 for item in set(source_family_ids))
                and payload.get("paired_clone_allocation") is True
                and payload.get("source_row_selection")
                == "NEAREST_STRICTLY_OUTSIDE_30MM_HANDOFF_FROM_FROZEN_CATALOG"
            )
            initial_probe_indices = payload.get("initial_probe_index_by_env")
            if (
                not isinstance(initial_probe_indices, list)
                or len(initial_probe_indices) != 10
                or any(type(item) is not int or item not in (0, 1) for item in initial_probe_indices)
                or any(
                    tuple(source_family_ids[position:position + 2]) != (source_family_ids[position], source_family_ids[position])
                    or tuple(initial_probe_indices[position:position + 2]) != (0, 1)
                    for position in range(0, 10, 2)
                )
            ):
                allocation_valid = False
        else:
            allocation_valid = bool(allocation_valid and len(set(source_family_ids)) == 10)
        if not allocation_valid:
            raise BoundaryPairedCollectionError("SIGNED_MARGIN_SOURCE_FAMILY_ALLOCATION_INVALID")
        if source_catalog is not None:
            catalog_payload = json.loads(Path(source_catalog).read_text(encoding="utf-8"))
            catalog_ids = {
                str(item.get("source_id"))
                for item in catalog_payload.get("sources", [])
                if isinstance(item, Mapping)
            }
            if not set(source_family_ids) <= catalog_ids:
                raise BoundaryPairedCollectionError("SIGNED_MARGIN_SOURCE_FAMILY_NOT_IN_CATALOG")
    variants: set[str] = set()
    probes: list[BoundaryProbe] = []
    for raw in raw_probes:
        probe = _probe_from_raw(raw, signed_margin_plan=signed_margin_plan)
        if probe.variant_id in variants:
            raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_VARIANT_INVALID")
        probes.append(probe)
        variants.add(probe.variant_id)
    result = dict(payload)
    result["probes"] = tuple(probes)
    result["plan_path"] = str(resolved)
    result["plan_sha256"] = sha256(resolved)
    result["signed_margin_telemetry"] = signed_margin_plan
    result["row_schema"] = (
        (
            (
                (
                    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA
                    if v6_diagnostic_receipt
                    else SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA
                    if wrist_rgbd_receipt
                    else SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA
                ) if causal_state_receipt else SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA
            ) if paired_clone_plan else SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA
        ) if signed_margin_plan
        else BOUNDARY_PAIRED_ROW_SCHEMA
    )
    result["paired_clone_allocation"] = paired_clone_plan
    result["causal_student_state_receipt"] = causal_state_receipt
    result["wrist_rgbd_receipt"] = wrist_rgbd_receipt
    result["v6_diagnostic_receipt"] = v6_diagnostic_receipt
    return result


def build_boundary_paired_plan(*, source_catalog: Path) -> dict[str, Any]:
    """Return a plan whose only source authority is a frozen source catalog."""

    catalog = Path(source_catalog).resolve()
    if not catalog.is_file():
        raise BoundaryPairedCollectionError("SOURCE_CATALOG_MISSING")
    try:
        payload = json.loads(catalog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BoundaryPairedCollectionError("SOURCE_CATALOG_UNREADABLE") from error
    sources = payload.get("sources") if isinstance(payload, Mapping) else None
    if (
        not isinstance(sources, list)
        or len(sources) < 3
        or any(
            not isinstance(source, Mapping)
            or not isinstance(source.get("source_id"), str)
            or not source.get("source_id")
            for source in sources
        )
    ):
        raise BoundaryPairedCollectionError("SOURCE_CATALOG_PROVENANCE_INSUFFICIENT")
    return {
        "schema": BOUNDARY_PAIRED_PLAN_SCHEMA,
        "contract_version": BOUNDARY_PAIRED_CONTRACT_VERSION,
        "source_catalog_path": str(catalog),
        "source_catalog_sha256": sha256(catalog),
        "source_family_count": len(sources),
        "source_family_ids": [str(source["source_id"]) for source in sources],
        "source_selection_authority": "FROZEN_CANDIDATE_A_CONTACT_FREE_OPEN_SOURCE_CATALOG",
        "teacher_label_authority": "UNCHANGED_LIVE_PRIVILEGED_GEOMETRY_PREDICATE",
        "label_assigned_by_plan": False,
        "paired_unit": "SAME_SOURCE_SAMPLE_REPLAYED_ACROSS_LABEL_AGNOSTIC_PROBES",
        "contact_free_at_direct_init_required": True,
        "canonical_open_at_direct_init_required": True,
        "owner_geometry_valid_required": True,
        "forbidden_collision_required": False,
        "sac_update": 0,
        "student_optimizer_update": 0,
        "privileged_hard_gate": False,
        "student_privileged_input_count": 0,
        "wandb_training_run": False,
        "probes": [probe.receipt() for probe in DEFAULT_BOUNDARY_PROBES],
    }


def build_signed_margin_boundary_plan(
    *,
    source_catalog: Path,
    source_family_ids: Sequence[str],
) -> dict[str, Any]:
    """Build one V2 exact-telemetry plan for ten allocated source families.

    The caller decides allocation solely from the frozen catalog.  This plan
    never sees a teacher label, a result, or a learned score.
    """

    catalog = Path(source_catalog).resolve()
    base = build_boundary_paired_plan(source_catalog=catalog)
    requested = tuple(str(item) for item in source_family_ids)
    known = set(base["source_family_ids"])
    if len(requested) != 10 or len(set(requested)) != 10 or not set(requested) <= known:
        raise BoundaryPairedCollectionError("SIGNED_MARGIN_SOURCE_FAMILY_ALLOCATION_INVALID")
    return {
        **base,
        "schema": SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA,
        "contract_version": SIGNED_MARGIN_BOUNDARY_CONTRACT_VERSION,
        "row_schema": SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
        "signed_margin_telemetry": True,
        "source_family_ids": list(requested),
        "source_family_allocation_authority": "FROZEN_CATALOG_HASH_DETERMINISTIC_BATCH_ALLOCATION",
        "probes": [probe.receipt() for probe in SIGNED_MARGIN_BOUNDARY_PROBES],
    }


def build_signed_margin_paired_clone_plan(
    *,
    source_catalog: Path,
    source_family_ids: Sequence[str],
    comparison_probe_variant: str,
) -> dict[str, Any]:
    """Build a two-clone-per-source exact-margin plan.

    A 3K collection commonly reaches the pre-CLOSE window only once per
    clone.  Sequentially advancing one clone through nine probes therefore
    produced centre-only data.  This V3 collection uses two independent
    simulator clones initialized from the *same immutable source row*: one
    ``CENTER`` clone and one bounded comparison-probe clone.  It changes only
    collection scheduling, never a runtime threshold or a teacher label.
    """

    catalog = Path(source_catalog).resolve()
    base = build_boundary_paired_plan(source_catalog=catalog)
    families = tuple(str(item) for item in source_family_ids)
    known = set(base["source_family_ids"])
    if len(families) != 5 or len(set(families)) != 5 or not set(families) <= known:
        raise BoundaryPairedCollectionError("SIGNED_MARGIN_PAIRED_CLONE_FAMILY_ALLOCATION_INVALID")
    probes_by_id = {probe.variant_id: probe for probe in SIGNED_MARGIN_BOUNDARY_PROBES}
    if (
        comparison_probe_variant == "CENTER"
        or comparison_probe_variant not in probes_by_id
    ):
        raise BoundaryPairedCollectionError("SIGNED_MARGIN_PAIRED_CLONE_PROBE_INVALID")
    probes = (probes_by_id["CENTER"], probes_by_id[comparison_probe_variant])
    per_env_families = [family for family in families for _ in range(2)]
    return {
        **base,
        "schema": SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA,
        "contract_version": SIGNED_MARGIN_PAIRED_CLONE_CONTRACT_VERSION,
        "row_schema": SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
        "signed_margin_telemetry": True,
        "paired_clone_allocation": True,
        "source_family_ids": per_env_families,
        "source_family_unique_ids": list(families),
        "source_family_allocation_authority": "FROZEN_CATALOG_HASH_DETERMINISTIC_BATCH_ALLOCATION",
        "clone_pairing_authority": "SAME_IMMUTABLE_SOURCE_ROW_CENTER_AND_COMPARISON_PROBE",
        "source_row_selection": "NEAREST_STRICTLY_OUTSIDE_30MM_HANDOFF_FROM_FROZEN_CATALOG",
        "initial_probe_index_by_env": [index for _ in families for index in (0, 1)],
        "probes": [probe.receipt() for probe in probes],
    }


def build_signed_margin_paired_clone_state_plan(
    *,
    source_catalog: Path,
    source_family_ids: Sequence[str],
    comparison_probe_variant: str,
) -> dict[str, Any]:
    """Build the V4 paired-clone plan with timestamp-aligned causal state.

    This changes recording schema only.  It neither changes the geometry
    predicate nor exposes privileged geometry to a student input.
    """

    base = build_signed_margin_paired_clone_plan(
        source_catalog=source_catalog,
        source_family_ids=source_family_ids,
        comparison_probe_variant=comparison_probe_variant,
    )
    return {
        **base,
        "schema": SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA,
        "contract_version": SIGNED_MARGIN_PAIRED_CLONE_STATE_CONTRACT_VERSION,
        "row_schema": SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA,
        "causal_student_state_receipt": True,
        "student_state_authority": "CURRENT_DEPLOYABLE_ROBOT_STATE_AT_STUDENT_FEATURE_TIMESTAMP",
        "student_state_forbidden_fields": [
            "cube_gt",
            "relative_pose_root_m",
            "privileged_geometry",
        ],
    }


def build_signed_margin_paired_clone_state_wrist_plan(
    *,
    source_catalog: Path,
    source_family_ids: Sequence[str],
    comparison_probe_variant: str,
) -> dict[str, Any]:
    """Build the V5 plan with exact timestamp-aligned Wrist RGB-D refs."""

    base = build_signed_margin_paired_clone_state_plan(
        source_catalog=source_catalog,
        source_family_ids=source_family_ids,
        comparison_probe_variant=comparison_probe_variant,
    )
    return {
        **base,
        "schema": SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
        "contract_version": SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_CONTRACT_VERSION,
        "row_schema": SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
        "wrist_rgbd_receipt": True,
        "wrist_rgbd_authority": "CURRENT_25HZ_RIGHT_WRIST_SENSOR_FRAME_REFERENCED_BY_50HZ_CONTROL_ROW",
        "wrist_rgbd_student_fields": ["rgb", "depth_m", "depth_valid"],
        "wrist_rgbd_forbidden_student_fields": [
            "cube_gt", "relative_pose_root_m", "privileged_geometry"
        ],
    }


def build_signed_margin_paired_clone_state_wrist_v6_plan(
    *,
    source_catalog: Path,
    source_family_ids: Sequence[str],
    comparison_probe_variant: str,
) -> dict[str, Any]:
    """Build V6 diagnostic collection without changing V5's authority.

    The sole differences from V5 are durable camera-pose, segmentation, and
    GT-visibility receipts.  They are diagnostics/teacher fields and are
    explicitly excluded from student inputs and runtime CLOSE authority.
    """

    base = build_signed_margin_paired_clone_state_wrist_plan(
        source_catalog=source_catalog,
        source_family_ids=source_family_ids,
        comparison_probe_variant=comparison_probe_variant,
    )
    return {
        **base,
        "schema": SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
        "contract_version": SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_CONTRACT_VERSION,
        "row_schema": SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
        "v6_diagnostic_receipt": True,
        "v6_diagnostic_authority": (
            "LIVE_RIGHT_WRIST_CAMERA_RGBD_INSTANCE_SEMANTIC_AND_AUTHORED_OPTICAL_POSE"
        ),
        "v6_diagnostic_student_input": False,
        "student_privileged_input_count": 0,
    }


def apply_probe_to_nominal_pose(
    nominal_pose_root_m_xyzw: Sequence[float], probe: BoundaryProbe
) -> tuple[float, ...]:
    """Return only a bounded nominal-EE target variant; no source is mutated."""

    nominal = np.asarray(nominal_pose_root_m_xyzw, dtype=np.float64)
    if nominal.shape != (7,) or not np.isfinite(nominal).all():
        raise BoundaryPairedCollectionError("NOMINAL_POSE_INVALID")
    updated = nominal.copy()
    updated[:3] += np.asarray(probe.nominal_offset_root_m, dtype=np.float64)
    if probe.orientation_root_z_deg:
        half = math.radians(probe.orientation_root_z_deg) / 2.0
        delta = np.asarray((0.0, 0.0, math.sin(half), math.cos(half)), dtype=np.float64)
        x1, y1, z1, w1 = delta
        x2, y2, z2, w2 = updated[3:]
        rotated = np.asarray((
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ), dtype=np.float64)
        norm = float(np.linalg.norm(rotated))
        if norm <= 1.0e-12:
            raise BoundaryPairedCollectionError("BOUNDARY_PAIRED_ORIENTATION_COMPOSE_DEGENERATE")
        updated[3:] = rotated / norm
    return tuple(float(value) for value in updated)


def boundary_pair_id(*, source_sample_id: str, generation: int) -> str:
    """Stable source-sample pair identity, independent of teacher outcome."""

    if type(generation) is not int or generation < 0:
        raise BoundaryPairedCollectionError("BOUNDARY_PAIR_GENERATION_INVALID")
    source_family_id_from_sample_id(source_sample_id)
    return f"{source_sample_id}::pair-{generation:06d}"


__all__ = [
    "BOUNDARY_PAIRED_CONTRACT_VERSION",
    "BOUNDARY_PAIRED_PLAN_SCHEMA",
    "BOUNDARY_PAIRED_ROW_SCHEMA",
    "SIGNED_MARGIN_BOUNDARY_CONTRACT_VERSION",
    "SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA",
    "SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_CONTRACT_VERSION",
    "SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_CONTRACT_VERSION",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_PLAN_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_ROW_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_CONTRACT_VERSION",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_CONTRACT_VERSION",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA",
    "SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA",
    "BoundaryPairedCollectionError",
    "BoundaryProbe",
    "DEFAULT_BOUNDARY_PROBES",
    "SIGNED_MARGIN_BOUNDARY_PROBES",
    "apply_probe_to_nominal_pose",
    "boundary_pair_id",
    "build_boundary_paired_plan",
    "build_signed_margin_boundary_plan",
    "build_signed_margin_paired_clone_plan",
    "build_signed_margin_paired_clone_state_plan",
    "build_signed_margin_paired_clone_state_wrist_plan",
    "build_signed_margin_paired_clone_state_wrist_v6_plan",
    "load_boundary_paired_plan",
    "sha256",
    "source_family_id_from_sample_id",
]
