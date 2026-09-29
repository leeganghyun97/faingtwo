# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from geniesim.rl.sac.stage1a_boundary_paired_collection import (
    BOUNDARY_PAIRED_ROW_SCHEMA,
    DEFAULT_BOUNDARY_PROBES,
    SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA,
    SIGNED_MARGIN_BOUNDARY_PROBES,
    SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA,
    apply_probe_to_nominal_pose,
    build_boundary_paired_plan,
    build_signed_margin_boundary_plan,
    build_signed_margin_paired_clone_plan,
    build_signed_margin_paired_clone_state_wrist_plan,
    build_signed_margin_paired_clone_state_wrist_v6_plan,
    load_boundary_paired_plan,
)
from geniesim.rl.sac.stage1a_signed_readiness_margin import (
    SIGNED_MARGIN_TARGET_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "artifacts/g2_stage1a_preclose_collection_source_catalog_20260928_v1.json"
AUDIT = ROOT / "scripts/audit_g2_stage1a_boundary_paired_dataset.py"
FINALIZER = ROOT / "scripts/finalize_g2_stage1a_boundary_teacher_student.py"
SUPERVISOR = ROOT / "scripts/run_g2_stage1a_boundary_paired_collection_batches.py"
SIGNED_SUPERVISOR = ROOT / "scripts/run_g2_stage1a_signed_margin_boundary_collection.py"


def _audit_module():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("g2_boundary_paired_audit", AUDIT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def _finalizer_module():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("g2_boundary_student_finalizer", FINALIZER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def _raw_row(*, family: str, sample: str, pair: str, variant: str, target: bool, episode: str, index: int) -> dict[str, object]:
    feature = np.zeros(128, dtype=np.float64)
    feature[0] = 2.0 if target else -2.0
    feature[1] = float(index) / 1000.0
    return {
        "schema": BOUNDARY_PAIRED_ROW_SCHEMA,
        "row_id": f"row-{pair}-{index}",
        "env_id": index % 10,
        "episode_id": episode,
        "source_family_id": family,
        "source_sample_id": sample,
        "source_provenance": {
            "hdf5_path": f"/frozen/{family}.hdf5",
            "hdf5_sha256": "a" * 64,
            "planner_sidecar_sha256": "b" * 64,
            "catalog_sha256": "c" * 64,
        },
        "boundary_pair_id": pair,
        "boundary_variant_id": variant,
        "collection_batch_id": "batch-0",
        "boundary_nominal_offset_root_m": [0.0, 0.0, 0.0],
        "contact_free_at_direct_init": True,
        "canonical_open_at_direct_init": True,
        "forbidden_collision_before_supervision": False,
        "pre_close_candidate": True,
        "close_latched_before_supervision": False,
        "teacher_geometry_timestamp_s": float(index) / 25.0,
        "student_feature_timestamp_s": float(index) / 25.0,
        "gru_reset_generation": index,
        "privileged_close_ready_target": target,
        "privileged_close_ready_score": 1.0 if target else 0.0,
        "negative_reason": "NONE" if target else "PAD_GEOMETRY_NOT_READY",
        "inner_pad_cube_gap_mm": 1.0,
        "outer_pad_cube_gap_mm": 1.1,
        "minimum_primary_pad_cube_gap_mm": 1.0,
        "gripper_aperture_mm": 45.0,
        "cube_effective_width_mm": 40.0,
        "orientation_error_deg": 1.0,
        "cube_between_primary_pads": True,
        "aperture_geometrically_compatible": True,
        "owner_valid": True,
        "geometry_valid": True,
        "relative_pose_root_m": [0.02, 0.0, 0.0],
        "frozen_student_feature_128d": feature.tolist(),
        "student_privileged_input_count": 0,
    }


def test_plan_is_label_agnostic_and_hash_attested(tmp_path: Path) -> None:
    plan = build_boundary_paired_plan(source_catalog=CATALOG)
    assert plan["label_assigned_by_plan"] is False
    assert plan["source_family_count"] >= 3
    assert all("teacher_target" not in probe for probe in plan["probes"])
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_boundary_paired_plan(path, source_catalog=CATALOG)
    assert len(loaded["probes"]) == len(DEFAULT_BOUNDARY_PROBES)
    shifted = apply_probe_to_nominal_pose((0.5, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0), loaded["probes"][1])
    assert shifted[:3] == (0.5, 0.002, 0.7)


def test_signed_margin_plan_binds_exactly_ten_families_and_keeps_labels_live(tmp_path: Path) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    families = tuple(source["source_id"] for source in catalog["sources"][:10])
    plan = build_signed_margin_boundary_plan(
        source_catalog=CATALOG, source_family_ids=families
    )
    assert plan["schema"] == SIGNED_MARGIN_BOUNDARY_PLAN_SCHEMA
    assert plan["signed_margin_telemetry"] is True
    assert tuple(plan["source_family_ids"]) == families
    assert len(plan["probes"]) == len(SIGNED_MARGIN_BOUNDARY_PROBES)
    assert all(probe["label_assigned_by_plan"] is False for probe in plan["probes"])
    path = tmp_path / "signed-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_boundary_paired_plan(path, source_catalog=CATALOG)
    assert loaded["row_schema"] == SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA
    rotated = apply_probe_to_nominal_pose(
        (0.5, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
        next(item for item in loaded["probes"] if item.orientation_root_z_deg == 5.0),
    )
    assert np.isclose(rotated[5], np.sin(np.deg2rad(5.0) / 2.0))
    assert np.isclose(rotated[6], np.cos(np.deg2rad(5.0) / 2.0))


def test_signed_margin_paired_clone_plan_binds_same_source_to_two_probes(tmp_path: Path) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    families = tuple(source["source_id"] for source in catalog["sources"][:5])
    plan = build_signed_margin_paired_clone_plan(
        source_catalog=CATALOG,
        source_family_ids=families,
        comparison_probe_variant="CONTAINMENT_ROOT_Y_PLUS_2MM",
    )
    assert plan["schema"] == SIGNED_MARGIN_PAIRED_CLONE_PLAN_SCHEMA
    assert plan["row_schema"] == SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA
    assert plan["paired_clone_allocation"] is True
    assert plan["source_family_ids"] == [family for source in families for family in (source, source)]
    assert plan["initial_probe_index_by_env"] == [0, 1] * 5
    path = tmp_path / "paired-clone-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_boundary_paired_plan(path, source_catalog=CATALOG)
    assert loaded["paired_clone_allocation"] is True


def test_v5_plan_requires_timestamp_aligned_wrist_rgbd_without_student_geometry(
    tmp_path: Path,
) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    families = tuple(source["source_id"] for source in catalog["sources"][5:10])
    plan = build_signed_margin_paired_clone_state_wrist_plan(
        source_catalog=CATALOG,
        source_family_ids=families,
        comparison_probe_variant="CONTAINMENT_ROOT_Y_MINUS_2MM",
    )
    assert plan["schema"] == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_PLAN_SCHEMA
    assert plan["row_schema"] == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA
    assert plan["causal_student_state_receipt"] is True
    assert plan["wrist_rgbd_receipt"] is True
    assert plan["student_privileged_input_count"] == 0
    assert set(plan["wrist_rgbd_forbidden_student_fields"]) == {
        "cube_gt", "relative_pose_root_m", "privileged_geometry"
    }
    path = tmp_path / "v5-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_boundary_paired_plan(path, source_catalog=CATALOG)
    assert loaded["wrist_rgbd_receipt"] is True
    assert loaded["row_schema"] == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_ROW_SCHEMA


def test_v6_plan_is_a_separate_teacher_only_diagnostic_authority(tmp_path: Path) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    families = tuple(source["source_id"] for source in catalog["sources"][10:15])
    plan = build_signed_margin_paired_clone_state_wrist_v6_plan(
        source_catalog=CATALOG,
        source_family_ids=families,
        comparison_probe_variant="CONTAINMENT_ROOT_Y_PLUS_2MM",
    )
    assert plan["schema"] == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_PLAN_SCHEMA
    assert plan["row_schema"] == SIGNED_MARGIN_PAIRED_CLONE_STATE_WRIST_V6_ROW_SCHEMA
    assert plan["v6_diagnostic_receipt"] is True
    assert plan["v6_diagnostic_student_input"] is False
    assert plan["student_privileged_input_count"] == 0
    path = tmp_path / "v6-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_boundary_paired_plan(path, source_catalog=CATALOG)
    assert loaded["v6_diagnostic_receipt"] is True
    assert loaded["wrist_rgbd_receipt"] is True
    assert loaded["source_family_ids"] == [
        family for family in families for _ in range(2)
    ]


def test_signed_margin_rows_require_actual_containment_and_binary_parity(tmp_path: Path) -> None:
    raw = _raw_row(
        family="signed-family", sample="signed-family/row-000000",
        pair="signed-family/row-000000::pair-000000", variant="CENTER",
        target=True, episode="episode-signed", index=0,
    )
    raw.update({
        "schema": SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
        "signed_margin_telemetry_schema": SIGNED_MARGIN_TARGET_SCHEMA,
        "inner_containment_margin_mm": 2.0,
        "outer_containment_margin_mm": 1.0,
        "primary_pad_containment_margin_mm": 1.0,
        "min_pad_surface_gap_mm": 0.25,
        "aperture_margin_mm": 5.0,
        "orientation_margin_deg": 14.0,
        "nominal_residual_mm": 17.0,
        "inner_toward_world_m": 0.1,
        "outer_toward_world_m": 0.15,
        "cube_projection_world_m": 0.125,
        "safety_valid": True,
        "perturbation_type": "CENTER",
        "perturbation_value": {"nominal_offset_root_m": [0.0, 0.0, 0.0]},
    })
    path = tmp_path / "signed.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    audit_module = _audit_module()
    loaded = audit_module._load_rows(path)
    assert loaded[0]["signed_margin"] is not None
    invalid = dict(raw)
    invalid["privileged_close_ready_target"] = False
    invalid["privileged_close_ready_score"] = 0.0
    invalid["negative_reason"] = "PAD_GEOMETRY_NOT_READY"
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    try:
        audit_module._load_rows(bad_path)
    except audit_module.BoundaryAuditError as error:
        assert "SIGNED_MARGIN_BINARY_PARITY_INVALID" in str(error)
    else:
        raise AssertionError("V2 signed/binary parity mismatch must fail closed")


def test_audit_uses_atomic_pairs_and_spreads_source_families(tmp_path: Path) -> None:
    sidecar = tmp_path / "rows.jsonl"
    rows: list[dict[str, object]] = []
    index = 0
    # Eight source families make the requested strict source-family-disjoint
    # allocation possible (train=3, validation=3, heldout=2). Every pair
    # contains actual labels from both probe variants; pairs, not individual
    # rows, are the atomic split unit.
    for family_index in range(8):
        family = f"family-{family_index}"
        for pair_index in range(2):
            sample = f"{family}/row-{pair_index:06d}"
            pair = f"{sample}::pair-000000"
            for target, variant in ((True, "CENTER"), (False, "LATERAL_PLUS_2MM")):
                for duplicate in range(25):
                    rows.append(_raw_row(
                        family=family,
                        sample=sample,
                        pair=pair,
                        variant=variant,
                        target=target,
                        episode=f"episode-{pair_index}-{variant}",
                        index=index,
                    ))
                    index += 1
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = _audit_module().audit([sidecar])
    assert report["BOUNDARY_PAIRED_ROW_COUNT"] == len(rows)
    assert report["SOURCE_LABEL_CONFOUNDING"] == "NO"
    assert report["SOURCE_FAMILY_DISJOINT_SPLIT"] == "YES"
    assert report["PAIR_LEAKAGE_COUNT"] == 0
    assert report["TRAIN_POS_NEG"] == {"positive": 150, "negative": 150}
    assert report["VALIDATION_POS_NEG"] == {"positive": 150, "negative": 150}
    assert report["HELDOUT_POS_NEG"] == {"positive": 100, "negative": 100}
    assert report["OFFLINE_CONTRACT_PASS"] is True
    assert report["3K_AUTHORIZED"] == "YES"


def test_passed_boundary_audit_freezes_linear_student_without_runtime_update(tmp_path: Path) -> None:
    sidecar = tmp_path / "rows.jsonl"
    rows: list[dict[str, object]] = []
    index = 0
    for family_index in range(8):
        family = f"family-{family_index}"
        for target, variant in ((True, "CENTER"), (False, "LATERAL_PLUS_2MM")):
            for duplicate in range(50):
                sample = f"{family}/row-000000"
                rows.append(_raw_row(
                    family=family,
                    sample=sample,
                    pair=f"{sample}::pair-000000",
                    variant=variant,
                    target=target,
                    episode=f"episode-{variant}",
                    index=index,
                ))
                index += 1
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = _audit_module().audit([sidecar])
    assert audit["OFFLINE_CONTRACT_PASS"] is True
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    frozen = _finalizer_module().finalize(
        sidecars=[sidecar], audit_path=audit_path, output_dir=tmp_path / "frozen"
    )
    assert frozen["STUDENT_HEAD_FROZEN"] == "YES"
    assert frozen["STUDENT_OPTIMIZER_UPDATE"] == 0
    assert frozen["CALIBRATION_SOURCE"] == "VALIDATION_ONLY"
    assert Path(str(frozen["CONTRACT_PATH"])).is_file()
    assert Path(str(frozen["INITIALIZATION_PATH"])).is_file()


def test_vector_collector_keeps_boundary_probes_update_free_and_teacher_labeled() -> None:
    smoke = (ROOT / "source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts/run_g2_stage1a_vector_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "boundary_paired_plan: Mapping[str, Any] | None = None" in smoke
    assert "BOUNDARY_PAIRED_COLLECTION_REQUIRES_CATALOGGED_UPDATE_FREE_COLLECTION" in smoke
    assert '"student_optimizer_update": 0 if boundary_probes else None' in smoke
    assert '"privileged_hard_gate": False if boundary_probes else None' in smoke
    assert "boundary_pair_id(" in smoke
    assert "--boundary-paired-plan" in launcher
    assert "BOUNDARY_PAIRED_PLAN_REQUIRES_CATALOGGED_UPDATE_FREE_COLLECTION" in launcher


def test_startup_receipts_do_not_create_the_runtime_output_directory() -> None:
    runner = ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
    spec = importlib.util.spec_from_file_location("g2_vector_runner_receipts", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = ROOT / "artifacts" / "future_run" / "REPORT.json"
    assert module._preflight_receipt_path(report, kind="launch") == (
        ROOT / "artifacts" / "future_run_launch.json"
    )
    assert module._preflight_receipt_path(report, kind="physics_smoke") == (
        ROOT / "artifacts" / "future_run_physics_smoke.json"
    )


def test_collection_supervisor_uses_receipt_not_training_only_exit_code() -> None:
    spec = importlib.util.spec_from_file_location("g2_boundary_batch_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = {
        "COLLECTION_COMPLETED": True,
        "accepted_transitions": 3000,
        "PRECLOSE_COLLECTION_ONLY": True,
        "sac_update_count": 0,
        "student_optimizer_update_count": 0,
        "vector_contract_pass": True,
        "source_freeze_match": True,
        "boundary_paired_collection": {"enabled": True},
        "ACTION_BOUND_VIOLATION": 0,
        "GRIPPER_AUTHORITY_VIOLATION": 0,
        "RUNTIME_HARDSTOP": 0,
        "FORBIDDEN_COLLISION": 0,
        "failure_event_count": 0,
    }
    assert module._valid_collection_payload(valid) is True
    invalid = dict(valid)
    invalid["FORBIDDEN_COLLISION"] = 1
    assert module._valid_collection_payload(invalid) is False


def test_signed_margin_supervisor_uses_nonoverlapping_frozen_catalog_allocations() -> None:
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "g2_signed_margin_boundary_supervisor", SIGNED_SUPERVISOR
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    allocations = [module._allocation(CATALOG.resolve(), index) for index in range(1, 4)]
    assert all(len(item) == 5 for item in allocations)
    assert len(set().union(*map(set, allocations))) == 15
    families = allocations[0]
    valid = {
        "COLLECTION_COMPLETED": True,
        "accepted_transitions": 3000,
        "PRECLOSE_COLLECTION_ONLY": True,
        "sac_update_count": 0,
        "student_optimizer_update_count": 0,
        "vector_contract_pass": True,
        "source_freeze_match": True,
        "boundary_paired_collection": {
            "enabled": True,
            "schema": "g2_stage1a_signed_margin_paired_clone_plan_v3",
            "row_schema": "g2_stage1a_signed_margin_teacher_row_v3",
            "signed_margin_telemetry": True,
            "paired_clone_allocation": True,
            "source_family_ids": [
                family for family_id in families for family in (family_id, family_id)
            ],
            "privileged_hard_gate": False,
            "student_privileged_input_count": 0,
        },
        "ACTION_BOUND_VIOLATION": 0,
        "GRIPPER_AUTHORITY_VIOLATION": 0,
        "RUNTIME_HARDSTOP": 0,
        "FORBIDDEN_COLLISION": 0,
        "failure_event_count": 0,
    }
    assert module._signed_receipt_valid(valid, family_ids=families) is True
