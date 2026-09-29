# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import json
from pathlib import Path

from geniesim.rl.sac.stage1a_vector_initial_states import (
    VECTOR_INITIAL_STATE_SCHEMA,
    select_stage1a_vector_initial_states,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "artifacts/g2_stage1a_preclose_collection_source_catalog_20260928_v1.json"


def test_vector_initial_state_selector_uses_ten_distinct_immutable_open_rows() -> None:
    selection = select_stage1a_vector_initial_states(num_envs=10)
    receipt = selection.receipt()
    assert receipt["schema"] == VECTOR_INITIAL_STATE_SCHEMA
    assert receipt["sample_count"] == 10
    assert receipt["distinct_source_sample_ids"] is True
    assert receipt["gripper_open_all"] is True
    assert receipt["writes_to_source_artifact"] is False
    assert len(set(receipt["source_row_indices"])) == 10
    assert selection.source_candidate_count >= 10


def test_seeded_collection_selector_covers_disjoint_approved_state_strata() -> None:
    selection = select_stage1a_vector_initial_states(
        num_envs=10,
        selection_seed=20260932,
    )
    receipt = selection.receipt()
    indices = receipt["source_row_indices"]
    assert receipt["distinct_source_sample_ids"] is True
    assert "STRATIFIED_ACROSS_ALL_VALID" in receipt["selection_rule"]
    assert "WITHOUT_TEACHER_OR_OUTCOME_LOOKUP" in receipt["selection_rule"]
    assert len(set(indices)) == 10
    # One sample per disjoint stratum must span more than the former narrow
    # near-handoff tail while still using only attested source rows.
    assert max(indices) - min(indices) > selection.source_candidate_count // 2


def test_explicit_signed_margin_family_allocation_is_exact_and_label_agnostic() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    families = tuple(source["source_id"] for source in catalog["sources"][:10])
    selection = select_stage1a_vector_initial_states(
        num_envs=10,
        selection_seed=20260929,
        collection_source_catalog=CATALOG,
        collection_source_family_ids=families,
    )
    selected_families = {sample.sample_id.rsplit("/row-", 1)[0] for sample in selection.samples}
    assert selected_families == set(families)
    assert "EXPLICIT_FAMILY_ALLOCATION" in selection.receipt()["selection_rule"]


def test_paired_clone_allocation_reuses_only_the_explicit_source_pair() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    families = tuple(source["source_id"] for source in catalog["sources"][:5])
    allocation = tuple(family for source in families for family in (source, source))
    selection = select_stage1a_vector_initial_states(
        num_envs=10,
        selection_seed=20260930,
        collection_source_catalog=CATALOG,
        collection_source_family_ids=allocation,
        collection_allow_duplicate_source_samples=True,
    )
    assert all(
        selection.samples[index].sample_id == selection.samples[index + 1].sample_id
        for index in range(0, 10, 2)
    )
    assert len({sample.sample_id for sample in selection.samples}) == 5
    assert "PAIRED_CLONE" in selection.receipt()["selection_rule"]
