# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable, distinct cuRobo source states for the Stage-1A vector runner.

This is deliberately a read-only selector.  It does not create Isaac, reset an
environment, plan, write an articulation target, or alter a source HDF5 file.
The selector exists because the scalar Stage-1A runner used one source row;
using that row ten times would make a vector rollout a hidden broadcast at
reset even if its later action rows differed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
    PregraspInitialState,
    selected_pregrasp_initial_state,
)
from geniesim.rl.isaaclab.g2_policy_branch.keyboard_v3_curobo_planner import (
    nominal_grasp_pose_root_m_xyzw_for_cube,
)
from geniesim.rl.sac.keyboard_grasp_contract import canonical_keyboard_grasp_contract


VECTOR_INITIAL_STATE_SCHEMA = "g2_stage1a_vector_initial_state_selection_v1"
COLLECTION_SOURCE_CATALOG_SCHEMA = "g2_stage1a_preclose_collection_source_catalog_v1"
_SOURCE_GUARD_ROWS = 3
# The reset-fixed vector route restores the source arm/cube state before its
# measured canonical OPEN trajectory has physically cleared the four-bar.  A
# source that was recorded OPEN but is closer than this observed closed-jaw
# clearance can make pad contact during that non-policy restore interval.
# This is a source-eligibility precondition, not a runtime CLOSE/safety
# threshold: it is evaluated solely from the immutable source EE/cube poses.
MEASURED_OPEN_RESTORE_MIN_EE_CUBE_CENTER_DISTANCE_M = 0.050


class Stage1AVectorInitialStateError(ValueError):
    """Raised before a non-distinct or non-authoritative source row is used."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packet_json(value: Any) -> dict[str, Any]:
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise Stage1AVectorInitialStateError("SOURCE_PACKET_RECEIPT_NOT_MAPPING")
    return parsed


@dataclass(frozen=True)
class VectorInitialStateSelection:
    """Distinct pre-contact source states and their shared source provenance."""

    samples: tuple[PregraspInitialState, ...]
    nominal_grasp_pose_root_m_xyzw: tuple[float, ...]
    source_hdf5_path: str
    source_hdf5_sha256: str
    source_candidate_count: int
    selection_rule: str
    schema: str = VECTOR_INITIAL_STATE_SCHEMA

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sample_count": len(self.samples),
            "sample_ids": [sample.sample_id for sample in self.samples],
            "source_row_indices": [int(sample.hdf5_row_index) for sample in self.samples],
            "source_control_steps": [int(sample.source_control_step) for sample in self.samples],
            "nominal_grasp_pose_root_m_xyzw": list(self.nominal_grasp_pose_root_m_xyzw),
            "source_hdf5_path": self.source_hdf5_path,
            "source_hdf5_sha256": self.source_hdf5_sha256,
            "source_candidate_count": int(self.source_candidate_count),
            "selection_rule": self.selection_rule,
            "distinct_source_sample_ids": len({sample.sample_id for sample in self.samples})
            == len(self.samples),
            "gripper_open_all": all(sample.gripper_state == "OPEN" for sample in self.samples),
            "writes_to_source_artifact": False,
        }


def _catalog_entries(catalog_path: Path) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Load an explicit, hash-attested collection-only source catalog."""

    resolved = catalog_path.resolve()
    if not resolved.is_file():
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_CATALOG_MISSING")
    catalog_sha256 = _sha256(resolved)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1AVectorInitialStateError(
            "COLLECTION_SOURCE_CATALOG_UNREADABLE"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != COLLECTION_SOURCE_CATALOG_SCHEMA
        or payload.get("collection_only") is not True
        or payload.get("teacher_or_outcome_lookup_used_for_selection") is not False
        or payload.get("baseline_variant") != "LEGACY_CANDIDATE_A_LEFT_ARM_DOWN_V2"
        or payload.get("candidate_asset_sha256")
        != "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
    ):
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_CATALOG_CONTRACT_INVALID")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(isinstance(item, Mapping) for item in sources):
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_CATALOG_SOURCES_INVALID")
    return catalog_sha256, tuple(sources)


def _catalog_source_candidates(
    *,
    base: PregraspInitialState,
    entry: Mapping[str, Any],
) -> tuple[list[PregraspInitialState], tuple[float, ...]]:
    """Return only source-attested Open/contact-free rows from one catalog entry."""

    required_entry = (
        "source_id",
        "hdf5_path",
        "hdf5_sha256",
        "planner_sidecar_path",
        "planner_sidecar_sha256",
        "cube_pose_robot_root_m_xyzw",
        "control_hz",
        "rgbd_hz",
        "candidate_asset_sha256",
    )
    if any(name not in entry for name in required_entry):
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_ENTRY_INCOMPLETE")
    if (
        int(entry["control_hz"]) != 50
        or int(entry["rgbd_hz"]) != 25
        or entry["candidate_asset_sha256"]
        != "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
    ):
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_ENTRY_CONTRACT_INVALID")
    source_path = Path(str(entry["hdf5_path"])).resolve()
    sidecar_path = Path(str(entry["planner_sidecar_path"])).resolve()
    expected_source_sha = str(entry["hdf5_sha256"])
    expected_sidecar_sha = str(entry["planner_sidecar_sha256"])
    if (
        not source_path.is_file()
        or not sidecar_path.is_file()
        or _sha256(source_path) != expected_source_sha
        or _sha256(sidecar_path) != expected_sidecar_sha
    ):
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_HASH_MISMATCH")
    cube_pose = tuple(float(value) for value in entry["cube_pose_robot_root_m_xyzw"])
    if len(cube_pose) != 7 or not np.isfinite(cube_pose).all():
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_CUBE_POSE_INVALID")
    nominal_pose = tuple(
        float(value)
        for value in nominal_grasp_pose_root_m_xyzw_for_cube(cube_pose[:3])
    )
    nominal_position = np.asarray(nominal_pose[:3], dtype=np.float64)
    handoff_m = float(canonical_keyboard_grasp_contract().curobo_default_handoff_m)
    with h5py.File(source_path, "r") as source:
        required = (
            "rows/control_step",
            "rows/ee_pose_robot_root_m_xyzw",
            "rows/gripper_state_open",
            "rows/packet_receipt_json",
            "rows/right_arm_joint_position_rad",
            "rows/right_arm_joint_velocity_rad_s",
            "rows/right_wrist_depth_valid",
            "rows/right_wrist_rgb",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise Stage1AVectorInitialStateError(
                f"COLLECTION_SOURCE_HDF5_SCHEMA_MISSING:{missing}"
            )
        rows = source["rows"]
        ee = np.asarray(rows["ee_pose_robot_root_m_xyzw"], dtype=np.float64)
        q = np.asarray(rows["right_arm_joint_position_rad"], dtype=np.float64)
        qd = np.asarray(rows["right_arm_joint_velocity_rad_s"], dtype=np.float64)
        steps = np.asarray(rows["control_step"], dtype=np.int64)
        gripper_open = np.asarray(rows["gripper_state_open"], dtype=np.float64).reshape(-1)
        packets = [_packet_json(value) for value in rows["packet_receipt_json"][:]]
        row_count = int(ee.shape[0])
        expected = (row_count, 7)
        if (
            ee.shape != expected
            or q.shape != expected
            or qd.shape != expected
            or steps.shape != (row_count,)
            or gripper_open.shape != (row_count,)
            or len(packets) != row_count
        ):
            raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_HDF5_SHAPE_INVALID")
        distance = np.linalg.norm(ee[:, :3] - nominal_position, axis=1)
        candidates: list[int] = []
        for index in range(row_count - _SOURCE_GUARD_ROWS):
            stop = index + _SOURCE_GUARD_ROWS + 1
            receipt_window = packets[index:stop]
            contact_free = all(
                receipt.get("contact") is False
                and receipt.get("forbidden_collision") is False
                and receipt.get("gripper_close") is False
                and receipt.get("terminal") is False
                for receipt in receipt_window
            )
            valid_depth = bool(np.asarray(rows["right_wrist_depth_valid"][index]).any())
            rgb = np.asarray(rows["right_wrist_rgb"][index])
            valid_rgb = rgb.ndim == 3 and rgb.shape[-1] == 3
            if (
                np.isfinite(ee[index:stop]).all()
                and np.isfinite(q[index:stop]).all()
                and np.isfinite(qd[index:stop]).all()
                and np.all(distance[index:stop] > handoff_m)
                and np.all(gripper_open[index:stop] == 1.0)
                and contact_free
                and valid_depth
                and valid_rgb
            ):
                candidates.append(index)
        source_id = str(entry["source_id"])
        return (
            [
                replace(
                    base,
                    sample_id=f"{source_id}/row-{index:06d}",
                    hdf5_path=str(source_path),
                    hdf5_sha256=expected_source_sha,
                    report_path=str(sidecar_path),
                    report_sha256=expected_sidecar_sha,
                    hdf5_row_index=int(index),
                    source_control_step=int(steps[index]),
                    right_arm_q_rad=tuple(float(value) for value in q[index]),
                    right_arm_qd_rad_s=tuple(float(value) for value in qd[index]),
                    ee_pose_robot_root_m_xyzw=tuple(float(value) for value in ee[index]),
                    cube_pose_robot_root_m_xyzw=cube_pose,
                )
                for index in candidates
            ],
            nominal_pose,
        )


def _select_catalog_collection_initial_states(
    *,
    base: PregraspInitialState,
    num_envs: int,
    selection_seed: int,
    catalog_path: Path,
    source_family_ids: tuple[str, ...] | None = None,
    allow_duplicate_source_samples: bool = False,
    prefer_nearest_handoff_source: bool = False,
) -> VectorInitialStateSelection:
    catalog_sha256, entries = _catalog_entries(catalog_path)
    if source_family_ids is not None:
        if (
            len(source_family_ids) != num_envs
            or not all(isinstance(item, str) and item for item in source_family_ids)
        ):
            raise Stage1AVectorInitialStateError(
                "COLLECTION_SOURCE_FAMILY_ALLOCATION_INVALID"
            )
        by_id = {str(entry.get("source_id")): entry for entry in entries}
        if not set(source_family_ids) <= set(by_id):
            raise Stage1AVectorInitialStateError(
                "COLLECTION_SOURCE_FAMILY_ALLOCATION_NOT_IN_CATALOG"
            )
        if len(set(source_family_ids)) != num_envs and not allow_duplicate_source_samples:
            raise Stage1AVectorInitialStateError(
                "COLLECTION_SOURCE_FAMILY_DUPLICATION_NOT_AUTHORIZED"
            )
        if allow_duplicate_source_samples:
            # Pair-clone boundary collection deliberately reuses one immutable
            # source row in two isolated simulation clones.  Select once per
            # family, then fan out the read-only state to its explicitly
            # allocated clone positions.  This is not a normal vector reset
            # and is forbidden unless a V3 paired-clone plan attests it.
            generator = np.random.default_rng(selection_seed)
            selected_by_family: dict[str, PregraspInitialState] = {}
            nominal_by_family: dict[str, tuple[float, ...]] = {}
            candidate_count = 0
            for family in dict.fromkeys(source_family_ids):
                candidates, nominal_pose = _catalog_source_candidates(
                    base=base, entry=by_id[family]
                )
                if not candidates:
                    raise Stage1AVectorInitialStateError(
                        "COLLECTION_SOURCE_HAS_NO_ELIGIBLE_ROWS"
                    )
                candidate_count += len(candidates)
                if prefer_nearest_handoff_source:
                    # The catalog already proves every candidate is clean,
                    # OPEN and strictly outside the 30-mm handoff.  For the
                    # boundary teacher collector choose the nearest such row
                    # so all paired clones can actually reach the same
                    # pre-CLOSE window inside a bounded 3K run.  This uses
                    # only immutable robot/cube geometry, never a teacher
                    # label, rollout result, or learned score.
                    nominal_position = np.asarray(nominal_pose[:3], dtype=np.float64)
                    selected_by_family[family] = min(
                        candidates,
                        key=lambda sample: (
                            float(np.linalg.norm(
                                np.asarray(sample.ee_pose_robot_root_m_xyzw[:3], dtype=np.float64)
                                - nominal_position
                            )),
                            int(sample.hdf5_row_index),
                        ),
                    )
                else:
                    selected_by_family[family] = candidates[
                        int(generator.integers(0, len(candidates)))
                    ]
                nominal_by_family[family] = nominal_pose
            samples = [selected_by_family[family] for family in source_family_ids]
            nominal_by_sample = [nominal_by_family[family] for family in source_family_ids]
            selection = VectorInitialStateSelection(
                samples=tuple(samples),
                nominal_grasp_pose_root_m_xyzw=nominal_by_sample[0],
                source_hdf5_path=f"COLLECTION_CATALOG:{catalog_path.resolve()}",
                source_hdf5_sha256=catalog_sha256,
                source_candidate_count=candidate_count,
                selection_rule=(
                    (
                        "EXPLICIT_PAIRED_CLONE_NEAREST_30MM_HANDOFF_SOURCE_FROM_"
                        "FROZEN_CATALOG_WITHOUT_TEACHER_OR_OUTCOME_LOOKUP"
                        if prefer_nearest_handoff_source else
                        "EXPLICIT_PAIRED_CLONE_FAMILY_ALLOCATION_FROM_FROZEN_CATALOG_"
                        "WITHOUT_TEACHER_OR_OUTCOME_LOOKUP"
                    )
                ),
            )
            receipt = selection.receipt()
            if receipt["distinct_source_sample_ids"] or not receipt["gripper_open_all"]:
                raise Stage1AVectorInitialStateError(
                    "COLLECTION_PAIRED_CLONE_SELECTION_VALIDATION_FAILED"
                )
            return selection
        entries = tuple(by_id[family] for family in source_family_ids)
    if len(entries) < num_envs:
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_CATALOG_TOO_SMALL")
    generator = np.random.default_rng(selection_seed)
    entry_positions = np.sort(generator.choice(len(entries), size=num_envs, replace=False))
    samples: list[PregraspInitialState] = []
    nominal_by_sample: list[tuple[float, ...]] = []
    candidate_count = 0
    for position in entry_positions:
        candidates, nominal_pose = _catalog_source_candidates(
            base=base,
            entry=entries[int(position)],
        )
        if not candidates:
            raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_HAS_NO_ELIGIBLE_ROWS")
        candidate_count += len(candidates)
        samples.append(candidates[int(generator.integers(0, len(candidates)))])
        nominal_by_sample.append(nominal_pose)
    if len({sample.sample_id for sample in samples}) != num_envs:
        raise Stage1AVectorInitialStateError("COLLECTION_CATALOG_SELECTION_DUPLICATE")
    selection = VectorInitialStateSelection(
        samples=tuple(samples),
        nominal_grasp_pose_root_m_xyzw=nominal_by_sample[0],
        source_hdf5_path=f"COLLECTION_CATALOG:{catalog_path.resolve()}",
        source_hdf5_sha256=catalog_sha256,
        source_candidate_count=candidate_count,
        selection_rule=(
            "EXPLICIT_FAMILY_ALLOCATION_FROM_FROZEN_CATALOG_"
            "WITHOUT_TEACHER_OR_OUTCOME_LOOKUP"
            if source_family_ids is not None else
            "SEEDED_WITHOUT_REPLACEMENT_ACROSS_FROZEN_CANDIDATE_A_V2_"
            "MULTISOURCE_CONTACT_FREE_OPEN_CATALOG_WITHOUT_TEACHER_OR_OUTCOME_LOOKUP"
        ),
    )
    receipt = selection.receipt()
    if not receipt["distinct_source_sample_ids"] or not receipt["gripper_open_all"]:
        raise Stage1AVectorInitialStateError("COLLECTION_CATALOG_SELECTION_VALIDATION_FAILED")
    return selection


def select_catalog_collection_sample_for_source_family(
    *,
    catalog_path: Path,
    source_family_id: str,
    selection_seed: int,
    exclude_sample_ids: tuple[str, ...] = (),
) -> PregraspInitialState:
    """Select one new immutable row from one already-attested source family.

    Boundary-paired collection replays several bounded nominal-target probes
    from the *same* source sample, then advances to a different HDF5 row in
    that same frozen source family.  Selection uses neither a teacher label
    nor a rollout outcome.  The function is intentionally read-only and does
    not create an Isaac environment.
    """

    if type(selection_seed) is not int:
        raise Stage1AVectorInitialStateError("COLLECTION_FAMILY_SEED_INVALID")
    if not isinstance(source_family_id, str) or not source_family_id:
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_FAMILY_INVALID")
    catalog_sha256, entries = _catalog_entries(Path(catalog_path))
    del catalog_sha256  # The caller freezes the catalog path/hash separately.
    matching = [entry for entry in entries if entry.get("source_id") == source_family_id]
    if len(matching) != 1:
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_FAMILY_NOT_UNIQUE")
    base = selected_pregrasp_initial_state()
    candidates, _nominal = _catalog_source_candidates(base=base, entry=matching[0])
    excluded = set(exclude_sample_ids)
    remaining = [sample for sample in candidates if sample.sample_id not in excluded]
    if not remaining:
        # A short source can legitimately have only one source-attested row.
        # Replaying it is explicit rather than silently falling back to a
        # different provenance family.
        remaining = candidates
    if not remaining:
        raise Stage1AVectorInitialStateError("COLLECTION_SOURCE_FAMILY_HAS_NO_ROWS")
    generator = np.random.default_rng(selection_seed)
    return remaining[int(generator.integers(0, len(remaining)))]


def select_stage1a_vector_initial_states(
    *,
    num_envs: int,
    selection_seed: int | None = None,
    collection_source_catalog: Path | None = None,
    collection_source_family_ids: tuple[str, ...] | None = None,
    collection_allow_duplicate_source_samples: bool = False,
    collection_prefer_nearest_handoff_source: bool = False,
    measured_open_restore_min_ee_cube_center_distance_m: float | None = None,
) -> VectorInitialStateSelection:
    """Select ``num_envs`` distinct, source-attested OPEN rows.

    Each selected row and its next three rows must remain contact-free, OPEN,
    and above the already-authoritative 30-mm handoff.  The normal runtime
    keeps its historical near-handoff tail selection exactly.  A
    collection-only seeded request instead covers the already-attested
    candidate route in disjoint state strata, without looking at a teacher
    label or modifying any source state.
    """

    if type(num_envs) is not int or num_envs <= 0:
        raise Stage1AVectorInitialStateError("NUM_ENVS_MUST_BE_POSITIVE_INT")
    if measured_open_restore_min_ee_cube_center_distance_m is not None and (
        not np.isfinite(measured_open_restore_min_ee_cube_center_distance_m)
        or float(measured_open_restore_min_ee_cube_center_distance_m) <= 0.0
    ):
        raise Stage1AVectorInitialStateError(
            "MEASURED_OPEN_RESTORE_SOURCE_DISTANCE_INVALID"
        )
    base = selected_pregrasp_initial_state()
    if collection_source_catalog is not None:
        if selection_seed is None:
            raise Stage1AVectorInitialStateError(
                "COLLECTION_SOURCE_CATALOG_REQUIRES_SEEDED_COLLECTION"
            )
        return _select_catalog_collection_initial_states(
            base=base,
            num_envs=num_envs,
            selection_seed=selection_seed,
            catalog_path=Path(collection_source_catalog),
            source_family_ids=collection_source_family_ids,
            allow_duplicate_source_samples=collection_allow_duplicate_source_samples,
            prefer_nearest_handoff_source=collection_prefer_nearest_handoff_source,
        )
    if (
        collection_source_family_ids is not None
        or collection_allow_duplicate_source_samples
        or collection_prefer_nearest_handoff_source
    ):
        raise Stage1AVectorInitialStateError(
            "COLLECTION_SOURCE_FAMILY_ALLOCATION_REQUIRES_CATALOG"
        )
    source_path = Path(base.hdf5_path).resolve()
    if not source_path.is_file() or _sha256(source_path) != base.hdf5_sha256:
        raise Stage1AVectorInitialStateError("SOURCE_HDF5_HASH_MISMATCH")
    nominal_pose = tuple(
        float(value)
        for value in nominal_grasp_pose_root_m_xyzw_for_cube(
            base.cube_pose_robot_root_m_xyzw[:3]
        )
    )
    nominal_position = np.asarray(nominal_pose[:3], dtype=np.float64)
    handoff_m = float(canonical_keyboard_grasp_contract().curobo_default_handoff_m)

    with h5py.File(source_path, "r") as source:
        required = (
            "rows/control_step",
            "rows/ee_pose_robot_root_m_xyzw",
            "rows/gripper_state_open",
            "rows/packet_receipt_json",
            "rows/right_arm_joint_position_rad",
            "rows/right_arm_joint_velocity_rad_s",
            "rows/right_wrist_depth_valid",
            "rows/right_wrist_rgb",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise Stage1AVectorInitialStateError(
                f"SOURCE_HDF5_SCHEMA_MISSING:{missing}"
            )
        rows = source["rows"]
        ee = np.asarray(rows["ee_pose_robot_root_m_xyzw"], dtype=np.float64)
        q = np.asarray(rows["right_arm_joint_position_rad"], dtype=np.float64)
        qd = np.asarray(rows["right_arm_joint_velocity_rad_s"], dtype=np.float64)
        steps = np.asarray(rows["control_step"], dtype=np.int64)
        gripper_open = np.asarray(rows["gripper_state_open"], dtype=np.float64).reshape(-1)
        packets = [_packet_json(value) for value in rows["packet_receipt_json"][:]]
        row_count = int(ee.shape[0])
        expected = (row_count, 7)
        if (
            ee.shape != expected
            or q.shape != expected
            or qd.shape != expected
            or steps.shape != (row_count,)
            or gripper_open.shape != (row_count,)
            or len(packets) != row_count
        ):
            raise Stage1AVectorInitialStateError("SOURCE_HDF5_SHAPE_INVALID")
        distance = np.linalg.norm(ee[:, :3] - nominal_position, axis=1)
        candidates: list[int] = []
        for index in range(row_count - _SOURCE_GUARD_ROWS):
            stop = index + _SOURCE_GUARD_ROWS + 1
            receipt_window = packets[index:stop]
            contact_free = all(
                receipt.get("contact") is False
                and receipt.get("forbidden_collision") is False
                and receipt.get("gripper_close") is False
                and receipt.get("terminal") is False
                for receipt in receipt_window
            )
            valid_depth = bool(np.asarray(rows["right_wrist_depth_valid"][index]).any())
            rgb = np.asarray(rows["right_wrist_rgb"][index])
            valid_rgb = rgb.ndim == 3 and rgb.shape[-1] == 3
            if (
                np.isfinite(ee[index:stop]).all()
                and np.isfinite(q[index:stop]).all()
                and np.isfinite(qd[index:stop]).all()
                and np.all(distance[index:stop] > handoff_m)
                and (
                    measured_open_restore_min_ee_cube_center_distance_m is None
                    or np.all(
                        distance[index:stop]
                        >= float(
                            measured_open_restore_min_ee_cube_center_distance_m
                        )
                    )
                )
                and np.all(gripper_open[index:stop] == 1.0)
                and contact_free
                and valid_depth
                and valid_rgb
            ):
                candidates.append(index)
        if len(candidates) < num_envs:
            raise Stage1AVectorInitialStateError(
                f"INSUFFICIENT_DISTINCT_SOURCE_ROWS:{len(candidates)}<{num_envs}"
            )
        # Keep every source row immutable.  Sampling from a near-handoff tail
        # yields useful FAR_REACH / LOCAL_GRASP diversity while retaining the
        # existing 30-mm safety guard window.
        tail_start = max(0, len(candidates) - max(num_envs * 4, num_envs))
        tail = candidates[tail_start:]
        if selection_seed is None:
            # Preserve the existing fixed runtime selection exactly.  This is
            # the default for training/evaluation and its receipt remains
            # byte-for-byte compatible with earlier vector artifacts.
            positions = np.rint(
                np.linspace(0, len(tail) - 1, num=num_envs)
            ).astype(int)
            chosen = [tail[int(position)] for position in positions]
            selection_rule = (
                "EVENLY_SUBSAMPLE_LAST_MAX_4X_NUM_ENVS_VALID_CONTACT_FREE_OPEN_"
                "ROWS_WITH_CURRENT_AND_NEXT_THREE_ROWS_ABOVE_EXISTING_30MM_HANDOFF"
                + (
                    "_AND_MEASURED_OPEN_RESTORE_SOURCE_CLEARANCE"
                    if measured_open_restore_min_ee_cube_center_distance_m is not None
                    else ""
                )
            )
        else:
            if type(selection_seed) is not int:
                raise Stage1AVectorInitialStateError("SELECTION_SEED_MUST_BE_INT_OR_NONE")
            # Collection-only diversity: select one source row from every
            # disjoint portion of the already-approved candidate route.  The
            # old tail-only sample repeatedly started the ten environments in
            # one narrow approach slice, which made natural pre-CLOSE labels
            # episode-pure.  This is state-distribution coverage only: every
            # selected row still satisfies the immutable OPEN/contact-free
            # and existing 30-mm guard above, and no geometry/teacher outcome
            # participates in selection.
            generator = np.random.default_rng(selection_seed)
            strata = np.array_split(np.asarray(candidates, dtype=np.int64), num_envs)
            if any(stratum.size == 0 for stratum in strata):
                raise Stage1AVectorInitialStateError("VECTOR_SOURCE_STRATUM_EMPTY")
            chosen = [
                int(stratum[int(generator.integers(0, stratum.size))])
                for stratum in strata
            ]
            selection_rule = (
                "SEEDED_STRATIFIED_ACROSS_ALL_VALID_CONTACT_FREE_OPEN_ROWS_"
                "WITH_CURRENT_AND_NEXT_THREE_ROWS_ABOVE_EXISTING_30MM_HANDOFF_"
                "WITHOUT_TEACHER_OR_OUTCOME_LOOKUP"
            )
        if len(set(chosen)) != num_envs:
            raise Stage1AVectorInitialStateError("VECTOR_SOURCE_SELECTION_DUPLICATE")
        samples = tuple(
            replace(
                base,
                sample_id=f"{source_path.parent.name}/row-{index:06d}",
                hdf5_row_index=int(index),
                source_control_step=int(steps[index]),
                right_arm_q_rad=tuple(float(value) for value in q[index]),
                right_arm_qd_rad_s=tuple(float(value) for value in qd[index]),
                ee_pose_robot_root_m_xyzw=tuple(float(value) for value in ee[index]),
            )
            for index in chosen
        )

    selection = VectorInitialStateSelection(
        samples=samples,
        nominal_grasp_pose_root_m_xyzw=nominal_pose,
        source_hdf5_path=str(source_path),
        source_hdf5_sha256=base.hdf5_sha256,
        source_candidate_count=len(candidates),
        selection_rule=selection_rule,
    )
    receipt = selection.receipt()
    if not receipt["distinct_source_sample_ids"] or not receipt["gripper_open_all"]:
        raise Stage1AVectorInitialStateError("VECTOR_SOURCE_SELECTION_VALIDATION_FAILED")
    return selection


__all__ = [
    "Stage1AVectorInitialStateError",
    "VECTOR_INITIAL_STATE_SCHEMA",
    "VectorInitialStateSelection",
    "select_stage1a_vector_initial_states",
]
