#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Create a derived manifest separating legacy binary and exact-margin rows.

The command never rewrites JSONL/HDF5 inputs.  Legacy V1 rows remain usable
for binary auxiliary supervision and representation analysis, but are never
assigned a made-up continuous target.  V2 rows are admitted to the signed
loss only after the shared boundary audit verified binary/margin parity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_g2_stage1a_boundary_paired_dataset import (
    _canonical_paired_groups,
    _load_rows,
    _split_units,
)
from geniesim.rl.sac.stage1a_boundary_paired_collection import (
    BOUNDARY_PAIRED_ROW_SCHEMA,
    SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
    SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
)


SCHEMA = "g2_stage1a_signed_margin_preprocess_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical, _summary = _canonical_paired_groups(rows)
    return canonical


def _counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "positive": sum(bool(row["target"]) for row in rows),
        "negative": sum(not bool(row["target"]) for row in rows),
    }


def _legacy_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    units = _split_units(rows)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    split = {
        name: [row for pair in pairs for row in by_pair[pair]]
        for name, pairs in units.items()
    }
    return {
        "available": True,
        "family_disjoint": True,
        "counts": {name: _counts(value) for name, value in split.items()},
        "families": {
            name: sorted({str(row["source_family_id"]) for row in value})
            for name, value in split.items()
        },
    }


def preprocess(
    *, legacy_sidecars: list[Path], signed_sidecars: list[Path]
) -> dict[str, Any]:
    all_paths = [*legacy_sidecars, *signed_sidecars]
    if any("batch-10" in str(path) for path in all_paths):
        raise ValueError("BATCH10_SOURCE_FREEZE_MISMATCH_FORBIDDEN")
    loaded = [row for path in all_paths for row in _load_rows(path)]
    legacy = _canonical([
        row for row in loaded
        if row["raw"].get("schema") == BOUNDARY_PAIRED_ROW_SCHEMA
    ])
    signed = _canonical([
        row for row in loaded
        if row["raw"].get("schema") in {
            SIGNED_MARGIN_BOUNDARY_ROW_SCHEMA,
            SIGNED_MARGIN_PAIRED_CLONE_ROW_SCHEMA,
        }
    ])
    if any(row.get("signed_margin") is None for row in signed):
        raise ValueError("SIGNED_MARGIN_ROW_NOT_AUDIT_VALID")
    return {
        "SCHEMA": SCHEMA,
        "PREPROCESS_DERIVED_ONLY": True,
        "RAW_INPUTS_MODIFIED": "NO",
        "ISAAC_STARTED": "NO",
        "SAC_UPDATE": 0,
        "STUDENT_OPTIMIZER_UPDATE": 0,
        "PRIVILEGED_HARD_GATE": "NO",
        "STUDENT_PRIVILEGED_INPUT_COUNT": 0,
        "LEGACY_BINARY_ONLY": {
            "CANONICAL_ROWS": len(legacy),
            "COUNTS": _counts(legacy),
            "SIGNED_MARGIN_ROWS": 0,
            "ROLE": "BINARY_AUXILIARY_AND_REPRESENTATION_ONLY",
            "SPLIT": _legacy_split(legacy),
        },
        "EXACT_SIGNED_MARGIN": {
            "CANONICAL_ROWS": len(signed),
            "COUNTS": _counts(signed),
            "SIGNED_MARGIN_ROWS": len(signed),
            "ROLE": "BINARY_PLUS_SIGNED_MARGIN_SUPERVISION",
            "SOURCE_FAMILY_COUNT": len({str(row["source_family_id"]) for row in signed}),
            "MARGIN_RANGE": (
                {
                    "min": min(float(row["signed_margin"]) for row in signed),
                    "max": max(float(row["signed_margin"]) for row in signed),
                }
                if signed else None
            ),
        },
        "INPUT_RECEIPTS": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in all_paths
        ],
        "BATCH10_INCLUDED": "NO",
        "NEXT": (
            "COLLECT_EXACT_SIGNED_MARGIN_ROWS"
            if not signed else "RUN_SIGNED_MARGIN_OFFLINE_AUDIT"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-sidecar", type=Path, action="append", default=[])
    parser.add_argument("--signed-sidecar", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("SIGNED_MARGIN_PREPROCESS_REFUSES_OVERWRITE")
    report = preprocess(
        legacy_sidecars=args.legacy_sidecar,
        signed_sidecars=args.signed_sidecar,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "LEGACY_BINARY_ONLY_ROWS": report["LEGACY_BINARY_ONLY"]["CANONICAL_ROWS"],
        "EXACT_SIGNED_MARGIN_ROWS": report["EXACT_SIGNED_MARGIN"]["CANONICAL_ROWS"],
        "RAW_INPUTS_MODIFIED": report["RAW_INPUTS_MODIFIED"],
        "NEXT": report["NEXT"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
