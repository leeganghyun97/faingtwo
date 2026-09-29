#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validate canonical v3 episodes and emit a GRU training-readiness receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from geniesim.rl.sac.keyboard_v3_dataset import training_readiness


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    root = args.collection_root.expanduser().resolve()
    episode_root = root / "episodes" if (root / "episodes").is_dir() else root
    paths = sorted(episode_root.glob("*.hdf5"))
    receipt = training_readiness(paths, split_seed=args.split_seed)
    receipt["collection_root"] = str(root)
    receipt["episode_files_discovered"] = len(paths)
    _atomic_json(args.output.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["gru_bc_training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
