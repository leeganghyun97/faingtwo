# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_method_matrix_is_exactly_a_to_g() -> None:
    payload = json.loads(
        (ROOT / "configs/reproducibility/methods_a_to_g.json").read_text(encoding="utf-8")
    )
    assert sorted(payload["methods"]) == list("ABCDEFG")
    assert all("runtime_variant_6000" in value for value in payload["methods"].values())


def test_repro_shell_entrypoints_use_strict_mode() -> None:
    names = [
        "bootstrap.sh", "preflight.sh", "smoke_test.sh", "collect_data.sh",
        "validate_dataset.sh", *(f"run_method_{name}.sh" for name in "abcdefg"),
    ]
    for name in names:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text, name


def test_all_method_entrypoints_dry_run() -> None:
    for name in "abcdefg":
        result = subprocess.run(
            [str(ROOT / "scripts" / f"run_method_{name}.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["method"] == name.upper()
        assert receipt["execute_live"] is False
        assert receipt["num_envs"] == 25


def test_synthetic_dataset_fixture() -> None:
    result = subprocess.run(
        [str(ROOT / "scripts" / "validate_dataset.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"
