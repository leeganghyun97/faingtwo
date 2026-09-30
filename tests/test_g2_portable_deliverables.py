# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _object(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_keyboard_release_is_ros2_free_and_contract_complete() -> None:
    spec = _object("configs/reproducibility/keyboard_collection_release.json")
    assert spec["student_privileged_input_count"] == 0
    assert spec["real_robot_authority"] is False
    assert "ros2" in spec["excluded_path_components"]
    assert "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py" in spec["dependency_roots"]
    required = set(spec["explicit_files"])
    assert "configs/g2_policy_branch/keyboard_collection_v3.json" in required
    assert "source/data_collection/config/robot_cfg/G2/G2_omnipicker_fixed_dual.urdf" in required
    assert "source/data_collection/config/curobo/configs/robot/G2_omnipicker_fixed_dual.yml" in required


def test_transfer_bundle_uses_environment_resolved_sources() -> None:
    spec = _object("configs/reproducibility/g2_transfer_bundle.json")
    entries = spec["entries"]
    assert entries
    assert all("source_environment" in row and "source" not in row for row in entries)
    roles = {role for row in entries for role in row["roles"]}
    assert {f"METHOD_{name}" for name in "ABCDEFG"}.issubset(roles)


def test_stage1a_runbook_and_entrypoints_exist() -> None:
    runbook = (ROOT / "README_G2_STAGE1A.md").read_text(encoding="utf-8")
    assert "Student privileged input" not in runbook or "student privileged input" in runbook
    assert "Grasp Success" in runbook
    assert "resume" in runbook
    for index, name in enumerate(
        (
            "check_env",
            "static_tests",
            "app_smoke",
            "physics_smoke",
            "open_restore_smoke",
            "stage1a_smoke",
            "train_3k",
            "train_15k",
            "resume",
            "eval",
        )
    ):
        path = ROOT / "scripts/g2" / f"{index:02d}_{name}.sh"
        assert path.is_file()
        assert path.stat().st_mode & 0o111


def test_new_portable_files_have_no_local_user_path_or_secret_assignment() -> None:
    paths = (
        ROOT / "README_G2_STAGE1A.md",
        ROOT / "configs/reproducibility/keyboard_collection_release.json",
        ROOT / "configs/reproducibility/g2_transfer_bundle.json",
        ROOT / "scripts/repro/build_portable_deliverables.py",
        ROOT / "scripts/repro/keyboard_release_preflight.py",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/home/fain" not in text
        assert "/data/fain" not in text
        assert "WANDB_API_KEY=" not in text


def test_live_wrappers_bind_the_current_clone_source_tree() -> None:
    common = (ROOT / "scripts/g2/_common.sh").read_text(encoding="utf-8")
    installer = (ROOT / "scripts/install_minimal_isaac_deps.sh").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts/repro/run_method.py").read_text(encoding="utf-8")
    assert 'export PYTHONPATH="${G2_REPO_ROOT}/source' in common
    assert 'pip install -e "${root}/source"' in installer
    assert 'source_root = str(ROOT / "source")' in launcher
    assert "choices=(10, 25)" in launcher
