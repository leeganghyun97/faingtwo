# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _portable_builder():
    path = ROOT / "scripts/repro/build_portable_deliverables.py"
    spec = importlib.util.spec_from_file_location("portable_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _method_launcher():
    path = ROOT / "scripts/repro/run_method.py"
    spec = importlib.util.spec_from_file_location("method_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_matrix_is_exactly_a_to_g() -> None:
    payload = json.loads(
        (ROOT / "configs/reproducibility/methods_a_to_g.json").read_text(encoding="utf-8")
    )
    assert sorted(payload["methods"]) == list("ABCDEFG")
    assert all("runtime_variant_6000" in value for value in payload["methods"].values())


def test_repro_shell_entrypoints_use_strict_mode() -> None:
    names = [
        "bootstrap.sh", "preflight.sh", "smoke_test.sh", "collect_data.sh",
        "validate_dataset.sh", "install_minimal_isaac_deps.sh",
        "run_minimal_stage1a_training.sh",
        *(f"run_method_{name}.sh" for name in "abcdefg"),
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
        authority = receipt["training_authority"]
        assert authority["training_urdf"].endswith("G2_omnipicker_fixed_dual.urdf")
        assert authority["teleop_translation_m_per_normalized"] == 0.0225
        assert authority["stage1a_final_xyz_max_norm_m"] == 0.0045
        assert authority["residual_effective_max_norm_m"] == 0.00045
        expected_count = 8 if name.upper() in {"F", "G"} else 7
        assert receipt["required_asset_count"] == expected_count
        assert receipt["required_asset_contract_pass"] is all(
            item["pass"] for item in receipt["required_assets"]
        )
        assert receipt["asset_hash_mismatches"] == [
            item["id"]
            for item in receipt["required_assets"]
            if item["hash_match"] is False and item["exists"]
        ]
        asset_ids = {item["id"] for item in receipt["required_assets"]}
        assert {
            "candidate_a_usd",
            "human_grasp_gru_checkpoint",
            "stage1a_pregrasp_hdf5",
            "stage1a_pregrasp_report",
            "far_reach_bc_checkpoint",
            "stage1a_residual_actor_checkpoint",
            "geniesim_asset_pack",
        } <= asset_ids
        assert (
            "conditional_coral_frozen_student" in asset_ids
        ) is (name.upper() in {"F", "G"})


def test_direct_method_launcher_loads_portable_environment(
    tmp_path: Path, monkeypatch,
) -> None:
    launcher = _method_launcher()
    environment_key = "GENIESIM_TEST_PORTABLE_ENV_RECEIPT"
    monkeypatch.delenv(environment_key, raising=False)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        f"# generated receipt\n{environment_key}=portable-value\n",
        encoding="utf-8",
    )

    launcher.load_repository_environment()

    assert launcher.os.environ[environment_key] == "portable-value"


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


def test_g2_training_authority_covers_teleop_urdf_and_action_scales() -> None:
    manifest = json.loads(
        (ROOT / "configs/reproducibility/g2_training_authority.json").read_text(
            encoding="utf-8"
        )
    )
    robot = manifest["robot_description"]
    teleop = manifest["teleoperation_environment"]
    action = manifest["action_contract"]
    timing = manifest["timing_contract"]
    assert robot["training_urdf"].endswith("G2_omnipicker_fixed_dual.urdf")
    assert teleop["canonical_for_stage1a_collection"] == "ISAACLAB_KEYBOARD_V3"
    assert action["teleop_translation_m_per_normalized"] == 0.0225
    assert action["stage1a_final_xyz_max_norm_m"] == 0.0045
    assert action["residual_raw_max_norm_m"] == 0.0045
    assert action["residual_alpha"] == 0.10
    assert action["residual_effective_max_norm_m"] == 0.00045
    assert action["residual_has_gripper_authority"] is False
    assert timing == {
        "control_hz": 50,
        "dataset_row_hz": 50,
        "rgbd_hz": 25,
        "physics_hz": 500,
        "physics_substeps_per_control": 10,
    }


def test_g2_training_authority_file_hashes_match() -> None:
    manifest = json.loads(
        (ROOT / "configs/reproducibility/g2_training_authority.json").read_text(
            encoding="utf-8"
        )
    )
    paths = set()
    for item in manifest["local_files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert item["path"] not in paths
        paths.add(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert manifest["robot_description"]["training_urdf"] in paths
    assert manifest["teleoperation_environment"]["environment_config"] in paths
    assert manifest["teleoperation_environment"]["physical_action_adapter"] in paths


def test_minimal_training_bundle_and_route_are_bounded() -> None:
    spec = json.loads(
        (ROOT / "configs/reproducibility/minimal_training_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["schema"] == "geniesim_minimal_training_bundle_spec_v1"
    training = spec["minimal_training"]
    assert training == {
        "accepted_transitions": 3000,
        "num_envs": 10,
        "replay": "HER_FORCE",
        "runtime_variant": "V3_BASELINE",
        "wandb_default": "offline",
    }
    targets = [Path(item["target"]) for item in spec["entries"]]
    assert len(targets) == len(set(targets))
    assert all(not target.is_absolute() and ".." not in target.parts for target in targets)
    target_strings = {target.as_posix() for target in targets}
    assert "artifacts/external/far_reach_bc_best.pt" in target_strings
    assert "artifacts/external/stage1a_pregrasp_initial_states.hdf5" in target_strings
    assert "source/geniesim/assets/robot/G2_omnipicker" in target_strings

    launcher = (ROOT / "scripts/run_minimal_stage1a_training.sh").read_text(
        encoding="utf-8"
    )
    assert "--num-envs 10" in launcher
    assert "--accepted-transitions 3000" in launcher
    assert "--runtime-variant V3_BASELINE" in launcher
    assert "--wandb-mode" in launcher

    installer = (ROOT / "scripts/repro/minimal_training_bundle.py").read_text(
        encoding="utf-8"
    )
    assert "--frozen-student-checkpoint" in installer
    assert "FROZEN_STUDENT_CHECKPOINT_HASH_MISMATCH" in installer

    deliverable_builder = (
        ROOT / "scripts/repro/build_portable_deliverables.py"
    ).read_text(encoding="utf-8")
    assert "GENIESIM_BUNDLE_ROOT" in deliverable_builder
    assert "GENIESIM_REPO_ROOT" in deliverable_builder

    guide = (ROOT / "README_G2_STAGE1A.md").read_text(encoding="utf-8")
    install_section = guide.split("## 5. 새 PC 설치", 1)[1].split(
        "## 6. 실행 순서", 1
    )[0]
    assert "--frozen-student-checkpoint" in install_section
    assert "cp .env.example .env" not in install_section


def test_all_live_methods_require_actual_runtime_initializers() -> None:
    assets = json.loads(
        (ROOT / "configs/reproducibility/external_assets.json").read_text(
            encoding="utf-8"
        )
    )["assets"]
    by_id = {item["id"]: item for item in assets}
    required = {
        "human_grasp_gru_checkpoint",
        "far_reach_bc_checkpoint",
        "stage1a_residual_actor_checkpoint",
        "stage1a_pregrasp_hdf5",
        "stage1a_pregrasp_report",
    }
    assert required <= set(by_id)
    assert all("all_live_methods" in by_id[item]["required_for"] for item in required)


def test_transfer_bundle_binds_exact_source_repository() -> None:
    receipt = _portable_builder().source_repository_receipt()
    assert receipt["branch"] == "stage1a-portable-training"
    assert len(receipt["commit"]) == 40
    assert receipt["url"].endswith("/leeganghyun97/genie_sim.git")


def test_open_table_clearance_is_frozen_and_in_portable_static_suite() -> None:
    runtime_launcher = (
        ROOT / "scripts/run_g2_stage1a_vector_runtime.py"
    ).read_text(encoding="utf-8")
    static_suite = (ROOT / "scripts/g2/01_static_tests.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'ROOT / "source/geniesim/rl/sac/stage1a_open_table_clearance.py"'
        in runtime_launcher
    )
    assert '"${root}/tests/test_g2_stage1a_open_table_clearance.py"' in static_suite


def test_all_runtime_assets_declare_portable_bundle_paths() -> None:
    assets = json.loads(
        (ROOT / "configs/reproducibility/external_assets.json").read_text(
            encoding="utf-8"
        )
    )["assets"]
    assert assets
    for asset in assets:
        assert asset["artifact_type"]
        path = Path(asset["bundle_path"])
        assert not path.is_absolute()
        assert ".." not in path.parts

    builder = (ROOT / "scripts/repro/build_portable_deliverables.py").read_text(
        encoding="utf-8"
    )
    assert "METHOD_A_TO_G_INPUTS.json" in builder
    assert "METHOD_BUNDLE_ASSET_MISSING" in builder
    assert "METHOD_BUNDLE_ASSET_HASH_MISMATCH" in builder
