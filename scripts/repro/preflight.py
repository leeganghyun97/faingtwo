#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Static/live reproducibility preflight without starting Isaac or a robot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
METHODS = ROOT / "configs/reproducibility/methods_a_to_g.json"
ASSETS = ROOT / "configs/reproducibility/external_assets.json"
TRAINING_AUTHORITY = ROOT / "configs/reproducibility/g2_training_authority.json"
FIXTURE = ROOT / "tests/fixtures/reproducibility/sample_dataset/manifest.json"

CANONICAL_FILES = (
    METHODS,
    ASSETS,
    TRAINING_AUTHORITY,
    ROOT / "scripts/repro/run_method.py",
    ROOT / "scripts/repro/minimal_training_bundle.py",
    ROOT / "scripts/install_minimal_isaac_deps.sh",
    ROOT / "scripts/run_minimal_stage1a_training.sh",
    ROOT / "configs/reproducibility/minimal_training_bundle.json",
    ROOT / "requirements-minimal-training.txt",
    ROOT / "scripts/preflight.sh",
    ROOT / "scripts/smoke_test.sh",
    ROOT / "scripts/collect_data.sh",
    ROOT / "scripts/validate_dataset.sh",
    ROOT / "README_G2_STAGE1A.md",
    ROOT / "configs/reproducibility/keyboard_collection_release.json",
    ROOT / "configs/reproducibility/g2_transfer_bundle.json",
    ROOT / "scripts/repro/build_portable_deliverables.py",
    ROOT / "scripts/repro/keyboard_release_preflight.py",
    *(ROOT / "scripts/g2" / name for name in (
        "00_check_env.sh",
        "01_static_tests.sh",
        "02_app_smoke.sh",
        "03_physics_smoke.sh",
        "04_open_restore_smoke.sh",
        "05_stage1a_smoke.sh",
        "06_train_3k.sh",
        "07_train_15k.sh",
        "08_resume.sh",
        "09_eval.sh",
    )),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=False, capture_output=True
    )
    if result.returncode != 0:
        return []
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def training_authority_checks() -> list[dict]:
    """Verify the exact G2 description, teleop and action authorities.

    These checks are intentionally simulator-free.  A clone must prove that it
    has the same robot-description/config/action inputs before importing Isaac
    or creating a GPU context.
    """

    receipts: list[dict] = []
    try:
        authority = load_json(TRAINING_AUTHORITY)
        local_files = authority["local_files"]
        action = authority["action_contract"]
        timing = authority["timing_contract"]
        robot = authority["robot_description"]
        teleop = authority["teleoperation_environment"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return [{"check": "g2_training_authority_manifest", "pass": False, "error": str(exc)}]

    seen: set[str] = set()
    for item in local_files:
        relative = str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        valid = (
            bool(relative)
            and relative not in seen
            and len(expected) == 64
            and actual == expected
        )
        receipts.append(
            {
                "check": f"g2_authority_file:{relative}",
                "pass": valid,
                "role": item.get("role"),
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
        seen.add(relative)

    expected_teleop_paths = {
        str(teleop[key])
        for key in (
            "environment_config",
            "physical_action_adapter",
            "redundancy_controller",
            "keyboard_runtime",
            "terminal_driver",
            "collection_config",
        )
    }
    receipts.append(
        {
            "check": "g2_teleoperation_environment_in_authority_closure",
            "pass": expected_teleop_paths.issubset(seen),
            "paths": sorted(expected_teleop_paths),
        }
    )
    receipts.append(
        {
            "check": "g2_training_urdf_in_authority_closure",
            "pass": (
                robot.get("training_urdf") in seen
                and robot.get("training_urdf_sha256")
                == next(
                    (
                        row.get("sha256")
                        for row in local_files
                        if row.get("path") == robot.get("training_urdf")
                    ),
                    None,
                )
            ),
            "path": robot.get("training_urdf"),
        }
    )

    residual_scale_ok = abs(
        float(action.get("residual_raw_max_norm_m", float("nan")))
        * float(action.get("residual_alpha", float("nan")))
        - float(action.get("residual_effective_max_norm_m", float("nan")))
    ) <= 1.0e-12
    action_ok = (
        action.get("frame") == "robot_root"
        and action.get("translation_unit") == "m"
        and action.get("teleop_translation_m_per_normalized") == 0.0225
        and action.get("teleop_rotation_rad_per_normalized") == 0.28125
        and action.get("terminal_maximum_translation_step_m") == 0.0045
        and action.get("stage1a_final_xyz_max_norm_m") == 0.0045
        and action.get("residual_raw_max_norm_m") == 0.0045
        and action.get("residual_effective_max_norm_m") == 0.00045
        and action.get("residual_has_gripper_authority") is False
        and residual_scale_ok
    )
    receipts.append(
        {
            "check": "g2_action_scale_contract",
            "pass": action_ok,
            "teleop_translation_m_per_normalized": action.get("teleop_translation_m_per_normalized"),
            "stage1a_final_xyz_max_norm_m": action.get("stage1a_final_xyz_max_norm_m"),
            "residual_effective_max_norm_m": action.get("residual_effective_max_norm_m"),
        }
    )
    receipts.append(
        {
            "check": "g2_timing_contract",
            "pass": (
                timing.get("control_hz") == 50
                and timing.get("dataset_row_hz") == 50
                and timing.get("rgbd_hz") == 25
                and timing.get("physics_hz") == 500
                and timing.get("physics_substeps_per_control") == 10
            ),
        }
    )

    try:
        collection = load_json(ROOT / str(teleop["collection_config"]))
        collection_ok = (
            collection.get("action_frame") == action.get("frame")
            and collection.get("translation_unit") == action.get("translation_unit")
            and collection.get("maximum_translation_norm_m")
            == action.get("stage1a_final_xyz_max_norm_m")
            and collection.get("control_rate_hz") == timing.get("control_hz")
            and collection.get("dataset_row_rate_hz") == timing.get("dataset_row_hz")
            and collection.get("rgbd_capture_rate_hz") == timing.get("rgbd_hz")
            and collection.get("physics_rate_hz") == timing.get("physics_hz")
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        collection_ok = False
    receipts.append({"check": "g2_keyboard_v3_config_matches_action_authority", "pass": collection_ok})
    return receipts


def static_checks() -> list[dict]:
    receipts: list[dict] = []
    for path in CANONICAL_FILES:
        receipts.append({"check": f"file:{path.relative_to(ROOT)}", "pass": path.is_file()})

    try:
        methods = load_json(METHODS)["methods"]
        method_ok = sorted(methods) == list("ABCDEFG")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        method_ok = False
    receipts.append({"check": "method_A_to_G_config", "pass": method_ok})

    try:
        fixture = load_json(FIXTURE)
        fixture_ok = fixture.get("synthetic") is True and fixture.get("frame") == "robot_root"
    except (OSError, ValueError, json.JSONDecodeError):
        fixture_ok = False
    receipts.append({"check": "synthetic_fixture", "pass": fixture_ok})
    receipts.extend(training_authority_checks())

    absolute_hits: list[str] = []
    for path in CANONICAL_FILES:
        if path.is_file() and re.search(r"/(?:home|data)/fain(?:/|\b)", path.read_text(errors="replace")):
            absolute_hits.append(str(path.relative_to(ROOT)))
    receipts.append({"check": "canonical_user_absolute_paths", "pass": not absolute_hits, "paths": absolute_hits})

    secret_hits: list[str] = []
    forbidden_tracked: list[str] = []
    secret_patterns = (
        re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(rb"\bghp_[A-Za-z0-9]{20,}"),
        re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}"),
        re.compile(rb"\b(?:WANDB|OPENAI)_API_KEY\s*=\s*[^\s#'\"]+"),
    )
    forbidden_roots = {"artifacts", "wandb", "output", "outputs", "datasets", "data"}
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if relative.name == ".env" or (relative.parts and relative.parts[0] in forbidden_roots):
            forbidden_tracked.append(str(relative))
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if any(pattern.search(content) for pattern in secret_patterns):
            secret_hits.append(str(relative))
    receipts.append({"check": "tracked_secret_scan", "pass": not secret_hits, "paths": secret_hits})
    receipts.append({"check": "tracked_runtime_data", "pass": not forbidden_tracked, "paths": forbidden_tracked})

    broken: list[str] = []
    for path in tracked_files():
        if path.is_symlink() and not path.exists():
            broken.append(str(path.relative_to(ROOT)))
    # These upstream links depend on the optional teleop/native-SDK install and
    # are outside the Stage-1A canonical closure. Keep them visible instead of
    # pretending they resolve in a source-only clone.
    documented_external_links = {
        "source/geniesim/teleop/tasks",
        "source/teleop/app/share/genie_robot_description/urdf/G2/model.srdf",
        "source/teleop/app/share/genie_robot_description/urdf/G2/model.urdf",
        "source/teleop/app/vendors/lib/libpinocchio_casadi.so",
    }
    unexpected = sorted(set(broken) - documented_external_links)
    receipts.append({
        "check": "tracked_broken_symlinks",
        "pass": not unexpected,
        "unexpected_paths": unexpected,
        "documented_external_paths": sorted(set(broken) & documented_external_links),
        "classification": "PASS" if not broken else "PASS_WITH_EXTERNAL_TELEOP_EXCLUSIONS",
    })
    return receipts


def live_checks(method: str | None, *, collection: bool = False) -> list[dict]:
    receipts: list[dict] = []
    python = Path(os.environ.get("GENIESIM_ISAAC_PYTHON", ""))
    receipts.append({"check": "isaac_python", "pass": python.is_file(), "path": str(python)})
    if python.is_file():
        imports = (
            "import importlib.metadata as m; import torch; import h5py; "
            "import numpy; import wandb; "
        )
        prints = (
            "print(m.version('isaacsim')); print(m.version('isaaclab')); "
            "print(torch.__version__); print(h5py.__version__); "
            "print(numpy.__version__); print(wandb.__version__)"
        )
        if collection:
            imports += "import curobo; "
            prints += "; print(getattr(curobo, '__version__', 'UNKNOWN'))"
        probe = subprocess.run(
            [
                str(python),
                "-c",
                imports + prints,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        receipts.append(
            {
                "check": (
                    "isaac_torch_curobo_packages"
                    if collection
                    else "isaac_training_runtime_packages"
                ),
                "pass": probe.returncode == 0,
                "versions": probe.stdout.splitlines(),
                "stderr": probe.stderr.strip(),
            }
        )
        source_freeze_probe = subprocess.run(
            [
                str(python),
                str(ROOT / "scripts/run_g2_stage1a_vector_runtime.py"),
                "--num-envs",
                "10",
                "--accepted-transitions",
                "3000",
                "--output-dir",
                "/tmp/geniesim-source-freeze-unused",
                "--report",
                "/tmp/geniesim-source-freeze-unused.json",
                "--verify-source-freeze-only",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        try:
            source_freeze_payload = json.loads(source_freeze_probe.stdout)
        except json.JSONDecodeError:
            source_freeze_payload = {}
        receipts.append(
            {
                "check": "stage1a_runtime_source_freeze",
                "pass": (
                    source_freeze_probe.returncode == 0
                    and source_freeze_payload.get("SOURCE_FREEZE") == "PASS"
                    and source_freeze_payload.get("SOURCE_FREEZE_COMPLETE") is True
                ),
                "profile": source_freeze_payload.get("freeze_profile"),
                "manifest_sha256": source_freeze_payload.get("manifest_sha256"),
                "stderr": source_freeze_probe.stderr.strip(),
            }
        )

    manifest = load_json(ASSETS)
    required_for = {"isaac_scene"}
    if method:
        required_for.update({"all_live_methods", f"method_{method}"})
    elif collection:
        required_for.add("collection")
    else:
        required_for.add("all_live_methods")
    for item in manifest["assets"]:
        if not required_for.intersection(item["required_for"]):
            continue
        configured = os.environ.get(item["environment_variable"], "")
        path = Path(configured).expanduser() if configured else ROOT / item["legacy_relative_path"]
        exists = path.is_file() if item["sha256"] else path.is_dir()
        match = exists and (item["sha256"] is None or sha256(path) == item["sha256"])
        receipts.append({"check": f"asset:{item['id']}", "pass": bool(match), "path": str(path), "hash_checked": item["sha256"] is not None})

    # cuRobo resolves its URDF inside the external Genie Sim asset pack.  That
    # mirror must be byte-identical to the repository-owned training URDF;
    # merely having two files with the same name is not sufficient.
    authority = load_json(TRAINING_AUTHORITY)
    robot = authority["robot_description"]
    asset_root_configured = os.environ.get("GENIESIM_ASSET_ROOT", "")
    asset_root = (
        Path(asset_root_configured).expanduser()
        if asset_root_configured
        else ROOT / "source/geniesim/assets"
    )
    mirror = asset_root / robot["curobo_asset_pack_mirror"]
    training_urdf = ROOT / robot["training_urdf"]
    mirror_ok = (
        mirror.is_file()
        and training_urdf.is_file()
        and sha256(mirror) == sha256(training_urdf) == robot["training_urdf_sha256"]
    )
    receipts.append(
        {
            "check": "g2_curobo_urdf_mirror_parity",
            "pass": mirror_ok,
            "training_urdf": str(training_urdf),
            "asset_pack_mirror": str(mirror),
        }
    )

    if collection:
        receipts.extend(
            [
                {
                    "check": "teleop_display",
                    "pass": bool(os.environ.get("DISPLAY")),
                    "display": os.environ.get("DISPLAY"),
                },
                {
                    "check": "teleop_gnome_terminal",
                    "pass": shutil.which("gnome-terminal") is not None,
                    "path": shutil.which("gnome-terminal"),
                },
            ]
        )

    # The source-freeze authority requires the artifact bundle in its original
    # repository-relative layout, not only the top-level USD.
    bundle_receipts = (
        "artifacts/g2_bounded_passive_range_qualification_20260921/CANDIDATE_A_MANIFEST.json",
        "artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/A_source_min_q3_10deg_q4_11p25deg/bounded-range-contract.json",
        "source/geniesim/assets/robot/G2_omnipicker/robot_fix.usda",
    )
    for relative in bundle_receipts:
        receipts.append({"check": f"legacy_source_freeze:{relative}", "pass": (ROOT / relative).is_file()})

    # Resolve the complete Candidate-A dependency/provenance chain before a
    # multi-minute Isaac bootstrap. Historical receipts may rebase only their
    # exact repository-relative suffix; hashes and dependency closure remain
    # strict.
    try:
        source_root = str(ROOT / "source")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from geniesim.rl.isaaclab.g2_policy_branch.contact_free_candidate_a_binding import (
            resolve_contact_free_candidate_a,
        )

        candidate_receipt = resolve_contact_free_candidate_a(repo_root=ROOT)
        candidate_bundle_ok = (
            candidate_receipt.candidate_asset_sha256
            == "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
            and candidate_receipt.dependency_manifest_sha256
            == "3f2120408d91778e47a22d88090db5f66970fad559d211eb08b88ff952e24bad"
        )
        candidate_error = None
    except Exception as exc:  # fail-closed receipt, never weaken the resolver
        candidate_bundle_ok = False
        candidate_error = f"{type(exc).__name__}:{exc}"
    receipts.append(
        {
            "check": "candidate_a_complete_authority_bundle",
            "pass": candidate_bundle_ok,
            "error": candidate_error,
        }
    )

    try:
        from geniesim.rl.isaaclab.g2_policy_branch.candidate_a_left_arm_down_v2 import (
            selected_pregrasp_initial_state,
        )

        pregrasp = selected_pregrasp_initial_state()
        pregrasp_ok = Path(pregrasp.hdf5_path).is_file() and Path(
            pregrasp.report_path
        ).is_file()
        pregrasp_paths = [pregrasp.hdf5_path, pregrasp.report_path]
        pregrasp_error = None
    except Exception as exc:  # fail-closed receipt
        pregrasp_ok = False
        pregrasp_paths = []
        pregrasp_error = f"{type(exc).__name__}:{exc}"
    receipts.append(
        {
            "check": "stage1a_pregrasp_evidence_authority",
            "pass": pregrasp_ok,
            "paths": pregrasp_paths,
            "error": pregrasp_error,
        }
    )
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("static", "live"), default="static")
    parser.add_argument("--method", choices=tuple("ABCDEFG"))
    parser.add_argument(
        "--collection",
        action="store_true",
        help="require the canonical Keyboard-v3 teleoperation dependencies",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipts = static_checks()
    if args.profile == "live":
        receipts.extend(live_checks(args.method, collection=args.collection))
    payload = {
        "schema": "geniesim_repro_preflight_v2",
        "profile": args.profile,
        "method": args.method,
        "collection": args.collection,
        "status": "PASS" if all(row["pass"] for row in receipts) else "FAIL",
        "checks": receipts,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
