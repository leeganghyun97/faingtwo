# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "g2_keyboard_v3_live_runner_artifact_test", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_post_reset_receipt_creates_run_directory(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "new_run" / "RUNTIME_REPLAN_POST_RESET_INPUT.json"
    runner._atomic_json(output, {"stage": "POST_RESET", "finite": 1.0})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "stage": "POST_RESET",
    }


def test_keyboard_v3_freeze_profile_remains_fail_closed() -> None:
    runner = _load_runner()
    freeze = runner._source_freeze(keyboard_v3_branch=True)
    assert freeze["SOURCE_FREEZE"] == "PASS"
    assert freeze["freeze_profile"] == "KEYBOARD_V3_CANDIDATE_A_ONE_SHOT"
    assert freeze["authorized_branch_delta"]
    assert all(row["match"] for row in freeze["keyboard_v3_sources"].values())


def test_live_runner_explicitly_requests_kit_visualizer_for_gui_keyboard() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._run_live)
    assert 'visualizer=["kit"] if gui else ["none"]' in source
    assert "visualizer_explicit=True" in source
    assert "headless=not gui" in source


def test_final_collection_launcher_uses_direct_pregrasp_16_22mm_path() -> None:
    shell = (ROOT / "scripts/run_g2_keyboard_v3_terminal_collection.sh").read_text(
        encoding="utf-8"
    )
    runner = RUNNER.read_text(encoding="utf-8")
    assert "--candidate-a-contact-free-runtime-replan" not in shell
    assert "--keyboard-v3-direct-pregrasp-v2" in shell
    assert "--target-residual-mm auto|16..22" in shell
    assert "target_residual_mm=\"auto\"" in shell
    assert "choices=(\n            0.016, 0.017, 0.018" in runner
    assert "or not args.keyboard_v3_direct_pregrasp_v2" in runner
    assert "or args.candidate_a_contact_free_runtime_replan" in runner
    assert "terminal_keyboard=True" in runner
    assert "--keyboard-v3-continuous-session" in shell
    assert "--keyboard-v3-terminal-smoothing-steps" in shell
    assert "0 if keyboard_v3_direct_pregrasp_v2 else 1" in runner
    assert "_restore_keyboard_v3_handoff_state" in runner
    assert "baseline_metadata=baseline_metadata" in runner
    assert '"continuous_session_episode_index": episode_offset' not in runner


def test_continuous_session_episode_ids_are_deterministic() -> None:
    runner = _load_runner()
    assert runner._numbered_episode_id("episode-000001", 0) == "episode-000001"
    assert runner._numbered_episode_id("episode-000001", 9) == "episode-000010"


def test_direct_pregrasp_continuous_session_reuses_one_live_process(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    executed: list[str] = []
    restore_count = 0

    def run_episode(
        episode_id: str,
        report_path: Path,
        restore_receipt: dict | None,
    ) -> int:
        executed.append(episode_id)
        report_path.write_text(
            json.dumps(
                {
                    "keyboard_v3_live_driver_qualified": True,
                    "save_class": "CANONICAL",
                    "saved_path": f"{episode_id}.h5",
                    "session_stop_requested": False,
                    "runtime_error": None,
                    "restore_receipt": restore_receipt,
                }
            ),
            encoding="utf-8",
        )
        return 0

    def restore() -> dict:
        nonlocal restore_count
        restore_count += 1
        return {
            "restored_ee_error_m": 0.0,
            "target_to_measured_max_rad": 0.0,
        }

    status = runner._run_keyboard_v3_session(
        collection_root=tmp_path,
        initial_episode_id="episode-000001",
        initial_report_path=tmp_path / "KEYBOARD_V3_LIVE_REPORT_episode-000001.json",
        continuous_session=True,
        maximum_episodes=3,
        startup_curobo_plan_count=0,
        initial_restore_receipt={"restored_ee_error_m": 0.0},
        run_episode=run_episode,
        restore_for_next_episode=restore,
    )

    progress = json.loads(
        (tmp_path / "KEYBOARD_V3_SESSION_episode-000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert status == 0
    assert executed == ["episode-000001", "episode-000002", "episode-000003"]
    assert restore_count == 2
    assert progress["schema"] == "g2_keyboard_v3_continuous_session_v2"
    assert progress["completed_episode_count"] == 3
    assert progress["startup_curobo_plan_count"] == 0


def test_continuous_session_keeps_process_alive_after_episode_error(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    executed: list[str] = []
    restore_count = 0

    def run_episode(
        episode_id: str,
        report_path: Path,
        _restore_receipt: dict | None,
    ) -> int:
        executed.append(episode_id)
        failed = len(executed) == 2
        quit_requested = len(executed) == 3
        report_path.write_text(
            json.dumps(
                {
                    "keyboard_v3_live_driver_qualified": not failed,
                    "save_class": "REJECTED" if failed else "CANONICAL",
                    "saved_path": f"{episode_id}.h5",
                    "session_stop_requested": quit_requested,
                    "runtime_error": "SAFETY_STOP" if failed else None,
                }
            ),
            encoding="utf-8",
        )
        return 2 if failed else 0

    def restore() -> dict:
        nonlocal restore_count
        restore_count += 1
        return {"restored_ee_error_m": 0.0}

    status = runner._run_keyboard_v3_session(
        collection_root=tmp_path,
        initial_episode_id="episode-000010",
        initial_report_path=tmp_path / "KEYBOARD_V3_LIVE_REPORT_episode-000010.json",
        continuous_session=True,
        maximum_episodes=0,
        startup_curobo_plan_count=0,
        initial_restore_receipt=None,
        run_episode=run_episode,
        restore_for_next_episode=restore,
    )

    assert status == 0
    assert executed == ["episode-000010", "episode-000011", "episode-000012"]
    assert restore_count == 2
    progress = json.loads(
        (tmp_path / "KEYBOARD_V3_SESSION_episode-000010.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["failed_episode_count"] == 1
    assert progress["process_lifetime_authority"] == (
        "EXPLICIT_OPERATOR_QUIT_OR_CONFIGURED_EPISODE_BOUND"
    )


def test_terminal_status_rendering_is_decimated_without_changing_control_rate() -> None:
    live_driver = (
        ROOT
        / "source/geniesim/rl/isaaclab/g2_policy_branch/keyboard_v3_live_driver.py"
    ).read_text(encoding="utf-8")
    assert "terminal_status_print_interval_steps = 5" in live_driver
    assert "control_step % terminal_status_print_interval_steps == 0" in live_driver
    assert '"terminal_status_rate_hz"' in live_driver
    assert "contract.control_hz / terminal_status_print_interval_steps" in live_driver
