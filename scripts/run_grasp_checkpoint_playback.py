#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Evaluation-only Stable-Grasp playback for Stage-1A periodic checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch

from geniesim.rl.sac.residual_sac_runtime import (
    residual_sac_actor_checkpoint_payload,
)
from geniesim.rl.sac.stage1a_real_sac_coordinator import (
    STAGE1A_PERIODIC_CHECKPOINT_SCHEMA,
)
from geniesim.rl.sac.stage2_sac import SACAgent, SACConfig


ROOT = Path(__file__).resolve().parents[1]
LIVE_RUNNER = ROOT / "scripts/diagnostics/run_g2_curobo_planner_live_smoke.py"
CANDIDATE_A = (
    ROOT
    / "artifacts/g2_bounded_passive_range_qualification_20260921/"
    "candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
)
CANDIDATE_A_SHA256 = (
    "d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _extract_actor(checkpoint: Path, destination: Path) -> tuple[int, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != STAGE1A_PERIODIC_CHECKPOINT_SCHEMA:
        raise RuntimeError("PLAYBACK_REQUIRES_PERIODIC_STAGE1A_CHECKPOINT")
    agent_state = payload["agent"]
    config = SACConfig(**dict(agent_state["config"]))
    if config.action_dim != 3:
        raise RuntimeError("PLAYBACK_CHECKPOINT_ACTION_DIM_NOT_3")
    agent = SACAgent(config, device="cpu")
    agent.load_state_dict(agent_state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        residual_sac_actor_checkpoint_payload(
            agent.actor,
            human_grasp_checkpoint_sha256=payload["bc_checkpoint_sha256"],
        ),
        destination,
    )
    return int(payload["accepted_transitions"]), _sha256(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="show the Isaac Sim Kit viewport during evaluation-only playback",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit("CHECKPOINT_NOT_FOUND")
    if not 1 <= args.episodes <= 100:
        raise SystemExit("EPISODES_MUST_BE_1_TO_100")
    if not args.deterministic:
        raise SystemExit("ONLY_DETERMINISTIC_EVALUATION_IS_VALIDATED")
    if output_dir.exists():
        raise SystemExit("PLAYBACK_OUTPUT_REFUSES_OVERWRITE")
    output_dir.mkdir(parents=True)

    actor_path = output_dir / "evaluation_actor.pt"
    checkpoint_step, actor_sha256 = _extract_actor(checkpoint, actor_path)
    episode_reports: list[dict[str, Any]] = []
    environment = dict(os.environ)
    source_path = str(ROOT / "source")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    for episode in range(args.episodes):
        episode_root = output_dir / f"episode_{episode:04d}"
        report_path = episode_root / "report.json"
        runtime_root = episode_root / "runtime"
        command = [
            sys.executable,
            str(LIVE_RUNNER),
            "--execute-live",
            "--seed",
            str(args.seed + episode),
            "--output",
            str(report_path),
            "--episode-horizon-steps",
            "1600",
            "--ordinary-tracking-cap",
            "20",
            "--critical-tracking-cap",
            "20",
            "--final-settle-cap",
            "20",
            "--planner-active-velocity-limit-rad-s",
            "0.8",
            "--diagnostic-asset-variant",
            "custom",
            "--diagnostic-asset-path",
            str(CANDIDATE_A),
            "--diagnostic-asset-sha256",
            CANDIDATE_A_SHA256,
            "--keyboard-v3-direct-pregrasp-v2",
            "--stage1a-hybrid-activation-smoke",
            "--stage1a-stable-playback",
            "--stage1a-hybrid-activation-max-steps",
            "512",
            "--stage1a-playback-actor-checkpoint",
            str(actor_path),
            "--stage1a-playback-actor-checkpoint-sha256",
            actor_sha256,
            "--stage1a-output-dir",
            str(runtime_root),
        ]
        if args.gui:
            command.append("--gui")
        if args.record_video:
            command.extend(
                [
                    "--stage1a-playback-video-path",
                    str(episode_root / "right_wrist.mp4"),
                ]
            )
        completed = subprocess.run(command, env=environment, check=False)
        runtime_report = runtime_root / "HYBRID_ACTIVATION_SMOKE_REPORT.json"
        if not runtime_report.is_file():
            raise RuntimeError(
                f"PLAYBACK_EPISODE_REPORT_MISSING:{episode}:exit={completed.returncode}"
            )
        report = json.loads(runtime_report.read_text())
        report["PROCESS_EXIT_CODE"] = completed.returncode
        episode_reports.append(report)

    def total(name: str) -> int:
        result = 0
        for row in episode_reports:
            value = row.get(name, 0)
            if isinstance(value, str):
                result += int(value.upper() in {"YES", "TRUE", "PASS"})
            else:
                result += int(value or 0)
        return result

    def timings(name: str) -> list[float]:
        return [
            float(row[name])
            for row in episode_reports
            if row.get(name) is not None
        ]

    residual_means = timings("RESIDUAL_MEAN_MM")
    residual_p95s = timings("RESIDUAL_P95_MM")
    residual_maxes = timings("RESIDUAL_MAX_MM")
    minimum_residuals = timings("MIN_NOMINAL_RESIDUAL_MM")
    result = {
        "schema": "g2_stage1a_stable_grasp_checkpoint_playback_v1",
        "CHECKPOINT": str(checkpoint),
        "CHECKPOINT_SHA256": _sha256(checkpoint),
        "CHECKPOINT_STEP": checkpoint_step,
        "EPISODES": args.episodes,
        "DETERMINISTIC": True,
        "GUI": bool(args.gui),
        "EVALUATION_ONLY": True,
        "CLOSE_COUNT": total("CLOSE_COUNT"),
        "CONTACT_COUNT": total("CONTACT_RATE"),
        "BILATERAL_COUNT": total("BILATERAL_RATE"),
        "STABLE_COUNT": total("STABLE_RATE"),
        "CONTACT_RATE": total("CONTACT_RATE") / args.episodes,
        "BILATERAL_RATE": total("BILATERAL_RATE") / args.episodes,
        "STABLE_RATE": total("STABLE_RATE") / args.episodes,
        "TIME_TO_CONTACT_MS": _mean(timings("TIME_TO_CONTACT_MS")),
        "TIME_TO_BILATERAL_MS": _mean(timings("TIME_TO_BILATERAL_MS")),
        "TIME_TO_STABLE_MS": _mean(timings("TIME_TO_STABLE_MS")),
        "HOVER_RATE": _mean(timings("HOVER_RATE")),
        "MIN_NOMINAL_RESIDUAL_MM": (
            None if not minimum_residuals else min(minimum_residuals)
        ),
        "RESIDUAL_MEAN_MM": _mean(residual_means),
        "RESIDUAL_P95_MM": (
            None if not residual_p95s else max(residual_p95s)
        ),
        "RESIDUAL_MAX_MM": (
            None if not residual_maxes else max(residual_maxes)
        ),
        "JOINT_LIMIT_VIOLATION": total("JOINT_LIMIT_VIOLATION"),
        "VELOCITY_LIMIT_VIOLATION": total("VELOCITY_LIMIT_VIOLATION"),
        "FORBIDDEN_COLLISION": total("FORBIDDEN_COLLISION"),
        "RUNTIME_HARDSTOP": total("RUNTIME_HARDSTOP"),
        "QDD_AUTHORITY": "DIAGNOSTIC_ONLY",
        "GRADIENT_UPDATE_COUNT": total("GRADIENT_UPDATE_COUNT"),
        "REPLAY_BUFFER_WRITE_COUNT": total("REPLAY_BUFFER_WRITE_COUNT"),
        "TRAINING_TRANSITION_COUNT_INCREMENT": total(
            "TRAINING_TRANSITION_COUNT_INCREMENT"
        ),
        "WANDB_TRAINING_RUN_TOUCHED": False,
        "AUTO_PLAY": False,
        "episode_reports": [
            str(output_dir / f"episode_{index:04d}" / "runtime" /
                "HYBRID_ACTIVATION_SMOKE_REPORT.json")
            for index in range(args.episodes)
        ],
    }
    _atomic_json(output_dir / "PLAYBACK_SUMMARY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
