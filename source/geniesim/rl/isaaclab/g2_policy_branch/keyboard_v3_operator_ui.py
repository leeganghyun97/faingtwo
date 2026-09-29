# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail-closed keyboard-v3 episode-result UI and publication routing.

ENTER never labels success.  It stops row acquisition and enters a mandatory
result-selection state.  Only an explicit numeric selection can publish to
the canonical or rejected episode directory; DISCARD publishes no HDF5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .keyboard_v3_collection_contract import COLLECTION_TARGETS_MM

from geniesim.rl.sac.keyboard_v3_dataset import (
    append_keyboard_v3_episode_v1_layout,
    validate_episode,
    write_keyboard_v3_episode,
    write_rejected_keyboard_v3_episode,
)

from .keyboard_v3_runtime import KeyboardV3EpisodeRecorder


KEYBOARD_V3_OPERATOR_UI_SCHEMA = "g2_keyboard_v3_operator_result_ui_v1"


class OperatorUIError(RuntimeError):
    pass


class OperatorUIState(str, Enum):
    RECORDING = "RECORDING"
    SELECT_RESULT = "SELECT_RESULT"
    COMMITTED = "COMMITTED"
    DISCARDED = "DISCARDED"


class EpisodeResult(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL_EARLY_CLOSE = "FAIL_EARLY_CLOSE"
    FAIL_LATE_CLOSE = "FAIL_LATE_CLOSE"
    FAIL_LATERAL_MISALIGNMENT = "FAIL_LATERAL_MISALIGNMENT"
    FAIL_HEIGHT_MISALIGNMENT = "FAIL_HEIGHT_MISALIGNMENT"
    FAIL_OPERATOR_ABORT = "FAIL_OPERATOR_ABORT"
    DISCARD = "DISCARD"


RESULT_KEY_MAP = {
    "1": EpisodeResult.SUCCESS,
    "2": EpisodeResult.FAIL_EARLY_CLOSE,
    "3": EpisodeResult.FAIL_LATE_CLOSE,
    "4": EpisodeResult.FAIL_LATERAL_MISALIGNMENT,
    "5": EpisodeResult.FAIL_HEIGHT_MISALIGNMENT,
    "6": EpisodeResult.FAIL_OPERATOR_ABORT,
    "0": EpisodeResult.DISCARD,
}


@dataclass(frozen=True)
class EpisodePublication:
    episode_id: str
    target_residual_mm: int
    row_count: int
    close_event_count: int
    nonzero_xyz_row_count: int
    success: bool
    failure_type: str
    save_path: str | None
    aggregate_save_path: str | None
    aggregate_demo_id: str | None
    validator_result: str
    canonical_training_registered: bool
    result: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": KEYBOARD_V3_OPERATOR_UI_SCHEMA,
            **self.__dict__,
        }

    def console_lines(self) -> tuple[str, ...]:
        return (
            f"EPISODE_ID: {self.episode_id}",
            f"TARGET_RESIDUAL_MM: {self.target_residual_mm}",
            f"ROW_COUNT: {self.row_count}",
            f"CLOSE_EVENT_COUNT: {self.close_event_count}",
            f"NONZERO_XYZ_ROW_COUNT: {self.nonzero_xyz_row_count}",
            f"SUCCESS: {str(self.success).lower()}",
            f"FAILURE_TYPE: {self.failure_type}",
            f"SAVE_PATH: {self.save_path if self.save_path is not None else 'NONE_DISCARDED'}",
            "V1_AGGREGATE_PATH: "
            f"{self.aggregate_save_path if self.aggregate_save_path is not None else 'NONE_DISCARDED'}",
            "V1_AGGREGATE_DEMO_ID: "
            f"{self.aggregate_demo_id if self.aggregate_demo_id is not None else 'NONE_DISCARDED'}",
            f"VALIDATOR_RESULT: {self.validator_result}",
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"result receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_distribution_summary(receipt_directory: Path) -> dict[str, Any]:
    """Rebuild a non-authoritative operator summary from immutable receipts."""

    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(receipt_directory.glob("*.result.json"))
    ]
    by_target = {target: 0 for target in COLLECTION_TARGETS_MM}
    success_by_target = {target: 0 for target in COLLECTION_TARGETS_MM}
    fail_by_target = {target: 0 for target in COLLECTION_TARGETS_MM}
    valid = rejected = discarded = safe = unsafe = invalid = 0
    for receipt in receipts:
        target = int(receipt["target_residual_mm"])
        if target not in by_target:
            # Historical 25/30/35-mm receipts remain readable but do not
            # participate in the new balanced 16--22-mm collection.
            continue
        by_target[target] += 1
        if receipt["result"] == EpisodeResult.DISCARD.value:
            discarded += 1
            continue
        if bool(receipt["canonical_training_registered"]):
            valid += 1
            if bool(receipt["success"]):
                safe += 1
                success_by_target[target] += 1
            else:
                unsafe += 1
                fail_by_target[target] += 1
        else:
            rejected += 1
            invalid += 1
    values = list(by_target.values())
    summary = {
        "schema": "g2_keyboard_v3_target_distribution_v1",
        "total_episodes": len(receipts),
        "valid_episodes": valid,
        "rejected_episodes": rejected,
        "discarded_episodes": discarded,
        "safe_count": safe,
        "unsafe_count": unsafe,
        "invalid_count": invalid,
        "count_by_target_mm": {str(key): value for key, value in by_target.items()},
        "success_by_target_mm": {
            str(key): value for key, value in success_by_target.items()
        },
        "fail_by_target_mm": {
            str(key): value for key, value in fail_by_target.items()
        },
        "distribution_imbalance_warning": max(values) - min(values) > 1,
    }
    path = receipt_directory.parent / "COLLECTION_DISTRIBUTION.json"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return summary


class KeyboardV3OperatorUI:
    """One-episode result state machine.  It never starts the next episode."""

    def __init__(
        self,
        recorder: KeyboardV3EpisodeRecorder,
        *,
        canonical_episode_directory: str | Path,
        rejected_episode_directory: str | Path,
        receipt_directory: str | Path,
    ) -> None:
        self.recorder = recorder
        self.canonical_episode_directory = Path(canonical_episode_directory).resolve()
        self.rejected_episode_directory = Path(rejected_episode_directory).resolve()
        self.receipt_directory = Path(receipt_directory).resolve()
        self.state = OperatorUIState.RECORDING
        self.selected_result: EpisodeResult | None = None
        self.publication: EpisodePublication | None = None

    @property
    def recording_enabled(self) -> bool:
        return self.state is OperatorUIState.RECORDING

    @property
    def result_selection_active(self) -> bool:
        return self.state is OperatorUIState.SELECT_RESULT

    @property
    def next_episode_allowed(self) -> bool:
        return self.state in (OperatorUIState.COMMITTED, OperatorUIState.DISCARDED)

    def press_enter(self) -> None:
        if self.state is not OperatorUIState.RECORDING:
            raise OperatorUIError("ENTER is valid exactly once during RECORDING")
        self.state = OperatorUIState.SELECT_RESULT
        print("END_EPISODE_AND_SELECT_RESULT", flush=True)
        print("1=SUCCESS 2=FAIL_EARLY_CLOSE 3=FAIL_LATE_CLOSE", flush=True)
        print("4=FAIL_LATERAL_MISALIGNMENT 5=FAIL_HEIGHT_MISALIGNMENT", flush=True)
        print("6=FAIL_OPERATOR_ABORT 0=DISCARD", flush=True)

    def choose(self, key: str) -> EpisodePublication:
        if self.state is not OperatorUIState.SELECT_RESULT:
            raise OperatorUIError("result selection is not active")
        if key not in RESULT_KEY_MAP:
            raise OperatorUIError("result key must be one of 0..6")
        result = RESULT_KEY_MAP[key]
        self.selected_result = result
        episode_id = self.recorder.episode_id
        target_residual_mm = int(
            round(self.recorder.planner.condition.backoff_m * 1000.0)
        )
        if result is EpisodeResult.DISCARD:
            # Build first only for deterministic summary counts; never write
            # an HDF5 episode or register it as training data.
            episode = self.recorder.build_episode(
                success=False, failure_type="OPERATOR_ABORT"
            )
            validation = validate_episode(episode)
            publication = EpisodePublication(
                episode_id=episode_id,
                target_residual_mm=target_residual_mm,
                row_count=validation.rows,
                close_event_count=validation.close_event_count,
                nonzero_xyz_row_count=validation.nonzero_micro_approach_rows,
                success=False,
                failure_type="DISCARDED",
                save_path=None,
                aggregate_save_path=None,
                aggregate_demo_id=None,
                validator_result="NOT_RUN_FOR_TRAINING_DISCARDED",
                canonical_training_registered=False,
                result=result.value,
            )
            self.recorder.mark_discarded()
            self.state = OperatorUIState.DISCARDED
        else:
            success = result is EpisodeResult.SUCCESS
            failure_type = "NONE" if success else result.value
            episode = self.recorder.build_episode(
                success=success, failure_type=failure_type
            )
            validation = validate_episode(episode)
            if validation.passed:
                path = self.canonical_episode_directory / f"{episode_id}.hdf5"
                write_keyboard_v3_episode(path, episode)
                aggregate = append_keyboard_v3_episode_v1_layout(
                    self.canonical_episode_directory.parent
                    / "g2_keyboard_v3_rgbd.hdf5",
                    episode,
                    training_eligible=True,
                )
                validator_result = "PASS_CANONICAL_REGISTERED"
                canonical = True
            else:
                path = self.rejected_episode_directory / f"{episode_id}.hdf5"
                write_rejected_keyboard_v3_episode(path, episode)
                aggregate = append_keyboard_v3_episode_v1_layout(
                    self.rejected_episode_directory.parent
                    / "g2_keyboard_v3_rejected_rgbd.hdf5",
                    episode,
                    training_eligible=False,
                )
                validator_result = "FAIL_REJECTED:" + ",".join(validation.errors)
                canonical = False
            publication = EpisodePublication(
                episode_id=episode_id,
                target_residual_mm=target_residual_mm,
                row_count=validation.rows,
                close_event_count=validation.close_event_count,
                nonzero_xyz_row_count=validation.nonzero_micro_approach_rows,
                success=success,
                failure_type=failure_type,
                save_path=str(path),
                aggregate_save_path=aggregate.path,
                aggregate_demo_id=aggregate.demo_id,
                validator_result=validator_result,
                canonical_training_registered=canonical,
                result=result.value,
            )
            self.recorder.mark_discarded()
            self.state = OperatorUIState.COMMITTED
        self.publication = publication
        _atomic_json(
            self.receipt_directory / f"{episode_id}.result.json",
            publication.payload(),
        )
        for line in publication.console_lines():
            print(line, flush=True)
        distribution = _write_distribution_summary(self.receipt_directory)
        for target in COLLECTION_TARGETS_MM:
            print(
                f"TARGET_{target}MM_COUNT: "
                f"{distribution['count_by_target_mm'][str(target)]}",
                flush=True,
            )
        for name in (
            "total_episodes", "valid_episodes", "rejected_episodes",
            "discarded_episodes", "safe_count", "unsafe_count", "invalid_count",
        ):
            print(f"{name.upper()}: {distribution[name]}", flush=True)
        print(
            "TARGET_DISTRIBUTION_IMBALANCE_WARNING: "
            f"{str(distribution['distribution_imbalance_warning']).upper()}",
            flush=True,
        )
        return publication

    def handle_key(self, key: str) -> EpisodePublication | None:
        """Runtime-safe callback: irrelevant result keys cannot label a live episode."""

        if key == "ENTER":
            self.press_enter()
            return None
        if key in RESULT_KEY_MAP:
            if not self.result_selection_active:
                return None
            return self.choose(key)
        return None

    def bind_keyboard_callbacks(self, keyboard: Any) -> None:
        """Bind exact prompt keys to an Isaac keyboard-like callback API."""

        add_callback = getattr(keyboard, "add_callback", None)
        if not callable(add_callback):
            raise OperatorUIError("keyboard does not expose add_callback")
        add_callback("ENTER", lambda: self.handle_key("ENTER"))
        for key in RESULT_KEY_MAP:
            add_callback(key, lambda selected=key: self.handle_key(selected))


def create_omni_result_panel(
    controller: KeyboardV3OperatorUI,
    *,
    title: str = "G2 keyboard-v3 episode result",
) -> Any:
    """Create the live numeric result panel; import remains Isaac-free."""

    import omni.ui as ui

    window = ui.Window(title, width=510, height=330)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label("ENTER: END_EPISODE_AND_SELECT_RESULT")
            ui.Separator()
            for key, result in RESULT_KEY_MAP.items():
                ui.Button(
                    f"{key} = {result.value}",
                    clicked_fn=lambda selected=key: controller.handle_key(selected),
                )
    return window


__all__ = [
    "EpisodePublication",
    "EpisodeResult",
    "KeyboardV3OperatorUI",
    "OperatorUIError",
    "OperatorUIState",
    "RESULT_KEY_MAP",
    "create_omni_result_panel",
]
