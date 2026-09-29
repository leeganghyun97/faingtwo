# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Non-blocking typed 4-D terminal input for Keyboard V3.

The terminal is the command authority; Isaac viewport focus is irrelevant.
Terminal attributes and signal handlers are restored on every normal/error
exit.  No legacy 8-D vector is constructed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import select
import signal
import sys
import termios
import time
import tty
from typing import Any, Callable, TextIO


KEYBOARD_V3_TERMINAL_INPUT_SCHEMA = "g2_keyboard_v3_terminal_4d_v2"
DEFAULT_TRANSLATION_STEP_M = 0.001
DEFAULT_MOTION_MIN_INTERVAL_S = 0.050
MAXIMUM_TRANSLATION_STEP_M = 0.0045
DEFAULT_SMOOTHING_STEPS = 1
MAXIMUM_SMOOTHING_STEPS = 10


class KeyboardV3TerminalInputError(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalPollResult:
    action_4d_metric_root_m: tuple[float, float, float, float]
    event: str | None
    key: str | None


class KeyboardV3TerminalInput:
    """One-key-per-control-epoch cbreak input with persistent gripper state."""

    KEY_TO_XYZ = {
        "w": (1.0, 0.0, 0.0),
        "s": (-1.0, 0.0, 0.0),
        "a": (0.0, 1.0, 0.0),
        "d": (0.0, -1.0, 0.0),
        "q": (0.0, 0.0, 1.0),
        "e": (0.0, 0.0, -1.0),
    }

    def __init__(
        self,
        *,
        translation_step_m: float = DEFAULT_TRANSLATION_STEP_M,
        motion_min_interval_s: float = DEFAULT_MOTION_MIN_INTERVAL_S,
        smoothing_steps: int = DEFAULT_SMOOTHING_STEPS,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < float(translation_step_m) <= MAXIMUM_TRANSLATION_STEP_M:
            raise KeyboardV3TerminalInputError("translation step must be in (0, 0.0045] m")
        if not 0.0 <= float(motion_min_interval_s) <= 1.0:
            raise KeyboardV3TerminalInputError(
                "motion minimum interval must be in [0, 1] seconds"
            )
        if (
            isinstance(smoothing_steps, bool)
            or not 1 <= int(smoothing_steps) <= MAXIMUM_SMOOTHING_STEPS
        ):
            raise KeyboardV3TerminalInputError(
                f"smoothing steps must be in [1, {MAXIMUM_SMOOTHING_STEPS}]"
            )
        self.translation_step_m = float(translation_step_m)
        self.motion_min_interval_s = float(motion_min_interval_s)
        self.smoothing_steps = int(smoothing_steps)
        unnormalized = tuple(
            1.0
            - math.cos(
                2.0 * math.pi * float(index + 1) / float(self.smoothing_steps + 1)
            )
            for index in range(self.smoothing_steps)
        )
        normalizer = sum(unnormalized)
        self._smoothing_weights = tuple(value / normalizer for value in unnormalized)
        self._active_motion_pulses: list[tuple[tuple[float, float, float], int]] = []
        self.maximum_concurrent_motion_pulses = 0
        self.cross_axis_tail_cancel_count = 0
        self._clock = clock
        self._last_motion_time_s: float | None = None
        self.suppressed_motion_key_count = 0
        self.stream = stream if stream is not None else sys.stdin
        if not self.stream.isatty():
            raise KeyboardV3TerminalInputError("terminal keyboard requires a TTY")
        self.fd = self.stream.fileno()
        self._original_termios = termios.tcgetattr(self.fd)
        self._previous_signal_handlers: dict[int, Any] = {}
        self.closed = False
        self.gripper_closed = False
        tty.setcbreak(self.fd)
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        previous = self._previous_signal_handlers.get(signum, signal.SIG_DFL)
        self.close()
        if callable(previous):
            previous(signum, _frame)
        elif previous == signal.SIG_IGN:
            return
        else:
            raise KeyboardInterrupt(f"terminal input interrupted by signal {signum}")

    def _read_one(self) -> str | None:
        readable, _, _ = select.select([self.fd], [], [], 0.0)
        if not readable:
            return None
        raw = os.read(self.fd, 1)
        if not raw:
            return None
        if raw in (b"\r", b"\n"):
            return "ENTER"
        return raw.decode("utf-8", errors="replace").lower()

    def poll(self) -> TerminalPollResult:
        if self.closed:
            raise KeyboardV3TerminalInputError("terminal input is closed")
        key = self._read_one()
        event: str | None = None
        xyz = [0.0, 0.0, 0.0]
        if key in self.KEY_TO_XYZ:
            now = float(self._clock())
            if (
                self._last_motion_time_s is None
                or now - self._last_motion_time_s
                >= self.motion_min_interval_s - 1.0e-12
            ):
                new_direction = self.KEY_TO_XYZ[key]
                new_axis = next(
                    axis for axis, value in enumerate(new_direction) if value != 0.0
                )
                # A shaped pulse may still have several control samples left
                # when the operator changes axes.  Never add that old tail to
                # the new key: W->E must emit only -Z, not a hidden X/Z
                # diagonal.  Same-axis repeats may overlap because they remain
                # axis pure and preserve the accepted pulse distance.
                active_axes = {
                    next(
                        axis
                        for axis, value in enumerate(direction)
                        if value != 0.0
                    )
                    for direction, _index in self._active_motion_pulses
                }
                if active_axes and active_axes != {new_axis}:
                    self._active_motion_pulses.clear()
                    self.cross_axis_tail_cancel_count += 1
                self._active_motion_pulses.append((new_direction, 0))
                self.maximum_concurrent_motion_pulses = max(
                    self.maximum_concurrent_motion_pulses,
                    len(self._active_motion_pulses),
                )
                self._last_motion_time_s = now
            else:
                self.suppressed_motion_key_count += 1
        elif key == "k":
            # V3 permits exactly one OPEN->CLOSE edge per episode and rejects
            # reopening. Make K idempotent so terminal key-repeat cannot turn
            # one intended close into REOPEN_AFTER_CLOSE_FORBIDDEN. Also stop
            # any remaining cosine-shaped translation tail at close onset;
            # otherwise a prior motion key can keep advancing during CLOSE.
            already_closed = self.gripper_closed
            self.gripper_closed = True
            self._active_motion_pulses.clear()
            event = "GRIPPER_CLOSE_HELD" if already_closed else "GRIPPER_CLOSE"
        elif key == "ENTER":
            event = "ENTER"
        elif key in tuple(str(value) for value in range(7)):
            event = f"RESULT_{key}"
        elif key == "r":
            event = "RESET_REQUEST"
        elif key in ("x", "\x1b"):
            event = "QUIT_REQUEST"
        remaining: list[tuple[tuple[float, float, float], int]] = []
        for direction, smoothing_index in self._active_motion_pulses:
            weight = self._smoothing_weights[smoothing_index]
            for axis in range(3):
                xyz[axis] += self.translation_step_m * weight * direction[axis]
            next_index = smoothing_index + 1
            if next_index < self.smoothing_steps:
                remaining.append((direction, next_index))
        self._active_motion_pulses = remaining
        xyz_norm = math.sqrt(sum(value * value for value in xyz))
        if xyz_norm > MAXIMUM_TRANSLATION_STEP_M + 1.0e-12:
            # Never hide an input-rate/configuration error by clipping it.
            raise KeyboardV3TerminalInputError(
                f"smoothed translation exceeds 4.5 mm authority: {xyz_norm}"
            )
        return TerminalPollResult(
            action_4d_metric_root_m=(
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                1.0 if self.gripper_closed else 0.0,
            ),
            event=event,
            key=key,
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._original_termios)
        finally:
            for signum, previous in self._previous_signal_handlers.items():
                signal.signal(signum, previous)
            self._previous_signal_handlers.clear()
            self.closed = True

    def __enter__(self) -> "KeyboardV3TerminalInput":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "KEYBOARD_V3_TERMINAL_INPUT_SCHEMA",
    "DEFAULT_SMOOTHING_STEPS",
    "MAXIMUM_SMOOTHING_STEPS",
    "DEFAULT_MOTION_MIN_INTERVAL_S",
    "DEFAULT_TRANSLATION_STEP_M",
    "MAXIMUM_TRANSLATION_STEP_M",
    "KeyboardV3TerminalInput",
    "KeyboardV3TerminalInputError",
    "TerminalPollResult",
]
