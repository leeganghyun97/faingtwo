# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Typed legacy-keyboard to canonical keyboard-v3 action conversion.

The Isaac keyboard device exposes the established eight-component physical
command ``[dx, dy, dz, rx, ry, rz, elbow, gripper]``.  Keyboard-v3 has a
different public schema: ``[dx, dy, dz, g]`` in robot-root metres.  This
module is the single, explicit boundary between those schemas.  It rejects
non-zero hidden axes and out-of-contract values; it never slices, clips, or
silently drops a component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geniesim.rl.sac.keyboard_v3_dataset import canonical_action_4d


LEGACY_KEYBOARD_COMPONENTS = (
    "dx_m",
    "dy_m",
    "dz_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
    "elbow_rad",
    "gripper_sign",
)
KEYBOARD_V3_COMPONENTS = ("dx_m", "dy_m", "dz_m", "g")


class KeyboardV3ActionAdapterError(ValueError):
    """Raised before an ambiguous legacy command can enter keyboard-v3."""


@dataclass(frozen=True)
class KeyboardV3ActionAdapterReceipt:
    action_4d: np.ndarray
    source_schema: str = "legacy_keyboard_physical_8d"
    destination_schema: str = "keyboard_v3_metric_4d"
    hidden_component_count: int = 0
    silent_clipping_count: int = 0


def adapt_legacy_keyboard_physical_8d(value: Any) -> KeyboardV3ActionAdapterReceipt:
    """Validate and explicitly map an established physical 8-D command.

    The legacy gripper sign contract is ``negative=CLOSE`` and
    ``non-negative=OPEN``.  The device must emit exact ``-1`` or ``+1``;
    accepting an arbitrary scalar here would silently create a new command
    contract.  Rotation and elbow must be exact zero because keyboard-v3 does
    not own those axes.
    """

    legacy = np.asarray(value, dtype=np.float64)
    if legacy.shape != (8,) or not np.isfinite(legacy).all():
        raise KeyboardV3ActionAdapterError(
            "legacy keyboard command must contain exactly eight finite values"
        )
    forbidden = legacy[3:7]
    if not np.array_equal(forbidden, np.zeros(4, dtype=np.float64)):
        raise KeyboardV3ActionAdapterError(
            "legacy rotation/elbow components must be exact zero for keyboard-v3"
        )
    gripper_sign = float(legacy[7])
    if gripper_sign not in (-1.0, 1.0):
        raise KeyboardV3ActionAdapterError(
            "legacy gripper sign must be exact OPEN=+1 or CLOSE=-1"
        )
    action = canonical_action_4d(
        np.asarray(
            [legacy[0], legacy[1], legacy[2], 1.0 if gripper_sign < 0.0 else 0.0],
            dtype=np.float64,
        )
    )
    return KeyboardV3ActionAdapterReceipt(action_4d=action)


__all__ = [
    "KEYBOARD_V3_COMPONENTS",
    "LEGACY_KEYBOARD_COMPONENTS",
    "KeyboardV3ActionAdapterError",
    "KeyboardV3ActionAdapterReceipt",
    "adapt_legacy_keyboard_physical_8d",
]
