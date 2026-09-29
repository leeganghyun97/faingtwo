# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Display-only v1-parity operator views for keyboard-v3 collection.

The perspective viewport and the two task-camera panels never alter recorded
sensor render products or the policy observation.  Omniverse imports are
delayed until ``create`` so static tests and offline replay remain Isaac-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geniesim.rl.sac.keyboard_v3_dataset import KEYBOARD_V3_OPERATOR_VIEW_NAMES

from ..g2_asset_camera_pose import (
    HEAD_CAMERA_RELATIVE_PRIM,
    RIGHT_WRIST_CAMERA_RELATIVE_PRIM,
)


PERSPECTIVE_BASELINE_EYE_M = (1.45, -1.45, 1.65)
PERSPECTIVE_TASK_FOCUS_M = (0.50, -0.23, 0.82)
PERSPECTIVE_DISTANCE_SCALE = 0.5
PERSPECTIVE_OPERATOR_EYE_M = tuple(
    focus + PERSPECTIVE_DISTANCE_SCALE * (position - focus)
    for position, focus in zip(
        PERSPECTIVE_BASELINE_EYE_M, PERSPECTIVE_TASK_FOCUS_M, strict=True
    )
)


class KeyboardV3OperatorViewError(RuntimeError):
    pass


@dataclass
class KeyboardV3OperatorViews:
    """Own exactly perspective + head + right-wrist collection views."""

    windows: list[Any] = field(default_factory=list)
    created: bool = False
    bindings: dict[str, str] = field(default_factory=dict)

    def create(self) -> None:
        if self.created:
            raise KeyboardV3OperatorViewError("operator views already created")
        import omni.kit.viewport.utility as viewport_utility
        import omni.usd
        from isaacsim.core.rendering_manager import ViewportManager
        from pxr import Sdf, UsdGeom

        paths = {
            "perspective": "/OmniverseKit_Persp",
            "head": f"/World/envs/env_0/{HEAD_CAMERA_RELATIVE_PRIM}",
            "right_wrist": f"/World/envs/env_0/{RIGHT_WRIST_CAMERA_RELATIVE_PRIM}",
        }
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise KeyboardV3OperatorViewError("operator-view stage unavailable")
            for name in ("head", "right_wrist"):
                prim = stage.GetPrimAtPath(paths[name])
                if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                    raise KeyboardV3OperatorViewError(
                        f"operator camera prim invalid: {name}:{paths[name]}"
                    )

            ViewportManager.set_camera_view(
                paths["perspective"],
                eye=list(PERSPECTIVE_OPERATOR_EYE_M),
                target=list(PERSPECTIVE_TASK_FOCUS_M),
            )
            primary = viewport_utility.get_active_viewport()
            if primary is None:
                raise KeyboardV3OperatorViewError(
                    "primary operator viewport unavailable"
                )
            primary.set_active_camera(paths["perspective"])
            if str(primary.get_active_camera()) != paths["perspective"]:
                raise KeyboardV3OperatorViewError(
                    "perspective viewport binding failed"
                )

            for index, (label, name) in enumerate(
                (
                    ("G2 Head Camera", "head"),
                    ("G2 Right Wrist Camera", "right_wrist"),
                )
            ):
                window = viewport_utility.create_viewport_window(
                    name=label,
                    width=512,
                    height=384,
                    position_x=30 + 540 * index,
                    position_y=90,
                    camera_path=Sdf.Path(paths[name]),
                )
                if window is None:
                    raise KeyboardV3OperatorViewError(
                        f"operator viewport creation failed: {name}"
                    )
                window.viewport_api.resolution = (256, 192)
                if str(window.viewport_api.get_active_camera()) != paths[name]:
                    window.viewport_api.set_active_camera(Sdf.Path(paths[name]))
                if str(window.viewport_api.get_active_camera()) != paths[name]:
                    raise KeyboardV3OperatorViewError(
                        f"operator viewport binding failed: {name}"
                    )
                self.windows.append(window)
            self.created = True
            self.bindings = dict(paths)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        while self.windows:
            window = self.windows.pop()
            viewport_api = getattr(window, "viewport_api", None)
            if viewport_api is not None:
                try:
                    viewport_api.updates_enabled = False
                except Exception:
                    pass
            destroy = getattr(window, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass
        self.created = False

    def receipt(self) -> dict[str, Any]:
        """Return display bindings without changing policy observations."""

        return {
            "created": bool(self.created),
            "head_camera_binding": self.bindings.get("head"),
            "wrist_camera_binding": self.bindings.get("right_wrist"),
            "head_rgb_visible": bool(self.created and len(self.windows) >= 1),
            "wrist_rgb_visible": bool(self.created and len(self.windows) >= 2),
            "policy_observation_changed": False,
        }


__all__ = [
    "KEYBOARD_V3_OPERATOR_VIEW_NAMES",
    "KeyboardV3OperatorViewError",
    "KeyboardV3OperatorViews",
    "PERSPECTIVE_OPERATOR_EYE_M",
    "PERSPECTIVE_TASK_FOCUS_M",
]
