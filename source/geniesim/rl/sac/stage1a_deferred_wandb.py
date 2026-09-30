# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lazy W&B materialization for measured-reset-gated Stage-1A runs."""

from __future__ import annotations

from typing import Any, Callable


class DeferredWandbRun:
    """Create the external W&B run only when the first metric is logged.

    Reset-fixed Stage-1A performs lengthy reset-only physics before policy
    exposure is legal.  Constructing a W&B run before that gate produces an
    externally visible training run even if measured OPEN parity later fails.
    This proxy preserves the ordinary run interface but materializes it only
    on the first post-gate ``log`` (or when a completed report asks for its
    identity).  ``finish`` before materialization is deliberately a no-op.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._run: Any | None = None
        self._initialization_count = 0

    @property
    def initialized(self) -> bool:
        return self._run is not None

    @property
    def initialization_count(self) -> int:
        return self._initialization_count

    def _materialize(self) -> Any:
        if self._run is None:
            self._run = self._factory()
            self._initialization_count += 1
        return self._run

    def log(self, *args: Any, **kwargs: Any) -> Any:
        return self._materialize().log(*args, **kwargs)

    @property
    def summary(self) -> Any:
        return self._materialize().summary

    @property
    def id(self) -> Any:
        return self._materialize().id

    @property
    def url(self) -> Any:
        return self._materialize().url

    def finish(self, *args: Any, **kwargs: Any) -> Any | None:
        if self._run is None:
            return None
        return self._run.finish(*args, **kwargs)


__all__ = ["DeferredWandbRun"]
