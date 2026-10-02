"""Lightweight public data contracts shared across TERRA boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from terra._methods import RetargetingMethod

if TYPE_CHECKING:
    from terra._musclemimic import TerrainSpec, Trajectory


@dataclass(frozen=True, slots=True)
class RetargetResult:
    """In-memory output of one retargeting run.

    ``trajectory`` is ready for playback after motion extension. ``analysis``
    records method, source, model, solver, and terrain settings. ``terrain`` is
    the geometry that must accompany playback, or ``None`` for flat ground. Use
    :func:`terra.save_retarget_result` to publish the fields atomically.
    """

    trajectory: Trajectory
    analysis: dict[str, object]
    terrain: TerrainSpec | None
    method: RetargetingMethod
    source_path: Path


__all__ = ["RetargetResult"]
