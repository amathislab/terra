"""Serialize the terrain geometry paired with a retargeted trajectory.

Terrain metadata files retain the plain :class:`TerrainSpec` JSON shape so terrain
readers can consume them directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from terra._files import atomic_write
from terra._musclemimic import TerrainSpec


@dataclass(frozen=True, slots=True)
class TerrainMetadata:
    """Terrain geometry stored beside a published trajectory.

    ``to_dict`` and ``save`` use the plain MuscleMimic TerrainSpec JSON
    representation. Load a published ``_terrain.json`` sidecar with
    :meth:`load`, or pass its path to ``terra retarget --terrain`` when
    retargeting another motion on known support geometry. A reconstruction
    cohort's full fit report is a different JSON record.
    """

    terrain: TerrainSpec

    @classmethod
    def from_terrain(cls, terrain: TerrainSpec) -> TerrainMetadata:
        """Create metadata for a terrain-aware retargeting result."""
        if not isinstance(terrain, TerrainSpec):
            raise TypeError(f"terrain must be a TerrainSpec, got {type(terrain).__name__}")
        return cls(terrain=terrain)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical plain-TerrainSpec JSON representation."""
        return self.terrain.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TerrainMetadata:
        """Parse a terrain metadata file in the current TerrainSpec shape."""
        if not isinstance(value, Mapping):
            raise ValueError("terrain metadata must be an object")
        unexpected = sorted((key for key in value if key not in {"boxes", "provenance"}), key=repr)
        if unexpected:
            raise ValueError(f"unexpected terrain metadata keys: {unexpected}")
        try:
            terrain = TerrainSpec.from_dict(dict(value))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid terrain geometry: {exc}") from exc
        return cls(terrain=terrain)

    @classmethod
    def load(cls, path: str | Path) -> TerrainMetadata:
        """Load and validate terrain metadata from JSON."""
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid terrain metadata JSON: {exc}") from exc
        return cls.from_dict(value)

    def save(self, path: str | Path) -> None:
        """Atomically write canonical, deterministic JSON."""
        destination = Path(path)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        atomic_write(
            destination,
            lambda temporary: temporary.write_text(payload, encoding="utf-8"),
        )


__all__ = ["TerrainMetadata"]
