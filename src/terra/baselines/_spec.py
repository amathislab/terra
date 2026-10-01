"""Immutable metadata shared by TERRA's comparison baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class BaselineSpec:
    """Describe one retargeting method used for comparison."""

    key: str
    label: str
    dependency_extra: str | None = None
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or "/" in self.key or self.key in {".", "..", "terra"}:
            raise ValueError(f"invalid baseline key: {self.key!r}")
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    def resolved_config(self, overrides: Mapping[str, object] | None = None) -> dict[str, object]:
        """Return a mutable invocation config without mutating the baseline definition."""
        resolved = dict(self.config)
        resolved.update(overrides or {})
        return resolved
