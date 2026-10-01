"""Locate the dataset policies shipped with TERRA.

Dataset TOMLs describe conversion-independent processing policy. They are package
data so an installed ``terra`` command and a source checkout use the same files.
Motion data, generated artifacts, and body models remain outside the package and
are resolved through :class:`terra.paths.StorageRoots`.
"""

from __future__ import annotations

from pathlib import Path

DATASET_CONFIG_NAMES = (
    "amass",
    "darmstadt",
    "gait120",
    "prism",
    "vielemeyer",
)


def bundled_dataset_config(name: str) -> Path:
    """Return one validated package-owned dataset config path."""

    normalized = name.strip().casefold()
    if normalized not in DATASET_CONFIG_NAMES:
        choices = ", ".join(DATASET_CONFIG_NAMES)
        raise FileNotFoundError(f"unknown bundled dataset config {name!r}; expected one of {choices}")
    path = Path(__file__).with_name("configs") / f"{normalized}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"installed package is missing dataset config: {path}")
    return path.resolve()


__all__ = ["DATASET_CONFIG_NAMES", "bundled_dataset_config"]
