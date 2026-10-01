"""Named dataset selections are optional; the code release ships none."""

from __future__ import annotations

from pathlib import Path

DATASET_SELECTION_NAMES: tuple[str, ...] = ()


def bundled_dataset_selection(name: str) -> Path:
    """Explain how to supply a motion selection in the code-only release."""

    raise FileNotFoundError(
        f"no bundled dataset selection {name!r}; pass a CSV path with --selection-manifest"
    )


__all__ = ["DATASET_SELECTION_NAMES", "bundled_dataset_selection"]
