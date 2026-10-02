"""Locate trusted PRISM takes inside the supported extraction layouts."""

from __future__ import annotations

from pathlib import Path


def resolve_data_root(path: Path) -> Path:
    """Resolve either the extraction root or the directory containing subjects."""
    path = path.expanduser().resolve()
    candidates = (
        path,
        path / "data" / "PRISM",
        path / "PRISM" / "data" / "PRISM",
    )
    for candidate in candidates:
        if any(candidate.glob("subj*/take*.pkl")):
            return candidate
    raise FileNotFoundError(f"could not find subj*/take*.pkl below {path}; pass the extracted PRISM root")


__all__ = ["resolve_data_root"]
