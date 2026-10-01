"""Adapters from official PRISM takes to TERRA terrain inputs."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_take(path: Path) -> dict[str, Any]:
    """Load an official PRISM pickle.

    Pickle permits code execution. The caller is responsible for using only files
    obtained from the official PRISM release.
    """
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def observed_support_points(take: Mapping[str, Any], *, stride: int = 1) -> np.ndarray:
    """Return world-frame CoP samples for frames with measured insole activity."""
    if stride < 1:
        raise ValueError("stride must be positive")
    points = []
    for side in ("L_Foot", "R_Foot"):
        foot = take["insole"][side]
        contacts = np.asarray(foot["contacts"], dtype=bool)
        active = np.any(contacts, axis=1)
        cop = np.asarray(foot["CoP_world"], dtype=float)
        if cop.shape != (len(active), 3):
            raise ValueError(f"{side} CoP_world must have shape ({len(active)}, 3), got {cop.shape}")
        valid = active & np.all(np.isfinite(cop), axis=1)
        points.append(cop[valid][::stride])
    return np.concatenate(points, axis=0) if points else np.empty((0, 3))


def observed_support_xy(take: Mapping[str, Any], *, stride: int = 1) -> np.ndarray:
    """Return world-frame CoP XY samples for frames with measured insole activity."""
    return observed_support_points(take, stride=stride)[:, :2]


def selected_take_paths(data_root: Path, take_ids: Sequence[str] = ()) -> list[Path]:
    """Select subject/take paths, accepting IDs as ``subj001/take002``."""
    from .root import resolve_data_root

    root = resolve_data_root(Path(data_root))
    if not take_ids:
        return sorted(root.glob("subj*/take*.pkl"))
    paths = []
    for take_id in take_ids:
        subject, separator, take = take_id.replace("_", "/", 1).partition("/")
        if not separator:
            raise ValueError(f"take ID must be subjNNN/takeNNN, got {take_id!r}")
        path = root / subject / f"{take.removesuffix('.pkl')}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths
