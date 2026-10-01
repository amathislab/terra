"""Pure geometry helpers for publication-scene assembly."""

from __future__ import annotations

import math

import numpy as np

from terra.figures.manifest import CellSpec, LayoutSpec


def rotation_z(degrees: float) -> np.ndarray:
    """Return a three-dimensional right-handed Z rotation."""

    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def resolve_frames(cell: CellSpec, n_frames: int) -> tuple[int, ...]:
    """Resolve explicit indices or normalized fractions against a trajectory."""

    if n_frames < 1:
        raise ValueError("trajectory must contain at least one frame")
    if cell.frames is not None:
        if cell.frames[-1] >= n_frames:
            raise ValueError(
                f"{cell.motion}: frame {cell.frames[-1]} lies outside a {n_frames}-frame trajectory"
            )
        return cell.frames
    assert cell.fractions is not None
    resolved = tuple(round(fraction * (n_frames - 1)) for fraction in cell.fractions)
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError(f"{cell.motion}: frame fractions collapse to duplicate indices")
    return resolved


def resolve_highlight_frame(cell: CellSpec, frames: tuple[int, ...]) -> int:
    """Return the opaque frame, retaining the historical last-frame default."""

    highlight = frames[-1] if cell.highlight_frame is None else cell.highlight_frame
    if highlight not in frames:
        raise ValueError(f"{cell.motion}: highlighted frame {highlight} is absent from resolved frames")
    return highlight


def history_alpha_by_frame(
    frames: tuple[int, ...],
    highlight: int,
    alpha_values: tuple[float, ...],
) -> dict[int, float]:
    """Assign the strongest ghost opacity to the temporally nearest sample."""

    ghosts = sorted((frame for frame in frames if frame != highlight), key=lambda frame: abs(frame - highlight), reverse=True)
    offset = max(0, len(alpha_values) - len(ghosts))
    return {
        frame: alpha_values[min(offset + index, len(alpha_values) - 1)]
        for index, frame in enumerate(ghosts)
    }


def alignment_yaw(root_xy: np.ndarray, frames: tuple[int, ...], override: float | None) -> float:
    """Rigid yaw that aligns a selected motion window with world +X."""

    if override is not None:
        return float(override)
    lo, hi = frames[0], frames[-1]
    path = np.asarray(root_xy[lo : hi + 1], dtype=float)
    if len(path) < 2:
        return 0.0
    centered = path - np.mean(path, axis=0)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] < 1e-6:
        return 0.0
    axis = vh[0]
    displacement = np.asarray(root_xy[hi] - root_xy[lo], dtype=float)
    if np.linalg.norm(displacement) > 0.05:
        if np.dot(axis, displacement) < 0:
            axis = -axis
    elif axis[0] < 0:
        axis = -axis
    return -math.degrees(math.atan2(axis[1], axis[0]))


def cell_origin(layout: LayoutSpec, row: int, column: int) -> np.ndarray:
    """Metric grid origin; row zero is the visually distant row by convention."""

    x = (column - (layout.columns - 1) / 2.0) * layout.cell_pitch[0]
    y = ((layout.rows - 1) / 2.0 - row) * layout.cell_pitch[1]
    return np.asarray((x, y, 0.0))


__all__ = [
    "alignment_yaw",
    "cell_origin",
    "history_alpha_by_frame",
    "resolve_frames",
    "resolve_highlight_frame",
    "rotation_z",
]
