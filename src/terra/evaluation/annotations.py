"""Per-frame evaluator arrays consumed by visualization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

# Array key -> (rank, NumPy dtype kind). All arrays use the trajectory frame as axis 0;
# rank-two arrays use the stable [left, right] ordering on axis 1.
RENDER_ARRAY_SPECS: dict[str, tuple[int, str]] = {
    "contact_gap": (1, "b"),
    "environment_penetrating": (1, "b"),
    "forefoot_up": (1, "b"),
    "hindfoot_up": (1, "b"),
    "self_collision_bad": (1, "b"),
    "skating": (1, "b"),
    "sole_vertical_clearance_m": (2, "f"),
    "source_stance": (2, "b"),
    "stance_slip_failure": (1, "b"),
    "support_floating": (2, "b"),
    "support_penetrating": (2, "b"),
    "swing_clearance_failure": (1, "b"),
    "swing_scraping": (1, "b"),
}
FRAME_METADATA_KEYS = frozenset({"method", "motion", "n_frames"})


class AnnotationError(ValueError):
    """Raised when visualization annotations are missing or malformed."""


def frame_metadata(*, motion: str, method: str, n_frames: int) -> dict[str, np.ndarray]:
    """Return the identifying scalars embedded in a frame archive."""

    return {
        "method": np.asarray(method),
        "motion": np.asarray(motion),
        "n_frames": np.asarray(n_frames),
    }


def validate_frame_archive(
    archive: Mapping[str, Any],
    *,
    motion: str,
    n_frames: int,
    method: str | None = None,
) -> None:
    """Validate one evaluator frame archive before rendering it."""

    files = set(getattr(archive, "files", archive))
    missing = (set(RENDER_ARRAY_SPECS) | FRAME_METADATA_KEYS) - files
    if missing:
        raise AnnotationError(f"frame annotations are incomplete: missing {sorted(missing)}")

    def scalar(key: str) -> Any:
        value = np.asarray(archive[key])
        if value.shape != ():
            raise AnnotationError(f"frame metadata {key!r} must be scalar")
        return value.item()

    for key, expected in {"motion": motion, "n_frames": n_frames}.items():
        if scalar(key) != expected:
            raise AnnotationError(
                f"frame annotation {key!r} does not match: expected {expected!r}, found {scalar(key)!r}"
            )
    recorded_method = str(scalar("method"))
    if not recorded_method:
        raise AnnotationError("frame annotation method is empty")
    if method is not None and recorded_method != method:
        raise AnnotationError(f"frame annotations are for method {recorded_method!r}, not requested method {method!r}")

    for key, (ndim, dtype_kind) in RENDER_ARRAY_SPECS.items():
        value = np.asarray(archive[key])
        if value.ndim != ndim or len(value) != n_frames:
            raise AnnotationError(f"frame annotation {key!r} must have shape ({n_frames}, ...), found {value.shape}")
        if ndim == 2 and value.shape[1] != 2:
            raise AnnotationError(f"frame annotation {key!r} must use [left, right] columns")
        if value.dtype.kind != dtype_kind:
            raise AnnotationError(
                f"frame annotation {key!r} must have dtype kind {dtype_kind!r}, found {value.dtype.kind!r}"
            )


__all__ = [
    "FRAME_METADATA_KEYS",
    "RENDER_ARRAY_SPECS",
    "AnnotationError",
    "frame_metadata",
    "validate_frame_archive",
]
