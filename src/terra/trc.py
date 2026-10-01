"""TRC marker-trajectory parsing and MuscleMimic archive preparation."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

TrcUpAxis = Literal["y", "z"]

_MARKER_ARCHIVE_VERSION = 1


@dataclass(frozen=True)
class TrcData:
    """Parsed marker samples in metres and a Z-up coordinate frame.

    ``positions`` has shape (frames, markers, 3), ordered by ``labels``.
    Missing or all-zero marker triples are NaN. ``fps``, source frame numbers,
    and timestamps retain the recording's sampling information.
    """

    positions: np.ndarray
    labels: tuple[str, ...]
    fps: float
    frame_numbers: np.ndarray
    times: np.ndarray


def _units_divisor(unit: str) -> float:
    normalized = unit.strip().casefold()
    divisors = {
        "mm": 1000.0,
        "millimeter": 1000.0,
        "millimeters": 1000.0,
        "cm": 100.0,
        "centimeter": 100.0,
        "centimeters": 100.0,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
    }
    if normalized not in divisors:
        raise ValueError(f"Unsupported TRC unit {normalized!r}")
    return divisors[normalized]


def load_trc(path: str | Path, *, up_axis: TrcUpAxis = "y") -> TrcData:
    """Read and validate a TRC marker trajectory.

    The file header supplies units, frame count, marker labels, and frame rate.
    TRC records axis names but does not identify the laboratory's vertical axis:
    ``up_axis="y"`` applies the Gait120 ``[X, -Z, Y]`` transform;
    ``up_axis="z"`` preserves recorded XYZ. Coordinates are returned in
    metres and Z-up. Empty or all-zero marker triples become NaN, while
    timestamps and frame numbers must increase strictly.
    """

    if up_axis not in {"y", "z"}:
        raise ValueError("trc_up_axis must be 'y' or 'z'")

    trc_path = Path(path).expanduser().resolve()
    if not trc_path.is_file():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")
    if trc_path.suffix.casefold() != ".trc":
        raise ValueError(f"expected a .trc marker trajectory, got {trc_path.name!r}")

    lines = trc_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError(f"TRC has only {len(lines)} lines")

    meta_names = [part.strip() for part in lines[1].split("\t")]
    meta_values = [part.strip() for part in lines[2].split("\t")]
    metadata = dict(zip(meta_names, meta_values, strict=False))
    try:
        n_markers = int(float(metadata["NumMarkers"]))
        expected_frames = int(float(metadata["NumFrames"]))
        fps = float(metadata["DataRate"])
    except (KeyError, OverflowError, ValueError) as exc:
        raise ValueError("TRC metadata must define numeric DataRate, NumFrames, and NumMarkers") from exc
    if n_markers <= 0:
        raise ValueError("TRC NumMarkers must be positive")
    if expected_frames <= 0:
        raise ValueError("TRC NumFrames must be positive")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("TRC DataRate must be positive and finite")
    divisor = _units_divisor(metadata.get("Units", "mm"))

    label_columns = lines[3].split("\t")
    label_indices = [2 + 3 * marker_idx for marker_idx in range(n_markers)]
    if not label_indices or label_indices[-1] >= len(label_columns):
        raise ValueError("TRC marker-label header is shorter than NumMarkers")
    labels = tuple(label_columns[index].strip() for index in label_indices)
    if any(not label for label in labels):
        raise ValueError("TRC contains an empty marker label")

    frames: list[int] = []
    times: list[float] = []
    rows: list[list[float]] = []
    needed_columns = 2 + 3 * n_markers
    for line in lines[5:]:
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < needed_columns:
            columns.extend([""] * (needed_columns - len(columns)))
        try:
            frames.append(int(float(columns[0])))
            times.append(float(columns[1]))
        except ValueError as exc:
            raise ValueError(f"TRC contains an invalid frame number or timestamp: {line!r}") from exc
        values = []
        for value in columns[2:needed_columns]:
            try:
                values.append(float(value))
            except ValueError:
                values.append(float("nan"))
        rows.append(values)

    if len(rows) != expected_frames:
        raise ValueError(f"TRC header reports {expected_frames} frames but contains {len(rows)}")
    frame_numbers = np.asarray(frames, dtype=np.int64)
    timestamps = np.asarray(times, dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("TRC timestamps must be finite and strictly increasing")
    if np.any(np.diff(frame_numbers) <= 0):
        raise ValueError("TRC frame numbers must be strictly increasing")
    positions = np.asarray(rows, dtype=np.float32).reshape(-1, n_markers, 3) / divisor
    if up_axis == "y":
        positions = positions[..., [0, 2, 1]].copy()
        positions[..., 1] *= -1.0
    positions[np.all(np.isclose(positions, 0.0), axis=-1)] = np.nan
    return TrcData(
        positions=positions,
        labels=labels,
        fps=fps,
        frame_numbers=frame_numbers,
        times=timestamps,
    )


def prepare_trc_marker_archive(
    source_path: str | Path,
    cache_root: str | Path,
    *,
    up_axis: TrcUpAxis = "y",
) -> Path:
    """Cache a normalized TRC for the shared marker-fitting pipeline.

    The archive contains ``positions`` (metres, Z-up), ``labels``, ``fps``,
    source frame numbers and times, the chosen up-axis, and a schema version.
    Its filename hashes the source bytes and axis choice, so repeating the
    same conversion returns the existing archive path.
    """

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TRC file not found: {source}")
    digest = hashlib.sha256()
    digest.update(f"terra-trc-marker-archive-v{_MARKER_ARCHIVE_VERSION}\0{up_axis}\0".encode())
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    archive_root = Path(cache_root).expanduser().resolve() / ".trc_marker_cache"
    archive_path = archive_root / f"{digest.hexdigest()}.npz"
    if archive_path.is_file():
        return archive_path

    motion = load_trc(source, up_axis=up_axis)
    archive_root.mkdir(parents=True, exist_ok=True)
    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".trc-", suffix=".npz", dir=archive_root, delete=False) as stream:
            staged_path = Path(stream.name)
            np.savez_compressed(
                stream,
                positions=motion.positions,
                labels=np.asarray(motion.labels),
                fps=np.asarray(motion.fps, dtype=np.float32),
                frame_numbers=motion.frame_numbers,
                times=motion.times,
                trc_up_axis=np.asarray(up_axis),
                archive_version=np.asarray(_MARKER_ARCHIVE_VERSION, dtype=np.int64),
            )
        staged_path.replace(archive_path)
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
    return archive_path


__all__ = ["TrcData", "TrcUpAxis", "load_trc", "prepare_trc_marker_archive"]
