"""Audit Gait120 and build TERRA-ready AMASS/SMPL-H trajectories.

The original dataset stores up to two TRC gait cycles per locomotion trial and
one transition per stool trial.  The companion ``Gait120-EMG`` tree stores the
same subject/movement/trial/step hierarchy in SciPy-readable
``ConvertedData.mat`` files.  This script joins those sources by their shared
key, concatenates the available TRC steps, fits SMPL-H with MoSh++-style, and
writes AMASS-like ``.npz`` files.

The Stage-I shape/marker calibration is reused for every motion from a subject.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from terra._revision import write_git_commit
from terra.datasets.biomechanics import (
    biomechanics_path,
    empty_grf,
    motion_clock,
    resample_linear,
    validate_biomechanics,
    write_biomechanics,
)
from terra.paths import StorageRoots
from terra.trc import TrcData, load_trc


@dataclass(frozen=True)
class MovementMetadata:
    expected_family: str
    terrain_class: str
    steps: tuple[int, ...]


MOVEMENT_METADATA = {
    "LevelWalking": MovementMetadata("flat", "flat", (1, 2)),
    "SlopeAscent": MovementMetadata("ramp", "ramp_up", (1, 2)),
    "SlopeDescent": MovementMetadata("ramp", "ramp_down", (1, 2)),
    "StairAscent": MovementMetadata("steps", "stairs_up", (1, 2)),
    "StairDescent": MovementMetadata("steps", "stairs_down", (1, 2)),
    # The Gait120 protocol extracts one complete transition, not two gait
    # cycles, for each stool trial.
    "SitToStand": MovementMetadata("steps", "chair_sit", (1,)),
    "StandToSit": MovementMetadata("steps", "chair_sit", (1,)),
}
TARGET_MOVEMENTS = tuple(MOVEMENT_METADATA)
CHAIR_MOVEMENTS = ("SitToStand", "StandToSit")
CHAIR_SELECTION_SEED = "terra-gait120-chair-v1"
EXPECTED_EMG_CHANNELS = (
    "VastusLateralis",
    "RectusFemoris",
    "VastusMedialis",
    "TibialisAnterior",
    "BicepsFemoris",
    "Semitendinosus",
    "GastrocnemuisMedialis",
    "GastrocnemiusLateralis",
    "SoleusMedialis",
    "SoleusLateralis",
    "PeroneusLongus",
    "PeroneusBrevis",
)
GAIT120_EMG_FPS = 2000.0
AUDIT_VERSION = 1
MARKER_ARCHIVE_VERSION = 1


@dataclass
class StepRecord:
    subject: int
    movement: str
    trial: int
    step: int
    trc_path: str
    mot_path: str
    emg_path: str
    trc_exists: bool = False
    trc_valid: bool = False
    trc_frames: int = 0
    marker_count: int = 0
    marker_fps: float | None = None
    mot_exists: bool = False
    emg_exists: bool = False
    emg_valid: bool = False
    emg_channels: int = 0
    emg_samples_min: int = 0
    emg_samples_max: int = 0
    emg_native_samples_min: int = 0
    emg_native_samples_max: int = 0
    emg_all_finite: bool = False
    usable_pair: bool = False
    error: str = ""


@dataclass
class ClipRecord:
    subject: int
    movement: str
    trial: int
    paired_steps: str
    marker_archive: str
    emg_archive: str
    output_path: str
    motion: str
    emg_path: str
    trc_paths: str
    mot_paths: str
    dataset: str = "gait120"
    expected_family: str = ""
    terrain_class: str = ""
    biomechanics_archive: str = ""
    source_frames: int = 0
    output_frames: int = 0
    fps: float | None = None
    marker_error_mean_mm: float | None = None
    marker_error_p95_mm: float | None = None
    marker_error_max_mm: float | None = None
    role: str = "retarget"
    fit_passed: bool = False
    fit_failure_reason: str = "not_validated"
    status: str = "pending"
    error: str = ""


def _subject_block(subject: int) -> str:
    start = ((subject - 1) // 10) * 10 + 1
    return f"Gait120_{start:03d}_to_{start + 9:03d}"


def _parse_int_ranges(value: str, *, lower: int, upper: int) -> list[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise argparse.ArgumentTypeError(f"Invalid integer/range {part!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end or start < lower or end > upper:
            raise argparse.ArgumentTypeError(f"Range {part!r} must be within {lower}-{upper}")
        selected.update(range(start, end + 1))
    if not selected:
        raise argparse.ArgumentTypeError("At least one value is required")
    return sorted(selected)


def load_gait120_trc(path: str | Path) -> TrcData:
    """Read one Gait120 TRC and convert its Y-up coordinates to Z-up metres.

    Gait120's files use ``X=lateral, Y=up, Z=progression``.  The existing
    level-walking C3Ds in ``Gait120-EMG`` encode the same samples as
    ``[X, -Z, Y]``.  Applying that exact transform keeps the new non-flat
    motions in the coordinate system already validated by the SMPL fitter and
    expected by TERRA.
    """

    return load_trc(path, up_axis="y")


def _mat_record(value: Any) -> np.void:
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.reshape(-1)[0]
    if not isinstance(current, np.void) or current.dtype.names is None:
        raise ValueError(f"Expected a scalar MATLAB struct, got {type(current).__name__}")
    return current


def _numeric_mat_value(value: Any) -> np.ndarray:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.reshape(-1)[0]
    return np.asarray(current, dtype=np.float64)


def _emg_step_metadata(movement_record: np.void, trial: int, step: int) -> dict[str, Any]:
    trial_name = f"Trial{trial:02d}"
    step_name = f"Step{step:02d}"
    if trial_name not in (movement_record.dtype.names or ()):
        raise KeyError(trial_name)
    trial_record = _mat_record(movement_record[trial_name])
    if step_name not in (trial_record.dtype.names or ()):
        raise KeyError(f"{trial_name}/{step_name}")
    step_record = _mat_record(trial_record[step_name])
    sample_counts_by_field: dict[str, list[int]] = {}
    channels_by_field: dict[str, tuple[str, ...]] = {}
    all_finite = True
    for field in ("EMGs_interpolated", "EMGs_norm"):
        if field not in (step_record.dtype.names or ()):
            raise KeyError(f"{trial_name}/{step_name}/{field}")
        emg_record = _mat_record(step_record[field])
        channels = tuple(emg_record.dtype.names or ())
        channels_by_field[field] = channels
        sample_counts_by_field[field] = []
        for channel in channels:
            values = _numeric_mat_value(emg_record[channel])
            sample_counts_by_field[field].append(int(values.size))
            all_finite &= bool(values.size and np.isfinite(values).all())
    channels = channels_by_field["EMGs_interpolated"]
    sample_counts = sample_counts_by_field["EMGs_interpolated"]
    native_counts = sample_counts_by_field["EMGs_norm"]
    return {
        "channels": channels,
        "samples_min": min(sample_counts, default=0),
        "samples_max": max(sample_counts, default=0),
        "native_samples_min": min(native_counts, default=0),
        "native_samples_max": max(native_counts, default=0),
        "all_finite": all_finite,
        "valid": (
            channels == EXPECTED_EMG_CHANNELS
            and channels_by_field["EMGs_norm"] == EXPECTED_EMG_CHANNELS
            and sample_counts == [101] * len(EXPECTED_EMG_CHANNELS)
            and bool(native_counts)
            and min(native_counts) > 0
            and len(set(native_counts)) == 1
            and all_finite
        ),
    }


def _emg_step_values(movement_record: np.void, trial: int, step: int) -> tuple[tuple[str, ...], np.ndarray]:
    """Return one processed step as ``(samples, channels)`` float32 values."""

    trial_name = f"Trial{trial:02d}"
    step_name = f"Step{step:02d}"
    if trial_name not in (movement_record.dtype.names or ()):
        raise KeyError(trial_name)
    trial_record = _mat_record(movement_record[trial_name])
    if step_name not in (trial_record.dtype.names or ()):
        raise KeyError(f"{trial_name}/{step_name}")
    step_record = _mat_record(trial_record[step_name])
    if "EMGs_interpolated" not in (step_record.dtype.names or ()):
        raise KeyError(f"{trial_name}/{step_name}/EMGs_interpolated")
    emg_record = _mat_record(step_record["EMGs_interpolated"])
    channels = tuple(emg_record.dtype.names or ())
    arrays = [_numeric_mat_value(emg_record[channel]).reshape(-1) for channel in channels]
    lengths = {array.size for array in arrays}
    if channels != EXPECTED_EMG_CHANNELS:
        raise ValueError(f"Unexpected EMG channels: {channels}")
    if lengths != {101}:
        raise ValueError(f"Expected 101 interpolated samples per channel, got {sorted(lengths)}")
    values = np.stack(arrays, axis=-1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("EMG values contain NaN or infinity")
    return channels, values


def _emg_native_step_values(
    movement_record: np.void,
    trial: int,
    step: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the MVC-normalized EMG envelope at its native 2 kHz clock."""

    trial_name = f"Trial{trial:02d}"
    step_name = f"Step{step:02d}"
    trial_record = _mat_record(movement_record[trial_name])
    step_record = _mat_record(trial_record[step_name])
    emg_record = _mat_record(step_record["EMGs_norm"])
    channels = tuple(emg_record.dtype.names or ())
    arrays = [_numeric_mat_value(emg_record[channel]).reshape(-1) for channel in channels]
    lengths = {array.size for array in arrays}
    if channels != EXPECTED_EMG_CHANNELS or len(lengths) != 1:
        raise ValueError(f"Unexpected native EMG layout: channels={channels}, lengths={sorted(lengths)}")
    values = np.stack(arrays, axis=-1).astype(np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Native EMG is empty or contains NaN/infinity")
    return channels, values


def _load_subject_emg_metadata(emg_path: Path, movements: tuple[str, ...]) -> dict[tuple[str, int, int], dict]:
    if not emg_path.exists():
        return {}
    loaded = loadmat(emg_path, variable_names=list(movements), struct_as_record=True, squeeze_me=False)
    output: dict[tuple[str, int, int], dict] = {}
    for movement in movements:
        if movement not in loaded:
            continue
        movement_record = _mat_record(loaded[movement])
        for trial in range(1, 6):
            for step in MOVEMENT_METADATA[movement].steps:
                try:
                    output[(movement, trial, step)] = _emg_step_metadata(movement_record, trial, step)
                except (KeyError, ValueError, TypeError, IndexError) as exc:
                    output[(movement, trial, step)] = {"valid": False, "error": str(exc)}
    return output


def audit_dataset(
    *,
    original_root: Path,
    emg_root: Path,
    output_root: Path,
    subjects: list[int],
    movements: tuple[str, ...],
    trials: list[int],
) -> tuple[list[StepRecord], list[ClipRecord], dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    step_records: list[StepRecord] = []

    for subject in subjects:
        emg_path = emg_root / _subject_block(subject) / f"S{subject:03d}" / "EMG" / "ConvertedData.mat"
        try:
            emg_metadata = _load_subject_emg_metadata(emg_path, movements)
            emg_load_error = ""
        except Exception as exc:  # keep auditing kinematics when one MAT file is corrupt
            emg_metadata = {}
            emg_load_error = f"{type(exc).__name__}: {exc}"

        subject_root = original_root / f"S{subject:03d}"
        for movement in movements:
            movement_spec = MOVEMENT_METADATA[movement]
            for trial in trials:
                for step in movement_spec.steps:
                    trc_path = (
                        subject_root / "MotionCapture" / movement / "TRC" / f"Trial{trial:02d}" / f"Step{step:02d}.trc"
                    )
                    mot_path = (
                        subject_root / "MotionCapture" / movement / "MOT" / f"Trial{trial:02d}" / f"Step{step:02d}.mot"
                    )
                    record = StepRecord(
                        subject=subject,
                        movement=movement,
                        trial=trial,
                        step=step,
                        trc_path=str(trc_path),
                        mot_path=str(mot_path),
                        emg_path=str(emg_path),
                        trc_exists=trc_path.exists(),
                        mot_exists=mot_path.exists(),
                        emg_exists=emg_path.exists(),
                    )
                    errors: list[str] = []
                    if trc_path.exists():
                        try:
                            trc = load_gait120_trc(trc_path)
                            record.trc_valid = True
                            record.trc_frames = len(trc.positions)
                            record.marker_count = len(trc.labels)
                            record.marker_fps = trc.fps
                        except Exception as exc:
                            errors.append(f"TRC {type(exc).__name__}: {exc}")
                    emg_step = emg_metadata.get((movement, trial, step))
                    if emg_step is not None:
                        record.emg_valid = bool(emg_step.get("valid"))
                        record.emg_channels = len(emg_step.get("channels", ()))
                        record.emg_samples_min = int(emg_step.get("samples_min", 0))
                        record.emg_samples_max = int(emg_step.get("samples_max", 0))
                        record.emg_native_samples_min = int(emg_step.get("native_samples_min", 0))
                        record.emg_native_samples_max = int(emg_step.get("native_samples_max", 0))
                        record.emg_all_finite = bool(emg_step.get("all_finite"))
                        if record.trc_valid:
                            expected_native = round(record.trc_frames * GAIT120_EMG_FPS / record.marker_fps)
                            if (
                                record.emg_native_samples_min != expected_native
                                or record.emg_native_samples_max != expected_native
                            ):
                                record.emg_valid = False
                                errors.append(
                                    f"EMG native samples {record.emg_native_samples_min}-"
                                    f"{record.emg_native_samples_max}, expected {expected_native}"
                                )
                        if emg_step.get("error"):
                            errors.append(f"EMG {emg_step['error']}")
                    elif emg_load_error:
                        errors.append(f"EMG {emg_load_error}")
                    record.usable_pair = record.trc_valid and record.emg_valid
                    record.error = "; ".join(errors)
                    step_records.append(record)

    clips: list[ClipRecord] = []
    grouped: dict[tuple[int, str, int], list[StepRecord]] = defaultdict(list)
    for record in step_records:
        grouped[(record.subject, record.movement, record.trial)].append(record)
    for (subject, movement, trial), steps in sorted(grouped.items()):
        paired = sorted((record for record in steps if record.usable_pair), key=lambda record: record.step)
        if not paired:
            continue
        rel = Path("Gait120") / f"S{subject:03d}" / movement / f"Trial{trial:02d}"
        marker_archive = output_root / ".markers" / rel / "AllSteps_markers.npz"
        emg_archive = output_root / rel / "AllSteps_emg.npz"
        output_path = output_root / rel / "AllSteps_stageii.npz"
        biomechanics_archive = biomechanics_path(output_path)
        motion = str(rel / "AllSteps_stageii")
        clips.append(
            ClipRecord(
                subject=subject,
                movement=movement,
                trial=trial,
                paired_steps=";".join(str(record.step) for record in paired),
                marker_archive=str(marker_archive),
                emg_archive=str(emg_archive),
                output_path=str(output_path),
                motion=motion,
                emg_path=paired[0].emg_path,
                trc_paths=";".join(record.trc_path for record in paired),
                mot_paths=";".join(record.mot_path for record in paired if record.mot_exists),
                expected_family=MOVEMENT_METADATA[movement].expected_family,
                terrain_class=MOVEMENT_METADATA[movement].terrain_class,
                biomechanics_archive=str(biomechanics_archive),
                role="calibration" if movement == "LevelWalking" else "retarget",
            )
        )

    summary_by_movement = {}
    for movement in movements:
        rows = [row for row in step_records if row.movement == movement]
        movement_clips = [clip for clip in clips if clip.movement == movement]
        summary_by_movement[movement] = {
            "expected_steps": len(subjects) * len(trials) * len(MOVEMENT_METADATA[movement].steps),
            "trc_steps": sum(row.trc_exists for row in rows),
            "valid_trc_steps": sum(row.trc_valid for row in rows),
            "mot_steps": sum(row.mot_exists for row in rows),
            "valid_emg_steps": sum(row.emg_valid for row in rows),
            "paired_marker_emg_steps": sum(row.usable_pair for row in rows),
            "paired_clips": len(movement_clips),
            "subjects_with_paired_clip": len({clip.subject for clip in movement_clips}),
            "complete_subjects_five_trials": sum(
                all(
                    any(clip.subject == subject and clip.movement == movement and clip.trial == trial for clip in clips)
                    for trial in trials
                )
                for subject in subjects
            ),
        }

    audit = {
        "audit_version": AUDIT_VERSION,
        "original_root": str(original_root.resolve()),
        "emg_root": str(emg_root.resolve()),
        "output_root": str(output_root.resolve()),
        "subjects_requested": subjects,
        "movements": list(movements),
        "trials": trials,
        "expected_emg_channels": list(EXPECTED_EMG_CHANNELS),
        "summary": {
            "original_subject_directories": sum((original_root / f"S{subject:03d}").is_dir() for subject in subjects),
            "converted_emg_files": sum(
                (emg_root / _subject_block(subject) / f"S{subject:03d}" / "EMG" / "ConvertedData.mat").is_file()
                for subject in subjects
            ),
            "expected_steps": len(subjects)
            * len(trials)
            * sum(len(MOVEMENT_METADATA[movement].steps) for movement in movements),
            "valid_trc_steps": sum(row.trc_valid for row in step_records),
            "mot_steps": sum(row.mot_exists for row in step_records),
            "valid_emg_steps": sum(row.emg_valid for row in step_records),
            "paired_marker_emg_steps": sum(row.usable_pair for row in step_records),
            "paired_clips": len(clips),
        },
        "by_movement": summary_by_movement,
    }
    _write_csv(output_root / "steps.csv", [asdict(row) for row in step_records])
    _write_manifest(_progress_manifest(output_root), clips, output_root=output_root)
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return step_records, clips, audit


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _progress_manifest(output_root: Path) -> Path:
    return output_root / ".conversion-progress" / "manifest.csv"


def _write_manifest(path: Path, clips: list[ClipRecord], *, output_root: Path) -> None:
    """Write every clip with relocatable paths and its individual fit status."""

    root = output_root.resolve()
    rows = []
    for clip in clips:
        row = asdict(clip)
        for field_name in ("marker_archive", "emg_archive", "output_path", "biomechanics_archive"):
            if not row[field_name]:
                continue
            value = Path(row[field_name])
            resolved = value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()
            try:
                row[field_name] = resolved.relative_to(root).as_posix()
            except ValueError:
                row[field_name] = str(resolved)
        rows.append(row)
    _write_csv(path, rows)


def _combine_paired_steps(trc_paths: list[Path]) -> tuple[np.ndarray, tuple[str, ...], float, np.ndarray, np.ndarray]:
    loaded = [load_gait120_trc(path) for path in trc_paths]
    labels = loaded[0].labels
    fps = loaded[0].fps
    positions: list[np.ndarray] = []
    source_frames: list[np.ndarray] = []
    source_steps: list[np.ndarray] = []
    last_frame: int | None = None
    for step_index, trc in enumerate(loaded, start=1):
        if trc.labels != labels:
            raise ValueError(f"Marker labels differ between {trc_paths[0]} and {trc_paths[step_index - 1]}")
        if not np.isclose(trc.fps, fps):
            raise ValueError(f"Marker rates differ between steps: {fps} and {trc.fps}")
        keep = np.ones(len(trc.frame_numbers), dtype=bool)
        if last_frame is not None and np.any(trc.frame_numbers > last_frame):
            keep &= trc.frame_numbers > last_frame
        positions.append(trc.positions[keep])
        source_frames.append(trc.frame_numbers[keep])
        source_steps.append(np.full(int(keep.sum()), step_index, dtype=np.int8))
        if keep.any():
            last_frame = int(trc.frame_numbers[keep][-1])
    return (
        np.concatenate(positions),
        labels,
        fps,
        np.concatenate(source_frames),
        np.concatenate(source_steps),
    )


def prepare_marker_archives(clips: list[ClipRecord], *, redo: bool = False) -> None:
    for index, clip in enumerate(clips, start=1):
        archive_path = Path(clip.marker_archive)
        if archive_path.exists() and not redo:
            try:
                with np.load(archive_path, allow_pickle=False) as data:
                    clip.source_frames = int(data["positions"].shape[0])
                    clip.fps = float(np.asarray(data["fps"]).reshape(()))
                continue
            except Exception:
                pass
        trc_paths = [Path(value) for value in clip.trc_paths.split(";") if value]
        positions, labels, fps, source_frames, source_steps = _combine_paired_steps(trc_paths)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = archive_path.with_name(f".{archive_path.name}.tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                positions=positions,
                labels=np.asarray(labels),
                fps=np.array(fps, dtype=np.float32),
                source_frames=source_frames,
                source_steps=source_steps,
                source_files=np.asarray([str(path.resolve()) for path in trc_paths]),
                coordinate_transform=np.array("Gait120 TRC [X,Y,Z] -> Z-up [X,-Z,Y], units -> metres"),
                archive_version=np.array(MARKER_ARCHIVE_VERSION, dtype=np.int64),
            )
        tmp.replace(archive_path)
        clip.source_frames = len(positions)
        clip.fps = fps
        if index % 100 == 0 or index == len(clips):
            print(f"Prepared marker archives: {index}/{len(clips)}", flush=True)


def prepare_emg_archives(clips: list[ClipRecord], *, redo: bool = False) -> None:
    """Export paired processed EMG without repeatedly loading the large MAT files."""

    grouped: dict[int, list[ClipRecord]] = defaultdict(list)
    for clip in clips:
        grouped[clip.subject].append(clip)

    completed = 0
    for _subject, subject_clips in sorted(grouped.items()):
        emg_path = Path(subject_clips[0].emg_path)
        movements = sorted({clip.movement for clip in subject_clips})
        loaded = loadmat(emg_path, variable_names=movements, struct_as_record=True, squeeze_me=False)
        movement_records = {movement: _mat_record(loaded[movement]) for movement in movements}
        for clip in subject_clips:
            archive_path = Path(clip.emg_archive)
            if archive_path.exists() and not redo:
                try:
                    _validate_emg_file(archive_path, expected_steps=clip.paired_steps)
                    completed += 1
                    continue
                except Exception:
                    pass
            steps = np.asarray([int(value) for value in clip.paired_steps.split(";") if value], dtype=np.int8)
            channel_names: tuple[str, ...] | None = None
            step_values = []
            for step in steps:
                channels, values = _emg_step_values(movement_records[clip.movement], clip.trial, int(step))
                if channel_names is not None and channels != channel_names:
                    raise ValueError(f"EMG channel order differs within {clip.motion}")
                channel_names = channels
                step_values.append(values)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = archive_path.with_name(f".{archive_path.name}.tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(
                    fh,
                    emg=np.stack(step_values),
                    channels=np.asarray(channel_names),
                    steps=steps,
                    samples_per_step=np.array(101, dtype=np.int64),
                    source_mat=np.array(str(emg_path.resolve())),
                    movement=np.array(clip.movement),
                    trial=np.array(clip.trial, dtype=np.int64),
                    archive_version=np.array(1, dtype=np.int64),
                )
            tmp.replace(archive_path)
            completed += 1
        if completed % 100 == 0 or completed == len(clips):
            print(f"Prepared EMG archives: {completed}/{len(clips)}", flush=True)


def _load_gait120_mot(path: Path) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    try:
        end_header = next(index for index, line in enumerate(lines) if line.strip().casefold() == "endheader")
    except StopIteration as exc:
        raise ValueError("MOT has no endheader") from exc
    names = tuple(lines[end_header + 1].split())
    values = np.loadtxt(lines[end_header + 2 :], dtype=np.float64, ndmin=2)
    if not names or values.shape[1] != len(names) or names[0] != "time":
        raise ValueError(f"Invalid MOT table: {len(names)} names, shape {values.shape}")
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("MOT timestamps/values are non-finite or unordered")
    return values[:, 0], names, values


def _z_up(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values)[..., (0, 2, 1)].copy()
    transformed[..., 1] *= -1.0
    return transformed


def _gait120_marker_timeline(
    trc_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], float]:
    loaded = [load_gait120_trc(path) for path in trc_paths]
    positions = []
    times = []
    last_frame = None
    for trc in loaded:
        if trc.labels != loaded[0].labels or not np.isclose(trc.fps, loaded[0].fps):
            raise ValueError("Gait120 step marker layouts or rates disagree")
        keep = np.ones(len(trc.frame_numbers), dtype=bool)
        if last_frame is not None:
            keep &= trc.frame_numbers > last_frame
        positions.append(trc.positions[keep])
        times.append(trc.times[keep])
        if keep.any():
            last_frame = int(trc.frame_numbers[keep][-1])
    marker_times = np.concatenate(times)
    if np.any(np.diff(marker_times) <= 0):
        raise ValueError("Combined Gait120 marker timestamps are not strictly increasing")
    return np.concatenate(positions), marker_times, loaded[0].labels, loaded[0].fps


def _gait120_emg(
    clip: ClipRecord,
    movement_record: np.void,
    marker_times: np.ndarray,
) -> dict[str, Any]:
    trc_paths = [Path(value) for value in clip.trc_paths.split(";") if value]
    steps = [int(value) for value in clip.paired_steps.split(";") if value]
    if len(trc_paths) != len(steps):
        raise ValueError(f"TRC/step counts disagree for {clip.motion}")
    chunks = []
    timestamps = []
    channel_names = None
    native_fps = None
    last_time = None
    for step, trc_path in zip(steps, trc_paths, strict=True):
        trc = load_gait120_trc(trc_path)
        channels, values = _emg_native_step_values(movement_record, clip.trial, step)
        ratio = len(values) / len(trc.positions)
        samples_per_marker = round(ratio)
        if not np.isclose(ratio, samples_per_marker) or samples_per_marker < 1:
            raise ValueError(f"EMG/marker ratio is {ratio:g} for {trc_path}, expected a positive integer")
        step_fps = trc.fps * samples_per_marker
        if not np.isclose(step_fps, GAIT120_EMG_FPS):
            raise ValueError(f"Native EMG rate is {step_fps:g} Hz, expected {GAIT120_EMG_FPS:g} Hz")
        if native_fps is not None and not np.isclose(step_fps, native_fps):
            raise ValueError("Native EMG rates disagree between paired steps")
        native_fps = step_fps
        step_times = trc.times[0] + np.arange(len(values), dtype=np.float64) / step_fps
        keep = np.ones(len(step_times), dtype=bool) if last_time is None else step_times > last_time + 1e-10
        chunks.append(values[keep])
        timestamps.append(step_times[keep])
        if keep.any():
            last_time = float(step_times[keep][-1])
        if channel_names is not None and channels != channel_names:
            raise ValueError("Native EMG channel order differs between steps")
        channel_names = channels
    assert native_fps is not None and channel_names is not None
    emg_native = np.concatenate(chunks)
    emg_time = np.concatenate(timestamps) - marker_times[0]
    clock, _ = motion_clock(clip.output_path)
    return {
        "emg_available": np.array(True),
        "emg": resample_linear(emg_time, emg_native, clock),
        "emg_native": emg_native,
        "emg_native_time_s": emg_time,
        "emg_channels": np.asarray(channel_names),
        "emg_muscles": np.asarray([name.replace("Gastrocnemuis", "Gastrocnemius") for name in channel_names]),
        "emg_channel_types": np.asarray(["muscle"] * len(channel_names)),
        "emg_channel_sides": np.asarray(["right"] * len(channel_names)),
        "emg_units": np.array("dimensionless MVC-normalized envelope"),
        "emg_processing": np.array("release EMGs_norm at native rate; linear sampling on motion clock"),
        "emg_source_paths": np.asarray([str(Path(clip.emg_path).resolve())]),
    }


def _gait120_plate_channels(names: tuple[str, ...]) -> list[int]:
    plates = sorted(
        {int(match.group(1)) for name in names if (match := re.fullmatch(r"ground_force(\d+)_vx", name)) is not None}
    )
    if not plates:
        raise ValueError("MOT has no ground_force*_vx columns")
    return plates


def _gait120_grf(
    clip: ClipRecord,
    marker_positions: np.ndarray,
    marker_times: np.ndarray,
    marker_labels: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mot_paths = [Path(value) for value in clip.mot_paths.split(";") if value and Path(value).is_file()]
    clock, _ = motion_clock(clip.output_path)
    if not mot_paths:
        return empty_grf(len(clock)), {
            "grf_recorded": np.zeros(len(clock), dtype=bool),
            "grf_recorded_native": np.empty(0, dtype=bool),
        }
    loaded = [_load_gait120_mot(path) for path in mot_paths]
    plates = _gait120_plate_channels(loaded[0][1])
    if any(_gait120_plate_channels(names) != plates for _time, names, _values in loaded):
        raise ValueError("Ground-force plate columns disagree between steps")
    native_fps_values = [1.0 / float(np.median(np.diff(times))) for times, _names, _values in loaded]
    native_fps = float(np.median(native_fps_values))
    if not all(np.isclose(value, native_fps, rtol=1e-5) for value in native_fps_values):
        raise ValueError(f"Ground-force sample rates disagree: {native_fps_values}")
    samples = round((marker_times[-1] - marker_times[0]) * native_fps) + 1
    native_time = np.arange(samples, dtype=np.float64) / native_fps
    force = np.zeros((samples, len(plates), 3), dtype=np.float32)
    moment = np.zeros_like(force)
    cop = np.zeros_like(force)
    recorded = np.zeros(samples, dtype=bool)
    for absolute_time, names, values in loaded:
        indices = np.rint((absolute_time - marker_times[0]) * native_fps).astype(np.int64)
        in_range = (indices >= 0) & (indices < samples)
        indices = indices[in_range]
        values = values[in_range]
        column = {name: index for index, name in enumerate(names)}
        for plate_index, plate in enumerate(plates):
            force[indices, plate_index] = _z_up(values[:, [column[f"ground_force{plate}_v{axis}"] for axis in "xyz"]])
            cop[indices, plate_index] = _z_up(values[:, [column[f"ground_force{plate}_p{axis}"] for axis in "xyz"]])
            moment[indices, plate_index] = _z_up(values[:, [column[f"ground_torque{plate}_{axis}"] for axis in "xyz"]])
        recorded[indices] = True
    valid = recorded[:, None] & (np.linalg.norm(force, axis=-1) > 20.0)

    feet = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        indices = [marker_labels.index(f"{prefix}HEE"), marker_labels.index(f"{prefix}TOE")]
        feet[side] = np.nanmean(marker_positions[:, indices], axis=1)
    absolute_native_time = marker_times[0] + native_time
    feet_native = {
        side: np.stack(
            [np.interp(absolute_native_time, marker_times, centres[:, axis]) for axis in range(3)],
            axis=-1,
        )
        for side, centres in feet.items()
    }
    labels = []
    assignments = []
    for plate_index, plate in enumerate(plates):
        active = valid[:, plate_index]
        if not active.any():
            label = f"plate{plate}"
            distances = {"left": float("nan"), "right": float("nan")}
        else:
            distances = {
                side: float(np.nanmedian(np.linalg.norm(cop[active, plate_index] - centre[active], axis=-1)))
                for side, centre in feet_native.items()
            }
            ordered = sorted(distances, key=distances.get)
            label = "combined" if distances[ordered[1]] - distances[ordered[0]] < 0.10 else ordered[0]
        if label in labels:
            label = f"{label}_plate{plate}"
        labels.append(label)
        assignments.append(
            f"plate{plate}:{label}:left_distance_m={distances['left']:.6g}:right_distance_m={distances['right']:.6g}"
        )
    aligned_force = resample_linear(native_time, force, clock)
    aligned_moment = resample_linear(native_time, moment, clock)
    aligned_cop = resample_linear(native_time, cop, clock)
    aligned_recorded = resample_linear(native_time, recorded.astype(np.float32), clock) > 0.5
    aligned_valid = aligned_recorded[:, None] & (np.linalg.norm(aligned_force, axis=-1) > 20.0)
    return (
        {
            "grf_available": np.array(True),
            "grf_cop_available": np.array(True),
            "grf_force": aligned_force,
            "grf_moment": aligned_moment,
            "grf_cop": aligned_cop,
            "grf_valid": aligned_valid,
            "grf_force_native": force,
            "grf_moment_native": moment,
            "grf_cop_native": cop,
            "grf_valid_native": valid,
            "grf_native_time_s": native_time,
            "grf_channels": np.asarray(labels),
            "grf_source_paths": np.asarray([str(path.resolve()) for path in mot_paths]),
        },
        {
            "grf_platform_assignment": np.asarray(assignments),
            "grf_recorded": aligned_recorded,
            "grf_recorded_native": recorded,
            "grf_contact_threshold_n": np.array(20.0, dtype=np.float32),
        },
    )


def prepare_biomechanics_archives(clips: list[ClipRecord], *, redo: bool = False) -> None:
    """Create native and frame-synchronized EMG/GRF sidecars for fitted clips."""

    grouped: dict[int, list[ClipRecord]] = defaultdict(list)
    for clip in clips:
        grouped[clip.subject].append(clip)
    completed = 0
    for _subject, subject_clips in sorted(grouped.items()):
        eligible = [clip for clip in subject_clips if Path(clip.output_path).is_file()]
        if not eligible:
            continue
        emg_path = Path(eligible[0].emg_path)
        movements = sorted({clip.movement for clip in eligible})
        loaded = loadmat(emg_path, variable_names=movements, struct_as_record=True, squeeze_me=False)
        movement_records = {movement: _mat_record(loaded[movement]) for movement in movements}
        for clip in eligible:
            output = Path(clip.biomechanics_archive or biomechanics_path(clip.output_path))
            clip.biomechanics_archive = str(output)
            if output.exists() and not redo:
                try:
                    validate_biomechanics(output, motion_path=clip.output_path)
                    completed += 1
                    continue
                except Exception:
                    pass
            trc_paths = [Path(value) for value in clip.trc_paths.split(";") if value]
            marker_positions, marker_times, marker_labels, _marker_fps = _gait120_marker_timeline(trc_paths)
            clock, fps = motion_clock(clip.output_path)
            emg = _gait120_emg(clip, movement_records[clip.movement], marker_times)
            grf, grf_metadata = _gait120_grf(
                clip,
                marker_positions,
                marker_times,
                marker_labels,
            )
            write_biomechanics(
                output,
                dataset="gait120",
                motion=clip.motion,
                motion_time_s=clock,
                motion_fps=fps,
                synchronization=(
                    "TRC, normalized EMG, and force MOT share absolute capture timestamps; "
                    "overlapping paired-step samples are de-duplicated"
                ),
                emg=emg,
                grf=grf,
                metadata={
                    **grf_metadata,
                    "source_start_time_s": np.array(marker_times[0], dtype=np.float64),
                    "motion_source_frame": (
                        int(load_gait120_trc(trc_paths[0]).frame_numbers[0])
                        + np.rint(clock * _marker_fps).astype(np.int64)
                    ),
                    "source_trc_paths": np.asarray([str(path.resolve()) for path in trc_paths]),
                },
            )
            completed += 1
        print(f"Prepared biomechanical sidecars: {completed}/{sum(map(len, grouped.values()))}", flush=True)


def _validate_smplh_file(path: Path, *, enforce_knee_hinge: bool = False) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        poses = np.asarray(data["poses"])
        trans = np.asarray(data["trans"])
        betas = np.asarray(data["betas"])
        fps = float(np.asarray(data["mocap_framerate"]).reshape(()))
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"poses has shape {poses.shape}, expected (T, 156)")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"trans has shape {trans.shape}, expected {(poses.shape[0], 3)}")
    if betas.ndim != 1 or betas.size < 10:
        raise ValueError(f"betas has shape {betas.shape}")
    if poses.shape[0] < 2 or not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("trajectory is empty or contains non-finite pose/translation values")
    if enforce_knee_hinge:
        nonhinge = np.asarray(poses[:, (13, 14, 16, 17)], dtype=float)
        maximum = float(np.max(np.abs(nonhinge), initial=0.0))
        if maximum > 1e-4:
            raise ValueError(f"knee hinge gate failed: non-flexion component reaches {np.degrees(maximum):.4f} deg")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid fps={fps}")
    return {"frames": int(poses.shape[0]), "fps": fps}


def _validate_emg_file(path: Path, *, expected_steps: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"emg", "channels", "steps", "samples_per_step", "source_mat"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        emg = np.asarray(data["emg"])
        channels = tuple(str(value) for value in np.asarray(data["channels"]).reshape(-1))
        steps = np.asarray(data["steps"], dtype=np.int64).reshape(-1)
    expected = np.asarray([int(value) for value in expected_steps.split(";") if value], dtype=np.int64)
    if channels != EXPECTED_EMG_CHANNELS:
        raise ValueError(f"unexpected channel order: {channels}")
    if not np.array_equal(steps, expected):
        raise ValueError(f"steps {steps.tolist()} do not match {expected.tolist()}")
    if emg.shape != (len(steps), 101, len(EXPECTED_EMG_CHANNELS)):
        raise ValueError(f"emg has shape {emg.shape}")
    if not np.isfinite(emg).all():
        raise ValueError("emg contains non-finite values")
    return {"steps": len(steps), "samples": int(emg.shape[0] * emg.shape[1])}


def _fit_subject(job: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ["JAX_PLATFORMS"] = "cuda" if job["device"] == "cuda" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(job["torch_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(job["torch_threads"]))

    import torch

    torch.set_num_threads(job["torch_threads"])
    from terra.datasets.marker_fitting import require_paper_pose_prior

    require_paper_pose_prior()
    from musclemimic.web_viewer.c3d_to_smpl import fit_smpl_to_c3d, save_motion_data_as_amass_smplh_npz

    subject = int(job["subject"])
    state_path = Path(job["stage1_state_root"]) / f"S{subject:03d}.npz"
    results: list[dict[str, Any]] = []
    for raw_clip in job["clips"]:
        clip = ClipRecord(**raw_clip)
        output_path = Path(clip.output_path)
        fit_metadata_path = output_path.with_suffix(".fit.json")
        metadata = json.loads(fit_metadata_path.read_text()) if fit_metadata_path.exists() else {}
        compatible_hinge = bool(metadata.get("enforce_knee_hinge", False)) == bool(job["enforce_knee_hinge"])
        compatible_torso = np.isclose(
            float(metadata.get("stage2_torso_frame_weight", 0.0)),
            float(job["stage2_torso_frame_weight"]),
        )
        if output_path.exists() and not job["redo"] and compatible_hinge and compatible_torso:
            try:
                validation = _validate_smplh_file(
                    output_path,
                    enforce_knee_hinge=job["enforce_knee_hinge"],
                )
                clip.output_frames = validation["frames"]
                clip.fps = validation["fps"]
                clip.status = "existing"
                if fit_metadata_path.exists():
                    fit_metadata = json.loads(fit_metadata_path.read_text())
                    clip.marker_error_mean_mm = fit_metadata.get("marker_error_mean_mm")
                    clip.marker_error_p95_mm = fit_metadata.get("marker_error_p95_mm")
                    clip.marker_error_max_mm = fit_metadata.get("marker_error_max_mm")
                    if clip.marker_error_max_mm is None:
                        clip.marker_error_max_mm = fit_metadata.get("marker_error", {}).get("max_mm")
                results.append(asdict(clip))
                print(f"[S{subject:03d}] existing {clip.motion}", flush=True)
                continue
            except Exception:
                pass

        started = time.time()
        try:
            motion_data = fit_smpl_to_c3d(
                clip.marker_archive,
                job["smpl_model_path"],
                surface_model_type="smplh",
                gender=job["gender"],
                target_fps=job["target_fps"],
                stage1_iters=job["stage1_iters"],
                stage2_iters=job["stage2_iters"],
                n_ref_frames=job["n_ref_frames"],
                stage1_shape_solver=job["stage1_shape_solver"],
                device=job["device"],
                stage2_solver=job["stage2_solver"],
                strict_frame_picking=False,
                least_avail_markers=job["least_avail_markers"],
                optimize_toes=True,
                enforce_knee_hinge=job["enforce_knee_hinge"],
                stage1_state_path=str(state_path),
                stage2_torso_frame_weight=job["stage2_torso_frame_weight"],
            )
            save_motion_data_as_amass_smplh_npz(motion_data, output_path)
            validation = _validate_smplh_file(
                output_path,
                enforce_knee_hinge=job["enforce_knee_hinge"],
            )
            marker_error = motion_data.get("debug", {}).get("marker_error", {})
            clip.output_frames = validation["frames"]
            clip.fps = validation["fps"]
            clip.marker_error_mean_mm = marker_error.get("mean_mm")
            clip.marker_error_p95_mm = marker_error.get("p95_mm")
            clip.marker_error_max_mm = marker_error.get("max_mm")
            clip.status = "generated"
            fit_metadata = {
                "motion": clip.motion,
                "source_marker_archive": str(Path(clip.marker_archive).resolve()),
                "stage1_state": str(state_path.resolve()),
                "stage1_reused": bool(motion_data.get("debug", {}).get("stagei", {}).get("reused")),
                "surface_model_type": "smplh",
                "gender": job["gender"],
                "smpl_model_path": str(Path(job["smpl_model_path"]).resolve()),
                "target_fps": job["target_fps"],
                "stage1_iters": job["stage1_iters"],
                "stage2_iters": job["stage2_iters"],
                "n_ref_frames": job["n_ref_frames"],
                "device": job["device"],
                "stage2_solver": job["stage2_solver"],
                "enforce_knee_hinge": job["enforce_knee_hinge"],
                "stage2_torso_frame_weight": job["stage2_torso_frame_weight"],
                "marker_error_mean_mm": clip.marker_error_mean_mm,
                "marker_error_p95_mm": clip.marker_error_p95_mm,
                "marker_error_max_mm": clip.marker_error_max_mm,
                "marker_error": marker_error,
                "elapsed_seconds": time.time() - started,
            }
            fit_metadata_path.write_text(json.dumps(fit_metadata, indent=2) + "\n")
            print(
                f"[S{subject:03d}] generated {clip.motion} {clip.output_frames}f "
                f"mean={clip.marker_error_mean_mm:.1f}mm {time.time() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:
            clip.status = "failed"
            clip.error = f"{type(exc).__name__}: {exc}"
            error_path = output_path.with_suffix(".error.log")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc())
            print(f"[S{subject:03d}] FAILED {clip.motion}: {clip.error}", flush=True)
        results.append(asdict(clip))
    return results


def fit_clips(
    clips: list[ClipRecord],
    *,
    output_root: Path,
    workers: int,
    redo: bool,
    smpl_model_path: str,
    gender: str,
    target_fps: float | None,
    stage1_iters: int,
    stage2_iters: int,
    n_ref_frames: int,
    stage1_shape_solver: str,
    least_avail_markers: float,
    device: str,
    stage2_solver: str,
    enforce_knee_hinge: bool,
    stage2_torso_frame_weight: float = 0.0,
    stage1_state_root: Path | None = None,
) -> list[ClipRecord]:
    grouped: dict[int, list[ClipRecord]] = defaultdict(list)
    for clip in clips:
        grouped[clip.subject].append(clip)
    torch_threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    jobs = []
    for subject, subject_clips in sorted(grouped.items()):
        jobs.append(
            {
                "subject": subject,
                "clips": [asdict(clip) for clip in subject_clips],
                "output_root": str(output_root),
                "stage1_state_root": str(stage1_state_root or (output_root / ".stage1")),
                "redo": redo,
                "smpl_model_path": smpl_model_path,
                "gender": gender,
                "target_fps": target_fps,
                "stage1_iters": stage1_iters,
                "stage2_iters": stage2_iters,
                "n_ref_frames": n_ref_frames,
                "stage1_shape_solver": stage1_shape_solver,
                "least_avail_markers": least_avail_markers,
                "device": device,
                "stage2_solver": stage2_solver,
                "enforce_knee_hinge": enforce_knee_hinge,
                "stage2_torso_frame_weight": stage2_torso_frame_weight,
                "torch_threads": torch_threads,
            }
        )

    all_results: list[ClipRecord] = []
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            all_results.extend(ClipRecord(**row) for row in _fit_subject(job))
            _write_manifest(
                _progress_manifest(output_root),
                sorted(all_results, key=_clip_sort_key),
                output_root=output_root,
            )
            print(f"Completed subjects: {index}/{len(jobs)}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(_fit_subject, job): job["subject"] for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                subject = futures[future]
                try:
                    all_results.extend(ClipRecord(**row) for row in future.result())
                except Exception as exc:
                    print(f"[S{subject:03d}] worker failed: {type(exc).__name__}: {exc}", flush=True)
                    for raw_clip in next(job["clips"] for job in jobs if job["subject"] == subject):
                        clip = ClipRecord(**raw_clip)
                        clip.status = "failed"
                        clip.error = f"worker {type(exc).__name__}: {exc}"
                        all_results.append(clip)
                _write_manifest(
                    _progress_manifest(output_root),
                    sorted(all_results, key=_clip_sort_key),
                    output_root=output_root,
                )
                print(f"Completed subjects: {index}/{len(jobs)}", flush=True)
    return sorted(all_results, key=_clip_sort_key)


def _clip_sort_key(clip: ClipRecord) -> tuple[int, int, int]:
    return clip.subject, TARGET_MOVEMENTS.index(clip.movement), clip.trial


def _stable_clip_key(seed: str, clip: ClipRecord) -> bytes:
    return hashlib.sha256(f"{seed}:{clip.motion}".encode()).digest()


def collect_chair_conversion_clips(
    clips: list[ClipRecord],
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Return every chair target that has a Trial01 walking calibration.

    Conversion deliberately happens before cohort selection: poor target fits
    and every target belonging to a poor subject calibration can therefore be
    excluded without reducing or biasing the published cohort.

    Returns ``(targets, targets_and_calibrations)``.
    """

    calibrations = {clip.subject: clip for clip in clips if clip.movement == "LevelWalking" and clip.trial == 1}
    targets = [clip for clip in clips if clip.movement in CHAIR_MOVEMENTS and clip.subject in calibrations]
    for clip in targets:
        clip.role = "retarget"

    selected_subjects = {clip.subject for clip in targets}
    selected_calibrations = []
    for subject in sorted(selected_subjects):
        calibration = calibrations[subject]
        calibration.role = "calibration"
        selected_calibrations.append(calibration)
    return targets, sorted(targets + selected_calibrations, key=_clip_sort_key)


def select_balanced_chair_clips(
    clips: list[ClipRecord],
    *,
    per_movement: int,
    seed: str = CHAIR_SELECTION_SEED,
    fit_passed_only: bool = False,
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Select paired stool transitions with exact direction/trial balance.

    Only subjects with a usable Trial01 level-walking calibration are eligible.
    Subject counts are balanced across both directions before a stable hash
    breaks ties, maximizing participant coverage without relying on row order.

    Returns ``(targets, targets_and_calibrations)``.
    """

    if per_movement <= 0 or per_movement % 5:
        raise ValueError("chair motions per movement must be positive and divisible by five")
    if not seed:
        raise ValueError("chair selection seed must be non-empty")

    calibrations = {
        clip.subject: clip
        for clip in clips
        if clip.movement == "LevelWalking" and clip.trial == 1 and (clip.fit_passed or not fit_passed_only)
    }
    subject_counts: Counter[int] = Counter()
    selected: list[ClipRecord] = []
    per_trial = per_movement // 5
    for movement in CHAIR_MOVEMENTS:
        for trial in range(1, 6):
            candidates = [
                clip
                for clip in clips
                if clip.movement == movement
                and clip.trial == trial
                and clip.subject in calibrations
                and (clip.fit_passed or not fit_passed_only)
            ]
            if len(candidates) < per_trial:
                qualifier = " successful" if fit_passed_only else ""
                raise ValueError(
                    f"Gait120 {movement} Trial{trial:02d} has only {len(candidates)}{qualifier} paired clips "
                    f"with calibration; cannot select {per_trial}"
                )
            for _ in range(per_trial):
                chosen = min(
                    candidates,
                    key=lambda clip: (
                        subject_counts[clip.subject],
                        _stable_clip_key(f"{seed}:{movement}", clip),
                    ),
                )
                candidates.remove(chosen)
                chosen.role = "retarget"
                selected.append(chosen)
                subject_counts[chosen.subject] += 1

    selected_subjects = {clip.subject for clip in selected}
    selected_calibrations = []
    for subject in sorted(selected_subjects):
        calibration = calibrations[subject]
        calibration.role = "calibration"
        selected_calibrations.append(calibration)
    return selected, sorted(selected + selected_calibrations, key=_clip_sort_key)


def write_chair_selection(path: Path, targets: list[ClipRecord]) -> None:
    """Publish the exact target-only selection consumed by every later stage."""

    rows = []
    for clip in targets:
        rows.append(
            {
                "motion": clip.motion,
                "dataset": "gait120",
                "subject": f"S{clip.subject:03d}",
                "movement": clip.movement,
                "trial": f"Trial{clip.trial:02d}",
                "terrain_class": clip.terrain_class,
                "expected_family": clip.expected_family,
                "calibration_motion": (f"Gait120/S{clip.subject:03d}/LevelWalking/Trial01/AllSteps_stageii"),
            }
        )
    _write_csv(path, rows)


def _filter_clips_by_selection_manifest(clips: list[ClipRecord], manifest: Path) -> list[ClipRecord]:
    """Keep explicitly selected Gait120 motions and their subject calibrations."""
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    retarget_motions: set[str] = set()
    calibration_motions: set[str] = set()
    for row in rows:
        if row.get("dataset", "").casefold() != "gait120":
            continue
        motion = row.get("motion", "").strip()
        calibration = row.get("calibration_motion", "").strip()
        if motion:
            retarget_motions.add(motion)
        if calibration:
            calibration_motions.add(calibration)
    requested = retarget_motions | calibration_motions
    if not requested:
        raise ValueError(f"selection manifest contains no Gait120 motions: {manifest}")
    available = {clip.motion: clip for clip in clips}
    missing = sorted(requested - set(available))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"selection manifest requests {len(missing)} unavailable Gait120 motion(s): {preview}{suffix}")
    selected = []
    for motion in requested:
        clip = available[motion]
        clip.role = "retarget" if motion in retarget_motions else "calibration"
        selected.append(clip)
    return sorted(selected, key=_clip_sort_key)


def _summarize_marker_fit_quality(clips: list[ClipRecord]) -> dict:
    """Identify clip-level fit outliers with a robust corpus-relative guard."""

    rows = [
        (clip.motion, float(clip.marker_error_mean_mm))
        for clip in clips
        if clip.marker_error_mean_mm is not None and np.isfinite(clip.marker_error_mean_mm)
    ]
    motions_with_metrics = {motion for motion, _error in rows}
    missing = sorted(clip.motion for clip in clips if clip.motion not in motions_with_metrics)
    if not rows:
        return {
            "clips_with_metrics": 0,
            "median_mean_error_mm": None,
            "mad_mm": None,
            "p95_mean_error_mm": None,
            "max_mean_error_mm": None,
            "robust_outlier_threshold_mm": None,
            "absolute_outlier_ceiling_mm": 20.0,
            "missing_metrics": missing,
            "outliers": [],
            "ready": False,
        }

    values = np.asarray([error for _motion, error in rows], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_threshold = median + max(5.0 * 1.4826 * mad, 1.0)
    threshold = min(20.0, robust_threshold)
    outliers = [
        {"motion": motion, "marker_error_mean_mm": error}
        for motion, error in sorted(rows, key=lambda item: item[1], reverse=True)
        if error > threshold
    ]
    return {
        "clips_with_metrics": len(rows),
        "median_mean_error_mm": median,
        "mad_mm": mad,
        "p95_mean_error_mm": float(np.percentile(values, 95)),
        "max_mean_error_mm": float(np.max(values)),
        "robust_outlier_threshold_mm": threshold,
        "absolute_outlier_ceiling_mm": 20.0,
        "missing_metrics": missing,
        "outliers": outliers,
        "ready": not missing and not outliers,
    }


def _apply_fit_quality(
    clips: list[ClipRecord],
    *,
    validation_failures: list[dict],
    emg_validation_failures: list[dict] | None = None,
    biomechanics_validation_failures: list[dict] | None = None,
    marker_fit_quality: dict,
) -> list[dict]:
    """Set the manifest quality gate and return auditable rejection rows."""

    reasons: dict[str, list[str]] = defaultdict(list)
    for label, rows in (("smplh_validation", validation_failures),):
        for row in rows:
            reasons[row["motion"]].append(f"{label}: {row['error']}")
    for row in emg_validation_failures or []:
        reasons[row["motion"]].append(f"emg_validation: {row['error']}")
    for row in biomechanics_validation_failures or []:
        reasons[row["motion"]].append(f"biomechanics_validation: {row['error']}")
    for motion in marker_fit_quality["missing_metrics"]:
        reasons[motion].append("marker_fit: missing mean marker error")
    threshold = marker_fit_quality["robust_outlier_threshold_mm"]
    for row in marker_fit_quality["outliers"]:
        reasons[row["motion"]].append(
            f"marker_fit: mean {row['marker_error_mean_mm']:.6g} mm exceeds {threshold:.6g} mm"
        )
    for clip in clips:
        if clip.status in {"failed", "missing", "error"}:
            detail = f": {clip.error}" if clip.error else ""
            reasons[clip.motion].append(f"conversion status {clip.status}{detail}")

    for clip in clips:
        clip.fit_passed = not reasons[clip.motion]
        clip.fit_failure_reason = "; ".join(reasons[clip.motion])

    # A target cannot be benchmark-ready when its subject-specific level-walking
    # calibration failed, even if the target clip itself happened to pass.
    failed_calibration_subjects = {clip.subject for clip in clips if clip.role == "calibration" and not clip.fit_passed}
    for clip in clips:
        if clip.role == "retarget" and clip.subject in failed_calibration_subjects:
            dependency_reason = "calibration_fit: subject level-walking calibration failed"
            if dependency_reason not in reasons[clip.motion]:
                reasons[clip.motion].append(dependency_reason)
            clip.fit_passed = False
            clip.fit_failure_reason = "; ".join(reasons[clip.motion])

    return [
        {
            "motion": clip.motion,
            "role": clip.role,
            "subject": clip.subject,
            "reason": clip.fit_failure_reason,
            "marker_error_mean_mm": clip.marker_error_mean_mm,
            "marker_error_p95_mm": clip.marker_error_p95_mm,
            "marker_error_max_mm": clip.marker_error_max_mm,
        }
        for clip in clips
        if not clip.fit_passed
    ]


def validate_dataset(
    clips: list[ClipRecord],
    *,
    output_root: Path,
    publish_manifest: bool = True,
    chair_per_movement: int | None = None,
    chair_selection_output: Path | None = None,
) -> dict:
    status_counts = Counter(clip.status for clip in clips)
    failures = []
    frames = 0
    fps_values = set()
    emg_failures = []
    valid_emg_files = 0
    biomechanics_failures = []
    valid_biomechanics_files = 0
    for clip in clips:
        path = Path(clip.output_path)
        try:
            validation = _validate_smplh_file(path)
            clip.output_frames = validation["frames"]
            clip.fps = validation["fps"]
            frames += validation["frames"]
            fps_values.add(validation["fps"])
        except Exception as exc:
            failures.append({"motion": clip.motion, "error": f"{type(exc).__name__}: {exc}"})
        try:
            _validate_emg_file(Path(clip.emg_archive), expected_steps=clip.paired_steps)
            valid_emg_files += 1
        except Exception as exc:
            emg_failures.append({"motion": clip.motion, "error": f"{type(exc).__name__}: {exc}"})
        try:
            sidecar = Path(clip.biomechanics_archive or biomechanics_path(clip.output_path))
            clip.biomechanics_archive = str(sidecar)
            validate_biomechanics(sidecar, motion_path=clip.output_path)
            valid_biomechanics_files += 1
        except Exception as exc:
            biomechanics_failures.append({"motion": clip.motion, "error": f"{type(exc).__name__}: {exc}"})

    expected_motions = {clip.motion for clip in clips}
    actual_motions = {
        str(path.relative_to(output_root).with_suffix(""))
        for path in (output_root / "Gait120").glob("S*/**/AllSteps_stageii.npz")
    }
    marker_fit_quality = _summarize_marker_fit_quality(clips)
    quality_rejections = _apply_fit_quality(
        clips,
        validation_failures=failures,
        emg_validation_failures=emg_failures,
        biomechanics_validation_failures=biomechanics_failures,
        marker_fit_quality=marker_fit_quality,
    )
    retarget_clips = [clip for clip in clips if clip.role == "retarget"]
    calibration_clips = [clip for clip in clips if clip.role == "calibration"]
    report = {
        "manifest_clips": len(clips),
        "valid_smplh_files": len(clips) - len(failures),
        "valid_paired_emg_files": valid_emg_files,
        "valid_biomechanics_files": valid_biomechanics_files,
        "total_output_frames": frames,
        "fps_values": sorted(fps_values),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_manifest_outputs": sorted(expected_motions - actual_motions),
        "unexpected_outputs": sorted(actual_motions - expected_motions),
        "validation_failures": failures,
        "emg_validation_failures": emg_failures,
        "biomechanics_validation_failures": biomechanics_failures,
        "marker_fit_quality": marker_fit_quality,
        "marker_fit_quality_ready": marker_fit_quality["ready"],
        "quality_passed_retarget": sum(clip.fit_passed for clip in retarget_clips),
        "quality_failed_retarget": sum(not clip.fit_passed for clip in retarget_clips),
        "quality_passed_calibration": sum(clip.fit_passed for clip in calibration_clips),
        "quality_failed_calibration": sum(not clip.fit_passed for clip in calibration_clips),
        "conversion_quality_rejections": quality_rejections,
        "terra_ready": not failures and expected_motions == actual_motions,
    }
    report["benchmark_ready"] = (
        report["terra_ready"]
        and not emg_failures
        and not biomechanics_failures
        and not marker_fit_quality["missing_metrics"]
        and any(clip.fit_passed for clip in retarget_clips)
    )
    if chair_per_movement is not None:
        if chair_selection_output is None:
            raise ValueError("chair_selection_output is required for chair cohort selection")
        try:
            chair_targets, _ = select_balanced_chair_clips(
                clips,
                per_movement=chair_per_movement,
                fit_passed_only=True,
            )
        except ValueError as exc:
            report["chair_selection"] = {
                "ready": False,
                "requested_per_movement": chair_per_movement,
                "selected_targets": 0,
                "output": str(chair_selection_output),
                "error": str(exc),
            }
            report["benchmark_ready"] = False
        else:
            write_chair_selection(chair_selection_output, chair_targets)
            report["chair_selection"] = {
                "ready": True,
                "requested_per_movement": chair_per_movement,
                "selected_targets": len(chair_targets),
                "selected_by_movement": dict(sorted(Counter(clip.movement for clip in chair_targets).items())),
                "selected_by_movement_trial": {
                    f"{movement}/Trial{trial:02d}": count
                    for (movement, trial), count in sorted(
                        Counter((clip.movement, clip.trial) for clip in chair_targets).items()
                    )
                },
                "output": str(chair_selection_output),
                "error": "",
            }
    report["paired_dataset_ready"] = (
        report["terra_ready"] and not emg_failures and not biomechanics_failures and marker_fit_quality["ready"]
    )
    report["manifest_published"] = bool(publish_manifest)
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest_path = output_root / "manifest.csv" if report["manifest_published"] else _progress_manifest(output_root)
    _write_manifest(manifest_path, clips, output_root=output_root)
    return report


def _build_parser(roots: StorageRoots | None = None) -> argparse.ArgumentParser:
    roots = roots or StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(
        prog="terra convert gait120",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=roots.data_root / "Gait120-original" / "extracted",
        help="Root containing S001 ... S120 from the original Figshare release",
    )
    parser.add_argument(
        "--emg-root",
        type=Path,
        default=roots.data_root / "Gait120-EMG",
        help="Root of the processed, SciPy-readable Gait120-EMG dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=roots.artifact_root / "gait120" / "smplh",
        help="Output directory and AMASS dataset root",
    )
    parser.add_argument(
        "--stage1-state-root",
        type=Path,
        help="Optional existing subject-calibration root for isolated Stage-II refits",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Fit only Gait120 rows in this explicit CSV, plus their calibration motions",
    )
    parser.add_argument(
        "--chair-per-movement",
        type=int,
        help=(
            "Convert every eligible SitToStand and StandToSit clip, then select this many "
            "successful fits per movement, balanced across trials and subjects"
        ),
    )
    parser.add_argument(
        "--chair-selection-output",
        type=Path,
        help=(
            "Artifact path for the generated target-only chair selection; defaults to OUTPUT_ROOT/chair_N_selection.csv"
        ),
    )
    parser.add_argument("--subjects", default="1-120", help="Comma-separated subjects/ranges")
    parser.add_argument("--trials", default="1-5", help="Comma-separated trials/ranges")
    parser.add_argument(
        "--movements",
        nargs="+",
        choices=TARGET_MOVEMENTS,
        default=list(TARGET_MOVEMENTS),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--redo", action="store_true", help="Rebuild marker archives and existing SMPL-H files")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Fit only the first N paired clips (smoke tests)")
    parser.add_argument("--smpl-model-path", type=Path, default=roots.model_root)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch Stage-II device; auto selects CUDA when visible",
    )
    parser.add_argument("--target-fps", type=float, default=50.0, help="Downsample 100 Hz TRCs before fitting")
    parser.add_argument("--stage1-iters", type=int, default=100)
    parser.add_argument("--stage2-iters", type=int, default=80)
    parser.add_argument(
        "--stage2-solver",
        choices=("batched_lbfgs", "frame_lbfgs"),
        default="batched_lbfgs",
        help="Batched is GPU-oriented; frame_lbfgs reproduces the original sequential solver",
    )
    parser.add_argument("--n-ref-frames", type=int, default=12)
    parser.add_argument(
        "--stage1-shape-solver",
        choices=("joint_dogleg", "joint_dogleg_jax"),
        default="joint_dogleg_jax",
    )
    parser.add_argument("--least-avail-markers", type=float, default=0.8)
    parser.add_argument(
        "--enforce-knee-hinge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain SMPL-H knees to their flexion axis during marker fitting",
    )
    parser.add_argument(
        "--stage2-torso-frame-weight",
        type=float,
        default=0.0,
        help="Soft weight on marker-derived pelvis/shoulder frame consistency",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    roots = StorageRoots.from_environment(Path.cwd())
    args = _build_parser(roots).parse_args(arguments)
    args.original_root = roots.resolve_input(args.original_root, base=Path.cwd())
    args.emg_root = roots.resolve_input(args.emg_root, base=Path.cwd())
    args.output_root = roots.resolve_artifact(args.output_root, base=Path.cwd())
    args.smpl_model_path = roots.resolve_model(args.smpl_model_path, base=Path.cwd())
    args.stage1_state_root = roots.resolve_artifact(args.stage1_state_root, base=Path.cwd())
    args.selection_manifest = roots.resolve_input(args.selection_manifest, base=Path.cwd())
    args.chair_selection_output = roots.resolve_artifact(args.chair_selection_output, base=Path.cwd())
    assert args.original_root is not None
    assert args.emg_root is not None
    assert args.output_root is not None
    assert args.smpl_model_path is not None
    subjects = _parse_int_ranges(args.subjects, lower=1, upper=120)
    trials = _parse_int_ranges(args.trials, lower=1, upper=5)
    movements = tuple(args.movements)
    if args.selection_manifest is not None and args.chair_per_movement is not None:
        raise SystemExit("--selection-manifest and --chair-per-movement are mutually exclusive")
    if args.chair_selection_output is not None and args.chair_per_movement is None:
        raise SystemExit("--chair-selection-output requires --chair-per-movement")
    if args.chair_per_movement is not None:
        required = {"LevelWalking", *CHAIR_MOVEMENTS}
        missing = sorted(required - set(movements))
        if missing:
            raise SystemExit("--chair-per-movement requires --movements to include " + ", ".join(missing))
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    for path, description in ((args.original_root, "original root"), (args.emg_root, "EMG root")):
        if not path.is_dir():
            raise SystemExit(f"Gait120 {description} does not exist: {path}")
    write_git_commit(args.output_root)

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")

    started = time.time()
    print("Auditing paired Gait120 markers, EMG, and available ground forces...", flush=True)
    _, clips, audit = audit_dataset(
        original_root=args.original_root,
        emg_root=args.emg_root,
        output_root=args.output_root,
        subjects=subjects,
        movements=movements,
        trials=trials,
    )
    chair_selection_output = None
    if args.selection_manifest is not None:
        clips = _filter_clips_by_selection_manifest(clips, args.selection_manifest)
        print(
            f"Selection filter: {len(clips)} Gait120 retarget/calibration clips",
            flush=True,
        )
    elif args.chair_per_movement is not None:
        targets, clips = collect_chair_conversion_clips(clips)
        chair_selection_output = args.chair_selection_output or (
            args.output_root / f"chair_{2 * args.chair_per_movement}_selection.csv"
        )
        print(
            f"Chair conversion pool: {len(targets)} targets and {len(clips) - len(targets)} "
            f"subject calibrations; selection follows fit validation -> {chair_selection_output}",
            flush=True,
        )
    print(json.dumps(audit["summary"], indent=2), flush=True)
    for movement, summary in audit["by_movement"].items():
        print(f"{movement}: {summary}", flush=True)
    if args.audit_only:
        print(f"Audit complete in {time.time() - started:.1f}s -> {args.output_root / 'audit.json'}")
        return 0

    print(f"Preparing {len(clips)} paired marker archives...", flush=True)
    prepare_marker_archives(clips, redo=args.redo)
    print(f"Preparing {len(clips)} paired EMG archives...", flush=True)
    prepare_emg_archives(clips, redo=args.redo)
    _write_manifest(_progress_manifest(args.output_root), clips, output_root=args.output_root)
    if args.prepare_only:
        print(f"Preparation complete in {time.time() - started:.1f}s -> {args.output_root}")
        return 0

    if args.limit is not None:
        clips = clips[: args.limit]
    print(
        f"Fitting {len(clips)} clips on {args.workers} worker(s): target_fps={args.target_fps:g}, "
        f"Stage-I={args.stage1_iters} once/subject, Stage-II={args.stage2_iters} "
        f"({args.stage2_solver}), device={device}",
        flush=True,
    )
    fitted = fit_clips(
        clips,
        output_root=args.output_root,
        workers=args.workers,
        redo=args.redo,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        target_fps=args.target_fps,
        stage1_iters=args.stage1_iters,
        stage2_iters=args.stage2_iters,
        n_ref_frames=args.n_ref_frames,
        stage1_shape_solver=args.stage1_shape_solver,
        least_avail_markers=args.least_avail_markers,
        device=device,
        stage2_solver=args.stage2_solver,
        enforce_knee_hinge=args.enforce_knee_hinge,
        stage2_torso_frame_weight=args.stage2_torso_frame_weight,
        stage1_state_root=args.stage1_state_root,
    )
    print("Preparing motion-synchronized EMG/GRF sidecars...", flush=True)
    prepare_biomechanics_archives(fitted, redo=args.redo)
    report = validate_dataset(
        fitted,
        output_root=args.output_root,
        publish_manifest=args.limit is None,
        chair_per_movement=args.chair_per_movement if args.limit is None else None,
        chair_selection_output=chair_selection_output,
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Finished in {(time.time() - started) / 60:.1f} min -> {args.output_root}", flush=True)
    return 0 if report["benchmark_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
