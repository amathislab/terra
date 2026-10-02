"""Inspect Gait120 and build TERRA-ready AMASS/SMPL-H trajectories.

The original dataset stores up to two TRC gait cycles per locomotion trial and
one transition per stool trial. This converter concatenates available TRC steps,
fits SMPL-H, and writes AMASS-like ``.npz`` motion files.

The Stage-I shape/marker calibration is reused for every motion from a subject.
"""

from __future__ import annotations

import argparse
import csv
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

from terra._revision import write_git_commit
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
REPORT_VERSION = 1
MARKER_ARCHIVE_VERSION = 1


@dataclass
class StepRecord:
    subject: int
    movement: str
    trial: int
    step: int
    trc_path: str
    trc_exists: bool = False
    trc_valid: bool = False
    trc_frames: int = 0
    marker_count: int = 0
    marker_fps: float | None = None
    error: str = ""


@dataclass
class ClipRecord:
    subject: int
    movement: str
    trial: int
    paired_steps: str
    marker_archive: str
    output_path: str
    motion: str
    trc_paths: str
    dataset: str = "gait120"
    expected_family: str = ""
    terrain_class: str = ""
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

    Gait120 uses X=lateral, Y=up, Z=progression. Convert to Z-up coordinates
    with [X, -Z, Y] before surface fitting.
    """

    return load_trc(path, up_axis="y")


def inspect_dataset(
    *,
    original_root: Path,
    output_root: Path,
    subjects: list[int],
    movements: tuple[str, ...],
    trials: list[int],
) -> tuple[list[StepRecord], list[ClipRecord], dict]:
    """Discover usable marker steps and report missing or corrupt recordings."""
    output_root.mkdir(parents=True, exist_ok=True)
    step_records = []
    grouped = defaultdict(list)
    for subject in subjects:
        for movement in movements:
            for trial in trials:
                for step in MOVEMENT_METADATA[movement].steps:
                    path = (
                        original_root
                        / f"S{subject:03d}"
                        / "MotionCapture"
                        / movement
                        / "TRC"
                        / f"Trial{trial:02d}"
                        / f"Step{step:02d}.trc"
                    )
                    record = StepRecord(subject, movement, trial, step, str(path), trc_exists=path.is_file())
                    if record.trc_exists:
                        try:
                            trc = load_gait120_trc(path)
                            record.trc_valid = True
                            record.trc_frames = len(trc.positions)
                            record.marker_count = len(trc.labels)
                            record.marker_fps = trc.fps
                        except Exception as exc:
                            record.error = f"TRC {type(exc).__name__}: {exc}"
                    step_records.append(record)
                    grouped[(subject, movement, trial)].append(record)
    clips = []
    for (subject, movement, trial), steps in sorted(grouped.items()):
        paired = sorted((row for row in steps if row.trc_valid), key=lambda row: row.step)
        if not paired:
            continue
        relative = Path("Gait120") / f"S{subject:03d}" / movement / f"Trial{trial:02d}"
        spec = MOVEMENT_METADATA[movement]
        clips.append(
            ClipRecord(
                subject=subject,
                movement=movement,
                trial=trial,
                paired_steps=";".join(str(row.step) for row in paired),
                marker_archive=str(output_root / ".markers" / relative / "AllSteps_markers.npz"),
                output_path=str(output_root / relative / "AllSteps_stageii.npz"),
                motion=(relative / "AllSteps_stageii").as_posix(),
                trc_paths=";".join(row.trc_path for row in paired),
                expected_family=spec.expected_family,
                terrain_class=spec.terrain_class,
                role="calibration" if movement == "LevelWalking" else "retarget",
            )
        )
    by_movement = {}
    for movement in movements:
        rows = [row for row in step_records if row.movement == movement]
        movement_clips = [clip for clip in clips if clip.movement == movement]
        by_movement[movement] = {
            "expected_steps": len(rows),
            "trc_steps": sum(row.trc_exists for row in rows),
            "valid_trc_steps": sum(row.trc_valid for row in rows),
            "clips": len(movement_clips),
            "subjects_with_clip": len({clip.subject for clip in movement_clips}),
        }
    report = {
        "report_version": REPORT_VERSION,
        "original_root": str(original_root.resolve()),
        "output_root": str(output_root.resolve()),
        "subjects_requested": subjects,
        "movements": list(movements),
        "trials": trials,
        "summary": {
            "original_subject_directories": sum((original_root / f"S{subject:03d}").is_dir() for subject in subjects),
            "expected_steps": len(step_records),
            "valid_trc_steps": sum(row.trc_valid for row in step_records),
            "clips": len(clips),
        },
        "by_movement": by_movement,
    }
    _write_csv(output_root / "steps.csv", [asdict(row) for row in step_records])
    _write_manifest(_progress_manifest(output_root), clips, output_root=output_root)
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return step_records, clips, report


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
        for field_name in ("marker_archive", "output_path"):
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


def _fit_subject(job: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ["JAX_PLATFORMS"] = "cuda" if job["device"] == "cuda" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(job["torch_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(job["torch_threads"]))

    import torch

    torch.set_num_threads(job["torch_threads"])

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
    marker_fit_quality: dict,
) -> list[dict]:
    """Set the manifest quality gate and return inspectable rejection rows."""

    reasons: dict[str, list[str]] = defaultdict(list)
    for label, rows in (("smplh_validation", validation_failures),):
        for row in rows:
            reasons[row["motion"]].append(f"{label}: {row['error']}")
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


def validate_dataset(clips: list[ClipRecord], *, output_root: Path, publish_manifest: bool = True) -> dict:
    """Validate fitted motions and publish both passing and failed manifest rows."""
    failures = []
    frames = 0
    fps_values = set()
    for clip in clips:
        try:
            validation = _validate_smplh_file(Path(clip.output_path))
            clip.output_frames = validation["frames"]
            clip.fps = validation["fps"]
            frames += validation["frames"]
            fps_values.add(validation["fps"])
        except Exception as exc:
            failures.append({"motion": clip.motion, "error": f"{type(exc).__name__}: {exc}"})
    expected = {clip.motion for clip in clips}
    actual = {
        path.relative_to(output_root).with_suffix("").as_posix()
        for path in (output_root / "Gait120").glob("S*/**/AllSteps_stageii.npz")
    }
    quality = _summarize_marker_fit_quality(clips)
    rejections = _apply_fit_quality(clips, validation_failures=failures, marker_fit_quality=quality)
    targets = [clip for clip in clips if clip.role == "retarget"]
    calibrations = [clip for clip in clips if clip.role == "calibration"]
    ready = not failures and expected == actual
    report = {
        "manifest_clips": len(clips),
        "valid_smplh_files": len(clips) - len(failures),
        "total_output_frames": frames,
        "fps_values": sorted(fps_values),
        "status_counts": dict(sorted(Counter(clip.status for clip in clips).items())),
        "missing_manifest_outputs": sorted(expected - actual),
        "unexpected_outputs": sorted(actual - expected),
        "validation_failures": failures,
        "marker_fit_quality": quality,
        "marker_fit_quality_ready": quality["ready"],
        "quality_passed_retarget": sum(clip.fit_passed for clip in targets),
        "quality_failed_retarget": sum(not clip.fit_passed for clip in targets),
        "quality_passed_calibration": sum(clip.fit_passed for clip in calibrations),
        "quality_failed_calibration": sum(not clip.fit_passed for clip in calibrations),
        "conversion_quality_rejections": rejections,
        "terra_ready": ready,
        "benchmark_ready": ready and not quality["missing_metrics"] and any(clip.fit_passed for clip in targets),
        "manifest_published": bool(publish_manifest),
    }
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = output_root / "manifest.csv" if publish_manifest else _progress_manifest(output_root)
    _write_manifest(manifest, clips, output_root=output_root)
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
    parser.add_argument("--subjects", default="1-120", help="Comma-separated subjects/ranges")
    parser.add_argument("--trials", default="1-5", help="Comma-separated trials/ranges")
    parser.add_argument(
        "--movements",
        nargs="+",
        choices=TARGET_MOVEMENTS,
        default=list(TARGET_MOVEMENTS),
    )
    parser.add_argument("--inspect-only", action="store_true")
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
    args.output_root = roots.resolve_artifact(args.output_root, base=Path.cwd())
    args.smpl_model_path = roots.resolve_model(args.smpl_model_path, base=Path.cwd())
    args.stage1_state_root = roots.resolve_artifact(args.stage1_state_root, base=Path.cwd())
    args.selection_manifest = roots.resolve_input(args.selection_manifest, base=Path.cwd())
    assert args.original_root is not None
    assert args.output_root is not None
    assert args.smpl_model_path is not None
    subjects = _parse_int_ranges(args.subjects, lower=1, upper=120)
    trials = _parse_int_ranges(args.trials, lower=1, upper=5)
    movements = tuple(args.movements)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not args.original_root.is_dir():
        raise SystemExit(f"Gait120 original root does not exist: {args.original_root}")
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
    print("Inspecting Gait120 marker recordings...", flush=True)
    _, clips, report = inspect_dataset(
        original_root=args.original_root,
        output_root=args.output_root,
        subjects=subjects,
        movements=movements,
        trials=trials,
    )
    if args.selection_manifest is not None:
        clips = _filter_clips_by_selection_manifest(clips, args.selection_manifest)
        print(
            f"Selection filter: {len(clips)} Gait120 retarget/calibration clips",
            flush=True,
        )
    print(json.dumps(report["summary"], indent=2), flush=True)
    for movement, summary in report["by_movement"].items():
        print(f"{movement}: {summary}", flush=True)
    if args.inspect_only:
        print(f"Inspection complete in {time.time() - started:.1f}s -> {args.output_root / 'report.json'}")
        return 0

    print(f"Preparing {len(clips)} paired marker archives...", flush=True)
    prepare_marker_archives(clips, redo=args.redo)
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
    report = validate_dataset(
        fitted,
        output_root=args.output_root,
        publish_manifest=args.limit is None,
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Finished in {(time.time() - started) / 60:.1f} min -> {args.output_root}", flush=True)
    return 0 if report["benchmark_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
