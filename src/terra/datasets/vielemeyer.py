"""Convert the complete Vielemeyer C3D release to SMPL-H."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from terra._revision import write_git_commit
from terra.datasets.marker_fitting import (
    Clip,
    _fit_clip,
    _stage1_state_path,
    _write_manifest,
)
from terra.paths import StorageRoots

FAMILIES = {
    "level_up": ("flat", "flat", 0.0),
    "level_down": ("flat", "flat", 0.0),
    "ramp_10_up": ("ramp", "ramp_up", 10.0),
    "ramp_10_down": ("ramp", "ramp_down", 10.0),
    "ramp_75_up": ("ramp", "ramp_up", 7.5),
    "ramp_75_down": ("ramp", "ramp_down", 7.5),
}


def _number(value: str | Path) -> int:
    match = re.search(r"(\d+)$", Path(value).stem)
    if match is None:
        raise ValueError(f"no numeric suffix in {value}")
    return int(match.group(1))


def _inventory(input_root: Path, output_root: Path) -> list[dict]:
    rows = []
    paths = list((input_root / "raw").glob("Ref_*/*/*.c3d"))
    paths.extend((input_root / "raw-incomplete").glob("Ref_*/*/*.c3d"))
    for path in sorted(paths):
        subject = f"Ref{_number(path.parents[1]):02d}"
        condition = path.parent.name
        family, terrain_class, slope = FAMILIES[condition]
        motion = f"Vielemeyer/{subject}/{condition}/{path.stem}_stageii"
        rows.append(
            {
                "dataset": "vielemeyer",
                "subject": subject,
                "condition": condition,
                "direction": "up" if condition.endswith("up") else "down",
                "configuration": "",
                "trial": path.stem,
                "motion": motion,
                "source_path": str(path.resolve()),
                "marker_path": str(path.resolve()),
                "output_path": str((output_root / f"{motion}.npz").resolve()),
                "expected_family": family,
                "terrain_class": terrain_class,
                "expected_slope_deg": slope,
                "expected_riser_m": "",
                "calibration_motion": "",
                "role": "retarget",
            }
        )
    return rows


def _clip(row: dict, *, role: str, calibration_motion: str) -> Clip:
    return Clip(
        dataset="vielemeyer",
        subject=row["subject"],
        condition=row["condition"],
        role=role,
        motion=row["motion"],
        source_path=row["source_path"],
        marker_path=row["marker_path"],
        output_path=row["output_path"],
        expected_family=row["expected_family"],
        terrain_class=row["terrain_class"],
        expected_slope_deg=float(row["expected_slope_deg"]),
        calibration_motion=calibration_motion,
    )


def _clips(rows: list[dict]) -> list[Clip]:
    selected_by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        selected_by_subject[row["subject"]].append(row)

    clips = []
    for subject in sorted(selected_by_subject):
        levels = sorted(
            (row for row in selected_by_subject[subject] if row["expected_family"] == "flat"),
            key=lambda row: row["motion"],
        )
        if not levels:
            raise ValueError(f"Vielemeyer subject {subject} has no level calibration trial")
        level = levels[0]
        calibration_motion = level["motion"]
        for row in selected_by_subject[subject]:
            row["calibration_motion"] = calibration_motion
            clips.append(
                _clip(
                    row,
                    role="retarget",
                    calibration_motion=calibration_motion,
                )
            )
    return clips


def _fit_subject_clips(clips: list[Clip], args: argparse.Namespace) -> Iterator[Clip]:
    """Fit one subject after selecting a passing same-subject level calibration.

    The first level clip remains the deterministic default. Unlike ordinary clips, a
    calibration clip always receives the longer Stage-II retry when it misses any marker
    residual gate. If it still fails, each subsequent level clip is tried with a fresh
    Stage-I state until one passes. Failed probes are then refitted against the selected
    state, so every published subject motion uses one common calibration. Any other clip
    that misses the gate also receives the longer Stage-II retry before rejection.
    """

    if not clips:
        return
    subject = clips[0].subject
    if any(clip.subject != subject for clip in clips):
        raise ValueError("Vielemeyer calibration selection requires exactly one subject")
    candidates = sorted(
        (clip for clip in clips if clip.expected_family == "flat"),
        key=lambda clip: clip.motion,
    )
    if not candidates:
        raise ValueError(f"Vielemeyer subject {subject} has no level calibration trial")

    state_path = _stage1_state_path(candidates[0], args)
    probes: dict[str, Clip] = {}
    rejected_probes: set[str] = set()
    selected: Clip | None = None
    selected_state_path: Path | None = None
    for candidate in candidates:
        # Give every candidate a stable, independent Stage-I cache. This both prevents a
        # rejected state from leaking into the next probe and keeps successful calibration
        # selection resumable without claiming that a state came from a different motion.
        candidate_args = copy.copy(args)
        candidate_args.stage1_state_root = (
            state_path.parent.parent
            / ".calibration-candidates"
            / subject
            / f"{candidate.condition}--{Path(candidate.output_path).stem}"
        )
        candidate_state_path = _stage1_state_path(candidate, candidate_args)
        result = _fit_clip(
            candidate,
            candidate_args,
            force_refit=not candidate_state_path.is_file(),
            retry_failed_fit=True,
        )
        probes[result.motion] = result
        if result.fit_passed:
            selected = result
            selected_state_path = candidate_state_path
            break
        rejected_probes.add(result.motion)

    if selected is None:
        state_path.unlink(missing_ok=True)
        reason = (
            f"CalibrationError: no same-subject level motion passed the "
            f"{args.max_mean_mm:g}/{args.max_p95_mm:g}/{args.max_error_mm:g} mm marker gate"
        )
        for clip in clips:
            result = probes.get(clip.motion, clip)
            if result.motion not in probes:
                result.status = "failed"
                result.error = reason
            yield result
        return

    assert selected_state_path is not None
    state_changed = True
    if state_path.is_file():
        with (
            np.load(state_path, allow_pickle=False) as current,
            np.load(selected_state_path, allow_pickle=False) as selected_state,
        ):
            state_changed = current.files != selected_state.files or any(
                not np.array_equal(current[name], selected_state[name]) for name in current.files
            )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_changed:
        temporary_state = state_path.with_name(f".{state_path.name}.tmp")
        shutil.copyfile(selected_state_path, temporary_state)
        temporary_state.replace(state_path)
    calibration_motion = selected.motion
    print(f"Vielemeyer {subject}: calibration {calibration_motion}", flush=True)
    for clip in clips:
        clip.calibration_motion = calibration_motion
        if clip.motion == calibration_motion:
            selected.calibration_motion = calibration_motion
            yield selected
            continue
        # A changed calibration invalidates every prior subject fit. Failed existing clips
        # also receive the full retry even when their calibration state is unchanged.
        yield _fit_clip(
            clip,
            args,
            force_refit=state_changed or clip.motion in rejected_probes,
            retry_failed_fit=True,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    roots = StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(
        prog="terra convert vielemeyer",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=roots.data_root / "Vielemeyer-Ramp-Walking",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=roots.artifact_root / "vielemeyer" / "smplh",
    )
    parser.add_argument("--smpl-model-path", type=Path, default=roots.model_root)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--redo", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = StorageRoots.from_environment(Path.cwd())
    args.vielemeyer_root = roots.resolve_input(args.input_root, base=Path.cwd())
    args.output_root = roots.resolve_artifact(args.output_root, base=Path.cwd())
    smpl_model_path = roots.resolve_model(args.smpl_model_path, base=Path.cwd())
    assert args.vielemeyer_root is not None
    assert args.output_root is not None
    assert smpl_model_path is not None
    args.smpl_model_path = str(smpl_model_path)
    args.target_fps = 50.0
    args.stage1_iters = 100
    args.stage2_iters = 80
    args.n_ref_frames = 12
    args.stage1_shape_solver = "joint_dogleg_jax"
    args.least_avail_markers = 0.8
    args.vielemeyer_foot_marker_weight = 1.0
    args.vielemeyer_heel_marker_weight = None
    args.enforce_knee_hinge = True
    args.max_mean_mm = 20.0
    args.max_p95_mm = 40.0
    args.max_error_mm = 60.0
    args.redo_stage1 = False

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_git_commit(args.output_root)
    inventory = _inventory(args.vielemeyer_root, args.output_root)
    clips = _clips(inventory)

    results = []
    manifest = args.output_root / "manifest.csv"
    clips_by_subject: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        clips_by_subject[clip.subject].append(clip)
    for subject in sorted(clips_by_subject):
        for result in _fit_subject_clips(clips_by_subject[subject], args):
            results.append(result)
            print(f"[{len(results)}/{len(clips)}] {result.motion}", flush=True)
            _write_manifest(manifest, results)
    failures = sum(not clip.fit_passed and clip.role == "retarget" for clip in results)
    passed = sum(clip.fit_passed and clip.role == "retarget" for clip in results)
    print(f"Vielemeyer: {passed} passed, {failures} failed -> {manifest}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
