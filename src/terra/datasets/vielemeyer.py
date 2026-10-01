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
from terra.datasets.biomechanics import (
    biomechanics_path,
    empty_emg,
    motion_clock,
    resample_linear,
    validate_biomechanics,
    write_biomechanics,
)
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
                "biomechanics_path": str(
                    biomechanics_path(output_root / f"{motion}.npz").resolve()
                ),
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
        biomechanics_path=row["biomechanics_path"],
        expected_family=row["expected_family"],
        terrain_class=row["terrain_class"],
        expected_slope_deg=float(row["expected_slope_deg"]),
        calibration_motion=calibration_motion,
    )


def _preflight_biomechanics(clips: list[Clip]) -> None:
    """Fail before fitting if a C3D does not contain its advertised force plates."""

    import ezc3d

    failures = []
    for clip in clips:
        try:
            c3d = ezc3d.c3d(clip.source_path)
            used = int(np.asarray(c3d["parameters"]["FORCE_PLATFORM"]["USED"]["value"]).reshape(-1)[0])
            if used < 1:
                raise ValueError("FORCE_PLATFORM:USED is zero")
        except Exception as exc:
            failures.append(f"{clip.motion}: {type(exc).__name__}: {exc}")
    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures[:20])
        suffix = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
        raise FileNotFoundError(
            f"Vielemeyer biomechanical preflight failed for {len(failures)} C3D files:\n"
            f"{preview}{suffix}"
        )


def _foot_centres(c3d: dict, analog_frames: int) -> dict[str, np.ndarray]:
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    points = np.asarray(c3d["data"]["points"][:3], dtype=np.float64)
    point_frames = points.shape[-1]
    centres = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        marker_indices = [labels.index(f"{prefix}HEE"), labels.index(f"{prefix}TOE")]
        centre = np.nanmean(np.stack([points[:, index, :].T for index in marker_indices]), axis=0)
        source_clock = np.linspace(0.0, 1.0, point_frames)
        analog_clock = np.linspace(0.0, 1.0, analog_frames)
        centres[side] = np.stack(
            [np.interp(analog_clock, source_clock, centre[:, axis]) for axis in range(3)],
            axis=-1,
        )
    return centres


def _extract_vielemeyer_grf(source_path: str | Path) -> tuple[dict, list[str]]:
    """Extract force plates, assign each contact to its nearest foot, and use SI units."""

    import ezc3d

    c3d = ezc3d.c3d(str(source_path), extract_forceplat_data=True)
    platforms = c3d["data"]["platform"]
    if not platforms:
        raise ValueError("C3D contains no extractable force-platform samples")
    lengths = {np.asarray(platform["force"]).shape[-1] for platform in platforms}
    if len(lengths) != 1:
        raise ValueError(f"Force-platform sample counts disagree: {sorted(lengths)}")
    samples = lengths.pop()
    analog_fps = float(c3d["header"]["analogs"]["frame_rate"])
    native_time = np.arange(samples, dtype=np.float64) / analog_fps
    feet = _foot_centres(c3d, samples)
    force = np.zeros((samples, 2, 3), dtype=np.float64)
    moment = np.zeros_like(force)
    cop_weighted = np.zeros_like(force)
    cop_weight = np.zeros((samples, 2), dtype=np.float64)
    assignments = []
    platform_forces = []
    platform_moments = []
    platform_cops = []
    platform_valid = []
    for index, platform in enumerate(platforms, start=1):
        platform_force = np.asarray(platform["force"], dtype=np.float64).T
        platform_moment = np.asarray(platform["moment"], dtype=np.float64).T
        platform_cop = np.asarray(platform["center_of_pressure"], dtype=np.float64).T
        if platform.get("unit_force") != "N" or platform.get("unit_moment") != "Nmm" or platform.get("unit_position") != "mm":
            raise ValueError(
                f"Unsupported platform units {platform.get('unit_force')}/"
                f"{platform.get('unit_moment')}/{platform.get('unit_position')}"
            )
        safe_cop = np.nan_to_num(platform_cop, nan=0.0, posinf=0.0, neginf=0.0)
        active = np.linalg.norm(platform_force, axis=-1) > 20.0
        platform_forces.append(platform_force.astype(np.float32))
        platform_moments.append((platform_moment / 1000.0).astype(np.float32))
        platform_cops.append((safe_cop / 1000.0).astype(np.float32))
        platform_valid.append(active)
        if not active.any():
            assignments.append(f"platform{index}:inactive")
            continue
        distances = {
            side: float(np.nanmedian(np.linalg.norm(platform_cop[active] - centre[active], axis=-1)))
            for side, centre in feet.items()
        }
        side = min(distances, key=distances.get)
        side_index = ("left", "right").index(side)
        assignments.append(f"platform{index}:{side}")
        force[:, side_index] += platform_force
        moment[:, side_index] += platform_moment / 1000.0
        weight = np.where(active, np.abs(platform_force[:, 2]), 0.0)
        cop_weighted[:, side_index] += safe_cop / 1000.0 * weight[:, None]
        cop_weight[:, side_index] += weight
    cop = np.zeros_like(force)
    np.divide(cop_weighted, cop_weight[..., None], out=cop, where=cop_weight[..., None] > 0)
    valid = cop_weight > 0
    return (
        {
            "native_time_s": native_time,
            "force": force.astype(np.float32),
            "moment": moment.astype(np.float32),
            "cop": cop.astype(np.float32),
            "valid": valid,
            "channels": np.asarray(("left", "right")),
            "fps": analog_fps,
            "marker_fps": float(c3d["header"]["points"]["frame_rate"]),
            "marker_first_frame": int(c3d["header"]["points"]["first_frame"]),
            "platform_force": np.stack(platform_forces, axis=1),
            "platform_moment": np.stack(platform_moments, axis=1),
            "platform_cop": np.stack(platform_cops, axis=1),
            "platform_valid": np.stack(platform_valid, axis=1),
        },
        assignments,
    )


def _prepare_biomechanics(clip: Clip, *, redo: bool = False) -> None:
    output = Path(clip.biomechanics_path or biomechanics_path(clip.output_path))
    clip.biomechanics_path = str(output)
    if output.exists() and not redo:
        try:
            validate_biomechanics(output, motion_path=clip.output_path)
            return
        except Exception:
            pass
    clock, fps = motion_clock(clip.output_path)
    native, assignments = _extract_vielemeyer_grf(clip.source_path)
    aligned_force = resample_linear(native["native_time_s"], native["force"], clock)
    aligned_moment = resample_linear(native["native_time_s"], native["moment"], clock)
    aligned_cop = resample_linear(native["native_time_s"], native["cop"], clock)
    aligned_valid = np.linalg.norm(aligned_force, axis=-1) > 20.0
    grf = {
        "grf_available": np.array(True),
        "grf_cop_available": np.array(True),
        "grf_force": aligned_force,
        "grf_moment": aligned_moment,
        "grf_cop": aligned_cop,
        "grf_valid": aligned_valid,
        "grf_force_native": native["force"],
        "grf_moment_native": native["moment"],
        "grf_cop_native": native["cop"],
        "grf_valid_native": native["valid"],
        "grf_native_time_s": native["native_time_s"],
        "grf_channels": native["channels"],
        "grf_source_paths": np.asarray([str(Path(clip.source_path).resolve())]),
    }
    write_biomechanics(
        output,
        dataset="vielemeyer",
        motion=clip.motion,
        motion_time_s=clock,
        motion_fps=fps,
        synchronization="C3D point and analog clocks share their recorded trial origin",
        emg=empty_emg(len(clock)),
        grf=grf,
        metadata={
            "grf_platform_assignment": np.asarray(assignments),
            "grf_platform_force_native": native["platform_force"],
            "grf_platform_moment_native": native["platform_moment"],
            "grf_platform_cop_native": native["platform_cop"],
            "grf_platform_valid_native": native["platform_valid"],
            "grf_contact_threshold_n": np.array(20.0, dtype=np.float32),
            "motion_source_frame": (
                native["marker_first_frame"]
                + np.rint(clock * native["marker_fps"]).astype(np.int64)
            ),
        },
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
    print(f"Checking force-platform data in {len(clips)} C3D files...", flush=True)
    _preflight_biomechanics(clips)

    results = []
    manifest = args.output_root / "manifest.csv"
    clips_by_subject: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        clips_by_subject[clip.subject].append(clip)
    for subject in sorted(clips_by_subject):
        for result in _fit_subject_clips(clips_by_subject[subject], args):
            if Path(result.output_path).is_file():
                try:
                    _prepare_biomechanics(result, redo=args.redo)
                except Exception as exc:
                    result.fit_passed = False
                    result.status = "failed"
                    result.error = f"BiomechanicsError: {type(exc).__name__}: {exc}"
            results.append(result)
            print(f"[{len(results)}/{len(clips)}] {result.motion}", flush=True)
            _write_manifest(manifest, results)
    failures = sum(not clip.fit_passed and clip.role == "retarget" for clip in results)
    passed = sum(clip.fit_passed and clip.role == "retarget" for clip in results)
    print(f"Vielemeyer: {passed} passed, {failures} failed -> {manifest}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
