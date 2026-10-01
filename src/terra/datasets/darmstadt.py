"""Convert the complete Darmstadt MATLAB marker release to SMPL-H."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from terra._revision import write_git_commit
from terra.datasets.biomechanics import (
    biomechanics_path,
    motion_clock,
    resample_linear,
    validate_biomechanics,
    write_biomechanics,
)
from terra.datasets.marker_fitting import (
    DARMSTADT_MARKERS,
    Clip,
    _fit_clip,
    _interpolate,
    _touchdown_crop,
    _write_manifest,
)
from terra.mat import extract_mat_markers
from terra.paths import StorageRoots

RISERS = (0.10, 0.17, 0.24, 0.10, 0.17, 0.24)
DARMSTADT_EMG_FIELDS = tuple(
    f"{muscle}_{side}"
    for side in ("r", "l")
    for muscle in ("bcf", "foo", "gas", "rcf", "sha", "sol", "tib", "vas")
)
DARMSTADT_MUSCLES = {
    "bcf": "BicepsFemoris",
    "foo": "",
    "gas": "GastrocnemiusLateralis",
    "rcf": "RectusFemoris",
    "sha": "",
    "sol": "Soleus",
    "tib": "TibialisAnterior",
    "vas": "VastusLateralis",
}
DARMSTADT_EMG_TYPES = tuple(
    "non_muscle_sensor" if field[:3] in {"foo", "sha"} else "muscle"
    for field in DARMSTADT_EMG_FIELDS
)
DARMSTADT_MAT_MARKER_SCHEMA: dict[str, object] = {
    "version": 1,
    "record_path": [
        "Marker",
        {"selector": "configuration"},
        {"selector": "trial"},
    ],
    "marker_fields": dict(DARMSTADT_MARKERS),
    "marker_field_axis_order": "tc",
    "fps_path": "Marker_fs",
    "units": "m",
    "axes": ["x", "y", "z"],
    "zero_is_missing": False,
}


def _number(value: str | Path) -> int:
    match = re.search(r"(\d+)$", Path(value).stem)
    if match is None:
        raise ValueError(f"no numeric suffix in {value}")
    return int(match.group(1))


def _inventory(input_root: Path, output_root: Path) -> list[dict]:
    from scipy.io import loadmat

    rows = []
    touchdown_root = input_root / "touchdowns" / "Processed" / "Touchdowns"
    for marker_path in sorted(input_root.glob("Marker*.mat"), key=_number):
        number = _number(marker_path)
        subject = f"D{number:02d}"
        touchdown_path = touchdown_root / f"Touchdowns{number}.mat"
        emg_path = _signal_path(input_root, "EMG", number)
        forces_path = _signal_path(input_root, "Forces", number)
        touchdown = loadmat(touchdown_path, struct_as_record=False, squeeze_me=True)
        for configuration in range(1, 7):
            for direction in ("ascent", "descent"):
                trials = np.atleast_1d(touchdown[f"TD_{direction}"][configuration - 1])
                for trial in range(1, len(trials) + 1):
                    motion = f"Darmstadt/{subject}/config{configuration:02d}/trial{trial:02d}/{direction}_stageii"
                    marker_output = (
                        output_root
                        / ".markers"
                        / "Darmstadt"
                        / subject
                        / f"config{configuration:02d}"
                        / f"trial{trial:02d}_{direction}.npz"
                    )
                    rows.append(
                        {
                            "dataset": "darmstadt",
                            "subject": subject,
                            "condition": f"riser_{RISERS[configuration - 1]:.2f}",
                            "direction": direction,
                            "configuration": configuration,
                            "trial": trial,
                            "motion": motion,
                            "source_path": str(marker_path.resolve()),
                            "touchdown_path": str(touchdown_path.resolve()),
                            "emg_path": str(emg_path.resolve()),
                            "forces_path": str(forces_path.resolve()),
                            "marker_path": str(marker_output.resolve()),
                            "output_path": str((output_root / f"{motion}.npz").resolve()),
                            "biomechanics_path": str(
                                biomechanics_path(output_root / f"{motion}.npz").resolve()
                            ),
                            "expected_family": "steps",
                            "terrain_class": ("stairs_up" if direction == "ascent" else "stairs_down"),
                            "expected_slope_deg": "",
                            "expected_riser_m": RISERS[configuration - 1],
                            "calibration_motion": "",
                            "role": "retarget",
                        }
                    )
    return rows


def _signal_path(input_root: Path, kind: str, subject: int) -> Path:
    """Resolve both the official release layout and locally extracted flat layout."""

    filename = f"{kind}{subject}.mat"
    candidates = (
        input_root / "Preprocessed" / kind / filename,
        input_root / kind / filename,
        input_root / filename,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _preflight_biomechanics(rows: list[dict]) -> None:
    """Require trial-level EMG and force files; processed group means are not substitutes."""

    required = {
        Path(row[field])
        for row in rows
        for field in ("source_path", "touchdown_path", "emg_path", "forces_path")
    }
    missing = sorted(path for path in required if not path.is_file())
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:30])
        suffix = f"\n  ... and {len(missing) - 30} more" if len(missing) > 30 else ""
        raise FileNotFoundError(
            "Darmstadt biomechanical preflight is missing trial-level source files. "
            "Extract Preprocessed/EMG/EMG1..12.mat and "
            "Preprocessed/Forces/Forces1..12.mat from the official release; the "
            "FullyProcessed 100-point group averages are not time-synchronized trials.\n"
            f"{preview}{suffix}"
        )


def _export_subject(rows: list[dict], args: argparse.Namespace) -> dict[str, int]:
    from scipy.io import loadmat

    marker_data = loadmat(rows[0]["source_path"], struct_as_record=False, squeeze_me=True)
    touchdown_data = loadmat(rows[0]["touchdown_path"], struct_as_record=False, squeeze_me=True)
    fps = float(np.asarray(marker_data["Marker_fs"]).reshape(()))
    touchdown_fps = float(np.asarray(touchdown_data["TD_fs"]).reshape(()))
    if not np.isclose(fps, touchdown_fps):
        raise ValueError(f"marker rate {fps} != touchdown rate {touchdown_fps}")

    marker_counts = {}
    for row in rows:
        configuration = int(row["configuration"]) - 1
        trial = int(row["trial"]) - 1
        marker_motion = extract_mat_markers(
            marker_data,
            DARMSTADT_MAT_MARKER_SCHEMA,
            selectors={"configuration": configuration, "trial": trial},
        )
        annotation = np.atleast_1d(touchdown_data[f"TD_{row['direction']}"][configuration])[trial]
        positions = marker_motion.positions
        start, end = _touchdown_crop(
            annotation,
            len(positions),
            fps,
            args.padding_s,
        )
        positions = positions[start:end]
        labels = np.asarray(marker_motion.labels)
        available = np.all(np.isfinite(positions).any(axis=0), axis=1)
        positions = _interpolate(positions[:, available])
        labels = labels[available]
        output = Path(row["marker_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.redo or not output.exists():
            np.savez_compressed(
                output,
                positions=positions,
                labels=labels,
                fps=np.array(fps, dtype=np.float32),
                source_path=np.array(str(Path(row["source_path"]).resolve())),
                touchdown_path=np.array(str(Path(row["touchdown_path"]).resolve())),
                crop_start_frame=np.array(start, dtype=np.int64),
                crop_end_frame=np.array(end, dtype=np.int64),
                configuration=np.array(configuration + 1, dtype=np.int64),
                trial=np.array(trial + 1, dtype=np.int64),
                direction=np.array(row["direction"]),
            )
        marker_counts[row["motion"]] = len(labels)
    return marker_counts


def _clip(row: dict) -> Clip:
    return Clip(
        dataset="darmstadt",
        subject=row["subject"],
        condition=row["condition"],
        role="retarget",
        motion=row["motion"],
        source_path=row["source_path"],
        marker_path=row["marker_path"],
        output_path=row["output_path"],
        biomechanics_path=row["biomechanics_path"],
        expected_family="steps",
        terrain_class=row["terrain_class"],
        expected_riser_m=float(row["expected_riser_m"]),
    )


def _mat_trial(container, configuration: int, trial: int):
    configurations = np.asarray(container, dtype=object).reshape(-1)
    trials = np.asarray(configurations[configuration - 1], dtype=object).reshape(-1)
    return trials[trial - 1]


def _trial_field(trial, field: str) -> np.ndarray:
    if hasattr(trial, field):
        value = getattr(trial, field)
    elif isinstance(trial, np.void) and trial.dtype.names and field in trial.dtype.names:
        value = trial[field]
    else:
        names = getattr(trial, "_fieldnames", ()) or getattr(getattr(trial, "dtype", None), "names", ()) or ()
        raise KeyError(f"Darmstadt trial has no {field!r}; available fields: {tuple(names)}")
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _load_biomechanics_sources(row: dict) -> dict:
    from scipy.io import loadmat

    emg = loadmat(row["emg_path"], struct_as_record=False, squeeze_me=True)
    forces = loadmat(row["forces_path"], struct_as_record=False, squeeze_me=True)
    return {
        "emg": emg["Emg"],
        "emg_fps": float(np.asarray(emg["Emg_fs"]).reshape(())),
        "emg_unit": " ".join(str(value) for value in np.asarray(emg["Emg_unit"]).reshape(-1)),
        "forces": forces["Forces"],
        "forces_fps": float(np.asarray(forces["Forces_fs"]).reshape(())),
        "forces_unit": " ".join(
            str(value) for value in np.asarray(forces["Forces_unit"]).reshape(-1)
        ),
    }


def _emg_envelope(values: np.ndarray, fps: float) -> np.ndarray:
    """Apply the release's zero-lag bandpass/rectification plus a 6 Hz envelope."""

    from scipy.signal import butter, sosfiltfilt

    if fps <= 900.0:
        raise ValueError(f"Darmstadt EMG rate {fps:g} Hz cannot represent the documented 450 Hz bandpass")
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    if not np.isfinite(centered).all():
        raise ValueError("Darmstadt EMG contains NaN or infinity")
    bandpass = butter(4, (20.0, 450.0), btype="bandpass", fs=fps, output="sos")
    lowpass = butter(4, 6.0, btype="lowpass", fs=fps, output="sos")
    rectified = np.abs(sosfiltfilt(bandpass, centered, axis=0))
    return np.maximum(sosfiltfilt(lowpass, rectified, axis=0), 0.0)


def _prepare_biomechanics(
    clip: Clip,
    row: dict,
    sources: dict,
    *,
    redo: bool = False,
) -> None:
    output = Path(clip.biomechanics_path or biomechanics_path(clip.output_path))
    clip.biomechanics_path = str(output)
    if output.exists() and not redo:
        try:
            validate_biomechanics(output, motion_path=clip.output_path)
            return
        except Exception:
            pass
    clock, fps = motion_clock(clip.output_path)
    with np.load(clip.marker_path, allow_pickle=False) as markers:
        crop_start = int(np.asarray(markers["crop_start_frame"]).reshape(()))
        crop_end = int(np.asarray(markers["crop_end_frame"]).reshape(()))
        marker_fps = float(np.asarray(markers["fps"]).reshape(()))
    configuration = int(row["configuration"])
    trial_index = int(row["trial"])

    emg_trial = _mat_trial(sources["emg"], configuration, trial_index)
    emg_columns = [_trial_field(emg_trial, field) for field in DARMSTADT_EMG_FIELDS]
    emg_length = min(map(len, emg_columns))
    emg_raw_full = np.stack([column[:emg_length] for column in emg_columns], axis=-1)
    envelope_cache = sources.setdefault("emg_envelope_cache", {})
    cache_key = (configuration, trial_index)
    if cache_key not in envelope_cache:
        envelope_cache[cache_key] = _emg_envelope(
            emg_raw_full,
            sources["emg_fps"],
        ).astype(np.float32)
    emg_envelope_full = envelope_cache[cache_key]
    emg_start = round(crop_start / marker_fps * sources["emg_fps"])
    emg_end = round(crop_end / marker_fps * sources["emg_fps"])
    if emg_end > emg_length:
        raise ValueError(
            f"EMG ends at sample {emg_length}, before synchronized crop sample {emg_end}"
        )
    emg_raw = emg_raw_full[emg_start:emg_end].astype(np.float32)
    emg_envelope = emg_envelope_full[emg_start:emg_end].astype(np.float32)
    emg_time = np.arange(len(emg_raw), dtype=np.float64) / sources["emg_fps"]
    emg = {
        "emg_available": np.array(True),
        "emg": resample_linear(emg_time, emg_envelope, clock),
        "emg_native": emg_raw,
        "emg_native_time_s": emg_time,
        "emg_channels": np.asarray(DARMSTADT_EMG_FIELDS),
        "emg_muscles": np.asarray(
            [DARMSTADT_MUSCLES[field[:3]] for field in DARMSTADT_EMG_FIELDS]
        ),
        "emg_channel_types": np.asarray(DARMSTADT_EMG_TYPES),
        "emg_channel_sides": np.asarray(
            ["left" if field.endswith("_l") else "right" for field in DARMSTADT_EMG_FIELDS]
        ),
        "emg_units": np.array(
            f"{sources['emg_unit']} (native raw; aligned linear envelope)"
        ),
        "emg_processing": np.array(
            "aligned: whole-trial mean removal, zero-phase 20-450 Hz order-4 "
            "Butterworth, rectification, zero-phase 6 Hz order-4 Butterworth"
        ),
        "emg_source_paths": np.asarray([str(Path(row["emg_path"]).resolve())]),
    }

    force_trial = _mat_trial(sources["forces"], configuration, trial_index)
    side_arrays = []
    side_valid = []
    for field in ("All_l", "All_r"):
        if hasattr(force_trial, field):
            values = np.asarray(getattr(force_trial, field), dtype=np.float64)
        elif isinstance(force_trial, np.void) and force_trial.dtype.names and field in force_trial.dtype.names:
            values = np.asarray(force_trial[field], dtype=np.float64)
        else:
            raise KeyError(f"Darmstadt force trial has no {field!r}")
        values = np.squeeze(values)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"{field} has shape {values.shape}, expected (samples, 6)")
        side_valid.append(np.isfinite(values[:, :3]).all(axis=-1))
        side_arrays.append(np.nan_to_num(values, nan=0.0))
    force_values = np.stack(side_arrays, axis=1)
    force_valid = np.stack(side_valid, axis=1)
    if not np.isclose(sources["forces_fps"], marker_fps):
        raise ValueError(f"force rate {sources['forces_fps']} != marker rate {marker_fps}")
    if crop_end > len(force_values):
        raise ValueError(f"Forces end at {len(force_values)}, before crop frame {crop_end}")
    force_values = force_values[crop_start:crop_end]
    force_valid = force_valid[crop_start:crop_end]
    force_time = np.arange(len(force_values), dtype=np.float64) / sources["forces_fps"]
    native_force = force_values[..., :3].astype(np.float32)
    compact_force_units = sources["forces_unit"].casefold().replace(" ", "")
    if "nmm" in compact_force_units:
        moment_scale = 1e-3
    elif "nm" in compact_force_units:
        moment_scale = 1.0
    elif compact_force_units.endswith("n") and "force" in compact_force_units:
        # The release labels the six-column All_l/All_r arrays only as force/N,
        # although its documentation defines columns 4:6 as plate moments. Their
        # published values are already in Nm (free moments are single-digit values).
        moment_scale = 1.0
    else:
        raise ValueError(f"Cannot convert Darmstadt moment units: {sources['forces_unit']!r}")
    native_moment = (force_values[..., 3:] * moment_scale).astype(np.float32)
    aligned_force = resample_linear(force_time, native_force, clock)
    aligned_moment = resample_linear(force_time, native_moment, clock)
    aligned_valid = resample_linear(force_time, force_valid.astype(np.float32), clock) > 0.5
    native_cop = np.full_like(native_force, np.nan)
    aligned_cop = np.full_like(aligned_force, np.nan)
    grf = {
        "grf_available": np.array(True),
        "grf_cop_available": np.array(False),
        "grf_force": aligned_force,
        "grf_moment": aligned_moment,
        "grf_cop": aligned_cop,
        "grf_valid": aligned_valid,
        "grf_force_native": native_force,
        "grf_moment_native": native_moment,
        "grf_cop_native": native_cop,
        "grf_valid_native": force_valid,
        "grf_native_time_s": force_time,
        "grf_channels": np.asarray(("left", "right")),
        "grf_source_paths": np.asarray([str(Path(row["forces_path"]).resolve())]),
    }
    write_biomechanics(
        output,
        dataset="darmstadt",
        motion=clip.motion,
        motion_time_s=clock,
        motion_fps=fps,
        synchronization=(
            "All streams share trial start; marker/force cropped by touchdowns at 200 Hz; "
            "native EMG crop derived from the same physical interval"
        ),
        emg=emg,
        grf=grf,
        metadata={
            "source_crop_start_frame": np.array(crop_start, dtype=np.int64),
            "source_crop_end_frame": np.array(crop_end, dtype=np.int64),
            "source_marker_fps": np.array(marker_fps, dtype=np.float64),
            "motion_source_frame": crop_start + np.rint(clock * marker_fps).astype(np.int64),
            "source_emg_units": np.array(sources["emg_unit"]),
            "source_force_units": np.array(sources["forces_unit"]),
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    roots = StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(
        prog="terra convert darmstadt",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=roots.data_root / "Darmstadt-Stair-Ambulation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=roots.artifact_root / "darmstadt" / "smplh",
    )
    parser.add_argument("--smpl-model-path", type=Path, default=roots.model_root)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--redo", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = StorageRoots.from_environment(Path.cwd())
    args.darmstadt_root = roots.resolve_input(args.input_root, base=Path.cwd())
    args.output_root = roots.resolve_artifact(args.output_root, base=Path.cwd())
    smpl_model_path = roots.resolve_model(args.smpl_model_path, base=Path.cwd())
    assert args.darmstadt_root is not None
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
    args.padding_s = 0.85
    args.redo_stage1 = False

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_git_commit(args.output_root)
    inventory = _inventory(args.darmstadt_root, args.output_root)
    print("Checking trial-level EMG and force sources...", flush=True)
    _preflight_biomechanics(inventory)

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in inventory:
        by_subject[row["subject"]].append(row)
    marker_counts: dict[str, int] = {}
    for index, subject in enumerate(sorted(by_subject), 1):
        print(f"[export {index}/{len(by_subject)}] {subject}", flush=True)
        marker_counts.update(_export_subject(by_subject[subject], args))

    ordered = []
    for subject in sorted(by_subject):
        ordered.extend(
            sorted(
                by_subject[subject],
                key=lambda row: (
                    -marker_counts[row["motion"]],
                    row["motion"],
                ),
            )
        )
    results = []
    manifest = args.output_root / "manifest.csv"
    loaded_subject = ""
    biomechanics_sources = None
    for index, row in enumerate(ordered, 1):
        clip = _clip(row)
        print(
            f"[fit {index}/{len(ordered)}] markers={marker_counts[row['motion']]} {clip.motion}",
            flush=True,
        )
        result = _fit_clip(clip, args)
        if Path(result.output_path).is_file():
            try:
                if loaded_subject != result.subject:
                    biomechanics_sources = _load_biomechanics_sources(row)
                    loaded_subject = result.subject
                assert biomechanics_sources is not None
                _prepare_biomechanics(result, row, biomechanics_sources, redo=args.redo)
            except Exception as exc:
                result.fit_passed = False
                result.status = "failed"
                result.error = f"BiomechanicsError: {type(exc).__name__}: {exc}"
        results.append(result)
        _write_manifest(manifest, results)
    failures = sum(not clip.fit_passed for clip in results)
    print(f"Darmstadt: {len(results) - failures} passed, {failures} failed -> {manifest}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
