"""Convert the complete Darmstadt MATLAB marker release to SMPL-H."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from terra._revision import write_git_commit
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
                            "marker_path": str(marker_output.resolve()),
                            "output_path": str((output_root / f"{motion}.npz").resolve()),
                            "expected_family": "steps",
                            "terrain_class": ("stairs_up" if direction == "ascent" else "stairs_down"),
                            "expected_slope_deg": "",
                            "expected_riser_m": RISERS[configuration - 1],
                            "calibration_motion": "",
                            "role": "retarget",
                        }
                    )
    return rows


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
        expected_family="steps",
        terrain_class=row["terrain_class"],
        expected_riser_m=float(row["expected_riser_m"]),
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
    for index, row in enumerate(ordered, 1):
        clip = _clip(row)
        print(
            f"[fit {index}/{len(ordered)}] markers={marker_counts[row['motion']]} {clip.motion}",
            flush=True,
        )
        result = _fit_clip(clip, args)
        results.append(result)
        _write_manifest(manifest, results)
    failures = sum(not clip.fit_passed for clip in results)
    print(f"Darmstadt: {len(results) - failures} passed, {failures} failed -> {manifest}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
