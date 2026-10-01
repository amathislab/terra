#!/usr/bin/env python3
"""Export PRISM's optical-MoCap SMPL labels to the common SMPL-H dataset contract."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from terra._revision import write_git_commit
from terra.paths import StorageRoots

from .adapter import load_take, selected_take_paths

FIELDS = (
    "motion",
    "dataset",
    "subject",
    "condition",
    "source_path",
    "output_path",
    "calibration_motion",
    "role",
    "fit_passed",
    "status",
    "frames",
    "fps",
    "error",
)


def _constant_betas(value: object, frames: int) -> np.ndarray:
    betas = np.asarray(value, dtype=np.float32)
    if betas.ndim == 1:
        result = betas
    elif betas.ndim == 2 and betas.shape[0] in {1, frames}:
        if betas.shape[0] == frames and not np.allclose(betas, betas[:1], atol=1e-5):
            raise ValueError("PRISM shape coefficients vary across frames")
        result = betas[0]
    else:
        raise ValueError(f"unexpected PRISM betas shape {betas.shape}")
    if len(result) < 10:
        raise ValueError(f"PRISM betas must contain at least 10 coefficients, got {len(result)}")
    return np.pad(result[:16], (0, max(0, 16 - len(result[:16]))))


def export_take(path: Path, output_root: Path, *, overwrite: bool) -> dict[str, object]:
    take = load_take(path)
    params = take["smpl_params"]
    poses = np.asarray(params["poses"], dtype=np.float32)
    trans = np.asarray(params["trans"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"PRISM poses must have shape (F, 72), got {poses.shape}")
    if trans.shape != (len(poses), 3):
        raise ValueError(f"PRISM trans must have shape ({len(poses)}, 3), got {trans.shape}")
    root_offset = np.asarray(params["root_offset"], dtype=np.float32).reshape(1, 3)
    fps = float(take["info"]["data_info"]["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid PRISM frame rate {fps}")
    pose_smplh = np.zeros((len(poses), 156), dtype=np.float32)
    pose_smplh[:, :72] = poses
    betas = _constant_betas(params["betas"], len(poses))
    gender = str(params["gender"]).strip().casefold()
    if gender not in {"neutral", "male", "female"}:
        raise ValueError(f"unsupported PRISM gender {gender!r}")

    subject = path.parent.name
    take_name = path.stem
    motion = f"PRISM/{subject}/{take_name}_poses"
    output = output_root / f"{motion}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not output.exists():
        np.savez_compressed(
            output,
            poses=pose_smplh,
            trans=trans + root_offset,
            betas=betas.astype(np.float32),
            gender=np.asarray(gender),
            mocap_framerate=np.asarray(fps, dtype=np.float32),
        )
    return {
        "motion": motion,
        "dataset": "prism",
        "subject": subject,
        "condition": take_name,
        "source_path": str(path.resolve()),
        "output_path": str(output.resolve()),
        "calibration_motion": motion,
        "role": "retarget",
        "fit_passed": True,
        "status": "converted",
        "frames": len(poses),
        "fps": fps,
        "error": "",
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    roots = StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(prog="terra convert prism", description=__doc__)
    parser.add_argument("data_root", nargs="?", type=Path, default=roots.data_root / "PRISM")
    parser.add_argument("--output-root", type=Path, default=roots.artifact_root / "prism" / "smplh")
    parser.add_argument("--take", action="append", default=[])
    parser.add_argument("--max-takes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    data_root = roots.resolve_input(args.data_root, base=Path.cwd())
    output_root = roots.resolve_artifact(args.output_root, base=Path.cwd())
    assert data_root is not None
    assert output_root is not None

    if args.take:
        paths = selected_take_paths(data_root, args.take)
    else:
        paths = selected_take_paths(data_root)
    if args.max_takes is not None:
        paths = paths[: args.max_takes]
    if not paths:
        raise SystemExit(f"no PRISM takes found below {data_root}")
    write_git_commit(output_root)

    rows: list[dict[str, object]] = []
    manifest = output_root / "manifest.csv"
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.parent.name}/{path.stem}", flush=True)
        try:
            row = export_take(path, output_root, overwrite=args.overwrite)
        except Exception as exc:
            row = {
                "motion": f"PRISM/{path.parent.name}/{path.stem}_poses",
                "dataset": "prism",
                "subject": path.parent.name,
                "condition": path.stem,
                "source_path": str(path.resolve()),
                "role": "retarget",
                "fit_passed": False,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        _write_manifest(manifest, rows)
    failures = sum(not bool(row["fit_passed"]) for row in rows)
    print(f"PRISM: {len(rows) - failures}/{len(rows)} converted -> {manifest}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
