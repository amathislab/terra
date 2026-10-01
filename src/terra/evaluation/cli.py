"""Installed command for explicit-manifest unified retargeting evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from terra._revision import write_git_commit
from terra.evaluation.annotations import frame_metadata
from terra.evaluation.audit import audit_metric_rows
from terra.evaluation.evaluator import (
    PER_MOTION_FIELDS,
    Thresholds,
    evaluate_method_motion,
)
from terra.evaluation.registry import AUTHORITATIVE_METRICS
from terra.evaluation.reporting import (
    aggregate,
    aggregate_joint_limit_sensitivity,
    write_joint_limit_sensitivity,
    write_summary_tables,
)
from terra.evaluation.settings import (
    BENCHMARK_THRESHOLDS,
    UNIFIED_AGGREGATION,
    UNIFIED_THRESHOLDS,
    UNIFIED_TIMELINE,
)


def parse_assignment(value: str, what: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{what} must be LABEL=value, got {value!r}")
    label, assigned = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError(f"{what} label may not be empty")
    return label.strip(), assigned.strip()


def read_motion_ids(path: Path) -> list[str]:
    """Read exact motion IDs from an explicit CSV manifest or line list."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "motion" not in reader.fieldnames:
                raise ValueError(f"motion manifest has no motion column: {path}")
            motions = [row.get("motion", "").strip() for row in reader]
    else:
        motions = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not motions or any(not motion for motion in motions):
        raise ValueError(f"motion manifest is empty or incomplete: {path}")
    if len(motions) != len(set(motions)):
        raise ValueError(f"motion manifest contains duplicate IDs: {path}")
    return motions


def read_common_intervals(path: Path) -> dict[str, tuple[float, float]]:
    """Read one consistent frozen benchmark interval for every motion in a CSV."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"motion", "common_start_s", "common_end_s"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"common-interval CSV is missing columns: {', '.join(sorted(missing))}")
        intervals: dict[str, tuple[float, float]] = {}
        for row in reader:
            motion = row.get("motion", "").strip()
            start_text = row.get("common_start_s", "").strip()
            end_text = row.get("common_end_s", "").strip()
            if not motion or not start_text or not end_text:
                continue
            start, end = float(start_text), float(end_text)
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError(f"invalid common interval for {motion!r}: {start_text}..{end_text}")
            previous = intervals.get(motion)
            if previous is not None and not (
                math.isclose(previous[0], start, abs_tol=1.0e-9) and math.isclose(previous[1], end, abs_tol=1.0e-9)
            ):
                raise ValueError(f"inconsistent common intervals for {motion!r}")
            intervals[motion] = (start, end)
    if not intervals:
        raise ValueError(f"common-interval CSV contains no usable rows: {path}")
    return intervals


def manifest_is_flat(label: str, path: Path) -> bool:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        terrain_classes = {row.get("terrain_class", "").strip().lower() for row in rows}
        terrain_classes.discard("")
        if terrain_classes:
            return terrain_classes <= {"flat", "level", "level_ground"}
    return label.strip().lower() in {"flat", "level", "level ground"}


def validate_class_motions(
    label: str,
    path: Path,
    motions: list[str],
    *,
    allow_flat_name_conflicts: bool = False,
) -> None:
    """Reject known non-flat IDs from a manifest explicitly declared flat."""
    if not manifest_is_flat(label, path) or allow_flat_name_conflicts:
        return
    terrain_tokens = ("beam", "stair", "step", "ramp", "chair", "seat", "stone", "obstacle")
    bad = [motion for motion in motions if any(token in motion.lower() for token in terrain_tokens)]
    if bad:
        raise ValueError(f"flat class {label!r} contains terrain motion {bad[0]!r}")


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = tuple(ordered or ("motion",))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def method_counts(rows: list[dict], methods: list[tuple[str, str]]) -> list[dict[str, int | str]]:
    """Count successful and failed evaluations for each requested method."""
    counts = []
    for method_label, _method_subdir in methods:
        requested = [row for row in rows if row["method"] == method_label]
        successful = sum(not row["error"] for row in requested)
        counts.append(
            {
                "method": method_label,
                "successful": successful,
                "failed": len(requested) - successful,
                "total": len(requested),
            }
        )
    return counts


def quality_csv_name(method: str) -> str:
    suffix = method.removeprefix("terra_")
    return "quality.csv" if method == "terra" else f"quality_{suffix}.csv"


def method_quality_root(quality_root: Path, method: str) -> Path:
    return quality_root if method == "terra" else quality_root / method


def _publish_detail(result, quality_root: Path, method: str) -> None:
    if result.row["error"]:
        return
    root = method_quality_root(quality_root, method)
    stem = result.row["motion"].replace("/", "__")
    detail = root / "quality" / f"{stem}.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(json.dumps(result.detail, indent=2) + "\n")
    frames = root / "frames" / f"{stem}.npz"
    frames.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frames,
        **result.annotations,
        **frame_metadata(
            motion=result.row["motion"],
            method=method,
            n_frames=int(result.row["frames"]),
        ),
    )


def _worker(job: tuple) -> tuple[dict, dict]:
    (
        method_label,
        method_subdir,
        motion_class,
        motion,
        terrain_method,
        force_flat,
        thresholds,
        cache_root,
        source_root,
        method_subdirs,
        common_interval_override,
        quality_root,
    ) = job
    result = evaluate_method_motion(
        method_label=method_label,
        method_subdir=method_subdir,
        motion_class=motion_class,
        motion=motion,
        terrain_method=terrain_method,
        force_flat=force_flat,
        thresholds=Thresholds(**thresholds),
        cache_root=Path(cache_root),
        source_root=Path(source_root) if source_root is not None else None,
        all_method_subdirs=method_subdirs,
        common_interval_override=common_interval_override,
    )
    _publish_detail(result, Path(quality_root), method_subdir)
    return result.row, result.quality


def _definitions_markdown() -> str:
    lines = [
        "# Authoritative metric registry",
        "",
        "| Key | Family | Formula | Denominator | Source |",
        "|---|---|---|---|---|",
    ]
    for spec in AUTHORITATIVE_METRICS:
        cells = [spec.key, spec.family, spec.formula, spec.denominator, spec.source]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    methods = [parse_assignment(value, "--method") for value in args.method]
    classes = [
        (label, Path(path).expanduser().resolve())
        for label, path in (parse_assignment(value, "--motion-class") for value in args.motion_class)
    ]
    if len({label for label, _ in methods}) != len(methods) or len({subdir for _, subdir in methods}) != len(methods):
        raise SystemExit("method labels and cache subdirectories must be unique")
    class_motions: dict[str, list[str]] = {}
    for label, manifest in classes:
        if not manifest.is_file():
            raise SystemExit(f"motion manifest does not exist: {manifest}")
        motions = read_motion_ids(manifest)
        validate_class_motions(
            label,
            manifest,
            motions,
            allow_flat_name_conflicts=args.allow_flat_name_conflicts,
        )
        class_motions[label] = motions[: args.limit] if args.limit is not None else motions
    thresholds = Thresholds(
        penetration_m=args.penetration_tol,
        support_penetration_m=args.support_penetration_tol,
        foot_contact_height_m=args.foot_contact_height,
        source_contact_speed_m_s=args.source_contact_speed,
        skating_speed_m_s=args.skating_speed,
        terrain_contact_m=args.terrain_contact_tol,
        joint_limit=args.joint_limit_tol,
        joint_limit_linear_m=args.joint_limit_linear_tol,
        tendon_jump=args.tendon_jump_threshold,
        self_collision_m=args.self_collision_tol,
    )
    thresholds.validate()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    cache_root = str(args.cache_root.resolve())
    source_root = str(args.source_root.resolve()) if args.source_root is not None else None
    method_subdirs = tuple(subdir for _label, subdir in methods)
    terrain_method = None if args.terrain_method == "self" else args.terrain_method
    common_intervals = read_common_intervals(args.common_intervals.resolve()) if args.common_intervals else None
    if common_intervals is not None:
        missing_intervals = sorted(
            motion for motions in class_motions.values() for motion in motions if motion not in common_intervals
        )
        if missing_intervals:
            raise SystemExit(
                f"frozen common intervals are missing {len(missing_intervals)} requested motion(s): "
                f"{missing_intervals[0]!r}"
            )
    jobs = [
        (
            method_label,
            method_subdir,
            class_label,
            motion,
            terrain_method,
            manifest_is_flat(class_label, manifest),
            asdict(thresholds),
            cache_root,
            source_root,
            method_subdirs,
            None if common_intervals is None else common_intervals[motion],
            str(args.quality_out),
        )
        for class_label, manifest in classes
        for motion in class_motions[class_label]
        for method_label, method_subdir in methods
    ]
    if not jobs:
        raise SystemExit("no method-motion pairs were requested")
    args.out.mkdir(parents=True, exist_ok=True)
    args.quality_out.mkdir(parents=True, exist_ok=True)
    write_git_commit(args.out)
    if args.quality_out.resolve() != args.out.resolve():
        write_git_commit(args.quality_out)
    started = time.time()
    rows: list[dict] = []
    quality_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(_worker, job): job[:4] for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            row, quality = future.result()
            rows.append(row)
            quality_rows.append(quality)
            status = "ok" if not row["error"] else f"ERROR {row['error']}"
            print(
                f"[{index}/{len(jobs)} {(time.time() - started) / 60:5.1f}m] {row['method']} {row['motion']} {status}",
                flush=True,
            )
    method_order = {label: index for index, (label, _subdir) in enumerate(methods)}
    class_order = {label: index for index, (label, _manifest) in enumerate(classes)}
    rows.sort(key=lambda row: (method_order[row["method"]], class_order[row["motion_class"]], row["motion"]))
    quality_rows.sort(key=lambda row: (method_subdirs.index(row["method"]), row["motion"]))
    _write_csv(args.out / "per_motion.csv", rows, PER_MOTION_FIELDS)
    metric_audit = audit_metric_rows(rows)
    (args.out / "audit.json").write_text(json.dumps(metric_audit, indent=2) + "\n")
    _write_csv(
        args.out / "method_counts.csv",
        method_counts(rows, methods),
        ("method", "successful", "failed", "total"),
    )
    summaries = aggregate(rows, methods, classes)
    write_summary_tables(args.out, summaries)
    write_joint_limit_sensitivity(args.out, aggregate_joint_limit_sensitivity(rows, methods, classes))
    (args.out / "thresholds.csv").write_text(
        "name,value\n"
        + "\n".join(
            [
                *(f"continuous.{key},{value}" for key, value in UNIFIED_THRESHOLDS["continuous"].items()),
                *(f"phase.{key},{value}" for key, value in UNIFIED_THRESHOLDS["phase"].items()),
            ]
        )
        + "\n"
    )
    _write_csv(
        args.out / "timeline_intersections.csv",
        [
            {
                key: row[key]
                for key in ("method", "motion_class", "motion", "common_start_s", "common_end_s", "common_duration_s")
            }
            for row in rows
            if not row["error"]
        ],
    )
    (args.out / "definitions.md").write_text(_definitions_markdown())
    for _label, method in methods:
        selected = [row for row in quality_rows if row["method"] == method]
        path = args.quality_out / quality_csv_name(method)
        _write_csv(path, selected)

    run_metadata = {
        "methods": dict(methods),
        "manifests": {label: str(path) for label, path in classes},
        "motion_selections": class_motions,
        "thresholds": UNIFIED_THRESHOLDS,
        "timeline": UNIFIED_TIMELINE,
        "aggregation": UNIFIED_AGGREGATION,
        "options": {
            "allow_missing": args.allow_missing,
            "allow_flat_name_conflicts": args.allow_flat_name_conflicts,
            "cache_root": cache_root,
            "flat_ground_classes": [label for label, manifest in classes if manifest_is_flat(label, manifest)],
            "limit_per_class": args.limit,
            "quality_out": str(args.quality_out.resolve()),
            "source_root": source_root,
            "terrain_method": args.terrain_method,
            "common_intervals": (None if args.common_intervals is None else str(args.common_intervals.resolve())),
        },
    }
    (args.out / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    errors = [row for row in rows if row["error"]]
    if not metric_audit["passed"]:
        return 2
    return 1 if errors and not args.allow_missing else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="terra evaluate metrics", description=__doc__)
    result.add_argument("--motion-class", action="append", required=True, metavar="LABEL=MANIFEST")
    result.add_argument("--method", action="append", required=True, metavar="LABEL=CACHE_SUBDIR")
    result.add_argument(
        "--terrain-method",
        default="terra",
        help="Cache subdirectory holding shared terrain, or 'self' for each result's own recorded terrain.",
    )
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument(
        "--source-root",
        type=Path,
        help="Explicit source-motion root; analysis metadata source paths remain a fallback.",
    )
    result.add_argument(
        "--common-intervals",
        type=Path,
        help=("CSV with motion/common_start_s/common_end_s fields; require and reuse these frozen scoring windows."),
    )
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--quality-out", type=Path, required=True)
    result.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    result.add_argument("--limit", type=int)
    result.add_argument("--allow-missing", action="store_true")
    result.add_argument(
        "--allow-flat-name-conflicts",
        action="store_true",
        help=(
            "Honor an explicitly flat manifest even when motion names contain terrain tokens; "
            "the override is recorded in run.json."
        ),
    )
    result.add_argument("--penetration-tol", type=float, default=BENCHMARK_THRESHOLDS["penetration_m"])
    result.add_argument(
        "--support-penetration-tol",
        type=float,
        default=BENCHMARK_THRESHOLDS["support_penetration_m"],
    )
    result.add_argument("--foot-contact-height", type=float, default=BENCHMARK_THRESHOLDS["foot_contact_height_m"])
    result.add_argument(
        "--source-contact-speed",
        type=float,
        default=BENCHMARK_THRESHOLDS["source_contact_speed_m_s"],
    )
    result.add_argument("--skating-speed", type=float, default=BENCHMARK_THRESHOLDS["skating_speed_m_s"])
    result.add_argument("--terrain-contact-tol", type=float, default=BENCHMARK_THRESHOLDS["terrain_contact_m"])
    result.add_argument("--joint-limit-tol", type=float, default=BENCHMARK_THRESHOLDS["joint_limit"])
    result.add_argument(
        "--joint-limit-linear-tol",
        type=float,
        default=BENCHMARK_THRESHOLDS["joint_limit_linear_m"],
    )
    result.add_argument("--tendon-jump-threshold", type=float, default=BENCHMARK_THRESHOLDS["tendon_jump"])
    result.add_argument("--self-collision-tol", type=float, default=BENCHMARK_THRESHOLDS["self_collision_m"])
    return result


def metrics_main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


def _command_help() -> str:
    return """usage: terra evaluate COMMAND [ARGS ...]

Evaluation commands:
  metrics ...          score explicit method-motion cohorts with authoritative metrics
  dataset ...          run retargeting metrics using one packaged dataset configuration


Use `terra evaluate COMMAND --help` for command-specific options.
"""


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_command_help(), end="")
        return 0
    command, *remaining = arguments
    if command == "metrics":
        return metrics_main(remaining)
    if command == "dataset":
        from terra.evaluation.dataset import main as dataset_main

        return dataset_main(remaining)
    print(f"terra evaluate: unknown command {command!r}", file=sys.stderr)
    print(_command_help(), file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
