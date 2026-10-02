"""Run the common SMPL-H -> calibrated terrain -> TERRA pipeline for one dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import signal
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from terra._files import atomic_write
from terra._revision import write_git_commit
from terra.datasets.config import resolve_dataset_config

if TYPE_CHECKING:
    from terra.dataset_pipeline import DatasetConfig, MotionRecord

STATUS_FIELDS = (
    "motion",
    "dataset",
    "subject",
    "condition",
    "benchmark_group",
    "terrain_class",
    "expected_family",
    "fit_passed",
    "marker_fit_mean_mm",
    "marker_fit_p95_mm",
    "marker_fit_max_mm",
    "status",
    "frames",
    "frequency",
    "terrain_model",
    "terrain_family_selected",
    "terrain_family_correct",
    "terrain_validation_passed",
    "trajectory_path",
    "analysis_path",
    "terrain_path",
    "elapsed_seconds",
    "error",
)
METHODS = ("terra", "omniretarget", "gmr", "smpl")


def _positive_environment_default(name: str, fallback: int) -> int:
    raw = os.environ.get(name, str(fallback))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def terminal_failure_count(rows: list[dict], *, allow_conversion_failures: bool = False) -> int:
    """Count terminal failures while optionally accepting recorded conversion rejects."""
    accepted = {"ok", "cached"}
    if allow_conversion_failures:
        accepted.add("conversion_failed")
    return sum(row["status"] not in accepted for row in rows)


def read_motion_selection_rows(path: Path) -> list[dict[str, str]]:
    """Read unique selection rows from a CSV or one-ID-per-line TXT file."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"motion selection is empty: {path}")
    header = [field.strip() for field in next(csv.reader([lines[0]]))]
    if header.count("motion") > 1:
        raise ValueError(f"motion selection contains duplicate motion columns: {path}")
    if "motion" in header:
        rows = list(csv.DictReader(lines))
    else:
        rows = [{"motion": line.strip()} for line in lines if line.strip()]
    motions = [str(row.get("motion") or "").strip() for row in rows]
    if not motions or any(not motion for motion in motions):
        raise ValueError(f"motion selection contains an empty ID: {path}")
    if len(motions) != len(set(motions)):
        raise ValueError(f"motion selection contains duplicate IDs: {path}")
    return [row | {"motion": motion} for row, motion in zip(rows, motions, strict=True)]


def read_motion_selection(path: Path) -> list[str]:
    """Read ordered motion IDs from a CSV or one-ID-per-line TXT file."""
    return [row["motion"] for row in read_motion_selection_rows(path)]


def _selection_path(value: Path, config: DatasetConfig) -> Path:
    """Resolve an explicit selection file against the input storage roots."""

    resolved = config.storage_roots.resolve_input(value, base=Path.cwd())
    assert resolved is not None
    return resolved


def _worker(config: DatasetConfig, record: MotionRecord, overwrite: bool) -> dict:
    from terra.dataset_pipeline import run_motion

    try:
        return run_motion(config, record, overwrite=overwrite)
    except Exception as exc:  # keep one bad motion from discarding the dataset yield
        return {
            "motion": record.motion,
            "dataset": config.name,
            "subject": record.subject,
            "condition": record.condition,
            "benchmark_group": record.metadata.get("benchmark_group", ""),
            "terrain_class": record.terrain_class,
            "expected_family": record.expected_family,
            "fit_passed": record.fit_passed,
            "status": "failed",
            "elapsed_seconds": 0.0,
            "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000],
            "traceback": traceback.format_exc(),
        }


def _failed_worker_result(
    config: DatasetConfig,
    record: MotionRecord,
    error: str,
    *,
    elapsed_seconds: float = 0.0,
) -> dict:
    """Build the common status row for a worker that failed outside Python."""
    return {
        "motion": record.motion,
        "dataset": config.name,
        "subject": record.subject,
        "condition": record.condition,
        "benchmark_group": record.metadata.get("benchmark_group", ""),
        "terrain_class": record.terrain_class,
        "expected_family": record.expected_family,
        "fit_passed": record.fit_passed,
        "status": "failed",
        "elapsed_seconds": elapsed_seconds,
        "error": error.replace("\n", " ")[:1000],
    }


def _isolated_worker_entry(connection, config, record, overwrite, worker) -> None:
    """Send one worker result to its parent from a disposable process."""
    try:
        connection.send(worker(config, record, overwrite))
    finally:
        connection.close()


def _worker_exit_error(exitcode: int | None) -> str:
    """Describe a subprocess exit without hiding a native signal such as SIGABRT."""
    if exitcode is None:
        return "isolated SMPL worker ended without reporting an exit status"
    if exitcode < 0:
        number = -exitcode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = f"signal {number}"
        return f"isolated SMPL worker terminated by {name} (signal {number})"
    return f"isolated SMPL worker exited with code {exitcode}"


def _run_worker_isolated(
    config: DatasetConfig,
    record: MotionRecord,
    overwrite: bool,
    *,
    worker=_worker,
    mp_context=None,
) -> dict:
    """Run one motion in a fresh process so a native abort stays motion-local.

    The upstream SMPL-fit baseline enters MuJoCo native code once per frame. A native
    abort cannot be converted into a Python exception, so neither ``try/except`` nor a
    one-worker executor protects the dataset driver. A disposable process is the only
    boundary that can preserve the baseline and let the parent record the failure.
    """
    context = mp_context or multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_worker_entry,
        args=(send, config, record, overwrite, worker),
    )
    started = time.monotonic()
    result = None
    process.start()
    send.close()
    try:
        # Read before joining: a sufficiently large traceback can fill the pipe and make
        # a child wait for its parent while the parent waits for the child.
        while process.is_alive():
            if receive.poll(0.1):
                try:
                    result = receive.recv()
                except EOFError:
                    pass
                break
        process.join()
        if result is None and receive.poll():
            try:
                result = receive.recv()
            except EOFError:
                pass
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
            process.join()

    elapsed = time.monotonic() - started
    exitcode = process.exitcode
    process.close()
    if exitcode != 0:
        return _failed_worker_result(
            config,
            record,
            _worker_exit_error(exitcode),
            elapsed_seconds=elapsed,
        )
    if result is None:
        return _failed_worker_result(
            config,
            record,
            "isolated SMPL worker exited cleanly without returning a result",
            elapsed_seconds=elapsed,
        )
    return result


def _smpl_needs_isolation(
    config: DatasetConfig,
    record: MotionRecord,
    overwrite: bool,
) -> bool:
    """Return whether this row can enter the native SMPL optimizer."""
    from terra.artifacts import retarget_cache_paths

    if config.method != "smpl" or not record.fit_passed:
        return False
    paths = retarget_cache_paths(
        config.cache_root,
        record.motion,
        method=config.method,
        env_name=config.env_name,
    )
    return overwrite or not (paths.trajectory_path.is_file() and paths.analysis_path.is_file())


def _run_record(config: DatasetConfig, record: MotionRecord, overwrite: bool) -> dict:
    """Run a record with a native-crash boundary only where the SMPL fit needs it."""
    if _smpl_needs_isolation(config, record, overwrite):
        return _run_worker_isolated(config, record, overwrite)
    return _worker(config, record, overwrite)


def _write_status(path: Path, records: list[MotionRecord], completed: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            if record.motion in completed:
                writer.writerow(completed[record.motion])
    temporary.replace(path)


def _dry_run(config: DatasetConfig, records: list[MotionRecord]) -> None:
    missing_sources = [record.motion for record in records if not record.source_path.is_file()]
    missing_calibrations = [
        record.motion
        for record in records
        if record.calibration_path is not None and not record.calibration_path.is_file()
    ]
    missing_terrain_records = (
        []
        if config.terrain_source_dir is None
        else [
            record.motion
            for record in records
            if not (config.terrain_source_dir / f"{record.motion.replace('/', '__')}.json").is_file()
        ]
    )
    payload = {
        "dataset": config.name,
        "config": str(config.path),
        "input_root": str(config.input_root),
        "manifest": str(config.manifest_path or ""),
        "method": config.method,
        "terrain_fit": config.terrain_fit,
        "retarget_overrides": config.retarget_overrides,
        "method_overrides": config.method_overrides,
        "terrain": {
            "mode": config.terrain_mode,
            "calibration": config.calibration_mode,
            "contact_source": "kinematic",
            "contact_joints": list(config.contact_joints),
            "fit": config.terrain_fit,
            "source_dir": str(config.terrain_source_dir or ""),
            "source_method": config.terrain_source_method or "",
        },
        "cache_root": str(config.cache_root),
        "reference_cache_root": str(config.reference_cache_root),
        "run_root": str(config.run_root),
        "storage_roots": config.storage_roots.as_dict(),
        "motions": len(records),
        "conversion_failures": sum(not record.fit_passed for record in records),
        "missing_source_files": len(missing_sources),
        "missing_calibration_files": len(missing_calibrations),
        "missing_terrain_records": len(missing_terrain_records),
        "missing_source_preview": missing_sources[:5],
        "missing_calibration_preview": missing_calibrations[:5],
        "missing_terrain_preview": missing_terrain_records[:5],
        "preview": [
            {
                "motion": record.motion,
                "source": str(record.source_path),
                "calibration": str(record.calibration_path or ""),
                "fit_passed": record.fit_passed,
                "terrain_class": record.terrain_class,
                "expected_family": record.expected_family,
            }
            for record in records[:5]
        ],
    }
    print(json.dumps(payload, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="terra run",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Bundled dataset name or an explicit TOML path",
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=("CSV with a motion column or TXT with one motion ID per line."),
    )
    parser.add_argument(
        "--all-motions",
        action="store_true",
        help="Explicitly process every discovered motion when the dataset has no conversion manifest",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Exact solver input rate for the GMR or MM-SMPL solve-rate experiment.",
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--terrain-contact-margin",
        type=float,
        nargs=2,
        metavar=("LONGITUDINAL_M", "LATERAL_M"),
        help="Override the terrain fitter's required support margins.",
    )
    parser.add_argument(
        "--terrain-max-extension",
        type=float,
        nargs=2,
        metavar=("LONGITUDINAL_M", "LATERAL_M"),
        help="Override optional box growth beyond required support.",
    )
    parser.add_argument(
        "--terrain-dir",
        type=Path,
        help="Directory of per-motion reconstruction records to use instead of fitting terrain.",
    )
    parser.add_argument(
        "--terrain-method",
        help="Expected reconstruction method in --terrain-dir (terra).",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        help="Retargeting method; baselines require completed TERRA artifacts for the same cohort",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=_positive_environment_default("TERRA_THREADS_PER_WORKER", 1),
        help="BLAS/OpenMP/XLA CPU threads available to each worker",
    )
    parser.add_argument("--motion", action="append", help="Run only this motion; repeatable")
    parser.add_argument(
        "--terrain-class",
        action="append",
        help=(
            "Run only converted rows with this exact terrain_class; repeatable. "
            "Explicitly selected calibration rows are promoted to retarget inputs."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-conversion-failures",
        action="store_true",
        help="Return success when every failure is a pre-existing conversion rejection",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.threads_per_worker < 1:
        raise SystemExit("--threads-per-worker must be positive")
    os.environ["TERRA_RUN_WORKERS"] = str(args.workers)
    os.environ["TERRA_THREADS_PER_WORKER"] = str(args.threads_per_worker)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads_per_worker)
    eigen_threads = "false" if args.threads_per_worker == 1 else "true"
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen={eigen_threads} intra_op_parallelism_threads={args.threads_per_worker}"
    )

    from terra.dataset_pipeline import ensure_robot_shape, load_dataset_config, load_motion_records

    config = load_dataset_config(resolve_dataset_config(args.config))
    replacements = {}
    for name in ("input_root", "manifest_path", "cache_root", "run_root"):
        cli_name = "manifest" if name == "manifest_path" else name
        value = getattr(args, cli_name)
        if value is not None:
            resolver = (
                config.storage_roots.resolve_input
                if name in {"input_root", "manifest_path"}
                else config.storage_roots.resolve_artifact
            )
            replacements[name] = resolver(value, base=Path.cwd())
    if "cache_root" in replacements and config.reference_cache_root == config.cache_root:
        replacements["reference_cache_root"] = replacements["cache_root"]
    if args.method is not None:
        replacements["method"] = args.method
        if args.run_root is None and args.method != config.method:
            replacements["run_root"] = config.run_root / args.method
    selected_method = str(replacements.get("method", config.method))
    if args.terrain_dir is not None:
        if selected_method != "terra":
            raise SystemExit("--terrain-dir is supported only with --method terra")
        source_method = args.terrain_method or config.terrain_source_method
        if not source_method:
            raise SystemExit("--terrain-method is required with --terrain-dir")
        replacements.update(
            terrain_mode="precomputed",
            terrain_source_dir=config.storage_roots.resolve_input(args.terrain_dir, base=Path.cwd()),
            terrain_source_method=source_method.strip().casefold(),
        )
    elif args.terrain_method is not None:
        if config.terrain_source_dir is None:
            raise SystemExit("--terrain-method requires --terrain-dir or a configured [terrain].source_dir")
        replacements["terrain_source_method"] = args.terrain_method.strip().casefold()
    if args.target_fps is not None:
        if not math.isfinite(args.target_fps) or args.target_fps <= 0:
            raise SystemExit("--target-fps must be positive and finite")
        if selected_method not in {"gmr", "smpl"}:
            raise SystemExit("--target-fps is supported only for --method gmr or --method smpl")
        method_overrides = dict(config.method_overrides)
        method_overrides["target_fps"] = float(args.target_fps)
        if selected_method == "gmr":
            method_overrides["exact_target_fps"] = True
        replacements["method_overrides"] = method_overrides
    terrain_fit = dict(config.terrain_fit)
    if args.terrain_contact_margin is not None:
        terrain_fit["contact_margin"] = tuple(args.terrain_contact_margin)
    if args.terrain_max_extension is not None:
        terrain_fit["max_extension"] = tuple(args.terrain_max_extension)
    if terrain_fit != config.terrain_fit:
        replacements["terrain_fit"] = terrain_fit
    for name, value in (
        ("--terrain-contact-margin", args.terrain_contact_margin),
        ("--terrain-max-extension", args.terrain_max_extension),
    ):
        if value is not None and any(not math.isfinite(component) or component < 0 for component in value):
            raise SystemExit(f"{name} values must be finite and non-negative")
    if replacements:
        config = replace(config, **replacements)
    if config.terrain_mode == "precomputed" and (
        args.terrain_contact_margin is not None or args.terrain_max_extension is not None
    ):
        raise SystemExit("terrain-fitting overrides cannot be used with precomputed terrain")

    if args.selection_manifest is not None and args.all_motions:
        raise SystemExit("--selection-manifest and --all-motions are mutually exclusive")
    if config.manifest_path is None and args.selection_manifest is None and not args.motion and not args.all_motions:
        raise SystemExit(
            f"{config.name} discovers motions directly; pass --selection-manifest, --motion, "
            "or --all-motions explicitly"
        )

    selection_path = None
    selection_rows = None
    if args.selection_manifest is not None:
        selection_path = _selection_path(args.selection_manifest, config)
        selection_rows = read_motion_selection_rows(selection_path)

    records = load_motion_records(
        config,
        include_non_retarget=selection_rows is not None or bool(args.terrain_class),
        selected_motions=(
            [row["motion"] for row in selection_rows]
            if selection_rows is not None and config.manifest_path is None
            else None
        ),
    )
    if selection_rows is not None:
        requested_order = [row["motion"] for row in selection_rows]
        by_motion = {record.motion: record for record in records}
        missing = [motion for motion in requested_order if motion not in by_motion]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise SystemExit(
                f"selection manifest contains {len(missing)} motion(s) absent from the "
                f"converted dataset: {preview}{suffix}"
            )
        records = []
        for selection in selection_rows:
            record = by_motion[selection["motion"]]
            records.append(
                replace(
                    record,
                    subject=selection.get("subject") or record.subject,
                    condition=selection.get("condition") or record.condition,
                    terrain_class=(selection.get("terrain_class") or record.terrain_class),
                    expected_family=(selection.get("expected_family") or record.expected_family),
                    metadata=record.metadata | selection,
                )
            )
    if args.motion:
        requested = set(args.motion)
        known = {record.motion for record in records}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"motion(s) absent from converted manifest: {', '.join(missing)}")
        records = [record for record in records if record.motion in requested]
    if args.terrain_class:
        requested_classes = {value.strip().casefold() for value in args.terrain_class}
        if "" in requested_classes:
            raise SystemExit("--terrain-class values must be non-empty")
        records = [record for record in records if record.terrain_class.strip().casefold() in requested_classes]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"no converted motions selected for {config.name}")
    if args.dry_run:
        _dry_run(config, records)
        return 0

    config.run_root.mkdir(parents=True, exist_ok=True)
    write_git_commit(config.run_root, repo_root=config.repo_root)
    write_git_commit(config.cache_root, repo_root=config.repo_root)
    ensure_robot_shape(config)
    method_assets: dict[str, dict[str, str]] = {}
    if config.method == "gmr":
        from terra.baselines.gmr import prepare_fitted_shape

        fitted_shape = prepare_fitted_shape(
            config.env_name,
            config.cache_root,
            config.smpl_model_path,
        )
        fitted_shape_metadata = fitted_shape.with_name(fitted_shape.name.replace("_shape.pkl", "_shape_metadata.json"))
        method_assets["gmr_fitted_shape"] = {
            "path": str(fitted_shape),
            "metadata_path": str(fitted_shape_metadata),
        }
        print(f"GMR fitted shape ready: {fitted_shape}", flush=True)
    completed: dict[str, dict] = {}
    status_path = config.run_root / "status.csv"
    jobs = [record for record in records if record.fit_passed]
    for record in records:
        if not record.fit_passed:
            completed[record.motion] = _worker(config, record, args.overwrite)
    _write_status(status_path, records, completed)
    if config.method == "smpl":
        print(
            "SMPL native isolation enabled: each uncached motion uses a disposable "
            f"process; concurrency={args.workers}",
            flush=True,
        )

    if args.workers == 1:
        for index, record in enumerate(jobs, 1):
            result = _run_record(config, record, args.overwrite)
            completed[record.motion] = result
            _write_status(status_path, records, completed)
            print(f"[{index}/{len(jobs)}] {result['status']} {record.motion}", flush=True)
    else:
        # SMPL already creates one disposable process per uncached motion. Threads only
        # coordinate those subprocesses here; another process pool would reintroduce a
        # shared failure domain and turn one native abort into BrokenProcessPool for the
        # whole cohort.
        if config.method == "smpl":
            pool = ThreadPoolExecutor(max_workers=args.workers)
        else:
            context = multiprocessing.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
        with pool:
            futures = {pool.submit(_run_record, config, record, args.overwrite): record for record in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failed_worker_result(
                        config,
                        record,
                        f"worker {type(exc).__name__}: {exc}",
                    )
                completed[record.motion] = result
                _write_status(status_path, records, completed)
                print(f"[{index}/{len(jobs)}] {result['status']} {record.motion}", flush=True)

    ordered = [completed[record.motion] for record in records]
    successful = [row for row in ordered if row["status"] in {"ok", "cached"}]
    manifest_path = config.run_root / "manifest.csv"

    def write_manifest(path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "motion",
                    "dataset",
                    "benchmark_group",
                    "terrain_class",
                    "passed",
                    "expected_family",
                    "marker_fit_mean_mm",
                    "marker_fit_p95_mm",
                    "marker_fit_max_mm",
                ),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(row | {"passed": 1} for row in successful)

    atomic_write(manifest_path, write_manifest)
    summary = {
        "dataset": config.name,
        "config": str(config.path),
        "input_root": str(config.input_root),
        "manifest": str(config.manifest_path or ""),
        "cache_root": str(config.cache_root),
        "reference_cache_root": str(config.reference_cache_root),
        "run_root": str(config.run_root),
        "env_name": config.env_name,
        "output_manifest": str(manifest_path),
        "status": str(status_path),
        "storage_roots": config.storage_roots.as_dict(),
        "method": config.method,
        "terrain_fit": config.terrain_fit,
        "terrain_source_dir": str(config.terrain_source_dir or ""),
        "terrain_source_method": config.terrain_source_method or "",
        "retarget_overrides": config.retarget_overrides,
        "method_overrides": config.method_overrides,
        "method_assets": method_assets,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "motions": len(records),
        "complete_dataset": (
            args.selection_manifest is None
            and not args.motion
            and not args.terrain_class
            and args.limit is None
            and (config.manifest_path is not None or args.all_motions)
        ),
        "selection_manifest": (str(selection_path) if selection_path is not None else ""),
        "all_motions": args.all_motions,
        "motion_filter": args.motion or [],
        "terrain_class_filter": args.terrain_class or [],
        "limit": args.limit,
        "allow_conversion_failures": args.allow_conversion_failures,
        "status_counts": {
            status: sum(row["status"] == status for row in ordered)
            for status in sorted({row["status"] for row in ordered})
        },
        "records": [asdict(record) | {"source_path": str(record.source_path)} for record in records],
    }
    run_path = config.run_root / "run.json"
    rendered_summary = json.dumps(summary, indent=2, default=str) + "\n"
    atomic_write(run_path, lambda path: path.write_text(rendered_summary, encoding="utf-8"))
    failures = terminal_failure_count(
        ordered,
        allow_conversion_failures=args.allow_conversion_failures,
    )
    successful_count = sum(row["status"] in {"ok", "cached"} for row in ordered)
    print(
        f"{config.name}: {successful_count}/{len(records)} succeeded, "
        f"{sum(row['status'] == 'conversion_failed' for row in ordered)} conversion rejected "
        f"-> {config.run_root}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
