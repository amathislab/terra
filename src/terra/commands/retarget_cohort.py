"""Run one retargeting method over a versioned, partitioned motion cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from terra._files import atomic_write
from terra._methods import SUPPORTED_METHODS
from terra._revision import write_git_commit
from terra.paths import StorageRoots

BASELINE_METHODS = tuple(method for method in SUPPORTED_METHODS if method != "terra")
SPEC_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RetargetCohortPartition:
    """One dataset-policy and artifact-cache boundary in a retargeting cohort."""

    key: str
    dataset: str
    config: Path
    selection: Path
    source_cache_root: str
    motions: int


@dataclass(frozen=True, slots=True)
class RetargetCohortSpec:
    """Validated immutable inputs for a multi-dataset retargeting comparison."""

    path: Path
    name: str
    source_selection: Path
    source_selection_sha256: str
    output_root: str
    methods: tuple[str, ...]
    partitions: tuple[RetargetCohortPartition, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(base: Path, value: object, label: str, *, suffix: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be a relative path: {value!r}")
    path = (base / candidate).resolve()
    if path.suffix.casefold() != suffix or not path.is_file():
        raise ValueError(f"{label} is not a readable {suffix} file: {path}")
    return path


def _csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if "motion" not in fields or not rows:
        raise ValueError(f"cohort selection is empty or has no motion column: {path}")
    motions = [row.get("motion", "").strip() for row in rows]
    if any(not motion for motion in motions) or len(motions) != len(set(motions)):
        raise ValueError(f"cohort selection contains empty or duplicate motions: {path}")
    return fields, rows


def _table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _config_identity(path: Path) -> tuple[str, str]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    dataset = _text(_table(payload.get("dataset"), f"{path} [dataset]").get("name"), "dataset.name")
    cache = _text(
        _table(payload.get("retarget"), f"{path} [retarget]").get("cache_root"),
        "retarget.cache_root",
    )
    return dataset, cache


def _cache_value_matches(value: str, absolute: str) -> bool:
    cache = Path(value)
    source = Path(absolute)
    if cache.is_absolute():
        return cache.resolve() == source.resolve()
    parts = cache.parts[1:] if cache.parts and cache.parts[0] == "runs" else cache.parts
    return bool(parts) and source.parts[-len(parts) :] == parts


def load_retarget_cohort_spec(path: str | Path) -> RetargetCohortSpec:
    """Load a cohort spec and prove that its partitions exactly cover its source rows."""

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    with spec_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(f"{spec_path} must declare schema_version = {SPEC_SCHEMA_VERSION}")
    base = spec_path.parent
    name = _text(payload.get("name"), "name")
    output_root = _text(payload.get("output_root"), "output_root")
    if Path(output_root).is_absolute() or not output_root.startswith("runs/"):
        raise ValueError("output_root must be a portable runs/... artifact path")
    source_selection = _relative_file(
        base,
        payload.get("source_selection"),
        "source_selection",
        suffix=".csv",
    )
    expected_source_hash = _text(payload.get("source_selection_sha256"), "source_selection_sha256")
    observed_source_hash = _sha256(source_selection)
    if expected_source_hash != observed_source_hash:
        raise ValueError(
            f"source selection digest mismatch: expected {expected_source_hash}, got {observed_source_hash}"
        )
    methods_value = payload.get("methods")
    if not isinstance(methods_value, list) or not methods_value:
        raise ValueError("methods must be a non-empty TOML array")
    methods = tuple(str(value) for value in methods_value)
    if len(methods) != len(set(methods)) or any(method not in BASELINE_METHODS for method in methods):
        raise ValueError(f"methods must be unique baseline methods chosen from {BASELINE_METHODS}")

    source_fields, source_rows = _csv_rows(source_selection)
    required_source_fields = {"motion", "dataset", "source_cache_root"}
    if not required_source_fields <= set(source_fields):
        raise ValueError(
            f"source selection is missing columns: {', '.join(sorted(required_source_fields - set(source_fields)))}"
        )
    source_by_root: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        root = row["source_cache_root"].strip()
        if not root:
            raise ValueError(f"source selection has an empty cache root for {row['motion']!r}")
        source_by_root.setdefault(root, []).append(row)

    partition_values = payload.get("partition")
    if not isinstance(partition_values, list) or not partition_values:
        raise ValueError("at least one [[partition]] table is required")
    partitions: list[RetargetCohortPartition] = []
    used_roots: set[str] = set()
    keys: set[str] = set()
    covered_motions: set[str] = set()
    for index, raw_partition in enumerate(partition_values):
        partition = _table(raw_partition, f"partition {index}")
        key = _text(partition.get("key"), f"partition {index}.key")
        if key in keys:
            raise ValueError(f"duplicate partition key: {key!r}")
        keys.add(key)
        dataset = _text(partition.get("dataset"), f"partition {key}.dataset")
        config = _relative_file(base, partition.get("config"), f"partition {key}.config", suffix=".toml")
        selection = _relative_file(
            base,
            partition.get("selection"),
            f"partition {key}.selection",
            suffix=".csv",
        )
        source_cache_root = _text(
            partition.get("source_cache_root"),
            f"partition {key}.source_cache_root",
        )
        if not Path(source_cache_root).is_absolute() and not source_cache_root.startswith("runs/"):
            raise ValueError(f"partition {key} source_cache_root must be absolute or a runs/... alias")
        if source_cache_root in used_roots:
            raise ValueError(f"source cache root appears in more than one partition: {source_cache_root}")
        used_roots.add(source_cache_root)
        motions = partition.get("motions")
        if not isinstance(motions, int) or isinstance(motions, bool) or motions < 1:
            raise ValueError(f"partition {key}.motions must be a positive integer")
        selection_fields, selection_rows = _csv_rows(selection)
        if selection_fields != source_fields:
            raise ValueError(f"partition {key} columns differ from the source selection")
        expected_rows = source_by_root.get(source_cache_root)
        if expected_rows is None:
            raise ValueError(f"partition {key} cache root is absent from the source selection")
        if selection_rows != expected_rows:
            raise ValueError(f"partition {key} is not the exact ordered source-cache subset")
        if len(selection_rows) != motions:
            raise ValueError(f"partition {key} expected {motions} motions but contains {len(selection_rows)}")
        if {row["dataset"] for row in selection_rows} != {dataset}:
            raise ValueError(f"partition {key} has a dataset mismatch")
        config_dataset, config_cache = _config_identity(config)
        if config_dataset != dataset:
            raise ValueError(f"partition {key} config dataset is {config_dataset!r}, expected {dataset!r}")
        if not _cache_value_matches(config_cache, source_cache_root):
            raise ValueError(f"partition {key} config cache {config_cache!r} does not match {source_cache_root!r}")
        covered_motions.update(row["motion"] for row in selection_rows)
        partitions.append(
            RetargetCohortPartition(
                key=key,
                dataset=dataset,
                config=config,
                selection=selection,
                source_cache_root=source_cache_root,
                motions=motions,
            )
        )

    if used_roots != set(source_by_root):
        missing = sorted(set(source_by_root) - used_roots)
        extra = sorted(used_roots - set(source_by_root))
        raise ValueError(f"partitions do not exactly cover source cache roots; missing={missing}, extra={extra}")
    if len(covered_motions) != len(source_rows):
        raise ValueError("partitions do not cover every source motion exactly once")
    return RetargetCohortSpec(
        path=spec_path,
        name=name,
        source_selection=source_selection,
        source_selection_sha256=observed_source_hash,
        output_root=output_root,
        methods=methods,
        partitions=tuple(partitions),
    )


def _record_path(
    spec: RetargetCohortSpec,
    method: str,
    roots: StorageRoots,
    output_root: Path | None = None,
) -> Path:
    output_root = output_root or roots.resolve_artifact(spec.output_root)
    assert output_root is not None
    return output_root / "cohorts" / method / "run.json"


def _publish_record(
    path: Path,
    spec: RetargetCohortSpec,
    method: str,
    workers: int,
    threads_per_worker: int,
    results: list[dict[str, object]],
    *,
    status: str,
    target_fps: float | None,
    cache_root: Path | None,
) -> None:
    payload = {
        "schema": "terra.retarget-cohort-run.v1",
        "name": spec.name,
        "spec": str(spec.path),
        "source_selection": str(spec.source_selection),
        "source_selection_sha256": spec.source_selection_sha256,
        "method": method,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "target_fps": target_fps,
        "cache_root": None if cache_root is None else str(cache_root),
        "motions": sum(partition.motions for partition in spec.partitions),
        "status": status,
        "partitions": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2) + "\n"
    atomic_write(path, lambda temporary: temporary.write_text(rendered, encoding="utf-8"))
    write_git_commit(path.parent)


def run_retarget_cohort(
    spec: RetargetCohortSpec,
    method: str,
    *,
    workers: int,
    threads_per_worker: int,
    target_fps: float | None = None,
    cache_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    """Execute all partitions, continuing after a partition records motion failures."""

    if method not in spec.methods:
        raise ValueError(f"method {method!r} is not enabled by {spec.path}; choose one of {spec.methods}")
    if workers < 1 or threads_per_worker < 1:
        raise ValueError("workers and threads-per-worker must be positive")
    if target_fps is not None and (target_fps <= 0 or not math.isfinite(target_fps)):
        raise ValueError("target_fps must be positive and finite")
    if target_fps is not None and method not in {"gmr", "smpl"}:
        raise ValueError("target_fps is supported only for GMR and MM-SMPL")
    from terra.commands.run import main as run_dataset
    from terra.dataset_pipeline import load_dataset_config

    roots = StorageRoots.from_environment(Path.cwd())
    experiment_cache_root = roots.resolve_artifact(cache_root, base=Path.cwd()) if cache_root is not None else None
    experiment_output_root = roots.resolve_artifact(output_root, base=Path.cwd()) if output_root is not None else None
    record_path = _record_path(spec, method, roots, experiment_output_root)
    results: list[dict[str, object]] = []
    for index, partition in enumerate(spec.partitions, 1):
        config = load_dataset_config(partition.config, storage_roots=roots)
        method_run_root = (
            experiment_output_root / "partitions" / partition.key / method
            if experiment_output_root is not None
            else config.run_root / method
        )
        print(
            f"[{index}/{len(spec.partitions)}] {method} {partition.key}: {partition.motions} motions",
            flush=True,
        )
        arguments = [
            str(partition.config),
            "--selection-manifest",
            str(partition.selection),
            "--method",
            method,
            "--workers",
            str(workers),
            "--threads-per-worker",
            str(threads_per_worker),
        ]
        if experiment_cache_root is not None:
            arguments.extend(("--cache-root", str(experiment_cache_root)))
        if experiment_output_root is not None:
            arguments.extend(("--run-root", str(method_run_root)))
        if target_fps is not None:
            arguments.extend(("--target-fps", str(target_fps)))
        if overwrite:
            arguments.append("--overwrite")
        if dry_run:
            arguments.append("--dry-run")
        exit_code = run_dataset(arguments)
        results.append(
            {
                "key": partition.key,
                "dataset": partition.dataset,
                "motions": partition.motions,
                "selection": str(partition.selection),
                "cache_root": str(experiment_cache_root or config.cache_root),
                "reference_cache_root": str(config.cache_root),
                "run_root": str(method_run_root),
                "manifest": str(method_run_root / "manifest.csv"),
                "status_table": str(method_run_root / "status.csv"),
                "exit_code": exit_code,
            }
        )
        if not dry_run:
            _publish_record(
                record_path,
                spec,
                method,
                workers,
                threads_per_worker,
                results,
                status="running",
                target_fps=target_fps,
                cache_root=experiment_cache_root,
            )
    exit_code = 0 if all(result["exit_code"] == 0 for result in results) else 2
    if not dry_run:
        _publish_record(
            record_path,
            spec,
            method,
            workers,
            threads_per_worker,
            results,
            status="complete" if exit_code == 0 else "completed_with_failures",
            target_fps=target_fps,
            cache_root=experiment_cache_root,
        )
        print(f"Cohort record: {record_path}", flush=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="terra retarget cohort",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("spec", type=Path, help="versioned retargeting-cohort TOML")
    parser.add_argument("--method", required=True, choices=BASELINE_METHODS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Solve every selected GMR or MM-SMPL motion on this exact temporal grid.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Separate artifact cache for an experimental variant.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Separate status/provenance root for an experimental variant.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the complete spec and every dataset input without publishing outputs",
    )
    args = parser.parse_args(argv)
    try:
        if args.target_fps is not None and (args.cache_root is None or args.output_root is None):
            parser.error("--target-fps requires --cache-root and --output-root to isolate the experiment")
        spec = load_retarget_cohort_spec(args.spec)
        return run_retarget_cohort(
            spec,
            args.method,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            target_fps=args.target_fps,
            cache_root=args.cache_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2


__all__ = [
    "BASELINE_METHODS",
    "RetargetCohortPartition",
    "RetargetCohortSpec",
    "load_retarget_cohort_spec",
    "main",
    "run_retarget_cohort",
]
