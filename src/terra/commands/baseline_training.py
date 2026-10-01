"""Build matched, temporally bounded policy selections for retargeting baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from terra._files import atomic_write
from terra._methods import RetargetingMethod, validate_method
from terra._revision import write_git_commit
from terra.artifacts import retarget_cache_paths
from terra.commands.materialize import read_manifest
from terra.paths import StorageRoots
from terra.training_segments import (
    SEGMENT_FIELDS,
    SEGMENT_POLICY,
    expand_training_segments,
    publish_segmented_selection,
    selection_segment,
)

_SUCCESS_STATUSES = frozenset(("ok", "cached"))
_STATUS_VALUES = _SUCCESS_STATUSES | {"failed"}


@dataclass(frozen=True, slots=True)
class _CohortCoverage:
    method: RetargetingMethod
    record_path: Path
    successful: frozenset[str]
    failed: frozenset[str]
    cache_roots: Mapping[str, Path]
    target_fps: float | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError as error:
        raise FileNotFoundError(f"cohort table is unavailable: {path}") from error
    if reader.fieldnames is None or "motion" not in reader.fieldnames:
        raise ValueError(f"cohort table has no motion column: {path}")
    if not rows and not allow_empty:
        raise ValueError(f"cohort table is empty: {path}")
    motions = [row.get("motion", "").strip() for row in rows]
    if any(not motion for motion in motions) or len(motions) != len(set(motions)):
        raise ValueError(f"cohort table contains empty or duplicate motion IDs: {path}")
    return rows


def _record_path(value: object, *, record: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"cohort record {record} has no {field}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else record.parent / path).resolve()


def _load_cohort_coverage(
    record_path: Path,
    *,
    source_selection_sha256: str,
    source_motions: frozenset[str],
) -> _CohortCoverage:
    record = record_path.expanduser().resolve()
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid retargeting cohort record {record}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != "terra.retarget-cohort-run.v1":
        raise ValueError(f"unsupported retargeting cohort record: {record}")
    method = validate_method(str(payload.get("method", "")))
    if method == "terra":
        raise ValueError(f"baseline cohort record cannot use method='terra': {record}")
    if payload.get("status") not in {"complete", "completed_with_failures"}:
        raise ValueError(f"retargeting cohort is not terminal: {record}")
    raw_target_fps = payload.get("target_fps")
    if raw_target_fps is None:
        target_fps = None
    elif isinstance(raw_target_fps, bool) or not isinstance(raw_target_fps, int | float):
        raise ValueError(f"retargeting cohort has invalid target_fps: {record}")
    else:
        target_fps = float(raw_target_fps)
        if target_fps <= 0 or not math.isfinite(target_fps):
            raise ValueError(f"retargeting cohort has invalid target_fps: {record}")
    if payload.get("source_selection_sha256") != source_selection_sha256:
        raise ValueError(f"retargeting cohort source selection digest does not match: {record}")
    partitions = payload.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError(f"retargeting cohort has no completed partitions: {record}")

    successful: set[str] = set()
    failed: set[str] = set()
    cache_roots: dict[str, Path] = {}
    for partition in partitions:
        if not isinstance(partition, Mapping):
            raise ValueError(f"retargeting cohort has an invalid partition record: {record}")
        expected_count = partition.get("motions")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"retargeting cohort partition has invalid motion count: {record}")
        selection_path = _record_path(partition.get("selection"), record=record, field="selection")
        status_path = _record_path(partition.get("status_table"), record=record, field="status_table")
        manifest_path = _record_path(partition.get("manifest"), record=record, field="manifest")
        cache_root = _record_path(partition.get("cache_root"), record=record, field="cache_root")
        selection_rows = _csv_rows(selection_path)
        status_rows = _csv_rows(status_path)
        manifest_rows = _csv_rows(manifest_path, allow_empty=True)
        selected = {row["motion"] for row in selection_rows}
        status_ids = {row["motion"] for row in status_rows}
        manifest_ids = {row["motion"] for row in manifest_rows}
        statuses = {row.get("status", "").strip() for row in status_rows}
        partition_success = {row["motion"] for row in status_rows if row.get("status", "").strip() in _SUCCESS_STATUSES}
        partition_failed = {row["motion"] for row in status_rows if row.get("status", "").strip() == "failed"}
        if (
            len(selected) != expected_count
            or status_ids != selected
            or not statuses <= _STATUS_VALUES
            or manifest_ids != partition_success
            or partition_success & partition_failed
            or partition_success | partition_failed != selected
        ):
            raise ValueError(f"retargeting cohort partition coverage is inconsistent: {status_path}")
        overlap = (successful | failed) & selected
        if overlap:
            raise ValueError(f"motion occurs in multiple retargeting cohort partitions: {min(overlap)!r}")
        successful.update(partition_success)
        failed.update(partition_failed)
        cache_roots.update(dict.fromkeys(selected, cache_root))

    covered = successful | failed
    if covered != source_motions or payload.get("motions") != len(source_motions):
        raise ValueError(f"retargeting cohort does not exactly cover the source selection: {record}")
    return _CohortCoverage(
        method=method,
        record_path=record,
        successful=frozenset(successful),
        failed=frozenset(failed),
        cache_roots=cache_roots,
        target_fps=target_fps,
    )


def _rebound_row(
    source_row: Mapping[str, str],
    *,
    method: RetargetingMethod,
    cache_root: Path,
    allow_isolated_cache_root: bool,
) -> dict[str, str]:
    motion = source_row["motion"]
    declared_root = Path(source_row["source_cache_root"]).expanduser().resolve()
    if declared_root != cache_root and not allow_isolated_cache_root:
        raise ValueError(
            f"baseline cache root differs from the frozen source selection for {motion!r}: "
            f"{cache_root} != {declared_root}; pass allow_isolated_cache_roots=True only for "
            "a provenance-backed experimental cohort"
        )
    if method == "terra":
        return {str(key): str(value) for key, value in source_row.items()} | {"retargeting_method": method}
    paths = retarget_cache_paths(cache_root, motion, method=method)
    return {str(key): str(value) for key, value in source_row.items()} | {
        "source_cache_root": str(cache_root),
        "trajectory_relpath": str(paths.trajectory_path.relative_to(cache_root)),
        "analysis_relpath": str(paths.analysis_path.relative_to(cache_root)),
        "terrain_relpath": (
            str(paths.terrain_path.relative_to(cache_root)) if source_row.get("terrain_relpath", "") else ""
        ),
        "retargeting_method": method,
    }


def _selection_signature(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, ...]]:
    fields = ("motion", "dataset", "motion_type", "split", *SEGMENT_FIELDS)
    return [tuple(str(row.get(field, "")) for field in fields) for row in rows]


def build_baseline_training_selections(
    source_selection: Path,
    cohort_records: Sequence[Path],
    *,
    storage_roots: StorageRoots | None = None,
    base: Path | None = None,
    trigger_seconds: float = 20.0,
    maximum_segment_seconds: float = 10.0,
    allow_isolated_cache_roots: bool = False,
) -> tuple[
    dict[RetargetingMethod, list[dict[str, str]]],
    dict[RetargetingMethod, dict[str, object]],
    dict[str, object],
]:
    """Build method-specific views over one exact shared-success intersection."""

    source_path = source_selection.expanduser().resolve()
    source_rows = read_manifest(source_path)
    if any(selection_segment(row) is not None for row in source_rows):
        raise ValueError("baseline training source selection must contain unsplit source clips")
    source_order = [row["motion"] for row in source_rows]
    source_motions = frozenset(source_order)
    source_digest = _sha256(source_path)
    if not cohort_records:
        raise ValueError("at least one baseline cohort record is required")

    coverages = [
        _load_cohort_coverage(
            path,
            source_selection_sha256=source_digest,
            source_motions=source_motions,
        )
        for path in cohort_records
    ]
    methods = [coverage.method for coverage in coverages]
    if len(methods) != len(set(methods)):
        raise ValueError("baseline cohort records must use distinct retargeting methods")
    coverage_by_method = {coverage.method: coverage for coverage in coverages}
    shared = set(source_motions)
    for coverage in coverages:
        shared.intersection_update(coverage.successful)
    shared_order = [motion for motion in source_order if motion in shared]
    if not shared_order:
        raise ValueError("retargeting methods have no shared successful motions")

    roots = storage_roots or StorageRoots.from_environment(Path.cwd())
    root_base = Path.cwd() if base is None else base
    rows_by_method: dict[RetargetingMethod, list[dict[str, str]]] = {}
    audits_by_method: dict[RetargetingMethod, dict[str, object]] = {}
    reference_signature: list[tuple[str, ...]] | None = None
    for method in ("terra", *methods):
        selected_method = validate_method(method)
        rebound = []
        for source_row in source_rows:
            motion = source_row["motion"]
            if motion not in shared:
                continue
            source_root = Path(source_row["source_cache_root"]).expanduser().resolve()
            if selected_method != "terra":
                source_root = coverage_by_method[selected_method].cache_roots[motion]
            rebound.append(
                _rebound_row(
                    source_row,
                    method=selected_method,
                    cache_root=source_root,
                    allow_isolated_cache_root=allow_isolated_cache_roots,
                )
            )
        segmented, segment_audit = expand_training_segments(
            rebound,
            storage_roots=roots,
            base=root_base,
            method=selected_method,
            trigger_seconds=trigger_seconds,
            maximum_segment_seconds=maximum_segment_seconds,
        )
        signature = _selection_signature(segmented)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(
                f"{selected_method} temporal views differ from TERRA; source frame counts/frequencies are not aligned"
            )
        rows_by_method[selected_method] = segmented
        audits_by_method[selected_method] = segment_audit | {
            "shared_source_selection": str(source_path),
            "shared_source_selection_sha256": source_digest,
        }

    audit: dict[str, object] = {
        "schema": "terra.retarget-baseline-training-selections.v1",
        "source_selection": str(source_path),
        "source_selection_sha256": source_digest,
        "source_motion_count": len(source_rows),
        "shared_source_motion_count": len(shared_order),
        "excluded_source_motion_count": len(source_rows) - len(shared_order),
        "retargeting_methods": ["terra", *methods],
        "segment_policy": SEGMENT_POLICY,
        "trigger_seconds": float(trigger_seconds),
        "maximum_segment_seconds": float(maximum_segment_seconds),
        "allow_isolated_cache_roots": allow_isolated_cache_roots,
        "output_motion_count": len(next(iter(rows_by_method.values()))),
        "split_counts": {
            split: sum(row.get("split", "train") == split for row in next(iter(rows_by_method.values())))
            for split in ("train", "evaluation", "test")
        },
        "method_coverage": {
            coverage.method: {
                "cohort_record": str(coverage.record_path),
                "cohort_record_sha256": _sha256(coverage.record_path),
                "target_fps": coverage.target_fps,
                "cache_roots": sorted({str(path) for path in coverage.cache_roots.values()}),
                "successful_source_motions": len(coverage.successful),
                "failed_source_motions": len(coverage.failed),
                "failed_motions": [motion for motion in source_order if motion in coverage.failed],
            }
            for coverage in coverages
        },
        "excluded_motions": [motion for motion in source_order if motion not in shared],
        "identical_motion_and_segment_identities": True,
    }
    return rows_by_method, audits_by_method, audit


def publish_baseline_training_selections(
    source_selection: Path,
    output_directory: Path,
    rows_by_method: Mapping[RetargetingMethod, Sequence[Mapping[str, str]]],
    audits_by_method: Mapping[RetargetingMethod, Mapping[str, object]],
    audit: Mapping[str, object],
    *,
    prefix: str = "universal_mixed_shared",
    overwrite: bool = False,
) -> dict[str, object]:
    """Publish matched method selections and one aggregate shared-cohort audit."""

    destination = output_directory.expanduser().resolve()
    outputs = {
        method: (
            destination / f"{prefix}_{method}_selection_segmented10s.csv",
            destination / f"{prefix}_{method}_segmented10s_audit.json",
        )
        for method in rows_by_method
    }
    aggregate_path = destination / f"{prefix}_audit.json"
    collisions = [path for pair in outputs.values() for path in pair if path.exists()]
    if aggregate_path.exists():
        collisions.append(aggregate_path)
    if collisions and not overwrite:
        raise FileExistsError(f"refusing to replace existing baseline training output: {collisions[0]}")

    published: dict[str, dict[str, object]] = {}
    for method, rows in rows_by_method.items():
        manifest_path, method_audit_path = outputs[method]
        payload = publish_segmented_selection(
            source_selection,
            manifest_path,
            method_audit_path,
            rows,
            audits_by_method[method],
            overwrite=overwrite,
        )
        published[method] = {
            "selection": str(manifest_path),
            "selection_sha256": payload["selection_sha256"],
            "audit": str(method_audit_path),
        }
    aggregate = dict(audit) | {
        "output_directory": str(destination),
        "methods": published,
    }
    atomic_write(
        aggregate_path,
        lambda temporary: temporary.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"),
    )
    write_git_commit(destination)
    return aggregate | {"audit": str(aggregate_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra train baseline-selections", description=__doc__)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--cohort-record", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="universal_mixed_shared")
    parser.add_argument("--trigger-seconds", type=float, default=20.0)
    parser.add_argument("--maximum-segment-seconds", type=float, default=10.0)
    parser.add_argument(
        "--allow-isolated-cache-roots",
        action="store_true",
        help=(
            "allow a terminal cohort record to rebind artifacts to its isolated experiment cache; "
            "the output audit records the opt-in, cache roots, and target frame rate"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        roots = StorageRoots.from_environment(Path.cwd())
        source = roots.resolve_input(args.source_selection, base=Path.cwd())
        records = [roots.resolve_artifact(path, base=Path.cwd()) for path in args.cohort_record]
        output = roots.resolve_artifact(args.out_dir, base=Path.cwd())
        assert source is not None and output is not None and all(record is not None for record in records)
        rows, method_audits, audit = build_baseline_training_selections(
            source,
            [record for record in records if record is not None],
            storage_roots=roots,
            base=Path.cwd(),
            trigger_seconds=args.trigger_seconds,
            maximum_segment_seconds=args.maximum_segment_seconds,
            allow_isolated_cache_roots=args.allow_isolated_cache_roots,
        )
        payload = publish_baseline_training_selections(
            source,
            output,
            rows,
            method_audits,
            audit,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_baseline_training_selections",
    "main",
    "publish_baseline_training_selections",
]
