"""Assemble completed retargeting evaluations into one benchmark report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from terra._revision import write_git_commit
from terra.evaluation.audit import AUDIT_SCHEMA, audit_timing_contexts
from terra.evaluation.dataset import RETARGET_EVALUATION_SCHEMA
from terra.evaluation.reporting import HEADLINE_TABLE_SECTIONS, write_tables

BENCHMARK_SCHEMA = "terra.dataset-benchmark"
METHOD_DISPLAY_NAMES = {
    "terra": "TERRA",
    "omniretarget": "OmniRetarget",
    "gmr": "GMR",
    "smpl": "SMPL",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"evaluation output not found: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["motion"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _quality_name(method: str) -> str:
    suffix = method.removeprefix("terra_")
    return "quality.csv" if method == "terra" else f"quality_{suffix}.csv"


def _load_evaluation(root: Path) -> dict[str, Any]:
    evaluation_root = root.expanduser().resolve()
    metadata_path = evaluation_root / "evaluation.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dataset evaluation: {metadata_path}") from error
    if not isinstance(metadata, dict) or metadata.get("schema") != RETARGET_EVALUATION_SCHEMA:
        raise ValueError(f"retargeting evaluation schema is missing or obsolete: {metadata_path}")
    if Path(str(metadata.get("output_root", ""))).expanduser().resolve() != evaluation_root:
        raise ValueError(f"dataset evaluation was published for a different output root: {metadata_path}")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"dataset evaluation has no dataset name: {metadata_path}")
    config = Path(str(metadata.get("config", ""))).expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"{dataset} evaluation config is missing")
    manifest = Path(str(metadata.get("manifest", ""))).expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"{dataset} evaluation manifest is missing")
    methods = metadata.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError(f"{dataset} evaluation has no method mapping")
    if any(not isinstance(label, str) or not isinstance(method, str) for label, method in methods.items()):
        raise ValueError(f"{dataset} evaluation method mapping is invalid")
    if len(set(methods.values())) != len(methods):
        raise ValueError(f"{dataset} evaluation contains duplicate cache methods")
    stages = metadata.get("stages")
    if not isinstance(stages, dict) or set(stages) != {"metrics"}:
        raise ValueError(f"{dataset} retargeting evaluation must contain only the metrics stage")
    failed = [name for name, stage in stages.items() if not isinstance(stage, dict) or stage.get("exit_code") != 0]
    if failed:
        raise ValueError(f"{dataset} evaluation contains failed stages: {', '.join(failed)}")

    metrics_path = evaluation_root / "metrics" / "run.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metrics for {dataset}: {metrics_path}") from error
    if not isinstance(metrics, dict):
        raise ValueError(f"metric run metadata must contain an object: {metrics_path}")
    metric_methods = metrics.get("methods")
    if not isinstance(metric_methods, dict) or metric_methods != methods:
        raise ValueError(f"{dataset} metric methods do not match the dataset evaluation")
    metric_manifests = metrics.get("manifests")
    if not isinstance(metric_manifests, dict) or {
        Path(str(path)).expanduser().resolve() for path in metric_manifests.values()
    } != {manifest}:
        raise ValueError(f"{dataset} metric manifest does not match the dataset evaluation")
    selections = metrics.get("motion_selections")
    if not isinstance(selections, dict) or any(not isinstance(value, list) for value in selections.values()):
        raise ValueError(f"{dataset} metric motion selection is missing")

    return {
        "root": evaluation_root,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "dataset": dataset,
        "manifest": manifest,
        "methods": {str(label): str(method) for label, method in methods.items()},
        "metrics": metrics,
    }


def build_benchmark(evaluation_roots: list[Path], output_root: Path) -> dict[str, Any]:
    if not evaluation_roots:
        raise ValueError("at least one evaluation directory is required")
    evaluations = [_load_evaluation(path) for path in evaluation_roots]
    names = [item["dataset"] for item in evaluations]
    if len(names) != len(set(names)):
        raise ValueError("benchmark dataset names must be unique")

    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    method_count_rows: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    for item in evaluations:
        dataset = item["dataset"]
        root = item["root"]
        methods = item["methods"]
        rows = _read_csv(root / "metrics" / "per_motion.csv")
        audit_path = root / "metrics" / "audit.json"
        try:
            metric_audit = json.loads(audit_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{dataset} metric audit is missing: {audit_path}") from error
        if (
            not isinstance(metric_audit, dict)
            or metric_audit.get("schema") != AUDIT_SCHEMA
            or metric_audit.get("passed") is not True
        ):
            raise ValueError(f"{dataset} metric audit did not pass: {audit_path}")
        timing_audit = audit_timing_contexts(rows)
        if not timing_audit["passed"]:
            preview = "; ".join(timing_audit["issues"][:3])
            raise ValueError(f"{dataset} T_frame timing audit did not pass: {preview}")
        selected = sum(len(record) for record in item["metrics"]["motion_selections"].values())
        expected_pairs = selected * len(methods)
        pairs = [(row.get("method", ""), row.get("motion", "")) for row in rows]
        if len(rows) != expected_pairs or len(set(pairs)) != expected_pairs:
            raise ValueError(f"{dataset} metrics are incomplete: expected {expected_pairs} unique method-motion rows")
        metric_rows.extend(row | {"dataset": dataset} for row in rows)
        summary_rows.extend(row | {"dataset": dataset} for row in _read_csv(root / "metrics" / "summary.csv"))
        dataset_counts: dict[str, dict[str, int]] = {}
        for label in methods:
            method_rows = [row for row in rows if row.get("method") == label]
            successful = sum(not row.get("error", "").strip() for row in method_rows)
            counts = {
                "successful": successful,
                "failed": len(method_rows) - successful,
                "total": len(method_rows),
            }
            dataset_counts[label] = counts
            method_count_rows.append({"dataset": dataset, "method": label} | counts)
        for method in methods.values():
            path = root / "quality" / _quality_name(method)
            method_rows = _read_csv(path)
            if len(method_rows) != selected:
                raise ValueError(f"{dataset}/{method} quality rows do not match the selected denominator")
            quality_rows.extend(row | {"dataset": dataset, "method": method} for row in method_rows)
        datasets.append(
            {
                "dataset": dataset,
                "evaluation": str(item["metadata_path"]),
                "manifest": str(item["manifest"]),
                "selected_motions": selected,
                "evaluated_method_motion_pairs": expected_pairs,
                "method_counts": dataset_counts,
                "methods": methods,
                "metric_audit": str(audit_path),
                "timing_mode": timing_audit["mode"],
                "timing_context": (
                    timing_audit["distinct_contexts"][0] if timing_audit["distinct_contexts"] else None
                ),
                "timing_warnings": timing_audit["warnings"],
            }
        )

    combined_timing_audit = audit_timing_contexts(metric_rows)
    if not combined_timing_audit["passed"]:
        preview = "; ".join(combined_timing_audit["issues"][:3])
        raise ValueError(f"combined T_frame timing audit did not pass: {preview}")

    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_git_commit(output)
    _write_csv(output / "metrics" / "per_motion.csv", metric_rows)
    _write_csv(output / "metrics" / "summary.csv", summary_rows)
    table_rows = [
        row
        | {
            "motion_class": row["dataset"],
            "method": METHOD_DISPLAY_NAMES.get(str(row["method"]), str(row["method"])),
        }
        for row in summary_rows
    ]
    write_tables(
        output / "metrics",
        table_rows,
        sections=HEADLINE_TABLE_SECTIONS,
        group_label="Dataset",
    )
    _write_csv(output / "method_counts.csv", method_count_rows)
    _write_csv(output / "quality.csv", quality_rows)
    method_counts = Counter(row["method"] for row in metric_rows)
    payload = {
        "schema": BENCHMARK_SCHEMA,
        "datasets": datasets,
        "selected_motions": sum(item["selected_motions"] for item in datasets),
        "evaluated_method_motion_pairs": len(metric_rows),
        "metric_rows_by_method": dict(sorted(method_counts.items())),
        "method_counts": method_count_rows,
        "quality_rows": len(quality_rows),
        "timing_mode": combined_timing_audit["mode"],
        "timing_context": (
            combined_timing_audit["distinct_contexts"][0]
            if combined_timing_audit["distinct_contexts"]
            else None
        ),
        "timing_warnings": combined_timing_audit["warnings"],
    }
    (output / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Benchmark: {payload['selected_motions']} motions, "
        f"{payload['evaluated_method_motion_pairs']} method-motion pairs -> {output}"
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="terra benchmark", description=__doc__)
    result.add_argument(
        "--retarget-evaluation",
        dest="evaluation",
        action="append",
        required=True,
        type=Path,
        help="completed retargeting evaluation directory; repeat for every dataset",
    )
    result.add_argument("--output-root", type=Path, required=True)
    return result


def retargeting_main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        build_benchmark(args.evaluation, args.output_root)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "reconstruction":
        from terra.evaluation.reconstruction_benchmark import main as reconstruction_main

        return reconstruction_main(arguments[1:])
    if arguments and arguments[0] == "failure-review":
        from terra.evaluation.failure_review import main as failure_review_main

        return failure_review_main(arguments[1:])
    if arguments and arguments[0] == "terrain-retargeting":
        from terra.evaluation.terrain_retargeting_report import main as terrain_retargeting_main

        return terrain_retargeting_main(arguments[1:])
    if arguments and arguments[0] == "retargeting":
        arguments = arguments[1:]
    return retargeting_main(arguments)


__all__ = ["BENCHMARK_SCHEMA", "build_benchmark", "main", "retargeting_main"]
