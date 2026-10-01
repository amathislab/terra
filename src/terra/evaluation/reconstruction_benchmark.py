"""Combine completed apparatus and PRISM reconstruction evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from terra._revision import write_git_commit
from terra.benchmarking.reconstruction.matrix import DEFAULT_MATRIX, load_matrix
from terra.benchmarking.reconstruction.provenance import (
    canonical_json,
    file_sha256,
    load_evaluation_provenance,
)
from terra.benchmarking.reconstruction.registry import RECONSTRUCTION_METHODS, create_method
from terra.evaluation.reconstruction_statistics import (
    _apparatus_summary as _apparatus_summary,
)
from terra.evaluation.reconstruction_statistics import (
    _base_summary as _base_summary,
)
from terra.evaluation.reconstruction_statistics import (
    _classification_stats as _classification_stats,
)
from terra.evaluation.reconstruction_statistics import (
    _metric_fields as _metric_fields,
)
from terra.evaluation.reconstruction_statistics import (
    _pooled_family_summary as _pooled_family_summary,
)
from terra.evaluation.reconstruction_statistics import (
    _prism_summary as _prism_summary,
)
from terra.evaluation.reconstruction_statistics import (
    _stats as _stats,
)
from terra.evaluation.reconstruction_statistics import (
    _truth as _truth,
)

RECONSTRUCTION_BENCHMARK_SCHEMA = "terra.reconstruction-benchmark"
APPARATUS_DATASETS = ("gait120", "darmstadt", "vielemeyer")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"reconstruction evaluation not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"reconstruction evaluation is empty: {path}")
    return rows


def _input_record(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
    }


def _motion_set(path: Path, rows: list[dict[str, str]]) -> set[str]:
    motions = [row.get("motion", "").strip() for row in rows]
    if any(not motion for motion in motions):
        raise ValueError(f"reconstruction evaluation has a missing motion identifier: {path}")
    if len(motions) != len(set(motions)):
        raise ValueError(f"reconstruction evaluation has duplicate motions: {path}")
    return set(motions)


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


def _format_metric(row: dict[str, Any], key: str, decimals: int, *, latex: bool = False) -> str:
    count = int(row.get(f"{key}_n", 0) or 0)
    if not count:
        return "N/A"
    mean = float(row[f"{key}_mean"])
    std = float(row[f"{key}_std"])
    if latex:
        return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f} [{count}]"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f} [n={count}]"


def _display_method(method: str) -> str:
    return RECONSTRUCTION_METHODS[method].display_name


def _markdown_tables(summary: list[dict[str, Any]]) -> str:
    apparatus = [row for row in summary if row["dataset"] in APPARATUS_DATASETS]
    prism = [row for row in summary if row["dataset"] == "prism"]
    lines = [
        "# Terrain-reconstruction benchmark",
        "",
        "Values are per-motion mean ± population standard deviation. The bracketed denominator is the number "
        "of motions for which that metric is defined. Family accuracy is N/A for representations without "
        "terrain families; otherwise it uses every labeled motion and counts an unavailable prediction as "
        "incorrect.",
        "",
        "## Apparatus datasets",
        "",
        "| Dataset | Method | Success/total | Family accuracy (%) | Ramp-angle MAE (deg) | Step-height MAE (mm) | Stool-height MAE (mm) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    previous = None
    for row in apparatus:
        dataset = str(row["dataset"])
        dataset_cell = f"**{dataset}**" if dataset != previous else ""
        lines.append(
            f"| {dataset_cell} | {_display_method(str(row['method']))} | "
            f"{row['successful']}/{row['total']} | {_format_metric(row, 'family_accuracy_pct', 1)} | "
            f"{_format_metric(row, 'ramp_angle_mae_deg', 3)} | "
            f"{_format_metric(row, 'step_height_mae_mm', 2)} | "
            f"{_format_metric(row, 'seat_height_mae_mm', 2)} |"
        )
        previous = dataset
    lines.extend(
        [
            "",
            "## PRISM mesh development evaluation",
            "",
            "| Method | Success/total | Foot support MAE (mm) | Seated support MAE (mm) | "
            "Observed support MAE (mm) | Raised-terrain coverage (%) | Flat-terrain coverage (%) | "
            "Footprint IoU (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in prism:
        lines.append(
            f"| {_display_method(str(row['method']))} | {row['successful']}/{row['total']} | "
            f"{_format_metric(row, 'foot_height_mae_mm', 2)} | "
            f"{_format_metric(row, 'seated_height_mae_mm', 2)} | "
            f"{_format_metric(row, 'observed_height_mae_mm', 2)} | "
            f"{_format_metric(row, 'raised_terrain_coverage_pct', 1)} | "
            f"{_format_metric(row, 'flat_terrain_coverage_pct', 1)} | "
            f"{_format_metric(row, 'footprint_iou_pct', 1)} |"
        )
    return "\n".join(lines) + "\n"


def _latex_tables(summary: list[dict[str, Any]]) -> str:
    def escape(value: object) -> str:
        text = str(value)
        for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
            text = text.replace(old, new)
        return text

    apparatus = [row for row in summary if row["dataset"] in APPARATUS_DATASETS]
    prism = [row for row in summary if row["dataset"] == "prism"]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Method & Success/total & Family accuracy (\%) & Ramp-angle MAE (deg) & Step-height MAE (mm) & Stool-height MAE (mm) \\",
        r"\midrule",
    ]
    previous = None
    for row in apparatus:
        dataset = str(row["dataset"])
        cells = [
            escape(dataset) if dataset != previous else "",
            escape(_display_method(str(row["method"]))),
            f"{row['successful']}/{row['total']}",
            _format_metric(row, "family_accuracy_pct", 1, latex=True),
            _format_metric(row, "ramp_angle_mae_deg", 3, latex=True),
            _format_metric(row, "step_height_mae_mm", 2, latex=True),
            _format_metric(row, "seat_height_mae_mm", 2, latex=True),
        ]
        lines.append(" & ".join(cells) + r" \\")
        previous = dataset
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Method & Success/total & Foot MAE (mm) & Seated MAE (mm) & Observed MAE (mm) & "
            r"Raised coverage (\%) & Flat coverage (\%) & Footprint IoU (\%) \\",
            r"\midrule",
        ]
    )
    for row in prism:
        cells = [
            escape(_display_method(str(row["method"]))),
            f"{row['successful']}/{row['total']}",
            _format_metric(row, "foot_height_mae_mm", 2, latex=True),
            _format_metric(row, "seated_height_mae_mm", 2, latex=True),
            _format_metric(row, "observed_height_mae_mm", 2, latex=True),
            _format_metric(row, "raised_terrain_coverage_pct", 1, latex=True),
            _format_metric(row, "flat_terrain_coverage_pct", 1, latex=True),
            _format_metric(row, "footprint_iou_pct", 1, latex=True),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_reconstruction_benchmark(
    apparatus_root: Path,
    prism_root: Path,
    output_root: Path,
    *,
    supplemental_apparatus: list[tuple[str, Path]] | None = None,
    matrix_path: Path = DEFAULT_MATRIX,
) -> dict[str, Any]:
    apparatus = apparatus_root.expanduser().resolve()
    prism = prism_root.expanduser().resolve()
    methods = tuple(RECONSTRUCTION_METHODS)
    per_motion: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    supplemental = supplemental_apparatus or []
    unknown = sorted({dataset for dataset, _root in supplemental} - set(APPARATUS_DATASETS))
    if unknown:
        raise ValueError(f"unknown supplemental apparatus dataset(s): {', '.join(unknown)}")
    supplemental_by_dataset = {
        dataset: [root.expanduser().resolve() for name, root in supplemental if name == dataset]
        for dataset in APPARATUS_DATASETS
    }
    inputs: list[str] = []
    input_artifacts: list[dict[str, Any]] = []
    source_identities: dict[str, dict[str, str]] = {}
    voronoi_method_identities: set[str] = set()
    matrix = load_matrix(matrix_path)
    expected_voronoi = create_method(
        "voronoi",
        motions=("provenance/audit",),
        options=matrix.methods["voronoi"].options,
    ).options

    def audit_provenance(path: Path, method: str) -> dict[str, Any]:
        evaluation = load_evaluation_provenance(path, method)
        primary = evaluation["primary_reconstruction"]
        method_identity = primary["method_identity"]
        source = method_identity["source"]
        if source["state"] == "dirty":
            raise ValueError(f"publication benchmark rejects dirty reconstruction source: {path}")
        source_key = canonical_json(source)
        source_identities.setdefault(source_key, source)
        if method == "voronoi":
            if primary["resolved_options"] != expected_voronoi:
                raise ValueError(f"Voronoi options do not match frozen matrix {matrix.path}: {path}")
            voronoi_method_identities.add(primary["method_identity_sha256"])
        return evaluation

    for dataset in APPARATUS_DATASETS:
        method_motions: dict[str, set[str]] = {}
        for method in methods:
            paths = [
                apparatus / dataset / method / "evaluation" / "per_motion.csv",
                *(root / method / "evaluation" / "per_motion.csv" for root in supplemental_by_dataset[dataset]),
            ]
            rows: list[dict[str, str]] = []
            seen: set[str] = set()
            for path in paths:
                cohort_rows = _read_csv(path)
                evaluation_provenance = audit_provenance(path, method)
                cohort_motions = _motion_set(path, cohort_rows)
                overlap = seen & cohort_motions
                if overlap:
                    preview = ", ".join(sorted(overlap)[:3])
                    raise ValueError(f"supplemental {dataset}/{method} cohort overlaps an earlier cohort: {preview}")
                seen.update(cohort_motions)
                rows.extend(cohort_rows)
                inputs.append(str(path))
                input_artifacts.append(
                    _input_record(path, cohort_rows)
                    | {
                        "evaluation_provenance_sha256": file_sha256(path.parent / "evaluation.json"),
                        "method_identity_sha256": evaluation_provenance["primary_reconstruction"][
                            "method_identity_sha256"
                        ],
                        "scientific_identity_sha256": evaluation_provenance["primary_reconstruction"][
                            "scientific_identity_sha256"
                        ],
                    }
                )
            method_motions[method] = seen
            per_motion.extend(row | {"benchmark_dataset": dataset, "benchmark_method": method} for row in rows)
            summaries.append(_apparatus_summary(dataset, method, rows))
        reference_method = methods[0]
        reference_motions = method_motions[reference_method]
        for method in methods[1:]:
            if method_motions[method] != reference_motions:
                missing = len(reference_motions - method_motions[method])
                extra = len(method_motions[method] - reference_motions)
                raise ValueError(
                    f"{dataset}/{method} motion set differs from {reference_method}: {missing} missing, {extra} extra"
                )
    prism_motions: dict[str, set[str]] = {}
    for method in methods:
        path = prism / method / "evaluation" / "prism_mesh" / "per_motion.csv"
        rows = _read_csv(path)
        evaluation_provenance = audit_provenance(path, method)
        prism_motions[method] = _motion_set(path, rows)
        inputs.append(str(path))
        input_artifacts.append(
            _input_record(path, rows)
            | {
                "evaluation_provenance_sha256": file_sha256(path.parent / "evaluation.json"),
                "method_identity_sha256": evaluation_provenance["primary_reconstruction"]["method_identity_sha256"],
                "scientific_identity_sha256": evaluation_provenance["primary_reconstruction"][
                    "scientific_identity_sha256"
                ],
            }
        )
        per_motion.extend(row | {"benchmark_dataset": "prism", "benchmark_method": method} for row in rows)
        summaries.append(_prism_summary(method, rows))
    reference_method = methods[0]
    for method in methods[1:]:
        if prism_motions[method] != prism_motions[reference_method]:
            missing = len(prism_motions[reference_method] - prism_motions[method])
            extra = len(prism_motions[method] - prism_motions[reference_method])
            raise ValueError(
                f"prism/{method} motion set differs from {reference_method}: {missing} missing, {extra} extra"
            )
    if len(source_identities) != 1:
        raise ValueError(
            "reconstruction evaluations were not produced by one identical source tree: "
            f"{len(source_identities)} source identities"
        )
    if len(voronoi_method_identities) != 1:
        raise ValueError(
            "Voronoi reconstruction cohorts do not share one method identity: "
            f"{len(voronoi_method_identities)} identities"
        )

    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_git_commit(output)
    pooled_family = _pooled_family_summary(per_motion)
    _write_csv(output / "per_motion.csv", per_motion)
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "pooled_family_accuracy.csv", pooled_family)
    _write_csv(
        output / "method_counts.csv",
        [{key: row[key] for key in ("dataset", "method", "successful", "failed", "total")} for row in summaries],
    )
    (output / "table.md").write_text(_markdown_tables(summaries))
    (output / "table.tex").write_text(_latex_tables(summaries))
    payload = {
        "schema": RECONSTRUCTION_BENCHMARK_SCHEMA,
        "apparatus_root": str(apparatus),
        "prism_root": str(prism),
        "inputs": inputs,
        "input_artifacts": input_artifacts,
        "matrix": {
            "path": str(matrix.path),
            "sha256": file_sha256(matrix.path),
        },
        "source_identity": next(iter(source_identities.values())),
        "voronoi_method_identity_sha256": next(iter(voronoi_method_identities)),
        "supplemental_apparatus": [
            {"dataset": dataset, "root": str(root.expanduser().resolve())} for dataset, root in supplemental
        ],
        "datasets": [*APPARATUS_DATASETS, "prism"],
        "methods": list(methods),
        "method_motion_rows": len(per_motion),
        "summary_rows": len(summaries),
        "pooled_family_accuracy": pooled_family,
    }
    (output / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Reconstruction benchmark: {len(summaries)} dataset-method rows -> {output}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="terra benchmark reconstruction",
        description=__doc__,
    )
    result.add_argument("--apparatus-root", type=Path, required=True)
    result.add_argument(
        "--supplemental-apparatus",
        action="append",
        default=[],
        metavar="DATASET=ROOT",
        help=(
            "additional disjoint apparatus cohort whose root contains METHOD/evaluation/per_motion.csv; "
            "repeat as needed"
        ),
    )
    result.add_argument("--prism-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help="frozen matrix whose resolved Voronoi configuration every input must match",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        supplemental: list[tuple[str, Path]] = []
        for value in args.supplemental_apparatus:
            dataset, separator, root = value.partition("=")
            if not separator or not dataset.strip() or not root.strip():
                raise ValueError("--supplemental-apparatus must be DATASET=ROOT")
            supplemental.append((dataset.strip(), Path(root.strip())))
        build_reconstruction_benchmark(
            args.apparatus_root,
            args.prism_root,
            args.output_root,
            supplemental_apparatus=supplemental,
            matrix_path=args.matrix,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


__all__ = ["RECONSTRUCTION_BENCHMARK_SCHEMA", "build_reconstruction_benchmark", "main"]
