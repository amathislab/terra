"""Compare TERRA retargeting on TERRA and precomputed Voronoi terrain."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from terra._revision import write_git_commit
from terra.artifacts import retarget_cache_paths
from terra.benchmarking.reconstruction.provenance import file_sha256
from terra.evaluation.benchmark import BENCHMARK_SCHEMA, _load_evaluation
from terra.evaluation.reporting import aggregate, summary_fields

REPORT_SCHEMA = "terra.voronoi-terrain-retargeting-report.v1"
CANONICAL_BENCHMARK_SCHEMAS = {
    BENCHMARK_SCHEMA,
    "terra.manuscript-retargeting-benchmark",
    "terra.final-table2-retargeting",
}
VORONOI_LABEL = "terra-voronoi"
DISPLAY_METHODS = ("TERRA terrain", "Voronoi terrain")
DATASET_ORDER = ("gait120", "darmstadt", "vielemeyer", "amass", "prism")
_ANALYSIS_JSON_PREFIX = "__terra_json__:"
ANGULAR_JOINT_LIMIT_FIELD = "joint_limit_duration_pct_at_1em02_rad"
TABLE_METRICS = (
    ("joint_limit_duration_pct", "Joint-limit duration (%)", 2),
    ("tendon_jump_duration_pct", "Tendon-jump duration (%)", 3),
    ("self_collision_duration_pct", "Inter-leg collision duration (%)", 2),
    ("penetration_duration_pct", "Penetration duration (%)", 2),
    ("skating_duration_pct", "Skating duration (%)", 2),
    ("floating_duration_pct", "Floating duration (%)", 2),
    ("pelvis_relative_landmark_rmse_mm", "Landmark RMSE (mm)", 2),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required CSV does not exist: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _assignment(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--evaluation must use PARTITION=PATH syntax") from exc
    name = name.strip().casefold()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("--evaluation contains an empty partition or path")
    return name, Path(raw_path).expanduser().resolve()


def _success(row: dict[str, str]) -> bool:
    return not row.get("error", "").strip()


def _report_metric_row(row: dict[str, str]) -> dict[str, str]:
    angular_duration = row.get(ANGULAR_JOINT_LIMIT_FIELD, "")
    if _success(row) and angular_duration == "":
        raise ValueError(f"successful metric row is missing {ANGULAR_JOINT_LIMIT_FIELD}: {row.get('motion', '')}")
    return row | {"joint_limit_duration_pct": angular_duration}


def _terrain_source(path: Path) -> dict[str, Any]:
    """Read only the small provenance member, not each archive's large metric arrays."""

    with np.load(path, allow_pickle=False) as archive:
        if "terrain_reconstruction_source" not in archive.files:
            raise ValueError(f"analysis has no terrain_reconstruction_source: {path}")
        value: object = archive["terrain_reconstruction_source"]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, str) and value.startswith(_ANALYSIS_JSON_PREFIX):
        value = json.loads(value.removeprefix(_ANALYSIS_JSON_PREFIX))
    if not isinstance(value, dict):
        raise ValueError(f"analysis terrain_reconstruction_source is not an object: {path}")
    return value


def _verify_terrain_identities(item: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    cache_root = Path(str(item["metadata"]["cache_root"])).expanduser().resolve()
    identities: Counter[str] = Counter()
    checked = 0
    for row in rows:
        if not _success(row):
            continue
        analysis_path = retarget_cache_paths(cache_root, row["motion"], method="terra").analysis_path
        source = _terrain_source(analysis_path)
        if source.get("method") != "voronoi":
            raise ValueError(f"successful alternative artifact is not bound to Voronoi terrain: {analysis_path}")
        required = ("method_identity_sha256", "scientific_identity_sha256", "record_sha256")
        if any(not isinstance(source.get(key), str) or len(source[key]) != 64 for key in required):
            raise ValueError(f"alternative artifact has incomplete terrain provenance: {analysis_path}")
        identities[json.dumps({key: source[key] for key in required[:2]}, sort_keys=True)] += 1
        checked += 1
    return {
        "checked_successful_artifacts": checked,
        "reconstruction_identities": [json.loads(value) | {"motions": count} for value, count in identities.items()],
    }


def _format(mean: object, spread: object, decimals: int) -> str:
    try:
        return f"{float(mean):.{decimals}f} ± {float(spread):.{decimals}f}"
    except (TypeError, ValueError):
        return "--"


def _success_cell(row: dict[str, Any]) -> str:
    successful = int(row["n_motions"])
    failed = int(row["n_errors"])
    return f"{successful}/{successful + failed}"


def _markdown(summary: list[dict[str, Any]]) -> str:
    headers = ["Dataset", "Terrain used by TERRA", "Success", *(label for _key, label, _digits in TABLE_METRICS)]
    lines = [
        "# TERRA retargeting with Voronoi terrain",
        "",
        "Values are per-motion mean ± population standard deviation across finite common-success motions.",
        "The solver, source motion, fitted body shape, and dataset-specific weights are unchanged; only the",
        "precomputed terrain record differs.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in summary:
        cells = [row["motion_class"], row["method"], _success_cell(row)]
        cells.extend(_format(row[f"{key}_mean"], row[f"{key}_std"], decimals) for key, _label, decimals in TABLE_METRICS)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _latex(summary: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llr" + "r" * len(TABLE_METRICS) + "}",
        r"\toprule",
        "Dataset & Terrain & Success & "
        + " & ".join(label.replace("%", r"\%").replace("±", r"$\pm$") for _key, label, _digits in TABLE_METRICS)
        + r" \\",
        r"\midrule",
    ]
    for row in summary:
        cells = [str(row["motion_class"]), str(row["method"]), _success_cell(row)]
        cells.extend(
            _format(row[f"{key}_mean"], row[f"{key}_std"], decimals).replace("±", r"$\pm$")
            for key, _label, decimals in TABLE_METRICS
        )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def build_report(
    canonical_benchmark: Path,
    evaluations: list[tuple[str, Path]],
    output_root: Path,
) -> dict[str, Any]:
    """Build a matched, per-motion terrain-source comparison."""

    canonical_root = canonical_benchmark.expanduser().resolve()
    metadata_path = canonical_root / "benchmark.json"
    try:
        canonical_metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical retargeting benchmark: {metadata_path}") from exc
    if not isinstance(canonical_metadata, dict) or canonical_metadata.get("schema") not in CANONICAL_BENCHMARK_SCHEMAS:
        raise ValueError(f"canonical retargeting benchmark schema is missing or obsolete: {metadata_path}")
    canonical_rows = [row for row in _read_csv(canonical_root / "metrics" / "per_motion.csv") if row["method"] == "terra"]
    canonical_by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    for row in canonical_rows:
        dataset = row.get("dataset", row.get("motion_class", "")).casefold()
        motions = canonical_by_dataset.setdefault(dataset, {})
        if row["motion"] in motions:
            raise ValueError(f"canonical benchmark repeats {dataset}/{row['motion']}")
        motions[row["motion"]] = row

    partition_names = [name for name, _path in evaluations]
    if len(partition_names) != len(set(partition_names)):
        raise ValueError("evaluation partition names must be unique")
    alternative_by_dataset: dict[str, list[dict[str, str]]] = {}
    input_records: list[dict[str, Any]] = []
    for partition, path in evaluations:
        item = _load_evaluation(path)
        if item["methods"] != {VORONOI_LABEL: "terra"}:
            raise ValueError(
                f"{partition} must evaluate exactly --method {VORONOI_LABEL}=terra; got {item['methods']}"
            )
        rows = _read_csv(item["root"] / "metrics" / "per_motion.csv")
        selected = [motion for values in item["metrics"]["motion_selections"].values() for motion in values]
        if len(rows) != len(selected) or {row["motion"] for row in rows} != set(selected):
            raise ValueError(f"{partition} does not contain one metric row per selected motion")
        dataset = str(item["dataset"]).casefold()
        alternative_by_dataset.setdefault(dataset, []).extend(rows)
        input_records.append(
            {
                "partition": partition,
                "dataset": dataset,
                "evaluation": str(item["metadata_path"]),
                "evaluation_sha256": file_sha256(item["metadata_path"]),
                "metrics_sha256": file_sha256(item["root"] / "metrics" / "per_motion.csv"),
                "motions": len(rows),
                "successful": sum(_success(row) for row in rows),
                "terrain_provenance_audit": _verify_terrain_identities(item, rows),
            }
        )

    if set(alternative_by_dataset) != set(DATASET_ORDER):
        raise ValueError(
            f"alternative evaluations must cover {DATASET_ORDER}; got {tuple(sorted(alternative_by_dataset))}"
        )
    method_identities = {
        identity["method_identity_sha256"]
        for record in input_records
        for identity in record["terrain_provenance_audit"]["reconstruction_identities"]
    }
    if len(method_identities) != 1:
        raise ValueError(f"alternative artifacts must use one Voronoi method identity; got {sorted(method_identities)}")
    combined: list[dict[str, str]] = []
    classes: list[tuple[str, Path]] = []
    requested_by_dataset: dict[str, int] = {}
    common_success_by_dataset: dict[str, int] = {}
    for dataset in DATASET_ORDER:
        alternative_rows = alternative_by_dataset[dataset]
        alternative_motions = [row["motion"] for row in alternative_rows]
        if len(alternative_motions) != len(set(alternative_motions)):
            raise ValueError(f"alternative {dataset} partitions overlap")
        canonical = canonical_by_dataset.get(dataset, {})
        if set(alternative_motions) != set(canonical):
            missing = sorted(set(canonical) - set(alternative_motions))
            extra = sorted(set(alternative_motions) - set(canonical))
            raise ValueError(f"{dataset} motion denominator differs from canonical; missing={missing[:3]}, extra={extra[:3]}")
        canonical_errors = [motion for motion, row in canonical.items() if not _success(row)]
        if canonical_errors:
            raise ValueError(f"canonical TERRA benchmark contains failures for {dataset}: {canonical_errors[:3]}")
        common_success = {row["motion"] for row in alternative_rows if _success(row)}
        requested_by_dataset[dataset] = len(alternative_motions)
        common_success_by_dataset[dataset] = len(common_success)
        combined.extend(
            _report_metric_row(canonical[motion]) | {"method": DISPLAY_METHODS[0], "motion_class": dataset}
            for motion in alternative_motions
            if motion in common_success
        )
        combined.extend(
            _report_metric_row(row) | {"method": DISPLAY_METHODS[1], "motion_class": dataset}
            for row in alternative_rows
        )
        classes.append((dataset, Path(".")))
    summary = aggregate(combined, [(name, "terra") for name in DISPLAY_METHODS], classes)
    for row in summary:
        if row["method"] == DISPLAY_METHODS[0]:
            row["n_motions"] = requested_by_dataset[row["motion_class"]]
            row["n_errors"] = 0
    deltas: list[dict[str, Any]] = []
    for index in range(0, len(summary), 2):
        canonical, alternative = summary[index : index + 2]
        delta = {"dataset": canonical["motion_class"]}
        for key, _label, _digits in TABLE_METRICS:
            delta[f"{key}_mean_delta"] = float(alternative[f"{key}_mean"]) - float(canonical[f"{key}_mean"])
        deltas.append(delta)

    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_git_commit(output)
    _write_csv(output / "summary.csv", summary, summary_fields())
    _write_csv(
        output / "deltas.csv",
        deltas,
        ("dataset", *(f"{key}_mean_delta" for key, _label, _digits in TABLE_METRICS)),
    )
    (output / "table.md").write_text(_markdown(summary))
    (output / "table.tex").write_text(_latex(summary))
    payload = {
        "schema": REPORT_SCHEMA,
        "report_script": str(Path(__file__).resolve()),
        "report_script_sha256": file_sha256(Path(__file__).resolve()),
        "voronoi_method_identity_sha256": next(iter(method_identities)),
        "canonical_benchmark": str(metadata_path),
        "canonical_benchmark_sha256": file_sha256(metadata_path),
        "canonical_metrics_sha256": file_sha256(canonical_root / "metrics" / "per_motion.csv"),
        "evaluations": input_records,
        "motions": sum(len(rows) for rows in alternative_by_dataset.values()),
        "successful": sum(_success(row) for rows in alternative_by_dataset.values() for row in rows),
        "failed": sum(not _success(row) for rows in alternative_by_dataset.values() for row in rows),
        "matched_metric_motions": sum(common_success_by_dataset.values()),
        "joint_limit_metric": {
            "source_field": ANGULAR_JOINT_LIMIT_FIELD,
            "joint_types": "limited hinge joints",
            "tolerance_rad": 0.01,
        },
        "aggregation": (
            "per-motion arithmetic mean and population standard deviation over finite motions successful "
            "with both terrain sources; success counts retain the full requested denominator"
        ),
        "comparison_control": (
            "identical source motion, fitted body shape, TERRA solver, and dataset weights; "
            "only the precomputed terrain reconstruction changes"
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Voronoi-terrain retargeting report: {payload['successful']}/{payload['motions']} -> {output}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra benchmark terrain-retargeting", description=__doc__)
    parser.add_argument("--canonical-benchmark", required=True, type=Path)
    parser.add_argument("--evaluation", action="append", required=True, type=_assignment)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        build_report(args.canonical_benchmark, args.evaluation, args.output_root)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


__all__ = ["REPORT_SCHEMA", "build_report", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
