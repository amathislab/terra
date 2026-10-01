"""Combined terrain-reconstruction benchmark reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from terra.benchmarking.reconstruction.matrix import DEFAULT_MATRIX, load_matrix
from terra.benchmarking.reconstruction.provenance import (
    METHOD_IDENTITY_SCHEMA,
    RUN_PROVENANCE_SCHEMA,
    SCIENTIFIC_IDENTITY_SCHEMA,
    content_sha256,
    write_evaluation_provenance,
)
from terra.benchmarking.reconstruction.registry import RECONSTRUCTION_METHODS, create_method
from terra.evaluation.reconstruction_benchmark import (
    APPARATUS_DATASETS,
    RECONSTRUCTION_BENCHMARK_SCHEMA,
    build_reconstruction_benchmark,
)

_MATRIX = load_matrix(DEFAULT_MATRIX)
_VORONOI_OPTIONS = create_method(
    "voronoi",
    motions=("fixture/motion",),
    options=_MATRIX.methods["voronoi"].options,
).options
_SOURCE = {
    "git_commit": "0" * 40,
    "state": "clean",
    "tree_sha256": "1" * 64,
}


def _method_root(path: Path) -> Path:
    return path.parents[2] if path.parent.name == "prism_mesh" else path.parents[1]


def _options(method: str) -> dict[str, object]:
    return _VORONOI_OPTIONS if method == "voronoi" else {"fixture_method": method}


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    options: dict[str, object] | None = None,
    source: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    method_root = _method_root(path)
    method = method_root.name
    resolved_options = _options(method) if options is None else options
    method_identity = {
        "schema": METHOD_IDENTITY_SCHEMA,
        "method": method,
        "resolved_options": resolved_options,
        "source": _SOURCE if source is None else source,
    }
    method_hash = content_sha256(method_identity)
    scientific_identity = {
        "schema": SCIENTIFIC_IDENTITY_SCHEMA,
        "method_identity_sha256": method_hash,
        "selection_sha256": content_sha256([row["motion"] for row in rows]),
        "dataset_config_sha256": None,
        "matrix_sha256": content_sha256("fixture-matrix"),
    }
    provenance = {
        "schema": RUN_PROVENANCE_SCHEMA,
        "method_identity": method_identity,
        "method_identity_sha256": method_hash,
        "scientific_identity": scientific_identity,
        "scientific_identity_sha256": content_sha256(scientific_identity),
        "inputs": {
            "selection": {
                "path": "fixture-selection.csv",
                "sha256": scientific_identity["selection_sha256"],
            },
            "dataset_config": None,
            "matrix": {
                "path": "fixture-matrix.toml",
                "sha256": scientific_identity["matrix_sha256"],
            },
        },
    }
    run_path = method_root / "terrain" / "run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        json.dumps(
            {
                "method": method,
                "motions": len(rows),
                "options": resolved_options,
                "provenance": provenance,
            }
        )
    )
    write_evaluation_provenance(path.parent, path, run_path)


def _minimal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    apparatus = tmp_path / "apparatus"
    prism = tmp_path / "prism"
    for dataset in APPARATUS_DATASETS:
        for method in RECONSTRUCTION_METHODS:
            _write_csv(
                apparatus / dataset / method / "evaluation" / "per_motion.csv",
                [{"motion": f"{dataset}/one", "terrain_available": True, "error": ""}],
            )
    for method in RECONSTRUCTION_METHODS:
        _write_csv(
            prism / method / "evaluation" / "prism_mesh" / "per_motion.csv",
            [{"motion": "prism/one", "candidate_fit_status": "ok"}],
        )
    return apparatus, prism


def test_combines_apparatus_and_prism_reconstruction_reports(tmp_path):
    apparatus = tmp_path / "reconstruction"
    prism = tmp_path / "terrain-reconstruction" / "prism"
    for dataset in APPARATUS_DATASETS:
        for method in RECONSTRUCTION_METHODS:
            failed = method == "voronoi"
            _write_csv(
                apparatus / dataset / method / "evaluation" / "per_motion.csv",
                [
                    {
                        "motion": f"{dataset}/one",
                        "terrain_available": "True",
                        "expected_family": "ramp",
                        "selected_family": "" if method == "voronoi" else "ramp",
                        "family_correct": "" if method == "voronoi" else "True",
                        "slope_abs_error_deg": "" if method == "voronoi" else 1.0,
                        "step_contact_height_mae_m": 0.010,
                        "seat_height_abs_error_m": 0.020,
                        "error": "",
                    },
                    {
                        "motion": f"{dataset}/two",
                        "terrain_available": str(not failed),
                        "expected_family": "ramp",
                        "selected_family": "",
                        "family_correct": "",
                        "slope_abs_error_deg": "" if method == "voronoi" else 3.0,
                        "step_contact_height_mae_m": 0.030,
                        "seat_height_abs_error_m": 0.040,
                        "error": "fit failed" if failed else "",
                    },
                ],
            )
    for method in RECONSTRUCTION_METHODS:
        failed = method == "voronoi"
        _write_csv(
            prism / method / "evaluation" / "prism_mesh" / "per_motion.csv",
            [
                {
                    "motion": "prism/one",
                    "candidate_fit_status": "ok",
                    "primary_eligible": True,
                    "foot_height_mae_mm": 10.0,
                    "seated_height_mae_mm": 20.0,
                    "observed_height_mae_mm": 15.0,
                    "raised_terrain_coverage": 0.9,
                    "flat_terrain_coverage": 0.95,
                    "full_footprint_iou": 0.5,
                },
                {
                    "motion": "prism/two",
                    "candidate_fit_status": "failed" if failed else "cached",
                    "primary_eligible": False,
                    "foot_height_mae_mm": 30.0,
                    "seated_height_mae_mm": 40.0,
                    "observed_height_mae_mm": 35.0,
                    "raised_terrain_coverage": 0.7,
                    "flat_terrain_coverage": 0.85,
                    "full_footprint_iou": 0.7,
                },
            ],
        )

    output = tmp_path / "benchmark" / "reconstruction"
    payload = build_reconstruction_benchmark(apparatus, prism, output)

    assert payload["schema"] == RECONSTRUCTION_BENCHMARK_SCHEMA
    assert payload["summary_rows"] == 16
    assert payload["method_motion_rows"] == 32
    summaries = list(csv.DictReader((output / "summary.csv").open()))
    gait_terra = next(row for row in summaries if row["dataset"] == "gait120" and row["method"] == "terra")
    assert gait_terra["successful"] == "2"
    assert float(gait_terra["family_accuracy_pct_mean"]) == 50.0
    assert float(gait_terra["ramp_angle_mae_deg_mean"]) == 2.0
    assert float(gait_terra["step_height_mae_mm_mean"]) == 20.0
    assert float(gait_terra["seat_height_mae_mm_mean"]) == 30.0
    gait_voronoi = next(row for row in summaries if row["dataset"] == "gait120" and row["method"] == "voronoi")
    assert gait_voronoi["family_accuracy_pct_n"] == "0"
    assert gait_voronoi["ramp_angle_mae_deg_n"] == "0"
    prism_voronoi = next(row for row in summaries if row["dataset"] == "prism" and row["method"] == "voronoi")
    assert prism_voronoi["successful"] == "1"
    assert prism_voronoi["failed"] == "1"
    assert float(prism_voronoi["observed_height_mae_mm_mean"]) == 15.0
    prism_terra = next(row for row in summaries if row["dataset"] == "prism" and row["method"] == "terra")
    assert prism_terra["observed_height_mae_mm_n"] == "1"
    assert float(prism_terra["observed_height_mae_mm_mean"]) == 15.0
    assert float(prism_terra["flat_terrain_coverage_pct_mean"]) == 90.0
    assert float(prism_terra["footprint_iou_pct_mean"]) == 60.0
    assert "[n=2]" in (output / "table.md").read_text()
    assert "N/A for representations without terrain families" in (output / "table.md").read_text()
    assert "|  | Voronoi | 1/2 | N/A | N/A |" in (output / "table.md").read_text()
    assert "TERRA w/o physical cues" in (output / "table.md").read_text()
    assert "Stool-height MAE" in (output / "table.md").read_text()
    assert "\\begin{tabular}" in (output / "table.tex").read_text()
    pooled = list(csv.DictReader((output / "pooled_family_accuracy.csv").open()))
    pooled_terra = next(row for row in pooled if row["method"] == "terra")
    assert pooled_terra["family_accuracy_pct_n"] == "6"
    assert float(pooled_terra["family_accuracy_pct_mean"]) == 50.0
    assert json.loads((output / "benchmark.json").read_text())["schema"] == RECONSTRUCTION_BENCHMARK_SCHEMA
    assert len((output / "GIT_COMMIT").read_text().strip()) == 40
    manifest = json.loads((output / "benchmark.json").read_text())
    assert len(manifest["input_artifacts"]) == 16
    first = manifest["input_artifacts"][0]
    assert first["sha256"] == hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest()


def test_combines_a_disjoint_supplemental_apparatus_cohort(tmp_path):
    apparatus = tmp_path / "reconstruction"
    supplemental = tmp_path / "chair"
    prism = tmp_path / "prism"
    for dataset in APPARATUS_DATASETS:
        for method in RECONSTRUCTION_METHODS:
            _write_csv(
                apparatus / dataset / method / "evaluation" / "per_motion.csv",
                [{"motion": f"{dataset}/base", "terrain_available": True, "error": ""}],
            )
    for method in RECONSTRUCTION_METHODS:
        _write_csv(
            supplemental / method / "evaluation" / "per_motion.csv",
            [
                {
                    "motion": "gait120/chair",
                    "terrain_available": True,
                    "seat_height_abs_error_m": 0.049,
                    "error": "",
                }
            ],
        )
        _write_csv(
            prism / method / "evaluation" / "prism_mesh" / "per_motion.csv",
            [{"motion": "prism/one", "candidate_fit_status": "ok"}],
        )

    output = tmp_path / "benchmark"
    payload = build_reconstruction_benchmark(
        apparatus,
        prism,
        output,
        supplemental_apparatus=[("gait120", supplemental)],
    )

    gait = list(csv.DictReader((output / "summary.csv").open()))
    gait_terra = next(row for row in gait if row["dataset"] == "gait120" and row["method"] == "terra")
    assert gait_terra["total"] == "2"
    assert gait_terra["seat_height_mae_mm_n"] == "1"
    assert float(gait_terra["seat_height_mae_mm_mean"]) == 49.0
    assert payload["method_motion_rows"] == 20
    assert payload["supplemental_apparatus"] == [{"dataset": "gait120", "root": str(supplemental.resolve())}]

    first_method = next(iter(RECONSTRUCTION_METHODS))
    _write_csv(
        supplemental / first_method / "evaluation" / "per_motion.csv",
        [{"motion": "gait120/base", "terrain_available": True, "error": ""}],
    )
    with pytest.raises(ValueError, match="overlaps an earlier cohort"):
        build_reconstruction_benchmark(
            apparatus,
            prism,
            tmp_path / "duplicate-benchmark",
            supplemental_apparatus=[("gait120", supplemental)],
        )


def test_rejects_voronoi_options_that_do_not_match_the_frozen_matrix(tmp_path):
    apparatus, prism = _minimal_inputs(tmp_path)
    path = apparatus / "darmstadt" / "voronoi" / "evaluation" / "per_motion.csv"
    _write_csv(
        path,
        [{"motion": "darmstadt/one", "terrain_available": True, "error": ""}],
        options={
            **_VORONOI_OPTIONS,
            "config": {
                **_VORONOI_OPTIONS["config"],
                "plateau_side_m": 0.2,
                "minimum_box_height_m": 0.04,
            },
            "terrain_links": ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "Pelvis"],
        },
    )

    with pytest.raises(ValueError, match="Voronoi options do not match frozen matrix"):
        build_reconstruction_benchmark(apparatus, prism, tmp_path / "benchmark")


def test_rejects_mixed_reconstruction_source_trees(tmp_path):
    apparatus, prism = _minimal_inputs(tmp_path)
    path = prism / "terra" / "evaluation" / "prism_mesh" / "per_motion.csv"
    _write_csv(
        path,
        [{"motion": "prism/one", "candidate_fit_status": "ok"}],
        source={**_SOURCE, "tree_sha256": "f" * 64},
    )

    with pytest.raises(ValueError, match="one identical source tree"):
        build_reconstruction_benchmark(apparatus, prism, tmp_path / "benchmark")


def test_rejects_a_tampered_evaluation_csv(tmp_path):
    apparatus, prism = _minimal_inputs(tmp_path)
    path = apparatus / "gait120" / "voronoi" / "evaluation" / "per_motion.csv"
    with path.open("a") as handle:
        handle.write("gait120/tampered,True\n")

    with pytest.raises(ValueError, match="evaluation CSV hash mismatch"):
        build_reconstruction_benchmark(apparatus, prism, tmp_path / "benchmark")
