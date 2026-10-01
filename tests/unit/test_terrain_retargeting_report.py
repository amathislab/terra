"""Tests for the matched Voronoi-terrain retargeting report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from terra.evaluation.evaluator import PER_MOTION_FIELDS
from terra.evaluation.terrain_retargeting_report import (
    ANGULAR_JOINT_LIMIT_FIELD,
    REPORT_SCHEMA,
    _terrain_source,
    build_report,
)


def _metric_row(dataset: str, motion: str, method: str, value: float, *, error: str = "") -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(PER_MOTION_FIELDS, "")
    row.update(method=method, motion_class=dataset, motion=motion, error=error)
    if not error:
        for field in PER_MOTION_FIELDS:
            if field.endswith(("_pct", "_mm", "_deg", "_s", "_m_s", "_residual_deg")):
                row[field] = value
        row.update(
            penetration_frame_depth_sum_mm=value,
            penetration_frame_depth_sq_sum_mm2=value * value,
            penetration_frame_depth_n=1,
            skating_frame_velocity_sum_m_s=value,
            skating_frame_velocity_sq_sum_m2_s2=value * value,
            skating_frame_velocity_n=1,
        )
        row[ANGULAR_JOINT_LIMIT_FIELD] = value
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*PER_MOTION_FIELDS, "dataset"), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_terrain_source_reads_only_json_envelope(tmp_path):
    path = tmp_path / "analysis.npz"
    payload = {"method": "voronoi", "record_sha256": "a" * 64}
    np.savez(
        path,
        terrain_reconstruction_source=np.asarray("__terra_json__:" + json.dumps(payload)),
        large_unrelated_array=np.ones((100, 100)),
    )

    assert _terrain_source(path) == payload


def test_build_report_matches_denominators_and_reports_failures(tmp_path, monkeypatch):
    import terra.evaluation.terrain_retargeting_report as module

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "benchmark.json").write_text(json.dumps({"schema": "terra.final-table2-retargeting"}))
    evaluations = []
    canonical_rows = []
    fake_items = {}
    for index, dataset in enumerate(module.DATASET_ORDER, 1):
        motion = f"Study/{dataset}"
        canonical_rows.append(_metric_row(dataset, motion, "terra", float(index)) | {"dataset": dataset})
        root = tmp_path / f"evaluation-{dataset}"
        metadata_path = root / "evaluation.json"
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text("{}")
        error = "missing artifact" if dataset == "amass" else ""
        _write_rows(root / "metrics/per_motion.csv", [_metric_row(dataset, motion, "terra-voronoi", index + 1, error=error)])
        fake_items[root.resolve()] = {
            "root": root.resolve(),
            "metadata_path": metadata_path.resolve(),
            "metadata": {"cache_root": str(tmp_path / "cache")},
            "dataset": dataset,
            "methods": {"terra-voronoi": "terra"},
            "metrics": {"motion_selections": {dataset: [motion]}},
        }
        evaluations.append((dataset, root))
    _write_rows(canonical / "metrics/per_motion.csv", canonical_rows)
    monkeypatch.setattr(module, "_load_evaluation", lambda path: fake_items[path.resolve()])
    monkeypatch.setattr(
        module,
        "_verify_terrain_identities",
        lambda _item, rows: {
            "checked_successful_artifacts": sum(not row["error"] for row in rows),
            "reconstruction_identities": [
                {
                    "method_identity_sha256": "a" * 64,
                    "scientific_identity_sha256": "b" * 64,
                    "motions": sum(not row["error"] for row in rows),
                }
            ],
        },
    )

    payload = build_report(canonical, evaluations, tmp_path / "report")

    assert payload["schema"] == REPORT_SCHEMA
    assert len(payload["report_script_sha256"]) == 64
    assert payload["voronoi_method_identity_sha256"] == "a" * 64
    assert payload["motions"] == 5
    assert payload["successful"] == 4
    assert payload["failed"] == 1
    assert payload["matched_metric_motions"] == 4
    assert payload["joint_limit_metric"]["tolerance_rad"] == 0.01
    table = (tmp_path / "report/table.md").read_text()
    assert "TERRA terrain" in table
    assert "Voronoi terrain" in table
    assert "0/1" in table
    summary = list(csv.DictReader((tmp_path / "report/summary.csv").open()))
    amass_control = next(
        row for row in summary if row["motion_class"] == "amass" and row["method"] == "TERRA terrain"
    )
    assert amass_control["n_motions"] == "1"
    assert amass_control["n_errors"] == "0"
    assert amass_control["penetration_duration_pct_n"] == "0"
    deltas = list(csv.DictReader((tmp_path / "report/deltas.csv").open()))
    assert float(deltas[0]["penetration_duration_pct_mean_delta"]) == 1.0
