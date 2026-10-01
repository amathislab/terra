"""Paired failure-review selection and frame recommendations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from terra.evaluation.failure_review import FAILURE_REVIEW_SCHEMA, METHODS, build_failure_review


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _annotations(path: Path, motion: str, method: str, *, failed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 12
    mask = np.zeros(n, dtype=bool)
    if failed:
        mask[4:8] = True
    np.savez_compressed(
        path,
        method=np.asarray(method),
        motion=np.asarray(motion),
        n_frames=np.asarray(n),
        common_active=np.ones(n, dtype=bool),
        environment_penetrating=mask,
        support_penetrating=np.zeros((n, 2), dtype=bool),
        sole_vertical_clearance_m=np.column_stack((-0.01 * mask, np.zeros(n))),
    )


def test_builds_unique_paired_failure_review(tmp_path):
    dataset = "example"
    artifact_root = tmp_path / "artifacts"
    motions = [f"Example/Motion{index:02d}" for index in range(9)]
    _write_csv(
        artifact_root / dataset / "retarget" / "manifest.csv",
        [
            {
                "motion": motion,
                "dataset": dataset,
                "terrain_class": "stairs_up" if index % 2 else "ramp_up",
            }
            for index, motion in enumerate(motions)
        ],
    )
    metric_rows = []
    for index, motion in enumerate(motions):
        for method in METHODS:
            metric_rows.append(
                {
                    "dataset": dataset,
                    "motion": motion,
                    "method": method,
                    "fps": 100.0,
                    "penetration_duration_pct": 0.0 if method == "terra" else 10.0 + index,
                    "error": "",
                }
            )
            quality = artifact_root / dataset / "evaluation" / "quality"
            frames = quality / "frames" if method == "terra" else quality / method / "frames"
            _annotations(frames / f"{motion.replace('/', '__')}.npz", motion, method, failed=method != "terra")
    benchmark = tmp_path / "per_motion.csv"
    _write_csv(benchmark, metric_rows)
    output = tmp_path / "review"

    payload = build_failure_review(
        benchmark,
        artifact_root,
        output,
        per_dataset=3,
        datasets=(dataset,),
    )

    assert payload["schema"] == FAILURE_REVIEW_SCHEMA
    assert payload["motions"] == 3
    assert payload["renders_expected"] == 12
    rows = list(csv.DictReader((output / "selections" / f"{dataset}.csv").open()))
    assert len(rows) == len({row["motion"] for row in rows}) == 3
    assert {row["highlight_baseline"] for row in rows} == {"omniretarget", "gmr", "smpl"}
    assert all(row["failure_mode"] == "penetration" for row in rows)
    assert all(row["highlight_frame"] in {"4", "5", "6", "7"} for row in rows)
    assert json.loads((output / "review.json").read_text())["renders_expected"] == 12
    assert len((output / "GIT_COMMIT").read_text().strip()) == 40
