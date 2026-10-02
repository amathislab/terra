"""Evaluation metadata must describe the thresholds used by workers."""

from __future__ import annotations

import json

from terra.evaluation import cli


def test_threshold_override_is_recorded_in_csv_and_run_metadata(tmp_path):
    manifest = tmp_path / "motions.txt"
    manifest.write_text("Study/Trial\n")
    output = tmp_path / "evaluation"
    args = cli.parser().parse_args(
        [
            "--motion-class",
            f"all={manifest}",
            "--method",
            "terra=terra",
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(output),
            "--quality-out",
            str(output),
            "--penetration-tol",
            "0.02",
            "--allow-missing",
        ]
    )

    assert cli.run(args) == 0
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["thresholds"]["continuous"]["penetration_m"] == 0.02
    assert "continuous.penetration_m,0.02" in (output / "thresholds.csv").read_text()
