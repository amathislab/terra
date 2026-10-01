"""Evaluation metadata must describe the thresholds used by workers."""

from __future__ import annotations

import json
from concurrent.futures import Future

from terra.evaluation import cli


class _ImmediatePool:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, _worker, job):
        future = Future()
        future.set_result((
            {"method": job[0], "motion_class": job[2], "motion": job[3], "error": "fixture"},
            {"method": job[1], "motion": job[3]},
        ))
        return future


def test_threshold_override_is_recorded_in_csv_and_run_metadata(monkeypatch, tmp_path):
    manifest = tmp_path / "motions.txt"
    manifest.write_text("Study/Trial\n")
    output = tmp_path / "evaluation"
    monkeypatch.setattr(cli, "ProcessPoolExecutor", _ImmediatePool)
    monkeypatch.setattr(cli, "aggregate", lambda *_args: [])
    monkeypatch.setattr(cli, "write_summary_tables", lambda *_args: None)
    monkeypatch.setattr(cli, "aggregate_joint_limit_sensitivity", lambda *_args: [])
    monkeypatch.setattr(cli, "write_joint_limit_sensitivity", lambda *_args: None)
    args = cli.parser().parse_args([
        "--motion-class", f"all={manifest}",
        "--method", "terra=terra",
        "--cache-root", str(tmp_path / "cache"),
        "--out", str(output),
        "--quality-out", str(output),
        "--penetration-tol", "0.02",
        "--allow-missing",
    ])

    assert cli.run(args) == 0
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["thresholds"]["continuous"]["penetration_m"] == 0.02
    assert "continuous.penetration_m,0.02" in (output / "thresholds.csv").read_text()
