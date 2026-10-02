"""Partial Gait120 conversions still publish inspectable manifests."""

from __future__ import annotations

import csv

from terra.datasets.gait120 import ClipRecord, validate_dataset


def test_failed_clip_remains_visible_in_public_manifest(tmp_path):
    clip = ClipRecord(
        subject=1,
        movement="StairAscent",
        trial=1,
        paired_steps="1,2",
        marker_archive="",
        output_path=str(tmp_path / "Gait120/S001/StairAscent/Trial01/AllSteps_stageii.npz"),
        motion="Gait120/S001/StairAscent/Trial01/AllSteps_stageii",
        trc_paths="",
        status="failed",
        error="marker fit failed",
    )

    report = validate_dataset([clip], output_root=tmp_path)

    assert report["manifest_published"] is True
    assert report["benchmark_ready"] is False
    with (tmp_path / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["motion"] == clip.motion
    assert rows[0]["fit_passed"] == "False"
    assert "marker fit failed" in rows[0]["fit_failure_reason"]
