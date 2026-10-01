"""Partial Gait120 conversions still publish inspectable manifests."""

from __future__ import annotations

import csv

import terra.datasets.gait120 as gait120
from terra.datasets.gait120 import ClipRecord, validate_dataset


def test_failed_clip_remains_visible_in_public_manifest(tmp_path):
    clip = ClipRecord(
        subject=1,
        movement="StairAscent",
        trial=1,
        paired_steps="1,2",
        marker_archive="",
        emg_archive="",
        output_path=str(tmp_path / "Gait120/S001/StairAscent/Trial01/AllSteps_stageii.npz"),
        motion="Gait120/S001/StairAscent/Trial01/AllSteps_stageii",
        emg_path="",
        trc_paths="",
        mot_paths="",
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


def test_ready_chair_selection_does_not_hide_failed_conversion(monkeypatch, tmp_path):
    good_path = tmp_path / "Gait120/S001/SitToStand/Trial01/AllSteps_stageii.npz"
    good_path.parent.mkdir(parents=True)
    good_path.touch()
    good = ClipRecord(
        subject=1, movement="SitToStand", trial=1, paired_steps="1",
        marker_archive="", emg_archive="", output_path=str(good_path),
        motion="Gait120/S001/SitToStand/Trial01/AllSteps_stageii",
        emg_path="", trc_paths="", mot_paths="", status="converted",
        marker_error_mean_mm=1.0,
    )
    failed = ClipRecord(
        subject=2, movement="SitToStand", trial=1, paired_steps="1",
        marker_archive="", emg_archive="",
        output_path=str(tmp_path / "Gait120/S002/SitToStand/Trial01/AllSteps_stageii.npz"),
        motion="Gait120/S002/SitToStand/Trial01/AllSteps_stageii",
        emg_path="", trc_paths="", mot_paths="", status="failed", error="fit failed",
    )

    def check_smplh(path):
        if path == good_path:
            return {"frames": 100, "fps": 50.0}
        raise FileNotFoundError(path)

    monkeypatch.setattr(gait120, "_validate_smplh_file", check_smplh)
    monkeypatch.setattr(gait120, "_validate_emg_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gait120, "validate_biomechanics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gait120, "select_balanced_chair_clips", lambda *_args, **_kwargs: ([good], {}))
    monkeypatch.setattr(gait120, "write_chair_selection", lambda *_args, **_kwargs: None)

    report = validate_dataset(
        [good, failed], output_root=tmp_path, chair_per_movement=1,
        chair_selection_output=tmp_path / "chairs.csv",
    )

    assert good.fit_passed is True
    assert failed.fit_passed is False
    assert report["chair_selection"]["ready"] is True
    assert report["benchmark_ready"] is False
    assert report["manifest_published"] is True
