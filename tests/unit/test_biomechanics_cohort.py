"""Tests for resumable multi-motion biomechanics comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import terra.biomechanics_cohort as cohort_module
from terra.biomechanics_cohort import CohortCase, compare_cohort, load_cohort_manifest
from terra.biomechanics_validation import NoSuccessfulRolloutError, TraceMatch


def _manifest(path: Path, *, subject: str = "D05") -> Path:
    fields = (
        "case_id",
        "motion",
        "dataset",
        "cache_subdir",
        "subject",
        "motion_type",
        "split",
        "evaluation_tier",
        "expected_emg",
        "expected_grf",
    )
    row = {
        "case_id": "darmstadt-d05-low-ascent",
        "motion": "Darmstadt/D05/config01/trial01/ascent_stageii",
        "dataset": "darmstadt",
        "cache_subdir": "darmstadt/cache",
        "subject": subject,
        "motion_type": "riser_0.10_ascent",
        "split": "test",
        "evaluation_tier": "primary",
        "expected_emg": "true",
        "expected_grf": "true",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_manifest_is_checked_against_registry_metadata_and_signal_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motion = "Darmstadt/D05/config01/trial01/ascent_stageii"
    match = TraceMatch(
        dataset="darmstadt",
        subject="D05",
        motion_type="riser_0.10_ascent",
        condition="riser_0.10",
        direction="ascent",
        motion=motion,
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    monkeypatch.setattr(cohort_module, "resolve_trace_match", lambda *_args, **_kwargs: match)
    monkeypatch.setattr(
        cohort_module,
        "load_trace",
        lambda _path: {"emg_available": np.array(True), "grf_available": np.array(True)},
    )

    cases = load_cohort_manifest(_manifest(tmp_path / "cohort.csv"), artifact_root=tmp_path)

    assert cases == [
        CohortCase(
            case_id="darmstadt-d05-low-ascent",
            motion=motion,
            dataset="darmstadt",
            subject="D05",
            motion_type="riser_0.10_ascent",
            split="test",
            evaluation_tier="primary",
            expected_emg=True,
            expected_grf=True,
            cache_root=tmp_path / "darmstadt/cache",
        )
    ]

    relocated = load_cohort_manifest(
        tmp_path / "cohort.csv", artifact_root=tmp_path, policy_root=tmp_path / "fresh-method"
    )
    assert relocated[0].cache_root == tmp_path / "fresh-method/darmstadt/cache"
    assert relocated[0].motion == cases[0].motion

    with pytest.raises(ValueError, match="does not match registry"):
        load_cohort_manifest(_manifest(tmp_path / "bad.csv", subject="D08"), artifact_root=tmp_path)


def _report(case: CohortCase, checkpoint: Path, output: Path, value: float) -> dict:
    return {
        "motion": case.motion,
        "checkpoint": str(checkpoint),
        "policy_mode": "deterministic",
        "seeds": [0],
        "summary": {
            "successful_rollouts": 1,
            "failed_rollouts": 0,
            "measured_rollouts": 1,
            "unmeasured_rollouts": 0,
            "measured_early_terminated_rollouts": 0,
            "completed_gaits": 2,
            "completed_gaits_in_early_terminated_rollouts": 0,
            "mean_emg_zero_lag_waveform_correlation": value,
            "median_emg_peak_phase_error_percent": 10.0,
            "mean_normal_grf_waveform_correlation": value + 0.1,
            "mean_normal_grf_rmse_bw": 0.2,
            "mean_normal_grf_rmse_percent_body_weight": 20.0,
            "mean_impulse_error_bw_phase": 0.1,
            "mean_impulse_relative_error_percent": 10.0,
        },
        "artifacts": {"summary": str(output / "summary.json")},
    }


def test_cohort_runner_is_failure_isolated_and_writes_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint_100"
    checkpoint.mkdir()
    cases = [
        CohortCase(
            case_id=f"case-{index}",
            motion=f"Darmstadt/D0{index}/motion",
            dataset="darmstadt",
            subject=f"D0{index}",
            motion_type="riser_0.10_ascent",
            split="test",
            evaluation_tier="primary",
            expected_emg=True,
            expected_grf=True,
            cache_root=tmp_path / "cache",
        )
        for index in (1, 2)
    ]
    monkeypatch.setattr(cohort_module, "load_cohort_manifest", lambda *_args, **_kwargs: cases)

    def compare(motion, checkpoint_path, *, output_dir, **_kwargs):
        case = next(value for value in cases if value.motion == motion)
        if case.case_id == "case-2":
            raise NoSuccessfulRolloutError(1)
        return _report(case, Path(checkpoint_path).resolve(), Path(output_dir), 0.4)

    monkeypatch.setattr(cohort_module, "compare_checkpoint", compare)
    (tmp_path / "manifest.csv").write_text("fixture manifest\n", encoding="utf-8")

    report = compare_cohort(
        tmp_path / "manifest.csv",
        checkpoint,
        output_dir=tmp_path / "output",
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        trials=1,
        deterministic=True,
    )

    assert report["completed_cases"] == 1
    assert report["failed_cases"] == 1
    with (tmp_path / "output/cases.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["complete", "failed"]
    assert rows[1]["error"] == "no successful rollout completed; physiological metrics cannot be computed"
    assert rows[1]["successful_rollouts"] == "0"
    assert rows[1]["failed_rollouts"] == "1"
    with (tmp_path / "output/aggregates.csv").open(newline="") as handle:
        overall = next(csv.DictReader(handle))
    assert overall["cases"] == "2"
    assert overall["completed_cases"] == "1"
    assert overall["successful_rollouts"] == "1"
    assert overall["failed_rollouts"] == "1"
    assert overall["measured_rollouts"] == "1"
    assert overall["unmeasured_rollouts"] == "1"
    assert overall["completed_gaits"] == "2"
    assert float(overall["mean_normal_grf_waveform_correlation"]) == pytest.approx(0.5)
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["checkpoint"] == str(checkpoint.resolve())
