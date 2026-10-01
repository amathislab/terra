"""Tests for the package-owned reconstruction registry and cohort contract."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from terra.benchmarking.reconstruction import (
    RECONSTRUCTION_METHODS,
    available_methods,
    cli,
    create_method,
    run_cohort,
)
from terra.benchmarking.reconstruction.core import (
    PreparedMotion,
    ReconstructionResult,
    load_selection,
)


def _selection(path: Path, *motions: str) -> Path:
    path.write_text("dataset,motion\n" + "".join(f"test,{motion}\n" for motion in motions))
    return path


@dataclass
class FakeMethod:
    fail_motion: str | None = None
    value: int = 1
    validation_passed: bool = True

    name = "fake"
    display_name = "Fake reconstruction"
    description = "Deterministic test plugin"

    @property
    def options(self) -> dict[str, Any]:
        return {"value": self.value}

    def prepare(self, motion: str) -> PreparedMotion:
        return PreparedMotion({"motion": motion})

    def fit(self, motion, prepared):
        if motion == self.fail_motion:
            raise RuntimeError("deliberate fit failure")
        return ReconstructionResult(
            terrain={"boxes": [], "provenance": {"test": True}},
            fit={"value": self.value},
            validation={
                "passed": self.validation_passed,
                "influenced_reconstruction": False,
            },
        )

    def summarize(self, result):
        return {"value": result.fit["value"]}


def test_registry_is_the_complete_supported_method_inventory():
    assert available_methods() == ("terra",)
    with pytest.raises(ValueError, match="unknown reconstruction method"):
        create_method("unknown", motions=("Study/A",))


def test_registry_labels_terra():
    assert RECONSTRUCTION_METHODS["terra"].display_name == "TERRA"


def test_cohort_cli_accepts_a_long_inline_options_object():
    payload = '{"config":{"max_merge_candidate_pairs":100000000},"padding":"' + "x" * 300 + '"}'

    assert cli._json_object(payload)["config"]["max_merge_candidate_pairs"] == 100_000_000


def test_selection_requires_unique_canonical_collision_free_motion_ids(tmp_path):
    valid = load_selection(_selection(tmp_path / "valid.csv", "Study/A", "Study/B"))
    assert valid.motions == ("Study/A", "Study/B")

    for name, motions, match in (
        ("alias.csv", ("Study/A.npz",), "already be canonical"),
        ("duplicate.csv", ("Study/A", "Study/A"), "duplicate"),
        ("collision.csv", ("A/B__C", "A__B/C"), "flatten"),
        ("case.csv", ("Study/A", "study/a"), "flatten"),
    ):
        with pytest.raises(ValueError, match=match):
            load_selection(_selection(tmp_path / name, *motions))


def test_cohort_publishes_one_schema_checkpoints_failures_and_preserves_order(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A", "Study/Bad", "Study/B")
    output = tmp_path / "out"
    result = run_cohort(FakeMethod(fail_motion="Study/Bad"), selection, output, progress=False)

    assert result.exit_code == 2
    with result.status_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["motion"] for row in rows] == ["Study/A", "Study/Bad", "Study/B"]
    assert [row["status"] for row in rows] == ["ok", "failed", "ok"]
    assert "RuntimeError: deliberate fit failure" in rows[1]["error"]
    assert not (output / "Study__Bad.json").exists()

    record = json.loads((output / "Study__A.json").read_text())
    assert set(record) == {
        "method",
        "method_display_name",
        "motion",
        "terrain",
        "fit",
        "validation",
    }
    assert set(rows[0]) == {
        "motion",
        "method",
        "status",
        "output",
        "elapsed_seconds",
        "error",
        "summary_json",
    }
    run = json.loads((output / "run.json").read_text())
    assert run["counts"] == {"ok": 2, "failed": 1}
    assert len((output / "GIT_COMMIT").read_text().strip()) == 40


def test_metric_threshold_does_not_mark_a_successful_method_as_failed(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    output = tmp_path / "out"

    result = run_cohort(
        FakeMethod(validation_passed=False),
        selection,
        output,
        progress=False,
    )

    assert result.exit_code == 0
    assert result.records[0]["status"] == "ok"
    assert result.records[0]["error"] == ""
    record = json.loads((output / "Study__A.json").read_text())
    assert record["validation"]["passed"] is False
    run = json.loads((output / "run.json").read_text())
    assert run["counts"] == {"ok": 1, "failed": 0}


def test_rerun_recomputes_output_and_failed_fit_preserves_existing_record(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    output = tmp_path / "out"
    first = run_cohort(FakeMethod(value=1), selection, output, progress=False)
    original = (output / "Study__A.json").read_bytes()
    assert first.exit_code == 0

    failed_rerun = run_cohort(
        FakeMethod(fail_motion="Study/A", value=2),
        selection,
        output,
        progress=False,
    )
    assert failed_rerun.exit_code == 2
    assert (output / "Study__A.json").read_bytes() == original

    replaced = run_cohort(FakeMethod(value=2), selection, output, progress=False)
    assert replaced.exit_code == 0
    assert json.loads((output / "Study__A.json").read_text())["fit"]["value"] == 2
    assert json.loads((output / "run.json").read_text())["options"] == {"value": 2}


def test_cohort_cli_forwards_registered_method_inputs(monkeypatch, tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return 17

    monkeypatch.setattr(cli, "run_registered", run)
    assert (
        cli.cohort_main(
            [
                "--method",
                "terra",
                "--motions",
                str(selection),
                "--dataset-config",
                str(tmp_path / "dataset.toml"),
                "--output-dir",
                str(tmp_path / "terra"),
            ]
        )
        == 17
    )
    assert calls[0][0][0] == "terra"
