import json

import pytest

from terra.evaluation.reconstruction_diff import compare, main


def _write_record(root, motion, *, terrain=None, fit=None, validation=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{motion.replace('/', '__')}.json"
    path.write_text(
        json.dumps(
            {
                "motion": motion,
                "terrain": {} if terrain is None else terrain,
                "fit": {} if fit is None else fit,
                "validation": {} if validation is None else validation,
            }
        )
    )


def test_reconstruction_diff_requires_exact_scientific_equality(tmp_path):
    manifest = tmp_path / "selection.csv"
    manifest.write_text("motion\nStudy/one\nStudy/two\n")
    before = tmp_path / "before"
    after = tmp_path / "after"
    for motion in ("Study/one", "Study/two"):
        _write_record(before, motion, terrain={"boxes": []}, fit={"model": "flat"}, validation={"passed": True})
        _write_record(after, motion, terrain={"boxes": []}, fit={"model": "flat"}, validation={"passed": True})

    report = compare(manifest, before, after)

    assert report["selected"] == 2
    assert report["identical"] == 2
    assert report["changed"] == 0
    assert report["errors"] == 0


def test_reconstruction_diff_fails_on_any_changed_or_missing_record(tmp_path):
    manifest = tmp_path / "selection.csv"
    manifest.write_text("motion\nStudy/changed\nStudy/missing\n")
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_record(before, "Study/changed", terrain={"boxes": []})
    _write_record(after, "Study/changed", terrain={"boxes": [{"name": "unexpected"}]})
    _write_record(before, "Study/missing")
    output = tmp_path / "audit"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--baseline-dir",
            str(before),
            "--candidate-dir",
            str(after),
            "--output",
            str(output),
        ]
    )

    report = json.loads((output / "report.json").read_text())
    assert exit_code == 1
    assert report["changed"] == 1
    assert report["errors"] == 1
    assert report["structurally_changed"] == 1
    assert report["numeric_only_changed"] == 0
    assert report["changed_records"][0]["motion"] == "Study/changed"
    assert report["changed_records"][0]["fields"] == ["terrain"]
    assert report["changed_records"][0]["differences"]["terrain"]["structural_differences"] == 1


def test_reconstruction_diff_can_report_changes_without_failing(tmp_path):
    manifest = tmp_path / "selection.csv"
    manifest.write_text("motion\nStudy/changed\n")
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_record(before, "Study/changed", terrain={"boxes": []})
    _write_record(after, "Study/changed", terrain={"boxes": [{"name": "expected-change"}]})
    output = tmp_path / "audit"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--baseline-dir",
            str(before),
            "--candidate-dir",
            str(after),
            "--output",
            str(output),
            "--allow-changes",
        ]
    )

    assert exit_code == 0
    assert json.loads((output / "report.json").read_text())["changed"] == 1


def test_reconstruction_diff_allow_changes_still_fails_on_missing_records(tmp_path):
    manifest = tmp_path / "selection.csv"
    manifest.write_text("motion\nStudy/missing\n")
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_record(before, "Study/missing")

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--baseline-dir",
            str(before),
            "--candidate-dir",
            str(after),
            "--output",
            str(tmp_path / "audit"),
            "--allow-changes",
        ]
    )

    assert exit_code == 1


def test_reconstruction_diff_reports_numerical_only_drift(tmp_path):
    manifest = tmp_path / "selection.csv"
    manifest.write_text("motion\nStudy/numerical\n")
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_record(before, "Study/numerical", fit={"height": 0.5, "samples": [1.0]})
    _write_record(after, "Study/numerical", fit={"height": 0.50000001, "samples": [1.0]})

    report = compare(manifest, before, after)

    assert report["changed"] == 1
    assert report["numeric_only_changed"] == 1
    assert report["structurally_changed"] == 0
    assert report["field_totals"]["fit"]["numeric_differences"] == 1
    assert report["field_totals"]["fit"]["max_abs_numeric_delta"] == pytest.approx(1e-8)
    difference = report["changed_records"][0]["differences"]["fit"]
    assert difference["max_abs_numeric_delta_path"] == "$.height"
