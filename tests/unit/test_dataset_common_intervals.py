"""Dataset evaluation must preserve fixed scoring windows across CLI stages."""

import json

import pytest

from terra.evaluation import cli, dataset


@pytest.fixture
def evaluation(tmp_path):
    config = tmp_path / "dataset.toml"
    config.write_text("""schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "none"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("motion,dataset,terrain_class,passed\nExample/Chair,example,seat,1\n")
    intervals = tmp_path / "intervals.csv"
    intervals.write_text("motion,common_start_s,common_end_s\nExample/Chair,0.02,1.48\n")
    output = tmp_path / "evaluation"
    argv = [str(config), "--manifest", str(manifest), "--output-root", str(output), "--method", "GMR=gmr"]
    return argv, intervals, output


def _metric_runner(monkeypatch):
    observed = []

    def run(argv):
        observed.append(cli.parser().parse_args(argv))
        return 0

    monkeypatch.setattr(cli, "metrics_main", run)
    return observed


def test_common_intervals_forwarded_and_recorded(evaluation, monkeypatch, capsys):
    argv, intervals, output = evaluation
    monkeypatch.chdir(intervals.parent)
    observed = _metric_runner(monkeypatch)
    argv += ["--common-intervals", intervals.name]
    assert dataset.main([*argv, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["common_intervals"] == str(intervals)
    assert not output.exists()
    assert dataset.main(argv) == 0
    assert observed[0].common_intervals == intervals
    assert cli.read_common_intervals(observed[0].common_intervals) == {"Example/Chair": (0.02, 1.48)}
    metadata = json.loads((output / "evaluation.json").read_text())
    assert metadata["common_intervals"] == str(intervals)
    assert metadata["stages"]["metrics"]["exit_code"] == 0


def test_default_does_not_request_common_intervals(evaluation, monkeypatch, capsys):
    argv, _, output = evaluation
    observed = _metric_runner(monkeypatch)
    assert dataset.main([*argv, "--dry-run"]) == 0
    assert "common_intervals" not in json.loads(capsys.readouterr().out)
    assert dataset.main(argv) == 0
    assert observed[0].common_intervals is None
    assert "common_intervals" not in json.loads((output / "evaluation.json").read_text())


def test_missing_intervals_fail_before_metrics(evaluation, monkeypatch):
    argv, intervals, output = evaluation
    observed = _metric_runner(monkeypatch)
    intervals.unlink()
    with pytest.raises(SystemExit, match="common scoring intervals file does not exist"):
        dataset.main([*argv, "--common-intervals", str(intervals)])
    assert not observed
    assert not output.exists()
