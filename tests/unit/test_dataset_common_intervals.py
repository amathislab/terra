"""Dataset evaluation must preserve frozen scoring windows across CLI stages."""

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


def _metric_runner(monkeypatch, *, recorded="forwarded"):
    observed = []

    def run(argv):
        # Exercise the actual receiving parser, including its Path conversion.
        args = cli.parser().parse_args(argv)
        observed.append(args)
        options = {
            "allow_missing": args.allow_missing,
            "allow_flat_name_conflicts": args.allow_flat_name_conflicts,
            "limit_per_class": args.limit,
            "terrain_method": args.terrain_method,
        }
        if recorded != "absent":
            options["common_intervals"] = (
                (str(args.common_intervals.resolve()) if args.common_intervals is not None else None)
                if recorded == "forwarded"
                else recorded
            )
        args.out.mkdir(parents=True)
        (args.out / "run.json").write_text(
            json.dumps(
                {
                    "methods": dict(value.split("=", 1) for value in args.method),
                    "manifests": dict(value.split("=", 1) for value in args.motion_class),
                    "options": options,
                }
            )
        )
        return 0

    monkeypatch.setattr(cli, "metrics_main", run)
    return observed


def test_frozen_intervals_forwarded_and_recorded(evaluation, monkeypatch, capsys):
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


@pytest.mark.parametrize("recorded", [None, "absent", "/different/intervals.csv"])
def test_requested_intervals_must_match_metric_run(evaluation, monkeypatch, recorded):
    argv, intervals, output = evaluation
    _metric_runner(monkeypatch, recorded=recorded)
    assert dataset.main([*argv, "--common-intervals", str(intervals)]) == 2
    stage = json.loads((output / "evaluation.json").read_text())["stages"]["metrics"]
    assert stage["exit_code"] == 2
    assert "options do not match" in stage["error"]


@pytest.mark.parametrize("recorded", [None, "absent"])
def test_default_preserves_schema_and_accepts_legacy_metadata(evaluation, monkeypatch, capsys, recorded):
    argv, _, output = evaluation
    observed = _metric_runner(monkeypatch, recorded=recorded)
    assert dataset.main([*argv, "--dry-run"]) == 0
    assert "common_intervals" not in json.loads(capsys.readouterr().out)
    assert dataset.main(argv) == 0
    assert observed[0].common_intervals is None
    assert "common_intervals" not in json.loads((output / "evaluation.json").read_text())


def test_unrequested_intervals_are_rejected(evaluation, monkeypatch):
    argv, intervals, output = evaluation
    _metric_runner(monkeypatch, recorded=str(intervals))
    assert dataset.main(argv) == 2
    assert json.loads((output / "evaluation.json").read_text())["stages"]["metrics"]["exit_code"] == 2


def test_missing_intervals_fail_before_metrics(evaluation, monkeypatch):
    argv, intervals, output = evaluation
    observed = _metric_runner(monkeypatch)
    intervals.unlink()
    with pytest.raises(SystemExit, match="common scoring intervals file does not exist"):
        dataset.main([*argv, "--common-intervals", str(intervals)])
    assert not observed
    assert not output.exists()
