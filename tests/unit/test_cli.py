"""Tests for the public retarget-and-publish command."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import terra.api as api
import terra.cli as cli
from terra.api import RetargetArtifacts


def test_c3d_cli_forwards_explicit_assets_and_publishes_named_motion(monkeypatch, tmp_path, capsys):
    source = tmp_path / "Trial 05.c3d"
    source.touch()
    output_root = tmp_path / "published"
    method_config = tmp_path / "method.json"
    c3d_options = tmp_path / "c3d.json"
    method_config.write_text('{"damping": 0.25}')
    c3d_options.write_text('{"stage2_iters": 12}')
    result = SimpleNamespace(terrain=SimpleNamespace(boxes=(object(),)))
    calls = {}

    def fake_retarget(path, **kwargs):
        calls.update(source=path, kwargs=kwargs)
        return result

    def fake_save(value, cache_root, motion_name, **kwargs):
        calls.update(result=value, cache_root=cache_root, motion_name=motion_name, save_kwargs=kwargs)
        return RetargetArtifacts(
            motion_name=str(motion_name),
            trajectory_path=output_root / "MyoFullBody/terra/Trial_05.npz",
            analysis_path=output_root / "MyoFullBody/terra/Trial_05_analysis.npz",
            terrain_path=output_root / "MyoFullBody/terra/Trial_05_terrain.json",
        )

    monkeypatch.setattr(api, "retarget", fake_retarget)
    monkeypatch.setattr(api, "save_retarget_result", fake_save)

    assert (
        cli.main(
            [
                str(source),
                "--output-root",
                str(output_root),
                "--smpl-model-path",
                str(tmp_path / "smplh"),
                "--c3d-model-path",
                str(tmp_path / "smplx"),
                "--config",
                str(method_config),
                "--c3d-options",
                str(c3d_options),
            ]
        )
        == 0
    )

    assert calls["source"] == source.resolve()
    assert calls["motion_name"] == "Trial_05"
    assert calls["kwargs"]["config"] == {"damping": 0.25}
    assert calls["kwargs"]["c3d_options"] == {"stage2_iters": 12, "clear_cache": False}
    assert calls["kwargs"]["cache_root"] == (output_root / ".terra-c3d-cache").resolve()
    assert calls["save_kwargs"] == {"env_name": "MyoFullBody", "overwrite": False}
    assert len((output_root / "GIT_COMMIT").read_text().strip()) == 40
    report = json.loads(capsys.readouterr().out)
    assert report["motion_name"] == "Trial_05"
    assert report["nonflat_terrain"] is True


def test_trc_cli_forwards_vertical_axis_and_marker_options(monkeypatch, tmp_path, capsys):
    source = tmp_path / "Trial 06.trc"
    source.touch()
    output_root = tmp_path / "published"
    result = SimpleNamespace(terrain=None)
    calls = {}

    def fake_retarget(path, **kwargs):
        calls.update(source=path, kwargs=kwargs)
        return result

    monkeypatch.setattr(api, "retarget", fake_retarget)
    monkeypatch.setattr(
        api,
        "save_retarget_result",
        lambda *_args, **_kwargs: RetargetArtifacts(
            motion_name="Trial_06",
            trajectory_path=output_root / "Trial_06.npz",
            analysis_path=output_root / "Trial_06_analysis.npz",
            terrain_path=None,
        ),
    )

    assert (
        cli.main(
            [
                str(source),
                "--output-root",
                str(output_root),
                "--c3d-model-path",
                str(tmp_path / "smplx"),
                "--trc-up-axis",
                "z",
            ]
        )
        == 0
    )

    assert calls["source"] == source.resolve()
    assert calls["kwargs"]["trc_up_axis"] == "z"
    assert calls["kwargs"]["c3d_options"] == {"clear_cache": False}
    assert calls["kwargs"]["cache_root"] == (output_root / ".terra-c3d-cache").resolve()
    assert json.loads(capsys.readouterr().out)["nonflat_terrain"] is False


def test_mat_cli_forwards_schema_and_nested_selectors(monkeypatch, tmp_path, capsys):
    source = tmp_path / "Trial 07.mat"
    source.touch()
    schema = tmp_path / "darmstadt.json"
    schema.write_text('{"positions_path": "markers", "labels": ["LANK"], "fps": 100}')
    output_root = tmp_path / "published"
    result = SimpleNamespace(terrain=None)
    calls = {}

    def fake_retarget(path, **kwargs):
        calls.update(source=path, kwargs=kwargs)
        return result

    monkeypatch.setattr(api, "retarget", fake_retarget)
    monkeypatch.setattr(
        api,
        "save_retarget_result",
        lambda *_args, **_kwargs: RetargetArtifacts(
            motion_name="Trial_07",
            trajectory_path=output_root / "Trial_07.npz",
            analysis_path=output_root / "Trial_07_analysis.npz",
            terrain_path=None,
        ),
    )

    assert (
        cli.main(
            [
                str(source),
                "--output-root",
                str(output_root),
                "--c3d-model-path",
                str(tmp_path / "smplx"),
                "--mat-schema",
                str(schema),
                "--mat-selector",
                "configuration=2",
                "--mat-selector",
                "trial=4",
            ]
        )
        == 0
    )

    assert calls["source"] == source.resolve()
    assert calls["kwargs"]["mat_schema"] == schema
    assert calls["kwargs"]["mat_selectors"] == {"configuration": 2, "trial": 4}
    assert json.loads(capsys.readouterr().out)["nonflat_terrain"] is False


def test_cli_resolves_runs_alias_against_artifact_root(monkeypatch, tmp_path, capsys):
    source = tmp_path / "trial.npz"
    source.touch()
    artifact_root = tmp_path / "external-artifacts"
    expected_root = artifact_root / "published"
    result = SimpleNamespace(terrain=None)
    calls = {}

    monkeypatch.setenv("TERRA_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setattr(api, "retarget", lambda *_args, **_kwargs: result)

    def fake_save(value, cache_root, motion_name, **kwargs):
        calls.update(cache_root=cache_root, motion_name=motion_name, kwargs=kwargs)
        return RetargetArtifacts(
            motion_name=str(motion_name),
            trajectory_path=expected_root / "MyoFullBody/terra/trial.npz",
            analysis_path=expected_root / "MyoFullBody/terra/trial_analysis.npz",
            terrain_path=None,
        )

    monkeypatch.setattr(api, "save_retarget_result", fake_save)

    assert cli.main([str(source), "--output-root", "runs/published"]) == 0

    assert calls["cache_root"] == expected_root
    report = json.loads(capsys.readouterr().out)
    assert report["storage_roots"]["artifact_root"] == str(artifact_root)


def test_json_options_must_be_objects(tmp_path):
    path = tmp_path / "options.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="JSON object"):
        cli._load_json_object(path, "options")


@pytest.mark.parametrize(("value", "expected"), [("AUTO", "auto"), ("None", None), ("terrain.json", "terrain.json")])
def test_terrain_sentinels_are_case_insensitive(value, expected):
    assert cli._terrain_argument(value) == expected


def test_cli_help_documents_public_asset_and_method_contract(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "SMPL-H model root" in help_text
    assert "SMPL-X/SMPL-H marker-fitting root" in help_text
    assert "required for marker input" in normalized_help
    assert "--trc-up-axis {y,z}" in normalized_help
    assert "--mat-schema MAT_SCHEMA" in normalized_help
    assert "--mat-selector NAME=INDEX" in normalized_help
    assert "{terra,omniretarget,gmr,smpl}" in help_text
    assert "auto requires terra or omniretarget" in normalized_help


def test_cli_reports_expected_input_errors_without_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                str(tmp_path / "missing.npz"),
                "--output-root",
                str(tmp_path / "published"),
            ]
        )

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "motion source not found" in stderr
    assert "Traceback" not in stderr


def test_cli_reports_runtime_preflight_errors_without_traceback(monkeypatch, tmp_path, capsys):
    source = tmp_path / "trial.c3d"
    source.touch()

    def fail_retarget(*_args, **_kwargs):
        raise RuntimeError("CUDA PyTorch is unavailable")

    monkeypatch.setattr(api, "retarget", fail_retarget)

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                str(source),
                "--output-root",
                str(tmp_path / "published"),
                "--c3d-model-path",
                str(tmp_path / "smplx"),
            ]
        )

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "CUDA PyTorch is unavailable" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("suffix", (".c3d", ".trc", ".mat"))
def test_cli_requires_explicit_marker_model_root(tmp_path, capsys, suffix):
    source = tmp_path / f"trial{suffix}"
    source.touch()

    with pytest.raises(SystemExit) as raised:
        cli.main([str(source), "--output-root", str(tmp_path / "published")])

    assert raised.value.code == 2
    assert "--c3d-model-path is required for marker input" in capsys.readouterr().err


def test_cli_requires_mat_schema(tmp_path, capsys):
    source = tmp_path / "trial.mat"
    source.touch()

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                str(source),
                "--output-root",
                str(tmp_path / "published"),
                "--c3d-model-path",
                str(tmp_path / "smplx"),
            ]
        )

    assert raised.value.code == 2
    assert "--mat-schema is required" in capsys.readouterr().err


@pytest.mark.parametrize("value", ("trial", "trial=nope"))
def test_mat_selector_requires_name_and_integer_index(value):
    with pytest.raises(Exception, match="MAT selector"):
        cli._mat_selector(value)
