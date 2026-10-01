"""Tests for the executable stable-API example shipped with the package."""

from __future__ import annotations

import json

from examples.retargeting import retarget_and_publish as example
from terra import RetargetArtifacts


def test_example_retargets_c3d_and_prints_published_artifacts(monkeypatch, tmp_path, capsys):
    source = tmp_path / "trial.c3d"
    source.touch()
    output_root = tmp_path / "published"
    smplh = tmp_path / "smplh"
    smplx = tmp_path / "smplx"
    result = object()
    captured = {}

    def fake_retarget(path, **kwargs):
        captured.update(source=path, kwargs=kwargs)
        return result

    def fake_save(value, root, name):
        captured.update(result=value, root=root, name=name)
        return RetargetArtifacts(
            motion_name=name,
            trajectory_path=root / "Trial.npz",
            analysis_path=root / "Trial_analysis.npz",
            terrain_path=root / "Trial_terrain.json",
        )

    monkeypatch.setattr(example, "retarget", fake_retarget)
    monkeypatch.setattr(example, "save_retarget_result", fake_save)

    assert (
        example.main(
            [
                str(source),
                "--output-root",
                str(output_root),
                "--name",
                "Study/Trial",
                "--smpl-model-path",
                str(smplh),
                "--c3d-model-path",
                str(smplx),
            ]
        )
        == 0
    )

    assert captured["source"] == source.resolve()
    assert captured["kwargs"] == {
        "method": "terra",
        "terrain": "auto",
        "smpl_model_path": smplh,
        "c3d_model_path": smplx,
        "cache_root": output_root.resolve() / ".terra-work",
    }
    assert captured["result"] is result
    assert captured["root"] == output_root.resolve()
    assert captured["name"] == "Study/Trial"
    payload = json.loads(capsys.readouterr().out)
    assert payload["motion_name"] == "Study/Trial"
    assert payload["terrain_path"].endswith("Trial_terrain.json")
