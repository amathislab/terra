from __future__ import annotations

import json

import pytest

from terra.commands.run import main as run_main


def test_amass_requires_an_explicit_scope() -> None:
    with pytest.raises(SystemExit, match="pass --selection-manifest, --motion, or --all-motions"):
        run_main(["amass", "--dry-run"])


def test_selection_and_all_motions_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        run_main(
            [
                "amass",
                "--selection-manifest",
                "selection.csv",
                "--all-motions",
                "--dry-run",
            ]
        )


@pytest.mark.parametrize(
    "filename,content",
    [
        ("selection.txt", "Study/Subject/walk\n"),
        ("selection.csv", "motion\nStudy/Subject/walk\n"),
    ],
)
def test_run_resolves_an_explicit_selection_file(tmp_path, capsys, filename, content):
    source = tmp_path / "inputs/Study/Subject/walk.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    selection = tmp_path / filename
    selection.write_text(content)
    assert (
        run_main(
            [
                "amass",
                "--input-root",
                str(tmp_path / "inputs"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--selection-manifest",
                str(selection),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["motions"] == 1
    assert payload["preview"][0]["motion"] == "Study/Subject/walk"

    assert payload["cache_root"] == str(tmp_path / "cache")
    assert payload["reference_cache_root"] == payload["cache_root"]
