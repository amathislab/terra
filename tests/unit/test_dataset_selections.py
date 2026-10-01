from __future__ import annotations

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
                "amass-nonflat",
                "--all-motions",
                "--dry-run",
            ]
        )
