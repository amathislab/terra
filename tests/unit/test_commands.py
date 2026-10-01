"""Tests for the installed package-owned command surface."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from terra import commands

REPO = Path(__file__).resolve().parents[2]


def test_root_help_lists_supported_commands_without_loading_workflows(monkeypatch, capsys) -> None:
    imported = []
    monkeypatch.setattr(commands.importlib, "import_module", lambda name: imported.append(name))

    assert commands.main(["--help"]) == 0

    output = capsys.readouterr().out
    assert all(
        name in output
        for name in (
            "convert",
            "retarget",
            "reconstruct",
            "run",
            "evaluate",
            "visualize",
            "train",
        )
    )
    assert imported == []


@pytest.mark.parametrize(
    ("arguments", "expected_module", "expected_function", "expected_arguments"),
    (
        (
            ["convert", "gait120", "--audit-only"],
            "terra.datasets.cli",
            "main",
            ["gait120", "--audit-only"],
        ),
        (
            ["retarget", "motion.npz", "--output-root", "runs/out"],
            "terra.cli",
            "main",
            ["motion.npz", "--output-root", "runs/out"],
        ),
        (
            ["reconstruct", "cohort", "--method", "terra"],
            "terra.benchmarking.reconstruction.cli",
            "command_main",
            ["cohort", "--method", "terra"],
        ),
        (
            ["evaluate", "metrics", "--motion-class", "all=cohort.csv"],
            "terra.evaluation.cli",
            "main",
            ["metrics", "--motion-class", "all=cohort.csv"],
        ),
        (["run", "amass", "--dry-run"], "terra.commands.run", "main", ["amass", "--dry-run"]),
        (["visualize", "--manifest", "cohort.csv"], "terra.visualization.cli", "main", ["--manifest", "cohort.csv"]),
        (["train", "run", "Study/Trial", "--dry-run"], "terra.training", "run_main", ["Study/Trial", "--dry-run"]),
        (["train", "preflight", "--allow-no-device"], "terra.training", "main", ["--allow-no-device"]),
        (["train", "select", "--help"], "terra.commands.selection", "main", ["--help"]),
        (["train", "segment", "--help"], "terra.commands.segment", "main", ["--help"]),
        (["train", "materialize", "--help"], "terra.commands.materialize", "main", ["--help"]),
    ),
)
def test_root_dispatch_is_lazy_and_forwards_arguments(
    monkeypatch,
    arguments,
    expected_module,
    expected_function,
    expected_arguments,
) -> None:
    calls = []

    def imported(name):
        calls.append(("import", name))

        def command(argv):
            calls.append((expected_function, list(argv)))
            return 17

        return SimpleNamespace(**{expected_function: command})

    monkeypatch.setattr(commands.importlib, "import_module", imported)

    assert commands.main(arguments) == 17
    assert calls == [
        ("import", expected_module),
        (expected_function, expected_arguments),
    ]


def test_train_help_describes_nested_commands_without_importing_training(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        commands.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )

    assert commands.main(["train", "--help"]) == 0

    output = capsys.readouterr().out
    assert "run [OPTIONS]" in output
    assert "PPO" in output
    assert "preflight" in output
    assert "select" in output
    assert "segment" in output
    assert "materialize" in output


def test_superseded_script_wrappers_are_removed() -> None:
    retired_wrappers = (
        "darmstadt/to_smplh.py",
        "gait120/merge_staging.py",
        "gait120/prepare_smplh.py",
        "gait120/to_smplh.py",
        "vielemeyer/to_smplh.py",
        "terra/_terrain_render.py",
        "terra/materialize_training_subset.py",
        "terra/run_dataset.py",
        "terra/run_terrain_reconstruction_matrix.py",
        "terra/score_terrain_subset.py",
        "terra/render_terrain_subset.py",
        "terra/training_cohort_review_manifest.py",
        "terra/validate_dataset_preflight.py",
    )
    assert all(not (REPO / "scripts" / path).exists() for path in retired_wrappers)

    for path in (
        REPO / "src/terra/commands/run.py",
        REPO / "src/terra/commands/materialize.py",
        REPO / "src/terra/commands/selection.py",
        REPO / "src/terra/visualization/render.py",
        REPO / "src/terra/visualization/cli.py",
        REPO / "src/terra/benchmarking/reconstruction/core.py",
    ):
        source = path.read_text()
        assert "scripts.terra" not in source
        assert "sys.path" not in source


def test_dataset_configs_have_one_package_owned_copy() -> None:
    from terra.datasets.config import DATASET_CONFIG_NAMES, bundled_dataset_config

    for name in DATASET_CONFIG_NAMES:
        assert bundled_dataset_config(name) == REPO / "src/terra/datasets/configs" / f"{name}.toml"
    assert not (REPO / "configs/datasets").exists()
    assert not (REPO / "src/terra/commands/dataset_configs").exists()


def test_wheel_contains_supported_commands_and_runs_root_help(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO / filename, source / filename)
    shutil.copytree(REPO / "src", source / "src")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list((source / "dist").glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        entry_points = archive.read(
            next(name for name in members if name.endswith(".dist-info/entry_points.txt"))
        ).decode()

    expected = {
        "terra/datasets/cli.py",
        "terra/datasets/gait120.py",
        "terra/datasets/darmstadt.py",
        "terra/datasets/vielemeyer.py",
        "terra/datasets/marker_fitting.py",
        "terra/datasets/prism/conversion.py",
        "terra/benchmarking/reconstruction/cli.py",
        "terra/commands/__init__.py",
        "terra/commands/run.py",
        "terra/commands/materialize.py",
        "terra/commands/selection.py",
        "terra/contracts.py",
        "terra/runtime.py",
        "terra/visualization/__init__.py",
        "terra/visualization/render.py",
        "terra/visualization/cli.py",
        "terra/datasets/config.py",
        "terra/datasets/configs/amass.toml",
        "terra/rl/configs/ppo_multi_motion.yaml",
    }
    assert expected <= members
    assert all(not name.startswith("scripts/") for name in members)
    assert "terra = terra.commands:main" in entry_points
    assert "terra-retarget" not in entry_points
    assert "terra-train" not in entry_points

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import terra\n"
                "from terra.commands import main\n"
                "assert '.whl' in terra.__file__\n"
                "assert main(['--help']) == 0\n"
                "for argv in (['run', '--help'], ['visualize', '--help'], "
                "['train', 'select', '--help'], ['train', 'segment', '--help'], "
                "['train', 'materialize', '--help'], "
                "['train', 'submit', '--help'], ['convert', '--help'], "
                "['convert', 'gait120', '--help'], ['convert', 'darmstadt', '--help'], "
                "['convert', 'vielemeyer', '--help'], ['convert', 'prism', '--help'], "
                "['reconstruct', '--help'], ['reconstruct', 'cohort', '--help'], "
                "['evaluate', '--help'], ['evaluate', 'metrics', '--help'], "
                "['evaluate', 'dataset', '--help']):\n"
                "    try:\n"
                "        main(argv)\n"
                "    except SystemExit as error:\n"
                "        assert error.code == 0\n"
            ),
        ],
        cwd=tmp_path,
        env={"PYTHONPATH": str(wheel)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "terra convert gait120" in completed.stdout
    assert "terra convert darmstadt" in completed.stdout
    assert "terra convert vielemeyer" in completed.stdout
    assert "terra convert prism" in completed.stdout
    assert "visualize" in completed.stdout
