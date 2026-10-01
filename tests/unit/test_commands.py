"""Tests for the installed bundled command surface."""

from __future__ import annotations

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
            ["convert", "gait120", "--inspect-only"],
            "terra.datasets.cli",
            "main",
            ["gait120", "--inspect-only"],
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


def test_bundled_dataset_configs_are_available() -> None:
    from terra.datasets.config import DATASET_CONFIG_NAMES, bundled_dataset_config

    for name in DATASET_CONFIG_NAMES:
        assert bundled_dataset_config(name).is_file()


@pytest.mark.slow
def test_wheel_contains_supported_commands_and_runs_root_help(tmp_path: Path, release_source: Path) -> None:
    source = release_source

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
