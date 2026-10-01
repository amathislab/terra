"""Installed, lazy command dispatch for the supported TERRA workflows."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Protocol, cast


class _Command(Protocol):
    def __call__(self, argv: Sequence[str] | None = None) -> int: ...


_COMMANDS = {
    "convert": (
        "terra.datasets.cli",
        "main",
        "convert a supported source dataset to SMPL-H motions",
    ),
    "biomechanics": (
        "terra.datasets.biomechanics_averages",
        "main",
        "build EMG/GRF traces or compare them with a policy rollout",
    ),
    "retarget": ("terra.cli", "main", "retarget and publish one SMPL-H, C3D, TRC, or MAT motion"),
    "evaluate": (
        "terra.evaluation.cli",
        "main",
        "evaluate explicit method-motion cohorts with the authoritative metrics",
    ),
    "benchmark": (
        "terra.evaluation.benchmark",
        "main",
        "assemble completed retargeting evaluations into one benchmark",
    ),
    "reconstruct": (
        "terra.benchmarking.reconstruction.cli",
        "command_main",
        "fit terrain with one method or a declared cohort matrix",
    ),
    "run": ("terra.commands.run", "main", "run a configured converted-motion dataset"),
    "visualize": ("terra.visualization.cli", "main", "render published artifacts for review"),
    "figure": (
        "terra.figures.cli",
        "main",
        "validate and render manifest-driven publication figures",
    ),
    "train": ("", "", "train a policy, run preflight, or materialize a cohort"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terra",
        description=(
            "Dataset conversion, terrain reconstruction, retargeting, evaluation, visualization, "
            "and policy-training workflows."
        ),
    )
    parser.add_argument("--version", action="store_true", help="print the installed package version")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_module, _function, description) in _COMMANDS.items():
        subcommands.add_parser(name, add_help=False, help=description)
    return parser


def _invoke(module_name: str, function_name: str, argv: Sequence[str]) -> int:
    function = cast(_Command, getattr(importlib.import_module(module_name), function_name))
    return int(function(argv))


def _train_help() -> str:
    return """usage: terra train COMMAND [ARGS ...]

Policy-training commands:
  run [OPTIONS]             launch PPO training
  preflight [OPTIONS]        validate algorithm config, JAX, Warp, and CUDA readiness
  select [OPTIONS]           verify current TERRA runs and build an ordered selection
  segment [OPTIONS]          split selection clips longer than a duration threshold
  retarget-selection [...]   build a baseline selection from audited artifact caches
  baseline-selections [...] build matched clipped selections for retargeting methods
  subset [OPTIONS]           select a stratified train cohort without splitting clips
  materialize [OPTIONS]      build one verified cross-dataset training cache

Use `terra train COMMAND --help` for command-specific options.
"""


def _train(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(_train_help(), end="")
        return 0
    command, *remaining = argv
    if command == "run":
        return _invoke("terra.training", "run_main", remaining)
    if command == "preflight":
        return _invoke("terra.training", "main", remaining)
    if command == "select":
        return _invoke("terra.commands.selection", "main", remaining)
    if command == "segment":
        return _invoke("terra.commands.segment", "main", remaining)
    if command == "retarget-selection":
        return _invoke("terra.commands.retarget_training_selection", "main", remaining)
    if command == "baseline-selections":
        return _invoke("terra.commands.baseline_training", "main", remaining)
    if command == "subset":
        return _invoke("terra.commands.subset", "main", remaining)
    if command == "materialize":
        return _invoke("terra.commands.materialize", "main", remaining)
    print(f"terra train: unknown command {command!r}", file=sys.stderr)
    print(_train_help(), file=sys.stderr, end="")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the installed ``terra`` command without importing heavy workflows."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments:
        parser.print_help()
        return 0
    if arguments == ["-h"] or arguments == ["--help"]:
        parser.print_help()
        return 0
    namespace, remaining = parser.parse_known_args(arguments)
    if namespace.version:
        from terra import __version__

        print(__version__)
        return 0
    if namespace.command is None:
        parser.error("a command is required")
    if namespace.command == "train":
        return _train(remaining)
    module_name, function_name, _description = _COMMANDS[namespace.command]
    return _invoke(module_name, function_name, remaining)


__all__ = ["main"]
