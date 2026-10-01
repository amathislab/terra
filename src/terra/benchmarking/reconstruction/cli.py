"""Command-line entry points for TERRA terrain reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import run_cohort
from .methods.terra import TerraMethod


def command_main(argv: list[str] | None = None) -> int:
    """Dispatch the installed ``terra reconstruct`` command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: terra reconstruct cohort [OPTIONS]\n\nFit terrain for a selected motion set.")
        return 0
    command, *remaining = arguments
    if command == "cohort":
        return cohort_main(remaining)
    raise SystemExit(f"unknown reconstruction command {command!r}; expected cohort")


def run_terrain_cohort(motions_path: Path, output_dir: Path, *, dataset_config: Path) -> int:
    return run_cohort(TerraMethod(dataset_config), motions_path, output_dir).exit_code


def cohort_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra reconstruct cohort", description=__doc__)
    parser.add_argument("--method", choices=("terra",), default="terra")
    parser.add_argument("--motions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return run_terrain_cohort(args.motions, args.output_dir, dataset_config=args.dataset_config)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2
