"""Command-line entry points for registered terrain reconstruction methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import load_selection, run_cohort
from .registry import available_methods, create_method


def _command_help() -> str:
    return """usage: terra reconstruct COMMAND [ARGS ...]

Terrain-reconstruction commands:
  cohort --method METHOD ...   fit one current manifest with one registered method
  matrix [OPTIONS]             plan or execute the configured multi-dataset matrix

Use `terra reconstruct COMMAND --help` for command-specific options.
"""


def command_main(argv: list[str] | None = None) -> int:
    """Dispatch the installed ``terra reconstruct`` command."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_command_help(), end="")
        return 0
    command, *remaining = arguments
    if command == "cohort":
        return cohort_main(remaining)
    if command == "matrix":
        from .matrix import matrix_main

        return matrix_main(remaining)
    raise SystemExit(f"unknown reconstruction command {command!r}; expected cohort or matrix")


def _json_object(value: str) -> dict[str, Any]:
    try:
        # Parse an inline object before treating it as a path. Calling stat on a long JSON
        # string can raise ENAMETOOLONG before json.loads ever sees the value.
        parsed = json.loads(value) if value.lstrip().startswith("{") else json.loads(Path(value).expanduser().read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError(f"options must be a JSON object or JSON file: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("options must contain a JSON object")
    return parsed


def run_registered(
    method_name: str,
    motions_path: Path,
    output_dir: Path,
    *,
    dataset_config: Path | None = None,
    options: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> int:
    selection = load_selection(motions_path)
    method = create_method(
        method_name,
        motions=selection.motions,
        dataset_config=dataset_config,
        options=options,
    )
    return run_cohort(
        method,
        motions_path,
        output_dir,
        overwrite=overwrite,
        dataset_config_path=dataset_config,
    ).exit_code


def cohort_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra reconstruct cohort", description=__doc__)
    parser.add_argument("--method", required=True, choices=available_methods())
    parser.add_argument("--motions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--options", type=_json_object, default={})
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run_registered(
            args.method,
            args.motions,
            args.output_dir,
            dataset_config=args.dataset_config,
            options=args.options,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2
