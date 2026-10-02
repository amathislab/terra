"""Command-line entry points for TERRA terrain reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from terra._files import atomic_write
from terra.datasets.config import resolve_dataset_config

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


def run_terrain_cohort(
    motions_path: Path | None,
    output_dir: Path,
    *,
    dataset_config: Path,
    cache_root: Path | None = None,
    model_root: Path | None = None,
    motion_paths: tuple[Path, ...] = (),
) -> int:
    method = TerraMethod(dataset_config, cache_root=cache_root, model_root=model_root, motion_paths=motion_paths)
    if motions_path is None:
        motions_path = output_dir / "selection.txt"
        selection = "".join(motion + "\n" for motion in method._records)
        atomic_write(motions_path, lambda temporary: temporary.write_text(selection))
    return run_cohort(method, motions_path, output_dir).exit_code


def cohort_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra reconstruct cohort", description=__doc__)
    parser.add_argument("--method", choices=("terra",), default="terra")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--motions", type=Path, help="CSV/TXT motion selection")
    selection.add_argument(
        "--motion", action="append", type=Path, help="SMPL-H archive below the dataset input root; repeatable"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-config", default=Path("amass"), type=Path, help="bundled dataset name or TOML path")
    parser.add_argument("--cache-root", type=Path, help="reuse a retargeting body-shape cache")
    parser.add_argument("--smpl-model-path", type=Path, help="override the neutral SMPL-H model root")
    args = parser.parse_args(argv)
    try:
        options = {"dataset_config": resolve_dataset_config(args.dataset_config)}
        if args.cache_root is not None:
            options["cache_root"] = args.cache_root
        if args.smpl_model_path is not None:
            options["model_root"] = args.smpl_model_path
        if args.motion:
            options["motion_paths"] = tuple(args.motion)
        return run_terrain_cohort(args.motions, args.output_dir, **options)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2
