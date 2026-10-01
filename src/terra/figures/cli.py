"""Command-line entry point for reproducible publication figures."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path


def _environment() -> None:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/terra-matplotlib")
    for variable in ("LP_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terra figure",
        description="Validate or render a versioned publication-figure manifest.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="resolve every required frozen artifact")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--cache-root", type=Path, required=True)
    render = subcommands.add_parser("render", help="render one manifest through the selected backend")
    render.add_argument("manifest", type=Path)
    render.add_argument("--cache-root", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--backend", choices=("mujoco", "blender"), default="mujoco")
    render.add_argument(
        "--blender-executable",
        default=os.environ.get("BLENDER", "blender"),
        help="Blender executable or path (only used by the blender backend)",
    )
    render.add_argument("--scale", type=float, default=1.0, help="resolution multiplier for previews")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from terra.figures.manifest import load_manifest

    manifest = load_manifest(arguments.manifest)
    _environment()
    from terra.figures.mujoco_renderer import render_figure as render_mujoco
    from terra.figures.mujoco_renderer import validate_artifacts

    if arguments.command == "validate":
        rows = validate_artifacts(manifest, arguments.cache_root)
        print(json.dumps({"figure": manifest.name, "cells": rows}, indent=2))
        return 0
    if arguments.backend == "blender":
        from terra.figures.blender_renderer import render_figure as render_blender

        summary = render_blender(
            manifest,
            cache_root=arguments.cache_root,
            output=arguments.out,
            blender_executable=arguments.blender_executable,
            scale=arguments.scale,
        )
    else:
        summary = render_mujoco(
            manifest,
            cache_root=arguments.cache_root,
            output=arguments.out,
            scale=arguments.scale,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
