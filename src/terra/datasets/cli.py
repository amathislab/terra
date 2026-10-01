"""Lazy command dispatch for dataset-specific source-to-SMPL-H converters."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Protocol, cast


class _Converter(Protocol):
    def __call__(self, argv: Sequence[str] | None = None) -> int: ...


_CONVERTERS = {
    "gait120": ("terra.datasets.gait120", "Gait120 TRC/MAT release"),
    "darmstadt": ("terra.datasets.darmstadt", "Darmstadt stair MATLAB release"),
    "vielemeyer": ("terra.datasets.vielemeyer", "Vielemeyer ramp C3D release"),
    "prism": ("terra.datasets.prism.conversion", "PRISM fitted-SMPL release"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terra convert",
        description=(
            "Convert a supported dataset's published source format to SMPL-H and write "
            "a manifest. MAT conversion is dataset-specific."
        ),
    )
    subcommands = parser.add_subparsers(dest="dataset", metavar="DATASET")
    for name, (_module, description) in _CONVERTERS.items():
        subcommands.add_parser(name, add_help=False, help=description)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one package-owned dataset converter without importing the others."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    namespace, remaining = parser.parse_known_args(arguments)
    if namespace.dataset is None:
        parser.error("a supported dataset is required")
    module_name, _description = _CONVERTERS[namespace.dataset]
    converter = cast(_Converter, importlib.import_module(module_name).main)
    return int(converter(remaining))


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
