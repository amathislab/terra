"""Create a deterministic stratified training subset from a versioned selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from terra.training_subset import (
    DEFAULT_SUBSET_SEED,
    publish_training_subset,
    select_stratified_training_subset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra train subset", description=__doc__)
    parser.add_argument("--selection", required=True, type=Path, help="source versioned selection CSV")
    parser.add_argument("--target-train-entries", required=True, type=int)
    parser.add_argument("--seed", default=DEFAULT_SUBSET_SEED)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        source = args.selection.expanduser().resolve()
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected, audit = select_stratified_training_subset(
            rows,
            target_entries=args.target_train_entries,
            seed=args.seed,
        )
        payload = publish_training_subset(
            args.selection,
            args.out,
            args.audit_out,
            selected,
            audit,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2))
    return 0


__all__ = ["main"]
