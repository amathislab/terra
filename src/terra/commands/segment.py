"""Split only training-selection clips longer than a configured duration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from terra._methods import SUPPORTED_METHODS
from terra.commands.materialize import read_manifest
from terra.paths import StorageRoots
from terra.training_segments import expand_training_segments, publish_segmented_selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra train segment", description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument(
        "--retargeting-method",
        choices=SUPPORTED_METHODS,
        default="terra",
        help="artifact namespace to validate and segment",
    )
    parser.add_argument("--trigger-seconds", type=float, default=20.0)
    parser.add_argument("--maximum-segment-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        roots = StorageRoots.from_environment(Path.cwd())
        source = roots.resolve_input(args.selection_manifest, base=Path.cwd())
        destination = roots.resolve_artifact(args.out, base=Path.cwd())
        audit_path = roots.resolve_artifact(args.audit_out, base=Path.cwd())
        assert source is not None and destination is not None and audit_path is not None
        rows, audit = expand_training_segments(
            read_manifest(source),
            storage_roots=roots,
            base=Path.cwd(),
            method=args.retargeting_method,
            trigger_seconds=args.trigger_seconds,
            maximum_segment_seconds=args.maximum_segment_seconds,
        )
        payload = publish_segmented_selection(
            source,
            destination,
            audit_path,
            rows,
            audit,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
