"""Minimal executable example of TERRA's stable file-oriented API.

Examples:
    python examples/retargeting/retarget_and_publish.py motion_poses.npz \
        --output-root retargeted_motions --name Study/Subject/Trial \
        --smpl-model-path /models/smplh

    python examples/retargeting/retarget_and_publish.py trial.c3d \
        --output-root retargeted_motions --name Study/Subject/Trial \
        --smpl-model-path /models/smplh --c3d-model-path /models/smplx

    python examples/retargeting/retarget_and_publish.py trial.trc \
        --output-root retargeted_motions --name Study/Subject/Trial \
        --smpl-model-path /models/smplh --c3d-model-path /models/smplx

    python examples/retargeting/retarget_and_publish.py trial.mat \
        --mat-schema marker-schema.json \
        --output-root retargeted_motions --name Study/Subject/Trial \
        --smpl-model-path /models/smplh --c3d-model-path /models/smplx

Use ``terra retarget`` for the complete CLI, including method configuration and marker
fitting options. This example intentionally shows only the stable Python lifecycle.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from terra import retarget, save_retarget_result


def main(argv: Sequence[str] | None = None) -> int:
    """Retarget one motion, publish its artifacts, and print their paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="AMASS-compatible .npz or marker .c3d/.trc/.mat motion")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--name", required=True, help="portable identifier such as Study/Subject/Trial")
    parser.add_argument("--terrain", default="auto", help="'auto', 'none', or terrain JSON path")
    parser.add_argument("--smpl-model-path", type=Path)
    parser.add_argument("--c3d-model-path", type=Path)
    parser.add_argument("--mat-schema", type=Path)
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    terrain = None if args.terrain.casefold() == "none" else args.terrain
    work_cache = output_root / ".terra-work"
    format_options = {"mat_schema": args.mat_schema} if source.suffix.casefold() == ".mat" else {}
    result = retarget(
        source,
        method="terra",
        terrain=terrain,
        smpl_model_path=args.smpl_model_path,
        c3d_model_path=args.c3d_model_path if source.suffix.casefold() in {".c3d", ".trc", ".mat"} else None,
        cache_root=work_cache,
        **format_options,
    )
    artifacts = save_retarget_result(result, output_root, args.name)
    print(
        json.dumps(
            {
                "motion_name": artifacts.motion_name,
                "trajectory_path": str(artifacts.trajectory_path),
                "analysis_path": str(artifacts.analysis_path),
                "terrain_path": str(artifacts.terrain_path) if artifacts.terrain_path else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
