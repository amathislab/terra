"""Command-line workflows for publishing retargeted motions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from terra._methods import SUPPORTED_METHODS
from terra._revision import write_git_commit
from terra.paths import StorageRoots

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _default_motion_name(source: Path) -> str:
    name = _UNSAFE_NAME.sub("_", source.stem).strip("._")
    if name:
        return name
    digest = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:8]
    return f"motion_{digest}"


def _load_json_object(value: str | Path | None, label: str) -> dict[str, object]:
    """Load a JSON object from an inline argument or a JSON file."""
    if value is None:
        return {}
    source = str(value).strip()
    try:
        payload = source if source.startswith(("{", "[")) else Path(source).expanduser().read_text()
        parsed = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be a JSON object or JSON file: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return parsed


def _terrain_argument(value: str):
    normalized = value.casefold()
    if normalized == "none":
        return None
    if normalized == "auto":
        return "auto"
    return value


def _mat_selector(value: str) -> tuple[str, int]:
    name, separator, raw_index = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("MAT selectors must use NAME=INDEX")
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("MAT selector indices must be integers") from exc
    return name.strip(), index


def main(argv: Sequence[str] | None = None) -> int:
    """Retarget one SMPL-H, C3D, TRC, or MAT motion and publish its output artifacts."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="terra retarget",
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "example: terra retarget motion_poses.npz --output-root retargeted_motions "
            "--name Study/Subject/Trial --smpl-model-path /models/smplh"
        ),
    )
    parser.add_argument("source", type=Path, help="AMASS-compatible .npz or marker .c3d/.trc/.mat motion")
    parser.add_argument("--output-root", required=True, type=Path, help="artifact cache root")
    parser.add_argument("--name", help="portable relative motion identifier; defaults to the source stem")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="terra", help="retargeting method")
    parser.add_argument(
        "--terrain",
        type=_terrain_argument,
        default="auto",
        help="'auto', 'none', or a TerrainSpec JSON path; auto requires terra or omniretarget",
    )
    parser.add_argument("--env-name", default="MyoFullBody", help="registered MuscleMimic target environment")
    parser.add_argument(
        "--smpl-model-path",
        type=Path,
        help="SMPL-H model root; defaults to TERRA_MODEL_ROOT",
    )
    parser.add_argument(
        "--c3d-model-path",
        type=Path,
        help="SMPL-X/SMPL-H marker-fitting root; required for marker input",
    )
    parser.add_argument("--config", help="method-configuration JSON object or file")
    parser.add_argument("--c3d-options", help="marker-fitting options JSON object or file")
    parser.add_argument(
        "--trc-up-axis",
        choices=("y", "z"),
        help="TRC vertical axis; y applies [X,-Z,Y], while z preserves XYZ",
    )
    parser.add_argument("--mat-schema", type=Path, help="JSON extraction schema required for .mat marker input")
    parser.add_argument(
        "--mat-selector",
        action="append",
        default=[],
        type=_mat_selector,
        metavar="NAME=INDEX",
        help="zero-based selector required by a nested MAT schema; repeat as needed",
    )
    parser.add_argument(
        "--work-cache",
        type=Path,
        help="marker fit/shape cache; defaults to <output-root>/.terra-c3d-cache",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing artifact set")
    args = parser.parse_args(arguments)

    try:
        # Keep argument discovery lightweight. Importing the numerical retargeting
        # stack is necessary only after argparse has accepted an actual run.
        from terra.api import retarget, retarget_cache_paths, save_retarget_result

        storage_roots = StorageRoots.from_environment(Path.cwd())
        source = args.source.expanduser().resolve()
        suffix = source.suffix.casefold()
        marker_source = suffix in {".c3d", ".trc", ".mat"}
        if marker_source:
            if args.c3d_model_path is None:
                parser.error("--c3d-model-path is required for marker input")
        elif args.c3d_model_path is not None or args.c3d_options is not None:
            parser.error("--c3d-model-path and --c3d-options require a marker source")
        if suffix != ".trc" and args.trc_up_axis is not None:
            parser.error("--trc-up-axis requires a .trc source")
        if suffix == ".mat" and args.mat_schema is None:
            parser.error("--mat-schema is required for .mat marker input")
        if suffix != ".mat" and (args.mat_schema is not None or args.mat_selector):
            parser.error("--mat-schema and --mat-selector require a .mat source")
        output_root = storage_roots.resolve_artifact(args.output_root, base=Path.cwd())
        assert output_root is not None
        motion_name = args.name or _default_motion_name(source)
        paths = retarget_cache_paths(output_root, motion_name, method=args.method, env_name=args.env_name)
        existing = [path for path in (paths.trajectory_path, paths.analysis_path, paths.terrain_path) if path.exists()]
        if existing and not args.overwrite:
            rendered = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"retargeted artifact(s) already exist: {rendered}")
        write_git_commit(output_root)
        config = _load_json_object(args.config, "method config")
        smpl_model_path = (
            storage_roots.resolve_model(args.smpl_model_path, base=Path.cwd())
            if args.smpl_model_path is not None
            else storage_roots.model_root
        )
        common = {
            "method": args.method,
            "terrain": args.terrain,
            "env_name": args.env_name,
            "config": config,
            "smpl_model_path": smpl_model_path,
        }
        if marker_source:
            c3d_options = _load_json_object(args.c3d_options, "marker options")
            c3d_options.setdefault("clear_cache", args.overwrite)
            work_cache = (
                storage_roots.resolve_artifact(args.work_cache, base=Path.cwd())
                if args.work_cache is not None
                else (output_root / ".terra-c3d-cache").resolve()
            )
            c3d_model_path = storage_roots.resolve_model(args.c3d_model_path, base=Path.cwd())
            marker_kwargs = {
                "c3d_model_path": c3d_model_path,
                "c3d_options": c3d_options,
                "cache_root": work_cache,
            }
            if suffix == ".trc":
                marker_kwargs["trc_up_axis"] = args.trc_up_axis or "y"
            elif suffix == ".mat":
                selectors = dict(args.mat_selector)
                if len(selectors) != len(args.mat_selector):
                    parser.error("--mat-selector names must be unique")
                marker_kwargs["mat_schema"] = args.mat_schema
                marker_kwargs["mat_selectors"] = selectors
            result = retarget(
                source,
                **common,
                **marker_kwargs,
            )
        else:
            result = retarget(source, **common, cache_root=output_root)

        artifacts = save_retarget_result(
            result,
            output_root,
            motion_name,
            env_name=args.env_name,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    payload = asdict(artifacts)
    for name in ("trajectory_path", "analysis_path", "terrain_path"):
        if payload[name] is not None:
            payload[name] = str(payload[name])
    payload["nonflat_terrain"] = bool(result.terrain and result.terrain.boxes)
    payload["storage_roots"] = storage_roots.as_dict()
    print(json.dumps(payload, indent=2))
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
