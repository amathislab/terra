"""Run retargeting metrics for one explicit dataset manifest.

This command scores retargeted motions using an explicit dataset manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from terra._revision import write_git_commit
from terra.dataset_pipeline import DatasetConfig, load_dataset_config
from terra.datasets.config import bundled_dataset_config

# Keep the existing wire value so completed retargeting-only evaluations remain usable.
RETARGET_EVALUATION_SCHEMA = "terra.dataset-evaluation"
DEFAULT_METHOD_LABELS = {
    "terra": "TERRA",
    "omniretarget": "OmniRetarget",
    "gmr": "GMR",
    "smpl": "MuscleMimic SMPL-fit",
}


def _config_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.suffix.casefold() == ".toml" or candidate.parent != Path("."):
        return candidate.resolve()
    return bundled_dataset_config(value)


def _assignment(value: str, flag: str) -> tuple[str, str]:
    try:
        label, assigned = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{flag} must use LABEL=VALUE syntax") from error
    label, assigned = label.strip(), assigned.strip()
    if not label or not assigned:
        raise argparse.ArgumentTypeError(f"{flag} contains an empty label or value")
    return label, assigned


def _methods(values: list[str] | None, config: DatasetConfig) -> list[tuple[str, str]]:
    if not values:
        return [(DEFAULT_METHOD_LABELS.get(config.method, config.method), config.method)]
    methods = [_assignment(value, "--method") for value in values]
    if len({label for label, _method in methods}) != len(methods):
        raise ValueError("evaluation method labels must be unique")
    if len({method for _label, method in methods}) != len(methods):
        raise ValueError("evaluation cache subdirectories must be unique")
    return methods


def _metric_arguments(
    *,
    config: DatasetConfig,
    manifest: Path,
    methods: list[tuple[str, str]],
    terrain_method: str,
    output_root: Path,
    workers: int,
    allow_flat_name_conflicts: bool,
    common_intervals: Path | None = None,
) -> list[str]:
    return [
        *(argument for label, method in methods for argument in ("--method", f"{label}={method}")),
        "--motion-class",
        f"{config.name}={manifest}",
        "--terrain-method",
        terrain_method,
        "--cache-root",
        str(config.cache_root),
        "--source-root",
        str(config.input_root),
        "--out",
        str(output_root / "metrics"),
        "--quality-out",
        str(output_root / "quality"),
        "--workers",
        str(workers),
        "--allow-missing",
        *(["--allow-flat-name-conflicts"] if allow_flat_name_conflicts else []),
        *(["--common-intervals", str(common_intervals)] if common_intervals is not None else []),
    ]


def _validate_metrics(
    *,
    output_root: Path,
    manifest: Path,
    methods: list[tuple[str, str]],
    terrain_method: str,
    allow_flat_name_conflicts: bool,
    common_intervals: Path | None = None,
) -> dict[str, str]:
    metadata = output_root / "metrics" / "run.json"
    try:
        payload = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metric run metadata: {metadata}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"metric run metadata must contain an object: {metadata}")
    recorded_methods = payload.get("methods")
    if not isinstance(recorded_methods, dict) or set(map(str, recorded_methods.values())) != {
        method for _label, method in methods
    }:
        raise ValueError("metric run method set does not match the dataset evaluation")
    manifests = payload.get("manifests")
    if not isinstance(manifests, dict) or {Path(str(path)).expanduser().resolve() for path in manifests.values()} != {
        manifest
    }:
        raise ValueError("metric run manifest does not match the dataset evaluation")
    options = payload.get("options")
    expected_options = {
        "allow_flat_name_conflicts": allow_flat_name_conflicts,
        "allow_missing": True,
        "limit_per_class": None,
        "terrain_method": terrain_method,
        "common_intervals": str(common_intervals) if common_intervals is not None else None,
    }
    if not isinstance(options, dict) or any(options.get(key) != value for key, value in expected_options.items()):
        raise ValueError("metric run options do not match the dataset evaluation")
    return {"run": str(metadata.resolve()), "output": str((output_root / "metrics").resolve())}


def evaluate_dataset(args: argparse.Namespace) -> int:
    config = load_dataset_config(_config_path(args.config))
    if args.cache_root is not None:
        cache_root = config.storage_roots.resolve_artifact(args.cache_root, base=Path.cwd())
        assert cache_root is not None
        config = replace(config, cache_root=cache_root)
    manifest = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    methods = _methods(args.method, config)
    terrain_method = args.terrain_method or config.method
    common_intervals = getattr(args, "common_intervals", None)
    if common_intervals is not None:
        common_intervals = common_intervals.expanduser().resolve()
        if not common_intervals.is_file():
            raise ValueError(f"common scoring intervals file does not exist: {common_intervals}")
    if not manifest.is_file():
        raise ValueError(f"evaluation manifest does not exist: {manifest}")
    if args.workers < 1:
        raise ValueError("worker count must be positive")

    plan = {
        "schema": RETARGET_EVALUATION_SCHEMA,
        "dataset": config.name,
        "config": str(config.path),
        "manifest": str(manifest),
        "methods": dict(methods),
        "terrain_method": terrain_method,
        "allow_flat_name_conflicts": args.allow_flat_name_conflicts,
        "cache_root": str(config.cache_root),
        "source_root": str(config.input_root),
        "output_root": str(output_root),
    }
    if common_intervals is not None:
        plan["common_intervals"] = str(common_intervals)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    write_git_commit(output_root)
    from terra.evaluation.cli import metrics_main

    stages: dict[str, dict[str, Any]] = {}
    metric_exit = metrics_main(
        _metric_arguments(
            config=config,
            manifest=manifest,
            methods=methods,
            terrain_method=terrain_method,
            output_root=output_root,
            workers=args.workers,
            allow_flat_name_conflicts=args.allow_flat_name_conflicts,
            common_intervals=common_intervals,
        )
    )
    stages["metrics"] = {"exit_code": metric_exit}
    if metric_exit == 0:
        try:
            stages["metrics"].update(
                _validate_metrics(
                    output_root=output_root,
                    manifest=manifest,
                    methods=methods,
                    terrain_method=terrain_method,
                    allow_flat_name_conflicts=args.allow_flat_name_conflicts,
                    common_intervals=common_intervals,
                )
            )
        except ValueError as error:
            stages["metrics"].update(exit_code=2, error=str(error))

    payload = plan | {"stages": stages}
    (output_root / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    failed = [name for name, stage in stages.items() if stage["exit_code"]]
    if failed:
        print(f"{config.name}: failed evaluation stages: {', '.join(failed)}")
        return 2
    print(f"{config.name}: retargeting evaluation -> {output_root}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="terra evaluate dataset", description=__doc__)
    result.add_argument("config", help="bundled dataset name or explicit TOML path")
    result.add_argument("--manifest", type=Path, required=True, help="exact current evaluation cohort")
    result.add_argument(
        "--method",
        action="append",
        metavar="LABEL=CACHE_SUBDIR",
        help="repeat for every retargeting method; defaults to the config method",
    )
    result.add_argument("--terrain-method", help="cache subdirectory whose terrain defines interaction metrics")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--cache-root",
        type=Path,
        help="Experimental retargeting cache to evaluate instead of the config default.",
    )
    result.add_argument(
        "--common-intervals",
        type=Path,
        help="CSV of frozen motion/common_start_s/common_end_s scoring windows to reuse across methods.",
    )
    result.add_argument("--workers", type=int, default=1)
    result.add_argument(
        "--allow-flat-name-conflicts",
        action="store_true",
        help="Honor an explicitly flat manifest when a frozen cohort contains terrain-like motion names.",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        return evaluate_dataset(parser().parse_args(argv))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


__all__ = ["RETARGET_EVALUATION_SCHEMA", "evaluate_dataset", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
