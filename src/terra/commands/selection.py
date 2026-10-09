"""Build a training selection from validated retargeted artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from terra._files import atomic_write
from terra._revision import write_git_commit
from terra.artifacts import validate_retarget_artifacts
from terra.commands.run import read_motion_selection
from terra.paths import StorageRoots
from terra.training_split import assign_training_splits, split_summary

SELECTION_FIELDS = (
    "motion",
    "dataset",
    "source_cache_root",
    "trajectory_relpath",
    "analysis_relpath",
    "terrain_relpath",
    "motion_type",
    "split",
)
TERRAIN_MODES = ("flat", "nonflat", "mixed")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid dataset run JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"dataset run must contain an object: {path}")
    return value


def _verified_run(path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    run_path = (path / "run.json" if path.is_dir() else path).expanduser().resolve()
    if not run_path.is_file():
        raise FileNotFoundError(f"dataset run record not found: {run_path}")
    payload = _load_json(run_path)
    if payload.get("method") != "terra":
        raise ValueError(f"training selections require a TERRA run, got {payload.get('method')!r}: {run_path}")
    dataset = payload.get("dataset")
    env_name = payload.get("env_name")
    cache_root = payload.get("cache_root")
    manifest_value = payload.get("output_manifest")
    if not all(isinstance(value, str) and value for value in (dataset, env_name, cache_root, manifest_value)):
        raise ValueError(f"dataset run is missing dataset, environment, cache root, or manifest: {run_path}")
    manifest = Path(manifest_value).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"dataset output manifest not found: {manifest}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"motion", "dataset", "passed"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"dataset output manifest is empty or missing {sorted(required)}: {manifest}")
    motions = [row.get("motion", "").strip() for row in rows]
    if any(not motion for motion in motions) or len(motions) != len(set(motions)):
        raise ValueError(f"dataset output manifest contains empty or duplicate motion IDs: {manifest}")
    for row in rows:
        if row["passed"].strip().casefold() not in {"1", "true", "yes"}:
            raise ValueError(f"dataset output manifest contains a non-passing row: {row['motion']!r}")
    payload = payload | {
        "_run_path": str(run_path),
        "_manifest_path": str(manifest),
    }
    return payload, rows


def _selection_row(
    cache_root: Path,
    motion: str,
    dataset: str,
    terrain_mode: str,
    motion_type: str = "",
    env_name: str = "MyoFullBody",
) -> dict[str, str]:
    validated = validate_retarget_artifacts(
        cache_root,
        motion,
        method="terra",
        env_name=env_name,
        require_nonflat_terrain=terrain_mode == "nonflat",
    )
    if terrain_mode == "flat" and validated.nonflat_terrain:
        raise ValueError(f"flat training selection contains non-flat terrain: {motion!r}")
    try:
        trajectory_relpath = str(validated.trajectory_path.relative_to(cache_root))
        analysis_relpath = str(validated.analysis_path.relative_to(cache_root))
        terrain_relpath = "" if validated.terrain_path is None else str(validated.terrain_path.relative_to(cache_root))
    except ValueError as error:
        raise ValueError(f"validated artifact escaped its cache root for {motion!r}") from error
    return {
        "motion": motion,
        "dataset": dataset,
        "source_cache_root": str(cache_root),
        "trajectory_relpath": trajectory_relpath,
        "analysis_relpath": analysis_relpath,
        "terrain_relpath": terrain_relpath,
        "motion_type": motion_type,
        "split": "train",
    }


def build_cache_selection(
    cache_root: Path,
    motions: list[str],
    *,
    dataset: str,
    terrain_mode: str = "mixed",
) -> list[dict[str, str]]:
    """Select validated artifacts directly from a retargeting cache."""
    if terrain_mode not in TERRAIN_MODES:
        raise ValueError(f"terrain_mode must be one of {', '.join(TERRAIN_MODES)}")
    if not dataset.strip() or not motions or len(motions) != len(set(motions)):
        raise ValueError("provide a dataset label and unique motion IDs")
    root = cache_root.expanduser().resolve()
    return [_selection_row(root, motion, dataset, terrain_mode) for motion in motions]


def build_selection(
    runs: list[Path],
    *,
    motions: list[str] | None = None,
    terrain_mode: str = "nonflat",
    duplicate_policy: str = "error",
) -> list[dict[str, str]]:
    """Validate current run records and return materialization-ready rows."""
    if terrain_mode not in TERRAIN_MODES:
        raise ValueError(f"terrain_mode must be one of {', '.join(TERRAIN_MODES)}")
    if duplicate_policy not in {"error", "first"}:
        raise ValueError("duplicate_policy must be 'error' or 'first'")
    available: dict[str, tuple[dict[str, object], dict[str, str]]] = {}
    ordered: list[str] = []
    for path in runs:
        run, manifest_rows = _verified_run(path)
        for manifest_row in manifest_rows:
            motion = manifest_row["motion"]
            if motion in available:
                if duplicate_policy == "first":
                    continue
                raise ValueError(f"motion appears in more than one dataset run: {motion!r}")
            available[motion] = (run, manifest_row)
            ordered.append(motion)

    selected = ordered if motions is None else motions
    missing = [motion for motion in selected if motion not in available]
    if missing:
        raise ValueError(f"selection contains motion absent from current TERRA runs: {missing[0]!r}")
    rows: list[dict[str, str]] = []
    for motion in selected:
        run, _manifest_row = available[motion]
        cache_root = Path(str(run["cache_root"])).expanduser().resolve()
        rows.append(
            _selection_row(
                cache_root,
                motion,
                str(run["dataset"]),
                terrain_mode,
                _manifest_row.get("terrain_class", "").strip(),
                str(run["env_name"]),
            )
        )
    return rows


def publish_selection(
    path: Path,
    rows: list[dict[str, str]],
    *,
    overwrite: bool = False,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Publish the ordered training-selection CSV."""
    destination = path.expanduser().resolve()
    if destination.suffix.casefold() != ".csv":
        raise ValueError("training selection output must use a .csv extension")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"training selection output already exists: {destination}")
    report = split_summary(rows)
    payload: dict[str, object] = {
        "selection": str(destination),
        "motions": [row["motion"] for row in rows],
        "split_summary": report,
    }
    resolved_report = None if report_path is None else report_path.expanduser().resolve()
    if resolved_report is not None:
        if resolved_report.suffix.casefold() != ".json":
            raise ValueError("training selection report output must use a .json extension")
        if resolved_report == destination:
            raise ValueError("training selection and report outputs must differ")
        if resolved_report.exists() and not overwrite:
            raise FileExistsError(f"training selection report already exists: {resolved_report}")
        payload["report"] = str(resolved_report)

    atomic_write(destination, lambda temporary: _write_csv(temporary, rows))
    if resolved_report is not None:
        rendered = json.dumps(payload, indent=2) + "\n"
        atomic_write(
            resolved_report,
            lambda temporary: temporary.write_text(rendered, encoding="utf-8"),
        )
    write_git_commit(destination.parent)
    return payload


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra train select", description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        help="current TERRA run directory or run.json; repeat for each dataset",
    )
    parser.add_argument("--cache-root", type=Path, help="retargeting cache for a direct artifact selection")
    parser.add_argument("--motion", action="append", help="motion ID in --cache-root; repeat to select more")
    parser.add_argument("--dataset", help="dataset label for a direct artifact selection")
    parser.add_argument(
        "--motions",
        type=Path,
        help="optional explicit ordered CSV/TXT motion list; defaults to every passing run row",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--report-out",
        type=Path,
        help="optional JSON path for the complete cohort and split summary",
    )
    parser.add_argument(
        "--terrain-mode",
        choices=TERRAIN_MODES,
        default="nonflat",
        help="require flat, non-flat, or allow a verified mixture of both terrain kinds",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first"),
        default="error",
        help=(
            "handle a motion present in multiple runs; 'first' preserves the first --run, "
            "which lets non-flat artifacts take precedence over a flat duplicate"
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        help=(
            "hold out this fraction as a fully isolated test set using a deterministic "
            "identity-disjoint stratified split"
        ),
    )
    parser.add_argument(
        "--evaluation-fraction",
        type=float,
        help="hold out this fraction for policy validation, disjoint from training and final test",
    )
    parser.add_argument(
        "--split-seed",
        default="terra-universal-tracking-split-v1",
        help="stable seed for the identity-disjoint split",
    )
    args = parser.parse_args(argv)
    try:
        roots = StorageRoots.from_environment(Path.cwd())
        if args.cache_root is not None:
            if args.run or args.motions:
                raise ValueError("--cache-root cannot be combined with --run or --motions")
            if not args.motion or not args.dataset:
                raise ValueError("--cache-root requires --motion and --dataset")
            cache_root = roots.resolve_artifact(args.cache_root, base=Path.cwd())
            assert cache_root is not None
            rows = build_cache_selection(cache_root, args.motion, dataset=args.dataset, terrain_mode=args.terrain_mode)
        else:
            if not args.run or args.motion or args.dataset:
                raise ValueError("provide --run, or use --cache-root with --motion and --dataset")
            run_paths = [roots.resolve_artifact(path, base=Path.cwd()) for path in args.run]
            resolved_runs = [path for path in run_paths if path is not None]
            motions_path = roots.resolve_input(args.motions, base=Path.cwd())
            selected_motions = None if motions_path is None else read_motion_selection(motions_path)
            rows = build_selection(
                resolved_runs,
                motions=selected_motions,
                terrain_mode=args.terrain_mode,
                duplicate_policy=args.duplicate_policy,
            )
        if args.test_fraction is not None or args.evaluation_fraction is not None:
            rows = assign_training_splits(
                rows,
                test_fraction=args.test_fraction,
                evaluation_fraction=args.evaluation_fraction,
                seed=args.split_seed,
            )
        output = roots.resolve_artifact(args.out, base=Path.cwd())
        assert output is not None
        report_output = roots.resolve_artifact(args.report_out, base=Path.cwd())
        payload = publish_selection(
            output,
            rows,
            overwrite=args.overwrite,
            report_path=report_output,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2))
    return 0


__all__ = [
    "SELECTION_FIELDS",
    "build_cache_selection",
    "build_selection",
    "main",
    "publish_selection",
]
