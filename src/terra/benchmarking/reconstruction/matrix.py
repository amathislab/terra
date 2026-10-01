"""Declarative multi-dataset terrain-reconstruction cohort runner.

The matrix describes only reconstruction.  Evaluation consumes its published records
as a separate stage, preventing scoring changes from altering reconstruction identity.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from terra.datasets.config import bundled_dataset_config
from terra.datasets.selections import bundled_dataset_selection
from terra.paths import StorageRoots

from .core import CohortResult, load_selection, run_cohort
from .registry import RECONSTRUCTION_METHODS, create_method

SCHEMA_VERSION = 2
REPO = Path(__file__).resolve().parents[4]
DEFAULT_MATRIX = Path(__file__).with_name("terrain_reconstruction.toml")


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table")
    return value


def _path(value: object, context: str, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty path string")
    expanded = Path(value).expanduser()
    return (expanded if expanded.is_absolute() else repo_root / expanded).resolve()


def _dataset_config_path(value: object, context: str, repo_root: Path) -> Path:
    if isinstance(value, str) and value.startswith("bundled:"):
        name = value.removeprefix("bundled:")
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"{context} has an invalid bundled dataset name: {value!r}")
        return bundled_dataset_config(name)
    return _path(value, context, repo_root)


def _selection_path(value: object, context: str, repo_root: Path) -> Path:
    if isinstance(value, str) and value.startswith("bundled:"):
        name = value.removeprefix("bundled:")
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"{context} has an invalid bundled selection name: {value!r}")
        return bundled_dataset_selection(name)
    return _path(value, context, repo_root)


@dataclass(frozen=True)
class MatrixMethod:
    name: str
    options: Mapping[str, Any]


@dataclass(frozen=True)
class MatrixDataset:
    name: str
    selection: Path | None
    dataset_config: Path | None
    methods: tuple[str, ...]


@dataclass(frozen=True)
class ReconstructionMatrix:
    path: Path
    output_root: Path
    methods: Mapping[str, MatrixMethod]
    datasets: Mapping[str, MatrixDataset]
    storage_roots: StorageRoots


@dataclass(frozen=True)
class MatrixJob:
    dataset: MatrixDataset
    method: MatrixMethod
    output_dir: Path
    publication_root: Path
    publication_root_source: str
    storage_roots: StorageRoots
    matrix_path: Path


def load_matrix(
    path: Path,
    *,
    repo_root: Path = REPO,
    storage_roots: StorageRoots | None = None,
    environment: Mapping[str, str] | None = None,
) -> ReconstructionMatrix:
    matrix_path = path.expanduser().resolve()
    with matrix_path.open("rb") as handle:
        data = tomllib.load(handle)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{matrix_path} must declare schema_version = {SCHEMA_VERSION}")
    unknown = sorted(set(data) - {"schema_version", "run", "methods", "datasets"})
    if unknown:
        raise ValueError(f"matrix has unsupported top-level tables: {', '.join(unknown)}")
    run = _mapping(data.get("run"), "[run]")
    if set(run) != {"output_root"}:
        raise ValueError("[run] must contain exactly output_root")
    roots = storage_roots or StorageRoots.from_environment(repo_root, environment=environment)
    output_root = roots.resolve_artifact(run["output_root"], base=Path.cwd(), environment=environment)
    assert output_root is not None

    methods: dict[str, MatrixMethod] = {}
    for name, raw in _mapping(data.get("methods"), "[methods]").items():
        if name not in RECONSTRUCTION_METHODS:
            raise ValueError(f"matrix method {name!r} is not registered")
        table = _mapping(raw, f"[methods.{name}]")
        unknown_method = sorted(set(table) - {"options"})
        if unknown_method:
            raise ValueError(f"[methods.{name}] has unsupported fields: {', '.join(unknown_method)}")
        options = table.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"[methods.{name}].options must be a table")
        methods[name] = MatrixMethod(name, options)
    if not methods:
        raise ValueError("matrix must select at least one method")

    datasets: dict[str, MatrixDataset] = {}
    for name, raw in _mapping(data.get("datasets"), "[datasets]").items():
        table = _mapping(raw, f"[datasets.{name}]")
        unknown_dataset = sorted(set(table) - {"selection", "config", "methods"})
        if unknown_dataset:
            raise ValueError(f"[datasets.{name}] has unsupported fields: {', '.join(unknown_dataset)}")
        selected = table.get("methods", list(methods))
        if not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected):
            raise ValueError(f"[datasets.{name}].methods must be a non-empty list of strings")
        if len(selected) != len(set(selected)):
            raise ValueError(f"[datasets.{name}].methods contains duplicates")
        unknown_methods = sorted(set(selected) - set(methods))
        if unknown_methods:
            raise ValueError(f"dataset {name!r} selects undefined methods: {', '.join(unknown_methods)}")
        datasets[name] = MatrixDataset(
            name=name,
            selection=(
                None
                if table.get("selection") is None
                else _selection_path(table["selection"], f"[datasets.{name}].selection", repo_root)
            ),
            dataset_config=(
                None
                if table.get("config") is None
                else _dataset_config_path(table["config"], f"[datasets.{name}].config", repo_root)
            ),
            methods=tuple(selected),
        )
    if not datasets:
        raise ValueError("matrix must select at least one dataset")
    return ReconstructionMatrix(matrix_path, output_root, methods, datasets, roots)


def build_jobs(
    matrix: ReconstructionMatrix,
    *,
    datasets: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    output_root: Path | None = None,
    manifests: Mapping[str, Path] | None = None,
) -> tuple[MatrixJob, ...]:
    selected_datasets = tuple(matrix.datasets) if datasets is None else tuple(datasets)
    selected_methods = None if methods is None else set(methods)
    unknown_datasets = sorted(set(selected_datasets) - set(matrix.datasets))
    if unknown_datasets:
        raise ValueError(f"unknown matrix datasets: {', '.join(unknown_datasets)}")
    if selected_methods is not None:
        unknown_methods = sorted(selected_methods - set(matrix.methods))
        if unknown_methods:
            raise ValueError(f"unknown matrix methods: {', '.join(unknown_methods)}")
    if output_root is None:
        root = matrix.output_root
        root_source = "matrix_config"
    elif output_root.expanduser().is_absolute():
        root = output_root.expanduser().resolve()
        root_source = "absolute_cli_override"
    else:
        resolved_root = matrix.storage_roots.resolve_artifact(output_root, base=Path.cwd())
        assert resolved_root is not None
        root = resolved_root
        root_source = "relative_cli_override"
    manifest_overrides = {} if manifests is None else dict(manifests)
    unknown_manifest_datasets = sorted(set(manifest_overrides) - set(matrix.datasets))
    if unknown_manifest_datasets:
        raise ValueError(f"manifest overrides name unknown datasets: {', '.join(unknown_manifest_datasets)}")
    jobs: list[MatrixJob] = []
    for dataset_name in selected_datasets:
        dataset = matrix.datasets[dataset_name]
        if dataset_name in manifest_overrides:
            dataset = replace(dataset, selection=manifest_overrides[dataset_name].expanduser().resolve())
        for method_name in dataset.methods:
            if selected_methods is not None and method_name not in selected_methods:
                continue
            jobs.append(
                MatrixJob(
                    dataset,
                    matrix.methods[method_name],
                    root / dataset.name / method_name / "terrain",
                    root,
                    root_source,
                    matrix.storage_roots,
                    matrix.path,
                )
            )
    if not jobs:
        raise ValueError("matrix selection produced no reconstruction jobs")
    return tuple(jobs)


def preflight(job: MatrixJob) -> tuple[dict[str, str], ...]:
    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "ok" if passed else "blocker", "detail": detail})

    selection_path = job.dataset.selection
    if selection_path is None or not selection_path.is_file():
        check(
            "selection",
            False,
            "not configured; pass --manifest DATASET=PATH" if selection_path is None else str(selection_path),
        )
    else:
        try:
            selection = load_selection(selection_path)
        except (OSError, TypeError, ValueError) as error:
            check("selection", False, str(error))
        else:
            check("selection", True, f"{len(selection.motions)} unique canonical motions")
    path = job.dataset.dataset_config
    check("dataset_config", path is not None and path.is_file(), "not configured" if path is None else str(path))
    return tuple(checks)


def execute_job(job: MatrixJob, *, overwrite: bool = False) -> CohortResult:
    if job.dataset.selection is None:
        raise ValueError(f"dataset {job.dataset.name!r} has no manifest; pass --manifest {job.dataset.name}=PATH")
    selection = load_selection(job.dataset.selection)
    method = create_method(
        job.method.name,
        motions=selection.motions,
        dataset_config=job.dataset.dataset_config,
        options=job.method.options,
    )
    return run_cohort(
        method,
        job.dataset.selection,
        job.output_dir,
        overwrite=overwrite,
        repo_root=REPO,
        dataset_config_path=job.dataset.dataset_config,
        matrix_path=job.matrix_path,
    )


def _dataset_paths(values: Sequence[str] | None, flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or ():
        try:
            dataset, rendered = value.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{flag} must use DATASET=PATH syntax: {value!r}") from error
        if not dataset or not rendered or dataset in result:
            raise ValueError(f"{flag} has an empty or duplicate dataset assignment: {value!r}")
        result[dataset] = Path(rendered)
    return result


def matrix_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra reconstruct matrix", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--manifest",
        action="append",
        metavar="DATASET=PATH",
        help="current generated cohort manifest; repeat once per selected dataset",
    )
    parser.add_argument("--execute", action="store_true", help="run fits; the default is a read-only plan")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        matrix = load_matrix(args.config)
        jobs = build_jobs(
            matrix,
            datasets=args.datasets,
            methods=args.methods,
            output_root=args.output_root,
            manifests=_dataset_paths(args.manifest, "--manifest"),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    plans = []
    blocked = False
    for job in jobs:
        checks = preflight(job)
        blocked = blocked or any(check["status"] == "blocker" for check in checks)
        plans.append(
            {
                "dataset": job.dataset.name,
                "method": job.method.name,
                "selection": None if job.dataset.selection is None else str(job.dataset.selection),
                "output_dir": str(job.output_dir),
                "checks": list(checks),
            }
        )
    print(
        json.dumps(
            {
                "matrix": str(matrix.path),
                "storage_roots": matrix.storage_roots.as_dict(),
                "jobs": plans,
            },
            indent=2,
        )
    )
    if not args.execute:
        return 2 if blocked else 0
    if blocked:
        print("matrix execution blocked by preflight", flush=True)
        return 2
    results = [execute_job(job, overwrite=args.overwrite) for job in jobs]
    return 2 if any(result.failed for result in results) else 0
