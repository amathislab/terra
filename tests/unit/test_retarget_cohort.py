from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from terra import cli
from terra.commands.retarget_cohort import (
    RetargetCohortPartition,
    RetargetCohortSpec,
    load_retarget_cohort_spec,
    run_retarget_cohort,
)

FIELDS = ("motion", "dataset", "source_cache_root")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_config(path: Path, dataset: str, cache: str, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[dataset]",
                f'name = "{dataset}"',
                "",
                "[input]",
                'root = "data/AMASS"',
                'glob = "**/*.npz"',
                "",
                "[terrain]",
                'mode = "flat"',
                'calibration = "none"',
                'contact_source = "kinematic"',
                'contact_joints = ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]',
                "calibrate_sites = true",
                "",
                "[terrain.fit]",
                "",
                "[retarget]",
                'method = "terra"',
                'env_name = "MyoFullBody"',
                'smpl_model_path = "smpl"',
                f'cache_root = "{cache}"',
                f'run_root = "runs/cohort/{key}"',
                "",
                "[retarget.overrides]",
                "",
            )
        )
    )


def _cohort(tmp_path: Path) -> Path:
    rows = [
        {"motion": "A/one", "dataset": "amass", "source_cache_root": "/users/artifacts/a/cache"},
        {"motion": "A/two", "dataset": "amass", "source_cache_root": "/users/artifacts/a/cache"},
        {"motion": "B/one", "dataset": "gait120", "source_cache_root": "/users/artifacts/b/cache"},
    ]
    source = tmp_path / "source.csv"
    _write_csv(source, rows)
    _write_csv(tmp_path / "selections/a.csv", rows[:2])
    _write_csv(tmp_path / "selections/b.csv", rows[2:])
    _write_config(tmp_path / "configs/a.toml", "amass", "runs/a/cache", "a")
    _write_config(tmp_path / "configs/b.toml", "gait120", "runs/b/cache", "b")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = tmp_path / "cohort.toml"
    spec.write_text(
        f'''schema_version = 1
name = "test-cohort"
source_selection = "source.csv"
source_selection_sha256 = "{digest}"
output_root = "runs/cohort"
methods = ["omniretarget", "gmr", "smpl"]

[[partition]]
key = "a"
dataset = "amass"
config = "configs/a.toml"
selection = "selections/a.csv"
source_cache_root = "/users/artifacts/a/cache"
motions = 2

[[partition]]
key = "b"
dataset = "gait120"
config = "configs/b.toml"
selection = "selections/b.csv"
source_cache_root = "/users/artifacts/b/cache"
motions = 1
'''
    )
    return spec


def test_cohort_spec_partitions_exactly_cover_pinned_source(tmp_path: Path) -> None:
    spec = load_retarget_cohort_spec(_cohort(tmp_path))

    assert spec.name == "test-cohort"
    assert spec.methods == ("omniretarget", "gmr", "smpl")
    assert [(partition.key, partition.motions) for partition in spec.partitions] == [("a", 2), ("b", 1)]


def test_cohort_spec_rejects_partition_drift(tmp_path: Path) -> None:
    spec_path = _cohort(tmp_path)
    _write_csv(
        tmp_path / "selections/a.csv",
        [{"motion": "A/two", "dataset": "amass", "source_cache_root": "/users/artifacts/a/cache"}],
    )

    with pytest.raises(ValueError, match="exact ordered source-cache subset"):
        load_retarget_cohort_spec(spec_path)


def test_cohort_runner_attempts_every_partition_and_reports_failure(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    selection = tmp_path / "selection.csv"
    config.write_text("placeholder")
    selection.write_text("motion\nA/one\n")
    spec = RetargetCohortSpec(
        path=tmp_path / "cohort.toml",
        name="test",
        source_selection=selection,
        source_selection_sha256="0" * 64,
        output_root="runs/cohort",
        methods=("gmr",),
        partitions=tuple(
            RetargetCohortPartition(
                key=key,
                dataset="amass",
                config=config,
                selection=selection,
                source_cache_root=f"/users/artifacts/{key}/cache",
                motions=1,
            )
            for key in ("a", "b")
        ),
    )
    calls: list[list[str]] = []

    def run(arguments):
        calls.append(arguments)
        return 2 if len(calls) == 1 else 0

    monkeypatch.setattr("terra.commands.run.main", run)
    monkeypatch.setattr(
        "terra.dataset_pipeline.load_dataset_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_root=tmp_path / "run",
            cache_root=tmp_path / "cache",
        ),
    )

    assert run_retarget_cohort(spec, "gmr", workers=3, threads_per_worker=2, dry_run=True) == 2
    assert len(calls) == 2
    assert all("--dry-run" in arguments for arguments in calls)
    assert all(
        arguments[arguments.index("--threads-per-worker") :][:2] == ["--threads-per-worker", "2"] for arguments in calls
    )


def test_cohort_rate_variant_uses_separate_cache_and_partition_outputs(monkeypatch, tmp_path: Path) -> None:
    spec = load_retarget_cohort_spec(_cohort(tmp_path))
    calls: list[list[str]] = []

    monkeypatch.setattr("terra.commands.run.main", lambda arguments: calls.append(arguments) or 0)
    monkeypatch.setattr(
        "terra.dataset_pipeline.load_dataset_config",
        lambda partition, **_kwargs: SimpleNamespace(
            run_root=tmp_path / Path(partition).stem,
            cache_root=tmp_path / f"canonical-{Path(partition).stem}",
        ),
    )

    result = run_retarget_cohort(
        spec,
        "gmr",
        workers=1,
        threads_per_worker=1,
        target_fps=100.0,
        cache_root=tmp_path / "rate100-cache",
        output_root=tmp_path / "rate100-output",
        dry_run=True,
    )

    assert result == 0
    assert len(calls) == 2
    for partition, arguments in zip(spec.partitions, calls, strict=True):
        assert arguments[arguments.index("--target-fps") :][:2] == ["--target-fps", "100.0"]
        assert arguments[arguments.index("--cache-root") :][:2] == [
            "--cache-root",
            str((tmp_path / "rate100-cache").resolve()),
        ]
        assert arguments[arguments.index("--run-root") :][:2] == [
            "--run-root",
            str((tmp_path / "rate100-output/partitions" / partition.key / "gmr").resolve()),
        ]


@pytest.mark.parametrize("method", ("omniretarget",))
def test_cohort_rate_variant_rejects_methods_without_explicit_rate_control(
    method: str,
    tmp_path: Path,
) -> None:
    spec = load_retarget_cohort_spec(_cohort(tmp_path))

    with pytest.raises(ValueError, match="only for GMR and MM-SMPL"):
        run_retarget_cohort(
            spec,
            method,
            workers=1,
            threads_per_worker=1,
            target_fps=100.0,
            cache_root=tmp_path / "rate100-cache",
            output_root=tmp_path / "rate100-output",
            dry_run=True,
        )


def test_retarget_cli_dispatches_cohort_without_loading_single_motion(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr("terra.commands.retarget_cohort.main", lambda arguments: seen.append(arguments) or 7)

    assert cli.main(["cohort", "cohort.toml", "--method", "gmr"]) == 7
    assert seen == [["cohort.toml", "--method", "gmr"]]
