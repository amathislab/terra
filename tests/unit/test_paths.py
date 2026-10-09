"""Tests for the bundled external-storage boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from terra.dataset_pipeline import load_dataset_config
from terra.datasets.config import bundled_dataset_config
from terra.paths import StorageRoots

REPO = Path(__file__).resolve().parents[2]


def test_storage_roots_use_external_user_defaults(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = StorageRoots.from_environment(REPO, environment={"HOME": str(home)})

    assert roots.data_root == home / ".local/share/terra/datasets"
    assert roots.artifact_root == home / ".local/state/terra/artifacts"
    assert roots.model_root == home / ".local/share/terra/models"
    assert roots.direct_cache_root == home / ".local/state/terra/artifacts/direct/cache"
    assert roots.resolve_input("data/AMASS") == roots.data_root / "AMASS"
    assert roots.resolve_input("runs/gait120/smplh") == roots.artifact_root / "gait120/smplh"
    assert roots.resolve_artifact("runs/gait120/cache") == roots.artifact_root / "gait120/cache"
    assert roots.resolve_model("smpl") == roots.model_root
    assert roots.resolve_input("custom/converted") == (REPO / "custom/converted").resolve()


def test_storage_roots_rebase_only_named_storage_prefixes(tmp_path: Path) -> None:
    environment = {
        "HOME": str(tmp_path / "home"),
        "TERRA_DATA_ROOT": str(tmp_path / "raw"),
        "TERRA_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "TERRA_MODEL_ROOT": str(tmp_path / "models"),
        "COHORT": "selected",
    }
    roots = StorageRoots.from_environment(REPO, environment=environment)

    assert roots.resolve_input("data/AMASS", environment=environment) == tmp_path / "raw/AMASS"
    assert roots.resolve_input("runs/gait120/smplh", environment=environment) == tmp_path / "artifacts/gait120/smplh"
    assert (
        roots.resolve_artifact("runs/$COHORT/cache", environment=environment) == tmp_path / "artifacts/selected/cache"
    )
    assert roots.resolve_model("smpl/SMPLH", environment=environment) == tmp_path / "models/SMPLH"
    assert roots.resolve_input("~/motion.npz", environment=environment) == tmp_path / "home/motion.npz"
    assert roots.resolve_artifact("custom/output", base=tmp_path, environment=environment) == tmp_path / "custom/output"


def test_storage_path_resolution_rejects_unresolved_variables() -> None:
    roots = StorageRoots.from_environment(REPO, environment={"HOME": "/tmp/terra-path-test-home"})

    with pytest.raises(ValueError, match="unresolved environment variable"):
        roots.resolve_input("data/$MISSING/motion.npz", environment={})


def test_explicit_storage_roots_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TERRA_DATA_ROOT must be an absolute path"):
        StorageRoots.from_environment(
            REPO,
            environment={"HOME": str(tmp_path), "TERRA_DATA_ROOT": "relative/data"},
        )


def test_runtime_defaults_follow_terra_storage_roots(monkeypatch, tmp_path: Path) -> None:
    from terra.runtime import resolve_cache_root, resolve_model_path, shape_cache_path

    model_root = tmp_path / "models"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    monkeypatch.setenv("TERRA_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("TERRA_ARTIFACT_ROOT", str(artifact_root))

    assert resolve_model_path(None) == model_root
    assert resolve_cache_root(None) == artifact_root / "direct/cache"
    assert shape_cache_path("MjxMyoFullBody", None).parent == artifact_root / "direct/cache/MyoFullBody"


@pytest.mark.parametrize(
    ("config_name", "expected_input"),
    (("amass", "raw/AMASS"), ("gait120", "artifacts/gait120/smplh")),
)
def test_dataset_configs_follow_overridden_roots_without_mutating_environment(
    config_name: str,
    expected_input: str,
    tmp_path: Path,
) -> None:
    environment = {
        "HOME": str(tmp_path / "home"),
        "TERRA_DATA_ROOT": str(tmp_path / "raw"),
        "TERRA_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "TERRA_MODEL_ROOT": str(tmp_path / "models"),
    }
    process_environment = dict(os.environ)

    config = load_dataset_config(
        bundled_dataset_config(config_name),
        environment=environment,
    )

    assert config.input_root == tmp_path / expected_input
    assert config.cache_root == tmp_path / f"artifacts/{config_name}/cache"
    assert config.run_root == tmp_path / f"artifacts/{config_name}/retarget"
    assert config.smpl_model_path == tmp_path / "models"
    assert config.storage_roots.as_dict() == {
        "repository_root": str(REPO),
        "data_root": str(tmp_path / "raw"),
        "artifact_root": str(tmp_path / "artifacts"),
        "model_root": str(tmp_path / "models"),
    }
    assert dict(os.environ) == process_environment


def test_dataset_runner_dry_run_reports_resolved_storage_roots_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from terra.commands.run import _dry_run

    environment = {
        "TERRA_DATA_ROOT": str(tmp_path / "raw"),
        "TERRA_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
    }
    process_environment = dict(os.environ)
    config = load_dataset_config(
        bundled_dataset_config("amass"),
        environment=environment,
    )

    _dry_run(config, [])

    payload = json.loads(capsys.readouterr().out)
    assert payload["storage_roots"]["data_root"] == str(tmp_path / "raw")
    assert payload["storage_roots"]["artifact_root"] == str(tmp_path / "artifacts")
    assert dict(os.environ) == process_environment
