"""Configuration and hardware-contract tests for TERRA FlashSAC training."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import terra.training as training
from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
from terra import RETARGET_ARTIFACT_FORMAT_VERSION, retarget_cache_paths
from terra.terrain.metadata import TerrainMetadata
from terra.training import main


def test_training_output_markers_cover_hydra_and_durable_checkpoints(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from terra.rl import experiment

    monkeypatch.setenv("TERRA_GIT_COMMIT", "b" * 40)
    runtime = SimpleNamespace(
        output_dir=str(tmp_path / "hydra"),
        cwd=str(tmp_path / "launch"),
    )
    monkeypatch.setattr(experiment.HydraConfig, "get", lambda: SimpleNamespace(runtime=runtime))
    config = OmegaConf.create(
        {
            "experiment": {
                "checkpoint_root": "checkpoints",
                "run_id": "study-run",
            }
        }
    )

    directories = experiment._write_output_markers(config)

    assert directories == (
        (tmp_path / "hydra").resolve(),
        (tmp_path / "launch/checkpoints/study-run").resolve(),
    )
    assert all((directory / "GIT_COMMIT").read_text() == f"{'b' * 40}\n" for directory in directories)


def _ppo_config(**overrides):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "experiment": {
                "training_backend": "mjx_warp",
                "num_envs": 16,
                "num_steps": 8,
                "num_minibatches": 4,
                "n_seeds": 1,
                "vmap_across_seeds": False,
                "distributed": {"num_devices": 4},
                "trajectory": {"sharding": "none"},
                "env_params": {"mjx_backend": "warp", "mjx_warp_graph_mode": "warp"},
                "validation": {
                    "active": True,
                    "evaluate_all": True,
                    "deterministic": False,
                    "minimum_total_rollouts": 100,
                    "rollouts_per_motion": 3,
                    "num_envs": 8,
                },
            }
        }
    )
    for path, value in overrides.items():
        OmegaConf.update(config, path, value, merge=False)
    return config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"experiment.distributed.num_devices": 0}, "at least 1"),
        ({"experiment.n_seeds": 2}, "exactly one seed"),
        ({"experiment.trajectory.sharding": "static"}, "initial PPO rollout"),
        ({"experiment.training_backend": "mjx_jax"}, "training_backend='mjx_warp'"),
        ({"experiment.env_params.mjx_backend": "jax"}, "mjx_backend='warp'"),
        ({"experiment.num_envs": 15}, "must be divisible"),
        ({"experiment.validation.num_envs": 7}, "must be divisible"),
        ({"experiment.num_steps": 3, "experiment.num_minibatches": 8}, "per-device rollout batch"),
    ],
)
def test_ppo_config_preflight_rejects_unsupported_topology(overrides, message):
    with pytest.raises(ValueError, match=message):
        training._validate_ppo_config(_ppo_config(**overrides))


@pytest.mark.parametrize("unsafe_mode", ("jax", "warp_staged", "warp_staged_ex"))
def test_ppo_config_preflight_rejects_unsafe_training_graph_modes(unsafe_mode):
    config = _ppo_config(**{"experiment.env_params.mjx_warp_graph_mode": unsafe_mode})

    with pytest.raises(ValueError, match="mjx_warp_graph_mode='warp'"):
        training._validate_ppo_config(config)


@pytest.mark.parametrize("unsafe_mode", ("jax", "warp_staged", "warp_staged_ex"))
def test_ppo_config_preflight_rejects_unsafe_validation_graph_modes(unsafe_mode):
    config = _ppo_config(**{"experiment.validation.env_params.mjx_warp_graph_mode": unsafe_mode})

    with pytest.raises(ValueError, match="validation environment must use mjx_warp_graph_mode='warp'"):
        training._validate_ppo_config(config)


def test_ppo_config_preflight_accepts_all_visible_device_selection():
    config = _ppo_config(**{"experiment.distributed.num_devices": -1})

    assert training._validate_ppo_config(config) == -1
    training._validate_ppo_environment_divisibility(config, 2, config_name="ppo_multi_motion")


def test_ppo_config_preflight_accepts_one_device():
    config = _ppo_config(**{"experiment.distributed.num_devices": 1})

    assert training._validate_ppo_config(config) == 1


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("experiment.validation.evaluate_all", False, "evaluate_all=true"),
        ("experiment.validation.deterministic", True, "deterministic=false"),
        ("experiment.validation.minimum_total_rollouts", 99, "at least 100"),
        ("experiment.validation.rollouts_per_motion", 0, "at least 1"),
    ],
)
def test_ppo_config_preflight_enforces_stochastic_validation_contract(path, value, message):
    config = _ppo_config(**{path: value})

    with pytest.raises(ValueError, match=message):
        training._validate_ppo_config(config)


def test_ppo_config_preflight_accepts_one_stochastic_rollout_per_motion():
    config = _ppo_config(**{"experiment.validation.rollouts_per_motion": 1})

    assert training._validate_ppo_config(config) == 4


def test_ppo_preflight_requires_configured_gpu_count_from_both_runtimes(monkeypatch):
    import jax
    import warp as wp

    config = _ppo_config()
    monkeypatch.setattr(training, "_config_path", lambda _name: Path("/configs/ppo_multi_motion.yaml"))
    monkeypatch.setattr(training, "_compose_training_config", lambda **_kwargs: config)
    monkeypatch.setattr(training, "_validate_training_config", lambda *_args, **_kwargs: "PPOJax")
    monkeypatch.setattr(jax, "devices", lambda _backend: ("gpu:0", "gpu:1", "gpu:2", "gpu:3"))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 2)

    report = training.training_preflight(algorithm="ppo", require_cuda=False)

    assert report.ready is False
    assert report.algorithm_key == "ppo"
    assert report.configured_jax_gpu_device_count == 4
    assert report.minimum_jax_gpu_device_count == 4
    assert "found 4 JAX GPU(s) and 2 Warp CUDA device(s)" in report.error


def test_ppo_preflight_rejects_warp_version_with_known_replica_corruption(monkeypatch):
    import jax
    import warp as wp

    config = _ppo_config()
    monkeypatch.setattr(training, "_config_path", lambda _name: Path("/configs/ppo_multi_motion.yaml"))
    monkeypatch.setattr(training, "_compose_training_config", lambda **_kwargs: config)
    monkeypatch.setattr(training, "_validate_training_config", lambda *_args, **_kwargs: "PPOJax")
    monkeypatch.setattr(jax, "devices", lambda _backend: ("gpu:0", "gpu:1", "gpu:2", "gpu:3"))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 4)
    monkeypatch.setattr(wp, "__version__", "1.9.0")

    report = training.training_preflight(algorithm="ppo", require_cuda=False)

    assert report.ready is False
    assert "silently corrupt non-zero pmap replicas" in report.error


def test_ppo_preflight_reports_missing_terra_runtime(monkeypatch):
    import jax
    import warp as wp

    config = _ppo_config()
    monkeypatch.setattr(training, "_config_path", lambda _name: Path("/configs/ppo_multi_motion.yaml"))
    monkeypatch.setattr(training, "_compose_training_config", lambda **_kwargs: config)
    monkeypatch.setattr(training, "_validate_training_config", lambda *_args, **_kwargs: "PPOJax")
    monkeypatch.setattr(training, "_ppo_warp_compatibility_error", lambda: "install the pinned terra integration")
    monkeypatch.setattr(jax, "devices", lambda _backend: ("gpu:0", "gpu:1", "gpu:2", "gpu:3"))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 4)

    report = training.training_preflight(algorithm="ppo", require_cuda=False)

    assert report.ready is False
    assert report.error == "install the pinned terra integration"


def test_preflight_cli_selects_ppo_and_emits_machine_readable_report(monkeypatch, capsys):
    report = training.TrainingPreflight(
        ready=True,
        jax_version="test",
        jax_backend="gpu",
        jax_gpu_devices=("gpu:0", "gpu:1"),
        warp_version="test",
        warp_cuda_device_count=2,
        algorithm="PPOJax",
        training_backend="mjx_warp",
        mjx_backend="warp",
        config_path="ppo_multi_motion.yaml",
        algorithm_key="ppo",
        configured_jax_gpu_device_count=-1,
        minimum_jax_gpu_device_count=2,
    )
    calls = []
    monkeypatch.setattr(
        training,
        "training_preflight",
        lambda *, algorithm, require_cuda, multi_motion: (
            calls.append((algorithm, require_cuda, multi_motion)) or report
        ),
    )

    assert main(["--algorithm", "ppo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == [("ppo", False, True)]
    assert payload["algorithm"] == "PPOJax"
    assert payload["minimum_jax_gpu_device_count"] == 2


def test_training_module_import_does_not_load_runtime_stack():
    code = """
import json
import sys
import terra.training

heavy = {
    "fullbody.experiment",
    "jax",
    "musclemimic.runner.engine",
    "numpy",
    "terra.api",
    "warp",
}
print(json.dumps(sorted(heavy & sys.modules.keys())))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd="/tmp",
    )

    assert json.loads(completed.stdout) == []


def test_preflight_cli_emits_machine_readable_report(capsys):
    assert main(["--allow-no-device"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["algorithm"] == "PPOJax"
    assert report["training_backend"] == "mjx_warp"


def test_accelerator_probe_suppresses_expected_native_stderr(monkeypatch, capfd):
    import jax
    import warp as wp

    def unavailable_jax(_backend):
        os.write(2, b"expected JAX driver diagnostic\n")
        logging.getLogger("jax._src.xla_bridge").error("expected JAX plugin diagnostic")
        raise RuntimeError("no GPU")

    def unavailable_warp():
        os.write(2, b"expected Warp driver diagnostic\n")
        return 0

    def cpu_backend():
        os.write(2, b"expected JAX backend diagnostic\n")
        return "cpu"

    monkeypatch.setattr(jax, "devices", unavailable_jax)
    monkeypatch.setattr(jax, "default_backend", cpu_backend)
    monkeypatch.setattr(wp, "get_cuda_device_count", unavailable_warp)

    report = training.training_preflight(require_cuda=False)

    assert report.ready is False
    assert capfd.readouterr().err == ""


def _write_artifacts(tmp_path: Path, *, method: str = "terra", nonflat: bool = True):
    paths = retarget_cache_paths(tmp_path, "Study/Trial")
    paths.trajectory_path.parent.mkdir(parents=True)
    np.savez(
        paths.trajectory_path,
        qpos=np.zeros((7, 8)),
        qvel=np.zeros((7, 7)),
        frequency=np.asarray(100.0),
    )
    boxes = (BoxSpec(pos=(0.0, 0.0, 0.1), size=(1.0, 1.0, 0.1), name="step"),) if nonflat else ()
    terrain = TerrainSpec(boxes=boxes)
    TerrainMetadata.from_terrain(terrain).save(paths.terrain_path)
    np.savez(
        paths.analysis_path,
        artifact_format_version=np.asarray(RETARGET_ARTIFACT_FORMAT_VERSION),
        motion_name=np.asarray("Study/Trial"),
        retargeting_method=np.asarray(method),
        trajectory_file=np.asarray(paths.trajectory_path.name),
        terrain_file=np.asarray(paths.terrain_path.name),
    )
    return paths


def test_launch_preflights_the_exact_hydra_overrides_and_environment(monkeypatch):
    observed = {}
    launch = SimpleNamespace(
        algorithm_key="ppo",
        command=("python", "-m", "terra.rl.experiment", "--config-name=ppo_multi_motion", "experiment.num_envs=8"),
        environment={"TERRA_VALIDATION_ENVS": "32"},
    )

    def preflight(**kwargs):
        observed["preflight"] = kwargs

    def run(command, *, env, check):
        observed["command"] = command
        observed["environment"] = env
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training, "training_preflight", preflight)
    monkeypatch.setattr(training.subprocess, "run", run)

    assert training.launch_training(launch) == 0
    assert observed["preflight"]["overrides"] == ("experiment.num_envs=8",)
    assert observed["preflight"]["environment"] == launch.environment
    assert observed["command"] == launch.command
    assert observed["environment"]["TERRA_VALIDATION_ENVS"] == "32"


def test_preflight_composes_user_overrides_with_launch_environment(monkeypatch):
    import jax
    import warp as wp

    captured = {}
    config = _ppo_config()

    def compose(*, overrides, **_kwargs):
        captured["overrides"] = overrides
        captured["validation_envs"] = os.environ["TERRA_VALIDATION_ENVS"]
        return config

    monkeypatch.delenv("TERRA_VALIDATION_ENVS", raising=False)
    monkeypatch.setattr(training, "_config_path", lambda _name: Path("/configs/ppo_multi_motion.yaml"))
    monkeypatch.setattr(training, "_compose_training_config", compose)
    monkeypatch.setattr(training, "_validate_training_config", lambda *_args, **_kwargs: "PPOJax")
    monkeypatch.setattr(training, "_ppo_warp_compatibility_error", lambda: None)
    monkeypatch.setattr(jax, "devices", lambda _backend: ())
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 0)

    training.training_preflight(
        require_cuda=False,
        overrides=("experiment.num_envs=8",),
        environment={"TERRA_VALIDATION_ENVS": "64"},
    )

    assert captured == {"overrides": ("experiment.num_envs=8",), "validation_envs": "64"}
    assert "TERRA_VALIDATION_ENVS" not in os.environ
