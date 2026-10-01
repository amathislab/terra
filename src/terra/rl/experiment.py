"""Hydra entry point for TERRA policy training."""

from __future__ import annotations

from pathlib import Path

import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from terra._revision import write_git_commit
from terra.rl import register_components
from terra.rl.backend import install_backend_integrations
from terra.rl.config_resolvers import register_config_resolvers
from terra.rl.hooks import TerraHooks

register_config_resolvers()


def _write_output_markers(config: DictConfig) -> tuple[Path, ...]:
    """Stamp the Hydra output and any durable checkpoint run directory."""

    runtime = HydraConfig.get().runtime
    directories = [Path(runtime.output_dir)]
    checkpoint_root = config.experiment.get("checkpoint_root")
    run_id = config.experiment.get("run_id")
    if checkpoint_root and run_id:
        root = Path(str(checkpoint_root)).expanduser()
        if not root.is_absolute():
            root = Path(runtime.cwd) / root
        directories.append(root / str(run_id))
    resolved = tuple(dict.fromkeys(path.resolve() for path in directories))
    for directory in resolved:
        write_git_commit(directory)
    return resolved


@hydra.main(version_base=None, config_path="configs", config_name="ppo_multi_motion")
def main(config: DictConfig) -> None:
    from musclemimic.runner.engine import run_experiment

    register_components()
    install_backend_integrations(str(config.experiment.get("algorithm", "PPOJax")))
    _write_output_markers(config)
    jax.config.update("jax_default_matmul_precision", str(config.experiment.get("matmul_precision", "high")))
    run_experiment(config, hooks=TerraHooks())


if __name__ == "__main__":
    main()
