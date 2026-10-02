"""Resolve and prepare external runtime assets without process-global mutation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terra._musclemimic import (
    OPTIMIZED_SHAPE_FILE_NAME,
    fit_smpl_shape,
)
from terra._revision import write_git_commit
from terra.paths import StorageRoots


def _storage_roots() -> StorageRoots:
    return StorageRoots.from_environment(Path.cwd())


def resolve_model_path(path: str | Path | None) -> Path:
    """Resolve an explicit SMPL-H root or ``TERRA_MODEL_ROOT``."""
    model_path = Path(path).expanduser().resolve() if path is not None else _storage_roots().model_root
    if not model_path.exists():
        raise FileNotFoundError(
            f"SMPL-H model root not found: {model_path}. Set TERRA_MODEL_ROOT or pass smpl_model_path explicitly."
        )
    return model_path


def resolve_cache_root(cache_root: str | Path | None) -> Path:
    """Resolve an explicit working cache or the TERRA artifact cache."""

    return (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else _storage_roots().direct_cache_root.resolve()
    )


def shape_cache_path(env_name: str, cache_root: str | Path | None) -> Path:
    """Return the fitted robot-shape path for one target environment."""
    root = resolve_cache_root(cache_root)
    return root / env_name.removeprefix("Mjx") / OPTIMIZED_SHAPE_FILE_NAME


def ensure_environment_registered(env_name: str) -> None:
    """Load backend registrations and reject an unknown target environment."""
    from musclemimic.environments import LocoEnv

    if env_name not in LocoEnv.registered_envs:
        supported = ", ".join(sorted(LocoEnv.registered_envs))
        raise ValueError(f"unknown MuscleMimic environment {env_name!r}; registered environments: {supported}")


def ensure_robot_shape(
    env_name: str,
    robot_conf: Any,
    model_path: Path,
    shape_path: Path,
    logger: Any,
) -> None:
    """Fit the environment-specific SMPL-H shape once when absent."""
    if shape_path.is_file():
        return
    shape_path.parent.mkdir(parents=True, exist_ok=True)
    write_git_commit(shape_path.parent)
    logger.info("Fitting the robot-specific SMPL-H shape: %s", shape_path)
    fit_smpl_shape(env_name, robot_conf, str(model_path), str(shape_path), logger)


__all__ = [
    "ensure_environment_registered",
    "ensure_robot_shape",
    "resolve_cache_root",
    "resolve_model_path",
    "shape_cache_path",
]
