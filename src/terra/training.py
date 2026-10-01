"""Policy-training readiness checks for TERRA trajectories."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from terra._methods import validate_method

TERRA_MULTI_GPU_PPO_CONFIG = "ppo_multi_motion"
TERRA_TRAINING_ALGORITHMS = ("ppo",)
_PPO_WARP_GRAPH_MODE = "warp"
_SUPPORTED_TRAINING_CONFIGS = frozenset((TERRA_MULTI_GPU_PPO_CONFIG,))
_EXPECTED_ALGORITHM_CLASS = {
    "ppo": "PPOJax",
}
_SAFE_TRAINING_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MANAGED_HYDRA_SETTINGS = (
    "experiment.task_factory.params.amass_dataset_conf",
    "experiment.validation.amass_dataset_conf",
    "experiment.validation.num_steps",
)
_ARTIFACT_DATASET_SETTINGS = _MANAGED_HYDRA_SETTINGS[:2]
_REQUIRED_DATASET_VALUES = {
    "load_paired_terrain": True,
    "allow_cache_download": False,
}
_METHOD_DATASET_FIELDS = ("retargeting_method", "output_cache_subdir")
_RETARGETING_METHOD_ENV = "TERRA_RETARGETING_METHOD"


@dataclass(frozen=True)
class TrainingPreflight:
    """Result of the installed training-stack readiness check.

    ``ready`` is true only when the selected PPO configuration, JAX GPU
    backend, and Warp CUDA devices meet the requested device count. The
    reported versions, backend names, device counts, and ``error`` make a
    failed ``terra train preflight`` actionable before launching a run.
    """

    ready: bool
    jax_version: str
    jax_backend: str
    jax_gpu_devices: tuple[str, ...]
    warp_version: str
    warp_cuda_device_count: int
    algorithm: str
    training_backend: str
    mjx_backend: str
    config_path: str
    error: str | None = None
    algorithm_key: str = "ppo"
    configured_jax_gpu_device_count: int = 1
    minimum_jax_gpu_device_count: int = 1


@dataclass(frozen=True)
class TrainingLaunch:
    """Prepared PPO subprocess and its verified motion cohort.

    ``command`` and ``environment`` can be inspected by ``terra train run
    --dry-run`` before :func:`launch_training` starts the process. Training,
    validation, and test identities come from the materialization record;
    trajectory and terrain paths have already passed artifact validation.
    """

    label: str
    materialization_record: Path
    cache_root: Path
    motion_names: tuple[str, ...]
    trajectory_paths: tuple[Path, ...]
    terrain_paths: tuple[Path | None, ...]
    command: tuple[str, ...]
    environment: dict[str, str]
    validation_motion_names: tuple[str, ...] = ()
    validation_trajectory_paths: tuple[Path, ...] = ()
    validation_terrain_paths: tuple[Path | None, ...] = ()
    test_motion_count: int = 0
    terrain_mode: str = "nonflat"
    retargeting_method: str = "terra"
    algorithm_key: str = "ppo"


def _config_path(config_name: str = TERRA_MULTI_GPU_PPO_CONFIG) -> Path:
    if config_name not in _SUPPORTED_TRAINING_CONFIGS:
        raise ValueError(f"unsupported training config: {config_name!r}")
    path = Path(__file__).resolve().parent / "rl" / "configs" / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"training config is not installed: {path}")
    return path


def _normalize_algorithm(algorithm: str) -> str:
    normalized = str(algorithm).strip().lower()
    if normalized not in TERRA_TRAINING_ALGORITHMS:
        choices = ", ".join(TERRA_TRAINING_ALGORITHMS)
        raise ValueError(f"unsupported training algorithm {algorithm!r}; choose one of: {choices}")
    return normalized


def _training_config_name(algorithm: str, *, multi_motion: bool) -> str:
    algorithm_key = _normalize_algorithm(algorithm)
    if algorithm_key == "ppo":
        if not multi_motion:
            raise ValueError("PPO training requires a materialized multi-motion selection")
        return TERRA_MULTI_GPU_PPO_CONFIG
    raise ValueError("PPO training requires a materialized selection")


def _is_multi_motion_config(config_name: str) -> bool:
    return config_name == TERRA_MULTI_GPU_PPO_CONFIG


def _compose_training_config(
    overrides: Sequence[str] = (),
    *,
    algorithm: str = "ppo",
    multi_motion: bool = True,
):
    from hydra import compose, initialize_config_dir

    from terra.rl.config_resolvers import register_config_resolvers

    register_config_resolvers()
    config_name = _training_config_name(algorithm, multi_motion=multi_motion)
    config_dir = _config_path(config_name).parent
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


@contextlib.contextmanager
def _temporary_environment(values: Mapping[str, str]):
    """Resolve launcher-managed Hydra environment values without leaking them."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def _validate_training_config(config, *, algorithm: str, config_name: str) -> str:
    from musclemimic.runner.engine import pick_algorithm, validate_training_backend

    algorithm_key = _normalize_algorithm(algorithm)
    validate_training_backend(config)
    resolved_algorithm = pick_algorithm(config).__name__
    expected_algorithm = _EXPECTED_ALGORITHM_CLASS[algorithm_key]
    if resolved_algorithm != expected_algorithm:
        raise ValueError(
            f"{config_name} resolved unexpected algorithm {resolved_algorithm!r}; expected {expected_algorithm!r}"
        )
    _validate_artifact_config(config, config_name=config_name)
    if algorithm_key == "ppo":
        _validate_ppo_config(config, config_name=config_name)
    return resolved_algorithm


def _validate_ppo_config(config, *, config_name: str = TERRA_MULTI_GPU_PPO_CONFIG) -> int:
    """Validate TERRA's single-host data-parallel PPO safety contract."""

    experiment = config.experiment
    if str(experiment.get("training_backend", "")) != "mjx_warp":
        raise ValueError(f"{config_name} must set experiment.training_backend='mjx_warp'")
    if str(experiment.env_params.get("mjx_backend", "")) != "warp":
        raise ValueError(f"{config_name} must set experiment.env_params.mjx_backend='warp'")
    distributed = experiment.get("distributed", None)
    if distributed is None or distributed.get("num_devices", None) is None:
        raise ValueError(f"{config_name} must set experiment.distributed.num_devices")
    configured_devices = int(distributed.num_devices)
    if configured_devices != -1 and configured_devices < 1:
        raise ValueError(f"{config_name} experiment.distributed.num_devices must be -1 or at least 1 for PPO")

    if int(experiment.get("n_seeds", 1)) != 1 or bool(experiment.get("vmap_across_seeds", False)):
        raise ValueError(f"{config_name} data-parallel PPO supports exactly one seed per process")

    trajectory = experiment.get("trajectory", {})
    sharding = str(trajectory.get("sharding", "none"))
    if sharding != "none":
        raise ValueError(f"{config_name} must set experiment.trajectory.sharding='none' for the initial PPO rollout")

    graph_mode = str(experiment.env_params.get("mjx_warp_graph_mode", ""))
    if graph_mode != _PPO_WARP_GRAPH_MODE:
        raise ValueError(
            f"{config_name} must set experiment.env_params.mjx_warp_graph_mode="
            f"'{_PPO_WARP_GRAPH_MODE}' for multi-GPU Warp safety"
        )
    validation = experiment.get("validation", {})
    if bool(validation.get("active", False)):
        if not bool(validation.get("evaluate_all", False)):
            raise ValueError(f"{config_name} active PPO validation must set validation.evaluate_all=true")
        if bool(validation.get("deterministic", False)):
            raise ValueError(f"{config_name} active PPO validation must set validation.deterministic=false")
        minimum_total_rollouts = int(validation.get("minimum_total_rollouts", 0))
        rollouts_per_motion = int(validation.get("rollouts_per_motion", 0))
        if minimum_total_rollouts < 100:
            raise ValueError(f"{config_name} validation.minimum_total_rollouts must be at least 100")
        if rollouts_per_motion < 1:
            raise ValueError(f"{config_name} validation.rollouts_per_motion must be at least 1")
        max_parallel_rollouts = int(validation.get("max_parallel_rollouts", 1024))
        if max_parallel_rollouts < 32 or max_parallel_rollouts % 32:
            raise ValueError(f"{config_name} validation.max_parallel_rollouts must be a multiple of 32 and at least 32")
        validation_env = validation.get("env_params", {}) or {}
        validation_backend = str(validation_env.get("mjx_backend", experiment.env_params.mjx_backend))
        validation_graph_mode = str(
            validation_env.get("mjx_warp_graph_mode", experiment.env_params.mjx_warp_graph_mode)
        )
        if validation_backend != "warp":
            raise ValueError(f"{config_name} validation environment must use mjx_backend='warp'")
        if validation_graph_mode != _PPO_WARP_GRAPH_MODE:
            raise ValueError(
                f"{config_name} validation environment must use "
                f"mjx_warp_graph_mode='{_PPO_WARP_GRAPH_MODE}' "
                "for multi-GPU Warp safety"
            )

    if configured_devices > 0:
        _validate_ppo_environment_divisibility(config, configured_devices, config_name=config_name)
    return configured_devices


def _validate_ppo_environment_divisibility(config, num_devices: int, *, config_name: str) -> None:
    environment_counts = [("experiment.num_envs", int(config.experiment.num_envs))]
    validation = config.experiment.get("validation", None)
    if validation is not None and bool(validation.get("active", False)):
        environment_counts.append(("experiment.validation.num_envs", int(validation.num_envs)))
    for setting, count in environment_counts:
        if count % num_devices:
            raise ValueError(f"{config_name} {setting} ({count}) must be divisible by num_devices ({num_devices})")
    ppo_config = config.experiment.get("ppo_config", {})
    num_steps = int(config.experiment.get("num_steps", ppo_config.get("num_steps")))
    num_minibatches = int(config.experiment.get("num_minibatches", ppo_config.get("num_minibatches")))
    local_batch_size = int(config.experiment.num_envs) // num_devices * num_steps
    if local_batch_size % num_minibatches:
        raise ValueError(
            f"{config_name} per-device rollout batch ({local_batch_size}) must be divisible by "
            f"experiment.num_minibatches ({num_minibatches})"
        )


def _unresolved_config_value(config: Mapping[str, object], path: str) -> object:
    value: object = config
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"{TERRA_MULTI_GPU_PPO_CONFIG} is missing required setting {path!r}")
        value = value[component]
    return value


def _validate_artifact_config(config, *, config_name: str = TERRA_MULTI_GPU_PPO_CONFIG) -> None:
    """Verify that the local config remains bound to published TERRA artifacts."""
    from omegaconf import OmegaConf

    unresolved = OmegaConf.to_container(config, resolve=False)
    if not isinstance(unresolved, Mapping):
        raise ValueError(f"{config_name} must resolve to a configuration object")

    multi_motion = _is_multi_motion_config(config_name)
    for dataset_path in _ARTIFACT_DATASET_SETTINGS:
        for field, expected in _REQUIRED_DATASET_VALUES.items():
            path = f"{dataset_path}.{field}"
            actual = _unresolved_config_value(unresolved, path)
            if actual != expected:
                raise ValueError(f"{config_name} must set {path}={expected!r}, got {actual!r}")
        for field in _METHOD_DATASET_FIELDS:
            path = f"{dataset_path}.{field}"
            actual = _unresolved_config_value(unresolved, path)
            if not isinstance(actual, str) or f"oc.env:{_RETARGETING_METHOD_ENV}" not in actual:
                raise ValueError(f"{config_name} setting {path} must be bound to {_RETARGETING_METHOD_ENV}")
        for field, environment_variable in (("cache_root", "TERRA_RETARGETED_MOTIONS"),):
            path = f"{dataset_path}.{field}"
            actual = _unresolved_config_value(unresolved, path)
            if not isinstance(actual, str) or f"oc.env:{environment_variable}" not in actual:
                raise ValueError(f"{config_name} setting {path} must be bound to {environment_variable}")
        motion_path = f"{dataset_path}.rel_dataset_path"
        motion_value = _unresolved_config_value(unresolved, motion_path)
        if multi_motion:
            if (
                not isinstance(motion_value, str)
                or "terra.motion_selection" not in motion_value
                or "oc.env:TERRA_MOTION_SELECTION_RECORD" not in motion_value
            ):
                raise ValueError(f"{config_name} setting {motion_path} must resolve from TERRA_MOTION_SELECTION_RECORD")
        elif not isinstance(motion_value, str) or "oc.env:TERRA_MOTION" not in motion_value:
            raise ValueError(f"{config_name} setting {motion_path} must be bound to TERRA_MOTION")
        terrain_requirement_path = f"{dataset_path}.require_nonflat_terrain"
        terrain_requirement = _unresolved_config_value(unresolved, terrain_requirement_path)
        if not isinstance(terrain_requirement, str) or "oc.env:TERRA_REQUIRE_NONFLAT_TERRAIN" not in (
            terrain_requirement
        ):
            raise ValueError(
                f"{config_name} setting {terrain_requirement_path} must be bound to TERRA_REQUIRE_NONFLAT_TERRAIN"
            )

    validation_steps = _unresolved_config_value(unresolved, "experiment.validation.num_steps")
    if not isinstance(validation_steps, str) or "oc.env:TERRA_VALIDATION_STEPS" not in validation_steps:
        raise ValueError(
            f"{config_name} setting experiment.validation.num_steps must be bound to TERRA_VALIDATION_STEPS"
        )
    if _is_multi_motion_config(config_name):
        validation_envs = _unresolved_config_value(unresolved, "experiment.validation.num_envs")
        validation_environment = "TERRA_VALIDATION_ENVS"
        if not isinstance(validation_envs, str) or f"oc.env:{validation_environment}" not in validation_envs:
            raise ValueError(
                f"{config_name} setting experiment.validation.num_envs must be bound to {validation_environment}"
            )


def _validate_hydra_overrides(
    overrides: Sequence[str],
    *,
    config_name: str = TERRA_MULTI_GPU_PPO_CONFIG,
) -> None:
    managed_settings = _MANAGED_HYDRA_SETTINGS
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if any(
            key == managed or key.startswith(f"{managed}.") or managed.startswith(f"{key}.")
            for managed in managed_settings
        ):
            raise ValueError(f"Hydra override {override!r} changes a launcher-managed artifact setting")


@contextlib.contextmanager
def _quiet_accelerator_probe():
    """Suppress native driver diagnostics during an availability probe.

    Both JAX and Warp write expected no-driver messages directly to file
    descriptor 2. Keep those diagnostics out of the JSON CLI while preserving
    exceptions for the caller to handle.
    """

    sys.stderr.flush()
    saved_stderr = os.dup(2)
    previous_logging_threshold = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w", encoding="utf-8") as null_stderr:
            os.dup2(null_stderr.fileno(), 2)
            with contextlib.redirect_stderr(null_stderr):
                yield
    finally:
        logging.disable(previous_logging_threshold)
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _ppo_warp_compatibility_error() -> str | None:
    """Return a clear integration error for an unsupported MuscleMimic revision."""
    try:
        from musclemimic.algorithms.common.ppo_distributed import assert_warp_multi_gpu_compatible
    except ImportError:
        return (
            "the installed MuscleMimic revision does not include TERRA's multi-GPU PPO runtime; "
            "install TERRA's pinned MuscleMimic terra revision before launching PPO"
        )
    try:
        assert_warp_multi_gpu_compatible()
    except RuntimeError as compatibility_error:
        return str(compatibility_error)
    return None


def training_preflight(
    *,
    algorithm: str = "ppo",
    require_cuda: bool = True,
    multi_motion: bool | None = None,
    overrides: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
) -> TrainingPreflight:
    """Validate the selected TERRA training config and its CUDA runtimes.

    Set ``require_cuda=False`` only for CPU CI and installation diagnostics. It
    validates the same configuration and software stack but reports unavailable
    accelerator hardware instead of raising.
    """

    import jax
    import warp as wp

    algorithm_key = _normalize_algorithm(algorithm)
    if multi_motion is None:
        multi_motion = algorithm_key == "ppo"
    config_name = _training_config_name(algorithm_key, multi_motion=multi_motion)
    config_path = _config_path(config_name)
    with _temporary_environment(environment or {}):
        config = _compose_training_config(overrides=overrides, algorithm=algorithm_key, multi_motion=multi_motion)
        resolved_algorithm = _validate_training_config(config, algorithm=algorithm_key, config_name=config_name)
        configured_devices = int(config.experiment.distributed.num_devices) if multi_motion else 1

    with _quiet_accelerator_probe():
        try:
            gpu_devices = tuple(str(device) for device in jax.devices("gpu"))
        except RuntimeError:
            gpu_devices = ()
        jax_backend = jax.default_backend()

        previous_warp_log_level = wp.config.log_level
        wp.config.log_level = max(previous_warp_log_level, wp.LOG_WARNING)
        try:
            warp_cuda_devices = int(wp.get_cuda_device_count())
        finally:
            wp.config.log_level = previous_warp_log_level
    minimum_devices = max(1, len(gpu_devices)) if configured_devices == -1 else configured_devices
    error = None
    if multi_motion:
        error = _ppo_warp_compatibility_error()
    if error is None and (len(gpu_devices) < minimum_devices or warp_cuda_devices < minimum_devices):
        error = (
            f"{algorithm_key} training requires at least {minimum_devices} CUDA device(s) visible to both JAX "
            f"and Warp; found {len(gpu_devices)} JAX GPU(s) and {warp_cuda_devices} Warp CUDA device(s)"
        )
    elif error is None and configured_devices == -1:
        _validate_ppo_environment_divisibility(config, len(gpu_devices), config_name=config_name)
    report = TrainingPreflight(
        ready=error is None,
        jax_version=jax.__version__,
        jax_backend=jax_backend,
        jax_gpu_devices=gpu_devices,
        warp_version=getattr(wp, "__version__", "unknown"),
        warp_cuda_device_count=warp_cuda_devices,
        algorithm=resolved_algorithm,
        training_backend=str(config.experiment.training_backend),
        mjx_backend=str(config.experiment.env_params.mjx_backend),
        config_path=str(config_path),
        error=error,
        algorithm_key=algorithm_key,
        configured_jax_gpu_device_count=configured_devices,
        minimum_jax_gpu_device_count=minimum_devices,
    )
    if require_cuda and error is not None:
        raise RuntimeError(error)
    return report


def _read_materialization_record(path: str | Path) -> tuple[Path, Mapping[str, object]]:
    record_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(record_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"materialization record is not valid JSON: {record_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"materialization record must contain a JSON object: {record_path}")
    return record_path, payload


def prepare_training_launch(
    materialization_record: str | Path,
    label: str,
    *,
    algorithm: str = "ppo",
    overrides: Sequence[str] = (),
    python_executable: str | Path | None = None,
) -> TrainingLaunch:
    """Validate a materialization record and prepare a PPO process.

    The record comes from ``terra train materialize`` and names published
    trajectory, analysis, and optional terrain artifacts. Every train and
    evaluation entry is revalidated before a :class:`TrainingLaunch` is
    returned. A test split contributes identities only, and when no
    evaluation split is supplied, training motions are reused for startup
    validation. This function does not start a subprocess; pass the result
    to :func:`launch_training` or use ``terra train run``.
    """

    from terra.artifacts import validate_retarget_artifacts

    algorithm_key = _normalize_algorithm(algorithm)
    config_name = _training_config_name(algorithm_key, multi_motion=True)
    if _SAFE_TRAINING_LABEL.fullmatch(label) is None:
        raise ValueError("training label must contain only letters, digits, '.', '_', and '-'")

    record_path, payload = _read_materialization_record(materialization_record)
    cache_value = payload.get("destination_cache")
    if not isinstance(cache_value, str) or not cache_value:
        raise ValueError("materialization record destination_cache must be a non-empty path")
    cache_root = Path(cache_value).expanduser()
    if not cache_root.is_absolute():
        raise ValueError("materialization record destination_cache must be absolute")
    cache_root = cache_root.resolve()

    rows = payload.get("motions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("materialization record must contain at least one motion")
    terrain_mode = payload.get("terrain_mode", "nonflat")
    if terrain_mode not in {"flat", "nonflat", "mixed"}:
        raise ValueError("materialization record terrain_mode must be 'flat', 'nonflat', or 'mixed'")
    retargeting_method = validate_method(str(payload.get("retargeting_method", "terra")))

    motion_names: list[str] = []
    trajectory_paths: list[Path] = []
    terrain_paths: list[Path | None] = []
    motion_frame_counts: list[int] = []
    evaluation_motion_names: list[str] = []
    evaluation_trajectory_paths: list[Path] = []
    evaluation_terrain_paths: list[Path | None] = []
    evaluation_frame_counts: list[int] = []
    test_motion_count = 0
    seen: set[str] = set()
    normalized_seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"materialization motion row {index} must be an object")
        required_text = ("motion", "dataset")
        invalid = [field for field in required_text if not isinstance(row.get(field), str) or not row[field]]
        if invalid:
            raise ValueError(f"materialization motion row {index} has empty or missing fields: {', '.join(invalid)}")
        motion_name = str(row["motion"])
        split = str(row.get("split", "train"))
        if split not in {"train", "evaluation", "test"}:
            raise ValueError(
                f"materialization motion row {index} has invalid split {split!r}; "
                "expected 'train', 'evaluation', or 'test'"
            )
        if motion_name in seen:
            raise ValueError(f"materialization record contains duplicate motion: {motion_name}")
        seen.add(motion_name)
        if split == "test":
            test_motion_count += 1
            continue
        recorded_paths = row.get("paths")
        if not isinstance(recorded_paths, Mapping):
            raise ValueError(f"materialization motion row {index} is missing artifact paths")
        required_path_fields = ["trajectory_relpath", "analysis_relpath"]
        if terrain_mode == "nonflat":
            required_path_fields.append("terrain_relpath")
        missing_paths = [
            field
            for field in required_path_fields
            if not isinstance(recorded_paths.get(field), str) or not recorded_paths[field]
        ]
        if missing_paths:
            raise ValueError(
                f"materialization motion row {index} has empty or missing artifact paths: {', '.join(missing_paths)}"
            )

        validated = validate_retarget_artifacts(
            cache_root,
            motion_name,
            method=retargeting_method,
            require_nonflat_terrain=terrain_mode == "nonflat",
        )
        if terrain_mode == "nonflat" and validated.terrain_path is None:
            raise RuntimeError(f"validated retargeting artifacts have no terrain path: {motion_name}")
        if terrain_mode == "flat" and validated.nonflat_terrain:
            raise ValueError(f"flat materialization record contains non-flat terrain: {motion_name}")
        if validated.motion_name in normalized_seen:
            raise ValueError(f"materialization record contains duplicate normalized motion: {validated.motion_name}")
        normalized_seen.add(validated.motion_name)
        expected_paths = {
            "trajectory_relpath": validated.trajectory_path,
            "analysis_relpath": validated.analysis_path,
        }
        recorded_terrain_path = recorded_paths.get("terrain_relpath")
        if validated.terrain_path is None:
            if recorded_terrain_path not in (None, ""):
                raise ValueError(f"materialization motion row {index} records terrain metadata absent from its cache")
        else:
            if not isinstance(recorded_terrain_path, str) or not recorded_terrain_path:
                raise ValueError(
                    f"materialization motion row {index} has empty or missing artifact paths: terrain_relpath"
                )
            expected_paths["terrain_relpath"] = validated.terrain_path
        for field, expected_path in expected_paths.items():
            recorded_path = Path(str(recorded_paths[field])).expanduser().resolve()
            if recorded_path != expected_path:
                raise ValueError(f"materialization motion row {index} {field} does not match its cache path")
        if split == "train":
            motion_names.append(validated.motion_name)
            trajectory_paths.append(validated.trajectory_path)
            terrain_paths.append(validated.terrain_path)
            motion_frame_counts.append(validated.num_frames)
        else:
            evaluation_motion_names.append(validated.motion_name)
            evaluation_trajectory_paths.append(validated.trajectory_path)
            evaluation_terrain_paths.append(validated.terrain_path)
            evaluation_frame_counts.append(validated.num_frames)

    if not motion_names:
        raise ValueError("materialization record must contain at least one train motion")
    if not evaluation_motion_names:
        evaluation_motion_names = list(motion_names)
        evaluation_trajectory_paths = list(trajectory_paths)
        evaluation_terrain_paths = list(terrain_paths)
        evaluation_frame_counts = list(motion_frame_counts)

    hydra_overrides = tuple(str(value) for value in overrides)
    if any(not value for value in hydra_overrides):
        raise ValueError("Hydra overrides cannot be empty")
    _validate_hydra_overrides(hydra_overrides, config_name=config_name)
    validation_envs = max(32, ((len(evaluation_motion_names) + 31) // 32) * 32)
    if algorithm_key == "ppo":
        from terra.rl.ppo_validation import (
            stochastic_validation_rollout_count,
            validation_lane_count,
        )

        validation_envs = validation_lane_count(stochastic_validation_rollout_count(len(evaluation_motion_names)))
    environment = {
        "TERRA_RETARGETED_MOTIONS": str(cache_root),
        "TERRA_MOTION_SELECTION_RECORD": str(record_path),
        "TERRA_SELECTION_SIZE": str(len(motion_names)),
        "TERRA_VALIDATION_SIZE": str(len(evaluation_motion_names)),
        "TERRA_TEST_SIZE": str(test_motion_count),
        "TERRA_VALIDATION_ENVS": str(validation_envs),
        "TERRA_VALIDATION_STEPS": str(max(evaluation_frame_counts)),
        "TERRA_TRAINING_SUBSET": label,
        "TERRA_REQUIRE_NONFLAT_TERRAIN": "true" if terrain_mode == "nonflat" else "false",
        _RETARGETING_METHOD_ENV: retargeting_method,
        "TERRA_TRAJECTORY_CACHE_ROOT": str(
            Path(os.environ.get("TERRA_ARTIFACT_ROOT", cache_root)).expanduser().resolve()
            / "training"
            / "trajectory-cache"
        ),
    }
    with _temporary_environment(environment):
        composed_config = _compose_training_config(
            hydra_overrides,
            algorithm=algorithm_key,
            multi_motion=True,
        )
    if algorithm_key == "ppo":
        validation_config = composed_config.experiment.validation
        rollout_count = stochastic_validation_rollout_count(
            len(evaluation_motion_names),
            minimum_total_rollouts=int(validation_config.minimum_total_rollouts),
            rollouts_per_motion=int(validation_config.rollouts_per_motion),
        )
        environment["TERRA_VALIDATION_ENVS"] = str(
            validation_lane_count(
                rollout_count,
                max_parallel_rollouts=int(validation_config.max_parallel_rollouts),
            )
        )
        with _temporary_environment(environment):
            composed_config = _compose_training_config(
                hydra_overrides,
                algorithm=algorithm_key,
                multi_motion=True,
            )
    else:
        configured_devices = int(composed_config.experiment.distributed.num_devices)
        if configured_devices > 0:
            environment["TERRA_VALIDATION_ENVS"] = str(
                (len(evaluation_motion_names) + configured_devices - 1) // configured_devices
            )
            with _temporary_environment(environment):
                composed_config = _compose_training_config(
                    hydra_overrides,
                    algorithm=algorithm_key,
                    multi_motion=True,
                )
    with _temporary_environment(environment):
        _validate_training_config(
            composed_config,
            algorithm=algorithm_key,
            config_name=config_name,
        )
    command = (
        str(python_executable or sys.executable),
        "-m",
        "terra.rl.experiment",
        f"--config-name={config_name}",
        *hydra_overrides,
    )
    return TrainingLaunch(
        label=label,
        materialization_record=record_path,
        cache_root=cache_root,
        motion_names=tuple(motion_names),
        trajectory_paths=tuple(trajectory_paths),
        terrain_paths=tuple(terrain_paths),
        validation_motion_names=tuple(evaluation_motion_names),
        validation_trajectory_paths=tuple(evaluation_trajectory_paths),
        validation_terrain_paths=tuple(evaluation_terrain_paths),
        test_motion_count=test_motion_count,
        command=command,
        environment=environment,
        terrain_mode=str(terrain_mode),
        retargeting_method=retargeting_method,
        algorithm_key=algorithm_key,
    )


def launch_training(launch: TrainingLaunch) -> int:
    """Require CUDA, then run a validated training experiment in a child process."""

    training_preflight(
        algorithm=launch.algorithm_key,
        require_cuda=True,
        overrides=launch.command[4:],
        environment=launch.environment,
    )
    environment = os.environ.copy()
    if "TERRA_MOTION_SELECTION_RECORD" in launch.environment:
        # File-backed cohort selection supersedes the legacy list variables.
        # Removing inherited copies also prevents unrelated stale values from
        # making a large-cohort exec fail with E2BIG.
        environment.pop("TERRA_MOTIONS", None)
        environment.pop("TERRA_VALIDATION_MOTIONS", None)
    # These values are consumed when the child imports JAX. Respect operator overrides.
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    environment.update(launch.environment)
    completed = subprocess.run(launch.command, env=environment, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public CUDA readiness check."""

    parser = argparse.ArgumentParser(
        prog="terra train preflight",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=TERRA_TRAINING_ALGORITHMS,
        default="ppo",
        help="training algorithm",
    )
    parser.add_argument(
        "--allow-no-device",
        action="store_true",
        help="report config/software readiness without requiring local CUDA hardware",
    )
    args = parser.parse_args(argv)
    try:
        report = training_preflight(
            algorithm=args.algorithm,
            require_cuda=False,
            multi_motion=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.ready or args.allow_no_device else 1


def config_overrides(value: Mapping[str, object], prefix: str = "") -> list[str]:
    """Convert a nested training configuration to typed Hydra overrides."""
    result = []
    for key, child in value.items():
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"invalid configuration key: {key!r}")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping) and child:
            result.extend(config_overrides(child, path))
        else:
            result.append(f"{path}={json.dumps(child, separators=(',', ':'), allow_nan=False)}")
    return result


def run_main(argv: Sequence[str] | None = None) -> int:
    """Validate a materialized motion selection and launch PPO."""
    parser = argparse.ArgumentParser(prog="terra train run", description=run_main.__doc__)
    parser.add_argument("--algorithm", choices=TERRA_TRAINING_ALGORITHMS, default="ppo")
    parser.add_argument("--materialization-record", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", type=Path, help="YAML experiment settings")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="append", default=[], dest="explicit_overrides")
    parser.add_argument("overrides", nargs="*", help="Hydra configuration overrides")
    args = parser.parse_intermixed_args(argv)
    try:
        settings = []
        if args.config:
            import yaml

            payload = yaml.safe_load(args.config.read_text())
            if not isinstance(payload, Mapping):
                raise ValueError("training config must contain a mapping")
            settings = config_overrides(payload)
        launch = prepare_training_launch(
            args.materialization_record,
            args.label,
            algorithm=args.algorithm,
            overrides=[*settings, *args.explicit_overrides, *args.overrides],
        )
        if args.dry_run:
            payload = asdict(launch)
            payload["preflight"] = asdict(
                training_preflight(
                    algorithm=launch.algorithm_key,
                    require_cuda=False,
                    overrides=launch.command[4:],
                    environment=launch.environment,
                )
            )
            print(json.dumps(payload, indent=2, default=str))
            return 0
        return launch_training(launch)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
