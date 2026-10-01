"""Evaluate repeated stochastic PPO rollouts on the MJX publication backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.runner.engine import build_metrics_handler, instantiate_validation_env, pick_algorithm
from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint
from terra.rl import register_components
from terra.rl.backend import install_backend_integrations
from terra.rl.collision_profiles import SELF_COLLISION_MODES, merge_disabled_contact_pairs
from terra.rl.ppo_validation import (
    TerraValidationSummary as _TerraValidationSummary,
)
from terra.rl.ppo_validation import (
    evaluate_policy_exhaustive,
    validation_lane_count,
)
from terra.rl.validation_render import _checkpoint_timestep, _upgrade_legacy_observation_config

_SUMMARY_TYPE = _TerraValidationSummary


DEFAULT_STOCHASTIC_ROLLOUTS_PER_MOTION = 5
BENCHMARK_METRICS = (
    "success",
    "early_termination",
    "horizon_timeout",
    "coverage",
    "return_per_frame",
    "tracking_error_m",
    "activation_energy",
    "episode_length",
)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A materialized held-out split independent of checkpoint configuration."""

    manifest: Path
    manifest_sha256: str
    split: str
    cache_root: Path
    terrain_mode: str
    retargeting_method: str
    motion_paths: tuple[str, ...]
    source_caches: dict[str, str | list[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_retargeting_method(config: Any) -> str:
    dataset = config.experiment.validation.get("amass_dataset_conf", None)
    if dataset is None:
        dataset = config.experiment.task_factory.params.get("amass_dataset_conf", {})
    return str(dataset.get("retargeting_method", "terra"))


def _materialize_evaluation_dataset(
    manifest: str | Path,
    cache_root: str | Path,
    *,
    split: str,
    terrain_mode: str,
    retargeting_method: str,
    link_mode: str,
) -> EvaluationDataset:
    """Materialize exactly one declared post-training evaluation split."""
    from terra.commands.materialize import materialize_subset, read_manifest

    manifest_path = Path(manifest).expanduser().resolve()
    rows = read_manifest(manifest_path)
    selected = [row for row in rows if str(row.get("split", "train")) == split]
    if not selected:
        raise ValueError(f"dataset manifest contains no {split!r} motions: {manifest_path}")
    payload = materialize_subset(
        [{**row, "split": "train"} for row in selected],
        Path(cache_root),
        link_mode=link_mode,
        method=retargeting_method,
        terrain_mode=terrain_mode,
    )
    motion_paths = tuple(str(row["motion"]) for row in payload["motions"])
    expected = tuple(str(row["motion"]) for row in selected)
    if motion_paths != expected:
        raise RuntimeError("evaluation materialization changed the ordered dataset")
    return EvaluationDataset(
        manifest=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        split=split,
        cache_root=Path(str(payload["destination_cache"])).resolve(),
        terrain_mode=terrain_mode,
        retargeting_method=retargeting_method,
        motion_paths=motion_paths,
        source_caches=dict(payload["source_caches"]),
    )


def _configure_evaluation_dataset(config: Any, dataset: EvaluationDataset) -> None:
    dataset_values = {
        "rel_dataset_path": list(dataset.motion_paths),
        "dataset_group": None,
        "retargeting_method": dataset.retargeting_method,
        "output_cache_subdir": dataset.retargeting_method,
        "cache_root": str(dataset.cache_root),
        "load_paired_terrain": True,
        "require_nonflat_terrain": dataset.terrain_mode == "nonflat",
        "clear_cache": False,
        "allow_cache_download": False,
        "trajectory_cache_type": "sparse",
    }
    with open_dict(config.experiment.task_factory.params):
        config.experiment.task_factory.params.amass_dataset_conf = dataset_values
        config.experiment.task_factory.params.trajectory_cache_root = ""
        config.experiment.task_factory.params.trajectory_cache_key = ""
    with open_dict(config.experiment.validation):
        config.experiment.validation.amass_dataset_conf = dataset_values
        config.experiment.validation.trajectory_cache_root = ""
        config.experiment.validation.trajectory_cache_key = ""


def _motion_paths(config: Any) -> list[str]:
    """Return the ordered motion list used by the checkpoint's validation env."""
    experiment = config.experiment
    validation = experiment.get("validation", {})
    dataset_config = validation.get("amass_dataset_conf", None)
    if dataset_config is None:
        dataset_config = experiment.task_factory.params.get("amass_dataset_conf", None)
    if dataset_config is None:
        raise ValueError("checkpoint has no AMASS validation dataset configuration")
    paths = dataset_config.get("rel_dataset_path", None)
    if isinstance(paths, str):
        paths = [paths]
    elif OmegaConf.is_list(paths):
        paths = list(paths)
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("checkpoint validation dataset must contain an explicit ordered motion list")
    if len(paths) != len(set(paths)):
        raise ValueError("checkpoint validation dataset contains duplicate motions")
    return paths


def _finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {field}: {result}")
    return result


def _configure_terrain_collision_margin(config: Any, value: float) -> float:
    """Apply one symmetric terrain contact-onset margin to an evaluation."""
    margin = float(value)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("terrain_collision_margin must be finite and non-negative")
    with open_dict(config.experiment.task_factory.params):
        config.experiment.task_factory.params.terrain_collision_margin = margin
    return margin


def _configure_termination_threshold(config: Any, value: float | None) -> dict[str, float]:
    """Optionally set and return the publication tracking thresholds."""
    validation = config.experiment.validation
    if validation.get("terminal_state_params", None) is None:
        with open_dict(validation):
            validation.terminal_state_params = {}
    terminal = validation.terminal_state_params
    if value is not None:
        threshold = float(value)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("termination_threshold must be finite and positive")
        with open_dict(terminal):
            terminal.mean_site_deviation_threshold = threshold
            terminal.core_upper_body_mean_site_deviation_threshold = threshold
    return {
        "global_mpjpe": _finite_float(
            terminal.mean_site_deviation_threshold,
            "global MPJPE termination threshold",
        ),
        "core_upper_body_mpjpe": _finite_float(
            terminal.core_upper_body_mean_site_deviation_threshold,
            "core-upper-body MPJPE termination threshold",
        ),
    }


def _summary_arrays(summary: Any, motion_count: int) -> dict[str, np.ndarray]:
    arrays = {
        "success": np.asarray(summary.per_motion_success, dtype=np.float64),
        "early_termination": np.asarray(summary.per_motion_early_termination, dtype=np.float64),
        "horizon_timeout": np.asarray(summary.per_motion_horizon_timeout, dtype=np.float64),
        "coverage": np.asarray(summary.per_motion_frame_coverage, dtype=np.float64),
        "return_per_frame": np.asarray(summary.per_motion_return_per_frame, dtype=np.float64),
        "tracking_error_m": np.asarray(summary.per_motion_tracking_error_m, dtype=np.float64),
        "activation_energy": np.asarray(summary.per_motion_activation_energy, dtype=np.float64),
        "episode_length": np.asarray(summary.per_motion_episode_length, dtype=np.float64),
    }
    for name, values in arrays.items():
        if values.shape != (motion_count,):
            raise RuntimeError(f"validation {name} shape {values.shape} does not match {motion_count} motions")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"validation {name} contains NaN or Inf")
    return arrays


def _aggregate(
    motion_paths: Sequence[str],
    stochastic_runs: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Aggregate stochastic repeats, then equal-weight motions."""
    if not stochastic_runs:
        raise ValueError("at least one stochastic rollout is required")
    motion_count = len(motion_paths)
    stochastic = {
        key: np.stack([np.asarray(run[key], dtype=np.float64) for run in stochastic_runs], axis=0)
        for key in BENCHMARK_METRICS
    }
    for name, values in stochastic.items():
        if values.shape != (len(stochastic_runs), motion_count):
            raise ValueError(f"stochastic {name} shape does not match repeats x motions")

    per_motion = {key: np.mean(values, axis=0) for key, values in stochastic.items()}
    summary = {
        "completion_rate": _finite_float(np.mean(per_motion["success"]), "completion rate"),
        "early_termination_rate": _finite_float(np.mean(per_motion["early_termination"]), "early termination rate"),
        "horizon_timeout_rate": _finite_float(np.mean(per_motion["horizon_timeout"]), "horizon timeout rate"),
        "mean_frame_coverage": _finite_float(np.mean(per_motion["coverage"]), "mean frame coverage"),
        "tracking_error_m": _finite_float(np.mean(per_motion["tracking_error_m"]), "tracking error"),
        "tracking_error_mm": _finite_float(1000.0 * np.mean(per_motion["tracking_error_m"]), "tracking error"),
        "activation_energy": _finite_float(np.mean(per_motion["activation_energy"]), "activation energy"),
        "mean_return_per_frame": _finite_float(np.mean(per_motion["return_per_frame"]), "return per frame"),
        "mean_episode_length": _finite_float(np.mean(per_motion["episode_length"]), "episode length"),
    }
    summary.update(
        {
            "completion_percent": 100.0 * summary["completion_rate"],
            "early_termination_percent": 100.0 * summary["early_termination_rate"],
            "horizon_timeout_percent": 100.0 * summary["horizon_timeout_rate"],
        }
    )
    motions = []
    for index, path in enumerate(motion_paths):
        rollouts = []
        for repeat in range(len(stochastic_runs)):
            rollout = {
                key: _finite_float(stochastic[key][repeat, index], f"stochastic {key}") for key in BENCHMARK_METRICS
            }
            for key in ("success", "early_termination", "horizon_timeout"):
                rollout[key] = bool(rollout[key] >= 0.5)
            rollouts.append(rollout)
        motions.append(
            {
                "index": index,
                "path": path,
                "stochastic": {
                    "summary": {
                        key: _finite_float(values[index], f"per-motion {key}") for key, values in per_motion.items()
                    },
                    "rollouts": rollouts,
                },
            }
        )
    return {
        "summary": {
            "motion_count": motion_count,
            "stochastic_rollouts_per_motion": len(stochastic_runs),
            "stochastic": summary,
        },
        "motions": motions,
    }


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    dataset: str | Path | None = None,
    dataset_cache: str | Path | None = None,
    dataset_split: str = "test",
    terrain_mode: str = "mixed",
    retargeting_method: str | None = None,
    link_mode: str = "hardlink",
    repeats: int = DEFAULT_STOCHASTIC_ROLLOUTS_PER_MOTION,
    first_seed: int = 0,
    num_envs: int | None = None,
    mjx_backend: str = "warp",
    terrain_collision_margin: float = 0.0,
    termination_threshold: float | None = None,
    disabled_contact_pairs: Sequence[Sequence[str]] = (),
    self_collision_mode: str = "all",
) -> dict[str, Any]:
    """Run repeated frame-zero stochastic sweeps on MJX-Warp by default."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if first_seed < 0:
        raise ValueError("first_seed must be non-negative")
    if mjx_backend not in {"jax", "warp"}:
        raise ValueError("mjx_backend must be 'jax' or 'warp'")
    if (dataset is None) != (dataset_cache is None):
        raise ValueError("dataset and dataset_cache must be provided together")
    if dataset_split not in {"train", "evaluation", "test"}:
        raise ValueError("dataset_split must be 'train', 'evaluation', or 'test'")
    if terrain_mode not in {"flat", "nonflat", "mixed"}:
        raise ValueError("terrain_mode must be 'flat', 'nonflat', or 'mixed'")
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError("link_mode must be 'hardlink' or 'copy'")

    checkpoint = str(Path(checkpoint).expanduser())
    config, raw_agent_state, metadata = load_checkpoint(checkpoint)
    OmegaConf.set_struct(config, False)
    _upgrade_legacy_observation_config(config)
    termination_thresholds = _configure_termination_threshold(config, termination_threshold)
    selected_method = retargeting_method or _checkpoint_retargeting_method(config)
    evaluation_dataset = None
    if dataset is not None:
        evaluation_dataset = _materialize_evaluation_dataset(
            dataset,
            dataset_cache,
            split=dataset_split,
            terrain_mode=terrain_mode,
            retargeting_method=selected_method,
            link_mode=link_mode,
        )
        _configure_evaluation_dataset(config, evaluation_dataset)
    terrain_collision_margin = _configure_terrain_collision_margin(config, terrain_collision_margin)
    checkpoint_contact_pairs = config.experiment.env_params.get("disabled_contact_pairs", ())
    contact_pairs = merge_disabled_contact_pairs(
        [*checkpoint_contact_pairs, *disabled_contact_pairs],
        self_collision_mode=self_collision_mode,
    )
    with open_dict(config.experiment.env_params):
        config.experiment.env_params.disabled_contact_pairs = contact_pairs
    motion_paths = _motion_paths(config)
    maximum_lanes = int(config.experiment.validation.get("max_parallel_rollouts", 1024))
    lane_count = (
        validation_lane_count(len(motion_paths), max_parallel_rollouts=maximum_lanes)
        if num_envs is None
        else int(num_envs)
    )
    if lane_count < 1:
        raise ValueError("num_envs must be positive")

    register_components()
    algorithm_cls = pick_algorithm(config)
    if algorithm_cls.__name__ != "PPOJax":
        raise ValueError(f"stochastic validation currently requires PPOJax, got {algorithm_cls.__name__}")
    install_backend_integrations("PPOJax")
    with open_dict(config.experiment.validation):
        config.experiment.validation.active = True
        config.experiment.validation.evaluate_all = True
        config.experiment.validation.start_from_beginning = True
        config.experiment.validation.deterministic = False
        # One exhaustive sweep supplies exactly one rollout per motion. Repeated
        # sweeps use independent, recorded seeds and are aggregated afterwards.
        config.experiment.validation.minimum_total_rollouts = 1
        config.experiment.validation.rollouts_per_motion = 1
        config.experiment.validation.num_envs = lane_count
        config.experiment.validation.global_num_envs = lane_count
        if config.experiment.validation.get("env_params", None) is None:
            config.experiment.validation.env_params = {}
    with open_dict(config.experiment.validation.env_params):
        config.experiment.validation.env_params.mjx_backend = mjx_backend

    env = None
    try:
        env = instantiate_validation_env(config)
        if int(env.th.n_trajectories) != len(motion_paths):
            raise RuntimeError(
                f"validation environment loaded {env.th.n_trajectories} trajectories for {len(motion_paths)} paths"
            )
        agent_conf = algorithm_cls.init_agent_conf(env, config)
        agent_state = align_agent_state(raw_agent_state, agent_conf)
        wrapped_env = wrap_env(env, config.experiment)
        metrics_handler = build_metrics_handler(config, wrapped_env)
        if metrics_handler is None:
            raise RuntimeError("validation metrics handler was not created")

        def evaluate(train_state, rng):
            return evaluate_policy_exhaustive(
                train_state,
                rng,
                wrapped_env,
                config.experiment,
                metrics_handler,
            )[0]

        stochastic_evaluator = jax.jit(evaluate)
        stochastic_runs = []
        seeds = list(range(first_seed, first_seed + repeats))
        for rollout_number, seed in enumerate(seeds, start=1):
            print(f"[StochasticValidation] rollout {rollout_number}/{repeats}, seed={seed}", flush=True)
            summary = jax.device_get(stochastic_evaluator(agent_state.train_state, jax.random.key(seed)))
            stochastic_runs.append(_summary_arrays(summary, len(motion_paths)))
        report = _aggregate(motion_paths, stochastic_runs)
        dataset_report = None
        if evaluation_dataset is not None:
            dataset_report = {
                **asdict(evaluation_dataset),
                "manifest": str(evaluation_dataset.manifest),
                "cache_root": str(evaluation_dataset.cache_root),
                "motion_paths": list(evaluation_dataset.motion_paths),
            }
        report.update(
            {
                "schema_version": 2,
                "evaluator_sha256": _sha256(Path(__file__).resolve()),
                "checkpoint": checkpoint,
                "checkpoint_global_timestep": _checkpoint_timestep(metadata),
                "backend": f"MJX-{mjx_backend.upper()}",
                "dataset": dataset_report,
                "dataset_source": "explicit_manifest" if dataset_report is not None else "checkpoint_validation",
                "disabled_contact_pairs": contact_pairs,
                "self_collision_mode": self_collision_mode,
                "terrain_collision_margin": terrain_collision_margin,
                "termination_thresholds_m": termination_thresholds,
                "evaluation_lanes": lane_count,
                "metric_definitions": {
                    "completion": "natural trajectory-end stochastic rollouts / attempted stochastic rollouts",
                    "early_termination": "absorbing stochastic rollouts / attempted stochastic rollouts",
                    "horizon_timeout": "stochastic rollouts without a terminal signal at the evaluation horizon / attempted stochastic rollouts",
                    "tracking_error_m": "per-step global mimic-site MPJPE; rollout mean, repeat mean, then equal-weight motion mean",
                    "activation_energy": "per-step mean squared muscle activation; rollout mean, repeat mean, then equal-weight motion mean",
                    "return_per_frame": "episode return divided by executed policy steps; repeat mean, then equal-weight motion mean",
                },
                "stochastic_seeds": seeds,
            }
        )
        return report
    finally:
        if env is not None:
            env.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Complete PPO checkpoint directory")
    parser.add_argument("--output", required=True, help="Destination JSON report")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Explicit selection CSV; the requested split is materialized independently of the checkpoint",
    )
    parser.add_argument("--dataset-cache", type=Path, help="Unified cache used to materialize --dataset")
    parser.add_argument(
        "--dataset-split",
        choices=("train", "evaluation", "test"),
        default="test",
        help="Selection-manifest split to evaluate",
    )
    parser.add_argument(
        "--terrain-mode",
        choices=("flat", "nonflat", "mixed"),
        default="mixed",
        help="Terrain contract for explicit dataset materialization",
    )
    parser.add_argument(
        "--retargeting-method",
        choices=("terra", "gmr", "smpl", "omniretarget"),
        help="Artifact method for --dataset; defaults to the checkpoint configuration",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How explicit dataset artifacts are materialized",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_STOCHASTIC_ROLLOUTS_PER_MOTION,
        help="Stochastic rollouts per motion",
    )
    parser.add_argument("--first-seed", type=int, default=0, help="First stochastic evaluation seed")
    parser.add_argument("--num-envs", type=int, default=None, help="Exhaustive validation lanes")
    parser.add_argument(
        "--mjx-backend",
        choices=("jax", "warp"),
        default="warp",
        help="MJX implementation used for the controlled evaluation",
    )
    parser.add_argument(
        "--terrain-collision-margin",
        type=float,
        default=0.0,
        help="Symmetric collision-onset margin, in metres, on each paired terrain box",
    )
    parser.add_argument(
        "--termination-threshold",
        type=float,
        default=None,
        help="Override both global and core-upper-body MPJPE termination thresholds, in metres",
    )
    parser.add_argument(
        "--self-collision-mode",
        choices=SELF_COLLISION_MODES,
        default="all",
        help="Keep all self-collisions, lower-body pairs only, or no explicit self-collisions",
    )
    parser.add_argument(
        "--disable-contact-pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("GEOM_A", "GEOM_B"),
        help="Remove one explicit MuJoCo contact pair before compiling the evaluation model; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_checkpoint(
        args.checkpoint,
        dataset=args.dataset,
        dataset_cache=args.dataset_cache,
        dataset_split=args.dataset_split,
        terrain_mode=args.terrain_mode,
        retargeting_method=args.retargeting_method,
        link_mode=args.link_mode,
        repeats=args.repeats,
        first_seed=args.first_seed,
        num_envs=args.num_envs,
        mjx_backend=args.mjx_backend,
        terrain_collision_margin=args.terrain_collision_margin,
        termination_threshold=args.termination_threshold,
        disabled_contact_pairs=args.disable_contact_pair,
        self_collision_mode=args.self_collision_mode,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"[StochasticValidation] report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
