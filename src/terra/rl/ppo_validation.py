"""TERRA's repeated stochastic, exhaustive PPO validation."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from musclemimic.utils import metrics as backend_metrics
from musclemimic.utils.metrics import QuantityContainer, ValidationSummary

DEFAULT_MINIMUM_TOTAL_ROLLOUTS = 100
DEFAULT_ROLLOUTS_PER_MOTION = 3
DEFAULT_MAX_PARALLEL_ROLLOUTS = 1024
PUBLICATION_STEP_METRIC_KEYS = ("err_site_abs", "activation_energy_raw")
_PROGRESS_BAR_WIDTH = 24

logger = logging.getLogger(__name__)


def _format_progress_bar(completed: int, total: int, *, width: int = _PROGRESS_BAR_WIDTH) -> str:
    """Return a compact fixed-width progress bar for durable cluster logs."""
    bounded_total = max(1, int(total))
    bounded_completed = min(max(0, int(completed)), bounded_total)
    filled = min(width, int(width * bounded_completed / bounded_total))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


class _ValidationProgressReporter:
    """Host callback that reports globally completed validation chunks once."""

    def __init__(
        self,
        *,
        motion_count: int,
        rollout_count: int,
        chunk_count: int,
        debug: bool,
    ) -> None:
        self.motion_count = int(motion_count)
        self.rollout_count = int(rollout_count)
        self.chunk_count = int(chunk_count)
        self.debug = bool(debug)
        self._lock = threading.Lock()
        self._validation_started_at = 0.0
        self._last_chunk_finished_at = 0.0
        self._last_chunk_index = -1

    def start(self, replica_index: np.ndarray, token: np.ndarray) -> np.ndarray:
        """Start a reporting session before the first validation chunk."""
        next_token = np.asarray(int(token) + 1, dtype=np.int32)
        if int(replica_index) != 0:
            return next_token
        now = time.monotonic()
        with self._lock:
            self._validation_started_at = now
            self._last_chunk_finished_at = now
            self._last_chunk_index = -1
            logger.info(
                "PPO validation progress %s 0/%s motions (0.0%%; chunk 0/%d)",
                _format_progress_bar(0, self.motion_count),
                f"{self.motion_count:,}",
                self.chunk_count,
            )
        return next_token

    def __call__(
        self,
        replica_index: np.ndarray,
        chunk_index: np.ndarray,
        completed_rollouts: np.ndarray,
        active_rollouts: np.ndarray,
        length_sum: np.ndarray,
        max_length: np.ndarray,
        token: np.ndarray,
    ) -> np.ndarray:
        """Log replica-zero progress after all replicas finish a chunk."""
        next_token = np.asarray(int(token) + 1, dtype=np.int32)
        if int(replica_index) != 0:
            return next_token

        current_chunk = int(chunk_index)
        now = time.monotonic()
        with self._lock:
            # Defensive fallback for direct calls that bypass ``start``.
            if self._validation_started_at == 0.0:
                self._validation_started_at = now
                self._last_chunk_finished_at = now
                self._last_chunk_index = -1
            if current_chunk <= self._last_chunk_index:
                return next_token

            completed = min(int(completed_rollouts), self.rollout_count)
            covered_motions = min(completed, self.motion_count)
            fraction = covered_motions / self.motion_count
            logger.info(
                "PPO validation progress %s %s/%s motions (%.1f%%; chunk %d/%d)",
                _format_progress_bar(covered_motions, self.motion_count),
                f"{covered_motions:,}",
                f"{self.motion_count:,}",
                100.0 * fraction,
                current_chunk + 1,
                self.chunk_count,
            )
            if self.debug:
                chunk_seconds = now - self._last_chunk_finished_at
                total_seconds = now - self._validation_started_at
                active = int(active_rollouts)
                mean_length = float(length_sum) / max(active, 1)
                logger.info(
                    "PPO validation debug chunk=%d/%d rollouts=%d/%d "
                    "active=%d mean_steps=%.1f max_steps=%d "
                    "chunk_seconds=%.3f total_seconds=%.3f",
                    current_chunk + 1,
                    self.chunk_count,
                    completed,
                    self.rollout_count,
                    active,
                    mean_length,
                    int(max_length),
                    chunk_seconds,
                    total_seconds,
                )
            self._last_chunk_finished_at = now
            self._last_chunk_index = current_chunk
        return next_token


@struct.dataclass
class TerraValidationSummary(ValidationSummary):
    """Validation summary with equal-weight per-motion publication metrics."""

    per_motion_early_termination: jax.Array = struct.field(default_factory=lambda: jnp.empty(0))
    per_motion_horizon_timeout: jax.Array = struct.field(default_factory=lambda: jnp.empty(0))
    per_motion_episode_length: jax.Array = struct.field(default_factory=lambda: jnp.empty(0))
    per_motion_tracking_error_m: jax.Array = struct.field(default_factory=lambda: jnp.empty(0))
    per_motion_activation_energy: jax.Array = struct.field(default_factory=lambda: jnp.empty(0))


def exhaustive_zero_summary(metrics_handler: Any) -> TerraValidationSummary:
    """Return the skip branch for TERRA's exhaustive validation summary.

    JAX traces both branches of the validation ``cond``, including updates where
    validation is not scheduled.  The skip value must therefore have exactly the
    same pytree type and per-motion shapes as :func:`evaluate_policy_exhaustive`.
    """
    motion_count = int(metrics_handler._trajectory_handler.n_trajectories)
    zeros = jnp.zeros((motion_count,), dtype=jnp.float32)
    base = metrics_handler.get_zero_container()
    values = {
        name: getattr(base, name)
        for name in ValidationSummary.__dataclass_fields__
    }
    empty_quantities = QuantityContainer()
    values.update(
        motion_count=jnp.asarray(motion_count, dtype=jnp.float32),
        per_motion_success=zeros,
        per_motion_frame_coverage=zeros,
        per_motion_return_per_frame=zeros,
        euclidean_distance=empty_quantities,
        dynamic_time_warping=empty_quantities,
        discrete_frechet_distance=empty_quantities,
        left_arm_euclidean_distance=empty_quantities,
        right_arm_euclidean_distance=empty_quantities,
    )
    return TerraValidationSummary(
        **values,
        per_motion_early_termination=zeros,
        per_motion_horizon_timeout=zeros,
        per_motion_episode_length=zeros,
        per_motion_tracking_error_m=zeros,
        per_motion_activation_energy=zeros,
    )


def stochastic_validation_rollout_count(
    motion_count: int,
    *,
    minimum_total_rollouts: int = DEFAULT_MINIMUM_TOTAL_ROLLOUTS,
    rollouts_per_motion: int = DEFAULT_ROLLOUTS_PER_MOTION,
) -> int:
    """Return the exact number of active rollouts for a motion cohort."""
    if motion_count < 1:
        raise ValueError("stochastic validation requires at least one motion")
    if minimum_total_rollouts < 1:
        raise ValueError("minimum_total_rollouts must be positive")
    if rollouts_per_motion < 1:
        raise ValueError("rollouts_per_motion must be positive")
    return max(minimum_total_rollouts, rollouts_per_motion * motion_count)


def validation_lane_count(
    rollout_count: int,
    *,
    max_parallel_rollouts: int = DEFAULT_MAX_PARALLEL_ROLLOUTS,
    allocation_quantum: int = 32,
) -> int:
    """Pad the requested lane pool and cap to the GPU allocation quantum."""
    if rollout_count < 1:
        raise ValueError("rollout_count must be positive")
    if allocation_quantum < 1:
        raise ValueError("allocation_quantum must be positive")
    if max_parallel_rollouts < 1:
        raise ValueError("max_parallel_rollouts must be positive")
    padded_rollouts = ((rollout_count + allocation_quantum - 1) // allocation_quantum) * allocation_quantum
    padded_cap = ((max_parallel_rollouts + allocation_quantum - 1) // allocation_quantum) * allocation_quantum
    return min(padded_rollouts, padded_cap)


def _rollout_settings(validation: Any, motion_count: int) -> tuple[int, int, int]:
    minimum_total = int(validation.get("minimum_total_rollouts", DEFAULT_MINIMUM_TOTAL_ROLLOUTS))
    per_motion = int(validation.get("rollouts_per_motion", DEFAULT_ROLLOUTS_PER_MOTION))
    return (
        stochastic_validation_rollout_count(
            motion_count,
            minimum_total_rollouts=minimum_total,
            rollouts_per_motion=per_motion,
        ),
        minimum_total,
        per_motion,
    )


def _freeze_completed_environments(completed: jax.Array, old_tree: Any, new_tree: Any) -> Any:
    """Keep terminal and padding lanes unchanged for the rest of the scan."""

    def select(old_value, new_value):
        if (
            hasattr(new_value, "shape")
            and new_value.shape
            and completed.ndim > 0
            and new_value.shape[0] == completed.shape[0]
        ):
            mask = completed
            while mask.ndim < new_value.ndim:
                mask = mask[..., None]
            return jnp.where(mask, old_value, new_value)
        return new_value

    return jax.tree.map(select, old_tree, new_tree)


def _lane_layout(
    *,
    motion_count: int,
    trajectory_lengths: tuple[int, ...],
    rollout_count: int,
    local_num_envs: int,
    replica_index: jax.Array,
    rollout_offset: int | jax.Array = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assign active lanes round-robin so every motion gets balanced repeats."""
    global_lane = (
        jnp.asarray(rollout_offset, dtype=jnp.int32)
        + replica_index * local_num_envs
        + jnp.arange(local_num_envs, dtype=jnp.int32)
    )
    active = global_lane < rollout_count
    trajectory_ids = jnp.mod(global_lane, motion_count)
    lengths = jnp.asarray(trajectory_lengths, dtype=jnp.float32)[trajectory_ids]
    return trajectory_ids, lengths, active


def _per_motion_mean(
    values: jax.Array,
    trajectory_ids: jax.Array,
    motion_count: int,
) -> jax.Array:
    """Average repeated rollout values by their assigned motion."""
    totals = jnp.zeros((motion_count,), dtype=jnp.float32).at[trajectory_ids].add(values.astype(jnp.float32))
    counts = jnp.zeros((motion_count,), dtype=jnp.float32).at[trajectory_ids].add(1.0)
    return totals / jnp.maximum(counts, 1.0)


def evaluate_policy_exhaustive(
    train_state: Any,
    rng: jax.Array,
    env: Any,
    config: Any,
    metrics_handler: Any,
    *,
    axis_name: str | None = None,
) -> tuple[TerraValidationSummary, jax.Array]:
    """Run balanced repeated rollouts over every motion on the existing GPU path.

    Rollouts are divided into fixed-size chunks so validation memory is bounded
    independently of cohort size. The final chunk may contain padded lanes; only
    the exact requested rollout count is active. Each active lane starts its
    assigned motion at frame zero and stops at its first terminal state.
    """
    trajectory_handler = metrics_handler._trajectory_handler
    motion_count = int(trajectory_handler.n_trajectories)
    if motion_count < 1:
        raise ValueError("validation.evaluate_all requires at least one trajectory")

    trajectory_lengths = tuple(metrics_handler._trajectory_lengths)
    if len(trajectory_lengths) != motion_count:
        raise ValueError(
            "validation trajectory lengths were not materialized before PPO tracing "
            f"({len(trajectory_lengths)} != {motion_count})"
        )

    rollout_count, _, _ = _rollout_settings(config.validation, motion_count)
    local_num_envs = int(config.validation.num_envs)
    global_num_envs = int(config.validation.get("global_num_envs", local_num_envs))
    if local_num_envs < 1 or global_num_envs < 1:
        raise ValueError("validation environment counts must be positive")
    if axis_name is None and local_num_envs != global_num_envs:
        raise ValueError("single-device validation requires local and global environment counts to match")
    chunk_count = (rollout_count + global_num_envs - 1) // global_num_envs
    progress_enabled = bool(config.validation.get("progress", True))
    progress_debug = bool(config.validation.get("progress_debug", False))
    progress_reporter = _ValidationProgressReporter(
        motion_count=motion_count,
        rollout_count=rollout_count,
        chunk_count=chunk_count,
        debug=progress_debug,
    )

    replica_index = jax.lax.axis_index(axis_name) if axis_name is not None else jnp.asarray(0, dtype=jnp.int32)
    horizon = int(metrics_handler._validation_horizon)
    deterministic = bool(config.validation.get("deterministic", False))

    def evaluate_chunk(carry, chunk_index):
        chunk_rng, progress_token = carry
        chunk_rng, reset_rng, rollout_rng = jax.random.split(chunk_rng, 3)
        trajectory_ids, local_trajectory_lengths, active = _lane_layout(
            motion_count=motion_count,
            trajectory_lengths=trajectory_lengths,
            rollout_count=rollout_count,
            local_num_envs=local_num_envs,
            replica_index=replica_index,
            rollout_offset=chunk_index * global_num_envs,
        )
        reset_keys = jax.random.split(reset_rng, local_num_envs)
        observation, env_state = env.reset_to(reset_keys, trajectory_ids)
        completed = ~active

        def step(carry, _unused):
            current_env_state, current_observation, current_completed, step_rng = carry
            was_completed = current_completed
            step_rng, action_rng = jax.random.split(step_rng)
            policy, _ = train_state.apply_fn(
                {"params": train_state.params, "run_stats": train_state.run_stats},
                current_observation,
            )
            action = policy.mode() if deterministic else policy.sample(seed=action_rng)
            action = jnp.where(was_completed[:, None], 0.0, action)

            (
                next_observation,
                reward,
                absorbing,
                done,
                info,
                next_env_state,
                _transition_state,
                _transition_observation,
            ) = env.step_with_transition(current_env_state, action)
            valid = ~was_completed
            next_completed = was_completed | done
            next_env_state = _freeze_completed_environments(was_completed, current_env_state, next_env_state)
            next_observation = jnp.where(was_completed[:, None], current_observation, next_observation)
            metric_keys = dict.fromkeys((*backend_metrics.VALIDATION_STEP_METRIC_KEYS, *PUBLICATION_STEP_METRIC_KEYS))
            step_metrics = {key: info[key] for key in metric_keys if key in info}
            output = {
                "reward": jnp.where(valid, reward, 0.0),
                "done": done & valid,
                "valid": valid,
                "absorbing": absorbing,
                "info": step_metrics,
            }
            return (
                next_env_state,
                next_observation,
                next_completed,
                step_rng,
            ), output

        _, rollout = jax.lax.scan(
            step,
            (env_state, observation, completed, rollout_rng),
            None,
            length=horizon,
        )
        valid = rollout["valid"]
        done = rollout["done"]
        lengths = jnp.sum(valid, axis=0).astype(jnp.float32)
        returns = jnp.sum(rollout["reward"], axis=0)
        finished = jnp.any(done, axis=0)
        early = active & jnp.any(done & rollout["absorbing"], axis=0)
        success = active & finished & ~early
        coverage = jnp.minimum(
            lengths / jnp.maximum(local_trajectory_lengths, 1.0),
            1.0,
        )
        return_per_frame = returns / jnp.maximum(lengths, 1.0)
        active_float = active.astype(jnp.float32)

        def per_motion_sum(values):
            return (
                jnp.zeros((motion_count,), dtype=jnp.float32)
                .at[trajectory_ids]
                .add(values.astype(jnp.float32) * active_float)
            )

        valid_step_count = jnp.sum(valid.astype(jnp.float32))
        step_numerators = {}
        step_denominators = {}
        for key, values in rollout["info"].items():
            mask = valid
            while mask.ndim < values.ndim:
                mask = mask[..., None]
            step_numerators[key] = jnp.sum(jnp.where(mask, values, 0.0))
            step_denominators[key] = valid_step_count * max(1, math.prod(values.shape[2:]))

        per_motion_step_metric_sum = {}
        for key in PUBLICATION_STEP_METRIC_KEYS:
            if key not in rollout["info"]:
                continue
            values = rollout["info"][key]
            mask = valid
            while mask.ndim < values.ndim:
                mask = mask[..., None]
            reduction_axes = (0, *range(2, values.ndim))
            per_rollout_sum = jnp.sum(jnp.where(mask, values, 0.0), axis=reduction_axes)
            per_rollout_denominator = lengths * max(1, math.prod(values.shape[2:]))
            per_rollout_mean = per_rollout_sum / jnp.maximum(per_rollout_denominator, 1.0)
            per_motion_step_metric_sum[key] = per_motion_sum(per_rollout_mean)

        chunk_metrics = {
            "episode_count": jnp.sum(active_float),
            "return_sum": jnp.sum(returns * active_float),
            "length_sum": jnp.sum(lengths * active_float),
            "max_length": jnp.max(lengths),
            "early_sum": jnp.sum(early.astype(jnp.float32)),
            "timeout_sum": jnp.sum((active & ~finished).astype(jnp.float32)),
            "success_sum": jnp.sum(success.astype(jnp.float32)),
            "coverage_sum": jnp.sum(coverage * active_float),
            "return_per_frame_sum": jnp.sum(return_per_frame * active_float),
            "per_motion_count": per_motion_sum(active_float),
            "per_motion_success_sum": per_motion_sum(success),
            "per_motion_early_sum": per_motion_sum(early),
            "per_motion_timeout_sum": per_motion_sum(active & ~finished),
            "per_motion_length_sum": per_motion_sum(lengths),
            "per_motion_coverage_sum": per_motion_sum(coverage),
            "per_motion_return_per_frame_sum": per_motion_sum(return_per_frame),
            "per_motion_step_metric_sum": per_motion_step_metric_sum,
            "step_numerators": step_numerators,
            "step_denominators": step_denominators,
        }
        if progress_enabled:
            global_active = chunk_metrics["episode_count"]
            global_length_sum = chunk_metrics["length_sum"]
            global_max_length = chunk_metrics["max_length"]
            if axis_name is not None:
                # Besides making progress global and truthful, this rendezvous
                # prevents one pmap replica from running arbitrarily far ahead
                # inside a long Warp validation executable.
                global_active = jax.lax.psum(global_active, axis_name=axis_name)
                global_length_sum = jax.lax.psum(global_length_sum, axis_name=axis_name)
                global_max_length = jax.lax.pmax(global_max_length, axis_name=axis_name)
            completed_rollouts = jnp.minimum(
                (chunk_index + 1) * global_num_envs,
                rollout_count,
            )
            progress_token = io_callback(
                progress_reporter,
                jax.ShapeDtypeStruct((), jnp.int32),
                replica_index,
                chunk_index,
                completed_rollouts,
                global_active,
                # This value depends on the completed horizon scan, ensuring
                # the callback cannot report a chunk before its rollout ends.
                global_length_sum,
                global_max_length,
                progress_token,
                ordered=False,
            )
        return (chunk_rng, progress_token), chunk_metrics

    progress_token = jnp.asarray(0, dtype=jnp.int32)
    if progress_enabled:
        progress_token = io_callback(
            progress_reporter.start,
            jax.ShapeDtypeStruct((), jnp.int32),
            replica_index,
            progress_token,
            ordered=False,
        )
    (next_rng, _progress_token), chunks = jax.lax.scan(
        evaluate_chunk,
        (rng, progress_token),
        jnp.arange(chunk_count, dtype=jnp.int32),
    )

    def global_sum(values):
        result = jnp.sum(values, axis=0)
        return jax.lax.psum(result, axis_name=axis_name) if axis_name is not None else result

    max_timestep = jnp.max(chunks["max_length"]).astype(jnp.int32)
    if axis_name is not None:
        max_timestep = jax.lax.pmax(max_timestep, axis_name=axis_name)
    episode_count = jnp.maximum(global_sum(chunks["episode_count"]), 1.0)
    return_sum = global_sum(chunks["return_sum"])
    length_sum = global_sum(chunks["length_sum"])
    early_sum = global_sum(chunks["early_sum"])
    timeout_sum = global_sum(chunks["timeout_sum"])
    success_sum = global_sum(chunks["success_sum"])
    coverage_sum = global_sum(chunks["coverage_sum"])
    return_per_frame_sum = global_sum(chunks["return_per_frame_sum"])
    per_motion_count = jnp.maximum(global_sum(chunks["per_motion_count"]), 1.0)
    per_motion_success = global_sum(chunks["per_motion_success_sum"]) / per_motion_count
    per_motion_early = global_sum(chunks["per_motion_early_sum"]) / per_motion_count
    per_motion_timeout = global_sum(chunks["per_motion_timeout_sum"]) / per_motion_count
    per_motion_length = global_sum(chunks["per_motion_length_sum"]) / per_motion_count
    per_motion_coverage = global_sum(chunks["per_motion_coverage_sum"]) / per_motion_count
    per_motion_return_per_frame = global_sum(chunks["per_motion_return_per_frame_sum"]) / per_motion_count

    step_means = {}
    for key, values in chunks["step_numerators"].items():
        numerator = global_sum(values)
        denominator = jnp.maximum(global_sum(chunks["step_denominators"][key]), 1.0)
        step_means[key] = numerator / denominator

    per_motion_step_means = {
        key: global_sum(values) / per_motion_count for key, values in chunks["per_motion_step_metric_sum"].items()
    }
    empty_per_motion = jnp.zeros((motion_count,), dtype=jnp.float32)

    summary_values = {
        key: step_means.get(key, jnp.asarray(0.0, dtype=jnp.float32))
        for key in backend_metrics.VALIDATION_STEP_METRIC_KEYS
        if key in ValidationSummary.__dataclass_fields__
    }
    quartile_size = max(1, (motion_count + 3) // 4)
    empty_quantities = QuantityContainer()
    return (
        TerraValidationSummary(
            episode_count=episode_count,
            mean_episode_return=return_sum / episode_count,
            mean_episode_length=length_sum / episode_count,
            max_timestep=max_timestep,
            early_termination_count=early_sum,
            early_termination_rate=early_sum / episode_count,
            horizon_timeout_count=timeout_sum,
            horizon_timeout_rate=timeout_sum / episode_count,
            motion_success_rate=success_sum / episode_count,
            mean_frame_coverage=coverage_sum / episode_count,
            worst_quartile_frame_coverage=jnp.mean(jnp.sort(per_motion_coverage)[:quartile_size]),
            mean_return_per_frame=return_per_frame_sum / episode_count,
            motion_count=jnp.asarray(motion_count, dtype=jnp.float32),
            per_motion_success=per_motion_success,
            per_motion_early_termination=per_motion_early,
            per_motion_horizon_timeout=per_motion_timeout,
            per_motion_episode_length=per_motion_length,
            per_motion_frame_coverage=per_motion_coverage,
            per_motion_return_per_frame=per_motion_return_per_frame,
            per_motion_tracking_error_m=per_motion_step_means.get("err_site_abs", empty_per_motion),
            per_motion_activation_energy=per_motion_step_means.get("activation_energy_raw", empty_per_motion),
            euclidean_distance=empty_quantities,
            dynamic_time_warping=empty_quantities,
            discrete_frechet_distance=empty_quantities,
            left_arm_euclidean_distance=empty_quantities,
            right_arm_euclidean_distance=empty_quantities,
            **summary_values,
        ),
        next_rng,
    )


__all__ = [
    "DEFAULT_MAX_PARALLEL_ROLLOUTS",
    "DEFAULT_MINIMUM_TOTAL_ROLLOUTS",
    "DEFAULT_ROLLOUTS_PER_MOTION",
    "PUBLICATION_STEP_METRIC_KEYS",
    "TerraValidationSummary",
    "evaluate_policy_exhaustive",
    "exhaustive_zero_summary",
    "stochastic_validation_rollout_count",
    "validation_lane_count",
]
