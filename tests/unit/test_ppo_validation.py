"""Unit tests for repeated stochastic PPO validation allocation."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from omegaconf import OmegaConf

from musclemimic.utils.metrics import ValidationSummary
from terra.rl.ppo_validation import (
    _format_progress_bar,
    _lane_layout,
    _per_motion_mean,
    evaluate_policy_exhaustive,
    exhaustive_zero_summary,
    stochastic_validation_rollout_count,
    validation_lane_count,
)


def test_progress_bar_is_bounded_and_fixed_width():
    assert _format_progress_bar(0, 100, width=10) == "[..........]"
    assert _format_progress_bar(50, 100, width=10) == "[#####.....]"
    assert _format_progress_bar(100, 100, width=10) == "[##########]"
    assert _format_progress_bar(101, 100, width=10) == "[##########]"


@pytest.mark.parametrize(
    ("motion_count", "rollout_count", "lane_count"),
    [
        (1, 100, 128),
        (30, 100, 128),
        (50, 150, 160),
        (150, 450, 480),
        (300, 900, 928),
        (1_000, 3_000, 1_024),
        (6_856, 20_568, 1_024),
    ],
)
def test_stochastic_validation_sizes_exact_rollouts_and_padded_gpu_lanes(
    motion_count,
    rollout_count,
    lane_count,
):
    actual_rollouts = stochastic_validation_rollout_count(motion_count)

    assert actual_rollouts == rollout_count
    assert validation_lane_count(actual_rollouts) == lane_count


def test_lane_layout_balances_nondivisible_minimum_across_motions():
    motion_count = 30
    trajectory_ids, lengths, active = _lane_layout(
        motion_count=motion_count,
        trajectory_lengths=tuple(range(1, motion_count + 1)),
        rollout_count=100,
        local_num_envs=128,
        replica_index=jnp.asarray(0),
    )
    active_ids = np.asarray(trajectory_ids)[np.asarray(active)]

    assert int(np.sum(active)) == 100
    assert np.array_equal(np.bincount(active_ids, minlength=motion_count)[:10], np.full(10, 4))
    assert np.array_equal(np.bincount(active_ids, minlength=motion_count)[10:], np.full(20, 3))
    assert np.array_equal(np.asarray(lengths)[:motion_count], np.arange(1, motion_count + 1))


def test_per_motion_metrics_are_means_over_repeated_rollouts():
    values = jnp.asarray([0.0, 1.0, 1.0, 1.0, 0.0, 0.5])
    trajectory_ids = jnp.asarray([0, 1, 2, 0, 1, 2])

    result = _per_motion_mean(values, trajectory_ids, 3)

    assert np.asarray(result) == pytest.approx([0.5, 0.5, 0.75])


class _StochasticPolicy:
    def __init__(self, observation):
        self._observation = observation

    def mode(self):
        return jnp.zeros_like(self._observation)

    def sample(self, *, seed):
        return random.uniform(seed, self._observation.shape)


class _TwoMotionEnvironment:
    def reset_to(self, _keys, trajectory_ids):
        steps = jnp.zeros_like(trajectory_ids)
        return steps[:, None].astype(jnp.float32), (trajectory_ids, steps)

    def step_with_transition(self, state, action):
        trajectory_ids, steps = state
        next_steps = steps + 1
        target_lengths = jnp.where(trajectory_ids == 0, 2, 3)
        done = next_steps >= target_lengths
        absorbing = done & (trajectory_ids == 0)
        observation = next_steps[:, None].astype(jnp.float32)
        info = {
            "err_site_abs": jnp.where(trajectory_ids == 0, 0.1, 0.3),
            "activation_energy_raw": jnp.where(trajectory_ids == 0, 0.2, 0.4),
        }
        return (
            observation,
            action[:, 0],
            absorbing,
            done,
            info,
            (trajectory_ids, next_steps),
            None,
            None,
        )


def test_chunked_exhaustive_validator_samples_policy_and_aggregates_repeats_per_motion():
    config = OmegaConf.create(
        {
            "validation": {
                "num_envs": 32,
                "global_num_envs": 32,
                "deterministic": False,
                "minimum_total_rollouts": 100,
                "rollouts_per_motion": 3,
            }
        }
    )
    train_state = SimpleNamespace(
        params={},
        run_stats={},
        apply_fn=lambda _variables, observation: (_StochasticPolicy(observation), None),
    )
    metrics_handler = SimpleNamespace(
        _trajectory_handler=SimpleNamespace(n_trajectories=2),
        _trajectory_lengths=(2, 3),
        _validation_horizon=3,
    )

    summary, _ = evaluate_policy_exhaustive(
        train_state,
        random.key(7),
        _TwoMotionEnvironment(),
        config,
        metrics_handler,
    )

    assert float(summary.episode_count) == 100
    assert float(summary.mean_episode_return) > 0.0
    assert float(summary.mean_episode_length) == pytest.approx(2.5)
    assert float(summary.early_termination_count) == 50
    assert float(summary.motion_success_rate) == pytest.approx(0.5)
    assert np.asarray(summary.per_motion_success) == pytest.approx([0.0, 1.0])
    assert np.asarray(summary.per_motion_early_termination) == pytest.approx([1.0, 0.0])
    assert np.asarray(summary.per_motion_horizon_timeout) == pytest.approx([0.0, 0.0])
    assert np.asarray(summary.per_motion_episode_length) == pytest.approx([2.0, 3.0])
    assert np.asarray(summary.per_motion_frame_coverage) == pytest.approx([1.0, 1.0])
    assert np.asarray(summary.per_motion_tracking_error_m) == pytest.approx([0.1, 0.3])
    assert np.asarray(summary.per_motion_activation_energy) == pytest.approx([0.2, 0.4])


def test_chunked_exhaustive_validator_reports_completed_chunks(caplog):
    config = OmegaConf.create(
        {
            "validation": {
                "num_envs": 32,
                "global_num_envs": 32,
                "deterministic": False,
                "minimum_total_rollouts": 100,
                "rollouts_per_motion": 3,
                "progress": True,
                "progress_debug": True,
            }
        }
    )
    train_state = SimpleNamespace(
        params={},
        run_stats={},
        apply_fn=lambda _variables, observation: (_StochasticPolicy(observation), None),
    )
    metrics_handler = SimpleNamespace(
        _trajectory_handler=SimpleNamespace(n_trajectories=2),
        _trajectory_lengths=(2, 3),
        _validation_horizon=3,
    )

    with caplog.at_level("INFO", logger="terra.rl.ppo_validation"):
        summary, _ = evaluate_policy_exhaustive(
            train_state,
            random.key(7),
            _TwoMotionEnvironment(),
            config,
            metrics_handler,
        )
        jax.block_until_ready(summary)
        jax.effects_barrier()

    progress_messages = [
        record.message for record in caplog.records if record.message.startswith("PPO validation progress")
    ]
    debug_messages = [
        record.message for record in caplog.records if record.message.startswith("PPO validation debug")
    ]
    assert len(progress_messages) == 5
    assert "0/2 motions (0.0%; chunk 0/4)" in progress_messages[0]
    assert "2/2 motions (100.0%; chunk 4/4)" in progress_messages[-1]
    assert len(debug_messages) == 4
    assert "rollouts=100/100" in debug_messages[-1]


def test_exhaustive_zero_summary_matches_evaluation_branch_pytree():
    config = OmegaConf.create(
        {
            "validation": {
                "num_envs": 32,
                "global_num_envs": 32,
                "deterministic": False,
                "minimum_total_rollouts": 100,
                "rollouts_per_motion": 3,
            }
        }
    )
    train_state = SimpleNamespace(
        params={},
        run_stats={},
        apply_fn=lambda _variables, observation: (_StochasticPolicy(observation), None),
    )
    metrics_handler = SimpleNamespace(
        _trajectory_handler=SimpleNamespace(n_trajectories=2),
        _trajectory_lengths=(2, 3),
        _validation_horizon=3,
        get_zero_container=lambda: ValidationSummary(),
    )

    evaluated, _ = evaluate_policy_exhaustive(
        train_state,
        random.key(7),
        _TwoMotionEnvironment(),
        config,
        metrics_handler,
    )
    skipped = exhaustive_zero_summary(metrics_handler)

    assert jax.tree.structure(skipped) == jax.tree.structure(evaluated)
    assert np.asarray(skipped.per_motion_success).shape == (2,)
    assert np.asarray(skipped.per_motion_early_termination).shape == (2,)


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires four JAX devices")
def test_exhaustive_validator_gathers_global_rollouts_across_four_devices():
    config = OmegaConf.create(
        {
            "validation": {
                "num_envs": 32,
                "global_num_envs": 128,
                "deterministic": False,
                "minimum_total_rollouts": 100,
                "rollouts_per_motion": 3,
            }
        }
    )
    train_state = SimpleNamespace(
        params={},
        run_stats={},
        apply_fn=lambda _variables, observation: (_StochasticPolicy(observation), None),
    )
    metrics_handler = SimpleNamespace(
        _trajectory_handler=SimpleNamespace(n_trajectories=2),
        _trajectory_lengths=(2, 3),
        _validation_horizon=3,
    )

    def evaluate(rng):
        return evaluate_policy_exhaustive(
            train_state,
            rng,
            _TwoMotionEnvironment(),
            config,
            metrics_handler,
            axis_name="devices",
        )[0]

    summaries = jax.pmap(evaluate, axis_name="devices")(random.split(random.key(11), 4))

    assert np.asarray(summaries.episode_count) == pytest.approx(np.full(4, 100.0))
    assert np.asarray(summaries.motion_success_rate) == pytest.approx(np.full(4, 0.5))
    assert np.asarray(summaries.per_motion_success) == pytest.approx(np.tile([0.0, 1.0], (4, 1)))
