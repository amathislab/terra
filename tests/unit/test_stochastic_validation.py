import numpy as np
from omegaconf import OmegaConf

from terra.rl.stochastic_validation import (
    _aggregate,
    _build_parser,
    _configure_termination_threshold,
    _configure_terrain_collision_margin,
)


def _run(success, coverage, returns=None, *, tracking=None, activation=None):
    if returns is None:
        returns = coverage
    if tracking is None:
        tracking = np.asarray(coverage, dtype=float) / 10.0
    if activation is None:
        activation = np.asarray(coverage, dtype=float) / 20.0
    success = np.asarray(success, dtype=float)
    return {
        "success": success,
        "early_termination": 1.0 - success,
        "horizon_timeout": np.zeros_like(success),
        "coverage": np.asarray(coverage, dtype=float),
        "return_per_frame": np.asarray(returns, dtype=float),
        "tracking_error_m": np.asarray(tracking, dtype=float),
        "activation_energy": np.asarray(activation, dtype=float),
        "episode_length": 100.0 * np.asarray(coverage, dtype=float),
    }


def test_aggregate_reports_five_stochastic_rollouts_and_equal_motion_means():
    stochastic = [
        _run([1, 0, 1, 0], [1.0, 0.3, 1.0, 0.7]),
        _run([0, 0, 1, 1], [0.2, 0.4, 1.0, 0.9]),
        _run([1, 0, 1, 1], [0.9, 0.2, 1.0, 0.8]),
        _run([1, 0, 1, 1], [1.0, 0.2, 1.0, 0.8]),
        _run([0, 0, 1, 1], [0.8, 0.4, 1.0, 0.8]),
    ]

    report = _aggregate(["a", "b", "c", "d"], stochastic)

    summary = report["summary"]
    assert summary["stochastic_rollouts_per_motion"] == 5
    assert np.isclose(summary["stochastic"]["completion_rate"], 12 / 20)
    assert np.isclose(summary["stochastic"]["early_termination_rate"], 8 / 20)
    assert summary["stochastic"]["horizon_timeout_rate"] == 0.0
    assert np.isclose(summary["stochastic"]["tracking_error_m"], 0.072)
    assert len(report["motions"][0]["stochastic"]["rollouts"]) == 5
    assert report["motions"][0]["stochastic"]["summary"]["success"] == 3 / 5
    assert report["motions"][0]["stochastic"]["rollouts"][0]["success"] is True


def test_parser_defaults_to_five_mjx_warp_rollouts_and_accepts_jax_override():
    defaults = _build_parser().parse_args(["--checkpoint", "/checkpoint", "--output", "/report.json"])
    args = _build_parser().parse_args(
        ["--checkpoint", "/checkpoint", "--output", "/report.json", "--mjx-backend", "jax"]
    )

    assert defaults.repeats == 5
    assert defaults.mjx_backend == "warp"
    assert defaults.termination_threshold is None
    assert args.mjx_backend == "jax"


def test_terrain_collision_margin_is_written_to_task_factory_config():
    config = OmegaConf.create({"experiment": {"task_factory": {"params": {}}}})

    assert _configure_terrain_collision_margin(config, 0.002) == 0.002
    assert config.experiment.task_factory.params.terrain_collision_margin == 0.002


def test_termination_threshold_overrides_both_validation_bounds():
    config = OmegaConf.create(
        {
            "experiment": {
                "validation": {
                    "terminal_state_params": {
                        "mean_site_deviation_threshold": 0.15,
                        "core_upper_body_mean_site_deviation_threshold": 0.15,
                    }
                }
            }
        }
    )

    thresholds = _configure_termination_threshold(config, 0.25)

    assert thresholds == {"global_mpjpe": 0.25, "core_upper_body_mpjpe": 0.25}
    assert config.experiment.validation.terminal_state_params.mean_site_deviation_threshold == 0.25
    assert config.experiment.validation.terminal_state_params.core_upper_body_mean_site_deviation_threshold == 0.25
