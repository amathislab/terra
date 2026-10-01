"""Narrow integration shims for the commit-pinned MuscleMimic backend."""

from __future__ import annotations

from terra.rl.metrics import TerraMetricsHandler
from terra.rl.trajectory import install_trajectory_stability

_TERRA_REWARD_METRICS = (
    ("activation_energy_raw", "reward/activation_energy_raw"),
    ("activation_floor_violation_raw", "reward/activation_floor_violation_raw"),
    ("activation_below_floor_fraction_raw", "reward/activation_below_floor_fraction_raw"),
    ("penalty_activation_floor", "reward/penalty_activation_floor"),
    ("penalty_activity_regularization", "reward/penalty_activity_regularization"),
    ("reward_core_upper_body", "reward/core_upper_body"),
    ("err_core_upper_body_relative", "errors/core_upper_body_relative"),
    ("reward_terminal_quality_bonus", "reward/terminal_quality_bonus"),
    ("reward_emg_correlation", "reward/emg_correlation"),
    ("emg_correlation_raw", "emg/correlation"),
    ("emg_correlation_defined", "emg/correlation_defined_fraction"),
    ("emg_supervision_active", "emg/supervision_fraction"),
    ("emg_target_channel_count", "emg/target_channel_count"),
    ("emg_prediction_std_raw", "emg/prediction_std"),
    ("emg_tracking_gate", "emg/tracking_gate"),
)


def _append_once(values: tuple, value: object) -> tuple:
    return values if value in values else (*values, value)


def _install_ppo_validation() -> None:
    """Install TERRA's repeated stochastic validator behind the upstream hook."""

    from musclemimic.algorithms.ppo import evaluation as ppo_evaluation
    from terra.rl.ppo_validation import evaluate_policy_exhaustive, exhaustive_zero_summary

    ppo_evaluation.evaluate_policy_exhaustive = evaluate_policy_exhaustive
    ppo_evaluation.exhaustive_zero_summary = exhaustive_zero_summary


def install_backend_integrations(algorithm: str = "PPOJax") -> None:
    """Install TERRA-owned metrics and logging into the pinned backend.

    MuscleMimic deliberately exposes experiment hooks for logging and video
    behavior, but its metrics-handler and reward-info registries are module
    level in the pinned revision.  Keep those compatibility assignments in one
    explicit, idempotent location until the backend grows corresponding hooks.
    """

    from musclemimic.runner import engine
    from musclemimic.utils import metrics as backend_metrics

    engine.MetricsHandler = TerraMetricsHandler
    install_trajectory_stability()
    for info_key, _ in _TERRA_REWARD_METRICS:
        backend_metrics.VALIDATION_STEP_METRIC_KEYS = _append_once(
            backend_metrics.VALIDATION_STEP_METRIC_KEYS,
            info_key,
        )

    algorithm_key = str(algorithm).strip().lower()
    if algorithm_key in {"ppo", "ppojax"}:
        _install_ppo_validation()
        return
