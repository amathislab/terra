"""Experiment hooks for TERRA training runs."""

from __future__ import annotations

import logging

from omegaconf import OmegaConf

from musclemimic.runner.logging import ExperimentHooks
from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder

logger = logging.getLogger(__name__)


class TerraValidationVideoRecorder(ValidationVideoRecorder):
    """Record validation with TERRA's terrain-safe reference renderer."""

    def _build_env_params(self, agent_conf, tag: str) -> dict:
        parameters = super()._build_env_params(agent_conf, tag)
        validation = agent_conf.config.experiment.get("validation", {})
        validation_environment = validation.get("env_params", {})
        if validation_environment:
            parameters.update(OmegaConf.to_container(validation_environment, resolve=True))
        # The generic recorder has already replaced the training goal with its
        # own visual type, so select TERRA's renderer from the source config.
        goal_type = agent_conf.config.experiment.env_params.get("goal_type", "TerraGoal")
        parameters["goal_type"] = {
            "TerraFullBodyTrackingGoal": "TerraFullBodyTrackingGoalVisual",
            "TerraFullBodyTrackingGoalVisual": "TerraFullBodyTrackingGoalVisual",
        }.get(goal_type, "TerraGoalVisual")
        return parameters

    def _build_task_params(self, agent_conf, motion_path: str | None = None) -> dict:
        parameters = super()._build_task_params(agent_conf, motion_path)
        if motion_path is not None:
            # A checkpoint cache key identifies its complete train/validation
            # cohort. Reusing it after pinning one motion can silently load the
            # first cached trajectory for every named render.
            parameters["trajectory_cache_root"] = ""
            parameters["trajectory_cache_key"] = ""
        return parameters


class TerraHooks(ExperimentHooks):
    """Use the local validation recorder with the generic training engine."""

    def build_video_recorder(self, result_dir: str, config) -> TerraValidationVideoRecorder | None:
        validation = config.experiment.get("validation", {})
        if not validation.get("active", False) or not validation.get("video_active", True):
            return None
        return TerraValidationVideoRecorder(
            video_dir=result_dir,
            frequency=int(validation.get("video_frequency", 10)),
            length=int(validation.get("video_length", 250)),
            deterministic=bool(validation.get("deterministic", True)),
            named_motions=validation.get("video_motions", ()),
        )

    def enrich_log(self, log_dict: dict, metrics_dict: dict, _env) -> None:
        if not metrics_dict.get("has_validation_update", False):
            return

        if float(metrics_dict.get("val_motion_count", 0.0)) > 0.0:
            success_rate = float(metrics_dict.get("val_motion_success_rate", 0.0))
        else:
            early_termination_rate = float(metrics_dict.get("val_early_termination_rate", 0.0))
            success_rate = 1.0 - early_termination_rate
        log_dict["Validation/success_rate"] = success_rate


__all__ = ["TerraHooks", "TerraValidationVideoRecorder"]
