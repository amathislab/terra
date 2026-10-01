"""Keep the effective MJX solver options when rebuilding an environment in MuJoCo."""

from __future__ import annotations

from typing import Any

import mujoco
from omegaconf import OmegaConf, open_dict


def materialize_mjx_model_options(config: Any) -> dict[str, int]:
    """Write ``MjxMyoFullBody``'s implicit defaults into the experiment config.

    ``MjxMyoFullBody`` supplies these values inside its constructor.  Native
    validation replaces that class with ``MyoFullBody``, so the values must be
    explicit in the config to keep both compiled models equivalent.
    """
    configured = config.experiment.env_params.get("model_option_conf", None)
    if configured is None:
        options = {
            "iterations": 4,
            "ls_iterations": 8,
            "disableflags": int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
        }
    elif OmegaConf.is_config(configured):
        options = dict(OmegaConf.to_container(configured, resolve=True))
    else:
        options = dict(configured)

    with open_dict(config.experiment.env_params):
        config.experiment.env_params.model_option_conf = options
    return options


__all__ = ["materialize_mjx_model_options"]
