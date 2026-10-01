from types import SimpleNamespace

import mujoco
from omegaconf import OmegaConf

from terra.rl.model_options import materialize_mjx_model_options


def test_materialize_mjx_model_options_uses_effective_environment_defaults():
    config = SimpleNamespace(
        experiment=SimpleNamespace(env_params=OmegaConf.create({"env_name": "MjxMyoFullBody"}))
    )

    options = materialize_mjx_model_options(config)

    assert options == {
        "iterations": 4,
        "ls_iterations": 8,
        "disableflags": int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
    }
    assert OmegaConf.to_container(config.experiment.env_params.model_option_conf) == options


def test_materialize_mjx_model_options_preserves_explicit_values():
    config = SimpleNamespace(
        experiment=SimpleNamespace(
            env_params=OmegaConf.create(
                {
                    "model_option_conf": {
                        "iterations": 12,
                        "ls_iterations": 3,
                        "disableflags": 0,
                    }
                }
            )
        )
    )

    assert materialize_mjx_model_options(config) == {
        "iterations": 12,
        "ls_iterations": 3,
        "disableflags": 0,
    }
