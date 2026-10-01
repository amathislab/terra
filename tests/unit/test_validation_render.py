import argparse

import numpy as np
import pytest

from terra.rl.validation_render import (
    _checkpoint_observation_dimension,
    _checkpoint_timestep,
    _parse_motion,
)


def test_motion_argument_requires_name_and_path():
    assert _parse_motion("stairs_up=Gait120/S100/StairAscent/Trial01/AllSteps_stageii") == {
        "name": "stairs_up",
        "path": "Gait120/S100/StairAscent/Trial01/AllSteps_stageii",
    }
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=DATASET_PATH"):
        _parse_motion("missing-separator")


def test_checkpoint_timestep_supports_metadata_objects_and_mappings():
    class Metadata:
        global_timestep = 123

    assert _checkpoint_timestep(Metadata()) == 123
    assert _checkpoint_timestep({"global_timestep": 456}) == 456


def test_checkpoint_observation_dimension_reads_ppo_running_statistics():
    state = {
        "run_stats": {
            "RunningMeanStd_0": {
                "count": np.asarray(10),
                "mean": np.zeros(1413),
                "var": np.ones(1413),
            }
        }
    }

    assert _checkpoint_observation_dimension(state) == 1413
