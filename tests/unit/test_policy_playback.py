"""Policy playback selects the dataset without changing policy inputs."""

from pathlib import Path
from runpy import run_path

import pytest
from omegaconf import OmegaConf

script = run_path(str(Path(__file__).resolve().parents[2] / "scripts/terra/play_policy.py"))


def test_motion_selection_uses_the_requested_split_and_order():
    record = {
        "motions": [
            {"motion": "upstairs07_poses", "split": "train"},
            {"motion": "KIT/3/downstairs", "split": "evaluation"},
            {"motion": "KIT/4/walk", "split": "test"},
        ]
    }
    assert script["selected_motions"](record, "train", None) == ["upstairs07_poses"]
    assert script["selected_motions"](record, "evaluation", None) == ["KIT/3/downstairs"]
    with pytest.raises(ValueError, match="absent from the train split"):
        script["selected_motions"](record, "train", ["KIT/4/walk"])


@pytest.mark.parametrize("video,show_reference", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tracking_threshold", [0.15, 0.25])
def test_playback_preserves_checkpoint_inputs_and_uses_the_new_dataset(
    tmp_path, video, show_reference, tracking_threshold
):
    config = OmegaConf.create(
        {
            "experiment": {
                "algorithm": "PPOJax",
                "actor_hidden_layers": [256, 128],
                "len_obs_history": 5,
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "headless": True,
                    "timestep": 0.002,
                    "n_substeps": 5,
                    "num_envs": 1024,
                    "mjx_backend": "warp",
                    "nconmax": 192,
                    "njmax": 768,
                    "enable_heightmap_observations": True,
                    "use_egocentric_root_observations": True,
                    "goal_type": "TerraGoal",
                    "goal_params": {"future_reference_stride": 10},
                },
                "validation": {
                    "init_state_type": "TrajInitialStateHandler",
                    "init_state_params": {},
                    "terminal_state_type": "TerraGlobalMPJPETerminalStateHandler",
                    "terminal_state_params": {
                        "mean_site_deviation_threshold": 0.15,
                        "core_upper_body_mean_site_deviation_threshold": 0.15,
                        "root_orientation_threshold": 1.0,
                    },
                },
                "task_factory": {
                    "params": {
                        "trajectory_cache_root": "/old/cache",
                        "trajectory_cache_key": "old",
                        "amass_dataset_conf": {"rel_dataset_path": ["old/motion"], "cache_root": "/old/dataset"},
                    }
                },
            }
        }
    )
    original = OmegaConf.to_container(config)
    record = {"destination_cache": str(tmp_path / "new-cache"), "terrain_mode": "mixed"}
    result = script["playback_config"](
        config,
        record,
        "upstairs07_poses",
        tmp_path if video else None,
        tracking_threshold=tracking_threshold,
        show_reference=show_reference,
    )
    assert OmegaConf.to_container(config) == original
    assert result.experiment.actor_hidden_layers == config.experiment.actor_hidden_layers
    assert result.experiment.len_obs_history == config.experiment.len_obs_history
    params = result.experiment.env_params
    assert params.enable_heightmap_observations
    assert params.use_egocentric_root_observations
    assert params.goal_params.future_reference_stride == 10
    assert params.terminal_state_params.mean_site_deviation_threshold == tracking_threshold
    assert params.terminal_state_params.core_upper_body_mean_site_deviation_threshold == tracking_threshold
    assert params.terminal_state_params.root_orientation_threshold == 1.0
    assert params.env_name == "MyoFullBody"
    assert params.headless == video
    assert params.th_params.fixed_start_conf == [0, 0]
    assert "mjx_backend" not in params
    factory = result.experiment.task_factory.params
    assert factory.trajectory_cache_key is None
    assert factory.amass_dataset_conf.rel_dataset_path == ["upstairs07_poses"]
    assert factory.amass_dataset_conf.cache_root == record["destination_cache"]
    assert factory.amass_dataset_conf.load_paired_terrain
    assert not factory.amass_dataset_conf.require_nonflat_terrain
    if video:
        assert params.recorder_params.fps == 100
        assert params.goal_type == ("TerraGoalVisual" if show_reference else "TerraGoal")
        assert params.viewer_size == [1280, 720]
        assert not params.show_debug_overlay
