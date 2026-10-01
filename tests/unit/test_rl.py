"""Regression tests for TERRA-owned reinforcement-learning behavior."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from musclemimic.core.reward.trajectory_based import MimicReward
from terra.rl.backend import install_backend_integrations
from terra.rl.environment import MjxMyoFullBody
from terra.rl.hooks import TerraHooks
from terra.rl.metrics import TerraMetricsHandler
from terra.rl.rewards import TerraReward
from terra.rl.termination import TerraGlobalMPJPETerminalStateHandler
from terra.rl.terrain import PairedBoxTerrain


@dataclass(frozen=True)
class _Carry:
    qvel_w_sum: np.ndarray
    root_vel_w_sum: np.ndarray
    traj_state: object = None
    reward_state: object = None

    def replace(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class _TrajectoryResetCarry:
    key: object
    selected_traj_idx: int
    sampling_weights: object = None
    traj_state: object = None

    def replace(self, **changes):
        return replace(self, **changes)


def test_mjx_jax_preserves_terra_body_terrain_contacts():
    spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>
            <body name="foot">
              <geom name="foot_collision" type="sphere" size="0.1" contype="1" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    env = object.__new__(MjxMyoFullBody)
    env.mjx_backend = "jax"

    result = env._modify_spec_for_mjx(spec)
    geoms = {geom.name: geom for geom in result.geoms}

    assert result is spec
    assert geoms["floor"].contype == 1
    assert geoms["floor"].conaffinity == 1
    assert geoms["foot_collision"].contype == 1
    assert geoms["foot_collision"].conaffinity == 0


def _reward_without_initialization(
    *,
    dynamic: bool = False,
    activation_floor: float = 0.0,
    activation_floor_coeff: float = 0.0,
) -> TerraReward:
    reward = object.__new__(TerraReward)
    reward._qvel_w_sum = 0.25
    reward._root_vel_w_sum = 0.5
    reward._use_dynamic_velocity_reward_weights = dynamic
    reward._activation_floor = activation_floor
    reward._activation_floor_coeff = activation_floor_coeff
    reward._emg_correlation_reward_weight = 0.0
    reward._emg_correlation_min_samples = 3
    reward._emg_prediction_std_threshold = 1e-6
    reward._emg_target_std_threshold = 1e-6
    reward._emg_max_channels = 12
    reward._core_upper_body_position_reward_weight = 1.0
    reward._terminal_quality_bonus_weight = 25.0
    reward._core_upper_body_tracking = lambda *_args: (np.asarray(0.8), np.asarray(0.1))
    return reward


class _RewardTrajectory:
    def __init__(self, at_end=False):
        self.at_end = at_end

    def reached_trajectory_end(self, _traj_state, _backend):
        return np.asarray(self.at_end)


def _reward_env(*, at_end=False):
    return SimpleNamespace(th=_RewardTrajectory(at_end=at_end))


def test_terra_reward_uses_configured_velocity_weights_and_logs_raw_activation(monkeypatch):
    captured = {}

    def fake_reward(self, *args):
        carry = args[-2]
        captured["carry"] = carry
        return np.asarray(1.0), carry, {}

    monkeypatch.setattr(MimicReward, "__call__", fake_reward)
    carry = _Carry(np.asarray(0.2), np.asarray(0.2))
    data = SimpleNamespace(act=np.asarray([0.25, 0.75]))

    _, next_carry, reward_info = _reward_without_initialization()(  # type: ignore[misc]
        None,
        None,
        None,
        False,
        {},
        _reward_env(),
        None,
        data,
        carry,
        np,
    )

    assert float(captured["carry"].qvel_w_sum) == 0.25
    assert float(captured["carry"].root_vel_w_sum) == 0.5
    assert next_carry is captured["carry"]
    assert float(reward_info["activation_energy_raw"]) == 0.3125
    assert float(reward_info["reward_core_upper_body"]) == 0.8
    assert float(reward_info["reward_terminal_quality_bonus"]) == 0.0
    assert not bool(reward_info["numerics_nonfinite_reward"])


@pytest.mark.parametrize("backend", [np, jnp])
def test_terra_reward_penalizes_silent_muscles_with_normalized_floor(monkeypatch, backend):
    def fake_reward(self, *args):
        carry = args[-2]
        return (
            backend.asarray(1.0),
            carry,
            {
                "penalty_total": backend.asarray(-0.2),
                "penalty_activation_energy": backend.asarray(-0.1),
            },
        )

    monkeypatch.setattr(MimicReward, "__call__", fake_reward)
    carry = _Carry(np.asarray(0.2), np.asarray(0.2))
    data = SimpleNamespace(act=backend.asarray([0.0, 0.01, 0.02, 0.04]))

    reward, _, reward_info = _reward_without_initialization(
        activation_floor=0.02,
        activation_floor_coeff=1.0,
    )(
        None,
        None,
        None,
        False,
        {},
        _reward_env(),
        None,
        data,
        carry,
        backend,
    )

    assert float(reward) == pytest.approx(1.4875)
    assert float(reward_info["activation_energy_raw"]) == pytest.approx(0.000525)
    assert float(reward_info["activation_floor_violation_raw"]) == pytest.approx(0.3125)
    assert float(reward_info["activation_below_floor_fraction_raw"]) == pytest.approx(0.5)
    assert float(reward_info["penalty_activation_floor"]) == pytest.approx(-0.3125)
    assert float(reward_info["penalty_activity_regularization"]) == pytest.approx(-0.4125)
    assert float(reward_info["penalty_total"]) == pytest.approx(-0.5125)


def test_terra_reward_contains_nonfinite_post_base_terms(monkeypatch):
    """TERRA additions must not undo MimicReward's finite scalar boundary."""

    def fake_reward(self, *args):
        carry = args[-2]
        return (
            np.asarray(1.0),
            carry,
            {
                "reward_qpos": np.asarray(np.nan),
                "reward_total": np.asarray(1.0),
            },
        )

    monkeypatch.setattr(MimicReward, "__call__", fake_reward)
    reward_fn = _reward_without_initialization()
    reward_fn._core_upper_body_tracking = lambda *_args: (
        np.asarray(np.nan),
        np.asarray(np.inf),
    )
    carry = _Carry(np.asarray(0.2), np.asarray(0.2))
    data = SimpleNamespace(act=np.asarray([np.inf]))

    reward, _, reward_info = reward_fn(  # type: ignore[misc]
        None,
        None,
        None,
        False,
        {},
        _reward_env(at_end=True),
        None,
        data,
        carry,
        np,
    )

    assert float(reward) == 0.0
    assert bool(reward_info["numerics_nonfinite_reward"])
    for name, value in reward_info.items():
        if name != "numerics_nonfinite_reward":
            assert np.isfinite(np.asarray(value)).all(), name


def test_terra_reward_can_preserve_curriculum_managed_weights(monkeypatch):
    def fake_reward(self, *args):
        carry = args[-2]
        return np.asarray(1.0), carry, {}

    monkeypatch.setattr(MimicReward, "__call__", fake_reward)
    carry = _Carry(np.asarray(0.1), np.asarray(0.15))
    data = SimpleNamespace(act=np.asarray([]))

    _, next_carry, reward_info = _reward_without_initialization(dynamic=True)(  # type: ignore[misc]
        None,
        None,
        None,
        False,
        {},
        _reward_env(),
        None,
        data,
        carry,
        np,
    )

    assert next_carry is carry
    assert float(reward_info["activation_energy_raw"]) == 0.0


def test_terra_reward_adds_quality_bonus_only_at_successful_trajectory_end(monkeypatch):
    def fake_reward(self, *args):
        return np.asarray(1.0), args[-2], {"reward_total": np.asarray(1.0)}

    monkeypatch.setattr(MimicReward, "__call__", fake_reward)
    carry = _Carry(np.asarray(0.2), np.asarray(0.2))
    data = SimpleNamespace(act=np.asarray([]))

    reward, _, reward_info = _reward_without_initialization()(
        None, None, None, False, {}, _reward_env(at_end=True), None, data, carry, np
    )
    failed_reward, _, failed_info = _reward_without_initialization()(
        None, None, None, True, {}, _reward_env(at_end=True), None, data, carry, np
    )

    assert float(reward) == pytest.approx(21.8)
    assert float(reward_info["reward_terminal_quality_bonus"]) == 20.0
    assert float(failed_reward) == pytest.approx(1.8)
    assert float(failed_info["reward_terminal_quality_bonus"]) == 0.0


def test_terra_metrics_do_not_shift_preserved_trajectory_roots():
    handler = object.__new__(TerraMetricsHandler)
    handler._preserve_trajectory_root_xy = True

    assert handler._get_root_xy_offset(SimpleNamespace()) is None


def test_backend_integrations_are_idempotent():
    from musclemimic.algorithms.ppo import update
    from musclemimic.runner import engine
    from musclemimic.utils import metrics

    install_backend_integrations()
    summarize = update.summarize_training_rollout
    install_backend_integrations()
    assert engine.MetricsHandler is TerraMetricsHandler
    assert update.summarize_training_rollout is summarize
    assert metrics.VALIDATION_STEP_METRIC_KEYS.count("activation_energy_raw") == 1


def test_validation_video_can_be_disabled_while_validation_stays_active(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "validation": {"active": True, "video_active": False},
                "policy_snapshots": {"active": False},
            }
        }
    )

    assert TerraHooks().build_video_recorder(str(tmp_path), config) is None


def test_validation_video_builds_named_motion_panel(tmp_path):
    motions = [
        "stairs_up=Gait120/S100/StairAscent/Trial01/AllSteps_stageii",
        "ramp_down=Gait120/S104/SlopeDescent/Trial01/AllSteps_stageii",
    ]
    config = OmegaConf.create(
        {
            "experiment": {
                "validation": {
                    "active": True,
                    "video_active": True,
                    "video_frequency": 1,
                    "video_length": 616,
                    "video_motions": motions,
                },
                "policy_snapshots": {"active": False},
            }
        }
    )

    recorder = TerraHooks().build_video_recorder(str(tmp_path), config)

    assert recorder is not None
    assert recorder.frequency == 1
    assert recorder.length == 616
    assert recorder.named_motions == tuple(tuple(motion.split("=", 1)) for motion in motions)


def test_frame_zero_reset_wrapper_is_scoped_by_handler_configuration():
    from loco_mujoco.trajectory.handler import TrajectoryHandler

    install_backend_integrations()
    handler = object.__new__(TrajectoryHandler)
    handler.random_start = True
    handler.start_from_random_step = True
    handler.frame_zero_reset_probability = 1.0
    handler.traj = SimpleNamespace(data=SimpleNamespace(split_points=np.asarray([0, 12])))
    carry = _TrajectoryResetCarry(key=None, selected_traj_idx=-1)

    _, next_carry = handler.reset_state(None, None, None, carry, np)

    assert int(next_carry.traj_state.subtraj_step_no) == 0
    assert int(next_carry.traj_state.subtraj_step_no_init) == 0


def test_paired_box_terrain_switches_geometry_and_height_with_trajectory():
    terrains = [
        {"boxes": [{"pos": [0.0, 0.0, 0.1], "size": [1.0, 0.5, 0.1], "name": "terrain_box_0"}]},
        {"boxes": [{"pos": [3.0, 0.0, 0.2], "size": [0.75, 0.5, 0.2], "name": "terrain_box_0"}]},
    ]
    placeholder_env = SimpleNamespace()
    terrain = PairedBoxTerrain(placeholder_env, terrains=terrains, contact_margin=0.003)
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    model = terrain.modify_spec(spec).compile()
    env = SimpleNamespace(_model=model)
    geom_id = terrain._resolve_geom_ids(env)[0]
    assert model.geom_margin[geom_id] == pytest.approx(0.003)

    carry_0 = SimpleNamespace(traj_state=SimpleNamespace(traj_no=0))
    carry_1 = SimpleNamespace(traj_state=SimpleNamespace(traj_no=1))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    terrain.update(env, model, data, carry_0, np)
    np.testing.assert_allclose(model.geom_pos[geom_id], [0.0, 0.0, 0.1])
    terrain.update(env, model, data, carry_1, np)
    np.testing.assert_allclose(model.geom_pos[geom_id], [3.0, 0.0, 0.2])
    np.testing.assert_allclose(data.geom_xpos[geom_id], [3.0, 0.0, 0.2])
    assert model.geom_rbound[geom_id] == pytest.approx(np.linalg.norm([0.75, 0.5, 0.2]))

    mjx_model = mujoco.mjx.put_model(model)
    mjx_data = mujoco.mjx.make_data(mjx_model)

    @jax.jit
    def geometry(trajectory_index):
        carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=trajectory_index))
        selected_model, selected_data, _ = terrain.update(env, mjx_model, mjx_data, carry, jnp)
        return (
            selected_model.geom_pos[geom_id],
            selected_model.geom_size[geom_id],
            selected_data.geom_xpos[geom_id],
        )

    positions, sizes, world_positions = jax.vmap(geometry)(jnp.asarray([0, 1]))
    np.testing.assert_allclose(np.asarray(positions), [[0.0, 0.0, 0.1], [3.0, 0.0, 0.2]])
    np.testing.assert_allclose(np.asarray(sizes), [[1.0, 0.5, 0.1], [0.75, 0.5, 0.2]])
    np.testing.assert_allclose(np.asarray(world_positions), [[0.0, 0.0, 0.1], [3.0, 0.0, 0.2]])

    warp_model = mujoco.mjx.put_model(model, impl="warp")

    @jax.jit
    def warp_geometry(trajectory_index):
        carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=trajectory_index))
        selected_model, _, _ = terrain.update(env, warp_model, None, carry, jnp)
        return selected_model.geom_aabb[geom_id]

    np.testing.assert_allclose(
        np.asarray(warp_geometry(jnp.asarray(1))),
        [[0.0, 0.0, 0.0], [0.75, 0.5, 0.2]],
    )

    @jax.jit
    def height(trajectory_index, x):
        carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=trajectory_index))
        return terrain.sample_heights_at_points(x, jnp.asarray([0.0]), None, carry, jnp)

    assert float(height(jnp.asarray(0), jnp.asarray([0.0]))[0]) == pytest.approx(0.2)
    assert float(height(jnp.asarray(1), jnp.asarray([3.0]))[0]) == pytest.approx(0.4)


def test_ppo_multi_motion_config_is_dedicated_and_safe_by_default(monkeypatch, tmp_path):
    from terra import training

    monkeypatch.setenv("TERRA_MOTIONS", '["Study/Z", "Study/A"]')
    monkeypatch.setenv("TERRA_RETARGETED_MOTIONS", str(tmp_path))
    monkeypatch.setenv("TERRA_VALIDATION_STEPS", "64")

    config = training._compose_training_config(algorithm="ppo", multi_motion=True)

    assert (
        training._validate_training_config(
            config,
            algorithm="ppo",
            config_name=training.TERRA_MULTI_GPU_PPO_CONFIG,
        )
        == "PPOJax"
    )
    assert config.experiment.algorithm == "PPOJax"
    assert config.experiment.distributed.num_devices == -1
    assert config.experiment.trajectory.sharding == "none"
    assert config.experiment.env_params.mjx_warp_graph_mode == "warp"
    assert config.experiment.validation.num_envs == 32
    assert config.experiment.validation.evaluate_all is True
    assert config.experiment.validation.deterministic is False
    assert config.experiment.validation.minimum_total_rollouts == 100
    assert config.experiment.validation.rollouts_per_motion == 3
    assert config.experiment.validation.max_parallel_rollouts == 1024
    assert config.experiment.validation.video_active is False
    assert list(config.experiment.validation.video_motions) == []
    assert config.experiment.exact_resume is False
    assert config.experiment.save_runtime_state is False
    assert config.experiment.async_checkpointing is False
    assert config.experiment.env_params.heightmap_grid_rows == 11
    assert config.experiment.env_params.heightmap_grid_forward_offset == 0.0
    assert config.experiment.env_params.goal_params.enable_future_reference_observations is True
    assert config.experiment.env_params.goal_params.future_reference_stride == 10
    assert config.experiment.env_params.goal_params.future_reference_horizon == 100
    assert config.experiment.env_params.goal_params.enable_motion_phase is False
    assert config.experiment.env_params.reward_params.activation_energy_coeff == 2.0
    assert config.experiment.env_params.reward_params.activation_floor == 0.02
    assert config.experiment.env_params.reward_params.activation_floor_coeff == 1.0
    assert config.experiment.ppo_config.num_steps == 40
    assert config.experiment.ppo_config.num_minibatches == 256
    assert config.experiment.ppo_config.gamma == 0.995
    assert config.experiment.ppo_config.gae_lambda == 0.97
    assert config.experiment.lr_schedule_type == "warmup_hold_cosine"
    assert config.experiment.hold_fraction == 0.55
    assert config.experiment.adaptive_termination.enabled is True
    assert config.experiment.adaptive_sampling.balance_motion_groups is True
    assert config.experiment.nonfinite_guard.enabled is True


class _IdentitySiteMapper:
    requires_mapping = False


class _ReferenceTrajectory:
    def __init__(self, current, initial):
        self.current = current
        self.initial = initial

    def get_current_traj_data(self, carry, backend):
        return self.current

    def get_init_traj_data(self, carry, backend):
        return self.initial


def _termination_fixture(*, reference_xy=(0.0, 0.0), preserve_root_xy=True, n_sites=2):
    reference_xy = np.asarray(reference_xy, dtype=np.float32)
    reference_sites = np.tile(
        np.asarray([reference_xy[0], reference_xy[1], 1.0], dtype=np.float32),
        (n_sites, 1),
    )
    reference_sites[:, 0] += np.arange(n_sites, dtype=np.float32)
    reference_qpos = np.asarray(
        [reference_xy[0], reference_xy[1], 1.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    reference = SimpleNamespace(site_xpos=reference_sites, qpos=reference_qpos)
    initial = SimpleNamespace(qpos=reference_qpos)
    env = SimpleNamespace(
        th=_ReferenceTrajectory(reference, initial),
        _goal=SimpleNamespace(_rel_site_ids=np.arange(n_sites), _site_mapper=_IdentitySiteMapper()),
        preserve_trajectory_root_xy=preserve_root_xy,
    )
    carry = SimpleNamespace(termination_threshold=np.asarray(0.25, dtype=np.float32))
    handler = object.__new__(TerraGlobalMPJPETerminalStateHandler)
    handler.enable_site_check = True
    handler.root_orientation_threshold = 1.0
    handler._root_qpos_ids_xy = np.asarray([0, 1])
    handler._root_qpos_ids_quat = np.asarray([3, 4, 5, 6])
    handler._has_exclusions = False
    handler._n_included = n_sites
    handler._include_indices = np.arange(n_sites)
    handler.core_upper_body_mean_site_deviation_threshold = None
    handler._core_upper_body_indices = np.asarray([], dtype=int)
    return handler, env, carry, reference


def test_global_mpjpe_uses_mean_distance_not_maximum():
    handler, env, carry, reference = _termination_fixture()
    current_sites = reference.site_xpos.copy()
    current_sites[1, 0] += 0.4
    state = SimpleNamespace(site_xpos=current_sites, qpos=reference.qpos.copy())

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert not bool(absorbing), "one 0.4 m keypoint error has a 0.2 m mean and should remain below threshold"


def test_global_mpjpe_terminates_above_quarter_meter():
    handler, env, carry, reference = _termination_fixture()
    state = SimpleNamespace(
        site_xpos=reference.site_xpos + np.asarray([0.26, 0.0, 0.0]),
        qpos=reference.qpos.copy(),
    )

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert bool(absorbing)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_global_mpjpe_terminates_nonfinite_site_state(invalid):
    handler, env, carry, reference = _termination_fixture()
    current_sites = reference.site_xpos.copy()
    current_sites[0, 0] = invalid
    state = SimpleNamespace(site_xpos=current_sites, qpos=reference.qpos.copy())

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert bool(absorbing)


def test_core_upper_body_mean_terminates_without_global_mean_violation():
    handler, env, carry, reference = _termination_fixture(n_sites=17)
    handler.core_upper_body_mean_site_deviation_threshold = 0.25
    handler._core_upper_body_indices = np.asarray([0, 1, 2, 3])
    current_sites = reference.site_xpos.copy()
    current_sites[:4, 2] -= 0.3
    state = SimpleNamespace(site_xpos=current_sites, qpos=reference.qpos.copy())

    global_violation = handler._check_mean_site_deviation(env, state, carry, np)
    core_violation = handler._check_core_upper_body_mean_site_deviation(env, state, carry, np)
    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert not bool(global_violation), "four 0.3 m errors average to only 0.071 m globally"
    assert bool(core_violation)
    assert bool(absorbing)


def test_core_upper_body_threshold_can_follow_training_curriculum():
    handler, env, carry, reference = _termination_fixture(n_sites=17)
    handler.core_upper_body_mean_site_deviation_threshold = 0.15
    handler.mean_site_deviation_threshold = 0.15
    handler.core_upper_body_uses_global_threshold = True
    handler.curriculum_initial_global_threshold = 0.25
    handler.core_upper_body_curriculum_initial_threshold = 0.20
    handler._core_upper_body_indices = np.asarray([0, 1, 2, 3])
    current_sites = reference.site_xpos.copy()
    current_sites[:4, 2] -= 0.20
    state = SimpleNamespace(site_xpos=current_sites, qpos=reference.qpos.copy())

    assert not bool(handler._check_core_upper_body_mean_site_deviation(env, state, carry, np))
    carry = SimpleNamespace(termination_threshold=np.asarray(0.15, dtype=np.float32))
    assert bool(handler._check_core_upper_body_mean_site_deviation(env, state, carry, np))
    handler.core_upper_body_uses_global_threshold = False
    assert bool(handler._check_core_upper_body_mean_site_deviation(env, state, carry, np))


def test_global_mpjpe_terminates_on_mjx_path():
    handler, env, carry, reference = _termination_fixture()
    state = SimpleNamespace(
        site_xpos=jnp.asarray(reference.site_xpos) + jnp.asarray([0.26, 0.0, 0.0]),
        qpos=jnp.asarray(reference.qpos),
    )

    absorbing, _ = handler.mjx_is_absorbing(env, jnp.asarray([]), {}, state, carry)

    assert bool(absorbing)


def test_global_mpjpe_does_not_shift_preserved_reference_xy():
    handler, env, carry, reference = _termination_fixture(reference_xy=(5.0, 3.0), preserve_root_xy=True)
    state = SimpleNamespace(site_xpos=reference.site_xpos.copy(), qpos=reference.qpos.copy())

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert not bool(absorbing)


def test_global_mpjpe_shifts_reference_xy_for_local_origin():
    handler, env, carry, reference = _termination_fixture(reference_xy=(5.0, 3.0), preserve_root_xy=False)
    state = SimpleNamespace(
        site_xpos=reference.site_xpos - np.asarray([5.0, 3.0, 0.0]),
        qpos=reference.qpos.copy(),
    )

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert not bool(absorbing)


def test_global_mpjpe_retains_root_orientation_guard():
    handler, env, carry, reference = _termination_fixture()
    half_angle = np.pi / 4.0
    rotated_qpos = reference.qpos.copy()
    rotated_qpos[3:7] = [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]
    state = SimpleNamespace(site_xpos=reference.site_xpos.copy(), qpos=rotated_qpos)

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert bool(absorbing)


def test_global_mpjpe_terminates_nonfinite_root_orientation():
    handler, env, carry, reference = _termination_fixture()
    invalid_qpos = reference.qpos.copy()
    invalid_qpos[3:7] = np.nan
    state = SimpleNamespace(site_xpos=reference.site_xpos.copy(), qpos=invalid_qpos)

    absorbing, _ = handler.is_absorbing(env, np.asarray([]), {}, state, carry)

    assert bool(absorbing)


def test_terra_termination_handler_registers_with_components():
    from terra.rl import register_components

    register_components()

    assert (
        TerminalStateHandler.registered["TerraGlobalMPJPETerminalStateHandler"] is TerraGlobalMPJPETerminalStateHandler
    )
