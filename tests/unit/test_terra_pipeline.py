import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from flax import struct

import terra.pipeline as terra_pipeline
from loco_mujoco.core import ObservationType
from loco_mujoco.core.domain_randomizer.default import DefaultRandomizer, DefaultRandomizerState
from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.core.terrain import BoxSpec, StaticTerrain, TerrainSpec
from loco_mujoco.task_factories.dataset_confs import AMASSDatasetConf
from loco_mujoco.task_factories.imitation_factory import ImitationFactory
from musclemimic.core.goals.trajectory import GoalTrajMimic
from musclemimic.core.reward.trajectory_based import MimicReward
from musclemimic.environments.base import LocoEnv
from terra.rl import register_components
from terra.rl.egocentric import (
    heading_rotation_matrix,
    root_velocity_in_heading_frame,
    rotation_matrix_from_quaternion,
)
from terra.rl.environment import (
    _delete_explicit_contact_pairs,
    _normalize_disabled_contact_pairs,
    _TerraObservationLayout,
)
from terra.rl.observations import TerraGoal
from terra.rl.rendering import TerraFullBodyTrackingGoalVisual, TerraGoalVisual
from terra.rl.rewards import TerraReward
from terra.rl.task_factory import TerraImitationFactory
from terra.rl.tracking import TerraFullBodyTrackingGoal, tracking_goal_dimension, validate_lookahead_steps
from terra.terrain.metadata import TerrainMetadata

MOTION = "EKUT/EKUT/234/WSUF03_poses"


def _set_single_motion_environment(monkeypatch, cache_root: Path) -> None:
    monkeypatch.setenv("TERRA_RETARGETED_MOTIONS", str(cache_root))
    monkeypatch.setenv("TERRA_MOTION", MOTION)


def _paired_conf(cache_root: Path) -> AMASSDatasetConf:
    return AMASSDatasetConf(
        rel_dataset_path=MOTION,
        retargeting_method="terra",
        output_cache_subdir="terra",
        cache_root=str(cache_root),
        load_paired_terrain=True,
        require_nonflat_terrain=True,
        allow_cache_download=False,
    )


def test_paired_amass_terrain_resolves_motion_and_metadata(tmp_path):
    cache_dir = tmp_path / "MyoFullBody" / "terra" / "EKUT" / "EKUT" / "234"
    cache_dir.mkdir(parents=True)
    motion_path = cache_dir / "WSUF03_poses.npz"
    motion_path.touch()
    terrain_path = cache_dir / "WSUF03_poses_terrain.json"
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.5, 0.5, 0.1), name="terrain_box_0"),))
    TerrainMetadata.from_terrain(terrain).save(terrain_path)

    paired = TerraImitationFactory._paired_terrain(
        "MjxMyoFullBody",
        _paired_conf(tmp_path),
        MOTION,
    )

    assert paired.motion_path == motion_path
    assert paired.terrain_path == terrain_path
    assert paired.metadata.terrain == terrain


def test_flat_paired_amass_motion_synthesizes_zero_height_terrain(tmp_path):
    cache_dir = tmp_path / "MyoFullBody" / "terra" / "EKUT" / "EKUT" / "234"
    cache_dir.mkdir(parents=True)
    motion_path = cache_dir / "WSUF03_poses.npz"
    motion_path.touch()
    config = dataclasses.replace(_paired_conf(tmp_path), require_nonflat_terrain=False)

    paired = TerraImitationFactory._paired_terrain("MjxMyoFullBody", config, MOTION)

    assert paired.motion_path == motion_path
    assert paired.terrain_path is None
    assert paired.metadata.terrain == TerrainSpec()


def test_all_flat_paired_motions_select_static_terrain(monkeypatch, tmp_path):
    motions = ["KIT/1/walk01_poses", "KIT/2/walk02_poses"]
    captured = {}

    monkeypatch.setattr(
        TerraImitationFactory,
        "get_amass_dataset_paths",
        staticmethod(lambda _config: motions),
    )
    monkeypatch.setattr(
        TerraImitationFactory,
        "_paired_terrain",
        classmethod(
            lambda _cls, _env_name, _config, motion: SimpleNamespace(
                motion_path=tmp_path / f"{motion}.npz",
                terrain_path=tmp_path / f"{motion}_terrain.json",
                metadata=TerrainMetadata.from_terrain(TerrainSpec()),
            )
        ),
    )

    def fake_make(_cls, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(th=SimpleNamespace(n_trajectories=len(motions)))

    monkeypatch.setattr(ImitationFactory, "make", classmethod(fake_make))
    config = AMASSDatasetConf(
        rel_dataset_path=motions,
        retargeting_method="terra",
        output_cache_subdir="terra",
        cache_root=str(tmp_path),
        load_paired_terrain=True,
        require_nonflat_terrain=False,
        allow_cache_download=False,
    )

    environment = TerraImitationFactory.make("MjxMyoFullBody", config)

    assert captured["terrain_type"] == "StaticTerrain"
    assert captured["terrain_params"] == {}
    assert captured["amass_dataset_conf"].load_paired_terrain is False
    assert environment.paired_motion_paths == tuple(tmp_path / f"{motion}.npz" for motion in motions)


def test_paired_nonflat_terrain_receives_collision_margin(monkeypatch, tmp_path):
    motions = ["Gait120/S001/StairDescent/Trial01", "Gait120/S002/StairDescent/Trial01"]
    captured = {}
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.5, 0.5, 0.1), name="terrain_box_0"),))
    monkeypatch.setattr(
        TerraImitationFactory,
        "get_amass_dataset_paths",
        staticmethod(lambda _config: motions),
    )
    monkeypatch.setattr(
        TerraImitationFactory,
        "_paired_terrain",
        classmethod(
            lambda _cls, _env_name, _config, motion: SimpleNamespace(
                motion_path=tmp_path / f"{motion}.npz",
                terrain_path=tmp_path / f"{motion}_terrain.json",
                metadata=TerrainMetadata.from_terrain(terrain),
            )
        ),
    )

    def fake_make(_cls, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(th=SimpleNamespace(n_trajectories=len(motions)))

    monkeypatch.setattr(ImitationFactory, "make", classmethod(fake_make))
    config = AMASSDatasetConf(
        rel_dataset_path=motions,
        retargeting_method="terra",
        output_cache_subdir="terra",
        cache_root=str(tmp_path),
        load_paired_terrain=True,
        require_nonflat_terrain=True,
        allow_cache_download=False,
    )

    TerraImitationFactory.make("MjxMyoFullBody", config, terrain_collision_margin=0.003)

    assert captured["terrain_type"] == "PairedBoxTerrain"
    assert captured["terrain_params"]["contact_margin"] == pytest.approx(0.003)


def test_gait120_motion_groups_are_stable_and_reset_mixture_is_validated():
    env = SimpleNamespace(th=SimpleNamespace())
    motions = [
        "Gait120/S001/StairAscent/trial01",
        "Gait120/S002/StairDescent/trial01",
        "Gait120/S003/SlopeAscent/trial01",
        "Gait120/S004/SlopeDescent/trial01",
        "Other/S005/Flat/trial01",
    ]

    TerraImitationFactory._configure_motion_groups(env, motions)
    TerraImitationFactory._configure_reset_mixture(env, 0.5, 0.25)

    assert env.trajectory_group_names == (
        "other",
        "slope_ascent",
        "slope_descent",
        "stair_ascent",
        "stair_descent",
    )
    assert env.trajectory_group_ids == (3, 4, 1, 2, 0)
    assert env.th.frame_zero_reset_probability == 0.5
    assert env.th.first_quarter_reset_probability == 0.25
    with pytest.raises(ValueError, match="sum to at most 1"):
        TerraImitationFactory._configure_reset_mixture(env, 0.8, 0.3)


def test_future_reference_observation_has_height_and_root_relative_ankles_only():
    goal = TerraGoal({"upper_body_xml_name": "torso", "sites_for_mimic": []})
    goal._future_reference_offsets = tuple(range(10, 101, 10))
    goal._root_qpos_full_ind = np.asarray([0, 1, 2, 3, 4, 5, 6])
    goal._future_ankle_model_ids = np.asarray([0, 1])
    goal._site_mapper = SimpleNamespace(requires_mapping=False)

    class Handler:
        @staticmethod
        def len_trajectory(_trajectory_id):
            return 61

        @staticmethod
        def get_traj_data_at(_trajectory_id, step, _carry, backend):
            step = backend.asarray(step)
            root = backend.asarray([2.0, -1.0, 1.0 + 0.01 * step])
            left = root + backend.asarray([0.2, 0.1, -0.9])
            right = root + backend.asarray([0.2, -0.1, -0.9])
            return SimpleNamespace(
                qpos=backend.concatenate((root, backend.asarray([1.0, 0.0, 0.0, 0.0]))),
                site_xpos=backend.stack((left, right)),
            )

    carry = SimpleNamespace(
        traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=20),
    )
    reference = Handler.get_traj_data_at(0, 20, carry, np)
    observation = goal._future_reference_observation(
        SimpleNamespace(th=Handler()),
        reference,
        carry,
        np,
    )

    assert observation.shape == (70,)
    expected_ankles = np.asarray([0.2, 0.1, -0.9, 0.2, -0.1, -0.9])
    for index, offset in enumerate(range(10, 101, 10)):
        clipped_offset = min(offset, 40)
        np.testing.assert_allclose(observation[index * 7], 0.01 * clipped_offset)
        np.testing.assert_allclose(observation[index * 7 + 1 : index * 7 + 7], expected_ankles)


def test_future_reference_observation_can_disable_extra_features():
    goal = TerraGoal({"upper_body_xml_name": "torso", "sites_for_mimic": []})
    parameters = {
        "enable_future_reference_observations": False,
        "future_reference_stride": 10,
        "future_reference_horizon": 100,
    }

    goal._configure_future_reference(parameters)
    observation = goal._future_reference_observation(
        SimpleNamespace(),
        SimpleNamespace(qpos=np.zeros(7, dtype=np.float32)),
        SimpleNamespace(),
        np,
    )

    assert goal._future_reference_offsets == ()
    assert observation.shape == (0,)
    assert parameters == {}


def test_terra_reward_defaults_to_global_root_and_joint_only_tracking(monkeypatch):
    captured = {}

    def capture_init(_self, _env, **kwargs):
        captured.update(kwargs)
        _self._info_props = {
            "sites_for_mimic": [
                "pelvis_mimic",
                "upper_body_mimic",
                "head_mimic",
                "left_shoulder_mimic",
                "right_shoulder_mimic",
            ]
        }

    monkeypatch.setattr(MimicReward, "__init__", capture_init)
    TerraReward(object())

    assert captured["global_root_tracking"] is True
    assert captured["joint_only_qpos_qvel"] is True
    assert captured["root_velocity_frame"] == "global"


def test_local_rl_components_are_registered():
    register_components()

    assert Goal.registered["TerraGoal"].__module__ == "terra.rl.observations"
    assert Goal.registered["TerraGoalVisual"].__module__ == "terra.rl.rendering"
    assert Goal.registered["TerraFullBodyTrackingGoal"] is TerraFullBodyTrackingGoal
    assert Goal.registered["TerraFullBodyTrackingGoalVisual"] is TerraFullBodyTrackingGoalVisual
    assert LocoEnv.registered_envs["MjxMyoFullBody"].__module__ == "terra.rl.environment"


def test_disabled_explicit_contact_pair_is_deleted_before_compilation():
    specification = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body><geom name="arm" size="0.1"/></body>
            <body><geom name="torso" size="0.1"/></body>
          </worldbody>
          <contact><pair geom1="arm" geom2="torso"/></contact>
        </mujoco>
        """
    )
    pairs = _normalize_disabled_contact_pairs([["torso", "arm"]])

    model = _delete_explicit_contact_pairs(specification, pairs).compile()

    assert model.npair == 0


def test_disabled_contact_pairs_reject_duplicates_and_missing_pairs():
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_disabled_contact_pairs([["arm", "torso"], ["torso", "arm"]])

    specification = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="does not exist"):
        _delete_explicit_contact_pairs(specification, (("arm", "torso"),))


def test_upstream_tracking_horizons_and_dimension_are_exact():
    assert validate_lookahead_steps([1, 20, 40, 60, 80]) == (1, 20, 40, 60, 80)
    assert tracking_goal_dimension(n_relative_sites=16, n_horizons=5) == 452
    with pytest.raises(ValueError, match="start with 1"):
        validate_lookahead_steps([20, 40])


def test_future_target_clamps_to_clip_end_and_marks_invalid():
    goal = TerraFullBodyTrackingGoal({"upper_body_xml_name": "torso", "sites_for_mimic": []})

    target, valid = goal._bounded_target_step(50, 20, 61, np)
    assert target == 60
    assert not bool(valid)

    target, valid = goal._bounded_target_step(20, 20, 61, np)
    assert target == 40
    assert bool(valid)


def test_fullbody_tracking_normalizes_parent_root_velocity_indices_for_jax(monkeypatch):
    specification = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis"><freejoint name="root"/><geom size="0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    model = specification.compile()
    data = mujoco.MjData(model)
    goal = TerraFullBodyTrackingGoal({"upper_body_xml_name": "torso", "sites_for_mimic": []})
    goal.lookahead_steps = (1,)
    goal._info_props = {"sites_for_mimic": ["pelvis_mimic"]}

    def initialize_parent(self, env, model, data, current_obs_size):
        self._root_qvel_ind = list(range(model.nv))

    monkeypatch.setattr(GoalTrajMimic, "_init_from_mj", initialize_parent)
    goal._init_from_mj(SimpleNamespace(root_free_joint_xml_name="root"), model, data, 0)

    assert isinstance(goal._root_qvel_indices, np.ndarray)
    np.testing.assert_array_equal(
        np.asarray(jnp.arange(model.nv)[goal._root_qvel_indices]),
        np.arange(model.nv),
    )


def test_heading_frame_root_velocity_is_invariant_to_global_yaw():
    root_qvel = np.asarray([1.0, 0.0, 0.25, 0.0, 0.1, 0.2])
    identity = rotation_matrix_from_quaternion(np.asarray([1.0, 0.0, 0.0, 0.0]), np)
    yaw_90 = rotation_matrix_from_quaternion(
        np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        np,
    )

    baseline = root_velocity_in_heading_frame(
        root_qvel,
        identity,
        heading_rotation_matrix(identity, np),
        np,
    )
    rotated_world_velocity = np.concatenate([yaw_90 @ root_qvel[:3], root_qvel[3:]])
    rotated = root_velocity_in_heading_frame(
        rotated_world_velocity,
        yaw_90,
        heading_rotation_matrix(yaw_90, np),
        np,
    )

    np.testing.assert_allclose(rotated, baseline, atol=1e-7)


def test_egocentric_root_layout_replaces_quaternion_but_keeps_joint_state_and_terrain():
    specification = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis"><freejoint name="root"/><joint name="hip" type="hinge"/></body>
          </worldbody>
        </mujoco>
        """
    )
    layout = SimpleNamespace(
        root_free_joint_xml_name="root",
        _enable_joint_pos_observations=True,
        _enable_joint_vel_observations=True,
        _enable_global_root_position_observation=False,
        _use_egocentric_root_observations=True,
        _enable_muscle_length_observations=False,
        _enable_muscle_velocity_observations=False,
        _enable_muscle_force_observations=False,
        _enable_muscle_excitation_observations=False,
        _enable_muscle_activation_observations=False,
        _enable_touch_sensor_observations=False,
        _enable_heightmap_observations=False,
    )

    observations = _TerraObservationLayout._get_observation_specification(layout, specification)
    names = [observation.name for observation in observations]

    assert names == ["q_root_height", "projected_gravity", "q_all_pos", "dq_free_joint_heading", "dq_all_vel"]


def test_flat_layout_uses_standard_heightmap_and_returns_negative_pelvis_height():
    specification = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis">
              <freejoint name="root"/>
              <geom type="sphere" size="0.1" mass="1.0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    layout = SimpleNamespace(
        root_free_joint_xml_name="root",
        _enable_joint_pos_observations=False,
        _enable_joint_vel_observations=False,
        _enable_muscle_length_observations=False,
        _enable_muscle_velocity_observations=False,
        _enable_muscle_force_observations=False,
        _enable_muscle_excitation_observations=False,
        _enable_muscle_activation_observations=False,
        _enable_touch_sensor_observations=False,
        _enable_heightmap_observations=True,
        _heightmap_grid_rows=11,
        _heightmap_grid_cols=11,
        _heightmap_grid_resolution=0.1,
        _heightmap_grid_forward_offset=0.0,
        _heightmap_body_name="pelvis",
    )

    observations = _TerraObservationLayout._get_observation_specification(layout, specification)
    assert len(observations) == 1
    heightmap = observations[0]
    assert type(heightmap) is ObservationType.HeightMatrix
    assert heightmap.dim == 121

    model = specification.compile()
    data = mujoco.MjData(model)
    data.qpos[:7] = np.asarray([0.0, 0.0, 1.75, 1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(
        root_free_joint_xml_name="root",
        root_body_name="pelvis",
        _terrain=StaticTerrain(None),
    )
    heightmap._init_from_mj(env, model, data, current_obs_size=0)

    carry = object()
    values, updated_carry = heightmap.get_obs_and_update_state(
        env,
        model,
        data,
        carry,
        np,
    )
    np.testing.assert_allclose(values, np.full(121, -1.75), atol=1e-12)
    assert updated_carry is carry

    mjx_model = mujoco.mjx.put_model(model)
    mjx_data = mujoco.mjx.put_data(model, data)
    mjx_values, _ = heightmap.get_obs_and_update_state(
        env,
        mjx_model,
        mjx_data,
        carry,
        jnp,
    )
    np.testing.assert_allclose(np.asarray(mjx_values), np.full(121, -1.75), atol=1e-6)


def test_visual_geometry_binding_uses_names_after_terrain_insertion():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="terrain_box_0" type="box" group="2" size="1 1 0.1"/>
            <body name="link_a"><geom name="bone_a" type="capsule" size="0.1 0.2"/></body>
            <body name="link_b"><geom name="bone_b" type="sphere" size="0.15"/></body>
          </worldbody>
        </mujoco>
        """
    )
    goal = TerraGoalVisual({"upper_body_xml_name": "torso", "sites_for_mimic": []})
    goal._geom_names = ("bone_a", "bone_b")
    goal._bound_model_id = None

    goal._bind_visual_geometries(model)

    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in goal._geom_ids)
    assert names == goal._geom_names
    np.testing.assert_array_equal(goal._geom_type[:, 0], model.geom_type[goal._geom_ids])


@struct.dataclass
class _PushCarry:
    key: object
    domain_randomizer_state: DefaultRandomizerState
    cur_step_in_episode: int


def test_velocity_push_changes_only_root_twist_and_reschedules(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='prefix'><joint type='slide'/><joint type='hinge'/><geom size='.1'/></body>"
        "<body name='pelvis'><freejoint name='root'/><geom size='.1'/></body></worldbody></mujoco>"
    )
    setup = SimpleNamespace(
        model=model,
        _get_all_info_properties=lambda: {"root_body_name": "pelvis", "root_free_joint_xml_name": "root"},
        obs_container=SimpleNamespace(get_randomizable_obs_indices=lambda: np.empty(0, dtype=int)),
    )
    randomizer = DefaultRandomizer(setup)
    randomizer.rand_conf = {
        "apply_velocity_pushes": True,
        "push_interval_range_s": [1.0, 1.0],
        "push_linear_velocity_max": [0.2, 0.1, 0.05],
        "push_angular_velocity_max": [0.3, 0.2, 0.1],
    }
    randomizer._root_qvel_ids = np.arange(2, 8, dtype=np.int32)
    state = DefaultRandomizerState(
        gravity=np.zeros(3),
        geom_friction=np.zeros((0, 3)),
        geom_stiffness=np.zeros(0),
        geom_damping=np.zeros(0),
        base_mass_to_add=0.0,
        com_displacement=np.zeros(3),
        link_mass_multipliers=np.ones(0),
        joint_friction_loss=np.zeros(0),
        joint_damping=np.zeros(0),
        joint_armature=np.zeros(0),
        next_push_step=np.asarray(5, dtype=np.int32),
    )
    carry = _PushCarry(key=None, domain_randomizer_state=state, cur_step_in_episode=5)
    data = SimpleNamespace(qvel=np.zeros(10, dtype=np.float32))
    env = SimpleNamespace(dt=0.01)
    monkeypatch.setattr(np.random, "uniform", lambda low, high: np.asarray(high))
    monkeypatch.setattr(np.random, "randint", lambda low, high: low)

    data, carry = randomizer._apply_velocity_push(env, data, carry, np)

    np.testing.assert_allclose(data.qvel[:2], 0.0)
    np.testing.assert_allclose(data.qvel[2:8], [0.2, 0.1, 0.05, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(data.qvel[8:], 0.0)
    assert int(carry.domain_randomizer_state.next_push_step) == 105


def test_solver_scratch_file_is_removed_when_retargeting_fails(monkeypatch):
    model = SimpleNamespace(
        nq=7,
        qpos0=np.zeros(7),
        njnt=0,
        jnt_limited=np.zeros(0, dtype=bool),
        jnt_qposadr=np.zeros(0, dtype=int),
        jnt_range=np.zeros((0, 2)),
    )
    scene = {
        "object_poses_src": np.zeros((1, 7)),
        "object_poses": np.zeros((1, 7)),
        "object_poses_augmented": np.zeros((1, 7)),
        "object_points_local_demo": np.zeros((1, 3)),
        "object_points_local": np.zeros((1, 3)),
    }
    context = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *_args: None),
        human_joints=np.zeros((1, 1, 3)),
        scene=scene,
        model=model,
        stage=terra_pipeline.SolveStage.CONSTRAINTS_ATTACHED,
    )
    captured = {}

    class FailingRetargeter:
        def retarget_motion(self, **kwargs):
            captured["kwargs"] = kwargs
            captured["scratch"] = Path(kwargs["dest_res_path"])
            raise RuntimeError("simulated solver failure")

    monkeypatch.setattr(
        terra_pipeline,
        "transform_from_human_to_world",
        lambda *_args: (np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(terra_pipeline, "joint_couplers", lambda _model: ())

    with pytest.raises(RuntimeError, match="solver failure"):
        terra_pipeline._solve(context, FailingRetargeter(), np.zeros((1, 2)), robot_dof=0)

    assert not captured["scratch"].exists()
    assert captured["kwargs"]["q_nominal_list"] is None
    assert captured["kwargs"]["original"] is True
    assert captured["kwargs"]["object_poses_augmented"] is scene["object_poses_augmented"]


def test_robot_assets_accept_explicit_public_api_paths(tmp_path):
    smpl_model_path = tmp_path / "smpl-models"
    smpl_model_path.mkdir()
    fitted_shape_path = tmp_path / "shape.npz"
    fitted_shape_path.touch()

    assets = terra_pipeline._resolve_robot_assets(
        "MjxMyoFullBody",
        smpl_model_path,
        fitted_shape_path,
    )

    assert assets.base_env_name == "MyoFullBody"
    assert assets.smpl_model_path == str(smpl_model_path)
    assert assets.fitted_shape_path == str(fitted_shape_path)


def test_robot_assets_reject_unknown_environment_before_loading_paths(tmp_path):
    with pytest.raises(ValueError, match="OmniRetarget not configured"):
        terra_pipeline._resolve_robot_assets("UnknownRobot", tmp_path / "smpl", tmp_path / "shape.npz")


def test_flat_postprocessing_measures_before_alignment_and_assembly(monkeypatch):
    calls = []
    trajectory = object()
    site_names = np.asarray(["site"])
    position_error = np.arange(12, dtype=np.float32).reshape(4, 3)
    ctx = SimpleNamespace(model=object(), on_terrain=False, stage=terra_pipeline.SolveStage.SOLVED)

    def advance(expected, target):
        assert ctx.stage is expected
        ctx.stage = target

    ctx.advance = advance
    retargeter = SimpleNamespace(demo_joints=["Pelvis"])

    monkeypatch.setattr(terra_pipeline.mujoco, "MjData", lambda model: object())
    monkeypatch.setattr(
        terra_pipeline,
        "landmark_error",
        lambda *_args, **_kwargs: calls.append("measure") or position_error,
    )
    monkeypatch.setattr(
        terra_pipeline,
        "align_to_ground",
        lambda *_args, **_kwargs: calls.append("align") or 0.012,
    )
    monkeypatch.setattr(
        terra_pipeline,
        "assemble_trajectory",
        lambda *_args, **_kwargs: calls.append("assemble") or (trajectory, site_names),
    )

    result = terra_pipeline._postprocess_solution(
        ctx,
        object(),
        "MyoFullBody",
        retargeter,
        np.zeros((4, 7)),
        "off",
    )

    assert calls == ["measure", "align", "assemble"]
    assert result.trajectory is trajectory
    assert result.site_names is site_names
    np.testing.assert_array_equal(result.position_error, position_error[1:-1])
    assert result.solver_penetration == pytest.approx(0.012)
