"""Unit tests for the cache-only benchmark table calculations."""

import numpy as np
import pytest

from terra.evaluation.cli import manifest_is_flat, parse_assignment, validate_class_motions
from terra.evaluation.evaluator import (
    Thresholds,
    benchmark_self_penetration,
    common_retargeting_rmse,
    floor_contact_preservation,
    foot_contact_metrics,
    joint_limit_excess_m,
    joint_limit_excess_rad,
    joint_limit_frame_mask,
    load_site_calibration_state,
    load_timeline,
    source_contact_on_output,
    source_floor_contact_on_output,
    source_probe_contact_on_output,
    terrain_contact_preservation,
)
from terra.evaluation.registry import AUTHORITATIVE_METRICS as METRICS
from terra.evaluation.reporting import (
    aggregate,
    aggregate_joint_limit_sensitivity,
    latex_table,
    markdown_table,
    mean_sem,
    mean_std,
    pooled_mean_std_sem,
)
from terra.evaluation.terrain import _declared_output_times, _max_pair_penetration


@pytest.mark.parametrize(
    ("method", "native_fps", "native_frames", "output_frames", "output_bounds"),
    [
        ("omniretarget", 50.0, 89, 178, (0.02, 1.78)),
        ("gmr", 50.0, 89, 178, (0.02, 1.78)),
        ("smpl", 12.5, 21, 168, (0.08, 1.68)),
    ],
)
def test_timeline_reconstructs_dataset_metadata_schema(
    tmp_path,
    method,
    native_fps,
    native_frames,
    output_frames,
    output_bounds,
):
    motion = "External/Subject01/trial"
    source_root = tmp_path / "source"
    source_path = source_root / f"{motion}.npz"
    source_path.parent.mkdir(parents=True)
    np.savez(
        source_path,
        pose_aa=np.zeros((91, 72)),
        trans=np.zeros((91, 3)),
        betas=np.zeros(16),
        gender=np.asarray("neutral"),
        fps=np.asarray(50.0),
    )
    cache = tmp_path / "cache"
    trajectory = cache / "MyoFullBody" / method / f"{motion}.npz"
    trajectory.parent.mkdir(parents=True)
    np.savez(trajectory, qpos=np.zeros((output_frames, 2)), frequency=np.asarray(100.0))
    np.savez(
        trajectory.with_name(trajectory.stem + "_analysis.npz"),
        native_fps=np.asarray(native_fps),
        native_frame_count=np.asarray(native_frames),
        trim_start_frames=np.asarray(1),
        trim_end_frames=np.asarray(1),
        source_path=np.asarray(str(source_path)),
    )
    timeline = load_timeline(cache, method, motion)

    assert timeline.source_frame_count == 91
    assert timeline.source_fps == pytest.approx(50.0)
    assert timeline.native_frame_count == native_frames
    assert timeline.native_fps == pytest.approx(native_fps)
    assert timeline.output_frame_count == output_frames
    assert timeline.output_times()[[0, -1]] == pytest.approx(output_bounds)
    output_times = _declared_output_times(str(trajectory), motion, output_frames, 100.0)
    assert output_times[[0, -1]] == pytest.approx(output_bounds)


@pytest.mark.parametrize("state", [False, True])
def test_benchmark_reads_source_calibration_state_from_analysis(tmp_path, state):
    analysis = tmp_path / "MyoFullBody" / "method" / "motion_analysis.npz"
    analysis.parent.mkdir(parents=True)
    np.savez(
        analysis,
        resolved_config_json=np.asarray(f'{{"calibrate_sites": {str(state).lower()}}}'),
    )

    assert load_site_calibration_state(tmp_path, "method", "motion") is state


def test_missing_analysis_has_unknown_calibration_state(tmp_path):
    assert load_site_calibration_state(tmp_path, "method", "motion") is None


def test_flat_class_rejects_beam_motion(tmp_path):
    manifest = tmp_path / "flat.csv"
    manifest.write_text("motion,terrain_class\nKIT/go_over_beam01_poses,flat\n")
    with pytest.raises(ValueError, match=r"flat class.*contains terrain motion"):
        validate_class_motions("Flat", manifest, ["KIT/go_over_beam01_poses"])


def test_explicit_flat_name_conflict_override_is_narrow_and_opt_in(tmp_path):
    manifest = tmp_path / "flat.csv"
    manifest.write_text("motion,terrain_class\nKIT/go_over_beam01_poses,flat\n")

    validate_class_motions(
        "Flat",
        manifest,
        ["KIT/go_over_beam01_poses"],
        allow_flat_name_conflicts=True,
    )


def test_canonical_rmse_splits_pelvis_world_and_relative_landmark_error():
    import mujoco

    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import SMPLH_TO_MYOFULLBODY

    child_bodies = "".join(f'<body name="{name}"/>' for name in list(SMPLH_TO_MYOFULLBODY.values())[1:])
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody><body name='pelvis'><freejoint/>"
        f"<geom type='sphere' size='.01' mass='1'/>{child_bodies}</body></worldbody></mujoco>"
    )
    qpos = np.zeros((2, model.nq))
    qpos[:, 0] = [10.0, 20.0]
    qpos[:, 3] = 1.0
    raw = qpos.copy()
    source = np.zeros((3, len(SMPLH_DEMO_JOINTS), 3))
    source[:, :, 0] = np.array([0.0, 10.0, 20.0])[:, None]

    pelvis_world, pelvis_relative, per_landmark = common_retargeting_rmse(
        model,
        qpos,
        output_times=np.array([0.0, 1.0]),
        source_joints=source,
        source_fps=1.0,
        active=np.array([True, True]),
    )

    # A one-frame best shift would make this zero. Fixed declared timestamps retain the
    # 10 m global error, while the pelvis-relative diagnostic removes it.
    assert pelvis_world == pytest.approx(10_000.0)
    assert pelvis_relative == pytest.approx(0.0)
    assert set(per_landmark) == set(SMPLH_TO_MYOFULLBODY)
    assert np.array_equal(qpos, raw)


def test_pelvis_world_rmse_does_not_average_other_landmarks_and_relative_excludes_pelvis():
    import mujoco

    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import SMPLH_TO_MYOFULLBODY

    child_bodies = "".join(f'<body name="{name}"/>' for name in list(SMPLH_TO_MYOFULLBODY.values())[1:])
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody><body name='pelvis'><freejoint/>"
        f"<geom type='sphere' size='.01' mass='1'/>{child_bodies}</body></worldbody></mujoco>"
    )
    qpos = np.zeros((2, model.nq))
    qpos[:, 0] = 10.0
    qpos[:, 3] = 1.0
    source = np.zeros((2, len(SMPLH_DEMO_JOINTS), 3))
    source[:, SMPLH_DEMO_JOINTS.index("Pelvis"), 0] = 9.0

    pelvis_world, pelvis_relative, _ = common_retargeting_rmse(
        model,
        qpos,
        output_times=np.array([0.0, 1.0]),
        source_joints=source,
        source_fps=1.0,
        active=np.array([True, True]),
    )

    # Pelvis error is 1 m. Every other landmark has 9 m of pelvis-relative error; the
    # aligned pelvis itself is excluded instead of diluting that value with a zero.
    assert pelvis_world == pytest.approx(1_000.0)
    assert pelvis_relative == pytest.approx(9_000.0)


def test_benchmark_metrics_report_the_two_explicit_fidelity_quantities():
    keys = {metric.key for metric in METRICS}
    assert "pelvis_world_rmse_mm" in keys
    assert "pelvis_relative_landmark_rmse_mm" in keys
    assert "retargeting_rmse_mm" not in keys
    assert "joint_limit_max_excess_deg" in keys


def test_joint_limit_metric_ignores_unlimited_joints_and_exposes_severity():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body>
              <joint name="unlimited" type="hinge" limited="false"/>
              <joint name="limited" type="hinge" limited="true" range="-1 1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.zeros((3, model.nq))
    qpos[:, model.joint("unlimited").qposadr[0]] = 10.0
    limited = model.joint("limited").qposadr[0]
    qpos[:, limited] = [0.0, 1.0005, 1.02]

    excess = joint_limit_excess_rad(model, qpos)

    np.testing.assert_allclose(excess, [0.0, 0.0005, 0.02])
    assert joint_limit_frame_mask(model, qpos, 1e-5).tolist() == [False, True, True]
    assert joint_limit_frame_mask(model, qpos, 1e-3).tolist() == [False, False, True]


def test_joint_limit_metric_keeps_hinge_radians_and_slide_metres_separate():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body>
              <joint name="hinge" type="hinge" limited="true" range="-1 1"/>
              <joint name="slide" type="slide" limited="true" range="-0.1 0.1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.zeros((2, model.nq))
    qpos[1, model.joint("hinge").qposadr[0]] = 1.01
    qpos[1, model.joint("slide").qposadr[0]] = 0.102

    np.testing.assert_allclose(joint_limit_excess_rad(model, qpos), [0.0, 0.01])
    np.testing.assert_allclose(joint_limit_excess_m(model, qpos), [0.0, 0.002])
    assert joint_limit_frame_mask(model, qpos, 0.02, linear_tol_m=0.001).tolist() == [False, True]


def test_joint_limit_sensitivity_keeps_numerical_and_material_thresholds_separate():
    methods = [("GMR", "gmr")]
    classes = [("Ramps", None)]
    row = {"method": "GMR", "motion_class": "Ramps", "error": ""}
    row.update(
        {
            "joint_limit_duration_pct_at_1em05_rad": 80.0,
            "joint_limit_duration_pct_at_1em04_rad": 40.0,
            "joint_limit_duration_pct_at_1em03_rad": 10.0,
            "joint_limit_duration_pct_at_1em02_rad": 0.0,
        }
    )

    sensitivity = aggregate_joint_limit_sensitivity([row], methods, classes)

    assert [result["duration_pct_mean"] for result in sensitivity] == [80.0, 40.0, 10.0, 0.0]
    assert all(result["n_motions"] == 1 for result in sensitivity)


def test_foot_skating_uses_desired_sticking_not_retargeted_contact():
    thresholds = Thresholds(skating_speed_m_s=0.1)
    clearance = np.zeros((5, 2))
    clearance[3, 0] = 0.030  # floating left foot
    clearance[0, 1] = 0.030  # right foot is still in swing before touchdown
    clearance[2, 1] = -0.002  # penetrating right foot
    source_contact = np.array(
        [
            [True, False],
            [True, True],
            [True, True],
            [True, False],
            [True, False],
        ]
    )
    position = np.zeros((5, 2, 3))
    position[:, 0, 0] = [0.000, 0.001, 0.021, 0.041, 0.042]
    # This is a 2 m/s displacement while the source says the right foot should stick. The
    # retargeted clearance does not suppress it: floating/contact are separate metrics.
    position[:, 1, 0] = [-0.200, 0.000, 0.000, 0.000, 0.000]

    result = foot_contact_metrics(clearance, position, source_contact, fps=10.0, thresholds=thresholds)

    assert result["source_contact_s"] == pytest.approx(0.5)
    assert "contact_preservation_pct" not in result
    # Floating asks whether the corresponding source-required foot is clear. The planted
    # right foot must not hide the floating left foot.
    assert result["source_support_s"] == pytest.approx(0.7)
    assert result["floating_duration_pct"] == pytest.approx(100.0 / 7.0)
    assert result["floating_max_height_mm"] == pytest.approx(30.0)
    assert result["support_penetration_duration_pct"] == 0.0
    assert result["support_penetration_max_depth_mm"] == pytest.approx(2.0)
    assert result["invalid_support_duration_pct"] == pytest.approx(100.0 / 7.0)
    assert result["skating_duration_pct"] == pytest.approx(60.0)
    assert result["skating_max_velocity_m_s"] == pytest.approx(0.8)
    assert result["skating_frame_velocity_n"] == 3
    assert result["skating_frame_velocity_sum_m_s"] == pytest.approx(2.4)


def test_foot_metrics_without_source_contact_are_missing_not_artificial_zeros():
    result = foot_contact_metrics(
        np.zeros((3, 2)),
        np.zeros((3, 2, 3)),
        np.zeros((3, 2), dtype=bool),
        fps=100.0,
        thresholds=Thresholds(),
    )
    assert result["source_contact_s"] == 0
    assert np.isnan(result["skating_duration_pct"])
    assert np.isnan(result["skating_max_velocity_m_s"])
    assert np.isnan(result["floating_duration_pct"])
    assert np.isnan(result["floating_max_height_mm"])
    assert np.isnan(result["support_penetration_duration_pct"])
    assert np.isnan(result["support_penetration_max_depth_mm"])
    assert np.isnan(result["invalid_support_duration_pct"])


def test_skating_velocity_pool_contains_only_upstream_defined_skating_frames():
    positions = np.zeros((3, 2, 3))
    positions[:, 0, 0] = [0.0, 0.004, 0.009]
    source_contact = np.array([[True, False]] * 3)

    result = foot_contact_metrics(
        np.zeros((3, 2)),
        positions,
        source_contact,
        fps=10.0,
        thresholds=Thresholds(skating_speed_m_s=0.3),
    )

    assert result["skating_duration_pct"] == 0.0
    assert result["skating_max_velocity_m_s"] == 0.0
    assert result["skating_frame_velocity_n"] == 0


def test_skating_uses_physical_velocity_and_practical_threshold_at_any_fps():
    def score(fps: float) -> dict:
        positions = np.zeros((20, 2, 3))
        positions[:, 0, 0] = np.arange(20) * 0.40 / fps
        source_contact = np.zeros((20, 2), dtype=bool)
        source_contact[:, 0] = True
        return foot_contact_metrics(
            np.zeros((20, 2)),
            positions,
            source_contact,
            fps=fps,
            thresholds=Thresholds(),
        )

    at_50_hz = score(50.0)
    at_100_hz = score(100.0)

    assert at_50_hz["skating_duration_pct"] == pytest.approx(97.5)
    assert at_100_hz["skating_duration_pct"] == pytest.approx(97.5)
    assert at_50_hz["skating_max_velocity_m_s"] == pytest.approx(0.40)
    assert at_100_hz["skating_max_velocity_m_s"] == pytest.approx(0.40)


def test_skating_matches_each_source_probe_to_the_corresponding_robot_point():
    fps = 10.0
    positions = np.zeros((4, 4, 3))
    positions[:, 0, 0] = np.arange(4) * 0.40 / fps  # moving left toe
    source_probes = np.zeros((4, 4), dtype=bool)
    source_probes[:, 2] = True  # only the stationary left ankle is desired to stick
    source_support = np.zeros((4, 2), dtype=bool)
    source_support[:, 0] = True

    result = foot_contact_metrics(
        np.zeros((4, 2)),
        positions,
        source_probes,
        fps,
        Thresholds(),
        source_support=source_support,
    )

    assert result["skating_duration_pct"] == 0.0
    assert result["skating_max_velocity_m_s"] == 0.0

    positions[:, 2, 0] = np.arange(4) * 0.40 / fps
    result = foot_contact_metrics(
        np.zeros((4, 2)),
        positions,
        source_probes,
        fps,
        Thresholds(),
        source_support=source_support,
    )
    # The common interval extends one frame past the final timestamp, so its
    # Voronoi cell represents 1.5 frame intervals and the initial cell 0.5.
    assert result["skating_duration_pct"] == pytest.approx(87.5)
    assert result["skating_max_velocity_m_s"] == pytest.approx(0.40)


def test_valid_source_flight_is_not_floating_and_support_uses_corresponding_foot():
    clearance = np.array(
        [
            [0.000, 0.000],
            [0.300, 0.250],  # intended source flight
            [0.030, 0.000],  # left stance floats while the other foot touches
            [0.000, 0.000],
        ]
    )
    source_sticking = np.ones((4, 2), dtype=bool)
    source_support = np.array(
        [
            [True, True],
            [False, False],
            [True, False],
            [True, True],
        ]
    )

    result = foot_contact_metrics(
        clearance,
        np.zeros((4, 2, 3)),
        source_sticking,
        fps=10.0,
        thresholds=Thresholds(),
        source_support=source_support,
    )

    assert result["floating_duration_pct"] == pytest.approx(20.0)
    assert result["floating_max_height_mm"] == pytest.approx(30.0)


def test_support_state_partitions_floating_penetrating_and_valid_contact():
    clearance = np.array(
        [
            [0.000, 0.000],
            [0.030, 0.000],  # floating
            [-0.006, 0.000],  # penetrating
            [-0.004, 0.000],  # inside the tolerance band
            [0.010, 0.000],
        ]
    )
    source_support = np.array([[True, False]] * len(clearance))
    result = foot_contact_metrics(
        clearance,
        np.zeros((len(clearance), 2, 3)),
        source_support,
        fps=10.0,
        thresholds=Thresholds(),
    )

    assert result["source_support_s"] == pytest.approx(0.5)
    assert result["floating_duration_pct"] == pytest.approx(20.0)
    assert result["support_penetration_duration_pct"] == pytest.approx(20.0)
    assert result["invalid_support_duration_pct"] == pytest.approx(40.0)
    assert result["support_penetration_max_depth_mm"] == pytest.approx(6.0)


def test_source_sticking_threshold_is_in_metres_per_second_at_any_fps():
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS

    def mask(fps):
        joints = np.zeros((20, len(SMPLH_DEMO_JOINTS), 3))
        x = np.arange(20) * 0.1 / fps
        for joint in ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle"):
            joints[:, SMPLH_DEMO_JOINTS.index(joint), 0] = x
        return source_contact_on_output(joints, fps, np.arange(20) / fps)

    assert mask(100.0).all()
    assert mask(50.0).all()


def test_source_contact_keeps_toe_and_ankle_probes_separate_before_side_union():
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS

    fps = 10.0
    joints = np.zeros((20, len(SMPLH_DEMO_JOINTS), 3))
    joints[:, SMPLH_DEMO_JOINTS.index("L_Toe"), 0] = np.arange(20) * 0.1
    output_times = np.arange(20) / fps

    probes = source_probe_contact_on_output(joints, fps, output_times)
    support = source_contact_on_output(joints, fps, output_times)

    assert not probes[:, 0].any()
    assert probes[:, 2].all()
    assert support[:, 0].all()


def test_terrain_contact_preservation_uses_desired_point_time_and_no_contact_is_missing():
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec

    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0, 0, 0.5), size=(0.5, 0.5, 0.5)),))
    source = np.array(
        [
            [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]],
        ]
    )
    robot_distance = np.array([[0.0, np.inf], [np.inf, 0.0]])
    result = terrain_contact_preservation(
        source,
        robot_distance,
        terrain,
        0.1,
        np.array([0.0, 1.0]),
        (0.0, 1.0),
    )
    assert result["desired_contact_point_s"] == pytest.approx(1.0)
    assert result["contact_preservation_pct"] == pytest.approx(50.0)

    flat = TerrainSpec()
    missing = terrain_contact_preservation(
        source,
        robot_distance,
        flat,
        0.1,
        np.array([0.0, 1.0]),
        (0.0, 1.0),
    )
    assert missing["desired_contact_point_s"] == 0.0
    assert missing["contact_preservation_pct"] == 100.0


def test_terrain_contact_preservation_matches_upstream_contact_independent_of_penetration():
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec

    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0, 0, 0.5), size=(0.5, 0.5, 0.5)),))
    source = np.array([[[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]])
    robot_distance = np.array([[-0.020, 0.0]])

    result = terrain_contact_preservation(
        source,
        robot_distance,
        terrain,
        0.1,
        np.array([0.0]),
        (0.0, 1.0),
    )

    assert result["desired_contact_point_s"] == pytest.approx(1.0)
    assert result["contact_preservation_pct"] == 100.0


def test_flat_floor_contact_preservation_retains_local_foot_time_metric():
    source_contact = np.array(
        [
            [True, False],
            [True, True],
        ]
    )
    clearance = np.array(
        [
            [0.000, 0.000],
            [0.030, -0.005],
        ]
    )
    result = floor_contact_preservation(
        clearance,
        source_contact,
        Thresholds(),
        np.array([0.0, 1.0]),
        (0.0, 1.0),
    )

    assert result["desired_contact_point_s"] == pytest.approx(1.5)
    assert result["contact_preservation_pct"] == pytest.approx(100.0 * 2 / 3)


def test_flat_floor_source_contact_excludes_a_slow_midair_pause():
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS

    joints = np.zeros((10, len(SMPLH_DEMO_JOINTS), 3))
    toe = SMPLH_DEMO_JOINTS.index("L_Toe")
    ankle = SMPLH_DEMO_JOINTS.index("L_Ankle")
    joints[:5, toe, 2] = 0.0
    joints[5:, toe, 2] = 1.0
    joints[:5, ankle, 2] = 0.0
    joints[5:, ankle, 2] = 1.0
    output_times = np.arange(10, dtype=float) / 10.0

    support_contact = source_contact_on_output(joints, 10.0, output_times, speed_m_s=0.3)
    floor_contact = source_floor_contact_on_output(joints, 10.0, output_times)

    assert np.array_equal(support_contact, floor_contact)
    assert not floor_contact[5:, 0].any()


def test_foot_metric_shapes_and_frequency_are_validated():
    with pytest.raises(ValueError, match="matching"):
        foot_contact_metrics(
            np.zeros((3, 2)),
            np.zeros((3, 2, 3)),
            np.zeros((4, 2), dtype=bool),
            100.0,
            Thresholds(),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        foot_contact_metrics(
            np.zeros((3, 2)),
            np.zeros((3, 2, 3)),
            np.zeros((3, 2), dtype=bool),
            0.0,
            Thresholds(),
        )
    with pytest.raises(ValueError, match="source_support"):
        foot_contact_metrics(
            np.zeros((3, 2)),
            np.zeros((3, 2, 3)),
            np.zeros((3, 2), dtype=bool),
            100.0,
            Thresholds(),
            source_support=np.zeros((4, 2), dtype=bool),
        )


def test_mean_sem_is_sample_sem_and_ignores_missing_values():
    mean, sem, n = mean_sem([1.0, 3.0, np.nan])
    assert (mean, n) == pytest.approx((2.0, 2))
    assert sem == pytest.approx(1.0)
    assert mean_sem([7.0]) == pytest.approx((7.0, 0.0, 1))
    empty = mean_sem([np.nan])
    assert np.isnan(empty[0]) and np.isnan(empty[1]) and empty[2] == 0


def test_mean_std_matches_upstream_population_standard_deviation():
    assert mean_std([1.0, 3.0, np.nan]) == pytest.approx((2.0, 1.0, 2))
    assert mean_std([7.0]) == pytest.approx((7.0, 0.0, 1))


def test_omniretarget_frame_observations_pool_from_sufficient_statistics():
    rows = [
        {"sum": 3.0, "square": 5.0, "count": 2},
        {"sum": 3.0, "square": 9.0, "count": 1},
    ]

    mean, std, sem, count = pooled_mean_std_sem(rows, ("sum", "square", "count"))

    assert count == 3
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert sem == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3))


def test_thresholds_reject_nonfinite_negative_and_zero_scale_values():
    defaults = Thresholds()
    assert defaults.penetration_m == pytest.approx(0.010)
    assert defaults.source_contact_speed_m_s == pytest.approx(0.300)
    assert defaults.skating_speed_m_s == pytest.approx(0.300)
    assert defaults.terrain_contact_m == pytest.approx(0.100)
    with pytest.raises(ValueError, match="penetration_m"):
        Thresholds(penetration_m=-1.0).validate()
    with pytest.raises(ValueError, match="self_collision_m"):
        Thresholds(self_collision_m=np.inf).validate()
    with pytest.raises(ValueError, match="foot_contact_height_m"):
        Thresholds(foot_contact_height_m=0.0).validate()
    with pytest.raises(ValueError, match="skating_speed_m_s"):
        Thresholds(skating_speed_m_s=0.0).validate()
    with pytest.raises(ValueError, match="terrain_contact_m"):
        Thresholds(terrain_contact_m=0.0).validate()


def test_benchmark_self_collision_excludes_all_body_contact_diagnostic():
    class Measurement:
        def __init__(self):
            self.per_frame = {
                "interleg_selfpen": np.array([0.0, 0.002, 0.0]),
                # Humerus/thorax and intended forearm/thigh contact can dominate this broader
                # diagnostic even when the contralateral legs are clean.
                "all_selfpen": np.array([0.020, 0.020, 0.020]),
            }

    assert np.array_equal(
        benchmark_self_penetration(Measurement()),
        np.array([0.0, 0.002, 0.0]),
    )
    assert next(m for m in METRICS if m.key == "self_collision_duration_pct").label == ("Inter-leg collision duration")


def test_interleg_penetration_uses_exact_distance_even_without_generated_contact():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="left" pos="0 0 0">
              <freejoint/>
              <geom name="left_geom" type="sphere" size="0.1" contype="1" conaffinity="1"/>
            </body>
            <body name="right" pos="0.15 0 0">
              <geom name="right_geom" type="sphere" size="0.1" contype="2" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left = model.geom("left_geom").id
    right = model.geom("right_geom").id

    assert data.ncon == 0
    assert _max_pair_penetration(model, data, [(left, right, "left/right")]) == pytest.approx(0.05)


def test_method_assignment_allows_empty_smpl_cache_subdirectory():
    assert parse_assignment("SMPL=", "--method") == ("SMPL", "")
    assert parse_assignment("TERRA=terra", "--method") == ("TERRA", "terra")


def test_flat_class_comes_from_manifest_metadata_not_only_display_name(tmp_path):
    manifest = tmp_path / "motions.csv"
    manifest.write_text("motion,terrain_class\nKIT/example,flat\n")
    assert manifest_is_flat("Level ground", manifest)

    manifest.write_text("motion,terrain_class\nKIT/example,ramp_up\n")
    assert not manifest_is_flat("Flat", manifest)


def test_summary_and_split_tables_group_methods_by_motion_class():
    methods = [("TERRA", "terra"), ("GMR", "gmr")]
    classes = [("Flat", None), ("Ramps", None)]
    rows = []
    for method, _ in methods:
        for motion_class, _ in classes:
            row = {
                "method": method,
                "motion_class": motion_class,
                "error": "",
            }
            row.update({metric.key: 1.0 for metric in METRICS})
            rows.append(row)

    summary = aggregate(rows, methods, classes)
    assert [(row["motion_class"], row["method"]) for row in summary] == [
        ("Flat", "TERRA"),
        ("Flat", "GMR"),
        ("Ramps", "TERRA"),
        ("Ramps", "GMR"),
    ]

    markdown = markdown_table(summary)
    first_markdown, second_markdown = markdown.split("### Table 2:")
    assert markdown.count("### Table ") == 2
    assert first_markdown.index("| **Flat** | TERRA") < first_markdown.index("|  | GMR")
    assert "Contact preservation (%)" in first_markdown
    assert "Joint-limit duration (%)" not in first_markdown
    assert "Joint-limit duration (%)" in second_markdown

    latex = latex_table(summary)
    first_latex, second_latex = latex.split(r"\par\medskip")
    assert latex.count(r"\begin{tabular}") == 2
    assert first_latex.index(r"Flat & TERRA") < first_latex.index(r" & GMR")
    assert "Contact preservation" in first_latex
    assert "Joint-limit duration" not in first_latex
    assert "Joint-limit duration" in second_latex


def test_summary_keeps_valid_motions_when_one_metric_value_is_unavailable():
    methods = [("GMR", "gmr")]
    classes = [("Flat", None)]
    rows = []
    for value in ("", 4.0):
        row = {"method": "GMR", "motion_class": "Flat", "error": ""}
        row.update({metric.key: 1.0 for metric in METRICS})
        row["contact_preservation_pct"] = value
        rows.append(row)

    summary = aggregate(rows, methods, classes)[0]
    assert summary["n_motions"] == 2
    assert summary["n_errors"] == 0
    assert summary["contact_preservation_pct_mean"] == pytest.approx(4.0)
    assert summary["contact_preservation_pct_n"] == 1
