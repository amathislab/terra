"""Focused contract and formula tests for the package-owned unified evaluator."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from terra.evaluation.annotations import (
    RENDER_ARRAY_SPECS,
    AnnotationError,
    frame_metadata,
)
from terra.evaluation.cli import method_counts, read_common_intervals
from terra.evaluation.evaluator import (
    Thresholds,
    coupler_metrics,
    evaluate_method_motion,
    frame_step_metrics,
    model,
    phase_metrics,
    solver_throughput,
)
from terra.evaluation.registry import (
    AUTHORITATIVE_METRICS,
    METRIC_REGISTRY,
    REQUIRED_FAMILIES,
    MetricDefinitionConflict,
    build_registry,
)
from terra.evaluation.reporting import aggregate
from terra.evaluation.terrain import Measurement, Stance, Swing, Tolerances, source_shape_path
from terra.visualization.render import Flags


def test_registry_covers_the_authoritative_metric_union_with_declared_lineage():
    families = {spec.family for spec in AUTHORITATIVE_METRICS}

    assert REQUIRED_FAMILIES <= families
    assert set(METRIC_REGISTRY) == {spec.key for spec in AUTHORITATIVE_METRICS}
    assert all(spec.formula and spec.denominator and spec.source for spec in AUTHORITATIVE_METRICS)


def test_evaluation_source_shape_uses_the_selected_cache_root(tmp_path):
    assert source_shape_path("MyoFullBody", tmp_path) == tmp_path / "MyoFullBody" / "shape_optimized.pkl"


def test_evaluator_uses_the_exact_finger_disabled_environment_model():
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    metric_model = model()
    environment_model = MyoFullBody(disable_fingers=True).model

    def names(instance, object_type, count):
        return tuple(mujoco.mj_id2name(instance, object_type, index) for index in range(count))

    assert (metric_model.nq, metric_model.nu, metric_model.ntendon) == (89, 354, 362)
    assert names(metric_model, mujoco.mjtObj.mjOBJ_ACTUATOR, metric_model.nu) == names(
        environment_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        environment_model.nu,
    )
    assert names(metric_model, mujoco.mjtObj.mjOBJ_TENDON, metric_model.ntendon) == names(
        environment_model,
        mujoco.mjtObj.mjOBJ_TENDON,
        environment_model.ntendon,
    )


def test_frozen_common_intervals_require_consistent_per_motion_windows(tmp_path):
    intervals = tmp_path / "per_motion.csv"
    intervals.write_text(
        "motion,method,common_start_s,common_end_s\n"
        "Study/A,smpl,0.08,2.16\n"
        "Study/A,gmr,0.08,2.16\n"
        "Study/B,smpl,,\n"
        "Study/B,gmr,0.10,1.90\n"
    )

    assert read_common_intervals(intervals) == {
        "Study/A": (0.08, 2.16),
        "Study/B": (0.10, 1.90),
    }

    intervals.write_text("motion,method,common_start_s,common_end_s\nStudy/A,smpl,0.08,2.16\nStudy/A,gmr,0.09,2.16\n")
    with pytest.raises(ValueError, match="inconsistent common intervals"):
        read_common_intervals(intervals)


def test_registry_rejects_conflicting_definitions():
    original = AUTHORITATIVE_METRICS[0]
    conflicting = replace(original, formula="a scientifically different formula")

    with pytest.raises(MetricDefinitionConflict, match=original.key):
        build_registry((original, conflicting))


def test_solver_throughput_emits_t_frame_per_solved_frame(tmp_path):
    analysis = tmp_path / "motion_analysis.npz"
    np.savez(
        analysis,
        benchmark_retarget_elapsed_s=np.asarray(5.0),
        benchmark_solved_frames=np.asarray(20),
        pos_error=np.zeros((20, 3)),
        benchmark_timing_context=np.asarray(
            '__terra_json__:{"cpu_request":8,"threads_per_worker":1,"worker_processes":8}'
        ),
    )

    result = solver_throughput(analysis, common_duration_s=2.0)

    assert result["solver_native_fps"] == pytest.approx(4.0)
    assert result["solver_seconds_per_motion_second"] == pytest.approx(2.5)
    assert result["t_frame_s"] == pytest.approx(0.25)
    assert result["solver_solved_frames"] == 20
    assert result["timing_source"] == "uniform_retarget_call"
    assert json.loads(result["timing_context_json"]) == {
        "cpu_request": 8,
        "threads_per_worker": 1,
        "worker_processes": 8,
    }


def test_solver_throughput_uses_legacy_producer_timing_without_retargeting(tmp_path):
    analysis = tmp_path / "motion_analysis.npz"
    np.savez(
        analysis,
        retarget_fps=np.asarray(5.0),
        pos_error=np.zeros((20, 3)),
    )

    result = solver_throughput(analysis, common_duration_s=2.0)

    assert result["solver_native_fps"] == pytest.approx(5.0)
    assert result["solver_seconds_per_motion_second"] == pytest.approx(2.0)
    assert result["t_frame_s"] == pytest.approx(0.2)
    assert result["solver_solved_frames"] == 20
    assert result["timing_source"] == "producer_retarget_fps"
    assert result["timing_context_json"] == ""


def test_aggregation_uses_successful_motions_and_a_finite_denominator_per_metric():
    methods = [("TERRA", "terra")]
    classes = [("Steps", Path("steps.csv"))]
    rows = [
        {
            "method": "TERRA",
            "motion_class": "Steps",
            "motion": "a",
            "error": "",
            "contact_preservation_pct": float("nan"),
            "floating_duration_pct": 10.0,
            "t_frame_s": 0.1,
        },
        {
            "method": "TERRA",
            "motion_class": "Steps",
            "motion": "b",
            "error": "",
            "contact_preservation_pct": 80.0,
            "floating_duration_pct": 20.0,
            "t_frame_s": 0.3,
        },
        {
            "method": "TERRA",
            "motion_class": "Steps",
            "motion": "failed",
            "error": "missing trajectory",
            "contact_preservation_pct": 100.0,
            "floating_duration_pct": 100.0,
            "t_frame_s": 0.01,
        },
    ]

    summary = aggregate(rows, methods, classes)[0]

    assert summary["n_motions"] == 2
    assert summary["n_errors"] == 1
    assert summary["contact_preservation_pct_mean"] == pytest.approx(80.0)
    assert summary["contact_preservation_pct_n"] == 1
    assert summary["floating_duration_pct_mean"] == pytest.approx(15.0)
    assert summary["floating_duration_pct_n"] == 2
    assert summary["t_frame_s_mean"] == pytest.approx(0.2)
    assert summary["t_frame_s_n"] == 2

    assert method_counts(rows, methods) == [{"method": "TERRA", "successful": 2, "failed": 1, "total": 3}]


def test_phase_metrics_use_only_complete_common_interval_phases_and_separate_swing_questions():
    measurement = Measurement("motion", 12, 1, Tolerances())
    measurement.stances = [
        Stance("left", 1, 4, 0.0, 0.0, 0.0, 0.05, 0.0, slip_end=3),
        Stance("right", 8, 11, 0.0, 0.0, 0.0, 0.05, 0.0, slip_end=10),
    ]
    measurement.swings = [
        Swing("left", 1, 4, peak=0.001, src_peak=0.080, touched=-0.010),
        # An invalid source clearance is excluded from relative clearance, but its exact
        # sole penetration remains measurable as scraping.
        Swing("right", 4, 7, peak=-0.010, src_peak=-0.020, touched=-0.010),
        Swing("left", 8, 11, peak=0.080, src_peak=0.080, touched=0.010),
    ]
    active = np.zeros(12, dtype=bool)
    active[1:7] = True

    values, annotations = phase_metrics(measurement, active)

    assert values["stance_slip_failure_pct"] == pytest.approx(100.0)
    assert values["swing_clearance_failure_pct"] == pytest.approx(100.0)
    assert values["swing_scrape_failure_pct"] == pytest.approx(100.0)
    assert values["swing_clearance_ratio_median"] == pytest.approx(0.001 / 0.080)
    assert annotations["stance_slip_failure"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert annotations["swing_clearance_failure"][1:4].all()
    assert annotations["swing_scraping"][1:7].all()
    assert not annotations["swing_scraping"][8:11].any()


def test_coupler_metrics_use_the_declared_polynomial_residual(monkeypatch):
    import terra._musclemimic as musclemimic_adapter

    metric_model = SimpleNamespace(
        njnt=2,
        jnt_qposadr=np.array([0, 1]),
        joint=lambda joint_id: SimpleNamespace(name=("independent", "dependent")[joint_id]),
    )
    monkeypatch.setattr(
        musclemimic_adapter,
        "joint_couplers",
        lambda _model: ((1, 0, (0.0, 2.0)),),
    )
    qpos = np.array([[1.0, 2.0], [1.0, 3.0], [99.0, -99.0]])

    values = coupler_metrics(metric_model, qpos, np.array([True, True, False]))

    assert values["coupler_mean_residual_deg"] == pytest.approx(np.degrees(0.5))
    assert values["coupler_max_residual_deg"] == pytest.approx(np.degrees(1.0))
    assert values["coupler_worst"] == "dependent"
    assert values["n_couplers"] == 1


def test_frame_step_metrics_select_hinges_and_common_transitions(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    import musclemimic.utils.retarget.msk_metrics as msk_metrics

    metric_model = SimpleNamespace(
        njnt=2,
        jnt_qposadr=np.array([0, 1]),
        joint=lambda joint_id: SimpleNamespace(
            name=("hip", "knee")[joint_id],
            type=mujoco.mjtJoint.mjJNT_HINGE,
        ),
    )
    monkeypatch.setattr(
        msk_metrics,
        "root_step_series",
        lambda _model, _qpos: (np.array([0.0, 2.0, 4.0]), np.array([0.0, 3.0, 6.0])),
    )
    qpos = np.radians(np.array([[0.0, 0.0], [10.0, 1.0], [20.0, 31.0]]))

    values, annotations = frame_step_metrics(
        metric_model,
        qpos,
        np.array([False, True, True]),
    )

    assert values["joint_step_max_deg"] == pytest.approx(30.0)
    assert values["joint_step_worst"] == "knee"
    assert values["joint_step_frame"] == 2
    assert values["root_translation_step_max_mm"] == pytest.approx(4.0)
    assert values["root_translation_step_p999_mm"] == pytest.approx(np.percentile([2.0, 4.0], 99.9))
    assert values["root_rotation_step_max_deg"] == pytest.approx(6.0)
    assert values["root_rotation_step_p999_deg"] == pytest.approx(np.percentile([3.0, 6.0], 99.9))
    assert annotations["joint_step_deg"] == pytest.approx([0.0, 10.0, 30.0])


def test_one_method_motion_measurement_populates_all_views_from_one_terrain_pass(tmp_path, monkeypatch):
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    import terra.evaluation.evaluator as evaluator
    import terra.source as source_module
    from musclemimic.utils.retarget.benchmark_timeline import MotionTimeline

    motion = "Study/Subject/Trial"
    monkeypatch.setenv("CONVERTED_AMASS_PATH", "must-not-change")
    trajectory = tmp_path / "MyoFullBody" / "method" / f"{motion}.npz"
    trajectory.parent.mkdir(parents=True)
    np.savez(trajectory, qpos=np.zeros((3, 1)), frequency=np.asarray(10.0))
    timeline = MotionTimeline(0.0, 0.2, 10.0, 3, 0.0, 0.2, 10.0, 3, 10.0, 3, 0, 0)

    per_frame = {
        "body_pen": np.zeros(3),
        "contact_gap_left": np.zeros(3, dtype=bool),
        "contact_gap_right": np.zeros(3, dtype=bool),
        "interleg_selfpen": np.zeros(3),
        "stance_left": np.ones(3, dtype=bool),
        "stance_right": np.zeros(3, dtype=bool),
        "support_left": np.zeros(3),
        "support_right": np.zeros(3),
        "vert_left": np.zeros(3),
        "vert_right": np.zeros(3),
    }

    class TerrainMeasurement:
        n_frames = 3
        fps = 10.0
        tol = Tolerances()
        stances = ()
        swings = ()

        def __init__(self):
            self.per_frame = per_frame

        def summary(self):
            return {"passed": 1, "n_fails": 0}

    count = 0

    def one_measure(*_args, **_kwargs):
        nonlocal count
        count += 1
        return TerrainMeasurement()

    monkeypatch.setattr(evaluator, "measure", one_measure)
    monkeypatch.setattr(evaluator, "to_json", lambda _measurement: {"source": "shared pass"})
    monkeypatch.setattr(evaluator, "load_timeline", lambda *_args, **_kwargs: timeline)
    monkeypatch.setattr(evaluator, "load_site_calibration_state", lambda *_args: None)
    monkeypatch.setattr(evaluator, "load_source_motion", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(evaluator, "model", lambda: SimpleNamespace(nq=1))
    joint_call = {}

    def source_joints(*_args, **kwargs):
        joint_call.update(kwargs)
        return np.zeros((3, len(SMPLH_DEMO_JOINTS), 3)), 10.0

    monkeypatch.setattr(source_module, "motion_world_joints", source_joints)
    monkeypatch.setattr(evaluator, "source_contact_on_output", lambda *_args, **_kwargs: np.zeros((3, 2), bool))
    monkeypatch.setattr(evaluator, "source_floor_contact_on_output", lambda *_args, **_kwargs: np.zeros((3, 2), bool))
    monkeypatch.setattr(evaluator, "robot_point_positions", lambda *_args, **_kwargs: np.zeros((3, 2, 3)))
    foot_values = {
        "source_contact_s": 0.1,
        "source_support_s": 0.1,
        "skating_duration_pct": 0.0,
        "skating_max_velocity_m_s": 0.0,
        "floating_duration_pct": 0.0,
        "floating_max_height_mm": 0.0,
        "support_penetration_duration_pct": 0.0,
        "support_penetration_max_depth_mm": 0.0,
        "invalid_support_duration_pct": 0.0,
    }
    foot_annotations = {
        "skating": np.zeros(3, bool),
        "skating_per_foot": np.zeros((3, 2), bool),
        "support_floating": np.zeros((3, 2), bool),
        "support_penetrating": np.zeros((3, 2), bool),
        "invalid_support": np.zeros((3, 2), bool),
        "foot_velocity_m_s": np.zeros((3, 2)),
    }
    monkeypatch.setattr(
        evaluator,
        "foot_contact_metrics",
        lambda *_args, **_kwargs: (foot_values, foot_annotations),
    )
    monkeypatch.setattr(
        evaluator,
        "floor_contact_preservation",
        lambda *_args, **_kwargs: {"desired_contact_point_s": 0.1, "contact_preservation_pct": 100.0},
    )
    monkeypatch.setattr(evaluator, "benchmark_self_penetration", lambda _measurement: np.zeros(3))
    monkeypatch.setattr(evaluator, "joint_limit_excess_rad", lambda *_args: np.zeros(3))
    monkeypatch.setattr(evaluator, "joint_limit_excess_m", lambda *_args: np.zeros(3))
    monkeypatch.setattr(evaluator, "tendon_discontinuity_series", lambda *_args: (np.zeros(3), np.zeros(3)))
    monkeypatch.setattr(
        evaluator,
        "frame_step_metrics",
        lambda *_args: (
            {
                "joint_step_max_deg": 0.0,
                "joint_step_p999_deg": 0.0,
                "joint_step_worst": None,
                "joint_step_frame": -1,
                "root_translation_step_max_mm": 0.0,
                "root_translation_step_p999_mm": 0.0,
                "root_rotation_step_max_deg": 0.0,
                "root_rotation_step_p999_deg": 0.0,
            },
            {},
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "coupler_metrics",
        lambda *_args: {
            "coupler_mean_residual_deg": 0.0,
            "coupler_max_residual_deg": 0.0,
            "coupler_worst": None,
            "n_couplers": 0,
        },
    )
    phase_values = {
        "stance_slip_failure_pct": 0.0,
        "swing_clearance_failure_pct": 0.0,
        "swing_scrape_failure_pct": 0.0,
        "swing_clearance_ratio_median": 1.0,
    }
    phase_annotations = {
        key: np.zeros(3, dtype=bool)
        for key in (
            "forefoot_up",
            "hindfoot_up",
            "stance_slip_failure",
            "swing_clearance_failure",
            "swing_scraping",
        )
    }
    monkeypatch.setattr(evaluator, "phase_metrics", lambda *_args: (phase_values, phase_annotations))
    monkeypatch.setattr(evaluator, "common_retargeting_rmse", lambda *_args: (0.0, 0.0, {"Pelvis": 0.0}))
    monkeypatch.setattr(
        evaluator,
        "solver_throughput",
        lambda *_args: {
            "solver_native_fps": 10.0,
            "solver_seconds_per_motion_second": 1.0,
            "t_frame_s": 0.1,
            "solver_solved_frames": 3,
        },
    )

    result = evaluate_method_motion(
        method_label="Method",
        method_subdir="method",
        motion_class="Steps",
        motion=motion,
        terrain_method="method",
        force_flat=True,
        thresholds=Thresholds(),
        cache_root=tmp_path,
        source_root=None,
        all_method_subdirs=("method",),
        common_interval_override=(0.05, 0.15),
    )

    assert result.row["error"] == ""
    assert count == 1
    assert os.environ["CONVERTED_AMASS_PATH"] == "must-not-change"
    assert result.detail == {"source": "shared pass"}
    assert result.quality["passed"] == 1
    assert result.row["contact_preservation_pct"] == pytest.approx(100.0)
    assert result.row["t_frame_s"] == pytest.approx(0.1)
    assert result.row["common_start_s"] == pytest.approx(0.05)
    assert result.row["common_end_s"] == pytest.approx(0.15)
    assert set(RENDER_ARRAY_SPECS) <= set(result.annotations)
    assert joint_call["fitted_shape_path"] == tmp_path / "MyoFullBody" / "shape_optimized.pkl"


def test_renderer_rejects_incomplete_frame_archives(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    np.savez(frames / "motion.npz", **frame_metadata(motion="motion", method="terra", n_frames=2))

    with pytest.raises(AnnotationError, match="incomplete"):
        Flags.load("motion", tmp_path, 2, method="terra")
