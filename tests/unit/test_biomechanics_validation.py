"""Tests for policy-to-EMG/GRF comparison and its frozen mappings."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terra.biomechanics_validation import (
    PhaseWindow,
    RolloutCapture,
    TraceMatch,
    _foot_contact_layout,
    _foot_grf_snapshot,
    _rollout_phase_windows,
    _vielemeyer_windows,
    _write_metrics,
    analyze_rollouts,
    emg_actuator_indices,
    match_rollout_to_trace,
    resolve_trace_match,
)


def test_vielemeyer_uses_annotated_stances_not_initial_partial_contact() -> None:
    time = np.arange(1801) / 1000.0
    valid = np.zeros((len(time), 3), dtype=bool)
    valid[:193, 2] = True
    valid[62:761, 1] = True
    valid[620:1337, 0] = True
    sidecar = {
        "grf_native_time_s": time,
        "grf_platform_valid_native": valid,
        "grf_platform_assignment": np.asarray(["platform1:left", "platform2:right", "platform3:left"]),
    }
    events = {
        "labels": ["Foot Off", "Foot Strike", "Foot Off", "Foot Strike", "Foot Off"],
        "sides": ["Left", "Right", "Right", "Left", "Left"],
        "time_s": np.asarray([2.845, 2.712, 3.411, 3.270, 3.987]),
        "fps": 200.0,
        "first_frame": 530,
        "frame_count": 371,
    }
    windows = _vielemeyer_windows(sidecar, events)
    assert [w.side for w in windows] == ["right", "left"]
    assert [w.label for w in windows] == ["contact1_platform2", "contact2_platform1"]
    assert windows[0].start_time_s == pytest.approx(0.06)
    assert windows[0].end_time_s == pytest.approx(0.755)


def test_metric_export_retains_correlation_support(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics(
        path, [{"signal": "emg", "paired_phase_samples": 98, "correlation_undefined_reason": "constant_policy_trace"}]
    )
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["paired_phase_samples"] == "98"
    assert row["correlation_undefined_reason"] == "constant_policy_trace"


def test_registry_resolves_retarget_cache_path_to_exact_motion(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    trace = trace_root / "gait120" / "S001" / "LevelWalking.npz"
    trace.parent.mkdir(parents=True)
    trace.touch()
    sidecar = tmp_path / "AllSteps_stageii_biomechanics.npz"
    sidecar.touch()
    row = {
        "dataset": "gait120",
        "subject": "S001",
        "motion_type": "LevelWalking",
        "condition": "LevelWalking",
        "direction": "",
        "motion": "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
        "motion_biomechanics_path": str(sidecar),
        "trace_path": str(trace),
    }
    with (trace_root / "motion_trace_matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    match = resolve_trace_match(
        tmp_path / "cache/MyoFullBody/terra/Gait120/S001/LevelWalking/Trial01/AllSteps_stageii.npz",
        trace_root=trace_root,
        artifact_root=tmp_path,
    )

    assert match.motion == row["motion"]
    assert match.trace_path == trace.resolve()
    assert match.sidecar_path == sidecar.resolve()


def test_fixed_emg_mapping_combines_biceps_femoris_heads() -> None:
    names = ("bflh_r", "bfsh_r", "recfem_r", "bflh_l", "bfsh_l")

    assert emg_actuator_indices(names, "BicepsFemoris", "right") == (0, 1)
    assert emg_actuator_indices(names, "RectusFemoris", "right") == (2,)
    assert emg_actuator_indices(names, "BicepsFemoris", "left") == (3, 4)
    with pytest.raises(ValueError, match="no fixed MyoFullBody actuator mapping"):
        emg_actuator_indices(names, "UnknownMuscle", "right")


def test_environment_case_override_disables_inherited_cohort_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    from loco_mujoco.task_factories import TaskFactory
    from terra.biomechanics_validation import _build_cpu_environment
    from terra.rl.hooks import TerraValidationVideoRecorder

    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "name": "TerraImitationFactory",
                    "params": {
                        "trajectory_cache_root": "/training/cache",
                        "trajectory_cache_key": "old-cohort-key",
                        "amass_dataset_conf": {"rel_dataset_path": ["old-motion"]},
                    },
                },
                "validation": {"amass_dataset_conf": {"rel_dataset_path": ["validation-motion"]}},
            }
        }
    )
    monkeypatch.setattr(TerraValidationVideoRecorder, "_build_env_params", lambda *_args: {})
    captured = []
    monkeypatch.setattr(
        TaskFactory, "get_factory_cls", lambda *_args: SimpleNamespace(make=lambda **kwargs: captured.append(kwargs))
    )
    _build_cpu_environment(config, ["subject-a"])
    _build_cpu_environment(config, ["subject-b"])
    assert [p["amass_dataset_conf"]["rel_dataset_path"] for p in captured] == [["subject-a"], ["subject-b"]]
    assert all(p["trajectory_cache_root"] == p["trajectory_cache_key"] == "" for p in captured)
    assert config.experiment.task_factory.params.trajectory_cache_key == "old-cohort-key"


def test_verified_reference_rejects_wrong_arrays_even_at_same_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from loco_mujoco.trajectory import Trajectory
    from loco_mujoco.trajectory.handler import TrajectoryHandler
    from terra.biomechanics_validation import _verified_reference

    qpos = np.arange(12, dtype=float).reshape(4, 3)
    data = SimpleNamespace(qpos=qpos, qvel=qpos, split_points=np.array([0, 4]))
    info = SimpleNamespace(frequency=100.0)
    monkeypatch.setattr(Trajectory, "load", lambda *_args, **_kwargs: SimpleNamespace(data=data, info=info))
    monkeypatch.setattr(TrajectoryHandler, "filter_and_extend", lambda *_args: (data, info))
    env = SimpleNamespace(
        paired_motion_path=tmp_path / "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii.npz",
        model=None,
        dt=0.01,
        th=SimpleNamespace(
            traj=SimpleNamespace(data=SimpleNamespace(qpos=qpos + 1.0, qvel=qpos, split_points=np.array([0, 4])))
        ),
    )
    match = TraceMatch(
        "gait120",
        "S001",
        "LevelWalking",
        "LevelWalking",
        "",
        "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
        tmp_path / "sidecar.npz",
        tmp_path / "trace.npz",
    )
    with pytest.raises(RuntimeError, match="loaded reference qpos"):
        _verified_reference(env, match, None)


def test_source_clock_masks_trimmed_frames_without_shifting_phase() -> None:
    from terra.biomechanics_validation import _resample_window

    source_time = np.linspace(0.02, 0.98, 97)
    phase = np.linspace(0.0, 100.0, 101)
    result = _resample_window(
        source_time, source_time[:, None], PhaseWindow(0, "cycle", "right", 0.0, 1.0), phase, source_boundary=True
    )
    assert np.isnan(result[:2]).all() and np.isnan(result[-2:]).all()
    np.testing.assert_allclose(result[2:-2, 0], phase[2:-2] / 100.0)
    with pytest.raises(ValueError, match="does not cover"):
        _resample_window(
            source_time, source_time[:, None], PhaseWindow(0, "wrong", "right", 0.0, 1.2), phase, source_boundary=True
        )


def test_real_mujoco_foot_forces_balance_body_weight() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep="0.002" gravity="0 0 -9.80665"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="pelvis" pos="0 0 0.1"><freejoint/>
          <body name="calcn_l" pos="0 0.15 0"><geom type="sphere" size="0.1" mass="1"/></body>
          <body name="calcn_r" pos="0 -0.15 0"><geom type="sphere" size="0.1" mass="1"/></body>
        </body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _step in range(1000):
        mujoco.mj_step(model, data)
    sides, roots = _foot_contact_layout(model, mujoco)
    forces = _foot_grf_snapshot(model, data, mujoco, sides, roots)
    np.testing.assert_allclose(forces[:, 2], [9.80665, 9.80665], rtol=1e-4)
    np.testing.assert_allclose(forces[:, :2], 0.0, atol=1e-8)


def _synthetic_trace(phase: np.ndarray, activation: np.ndarray, normal_grf_bw: np.ndarray) -> dict[str, np.ndarray]:
    emg = (5.0 + 2.0 * activation)[None, :, None]
    grf = np.zeros((1, len(phase), 1, 3), dtype=np.float64)
    grf[0, :, 0, 2] = normal_grf_bw * (20.0 * 9.80665)
    return {
        "phase_percent": phase,
        "phase_definition": np.array("published gait cycle (0-100%)"),
        "stride_positions": np.asarray(("cycle",)),
        "stride_sides": np.asarray(("right",)),
        "emg_mean": emg,
        "emg_std": np.zeros_like(emg),
        "emg_valid": np.ones_like(emg, dtype=bool),
        "emg_channels": np.asarray(("TA",)),
        "emg_muscles": np.asarray(("TibialisAnterior",)),
        "emg_channel_sides": np.asarray(("right",)),
        "grf_force_mean": grf,
        "grf_force_std": np.zeros_like(grf),
        "grf_valid": np.ones((1, len(phase), 1), dtype=bool),
        "grf_channels": np.asarray(("right",)),
        "grf_axes": np.asarray(("terra_x", "terra_y", "terra_z")),
        "grf_force_units": np.array("N"),
    }


def _synthetic_capture(seed: int, *, success: bool = True) -> RolloutCapture:
    time = np.linspace(0.0, 1.0, 101)
    activation = np.sin(np.pi * time) ** 2
    normal_grf_bw = 0.6 + 0.4 * np.sin(np.pi * time) ** 2
    grf = np.zeros((len(time), 2, 3), dtype=np.float64)
    grf[:, 1, 2] = normal_grf_bw * (10.0 * 9.80665)
    root = np.stack((time, np.zeros_like(time), np.ones_like(time)), axis=-1)
    return RolloutCapture(
        seed=seed,
        success=success,
        absorbing=not success,
        coverage=1.0 if success else 0.5,
        return_per_frame=1.0,
        time_s=time,
        actuator_names=("tibant_r",),
        activation=activation[:, None],
        grf_world_n=grf,
        root_position_m=root,
    )


def test_gait120_phase_windows_project_to_completed_retarget_clock(tmp_path: Path) -> None:
    capture = _synthetic_capture(1)
    capture.time_s = np.arange(242, dtype=np.float64) / 100.0
    capture.coverage = 241.0 / 242.0
    match = TraceMatch(
        dataset="gait120",
        subject="S098",
        motion_type="SlopeAscent",
        condition="SlopeAscent",
        direction="ascent",
        motion="Gait120/S098/SlopeAscent/Trial05/AllSteps_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    source = [
        PhaseWindow(0, "Step01", "right", 0.0, 1.27),
        PhaseWindow(0, "Step02", "right", 1.27, 2.45),
    ]

    mapped = _rollout_phase_windows(capture, match, source)

    expected_boundary = 2.41 * 1.27 / 2.45
    assert mapped[0].start_time_s == pytest.approx(0.0)
    assert mapped[0].end_time_s == pytest.approx(expected_boundary)
    assert mapped[1].start_time_s == pytest.approx(expected_boundary)
    assert mapped[1].end_time_s == pytest.approx(2.41)
    assert source[1].end_time_s == pytest.approx(2.45)

    capture.coverage = 0.9
    with pytest.raises(ValueError, match="rollout is incomplete"):
        _rollout_phase_windows(capture, match, source)


def test_source_boundary_tolerance_clips_natural_vielemeyer_completion(tmp_path: Path) -> None:
    capture = _synthetic_capture(1)
    capture.time_s = np.arange(170, dtype=np.float64) / 100.0
    capture.coverage = 169.0 / 170.0
    match = TraceMatch(
        dataset="vielemeyer",
        subject="Ref08",
        motion_type="ramp_10_down",
        condition="ramp_10_down",
        direction="down",
        motion="Vielemeyer/Ref08/ramp_10_down/example_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    source = [PhaseWindow(0, "contact2_platform2", "right", 0.48, 1.721)]

    mapped = _rollout_phase_windows(capture, match, source)

    assert mapped[0].start_time_s == pytest.approx(0.48)
    assert mapped[0].end_time_s == pytest.approx(1.69)
    assert source[0].end_time_s == pytest.approx(1.721)

    capture.coverage = 0.8
    assert _rollout_phase_windows(capture, match, source) == source


def test_vielemeyer_contact_channels_use_synchronized_foot_assignment(tmp_path: Path) -> None:
    time = np.linspace(0.0, 2.0, 201)
    force = np.zeros((len(time), 2, 3), dtype=np.float64)
    force[:, 0, 2] = 100.0 + time
    force[:, 1, 2] = 200.0 + time
    capture = RolloutCapture(
        seed=1,
        success=True,
        absorbing=False,
        coverage=1.0,
        return_per_frame=1.0,
        time_s=time,
        actuator_names=("unused",),
        activation=np.zeros((len(time), 1)),
        grf_world_n=force,
        root_position_m=np.zeros((len(time), 3)),
    )
    match = TraceMatch(
        dataset="vielemeyer",
        subject="Ref05",
        motion_type="ramp_10_up",
        condition="ramp_10_up",
        direction="up",
        motion="Vielemeyer/Ref05/ramp_10_up/trial_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    trace = {
        "phase_percent": np.asarray((0.0, 50.0, 100.0)),
        "stride_positions": np.asarray(("trial_average",)),
        "emg_muscles": np.asarray((), dtype=str),
        "emg_channel_sides": np.asarray((), dtype=str),
        "grf_channels": np.asarray(("contact1", "contact2")),
    }
    windows = [
        PhaseWindow(0, "contact1_platform1", "left", 0.0, 1.0, grf_channel_indices=(0,)),
        PhaseWindow(0, "contact2_platform2", "right", 1.0, 2.0, grf_channel_indices=(1,)),
    ]

    emg, grf, sources = match_rollout_to_trace(capture, match, trace, windows)

    assert emg.shape == (1, 3, 0)
    assert sources.shape == (1, 0)
    np.testing.assert_allclose(grf[0, :, 0, 2], (100.0, 100.5, 101.0))
    np.testing.assert_allclose(grf[0, :, 1, 2], (201.0, 201.5, 202.0))


def test_gait120_grf_uses_only_the_complete_contact_window(tmp_path: Path) -> None:
    time = np.linspace(0.0, 2.0, 201)
    force = np.zeros((len(time), 2, 3), dtype=np.float64)
    force[:, 0, 2] = 100.0 + time
    force[:, 1, 2] = 200.0 + time
    capture = RolloutCapture(
        seed=1,
        success=True,
        absorbing=False,
        coverage=1.0,
        return_per_frame=1.0,
        time_s=time,
        actuator_names=("unused",),
        activation=np.zeros((len(time), 1)),
        grf_world_n=force,
        root_position_m=np.zeros((len(time), 3)),
    )
    match = TraceMatch(
        dataset="gait120",
        subject="S016",
        motion_type="LevelWalking",
        condition="LevelWalking",
        direction="",
        motion="Gait120/S016/LevelWalking/Trial01/AllSteps_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    trace = {
        "phase_percent": np.asarray((0.0, 50.0, 100.0)),
        "stride_positions": np.asarray(("cycle",)),
        "emg_muscles": np.asarray((), dtype=str),
        "emg_channel_sides": np.asarray((), dtype=str),
        "grf_channels": np.asarray(("left", "right")),
    }
    windows = [
        PhaseWindow(0, "Step01", "right", 0.0, 1.0, grf_channel_indices=(0,)),
        PhaseWindow(0, "Step02", "right", 1.0, 2.0, grf_channel_indices=(1,)),
    ]

    _emg, grf, _sources = match_rollout_to_trace(capture, match, trace, windows)

    np.testing.assert_allclose(grf[0, :, 0, 2], (100.0, 100.5, 101.0))
    np.testing.assert_allclose(grf[0, :, 1, 2], (201.0, 201.5, 202.0))


def test_gait120_combined_chair_grf_sums_both_simulated_feet(tmp_path: Path) -> None:
    time = np.linspace(0.0, 1.0, 101)
    force = np.zeros((len(time), 2, 3), dtype=np.float64)
    force[:, 0, 2] = 100.0 + time
    force[:, 1, 2] = 200.0 + 2.0 * time
    capture = RolloutCapture(
        seed=1,
        success=True,
        absorbing=False,
        coverage=1.0,
        return_per_frame=1.0,
        time_s=time,
        actuator_names=("unused",),
        activation=np.zeros((len(time), 1)),
        grf_world_n=force,
        root_position_m=np.zeros((len(time), 3)),
    )
    match = TraceMatch(
        dataset="gait120",
        subject="S016",
        motion_type="SitToStand",
        condition="SitToStand",
        direction="",
        motion="Gait120/S016/SitToStand/Trial01/AllSteps_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    trace = {
        "phase_percent": np.asarray((0.0, 50.0, 100.0)),
        "stride_positions": np.asarray(("cycle",)),
        "emg_muscles": np.asarray((), dtype=str),
        "emg_channel_sides": np.asarray((), dtype=str),
        "grf_channels": np.asarray(("combined",)),
    }

    _emg, grf, _sources = match_rollout_to_trace(
        capture,
        match,
        trace,
        [PhaseWindow(0, "Step01", "right", 0.0, 1.0, grf_channel_indices=(0,))],
    )

    np.testing.assert_allclose(grf[0, :, 0, 2], (300.0, 301.5, 303.0))


def test_analysis_normalizes_before_averaging_and_excludes_failed_rollouts(tmp_path: Path) -> None:
    phase = np.linspace(0.0, 100.0, 101)
    activation = np.sin(np.pi * phase / 100.0) ** 2
    normal_grf_bw = 0.6 + 0.4 * activation
    trace = _synthetic_trace(phase, activation, normal_grf_bw)
    match = TraceMatch(
        dataset="gait120",
        subject="S001",
        motion_type="LevelWalking",
        condition="LevelWalking",
        direction="",
        motion="Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    captures = [_synthetic_capture(1), _synthetic_capture(2), _synthetic_capture(3, success=False)]

    arrays, rows, summary = analyze_rollouts(
        captures,
        match,
        trace,
        [PhaseWindow(0, "cycle", "right", 0.0, 1.0)],
        model_mass_kg=10.0,
        experimental_mass_kg_value=20.0,
    )

    assert arrays["generated_emg_trials_shape_normalized"].shape == (2, 1, 101, 1)
    np.testing.assert_allclose(
        arrays["generated_emg_mean_shape_normalized"][0, :, 0],
        activation,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        arrays["generated_grf_mean_body_weight"][0, :, 0, 2],
        normal_grf_bw,
        atol=1e-6,
    )
    assert summary["successful_rollouts"] == 2
    assert summary["failed_rollouts"] == 1
    assert summary["mean_emg_zero_lag_waveform_correlation"] == pytest.approx(1.0)
    assert summary["median_emg_peak_phase_error_percent"] == pytest.approx(0.0)
    assert summary["mean_normal_grf_waveform_correlation"] == pytest.approx(1.0)
    assert summary["mean_normal_grf_rmse_bw"] == pytest.approx(0.0, abs=1e-12)
    assert summary["mean_impulse_error_bw_phase"] == pytest.approx(0.0, abs=1e-12)
    assert {row["signal"] for row in rows} == {"emg", "grf"}


def test_analysis_counts_completed_gait_before_early_termination(tmp_path: Path) -> None:
    phase = np.linspace(0.0, 100.0, 101)
    activation = np.sin(np.pi * phase / 100.0) ** 2
    trace = _synthetic_trace(phase, activation, 0.6 + 0.4 * activation)
    match = TraceMatch(
        dataset="gait120",
        subject="S001",
        motion_type="LevelWalking",
        condition="LevelWalking",
        direction="",
        motion="Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
        sidecar_path=tmp_path / "sidecar.npz",
        trace_path=tmp_path / "trace.npz",
    )
    capture = _synthetic_capture(7, success=False)
    retained = 76
    capture.time_s = capture.time_s[:retained]
    capture.source_time_s = capture.time_s.copy()
    capture.reference_frame = np.arange(retained)
    capture.activation = capture.activation[:retained]
    capture.grf_world_n = capture.grf_world_n[:retained]
    capture.root_position_m = capture.root_position_m[:retained]
    capture.coverage = 0.75

    arrays, rows, summary = analyze_rollouts(
        [capture],
        match,
        trace,
        [
            PhaseWindow(0, "completed", "right", 0.0, 0.5),
            PhaseWindow(0, "interrupted", "right", 0.5, 1.0),
        ],
        model_mass_kg=10.0,
        experimental_mass_kg_value=20.0,
    )

    assert summary == {
        **summary,
        "successful_rollouts": 0,
        "failed_rollouts": 1,
        "measured_rollouts": 1,
        "unmeasured_rollouts": 0,
        "measured_early_terminated_rollouts": 1,
        "completed_gaits": 1,
        "completed_gaits_in_early_terminated_rollouts": 1,
    }
    np.testing.assert_array_equal(arrays["successful_seed"], [])
    np.testing.assert_array_equal(arrays["measured_seed"], [7])
    np.testing.assert_array_equal(arrays["completed_gait_seed"], [7])
    np.testing.assert_array_equal(arrays["completed_gait_source_window_index"], [0])
    assert arrays["generated_emg_gaits_shape_normalized"].shape == (1, 1, 101, 1)
    assert {row["completed_gaits"] for row in rows} == {1}


def _fake_name_to_id(_model, _kind, name):
    return {"calcn_l": 2, "toes_l": 3, "calcn_r": 4, "toes_r": 5}.get(name, -1)


def _fake_contact_force(_model, data, contact_index, output):
    output[:] = data.forces[contact_index]


_FAKE_MUJOCO = SimpleNamespace(
    mjtObj=SimpleNamespace(mjOBJ_BODY=1),
    mj_name2id=_fake_name_to_id,
    mj_contactForce=_fake_contact_force,
)


def test_contact_forces_are_summed_as_world_forces_acting_on_each_foot() -> None:
    model = SimpleNamespace(
        nu=0,
        nbody=6,
        ngeom=3,
        body_parentid=np.asarray((0, 0, 1, 2, 1, 4)),
        geom_bodyid=np.asarray((0, 2, 4)),
    )
    contacts = [
        SimpleNamespace(geom1=0, geom2=1, frame=np.asarray((0, 0, 1, 1, 0, 0, 0, 1, 0))),
        SimpleNamespace(geom1=2, geom2=0, frame=np.asarray((0, 0, -1, 1, 0, 0, 0, -1, 0))),
    ]
    data = SimpleNamespace(
        ncon=2,
        contact=contacts,
        forces=np.asarray(((100, 0, 0, 0, 0, 0), (200, 0, 0, 0, 0, 0))),
    )

    layout, robot_roots = _foot_contact_layout(model, _FAKE_MUJOCO)
    result = _foot_grf_snapshot(model, data, _FAKE_MUJOCO, layout, robot_roots)

    np.testing.assert_allclose(result, ((0.0, 0.0, 100.0), (0.0, 0.0, 200.0)))
