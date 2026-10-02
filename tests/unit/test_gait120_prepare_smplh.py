import csv

from terra.datasets.gait120 import (
    MOVEMENT_METADATA,
    ClipRecord,
    _apply_fit_quality,
    _filter_clips_by_selection_manifest,
    _summarize_marker_fit_quality,
    _write_manifest,
    fit_clips,
    inspect_dataset,
)


def _clip(tmp_path, motion, *, subject=1, movement="SlopeAscent", mean=10.0):
    rel = motion.removesuffix("_stageii")
    return ClipRecord(
        subject=subject,
        movement=movement,
        trial=1,
        paired_steps="1;2",
        marker_archive=str(tmp_path / ".markers" / f"{rel}_markers.npz"),
        output_path=str(tmp_path / f"{motion}.npz"),
        motion=motion,
        trc_paths="step1.trc;step2.trc",
        marker_error_mean_mm=mean,
        status="existing",
    )


def test_manifest_paths_are_relative_to_output_root(tmp_path):
    clip = _clip(tmp_path, "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii")
    manifest = tmp_path / "manifest.csv"

    _write_manifest(manifest, [clip], output_root=tmp_path)

    with manifest.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["dataset"] == "gait120"
    assert row["output_path"] == "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii.npz"
    assert not row["output_path"].startswith("runs/gait120/smplh/")


def test_selection_assigns_retarget_and_calibration_roles(tmp_path):
    target = _clip(tmp_path, "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii")
    calibration = _clip(
        tmp_path,
        "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
        movement="LevelWalking",
    )
    manifest = tmp_path / "selection.csv"
    manifest.write_text(
        "motion,dataset,calibration_motion\n"
        "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii,gait120,"
        "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii\n"
    )

    selected = _filter_clips_by_selection_manifest([target, calibration], manifest)

    assert {clip.motion: clip.role for clip in selected} == {
        target.motion: "retarget",
        calibration.motion: "calibration",
    }


def test_failed_calibration_rejects_dependent_target(tmp_path):
    calibration = _clip(
        tmp_path,
        "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
        movement="LevelWalking",
        mean=25.0,
    )
    calibration.role = "calibration"
    dependent = _clip(tmp_path, "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii", mean=10.0)
    independent = _clip(
        tmp_path,
        "Gait120/S002/SlopeAscent/Trial01/AllSteps_stageii",
        subject=2,
        mean=11.0,
    )
    clips = [calibration, dependent, independent]
    quality = _summarize_marker_fit_quality(clips)

    rejections = _apply_fit_quality(
        clips,
        validation_failures=[],
        marker_fit_quality=quality,
    )

    assert calibration.fit_passed is False
    assert dependent.fit_passed is False
    assert "calibration_fit" in dependent.fit_failure_reason
    assert independent.fit_passed is True
    assert {row["motion"] for row in rejections} == {calibration.motion, dependent.motion}


def test_fit_progress_does_not_replace_public_manifest(tmp_path, monkeypatch):
    clip = _clip(tmp_path, "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii")
    public_manifest = tmp_path / "manifest.csv"
    public_manifest.write_text("previous-complete-manifest\n")

    def fake_fit_subject(job):
        return job["clips"]

    monkeypatch.setattr("terra.datasets.gait120._fit_subject", fake_fit_subject)
    fit_clips(
        [clip],
        output_root=tmp_path,
        workers=1,
        redo=False,
        smpl_model_path="smpl",
        gender="neutral",
        target_fps=50.0,
        stage1_iters=1,
        stage2_iters=1,
        n_ref_frames=1,
        stage1_shape_solver="joint_dogleg_jax",
        least_avail_markers=0.8,
        device="cpu",
        stage2_solver="batched_lbfgs",
        enforce_knee_hinge=True,
    )

    assert public_manifest.read_text() == "previous-complete-manifest\n"
    assert (tmp_path / ".conversion-progress" / "manifest.csv").is_file()


def test_stool_tasks_are_single_transition_chair_motions():
    assert MOVEMENT_METADATA["SitToStand"].steps == (1,)
    assert MOVEMENT_METADATA["StandToSit"].steps == (1,)
    assert MOVEMENT_METADATA["SitToStand"].terrain_class == "chair_sit"
    assert MOVEMENT_METADATA["StandToSit"].expected_family == "steps"
    assert MOVEMENT_METADATA["StairAscent"].steps == (1, 2)


def test_inspection_counts_task_specific_step_cardinality(tmp_path):
    steps, clips, report = inspect_dataset(
        original_root=tmp_path / "original",
        output_root=tmp_path / "output",
        subjects=[1],
        movements=("LevelWalking", "SitToStand", "StandToSit"),
        trials=[1, 2],
    )

    assert clips == []
    assert len(steps) == report["summary"]["expected_steps"] == 8
    assert report["by_movement"]["LevelWalking"]["expected_steps"] == 4
    assert report["by_movement"]["SitToStand"]["expected_steps"] == 2
    assert report["by_movement"]["StandToSit"]["expected_steps"] == 2


def test_marker_only_gait120_input_can_be_prepared(tmp_path):
    import numpy as np

    from terra.datasets.gait120 import prepare_marker_archives

    original = tmp_path / "original"
    trial = original / "S001/MotionCapture/LevelWalking/TRC/Trial01"
    trial.mkdir(parents=True)
    for step in (1, 2):
        path = trial / f"Step{step:02d}.trc"
        path.write_text(
            "PathFileType\t4\t(X/Y/Z)\tstep.trc\n"
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
            "100\t100\t3\t2\tmm\n"
            "Frame#\tTime\tLANK\t\t\tRANK\t\t\n"
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n"
            + "".join(
                f"{frame + 1}\t{((step - 1) * 3 + frame) / 100:.2f}\t100\t200\t300\t400\t500\t600\n"
                for frame in range(3)
            )
        )
    _steps, clips, report = inspect_dataset(
        original_root=original,
        output_root=tmp_path / "output",
        subjects=[1],
        movements=("LevelWalking",),
        trials=[1],
    )
    assert report["summary"]["valid_trc_steps"] == 2
    assert len(clips) == 1 and clips[0].paired_steps == "1;2"
    prepare_marker_archives(clips)
    with np.load(clips[0].marker_archive, allow_pickle=False) as archive:
        assert archive["positions"].shape == (6, 2, 3)
        np.testing.assert_allclose(archive["positions"][0, 0], [0.1, -0.3, 0.2])
