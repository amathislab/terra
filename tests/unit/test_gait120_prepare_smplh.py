import csv
from collections import Counter

from terra.datasets.gait120 import (
    MOVEMENT_METADATA,
    ClipRecord,
    _apply_fit_quality,
    _filter_clips_by_selection_manifest,
    _summarize_marker_fit_quality,
    _write_manifest,
    audit_dataset,
    collect_chair_conversion_clips,
    fit_clips,
    select_balanced_chair_clips,
    write_chair_selection,
)


def _clip(tmp_path, motion, *, subject=1, movement="SlopeAscent", mean=10.0):
    rel = motion.removesuffix("_stageii")
    return ClipRecord(
        subject=subject,
        movement=movement,
        trial=1,
        paired_steps="1;2",
        marker_archive=str(tmp_path / ".markers" / f"{rel}_markers.npz"),
        emg_archive=str(tmp_path / f"{rel}_emg.npz"),
        output_path=str(tmp_path / f"{motion}.npz"),
        motion=motion,
        emg_path="source.mat",
        trc_paths="step1.trc;step2.trc",
        mot_paths="step1.mot;step2.mot",
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
    assert row["biomechanics_archive"] == ""
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


def test_emg_validation_failure_rejects_clip(tmp_path):
    clip = _clip(tmp_path, "Gait120/S001/SitToStand/Trial01/AllSteps_stageii")
    quality = _summarize_marker_fit_quality([clip])

    rejections = _apply_fit_quality(
        [clip],
        validation_failures=[],
        emg_validation_failures=[{"motion": clip.motion, "error": "non-finite EMG"}],
        marker_fit_quality=quality,
    )

    assert clip.fit_passed is False
    assert clip.fit_failure_reason == "emg_validation: non-finite EMG"
    assert [row["motion"] for row in rejections] == [clip.motion]


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


def test_audit_counts_task_specific_step_cardinality(tmp_path):
    steps, clips, audit = audit_dataset(
        original_root=tmp_path / "original",
        emg_root=tmp_path / "emg",
        output_root=tmp_path / "output",
        subjects=[1],
        movements=("LevelWalking", "SitToStand", "StandToSit"),
        trials=[1, 2],
    )

    assert clips == []
    assert len(steps) == audit["summary"]["expected_steps"] == 8
    assert audit["by_movement"]["LevelWalking"]["expected_steps"] == 4
    assert audit["by_movement"]["SitToStand"]["expected_steps"] == 2
    assert audit["by_movement"]["StandToSit"]["expected_steps"] == 2


def test_balanced_chair_selection_is_exact_reproducible_and_calibrated(tmp_path):
    clips = []
    for subject in range(1, 26):
        clips.append(
            _clip(
                tmp_path,
                f"Gait120/S{subject:03d}/LevelWalking/Trial01/AllSteps_stageii",
                subject=subject,
                movement="LevelWalking",
            )
        )
        for movement in ("SitToStand", "StandToSit"):
            for trial in range(1, 6):
                clip = _clip(
                    tmp_path,
                    f"Gait120/S{subject:03d}/{movement}/Trial{trial:02d}/AllSteps_stageii",
                    subject=subject,
                    movement=movement,
                )
                clip.trial = trial
                clip.paired_steps = "1"
                clip.terrain_class = "chair_sit"
                clip.expected_family = "steps"
                clips.append(clip)

    targets, selected = select_balanced_chair_clips(clips, per_movement=105)
    repeated, _ = select_balanced_chair_clips(clips, per_movement=105)

    assert [clip.motion for clip in targets] == [clip.motion for clip in repeated]
    assert len(targets) == 210
    assert Counter(clip.movement for clip in targets) == Counter(SitToStand=105, StandToSit=105)
    assert Counter((clip.movement, clip.trial) for clip in targets) == Counter(
        {(movement, trial): 21 for movement in ("SitToStand", "StandToSit") for trial in range(1, 6)}
    )
    target_subjects = {clip.subject for clip in targets}
    calibrations = [clip for clip in selected if clip.role == "calibration"]
    assert {clip.subject for clip in calibrations} == target_subjects
    assert all(clip.movement == "LevelWalking" and clip.trial == 1 for clip in calibrations)

    manifest = tmp_path / "chair_selection.csv"
    write_chair_selection(manifest, targets)
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 210
    assert {row["dataset"] for row in rows} == {"gait120"}
    assert {row["terrain_class"] for row in rows} == {"chair_sit"}
    assert all(
        row["calibration_motion"]
        == f"Gait120/{row['subject']}/LevelWalking/Trial01/AllSteps_stageii"
        for row in rows
    )


def test_chair_conversion_pool_precedes_successful_balanced_selection(tmp_path):
    clips = []
    for subject in range(1, 24):
        calibration = _clip(
            tmp_path,
            f"Gait120/S{subject:03d}/LevelWalking/Trial01/AllSteps_stageii",
            subject=subject,
            movement="LevelWalking",
        )
        calibration.fit_passed = subject != 1
        clips.append(calibration)
        for movement in ("SitToStand", "StandToSit"):
            for trial in range(1, 6):
                clip = _clip(
                    tmp_path,
                    f"Gait120/S{subject:03d}/{movement}/Trial{trial:02d}/AllSteps_stageii",
                    subject=subject,
                    movement=movement,
                )
                clip.trial = trial
                clip.fit_passed = subject != 2
                clips.append(clip)

    conversion_targets, conversion_clips = collect_chair_conversion_clips(clips)
    targets, selected = select_balanced_chair_clips(
        conversion_clips,
        per_movement=105,
        fit_passed_only=True,
    )

    assert len(conversion_targets) == 230
    assert len(conversion_clips) == 253
    assert len(targets) == 210
    assert {clip.subject for clip in targets}.isdisjoint({1, 2})
    assert all(clip.fit_passed for clip in selected)
    assert Counter((clip.movement, clip.trial) for clip in targets) == Counter(
        {(movement, trial): 21 for movement in ("SitToStand", "StandToSit") for trial in range(1, 6)}
    )
