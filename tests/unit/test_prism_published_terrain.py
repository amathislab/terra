"""Tests for PRISM reference-mesh scoring of published TERRA terrain."""

from __future__ import annotations

import numpy as np
import pytest

from musclemimic.retargeting import BoxSpec, TerrainSpec
from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
from terra.datasets.prism.evaluation import (
    _candidate_fit_statuses,
    _condition_group_for_terrain_class,
    _evaluate_motion,
    _fit_admission,
    _manifest_selections,
    _seated_support_xy,
    score_prepared_take,
    summarize,
    write_report,
)


def _cube_objects(center=(0.0, 0.0, 0.1), half=(0.2, 0.3, 0.1)):
    center = np.asarray(center)
    half = np.asarray(half)
    vertices = np.array([center + half * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = np.array(
        [
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
        ],
        dtype=int,
    )
    return {"box": {"vertices": vertices, "faces": faces}}


def _take(objects, *, trial_name="Stepping Boxes"):
    contacts = np.ones((4, 2), dtype=bool)
    cop = np.tile(np.array([[0.0, 0.0, 0.2]]), (4, 1))
    return {
        "info": {"data_info": {"fps": 100.0, "trial_name": trial_name}},
        "smpl_params": {"poses": np.zeros((4, 72))},
        "insole": {
            "L_Foot": {"contacts": contacts, "CoP_world": cop},
            "R_Foot": {"contacts": contacts, "CoP_world": cop},
        },
        "objects": objects,
    }


def _record(terrain, *, validation=True):
    return {
        "motion": "PRISM/subj001/take021_poses",
        "terrain": terrain.to_dict(),
        "fit": {"model": "per_level", "seat": {"seats": []}},
        "validation": {"passed": validation},
    }


def test_published_terrain_scores_exactly_against_heldout_mesh():
    objects = _cube_objects()
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))

    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(objects),
        _record(terrain),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )

    assert result["primary_pass"] is True
    assert result["primary_pass_with_internal_validation"] is True
    assert result["mesh_at_observed_support"]["height_mae_m"] == pytest.approx(0.0)
    assert result["mesh_at_observed_support"]["within_50mm"] == 1.0
    assert result["mesh_full"]["raised_footprint_iou"] == 1.0
    assert result["mesh_full"]["raised_terrain_coverage"] == 1.0
    assert result["mesh_full"]["flat_terrain_coverage"] == 1.0
    assert result["coordinate_audit"]["cop_to_mesh_centered_p95_m"] == pytest.approx(0.0)


def test_heldout_primary_gate_is_independent_of_method_internal_validation():
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))

    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(_cube_objects()),
        _record(terrain, validation=False),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )

    assert result["primary_pass"] is True
    assert result["primary_pass_with_internal_validation"] is False


def test_published_terrain_primary_gate_rejects_wrong_support_height():
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.05), size=(0.2, 0.3, 0.05)),))

    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(_cube_objects()),
        _record(terrain),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )

    assert result["mesh_at_observed_support"]["height_mae_m"] == pytest.approx(0.1)
    assert result["mesh_at_observed_support"]["within_50mm"] == 0.0
    assert result["primary_pass"] is False


def test_foot_query_filter_rejects_active_insole_sample_above_support():
    objects = _cube_objects()
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    take = _take(objects)
    take["insole"]["L_Foot"]["CoP_world"] = take["insole"]["L_Foot"]["CoP_world"].copy()
    take["insole"]["L_Foot"]["CoP_world"][0, 2] = 0.45

    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        take,
        _record(terrain),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )

    assert result["foot_query_filter"] == {
        "candidate_points": 8,
        "accepted_points": 7,
        "rejected_points": 1,
        "vertical_tolerance_m": 0.1,
    }
    assert result["mesh_at_foot_support"]["points"] == 7


def test_foot_only_control_cannot_claim_complete_chair_primary_support():
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    record = _record(terrain)
    record["fit"]["unsupported_support_kinds"] = ["pelvis"]
    objects = _cube_objects()
    objects["seat"] = _cube_objects(center=(0.8, 0.0, 0.1))["box"]

    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(objects, trial_name="Sitting Boxes"),
        record,
        np.array([[0.8, 0.0]]),
        terrain_class="chair_sit",
        cop_stride=1,
    )

    assert result["mesh_at_foot_support"]["raised_points"] > 0
    assert result["primary_support_complete"] is False
    assert result["primary_eligible"] is True
    assert result["primary_pass"] is False
    assert result["primary_ineligibility_reason"] is None


def test_prism_dataset_completeness_gate_includes_scorer_errors():
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(_cube_objects()),
        _record(terrain),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )

    summary = summarize(
        [result],
        [{"motion": "PRISM/subj002/take021_poses", "error": "current fit failed"}],
        bootstrap_samples=10,
        seed=1,
    )

    assert summary["selected"] == 2
    assert summary["dataset_gate"]["complete_observed_support"] is False


def test_seated_query_is_detected_from_source_motion_with_fixed_replay(monkeypatch, tmp_path):
    import terra.smplh
    import terra.source

    names = list(SMPLH_DEMO_JOINTS)
    joints = np.zeros((30, len(names), 3), dtype=float)
    joints[:, names.index("Pelvis"), :2] = [1.0, 12.0]
    for index, joint in enumerate(("L_Toe", "R_Toe", "L_Ankle", "R_Ankle")):
        joints[:, names.index(joint), :2] = [0.0, 0.01 * index]
    monkeypatch.setattr(terra.smplh, "load_smplh_motion", lambda _path: {})
    calls = []

    def replay(*args, **kwargs):
        calls.append((args, kwargs))
        return joints, 30.0

    monkeypatch.setattr(terra.source, "motion_world_joints", replay)
    points = _seated_support_xy(
        "PRISM/subj001/take021_poses",
        tmp_path / "source.npz",
        terrain_class="chair_sit",
    )

    np.testing.assert_allclose(points, [[1.0, 12.0]])
    assert calls[0][1]["env_name"] == "MyoFullBody"
    assert calls[0][1]["use_fitted_shape"] is True
    assert calls[0][1]["calibrate_sites"] is True


def test_nonchair_seated_query_does_not_replay_source(monkeypatch, tmp_path):
    import terra.smplh

    monkeypatch.setattr(
        terra.smplh,
        "load_smplh_motion",
        lambda _path: pytest.fail("non-chair geometry must not replay the source"),
    )

    points = _seated_support_xy(
        "PRISM/subj001/take021_poses",
        tmp_path / "source.npz",
        terrain_class="platform",
    )

    assert points.shape == (0, 2)


def test_summary_reports_subject_clustered_ci_and_registered_gate(tmp_path):
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(_cube_objects()),
        _record(terrain),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )

    summary = summarize([result], [], bootstrap_samples=50, seed=7)

    assert summary["dataset_gate"]["passed"] is True
    assert summary["overall"]["micro_observed"]["height_mae_mm"] == pytest.approx(0.0)
    ci = summary["subject_clustered_bootstrap_ci95"]["observed_height_mae_mm"]
    assert ci["estimate"] == pytest.approx(0.0)
    assert ci["ci95"] == pytest.approx([0.0, 0.0])

    write_report(tmp_path, [result], summary)
    report = (tmp_path / "REPORT.md").read_text()
    assert "Primary pass" in report
    assert "Raised-terrain coverage" in report
    assert "Reconstruction-input support diagnostics" not in report
    assert len((tmp_path / "GIT_COMMIT").read_text().strip()) == 40


def test_prism_condition_groups_match_frozen_geometry_classes_and_preserve_raw_label():
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    raw_condition = "Sitting Boxes "

    result = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(_cube_objects(), trial_name=raw_condition),
        _record(terrain),
        np.array([[0.0, 0.0]]),
        terrain_class="chair_sit",
        cop_stride=1,
    )

    assert result["condition"] == raw_condition
    assert result["condition_group"] == "Sitting"
    assert result["terrain_class"] == "chair_sit"
    summary = summarize([result], [], bootstrap_samples=10, seed=3)
    assert summary["conditions"]["Sitting"]["seated_height_mae_mm"]["mean"] == pytest.approx(0.0)
    assert _condition_group_for_terrain_class("platform") == "Stepping boxes"
    assert _condition_group_for_terrain_class("stairs_up_down") == "Stairs"
    with pytest.raises(ValueError, match="unsupported PRISM terrain_class"):
        _condition_group_for_terrain_class("unknown_geometry")


def test_summary_combines_only_protocol_variants_with_the_same_geometry_class():
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    base = score_prepared_take(
        "PRISM/subj001/take021_poses",
        _take(_cube_objects()),
        _record(terrain),
        np.empty((0, 2)),
        terrain_class="platform",
        cop_stride=1,
    )
    # Exact raw-label inventory of the frozen 31-take object cohort. The one trailing-space
    # label is intentional and must remain visible in per-motion output.
    variants = (
        ("Sitting Boxes", "chair_sit", 4),
        ("Sitting Boxes ", "chair_sit", 1),
        ("Sitting Boxes 1", "chair_sit", 1),
        ("Sitting Boxes 2", "chair_sit", 1),
        ("Sitting Boxes Dark", "chair_sit", 6),
        ("Stepping Boxes", "platform", 6),
        ("Stepping Boxes Dark", "platform", 6),
        ("Stepping Stair", "stairs_up_down", 6),
    )
    records = [
        base
        | {
            "condition": raw_condition,
            "condition_group": _condition_group_for_terrain_class(terrain_class),
            "terrain_class": terrain_class,
        }
        for raw_condition, terrain_class, count in variants
        for _ in range(count)
    ]

    summary = summarize(records, [], bootstrap_samples=10, seed=7)

    assert set(summary["conditions"]) == {"Sitting", "Stairs", "Stepping boxes"}
    assert summary["conditions"]["Sitting"]["takes"] == 13
    assert summary["conditions"]["Stepping boxes"]["takes"] == 12
    assert summary["conditions"]["Stairs"]["takes"] == 6
    assert summary["conditions"]["Sitting"]["terrain_classes"] == ["chair_sit"]
    assert summary["conditions"]["Sitting"]["raw_conditions"]["Sitting Boxes "] == 1


def test_manifest_condition_group_comes_from_required_frozen_terrain_class(tmp_path):
    manifest = tmp_path / "paired.csv"
    manifest.write_text(
        "motion,condition,terrain_class\nPRISM/subj001/take021_poses,arbitrary protocol label,stairs_up_down\n"
    )

    assert _manifest_selections(manifest) == [
        {
            "motion": "PRISM/subj001/take021_poses",
            "terrain_class": "stairs_up_down",
            "condition_group": "Stairs",
        }
    ]

    manifest.write_text("motion,terrain_class\nPRISM/subj001/take021_poses,new_geometry\n")
    with pytest.raises(ValueError, match="unsupported PRISM terrain_class"):
        _manifest_selections(manifest)


def test_failed_current_fit_status_blocks_stale_prism_record_before_read(tmp_path):
    motion = "PRISM/subj001/take021_poses"
    terrain_dir = tmp_path / "terrain"
    terrain_dir.mkdir()
    stale = terrain_dir / "PRISM__subj001__take021_poses.json"
    stale.write_text("this stale record must never be parsed")
    status_path = tmp_path / "status.csv"
    status_path.write_text(f"motion,status,error\n{motion},failed,request conflict\n")
    statuses = _candidate_fit_statuses(status_path)
    task = {
        "motion": motion,
        "terrain_dir": str(terrain_dir),
        **_fit_admission(motion, statuses, status_path),
    }

    with pytest.raises(RuntimeError, match="request conflict"):
        _evaluate_motion(task)

    assert stale.read_text() == "this stale record must never be parsed"


def test_missing_current_fit_status_is_an_explicit_prism_admission_failure(tmp_path):
    motion = "PRISM/subj001/take021_poses"
    status_path = tmp_path / "missing-status.csv"

    admission = _fit_admission(motion, _candidate_fit_statuses(status_path), status_path)

    assert admission["candidate_fit_status"] is None
    assert "fit status is missing" in admission["candidate_fit_admission_error"]
