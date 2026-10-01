"""Tests for scoring the terrain records used by dataset retargeting runs."""

from __future__ import annotations

import csv
import json
import math

import pytest

from terra.benchmarking.reconstruction.core import STATUS_FIELDS
from terra.benchmarking.reconstruction.provenance import (
    METHOD_IDENTITY_SCHEMA,
    RECORD_PROVENANCE_SCHEMA,
    RUN_PROVENANCE_SCHEMA,
    SCIENTIFIC_IDENTITY_SCHEMA,
    content_sha256,
)
from terra.evaluation.reconstruction import (
    evaluate,
    summarize,
)

METHOD = "fixture-method"


def _fixture_provenance():
    method_identity = {
        "schema": METHOD_IDENTITY_SCHEMA,
        "method": METHOD,
        "resolved_options": {},
        "source": {
            "git_commit": "0" * 40,
            "state": "clean",
            "tree_sha256": "1" * 64,
        },
    }
    method_hash = content_sha256(method_identity)
    scientific_identity = {
        "schema": SCIENTIFIC_IDENTITY_SCHEMA,
        "method_identity_sha256": method_hash,
        "selection_sha256": "2" * 64,
        "dataset_config_sha256": None,
        "matrix_sha256": None,
    }
    return {
        "schema": RUN_PROVENANCE_SCHEMA,
        "method_identity": method_identity,
        "method_identity_sha256": method_hash,
        "scientific_identity": scientific_identity,
        "scientific_identity_sha256": content_sha256(scientific_identity),
        "inputs": {
            "selection": {"path": "fixture.csv", "sha256": "2" * 64},
            "dataset_config": None,
            "matrix": None,
        },
    }


FIXTURE_PROVENANCE = _fixture_provenance()


def _run_metadata(motions):
    return {
        "method": METHOD,
        "motions": len(motions),
        "options": {},
        "provenance": FIXTURE_PROVENANCE,
    }


def _record_provenance():
    return {
        "schema": RECORD_PROVENANCE_SCHEMA,
        "method_identity_sha256": FIXTURE_PROVENANCE["method_identity_sha256"],
        "scientific_identity_sha256": FIXTURE_PROVENANCE["scientific_identity_sha256"],
    }


def _write_manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("motion", "dataset", "terrain_class", "expected_family"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_terrain(directory, motion, *, model, slope=None, boxes=1):
    ramp = {} if slope is None else {"slope_deg": slope}
    payload = {
        # This stored label is deliberately ignored: family must come from the fitted model.
        "selected_family": "ramp",
        "terrain": {"boxes": [{} for _ in range(boxes)]},
        "fit": {"model": model, "ramp": ramp},
        "validation": {
            "passed": True,
            "raised_contact_error_max": 0.004,
            "max_penetration": 0.002,
            "n_uncovered_contacts": 1,
        },
    }
    _write_payload(directory, motion, payload)


def _box(x, z, *, width=0.4, depth=0.4, pitch=0.0, name="terrain_box_baseline"):
    return {
        "pos": [x, 0.0, z / 2.0],
        "size": [width / 2.0, depth / 2.0, z / 2.0],
        "yaw": 0.0,
        "pitch": pitch,
        "name": name,
    }


def _step_reference_contacts(riser):
    """Fixed source-only contacts spanning the floor and first two treads."""

    contacts = []
    for joint, offset in (("L_Toe", 0.005), ("R_Toe", 0.005), ("L_Ankle", 0.040), ("R_Ankle", 0.040)):
        for ordinal, x in enumerate((-0.5, 0.0, 0.5)):
            contacts.append(
                {
                    "link": joint,
                    "kind": "foot",
                    "start": ordinal * 20,
                    "end": ordinal * 20 + 10,
                    "surface_xyz_m": [x, 0.0, offset + ordinal * riser],
                }
            )
    return contacts


def _write_payload(directory, motion, payload):
    manifest = directory.parent / "manifest.csv"
    scientific = {key: payload.get(key, {}) for key in ("terrain", "fit", "validation")}
    record = {
        "method": METHOD,
        "method_display_name": "Fixture",
        "motion": motion,
        "provenance": _record_provenance(),
        **scientific,
    }
    path = directory / f"{motion.replace('/', '__')}.json"
    path.write_text(json.dumps(record))
    with manifest.open(newline="") as handle:
        motions = [row["motion"] for row in csv.DictReader(handle)]
    status_path = directory / "status.csv"
    existing = {}
    if status_path.is_file():
        with status_path.open(newline="") as handle:
            existing = {row["motion"]: row for row in csv.DictReader(handle)}
    existing[motion] = {
        "motion": motion,
        "method": METHOD,
        "status": "ok",
        "output": str(path),
        "elapsed_seconds": "0",
        "error": "",
        "summary_json": "{}",
    }
    with status_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(existing[item] for item in motions if item in existing)
    (directory / "run.json").write_text(json.dumps(_run_metadata(motions)))


def _write_fit_status(path, rows):
    terrain = path.parent / "terrain"
    manifest = path.parent / "manifest.csv"
    existing = {}
    source_status = terrain / "status.csv"
    if source_status.is_file():
        with source_status.open(newline="") as handle:
            existing = {row["motion"]: row for row in csv.DictReader(handle)}
    for update in rows:
        motion = update["motion"]
        existing.setdefault(
            motion,
            {
                "motion": motion,
                "method": METHOD,
                "status": "failed",
                "output": str(terrain / f"{motion.replace('/', '__')}.json"),
                "elapsed_seconds": "0",
                "error": "",
                "summary_json": "",
            },
        ).update(update)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(existing.values())
    with manifest.open(newline="") as handle:
        motions = list(csv.DictReader(handle))
    (terrain / "run.json").write_text(json.dumps(_run_metadata(motions)))


def test_vielemeyer_scores_the_authoritative_terrain_against_nominal_angle(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_75_up/trial_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}],
    )
    _write_terrain(terrain, motion, model="ramp", slope=8.0)

    rows = evaluate(manifest, terrain)

    assert rows[0]["specification_source"] == "vielemeyer_nominal_ramp"
    assert rows[0]["expected_slope_deg"] == 7.5
    assert rows[0]["fitted_slope_deg"] == 8.0
    assert rows[0]["slope_abs_error_deg"] == 0.5
    assert rows[0]["family_correct"] is True
    assert rows[0]["raised_contact_error_max_mm"] == 4.0
    assert rows[0]["max_penetration_mm"] == 2.0


def test_gait120_chair_height_uses_published_stool_and_keeps_misses_in_denominator(tmp_path):
    motions = (
        "Gait120/S001/SitToStand/Trial01/AllSteps_stageii",
        "Gait120/S002/StandToSit/Trial01/AllSteps_stageii",
    )
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [
            {
                "motion": motion,
                "dataset": "gait120",
                "terrain_class": "chair_sit",
                "expected_family": "steps",
            }
            for motion in motions
        ],
    )
    _write_payload(
        terrain,
        motions[0],
        {
            "terrain": {
                "boxes": [
                    _box(0.0, 0.465, name="terrain_box_seat_0"),
                    _box(1.0, 0.10, name="terrain_box_baseline"),
                ]
            },
            "fit": {"model": "per_level"},
            "validation": {},
        },
    )
    _write_payload(
        terrain,
        motions[1],
        {
            "terrain": {"boxes": []},
            "fit": {"model": "per_level"},
            "validation": {},
        },
    )

    rows = evaluate(manifest, terrain)
    reconstructed, missed = rows

    assert reconstructed["specification_source"] == "gait120_published_stool"
    assert reconstructed["expected_seat_height_m"] == pytest.approx(0.490)
    assert reconstructed["raised_seat_support_present"] is True
    assert reconstructed["predicted_seat_height_m"] == pytest.approx(0.465)
    assert reconstructed["seat_height_abs_error_m"] == pytest.approx(0.025)
    assert missed["raised_seat_support_present"] is False
    assert missed["predicted_seat_height_m"] == pytest.approx(0.0)
    assert missed["seat_height_abs_error_m"] == pytest.approx(0.490)

    summary = summarize(rows)["datasets"]["gait120"]
    assert summary["seat_expected"] == 2
    assert summary["seat_available"] == 2
    assert summary["raised_seat_support_present"] == 1
    assert summary["seat_height_mae_mm"] == pytest.approx(257.5)
    assert summary["seat_height_abs_error_mm"]["max"] == pytest.approx(490.0)


def test_family_score_is_not_copied_from_the_expected_label(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_10_down/trial_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_down", "expected_family": "ramp"}],
    )
    _write_terrain(terrain, motion, model="per_level", boxes=3)

    row = evaluate(manifest, terrain)[0]

    assert row["selected_family"] == "steps"
    assert row["family_correct"] is False
    assert row["fitted_slope_deg"] is None
    assert row["slope_abs_error_deg"] is None


def test_summary_separates_geometry_coverage_from_internal_fit_diagnostics(tmp_path):
    motions = (
        "Vielemeyer/Ref01/ramp_10_up/first_stageii",
        "Vielemeyer/Ref01/ramp_75_down/second_stageii",
    )
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [
            {"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}
            for motion in motions
        ],
    )
    _write_terrain(terrain, motions[0], model="ramp", slope=9.75)
    _write_terrain(terrain, motions[1], model="per_level", boxes=2)

    item = summarize(evaluate(manifest, terrain))["datasets"]["vielemeyer"]

    assert item["family_correct"] == 1
    assert item["family_evaluated"] == 2
    assert item["slope_expected"] == 2
    assert item["slope_available"] == 1
    assert item["ramp_angle_mae_deg"] == pytest.approx(0.25)
    assert item["slope_abs_error_deg"]["mean"] == pytest.approx(0.25)
    assert item["slope_abs_error_deg"]["median"] == pytest.approx(0.25)
    assert item["validation_passed"] == 2


def test_failed_current_fit_status_blocks_stale_apparatus_record(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_10_up/stale_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}],
    )
    stale = terrain / f"{motion.replace('/', '__')}.json"
    stale.write_text("this stale record must never be parsed")
    status = tmp_path / "status.csv"
    _write_fit_status(status, [{"motion": motion, "status": "failed", "error": "request conflict"}])

    row = evaluate(manifest, terrain, status)[0]

    assert row["candidate_fit_status"] == "failed"
    assert row["terrain_record_present"] is True
    assert row["terrain_available"] is False
    assert "request conflict" in row["error"]
    assert row["selected_model"] is None


def test_missing_current_fit_status_fails_closed_without_reading_json(tmp_path):
    motion = "Darmstadt/D01/config01/trial01/ascent_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "darmstadt", "terrain_class": "stairs", "expected_family": "steps"}],
    )
    (terrain / f"{motion.replace('/', '__')}.json").write_text("stale")
    (terrain / "run.json").write_text(json.dumps(_run_metadata([motion])))

    with pytest.raises(FileNotFoundError, match="status table not found"):
        evaluate(manifest, terrain, tmp_path / "missing-status.csv")


def test_unknown_nonempty_heightfield_does_not_fabricate_steps_family(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_10_up/unknown_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}],
    )
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": [_box(0.0, 0.2)]},
            "fit": {"model": "voronoi"},
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain)[0]

    assert row["selected_family"] is None
    assert row["selected_family_source"] == "unavailable"
    assert row["family_correct"] is None
    assert row["fitted_slope_deg"] is None
    assert row["fitted_slope_source"] == "unavailable"
    assert row["slope_abs_error_deg"] is None


def test_empty_heightfield_does_not_establish_family_or_grade(tmp_path):
    ramp_motion = "Vielemeyer/Ref01/ramp_10_up/flat_stageii"
    stair_motion = "Darmstadt/Subject/config02/flat_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [
            {
                "motion": ramp_motion,
                "dataset": "vielemeyer",
                "terrain_class": "ramp_up",
                "expected_family": "ramp",
            },
            {
                "motion": stair_motion,
                "dataset": "darmstadt",
                "terrain_class": "stairs",
                "expected_family": "steps",
            },
        ],
    )
    for motion, model in ((ramp_motion, "flat"), (stair_motion, "other_baseline")):
        fit = {"model": model}
        if motion == stair_motion:
            fit["support_intervals"] = _step_reference_contacts(0.17)
        _write_payload(
            terrain,
            motion,
            {
                "terrain": {"boxes": []},
                "fit": fit,
                "validation": {},
            },
        )

    rows = {row["motion"]: row for row in evaluate(manifest, terrain, contact_reference_dir=terrain)}
    ramp = rows[ramp_motion]
    stair = rows[stair_motion]

    assert ramp["selected_family"] == "flat"
    assert ramp["selected_family_source"] == "terra_fit_report.model"
    assert ramp["family_correct"] is False
    assert ramp["fitted_slope_deg"] is None
    assert ramp["fitted_slope_source"] == "unavailable"
    assert ramp["slope_abs_error_deg"] is None
    assert stair["selected_family"] is None
    assert stair["selected_family_source"] == "unavailable"
    assert stair["family_correct"] is None
    assert stair["step_contact_height_mae_m"] == pytest.approx(0.255)

    summary = summarize(list(rows.values()))["datasets"]
    assert summary["vielemeyer"]["slope_available"] == 0
    assert summary["vielemeyer"]["ramp_angle_mae_deg"] is None
    assert summary["darmstadt"]["step_available"] == 1
    assert summary["darmstadt"]["step_height_mae_mm"] == pytest.approx(255.0)


def test_unknown_plateaus_do_not_establish_ramp_family_or_grade(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_10_up/voronoi_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}],
    )
    boxes = [_box(x, z) for x, z in ((0.0, 0.10), (0.5, 0.20), (1.0, 0.30))]
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": boxes},
            "fit": {"model": "voronoi"},
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain)[0]

    assert row["selected_family"] is None
    assert row["family_correct"] is None
    assert row["fitted_slope_deg"] is None
    assert row["fitted_slope_source"] == "unavailable"
    assert row["slope_abs_error_deg"] is None


def test_unknown_pitched_top_faces_directly_establish_ramp_and_slope(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_10_up/pitched_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}],
    )
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": [_box(0.0, 0.2, pitch=math.radians(-9.0))]},
            "fit": {"model": "other_baseline"},
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain)[0]

    assert row["selected_family"] == "ramp"
    assert row["selected_family_source"] == "terrain_geometry.sloped_top_faces"
    assert row["family_correct"] is True
    assert row["fitted_slope_deg"] == pytest.approx(9.0)
    assert row["fitted_slope_source"] == "terrain_geometry.box_pitch"


def test_least_squares_plane_is_scored_at_step_contacts(tmp_path):
    motion = "Darmstadt/Subject/config02/plane_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "darmstadt", "terrain_class": "stairs", "expected_family": "steps"}],
    )
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": [_box(0.0, 0.2, pitch=math.radians(-8.0))]},
            "fit": {
                "model": "least_squares_contact_plane",
                "support_intervals": _step_reference_contacts(0.17),
            },
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain, contact_reference_dir=terrain)[0]

    assert row["selected_family"] is None
    assert row["selected_family_source"] == "not_applicable.no_family_selection"
    assert row["family_correct"] is None
    assert row["fitted_slope_deg"] == pytest.approx(8.0)
    assert row["step_contact_height_mae_m"] is not None
    assert row["step_raised_contacts"] == 8
    summary = summarize([row])["datasets"]["darmstadt"]
    assert summary["family_evaluated"] == 0
    assert summary["step_available"] == 1
    assert summary["step_contacts"] == 8


def test_native_report_metrics_take_precedence_over_geometry_fallback(tmp_path):
    motion = "Vielemeyer/Ref01/ramp_10_up/native_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "vielemeyer", "terrain_class": "ramp_up", "expected_family": "ramp"}],
    )
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": [_box(0.0, 0.2, pitch=math.radians(-20.0))]},
            "fit": {"model": "ramp", "ramp": {"slope_deg": 8.0}},
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain)[0]

    assert row["selected_family_source"] == "terra_fit_report.model"
    assert row["fitted_slope_deg"] == 8.0
    assert row["fitted_slope_source"] == "terra_fit_report.ramp.slope_deg"


def test_native_stair_record_uses_contact_height_for_primary_step_score(tmp_path):
    motion = "Darmstadt/Subject/config02/native_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "darmstadt", "terrain_class": "stairs", "expected_family": "steps"}],
    )
    boxes = [_box(x, z) for x, z in ((0.0, 0.10), (0.5, 0.20), (1.0, 0.30))]
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": boxes},
            "fit": {
                "model": "stair_flight",
                "stair_flight": {"heights": [0.10, 0.27, 0.44]},
                "support_intervals": _step_reference_contacts(0.17),
            },
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain, contact_reference_dir=terrain)[0]

    assert row["selected_family"] == "steps"
    assert row["step_contact_height_mae_m"] is not None


def test_step_mae_queries_candidate_height_at_fixed_source_contacts(tmp_path):
    motion = "Darmstadt/Subject/config02/contact_height_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "darmstadt", "terrain_class": "stairs", "expected_family": "steps"}],
    )
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": [_box(0.0, 0.15), _box(0.5, 0.30)]},
            # This deliberately wrong global riser must not affect the benchmark score.
            "fit": {
                "model": "stair_flight",
                "stair_flight": {"heights": [0.12, 0.24, 0.36]},
                "support_intervals": _step_reference_contacts(0.17),
            },
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain, contact_reference_dir=terrain)[0]
    summary = summarize([row])["datasets"]["darmstadt"]

    assert row["step_reference_contacts"] == 12
    assert row["step_raised_contacts"] == 8
    assert row["step_contact_height_mae_m"] == pytest.approx(0.03)
    assert row["step_contact_height_max_m"] == pytest.approx(0.04)
    assert summary["step_available"] == 1
    assert summary["step_contacts"] == 8
    assert summary["step_height_mae_mm"] == pytest.approx(30.0)
    assert summary["step_height_micro_mae_mm"] == pytest.approx(30.0)


def test_step_ground_truth_pairs_half_riser_ankles_to_overlapping_toes(tmp_path):
    motion = "Gait120/S080/StairAscent/half_riser_ankle_stageii"
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [{"motion": motion, "dataset": "gait120", "terrain_class": "stairs", "expected_family": "steps"}],
    )
    contacts = []
    for joint, offset in (("L_Toe", 0.002), ("R_Toe", 0.004), ("L_Ankle", 0.052), ("R_Ankle", 0.054)):
        for ordinal, x in enumerate((-0.5, 0.0, 0.5)):
            contacts.append(
                {
                    "link": joint,
                    "kind": "foot",
                    "start": ordinal * 20 + (1 if joint.endswith("Ankle") else 0),
                    "end": ordinal * 20 + 10,
                    "surface_xyz_m": [x, 0.0, offset + ordinal * 0.1],
                }
            )
    _write_payload(
        terrain,
        motion,
        {
            "terrain": {"boxes": [_box(0.0, 0.1), _box(0.5, 0.2)]},
            "fit": {"model": "stair_flight", "support_intervals": contacts},
            "validation": {},
        },
    )

    row = evaluate(manifest, terrain, contact_reference_dir=terrain)[0]

    assert row["step_raised_contacts"] == 8
    assert row["step_contact_height_mae_m"] == pytest.approx(0.0)


def test_scattered_or_malformed_unknown_geometry_stays_unavailable(tmp_path):
    motions = ("unknown/scattered", "unknown/malformed")
    manifest = tmp_path / "manifest.csv"
    terrain = tmp_path / "terrain"
    terrain.mkdir()
    _write_manifest(
        manifest,
        [
            {"motion": motion, "dataset": "custom", "terrain_class": "stairs", "expected_family": "steps"}
            for motion in motions
        ],
    )
    scattered = [_box(x, z) for x, z in ((0.0, 0.10), (3.0, 0.20), (6.0, 0.30))]
    _write_payload(
        terrain,
        motions[0],
        {"terrain": {"boxes": scattered}, "fit": {"model": "other_baseline"}, "validation": {}},
    )
    malformed = _box(0.0, 0.1)
    malformed["size"] = [0.2, -0.2, 0.05]
    _write_payload(
        terrain,
        motions[1],
        {"terrain": {"boxes": [malformed]}, "fit": {"model": "other_baseline"}, "validation": {}},
    )

    rows = evaluate(manifest, terrain)

    assert all(row["selected_family"] is None for row in rows)
    assert all(row["family_correct"] is None for row in rows)
    assert all(row["fitted_slope_deg"] is None for row in rows)
