from __future__ import annotations

import numpy as np
import pytest

from musclemimic.retargeting import BoxSpec, TerrainSpec
from terra.datasets.prism.mesh_metrics import (
    coordinate_audit,
    mesh_height_at,
    score_height_fields,
    score_observed_support,
)


def cube_object(center=(0.0, 0.0, 0.1), half=(0.2, 0.3, 0.1)):
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


def test_mesh_height_selects_top_surface_and_floor():
    objects = cube_object()
    xy = np.array([[0.0, 0.0], [0.19, 0.29], [0.21, 0.0]])
    np.testing.assert_allclose(mesh_height_at(xy, objects), [0.2, 0.2, 0.0])


def test_identical_box_has_perfect_support_metrics():
    objects = cube_object()
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.2, 0.3, 0.1)),))
    score = score_height_fields(objects, terrain, resolution=0.02)
    assert score["raised_footprint_iou"] == 1.0
    assert score["raised_terrain_coverage"] == 1.0
    assert score["flat_terrain_coverage"] == 1.0
    assert score["height_mae_union_m"] < 1e-12
    assert score["support_f1_20mm"] == 1.0


def test_raised_and_flat_coverage_separate_misses_from_false_positives():
    objects = cube_object()
    undersized = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.1, 0.15, 0.1)),))
    oversized = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(0.3, 0.4, 0.1)),))
    domain_xy = np.asarray(((-0.6, -0.6), (0.6, 0.6)))

    missed = score_height_fields(
        objects,
        undersized,
        resolution=0.02,
        evaluation_domain_xy=domain_xy,
        evaluation_margin=0.0,
    )
    hallucinated = score_height_fields(
        objects,
        oversized,
        resolution=0.02,
        evaluation_domain_xy=domain_xy,
        evaluation_margin=0.0,
    )

    assert missed["raised_terrain_coverage"] < 1.0
    assert missed["flat_terrain_coverage"] == 1.0
    assert hallucinated["raised_terrain_coverage"] == 1.0
    assert hallucinated["flat_terrain_coverage"] < 1.0
    assert missed["evaluation_area_m2"] > 0.0
    assert missed["evaluation_area_m2"] == hallucinated["evaluation_area_m2"]


def test_observed_support_penalizes_wrong_height():
    objects = cube_object()
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.075), size=(0.2, 0.3, 0.075)),))
    score = score_observed_support(np.array([[0.0, 0.0]]), objects, terrain)
    assert score["raised_coverage"] == 1.0
    assert score["height_mae_m"] == pytest.approx(0.05)
    assert score["within_20mm"] == 0.0
    assert score["within_50mm"] == 1.0


def test_coordinate_audit_records_but_does_not_remove_sensor_offset():
    objects = cube_object()
    contacts = np.ones((3, 2), dtype=bool)
    cop = np.tile(np.array([[0.0, 0.0, 0.225]]), (3, 1))
    take = {
        "insole": {
            "L_Foot": {"contacts": contacts, "CoP_world": cop},
            "R_Foot": {"contacts": contacts, "CoP_world": cop},
        }
    }

    audit = coordinate_audit(take, objects)

    assert audit["raised_contact_frames"] == 6
    assert audit["cop_to_mesh_median_bias_m"] == pytest.approx(0.025)
    assert audit["cop_to_mesh_centered_p95_m"] == pytest.approx(0.0)
    assert audit["warning"] is None
