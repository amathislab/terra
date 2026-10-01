from __future__ import annotations

import json

import numpy as np
import pytest

from terra.figures.reconstruction_ablations import (
    METHODS,
    _axis_aligned_surface_faces,
    _box_height_at,
    _spatially_thin_contacts,
    select_prism_boxes_example,
    select_ramp_example,
)


def test_select_ramp_example_uses_median_duration_and_stable_tie_break(tmp_path) -> None:
    directory = tmp_path / "vielemeyer" / "contact-least-squares" / "terrain"
    directory.mkdir(parents=True)
    for name, frames in (("B", 20), ("A", 20), ("C", 70)):
        motion = f"Vielemeyer/{name}/ramp_10_down/trial"
        path = directory / f"Vielemeyer__{name}__ramp_10_down__trial.json"
        path.write_text(json.dumps({"motion": motion, "fit": {"n_frames": frames}}))

    motion, selection = select_ramp_example(tmp_path)

    assert motion == "Vielemeyer/A/ramp_10_down/trial"
    assert selection["median_frames"] == 20.0
    assert selection["candidate_count"] == 3


def test_select_prism_boxes_example_uses_median_reference_area(tmp_path) -> None:
    path = tmp_path / "prism" / "terra" / "evaluation" / "prism_mesh" / "per_motion.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {
                    "motion": f"PRISM/subj00{index}/take001_poses",
                    "condition_group": "Stepping boxes",
                    "mesh_full": {"gt_raised_area_m2": area},
                }
                for index, area in enumerate((0.2, 0.4, 1.8), start=1)
            ]
        )
    )

    motion, selection = select_prism_boxes_example(tmp_path)

    assert motion == "PRISM/subj002/take001_poses"
    assert selection["median_reference_area_m2"] == 0.4
    assert selection["candidate_count"] == 3


def test_box_height_query_uses_serialized_pitched_top_surface() -> None:
    box = {
        "pos": [0.0, 0.0, 0.1],
        "size": [1.0, 0.5, 0.2],
        "yaw": 0.0,
        "pitch": -0.1,
    }

    height = _box_height_at(box, np.asarray([0.0, 2.0]), np.asarray([0.0, 0.0]))

    assert height[0] == pytest.approx(0.1 + 0.2 / np.cos(0.1))
    assert height[1] == 0.0


def test_spatial_thinning_keeps_observed_points() -> None:
    points = np.asarray(((0.001, 0.001, 0.2), (0.009, 0.009, 0.21), (0.2, 0.2, 0.2)))

    selected = _spatially_thin_contacts(points, 0.05)

    assert len(selected) == 2
    assert all(any(np.array_equal(point, source) for source in points) for point in selected)


def test_axis_aligned_shell_omits_wall_between_equal_adjacent_boxes() -> None:
    record = {
        "terrain": {
            "boxes": [
                {"pos": [-0.5, 0.0, 0.1], "size": [0.5, 1.0, 0.1], "yaw": 0.0, "pitch": 0.0},
                {"pos": [0.5, 0.0, 0.1], "size": [0.5, 1.0, 0.1], "yaw": 0.0, "pitch": 0.0},
            ]
        }
    }
    frame = {
        "center": np.asarray((0.0, 0.0)),
        "traversal": np.asarray((1.0, 0.0)),
        "lateral": np.asarray((0.0, 1.0)),
    }

    faces, _, _ = _axis_aligned_surface_faces(record, frame, METHODS[1])

    vertical_faces = [face for face in faces if np.ptp(face[:, 2]) > 0.0]
    assert not any(np.allclose(face[:, 0], 0.0) for face in vertical_faces)
