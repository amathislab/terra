from __future__ import annotations

import numpy as np
import pytest

from terra.figures.methodology import (
    _farthest_point_sample,
    _fit_metrics,
    _inside_safe_frame,
    _minimum_spanning_edges,
    _motion_sweep_evidence,
    _projection_bounds,
    load_methodology_spec,
)


def _write(path, frames: str = "[10, 20, 30]", highlight: int = 20) -> None:
    path.write_text(
        f'''version = 1
[figure]
name = "method"
width = 2100
height = 2400
panel_width = 1000
panel_height = 900
[motion]
name = "Study/Subject/Trial"
frames = {frames}
highlight_frame = {highlight}
target_frame_offset = -1
[camera]
azimuth = 38.0
elevation = -25.0
distance = 4.8
lookat = [0.0, 0.0, 0.72]
field_of_view = 42.0
[assets]
smpl_model = "models/smpl"
fitted_shape = "shape.pkl"
terrain_report = "terrain.json"
'''
    )


def test_load_methodology_spec_resolves_paths_and_frames(tmp_path):
    path = tmp_path / "figure.toml"
    _write(path)

    spec = load_methodology_spec(path)

    assert spec.motion == "Study/Subject/Trial"
    assert spec.source_frames == (10, 20, 30)
    assert spec.highlight_frame == 20
    assert spec.target_frame_offset == -1
    assert spec.smpl_model == (tmp_path / "models/smpl").resolve()
    assert spec.camera_lookat == (0.0, 0.0, 0.72)
    assert spec.include_text is True
    assert spec.transparent_background is False
    assert len(spec.panel_cameras) == 4
    assert spec.panel_cameras[0].distance == 4.8
    assert spec.panel_cameras[0].projection == "perspective"
    assert spec.panel_cameras[0].ortho_scale is None


def test_load_methodology_spec_allows_panel_camera_overrides(tmp_path):
    path = tmp_path / "figure.toml"
    _write(path)
    path.write_text(
        path.read_text().replace(
            "[assets]",
            (
                "[camera.c]\ndistance = 3.1\nlookat = [0.1, 0.2, 0.4]\n"
                "field_of_view = 36.0\nprojection = \"orthographic\"\n"
                "ortho_scale = 1.8\n[assets]"
            ),
        )
    )

    spec = load_methodology_spec(path)

    assert spec.panel_cameras[2].distance == 3.1
    assert spec.panel_cameras[2].lookat == (0.1, 0.2, 0.4)
    assert spec.panel_cameras[2].field_of_view == 36.0
    assert spec.panel_cameras[2].projection == "orthographic"
    assert spec.panel_cameras[2].ortho_scale == 1.8
    assert spec.panel_cameras[1].distance == 4.8


def test_load_methodology_spec_allows_text_free_composition(tmp_path):
    path = tmp_path / "figure.toml"
    _write(path)
    path.write_text(
        path.read_text().replace(
            'name = "method"',
            'name = "method"\ninclude_text = false\ntransparent_background = true',
        )
    )

    spec = load_methodology_spec(path)

    assert spec.include_text is False
    assert spec.transparent_background is True


def test_load_methodology_spec_requires_highlight_in_frames(tmp_path):
    path = tmp_path / "figure.toml"
    _write(path, highlight=40)

    with pytest.raises(ValueError, match="highlight_frame must be included"):
        load_methodology_spec(path)


def test_load_methodology_spec_requires_strictly_increasing_frames(tmp_path):
    path = tmp_path / "figure.toml"
    _write(path, frames="[10, 10, 30]", highlight=10)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_methodology_spec(path)


def test_farthest_point_sample_is_deterministic_and_bounded():
    points = np.asarray([[float(x), float(y), 0.0] for x in range(5) for y in range(4)])

    first = _farthest_point_sample(points, 7)
    second = _farthest_point_sample(points, 7)

    assert first.shape == (7, 3)
    np.testing.assert_array_equal(first, second)
    assert all(any(np.array_equal(point, candidate) for candidate in points) for point in first)


def test_motion_sweep_evidence_preserves_continuous_landmark_paths():
    names = (
        "Pelvis",
        "L_Knee",
        "R_Knee",
        "L_Ankle",
        "R_Ankle",
        "L_Toe",
        "R_Toe",
        "L_Wrist",
        "R_Wrist",
    )
    joints = np.zeros((13, len(names), 3), dtype=float)
    joints[:, :, 0] = np.arange(13)[:, None]

    points, segments = _motion_sweep_evidence(
        joints,
        {name: index for index, name in enumerate(names)},
        (0, 12),
    )

    assert len(points) == len(names)
    assert len(segments) == 2 * len(names)
    np.testing.assert_array_equal(segments[0][0], np.asarray((0.0, 0.0, 0.0)))
    np.testing.assert_array_equal(segments[0][1], np.asarray((6.0, 0.0, 0.0)))


def test_interaction_backbone_connects_separated_landmark_groups():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.1, 0.0, 0.0),
        )
    )

    edges = _minimum_spanning_edges(points)

    assert len(edges) == len(points) - 1
    assert any(left < 2 <= right for left, right in edges)


def test_projection_audit_uses_normalized_safe_bounds():
    metadata = {
        "width": 100,
        "height": 100,
        "camera": {
            "position": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            "fovy_degrees": 90.0,
        },
    }
    points = np.asarray(((-0.5, -0.5, 1.0), (0.5, 0.5, 1.0)))

    bounds = _projection_bounds(points, metadata)

    np.testing.assert_allclose(bounds, (0.25, 0.25, 0.75, 0.75))
    assert _inside_safe_frame(bounds, 0.20)
    assert not _inside_safe_frame(bounds, 0.30)


def test_projection_audit_supports_orthographic_cameras():
    metadata = {
        "width": 200,
        "height": 100,
        "camera": {
            "position": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            "fovy_degrees": 90.0,
            "projection": "orthographic",
            "ortho_scale": 2.0,
        },
    }
    points = np.asarray(((-0.5, -0.25, 1.0), (0.5, 0.25, 5.0)))

    bounds = _projection_bounds(points, metadata)

    np.testing.assert_allclose(bounds, (0.25, 0.25, 0.75, 0.75))


def test_fit_metrics_preserve_reconstruction_quantities():
    report = {
        "terrain": {
            "provenance": {"joint_surface_offsets_m": {"R_Ankle": 0.031}}
        },
        "fit": {
            "stair_flight": {
                "height_model": {
                    "shared_riser": 0.2,
                    "raw_heights": [0.0, 0.21],
                    "fitted_heights": [0.0, 0.2],
                    "rms_adjustment": 0.007,
                }
            },
            "model_scores": {"stair_flight": {"raised_contact_error_max": 0.006}},
        }
    }

    metrics = _fit_metrics(report)

    assert metrics.shared_riser == 0.2
    assert metrics.raw_heights == (0.0, 0.21)
    assert metrics.fitted_heights == (0.0, 0.2)
    assert metrics.rms_adjustment == 0.007
    assert metrics.max_support_residual == 0.006
    assert metrics.sole_offset == 0.031
