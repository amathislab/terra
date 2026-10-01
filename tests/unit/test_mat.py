"""Tests for schema-driven MATLAB marker extraction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from terra.datasets.darmstadt import DARMSTADT_MAT_MARKER_SCHEMA
from terra.datasets.marker_fitting import DARMSTADT_MARKERS
from terra.mat import extract_mat_markers, load_mat_markers, prepare_mat_marker_archive


def _tensor_schema() -> dict[str, object]:
    return {
        "version": 1,
        "positions_path": "markers",
        "axis_order": "tmc",
        "labels_path": "marker_labels",
        "fps_path": "marker_rate",
        "units": "mm",
        "axes": ["x", "-z", "y"],
    }


def test_load_mat_markers_uses_declared_tensor_layout_units_and_axes(tmp_path):
    source = tmp_path / "markers.mat"
    positions = np.asarray(
        [
            [[1000, 2000, 3000], [0, 0, 0]],
            [[4000, 5000, 6000], [7000, 8000, 9000]],
        ],
        dtype=np.float32,
    )
    savemat(
        source,
        {
            "markers": positions,
            "marker_labels": np.asarray(["LANK", "RANK"], dtype=object),
            "marker_rate": np.asarray(100.0),
        },
    )

    motion = load_mat_markers(source, _tensor_schema())

    assert motion.labels == ("LANK", "RANK")
    assert motion.fps == 100.0
    np.testing.assert_allclose(motion.positions[0, 0], [1.0, -3.0, 2.0])
    assert np.isnan(motion.positions[0, 1]).all()


def test_darmstadt_schema_loads_scipy_nested_configuration_and_trial(tmp_path):
    source = tmp_path / "Marker1.mat"
    configurations = np.empty((1, 2), dtype=object)
    for configuration_index in range(2):
        trials = np.empty((1, 2), dtype=object)
        for trial_index in range(2):
            trials[0, trial_index] = {
                field: np.full(
                    (4, 3),
                    marker_index + 10 * trial_index + 100 * configuration_index,
                    dtype=np.float32,
                )
                for marker_index, (_label, field) in enumerate(DARMSTADT_MARKERS)
            }
        configurations[0, configuration_index] = trials
    savemat(source, {"Marker": configurations, "Marker_fs": np.asarray(200.0)})

    motion = load_mat_markers(
        source,
        DARMSTADT_MAT_MARKER_SCHEMA,
        selectors={"configuration": 1, "trial": 1},
    )

    assert motion.labels == tuple(label for label, _field in DARMSTADT_MARKERS)
    assert motion.positions.shape == (4, len(DARMSTADT_MARKERS), 3)
    assert motion.fps == 200.0
    np.testing.assert_allclose(motion.positions[:, 0], 110.0)


def test_darmstadt_schema_requires_both_nested_selectors():
    with pytest.raises(ValueError, match="selector 'configuration'"):
        extract_mat_markers(
            {"Marker": np.asarray([]), "Marker_fs": 100.0},
            DARMSTADT_MAT_MARKER_SCHEMA,
        )


def test_prepare_mat_marker_archive_is_content_addressed_and_musclemimic_compatible(tmp_path):
    source = tmp_path / "markers.mat"
    savemat(
        source,
        {
            "markers": np.ones((3, 2, 3), dtype=np.float32),
            "marker_labels": np.asarray(["LANK", "RANK"], dtype=object),
            "marker_rate": np.asarray(120.0),
        },
    )

    first = prepare_mat_marker_archive(source, _tensor_schema(), tmp_path / "cache")
    second = prepare_mat_marker_archive(source, _tensor_schema(), tmp_path / "cache")

    assert first == second
    from musclemimic.web_viewer.c3d.markers import load_marker_trajectory

    positions, labels, fps = load_marker_trajectory(first)
    assert positions.shape == (3, 2, 3)
    assert labels == ["LANK", "RANK"]
    assert fps == 120.0


def test_mat_schema_rejects_ambiguous_marker_storage(tmp_path):
    source = tmp_path / "markers.mat"
    source.touch()
    schema = {
        "positions_path": "markers",
        "marker_fields": {"LANK": "ankle"},
        "fps": 100,
    }

    with pytest.raises(ValueError, match="exactly one"):
        load_mat_markers(source, schema)
