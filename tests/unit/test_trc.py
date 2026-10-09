"""Tests for shared TRC parsing and marker-archive preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from terra.trc import load_trc, prepare_trc_marker_archive


def _write_trc(path: Path, *, units: str = "mm") -> None:
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\ttrial.trc\n"
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
        f"100\t100\t2\t2\t{units}\n"
        "Frame#\tTime\tLANK\t\t\tRANK\t\t\n"
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n"
        "1\t0.00\t1000\t2000\t3000\t0\t0\t0\n"
        "2\t0.01\t4000\t5000\t6000\t7000\t8000\t9000\n"
    )


def test_load_trc_normalizes_units_and_y_up_coordinates(tmp_path):
    source = tmp_path / "walk.trc"
    _write_trc(source)

    motion = load_trc(source)

    assert motion.labels == ("LANK", "RANK")
    assert motion.fps == 100.0
    np.testing.assert_array_equal(motion.frame_numbers, [1, 2])
    np.testing.assert_allclose(motion.times, [0.0, 0.01])
    np.testing.assert_allclose(motion.positions[0, 0], [1.0, -3.0, 2.0])
    assert np.isnan(motion.positions[0, 1]).all()


def test_load_trc_can_preserve_z_up_coordinates(tmp_path):
    source = tmp_path / "walk.trc"
    _write_trc(source)

    motion = load_trc(source, up_axis="z")

    np.testing.assert_allclose(motion.positions[1, 1], [7.0, 8.0, 9.0])


def test_load_trc_rejects_unknown_vertical_axis(tmp_path):
    source = tmp_path / "walk.trc"
    _write_trc(source)

    with pytest.raises(ValueError, match="trc_up_axis"):
        load_trc(source, up_axis="x")  # type: ignore[arg-type]


def test_prepare_trc_marker_archive_is_content_addressed_and_musclemimic_compatible(tmp_path):
    source = tmp_path / "walk.trc"
    _write_trc(source)

    first = prepare_trc_marker_archive(source, tmp_path / "cache")
    second = prepare_trc_marker_archive(source, tmp_path / "cache")

    assert first == second
    with np.load(first, allow_pickle=False) as archive:
        assert {"positions", "labels", "fps"} <= set(archive.files)
        assert archive["positions"].shape == (2, 2, 3)
        assert archive["labels"].tolist() == ["LANK", "RANK"]
        assert float(archive["fps"]) == 100.0

    from musclemimic.web_viewer.c3d.markers import load_marker_trajectory

    positions, labels, fps = load_marker_trajectory(first)
    assert positions.shape == (2, 2, 3)
    assert labels == ["LANK", "RANK"]
    assert fps == 100.0
