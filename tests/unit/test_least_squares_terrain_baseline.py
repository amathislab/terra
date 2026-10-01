"""Tests for the inferred-contact least-squares plane baseline."""

from __future__ import annotations

import numpy as np
import pytest

from terra.benchmarking.terrain import LeastSquaresPlaneConfig, fit_least_squares_contact_plane


def _stationary(left, right, *, frames=12, fps=20.0, config=None):
    joints = np.empty((frames, 2, 3), dtype=float)
    joints[:, 0] = left
    joints[:, 1] = right
    return fit_least_squares_contact_plane(
        joints,
        ("L_Toe", "R_Toe"),
        fps,
        config=config,
        input_stage="unit_test",
    )


def test_two_stationary_contacts_fit_one_affine_plane_without_terra_family_logic():
    terrain, report = _stationary((0.0, 0.0, 0.0), (1.0, 0.0, 0.2))

    assert report["model"] == "least_squares_contact_plane"
    assert report["n_contact_events"] == 2
    assert report["n_boxes"] == 1
    assert report["fit"]["gradient_xy"] == pytest.approx([0.2, 0.0])
    assert report["fit"]["slope_deg"] == pytest.approx(np.degrees(np.arctan(0.2)))
    assert report["fit"]["height_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert report["geometry"]["output"] == "one_finite_affine_plane"
    assert report["unsupported_support_kinds"] == ["pelvis"]
    assert len(terrain.boxes) == 1
    assert terrain.boxes[0].top_plane_at(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert terrain.boxes[0].top_plane_at(1.0, 0.0) == pytest.approx(0.2, abs=1e-12)
    assert [item["link"] for item in report["support_intervals"]] == ["L_Toe", "R_Toe"]
    assert all(
        item["evidence_stage"] == "inferred_slow_local_low_before_plane_fit" for item in report["support_intervals"]
    )


def test_floor_contacts_use_deterministic_flat_fallback_but_keep_inferred_intervals():
    terrain, report = _stationary((0.0, 0.0, 0.0), (0.4, 0.0, 0.0))

    assert terrain.boxes == ()
    assert report["model"] == "least_squares_contact_plane"
    assert report["n_contact_events"] == 2
    assert report["n_boxes"] == 0
    assert report["fit"] is not None
    assert report["geometry"]["output"] == "implicit_flat_floor"
    assert "nowhere above" in report["geometry"]["fallback_reason"]


def test_no_slow_contacts_falls_back_to_floor_without_fabricating_support():
    frames = 10
    joints = np.zeros((frames, 2, 3), dtype=float)
    joints[:, :, 0] = np.arange(frames)[:, None] * 0.1

    terrain, report = fit_least_squares_contact_plane(
        joints,
        ("L_Toe", "R_Toe"),
        10.0,
    )

    assert terrain.boxes == ()
    assert report["fit"] is None
    assert report["support_intervals"] == []
    assert report["geometry"]["fallback_reason"] == "no inferred contact events"


@pytest.mark.parametrize(("fps", "frames"), [(50.0, 5), (100.0, 10)])
def test_minimum_contact_duration_is_source_rate_invariant(fps, frames):
    joints = np.zeros((frames, 2, 3), dtype=float)
    _terrain, report = fit_least_squares_contact_plane(joints, ("L_Toe", "R_Toe"), fps)
    assert report["n_contact_events"] == 2
    assert report["contact_detection"]["L_Toe"]["minimum_run_frames"] == frames

    too_short = joints[:-1]
    _terrain, report = fit_least_squares_contact_plane(too_short, ("L_Toe", "R_Toe"), fps)
    assert report["n_contact_events"] == 0


def test_line_degenerate_fit_is_minimum_norm_and_slope_cap_is_explicit():
    config = LeastSquaresPlaneConfig(maximum_slope_deg=30.0)
    terrain, report = _stationary(
        (0.0, 0.0, 0.0),
        (0.001, 0.0, 1.0),
        config=config,
    )

    assert len(terrain.boxes) == 1
    assert report["fit"]["design_rank"] == 2
    assert report["fit"]["raw_slope_deg"] > 80.0
    assert report["fit"]["slope_was_capped"] is True
    assert report["fit"]["slope_deg"] == pytest.approx(30.0)
    assert report["fit"]["gradient_xy"][1] == pytest.approx(0.0)


def test_each_contact_event_has_equal_weight_regardless_of_duration():
    # L_Toe supplies one 5-frame event at z=0. R_Toe supplies one 15-frame event at
    # z=0.2. A frame-weighted fit would report a mean height of 0.15; event OLS reports
    # 0.10 at the centred intercept.
    joints = np.zeros((20, 2, 3), dtype=float)
    joints[:, 0, 0] = np.arange(20) * 0.1
    joints[:5, 0] = (0.0, 0.0, 0.0)
    joints[:, 1] = (1.0, 0.0, 0.2)
    terrain, report = fit_least_squares_contact_plane(
        joints,
        ("L_Toe", "R_Toe"),
        10.0,
        config=LeastSquaresPlaneConfig(local_window_s=0.1),
    )

    assert len(terrain.boxes) == 1
    assert report["n_contact_events"] == 2
    assert report["fit"]["intercept_at_center_m"] == pytest.approx(0.1)
    assert report["fit"]["n_equal_weight_events"] == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"velocity_threshold_m_s": 0.0}, "velocity_threshold"),
        ({"minimum_contact_duration_s": True}, "minimum_contact_duration"),
        ({"maximum_slope_deg": 60.0}, "less than 60"),
    ],
)
def test_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LeastSquaresPlaneConfig(**kwargs)


def test_motion_contract_is_strict_but_absent_requested_link_is_reported():
    joints = np.zeros((8, 1, 3), dtype=float)
    terrain, report = fit_least_squares_contact_plane(joints, ["L_Toe"], 20.0)
    assert terrain.boxes == ()
    assert report["contact_detection"]["R_Toe"]["available"] is False
    assert "absent" in report["warnings"][0]

    with pytest.raises(ValueError, match="shape"):
        fit_least_squares_contact_plane(np.zeros((8, 3)), ["L_Toe"], 20.0)
