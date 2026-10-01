from __future__ import annotations

import numpy as np

from terra.datasets.prism.adapter import observed_support_points, observed_support_xy


def synthetic_take():
    left = np.zeros((12, 2), dtype=bool)
    left[1:5, 0] = True
    left[6:10, 1] = True  # one-frame dropout should be bridged
    right = np.zeros((12, 2), dtype=bool)
    right[2:8] = True
    cop = np.column_stack((np.arange(12), np.zeros(12), np.zeros(12)))
    return {
        "info": {"data_info": {"fps": 10}},
        "insole": {
            "L_Foot": {"contacts": left, "CoP_world": cop},
            "R_Foot": {"contacts": right, "CoP_world": cop + np.array([0, 1, 0])},
        },
    }


def test_observed_support_uses_only_contact_frames():
    xy = observed_support_xy(synthetic_take())
    assert xy.shape == (14, 2)
    assert {tuple(point) for point in xy} == {
        *((frame, 0) for frame in (*range(1, 5), *range(6, 10))),
        *((frame, 1) for frame in range(2, 8)),
    }

    points = observed_support_points(synthetic_take())
    assert points.shape == (14, 3)
    np.testing.assert_array_equal(points[:, :2], xy)
