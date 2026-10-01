from __future__ import annotations

import numpy as np
import pytest

from terra.constants import MYOFULLBODY_SITE_CALIBRATION
from terra.figures.rl_scene import correspondence_dots, heightmap_offsets, reference_edges


def test_heightmap_offsets_match_locomujoco_row_major_convention():
    offsets = heightmap_offsets(3, 3, 0.1)

    np.testing.assert_allclose(
        offsets,
        (
            (-0.1, -0.1),
            (-0.1, 0.0),
            (-0.1, 0.1),
            (0.0, -0.1),
            (0.0, 0.0),
            (0.0, 0.1),
            (0.1, -0.1),
            (0.1, 0.0),
            (0.1, 0.1),
        ),
    )


def test_heightmap_forward_offset_shifts_only_the_local_x_axis():
    base = heightmap_offsets(11, 11, 0.1)
    shifted = heightmap_offsets(11, 11, 0.1, 0.5)

    np.testing.assert_allclose(shifted[:, 0] - base[:, 0], 0.5)
    np.testing.assert_allclose(shifted[:, 1], base[:, 1])


def test_reference_skeleton_connects_all_calibrated_sites_without_cycles():
    edges = reference_edges()
    nodes = set(range(len(MYOFULLBODY_SITE_CALIBRATION)))
    reached = {0}
    while True:
        expanded = reached | {
            right if left in reached else left for left, right in edges if left in reached or right in reached
        }
        if expanded == reached:
            break
        reached = expanded

    assert len(edges) == len(nodes) - 1
    assert reached == nodes


def test_correspondence_dots_omit_endpoints_and_preserve_pair_ids():
    current = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    target = np.asarray(((0.2, 0.0, 0.0), (1.0, 0.1, 0.0)))

    dots, pairs = correspondence_dots(current, target, spacing=0.055)

    assert set(pairs.tolist()) == {0, 1}
    assert np.all((dots[:, 0] > 0.0) & (dots[:, 0] < 1.2))
    for pair_index, (start, end) in enumerate(zip(current, target, strict=True)):
        pair_dots = dots[pairs == pair_index]
        assert len(pair_dots) >= 2
        assert not np.any(np.all(np.isclose(pair_dots, start), axis=1))
        assert not np.any(np.all(np.isclose(pair_dots, end), axis=1))


@pytest.mark.parametrize("rows,cols,resolution", [(0, 3, 0.1), (3, 0, 0.1), (3, 3, 0.0)])
def test_invalid_heightmap_shapes_are_rejected(rows: int, cols: int, resolution: float):
    with pytest.raises(ValueError, match="heightmap"):
        heightmap_offsets(rows, cols, resolution)
