from __future__ import annotations

import json

import numpy as np
import pytest

from terra.figures.smpl_assets import (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    SMPLAssetGeometry,
    apose_axis_angle,
    build_smpl_apose_bundle,
    smpl_to_blender_coordinates,
)


def _geometry() -> SMPLAssetGeometry:
    return SMPLAssetGeometry(
        vertices=np.asarray(((-0.6, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.0, 1.8)), dtype=np.float32),
        faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        joints=np.zeros((22, 3), dtype=np.float32),
        pose_axis_angle=apose_axis_angle(),
        shoulder_angle_degrees=45.0,
    )


def test_apose_rotates_only_the_shoulders_symmetrically():
    pose = apose_axis_angle(45.0)
    nonzero = np.flatnonzero(pose)

    assert nonzero.tolist() == [LEFT_SHOULDER * 3 + 2, RIGHT_SHOULDER * 3 + 2]
    assert pose[LEFT_SHOULDER * 3 + 2] == pytest.approx(-pose[RIGHT_SHOULDER * 3 + 2])
    assert abs(pose[LEFT_SHOULDER * 3 + 2]) == pytest.approx(np.pi / 4)


def test_smpl_coordinates_become_front_facing_z_up_and_grounded():
    points = np.asarray(((1.0, -2.0, 3.0), (-1.0, 4.0, -5.0)))

    converted = smpl_to_blender_coordinates(points, ground=-2.0)

    np.testing.assert_allclose(converted, ((1.0, -3.0, 0.0), (-1.0, 5.0, 6.0)))


def test_smpl_bundle_is_an_opaque_front_orthographic_rgba_asset():
    bundle = build_smpl_apose_bundle(_geometry(), width=600, height=750, samples=32)
    metadata = json.loads(str(bundle["metadata"]))

    assert metadata["transparent_background"] is True
    assert metadata["camera"]["projection"] == "orthographic"
    assert metadata["camera"]["forward"] == [0.0, 1.0, 0.0]
    assert metadata["shadow_catcher"]["z"] < 0.0
    assert bundle["source_vertices"].shape == (1, 3, 3)
    assert bundle["source_rgba"][0, 3] == 1.0


def test_apose_angle_validation_is_fail_closed():
    with pytest.raises(ValueError, match="shoulder angle"):
        apose_axis_angle(5.0)
