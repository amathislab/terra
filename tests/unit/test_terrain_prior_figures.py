from __future__ import annotations

import json
import math

import numpy as np
import pytest

from terra.figures.terrain_priors import (
    PRIOR_NAMES,
    _camera_payload,
    _clean_transparent_film,
    build_prior_bundle,
    chair_primitives,
    ramp_primitives,
    staircase_primitives,
)
from terra.terrain.ramps import RAMP_SLOPE_RANGE
from terra.terrain.seats import SEAT_SIZE_PRIOR
from terra.terrain.stairs import STAIR_MIN_LEVELS


def test_all_terrain_prior_bundles_are_rgba_shadow_catcher_scenes():
    for name in PRIOR_NAMES:
        bundle = build_prior_bundle(name, width=720, height=600, samples=32)
        metadata = json.loads(str(bundle["metadata"]))

        assert metadata["transparent_background"] is True
        assert metadata["shadow_catcher"]["z"] < 0.0
        assert metadata["camera"]["projection"] == "orthographic"
        assert bundle["boxes_position"].shape[0] == len(bundle["boxes_size"])
        assert bundle["boxes_pitched"].shape == (len(bundle["boxes_size"]),)


def test_chair_autoframe_keeps_extra_room_for_its_tall_shadow():
    chair = json.loads(str(build_prior_bundle("chairs")["metadata"]))
    boxes = json.loads(str(build_prior_bundle("boxes")["metadata"]))
    standard_margin = _camera_payload(chair_primitives(), width=1200, height=1000)

    assert chair["camera"]["ortho_scale"] > standard_margin["ortho_scale"]
    assert chair["camera"]["ortho_scale"] > 1.0
    assert boxes["camera"]["ortho_scale"] > chair["camera"]["ortho_scale"]


def test_ramp_uses_a_true_pitched_wedge_with_an_allowed_slope():
    ramp, landing = ramp_primitives()
    pitch = math.degrees(math.atan2(ramp.rotation[0][2], ramp.rotation[0][0]))

    assert ramp.pitched is True
    assert RAMP_SLOPE_RANGE[0] <= abs(pitch) <= RAMP_SLOPE_RANGE[1]
    assert landing.pitched is False
    assert landing.position[2] + landing.half_size[2] > 0.5


def test_staircase_is_contiguous_with_a_shared_riser():
    stairs = staircase_primitives()
    tops = np.asarray([item.position[2] + item.half_size[2] for item in stairs])
    left = np.asarray([item.position[0] - item.half_size[0] for item in stairs])
    right = np.asarray([item.position[0] + item.half_size[0] for item in stairs])

    assert len(stairs) >= STAIR_MIN_LEVELS
    np.testing.assert_allclose(np.diff(tops), np.diff(tops)[0])
    np.testing.assert_allclose(left[1:], right[:-1], atol=1e-12)


def test_chair_is_exactly_one_inferred_support_box():
    primitives = chair_primitives()

    assert len(primitives) == 1
    assert primitives[0].role == "inferred solid seat support"
    assert primitives[0].half_size[:2] == pytest.approx(SEAT_SIZE_PRIOR)
    assert primitives[0].kind == "terrain"


def test_unknown_terrain_prior_is_rejected():
    with pytest.raises(ValueError, match="unknown terrain prior"):
        build_prior_bundle("boulders")


def test_transparent_film_cleanup_removes_only_isolated_border_alpha(tmp_path):
    from PIL import Image

    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[0, 7] = (255, 255, 255, 48)
    pixels[3:6, 0:3] = (255, 255, 255, 48)
    path = tmp_path / "film.png"
    Image.fromarray(pixels, mode="RGBA").save(path)

    _clean_transparent_film(path)

    alpha = np.asarray(Image.open(path).convert("RGBA").getchannel("A"))
    assert alpha[0, 7] == 0
    assert np.all(alpha[3:6, 0:3] == 48)
