import mujoco
import pytest

from terra.rl.collision_profiles import _disabled_pairs_for_spec, merge_disabled_contact_pairs


def _collision_spec():
    return mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body name="thorax"><geom name="torso" size="0.1"/></body>
            <body name="humerus_r"><geom name="arm" size="0.1"/></body>
            <body name="femur_r"><geom name="right_leg" size="0.1"/></body>
            <body name="femur_l"><geom name="left_leg" size="0.1"/></body>
          </worldbody>
          <contact>
            <pair geom1="arm" geom2="torso"/>
            <pair geom1="arm" geom2="right_leg"/>
            <pair geom1="right_leg" geom2="left_leg"/>
          </contact>
        </mujoco>
        """
    )


def test_lower_body_only_removes_upper_pairs_but_preserves_leg_collision():
    assert _disabled_pairs_for_spec(_collision_spec(), "lower_body_only") == [
        ["arm", "torso"],
        ["arm", "right_leg"],
    ]


def test_none_removes_every_explicit_self_collision_pair():
    disabled = _disabled_pairs_for_spec(_collision_spec(), "none")

    assert len(disabled) == 3
    assert ["right_leg", "left_leg"] in disabled


def test_invalid_self_collision_mode_is_rejected():
    with pytest.raises(ValueError, match="self_collision_mode"):
        _disabled_pairs_for_spec(_collision_spec(), "upper")


def test_targeted_and_profile_pairs_are_deduplicated(monkeypatch):
    monkeypatch.setattr(
        "terra.rl.collision_profiles.disabled_self_collision_pairs",
        lambda _mode: [["arm", "torso"], ["arm", "leg"]],
    )

    assert merge_disabled_contact_pairs(
        [["torso", "arm"]],
        self_collision_mode="lower_body_only",
    ) == [["torso", "arm"], ["arm", "leg"]]
