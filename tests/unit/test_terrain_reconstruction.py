"""Tests for box terrain and its reconstruction from motion.

These guard silent failures rather than crashes. A terrain that is a few centimetres off,
or one that comes back empty because a gate rejected every raised contact, produces no
error anywhere: retargeting still converges, playback still looks like walking, and the
only symptom is a policy that cannot reproduce the motion. So the properties pinned here
are the ones with no downstream alarm - the height estimator's cancellation of the
joint-inside-foot offset, the stance detector's independence from the height datum, and
the agreement between the fitted geometry and the geoms the environment actually builds.
"""

import json

import numpy as np
import pytest

from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
from loco_mujoco.core.terrain.boxes import SEAT_GEOM_PREFIX
from terra.terrain import (
    DEFAULT_CONTACT_MARGIN,
    PELVIS_SEAT_OFFSET,
    SEAT_SPLIT_GAP,
    STAIR_SOLE_HALF,
    StanceEvent,
    calibrate_neutral_foot_pitch,
    cluster_levels,
    detect_seat_rests,
    detect_stance_events,
    fit_stair_flight,
    fit_terrain_from_motion,
    joint_surface_offsets,
    paired_sole_offsets,
    validate_terrain,
)
from terra.terrain.stance import _level_height

JOINTS = ["Pelvis", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]
FPS = 100.0

#: Offset of each contact joint's centre above the surface it rests on. Real, and the
#: reason absolute joint heights cannot be used as surface heights.
JOINT_OFFSET = {"L_Toe": 0.02, "R_Toe": 0.02, "L_Ankle": 0.09, "R_Ankle": 0.09}


def synth_walk(beam_height=0.10, beam_x=(-0.9, 0.9), n_beam_steps=4, stance_frames=60, swing_frames=40):
    """A walk along +x that steps up onto a beam, crosses it, and steps down.

    Feet alternate; each foot is stationary through stance and interpolates during swing.
    Heights carry `JOINT_OFFSET`, so anything that reads a surface height off a joint
    centre without cancelling the offset gets the wrong answer.
    """
    xs = [-1.8, -1.35]
    xs += list(np.linspace(beam_x[0] + 0.05, beam_x[1] - 0.05, n_beam_steps))
    xs += [1.35, 1.8]
    on_beam = [False, False] + [True] * n_beam_steps + [False, False]

    frames = []
    foot_state = {  # (x, surface_z) per foot, updated as it steps
        "L": (xs[0], 0.0),
        "R": (xs[0], 0.0),
    }
    for i, (x, beam) in enumerate(zip(xs, on_beam, strict=True)):
        side = "L" if i % 2 == 0 else "R"
        target = (x, beam_height if beam else 0.0)
        prev = foot_state[side]
        for f in range(swing_frames):  # swing: lift, translate, land
            a = (f + 1) / swing_frames
            lift = 0.18 * np.sin(np.pi * a)
            foot_state[side] = (
                prev[0] + a * (target[0] - prev[0]),
                prev[1] + a * (target[1] - prev[1]) + lift,
            )
            frames.append(dict(foot_state))
        foot_state[side] = target
        for _ in range(stance_frames):
            frames.append(dict(foot_state))

    out = np.zeros((len(frames), len(JOINTS), 3))
    for t, st in enumerate(frames):
        for side in ("L", "R"):
            x, z = st[side]
            y = 0.1 if side == "L" else -0.1
            out[t, JOINTS.index(f"{side}_Toe")] = [x + 0.08, y, z + JOINT_OFFSET[f"{side}_Toe"]]
            out[t, JOINTS.index(f"{side}_Ankle")] = [x, y, z + JOINT_OFFSET[f"{side}_Ankle"]]
        out[t, JOINTS.index("Pelvis")] = [0.5 * (st["L"][0] + st["R"][0]), 0.0, 0.9 + max(st["L"][1], st["R"][1])]
    return out


def profile_walk(height_of, xs=None):
    """A walk along +x whose footfalls take their height from `height_of(x)`.

    The counterpart of :func:`synth_walk` for surfaces defined by a *profile* rather than by
    a slab: pass a ramp and a staircase of the same rise over the same ground and the only
    thing that differs is whether the heights lie on a line.
    """
    if xs is None:
        xs = np.arange(-1.2, 2.2, 0.30)
    frames, state = [], {"L": (xs[0], 0.0), "R": (xs[0], 0.0)}
    for i, x in enumerate(xs):
        side = "L" if i % 2 == 0 else "R"
        prev, target = state[side], (x, height_of(x))
        for f in range(20):
            a = (f + 1) / 20
            state[side] = (
                prev[0] + a * (target[0] - prev[0]),
                prev[1] + a * (target[1] - prev[1]) + 0.15 * np.sin(np.pi * a),
            )
            frames.append(dict(state))
        state[side] = target
        frames.extend([dict(state)] * 40)

    out = np.zeros((len(frames), len(JOINTS), 3))
    for t, st in enumerate(frames):
        for side, y in (("L", 0.1), ("R", -0.1)):
            x, z = st[side]
            out[t, JOINTS.index(f"{side}_Toe")] = [x + 0.08, y, z + JOINT_OFFSET[f"{side}_Toe"]]
            out[t, JOINTS.index(f"{side}_Ankle")] = [x, y, z + JOINT_OFFSET[f"{side}_Ankle"]]
        out[t, JOINTS.index("Pelvis")] = [0.5 * (st["L"][0] + st["R"][0]), 0.0, 1.0 + max(st["L"][1], st["R"][1])]
    return out


#: ~18 deg of continuous incline, and a staircase climbing the same 0.6 m over the same run.
RAMP_HEIGHT = lambda x: float(np.clip(x, 0.0, 1.8)) / 3.0  # noqa: E731
STAIR_HEIGHT = lambda x: 0.15 * np.floor(np.clip(x, 0.0, 1.8) / 0.45)  # noqa: E731


@pytest.mark.parametrize("name", ["stair_flight", "ramp", "seat"])
def test_fit_rejects_invalid_model_modes_without_stance_evidence(name):
    motion = np.zeros((2, len(JOINTS), 3))

    with pytest.raises(ValueError, match=rf"{name} must be 'auto' or 'off'"):
        fit_terrain_from_motion(motion, JOINTS, FPS, **{name: "invalid"})


def test_fit_rejects_invalid_family_evidence_mode_without_stance_evidence():
    with pytest.raises(ValueError, match="family_evidence_mode must be one of"):
        fit_terrain_from_motion(
            np.zeros((2, len(JOINTS), 3)),
            JOINTS,
            FPS,
            family_evidence_mode="invalid",
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("shape", r"shape \(T>=2, J, 3\)"),
        ("frames", r"shape \(T>=2, J, 3\)"),
        ("names", "names for a joint array"),
        ("duplicate_names", "duplicate names"),
        ("finite", "only finite values"),
        ("fps", "positive and finite"),
    ],
)
def test_fit_rejects_invalid_motion_contract(failure, message):
    motion = np.zeros((2, len(JOINTS), 3))
    names = JOINTS
    fps = FPS
    if failure == "shape":
        motion = np.zeros((2, len(JOINTS)))
    elif failure == "frames":
        motion = motion[:1]
    elif failure == "names":
        names = JOINTS[:-1]
    elif failure == "duplicate_names":
        names = [*JOINTS[:-1], JOINTS[-2]]
    elif failure == "finite":
        motion[0, 0, 0] = np.nan
    else:
        fps = 0.0

    with pytest.raises(ValueError, match=message):
        fit_terrain_from_motion(motion, names, fps)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("contact_margin", (-0.01, 0.1)),
        ("max_extension", (0.1, np.inf)),
    ],
)
def test_fit_rejects_nonphysical_geometry_pairs(name, value):
    with pytest.raises(ValueError, match="finite non-negative"):
        fit_terrain_from_motion(np.zeros((2, len(JOINTS), 3)), JOINTS, FPS, **{name: value})


def test_physical_family_selector_separates_ramp_and_steps_without_a_trained_model():
    """Surface orientation and swing clearance separate matched support profiles."""

    neutral = calibrate_neutral_foot_pitch(synth_walk(beam_height=0.0), JOINTS, FPS)
    ramp_motion = profile_walk(RAMP_HEIGHT)
    angle = np.arctan(1.0 / 3.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for side in ("L", "R"):
        ankle = ramp_motion[:, JOINTS.index(f"{side}_Ankle")]
        toe = ramp_motion[:, JOINTS.index(f"{side}_Toe")]
        chord_xz = (toe - ankle)[:, [0, 2]] @ rotation.T
        toe[:, 0] = ankle[:, 0] + chord_xz[:, 0]
        toe[:, 2] = ankle[:, 2] + chord_xz[:, 1]
    ramp, ramp_report = fit_terrain_from_motion(ramp_motion, JOINTS, FPS, neutral_foot_pitch=neutral)
    steps, step_report = fit_terrain_from_motion(profile_walk(STAIR_HEIGHT), JOINTS, FPS, neutral_foot_pitch=neutral)

    assert ramp_report["family_evidence"]["model"] == "physical_surface"
    assert ramp_report["family_evidence"]["family"] == "ramp"
    assert ramp_report["model"] == "ramp"
    assert len(ramp.boxes) > 0
    assert step_report["family_evidence"]["family"] == "steps"
    assert step_report["model"] != "ramp"
    assert len(steps.boxes) > 0


def test_uncalibrated_family_fallback_keeps_legacy_residual_definition():
    """A threshold may not be reused with a newly parameterised residual."""

    from terra.terrain.family import classify_terrain_family

    evidence = classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {
            "profile_rms": 0.010,
            "legacy_profile_rms": 0.020,
        },
    )

    assert evidence["family"] == "steps"
    assert "profile residual" in evidence["reason"]

    calibrated = classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {
            "profile_rms": 0.010,
            "legacy_profile_rms": 0.020,
        },
        neutral_foot_pitch={"L": 0.0, "R": 0.0},
    )
    assert calibrated["family"] == "ramp"
    assert "no jointly supported" in calibrated["reason"]


def test_height_only_family_ablation_does_not_evaluate_physical_cues(monkeypatch):
    """The paper ablation must not let orientation or swing evidence leak into selection."""
    import terra.terrain.family as family

    def forbidden(*_args, **_kwargs):
        raise AssertionError("physical family cue evaluated by height-only ablation")

    monkeypatch.setattr(family, "surface_normal_evidence", forbidden)
    monkeypatch.setattr(family, "swing_clearance_evidence", forbidden)
    evidence = family.classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {
            "profile_rms": 0.030,
            "legacy_profile_rms": 0.010,
            "n_footfalls_on_incline": 2,
        },
        neutral_foot_pitch={"L": 0.0, "R": 0.0},
        family_evidence_mode="height_only",
    )

    assert evidence["family"] == "ramp"
    assert evidence["model"] == "height_profile_only"
    assert evidence["family_evidence_mode"] == "height_only"
    assert evidence["height_profile_residual_m"] == pytest.approx(0.010)
    assert evidence["surface_normal_family"] is None
    assert evidence["swing_clearance_family"] is None


def test_one_interior_footfall_cannot_identify_a_continuous_ramp():
    """One raised support is equally explained by a flat platform and an incline."""
    from terra.terrain.family import classify_terrain_family

    common = {
        "profile_rms": 0.010,
        "legacy_profile_rms": 0.010,
        "length": 2.0,
    }
    underdetermined = classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        common | {"n_footfalls_on_incline": 1},
    )
    observed = classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        common | {"n_footfalls_on_incline": 2},
    )

    assert underdetermined["family"] == "steps"
    assert "fewer than two observed support positions" in underdetermined["reason"]
    assert observed["family"] == "ramp"


@pytest.mark.parametrize(
    ("normal_family", "normal_margin", "swing_family"),
    [("steps", -0.6, "ramp"), ("ramp", 0.6, "steps")],
)
def test_sub_resolution_surface_normal_ties_are_resolved_symmetrically_by_swing(
    monkeypatch, normal_family, normal_margin, swing_family
):
    """The sign of angular noise must not decide which cue is allowed to break a tie."""
    import terra.terrain.family as family

    monkeypatch.setattr(
        family,
        "surface_normal_evidence",
        lambda *_args, **_kwargs: {
            "surface_normal_family": normal_family,
            "surface_normal_margin_deg": normal_margin,
            "surface_normal_step_error_deg": 4.0,
            "surface_normal_ramp_error_deg": 4.0,
            "surface_normal_n_intervals": 2,
            "surface_normal_slope_deg": 7.5,
        },
    )
    monkeypatch.setattr(
        family,
        "swing_clearance_evidence",
        lambda *_args, **_kwargs: {
            "swing_clearance_family": swing_family,
            "swing_clearance_margin": -0.1 if swing_family == "ramp" else 0.1,
        },
    )

    evidence = family.classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {"profile_rms": 0.010},
        neutral_foot_pitch={"L": 0.0, "R": 0.0},
    )

    assert evidence["family"] == swing_family
    assert "within positional resolution" in evidence["reason"]


def test_sub_resolution_normal_is_not_overturned_by_unresolved_swing_noise(monkeypatch):
    """A near-zero quartile contrast cannot defeat repeated surface-normal evidence."""
    import terra.terrain.family as family

    monkeypatch.setattr(
        family,
        "surface_normal_evidence",
        lambda *_args, **_kwargs: {
            "surface_normal_family": "steps",
            "surface_normal_margin_deg": -1.98,
            "surface_normal_step_error_deg": 4.0,
            "surface_normal_ramp_error_deg": 5.98,
            "surface_normal_n_intervals": 5,
            "surface_normal_slope_deg": 5.5,
        },
    )
    monkeypatch.setattr(
        family,
        "swing_clearance_evidence",
        lambda *_args, **_kwargs: {
            "swing_clearance_family": "ramp",
            "swing_clearance_margin": -0.028,
        },
    )

    evidence = family.classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {"profile_rms": 0.010},
        neutral_foot_pitch={"L": 0.0, "R": 0.0},
    )

    assert evidence["family"] == "steps"
    assert evidence["reason"] == "supported-foot surface normal"


def test_prior_only_short_ramp_loses_to_step_swing_without_flat_calibration(monkeypatch):
    """A minimum-length ramp prior must not turn one observed step into an incline."""
    import terra.terrain.family as family
    from terra.terrain.ramps import RAMP_MIN_LENGTH

    monkeypatch.setattr(
        family,
        "surface_normal_evidence",
        lambda *_args, **_kwargs: {
            "surface_normal_family": None,
            "surface_normal_margin_deg": None,
            "surface_normal_step_error_deg": None,
            "surface_normal_ramp_error_deg": None,
            "surface_normal_n_intervals": 0,
            "surface_normal_slope_deg": None,
        },
    )
    monkeypatch.setattr(
        family,
        "swing_clearance_evidence",
        lambda *_args, **_kwargs: {
            "swing_clearance_family": "steps",
            "swing_clearance_margin": 0.67,
            "swing_clearance_n": 2,
        },
    )

    evidence = family.classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {
            "profile_rms": 0.006,
            "legacy_profile_rms": 0.008,
            "length": RAMP_MIN_LENGTH,
        },
    )

    assert evidence["family"] == "steps"
    assert evidence["reason"] == ("ramp span is entirely minimum-length prior; swing clearance decides")


def test_long_near_threshold_profile_can_use_independent_ramp_swing_evidence(monkeypatch):
    """Two resolved motion cues may identify a ramp without neutral-foot calibration."""
    import terra.terrain.family as family

    monkeypatch.setattr(
        family,
        "surface_normal_evidence",
        lambda *_args, **_kwargs: {
            "surface_normal_family": None,
            "surface_normal_margin_deg": None,
            "surface_normal_step_error_deg": None,
            "surface_normal_ramp_error_deg": None,
            "surface_normal_n_intervals": 0,
            "surface_normal_slope_deg": None,
        },
    )
    monkeypatch.setattr(
        family,
        "swing_clearance_evidence",
        lambda *_args, **_kwargs: {
            "swing_clearance_family": "ramp",
            "swing_clearance_margin": -0.19,
        },
    )

    evidence = family.classify_terrain_family(
        np.zeros((10, len(JOINTS), 3)),
        JOINTS,
        FPS,
        (),
        {
            "profile_rms": 0.018,
            "profile_max": 0.036,
            "legacy_profile_rms": 0.018,
            "length": 3.0,
        },
    )

    assert evidence["family"] == "ramp"
    assert "within motion resolution" in evidence["reason"]


@pytest.mark.parametrize(
    "ramp",
    [
        {"profile_rms": 0.021, "profile_max": 0.036, "legacy_profile_rms": 0.021, "length": 3.0},
        {"profile_rms": 0.018, "profile_max": 0.041, "legacy_profile_rms": 0.018, "length": 3.0},
        {"profile_rms": 0.018, "profile_max": 0.036, "legacy_profile_rms": 0.018, "length": 0.80},
    ],
)
def test_ramp_swing_cannot_override_motion_resolution_or_observed_span(monkeypatch, ramp):
    """The second cue resolves only a bounded, observed profile ambiguity."""
    import terra.terrain.family as family

    monkeypatch.setattr(
        family,
        "surface_normal_evidence",
        lambda *_args, **_kwargs: {
            "surface_normal_family": None,
            "surface_normal_margin_deg": None,
            "surface_normal_step_error_deg": None,
            "surface_normal_ramp_error_deg": None,
            "surface_normal_n_intervals": 0,
            "surface_normal_slope_deg": None,
        },
    )
    monkeypatch.setattr(
        family,
        "swing_clearance_evidence",
        lambda *_args, **_kwargs: {
            "swing_clearance_family": "ramp",
            "swing_clearance_margin": -0.19,
        },
    )

    evidence = family.classify_terrain_family(np.zeros((10, len(JOINTS), 3)), JOINTS, FPS, (), ramp)

    assert evidence["family"] == "steps"


def raised(events, above=0.05):
    """Events on a surface higher than the floor, judged after removing the joint offset.

    Comparing raw heights against a single threshold would count every floor *ankle*
    contact as raised - it sits 9 cm up - which is the same conflation the fitter exists
    to avoid.
    """
    return [e for e in events if e.z - JOINT_OFFSET[e.joint] > above]


# ----------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------


def test_penetration_is_zero_on_the_top_face():
    """A foot resting on the surface must not read as passing through it.

    The free-space test rejects a candidate box wherever a body point is inside it. If
    depth were measured from the top face rather than the nearest face, every supporting
    footfall would veto its own support and the fit would collapse to nothing.
    """
    b = BoxSpec(pos=(0, 0, 0.05), size=(1.0, 0.3, 0.05))
    on_top = np.array([[0.0, 0.0, 0.10]])
    sunk = np.array([[0.0, 0.0, 0.08]])
    outside = np.array([[2.0, 0.0, 0.0]])

    assert b.penetration(on_top)[0] == pytest.approx(0.0, abs=1e-9)
    assert b.penetration(sunk)[0] == pytest.approx(0.02)
    assert b.penetration(outside)[0] == 0.0


def test_local_frame_roundtrip_with_yaw():
    b = BoxSpec(pos=(0.3, -0.2, 0.05), size=(1.0, 0.2, 0.05), yaw=0.7)
    pts = np.random.default_rng(0).normal(size=(20, 3))
    np.testing.assert_allclose(b.to_world(b.to_local(pts)), pts, atol=1e-12)


def test_height_at_respects_yaw():
    b = BoxSpec(pos=(0, 0, 0.05), size=(1.0, 0.1, 0.05), yaw=np.pi / 2)
    # Long axis now runs along y, so (0, 0.9) is on it and (0.9, 0) is not.
    assert TerrainSpec(boxes=(b,)).height_at(0.0, 0.9) == pytest.approx(0.10)
    assert TerrainSpec(boxes=(b,)).height_at(0.9, 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("yaw,pitch", [(0.0, 0.0), (0.7, 0.0), (0.0, -0.20), (-1.3, 0.35), (2.1, -0.45), (0.3, 0.9)])
def test_height_at_matches_a_mujoco_ray_cast(yaw, pitch):
    """The surface this module reports must be the surface MuJoCo's own geom has.

    The whole design rests on retargeting and simulation sharing one description, so this
    is the load-bearing check on `BoxSpec`: build the geom MuJoCo would build, drop a
    vertical ray onto it, and require `height_at` to agree everywhere.

    It is what caught the pitched box's real geometry. A tilted slab's vertical silhouette
    is longer than its top *face* by the projection of its end walls, so a query built
    around the top face alone reports floor over that strip - a foot against the raised end
    of a ramp reading z=0 with a wall in front of it.
    """
    mujoco = pytest.importorskip("mujoco")
    b = BoxSpec(pos=(0.5, 0.1, 0.35), size=(1.2, 0.6, 0.30), yaw=yaw, pitch=pitch)
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="r", type=mujoco.mjtGeom.mjGEOM_BOX, pos=b.pos, quat=b.quat, size=b.size)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(4)
    x, y = rng.uniform(-2.5, 3.0, 500), rng.uniform(-2.5, 2.5, 500)
    ours = np.asarray(TerrainSpec(boxes=(b,)).height_at(x, y))
    for i in range(len(x)):
        gid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(model, data, np.array([x[i], y[i], 20.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        # The floor wins wherever the box has tipped below it, which `height_at` reports
        # as 0 rather than as the negative height of the box's own face.
        hit = max(20.0 - dist, 0.0) if gid[0] >= 0 else 0.0
        assert ours[i] == pytest.approx(hit, abs=1e-9)


def test_a_ramp_surface_climbs_along_its_axis():
    """The reading that makes a ramp a ramp: height varies continuously along it."""
    # 3 m long, climbing 0.30 m, running along +x from the origin.
    rise, half = 0.30, 1.5
    pitch = -np.arctan2(rise, 2 * half)
    b = BoxSpec(pos=(half, 0.0, 0.0), size=(half / np.cos(pitch), 0.5, 0.4), pitch=pitch)
    # Anchor it so the low end's top face sits at z=0.
    n = b.rotation[:, 2]
    b = BoxSpec(pos=tuple(np.array(b.pos) + np.array([0, 0, rise / 2]) - b.size[2] * n), size=b.size, pitch=pitch)
    t = TerrainSpec(boxes=(b,))

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert float(t.height_at(2 * half * frac, 0.0)) == pytest.approx(rise * frac, abs=1e-9)
    # Beside it and beyond its far end there is no ramp, only floor.
    assert float(t.height_at(1.5, 2.0)) == pytest.approx(0.0)
    assert float(t.height_at(-0.5, 0.0)) == pytest.approx(0.0)


def test_every_fitted_box_is_findable_as_static_environment():
    """A terrain geom the environment selector cannot find is collided with by nothing.

    `scene_geom_ids` picks the static environment out of the model **by name prefix**, and
    hands that set both to the solver's non-penetration term and to the quality metric. A
    box named outside the convention is built, rendered, and then ignored by both, with no
    error anywhere: the fit gate passes, the video shows the ramp, and every clip solves
    with no terrain constraint while penetration reads exactly 0.0 mm. That is what naming
    the fitted ramp `terrain_ramp_0` did, and it cost a whole 40-motion run.
    """
    from loco_mujoco.core.terrain import BoxTerrain

    # Every producer in the module, not just the ramp: this is a property of the interface.
    for motion, kwargs in (
        (synth_walk(), {}),
        (profile_walk(RAMP_HEIGHT), {}),
        (synth_walk(beam_height=0.30, n_beam_steps=6), {"stair_flight": "off"}),
    ):
        terrain, _ = fit_terrain_from_motion(motion, JOINTS, FPS, **kwargs)
        for b in terrain.boxes:
            assert b.name.startswith(BoxTerrain.GEOM_PREFIX), f"{b.name!r} would be invisible to scene_geom_ids"


def test_a_box_named_outside_the_convention_is_refused_at_build():
    """The same rule, enforced where the geom is actually created."""
    from loco_mujoco.core.terrain import BoxTerrain

    mujoco = pytest.importorskip("mujoco")
    t = BoxTerrain.__new__(BoxTerrain)
    t._pack(TerrainSpec(boxes=(BoxSpec(pos=(0, 0, 0.1), size=(1.0, 0.5, 0.1), name="ramp"),)))
    t.rgba = (0.5, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="invisible to non-penetration"):
        t.modify_spec(mujoco.MjSpec())


def test_pitch_beyond_the_limit_is_refused():
    with pytest.raises(ValueError, match="is a wall"):
        BoxSpec(pos=(0, 0, 0.5), size=(1.0, 0.5, 0.2), pitch=np.deg2rad(75))


def _staircase(n: int = 4, riser: float = 0.20, run: float = 0.30) -> TerrainSpec:
    """`n` abutting treads climbing along +x, each `riser` above the last."""
    return TerrainSpec(
        boxes=tuple(
            BoxSpec(
                pos=((i + 0.5) * run, 0.0, (i + 1) * riser / 2),
                size=(run / 2, 0.5, (i + 1) * riser / 2),
                name=f"terrain_box_{i}",
            )
            for i in range(n)
        )
    )


def test_height_near_reads_a_step_before_the_foot_is_over_it():
    """A reach of a foot length is what lets a swing arc over a step instead of into it.

    The same reach is why the reading is only meaningful next to the quantity it is compared
    with: standing on a tread, the tread above is within reach, so this returns a height a
    whole riser above the surface the foot is actually on. See `source_sole_clearance`.
    """
    stairs = _staircase()
    mid = 0.45  # the middle of tread 1, whose top is at 0.40

    assert stairs.height_at(mid, 0.0) == pytest.approx(0.40)
    assert stairs.height_near(mid, 0.0, 0.0) == pytest.approx(0.40)
    # Tread 2 starts at x=0.60 and its top is 0.60; a 0.12 m reach does not span the 0.15 m
    # to it, a 0.20 m one does.
    assert stairs.height_near(mid, 0.0, 0.12) == pytest.approx(0.40)
    assert stairs.height_near(mid, 0.0, 0.20) == pytest.approx(0.60)

    # Short of the flight entirely, the step ahead is still read once it is within reach.
    assert stairs.height_near(-0.10, 0.0, 0.05) == pytest.approx(0.0)
    assert stairs.height_near(-0.10, 0.0, 0.12) == pytest.approx(0.20)


def test_serialisation_roundtrip():
    t = TerrainSpec(
        boxes=(BoxSpec(pos=(0.1, 0.2, 0.05), size=(1.0, 0.3, 0.05), yaw=0.2),), provenance={"source": "test"}
    )
    assert TerrainSpec.from_dict(json.loads(json.dumps(t.to_dict()))) == t


def test_surface_points_are_not_buried_under_boxes():
    """Floor samples under a box would be support the foot can never reach."""
    t = TerrainSpec(boxes=(BoxSpec(pos=(0, 0, 0.05), size=(1.0, 0.3, 0.05)),))
    pts = t.surface_points(spacing=0.2, ground_range=(-3, 3), ground_size=10)
    ground = pts[pts[:, 2] < 1e-9]
    assert not t.boxes[0].contains_xy(ground[:, 0], ground[:, 1]).any()


# ----------------------------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------------------------


def test_stance_detection_finds_raised_contacts():
    """The detector must not be anchored to the motion's lowest point.

    A height gate measured from the global minimum - the one flat-ground foot anchoring
    uses - accepts only floor contacts, so a beam walk reads as perfectly flat and the fit
    silently returns no terrain.
    """
    motion = synth_walk()
    events = detect_stance_events(motion, JOINTS, FPS)
    assert len(raised(events)) >= 4, "raised stances were rejected"
    assert len(events) - len(raised(events)) >= 4, "floor stances were rejected"


def test_local_window_must_stay_inside_a_stride():
    """Widening the local-height window past a stride re-breaks raised detection.

    With the window reaching the neighbouring footfall - which on a beam is on the floor -
    every raised contact is compared against the floor and discarded. This is the failure
    that produced a confident, empty terrain, and it has no error path.
    """
    motion = synth_walk()
    wide = detect_stance_events(motion, JOINTS, FPS, local_window_s=3.0)
    assert not raised(wide), "expected the known failure mode to reproduce"

    narrow = detect_stance_events(motion, JOINTS, FPS, local_window_s=0.3)
    assert raised(narrow)


def test_levels_cluster_in_offset_corrected_heights():
    """Two joints with different foot offsets must not split one surface into two levels.

    Toes rest 2 cm above the surface and ankles 9 cm, so on a flat walk the raw heights
    form two well-separated groups that look exactly like a floor and a 7 cm step. Only
    after each joint's own offset is removed do they collapse to the one surface they are.
    """
    flat = detect_stance_events(synth_walk(beam_height=0.0), JOINTS, FPS)
    assert len(cluster_levels(flat, offsets={})) == 2, "expected raw heights to split"
    assert len(cluster_levels(flat)) == 1

    beam = detect_stance_events(synth_walk(beam_height=0.10), JOINTS, FPS)
    levels = cluster_levels(beam)
    assert len(levels) == 2
    assert len(levels[0]) >= 4 and len(levels[1]) >= 4


def test_fitted_height_cancels_the_joint_offset():
    """The differential estimator must recover the true height, not the joint centre's.

    Toes sit 2 cm above the surface and ankles 9 cm. Taking absolute heights would put the
    surface somewhere between 12 cm and 19 cm for a 10 cm beam, and the two joints would
    disagree by 7 cm. Measured against each joint's own floor contacts, both are right.
    """
    motion = synth_walk(beam_height=0.10)
    terrain, report = fit_terrain_from_motion(motion, JOINTS, FPS)

    assert len(terrain) == 1
    assert terrain.boxes[0].top == pytest.approx(0.10, abs=0.005)

    level = next(lv for lv in report["levels"] if "skipped" not in lv)
    assert level["spread"] < 0.01, "per-joint estimates disagree; the offset did not cancel"
    for estimate in level["per_joint"].values():
        assert estimate == pytest.approx(0.10, abs=0.005)


def test_free_space_bounds_the_extent():
    """The box must stop short of where the body demonstrably passed through.

    Support evidence alone leaves the ends unbounded; only the step down onto the floor
    says where the beam ends.
    """
    motion = synth_walk(beam_x=(-0.9, 0.9))
    terrain, _ = fit_terrain_from_motion(motion, JOINTS, FPS)
    b = terrain.boxes[0]

    lo, hi = b.pos[0] - b.size[0], b.pos[0] + b.size[0]
    assert lo > -1.8 and hi < 1.8, "box swallowed the floor footfalls"
    # ...but it must still cover every raised contact.
    assert lo < -0.85 and hi > 0.85


def test_free_space_evidence_can_be_ablated_without_changing_support_events():
    """The paper ablation removes negative extent evidence, not positive contacts."""
    motion = synth_walk(beam_x=(-0.9, 0.9))
    bounded, bounded_report = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        max_extension=(1.0, 0.0),
    )
    support_only, support_only_report = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        max_extension=(1.0, 0.0),
        use_free_space_evidence=False,
    )

    assert bounded_report["n_stance_events"] == support_only_report["n_stance_events"]
    assert bounded_report["use_free_space_evidence"] is True
    assert support_only_report["use_free_space_evidence"] is False
    assert support_only.boxes[0].size[0] > bounded.boxes[0].size[0]


def test_a_single_hover_does_not_become_terrain():
    """One slow moment in mid-swing is not a surface."""
    motion = synth_walk(beam_height=0.0)  # flat walk
    motion[300:320, JOINTS.index("L_Toe"), 2] = 0.30  # a stationary hover
    motion[300:320, JOINTS.index("L_Toe"), 0] = motion[300, JOINTS.index("L_Toe"), 0]
    terrain, _ = fit_terrain_from_motion(motion, JOINTS, FPS)
    assert terrain.is_flat


def test_validation_rejects_a_wrong_height():
    """The gate has to fail when the terrain does not match, or it gates nothing."""
    motion = synth_walk(beam_height=0.10)
    good, _ = fit_terrain_from_motion(motion, JOINTS, FPS)
    assert validate_terrain(motion, JOINTS, good, FPS)["passed"]

    b = good.boxes[0]
    bad = TerrainSpec(boxes=(BoxSpec(pos=(b.pos[0], b.pos[1], b.pos[2] + 0.06), size=b.size, yaw=b.yaw),))
    assert not validate_terrain(motion, JOINTS, bad, FPS)["passed"]


def test_flat_motion_yields_flat_terrain():
    terrain, report = fit_terrain_from_motion(synth_walk(beam_height=0.0), JOINTS, FPS)
    assert terrain.is_flat
    assert not report["warnings"]


# ----------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------


def test_env_geoms_match_the_spec():
    """The geoms the environment builds must be the boxes that were fitted.

    This is the join between retargeting and simulation: if it slips, the solver
    constrains against one surface and the policy walks on another.
    """
    mujoco = pytest.importorskip("mujoco")
    from loco_mujoco.core.terrain import BoxTerrain

    spec = TerrainSpec(boxes=(BoxSpec(pos=(0.1, -0.2, 0.05), size=(1.2, 0.25, 0.05), yaw=0.3),))
    m_spec = mujoco.MjSpec()
    m_spec.worldbody.add_body(name="dummy").add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0])
    terrain = BoxTerrain.__new__(BoxTerrain)
    terrain.spec_ = spec
    terrain.rgba = (0.5, 0.5, 0.5, 1.0)
    model = terrain.modify_spec(m_spec).compile()

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_box_0")
    assert gid >= 0
    np.testing.assert_allclose(model.geom_pos[gid], spec.boxes[0].pos, atol=1e-9)
    np.testing.assert_allclose(model.geom_size[gid], spec.boxes[0].size, atol=1e-9)
    np.testing.assert_allclose(model.geom_quat[gid], spec.boxes[0].quat, atol=1e-9)


def test_sampled_heights_match_the_spec():
    """`sample_heights_at_points` feeds the height-map observation; it must agree."""
    from loco_mujoco.core.terrain import BoxTerrain

    spec = TerrainSpec(
        boxes=(
            BoxSpec(pos=(0.0, 0.0, 0.05), size=(1.0, 0.2, 0.05), yaw=0.4),
            BoxSpec(pos=(2.0, 0.0, 0.10), size=(0.4, 0.4, 0.10)),
            BoxSpec(pos=(-1.5, 0.3, 0.20), size=(0.8, 0.5, 0.25), yaw=-0.9, pitch=-0.18),
        )
    )
    terrain = BoxTerrain.__new__(BoxTerrain)
    terrain._pack(spec)

    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, 200)
    y = rng.uniform(-2, 2, 200)
    got = terrain.sample_heights_at_points(x, y, None, None, np)
    np.testing.assert_allclose(got, spec.height_at(x, y), atol=1e-12)


def test_sampled_heights_agree_across_backends():
    jnp = pytest.importorskip("jax.numpy")
    from loco_mujoco.core.terrain import BoxTerrain

    spec = TerrainSpec(
        boxes=(
            BoxSpec(pos=(0.0, 0.0, 0.05), size=(1.0, 0.2, 0.05), yaw=0.4),
            BoxSpec(pos=(1.4, 0.0, 0.12), size=(0.6, 0.4, 0.2), pitch=-0.3),
        )
    )
    terrain = BoxTerrain.__new__(BoxTerrain)
    terrain._pack(spec)

    x = np.linspace(-2, 2, 50)
    y = np.zeros(50)
    np.testing.assert_allclose(
        np.asarray(terrain.sample_heights_at_points(jnp.array(x), jnp.array(y), None, None, jnp)),
        terrain.sample_heights_at_points(x, y, None, None, np),
        atol=1e-6,
    )


def synth_stairs(n_steps=4, riser=0.20, tread=0.27, foot=0.15, stance_frames=60, swing_frames=40):
    """A flight climbed one footfall per tread, ending on a landing both feet stand on.

    That landing is where the fitter breaks, and the rest of the flight is the control. A
    tread with a single footfall is benign: its two collinear contacts make
    `_principal_yaw` return that foot's own long axis, which runs roughly along the flight,
    so the fitter's unchecked lateral direction points harmlessly across the steps. Two
    footfalls side by side instead define the *inter-foot* axis, perpendicular to travel -
    and then "lateral" points straight down the stairs and grows by `contact_margin` plus
    `max_extension[1]` with nothing checking what is already there.

    `foot` is SMPL's ankle-to-toe distance rather than a token offset, because it is what
    sets that growth: half the contact spread plus 0.05 plus 0.15 comes to 0.275 m here,
    against the 0.269-0.292 m measured on the KIT stair clips this reproduces.
    """
    top = (n_steps + 1) * riser
    landing = n_steps * tread
    # (x, z, side) in order; the last two are both feet arriving on the landing abreast.
    plan = [(-0.5, 0.0, "L"), (-0.25, 0.0, "R")]
    plan += [(i * tread, (i + 1) * riser, "LR"[i % 2]) for i in range(n_steps)]
    plan += [(landing, top, "LR"[n_steps % 2]), (landing, top, "RL"[n_steps % 2])]

    frames = []
    foot_state = {"L": (plan[0][0], 0.0), "R": (plan[0][0], 0.0)}
    for x, z, side in plan:
        prev = foot_state[side]
        for f in range(swing_frames):
            a = (f + 1) / swing_frames
            foot_state[side] = (prev[0] + a * (x - prev[0]), prev[1] + a * (z - prev[1]) + 0.10 * np.sin(np.pi * a))
            frames.append(dict(foot_state))
        foot_state[side] = (x, z)
        for _ in range(stance_frames):
            frames.append(dict(foot_state))

    out = np.zeros((len(frames), len(JOINTS), 3))
    for t, st in enumerate(frames):
        for side in ("L", "R"):
            fx, fz = st[side]
            y = 0.1 if side == "L" else -0.1
            out[t, JOINTS.index(f"{side}_Toe")] = [fx + foot, y, fz + JOINT_OFFSET[f"{side}_Toe"]]
            out[t, JOINTS.index(f"{side}_Ankle")] = [fx, y, fz + JOINT_OFFSET[f"{side}_Ankle"]]
        out[t, JOINTS.index("Pelvis")] = [0.5 * (st["L"][0] + st["R"][0]), 0.0, 0.9 + max(st["L"][1], st["R"][1])]
    return out


def test_a_step_does_not_swallow_the_tread_below_it():
    """The defect this pins: on a staircase every footfall was scored one riser too high.

    `_free_space_limit` bounds the *longitudinal* extent only; the lateral half-width was
    contact spread + `contact_margin` + `max_extension[1]` with nothing checking what was
    already there. Since a single footfall per tread makes the fitted yaw arbitrary, that
    unchecked direction can point down the flight, and `height_at` - which returns the
    highest covering box - then answers with the step above.

    Measured on `KIT/3/upstairs01` this was a 201.0 mm error, one exact riser.
    """
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    terrain, _ = fit_terrain_from_motion(motion, JOINTS, FPS)

    events = detect_stance_events(motion, JOINTS, FPS)
    for e in raised(events):
        own = e.z - JOINT_OFFSET[e.joint]
        here = float(np.median(terrain.height_at(e.xy[:, 0], e.xy[:, 1])))
        assert here == pytest.approx(own, abs=0.03), (
            f"a {e.joint} footfall on the {own:.2f} m tread is standing under a {here:.2f} m box"
        )
    assert validate_terrain(motion, JOINTS, terrain, FPS)["passed"]


# A relative measurement is only as good as the agreement between its two halves, and the
# gate is a relative measurement: a contact "rests on" a surface when its residual matches
# the offset that joint shows on the surface it stands on. Take that offset from a fresh
# estimate rather than from the fit, and the two can disagree by a whole riser. Measured on
# `EKUT/EKUT/265/WSDF05_poses`, where the right foot never reaches the floor: 7.9 mm of real
# error reported as 183.2 mm, which is the whole reason the stair-flight model looked like it
# regressed a motion.


def test_the_gate_scores_against_the_offsets_the_fit_used():
    """A foot that never reached the lowest surface must not read as a riser of error."""
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    # Lift the right foot's whole trajectory by a riser's worth of *measurement* error: the
    # signature of a foot whose lowest footfall is a tread up.
    events = detect_stance_events(motion, JOINTS, FPS)
    raw = joint_surface_offsets(events)
    skewed = dict(raw, R_Toe=raw["R_Toe"] + 0.20, R_Ankle=raw["R_Ankle"] + 0.20)

    terrain, report = fit_terrain_from_motion(motion, JOINTS, FPS)
    fitted = report["joint_offsets"]

    honest = validate_terrain(motion, JOINTS, terrain, FPS, offsets=fitted)
    assert honest["passed"], honest["raised_contact_error_max"]

    # The same terrain, scored against a convention the fit did not use.
    wrong = validate_terrain(motion, JOINTS, terrain, FPS, offsets=skewed)
    assert wrong["raised_contact_error_max"] > 0.15, (
        f"the test needs a convention mismatch large enough to matter, got {wrong['raised_contact_error_max']:.3f} m"
    )
    assert not wrong["passed"], "a mismatched baseline must be what fails, not the terrain"


def test_paired_offsets_undo_a_foot_that_never_reached_the_lowest_surface():
    """The estimate a high foot produces is wrong, and its partner's is the correction.

    `joint_surface_offsets` measures each joint against its *own* lowest group of contacts,
    which is that joint's height inside the foot only if the joint ever reached the lowest
    surface. On `EKUT/EKUT/265/WSDF05_poses` the right foot's lowest footfall is a tread up,
    so R_Ankle read +213 mm against L_Ankle's +37 - and clustering in those heights folded a
    real level into the floor, turning a five-level staircase into a three-level one.

    Feet are the same size, so the lower estimate is the sound one, and recovering it has to
    put the clustering back exactly where the true offsets would.
    """
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    events = detect_stance_events(motion, JOINTS, FPS)
    truth = dict(JOINT_OFFSET)
    high = dict(truth, R_Toe=truth["R_Toe"] + 0.20, R_Ankle=truth["R_Ankle"] + 0.20)

    assert paired_sole_offsets(high) == pytest.approx(truth)

    as_truth = [sorted((e.joint, e.start) for e in lv) for lv in cluster_levels(events, offsets=truth)]
    as_high = [sorted((e.joint, e.start) for e in lv) for lv in cluster_levels(events, offsets=high)]
    as_paired = [
        sorted((e.joint, e.start) for e in lv) for lv in cluster_levels(events, offsets=paired_sole_offsets(high))
    ]
    assert as_paired == as_truth, "pairing must restore the clustering the true offsets give"
    assert as_high != as_truth, "the test needs the skew to actually change the clustering"


def test_claimed_footfalls_are_what_bounds_the_lateral_extent():
    """Turning the bound off must bring the failure back, or the test above proves nothing.

    A passing test on a fixed fitter says nothing about which change fixed it. This pins the
    mechanism: with `exclude_claimed=False` the same motion mis-scores a contact by about a
    riser, and the error is lateral - free space, which is longitudinal, cannot catch it.

    Pinned to the per-level path, because on a staircase the flight model no longer takes it:
    its treads are a partition of one axis, so nothing it builds can cover anything else and
    the bound has nothing left to do. That is the point of the model, and the assertion at the
    end of this test is what says so.
    """

    def worst_overscore(terrain):
        worst = 0.0
        for e in raised(detect_stance_events(motion, JOINTS, FPS)):
            own = e.z - JOINT_OFFSET[e.joint]
            here = float(np.median(terrain.height_at(e.xy[:, 0], e.xy[:, 1])))
            worst = max(worst, here - own)
        return worst

    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    loose, _ = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        exclude_claimed=False,
        stair_flight="off",
        # Reproduce the historical optional growth explicitly. The universal
        # default is zero; this test isolates why claimed-footfall clipping is
        # still necessary when a caller requests extra geometry.
        max_extension=(0.0, 0.15),
    )
    assert worst_overscore(loose) > 0.15, (
        f"expected ~a riser of over-scoring without the bound, got {worst_overscore(loose):.3f} m"
    )

    # The same fit with the bound *still* off, but as one flight: disjoint treads make the
    # failure unreachable rather than corrected.
    as_flight, report = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        exclude_claimed=False,
        max_extension=(0.0, 0.15),
    )
    assert report["model"] == "stair_flight"
    assert worst_overscore(as_flight) <= 0.02


def test_the_bound_never_cuts_into_the_support_requirement():
    """Excluding a claimed footfall must not shrink a box off its own footfalls.

    The bound only moves faces that are already outside the support requirement; a surface
    that a foot demonstrably stood on must keep supporting it, and a conflict is reported
    rather than resolved in favour of the higher box.
    """
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    terrain, _ = fit_terrain_from_motion(motion, JOINTS, FPS)

    for e in raised(detect_stance_events(motion, JOINTS, FPS)):
        own = e.z - JOINT_OFFSET[e.joint]
        here = float(np.median(terrain.height_at(e.xy[:, 0], e.xy[:, 1])))
        assert here > own - 0.03, f"footfall at {own:.2f} m left unsupported (surface {here:.2f} m)"


def test_the_bound_leaves_a_single_level_alone():
    """A beam has nothing claimed below it, so the tuned flat-ground geometry must not move.

    `go_over_beam08` is the motion every terrain parameter was tuned on; a lateral bound
    that quietly narrowed it would invalidate that record.
    """
    motion = synth_walk(beam_height=0.10)
    bounded, _ = fit_terrain_from_motion(motion, JOINTS, FPS)
    loose, _ = fit_terrain_from_motion(motion, JOINTS, FPS, exclude_claimed=False)

    assert len(bounded) == len(loose) == 1
    for a, b in zip(bounded.boxes, loose.boxes, strict=True):
        assert a.size == pytest.approx(b.size, abs=1e-9)
        assert a.pos == pytest.approx(b.pos, abs=1e-9)


# --- the frame the fit assumes ---------------------------------------------------------


def test_fitted_heights_do_not_depend_on_the_input_datum():
    """Level heights are differences, so a motion handed over 10 cm high fits the same steps.

    This is what makes the fit safe to run on joints whose datum is uncertain: the ground
    level is defined as the lowest contact level, and every height is measured from it.
    """
    joints = synth_walk(beam_height=0.10)
    fps = FPS
    base, _ = fit_terrain_from_motion(joints, JOINTS, fps)
    lifted, _ = fit_terrain_from_motion(joints + np.array([0.0, 0.0, 0.10]), JOINTS, fps)

    assert len(base) == len(lifted)
    for a, b in zip(base.boxes, lifted.boxes, strict=True):
        assert a.top == pytest.approx(b.top, abs=1e-9)


# ----------------------------------------------------------------------------------------
# What a level is, and what a surface is
# ----------------------------------------------------------------------------------------
# Six defects fixed together on 2026-08-02, and every one of them is the same shape as the
# six before it: a relative measurement whose two halves were read against different
# references, or one statistic on one side and a different statistic on the other. The
# symptom was a staircase 0.07 m wide - narrower than a foot - rendered under a subject
# descending it.


def test_a_tread_is_never_narrower_than_the_footfalls_on_it():
    """The width bug: a foot standing on a tread vetoed that tread's own width.

    `fit_stair_flight` bounds the width by free space - the nearest body point that would
    end up inside a tread. That test read *raw* joint heights against a surface fitted from
    offset-corrected contacts, so a stance toe rolling through heel-off, tens of millimetres
    below the level it defines, counted as a body point passing through the staircase.
    Measured over the 59 fitted flights of the non-flat subset it pinched 10 below 0.25 m
    and five to `STAIR_SOLE_HALF` exactly: 0.07 m of tread for a 0.09 m foot.
    """
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    events = detect_stance_events(motion, JOINTS, FPS)
    offsets = paired_sole_offsets(joint_surface_offsets(events))
    levels = cluster_levels(events, offsets=offsets)

    # Sink each stance toe below the tread it stands on, as a real foot does mid-roll.
    rolled = motion.copy()
    for e in events:
        if e.joint.endswith("_Toe"):
            mid = (e.start + e.end) // 2
            rolled[mid - 2 : mid + 2, JOINTS.index(e.joint), 2] -= 0.05

    _, report = fit_stair_flight(rolled, [lv for lv in levels if len(lv) >= 2], offsets)
    assert report["width"] >= report["support_width"], (
        "the staircase is narrower than the footfalls it was fitted to; every contact "
        "outside it lands over no box at all"
    )
    assert report["width"] > 2 * STAIR_SOLE_HALF[1]


def test_a_stance_joint_inside_its_own_tread_is_not_a_free_space_violation():
    """The gate's other half of the same mistake, and the reason the width bug hid.

    `validate_terrain` scored every joint of every frame against the boxes. A probe joint
    resting on a surface is not a fixed height above it - `KIT/424/upstairs03_poses` has the
    L_Toe that *defines* the 0.813 m tread dip to 0.761 through heel-off - so support and
    free space were answering the same question about the same frames with two conventions,
    and the stricter one won. Those frames are already scored by the support term.
    """
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    terrain, report = fit_terrain_from_motion(motion, JOINTS, FPS)
    assert validate_terrain(motion, JOINTS, terrain, FPS, offsets=report["joint_offsets"])["passed"]

    rolled = motion.copy()
    for e in detect_stance_events(motion, JOINTS, FPS):
        if e.joint.endswith("_Toe"):
            mid = (e.start + e.end) // 2
            rolled[mid - 2 : mid + 2, JOINTS.index(e.joint), 2] -= 0.06
    check = validate_terrain(rolled, JOINTS, terrain, FPS, offsets=report["joint_offsets"])
    assert check["max_penetration"] <= 0.05, (
        f"a stance toe rolling inside its own tread read as {check['max_penetration']:.3f} m "
        "of the body passing through the staircase"
    )

    # A joint that is not standing on anything still has to fail, or nothing is gated.
    through = motion.copy()
    swing = JOINTS.index("Pelvis")
    through[:, swing, 0] = terrain.boxes[-1].pos[0]
    through[:, swing, 1] = terrain.boxes[-1].pos[1]
    through[:, swing, 2] = 0.5 * terrain.boxes[-1].top
    assert validate_terrain(through, JOINTS, terrain, FPS)["max_penetration"] > 0.05


def test_a_footfall_over_no_box_counts_as_unsupported():
    """A terrain covering nothing used to score perfectly.

    `raised` was selected on ``surface_z > 0`` - the *terrain's* answer - so a footfall the
    geometry does not reach dropped out of the sum instead of counting as the support
    failure it is. Over the non-flat subset that hid 52 uncovered footfalls, including six
    motions fitted flat ground for a stepping-stone or low-obstacle crossing.
    """
    motion = synth_walk(beam_height=0.10)
    terrain, report = fit_terrain_from_motion(motion, JOINTS, FPS)
    assert validate_terrain(motion, JOINTS, terrain, FPS, offsets=report["joint_offsets"])["passed"]

    empty = validate_terrain(motion, JOINTS, TerrainSpec(), FPS, offsets=report["joint_offsets"])
    assert empty["n_uncovered_contacts"] > 0
    assert empty["raised_contact_error_max"] == pytest.approx(0.10, abs=0.02)
    assert not empty["passed"], "flat ground must not pass a motion that crosses a beam"


def test_a_level_may_not_chain_across_a_whole_obstacle():
    """Single linkage let one 'level' span floor to obstacle top, and so built nothing.

    Every footfall of a low crossing is within `level_tol` of the next, so the chain runs
    from the floor to the top and the whole motion becomes one level - which, being the
    lowest, is the floor. `KIT/424/step_stones10_poses` put all 23 of its footfalls, 165 mm
    apart end to end, on one surface at h=0.048.
    """
    motion = synth_walk(beam_height=0.10)
    # A ramp of intermediate footfalls, each within `level_tol` of its neighbours.
    events = detect_stance_events(motion, JOINTS, FPS)
    offsets = joint_surface_offsets(events)
    ladder = []
    for i, e in enumerate(events):
        z = 0.035 * i
        ladder.append(type(e)(joint=e.joint, start=e.start, end=e.end, z=z + offsets.get(e.joint, 0.0), xy=e.xy))

    levels = cluster_levels(ladder, offsets=offsets)
    assert len(levels) > 1, "a 0.035 m ladder chained into one surface"
    for lv in levels:
        zs = [e.z - offsets.get(e.joint, 0.0) for e in lv]
        assert max(zs) - min(zs) <= 0.08 + 1e-9


def test_a_tread_stepped_on_once_is_still_a_tread():
    """`min_events_per_level` counts events, and toe plus ankle of one footfall are two.

    A fast descent rolls the toe through continuously while the ankle is briefly still, so a
    real tread can register a single event and be discarded. On `EKUT/EKUT/265/WSDF03_poses`
    that left one riser of 0.353 m where the other four measure 0.166-0.188, and every
    footfall on the missing tread scored against the one above - 182 mm.
    """
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    kept, _ = fit_terrain_from_motion(motion, JOINTS, FPS)

    # Take one tread's toe evidence away, leaving the ankle alone on that level.
    thinned = motion.copy()
    events = detect_stance_events(motion, JOINTS, FPS)
    offsets = paired_sole_offsets(joint_surface_offsets(events))
    levels = cluster_levels(events, offsets=offsets)
    victim = next(e for lv in levels[2:-1] for e in lv if e.joint.endswith("_Toe"))
    # Move it through its own stance so the speed gate stops calling it a contact.
    j = JOINTS.index(victim.joint)
    thinned[victim.start : victim.end, j, 0] += np.linspace(0, 0.5, victim.end - victim.start)

    thin_terrain, _ = fit_terrain_from_motion(thinned, JOINTS, FPS)
    assert len(thin_terrain) == len(kept), (
        f"lost a tread with the toe evidence: {len(thin_terrain)} boxes against {len(kept)}"
    )
    np.testing.assert_allclose(sorted(b.top for b in thin_terrain.boxes), sorted(b.top for b in kept.boxes), atol=0.02)


def test_a_footfall_straddling_a_riser_is_placed_by_its_centre():
    """`np.median` over per-frame box tops invents a surface halfway up the riser.

    `height_at` returns a box top; the median of an even number of them averages the two
    middle values. A footfall that settles across a riser got back a height no box has and
    scored as half a riser of error against both treads - three contacts at 89-98 mm on
    `KIT/424/upstairs_downstairs02_poses`, against a terrain that supports them exactly.
    """
    # Two abutting treads: the lower spans x in [0.0, 0.3], the upper [0.3, 0.6]. Abutting,
    # so every sample of the run is over one box or the other and the floor cannot stand in.
    lower = BoxSpec(pos=(0.15, 0.0, 0.10), size=(0.15, 0.5, 0.10), yaw=0.0)
    upper = BoxSpec(pos=(0.45, 0.0, 0.20), size=(0.15, 0.5, 0.20), yaw=0.0)
    terrain = TerrainSpec(boxes=(lower, upper))

    n = 40
    motion = np.zeros((n, len(JOINTS), 3))
    motion[:, :, 2] = 1.0
    for side, y in (("L", 0.1), ("R", -0.1)):
        # Both feet on the upper tread, with the toe run split evenly across the riser.
        for j in (f"{side}_Toe", f"{side}_Ankle"):
            motion[:, JOINTS.index(j)] = [0.40, y, upper.top + JOINT_OFFSET[j]]
        toe = JOINTS.index(f"{side}_Toe")
        motion[: n // 2, toe, 0] = 0.22  # over the lower tread for exactly half the run

    check = validate_terrain(motion, JOINTS, terrain, FPS, offsets={j: JOINT_OFFSET[j] for j in JOINT_OFFSET})
    tops = {round(lower.top, 9), round(upper.top, 9)}
    for r in check["contacts"]:
        # `probe_surface_z` is the unpaired reading, which is where the median was taken.
        assert round(r["probe_surface_z"], 9) in tops, (
            f"{r['joint']} is on {r['probe_surface_z']:.3f} m, a height no box has"
        )
        assert r["surface_z"] == pytest.approx(upper.top)


def test_a_foot_rests_on_the_highest_surface_any_part_of_it_is_over():
    """An ankle hanging back over the tread below is how one stands on a step.

    Toe and ankle are ~0.11 m apart along a ~0.25 m tread, so on a staircase they routinely
    straddle a riser. Scored on its own, the overhanging ankle reports a whole riser of
    unsupported contact - 207 mm on `KIT/3/walking_upstairs04_poses`. Boxes are solid from
    z=0, so a foot with any part over a taller box would be inside it: the highest surface
    under the foot is the one it must be resting on, which is also how
    `_terrain_metrics.surface_under_foot` reads the retargeted side.
    """
    # Two abutting treads: the lower spans x in [0.0, 0.3], the upper [0.3, 0.6].
    lower = BoxSpec(pos=(0.15, 0.0, 0.10), size=(0.15, 0.5, 0.10), yaw=0.0)
    upper = BoxSpec(pos=(0.45, 0.0, 0.20), size=(0.15, 0.5, 0.20), yaw=0.0)
    terrain = TerrainSpec(boxes=(lower, upper))

    n = 40
    motion = np.zeros((n, len(JOINTS), 3))
    motion[:, :, 2] = 1.0
    for side, y in (("L", 0.1), ("R", -0.1)):
        toe, ankle = f"{side}_Toe", f"{side}_Ankle"
        motion[:, JOINTS.index(toe)] = [0.39, y, upper.top + JOINT_OFFSET[toe]]
        # The ankle is a foot's length behind, over the tread below.
        motion[:, JOINTS.index(ankle)] = [0.28, y, upper.top + JOINT_OFFSET[ankle]]

    offsets = {j: JOINT_OFFSET[j] for j in JOINT_OFFSET}
    check = validate_terrain(motion, JOINTS, terrain, FPS, offsets=offsets)
    for r in check["contacts"]:
        assert r["surface_z"] == pytest.approx(upper.top), f"{r['joint']} placed on the {r['surface_z']:.2f} m surface"
    assert check["raised_contact_error_max"] < 0.01
    # The probe's own reading is kept, so a diagnosis can still see the two disagree.
    assert any(r["probe_surface_z"] == pytest.approx(lower.top) for r in check["contacts"])


def test_a_pitched_foot_bridging_tread_edges_uses_local_probe_support():
    """The inverse riser configuration is one coherent whole-foot hypothesis too.

    Some short clips place the toe and ankle on adjacent tread edges at their respective
    heights. Scoring both against the upper plane invents one riser of error; choosing the
    local hypothesis for the overlapping foot interval supports both exactly.
    """
    lower = BoxSpec(pos=(0.15, 0.0, 0.10), size=(0.15, 0.5, 0.10), yaw=0.0)
    upper = BoxSpec(pos=(0.45, 0.0, 0.20), size=(0.15, 0.5, 0.20), yaw=0.0)
    terrain = TerrainSpec(boxes=(lower, upper))
    motion = np.zeros((40, len(JOINTS), 3))
    motion[:, :, 2] = 1.0
    for side, y in (("L", 0.1), ("R", -0.1)):
        toe, ankle = f"{side}_Toe", f"{side}_Ankle"
        motion[:, JOINTS.index(toe)] = [0.39, y, upper.top + JOINT_OFFSET[toe]]
        motion[:, JOINTS.index(ankle)] = [0.28, y, lower.top + JOINT_OFFSET[ankle]]

    check = validate_terrain(
        motion, JOINTS, terrain, FPS, offsets={joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}
    )
    assert check["passed"]
    assert check["raised_contact_error_max"] < 0.01
    assert {contact["support_mode"] for contact in check["contacts"]} == {"local_probes"}
    selected = sorted({contact["selected_surface_z"] for contact in check["contacts"]})
    assert selected == pytest.approx([lower.top, upper.top])


def test_a_foot_on_a_slope_is_scored_at_each_probe_s_own_height():
    """The same rule as the step above, and on a ramp it must give the opposite answer.

    Straddling a riser, toe and ankle are both at the tread's height. Lying on an incline
    they are legitimately at *different* heights, a foot-length of slope apart, and taking
    the higher for both scores the ankle as buried. Measured on `EKUT/EKUT/265/SLP302`, a
    27 deg ramp fitted to 8 mm RMS: every ankle read 69-71 mm of error against 2 mm at its
    own surface, and the motion was refused as unrepresentable terrain.

    One rule covers both - take the surface the foot is standing on and read its *plane*
    where the probe is - which is why this test sits next to that one.
    """
    rise, half = 0.60, 1.5  # 1.5 m of climb over 3 m, ~11 deg
    pitch = -np.arctan2(rise, 2 * half)
    nz = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    ramp = BoxSpec(
        pos=tuple(np.array([half, 0.0, rise / 2]) - 0.5 * nz), size=(half / np.cos(pitch), 0.6, 0.5), pitch=pitch
    )
    terrain = TerrainSpec(boxes=(ramp,))
    slope = rise / (2 * half)

    n = 40
    motion = np.zeros((n, len(JOINTS), 3))
    motion[:, :, 2] = 2.0
    for side, y in (("L", 0.1), ("R", -0.1)):
        # A flat foot on the slope: the toe 0.11 m further up it than the ankle.
        for joint, x in ((f"{side}_Toe", 1.61), (f"{side}_Ankle", 1.50)):
            motion[:, JOINTS.index(joint)] = [x, y, slope * x + JOINT_OFFSET[joint]]

    offsets = {j: JOINT_OFFSET[j] for j in JOINT_OFFSET}
    check = validate_terrain(motion, JOINTS, terrain, FPS, offsets=offsets)
    assert check["n_raised_contacts"] == 4
    for r in check["contacts"]:
        own = slope * (1.61 if r["joint"].endswith("Toe") else 1.50)
        assert r["surface_z"] == pytest.approx(own, abs=1e-6), (
            f"{r['joint']} scored against {r['surface_z']:.3f} m, not its own {own:.3f} m"
        )
    assert check["raised_contact_error_max"] < 0.005
    assert check["passed"]


def test_a_sole_spanning_the_top_of_a_ramp_is_read_against_the_ground_under_it():
    """One plane may be extended across a foot only while the surfaces under it share a tilt.

    A sole at the top of a ramp has its toe on the flat landing and its heel a foot-length
    back down the incline. The landing is the higher surface, so it is the one the whole-foot
    rule picks - and a flat box's plane is a constant, so the heel is then scored against a
    datum that is 11 mm above the ground it is resting on. On `KIT/3/slope_up04_poses` that
    read a stance as 28 mm inside terrain that `mj_geomDistance` put 9 mm inside.

    Where the tilts differ there is no plane to extend, so each landmark is read against the
    surface actually beneath it. Two flat treads share a tilt, so the staircase case that
    this rule exists for - the overhanging ankle of the test below - is untouched.
    """
    rise, half = 0.30, 1.5
    pitch = -np.arctan2(rise, 2 * half)
    nz = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    incline = BoxSpec(
        pos=tuple(np.array([half, 0.0, rise / 2]) - 0.5 * nz), size=(half / np.cos(pitch), 0.6, 0.5), pitch=pitch
    )
    landing = BoxSpec(pos=(3.2, 0.0, rise / 2), size=(0.2, 0.6, rise / 2))
    terrain = TerrainSpec(boxes=(incline, landing))
    slope = rise / (2 * half)

    # Toe on the landing, heel 0.16 m back down the incline.
    toe, heel = np.array([3.05, 0.0]), np.array([2.89, 0.0])
    foot = np.stack([toe, heel])
    assert terrain.supporting_box(foot) is landing, "precondition: the landing is the higher"

    assert terrain.support_plane_at(foot, *heel) == pytest.approx(slope * heel[0], abs=1e-9), (
        "the heel is resting on the incline, not on the landing's plane extended back down it"
    )
    assert terrain.support_plane_at(foot, *toe) == pytest.approx(rise, abs=1e-9)
    # A landmark overhanging every box keeps the fallback the rule was written for.
    assert terrain.support_plane_at(foot, 3.6, 0.0) == pytest.approx(rise, abs=1e-9)


def test_the_surface_a_foot_stands_on_carries_a_tilt_as_well_as_a_height():
    """`support_normal_at` is the orientation half of `support_plane_at`, on one box.

    Anything calibrated against world-vertical instead of this reads the incline as part of
    whatever constant it is measuring - which is what held the retargeted forefoot off a
    descent by the slope angle over a foot length.
    """
    rise, half = 0.60, 1.5
    pitch = -np.arctan2(rise, 2 * half)
    nz = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    ramp = BoxSpec(
        pos=tuple(np.array([half, 0.0, rise / 2]) - 0.5 * nz), size=(half / np.cos(pitch), 0.6, 0.5), pitch=pitch
    )
    step = BoxSpec(pos=(4.0, 0.0, 0.1), size=(0.4, 0.6, 0.1))
    terrain = TerrainSpec(boxes=(ramp, step))

    on_ramp = np.array([[1.50, 0.1], [1.61, 0.1]])
    assert np.allclose(terrain.support_normal_at(on_ramp), nz), "the incline's own normal"
    # A flat-topped box and the bare floor are both upright, so nothing off a slope moves.
    assert np.allclose(terrain.support_normal_at(np.array([[4.0, 0.0], [4.1, 0.0]])), [0, 0, 1])
    assert np.allclose(terrain.support_normal_at(np.array([[9.0, 0.0]])), [0, 0, 1])


def test_a_ramp_is_fitted_where_a_staircase_is_not():
    """The discriminating evidence: whether the footfall heights lie on one straight line.

    Both motions climb 0.6 m over the same ground with the same stride, so support alone
    cannot separate them - the level model quantises either one into boxes that pass the
    gate. What differs is the residual about a single incline, which is mocap-scale on the
    ramp and half a riser on the stairs. See `RAMP_MAX_PROFILE_RMS`.
    """
    _, ramp_report = fit_terrain_from_motion(profile_walk(RAMP_HEIGHT), JOINTS, FPS)
    _, stair_report = fit_terrain_from_motion(profile_walk(STAIR_HEIGHT), JOINTS, FPS)

    assert ramp_report["model"] == "ramp", ramp_report["ramp"].get("rejected")
    assert ramp_report["ramp"]["slope_deg"] == pytest.approx(np.degrees(np.arctan(1 / 3)), abs=1.5)
    assert stair_report["model"] != "ramp"
    assert "profile residual" in stair_report["family_evidence"]["reason"]


def test_flat_walking_is_never_a_ramp():
    """Level ground fits a fraction of a degree of slope; only the floors keep it out."""
    _, report = fit_terrain_from_motion(synth_walk(beam_height=0.0), JOINTS, FPS)
    assert report["model"] != "ramp"
    assert report["ramp"]["rejected"]


def _incline_events(top_x: float, rise_per_m: float = 1 / 3) -> list:
    """Footfalls up an incline along +x, ending with a foot planted at `top_x`.

    Placed by hand rather than walked, because what this exercises is the fit's treatment of
    its *last* footfall, and where the least-squares breakpoint lands relative to that is
    what a synthesised walk cannot be made to control.
    """
    from terra.terrain import StanceEvent

    events, t = [], 0
    for i, x in enumerate(np.arange(-0.9, top_x + 1e-9, 0.30)):
        for joint, dx in ((f"{'LR'[i % 2]}_Toe", 0.08), (f"{'LR'[i % 2]}_Ankle", 0.0)):
            xy = np.repeat([[x + dx, 0.1 * (1 if i % 2 else -1)]], 30, axis=0)
            events.append(
                StanceEvent(
                    joint=joint, start=t, end=t + 30, z=max(0.0, x + dx) * rise_per_m + JOINT_OFFSET[joint], xy=xy
                )
            )
        t += 60
    return events


def test_the_top_of_a_ramp_reaches_a_foot_past_the_last_footfall():
    """A surface ending under the foot standing on it leaves half that foot over a drop.

    The landing used to be built only where a footfall lay strictly *past* the fitted
    incline, which reads the top footfall as needing no support beyond its own joint centre.
    Every KIT `slope_*` clip fitted an incline whose end sat within a foot-length of its
    topmost footfall - on `slope_down08` the fitted `u1` was 10 mm *above* the highest one,
    so no landing was built at all - and the retargeted figure opened the motion balanced on
    the top edge with its heel in the air over the full rise. `contact_margin` is the same
    support requirement the level model applies, and it belongs here for the same reason.
    """
    from terra.terrain import DEFAULT_CONTACT_MARGIN, fit_ramp
    from terra.terrain.stairs import STAIR_SOLE_HALF

    events = _incline_events(top_x=1.8)
    offsets = {j: JOINT_OFFSET[j] for j in JOINT_OFFSET}
    terrain, report = fit_ramp(np.zeros((1, 1, 3)), events, offsets)
    assert terrain is not None, report.get("rejected")

    # This is the case that used to build nothing: the least-squares breakpoint lands just
    # *past* the highest footfall, so no footfall is strictly beyond the incline.
    assert report["u_max"] < report["u1"], "precondition: no footfall lies past the fitted incline"
    assert report["landing_span"][1] - report["u_max"] >= DEFAULT_CONTACT_MARGIN[0] - 1e-9
    assert report["landing_span"][1] - report["u1"] >= (STAIR_SOLE_HALF[0] + DEFAULT_CONTACT_MARGIN[0] - 1e-9)

    # Read as heights, which is what the solver and the metric see: the foot planted at the
    # top stands on the ramp, not on the floor beside it.
    top = max(e.xy[0, 0] for e in events)  # the events run along +x
    for x in (top, top + DEFAULT_CONTACT_MARGIN[0] - 0.02):
        assert float(terrain.height_at(x, 0.1)) > 0.55, (
            f"x={x:.2f} m - a footfall, or the foot around it - is over the floor"
        )


def test_ramp_landing_covers_motion_after_the_last_complete_stance():
    """A clipped final step remains supported even when it is not a stance event."""
    from terra.terrain import fit_ramp

    events = _incline_events(top_x=1.8)
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}
    terminal_foot_x = 2.65
    motion_foot_xy = np.array(
        [
            point
            for x in np.linspace(-1.0, terminal_foot_x, 80)
            for point in ((x, -0.1), (x + 0.08, -0.1), (x, 0.1), (x + 0.08, 0.1))
        ]
    )

    terrain, report = fit_ramp(
        np.zeros((1, 1, 3)),
        events,
        offsets,
        motion_foot_xy=motion_foot_xy,
    )

    assert terrain is not None, report.get("rejected")
    required = report["motion_foot_span_u"][1] + STAIR_SOLE_HALF[0] + DEFAULT_CONTACT_MARGIN[0]
    assert report["landing_extent_source"] == "full_motion_foot_landmarks"
    assert report["landing_span"][1] >= required - 1e-9


def test_censored_terminal_step_extends_the_incline_only_when_both_probes_agree():
    """Endpoint foot geometry distinguishes another ramp step from a flat landing."""
    from terra.terrain import fit_ramp

    events = _incline_events(top_x=1.8)
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}
    body = np.zeros((1, 1, 3))
    _baseline, initial = fit_ramp(body, events, offsets)
    yaw = initial["yaw"]
    origin = np.asarray(initial["origin_xy"])
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    slope = np.tan(np.radians(initial["slope_deg"]))

    def terminal_points(flat: bool) -> dict[str, np.ndarray]:
        result = {}
        for name, terminal_u in (
            ("R_Ankle", initial["u1"] + 0.40),
            ("R_Toe", initial["u1"] + 0.52),
        ):
            xy = origin + terminal_u * axis
            surface = initial["rise"] if flat else slope * (terminal_u - initial["u0"])
            result[name] = np.repeat([[xy[0], xy[1], surface + offsets[name]]], 5, axis=0)
        return result

    _continued, continued = fit_ramp(
        body,
        events,
        offsets,
        motion_foot_xyz=terminal_points(flat=False),
    )
    _flat, flat = fit_ramp(
        body,
        events,
        offsets,
        motion_foot_xyz=terminal_points(flat=True),
    )

    extension = continued["terminal_incline_extension"]
    assert extension["extended_u1"] >= initial["u1"] + 0.52 + STAIR_SOLE_HALF[0] - 1e-9
    assert extension["evidence"]["R"]["matches_extrapolated_ramp"]
    assert "terminal_incline_extension" not in flat
    assert flat["u1"] == pytest.approx(initial["u1"])


def test_internal_50mm_diagnostic_cannot_replace_a_classified_ramp(monkeypatch):
    """Validation may flag a fit for review, but it has no model-selection authority."""
    import terra.terrain.fitting as fitting

    monkeypatch.setattr(
        fitting,
        "validate_terrain",
        lambda *_args, **_kwargs: {
            "raised_contact_error_max": 0.075,
            "max_penetration": 0.0,
            "passed": False,
        },
    )

    _terrain, report = fit_terrain_from_motion(profile_walk(RAMP_HEIGHT), JOINTS, FPS)

    assert report["family_evidence"]["family"] == "ramp"
    assert report["model"] == "ramp"
    assert not report["ramp_scores"]["ramp"]["passed"]


def test_three_support_events_fit_only_with_independent_family_authority():
    """The three-event escape hatch is explicit; ordinary ramp inference stays at four."""
    from terra.terrain import StanceEvent, fit_ramp

    offsets = {"L_Toe": 0.01, "R_Toe": 0.01}
    events = []
    for i, (joint, x) in enumerate((("L_Toe", 0.0), ("R_Toe", 0.6), ("L_Toe", 1.2))):
        events.append(
            StanceEvent(
                joint=joint,
                start=20 * i,
                end=20 * (i + 1),
                z=offsets[joint] + x / 3.0,
                xy=np.repeat([[x, 0.1 if joint.startswith("L") else -0.1]], 20, axis=0),
            )
        )
    body = np.zeros((60, len(JOINTS), 3))
    body[:, :, 2] = 2.0

    rejected, report = fit_ramp(body, events, offsets, fixed_joint_offsets=True)
    accepted, accepted_report = fit_ramp(body, events, offsets, fixed_joint_offsets=True, min_events=3)

    assert rejected is None
    assert report["rejected"] == "fewer than 4 stance events"
    assert accepted is not None, accepted_report.get("rejected")


def test_flat_reference_offsets_keep_a_high_opening_footfall_on_its_tread():
    """A probe first observed one riser up must not make that tread the floor."""
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)

    terrain, report = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        calibrated_joint_offsets=JOINT_OFFSET,
    )

    assert report["joint_offsets_source"] == "flat_reference"
    assert report["joint_offsets"] == pytest.approx(JOINT_OFFSET)
    assert validate_terrain(motion, JOINTS, terrain, FPS, offsets=report["joint_offsets"])["passed"]


def test_flat_reference_builds_a_lowest_observed_tread_above_the_floor():
    """A clipped descent need not contain a stationary contact on the world floor."""
    motion = synth_stairs(n_steps=4, riser=0.20, tread=0.27)
    motion[:, :, 2] += 0.08

    terrain, report = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        calibrated_joint_offsets=JOINT_OFFSET,
    )

    assert not report["ground_level"]["observed_floor"]
    assert report["ground_level"]["height"] == pytest.approx(0.08, abs=0.005)
    first_contact = min(
        detect_stance_events(motion, JOINTS, FPS),
        key=lambda event: event.z - JOINT_OFFSET[event.joint],
    )
    here = np.median(first_contact.xy, axis=0)
    assert terrain.height_at(*here) == pytest.approx(0.08, abs=0.01)
    strict = validate_terrain(
        motion,
        JOINTS,
        terrain,
        FPS,
        offsets=JOINT_OFFSET,
        compensate_sloped_offsets=False,
    )
    assert strict["passed"], strict["raised_contact_error_max"]


def test_flat_reference_offsets_remain_fixed_while_fitting_a_ramp():
    """A ramp may not replace an out-of-sample sole calibration with fitted intercepts.

    Jointly estimating the four probe intercepts is useful when a motion calibrates itself,
    but makes a common ramp-height error non-identifiable.  External marker datasets use a
    separate flat SMPL-H reference, so that measurement must remain the datum of the fit.
    """
    motion = profile_walk(RAMP_HEIGHT)

    terrain, report = fit_terrain_from_motion(
        motion,
        JOINTS,
        FPS,
        calibrated_joint_offsets=JOINT_OFFSET,
    )

    assert report["model"] == "ramp", report["ramp"].get("rejected")
    assert report["ramp"]["profile_joint_offsets_fixed"]
    assert report["ramp"]["profile_joint_offsets"] == pytest.approx(JOINT_OFFSET)
    strict = validate_terrain(
        motion,
        JOINTS,
        terrain,
        FPS,
        offsets=JOINT_OFFSET,
        compensate_sloped_offsets=False,
    )
    assert strict["support_baseline_mode"] == "fixed_calibration"
    assert strict["passed"], strict["raised_contact_error_max"]


def test_a_flat_reference_gate_cannot_normalize_away_a_low_ramp():
    """A ramp shifted below every planted foot is a support failure, not calibration.

    This is the exact false-pass signature found on Gait120 descent: learning one new
    residual per probe on the fitted incline hid a common 60 mm surface placement error.
    """
    rise, half = 0.60, 1.5
    pitch = -np.arctan2(rise, 2 * half)
    nz = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    centre = np.array([half, 0.0, rise / 2]) - 0.5 * nz
    centre[2] -= 0.06
    terrain = TerrainSpec(
        boxes=(
            BoxSpec(
                pos=tuple(centre),
                size=(half / np.cos(pitch), 0.6, 0.5),
                pitch=pitch,
            ),
        )
    )
    slope = rise / (2 * half)
    motion = np.zeros((40, len(JOINTS), 3))
    motion[:, :, 2] = 2.0
    for side, y in (("L", 0.1), ("R", -0.1)):
        for joint, x in ((f"{side}_Toe", 1.61), (f"{side}_Ankle", 1.50)):
            motion[:, JOINTS.index(joint)] = [x, y, slope * x + JOINT_OFFSET[joint]]

    self_calibrated = validate_terrain(motion, JOINTS, terrain, FPS, offsets=JOINT_OFFSET)
    strict = validate_terrain(
        motion,
        JOINTS,
        terrain,
        FPS,
        offsets=JOINT_OFFSET,
        compensate_sloped_offsets=False,
    )

    assert self_calibrated["passed"], "precondition: the old adaptive gate hides the shift"
    assert strict["raised_contact_error_max"] == pytest.approx(0.06, abs=0.005)
    assert not strict["passed"]


def test_a_ramp_landing_never_steps_up_over_its_own_incline():
    """The landing abuts the incline; it must not overlap back down it.

    Boxes resolve by height, so a flat box at the full rise sitting over ground the incline
    puts lower would *raise* that strip - a step at the top of a slope, in the geometry that
    exists to have none. A minimum landing length would have done exactly that.
    """
    from terra.terrain import fit_ramp

    offsets = {j: JOINT_OFFSET[j] for j in JOINT_OFFSET}
    terrain, report = fit_ramp(np.zeros((1, 1, 3)), _incline_events(top_x=1.8), offsets)
    assert len(terrain.boxes) == 2, "an incline and a landing"

    x = np.linspace(-1.5, 2.5, 800)
    h = terrain.height_at(x, 0.1)
    climb = np.tan(np.radians(report["slope_deg"])) * float(np.diff(x)[0])
    step = float(np.max(np.diff(h)))
    assert step <= climb + 1e-6, (
        f"the surface jumps {step * 1000:.2f} mm where the steepest it may climb over one "
        f"sample is {climb * 1000:.2f} mm"
    )


def test_the_riser_leaves_a_sole_on_both_sides_of_it():
    """A cut placed exactly at the lowest upper footfall puts that whole foot a tread down.

    An invariant of the partition rather than a historical regression: it holds on the code
    before this change too. It is here because moving the cut to per-footfall centres, which
    is what fixed `KIT/424/upstairs_downstairs02_poses`, makes the cap coincide with a
    footfall centre by construction - the fix was 188 mm of error from a one-micron tie, and
    nothing else in this file would notice it coming back.
    """
    riser = 0.20
    offsets = {j: JOINT_OFFSET[j] for j in JOINT_OFFSET}

    def footfall(joint, level, u, spread=0.03, n=30):
        """One probe at rest around `u` along +x, on the tread at `level` risers."""
        xs = np.linspace(u - spread, u + spread, n)
        return StanceEvent(
            joint=joint, start=0, end=n, z=level * riser + offsets[joint], xy=np.stack([xs, np.zeros(n)], axis=1)
        )

    # The upper tread's rearmost footfall is centred 0.07 m past the lower tread's frontmost
    # sample - inside the sole the riser rule reserves, so the cap binds and has to leave
    # that foot on its own tread rather than stand the riser through the middle of it.
    levels = [
        [footfall("L_Toe", 1, 0.20), footfall("L_Ankle", 1, 0.09)],
        [footfall("R_Toe", 2, 0.47), footfall("R_Ankle", 2, 0.36)],
    ]
    joints = np.zeros((30, len(JOINTS), 3))  # only the levels are read for the partition
    terrain, report = fit_stair_flight(joints, levels, offsets)

    assert report["misassigned"] == 0
    for level in levels:
        for e in level:
            centre = np.median(e.xy, axis=0)
            own = e.z - offsets[e.joint]
            here = float(terrain.height_at(centre[0], centre[1]))
            assert here == pytest.approx(own, abs=0.5 * riser), (
                f"a {e.joint} footfall on the {own:.2f} m tread was cut onto the {here:.2f} m one"
            )


def test_stair_risers_use_free_space_to_centre_full_feet_and_extend_landings():
    """Spare tread depth belongs on both sides of a sole, including the last stance."""
    offsets = {j: JOINT_OFFSET[j] for j in JOINT_OFFSET}

    def footfall(joint, level, u):
        xy = np.repeat([[u, 0.0]], 30, axis=0)
        return StanceEvent(
            joint=joint,
            start=0,
            end=30,
            z=0.1 * level + offsets[joint],
            xy=xy,
        )

    levels = [
        [footfall("L_Ankle", 1, 0.0), footfall("L_Toe", 1, 0.08)],
        [footfall("R_Ankle", 2, 0.60), footfall("R_Toe", 2, 0.68)],
        [footfall("L_Ankle", 3, 1.20), footfall("L_Toe", 3, 1.28)],
    ]
    _terrain, report = fit_stair_flight(np.zeros((30, len(JOINTS), 3)), levels, offsets)

    support = report["level_support_spans_u"]
    for cut, lower, upper in zip(
        [span[1] for span in report["spans"][:-1]],
        support[:-1],
        support[1:],
        strict=True,
    ):
        assert cut == pytest.approx(0.5 * (lower[1] + upper[0]))

    assert support[0][0] - report["spans"][0][0] >= (DEFAULT_CONTACT_MARGIN[0] - 1e-9)
    assert report["spans"][-1][1] - support[-1][1] >= (DEFAULT_CONTACT_MARGIN[0] - 1e-9)


def test_stair_flight_regularizes_small_motion_height_noise_to_one_shared_riser():
    """A coherent flight must not serialize one noisy riser per contact cluster."""
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}
    raw_heights = [0.074, 0.194, 0.285, 0.408]

    def level(index, height):
        return [
            StanceEvent(
                joint=joint,
                start=0,
                end=30,
                z=height + offsets[joint],
                xy=np.repeat([[0.60 * index + dx, 0.0]], 30, axis=0),
            )
            for joint, dx in (("L_Ankle", 0.0), ("L_Toe", 0.08))
        ]

    _terrain, report = fit_stair_flight(
        np.zeros((30, len(JOINTS), 3)),
        [level(index, height) for index, height in enumerate(raw_heights)],
        offsets,
    )

    model = report["height_model"]
    assert model["accepted"] is True
    assert model["model"] == "shared_riser_least_squares"
    assert model["raw_heights"] == pytest.approx(raw_heights)
    np.testing.assert_allclose(np.diff(report["heights"]), model["shared_riser"], atol=1e-12)
    assert model["max_abs_adjustment"] < 0.03


def test_stair_regularization_stays_inside_each_motion_support_observation():
    """A coherent prior may approach, but never invalidate, observed foot support."""
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}
    raw_heights = [0.0, 0.20, 0.40, 0.57]

    levels = []
    for index, height in enumerate(raw_heights):
        levels.append(
            [
                StanceEvent(
                    joint=joint,
                    start=0,
                    end=30,
                    z=height + offsets[joint],
                    xy=np.repeat([[0.60 * index + dx, 0.0]], 30, axis=0),
                )
                for joint, dx in (
                    ("L_Ankle", 0.0),
                    ("L_Toe", 0.08),
                )
            ]
        )
    # A weak 0.449 m stance cluster was rejected as a distinct tread and assigned to
    # the nearest accepted 0.40 m surface. The unconstrained line moves that tread down
    # 12 mm, which would make this still-real support observation miss by 61 mm.
    support_targets = [[height] for height in raw_heights]
    support_targets[2].append(0.449)

    _terrain, report = fit_stair_flight(
        np.zeros((30, len(JOINTS), 3)),
        levels,
        offsets,
        support_height_targets=support_targets,
    )

    model = report["height_model"]
    assert model["accepted"] is True
    assert model["model"] == "support_constrained_shared_riser"
    assert 0.0 < model["regularization_blend"] < 1.0
    assert model["support_residual_max_fitted"] > 0.05
    assert model["support_residual_max_effective"] < 0.05
    assert model["support_residual_max_effective"] == pytest.approx(0.05 - 1e-6)


def test_stair_flight_does_not_regularize_a_missing_or_irregular_level():
    """A shared-riser prior may not hide a gap larger than physical resolution."""
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}
    raw_heights = [0.0, 0.10, 0.50, 0.60]

    levels = []
    for index, height in enumerate(raw_heights):
        levels.append(
            [
                StanceEvent(
                    joint="L_Toe",
                    start=0,
                    end=30,
                    z=height + offsets["L_Toe"],
                    xy=np.repeat([[0.60 * index, 0.0]], 30, axis=0),
                )
            ]
        )

    _terrain, report = fit_stair_flight(np.zeros((30, len(JOINTS), 3)), levels, offsets)

    assert report["height_model"]["accepted"] is False
    assert report["height_model"]["model"] == "observed_levels"
    assert report["heights"] == pytest.approx(raw_heights)


def test_stair_landing_covers_motion_after_the_last_complete_stance():
    """The top landing covers terminal walking that stance detection cannot observe."""
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}

    def footfall(joint, level, u):
        xy = np.repeat([[u, 0.0]], 30, axis=0)
        return StanceEvent(
            joint=joint,
            start=0,
            end=30,
            z=0.1 * level + offsets[joint],
            xy=xy,
        )

    levels = [
        [footfall("L_Ankle", 1, 0.0), footfall("L_Toe", 1, 0.08)],
        [footfall("R_Ankle", 2, 0.60), footfall("R_Toe", 2, 0.68)],
        [footfall("L_Ankle", 3, 1.20), footfall("L_Toe", 3, 1.28)],
    ]
    motion_foot_xy = np.array([[-0.40, 0.0], [2.45, 0.0]])

    _terrain, report = fit_stair_flight(
        np.zeros((30, len(JOINTS), 3)),
        levels,
        offsets,
        motion_foot_xy=motion_foot_xy,
    )

    required = report["motion_foot_span_u"][1] + STAIR_SOLE_HALF[0] + DEFAULT_CONTACT_MARGIN[0]
    assert report["landing_extent_source"] == "full_motion_foot_landmarks"
    assert report["spans"][-1][1] >= required - 1e-9


def test_stair_terminal_extension_ignores_floor_transition_below_raised_tread():
    """A clipped descent must not stretch its lowest raised tread over the floor."""
    offsets = {joint: JOINT_OFFSET[joint] for joint in JOINT_OFFSET}

    def footfall(joint, level, u):
        xy = np.repeat([[u, 0.0]], 30, axis=0)
        return StanceEvent(
            joint=joint,
            start=0,
            end=30,
            z=0.1 * level + offsets[joint],
            xy=xy,
        )

    levels = [
        [footfall("L_Ankle", 1, 0.0), footfall("L_Toe", 1, 0.08)],
        [footfall("R_Ankle", 2, 0.60), footfall("R_Toe", 2, 0.68)],
        [footfall("L_Ankle", 3, 1.20), footfall("L_Toe", 3, 1.28)],
    ]
    motion_foot_xyz = {
        name: np.array(
            [
                [-1.20, 0.0, offsets[name]],
                [0.04, 0.0, 0.1 + offsets[name]],
                [1.24, 0.0, 0.3 + offsets[name]],
            ]
        )
        for name in offsets
    }

    _terrain, report = fit_stair_flight(
        np.zeros((30, len(JOINTS), 3)),
        levels,
        offsets,
        motion_foot_xyz=motion_foot_xyz,
    )

    first_support_lo = report["level_support_spans_u"][0][0]
    assert report["landing_extent_source"] == "clip_boundary_paired_foot_landmarks"
    assert report["spans"][0][0] >= first_support_lo - DEFAULT_CONTACT_MARGIN[0] - 1e-9
    assert report["terminal_motion_foot_spans_u"]["first"][0] > -1.0


def test_passing_stair_flight_beats_sparse_zero_penetration_pads(monkeypatch):
    """A passing structured flight must not lose to the smaller collision volume."""
    import terra.terrain.fitting as fitting

    def candidate_check(_evidence, terrain):
        is_flight = terrain.provenance.get("source") == "fit_stair_flight"
        return {
            "raised_contact_error_max": 0.0,
            "max_penetration": 0.045 if is_flight else 0.0,
            "passed": True,
        }

    monkeypatch.setattr(fitting, "_validation_score", candidate_check)

    _terrain, report = fit_terrain_from_motion(
        synth_stairs(n_steps=4, riser=0.10, tread=0.29),
        JOINTS,
        FPS,
    )

    assert report["model"] == "stair_flight"
    assert report["model_scores"]["stair_flight"]["max_penetration"] == pytest.approx(0.045)


def test_default_per_level_width_retains_generic_lateral_completion_prior():
    """A straight contact trace must not collapse a platform to sole width."""
    terrain, report = fit_terrain_from_motion(
        synth_stones(heights=(0.12,), gap=1.6),
        JOINTS,
        FPS,
        ramp="off",
        stair_flight="off",
    )

    assert report["model"] == "per_level"
    assert terrain.boxes
    assert min(2.0 * box.size[1] for box in terrain.boxes) >= 0.30


def test_repeated_trials_reject_isolated_slow_swing_height_clusters():
    """Rare levels are outliers only when genuine supports recur many times."""
    from terra.terrain.stance import StanceEvent, _accept_levels

    def events(count, height):
        return [StanceEvent("L_Toe", index, index + 5, height, np.zeros((5, 2))) for index in range(count)]

    levels = [events(250, 0.0), events(52, 0.20), events(89, 0.40), events(2, 0.53)]
    accepted = _accept_levels([0.0, 0.20, 0.40, 0.53], levels, 2, 0.05)

    assert accepted == [True, True, True, False]


def test_single_pass_motion_retains_one_observation_of_a_distinct_support():
    """The repetition gate must not erase a real stone in a short traversal."""
    from terra.terrain.stance import StanceEvent, _accept_levels

    def events(count, height):
        return [StanceEvent("L_Toe", index, index + 5, height, np.zeros((5, 2))) for index in range(count)]

    levels = [events(8, 0.0), events(2, 0.12), events(1, 0.24)]
    accepted = _accept_levels([0.0, 0.12, 0.24], levels, 2, 0.05)

    assert accepted == [True, True, True]


def synth_stones(heights=(0.10, 0.14), gap=1.6, stance_frames=60, swing_frames=40):
    """A walk along +x over stepping stones at the given heights, floor at each end.

    The stones are `gap` apart, well past `split_gap`, so they become separate boxes out of
    whatever level they land in; their heights are within `level_tol` of each other, so that
    level is one.
    """
    plan = [(-2.2, 0.0), (-1.7, 0.0)]
    plan += [(i * gap, h) for i, h in enumerate(heights)]
    plan += [((len(heights) - 1) * gap + 1.0, 0.0), ((len(heights) - 1) * gap + 1.5, 0.0)]

    frames = []
    foot = {"L": (plan[0][0], 0.0), "R": (plan[0][0], 0.0)}
    for i, (x, z) in enumerate(plan):
        side = "L" if i % 2 == 0 else "R"
        prev = foot[side]
        for f in range(swing_frames):
            a = (f + 1) / swing_frames
            foot[side] = (prev[0] + a * (x - prev[0]), prev[1] + a * (z - prev[1]) + 0.16 * np.sin(np.pi * a))
            frames.append(dict(foot))
        foot[side] = (x, z)
        for _ in range(stance_frames):
            frames.append(dict(foot))

    out = np.zeros((len(frames), len(JOINTS), 3))
    for t, st in enumerate(frames):
        for side in ("L", "R"):
            fx, fz = st[side]
            y = 0.1 if side == "L" else -0.1
            out[t, JOINTS.index(f"{side}_Toe")] = [fx + 0.08, y, fz + JOINT_OFFSET[f"{side}_Toe"]]
            out[t, JOINTS.index(f"{side}_Ankle")] = [fx, y, fz + JOINT_OFFSET[f"{side}_Ankle"]]
        out[t, JOINTS.index("Pelvis")] = [0.5 * (st["L"][0] + st["R"][0]), 0.0, 0.9 + max(st["L"][1], st["R"][1])]
    return out


def test_a_box_takes_its_height_from_its_own_footfalls():
    """Two stones one level apart in height are two surfaces, not one averaged surface.

    `_split_by_gap` has already decided these footfalls are disjoint footprints; giving each
    resulting box the *level* mean puts both wrong by half the disagreement. Every
    stepping-stone motion of the non-flat subset had this - `KIT/3/step_stones01_poses` read
    0.110/0.110 m for stones that measure 0.088 and 0.110.
    """
    motion = synth_stones(heights=(0.10, 0.125))
    events = detect_stance_events(motion, JOINTS, FPS)
    offsets = paired_sole_offsets(joint_surface_offsets(events))
    raised_levels = [lv for lv in cluster_levels(events, offsets=offsets) if _level_height(lv, offsets)[0] > 0.04]
    assert len(raised_levels) == 1, (
        f"the two stones have to land in one level for this to test anything; they are in {len(raised_levels)}"
    )

    terrain, report = fit_terrain_from_motion(motion, JOINTS, FPS)
    assert len(terrain) == 2, f"expected one box per stone, got {len(terrain)}"
    tops = sorted(b.top for b in terrain.boxes)
    assert tops[0] == pytest.approx(0.100, abs=0.008)
    assert tops[1] == pytest.approx(0.125, abs=0.008)
    assert validate_terrain(motion, JOINTS, terrain, FPS, offsets=report["joint_offsets"])["passed"]


def test_nearby_stepping_stones_do_not_merge_into_one_height_slab():
    """A step-length gap is still empty space, even when it is less than the old 0.60 m."""
    from terra.terrain import fit_terrain_from_motion

    motion = synth_stones(heights=(0.10, 0.125), gap=0.50)
    terrain, _ = fit_terrain_from_motion(motion, JOINTS, FPS)
    tops = sorted(round(b.top, 3) for b in terrain.boxes)
    assert len(terrain) == 2, f"nearby stones merged into {len(terrain)} box(es): {tops}"
    assert tops[0] == pytest.approx(0.100, abs=0.008)
    assert tops[1] == pytest.approx(0.125, abs=0.008)


def test_a_box_buried_under_a_taller_one_is_dropped():
    """A covered top face is a surface no foot can reach, and only confuses the solver.

    Boxes are solid from z=0, so `height_at` never returns a buried top; what the box does
    contribute is collision geometry inside other collision geometry, which gives the
    retargeter's non-penetration rows contact normals that cannot be satisfied together.
    """
    from terra.terrain.shapes import _drop_buried_boxes

    low = BoxSpec(pos=(0.0, 0.0, 0.05), size=(0.15, 0.15, 0.05), yaw=0.0, name="low")
    tall = BoxSpec(pos=(0.0, 0.0, 0.20), size=(1.2, 0.5, 0.20), yaw=0.0, name="tall")
    beside = BoxSpec(pos=(3.0, 0.0, 0.05), size=(0.15, 0.15, 0.05), yaw=0.0, name="beside")
    edge = BoxSpec(pos=(1.1, 0.0, 0.05), size=(0.3, 0.15, 0.05), yaw=0.0, name="edge")

    kept, dropped = _drop_buried_boxes([low, tall, beside, edge])
    assert [b.name for b in kept] == ["tall", "beside", "edge"]
    assert dropped == [("low", "tall")]

    # Nothing to bury against, and a box only partly covered, both survive untouched.
    assert _drop_buried_boxes([tall]) == ([tall], [])
    assert _drop_buried_boxes([low]) == ([low], [])


# ======================================================================================
# The seat: terrain evidence from the pelvis rather than from the feet
# ======================================================================================
#
# A chair leaves the feet on flat ground, so none of the tests above can fail if the seat
# channel breaks: the fit still returns a valid flat terrain and every foot-based number
# stays clean. These pin the discriminator instead - what separates sitting from the four
# other ways a motion puts the pelvis low and still.

#: Joint set for the seat tests. `JOINTS` above has no knees, and the conditional support
#: landmarks are exactly what the negative cases turn on.
SEAT_JOINTS = ["Pelvis", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe", "L_Knee", "R_Knee", "L_Wrist", "R_Wrist"]


def synth_sit(pelvis_z=0.56, pelvis_x=0.0, foot_x=0.40, knee_z=0.45, wrist_z=0.50, hold_s=2.0, n_walk_steps=6):
    """Walk in along +x, sit with the pelvis behind the planted feet, hold, then stand.

    The walk-in matters: it is what supplies the floor contacts the joint offsets are
    measured from, and `drop_seated_contacts` refuses to drop anything that would leave
    fewer than `MIN_STANDING_CONTACTS` of them.

    The knobs are the negative cases. Lower `pelvis_z` to sit on the floor, raise it to lean
    on a table, put `pelvis_x` under the feet to squat, drop `knee_z` to kneel, drop
    `wrist_z` to take the weight on the hands.

    The approach comes from in *front* of the chair and stops at it, because that is what a
    subject does and because the alternative is a motion that contradicts itself: feet that
    walk through the seat's own footprint are free-space evidence against the surface the
    pelvis is about to rest on, and the fitter is right to refuse it.
    """
    stance, swing = 40, 20
    frames = []
    state = {"L": (foot_x + 2.0, 0.0), "R": (foot_x + 2.0, 0.0)}

    xs = list(np.linspace(foot_x + 1.8, foot_x, n_walk_steps))
    for i, x in enumerate(xs):
        side = "L" if i % 2 == 0 else "R"
        prev = state[side]
        for f in range(swing):
            a = (f + 1) / swing
            state[side] = (prev[0] + a * (x - prev[0]), 0.15 * np.sin(np.pi * a))
            frames.append((dict(state), 0.90, 0.5 * (state["L"][0] + state["R"][0])))
        state[side] = (x, 0.0)
        frames.extend([(dict(state), 0.90, 0.5 * (state["L"][0] + state["R"][0]))] * stance)

    stand_x = 0.5 * (state["L"][0] + state["R"][0])
    for a in np.linspace(0.0, 1.0, 40):  # lower onto the seat
        frames.append((dict(state), 0.90 + a * (pelvis_z - 0.90), stand_x + a * (pelvis_x - stand_x)))
    frames.extend([(dict(state), pelvis_z, pelvis_x)] * int(hold_s * FPS))
    for a in np.linspace(0.0, 1.0, 40):  # and back up
        frames.append((dict(state), pelvis_z + a * (0.90 - pelvis_z), pelvis_x + a * (stand_x - pelvis_x)))
    frames.extend([(dict(state), 0.90, stand_x)] * stance)

    out = np.zeros((len(frames), len(SEAT_JOINTS), 3))
    idx = {n: SEAT_JOINTS.index(n) for n in SEAT_JOINTS}
    for t, (st, pz, px) in enumerate(frames):
        seated = pz < 0.88
        for side, y in (("L", 0.1), ("R", -0.1)):
            x, z = st[side]
            out[t, idx[f"{side}_Toe"]] = [x + 0.08, y, z + JOINT_OFFSET[f"{side}_Toe"]]
            out[t, idx[f"{side}_Ankle"]] = [x, y, z + JOINT_OFFSET[f"{side}_Ankle"]]
            # Knees and wrists only take their seated values once the figure is down; while
            # it is walking they ride at ordinary standing heights.
            out[t, idx[f"{side}_Knee"]] = [x - 0.05, y, knee_z if seated else 0.48]
            out[t, idx[f"{side}_Wrist"]] = [px + 0.10, y * 1.8, wrist_z if seated else 0.75]
        out[t, idx["Pelvis"]] = [px, 0.0, pz]
    return out


def seat_boxes(terrain):
    return [b for b in terrain.boxes if b.name.startswith(SEAT_GEOM_PREFIX)]


def test_a_chair_is_reconstructed_from_the_pelvis_alone():
    """The whole point: a surface the feet never touch, found because the pelvis rested on it.

    Both halves are pinned. The seat has to appear at the height that puts the *robot's*
    glutes on it - source pelvis minus `PELVIS_SEAT_OFFSET`, since the solver drives the
    pelvis body to the pelvis joint - and the ground under the feet has to stay flat, because
    a chair is not a step and nothing about it may reach the walking surface.
    """
    motion = synth_sit(pelvis_z=0.56)
    terrain, report = fit_terrain_from_motion(motion, SEAT_JOINTS, FPS)

    seats = seat_boxes(terrain)
    assert len(seats) == 1, f"expected one seat, got {[b.name for b in terrain.boxes]}"
    assert seats[0].top == pytest.approx(0.56 - PELVIS_SEAT_OFFSET, abs=0.005)
    assert not [b for b in terrain.boxes if not b.name.startswith(SEAT_GEOM_PREFIX)], (
        "the floor gained a box; a chair is not a step"
    )
    # The feet walked in along y=+-0.1 at z=0 and must still be over the floor.
    assert terrain.height_at(-1.0, 0.1) == pytest.approx(0.0)
    assert report["seat"]["n_rests"] >= 1
    assert "n_boundary_truncated_rests" not in report["seat"]


def test_a_seat_can_use_the_posed_body_surface_instead_of_a_pelvis_constant():
    """The source mesh supplies the height; the fitter must not learn it back."""

    motion = synth_sit(pelvis_z=0.56)
    rests = detect_seat_rests(motion, SEAT_JOINTS, FPS)
    support = [0.31] * len(rests)
    terrain, report = fit_terrain_from_motion(
        motion,
        SEAT_JOINTS,
        FPS,
        seat_support_heights=support,
    )

    seats = seat_boxes(terrain)
    assert len(seats) == 1
    assert seats[0].top == pytest.approx(0.31)
    assert report["seat_support_heights"] == support
    assert report["seat"]["height_source"] == "posed_body_surface"
    assert report["seat"]["seats"][0]["support_height_samples"] == support

    check = validate_terrain(
        motion,
        SEAT_JOINTS,
        terrain,
        FPS,
        offsets=report.get("joint_offsets"),
        seat_support_heights=report["seat_support_heights"],
    )
    assert check["passed"]
    assert check["seat_contact_error_max"] < 1e-9
    assert {entry["height_source"] for entry in check["seats"]} == {"posed_body_surface"}


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_a_seat_contact_truncated_by_a_clip_boundary_is_retained(boundary):
    """A recording boundary shortens the observation, not the physical contact."""

    motion = synth_sit(hold_s=1.0)
    pelvis_z = motion[:, SEAT_JOINTS.index("Pelvis"), 2]
    seated = np.flatnonzero(np.isclose(pelvis_z, 0.56))
    if boundary == "start":
        clipped = motion[seated[-6] :]
    else:
        clipped = motion[: seated[5] + 1]

    rests = detect_seat_rests(clipped, SEAT_JOINTS, FPS)

    assert len(rests) == 1
    assert rests[0].boundary == boundary
    assert rests[0].n_frames < int(0.4 * FPS)
    terrain, report = fit_terrain_from_motion(clipped, SEAT_JOINTS, FPS)
    assert len(seat_boxes(terrain)) == 1
    assert report["seat"]["n_boundary_truncated_rests"] == 1
    assert report["seat"]["seats"][0]["boundary_truncated"] == [boundary]


def test_a_short_interior_pelvis_pause_is_not_a_seat_contact():
    """Only the clip boundary can explain why the minimum duration was not observed."""

    motion = synth_sit(hold_s=0.1)

    assert not detect_seat_rests(motion, SEAT_JOINTS, FPS)


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_a_moving_clip_boundary_is_not_a_seat_contact(boundary):
    """A recording endpoint alone is not evidence that support continued off-camera."""

    motion = synth_sit(hold_s=0.0)
    pelvis_z = motion[:, SEAT_JOINTS.index("Pelvis"), 2]
    transition = np.flatnonzero((pelvis_z > 0.56) & (pelvis_z < 0.90))
    clipped = motion[transition[-1] :] if boundary == "start" else motion[: transition[0] + 1]

    assert not detect_seat_rests(clipped, SEAT_JOINTS, FPS)


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_a_one_frame_flexed_knee_boundary_pose_is_seat_evidence(boundary):
    """A seated posture can retain support when the recording contains no dwell time."""

    motion = synth_sit(hold_s=0.0)
    names = [*SEAT_JOINTS, "L_Hip", "R_Hip"]
    extended = np.zeros((len(motion), len(names), 3), dtype=float)
    extended[:, : len(SEAT_JOINTS)] = motion
    pelvis = motion[:, SEAT_JOINTS.index("Pelvis")]
    for side, y in (("L", 0.08), ("R", -0.08)):
        extended[:, names.index(f"{side}_Hip")] = pelvis + np.array([0.0, y, 0.0])

    pelvis_z = extended[:, names.index("Pelvis"), 2]
    lowered = np.flatnonzero((pelvis_z > 0.57) & (pelvis_z < 0.60))
    clipped = extended[lowered[-1] :] if boundary == "start" else extended[: lowered[0] + 1]

    rests = detect_seat_rests(clipped, names, FPS)

    assert len(rests) == 1
    assert rests[0].boundary == boundary
    assert rests[0].n_frames == 1
    assert rests[0].boundary_evidence == "flexed_knee_endpoint"


def test_source_derived_seat_height_uses_a_floor_threshold_not_a_chair_prior():
    """Low platforms are valid support; only floor-scale surfaces are suppressed."""

    motion = synth_sit(pelvis_z=0.56)
    n_rests = len(detect_seat_rests(motion, SEAT_JOINTS, FPS))
    low_platform, _ = fit_terrain_from_motion(
        motion,
        SEAT_JOINTS,
        FPS,
        seat_support_heights=[0.05] * n_rests,
    )
    floor_noise, _ = fit_terrain_from_motion(
        motion,
        SEAT_JOINTS,
        FPS,
        seat_support_heights=[0.03] * n_rests,
    )

    assert len(seat_boxes(low_platform)) == 1
    assert not seat_boxes(floor_noise)


def test_seat_support_height_count_and_values_are_validated():
    motion = synth_sit(pelvis_z=0.56)
    n_rests = len(detect_seat_rests(motion, SEAT_JOINTS, FPS))

    with pytest.raises(ValueError, match="one value per detected seat rest"):
        fit_terrain_from_motion(
            motion,
            SEAT_JOINTS,
            FPS,
            seat_support_heights=[0.31] * (n_rests + 1),
        )
    with pytest.raises(ValueError, match="must be finite"):
        fit_terrain_from_motion(
            motion,
            SEAT_JOINTS,
            FPS,
            seat_support_heights=[np.nan] * n_rests,
        )


@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"pelvis_z": 0.14}, "sitting on the floor: the surface is the floor"),
        ({"pelvis_z": 0.80}, "leaning on a table: the pelvis is not held from below"),
        ({"pelvis_x": 0.40}, "squatting: the pelvis is over its own feet"),
        ({"pelvis_z": 0.56, "wrist_z": 0.02}, "on all fours: the hands are carrying it"),
    ],
)
def test_the_other_ways_of_putting_the_pelvis_low_are_not_seats(kwargs, why):
    """Each negative is a real family in the archive, and each fails a different test.

    Statics is what ties them together: a body at rest whose weight *can* reach the ground
    through contacts already accounted for needs no extra surface to explain it. So the two
    that put the pelvis inside the support hull (squat) or add a contact to that hull (kneel,
    all fours) are refused by the hull test, and the two whose inferred surface is not a seat
    at all are refused by the height range.
    """
    terrain, _ = fit_terrain_from_motion(synth_sit(**kwargs), SEAT_JOINTS, FPS)
    assert not seat_boxes(terrain), f"built a seat for {why}"


def test_kneeling_is_not_sitting():
    """Kneeling is the negative that needs its own geometry, not a knob on `synth_sit`.

    It is the hardest of the five: the pelvis is low, still, and genuinely far from the feet,
    because the feet are behind it and are not what is carrying it. What is carrying it is the
    shins, and once the grounded knees join the support hull the pelvis is inside it - which
    is the whole statics argument, applied to the contact that is actually loaded. Over 96
    KIT `kneel*` clips this leaves none of them building a seat.
    """
    n = int(3.0 * FPS)
    motion = np.zeros((n, len(SEAT_JOINTS), 3))
    idx = {name: SEAT_JOINTS.index(name) for name in SEAT_JOINTS}
    for side, y in (("L", 0.1), ("R", -0.1)):
        motion[:, idx[f"{side}_Knee"]] = [0.20, y, 0.02]  # shins down, knees forward
        motion[:, idx[f"{side}_Ankle"]] = [-0.22, y, 0.06]  # feet tucked behind
        motion[:, idx[f"{side}_Toe"]] = [-0.34, y, 0.02]
        motion[:, idx[f"{side}_Wrist"]] = [0.35, y * 2, 0.55]
    motion[:, idx["Pelvis"]] = [-0.05, 0.0, 0.45]  # sitting back on the heels

    rests = detect_seat_rests(motion, SEAT_JOINTS, FPS)
    assert not rests, f"kneeling read as {len(rests)} seat rest(s)"

    # And it is the knees that do it: take them out of reach of the ground and the same
    # pose does read as a seat, which is what makes this test about the hull rather than
    # about the height.
    lifted = motion.copy()
    for side in ("L", "R"):
        lifted[:, idx[f"{side}_Knee"], 2] = 0.44
    assert detect_seat_rests(lifted, SEAT_JOINTS, FPS)


def test_walking_never_builds_a_seat():
    """The regression guard for every motion the pipeline already handles.

    The seat channel runs on every fit, so the whole non-flat corpus passes through it. It
    must be provably inert there, and `synth_walk` is the same motion the beam and stair
    tests above are built on.
    """
    terrain, report = fit_terrain_from_motion(synth_walk(), JOINTS, FPS)
    assert not seat_boxes(terrain)
    assert report["seat"]["n_rests"] == 0
    assert report["seat"]["n_seated_contacts_dropped"] == 0
    assert "n_boundary_truncated_rests" not in report["seat"]
    # And the terrain is exactly what the foot channel alone produces.
    off, off_report = fit_terrain_from_motion(synth_walk(), JOINTS, FPS, seat="off")
    assert terrain.boxes == off.boxes
    assert report["seat"]["mode"] == "auto"
    assert off_report["n_seats"] == 0
    assert off_report["seat"]["mode"] == "off"
    assert off_report["seat"]["n_rests"] == 0
    assert off_report["seat"]["n_seated_contacts_dropped"] == 0


def test_a_still_foot_under_a_seated_body_is_not_a_surface():
    """A foot parked in the air through a sit reads as a footfall, and must not build terrain.

    Contact detection is speed plus a *local* height minimum, which is what lets it find an
    elevated footfall with no height datum - and over a long sit the local window lies
    entirely inside the seated phase, so a foot resting on its heel with the toe 55 mm up is
    a local minimum for the whole of it. On `sitdown_standup-02-chair-hamada` that produced
    11 such contacts, worst 50 mm, and failed the fit gate on a clip whose floor is flat.

    The fit and the gate have to drop the same ones. Scoring against two conventions is the
    defect this module documents three times over, so the gate re-derives the rests from the
    terrain rather than being told about them.
    """
    motion = synth_sit(pelvis_z=0.56)
    # Park the right foot 55 mm up for the whole seated phase, as a real subject does.
    seated = motion[:, SEAT_JOINTS.index("Pelvis"), 2] < 0.88
    for j in ("R_Toe", "R_Ankle"):
        motion[seated, SEAT_JOINTS.index(j), 2] += 0.055

    terrain, report = fit_terrain_from_motion(motion, SEAT_JOINTS, FPS)
    assert report["seat"]["n_seated_contacts_dropped"] >= 1
    assert not [b for b in terrain.boxes if not b.name.startswith(SEAT_GEOM_PREFIX)], (
        "the parked foot was built as a surface"
    )
    check = validate_terrain(motion, SEAT_JOINTS, terrain, FPS, offsets=report.get("joint_offsets"))
    assert check["n_seated_contacts_dropped"] >= 1, "the gate kept what the fit dropped"
    assert check["passed"], check["raised_contact_error_max"]


def test_a_seat_is_solid_but_never_the_ground_under_a_foot():
    """`walkable` is the split between support and solidity, and both directions matter.

    Non-penetration must still see a chair - the robot may not pass through it - while every
    query about the surface *under a foot* must not, or a swing beside a chair is asked to
    clear 0.42 m and a stance beside one is scored against a seat it is standing next to.
    """
    seat = BoxSpec(pos=(0.0, 0.0, 0.2), size=(0.22, 0.22, 0.2), name="terrain_box_seat_0")
    step = BoxSpec(pos=(1.0, 0.0, 0.05), size=(0.3, 0.3, 0.05), name="terrain_box_0")
    terrain = TerrainSpec(boxes=(seat, step))

    assert terrain.height_at(0.0, 0.0) == pytest.approx(0.4)
    assert terrain.walkable.height_at(0.0, 0.0) == pytest.approx(0.0)
    assert terrain.walkable.height_at(1.0, 0.0) == pytest.approx(0.1)
    assert terrain.penetration(np.array([[0.0, 0.0, 0.2]])) > 0, "the seat stopped being solid"

    # Identity, not a copy, wherever there is no seat: this runs on every terrain fitted.
    flat = TerrainSpec(boxes=(step,))
    assert flat.walkable is flat


def test_every_seat_is_findable_as_static_environment():
    """A seat named outside `GEOM_PREFIX` is built, drawn, and collided with by nothing.

    The ramp work lost a whole 40-motion run to exactly this, with every signal reading
    clean, so the naming is pinned for the seat too.
    """
    from loco_mujoco.core.terrain.boxes import BoxTerrain

    terrain, _ = fit_terrain_from_motion(synth_sit(), SEAT_JOINTS, FPS)
    assert seat_boxes(terrain)
    for b in terrain.boxes:
        assert b.name.startswith(BoxTerrain.GEOM_PREFIX), b.name


def test_two_positions_of_one_chair_are_two_seats():
    """A subject who sits, moves the chair and sits again was supported by two surfaces.

    Merging them would put one box between the two, under neither sit. They are kept apart
    by `SEAT_SPLIT_GAP`; a subject who stands and sits again in the same place is one seat,
    which the single-seat test above already pins.
    """
    a = synth_sit(pelvis_z=0.56, pelvis_x=0.0, foot_x=0.40)
    b = synth_sit(pelvis_z=0.56, pelvis_x=3.2, foot_x=3.6)
    motion = np.concatenate([a, b], axis=0)
    terrain, _ = fit_terrain_from_motion(motion, SEAT_JOINTS, FPS)
    seats = seat_boxes(terrain)
    assert len(seats) == 2, f"got {len(seats)}: {[round(x.pos[0], 2) for x in seats]}"
    assert abs(seats[0].pos[0] - seats[1].pos[0]) > SEAT_SPLIT_GAP


def test_hull_distance_survives_the_degenerate_inputs_it_is_given():
    """Two feet side by side are four probes on nearly one line, every frame.

    A general hull routine either raises on that or returns a facet set with no interior,
    which is why this one is written out. Zero inside, exact outside, and no special case
    for the collinear and duplicate inputs a real motion supplies.
    """
    from terra.terrain.seats import _hull_distance

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert _hull_distance(np.array([0.5, 0.5]), square) == 0.0
    assert _hull_distance(np.array([2.0, 0.5]), square) == pytest.approx(1.0)
    assert _hull_distance(np.array([2.0, 2.0]), square) == pytest.approx(np.sqrt(2))

    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    assert _hull_distance(np.array([0.5, 0.3]), collinear) == pytest.approx(0.3)
    assert _hull_distance(np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]])) == 0.0
    assert _hull_distance(np.array([1.0, 0.0]), np.array([[0.0, 0.0]])) == pytest.approx(1.0)


def test_the_gate_scores_the_seat_it_was_given():
    """A seat the fit built and free space then cut away from under the pelvis is a failure.

    The foot terms cannot see it - the feet are on flat floor throughout - so the gate needs
    its own seat term, and `passed` has to depend on it or a terrain that supports nothing
    the motion did comes back clean.
    """
    motion = synth_sit(pelvis_z=0.56)
    terrain, report = fit_terrain_from_motion(motion, SEAT_JOINTS, FPS)
    good = validate_terrain(motion, SEAT_JOINTS, terrain, FPS, offsets=report.get("joint_offsets"))
    assert good["n_seat_rests"] >= 1
    assert good["seat_contact_error_max"] < 0.005
    assert good["passed"]

    # The same motion against a terrain whose seat is 0.15 m too low.
    sunk = TerrainSpec(
        boxes=tuple(
            BoxSpec(
                pos=(b.pos[0], b.pos[1], b.pos[2] - 0.075),
                size=(b.size[0], b.size[1], b.size[2] - 0.075),
                yaw=b.yaw,
                name=b.name,
            )
            if b.name.startswith(SEAT_GEOM_PREFIX)
            else b
            for b in terrain.boxes
        )
    )
    bad = validate_terrain(motion, SEAT_JOINTS, sunk, FPS, offsets=report.get("joint_offsets"))
    assert bad["seat_contact_error_max"] == pytest.approx(0.15, abs=0.01)
    assert not bad["passed"], "a seat 150 mm below the pelvis passed the gate"
