"""Tests for the terrain-retargeting quality metrics.

These guard the judgement, not the arithmetic. Every mode here has already passed a motion
it should have failed at least once: a stance-only test passed a beam crossing whose heel
never left the beam, and a clearance threshold fixed in millimetres passes a drag on a
staircase while failing a legitimate 20 mm shuffle along a beam. So what is pinned is the
part that decides pass or fail - that swing windows come from the gaps between a *foot's
own* stances, that the clearance and slip budgets scale with the source motion, and that
the phase-level counts a class table aggregates match the phases.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from terra.evaluation.annotations import RENDER_ARRAY_SPECS, frame_metadata
from terra.evaluation.terrain import (
    Measurement,
    Stance,
    Swing,
    Tolerances,
    _minimum_geom_distance_at_threshold,
    _on_timebase,
    _slip_window_end,
    _sole_support_distance,
    phases,
)
from terra.visualization.render import Flags, broadside_azimuth


@dataclass
class FakeEvent:
    """The fields of a stance event that `phases` reads."""

    joint: str
    start: int
    end: int


def test_source_joints_resample_on_declared_trimmed_output_clock():
    joints = np.zeros((6, 1, 3), dtype=float)
    joints[:, 0, 0] = np.arange(6) / 50.0
    output_times = np.array([0.02, 0.03, 0.04, 0.05, 0.06])

    aligned, fps = _on_timebase(joints, 50.0, len(output_times), 100.0, target_times_s=output_times)

    assert fps == pytest.approx(100.0)
    assert aligned[:, 0, 0] == pytest.approx(output_times)


def stance(
    side="left",
    start=0,
    end=50,
    surface_z=0.0,
    median_clear=0.0,
    min_clear=0.0,
    travel=0.05,
    src_travel=0.05,
    fore_clear=0.0,
    src_fore_clear=0.0,
    hind_clear=0.0,
    src_hind_clear=0.0,
) -> Stance:
    return Stance(
        side=side,
        start=start,
        end=end,
        surface_z=surface_z,
        median_clear=median_clear,
        min_clear=min_clear,
        travel=travel,
        src_travel=src_travel,
        fore_clear=fore_clear,
        src_fore_clear=src_fore_clear,
        hind_clear=hind_clear,
        src_hind_clear=src_hind_clear,
    )


def swing(side="left", start=50, end=90, peak=0.08, src_peak=0.08, touched=0.02, exact_peak=0.08) -> Swing:
    return Swing(side=side, start=start, end=end, peak=peak, src_peak=src_peak, touched=touched, exact_peak=exact_peak)


def measurement(stances=(), swings=(), tol=None, **kwargs) -> Measurement:
    m = Measurement(
        motion="test",
        n_frames=200,
        n_boxes=1,
        tol=tol or Tolerances(),
        face_pen={"top": 0.0, "side": 0.0},
        face_worst={"top": "", "side": ""},
        **kwargs,
    )
    m.stances, m.swings = list(stances), list(swings)
    return m


# --- phase windows ---------------------------------------------------------------------


def test_swings_are_the_gaps_between_one_foot_s_own_stances():
    events = [FakeEvent("L_Toe", 0, 50), FakeEvent("R_Toe", 30, 80), FakeEvent("L_Toe", 90, 140)]
    st, sw = phases(events, "left", 200)
    assert st == [(0, 50), (90, 140)]
    # 50-90 is the left foot in the air. The right foot's stance overlapping it is not a
    # boundary: interleaving both feet's events would leave 50-80 as "contact" and hide the
    # left swing entirely.
    assert sw == [(50, 90)]


def test_toe_and_ankle_events_are_merged_into_one_side_stance():
    events = [
        FakeEvent("L_Toe", 10, 40),
        FakeEvent("L_Ankle", 20, 50),
        FakeEvent("L_Toe", 80, 100),
    ]

    stance, swing = phases(events, "left", 120)

    assert stance == [(10, 50), (80, 100)]
    assert swing == [(50, 80)]


def test_nothing_is_expected_before_the_first_or_after_the_last_footfall():
    events = [FakeEvent("L_Toe", 40, 80), FakeEvent("L_Toe", 120, 160)]
    _, sw = phases(events, "left", 300)
    assert sw == [(80, 120)]


def test_a_stance_running_past_the_end_of_the_trajectory_is_clipped():
    # The retargeted trajectory can be a frame or two shorter than the source it is timed
    # from; an unclipped window indexes past the end of the distance arrays.
    st, _ = phases([FakeEvent("R_Toe", 100, 500)], "right", 300)
    assert st == [(100, 300)]


def test_a_stance_starting_past_the_end_is_dropped_rather_than_inverted():
    st, _ = phases([FakeEvent("R_Toe", 400, 500)], "right", 300)
    assert st == []


# --- support ---------------------------------------------------------------------------


def test_a_stance_foot_hovering_above_its_surface_is_floating():
    tol = Tolerances(float_tol=0.02)
    assert stance(median_clear=0.04).flags(tol) == ["FLOATING"]
    assert stance(median_clear=0.01).flags(tol) == []


def test_floating_is_judged_on_the_median_not_the_minimum():
    # A foot that touches down for a single frame in the middle of an otherwise airborne
    # stance is floating. Scored on the minimum it would read as perfect contact.
    tol = Tolerances(float_tol=0.02)
    assert "FLOATING" in stance(median_clear=0.05, min_clear=-0.001).flags(tol)


def test_a_stance_foot_inside_its_surface_is_penetrating():
    tol = Tolerances(pen_tol=0.005)
    assert stance(min_clear=-0.012).flags(tol) == ["PENETRATING"]
    # The solver's own contact tolerance is ~1 mm and must not read as a failure.
    assert stance(min_clear=-0.001).flags(tol) == []


def test_exact_selected_support_distance_has_no_ramp_projection_or_nearby_side_bias():
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec

    ramp = BoxSpec(
        pos=(0.0, 0.0, 0.20),
        size=(1.0, 0.50, 0.10),
        pitch=np.deg2rad(-10.0),
        name="ramp",
    )
    quat = " ".join(str(value) for value in ramp.quat)
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="4 4 0.1"/>
            <geom name="ramp" type="box" pos="{" ".join(str(v) for v in ramp.pos)}"
                  size="{" ".join(str(v) for v in ramp.size)}" quat="{quat}"/>
            <body name="foot">
              <freejoint/>
              <geom name="sole" type="ellipsoid" size="0.10 0.05 0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp")
    sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sole")
    terrain = TerrainSpec(boxes=(ramp,))
    support_geoms = {ramp: ramp_id}
    surface = np.array([0.0, 0.0, float(ramp.top_plane_at(0.0, 0.0))])
    normal = ramp.rotation[:, 2]

    for expected_gap in (0.0, 0.030, -0.010):
        data.qpos[:3] = surface + normal * (0.010 + expected_gap)
        data.qpos[3:7] = ramp.quat
        mujoco.mj_forward(model, data)
        measured = _sole_support_distance(
            model,
            data,
            [sole_id],
            terrain,
            support_geoms,
            floor_id,
        )
        assert measured == pytest.approx(expected_gap, abs=1e-6)

    # The discarded vertical construction compares the sole's downhill edge with the ramp
    # plane at its centre. It calls this exactly tangent pose a material penetration.
    data.qpos[:3] = surface + normal * 0.010
    data.qpos[3:7] = ramp.quat
    mujoco.mj_forward(model, data)
    floor_gap = mujoco.mj_geomDistance(model, data, sole_id, floor_id, 2.0, None)
    centre = data.geom_xpos[sole_id]
    old_vertical_gap = floor_gap - float(ramp.top_plane_at(centre[0], centre[1]))
    assert old_vertical_gap < -0.005

    # A sole on the floor just beside the ramp selects the floor. Its proximity to the
    # ramp's side cannot turn into apparent support contact.
    data.qpos[:3] = (0.0, ramp.size[1] + 0.02, 0.010)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    assert _sole_support_distance(
        model,
        data,
        [sole_id],
        terrain,
        support_geoms,
        floor_id,
    ) == pytest.approx(0.0, abs=1e-7)


def test_thresholded_geom_distance_does_not_turn_clipped_far_pairs_into_contact():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="terrain" type="box" pos="0 0 0" size="0.5 0.5 0.5"/>
            <body name="probe" pos="0 0 2">
              <geom name="robot" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot")
    threshold = 0.1

    clipped = mujoco.mj_geomDistance(model, data, robot, terrain, threshold, None)
    measured = _minimum_geom_distance_at_threshold(model, data, (robot,), (terrain,), threshold)

    assert clipped == threshold
    assert measured > threshold


# --- slip ------------------------------------------------------------------------------


def test_slip_is_measured_against_the_source_s_own_travel():
    tol = Tolerances(slip_tol=0.03)
    # 95 mm of absolute travel inside one stance, all of it in the source too. A slow,
    # careful beam crossing does exactly this, and an absolute budget fails every stance.
    assert stance(travel=0.095, src_travel=0.090).flags(tol) == []
    assert stance(travel=0.095, src_travel=0.020).flags(tol) == ["SLIP"]


def test_slip_liftoff_guard_is_short_and_trailing_only():
    tol = Tolerances()
    assert tol.slip_liftoff_s == pytest.approx(0.05)
    assert _slip_window_end(10, 70, fps=100.0, liftoff_s=tol.slip_liftoff_s) == 65
    assert _slip_window_end(10, 70, fps=120.0, liftoff_s=tol.slip_liftoff_s) == 64
    assert _slip_window_end(10, 12, fps=100.0, liftoff_s=tol.slip_liftoff_s) == 11


def test_a_foot_steadier_than_the_source_never_counts_as_slipping():
    assert stance(travel=0.01, src_travel=0.09).flags(Tolerances()) == []


# --- clearance -------------------------------------------------------------------------


def test_clearance_is_required_as_a_fraction_of_what_the_source_achieved():
    tol = Tolerances(clearance_frac=0.5, clearance_floor=0.010)
    # A 90 mm stair swing reproduced at 30 mm is a drag; a 20 mm beam shuffle reproduced at
    # 15 mm is the motion. A single absolute threshold cannot separate them.
    assert swing(peak=0.030, src_peak=0.090).dragging(tol)
    assert not swing(peak=0.015, src_peak=0.020).dragging(tol)


def test_the_floor_catches_a_swing_the_source_barely_lifted():
    tol = Tolerances(clearance_frac=0.5, clearance_floor=0.010)
    # Half of 4 mm is 2 mm, which any scraping foot clears. The floor is what makes a
    # source that itself barely lifts not license a foot that never leaves the surface.
    assert swing(peak=0.003, src_peak=0.004).dragging(tol)
    assert swing(peak=0.003, src_peak=0.004).required(tol) == pytest.approx(0.010)


def test_clearing_more_than_the_source_is_not_a_failure():
    assert not swing(peak=0.150, src_peak=0.090).dragging(Tolerances())


def test_a_swing_the_source_spent_below_its_own_surface_has_no_lift_to_be_a_fraction_of():
    """`src_peak <= 0` is a reference that does not describe the swing, not a small lift.

    It happens where the foot passes beside something taller than itself. Applying the
    absolute floor there failed the robot for not clearing a surface the human was under for
    the whole swing - eight swings of the subset, source peaks of -0 to -182 mm.
    """
    w = swing(peak=-0.001, src_peak=-0.047, touched=0.010)
    assert not w.measurable()
    assert not w.dragging(Tolerances())
    assert not w.scraping(Tolerances()), "and it never reached the terrain either"


def test_such_a_swing_is_still_caught_if_it_actually_hits_the_terrain():
    w = swing(peak=-0.001, src_peak=-0.047, touched=-0.020)
    assert not w.dragging(Tolerances())
    assert w.scraping(Tolerances()), "the absolute test needs no reference and still applies"


def test_a_source_that_barely_lifted_still_gets_the_floor():
    # The guard above must not swallow the case the floor exists for: a real but tiny lift.
    assert swing(peak=0.003, src_peak=0.004).dragging(Tolerances())


# --- the motion-level verdict ----------------------------------------------------------


def test_every_failing_phase_is_named_in_the_verdict():
    m = measurement(
        stances=[stance(median_clear=0.05), stance(start=100, end=150, min_clear=-0.02)],
        swings=[swing(peak=0.001, src_peak=0.090)],
    )
    reasons = m.fails()
    assert len(reasons) == 3
    assert any("floating" in r for r in reasons)
    assert any("penetrating" in r for r in reasons)
    assert any("cleared" in r for r in reasons)


def test_a_clean_motion_passes_and_a_body_through_a_box_side_does_not():
    m = measurement(stances=[stance()], swings=[swing()])
    assert m.summary()["passed"] == 1
    # The side face is the one a step-up approaches, and it hides behind a clean top-face
    # number whenever penetration is minimised over all faces at once.
    m.face_pen["side"] = 0.062
    assert m.summary()["passed"] == 0
    assert "side face" in " ".join(m.fails())


def test_legs_through_each_other_fails_a_motion_whose_feet_are_perfect():
    m = measurement(stances=[stance()], swings=[swing()], selfpen_worst=0.035)
    assert m.summary()["passed"] == 0
    assert m.summary()["n_floating"] == 0


def test_the_summary_counts_phases_not_motions():
    m = measurement(
        stances=[stance(median_clear=0.05), stance(start=60, end=100, median_clear=0.05), stance(start=110, end=150)],
        swings=[swing(peak=0.001, src_peak=0.090)],
    )
    s = m.summary()
    assert (s["n_stances"], s["n_floating"], s["n_swings"], s["n_dragging"]) == (3, 2, 1, 1)
    # The ratio to the source is what makes clearance comparable across a stair riser and
    # a beam shuffle, so it is reported alongside the raw millimetres.
    assert s["clear_ratio_median"] == pytest.approx(0.001 / 0.090)


def test_renderer_marks_brief_source_stance_floating_from_per_frame_score(tmp_path):
    """A brief lift must be visible even when the stance median still passes."""
    frames = tmp_path / "frames"
    frames.mkdir()
    arrays = {
        key: np.zeros((5, 2) if ndim == 2 else 5, dtype=bool if kind == "b" else float)
        for key, (ndim, kind) in RENDER_ARRAY_SPECS.items()
    }
    arrays["support_floating"][:, 0] = [False, False, True, False, False]
    arrays["source_stance"][:, 0] = [False, True, True, True, False]
    arrays["contact_gap"][:] = [False, True, True, False, False]
    np.savez_compressed(
        frames / "test.npz",
        **arrays,
        **frame_metadata(motion="test", method="terra", n_frames=5),
    )

    flags = Flags.load("test", tmp_path, 5)

    assert flags.masks["float"].tolist() == [False, False, True, False, False]
    assert flags.masks["contact_gap"].tolist() == [False, True, True, False, False]
    assert flags.stance["left"].tolist() == [False, True, True, True, False]


# --- a short "stance" is not a footfall -------------------------------------------------
# `detect_stance_events` marks a toe that is slow and low, which on a jump or a landing
# fires for a handful of frames. `0017_JumpingOnBench001` at 120 fps yields 66 such stances
# with a median duration of 117 ms against 500-700 ms for a walking footfall, and scoring
# each as a footfall is what let the two `obstacle` clips dominate every worst-list.


def test_a_stance_too_short_to_be_a_footfall_is_dropped():
    events = [FakeEvent("L_Toe", 0, 60), FakeEvent("L_Toe", 100, 108), FakeEvent("L_Toe", 150, 210)]
    kept, _ = phases(events, "left", 300, fps=120.0, min_stance_s=0.15)
    assert kept == [(0, 60), (150, 210)], "the 8-frame stance is 67 ms, not a footfall"


def test_dropping_a_short_stance_does_not_invent_one_long_swing_across_it():
    """The foot's whereabouts during a dropped stance is unknown, so the swings stay cut."""
    events = [FakeEvent("L_Toe", 0, 60), FakeEvent("L_Toe", 100, 108), FakeEvent("L_Toe", 150, 210)]
    _, swings = phases(events, "left", 300, fps=120.0, min_stance_s=0.15)
    assert swings == [(60, 100), (108, 150)]


def test_without_a_frame_rate_every_stance_is_kept():
    events = [FakeEvent("L_Toe", 0, 60), FakeEvent("L_Toe", 100, 108)]
    assert phases(events, "left", 300)[0] == [(0, 60), (100, 108)]


# --- the two swing questions ------------------------------------------------------------
# `dragging` is relative and so must be measured the same way on both sides; `scraping` is
# absolute and uses the exact geom query, which no foot pose can flatter.


def test_a_swing_that_lifts_as_much_as_the_source_is_not_dragging_however_low_its_toe_hangs():
    # The exact query sees a toe 2 mm off the ground; the like-for-like comparison against
    # the source does not, because the source estimate cannot see a human toe either.
    w = swing(peak=0.080, src_peak=0.090, exact_peak=0.002, touched=0.002)
    assert not w.dragging(Tolerances())
    assert not w.scraping(Tolerances())


def test_a_sole_that_enters_the_terrain_mid_swing_is_scraping_even_if_it_lifted_enough():
    w = swing(peak=0.080, src_peak=0.090, touched=-0.020)
    assert not w.dragging(Tolerances()), "precondition: it cleared what the source did"
    assert w.scraping(Tolerances())


def test_both_swing_modes_are_counted_and_named():
    m = measurement(swings=[swing(peak=0.001, src_peak=0.090, touched=-0.02)])
    s = m.summary()
    assert (s["n_dragging"], s["n_scraping"]) == (1, 1)
    assert len(m.fails()) == 2, "one swing can fail both ways and each must be reported"


# --- a foot standing on its heel --------------------------------------------------------
# Every other support quantity is a minimum over the whole sole, which a heel-only stance
# satisfies with its toes in the air.


def test_a_stance_on_the_heel_is_caught_where_the_source_stood_flat():
    s = stance(median_clear=0.0, min_clear=0.0, fore_clear=0.045, src_fore_clear=0.005)
    assert s.forefoot_up(Tolerances())
    assert "FOREFOOT_UP" in s.flags(Tolerances())


def test_a_heel_strike_the_human_also_made_is_not_a_defect():
    s = stance(fore_clear=0.045, src_fore_clear=0.050)
    assert not s.forefoot_up(Tolerances()), "the source's forefoot is up too"


def test_a_flat_stance_passes_and_an_unmeasured_one_is_not_guessed_at():
    assert not stance(fore_clear=0.002, src_fore_clear=0.001).forefoot_up(Tolerances())
    assert not stance(fore_clear=float("nan"), src_fore_clear=0.0).forefoot_up(Tolerances())


# --- a foot standing on its forefoot ----------------------------------------------------


def test_a_stance_on_the_forefoot_is_caught_where_the_source_stood_flat():
    s = stance(median_clear=0.0, min_clear=0.0, hind_clear=0.045, src_hind_clear=0.005)
    assert s.hindfoot_up(Tolerances())
    assert "HINDFOOT_UP" in s.flags(Tolerances())


def test_physiological_source_heel_rise_is_not_a_defect():
    s = stance(hind_clear=0.045, src_hind_clear=0.050)
    assert not s.hindfoot_up(Tolerances()), "the source's hindfoot is up too"


def test_a_flat_hindfoot_passes_and_an_unmeasured_one_is_not_guessed_at():
    assert not stance(hind_clear=0.002, src_hind_clear=0.001).hindfoot_up(Tolerances())
    assert not stance(hind_clear=float("nan"), src_hind_clear=0.0).hindfoot_up(Tolerances())


# --- camera ----------------------------------------------------------------------------


def test_the_camera_looks_across_the_direction_of_travel():
    # Walking along +x is viewed from -y or +y, never from behind: down a staircase the
    # body occludes exactly the feet the clip exists to show.
    walk_x = np.stack([np.linspace(0, 4, 50), np.zeros(50)], axis=1)
    assert broadside_azimuth(walk_x) == pytest.approx(90.0)
    walk_y = np.stack([np.zeros(50), np.linspace(0, 4, 50)], axis=1)
    assert broadside_azimuth(walk_y) == pytest.approx(180.0)


def test_a_motion_that_stays_put_keeps_the_default_camera():
    still = np.zeros((50, 2))
    assert broadside_azimuth(still, fallback=120.0) == 120.0
