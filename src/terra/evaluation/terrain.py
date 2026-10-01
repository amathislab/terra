"""Package-owned terrain interaction measurement for the unified evaluator.

This is the metric core of [check_beam_motion.py](check_beam_motion.py), lifted out so that
the single-motion report, the subset scoring and the video renderer all read the same
numbers. The renderer in particular has to: a clip whose foot pane is tinted on frames the
score sheet counts as clean is worse than no video at all.

Four things decide whether a terrain motion is reproducible, and none of them is landmark
error - a solution can track every landmark to a centimetre and still be walking through the
staircase:

* **support** - a planted foot resting *on* its surface, neither floating above it nor
  inside it;
* **clearance** - a swinging foot actually leaving that surface;
* **slip** - a planted foot staying where it landed;
* **self-collision** - the legs not passing through each other.

Plus **body-vs-terrain penetration**, split by which face of a box it is against: the whole
terrain record so far has been read off a distance minimised over all faces, so a body driven
through the *end* of a beam hides behind a clean top-face number.

Two rules keep the thresholds honest:

1. **Contact timing comes from the source motion.** The retargeted feet are the thing being
   judged, so their own contact detection cannot be the reference.
2. **Clearance and slip are scored against the source's own**, never absolutely. These
   motions are deliberately low and careful: a fixed clearance threshold either passes a
   drag or fails a legitimate shuffle, and a fixed slip budget fails every stance of a slow
   beam crossing that is tracking the source to within a centimetre.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from loco_mujoco.core.terrain.boxes import SEAT_GEOM_PREFIX
from terra.evaluation.settings import QUALITY_THRESHOLDS
from terra.paths import StorageRoots
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

#: Geoms making up each foot's sole, by the side used in the report.
FOOT_GEOMS = {"left": ("l_foot_col", "l_bofoot_col"), "right": ("r_foot_col", "r_bofoot_col")}
#: Geoms standing in for the buttocks: the ellipsoids MyoFullBody carries on its `pelvis`
#: body. They are the only collidable geometry between the hips and the thighs, and they
#: are what a seated robot rests on.
SEAT_GEOMS = ("r_pelvis_col", "l_pelvis_col")
#: The forefoot half of that sole - the ball and the toes. Scored separately because every
#: other support quantity here is a *minimum over the whole sole*, which a foot standing on
#: its heel satisfies with its toes in the air.
FOREFOOT_GEOMS = {"left": ("l_bofoot_col",), "right": ("r_bofoot_col",)}
#: Robot body standing in for each source contact joint, for the like-for-like swing
#: comparison. These are the pairs `SMPLH_TO_MYOFULLBODY` maps, and the offsets that turn
#: them into sole heights are the ones `robot_sole_offsets` measures at `qpos0`.
PROBE_BODIES = {
    "left": (("L_Ankle", "talus_l"), ("L_Toe", "toes_l")),
    "right": (("R_Ankle", "talus_r"), ("R_Toe", "toes_r")),
}
#: Body whose origin measures slip, and the source joint it is compared against. It has to
#: be one *fixed* point on the foot - tracking whichever sole geom is currently closest to
#: the ground reads heel-to-toe roll as 100+ mm of travel on a foot that never moved - **and
#: the two sides have to be the same point**. They were not: the robot's heel (`calcn`) was
#: compared against the human's *toe*, and a foot pivoting about a planted toe moves its heel
#: a long way while the toe does not. Measured over 91 stances of the worst-slipping motions,
#: that mismatch flagged 19% as slipping where the mapped ankle-to-ankle correspondence flags
#: 8% and toe-to-toe flags 4%. `talus` <-> `*_Ankle` is the pair `SMPLH_TO_MYOFULLBODY`
#: itself maps, and the ankle is the steadier of the two probes.
SLIP_BODY = {"left": "talus_l", "right": "talus_r"}
SLIP_JOINT = {"left": "L_Ankle", "right": "R_Ankle"}
#: Source contact joint driving each side's stance timing.
TOE_OF = {"left": "L_Toe", "right": "R_Toe"}

# OmniRetarget's terrain-contact metric evaluates the mapped collision body for
# both hands, both heels/ankles, and both toes.
TERRAIN_CONTACT_BODIES = (
    "3distph_l",
    "3distph_r",
    "talus_l",
    "talus_r",
    "toes_l",
    "toes_r",
)
TERRAIN_CONTACT_SOURCE_JOINTS = (
    "L_Middle3",
    "R_Middle3",
    "L_Ankle",
    "R_Ankle",
    "L_Toe",
    "R_Toe",
)


def signed_distance_to_terrain_boxes(points: np.ndarray, terrain) -> np.ndarray:
    """Exact signed distance from world-space points to the nearest terrain box."""
    points = np.asarray(points, dtype=float)
    flat = points.reshape(-1, 3)
    if not terrain.boxes:
        return np.full(points.shape[:-1], np.inf)
    distances = []
    for box in terrain.boxes:
        delta = np.abs(box.to_local(flat)) - np.asarray(box.size, dtype=float)
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
        inside = np.minimum(np.max(delta, axis=1), 0.0)
        distances.append(outside + inside)
    return np.min(distances, axis=0).reshape(points.shape[:-1])


def source_shape_path(model: str, cache_root: Path | None) -> Path:
    """Resolve source-landmark calibration from the evaluated artifact cache."""

    from terra.runtime import shape_cache_path

    return shape_cache_path(model, cache_root)


#: Source joints bracketing each side's contact patch, used for the source clearance
#: reference. Heel clearance is what a low shuffling step actually risks, and the ankle is
#: the only source joint that tracks it.
SOLE_JOINTS = {
    "left": tuple(joint for joint in DEFAULT_CONTACT_JOINTS if joint.startswith("L_")),
    "right": tuple(joint for joint in DEFAULT_CONTACT_JOINTS if joint.startswith("R_")),
}
#: Left/right pairs whose interpenetration means the pose is geometrically impossible, as
#: opposed to an arm brushing the ribcage. These prefixes intentionally select the distal
#: collision geoms on MyoFullBody; proximal capsule overlap has not been calibrated as a
#: quality gate and made the filter substantially stricter than visual review.
LIMB_PREFIXES = (("l_foot", "r_foot"), ("l_bofoot", "r_bofoot"))


@dataclass(frozen=True)
class Tolerances:
    """The budget each failure mode is allowed. Defaults match `check_beam_motion.py`."""

    float_tol: float = QUALITY_THRESHOLDS["float_tol"]  # m above which a stance foot floats
    pen_tol: float = QUALITY_THRESHOLDS["pen_tol"]  # m below a surface that counts as penetration
    clearance_frac: float = QUALITY_THRESHOLDS["clearance_frac"]  # of source peak swing clearance
    clearance_floor: float = QUALITY_THRESHOLDS["clearance_floor"]  # absolute peak swing clearance
    slip_tol: float = QUALITY_THRESHOLDS["slip_tol"]  # m of stance travel beyond the source
    #: The slow-and-low toe event includes the last few frames of pre-swing: the heel and
    #: then the whole foot begin moving while the toe is still low enough to be called in
    #: contact. Treating the maximum over that transition as planted-foot travel flags a
    #: visually stable stair descent for what happens during lift-off. Ignore only this
    #: short trailing edge; any earlier or sustained slide still contributes in full.
    slip_liftoff_s: float = QUALITY_THRESHOLDS["slip_liftoff_s"]
    selfpen_tol: float = QUALITY_THRESHOLDS["selfpen_tol"]  # m of left/right limb interpenetration
    scrape_tol: float = QUALITY_THRESHOLDS["scrape_tol"]  # m a swinging sole may enter terrain
    forefoot_tol: float = QUALITY_THRESHOLDS["forefoot_tol"]  # m forefoot clearance in stance
    hindfoot_tol: float = QUALITY_THRESHOLDS["hindfoot_tol"]  # m hindfoot clearance in stance
    #: Shortest source contact that counts as a footfall, seconds. A stance detected from
    #: "slow and low" fragments on dynamic motions - `0017_JumpingOnBench001` at 120 fps
    #: yields 66 "stances" with a median duration of 117 ms, 22 of them under 83 ms, where a
    #: walking footfall runs 500-700 ms. Each fragment was then scored as a footfall that
    #: must not float or slip, which is why the two `obstacle` clips dominated every
    #: worst-list. Over the subset this drops 95 of 997 stances and 29 of 125 stance
    #: failures, almost all of them in those two clips.
    min_stance_s: float = QUALITY_THRESHOLDS["min_stance_s"]
    #: How far the glutes may sit above a seat before the figure is not on it, metres, and
    #: how far into it they may go. Both are looser than the foot's `float_tol` and
    #: `pen_tol`, deliberately and for a reason that is about the model rather than about
    #: the retargeting: `r_pelvis_col`/`l_pelvis_col` are rigid ellipsoids standing in for
    #: the buttocks, which in a real sit compress by a centimetre or two, and the seat's own
    #: height is placed from a pelvis-to-glute offset measured at `qpos0` that moves ~20 mm
    #: over the pelvis tilts a seated pose actually uses (`PELVIS_SEAT_OFFSET`). Scoring at
    #: the foot's 20/5 mm would report that modelling slack as a retargeting failure. The
    #: continuous `seat_clear_*` columns are reported alongside so the threshold is not the
    #: only thing on offer.
    seat_float_tol: float = QUALITY_THRESHOLDS["seat_float_tol"]
    seat_pen_tol: float = QUALITY_THRESHOLDS["seat_pen_tol"]


@dataclass
class Stance:
    """One source-timed stance phase, and how the retargeted foot behaved through it."""

    side: str
    start: int
    end: int
    surface_z: float
    median_clear: float
    min_clear: float
    travel: float
    src_travel: float
    #: Exclusive end of the window used for slip. Support modes still use the complete
    #: source stance through ``end``.
    slip_end: int | None = None
    #: Median forefoot clearance over the stance, and the source's own over the same window.
    #: Every other quantity here is a *minimum* over the whole sole, which a foot resting on
    #: its heel satisfies with its toes in the air; this is the only one that can see that.
    fore_clear: float = float("nan")
    src_fore_clear: float = float("nan")
    #: Median hindfoot clearance over the stance, and the source's ankle-side proxy over
    #: the same window.  This is the mirror image of ``fore_clear`` and detects tip-toe
    #: support without treating physiological heel rise as a retargeting defect.
    hind_clear: float = float("nan")
    src_hind_clear: float = float("nan")

    @property
    def excess(self) -> float:
        return self.travel - self.src_travel

    def forefoot_up(self, tol: Tolerances) -> bool:
        """Whether the forefoot is off the surface where the source's own is on it.

        Judged against the source rather than absolutely: a stance the human spends on its
        heel is a heel strike, not a defect to correct.
        """
        if not np.isfinite(self.fore_clear) or not np.isfinite(self.src_fore_clear):
            return False
        return self.fore_clear > tol.forefoot_tol and self.src_fore_clear <= tol.forefoot_tol

    def hindfoot_up(self, tol: Tolerances) -> bool:
        """Whether the heel is off the surface where the source's own heel is on it."""
        if not np.isfinite(self.hind_clear) or not np.isfinite(self.src_hind_clear):
            return False
        return self.hind_clear > tol.hindfoot_tol and self.src_hind_clear <= tol.hindfoot_tol

    def flags(self, tol: Tolerances) -> list[str]:
        out = []
        if self.median_clear > tol.float_tol:
            out.append("FLOATING")
        if self.min_clear < -tol.pen_tol:
            out.append("PENETRATING")
        if self.excess > tol.slip_tol:
            out.append("SLIP")
        if self.forefoot_up(tol):
            out.append("FOREFOOT_UP")
        if self.hindfoot_up(tol):
            out.append("HINDFOOT_UP")
        return out


@dataclass
class Swing:
    """One swing between two stances of the same foot.

    ``peak`` and ``src_peak`` compare robot and source clearance using the same
    probe-point convention. ``touched`` records the closest exact sole-to-terrain
    distance, which detects scraping without a source reference.
    """

    side: str
    start: int
    end: int
    #: Peak clearance, robot and source measured the same way: the lowest probe point of the
    #: foot, each dropped by its own flat-stance offset, over the surface beneath the foot.
    peak: float
    src_peak: float
    touched: float  # closest the sole came to any terrain face, exact geom query
    #: Peak clearance from the exact geom query. Reported, not scored - see above.
    exact_peak: float = float("nan")

    def required(self, tol: Tolerances) -> float:
        return max(tol.clearance_floor, tol.clearance_frac * self.src_peak)

    def measurable(self) -> bool:
        """Whether the source demonstrated a lift for this swing to be a fraction of.

        `src_peak <= 0` means the source's own sole never rose above the surface beneath the
        foot over the whole swing - which is not a source that barely lifted, it is a
        surface reference that does not describe this swing. It happens where the foot
        passes beside something taller than itself: measured over the subset, eight swings
        had source peaks of -0 to -182 mm, and the 10 mm absolute floor was failing the
        robot for not clearing a surface the human was under for the entire swing.

        Where this is False the swing is judged by :meth:`scraping` alone, which asks the
        absolute question and needs no reference.
        """
        return self.src_peak > 0.0

    def dragging(self, tol: Tolerances) -> bool:
        return self.measurable() and self.peak < self.required(tol)

    def scraping(self, tol: Tolerances) -> bool:
        """Whether the sole entered the terrain during the swing."""
        return self.touched < -tol.scrape_tol


@dataclass
class Seat:
    """One source-timed seated phase, and how the retargeted glutes behaved through it.

    The pelvis counterpart of :class:`Stance`, and it exists because no foot-based mode can
    see this failure: while the subject is seated both feet are on flat floor, perfectly
    supported, and a robot whose glutes hang 80 mm above the chair or sink 60 mm into it
    scores a clean pass on every other quantity in this module.

    The gap is an exact `mj_geomDistance` from the glute ellipsoids to the seat geoms, so it
    needs no offset convention on either side - unlike the foot, where a source joint centre
    has to be dropped to a sole before the two can be compared.
    """

    start: int
    end: int
    surface_z: float
    median_clear: float  # median over the phase of the glute-to-seat gap
    min_clear: float
    src_pelvis_z: float

    def flags(self, tol: Tolerances) -> list[str]:
        out = []
        if self.median_clear > tol.seat_float_tol:
            out.append("SEAT_FLOATING")
        if self.min_clear < -tol.seat_pen_tol:
            out.append("SEAT_PENETRATING")
        return out


@dataclass
class Measurement:
    """Everything measured on one retargeted motion. `summary()` is the row form."""

    motion: str
    n_frames: int
    n_boxes: int
    tol: Tolerances
    fps: float = 100.0
    stances: list[Stance] = field(default_factory=list)
    swings: list[Swing] = field(default_factory=list)
    seats: list[Seat] = field(default_factory=list)
    face_pen: dict[str, float] = field(default_factory=dict)  # deepest, by box face
    face_worst: dict[str, str] = field(default_factory=dict)  # where, as "geom @ fN"
    selfpen_worst: float = 0.0
    selfpen_where: str = ""
    #: Foot phases not scored because the body was on a seat through them; see
    #: :func:`_not_while_seated`. Always 0 on a motion with no seat.
    n_seated_phases_dropped: int = 0
    selfpen_frac: float = 0.0
    #: Per-frame arrays, for the renderer: `dist`/`vert` per side, `selfpen`, and the
    #: boolean `swing_bad` marking the frames of swings that failed their clearance test.
    per_frame: dict = field(default_factory=dict)

    def fails(self) -> list[str]:
        """Every threshold violation, as a line of text. Empty means PASS."""
        out = []
        for s in self.stances:
            for f in s.flags(self.tol):
                if f == "FLOATING":
                    out.append(f"stance f{s.start}-{s.end} {s.side}: floating {s.median_clear * 1000:.0f} mm")
                elif f == "PENETRATING":
                    out.append(f"stance f{s.start}-{s.end} {s.side}: penetrating {-s.min_clear * 1000:.0f} mm")
                elif f == "FOREFOOT_UP":
                    out.append(
                        f"stance f{s.start}-{s.end} {s.side}: forefoot "
                        f"{s.fore_clear * 1000:.0f} mm up where the source's is "
                        f"{s.src_fore_clear * 1000:.0f}"
                    )
                elif f == "HINDFOOT_UP":
                    out.append(
                        f"stance f{s.start}-{s.end} {s.side}: hindfoot "
                        f"{s.hind_clear * 1000:.0f} mm up where the source's is "
                        f"{s.src_hind_clear * 1000:.0f}"
                    )
                else:
                    out.append(
                        f"stance f{s.start}-{s.end} {s.side}: travelled {s.excess * 1000:.0f} mm more than the source"
                    )
        for w in self.swings:
            if w.dragging(self.tol):
                out.append(
                    f"swing f{w.start}-{w.end} {w.side}: cleared {w.peak * 1000:.0f} mm, "
                    f"needed {w.required(self.tol) * 1000:.0f} "
                    f"(source {w.src_peak * 1000:.0f})"
                )
            if w.scraping(self.tol):
                out.append(f"swing f{w.start}-{w.end} {w.side}: sole entered the terrain by {-w.touched * 1000:.0f} mm")
        for t in self.seats:
            for f in t.flags(self.tol):
                if f == "SEAT_FLOATING":
                    out.append(f"seated f{t.start}-{t.end}: glutes {t.median_clear * 1000:.0f} mm above the seat")
                else:
                    out.append(f"seated f{t.start}-{t.end}: glutes {-t.min_clear * 1000:.0f} mm into the seat")
        for face, depth in self.face_pen.items():
            if depth > self.tol.pen_tol:
                out.append(f"body penetrates a box {face} face by {depth * 1000:.0f} mm")
        if self.selfpen_worst > self.tol.selfpen_tol:
            out.append(f"left/right limbs interpenetrate by {self.selfpen_worst * 1000:.0f} mm")
        return out

    def summary(self) -> dict:
        """One flat row of counts and extremes, suitable for a CSV.

        Counts are per phase, not per motion: a motion holds 4-40 stances and swings, and
        the rate over them is what the class table aggregates.

        Returns:
            A dict of the phase-quality columns, in millimetres
            except `clear_ratio_*`, which is dimensionless, and the counts.
        """
        tol = self.tol
        floating = [s for s in self.stances if s.median_clear > tol.float_tol]
        penetrating = [s for s in self.stances if s.min_clear < -tol.pen_tol]
        slipping = [s for s in self.stances if s.excess > tol.slip_tol]
        forefoot_up = [s for s in self.stances if s.forefoot_up(tol)]
        hindfoot_up = [s for s in self.stances if s.hindfoot_up(tol)]
        seat_bad = [t for t in self.seats if t.flags(tol)]
        dragging = [w for w in self.swings if w.dragging(tol)]
        scraping = [w for w in self.swings if w.scraping(tol)]
        med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")  # noqa: E731
        return {
            "motion": self.motion,
            "n_frames": self.n_frames,
            "n_boxes": self.n_boxes,
            "n_stances": len(self.stances),
            "n_swings": len(self.swings),
            "n_floating": len(floating),
            "n_penetrating": len(penetrating),
            "n_slipping": len(slipping),
            "n_forefoot_up": len(forefoot_up),
            "n_hindfoot_up": len(hindfoot_up),
            "n_dragging": len(dragging),
            "n_scraping": len(scraping),
            "n_seats": len(self.seats),
            "n_seat_bad": len(seat_bad),
            "n_seated_phases_dropped": self.n_seated_phases_dropped,
            # Continuous quantities: the median says what the motion is usually like, the
            # worst says whether it is reproducible at all.
            "stance_clear_median_mm": med([s.median_clear for s in self.stances]) * 1000,
            "stance_clear_worst_mm": max([s.median_clear for s in self.stances], default=0) * 1000,
            "stance_pen_worst_mm": -min([s.min_clear for s in self.stances], default=0) * 1000,
            "slip_excess_median_mm": med([s.excess for s in self.stances]) * 1000,
            "slip_excess_worst_mm": max([s.excess for s in self.stances], default=0) * 1000,
            # Clearance as a ratio to the source's own is the comparable number across a
            # 90 mm stair riser and a 20 mm beam shuffle.
            "clear_ratio_median": med([w.peak / w.src_peak for w in self.swings if w.src_peak > 1e-3]),
            "clear_ratio_worst": min(
                [w.peak / w.src_peak for w in self.swings if w.src_peak > 1e-3], default=float("nan")
            ),
            "swing_peak_median_mm": med([w.peak for w in self.swings]) * 1000,
            "src_peak_median_mm": med([w.src_peak for w in self.swings]) * 1000,
            "seat_clear_median_mm": med([t.median_clear for t in self.seats]) * 1000,
            "seat_clear_worst_mm": max([t.median_clear for t in self.seats], default=0) * 1000,
            "seat_pen_worst_mm": -min([t.min_clear for t in self.seats], default=0) * 1000,
            "body_pen_top_mm": self.face_pen.get("top", 0.0) * 1000,
            "body_pen_side_mm": self.face_pen.get("side", 0.0) * 1000,
            "selfpen_worst_mm": self.selfpen_worst * 1000,
            "selfpen_frames_pct": self.selfpen_frac * 100,
            "n_fails": len(self.fails()),
            "passed": int(not self.fails()),
        }


#: Grid resolution used to sample the terrain under a foot. 5x5 over the sole's own
#: footprint is enough to catch a tread edge crossing it; the quantity is a max, so more
#: samples only move it earlier by a millimetre or two.
FOOTPRINT_SAMPLES = 5


def _surface_under(terrain, points: np.ndarray, radius: float) -> float:
    """Highest terrain height anywhere under a foot and `radius` around it, in metres.

    A lookahead query - what is the foot over or about to meet - not the support datum.
    For "how far is this landmark above the surface it is standing on", use
    :func:`_support_plane_under`: a single height for a whole foot is right only while
    every surface is level, and reads the uphill edge of the footprint on a ramp.

    Args:
        terrain: The terrain to query.
        points: (n, 3) world positions of the geoms making up the sole.
        radius: How far each geom extends horizontally beyond its centre.

    Returns:
        The maximum of `terrain.height_at` over the sole's footprint. 0.0 on flat ground.
    """
    # Walkable, not everything solid: a chair beside the foot is not the surface it is
    # standing on. Inert wherever the terrain has no seat. See `TerrainSpec.walkable`.
    terrain = terrain.walkable
    xy = np.atleast_2d(np.asarray(points))[:, :2]
    gx = np.linspace(xy[:, 0].min() - radius, xy[:, 0].max() + radius, FOOTPRINT_SAMPLES)
    gy = np.linspace(xy[:, 1].min() - radius, xy[:, 1].max() + radius, FOOTPRINT_SAMPLES)
    grid_x, grid_y = np.meshgrid(gx, gy)
    return float(np.max(terrain.height_at(grid_x.ravel(), grid_y.ravel())))


def _support_plane_under(terrain, points: np.ndarray, footprint: np.ndarray | None = None) -> np.ndarray:
    """Height of the surface the foot is standing on, under each of `points`.

    The support datum for everything measured against the terrain in this module. One
    surface is chosen for the whole foot - the highest any part of the sole is over - and
    then evaluated at each point, so two landmarks of one foot are always read off the
    same surface but at their own positions. See
    :meth:`~loco_mujoco.core.terrain.TerrainSpec.support_plane_at`.

    Args:
        terrain: The terrain to query.
        points: (n, 3) world positions to evaluate the surface at.
        footprint: (m, 3) the whole sole, if `points` is only part of it. Defaults to
            `points`, which is right when they are the sole geoms themselves; pass the sole
            when asking about the forefoot alone, or a forefoot hanging over an edge picks
            a different surface from the foot it belongs to.

    Returns:
        (n,) heights. All zero on flat ground.
    """
    terrain = terrain.walkable  # the surface it stands on, never a seat beside it
    points = np.atleast_2d(np.asarray(points, dtype=float))
    foot_xy = np.atleast_2d(np.asarray(points if footprint is None else footprint))[:, :2]
    return np.array([terrain.support_plane_at(foot_xy, p[0], p[1]) for p in points])


def _sole_support_distance(
    model,
    data,
    foot_geoms: list[int],
    terrain,
    support_geoms: dict,
    floor_id: int,
) -> float:
    """Exact signed distance from a sole to its selected walkable support geom.

    Selecting the support and measuring its distance are deliberately separate steps. A
    minimum over every scene geom lets a nearby riser side masquerade as support for a foot
    hovering over the floor. Measuring world-vertical sole height against a ramp plane is
    also wrong: the lowest point of an inclined sole and the plane height at its geom centre
    are at different horizontal positions, producing a false penetration proportional to
    foot length and slope. ``mj_geomDistance`` against the selected solid has neither
    ambiguity and returns the signed surface-normal gap directly.

    Args:
        model: MuJoCo model containing the foot and terrain geoms.
        data: Forwarded MuJoCo data for the current pose.
        foot_geoms: Collision geoms making up one sole.
        terrain: Walkable :class:`TerrainSpec` used to choose the supporting box.
        support_geoms: Mapping from each walkable ``BoxSpec`` to its MuJoCo geom id.
        floor_id: MuJoCo floor geom id, used when no box lies under the sole footprint.

    Returns:
        Signed Euclidean distance in metres. Positive is a gap, negative is penetration.
    """
    footprint = np.asarray(data.geom_xpos[foot_geoms], dtype=float)
    box = terrain.supporting_box(footprint[:, :2])
    if box is None:
        selected_geom = floor_id
    else:
        try:
            selected_geom = support_geoms[box]
        except KeyError as exc:
            raise ValueError(f"selected support geom {box.name!r} has no MuJoCo mapping") from exc
    return float(min(mujoco.mj_geomDistance(model, data, geom, selected_geom, 2.0, None) for geom in foot_geoms))


def surface_over_run(terrain, xy: np.ndarray) -> float:
    """Terrain height under a stance, queried once at the run's median position.

    Not the median of the height under each frame. `height_at` returns a box top, and the
    median of an even number of them averages the two middle values: a stance that settles
    across a riser gets back a height halfway up it, which is a surface no box has and which
    then scores as half a riser against both treads. Same defect, same fix, as the support
    term of `validate_terrain`.

    Args:
        terrain: The terrain to query.
        xy: (n, 2) positions over the run.

    Returns:
        The box top under the run's centre, or 0.0 for the floor.
    """
    centre = np.median(np.atleast_2d(np.asarray(xy))[:, :2], axis=0)
    return float(terrain.walkable.height_at(centre[0], centre[1]))


def _on_timebase(
    joints: np.ndarray,
    fps: float,
    n: int,
    out_fps: float,
    target_times_s: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Resample source joints onto the retargeted trajectory's frame times.

    Linear in time, which is what `env.load_trajectory` does to the qpos on the way in, so
    the two halves of every frame-by-frame comparison in this module are on one clock.

    Args:
        joints: (T, J, 3) source joints at `fps`.
        fps: Source frame rate.
        n: Number of frames in the retargeted trajectory.
        out_fps: Its frame rate.

    Returns:
        ``((n, J, 3), out_fps)``. Returned unchanged, only truncated, when the rates already
        match - so nothing scored at 100 Hz moves by a floating-point epsilon.
    """
    if target_times_s is None and abs(fps - out_fps) < 1e-9:
        return joints[:n], fps
    t_src = np.arange(len(joints)) / fps
    t_out = np.arange(n, dtype=float) / out_fps if target_times_s is None else np.asarray(target_times_s, dtype=float)
    if t_out.shape != (n,) or not np.all(np.isfinite(t_out)):
        raise ValueError("target_times_s must contain one finite timestamp per output frame")
    if np.any(np.diff(t_out) <= 0):
        raise ValueError("target_times_s must be strictly increasing")
    out = np.empty((n, joints.shape[1], 3))
    for j in range(joints.shape[1]):
        for k in range(3):
            out[:, j, k] = np.interp(t_out, t_src, joints[:, j, k])
    return out, out_fps


def _declared_output_times(
    traj_path: str,
    motion: str,
    n: int,
    out_fps: float,
    source_root: Path | None = None,
) -> np.ndarray:
    """Read or reconstruct the trajectory's source-clock timestamps.

    TERRA removes one source frame at each end for central differences.  Consequently the
    first stored pose normally represents source time ``1 / native_fps``, not time zero.
    Contact annotations are expressed on the source clock, so silently rebuilding a local
    zero-based clock shifts every touchdown and foot-off relative to the robot trajectory.
    """
    from terra.evaluation.timeline import load_trajectory_timeline

    trajectory_path = Path(traj_path)
    analysis_path = trajectory_path.with_name(trajectory_path.stem + "_analysis.npz")
    timeline = load_trajectory_timeline(
        trajectory_path,
        analysis_path,
        motion,
        source_root=source_root,
    )
    times = timeline.output_times()
    if len(times) != n or not np.isclose(timeline.output_fps, out_fps, rtol=0.0, atol=1e-6):
        raise ValueError(f"resolved timeline disagrees with trajectory: {trajectory_path}")
    return times


def _not_while_seated(phases_, rests) -> tuple[list, int]:
    """Drop the stance/swing phases that happen while the pelvis is on a seat.

    **The feet are not what is carrying the body during a sit**, so neither question this
    module asks of a foot is meaningful there: a foot parked in the air is not a stance that
    is floating, and the long gap between two such parked phases is not a swing that failed
    to clear anything. Both are scored against a surface that is not bearing any load.

    Measured over the 40-motion chair subset, **47% of all flagged stances overlapped a
    seated phase** - `stance floating` on 21 of 40 motions and `forefoot up` on 29 - none of
    which is a retargeting defect.

    This is the same rule `drop_seated_contacts` applies in the fitter and in the fit gate,
    in its third and last place. It has to be all three: the whole point of the rule is that
    support evidence and support scoring share one convention, and a scorer that keeps what
    the fitter dropped is measuring the result against terrain built from different
    assumptions.

    Body-vs-terrain penetration and the seat term are deliberately *not* filtered - they run
    over every frame - so nothing about a seated phase goes unmeasured, it is only measured
    by the instrument that applies to it.

    Args:
        phases_: ``[(start, end), ...]`` for one side.
        rests: `SeatRest`s detected on the source.

    Returns:
        ``(kept, n_dropped)``.
    """
    if not rests:
        return list(phases_), 0
    kept = [(a, b) for a, b in phases_ if not any(min(b, r.end) > max(a, r.start) for r in rests)]
    return kept, len(phases_) - len(kept)


def _foot_geoms(model) -> dict[str, list[int]]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    out = {side: [g for g, n in enumerate(names) if n.startswith(pref)] for side, pref in FOOT_GEOMS.items()}
    for side, geoms in out.items():
        if not geoms:
            raise SystemExit(f"No {side} foot geoms matched {FOOT_GEOMS[side]}")
    return out


def _forefoot_geoms(model) -> dict[str, list[int]]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    out = {side: [g for g, n in enumerate(names) if n.startswith(pref)] for side, pref in FOREFOOT_GEOMS.items()}
    for side, geoms in out.items():
        if not geoms:
            raise SystemExit(f"No {side} forefoot geoms matched {FOREFOOT_GEOMS[side]}")
    return out


def _leg_pair_ids(model) -> list[tuple[int, int, str]]:
    """Geom-id pairs for the calibrated left/right distal segments, with a label."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    pairs = []
    for left_pref, right_pref in LIMB_PREFIXES:
        ls = [g for g, n in enumerate(names) if n.startswith(left_pref)]
        rs = [g for g, n in enumerate(names) if n.startswith(right_pref)]
        pairs += [(a, b, f"{names[a]}/{names[b]}") for a in ls for b in rs]
    return pairs


def _max_pair_penetration(model, data, pairs: list[tuple[int, int, str]]) -> float:
    """Deepest exact signed-distance penetration among named geometry pairs."""
    if not pairs:
        return 0.0
    return max(
        0.0,
        max(
            -float(mujoco.mj_geomDistance(model, data, first, second, 0.05, None))
            for first, second, _label in pairs
        ),
    )


def _minimum_geom_distance_at_threshold(
    model,
    data,
    first_geoms: tuple[int, ...],
    second_geoms: list[int] | tuple[int, ...],
    threshold_m: float,
) -> float:
    """Return a distance that remains strictly above an exceeded threshold.

    ``mj_geomDistance`` returns ``distmax`` when the true separation is at least
    that value.  Querying with ``distmax == threshold`` and then testing ``<=``
    therefore turns every distant pair into a false contact.  The next representable
    value preserves the metric boundary without doing unnecessary long-range work.
    """

    distmax = float(np.nextafter(threshold_m, np.inf))
    return min(
        float(mujoco.mj_geomDistance(model, data, first, second, distmax, None))
        for first in first_geoms
        for second in second_geoms
    )


def phases(events, side: str, n_frames: int, fps: float | None = None, min_stance_s: float = 0.0):
    """Stance and swing windows for one side, from source stance events.

    Swings are the gaps between consecutive stances. The leading and trailing gaps are
    dropped: outside the first and last footfall there is no evidence the foot was meant to
    be anywhere in particular.

    Args:
        events: Source stance events from `detect_stance_events`.
        side: ``"left"`` or ``"right"``.
        n_frames: Length of the retargeted trajectory; windows are clipped to it.
        fps: Source frame rate. Required to apply `min_stance_s`.
        min_stance_s: Drop stances shorter than this, seconds. `detect_stance_events` marks a
            foot probe that is slow and low, which on a jump or a landing fires for a few frames at
            a time; scoring those as footfalls is what let two `obstacle` clips dominate
            every worst-list. Dropped stances are *not* bridged - the swings around them are
            still cut at their edges - because the foot's whereabouts during those frames is
            genuinely unknown, and merging them would invent one long swing across a landing.
    """
    active = np.zeros(n_frames, dtype=bool)
    for event in events:
        if event.joint in SOLE_JOINTS[side]:
            active[max(0, event.start) : min(n_frames, event.end)] = True
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    stance = [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]
    swing = [(stance[index][1], stance[index + 1][0]) for index in range(len(stance) - 1)]
    if fps and min_stance_s > 0:
        stance = [(a, b) for a, b in stance if (b - a) / fps >= min_stance_s]
    return stance, [(a, b) for a, b in swing if b - a >= 3]


def _slip_window_end(start: int, end: int, fps: float, liftoff_s: float) -> int:
    """Exclusive end of the loaded-stance window used to judge foot slip."""
    tail = max(round(liftoff_s * fps), 0)
    return max(start + 1, end - tail)


def traj_paths(
    motion: str,
    model: str = "MyoFullBody",
    method: str = "terra",
    *,
    cache_root: Path | None = None,
) -> tuple[str, str]:
    """`(trajectory, terrain)` paths in the retargeting cache. The terrain may not exist."""
    if cache_root is None:
        cache_root = StorageRoots.from_environment(Path.cwd()).direct_cache_root
    traj = os.path.join(cache_root, model, method, f"{motion}.npz")
    return traj, traj.replace(".npz", "_terrain.json")


def measure(
    motion: str,
    model: str = "MyoFullBody",
    method: str = "terra",
    tol: Tolerances | None = None,
    keep_per_frame: bool = True,
    terrain_method: str | None = None,
    force_flat: bool = False,
    cache_root: Path | None = None,
    source_root: Path | None = None,
    terrain_contact_threshold_m: float = 0.1,
) -> Measurement:
    """Measure one retargeted motion against the terrain it was retargeted onto.

    Args:
        motion: AMASS motion name.
        model: Environment name; also selects the cache directory.
        method: Cache subdirectory holding the retargeted motion.
        terrain_method: Cache subdirectory holding the terrain metadata. Defaults to
            ``method``. Set this to the terrain-aware method
            when evaluating a flat-ground baseline against the same reconstructed scene.
        force_flat: Ignore any terrain metadata and evaluate against a plane. This is for a
            manifest explicitly designated as flat; a stale or exploratory fitted terrain file
            must not silently turn a flat-ground table row into a terrain measurement.
        tol: Thresholds; the defaults are the documented ones.
        keep_per_frame: Keep the per-frame arrays (the renderer needs them; a batch scorer
            does not, and they are the bulk of the memory).
        cache_root: Explicit root containing ``model/method`` trajectory artifacts. When
            omitted, the configured retargeting cache is used.

    Returns:
        A `Measurement`. `summary()` flattens it to a CSV row, `fails()` to the reasons.

    Raises:
        FileNotFoundError: If the motion has not been retargeted.
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import TerrainSpec
    from loco_mujoco.trajectory import Trajectory
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody
    from terra._musclemimic import scene_geom_ids
    from terra.collisions import self_collision_geom_pairs
    from terra.constants import MYOFULLBODY_SELF_COLLISION_PAIRS, SMPLH_TO_MYOFULLBODY
    from terra.contacts import robot_sole_offsets
    from terra.source import motion_world_joints
    from terra.terrain import (
        detect_seat_rests,
        detect_stance_events,
        joint_surface_offsets,
    )
    from terra.terrain.metadata import TerrainMetadata

    tol = tol or Tolerances()
    if not np.isfinite(terrain_contact_threshold_m) or terrain_contact_threshold_m <= 0:
        raise ValueError("terrain_contact_threshold_m must be positive and finite")
    traj_path, terrain_path = traj_paths(motion, model, method, cache_root=cache_root)
    if terrain_method is not None:
        terrain_path = traj_paths(motion, model, terrain_method, cache_root=cache_root)[1]
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"No retargeted motion at {traj_path}. Retarget it first.")

    metadata = (
        TerrainMetadata.from_terrain(TerrainSpec())
        if force_flat or not os.path.exists(terrain_path)
        else TerrainMetadata.load(terrain_path)
    )
    terrain = metadata.terrain
    source_terrain = metadata.terrain
    env_kwargs = {} if terrain.is_flat else {"terrain_type": "BoxTerrain", "terrain_params": terrain.to_env_params()}
    env = MyoFullBody(**env_kwargs, th_params={"random_start": False, "fixed_start_conf": (0, 0)})
    m = env._model
    data = mujoco.MjData(m)

    env_geoms = scene_geom_ids(m)
    feet = _foot_geoms(m)
    fore_geoms = _forefoot_geoms(m)
    leg_pairs = _leg_pair_ids(m)
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    slip_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for s, b in SLIP_BODY.items()}
    # The robot's own body-origin-above-sole offsets and the bodies they belong to, so the
    # swing comparison can be built the same way on both sides. See `Swing`.
    probe_off = robot_sole_offsets(m, SMPLH_TO_MYOFULLBODY)
    probe_body = {
        side: [(j, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)) for j, b in pairs]
        for side, pairs in PROBE_BODIES.items()
    }
    box_geoms = {g for g in env_geoms if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") != "floor"}
    target_ground = terrain.walkable
    support_geom = {}
    for index, box in enumerate(terrain.boxes):
        if box.name.startswith(SEAT_GEOM_PREFIX):
            continue
        name = box.name or f"terrain_box_{index}"
        geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"terrain support geom {name!r} is absent from the model")
        support_geom[box] = geom_id
    # The seat, and the glutes that have to rest on it. Both empty on every terrain without
    # a chair in it, which is what makes every seat quantity below inert there.
    seat_boxes = [
        g for g in box_geoms if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(SEAT_GEOM_PREFIX)
    ]
    glutes = [g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in SEAT_GEOMS) if g >= 0]
    terrain_contact_body_ids = tuple(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in TERRAIN_CONTACT_BODIES
    )
    if any(body_id < 0 for body_id in terrain_contact_body_ids):
        raise ValueError("terrain-contact metric body is absent from the model")
    terrain_contact_geoms = tuple(
        tuple(int(geom_id) for geom_id in range(m.ngeom) if int(m.geom_bodyid[geom_id]) == body_id)
        for body_id in terrain_contact_body_ids
    )
    if any(not geoms for geoms in terrain_contact_geoms):
        raise ValueError("terrain-contact metric body has no geometry")

    traj = Trajectory.load(traj_path)
    qpos = np.asarray(traj.data.qpos)
    n = len(qpos)

    # Compare source and robot on a common timebase. Retargeted trajectories use the
    # environment rate, which can differ from the source capture rate.
    from terra.evaluation.timeline import load_source_motion

    analysis_path = Path(traj_path).with_name(Path(traj_path).stem + "_analysis.npz")
    source_motion = load_source_motion(motion, analysis_path, source_root=source_root)
    joints, fps = motion_world_joints(
        motion,
        model,
        motion_data=source_motion,
        fitted_shape_path=source_shape_path(model, cache_root),
    )
    output_times = _declared_output_times(
        traj_path,
        motion,
        n,
        float(traj.info.frequency),
        source_root=source_root,
    )
    joints, fps = _on_timebase(joints, fps, n, float(traj.info.frequency), target_times_s=output_times)

    names = list(SMPLH_DEMO_JOINTS)
    source_contact_points = joints[:, [names.index(name) for name in TERRAIN_CONTACT_SOURCE_JOINTS]]
    desired_terrain_contact = (
        signed_distance_to_terrain_boxes(source_contact_points, source_terrain) <= terrain_contact_threshold_m
    )
    events = detect_stance_events(
        joints,
        names,
        fps,
        contact_joints=DEFAULT_CONTACT_JOINTS,
    )
    # Each source joint's centre sits a fixed distance inside the foot; measured on its own
    # lowest contacts it becomes the offset that turns a joint height into a sole height.
    stored_offsets = source_terrain.provenance.get("joint_surface_offsets_m")
    offsets = (
        {str(name): float(value) for name, value in stored_offsets.items()}
        if isinstance(stored_offsets, dict)
        else joint_surface_offsets(events)
    )

    # --- per-frame foot state ------------------------------------------------------------
    dist = {s: np.full(n, np.inf) for s in feet}  # sole -> environment, any face
    vert = {s: np.full(n, np.inf) for s in feet}  # sole above the surface beneath it
    fore = {s: np.full(n, np.inf) for s in feet}  # forefoot only, over the same surface
    hind = {s: np.full(n, np.inf) for s in feet}  # hindfoot only, over the same surface
    probe = {s: np.full(n, np.inf) for s in feet}  # the same, built the source's way
    support = {s: np.full(n, np.inf) for s in feet}  # exact distance to chosen support geom
    pos = {s: np.zeros((n, 3)) for s in feet}  # fixed foot point, for slip
    selfpen = np.zeros(n)
    # Deepest robot/environment and robot/robot contact on every frame.  The old terrain
    # quality report kept only box-face maxima and a calibrated distal-leg pair. The benchmark
    # table evaluates the complete contralateral-leg scope used by TERRA's self-collision
    # term.  Keep the all-body series as a diagnostic, but do not use it as a benchmark
    # failure mask: MyoFullBody's own qpos0 has both humerus capsules 21.5 mm inside the
    # thorax capsule, and sit/stand motions intentionally put the forearms on the thighs.
    # Counting either makes closer pose tracking look like worse self-collision.
    body_pen = np.zeros(n)
    interleg_selfpen = np.zeros(n)
    all_selfpen = np.zeros(n)
    terrain_contact_distance = np.full((n, len(terrain_contact_body_ids)), np.inf)
    interleg_geom_pairs = self_collision_geom_pairs(m, MYOFULLBODY_SELF_COLLISION_PAIRS)
    # Glutes to seat, exact. `inf` where the terrain has no seat, so nothing downstream can
    # mistake "not measured" for "resting on it".
    seat_gap = np.full(n, np.inf)
    face_pen = {"top": 0.0, "side": 0.0}
    face_worst: dict[str, str] = {"top": "", "side": ""}
    worst_pair, worst_depth = "", 0.0
    original_geom_margins = m.geom_margin.copy()
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(m, data)
        for side, geoms in feet.items():
            dist[side][i] = min(mujoco.mj_geomDistance(m, data, g, e, 0.5, None) for g in geoms for e in env_geoms)
            # Benchmark actual-contact state: exact signed distance to the specifically
            # selected walkable support. This is neither a method-specific contact decision
            # nor a nearest-side query; the scene geometry under the sole chooses the object.
            support[side][i] = _sole_support_distance(
                m,
                data,
                geoms,
                target_ground,
                support_geom,
                floor_id,
            )
            # Distance to the floor *plane* is the geom's lowest point measured vertically,
            # whatever its shape - so height above the local surface needs no per-geom-type
            # bounding maths, only the terrain height under the foot.
            #
            # **One surface per foot, evaluated under each point of it.** The surface is the
            # one the whole foot is standing on, chosen once from the highest terrain any
            # part of the sole is over, and then read at each geom's own position. Choosing
            # it per geom instead puts the two ends of one foot on different treads, and the
            # minimum over them flips by a whole riser as the foot crosses a nose: 11-143
            # frames per stair clip. Taking a single *height* for the whole foot instead -
            # the highest anywhere beneath it, which is what this did - is right only while
            # every surface is level. On a ramp it reads the uphill edge of the footprint,
            # so a foot lying perfectly flat on a 27 deg slope scores 66 mm of penetration
            # on every stance it has. `support_plane_at` is the rule that covers both, and
            # is the same one `validate_terrain` scores the source with.
            here = _support_plane_under(terrain, data.geom_xpos[geoms])
            vert[side][i] = min(
                mujoco.mj_geomDistance(m, data, g, floor_id, 2.0, None) - z for g, z in zip(geoms, here, strict=True)
            )
            # Both read off that same one surface, so the difference between them is a foot
            # pitch and never a tread step.
            fore_z = _support_plane_under(terrain, data.geom_xpos[fore_geoms[side]], footprint=data.geom_xpos[geoms])
            fore[side][i] = min(
                mujoco.mj_geomDistance(m, data, g, floor_id, 2.0, None) - z
                for g, z in zip(fore_geoms[side], fore_z, strict=True)
            )
            probe_xyz = np.array([data.xpos[b] for _, b in probe_body[side]])
            probe_z = _support_plane_under(terrain, probe_xyz, footprint=data.geom_xpos[geoms])
            probe_clear = np.asarray(
                [
                    data.xpos[b][2] - probe_off.get(j, 0.0) - z
                    for (j, b), z in zip(probe_body[side], probe_z, strict=True)
                ]
            )
            probe[side][i] = float(np.min(probe_clear))
            # The ankle-side body probe uses the same landmark and calibrated sole-offset
            # convention as ``src_hind`` below.  A rear collision capsule can still touch
            # while its anatomical ankle target visibly hovers, so the geom minimum is not
            # an adequate diagnostic for the reported tip-toe fit.
            hind_index = [j for j, _body in probe_body[side]].index(SLIP_JOINT[side])
            hind[side][i] = float(probe_clear[hind_index])
            pos[side][i] = data.xpos[slip_body[side]]

        if seat_boxes and glutes:
            seat_gap[i] = min(mujoco.mj_geomDistance(m, data, g, e, 1.0, None) for g in glutes for e in seat_boxes)

        depth = 0.0
        for a, b, label in leg_pairs:
            d = mujoco.mj_geomDistance(m, data, a, b, 0.05, None)
            if d < -depth:
                depth = -d
                if depth > worst_depth:
                    worst_depth, worst_pair = depth, f"{label} @ f{i}"
        selfpen[i] = depth

        # Do not infer geometric validity from ``data.contact``. MuJoCo's generated
        # contacts are solver-facing and may omit or under-report an already intersecting
        # pair. The benchmark definition names the exact contralateral-leg geometry scope,
        # so query every one of those pairs directly.
        interleg_selfpen[i] = _max_pair_penetration(m, data, interleg_geom_pairs)

        # OmniRetarget measures mapped robot collision bodies, not body origins.
        # Query only source-desired contacts; the other entries remain irrelevant
        # infinities and do not add an all-box scan to every body on every frame.
        for point in np.flatnonzero(desired_terrain_contact[i]):
            terrain_contact_distance[i, point] = _minimum_geom_distance_at_threshold(
                m,
                data,
                terrain_contact_geoms[point],
                box_geoms,
                terrain_contact_threshold_m,
            )

        # Match OmniRetarget's metric broad phase: temporarily expand margins to
        # 10 cm, collect nearby pairs, then use each contact's exact signed
        # distance. The margins are restored before any subsequent frame.
        try:
            m.geom_margin[:] = 0.1
            mujoco.mj_collision(m, data)
        finally:
            m.geom_margin[:] = original_geom_margins

        for c in range(data.ncon):
            con = data.contact[c]
            first_is_env = con.geom1 in env_geoms
            second_is_env = con.geom2 in env_geoms
            if first_is_env != second_is_env:
                # OmniRetarget uses the expanded contact list only as a broad phase,
                # then evaluates each candidate with the exact signed-distance query.
                exact_distance = float(
                    mujoco.mj_geomDistance(m, data, int(con.geom1), int(con.geom2), 0.1, None)
                )
                if exact_distance < 0:
                    body_pen[i] = max(body_pen[i], -exact_distance)
            elif not first_is_env and not second_is_env and con.dist < 0:
                all_selfpen[i] = max(all_selfpen[i], float(-con.dist))
            if (con.geom1 in box_geoms) == (con.geom2 in box_geoms):
                continue  # not a body-against-terrain contact
            if con.dist >= 0:
                continue
            # contact.frame's first row is the normal; near-vertical means a top face.
            face = "top" if abs(con.frame[2]) >= 0.7 else "side"
            if -con.dist > face_pen[face]:
                face_pen[face] = float(-con.dist)
                other = con.geom2 if con.geom1 in box_geoms else con.geom1
                face_worst[face] = f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)} @ f{i}"

    # --- source sole clearance reference ---------------------------------------------------
    # The source's own sole, as the lowest of its contact joints minus that joint's offset,
    # over the same datum the result is measured against: the highest terrain under the whole
    # foot. Sampling it at the joint centres instead makes the reference jump a riser at a
    # tread edge exactly where the result does not, and the ratio between them is what
    # decides dragging.
    #
    # Walkable, not the whole terrain - the same reason every other foot-facing query here
    # is (`_support_plane_under`, `_surface_under`, `surface_over_run`). Missed here, a
    # source foot that is genuinely on the floor while it passes under or beside a chair
    # reads a clearance of minus the seat height, which the dragging ratio then compares
    # the robot's real clearance against.
    ground = source_terrain.walkable
    src_clear, src_fore, src_hind = {}, {}, {}
    for side, jnames in SOLE_JOINTS.items():
        cols, foot_xy = {}, []
        for jn in jnames:
            j = joints[:, names.index(jn)]
            foot_xy.append(j[:, :2])
            cols[jn] = j[:, 2] - offsets.get(jn, 0.0)
        stacked = np.stack(foot_xy, axis=1)  # (T, n_joints, 2)
        # Read the same way as the robot's: one surface per foot, evaluated under each
        # landmark. The two are compared as a ratio, so a slope biasing only one of them
        # would show up as dragging that is not there.
        clear = {}
        for k, jn in enumerate(jnames):
            surf = np.array(
                [ground.support_plane_at(stacked[t], stacked[t, k, 0], stacked[t, k, 1]) for t in range(len(joints))]
            )
            clear[jn] = cols[jn] - surf
        src_clear[side] = np.min(np.stack(list(clear.values())), axis=0)
        # The source's forefoot, taken at its toe joint alone - the only source probe over
        # the ball of the foot. Compared against the robot's forefoot geoms, this says
        # whether a stance the robot spends on its heel is one the human spent flat.
        src_fore[side] = clear[TOE_OF[side]]
        src_hind[side] = clear[SLIP_JOINT[side]]

    out = Measurement(
        motion=motion,
        n_frames=n,
        n_boxes=len(terrain),
        tol=tol,
        fps=float(traj.info.frequency),
        face_pen=face_pen,
        face_worst=face_worst,
        selfpen_worst=float(worst_depth),
        selfpen_where=worst_pair,
        selfpen_frac=float(np.mean(selfpen > tol.selfpen_tol)),
    )

    # --- seated phases -------------------------------------------------------------------
    # Timed from the *source*, exactly as stances are, and for the same reason: the
    # retargeted pelvis is the thing being judged, so its own rest detection cannot be the
    # reference. An empty list wherever the source never sat, which is every locomotion clip.
    seat_bad = np.zeros(n, dtype=bool)
    rests = detect_seat_rests(joints, names, fps)
    for rest in rests:
        a, b = rest.start, min(rest.end, n)
        if b - a < 2 or not np.isfinite(seat_gap[a:b]).all():
            continue
        t = Seat(
            start=a,
            end=b,
            surface_z=float(terrain.height_at(*np.median(rest.xy, axis=0))),
            median_clear=float(np.median(seat_gap[a:b])),
            min_clear=float(seat_gap[a:b].min()),
            src_pelvis_z=rest.z,
        )
        out.seats.append(t)
        if t.flags(tol):
            seat_bad[a:b] = True

    swing_bad = np.zeros(n, dtype=bool)
    stance_mask = {side: np.zeros(n, dtype=bool) for side in ("left", "right")}
    n_seated_phases = 0
    for side in ("left", "right"):
        stance, swing = phases(events, side, n, fps=fps, min_stance_s=tol.min_stance_s)
        stance, dropped_st = _not_while_seated(stance, rests)
        swing, dropped_sw = _not_while_seated(swing, rests)
        n_seated_phases += dropped_st + dropped_sw
        for a, b in stance:
            stance_mask[side][a:b] = True
            seg = dist[side][a:b]
            slip_end = _slip_window_end(a, b, fps, tol.slip_liftoff_s)
            xy = pos[side][a:slip_end, :2]
            toe = joints[a:b, names.index(TOE_OF[side])]
            src = joints[a:slip_end, names.index(SLIP_JOINT[side])]
            out.stances.append(
                Stance(
                    side=side,
                    start=a,
                    end=b,
                    surface_z=surface_over_run(terrain, toe[:, :2]),
                    median_clear=float(np.median(seg)),
                    min_clear=float(seg.min()),
                    travel=float(np.linalg.norm(xy - xy[0], axis=1).max()),
                    # Same landmark on both sides; see `SLIP_BODY`.
                    src_travel=float(np.linalg.norm(src[:, :2] - src[0, :2], axis=1).max()),
                    slip_end=slip_end,
                    fore_clear=float(np.median(fore[side][a:b])),
                    src_fore_clear=float(np.median(src_fore[side][a:b])),
                    hind_clear=float(np.median(hind[side][a:b])),
                    src_hind_clear=float(np.median(src_hind[side][a:b])),
                )
            )
        for a, b in swing:
            # `peak` is the like-for-like pair with `src_peak`; the exact query is carried
            # alongside and scored only as `scraping`, which needs no source reference.
            w = Swing(
                side=side,
                start=a,
                end=b,
                peak=float(probe[side][a:b].max()),
                src_peak=float(src_clear[side][a:b].max()),
                touched=float(dist[side][a:b].min()),
                exact_peak=float(vert[side][a:b].max()),
            )
            out.swings.append(w)
            if w.dragging(tol) or w.scraping(tol):
                swing_bad[a:b] = True

    out.n_seated_phases_dropped = n_seated_phases
    out.stances.sort(key=lambda s: (s.start, s.side))
    out.swings.sort(key=lambda w: (w.start, w.side))
    if keep_per_frame:
        out.per_frame = {
            "dist_left": dist["left"],
            "dist_right": dist["right"],
            "support_left": support["left"],
            "support_right": support["right"],
            # These are the exact source-timed support/floating decisions used by the
            # benchmark. Phase-level quality flags deliberately ignore a brief lift near
            # toe-off, so relying on them alone made visually obvious hover frames render
            # with no warning band.
            "stance_left": stance_mask["left"],
            "stance_right": stance_mask["right"],
            "floating_left": stance_mask["left"] & (support["left"] > tol.float_tol),
            "floating_right": stance_mask["right"] & (support["right"] > tol.float_tol),
            # Rendering-only contact diagnostic. The formal floating threshold remains
            # 20 mm, but a 2+ mm gap has no physical MuJoCo contact point and is visibly
            # useful when reviewing the near-surface hover reported in descent clips.
            "contact_gap_left": stance_mask["left"] & (support["left"] > 0.002),
            "contact_gap_right": stance_mask["right"] & (support["right"] > 0.002),
            "vert_left": vert["left"],
            "vert_right": vert["right"],
            "hind_left": hind["left"],
            "hind_right": hind["right"],
            "selfpen": selfpen,
            "swing_bad": swing_bad,
            "seat_gap": seat_gap,
            "seat_bad": seat_bad,
            "body_pen": body_pen,
            "terrain_contact_desired": desired_terrain_contact,
            "terrain_contact_distance": terrain_contact_distance,
            "interleg_selfpen": interleg_selfpen,
            "all_selfpen": all_selfpen,
            "foot_pos_left": pos["left"],
            "foot_pos_right": pos["right"],
        }
    return out


def to_json(meas: Measurement) -> dict:
    """A JSON-safe dict of everything but the per-frame arrays."""
    return {
        "motion": meas.motion,
        "n_frames": meas.n_frames,
        "fps": meas.fps,
        "n_boxes": meas.n_boxes,
        "tolerances": asdict(meas.tol),
        "summary": meas.summary(),
        "fails": meas.fails(),
        "seats": [asdict(t) | {"flags": t.flags(meas.tol)} for t in meas.seats],
        "stances": [asdict(s) | {"excess": s.excess, "flags": s.flags(meas.tol)} for s in meas.stances],
        "swings": [
            asdict(w) | {"required": w.required(meas.tol), "dragging": w.dragging(meas.tol)} for w in meas.swings
        ],
        "face_pen": meas.face_pen,
        "face_worst": meas.face_worst,
        "selfpen_worst": meas.selfpen_worst,
        "selfpen_where": meas.selfpen_where,
        "selfpen_frames_frac": meas.selfpen_frac,
    }
