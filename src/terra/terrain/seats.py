"""Detect seated support phases and fit seat geometry from pelvis motion."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from terra._musclemimic import BoxSpec, TerrainSpec
from terra.terrain.shapes import (
    DEFAULT_FREE_SPACE_TOL,
    _exclude_claimed_points,
    _free_space_limit,
)
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS, DEFAULT_LEVEL_TOL, StanceEvent

logger = logging.getLogger(__name__)


#: Maximum pelvis speed for a seated rest, in meters per second.
SEAT_REST_SPEED = 0.15


#: Maximum interior knee angle for a one-frame boundary support observation.  This
#: corresponds to at least 45 degrees of flexion from a straight leg and is used only
#: when the recording boundary leaves no measurable pelvis dwell interval.
SEAT_BOUNDARY_MAX_KNEE_ANGLE_DEG = 135.0


#: Minimum duration of a seated rest in seconds.
SEAT_MIN_REST_S = 0.4


#: Minimum pelvis distance outside the support hull, in meters.
SEAT_MIN_OUTSIDE = 0.10


#: Maximum surface-relative height for conditional support joints, in meters.
SEAT_GROUNDED_Z = 0.15


#: Joints added to the support hull when grounded.
SEAT_CONDITIONAL_SUPPORT = (
    "L_Knee",
    "R_Knee",
    "L_Wrist",
    "R_Wrist",
    "L_Elbow",
    "R_Elbow",
    "L_Shoulder",
    "R_Shoulder",
)


#: Minimum raised support height in meters. This matches the terrain fitter's
#: raised-level threshold: a lower inferred surface is treated as local floor rather
#: than as a separate seat. In particular, this is not an ergonomic chair-height prior.
SEAT_MIN_HEIGHT = 0.04


#: Maximum inferred seat height in meters.
SEAT_MAX_HEIGHT = 0.60


#: Vertical distance from the pelvis origin to the seat surface, in meters.
PELVIS_SEAT_OFFSET = 0.16


#: Preferred seat half-extents as ``(depth, width)`` in meters.
SEAT_SIZE_PRIOR = (0.22, 0.22)


#: Required pelvis contact half-extents in meters.
SEAT_CONTACT_MARGIN = (0.10, 0.10)


#: Distance above which pelvis rests are assigned to different seats, in meters.
SEAT_SPLIT_GAP = 0.50


#: Geom-name prefix used for fitted seat boxes.
SEAT_GEOM_PREFIX = "terrain_box_seat"


#: Minimum standing contacts retained when seated contacts are removed.
MIN_STANDING_CONTACTS = 4


@dataclass
class SeatRest:
    """Describe a continuous interval of inferred seated support.

    Attributes:
        start: Index of the first frame in the interval.
        end: Exclusive index of the final frame in the interval.
        z: Median pelvis height during the interval, in meters.
        xy: Pelvis positions in the horizontal plane during the interval.
        outside: Distance from the pelvis to the active support hull, in meters.
        yaw: Direction from the pelvis toward the feet, in radians.
        boundary: ``"start"`` or ``"end"`` when the observed interval is shorter
            than the minimum duration only because it reaches that clip boundary.
        boundary_evidence: Kinematic evidence that justified a shortened boundary
            interval: measured low speed or a one-frame flexed-knee seated posture.
    """

    start: int
    end: int  # exclusive
    z: float  # pelvis height, the median over the run
    xy: np.ndarray = field(repr=False)  # (n, 2) pelvis positions over the run
    outside: float = 0.0  # how far outside the support hull the pelvis sat, metres
    yaw: float = 0.0  # direction from the pelvis towards the feet, radians
    boundary: str | None = None
    boundary_evidence: str | None = None

    @property
    def n_frames(self) -> int:
        """Return the number of frames in the interval."""
        return self.end - self.start


@dataclass(frozen=True)
class _SeatFrame:
    """Pelvis contact samples and the subject-aligned horizontal frame."""

    pelvis_xy: np.ndarray
    centre_xy: np.ndarray
    yaw: float
    local_cosine: float
    local_sine: float
    world_cosine: float
    world_sine: float

    @classmethod
    def from_rests(cls, rests: Sequence[SeatRest]) -> _SeatFrame:
        pelvis_xy = np.concatenate([rest.xy for rest in rests], axis=0)
        centre_xy = np.median(pelvis_xy, axis=0)
        headings = np.asarray([rest.yaw for rest in rests], dtype=float)
        mean_sine = float(np.mean(np.sin(headings)))
        mean_cosine = float(np.mean(np.cos(headings)))
        yaw = (
            float(np.arctan2(mean_sine, mean_cosine))
            if np.hypot(mean_sine, mean_cosine) > 1e-8
            else float(headings[0])
        )
        return cls(
            pelvis_xy=pelvis_xy,
            centre_xy=centre_xy,
            yaw=yaw,
            local_cosine=float(np.cos(-yaw)),
            local_sine=float(np.sin(-yaw)),
            world_cosine=float(np.cos(yaw)),
            world_sine=float(np.sin(yaw)),
        )

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world horizontal points into subject-forward coordinates."""

        delta = np.asarray(xy, dtype=float) - self.centre_xy
        longitudinal = self.local_cosine * delta[:, 0] - self.local_sine * delta[:, 1]
        lateral = self.local_sine * delta[:, 0] + self.local_cosine * delta[:, 1]
        return longitudinal, lateral

    def to_world(self, longitudinal: float, lateral: float) -> np.ndarray:
        """Map one subject-aligned offset back to a world horizontal position."""

        return self.centre_xy + np.array(
            [
                self.world_cosine * longitudinal - self.world_sine * lateral,
                self.world_sine * longitudinal + self.world_cosine * lateral,
            ]
        )


@dataclass(frozen=True)
class _SeatBounds:
    """Longitudinal and lateral seat-face coordinates in a :class:`_SeatFrame`."""

    u_lo: float
    u_hi: float
    v_lo: float
    v_hi: float

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float, float]) -> _SeatBounds:
        return cls(*values)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.u_lo, self.u_hi, self.v_lo, self.v_hi


def _knee_angle_deg(joints: np.ndarray, names: Sequence[str], frame: int, side: str) -> float | None:
    """Return one interior hip-knee-ankle angle, or ``None`` when unavailable."""

    required = [f"{side}_Hip", f"{side}_Knee", f"{side}_Ankle"]
    if any(name not in names for name in required):
        return None
    hip, knee, ankle = (np.asarray(joints[frame, names.index(name)], dtype=float) for name in required)
    upper = hip - knee
    lower = ankle - knee
    scale = float(np.linalg.norm(upper) * np.linalg.norm(lower))
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(upper, lower) / scale, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _flexed_knee_boundary_pose(joints: np.ndarray, names: Sequence[str], frame: int) -> bool:
    """Identify a seated-like endpoint when no pelvis dwell time was recorded."""

    angles = [_knee_angle_deg(joints, names, frame, side) for side in ("L", "R")]
    return all(angle is not None and angle <= SEAT_BOUNDARY_MAX_KNEE_ANGLE_DEG for angle in angles)


def _hull_distance(point: np.ndarray, pts: np.ndarray) -> float:
    """Measure the distance from a point to a two-dimensional convex hull.

    Args:
        point: Point whose distance is measured, with shape ``(2,)``.
        pts: Points defining the hull, with shape ``(n_points, 2)``.

    Returns:
        Zero when ``point`` lies inside the hull; otherwise, the shortest
        distance to the hull boundary in meters.
    """
    point = np.asarray(point, dtype=float)
    pts = np.unique(np.atleast_2d(np.asarray(pts, dtype=float)).round(9), axis=0)
    if len(pts) == 1:
        return float(np.linalg.norm(point - pts[0]))

    def cross(o, a, b) -> float:
        """Compute the signed two-dimensional cross product of three points.

        Args:
            o: Origin point.
            a: First endpoint.
            b: Second endpoint.

        Returns:
            Signed cross product of vectors ``o -> a`` and ``o -> b``.
        """
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    # Andrew's monotone chain, counter-clockwise. Collinear input collapses to the two
    # extreme points, which is exactly the segment the distance should be taken to.
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower, upper = [], []
    for chain, seq in ((lower, p), (upper, p[::-1])):
        for q in seq:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], q) <= 0:
                chain.pop()
            chain.append(q)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 2:
        hull = p[[0, -1]]

    # Distance to the boundary, zeroed when the point is inside every edge. A degenerate
    # (segment) hull has no interior, so the winding test is skipped and the distance to the
    # segment stands.
    d, inside = np.inf, len(hull) >= 3
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        ab = b - a
        t = np.clip(float((point - a) @ ab) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
        d = min(d, float(np.linalg.norm(point - (a + t * ab))))
        if cross(a, b, point) < 0:
            inside = False
    return 0.0 if inside else d


def detect_seat_rests(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    speed_ms: float = SEAT_REST_SPEED,
    min_rest_s: float = SEAT_MIN_REST_S,
    min_outside: float = SEAT_MIN_OUTSIDE,
    grounded_z: float = SEAT_GROUNDED_Z,
    terrain: TerrainSpec | None = None,
    allow_boundary_truncation: bool = True,
) -> list[SeatRest]:
    """Detect stationary pelvis intervals outside the active support hull.

    Args:
        joints: World-space joint positions with shape ``(n_frames, n_joints, 3)``.
        demo_joints: Joint names corresponding to the second axis of ``joints``.
        fps: Motion frame rate in frames per second.
        speed_ms: Maximum pelvis speed for a rest, in meters per second.
        min_rest_s: Minimum rest duration in seconds.
        min_outside: Required pelvis distance outside the support hull, in meters.
        grounded_z: Maximum surface-relative height for a conditional support
            joint, in meters.
        terrain: Previously fitted terrain used to determine local support heights.
        allow_boundary_truncation: Accept otherwise-valid support intervals at the
            first or last frame without requiring the complete minimum duration.

    Returns:
        Detected seat rests in chronological order.
    """
    names = list(demo_joints)
    if "Pelvis" not in names:
        logger.warning("No Pelvis joint in the source skeleton; no seat can be inferred")
        return []

    pelvis = joints[:, names.index("Pelvis")]
    feet = [names.index(j) for j in DEFAULT_CONTACT_JOINTS if j in names]
    conditional = [names.index(j) for j in SEAT_CONDITIONAL_SUPPORT if j in names]
    if not feet:
        return []

    speed = np.linalg.norm(np.diff(pelvis, axis=0), axis=-1) * fps
    speed = np.zeros(1, dtype=float) if not len(speed) else np.concatenate([speed[:1], speed])
    slow = speed < speed_ms
    # A clip may start after support began or stop before support ended.  A measured slow
    # interval reaching that boundary can therefore be shorter than the ordinary minimum
    # duration.  If only one boundary frame remains, bilateral knee flexion distinguishes
    # a seated endpoint from an arbitrary mid-motion cut without using a dataset label or
    # chair-height prior.
    posture_only: set[int] = set()
    if allow_boundary_truncation:
        for frame in {0, len(slow) - 1}:
            if not slow[frame] and _flexed_knee_boundary_pose(joints, names, frame):
                slow[frame] = True
                posture_only.add(frame)
    min_frames = max(int(min_rest_s * fps), 1)

    out: list[SeatRest] = []
    i, n_frames = 0, joints.shape[0]
    while i < n_frames:
        if not slow[i]:
            i += 1
            continue
        j = i
        while j < n_frames and slow[j]:
            j += 1
        boundary = None
        if allow_boundary_truncation and j - i < min_frames:
            if i == 0:
                boundary = "start"
            elif j == n_frames:
                boundary = "end"
        if j - i >= min_frames or boundary is not None:
            mid = (i + j) // 2
            here = np.median(pelvis[i:j, :2], axis=0)

            support = [joints[mid, k, :2] for k in feet]
            for k in conditional:
                floor = terrain.height_at(*joints[mid, k, :2]) if terrain is not None else 0.0
                if joints[mid, k, 2] - float(floor) < grounded_z:
                    support.append(joints[mid, k, :2])

            outside = _hull_distance(here, np.array(support))
            if outside > min_outside:
                towards = np.mean([joints[mid, k, :2] for k in feet], axis=0) - here
                boundary_evidence = None
                if boundary is not None:
                    boundary_evidence = (
                        "flexed_knee_endpoint" if {i, j - 1} & posture_only else "observed_slow_interval"
                    )
                out.append(
                    SeatRest(
                        start=i,
                        end=j,
                        z=float(np.median(pelvis[i:j, 2])),
                        xy=pelvis[i:j, :2].copy(),
                        outside=float(outside),
                        yaw=float(np.arctan2(towards[1], towards[0])),
                        boundary=boundary,
                        boundary_evidence=boundary_evidence,
                    )
                )
        i = j
    return out


def drop_seated_contacts(events: Sequence[StanceEvent], rests: Sequence[SeatRest]) -> tuple[list[StanceEvent], int]:
    """Remove stance events that overlap inferred seated support.

    Args:
        events: Stance events detected throughout the motion.
        rests: Inferred seated-support intervals.

    Returns:
        A pair containing the retained events and the number of removed events.
        All events are retained when removal would leave too few standing contacts.
    """
    if not rests:
        return list(events), 0
    # Any overlap at all, rather than a majority of the run. A still foot produces *one*
    # maximal event, so a foot planted before the subject sits, held through a 7 s sit and
    # still planted afterwards is a single run that is only part seated - and its
    # representative height is the median over the whole of it, which those seated frames
    # have already moved. On `sitdown_standup-02-chair-hamada` a majority rule kept exactly
    # those runs and left the gate failing at 51 mm.
    #
    # Identity, not equality: `StanceEvent` carries an array field, so the generated
    # `__eq__` returns an array and `in` raises on its truth value.
    # Boundary-truncated rests add a support hypothesis but do not rewrite the observed
    # foot evidence.  This makes the boundary rule additive: if its seat candidate is
    # rejected, the foot-terrain reconstruction is exactly the original one.
    complete_rests = [rest for rest in rests if rest.boundary is None]
    seated = {
        id(event)
        for event in events
        if any(min(event.end, rest.end) > max(event.start, rest.start) for rest in complete_rests)
    }
    kept = [e for e in events if id(e) not in seated]
    if len(kept) < MIN_STANDING_CONTACTS:
        return list(events), 0
    return kept, len(seated)


def _group_seat_rests(rests: Sequence[SeatRest], split_gap: float) -> list[list[SeatRest]]:
    """Group repeated rests on one seat while keeping moved seats separate."""

    groups: list[list[SeatRest]] = []
    for rest in sorted(rests, key=lambda value: value.z):
        here = np.median(rest.xy, axis=0)
        for group in groups:
            there = np.median(np.concatenate([value.xy for value in group], axis=0), axis=0)
            same_height = abs(rest.z - float(np.mean([value.z for value in group]))) <= DEFAULT_LEVEL_TOL
            same_position = float(np.linalg.norm(here - there)) <= split_gap
            if same_height and same_position:
                group.append(rest)
                break
        else:
            groups.append([rest])
    return groups


def _seat_report_entry(
    index: int,
    rests: Sequence[SeatRest],
    pelvis_offset: float,
    support_heights: Sequence[float] | None,
) -> tuple[float, dict[str, object]]:
    pelvis_z = float(np.mean([rest.z for rest in rests]))
    if support_heights is None:
        top = pelvis_z - pelvis_offset
        height_source = "fixed_pelvis_offset"
        samples: list[float] = []
    else:
        samples = [float(value) for value in support_heights]
        top = float(np.median(samples))
        height_source = "posed_body_surface"
    entry: dict[str, object] = {
        "index": index,
        "n_rests": len(rests),
        "frames": [(rest.start, rest.end) for rest in rests],
        "pelvis_z": pelvis_z,
        "top": top,
        "height_source": height_source,
        "outside": float(np.mean([rest.outside for rest in rests])),
    }
    boundaries = sorted({rest.boundary for rest in rests if rest.boundary is not None})
    if boundaries:
        entry["boundary_truncated"] = boundaries
        entry["boundary_evidence"] = sorted(
            {rest.boundary_evidence for rest in rests if rest.boundary_evidence is not None}
        )
    if samples:
        entry["support_height_samples"] = samples
    return top, entry


def _initial_seat_bounds(
    frame: _SeatFrame,
    size_prior: tuple[float, float],
    contact_margin: tuple[float, float],
) -> tuple[_SeatBounds, _SeatBounds, float]:
    """Return preferred bounds, required pelvis support, and support half-width."""

    longitudinal, lateral = frame.project(frame.pelvis_xy)
    required = _SeatBounds(
        u_lo=float(longitudinal.min()) - contact_margin[0],
        u_hi=float(longitudinal.max()) + contact_margin[0],
        v_lo=float(lateral.min()) - contact_margin[1],
        v_hi=float(lateral.max()) + contact_margin[1],
    )
    bounds = _SeatBounds(
        u_lo=min(-size_prior[0], required.u_lo),
        u_hi=max(size_prior[0], required.u_hi),
        v_lo=min(-size_prior[1], required.v_lo),
        v_hi=max(size_prior[1], required.v_hi),
    )
    support_half_width = max(abs(required.v_lo), abs(required.v_hi), size_prior[1])
    return bounds, required, support_half_width


def _constrain_seat_depth(
    body_points: np.ndarray,
    frame: _SeatFrame,
    bounds: _SeatBounds,
    required: _SeatBounds,
    support_half_width: float,
    top: float,
    *,
    free_space_tol: float,
    clearance: float,
    seat_index: int,
) -> tuple[_SeatBounds, str | None]:
    """Limit the seat depth to body free space."""

    lo_limit, hi_limit = _free_space_limit(
        body_points,
        frame.yaw,
        frame.centre_xy,
        support_half_width,
        top,
        free_space_tol,
        required.u_lo,
        required.u_hi,
        clearance,
    )
    u_lo = max(bounds.u_lo, lo_limit)
    u_hi = min(bounds.u_hi, hi_limit)
    warning = None
    if u_lo > required.u_lo or u_hi < required.u_hi:
        warning = (
            f"seat {seat_index}: free space ({lo_limit:.3f}, {hi_limit:.3f}) is tighter than the "
            f"pelvis contact patch ({required.u_lo:.3f}, {required.u_hi:.3f}); clamped to free "
            f"space, so the glutes may overhang the edge"
        )
        u_lo, u_hi = min(u_lo, u_hi), max(u_lo, u_hi)
    constrained = _SeatBounds(u_lo=u_lo, u_hi=u_hi, v_lo=bounds.v_lo, v_hi=bounds.v_hi)
    return constrained, warning


def _exclude_claimed_footfalls(
    claimed: np.ndarray,
    frame: _SeatFrame,
    bounds: _SeatBounds,
    required: _SeatBounds,
    *,
    clearance: float,
    seat_index: int,
) -> tuple[_SeatBounds, int | None, str | None]:
    """Trim a seat around standing contacts without cutting its pelvis support."""

    if not len(claimed):
        return bounds, None, None

    claimed_u, claimed_v = frame.project(claimed)
    values, unresolved = _exclude_claimed_points(
        bounds.as_tuple(),
        required.as_tuple(),
        claimed_u,
        claimed_v,
        clearance,
    )
    warning = None
    if unresolved:
        warning = (
            f"seat {seat_index}: {unresolved} footfall(s) lie inside the seat's own contact "
            f"patch, so the subject stood where they later sat and the two surfaces "
            f"overlap in xy"
        )
    return _SeatBounds.from_tuple(values), unresolved, warning


def _build_seat_box(frame: _SeatFrame, bounds: _SeatBounds, top: float, box_index: int) -> BoxSpec:
    """Construct one solid seat box from accepted horizontal bounds."""

    u_centre = 0.5 * (bounds.u_lo + bounds.u_hi)
    u_half = max(0.5 * (bounds.u_hi - bounds.u_lo), 1e-3)
    v_centre = 0.5 * (bounds.v_lo + bounds.v_hi)
    v_half = max(0.5 * (bounds.v_hi - bounds.v_lo), 1e-3)
    position = frame.to_world(u_centre, v_centre)
    return BoxSpec(
        pos=(float(position[0]), float(position[1]), top / 2),
        size=(u_half, v_half, top / 2),
        yaw=frame.yaw,
        # The prefix is what scene_geom_ids uses to select static environment
        # geometry. A differently named box would not be built or collided with.
        name=f"{SEAT_GEOM_PREFIX}_{box_index}",
    )


def _fit_seat_group(
    group: Sequence[SeatRest],
    *,
    group_index: int,
    box_index: int,
    body_points: np.ndarray,
    claimed: np.ndarray,
    pelvis_offset: float,
    size_prior: tuple[float, float],
    contact_margin: tuple[float, float],
    free_space_tol: float,
    clearance: float,
    min_height: float,
    max_height: float,
    support_heights: Sequence[float] | None,
) -> tuple[BoxSpec | None, dict[str, object], list[str]]:
    """Fit one grouped seat candidate and return its complete diagnostics."""

    top, entry = _seat_report_entry(group_index, group, pelvis_offset, support_heights)
    warnings = []
    if not min_height <= top <= max_height:
        entry["skipped"] = (
            f"inferred surface at {top * 1000:.0f} mm is outside the "
            f"{min_height * 1000:.0f}-{max_height * 1000:.0f} mm seat range; the subject "
            f"is on the ground, or standing and leaning on something"
        )
        return None, entry, warnings

    frame = _SeatFrame.from_rests(group)
    entry["yaw"] = frame.yaw
    bounds, required, support_half_width = _initial_seat_bounds(frame, size_prior, contact_margin)
    bounds, free_space_warning = _constrain_seat_depth(
        body_points,
        frame,
        bounds,
        required,
        support_half_width,
        top,
        free_space_tol=free_space_tol,
        clearance=clearance,
        seat_index=group_index,
    )
    if free_space_warning is not None:
        entry["clamped"] = True
        warnings.append(free_space_warning)

    bounds, unresolved, claimed_warning = _exclude_claimed_footfalls(
        claimed,
        frame,
        bounds,
        required,
        clearance=clearance,
        seat_index=group_index,
    )
    if unresolved is not None:
        entry["claimed_unresolved"] = unresolved
    if claimed_warning is not None:
        warnings.append(claimed_warning)

    box = _build_seat_box(frame, bounds, top, box_index)
    entry["box"] = box_index
    entry["extent_u"] = (bounds.u_lo, bounds.u_hi)
    entry["extent_v"] = (bounds.v_lo, bounds.v_hi)
    return box, entry, warnings


def fit_seat(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    rests: Sequence[SeatRest],
    stance_events: Sequence[StanceEvent] = (),
    pelvis_offset: float = PELVIS_SEAT_OFFSET,
    size_prior: tuple[float, float] = SEAT_SIZE_PRIOR,
    contact_margin: tuple[float, float] = SEAT_CONTACT_MARGIN,
    split_gap: float = SEAT_SPLIT_GAP,
    free_space_tol: float = DEFAULT_FREE_SPACE_TOL,
    clearance: float = 0.02,
    min_height: float = SEAT_MIN_HEIGHT,
    max_height: float = SEAT_MAX_HEIGHT,
    support_heights: Sequence[float] | None = None,
) -> tuple[list[BoxSpec], dict]:
    """Fit an oriented box to each inferred seat.

    Args:
        joints: World-space joint positions with shape ``(n_frames, n_joints, 3)``.
        demo_joints: Joint names corresponding to the second axis of ``joints``.
        rests: Inferred seated-support intervals.
        stance_events: Foot contacts that fitted seats must avoid covering.
        pelvis_offset: Vertical distance from the pelvis to the seat surface, in meters.
        size_prior: Preferred seat half-extents as ``(depth, width)`` in meters.
        contact_margin: Required half-extents around each pelvis rest, in meters.
        split_gap: Distance above which rests are assigned to separate seats, in meters.
        free_space_tol: Allowed penetration below the seat's top face, in meters.
        clearance: Margin between a seat face and a limiting point, in meters.
        min_height: Minimum accepted seat height in meters.
        max_height: Maximum accepted seat height in meters.
        support_heights: Optional source-body support height for every element of
            ``rests``. When supplied, grouped seats use the median of these posed
            surface observations instead of a fixed pelvis offset.

    Returns:
        A pair containing fitted seat boxes and a diagnostic report.
    """
    seat_entries: list[dict[str, object]] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "n_rests": len(rests),
        "seats": seat_entries,
        "warnings": warnings,
    }
    boundary_count = sum(rest.boundary is not None for rest in rests)
    if boundary_count:
        report["n_boundary_truncated_rests"] = boundary_count
    if not rests:
        if support_heights is not None and len(support_heights):
            raise ValueError("support_heights must be empty when there are no seat rests")
        return [], report

    support_by_rest: dict[int, float] | None = None
    if support_heights is not None:
        values = np.asarray(support_heights, dtype=float)
        if values.shape != (len(rests),):
            raise ValueError(
                f"support_heights must contain one value per seat rest; expected {len(rests)}, got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("support_heights must be finite")
        support_by_rest = {id(rest): float(value) for rest, value in zip(rests, values, strict=True)}

    groups = _group_seat_rests(rests, split_gap)
    body_points = joints.reshape(-1, 3)
    claimed = np.concatenate([event.xy for event in stance_events], axis=0) if stance_events else np.zeros((0, 2))
    boxes: list[BoxSpec] = []

    for group_index, group in enumerate(groups):
        group_support = None if support_by_rest is None else [support_by_rest[id(rest)] for rest in group]
        box, entry, group_warnings = _fit_seat_group(
            group,
            group_index=group_index,
            box_index=len(boxes),
            body_points=body_points,
            claimed=claimed,
            pelvis_offset=pelvis_offset,
            size_prior=size_prior,
            contact_margin=contact_margin,
            free_space_tol=free_space_tol,
            clearance=clearance,
            min_height=min_height,
            max_height=max_height,
            support_heights=group_support,
        )
        warnings.extend(group_warnings)
        seat_entries.append(entry)
        if box is not None:
            boxes.append(box)

    return boxes, report
