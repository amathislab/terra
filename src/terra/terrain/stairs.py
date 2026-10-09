"""Fit stair flights from motion-derived support levels."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from terra._musclemimic import BoxSpec, TerrainSpec
from terra.terrain.shapes import (
    DEFAULT_CONTACT_MARGIN,
    DEFAULT_CONTACT_TOL,
    DEFAULT_FREE_SPACE_TOL,
)
from terra.terrain.stance import StanceEvent

#: Robot sole half-extents as ``(longitudinal, lateral)`` distances in meters.
#:
#: The complete MyoFullBody sole is 0.24 x 0.09 m across its rear- and fore-foot geoms.
#: The previous 0.085 x 0.035 m values were the half-size of only ``*_foot_col4`` and
#: omitted the forefoot geoms.  That left a visually planted foot at a tread/ramp edge even
#: when the source contact probe itself was covered.  Terrain support is for the complete
#: target sole, so use the complete footprint here.
STAIR_SOLE_HALF = (0.12, 0.045)


#: Minimum number of raised levels required to fit a stair flight.
STAIR_MIN_LEVELS = 3


#: Preferred full stair width before applying free-space constraints, in meters.
STAIR_WIDTH_PRIOR = 1.00


#: Minimum number of body points needed to impose a stair-width limit.
STAIR_MIN_BLOCKING_POINTS = 10


#: A clip-boundary foot may extend a terminal landing only when both its ankle and
#: toe agree with that support level.  This prevents a descent's final swing toward
#: the floor from stretching the lowest raised tread across the floor.
STAIR_TERMINAL_HEIGHT_TOL = 0.03


#: Number of frames summarized at each clip boundary when looking for an incomplete
#: terminal support.  Terrain outside that boundary evidence is not identifiable.
STAIR_TERMINAL_WINDOW = 5


#: Largest per-level correction accepted when imposing a shared riser, in meters.
#:
#: A stair flight is a repeated geometric primitive, not a collection of unrelated box
#: tops.  The motion-derived contact heights are noisy observations of that primitive.
#: We accept the least-squares shared-riser model only when every correction stays within
#: the same 30 mm free-space resolution used by the physical validator.  Larger departures
#: remain raw observations rather than being forced into a regular staircase.
STAIR_HEIGHT_REGULARIZATION_TOL = DEFAULT_FREE_SPACE_TOL


#: Numerical clearance retained inside the physical support gate, in meters.
STAIR_SUPPORT_SAFETY_MARGIN = 1e-6


def _regularize_stair_heights(
    heights: Sequence[float],
    *,
    support_targets: Sequence[Sequence[float]] | None = None,
    max_adjustment: float = STAIR_HEIGHT_REGULARIZATION_TOL,
    support_tolerance: float = DEFAULT_CONTACT_TOL,
) -> tuple[list[float], dict[str, object]]:
    """Fit one motion-only shared riser to a coherent sequence of support levels.

    The level ordinal is supplied by the already ordered contact-height clusters.  No
    nominal apparatus height or dataset identity is consulted.  An adjustment
    bound prevents the structural prior from hiding a missing tread, a skipped step, or a
    badly estimated contact level.  When individual stance targets are supplied, the
    least-squares correction is blended back toward the observed level until every
    motion-derived target remains inside the same support gate used by validation.
    """

    raw = np.asarray(heights, dtype=float)
    if raw.ndim != 1 or len(raw) < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("stair heights must contain at least two finite values")
    if not np.isfinite(max_adjustment) or max_adjustment < 0.0:
        raise ValueError("max_adjustment must be finite and non-negative")
    if not np.isfinite(support_tolerance) or support_tolerance <= STAIR_SUPPORT_SAFETY_MARGIN:
        raise ValueError("support_tolerance must be finite and larger than the safety margin")

    targets: list[np.ndarray] | None = None
    if support_targets is not None:
        if len(support_targets) != len(raw):
            raise ValueError("support_targets must contain one sequence per stair height")
        targets = [np.asarray(level, dtype=float) for level in support_targets]
        if any(level.ndim != 1 or not len(level) or not np.all(np.isfinite(level)) for level in targets):
            raise ValueError("every support-target level must contain finite scalar heights")

    ordinal = np.arange(len(raw), dtype=float)
    design = np.column_stack([np.ones(len(raw)), ordinal])
    intercept, riser = np.linalg.lstsq(design, raw, rcond=None)[0]
    fitted = intercept + riser * ordinal
    correction = fitted - raw
    worst = float(np.max(np.abs(correction)))
    structurally_accepted = bool(riser > 0.0 and worst <= max_adjustment)
    blend = 1.0 if structurally_accepted else 0.0
    feasible_interval = [0.0, 1.0]
    residual_raw = residual_fitted = residual_effective = None
    if structurally_accepted and targets is not None:
        # Each event supplies |raw_i + blend * correction_i - target| <= tolerance.
        # Intersect those one-dimensional intervals and retain the largest feasible
        # blend: this preserves as much coherent shared-riser structure as the motion
        # itself supports, without querying any apparatus geometry or labels.
        tolerance = support_tolerance - STAIR_SUPPORT_SAFETY_MARGIN
        lower, upper = feasible_interval
        for raw_height, delta, level_targets in zip(raw, correction, targets, strict=True):
            for target in level_targets:
                residual = raw_height - target
                if abs(delta) <= np.finfo(float).eps:
                    if abs(residual) > tolerance:
                        lower, upper = 1.0, 0.0
                        break
                    continue
                bounds = sorted(((-tolerance - residual) / delta, (tolerance - residual) / delta))
                lower = max(lower, float(bounds[0]))
                upper = min(upper, float(bounds[1]))
            if lower > upper:
                break
        feasible_interval = [float(lower), float(upper)]
        blend = float(np.clip(upper, 0.0, 1.0)) if lower <= upper and upper > 0.0 else 0.0

        target_level = np.concatenate(targets)
        target_raw = np.concatenate([np.full(len(level), height) for height, level in zip(raw, targets, strict=True)])
        target_fitted = np.concatenate(
            [np.full(len(level), height) for height, level in zip(fitted, targets, strict=True)]
        )
        residual_raw = float(np.max(np.abs(target_raw - target_level)))
        residual_fitted = float(np.max(np.abs(target_fitted - target_level)))

    accepted = bool(structurally_accepted and blend > 0.0)
    effective = raw + blend * correction if accepted else raw
    if targets is not None:
        target_effective = np.concatenate(
            [np.full(len(level), height) for height, level in zip(effective, targets, strict=True)]
        )
        residual_effective = float(np.max(np.abs(target_effective - np.concatenate(targets))))

    model = "observed_levels"
    if accepted:
        model = "shared_riser_least_squares" if blend == 1.0 else "support_constrained_shared_riser"
    report: dict[str, object] = {
        "model": model,
        "accepted": accepted,
        "raw_heights": raw.tolist(),
        "fitted_heights": fitted.tolist(),
        "effective_heights": effective.tolist(),
        "shared_riser": float(riser),
        "max_abs_adjustment": worst,
        "max_abs_applied_adjustment": float(np.max(np.abs(effective - raw))),
        "rms_adjustment": float(np.sqrt(np.mean(correction**2))),
        "max_adjustment": float(max_adjustment),
        "regularization_blend": blend,
        "support_tolerance": float(support_tolerance),
        "support_safety_margin": STAIR_SUPPORT_SAFETY_MARGIN,
        "feasible_blend_interval": feasible_interval,
        "support_residual_max_raw": residual_raw,
        "support_residual_max_fitted": residual_fitted,
        "support_residual_max_effective": residual_effective,
        "evidence": "ordered motion-derived support levels only",
    }
    if not accepted:
        report["rejected"] = (
            "non-positive shared riser"
            if riser <= 0.0
            else (
                "a level would move beyond the physical-resolution bound"
                if not structurally_accepted
                else "no positive shared-riser blend satisfies the motion support gate"
            )
        )
    return effective.tolist(), report


def _flight_axis(level_xy: Sequence[np.ndarray]) -> tuple[float, np.ndarray]:
    """Estimate a stair flight's horizontal axis from level centroids.

    Args:
        level_xy: Horizontal footfall positions for each level, ordered by height.

    Returns:
        Flight yaw in radians and its horizontal origin. The axis points toward
        increasing support height.
    """
    centroids = np.array([xy.mean(axis=0) for xy in level_xy])
    origin = centroids.mean(axis=0)
    if len(centroids) < 2:
        return 0.0, origin
    d = centroids - origin
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    if np.dot(centroids[-1] - centroids[0], axis) < 0:
        axis = -axis
    return float(np.arctan2(axis[1], axis[0])), origin


@dataclass(frozen=True)
class _FlightFrame:
    """Horizontal coordinate frame shared by every tread in a stair flight."""

    yaw: float
    origin: np.ndarray
    cosine: float
    sine: float

    @classmethod
    def from_levels(cls, level_xy: Sequence[np.ndarray]) -> _FlightFrame:
        yaw, origin = _flight_axis(level_xy)
        return cls(yaw=yaw, origin=origin, cosine=float(np.cos(yaw)), sine=float(np.sin(yaw)))

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project horizontal points into longitudinal and lateral coordinates."""

        delta = np.asarray(xy, dtype=float) - self.origin
        longitudinal = delta[:, 0] * self.cosine + delta[:, 1] * self.sine
        lateral = -delta[:, 0] * self.sine + delta[:, 1] * self.cosine
        return longitudinal, lateral


def _flight_partition(
    level_u: Sequence[np.ndarray],
    level_event_u: Sequence[np.ndarray],
) -> tuple[list[float], int, int]:
    """Partition a stair flight into non-overlapping tread spans.

    Args:
        level_u: Per-frame footfall coordinates for each level, ordered by height.
        level_event_u: Median footfall coordinate for each stance event and level.

    Returns:
        Cut coordinates, the number of misassigned events, and the number of
        misassigned frames.
    """
    cuts: list[float] = []
    wrong_events = wrong_frames = 0
    for (lo_lvl, hi_lvl), (lo_med, hi_med) in zip(
        itertools.pairwise(level_u), itertools.pairwise(level_event_u), strict=True
    ):
        candidates = np.unique(np.concatenate([lo_med, hi_med]))
        mids = np.concatenate(
            [
                [candidates[0] - 0.05],
                0.5 * (candidates[:-1] + candidates[1:]),
                [candidates[-1] + 0.05],
            ]
        )
        n_ev = np.array([(lo_med > c).sum() + (hi_med < c).sum() for c in mids])
        n_fr = np.array([(lo_lvl > c).sum() + (hi_lvl < c).sum() for c in mids])
        best_i = int(np.lexsort((n_fr, n_ev))[0])
        best = mids[best_i]

        # A sole ahead of the lower tread's last footfall and a sole behind the upper
        # tread's first.  When there is free space between those complete-foot support
        # envelopes, put the riser at its midpoint.  The old arbitrary zero-error
        # separator could leave all of that spare tread on one side of a foot even though
        # the same geometry admitted a centred placement.
        room = float(lo_lvl.max()) + STAIR_SOLE_HALF[0]
        headroom = float(hi_lvl.min()) - STAIR_SOLE_HALF[0]
        if room <= headroom:
            best = 0.5 * (room + headroom)
        else:
            # The full support envelopes overlap, so both feet cannot be given their
            # requested margin.  Preserve the event/frame-optimal separator and make the
            # minimum concession needed to keep the upper foot on its own tread.
            best = max(best, headroom)
        if cuts and best <= cuts[-1]:
            best = cuts[-1] + 1e-4
        cuts.append(float(best))
        wrong_events += int((lo_med > best).sum() + (hi_med < best).sum())
        wrong_frames += int((lo_lvl > best).sum() + (hi_lvl < best).sum())
    return cuts, wrong_events, wrong_frames


def _project_levels(
    level_xy: Sequence[np.ndarray],
    levels: Sequence[Sequence[StanceEvent]],
    frame: _FlightFrame,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Project stance frames and event centres into the shared flight frame."""

    level_u = []
    level_event_u = []
    level_v = []
    for xy, level in zip(level_xy, levels, strict=True):
        longitudinal, lateral = frame.project(xy)
        level_u.append(longitudinal)
        level_event_u.append(np.array([float(np.median(frame.project(event.xy)[0])) for event in level]))
        level_v.append(lateral)
    return level_u, level_event_u, np.concatenate(level_v)


def _motion_envelope(
    motion_foot_xy: np.ndarray | None,
    motion_foot_xyz: Mapping[str, np.ndarray] | None,
    offsets: Mapping[str, float],
    heights: Sequence[float],
    level_u: Sequence[np.ndarray],
    frame: _FlightFrame,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, int]]:
    """Return clip-boundary extents for the two terminal support levels.

    A landmark merely passing through a tread's height is not evidence that the tread
    continues beneath it.  Extension therefore requires a complete foot (ankle and toe)
    to agree with the same support plane at the head or tail of the clip.  The detected
    stance events remain the fallback when that stronger evidence is absent.
    """

    if motion_foot_xyz:
        by_terminal: list[list[np.ndarray]] = [[], []]
        terminal_heights = (float(heights[0]), float(heights[-1]))
        points_by_name: dict[str, np.ndarray] = {}
        for name, raw_points in motion_foot_xyz.items():
            points = np.asarray(raw_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                raise ValueError("motion_foot_xyz values must have non-empty shape (N, 3)")
            if not np.isfinite(points).all():
                raise ValueError("motion_foot_xyz values must be finite")
            points_by_name[name] = points

        for side in ("L", "R"):
            pair = (f"{side}_Ankle", f"{side}_Toe")
            if not all(name in points_by_name and name in offsets for name in pair):
                continue
            n_frames = min(len(points_by_name[name]) for name in pair)
            window = min(STAIR_TERMINAL_WINDOW, n_frames)
            for boundary_slice in (slice(0, window), slice(n_frames - window, n_frames)):
                boundary_points = [np.median(points_by_name[name][boundary_slice], axis=0) for name in pair]
                surface_heights = [
                    float(point[2] - float(offsets[name])) for name, point in zip(pair, boundary_points, strict=True)
                ]
                for terminal, height in enumerate(terminal_heights):
                    if all(
                        abs(surface_height - height) <= STAIR_TERMINAL_HEIGHT_TOL for surface_height in surface_heights
                    ):
                        longitudinal, _ = frame.project(np.asarray(boundary_points, dtype=float)[:, :2])
                        by_terminal[terminal].append(longitudinal)

        first = np.concatenate(by_terminal[0]) if by_terminal[0] else np.asarray(level_u[0])
        last = np.concatenate(by_terminal[1]) if by_terminal[1] else np.asarray(level_u[-1])
        counts = {"first": len(first), "last": len(last)}
        return first, last, "clip_boundary_paired_foot_landmarks", counts

    if motion_foot_xy is None:
        return (
            np.asarray(level_u[0]),
            np.asarray(level_u[-1]),
            "terminal_stance_events",
            {"first": len(level_u[0]), "last": len(level_u[-1])},
        )

    points = np.asarray(motion_foot_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("motion_foot_xy must have non-empty shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("motion_foot_xy must be finite")
    longitudinal, _ = frame.project(points)
    return (
        longitudinal,
        longitudinal,
        "full_motion_foot_landmarks",
        {"first": len(longitudinal), "last": len(longitudinal)},
    )


def _tread_spans(
    level_u: Sequence[np.ndarray],
    cuts: Sequence[float],
    first_envelope_u: np.ndarray,
    last_envelope_u: np.ndarray,
    contact_margin: tuple[float, float],
) -> list[tuple[float, float]]:
    """Turn consecutive riser cuts into complete-foot support spans."""

    spans = []
    for index in range(len(level_u)):
        lo = (
            cuts[index - 1]
            if index
            else min(float(level_u[0].min()), float(first_envelope_u.min())) - STAIR_SOLE_HALF[0] - contact_margin[0]
        )
        hi = (
            cuts[index]
            if index < len(cuts)
            else max(float(level_u[-1].max()), float(last_envelope_u.max())) + STAIR_SOLE_HALF[0] + contact_margin[0]
        )
        spans.append((lo, hi))
    return spans


def _stair_width(
    joints: np.ndarray,
    frame: _FlightFrame,
    spans: Sequence[tuple[float, float]],
    heights: Sequence[float],
    lateral_footfalls: np.ndarray,
    *,
    free_space_tol: float,
    width_prior: float,
    clearance: float,
) -> tuple[float, float, int]:
    """Fit a shared width without enclosing unsupported body points."""

    support_width = float(np.abs(lateral_footfalls).max()) + STAIR_SOLE_HALF[1]
    body = joints.reshape(-1, 3)
    body_u, body_v = frame.project(body[:, :2])
    limit = np.inf
    n_blocking = 0
    for (lo, hi), height in zip(spans, heights, strict=True):
        inside = (
            (body_u >= lo) & (body_u <= hi) & (body[:, 2] < height - free_space_tol) & (np.abs(body_v) > support_width)
        )
        if int(inside.sum()) < STAIR_MIN_BLOCKING_POINTS:
            continue
        n_blocking += int(inside.sum())
        limit = min(limit, float(np.abs(body_v[inside]).min()) - clearance)

    requested = max(support_width, 0.5 * width_prior)
    # Never narrower than the footfalls, which would leave them over no box at all.
    width = requested if not np.isfinite(limit) else max(min(requested, limit), support_width)
    return width, support_width, n_blocking


def _build_stair_boxes(
    frame: _FlightFrame,
    heights: Sequence[float],
    spans: Sequence[tuple[float, float]],
    width: float,
) -> tuple[BoxSpec, ...]:
    """Build one solid box for every raised tread in the flight."""

    boxes = []
    for height, (lo, hi) in zip(heights, spans, strict=True):
        if height < 0.04:
            continue  # TerrainSpec already provides the floor at z=0.
        u_mid = 0.5 * (lo + hi)
        u_half = 0.5 * (hi - lo)
        centre = frame.origin + np.array([frame.cosine * u_mid, frame.sine * u_mid])
        boxes.append(
            BoxSpec(
                pos=(float(centre[0]), float(centre[1]), height / 2),
                size=(max(u_half, STAIR_SOLE_HALF[0]), max(width, STAIR_SOLE_HALF[1]), height / 2),
                yaw=frame.yaw,
                name=f"terrain_box_{len(boxes)}",
            )
        )
    return tuple(boxes)


def _stair_report(
    *,
    frame: _FlightFrame,
    heights: Sequence[float],
    height_model: Mapping[str, object],
    spans: Sequence[tuple[float, float]],
    level_u: Sequence[np.ndarray],
    level_event_u: Sequence[np.ndarray],
    first_envelope_u: np.ndarray,
    last_envelope_u: np.ndarray,
    envelope_source: str,
    envelope_counts: Mapping[str, int],
    width: float,
    support_width: float,
    n_blocking: int,
    misassigned_events: int,
    misassigned_frames: int,
) -> dict[str, object]:
    """Serialize stair geometry and fitting diagnostics."""

    return {
        "model": "stair_flight",
        "yaw": frame.yaw,
        "heights": list(heights),
        "height_model": dict(height_model),
        "spans": list(spans),
        "level_centres_u": [float(np.median(values)) for values in level_event_u],
        "level_support_spans_u": [
            (
                float(values.min()) - STAIR_SOLE_HALF[0],
                float(values.max()) + STAIR_SOLE_HALF[0],
            )
            for values in level_u
        ],
        "motion_foot_span_u": (
            float(first_envelope_u.min()),
            float(last_envelope_u.max()),
        ),
        "terminal_motion_foot_spans_u": {
            "first": (float(first_envelope_u.min()), float(first_envelope_u.max())),
            "last": (float(last_envelope_u.min()), float(last_envelope_u.max())),
        },
        "terminal_motion_foot_counts": dict(envelope_counts),
        "landing_extent_source": envelope_source,
        "width": width,
        "support_width": support_width,
        "n_blocking": n_blocking,
        "misassigned": misassigned_events,
        "misassigned_frames": misassigned_frames,
        "n_levels": len(level_u),
    }


def fit_stair_flight(
    joints: np.ndarray,
    levels: Sequence[Sequence[StanceEvent]],
    offsets: dict[str, float],
    free_space_tol: float = DEFAULT_FREE_SPACE_TOL,
    width_prior: float = STAIR_WIDTH_PRIOR,
    clearance: float = 0.02,
    contact_margin: tuple[float, float] = DEFAULT_CONTACT_MARGIN,
    motion_foot_xy: np.ndarray | None = None,
    motion_foot_xyz: Mapping[str, np.ndarray] | None = None,
    support_height_targets: Sequence[Sequence[float]] | None = None,
) -> tuple[TerrainSpec, dict]:
    """Fit support levels as a staircase with a shared axis and width.

    Args:
        joints: Source joint positions with shape ``(n_frames, n_joints, 3)``.
        levels: Stance events grouped by support height, lowest first.
        offsets: Surface-relative height offset for each contact joint.
        free_space_tol: Allowed penetration below a tread surface, in meters.
        width_prior: Preferred full width of the staircase in meters.
        clearance: Margin between a tread face and a blocking point, in meters.
        contact_margin: Extra longitudinal/lateral surface past a complete planted sole.
        motion_foot_xy: Optional horizontal positions of all foot landmarks throughout
            the recorded clip. These affect only the terminal landing extents, allowing
            them to support motion beyond the last complete detected stance.
        motion_foot_xyz: Optional per-landmark trajectories. When supplied, terminal
            landing extension uses only a complete ankle/toe pair at a clip boundary
            whose calibrated surface heights agree with the corresponding terminal level.
        support_height_targets: Optional motion-derived surface-height observations,
            grouped onto the fitted levels. This can include weak stance clusters that
            do not justify their own tread but must still be supported by the nearest
            accepted tread. When omitted, the fitted level events are used directly.

    Returns:
        A pair containing the fitted terrain and stair-fit diagnostics.
    """
    level_targets = [[event.z - offsets.get(event.joint, 0.0) for event in level] for level in levels]
    raw_heights = [float(np.mean(targets)) for targets in level_targets]
    support_targets = level_targets if support_height_targets is None else support_height_targets
    heights, height_model = _regularize_stair_heights(
        raw_heights,
        support_targets=support_targets,
    )
    height_model["support_target_counts"] = [len(targets) for targets in support_targets]
    level_xy = [np.vstack([event.xy for event in level]) for level in levels]
    frame = _FlightFrame.from_levels(level_xy)
    level_u, level_event_u, lateral_footfalls = _project_levels(level_xy, levels, frame)
    first_envelope_u, last_envelope_u, envelope_source, envelope_counts = _motion_envelope(
        motion_foot_xy,
        motion_foot_xyz,
        offsets,
        heights,
        level_u,
        frame,
    )
    cuts, misassigned, misassigned_frames = _flight_partition(level_u, level_event_u)
    spans = _tread_spans(
        level_u,
        cuts,
        first_envelope_u,
        last_envelope_u,
        contact_margin,
    )
    width, support_width, n_blocking = _stair_width(
        joints,
        frame,
        spans,
        heights,
        lateral_footfalls,
        free_space_tol=free_space_tol,
        width_prior=width_prior,
        clearance=clearance,
    )
    boxes = _build_stair_boxes(frame, heights, spans, width)
    report = _stair_report(
        frame=frame,
        heights=heights,
        height_model=height_model,
        spans=spans,
        level_u=level_u,
        level_event_u=level_event_u,
        first_envelope_u=first_envelope_u,
        last_envelope_u=last_envelope_u,
        envelope_source=envelope_source,
        envelope_counts=envelope_counts,
        width=width,
        support_width=support_width,
        n_blocking=n_blocking,
        misassigned_events=misassigned,
        misassigned_frames=misassigned_frames,
    )
    return TerrainSpec(boxes=boxes, provenance={"source": "fit_stair_flight"}), report
