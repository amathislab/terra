"""Fit continuous incline terrain from stance-event positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from terra._musclemimic import BoxSpec, TerrainSpec
from terra.terrain.shapes import DEFAULT_CONTACT_MARGIN, DEFAULT_FREE_SPACE_TOL
from terra.terrain.stairs import STAIR_MIN_BLOCKING_POINTS, STAIR_SOLE_HALF
from terra.terrain.stance import StanceEvent

#: Accepted fitted ramp slopes in degrees.
RAMP_SLOPE_RANGE = (2.0, 40.0)


#: Minimum accepted ramp rise in meters.
RAMP_MIN_RISE = 0.10


#: Minimum accepted horizontal ramp length in meters when the gradient must also be
#: inferred from the footfall profile.  This conservative span prevents a short local
#: height transition from being promoted to a continuous incline.
RAMP_MIN_LENGTH = 0.80


#: Minimum length when supported-foot orientation independently supplies the gradient.
#: Forty centimetres exceeds a full adult support footprint plus both contact margins,
#: while allowing a short source clip that does not expose a full stride-length incline.
RAMP_ORIENTED_MIN_LENGTH = 0.40


#: Maximum RMS footfall residual for a ramp profile, in meters.
RAMP_MAX_PROFILE_RMS = 0.016


#: Maximum footfall distance from the ramp axis in meters.
RAMP_MAX_LATERAL = 0.60


#: Preferred total ramp width before free-space clipping, in meters.
RAMP_WIDTH_PRIOR = 1.00


#: Additional depth below the floor at the low end of a ramp, in meters.
RAMP_BURIED_DEPTH = 0.05


def _ramp_profile(
    u: np.ndarray,
    z: np.ndarray,
    step: float = 0.02,
    fixed_slope: float | None = None,
    min_length: float = RAMP_MIN_LENGTH,
    search_margin: float = 0.30,
) -> tuple[float, float, float]:
    """Fit a flat-incline-flat height profile along one axis.

    Args:
        u: Footfall positions along the ramp axis, in meters.
        z: Footfall heights above the lowest event, in meters.
        step: Breakpoint-search grid spacing in meters.
        fixed_slope: Optional independently measured positive gradient.
        min_length: Minimum horizontal incline span in meters.
        search_margin: Distance beyond the observed support span searched for landings.

    Returns:
        Incline start, incline end, and fitted gradient. The gradient is zero
        when no rising profile is found.
    """
    grid = np.arange(u.min() - search_margin, u.max() + search_margin + 1e-9, step)
    best = (np.inf, 0.0, 0.0, 0.0)
    for i, u0 in enumerate(grid):
        for u1 in grid[i + 1 :]:
            if u1 - u0 < min_length:
                continue
            x = np.clip(u, u0, u1) - u0
            xx = float(x @ x)
            if xx <= 0.0:
                continue
            s = float(fixed_slope) if fixed_slope is not None else float(x @ z / xx)
            if s <= 0.0:
                continue
            r = z - s * x
            sse = float(r @ r)
            equal_fit = abs(sse - best[0]) <= 1e-12
            shorter = (u1 - u0) < (best[2] - best[1])
            if sse < best[0] - 1e-12 or (equal_fit and shorter):
                best = (sse, float(u0), float(u1), s)
    return best[1], best[2], best[3]


def _ramp_profile_with_joint_offsets(
    u: np.ndarray,
    z: np.ndarray,
    joints: Sequence[str],
    step: float = 0.02,
    fixed_slope: float | None = None,
    min_length: float = RAMP_MIN_LENGTH,
) -> tuple[float, float, float, dict[str, float], np.ndarray]:
    """Fit a flat-incline-flat profile with one anatomical offset per probe.

    A short clip need not show every contact probe on the lower landing.  Estimating a
    probe's offset from its lowest observed event then mistakes terrain height for anatomy:
    an ankle first seen 10 cm up an incline receives an offset 10 cm too large.  The ramp
    slope is identifiable from *changes within the same probe*, so this fit treats the four
    ankle/toe offsets as nuisance intercepts and estimates them jointly with the shared
    surface profile.

    Every breakpoint candidate has exactly the same nuisance parameters.  They therefore
    cannot make one terrain family more flexible than another, and no dataset-specific
    marker dimensions are assumed.

    Args:
        u: Event positions along the candidate ramp axis, in meters.
        z: Raw world-space event heights, in meters.
        joints: Contact-probe name for each event.
        step: Breakpoint-search grid spacing in meters.

    Returns:
        Incline start, incline end, gradient, fitted probe intercepts, and residuals.
        The gradient is zero when the repeated-probe observations do not identify a rising
        profile.
    """

    u = np.asarray(u, dtype=float)
    z = np.asarray(z, dtype=float)
    joint_names = np.asarray(joints, dtype=str)
    # Bilateral counterparts are the same anatomical probe class.  Keeping four free
    # intercepts makes terrain height and anatomy non-identifiable when, for example, the
    # left ankle appears only on lower treads and the right ankle only on upper ones.  The
    # rest of the terrain pipeline already enforces this symmetry in
    # ``paired_sole_offsets``; use the same physical constraint during profile fitting.
    classes = np.asarray([name.split("_", 1)[1] if name[:2] in {"L_", "R_"} else name for name in joint_names])
    labels, group = np.unique(classes, return_inverse=True)
    grid = np.arange(u.min() - 0.30, u.max() + 0.30 + 1e-9, step)
    best = (np.inf, 0.0, 0.0, 0.0, np.zeros(len(labels)), np.zeros_like(z))

    # Eliminate the nuisance intercepts analytically.  For a candidate profile coordinate
    # x, centring x and z within each probe leaves z_c = slope * x_c.  This is the fixed-
    # effects least-squares estimator, but avoids solving a matrix at every grid point.
    z_mean = np.array([np.mean(z[group == i]) for i in range(len(labels))])
    z_centre = z - z_mean[group]
    for i, u0 in enumerate(grid):
        for u1 in grid[i + 1 :]:
            if u1 - u0 < min_length:
                continue
            x = np.clip(u, u0, u1) - u0
            x_mean = np.array([np.mean(x[group == j]) for j in range(len(labels))])
            x_centre = x - x_mean[group]
            xx = float(x_centre @ x_centre)
            if xx <= 1e-12:
                continue
            slope = float(fixed_slope) if fixed_slope is not None else float(x_centre @ z_centre / xx)
            if slope <= 0.0:
                continue
            intercept = z_mean - slope * x_mean
            residual = z - intercept[group] - slope * x
            sse = float(residual @ residual)
            # A landing with no sampled contact produces a continuum of numerically equal
            # breakpoint fits.  Floating-point noise must not decide how much unsupported
            # terrain is invented.  Among equal fits, choose the shortest incline: the
            # minimum-support/Occam solution also minimizes avoidable body penetration.
            equal_fit = abs(sse - best[0]) <= 1e-12
            shorter = (u1 - u0) < (best[2] - best[1])
            if sse < best[0] - 1e-12 or (equal_fit and shorter):
                best = (sse, float(u0), float(u1), slope, intercept, residual)

    return (
        best[1],
        best[2],
        best[3],
        {str(name): float(best[4][group[index]]) for index, name in enumerate(joint_names)},
        np.asarray(best[5], dtype=float),
    )


@dataclass(frozen=True)
class _RampEvidence:
    """Footfall measurements expressed in a height-increasing ramp frame."""

    raw_z: np.ndarray
    offset_z: np.ndarray
    offset_datum: float
    legacy_z: np.ndarray
    origin: np.ndarray
    axis: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray
    yaw: float


def _collect_ramp_evidence(
    events: Sequence[StanceEvent],
    offsets: Mapping[str, float],
    fixed_joint_offsets: bool,
) -> _RampEvidence:
    """Project footfalls onto the principal path axis and orient it uphill."""
    xy = np.array([np.median(event.xy, axis=0) for event in events])
    raw_z = np.array([event.z for event in events])
    offset_z = np.array([event.z - offsets.get(event.joint, 0.0) for event in events])
    offset_datum = float(offset_z.min())
    legacy_z = offset_z - offset_datum

    origin = xy.mean(axis=0)
    centred_xy = xy - origin
    _, _, right_singular_vectors = np.linalg.svd(centred_xy, full_matrices=False)
    axis = right_singular_vectors[0]
    longitudinal = centred_xy @ axis
    lateral = centred_xy @ np.array([-axis[1], axis[0]])
    height_signal = offset_z if fixed_joint_offsets else raw_z
    if float(np.cov(longitudinal, height_signal)[0, 1]) < 0:
        axis = -axis
        longitudinal = -longitudinal
        lateral = -lateral
    return _RampEvidence(
        raw_z=raw_z,
        offset_z=offset_z,
        offset_datum=offset_datum,
        legacy_z=legacy_z,
        origin=origin,
        axis=axis,
        longitudinal=longitudinal,
        lateral=lateral,
        yaw=float(np.arctan2(axis[1], axis[0])),
    )


def _motion_envelope(
    evidence: _RampEvidence,
    events: Sequence[StanceEvent],
    motion_foot_xy: np.ndarray | None,
    motion_foot_xyz: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, str]:
    """Return the longitudinal span that the final ramp and landing must support."""
    if motion_foot_xy is None and motion_foot_xyz:
        envelope_xy = np.concatenate(
            [np.asarray(points, dtype=float)[:, :2] for points in motion_foot_xyz.values()],
            axis=0,
        )
        source = "full_motion_foot_landmarks"
    elif motion_foot_xy is None:
        envelope_xy = np.concatenate([event.xy for event in events], axis=0)
        source = "stance_events"
    else:
        envelope_xy = np.asarray(motion_foot_xy, dtype=float)
        if envelope_xy.ndim != 2 or envelope_xy.shape[1] != 2 or not len(envelope_xy):
            raise ValueError("motion_foot_xy must have non-empty shape (N, 2)")
        if not np.isfinite(envelope_xy).all():
            raise ValueError("motion_foot_xy must be finite")
        source = "full_motion_foot_landmarks"
    return (envelope_xy - evidence.origin) @ evidence.axis, source


def _fit_surface_profile(
    evidence: _RampEvidence,
    events: Sequence[StanceEvent],
    offsets: dict[str, float],
    fixed_slope: float | None,
    min_length: float,
    fixed_joint_offsets: bool,
) -> tuple[float, float, float, dict[str, float], np.ndarray]:
    """Fit the shared incline and anatomical probe offsets."""
    if fixed_joint_offsets:
        u0, u1, slope = _ramp_profile(
            evidence.longitudinal,
            evidence.offset_z,
            fixed_slope=fixed_slope,
            min_length=min_length,
            # A clipped descent may omit the lower landing. External offsets identify
            # absolute surface height, so the search may safely extend farther downhill.
            search_margin=0.60,
        )
        residual = evidence.offset_z - slope * (np.clip(evidence.longitudinal, u0, u1) - u0)
        return u0, u1, slope, dict(offsets), residual

    u0, u1, slope, fitted_offsets, residual = _ramp_profile_with_joint_offsets(
        evidence.longitudinal,
        evidence.raw_z,
        [event.joint for event in events],
        fixed_slope=fixed_slope,
        min_length=min_length,
    )
    return u0, u1, slope, fitted_offsets, residual


def _extend_terminal_incline(
    evidence: _RampEvidence,
    u0: float,
    u1: float,
    slope: float,
    fitted_offsets: Mapping[str, float],
    residual: np.ndarray,
    free_space_tol: float,
    motion_foot_xyz: Mapping[str, np.ndarray] | None,
    events: Sequence[StanceEvent],
    report: dict,
) -> tuple[float, np.ndarray]:
    """Extend a clipped incline when both terminal foot probes support its plane."""
    initial_u1 = u1
    terminal_evidence = {}
    if motion_foot_xyz:
        for side in ("L", "R"):
            pair = (f"{side}_Toe", f"{side}_Ankle")
            if not all(name in motion_foot_xyz and name in fitted_offsets for name in pair):
                continue
            probes = []
            for name in pair:
                points = np.asarray(motion_foot_xyz[name], dtype=float)
                tail = points[-min(5, len(points)) :]
                point = np.median(tail, axis=0)
                terminal_u = float((point[:2] - evidence.origin) @ evidence.axis)
                residual_m = float(point[2] - fitted_offsets[name] - slope * (terminal_u - u0))
                tail_u_max = float(np.max((tail[:, :2] - evidence.origin) @ evidence.axis))
                probes.append((name, terminal_u, residual_m, tail_u_max))
            matches = all(
                terminal_u > initial_u1 + 1e-3 and abs(residual_m) <= free_space_tol
                for _name, terminal_u, residual_m, _tail_u_max in probes
            )
            terminal_evidence[side] = {
                "matches_extrapolated_ramp": matches,
                "probes": {
                    name: {"u": terminal_u, "residual_m": residual_m}
                    for name, terminal_u, residual_m, _tail_u_max in probes
                },
            }
            if matches:
                u1 = max(u1, max(probe[3] for probe in probes) + STAIR_SOLE_HALF[0])

    if u1 <= initial_u1 + 1e-9:
        return u1, residual
    report["terminal_incline_extension"] = {
        "initial_u1": float(initial_u1),
        "extended_u1": float(u1),
        "evidence": terminal_evidence,
    }
    profile_z = evidence.raw_z - np.array([fitted_offsets[event.joint] for event in events])
    residual = profile_z - slope * (np.clip(evidence.longitudinal, u0, u1) - u0)
    return u1, residual


def _summarize_profile(
    evidence: _RampEvidence,
    u0: float,
    u1: float,
    slope: float,
    residual: np.ndarray,
    fitted_offsets: Mapping[str, float],
    fixed_joint_offsets: bool,
    slope_hint_deg: float | None,
    envelope_u: np.ndarray,
    envelope_source: str,
    report: dict,
) -> tuple[float, float]:
    """Record the fitted incline and its legacy-comparable diagnostics."""
    rise = slope * (u1 - u0)
    length = u1 - u0
    legacy_u0, legacy_u1, legacy_slope = _ramp_profile(
        evidence.longitudinal,
        evidence.legacy_z,
    )
    legacy_residual = evidence.legacy_z - legacy_slope * (
        np.clip(evidence.longitudinal, legacy_u0, legacy_u1) - legacy_u0
    )
    on_incline = np.sort(evidence.longitudinal[(evidence.longitudinal >= u0) & (evidence.longitudinal <= u1)])
    edges = np.concatenate([[u0], on_incline, [u1]])
    report.update(
        yaw=evidence.yaw,
        origin_xy=evidence.origin.tolist(),
        u0=u0,
        u1=u1,
        slope_deg=float(np.degrees(np.arctan(slope))),
        rise=float(rise),
        length=float(length),
        profile_rms=float(np.sqrt(np.mean(residual**2))),
        profile_max=float(np.abs(residual).max()),
        lateral_spread=float(np.abs(evidence.lateral).max()),
        n_footfalls_on_incline=len(on_incline),
        u_max=float(evidence.longitudinal.max()),
        largest_unsampled_gap=(float(np.max(np.diff(edges))) if len(edges) > 1 else float(length)),
        profile_joint_offsets=fitted_offsets,
        profile_joint_offsets_fixed=bool(fixed_joint_offsets),
        profile_surface_datum=float(evidence.offset_datum),
        legacy_profile_rms=float(np.sqrt(np.mean(legacy_residual**2))),
        slope_source=("supported_foot_orientation" if slope_hint_deg is not None else "footfalls"),
        motion_foot_span_u=(float(envelope_u.min()), float(envelope_u.max())),
        landing_extent_source=envelope_source,
    )
    return rise, length


def _rejection_reason(
    report: Mapping[str, float],
    rise: float,
    length: float,
    min_length: float,
    max_profile_rms: float | None,
) -> str | None:
    """Return the first failed geometric candidacy gate, if any."""
    lo_deg, hi_deg = RAMP_SLOPE_RANGE
    profile_limit = RAMP_MAX_PROFILE_RMS if max_profile_rms is None else float(max_profile_rms)
    conditions = (
        (
            lo_deg <= report["slope_deg"] <= hi_deg,
            f"slope {report['slope_deg']:.1f} deg outside {lo_deg:.0f}-{hi_deg:.0f} deg",
        ),
        (
            rise >= RAMP_MIN_RISE,
            f"rise {rise * 1000:.0f} mm below {RAMP_MIN_RISE * 1000:.0f} mm",
        ),
        (length >= min_length, f"length {length:.2f} m below {min_length:.2f} m"),
        (
            report["profile_rms"] <= profile_limit,
            f"footfalls miss the profile by {report['profile_rms'] * 1000:.1f} mm RMS, over the "
            f"{profile_limit * 1000:.0f} mm a ramp fits within; these are steps",
        ),
        (
            report["lateral_spread"] <= RAMP_MAX_LATERAL,
            f"footfalls spread {report['lateral_spread']:.2f} m off the axis, over "
            f"{RAMP_MAX_LATERAL:.2f} m; one pitched box is one straight incline",
        ),
    )
    return next((reason for accepted, reason in conditions if not accepted), None)


def _fit_ramp_width(
    joints: np.ndarray,
    evidence: _RampEvidence,
    u0: float,
    u1: float,
    slope: float,
    free_space_tol: float,
    width_prior: float,
    clearance: float,
    report: dict,
) -> float:
    """Fit the support width and clip it against independently blocking body points."""
    support_width = float(np.abs(evidence.lateral).max()) + STAIR_SOLE_HALF[1]
    body = joints.reshape(-1, 3)
    body_delta = body[:, :2] - evidence.origin
    body_u = body_delta @ evidence.axis
    body_v = body_delta @ np.array([-evidence.axis[1], evidence.axis[0]])
    surface = slope * (np.clip(body_u, u0, u1) - u0)
    inside = (
        (body_u >= u0 - STAIR_SOLE_HALF[0])
        & (body_u <= evidence.longitudinal.max() + STAIR_SOLE_HALF[0])
        & (body[:, 2] < surface - free_space_tol)
        & (np.abs(body_v) > support_width)
    )
    limit = np.inf
    if int(inside.sum()) >= STAIR_MIN_BLOCKING_POINTS:
        limit = float(np.abs(body_v[inside]).min()) - clearance
    asked = max(support_width, 0.5 * width_prior)
    width = asked if not np.isfinite(limit) else max(min(asked, limit), support_width)
    report.update(
        width=float(width),
        support_width=float(support_width),
        n_blocking=int(inside.sum()),
    )
    return width


def _build_ramp_terrain(
    evidence: _RampEvidence,
    envelope_u: np.ndarray,
    u0: float,
    u1: float,
    slope: float,
    rise: float,
    length: float,
    width: float,
    contact_margin: tuple[float, float],
    report: dict,
) -> TerrainSpec:
    """Materialize the pitched incline and its non-overlapping top landing."""
    pitch = -float(np.arctan(slope))
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    u_half = 0.5 * length / cos_pitch
    z_half = 0.5 * rise / cos_pitch + RAMP_BURIED_DEPTH
    midpoint = evidence.origin + evidence.axis * (0.5 * (u0 + u1))
    normal = np.array(
        [
            np.cos(evidence.yaw) * sin_pitch,
            np.sin(evidence.yaw) * sin_pitch,
            cos_pitch,
        ]
    )
    centre = np.array([midpoint[0], midpoint[1], 0.5 * rise]) - z_half * normal
    boxes = [
        BoxSpec(
            pos=tuple(centre),
            size=(u_half, width, z_half),
            yaw=evidence.yaw,
            pitch=pitch,
            name="terrain_box_ramp",
        )
    ]

    far = max(
        float(envelope_u.max()) + STAIR_SOLE_HALF[0] + contact_margin[0],
        float(u1) + STAIR_SOLE_HALF[0] + contact_margin[0],
    )
    if far > u1 + 1e-3:
        mid_u = 0.5 * (u1 + far)
        centre_xy = evidence.origin + evidence.axis * mid_u
        boxes.append(
            BoxSpec(
                pos=(float(centre_xy[0]), float(centre_xy[1]), rise / 2),
                size=(0.5 * (far - u1), width, rise / 2),
                yaw=evidence.yaw,
                name="terrain_box_ramp_landing",
            )
        )
        report["landing_span"] = (float(u1), far)
    return TerrainSpec(boxes=tuple(boxes), provenance={"source": "fit_ramp"})


def fit_ramp(
    joints: np.ndarray,
    events: Sequence[StanceEvent],
    offsets: dict[str, float],
    free_space_tol: float = DEFAULT_FREE_SPACE_TOL,
    width_prior: float = RAMP_WIDTH_PRIOR,
    clearance: float = 0.02,
    contact_margin: tuple[float, float] = DEFAULT_CONTACT_MARGIN,
    max_profile_rms: float | None = None,
    slope_hint_deg: float | None = None,
    min_length: float = RAMP_MIN_LENGTH,
    fixed_joint_offsets: bool = False,
    min_events: int = 4,
    motion_foot_xy: np.ndarray | None = None,
    motion_foot_xyz: Mapping[str, np.ndarray] | None = None,
) -> tuple[TerrainSpec | None, dict]:
    """Fit stance events as one straight continuous incline.

    Args:
        joints: Solver-space source joints with shape ``(T, J, 3)``.
        events: Stance events, including floor contacts.
        offsets: Per-joint vertical offsets in meters.
        free_space_tol: Vertical tolerance for blocking body points, in meters.
        width_prior: Preferred total ramp width in meters.
        clearance: Horizontal margin from blocking points in meters.
        contact_margin: Longitudinal and lateral contact margins in meters.
        max_profile_rms: Maximum footfall-profile residual in meters. ``None``
            uses :data:`RAMP_MAX_PROFILE_RMS`; callers with independent surface-family
            evidence may relax this candidacy gate without changing the geometric gates.
        slope_hint_deg: Optional incline angle independently measured from supported-foot
            orientation.  The breakpoint profile is then fitted at that fixed gradient.
        min_length: Minimum horizontal incline span.  The default remains conservative;
            callers with an independent orientation measurement may use
            :data:`RAMP_ORIENTED_MIN_LENGTH`.
        fixed_joint_offsets: Keep ``offsets`` fixed instead of estimating new anatomical
            probe intercepts from this ramp. This must be enabled when the offsets came
            from a separate flat reference; otherwise a common surface-height error can
            be absorbed into every fitted intercept and move the ramp away from the sole.
        min_events: Minimum support events required. Three is accepted only by callers
            that have independent multi-probe evidence for the continuous-incline family;
            the general fit keeps the conservative four-event default.
        motion_foot_xy: Optional horizontal positions of all foot landmarks throughout
            the recorded clip. These do not affect the ramp angle or family fit; they only
            ensure that the terminal landing supports motion after the last complete stance.
        motion_foot_xyz: Optional per-landmark trajectories used to recognize a terminal
            step censored by the end of the recording. The incline is extended only when
            both landmarks of that foot agree with its extrapolated surface.

    Returns:
        A ramp terrain or ``None``, plus a report containing fit measurements and
        any rejection reason.
    """
    if min_events < 3:
        raise ValueError("min_events must be at least 3")
    report: dict = {"model": "ramp", "n_events": len(events)}
    if len(events) < min_events:
        return None, {
            **report,
            "rejected": f"fewer than {min_events} stance events",
        }

    evidence = _collect_ramp_evidence(events, offsets, fixed_joint_offsets)
    envelope_u, envelope_source = _motion_envelope(
        evidence,
        events,
        motion_foot_xy,
        motion_foot_xyz,
    )
    fixed_slope = float(np.tan(np.radians(slope_hint_deg))) if slope_hint_deg is not None else None
    u0, u1, slope, fitted_offsets, residual = _fit_surface_profile(
        evidence,
        events,
        offsets,
        fixed_slope,
        min_length,
        fixed_joint_offsets,
    )
    if slope <= 0.0:
        return None, {**report, "rejected": "no rising profile fits these footfalls"}

    u1, residual = _extend_terminal_incline(
        evidence,
        u0,
        u1,
        slope,
        fitted_offsets,
        residual,
        free_space_tol,
        motion_foot_xyz,
        events,
        report,
    )
    rise, length = _summarize_profile(
        evidence,
        u0,
        u1,
        slope,
        residual,
        fitted_offsets,
        fixed_joint_offsets,
        slope_hint_deg,
        envelope_u,
        envelope_source,
        report,
    )
    rejected = _rejection_reason(report, rise, length, min_length, max_profile_rms)
    if rejected is not None:
        return None, {**report, "rejected": rejected}

    width = _fit_ramp_width(
        joints,
        evidence,
        u0,
        u1,
        slope,
        free_space_tol,
        width_prior,
        clearance,
        report,
    )
    terrain = _build_ramp_terrain(
        evidence,
        envelope_u,
        u0,
        u1,
        slope,
        rise,
        length,
        width,
        contact_margin,
        report,
    )
    return terrain, report
