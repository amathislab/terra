"""Validate fitted terrain against the source motion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from terra._musclemimic import TerrainSpec
from terra.terrain.seats import (
    PELVIS_SEAT_OFFSET,
    SEAT_GEOM_PREFIX,
    SeatRest,
    detect_seat_rests,
    drop_seated_contacts,
)
from terra.terrain.shapes import DEFAULT_CONTACT_TOL
from terra.terrain.stance import (
    DEFAULT_CONTACT_JOINTS,
    DEFAULT_LEVEL_TOL,
    DEFAULT_STANCE_SPEED,
    StanceEvent,
    _kinematic_interval_events,
    detect_stance_events,
    joint_surface_offsets,
)


def _contact_residuals(events: Sequence[StanceEvent], ground: TerrainSpec) -> list[dict]:
    """Measure each stance probe against its local and whole-foot support surfaces."""

    # Evaluating the probe at its own median position avoids averaging two box tops
    # into a height that no surface has.
    probe_surfaces = [float(ground.height_at(*np.median(event.xy, axis=0))) for event in events]
    residuals = []
    for index, event in enumerate(events):
        # A rigid foot can span multiple probes. On steps it rests on the highest
        # overlapping surface, while on slopes each probe legitimately has a distinct
        # height; support_plane_at implements that geometric distinction.
        foot_xy = np.array(
            [
                np.median(other.xy, axis=0)
                for other in events
                if other.joint[0] == event.joint[0] and other.start < event.end and event.start < other.end
            ]
        )
        here = np.median(event.xy, axis=0)
        surface = ground.support_plane_at(foot_xy, here[0], here[1])
        residuals.append(
            {
                "joint": event.joint,
                "start": event.start,
                "end": event.end,
                "contact_z": event.z,
                "surface_z": surface,
                "residual": event.z - surface,
                "probe_surface_z": probe_surfaces[index],
            }
        )
    return residuals


def _support_baselines(
    events: Sequence[StanceEvent],
    residuals: Sequence[dict],
    offsets: dict[str, float] | None,
    compensate_sloped_offsets: bool,
    ground: TerrainSpec,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve calibrated and optional slope-compensated probe offsets."""

    baseline = joint_surface_offsets(events) if offsets is None else dict(offsets)
    support_baseline = dict(baseline)
    if not compensate_sloped_offsets or not any(box.is_sloped for box in ground.boxes):
        return baseline, support_baseline

    # Rotating a flat sole onto an incline changes its probes' vertical offsets even
    # though their normal distance to the surface is unchanged. One constant median
    # per probe compensates orientation without fitting each event independently.
    for joint in baseline:
        values = [
            residual["contact_z"] - residual["probe_surface_z"]
            for residual in residuals
            if residual["joint"] == joint and residual["probe_surface_z"] > 1e-6
        ]
        if values:
            support_baseline[joint] = float(np.median(values))
    return baseline, support_baseline


def _free_body_penetration(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    terrain: TerrainSpec,
    events: Sequence[StanceEvent],
    fps: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Measure penetration outside padded, already-scored support intervals."""

    body = joints.reshape(-1, 3)
    supported = np.zeros(joints.shape[:2], dtype=bool)
    joint_index = {name: index for index, name in enumerate(demo_joints)}
    # Contact onset is frame-quantized and speed detection waits for settlement.
    # Twenty milliseconds covers that uncertainty without hiding a swing arc.
    contact_padding = max(1, round(0.020 * fps))
    for event in events:
        if event.joint not in joint_index:
            continue
        supported[
            max(0, event.start - contact_padding) : min(len(joints), event.end + contact_padding),
            joint_index[event.joint],
        ] = True

    penetration = terrain.penetration(body)
    free_penetration = np.where(~supported.reshape(-1), penetration, 0.0)
    worst = int(np.argmax(free_penetration)) if len(free_penetration) else 0
    below_floor = float(np.minimum(body[:, 2], 0.0).min())
    return body, free_penetration, worst, below_floor


def _raised_contact_groups(
    residuals: Sequence[dict], baseline: Mapping[str, float]
) -> tuple[list[dict], list[list[int]]]:
    """Select raised contacts and group overlapping probes on the same foot."""

    # Select on measured contact height as well as fitted coverage. Selecting only
    # by surface height lets an uncovered raised contact disappear from validation.
    raised = [
        residual
        for residual in residuals
        if residual["contact_z"] - baseline.get(residual["joint"], 0.0) > DEFAULT_LEVEL_TOL
        or max(residual["surface_z"], residual["probe_surface_z"]) > 1e-6
    ]

    pending = set(range(len(raised)))
    groups: list[list[int]] = []
    while pending:
        group = {pending.pop()}
        changed = True
        while changed:
            changed = False
            for index in list(pending):
                candidate = raised[index]
                if any(
                    candidate["joint"][0] == raised[member]["joint"][0]
                    and candidate["start"] < raised[member]["end"]
                    and raised[member]["start"] < candidate["end"]
                    for member in group
                ):
                    pending.remove(index)
                    group.add(index)
                    changed = True
        groups.append(sorted(group))
    return raised, groups


def _score_raised_contacts(
    raised: Sequence[dict], groups: Sequence[Sequence[int]], support_baseline: Mapping[str, float]
) -> list[float]:
    """Choose one support hypothesis per overlapping foot interval and score it."""

    errors = []
    for group in groups:
        plane_errors = [
            abs(raised[index]["residual"] - support_baseline.get(raised[index]["joint"], 0.0)) for index in group
        ]
        local_errors = [
            abs(
                raised[index]["contact_z"]
                - raised[index]["probe_surface_z"]
                - support_baseline.get(raised[index]["joint"], 0.0)
            )
            for index in group
        ]
        plane_key = (max(plane_errors), float(np.mean(plane_errors)))
        local_key = (max(local_errors), float(np.mean(local_errors)))
        mode = "local_probes" if local_key < plane_key else "shared_plane"
        selected_errors = local_errors if mode == "local_probes" else plane_errors
        for position, (index, error) in enumerate(zip(group, selected_errors, strict=True)):
            residual = raised[index]
            residual["support_mode"] = mode
            residual["support_plane_error"] = plane_errors[position]
            residual["local_probe_error"] = local_errors[position]
            residual["selected_surface_z"] = (
                residual["probe_surface_z"] if mode == "local_probes" else residual["surface_z"]
            )
            residual["error"] = error
            residual["has_baseline"] = residual["joint"] in support_baseline
            residual["uncovered"] = residual["selected_surface_z"] <= 1e-6
            errors.append(error)
    return errors


def _seat_residuals(
    seat_rests: Sequence[SeatRest],
    terrain: TerrainSpec,
    pelvis_seat_offset: float,
    seat_support_heights: Sequence[float] | None,
) -> list[dict]:
    """Score pelvis support against fitted seat surfaces."""

    if seat_support_heights is not None:
        support = np.asarray(seat_support_heights, dtype=float)
        if support.shape != (len(seat_rests),):
            raise ValueError(
                "seat_support_heights must contain one value per seat rest; "
                f"expected {len(seat_rests)}, got shape {support.shape}"
            )
        if not np.all(np.isfinite(support)):
            raise ValueError("seat_support_heights must be finite")
    else:
        support = None

    seats = []
    for index, rest in enumerate(seat_rests):
        here = np.median(rest.xy, axis=0)
        surface = float(terrain.height_at(*here))
        expected = rest.z - pelvis_seat_offset if support is None else float(support[index])
        seats.append(
            {
                "start": rest.start,
                "end": rest.end,
                "pelvis_z": rest.z,
                "expected_surface_z": expected,
                "height_source": ("fixed_pelvis_offset" if support is None else "posed_body_surface"),
                "surface_z": surface,
                "outside": rest.outside,
                "error": abs(expected - surface),
                "uncovered": surface <= 1e-6,
            }
        )
    return seats


def validate_terrain(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    terrain: TerrainSpec,
    fps: float,
    contact_joints: Sequence[str] = DEFAULT_CONTACT_JOINTS,
    stance_speed: float = DEFAULT_STANCE_SPEED,
    contact_tol: float = DEFAULT_CONTACT_TOL,
    offsets: dict[str, float] | None = None,
    seat_rests: Sequence[SeatRest] | None = None,
    pelvis_seat_offset: float = PELVIS_SEAT_OFFSET,
    seat_support_heights: Sequence[float] | None = None,
    allow_boundary_truncation: bool = True,
    compensate_sloped_offsets: bool = True,
    _kinematic_intervals_s: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> dict:
    """Measure terrain support and body clearance for a source motion.

    Args:
        joints: Source joint positions with shape ``(n_frames, n_joints, 3)``.
        demo_joints: Joint names corresponding to the second axis of ``joints``.
        terrain: Fitted terrain to validate.
        fps: Motion frame rate in frames per second.
        contact_joints: Joint names used as contact probes.
        stance_speed: Maximum probe speed for detecting stance, in meters per second.
        contact_tol: Maximum accepted contact error and body penetration, in meters.
        offsets: Surface-relative height for each contact joint. Estimated from the
            motion when omitted.
        seat_rests: Seated-support intervals to evaluate. When omitted, intervals are
            detected only if the terrain contains a fitted seat.
        pelvis_seat_offset: Vertical pelvis-to-seat distance used during fitting, in meters.
        seat_support_heights: Optional posed body-surface height for every element
            of ``seat_rests``. When supplied, these are the expected seat surfaces.
        allow_boundary_truncation: Apply the clip-boundary duration rule when this
            function must detect its own seat rests.
        compensate_sloped_offsets: Re-estimate each probe's vertical residual on sloped
            support. This compensates for a flat sole rotating onto an incline, but must be
            disabled when ``offsets`` are an out-of-sample flat-reference calibration:
            otherwise the terrain under test can hide a common vertical placement error
            by learning it back as a new ramp-specific offset.
        _kinematic_intervals_s: Internal kinematic intervals derived by a reconstruction
            method. Production TERRA callers omit this and use the shared detector.

    Returns:
        Validation metrics and a ``passed`` flag indicating whether all errors are
        within ``contact_tol``.
    """
    joints = np.asarray(joints, dtype=float)
    events = (
        detect_stance_events(joints, demo_joints, fps, contact_joints=contact_joints, speed_ms=stance_speed)
        if _kinematic_intervals_s is None
        else _kinematic_interval_events(joints, demo_joints, fps, _kinematic_intervals_s, contact_joints=contact_joints)
    )
    if seat_rests is None:
        has_seat = any(b.name.startswith(SEAT_GEOM_PREFIX) for b in terrain.boxes)
        seat_rests = (
            detect_seat_rests(
                joints,
                demo_joints,
                fps,
                allow_boundary_truncation=allow_boundary_truncation,
            )
            if has_seat
            else ()
        )
    events, n_seated = drop_seated_contacts(events, seat_rests)

    # Feet read only walkable support; free-body penetration still reads every solid,
    # including seats.
    ground = terrain.walkable
    residuals = _contact_residuals(events, ground)
    baseline, support_baseline = _support_baselines(
        events,
        residuals,
        offsets,
        compensate_sloped_offsets,
        ground,
    )
    body, pen_free, worst, below_floor = _free_body_penetration(joints, demo_joints, terrain, events, fps)
    raised, groups = _raised_contact_groups(residuals, baseline)
    errors = _score_raised_contacts(raised, groups, support_baseline)
    seats = _seat_residuals(
        seat_rests,
        terrain,
        pelvis_seat_offset,
        seat_support_heights,
    )
    seat_errors = [s["error"] for s in seats]

    return {
        "n_contacts": len(residuals),
        "n_raised_contacts": len(raised),
        "n_uncovered_contacts": sum(1 for r in raised if r["uncovered"]),
        "n_seated_contacts_dropped": n_seated,
        "contacts": residuals,
        "ground_baseline_residual": baseline,
        "support_baseline_residual": support_baseline,
        "support_baseline_mode": ("slope_compensated" if compensate_sloped_offsets else "fixed_calibration"),
        "raised_contact_error_mean": float(np.mean(errors)) if errors else 0.0,
        "raised_contact_error_max": float(np.max(errors)) if errors else 0.0,
        "max_penetration": float(pen_free.max()) if len(pen_free) else 0.0,
        "max_penetration_point": body[worst].tolist() if len(pen_free) else None,
        "n_points_penetrating": int((pen_free > contact_tol).sum()),
        "min_body_z": below_floor,
        "seats": seats,
        "n_seat_rests": len(seats),
        "n_uncovered_seat_rests": sum(1 for s in seats if s["uncovered"]),
        "seat_contact_error_max": float(np.max(seat_errors)) if seat_errors else 0.0,
        "passed": (
            (float(np.max(errors)) if errors else 0.0) <= contact_tol
            and (float(pen_free.max()) if len(pen_free) else 0.0) <= contact_tol
            and (float(np.max(seat_errors)) if seat_errors else 0.0) <= contact_tol
        ),
    }
