"""Fit per-level, stair, ramp, and seat terrain models to a motion."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from terra._musclemimic import BoxSpec, TerrainSpec
from terra.terrain.family import FAMILY_EVIDENCE_MODES, FAMILY_EVIDENCE_PHYSICAL, classify_terrain_family
from terra.terrain.ramps import (
    RAMP_ORIENTED_MIN_LENGTH,
    RAMP_WIDTH_PRIOR,
    fit_ramp,
)
from terra.terrain.seats import (
    PELVIS_SEAT_OFFSET,
    SEAT_GEOM_PREFIX,
    SeatRest,
    detect_seat_rests,
    drop_seated_contacts,
    fit_seat,
)
from terra.terrain.shapes import (
    CONFLICT_REJECT_MAX_HEIGHT,
    CONFLICT_REJECT_MIN_RATIO,
    DEFAULT_AMBIGUOUS_SPLIT_GAP,
    DEFAULT_CONTACT_MARGIN,
    DEFAULT_EXCLUDE_CLAIMED,
    DEFAULT_FREE_SPACE_TOL,
    DEFAULT_MAX_EXTENSION,
    DEFAULT_SPLIT_GAP,
    _as_pair,
    _drop_buried_boxes,
    _exclude_claimed_points,
    _free_space_limit,
    _principal_yaw,
    _split_by_gap,
)
from terra.terrain.stairs import STAIR_MIN_LEVELS, STAIR_WIDTH_PRIOR, fit_stair_flight
from terra.terrain.stance import (
    DEFAULT_CONTACT_JOINTS,
    DEFAULT_LEVEL_TOL,
    DEFAULT_STANCE_SPEED,
    WEAK_LEVEL_MERGE_GAP,
    StanceEvent,
    _accept_levels,
    _level_height,
    cluster_levels,
    detect_stance_events,
    joint_surface_offsets,
    paired_sole_offsets,
)
from terra.terrain.validation import validate_terrain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FitEvidence:
    """Measurements and configuration shared by the terrain candidate fits."""

    joints: np.ndarray
    demo_joints: tuple[str, ...]
    fps: float
    contact_joints: tuple[str, ...]
    stance_speed: float
    free_space_tol: float
    use_free_space_evidence: bool
    contact_margin: tuple[float, float]
    max_extension: tuple[float, float]
    clearance: float
    exclude_claimed: bool
    pelvis_seat_offset: float
    seat_support_heights: tuple[float, ...] | None
    events: list[StanceEvent]
    family_events: list[StanceEvent]
    rests: list[SeatRest]
    n_seated_contacts: int
    offsets: dict[str, float]
    offsets_source: str
    levels: list[list[StanceEvent]]
    motion_foot_xy: np.ndarray | None
    motion_foot_xyz: dict[str, np.ndarray]


@dataclass(frozen=True)
class _LevelGeometry:
    """Horizontal coordinates and reporting state for one support level."""

    index: int
    events: list[StanceEvent]
    entry: dict
    point_event: np.ndarray
    yaw: float
    centre_xy: np.ndarray
    cos_neg_yaw: float
    sin_neg_yaw: float
    longitudinal: np.ndarray
    lateral: np.ndarray


def _validate_motion_input(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    """Normalize one joint motion and reject ambiguous array/name contracts."""
    try:
        motion = np.asarray(joints, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("joints must be a real numeric array with shape (T, J, 3)") from exc
    if motion.ndim != 3 or motion.shape[2] != 3 or motion.shape[0] < 2:
        raise ValueError(f"joints must have shape (T>=2, J, 3), got {motion.shape}")
    if not np.all(np.isfinite(motion)):
        raise ValueError("joints must contain only finite values")

    names = tuple(demo_joints)
    if len(names) != motion.shape[1]:
        raise ValueError(f"demo_joints has {len(names)} names for a joint array with {motion.shape[1]} columns")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("demo_joints must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("demo_joints must not contain duplicate names")

    try:
        frame_rate = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fps must be positive and finite, got {fps!r}") from exc
    if not np.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps!r}")
    return motion, names, frame_rate


def _validate_model_mode(name: str, value: str) -> None:
    if value not in ("auto", "off"):
        raise ValueError(f"{name} must be 'auto' or 'off', got {value!r}")


def _nonnegative_pair(value: object, name: str) -> tuple[float, float]:
    try:
        pair = _as_pair(value, name)
    except TypeError as exc:
        raise ValueError(f"{name} must be a scalar or two-value sequence, got {value!r}") from exc
    if not all(np.isfinite(component) and component >= 0.0 for component in pair):
        raise ValueError(f"{name} must contain finite non-negative values, got {value!r}")
    return pair


def _collect_fit_evidence(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    stance_speed: float,
    level_tol: float,
    free_space_tol: float,
    use_free_space_evidence: bool,
    contact_margin: tuple[float, float],
    max_extension: tuple[float, float],
    clearance: float,
    exclude_claimed: bool,
    seat: str,
    pelvis_seat_offset: float,
    calibrated_joint_offsets: Mapping[str, float] | None,
    calibrated_joint_offsets_source: str | None,
    seat_support_heights: Sequence[float] | None,
    allow_boundary_truncated_support: bool,
) -> _FitEvidence:
    """Collect stance, seat, and anatomical evidence used by every candidate."""
    names = list(demo_joints)
    foot_indices = [names.index(name) for name in DEFAULT_CONTACT_JOINTS if name in names]
    motion_foot_xyz = {name: joints[:, names.index(name), :] for name in DEFAULT_CONTACT_JOINTS if name in names}
    motion_foot_xy = joints[:, foot_indices, :2].reshape(-1, 2) if foot_indices else None

    events = detect_stance_events(
        joints,
        demo_joints,
        fps,
        contact_joints=DEFAULT_CONTACT_JOINTS,
        speed_ms=stance_speed,
    )
    family_events = events

    rests = (
        detect_seat_rests(
            joints,
            demo_joints,
            fps,
            allow_boundary_truncation=allow_boundary_truncated_support,
        )
        if seat == "auto"
        else []
    )
    support_heights: tuple[float, ...] | None = None
    if seat_support_heights is not None:
        values = np.asarray(seat_support_heights, dtype=float)
        if values.shape != (len(rests),):
            raise ValueError(
                "seat_support_heights must contain one value per detected seat rest; "
                f"expected {len(rests)}, got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("seat_support_heights must be finite")
        support_heights = tuple(float(value) for value in values)
    events, n_seated_contacts = drop_seated_contacts(events, rests)

    if calibrated_joint_offsets is None:
        offsets = paired_sole_offsets(joint_surface_offsets(events, tol=level_tol))
        offsets_source = "motion"
    else:
        offsets = {str(name): float(value) for name, value in calibrated_joint_offsets.items()}
        missing = sorted(set(DEFAULT_CONTACT_JOINTS) - set(offsets))
        if missing:
            raise ValueError(f"calibrated_joint_offsets is missing {missing}")
        if not all(np.isfinite(value) for value in offsets.values()):
            raise ValueError("calibrated_joint_offsets must be finite")
        offsets_source = calibrated_joint_offsets_source or "flat_reference"

    return _FitEvidence(
        joints=joints,
        demo_joints=tuple(demo_joints),
        fps=fps,
        contact_joints=DEFAULT_CONTACT_JOINTS,
        stance_speed=stance_speed,
        free_space_tol=free_space_tol,
        use_free_space_evidence=use_free_space_evidence,
        contact_margin=contact_margin,
        max_extension=max_extension,
        clearance=clearance,
        exclude_claimed=exclude_claimed,
        pelvis_seat_offset=pelvis_seat_offset,
        seat_support_heights=support_heights,
        events=events,
        family_events=family_events,
        rests=rests,
        n_seated_contacts=n_seated_contacts,
        offsets=offsets,
        offsets_source=offsets_source,
        levels=cluster_levels(events, tol=level_tol, offsets=offsets),
        motion_foot_xy=motion_foot_xy,
        motion_foot_xyz=motion_foot_xyz,
    )


def _build_report(evidence: _FitEvidence) -> tuple[dict, dict]:
    """Create the mutable fit report and persisted terrain metadata."""
    report: dict = {
        "n_frames": int(evidence.joints.shape[0]),
        "n_stance_events": len(evidence.events),
        "n_family_evidence_events": len(evidence.family_events),
        "n_levels": len(evidence.levels),
        "levels": [],
        "free_space_tol": evidence.free_space_tol,
        "use_free_space_evidence": evidence.use_free_space_evidence,
        "warnings": [],
        "joint_offsets_source": evidence.offsets_source,
        "stance_timing_source": "kinematic",
    }
    if evidence.seat_support_heights is not None:
        report["seat_support_heights"] = list(evidence.seat_support_heights)
    provenance = {
        "source": "fit_terrain_from_motion",
        "n_frames": int(evidence.joints.shape[0]),
        "fps": float(evidence.fps),
        "stance_speed": evidence.stance_speed,
        "free_space_tol": evidence.free_space_tol,
        "use_free_space_evidence": evidence.use_free_space_evidence,
        "contact_margin": evidence.contact_margin,
        "max_extension": list(evidence.max_extension),
        "joint_offsets_source": evidence.offsets_source,
        "stance_timing_source": "kinematic",
        "seat_height_source": (
            "posed_body_surface" if evidence.seat_support_heights is not None else "fixed_pelvis_offset"
        ),
        "joint_surface_offsets_m": {str(name): float(value) for name, value in evidence.offsets.items()},
    }
    if evidence.seat_support_heights is not None:
        provenance["seat_support_heights_m"] = list(evidence.seat_support_heights)
    return report, provenance


def _fit_seat_candidate(evidence: _FitEvidence, seat: str, report: dict) -> list[BoxSpec]:
    """Fit seat boxes independently of the competing foot-support models."""
    if seat == "off":
        report["seat"] = {
            "mode": "off",
            "n_rests": 0,
            "seats": [],
            "warnings": [],
            "n_seated_contacts_dropped": 0,
            "pelvis_offset": evidence.pelvis_seat_offset,
        }
        report["n_seats"] = 0
        return []
    seat_boxes, seat_report = fit_seat(
        evidence.joints if evidence.use_free_space_evidence else evidence.joints[:0],
        evidence.demo_joints,
        evidence.rests,
        stance_events=evidence.events if evidence.use_free_space_evidence else (),
        pelvis_offset=evidence.pelvis_seat_offset,
        free_space_tol=evidence.free_space_tol,
        clearance=evidence.clearance,
        support_heights=evidence.seat_support_heights,
    )
    seat_report["n_seated_contacts_dropped"] = evidence.n_seated_contacts
    seat_report["pelvis_offset"] = evidence.pelvis_seat_offset
    seat_report["height_source"] = (
        "posed_body_surface" if evidence.seat_support_heights is not None else "fixed_pelvis_offset"
    )
    seat_report["mode"] = "auto"
    report["seat"] = seat_report
    report["warnings"].extend(seat_report["warnings"])
    report["n_seats"] = len(seat_boxes)
    return seat_boxes


def _finish_terrain(
    boxes: Sequence[BoxSpec],
    seat_boxes: Sequence[BoxSpec],
    report: dict,
    provenance: dict,
    model: str,
    **extra,
) -> TerrainSpec:
    """Combine a selected foot-support model with uniquely named seat boxes."""
    seats = [
        BoxSpec(
            pos=box.pos,
            size=box.size,
            yaw=box.yaw,
            pitch=box.pitch,
            name=f"{SEAT_GEOM_PREFIX}_{index}",
        )
        for index, box in enumerate(seat_boxes)
    ]
    report["model"] = model
    return TerrainSpec(
        boxes=tuple(boxes) + tuple(seats),
        provenance={**provenance, "model": model, "n_seats": len(seats), **extra},
    )


def _fit_ramp_geometry(
    evidence: _FitEvidence,
    events: Sequence[StanceEvent],
    width_prior: float,
    **overrides,
) -> tuple[TerrainSpec | None, dict]:
    """Fit one ramp geometry using the common physical evidence."""
    options = {
        "free_space_tol": evidence.free_space_tol,
        "width_prior": width_prior,
        "clearance": evidence.clearance,
        "contact_margin": evidence.contact_margin,
        "max_profile_rms": float("inf"),
        "fixed_joint_offsets": evidence.offsets_source != "motion",
        "motion_foot_xy": evidence.motion_foot_xy,
        "motion_foot_xyz": evidence.motion_foot_xyz,
        **overrides,
    }
    joints = evidence.joints if evidence.use_free_space_evidence else evidence.joints[:0]
    return fit_ramp(joints, events, evidence.offsets, **options)


def _classify_short_ramp(
    evidence: _FitEvidence,
    ramp_terrain: TerrainSpec | None,
    ramp_report: dict,
    width_prior: float,
    neutral_foot_pitch: Mapping[str, float] | None,
    family_evidence_mode: str,
) -> tuple[TerrainSpec | None, dict, dict | None]:
    """Use extra orientation probes to classify a three-contact annotated clip."""
    if not (
        ramp_report.get("slope_deg") is None
        and 2 <= len(evidence.events) <= 3
        and len(evidence.family_events) >= 4
        and evidence.family_events is not evidence.events
    ):
        return ramp_terrain, ramp_report, None

    _, family_ramp_report = _fit_ramp_geometry(evidence, evidence.family_events, width_prior)
    if family_ramp_report.get("slope_deg") is None:
        return ramp_terrain, ramp_report, None

    family = classify_terrain_family(
        evidence.joints,
        evidence.demo_joints,
        evidence.fps,
        evidence.family_events,
        family_ramp_report,
        neutral_foot_pitch=neutral_foot_pitch,
        family_evidence_mode=family_evidence_mode,
    )
    if family["family"] == "ramp":
        if len(evidence.events) >= 3:
            ramp_terrain, ramp_report = _fit_ramp_geometry(
                evidence,
                evidence.events,
                width_prior,
                min_events=3,
            )
        else:
            # Two annotated toe stances can still expose four independently calibrated
            # ankle/toe support probes.  Once those probes identify a continuous plane,
            # retain their fitted geometry instead of publishing two horizontal pads.
            ramp_terrain, ramp_report = _fit_ramp_geometry(
                evidence,
                evidence.family_events,
                width_prior,
            )
            ramp_report["short_ramp_geometry_source"] = "family_ankle_toe_probes"
    return ramp_terrain, ramp_report, family


def _validation_score(evidence: _FitEvidence, terrain: TerrainSpec) -> dict:
    """Score a candidate with the same support convention used by the public gate."""
    return validate_terrain(
        evidence.joints,
        evidence.demo_joints,
        terrain,
        evidence.fps,
        contact_joints=evidence.contact_joints,
        stance_speed=evidence.stance_speed,
        offsets=evidence.offsets,
        seat_rests=evidence.rests,
        pelvis_seat_offset=evidence.pelvis_seat_offset,
        seat_support_heights=evidence.seat_support_heights,
        compensate_sloped_offsets=evidence.offsets_source != "flat_reference",
    )


def _physical_score(check: Mapping[str, object]) -> tuple[float, float]:
    """Order candidate fits by support error, then free-body penetration."""
    return float(check["raised_contact_error_max"]), float(check["max_penetration"])


def _validation_summary(check: Mapping[str, object]) -> dict[str, float | bool]:
    """Keep only the stable physical metrics needed in model-selection reports."""
    return {
        "raised_contact_error_max": float(check["raised_contact_error_max"]),
        "max_penetration": float(check["max_penetration"]),
        "passed": bool(check["passed"]),
    }


def _refine_ramp_from_orientation(
    evidence: _FitEvidence,
    ramp_terrain: TerrainSpec | None,
    ramp_report: dict,
    family: dict,
    width_prior: float,
) -> tuple[TerrainSpec | None, dict]:
    """Adopt an orientation-derived slope only when it improves physical support."""
    orientation_slope = family.get("surface_normal_slope_deg")
    if ramp_terrain is None or family["family"] != "ramp" or orientation_slope is None:
        return ramp_terrain, ramp_report

    refined_terrain, refined_report = _fit_ramp_geometry(
        evidence,
        evidence.events,
        width_prior,
        slope_hint_deg=float(orientation_slope),
        min_length=RAMP_ORIENTED_MIN_LENGTH,
    )
    if refined_terrain is None:
        return ramp_terrain, ramp_report

    refined_report["initial_footfall_slope_deg"] = ramp_report["slope_deg"]
    initial_check = _validation_score(evidence, ramp_terrain)
    refined_check = _validation_score(evidence, refined_terrain)
    if _physical_score(refined_check) < _physical_score(initial_check):
        return refined_terrain, refined_report

    ramp_report["orientation_refinement_rejected"] = {
        "slope_deg": refined_report["slope_deg"],
        "support_error_max": refined_check["raised_contact_error_max"],
        "max_penetration": refined_check["max_penetration"],
    }
    return ramp_terrain, ramp_report


def _fit_ramp_candidate(
    evidence: _FitEvidence,
    width_prior: float,
    neutral_foot_pitch: Mapping[str, float] | None,
    family_evidence_mode: str,
    report: dict,
) -> TerrainSpec | None:
    """Fit and physically classify the continuous-incline candidate."""
    ramp_terrain, ramp_report = _fit_ramp_geometry(evidence, evidence.events, width_prior)
    ramp_terrain, ramp_report, preclassified_family = _classify_short_ramp(
        evidence,
        ramp_terrain,
        ramp_report,
        width_prior,
        neutral_foot_pitch,
        family_evidence_mode,
    )
    if ramp_report.get("slope_deg") is not None:
        family = (
            preclassified_family
            if preclassified_family is not None
            else classify_terrain_family(
                evidence.joints,
                evidence.demo_joints,
                evidence.fps,
                evidence.family_events,
                ramp_report,
                neutral_foot_pitch=neutral_foot_pitch,
                family_evidence_mode=family_evidence_mode,
            )
        )
    else:
        family = {
            "family": "steps",
            "model": "physical_surface" if family_evidence_mode == FAMILY_EVIDENCE_PHYSICAL else "height_profile_only",
            "reason": "no geometrically valid continuous-incline candidate",
            "family_evidence_mode": family_evidence_mode,
        }
    report["family_evidence"] = family
    ramp_terrain, ramp_report = _refine_ramp_from_orientation(
        evidence,
        ramp_terrain,
        ramp_report,
        family,
        width_prior,
    )
    if family["family"] == "steps" and ramp_terrain is not None:
        ramp_report["profile_candidate_available"] = True
        ramp_report["rejected"] = "independent surface-normal and swing-clearance evidence favors discrete support"
        ramp_terrain = None
    report["ramp"] = ramp_report
    return ramp_terrain


def _record_ground_evidence(evidence: _FitEvidence, report: dict) -> None:
    """Record the lowest support level before fitting candidate families."""
    ground = evidence.levels[0]
    report["joint_offsets"] = evidence.offsets
    report["pelvis_seat_offset"] = evidence.pelvis_seat_offset
    report["ground_level"] = {
        "n_events": len(ground),
        "mean_z_by_joint": {
            joint: float(np.mean([event.z for event in ground if event.joint == joint]))
            for joint in sorted({event.joint for event in ground})
        },
    }


def _fit_level_candidates(
    evidence: _FitEvidence,
    min_level_height: float,
    min_events_per_level: int,
    stair_flight: str,
    stair_width_prior: float,
    report: dict,
) -> tuple[list[bool], bool, TerrainSpec | None]:
    """Accept measured support levels and optionally fit a staircase flight."""
    level_heights = [_level_height(level, evidence.offsets)[0] for level in evidence.levels]
    ground_observed = level_heights[0] < min_level_height
    report["ground_level"]["observed_floor"] = bool(ground_observed)
    report["ground_level"]["height"] = float(level_heights[0])
    accepted = _accept_levels(
        level_heights,
        evidence.levels,
        min_events_per_level,
        min_level_height,
    )
    report["accepted_levels"] = accepted

    usable = [level for level, is_accepted in zip(evidence.levels, accepted, strict=True) if is_accepted]
    support_height_targets = [[event.z - evidence.offsets.get(event.joint, 0.0) for event in level] for level in usable]
    usable_heights = np.array([float(np.mean(targets)) for targets in support_height_targets])
    reassigned_constraints: list[dict[str, object]] = []
    # A weak cluster may be too small to create a new tread, but validation still treats
    # its stance as motion evidence. Assign every such observation to the closest accepted
    # surface so shared-riser regularization cannot move that surface out from under it.
    for source_index, (level, is_accepted) in enumerate(zip(evidence.levels, accepted, strict=True)):
        if is_accepted or not len(usable_heights):
            continue
        for event in level:
            target = float(event.z - evidence.offsets.get(event.joint, 0.0))
            target_index = int(np.argmin(np.abs(usable_heights - target)))
            support_height_targets[target_index].append(target)
            reassigned_constraints.append(
                {
                    "source_level": source_index,
                    "target_level": target_index,
                    "joint": event.joint,
                    "height": target,
                    "nearest_level_distance": abs(float(usable_heights[target_index]) - target),
                }
            )
    report["stair_support_constraints"] = {
        "source": "all motion-derived stance heights",
        "target_counts": [len(targets) for targets in support_height_targets],
        "reassigned": reassigned_constraints,
    }
    n_raised = sum(1 for level in usable if _level_height(level, evidence.offsets)[0] >= min_level_height)
    flight_terrain = None
    if stair_flight == "auto" and n_raised >= STAIR_MIN_LEVELS and (len(usable) > n_raised or not ground_observed):
        flight_terrain, flight_report = fit_stair_flight(
            evidence.joints if evidence.use_free_space_evidence else evidence.joints[:0],
            usable,
            evidence.offsets,
            free_space_tol=evidence.free_space_tol,
            width_prior=stair_width_prior,
            clearance=evidence.clearance,
            contact_margin=evidence.contact_margin,
            motion_foot_xy=evidence.motion_foot_xy,
            motion_foot_xyz=evidence.motion_foot_xyz,
            support_height_targets=support_height_targets,
        )
        report["stair_flight"] = flight_report
    return accepted, ground_observed, flight_terrain


def _make_level_geometry(
    index: int,
    events: list[StanceEvent],
    entry: dict,
) -> _LevelGeometry:
    """Express one support level in its principal horizontal coordinates."""
    contacts = np.concatenate([event.xy for event in events], axis=0)
    point_event = np.repeat(np.arange(len(events)), [len(event.xy) for event in events])
    yaw = _principal_yaw(contacts)
    centre_xy = contacts.mean(axis=0)
    cos_neg_yaw, sin_neg_yaw = np.cos(-yaw), np.sin(-yaw)
    delta = contacts - centre_xy
    longitudinal = cos_neg_yaw * delta[:, 0] - sin_neg_yaw * delta[:, 1]
    lateral = sin_neg_yaw * delta[:, 0] + cos_neg_yaw * delta[:, 1]
    entry["yaw"] = yaw
    entry["contact_spread_z"] = float(np.ptp([event.z for event in events]))
    entry["boxes"] = []
    return _LevelGeometry(
        index=index,
        events=events,
        entry=entry,
        point_event=point_event,
        yaw=yaw,
        centre_xy=centre_xy,
        cos_neg_yaw=cos_neg_yaw,
        sin_neg_yaw=sin_neg_yaw,
        longitudinal=longitudinal,
        lateral=lateral,
    )


def _exclude_lower_support(
    evidence: _FitEvidence,
    geometry: _LevelGeometry,
    group_index: int,
    contact_indices: np.ndarray,
    level_contacts: Sequence[np.ndarray],
    required_bounds: tuple[float, float, float, float],
    fitted_bounds: tuple[float, float, float, float],
    box_height: float,
    box_report: dict,
    report: dict,
) -> tuple[tuple[float, float, float, float], bool]:
    """Trim a box around lower contacts and reject overwhelming conflicts."""
    claimed = np.concatenate(level_contacts[: geometry.index], axis=0)
    delta = claimed - geometry.centre_xy
    claimed_u = geometry.cos_neg_yaw * delta[:, 0] - geometry.sin_neg_yaw * delta[:, 1]
    claimed_v = geometry.sin_neg_yaw * delta[:, 0] + geometry.cos_neg_yaw * delta[:, 1]
    fitted_bounds, unresolved = _exclude_claimed_points(
        fitted_bounds,
        required_bounds,
        claimed_u,
        claimed_v,
        evidence.clearance,
    )
    box_report["claimed_unresolved"] = unresolved
    if unresolved:
        report["warnings"].append(
            f"level {geometry.index} box {group_index}: {unresolved} footfall(s) of a lower "
            "surface lie inside this box's own support requirement and cannot be excluded; "
            "two surfaces overlap in xy and the higher one wins"
        )

    own_frames = len(contact_indices)
    ratio = unresolved / own_frames if own_frames else 0.0
    box_report["own_frames"] = own_frames
    box_report["conflict_ratio"] = ratio
    rejected = box_height < CONFLICT_REJECT_MAX_HEIGHT and ratio >= CONFLICT_REJECT_MIN_RATIO
    if rejected:
        box_report["rejected"] = (
            f"{unresolved} conflicting lower-level frame(s) against {own_frames} of its own (ratio {ratio:.2f})"
        )
        report["warnings"].append(
            f"level {geometry.index} box {group_index}: dropped, {box_report['rejected']} at "
            f"{box_height * 1000:.0f} mm - a real surface this low would not be this "
            "outnumbered by the floor"
        )
        geometry.entry["boxes"].append(box_report)
    return fitted_bounds, rejected


def _fit_level_box(
    evidence: _FitEvidence,
    geometry: _LevelGeometry,
    group_index: int,
    contact_indices: np.ndarray,
    body_points: np.ndarray,
    level_contacts: Sequence[np.ndarray],
    min_level_height: float,
    boxes: list[BoxSpec],
    report: dict,
) -> None:
    """Fit one free-space-bounded box around a connected contact group."""
    u = geometry.longitudinal
    v = geometry.lateral
    u_lo, u_hi = float(u[contact_indices].min()), float(u[contact_indices].max())
    v_lo, v_hi = float(v[contact_indices].min()), float(v[contact_indices].max())

    event_indices = sorted(set(geometry.point_event[contact_indices].tolist()))
    own_events = [geometry.events[index] for index in event_indices]
    box_height, box_height_info = _level_height(own_events, evidence.offsets)
    if box_height < min_level_height:
        return

    u_lo_req = u_lo - evidence.contact_margin[0]
    u_hi_req = u_hi + evidence.contact_margin[0]
    v_lo_req = v_lo - evidence.contact_margin[1]
    v_hi_req = v_hi + evidence.contact_margin[1]

    v_centre = 0.5 * (v_lo_req + v_hi_req)
    v_half = 0.5 * (v_hi_req - v_lo_req) + evidence.max_extension[1]
    v_lo_fit, v_hi_fit = v_centre - v_half, v_centre + v_half
    lo_limit, hi_limit = _free_space_limit(
        body_points,
        geometry.yaw,
        geometry.centre_xy,
        abs(v_centre) + v_half,
        box_height,
        evidence.free_space_tol,
        u_lo_req,
        u_hi_req,
        evidence.clearance,
    )
    u_lo_fit = max(u_lo_req - evidence.max_extension[0], lo_limit)
    u_hi_fit = min(u_hi_req + evidence.max_extension[0], hi_limit)
    box_report = {
        "height": box_height,
        "height_spread": box_height_info["spread"],
        "support_u": (u_lo, u_hi),
        "required_u": (u_lo_req, u_hi_req),
        "fitted_u": (u_lo_fit, u_hi_fit),
        "support_v": (v_lo, v_hi),
        "fitted_v_half": v_half,
    }
    if u_lo_fit > u_lo_req or u_hi_fit < u_hi_req:
        report["warnings"].append(
            f"level {geometry.index} box {group_index}: free space ({lo_limit:.3f}, "
            f"{hi_limit:.3f}) is tighter than the contact footprint ({u_lo_req:.3f}, "
            f"{u_hi_req:.3f}); clamped to free space, so a footfall may overhang the edge"
        )
        box_report["clamped"] = True
        u_lo_fit, u_hi_fit = min(u_lo_fit, u_hi_fit), max(u_lo_fit, u_hi_fit)

    if evidence.exclude_claimed and geometry.index:
        fitted_bounds, rejected = _exclude_lower_support(
            evidence,
            geometry,
            group_index,
            contact_indices,
            level_contacts,
            (u_lo_req, u_hi_req, v_lo_req, v_hi_req),
            (u_lo_fit, u_hi_fit, v_lo_fit, v_hi_fit),
            box_height,
            box_report,
            report,
        )
        u_lo_fit, u_hi_fit, v_lo_fit, v_hi_fit = fitted_bounds
        if rejected:
            return

    u_centre = 0.5 * (u_lo_fit + u_hi_fit)
    u_half = max(0.5 * (u_hi_fit - u_lo_fit), 1e-3)
    v_centre = 0.5 * (v_lo_fit + v_hi_fit)
    v_half = max(0.5 * (v_hi_fit - v_lo_fit), 1e-3)
    box_report["fitted_v"] = (v_lo_fit, v_hi_fit)
    cos_yaw, sin_yaw = np.cos(geometry.yaw), np.sin(geometry.yaw)
    pos_xy = geometry.centre_xy + np.array(
        [
            cos_yaw * u_centre - sin_yaw * v_centre,
            sin_yaw * u_centre + cos_yaw * v_centre,
        ]
    )
    boxes.append(
        BoxSpec(
            pos=(float(pos_xy[0]), float(pos_xy[1]), box_height / 2),
            size=(u_half, max(v_half, 1e-3), box_height / 2),
            yaw=geometry.yaw,
            name=f"terrain_box_{len(boxes)}",
        )
    )
    box_report["box"] = len(boxes) - 1
    geometry.entry["boxes"].append(box_report)


def _fit_level(
    evidence: _FitEvidence,
    index: int,
    accepted: Sequence[bool],
    body_points: np.ndarray,
    level_contacts: Sequence[np.ndarray],
    min_level_height: float,
    split_gap: float,
    ambiguous_split_gap: float,
    boxes: list[BoxSpec],
    report: dict,
) -> None:
    """Fit every connected box supported by one accepted height level."""
    level = evidence.levels[index]
    height, height_info = _level_height(level, evidence.offsets)
    entry = {
        "index": index,
        "n_events": len(level),
        "height": height,
        "joints": sorted({event.joint for event in level}),
        **height_info,
    }
    if height < min_level_height:
        entry["skipped"] = f"height {height:.3f} m below min_level_height"
        report["levels"].append(entry)
        return
    if not accepted[index]:
        entry["skipped"] = (
            f"only {len(level)} contact(s), below min_events_per_level, and an accepted "
            f"level within {WEAK_LEVEL_MERGE_GAP:.3f} m explains it"
        )
        report["levels"].append(entry)
        return
    if height_info["spread"] > 0.02:
        report["warnings"].append(
            f"level {index}: per-joint height estimates disagree by "
            f"{height_info['spread'] * 1000:.0f} mm ({height_info['per_joint']}); the "
            "surface may be sloped, or the level may be merging two real surfaces"
        )

    geometry = _make_level_geometry(index, level, entry)
    spatial_gap = min(split_gap, ambiguous_split_gap) if height_info["spread"] > 0.02 else split_gap
    for group_index, contact_indices in enumerate(_split_by_gap(geometry.longitudinal, spatial_gap)):
        _fit_level_box(
            evidence,
            geometry,
            group_index,
            contact_indices,
            body_points,
            level_contacts,
            min_level_height,
            boxes,
            report,
        )
    report["levels"].append(entry)


def _fit_per_level_candidate(
    evidence: _FitEvidence,
    accepted: Sequence[bool],
    ground_observed: bool,
    min_level_height: float,
    split_gap: float,
    ambiguous_split_gap: float,
    report: dict,
    provenance: dict,
) -> TerrainSpec:
    """Fit independent support boxes for every accepted discrete height level."""
    body_points = evidence.joints.reshape(-1, 3) if evidence.use_free_space_evidence else np.zeros((0, 3), dtype=float)
    level_contacts = [np.concatenate([event.xy for event in level], axis=0) for level in evidence.levels]
    boxes: list[BoxSpec] = []
    first_terrain_level = 1 if ground_observed else 0
    for index in range(first_terrain_level, len(evidence.levels)):
        _fit_level(
            evidence,
            index,
            accepted,
            body_points,
            level_contacts,
            min_level_height,
            split_gap,
            ambiguous_split_gap,
            boxes,
            report,
        )

    boxes, buried = _drop_buried_boxes(boxes)
    for name, covering_box in buried:
        report["warnings"].append(
            f"{name}: top face lies entirely under {covering_box}; dropped, since no foot "
            "can reach a surface another box covers"
        )
    report["n_buried_boxes"] = len(buried)
    return TerrainSpec(boxes=tuple(boxes), provenance=provenance)


def _select_terrain_model(
    evidence: _FitEvidence,
    ramp_terrain: TerrainSpec | None,
    flight_terrain: TerrainSpec | None,
    per_level_terrain: TerrainSpec,
    seat_boxes: Sequence[BoxSpec],
    report: dict,
    provenance: dict,
) -> TerrainSpec:
    """Select the classified family and best-supported representation."""
    if ramp_terrain is not None and len(ramp_terrain):
        ramp_check = _validation_score(evidence, ramp_terrain)
        report["ramp_scores"] = {"ramp": _validation_summary(ramp_check)}
        return _finish_terrain(
            ramp_terrain.boxes,
            seat_boxes,
            report,
            provenance,
            "ramp",
            slope_deg=report["ramp"]["slope_deg"],
        )

    if flight_terrain is not None and len(flight_terrain):
        per_level_check = _validation_score(evidence, per_level_terrain)
        flight_check = _validation_score(evidence, flight_terrain)
        report["model_scores"] = {
            model: _validation_summary(check)
            for model, check in (
                ("per_level", per_level_check),
                ("stair_flight", flight_check),
            )
        }
        # A coherent staircase is the structurally stronger explanation.  Once it
        # passes the same physical gate as the sparse alternative, a tiny raw
        # penetration difference must not reward isolated per-foot pads.  If neither
        # candidate passes, retain the physical score as a diagnostic fallback.
        select_flight = bool(flight_check["passed"]) or (
            not bool(per_level_check["passed"]) and _physical_score(flight_check) <= _physical_score(per_level_check)
        )
        if select_flight:
            return _finish_terrain(
                flight_terrain.boxes,
                seat_boxes,
                report,
                provenance,
                "stair_flight",
            )
    return _finish_terrain(
        per_level_terrain.boxes,
        seat_boxes,
        report,
        provenance,
        "per_level",
    )


def fit_terrain_from_motion(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    stance_speed: float = DEFAULT_STANCE_SPEED,
    level_tol: float = DEFAULT_LEVEL_TOL,
    free_space_tol: float = DEFAULT_FREE_SPACE_TOL,
    contact_margin: tuple[float, float] = DEFAULT_CONTACT_MARGIN,
    max_extension: tuple[float, float] = DEFAULT_MAX_EXTENSION,
    min_level_height: float = 0.04,
    min_events_per_level: int = 2,
    split_gap: float = DEFAULT_SPLIT_GAP,
    ambiguous_split_gap: float = DEFAULT_AMBIGUOUS_SPLIT_GAP,
    clearance: float = 0.02,
    exclude_claimed: bool = DEFAULT_EXCLUDE_CLAIMED,
    stair_flight: str = "auto",
    stair_width_prior: float = STAIR_WIDTH_PRIOR,
    ramp: str = "auto",
    ramp_width_prior: float = RAMP_WIDTH_PRIOR,
    neutral_foot_pitch: Mapping[str, float] | None = None,
    family_evidence_mode: str = FAMILY_EVIDENCE_PHYSICAL,
    seat: str = "auto",
    pelvis_seat_offset: float = PELVIS_SEAT_OFFSET,
    calibrated_joint_offsets: Mapping[str, float] | None = None,
    seat_support_heights: Sequence[float] | None = None,
    use_free_space_evidence: bool = True,
    calibrated_joint_offsets_source: str | None = None,
    allow_boundary_truncated_support: bool = True,
) -> tuple[TerrainSpec, dict]:
    """Fit the best-supported box terrain model to a source motion.

    Args:
        joints: Solver-space source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        stance_speed: Maximum contact-joint speed in meters per second.
        level_tol: Maximum height difference within a support level, in meters.
        free_space_tol: Vertical tolerance used to identify body points inside a box.
        use_free_space_evidence: Whether non-support body trajectories constrain fitted
            surface extents. When false, fitting retains positive support evidence and
            downstream physical validation but does not use body points to trim level,
            ramp, stair, or seat geometry.
        contact_margin: Longitudinal and lateral support margins in meters.
        max_extension: Maximum longitudinal and lateral box growth in meters.
        min_level_height: Minimum raised-surface height in meters.
        min_events_per_level: Minimum stance events required for a support level.
        split_gap: Longitudinal gap that separates boxes on one level.
        ambiguous_split_gap: Split gap used when per-joint heights disagree.
        clearance: Horizontal clearance from body points in meters.
        exclude_claimed: Whether to exclude contacts assigned to lower levels.
        stair_flight: Stair model mode, ``"auto"`` or ``"off"``.
        stair_width_prior: Preferred total staircase width in meters.
        ramp: Ramp model mode, ``"auto"`` or ``"off"``.
        ramp_width_prior: Preferred total ramp width in meters.
        neutral_foot_pitch: Optional left/right ankle-to-toe pitch measured on an
            automatically flat reference motion.  This is skeleton/marker calibration,
            not a terrain-family label.  If omitted, selection conservatively falls back
            to the direct footfall-profile residual.
        family_evidence_mode: ``"physical"`` uses supported-foot normals and swing
            clearance to select ramp versus steps. ``"height_only"`` is the paper
            ablation and uses only the frozen contact-height residual rule.
        seat: Seat model mode, ``"auto"`` or ``"off"``.
        pelvis_seat_offset: Expected pelvis-center height above a seat in meters.
        calibrated_joint_offsets: Optional anatomical ankle/toe heights measured on a
            separate flat reference clip. This prevents a short non-flat clip whose probe
            never reaches the floor from absorbing a tread or ramp rise into that probe's
            anatomical offset.
        calibrated_joint_offsets_source: Optional metadata label for explicitly
            supplied offsets. This does not change their numerical interpretation.
        seat_support_heights: Optional posed body-surface height for every detected
            seated rest. When supplied, seat heights preserve pose and shape instead of
            using a universal pelvis offset.
        allow_boundary_truncated_support: Accept otherwise-valid support intervals
            touching a clip boundary without requiring their complete minimum duration.

    Returns:
        Fitted terrain and a report containing detected support, model scores,
        fitted extents, and warnings.

    Raises:
        ValueError: If the motion schema, frame rate, model mode, or geometric
            pair is invalid.
    """
    joints, demo_joints, fps = _validate_motion_input(joints, demo_joints, fps)
    contact_margin = _nonnegative_pair(contact_margin, "contact_margin")
    max_extension = _nonnegative_pair(max_extension, "max_extension")
    _validate_model_mode("stair_flight", stair_flight)
    _validate_model_mode("ramp", ramp)
    _validate_model_mode("seat", seat)
    if family_evidence_mode not in FAMILY_EVIDENCE_MODES:
        supported = ", ".join(FAMILY_EVIDENCE_MODES)
        raise ValueError(f"family_evidence_mode must be one of {supported}, got {family_evidence_mode!r}")
    if not isinstance(use_free_space_evidence, bool):
        raise ValueError("use_free_space_evidence must be a boolean")
    if calibrated_joint_offsets_source is not None and (
        not isinstance(calibrated_joint_offsets_source, str) or not calibrated_joint_offsets_source
    ):
        raise ValueError("calibrated_joint_offsets_source must be a non-empty string or None")
    evidence = _collect_fit_evidence(
        joints,
        demo_joints,
        fps,
        stance_speed,
        level_tol,
        free_space_tol,
        use_free_space_evidence,
        contact_margin,
        max_extension,
        clearance,
        exclude_claimed,
        seat,
        pelvis_seat_offset,
        calibrated_joint_offsets,
        calibrated_joint_offsets_source,
        seat_support_heights,
        allow_boundary_truncated_support,
    )
    report, provenance = _build_report(evidence)
    seat_boxes = _fit_seat_candidate(evidence, seat, report)

    if not evidence.levels:
        report["warnings"].append("no stance events detected; returning flat terrain")
        return _finish_terrain((), seat_boxes, report, provenance, "flat"), report

    _record_ground_evidence(evidence, report)
    ramp_terrain = None
    if ramp == "auto":
        ramp_terrain = _fit_ramp_candidate(
            evidence,
            ramp_width_prior,
            neutral_foot_pitch,
            family_evidence_mode,
            report,
        )

    accepted, ground_observed, flight_terrain = _fit_level_candidates(
        evidence,
        min_level_height,
        min_events_per_level,
        stair_flight,
        stair_width_prior,
        report,
    )
    terrain = _fit_per_level_candidate(
        evidence,
        accepted,
        ground_observed,
        min_level_height,
        split_gap,
        ambiguous_split_gap,
        report,
        provenance,
    )

    selected = _select_terrain_model(
        evidence,
        ramp_terrain,
        flight_terrain,
        terrain,
        seat_boxes,
        report,
        provenance,
    )
    return selected, report
