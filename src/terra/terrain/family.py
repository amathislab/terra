"""Physically identify a continuous incline versus discrete treads.

Sparse footfall heights do not identify the surface between contacts: a staircase and a
ramp can pass through the same support points.  The selector here therefore uses two
measurements that are independent of the fitted heights:

* while a foot is supported, its ankle-to-toe chord follows a ramp's surface normal but
  stays near its neutral orientation on a stair tread; and
* during a height-changing swing, a stair nose forces an early lift (ascent) or delayed
  drop (descent) relative to the straight line between supports.

The only calibration is performed based on the neutral chord pitch of each foot.
This can be measured from an automatically flat reference motion or by
explicitly supplying a representation's model-rest value.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from terra.terrain.stance import DEFAULT_CONTACT_JOINTS, DEFAULT_LEVEL_TOL, StanceEvent

# The fitted SMPL-H robot's ankle-to-toe chords in its neutral pose.  These are skeleton
# metadata, not statistics of a terrain dataset.  Other skeleton adapters should pass a
# measured neutral pitch instead of using this fallback.
SMPLH_NEUTRAL_FOOT_PITCH_DEG = {"L": -24.83603286743164, "R": -19.05824851989746}

# Five millimetres of independent endpoint error on a roughly 20 cm foot chord changes its
# angle by about two degrees.  Inside this resolution the surface-normal cue is explicitly
# considered ambiguous and the independent obstacle-clearance cue may decide.
SURFACE_NORMAL_RESOLUTION_DEG = 2.0

# Five percent of one support-level change is below the temporal/pose resolution of the
# quartile swing statistic.  A weaker disagreement must not overturn a supported-foot
# normal observed over several stance intervals.
SWING_CLEARANCE_RESOLUTION = 0.05

# When the direct ramp residual falls in the narrow band between the strict 16 mm
# single-cue threshold and half of one 40 mm support-level resolution, independent
# ramp-like swing evidence may resolve the ambiguity. This bound is expressed from the
# fitter's motion resolution rather than a dataset or apparatus measurement.
RAMP_AMBIGUOUS_PROFILE_RMS = 0.5 * DEFAULT_LEVEL_TOL

# A continuous incline is not identifiable from one interior support position and two
# endpoint landings: the same three heights are explained by one raised horizontal
# surface. Two distinct footfalls on the candidate incline are the minimum motion-only
# evidence for continuity.
RAMP_MIN_INTERIOR_FOOTFALLS = 2

# Height-only mode uses the same support evidence without orientation or swing cues.
FAMILY_EVIDENCE_PHYSICAL = "physical"
FAMILY_EVIDENCE_HEIGHT_ONLY = "height_only"
FAMILY_EVIDENCE_MODES = (FAMILY_EVIDENCE_PHYSICAL, FAMILY_EVIDENCE_HEIGHT_ONLY)

# At the halfway point between successive supports, an ideal stair nose has introduced
# half of the level discontinuity relative to their connecting line.  On descent, the
# analogous early-versus-late contrast is one third of a level change.  Both are
# dimensionless geometric reference values, not fitted thresholds.
STAIR_ASCENT_MIDPOINT_CLEARANCE = 0.5
STAIR_DESCENT_EARLY_LATE_CLEARANCE = 1.0 / 3.0


def calibrate_neutral_foot_pitch(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    *,
    speed_gate: float = 0.30,
    events: Sequence[StanceEvent] | None = None,
) -> dict[str, float]:
    """Measure neutral ankle-to-toe pitch from an automatically flat reference clip.

    This estimates anatomy/marker placement only.  It does not inspect a terrain label or
    tune a decision boundary.  Both landmarks must be slow, which excludes swing and the
    rotating parts of heel strike and toe off.
    """

    names = list(demo_joints)
    result: dict[str, float] = {}
    for side in ("L", "R"):
        ankle_name, toe_name = f"{side}_Ankle", f"{side}_Toe"
        if ankle_name not in names or toe_name not in names:
            continue
        ankle = np.asarray(joints[:, names.index(ankle_name)], dtype=float)
        toe = np.asarray(joints[:, names.index(toe_name)], dtype=float)
        ankle_speed = np.r_[0.0, np.linalg.norm(np.diff(ankle, axis=0), axis=1) * fps]
        toe_speed = np.r_[0.0, np.linalg.norm(np.diff(toe, axis=0), axis=1) * fps]
        chord = toe - ankle
        horizontal = np.linalg.norm(chord[:, :2], axis=1)
        pitch = np.degrees(np.arctan2(chord[:, 2], horizontal))
        usable = (ankle_speed < speed_gate) & (toe_speed < speed_gate) & np.isfinite(pitch) & (horizontal > 1e-6)
        if events is not None:
            supported = np.zeros(len(joints), dtype=bool)
            for event in events:
                if event.joint in (ankle_name, toe_name):
                    supported[event.start : event.end] = True
            usable &= supported
        if np.any(usable):
            result[side] = float(np.median(pitch[usable]))
    return result


def _supported_intervals(events: Sequence[StanceEvent], side: str, n_frames: int):
    ankle = np.zeros(n_frames, dtype=bool)
    toe = np.zeros(n_frames, dtype=bool)
    for event in events:
        if event.joint == f"{side}_Ankle":
            ankle[event.start : event.end] = True
        elif event.joint == f"{side}_Toe":
            toe[event.start : event.end] = True
    for supported, run in itertools.groupby(enumerate(ankle & toe), key=lambda item: item[1]):
        if not supported:
            continue
        indices = np.fromiter((index for index, _ in run), dtype=int)
        if len(indices) >= 3:
            yield indices


def surface_normal_evidence(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    events: Sequence[StanceEvent],
    ramp: Mapping[str, object],
    neutral_pitch: Mapping[str, float] | None,
) -> dict[str, float | int | str | None]:
    """Compare supported-foot orientation under ramp and flat-tread hypotheses."""

    names = list(demo_joints)
    required = set(DEFAULT_CONTACT_JOINTS)
    if (
        not required.issubset(names)
        or neutral_pitch is None
        or any(side not in neutral_pitch for side in ("L", "R"))
        or ramp.get("yaw") is None
        or ramp.get("slope_deg") is None
    ):
        return {
            "surface_normal_family": None,
            "surface_normal_margin_deg": None,
            "surface_normal_step_error_deg": None,
            "surface_normal_ramp_error_deg": None,
            "surface_normal_n_intervals": 0,
            "surface_normal_slope_deg": None,
        }

    axis = np.array([np.cos(float(ramp["yaw"])), np.sin(float(ramp["yaw"]))])
    grade = float(np.tan(np.radians(float(ramp["slope_deg"]))))
    step_errors: list[float] = []
    ramp_errors: list[float] = []
    grade_estimates: list[float] = []
    for side in ("L", "R"):
        ankle = np.asarray(joints[:, names.index(f"{side}_Ankle")], dtype=float)
        toe = np.asarray(joints[:, names.index(f"{side}_Toe")], dtype=float)
        for indices in _supported_intervals(events, side, len(joints)):
            chord = toe[indices] - ankle[indices]
            horizontal = np.linalg.norm(chord[:, :2], axis=1)
            valid = np.isfinite(chord).all(axis=1) & (horizontal > 1e-6)
            if not np.any(valid):
                continue
            direction = chord[valid, :2] / horizontal[valid, None]
            observed = np.degrees(np.arctan2(chord[valid, 2], horizontal[valid]))
            predicted_surface = np.degrees(np.arctan(grade * (direction @ axis)))
            neutral = float(neutral_pitch[side])
            step_errors.append(float(np.median(np.abs(observed - neutral))))
            ramp_errors.append(float(np.median(np.abs(observed - neutral - predicted_surface))))
            projection = direction @ axis
            informative = np.abs(projection) > 0.30
            if np.any(informative):
                delta = np.radians(observed[informative] - neutral)
                estimates = np.tan(delta) / projection[informative]
                grade_estimates.extend(estimates[np.isfinite(estimates) & (estimates > 0.0)].tolist())

    if not step_errors:
        return {
            "surface_normal_family": None,
            "surface_normal_margin_deg": None,
            "surface_normal_step_error_deg": None,
            "surface_normal_ramp_error_deg": None,
            "surface_normal_n_intervals": 0,
            "surface_normal_slope_deg": None,
        }
    step_error = float(np.median(step_errors))
    ramp_error = float(np.median(ramp_errors))
    margin = step_error - ramp_error
    return {
        "surface_normal_family": "ramp" if margin > 0.0 else "steps",
        "surface_normal_margin_deg": margin,
        "surface_normal_step_error_deg": step_error,
        "surface_normal_ramp_error_deg": ramp_error,
        "surface_normal_n_intervals": len(step_errors),
        "surface_normal_slope_deg": (
            float(np.degrees(np.arctan(np.median(grade_estimates)))) if grade_estimates else None
        ),
    }


def swing_clearance_evidence(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    events: Sequence[StanceEvent],
) -> dict[str, float | int | str | None]:
    """Measure the dimensionless stair-nose signature in height-changing swings."""

    names = list(demo_joints)
    if not set(DEFAULT_CONTACT_JOINTS).issubset(names):
        return {"swing_clearance_family": None, "swing_clearance_margin": None}

    by_direction: dict[str, dict[str, list[float]]] = {
        direction: {key: [] for key in ("q25", "mid", "q75")} for direction in ("ascent", "descent")
    }
    for side in ("L", "R"):
        ankle = np.asarray(joints[:, names.index(f"{side}_Ankle")], dtype=float)
        toe = np.asarray(joints[:, names.index(f"{side}_Toe")], dtype=float)
        intervals: list[tuple[int, int]] = []
        for event in sorted(
            (event for event in events if event.joint.startswith(side)),
            key=lambda event: event.start,
        ):
            if intervals and event.start <= intervals[-1][1] + 2:
                intervals[-1] = (
                    min(intervals[-1][0], event.start),
                    max(intervals[-1][1], event.end),
                )
            else:
                intervals.append((event.start, event.end))
        for (_, end), (start, _) in itertools.pairwise(intervals):
            if start <= end + 2:
                continue
            foot_z = np.minimum(ankle[end : start + 1, 2], toe[end : start + 1, 2])
            dz = float(foot_z[-1] - foot_z[0])
            scale = abs(dz)
            if scale <= DEFAULT_LEVEL_TOL:
                continue
            line_error = foot_z - np.linspace(float(foot_z[0]), float(foot_z[-1]), len(foot_z))
            values = {
                "q25": float(line_error[round(0.25 * (len(line_error) - 1))] / scale),
                "mid": float(line_error[round(0.50 * (len(line_error) - 1))] / scale),
                "q75": float(line_error[round(0.75 * (len(line_error) - 1))] / scale),
            }
            destination = by_direction["ascent" if dz > 0.0 else "descent"]
            for key, value in values.items():
                destination[key].append(value)

    scores: list[tuple[int, float, str]] = []
    result: dict[str, float | int | str | None] = {}
    for direction, values in by_direction.items():
        count = len(values["mid"])
        result[f"swing_clearance_{direction}_n"] = count
        medians = {key: float(np.median(samples)) if samples else None for key, samples in values.items()}
        for key, value in medians.items():
            result[f"swing_clearance_{direction}_{key}"] = value
        if direction == "ascent" and medians["mid"] is not None:
            score = medians["mid"] - STAIR_ASCENT_MIDPOINT_CLEARANCE
            scores.append((count, float(score), direction))
        elif direction == "descent" and medians["q25"] is not None:
            score = medians["q25"] - medians["q75"] - STAIR_DESCENT_EARLY_LATE_CLEARANCE
            scores.append((count, float(score), direction))

    if not scores:
        return {
            **result,
            "swing_clearance_family": None,
            "swing_clearance_margin": None,
            "swing_clearance_direction": None,
        }
    # A clip can contain both ascent and descent.  Prefer the direction represented by more
    # height-changing swings; a tie is resolved by the larger dimensionless separation.
    count, score, direction = max(scores, key=lambda value: (value[0], abs(value[1])))
    return {
        **result,
        "swing_clearance_family": "steps" if score > 0.0 else "ramp",
        "swing_clearance_margin": score,
        "swing_clearance_direction": direction,
        "swing_clearance_n": count,
    }


def classify_terrain_family(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    events: Sequence[StanceEvent],
    ramp: Mapping[str, object],
    *,
    neutral_foot_pitch: Mapping[str, float] | None = None,
    neutral_foot_pitch_source: str = "provided_flat_reference",
    family_evidence_mode: str = FAMILY_EVIDENCE_PHYSICAL,
) -> dict[str, object]:
    """Select ``"ramp"`` or ``"steps"`` from explicit physical evidence."""

    if family_evidence_mode not in FAMILY_EVIDENCE_MODES:
        supported = ", ".join(FAMILY_EVIDENCE_MODES)
        raise ValueError(f"family_evidence_mode must be one of {supported}, got {family_evidence_mode!r}")

    if family_evidence_mode == FAMILY_EVIDENCE_HEIGHT_ONLY:
        # Height-only selection uses the contact-height residual without foot
        # normals or swing-clearance evidence.
        from terra.terrain.ramps import RAMP_MAX_PROFILE_RMS

        residual = ramp.get("contact_profile_rms", ramp.get("profile_rms"))
        family = "ramp" if residual is not None and float(residual) <= RAMP_MAX_PROFILE_RMS else "steps"
        reason = "contact-height profile residual only (physical family cues disabled)"
        incline_footfalls = ramp.get("n_footfalls_on_incline")
        continuity_observed = incline_footfalls is None or int(incline_footfalls) >= RAMP_MIN_INTERIOR_FOOTFALLS
        if family == "ramp" and not continuity_observed:
            family = "steps"
            reason = "fewer than two observed support positions lie on the candidate incline"
        return {
            "family": family,
            "model": "height_profile_only",
            "reason": reason,
            "family_evidence_mode": family_evidence_mode,
            "neutral_source": "disabled_ablation",
            "neutral_foot_pitch_deg": None,
            "height_profile_residual_m": None if residual is None else float(residual),
            "surface_normal_family": None,
            "surface_normal_margin_deg": None,
            "surface_normal_step_error_deg": None,
            "surface_normal_ramp_error_deg": None,
            "surface_normal_n_intervals": 0,
            "surface_normal_slope_deg": None,
            "swing_clearance_family": None,
            "swing_clearance_margin": None,
            "swing_clearance_direction": None,
            "swing_clearance_n": 0,
        }

    neutral_source = neutral_foot_pitch_source
    neutral = neutral_foot_pitch
    if neutral is None:
        # Do not silently impose one skeleton's rest chord on another dataset.  A caller
        # that wants the SMPL-H model-rest prior can pass
        # ``SMPLH_NEUTRAL_FOOT_PITCH_DEG`` explicitly.  Without calibration, the direct
        # height profile is safer and preserved the pre-existing AMASS behavior.
        neutral_source = "unavailable"

    normal = surface_normal_evidence(joints, demo_joints, events, ramp, neutral)
    swing = swing_clearance_evidence(joints, demo_joints, events)
    normal_family = normal["surface_normal_family"]
    swing_family = swing["swing_clearance_family"]

    if normal_family is not None:
        family = str(normal_family)
        reason = "supported-foot surface normal"
        normal_margin = float(normal["surface_normal_margin_deg"])
        if (
            abs(normal_margin) <= SURFACE_NORMAL_RESOLUTION_DEG
            and swing_family is not None
            and swing_family != family
            and abs(float(swing["swing_clearance_margin"])) >= SWING_CLEARANCE_RESOLUTION
        ):
            # Sub-resolution angular differences have no reliable sign; use the
            # independently resolved swing cue to break the tie.
            family = str(swing_family)
            reason = "surface normal is within positional resolution; swing clearance decides"
    else:
        # With no neutral orientation, swing height alone is not decisive: ordinary foot
        # clearance divided by a small ramp rise can look exactly like stair clearance.
        # Use the candidate's direct height residual as the conservative primary fallback;
        # the swing measurement remains in the report.  This branch is mostly relevant to
        # minimal/custom skeletons whose adapter supplied no neutral-foot calibration.
        from terra.terrain.ramps import RAMP_MAX_PROFILE_RMS, RAMP_MIN_LENGTH

        # The 16 mm fallback threshold was established for the original residual after
        # subtracting pre-estimated sole offsets.  The joint-intercept ramp fit reports a
        # second residual with different flexibility; applying the old threshold to that
        # new statistic silently changed AMASS decisions despite identical units.  Keep
        # the uncalibrated fallback's statistic/threshold pair as one definition.
        # If calibration was supplied but a particular clip contains no jointly supported
        # foot interval, retain the calibrated path's joint-intercept profile rather than
        # switching algorithms merely because its primary cue is missing.
        residual = (
            ramp.get("contact_profile_rms", ramp.get("profile_rms")) if neutral is None else ramp.get("profile_rms")
        )
        family = "ramp" if residual is not None and float(residual) <= RAMP_MAX_PROFILE_RMS else "steps"
        reason = (
            "footfall profile residual (neutral surface-normal calibration unavailable)"
            if neutral is None
            else "footfall profile residual (no jointly supported foot interval)"
        )
        incline_footfalls = ramp.get("n_footfalls_on_incline")
        continuity_observed = incline_footfalls is None or int(incline_footfalls) >= RAMP_MIN_INTERIOR_FOOTFALLS
        if neutral is None and family == "ramp" and not continuity_observed:
            family = "steps"
            reason = "fewer than two observed support positions lie on the candidate incline"
        elif (
            neutral is None
            and family == "ramp"
            and swing_family == "steps"
            and abs(float(swing["swing_clearance_margin"])) >= SWING_CLEARANCE_RESOLUTION
            and ramp.get("length") is not None
            and float(ramp["length"]) <= RAMP_MIN_LENGTH + 1e-9
        ):
            # The candidate has not observed a continuous support span: its entire
            # longitudinal extent comes from the ramp fitter's minimum-length prior.
            # In that underdetermined case, an independently observed stair-nose swing
            # is stronger evidence than a low residual through a handful of footfalls.
            family = "steps"
            reason = "ramp span is entirely minimum-length prior; swing clearance decides"
        elif (
            neutral is None
            and family == "steps"
            and residual is not None
            and float(residual) <= RAMP_AMBIGUOUS_PROFILE_RMS
            and ramp.get("profile_max") is not None
            and float(ramp["profile_max"]) <= DEFAULT_LEVEL_TOL
            and swing_family == "ramp"
            and abs(float(swing["swing_clearance_margin"])) >= SWING_CLEARANCE_RESOLUTION
            and ramp.get("length") is not None
            and float(ramp["length"]) > RAMP_MIN_LENGTH + 1e-9
            and continuity_observed
        ):
            # No neutral-foot calibration is available, but two independent kinematic
            # observations agree: all footfalls lie within one support-level resolution
            # of a long continuous profile, and the height-changing swings lack the
            # stair-nose signature. The strict 16 mm residual remains sufficient by
            # itself; only this narrow ambiguous band requires the second cue.
            family = "ramp"
            reason = (
                "footfall profile lies within motion resolution; "
                "independent swing clearance supports a continuous incline"
            )

    return {
        "family": family,
        "model": "physical_surface",
        "reason": reason,
        "family_evidence_mode": family_evidence_mode,
        "neutral_source": neutral_source,
        "neutral_foot_pitch_deg": dict(neutral) if neutral is not None else None,
        **normal,
        **swing,
    }


__all__ = [
    "FAMILY_EVIDENCE_HEIGHT_ONLY",
    "FAMILY_EVIDENCE_MODES",
    "FAMILY_EVIDENCE_PHYSICAL",
    "RAMP_AMBIGUOUS_PROFILE_RMS",
    "RAMP_MIN_INTERIOR_FOOTFALLS",
    "SMPLH_NEUTRAL_FOOT_PITCH_DEG",
    "STAIR_ASCENT_MIDPOINT_CLEARANCE",
    "STAIR_DESCENT_EARLY_LATE_CLEARANCE",
    "SURFACE_NORMAL_RESOLUTION_DEG",
    "SWING_CLEARANCE_RESOLUTION",
    "calibrate_neutral_foot_pitch",
    "classify_terrain_family",
    "surface_normal_evidence",
    "swing_clearance_evidence",
]
