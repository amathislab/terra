"""A deliberately simple inferred-contact least-squares terrain baseline.

The baseline detects locally low, slow foot-joint intervals, reduces every interval to one
world-space point, and fits one affine height plane with ordinary least squares.  It is
intentionally unable to represent stairs, multiple support objects, seats, collision
holes, or disconnected surfaces.  The finite output patch is only a serialization of
the fitted plane over the observed contact envelope; ground-truth geometry is never read.

This module is independent of TERRA's terrain fitter and stance implementation.  Keeping
the small detector here makes the baseline's evidence and failure modes auditable rather
than silently inheriting TERRA's level, family, offset, or free-space logic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from terra._musclemimic import BoxSpec, TerrainSpec
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS


@dataclass(frozen=True)
class LeastSquaresPlaneConfig:
    """Frozen assumptions for the inferred-contact affine-plane baseline."""

    velocity_threshold_m_s: float = 0.30
    minimum_contact_duration_s: float = 0.10
    local_window_s: float = 0.30
    local_height_tolerance_m: float = 0.05
    footprint_margin_m: float = 0.15
    minimum_footprint_side_m: float = 0.30
    minimum_raised_height_m: float = 0.02
    # BoxSpec's collision representation stops at 60 degrees. Keep the default one
    # degree inside that hard boundary; this is a serialization guard, not an apparatus
    # slope prior.
    maximum_slope_deg: float = 59.0
    buried_depth_m: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            "velocity_threshold_m_s",
            "minimum_contact_duration_s",
            "local_window_s",
            "local_height_tolerance_m",
            "minimum_footprint_side_m",
            "maximum_slope_deg",
            "buried_depth_m",
        )
        nonnegative = ("footprint_margin_m", "minimum_raised_height_m")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, value)
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, value)
        if self.maximum_slope_deg >= 60.0:
            raise ValueError("maximum_slope_deg must be less than 60 degrees")


@dataclass(frozen=True)
class _ContactEvent:
    link: str
    start: int
    end: int
    point: np.ndarray


def _validate_motion(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
) -> tuple[np.ndarray, tuple[str, ...], float, tuple[str, ...]]:
    try:
        motion = np.asarray(joints, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("joints must be a real numeric array with shape (T, J, 3)") from error
    if motion.ndim != 3 or motion.shape[0] < 2 or motion.shape[2] != 3:
        raise ValueError(f"joints must have shape (T>=2, J, 3), got {motion.shape}")
    if not np.all(np.isfinite(motion)):
        raise ValueError("joints must contain only finite values")

    names = tuple(demo_joints)
    if len(names) != motion.shape[1]:
        raise ValueError(f"demo_joints has {len(names)} names for {motion.shape[1]} columns")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("demo_joints must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("demo_joints must not contain duplicates")

    try:
        frame_rate = float(fps)
    except (TypeError, ValueError) as error:
        raise ValueError("fps must be finite and positive") from error
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("fps must be finite and positive")

    return motion, names, frame_rate, DEFAULT_CONTACT_JOINTS


def _true_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True) if end - start >= minimum_length]


def _infer_contact_events(
    joints: np.ndarray,
    names: tuple[str, ...],
    fps: float,
    links: tuple[str, ...],
    config: LeastSquaresPlaneConfig,
) -> tuple[list[_ContactEvent], dict[str, dict[str, object]], list[str]]:
    events: list[_ContactEvent] = []
    diagnostics: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    half_window = max(1, int(config.local_window_s * fps))
    minimum_run_frames = max(1, math.ceil(config.minimum_contact_duration_s * fps - 1e-12))

    for link in links:
        if link not in names:
            warnings.append(f"contact link {link!r} is absent and was skipped")
            diagnostics[link] = {
                "available": False,
                "slow_frames": 0,
                "slow_runs": 0,
                "accepted_runs": 0,
            }
            continue
        points = joints[:, names.index(link)]
        speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
        # Match every frame to one forward-difference estimate while preserving the
        # first sample.  This requires only T>=2 and keeps a motion-start stance visible.
        speed = np.concatenate((speed[:1], speed))
        slow = speed < config.velocity_threshold_m_s
        slow_runs = _true_runs(slow, minimum_run_frames)
        accepted = 0
        for start, end in slow_runs:
            point = np.median(points[start:end], axis=0)
            middle = (start + end) // 2
            lower = max(0, middle - half_window)
            upper = min(len(points), middle + half_window)
            if point[2] > float(points[lower:upper, 2].min()) + config.local_height_tolerance_m:
                continue
            events.append(_ContactEvent(link=link, start=start, end=end, point=point))
            accepted += 1
        diagnostics[link] = {
            "available": True,
            "minimum_run_frames": minimum_run_frames,
            "slow_frames": int(slow.sum()),
            "slow_runs": len(slow_runs),
            "accepted_runs": accepted,
            "speed_min_m_s": float(speed.min()),
            "speed_median_m_s": float(np.median(speed)),
        }

    events.sort(key=lambda event: (event.start, event.end, event.link, *event.point.tolist()))
    return events, diagnostics, warnings


def _expanded_bounds(values: np.ndarray, margin: float, minimum_side: float) -> tuple[float, float]:
    lower = float(values.min()) - margin
    upper = float(values.max()) + margin
    if upper - lower < minimum_side:
        centre = 0.5 * (lower + upper)
        lower = centre - 0.5 * minimum_side
        upper = centre + 0.5 * minimum_side
    return lower, upper


def _plane_box(
    points: np.ndarray,
    centre_xy: np.ndarray,
    gradient: np.ndarray,
    intercept_at_centre: float,
    config: LeastSquaresPlaneConfig,
) -> tuple[BoxSpec | None, dict[str, object]]:
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > 1e-12:
        axis_u = gradient / gradient_norm
    else:
        axis_u = np.array([1.0, 0.0])
    axis_v = np.array([-axis_u[1], axis_u[0]])
    centred = points[:, :2] - centre_xy
    coordinate_u = centred @ axis_u
    coordinate_v = centred @ axis_v
    u0, u1 = _expanded_bounds(
        coordinate_u,
        config.footprint_margin_m,
        config.minimum_footprint_side_m,
    )
    v0, v1 = _expanded_bounds(
        coordinate_v,
        config.footprint_margin_m,
        config.minimum_footprint_side_m,
    )
    middle_u = 0.5 * (u0 + u1)
    middle_v = 0.5 * (v0 + v1)
    top_centre_xy = centre_xy + middle_u * axis_u + middle_v * axis_v
    top_centre_z = float(intercept_at_centre + gradient @ (top_centre_xy - centre_xy))
    half_run = 0.5 * (u1 - u0)
    half_width = 0.5 * (v1 - v0)
    minimum_top = top_centre_z - gradient_norm * half_run
    maximum_top = top_centre_z + gradient_norm * half_run
    geometry = {
        "axis_u_xy": axis_u.tolist(),
        "axis_v_xy": axis_v.tolist(),
        "u_bounds_m": [u0, u1],
        "v_bounds_m": [v0, v1],
        "top_height_range_m": [minimum_top, maximum_top],
        "top_centre_xyz_m": [*top_centre_xy.tolist(), top_centre_z],
    }
    if maximum_top <= config.minimum_raised_height_m:
        return None, geometry | {
            "output": "implicit_flat_floor",
            "fallback_reason": "fitted plane is nowhere above minimum_raised_height_m within its support envelope",
        }

    yaw = math.atan2(float(axis_u[1]), float(axis_u[0]))
    pitch = -math.atan(gradient_norm)
    cosine_pitch = math.cos(pitch)
    half_u_local = half_run / cosine_pitch
    # Make the lower face sit below z=0 even at the plane's highest end.  This preserves
    # TerrainSpec's solid-from-floor convention without altering the fitted top plane.
    half_thickness = (max(maximum_top, 0.0) + config.buried_depth_m) / (2.0 * cosine_pitch)
    normal = np.array(
        [
            math.cos(yaw) * math.sin(pitch),
            math.sin(yaw) * math.sin(pitch),
            cosine_pitch,
        ]
    )
    top_centre = np.array([*top_centre_xy, top_centre_z])
    box_centre = top_centre - half_thickness * normal
    box = BoxSpec(
        pos=tuple(box_centre),
        size=(half_u_local, half_width, half_thickness),
        yaw=yaw,
        pitch=pitch,
        name="terrain_box_least_squares_plane_0000",
    )
    return box, geometry | {
        "output": "one_finite_affine_plane",
        "fallback_reason": None,
        "yaw_rad": yaw,
        "pitch_rad": pitch,
    }


def fit_least_squares_contact_plane(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    *,
    config: LeastSquaresPlaneConfig | None = None,
    input_stage: str = "world_space_motion_landmarks",
) -> tuple[TerrainSpec, dict[str, object]]:
    """Infer four-joint foot contacts and fit one finite affine plane by ordinary least squares.

    Each accepted contact interval contributes exactly one median XYZ point, so a long
    pause has the same weight as one ordinary stance.  The centred least-squares solve is
    deterministic for point- or line-degenerate trajectories through NumPy's minimum-norm
    solution.  A slope cap only prevents an unrepresentable box from being serialized; it
    is reported alongside the uncapped estimate.
    """

    resolved = config or LeastSquaresPlaneConfig()
    if not isinstance(resolved, LeastSquaresPlaneConfig):
        raise TypeError("config must be LeastSquaresPlaneConfig")
    if not isinstance(input_stage, str) or not input_stage.strip():
        raise ValueError("input_stage must be a non-empty string")
    motion, names, frame_rate, links = _validate_motion(joints, demo_joints, fps)
    events, detector, warnings = _infer_contact_events(motion, names, frame_rate, links, resolved)
    support_intervals = [
        {
            "link": event.link,
            "kind": "foot",
            "start": event.start,
            "end": event.end,
            "surface_xyz_m": event.point.tolist(),
            "evidence_stage": "inferred_slow_local_low_before_plane_fit",
        }
        for event in events
    ]
    common_report: dict[str, object] = {
        "model": "least_squares_contact_plane",
        "reconstruction_only": True,
        "input_stage": input_stage,
        "n_frames": len(motion),
        "fps": frame_rate,
        "contact_links": list(links),
        "contact_source": "kinematic_foot_speed_and_local_height",
        "contact_detection": detector,
        "support_intervals": support_intervals,
        "n_contact_events": len(events),
        "evidence_completeness": "foot_only",
        "unsupported_support_kinds": ["pelvis"],
        "evidence_limitation": "The baseline detects feet only and cannot reconstruct seats or pelvis support.",
        "config": asdict(resolved),
        "assumptions": {
            "coordinate_frame": "world_z_up",
            "units": "metres",
            "event_weighting": "one equally weighted median XYZ point per accepted stance interval",
            "surface_model": "one affine plane z = a*x + b*y + c",
            "fit_loss": "unweighted ordinary least squares in vertical height",
            "scene_extent": "oriented inferred-contact envelope plus fixed margin",
            "floor": "TerrainSpec implicit z=0 floor clips portions of the finite plane below zero",
            "anatomical_offsets": "none; calibrated foot landmarks are treated as contact-surface probes",
            "level_clustering": "none",
            "terrain_family_selection": "none",
            "seat_model": "none",
            "collision_carving": "none",
        },
        "warnings": warnings,
    }

    provenance = {
        "source": "fit_least_squares_contact_plane",
        "input_stage": input_stage,
        "config": asdict(resolved),
    }
    if not events:
        report = common_report | {
            "fit": None,
            "geometry": {
                "output": "implicit_flat_floor",
                "fallback_reason": "no inferred contact events",
            },
            "n_boxes": 0,
        }
        return TerrainSpec(provenance=provenance | {"fallback": "no inferred contact events"}), report

    points = np.stack([event.point for event in events])
    centre_xy = points[:, :2].mean(axis=0)
    centred_xy = points[:, :2] - centre_xy
    design = np.column_stack((centred_xy, np.ones(len(points))))
    coefficients, _residual_sum, rank, singular_values = np.linalg.lstsq(design, points[:, 2], rcond=None)
    raw_gradient = coefficients[:2]
    raw_intercept = float(coefficients[2])
    raw_norm = float(np.linalg.norm(raw_gradient))
    cap = math.tan(math.radians(resolved.maximum_slope_deg))
    capped = raw_norm > cap
    gradient = raw_gradient * (cap / raw_norm) if capped else raw_gradient.copy()
    # Centring makes the OLS intercept the mean observed z. Recompute it explicitly after
    # slope clipping so this invariant remains clear even if the design changes later.
    intercept = float(np.mean(points[:, 2] - centred_xy @ gradient))
    predicted = intercept + centred_xy @ gradient
    residuals = points[:, 2] - predicted
    box, geometry = _plane_box(points, centre_xy, gradient, intercept, resolved)
    slope_deg = math.degrees(math.atan(float(np.linalg.norm(gradient))))
    report = common_report | {
        "fit": {
            "equation_center_xy_m": centre_xy.tolist(),
            "gradient_xy": gradient.tolist(),
            "intercept_at_center_m": intercept,
            "slope_deg": slope_deg,
            "raw_gradient_xy": raw_gradient.tolist(),
            "raw_intercept_at_center_m": raw_intercept,
            "raw_slope_deg": math.degrees(math.atan(raw_norm)),
            "slope_was_capped": capped,
            "design_rank": int(rank),
            "design_singular_values": singular_values.tolist(),
            "height_rmse_m": float(np.sqrt(np.mean(np.square(residuals)))),
            "height_mae_m": float(np.mean(np.abs(residuals))),
            "height_residual_max_abs_m": float(np.max(np.abs(residuals))),
            "n_equal_weight_events": len(events),
        },
        "geometry": geometry,
        "n_boxes": int(box is not None),
    }
    terrain = TerrainSpec(
        boxes=() if box is None else (box,),
        provenance=provenance
        | {
            "plane": {
                "center_xy_m": centre_xy.tolist(),
                "gradient_xy": gradient.tolist(),
                "intercept_at_center_m": intercept,
            },
            "geometry": geometry,
        },
    )
    return terrain, report


__all__ = [
    "LeastSquaresPlaneConfig",
    "fit_least_squares_contact_plane",
]
