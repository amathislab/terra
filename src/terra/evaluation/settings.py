"""Shared thresholds and aggregation settings for TERRA evaluation."""

from __future__ import annotations

from typing import Any

BENCHMARK_THRESHOLDS: dict[str, float] = {
    # OmniRetarget evaluates strict intersections deeper than 1 cm.
    "penetration_m": 0.010,
    "support_penetration_m": 0.005,
    "foot_contact_height_m": 0.020,
    # The shared four-joint kinematic contact detector uses TERRA's default.
    "source_contact_speed_m_s": 0.300,
    # Retain the established study threshold after converting displacement to a
    # physical horizontal velocity.  A 0.01 m/s threshold classifies normal planted-
    # foot motion and solver noise as skating in practice.
    "skating_speed_m_s": 0.300,
    "terrain_contact_m": 0.100,
    "joint_limit": 1e-5,
    "joint_limit_linear_m": 1e-5,
    "tendon_jump": 0.050,
    "self_collision_m": 0.001,
}
QUALITY_THRESHOLDS: dict[str, float] = {
    "float_tol": 0.02,
    "pen_tol": 0.005,
    "clearance_frac": 0.5,
    "clearance_floor": 0.010,
    "slip_tol": 0.03,
    "slip_liftoff_s": 0.05,
    "selfpen_tol": 0.005,
    "scrape_tol": 0.005,
    "forefoot_tol": 0.02,
    "hindfoot_tol": 0.02,
    "min_stance_s": 0.15,
    "seat_float_tol": 0.05,
    "seat_pen_tol": 0.02,
}

UNIFIED_TIMELINE: dict[str, str] = {
    "clock": "declared source timestamps and declared retargeted output timestamps",
    "alignment": "fixed common source-time intersection across requested methods",
    "shift_search": "none",
    "duration_weighting": "sample-interval weighted",
    "phase_boundary_policy": "include only complete source-derived phases wholly inside the common interval",
}
UNIFIED_AGGREGATION: dict[str, str] = {
    "unit": "motion",
    "across_motions": (
        "unweighted arithmetic mean and population standard deviation, except OmniRetarget "
        "penetration-depth and skating-velocity observations pooled over eligible frames"
    ),
    "uncertainty": "sample standard error retained in summary.csv",
    "within_motion_durations": "sample-interval weighted percentages",
    "undefined_values": "omitted per metric with finite sample count reported",
    "phase_metrics": "per-motion ratios over eligible complete phases, then unweighted across motions",
}
UNIFIED_THRESHOLDS: dict[str, Any] = {
    "continuous": BENCHMARK_THRESHOLDS,
    "phase": QUALITY_THRESHOLDS,
}

__all__ = [
    "BENCHMARK_THRESHOLDS",
    "QUALITY_THRESHOLDS",
    "UNIFIED_AGGREGATION",
    "UNIFIED_THRESHOLDS",
    "UNIFIED_TIMELINE",
]
