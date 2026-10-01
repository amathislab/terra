"""Metric definitions for retargeting evaluation.

Every output column names its formula lineage and denominator here. A duplicate key with a
different definition is a scientific conflict, not an alias: callers must select or add an
explicit decision before results can be published.
"""

from __future__ import annotations

from dataclasses import dataclass


class MetricDefinitionConflict(ValueError):  # noqa: N818 - scientific conflict, not runtime failure
    """Raised when two scientifically different formulas claim the same metric key."""


@dataclass(frozen=True)
class MetricSpec:
    """One scalar emitted for each method-motion pair."""

    key: str
    label: str
    unit: str
    decimals: int
    family: str
    formula: str
    denominator: str
    source: str
    aggregation: str = "unweighted mean across finite per-motion values"
    per_frame_key: str | None = None
    higher_is_better: bool = False
    comparison_role: str = "headline"


def _spec(
    key: str,
    label: str,
    unit: str,
    decimals: int,
    family: str,
    formula: str,
    denominator: str,
    source: str,
    **kwargs,
) -> MetricSpec:
    return MetricSpec(key, label, unit, decimals, family, formula, denominator, source, **kwargs)


AUTHORITATIVE_METRICS = (
    _spec(
        "penetration_duration_pct",
        "Environment pen. duration",
        "%",
        2,
        "environment_penetration",
        "time-weighted frames where deepest enabled robot/environment contact exceeds penetration_m",
        "common source-time duration",
        "OmniRetarget evaluate_penetration with its 10 cm broad phase and 10 mm tolerance",
        per_frame_key="environment_penetrating",
    ),
    _spec(
        "penetration_max_depth_mm",
        "Environment pen. max depth",
        "mm",
        2,
        "environment_penetration",
        "maximum robot/environment depth in each penetrating frame, pooled as in OmniRetarget",
        "frames exceeding penetration_m on the common source-time interval",
        "OmniRetarget evaluate_penetration with its exact-distance query and per-frame maximum",
        aggregation="pooled mean and population standard deviation over penetrating-frame maxima",
        per_frame_key="body_pen",
    ),
    _spec(
        "support_penetration_duration_pct",
        "Support pen. duration",
        "%",
        2,
        "support_penetration",
        "source-required support foot-time below its selected walkable support by support_penetration_m",
        "desired left/right source support foot-time",
        "TERRA benchmark foot_contact_metrics",
        per_frame_key="support_penetrating",
    ),
    _spec(
        "support_penetration_max_depth_mm",
        "Support pen. max depth",
        "mm",
        2,
        "support_penetration",
        "deepest corresponding sole depth during source-required support",
        "maximum over desired source support foot-time",
        "TERRA benchmark foot_contact_metrics",
    ),
    _spec(
        "skating_duration_pct",
        "Skating duration",
        "%",
        2,
        "skating_slip",
        "source-sticking time where any mapped robot toe/ankle probe exceeds skating_speed_m_s",
        "time where any of the four source contact probes is required to stick",
        "OmniRetarget detect_foot_sliding duration rule with physical velocity units, the study's 0.30 m/s threshold, and TERRA's four-probe contact source",
        per_frame_key="skating",
    ),
    _spec(
        "skating_max_velocity_m_s",
        "Skating max velocity",
        "m/s",
        3,
        "skating_slip",
        "maximum violating mapped toe/ankle velocity in each skating frame, pooled as in OmniRetarget",
        "skating frames on the common source-time interval",
        "OmniRetarget detect_foot_sliding aggregation with physical velocity units and four mapped contact probes",
        aggregation="pooled mean and population standard deviation over skating-frame maxima",
    ),
    _spec(
        "stance_slip_failure_pct",
        "Stance slip failures",
        "% phases",
        2,
        "skating_slip",
        "source stance phases whose robot travel exceeds source travel by slip_tol",
        "source stance phases wholly contained in the common source-time interval",
        "TERRA terrain Stance.excess phase gate",
        per_frame_key="stance_slip_failure",
        comparison_role="diagnostic",
    ),
    _spec(
        "floating_duration_pct",
        "Floating duration",
        "%",
        2,
        "floating",
        "source-required support foot-time above selected support by foot_contact_height_m",
        "desired left/right source support foot-time",
        "TERRA benchmark foot_contact_metrics",
        per_frame_key="support_floating",
    ),
    _spec(
        "floating_max_height_mm",
        "Floating max height",
        "mm",
        2,
        "floating",
        "maximum corresponding sole gap during source-required support",
        "maximum over desired source support foot-time",
        "TERRA benchmark foot_contact_metrics",
    ),
    _spec(
        "invalid_support_duration_pct",
        "Invalid support duration",
        "%",
        2,
        "support_validity",
        "union of mutually exclusive source-required support penetration and floating masks",
        "desired left/right source support foot-time",
        "TERRA benchmark foot_contact_metrics",
        per_frame_key="invalid_support",
    ),
    _spec(
        "contact_preservation_pct",
        "Contact preservation",
        "%",
        2,
        "contact_preservation",
        "time where every desired mapped hand/ankle/toe terrain contact remains within terrain_contact_m",
        "time with at least one desired source terrain contact; 100% when none exists",
        "OmniRetarget evaluate_terrain_contact_precision with corrected per-frame robot FK",
        higher_is_better=True,
    ),
    _spec(
        "swing_clearance_failure_pct",
        "Swing clearance failures",
        "% phases",
        2,
        "swing_clearance",
        "measurable swings whose like-for-like robot peak is below max(clearance_floor, clearance_frac*source_peak)",
        "source_peak > 0 swings wholly contained in the common source-time interval",
        "TERRA terrain Swing.dragging calibrated phase definition",
        per_frame_key="swing_clearance_failure",
    ),
    _spec(
        "swing_scrape_failure_pct",
        "Swing scrape failures",
        "% phases",
        2,
        "swing_clearance",
        "swings whose exact sole-to-terrain distance enters terrain by scrape_tol",
        "source-derived swing phases wholly contained in the common source-time interval",
        "TERRA terrain Swing.scraping absolute definition",
        per_frame_key="swing_scraping",
        comparison_role="diagnostic",
    ),
    _spec(
        "swing_clearance_ratio_median",
        "Swing clearance/source median",
        "ratio",
        3,
        "swing_clearance",
        "median robot/source like-for-like peak-clearance ratio",
        "finite positive source-peak swings wholly contained in the common source-time interval",
        "TERRA terrain Swing peak/src_peak",
        higher_is_better=True,
        comparison_role="diagnostic",
    ),
    _spec(
        "joint_limit_duration_pct",
        "Joint-limit duration",
        "%",
        2,
        "joint_limit_validity",
        "time-weighted frames with a limited hinge outside range by joint_limit or a limited slide outside range by joint_limit_linear_m",
        "common source-time duration",
        "TERRA benchmark angular and linear joint-limit excess",
        per_frame_key="joint_limit_bad",
    ),
    _spec(
        "joint_limit_max_excess_deg",
        "Joint-limit max excess",
        "deg",
        3,
        "joint_limit_validity",
        "maximum angular range excess over limited hinge joints",
        "maximum over common source-time samples and hinge joints",
        "TERRA benchmark joint_limit_excess_rad",
    ),
    _spec(
        "coupler_mean_residual_deg",
        "Coupler mean residual",
        "deg",
        3,
        "coupler_validity",
        "mean absolute dependent-minus-polynomial(independent) joint equality residual",
        "all configured joint couplers and common source-time frames",
        "MuscleMimic msk_metrics.coupler_residual_stats",
    ),
    _spec(
        "coupler_max_residual_deg",
        "Coupler max residual",
        "deg",
        3,
        "coupler_validity",
        "maximum absolute joint equality residual",
        "maximum over configured joint couplers and common source-time frames",
        "MuscleMimic msk_metrics.coupler_residual_stats",
    ),
    _spec(
        "tendon_jump_duration_pct",
        "Tendon-jump duration",
        "%",
        3,
        "tendon_discontinuity",
        "time-weighted active frame transitions with adaptive relative tendon event above tendon_jump",
        "common source-time frame-transition duration",
        "TERRA benchmark adaptive tendon event definition",
        per_frame_key="tendon_jump_bad",
    ),
    _spec(
        "tendon_max_jump",
        "Tendon max jump",
        "relative",
        4,
        "tendon_discontinuity",
        "maximum adaptive relative tendon event above tendon_jump",
        "maximum over active common-time transitions",
        "MuscleMimic tendon_stats_and_events with the configured tendon_jump threshold",
    ),
    _spec(
        "tendon_max_step_mm",
        "Tendon max physical step",
        "mm/frame",
        3,
        "tendon_discontinuity",
        "maximum absolute physical tendon-length change",
        "maximum over active common-time transitions and tendons",
        "MuscleMimic tendon_stats_and_events tendon_max_mm",
        comparison_role="diagnostic",
    ),
    _spec(
        "joint_step_max_deg",
        "Joint step max",
        "deg/frame",
        2,
        "frame_step_discontinuity",
        "maximum absolute hinge qpos frame step; free/ball joints excluded",
        "maximum over active common-time transitions and hinge joints",
        "MuscleMimic msk_metrics.qpos_step_stats",
        per_frame_key="joint_step_deg",
    ),
    _spec(
        "joint_step_p999_deg",
        "Joint step p99.9",
        "deg/frame",
        3,
        "frame_step_discontinuity",
        "99.9th percentile absolute hinge qpos frame step; free/ball joints excluded",
        "active common-time transitions and hinge joints",
        "MuscleMimic msk_metrics.qpos_step_stats",
        comparison_role="diagnostic",
    ),
    _spec(
        "root_translation_step_max_mm",
        "Root translation step max",
        "mm/frame",
        2,
        "frame_step_discontinuity",
        "maximum free-root Euclidean translation step",
        "maximum over active common-time transitions",
        "MuscleMimic msk_metrics.root_step_series",
        per_frame_key="root_translation_step_mm",
        comparison_role="diagnostic",
    ),
    _spec(
        "root_translation_step_p999_mm",
        "Root translation step p99.9",
        "mm/frame",
        3,
        "frame_step_discontinuity",
        "99.9th percentile free-root Euclidean translation step",
        "active common-time transitions",
        "MuscleMimic msk_metrics.root_step_stats",
        comparison_role="diagnostic",
    ),
    _spec(
        "root_rotation_step_max_deg",
        "Root rotation step max",
        "deg/frame",
        2,
        "frame_step_discontinuity",
        "maximum normalized quaternion geodesic root rotation step with q/-q equivalence",
        "maximum over active common-time transitions",
        "MuscleMimic msk_metrics.root_step_series",
        per_frame_key="root_rotation_step_deg",
        comparison_role="diagnostic",
    ),
    _spec(
        "root_rotation_step_p999_deg",
        "Root rotation step p99.9",
        "deg/frame",
        3,
        "frame_step_discontinuity",
        "99.9th percentile normalized quaternion geodesic root rotation step with q/-q equivalence",
        "active common-time transitions",
        "MuscleMimic msk_metrics.root_step_stats",
        comparison_role="diagnostic",
    ),
    _spec(
        "self_collision_duration_pct",
        "Inter-leg collision duration",
        "%",
        2,
        "self_collision",
        "time-weighted frames with constrained left/right leg-pair depth above self_collision_m",
        "common source-time duration",
        "TERRA benchmark benchmark_self_penetration",
        per_frame_key="self_collision_bad",
    ),
    _spec(
        "self_collision_max_depth_mm",
        "Inter-leg collision max depth",
        "mm",
        2,
        "self_collision",
        "maximum constrained left/right leg-pair interpenetration",
        "maximum over common source-time frames",
        "TERRA terrain Measurement.selfpen_worst",
    ),
    _spec(
        "pelvis_world_rmse_mm",
        "Pelvis world RMSE",
        "mm",
        2,
        "fidelity",
        "RMS Euclidean source/robot pelvis error at declared timestamps",
        "active common source-time frames",
        "TERRA benchmark common_retargeting_rmse; no rigid or temporal alignment search",
    ),
    _spec(
        "pelvis_relative_landmark_rmse_mm",
        "Pelvis-relative landmark RMSE",
        "mm",
        2,
        "fidelity",
        "RMS Euclidean error of 17 non-pelvis mapped landmarks after per-frame pelvis subtraction",
        "active common source-time frames and mapped non-pelvis landmarks",
        "TERRA benchmark common_retargeting_rmse; no rigid or temporal alignment search",
    ),
    _spec(
        "solver_native_fps",
        "Solver throughput",
        "solved frames/s",
        2,
        "solver_throughput",
        "producer-native solved frames divided by the producer-recorded retarget duration",
        "producer-recorded retarget timing",
        "retargeter analysis metadata",
        higher_is_better=True,
        comparison_role="diagnostic",
    ),
    _spec(
        "solver_seconds_per_motion_second",
        "Solver cost per motion second",
        "s/s",
        3,
        "solver_throughput",
        "producer-recorded retarget duration divided by common motion duration",
        "common source-time motion seconds",
        "TERRA benchmark solver_throughput",
    ),
    _spec(
        "t_frame_s",
        "T_frame",
        "s/solved frame",
        4,
        "solver_throughput",
        "producer-recorded retarget duration divided by producer-native solved frames",
        "one producer-native solved frame",
        "retargeter analysis metadata",
    ),
)


def build_registry(specs=AUTHORITATIVE_METRICS) -> dict[str, MetricSpec]:
    """Build a key registry and reject ambiguous duplicate scientific definitions."""
    registry: dict[str, MetricSpec] = {}
    for spec in specs:
        previous = registry.get(spec.key)
        if previous is not None and previous != spec:
            raise MetricDefinitionConflict(
                f"metric {spec.key!r} has conflicting definitions from {previous.source!r} and {spec.source!r}"
            )
        registry[spec.key] = spec
    return registry


METRIC_REGISTRY = build_registry()

REQUIRED_FAMILIES = frozenset(
    {
        "environment_penetration",
        "support_penetration",
        "contact_preservation",
        "floating",
        "skating_slip",
        "swing_clearance",
        "joint_limit_validity",
        "coupler_validity",
        "tendon_discontinuity",
        "frame_step_discontinuity",
        "self_collision",
        "fidelity",
        "solver_throughput",
    }
)
