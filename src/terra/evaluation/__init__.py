"""Metrics and reporting, loaded lazily so numerical summaries need no simulator."""

from importlib import import_module

_EXPORTS = {
    "DIAGNOSTIC_FIELDS": ("terra.evaluation.evaluator", "DIAGNOSTIC_FIELDS"),
    "IDENTITY_FIELDS": ("terra.evaluation.evaluator", "IDENTITY_FIELDS"),
    "JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD": ("terra.evaluation.evaluator", "JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD"),
    "PER_MOTION_FIELDS": ("terra.evaluation.evaluator", "PER_MOTION_FIELDS"),
    "Thresholds": ("terra.evaluation.evaluator", "Thresholds"),
    "UnifiedMeasurement": ("terra.evaluation.evaluator", "UnifiedMeasurement"),
    "benchmark_self_penetration": ("terra.evaluation.evaluator", "benchmark_self_penetration"),
    "common_retargeting_rmse": ("terra.evaluation.evaluator", "common_retargeting_rmse"),
    "coupler_metrics": ("terra.evaluation.evaluator", "coupler_metrics"),
    "evaluate_method_motion": ("terra.evaluation.evaluator", "evaluate_method_motion"),
    "floor_contact_preservation": ("terra.evaluation.evaluator", "floor_contact_preservation"),
    "foot_contact_metrics": ("terra.evaluation.evaluator", "foot_contact_metrics"),
    "frame_step_metrics": ("terra.evaluation.evaluator", "frame_step_metrics"),
    "joint_limit_excess_rad": ("terra.evaluation.evaluator", "joint_limit_excess_rad"),
    "joint_limit_excess_m": ("terra.evaluation.evaluator", "joint_limit_excess_m"),
    "joint_limit_frame_mask": ("terra.evaluation.evaluator", "joint_limit_frame_mask"),
    "load_site_calibration_state": ("terra.evaluation.evaluator", "load_site_calibration_state"),
    "load_timeline": ("terra.evaluation.evaluator", "load_timeline"),
    "robot_body_positions": ("terra.evaluation.evaluator", "robot_body_positions"),
    "robot_point_positions": ("terra.evaluation.evaluator", "robot_point_positions"),
    "source_contact_mask": ("terra.evaluation.evaluator", "source_contact_mask"),
    "source_contact_on_output": ("terra.evaluation.evaluator", "source_contact_on_output"),
    "source_floor_contact_on_output": ("terra.evaluation.evaluator", "source_floor_contact_on_output"),
    "source_probe_contact_on_output": ("terra.evaluation.evaluator", "source_probe_contact_on_output"),
    "terrain_contact_preservation": ("terra.evaluation.evaluator", "terrain_contact_preservation"),
    "METRICS": ("terra.evaluation.registry", "METRICS"),
    "METRIC_REGISTRY": ("terra.evaluation.registry", "METRIC_REGISTRY"),
    "REQUIRED_FAMILIES": ("terra.evaluation.registry", "REQUIRED_FAMILIES"),
    "MetricDefinitionConflict": ("terra.evaluation.registry", "MetricDefinitionConflict"),
    "MetricSpec": ("terra.evaluation.registry", "MetricSpec"),
    "build_registry": ("terra.evaluation.registry", "build_registry"),
    "BENCHMARK_THRESHOLDS": ("terra.evaluation.settings", "BENCHMARK_THRESHOLDS"),
    "QUALITY_THRESHOLDS": ("terra.evaluation.settings", "QUALITY_THRESHOLDS"),
    "UNIFIED_AGGREGATION": ("terra.evaluation.settings", "UNIFIED_AGGREGATION"),
    "UNIFIED_THRESHOLDS": ("terra.evaluation.settings", "UNIFIED_THRESHOLDS"),
    "UNIFIED_TIMELINE": ("terra.evaluation.settings", "UNIFIED_TIMELINE"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BENCHMARK_THRESHOLDS",
    "DIAGNOSTIC_FIELDS",
    "IDENTITY_FIELDS",
    "JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD",
    "METRICS",
    "METRIC_REGISTRY",
    "PER_MOTION_FIELDS",
    "QUALITY_THRESHOLDS",
    "REQUIRED_FAMILIES",
    "UNIFIED_AGGREGATION",
    "UNIFIED_THRESHOLDS",
    "UNIFIED_TIMELINE",
    "MetricDefinitionConflict",
    "MetricSpec",
    "Thresholds",
    "UnifiedMeasurement",
    "benchmark_self_penetration",
    "build_registry",
    "common_retargeting_rmse",
    "coupler_metrics",
    "evaluate_method_motion",
    "floor_contact_preservation",
    "foot_contact_metrics",
    "frame_step_metrics",
    "joint_limit_excess_m",
    "joint_limit_excess_rad",
    "joint_limit_frame_mask",
    "load_site_calibration_state",
    "load_timeline",
    "robot_body_positions",
    "robot_point_positions",
    "source_contact_mask",
    "source_contact_on_output",
    "source_floor_contact_on_output",
    "source_probe_contact_on_output",
    "terrain_contact_preservation",
]
