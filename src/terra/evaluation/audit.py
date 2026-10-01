"""Independent consistency checks for retargeting benchmark rows."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

AUDIT_SCHEMA = "terra.retarget-metric-audit"
PERCENTAGE_ATOL = 1e-9

PERCENTAGE_KEYS = (
    "penetration_duration_pct",
    "skating_duration_pct",
    "floating_duration_pct",
    "contact_preservation_pct",
    "joint_limit_duration_pct",
    "tendon_jump_duration_pct",
    "self_collision_duration_pct",
    "support_penetration_duration_pct",
    "invalid_support_duration_pct",
)
NONNEGATIVE_KEYS = (
    "penetration_max_depth_mm",
    "skating_max_velocity_m_s",
    "floating_max_height_mm",
    "joint_limit_max_excess_deg",
    "tendon_max_jump",
    "self_collision_max_depth_mm",
    "support_penetration_max_depth_mm",
    "t_frame_s",
)
OPTIONAL_METRIC_DENOMINATORS = {
    "skating_duration_pct": "source_contact_s",
    "skating_max_velocity_m_s": "source_contact_s",
    "floating_duration_pct": "source_support_s",
    "floating_max_height_mm": "source_support_s",
    "support_penetration_duration_pct": "source_support_s",
    "support_penetration_max_depth_mm": "source_support_s",
    "invalid_support_duration_pct": "source_support_s",
    "contact_preservation_pct": "desired_contact_point_s",
}
POOLED_IDENTITIES = (
    (
        "penetration_max_depth_mm",
        "penetration_frame_depth_sum_mm",
        "penetration_frame_depth_sq_sum_mm2",
        "penetration_frame_depth_n",
    ),
    (
        "skating_max_velocity_m_s",
        "skating_frame_velocity_sum_m_s",
        "skating_frame_velocity_sq_sum_m2_s2",
        "skating_frame_velocity_n",
    ),
)
DENOMINATOR_TRANSITIONS = {
    "source_contact_s": "source_contact_transitions",
    "source_support_s": "source_support_transitions",
    "desired_contact_point_s": "desired_contact_transitions",
}
TIMING_FIELDS = (
    "cpu_model",
    "cpu_request",
    "worker_processes",
    "threads_per_worker",
    "omp_num_threads",
    "mkl_num_threads",
    "openblas_num_threads",
    "xla_flags",
)


def _identity(row: dict[str, Any]) -> str:
    return f"{row.get('method', '?')}/{row.get('motion_class', '?')}/{row.get('motion', '?')}"


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"{key} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} is not finite")
    return number


def _close(first: float, second: float, *, atol: float = 1e-9) -> bool:
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=atol)


def _metric_number(row: dict[str, Any], key: str) -> float | None:
    """Return a metric value, allowing undefined values only with no eligible data."""

    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"{key} is missing")
    number = float(value)
    if math.isfinite(number):
        return number
    denominator_key = OPTIONAL_METRIC_DENOMINATORS.get(key)
    if denominator_key is not None and _number(row, denominator_key) == 0.0:
        return None
    raise ValueError(f"{key} is not finite")


def _row_issues(row: dict[str, Any]) -> list[str]:
    if str(row.get("error", "")).strip():
        return []
    issues: list[str] = []

    for key in PERCENTAGE_KEYS:
        try:
            value = _metric_number(row, key)
            if value is None:
                continue
            if value < -PERCENTAGE_ATOL or value > 100.0 + PERCENTAGE_ATOL:
                issues.append(f"{key}={value:g} is outside [0, 100]")
        except (TypeError, ValueError) as error:
            issues.append(str(error))
    for key in NONNEGATIVE_KEYS:
        try:
            value = _metric_number(row, key)
            if value is None:
                continue
            if value < 0.0:
                issues.append(f"{key}={value:g} is negative")
        except (TypeError, ValueError) as error:
            issues.append(str(error))

    try:
        frames = _number(row, "frames")
        fps = _number(row, "fps")
        common_duration = _number(row, "common_duration_s")
        if frames < 2 or not frames.is_integer():
            issues.append(f"frames={frames:g} is not an integer >= 2")
        if fps <= 0:
            issues.append(f"fps={fps:g} is not positive")
        if common_duration <= 0:
            issues.append(f"common_duration_s={common_duration:g} is not positive")
    except (TypeError, ValueError) as error:
        issues.append(str(error))
        common_duration = float("nan")

    for key in DENOMINATOR_TRANSITIONS:
        try:
            if _number(row, key) < 0.0:
                issues.append(f"{key} is negative")
        except (TypeError, ValueError) as error:
            issues.append(str(error))

    try:
        t_frame = _number(row, "t_frame_s")
        native_fps = _number(row, "solver_native_fps")
        if not _close(t_frame * native_fps, 1.0, atol=1e-7):
            issues.append("t_frame_s is not the inverse of solver_native_fps")
        solved_frames = _number(row, "solver_solved_frames")
        normalized_cost = _number(row, "solver_seconds_per_motion_second")
        expected_cost = solved_frames * t_frame / common_duration
        if not _close(normalized_cost, expected_cost, atol=1e-7):
            issues.append("solver_seconds_per_motion_second disagrees with solved frames and T_frame")
    except (TypeError, ValueError) as error:
        issues.append(str(error))

    for metric, sum_key, square_key, count_key in POOLED_IDENTITIES:
        try:
            total = _number(row, sum_key)
            square_total = _number(row, square_key)
            count = _number(row, count_key)
            if count < 0 or not count.is_integer():
                issues.append(f"{count_key}={count:g} is not a non-negative integer")
                continue
            if total < 0 or square_total < 0:
                issues.append(f"{metric} sufficient statistics are negative")
                continue
            value = _metric_number(row, metric)
            if count == 0:
                if value is None:
                    if not (_close(total, 0.0) and _close(square_total, 0.0)):
                        issues.append(f"{metric} has observations without eligible source data")
                elif not (_close(value, 0.0) and _close(total, 0.0) and _close(square_total, 0.0)):
                    issues.append(f"{metric} has non-zero values with no pooled observations")
            else:
                if value is None:
                    issues.append(f"{metric} is undefined with pooled observations")
                    continue
                if not _close(value, total / count, atol=1e-7):
                    issues.append(f"{metric} is not the mean of its pooled frame observations")
                if square_total + 1e-9 < total * total / count:
                    issues.append(f"{metric} pooled square sum violates non-negative variance")
        except (TypeError, ValueError) as error:
            issues.append(str(error))

    if _number(row, "source_support_s") > 0.0:
        try:
            invalid = _number(row, "invalid_support_duration_pct")
            floating = _number(row, "floating_duration_pct")
            penetrating = _number(row, "support_penetration_duration_pct")
            if not _close(invalid, floating + penetrating, atol=1e-7):
                issues.append("invalid support is not floating duration plus support penetration duration")
        except (TypeError, ValueError) as error:
            issues.append(str(error))
    return issues


def audit_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check per-row formulas and cross-method common-time denominators."""

    issues: list[str] = []
    successful = [row for row in rows if not str(row.get("error", "")).strip()]
    for row in successful:
        issues.extend(f"{_identity(row)}: {issue}" for issue in _row_issues(row))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        groups[(str(row.get("motion_class", "")), str(row.get("motion", "")))].append(row)
    for (motion_class, motion), group in groups.items():
        if len(group) < 2:
            continue
        reference = group[0]
        for row in group[1:]:
            for key in ("common_start_s", "common_end_s", "common_duration_s"):
                try:
                    if not _close(_number(reference, key), _number(row, key), atol=1e-9):
                        issues.append(f"{motion_class}/{motion}: methods disagree on {key}")
                except (TypeError, ValueError) as error:
                    issues.append(f"{motion_class}/{motion}: {error}")
            try:
                reference_fps = _number(reference, "fps")
                row_fps = _number(row, "fps")
                fallback_tolerance = 2.0 / min(reference_fps, row_fps)
                for key, transition_key in DENOMINATOR_TRANSITIONS.items():
                    try:
                        transitions = max(_number(reference, transition_key), _number(row, transition_key))
                        if transitions < 0 or not transitions.is_integer():
                            raise ValueError(f"{transition_key} is not a non-negative integer")
                        # A Boolean step function's sampled integral can differ from its
                        # exact integral by at most half a sample interval per transition
                        # and boundary. Sum the bounds for the two output grids.
                        sampling_tolerance = (transitions + 1.0) * 0.5 * (
                            1.0 / reference_fps + 1.0 / row_fps
                        )
                    except (TypeError, ValueError):
                        sampling_tolerance = fallback_tolerance
                    if abs(_number(reference, key) - _number(row, key)) > sampling_tolerance + 1e-9:
                        issues.append(f"{motion_class}/{motion}: methods disagree on source-derived {key}")
            except (TypeError, ValueError) as error:
                issues.append(f"{motion_class}/{motion}: {error}")

    issues = list(dict.fromkeys(issues))
    return {
        "schema": AUDIT_SCHEMA,
        "passed": not issues,
        "rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(rows) - len(successful),
        "issues": issues,
    }


def audit_timing_contexts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit uniform timings, or identify an all-legacy producer-timing report."""

    issues: list[str] = []
    warnings: list[str] = []
    contexts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    legacy_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("error", "")).strip():
            continue
        raw = row.get("timing_context_json", "")
        try:
            context = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            context = None
        if not isinstance(context, dict):
            if str(row.get("timing_source", "")).strip() == "uniform_retarget_call":
                issues.append(f"{_identity(row)}: uniform benchmark timing context is missing")
                continue
            try:
                if _number(row, "t_frame_s") <= 0 or _number(row, "solver_native_fps") <= 0:
                    raise ValueError("producer timing is not positive")
                legacy_rows.append(row)
            except (TypeError, ValueError) as error:
                issues.append(f"{_identity(row)}: benchmark timing context is missing and {error}")
            continue
        missing = [key for key in TIMING_FIELDS if key not in context]
        if missing:
            issues.append(f"{_identity(row)}: timing context is missing {', '.join(missing)}")
            continue
        try:
            integers = {
                key: int(context[key])
                for key in TIMING_FIELDS
                if key not in {"cpu_model", "xla_flags"}
            }
            if any(value < 1 for value in integers.values()):
                raise ValueError("timing allocation values must be positive")
            threads = integers["threads_per_worker"]
            if any(integers[key] != threads for key in ("omp_num_threads", "mkl_num_threads", "openblas_num_threads")):
                raise ValueError("BLAS/OpenMP thread counts disagree with threads_per_worker")
            if integers["worker_processes"] * threads > integers["cpu_request"]:
                raise ValueError("worker/thread allocation oversubscribes requested CPUs")
            if not str(context["cpu_model"]).strip():
                raise ValueError("cpu_model is empty")
            eigen_threads = "false" if threads == 1 else "true"
            expected_xla = (
                f"--xla_cpu_multi_thread_eigen={eigen_threads} "
                f"intra_op_parallelism_threads={threads}"
            )
            if str(context["xla_flags"]).strip() != expected_xla:
                raise ValueError("XLA CPU thread controls disagree with threads_per_worker")
        except (TypeError, ValueError) as error:
            issues.append(f"{_identity(row)}: {error}")
            continue
        contexts.append((row, {key: context[key] for key in TIMING_FIELDS}))

    distinct = {json.dumps(context, sort_keys=True) for _row, context in contexts}
    if contexts and legacy_rows:
        issues.append("uniform benchmark timings and legacy producer timings cannot be mixed")
    if len(distinct) > 1:
        issues.append("successful method-motion rows were produced with different timing allocations")
    if legacy_rows and not contexts:
        warnings.append(
            "T_frame uses producer-recorded retarget_fps; timer scopes and compute allocations "
            "are not standardized across methods"
        )
    mode = "uniform_retarget_call" if contexts and not legacy_rows else "producer_retarget_fps"
    if not contexts and not legacy_rows:
        mode = "missing"
        issues.append("no successful rows contain usable timing data")
    return {
        "schema": AUDIT_SCHEMA,
        "passed": not issues,
        "mode": mode,
        "warnings": warnings,
        "successful_rows_with_context": len(contexts),
        "successful_rows_with_legacy_timing": len(legacy_rows),
        "distinct_contexts": [json.loads(value) for value in sorted(distinct)],
        "issues": issues,
    }


__all__ = ["AUDIT_SCHEMA", "audit_metric_rows", "audit_timing_contexts"]
