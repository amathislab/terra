"""NumPy-only statistics shared by fresh evaluations and saved-result reproduction."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

APPARATUS_DATASETS = ("gait120", "darmstadt", "vielemeyer")
RECONSTRUCTION_METHOD_NAMES = (
    "contact-least-squares",
    "voronoi",
    "terra-no-sole-offsets",
    "terra-no-physical-cues",
    "terra",
)
CONSISTENCY_SCHEMA = "terra.reconstruction-consistency.v1"


def _truth(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _stats(rows: list[dict[str, str]], field: str, *, scale: float = 1.0) -> dict[str, float | int | None]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "").strip()
        if not raw:
            continue
        try:
            value = float(raw) * scale
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
        "n": len(values),
    }


def _classification_stats(
    rows: list[dict[str, str]],
    field: str,
    *,
    expected_field: str,
    prediction_field: str,
) -> dict[str, float | int | None]:
    """Score a family-producing representation on every labeled motion."""

    labeled = [row for row in rows if row.get(expected_field, "").strip()]
    if not any(row.get(prediction_field, "").strip() for row in labeled):
        return {"mean": None, "std": None, "n": 0}
    values = [100.0 if _truth(row.get(field, "")) else 0.0 for row in labeled]
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
        "n": len(values),
    }


def _metric_fields(name: str, values: dict[str, float | int | None]) -> dict[str, float | int | None]:
    return {f"{name}_{suffix}": value for suffix, value in values.items()}


def _base_summary(
    dataset: str,
    method: str,
    rows: list[dict[str, str]],
    successful: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "total": len(rows),
    }


def _apparatus_summary(dataset: str, method: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error", "").strip() and _truth(row.get("terrain_available", ""))]
    return (
        _base_summary(dataset, method, rows, successful)
        | _metric_fields(
            "family_accuracy_pct",
            _classification_stats(
                successful,
                "family_correct",
                expected_field="expected_family",
                prediction_field="selected_family",
            ),
        )
        | _metric_fields("ramp_angle_mae_deg", _stats(successful, "slope_abs_error_deg"))
        | _metric_fields("step_height_mae_mm", _stats(successful, "step_contact_height_mae_m", scale=1000.0))
        | _metric_fields("seat_height_mae_mm", _stats(successful, "seat_height_abs_error_m", scale=1000.0))
    )


def _pooled_family_summary(per_motion: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool the labeled apparatus denominator across dataset partitions."""

    result: list[dict[str, Any]] = []
    for method in RECONSTRUCTION_METHOD_NAMES:
        rows = [
            row
            for row in per_motion
            if row.get("benchmark_dataset") in APPARATUS_DATASETS
            and row.get("benchmark_method") == method
            and _truth(row.get("terrain_available", ""))
            and not str(row.get("error", "")).strip()
        ]
        values = _classification_stats(
            rows,
            "family_correct",
            expected_field="expected_family",
            prediction_field="selected_family",
        )
        result.append(
            {
                "method": method,
                "family_accuracy_pct_mean": values["mean"],
                "family_accuracy_pct_std": values["std"],
                "family_accuracy_pct_n": values["n"],
            }
        )
    return result


def _prism_summary(method: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("candidate_fit_status", "").strip().casefold() in {"ok", "cached"}]
    primary_eligible = [row for row in successful if _truth(row.get("primary_eligible", "true"))]
    return (
        _base_summary("prism", method, rows, successful)
        | _metric_fields("foot_height_mae_mm", _stats(successful, "foot_height_mae_mm"))
        | _metric_fields("seated_height_mae_mm", _stats(successful, "seated_height_mae_mm"))
        | _metric_fields(
            "observed_height_mae_mm",
            _stats(primary_eligible, "observed_height_mae_mm"),
        )
        | _metric_fields(
            "raised_terrain_coverage_pct",
            _stats(successful, "raised_terrain_coverage", scale=100.0),
        )
        | _metric_fields(
            "flat_terrain_coverage_pct",
            _stats(successful, "flat_terrain_coverage", scale=100.0),
        )
        | _metric_fields(
            "footprint_iou_pct",
            _stats(successful, "full_footprint_iou", scale=100.0),
        )
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _metric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int | None]:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) not in (None, "") and math.isfinite(float(row[key]))],
        dtype=float,
    )
    if not values.size:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
    }


def _aggregate_method(rows: Sequence[Mapping[str, Any]], total: int) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    counts = {
        key: sum(int(row.get(key) or 0) for row in successful)
        for key in (
            "contact_n",
            "contact_consistent_n",
            "contact_penetrating_n",
            "contact_floating_n",
            "raised_true_positive_n",
            "raised_false_positive_n",
            "raised_false_negative_n",
            "raised_true_negative_n",
            "free_space_query_n",
            "free_space_penetrating_n",
        )
    }
    tp = counts["raised_true_positive_n"]
    fp = counts["raised_false_positive_n"]
    fn = counts["raised_false_negative_n"]
    contact_n = counts["contact_n"]
    free_n = counts["free_space_query_n"]
    metrics = {
        key: _metric(successful, key)
        for key in (
            "contact_height_mae_mm",
            "contact_height_rmse_mm",
            "contact_height_p95_mm",
            "contact_height_max_mm",
            "contact_height_bias_mm",
            "support_pair_height_mae_mm",
            "contact_height_correlation",
            "n_boxes",
            "n_walkable_boxes",
            "free_space_penetrating_pct",
            "free_space_penetration_max_mm",
            "free_space_penetration_mean_positive_mm",
        )
    }
    return {
        "selected": total,
        "successful": len(successful),
        "failed": total - len(successful),
        "failure_pct": 100.0 * (total - len(successful)) / total if total else None,
        "contact_events": contact_n,
        "contact_consistent_pct": _ratio(counts["contact_consistent_n"], contact_n),
        "contact_penetrating_pct": _ratio(counts["contact_penetrating_n"], contact_n),
        "contact_floating_pct": _ratio(counts["contact_floating_n"], contact_n),
        "raised_precision_pct": _ratio(tp, tp + fp),
        "raised_recall_pct": _ratio(tp, tp + fn),
        "raised_f1_pct": _ratio(2 * tp, 2 * tp + fp + fn),
        "raised_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": counts["raised_true_negative_n"]},
        "free_space_queries": free_n,
        "free_space_penetrating_pct_pooled": _ratio(counts["free_space_penetrating_n"], free_n),
        "motions_with_free_space_violation": sum(bool(row.get("free_space_violation")) for row in successful),
        "metrics": metrics,
    }


def summarize(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], total: int) -> dict[str, Any]:
    by_method = {label: [row for row in rows if row["method"] == label] for label in labels}
    common_motions = set.intersection(
        *({str(row["motion"]) for row in by_method[label] if not row.get("error")} for label in labels)
    )
    return {
        "schema": CONSISTENCY_SCHEMA,
        "selected_motions": total,
        "common_success_motions": len(common_motions),
        "methods": {label: _aggregate_method(by_method[label], total) for label in labels},
        "common_success_methods": {
            label: _aggregate_method(
                [row for row in by_method[label] if row["motion"] in common_motions],
                len(common_motions),
            )
            for label in labels
        },
    }
