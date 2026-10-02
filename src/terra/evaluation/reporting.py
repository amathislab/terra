"""Aggregation and table views over unified per-motion metric records."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from terra.evaluation.evaluator import JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD, joint_limit_sensitivity_key
from terra.evaluation.registry import METRICS, MetricSpec

INTERACTION_FAMILIES = {
    "environment_penetration",
    "support_penetration",
    "skating_slip",
    "floating",
    "support_validity",
    "contact_preservation",
    "swing_clearance",
}
TABLE_SECTIONS = (
    (
        "Terrain interaction and contact",
        tuple(spec for spec in METRICS if spec.family in INTERACTION_FAMILIES),
    ),
    (
        "Biomechanical constraints, fidelity, discontinuity, and runtime",
        tuple(spec for spec in METRICS if spec.family not in INTERACTION_FAMILIES),
    ),
)
HEADLINE_METRIC_KEYS = (
    "penetration_duration_pct",
    "penetration_max_depth_mm",
    "skating_duration_pct",
    "skating_max_velocity_m_s",
    "floating_duration_pct",
    "floating_max_height_mm",
    "contact_preservation_pct",
    "joint_limit_duration_pct",
    "tendon_jump_duration_pct",
    "self_collision_duration_pct",
    "t_frame_s",
)
HEADLINE_METRICS = {
    spec.key: spec for spec in METRICS if spec.key in HEADLINE_METRIC_KEYS
}
HEADLINE_TABLE_SECTIONS = (
    (
        "Terrain interaction and contact",
        tuple(
            HEADLINE_METRICS[key]
            for key in (
                "penetration_duration_pct",
                "penetration_max_depth_mm",
                "skating_duration_pct",
                "skating_max_velocity_m_s",
                "floating_duration_pct",
                "floating_max_height_mm",
                "contact_preservation_pct",
            )
        ),
    ),
    (
        "Biomechanical constraints and runtime",
        tuple(
            HEADLINE_METRICS[key]
            for key in (
                "joint_limit_duration_pct",
                "tendon_jump_duration_pct",
                "self_collision_duration_pct",
                "t_frame_s",
            )
        ),
    ),
)

POOLED_OBSERVATION_METRICS = {
    "penetration_max_depth_mm": (
        "penetration_frame_depth_sum_mm",
        "penetration_frame_depth_sq_sum_mm2",
        "penetration_frame_depth_n",
    ),
    "skating_max_velocity_m_s": (
        "skating_frame_velocity_sum_m_s",
        "skating_frame_velocity_sq_sum_m2_s2",
        "skating_frame_velocity_n",
    ),
}


def mean_sem(values) -> tuple[float, float, int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    return (
        float(values.mean()),
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        len(values),
    )


def mean_std(values) -> tuple[float, float, int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    return float(values.mean()), float(values.std(ddof=0)), len(values)


def pooled_mean_std_sem(rows: list[dict], fields: tuple[str, str, str]) -> tuple[float, float, float, int]:
    """Aggregate upstream OmniRetarget per-frame observations from sufficient statistics."""
    sum_field, square_field, count_field = fields
    count = sum(int(float(row.get(count_field, 0) or 0)) for row in rows)
    if count == 0:
        return 0.0, 0.0, 0.0, 0
    total = sum(float(row.get(sum_field, 0) or 0) for row in rows)
    square_total = sum(float(row.get(square_field, 0) or 0) for row in rows)
    mean = total / count
    centered = max(0.0, square_total - count * mean * mean)
    population_std = float(np.sqrt(centered / count))
    sample_sem = float(np.sqrt(centered / (count - 1)) / np.sqrt(count)) if count > 1 else 0.0
    return mean, population_std, sample_sem, count


def summary_fields(metrics: tuple[MetricSpec, ...] = METRICS) -> tuple[str, ...]:
    return (
        "method",
        "motion_class",
        "n_motions",
        "n_errors",
        *(
            field
            for spec in metrics
            for field in (f"{spec.key}_mean", f"{spec.key}_std", f"{spec.key}_sem", f"{spec.key}_n")
        ),
    )


SUMMARY_FIELDS = summary_fields()


def aggregate(rows: list[dict], methods, classes) -> list[dict]:
    """Unweighted per-motion summaries with metric-specific finite denominators."""
    out = []
    for class_label, _manifest in classes:
        for method_label, _subdir in methods:
            requested = [row for row in rows if row["method"] == method_label and row["motion_class"] == class_label]
            valid = [row for row in requested if not row["error"]]
            summary = {
                "method": method_label,
                "motion_class": class_label,
                "n_motions": len(valid),
                "n_errors": len(requested) - len(valid),
            }
            for spec in METRICS:
                if spec.key in POOLED_OBSERVATION_METRICS:
                    mean, std, sem, n = pooled_mean_std_sem(valid, POOLED_OBSERVATION_METRICS[spec.key])
                else:
                    values = [row[spec.key] for row in valid if row.get(spec.key) not in (None, "")]
                    mean, std, n = mean_std(values)
                    _mean, sem, _n = mean_sem(values)
                summary.update(
                    {
                        f"{spec.key}_mean": mean,
                        f"{spec.key}_std": std,
                        f"{spec.key}_sem": sem,
                        f"{spec.key}_n": n,
                    }
                )
            out.append(summary)
    return out


def aggregate_joint_limit_sensitivity(rows: list[dict], methods, classes) -> list[dict]:
    out = []
    for class_label, _manifest in classes:
        for method_label, _subdir in methods:
            valid = [
                row
                for row in rows
                if row["method"] == method_label and row["motion_class"] == class_label and not row["error"]
            ]
            for tolerance in JOINT_LIMIT_SENSITIVITY_TOLERANCES_RAD:
                values = [
                    row[joint_limit_sensitivity_key(tolerance)]
                    for row in valid
                    if row.get(joint_limit_sensitivity_key(tolerance)) not in (None, "")
                ]
                mean, std, n = mean_std(values)
                _mean, sem, _n = mean_sem(values)
                out.append(
                    {
                        "method": method_label,
                        "motion_class": class_label,
                        "tolerance_rad": tolerance,
                        "tolerance_deg": float(np.degrees(tolerance)),
                        "duration_pct_mean": mean,
                        "duration_pct_std": std,
                        "duration_pct_sem": sem,
                        "n_motions": n,
                    }
                )
    return out


def write_joint_limit_sensitivity(out: Path, rows: list[dict]) -> None:
    fields = (
        "method",
        "motion_class",
        "tolerance_rad",
        "tolerance_deg",
        "duration_pct_mean",
        "duration_pct_std",
        "duration_pct_sem",
        "n_motions",
    )
    with (out / "joint_limit_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(mean: object, spread: object, decimals: int, *, latex: bool = False) -> str:
    try:
        mean, spread = float(mean), float(spread)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(mean):
        return "--"
    separator = r" $\pm$ " if latex else " ± "
    return f"{mean:.{decimals}f}{separator}{spread:.{decimals}f}"


def _success_count(row: dict) -> str:
    successful = int(float(row["n_motions"]))
    failed = int(float(row.get("n_errors", 0) or 0))
    return f"{successful}/{successful + failed}"


def markdown_table(
    summary: list[dict],
    *,
    sections=TABLE_SECTIONS,
    group_label: str = "Motion class",
) -> str:
    lines = []
    for section_index, (title, metrics) in enumerate(sections, 1):
        if lines:
            lines.append("")
        headers = [
            group_label,
            "Method",
            "Success/total",
            *(f"{spec.label} ({spec.unit})" for spec in metrics),
        ]
        lines += [
            f"### Table {section_index}: {title}",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
        ]
        previous = None
        for row in summary:
            motion_class = row["motion_class"]
            cells = [
                f"**{motion_class}**" if motion_class != previous else "",
                row["method"],
                _success_count(row),
            ]
            cells.extend(
                _format_value(
                    row.get(f"{spec.key}_mean"),
                    row.get(f"{spec.key}_std"),
                    spec.decimals,
                )
                for spec in metrics
            )
            lines.append("| " + " | ".join(cells) + " |")
            previous = motion_class
    return "\n".join(lines)


def latex_table(
    summary: list[dict],
    *,
    sections=TABLE_SECTIONS,
    group_label: str = "Motion class",
) -> str:
    def escape(value: object) -> str:
        text = str(value)
        for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            text = text.replace(old, new)
        return text

    lines = []
    for section_index, (title, metrics) in enumerate(sections):
        if section_index:
            lines += ["", r"\par\medskip", ""]
        headers = [
            group_label,
            "Method",
            "Success/total",
            *(f"{escape(spec.label)} ({escape(spec.unit)})" for spec in metrics),
        ]
        lines += [
            rf"\begin{{tabular}}{{{'llr' + 'r' * len(metrics)}}}",
            r"\toprule",
            rf"\multicolumn{{{len(headers)}}}{{c}}{{\textbf{{{escape(title)}}}}} \\",
            r"\midrule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
        previous = None
        for row in summary:
            motion_class = row["motion_class"]
            cells = [
                escape(motion_class) if motion_class != previous else "",
                escape(row["method"]),
                _success_count(row),
            ]
            cells.extend(
                _format_value(
                    row.get(f"{spec.key}_mean"),
                    row.get(f"{spec.key}_std"),
                    spec.decimals,
                    latex=True,
                )
                for spec in metrics
            )
            lines.append(" & ".join(cells) + r" \\")
            previous = motion_class
        lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_summary_tables(out: Path, summary: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (out / "table.md").write_text(markdown_table(summary) + "\n")
    (out / "table.tex").write_text(latex_table(summary) + "\n")


def write_tables(
    out: Path,
    summary: list[dict],
    *,
    sections=TABLE_SECTIONS,
    group_label: str = "Motion class",
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    markdown = markdown_table(summary, sections=sections, group_label=group_label)
    latex = latex_table(summary, sections=sections, group_label=group_label)
    (out / "table.md").write_text(markdown + "\n")
    (out / "table.tex").write_text(latex + "\n")
