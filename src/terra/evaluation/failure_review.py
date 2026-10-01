"""Select paired retargeting failures and their representative frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra._revision import write_git_commit

FAILURE_REVIEW_SCHEMA = "terra.retargeting-failure-review"
DATASETS = ("gait120", "vielemeyer", "darmstadt", "amass", "prism")
METHODS = ("terra", "omniretarget", "gmr", "smpl")
BASELINES = METHODS[1:]


@dataclass(frozen=True, slots=True)
class FailureMode:
    """One scalar ranking metric and its corresponding frame annotation."""

    name: str
    metric: str
    higher_is_better: bool = False
    dynamic: bool = False


FAILURE_MODES = {
    mode.name: mode
    for mode in (
        FailureMode("penetration", "penetration_duration_pct"),
        FailureMode("skating", "skating_duration_pct", dynamic=True),
        FailureMode("joint_limit", "joint_limit_duration_pct"),
        FailureMode("contact_gap", "contact_preservation_pct", higher_is_better=True),
        FailureMode("self_collision", "self_collision_duration_pct"),
        FailureMode("floating", "floating_duration_pct"),
        FailureMode("tendon_jump", "tendon_jump_duration_pct", dynamic=True),
    )
}

# Ten slots per baseline produce 30 unique motions per dataset. The second
# penetration/skating/contact pass samples a lower quantile so the review set
# contains clear but not exclusively extreme failures.
DEFAULT_SLOTS = (
    ("penetration", 0.90),
    ("skating", 0.90),
    ("joint_limit", 0.90),
    ("contact_gap", 0.90),
    ("self_collision", 0.90),
    ("floating", 0.90),
    ("tendon_jump", 0.90),
    ("penetration", 0.75),
    ("skating", 0.75),
    ("contact_gap", 0.75),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required CSV does not exist: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["motion"])
        writer.writeheader()
        writer.writerows(rows)


def _finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _gap(terra: dict[str, str], baseline: dict[str, str], mode: FailureMode) -> float | None:
    terra_value = _finite(terra, mode.metric)
    baseline_value = _finite(baseline, mode.metric)
    if terra_value is None or baseline_value is None:
        return None
    return (
        terra_value - baseline_value
        if mode.higher_is_better
        else baseline_value - terra_value
    )


def _annotation_path(artifact_root: Path, dataset: str, method: str, motion: str) -> Path:
    stem = motion.replace("/", "__") + ".npz"
    quality = artifact_root / dataset / "evaluation" / "quality"
    return quality / "frames" / stem if method == "terra" else quality / method / "frames" / stem


def _mask(archive: Any, mode: str) -> np.ndarray:
    if mode == "penetration":
        return np.asarray(archive["environment_penetrating"], dtype=bool) | np.any(
            np.asarray(archive["support_penetrating"], dtype=bool), axis=1
        )
    if mode == "skating":
        return np.asarray(archive["skating"], dtype=bool) | np.asarray(
            archive["stance_slip_failure"], dtype=bool
        )
    if mode == "floating":
        return np.any(np.asarray(archive["support_floating"], dtype=bool), axis=1)
    return np.asarray(
        archive[
            {
                "contact_gap": "contact_gap",
                "self_collision": "self_collision_bad",
                "joint_limit": "joint_limit_bad",
                "tendon_jump": "tendon_jump_bad",
            }[mode]
        ],
        dtype=bool,
    )


def _severity(archive: Any, mode: str, length: int) -> np.ndarray:
    if mode == "penetration":
        clearance = np.asarray(archive["sole_vertical_clearance_m"], dtype=float)[:length]
        return np.maximum(0.0, -np.min(clearance, axis=1))
    if mode == "skating":
        return np.max(np.asarray(archive["foot_velocity_m_s"], dtype=float)[:length], axis=1)
    if mode in {"floating", "contact_gap"}:
        clearance = np.asarray(archive["sole_vertical_clearance_m"], dtype=float)[:length]
        return np.maximum(0.0, np.max(clearance, axis=1))
    if mode == "joint_limit":
        return np.asarray(archive["joint_limit_excess_rad"], dtype=float)[:length]
    if mode == "tendon_jump":
        return np.asarray(archive["tendon_relative_event"], dtype=float)[:length]
    return np.zeros(length, dtype=float)


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, splits) if len(run)]


def recommend_frames(
    baseline_path: Path,
    terra_path: Path,
    mode: FailureMode,
) -> dict[str, Any] | None:
    """Choose baseline-only frames at the same timeline indices as TERRA."""

    if not baseline_path.is_file() or not terra_path.is_file():
        return None
    with np.load(baseline_path, allow_pickle=False) as baseline, np.load(
        terra_path, allow_pickle=False
    ) as terra:
        length = min(int(baseline["n_frames"]), int(terra["n_frames"]))
        baseline_mask = _mask(baseline, mode.name)[:length]
        terra_mask = _mask(terra, mode.name)[:length]
        active = np.asarray(baseline.get("common_active", np.ones(length)), dtype=bool)[:length]
        active &= np.asarray(terra.get("common_active", np.ones(length)), dtype=bool)[:length]
        contrast = active & baseline_mask & ~terra_mask
        runs = _runs(contrast)
        if not runs:
            return None
        severity = _severity(baseline, mode.name, length)
        run = max(runs, key=lambda values: (len(values), float(np.max(severity[values]))))
        highlight = int(run[int(np.argmax(severity[run]))])
        if mode.name == "skating":
            frames = tuple(sorted({int(run[0]), highlight, int(run[-1])}))
        elif mode.name == "tendon_jump":
            frames = tuple(sorted({max(0, highlight - 1), highlight}))
        else:
            frames = (highlight,)
        return {
            "highlight_frame": highlight,
            "frames": frames,
            "contrast_run_start": int(run[0]),
            "contrast_run_end": int(run[-1]),
            "contrast_run_frames": len(run),
            "frame_severity": float(severity[highlight]),
        }


def _complete_cohort(rows: list[dict[str, str]], dataset: str) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("dataset") != dataset or row.get("method", "").casefold() not in METHODS:
            continue
        grouped.setdefault(row["motion"], {})[row["method"].casefold()] = row
    return {
        motion: methods
        for motion, methods in grouped.items()
        if set(methods) == set(METHODS) and all(not methods[method].get("error", "").strip() for method in METHODS)
    }


def _ordered_candidates(
    candidates: list[tuple[str, float]],
    target_quantile: float,
    manifest: dict[str, dict[str, str]],
    class_counts: Counter[str],
) -> list[tuple[str, float]]:
    ranked = sorted(candidates, key=lambda item: (item[1], item[0]))
    target = round(target_quantile * (len(ranked) - 1))
    rank = {motion: index for index, (motion, _gap_value) in enumerate(ranked)}
    return sorted(
        ranked,
        key=lambda item: (
            class_counts[manifest[item[0]].get("terrain_class", "unclassified")],
            abs(rank[item[0]] - target),
            -item[1],
            item[0],
        ),
    )


def _select_one(
    *,
    cohort: dict[str, dict[str, dict[str, str]]],
    manifest: dict[str, dict[str, str]],
    artifact_root: Path,
    dataset: str,
    baseline: str,
    mode: FailureMode,
    target_quantile: float,
    used: set[str],
    class_counts: Counter[str],
) -> dict[str, Any] | None:
    candidates = []
    for motion, methods in cohort.items():
        if motion in used or motion not in manifest:
            continue
        gap = _gap(methods["terra"], methods[baseline], mode)
        if gap is not None and gap > 0:
            candidates.append((motion, gap))
    if not candidates:
        return None
    terra_value_key = mode.metric
    for motion, gap in _ordered_candidates(candidates, target_quantile, manifest, class_counts):
        recommendation = recommend_frames(
            _annotation_path(artifact_root, dataset, baseline, motion),
            _annotation_path(artifact_root, dataset, "terra", motion),
            mode,
        )
        if recommendation is None:
            continue
        methods = cohort[motion]
        baseline_value = _finite(methods[baseline], terra_value_key)
        terra_value = _finite(methods["terra"], terra_value_key)
        assert baseline_value is not None and terra_value is not None
        return {
            "motion": motion,
            "dataset": dataset,
            "source_dataset": dataset,
            "terrain_class": manifest[motion].get("terrain_class", "unclassified"),
            "highlight_baseline": baseline,
            "failure_mode": mode.name,
            "metric": mode.metric,
            "baseline_value": baseline_value,
            "terra_value": terra_value,
            "metric_gap": gap,
            **recommendation,
        }
    return None


def _select_dataset(
    rows: list[dict[str, str]],
    artifact_root: Path,
    dataset: str,
    per_dataset: int,
) -> list[dict[str, Any]]:
    if per_dataset % len(BASELINES):
        raise ValueError(f"per-dataset must be divisible by {len(BASELINES)}")
    slots_per_baseline = per_dataset // len(BASELINES)
    if slots_per_baseline > len(DEFAULT_SLOTS):
        raise ValueError(f"per-dataset supports at most {len(DEFAULT_SLOTS) * len(BASELINES)} motions")
    manifest_rows = _read_csv(artifact_root / dataset / "retarget" / "manifest.csv")
    manifest = {row["motion"]: row for row in manifest_rows}
    cohort = _complete_cohort(rows, dataset)
    used: set[str] = set()
    class_counts: Counter[str] = Counter()
    selections: list[dict[str, Any]] = []
    for slot_index in range(slots_per_baseline):
        requested_mode, target = DEFAULT_SLOTS[slot_index]
        for baseline in BASELINES:
            modes = (requested_mode, *(name for name in FAILURE_MODES if name != requested_mode))
            selected = None
            for mode_name in modes:
                selected = _select_one(
                    cohort=cohort,
                    manifest=manifest,
                    artifact_root=artifact_root,
                    dataset=dataset,
                    baseline=baseline,
                    mode=FAILURE_MODES[mode_name],
                    target_quantile=target,
                    used=used,
                    class_counts=class_counts,
                )
                if selected is not None:
                    selected["requested_failure_mode"] = requested_mode
                    break
            if selected is None:
                raise ValueError(
                    f"could not select {per_dataset} unique contrasted motions for {dataset}; "
                    f"stopped at {len(selections)}"
                )
            motion = str(selected["motion"])
            used.add(motion)
            class_counts[str(selected["terrain_class"])] += 1
            selections.append(selected)
    for index, row in enumerate(selections):
        row["review_index"] = index
        row["frames"] = ";".join(str(frame) for frame in row["frames"])
        fps = _finite(cohort[str(row["motion"])][str(row["highlight_baseline"])], "fps")
        row["highlight_time_s"] = None if not fps else float(row["highlight_frame"]) / fps
    return selections


def _write_report(
    output: Path,
    selections: list[dict[str, Any]],
    datasets: tuple[str, ...],
) -> None:
    lines = [
        "# Paired retargeting failure review",
        "",
        "Every selected motion has successful TERRA, OmniRetarget, GMR, and SMPL artifacts. "
        "The proposed frames contain a baseline failure annotation that is absent from TERRA "
        "at the same timeline index.",
        "",
        "| Dataset | Motions | Terrain classes | OmniRetarget | GMR | SMPL |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for dataset in datasets:
        rows = [row for row in selections if row["dataset"] == dataset]
        methods = Counter(str(row["highlight_baseline"]) for row in rows)
        classes = Counter(str(row["terrain_class"]) for row in rows)
        class_text = ", ".join(f"{name}: {count}" for name, count in sorted(classes.items()))
        lines.append(
            f"| {dataset} | {len(rows)} | {class_text} | {methods['omniretarget']} | "
            f"{methods['gmr']} | {methods['smpl']} |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Failure modes selected |",
            "|---|---|",
        ]
    )
    for dataset in datasets:
        modes = Counter(
            str(row["failure_mode"])
            for row in selections
            if row["dataset"] == dataset
        )
        mode_text = ", ".join(f"{name}: {count}" for name, count in sorted(modes.items()))
        lines.append(f"| {dataset} | {mode_text} |")
    lines.extend(
        [
            "",
            "`frame_candidates.csv` gives the baseline, failure mode, matched frame or frame set, "
            "scalar metric gap, and annotation severity for every selected motion.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def build_failure_review(
    benchmark_csv: Path,
    artifact_root: Path,
    output_root: Path,
    *,
    per_dataset: int = 30,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, Any]:
    """Write paired review manifests and baseline-only frame recommendations."""

    benchmark = benchmark_csv.expanduser().resolve()
    artifacts = artifact_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    rows = _read_csv(benchmark)
    selections = [
        row
        for dataset in datasets
        for row in _select_dataset(rows, artifacts, dataset, per_dataset)
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_git_commit(output)
    _write_csv(output / "selection.csv", selections)
    _write_csv(output / "frame_candidates.csv", selections)
    for dataset in datasets:
        _write_csv(
            output / "selections" / f"{dataset}.csv",
            [row for row in selections if row["dataset"] == dataset],
        )
    payload = {
        "schema": FAILURE_REVIEW_SCHEMA,
        "benchmark_csv": str(benchmark),
        "artifact_root": str(artifacts),
        "datasets": list(datasets),
        "methods": list(METHODS),
        "motions_per_dataset": per_dataset,
        "motions": len(selections),
        "renders_expected": len(selections) * len(METHODS),
        "selection_rule": (
            "ten paired slots per baseline, stratified by failure mode and terrain class; "
            "90th/75th-percentile gaps; baseline failure absent from TERRA at matched frames"
        ),
    }
    (output / "review.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(output, selections, datasets)
    print(
        f"Failure review: {len(selections)} motions, {payload['renders_expected']} paired renders -> {output}"
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="terra benchmark failure-review",
        description=__doc__,
    )
    result.add_argument("--benchmark-csv", type=Path, required=True)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--per-dataset", type=int, default=30)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        build_failure_review(
            args.benchmark_csv,
            args.artifact_root,
            args.output_root,
            per_dataset=args.per_dataset,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


__all__ = ["FAILURE_REVIEW_SCHEMA", "build_failure_review", "main", "recommend_frames"]
