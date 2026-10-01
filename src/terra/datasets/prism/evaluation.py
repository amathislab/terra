#!/usr/bin/env python3
"""Score a published PRISM terrain reconstruction against reference object meshes.

This evaluator never reconstructs terrain.  It first loads the immutable terrain record
published by ``run_dataset.py`` and derives any seated-support query locations from the
same converted source motion.  Only then does it open the trusted PRISM take and load its
object meshes for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from terra._revision import write_git_commit
from terra.benchmarking.reconstruction.provenance import write_evaluation_provenance
from terra.evaluation.reconstruction import (
    ReconstructionInputs,
    load_reconstruction_inputs,
    validate_reconstruction_record,
)
from terra.paths import StorageRoots

from .adapter import load_take, observed_support_points, observed_support_xy
from .mesh_metrics import (
    coordinate_audit,
    mesh_height_at,
    score_height_fields,
    score_observed_support,
)
from .root import resolve_data_root

PRIMARY_COVERAGE_GATE = 0.95
PRIMARY_WITHIN_50MM_GATE = 0.95
DATASET_TAKE_PASS_GATE = 0.90
DATASET_MACRO_MAE_GATE_M = 0.020
CURRENT_FIT_STATUSES = {"ok", "cached"}
FOOTPRINT_WORKSPACE_MARGIN_M = 0.50
FOOT_SUPPORT_VERTICAL_TOLERANCE_M = 0.10

# The frozen PRISM manifests distinguish geometry with ``terrain_class`` even when the
# source protocol's free-text trial labels differ by lighting, repetition, or stray
# whitespace. Keep this mapping deliberately closed: a new terrain class must be reviewed
# before it can be combined with an existing report row.
CONDITION_GROUP_BY_TERRAIN_CLASS = {
    "chair_sit": "Sitting",
    "platform": "Stepping boxes",
    "stairs_up_down": "Stairs",
}


def _condition_group_for_terrain_class(terrain_class: str) -> str:
    try:
        return CONDITION_GROUP_BY_TERRAIN_CLASS[terrain_class]
    except KeyError as exc:
        supported = ", ".join(sorted(CONDITION_GROUP_BY_TERRAIN_CLASS))
        raise ValueError(f"unsupported PRISM terrain_class {terrain_class!r}; expected one of {supported}") from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _motion_take(motion: str) -> tuple[str, str]:
    parts = motion.split("/")
    if len(parts) != 3 or parts[0] != "PRISM" or not parts[2].endswith("_poses"):
        raise ValueError(f"invalid PRISM motion ID {motion!r}")
    subject = parts[1]
    take = parts[2].removesuffix("_poses")
    if not subject.startswith("subj") or not take.startswith("take"):
        raise ValueError(f"invalid PRISM motion ID {motion!r}")
    return subject, take


def _terrain_record_path(terrain_dir: Path, motion: str) -> Path:
    return terrain_dir / f"{motion.replace('/', '__')}.json"


def _candidate_fit_statuses(path: Path) -> dict[str, dict[str, str]]:
    """Read current candidate admissions without trusting leftover terrain JSON."""

    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if "motion" not in fields or "status" not in fields:
            raise ValueError(f"candidate fit status must contain motion and status columns: {path}")
        rows = list(reader)
    result: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        motion = (row.get("motion") or "").strip()
        status = (row.get("status") or "").strip()
        if not motion or not status:
            raise ValueError(f"candidate fit status row {index} has an empty motion or status: {path}")
        if motion in result:
            raise ValueError(f"candidate fit status contains duplicate motion {motion!r}: {path}")
        result[motion] = row | {"motion": motion, "status": status}
    return result


def _fit_admission(
    motion: str,
    statuses: dict[str, dict[str, str]],
    status_path: Path,
) -> dict[str, str | None]:
    row = statuses.get(motion)
    if row is None:
        return {
            "candidate_fit_status": None,
            "candidate_fit_admission_error": (f"current candidate fit status is missing for {motion!r}: {status_path}"),
        }
    status = row["status"]
    if status in CURRENT_FIT_STATUSES:
        return {"candidate_fit_status": status, "candidate_fit_admission_error": ""}
    detail = (row.get("error") or "").strip()
    suffix = f"; {detail}" if detail else ""
    return {
        "candidate_fit_status": status,
        "candidate_fit_admission_error": f"current candidate fit status is {status!r}{suffix}",
    }


def _terrain_from_record(record: dict[str, Any]) -> Any:
    from terra._musclemimic import TerrainSpec

    value = record.get("terrain")
    if value is None:
        value = {"boxes": []}
    if not isinstance(value, dict):
        raise ValueError("published terrain geometry must be an object or null")
    return TerrainSpec.from_dict(value)


def _seated_support_xy(
    motion: str,
    source_path: Path,
    *,
    terrain_class: str,
    smpl_model_path: Path | None = None,
    fitted_shape_path: Path | None = None,
) -> np.ndarray:
    """Derive seated-support queries from the converted source motion alone."""
    if terrain_class != "chair_sit":
        return np.empty((0, 2), dtype=float)

    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.smplh import load_smplh_motion
    from terra.source import motion_world_joints
    from terra.terrain import detect_seat_rests

    motion_data = load_smplh_motion(source_path)
    joints, fps = motion_world_joints(
        motion,
        env_name="MyoFullBody",
        use_fitted_shape=True,
        motion_data=motion_data,
        calibrate_sites=True,
        smpl_model_path=smpl_model_path,
        fitted_shape_path=fitted_shape_path,
    )
    joint_order = list(SMPLH_DEMO_JOINTS)
    if joints.ndim != 3 or joints.shape[1] != len(joint_order) or joints.shape[2] < 2:
        raise ValueError("replayed source landmarks have an unexpected shape")
    pelvis = joint_order.index("Pelvis")
    rests = detect_seat_rests(joints, joint_order, fps)
    points = [np.median(joints[rest.start : rest.end, pelvis, :2], axis=0) for rest in rests]
    return np.asarray(points, dtype=float).reshape(-1, 2)


def score_prepared_take(
    motion: str,
    take: dict[str, Any],
    terrain_record: dict[str, Any],
    seated_xy: np.ndarray,
    *,
    terrain_class: str,
    grid_resolution: float = 0.02,
    cop_stride: int = 10,
) -> dict[str, Any]:
    """Score one already-published terrain against the reference geometry."""
    terrain = _terrain_from_record(terrain_record)
    fit_report = terrain_record.get("fit") or {}
    validation = terrain_record.get("validation") or {}
    if not isinstance(fit_report, dict) or not isinstance(validation, dict):
        raise ValueError("published terrain fit and validation records must be objects")
    # Evaluation boundary: no object value is accessed until the frozen TerrainSpec exists.
    objects = take.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("PRISM take has no reference object meshes")
    foot_candidates = observed_support_points(take, stride=cop_stride)
    candidate_surface = mesh_height_at(foot_candidates[:, :2], objects)
    foot_consistent = (
        np.abs(foot_candidates[:, 2] - candidate_surface) <= FOOT_SUPPORT_VERTICAL_TOLERANCE_M
    )
    foot_xy = foot_candidates[foot_consistent, :2]
    seated_xy = np.asarray(seated_xy, dtype=float).reshape(-1, 2)
    observed_xy = np.concatenate((foot_xy, seated_xy), axis=0)
    foot = score_observed_support(foot_xy, objects, terrain)
    seated = score_observed_support(seated_xy, objects, terrain)
    observed = score_observed_support(observed_xy, objects, terrain)
    workspace_xy = observed_support_xy(take, stride=1)
    mesh_full = score_height_fields(
        objects,
        terrain,
        resolution=grid_resolution,
        evaluation_domain_xy=workspace_xy,
        evaluation_margin=FOOTPRINT_WORKSPACE_MARGIN_M,
    )
    raw_unsupported = fit_report.get("unsupported_support_kinds")
    if raw_unsupported is None:
        raw_unsupported = []
    if not isinstance(raw_unsupported, (list, tuple)) or any(
        not isinstance(kind, str) or not kind.strip() for kind in raw_unsupported
    ):
        raise ValueError("published fit unsupported_support_kinds must be a list of non-empty strings")
    unsupported_support_kinds = sorted({kind.strip().casefold() for kind in raw_unsupported})
    primary_support_complete = not (terrain_class == "chair_sit" and "pelvis" in unsupported_support_kinds)
    has_raised_observed_support = int(observed.get("raised_points", 0)) > 0
    eligible = has_raised_observed_support
    primary_ineligibility_reasons = []
    if not has_raised_observed_support:
        primary_ineligibility_reasons.append("no raised observed-support query")
    primary_ineligibility_reason = "; ".join(primary_ineligibility_reasons) or None
    primary_pass = bool(
        eligible
        and float(observed["raised_coverage"]) >= PRIMARY_COVERAGE_GATE
        and float(observed["within_50mm"]) >= PRIMARY_WITHIN_50MM_GATE
    )
    primary_pass_with_internal_validation = bool(primary_pass and validation.get("passed") is True)
    subject, take_name = _motion_take(motion)
    data_info = take["info"]["data_info"]
    condition = data_info.get("trial_name", take_name)
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("PRISM take trial_name must be a non-empty string")
    condition_group = _condition_group_for_terrain_class(terrain_class)
    return {
        "motion": motion,
        "subject": subject,
        "take": take_name,
        # Preserve the source label verbatim, including the release's trailing whitespace.
        "condition": condition,
        "condition_group": condition_group,
        "terrain_class": terrain_class,
        "frames": len(take["smpl_params"]["poses"]),
        "fps": float(data_info["fps"]),
        "object_names": sorted(objects),
        "selected_model": fit_report.get("model"),
        "internal_validation_passed": validation.get("passed"),
        "coordinate_audit": coordinate_audit(take, objects),
        "mesh_at_foot_support": foot,
        "foot_query_filter": {
            "candidate_points": len(foot_candidates),
            "accepted_points": int(foot_consistent.sum()),
            "rejected_points": int((~foot_consistent).sum()),
            "vertical_tolerance_m": FOOT_SUPPORT_VERTICAL_TOLERANCE_M,
        },
        "mesh_at_seated_support": seated,
        "mesh_at_observed_support": observed,
        "mesh_full": mesh_full,
        "unsupported_support_kinds": unsupported_support_kinds,
        "primary_support_complete": primary_support_complete,
        "primary_eligible": eligible,
        "primary_ineligibility_reason": primary_ineligibility_reason,
        "primary_pass": primary_pass,
        "primary_pass_with_internal_validation": primary_pass_with_internal_validation,
    }


def _evaluate_motion(task: dict[str, Any]) -> dict[str, Any]:
    motion = str(task["motion"])
    admission_error = task.get("candidate_fit_admission_error")
    if admission_error:
        # This check intentionally precedes even stat/read access to the terrain record.
        # A failed current request must never fall back to an older JSON at the same path.
        raise RuntimeError(str(admission_error))
    terrain_path = _terrain_record_path(Path(task["terrain_dir"]), motion)
    cohort = task.get("reconstruction_cohort")
    if not isinstance(cohort, ReconstructionInputs):
        raise ValueError("PRISM scoring requires a verified reconstruction cohort")
    terrain_record = validate_reconstruction_record(
        terrain_path,
        motion=motion,
        status=cohort.statuses[motion],
        cohort=cohort,
    )
    _terrain_from_record(terrain_record)
    # Query locations are derived before the raw take (and therefore its mesh) is opened.
    source_path = Path(task["input_root"]) / f"{motion}.npz"
    if not source_path.is_file():
        raise FileNotFoundError(f"converted PRISM source motion not found: {source_path}")
    seated_xy = _seated_support_xy(
        motion,
        source_path,
        terrain_class=str(task["terrain_class"]),
        smpl_model_path=Path(task["smpl_model_path"]),
        fitted_shape_path=Path(task["fitted_shape_path"]),
    )

    subject, take_name = _motion_take(motion)
    take_path = Path(task["data_root"]) / subject / f"{take_name}.pkl"
    if not take_path.is_file():
        raise FileNotFoundError(f"trusted PRISM take not found: {take_path}")
    take = load_take(take_path)
    result = score_prepared_take(
        motion,
        take,
        terrain_record,
        seated_xy,
        terrain_class=str(task["terrain_class"]),
        grid_resolution=float(task["grid_resolution"]),
        cop_stride=int(task["cop_stride"]),
    )
    result.update(
        candidate_fit_status=task.get("candidate_fit_status"),
        source_path=str(source_path),
        take_path=str(take_path),
        terrain_record_path=str(terrain_path),
        seated_query_points=len(seated_xy),
    )
    return result


def _value(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _stats(values: list[float | None], *, scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(
        [float(value) * scale for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _micro_observed(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [record["mesh_at_observed_support"] for record in records]
    raised = sum(int(score.get("raised_points", 0)) for score in scores)
    if not raised:
        return {"raised_points": 0}
    return {
        "raised_points": raised,
        "height_mae_mm": 1000.0 * sum(float(score.get("height_abs_error_sum_m", 0.0)) for score in scores) / raised,
        "raised_coverage": sum(int(score.get("covered_raised_points", 0)) for score in scores) / raised,
        "within_20mm": sum(int(score.get("within_20mm_count", 0)) for score in scores) / raised,
        "within_50mm": sum(int(score.get("within_50mm_count", 0)) for score in scores) / raised,
    }


def _micro_full(records: list[dict[str, Any]], tolerance: str) -> dict[str, Any]:
    scores = [record["mesh_full"] for record in records]
    gt = sum(int(score.get("gt_raised_cells", 0)) for score in scores)
    pred = sum(int(score.get("pred_raised_cells", 0)) for score in scores)
    gt_flat = sum(int(score.get("gt_flat_cells", 0)) for score in scores)
    union = sum(int(score.get("raised_union_cells", 0)) for score in scores)
    overlap = sum(int(score.get("raised_overlap_cells", 0)) for score in scores)
    flat_overlap = sum(int(score.get("flat_overlap_cells", 0)) for score in scores)
    matched = sum(int(score.get(f"support_matched_cells_{tolerance}", 0)) for score in scores)
    precision = matched / pred if pred else None
    recall = matched / gt if gt else None
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "gt_raised_cells": gt,
        "pred_raised_cells": pred,
        "raised_footprint_iou": overlap / union if union else None,
        "raised_terrain_coverage": overlap / gt if gt else None,
        "flat_terrain_coverage": flat_overlap / gt_flat if gt_flat else None,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record["primary_eligible"]]
    foot = "mesh_at_foot_support"
    seated = "mesh_at_seated_support"
    observed = "mesh_at_observed_support"
    full = "mesh_full"
    pass_rate = sum(record["primary_pass"] for record in eligible) / len(eligible) if eligible else None
    raw_conditions: dict[str, int] = defaultdict(int)
    for record in records:
        raw_conditions[record["condition"]] += 1
    return {
        "takes": len(records),
        "subjects": len({record["subject"] for record in records}),
        "objects": sum(len(record["object_names"]) for record in records),
        "terrain_classes": sorted({record["terrain_class"] for record in records}),
        "raw_conditions": dict(sorted(raw_conditions.items())),
        "internal_validation_pass_rate": (
            sum(record["internal_validation_passed"] is True for record in records) / len(records) if records else None
        ),
        "primary_eligible": len(eligible),
        "primary_support_complete": sum(record.get("primary_support_complete", True) for record in records),
        "primary_support_incomplete": sum(not record.get("primary_support_complete", True) for record in records),
        "primary_passes": sum(record["primary_pass"] for record in eligible),
        "primary_pass_rate": pass_rate,
        "primary_passes_with_internal_validation": sum(
            record["primary_pass_with_internal_validation"] for record in eligible
        ),
        "coordinate_warnings": sum(bool(_value(record, "coordinate_audit", "warning")) for record in records),
        "foot_height_mae_mm": _stats([_value(record, foot, "height_mae_m") for record in records], scale=1000.0),
        "seated_height_mae_mm": _stats([_value(record, seated, "height_mae_m") for record in records], scale=1000.0),
        "observed_height_mae_mm": _stats(
            [_value(record, observed, "height_mae_m") for record in eligible], scale=1000.0
        ),
        "observed_height_p95_mm": _stats(
            [_value(record, observed, "height_p95_m") for record in eligible], scale=1000.0
        ),
        "observed_raised_coverage": _stats([_value(record, observed, "raised_coverage") for record in eligible]),
        "observed_within_50mm": _stats([_value(record, observed, "within_50mm") for record in eligible]),
        "full_height_mae_union_mm": _stats(
            [_value(record, full, "height_mae_union_m") for record in records], scale=1000.0
        ),
        "full_footprint_iou": _stats([_value(record, full, "raised_footprint_iou") for record in records]),
        "raised_terrain_coverage": _stats([_value(record, full, "raised_terrain_coverage") for record in records]),
        "flat_terrain_coverage": _stats([_value(record, full, "flat_terrain_coverage") for record in records]),
        "footprint_evaluation_area_m2": _stats([_value(record, full, "evaluation_area_m2") for record in records]),
        "full_support_f1_20mm": _stats([_value(record, full, "support_f1_20mm") for record in records]),
        "full_support_f1_50mm": _stats([_value(record, full, "support_f1_50mm") for record in records]),
        "micro_observed": _micro_observed(eligible),
        "micro_full_20mm": _micro_full(records, "20mm"),
        "micro_full_50mm": _micro_full(records, "50mm"),
    }


def _cluster_bootstrap_ci(
    records: list[dict[str, Any]],
    accessor: Callable[[dict[str, Any]], float | bool | None],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = accessor(record)
        if value is not None and math.isfinite(float(value)):
            groups[record["subject"]].append(record)
    values = [float(accessor(record)) for rows in groups.values() for record in rows]
    if not values:
        return {"n_takes": 0, "n_subjects": 0, "estimate": None, "ci95": [None, None]}
    subjects = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        replicate = [float(accessor(record)) for subject in sampled for record in groups[str(subject)]]
        estimates[index] = np.mean(replicate)
    return {
        "n_takes": len(values),
        "n_subjects": len(subjects),
        "estimate": float(np.mean(values)),
        "ci95": [float(value) for value in np.percentile(estimates, [2.5, 97.5])],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def summarize(
    records: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        conditions[record["condition_group"]].append(record)
    overall = _group_summary(records)
    ci_accessors: dict[str, Callable[[dict[str, Any]], float | bool | None]] = {
        "primary_pass_rate": lambda row: row["primary_pass"] if row["primary_eligible"] else None,
        "observed_height_mae_mm": lambda row: (
            1000.0 * float(_value(row, "mesh_at_observed_support", "height_mae_m")) if row["primary_eligible"] else None
        ),
        "observed_raised_coverage": lambda row: (
            _value(row, "mesh_at_observed_support", "raised_coverage") if row["primary_eligible"] else None
        ),
        "observed_within_50mm": lambda row: (
            _value(row, "mesh_at_observed_support", "within_50mm") if row["primary_eligible"] else None
        ),
        "full_footprint_iou": lambda row: _value(row, "mesh_full", "raised_footprint_iou"),
        "raised_terrain_coverage": lambda row: _value(row, "mesh_full", "raised_terrain_coverage"),
        "flat_terrain_coverage": lambda row: _value(row, "mesh_full", "flat_terrain_coverage"),
        "full_support_f1_50mm": lambda row: _value(row, "mesh_full", "support_f1_50mm"),
    }
    confidence_intervals = {
        name: _cluster_bootstrap_ci(
            records,
            accessor,
            samples=bootstrap_samples,
            seed=seed + index,
        )
        for index, (name, accessor) in enumerate(ci_accessors.items())
    }
    dataset_gate = {
        "complete_observed_support": overall["primary_eligible"] == len(records) + len(errors),
        "take_pass_rate_at_least_90pct": (
            overall["primary_pass_rate"] is not None and overall["primary_pass_rate"] >= DATASET_TAKE_PASS_GATE
        ),
        "macro_observed_height_mae_at_most_20mm": (
            overall["observed_height_mae_mm"]["mean"] is not None
            and overall["observed_height_mae_mm"]["mean"] <= 1000.0 * DATASET_MACRO_MAE_GATE_M
        ),
    }
    dataset_gate["passed"] = all(dataset_gate.values())
    return {
        "dataset": "prism",
        "selected": len(records) + len(errors),
        "scored": len(records),
        "errors": errors,
        "definitions": {
            "ground_truth": "PRISM object-mesh upper support envelope in the documented world frame",
            "observed_support": (
                "measured insole CoP XY consistent within 0.10 m of the reference support envelope, plus pelvis "
                "XY detected from the converted source motion; the query set is identical for every method"
            ),
            "full_mesh": ("20 mm XY raster over the method-independent per-take foot-contact/object workspace"),
            "raised_terrain_coverage": (
                "fraction of ground-truth raised raster cells predicted raised; horizontal footprint recall"
            ),
            "flat_terrain_coverage": (
                "fraction of ground-truth flat raster cells predicted flat within the method-independent "
                "per-take foot-contact/object workspace; horizontal specificity"
            ),
            "footprint_workspace": (
                "axis-aligned bounds of all measured foot-contact CoP XY and reference object vertices, "
                f"padded by {FOOTPRINT_WORKSPACE_MARGIN_M:.2f} m; identical for every method on a take"
            ),
            "development_use": (
                "PRISM meshes were used only to select the shared seated-support height rule; "
                "the frozen mesh files are not read during reconstruction"
            ),
            "evaluation_isolation": "object meshes are opened only after loading the published terrain record",
            "primary_pass": (
                "reference observed-support coverage and height; internal validation is reported separately"
            ),
            "primary_support_completeness": (
                "fit.unsupported_support_kinds is a method diagnostic only; it does not change the shared "
                "evaluation queries or denominator"
            ),
            "candidate_fit_status": (
                "when --fit-status is supplied, only current ok/cached rows are admitted; "
                "failed or missing rows remain selected and stale terrain JSON is not read"
            ),
            "condition_grouping": (
                "canonical condition_group comes from the frozen manifest terrain_class; "
                "condition preserves take.info.data_info.trial_name verbatim"
            ),
        },
        "condition_group_by_terrain_class": CONDITION_GROUP_BY_TERRAIN_CLASS,
        "take_gate": {
            "raised_support_coverage": PRIMARY_COVERAGE_GATE,
            "observed_within_50mm": PRIMARY_WITHIN_50MM_GATE,
            "requires_internal_validation": False,
        },
        "dataset_gate": dataset_gate,
        "overall": overall,
        "conditions": {condition: _group_summary(conditions[condition]) for condition in sorted(conditions)},
        "subject_clustered_bootstrap_ci95": confidence_intervals,
    }


def _flat_row(record: dict[str, Any]) -> dict[str, Any]:
    foot = record["mesh_at_foot_support"]
    seated = record["mesh_at_seated_support"]
    observed = record["mesh_at_observed_support"]
    full = record["mesh_full"]
    audit = record["coordinate_audit"]
    return {
        "motion": record["motion"],
        "subject": record["subject"],
        "take": record["take"],
        "condition": record["condition"],
        "condition_group": record["condition_group"],
        "terrain_class": record["terrain_class"],
        "candidate_fit_status": record.get("candidate_fit_status"),
        "objects": len(record["object_names"]),
        "selected_model": record["selected_model"],
        "internal_validation_passed": record["internal_validation_passed"],
        "primary_eligible": record["primary_eligible"],
        "primary_support_complete": record.get("primary_support_complete", True),
        "primary_ineligibility_reason": record.get("primary_ineligibility_reason"),
        "primary_pass": record["primary_pass"],
        "primary_pass_with_internal_validation": record["primary_pass_with_internal_validation"],
        "foot_raised_points": foot["raised_points"],
        "foot_height_mae_mm": None if foot.get("height_mae_m") is None else 1000.0 * foot["height_mae_m"],
        "seated_raised_points": seated["raised_points"],
        "seated_height_mae_mm": (None if seated.get("height_mae_m") is None else 1000.0 * seated["height_mae_m"]),
        "observed_raised_points": observed["raised_points"],
        "observed_height_mae_mm": (None if observed.get("height_mae_m") is None else 1000.0 * observed["height_mae_m"]),
        "observed_height_p95_mm": (None if observed.get("height_p95_m") is None else 1000.0 * observed["height_p95_m"]),
        "observed_height_p99_mm": (None if observed.get("height_p99_m") is None else 1000.0 * observed["height_p99_m"]),
        "observed_height_max_mm": (None if observed.get("height_max_m") is None else 1000.0 * observed["height_max_m"]),
        "observed_raised_coverage": observed.get("raised_coverage"),
        "observed_within_20mm": observed.get("within_20mm"),
        "observed_within_50mm": observed.get("within_50mm"),
        "full_height_mae_union_mm": (
            None if full.get("height_mae_union_m") is None else 1000.0 * full["height_mae_union_m"]
        ),
        "full_height_p95_union_mm": (
            None if full.get("height_p95_union_m") is None else 1000.0 * full["height_p95_union_m"]
        ),
        "full_footprint_iou": full.get("raised_footprint_iou"),
        "raised_terrain_coverage": full.get("raised_terrain_coverage"),
        "flat_terrain_coverage": full.get("flat_terrain_coverage"),
        "footprint_evaluation_area_m2": full.get("evaluation_area_m2"),
        "full_support_f1_20mm": full.get("support_f1_20mm"),
        "full_support_f1_50mm": full.get("support_f1_50mm"),
        "coordinate_centered_p95_mm": (
            None if audit.get("cop_to_mesh_centered_p95_m") is None else 1000.0 * audit["cop_to_mesh_centered_p95_m"]
        ),
        "coordinate_warning": audit.get("warning"),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def write_report(
    output: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_git_commit(output)
    _write_json(output / "per_motion.json", records)
    _write_json(output / "summary.json", summary)
    rows = [_flat_row(record) for record in records]
    fields = list(rows[0]) if rows else ["motion", "error"]
    with (output / "per_motion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    overall = summary["overall"]
    mae = overall["observed_height_mae_mm"]
    internal_pass_pct = (
        None if overall["internal_validation_pass_rate"] is None else 100.0 * overall["internal_validation_pass_rate"]
    )
    lines = [
        "# PRISM object-mesh terrain reconstruction development evaluation",
        "",
        "The scored geometry is the frozen published terrain record. PRISM meshes informed only the shared "
        "seated-support height rule; individual mesh files are not read during reconstruction, and no rigid "
        "alignment or per-take mesh-derived offset is fitted.",
        "",
        "| Takes scored | Primary pass | Primary + internal validation | Internal validation | Observed height MAE mean / median / p95 (mm) | Raised-terrain coverage mean | Flat-terrain coverage mean | Footprint IoU mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['scored']}/{summary['selected']} | {overall['primary_passes']}/{overall['primary_eligible']} "
        f"({_fmt(None if overall['primary_pass_rate'] is None else 100 * overall['primary_pass_rate'])}%) | "
        f"{overall['primary_passes_with_internal_validation']}/{overall['primary_eligible']} | "
        f"{_fmt(internal_pass_pct)}% | "
        f"{_fmt(mae['mean'])} / {_fmt(mae['median'])} / {_fmt(mae['p95'])} | "
        f"{_fmt(overall['raised_terrain_coverage']['mean'], 3)} | "
        f"{_fmt(overall['flat_terrain_coverage']['mean'], 3)} | "
        f"{_fmt(overall['full_footprint_iou']['mean'], 3)} |",
        "",
        "Primary per-take pass requires at least 95% raised-support coverage and at least 95% of observed "
        "raised-support queries within 50 mm. Method-internal validation is reported "
        "separately (and as a combined diagnostic), because it is not a common mesh-reference test. "
        "Source-derived seated queries and the denominator are identical for every method. "
        "Whole-object extent scores are secondary because unobserved surfaces are not identifiable "
        "from body motion alone.",
        "Raised-terrain coverage is footprint recall: the fraction of GT-raised raster cells predicted "
        "raised. Flat-terrain coverage is footprint specificity: the fraction of GT-flat raster cells "
        "predicted flat. Both use the same 20 mm raised threshold and the same method-independent per-take "
        "workspace: the axis-aligned bounds of measured foot-contact CoP and object geometry, padded by "
        f"{FOOTPRINT_WORKSPACE_MARGIN_M:.2f} m. The finite domain makes flat coverage well-defined; it is "
        "not a claim about infinite global free space.",
        "",
    ]
    lines.extend(
        [
            "",
            "Condition rows use the frozen manifest's geometry-bearing `terrain_class`; "
            "`per_motion.json` and `per_motion.csv` retain each take's raw trial label.",
            "",
            "| Condition group | Takes | Primary pass | Foot height MAE mean (mm) | Seated height MAE mean (mm) | Observed MAE mean (mm) | Observed-support coverage mean | Raised-terrain coverage mean | Flat-terrain coverage mean | Footprint IoU mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, item in summary["conditions"].items():
        lines.append(
            f"| {condition} | {item['takes']} | {item['primary_passes']}/{item['primary_eligible']} | "
            f"{_fmt(item['foot_height_mae_mm']['mean'])} | "
            f"{_fmt(item['seated_height_mae_mm']['mean'])} | "
            f"{_fmt(item['observed_height_mae_mm']['mean'])} | "
            f"{_fmt(item['observed_raised_coverage']['mean'], 3)} | "
            f"{_fmt(item['raised_terrain_coverage']['mean'], 3)} | "
            f"{_fmt(item['flat_terrain_coverage']['mean'], 3)} | "
            f"{_fmt(item['full_footprint_iou']['mean'], 3)} |"
        )
    lines.extend(
        [
            "",
            "| Metric | Estimate | Subject-clustered bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for name, item in summary["subject_clustered_bootstrap_ci95"].items():
        lines.append(
            f"| {name} | {_fmt(item['estimate'], 3)} | [{_fmt(item['ci95'][0], 3)}, {_fmt(item['ci95'][1], 3)}] |"
        )
    gate = summary["dataset_gate"]
    lines.extend(
        [
            "",
            f"Pre-registered dataset gate: **{'PASS' if gate['passed'] else 'FAIL'}**. "
            f"Complete observed-support coverage={gate['complete_observed_support']}; "
            f"take pass rate >=90%={gate['take_pass_rate_at_least_90pct']}; "
            f"macro observed MAE <=20 mm={gate['macro_observed_height_mae_at_most_20mm']}.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def _manifest_selections(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    motions = [row.get("motion", "").strip() for row in rows]
    if not motions or any(not motion for motion in motions):
        raise ValueError(f"manifest contains no complete motion column: {path}")
    if len(motions) != len(set(motions)):
        raise ValueError(f"manifest contains duplicate motions: {path}")
    selections = []
    for motion, row in zip(motions, rows, strict=True):
        _motion_take(motion)
        terrain_class = row.get("terrain_class", "").strip()
        if not terrain_class:
            raise ValueError(f"manifest row for {motion} has no terrain_class: {path}")
        condition_group = _condition_group_for_terrain_class(terrain_class)
        selections.append(
            {
                "motion": motion,
                "terrain_class": terrain_class,
                "condition_group": condition_group,
            }
        )
    return selections


def main(argv: list[str] | None = None) -> int:
    roots = StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(prog="terra evaluate prism-mesh", description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--terrain-dir", type=Path, required=True)
    parser.add_argument(
        "--fit-status",
        type=Path,
        help="optional current candidate status.csv; only ok/cached rows may be scored",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--smpl-model-path",
        type=Path,
        default=roots.model_root,
        help="SMPL-H model root used to replay the converted source motion",
    )
    parser.add_argument(
        "--fitted-shape-path",
        type=Path,
        help="MyoFullBody fitted-shape file; defaults to INPUT_ROOT/../cache/MyoFullBody/shape_optimized.pkl",
    )
    parser.add_argument("--data-root", type=Path, default=roots.data_root / "PRISM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--grid-resolution", type=float, default=0.02)
    parser.add_argument("--cop-stride", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.cop_stride < 1 or args.bootstrap_samples < 1:
        raise SystemExit("workers, cop-stride, and bootstrap-samples must be positive")
    if not math.isfinite(args.grid_resolution) or args.grid_resolution <= 0:
        raise SystemExit("grid-resolution must be finite and positive")

    manifest = args.manifest.resolve()
    selections = _manifest_selections(manifest)
    cohort = load_reconstruction_inputs(manifest, args.terrain_dir, args.fit_status)
    fit_status_path = cohort.status_path
    data_root = resolve_data_root(args.data_root)
    input_root = args.input_root.resolve()
    smpl_model_path = args.smpl_model_path.resolve()
    fitted_shape_path = (
        args.fitted_shape_path.resolve()
        if args.fitted_shape_path is not None
        else input_root.parent / "cache" / "MyoFullBody" / "shape_optimized.pkl"
    )
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL-H model root not found: {smpl_model_path}")
    if not fitted_shape_path.is_file():
        raise FileNotFoundError(f"MyoFullBody fitted-shape file not found: {fitted_shape_path}")
    common = {
        "terrain_dir": str(cohort.terrain_dir),
        "input_root": str(input_root),
        "data_root": str(data_root),
        "smpl_model_path": str(smpl_model_path),
        "fitted_shape_path": str(fitted_shape_path),
        "grid_resolution": args.grid_resolution,
        "cop_stride": args.cop_stride,
        "reconstruction_cohort": cohort,
    }
    tasks = [
        common | selection | _fit_admission(selection["motion"], cohort.statuses, fit_status_path)
        for selection in selections
    ]
    records: list[dict[str, Any] | None] = [None] * len(tasks)
    errors: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, task in enumerate(tasks):
            print(f"[{index + 1:02d}/{len(tasks):02d}] {task['motion']}", flush=True)
            try:
                records[index] = _evaluate_motion(task)
            except Exception as exc:
                errors.append(
                    {
                        "motion": task["motion"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(_evaluate_motion, task): index for index, task in enumerate(tasks)}
            for completed, future in enumerate(as_completed(futures), 1):
                index = futures[future]
                motion = tasks[index]["motion"]
                try:
                    records[index] = future.result()
                    print(f"[{completed:02d}/{len(tasks):02d}] ok {motion}", flush=True)
                except Exception as exc:
                    errors.append(
                        {
                            "motion": motion,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[{completed:02d}/{len(tasks):02d}] FAILED {motion}: {exc}", flush=True)

    successful = [record for record in records if record is not None]
    summary = summarize(
        successful,
        errors,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = args.output.resolve()
    write_report(output, successful, summary)
    write_evaluation_provenance(
        output,
        output / "per_motion.csv",
        cohort.run_path,
    )
    print(json.dumps(_json_safe(summary), indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
