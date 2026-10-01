"""Ground-truth-free motion--terrain consistency evaluation.

This evaluator deliberately does not claim geometric accuracy.  It queries every
reconstruction at one frozen set of source-only kinematic contact locations and asks
whether the reconstructed surface is consistent with those observations.  The contact
detector, anatomical offset estimate, thresholds, and denominator are therefore shared
by every method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra._revision import write_git_commit
from terra.benchmarking.reconstruction.core import load_selection
from terra.benchmarking.reconstruction.provenance import (
    RECORD_PROVENANCE_SCHEMA,
    file_sha256,
    validate_run_provenance,
)
from terra.evaluation.reconstruction_statistics import (
    _aggregate_method as _aggregate_method,
)
from terra.evaluation.reconstruction_statistics import (
    _metric as _metric,
)
from terra.evaluation.reconstruction_statistics import (
    _ratio as _ratio,
)
from terra.evaluation.reconstruction_statistics import (
    summarize as summarize,
)
from terra.terrain.metadata import TerrainMetadata

CURRENT_FIT_STATUSES = frozenset({"ok", "cached"})
CONSISTENCY_SCHEMA = "terra.reconstruction-consistency.v1"
DEFAULT_LEVEL_THRESHOLD_M = 0.04
DEFAULT_CONTACT_TOLERANCE_M = 0.05
DEFAULT_FREE_SPACE_TOLERANCE_M = 0.03


@dataclass(frozen=True)
class ContactQuery:
    """One source-only, interval-median foot-contact query."""

    joint: str
    start: int
    end: int
    xyz: np.ndarray


@dataclass(frozen=True)
class Cohort:
    label: str
    root: Path
    method: str
    run: Mapping[str, Any]
    statuses: Mapping[str, Mapping[str, str]]
    legacy_provenance: bool
    method_identity_sha256: str | None
    scientific_identity_sha256: str | None


def _finite_number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _record_path(root: Path, motion: str) -> Path:
    return root / f"{motion.replace('/', '__')}.json"


def _load_statuses(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"reconstruction status table not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"motion", "method", "status", "output", "error"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"status table is missing fields {sorted(missing)}: {path}")
        rows = list(reader)
    result: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(rows, start=2):
        row = {str(key): str(value or "").strip() for key, value in raw.items()}
        motion = row["motion"]
        if not motion or not row["method"] or not row["status"]:
            raise ValueError(f"status row {index} has an empty identity field: {path}")
        if motion in result:
            raise ValueError(f"status table contains duplicate motion {motion!r}: {path}")
        result[motion] = row
    return result


def load_cohort(
    label: str,
    root: Path,
    motions: Sequence[str],
    *,
    allow_legacy_provenance: bool = False,
) -> Cohort:
    """Validate one reconstruction directory and its exact cohort denominator."""

    resolved = root.expanduser().resolve()
    run_path = resolved / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"reconstruction run metadata not found: {run_path}")
    try:
        run = json.loads(run_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction run metadata: {run_path}") from error
    if not isinstance(run, dict):
        raise ValueError(f"reconstruction run metadata must contain an object: {run_path}")
    method = run.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"reconstruction run has no method: {run_path}")
    if run.get("motions") != len(motions):
        raise ValueError(f"reconstruction run motion count does not match the manifest: {run_path}")

    legacy = not isinstance(run.get("provenance"), dict)
    if legacy:
        if not allow_legacy_provenance:
            raise ValueError(
                f"reconstruction run has no content-addressed provenance: {run_path}; "
                "use --allow-legacy-provenance only for explicitly exploratory rescoring"
            )
        method_hash = scientific_hash = None
    else:
        provenance = validate_run_provenance(run, run_path)
        method_hash = provenance.method_identity_sha256
        scientific_hash = provenance.scientific_identity_sha256

    statuses = _load_statuses(resolved / "status.csv")
    if set(statuses) != set(motions):
        missing = sorted(set(motions) - set(statuses))
        extra = sorted(set(statuses) - set(motions))
        raise ValueError(
            f"status denominator does not match the manifest for {label!r}; missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(row["method"] != method for row in statuses.values()):
        raise ValueError(f"status method does not match run metadata for {label!r}")
    return Cohort(
        label=label,
        root=resolved,
        method=method,
        run=run,
        statuses=statuses,
        legacy_provenance=legacy,
        method_identity_sha256=method_hash,
        scientific_identity_sha256=scientific_hash,
    )


def load_record(cohort: Cohort, motion: str) -> dict[str, Any]:
    status = cohort.statuses[motion]
    if status["status"] not in CURRENT_FIT_STATUSES:
        detail = f"; {status['error']}" if status["error"] else ""
        raise ValueError(f"candidate fit status is {status['status']!r}{detail}")
    path = _record_path(cohort.root, motion)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"reconstruction record not found: {path}")
    if Path(status["output"]).name != path.name:
        raise ValueError(f"status output does not name the expected record for {motion!r}")
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction record: {path}") from error
    if not isinstance(record, dict) or record.get("motion") != motion or record.get("method") != cohort.method:
        raise ValueError(f"reconstruction record identity mismatch for {motion!r}: {path}")
    if not all(isinstance(record.get(key), dict) for key in ("terrain", "fit", "validation")):
        raise ValueError(f"reconstruction scientific result is incomplete for {motion!r}: {path}")
    if not cohort.legacy_provenance:
        expected = {
            "schema": RECORD_PROVENANCE_SCHEMA,
            "method_identity_sha256": cohort.method_identity_sha256,
            "scientific_identity_sha256": cohort.scientific_identity_sha256,
        }
        if record.get("provenance") != expected:
            raise ValueError(f"reconstruction record scientific identity mismatch for {motion!r}")
    return record


def reference_contacts(record: Mapping[str, Any]) -> tuple[ContactQuery, ...]:
    """Read source contacts serialized before the reference method fitted terrain."""

    intervals = record.get("fit", {}).get("support_intervals")
    if not isinstance(intervals, list):
        raise ValueError("contact reference fit has no support_intervals list")
    contacts: list[ContactQuery] = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise ValueError(f"contact reference interval {index} is not an object")
        if interval.get("kind") != "foot":
            continue
        joint = interval.get("link")
        start, end = interval.get("start"), interval.get("end")
        try:
            xyz = np.asarray(interval.get("surface_xyz_m"), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"contact reference interval {index} has invalid surface_xyz_m") from error
        valid_frames = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end
        )
        if not isinstance(joint, str) or not joint or xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f"contact reference interval {index} has an invalid contact point")
        if not valid_frames:
            raise ValueError(f"contact reference interval {index} has invalid frame bounds")
        contacts.append(ContactQuery(joint, start, end, xyz))
    if not contacts:
        raise ValueError("contact reference contains no foot contacts")
    return tuple(contacts)


def source_surface_offsets(
    contacts: Sequence[ContactQuery],
    *,
    level_threshold_m: float = DEFAULT_LEVEL_THRESHOLD_M,
) -> dict[str, float]:
    """Estimate paired anatomical offsets from only the lowest source-contact cluster."""

    if not math.isfinite(level_threshold_m) or level_threshold_m <= 0.0:
        raise ValueError("level_threshold_m must be finite and positive")
    offsets: dict[str, float] = {}
    for joint in sorted({contact.joint for contact in contacts}):
        heights = sorted(float(contact.xyz[2]) for contact in contacts if contact.joint == joint)
        lowest = [heights[0]]
        for height in heights[1:]:
            if height - lowest[-1] > level_threshold_m:
                break
            lowest.append(height)
        offsets[joint] = float(np.mean(lowest))
    # Left/right probes of the same anatomical class should have one physical offset.
    # The lower estimate prevents a side first observed on raised terrain from defining
    # the datum, matching the production source-only calibration rule.
    for left, right in (("L_Ankle", "R_Ankle"), ("L_Toe", "R_Toe")):
        if left in offsets and right in offsets:
            paired = min(offsets[left], offsets[right])
            offsets[left] = offsets[right] = paired
    return offsets


def _support_pair_errors(
    contacts: Sequence[ContactQuery],
    target: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Compare toe--ankle height differences for overlapping same-foot contacts."""

    errors: list[float] = []
    for toe_index, toe in enumerate(contacts):
        if not toe.joint.endswith("_Toe"):
            continue
        for ankle_index, ankle in enumerate(contacts):
            if ankle.joint != f"{toe.joint[0]}_Ankle":
                continue
            if toe.start < ankle.end and ankle.start < toe.end:
                observed_delta = target[toe_index] - target[ankle_index]
                predicted_delta = predicted[toe_index] - predicted[ankle_index]
                errors.append(abs(float(predicted_delta - observed_delta)))
    return np.asarray(errors, dtype=float)


def score_contacts(
    contacts: Sequence[ContactQuery],
    terrain: Any,
    *,
    level_threshold_m: float = DEFAULT_LEVEL_THRESHOLD_M,
    contact_tolerance_m: float = DEFAULT_CONTACT_TOLERANCE_M,
) -> dict[str, float | int | None]:
    """Score one terrain at a fixed set of source-only interval-median contacts."""

    if not contacts:
        raise ValueError("contacts must not be empty")
    if not math.isfinite(contact_tolerance_m) or contact_tolerance_m <= 0.0:
        raise ValueError("contact_tolerance_m must be finite and positive")
    offsets = source_surface_offsets(contacts, level_threshold_m=level_threshold_m)
    target = np.asarray([contact.xyz[2] - offsets[contact.joint] for contact in contacts], dtype=float)
    ground = terrain.walkable
    predicted = np.asarray(
        [ground.height_at(float(contact.xyz[0]), float(contact.xyz[1])) for contact in contacts],
        dtype=float,
    )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("candidate terrain returned a non-finite contact height")
    error = predicted - target
    absolute = np.abs(error)
    raised_target = target > level_threshold_m
    raised_prediction = predicted > level_threshold_m
    tp = int(np.sum(raised_target & raised_prediction))
    fp = int(np.sum(~raised_target & raised_prediction))
    fn = int(np.sum(raised_target & ~raised_prediction))
    tn = int(np.sum(~raised_target & ~raised_prediction))
    pair_error = _support_pair_errors(contacts, target, predicted)
    correlation = None
    if float(np.ptp(target)) >= 2.0 * level_threshold_m and float(np.std(predicted)) > 1e-12:
        value = float(np.corrcoef(target, predicted)[0, 1])
        correlation = value if math.isfinite(value) else None
    return {
        "contact_n": len(contacts),
        "contact_height_mae_mm": float(np.mean(absolute) * 1000.0),
        "contact_height_rmse_mm": float(np.sqrt(np.mean(np.square(error))) * 1000.0),
        "contact_height_p95_mm": float(np.percentile(absolute, 95) * 1000.0),
        "contact_height_max_mm": float(np.max(absolute) * 1000.0),
        "contact_height_bias_mm": float(np.mean(error) * 1000.0),
        "contact_consistent_n": int(np.sum(absolute <= contact_tolerance_m)),
        "contact_penetrating_n": int(np.sum(error > contact_tolerance_m)),
        "contact_floating_n": int(np.sum(error < -contact_tolerance_m)),
        "contact_consistent_pct": float(np.mean(absolute <= contact_tolerance_m) * 100.0),
        "contact_penetrating_pct": float(np.mean(error > contact_tolerance_m) * 100.0),
        "contact_floating_pct": float(np.mean(error < -contact_tolerance_m) * 100.0),
        "raised_target_n": int(np.sum(raised_target)),
        "raised_prediction_n": int(np.sum(raised_prediction)),
        "raised_true_positive_n": tp,
        "raised_false_positive_n": fp,
        "raised_false_negative_n": fn,
        "raised_true_negative_n": tn,
        "raised_precision_pct": _ratio(tp, tp + fp),
        "raised_recall_pct": _ratio(tp, tp + fn),
        "raised_f1_pct": _ratio(2 * tp, 2 * tp + fp + fn),
        "support_pair_n": int(pair_error.size),
        "support_pair_height_mae_mm": (float(np.mean(pair_error) * 1000.0) if pair_error.size else None),
        "contact_height_correlation": correlation,
        "source_support_height_span_mm": float(np.ptp(target) * 1000.0),
    }


def _source_input_matches(
    reference: Mapping[str, Any], identity: Mapping[str, Any], joints: np.ndarray, fps: float
) -> None:
    recorded = reference.get("fit", {}).get("input")
    if not isinstance(recorded, dict):
        raise ValueError("contact reference has no fitted-motion input identity")
    metadata = recorded.get("joints")
    if not isinstance(metadata, dict) or metadata.get("shape") != list(joints.shape):
        raise ValueError("recomputed source joint shape does not match the contact reference")
    if not math.isclose(_finite_number(recorded.get("fps"), "recorded fps"), fps, abs_tol=1e-9):
        raise ValueError("recomputed source fps does not match the contact reference")
    expected_normalization = recorded.get("normalization")
    if expected_normalization is not None and identity.get("normalization") != expected_normalization:
        raise ValueError("recomputed source normalization does not match the contact reference")


def load_source_landmarks(
    motion: str,
    reference: Mapping[str, Any],
    *,
    source_root: Path,
    smpl_model_path: Path,
    fitted_shape_cache: Path,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    """Recompute the same fitted source landmarks used by reconstruction."""

    from terra.benchmarking.reconstruction.methods.common import prepare_motion_landmarks

    landmarks, identity = prepare_motion_landmarks(
        motion,
        "MyoFullBody",
        source_path=source_root / f"{motion}.npz",
        smpl_model_path=smpl_model_path,
        cache_root=fitted_shape_cache,
    )
    _source_input_matches(reference, identity, landmarks.joints, landmarks.fps)
    return landmarks.joints, landmarks.joint_names, landmarks.fps


def score_free_space(
    contacts: Sequence[ContactQuery],
    terrain: Any,
    joints: np.ndarray,
    joint_names: Sequence[str],
    fps: float,
    *,
    tolerance_m: float = DEFAULT_FREE_SPACE_TOLERANCE_M,
) -> dict[str, float | int | bool]:
    """Measure candidate-solid penetration by non-support source landmarks."""

    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("free-space tolerance must be finite and positive")
    motion = np.asarray(joints, dtype=float)
    if motion.ndim != 3 or motion.shape[2] != 3 or not np.all(np.isfinite(motion)):
        raise ValueError("source joints must have finite shape (T, J, 3)")
    names = tuple(joint_names)
    if len(names) != motion.shape[1]:
        raise ValueError("joint_names length does not match source joints")
    supported = np.zeros(motion.shape[:2], dtype=bool)
    name_to_index = {name: index for index, name in enumerate(names)}
    padding = max(1, round(0.020 * fps))
    for contact in contacts:
        if contact.joint not in name_to_index:
            continue
        supported[
            max(0, contact.start - padding) : min(len(motion), contact.end + padding),
            name_to_index[contact.joint],
        ] = True
    penetration = np.asarray(terrain.penetration(motion.reshape(-1, 3)), dtype=float).reshape(motion.shape[:2])
    if penetration.shape != supported.shape or not np.all(np.isfinite(penetration)):
        raise ValueError("candidate terrain returned invalid penetration depths")
    free = penetration[~supported]
    violating = free > tolerance_m
    positive = free[free > 0.0]
    return {
        "free_space_query_n": int(free.size),
        "free_space_penetrating_n": int(np.sum(violating)),
        "free_space_penetrating_pct": float(np.mean(violating) * 100.0),
        "free_space_penetration_max_mm": float(np.max(free) * 1000.0) if free.size else 0.0,
        "free_space_penetration_mean_positive_mm": (float(np.mean(positive) * 1000.0) if positive.size else 0.0),
        "free_space_violation": bool(np.any(violating)),
    }


def _evaluate_motions(
    motions: Sequence[str],
    reference: Cohort,
    cohorts: Sequence[Cohort],
    *,
    level_threshold_m: float = DEFAULT_LEVEL_THRESHOLD_M,
    contact_tolerance_m: float = DEFAULT_CONTACT_TOLERANCE_M,
    free_space_tolerance_m: float = DEFAULT_FREE_SPACE_TOLERANCE_M,
    source_root: Path | None = None,
    smpl_model_path: Path | None = None,
    fitted_shape_cache: Path | None = None,
) -> list[dict[str, Any]]:
    source_requested = source_root is not None
    rows: list[dict[str, Any]] = []
    for motion in motions:
        try:
            reference_record = load_record(reference, motion)
            contacts = reference_contacts(reference_record)
            source = (
                load_source_landmarks(
                    motion,
                    reference_record,
                    source_root=source_root,
                    smpl_model_path=smpl_model_path,
                    fitted_shape_cache=fitted_shape_cache,
                )
                if source_requested
                else None
            )
            reference_error = ""
        except Exception as error:
            contacts = ()
            source = None
            reference_error = f"contact reference: {type(error).__name__}: {error}".replace("\n", " ")[:1000]
        for cohort in cohorts:
            row: dict[str, Any] = {
                "method": cohort.label,
                "reconstruction_method": cohort.method,
                "motion": motion,
                "candidate_fit_status": cohort.statuses[motion]["status"],
                "legacy_provenance": cohort.legacy_provenance,
                "error": reference_error,
            }
            if reference_error:
                rows.append(row)
                continue
            try:
                record = load_record(cohort, motion)
                terrain = TerrainMetadata.from_dict(record["terrain"]).terrain
                row.update(
                    score_contacts(
                        contacts,
                        terrain,
                        level_threshold_m=level_threshold_m,
                        contact_tolerance_m=contact_tolerance_m,
                    )
                )
                row["n_boxes"] = len(terrain.boxes)
                row["n_walkable_boxes"] = len(terrain.walkable.boxes)
                if source is not None:
                    row.update(
                        score_free_space(
                            contacts,
                            terrain,
                            *source,
                            tolerance_m=free_space_tolerance_m,
                        )
                    )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}".replace("\n", " ")[:1000]
            rows.append(row)
    return rows


def evaluate(
    motions: Sequence[str],
    reference: Cohort,
    cohorts: Sequence[Cohort],
    *,
    level_threshold_m: float = DEFAULT_LEVEL_THRESHOLD_M,
    contact_tolerance_m: float = DEFAULT_CONTACT_TOLERANCE_M,
    free_space_tolerance_m: float = DEFAULT_FREE_SPACE_TOLERANCE_M,
    source_root: Path | None = None,
    smpl_model_path: Path | None = None,
    fitted_shape_cache: Path | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate all method-motion pairs, preserving failed fits in the denominator."""

    source_requested = source_root is not None or smpl_model_path is not None or fitted_shape_cache is not None
    if source_requested and (source_root is None or smpl_model_path is None or fitted_shape_cache is None):
        raise ValueError("source-root, smpl-model-path, and fitted-shape-cache must be supplied together")
    if workers < 1:
        raise ValueError("workers must be positive")
    options = {
        "level_threshold_m": level_threshold_m,
        "contact_tolerance_m": contact_tolerance_m,
        "free_space_tolerance_m": free_space_tolerance_m,
        "source_root": source_root,
        "smpl_model_path": smpl_model_path,
        "fitted_shape_cache": fitted_shape_cache,
    }
    if workers == 1 or len(motions) < 2:
        return _evaluate_motions(motions, reference, cohorts, **options)
    chunks = [tuple(chunk) for chunk in np.array_split(np.asarray(motions, dtype=object), min(workers, len(motions)))]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_evaluate_motions, chunk, reference, cohorts, **options) for chunk in chunks]
        return [row for future in futures for row in future.result()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["motion"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: object, decimals: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{decimals}f}"


def _table(summary: Mapping[str, Any], *, common_success: bool = False) -> str:
    section = "common_success_methods" if common_success else "methods"
    title = "Common-success sensitivity" if common_success else "All requested motions"
    lines = [
        f"## {title}",
        "",
        "| Method | Success | Contact MAE (mm) | Within tolerance (%) | Penetrating / floating (%) | Raised P / R / F1 (%) | Free-space violation motions | Boxes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in summary[section].items():
        metrics = item["metrics"]
        mae = metrics["contact_height_mae_mm"]
        boxes = metrics["n_boxes"]
        free = (
            "n/a"
            if not item["free_space_queries"]
            else f"{item['motions_with_free_space_violation']}/{item['successful']}"
        )
        lines.append(
            f"| {label} | {item['successful']}/{item['selected']} | "
            f"{_format(mae['mean'])} ± {_format(mae['std'])} [n={mae['n']}] | "
            f"{_format(item['contact_consistent_pct'], 1)} | "
            f"{_format(item['contact_penetrating_pct'], 1)} / {_format(item['contact_floating_pct'], 1)} | "
            f"{_format(item['raised_precision_pct'], 1)} / {_format(item['raised_recall_pct'], 1)} / "
            f"{_format(item['raised_f1_pct'], 1)} | {free} | "
            f"{_format(boxes['mean'], 1)} ± {_format(boxes['std'], 1)} |"
        )
    return "\n".join(lines)


def write_report(output: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_motion.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    explanation = [
        "# Ground-truth-free reconstruction consistency",
        "",
        "These are source-motion consistency diagnostics, not terrain-accuracy metrics. Every method is queried at "
        "the same contact locations, using anatomical offsets estimated once from the reference contacts. Positive "
        "signed error means the predicted support lies above the observed sole (penetration); negative error means "
        "it lies below (floating). Raised-support precision/recall classifies both observed and predicted support "
        "with the frozen level threshold. Free-space penetration uses non-support source landmarks and is reported "
        "only when the source assets were explicitly supplied.",
        "",
        _table(summary),
        "",
        _table(summary, common_success=True),
        "",
    ]
    (output / "table.md").write_text("\n".join(explanation))
    write_git_commit(output)


def _assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--method must be LABEL=DIR, got {value!r}")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"--method must be LABEL=DIR, got {value!r}")
    return label.strip(), Path(path).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="terra evaluate reconstruction-consistency", description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--contact-reference-dir", type=Path, required=True)
    result.add_argument("--method", action="append", type=_assignment, required=True, metavar="LABEL=DIR")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--level-threshold", type=float, default=DEFAULT_LEVEL_THRESHOLD_M)
    result.add_argument("--contact-tolerance", type=float, default=DEFAULT_CONTACT_TOLERANCE_M)
    result.add_argument("--free-space-tolerance", type=float, default=DEFAULT_FREE_SPACE_TOLERANCE_M)
    result.add_argument("--source-root", type=Path)
    result.add_argument("--smpl-model-path", type=Path)
    result.add_argument("--fitted-shape-cache", type=Path, help="cache root containing MyoFullBody/shape_optimized.pkl")
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--allow-failures", action="store_true")
    result.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help="Allow exploratory scoring of pre-provenance records; the limitation is marked in every output row.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    selection = load_selection(args.manifest)
    labels = [label for label, _path in args.method]
    if len(labels) != len(set(labels)):
        raise SystemExit("--method labels must be unique")
    reference = load_cohort(
        "contact-reference",
        args.contact_reference_dir,
        selection.motions,
        allow_legacy_provenance=args.allow_legacy_provenance,
    )
    cohorts = [
        load_cohort(
            label,
            path,
            selection.motions,
            allow_legacy_provenance=args.allow_legacy_provenance,
        )
        for label, path in args.method
    ]
    rows = evaluate(
        selection.motions,
        reference,
        cohorts,
        level_threshold_m=args.level_threshold,
        contact_tolerance_m=args.contact_tolerance,
        free_space_tolerance_m=args.free_space_tolerance,
        source_root=None if args.source_root is None else args.source_root.expanduser().resolve(),
        smpl_model_path=None if args.smpl_model_path is None else args.smpl_model_path.expanduser().resolve(),
        fitted_shape_cache=(
            None if args.fitted_shape_cache is None else args.fitted_shape_cache.expanduser().resolve()
        ),
        workers=args.workers,
    )
    manifest_rows = {row["motion"]: row for row in selection.rows}
    for row in rows:
        source = manifest_rows[row["motion"]]
        row.update(
            benchmark_group=source.get("benchmark_group", ""),
            terrain_class=source.get("terrain_class", ""),
            expected_family=source.get("expected_family", ""),
        )
    summary = summarize(rows, labels, len(selection.motions))
    for field in ("benchmark_group", "terrain_class"):
        summary[f"by_{field}"] = {
            value: summarize(
                [row for row in rows if row.get(field) == value],
                labels,
                sum(source.get(field) == value for source in selection.rows),
            )
            for value in sorted({source.get(field, "") for source in selection.rows} - {""})
        }
    summary["definitions"] = {
        "contact_reference": (
            "interval-median foot contacts detected before the contact-reference terrain fit; one event, one vote"
        ),
        "source_surface_offset": (
            "mean of each probe's lowest contact-height cluster, reconciled to the lower bilateral estimate"
        ),
        "contact_signed_error": "predicted terrain height minus source contact height after anatomical offset",
        "level_threshold_m": args.level_threshold,
        "contact_tolerance_m": args.contact_tolerance,
        "free_space_tolerance_m": args.free_space_tolerance,
        "support_pair": "absolute error in toe-minus-ankle support-height difference during overlapping contacts",
        "raised_classification": "source or predicted support height strictly above level_threshold_m",
        "aggregation": "continuous metrics are unweighted per-motion mean and population standard deviation",
        "interpretation": "in-sample motion consistency, not independent geometric accuracy",
    }
    output = args.output.expanduser().resolve()
    write_report(output, rows, summary)
    inputs = {
        "schema": CONSISTENCY_SCHEMA,
        "manifest": {"path": str(selection.path), "sha256": file_sha256(selection.path)},
        "contact_reference": {
            "path": str(reference.root),
            "run_sha256": file_sha256(reference.root / "run.json"),
            "status_sha256": file_sha256(reference.root / "status.csv"),
            "legacy_provenance": reference.legacy_provenance,
        },
        "methods": {
            cohort.label: {
                "path": str(cohort.root),
                "method": cohort.method,
                "run_sha256": file_sha256(cohort.root / "run.json"),
                "status_sha256": file_sha256(cohort.root / "status.csv"),
                "legacy_provenance": cohort.legacy_provenance,
            }
            for cohort in cohorts
        },
        "source_landmarks_recomputed": args.source_root is not None,
        "source_root": None if args.source_root is None else str(args.source_root.expanduser().resolve()),
        "smpl_model_path": (None if args.smpl_model_path is None else str(args.smpl_model_path.expanduser().resolve())),
        "fitted_shape_cache": (
            None if args.fitted_shape_cache is None else str(args.fitted_shape_cache.expanduser().resolve())
        ),
        "per_motion_sha256": file_sha256(output / "per_motion.csv"),
        "summary_sha256": file_sha256(output / "summary.json"),
    }
    (output / "evaluation.json").write_text(json.dumps(inputs, indent=2, allow_nan=False) + "\n")
    errors = sum(bool(row.get("error")) for row in rows)
    print((output / "table.md").read_text(), end="")
    empty_methods = [label for label, item in summary["methods"].items() if item["successful"] == 0]
    if empty_methods:
        print(f"No successful evaluations for: {', '.join(empty_methods)}")
        return 1
    return 1 if errors and not args.allow_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
