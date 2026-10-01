"""Run a checkpoint across a versioned cohort of biomechanical comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra.biomechanics_validation import (
    COMPARISON_SCHEMA_VERSION,
    NoSuccessfulRolloutError,
    compare_checkpoint,
    load_trace,
    resolve_trace_match,
)

COHORT_SCHEMA_VERSION = 4
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_REQUIRED_COLUMNS = (
    "case_id",
    "motion",
    "dataset",
    "cache_subdir",
    "subject",
    "motion_type",
    "split",
    "evaluation_tier",
    "expected_emg",
    "expected_grf",
)
_SUMMARY_METRICS = (
    "mean_emg_zero_lag_waveform_correlation",
    "median_emg_peak_phase_error_percent",
    "mean_normal_grf_waveform_correlation",
    "mean_normal_grf_rmse_bw",
    "mean_normal_grf_rmse_percent_body_weight",
    "mean_impulse_error_bw_phase",
    "mean_impulse_relative_error_percent",
)


@dataclass(frozen=True)
class CohortCase:
    case_id: str
    motion: str
    dataset: str
    subject: str
    motion_type: str
    split: str
    evaluation_tier: str
    expected_emg: bool
    expected_grf: bool
    cache_root: Path


def _boolean(value: str, *, field: str, row: int) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cohort manifest row {row} has invalid {field} boolean: {value!r}")


def _scalar_boolean(trace: dict[str, np.ndarray], key: str) -> bool:
    return bool(np.asarray(trace[key]).reshape(()))


def load_cohort_manifest(
    manifest: str | Path,
    *,
    artifact_root: str | Path,
    trace_root: str | Path | None = None,
    policy_root: str | Path | None = None,
    case_ids: Iterable[str] = (),
    tiers: Iterable[str] = (),
) -> list[CohortCase]:
    """Validate a cohort manifest against the exact motion-to-trace registry."""

    path = Path(manifest).expanduser().resolve()
    artifacts = Path(artifact_root).expanduser().resolve()
    traces = Path(trace_root).expanduser().resolve() if trace_root else artifacts / "biomechanics/validation-traces"
    selected_ids = set(case_ids)
    selected_tiers = set(tiers)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in _REQUIRED_COLUMNS if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"cohort manifest is missing columns: {', '.join(missing)}")
        rows = list(reader)
    cases: list[CohortCase] = []
    seen_ids: set[str] = set()
    seen_motions: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        case_id = row["case_id"].strip()
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError(f"cohort manifest row {row_number} has invalid case_id: {case_id!r}")
        if case_id in seen_ids:
            raise ValueError(f"cohort manifest repeats case_id {case_id!r}")
        seen_ids.add(case_id)
        motion = row["motion"].strip()
        if motion in seen_motions:
            raise ValueError(f"cohort manifest repeats motion {motion!r}")
        seen_motions.add(motion)
        if selected_ids and case_id not in selected_ids:
            continue
        tier = row["evaluation_tier"].strip()
        if selected_tiers and tier not in selected_tiers:
            continue
        match = resolve_trace_match(motion, trace_root=traces, artifact_root=artifacts)
        declared = (row["dataset"].strip(), row["subject"].strip(), row["motion_type"].strip())
        resolved = (match.dataset, match.subject, match.motion_type)
        if declared != resolved:
            raise ValueError(
                f"cohort manifest row {row_number} metadata {declared!r} does not match registry {resolved!r}"
            )
        expected_emg = _boolean(row["expected_emg"], field="expected_emg", row=row_number)
        expected_grf = _boolean(row["expected_grf"], field="expected_grf", row=row_number)
        trace = load_trace(match.trace_path)
        available = (_scalar_boolean(trace, "emg_available"), _scalar_boolean(trace, "grf_available"))
        if (expected_emg, expected_grf) != available:
            raise ValueError(
                f"cohort manifest row {row_number} expected signals {(expected_emg, expected_grf)!r} "
                f"do not match trace {available!r}"
            )
        cache_subdir = Path(row["cache_subdir"].strip())
        if not str(cache_subdir) or cache_subdir.is_absolute() or ".." in cache_subdir.parts:
            raise ValueError(f"cohort manifest row {row_number} has unsafe cache_subdir: {cache_subdir!s}")
        cases.append(
            CohortCase(
                case_id=case_id,
                motion=match.motion,
                dataset=match.dataset,
                subject=match.subject,
                motion_type=match.motion_type,
                split=row["split"].strip(),
                evaluation_tier=tier,
                expected_emg=expected_emg,
                expected_grf=expected_grf,
                cache_root=(
                    Path(policy_root).expanduser().resolve() / match.dataset / "cache"
                    if policy_root is not None
                    else artifacts / cache_subdir
                ),
            )
        )
    if selected_ids - seen_ids:
        raise ValueError(f"unknown cohort case ids: {', '.join(sorted(selected_ids - seen_ids))}")
    if not cases:
        raise ValueError("cohort selection is empty")
    return cases


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty cohort table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _existing_report(
    path: Path,
    *,
    case: CohortCase,
    checkpoint: Path,
    trials: int,
    first_seed: int,
    deterministic: bool,
    termination_threshold_m: float | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    # Earlier versions can contain wrong-motion materialized-cache collisions
    # in every dataset. They are never safe to resume after the identity audit.
    report_schema = report.get("schema_version")
    if report_schema != COMPARISON_SCHEMA_VERSION:
        return None
    source_hashes = report.get("source_sha256", {})
    from terra import biomechanics_validation

    if (
        source_hashes.get("evaluation")
        != hashlib.sha256(Path(biomechanics_validation.__file__).read_bytes()).hexdigest()
    ):
        return None
    expected_seeds = list(range(first_seed, first_seed + trials))
    compatible = (
        report.get("motion") == case.motion
        and Path(str(report.get("checkpoint", ""))).resolve() == checkpoint
        and report.get("policy_mode") == ("deterministic" if deterministic else "stochastic")
        and report.get("seeds") == expected_seeds
        and report.get("termination_threshold_m") == termination_threshold_m
        and report.get("reference", {}).get("motion") == case.motion
        and report.get("reference", {}).get("arrays_verified") is True
        and Path(report.get("reference", {}).get("cache_root", "")).resolve() == case.cache_root.resolve()
    )
    if not compatible:
        raise ValueError(f"existing comparison is incompatible with this run: {path}; use --force to replace it")
    return report


def _case_row(case: CohortCase, output: Path, report: dict[str, Any] | None, error: Exception | None) -> dict[str, Any]:
    summary = report.get("summary", {}) if report else {}
    rollout_failure = error if isinstance(error, NoSuccessfulRolloutError) else None
    reference = report.get("reference", {}) if report else getattr(error, "reference", {})
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "motion": case.motion,
        "dataset": case.dataset,
        "subject": case.subject,
        "motion_type": case.motion_type,
        "split": case.split,
        "evaluation_tier": case.evaluation_tier,
        "expected_emg": case.expected_emg,
        "expected_grf": case.expected_grf,
        "status": "complete" if report else "failed",
        "output_dir": str(output),
        "successful_rollouts": summary.get(
            "successful_rollouts", rollout_failure.successful_rollouts if rollout_failure else ""
        ),
        "failed_rollouts": summary.get("failed_rollouts", rollout_failure.failed_rollouts if rollout_failure else ""),
        "measured_rollouts": summary.get(
            "measured_rollouts", getattr(rollout_failure, "measured_rollouts", 0) if rollout_failure else ""
        ),
        "unmeasured_rollouts": summary.get(
            "unmeasured_rollouts",
            (rollout_failure.successful_rollouts + rollout_failure.failed_rollouts if rollout_failure else ""),
        ),
        "measured_early_terminated_rollouts": summary.get(
            "measured_early_terminated_rollouts",
            getattr(rollout_failure, "measured_early_terminated_rollouts", 0) if rollout_failure else "",
        ),
        "completed_gaits": summary.get(
            "completed_gaits", getattr(rollout_failure, "completed_gaits", 0) if rollout_failure else ""
        ),
        "completed_gaits_in_early_terminated_rollouts": summary.get(
            "completed_gaits_in_early_terminated_rollouts", 0 if rollout_failure else ""
        ),
        "error_type": type(error).__name__ if error else "",
        "error": str(error) if error else "",
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "reference_verified": reference.get("arrays_verified", False),
        "reference_qpos_sha256": reference.get("loaded_qpos_sha256", ""),
        "reference_frames": reference.get("loaded_frames", ""),
        "reference_method": reference.get("retargeting_method", ""),
    }
    row.update({metric: summary.get(metric, "") for metric in _SUMMARY_METRICS})
    return row


def _finite_mean(rows: Sequence[dict[str, Any]], field: str, *, median: bool = False) -> float | str:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return ""
    return float(np.median(values) if median else np.mean(values))


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys: list[tuple[str, str, str]] = [("all", "all", "all")]
    group_keys.extend(sorted({(row["dataset"], "all", row["evaluation_tier"]) for row in rows}))
    group_keys.extend(sorted({(row["dataset"], row["motion_type"], row["evaluation_tier"]) for row in rows}))
    result = []
    for dataset, motion_type, tier in group_keys:
        selected = [
            row
            for row in rows
            if (dataset == "all" or row["dataset"] == dataset)
            and (motion_type == "all" or row["motion_type"] == motion_type)
            and (tier == "all" or row["evaluation_tier"] == tier)
        ]
        complete = [row for row in selected if row["status"] == "complete"]
        result.append(
            {
                "dataset": dataset,
                "motion_type": motion_type,
                "evaluation_tier": tier,
                "cases": len(selected),
                "completed_cases": len(complete),
                "failed_cases": len(selected) - len(complete),
                "successful_rollouts": sum(int(row["successful_rollouts"] or 0) for row in selected),
                "failed_rollouts": sum(int(row["failed_rollouts"] or 0) for row in selected),
                "measured_rollouts": sum(int(row["measured_rollouts"] or 0) for row in selected),
                "unmeasured_rollouts": sum(int(row["unmeasured_rollouts"] or 0) for row in selected),
                "measured_early_terminated_rollouts": sum(
                    int(row["measured_early_terminated_rollouts"] or 0) for row in selected
                ),
                "completed_gaits": sum(int(row["completed_gaits"] or 0) for row in selected),
                "completed_gaits_in_early_terminated_rollouts": sum(
                    int(row["completed_gaits_in_early_terminated_rollouts"] or 0) for row in selected
                ),
                **{
                    metric: _finite_mean(
                        complete,
                        metric,
                        median=metric == "median_emg_peak_phase_error_percent",
                    )
                    for metric in _SUMMARY_METRICS
                },
            }
        )
    return result


def compare_cohort(
    manifest: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    data_root: str | Path,
    artifact_root: str | Path,
    trace_root: str | Path | None = None,
    policy_root: str | Path | None = None,
    trials: int = 10,
    first_seed: int = 0,
    deterministic: bool = False,
    case_ids: Iterable[str] = (),
    tiers: Iterable[str] = (),
    force: bool = False,
    termination_threshold_m: float | None = None,
) -> dict[str, Any]:
    """Run or resume every selected comparison and write cohort-level tables."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if termination_threshold_m is not None and termination_threshold_m <= 0:
        raise ValueError("termination_threshold_m must be positive")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_path}")
    artifacts = Path(artifact_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    cases = load_cohort_manifest(
        manifest,
        artifact_root=artifacts,
        trace_root=trace_root,
        policy_root=policy_root,
        case_ids=case_ids,
        tiers=tiers,
    )
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_output = output / "cases" / case.case_id
        report = None
        error = None
        try:
            if not force:
                report = _existing_report(
                    case_output / "summary.json",
                    case=case,
                    checkpoint=checkpoint_path,
                    trials=trials,
                    first_seed=first_seed,
                    deterministic=deterministic,
                    termination_threshold_m=termination_threshold_m,
                )
            if report is None:
                print(f"[BiomechanicsCohort] case {index}/{len(cases)}: {case.case_id}", flush=True)
                report = compare_checkpoint(
                    case.motion,
                    checkpoint_path,
                    output_dir=case_output,
                    data_root=data_root,
                    artifact_root=artifacts,
                    trace_root=trace_root,
                    cache_root=case.cache_root,
                    trials=trials,
                    first_seed=first_seed,
                    deterministic=deterministic,
                    termination_threshold_m=termination_threshold_m,
                )
            else:
                print(f"[BiomechanicsCohort] resume: {case.case_id} already complete", flush=True)
        except Exception as exc:  # Keep the remaining cohort observable and resumable.
            error = exc
            print(f"[BiomechanicsCohort] failed: {case.case_id}: {type(exc).__name__}: {exc}", flush=True)
        rows.append(_case_row(case, case_output, report, error))
        _atomic_csv(output / "cases.csv", rows)
    aggregates = _aggregate_rows(rows)
    _atomic_csv(output / "aggregates.csv", aggregates)
    failed = sum(row["status"] == "failed" for row in rows)
    report = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "manifest": str(Path(manifest).expanduser().resolve()),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "source_sha256": {
            "validation": hashlib.sha256(
                Path(__file__).with_name("biomechanics_validation.py").read_bytes()
            ).hexdigest(),
            "cohort": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "averages": hashlib.sha256(
                (Path(__file__).parent / "datasets/biomechanics_averages.py").read_bytes()
            ).hexdigest(),
        },
        "checkpoint": str(checkpoint_path),
        "policy_mode": "deterministic" if deterministic else "stochastic",
        "trials_per_case": trials,
        "termination_threshold_m": termination_threshold_m,
        "first_seed": first_seed,
        "selected_tiers": sorted(set(tiers)),
        "cases": len(rows),
        "completed_cases": len(rows) - failed,
        "failed_cases": failed,
        "artifacts": {
            "cases": str(output / "cases.csv"),
            "aggregates": str(output / "aggregates.csv"),
            "summary": str(output / "summary.json"),
        },
    }
    _atomic_json(output / "summary.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    from terra.paths import StorageRoots

    repository_root = Path(__file__).resolve().parents[2]
    roots = StorageRoots.from_environment(repository_root)
    parser = argparse.ArgumentParser(
        prog="terra biomechanics cohort",
        description="Run a checkpoint over a versioned EMG/GRF comparison cohort.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=roots.data_root)
    parser.add_argument("--artifact-root", type=Path, default=roots.artifact_root)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--policy-root", type=Path, help="override method caches with ROOT/<dataset>/cache")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--case-id", action="append", default=[], help="run only this case; repeat as needed")
    parser.add_argument("--tier", action="append", default=[], help="run only this evaluation tier; repeat as needed")
    parser.add_argument("--force", action="store_true", help="replace compatible completed case outputs")
    parser.add_argument("--allow-failures", action="store_true", help="return success when one or more cases fail")
    parser.add_argument(
        "--termination-threshold-m",
        type=float,
        help="override both global and core upper-body mean site-deviation termination thresholds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_cohort(
        args.manifest,
        args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        trace_root=args.trace_root,
        policy_root=args.policy_root,
        trials=args.trials,
        first_seed=args.first_seed,
        deterministic=args.deterministic,
        case_ids=args.case_id,
        tiers=args.tier,
        force=args.force,
        termination_threshold_m=args.termination_threshold_m,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed_cases"] and not args.allow_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
