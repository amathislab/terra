"""Require exact scientific equality between two reconstruction cohorts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from terra._revision import write_git_commit
from terra.benchmarking.reconstruction.core import load_selection

SCIENTIFIC_FIELDS = ("terrain", "fit", "validation")
MAX_REPORTED_STRUCTURAL_PATHS = 20


def _difference_summary(before: Any, after: Any) -> dict[str, Any]:
    """Describe exact JSON differences without hiding floating-point drift."""

    summary: dict[str, Any] = {
        "numeric_differences": 0,
        "max_abs_numeric_delta": 0.0,
        "max_abs_numeric_delta_path": None,
        "structural_differences": 0,
        "structural_paths": [],
    }

    def structural(path: str) -> None:
        summary["structural_differences"] += 1
        paths = summary["structural_paths"]
        if len(paths) < MAX_REPORTED_STRUCTURAL_PATHS:
            paths.append(path)

    def walk(left: Any, right: Any, path: str) -> None:
        if left == right:
            return
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            delta = abs(float(left) - float(right))
            summary["numeric_differences"] += 1
            if not math.isfinite(delta) or delta > summary["max_abs_numeric_delta"]:
                summary["max_abs_numeric_delta"] = delta
                summary["max_abs_numeric_delta_path"] = path
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = set(left)
            right_keys = set(right)
            for key in sorted(left_keys - right_keys):
                structural(f"{path}.{key}:removed")
            for key in sorted(right_keys - left_keys):
                structural(f"{path}.{key}:added")
            for key in sorted(left_keys & right_keys):
                walk(left[key], right[key], f"{path}.{key}")
            return
        if (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes))
            and isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
        ):
            if len(left) != len(right):
                structural(f"{path}.length:{len(left)}->{len(right)}")
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
                walk(left_item, right_item, f"{path}[{index}]")
            return
        structural(path)

    walk(before, after, "$")
    return summary


def _record_path(root: Path, motion: str) -> Path:
    return root / f"{motion.replace('/', '__')}.json"


def _load_record(path: Path, motion: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"record is missing or unsafe: {path}")
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"record is unreadable: {path}: {error}") from error
    if not isinstance(record, dict) or record.get("motion") != motion:
        raise ValueError(f"record identity does not match {motion!r}: {path}")
    for field in SCIENTIFIC_FIELDS:
        if not isinstance(record.get(field), dict):
            raise ValueError(f"record has no {field!r} object: {path}")
    return record


def compare(manifest: Path, baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Compare every selected motion and return a publication-ready audit."""

    selection = load_selection(manifest)
    baseline_root = baseline_dir.expanduser().resolve()
    candidate_root = candidate_dir.expanduser().resolve()
    changed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    field_totals = {
        field: {
            "changed_records": 0,
            "numeric_differences": 0,
            "max_abs_numeric_delta": 0.0,
            "structural_differences": 0,
        }
        for field in SCIENTIFIC_FIELDS
    }
    for motion in selection.motions:
        try:
            baseline = _load_record(_record_path(baseline_root, motion), motion)
            candidate = _load_record(_record_path(candidate_root, motion), motion)
        except ValueError as error:
            errors.append({"motion": motion, "error": str(error)})
            continue
        fields = [field for field in SCIENTIFIC_FIELDS if baseline[field] != candidate[field]]
        if fields:
            differences = {field: _difference_summary(baseline[field], candidate[field]) for field in fields}
            for field, summary in differences.items():
                totals = field_totals[field]
                totals["changed_records"] += 1
                totals["numeric_differences"] += summary["numeric_differences"]
                totals["max_abs_numeric_delta"] = max(
                    totals["max_abs_numeric_delta"], summary["max_abs_numeric_delta"]
                )
                totals["structural_differences"] += summary["structural_differences"]
            changed.append({"motion": motion, "fields": fields, "differences": differences})
    structurally_changed = sum(
        any(summary["structural_differences"] for summary in record["differences"].values()) for record in changed
    )
    return {
        "selected": len(selection.motions),
        "identical": len(selection.motions) - len(changed) - len(errors),
        "changed": len(changed),
        "errors": len(errors),
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "numeric_only_changed": len(changed) - structurally_changed,
        "structurally_changed": structurally_changed,
        "field_totals": field_totals,
        "changed_records": changed,
        "error_records": errors,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="terra evaluate reconstruction-diff", description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--baseline-dir", type=Path, required=True)
    result.add_argument("--candidate-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--allow-changes",
        action="store_true",
        help="Report scientific differences without failing; missing or invalid records still fail.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = compare(args.manifest, args.baseline_dir, args.candidate_dir)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_git_commit(output)
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] or (report["changed"] and not args.allow_changes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
