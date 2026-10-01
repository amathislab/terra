"""Common terrain-reconstruction execution and publication utilities.

Reconstruction methods own only scientific preparation and fitting. This module owns
the cohort denominator, cache validation, atomic record publication, and status
checkpoints. Records from every registered method therefore use the same layout and
cache behavior.
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from terra._files import atomic_write
from terra._revision import write_git_commit
from terra.artifacts import normalize_motion_name

from .provenance import RECORD_PROVENANCE_SCHEMA, build_run_provenance

STATUS_FIELDS = (
    "motion",
    "method",
    "status",
    "output",
    "elapsed_seconds",
    "error",
    "summary_json",
)
MAX_ERROR_CHARS = 2000


def _json_default(value: object) -> object:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a runtime dependency
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_json(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n"
    atomic_write(path, lambda temporary: temporary.write_text(rendered))


def _motion_rows(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    """Read an ordered TXT or CSV selection without importing a workflow script."""

    selection = path.expanduser().resolve()
    text = selection.read_text()
    if selection.suffix.casefold() == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        fields = tuple(reader.fieldnames or ())
        if "motion" not in fields:
            raise ValueError(f"selection CSV has no motion column: {selection}")
        rows = tuple(dict(row) for row in reader)
    else:
        fields = ("motion",)
        rows = tuple(
            {"motion": line.strip()} for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
        )
    if not rows:
        raise ValueError("motion selection must not be empty")
    return fields, rows


@dataclass(frozen=True)
class Selection:
    path: Path
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    @property
    def motions(self) -> tuple[str, ...]:
        return tuple(row["motion"] for row in self.rows)


def load_selection(path: Path) -> Selection:
    fields, raw_rows = _motion_rows(path)
    rows: list[dict[str, str]] = []
    owners: dict[str, str] = {}
    filenames: dict[str, str] = {}
    for raw in raw_rows:
        value = raw.get("motion")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("motion selection contains an empty identifier")
        canonical = normalize_motion_name(value).as_posix()
        if canonical != value:
            raise ValueError(f"motion IDs must already be canonical: {value!r} != {canonical!r}")
        if canonical in owners:
            raise ValueError(f"motion selection contains duplicate ID {canonical!r}")
        filename = f"{canonical.replace('/', '__')}.json"
        collision = filename.casefold()
        if collision in filenames:
            raise ValueError(
                f"motion IDs flatten to the same output filename: {filenames[collision]!r} and {canonical!r}"
            )
        owners[canonical] = value
        filenames[collision] = canonical
        rows.append(dict(raw) | {"motion": canonical})
    return Selection(
        path=path.expanduser().resolve(),
        fields=fields,
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class PreparedMotion:
    """Opaque method state plus JSON-compatible inputs for the fit report."""

    state: object
    input: Mapping[str, Any]


@dataclass(frozen=True)
class ReconstructionResult:
    terrain: Mapping[str, Any]
    fit: Mapping[str, Any]
    validation: Mapping[str, Any]


@runtime_checkable
class ReconstructionMethod(Protocol):
    """Scientific plugin consumed by :func:`run_cohort`."""

    name: str
    display_name: str
    description: str

    @property
    def options(self) -> Mapping[str, Any]: ...

    def prepare(self, motion: str) -> PreparedMotion: ...

    def fit(
        self,
        motion: str,
        prepared: PreparedMotion,
    ) -> ReconstructionResult: ...

    def summarize(self, result: ReconstructionResult) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CohortResult:
    method: str
    status_path: Path
    records: tuple[Mapping[str, Any], ...]

    @property
    def failed(self) -> int:
        return sum(record["status"] == "failed" for record in self.records)

    @property
    def exit_code(self) -> int:
        return 2 if self.failed else 0


def _record(
    method: ReconstructionMethod,
    motion: str,
    result: ReconstructionResult,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method": method.name,
        "method_display_name": method.display_name,
        "motion": motion,
        "provenance": {
            "schema": RECORD_PROVENANCE_SCHEMA,
            "method_identity_sha256": provenance["method_identity_sha256"],
            "scientific_identity_sha256": provenance["scientific_identity_sha256"],
        },
        "terrain": dict(result.terrain),
        "fit": dict(result.fit),
        "validation": dict(result.validation),
    }


def cached_record(
    path: Path,
    method: ReconstructionMethod,
    motion: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    if path.is_symlink():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("method") != method.name:
        return None
    if record.get("motion") != motion:
        return None
    expected_provenance = {
        "schema": RECORD_PROVENANCE_SCHEMA,
        "method_identity_sha256": provenance["method_identity_sha256"],
        "scientific_identity_sha256": provenance["scientific_identity_sha256"],
    }
    if record.get("provenance") != expected_provenance:
        return None
    scientific_result = {key: record.get(key) for key in ("terrain", "fit", "validation")}
    if not all(isinstance(value, dict) for value in scientific_result.values()):
        return None
    return record


def _status_row(
    *,
    motion: str,
    method: ReconstructionMethod,
    status: str,
    output: Path,
    elapsed: float,
    summary: Mapping[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, str]:
    error_text = "" if error is None else f"{type(error).__name__}: {error}".replace("\n", " ")
    if len(error_text) > MAX_ERROR_CHARS:
        error_text = error_text[: MAX_ERROR_CHARS - 3] + "..."
    return {
        "motion": motion,
        "method": method.name,
        "status": status,
        "output": str(output),
        "elapsed_seconds": f"{elapsed:.6f}",
        "error": error_text,
        "summary_json": "" if summary is None else json.dumps(summary, sort_keys=True, allow_nan=False),
    }


def _write_status(path: Path, motions: Sequence[str], rows: Mapping[str, Mapping[str, str]]) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[motion] for motion in motions if motion in rows)

    atomic_write(path, write)


def run_cohort(
    method: ReconstructionMethod,
    selection_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    repo_root: Path | None = None,
    dataset_config_path: Path | None = None,
    matrix_path: Path | None = None,
    progress: bool = True,
) -> CohortResult:
    """Run one registered method over one immutable ordered denominator."""

    selection = load_selection(selection_path)
    output_root = output_dir.expanduser().resolve()
    outputs = {motion: output_root / f"{motion.replace('/', '__')}.json" for motion in selection.motions}
    provenance = build_run_provenance(
        method.name,
        method.options,
        selection.path,
        dataset_config_path=dataset_config_path,
        matrix_path=matrix_path,
        repo_root=repo_root,
    )
    reusable: dict[str, dict[str, Any]] = {}
    if not overwrite:
        for motion, output in outputs.items():
            if not (output.exists() or output.is_symlink()):
                continue
            existing = cached_record(output, method, motion, provenance)
            if existing is None:
                raise FileExistsError(
                    "existing output does not match the current scientific identity: "
                    f"{output}; use a fresh output root or pass --overwrite"
                )
            reusable[motion] = existing
    write_git_commit(output_root, repo_root=repo_root)
    status_path = output_root / "status.csv"
    rows: dict[str, dict[str, str]] = {}
    for index, motion in enumerate(selection.motions, 1):
        started = time.perf_counter()
        output = outputs[motion]
        try:
            existing = reusable.get(motion)
            if existing is None:
                prepared = method.prepare(motion)
                result = method.fit(motion, prepared)
                record = _record(method, motion, result, provenance)
                atomic_json(output, record)
                state = "ok"
                summary = method.summarize(result)
            else:
                state = "cached"
                result = ReconstructionResult(
                    terrain=existing["terrain"],
                    fit=existing["fit"],
                    validation=existing["validation"],
                )
                summary = method.summarize(result)
            row = _status_row(
                motion=motion,
                method=method,
                status=state,
                output=output,
                elapsed=time.perf_counter() - started,
                summary=summary,
            )
        except Exception as error:  # keep one bad motion from discarding the denominator
            row = _status_row(
                motion=motion,
                method=method,
                status="failed",
                output=output,
                elapsed=time.perf_counter() - started,
                error=error,
            )
        rows[motion] = row
        _write_status(status_path, selection.motions, rows)
        if progress:
            print(f"[{index}/{len(selection.motions)}] {row['status']} {motion}", flush=True)

    run_payload = {
        "method": method.name,
        "selection": str(selection.path),
        "motions": len(selection.motions),
        "status": str(status_path),
        "counts": {state: sum(row["status"] == state for row in rows.values()) for state in ("ok", "cached", "failed")},
        "options": provenance["method_identity"]["resolved_options"],
        "provenance": provenance,
    }
    atomic_json(output_root / "run.json", run_payload)
    return CohortResult(method.name, status_path, tuple(rows[motion] for motion in selection.motions))
