"""Build one audited policy selection from retargeted artifacts in isolated caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from terra._methods import RetargetingMethod, validate_method
from terra.artifacts import retarget_cache_paths
from terra.commands.materialize import read_manifest
from terra.paths import StorageRoots
from terra.training_segments import expand_training_segments, publish_segmented_selection

SPEC_SCHEMA_VERSION = 1
_SUCCESS_STATUSES = frozenset(("ok", "cached"))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class ArtifactGroup:
    """One exact source subset backed by one method-specific cache."""

    key: str
    selection: Path
    selection_sha256: str
    cache_root: Path
    status_table: Path | None


@dataclass(frozen=True, slots=True)
class ProvenanceFile:
    """An immutable upstream audit required by the selection."""

    key: str
    path: Path
    sha256: str
    schema: str | None
    require_passed: bool


@dataclass(frozen=True, slots=True)
class RetargetTrainingSelectionSpec:
    """Validated inputs for one retargeting method's policy cohort."""

    path: Path
    name: str
    method: RetargetingMethod
    source_selection: Path
    source_selection_sha256: str
    producer_commit: str
    producer_image: str
    required_frequency_hz: float
    fitted_shape_relpath: Path | None
    fitted_shape_sha256: str | None
    groups: tuple[ArtifactGroup, ...]
    provenance: tuple[ProvenanceFile, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _positive_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _sha256_value(value: object, field: str) -> str:
    result = _text(value, field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _path(base: Path, value: object, field: str, *, require_file: bool = True) -> Path:
    raw = _text(value, field)
    candidate = Path(raw).expanduser()
    path = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if require_file and not path.is_file():
        raise FileNotFoundError(f"{field} is unavailable: {path}")
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    rows = read_manifest(path)
    motions = [row.get("motion", "").strip() for row in rows]
    if not rows or any(not motion for motion in motions) or len(motions) != len(set(motions)):
        raise ValueError(f"selection contains empty or duplicate motion IDs: {path}")
    return rows


def load_retarget_training_selection_spec(path: str | Path) -> RetargetTrainingSelectionSpec:
    """Load a spec and prove that its groups exactly partition the frozen source rows."""

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    with spec_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(f"{spec_path} must declare schema_version = {SPEC_SCHEMA_VERSION}")
    base = spec_path.parent
    name = _text(payload.get("name"), "name")
    method = validate_method(_text(payload.get("method"), "method"))
    if method == "terra":
        raise ValueError("retarget training selection method must be a baseline method")
    source_selection = _path(base, payload.get("source_selection"), "source_selection")
    source_digest = _sha256(source_selection)
    expected_source_digest = _sha256_value(payload.get("source_selection_sha256"), "source_selection_sha256")
    if source_digest != expected_source_digest:
        raise ValueError(f"source selection digest mismatch: expected {expected_source_digest}, got {source_digest}")
    producer_commit = _text(payload.get("producer_commit"), "producer_commit")
    if _GIT_COMMIT.fullmatch(producer_commit) is None:
        raise ValueError("producer_commit must be a lowercase 40-character Git commit")
    producer_image = _text(payload.get("producer_image"), "producer_image")
    required_frequency = _positive_finite(payload.get("required_frequency_hz"), "required_frequency_hz")

    fitted_shape_value = payload.get("fitted_shape_relpath")
    fitted_shape_digest_value = payload.get("fitted_shape_sha256")
    if (fitted_shape_value is None) != (fitted_shape_digest_value is None):
        raise ValueError("fitted_shape_relpath and fitted_shape_sha256 must be specified together")
    fitted_shape_relpath = None
    fitted_shape_digest = None
    if fitted_shape_value is not None:
        fitted_shape_relpath = Path(_text(fitted_shape_value, "fitted_shape_relpath"))
        if fitted_shape_relpath.is_absolute() or ".." in fitted_shape_relpath.parts:
            raise ValueError("fitted_shape_relpath must be a safe relative path")
        fitted_shape_digest = _sha256_value(fitted_shape_digest_value, "fitted_shape_sha256")

    source_rows = _rows(source_selection)
    source_by_motion = {row["motion"]: row for row in source_rows}
    group_values = payload.get("artifact_group")
    if not isinstance(group_values, list) or not group_values:
        raise ValueError("at least one [[artifact_group]] is required")
    groups: list[ArtifactGroup] = []
    group_for_motion: dict[str, str] = {}
    keys: set[str] = set()
    for index, value in enumerate(group_values):
        if not isinstance(value, Mapping):
            raise ValueError(f"artifact_group {index} must be a TOML table")
        key = _text(value.get("key"), f"artifact_group {index}.key")
        if key in keys:
            raise ValueError(f"duplicate artifact group key: {key!r}")
        keys.add(key)
        selection = _path(base, value.get("selection"), f"artifact_group {key}.selection")
        selection_digest = _sha256(selection)
        expected_selection_digest = _sha256_value(
            value.get("selection_sha256"),
            f"artifact_group {key}.selection_sha256",
        )
        if selection_digest != expected_selection_digest:
            raise ValueError(
                f"artifact group {key} selection digest mismatch: "
                f"expected {expected_selection_digest}, got {selection_digest}"
            )
        cache_root = _path(
            base,
            value.get("cache_root"),
            f"artifact_group {key}.cache_root",
            require_file=False,
        )
        if not cache_root.is_absolute():
            raise ValueError(f"artifact_group {key}.cache_root must resolve to an absolute path")
        raw_status = value.get("status_table")
        status_table = (
            None
            if raw_status is None
            else _path(base, raw_status, f"artifact_group {key}.status_table", require_file=False)
        )
        group_rows = _rows(selection)
        for row in group_rows:
            motion = row["motion"]
            source_row = source_by_motion.get(motion)
            if source_row is None:
                raise ValueError(f"artifact group {key} contains motion absent from source: {motion!r}")
            if row != source_row:
                raise ValueError(f"artifact group {key} row differs from frozen source row: {motion!r}")
            previous = group_for_motion.setdefault(motion, key)
            if previous != key:
                raise ValueError(f"motion occurs in artifact groups {previous!r} and {key!r}: {motion!r}")
        groups.append(
            ArtifactGroup(
                key=key,
                selection=selection,
                selection_sha256=selection_digest,
                cache_root=cache_root,
                status_table=status_table,
            )
        )
    missing = [row["motion"] for row in source_rows if row["motion"] not in group_for_motion]
    if missing:
        raise ValueError(f"artifact groups do not cover {len(missing)} source motions; first={missing[0]!r}")

    provenance_values = payload.get("provenance", [])
    if not isinstance(provenance_values, list):
        raise ValueError("[[provenance]] entries must be TOML tables")
    provenance: list[ProvenanceFile] = []
    provenance_keys: set[str] = set()
    for index, value in enumerate(provenance_values):
        if not isinstance(value, Mapping):
            raise ValueError(f"provenance {index} must be a TOML table")
        key = _text(value.get("key"), f"provenance {index}.key")
        if key in provenance_keys:
            raise ValueError(f"duplicate provenance key: {key!r}")
        provenance_keys.add(key)
        raw_schema = value.get("schema")
        require_passed = value.get("require_passed", False)
        if not isinstance(require_passed, bool):
            raise ValueError(f"provenance {key}.require_passed must be a boolean")
        provenance.append(
            ProvenanceFile(
                key=key,
                path=_path(base, value.get("path"), f"provenance {key}.path", require_file=False),
                sha256=_sha256_value(value.get("sha256"), f"provenance {key}.sha256"),
                schema=None if raw_schema is None else _text(raw_schema, f"provenance {key}.schema"),
                require_passed=require_passed,
            )
        )
    return RetargetTrainingSelectionSpec(
        path=spec_path,
        name=name,
        method=method,
        source_selection=source_selection,
        source_selection_sha256=source_digest,
        producer_commit=producer_commit,
        producer_image=producer_image,
        required_frequency_hz=required_frequency,
        fitted_shape_relpath=fitted_shape_relpath,
        fitted_shape_sha256=fitted_shape_digest,
        groups=tuple(groups),
        provenance=tuple(provenance),
    )


def _verify_status(group: ArtifactGroup, required_frequency_hz: float) -> dict[str, object] | None:
    if group.status_table is None:
        return None
    if not group.status_table.is_file():
        raise FileNotFoundError(f"artifact group {group.key} status table is unavailable: {group.status_table}")
    selected = {row["motion"] for row in _rows(group.selection)}
    with group.status_table.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    motions = [str(row.get("motion", "")).strip() for row in rows]
    if len(rows) != len(selected) or set(motions) != selected or len(motions) != len(set(motions)):
        raise ValueError(f"artifact group {group.key} status table does not exactly cover its selection")
    unsuccessful = [row for row in rows if str(row.get("status", "")).strip() not in _SUCCESS_STATUSES]
    if unsuccessful:
        raise ValueError(
            f"artifact group {group.key} has {len(unsuccessful)} unsuccessful motions; "
            f"first={unsuccessful[0].get('motion', '')!r}"
        )
    for row in rows:
        try:
            frequency = float(str(row.get("frequency", "")).strip())
        except ValueError as error:
            raise ValueError(f"artifact group {group.key} status row has invalid frequency") from error
        if not math.isclose(frequency, required_frequency_hz, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"artifact group {group.key} status row {row.get('motion', '')!r} has frequency {frequency:.12g} Hz"
            )
    return {
        "path": str(group.status_table),
        "sha256": _sha256(group.status_table),
        "rows": len(rows),
        "all_successful": True,
        "frequency_hz": required_frequency_hz,
    }


def _verify_provenance(provenance: ProvenanceFile) -> dict[str, object]:
    if not provenance.path.is_file():
        raise FileNotFoundError(f"required provenance is unavailable: {provenance.path}")
    digest = _sha256(provenance.path)
    if digest != provenance.sha256:
        raise ValueError(f"provenance {provenance.key} digest mismatch: expected {provenance.sha256}, got {digest}")
    if provenance.schema is not None or provenance.require_passed:
        try:
            payload = json.loads(provenance.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"provenance {provenance.key} is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"provenance {provenance.key} must contain a JSON object")
        if provenance.schema is not None and payload.get("schema") != provenance.schema:
            raise ValueError(f"provenance {provenance.key} has an unexpected schema")
        if provenance.require_passed and payload.get("passed") is not True:
            raise ValueError(f"provenance {provenance.key} is not a passing audit")
    return {
        "key": provenance.key,
        "path": str(provenance.path),
        "sha256": digest,
        "schema": provenance.schema,
        "passed": True if provenance.require_passed else None,
    }


def build_retarget_training_selection(
    spec: RetargetTrainingSelectionSpec,
    *,
    storage_roots: StorageRoots | None = None,
    base: Path | None = None,
    trigger_seconds: float = 20.0,
    maximum_segment_seconds: float = 10.0,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Validate all artifacts and derive temporal views from their actual frame grids."""

    root_base = Path.cwd() if base is None else base
    roots = storage_roots or StorageRoots.from_environment(root_base)
    source_rows = _rows(spec.source_selection)
    group_by_motion: dict[str, ArtifactGroup] = {}
    group_audits: list[dict[str, object]] = []
    checked_caches: set[Path] = set()
    for group in spec.groups:
        group_rows = _rows(group.selection)
        group_by_motion.update({row["motion"]: group for row in group_rows})
        if group.cache_root not in checked_caches:
            commit_marker = group.cache_root / "GIT_COMMIT"
            if not commit_marker.is_file() or commit_marker.read_text(encoding="utf-8").strip() != spec.producer_commit:
                raise ValueError(f"artifact cache {group.cache_root} was not produced by commit {spec.producer_commit}")
            if spec.fitted_shape_relpath is not None:
                fitted_shape = group.cache_root / spec.fitted_shape_relpath
                if not fitted_shape.is_file() or _sha256(fitted_shape) != spec.fitted_shape_sha256:
                    raise ValueError(f"artifact cache {group.cache_root} does not contain the pinned fitted shape")
            checked_caches.add(group.cache_root)
        group_audits.append(
            {
                "key": group.key,
                "selection": str(group.selection),
                "selection_sha256": group.selection_sha256,
                "motions": len(group_rows),
                "cache_root": str(group.cache_root),
                "status": _verify_status(group, spec.required_frequency_hz),
            }
        )

    rebound: list[dict[str, str]] = []
    for source_row in source_rows:
        motion = source_row["motion"]
        group = group_by_motion[motion]
        paths = retarget_cache_paths(group.cache_root, motion, method=spec.method)
        rebound.append(
            {str(key): str(value) for key, value in source_row.items()}
            | {
                "source_cache_root": str(group.cache_root),
                "trajectory_relpath": str(paths.trajectory_path.relative_to(group.cache_root)),
                "analysis_relpath": str(paths.analysis_path.relative_to(group.cache_root)),
                "terrain_relpath": (
                    str(paths.terrain_path.relative_to(group.cache_root))
                    if source_row.get("terrain_relpath", "").strip()
                    else ""
                ),
                "retargeting_method": spec.method,
            }
        )
    segmented, segment_audit = expand_training_segments(
        rebound,
        storage_roots=roots,
        base=root_base,
        method=spec.method,
        trigger_seconds=trigger_seconds,
        maximum_segment_seconds=maximum_segment_seconds,
        required_frequency_hz=spec.required_frequency_hz,
    )
    audit = segment_audit | {
        "schema": "terra.retarget-training-selection.v1",
        "name": spec.name,
        "spec": str(spec.path),
        "spec_sha256": _sha256(spec.path),
        "source_selection": str(spec.source_selection),
        "source_selection_sha256": spec.source_selection_sha256,
        "producer_commit": spec.producer_commit,
        "producer_image": spec.producer_image,
        "fitted_shape_relpath": None if spec.fitted_shape_relpath is None else str(spec.fitted_shape_relpath),
        "fitted_shape_sha256": spec.fitted_shape_sha256,
        "artifact_groups": group_audits,
        "provenance": [_verify_provenance(item) for item in spec.provenance],
        "all_source_artifacts_validated": True,
    }
    return segmented, audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra train retarget-selection", description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--trigger-seconds", type=float, default=20.0)
    parser.add_argument("--maximum-segment-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        roots = StorageRoots.from_environment(Path.cwd())
        spec_path = roots.resolve_input(args.spec, base=Path.cwd())
        destination = roots.resolve_artifact(args.out, base=Path.cwd())
        audit_path = roots.resolve_artifact(args.audit_out, base=Path.cwd())
        assert spec_path is not None and destination is not None and audit_path is not None
        spec = load_retarget_training_selection_spec(spec_path)
        rows, audit = build_retarget_training_selection(
            spec,
            storage_roots=roots,
            base=Path.cwd(),
            trigger_seconds=args.trigger_seconds,
            maximum_segment_seconds=args.maximum_segment_seconds,
        )
        payload = publish_segmented_selection(
            spec.source_selection,
            destination,
            audit_path,
            rows,
            audit,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactGroup",
    "ProvenanceFile",
    "RetargetTrainingSelectionSpec",
    "build_retarget_training_selection",
    "load_retarget_training_selection_spec",
    "main",
]
