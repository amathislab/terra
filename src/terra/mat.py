"""Schema-driven MATLAB marker extraction for the public retargeting API."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

MatSchemaInput: TypeAlias = Mapping[str, object] | str | Path

_MAT_MARKER_ARCHIVE_VERSION = 1
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class MatMarkerData:
    """One marker trajectory extracted from a MATLAB container."""

    positions: np.ndarray
    labels: tuple[str, ...]
    fps: float


def load_mat_schema(schema: MatSchemaInput) -> dict[str, object]:
    """Load and minimally validate a MAT marker schema."""

    if isinstance(schema, Mapping):
        loaded = dict(schema)
    else:
        path = Path(schema).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MAT schema not found: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"MAT schema is not valid JSON: {path}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"MAT schema must contain a JSON object: {path}")

    version = loaded.get("version", 1)
    if version != 1:
        raise ValueError(f"unsupported MAT schema version {version!r}; expected 1")
    has_tensor = "positions_path" in loaded
    has_fields = "marker_fields" in loaded
    if has_tensor == has_fields:
        raise ValueError("MAT schema must define exactly one of positions_path or marker_fields")
    if "fps" not in loaded and "fps_path" not in loaded:
        raise ValueError("MAT schema must define fps or fps_path")
    return loaded


def _unwrap_singleton(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def _field(value: Any, name: str) -> Any:
    current = _unwrap_singleton(value)
    if isinstance(current, Mapping):
        if name not in current:
            raise KeyError(name)
        return current[name]
    if isinstance(current, np.void) and current.dtype.names and name in current.dtype.names:
        return current[name]
    if hasattr(current, name):
        return getattr(current, name)
    names = getattr(current, "_fieldnames", ()) or getattr(getattr(current, "dtype", None), "names", ()) or ()
    raise KeyError(f"MAT value has no field {name!r}; available fields: {tuple(names)}")


def _path_components(path: object, label: str) -> list[object]:
    if isinstance(path, str):
        components: list[object] = [part for part in path.split(".") if part]
    elif isinstance(path, Sequence) and not isinstance(path, (bytes, bytearray)):
        components = list(path)
    else:
        raise ValueError(f"MAT schema {label} must be a dotted string or path array")
    if not components:
        raise ValueError(f"MAT schema {label} cannot be empty")
    return components


def _resolve_path(
    root: Any,
    path: object,
    selectors: Mapping[str, int],
    *,
    label: str,
) -> Any:
    current = root
    for component in _path_components(path, label):
        if isinstance(component, str):
            current = _field(current, component)
            continue
        if isinstance(component, Integral) and not isinstance(component, bool):
            index = int(component)
        elif isinstance(component, Mapping) and set(component) == {"selector"}:
            selector = component["selector"]
            if not isinstance(selector, str) or not selector:
                raise ValueError(f"MAT schema {label} contains an invalid selector")
            if selector not in selectors:
                raise ValueError(f"MAT selector {selector!r} is required by {label}")
            index = selectors[selector]
        else:
            raise ValueError(f"MAT schema {label} contains an invalid path component: {component!r}")
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError(f"MAT index in {label} must be an integer")
        index = int(index)
        values = np.asarray(current, dtype=object).reshape(-1)
        try:
            current = values[index]
        except IndexError as exc:
            raise IndexError(f"MAT index {index} is outside {label} with {len(values)} values") from exc
    return _unwrap_singleton(current)


def _labels(value: Any) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"} and array.ndim == 2 and array.shape[1] > 1:
        values = ["".join(str(part) for part in row).strip() for row in array]
    else:
        values = [str(_unwrap_singleton(item)).strip() for item in np.asarray(value, dtype=object).reshape(-1)]
    labels = tuple(values)
    if not labels or any(not label for label in labels):
        raise ValueError("MAT marker labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("MAT marker labels must be unique")
    return labels


def _scale_for_units(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("MAT schema units must be a string")
    unit = value.strip().casefold()
    divisors = {
        "mm": 1000.0,
        "millimeter": 1000.0,
        "millimeters": 1000.0,
        "cm": 100.0,
        "centimeter": 100.0,
        "centimeters": 100.0,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
    }
    if unit not in divisors:
        raise ValueError(f"unsupported MAT marker unit {unit!r}")
    return divisors[unit]


def _axis_order(value: object, *, dimensions: str, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError(f"MAT schema {label} must be a string")
    normalized = value.casefold()
    if len(normalized) != len(dimensions) or set(normalized) != set(dimensions):
        raise ValueError(f"MAT schema {label} must be a permutation of {dimensions!r}")
    return tuple(normalized.index(axis) for axis in dimensions)


def _coordinate_transform(value: object) -> tuple[tuple[int, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        raise ValueError("MAT schema axes must contain three signed axes")
    output: list[tuple[int, float]] = []
    used: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError("MAT schema axes must contain strings")
        normalized = raw.strip().casefold()
        sign = -1.0 if normalized.startswith("-") else 1.0
        axis = normalized.lstrip("+-")
        if axis not in _AXIS_INDEX or axis in used:
            raise ValueError("MAT schema axes must be a signed permutation of x, y, and z")
        used.add(axis)
        output.append((_AXIS_INDEX[axis], sign))
    return tuple(output)


def _numeric_scalar(value: Any, label: str) -> float:
    try:
        scalar = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MAT {label} must resolve to one numeric value") from exc
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"MAT {label} must be positive and finite")
    return scalar


def extract_mat_markers(
    contents: Mapping[str, object],
    schema: MatSchemaInput,
    *,
    selectors: Mapping[str, int] | None = None,
) -> MatMarkerData:
    """Extract one marker trajectory from already loaded MATLAB contents."""

    resolved = load_mat_schema(schema)
    selected = dict(selectors or {})
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in selected.values()):
        raise ValueError("MAT selector values must be integers")
    selected = {name: int(value) for name, value in selected.items()}

    record = contents
    if "record_path" in resolved:
        record = _resolve_path(contents, resolved["record_path"], selected, label="record_path")

    if "marker_fields" in resolved:
        marker_fields = resolved["marker_fields"]
        if not isinstance(marker_fields, Mapping) or not marker_fields:
            raise ValueError("MAT schema marker_fields must be a non-empty object")
        labels = _labels(list(marker_fields))
        field_order = _axis_order(
            resolved.get("marker_field_axis_order", "tc"), dimensions="tc", label="marker_field_axis_order"
        )
        columns = []
        for marker_label, marker_path in marker_fields.items():
            raw = _resolve_path(record, marker_path, selected, label=f"marker_fields.{marker_label}")
            column = np.asarray(raw, dtype=np.float32)
            if column.ndim != 2:
                raise ValueError(f"MAT marker field {marker_label!r} must be two-dimensional")
            column = np.transpose(column, field_order)
            if column.shape[1] != 3:
                raise ValueError(f"MAT marker field {marker_label!r} must have three coordinate columns")
            columns.append(column)
        lengths = {len(column) for column in columns}
        if len(lengths) != 1:
            raise ValueError(f"MAT marker field lengths disagree: {sorted(lengths)}")
        positions = np.stack(columns, axis=1)
    else:
        raw = _resolve_path(record, resolved["positions_path"], selected, label="positions_path")
        positions = np.asarray(raw, dtype=np.float32)
        if positions.ndim != 3:
            raise ValueError("MAT marker positions must be a three-dimensional array")
        order = _axis_order(resolved.get("axis_order", "tmc"), dimensions="tmc", label="axis_order")
        positions = np.transpose(positions, order)
        if positions.shape[2] != 3:
            raise ValueError("MAT marker positions must have three coordinates")
        if "labels" in resolved:
            labels = _labels(resolved["labels"])
        elif "labels_path" in resolved:
            labels = _labels(_resolve_path(record, resolved["labels_path"], selected, label="labels_path"))
        else:
            raise ValueError("tensor MAT schemas must define labels or labels_path")

    if positions.shape[1] != len(labels):
        raise ValueError(
            f"MAT positions contain {positions.shape[1]} markers but the schema provides {len(labels)} labels"
        )
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("MAT marker positions must contain at least one frame and marker")
    if np.isinf(positions).any():
        raise ValueError("MAT marker positions cannot contain infinite values")

    fps = (
        _numeric_scalar(resolved["fps"], "fps")
        if "fps" in resolved
        else _numeric_scalar(_resolve_path(contents, resolved["fps_path"], selected, label="fps_path"), "fps")
    )
    positions = positions / _scale_for_units(resolved.get("units", "m"))
    transform = _coordinate_transform(resolved.get("axes", ("x", "y", "z")))
    source_positions = positions
    positions = np.stack([sign * source_positions[..., index] for index, sign in transform], axis=-1)
    zero_is_missing = resolved.get("zero_is_missing", True)
    if not isinstance(zero_is_missing, bool):
        raise ValueError("MAT schema zero_is_missing must be a boolean")
    if zero_is_missing:
        positions[np.all(np.isclose(positions, 0.0), axis=-1)] = np.nan
    return MatMarkerData(positions=positions.astype(np.float32, copy=False), labels=labels, fps=fps)


def load_mat_markers(
    path: str | Path,
    schema: MatSchemaInput,
    *,
    selectors: Mapping[str, int] | None = None,
) -> MatMarkerData:
    """Extract one marker trajectory using an explicit MAT schema.

    ``schema`` may be a JSON file path or mapping. It must name exactly one
    position tensor or set of marker fields and supply a frame rate; it can
    also describe labels, units, and coordinate axes. ``selectors`` choose
    zero-based entries in a nested record. Returned positions are in metres
    and Z-up, with missing markers represented by NaNs. SciPy-readable MAT
    files are supported; MATLAB v7.3/HDF5 files must be exported as v7.2.

    Raises:
        FileNotFoundError: If the MAT file or schema path is absent.
        ValueError: If the file, schema, or selected markers are invalid.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MAT file not found: {source}")
    if source.suffix.casefold() != ".mat":
        raise ValueError(f"expected a .mat marker container, got {source.name!r}")
    resolved_schema = load_mat_schema(schema)
    from scipy.io import loadmat

    try:
        contents = loadmat(source, struct_as_record=False, squeeze_me=True)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3/HDF5 files are not supported; export a v7.2 MAT file") from exc
    return extract_mat_markers(contents, resolved_schema, selectors=selectors)


def prepare_mat_marker_archive(
    source_path: str | Path,
    schema: MatSchemaInput,
    cache_root: str | Path,
    *,
    selectors: Mapping[str, int] | None = None,
) -> Path:
    """Convert a schema-selected MAT trajectory to MuscleMimic's marker archive."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MAT file not found: {source}")
    if source.suffix.casefold() != ".mat":
        raise ValueError(f"expected a .mat marker container, got {source.name!r}")
    resolved_schema = load_mat_schema(schema)
    selected = dict(selectors or {})
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in selected.values()):
        raise ValueError("MAT selector values must be integers")
    selected = {name: int(value) for name, value in selected.items()}
    try:
        identity = json.dumps(
            {"schema": resolved_schema, "selectors": selected},
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("MAT schema and selectors must contain JSON-compatible values") from exc
    digest = hashlib.sha256()
    digest.update(f"terra-mat-marker-archive-v{_MAT_MARKER_ARCHIVE_VERSION}\0{identity}\0".encode())
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    archive_root = Path(cache_root).expanduser().resolve() / ".mat_marker_cache"
    archive_path = archive_root / f"{digest.hexdigest()}.npz"
    if archive_path.is_file():
        return archive_path

    motion = load_mat_markers(source, resolved_schema, selectors=selected)
    archive_root.mkdir(parents=True, exist_ok=True)
    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".mat-", suffix=".npz", dir=archive_root, delete=False) as stream:
            staged_path = Path(stream.name)
            np.savez_compressed(
                stream,
                positions=motion.positions,
                labels=np.asarray(motion.labels),
                fps=np.asarray(motion.fps, dtype=np.float32),
                mat_schema=np.asarray(identity),
                archive_version=np.asarray(_MAT_MARKER_ARCHIVE_VERSION, dtype=np.int64),
            )
        staged_path.replace(archive_path)
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
    return archive_path


__all__ = [
    "MatMarkerData",
    "MatSchemaInput",
    "extract_mat_markers",
    "load_mat_markers",
    "load_mat_schema",
    "prepare_mat_marker_archive",
]
