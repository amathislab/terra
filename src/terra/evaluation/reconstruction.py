"""Score reconstructed terrain against a dataset's available reference geometry.

Family and ramp-angle metrics are included only when the reconstruction representation
provides them; step-height MAE is included where references exist. The PRISM mesh
comparison is a separate ``terra evaluate prism-mesh`` command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra._revision import write_git_commit
from terra.benchmarking.reconstruction.core import load_selection
from terra.benchmarking.reconstruction.provenance import (
    RECORD_PROVENANCE_SCHEMA,
    ValidatedRunProvenance,
    file_sha256,
    validate_run_provenance,
    write_evaluation_provenance,
)
from terra.terrain.metadata import TerrainMetadata
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

DARMSTADT_RISERS_M = (0.10, 0.17, 0.24, 0.10, 0.17, 0.24)
# Published Gait120 apparatus specification (Boo et al., Scientific Data, 2025).
GAIT120_STOOL_HEIGHT_M = 0.490
CURRENT_FIT_STATUSES = {"ok", "cached"}

_MIN_RAISED_LEVEL_M = 0.03


@dataclass(frozen=True)
class _SurfacePatch:
    """One valid, non-seat top surface read from the TerrainSpec wire format."""

    height: float
    area: float
    pitch_deg: float


@dataclass(frozen=True)
class _GeometryMetrics:
    """Conservative metrics available from serialized geometry alone."""

    family: str | None = None
    family_source: str = "unavailable"
    slope_deg: float | None = None
    slope_source: str = "unavailable"


@dataclass(frozen=True)
class _ExpectedGeometry:
    """Dataset apparatus dimensions available independently of reconstruction."""

    slope_deg: float | None = None
    riser_m: float | None = None
    seat_height_m: float | None = None
    source: str = "unavailable"


@dataclass(frozen=True)
class _SeatHeightPrediction:
    """Failure-aware scalar chair prediction read only from serialized geometry."""

    height_m: float | None = None
    raised_support_present: bool | None = None
    source: str = "unavailable"


@dataclass(frozen=True)
class _ContactPoint:
    """One method-independent kinematic contact used by the step scorer."""

    joint: str
    start: int
    end: int
    xyz: np.ndarray


@dataclass(frozen=True)
class _StepContactScore:
    """Predicted-vs-nominal surface-height errors at raised contacts."""

    reference_contacts: int
    raised_contacts: int
    errors_m: np.ndarray
    bias_m: np.ndarray


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _condition(row: dict[str, str]) -> str:
    if row.get("condition"):
        return row["condition"]
    motion = row.get("motion", "")
    parts = motion.split("/")
    dataset = row.get("dataset", "").lower()
    if dataset == "vielemeyer" and len(parts) > 2:
        return parts[2]
    if dataset == "gait120" and len(parts) > 2:
        return parts[2]
    if dataset == "darmstadt" and len(parts) > 2:
        return parts[2]
    return row.get("terrain_class", "")


def _expected_geometry(row: dict[str, str]) -> _ExpectedGeometry:
    slope = _number(row.get("expected_slope_deg"))
    riser = _number(row.get("expected_riser_m"))
    seat_height = _number(row.get("expected_seat_height_m"))
    if slope is not None or riser is not None or seat_height is not None:
        return _ExpectedGeometry(slope, riser, seat_height, "manifest")

    dataset = row.get("dataset", "").lower()
    condition = _condition(row).lower()
    motion = row.get("motion", "").lower()
    if dataset == "vielemeyer":
        match = re.search(r"(?:^|/)ramp_(75|10)_(?:up|down)(?:/|$)", motion)
        if match:
            return _ExpectedGeometry(
                slope_deg=7.5 if match.group(1) == "75" else 10.0,
                source="vielemeyer_nominal_ramp",
            )
    if dataset == "gait120":
        if condition.startswith("slope"):
            return _ExpectedGeometry(slope_deg=10.0, source="gait120_benchmark_nominal")
        if condition.startswith("stair"):
            return _ExpectedGeometry(riser_m=0.100, source="gait120_benchmark_nominal")
        if row.get("terrain_class", "").strip().lower().startswith("chair") or condition in {
            "sittostand",
            "standtosit",
        }:
            return _ExpectedGeometry(
                seat_height_m=GAIT120_STOOL_HEIGHT_M,
                source="gait120_published_stool",
            )
    if dataset == "darmstadt":
        match = re.search(r"(?:^|/)config(0[1-6])(?:/|$)", motion)
        if match:
            return _ExpectedGeometry(
                riser_m=DARMSTADT_RISERS_M[int(match.group(1)) - 1],
                source="darmstadt_nominal_riser",
            )
    return _ExpectedGeometry()


def _expected_family(row: dict[str, str]) -> str | None:
    expected = row.get("expected_family", "").strip().lower()
    if expected:
        return expected
    terrain_class = row.get("terrain_class", "").strip().lower()
    if terrain_class.startswith("ramp"):
        return "ramp"
    if terrain_class.startswith(("stair", "platform", "chair", "beam", "uneven", "low_step")):
        return "steps"
    return None


def _vector(value: object, length: int) -> np.ndarray | None:
    """Read one finite numeric vector without trusting an external terrain file."""
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        return None
    return result


def _surface_patches(
    terrain: dict[str, Any] | None,
    *,
    include_seats: bool = False,
) -> tuple[_SurfacePatch, ...] | None:
    """Parse box top surfaces, returning ``None`` when geometry is malformed.

    The fallback must not silently infer a metric from the well-formed subset of a broken
    scene. Seat boxes are excluded by default, matching ``TerrainSpec.walkable``; the
    stool-height scorer includes them explicitly.
    """

    if not isinstance(terrain, dict) or "boxes" not in terrain:
        return None
    boxes = terrain["boxes"]
    if not isinstance(boxes, list):
        return None
    patches: list[_SurfacePatch] = []
    for box in boxes:
        if not isinstance(box, dict):
            return None
        if not include_seats and str(box.get("name", "")).startswith("terrain_box_seat"):
            continue
        pos = _vector(box.get("pos"), 3)
        size = _vector(box.get("size"), 3)
        pitch = _number(box.get("pitch", 0.0))
        if pos is None or size is None or pitch is None or np.any(size <= 0.0):
            return None
        if abs(pitch) >= math.pi / 2.0:
            return None
        # For horizontal boxes this is the exact top height. A pitched surface's grade
        # comes directly from pitch.
        height = float(pos[2] + size[2])
        patches.append(
            _SurfacePatch(
                height=height,
                area=float(4.0 * size[0] * size[1]),
                pitch_deg=abs(math.degrees(pitch)),
            )
        )
    return tuple(patches)


def _predicted_seat_height(terrain: dict[str, Any] | None) -> _SeatHeightPrediction:
    """Read a scalar chair prediction from geometry without using method-specific labels.

    Gait120 chair trials contain one floor and one horizontal stool top. The highest
    horizontal raised top is therefore the reconstruction's stool-height prediction.
    An empty scene, a floor-only scene, or a purely pitched scene predicts that no raised
    chair support exists and receives a numeric 0 m prediction. Malformed geometry remains
    unavailable rather than being partially interpreted.
    """

    patches = _surface_patches(terrain, include_seats=True)
    if patches is None:
        return _SeatHeightPrediction()
    raised = [
        patch.height
        for patch in patches
        if patch.pitch_deg < 0.5 and patch.height > _MIN_RAISED_LEVEL_M
    ]
    if not raised:
        return _SeatHeightPrediction(
            height_m=0.0,
            raised_support_present=False,
            source="terrain_geometry.no_raised_horizontal_support",
        )
    return _SeatHeightPrediction(
        height_m=max(raised),
        raised_support_present=True,
        source="terrain_geometry.highest_raised_horizontal_top",
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_weights = values[order], weights[order]
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), 0.5 * weights.sum(), side="left")])


def _direct_box_slope(patches: tuple[_SurfacePatch, ...]) -> float | None:
    """Return a common material top-face pitch, not an average of unrelated ramps."""

    sloped = [patch for patch in patches if patch.pitch_deg >= 0.5]
    if not sloped:
        return None
    values = np.asarray([patch.pitch_deg for patch in sloped], dtype=float)
    weights = np.asarray([patch.area for patch in sloped], dtype=float)
    slope = _weighted_median(values, weights)
    tolerance = max(0.5, 0.10 * slope)
    agreeing_area = float(weights[np.abs(values - slope) <= tolerance].sum())
    if agreeing_area < 0.9 * float(weights.sum()):
        return None
    return slope


def _geometry_metrics(terrain: dict[str, Any] | None) -> _GeometryMetrics:
    patches = _surface_patches(terrain)
    if not patches:
        return _GeometryMetrics()

    direct_slope = _direct_box_slope(patches)
    if direct_slope is not None:
        return _GeometryMetrics(
            family="ramp",
            family_source="terrain_geometry.sloped_top_faces",
            slope_deg=direct_slope,
            slope_source="terrain_geometry.box_pitch",
        )
    return _GeometryMetrics()


def _selected_family(
    model: str | None,
    terrain: dict[str, Any] | None,
    geometry: _GeometryMetrics,
) -> tuple[str | None, str]:
    """Derive family without consulting the ground-truth class label.

    Current TERRA model labels map directly to their terrain family. An unknown
    representation is classified only when serialized geometry establishes the answer.
    """

    # The contact least-squares baseline always fits one affine plane; a pitched
    # output therefore reflects the fitted slope, not a ramp-versus-step family
    # decision.  Keep its slope available for grade error without fabricating a
    # family prediction from the serialized box.
    if model == "least_squares_contact_plane":
        return None, "not_applicable.no_family_selection"

    boxes = (terrain or {}).get("boxes") or []
    if model == "ramp":
        return "ramp", "terra_fit_report.model"
    if model in {"flat", "per_level", "stair_flight"}:
        return ("steps" if boxes else "flat"), "terra_fit_report.model"
    return geometry.family, geometry.family_source


def _terrain_metadata_path(terrain_dir: Path, motion: str) -> Path:
    return terrain_dir / f"{motion.replace('/', '__')}.json"


def _candidate_fit_statuses(path: Path) -> dict[str, dict[str, str]]:
    """Read the reconstruction status table that admits records to evaluation."""

    if not path.is_file():
        raise FileNotFoundError(f"reconstruction status table not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = {"motion", "method", "status", "output", "error"}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"reconstruction status table is missing fields {', '.join(missing)}: {path}")
        rows = list(reader)
    result: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        motion = (row.get("motion") or "").strip()
        status = (row.get("status") or "").strip()
        method = (row.get("method") or "").strip()
        if not motion or not status or not method:
            raise ValueError(f"reconstruction status row {index} has an empty identity field: {path}")
        if motion in result:
            raise ValueError(f"reconstruction status contains duplicate motion {motion!r}: {path}")
        result[motion] = row | {
            "motion": motion,
            "method": method,
            "status": status,
        }
    return result


def _fit_admission_error(
    motion: str,
    statuses: dict[str, dict[str, str]],
    status_path: Path,
) -> tuple[str | None, str]:
    row = statuses.get(motion)
    if row is None:
        return None, f"current candidate fit status is missing for {motion!r}: {status_path}"
    status = row["status"]
    if status in CURRENT_FIT_STATUSES:
        return status, ""
    detail = (row.get("error") or "").strip()
    suffix = f"; {detail}" if detail else ""
    return status, f"current candidate fit status is {status!r}{suffix}"


@dataclass(frozen=True)
class ReconstructionInputs:
    """Reconstruction cohort consumed by the apparatus scorer."""

    manifest: Path
    terrain_dir: Path
    run_path: Path
    status_path: Path
    method: str
    statuses: dict[str, dict[str, str]]
    provenance: ValidatedRunProvenance
    run_sha256: str


def load_reconstruction_inputs(
    manifest: Path,
    terrain_dir: Path,
    fit_status: Path | None,
) -> ReconstructionInputs:
    """Load the cohort metadata before any candidate record is opened."""

    selection = load_selection(manifest)
    root = terrain_dir.expanduser().resolve()
    run_path = root / "run.json"
    status_path = (fit_status or root / "status.csv").expanduser().resolve()
    if not run_path.is_file():
        raise FileNotFoundError(f"reconstruction cohort metadata not found: {run_path}")
    try:
        run = json.loads(run_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction cohort metadata: {run_path}") from error
    if not isinstance(run, dict):
        raise ValueError(f"reconstruction cohort metadata must contain an object: {run_path}")
    provenance = validate_run_provenance(run, run_path)
    method = run.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"reconstruction cohort has no method: {run_path}")
    if run.get("motions") != len(selection.motions):
        raise ValueError("reconstruction cohort motion count does not match the manifest")
    statuses = _candidate_fit_statuses(status_path)
    if set(statuses) != set(selection.motions):
        missing = sorted(set(selection.motions) - set(statuses))
        extra = sorted(set(statuses) - set(selection.motions))
        raise ValueError(
            f"reconstruction status denominator does not match the manifest; missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(row["method"] != method for row in statuses.values()):
        raise ValueError("reconstruction status method does not match run metadata")
    return ReconstructionInputs(
        manifest=selection.path,
        terrain_dir=root,
        run_path=run_path,
        status_path=status_path,
        method=method,
        statuses=statuses,
        provenance=provenance,
        run_sha256=file_sha256(run_path),
    )


def validate_reconstruction_record(
    path: Path,
    *,
    motion: str,
    status: dict[str, str],
    cohort: ReconstructionInputs,
) -> dict[str, Any]:
    """Load one reconstruction record and verify its identity and shape."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"reconstruction record not found: {path}")
    recorded_output = Path(status["output"])
    if recorded_output.name != path.name:
        raise ValueError(f"status output does not name the expected record for {motion!r}")
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction record: {path}") from error
    if not isinstance(record, dict):
        raise ValueError(f"reconstruction record must contain an object: {path}")
    if record.get("motion") != motion or record.get("method") != cohort.method:
        raise ValueError(f"reconstruction record identity mismatch for {motion!r}")
    expected_provenance = {
        "schema": RECORD_PROVENANCE_SCHEMA,
        "method_identity_sha256": cohort.provenance.method_identity_sha256,
        "scientific_identity_sha256": cohort.provenance.scientific_identity_sha256,
    }
    if record.get("provenance") != expected_provenance:
        raise ValueError(f"reconstruction record scientific identity mismatch for {motion!r}")
    scientific = {key: record.get(key) for key in ("terrain", "fit", "validation")}
    if not all(isinstance(value, dict) for value in scientific.values()):
        raise ValueError(f"reconstruction scientific result is incomplete for {motion!r}")
    return record


def _default_contact_reference_dir(terrain_dir: Path) -> Path:
    """Locate the sibling contact-least-squares records for this dataset."""

    root = terrain_dir.expanduser().resolve()
    if root.name != "terrain" or len(root.parents) < 2:
        raise ValueError(
            "--contact-reference-dir is required because TERRAIN_DIR does not use the "
            "DATASET/METHOD/terrain layout"
        )
    return root.parent.parent / "contact-least-squares" / "terrain"


def _reference_contact_points(record: dict[str, Any]) -> tuple[_ContactPoint, ...]:
    """Read source-only contacts serialized before contact-plane fitting.

    The contact least-squares baseline publishes the median source landmark position for
    every inferred interval.  Those positions do not depend on the plane it subsequently
    fits, so all reconstruction methods can be queried at the same coordinates.
    """

    report = record.get("fit") or {}
    intervals = report.get("support_intervals")
    if not isinstance(intervals, list):
        raise ValueError("contact reference fit has no support_intervals list")
    contacts: list[_ContactPoint] = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise ValueError(f"contact reference interval {index} is not an object")
        if interval.get("kind") != "foot":
            continue
        joint = interval.get("link")
        point = _vector(interval.get("surface_xyz_m"), 3)
        start, end = interval.get("start"), interval.get("end")
        valid_frames = (
            isinstance(start, (int, np.integer))
            and not isinstance(start, (bool, np.bool_))
            and isinstance(end, (int, np.integer))
            and not isinstance(end, (bool, np.bool_))
            and 0 <= int(start) < int(end)
        )
        if joint not in DEFAULT_CONTACT_JOINTS or point is None or not valid_frames:
            raise ValueError(f"contact reference interval {index} has no valid toe/ankle surface_xyz_m")
        contacts.append(_ContactPoint(str(joint), int(start), int(end), point))
    if not contacts:
        raise ValueError("contact reference contains no kinematic toe/ankle contacts")
    return tuple(contacts)


def _wrapped_grid_delta(value: np.ndarray, period: float) -> np.ndarray:
    """Signed distance to the nearest nominal step-height repetition."""

    return (value + 0.5 * period) % period - 0.5 * period


def _toe_contact_offset(contacts: tuple[_ContactPoint, ...], riser_m: float) -> float:
    """Fit the common toe landmark offset modulo the known riser height.

    A cropped traversal need not place every landmark on the floor.  Taking its minimum
    height can therefore mistake the first observed tread for the floor. The apparatus
    riser makes the toe-to-surface offset identifiable modulo one tread; pooling left and
    right toes and choosing the robust circular medoid recovers it without looking at any
    candidate terrain.
    """

    if not math.isfinite(riser_m) or riser_m <= 0.0:
        raise ValueError("expected riser height must be finite and positive")
    selected = [contact for contact in contacts if contact.joint.endswith("_Toe")]
    if not selected:
        raise ValueError("contact reference has no toe contacts")
    heights = np.asarray([contact.xyz[2] for contact in selected], dtype=float)
    wrapped = _wrapped_grid_delta(heights, riser_m)
    candidates = np.unique(wrapped)

    def key(candidate: float) -> tuple[float, float, float]:
        residual = np.abs(_wrapped_grid_delta(heights - candidate, riser_m))
        return float(np.median(residual)), float(np.mean(residual)), abs(float(candidate))

    return float(min(candidates, key=key))


def _contact_tread_ordinals(
    contacts: tuple[_ContactPoint, ...],
    riser_m: float,
) -> tuple[int, ...]:
    """Assign toe and ankle observations to the same physical tread.

    Toe landmarks lie close to the contact surface, so their height modulo the known
    riser identifies tread ordinals. An ankle can lie close to half a riser above its
    surface; independently rounding ankle heights therefore aliases between adjacent
    treads. Each ankle is instead paired to the overlapping same-side toe interval (or the
    nearest same-side interval when the two contact detectors have a small temporal gap).
    """

    toe_offset = _toe_contact_offset(contacts, riser_m)
    ordinals: list[int | None] = [None] * len(contacts)
    toe_indices: dict[str, list[int]] = {"L": [], "R": []}
    for index, contact in enumerate(contacts):
        if not contact.joint.endswith("_Toe"):
            continue
        ordinal = max(0, math.floor((contact.xyz[2] - toe_offset) / riser_m + 0.5))
        ordinals[index] = ordinal
        toe_indices[contact.joint[0]].append(index)

    for index, contact in enumerate(contacts):
        if not contact.joint.endswith("_Ankle"):
            continue
        candidates = toe_indices[contact.joint[0]]
        if not candidates:
            raise ValueError(f"contact reference has no same-side toe contacts for {contact.joint}")
        midpoint = 0.5 * (contact.start + contact.end)

        def pairing_key(
            toe_index: int,
            contact: _ContactPoint = contact,
            midpoint: float = midpoint,
        ) -> tuple[int, float, int]:
            toe = contacts[toe_index]
            overlap = max(0, min(contact.end, toe.end) - max(contact.start, toe.start))
            midpoint_gap = abs(midpoint - 0.5 * (toe.start + toe.end))
            return -overlap, midpoint_gap, toe_index

        paired = min(candidates, key=pairing_key)
        ordinals[index] = ordinals[paired]

    if any(value is None for value in ordinals):
        raise ValueError("contact reference contains an unsupported contact joint")
    return tuple(int(value) for value in ordinals)


def _score_step_contacts(
    terrain: dict[str, Any],
    reference_record: dict[str, Any],
    riser_m: float,
) -> _StepContactScore:
    """Compare reconstructed and nominal tread height at each raised contact."""

    contacts = _reference_contact_points(reference_record)
    ordinals = _contact_tread_ordinals(contacts, riser_m)
    candidate = TerrainMetadata.from_dict(terrain).terrain.walkable
    errors: list[float] = []
    biases: list[float] = []
    for contact, ordinal in zip(contacts, ordinals, strict=True):
        if ordinal == 0:
            continue
        expected = ordinal * riser_m
        predicted = float(candidate.height_at(contact.xyz[0], contact.xyz[1]))
        if not math.isfinite(predicted):
            raise ValueError("candidate terrain returned a non-finite contact height")
        bias = predicted - expected
        biases.append(bias)
        errors.append(abs(bias))
    if not errors:
        raise ValueError("contact reference contains no raised step contacts")
    return _StepContactScore(
        reference_contacts=len(contacts),
        raised_contacts=len(errors),
        errors_m=np.asarray(errors, dtype=float),
        bias_m=np.asarray(biases, dtype=float),
    )


def evaluate(
    manifest: Path,
    terrain_dir: Path,
    fit_status: Path | None = None,
    contact_reference_dir: Path | None = None,
) -> list[dict[str, Any]]:
    cohort = load_reconstruction_inputs(manifest, terrain_dir, fit_status)
    with cohort.manifest.open(newline="") as handle:
        selected = list(csv.DictReader(handle))
    if not selected or any(not row.get("motion", "").strip() for row in selected):
        raise ValueError(f"manifest contains no complete motion column: {manifest}")

    reference_cohort: ReconstructionInputs | None = None

    def contact_reference() -> ReconstructionInputs:
        nonlocal reference_cohort
        if reference_cohort is None:
            reference_root = (
                _default_contact_reference_dir(cohort.terrain_dir)
                if contact_reference_dir is None
                else contact_reference_dir
            )
            reference_cohort = load_reconstruction_inputs(manifest, reference_root, None)
        return reference_cohort

    rows: list[dict[str, Any]] = []
    for source in selected:
        motion = source["motion"].strip()
        path = _terrain_metadata_path(cohort.terrain_dir, motion)
        expected = _expected_geometry(source)
        slope_gt = expected.slope_deg
        riser_gt = expected.riser_m
        seat_height_gt = expected.seat_height_m
        expected_family = _expected_family(source)
        status = cohort.statuses[motion]
        row: dict[str, Any] = {
            "motion": motion,
            "reconstruction_method": cohort.method,
            "dataset": source.get("dataset", "").lower(),
            "condition": _condition(source),
            "terrain_class": source.get("terrain_class", ""),
            "expected_family": expected_family,
            "specification_source": expected.source,
            "expected_slope_deg": slope_gt,
            "expected_riser_m": riser_gt,
            "expected_seat_height_m": seat_height_gt,
            "candidate_fit_status": status["status"],
            "terrain_record_present": path.is_file(),
            "terrain_available": False,
            "selected_model": None,
            "selected_family": None,
            "selected_family_source": "unavailable",
            "family_correct": None,
            "validation_passed": None,
            "raised_contact_error_max_mm": None,
            "max_penetration_mm": None,
            "n_uncovered_contacts": None,
            "fitted_slope_deg": None,
            "fitted_slope_source": "unavailable",
            "slope_abs_error_deg": None,
            "step_contact_reference_method": None,
            "step_reference_contacts": None,
            "step_raised_contacts": None,
            "step_contact_height_mae_m": None,
            "step_contact_height_p95_m": None,
            "step_contact_height_max_m": None,
            "step_contact_height_bias_m": None,
            "step_contact_height_abs_error_sum_m": None,
            "raised_seat_support_present": None,
            "predicted_seat_height_m": None,
            "predicted_seat_height_source": "unavailable",
            "seat_height_abs_error_m": None,
            "n_boxes": None,
            "error": "",
        }
        _candidate_status, admission_error = _fit_admission_error(
            motion,
            cohort.statuses,
            cohort.status_path,
        )
        if admission_error:
            # Do not even parse a candidate when the current request failed.
            row["error"] = admission_error
            rows.append(row)
            continue
        try:
            payload = validate_reconstruction_record(
                path,
                motion=motion,
                status=status,
                cohort=cohort,
            )
            row["terrain_available"] = True
            report = payload.get("fit") or {}
            validation = payload.get("validation") or {}
            terrain = payload.get("terrain") or {}
            model = report.get("model")
            geometry = _geometry_metrics(terrain)
            selected_family, selected_family_source = _selected_family(model, terrain, geometry)
            ramp = report.get("ramp") or {}
            native_slope = _number(ramp.get("slope_deg")) if model == "ramp" else None
            fitted_slope = native_slope if native_slope is not None else geometry.slope_deg
            fitted_slope_source = (
                "terra_fit_report.ramp.slope_deg" if native_slope is not None else geometry.slope_source
            )
            seat_prediction = _predicted_seat_height(terrain) if seat_height_gt is not None else None
            step_score = None
            reference = None
            if riser_gt is not None:
                reference = contact_reference()
                _reference_status, reference_error = _fit_admission_error(
                    motion,
                    reference.statuses,
                    reference.status_path,
                )
                if reference_error:
                    raise ValueError(reference_error)
                reference_record = validate_reconstruction_record(
                    _terrain_metadata_path(reference.terrain_dir, motion),
                    motion=motion,
                    status=reference.statuses[motion],
                    cohort=reference,
                )
                step_score = _score_step_contacts(terrain, reference_record, riser_gt)
            row.update(
                selected_model=model,
                selected_family=selected_family,
                selected_family_source=selected_family_source,
                family_correct=(
                    selected_family == expected_family
                    if selected_family is not None and expected_family is not None
                    else None
                ),
                validation_passed=validation.get("passed"),
                raised_contact_error_max_mm=(
                    1000.0 * value
                    if (value := _number(validation.get("raised_contact_error_max"))) is not None
                    else None
                ),
                max_penetration_mm=(
                    1000.0 * value if (value := _number(validation.get("max_penetration"))) is not None else None
                ),
                n_uncovered_contacts=validation.get("n_uncovered_contacts"),
                fitted_slope_deg=fitted_slope,
                fitted_slope_source=fitted_slope_source,
                slope_abs_error_deg=(
                    abs(fitted_slope - slope_gt) if fitted_slope is not None and slope_gt is not None else None
                ),
                step_contact_reference_method=(reference.method if reference is not None else None),
                step_reference_contacts=(step_score.reference_contacts if step_score is not None else None),
                step_raised_contacts=(step_score.raised_contacts if step_score is not None else None),
                step_contact_height_mae_m=(
                    float(np.mean(step_score.errors_m)) if step_score is not None else None
                ),
                step_contact_height_p95_m=(
                    float(np.percentile(step_score.errors_m, 95)) if step_score is not None else None
                ),
                step_contact_height_max_m=(
                    float(np.max(step_score.errors_m)) if step_score is not None else None
                ),
                step_contact_height_bias_m=(
                    float(np.mean(step_score.bias_m)) if step_score is not None else None
                ),
                step_contact_height_abs_error_sum_m=(
                    float(np.sum(step_score.errors_m)) if step_score is not None else None
                ),
                raised_seat_support_present=(
                    seat_prediction.raised_support_present if seat_prediction is not None else None
                ),
                predicted_seat_height_m=(seat_prediction.height_m if seat_prediction is not None else None),
                predicted_seat_height_source=(
                    seat_prediction.source if seat_prediction is not None else "unavailable"
                ),
                seat_height_abs_error_m=(
                    abs(seat_prediction.height_m - seat_height_gt)
                    if seat_prediction is not None and seat_prediction.height_m is not None
                    else None
                ),
                n_boxes=len(terrain.get("boxes") or []),
            )
        except Exception as exc:  # preserve the failed row and make the stage fail at the end
            row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        rows.append(row)
    return rows


def _metric(values: Iterable[float | None], scale: float = 1.0) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None], dtype=float) * scale
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in group if row["terrain_available"] and not row["error"]]
    family = [row for row in available if row["family_correct"] is not None]
    validation = [row for row in available if row["validation_passed"] is not None]
    slopes = [row for row in group if row["expected_slope_deg"] is not None]
    steps = [row for row in group if row["expected_riser_m"] is not None]
    seats = [row for row in group if row["expected_seat_height_m"] is not None]
    slope_error = _metric(row["slope_abs_error_deg"] for row in slopes)
    step_height_error = _metric((row["step_contact_height_mae_m"] for row in steps), scale=1000.0)
    step_contacts = sum(int(row["step_raised_contacts"] or 0) for row in steps)
    step_error_sum = sum(float(row["step_contact_height_abs_error_sum_m"] or 0.0) for row in steps)
    seat_height_error = _metric((row["seat_height_abs_error_m"] for row in seats), scale=1000.0)
    return {
        "selected": len(group),
        "terrain_available": len(available),
        "errors": sum(bool(row["error"]) for row in group),
        "family_correct": sum(row["family_correct"] is True for row in family),
        "family_evaluated": len(family),
        "family_accuracy_pct": (
            100.0 * sum(row["family_correct"] is True for row in family) / len(family) if family else None
        ),
        "validation_passed": sum(row["validation_passed"] is True for row in validation),
        "validation_evaluated": len(validation),
        "raised_contact_error_max_mm": _metric(row["raised_contact_error_max_mm"] for row in available),
        "max_penetration_mm": _metric(row["max_penetration_mm"] for row in available),
        "motions_with_uncovered_contacts": sum(int(row["n_uncovered_contacts"] or 0) > 0 for row in available),
        "uncovered_contacts_total": sum(int(row["n_uncovered_contacts"] or 0) for row in available),
        "slope_expected": len(slopes),
        "slope_available": sum(row["slope_abs_error_deg"] is not None for row in slopes),
        "ramp_angle_mae_deg": slope_error["mean"],
        "slope_abs_error_deg": slope_error,
        "step_expected": len(steps),
        "step_available": sum(row["step_contact_height_mae_m"] is not None for row in steps),
        "step_contacts": step_contacts,
        "step_height_mae_mm": step_height_error["mean"],
        "step_height_micro_mae_mm": (1000.0 * step_error_sum / step_contacts if step_contacts else None),
        "step_height_per_motion_mae_mm": step_height_error,
        "seat_expected": len(seats),
        "seat_available": sum(row["seat_height_abs_error_m"] is not None for row in seats),
        "raised_seat_support_present": sum(row["raised_seat_support_present"] is True for row in seats),
        "seat_height_mae_mm": seat_height_error["mean"],
        "seat_height_abs_error_mm": seat_height_error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conditions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        datasets[row["dataset"]].append(row)
        conditions[(row["dataset"], row["condition"])].append(row)
    return {
        "motions": len(rows),
        "definitions": {
            "ground_truth": (
                "apparatus family, nominal ramp angle, nominal tread height at source contacts, and "
                "the published 0.490 m Gait120 stool height"
            ),
            "in_sample": "source contact residual, penetration, uncovered contacts, and validation pass",
            "family_rule": (
                "use the reconstruction method's recorded family when available; otherwise classify only when "
                "their serialized geometry establishes a family, never from the expected label"
            ),
            "geometry_fallback": (
                "a consistently pitched top surface directly establishes ramp family and grade; "
                "piecewise-horizontal heightfields establish neither"
            ),
            "metric_source_fields": (
                "selected_family_source and fitted_slope_source identify recorded report values, "
                "geometry-derived values, or unavailable metrics"
            ),
            "step_height": (
                "query every method at the same source-only kinematic toe/ankle contact XY; assign the "
                "contact to a nominal tread from its source height and the published riser, then compare "
                "predicted and nominal surface height; dataset MAE is the mean of per-motion contact MAEs"
            ),
            "seat_height": (
                "for Gait120 chair trials, compare the published 0.490 m stool height with the highest "
                "horizontal raised top in the prediction; no raised support is a numeric 0 m prediction, "
                "and malformed geometry is unavailable"
            ),
            "status_rule": (
                "only ok/cached status rows are scored; failed or missing rows remain in the selected cohort "
                "but their terrain files are not read"
            ),
        },
        "datasets": {dataset: _group_summary(datasets[dataset]) for dataset in sorted(datasets)},
        "conditions": {
            f"{dataset}/{condition}": _group_summary(conditions[(dataset, condition)])
            for dataset, condition in sorted(conditions)
        },
    }


def _format(value: object, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def write_report(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["motion", "error"]
    with (output / "per_motion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Dataset terrain-reconstruction quality",
        "",
        "Family accuracy, ramp-angle MAE, step-height MAE, and stool-height MAE use independent apparatus "
        "specifications. "
        "Step-height MAE queries every reconstruction at the same source-only kinematic contact points "
        "and compares its surface height with the nominal tread height at that contact. "
        "Source-contact residuals, penetration, uncovered contacts, and the validation decision are "
        "in-sample fit diagnostics and are not presented as independent ground truth. Native ramp "
        "metrics are preferred; otherwise a consistently pitched top surface can directly establish "
        "grade. Piecewise-horizontal heightfields have no ramp-grade or terrain-family score.",
        "",
        "| Dataset | Terrain records | Family accuracy | Internal validation | Raised-contact error median / p95 / max (mm) | Penetration median / p95 / max (mm) | Motions / contacts uncovered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in summary["datasets"].items():
        raised = item["raised_contact_error_max_mm"]
        penetration = item["max_penetration_mm"]
        lines.append(
            f"| {dataset} | {item['terrain_available']}/{item['selected']} | "
            f"{item['family_correct']}/{item['family_evaluated']} "
            f"({_format(item['family_accuracy_pct'])}%) | "
            f"{item['validation_passed']}/{item['validation_evaluated']} | "
            f"{_format(raised['median'])} / {_format(raised['p95'])} / {_format(raised['max'])} | "
            f"{_format(penetration['median'])} / {_format(penetration['p95'])} / "
            f"{_format(penetration['max'])} | {item['motions_with_uncovered_contacts']} / "
            f"{item['uncovered_contacts_total']} |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Geometry | Scoreable / expected | MAE / p95 / max |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset, item in summary["datasets"].items():
        if item["slope_expected"]:
            metric = item["slope_abs_error_deg"]
            lines.append(
                f"| {dataset} | Ramp slope (deg) | {item['slope_available']}/{item['slope_expected']} | "
                f"{_format(metric['mean'], 3)} / {_format(metric['p95'], 3)} / "
                f"{_format(metric['max'], 3)} |"
            )
        if item["step_expected"]:
            metric = item["step_height_per_motion_mae_mm"]
            lines.append(
                f"| {dataset} | Contact-conditioned step height (mm) | {item['step_available']}/"
                f"{item['step_expected']} | "
                f"{_format(metric['mean'], 3)} / {_format(metric['p95'], 3)} / "
                f"{_format(metric['max'], 3)} |"
            )
        if item["seat_expected"]:
            metric = item["seat_height_abs_error_mm"]
            lines.append(
                f"| {dataset} | Stool height (mm) | {item['seat_available']}/"
                f"{item['seat_expected']} | "
                f"{_format(metric['mean'], 3)} / {_format(metric['p95'], 3)} / "
                f"{_format(metric['max'], 3)} |"
            )
    lines.extend(
        [
            "",
            "Ramp length, height, and world placement are not reported because Vielemeyer publishes "
            "nominal angle but no per-trial surveyed pose for those quantities.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terra evaluate reconstruction", description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--terrain-dir", type=Path, required=True)
    parser.add_argument(
        "--fit-status",
        type=Path,
        help="Reconstruction status CSV; defaults to TERRAIN_DIR/status.csv.",
    )
    parser.add_argument(
        "--contact-reference-dir",
        type=Path,
        help=(
            "Contact least-squares terrain directory supplying the fixed source-only contact points; "
            "defaults to the sibling contact-least-squares/terrain directory."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rows = evaluate(
        args.manifest,
        args.terrain_dir,
        args.fit_status,
        args.contact_reference_dir,
    )
    summary = summarize(rows)
    write_report(args.output, rows, summary)
    write_git_commit(args.output)
    cohort = load_reconstruction_inputs(args.manifest, args.terrain_dir, args.fit_status)
    references: list[Path] = []
    if any(row.get("step_contact_reference_method") for row in rows):
        reference_root = (
            _default_contact_reference_dir(cohort.terrain_dir)
            if args.contact_reference_dir is None
            else args.contact_reference_dir
        )
        reference = load_reconstruction_inputs(args.manifest, reference_root, None)
        if reference.run_path != cohort.run_path:
            references.append(reference.run_path)
    write_evaluation_provenance(
        args.output,
        args.output / "per_motion.csv",
        cohort.run_path,
        reference_run_paths=references,
    )
    print(json.dumps(summary, indent=2))
    errors = [row for row in rows if row["error"]]
    if errors:
        print(f"{len(errors)} terrain record(s) missing or invalid")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
