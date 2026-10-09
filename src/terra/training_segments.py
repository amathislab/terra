"""Deterministic temporal segmentation for versioned training selections."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from terra._files import atomic_write
from terra._methods import RetargetingMethod, validate_method
from terra._revision import write_git_commit
from terra.artifacts import validate_retarget_artifacts
from terra.paths import StorageRoots

SEGMENT_POLICY = "balanced-temporal-v2"
SEGMENT_REQUIRED_FIELDS = (
    "source_motion",
    "segment_start_frame",
    "segment_end_frame_exclusive",
    "segment_index",
    "segment_count",
    "source_num_frames",
    "source_frequency_hz",
    "segment_policy",
)
SEGMENT_FIELDS = (*SEGMENT_REQUIRED_FIELDS, "segment_trigger_seconds", "segment_maximum_seconds")


@dataclass(frozen=True, slots=True)
class TemporalSegment:
    """A half-open frame interval in one source trajectory."""

    start_frame: int
    end_frame_exclusive: int

    @property
    def frames(self) -> int:
        return self.end_frame_exclusive - self.start_frame


@dataclass(frozen=True, slots=True)
class SelectionSegment:
    """Selected frame interval encoded in a training-selection row."""

    source_motion: str
    start_frame: int
    end_frame_exclusive: int
    index: int
    count: int
    source_num_frames: int
    source_frequency_hz: float
    policy: str
    trigger_seconds: float = 20.0
    maximum_segment_seconds: float = 10.0


def _positive_finite(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def temporal_segments(
    num_frames: int,
    frequency_hz: float,
    *,
    trigger_seconds: float = 20.0,
    maximum_segment_seconds: float = 10.0,
) -> tuple[TemporalSegment, ...]:
    """Partition only clips above the trigger into balanced bounded segments.

    Clips at or below ``trigger_seconds`` are returned unchanged. Longer clips
    use the minimum possible number of contiguous segments. Frames are divided
    as evenly as possible, which retains the complete clip without producing a
    short tail segment.
    """

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 2:
        raise ValueError("num_frames must be an integer of at least two")
    frequency = _positive_finite(frequency_hz, "frequency_hz")
    trigger = _positive_finite(trigger_seconds, "trigger_seconds")
    maximum = _positive_finite(maximum_segment_seconds, "maximum_segment_seconds")
    if maximum > trigger:
        raise ValueError("maximum_segment_seconds cannot exceed trigger_seconds")

    if num_frames / frequency <= trigger:
        return (TemporalSegment(0, num_frames),)

    maximum_frames = math.floor(maximum * frequency + 1.0e-9)
    if maximum_frames < 2:
        raise ValueError("maximum_segment_seconds must contain at least two frames")
    count = math.ceil(num_frames / maximum_frames)
    base_frames, longer_segments = divmod(num_frames, count)
    lengths = tuple(base_frames + (index < longer_segments) for index in range(count))
    if min(lengths) < 2 or max(lengths) > maximum_frames:
        raise RuntimeError("internal temporal partition violated its frame bounds")

    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return tuple(
        TemporalSegment(start, end)
        for start, end in pairwise(boundaries)
    )


def segmented_motion_name(source_motion: str, index: int, count: int) -> str:
    """Return the stable cache identity for one derived temporal segment."""

    if not source_motion:
        raise ValueError("source_motion must be non-empty")
    if count < 2 or not 1 <= index <= count:
        raise ValueError("segment index/count must identify one of at least two segments")
    return f"{source_motion}/__terra_segment_{index:03d}_of_{count:03d}"


def selection_segment(row: Mapping[str, str]) -> SelectionSegment | None:
    """Decode and fully validate optional segment fields from a manifest row."""

    values = {field: str(row.get(field, "")).strip() for field in SEGMENT_FIELDS}
    populated = {field for field, value in values.items() if value}
    if not populated:
        return None
    missing = set(SEGMENT_REQUIRED_FIELDS) - populated
    if missing:
        raise ValueError(
            f"segmented selection row {row.get('motion', '')!r} is missing: {', '.join(sorted(missing))}"
        )
    try:
        segment = SelectionSegment(
            source_motion=values["source_motion"],
            start_frame=int(values["segment_start_frame"]),
            end_frame_exclusive=int(values["segment_end_frame_exclusive"]),
            index=int(values["segment_index"]),
            count=int(values["segment_count"]),
            source_num_frames=int(values["source_num_frames"]),
            source_frequency_hz=float(values["source_frequency_hz"]),
            policy=values["segment_policy"],
            trigger_seconds=float(values["segment_trigger_seconds"] or 20.0),
            maximum_segment_seconds=float(values["segment_maximum_seconds"] or 10.0),
        )
    except ValueError as error:
        raise ValueError(f"segmented selection row {row.get('motion', '')!r} has invalid numeric fields") from error
    if segment.policy != SEGMENT_POLICY:
        raise ValueError(f"unsupported segment policy {segment.policy!r}")
    expected = temporal_segments(
        segment.source_num_frames, segment.source_frequency_hz,
        trigger_seconds=segment.trigger_seconds, maximum_segment_seconds=segment.maximum_segment_seconds,
    )
    if segment.count != len(expected) or not 1 <= segment.index <= segment.count:
        raise ValueError(f"segmented selection row {row.get('motion', '')!r} has inconsistent index/count")
    bounds = expected[segment.index - 1]
    if (segment.start_frame, segment.end_frame_exclusive) != (
        bounds.start_frame,
        bounds.end_frame_exclusive,
    ):
        raise ValueError(f"segmented selection row {row.get('motion', '')!r} has non-canonical frame bounds")
    expected_motion = segmented_motion_name(segment.source_motion, segment.index, segment.count)
    if row.get("motion") != expected_motion:
        raise ValueError(
            f"segmented selection motion must be {expected_motion!r}, got {row.get('motion')!r}"
        )
    return segment


def _resolved_source_root(
    row: Mapping[str, str],
    roots: StorageRoots,
    *,
    base: Path,
) -> Path:
    source = roots.resolve_artifact(row.get("source_cache_root", ""), base=base)
    if source is None:
        raise ValueError(f"selection has no source cache for {row.get('motion', '')!r}")
    return source


def _verified_source_paths(
    row: Mapping[str, str],
    source_root: Path,
    *,
    method: RetargetingMethod,
):
    motion = row.get("motion", "").strip()
    row_method = row.get("retargeting_method", "").strip()
    if row_method and row_method != method:
        raise ValueError(
            f"selection row {motion!r} declares retargeting_method={row_method!r}, expected {method!r}"
        )
    validated = validate_retarget_artifacts(source_root, motion, method=method)
    expected = {
        "trajectory_relpath": validated.trajectory_path,
        "analysis_relpath": validated.analysis_path,
    }
    if validated.terrain_path is not None:
        expected["terrain_relpath"] = validated.terrain_path
    for field, expected_path in expected.items():
        relative = row.get(field, "").strip()
        if not relative:
            raise ValueError(f"selection row {motion!r} is missing {field}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe artifact path for {motion!r}: {path}")
        if (source_root / path).resolve() != expected_path:
            raise ValueError(f"selection row {motion!r} {field} does not match its canonical artifact")
    if validated.terrain_path is None and row.get("terrain_relpath", "").strip():
        raise ValueError(f"selection row {motion!r} declares terrain absent from its validated artifacts")
    return validated


def expand_training_segments(
    rows: Sequence[Mapping[str, str]],
    *,
    storage_roots: StorageRoots | None = None,
    base: Path | None = None,
    method: RetargetingMethod = "terra",
    trigger_seconds: float = 20.0,
    maximum_segment_seconds: float = 10.0,
    required_frequency_hz: float | None = None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Expand long source rows while preserving every row's dataset and split."""

    if not rows:
        raise ValueError("training selection must contain at least one motion")
    root_base = Path.cwd() if base is None else base
    roots = storage_roots or StorageRoots.from_environment(root_base)
    selected_method = validate_method(method)
    required_frequency = (
        None
        if required_frequency_hz is None
        else _positive_finite(required_frequency_hz, "required_frequency_hz")
    )
    output: list[dict[str, str]] = []
    source_motions: set[str] = set()
    output_motions: set[str] = set()
    duration_hours = 0.0
    maximum_source_duration = 0.0
    maximum_output_duration = 0.0
    long_clips: list[dict[str, object]] = []
    by_dataset: dict[str, dict[str, int]] = defaultdict(
        lambda: {"source_motions": 0, "segmented_sources": 0, "output_motions": 0}
    )
    by_split: dict[str, dict[str, int]] = defaultdict(
        lambda: {"source_motions": 0, "segmented_sources": 0, "output_motions": 0}
    )

    for original in rows:
        row = {str(key): str(value) for key, value in original.items()}
        motion = row.get("motion", "").strip()
        dataset = row.get("dataset", "").strip()
        split = row.get("split", "train").strip() or "train"
        if not motion or motion in source_motions:
            raise ValueError(f"selection contains an empty or duplicate source motion: {motion!r}")
        if any(row.get(field, "").strip() for field in SEGMENT_FIELDS):
            raise ValueError(f"selection is already segmented: {motion!r}")
        source_motions.add(motion)
        source_root = _resolved_source_root(row, roots, base=root_base)
        validated = _verified_source_paths(row, source_root, method=selected_method)
        if required_frequency is not None and not math.isclose(
            validated.frequency,
            required_frequency,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"selection row {motion!r} has frequency {validated.frequency:.12g} Hz, "
                f"expected {required_frequency:.12g} Hz"
            )
        segments = temporal_segments(
            validated.num_frames,
            validated.frequency,
            trigger_seconds=trigger_seconds,
            maximum_segment_seconds=maximum_segment_seconds,
        )
        source_duration = validated.num_frames / validated.frequency
        duration_hours += source_duration / 3600.0
        maximum_source_duration = max(maximum_source_duration, source_duration)
        by_dataset[dataset]["source_motions"] += 1
        by_split[split]["source_motions"] += 1

        if len(segments) == 1:
            expanded = row | dict.fromkeys(SEGMENT_FIELDS, "")
            output.append(expanded)
            output_motions.add(motion)
            by_dataset[dataset]["output_motions"] += 1
            by_split[split]["output_motions"] += 1
            maximum_output_duration = max(maximum_output_duration, source_duration)
            continue

        by_dataset[dataset]["segmented_sources"] += 1
        by_split[split]["segmented_sources"] += 1
        segment_records = []
        for zero_based_index, segment in enumerate(segments):
            index = zero_based_index + 1
            segment_motion = segmented_motion_name(motion, index, len(segments))
            if segment_motion in output_motions:
                raise ValueError(f"segmentation produced duplicate motion {segment_motion!r}")
            output_motions.add(segment_motion)
            segment_duration = segment.frames / validated.frequency
            maximum_output_duration = max(maximum_output_duration, segment_duration)
            expanded = row | {
                "motion": segment_motion,
                "source_motion": motion,
                "segment_start_frame": str(segment.start_frame),
                "segment_end_frame_exclusive": str(segment.end_frame_exclusive),
                "segment_index": str(index),
                "segment_count": str(len(segments)),
                "source_num_frames": str(validated.num_frames),
                "source_frequency_hz": format(validated.frequency, ".12g"),
                "segment_policy": SEGMENT_POLICY,
                "segment_trigger_seconds": format(float(trigger_seconds), ".12g"),
                "segment_maximum_seconds": format(float(maximum_segment_seconds), ".12g"),
            }
            output.append(expanded)
            by_dataset[dataset]["output_motions"] += 1
            by_split[split]["output_motions"] += 1
            segment_records.append(asdict(segment) | {"duration_s": segment_duration})
        long_clips.append(
            {
                "motion": motion,
                "dataset": dataset,
                "split": split,
                "frames": validated.num_frames,
                "frequency_hz": validated.frequency,
                "duration_s": source_duration,
                "segments": segment_records,
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "retargeting_method": selected_method,
        "required_frequency_hz": required_frequency,
        "segment_policy": SEGMENT_POLICY,
        "trigger_seconds": float(trigger_seconds),
        "maximum_segment_seconds": float(maximum_segment_seconds),
        "source_motion_count": len(source_motions),
        "segmented_source_count": len(long_clips),
        "output_motion_count": len(output),
        "additional_motion_count": len(output) - len(source_motions),
        "source_duration_hours": duration_hours,
        "maximum_source_duration_s": maximum_source_duration,
        "maximum_output_duration_s": maximum_output_duration,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_split": dict(sorted(by_split.items())),
        "long_clips": long_clips,
    }
    return output, report


def publish_segmented_selection(
    source_manifest: Path,
    destination_manifest: Path,
    report_path: Path,
    rows: Sequence[Mapping[str, str]],
    report: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Atomically publish a segmented selection and its complete report."""

    source = source_manifest.expanduser().resolve()
    destination = destination_manifest.expanduser().resolve()
    resolved_report = report_path.expanduser().resolve()
    if destination.suffix.casefold() != ".csv":
        raise ValueError("segmented training selection must use a .csv extension")
    if resolved_report.suffix.casefold() != ".json":
        raise ValueError("segmentation report must use a .json extension")
    if destination in {source, resolved_report} or resolved_report == source:
        raise ValueError("source selection, segmented selection, and report paths must differ")
    collisions = [path for path in (destination, resolved_report) if path.exists()]
    if collisions and not overwrite:
        raise FileExistsError(f"refusing to replace existing segmentation output: {collisions[0]}")
    if not rows:
        raise ValueError("segmented training selection must be non-empty")

    fieldnames = list(rows[0])
    for field in SEGMENT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    def write_csv(path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(destination, write_csv)
    payload = dict(report) | {
        "source_selection": str(source),
        "selection": str(destination),
    }
    atomic_write(
        resolved_report,
        lambda temporary: temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"),
    )
    write_git_commit(destination.parent)
    return payload


__all__ = [
    "SEGMENT_FIELDS",
    "SEGMENT_POLICY",
    "SelectionSegment",
    "TemporalSegment",
    "expand_training_segments",
    "publish_segmented_selection",
    "segmented_motion_name",
    "selection_segment",
    "temporal_segments",
]
