"""Materialize a cross-dataset TERRA training subset into one cache root."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from terra._files import atomic_write, file_sha256
from terra._methods import SUPPORTED_METHODS, RetargetingMethod, validate_method
from terra._revision import write_git_commit
from terra.artifacts import (
    RetargetSegmentRequest,
    load_retarget_analysis,
    retarget_cache_paths,
    save_retarget_segments,
    validate_retarget_artifacts,
)
from terra.paths import StorageRoots
from terra.training_segments import SelectionSegment, selection_segment

ARTIFACT_FIELDS = (
    "trajectory_relpath",
    "analysis_relpath",
    "terrain_relpath",
)
TERRAIN_MODES = ("flat", "nonflat", "mixed")
SPLITS = ("train", "evaluation", "test")


@dataclass(frozen=True, slots=True)
class _ResolvedRow:
    row: dict[str, str]
    source_root: Path
    split: str
    segment: SelectionSegment | None


def _atomic_transfer(source: Path, destination: Path, mode: str, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) == file_sha256(source):
            return
        if not overwrite:
            raise FileExistsError(f"destination differs from frozen source: {destination}")
    staged = destination.with_name(f".{destination.name}.tmp")
    staged.unlink(missing_ok=True)
    try:
        if mode == "hardlink":
            os.link(source, staged)
        else:
            shutil.copy2(source, staged)
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a nonempty selection CSV with unique portable motion IDs.

    The required columns name each source cache and its published trajectory,
    analysis, and optional terrain path. This checks the table shape; actual
    paths, artifact identity, terrain mode, and segment provenance are checked
    later by :func:`materialize_subset`.
    """

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "motion",
        "dataset",
        "source_cache_root",
        "trajectory_relpath",
        "analysis_relpath",
        "terrain_relpath",
    }
    missing = required - set(rows[0] if rows else ())
    if not rows:
        raise ValueError(f"training subset manifest is empty: {path}")
    if missing:
        raise ValueError(f"training subset manifest is missing columns: {', '.join(sorted(missing))}")
    motions = [row["motion"] for row in rows]
    if any(not motion for motion in motions) or len(motions) != len(set(motions)):
        raise ValueError(f"training subset contains empty or duplicate motion IDs: {path}")
    return rows


def _canonical_source_artifact(
    source_root: Path,
    relative_value: str,
    expected: Path,
    *,
    motion: str,
    field: str,
) -> None:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path for {motion}: {relative}")
    if (source_root / relative).resolve() != expected:
        raise ValueError(f"segmented selection {field} does not match its source artifact: {motion!r}")


def _segment_analysis_matches(path: Path, segment: SelectionSegment) -> bool:
    analysis = load_retarget_analysis(path)
    expected = {
        "segment_policy": segment.policy,
        "segment_source_motion": segment.source_motion,
        "segment_source_num_frames": segment.source_num_frames,
        "segment_start_frame": segment.start_frame,
        "segment_end_frame_exclusive": segment.end_frame_exclusive,
        "segment_index": segment.index,
        "segment_count": segment.count,
    }
    frequency = analysis.get("segment_source_frequency_hz")
    return (
        all(analysis.get(field) == value for field, value in expected.items())
        and isinstance(frequency, int | float)
        and math.isclose(float(frequency), segment.source_frequency_hz, rel_tol=1.0e-11)
    )


def _validate_segment_group(items: Sequence[_ResolvedRow]) -> None:
    """Require one complete, consistently classified set of sibling segments."""

    first = items[0]
    segment = first.segment
    assert segment is not None
    expected_indices = set(range(1, segment.count + 1))
    observed_indices = {item.segment.index for item in items if item.segment is not None}
    if len(items) != segment.count or observed_indices != expected_indices:
        raise ValueError(
            f"segmented selection must include every interval for {segment.source_motion!r}; "
            f"expected indices 1..{segment.count}, got {sorted(observed_indices)}"
        )

    shared_row_fields = ("dataset", "motion_type", "trajectory_relpath", "analysis_relpath", "terrain_relpath")
    expected_row_values = tuple(first.row.get(field, "") for field in shared_row_fields)
    expected_segment_values = (
        segment.source_motion,
        segment.count,
        segment.source_num_frames,
        segment.source_frequency_hz,
        segment.policy,
        segment.trigger_seconds,
        segment.maximum_segment_seconds,
    )
    for item in items:
        current = item.segment
        assert current is not None
        if item.source_root != first.source_root or item.split != first.split:
            raise ValueError(f"sibling segments cross source roots or splits: {segment.source_motion!r}")
        if tuple(item.row.get(field, "") for field in shared_row_fields) != expected_row_values:
            raise ValueError(f"sibling segments have inconsistent source metadata: {segment.source_motion!r}")
        if (
            current.source_motion,
            current.count,
            current.source_num_frames,
            current.source_frequency_hz,
            current.policy,
            current.trigger_seconds,
            current.maximum_segment_seconds,
        ) != expected_segment_values:
            raise ValueError(f"sibling segments have inconsistent temporal metadata: {segment.source_motion!r}")


def _materialize_segment_group(
    items: Sequence[_ResolvedRow],
    destination_cache: Path,
    *,
    overwrite: bool,
    method: RetargetingMethod,
    env_name: str,
    terrain_mode: str,
) -> dict[str, dict[str, str]]:
    """Materialize sibling intervals while loading their source archive once."""

    first = items[0]
    row = first.row
    segment = first.segment
    assert segment is not None
    source_root = first.source_root
    source_paths = retarget_cache_paths(source_root, segment.source_motion, method=method, env_name=env_name)
    _canonical_source_artifact(
        source_root,
        row["trajectory_relpath"],
        source_paths.trajectory_path,
        motion=segment.source_motion,
        field="trajectory_relpath",
    )
    _canonical_source_artifact(
        source_root,
        row["analysis_relpath"],
        source_paths.analysis_path,
        motion=segment.source_motion,
        field="analysis_relpath",
    )
    if row["terrain_relpath"]:
        _canonical_source_artifact(
            source_root,
            row["terrain_relpath"],
            source_paths.terrain_path,
            motion=segment.source_motion,
            field="terrain_relpath",
        )
    elif source_paths.terrain_path.exists():
        raise ValueError(f"segmented selection omits its source terrain artifact: {segment.source_motion!r}")

    requests: list[RetargetSegmentRequest] = []
    copied_by_motion: dict[str, dict[str, str]] = {}
    for item in items:
        item_segment = item.segment
        assert item_segment is not None
        destination_paths = retarget_cache_paths(
            destination_cache,
            item.row["motion"],
            method=method,
            env_name=env_name,
        )
        required_outputs = [destination_paths.trajectory_path, destination_paths.analysis_path]
        if row["terrain_relpath"]:
            required_outputs.append(destination_paths.terrain_path)
        existing = [path for path in required_outputs if path.exists()]
        if existing and len(existing) != len(required_outputs) and not overwrite:
            raise FileExistsError(f"segment artifacts are incomplete for {item.row['motion']!r}")
        if len(existing) == len(required_outputs) and not overwrite:
            try:
                validated = validate_retarget_artifacts(
                    destination_cache,
                    item.row["motion"],
                    method=method,
                    env_name=env_name,
                    require_nonflat_terrain=terrain_mode == "nonflat",
                )
            except (FileNotFoundError, ValueError) as error:
                raise FileExistsError(
                    f"existing segment artifacts are invalid for {item.row['motion']!r}"
                ) from error
            if (
                validated.num_frames != item_segment.end_frame_exclusive - item_segment.start_frame
                or not _segment_analysis_matches(validated.analysis_path, item_segment)
            ):
                raise FileExistsError(f"existing segment artifacts differ from the manifest: {item.row['motion']!r}")
            copied_by_motion[item.row["motion"]] = _materialized_paths(validated)
            continue
        requests.append(
            RetargetSegmentRequest(
                motion_name=item.row["motion"],
                start_frame=item_segment.start_frame,
                end_frame_exclusive=item_segment.end_frame_exclusive,
                segment_index=item_segment.index,
                segment_count=item_segment.count,
                segment_policy=item_segment.policy,
            )
        )

    if requests:
        save_retarget_segments(
            source_root,
            segment.source_motion,
            destination_cache,
            requests,
            method=method,
            env_name=env_name,
            overwrite=overwrite,
        )
    for item in items:
        if item.row["motion"] in copied_by_motion:
            continue
        validated = validate_retarget_artifacts(
            destination_cache,
            item.row["motion"],
            method=method,
            env_name=env_name,
            require_nonflat_terrain=terrain_mode == "nonflat",
        )
        copied_by_motion[item.row["motion"]] = _materialized_paths(validated)
    return copied_by_motion


def _materialized_paths(validated) -> dict[str, str]:
    copied = {
        "trajectory_relpath": str(validated.trajectory_path),
        "analysis_relpath": str(validated.analysis_path),
    }
    if validated.terrain_path is not None:
        copied["terrain_relpath"] = str(validated.terrain_path)
    return copied


def materialize_subset(
    rows: list[dict[str, str]],
    destination_cache: Path,
    *,
    policy_root: Path | None = None,
    link_mode: str = "hardlink",
    overwrite: bool = False,
    method: RetargetingMethod = "terra",
    env_name: str = "MyoFullBody",
    terrain_mode: str = "nonflat",
    storage_roots: StorageRoots | None = None,
) -> dict[str, object]:
    """Validate and collect a declared training subset into one cache.

    Each CSV row needs ``motion``, ``dataset``, ``source_cache_root``, and
    relative trajectory, analysis, and terrain paths. ``split`` may be
    ``train``, ``evaluation``, or ``test``; a test row retains only its
    identity. ``terrain_mode`` can require flat or non-flat terrain, or
    accept a verified mixture. The returned JSON-compatible mapping is
    the record consumed by :func:`terra.training.prepare_training_launch`.
    """

    if terrain_mode not in TERRAIN_MODES:
        raise ValueError(f"terrain_mode must be one of {', '.join(TERRAIN_MODES)}")
    selected_method = validate_method(method)
    roots = storage_roots or StorageRoots.from_environment(Path.cwd())
    resolved_destination = roots.resolve_artifact(destination_cache, base=Path.cwd())
    assert resolved_destination is not None
    destination_cache = resolved_destination
    write_git_commit(destination_cache)
    resolved_policy_root = roots.resolve_artifact(policy_root, base=Path.cwd()) if policy_root is not None else None
    row_source_roots: list[Path] = []
    dataset_source_roots: dict[str, set[Path]] = defaultdict(set)
    for row in rows:
        source = (
            resolved_policy_root / row["dataset"] / "cache"
            if resolved_policy_root is not None
            else roots.resolve_artifact(row["source_cache_root"], base=Path.cwd())
        )
        assert source is not None
        if source == destination_cache:
            raise ValueError("source and destination cache roots must differ")
        row_source_roots.append(source)
        dataset_source_roots[row["dataset"]].add(source)

    resolved_rows: list[_ResolvedRow] = []
    segment_groups: dict[tuple[Path, str], list[_ResolvedRow]] = defaultdict(list)
    for row, source_root in zip(rows, row_source_roots, strict=True):
        row_method = row.get("retargeting_method", "").strip()
        if row_method and row_method != selected_method:
            raise ValueError(
                f"selection row {row['motion']!r} declares retargeting_method={row_method!r}, "
                f"expected {selected_method!r}"
            )
        segment = selection_segment(row)
        split = row.get("split", "train").strip() or "train"
        if split not in SPLITS:
            raise ValueError(
                f"training subset split must be one of {', '.join(SPLITS)}, "
                f"got {split!r} for {row['motion']!r}"
            )
        resolved = _ResolvedRow(row=row, source_root=source_root, split=split, segment=segment)
        resolved_rows.append(resolved)
        if segment is not None:
            segment_groups[(source_root, segment.source_motion)].append(resolved)

    unsegmented_motions = {item.row["motion"] for item in resolved_rows if item.segment is None}
    segmented_sources = {item.segment.source_motion for item in resolved_rows if item.segment is not None}
    overlap = sorted(unsegmented_motions & segmented_sources)
    if overlap:
        raise ValueError(f"selection includes both a source and its temporal segments: {overlap[0]!r}")

    segment_outputs: dict[str, dict[str, str]] = {}
    for items in segment_groups.values():
        _validate_segment_group(items)
        if items[0].split != "test":
            segment_outputs.update(
                _materialize_segment_group(
                    items,
                    destination_cache,
                    overwrite=overwrite,
                    method=selected_method,
                    env_name=env_name,
                    terrain_mode=terrain_mode,
                )
            )

    materialized: list[dict[str, object]] = []
    for item in resolved_rows:
        row = item.row
        source_root = item.source_root
        split = item.split
        segment = item.segment
        if split == "test":
            # Retain held-out identities in provenance without copying or
            # validating their artifacts as part of a training launch.
            materialized.append(
                {
                    "motion": row["motion"],
                    "dataset": row["dataset"],
                    "motion_type": row.get("motion_type", ""),
                    "split": split,
                    "source_motion": "" if segment is None else segment.source_motion,
                    "segment_start_frame": None if segment is None else segment.start_frame,
                    "segment_end_frame_exclusive": None if segment is None else segment.end_frame_exclusive,
                    "paths": {},
                }
            )
            continue
        if segment is None:
            copied: dict[str, str] = {}
            for relative_field in ARTIFACT_FIELDS:
                if terrain_mode != "nonflat" and relative_field == "terrain_relpath" and not row[relative_field]:
                    continue
                relative = Path(row[relative_field])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe artifact path for {row['motion']}: {relative}")
                source = source_root / relative
                destination = destination_cache / relative
                if not source.is_file():
                    if terrain_mode != "nonflat" and relative_field == "terrain_relpath":
                        continue
                    raise FileNotFoundError(f"source artifact is missing: {source}")
                _atomic_transfer(source, destination, link_mode, overwrite=overwrite)
                copied[relative_field] = str(destination)
        else:
            copied = segment_outputs[row["motion"]]
        validated = validate_retarget_artifacts(
            destination_cache,
            row["motion"],
            method=selected_method,
            env_name=env_name,
            require_nonflat_terrain=terrain_mode == "nonflat",
        )
        if terrain_mode == "flat":
            if validated.nonflat_terrain:
                raise ValueError(f"flat training selection contains non-flat terrain: {row['motion']!r}")
        materialized.append(
            {
                "motion": row["motion"],
                "dataset": row["dataset"],
                "motion_type": row.get("motion_type", ""),
                "split": split,
                "source_motion": "" if segment is None else segment.source_motion,
                "segment_start_frame": None if segment is None else segment.start_frame,
                "segment_end_frame_exclusive": None if segment is None else segment.end_frame_exclusive,
                "paths": copied,
            }
        )

    split_counts = {
        split: sum(row["split"] == split for row in materialized)
        for split in SPLITS
    }
    if split_counts["train"] < 1:
        raise ValueError("training subset must contain at least one train motion")

    return {
        "destination_cache": str(destination_cache),
        "retargeting_method": selected_method,
        "link_mode": link_mode,
        "terrain_mode": terrain_mode,
        "robot_shape": None,
        "source_caches": {
            dataset: (
                str(next(iter(roots_for_dataset)))
                if len(roots_for_dataset) == 1
                else [str(root) for root in sorted(roots_for_dataset)]
            )
            for dataset, roots_for_dataset in sorted(dataset_source_roots.items())
        },
        "storage_roots": roots.as_dict(),
        "split_counts": split_counts,
        "motions": materialized,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra train materialize", description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--destination-cache", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--retargeting-method",
        choices=SUPPORTED_METHODS,
        default="terra",
        help="artifact namespace to materialize",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env-name", default="MyoFullBody")
    parser.add_argument(
        "--terrain-mode",
        choices=TERRAIN_MODES,
        default="nonflat",
        help="require flat, non-flat, or allow a verified mixture of both terrain kinds",
    )
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)

    try:
        roots = StorageRoots.from_environment(Path.cwd())
        selection_manifest = roots.resolve_input(args.selection_manifest, base=Path.cwd())
        assert selection_manifest is not None
        payload = materialize_subset(
            read_manifest(selection_manifest),
            args.destination_cache,
            policy_root=args.policy_root,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
            method=args.retargeting_method,
            env_name=args.env_name,
            terrain_mode=args.terrain_mode,
            storage_roots=roots,
        )
        payload["selection"] = {
            "manifest": str(selection_manifest),
        }
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.record is not None:
        record = roots.resolve_artifact(args.record, base=Path.cwd())
        assert record is not None
        atomic_write(record, lambda path: path.write_text(rendered, encoding="utf-8"))
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
