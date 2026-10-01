"""Tests for bounded temporal views of long training trajectories."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

import terra.artifacts as artifacts
from terra.artifacts import (
    load_retarget_analysis,
    retarget_cache_paths,
    save_retarget_result,
    validate_retarget_artifacts,
)
from terra.commands.materialize import materialize_subset
from terra.contracts import RetargetResult
from terra.training_segments import (
    SEGMENT_POLICY,
    expand_training_segments,
    segmented_motion_name,
    selection_segment,
    temporal_segments,
)


class _TrajectoryArchive:
    def __init__(self, frames: int, frequency: float) -> None:
        self.frames = frames
        self.frequency = frequency

    def save(self, path: str) -> None:
        frames = self.frames
        qpos = np.arange(frames * 8, dtype=np.float32).reshape(frames, 8)
        np.savez(
            path,
            qpos=qpos,
            qvel=-qpos[:, :7],
            site_xpos=np.arange(frames * 2 * 3, dtype=np.float32).reshape(frames, 2, 3),
            split_points=np.asarray([0, frames], dtype=np.int32),
            frequency=np.asarray(self.frequency),
            joint_names=np.asarray(["root", "hip"]),
        )


def _source_row(
    tmp_path: Path,
    *,
    frames: int,
    frequency: float,
    split: str = "train",
    method: str = "terra",
) -> dict[str, str]:
    cache_root = tmp_path / "source-cache"
    motion = "PRISM/subj001/take001_poses"
    save_retarget_result(
        RetargetResult(
            trajectory=_TrajectoryArchive(frames, frequency),
            analysis={"source_diagnostic": "retargeted-parent-only"},
            terrain=None,
            method=method,
            source_path=tmp_path / "source-motion.npz",
        ),
        cache_root,
        motion,
    )
    paths = retarget_cache_paths(cache_root, motion, method=method)
    return {
        "motion": motion,
        "dataset": "prism",
        "source_cache_root": str(cache_root),
        "trajectory_relpath": str(paths.trajectory_path.relative_to(cache_root)),
        "analysis_relpath": str(paths.analysis_path.relative_to(cache_root)),
        "terrain_relpath": "",
        "motion_type": "uneven_terrain",
        "split": split,
        "retargeting_method": method,
    }


def test_temporal_segments_leave_twenty_seconds_and_shorter_untouched() -> None:
    for frames, frequency in ((2_000, 100.0), (1_000, 50.0), (1_999, 100.0)):
        segments = temporal_segments(frames, frequency)
        assert len(segments) == 1
        assert segments[0].start_frame == 0
        assert segments[0].end_frame_exclusive == frames


def test_temporal_segments_balance_long_clips_without_dropping_or_repeating_frames() -> None:
    segments = temporal_segments(2_501, 100.0)

    assert [segment.frames for segment in segments] == [834, 834, 833]
    assert segments[0].start_frame == 0
    assert segments[-1].end_frame_exclusive == 2_501
    assert all(left.end_frame_exclusive == right.start_frame for left, right in pairwise(segments))
    assert all(segment.frames <= 1_000 for segment in segments)


@pytest.mark.parametrize(
    ("frames", "frequency", "message"),
    ((1, 100.0, "at least two"), (100, 0.0, "positive and finite")),
)
def test_temporal_segments_reject_invalid_source_metadata(frames: int, frequency: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        temporal_segments(frames, frequency)


def test_expand_and_materialize_long_selection_preserves_split_and_exact_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _source_row(tmp_path, frames=2_501, frequency=100.0, split="test")
    expanded, report = expand_training_segments([row], base=tmp_path)

    assert len(expanded) == 3
    assert {item["split"] for item in expanded} == {"test"}
    assert [item["motion"] for item in expanded] == [
        segmented_motion_name(row["motion"], index, 3) for index in range(1, 4)
    ]
    assert report["source_motion_count"] == 1
    assert report["segmented_source_count"] == 1
    assert report["output_motion_count"] == 3
    assert report["maximum_output_duration_s"] == pytest.approx(8.34)

    for item in expanded:
        segment = selection_segment(item)
        assert segment is not None
        assert segment.source_motion == row["motion"]
        assert segment.policy == SEGMENT_POLICY

    # Promote this isolated fixture to train only to exercise artifact creation;
    # the production manifest retains the inherited test assignment above.
    train_rows = [item | {"split": "train"} for item in expanded]
    destination = tmp_path / "training-cache"
    source_loads = 0
    original_load = artifacts._load_segment_source_archive

    def counted_load(*args, **kwargs):
        nonlocal source_loads
        source_loads += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_load_segment_source_archive", counted_load)
    payload = materialize_subset(train_rows, destination, terrain_mode="mixed")

    assert source_loads == 1
    assert payload["split_counts"] == {"train": 3, "evaluation": 0, "test": 0}
    source_path = Path(row["source_cache_root"]) / row["trajectory_relpath"]
    with np.load(source_path, allow_pickle=False) as source:
        source_qpos = source["qpos"]
    observed = []
    for materialized, manifest_row in zip(payload["motions"], train_rows, strict=True):
        segment = selection_segment(manifest_row)
        assert segment is not None
        trajectory_path = Path(materialized["paths"]["trajectory_relpath"])
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            np.testing.assert_array_equal(
                trajectory["qpos"],
                source_qpos[segment.start_frame : segment.end_frame_exclusive],
            )
            np.testing.assert_array_equal(
                trajectory["split_points"],
                [0, segment.end_frame_exclusive - segment.start_frame],
            )
            observed.append(trajectory["qpos"])
        validated = validate_retarget_artifacts(destination, manifest_row["motion"])
        assert validated.num_frames <= 1_000
        analysis = load_retarget_analysis(validated.analysis_path)
        assert analysis["segment_start_frame"] == segment.start_frame
        assert analysis["segment_end_frame_exclusive"] == segment.end_frame_exclusive
        assert "source_diagnostic" not in analysis

    np.testing.assert_array_equal(np.concatenate(observed), source_qpos)
    assert not retarget_cache_paths(destination, row["motion"]).trajectory_path.exists()


def test_selection_segment_rejects_tampered_bounds() -> None:
    row = {
        "motion": "Study/Trial/__terra_segment_001_of_003",
        "source_motion": "Study/Trial",
        "segment_start_frame": "1",
        "segment_end_frame_exclusive": "668",
        "segment_index": "1",
        "segment_count": "3",
        "source_num_frames": "2001",
        "source_frequency_hz": "100",
        "segment_policy": SEGMENT_POLICY,
    }

    with pytest.raises(ValueError, match="non-canonical frame bounds"):
        selection_segment(row)


def test_baseline_segments_preserve_method_namespace_and_identity(tmp_path: Path) -> None:
    row = _source_row(tmp_path, frames=2_501, frequency=100.0, method="gmr")
    expanded, report = expand_training_segments([row], base=tmp_path, method="gmr")
    destination = tmp_path / "gmr-training-cache"

    payload = materialize_subset(expanded, destination, method="gmr", terrain_mode="mixed")

    assert report["retargeting_method"] == "gmr"
    assert payload["retargeting_method"] == "gmr"
    assert payload["split_counts"] == {"train": 3, "evaluation": 0, "test": 0}
    for item in payload["motions"]:
        assert "/MyoFullBody/gmr/" in item["paths"]["trajectory_relpath"]
        validated = validate_retarget_artifacts(destination, item["motion"], method="gmr")
        assert validated.method == "gmr"
    assert not retarget_cache_paths(destination, expanded[0]["motion"], method="terra").trajectory_path.exists()


def test_materialization_rejects_incomplete_or_cross_split_segment_groups(tmp_path: Path) -> None:
    row = _source_row(tmp_path, frames=2_501, frequency=100.0)
    expanded, _audit = expand_training_segments([row], base=tmp_path)

    with pytest.raises(ValueError, match="include every interval"):
        materialize_subset(expanded[:-1], tmp_path / "incomplete-cache", terrain_mode="mixed")

    mixed_splits = [expanded[0], expanded[1] | {"split": "test"}, expanded[2]]
    with pytest.raises(ValueError, match="cross source roots or splits"):
        materialize_subset(mixed_splits, tmp_path / "cross-split-cache", terrain_mode="mixed")


def test_custom_segment_bounds_survive_materialization(tmp_path: Path):
    row = _source_row(tmp_path, frames=2_501, frequency=100.0)
    expanded, report = expand_training_segments(
        [row], base=tmp_path, trigger_seconds=12.0, maximum_segment_seconds=6.0,
    )

    assert len(expanded) == 5
    assert report["maximum_segment_seconds"] == 6.0
    assert all(selection_segment(item).maximum_segment_seconds == 6.0 for item in expanded)
    payload = materialize_subset(expanded, tmp_path / "custom-cache", terrain_mode="mixed")
    assert payload["split_counts"]["train"] == 5
