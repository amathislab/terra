"""Tests for deterministic, clip-complete training subsets."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from terra.training_segments import SEGMENT_POLICY, segmented_motion_name, temporal_segments
from terra.training_subset import select_stratified_training_subset


def _row(
    motion: str,
    dataset: str,
    motion_type: str,
    split: str = "train",
) -> dict[str, str]:
    return {
        "motion": motion,
        "dataset": dataset,
        "motion_type": motion_type,
        "split": split,
        "source_motion": "",
        "segment_start_frame": "",
        "segment_end_frame_exclusive": "",
        "segment_index": "",
        "segment_count": "",
        "source_num_frames": "",
        "source_frequency_hz": "",
        "segment_policy": "",
    }


def _segmented_rows(source: str, dataset: str, motion_type: str) -> list[dict[str, str]]:
    frame_count = 2_001
    frequency = 100.0
    segments = temporal_segments(frame_count, frequency)
    return [
        _row(segmented_motion_name(source, index, len(segments)), dataset, motion_type)
        | {
            "source_motion": source,
            "segment_start_frame": str(segment.start_frame),
            "segment_end_frame_exclusive": str(segment.end_frame_exclusive),
            "segment_index": str(index),
            "segment_count": str(len(segments)),
            "source_num_frames": str(frame_count),
            "source_frequency_hz": str(frequency),
            "segment_policy": SEGMENT_POLICY,
        }
        for index, segment in enumerate(segments, start=1)
    ]


def _selection() -> list[dict[str, str]]:
    rows = []
    for subject in range(1, 13):
        rows.extend(
            (
                _row(
                    f"Gait120/S{subject:03d}/LevelWalking/Trial01/AllSteps_stageii",
                    "gait120",
                    "flat_locomotion",
                ),
                _row(
                    f"Gait120/S{subject:03d}/StairAscent/Trial01/AllSteps_stageii",
                    "gait120",
                    "stairs_up",
                ),
            )
        )
    rows.extend(_segmented_rows("PRISM/subj001/long_poses", "prism", "platform"))
    rows.extend(_row(f"KIT/{subject}/sit{subject:02d}_poses", "amass", "chair_sit") for subject in range(1, 13))
    rows.extend(
        _row(
            f"Gait120/S{subject:03d}/SlopeDescent/Trial01/AllSteps_stageii",
            "gait120",
            "ramp_down",
            "test",
        )
        for subject in range(20, 23)
    )
    return rows


def test_training_subset_is_exact_deterministic_stratified_and_clip_complete():
    rows = _selection()

    selected, audit = select_stratified_training_subset(rows, target_entries=20, seed="medium-v1")
    repeated, _ = select_stratified_training_subset(rows, target_entries=20, seed="medium-v1")

    assert selected == repeated
    assert Counter(row["split"] for row in selected) == {"train": 20, "test": 3}
    assert audit["selected_train_entries"] == 20
    assert audit["retained_test_entries"] == 3
    assert audit["complete_segment_groups"] is True
    assert {row["motion"] for row in selected if row["split"] == "test"} == {
        row["motion"] for row in rows if row["split"] == "test"
    }

    strata = Counter((row["dataset"], row["motion_type"]) for row in selected if row["split"] == "train")
    assert set(strata) == {
        ("amass", "chair_sit"),
        ("gait120", "flat_locomotion"),
        ("gait120", "stairs_up"),
        ("prism", "platform"),
    }

    selected_segments: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        if row["source_motion"]:
            selected_segments[row["source_motion"]].append(int(row["segment_index"]))
    assert selected_segments in ({}, {"PRISM/subj001/long_poses": [1, 2, 3]})


def test_training_subset_rejects_incomplete_segment_groups():
    rows = _selection()
    rows.pop(next(index for index, row in enumerate(rows) if row["segment_index"] == "2"))

    with pytest.raises(ValueError, match="missing sibling intervals"):
        select_stratified_training_subset(rows, target_entries=20)
