"""Tests for identity-disjoint universal policy-training splits."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from terra.training_split import (
    assign_stratified_splits,
    assign_training_splits,
    canonical_motion_type,
    split_identity,
    split_summary,
)


def _row(motion: str, dataset: str, terrain_class: str) -> dict[str, str]:
    return {
        "motion": motion,
        "dataset": dataset,
        "terrain_class": terrain_class,
    }


def _cohort() -> list[dict[str, str]]:
    rows = []
    for subject in range(1, 21):
        for movement, terrain_class in (
            ("LevelWalking", "flat"),
            ("SlopeAscent", "ramp_up"),
            ("SlopeDescent", "ramp_down"),
            ("StairAscent", "stairs_up"),
            ("StairDescent", "stairs_down"),
            ("SitToStand", "chair_sit"),
            ("StandToSit", "chair_sit"),
        ):
            rows.append(
                _row(
                    f"Gait120/S{subject:03d}/{movement}/Trial01/AllSteps_stageii",
                    "gait120",
                    terrain_class,
                )
            )
    for subject in range(1, 11):
        rows.extend(
            (
                _row(
                    f"PRISM/subj{subject:03d}/take{subject:03d}_poses",
                    "prism",
                    "platform",
                ),
                _row(
                    f"PRISM/subj{subject:03d}/take{subject + 100:03d}_poses",
                    "prism",
                    "stairs_up_down",
                ),
            )
        )
    return rows


def test_gait120_motion_types_are_direction_specific():
    assert (
        canonical_motion_type(
            _row(
                "Gait120/S001/SitToStand/Trial01/AllSteps_stageii",
                "gait120",
                "chair_sit",
            )
        )
        == "chair_sit_to_stand"
    )
    assert (
        canonical_motion_type(
            _row(
                "Gait120/S001/StandToSit/Trial01/AllSteps_stageii",
                "gait120",
                "chair_sit",
            )
        )
        == "chair_stand_to_sit"
    )
    assert split_identity("KIT/12/walk01_poses") == ("AMASS", "KIT/12")
    assert split_identity("CMU/CMU/106/106_07_poses") == (
        "AMASS",
        "CMU/CMU/106",
    )
    assert split_identity("EyesJapanDataset/Eyes_Japan_Dataset/aita/walk_poses") == (
        "AMASS",
        "EyesJapanDataset/Eyes_Japan_Dataset/aita",
    )


def test_split_is_deterministic_identity_disjoint_and_stratified():
    rows = _cohort()
    first = assign_stratified_splits(rows, holdout_fraction=0.2, seed="test-v1")
    repeated = assign_stratified_splits(rows, holdout_fraction=0.2, seed="test-v1")
    alternate = assign_stratified_splits(rows, holdout_fraction=0.2, seed="test-v2")

    assert first == repeated
    assert [row["split"] for row in first] != [row["split"] for row in alternate]
    by_identity = defaultdict(set)
    for row in first:
        by_identity[split_identity(row["motion"])].add(row["split"])
    assert all(len(splits) == 1 for splits in by_identity.values())

    counts = Counter((row["dataset"], row["motion_type"], row["split"]) for row in first)
    for motion_type in {
        "flat_locomotion",
        "ramp_up",
        "ramp_down",
        "stairs_up",
        "stairs_down",
        "chair_sit_to_stand",
        "chair_stand_to_sit",
    }:
        assert counts[("gait120", motion_type, "test")] == 4
        assert counts[("gait120", motion_type, "train")] == 16
    assert split_summary(first)["splits"] == {"test": 32, "train": 128}


def test_split_requires_motion_type_and_two_identities_per_domain():
    with pytest.raises(ValueError, match="motion type is unavailable"):
        assign_stratified_splits([{"motion": "Study/S01/trial", "dataset": "study"}])
    with pytest.raises(ValueError, match="fewer than two identities"):
        assign_stratified_splits([_row("PRISM/subj001/take001", "prism", "platform")])


def test_split_keeps_single_identity_motion_types_in_training():
    rows = [
        _row("KIT/1/beam01_poses", "amass", "beam_balance"),
        _row("KIT/2/walk01_poses", "amass", "flat"),
        _row("KIT/3/walk01_poses", "amass", "flat"),
        _row("KIT/4/walk01_poses", "amass", "flat"),
    ]

    split = assign_stratified_splits(rows, holdout_fraction=0.25, seed="test-v1")

    assert next(row for row in split if row["motion"] == "KIT/1/beam01_poses")["split"] == "train"
    flat_splits = {row["split"] for row in split if row["motion_type"] == "flat_locomotion"}
    assert flat_splits == {"train", "test"}


def test_split_summary_rejects_identity_leakage():
    rows = [
        _row("Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii", "gait120", "ramp_up") | {"split": "train"},
        _row("Gait120/S001/SlopeAscent/Trial02/AllSteps_stageii", "gait120", "ramp_up") | {"split": "evaluation"},
    ]
    with pytest.raises(ValueError, match="identity leakage"):
        split_summary(rows)


def test_evaluation_fraction_creates_a_real_identity_disjoint_validation_split():
    rows = assign_training_splits(
        _cohort(), evaluation_fraction=0.15, test_fraction=0.15, seed="three-way-v1",
    )
    report = split_summary(rows)

    assert set(report["splits"]) == {"train", "evaluation", "test"}
    assert all(report["splits"][name] > 0 for name in report["splits"])
    identities = defaultdict(set)
    for row in rows:
        identities[split_identity(row["motion"])].add(row["split"])
    assert all(len(splits) == 1 for splits in identities.values())
    assert rows == assign_training_splits(
        _cohort(), evaluation_fraction=0.15, test_fraction=0.15, seed="three-way-v1",
    )


def test_select_cli_creates_evaluation_rows(monkeypatch, tmp_path):
    from terra.commands import selection

    observed = {}
    monkeypatch.setattr(selection, "build_selection", lambda *_args, **_kwargs: _cohort())

    def publish(_path, rows, **_kwargs):
        observed["rows"] = rows
        return {"splits": split_summary(rows)["splits"]}

    monkeypatch.setattr(selection, "publish_selection", publish)

    assert selection.main([
        "--run", str(tmp_path / "run"),
        "--out", str(tmp_path / "selection.csv"),
        "--evaluation-fraction", "0.15",
        "--test-fraction", "0.15",
    ]) == 0
    assert set(split_summary(observed["rows"])["splits"]) == {"train", "evaluation", "test"}
