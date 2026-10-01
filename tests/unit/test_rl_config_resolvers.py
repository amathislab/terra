"""Tests for file-backed reinforcement-learning config values."""

from __future__ import annotations

import json

import pytest

from terra.rl.config_resolvers import motion_selection, motion_selection_hash


def test_motion_selection_reads_splits_and_keeps_test_fully_held_out(monkeypatch, tmp_path):
    record = tmp_path / "materialization.json"
    record.write_text(
        json.dumps(
            {
                "motions": [
                    {"motion": "Study/TrainA", "split": "train"},
                    {"motion": "Study/EvalA", "split": "evaluation"},
                    {"motion": "Study/TestA", "split": "test"},
                    {"motion": "Study/TrainB"},
                ]
            }
        )
    )
    monkeypatch.setenv("TERRA_SELECTION_SIZE", "2")
    monkeypatch.setenv("TERRA_VALIDATION_SIZE", "1")
    monkeypatch.setenv("TERRA_TEST_SIZE", "1")

    assert motion_selection("train", str(record)) == ["Study/TrainA", "Study/TrainB"]
    assert motion_selection("validation", str(record)) == ["Study/EvalA"]


def test_motion_selection_validates_launcher_counts(monkeypatch, tmp_path):
    record = tmp_path / "materialization.json"
    record.write_text(json.dumps({"motions": [{"motion": "Study/Train", "split": "train"}]}))
    monkeypatch.setenv("TERRA_SELECTION_SIZE", "2")

    with pytest.raises(ValueError, match="does not match the materialization record count 1"):
        motion_selection("train", str(record))


def test_motion_selection_validation_falls_back_to_training(monkeypatch):
    monkeypatch.delenv("TERRA_VALIDATION_MOTIONS", raising=False)
    monkeypatch.setenv("TERRA_MOTIONS", '["Study/Zeta","Study/Alpha"]')

    assert motion_selection("validation") == ["Study/Zeta", "Study/Alpha"]


def test_motion_selection_hash_is_role_specific_and_ordered(monkeypatch, tmp_path):
    record = tmp_path / "materialization.json"
    record.write_text(
        json.dumps(
            {
                "motions": [
                    {"motion": "Study/A", "split": "train"},
                    {"motion": "Study/B", "split": "train"},
                    {"motion": "Study/D", "split": "evaluation"},
                    {"motion": "Study/C", "split": "evaluation"},
                ]
            }
        )
    )
    monkeypatch.setenv("TERRA_SELECTION_SIZE", "2")
    monkeypatch.setenv("TERRA_VALIDATION_SIZE", "2")
    monkeypatch.setenv("TERRA_TEST_SIZE", "0")

    train_hash = motion_selection_hash("train", str(record))
    validation_hash = motion_selection_hash("validation", str(record))

    assert len(train_hash) == 64
    assert train_hash != validation_hash


def test_motion_selection_hash_separates_retargeting_methods(monkeypatch, tmp_path):
    record = tmp_path / "materialization.json"
    payload = {
        "retargeting_method": "terra",
        "motions": [{"motion": "Study/A", "split": "train"}],
    }
    record.write_text(json.dumps(payload))
    monkeypatch.setenv("TERRA_SELECTION_SIZE", "1")
    monkeypatch.setenv("TERRA_VALIDATION_SIZE", "1")
    monkeypatch.setenv("TERRA_TEST_SIZE", "0")

    terra_hash = motion_selection_hash("train", str(record))
    payload["retargeting_method"] = "gmr"
    record.write_text(json.dumps(payload))

    assert motion_selection_hash("train", str(record)) != terra_hash
