"""Tests for provenance-backed training selections over isolated retarget caches."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from terra.artifacts import retarget_cache_paths, save_retarget_result
from terra.commands.retarget_training_selection import (
    build_retarget_training_selection,
    load_retarget_training_selection_spec,
)
from terra.contracts import RetargetResult
from terra.training_segments import selection_segment


class _Trajectory:
    def __init__(self, frames: int, frequency: float) -> None:
        self.frames = frames
        self.frequency = frequency

    def save(self, path: str) -> None:
        qpos = np.arange(self.frames * 8, dtype=np.float32).reshape(self.frames, 8)
        np.savez(
            path,
            qpos=qpos,
            qvel=qpos[:, :7],
            split_points=np.asarray([0, self.frames], dtype=np.int32),
            frequency=np.asarray(self.frequency),
        )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, second_frequency: float = 100.0) -> Path:
    commit = "1" * 40
    shape = b"one-frozen-gmr-shape"
    cache_a = tmp_path / "table2-cache"
    cache_b = tmp_path / "flat-cache"
    definitions = (
        ("Study/Long", 2_501, 100.0, cache_a, "train"),
        ("Study/Short", 801, second_frequency, cache_b, "test"),
    )
    source_rows: list[dict[str, str]] = []
    for motion, frames, frequency, cache, split in definitions:
        save_retarget_result(
            RetargetResult(
                trajectory=_Trajectory(frames, frequency),
                analysis={},
                terrain=None,
                method="gmr",
                source_path=tmp_path / "source.npz",
            ),
            cache,
            motion,
        )
        (cache / "GIT_COMMIT").write_text(commit + "\n", encoding="utf-8")
        fitted_shape = cache / "MyoFullBody/gmr/myofullbody_shape.pkl"
        fitted_shape.parent.mkdir(parents=True, exist_ok=True)
        fitted_shape.write_bytes(shape)
        terra_paths = retarget_cache_paths(tmp_path / "terra-cache", motion)
        source_rows.append(
            {
                "motion": motion,
                "dataset": "study",
                "source_cache_root": str(tmp_path / "terra-cache"),
                "trajectory_relpath": str(terra_paths.trajectory_path.relative_to(tmp_path / "terra-cache")),
                "analysis_relpath": str(terra_paths.analysis_path.relative_to(tmp_path / "terra-cache")),
                "terrain_relpath": "",
                "motion_type": "locomotion",
                "split": split,
            }
        )
    source = tmp_path / "source.csv"
    group_a = tmp_path / "group-a.csv"
    group_b = tmp_path / "group-b.csv"
    _write_csv(source, source_rows)
    _write_csv(group_a, source_rows[:1])
    _write_csv(group_b, source_rows[1:])
    status = tmp_path / "flat-status.csv"
    _write_csv(
        status,
        [
            {
                "motion": "Study/Short",
                "status": "ok",
                "frequency": str(second_frequency),
            }
        ],
    )
    provenance = tmp_path / "upstream-audit.json"
    provenance.write_text(
        json.dumps({"schema": "study.audit.v1", "passed": True}) + "\n",
        encoding="utf-8",
    )
    spec = tmp_path / "selection.toml"
    spec.write_text(
        f'''schema_version = 1
name = "study-gmr-final100"
method = "gmr"
source_selection = "source.csv"
source_selection_sha256 = "{_sha256(source)}"
producer_commit = "{commit}"
producer_image = "registry.example/gmr@sha256:{"2" * 64}"
required_frequency_hz = 100.0
fitted_shape_relpath = "MyoFullBody/gmr/myofullbody_shape.pkl"
fitted_shape_sha256 = "{hashlib.sha256(shape).hexdigest()}"

[[provenance]]
key = "upstream"
path = "upstream-audit.json"
sha256 = "{_sha256(provenance)}"
schema = "study.audit.v1"
require_passed = true

[[artifact_group]]
key = "table2"
selection = "group-a.csv"
selection_sha256 = "{_sha256(group_a)}"
cache_root = "{cache_a}"

[[artifact_group]]
key = "flat"
selection = "group-b.csv"
selection_sha256 = "{_sha256(group_b)}"
cache_root = "{cache_b}"
status_table = "{status}"
''',
        encoding="utf-8",
    )
    return spec


def test_build_retarget_training_selection_validates_provenance_and_actual_frames(tmp_path: Path) -> None:
    spec = load_retarget_training_selection_spec(_fixture(tmp_path))

    rows, audit = build_retarget_training_selection(spec, base=tmp_path)

    assert [row["motion"] for row in rows] == [
        "Study/Long/__terra_segment_001_of_003",
        "Study/Long/__terra_segment_002_of_003",
        "Study/Long/__terra_segment_003_of_003",
        "Study/Short",
    ]
    assert all(row["retargeting_method"] == "gmr" for row in rows)
    assert all("/gmr/" in row["trajectory_relpath"] for row in rows)
    assert selection_segment(rows[0]).source_num_frames == 2_501  # type: ignore[union-attr]
    assert audit["source_motion_count"] == 2
    assert audit["output_motion_count"] == 4
    assert audit["required_frequency_hz"] == 100.0
    assert audit["all_source_artifacts_validated"] is True
    assert audit["artifact_groups"][1]["status"]["all_successful"] is True  # type: ignore[index]
    assert audit["provenance"][0]["passed"] is True  # type: ignore[index]


def test_build_retarget_training_selection_rejects_nonmatching_rate(tmp_path: Path) -> None:
    spec = load_retarget_training_selection_spec(_fixture(tmp_path, second_frequency=50.0))

    with pytest.raises(ValueError, match="has frequency 50"):
        build_retarget_training_selection(spec, base=tmp_path)


def test_build_retarget_training_selection_rejects_phantom_terrain(tmp_path: Path) -> None:
    spec_path = _fixture(tmp_path)
    for selection in (tmp_path / "source.csv", tmp_path / "group-a.csv"):
        rows = list(csv.DictReader(selection.open(newline="")))
        rows[0]["terrain_relpath"] = "MyoFullBody/terra/Study/Long_terrain.json"
        _write_csv(selection, rows)
    text = spec_path.read_text(encoding="utf-8")
    text = text.replace(
        next(line for line in text.splitlines() if line.startswith("source_selection_sha256 =")),
        f'source_selection_sha256 = "{_sha256(tmp_path / "source.csv")}"',
    )
    text = text.replace(
        next(
            line
            for line in text.splitlines()
            if line.startswith("selection_sha256 =")
        ),
        f'selection_sha256 = "{_sha256(tmp_path / "group-a.csv")}"',
        1,
    )
    spec_path.write_text(text, encoding="utf-8")
    spec = load_retarget_training_selection_spec(spec_path)

    with pytest.raises(ValueError, match="declares terrain absent"):
        build_retarget_training_selection(spec, base=tmp_path)


def test_load_retarget_training_selection_spec_requires_exact_group_union(tmp_path: Path) -> None:
    spec_path = _fixture(tmp_path)
    text = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(text.replace('selection = "group-b.csv"', 'selection = "group-a.csv"'), encoding="utf-8")

    with pytest.raises(ValueError, match="selection digest mismatch"):
        load_retarget_training_selection_spec(spec_path)
