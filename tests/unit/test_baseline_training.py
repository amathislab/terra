"""Tests for matched baseline policy cohorts and temporal views."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from terra.artifacts import retarget_cache_paths, save_retarget_result, validate_retarget_artifacts
from terra.commands.baseline_training import (
    build_baseline_training_selections,
    publish_baseline_training_selections,
)
from terra.commands.materialize import materialize_subset
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


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cohort_record(
    tmp_path: Path,
    *,
    method: str,
    source_selection: Path,
    cache_root: Path,
    motions: list[str],
    failed: set[str],
    target_fps: float | None = None,
) -> Path:
    root = tmp_path / f"{method}-run"
    selection = root / "selection.csv"
    status = root / "status.csv"
    manifest = root / "manifest.csv"
    _write_csv(selection, ("motion", "dataset"), [{"motion": motion, "dataset": "study"} for motion in motions])
    _write_csv(
        status,
        ("motion", "status"),
        [{"motion": motion, "status": "failed" if motion in failed else "ok"} for motion in motions],
    )
    _write_csv(
        manifest,
        ("motion", "dataset"),
        [{"motion": motion, "dataset": "study"} for motion in motions if motion not in failed],
    )
    record = root / "run.json"
    record.write_text(
        json.dumps(
            {
                "schema": "terra.retarget-cohort-run.v1",
                "method": method,
                "status": "completed_with_failures" if failed else "complete",
                "motions": len(motions),
                "source_selection_sha256": hashlib.sha256(source_selection.read_bytes()).hexdigest(),
                "target_fps": target_fps,
                "partitions": [
                    {
                        "key": "study",
                        "motions": len(motions),
                        "selection": str(selection),
                        "status_table": str(status),
                        "manifest": str(manifest),
                        "cache_root": str(cache_root),
                    }
                ],
            }
        )
    )
    return record


def test_baseline_builder_uses_shared_successes_and_identical_segments(tmp_path: Path) -> None:
    cache_root = tmp_path / "source-cache"
    definitions = [
        ("Study/Long", 2_501, "train"),
        ("Study/Short", 1_000, "test"),
        ("Study/Failed", 500, "train"),
    ]
    source_rows = []
    for motion, frames, split in definitions:
        for method in ("terra", "gmr", "smpl"):
            if method == "smpl" and motion == "Study/Failed":
                continue
            save_retarget_result(
                RetargetResult(
                    trajectory=_Trajectory(frames, 100.0),
                    analysis={},
                    terrain=None,
                    method=method,
                    source_path=tmp_path / f"{motion.replace('/', '-')}.npz",
                ),
                cache_root,
                motion,
            )
        terra_paths = retarget_cache_paths(cache_root, motion)
        source_rows.append(
            {
                "motion": motion,
                "dataset": "study",
                "source_cache_root": str(cache_root),
                "trajectory_relpath": str(terra_paths.trajectory_path.relative_to(cache_root)),
                "analysis_relpath": str(terra_paths.analysis_path.relative_to(cache_root)),
                "terrain_relpath": "",
                "motion_type": "mixed",
                "split": split,
            }
        )
    source_selection = tmp_path / "source-selection.csv"
    _write_csv(source_selection, tuple(source_rows[0]), source_rows)
    motions = [motion for motion, _frames, _split in definitions]
    gmr_record = _cohort_record(
        tmp_path,
        method="gmr",
        source_selection=source_selection,
        cache_root=cache_root,
        motions=motions,
        failed=set(),
    )
    smpl_record = _cohort_record(
        tmp_path,
        method="smpl",
        source_selection=source_selection,
        cache_root=cache_root,
        motions=motions,
        failed={"Study/Failed"},
    )

    rows, method_audits, audit = build_baseline_training_selections(
        source_selection,
        [gmr_record, smpl_record],
        base=tmp_path,
    )

    assert set(rows) == {"terra", "gmr", "smpl"}
    assert audit["source_motion_count"] == 3
    assert audit["shared_source_motion_count"] == 2
    assert audit["excluded_motions"] == ["Study/Failed"]
    assert audit["output_motion_count"] == 4
    assert audit["split_counts"] == {"train": 3, "evaluation": 0, "test": 1}
    identities = {method: [row["motion"] for row in selection] for method, selection in rows.items()}
    assert identities["terra"] == identities["gmr"] == identities["smpl"]
    assert sum(selection_segment(row) is not None for row in rows["gmr"]) == 3
    assert all(row["retargeting_method"] == "gmr" for row in rows["gmr"])
    assert all("/gmr/" in row["trajectory_relpath"] for row in rows["gmr"])
    assert method_audits["gmr"]["segment_policy"] == "over20s-balanced-max10s-v1"

    published = publish_baseline_training_selections(
        source_selection,
        tmp_path / "published",
        rows,
        method_audits,
        audit,
    )
    assert published["identical_motion_and_segment_identities"] is True
    assert Path(published["methods"]["smpl"]["selection"]).is_file()

    materialized = materialize_subset(
        rows["gmr"],
        tmp_path / "materialized-gmr",
        method="gmr",
        terrain_mode="mixed",
    )
    assert materialized["retargeting_method"] == "gmr"
    assert materialized["split_counts"] == {"train": 3, "evaluation": 0, "test": 1}
    for item in materialized["motions"]:
        if item["split"] != "test":
            validate_retarget_artifacts(tmp_path / "materialized-gmr", item["motion"], method="gmr")


def test_baseline_builder_explicitly_supports_provenance_backed_isolated_cache(tmp_path: Path) -> None:
    source_cache = tmp_path / "source-cache"
    isolated_cache = tmp_path / "gmr-rate100-cache"
    motion = "Study/Long"
    for method, cache_root in (("terra", source_cache), ("gmr", isolated_cache)):
        save_retarget_result(
            RetargetResult(
                trajectory=_Trajectory(2_501, 100.0),
                analysis={},
                terrain=None,
                method=method,
                source_path=tmp_path / "source.npz",
            ),
            cache_root,
            motion,
        )
    terra_paths = retarget_cache_paths(source_cache, motion)
    source_selection = tmp_path / "source-selection.csv"
    _write_csv(
        source_selection,
        (
            "motion",
            "dataset",
            "source_cache_root",
            "trajectory_relpath",
            "analysis_relpath",
            "terrain_relpath",
            "motion_type",
            "split",
        ),
        [
            {
                "motion": motion,
                "dataset": "study",
                "source_cache_root": str(source_cache),
                "trajectory_relpath": str(terra_paths.trajectory_path.relative_to(source_cache)),
                "analysis_relpath": str(terra_paths.analysis_path.relative_to(source_cache)),
                "terrain_relpath": "",
                "motion_type": "mixed",
                "split": "train",
            }
        ],
    )
    record = _cohort_record(
        tmp_path,
        method="gmr",
        source_selection=source_selection,
        cache_root=isolated_cache,
        motions=[motion],
        failed=set(),
        target_fps=100.0,
    )

    with pytest.raises(ValueError, match="allow_isolated_cache_roots=True"):
        build_baseline_training_selections(source_selection, [record], base=tmp_path)

    rows, _method_audits, audit = build_baseline_training_selections(
        source_selection,
        [record],
        base=tmp_path,
        allow_isolated_cache_roots=True,
    )

    assert audit["allow_isolated_cache_roots"] is True
    assert audit["method_coverage"]["gmr"]["target_fps"] == 100.0
    assert audit["method_coverage"]["gmr"]["cache_roots"] == [str(isolated_cache)]
    assert all(row["source_cache_root"] == str(isolated_cache) for row in rows["gmr"])
    assert [row["motion"] for row in rows["terra"]] == [row["motion"] for row in rows["gmr"]]
