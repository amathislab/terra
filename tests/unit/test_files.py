"""Failure-safety tests for shared filesystem primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from terra._files import commit_staged_files


def test_commit_staged_files_restores_existing_set_on_failure(monkeypatch, tmp_path):
    trajectory = tmp_path / "motion.npz"
    analysis = tmp_path / "motion_analysis.npz"
    staged_trajectory = tmp_path / ".staged-motion.npz"
    staged_analysis = tmp_path / ".staged-analysis.npz"
    trajectory.write_bytes(b"old trajectory")
    analysis.write_bytes(b"old analysis")
    staged_trajectory.write_bytes(b"new trajectory")
    staged_analysis.write_bytes(b"new analysis")

    original_replace = Path.replace

    def fail_analysis_commit(path, target):
        if path == staged_analysis:
            raise OSError("simulated commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_analysis_commit)

    with pytest.raises(OSError, match="commit failure"):
        commit_staged_files(
            (
                (trajectory, staged_trajectory),
                (analysis, staged_analysis),
            )
        )

    assert trajectory.read_bytes() == b"old trajectory"
    assert analysis.read_bytes() == b"old analysis"
