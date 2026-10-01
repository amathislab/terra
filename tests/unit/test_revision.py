from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import terra._revision as revision
from terra._revision import GIT_COMMIT_FILE, git_commit, write_git_commit


def test_write_git_commit_uses_explicit_environment(monkeypatch, tmp_path):
    commit = "a" * 40
    monkeypatch.setenv("TERRA_GIT_COMMIT", commit)

    marker = write_git_commit(tmp_path / "result")

    assert marker == (tmp_path / "result" / GIT_COMMIT_FILE).resolve()
    assert marker.read_text() == f"{commit}\n"


def test_git_commit_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("TERRA_GIT_COMMIT", "abc123")

    with pytest.raises(ValueError, match="full 40-character lowercase Git hash"):
        git_commit(Path.cwd())


def test_packaged_commit_precedes_an_unrelated_working_tree(monkeypatch, tmp_path):
    packaged = "b" * 40
    marker = tmp_path / "_build_commit.txt"
    marker.write_text(f"{packaged}\n")
    monkeypatch.delenv("TERRA_GIT_COMMIT", raising=False)
    monkeypatch.setattr(revision, "PACKAGED_GIT_COMMIT_FILE", marker)
    monkeypatch.setattr(revision, "_git_commit_at", lambda _path: "d" * 40)

    assert git_commit() == packaged


def test_source_archive_inside_an_unrelated_git_checkout_remains_packaged(tmp_path):
    from terra.benchmarking.reconstruction.provenance import _source_state

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    package = tmp_path / "archive/src/terra"
    terrain = package / "terrain"
    terrain.mkdir(parents=True)
    (terrain / "fitting.py").write_text("VALUE = 1\n")
    assert _source_state(None, package) == "packaged"
    assert _source_state(tmp_path / "archive", package) == "packaged"
    assert _source_state(tmp_path, package) == "dirty"


def test_unstamped_source_copy_records_unknown_without_blocking_work(monkeypatch, tmp_path):
    monkeypatch.delenv("TERRA_GIT_COMMIT", raising=False)
    monkeypatch.setattr(revision, "_packaged_git_commit", lambda: None)
    monkeypatch.setattr(revision, "_git_commit_at", lambda _path: None)

    marker = write_git_commit(tmp_path / "result")

    assert marker.read_text() == "unknown\n"
