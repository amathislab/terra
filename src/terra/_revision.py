"""Write the source Git revision beside generated data and results."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from terra._files import atomic_write

GIT_COMMIT_FILE = "GIT_COMMIT"
PACKAGED_GIT_COMMIT_FILE = Path(__file__).with_name("_build_commit.txt")
_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _packaged_git_commit() -> str | None:
    try:
        commit = PACKAGED_GIT_COMMIT_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return commit if _FULL_GIT_SHA.fullmatch(commit) is not None else None


def _git_commit_at(candidate: Path) -> str | None:
    try:
        commit = (
            subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "--verify", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit if _FULL_GIT_SHA.fullmatch(commit) is not None else None


def git_commit(repo_root: Path | None = None) -> str:
    """Return the full commit hash for the TERRA source being executed."""

    supplied = os.environ.get("TERRA_GIT_COMMIT", "").strip().lower()
    if supplied:
        if _FULL_GIT_SHA.fullmatch(supplied) is None:
            raise ValueError("TERRA_GIT_COMMIT must be a full 40-character lowercase Git hash")
        return supplied

    if repo_root is not None:
        commit = _git_commit_at(repo_root.expanduser().resolve())
        if commit is not None:
            return commit
    packaged = _packaged_git_commit()
    if packaged is not None:
        return packaged
    archive_marker = Path(__file__).resolve().parents[2] / ".git_archival.txt"
    if archive_marker.is_file():
        archived = archive_marker.read_text(encoding="utf-8").strip().lower()
        if _FULL_GIT_SHA.fullmatch(archived):
            return archived
    commit = _git_commit_at(Path(__file__).resolve().parents[2])
    if commit is not None:
        return commit
    raise RuntimeError(
        "cannot determine the TERRA Git commit; install an official wheel, run from a Git checkout, "
        "or set TERRA_GIT_COMMIT for a custom source bundle"
    )


def write_git_commit(directory: Path, *, repo_root: Path | None = None) -> Path:
    """Write ``GIT_COMMIT`` in *directory* and return its path."""

    root = directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / GIT_COMMIT_FILE
    commit = git_commit(repo_root)
    atomic_write(marker, lambda path: path.write_text(f"{commit}\n", encoding="utf-8"))
    return marker


__all__ = ["GIT_COMMIT_FILE", "PACKAGED_GIT_COMMIT_FILE", "git_commit", "write_git_commit"]
