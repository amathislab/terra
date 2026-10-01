"""Write the source Git revision beside generated data and results."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from terra._files import atomic_write

GIT_COMMIT_FILE = "GIT_COMMIT"


def git_commit(repo_root: Path | None = None) -> str:
    """Return the checkout commit, an explicit override, or ``unknown``."""
    supplied = os.environ.get("TERRA_GIT_COMMIT", "").strip()
    if supplied:
        return supplied
    root = Path(__file__).resolve().parents[2] if repo_root is None else repo_root.expanduser().resolve()
    if not (root / ".git").exists():
        return "unknown"
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_git_commit(directory: Path, *, repo_root: Path | None = None) -> Path:
    """Write ``GIT_COMMIT`` in *directory* and return its path."""
    root = directory.expanduser().resolve()
    marker = root / GIT_COMMIT_FILE
    commit = git_commit(repo_root)
    atomic_write(marker, lambda path: path.write_text(f"{commit}\n", encoding="utf-8"))
    return marker


__all__ = ["GIT_COMMIT_FILE", "git_commit", "write_git_commit"]
