"""Setuptools build hooks that embed the source Git commit in release artifacts."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from setuptools import build_meta as _setuptools

_ROOT = Path(__file__).resolve().parent
_MARKER = _ROOT / "src" / "terra" / "_build_commit.txt"
_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _valid_commit(value: str) -> str | None:
    normalized = value.strip().lower()
    return normalized if _FULL_GIT_SHA.fullmatch(normalized) is not None else None


def _source_commit() -> str | None:
    supplied = os.environ.get("TERRA_GIT_COMMIT", "")
    if supplied:
        commit = _valid_commit(supplied)
        if commit is None:
            raise ValueError("TERRA_GIT_COMMIT must be a full 40-character lowercase Git hash")
        return commit
    if _MARKER.is_file():
        commit = _valid_commit(_MARKER.read_text(encoding="utf-8"))
        if commit is not None:
            return commit
    archive_marker = _ROOT / ".git_archival.txt"
    if archive_marker.is_file():
        commit = _valid_commit(archive_marker.read_text(encoding="utf-8"))
        if commit is not None:
            return commit
    try:
        value = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return _valid_commit(value)


@contextmanager
def _embedded_commit() -> Iterator[None]:
    previous = _MARKER.read_bytes() if _MARKER.is_file() else None
    commit = _source_commit()
    if commit is not None:
        _MARKER.write_text(f"{commit}\n", encoding="utf-8")
    try:
        yield
    finally:
        if previous is None:
            _MARKER.unlink(missing_ok=True)
        else:
            _MARKER.write_bytes(previous)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel containing the source commit marker."""

    build_root = _ROOT / "build"
    if build_root.is_symlink():
        raise RuntimeError(f"refusing to clean symlinked setuptools build directory: {build_root}")
    if build_root.is_dir():
        shutil.rmtree(build_root)
    with _embedded_commit():
        return _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings=None):
    """Resolve wheel requirements with the generated package-data file present."""

    with _embedded_commit():
        return _setuptools.get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare wheel metadata with the generated package-data file present."""

    with _embedded_commit():
        return _setuptools.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_sdist(sdist_directory, config_settings=None):
    """Build an sdist containing the source commit marker."""

    with _embedded_commit():
        return _setuptools.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_sdist(config_settings=None):
    """Resolve sdist requirements with the generated package-data file present."""

    with _embedded_commit():
        return _setuptools.get_requires_for_build_sdist(config_settings)


def __getattr__(name: str):
    """Delegate the remaining PEP 517 hooks to setuptools."""

    return getattr(_setuptools, name)
