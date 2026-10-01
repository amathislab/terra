"""Small filesystem primitives shared by public workflows."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def staged_write(path: Path, writer: Callable[[Path], object]) -> Iterator[Path]:
    """Write a sibling temporary file and yield it without publishing it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _backup_file(path: Path) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.backup.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    try:
        os.link(path, backup)
    except OSError:
        shutil.copy2(path, backup)
    return backup


def commit_staged_files(operations: Sequence[tuple[Path, Path | None]]) -> None:
    """Commit ordered replacements/removals and restore old files on failure.

    A ``None`` staged path removes its destination. Callers should put their
    metadata file last so readers never accept an incompletely committed set.
    """

    destinations = tuple(destination for destination, _staged in operations)
    if len(set(destinations)) != len(destinations):
        raise ValueError("each staged-file destination must be unique")
    missing_staged = [staged for _destination, staged in operations if staged is not None and not staged.is_file()]
    if missing_staged:
        raise FileNotFoundError(f"staged output does not exist: {missing_staged[0]}")

    backups = {destination: _backup_file(destination) for destination in destinations if destination.is_file()}
    try:
        for destination, staged in operations:
            if staged is None:
                destination.unlink(missing_ok=True)
            else:
                staged.replace(destination)
    except BaseException:
        for destination in reversed(destinations):
            backup = backups.get(destination)
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                backup.replace(destination)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def atomic_write(path: Path, writer: Callable[[Path], object]) -> None:
    """Replace one file only after ``writer`` completes successfully."""

    with staged_write(path, writer) as temporary_path:
        commit_staged_files(((path, temporary_path),))


__all__ = ["atomic_write", "commit_staged_files", "file_sha256", "staged_write"]
