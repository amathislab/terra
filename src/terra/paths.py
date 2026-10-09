"""External locations for motion data, generated artifacts, and body models.

Set ``TERRA_DATA_ROOT``, ``TERRA_ARTIFACT_ROOT``, and ``TERRA_MODEL_ROOT`` to
absolute paths for an explicit installation layout.  When they are unset, TERRA
uses the platform's user data/state directories instead of writing into the
source checkout.

This module only reads an environment mapping.  It never writes to
``os.environ``; callers that launch subprocesses remain responsible for building
an explicit child environment.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT_ENV = "TERRA_DATA_ROOT"
ARTIFACT_ROOT_ENV = "TERRA_ARTIFACT_ROOT"
MODEL_ROOT_ENV = "TERRA_MODEL_ROOT"

_ENVIRONMENT_REFERENCE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})")


def _expand_environment(value: str, environment: Mapping[str, str]) -> str:
    """Expand shell-style variables from an explicit, read-only mapping."""

    def replace(match: re.Match[str]) -> str:
        reference = match.group(0)
        name = reference[2:-1] if reference.startswith("${") else reference[1:]
        return environment.get(name, reference)

    expanded = _ENVIRONMENT_REFERENCE.sub(replace, value)
    if _ENVIRONMENT_REFERENCE.search(expanded) is not None:
        raise ValueError(f"path contains an unresolved environment variable: {value}")
    if expanded == "~" or expanded.startswith("~/"):
        home = environment.get("HOME")
        if home:
            return f"{home}{expanded[1:]}"
    return os.path.expanduser(expanded)


def _absolute(value: str | Path, *, base: Path, environment: Mapping[str, str]) -> Path:
    expanded = Path(_expand_environment(str(value), environment))
    return (expanded if expanded.is_absolute() else base / expanded).resolve()


@dataclass(frozen=True)
class StorageRoots:
    """Absolute roots for source data, generated artifacts, and body models.

    :meth:`from_environment` reads ``TERRA_DATA_ROOT``,
    ``TERRA_ARTIFACT_ROOT``, and ``TERRA_MODEL_ROOT`` without changing
    environment variables. Dataset configs may use ``data/``, ``runs/``,
    and ``smpl/`` as aliases for those roots. Unset roots use user data and
    state directories outside the checkout.
    """

    repository_root: Path
    data_root: Path
    artifact_root: Path
    model_root: Path

    @classmethod
    def from_environment(
        cls,
        repository_root: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> StorageRoots:
        """Resolve roots without mutating the process environment.

        Explicit ``TERRA_*_ROOT`` values must be absolute. A copy of the supplied
        mapping is retained so later mutations by the caller do not change path
        expansion behavior.
        """

        repository = Path(repository_root).expanduser().resolve()
        values = dict(os.environ if environment is None else environment)

        home = Path(values.get("HOME", Path.home())).expanduser().resolve()
        data_home = _absolute(
            values.get("XDG_DATA_HOME", home / ".local" / "share"),
            base=home,
            environment=values,
        )
        state_home = _absolute(
            values.get("XDG_STATE_HOME", home / ".local" / "state"),
            base=home,
            environment=values,
        )

        def root(variable: str, default: Path) -> Path:
            value = values.get(variable)
            if value in (None, ""):
                return default.resolve()
            expanded = Path(_expand_environment(value, values)).expanduser()
            if not expanded.is_absolute():
                raise ValueError(f"{variable} must be an absolute path, got {value!r}")
            return expanded.resolve()

        return cls(
            repository_root=repository,
            data_root=root(DATA_ROOT_ENV, data_home / "terra" / "datasets"),
            artifact_root=root(ARTIFACT_ROOT_ENV, state_home / "terra" / "artifacts"),
            model_root=root(MODEL_ROOT_ENV, data_home / "terra" / "models"),
        )

    @property
    def direct_cache_root(self) -> Path:
        """Cache root for direct API retargeting and evaluation calls."""

        return self.artifact_root / "direct" / "cache"

    def _resolve_path(
        self,
        value: str | Path | None,
        *,
        storage_aliases: Mapping[str, Path],
        base: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> Path | None:
        if value in (None, ""):
            return None
        expanded = Path(_expand_environment(str(value), os.environ if environment is None else environment))
        if expanded.is_absolute():
            return expanded.resolve()
        if expanded.parts and expanded.parts[0] in storage_aliases:
            return storage_aliases[expanded.parts[0]].joinpath(*expanded.parts[1:]).resolve()
        return ((base or self.repository_root) / expanded).resolve()

    def resolve_input(
        self,
        value: str | Path | None,
        *,
        base: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> Path | None:
        """Resolve an input path against the configured data or artifact root."""

        return self._resolve_path(
            value,
            storage_aliases={"data": self.data_root, "runs": self.artifact_root},
            base=base,
            environment=environment,
        )

    def resolve_artifact(
        self,
        value: str | Path | None,
        *,
        base: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> Path | None:
        """Resolve an output or cache path against the artifact root."""

        return self._resolve_path(
            value,
            storage_aliases={"runs": self.artifact_root},
            base=base,
            environment=environment,
        )

    def resolve_model(
        self,
        value: str | Path | None,
        *,
        base: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> Path | None:
        """Resolve a body-model path against the model root."""

        return self._resolve_path(
            value,
            storage_aliases={"smpl": self.model_root},
            base=base,
            environment=environment,
        )

    def as_dict(self) -> dict[str, str]:
        """Return the resolved roots as JSON-ready strings."""

        return {
            "repository_root": str(self.repository_root),
            "data_root": str(self.data_root),
            "artifact_root": str(self.artifact_root),
            "model_root": str(self.model_root),
        }


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "DATA_ROOT_ENV",
    "MODEL_ROOT_ENV",
    "StorageRoots",
]
