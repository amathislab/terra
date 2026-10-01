"""Registry for the supported TERRA terrain reconstruction method."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import ReconstructionMethod
from .methods import TerraMethod


@dataclass(frozen=True)
class MethodDefinition:
    name: str
    display_name: str
    evidence_mode: str
    factory: Callable[..., ReconstructionMethod]


def _terra(*, motions: tuple[str, ...], dataset_config: Path | None, options: Mapping[str, Any]) -> ReconstructionMethod:
    if options:
        raise ValueError(f"method 'terra' has unsupported option(s): {', '.join(sorted(options))}")
    if dataset_config is None:
        raise ValueError("terra requires dataset_config")
    return TerraMethod(dataset_config, "full")


RECONSTRUCTION_METHODS: dict[str, MethodDefinition] = {
    "terra": MethodDefinition(
        "terra",
        "TERRA",
        "Terrain reconstruction from kinematic contacts and free-space evidence.",
        _terra,
    ),
}


def available_methods() -> tuple[str, ...]:
    return tuple(RECONSTRUCTION_METHODS)


def create_method(
    name: str,
    *,
    motions: Sequence[str],
    dataset_config: Path | None = None,
    options: Mapping[str, Any] | None = None,
) -> ReconstructionMethod:
    """Instantiate one validated registered method."""

    try:
        definition = RECONSTRUCTION_METHODS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown reconstruction method {name!r}; choose from {', '.join(available_methods())}"
        ) from error
    method = definition.factory(
        motions=tuple(motions),
        dataset_config=dataset_config,
        options={} if options is None else options,
    )
    if method.name != name:
        raise RuntimeError(f"registry factory for {name!r} returned method {method.name!r}")
    return method
