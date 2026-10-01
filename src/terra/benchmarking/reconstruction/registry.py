"""Registry for the four terrain-reconstruction benchmark methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terra.benchmarking.terrain import LeastSquaresPlaneConfig, VoronoiConfig
from terra.dataset_pipeline import load_dataset_config, load_motion_records

from .core import ReconstructionMethod
from .methods import LeastSquaresMethod, TerraMethod, VoronoiMethod


@dataclass(frozen=True)
class MethodDefinition:
    name: str
    display_name: str
    evidence_mode: str
    factory: Callable[..., ReconstructionMethod]


def _unexpected(options: Mapping[str, Any], allowed: set[str], method: str) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"method {method!r} has unsupported option(s): {', '.join(unknown)}")


def _landmark_inputs(dataset_config: Path | None) -> tuple[dict[str, Path], Path | None, Path | None]:
    if dataset_config is None:
        return {}, None, None
    config = load_dataset_config(dataset_config)
    return (
        {record.motion: record.source_path for record in load_motion_records(config)},
        config.smpl_model_path,
        config.cache_root,
    )


def _contact_least_squares(
    *,
    motions: tuple[str, ...],
    dataset_config: Path | None,
    options: Mapping[str, Any],
) -> ReconstructionMethod:
    _unexpected(options, {"env_name", "config"}, "contact-least-squares")
    config = options.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("contact-least-squares.config must be a table")
    source_paths, model_path, cache_root = _landmark_inputs(dataset_config)
    return LeastSquaresMethod(
        config=LeastSquaresPlaneConfig(**dict(config)),
        env_name=str(options.get("env_name", "MyoFullBody")),
        source_paths=source_paths,
        smpl_model_path=model_path,
        cache_root=cache_root,
    )


def _voronoi(
    *,
    motions: tuple[str, ...],
    dataset_config: Path | None,
    options: Mapping[str, Any],
) -> ReconstructionMethod:
    allowed = {
        "env_name",
        "terrain_links",
        "pelvis_link",
        "seat_height_source",
        "link_offsets",
        "config",
    }
    _unexpected(options, allowed, "voronoi")
    raw_config = options.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("voronoi.config must be a table")
    config = {
        "grid_size_m": 0.10,
        "plateau_side_m": 1.0,
        "height_merge_tolerance_m": 0.10,
        "minimum_box_height_m": 0.0001,
    }
    config.update(raw_config)
    seat_height_source = str(options.get("seat_height_source", "shared_posed_posterior_body_surface"))
    default_offsets = {"L_Toe": 0.0, "R_Toe": 0.0}
    if seat_height_source == "fixed_link_offset":
        default_offsets["Pelvis"] = 0.16
    raw_offsets = options.get("link_offsets", default_offsets)
    if not isinstance(raw_offsets, Mapping):
        raise ValueError("voronoi.link_offsets must be a table")
    source_paths, model_path, cache_root = _landmark_inputs(dataset_config)
    return VoronoiMethod(
        motions=motions,
        config=VoronoiConfig(**config),
        env_name=str(options.get("env_name", "MyoFullBody")),
        terrain_links=tuple(options.get("terrain_links", ("L_Toe", "R_Toe", "Pelvis"))),
        pelvis_link=options.get("pelvis_link", "Pelvis"),
        seat_height_source=seat_height_source,
        posed_seat_frame=(load_dataset_config(dataset_config).posed_seat_frame or "normalized")
        if dataset_config is not None
        else "normalized",
        link_offsets={str(key): float(value) for key, value in raw_offsets.items()},
        source_paths=source_paths,
        smpl_model_path=model_path,
        cache_root=cache_root,
    )


def _terra(profile: str, method_id: str) -> Callable[..., ReconstructionMethod]:
    def create(
        *,
        motions: tuple[str, ...],
        dataset_config: Path | None,
        options: Mapping[str, Any],
    ) -> ReconstructionMethod:
        _unexpected(options, set(), method_id)
        if dataset_config is None:
            raise ValueError(f"{method_id} requires dataset_config")
        return TerraMethod(dataset_config, profile)

    return create


RECONSTRUCTION_METHODS: dict[str, MethodDefinition] = {
    "contact-least-squares": MethodDefinition(
        "contact-least-squares",
        "Contact least squares",
        "Kinematic toe and ankle contacts fitted with one affine plane.",
        _contact_least_squares,
    ),
    "voronoi": MethodDefinition(
        "voronoi",
        "Voronoi",
        "Discrete-height Voronoi reconstruction based on TIP and SceneBot.",
        _voronoi,
    ),
    "terra-no-physical-cues": MethodDefinition(
        "terra-no-physical-cues",
        "TERRA w/o physical cues",
        "TERRA with ramp-versus-step selection based only on contact-height residuals.",
        _terra("no-physical-cues", "terra-no-physical-cues"),
    ),
    "terra": MethodDefinition(
        "terra",
        "TERRA",
        "Terrain reconstruction from kinematic contacts and free-space evidence.",
        _terra("full", "terra"),
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
