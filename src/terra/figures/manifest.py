"""Versioned, renderer-independent figure manifests."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _table(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _vector(
    value: object,
    name: str,
    length: int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    result = tuple(_number(item, f"{name}[{index}]", minimum=minimum) for index, item in enumerate(value))
    if maximum is not None and any(item > maximum for item in result):
        raise ValueError(f"{name} values must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """Global grid, camera, and raster settings shared by all cells."""

    rows: int
    columns: int
    width: int
    height: int
    cell_size: tuple[float, float]
    cell_pitch: tuple[float, float]
    pad_thickness: float
    decorative_border: int
    camera_azimuth: float
    camera_elevation: float
    camera_distance: float
    camera_lookat_z: float
    field_of_view: float
    camera_lookat_x: float = 0.0
    camera_lookat_y: float = 0.0
    horizon_extent: float = 0.0


@dataclass(frozen=True, slots=True)
class StyleSpec:
    """Scientific color grammar for the shared scene."""

    background: tuple[float, float, float]
    pad: tuple[float, float, float, float]
    terrain: tuple[float, float, float, float]
    history: tuple[float, float, float]
    history_alpha: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LightingSpec:
    """Renderer-neutral studio-lighting controls."""

    ambient: tuple[float, float, float]
    headlight_diffuse: tuple[float, float, float]
    key_direction: tuple[float, float, float]
    key_diffuse: tuple[float, float, float]
    key_specular: tuple[float, float, float]
    fill_direction: tuple[float, float, float]
    fill_diffuse: tuple[float, float, float]
    key_samples: int
    key_spread: float
    shadow_map_size: int
    shadow_clip: float
    ambient_strength: float = 0.18
    direct_light_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One frozen motion-terrain pair placed into a grid slot."""

    row: int
    column: int
    motion: str
    label: str
    frames: tuple[int, ...] | None
    fractions: tuple[float, ...] | None
    yaw_degrees: float | None
    anchor_xy: tuple[float, float] | None
    highlight_frame: int | None = None
    loop_frames: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class FigureManifest:
    """Complete renderer-independent description of one publication figure."""

    version: int
    name: str
    model: str
    method: str
    layout: LayoutSpec
    style: StyleSpec
    lighting: LightingSpec
    cells: tuple[CellSpec, ...]
    path: Path


def _layout(value: object) -> LayoutSpec:
    table = _table(value, "layout")
    rows = _integer(table.get("rows"), "layout.rows", minimum=1)
    columns = _integer(table.get("columns"), "layout.columns", minimum=1)
    return LayoutSpec(
        rows=rows,
        columns=columns,
        width=_integer(table.get("width", 4300), "layout.width", minimum=64),
        height=_integer(table.get("height", 1650), "layout.height", minimum=64),
        cell_size=_vector(table.get("cell_size", [4.2, 2.7]), "layout.cell_size", 2, minimum=0.1),
        cell_pitch=_vector(table.get("cell_pitch", [4.7, 3.3]), "layout.cell_pitch", 2, minimum=0.1),
        pad_thickness=_number(table.get("pad_thickness", 0.06), "layout.pad_thickness", minimum=0.001),
        decorative_border=_integer(table.get("decorative_border", 1), "layout.decorative_border"),
        camera_azimuth=_number(table.get("camera_azimuth", 120.0), "layout.camera_azimuth"),
        camera_elevation=_number(table.get("camera_elevation", -28.0), "layout.camera_elevation"),
        camera_distance=_number(table.get("camera_distance", 25.0), "layout.camera_distance", minimum=0.1),
        camera_lookat_z=_number(table.get("camera_lookat_z", 0.75), "layout.camera_lookat_z"),
        field_of_view=_number(table.get("field_of_view", 35.0), "layout.field_of_view", minimum=1.0),
        camera_lookat_x=_number(table.get("camera_lookat_x", 0.0), "layout.camera_lookat_x"),
        camera_lookat_y=_number(table.get("camera_lookat_y", 0.0), "layout.camera_lookat_y"),
        horizon_extent=_number(table.get("horizon_extent", 0.0), "layout.horizon_extent", minimum=0.0),
    )


def _style(value: object) -> StyleSpec:
    table = _table(value or {}, "style")
    history_alpha_value = table.get("history_alpha", [0.16, 0.28, 0.46])
    if not isinstance(history_alpha_value, list) or not history_alpha_value:
        raise ValueError("style.history_alpha must contain at least one opacity")
    history_alpha = tuple(
        _number(item, f"style.history_alpha[{index}]", minimum=0.0)
        for index, item in enumerate(history_alpha_value)
    )
    if any(item > 1.0 for item in history_alpha):
        raise ValueError("style.history_alpha values must be <= 1")
    return StyleSpec(
        background=_vector(table.get("background", [0.961, 0.953, 0.929]), "style.background", 3, minimum=0, maximum=1),
        pad=_vector(table.get("pad", [0.84, 0.84, 0.81, 1.0]), "style.pad", 4, minimum=0, maximum=1),
        terrain=_vector(table.get("terrain", [0.54, 0.59, 0.62, 1.0]), "style.terrain", 4, minimum=0, maximum=1),
        history=_vector(table.get("history", [0.40, 0.52, 0.62]), "style.history", 3, minimum=0, maximum=1),
        history_alpha=history_alpha,
    )


def _lighting(value: object) -> LightingSpec:
    table = _table(value or {}, "lighting")
    key_direction = _vector(table.get("key_direction", [-0.35, 0.45, -0.82]), "lighting.key_direction", 3)
    fill_direction = _vector(table.get("fill_direction", [0.65, -0.25, -0.72]), "lighting.fill_direction", 3)
    if sum(component * component for component in key_direction) < 1e-8:
        raise ValueError("lighting.key_direction must be non-zero")
    if sum(component * component for component in fill_direction) < 1e-8:
        raise ValueError("lighting.fill_direction must be non-zero")
    key_samples = _integer(table.get("key_samples", 3), "lighting.key_samples", minimum=1)
    if key_samples > 8:
        raise ValueError("lighting.key_samples must be <= 8")
    return LightingSpec(
        ambient=_vector(table.get("ambient", [0.16, 0.16, 0.17]), "lighting.ambient", 3, minimum=0),
        headlight_diffuse=_vector(
            table.get("headlight_diffuse", [0.10, 0.10, 0.11]),
            "lighting.headlight_diffuse",
            3,
            minimum=0,
        ),
        key_direction=key_direction,
        key_diffuse=_vector(table.get("key_diffuse", [0.92, 0.88, 0.80]), "lighting.key_diffuse", 3, minimum=0),
        key_specular=_vector(
            table.get("key_specular", [0.22, 0.20, 0.17]),
            "lighting.key_specular",
            3,
            minimum=0,
        ),
        fill_direction=fill_direction,
        fill_diffuse=_vector(
            table.get("fill_diffuse", [0.20, 0.24, 0.30]),
            "lighting.fill_diffuse",
            3,
            minimum=0,
        ),
        key_samples=key_samples,
        key_spread=_number(table.get("key_spread", 0.08), "lighting.key_spread", minimum=0.0),
        shadow_map_size=_integer(table.get("shadow_map_size", 8192), "lighting.shadow_map_size", minimum=1024),
        shadow_clip=_number(table.get("shadow_clip", 1.05), "lighting.shadow_clip", minimum=0.1),
        ambient_strength=_number(
            table.get("ambient_strength", 0.18),
            "lighting.ambient_strength",
            minimum=0.0,
        ),
        direct_light_scale=_number(
            table.get("direct_light_scale", 1.0),
            "lighting.direct_light_scale",
            minimum=0.0,
        ),
    )


def _cell(value: object, index: int, layout: LayoutSpec) -> CellSpec:
    table = _table(value, f"cells[{index}]")
    row = _integer(table.get("row"), f"cells[{index}].row")
    column = _integer(table.get("column"), f"cells[{index}].column")
    if row >= layout.rows or column >= layout.columns:
        raise ValueError(f"cells[{index}] slot ({row}, {column}) lies outside the declared grid")

    frames_value = table.get("frames")
    fractions_value = table.get("fractions")
    if (frames_value is None) == (fractions_value is None):
        raise ValueError(f"cells[{index}] must define exactly one of frames or fractions")
    frames: tuple[int, ...] | None = None
    fractions: tuple[float, ...] | None = None
    if frames_value is not None:
        if not isinstance(frames_value, list) or not frames_value:
            raise ValueError(f"cells[{index}].frames must be a non-empty array")
        frames = tuple(_integer(item, f"cells[{index}].frames", minimum=0) for item in frames_value)
        if tuple(sorted(set(frames))) != frames:
            raise ValueError(f"cells[{index}].frames must be strictly increasing")
    else:
        if not isinstance(fractions_value, list) or not fractions_value:
            raise ValueError(f"cells[{index}].fractions must be a non-empty array")
        fractions = tuple(
            _number(item, f"cells[{index}].fractions", minimum=0.0) for item in fractions_value
        )
        if any(item > 1.0 for item in fractions) or tuple(sorted(set(fractions))) != fractions:
            raise ValueError(f"cells[{index}].fractions must be strictly increasing values in [0, 1]")

    yaw = table.get("yaw_degrees")
    anchor = table.get("anchor_xy")
    highlight = table.get("highlight_frame")
    loop_value = table.get("loop_frames")
    loop_frames: tuple[int, int] | None = None
    if loop_value is not None:
        if not isinstance(loop_value, list) or len(loop_value) != 2:
            raise ValueError(f"cells[{index}].loop_frames must contain exactly two frames")
        loop_frames = tuple(
            _integer(item, f"cells[{index}].loop_frames", minimum=0)
            for item in loop_value
        )
        if loop_frames[0] >= loop_frames[1]:
            raise ValueError(f"cells[{index}].loop_frames must be strictly increasing")
        if frames is not None and (
            loop_frames[0] > frames[0] or loop_frames[1] < frames[-1]
        ):
            raise ValueError(
                f"cells[{index}].loop_frames must encompass all explicit display frames"
            )
    highlight_frame = (
        None
        if highlight is None
        else _integer(highlight, f"cells[{index}].highlight_frame", minimum=0)
    )
    if highlight_frame is not None:
        if frames is None:
            raise ValueError(f"cells[{index}].highlight_frame requires explicit frames")
        if highlight_frame not in frames:
            raise ValueError(f"cells[{index}].highlight_frame must be included in frames")
    return CellSpec(
        row=row,
        column=column,
        motion=_text(table.get("motion"), f"cells[{index}].motion"),
        label=_text(table.get("label", table.get("motion")), f"cells[{index}].label"),
        frames=frames,
        fractions=fractions,
        yaw_degrees=None if yaw is None else _number(yaw, f"cells[{index}].yaw_degrees"),
        anchor_xy=None if anchor is None else _vector(anchor, f"cells[{index}].anchor_xy", 2),
        highlight_frame=highlight_frame,
        loop_frames=loop_frames,
    )


def load_manifest(path: str | Path) -> FigureManifest:
    """Load and validate a schema-version-1 TOML figure manifest."""

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("rb") as handle:
        raw = tomllib.load(handle)
    version = _integer(raw.get("version"), "version", minimum=1)
    if version != 1:
        raise ValueError(f"unsupported figure manifest version {version}")
    figure = _table(raw.get("figure"), "figure")
    layout = _layout(raw.get("layout"))
    style = _style(raw.get("style", {}))
    lighting = _lighting(raw.get("lighting", {}))
    cell_values = raw.get("cells")
    if not isinstance(cell_values, list) or not cell_values:
        raise ValueError("cells must be a non-empty array of tables")
    cells = tuple(_cell(value, index, layout) for index, value in enumerate(cell_values))
    slots = tuple((cell.row, cell.column) for cell in cells)
    if len(slots) != len(set(slots)):
        raise ValueError("each figure cell must occupy a unique row/column slot")
    return FigureManifest(
        version=version,
        name=_text(figure.get("name"), "figure.name"),
        model=_text(figure.get("model", "MyoFullBody"), "figure.model"),
        method=_text(figure.get("method", "terra"), "figure.method"),
        layout=layout,
        style=style,
        lighting=lighting,
        cells=cells,
        path=manifest_path,
    )


__all__ = ["CellSpec", "FigureManifest", "LayoutSpec", "LightingSpec", "StyleSpec", "load_manifest"]
