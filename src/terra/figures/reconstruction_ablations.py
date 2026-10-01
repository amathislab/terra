"""Render the compact six-panel terrain-reconstruction comparison.

The figure deliberately uses the source-only contact coordinates serialized by the
contact least-squares reconstruction.  Those coordinates are recorded before its plane
fit and are therefore a common observation, not an output of any displayed method.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MethodStyle:
    name: str
    label: str
    description: str
    color: str


METHODS = (
    MethodStyle("contact-least-squares", "Contact LS", "single plane", "#8A8D91"),
    MethodStyle("voronoi", "Voronoi", "piecewise-flat", "#D98200"),
    MethodStyle("terra", "TERRA", "structured primitives", "#008F6A"),
)
DEFAULT_RAMP_CONDITION = "ramp_10_down"
DEFAULT_PRISM_CONDITION = "Stepping boxes"
CONTACT_COLOR = "#17191C"
PRISM_RAISED_CONTACT_THRESHOLD_M = 0.08
PRISM_CONTACT_BIN_M = 0.05


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_path(root: Path, dataset: str, method: str, motion: str) -> Path:
    return root / dataset / method / "terrain" / f"{motion.replace('/', '__')}.json"


def select_ramp_example(
    reconstruction_root: Path,
    *,
    condition: str = DEFAULT_RAMP_CONDITION,
) -> tuple[str, dict[str, Any]]:
    """Select the median-duration ramp within one declared apparatus condition."""

    directory = reconstruction_root / "vielemeyer" / "contact-least-squares" / "terrain"
    candidates: list[tuple[str, int]] = []
    for path in sorted(directory.glob(f"*__{condition}__*.json")):
        record = _load_json(path)
        motion = str(record.get("motion", ""))
        frames = (record.get("fit") or {}).get("n_frames")
        if motion and isinstance(frames, int) and frames > 0:
            candidates.append((motion, frames))
    if not candidates:
        raise ValueError(f"no Vielemeyer {condition!r} reconstruction records found in {directory}")
    target = float(np.median([frames for _, frames in candidates]))
    motion, frames = min(candidates, key=lambda item: (abs(item[1] - target), item[0]))
    return motion, {
        "rule": "nearest median source-frame count within the declared Vielemeyer condition",
        "condition": condition,
        "candidate_count": len(candidates),
        "median_frames": target,
        "selected_frames": frames,
    }


def select_prism_boxes_example(
    reconstruction_root: Path,
    *,
    condition: str = DEFAULT_PRISM_CONDITION,
) -> tuple[str, dict[str, Any]]:
    """Select the PRISM box take nearest its condition's median reference area."""

    path = reconstruction_root / "prism" / "terra" / "evaluation" / "prism_mesh" / "per_motion.json"
    records = _load_json(path)
    candidates: list[tuple[str, float]] = []
    for record in records:
        if record.get("condition_group") != condition:
            continue
        area = ((record.get("mesh_full") or {}).get("gt_raised_area_m2"))
        motion = str(record.get("motion", ""))
        if motion and isinstance(area, (int, float)) and math.isfinite(float(area)):
            candidates.append((motion, float(area)))
    if not candidates:
        raise ValueError(f"no PRISM {condition!r} evaluation rows found in {path}")
    target = float(np.median([area for _, area in candidates]))
    motion, area = min(candidates, key=lambda item: (abs(item[1] - target), item[0]))
    return motion, {
        "rule": "nearest median ground-truth raised footprint area within the declared PRISM condition",
        "condition": condition,
        "candidate_count": len(candidates),
        "median_reference_area_m2": target,
        "selected_reference_area_m2": area,
        "selection_input": str(path),
        "selection_input_sha256": _sha256(path),
    }


def _source_contacts(record: dict[str, Any], *, toes_only: bool = False) -> np.ndarray:
    """Read pre-fit median foot-landmark positions from a contact-LS record."""

    intervals = (record.get("fit") or {}).get("support_intervals")
    if not isinstance(intervals, list):
        raise ValueError("contact least-squares record has no support_intervals list")
    points = []
    for interval in intervals:
        if not isinstance(interval, dict) or interval.get("kind") != "foot":
            continue
        if toes_only and interval.get("link") not in {"L_Toe", "R_Toe"}:
            continue
        point = np.asarray(interval.get("surface_xyz_m"), dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("contact least-squares record contains an invalid foot contact")
        points.append(point)
    if not points:
        raise ValueError("contact least-squares record contains no foot contacts")
    return np.asarray(points)


def _spatially_thin_contacts(points: np.ndarray, spacing: float) -> np.ndarray:
    """Keep one observed contact nearest each occupied XY-bin median."""

    points = np.asarray(points, dtype=float)
    keys = np.floor(points[:, :2] / spacing + 0.5).astype(int)
    selected = []
    for key in sorted(map(tuple, np.unique(keys, axis=0))):
        members = points[np.all(keys == key, axis=1)]
        center = np.median(members, axis=0)
        index = int(np.argmin(np.linalg.norm(members - center, axis=1)))
        selected.append(members[index])
    return np.asarray(selected)


def _box_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Return the serialized BoxSpec world-from-box rotation Rz(yaw) @ Ry(pitch)."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.asarray(
        (
            (cy * cp, -sy, cy * sp),
            (sy * cp, cy, sy * sp),
            (-sp, 0.0, cp),
        )
    )


def _top_face(box: dict[str, Any]) -> np.ndarray:
    center = np.asarray(box["pos"], dtype=float)
    half = np.asarray(box["size"], dtype=float)
    if center.shape != (3,) or half.shape != (3,) or np.any(half <= 0):
        raise ValueError(f"invalid serialized terrain box: {box!r}")
    local = np.asarray(
        (
            (-half[0], -half[1], half[2]),
            (half[0], -half[1], half[2]),
            (half[0], half[1], half[2]),
            (-half[0], half[1], half[2]),
        )
    )
    rotation = _box_rotation(float(box.get("yaw", 0.0)), float(box.get("pitch", 0.0)))
    return local @ rotation.T + center


def _terrain_faces(record: dict[str, Any]) -> list[np.ndarray]:
    terrain = record.get("terrain") or {}
    boxes = terrain.get("boxes") or []
    if not isinstance(boxes, list):
        raise ValueError("terrain boxes must be a list")
    return [_top_face(box) for box in boxes]


def _box_height_at(box: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a serialized BoxSpec with the same vertical ray-slab contract."""

    center = np.asarray(box["pos"], dtype=float)
    half = np.asarray(box["size"], dtype=float)
    yaw = float(box.get("yaw", 0.0))
    pitch = float(box.get("pitch", 0.0))
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = x - center[0], y - center[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    q = cy * dx + sy * dy
    coefficients = (cp * q + sp * center[2], -sy * dx + cy * dy, sp * q - cp * center[2])
    directions = (-sp, 0.0, cp)
    lo = np.full(np.broadcast(x, y).shape, -np.inf)
    hi = np.full(lo.shape, np.inf)
    for coefficient, direction, extent in zip(coefficients, directions, half, strict=True):
        if abs(direction) < 1e-12:
            miss = np.abs(coefficient) > extent
            lo, hi = np.where(miss, np.inf, lo), np.where(miss, -np.inf, hi)
            continue
        first, second = (-extent - coefficient) / direction, (extent - coefficient) / direction
        lo = np.maximum(lo, np.minimum(first, second))
        hi = np.minimum(hi, np.maximum(first, second))
    return np.where(lo <= hi, hi, 0.0)


def _terrain_height_at(record: dict[str, Any], xy: np.ndarray) -> np.ndarray:
    """Query the implicit floor plus every serialized terrain box."""

    xy = np.asarray(xy, dtype=float)
    height = np.zeros(len(xy), dtype=float)
    for box in (record.get("terrain") or {}).get("boxes") or []:
        height = np.maximum(height, _box_height_at(box, xy[:, 0], xy[:, 1]))
    return height


def _presentation_frame(contacts: np.ndarray) -> dict[str, np.ndarray]:
    """Center a row while preserving the source world-XY orientation."""

    center = np.median(contacts[:, :2], axis=0)
    traversal = np.asarray((1.0, 0.0))
    lateral = np.asarray((0.0, 1.0))
    return {"center": center, "traversal": traversal, "lateral": lateral}


def _transform(points: np.ndarray, frame: dict[str, np.ndarray]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    centered = points[:, :2] - frame["center"]
    return np.column_stack(
        (
            centered @ frame["traversal"],
            centered @ frame["lateral"],
            points[:, 2],
        )
    )


def _bounds(
    records: dict[str, dict[str, Any]],
    contacts: np.ndarray,
    frame: dict[str, np.ndarray],
    *,
    contact_focus: bool = False,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    values = [_transform(contacts, frame)]
    for record in records.values():
        values.extend(_transform(face, frame) for face in _terrain_faces(record))
    points = np.concatenate(values)
    lo, hi = points.min(axis=0), points.max(axis=0)
    spans = np.maximum(hi - lo, (0.4, 0.4, 0.15))
    minimum_pad = 0.14 if contact_focus else 0.10
    xy_pad = np.maximum(0.06 * spans[:2], minimum_pad)
    xlim = (float(lo[0] - xy_pad[0]), float(hi[0] + xy_pad[0]))
    ylim = (float(lo[1] - xy_pad[1]), float(hi[1] + xy_pad[1]))
    zlim = (0.0, float(max(0.15, hi[2] + 0.13 * spans[2])))
    return xlim, ylim, zlim


def _darken(color: str, amount: float = 0.72) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return tuple(amount * channel for channel in to_rgb(color))


def _lighten(color: str, amount: float = 0.58) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    rgb = to_rgb(color)
    return tuple(channel + amount * (1.0 - channel) for channel in rgb)


def _surface_color(height: float, style: MethodStyle) -> str | tuple[float, float, float]:
    return style.color if height > PRISM_RAISED_CONTACT_THRESHOLD_M else _lighten(style.color)


def _axis_aligned_surface_faces(
    record: dict[str, Any],
    frame: dict[str, np.ndarray],
    style: MethodStyle,
) -> tuple[list[np.ndarray], list[Any], list[tuple[float, float, float, float]]]:
    """Build an opaque height-field shell without overlapping internal box skirts.

    Voronoi outputs are coalesced into adjacent axis-aligned boxes. Skirting every box
    independently creates intersecting faces that mplot3d cannot reliably depth-sort.
    Tops retain their exact serialized rectangles; vertical walls are emitted only
    where the union height actually changes.
    """

    boxes = (record.get("terrain") or {}).get("boxes") or []
    if not boxes or any(
        abs(float(box.get("yaw", 0.0))) > 1e-10 or abs(float(box.get("pitch", 0.0))) > 1e-10
        for box in boxes
    ):
        return [], [], []

    x_edges = np.unique(
        np.round(
            [float(box["pos"][0]) + sign * float(box["size"][0]) for box in boxes for sign in (-1, 1)],
            decimals=10,
        )
    )
    y_edges = np.unique(
        np.round(
            [float(box["pos"][1]) + sign * float(box["size"][1]) for box in boxes for sign in (-1, 1)],
            decimals=10,
        )
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    sample_x, sample_y = np.meshgrid(x_centers, y_centers)
    samples = np.column_stack((sample_x.ravel(), sample_y.ravel()))
    heights = _terrain_height_at(record, samples).reshape(len(y_centers), len(x_centers))

    faces: list[np.ndarray] = []
    colors: list[Any] = []
    edges: list[tuple[float, float, float, float]] = []

    def append_wall(world: np.ndarray, height: float) -> None:
        faces.append(_transform(world, frame))
        colors.append(_darken(_surface_color(height, style)))
        edges.append((0.0, 0.0, 0.0, 0.0))

    for edge_index, x_value in enumerate(x_edges):
        for y_index, (y_first, y_second) in enumerate(pairwise(y_edges)):
            left = heights[y_index, edge_index - 1] if edge_index else 0.0
            right = heights[y_index, edge_index] if edge_index < len(x_centers) else 0.0
            lower, upper = sorted((float(left), float(right)))
            if upper - lower <= 1e-10:
                continue
            append_wall(
                np.asarray(
                    (
                        (x_value, y_first, lower),
                        (x_value, y_second, lower),
                        (x_value, y_second, upper),
                        (x_value, y_first, upper),
                    )
                ),
                upper,
            )

    for edge_index, y_value in enumerate(y_edges):
        for x_index, (x_first, x_second) in enumerate(pairwise(x_edges)):
            below = heights[edge_index - 1, x_index] if edge_index else 0.0
            above = heights[edge_index, x_index] if edge_index < len(y_centers) else 0.0
            lower, upper = sorted((float(below), float(above)))
            if upper - lower <= 1e-10:
                continue
            append_wall(
                np.asarray(
                    (
                        (x_first, y_value, lower),
                        (x_second, y_value, lower),
                        (x_second, y_value, upper),
                        (x_first, y_value, upper),
                    )
                ),
                upper,
            )

    # Retain the coalesced top rectangles so the Voronoi structure stays visible
    # without introducing artificial grid lines from the wall construction.
    for world_face in _terrain_faces(record):
        top = _transform(world_face, frame)
        faces.append(top)
        colors.append(_surface_color(float(np.max(top[:, 2])), style))
        edges.append((1.0, 1.0, 1.0, 0.72))
    return faces, colors, edges


def _plot_panel(
    axis: Any,
    record: dict[str, Any],
    contacts: np.ndarray,
    frame: dict[str, np.ndarray],
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    style: MethodStyle,
    *,
    zoom: float,
    vertical_exaggeration: float,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xlim, ylim, zlim = limits
    # Contact markers are explanatory overlays and must remain visible even when a
    # reconstructed surface sits slightly above the source landmark. Collections still
    # depth-sort their own faces, but cross-collection order is explicit.
    axis.computed_zorder = False
    floor = np.asarray(
        (
            (xlim[0], ylim[0], 0.0),
            (xlim[1], ylim[0], 0.0),
            (xlim[1], ylim[1], 0.0),
            (xlim[0], ylim[1], 0.0),
        )
    )
    axis.add_collection3d(
        Poly3DCollection([floor], facecolor="#ECEBE7", edgecolor="none", alpha=1.0, zorder=0)
    )

    terrain_faces, terrain_colors, terrain_edges = _axis_aligned_surface_faces(record, frame, style)
    if not terrain_faces:
        for world_face in _terrain_faces(record):
            top = _transform(world_face, frame)
            top_color = _surface_color(float(np.max(top[:, 2])), style)
            for index in range(4):
                first = top[index]
                second = top[(index + 1) % 4]
                terrain_faces.append(
                    np.asarray(
                        (
                            first,
                            second,
                            (second[0], second[1], 0.0),
                            (first[0], first[1], 0.0),
                        )
                    )
                )
                terrain_colors.append(_darken(top_color))
                terrain_edges.append((0.0, 0.0, 0.0, 0.0))
            terrain_faces.append(top)
            terrain_colors.append(top_color)
            terrain_edges.append((1.0, 1.0, 1.0, 0.72))
    if terrain_faces:
        axis.add_collection3d(
            Poly3DCollection(
                terrain_faces,
                facecolor=terrain_colors,
                edgecolor=terrain_edges,
                linewidth=0.28,
                alpha=1.0,
                zorder=2,
            )
        )

    evidence = _transform(contacts, frame)
    surface_height = _terrain_height_at(record, contacts[:, :2])
    surface = _transform(np.column_stack((contacts[:, :2], surface_height)), frame)
    for point, support in zip(evidence, surface, strict=True):
        axis.plot(
            (point[0], support[0]),
            (point[1], support[1]),
            (point[2], support[2]),
            color=CONTACT_COLOR,
            linestyle=(0, (1.0, 1.8)),
            linewidth=0.80,
            alpha=0.78,
            zorder=8,
        )
    marker_size = float(np.clip(100.0 / len(evidence), 3.5, 14.0))
    axis.scatter(
        evidence[:, 0],
        evidence[:, 1],
        evidence[:, 2],
        s=marker_size,
        c=CONTACT_COLOR,
        marker="x",
        linewidths=0.68,
        alpha=1.0,
        depthshade=False,
        zorder=10,
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_zlim(*zlim)
    spans = (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    axis.set_box_aspect(
        (spans[0], spans[1], vertical_exaggeration * spans[2]),
        zoom=zoom,
    )
    axis.view_init(elev=32.0, azim=-58.0)
    axis.set_axis_off()


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_reconstruction_ablations(
    reconstruction_root: str | Path,
    output_stem: str | Path,
    *,
    ramp_motion: str | None = None,
    prism_motion: str | None = None,
    prism_reconstruction_root: str | Path | None = None,
) -> dict[str, Any]:
    """Render PNG/PDF outputs and return their auditable provenance record."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(reconstruction_root).expanduser().resolve()
    prism_root = root if prism_reconstruction_root is None else Path(prism_reconstruction_root).expanduser().resolve()
    destination = Path(output_stem).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]

    if ramp_motion is None:
        ramp_motion, ramp_selection = select_ramp_example(root)
    else:
        ramp_selection = {"rule": "explicit motion override"}
    if prism_motion is None:
        prism_motion, prism_selection = select_prism_boxes_example(prism_root)
    else:
        prism_selection = {"rule": "explicit motion override"}

    rows = (
        ("Ramp", "vielemeyer", ramp_motion, ramp_selection),
        ("PRISM boxes", "prism", prism_motion, prism_selection),
    )
    figure = plt.figure(figsize=(3.45, 2.55), dpi=180)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.995,
        wspace=-0.08,
        hspace=-0.08,
    )
    provenance_rows = []

    for row_index, (label, dataset, motion, selection) in enumerate(rows):
        source_root = prism_root if dataset == "prism" else root
        paths = {method.name: _record_path(source_root, dataset, method.name, motion) for method in METHODS}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing matched reconstruction record(s): {missing}")
        records = {name: _load_json(path) for name, path in paths.items()}
        for name, record in records.items():
            if record.get("motion") != motion or record.get("method") != name:
                raise ValueError(f"reconstruction identity mismatch in {paths[name]}")
        source_contacts = _source_contacts(records["contact-least-squares"])
        toe_contacts = _source_contacts(records["contact-least-squares"], toes_only=True)
        contacts = toe_contacts
        contact_focus = dataset == "prism"
        if contact_focus:
            contacts = toe_contacts[toe_contacts[:, 2] > PRISM_RAISED_CONTACT_THRESHOLD_M]
            if not len(contacts):
                raise ValueError(f"PRISM motion has no raised foot contacts above {PRISM_RAISED_CONTACT_THRESHOLD_M} m")
            contacts = _spatially_thin_contacts(contacts, PRISM_CONTACT_BIN_M)
        frame = _presentation_frame(contacts)
        limits = _bounds(records, contacts, frame, contact_focus=contact_focus)
        panel_zoom = 1.12 if contact_focus else 1.23
        vertical_exaggeration = 4.0 if contact_focus else 2.0

        for column_index, style in enumerate(METHODS):
            axis = figure.add_subplot(grid[row_index, column_index], projection="3d")
            _plot_panel(
                axis,
                records[style.name],
                contacts,
                frame,
                limits,
                style,
                zoom=panel_zoom,
                vertical_exaggeration=vertical_exaggeration,
            )

        try:
            relative_paths = {name: str(path.relative_to(repository)) for name, path in paths.items()}
        except ValueError:
            relative_paths = {name: str(path) for name, path in paths.items()}
        provenance_rows.append(
            {
                "label": label,
                "dataset": dataset,
                "motion": motion,
                "selection": selection,
                "source_contact_count": len(source_contacts),
                "source_toe_contact_count": len(toe_contacts),
                "displayed_contact_count": len(contacts),
                "displayed_contact_filter": (
                    f"toe landmark z > {PRISM_RAISED_CONTACT_THRESHOLD_M} m; one observed point per "
                    f"{PRISM_CONTACT_BIN_M} m XY bin"
                    if contact_focus
                    else "all source toe contacts"
                ),
                "contact_source": "contact-least-squares toe support_intervals[*].surface_xyz_m before plane fitting",
                "records": {
                    name: {"path": relative_paths[name], "sha256": _sha256(path)}
                    for name, path in paths.items()
                },
                "presentation_frame": {
                    "center_world_xy_m": frame["center"].tolist(),
                    "traversal_axis_world_xy": frame["traversal"].tolist(),
                    "lateral_axis_world_xy": frame["lateral"].tolist(),
                },
                "plot_limits_m": {
                    "x": list(limits[0]),
                    "y": list(limits[1]),
                    "z": list(limits[2]),
                },
                "panel_zoom": panel_zoom,
                "vertical_display_exaggeration": vertical_exaggeration,
            }
        )

    png = destination.with_suffix(".png")
    pdf = destination.with_suffix(".pdf")
    figure.savefig(png, dpi=320, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)

    report = {
        "schema": "terra.figure.reconstruction-ablations.v1",
        "source_git_commit": _git_commit(repository),
        "reconstruction_root": str(root),
        "prism_reconstruction_root": str(prism_root),
        "methods": [
            {
                "name": style.name,
                "label": style.label,
                "description": style.description,
                "color": style.color,
            }
            for style in METHODS
        ],
        "rows": provenance_rows,
        "presentation": {
            "coordinate_frame": "per-row centering only; source world XY orientation and world Z retained",
            "camera_elevation_deg": 32.0,
            "camera_azimuth_deg": -58.0,
            "vertical_display_exaggeration": "2x for ramp; 4x for PRISM boxes",
            "surface_geometry": (
                "exact serialized box top faces; axis-aligned outputs use an exposed union shell, "
                "with vertical sides only at footprint boundaries and height changes"
            ),
            "contacts": (
                "all serialized ramp toe intervals and spatially thinned raised PRISM toe intervals; "
                "every marker is an observed source point, not a fitted terrain sample"
            ),
            "contact_marker_vertical_offset_m": 0.0,
            "contact_residual_lines": "vertical dotted segment from each displayed source toe to the queried terrain height",
            "terrain_opacity": 1.0,
            "low_surface_height_threshold_m": PRISM_RAISED_CONTACT_THRESHOLD_M,
            "low_surface_lightening": 0.58,
            "layout": "one-column 2x3 grid",
            "figure_size_inches": [3.45, 2.55],
            "embedded_text": False,
        },
        "outputs": {
            "png": {"path": str(png), "sha256": _sha256(png)},
            "pdf": {"path": str(pdf), "sha256": _sha256(pdf)},
        },
    }
    provenance_path = destination.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_PRISM_CONDITION",
    "DEFAULT_RAMP_CONDITION",
    "METHODS",
    "render_reconstruction_ablations",
    "select_prism_boxes_example",
    "select_ramp_example",
]
