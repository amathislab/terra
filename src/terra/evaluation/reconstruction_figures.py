"""Build reproducible reconstruction-benchmark figures and scene renders.

The command consumes completed evaluator outputs and immutable reconstruction records.
Representative motions are selected by a method-independent ground-truth statistic for
PRISM and by median source-motion duration within each apparatus condition. The selection
rule and exact motion IDs are written beside the figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terra._revision import write_git_commit
from terra.datasets.prism.adapter import load_take, observed_support_points, observed_support_xy
from terra.datasets.prism.mesh_metrics import evaluation_grid, mesh_height_at
from terra.evaluation.reconstruction import (
    _contact_tread_ordinals,
    _expected_geometry,
    _reference_contact_points,
)
from terra.terrain.metadata import TerrainMetadata


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    color: str


METHODS = (
    Method("contact-least-squares", "Contact least squares", "#7f7f7f"),
    Method("voronoi", "Voronoi", "#d55e00"),
    Method("terra-no-physical-cues", "TERRA w/o physical cues", "#0072b2"),
    Method("terra", "TERRA", "#009e73"),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}
GT_COLOR = "#111111"


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _record_path(directory: Path, motion: str) -> Path:
    return directory / f"{motion.replace('/', '__')}.json"


def _terrain(directory: Path, motion: str) -> Any:
    record = _json(_record_path(directory, motion))
    return TerrainMetadata.from_dict(record.get("terrain") or {"boxes": []}).terrain


def _evaluation_records(root: Path, method: str) -> list[dict[str, Any]]:
    path = root / method / "evaluation" / "prism_mesh" / "per_motion.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = _json(path)
    if not isinstance(records, list):
        raise ValueError(f"PRISM per-motion evaluation must be a list: {path}")
    return records


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(record: Mapping[str, Any], *path: str) -> float | None:
    value: Any = record
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return _finite(value)


def select_prism_representatives(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Choose the take nearest the median GT raised area in each geometry class."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["condition_group"]), []).append(record)
    selected = {}
    for condition, rows in grouped.items():
        values = np.asarray([_metric(row, "mesh_full", "gt_raised_area_m2") for row in rows], dtype=float)
        target = float(np.median(values))
        choice = min(
            zip(rows, values, strict=True),
            key=lambda item: (abs(float(item[1]) - target), str(item[0]["motion"])),
        )[0]
        selected[condition] = str(choice["motion"])
    return dict(sorted(selected.items()))


def _terrain_arrays(
    take: Mapping[str, Any],
    terrains: Mapping[str, Any],
    *,
    resolution: float = 0.025,
    margin: float = 0.15,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, dict[str, np.ndarray]]:
    objects = take["objects"]
    domain = observed_support_xy(take, stride=1)
    arbitrary = next(iter(terrains.values()))
    xy, shape = evaluation_grid(
        objects,
        arbitrary,
        resolution=resolution,
        margin=margin,
        domain_xy=domain,
    )
    gt = mesh_height_at(xy, objects)
    predicted = {
        name: np.asarray(terrain.height_at(xy[:, 0], xy[:, 1]), dtype=float) for name, terrain in terrains.items()
    }
    return xy, shape, gt, predicted


def _extent(xy: np.ndarray) -> tuple[float, float, float, float]:
    x = np.unique(xy[:, 0])
    y = np.unique(xy[:, 1])
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    return float(x.min() - dx / 2), float(x.max() + dx / 2), float(y.min() - dy / 2), float(y.max() + dy / 2)


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def render_prism_heightfields(
    prism_root: Path,
    data_root: Path,
    representatives: Mapping[str, str],
    output: Path,
) -> None:
    """Render GT and every fitted method on shared per-take grids."""

    plt = _mpl()
    columns = (Method("ground-truth", "PRISM ground truth", GT_COLOR), *METHODS)
    figure, axes = plt.subplots(len(representatives), len(columns), figsize=(16.5, 8.0), squeeze=False)
    image = None
    for row_index, (condition, motion) in enumerate(representatives.items()):
        _, subject, take_name = motion.split("/")
        take = load_take(data_root / subject / f"{take_name.removesuffix('_poses')}.pkl")
        terrains = {
            method.name: _terrain(prism_root / method.name / "terrain", motion) for method in METHODS
        }
        xy, shape, gt, predicted = _terrain_arrays(take, terrains)
        extent = _extent(xy)
        maximum = max(float(gt.max()), *(float(values.max()) for values in predicted.values()), 0.25)
        support = observed_support_xy(take, stride=40)
        for column_index, method in enumerate(columns):
            axis = axes[row_index, column_index]
            values = gt if method.name == "ground-truth" else predicted[method.name]
            image = axis.imshow(
                values.reshape(shape),
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=maximum,
                interpolation="nearest",
            )
            axis.contour(
                xy[:, 0].reshape(shape),
                xy[:, 1].reshape(shape),
                (gt > 0.02).reshape(shape),
                levels=[0.5],
                colors="white",
                linewidths=0.8,
            )
            if len(support):
                axis.scatter(support[:, 0], support[:, 1], s=2, c="black", alpha=0.25, linewidths=0)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(method.label)
            if column_index == 0:
                axis.set_ylabel(f"{condition}\n{motion.split('/')[1]}")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.70, pad=0.01)
    colorbar.set_label("support height (m)")
    figure.suptitle("PRISM terrain support envelopes — white contour is the reference object footprint", y=0.995)
    _save_figure(figure, output / "prism_heightfields.png")
    plt.close(figure)


def render_prism_errors(
    prism_root: Path,
    data_root: Path,
    evaluation_root: Path,
    representatives: Mapping[str, str],
    output: Path,
) -> None:
    """Render signed height errors and reference-mesh per-take metrics."""

    plt = _mpl()
    score_rows = {
        method.name: {str(row["motion"]): row for row in _evaluation_records(evaluation_root, method.name)}
        for method in METHODS
    }
    figure, axes = plt.subplots(len(representatives), len(METHODS), figsize=(14.0, 8.0), squeeze=False)
    image = None
    for row_index, (condition, motion) in enumerate(representatives.items()):
        _, subject, take_name = motion.split("/")
        take = load_take(data_root / subject / f"{take_name.removesuffix('_poses')}.pkl")
        terrains = {
            method.name: _terrain(prism_root / method.name / "terrain", motion) for method in METHODS
        }
        xy, shape, gt, predicted = _terrain_arrays(take, terrains)
        extent = _extent(xy)
        gt_raised = gt > 0.02
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            pred = predicted[method.name]
            union = gt_raised | (pred > 0.02)
            error_cm = np.ma.masked_where(~union.reshape(shape), 100.0 * (pred - gt).reshape(shape))
            image = axis.imshow(
                error_cm,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-20.0,
                vmax=20.0,
                interpolation="nearest",
            )
            axis.contour(
                xy[:, 0].reshape(shape),
                xy[:, 1].reshape(shape),
                gt_raised.reshape(shape),
                levels=[0.5],
                colors="black",
                linewidths=0.8,
            )
            score = score_rows[method.name][motion]
            mae = _metric(score, "mesh_at_observed_support", "height_mae_m")
            iou = _metric(score, "mesh_full", "raised_footprint_iou")
            axis.text(
                0.02,
                0.02,
                f"support MAE {1000 * float(mae):.1f} mm\nfootprint IoU {float(iou):.3f}",
                transform=axis.transAxes,
                fontsize=7,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(method.label)
            if column_index == 0:
                axis.set_ylabel(condition)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.70, pad=0.01)
    colorbar.set_label("predicted - ground-truth height (cm); clipped at ±20 cm")
    figure.suptitle("PRISM reconstruction error — black contour is the reference object footprint", y=0.995)
    _save_figure(figure, output / "prism_height_errors.png")
    plt.close(figure)


def _representative_frame(take: Mapping[str, Any]) -> int:
    """Choose a frame with the strongest reference-mesh terrain interaction."""

    names = ("Pelvis", "L_Foot", "R_Foot")
    positions = {
        name: np.asarray(take["imu_gt"][name]["pos_world"], dtype=float) for name in names
    }
    frames = min(len(value) for value in positions.values())
    sampled = np.arange(0, frames, max(1, frames // 1000), dtype=int)
    support_xy = np.concatenate([positions[name][sampled, :2] for name in names])
    support_height = mesh_height_at(support_xy, take["objects"]).reshape(len(names), -1)
    return int(sampled[int(np.argmax(support_height.max(axis=0)))])


def render_prism_perspectives(
    prism_root: Path,
    data_root: Path,
    representatives: Mapping[str, str],
    output: Path,
) -> None:
    """Render reference meshes, Voronoi, and TERRA from a shared 3-D viewpoint."""

    plt = _mpl()
    panels = (
        Method("ground-truth", "PRISM ground truth", GT_COLOR),
        METHOD_BY_NAME["voronoi"],
        METHOD_BY_NAME["terra"],
    )
    figure = plt.figure(figsize=(12.5, 10.0))
    links = (
        ("Pelvis", "Head"),
        ("Pelvis", "L_Knee"),
        ("L_Knee", "L_Foot"),
        ("Pelvis", "R_Knee"),
        ("R_Knee", "R_Foot"),
        ("Head", "L_Wrist"),
        ("Head", "R_Wrist"),
    )
    landmark_names = sorted({name for link in links for name in link})
    for row_index, (condition, motion) in enumerate(representatives.items()):
        _, subject, take_name = motion.split("/")
        take = load_take(data_root / subject / f"{take_name.removesuffix('_poses')}.pkl")
        terrains = {
            method.name: _terrain(prism_root / method.name / "terrain", motion)
            for method in panels
            if method.name != "ground-truth"
        }
        xy, shape, gt, predicted = _terrain_arrays(take, terrains, resolution=0.04)
        x = xy[:, 0].reshape(shape)
        y = xy[:, 1].reshape(shape)
        frame = _representative_frame(take)
        landmarks = {
            name: np.asarray(take["imu_gt"][name]["pos_world"], dtype=float)[frame]
            for name in landmark_names
        }
        maximum = max(float(gt.max()), *(float(values.max()) for values in predicted.values()), 0.25)
        for column_index, panel in enumerate(panels):
            axis = figure.add_subplot(
                len(representatives),
                len(panels),
                row_index * len(panels) + column_index + 1,
                projection="3d",
            )
            values = gt if panel.name == "ground-truth" else predicted[panel.name]
            surface_color = "#339ca3" if panel.name == "ground-truth" else panel.color
            axis.plot_surface(
                x,
                y,
                values.reshape(shape),
                color=surface_color,
                linewidth=0,
                antialiased=False,
                alpha=0.78,
            )
            if panel.name != "ground-truth":
                axis.contour(
                    x,
                    y,
                    (gt > 0.02).reshape(shape),
                    levels=[0.5],
                    zdir="z",
                    offset=0.0,
                    colors="black",
                    linewidths=1.0,
                )
            for first, second in links:
                points = np.vstack((landmarks[first], landmarks[second]))
                axis.plot(points[:, 0], points[:, 1], points[:, 2], color="#202228", lw=1.7)
            axis.scatter(
                [landmarks[name][0] for name in landmark_names],
                [landmarks[name][1] for name in landmark_names],
                [landmarks[name][2] for name in landmark_names],
                color="#202228",
                s=8,
                depthshade=False,
            )
            axis.set_xlim(float(x.min()), float(x.max()))
            axis.set_ylim(float(y.min()), float(y.max()))
            axis.set_zlim(0.0, max(float(landmarks["Head"][2]) + 0.1, maximum + 0.2))
            axis.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), 1.5))
            axis.view_init(elev=27, azim=-58)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([0.0, round(maximum, 2)])
            axis.set_zlabel("height (m)", labelpad=-3)
            if row_index == 0:
                axis.set_title(panel.label)
            if column_index == 0:
                axis.text2D(
                    -0.10,
                    0.50,
                    f"{condition}\n{subject}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                )
    figure.suptitle("PRISM scene reconstructions with source motion pose", y=0.99)
    figure.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.95, wspace=0.02, hspace=0.06)
    _save_figure(figure, output / "prism_perspective_scenes.png")
    plt.close(figure)


def render_prism_metric_distributions(evaluation_root: Path, output: Path) -> None:
    """Show paired per-take distributions for the benchmark's primary metrics."""

    plt = _mpl()
    metric_specs = (
        (("mesh_at_observed_support", "height_mae_m"), "Observed support MAE (mm)", 1000.0),
        (("mesh_full", "raised_footprint_iou"), "Raised footprint IoU", 1.0),
        (("mesh_full", "raised_terrain_coverage"), "Raised-terrain coverage", 1.0),
        (("mesh_full", "flat_terrain_coverage"), "Flat-terrain coverage", 1.0),
        (("mesh_full", "support_f1_50mm"), "Support F1 @ 50 mm", 1.0),
    )
    all_rows = {method.name: _evaluation_records(evaluation_root, method.name) for method in METHODS}
    figure, axes = plt.subplots(1, len(metric_specs), figsize=(16.5, 3.8), squeeze=False)
    for axis, (path, label, scale) in zip(axes[0], metric_specs, strict=True):
        values = []
        for method in METHODS:
            method_values = [
                scale * value
                for row in all_rows[method.name]
                if (value := _metric(row, *path)) is not None
            ]
            values.append(method_values)
        plot = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
        for box, method in zip(plot["boxes"], METHODS, strict=True):
            box.set(facecolor=method.color, alpha=0.68, edgecolor=method.color)
        rng = np.random.default_rng(0)
        for index, (method, method_values) in enumerate(zip(METHODS, values, strict=True), start=1):
            jitter = rng.uniform(-0.11, 0.11, len(method_values))
            axis.scatter(index + jitter, method_values, s=8, color=method.color, alpha=0.42, linewidths=0)
        axis.set_ylabel(label)
        axis.set_xticks(range(1, len(METHODS) + 1), [method.label for method in METHODS], rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.20)
    figure.suptitle("PRISM paired 31-take reconstruction benchmark", y=1.02)
    _save_figure(figure, output / "prism_metric_distributions.png")
    plt.close(figure)


def _apparatus_group(dataset: str, motion: str, condition: str) -> str:
    parts = motion.split("/")
    if dataset in {"gait120", "darmstadt", "vielemeyer"} and len(parts) > 2:
        return parts[2]
    return condition


def select_apparatus_representatives(
    dataset: str,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    """Choose the motion nearest the median source duration within each condition."""

    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        motion = str(row["motion"])
        expected_family = str(row.get("expected_family", "")).strip().casefold()
        if expected_family not in {"ramp", "steps"}:
            continue
        group = _apparatus_group(dataset, motion, str(row.get("condition", "")))
        value = _finite(row.get("frames"))
        if value is None:
            value = _finite(row.get("source_frames"))
        if value is not None:
            grouped.setdefault(group, []).append((motion, value))
    selected = {}
    for group, values in grouped.items():
        target = float(np.median([value for _, value in values]))
        selected[group] = min(values, key=lambda item: (abs(item[1] - target), item[0]))[0]
    return dict(sorted(selected.items()))


def _principal_coordinate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    coordinate = centered @ axis
    return coordinate, center, axis


def _apparatus_panel(
    axis: Any,
    dataset: str,
    motion: str,
    apparatus_root: Path,
    row: Mapping[str, str],
) -> None:
    reference = _json(_record_path(apparatus_root / "contact-least-squares" / "terrain", motion))
    contacts = _reference_contact_points(reference)
    xy = np.stack([contact.xyz[:2] for contact in contacts])
    coordinate, center, direction = _principal_coordinate(xy)
    slope, riser, _ = _expected_geometry(dict(row))
    if riser is not None:
        ordinals = _contact_tread_ordinals(contacts, riser)
        expected = np.asarray(ordinals, dtype=float) * riser
        order = np.argsort(coordinate, kind="stable")
        axis.scatter(
            coordinate[order],
            1000.0 * expected[order],
            color=GT_COLOR,
            marker="x",
            s=22,
            label="Ground truth",
        )
        for method in METHODS:
            terrain = _terrain(apparatus_root / method.name / "terrain", motion)
            height = np.asarray(terrain.height_at(xy[:, 0], xy[:, 1]), dtype=float)
            axis.plot(coordinate[order], 1000.0 * height[order], "o", ms=2.8, color=method.color, alpha=0.78)
        axis.set_ylabel("support height (mm)")
        return

    if slope is None:
        raise ValueError(f"motion has neither slope nor riser reference: {motion}")
    source_z = np.asarray([contact.xyz[2] for contact in contacts])
    if np.corrcoef(coordinate, source_z)[0, 1] < 0:
        direction = -direction
        coordinate = -coordinate
    samples = np.linspace(float(coordinate.min()), float(coordinate.max()), 160)
    query = center + samples[:, None] * direction
    expected = np.tan(np.radians(slope)) * (samples - samples.min())
    axis.plot(samples, 1000.0 * expected, color=GT_COLOR, lw=2.0, label=f"Nominal {slope:g}°")
    for method in METHODS:
        terrain = _terrain(apparatus_root / method.name / "terrain", motion)
        height = np.asarray(terrain.height_at(query[:, 0], query[:, 1]), dtype=float)
        height -= float(height.min())
        axis.plot(samples, 1000.0 * height, color=method.color, lw=1.15, alpha=0.90)
    axis.set_ylabel("height above profile minimum (mm)")


def render_apparatus_profiles(
    apparatus_root: Path,
    manifests: Mapping[str, Path],
    output: Path,
) -> dict[str, dict[str, str]]:
    """Render representative known-apparatus profiles for all three datasets."""

    plt = _mpl()
    selections: dict[str, dict[str, str]] = {}
    for dataset in ("gait120", "darmstadt", "vielemeyer"):
        terrain_root = apparatus_root / dataset
        with manifests[dataset].open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        manifest_rows = {row["motion"]: row for row in source_rows}
        selected = select_apparatus_representatives(dataset, source_rows)
        selections[dataset] = selected
        count = len(selected)
        columns = 2 if count <= 4 else 3
        rows_count = math.ceil(count / columns)
        figure, axes = plt.subplots(rows_count, columns, figsize=(5.2 * columns, 3.7 * rows_count), squeeze=False)
        for axis, (group, motion) in zip(axes.ravel(), selected.items(), strict=False):
            _apparatus_panel(axis, dataset, motion, terrain_root, manifest_rows[motion])
            axis.set_title(f"{group} — {motion.split('/')[1]}")
            axis.set_xlabel("distance along traversal (m)")
            axis.grid(alpha=0.20)
        for axis in axes.ravel()[count:]:
            axis.set_visible(False)
        handles = [plt.Line2D([], [], color=GT_COLOR, lw=2, label="Ground truth")]
        handles.extend(plt.Line2D([], [], color=method.color, lw=2, label=method.label) for method in METHODS)
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
        figure.suptitle(f"{dataset.capitalize()} representative terrain reconstructions", y=0.98)
        figure.subplots_adjust(
            bottom=0.19 if rows_count == 1 else 0.13,
            top=0.88,
            hspace=0.52,
            wspace=0.24,
        )
        _save_figure(figure, output / f"{dataset}_representative_profiles.png")
        plt.close(figure)
    return selections


def _convex_hull(points: np.ndarray) -> np.ndarray:
    values = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(values) <= 2:
        return np.asarray(values)

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for chain, sequence in ((lower, values), (upper, reversed(values))):
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _box_polygon(box: Any) -> np.ndarray:
    local = np.asarray(
        [
            [-box.size[0], -box.size[1]],
            [box.size[0], -box.size[1]],
            [box.size[0], box.size[1]],
            [-box.size[0], box.size[1]],
        ],
        dtype=float,
    )
    return local @ np.asarray(box.rotation, dtype=float)[:2, :2].T + np.asarray(box.pos[:2])


def _render_video(
    motion: str,
    prism_root: Path,
    evaluation_root: Path,
    data_root: Path,
    destination: Path,
    *,
    stride: int,
    fps: int,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    _, subject, take_name = motion.split("/")
    take = load_take(data_root / subject / f"{take_name.removesuffix('_poses')}.pkl")
    terrains = {method.name: _terrain(prism_root / method.name / "terrain", motion) for method in METHODS}
    scores = {
        method.name: next(row for row in _evaluation_records(evaluation_root, method.name) if row["motion"] == motion)
        for method in METHODS
    }
    objects = take["objects"]
    landmarks = {
        name: np.asarray(take["imu_gt"][name]["pos_world"], dtype=float)
        for name in ("Pelvis", "Head", "L_Knee", "R_Knee", "L_Foot", "R_Foot", "L_Wrist", "R_Wrist")
    }
    n_frames = min(len(value) for value in landmarks.values())
    gt_polygons = []
    for obj in objects.values():
        vertices = np.asarray(obj["vertices"], dtype=float).reshape(-1, 3)
        gt_polygons.append((_convex_hull(vertices[:, :2]), float(vertices[:, 2].max())))
    predicted = {
        name: [(_box_polygon(box), float(np.asarray(box.pos)[2] + np.asarray(box.size)[2])) for box in terrain.boxes]
        for name, terrain in terrains.items()
    }
    bounds = [polygon for polygon, _ in gt_polygons]
    bounds.extend(polygon for values in predicted.values() for polygon, _ in values)
    bounds.extend(value[:: max(1, len(value) // 1000), :2] for value in landmarks.values())
    all_xy = np.concatenate(bounds)
    lo, hi = all_xy.min(axis=0) - 0.18, all_xy.max(axis=0) + 0.18

    canvas = (1600, 900)
    panels = (Method("ground-truth", "PRISM ground truth", GT_COLOR), *METHODS)
    cell_width, cell_height = 520, 385
    origins = [(20 + column * cell_width, 80 + row * cell_height) for row in range(2) for column in range(3)]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = small = title_font = ImageFont.load_default()

    def project(points: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
        span = np.maximum(hi - lo, 1e-6)
        scale = min((cell_width - 24) / span[0], (cell_height - 58) / span[1])
        pad = np.asarray([cell_width - 24, cell_height - 58]) - span * scale
        q = (np.asarray(points) - lo) * scale
        return np.column_stack(
            (
                origin[0] + 12 + pad[0] / 2 + q[:, 0],
                origin[1] + cell_height - 12 - pad[1] / 2 - q[:, 1],
            )
        )

    def tuples(points: np.ndarray) -> list[tuple[int, int]]:
        return [(round(float(x)), round(float(y))) for x, y in points]

    import imageio_ffmpeg

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.staging-{os.getpid()}.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{canvas[0]}x{canvas[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    links = (
        ("Pelvis", "Head"),
        ("Pelvis", "L_Knee"),
        ("L_Knee", "L_Foot"),
        ("Pelvis", "R_Knee"),
        ("R_Knee", "R_Foot"),
        ("Head", "L_Wrist"),
        ("Head", "R_Wrist"),
    )
    rendered = 0
    try:
        assert process.stdin is not None
        for frame in range(0, n_frames, stride):
            image = Image.new("RGB", canvas, (248, 248, 246))
            draw = ImageDraw.Draw(image, "RGBA")
            draw.text(
                (24, 18),
                f"PRISM reconstruction comparison — {motion}",
                font=title_font,
                fill=(20, 24, 30, 255),
            )
            draw.text(
                (1220, 24),
                f"t = {frame / float(take['info']['data_info']['fps']):.1f} s",
                font=font,
                fill=(60, 65, 72, 255),
            )
            for panel, origin in zip(panels, origins, strict=True):
                draw.rectangle(
                    (origin[0], origin[1], origin[0] + cell_width - 10, origin[1] + cell_height - 10),
                    fill=(255, 255, 255, 255),
                    outline=(210, 212, 214, 255),
                    width=1,
                )
                for polygon, _height in gt_polygons:
                    points = tuples(project(polygon, origin))
                    if panel.name == "ground-truth":
                        draw.polygon(points, fill=(34, 150, 155, 135), outline=(0, 91, 96, 255))
                    else:
                        draw.line([*points, points[0]], fill=(0, 91, 96, 220), width=2)
                if panel.name != "ground-truth":
                    for polygon, _height in predicted[panel.name]:
                        points = tuples(project(polygon, origin))
                        draw.polygon(points, fill=(230, 132, 38, 90), outline=(205, 92, 8, 230))
                for first, second in links:
                    points = project(np.vstack((landmarks[first][frame, :2], landmarks[second][frame, :2])), origin)
                    draw.line(tuples(points), fill=(28, 30, 36, 230), width=3)
                for name in ("Pelvis", "Head", "L_Foot", "R_Foot"):
                    x, y = project(landmarks[name][frame, :2][None], origin)[0]
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(20, 20, 22, 255))
                draw.text((origin[0] + 12, origin[1] + 10), panel.label, font=font, fill=(20, 24, 30, 255))
                if panel.name != "ground-truth":
                    mae = 1000.0 * float(_metric(scores[panel.name], "mesh_at_observed_support", "height_mae_m"))
                    iou = float(_metric(scores[panel.name], "mesh_full", "raised_footprint_iou"))
                    draw.text(
                        (origin[0] + 12, origin[1] + 34),
                        f"support MAE {mae:.1f} mm   IoU {iou:.3f}",
                        font=small,
                        fill=(72, 76, 82, 255),
                    )
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            rendered += 1
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        os.replace(temporary, destination)
    finally:
        if process.poll() is None:
            process.kill()
        temporary.unlink(missing_ok=True)
    return {"motion": motion, "frames": n_frames, "rendered_frames": rendered, "output": str(destination)}


def render_prism_videos(
    prism_root: Path,
    evaluation_root: Path,
    data_root: Path,
    representatives: Mapping[str, str],
    output: Path,
    *,
    stride: int,
    fps: int,
) -> list[dict[str, Any]]:
    results = []
    for condition, motion in representatives.items():
        destination = output / f"prism_{condition.casefold().replace(' ', '_')}_comparison.mp4"
        results.append(
            _render_video(
                motion,
                prism_root,
                evaluation_root,
                data_root,
                destination,
                stride=stride,
                fps=fps,
            )
            | {"condition": condition}
        )
    return results


def _write_prism_table(evaluation_root: Path, output: Path) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        summary = _json(evaluation_root / method.name / "evaluation" / "prism_mesh" / "summary.json")
        overall = summary["overall"]
        rows.append(
            {
                "method": method.label,
                "scored": summary["scored"],
                "selected": summary["selected"],
                "primary_passes": overall["primary_passes"],
                "observed_height_mae_mm": overall["observed_height_mae_mm"]["mean"],
                "observed_within_50mm": overall["observed_within_50mm"]["mean"],
                "raised_terrain_coverage": overall["raised_terrain_coverage"]["mean"],
                "flat_terrain_coverage": overall["flat_terrain_coverage"]["mean"],
                "footprint_iou": overall["full_footprint_iou"]["mean"],
                "support_f1_50mm": overall["full_support_f1_50mm"]["mean"],
            }
        )
    with (output / "prism_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_voronoi_sensitivity(
    summaries: Mapping[str, Path],
    output: Path,
) -> list[dict[str, Any]]:
    """Compare completed Voronoi variants on the same PRISM cohort."""

    rows = []
    for label, path in summaries.items():
        summary = _json(path)
        overall = summary["overall"]
        rows.append(
            {
                "variant": label,
                "selected": summary["selected"],
                "scored": summary["scored"],
                "primary_passes": overall["primary_passes"],
                "observed_height_mae_mm": overall["observed_height_mae_mm"]["mean"],
                "raised_terrain_coverage": overall["raised_terrain_coverage"]["mean"],
                "flat_terrain_coverage": overall["flat_terrain_coverage"]["mean"],
                "footprint_iou": overall["full_footprint_iou"]["mean"],
                "support_f1_50mm": overall["full_support_f1_50mm"]["mean"],
                "source_summary": str(path),
            }
        )
    with (output / "voronoi_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt = _mpl()
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    labels = [str(row["variant"]) for row in rows]
    colors = ["#d55e00", "#e69f00", "#999999", "#56b4e9"][: len(rows)]
    axes[0].bar(labels, [float(row["observed_height_mae_mm"]) for row in rows], color=colors)
    axes[0].set_ylabel("Observed support MAE (mm)")
    axes[1].bar(labels, [float(row["primary_passes"]) for row in rows], color=colors)
    axes[1].set_ylabel("Primary passes (of 31)")
    width = 0.24
    x = np.arange(len(rows))
    axes[2].bar(x - width, [float(row["raised_terrain_coverage"]) for row in rows], width, label="Raised coverage")
    axes[2].bar(x, [float(row["flat_terrain_coverage"]) for row in rows], width, label="Flat coverage")
    axes[2].bar(x + width, [float(row["footprint_iou"]) for row in rows], width, label="Footprint IoU")
    axes[2].set_ylabel("Fraction")
    axes[2].set_xticks(x, labels)
    axes[2].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.20)
    figure.suptitle("Voronoi influence-width sensitivity on the same 31 PRISM takes")
    figure.subplots_adjust(bottom=0.22, top=0.84, wspace=0.34)
    _save_figure(figure, output / "voronoi_sensitivity.png")
    plt.close(figure)
    return rows


def render_foot_query_sensitivity(
    prism_root: Path,
    data_root: Path,
    motions: Sequence[str],
    output: Path,
    *,
    tolerances_m: Sequence[float] = (0.05, 0.075, 0.10, 0.15),
) -> list[dict[str, Any]]:
    """Audit support-height means against the insole vertical-consistency threshold."""

    rows = []
    for tolerance in tolerances_m:
        values: dict[str, list[float]] = {method.name: [] for method in METHODS}
        candidate_points = rejected_points = raised_points = 0
        for motion in motions:
            _, subject, take_name = motion.split("/")
            take = load_take(data_root / subject / f"{take_name.removesuffix('_poses')}.pkl")
            candidates = observed_support_points(take, stride=10)
            gt = mesh_height_at(candidates[:, :2], take["objects"])
            accepted = np.abs(candidates[:, 2] - gt) <= tolerance
            raised = accepted & (gt > 0.02)
            candidate_points += len(candidates)
            rejected_points += int((~accepted).sum())
            raised_points += int(raised.sum())
            if not np.any(raised):
                continue
            for method in METHODS:
                terrain = _terrain(prism_root / method.name / "terrain", motion)
                predicted = np.asarray(
                    terrain.height_at(candidates[raised, 0], candidates[raised, 1]),
                    dtype=float,
                )
                values[method.name].append(1000.0 * float(np.mean(np.abs(predicted - gt[raised]))))
        row: dict[str, Any] = {
            "vertical_tolerance_m": tolerance,
            "candidate_points": candidate_points,
            "rejected_points": rejected_points,
            "raised_points": raised_points,
            "defined_motions": len(values["terra"]),
        }
        row.update(
            {f"{method.name}_foot_height_mae_mm": float(np.mean(values[method.name])) for method in METHODS}
        )
        rows.append(row)
    with (output / "foot_query_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt = _mpl()
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    x = [1000.0 * float(row["vertical_tolerance_m"]) for row in rows]
    for method in METHODS:
        y = [float(row[f"{method.name}_foot_height_mae_mm"]) for row in rows]
        axis.plot(x, y, marker="o", ms=4, lw=1.6, color=method.color, label=method.label)
    axis.axvline(100.0, color="black", ls="--", lw=1.0, label="Benchmark threshold")
    axis.set_xlabel("Insole-to-mesh vertical consistency threshold (mm)")
    axis.set_ylabel("Foot support height MAE (mm)")
    axis.grid(alpha=0.20)
    axis.legend(fontsize=7, ncol=2, frameon=False)
    figure.suptitle("PRISM foot-query threshold sensitivity")
    _save_figure(figure, output / "foot_query_sensitivity.png")
    plt.close(figure)
    return rows


def _write_report(
    output: Path,
    prism_rows: Sequence[Mapping[str, Any]],
    prism_selection: Mapping[str, str],
    apparatus_selection: Mapping[str, Mapping[str, str]],
    videos: Sequence[Mapping[str, Any]],
    voronoi_sensitivity: Sequence[Mapping[str, Any]],
    foot_query_sensitivity: Sequence[Mapping[str, Any]],
) -> None:
    maximum_foot_mae_range = max(
        (
            max(float(row[f"{method.name}_foot_height_mae_mm"]) for row in foot_query_sensitivity)
            - min(float(row[f"{method.name}_foot_height_mae_mm"]) for row in foot_query_sensitivity)
            for method in METHODS
        ),
        default=0.0,
    )
    lines = [
        "# Reconstruction benchmark audit figures",
        "",
        "PRISM representatives are selected by ground-truth raised area: each is the take nearest the median "
        "within its geometry class. Apparatus representatives are the motions nearest the median source-motion "
        "duration within each condition. These deterministic rules do not use reconstruction performance or "
        "visual appearance.",
        "",
        "## PRISM metrics",
        "",
        "| Method | Primary accuracy gate | Observed support MAE (mm) | Within 50 mm | "
        "Raised coverage | Flat coverage | Footprint IoU | F1 @50 mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prism_rows:
        lines.append(
            f"| {row['method']} | {row['primary_passes']}/{row['scored']} | "
            f"{float(row['observed_height_mae_mm']):.2f} | {float(row['observed_within_50mm']):.3f} | "
            f"{float(row['raised_terrain_coverage']):.3f} | {float(row['flat_terrain_coverage']):.3f} | "
            f"{float(row['footprint_iou']):.3f} | {float(row['support_f1_50mm']):.3f} |"
        )
    if voronoi_sensitivity:
        lines.extend(
            [
                "",
                "## Voronoi sensitivity",
                "",
                "| Variant | Primary passes | Observed support MAE (mm) | Raised coverage | "
                "Flat coverage | Footprint IoU |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in voronoi_sensitivity:
            lines.append(
                f"| {row['variant']} | {row['primary_passes']}/{row['scored']} | "
                f"{float(row['observed_height_mae_mm']):.2f} | "
                f"{float(row['raised_terrain_coverage']):.3f} | "
                f"{float(row['flat_terrain_coverage']):.3f} | "
                f"{float(row['footprint_iou']):.3f} |"
            )
        lines.extend(
            [
                "",
                "The benchmark uses TIP's 1 m x 1 m influence bound. Following SceneBot's algorithm order, "
                "interaction edges are collision-tested at their spatial locations before square plateaus are "
                "constructed; collision-causing cells are carved only after Voronoi terrain reconstruction.",
            ]
        )
    lines.extend(["", "## Representative motions", ""])
    for condition, motion in prism_selection.items():
        lines.append(f"- PRISM {condition}: `{motion}`")
    for dataset, selections in apparatus_selection.items():
        for condition, motion in selections.items():
            lines.append(f"- {dataset} {condition}: `{motion}`")
    figure_inventory = [
        "",
        "## Figure inventory",
        "",
        "- [PRISM height fields](prism_heightfields.png)",
        "- [PRISM signed height errors](prism_height_errors.png)",
        "- [PRISM metric distributions](prism_metric_distributions.png)",
        "- [PRISM perspective scenes](prism_perspective_scenes.png)",
    ]
    if voronoi_sensitivity:
        figure_inventory.append("- [Voronoi influence-width sensitivity](voronoi_sensitivity.png)")
    if foot_query_sensitivity:
        figure_inventory.append("- [PRISM foot-query threshold sensitivity](foot_query_sensitivity.png)")
    figure_inventory.extend(
        [
            "- [Gait120 apparatus profiles](gait120_representative_profiles.png)",
            "- [Darmstadt apparatus profiles](darmstadt_representative_profiles.png)",
            "- [Vielemeyer apparatus profiles](vielemeyer_representative_profiles.png)",
        ]
    )
    lines.extend(figure_inventory)
    if videos:
        lines.extend(["", "## Scene renders", ""])
        for video in videos:
            lines.append(f"- [{video['condition']}]({Path(str(video['output'])).name})")
    lines.extend(
        [
            "",
            "White/black PRISM contours are reference object footprints. Black dots in height-field panels are "
            "measured insole support samples. Error maps show predicted minus ground-truth support height and "
            "are clipped at ±20 cm only for color legibility; the printed metrics use unclipped values.",
            "The perspective panels and MP4s overlay source kinematics for spatial context; they are scene renders, "
            "not policy rollouts or retargeting evaluations on baseline terrain.",
            f"Across the audited insole vertical-consistency thresholds, method-level foot-support MAEs change "
            f"by at most {maximum_foot_mae_range:.2f} mm; the reported ordering is not driven by the "
            "100 mm threshold.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def _assignments(values: Iterable[str], flag: str) -> dict[str, Path]:
    result = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not dataset or not path or dataset in result:
            raise ValueError(f"{flag} must use unique DATASET=PATH values")
        result[dataset] = Path(path).expanduser().resolve()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="terra reconstruction-figures", description=__doc__)
    parser.add_argument("--prism-root", type=Path, required=True, help="root containing METHOD/terrain")
    parser.add_argument(
        "--prism-evaluation-root",
        type=Path,
        required=True,
        help="root containing METHOD/evaluation/prism_mesh",
    )
    parser.add_argument("--prism-data-root", type=Path, required=True)
    parser.add_argument("--apparatus-root", type=Path, required=True)
    parser.add_argument("--manifest", action="append", default=[], help="DATASET=manifest.csv; repeat per dataset")
    parser.add_argument(
        "--voronoi-sensitivity",
        action="append",
        default=[],
        help="LABEL=summary.json for completed PRISM Voronoi variants",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=80)
    parser.add_argument("--video-fps", type=int, default=15)
    args = parser.parse_args(argv)
    if args.video_stride < 1 or args.video_fps < 1:
        parser.error("video-stride and video-fps must be positive")
    try:
        manifests = _assignments(args.manifest, "--manifest")
        sensitivity_summaries = _assignments(args.voronoi_sensitivity, "--voronoi-sensitivity")
    except ValueError as error:
        parser.error(str(error))
    required = {"gait120", "darmstadt", "vielemeyer"}
    if set(manifests) != required:
        parser.error(f"--manifest must define exactly {', '.join(sorted(required))}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_git_commit(output)
    prism_root = args.prism_root.expanduser().resolve()
    evaluation_root = args.prism_evaluation_root.expanduser().resolve()
    data_root = args.prism_data_root.expanduser().resolve()
    reference_records = _evaluation_records(evaluation_root, "terra")
    prism_selection = select_prism_representatives(reference_records)
    render_prism_heightfields(prism_root, data_root, prism_selection, output)
    render_prism_errors(prism_root, data_root, evaluation_root, prism_selection, output)
    render_prism_perspectives(prism_root, data_root, prism_selection, output)
    render_prism_metric_distributions(evaluation_root, output)
    apparatus_selection = render_apparatus_profiles(
        args.apparatus_root.expanduser().resolve(),
        manifests,
        output,
    )
    videos = (
        render_prism_videos(
            prism_root,
            evaluation_root,
            data_root,
            prism_selection,
            output,
            stride=args.video_stride,
            fps=args.video_fps,
        )
        if args.videos
        else []
    )
    prism_rows = _write_prism_table(evaluation_root, output)
    sensitivity_rows = (
        render_voronoi_sensitivity(sensitivity_summaries, output) if sensitivity_summaries else []
    )
    foot_query_rows = render_foot_query_sensitivity(
        prism_root,
        data_root,
        [str(record["motion"]) for record in reference_records],
        output,
    )
    selection = {
        "rules": {
            "prism": "nearest to condition median ground-truth raised area",
            "apparatus": "nearest to condition median source-motion duration",
        },
        "prism": prism_selection,
        "apparatus": apparatus_selection,
        "videos": videos,
        "voronoi_sensitivity": {
            label: str(path) for label, path in sensitivity_summaries.items()
        },
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    _write_report(
        output,
        prism_rows,
        prism_selection,
        apparatus_selection,
        videos,
        sensitivity_rows,
        foot_query_rows,
    )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "render_apparatus_profiles",
    "render_foot_query_sensitivity",
    "render_prism_errors",
    "render_prism_heightfields",
    "render_prism_metric_distributions",
    "render_prism_perspectives",
    "render_prism_videos",
    "render_voronoi_sensitivity",
    "select_apparatus_representatives",
    "select_prism_representatives",
]
