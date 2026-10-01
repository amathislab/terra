"""Publication rendering for TERRA's paired-scene RL training input."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from terra._files import atomic_write
from terra.constants import MYOFULLBODY_SITE_CALIBRATION
from terra.figures.blender_renderer import export_scene_bundle
from terra.figures.geometry import alignment_yaw, rotation_z
from terra.figures.manifest import (
    CellSpec,
    FigureManifest,
    LayoutSpec,
    LightingSpec,
    StyleSpec,
)
from terra.figures.mujoco_renderer import _artifact_paths, _model, _sha256
from terra.figures.terrain_priors import (
    _audit_content_margin,
    _clean_transparent_film,
    _content_bounds,
    _render_bundle,
)

BACKGROUND = (0.975, 0.973, 0.963)
PAD_GRAY = (0.90, 0.90, 0.88, 1.0)
TERRAIN_GRAY = (0.47, 0.51, 0.53, 1.0)
REFERENCE_CYAN = (0.00, 0.57, 0.78, 0.98)
CURRENT_SITE_ORANGE = (0.98, 0.43, 0.10, 0.98)
CORRESPONDENCE_BLACK = (0.025, 0.025, 0.025, 0.88)
HEIGHTMAP_LOW = (0.02, 0.36, 0.68)
HEIGHTMAP_HIGH = (0.98, 0.59, 0.12)

# Edges are named in the SMPL-H landmark convention used by the calibrated
# MyoFullBody mimic sites. The result is one connected, anatomically readable
# 17-landmark reference skeleton.
REFERENCE_EDGE_NAMES = (
    ("Pelvis", "Spine"),
    ("Spine", "Head"),
    ("Spine", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Spine", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),
)


@dataclass(frozen=True, slots=True)
class RLSceneSpec:
    """Data and presentation choices for one paired-scene RL asset."""

    motion: str = "KIT/3/upstairs09_poses"
    frame: int = 329
    reference_frame: int = 424
    width: int = 1200
    height: int = 1500
    samples: int = 128
    grid_rows: int = 11
    grid_cols: int = 11
    grid_resolution: float = 0.1
    grid_forward_offset: float = 0.0
    alignment_yaw_degrees: float | None = -84.47901043536575
    camera_azimuth_degrees: float = 43.0
    camera_elevation_degrees: float = -25.0
    camera_usable_fraction: float = 0.74
    reference_view_offset: float = 0.025


def heightmap_offsets(
    rows: int,
    cols: int,
    resolution: float,
    forward_offset: float = 0.0,
) -> np.ndarray:
    """Return the exact row-major local offsets used by ``HeightMatrix``."""

    if rows < 1 or cols < 1:
        raise ValueError("heightmap rows and columns must be positive")
    if resolution <= 0.0:
        raise ValueError("heightmap resolution must be positive")
    half_rows = (rows - 1) / 2.0
    half_cols = (cols - 1) / 2.0
    return np.asarray(
        [
            (
                (row - half_rows) * resolution + forward_offset,
                (column - half_cols) * resolution,
            )
            for row in range(rows)
            for column in range(cols)
        ],
        dtype=float,
    )


def reference_edges() -> tuple[tuple[int, int], ...]:
    """Return reference-skeleton indices in calibrated mimic-site order."""

    index = {
        source_joint: position for position, (_site, source_joint, _body) in enumerate(MYOFULLBODY_SITE_CALIBRATION)
    }
    return tuple((index[left], index[right]) for left, right in REFERENCE_EDGE_NAMES)


def correspondence_dots(
    current: np.ndarray,
    target: np.ndarray,
    *,
    spacing: float = 0.055,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample dotted 3-D correspondences between paired mimic sites.

    The returned pair index identifies which calibrated site generated each
    dot. Endpoints are omitted because they receive dedicated site markers.
    """

    current_points = np.asarray(current, dtype=float)
    target_points = np.asarray(target, dtype=float)
    if current_points.shape != target_points.shape or current_points.ndim != 2 or current_points.shape[1] != 3:
        raise ValueError("current and target mimic sites must have matching shape (N, 3)")
    if spacing <= 0.0:
        raise ValueError("correspondence dot spacing must be positive")
    dots = []
    pair_indices = []
    for pair_index, (start, end) in enumerate(zip(current_points, target_points, strict=True)):
        distance = float(np.linalg.norm(end - start))
        count = max(2, math.ceil(distance / spacing))
        for fraction in np.linspace(0.16, 0.84, count):
            dots.append((1.0 - fraction) * start + fraction * end)
            pair_indices.append(pair_index)
    return np.asarray(dots, dtype=float), np.asarray(pair_indices, dtype=np.int16)


def _validate_spec(spec: RLSceneSpec) -> None:
    if not spec.motion.strip():
        raise ValueError("RL scene motion must be non-empty")
    if spec.frame < 0:
        raise ValueError("RL scene frame must be non-negative")
    if spec.reference_frame <= spec.frame:
        raise ValueError("RL scene reference frame must be later than the agent frame")
    if spec.width < 64 or spec.height < 64:
        raise ValueError("RL scene dimensions must be at least 64 pixels")
    if spec.samples < 1:
        raise ValueError("RL scene samples must be positive")
    heightmap_offsets(
        spec.grid_rows,
        spec.grid_cols,
        spec.grid_resolution,
        spec.grid_forward_offset,
    )
    if not 0.5 <= spec.camera_usable_fraction <= 0.9:
        raise ValueError("camera usable fraction must lie in [0.5, 0.9]")
    if not 0.0 <= spec.reference_view_offset <= 0.1:
        raise ValueError("reference view offset must lie in [0, 0.1] m")


def _scene_manifest(
    spec: RLSceneSpec,
    *,
    anchor_xy: np.ndarray,
    yaw_degrees: float,
    provenance_path: Path,
) -> FigureManifest:
    layout = LayoutSpec(
        rows=1,
        columns=1,
        width=spec.width,
        height=spec.height,
        cell_size=(2.25, 1.72),
        cell_pitch=(2.25, 1.72),
        pad_thickness=0.045,
        decorative_border=0,
        camera_azimuth=spec.camera_azimuth_degrees,
        camera_elevation=spec.camera_elevation_degrees,
        camera_distance=5.0,
        camera_lookat_z=1.05,
        field_of_view=38.0,
    )
    style = StyleSpec(
        background=BACKGROUND,
        pad=PAD_GRAY,
        terrain=TERRAIN_GRAY,
        history=REFERENCE_CYAN[:3],
        history_alpha=(0.30,),
    )
    lighting = LightingSpec(
        ambient=(0.23, 0.23, 0.24),
        headlight_diffuse=(0.08, 0.08, 0.09),
        key_direction=(-0.35, 0.45, -0.82),
        key_diffuse=(0.86, 0.82, 0.75),
        key_specular=(0.28, 0.25, 0.21),
        fill_direction=(0.65, -0.25, -0.72),
        fill_diffuse=(0.24, 0.28, 0.34),
        key_samples=5,
        key_spread=0.18,
        shadow_map_size=8192,
        shadow_clip=1.05,
        ambient_strength=0.27,
        direct_light_scale=1.08,
    )
    return FigureManifest(
        version=1,
        name="paired-scene-rl-training",
        model="MyoFullBody",
        method="terra",
        layout=layout,
        style=style,
        lighting=lighting,
        cells=(
            CellSpec(
                row=0,
                column=0,
                motion=spec.motion,
                label="paired-scene RL input",
                frames=(spec.frame, spec.reference_frame),
                fractions=None,
                yaw_degrees=yaw_degrees,
                anchor_xy=tuple(float(value) for value in anchor_xy),
                highlight_frame=spec.frame,
            ),
        ),
        path=provenance_path,
    )


def _root_heading(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _scene_transform(points: np.ndarray, anchor_xy: np.ndarray, yaw_degrees: float) -> np.ndarray:
    values = np.asarray(points, dtype=float).copy()
    values[..., :2] -= anchor_xy
    return values @ rotation_z(yaw_degrees).T


def _mimic_site_positions(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, tuple[str, ...]]:
    points = []
    names = []
    for site_name, source_joint, _body_name in MYOFULLBODY_SITE_CALIBRATION:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"MyoFullBody mimic site is missing: {site_name}")
        points.append(np.asarray(data.site_xpos[site_id], dtype=float).copy())
        names.append(source_joint)
    return np.stack(points), tuple(names)


def _box_corners(position: np.ndarray, half_size: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    local = np.asarray(
        [
            (sx * half_size[0], sy * half_size[1], sz * half_size[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    return local @ rotation.T + position


def _bundle_points(arrays: dict[str, np.ndarray]) -> np.ndarray:
    points = [
        _box_corners(position, size, rotation)
        for position, size, rotation in zip(
            arrays["boxes_position"],
            arrays["boxes_size"],
            arrays["boxes_rotation"],
            strict=True,
        )
    ]
    for mesh_id, transform in zip(arrays["instance_mesh"], arrays["instance_transform"], strict=True):
        vertices = arrays[f"mesh_{int(mesh_id)}_vertices"]
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        points.append(world)
    if len(arrays["tendon_endpoints"]):
        points.append(arrays["tendon_endpoints"].reshape(-1, 3))
    return np.concatenate(points, axis=0)


def _camera_payload(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    azimuth_degrees: float,
    elevation_degrees: float,
    usable_fraction: float,
) -> dict[str, object]:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    forward = np.asarray(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ),
        dtype=float,
    )
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    projected_right = points @ right
    projected_up = points @ up
    centre_right = 0.5 * (float(np.min(projected_right)) + float(np.max(projected_right)))
    centre_up = 0.5 * (float(np.min(projected_up)) + float(np.max(projected_up)))
    lookat = np.mean(points, axis=0)
    lookat += (centre_right - float(lookat @ right)) * right
    lookat += (centre_up - float(lookat @ up)) * up
    aspect = width / height
    ortho_scale = (
        max(
            float(np.ptp(projected_right)),
            float(np.ptp(projected_up)) * aspect,
        )
        / usable_fraction
    )
    distance = 5.0
    return {
        "position": (lookat - distance * forward).tolist(),
        "forward": forward.tolist(),
        "up": up.tolist(),
        "fovy_degrees": 38.0,
        "projection": "orthographic",
        "ortho_scale": ortho_scale,
    }


def _height_colors(heights: np.ndarray) -> np.ndarray:
    values = np.asarray(heights, dtype=float)
    span = float(np.ptp(values))
    normalized = np.zeros_like(values) if span < 1e-9 else (values - float(np.min(values))) / span
    low = np.asarray(HEIGHTMAP_LOW)
    high = np.asarray(HEIGHTMAP_HIGH)
    rgb = low + normalized[:, None] * (high - low)
    return np.column_stack((rgb, np.full(len(rgb), 0.88)))


def build_rl_scene_bundle(
    spec: RLSceneSpec,
    *,
    cache_root: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Build a Blender bundle from one exact trajectory/terrain pair."""

    _validate_spec(spec)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_root).expanduser().resolve()

    # A provisional manifest resolves the exact cache paths and builds the same
    # model used by the publication renderer.
    provisional = _scene_manifest(
        spec,
        anchor_xy=np.zeros(2),
        yaw_degrees=0.0,
        provenance_path=destination,
    )
    trajectory_path, terrain_path, analysis_path = _artifact_paths(provisional, cache, spec.motion)
    from loco_mujoco.core.terrain import TerrainSpec
    from loco_mujoco.trajectory import Trajectory

    qpos = np.asarray(Trajectory.load(str(trajectory_path)).data.qpos, dtype=float)
    if not 0 <= spec.frame < len(qpos):
        raise ValueError(f"frame {spec.frame} lies outside the {len(qpos)}-frame trajectory")
    if not 0 <= spec.reference_frame < len(qpos):
        raise ValueError(f"reference frame {spec.reference_frame} lies outside the {len(qpos)}-frame trajectory")
    anchor_xy = np.asarray(qpos[spec.frame, :2], dtype=float)
    window = tuple(range(max(0, spec.frame - 40), min(len(qpos), spec.frame + 41)))
    yaw_degrees = alignment_yaw(qpos[:, :2], window, spec.alignment_yaw_degrees)
    manifest = _scene_manifest(
        spec,
        anchor_xy=anchor_xy,
        yaw_degrees=yaw_degrees,
        provenance_path=destination,
    )

    base_bundle = destination.with_name(f".{destination.stem}.base.npz")
    try:
        _metadata, cell_provenance = export_scene_bundle(
            manifest,
            cache_root=cache,
            output=base_bundle,
            scale=1.0,
        )
        with np.load(base_bundle, allow_pickle=False) as base:
            arrays = {name: np.asarray(base[name]).copy() for name in base.files}
    finally:
        base_bundle.unlink(missing_ok=True)

    model = _model(manifest, spec.width, spec.height)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[spec.frame]
    mujoco.mj_forward(model, data)

    current_sites, source_joint_names = _mimic_site_positions(model, data)
    transformed_current_sites = _scene_transform(current_sites, anchor_xy, yaw_degrees)
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body_id < 0:
        raise ValueError("MyoFullBody pelvis body is missing")
    pelvis = np.asarray(data.xpos[pelvis_body_id], dtype=float).copy()
    root_yaw = _root_heading(np.asarray(data.qpos[3:7], dtype=float))
    data.qpos[:] = qpos[spec.reference_frame]
    mujoco.mj_forward(model, data)
    target_sites, target_source_joint_names = _mimic_site_positions(model, data)
    if target_source_joint_names != source_joint_names:
        raise AssertionError("current and target mimic-site order changed within one model")
    transformed_target_sites = _scene_transform(target_sites, anchor_xy, yaw_degrees)
    offsets = heightmap_offsets(
        spec.grid_rows,
        spec.grid_cols,
        spec.grid_resolution,
        spec.grid_forward_offset,
    )
    cosine, sine = math.cos(root_yaw), math.sin(root_yaw)
    sample_xy = np.column_stack(
        (
            pelvis[0] + cosine * offsets[:, 0] - sine * offsets[:, 1],
            pelvis[1] + sine * offsets[:, 0] + cosine * offsets[:, 1],
        )
    )
    terrain = TerrainSpec.load(str(terrain_path)).walkable
    absolute_heights = np.asarray(terrain.height_at(sample_xy[:, 0], sample_xy[:, 1]), dtype=float)
    current_site_support = np.asarray(terrain.height_at(current_sites[:, 0], current_sites[:, 1]), dtype=float)
    target_site_support = np.asarray(terrain.height_at(target_sites[:, 0], target_sites[:, 1]), dtype=float)
    sample_world = np.column_stack((sample_xy, absolute_heights))
    transformed_samples = _scene_transform(sample_world, anchor_xy, yaw_degrees)

    camera = _camera_payload(
        _bundle_points(arrays),
        width=spec.width,
        height=spec.height,
        azimuth_degrees=spec.camera_azimuth_degrees,
        elevation_degrees=spec.camera_elevation_degrees,
        usable_fraction=spec.camera_usable_fraction,
    )
    forward = np.asarray(camera["forward"], dtype=float)
    displayed_current_sites = transformed_current_sites - spec.reference_view_offset * forward
    displayed_target_sites = transformed_target_sites - spec.reference_view_offset * forward
    dots, dot_pair_indices = correspondence_dots(displayed_current_sites, displayed_target_sites)
    height_point_radius = min(0.017, 0.17 * spec.grid_resolution)
    displayed_height_samples = transformed_samples.copy()
    displayed_height_samples[:, 2] += height_point_radius + 0.003
    point_positions = np.concatenate(
        (displayed_height_samples, displayed_current_sites, displayed_target_sites, dots),
        axis=0,
    )
    point_radii = np.concatenate(
        (
            np.full(len(displayed_height_samples), height_point_radius),
            np.full(len(displayed_current_sites), 0.020),
            np.full(len(displayed_target_sites), 0.022),
            np.full(len(dots), 0.010),
        )
    )
    point_rgba = np.concatenate(
        (
            _height_colors(absolute_heights),
            np.tile(CURRENT_SITE_ORANGE, (len(displayed_current_sites), 1)),
            np.tile(REFERENCE_CYAN, (len(displayed_target_sites), 1)),
            np.tile(CORRESPONDENCE_BLACK, (len(dots), 1)),
        ),
        axis=0,
    )
    arrays["overlay_points_position"] = point_positions.astype(np.float32)
    arrays["overlay_points_radius"] = point_radii.astype(np.float32)
    arrays["overlay_points_rgba"] = point_rgba.astype(np.float32)

    metadata = json.loads(str(arrays["metadata"]))
    metadata.update(
        {
            "schema_version": 2,
            "name": "paired-scene-rl-training",
            "transparent_background": True,
            "samples": spec.samples,
            "exposure": 0.58,
            "camera": camera,
            "bevel": {"pad": 0.016, "terrain": 0.018},
            "shadow_catcher": {"size": 5.0, "z": -0.050},
            "rl_scene": {
                "agent_state": "cached qpos at the agent frame (not a claimed policy rollout)",
                "reference": "translucent cached future pose with 17 calibrated mimic sites",
                "correspondence": "dotted current-to-future mimic-site pairs",
                "heightmap": "exact HeightMatrix samples rendered as spherical points",
            },
        }
    )
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(destination, **arrays)

    relative_heights = absolute_heights - pelvis[2]
    return {
        "bundle": str(destination),
        "motion": spec.motion,
        "frame": spec.frame,
        "reference_frame": spec.reference_frame,
        "reference_frame_offset": spec.reference_frame - spec.frame,
        "anchor_xy": anchor_xy.tolist(),
        "alignment_yaw_degrees": yaw_degrees,
        "root_heading_degrees": math.degrees(root_yaw),
        "camera": camera,
        "heightmap": {
            "rows": spec.grid_rows,
            "cols": spec.grid_cols,
            "resolution_m": spec.grid_resolution,
            "forward_offset_m": spec.grid_forward_offset,
            "sample_xy_world_m": sample_xy.tolist(),
            "absolute_height_m": absolute_heights.reshape(spec.grid_rows, spec.grid_cols).tolist(),
            "pelvis_relative_height_m": relative_heights.reshape(spec.grid_rows, spec.grid_cols).tolist(),
        },
        "reference": {
            "site_names": [item[0] for item in MYOFULLBODY_SITE_CALIBRATION],
            "source_joint_names": list(source_joint_names),
            "current_site_xyz_world_m": current_sites.tolist(),
            "target_site_xyz_world_m": target_sites.tolist(),
            "current_site_support_height_m": current_site_support.tolist(),
            "target_site_support_height_m": target_site_support.tolist(),
            "correspondence_pair_count": len(current_sites),
            "correspondence_dot_count": len(dots),
            "correspondence_dot_pair_indices": dot_pair_indices.tolist(),
            "display_view_offset_m": spec.reference_view_offset,
        },
        "inputs": {
            "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
            "terrain": {"path": str(terrain_path), "sha256": _sha256(terrain_path)},
            "analysis": {"path": str(analysis_path), "sha256": _sha256(analysis_path)},
        },
        "cell": cell_provenance[0],
    }


def render_rl_scene(
    spec: RLSceneSpec,
    *,
    cache_root: str | Path,
    blender_executable: str | Path,
    output: str | Path,
    scale: float = 1.0,
) -> dict[str, object]:
    """Render and audit a transparent paired-scene RL training asset."""

    if not 0.1 <= scale <= 1.0:
        raise ValueError(f"scale must lie in [0.1, 1.0], got {scale}")
    blender = Path(blender_executable).expanduser().resolve()
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    destination = Path(output).expanduser().resolve()
    if destination.suffix.casefold() != ".png":
        raise ValueError("RL scene output must use the .png extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_spec = RLSceneSpec(
        motion=spec.motion,
        frame=spec.frame,
        reference_frame=spec.reference_frame,
        width=max(64, round(spec.width * scale)),
        height=max(64, round(spec.height * scale)),
        samples=max(16, round(spec.samples * scale)),
        grid_rows=spec.grid_rows,
        grid_cols=spec.grid_cols,
        grid_resolution=spec.grid_resolution,
        grid_forward_offset=spec.grid_forward_offset,
        alignment_yaw_degrees=spec.alignment_yaw_degrees,
        camera_azimuth_degrees=spec.camera_azimuth_degrees,
        camera_elevation_degrees=spec.camera_elevation_degrees,
        camera_usable_fraction=spec.camera_usable_fraction,
        reference_view_offset=spec.reference_view_offset,
    )
    with tempfile.TemporaryDirectory(prefix="terra-rl-scene-") as temporary_directory:
        temporary = Path(temporary_directory)
        bundle = temporary / "rl-scene.npz"
        raw_render = temporary / "rl-scene.png"
        diagnostics = build_rl_scene_bundle(render_spec, cache_root=cache_root, output=bundle)
        # The generated bundle is intentionally temporary; provenance records
        # the recipe and exact inputs rather than a dead temporary path.
        diagnostics.pop("bundle", None)
        renderer = _render_bundle(bundle, raw_render, blender)
        atomic_write(destination, lambda path: shutil.copyfile(raw_render, path))
    _clean_transparent_film(destination)
    bounds = _content_bounds(destination)
    _audit_content_margin(bounds, width=render_spec.width, height=render_spec.height)
    renderer.update(
        {
            "output": str(destination),
            "sha256": _sha256(destination),
            "alpha_cleanup_threshold": 15,
        }
    )
    renderer_path = destination.with_suffix(".renderer.json")
    atomic_write(
        renderer_path,
        lambda path: path.write_text(json.dumps(renderer, indent=2, sort_keys=True) + "\n"),
    )
    provenance = {
        "schema_version": 1,
        "description": "TERRA paired-scene RL training input: MyoFullBody, future reference ghost, correspondences, and local heightmap",
        "configuration": {
            "motion": spec.motion,
            "frame": spec.frame,
            "reference_frame": spec.reference_frame,
            "reference_frame_offset": spec.reference_frame - spec.frame,
            "dimensions": [render_spec.width, render_spec.height],
            "samples": render_spec.samples,
            "scale": scale,
            "grid_rows": spec.grid_rows,
            "grid_cols": spec.grid_cols,
            "grid_resolution_m": spec.grid_resolution,
            "grid_forward_offset_m": spec.grid_forward_offset,
            "alignment_yaw_degrees": spec.alignment_yaw_degrees,
            "camera_azimuth_degrees": spec.camera_azimuth_degrees,
            "camera_elevation_degrees": spec.camera_elevation_degrees,
            "camera_usable_fraction": spec.camera_usable_fraction,
        },
        "scientific_contract": {
            "agent": (f"exact cached configuration at frame {spec.frame}; the asset does not claim a learned rollout"),
            "reference": (
                f"exact cached future configuration at frame {spec.reference_frame}, "
                f"displayed {spec.reference_frame - spec.frame} frames after the agent"
            ),
            "correspondence": "17 dotted current-to-future calibrated mimic-site pairs",
            "heightmap": "pelvis-centered, root-heading-aligned, row-major HeightMatrix samples rendered as dots",
            "terrain": "walkable surface paired with the cached retargeted trajectory",
        },
        "diagnostics": diagnostics,
        "renderer": renderer,
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
            "content_bounds": bounds,
        },
    }
    provenance_path = destination.with_suffix(".provenance.json")
    atomic_write(
        provenance_path,
        lambda path: path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n"),
    )
    return {
        "output": str(destination),
        "renderer": str(renderer_path),
        "provenance": str(provenance_path),
        "width": render_spec.width,
        "height": render_spec.height,
        "content_bounds": bounds,
        "motion": spec.motion,
        "frame": spec.frame,
        "reference_frame": spec.reference_frame,
        "device": renderer.get("device"),
    }


__all__ = [
    "REFERENCE_EDGE_NAMES",
    "RLSceneSpec",
    "build_rl_scene_bundle",
    "correspondence_dots",
    "heightmap_offsets",
    "reference_edges",
    "render_rl_scene",
]
