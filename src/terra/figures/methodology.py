"""Data-driven Blender pipeline for the four-panel TERRA methodology figure."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay

from terra._files import atomic_write
from terra.constants import SMPLH_TO_MYOFULLBODY
from terra.figures.blender_renderer import export_scene_bundle
from terra.figures.geometry import rotation_z
from terra.figures.manifest import (
    CellSpec,
    FigureManifest,
    LayoutSpec,
    LightingSpec,
    StyleSpec,
)
from terra.figures.mujoco_renderer import _artifact_paths, _model
from terra.smplh import load_smplh_motion
from terra.source import get_robot_height, normalized_motion_landmarks, terrain_scene
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS, detect_stance_events

SOURCE_NEUTRAL = (0.28, 0.47, 0.63, 1.0)
SOURCE_GHOST = (0.25, 0.38, 0.48)
FIT_BLUE = (0.00, 0.32, 0.52, 1.0)
LANDMARK_BLUE = (0.00, 0.45, 0.70, 1.0)
SUPPORT_CYAN = (0.10, 0.55, 0.75, 1.0)
FREE_SPACE_CORAL = (0.84, 0.37, 0.00, 0.56)
INTERACTION_TEAL = (0.00, 0.62, 0.45, 0.50)
CONSTRAINT_GREEN = (0.16, 0.68, 0.35, 1.0)
CLEARANCE_AMBER = (0.90, 0.62, 0.00, 1.0)
TERRAIN_GRAY = (0.47, 0.51, 0.53, 1.0)
PAD_GRAY = (0.90, 0.90, 0.88, 1.0)
DIMENSION_CHARCOAL = (0.18, 0.21, 0.23, 0.94)
BACKGROUND = (0.975, 0.973, 0.963)


@dataclass(frozen=True, slots=True)
class PanelCameraSpec:
    """Per-panel framing while preserving the figure's shared view direction."""

    distance: float
    lookat: tuple[float, float, float]
    field_of_view: float
    projection: str
    ortho_scale: float | None


@dataclass(frozen=True, slots=True)
class MethodologySpec:
    """Frozen inputs and presentation choices for Figure 2."""

    path: Path
    name: str
    motion: str
    source_frames: tuple[int, ...]
    highlight_frame: int
    target_frame_offset: int
    width: int
    height: int
    include_text: bool
    transparent_background: bool
    panel_width: int
    panel_height: int
    camera_azimuth: float
    camera_elevation: float
    camera_distance: float
    camera_lookat: tuple[float, float, float]
    field_of_view: float
    panel_cameras: tuple[PanelCameraSpec, ...]
    smpl_model: Path
    fitted_shape: Path
    terrain_report: Path


@dataclass(frozen=True, slots=True)
class FitMetrics:
    """Quantitative reconstruction results shown directly in panel C."""

    shared_riser: float
    raw_heights: tuple[float, ...]
    fitted_heights: tuple[float, ...]
    rms_adjustment: float
    max_support_residual: float
    sole_offset: float


@dataclass(frozen=True, slots=True)
class Callout:
    """A vector annotation placed in normalized panel-image coordinates."""

    label: str
    label_xy: tuple[float, float]
    target_xy: tuple[float, float]
    color: tuple[float, float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def load_methodology_spec(path: str | Path) -> MethodologySpec:
    """Load and validate the specialized Figure 2 TOML manifest."""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("version") != 1:
        raise ValueError("methodology figure manifest version must be 1")
    figure = raw.get("figure")
    motion = raw.get("motion")
    camera = raw.get("camera")
    assets = raw.get("assets")
    if not all(isinstance(table, dict) for table in (figure, motion, camera, assets)):
        raise ValueError("figure, motion, camera, and assets must be TOML tables")
    frames_value = motion.get("frames")
    if not isinstance(frames_value, list) or not frames_value:
        raise ValueError("motion.frames must be a non-empty integer array")
    frames = tuple(_integer(value, "motion.frames") for value in frames_value)
    if tuple(sorted(set(frames))) != frames:
        raise ValueError("motion.frames must be strictly increasing")
    highlight = _integer(motion.get("highlight_frame"), "motion.highlight_frame")
    if highlight not in frames:
        raise ValueError("motion.highlight_frame must be included in motion.frames")
    lookat_value = camera.get("lookat")
    if not isinstance(lookat_value, list) or len(lookat_value) != 3:
        raise ValueError("camera.lookat must contain three numbers")
    name = figure.get("name")
    motion_name = motion.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("figure.name must be a non-empty string")
    if not isinstance(motion_name, str) or not motion_name.strip():
        raise ValueError("motion.name must be a non-empty string")
    include_text = figure.get("include_text", True)
    if not isinstance(include_text, bool):
        raise ValueError("figure.include_text must be a boolean")
    transparent_background = figure.get("transparent_background", False)
    if not isinstance(transparent_background, bool):
        raise ValueError("figure.transparent_background must be a boolean")
    camera_distance = _number(camera.get("distance", 4.8), "camera.distance")
    camera_lookat = tuple(_number(value, "camera.lookat") for value in lookat_value)
    field_of_view = _number(camera.get("field_of_view", 42.0), "camera.field_of_view")
    projection_value = camera.get("projection", "perspective")
    if not isinstance(projection_value, str):
        raise ValueError("camera.projection must be a string")
    base_projection = projection_value.casefold()
    if base_projection not in {"perspective", "orthographic"}:
        raise ValueError("camera.projection must be perspective or orthographic")
    base_ortho_scale_value = camera.get("ortho_scale")
    base_ortho_scale = (
        None
        if base_ortho_scale_value is None
        else _number(base_ortho_scale_value, "camera.ortho_scale")
    )
    panel_cameras = []
    for panel in "abcd":
        panel_camera = camera.get(panel, {})
        if not isinstance(panel_camera, dict):
            raise ValueError(f"camera.{panel} must be a TOML table")
        panel_lookat_value = panel_camera.get("lookat", list(camera_lookat))
        if not isinstance(panel_lookat_value, list) or len(panel_lookat_value) != 3:
            raise ValueError(f"camera.{panel}.lookat must contain three numbers")
        panel_projection_value = panel_camera.get("projection", base_projection)
        if not isinstance(panel_projection_value, str):
            raise ValueError(f"camera.{panel}.projection must be a string")
        panel_projection = panel_projection_value.casefold()
        if panel_projection not in {"perspective", "orthographic"}:
            raise ValueError(
                f"camera.{panel}.projection must be perspective or orthographic"
            )
        panel_ortho_scale_value = panel_camera.get("ortho_scale", base_ortho_scale)
        panel_ortho_scale = (
            None
            if panel_ortho_scale_value is None
            else _number(panel_ortho_scale_value, f"camera.{panel}.ortho_scale")
        )
        if panel_projection == "orthographic" and (
            panel_ortho_scale is None or panel_ortho_scale <= 0.0
        ):
            raise ValueError(
                f"camera.{panel}.ortho_scale must be positive for an orthographic camera"
            )
        panel_cameras.append(
            PanelCameraSpec(
                distance=_number(
                    panel_camera.get("distance", camera_distance),
                    f"camera.{panel}.distance",
                ),
                lookat=tuple(
                    _number(value, f"camera.{panel}.lookat") for value in panel_lookat_value
                ),
                field_of_view=_number(
                    panel_camera.get("field_of_view", field_of_view),
                    f"camera.{panel}.field_of_view",
                ),
                projection=panel_projection,
                ortho_scale=panel_ortho_scale,
            )
        )
    return MethodologySpec(
        path=source,
        name=name.strip(),
        motion=motion_name.strip(),
        source_frames=frames,
        highlight_frame=highlight,
        target_frame_offset=int(motion.get("target_frame_offset", 0)),
        width=_integer(figure.get("width", 2100), "figure.width", 512),
        height=_integer(figure.get("height", 2400), "figure.height", 512),
        include_text=include_text,
        transparent_background=transparent_background,
        panel_width=_integer(figure.get("panel_width", 1000), "figure.panel_width", 256),
        panel_height=_integer(figure.get("panel_height", 900), "figure.panel_height", 256),
        camera_azimuth=_number(camera.get("azimuth", 38.0), "camera.azimuth"),
        camera_elevation=_number(camera.get("elevation", -25.0), "camera.elevation"),
        camera_distance=camera_distance,
        camera_lookat=camera_lookat,
        field_of_view=field_of_view,
        panel_cameras=tuple(panel_cameras),
        smpl_model=_path(assets.get("smpl_model"), "assets.smpl_model", source.parent),
        fitted_shape=_path(assets.get("fitted_shape"), "assets.fitted_shape", source.parent),
        terrain_report=_path(assets.get("terrain_report"), "assets.terrain_report", source.parent),
    )


def _panel_camera(spec: MethodologySpec, panel: str) -> PanelCameraSpec:
    if panel not in "abcd" or len(panel) != 1:
        raise ValueError(f"unknown methodology panel: {panel!r}")
    return spec.panel_cameras[ord(panel) - ord("a")]


def _camera_payload(spec: MethodologySpec, panel: str) -> dict[str, object]:
    azimuth = math.radians(spec.camera_azimuth)
    elevation = math.radians(spec.camera_elevation)
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
    camera = _panel_camera(spec, panel)
    lookat = np.asarray(camera.lookat, dtype=float)
    payload = {
        "position": (lookat - camera.distance * forward).tolist(),
        "forward": forward.tolist(),
        "up": up.tolist(),
        "fovy_degrees": camera.field_of_view,
        "projection": camera.projection,
    }
    if camera.ortho_scale is not None:
        payload["ortho_scale"] = camera.ortho_scale
    return payload


def _projection_bounds(
    points: np.ndarray,
    metadata: dict[str, object],
) -> tuple[float, float, float, float]:
    """Project world points to normalized image coordinates for framing audits."""

    camera = metadata["camera"]
    position = np.asarray(camera["position"], dtype=float)
    forward = np.asarray(camera["forward"], dtype=float)
    up = np.asarray(camera["up"], dtype=float)
    right = np.cross(forward, up)
    values = np.asarray(points, dtype=float).reshape(-1, 3) - position
    depth = values @ forward
    if np.any(depth <= 0.0):
        raise ValueError("publication framing audit found geometry behind the camera")
    aspect = float(metadata["width"]) / float(metadata["height"])
    if str(camera.get("projection", "perspective")).casefold() == "orthographic":
        half_width = 0.5 * float(camera["ortho_scale"])
        half_height = half_width / aspect
        normalized_x = 0.5 * (1.0 + (values @ right) / half_width)
        normalized_y = 0.5 * (1.0 - (values @ up) / half_height)
    else:
        tangent = math.tan(math.radians(float(camera["fovy_degrees"])) / 2.0)
        normalized_x = 0.5 * (1.0 + (values @ right) / (depth * tangent * aspect))
        normalized_y = 0.5 * (1.0 - (values @ up) / (depth * tangent))
    return (
        float(np.min(normalized_x)),
        float(np.min(normalized_y)),
        float(np.max(normalized_x)),
        float(np.max(normalized_y)),
    )


def _projection_target(
    point: np.ndarray,
    metadata: dict[str, object],
) -> tuple[float, float]:
    bounds = _projection_bounds(np.asarray(point, dtype=float).reshape(1, 3), metadata)
    return bounds[0], bounds[1]


def _inside_safe_frame(
    bounds: tuple[float, float, float, float],
    margin: float,
) -> bool:
    return (
        bounds[0] >= margin
        and bounds[1] >= margin
        and bounds[2] <= 1.0 - margin
        and bounds[3] <= 1.0 - margin
    )


def _instance_world_vertices(
    arrays: dict[str, np.ndarray],
    *,
    include_history: bool = False,
) -> np.ndarray:
    values = []
    for mesh_id, transform, history in zip(
        arrays["instance_mesh"],
        arrays["instance_transform"],
        arrays["instance_history"],
        strict=True,
    ):
        if bool(history) and not include_history:
            continue
        vertices = arrays[f"mesh_{int(mesh_id)}_vertices"]
        homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1)
        values.append((homogeneous @ transform.T)[:, :3])
    if len(arrays["tendon_endpoints"]):
        values.append(np.asarray(arrays["tendon_endpoints"], dtype=float).reshape(-1, 3))
    if not values:
        raise ValueError("publication framing audit found no highlighted target geometry")
    return np.concatenate(values, axis=0)


def _empty_bundle(spec: MethodologySpec, scale: float, panel: str) -> dict[str, np.ndarray]:
    width = max(64, round(spec.panel_width * scale))
    height = max(64, round(spec.panel_height * scale))
    metadata = {
        "schema_version": 2,
        "width": width,
        "height": height,
        "background": BACKGROUND,
        "transparent_background": spec.transparent_background,
        "camera": _camera_payload(spec, panel),
        "mesh_ids": [],
        "bevel": {"pad": 0.003, "terrain": 0.025, "candidate": 0.006},
        "samples": 64 if scale < 1.0 else 96,
    }
    return {
        "boxes_position": np.asarray(((0.0, 0.0, -0.006),), dtype=np.float32),
        "boxes_size": np.asarray(((6.0, 6.0, 0.006),), dtype=np.float32),
        "boxes_rotation": np.asarray((np.eye(3),), dtype=np.float32),
        "boxes_rgba": np.asarray((PAD_GRAY,), dtype=np.float32),
        "boxes_kind": np.asarray(("pad",)),
        "instance_mesh": np.empty(0, dtype=np.int32),
        "instance_transform": np.empty((0, 4, 4), dtype=np.float32),
        "instance_rgba": np.empty((0, 4), dtype=np.float32),
        "instance_history": np.empty(0, dtype=bool),
        "instance_cell": np.empty(0, dtype=np.int16),
        "instance_frame": np.empty(0, dtype=np.int32),
        "tendon_endpoints": np.empty((0, 2, 3), dtype=np.float32),
        "tendon_radius": np.empty(0, dtype=np.float32),
        "tendon_rgba": np.empty((0, 4), dtype=np.float32),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def _set_overlays(
    arrays: dict[str, np.ndarray],
    points: list[tuple[np.ndarray, float, tuple[float, float, float, float]]],
    segments: list[tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float]]],
) -> None:
    arrays["overlay_points_position"] = np.asarray([entry[0] for entry in points], dtype=np.float32).reshape(-1, 3)
    arrays["overlay_points_radius"] = np.asarray([entry[1] for entry in points], dtype=np.float32)
    arrays["overlay_points_rgba"] = np.asarray([entry[2] for entry in points], dtype=np.float32).reshape(-1, 4)
    arrays["overlay_segments_endpoints"] = np.asarray(
        [[entry[0], entry[1]] for entry in segments], dtype=np.float32
    ).reshape(-1, 2, 3)
    arrays["overlay_segments_radius"] = np.asarray([entry[2] for entry in segments], dtype=np.float32)
    arrays["overlay_segments_rgba"] = np.asarray([entry[3] for entry in segments], dtype=np.float32).reshape(-1, 4)


def _farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministically retain spatial coverage without a dense point cloud."""

    values = np.asarray(points, dtype=float)
    if len(values) <= count:
        return values
    centroid = np.mean(values, axis=0)
    selected = [int(np.argmax(np.linalg.norm(values - centroid, axis=1)))]
    minimum_squared_distance = np.sum((values - values[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(minimum_squared_distance))
        selected.append(index)
        squared_distance = np.sum((values - values[index]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, squared_distance)
    return values[np.asarray(selected)]


def _world_transform(
    anchor: np.ndarray,
    yaw_degrees: float,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    rotation = rotation_z(yaw_degrees)

    def transform(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float).copy()
        values[..., :2] -= anchor
        return values @ rotation.T

    return rotation, transform


def _source_geometry(
    spec: MethodologySpec,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], float, dict[str, object]]:
    """Evaluate exact fitted-shape SMPL-H meshes and normalized landmarks."""

    from loco_mujoco.smpl import SMPLH_Parser
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS, demo_joint_permutation

    motion = load_smplh_motion(source_path)
    robot_height = get_robot_height(str(spec.fitted_shape), str(spec.smpl_model))
    joints, _rotations, fps, normalization = normalized_motion_landmarks(
        motion,
        str(spec.smpl_model),
        str(spec.fitted_shape),
        robot_height,
        use_fitted_shape=True,
        calibrate_sites=True,
        return_normalization=True,
    )
    shape, shape_scale, *_ = joblib.load(spec.fitted_shape)
    fitted_betas = shape.detach() if hasattr(shape, "detach") else torch.as_tensor(shape)
    fitted_betas = fitted_betas.reshape(1, -1).float()
    neutral_betas = torch.zeros_like(fitted_betas)
    frames = np.asarray(spec.source_frames, dtype=int)
    pose = torch.as_tensor(np.asarray(motion["pose_aa"])[frames]).float()
    pose = torch.cat((pose, torch.zeros((len(pose), 156 - pose.shape[1]))), dim=1)
    translation = torch.as_tensor(np.asarray(motion["trans"])[frames]).float()
    parser = SMPLH_Parser(model_path=str(spec.smpl_model), gender="neutral")
    with torch.no_grad():
        vertices, _ = parser.get_joints_verts(
            pose,
            th_betas=neutral_betas.repeat(len(frames), 1),
            th_trans=translation,
        )
        transforms = parser.get_joint_transformations(
            pose.reshape(len(frames), -1, 3),
            neutral_betas.repeat(len(frames), 1),
            translation,
        )
    vertices = vertices.detach().cpu().numpy()
    transforms = transforms.detach().cpu().numpy()
    neutral_joints = transforms[..., :3, 3][:, demo_joint_permutation(), :]
    neutral_root = neutral_joints[:, :1]
    fitted_scale = float(np.asarray(shape_scale.detach() if hasattr(shape_scale, "detach") else shape_scale).reshape(-1)[0])
    fitted_root = joints[frames, :1]
    vertices = (vertices - neutral_root) * fitted_scale
    vertices *= float(normalization["uniform_scale"])
    vertices += fitted_root
    normalization = dict(normalization)
    normalization["display_body_shape"] = "SMPL-H neutral template (zero beta)"
    return (
        vertices,
        np.asarray(parser.faces, dtype=np.int32),
        joints,
        tuple(SMPLH_DEMO_JOINTS),
        float(fps),
        normalization,
    )


def _box_edges(center: np.ndarray, half_size: np.ndarray, rotation: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    corners = signs * half_size
    corners = corners @ rotation.T + center
    edges = []
    for left in range(8):
        for right in range(left + 1, 8):
            if int(np.count_nonzero(signs[left] != signs[right])) == 1:
                edges.append((corners[left], corners[right]))
    return edges


def _terrain_boxes(terrain, transform, global_rotation: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    positions, sizes, rotations = [], [], []
    for box in terrain.boxes:
        cosine, sine = math.cos(float(box.pitch)), math.sin(float(box.pitch))
        pitch = np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))
        box_rotation = rotation_z(math.degrees(float(box.yaw))) @ pitch
        positions.append(transform(np.asarray(box.pos, dtype=float)))
        sizes.append(np.asarray(box.size, dtype=float))
        rotations.append(global_rotation @ box_rotation)
    return positions, sizes, rotations


def _motion_sweep_evidence(
    transformed_joints: np.ndarray,
    name_index: dict[str, int],
    frame_range: tuple[int, int],
) -> tuple[
    list[tuple[np.ndarray, float, tuple[float, float, float, float]]],
    list[tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float]]],
]:
    """Trace the measured body sweep that supplies negative free-space evidence.

    Continuous trajectories are easier to interpret than an unstructured sample
    cloud: every orange curve is an observed landmark path, and the intervening
    volume therefore cannot be occupied by a reconstructed terrain surface.
    """

    evidence_joints = (
        "Pelvis",
        "L_Knee",
        "R_Knee",
        "L_Ankle",
        "R_Ankle",
        "L_Toe",
        "R_Toe",
        "L_Wrist",
        "R_Wrist",
    )
    first, last = frame_range
    sampled_frames = np.arange(first, last + 1, 6, dtype=int)
    curve_color = (*FREE_SPACE_CORAL[:3], 0.32)
    point_color = (*FREE_SPACE_CORAL[:3], 0.62)
    points = []
    segments = []
    for name in evidence_joints:
        path = transformed_joints[sampled_frames, name_index[name]]
        radius = 0.006 if name in {"L_Ankle", "R_Ankle", "L_Toe", "R_Toe"} else 0.0045
        segments.extend(
            (left, right, radius, curve_color)
            for left, right in pairwise(path)
        )
        points.extend((point, 0.011, point_color) for point in path[::4])
    return points, segments


def _target_manifest(
    spec: MethodologySpec,
    anchor: np.ndarray,
    yaw_degrees: float,
    scale: float,
) -> FigureManifest:
    target_frames = tuple(frame + spec.target_frame_offset for frame in spec.source_frames)
    target_highlight = spec.highlight_frame + spec.target_frame_offset
    if target_highlight < 0:
        raise ValueError(f"target frame offset produces a negative frame: {target_highlight}")
    if target_frames[0] < 0:
        raise ValueError(
            f"target frame offset produces a negative frame: {target_frames[0]}"
        )
    camera = _panel_camera(spec, "d")
    layout = LayoutSpec(
        rows=1,
        columns=1,
        width=max(64, round(spec.panel_width * scale)),
        height=max(64, round(spec.panel_height * scale)),
        cell_size=(12.0, 12.0),
        cell_pitch=(12.0, 12.0),
        pad_thickness=0.012,
        decorative_border=0,
        camera_azimuth=spec.camera_azimuth,
        camera_elevation=spec.camera_elevation,
        camera_distance=camera.distance,
        camera_lookat_z=camera.lookat[2],
        field_of_view=camera.field_of_view,
        camera_lookat_x=camera.lookat[0],
        camera_lookat_y=camera.lookat[1],
        horizon_extent=0.0,
    )
    style = StyleSpec(
        background=BACKGROUND,
        pad=PAD_GRAY,
        terrain=TERRAIN_GRAY,
        history=(0.30, 0.40, 0.48),
        history_alpha=(0.12, 0.20, 0.30, 0.42),
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
    )
    return FigureManifest(
        version=1,
        name=f"{spec.name}-target-panel",
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
                label="terrain-aware retargeting",
                frames=target_frames,
                fractions=None,
                yaw_degrees=yaw_degrees,
                anchor_xy=tuple(float(value) for value in anchor),
                highlight_frame=target_highlight,
            ),
        ),
        path=spec.path,
    )


def _target_landmarks(
    manifest: FigureManifest,
    cache_root: Path,
    transform,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model = _model(manifest, manifest.layout.width, manifest.layout.height)
    trajectory_path, _terrain_path, _analysis_path = _artifact_paths(
        manifest, cache_root, manifest.cells[0].motion
    )
    from loco_mujoco.trajectory import Trajectory

    qpos = np.asarray(Trajectory.load(str(trajectory_path)).data.qpos, dtype=float)
    frame = int(manifest.cells[0].highlight_frame)
    if not 0 <= frame < len(qpos):
        raise ValueError(f"target highlight frame {frame} lies outside {len(qpos)} frames")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(model, data)
    by_name = {}
    for source_name, body_name in SMPLH_TO_MYOFULLBODY.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MyoFullBody body not found for landmark {source_name}: {body_name}")
        by_name[source_name] = transform(np.asarray(data.xpos[body_id], dtype=float))
    return np.stack([by_name[name] for name in SMPLH_TO_MYOFULLBODY]), by_name


def _minimum_spanning_edges(points: np.ndarray) -> list[tuple[int, int]]:
    """Return a deterministic Euclidean backbone connecting every landmark."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("interaction landmarks must have shape (N, 3)")
    if len(values) < 2:
        return []
    connected = {0}
    edges = []
    while len(connected) < len(values):
        _distance, left, right = min(
            (
                float(np.linalg.norm(values[left] - values[right])),
                left,
                right,
            )
            for left in connected
            for right in range(len(values))
            if right not in connected
        )
        edges.append(tuple(sorted((left, right))))
        connected.add(right)
    return edges


def _interaction_graph(
    terrain,
    source_landmarks: np.ndarray,
    transform,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Build the exact source/scene neighborhood graph used by retargeting."""

    scene = terrain_scene(
        terrain,
        n_frames,
        spacing=0.20,
        ground_range=(-3.0, 3.0),
        ground_size=8,
    )
    scene_points = transform(np.asarray(scene["object_points_local"], dtype=float))
    vertices = np.concatenate((source_landmarks, scene_points), axis=0)
    tetrahedra = Delaunay(vertices, qhull_options="QJ").simplices
    edges = set()
    for tetrahedron in tetrahedra:
        for left_index in range(4):
            for right_index in range(left_index + 1, 4):
                edge = tuple(
                    sorted(
                        (
                            int(tetrahedron[left_index]),
                            int(tetrahedron[right_index]),
                        )
                    )
                )
                if edge[0] < len(source_landmarks):
                    edges.add(edge)
    candidates = sorted(
        (
            edge
            for edge in edges
            if np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]) <= 1.15
        ),
        key=lambda edge: float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])),
    )
    spanning_edges = _minimum_spanning_edges(source_landmarks)
    body_edges = [edge for edge in candidates if edge[1] < len(source_landmarks)][:48]
    scene_edges = [edge for edge in candidates if edge[1] >= len(source_landmarks)][:48]
    selected_edges = sorted(
        {*spanning_edges, *body_edges, *scene_edges},
        key=lambda edge: float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])),
    )
    scene_indices = sorted(
        {
            index - len(source_landmarks)
            for edge in selected_edges
            for index in edge
            if index >= len(source_landmarks)
        }
    )
    display_scene = scene_points[np.asarray(scene_indices, dtype=int)]
    return scene_points, vertices, selected_edges, display_scene


def _npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _write_bundle(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _source_colors(spec: MethodologySpec) -> np.ndarray:
    alphas = np.linspace(0.18, 0.34, max(len(spec.source_frames) - 1, 1))
    colors = []
    ghost_index = 0
    for frame in spec.source_frames:
        if frame == spec.highlight_frame:
            colors.append(SOURCE_NEUTRAL)
        else:
            colors.append((*SOURCE_GHOST, float(alphas[min(ghost_index, len(alphas) - 1)])))
            ghost_index += 1
    return np.asarray(colors, dtype=np.float32)


def build_methodology_bundles(
    spec: MethodologySpec,
    *,
    amass_root: str | Path,
    cache_root: str | Path,
    output_directory: str | Path,
    scale: float,
) -> dict[str, object]:
    """Build the four Blender bundles from one source/terrain/target triplet."""

    if not 0 < scale <= 1.0:
        raise ValueError(f"scale must lie in (0, 1], got {scale}")
    amass = Path(amass_root).expanduser().resolve()
    cache = Path(cache_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    source_path = amass / f"{spec.motion}.npz"
    if not source_path.is_file():
        raise FileNotFoundError(f"source SMPL-H motion not found: {source_path}")
    for asset in (spec.smpl_model, spec.fitted_shape, spec.terrain_report):
        if not asset.exists():
            raise FileNotFoundError(f"methodology figure asset not found: {asset}")

    from loco_mujoco.core.terrain import TerrainSpec

    target_stub = _target_manifest(spec, np.zeros(2), 0.0, scale)
    _trajectory_path, terrain_path, analysis_path = _artifact_paths(target_stub, cache, spec.motion)
    terrain = TerrainSpec.load(str(terrain_path))
    report = json.loads(spec.terrain_report.read_text())
    if report.get("motion") != spec.motion:
        raise ValueError(f"terrain report is for {report.get('motion')!r}, expected {spec.motion!r}")
    if not terrain.boxes:
        raise ValueError("methodology figure requires a reconstructed non-flat terrain")

    anchor = np.mean([np.asarray(box.pos[:2], dtype=float) for box in terrain.boxes], axis=0)
    yaw_degrees = -math.degrees(float(terrain.boxes[0].yaw))
    global_rotation, transform = _world_transform(anchor, yaw_degrees)
    vertices, faces, joints, joint_names, fps, normalization = _source_geometry(spec, source_path)
    transformed_vertices = transform(vertices)
    transformed_joints = transform(joints)
    selected_index = {frame: index for index, frame in enumerate(spec.source_frames)}
    highlight_source = transformed_vertices[selected_index[spec.highlight_frame]]
    name_index = {name: index for index, name in enumerate(joint_names)}

    events = detect_stance_events(joints, joint_names, fps)
    offsets = {
        str(name): float(value)
        for name, value in report["terrain"]["provenance"]["joint_surface_offsets_m"].items()
    }
    support_points = []
    support_segments = []
    for event in events:
        xy = np.median(event.xy, axis=0)
        surface = np.asarray((xy[0], xy[1], event.z - offsets.get(event.joint, 0.0)))
        middle = min((event.start + event.end) // 2, len(joints) - 1)
        probe = joints[middle, name_index[event.joint]]
        support_points.append(transform(surface))
        support_segments.append((transform(surface), transform(probe)))

    terrain_positions, terrain_sizes, terrain_rotations = _terrain_boxes(
        terrain, transform, global_rotation
    )
    terrain_edges = [
        edge
        for position, size, rotation in zip(
            terrain_positions, terrain_sizes, terrain_rotations, strict=True
        )
        for edge in _box_edges(position, size, rotation)
    ]
    highlight_joints = transformed_joints[spec.highlight_frame]
    source_landmarks = np.stack(
        [highlight_joints[name_index[name]] for name in SMPLH_TO_MYOFULLBODY]
    )
    scene_points, interaction_vertices, graph_edges, display_scene = _interaction_graph(
        terrain,
        source_landmarks,
        transform,
        len(joints),
    )

    panels: dict[str, Path] = {}

    # Panel A: exact source meshes, paired landmarks, contact probes, and root path.
    panel_a = _empty_bundle(spec, scale, "a")
    panel_a["source_vertices"] = transformed_vertices.astype(np.float32)
    panel_a["source_faces"] = faces
    panel_a["source_rgba"] = _source_colors(spec)
    points_a = []
    for name in SMPLH_TO_MYOFULLBODY:
        points_a.append((highlight_joints[name_index[name]], 0.025, LANDMARK_BLUE))
    for name in DEFAULT_CONTACT_JOINTS:
        points_a.append((highlight_joints[name_index[name]], 0.041, SUPPORT_CYAN))
    pelvis_path = transformed_joints[::10, name_index["Pelvis"]]
    segments_a = [
        (left, right, 0.008, (*LANDMARK_BLUE[:3], 0.46))
        for left, right in pairwise(pelvis_path)
    ]
    _set_overlays(panel_a, points_a, segments_a)
    panels["a"] = output / "a-motion.npz"
    _write_bundle(panels["a"], panel_a)

    # Panel B: measured body-sweep/free-space trajectories, stance supports, and reconstruction.
    panel_b = _empty_bundle(spec, scale, "b")
    sweep_points, sweep_segments = _motion_sweep_evidence(
        transformed_joints,
        name_index,
        (spec.source_frames[0], spec.source_frames[-1]),
    )
    points_b = list(sweep_points)
    points_b.extend((point, 0.043, SUPPORT_CYAN) for point in support_points)
    segments_b = list(sweep_segments)
    segments_b.extend(
        (surface, probe, 0.008, (*SUPPORT_CYAN[:3], 0.82))
        for surface, probe in support_segments
    )
    segments_b.extend(
        (left, right, 0.010, (*FIT_BLUE[:3], 0.94)) for left, right in terrain_edges
    )
    _set_overlays(panel_b, points_b, segments_b)
    panels["b"] = output / "b-reconstruction-evidence.npz"
    _write_bundle(panels["b"], panel_b)

    # Panel C: source landmarks connected to the sampled terrain interaction graph.
    panel_c = _empty_bundle(spec, scale, "c")
    panel_c["boxes_position"] = np.concatenate(
        (panel_c["boxes_position"], np.asarray(terrain_positions, dtype=np.float32))
    )
    panel_c["boxes_size"] = np.concatenate(
        (panel_c["boxes_size"], np.asarray(terrain_sizes, dtype=np.float32))
    )
    panel_c["boxes_rotation"] = np.concatenate(
        (panel_c["boxes_rotation"], np.asarray(terrain_rotations, dtype=np.float32))
    )
    terrain_colors = np.asarray(
        [
            (0.43 + 0.035 * index, 0.47 + 0.035 * index, 0.49 + 0.035 * index, 1.0)
            for index in range(len(terrain.boxes))
        ],
        dtype=np.float32,
    )
    panel_c["boxes_rgba"] = np.concatenate((panel_c["boxes_rgba"], terrain_colors))
    panel_c["boxes_kind"] = np.concatenate(
        (panel_c["boxes_kind"], np.asarray(["terrain"] * len(terrain.boxes)))
    )
    panel_c["source_vertices"] = highlight_source[None].astype(np.float32)
    panel_c["source_faces"] = faces
    panel_c["source_rgba"] = np.asarray(((*SOURCE_NEUTRAL[:3], 0.50),), dtype=np.float32)
    interaction_color = (*INTERACTION_TEAL[:3], 0.78)
    points_c = [(point, 0.019, interaction_color) for point in display_scene]
    points_c.extend((point, 0.026, LANDMARK_BLUE) for point in source_landmarks)
    segments_c = [
        (left, right, 0.008, (*FIT_BLUE[:3], 0.76)) for left, right in terrain_edges
    ]
    segments_c.extend(
        (
            interaction_vertices[left],
            interaction_vertices[right],
            0.007,
            interaction_color,
        )
        for left, right in graph_edges
    )
    _set_overlays(panel_c, points_c, segments_c)
    panels["c"] = output / "c-interaction-mesh.npz"
    _write_bundle(panels["c"], panel_c)

    # Panel D: exact retargeted MyoFullBody sequence and highlighted contact constraints.
    target_manifest = _target_manifest(spec, anchor, yaw_degrees, scale)
    target_bundle_path = output / "d-retargeting.npz"
    export_scene_bundle(
        target_manifest,
        cache_root=cache,
        output=target_bundle_path,
        scale=1.0,
    )
    panel_d = _npz_dict(target_bundle_path)
    target_metadata = json.loads(str(panel_d["metadata"]))
    target_metadata["schema_version"] = 2
    target_metadata["samples"] = 64 if scale < 1.0 else 96
    target_metadata["camera"] = _camera_payload(spec, "d")
    target_metadata["transparent_background"] = spec.transparent_background
    panel_d["metadata"] = np.asarray(json.dumps(target_metadata, sort_keys=True))
    _target_landmark_array, target_by_name = _target_landmarks(
        target_manifest,
        cache,
        transform,
    )
    correspondence_names = (
        "Pelvis",
        "Head",
        "L_Wrist",
        "R_Wrist",
        "L_Ankle",
        "R_Ankle",
        "L_Toe",
        "R_Toe",
    )
    points_d = [(target_by_name[name], 0.021, LANDMARK_BLUE) for name in correspondence_names]
    segments_d = []
    for name in ("R_Ankle", "R_Toe"):
        target = target_by_name[name]
        original_xy = global_rotation.T @ target
        original_xy[:2] += anchor
        support = np.asarray(
            (original_xy[0], original_xy[1], float(terrain.height_at(original_xy[0], original_xy[1])))
        )
        support = transform(support)
        points_d.append((support, 0.034, CONSTRAINT_GREEN))
        segments_d.append((support, target, 0.010, CONSTRAINT_GREEN))
    swing = target_by_name["L_Toe"]
    original_swing = global_rotation.T @ swing
    original_swing[:2] += anchor
    below = np.asarray(
        (
            original_swing[0],
            original_swing[1],
            float(terrain.height_at(original_swing[0], original_swing[1])),
        )
    )
    below = transform(below)
    points_d.append((below, 0.027, CLEARANCE_AMBER))
    segments_d.append((below, swing, 0.009, CLEARANCE_AMBER))
    _set_overlays(panel_d, points_d, segments_d)
    _write_bundle(target_bundle_path, panel_d)
    panels["d"] = target_bundle_path

    safe_margin = 0.025
    panel_a_metadata = json.loads(str(panel_a["metadata"]))
    panel_d_metadata = json.loads(str(panel_d["metadata"]))
    source_bounds = _projection_bounds(transformed_vertices, panel_a_metadata)
    target_bounds = _projection_bounds(
        _instance_world_vertices(panel_d, include_history=True),
        panel_d_metadata,
    )
    framing_a_safe = _inside_safe_frame(source_bounds, safe_margin)
    framing_d_safe = _inside_safe_frame(target_bounds, safe_margin)
    if not framing_a_safe or not framing_d_safe:
        raise ValueError(
            "publication framing audit failed: "
            f"source={source_bounds}, target={target_bounds}, margin={safe_margin}"
        )
    annotation_targets = {}
    sole_joint = "R_Ankle"
    sole_offset = float(offsets[sole_joint])

    return {
        "panels": {name: str(path) for name, path in panels.items()},
        "source": str(source_path),
        "terrain": str(terrain_path),
        "analysis": str(analysis_path),
        "anchor_xy": anchor.tolist(),
        "yaw_degrees": yaw_degrees,
        "source_frames": list(spec.source_frames),
        "target_frames": list(target_manifest.cells[0].frames),
        "highlight_frame": spec.highlight_frame,
        "stance_events": len(events),
        "free_space_trajectory_joints": 9,
        "free_space_points_displayed": len(sweep_points),
        "free_space_segments_displayed": len(sweep_segments),
        "sole_offset_joint": sole_joint,
        "sole_offset_m": sole_offset,
        "interaction_scene_points": len(scene_points),
        "interaction_scene_points_displayed": len(display_scene),
        "interaction_edges_displayed": len(graph_edges),
        "interaction_body_edges_displayed": sum(
            right < len(source_landmarks) for _left, right in graph_edges
        ),
        "interaction_scene_edges_displayed": sum(
            right >= len(source_landmarks) for _left, right in graph_edges
        ),
        "interaction_landmark_backbone_edges": len(source_landmarks) - 1,
        "framing_audit": {
            "safe_margin_fraction": safe_margin,
            "source_trail_bounds": source_bounds,
            "source_trail_safe": framing_a_safe,
            "target_trail_bounds": target_bounds,
            "target_trail_safe": framing_d_safe,
        },
        "annotation_targets": annotation_targets,
        "normalization": normalization,
    }


def _resolve_blender(executable: str | Path) -> Path:
    value = str(executable)
    resolved = shutil.which(value)
    path = Path(resolved) if resolved is not None else Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Blender executable not found: {path}")
    return path


def _render_bundle(
    bundle: Path,
    output: Path,
    report: Path,
    blender: Path,
) -> dict[str, object]:
    scene_script = Path(__file__).with_name("blender_scene.py")
    with tempfile.TemporaryDirectory(prefix="terra-methodology-panel-") as temporary_directory:
        temporary = Path(temporary_directory)
        rendered = temporary / "render.png"
        renderer_report = temporary / "renderer.json"
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(scene_script),
            "--",
            "--bundle",
            str(bundle),
            "--out",
            str(rendered),
            "--report",
            str(renderer_report),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not rendered.is_file() or not renderer_report.is_file():
            detail = (completed.stdout + "\n" + completed.stderr).strip()
            raise RuntimeError(
                f"Blender panel render failed with exit code {completed.returncode}:\n{detail[-8000:]}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(output, lambda destination: shutil.copyfile(rendered, destination))
        payload = json.loads(renderer_report.read_text())
        payload["output"] = str(output)
        payload["sha256"] = _sha256(output)
        atomic_write(
            report,
            lambda destination: destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            ),
        )
        return payload


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    return ImageFont.truetype(str(path), size) if path is not None else ImageFont.load_default()


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    tip: tuple[int, int],
    direction: tuple[float, float],
    size: int,
    color: tuple[int, int, int],
) -> None:
    vector = np.asarray(direction, dtype=float)
    vector /= np.linalg.norm(vector)
    perpendicular = np.asarray((-vector[1], vector[0]))
    tip_value = np.asarray(tip, dtype=float)
    base = tip_value - vector * size
    polygon = [tip_value, base + perpendicular * size * 0.46, base - perpendicular * size * 0.46]
    draw.polygon([tuple(np.rint(point).astype(int)) for point in polygon], fill=color)


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGBA")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGBA", size, (0, 0, 0, 0))
    result.alpha_composite(
        copy,
        ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2),
    )
    return result


def _fit_metrics(report: dict[str, object]) -> FitMetrics:
    stair_flight = report["fit"]["stair_flight"]
    height_model = stair_flight["height_model"]
    score = report["fit"]["model_scores"]["stair_flight"]
    sole_offset = report["terrain"]["provenance"]["joint_surface_offsets_m"]["R_Ankle"]
    return FitMetrics(
        shared_riser=float(height_model["shared_riser"]),
        raw_heights=tuple(float(value) for value in height_model["raw_heights"]),
        fitted_heights=tuple(float(value) for value in height_model["fitted_heights"]),
        rms_adjustment=float(height_model["rms_adjustment"]),
        max_support_residual=float(score["raised_contact_error_max"]),
        sole_offset=float(sole_offset),
    )


def _panel_titles() -> dict[str, str]:
    return {
        "a": "(a) Motion representation",
        "b": "(b) Terrain reconstruction",
        "c": "(c) Interaction mesh",
        "d": "(d) Retargeted motion",
    }


def _panel_callouts(
    metrics: FitMetrics,
    annotation_targets: dict[str, tuple[float, float]],
) -> dict[str, tuple[Callout, ...]]:
    del metrics, annotation_targets

    return {
        "a": (
            Callout("paired joints", (0.04, 0.10), (0.55, 0.28), LANDMARK_BLUE),
            Callout("root path", (0.04, 0.69), (0.40, 0.61), LANDMARK_BLUE),
        ),
        "b": (
            Callout("body-sweep evidence", (0.04, 0.10), (0.46, 0.31), FREE_SPACE_CORAL),
            Callout("stance support", (0.04, 0.80), (0.30, 0.83), SUPPORT_CYAN),
            Callout("fitted stairs", (0.58, 0.73), (0.70, 0.64), FIT_BLUE),
        ),
        "c": (
            Callout("interaction edges", (0.03, 0.10), (0.53, 0.44), INTERACTION_TEAL),
            Callout("scene samples", (0.57, 0.76), (0.55, 0.62), INTERACTION_TEAL),
        ),
        "d": (
            Callout("retargeted trail", (0.03, 0.11), (0.35, 0.53), LANDMARK_BLUE),
            Callout("foot constraints", (0.54, 0.73), (0.54, 0.57), CONSTRAINT_GREEN),
        ),
    }


def _rect_xy(
    rectangle: tuple[int, int, int, int],
    normalized: tuple[float, float],
) -> tuple[int, int]:
    left, top, right, bottom = rectangle
    return (
        round(left + normalized[0] * (right - left)),
        round(top + normalized[1] * (bottom - top)),
    )


def _rgb(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    return tuple(round(value * 255) for value in rgba[:3])


def _draw_callout_png(
    draw: ImageDraw.ImageDraw,
    rectangle: tuple[int, int, int, int],
    callout: Callout,
    *,
    scale: float,
) -> None:
    font = _font(max(round(44 * scale), 14), bold=True)
    label = _rect_xy(rectangle, callout.label_xy)
    target = _rect_xy(rectangle, callout.target_xy)
    swatch = max(round(8 * scale), 3)
    gap = max(round(10 * scale), 4)
    color = _rgb(callout.color)
    center = (label[0] + swatch // 2, label[1] + max(round(22 * scale), 7))
    line_width = max(round(3 * scale), 1)
    draw.line((center, target), fill=color, width=line_width)
    radius = max(round(5 * scale), 2)
    draw.ellipse(
        (target[0] - radius, target[1] - radius, target[0] + radius, target[1] + radius),
        fill=color,
    )
    draw.rectangle(
        (
            label[0],
            label[1] + max(round(5 * scale), 2),
            label[0] + swatch,
            label[1] + max(round(39 * scale), 13),
        ),
        fill=color,
    )
    draw.text(
        (label[0] + swatch + gap, label[1]),
        callout.label,
        font=font,
        fill=(35, 40, 42),
        stroke_width=max(round(5 * scale), 2),
        stroke_fill=(250, 250, 247),
    )


def _draw_sole_offset_inset_png(
    draw: ImageDraw.ImageDraw,
    rectangle: tuple[int, int, int, int],
    metrics: FitMetrics,
    target: tuple[float, float],
    *,
    scale: float,
) -> None:
    left, top, right, bottom = rectangle
    width, height = right - left, bottom - top
    x = round(left + 0.035 * width)
    y = round(top + 0.035 * height)
    inset_width = round(0.47 * width)
    inset_height = round(0.31 * height)
    inset = (x, y, x + inset_width, y + inset_height)
    target_xy = _rect_xy(rectangle, target)
    leader_start = (inset[2], round(inset[1] + 0.78 * inset_height))
    leader_color = _rgb(LANDMARK_BLUE)
    leader_width = max(round(3 * scale), 1)
    draw.line((leader_start, target_xy), fill=leader_color, width=leader_width)
    target_radius = max(round(6 * scale), 2)
    draw.ellipse(
        (
            target_xy[0] - target_radius,
            target_xy[1] - target_radius,
            target_xy[0] + target_radius,
            target_xy[1] + target_radius,
        ),
        fill=leader_color,
        outline=(250, 250, 247),
        width=max(round(2 * scale), 1),
    )
    draw.rounded_rectangle(
        inset,
        radius=max(round(12 * scale), 4),
        fill=(250, 250, 247),
        outline=(143, 149, 151),
        width=max(round(2 * scale), 1),
    )
    title_font = _font(max(round(38 * scale), 13), bold=True)
    label_font = _font(max(round(26 * scale), 9), bold=True)
    note_font = _font(max(round(23 * scale), 8))
    draw.text(
        (x + round(18 * scale), y + round(10 * scale)),
        "sole-offset calibration",
        font=title_font,
        fill=(35, 40, 42),
    )
    draw.text(
        (x + round(18 * scale), y + round(47 * scale)),
        "R_Ankle · magnified view",
        font=note_font,
        fill=(75, 80, 82),
    )
    diagram_top = y + round(82 * scale)
    diagram_bottom = inset[3] - round(27 * scale)
    probe_x = x + round(0.29 * inset_width)
    bracket_x = x + round(0.46 * inset_width)
    ankle_y = diagram_top + round(0.20 * (diagram_bottom - diagram_top))
    sole_y = diagram_top + round(0.70 * (diagram_bottom - diagram_top))
    surface_left = x + round(0.12 * inset_width)
    surface_right = x + round(0.88 * inset_width)
    cap = max(round(10 * scale), 4)
    line_width = max(round(4 * scale), 2)
    draw.line(
        (surface_left, sole_y, surface_right, sole_y),
        fill=_rgb(SUPPORT_CYAN),
        width=max(round(7 * scale), 3),
    )
    draw.line((bracket_x, ankle_y, bracket_x, sole_y), fill=(53, 59, 61), width=line_width)
    draw.line((bracket_x - cap, ankle_y, bracket_x + cap, ankle_y), fill=(53, 59, 61), width=line_width)
    draw.line((bracket_x - cap, sole_y, bracket_x + cap, sole_y), fill=(53, 59, 61), width=line_width)
    point_radius = max(round(10 * scale), 4)
    for point_y, color in ((ankle_y, LANDMARK_BLUE), (sole_y, SUPPORT_CYAN)):
        draw.ellipse(
            (
                probe_x - point_radius,
                point_y - point_radius,
                probe_x + point_radius,
                point_y + point_radius,
            ),
            fill=_rgb(color),
            outline=(250, 250, 247),
            width=max(round(3 * scale), 1),
        )
    text_x = bracket_x + round(18 * scale)
    draw.text(
        (text_x, round((ankle_y + sole_y) / 2 - 18 * scale)),
        f"δ = {1000.0 * metrics.sole_offset:.0f} mm",
        font=label_font,
        fill=(35, 40, 42),
    )
    draw.text(
        (probe_x + round(16 * scale), ankle_y - round(15 * scale)),
        "ankle landmark",
        font=note_font,
        fill=_rgb(LANDMARK_BLUE),
    )
    draw.text(
        (surface_left, sole_y + round(8 * scale)),
        "sole / support surface",
        font=note_font,
        fill=(42, 111, 119),
    )


def _compose_png(
    spec: MethodologySpec,
    panel_images: dict[str, Path],
    output: Path,
    *,
    scale: float,
    metrics: FitMetrics,
    annotation_targets: dict[str, tuple[float, float]],
) -> tuple[
    dict[str, tuple[int, int, int, int]],
    dict[str, tuple[int, int, int, int]],
]:
    width = round(spec.width * scale)
    height = round(spec.height * scale)
    background = tuple(round(value * 255) for value in BACKGROUND)
    canvas_background = (0, 0, 0, 0) if spec.transparent_background else (*background, 255)
    canvas = Image.new("RGBA", (width, height), canvas_background)
    draw = ImageDraw.Draw(canvas)
    margin = round(58 * scale)
    column_gap = round(72 * scale)
    row_gap = round(86 * scale)
    panel_width = (width - 2 * margin - column_gap) // 2
    panel_height = (height - 2 * margin - row_gap) // 2
    header_height = round(82 * scale) if spec.include_text else 0
    inner = round(14 * scale)
    image_top_offset = header_height if spec.include_text else inner
    image_size = (panel_width - 2 * inner, panel_height - image_top_offset - inner)
    boxes = {
        "a": (margin, margin),
        "b": (margin + panel_width + column_gap, margin),
        "c": (margin, margin + panel_height + row_gap),
        "d": (margin + panel_width + column_gap, margin + panel_height + row_gap),
    }
    titles = _panel_titles()
    title_font = _font(max(round(54 * scale), 18), bold=True)
    panel_rectangles = {}
    content_rectangles = {}
    for key, (left, top) in boxes.items():
        right, bottom = left + panel_width, top + panel_height
        panel_rectangles[key] = (left, top, right, bottom)
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=round(18 * scale),
            fill=None if spec.transparent_background else (250, 250, 247, 255),
            outline=(176, 181, 182),
            width=max(round(2 * scale), 1),
        )
        if spec.include_text:
            draw.text(
                (left + round(22 * scale), top + round(10 * scale)),
                titles[key],
                font=title_font,
                fill=(31, 36, 38),
            )
        image = _fit_panel(Image.open(panel_images[key]), image_size)
        image_left, image_top = left + inner, top + image_top_offset
        canvas.alpha_composite(image, (image_left, image_top))
        content_rectangles[key] = (
            image_left,
            image_top,
            image_left + image_size[0],
            image_top + image_size[1],
        )

    if spec.include_text:
        callouts = _panel_callouts(metrics, annotation_targets)
        for key, entries in callouts.items():
            for callout in entries:
                _draw_callout_png(draw, content_rectangles[key], callout, scale=scale)

    arrow_color = (57, 67, 71)
    line_width = max(round(5 * scale), 2)
    head = max(round(20 * scale), 8)
    for row in ("a", "c"):
        left_box = panel_rectangles[row]
        right_box = panel_rectangles["b" if row == "a" else "d"]
        y = (left_box[1] + left_box[3]) // 2
        start = (left_box[2] + round(10 * scale), y)
        end = (right_box[0] - round(10 * scale), y)
        draw.line((start, end), fill=arrow_color, width=line_width)
        _arrow_head(draw, end, (1.0, 0.0), head, arrow_color)

    top_right = panel_rectangles["b"]
    bottom_left = panel_rectangles["c"]
    start = ((top_right[0] + top_right[2]) // 2, top_right[3] + round(5 * scale))
    end = ((bottom_left[0] + bottom_left[2]) // 2, bottom_left[1] - round(5 * scale))
    middle_y = (start[1] + end[1]) // 2
    path = [start, (start[0], middle_y), (end[0], middle_y), end]
    draw.line(path, fill=arrow_color, width=line_width, joint="curve")
    _arrow_head(draw, end, (0.0, 1.0), head, arrow_color)

    final_canvas = canvas if spec.transparent_background else canvas.convert("RGB")
    atomic_write(output, lambda destination: final_canvas.save(destination, format="PNG"))
    return panel_rectangles, content_rectangles


def _svg_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _hex_color(rgba: tuple[float, float, float, float]) -> str:
    return "#" + "".join(f"{round(value * 255):02x}" for value in rgba[:3])


def _svg_callouts(
    content_rectangles: dict[str, tuple[int, int, int, int]],
    metrics: FitMetrics,
    annotation_targets: dict[str, tuple[float, float]],
    *,
    scale: float,
) -> str:
    font_size = 44 * scale
    swatch = 8 * scale
    gap = 10 * scale
    elements = []
    for key, entries in _panel_callouts(metrics, annotation_targets).items():
        rectangle = content_rectangles[key]
        for callout in entries:
            label_x, label_y = _rect_xy(rectangle, callout.label_xy)
            target_x, target_y = _rect_xy(rectangle, callout.target_xy)
            center_x = label_x + 0.5 * swatch
            center_y = label_y + 22 * scale
            color = _hex_color(callout.color)
            elements.extend(
                (
                    f'<line x1="{center_x:.1f}" y1="{center_y:.1f}" x2="{target_x}" y2="{target_y}" '
                    f'stroke="{color}" stroke-width="{max(3 * scale, 1):.1f}"/>',
                    f'<circle cx="{target_x}" cy="{target_y}" r="{max(5 * scale, 2):.1f}" fill="{color}"/>',
                    f'<rect x="{label_x}" y="{label_y + 5 * scale:.1f}" '
                    f'width="{swatch:.1f}" height="{34 * scale:.1f}" fill="{color}"/>',
                    f'<text x="{label_x + swatch + gap:.1f}" '
                    f'y="{label_y + 0.82 * font_size:.1f}" class="callout-text">'
                    f'{callout.label}</text>',
                )
            )
    return "".join(elements)


def _svg_sole_offset_inset(
    rectangle: tuple[int, int, int, int],
    metrics: FitMetrics,
    target: tuple[float, float],
    *,
    scale: float,
) -> str:
    left, top, right, bottom = rectangle
    width, height = right - left, bottom - top
    x = left + 0.035 * width
    y = top + 0.035 * height
    inset_width = 0.47 * width
    inset_height = 0.31 * height
    target_x, target_y = _rect_xy(rectangle, target)
    leader_x = x + inset_width
    leader_y = y + 0.78 * inset_height
    diagram_top = y + 82 * scale
    diagram_bottom = y + inset_height - 27 * scale
    probe_x = x + 0.29 * inset_width
    bracket_x = x + 0.46 * inset_width
    ankle_y = diagram_top + 0.20 * (diagram_bottom - diagram_top)
    sole_y = diagram_top + 0.70 * (diagram_bottom - diagram_top)
    surface_left = x + 0.12 * inset_width
    surface_right = x + 0.88 * inset_width
    cap = max(10 * scale, 4)
    landmark = _hex_color(LANDMARK_BLUE)
    support = _hex_color(SUPPORT_CYAN)
    elements = [
        f'<line x1="{leader_x:.1f}" y1="{leader_y:.1f}" x2="{target_x}" y2="{target_y}" '
        f'stroke="{landmark}" stroke-width="{max(3 * scale, 1):.1f}"/>',
        f'<circle cx="{target_x}" cy="{target_y}" r="{max(6 * scale, 2):.1f}" '
        f'fill="{landmark}" stroke="#fafaf7" stroke-width="{max(2 * scale, 1):.1f}"/>',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{inset_width:.1f}" height="{inset_height:.1f}" '
        f'rx="{max(12 * scale, 4):.1f}" fill="#fafaf7" stroke="#8f9597" '
        f'stroke-width="{max(2 * scale, 1):.1f}"/>',
        f'<text x="{x + 18 * scale:.1f}" y="{y + 35 * scale:.1f}" class="inset-title">sole-offset calibration</text>',
        f'<text x="{x + 18 * scale:.1f}" y="{y + 67 * scale:.1f}" class="inset-metric">R_Ankle · magnified view</text>',
        f'<line x1="{surface_left:.1f}" y1="{sole_y:.1f}" x2="{surface_right:.1f}" y2="{sole_y:.1f}" '
        f'stroke="{support}" stroke-width="{max(7 * scale, 3):.1f}"/>',
        f'<line x1="{bracket_x:.1f}" y1="{ankle_y:.1f}" x2="{bracket_x:.1f}" y2="{sole_y:.1f}" '
        f'stroke="#353b3d" stroke-width="{max(4 * scale, 2):.1f}"/>',
        f'<line x1="{bracket_x-cap:.1f}" y1="{ankle_y:.1f}" x2="{bracket_x+cap:.1f}" y2="{ankle_y:.1f}" '
        f'stroke="#353b3d" stroke-width="{max(4 * scale, 2):.1f}"/>',
        f'<line x1="{bracket_x-cap:.1f}" y1="{sole_y:.1f}" x2="{bracket_x+cap:.1f}" y2="{sole_y:.1f}" '
        f'stroke="#353b3d" stroke-width="{max(4 * scale, 2):.1f}"/>',
        f'<circle cx="{probe_x:.1f}" cy="{ankle_y:.1f}" r="{max(10 * scale, 4):.1f}" '
        f'fill="{landmark}" stroke="#fafaf7" stroke-width="{max(3 * scale, 1):.1f}"/>',
        f'<circle cx="{probe_x:.1f}" cy="{sole_y:.1f}" r="{max(10 * scale, 4):.1f}" '
        f'fill="{support}" stroke="#fafaf7" stroke-width="{max(3 * scale, 1):.1f}"/>',
        f'<text x="{bracket_x + 18 * scale:.1f}" y="{(ankle_y + sole_y) / 2 + 8 * scale:.1f}" '
        f'class="inset-label">δ = {1000.0 * metrics.sole_offset:.0f} mm</text>',
        f'<text x="{probe_x + 16 * scale:.1f}" y="{ankle_y - 7 * scale:.1f}" '
        f'class="inset-note" fill="{landmark}">ankle landmark</text>',
        f'<text x="{surface_left:.1f}" y="{sole_y + 28 * scale:.1f}" '
        f'class="inset-note" fill="#2a6f77">sole / support surface</text>',
    ]
    return "".join(elements)


def _compose_svg(
    spec: MethodologySpec,
    panel_images: dict[str, Path],
    output: Path,
    rectangles: dict[str, tuple[int, int, int, int]],
    content_rectangles: dict[str, tuple[int, int, int, int]],
    *,
    scale: float,
    metrics: FitMetrics,
    annotation_targets: dict[str, tuple[float, float]],
) -> None:
    width = round(spec.width * scale)
    height = round(spec.height * scale)
    title_size = max(round(54 * scale), 18)
    titles = _panel_titles()
    images, labels = [], []
    for key, (left, top, _right, _bottom) in rectangles.items():
        content_left, content_top, content_right, content_bottom = content_rectangles[key]
        images.append(
            f'<image x="{content_left}" y="{content_top}" width="{content_right - content_left}" '
            f'height="{content_bottom - content_top}" '
            f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{_svg_data(panel_images[key])}"/>'
        )
        if spec.include_text:
            labels.append(
                f'<text x="{left + round(22 * scale)}" y="{top + round(60 * scale)}" class="title">{titles[key]}</text>'
            )
    a, b, c, d = (rectangles[key] for key in "abcd")
    arrows = [
        f'<path d="M {a[2] + 10 * scale:.1f} {(a[1] + a[3]) / 2:.1f} H {b[0] - 10 * scale:.1f}" class="arrow"/>',
        f'<path d="M {c[2] + 10 * scale:.1f} {(c[1] + c[3]) / 2:.1f} H {d[0] - 10 * scale:.1f}" class="arrow"/>',
        f'<path d="M {(b[0] + b[2]) / 2:.1f} {b[3] + 5 * scale:.1f} V {(b[3] + c[1]) / 2:.1f} '
        f'H {(c[0] + c[2]) / 2:.1f} V {c[1] - 5 * scale:.1f}" class="arrow"/>',
    ]
    panels = [
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" rx="{18*scale:.1f}" class="panel"/>'
        for left, top, right, bottom in rectangles.values()
    ]
    callouts = ""
    if spec.include_text:
        callouts = _svg_callouts(
            content_rectangles,
            metrics,
            annotation_targets,
            scale=scale,
        )
    panel_fill = "none" if spec.transparent_background else "#fafaf7"
    background_element = (
        "" if spec.transparent_background else '<rect width="100%" height="100%" fill="#f5f3ed"/>'
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#394347"/></marker>
  <style>
    .panel {{ fill:{panel_fill}; stroke:#b0b5b6; stroke-width:{max(2*scale,1):.1f}; }}
    .title {{ font:700 {title_size}px Arial,Helvetica,sans-serif; fill:#1f2426; }}
    .callout-text {{ font:700 {max(round(44*scale),14)}px Arial,Helvetica,sans-serif; fill:#23282a; stroke:#fafaf7; stroke-width:{max(5*scale,2):.1f}; paint-order:stroke; stroke-linejoin:round; }}
    .inset-title {{ font:700 {max(round(40*scale),13)}px Arial,Helvetica,sans-serif; fill:#23282a; }}
    .inset-metric {{ font:400 {max(round(28*scale),10)}px Arial,Helvetica,sans-serif; fill:#4b5052; }}
    .inset-label {{ font:700 {max(round(26*scale),9)}px Arial,Helvetica,sans-serif; fill:#23282a; }}
    .inset-note {{ font:400 {max(round(23*scale),8)}px Arial,Helvetica,sans-serif; }}
    .arrow {{ fill:none; stroke:#394347; stroke-width:{max(5*scale,2):.1f}; stroke-linejoin:round; marker-end:url(#arrowhead); }}
  </style>
</defs>
{background_element}
{''.join(panels)}
{''.join(images)}
{''.join(labels)}
{callouts}
{''.join(arrows)}
</svg>'''
    atomic_write(output, lambda destination: destination.write_text(svg))


def render_methodology_figure(
    spec: MethodologySpec,
    *,
    amass_root: str | Path,
    cache_root: str | Path,
    blender_executable: str | Path,
    output: str | Path,
    scale: float = 1.0,
) -> dict[str, object]:
    """Build, render, and compose the complete 2x2 methodology figure."""

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.casefold() != ".png":
        raise ValueError("methodology figure output must use the .png extension")
    blender = _resolve_blender(blender_executable)
    panel_directory = output_path.parent / f"{output_path.stem}-panels"
    panel_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terra-methodology-") as temporary_directory:
        temporary = Path(temporary_directory)
        diagnostics = build_methodology_bundles(
            spec,
            amass_root=amass_root,
            cache_root=cache_root,
            output_directory=temporary,
            scale=scale,
        )
        panel_images = {}
        panel_reports = {}
        panel_names = {
            "a": "a-motion",
            "b": "b-reconstruction-evidence",
            "c": "c-interaction-mesh",
            "d": "d-retargeting",
        }
        for key, stem in panel_names.items():
            image_path = panel_directory / f"{stem}.png"
            report_path = panel_directory / f"{stem}.renderer.json"
            panel_reports[key] = _render_bundle(
                Path(diagnostics["panels"][key]), image_path, report_path, blender
            )
            panel_images[key] = image_path

    terrain_report = json.loads(spec.terrain_report.read_text())
    metrics = _fit_metrics(terrain_report)
    annotation_targets = {
        key: tuple(float(value) for value in values)
        for key, values in diagnostics["annotation_targets"].items()
    }
    rectangles, content_rectangles = _compose_png(
        spec,
        panel_images,
        output_path,
        scale=scale,
        metrics=metrics,
        annotation_targets=annotation_targets,
    )
    svg_path = output_path.with_suffix(".svg")
    _compose_svg(
        spec,
        panel_images,
        svg_path,
        rectangles,
        content_rectangles,
        scale=scale,
        metrics=metrics,
        annotation_targets=annotation_targets,
    )
    version = subprocess.run(
        [str(blender), "--background", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    source_path = Path(amass_root).expanduser().resolve() / f"{spec.motion}.npz"
    cache = Path(cache_root).expanduser().resolve()
    target_manifest = _target_manifest(spec, np.asarray(diagnostics["anchor_xy"]), float(diagnostics["yaw_degrees"]), scale)
    trajectory_path, terrain_path, analysis_path = _artifact_paths(target_manifest, cache, spec.motion)
    provenance = {
        "schema_version": 1,
        "manifest": {"path": str(spec.path), "sha256": _sha256(spec.path)},
        "configuration": asdict(spec) | {"path": str(spec.path), "smpl_model": str(spec.smpl_model), "fitted_shape": str(spec.fitted_shape), "terrain_report": str(spec.terrain_report)},
        "inputs": {
            "source_motion": {"path": str(source_path), "sha256": _sha256(source_path)},
            "smpl_model": {"path": str(spec.smpl_model), "sha256": _sha256(spec.smpl_model / "SMPLH_NEUTRAL.pkl")},
            "fitted_shape": {"path": str(spec.fitted_shape), "sha256": _sha256(spec.fitted_shape)},
            "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
            "terrain": {"path": str(terrain_path), "sha256": _sha256(terrain_path)},
            "analysis": {"path": str(analysis_path), "sha256": _sha256(analysis_path)},
            "terrain_report": {"path": str(spec.terrain_report), "sha256": _sha256(spec.terrain_report)},
        },
        "diagnostics": diagnostics | {"panels": None},
        "renderer": {"backend": "blender-cycles", "blender_version": version, "scale": scale, "panels": panel_reports},
        "outputs": {
            "png": {"path": str(output_path), "sha256": _sha256(output_path)},
            "svg": {"path": str(svg_path), "sha256": _sha256(svg_path)},
            "panels": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in panel_images.items()},
        },
    }
    provenance_path = output_path.with_suffix(".provenance.json")
    atomic_write(
        provenance_path,
        lambda destination: destination.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n"),
    )
    return {
        "output": str(output_path),
        "svg": str(svg_path),
        "provenance": str(provenance_path),
        "panels": {key: str(path) for key, path in panel_images.items()},
        "width": round(spec.width * scale),
        "height": round(spec.height * scale),
        "motion": spec.motion,
        "diagnostics": {key: value for key, value in diagnostics.items() if key != "panels"},
    }


__all__ = [
    "MethodologySpec",
    "build_methodology_bundles",
    "load_methodology_spec",
    "render_methodology_figure",
]
