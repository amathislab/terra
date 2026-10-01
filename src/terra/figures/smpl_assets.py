"""Standalone publication assets generated from the neutral SMPL-H model."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terra._files import atomic_write
from terra.figures.terrain_priors import (
    _audit_content_margin,
    _clean_transparent_film,
    _content_bounds,
    _render_bundle,
    _sha256,
)

BACKGROUND = (0.975, 0.973, 0.963)
NEUTRAL_BLUE_GRAY = (0.16, 0.42, 0.61, 1.0)
SMPLH_POSE_DIMENSION = 156
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_WRIST = 20
RIGHT_WRIST = 21


@dataclass(frozen=True, slots=True)
class SMPLAssetGeometry:
    """One evaluated neutral SMPL-H body and its audit landmarks."""

    vertices: np.ndarray
    faces: np.ndarray
    joints: np.ndarray
    pose_axis_angle: np.ndarray
    shoulder_angle_degrees: float


def apose_axis_angle(shoulder_angle_degrees: float = 45.0) -> np.ndarray:
    """Return a neutral full SMPL-H pose with symmetric A-pose shoulders."""

    if not 20.0 <= shoulder_angle_degrees <= 70.0:
        raise ValueError("A-pose shoulder angle must lie in [20, 70] degrees")
    angle = math.radians(shoulder_angle_degrees)
    pose = np.zeros(SMPLH_POSE_DIMENSION, dtype=np.float32)
    pose[LEFT_SHOULDER * 3 + 2] = -angle
    pose[RIGHT_SHOULDER * 3 + 2] = angle
    return pose


def smpl_to_blender_coordinates(points: np.ndarray, *, ground: float) -> np.ndarray:
    """Convert SMPL's Y-up, +Z-front frame to Blender's Z-up frame."""

    values = np.asarray(points, dtype=float)
    result = np.empty_like(values)
    result[..., 0] = values[..., 0]
    result[..., 1] = -values[..., 2]
    result[..., 2] = values[..., 1] - ground
    return result


def neutral_apose_geometry(
    model_directory: str | Path,
    *,
    shoulder_angle_degrees: float = 45.0,
) -> SMPLAssetGeometry:
    """Evaluate the zero-beta neutral SMPL-H model in a canonical A-pose."""

    import torch

    from loco_mujoco.smpl import SMPLH_Parser

    directory = Path(model_directory).expanduser().resolve()
    model = directory / "SMPLH_NEUTRAL.pkl"
    if not model.is_file():
        raise FileNotFoundError(f"neutral SMPL-H model not found: {model}")
    pose = apose_axis_angle(shoulder_angle_degrees)
    parser = SMPLH_Parser(model_path=str(directory), gender="neutral")
    with torch.no_grad():
        vertices, joints = parser.get_joints_verts(
            torch.as_tensor(pose).reshape(1, -1),
            th_betas=torch.zeros((1, 16), dtype=torch.float32),
            th_trans=torch.zeros((1, 3), dtype=torch.float32),
        )
    smpl_vertices = vertices[0].detach().cpu().numpy()
    smpl_joints = joints[0].detach().cpu().numpy()
    ground = float(np.min(smpl_vertices[:, 1]))
    world_vertices = smpl_to_blender_coordinates(smpl_vertices, ground=ground)
    world_joints = smpl_to_blender_coordinates(smpl_joints, ground=ground)
    centre_x = 0.5 * (float(np.min(world_vertices[:, 0])) + float(np.max(world_vertices[:, 0])))
    world_vertices[:, 0] -= centre_x
    world_joints[:, 0] -= centre_x
    geometry = SMPLAssetGeometry(
        vertices=np.asarray(world_vertices, dtype=np.float32),
        faces=np.asarray(parser.faces, dtype=np.int32),
        joints=np.asarray(world_joints, dtype=np.float32),
        pose_axis_angle=pose,
        shoulder_angle_degrees=shoulder_angle_degrees,
    )
    _audit_apose_geometry(geometry)
    return geometry


def _audit_apose_geometry(geometry: SMPLAssetGeometry) -> None:
    if geometry.vertices.shape != (6890, 3):
        raise ValueError(f"unexpected neutral SMPL-H vertex shape: {geometry.vertices.shape}")
    if geometry.faces.shape != (13776, 3):
        raise ValueError(f"unexpected neutral SMPL-H face shape: {geometry.faces.shape}")
    if not np.isfinite(geometry.vertices).all() or not np.isfinite(geometry.joints).all():
        raise ValueError("neutral SMPL-H A-pose contains non-finite geometry")
    if abs(float(np.min(geometry.vertices[:, 2]))) > 1e-5:
        raise ValueError("neutral SMPL-H A-pose is not grounded")
    shoulders = geometry.joints[[LEFT_SHOULDER, RIGHT_SHOULDER]]
    wrists = geometry.joints[[LEFT_WRIST, RIGHT_WRIST]]
    drops = shoulders[:, 2] - wrists[:, 2]
    if np.min(drops) < 0.20 or np.max(np.abs(drops - np.mean(drops))) > 0.02:
        raise ValueError(f"shoulder rotations do not produce a symmetric A-pose: wrist drops {drops}")


def _front_camera(vertices: np.ndarray, *, width: int, height: int) -> dict[str, object]:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    centre = 0.5 * (minimum + maximum)
    aspect = width / height
    horizontal_span = float(maximum[0] - minimum[0])
    vertical_span = float(maximum[2] - minimum[2])
    # Blender's orthographic scale is the vertical view span. Preserve 12% top
    # and bottom margin while also fitting the A-pose hands horizontally.
    ortho_scale = max(vertical_span / 0.76, horizontal_span / (aspect * 0.68))
    lookat = (float(centre[0]), 0.0, float(centre[2]))
    return {
        "position": (lookat[0], -4.5, lookat[2]),
        "forward": (0.0, 1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "fovy_degrees": 36.0,
        "projection": "orthographic",
        "ortho_scale": ortho_scale,
    }


def build_smpl_apose_bundle(
    geometry: SMPLAssetGeometry,
    *,
    width: int = 1200,
    height: int = 1500,
    samples: int = 96,
) -> dict[str, np.ndarray]:
    """Build a front-view Blender bundle for a neutral SMPL-H A-pose."""

    if width < 64 or height < 64:
        raise ValueError("SMPL asset dimensions must be at least 64 pixels")
    if samples < 1:
        raise ValueError("SMPL asset samples must be positive")
    metadata = {
        "schema_version": 2,
        "name": "smplh-neutral-apose-front",
        "width": int(width),
        "height": int(height),
        "background": BACKGROUND,
        "transparent_background": True,
        "camera": _front_camera(geometry.vertices, width=width, height=height),
        "mesh_ids": [],
        "bevel": {},
        "samples": int(samples),
        "ambient_strength": 0.20,
        "direct_light_scale": 1.0,
        "shadow_catcher": {"size": 10.0, "z": -0.008},
    }
    return {
        "boxes_position": np.empty((0, 3), dtype=np.float32),
        "boxes_size": np.empty((0, 3), dtype=np.float32),
        "boxes_rotation": np.empty((0, 3, 3), dtype=np.float32),
        "boxes_rgba": np.empty((0, 4), dtype=np.float32),
        "boxes_kind": np.empty(0, dtype="U1"),
        "boxes_pitched": np.empty(0, dtype=bool),
        "instance_mesh": np.empty(0, dtype=np.int32),
        "instance_transform": np.empty((0, 4, 4), dtype=np.float32),
        "instance_rgba": np.empty((0, 4), dtype=np.float32),
        "instance_history": np.empty(0, dtype=bool),
        "instance_cell": np.empty(0, dtype=np.int16),
        "instance_frame": np.empty(0, dtype=np.int32),
        "tendon_endpoints": np.empty((0, 2, 3), dtype=np.float32),
        "tendon_radius": np.empty(0, dtype=np.float32),
        "tendon_rgba": np.empty((0, 4), dtype=np.float32),
        "source_vertices": geometry.vertices[None],
        "source_faces": geometry.faces,
        "source_rgba": np.asarray((NEUTRAL_BLUE_GRAY,), dtype=np.float32),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def render_neutral_apose(
    *,
    model_directory: str | Path,
    blender_executable: str | Path,
    output: str | Path,
    shoulder_angle_degrees: float = 45.0,
    width: int = 1200,
    height: int = 1500,
    samples: int = 96,
    scale: float = 1.0,
) -> dict[str, object]:
    """Render and audit a neutral front-view SMPL-H A-pose asset."""

    if not 0.1 <= scale <= 1.0:
        raise ValueError(f"scale must lie in [0.1, 1.0], got {scale}")
    model_root = Path(model_directory).expanduser().resolve()
    model = model_root / "SMPLH_NEUTRAL.pkl"
    blender = Path(blender_executable).expanduser().resolve()
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_width = max(64, round(width * scale))
    render_height = max(64, round(height * scale))
    render_samples = max(16, round(samples * scale))
    geometry = neutral_apose_geometry(
        model_root,
        shoulder_angle_degrees=shoulder_angle_degrees,
    )
    arrays = build_smpl_apose_bundle(
        geometry,
        width=render_width,
        height=render_height,
        samples=render_samples,
    )
    metadata = json.loads(str(arrays["metadata"]))
    with tempfile.TemporaryDirectory(prefix="terra-smpl-asset-") as temporary_directory:
        temporary = Path(temporary_directory)
        bundle = temporary / "smpl-neutral-apose-front.npz"
        raw_render = temporary / "smpl-neutral-apose-front.png"
        np.savez_compressed(bundle, **arrays)
        renderer = _render_bundle(bundle, raw_render, blender)
        atomic_write(destination, lambda path: shutil.copyfile(raw_render, path))
    _clean_transparent_film(destination)
    bounds = _content_bounds(destination)
    _audit_content_margin(bounds, width=render_width, height=render_height)
    renderer["output"] = str(destination)
    renderer["sha256"] = _sha256(destination)
    renderer["alpha_cleanup_threshold"] = 15
    report_path = destination.with_suffix(".renderer.json")
    atomic_write(
        report_path,
        lambda path: path.write_text(json.dumps(renderer, indent=2, sort_keys=True) + "\n"),
    )
    shoulders = geometry.joints[[LEFT_SHOULDER, RIGHT_SHOULDER]]
    wrists = geometry.joints[[LEFT_WRIST, RIGHT_WRIST]]
    provenance = {
        "schema_version": 1,
        "description": "Zero-beta neutral SMPL-H body in a symmetric A-pose",
        "model": {"path": str(model), "sha256": _sha256(model)},
        "pose": {
            "shoulder_angle_degrees": shoulder_angle_degrees,
            "left_shoulder_axis_angle": geometry.pose_axis_angle[LEFT_SHOULDER * 3 : LEFT_SHOULDER * 3 + 3].tolist(),
            "right_shoulder_axis_angle": geometry.pose_axis_angle[RIGHT_SHOULDER * 3 : RIGHT_SHOULDER * 3 + 3].tolist(),
            "wrist_drop_below_shoulder_m": (shoulders[:, 2] - wrists[:, 2]).tolist(),
            "all_other_pose_parameters_zero": True,
            "betas": "16 zeros",
        },
        "geometry": {
            "vertices": len(geometry.vertices),
            "faces": len(geometry.faces),
            "world_extent_m": np.ptp(geometry.vertices, axis=0).tolist(),
            "coordinate_frame": "Blender Z-up; front is camera-facing -Y",
        },
        "camera": metadata["camera"],
        "dimensions": [render_width, render_height],
        "content_bounds": bounds,
        "renderer": renderer,
        "output": {"path": str(destination), "sha256": _sha256(destination)},
    }
    provenance_path = destination.with_suffix(".provenance.json")
    atomic_write(
        provenance_path,
        lambda path: path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n"),
    )
    return {**provenance, "provenance": str(provenance_path), "renderer_report": str(report_path)}


__all__ = [
    "SMPLAssetGeometry",
    "apose_axis_angle",
    "build_smpl_apose_bundle",
    "neutral_apose_geometry",
    "render_neutral_apose",
    "smpl_to_blender_coordinates",
]
