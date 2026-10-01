"""Blender scene export and headless rendering for publication figures."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from terra._files import atomic_write
from terra.figures.geometry import (
    alignment_yaw,
    cell_origin,
    history_alpha_by_frame,
    resolve_frames,
    resolve_highlight_frame,
    rotation_z,
)
from terra.figures.manifest import FigureManifest
from terra.figures.mujoco_renderer import _artifact_paths, _model, _sha256, validate_artifacts


def _visual_transform(
    source: mujoco.MjvGeom,
    rotation: np.ndarray,
    anchor: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(source.pos, dtype=np.float64).copy()
    position[:2] -= anchor
    position = rotation @ position + offset
    matrix = rotation @ np.asarray(source.mat, dtype=np.float64)
    return position, matrix


def _camera_payload(
    scene: mujoco.MjvScene,
    manifest: FigureManifest,
) -> dict[str, object]:
    positions = np.stack((scene.camera[0].pos, scene.camera[1].pos))
    return {
        "position": np.mean(positions, axis=0).tolist(),
        "forward": np.asarray(scene.camera[0].forward, dtype=float).tolist(),
        "up": np.asarray(scene.camera[0].up, dtype=float).tolist(),
        "fovy_degrees": manifest.layout.field_of_view,
    }


def export_scene_bundle(
    manifest: FigureManifest,
    *,
    cache_root: str | Path,
    output: str | Path,
    scale: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Freeze exact cached geometry and transforms into a Blender-readable NPZ."""

    output_path = Path(output).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    width = max(64, round(manifest.layout.width * scale))
    height = max(64, round(manifest.layout.height * scale))
    model = _model(manifest, width, height)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    scene = mujoco.MjvScene(model, maxgeom=2500)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (
        manifest.layout.camera_lookat_x,
        manifest.layout.camera_lookat_y,
        manifest.layout.camera_lookat_z,
    )
    camera.distance = manifest.layout.camera_distance
    camera.azimuth = manifest.layout.camera_azimuth
    camera.elevation = manifest.layout.camera_elevation
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    boxes_position: list[np.ndarray] = []
    boxes_size: list[np.ndarray] = []
    boxes_rotation: list[np.ndarray] = []
    boxes_rgba: list[tuple[float, ...]] = []
    boxes_kind: list[str] = []
    boxes_pitched: list[bool] = []
    border = manifest.layout.decorative_border
    if manifest.layout.horizon_extent > 0:
        boxes_position.append(
            np.asarray((0.0, 0.0, -manifest.layout.pad_thickness / 2.0 - 0.001))
        )
        boxes_size.append(
            np.asarray(
                (
                    manifest.layout.horizon_extent,
                    manifest.layout.horizon_extent,
                    manifest.layout.pad_thickness / 2.0,
                )
            )
        )
        boxes_rotation.append(np.eye(3))
        boxes_rgba.append(manifest.style.pad)
        boxes_kind.append("pad")
        boxes_pitched.append(False)
    half_pad = np.asarray(
        (
            manifest.layout.cell_size[0] / 2.0,
            manifest.layout.cell_size[1] / 2.0,
            manifest.layout.pad_thickness / 2.0,
        )
    )
    for row in range(-border, manifest.layout.rows + border):
        for column in range(-border, manifest.layout.columns + border):
            position = cell_origin(manifest.layout, row, column)
            position[2] = -manifest.layout.pad_thickness / 2.0
            boxes_position.append(position)
            boxes_size.append(half_pad)
            boxes_rotation.append(np.eye(3))
            boxes_rgba.append(manifest.style.pad)
            boxes_kind.append("pad")
            boxes_pitched.append(False)

    instance_mesh: list[int] = []
    instance_transform: list[np.ndarray] = []
    instance_rgba: list[tuple[float, ...]] = []
    instance_history: list[bool] = []
    instance_cell: list[int] = []
    instance_frame: list[int] = []
    tendon_endpoints: list[np.ndarray] = []
    tendon_radius: list[float] = []
    tendon_rgba: list[np.ndarray] = []
    cell_provenance: list[dict[str, object]] = []
    used_meshes: set[int] = set()

    from loco_mujoco.core.terrain import TerrainSpec
    from loco_mujoco.trajectory import Trajectory

    for cell_index, cell in enumerate(manifest.cells):
        trajectory_path, terrain_path, analysis_path = _artifact_paths(manifest, cache_path, cell.motion)
        qpos = np.asarray(Trajectory.load(str(trajectory_path)).data.qpos, dtype=float)
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise ValueError(f"{cell.motion}: qpos shape {qpos.shape} does not match model nq={model.nq}")
        frames = resolve_frames(cell, len(qpos))
        highlight_frame = resolve_highlight_frame(cell, frames)
        anchor = (
            np.asarray(cell.anchor_xy, dtype=float)
            if cell.anchor_xy is not None
            else np.mean(qpos[np.asarray(frames), :2], axis=0)
        )
        yaw = alignment_yaw(qpos[:, :2], frames, cell.yaw_degrees)
        rotation = rotation_z(yaw)
        offset = cell_origin(manifest.layout, cell.row, cell.column)

        terrain = TerrainSpec.load(str(terrain_path))
        for box in terrain.boxes:
            position = np.asarray(box.pos, dtype=float)
            position[:2] -= anchor
            position = rotation @ position + offset
            cosine, sine = math.cos(float(box.pitch)), math.sin(float(box.pitch))
            pitch = np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))
            box_rotation = rotation_z(math.degrees(float(box.yaw))) @ pitch
            boxes_position.append(position)
            boxes_size.append(np.asarray(box.size, dtype=float))
            boxes_rotation.append(rotation @ box_rotation)
            boxes_rgba.append(manifest.style.terrain)
            boxes_kind.append("terrain")
            boxes_pitched.append(abs(float(box.pitch)) > 1e-6)

        ghost_alpha = history_alpha_by_frame(frames, highlight_frame, manifest.style.history_alpha)
        for frame in frames:
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            current = frame == highlight_frame
            history_rgba = (*manifest.style.history, ghost_alpha.get(frame, 1.0))
            for source_index in range(scene.ngeom):
                source = scene.geoms[source_index]
                model_mesh = (
                    source.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and source.type == mujoco.mjtGeom.mjGEOM_MESH
                )
                tendon = source.objtype == mujoco.mjtObj.mjOBJ_TENDON
                if model_mesh:
                    position, matrix = _visual_transform(source, rotation, anchor, offset)
                    transform = np.eye(4)
                    transform[:3, :3] = matrix
                    transform[:3, 3] = position
                    mesh_id = int(model.geom_dataid[int(source.objid)])
                    used_meshes.add(mesh_id)
                    instance_mesh.append(mesh_id)
                    instance_transform.append(transform)
                    instance_rgba.append(tuple(source.rgba) if current else history_rgba)
                    instance_history.append(not current)
                    instance_cell.append(cell_index)
                    instance_frame.append(frame)
                elif current and tendon:
                    if source.type != mujoco.mjtGeom.mjGEOM_CAPSULE:
                        continue
                    position, matrix = _visual_transform(source, rotation, anchor, offset)
                    axis = matrix[:, 2]
                    half_length = float(source.size[2])
                    tendon_endpoints.append(np.stack((position - axis * half_length, position + axis * half_length)))
                    tendon_radius.append(float(source.size[0]))
                    tendon_rgba.append(np.asarray(source.rgba, dtype=float))

        cell_provenance.append(
            {
                "row": cell.row,
                "column": cell.column,
                "label": cell.label,
                "motion": cell.motion,
                "frames": list(frames),
                "highlight_frame": highlight_frame,
                "anchor_xy": anchor.tolist(),
                "yaw_degrees": yaw,
                "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
                "terrain": {"path": str(terrain_path), "sha256": _sha256(terrain_path)},
                "analysis": {"path": str(analysis_path), "sha256": _sha256(analysis_path)},
            }
        )

    arrays: dict[str, np.ndarray] = {
        "boxes_position": np.asarray(boxes_position, dtype=np.float32),
        "boxes_size": np.asarray(boxes_size, dtype=np.float32),
        "boxes_rotation": np.asarray(boxes_rotation, dtype=np.float32),
        "boxes_rgba": np.asarray(boxes_rgba, dtype=np.float32),
        "boxes_kind": np.asarray(boxes_kind),
        "boxes_pitched": np.asarray(boxes_pitched, dtype=bool),
        "instance_mesh": np.asarray(instance_mesh, dtype=np.int32),
        "instance_transform": np.asarray(instance_transform, dtype=np.float32),
        "instance_rgba": np.asarray(instance_rgba, dtype=np.float32),
        "instance_history": np.asarray(instance_history, dtype=bool),
        "instance_cell": np.asarray(instance_cell, dtype=np.int16),
        "instance_frame": np.asarray(instance_frame, dtype=np.int32),
        "tendon_endpoints": np.asarray(tendon_endpoints, dtype=np.float32).reshape(-1, 2, 3),
        "tendon_radius": np.asarray(tendon_radius, dtype=np.float32),
        "tendon_rgba": np.asarray(tendon_rgba, dtype=np.float32).reshape(-1, 4),
    }
    for mesh_id in sorted(used_meshes):
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        arrays[f"mesh_{mesh_id}_vertices"] = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=np.float32
        )
        arrays[f"mesh_{mesh_id}_faces"] = np.asarray(
            model.mesh_face[face_start : face_start + face_count], dtype=np.int32
        )

    metadata = {
        "schema_version": 1,
        "width": width,
        "height": height,
        "background": manifest.style.background,
        "ambient_strength": manifest.lighting.ambient_strength,
        "direct_light_scale": manifest.lighting.direct_light_scale,
        "camera": _camera_payload(scene, manifest),
        "mesh_ids": sorted(used_meshes),
        "bevel": {"pad": 0.012, "terrain": 0.025},
        "ramp_fill": {
            "mode": "vertical-skirt-wedge",
            "preserves": "pitched-box-top-plane",
        },
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return metadata, cell_provenance


def render_figure(
    manifest: FigureManifest,
    *,
    cache_root: str | Path,
    output: str | Path,
    blender_executable: str | Path,
    scale: float = 1.0,
) -> dict[str, object]:
    """Export the exact scene, invoke Blender headlessly, and write provenance."""

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.casefold() != ".png":
        raise ValueError("Blender figure output must use the .png extension")
    executable_value = str(blender_executable)
    resolved_executable = shutil.which(executable_value)
    blender_path = (
        Path(resolved_executable)
        if resolved_executable is not None
        else Path(executable_value).expanduser().resolve()
    )
    if not blender_path.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender_path}")
    validate_artifacts(manifest, cache_root)
    scene_script = Path(__file__).with_name("blender_scene.py")

    with tempfile.TemporaryDirectory(prefix="terra-blender-") as temporary_directory:
        temporary = Path(temporary_directory)
        bundle_path = temporary / "scene.npz"
        rendered_path = temporary / "render.png"
        report_path = temporary / "renderer.json"
        metadata, cells = export_scene_bundle(
            manifest,
            cache_root=cache_root,
            output=bundle_path,
            scale=scale,
        )
        command = [
            str(blender_path),
            "--background",
            "--factory-startup",
            "--python",
            str(scene_script),
            "--",
            "--bundle",
            str(bundle_path),
            "--out",
            str(rendered_path),
            "--report",
            str(report_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not rendered_path.is_file() or not report_path.is_file():
            detail = (completed.stdout + "\n" + completed.stderr).strip()
            raise RuntimeError(f"Blender render failed with exit code {completed.returncode}:\n{detail[-8000:]}")
        atomic_write(output_path, lambda destination: shutil.copyfile(rendered_path, destination))
        render_report = json.loads(report_path.read_text())

    version = subprocess.run(
        [str(blender_path), "--background", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance = {
        "schema_version": 1,
        "manifest": {"path": str(manifest.path), "sha256": _sha256(manifest.path)},
        "configuration": {
            "version": manifest.version,
            "name": manifest.name,
            "model": manifest.model,
            "method": manifest.method,
            "layout": asdict(manifest.layout),
            "style": asdict(manifest.style),
            "lighting": asdict(manifest.lighting),
            "cells": [asdict(cell) for cell in manifest.cells],
        },
        "renderer": {
            "backend": "blender-cycles",
            "blender_version": version,
            "width": metadata["width"],
            "height": metadata["height"],
            "scale": scale,
            "bevel": metadata["bevel"],
            "ramp_fill": metadata["ramp_fill"],
            **render_report,
        },
        "cells": cells,
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    atomic_write(
        provenance_path,
        lambda destination: destination.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n"),
    )
    return {
        "output": str(output_path),
        "provenance": str(provenance_path),
        "width": metadata["width"],
        "height": metadata["height"],
        "n_cells": len(manifest.cells),
        "backend": "blender-cycles",
    }


__all__ = ["export_scene_bundle", "render_figure"]
