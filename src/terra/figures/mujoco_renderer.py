"""Shared-camera MuJoCo backend for manifest-driven publication figures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")

import mujoco
import numpy as np
from PIL import Image

from terra._files import atomic_write
from terra.artifacts import retarget_cache_paths
from terra.figures.geometry import (
    alignment_yaw,
    cell_origin,
    history_alpha_by_frame,
    resolve_frames,
    resolve_highlight_frame,
    rotation_z,
)
from terra.figures.manifest import FigureManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_visual_geom(
    destination: mujoco.MjvGeom,
    source: mujoco.MjvGeom,
    *,
    rotation: np.ndarray,
    anchor_xy: np.ndarray,
    offset: np.ndarray,
    rgba: tuple[float, float, float, float] | None,
) -> None:
    position = np.asarray(source.pos, dtype=np.float64).copy()
    position[:2] -= anchor_xy
    position = rotation @ position + offset
    matrix = rotation @ np.asarray(source.mat, dtype=np.float64)
    color = np.asarray(source.rgba if rgba is None else rgba, dtype=np.float32)
    mujoco.mjv_initGeom(
        destination,
        source.type,
        np.asarray(source.size, dtype=np.float64),
        position,
        matrix.reshape(-1),
        color,
    )
    for field in (
        "category",
        "dataid",
        "emission",
        "matid",
        "modelrbound",
        "objid",
        "objtype",
        "reflectance",
        "segid",
        "shininess",
        "specular",
        "texcoord",
        "texid",
        "texuniform",
        "transparent",
        "type",
    ):
        setattr(destination, field, getattr(source, field))
    destination.texrepeat[:] = source.texrepeat
    destination.pos[:] = position
    destination.mat[:] = matrix
    destination.rgba[:] = color
    if rgba is not None:
        destination.matid = -1
        destination.texid = -1
        destination.transparent = int(color[3] < 1.0)


def _add_box(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    size: tuple[float, float, float] | np.ndarray,
    matrix: np.ndarray,
    rgba: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"publication scene exceeds its {scene.maxgeom}-geometry capacity")
    destination = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        destination,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(size, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.asarray(matrix, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    destination.category = mujoco.mjtCatBit.mjCAT_STATIC
    destination.specular = 0.12
    destination.shininess = 0.22
    destination.reflectance = 0.0
    scene.ngeom += 1


def _pitch_rotation(pitch: float) -> np.ndarray:
    cosine, sine = math.cos(pitch), math.sin(pitch)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _uniform_skybox(model: mujoco.MjModel, rgb: tuple[float, float, float]) -> None:
    color = np.asarray(tuple(round(255 * channel) for channel in rgb), dtype=np.uint8)
    for texture_id in range(model.ntex):
        if model.tex_type[texture_id] != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            continue
        start = int(model.tex_adr[texture_id])
        count = int(model.tex_width[texture_id] * model.tex_height[texture_id] * 3)
        model.tex_data[start : start + count].reshape(-1, 3)[:] = color


def _scene_extent(manifest: FigureManifest) -> float:
    border = manifest.layout.decorative_border
    x_extent = (
        ((manifest.layout.columns - 1) / 2.0 + border) * manifest.layout.cell_pitch[0]
        + manifest.layout.cell_size[0] / 2.0
    )
    y_extent = (
        ((manifest.layout.rows - 1) / 2.0 + border) * manifest.layout.cell_pitch[1]
        + manifest.layout.cell_size[1] / 2.0
    )
    return max(3.0, x_extent, y_extent)


def _set_directional_light(
    light: mujoco.MjvLight,
    *,
    direction: tuple[float, float, float] | np.ndarray,
    ambient: tuple[float, float, float],
    diffuse: tuple[float, float, float],
    specular: tuple[float, float, float],
    cast_shadow: bool,
    headlight: bool = False,
) -> None:
    normalized = np.asarray(direction, dtype=np.float32)
    normalized /= np.linalg.norm(normalized)
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.headlight = int(headlight)
    light.castshadow = int(cast_shadow)
    light.pos[:] = (0.0, 0.0, 0.0)
    light.dir[:] = normalized
    light.ambient[:] = ambient
    light.diffuse[:] = diffuse
    light.specular[:] = specular
    light.attenuation[:] = (1.0, 0.0, 0.0)
    light.cutoff = 45.0
    light.exponent = 10.0
    light.bulbradius = 0.05
    light.intensity = 1.0
    light.range = 0.0
    light.texid = -1


def _configure_lighting(scene: mujoco.MjvScene, manifest: FigureManifest) -> None:
    lighting = manifest.lighting
    camera_direction = np.asarray(scene.lights[0].dir, dtype=np.float32).copy()
    scene.nlight = 2 + lighting.key_samples
    _set_directional_light(
        scene.lights[0],
        direction=camera_direction,
        ambient=lighting.ambient,
        diffuse=lighting.headlight_diffuse,
        specular=(0.02, 0.02, 0.02),
        cast_shadow=False,
        headlight=True,
    )

    key = np.asarray(lighting.key_direction, dtype=np.float32)
    key /= np.linalg.norm(key)
    tangent = np.cross(key, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(key, tangent)
    key_diffuse = np.asarray(lighting.key_diffuse) / lighting.key_samples
    key_specular = np.asarray(lighting.key_specular) / lighting.key_samples
    for sample in range(lighting.key_samples):
        angle = math.tau * sample / lighting.key_samples
        jitter = math.cos(angle) * tangent + math.sin(angle) * bitangent
        direction = key if lighting.key_samples == 1 else key + lighting.key_spread * jitter
        _set_directional_light(
            scene.lights[1 + sample],
            direction=direction,
            ambient=(0.0, 0.0, 0.0),
            diffuse=key_diffuse,
            specular=key_specular,
            cast_shadow=True,
        )

    _set_directional_light(
        scene.lights[1 + lighting.key_samples],
        direction=lighting.fill_direction,
        ambient=(0.0, 0.0, 0.0),
        diffuse=lighting.fill_diffuse,
        specular=(0.03, 0.04, 0.05),
        cast_shadow=False,
    )


def _model(manifest: FigureManifest, width: int, height: int) -> mujoco.MjModel:
    if manifest.model != "MyoFullBody":
        raise ValueError(f"the MuJoCo figure backend currently supports MyoFullBody, got {manifest.model!r}")
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    environment = MyoFullBody(th_params={"random_start": False, "fixed_start_conf": (0, 0)})
    model = environment._model
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.global_.fovy = manifest.layout.field_of_view
    model.vis.quality.offsamples = 4
    model.vis.quality.shadowsize = manifest.lighting.shadow_map_size
    model.vis.map.shadowclip = manifest.lighting.shadow_clip
    model.vis.map.znear = 0.005
    model.vis.map.zfar = 3.0
    model.stat.extent = _scene_extent(manifest)
    model.vis.rgba.haze[:] = (*manifest.style.background, 1.0)
    model.vis.headlight.ambient[:] = manifest.lighting.ambient
    model.vis.headlight.diffuse[:] = manifest.lighting.headlight_diffuse
    model.vis.headlight.specular[:] = (0.02, 0.02, 0.02)
    _uniform_skybox(model, manifest.style.background)
    return model


def _artifact_paths(manifest: FigureManifest, cache_root: Path, motion: str) -> tuple[Path, Path, Path]:
    paths = retarget_cache_paths(cache_root, motion, method=manifest.method, env_name=manifest.model)
    missing = [path for path in (paths.trajectory_path, paths.terrain_path, paths.analysis_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{motion}: missing required figure artifact {missing[0]}")
    return paths.trajectory_path, paths.terrain_path, paths.analysis_path


def validate_artifacts(manifest: FigureManifest, cache_root: str | Path) -> tuple[dict[str, str], ...]:
    """Resolve every manifest cell and fail before rendering if any artifact is absent."""

    resolved_root = Path(cache_root).expanduser().resolve()
    rows = []
    for cell in manifest.cells:
        trajectory, terrain, analysis = _artifact_paths(manifest, resolved_root, cell.motion)
        rows.append(
            {
                "motion": cell.motion,
                "trajectory": str(trajectory),
                "terrain": str(terrain),
                "analysis": str(analysis),
            }
        )
    return tuple(rows)


def _manifest_payload(manifest: FigureManifest) -> dict[str, object]:
    return {
        "version": manifest.version,
        "name": manifest.name,
        "model": manifest.model,
        "method": manifest.method,
        "layout": asdict(manifest.layout),
        "style": asdict(manifest.style),
        "lighting": asdict(manifest.lighting),
        "cells": [asdict(cell) for cell in manifest.cells],
    }


def render_figure(
    manifest: FigureManifest,
    *,
    cache_root: str | Path,
    output: str | Path,
    scale: float = 1.0,
) -> dict[str, object]:
    """Render one exact multi-environment scene and write its provenance sidecar."""

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.casefold() != ".png":
        raise ValueError("MuJoCo figure output must use the .png extension")
    cache_path = Path(cache_root).expanduser().resolve()
    resolved = validate_artifacts(manifest, cache_path)
    width = max(64, round(manifest.layout.width * scale))
    height = max(64, round(manifest.layout.height * scale))

    from loco_mujoco.core.terrain import TerrainSpec
    from loco_mujoco.trajectory import Trajectory

    model = _model(manifest, width, height)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    source_scene = mujoco.MjvScene(model, maxgeom=2500)
    source_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(source_camera)

    histories = max(
        (len(cell.frames or cell.fractions or ()) - 1 for cell in manifest.cells),
        default=0,
    )
    max_geometry = 5000 + len(manifest.cells) * (1200 + 120 * histories)
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

    renderer = mujoco.Renderer(model, width=width, height=height, max_geom=max_geometry)
    cell_provenance: list[dict[str, object]] = []
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        scene = renderer.scene
        scene.ngeom = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
        _configure_lighting(scene, manifest)

        identity = np.eye(3)
        border = manifest.layout.decorative_border
        if manifest.layout.horizon_extent > 0:
            underlay_position = np.asarray(
                (0.0, 0.0, -manifest.layout.pad_thickness / 2.0 - 0.001)
            )
            underlay_size = (
                manifest.layout.horizon_extent,
                manifest.layout.horizon_extent,
                manifest.layout.pad_thickness / 2.0,
            )
            _add_box(scene, underlay_position, underlay_size, identity, manifest.style.pad)
        half_pad = (
            manifest.layout.cell_size[0] / 2.0,
            manifest.layout.cell_size[1] / 2.0,
            manifest.layout.pad_thickness / 2.0,
        )
        for row in range(-border, manifest.layout.rows + border):
            for column in range(-border, manifest.layout.columns + border):
                origin = cell_origin(manifest.layout, row, column)
                origin[2] = -manifest.layout.pad_thickness / 2.0
                _add_box(scene, origin, half_pad, identity, manifest.style.pad)

        for cell, paths in zip(manifest.cells, resolved, strict=True):
            trajectory_path = Path(paths["trajectory"])
            terrain_path = Path(paths["terrain"])
            analysis_path = Path(paths["analysis"])
            trajectory = Trajectory.load(str(trajectory_path))
            qpos = np.asarray(trajectory.data.qpos, dtype=float)
            if qpos.ndim != 2 or qpos.shape[1] != model.nq:
                raise ValueError(
                    f"{cell.motion}: qpos shape {qpos.shape} does not match model nq={model.nq}"
                )
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
                box_rotation = rotation_z(math.degrees(float(box.yaw))) @ _pitch_rotation(float(box.pitch))
                _add_box(
                    scene,
                    position,
                    np.asarray(box.size, dtype=float),
                    rotation @ box_rotation,
                    manifest.style.terrain,
                )

            ghost_alpha = history_alpha_by_frame(frames, highlight_frame, manifest.style.history_alpha)
            for frame in frames:
                data.qpos[:] = qpos[frame]
                mujoco.mj_forward(model, data)
                mujoco.mjv_updateScene(
                    model,
                    data,
                    option,
                    None,
                    source_camera,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    source_scene,
                )
                current = frame == highlight_frame
                history_rgba = (*manifest.style.history, ghost_alpha.get(frame, 1.0))
                for source_index in range(source_scene.ngeom):
                    source_geom = source_scene.geoms[source_index]
                    model_mesh = (
                        source_geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                        and source_geom.type == mujoco.mjtGeom.mjGEOM_MESH
                    )
                    tendon = source_geom.objtype == mujoco.mjtObj.mjOBJ_TENDON
                    if current:
                        if not (model_mesh or tendon):
                            continue
                        rgba = None
                    else:
                        if not model_mesh:
                            continue
                        rgba = history_rgba
                    if scene.ngeom >= scene.maxgeom:
                        raise RuntimeError(f"publication scene exceeds its {scene.maxgeom}-geometry capacity")
                    _copy_visual_geom(
                        scene.geoms[scene.ngeom],
                        source_geom,
                        rotation=rotation,
                        anchor_xy=anchor,
                        offset=offset,
                        rgba=rgba,
                    )
                    scene.ngeom += 1

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

        pixels = renderer.render()
    finally:
        renderer.close()

    atomic_write(output_path, lambda temporary: Image.fromarray(pixels).save(temporary, format="PNG"))
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance = {
        "schema_version": 1,
        "manifest": {"path": str(manifest.path), "sha256": _sha256(manifest.path)},
        "configuration": _manifest_payload(manifest),
        "renderer": {
            "backend": "mujoco",
            "mujoco_version": mujoco.__version__,
            "width": width,
            "height": height,
            "scale": scale,
            "scene_geometries": int(scene.ngeom),
        },
        "cells": cell_provenance,
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    atomic_write(
        provenance_path,
        lambda temporary: temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n"),
    )
    return {
        "output": str(output_path),
        "provenance": str(provenance_path),
        "width": width,
        "height": height,
        "n_cells": len(manifest.cells),
        "n_geometries": int(scene.ngeom),
    }


__all__ = ["render_figure", "validate_artifacts"]
