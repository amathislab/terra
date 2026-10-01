"""Live-motion MuJoCo renderer for the manuscript overview video."""

from __future__ import annotations

import math
import subprocess
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from terra.figures.blender_renderer import _visual_transform
from terra.figures.geometry import (
    cell_origin,
    history_alpha_by_frame,
    rotation_z,
)
from terra.figures.manifest import FigureManifest
from terra.figures.mujoco_renderer import (
    _add_box,
    _configure_lighting,
    _copy_visual_geom,
    _model,
    _pitch_rotation,
    _sha256,
    validate_artifacts,
)


def _live_geometry(
    destination_scene: mujoco.MjvScene,
    source_scene: mujoco.MjvScene,
    *,
    rotation: np.ndarray,
    anchor: np.ndarray,
    offset: np.ndarray,
    tendon_line_width: float,
) -> None:
    tendon_rgba = np.asarray((0.95, 0.30, 0.30, 1.0), dtype=np.float32)
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    zero = np.zeros(3, dtype=np.float64)
    for source_index in range(source_scene.ngeom):
        source = source_scene.geoms[source_index]
        model_mesh = source.objtype == mujoco.mjtObj.mjOBJ_GEOM and source.type == mujoco.mjtGeom.mjGEOM_MESH
        tendon = source.objtype == mujoco.mjtObj.mjOBJ_TENDON and source.type == mujoco.mjtGeom.mjGEOM_CAPSULE
        if not (model_mesh or tendon):
            continue
        if destination_scene.ngeom >= destination_scene.maxgeom:
            raise RuntimeError(f"animated scene exceeds its {destination_scene.maxgeom}-geometry capacity")
        destination = destination_scene.geoms[destination_scene.ngeom]
        if model_mesh:
            _copy_visual_geom(
                destination,
                source,
                rotation=rotation,
                anchor_xy=anchor,
                offset=offset,
                rgba=None,
            )
        else:
            position, matrix = _visual_transform(source, rotation, anchor, offset)
            axis = matrix[:, 2]
            half_length = float(source.size[2])
            mujoco.mjv_initGeom(
                destination,
                mujoco.mjtGeom.mjGEOM_LINE,
                zero,
                position,
                identity,
                tendon_rgba,
            )
            mujoco.mjv_connector(
                destination,
                mujoco.mjtGeom.mjGEOM_LINE,
                tendon_line_width,
                position - axis * half_length,
                position + axis * half_length,
            )
        destination_scene.ngeom += 1


def _camera_progress(frame_index: int, frame_count: int) -> float:
    progress = min(max(frame_index / max(frame_count - 1, 1), 0.0), 1.0)
    return progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)


def render_video(
    manifest: FigureManifest,
    *,
    cache_root: str | Path,
    output: str | Path,
    ffmpeg: str | Path,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
    camera_frame_count: int | None = None,
    start_zoom: float,
    frame_start: int = 0,
    frame_end: int | None = None,
    tendon_line_width: float = 1.8,
) -> dict[str, object]:
    """Render looping live bodies and fixed ghosts through one smooth 3D camera."""

    if width < 64 or height < 64 or width % 2 or height % 2:
        raise ValueError("video dimensions must be even and at least 64 pixels")
    if fps <= 0 or frame_count < 2:
        raise ValueError("video timing must contain at least two frames at a positive rate")
    resolved_camera_frame_count = frame_count if camera_frame_count is None else camera_frame_count
    if not 2 <= resolved_camera_frame_count <= frame_count:
        raise ValueError("camera_frame_count must be between two and frame_count")
    resolved_frame_end = frame_count if frame_end is None else frame_end
    if not 0 <= frame_start < resolved_frame_end <= frame_count:
        raise ValueError("frame range must satisfy 0 <= start < end <= frame_count")
    if start_zoom < 1.0:
        raise ValueError("start_zoom must be at least one")
    if tendon_line_width <= 0.0:
        raise ValueError("tendon_line_width must be positive")

    output_path = Path(output).expanduser().resolve()
    ffmpeg_path = Path(ffmpeg).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    resolved = validate_artifacts(manifest, cache_path)

    from loco_mujoco.core.terrain import TerrainSpec
    from loco_mujoco.trajectory import Trajectory

    video_manifest = replace(
        manifest,
        lighting=replace(
            manifest.lighting,
            key_samples=1,
            key_spread=0.0,
            shadow_map_size=2048,
        ),
    )
    model = _model(video_manifest, width, height)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    source_scene = mujoco.MjvScene(model, maxgeom=2500)
    source_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(source_camera)

    maximum_geometry = 5000 + len(manifest.cells) * 1450
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
    renderer = mujoco.Renderer(
        model,
        width=width,
        height=height,
        max_geom=maximum_geometry,
    )

    trajectories: list[np.ndarray] = []
    frequencies: list[float] = []
    anchors: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    cell_provenance: list[dict[str, object]] = []
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        scene = renderer.scene
        scene.ngeom = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
        _configure_lighting(scene, video_manifest)

        identity = np.eye(3)
        border = manifest.layout.decorative_border
        if manifest.layout.horizon_extent > 0:
            _add_box(
                scene,
                np.asarray((0.0, 0.0, -manifest.layout.pad_thickness / 2.0 - 0.001)),
                (
                    manifest.layout.horizon_extent,
                    manifest.layout.horizon_extent,
                    manifest.layout.pad_thickness / 2.0,
                ),
                identity,
                manifest.style.pad,
            )
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
            if cell.frames is None:
                raise ValueError(f"{cell.motion}: overview video requires explicit display frames")
            trajectory_path = Path(paths["trajectory"])
            terrain_path = Path(paths["terrain"])
            analysis_path = Path(paths["analysis"])
            trajectory = Trajectory.load(str(trajectory_path))
            qpos = np.asarray(trajectory.data.qpos, dtype=np.float64)
            frequency = float(trajectory.info.frequency)
            if qpos.ndim != 2 or qpos.shape[1] != model.nq:
                raise ValueError(f"{cell.motion}: qpos shape {qpos.shape} does not match model nq={model.nq}")
            if not math.isfinite(frequency) or frequency <= 0.0:
                raise ValueError(f"{cell.motion}: invalid trajectory frequency {frequency}")
            anchor = (
                np.asarray(cell.anchor_xy, dtype=np.float64)
                if cell.anchor_xy is not None
                else np.mean(qpos[np.asarray(cell.frames), :2], axis=0)
            )
            rotation = rotation_z(float(cell.yaw_degrees or 0.0))
            offset = cell_origin(manifest.layout, cell.row, cell.column)
            terrain = TerrainSpec.load(str(terrain_path))
            for box in terrain.boxes:
                position = np.asarray(box.pos, dtype=np.float64)
                position[:2] -= anchor
                position = rotation @ position + offset
                box_rotation = rotation_z(math.degrees(float(box.yaw))) @ _pitch_rotation(float(box.pitch))
                _add_box(
                    scene,
                    position,
                    np.asarray(box.size, dtype=np.float64),
                    rotation @ box_rotation,
                    manifest.style.terrain,
                )

            highlight_frame = int(cell.highlight_frame)
            ghost_alpha = history_alpha_by_frame(
                cell.frames,
                highlight_frame,
                manifest.style.history_alpha,
            )
            for ghost_frame in cell.frames:
                if ghost_frame == highlight_frame:
                    continue
                data.qpos[:] = qpos[ghost_frame]
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
                alpha = ghost_alpha[ghost_frame]
                rgba = (
                    *(
                        alpha * np.asarray(manifest.style.history)
                        + (1.0 - alpha) * np.asarray(manifest.style.background)
                    ),
                    1.0,
                )
                for source_index in range(source_scene.ngeom):
                    source = source_scene.geoms[source_index]
                    if not (source.objtype == mujoco.mjtObj.mjOBJ_GEOM and source.type == mujoco.mjtGeom.mjGEOM_MESH):
                        continue
                    _copy_visual_geom(
                        scene.geoms[scene.ngeom],
                        source,
                        rotation=rotation,
                        anchor_xy=anchor,
                        offset=offset,
                        rgba=rgba,
                    )
                    scene.ngeom += 1

            trajectories.append(qpos)
            frequencies.append(frequency)
            anchors.append(anchor)
            rotations.append(rotation)
            offsets.append(offset)
            segment = cell.loop_frames or (cell.frames[0], cell.frames[-1])
            if not 0 <= segment[0] < segment[1] < len(qpos):
                raise ValueError(f"{cell.motion}: loop frames {segment} lie outside the {len(qpos)}-frame trajectory")
            segments.append(segment)
            cell_provenance.append(
                {
                    "row": cell.row,
                    "column": cell.column,
                    "motion": cell.motion,
                    "frequency": frequency,
                    "start_frame": segment[0],
                    "end_frame": segment[1],
                    "loop_duration_seconds": ((segment[1] - segment[0] + 1) / frequency),
                    "trajectory": {
                        "path": str(trajectory_path),
                        "sha256": _sha256(trajectory_path),
                    },
                    "terrain": {
                        "path": str(terrain_path),
                        "sha256": _sha256(terrain_path),
                    },
                    "analysis": {
                        "path": str(analysis_path),
                        "sha256": _sha256(analysis_path),
                    },
                }
            )

        static_geometry = int(scene.ngeom)
        central_index = min(
            range(len(manifest.cells)),
            key=lambda index: (
                (manifest.cells[index].row - manifest.layout.rows / 2.0) ** 2
                + (manifest.cells[index].column - manifest.layout.columns / 2.0) ** 2
            ),
        )
        central_start = trajectories[central_index][segments[central_index][0], :3].copy()
        central_start[:2] -= anchors[central_index]
        start_lookat = rotations[central_index] @ central_start + offsets[central_index]
        end_lookat = np.asarray(
            (
                manifest.layout.camera_lookat_x,
                manifest.layout.camera_lookat_y,
                manifest.layout.camera_lookat_z,
            ),
            dtype=np.float64,
        )
        start_distance = manifest.layout.camera_distance / start_zoom
        end_distance = manifest.layout.camera_distance

        command = [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            if encoder.stdin is None:
                raise RuntimeError("failed to open ffmpeg video input")
            for frame_index in range(frame_start, resolved_frame_end):
                elapsed = frame_index / fps
                scene.ngeom = static_geometry
                for qpos, frequency, anchor, rotation, offset, segment in zip(
                    trajectories,
                    frequencies,
                    anchors,
                    rotations,
                    offsets,
                    segments,
                    strict=True,
                ):
                    start_frame, end_frame = segment
                    segment_length = end_frame - start_frame + 1
                    source_frame = start_frame + (math.floor(elapsed * frequency + 1e-9) % segment_length)
                    data.qpos[:] = qpos[source_frame]
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
                    _live_geometry(
                        scene,
                        source_scene,
                        rotation=rotation,
                        anchor=anchor,
                        offset=offset,
                        tendon_line_width=tendon_line_width,
                    )

                eased = _camera_progress(frame_index, resolved_camera_frame_count)
                camera.lookat[:] = start_lookat + eased * (end_lookat - start_lookat)
                camera.distance = start_distance * math.exp(math.log(end_distance / start_distance) * eased)
                mujoco.mjv_updateCamera(model, data, camera, scene)
                pixels = renderer.render()
                encoder.stdin.write(np.ascontiguousarray(pixels).tobytes())
                if (
                    frame_index == frame_start
                    or (frame_index - frame_start + 1) % fps == 0
                    or frame_index + 1 == resolved_frame_end
                ):
                    print(
                        f"rendered global frame {frame_index + 1}/{frame_count}; "
                        f"segment {frame_index - frame_start + 1}/"
                        f"{resolved_frame_end - frame_start}",
                        flush=True,
                    )
            encoder.stdin.close()
            return_code = encoder.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        finally:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
    finally:
        renderer.close()

    central_cell = manifest.cells[central_index]
    return {
        "backend": "mujoco-osmesa",
        "mujoco_version": mujoco.__version__,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": resolved_frame_end - frame_start,
        "duration_seconds": (resolved_frame_end - frame_start) / fps,
        "global_frame_count": frame_count,
        "camera_frame_count": resolved_camera_frame_count,
        "global_frame_start": frame_start,
        "global_frame_end_exclusive": resolved_frame_end,
        "camera_interpolation": "quintic-smootherstep-geometric-distance",
        "start_zoom": start_zoom,
        "static_geometries": static_geometry,
        "last_frame_geometries": int(scene.ngeom),
        "tendon_rendering": {
            "primitive": "screen-space-line",
            "width_pixels": tendon_line_width,
        },
        "video_lighting": {
            "key_samples": video_manifest.lighting.key_samples,
            "shadow_map_size": video_manifest.lighting.shadow_map_size,
            "cast_shadows": False,
            "ghost_alpha_mode": "baked-opaque-against-background",
        },
        "central_cell": {
            "row": central_cell.row,
            "column": central_cell.column,
            "motion": central_cell.motion,
        },
        "cells": cell_provenance,
    }


__all__ = ["render_video"]
