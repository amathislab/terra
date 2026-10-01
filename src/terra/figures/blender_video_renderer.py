"""Blender animation export and frame rendering for the overview video."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from terra.figures.blender_renderer import (
    _camera_payload,
    _visual_transform,
    export_scene_bundle,
)
from terra.figures.geometry import cell_origin, rotation_z
from terra.figures.manifest import FigureManifest
from terra.figures.mujoco_renderer import _artifact_paths, _model
from terra.figures.mujoco_video_renderer import _camera_progress


def _animated_geometry(
    model: mujoco.MjModel,
    scene: mujoco.MjvScene,
    *,
    rotation: np.ndarray,
    anchor: np.ndarray,
    offset: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    mesh_ids: list[int] = []
    transforms: list[np.ndarray] = []
    tendon_endpoints: list[np.ndarray] = []
    tendon_radii: list[float] = []
    for source_index in range(scene.ngeom):
        source = scene.geoms[source_index]
        model_mesh = source.objtype == mujoco.mjtObj.mjOBJ_GEOM and source.type == mujoco.mjtGeom.mjGEOM_MESH
        tendon = source.objtype == mujoco.mjtObj.mjOBJ_TENDON and source.type == mujoco.mjtGeom.mjGEOM_CAPSULE
        if model_mesh:
            position, matrix = _visual_transform(source, rotation, anchor, offset)
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = matrix
            transform[:3, 3] = position
            mesh_ids.append(int(model.geom_dataid[int(source.objid)]))
            transforms.append(transform)
        elif tendon:
            position, matrix = _visual_transform(source, rotation, anchor, offset)
            axis = matrix[:, 2]
            half_length = float(source.size[2])
            tendon_endpoints.append(
                np.stack(
                    (
                        position - axis * half_length,
                        position + axis * half_length,
                    )
                )
            )
            tendon_radii.append(float(source.size[0]))
    return (
        mesh_ids,
        np.asarray(transforms, dtype=np.float32).reshape(-1, 4, 4),
        np.asarray(tendon_endpoints, dtype=np.float32).reshape(-1, 2, 3),
        np.asarray(tendon_radii, dtype=np.float32),
    )


def build_static_bundle(
    manifest: FigureManifest,
    *,
    cache_root: str | Path,
    output: str | Path,
    width: int,
    height: int,
    floor_grid: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Export the exact still-figure scene used as the animation foundation."""

    scale = width / manifest.layout.width
    expected_height = round(manifest.layout.height * scale)
    if expected_height != height:
        raise ValueError(
            "video dimensions must preserve the figure aspect ratio: "
            f"{width} px wide requires {expected_height} px high"
        )
    metadata, cells = export_scene_bundle(
        manifest,
        cache_root=cache_root,
        output=output,
        scale=scale,
    )
    if floor_grid is not None:
        output_path = Path(output).expanduser().resolve()
        with np.load(output_path, allow_pickle=False) as frozen:
            arrays = {key: np.asarray(frozen[key]) for key in frozen.files}
        metadata = {**metadata, "floor_grid": floor_grid}
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(output_path, **arrays)
    return metadata, cells


def export_animation_bundle(
    manifest: FigureManifest,
    *,
    cache_root: str | Path,
    static_bundle: str | Path,
    cell_provenance: list[dict[str, object]],
    output: str | Path,
    fps: int,
    global_frame_count: int,
    camera_frame_count: int,
    frame_start: int,
    frame_end: int,
    start_zoom: float,
    samples: int,
    device: str,
) -> dict[str, object]:
    """Add one compact range of live geometry and camera samples to a bundle."""

    if not 0 <= frame_start < frame_end <= global_frame_count:
        raise ValueError("invalid global animation frame range")
    if not 2 <= camera_frame_count <= global_frame_count:
        raise ValueError("camera_frame_count must be between two and global_frame_count")
    if fps <= 0 or samples <= 0:
        raise ValueError("fps and Cycles samples must be positive")
    if start_zoom < 1.0:
        raise ValueError("start_zoom must be at least one")
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unsupported Cycles device request: {device}")

    cache_path = Path(cache_root).expanduser().resolve()
    static_path = Path(static_bundle).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    with np.load(static_path, allow_pickle=False) as frozen:
        arrays = {key: np.asarray(frozen[key]) for key in frozen.files}
    metadata = json.loads(str(arrays["metadata"]))
    live_instance_indices = np.flatnonzero(~arrays["instance_history"])
    expected_mesh_ids = arrays["instance_mesh"][live_instance_indices]

    model = _model(manifest, int(metadata["width"]), int(metadata["height"]))
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    scene = mujoco.MjvScene(model, maxgeom=2500)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = manifest.layout.camera_azimuth
    camera.elevation = manifest.layout.camera_elevation

    from loco_mujoco.trajectory import Trajectory

    trajectories: list[np.ndarray] = []
    frequencies: list[float] = []
    anchors: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    for cell, provenance in zip(manifest.cells, cell_provenance, strict=True):
        if cell.frames is None:
            raise ValueError(f"{cell.motion}: video requires explicit display frames")
        trajectory_path = _artifact_paths(manifest, cache_path, cell.motion)[0]
        trajectory = Trajectory.load(str(trajectory_path))
        qpos = np.asarray(trajectory.data.qpos, dtype=np.float64)
        frequency = float(trajectory.info.frequency)
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise ValueError(f"{cell.motion}: qpos shape {qpos.shape} does not match model nq={model.nq}")
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(f"{cell.motion}: invalid trajectory frequency {frequency}")
        trajectories.append(qpos)
        frequencies.append(frequency)
        anchors.append(np.asarray(provenance["anchor_xy"], dtype=np.float64))
        rotations.append(rotation_z(float(provenance["yaw_degrees"])))
        offsets.append(cell_origin(manifest.layout, cell.row, cell.column))
        segment = cell.loop_frames or (int(cell.frames[0]), int(cell.frames[-1]))
        if not 0 <= segment[0] < segment[1] < len(qpos):
            raise ValueError(f"{cell.motion}: loop frames {segment} lie outside the {len(qpos)}-frame trajectory")
        segments.append(segment)

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

    range_length = frame_end - frame_start
    body_transforms = np.empty(
        (range_length, len(live_instance_indices), 4, 4),
        dtype=np.float32,
    )
    endpoints_by_frame: list[np.ndarray] = []
    radii_by_frame: list[np.ndarray] = []
    camera_position = np.empty((range_length, 3), dtype=np.float32)
    camera_forward = np.empty((range_length, 3), dtype=np.float32)
    camera_up = np.empty((range_length, 3), dtype=np.float32)
    verified_mesh_order = False
    maximum_tendon_segments = 0

    for local_index, global_index in enumerate(range(frame_start, frame_end)):
        elapsed = global_index / fps
        frame_mesh_ids: list[int] = []
        frame_transforms: list[np.ndarray] = []
        frame_endpoints: list[np.ndarray] = []
        frame_radii: list[np.ndarray] = []
        for qpos, frequency, anchor, rotation, offset, segment in zip(
            trajectories,
            frequencies,
            anchors,
            rotations,
            offsets,
            segments,
            strict=True,
        ):
            segment_start, segment_end = segment
            segment_length = segment_end - segment_start + 1
            source_frame = segment_start + (math.floor(elapsed * frequency + 1e-9) % segment_length)
            data.qpos[:] = qpos[source_frame]
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
            mesh_ids, transforms, tendon_endpoints, tendon_radii = _animated_geometry(
                model,
                scene,
                rotation=rotation,
                anchor=anchor,
                offset=offset,
            )
            frame_mesh_ids.extend(mesh_ids)
            frame_transforms.append(transforms)
            frame_endpoints.append(tendon_endpoints)
            frame_radii.append(tendon_radii)

        if not verified_mesh_order:
            actual_mesh_ids = np.asarray(frame_mesh_ids, dtype=np.int32)
            if not np.array_equal(actual_mesh_ids, expected_mesh_ids):
                raise RuntimeError("animated mesh ordering does not match the still-figure instances")
            verified_mesh_order = True
        body_transforms[local_index] = np.concatenate(frame_transforms, axis=0)
        endpoints = np.concatenate(frame_endpoints, axis=0)
        radii = np.concatenate(frame_radii, axis=0)
        endpoints_by_frame.append(endpoints)
        radii_by_frame.append(radii)
        maximum_tendon_segments = max(maximum_tendon_segments, len(endpoints))

        eased = _camera_progress(global_index, camera_frame_count)
        camera.lookat[:] = start_lookat + eased * (end_lookat - start_lookat)
        camera.distance = start_distance * math.exp(math.log(end_distance / start_distance) * eased)
        mujoco.mjv_updateCamera(model, data, camera, scene)
        payload = _camera_payload(scene, manifest)
        camera_position[local_index] = payload["position"]
        camera_forward[local_index] = payload["forward"]
        camera_up[local_index] = payload["up"]

    tendon_endpoints = np.zeros(
        (range_length, maximum_tendon_segments, 2, 3),
        dtype=np.float32,
    )
    tendon_radii = np.zeros(
        (range_length, maximum_tendon_segments),
        dtype=np.float32,
    )
    tendon_counts = np.empty(range_length, dtype=np.int32)
    for local_index, (endpoints, radii) in enumerate(zip(endpoints_by_frame, radii_by_frame, strict=True)):
        count = len(endpoints)
        tendon_endpoints[local_index, :count] = endpoints
        tendon_radii[local_index, :count] = radii
        tendon_counts[local_index] = count

    metadata["samples"] = samples
    metadata["video"] = {
        "fps": fps,
        "global_frame_count": global_frame_count,
        "camera_frame_count": camera_frame_count,
        "frame_start": frame_start,
        "frame_end_exclusive": frame_end,
        "start_zoom": start_zoom,
        "device": device,
        "camera_interpolation": "quintic-smootherstep-geometric-distance",
        "central_cell": {
            "row": manifest.cells[central_index].row,
            "column": manifest.cells[central_index].column,
            "motion": manifest.cells[central_index].motion,
        },
        "loop_frames": [list(segment) for segment in segments],
    }
    arrays.update(
        {
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
            "animation_global_frames": np.arange(frame_start, frame_end, dtype=np.int32),
            "animation_body_indices": live_instance_indices.astype(np.int32),
            "animation_body_transform": body_transforms,
            "animation_tendon_endpoints": tendon_endpoints,
            "animation_tendon_radius": tendon_radii,
            "animation_tendon_count": tendon_counts,
            "animation_camera_position": camera_position,
            "animation_camera_forward": camera_forward,
            "animation_camera_up": camera_up,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    return {
        "frame_start": frame_start,
        "frame_end_exclusive": frame_end,
        "frames": range_length,
        "live_body_instances": len(live_instance_indices),
        "maximum_tendon_segments": maximum_tendon_segments,
        "bundle_bytes": output_path.stat().st_size,
    }


def render_animation_bundle(
    *,
    blender_executable: str | Path,
    scene_script: str | Path,
    bundle: str | Path,
    frames_directory: str | Path,
    report: str | Path,
) -> dict[str, object]:
    """Invoke Blender for one already-exported animation range."""

    executable_value = str(blender_executable)
    executable_resolved = shutil.which(executable_value)
    blender = (
        Path(executable_resolved) if executable_resolved is not None else Path(executable_value).expanduser().resolve()
    )
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(Path(scene_script).expanduser().resolve()),
        "--",
        "--bundle",
        str(Path(bundle).expanduser().resolve()),
        "--frames-dir",
        str(Path(frames_directory).expanduser().resolve()),
        "--report",
        str(Path(report).expanduser().resolve()),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    report_path = Path(report)
    if completed.returncode != 0 or not report_path.is_file():
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"Blender animation render failed with exit code {completed.returncode}:\n{detail[-12000:]}")
    result = json.loads(report_path.read_text())
    result["blender_stdout_tail"] = completed.stdout[-2000:]
    return result


__all__ = [
    "build_static_bundle",
    "export_animation_bundle",
    "render_animation_bundle",
]
