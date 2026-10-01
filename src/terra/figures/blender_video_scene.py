"""Render an animated TERRA scene bundle inside Blender Cycles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_scene as still


def _arguments() -> argparse.Namespace:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[separator + 1 :])


def _bodies(bundle, material_for) -> list[bpy.types.Object]:
    dynamic_indices = {int(value) for value in bundle["animation_body_indices"]}
    mesh_cache: dict[tuple[int, str, tuple[float, ...]], bpy.types.Mesh] = {}
    live_objects: list[bpy.types.Object] = []
    for index, mesh_id_value in enumerate(bundle["instance_mesh"]):
        mesh_id = int(mesh_id_value)
        rgba = bundle["instance_rgba"][index]
        history = bool(bundle["instance_history"][index])
        kind = "ghost" if history else "body"
        color_key = tuple(round(float(value), 3) for value in rgba)
        key = (mesh_id, kind, color_key)
        if key not in mesh_cache:
            material = material_for(rgba, kind)
            mesh_cache[key] = still._mesh_data(
                bundle,
                mesh_id,
                material,
                f"{kind}-{len(mesh_cache)}",
            )
        obj = bpy.data.objects.new(f"{kind}-{index:05d}", mesh_cache[key])
        bpy.context.scene.collection.objects.link(obj)
        obj.matrix_world = Matrix(bundle["instance_transform"][index].tolist())
        if index in dynamic_indices:
            live_objects.append(obj)
    if len(live_objects) != len(dynamic_indices):
        raise RuntimeError(f"created {len(live_objects)} live bodies for {len(dynamic_indices)} indices")
    return live_objects


def _animated_tendons(
    endpoints: np.ndarray,
    radii: np.ndarray,
    material: bpy.types.Material,
) -> bpy.types.Curve:
    curve = bpy.data.curves.new("animated-tendons", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = 0.005
    curve.bevel_resolution = 2
    curve.materials.append(material)
    for segment_endpoints, radius in zip(endpoints, radii, strict=True):
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        point_radius = float(radius) / curve.bevel_depth
        for point, coordinate in zip(
            spline.points,
            segment_endpoints,
            strict=True,
        ):
            point.co = (*coordinate, 1.0)
            point.radius = point_radius
    obj = bpy.data.objects.new("animated-tendons", curve)
    bpy.context.scene.collection.objects.link(obj)
    return curve


def _update_tendons(
    curve: bpy.types.Curve,
    endpoints: np.ndarray,
    radii: np.ndarray,
) -> None:
    inverse_bevel = 1.0 / curve.bevel_depth
    for spline, segment_endpoints, radius in zip(
        curve.splines,
        endpoints,
        radii,
        strict=True,
    ):
        point_radius = float(radius) * inverse_bevel
        for point, coordinate in zip(
            spline.points,
            segment_endpoints,
            strict=True,
        ):
            point.co = (*coordinate, 1.0)
            point.radius = point_radius
    curve.update_tag()


def _set_camera_pose(
    camera: bpy.types.Object,
    position: np.ndarray,
    forward_value: np.ndarray,
    up_value: np.ndarray,
) -> None:
    camera.location = position
    forward = Vector(forward_value)
    up = Vector(up_value)
    right = forward.cross(up).normalized()
    corrected_up = right.cross(forward).normalized()
    rotation = Matrix((right, corrected_up, -forward)).transposed()
    camera.rotation_euler = rotation.to_euler()


def _configure_device(requested: str, report: dict[str, object]) -> None:
    scene = bpy.context.scene
    detected = str(report["device"])
    if requested == "gpu" and detected == "CPU":
        raise RuntimeError("Cycles GPU rendering was requested, but Blender detected only the CPU")
    if requested == "cpu":
        scene.cycles.device = "CPU"
        report["device"] = "CPU"


def main() -> None:
    args = _arguments()
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    still._clear_scene()
    with np.load(args.bundle, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"]))
        video = metadata["video"]
        material_for = still._material_cache()
        for index, position in enumerate(bundle["boxes_position"]):
            kind = str(bundle["boxes_kind"][index])
            pitched = (
                bool(bundle["boxes_pitched"][index])
                if "boxes_pitched" in bundle.files
                else kind == "terrain" and abs(float(bundle["boxes_rotation"][index][2, 0])) > 1e-6
            )
            add_box = still._ramp_wedge if pitched and kind == "terrain" else still._box
            add_box(
                f"{kind}-{index:03d}",
                position,
                bundle["boxes_size"][index],
                bundle["boxes_rotation"][index],
                material_for(bundle["boxes_rgba"][index], kind),
                float(metadata["bevel"].get(kind, 0.012)),
            )
        still._floor_grid(bundle, metadata, material_for)
        live_objects = _bodies(bundle, material_for)
        initial_endpoints = bundle["animation_tendon_endpoints"][0]
        initial_radii = bundle["animation_tendon_radius"][0]
        tendon_rgba = np.asarray(bundle["tendon_rgba"][0], dtype=float).copy()
        tendon_rgba[3] = 1.0
        tendon_curve = _animated_tendons(
            initial_endpoints,
            initial_radii,
            material_for(tendon_rgba, "tendon"),
        )
        still._source_bodies(bundle, material_for)
        still._overlay_segments(bundle, material_for)
        still._overlay_points(bundle, material_for)
        still._shadow_catcher(metadata)
        still._camera(metadata)
        camera = bpy.context.scene.camera
        report = still._render_settings(metadata, args.frames_dir / "frame.png")
        _configure_device(str(video["device"]), report)
        report["blender_version"] = bpy.app.version_string
        report["shadow_catcher"] = isinstance(metadata.get("shadow_catcher"), dict)

        direct_light_scale = float(metadata.get("direct_light_scale", 1.0))
        still._area_light(
            "key",
            (-9.0, -10.0, 18.0),
            1950.0 * direct_light_scale,
            7.0,
            (1.0, 0.91, 0.80),
        )
        still._area_light(
            "fill",
            (13.0, -2.0, 10.0),
            380.0 * direct_light_scale,
            9.0,
            (0.72, 0.83, 1.0),
        )
        still._area_light(
            "rim",
            (2.0, 13.0, 15.0),
            520.0 * direct_light_scale,
            8.0,
            (1.0, 0.93, 0.84),
        )

        scene = bpy.context.scene
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.use_file_extension = True
        scene.render.fps = int(video["fps"])
        rendered: list[int] = []
        skipped: list[int] = []
        global_frames = bundle["animation_global_frames"]
        for local_index, global_value in enumerate(global_frames):
            global_index = int(global_value)
            destination = args.frames_dir / f"frame-{global_index:06d}.png"
            if destination.is_file() and destination.stat().st_size > 0:
                skipped.append(global_index)
                continue
            for obj, transform in zip(
                live_objects,
                bundle["animation_body_transform"][local_index],
                strict=True,
            ):
                obj.matrix_world = Matrix(transform.tolist())
            endpoints = bundle["animation_tendon_endpoints"][local_index]
            radii = bundle["animation_tendon_radius"][local_index]
            _update_tendons(tendon_curve, endpoints, radii)
            _set_camera_pose(
                camera,
                bundle["animation_camera_position"][local_index],
                bundle["animation_camera_forward"][local_index],
                bundle["animation_camera_up"][local_index],
            )
            scene.frame_set(global_index + 1)
            temporary = args.frames_dir / f".frame-{global_index:06d}.render.png"
            scene.render.filepath = str(temporary)
            bpy.ops.render.render(write_still=True)
            if not temporary.is_file():
                raise RuntimeError(f"Cycles did not produce {temporary}")
            os.replace(temporary, destination)
            rendered.append(global_index)
            print(
                f"rendered global Blender frame {global_index + 1}/{video['global_frame_count']}",
                flush=True,
            )

        report.update(
            {
                "frame_start": int(video["frame_start"]),
                "frame_end_exclusive": int(video["frame_end_exclusive"]),
                "rendered_frames": rendered,
                "skipped_frames": skipped,
                "live_body_instances": len(live_objects),
                "maximum_tendon_segments": int(bundle["animation_tendon_endpoints"].shape[1]),
                "tendon_primitive": "beveled-poly-curve",
                "camera_interpolation": video["camera_interpolation"],
            }
        )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
