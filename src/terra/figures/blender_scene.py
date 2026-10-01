"""Build and render a TERRA scene bundle inside Blender.

This module intentionally imports only Blender-bundled packages. It is invoked
by ``blender --background --python`` rather than imported by the TERRA runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _arguments() -> argparse.Namespace:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(sys.argv[separator + 1 :])


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def _principled_material(
    name: str,
    rgba: tuple[float, float, float, float],
    *,
    roughness: float,
    transparent: bool,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = rgba
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*rgba[:3], 1.0)
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Specular IOR Level"].default_value = 0.34
    if emission_strength > 0.0:
        principled.inputs["Emission Color"].default_value = (*rgba[:3], 1.0)
        principled.inputs["Emission Strength"].default_value = emission_strength
    if transparent and rgba[3] < 1.0:
        transparent_node = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = rgba[3]
        links.new(transparent_node.outputs[0], mix.inputs[1])
        links.new(principled.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs[0])
        material.surface_render_method = "DITHERED"
        material.use_transparent_shadow = True
    else:
        links.new(principled.outputs[0], output.inputs[0])
    return material


def _material_cache():
    cache: dict[tuple[object, ...], bpy.types.Material] = {}

    def get(rgba: np.ndarray, kind: str) -> bpy.types.Material:
        rounded = tuple(round(float(value), 3) for value in rgba)
        key = (kind, *rounded)
        if key not in cache:
            roughness = {
                "pad": 0.72,
                "floor-grid": 0.78,
                "terrain": 0.58,
                "ghost": 0.48,
                "body": 0.42,
                "tendon": 0.38,
                "muscle-glow": 0.24,
                "dof-arrow": 0.28,
                "landmark-sphere": 0.30,
                "source": 0.52,
                "overlay": 0.46,
                "candidate": 0.62,
            }.get(kind, 0.52)
            cache[key] = _principled_material(
                f"{kind}-{'-'.join(f'{value:.3f}' for value in rounded)}",
                rounded,
                roughness=roughness,
                transparent=kind in {"ghost", "source", "overlay", "candidate"},
                emission_strength=(
                    0.825
                    if kind == "muscle-glow"
                    else 1.15 if kind == "dof-arrow"
                    else 1.0 if kind == "landmark-sphere"
                    else 0.22 if kind == "overlay" else 0.0
                ),
            )
        return cache[key]

    return get


def _box(
    name: str,
    position: np.ndarray,
    half_size: np.ndarray,
    rotation: np.ndarray,
    material: bpy.types.Material,
    bevel_width: float,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=position)
    obj = bpy.context.object
    obj.name = name
    transform = np.eye(4)
    transform[:3, :3] = rotation @ np.diag(half_size)
    transform[:3, 3] = position
    obj.matrix_world = Matrix(transform.tolist())
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("publication bevel", "BEVEL")
    bevel.width = min(bevel_width, float(np.min(half_size)) * 0.45)
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    bevel.harden_normals = True


def _floor_grid(bundle, metadata: dict[str, object], material_for) -> None:
    """Add a metrically spaced grid just above the largest horizontal pad.

    Overview bundles contain both a local cell pad and a much larger horizon
    pad. Drawing the grid only once on the horizon avoids coincident geometry
    while retaining an exact, world-aligned metric reference.
    """

    specification = metadata.get("floor_grid")
    if not isinstance(specification, dict):
        return
    spacing = float(specification["spacing"])
    line_width = float(specification["line_width"])
    rgba = np.asarray(specification["rgba"], dtype=float)
    if spacing <= 0.0 or line_width <= 0.0:
        raise ValueError("floor-grid spacing and line width must be positive")
    if rgba.shape != (4,):
        raise ValueError(f"floor-grid RGBA must have four components, got {rgba.shape}")

    candidates: list[tuple[float, int]] = []
    for index, kind_value in enumerate(bundle["boxes_kind"]):
        if str(kind_value) != "pad":
            continue
        rotation = np.asarray(bundle["boxes_rotation"][index], dtype=float)
        if not np.allclose(rotation, np.eye(3), atol=1e-5):
            continue
        half_size = np.asarray(bundle["boxes_size"][index], dtype=float)
        candidates.append((float(half_size[0] * half_size[1]), index))
    if not candidates:
        return

    index = max(candidates)[1]
    center = np.asarray(bundle["boxes_position"][index], dtype=float)
    half_size = np.asarray(bundle["boxes_size"][index], dtype=float)
    x_min, x_max = center[0] - half_size[0], center[0] + half_size[0]
    y_min, y_max = center[1] - half_size[1], center[1] + half_size[1]
    z = center[2] + half_size[2] + float(specification.get("z_offset", 0.002))
    half_width = 0.5 * line_width

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []

    def add_quad(corners: tuple[tuple[float, float, float], ...]) -> None:
        start = len(vertices)
        vertices.extend(corners)
        faces.append((start, start + 1, start + 2, start + 3))

    x_first = math.ceil(x_min / spacing) * spacing
    x_last = math.floor(x_max / spacing) * spacing
    for x in np.arange(x_first, x_last + spacing * 0.5, spacing):
        add_quad(
            (
                (float(x - half_width), float(y_min), float(z)),
                (float(x + half_width), float(y_min), float(z)),
                (float(x + half_width), float(y_max), float(z)),
                (float(x - half_width), float(y_max), float(z)),
            )
        )
    y_first = math.ceil(y_min / spacing) * spacing
    y_last = math.floor(y_max / spacing) * spacing
    for y in np.arange(y_first, y_last + spacing * 0.5, spacing):
        add_quad(
            (
                (float(x_min), float(y - half_width), float(z)),
                (float(x_max), float(y - half_width), float(z)),
                (float(x_max), float(y + half_width), float(z)),
                (float(x_min), float(y + half_width), float(z)),
            )
        )

    mesh = bpy.data.meshes.new("publication-floor-grid")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material_for(rgba, "floor-grid"))
    mesh.update()
    obj = bpy.data.objects.new("publication-floor-grid", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.visible_shadow = False


def _ramp_wedge(
    name: str,
    position: np.ndarray,
    half_size: np.ndarray,
    rotation: np.ndarray,
    material: bpy.types.Material,
    bevel_width: float,
) -> None:
    """Render a pitched collision box as a solid wedge with an exact top plane.

    Rotating a cuboid also tilts its underside, which exposes a dark triangular
    undercut where the ramp meets a flat landing. The publication mesh keeps the
    four transformed top vertices unchanged and extends them vertically to the
    lowest corner of the original cuboid. Simulation geometry is unaffected.
    """

    x, y, z = (float(value) for value in half_size)
    top_local = np.asarray(
        ((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)),
        dtype=float,
    )
    top = top_local @ rotation.T + position
    all_local = np.asarray(
        [(sx * x, sy * y, sz * z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )
    bottom_z = float(np.min((all_local @ rotation.T + position)[:, 2]))
    bottom = top.copy()
    bottom[:, 2] = bottom_z
    vertices = np.vstack((top, bottom))
    faces = (
        (0, 1, 2, 3),
        (7, 6, 5, 4),
        (4, 5, 1, 0),
        (5, 6, 2, 1),
        (6, 7, 3, 2),
        (7, 4, 0, 3),
    )
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bevel = obj.modifiers.new("publication bevel", "BEVEL")
    bevel.width = min(bevel_width, float(np.min(half_size)) * 0.45)
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    bevel.harden_normals = True


def _mesh_data(bundle, mesh_id: int, material: bpy.types.Material, variant: str) -> bpy.types.Mesh:
    vertices = bundle[f"mesh_{mesh_id}_vertices"]
    faces = bundle[f"mesh_{mesh_id}_faces"]
    mesh = bpy.data.meshes.new(f"myofullbody-{mesh_id}-{variant}")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.update()
    return mesh


def _bodies(bundle, material_for) -> None:
    mesh_cache: dict[tuple[int, str, tuple[float, ...]], bpy.types.Mesh] = {}
    for index, mesh_id_value in enumerate(bundle["instance_mesh"]):
        mesh_id = int(mesh_id_value)
        rgba = bundle["instance_rgba"][index]
        history = bool(bundle["instance_history"][index])
        kind = "ghost" if history else "body"
        color_key = tuple(round(float(value), 3) for value in rgba)
        key = (mesh_id, kind, color_key)
        if key not in mesh_cache:
            material = material_for(rgba, kind)
            mesh_cache[key] = _mesh_data(bundle, mesh_id, material, f"{kind}-{len(mesh_cache)}")
        obj = bpy.data.objects.new(f"{kind}-{index:05d}", mesh_cache[key])
        bpy.context.scene.collection.objects.link(obj)
        obj.matrix_world = Matrix(bundle["instance_transform"][index].tolist())


def _source_bodies(bundle, material_for) -> None:
    """Add optional world-space SMPL-H meshes to a diagnostic bundle."""

    if "source_vertices" not in bundle.files:
        return
    faces = bundle["source_faces"]
    for index, (vertices, rgba) in enumerate(zip(bundle["source_vertices"], bundle["source_rgba"], strict=True)):
        mesh = bpy.data.meshes.new(f"smplh-{index:02d}")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.materials.append(material_for(rgba, "source"))
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        mesh.update()
        obj = bpy.data.objects.new(f"smplh-{index:02d}", mesh)
        bpy.context.scene.collection.objects.link(obj)


def _overlay_points(bundle, material_for, *, material_kind: str = "overlay") -> None:
    """Render optional scientific landmarks as instanced icospheres."""

    if "overlay_points_position" not in bundle.files:
        return
    positions = bundle["overlay_points_position"]
    radii = bundle["overlay_points_radius"]
    rgba_values = bundle["overlay_points_rgba"]
    groups: dict[tuple[float, ...], bpy.types.Mesh] = {}
    for index, (position, radius, rgba) in enumerate(zip(positions, radii, rgba_values, strict=True)):
        color = tuple(round(float(value), 3) for value in rgba)
        key = (*color, round(float(radius), 4))
        if key not in groups:
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=float(radius))
            template = bpy.context.object
            template.name = f"overlay-point-template-{len(groups):02d}"
            template.data.name = template.name
            template.data.materials.append(material_for(rgba, material_kind))
            groups[key] = template.data
            bpy.data.objects.remove(template, do_unlink=True)
        obj = bpy.data.objects.new(f"overlay-point-{index:04d}", groups[key])
        bpy.context.scene.collection.objects.link(obj)
        obj.location = position
        obj.visible_shadow = False


def _overlay_segments(bundle, material_for, *, material_kind: str = "overlay") -> None:
    """Render optional correspondence, trajectory, and evidence line segments."""

    if "overlay_segments_endpoints" not in bundle.files:
        return
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, (radius, rgba) in enumerate(
        zip(bundle["overlay_segments_radius"], bundle["overlay_segments_rgba"], strict=True)
    ):
        key = (*tuple(round(float(value), 3) for value in rgba), round(float(radius), 4))
        groups.setdefault(key, []).append(index)
    for group_index, indices in enumerate(groups.values()):
        first = indices[0]
        radius = float(bundle["overlay_segments_radius"][first])
        rgba = bundle["overlay_segments_rgba"][first]
        curve = bpy.data.curves.new(f"overlay-segments-{group_index:02d}", type="CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = radius
        curve.bevel_resolution = 2
        curve.materials.append(material_for(rgba, material_kind))
        for index in indices:
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            for point, coordinate in zip(
                spline.points,
                bundle["overlay_segments_endpoints"][index],
                strict=True,
            ):
                point.co = (*coordinate, 1.0)
        obj = bpy.data.objects.new(f"overlay-segments-{group_index:02d}", curve)
        bpy.context.scene.collection.objects.link(obj)
        obj.visible_shadow = False


def _overlay_arrows(bundle, material_for, *, material_kind: str = "overlay") -> None:
    """Render optional vector arrows as cylindrical shafts with conical heads."""

    if "overlay_arrows_endpoints" not in bundle.files:
        return
    endpoints = bundle["overlay_arrows_endpoints"]
    radii = bundle["overlay_arrows_radius"]
    rgba_values = bundle["overlay_arrows_rgba"]
    head_lengths = bundle["overlay_arrows_head_length"] if "overlay_arrows_head_length" in bundle.files else 5.0 * radii
    head_radii = bundle["overlay_arrows_head_radius"] if "overlay_arrows_head_radius" in bundle.files else 2.4 * radii
    for index, (pair, radius, rgba, head_length, head_radius) in enumerate(
        zip(endpoints, radii, rgba_values, head_lengths, head_radii, strict=True)
    ):
        start = Vector(pair[0])
        end = Vector(pair[1])
        direction = end - start
        length = direction.length
        if length <= 1e-8:
            continue
        direction.normalize()
        head_length = min(float(head_length), 0.45 * length)
        shaft_end = end - head_length * direction
        material = material_for(rgba, material_kind)

        curve = bpy.data.curves.new(f"overlay-arrow-shaft-{index:03d}", type="CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = float(radius)
        curve.bevel_resolution = 3
        curve.materials.append(material)
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        for point, coordinate in zip(spline.points, (start, shaft_end), strict=True):
            point.co = (*coordinate, 1.0)
        shaft = bpy.data.objects.new(f"overlay-arrow-shaft-{index:03d}", curve)
        bpy.context.scene.collection.objects.link(shaft)
        shaft.visible_shadow = False

        midpoint = shaft_end + 0.5 * head_length * direction
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=float(head_radius),
            radius2=0.0,
            depth=head_length,
            location=midpoint,
        )
        head = bpy.context.object
        head.name = f"overlay-arrow-head-{index:03d}"
        head.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        head.data.materials.append(material)
        head.visible_shadow = False


def _tendons(bundle, material_for, *, material_kind: str = "tendon") -> None:
    if len(bundle["tendon_radius"]) == 0:
        return
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, rgba in enumerate(bundle["tendon_rgba"]):
        key = tuple(round(float(value), 3) for value in rgba)
        groups.setdefault(key, []).append(index)
    for group_index, (rgba, indices) in enumerate(groups.items()):
        curve = bpy.data.curves.new(f"tendons-{group_index}", type="CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = 0.005
        curve.bevel_resolution = 2
        curve.resolution_u = 1
        curve.materials.append(material_for(np.asarray(rgba), material_kind))
        for index in indices:
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            endpoints = bundle["tendon_endpoints"][index]
            radius = float(bundle["tendon_radius"][index]) / curve.bevel_depth
            for point, coordinate in zip(spline.points, endpoints, strict=True):
                point.co = (*coordinate, 1.0)
                point.radius = radius
        obj = bpy.data.objects.new(f"tendons-{group_index}", curve)
        bpy.context.scene.collection.objects.link(obj)


def _point_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
    color: tuple[float, float, float],
) -> None:
    light = bpy.data.lights.new(name, type="AREA")
    light.energy = energy
    light.shape = "DISK"
    light.size = size
    light.color = color
    light.use_shadow = True
    light.use_shadow_jitter = True
    light.shadow_jitter_overblur = 20.0
    obj = bpy.data.objects.new(name, light)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    _point_at(obj, (0.0, 0.0, 0.5))


def _shadow_catcher(metadata: dict[str, object]) -> None:
    """Add an optional transparent-film plane that retains contact shadows."""

    specification = metadata.get("shadow_catcher")
    if not isinstance(specification, dict):
        return
    size = float(specification.get("size", 3.0))
    z = float(specification.get("z", -0.012))
    if size <= 0.0:
        raise ValueError("shadow-catcher size must be positive")
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = "publication-shadow-catcher"
    obj.is_shadow_catcher = True


def _camera(metadata: dict[str, object]) -> None:
    camera_meta = metadata["camera"]
    camera_data = bpy.data.cameras.new("publication-camera")
    projection = str(camera_meta.get("projection", "perspective")).casefold()
    if projection == "orthographic":
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = float(camera_meta["ortho_scale"])
    else:
        camera_data.type = "PERSP"
        camera_data.sensor_fit = "VERTICAL"
        camera_data.sensor_height = 32.0
        camera_data.lens = camera_data.sensor_height / (
            2.0 * math.tan(math.radians(float(camera_meta["fovy_degrees"])) / 2.0)
        )
    camera_data.dof.use_dof = False
    camera = bpy.data.objects.new("publication-camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = camera_meta["position"]
    forward = Vector(camera_meta["forward"])
    up = Vector(camera_meta["up"])
    right = forward.cross(up).normalized()
    corrected_up = right.cross(forward).normalized()
    rotation = Matrix((right, corrected_up, -forward)).transposed()
    camera.rotation_euler = rotation.to_euler()
    bpy.context.scene.camera = camera


def _cycles_device() -> str:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.refresh_devices()
        except TypeError:
            continue
        candidates = [device for device in preferences.devices if device.type == backend]
        if not candidates:
            continue
        preferred = next((device for device in candidates if "RTX" in device.name), candidates[0])
        for device in preferences.devices:
            device.use = device == preferred
        return f"{preferred.type}:{preferred.name}"
    return "CPU"


def _render_settings(metadata: dict[str, object], output: Path) -> dict[str, object]:
    scene = bpy.context.scene
    device = _cycles_device()
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(metadata["width"])
    scene.render.resolution_y = int(metadata["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    transparent_background = bool(metadata.get("transparent_background", False))
    scene.render.image_settings.color_mode = "RGBA" if transparent_background else "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output)
    scene.render.film_transparent = transparent_background
    scene.render.use_persistent_data = True
    scene.render.image_settings.color_management = "FOLLOW_SCENE"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = float(metadata.get("exposure", 0.15))
    samples = int(metadata.get("samples", 72 if int(metadata["width"]) < 3000 else 144))
    scene.cycles.device = "GPU" if device != "CPU" else "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_light_tree = True
    background = metadata["background"]
    ambient_background = metadata.get("ambient_background", background)
    world = bpy.data.worlds.new("publication-world")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    ambient = nodes.new("ShaderNodeBackground")
    ambient.inputs["Color"].default_value = (*ambient_background, 1.0)
    ambient.inputs["Strength"].default_value = float(metadata.get("ambient_strength", 0.18))
    visible = nodes.new("ShaderNodeBackground")
    visible.inputs["Color"].default_value = (*background, 1.0)
    visible.inputs["Strength"].default_value = 2.00
    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(ambient.outputs[0], mix.inputs[1])
    links.new(visible.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])
    scene.world = world
    return {
        "engine": "CYCLES",
        "device": device,
        "samples": samples,
        "denoising": "OPENIMAGEDENOISE",
        "transparent_background": transparent_background,
    }


def main() -> None:
    args = _arguments()
    _clear_scene()
    with np.load(args.bundle, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"]))
        material_for = _material_cache()
        for index, position in enumerate(bundle["boxes_position"]):
            kind = str(bundle["boxes_kind"][index])
            pitched = (
                bool(bundle["boxes_pitched"][index])
                if "boxes_pitched" in bundle.files
                else kind == "terrain" and abs(float(bundle["boxes_rotation"][index][2, 0])) > 1e-6
            )
            add_box = _ramp_wedge if pitched and kind == "terrain" else _box
            add_box(
                f"{kind}-{index:03d}",
                position,
                bundle["boxes_size"][index],
                bundle["boxes_rotation"][index],
                material_for(bundle["boxes_rgba"][index], kind),
                float(metadata["bevel"].get(kind, 0.012)),
            )
        _floor_grid(bundle, metadata, material_for)
        _bodies(bundle, material_for)
        _source_bodies(bundle, material_for)
        _tendons(
            bundle,
            material_for,
            material_kind=str(metadata.get("tendon_material_kind", "tendon")),
        )
        overlay_material_kind = str(metadata.get("overlay_material_kind", "overlay"))
        _overlay_segments(bundle, material_for, material_kind=overlay_material_kind)
        _overlay_arrows(bundle, material_for, material_kind=overlay_material_kind)
        _overlay_points(
            bundle,
            material_for,
            material_kind=str(metadata.get("overlay_points_material_kind", overlay_material_kind)),
        )
        _shadow_catcher(metadata)
        _camera(metadata)
        report = _render_settings(metadata, args.out)
        report["blender_version"] = bpy.app.version_string
        report["shadow_catcher"] = isinstance(metadata.get("shadow_catcher"), dict)

    direct_light_scale = float(metadata.get("direct_light_scale", 1.0))
    _area_light("key", (-9.0, -10.0, 18.0), 1950.0 * direct_light_scale, 7.0, (1.0, 0.91, 0.80))
    _area_light("fill", (13.0, -2.0, 10.0), 380.0 * direct_light_scale, 9.0, (0.72, 0.83, 1.0))
    _area_light("rim", (2.0, 13.0, 15.0), 520.0 * direct_light_scale, 8.0, (1.0, 0.93, 0.84))
    bpy.ops.render.render(write_still=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
