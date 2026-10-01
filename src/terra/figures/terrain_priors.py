"""Publication rendering for the geometric terrain priors used by TERRA."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from terra._files import atomic_write
from terra.terrain.ramps import RAMP_BURIED_DEPTH, RAMP_SLOPE_RANGE, RAMP_WIDTH_PRIOR
from terra.terrain.seats import SEAT_MAX_HEIGHT, SEAT_MIN_HEIGHT, SEAT_SIZE_PRIOR
from terra.terrain.stairs import STAIR_MIN_LEVELS, STAIR_WIDTH_PRIOR

BACKGROUND = (0.975, 0.973, 0.963)
PRIOR_BLUE = (0.055, 0.36, 0.56, 1.0)
PRIOR_BLUE_LIGHT = (0.13, 0.51, 0.70, 1.0)
PRIOR_BLUE_DARK = (0.035, 0.24, 0.39, 1.0)
PRIOR_NAMES = ("boxes", "ramps", "staircases", "chairs")


@dataclass(frozen=True, slots=True)
class Primitive:
    """One beveled cuboid or pitched terrain wedge in a Blender scene bundle."""

    position: tuple[float, float, float]
    half_size: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    rgba: tuple[float, float, float, float]
    kind: str = "terrain"
    pitched: bool = False
    role: str = "inferred terrain"


def _rotation_y(angle_radians: float) -> tuple[tuple[float, float, float], ...]:
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    return (
        (cosine, 0.0, sine),
        (0.0, 1.0, 0.0),
        (-sine, 0.0, cosine),
    )


IDENTITY = _rotation_y(0.0)


def box_primitives() -> tuple[Primitive, ...]:
    """Return independent support boxes representative of the generic level prior."""

    return (
        Primitive((-0.56, 0.08, 0.18), (0.34, 0.39, 0.18), IDENTITY, PRIOR_BLUE_LIGHT),
        Primitive((0.12, -0.08, 0.32), (0.40, 0.35, 0.32), IDENTITY, PRIOR_BLUE),
        Primitive((0.75, 0.13, 0.13), (0.22, 0.28, 0.13), IDENTITY, PRIOR_BLUE_DARK),
    )


def ramp_primitives(slope_degrees: float = 24.0) -> tuple[Primitive, ...]:
    """Return the fitted pitched box as a vertical-skirt ramp plus its landing."""

    if not RAMP_SLOPE_RANGE[0] <= slope_degrees <= RAMP_SLOPE_RANGE[1]:
        raise ValueError(f"ramp slope must lie in {RAMP_SLOPE_RANGE}, got {slope_degrees}")
    length = 1.38
    slope = math.tan(math.radians(slope_degrees))
    rise = length * slope
    pitch = -math.atan(slope)
    cosine, sine = math.cos(pitch), math.sin(pitch)
    longitudinal_half_size = 0.5 * length / cosine
    vertical_half_size = 0.5 * rise / cosine + RAMP_BURIED_DEPTH
    normal = np.asarray((sine, 0.0, cosine), dtype=float)
    centre = np.asarray((0.0, 0.0, 0.5 * rise), dtype=float) - vertical_half_size * normal
    landing_length = 0.38
    return (
        Primitive(
            tuple(float(value) for value in centre),
            (longitudinal_half_size, 0.5 * RAMP_WIDTH_PRIOR, vertical_half_size),
            _rotation_y(pitch),
            PRIOR_BLUE,
            pitched=True,
            role="inferred continuous incline",
        ),
        Primitive(
            (0.5 * length + 0.5 * landing_length, 0.0, 0.5 * rise),
            (0.5 * landing_length, 0.5 * RAMP_WIDTH_PRIOR, 0.5 * rise),
            IDENTITY,
            PRIOR_BLUE_DARK,
            role="terminal landing",
        ),
    )


def staircase_primitives() -> tuple[Primitive, ...]:
    """Return four solid treads obeying the shared-riser staircase prior."""

    count = max(4, STAIR_MIN_LEVELS)
    tread_depth = 0.36
    riser = 0.17
    first = -0.5 * count * tread_depth
    colors = (PRIOR_BLUE_LIGHT, PRIOR_BLUE, PRIOR_BLUE, PRIOR_BLUE_DARK)
    return tuple(
        Primitive(
            (first + (index + 0.5) * tread_depth, 0.0, 0.5 * (index + 1) * riser),
            (0.5 * tread_depth, 0.5 * STAIR_WIDTH_PRIOR, 0.5 * (index + 1) * riser),
            IDENTITY,
            colors[index],
            role="shared-riser tread",
        )
        for index in range(count)
    )


def chair_primitives() -> tuple[Primitive, ...]:
    """Return exactly the solid seat-support box inferred by TERRA."""

    seat_height = 0.46
    if not SEAT_MIN_HEIGHT <= seat_height <= SEAT_MAX_HEIGHT:
        raise AssertionError("illustrative chair height violates the TERRA seat prior")
    depth_half, width_half = SEAT_SIZE_PRIOR
    return (
        Primitive(
            (0.0, 0.0, 0.5 * seat_height),
            (depth_half, width_half, 0.5 * seat_height),
            IDENTITY,
            PRIOR_BLUE,
            role="inferred solid seat support",
        ),
    )


def prior_primitives(name: str) -> tuple[Primitive, ...]:
    """Return the primitives for one named terrain family."""

    builders = {
        "boxes": box_primitives,
        "ramps": ramp_primitives,
        "staircases": staircase_primitives,
        "chairs": chair_primitives,
    }
    try:
        return builders[name]()
    except KeyError as error:
        raise ValueError(f"unknown terrain prior {name!r}; expected one of {PRIOR_NAMES}") from error


def _primitive_points(primitives: tuple[Primitive, ...]) -> np.ndarray:
    """Return visible presentation-mesh corners for deterministic framing."""

    points = []
    signs = (-1.0, 1.0)
    for primitive in primitives:
        position = np.asarray(primitive.position, dtype=float)
        half_size = np.asarray(primitive.half_size, dtype=float)
        rotation = np.asarray(primitive.rotation, dtype=float)
        if primitive.pitched:
            x, y, z = half_size
            top = np.asarray(((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z))) @ rotation.T + position
            bottom = top.copy()
            bottom[:, 2] = 0.0
            corners = np.concatenate((top, bottom), axis=0)
        else:
            corners = (
                np.asarray(
                    [
                        (sx * half_size[0], sy * half_size[1], sz * half_size[2])
                        for sx in signs
                        for sy in signs
                        for sz in signs
                    ]
                )
                @ rotation.T
                + position
            )
        corners[:, 2] = np.maximum(corners[:, 2], 0.0)
        points.append(corners)
    return np.concatenate(points, axis=0)


def _camera_payload(
    primitives: tuple[Primitive, ...],
    *,
    width: int,
    height: int,
    usable_fraction: float = 0.58,
) -> dict[str, object]:
    """Fit one shared view direction tightly around a terrain family."""

    azimuth = math.radians(43.0)
    elevation = math.radians(-27.0)
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
    points = _primitive_points(primitives)
    projected_right = points @ right
    projected_up = points @ up
    centre_right = 0.5 * (float(np.min(projected_right)) + float(np.max(projected_right)))
    centre_up = 0.5 * (float(np.min(projected_up)) + float(np.max(projected_up)))
    lookat = np.mean(points, axis=0)
    lookat += (centre_right - float(lookat @ right)) * right
    lookat += (centre_up - float(lookat @ up)) * up
    aspect = width / height
    # The remaining margin is not decorative whitespace: it retains the area
    # lights' soft contact shadows.
    ortho_scale = (
        max(
            float(np.ptp(projected_right)),
            float(np.ptp(projected_up)) * aspect,
        )
        / usable_fraction
    )
    distance = 4.8
    return {
        "position": (lookat - distance * forward).tolist(),
        "forward": forward.tolist(),
        "up": up.tolist(),
        "fovy_degrees": 38.0,
        "projection": "orthographic",
        "ortho_scale": ortho_scale,
    }


def build_prior_bundle(
    name: str,
    *,
    width: int = 1200,
    height: int = 1000,
    samples: int = 96,
) -> dict[str, np.ndarray]:
    """Build a renderer-independent NPZ payload for one terrain prior."""

    if width < 64 or height < 64:
        raise ValueError("terrain-prior dimensions must be at least 64 pixels")
    if samples < 1:
        raise ValueError("terrain-prior samples must be positive")
    primitives = prior_primitives(name)
    metadata = {
        "schema_version": 2,
        "name": f"terrain-prior-{name}",
        "width": int(width),
        "height": int(height),
        "background": BACKGROUND,
        "transparent_background": True,
        "camera": _camera_payload(
            primitives,
            width=width,
            height=height,
            usable_fraction=0.52 if name == "chairs" else 0.58,
        ),
        "mesh_ids": [],
        "bevel": {"terrain": 0.018, "body": 0.010, "candidate": 0.010},
        "samples": int(samples),
        "ambient_strength": 0.22,
        "direct_light_scale": 1.0,
        # Keep the finite catcher boundary well outside every auto-fitted camera.
        "shadow_catcher": {"size": 10.0, "z": -0.008},
    }
    empty_transform = np.empty((0, 4, 4), dtype=np.float32)
    return {
        "boxes_position": np.asarray([item.position for item in primitives], dtype=np.float32),
        "boxes_size": np.asarray([item.half_size for item in primitives], dtype=np.float32),
        "boxes_rotation": np.asarray([item.rotation for item in primitives], dtype=np.float32),
        "boxes_rgba": np.asarray([item.rgba for item in primitives], dtype=np.float32),
        "boxes_kind": np.asarray([item.kind for item in primitives]),
        "boxes_pitched": np.asarray([item.pitched for item in primitives], dtype=bool),
        "instance_mesh": np.empty(0, dtype=np.int32),
        "instance_transform": empty_transform,
        "instance_rgba": np.empty((0, 4), dtype=np.float32),
        "instance_history": np.empty(0, dtype=bool),
        "instance_cell": np.empty(0, dtype=np.int16),
        "instance_frame": np.empty(0, dtype=np.int32),
        "tendon_endpoints": np.empty((0, 2, 3), dtype=np.float32),
        "tendon_radius": np.empty(0, dtype=np.float32),
        "tendon_rgba": np.empty((0, 4), dtype=np.float32),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_bundle(bundle: Path, output: Path, blender: Path) -> dict[str, object]:
    scene_script = Path(__file__).with_name("blender_scene.py")
    renderer_report = bundle.with_suffix(".renderer.json")
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
        str(output),
        "--report",
        str(renderer_report),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not output.is_file() or not renderer_report.is_file():
        details = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"Blender terrain-prior render failed ({completed.returncode}):\n{details[-8000:]}")
    payload = json.loads(renderer_report.read_text())
    payload["raw_sha256"] = _sha256(output)
    return payload


def _clean_transparent_film(path: Path, threshold: int = 15) -> None:
    """Discard imperceptible catcher alpha and isolated border pixels."""

    with Image.open(path) as source:
        image = source.convert("RGBA")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8).copy()
    alpha[alpha <= threshold] = 0
    retained = alpha > 0
    height, width = alpha.shape
    border = {
        *((0, column) for column in range(width)),
        *((height - 1, column) for column in range(width)),
        *((row, 0) for row in range(height)),
        *((row, width - 1) for row in range(height)),
    }
    for row, column in border:
        neighborhood = retained[max(0, row - 1) : min(height, row + 2), max(0, column - 1) : min(width, column + 2)]
        if retained[row, column] and np.count_nonzero(neighborhood) == 1:
            alpha[row, column] = 0
    image.putalpha(Image.fromarray(alpha, mode="L"))
    atomic_write(path, lambda destination: image.save(destination, format="PNG", optimize=True))


def _content_bounds(path: Path) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        alpha = image.convert("RGBA").getchannel("A")
        bounds = alpha.getbbox()
    if bounds is None:
        raise ValueError(f"terrain-prior render is fully transparent: {path}")
    return bounds


def _audit_content_margin(
    bounds: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> None:
    required = max(2, round(0.005 * min(width, height)))
    margins = (bounds[0], bounds[1], width - bounds[2], height - bounds[3])
    if min(margins) < required:
        raise ValueError(
            f"terrain-prior content margin {margins} is below the required {required}px; "
            "geometry or its contact shadow is clipped"
        )


def _compose_strip(images: list[Path], output: Path) -> None:
    opened = [Image.open(path).convert("RGBA") for path in images]
    try:
        gap = max(8, opened[0].width // 40)
        canvas = Image.new(
            "RGBA",
            (sum(image.width for image in opened) + gap * (len(opened) - 1), max(image.height for image in opened)),
            (0, 0, 0, 0),
        )
        x = 0
        for image in opened:
            canvas.alpha_composite(image, (x, 0))
            x += image.width + gap
        atomic_write(output, lambda destination: canvas.save(destination, format="PNG", optimize=True))
    finally:
        for image in opened:
            image.close()


def render_terrain_priors(
    *,
    output_directory: str | Path,
    blender_executable: str | Path,
    width: int = 1200,
    height: int = 1000,
    scale: float = 1.0,
    samples: int = 96,
) -> dict[str, object]:
    """Render all terrain priors and a text-free horizontal composition."""

    if not 0.1 <= scale <= 1.0:
        raise ValueError(f"scale must lie in [0.1, 1.0], got {scale}")
    executable = Path(blender_executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Blender executable not found: {executable}")
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    render_width = max(64, round(width * scale))
    render_height = max(64, round(height * scale))
    render_samples = max(16, round(samples * scale))
    outputs: dict[str, object] = {}
    final_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="terra-terrain-priors-") as temporary_directory:
        temporary = Path(temporary_directory)
        for name in PRIOR_NAMES:
            bundle = temporary / f"{name}.npz"
            arrays = build_prior_bundle(
                name,
                width=render_width,
                height=render_height,
                samples=render_samples,
            )
            np.savez_compressed(bundle, **arrays)
            metadata = json.loads(str(arrays["metadata"]))
            temporary_render = temporary / f"{name}.png"
            final_render = destination / f"{name}.png"
            report = destination / f"{name}.renderer.json"
            renderer = _render_bundle(bundle, temporary_render, executable)
            atomic_write(final_render, lambda path, source=temporary_render: shutil.copyfile(source, path))
            _clean_transparent_film(final_render)
            bounds = _content_bounds(final_render)
            _audit_content_margin(bounds, width=render_width, height=render_height)
            renderer["output"] = str(final_render)
            renderer["sha256"] = _sha256(final_render)
            renderer["alpha_cleanup_threshold"] = 15
            atomic_write(
                report,
                lambda path, payload=renderer: path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"),
            )
            primitives = prior_primitives(name)
            outputs[name] = {
                "image": str(final_render),
                "sha256": _sha256(final_render),
                "content_bounds": bounds,
                "primitive_count": len(primitives),
                "roles": sorted({primitive.role for primitive in primitives}),
                "camera": metadata["camera"],
                "renderer": renderer,
            }
            final_paths.append(final_render)
    strip = destination / "terrain-priors-strip.png"
    _compose_strip(final_paths, strip)
    provenance = {
        "schema_version": 1,
        "description": "Matched transparent Blender renders of TERRA terrain priors",
        "dimensions": [render_width, render_height],
        "scale": scale,
        "samples": render_samples,
        "view": {
            "azimuth_degrees": 43.0,
            "elevation_degrees": -27.0,
            "projection": "orthographic",
            "per_family_autoframe": True,
        },
        "outputs": outputs,
        "strip": {"path": str(strip), "sha256": _sha256(strip)},
        "geometry_contract": {
            "ramps": {
                "slope_range_degrees": RAMP_SLOPE_RANGE,
                "width_prior_m": RAMP_WIDTH_PRIOR,
                "buried_depth_m": RAMP_BURIED_DEPTH,
            },
            "staircases": {
                "minimum_levels": STAIR_MIN_LEVELS,
                "width_prior_m": STAIR_WIDTH_PRIOR,
            },
            "chairs": {
                "seat_height_range_m": [SEAT_MIN_HEIGHT, SEAT_MAX_HEIGHT],
                "seat_half_extents_m": SEAT_SIZE_PRIOR,
                "representation": "solid seat-support box",
            },
        },
    }
    provenance_path = destination / "provenance.json"
    atomic_write(
        provenance_path,
        lambda path: path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n"),
    )
    return {**provenance, "provenance": str(provenance_path)}


__all__ = [
    "PRIOR_NAMES",
    "Primitive",
    "box_primitives",
    "build_prior_bundle",
    "chair_primitives",
    "prior_primitives",
    "ramp_primitives",
    "render_terrain_priors",
    "staircase_primitives",
]
