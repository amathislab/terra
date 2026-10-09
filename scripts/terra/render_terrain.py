"""Render reconstruction JSON files as PNG images with MuJoCo."""

import argparse
import itertools
import json
import os
from pathlib import Path

# Select the headless backend before MuJoCo is imported.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image

from loco_mujoco.core.terrain import TerrainSpec
from terra.visualization.scene import apply_scene_style


def render_terrain(source: Path, output: Path, *, width=1280, height=960, azimuth=135.0, elevation=-30.0):
    payload = json.loads(source.read_text())
    value = payload.get("terrain", payload)
    if "boxes" not in value:
        raise ValueError(f"No terrain boxes field in {source}")
    terrain = TerrainSpec.from_dict(value)
    spec = mujoco.MjSpec.from_string(
        """<mujoco>
          <asset>
            <texture name="grid" type="2d" builtin="checker" width="256" height="256"
                     rgb1="0.90 0.91 0.92" rgb2="0.76 0.79 0.82"/>
            <material name="floor" texture="grid" texrepeat="4 4" texuniform="true"/>
          </asset>
          <worldbody>
            <light pos="0 0 8" dir="-0.3 -0.5 -1" directional="true"/>
            <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
          </worldbody>
        </mujoco>"""
    )
    for index, box in enumerate(terrain.boxes):
        spec.worldbody.add_geom(
            name=f"terrain_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box.pos,
            size=box.size,
            quat=box.quat,
            rgba=(0.36, 0.58, 0.78, 1.0),
        )
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    corners = (
        np.concatenate([np.asarray(box.pos) + (signs * box.size) @ box.rotation.T for box in terrain.boxes])
        if terrain.boxes
        else np.asarray([[-1, -1, 0], [1, 1, 0]])
    )
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    center = (lower + upper) / 2
    extent = max(1.0, float(np.linalg.norm(upper - lower)))
    model = apply_scene_style(spec).compile()
    model.stat.center[:] = center
    model.stat.extent = extent
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = center
    camera.distance = 1.5 * extent / min(1.0, width / height)
    camera.azimuth = azimuth
    camera.elevation = elevation
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        Image.fromarray(renderer.render()).save(output)
    return len(terrain.boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="reconstruction JSON file or cohort output directory")
    parser.add_argument("--output-dir", type=Path, help="save images here; defaults to beside each JSON file")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--azimuth", type=float, default=135.0, help="camera angle around the terrain, in degrees")
    parser.add_argument(
        "--elevation", type=float, default=-30.0, help="camera elevation in degrees; negative looks down"
    )
    args = parser.parse_args()
    source = args.source.expanduser()
    paths = sorted(path for path in source.glob("*.json") if path.name != "run.json") if source.is_dir() else [source]
    if not paths:
        parser.error(f"No reconstruction JSON files in {source}")
    for path in paths:
        output = (args.output_dir.expanduser() if args.output_dir else path.parent) / f"{path.stem}.png"
        boxes = render_terrain(
            path, output, width=args.width, height=args.height, azimuth=args.azimuth, elevation=args.elevation
        )
        print(f"{output}: {boxes} terrain boxes")


if __name__ == "__main__":
    main()
