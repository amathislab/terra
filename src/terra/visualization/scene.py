"""Scene asset changes for TERRA viewers."""

from pathlib import Path

import mujoco


def remove_scene_logo(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Remove the upstream logo texture before the environment is compiled."""
    for texture in list(spec.textures):
        if (
            texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX
            and Path(texture.file).name == "musclemimic_skybox.png"
        ):
            spec.delete(texture)
    return spec
