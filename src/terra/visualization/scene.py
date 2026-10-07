"""Gray scenes with steel-blue terrain and MuJoCo muscle colors."""

import mujoco
import numpy as np

BACKGROUND = (0.88, 0.89, 0.90)
FLOOR_LIGHT = (0.72, 0.74, 0.76)
FLOOR_DARK = (0.67, 0.69, 0.71)
STEEL_BLUE = (0.38, 0.49, 0.58)
REFERENCE_RGBA = (*STEEL_BLUE, 0.35)


def apply_scene_style(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Set display assets and lights without changing geometry or dynamics."""
    for texture in list(spec.textures):
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(texture)
        elif texture.name in {"texplane", "grid"}:
            texture.rgb1 = FLOOR_LIGHT
            texture.rgb2 = FLOOR_DARK
    spec.add_texture(
        name="terra_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=256,
        height=1536,
        rgb1=BACKGROUND,
        rgb2=BACKGROUND,
    )
    spec.visual.rgba.haze = (*BACKGROUND, 0.0)
    spec.visual.headlight.diffuse = (0.4, 0.4, 0.4)
    spec.visual.headlight.ambient = (0.3, 0.3, 0.3)
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.quality.numslices = 32
    spec.visual.quality.numstacks = 16
    spec.visual.map.shadowscale = 1.0
    floor_materials = {geom.material for geom in spec.geoms if geom.name == "floor"}
    for material in spec.materials:
        if material.name in floor_materials:
            material.reflectance = 0.0
            material.specular = 0.0
            material.rgba = (1.0, 1.0, 1.0, 1.0)
    for geom in spec.geoms:
        if geom.name == "floor":
            geom.rgba = (1.0, 1.0, 1.0, 1.0)
        elif geom.name.startswith("terrain_"):
            geom.rgba = (*STEEL_BLUE, 1.0)
    lights = list(spec.lights)
    key = lights[0] if lights else spec.worldbody.add_light(name="terra_key")
    for light in lights[1:]:
        light.castshadow = False
    key.type = mujoco.mjtLightType.mjLIGHT_SPOT
    key.pos = (-3.0, -4.0, 6.0)
    key.dir = (2.5, 4.4, -4.5)
    key.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    key.cutoff = 40.0
    key.diffuse = (0.8, 0.8, 0.8)
    key.specular = (0.1, 0.1, 0.1)
    key.castshadow = True
    return spec


def update_muscle_colors(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Use the model's MuJoCo actuator palette for muscle activation."""
    muscles = (
        (model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON)
        & (model.actuator_actadr >= 0)
    )
    tendon_ids = model.actuator_trnid[muscles, 0]
    activation = np.clip(data.act[model.actuator_actadr[muscles]], 0.0, 1.0)[:, None]
    inactive = model.vis.rgba.actuator[:3]
    active = model.vis.rgba.actuatorpositive[:3]
    model.tendon_rgba[tendon_ids, :3] = inactive + activation * (active - inactive)
