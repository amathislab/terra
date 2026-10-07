"""Muscle display colors follow activation without changing simulation state."""

import mujoco
import numpy as np
import pytest

from terra.visualization.scene import update_muscle_colors


@pytest.mark.parametrize(
    ("activation", "expected"),
    [(0.0, (0.2, 0.3, 0.95)), (0.5, (0.575, 0.3, 0.625)), (1.0, (0.95, 0.3, 0.3))],
)
def test_muscle_activation_colors_preserve_state_and_tendon_opacity(activation, expected):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody><body><joint name="joint"/><geom type="capsule" size="0.1 0.2"/></body></worldbody>
      <tendon><fixed name="muscle"><joint joint="joint" coef="1"/></fixed></tendon>
      <actuator><muscle tendon="muscle" lengthrange="0.1 1.0"/></actuator>
    </mujoco>
    """)
    model.tendon_rgba[0, 3] = 0.7
    data = mujoco.MjData(model)
    data.act[:] = activation
    state = data.act.copy()
    update_muscle_colors(model, data)
    np.testing.assert_allclose(model.tendon_rgba[0, :3], expected)
    np.testing.assert_allclose(model.tendon_rgba[0, 3], 0.7)
    np.testing.assert_array_equal(data.act, state)


def test_environment_styles_terrain_added_after_robot_initialization():
    from terra.rl.environment import MyoFullBody
    from terra.visualization.scene import STEEL_BLUE

    env = MyoFullBody(
        terrain_type="BoxTerrain",
        terrain_params={"boxes": [{"pos": [0, 0, 0.1], "size": [1, 1, 0.1]}]},
    )
    try:
        model = env._model
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_box_0")
        assert geom >= 0
        np.testing.assert_allclose(model.geom_rgba[geom], (*STEEL_BLUE, 1.0))
        assert model.light_castshadow.any()
    finally:
        env.stop()
