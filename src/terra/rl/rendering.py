"""Reference-motion rendering for terrain-conditioned environments."""

from __future__ import annotations

from copy import deepcopy
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from musclemimic.core.goals.trajectory import GoalTrajMimicv2
from terra.rl.observations import CompactTerraGoalMixin, terra_goal_parameters
from terra.rl.tracking import TerraFullBodyTrackingGoal


class _TerraReferenceVisualMixin:
    """Keep the ghost reference aligned with TERRA's paired terrain."""

    _geom_names: tuple[str, ...] = ()
    _bound_model_id: int | None = None

    def _init_from_mj(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        current_obs_size: int,
    ) -> None:
        super()._init_from_mj(env, model, data, current_obs_size)
        self._geom_names: tuple[str, ...] = ()
        self._bound_model_id: int | None = None
        self._geom_names = (
            ()
            if self._geom_ids is None
            else tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) for geom_id in self._geom_ids)
        )

    def _bind_visual_geometries(self, model: MjModel) -> None:
        if self._bound_model_id == id(model):
            return

        geom_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._geom_names],
            dtype=np.int32,
        )
        if np.any(geom_ids < 0):
            missing = [name for name, geom_id in zip(self._geom_names, geom_ids, strict=True) if geom_id < 0]
            raise RuntimeError(f"Reference model is missing visual geometries: {missing}")
        self._geom_ids = geom_ids
        self._geom_bodyid = np.asarray(model.geom_bodyid[geom_ids])
        self._geom_type = np.asarray(model.geom_type[geom_ids]).reshape(-1, 1)
        self._geom_size = np.asarray(model.geom_size[geom_ids])
        self._geom_dataid = np.asarray(model.geom_dataid[geom_ids]).reshape(-1, 1)
        self._geom_group = np.asarray(model.geom_group[geom_ids]).reshape(-1, 1)
        self._bound_model_id = id(model)

    def set_visuals(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> Any:
        if not self.visualize_goal or not self._enable_enhanced_vis:
            return carry

        cpu_model = env._model
        self._bind_visual_geometries(cpu_model)
        visual_slots = np.asarray(self.visual_geoms_idx, dtype=np.intp).reshape(-1)
        reference = env.th.get_current_traj_data(carry, backend)
        qpos = reference.qpos

        if not env.preserve_trajectory_root_xy:
            initial = env.th.get_init_traj_data(carry, backend)
            if backend == jnp:
                qpos = qpos.at[:2].add(-initial.qpos[:2])
            else:
                qpos = np.array(qpos, copy=True)
                qpos[:2] -= initial.qpos[:2]

        if backend == jnp:
            slots = jnp.asarray(visual_slots)
            data = data.replace(qpos=qpos, qvel=reference.qvel)
            data = mujoco.mjx.kinematics(env.sys, data)
            geoms = carry.user_scene.geoms
            geoms = geoms.replace(
                pos=geoms.pos.at[slots].set(data.geom_xpos[self._geom_ids]),
                mat=geoms.mat.at[slots].set(data.geom_xmat[self._geom_ids].reshape(-1, 9)),
                type=geoms.type.at[slots].set(self._geom_type),
                size=geoms.size.at[slots].set(self._geom_size),
                rgba=geoms.rgba.at[slots].set(self._geom_rgba),
                dataid=geoms.dataid.at[slots].set(self._geom_dataid),
            )
            return carry.replace(user_scene=carry.user_scene.replace(geoms=geoms))

        reference_data = deepcopy(data)
        reference_data.qpos = qpos
        reference_data.qvel = reference.qvel
        mujoco.mj_kinematics(model, reference_data)
        geoms = carry.user_scene.geoms
        geoms.pos[visual_slots] = reference_data.geom_xpos[self._geom_ids]
        geoms.mat[visual_slots] = reference_data.geom_xmat[self._geom_ids].reshape(-1, 9)
        geoms.type[visual_slots] = self._geom_type
        geoms.size[visual_slots] = self._geom_size
        geoms.rgba[visual_slots] = self._geom_rgba
        geoms.dataid[visual_slots] = self._geom_dataid
        return carry.replace(user_scene=carry.user_scene)


class TerraGoalVisual(CompactTerraGoalMixin, _TerraReferenceVisualMixin, GoalTrajMimicv2):
    """Render the compact legacy TERRA goal on its paired terrain."""

    def __init__(self, info_props: dict, rel_body_names: list[str] | None = None, **parameters: Any):
        self._configure_future_reference(parameters)
        super().__init__(
            info_props,
            rel_body_names=rel_body_names,
            **terra_goal_parameters(parameters),
        )


class TerraFullBodyTrackingGoalVisual(
    _TerraReferenceVisualMixin,
    TerraFullBodyTrackingGoal,
    GoalTrajMimicv2,
):
    """Render the egocentric full-body tracking goal on its paired terrain."""


__all__ = ["TerraFullBodyTrackingGoalVisual", "TerraGoalVisual"]
