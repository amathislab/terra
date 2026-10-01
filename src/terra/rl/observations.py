"""Compact reference-motion observations used by TERRA policies."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from musclemimic.core.goals.trajectory import GoalTrajMimic


def terra_goal_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Force the observation contract shared by training and rendering.

    ``n_step_lookahead=1`` means the current reference frame only. The parent
    flags are disabled because :class:`CompactTerraGoalMixin` constructs the
    smaller observation explicitly.
    """

    parameters["n_step_lookahead"] = 1
    parameters["n_step_stride"] = 1
    parameters["enable_motion_phase"] = False
    parameters["use_concise_lookahead"] = False
    parameters["enable_mimic_site_rpos_observations"] = False
    parameters["enable_global_root_tracking_observations"] = False
    return parameters


class CompactTerraGoalMixin:
    """Compact current and sparse-future goal shared by actor, critic, and renderer.

    The goal portion is ordered as target qpos without global root XY, target
    qvel, target-minus-simulated root XY, and target-minus-simulated
    pelvis-relative mimic-site positions. Current qpos/qvel are supplied by the
    physical observation container. Future reference cues contain only root
    height deltas and root-relative ankle positions. No phase or continuation
    flag is exposed.
    """

    def _configure_future_reference(self, parameters: dict[str, Any]) -> None:
        self._enable_future_reference_observations = bool(parameters.pop("enable_future_reference_observations", False))
        self._future_reference_stride = int(parameters.pop("future_reference_stride", 10))
        self._future_reference_horizon = int(parameters.pop("future_reference_horizon", 100))
        if not self._enable_future_reference_observations:
            self._future_reference_offsets = ()
            return
        if self._future_reference_stride <= 0:
            raise ValueError("future_reference_stride must be positive")
        if self._future_reference_horizon < self._future_reference_stride:
            raise ValueError("future_reference_horizon must be at least future_reference_stride")
        if self._future_reference_horizon % self._future_reference_stride != 0:
            raise ValueError("future_reference_horizon must be divisible by future_reference_stride")
        self._future_reference_offsets = tuple(
            range(
                self._future_reference_stride,
                self._future_reference_horizon + 1,
                self._future_reference_stride,
            )
        )

    def _init_from_mj(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        current_obs_size: int,
    ) -> None:
        super()._init_from_mj(env, model, data, current_obs_size)
        n_relative_sites = max(0, len(self._rel_site_ids) - 1)
        if self._future_reference_offsets:
            site_names = list(self._info_props["sites_for_mimic"])
            ankle_names = ("left_ankle_mimic", "right_ankle_mimic")
            missing_ankles = [name for name in ankle_names if name not in site_names]
            if missing_ankles:
                raise ValueError(
                    "future reference observation requires ankle mimic sites: " + ", ".join(missing_ankles)
                )
            ankle_model_ids = self._rel_site_ids[
                np.asarray([site_names.index(name) for name in ankle_names], dtype=int)
            ]
            # Keep model ids here. The trajectory mapper is finalized later by
            # ``init_from_traj``, so conversion must happen when observations are
            # built rather than against its provisional site order.
            self._future_ankle_model_ids = ankle_model_ids
        else:
            self._future_ankle_model_ids = np.asarray([], dtype=int)
        future_dim = len(self._future_reference_offsets) * 7
        self._dim = len(self._qpos_ind) + len(self._qvel_ind) + 2 + 3 * n_relative_sites + future_dim
        self._size_additional_observation = 0
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.arange(data.userdata.size)
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)

    def get_obs_and_update_state(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        reference = env.th.get_current_traj_data(carry, backend)
        rel_site_ids = self._rel_site_ids
        rel_body_ids = self._site_bodyid[rel_site_ids]

        trajectory_site_indices = None
        if self._site_mapper.requires_mapping:
            trajectory_site_indices = self._site_mapper.model_ids_to_traj_indices(rel_site_ids)

        current_site_rpos, _, _ = calculate_relative_site_quantities(
            data,
            rel_site_ids,
            rel_body_ids,
            self._body_rootid,
            backend,
        )
        target_site_rpos, _, _ = calculate_relative_site_quantities(
            reference,
            rel_site_ids,
            rel_body_ids,
            self._body_rootid,
            backend,
            trajectory_site_indices=trajectory_site_indices,
        )

        root_xy_indices = self._root_qpos_full_ind[:2]
        if backend == jnp:
            root_xy_indices = backend.asarray(root_xy_indices)
        target_root_xy = reference.qpos[root_xy_indices]
        if not getattr(env, "preserve_trajectory_root_xy", False):
            initial_reference = env.th.get_init_traj_data(carry, backend)
            target_root_xy = target_root_xy - initial_reference.qpos[root_xy_indices]

        future_reference_goal = self._future_reference_observation(
            env,
            reference,
            carry,
            backend,
        )

        if self.visualize_goal:
            carry = self.set_visuals(env, model, data, carry, backend)

        goal = backend.concatenate(
            (
                reference.qpos[self._qpos_ind],
                reference.qvel[self._qvel_ind],
                target_root_xy - data.qpos[root_xy_indices],
                backend.ravel(target_site_rpos - current_site_rpos),
                future_reference_goal,
            )
        )
        return goal, carry

    def _future_reference_observation(
        self,
        env: Any,
        reference: Any,
        carry: Any,
        backend: ModuleType,
    ) -> np.ndarray | jnp.ndarray:
        """Return +stride..+horizon height and root-relative ankle cues."""
        if not self._future_reference_offsets:
            return backend.zeros((0,), dtype=reference.qpos.dtype)

        root_xyz_indices = self._root_qpos_full_ind[:3]
        ankle_trajectory_ids = (
            self._site_mapper.model_ids_to_traj_indices(self._future_ankle_model_ids)
            if self._site_mapper.requires_mapping
            else self._future_ankle_model_ids
        )
        if backend == jnp:
            root_xyz_indices = backend.asarray(root_xyz_indices)
            ankle_trajectory_ids = backend.asarray(ankle_trajectory_ids)
        current_reference_root_z = reference.qpos[root_xyz_indices[2]]
        trajectory_length = env.th.len_trajectory(carry.traj_state.traj_no)
        future_reference_components = []
        for offset in self._future_reference_offsets:
            future_step = carry.traj_state.subtraj_step_no + offset
            if backend == jnp:
                future_step = backend.clip(future_step, 0, trajectory_length - 1)
            else:
                future_step = max(0, min(future_step, trajectory_length - 1))
            future_reference = env.th.get_traj_data_at(
                carry.traj_state.traj_no,
                future_step,
                carry,
                backend,
            )
            future_root_xyz = future_reference.qpos[root_xyz_indices]
            future_ankles_root_relative = future_reference.site_xpos[ankle_trajectory_ids] - future_root_xyz
            future_reference_components.extend(
                (
                    backend.atleast_1d(future_root_xyz[2] - current_reference_root_z),
                    backend.ravel(future_ankles_root_relative),
                )
            )
        future_reference_goal = backend.concatenate(future_reference_components)
        return future_reference_goal


class TerraGoal(CompactTerraGoalMixin, GoalTrajMimic):
    """Compact imitation goal with sparse future height and ankle cues."""

    def __init__(self, info_props: dict, rel_body_names: list[str] | None = None, **parameters: Any):
        self._configure_future_reference(parameters)
        super().__init__(
            info_props,
            rel_body_names=rel_body_names,
            **terra_goal_parameters(parameters),
        )


__all__ = ["CompactTerraGoalMixin", "TerraGoal", "terra_goal_parameters"]
