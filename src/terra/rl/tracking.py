"""Egocentric immediate-error and future-intent trajectory observations."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from numbers import Integral
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils.math import calc_site_velocities
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid
from musclemimic.core.goals.trajectory import GoalTrajMimic
from terra.rl.egocentric import (
    heading_rotation_matrix,
    root_velocity_in_heading_frame,
    rotation_error_in_frame,
    rotation_matrix_from_quaternion,
    world_to_heading,
)

DEFAULT_LOOKAHEAD_STEPS = (1, 20, 40, 60, 80)
ROOT_TRACKING_ERROR_DIM = 12
SITE_TRACKING_ERROR_DIM = 12
FUTURE_ROOT_INTENT_DIM = 12
FUTURE_SITE_INTENT_DIM = 3
FUTURE_METADATA_DIM = 2


def validate_lookahead_steps(steps: Sequence[int]) -> tuple[int, ...]:
    """Validate tracking horizons measured in 100 Hz control steps."""

    steps = tuple(steps)
    if any(isinstance(step, bool) or not isinstance(step, Integral) for step in steps):
        raise TypeError("lookahead_steps must contain integers")
    normalized = tuple(int(step) for step in steps)
    if not normalized or normalized[0] != 1:
        raise ValueError("lookahead_steps must start with 1")
    if any(current >= following for current, following in pairwise(normalized)):
        raise ValueError("lookahead_steps must be strictly increasing")
    return normalized


def tracking_goal_dimension(n_relative_sites: int, n_horizons: int) -> int:
    """Return the flattened immediate-target plus future-intent dimension."""

    if n_relative_sites < 0:
        raise ValueError("n_relative_sites must be non-negative")
    if n_horizons < 1:
        raise ValueError("n_horizons must be at least one")
    immediate = ROOT_TRACKING_ERROR_DIM + SITE_TRACKING_ERROR_DIM * n_relative_sites
    future = FUTURE_ROOT_INTENT_DIM + FUTURE_SITE_INTENT_DIM * n_relative_sites + FUTURE_METADATA_DIM
    return immediate + future * (n_horizons - 1)


def _site_kinematics(
    data: Any,
    site_ids: Any,
    site_body_ids: Any,
    body_root_ids: Any,
    backend: ModuleType,
    trajectory_site_indices: Any = None,
) -> tuple[Any, Any, Any]:
    data_indices = site_ids if trajectory_site_indices is None else trajectory_site_indices
    positions = data.site_xpos[data_indices]
    rotations = data.site_xmat[data_indices].reshape(-1, 3, 3)
    root_body_ids = body_root_ids[site_body_ids]
    velocities = calc_site_velocities(
        site_ids,
        data,
        site_body_ids,
        root_body_ids,
        backend,
        flg_local=False,
        trajectory_site_indices=trajectory_site_indices,
    )
    return positions, rotations, velocities


def _root_tracking_error(
    current_position: Any,
    current_rotation: Any,
    current_qvel: Any,
    target_position: Any,
    target_rotation: Any,
    target_qvel: Any,
    heading_rotation: Any,
    backend: ModuleType,
) -> Any:
    position_error = world_to_heading(target_position - current_position, heading_rotation, backend)
    rotation_error = rotation_error_in_frame(
        current_rotation,
        target_rotation,
        heading_rotation,
        backend,
    )
    current_velocity = root_velocity_in_heading_frame(
        current_qvel,
        current_rotation,
        heading_rotation,
        backend,
    )
    target_velocity = root_velocity_in_heading_frame(
        target_qvel,
        target_rotation,
        heading_rotation,
        backend,
    )
    return backend.concatenate([position_error, rotation_error, target_velocity - current_velocity])


def _site_tracking_error(
    current: tuple[Any, Any, Any],
    target: tuple[Any, Any, Any],
    heading_rotation: Any,
    backend: ModuleType,
) -> Any:
    current_positions, current_rotations, current_velocities = current
    target_positions, target_rotations, target_velocities = target

    current_relative_positions = current_positions[1:] - current_positions[0]
    target_relative_positions = target_positions[1:] - target_positions[0]
    position_errors = world_to_heading(
        target_relative_positions - current_relative_positions,
        heading_rotation,
        backend,
    )
    rotation_errors = rotation_error_in_frame(
        current_rotations[1:],
        target_rotations[1:],
        heading_rotation,
        backend,
    )
    velocity_errors_world = (
        target_velocities[1:] - target_velocities[0]
        - (current_velocities[1:] - current_velocities[0])
    )
    # MuJoCo spatial velocities are [angular, linear].
    velocity_errors = backend.concatenate(
        [
            world_to_heading(velocity_errors_world[:, :3], heading_rotation, backend),
            world_to_heading(velocity_errors_world[:, 3:], heading_rotation, backend),
        ],
        axis=-1,
    )
    return backend.concatenate(
        [
            backend.ravel(position_errors),
            backend.ravel(rotation_errors),
            backend.ravel(velocity_errors),
        ]
    )


def _future_intent(
    current_root_position: Any,
    current_root_rotation: Any,
    target_root_position: Any,
    target_root_rotation: Any,
    target_root_qvel: Any,
    target_site_positions: Any,
    heading_rotation: Any,
    time_offset_seconds: Any,
    is_valid: Any,
    backend: ModuleType,
) -> Any:
    root_displacement = world_to_heading(
        target_root_position - current_root_position,
        heading_rotation,
        backend,
    )
    root_rotation = rotation_error_in_frame(
        current_root_rotation,
        target_root_rotation,
        heading_rotation,
        backend,
    )
    root_velocity = root_velocity_in_heading_frame(
        target_root_qvel,
        target_root_rotation,
        heading_rotation,
        backend,
    )
    relative_site_positions = world_to_heading(
        target_site_positions[1:] - target_site_positions[0],
        heading_rotation,
        backend,
    )
    metadata = backend.stack([time_offset_seconds, is_valid])
    return backend.concatenate(
        [root_displacement, root_rotation, root_velocity, backend.ravel(relative_site_positions), metadata]
    )


class TerraFullBodyTrackingGoal(GoalTrajMimic):
    """Full +1-frame tracking error followed by compact future targets.

    The target root is left in its paired-terrain coordinates when TERRA is
    configured to preserve reference XY. Otherwise it is shifted by the
    episode's initial reference XY, using the flat-ground reference origin.
    """

    def __init__(
        self,
        info_props: dict,
        rel_body_names: list[str] | None = None,
        lookahead_steps: Sequence[int] = DEFAULT_LOOKAHEAD_STEPS,
        include_support_intent: bool = False,
        **parameters: Any,
    ) -> None:
        if include_support_intent:
            raise ValueError("TERRA's paired-terrain tracking goal does not use flat-floor support intent")
        self.lookahead_steps = validate_lookahead_steps(lookahead_steps)
        self._root_qpos_indices = None
        self._root_qvel_indices = None

        # Allow this goal to be selected as an override of TerraGoal
        # without leaking its future-reference parameters into the parent.
        for compact_key in (
            "enable_future_reference_observations",
            "future_reference_stride",
            "future_reference_horizon",
        ):
            parameters.pop(compact_key, None)
        parameters.update(
            n_step_lookahead=1,
            n_step_stride=1,
            enable_motion_phase=False,
            use_concise_lookahead=False,
            enable_mimic_site_rpos_observations=False,
            enable_global_root_tracking_observations=False,
        )
        super().__init__(info_props, rel_body_names=rel_body_names, **parameters)

    def _init_from_mj(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        current_obs_size: int,
    ) -> None:
        super()._init_from_mj(env, model, data, current_obs_size)
        root_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            env.root_free_joint_xml_name,
        )
        self._root_qpos_indices = np.asarray(mj_jntid2qposid(root_joint_id, model))
        # GoalTrajMimic exposes these as a Python list. Newer JAX releases
        # require advanced indices to be an array rather than a list.
        self._root_qvel_indices = np.asarray(self._root_qvel_ind)
        n_relative_sites = len(self._info_props["sites_for_mimic"]) - 1
        self._dim = tracking_goal_dimension(n_relative_sites, len(self.lookahead_steps))
        self._size_additional_observation = 0
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)

    def _trajectory_site_indices(self, backend: ModuleType) -> Any:
        if not self._site_mapper.requires_mapping:
            return None
        indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
        return backend.asarray(indices) if backend is jnp else indices

    @staticmethod
    def _bounded_target_step(
        current_step: Any,
        step_offset: int,
        trajectory_length: Any,
        backend: ModuleType,
    ) -> tuple[Any, Any]:
        requested_step = current_step + step_offset
        is_valid = requested_step < trajectory_length
        if backend is jnp:
            return backend.clip(requested_step, 0, trajectory_length - 1), is_valid
        return max(0, min(requested_step, trajectory_length - 1)), is_valid

    def get_obs_and_update_state(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        observation = self._tracking_observation(env, data, carry, backend)
        if self.visualize_goal:
            carry = self.set_visuals(env, model, data, carry, backend)
        return observation, carry

    def _tracking_observation(
        self,
        env: Any,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> np.ndarray | jnp.ndarray:
        traj_state = carry.traj_state
        root_qpos_indices = self._root_qpos_indices
        root_qvel_indices = self._root_qvel_indices
        trajectory_site_indices = self._trajectory_site_indices(backend)

        current_root_qpos = data.qpos[root_qpos_indices]
        current_root_position = current_root_qpos[:3]
        current_root_rotation = rotation_matrix_from_quaternion(current_root_qpos[3:7], backend)
        heading_rotation = heading_rotation_matrix(current_root_rotation, backend)
        current_root_qvel = data.qvel[root_qvel_indices]
        dtype = backend.asarray(current_root_position).dtype

        site_ids = self._rel_site_ids
        site_body_ids = self._site_bodyid[site_ids]
        current_sites = _site_kinematics(
            data,
            site_ids,
            site_body_ids,
            self._body_rootid,
            backend,
        )

        if getattr(env, "preserve_trajectory_root_xy", False):
            position_offset = backend.zeros(3, dtype=dtype)
        else:
            initial_reference = env.th.get_traj_data_at(
                traj_state.traj_no,
                traj_state.subtraj_step_no_init,
                carry,
                backend,
            )
            initial_xy = backend.asarray(initial_reference.qpos[root_qpos_indices[:2]], dtype=dtype)
            position_offset = backend.concatenate([initial_xy, backend.zeros(1, dtype=dtype)])

        trajectory_length = env.th.len_trajectory(traj_state.traj_no)
        components = []
        for horizon_index, step_offset in enumerate(self.lookahead_steps):
            target_step, is_valid = self._bounded_target_step(
                traj_state.subtraj_step_no,
                step_offset,
                trajectory_length,
                backend,
            )
            target_data = env.th.get_traj_data_at(
                traj_state.traj_no,
                target_step,
                carry,
                backend,
            )
            target_root_qpos = target_data.qpos[root_qpos_indices]
            target_root_position = target_root_qpos[:3] - position_offset
            target_root_rotation = rotation_matrix_from_quaternion(target_root_qpos[3:7], backend)
            target_root_qvel = target_data.qvel[root_qvel_indices]

            if horizon_index == 0:
                target_sites = _site_kinematics(
                    target_data,
                    site_ids,
                    site_body_ids,
                    self._body_rootid,
                    backend,
                    trajectory_site_indices=trajectory_site_indices,
                )
                components.extend(
                    [
                        _root_tracking_error(
                            current_root_position,
                            current_root_rotation,
                            current_root_qvel,
                            target_root_position,
                            target_root_rotation,
                            target_root_qvel,
                            heading_rotation,
                            backend,
                        ),
                        _site_tracking_error(current_sites, target_sites, heading_rotation, backend),
                    ]
                )
                continue

            target_site_indices = site_ids if trajectory_site_indices is None else trajectory_site_indices
            target_site_positions = target_data.site_xpos[target_site_indices]
            components.append(
                _future_intent(
                    current_root_position,
                    current_root_rotation,
                    target_root_position,
                    target_root_rotation,
                    target_root_qvel,
                    target_site_positions,
                    heading_rotation,
                    backend.asarray(step_offset * env.dt, dtype=dtype),
                    backend.asarray(is_valid, dtype=dtype),
                    backend,
                )
            )

        return backend.concatenate(components)


__all__ = [
    "DEFAULT_LOOKAHEAD_STEPS",
    "TerraFullBodyTrackingGoal",
    "tracking_goal_dimension",
    "validate_lookahead_steps",
]
