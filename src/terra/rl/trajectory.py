"""Trajectory reset sampling shared by PPO training and evaluation."""

import jax
import jax.numpy as jnp
import numpy as np


def _install_frame_zero_reset_mixture() -> None:
    from loco_mujoco.trajectory.handler import TrajectoryHandler

    original = TrajectoryHandler.reset_state
    if getattr(original, "__terra_frame_zero_mixture__", False):
        return

    def reset_state(self, env, model, data, carry, backend):
        selected_trajectory = getattr(carry, "selected_traj_idx", None)
        data, carry = original(self, env, model, data, carry, backend)
        probability = float(getattr(self, "frame_zero_reset_probability", 0.0))
        random_frame_reset = bool(self.random_start and self.start_from_random_step)
        if probability <= 0.0 or not random_frame_reset:
            return data, carry
        if not 0.0 <= probability <= 1.0:
            raise ValueError("frame_zero_reset_probability must be in [0, 1]")
        if backend == jnp:
            key, mixture_key = jax.random.split(carry.key)
            use_frame_zero = jax.random.bernoulli(mixture_key, probability)
            if selected_trajectory is not None:
                use_frame_zero = jnp.logical_and(use_frame_zero, selected_trajectory < 0)
            step = jnp.where(use_frame_zero, jnp.asarray(0, dtype=jnp.int32), carry.traj_state.subtraj_step_no)
            traj_state = carry.traj_state.replace(subtraj_step_no=step, subtraj_step_no_init=step)
            return data, carry.replace(key=key, traj_state=traj_state)
        eligible = selected_trajectory is None or int(selected_trajectory) < 0
        use_frame_zero = eligible and np.random.random() < probability
        if use_frame_zero:
            traj_state = carry.traj_state.replace(subtraj_step_no=0, subtraj_step_no_init=0)
            carry = carry.replace(traj_state=traj_state)
        return data, carry

    reset_state.__terra_frame_zero_mixture__ = True
    TrajectoryHandler.reset_state = reset_state


def install_trajectory_stability() -> None:
    _install_frame_zero_reset_mixture()
