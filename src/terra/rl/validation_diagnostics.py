"""Frame-zero diagnostics for deterministic validation renders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def _float(value: Any) -> float:
    return float(np.asarray(value).item())


def _canonical_reference_qpos(env: Any, reference: Any, carry: Any) -> np.ndarray:
    """Match the root-XY convention used by ``set_sim_state_from_traj_data``."""
    qpos = np.asarray(reference.qpos).copy()
    if getattr(env, "preserve_trajectory_root_xy", False):
        return qpos

    initial_reference = env.th.get_init_traj_data(carry, np)
    free_joint_xy = np.asarray(env.free_jnt_qpos_id, dtype=int)[:, :2].reshape(-1)
    root_xy = free_joint_xy[:2]
    qpos[free_joint_xy] -= np.asarray(initial_reference.qpos)[root_xy]
    return qpos


def _root_orientation_error(handler: Any, data: Any, reference: Any) -> float | None:
    quaternion_ids = getattr(handler, "_root_qpos_ids_quat", None)
    if quaternion_ids is None:
        return None
    current = np.asarray(data.qpos)[quaternion_ids]
    target = np.asarray(reference.qpos)[quaternion_ids]
    current_norm = np.linalg.norm(current)
    target_norm = np.linalg.norm(target)
    if current_norm <= 0.0 or target_norm <= 0.0:
        return float("inf")
    dot = abs(float(np.dot(current / current_norm, target / target_norm)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _core_threshold(handler: Any, carry: Any) -> float | None:
    threshold = getattr(handler, "core_upper_body_mean_site_deviation_threshold", None)
    if threshold is None:
        return None
    if not getattr(handler, "core_upper_body_uses_global_threshold", False):
        return float(threshold)

    global_final = float(handler.mean_site_deviation_threshold)
    global_initial = float(handler.curriculum_initial_global_threshold)
    progress = np.clip(
        (_float(carry.termination_threshold) - global_final)
        / max(global_initial - global_final, 1e-8),
        0.0,
        1.0,
    )
    core_final = float(handler.core_upper_body_mean_site_deviation_threshold)
    core_initial = float(handler.core_upper_body_curriculum_initial_threshold)
    return float(core_final + progress * (core_initial - core_final))


def _terrain_height(env: Any, x: float, y: float, carry: Any) -> float | None:
    sampler = getattr(getattr(env, "_terrain", None), "sample_heights_at_points", None)
    if sampler is None:
        return None
    try:
        height = sampler(
            np.asarray([x]),
            np.asarray([y]),
            env._model,
            carry,
            np,
        )
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return float(np.asarray(height).reshape(-1)[0])


def _site_terrain_gaps(env: Any, carry: Any) -> dict[str, float]:
    gaps = {}
    for name in (
        "left_ankle_mimic",
        "left_toes_mimic",
        "right_ankle_mimic",
        "right_toes_mimic",
    ):
        site_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            continue
        position = np.asarray(env._data.site_xpos[site_id])
        height = _terrain_height(env, float(position[0]), float(position[1]), carry)
        if height is not None:
            gaps[name] = float(position[2] - height)
    return gaps


def audit_validation_initialization(env: Any, motion_path: str | None = None) -> dict[str, Any]:
    """Measure whether a validation reset reproduces trajectory frame zero."""
    observation = np.asarray(env._obs)
    data = env._data
    carry = env._additional_carry
    trajectory_state = carry.traj_state
    trajectory_index = int(trajectory_state.traj_no)
    trajectory_frame = int(trajectory_state.subtraj_step_no)
    reference = env.th.get_current_traj_data(carry, np)
    canonical_qpos = _canonical_reference_qpos(env, reference, carry)
    qpos_error = np.abs(np.asarray(data.qpos) - canonical_qpos)
    qvel_error = np.abs(np.asarray(data.qvel) - np.asarray(reference.qvel))

    terminal_handler = env._terminal_state_handler
    site_deviations = terminal_handler._site_deviations(env, data, carry, np)
    if site_deviations is None:
        site_deviations_array = np.asarray([], dtype=float)
    else:
        site_deviations_array = np.asarray(site_deviations, dtype=float)

    core_indices = np.asarray(getattr(terminal_handler, "_core_upper_body_indices", ()), dtype=int)
    core_mean = None
    if site_deviations_array.size and core_indices.size:
        core_mean = float(np.mean(site_deviations_array[core_indices]))
    global_mean = float(np.mean(site_deviations_array)) if site_deviations_array.size else None
    global_max = float(np.max(site_deviations_array)) if site_deviations_array.size else None
    orientation_error = _root_orientation_error(terminal_handler, data, reference)
    global_threshold = _float(carry.termination_threshold)
    core_threshold = _core_threshold(terminal_handler, carry)
    orientation_threshold = getattr(terminal_handler, "root_orientation_threshold", None)
    would_terminate = bool(
        (global_mean is not None and (not np.isfinite(global_mean) or global_mean > global_threshold))
        or (core_mean is not None and core_threshold is not None and core_mean > core_threshold)
        or (
            orientation_error is not None
            and orientation_threshold is not None
            and orientation_error > float(orientation_threshold)
        )
    )

    root_xyz = np.asarray(data.qpos)[:3]
    root_terrain_height = _terrain_height(env, float(root_xyz[0]), float(root_xyz[1]), carry)
    paired_motion = getattr(env, "paired_motion_path", None)

    return {
        "motion_path": motion_path,
        "materialized_motion": str(Path(paired_motion)) if paired_motion is not None else None,
        "trajectory_index": trajectory_index,
        "trajectory_frame": trajectory_frame,
        "trajectory_length": int(env.th.len_trajectory(trajectory_index)),
        "control_dt": float(env.dt),
        "observation_dimension": int(observation.size),
        "observation_all_finite": bool(np.isfinite(observation).all()),
        "preserve_trajectory_root_xy": bool(getattr(env, "preserve_trajectory_root_xy", False)),
        "initial_state_handler": type(env._init_state_handler).__name__,
        "domain_randomizer": type(env._domain_randomizer).__name__,
        "terminal_state_handler": type(terminal_handler).__name__,
        "terrain": type(env._terrain).__name__,
        "qpos_max_abs_error": float(np.max(qpos_error)),
        "qpos_mean_abs_error": float(np.mean(qpos_error)),
        "qvel_max_abs_error": float(np.max(qvel_error)),
        "qvel_mean_abs_error": float(np.mean(qvel_error)),
        "global_site_mean_error": global_mean,
        "global_site_max_error": global_max,
        "core_site_mean_error": core_mean,
        "root_orientation_error_radians": orientation_error,
        "global_site_threshold": global_threshold,
        "core_site_threshold": core_threshold,
        "root_orientation_threshold_radians": (
            None if orientation_threshold is None else float(orientation_threshold)
        ),
        "would_terminate_at_reset": would_terminate,
        "root_terrain_height": root_terrain_height,
        "root_height_above_terrain": (
            None if root_terrain_height is None else float(root_xyz[2] - root_terrain_height)
        ),
        "mimic_site_terrain_gaps": _site_terrain_gaps(env, carry),
    }


__all__ = ["audit_validation_initialization"]
