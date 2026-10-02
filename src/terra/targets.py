"""Build mimic-site orientation targets and joint-coupler constraints."""

from __future__ import annotations

import joblib
import mujoco
import numpy as np
from omegaconf import DictConfig

from terra._musclemimic import SMPLH_BONE_ORDER_NAMES


def build_orientation_targets(
    env,
    robot_conf: DictConfig,
    fitted_shape_path: str,
    smpl_rotations: np.ndarray,
) -> dict:
    """Build world-space orientation targets for robot mimic sites.

    Args:
        env: Environment containing mimic-site names and the MuJoCo model.
        robot_conf: Robot configuration containing site-to-SMPL joint matches.
        fitted_shape_path: Optimized shape cache containing site alignments.
        smpl_rotations: World rotations with shape ``(T, 52, 3, 3)`` in
            SMPL-H bone order.

    Returns:
        Site IDs, rotation matrices, and ordered site names.

    Raises:
        ValueError: If cached alignments and environment sites are inconsistent.
    """
    _, _, _, smpl2robot_rot_mat, *_ = joblib.load(fitted_shape_path)
    all_sites = list(env.sites_for_mimic)
    if smpl2robot_rot_mat.shape[0] != len(all_sites):
        raise ValueError(
            f"shape_optimized.pkl holds {smpl2robot_rot_mat.shape[0]} alignments for "
            f"{len(all_sites)} mimic sites. The cache is stale - delete it and re-run the "
            "`smpl` retargeting path to regenerate."
        )

    matches = robot_conf.site_joint_matches
    site_ids, targets, names = [], [], []
    for b, site in enumerate(all_sites):  # `smpl2robot_rot_mat` is indexed by `sites_for_mimic`
        j = SMPLH_BONE_ORDER_NAMES.index(matches[site].smpl_joint)
        sid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise ValueError(f"Mimic site '{site}' not found in the compiled model.")
        site_ids.append(sid)
        targets.append(np.einsum("tij,jk->tik", smpl_rotations[:, j], smpl2robot_rot_mat[b]))
        names.append(site)

    return {
        "site_ids": np.asarray(site_ids, dtype=int),
        "targets": np.stack(targets, axis=1),  # (T, K, 3, 3)
        "names": names,
    }
