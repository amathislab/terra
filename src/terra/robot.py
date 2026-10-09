"""Configure the OmniRetarget retargeter for a MuscleMimic robot model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import SimpleNamespace

import mujoco
import numpy as np

from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS, require_omniretarget
from terra.constants import AXIAL_ROTATION_JOINTS, MTP_JOINTS, TRUNK_JOINT_PREFIXES
from terra.defaults import (
    DEFAULT_AXIAL_SMOOTH_WEIGHT,
    DEFAULT_TRUNK_Q_DIAG,
    DEFAULT_TRUNK_SMOOTH_WEIGHT,
)


def build_task_constants(
    robot_dof: int,
    robot_height: float,
    robot_xml_path: str,
    joints_mapping: dict[str, str],
    foot_links: dict[str, str],
    object_name: str = "ground",
) -> SimpleNamespace:
    """Build the task constants required by the OmniRetarget retargeter.

    Args:
        robot_dof: Number of robot coordinates after the floating root.
        robot_height: Robot stature in meters.
        robot_xml_path: Robot MuJoCo XML path.
        joints_mapping: Mapping from SMPL joint names to robot body names.
        foot_links: Mapping from foot labels to robot body names.
        object_name: Name of the interaction-mesh scene object.

    Returns:
        Namespace containing the OmniRetarget task-constant fields.
    """
    require_omniretarget()
    return SimpleNamespace(
        ROBOT_URDF_FILE=str(robot_xml_path).replace(".xml", ".urdf"),
        ROBOT_DOF=robot_dof,
        ROBOT_HEIGHT=robot_height,
        OBJECT_NAME=object_name,
        OBJECT_URDF_FILE=None,
        OBJECT_MESH_FILE=None,
        DEMO_JOINTS=list(SMPLH_DEMO_JOINTS),
        JOINTS_MAPPING=dict(joints_mapping),
        FOOT_STICKING_LINKS=list(foot_links.keys()),
        MANUAL_LB={},
        MANUAL_UB={},
        MANUAL_COST={},
        NOMINAL_TRACKING_INDICES=[],
    )


def trunk_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """Find generalized-position indices for trunk and lumbar joints.

    Args:
        model: MuJoCo robot model.

    Returns:
        Integer array of matching generalized-position indices.
    """
    idx = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if name.startswith(TRUNK_JOINT_PREFIXES):
            idx.append(int(model.jnt_qposadr[i]))
    return np.asarray(idx, dtype=int)


def axial_rotation_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """Find generalized-position indices for axial-rotation joints.

    Args:
        model: MuJoCo robot model.

    Returns:
        Integer array of matching generalized-position indices.
    """
    return _qpos_indices_of(model, AXIAL_ROTATION_JOINTS)


def mtp_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """Find generalized-position indices for metatarsophalangeal joints.

    Args:
        model: MuJoCo robot model.

    Returns:
        Integer array of matching generalized-position indices.
    """
    return _qpos_indices_of(model, MTP_JOINTS)


def _qpos_indices_of(model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    """Find generalized-position indices for named joints.

    Args:
        model: MuJoCo robot model.
        names: Joint names to select.

    Returns:
        Matching indices in model order. Missing names are omitted.
    """
    idx = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if name in names:
            idx.append(int(model.jnt_qposadr[i]))
    return np.asarray(idx, dtype=int)


def swap_model_and_fix_limits(
    retargeter,
    model: mujoco.MjModel,
    robot_dof: int,
    foot_links: dict[str, str],
    logger: logging.Logger,
    base_smooth_weight: float = 0.2,
    trunk_smooth_weight: float = DEFAULT_TRUNK_SMOOTH_WEIGHT,
    trunk_q_diag: float = DEFAULT_TRUNK_Q_DIAG,
    axial_smooth_weight: float | None = DEFAULT_AXIAL_SMOOTH_WEIGHT,
    mtp_smooth_weight: float | None = None,
) -> None:
    """Bind a retargeter to an environment model and rebuild derived arrays.

    Args:
        retargeter: Retargeter to update in place.
        model: Environment MuJoCo model.
        robot_dof: Number of robot coordinates after the floating root.
        foot_links: Mapping from foot labels to robot body names.
        logger: Logger for the model-swap summary.
        base_smooth_weight: Default temporal damping weight.
        trunk_smooth_weight: Temporal damping weight for trunk joints.
        trunk_q_diag: Absolute-angle penalty for trunk joints.
        axial_smooth_weight: Optional damping weight for axial-rotation joints.
        mtp_smooth_weight: Optional damping weight for toe joints.

    Raises:
        ValueError: If rebuilt joint-limit arrays do not match ``model.nq``.
    """
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.nq = model.nq

    retargeter.q_a_indices = np.arange(7 + retargeter.q_a_init_idx, 7 + robot_dof)
    retargeter.nq_a = len(retargeter.q_a_indices)

    # Rebuild limits, excluding the free joint by type (not by name).
    large = 1e6
    non_free = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    lower = np.concatenate([-large * np.ones(7), model.jnt_range[non_free, 0]])
    upper = np.concatenate([large * np.ones(7), model.jnt_range[non_free, 1]])
    if len(lower) != model.nq:
        raise ValueError(f"Limit array length {len(lower)} != nq {model.nq}; check joint layout.")

    retargeter.q_a_lb = lower[retargeter.q_a_indices]
    retargeter.q_a_ub = upper[retargeter.q_a_indices]

    # Per-joint regularisation. q_a index equals qpos index here, because `q_a_indices` is
    # arange(0, nq) for the default `q_a_init_idx` of -7.
    retargeter.Q_diag = np.zeros(retargeter.nq_a)
    smooth = np.full(retargeter.nq_a, float(base_smooth_weight))

    trunk = trunk_qpos_indices(model)
    if trunk.size:
        inside = trunk[trunk < retargeter.nq_a]
        retargeter.Q_diag[inside] = trunk_q_diag
        smooth[inside] = trunk_smooth_weight

    axial = axial_rotation_qpos_indices(model)
    if axial.size and axial_smooth_weight is not None:
        inside = axial[axial < retargeter.nq_a]
        smooth[inside] = axial_smooth_weight

    mtp = mtp_qpos_indices(model)
    if mtp.size and mtp_smooth_weight is not None:
        inside = mtp[mtp < retargeter.nq_a]
        smooth[inside] = mtp_smooth_weight

    retargeter.smooth_weight = smooth

    # Labels, not body names, so OmniRetarget's `"left" in key` test fires. See MYOFULLBODY_FOOT_LINKS.
    retargeter.foot_links = dict(foot_links)

    retargeter.has_dynamic_object = model.nq > 7 + robot_dof

    logger.info(f"Model swapped (nq={model.nq}, nq_a={retargeter.nq_a}); limits rebuilt, foot links relabelled")
