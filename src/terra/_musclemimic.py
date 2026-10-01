"""Validated compatibility boundary for the required MuscleMimic integration.

The pinned integration uses the same distribution name as other MuscleMimic packages.
TERRA therefore validates the capabilities it uses instead of treating distribution
metadata as proof of compatibility. Keep dependency imports in this module so a wrong
installation fails once with an actionable message.
"""

from __future__ import annotations

from importlib import import_module

_INSTALL_HINT = (
    "TERRA requires the amathislab/musclemimic_terra_release package, including its "
    "stable musclemimic.retargeting integration API. Install terra-retargeting "
    "with its pinned dependencies; a different 'musclemimic' distribution is "
    "not compatible."
)
_REQUIRED_RETARGETING_EXPORTS = (
    "OPTIMIZED_SHAPE_FILE_NAME",
    "SEAT_GEOM_PREFIX",
    "SMPLH_BONE_ORDER_NAMES",
    "BoxSpec",
    "Mujoco",
    "TerrainSpec",
    "Trajectory",
    "TrajectoryCacheType",
    "TrajectoryData",
    "TrajectoryInfo",
    "TrajectoryModel",
    "extend_motion",
    "fit_gmr_motion",
    "fit_smpl_motion",
    "fit_smpl_shape",
    "joint_couplers",
    "load_robot_conf_file",
    "materialize_trajectory",
    "max_penetration_with_geoms",
    "retarget_c3d_to_trajectory",
    "retargeting_cache_dir",
    "scene_geom_ids",
)

try:
    _retargeting = import_module("musclemimic.retargeting")
except ImportError as exc:
    raise ImportError(f"{_INSTALL_HINT} Original import error: {exc}") from exc

_missing = tuple(name for name in _REQUIRED_RETARGETING_EXPORTS if not hasattr(_retargeting, name))
if _missing:
    raise ImportError(f"{_INSTALL_HINT} Missing integration export(s): {', '.join(_missing)}")

try:
    _torso_frames = import_module("musclemimic.utils.torso_frame")
    torso_frame = _torso_frames.torso_frame
except (AttributeError, ImportError) as exc:
    raise ImportError(f"{_INSTALL_HINT} Missing torso-frame integration: {exc}") from exc

OPTIMIZED_SHAPE_FILE_NAME = _retargeting.OPTIMIZED_SHAPE_FILE_NAME
SEAT_GEOM_PREFIX = _retargeting.SEAT_GEOM_PREFIX
SMPLH_BONE_ORDER_NAMES = _retargeting.SMPLH_BONE_ORDER_NAMES
BoxSpec = _retargeting.BoxSpec
Mujoco = _retargeting.Mujoco
TerrainSpec = _retargeting.TerrainSpec
Trajectory = _retargeting.Trajectory
TrajectoryCacheType = _retargeting.TrajectoryCacheType
TrajectoryData = _retargeting.TrajectoryData
TrajectoryInfo = _retargeting.TrajectoryInfo
TrajectoryModel = _retargeting.TrajectoryModel
extend_motion = _retargeting.extend_motion
fit_gmr_motion = _retargeting.fit_gmr_motion
fit_smpl_motion = _retargeting.fit_smpl_motion
fit_smpl_shape = _retargeting.fit_smpl_shape
joint_couplers = _retargeting.joint_couplers
load_robot_conf_file = _retargeting.load_robot_conf_file
materialize_trajectory = _retargeting.materialize_trajectory
max_penetration_with_geoms = _retargeting.max_penetration_with_geoms
retarget_c3d_to_trajectory = _retargeting.retarget_c3d_to_trajectory
retargeting_cache_dir = _retargeting.retargeting_cache_dir
scene_geom_ids = _retargeting.scene_geom_ids

__all__ = [*_REQUIRED_RETARGETING_EXPORTS, "torso_frame"]
