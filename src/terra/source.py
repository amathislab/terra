"""Convert AMASS motion data and terrain surfaces into solver inputs."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from terra._musclemimic import SMPLH_BONE_ORDER_NAMES
from terra.baselines.omniretarget import (
    SMPLH_DEMO_JOINTS,
    demo_joint_permutation,
    preprocess_motion_data,
    require_omniretarget,
)
from terra.constants import (
    MYOFULLBODY_SITE_CALIBRATION,
    NO_MAT_HEIGHT,
)

if TYPE_CHECKING:
    from terra.terrain.seats import SeatRest


#: Vertices used for seated support must be predominantly attached to the pelvis or
#: proximal hips. This is an anatomical membership rule, not an object-dataset prior.
SEATED_SURFACE_MIN_SKIN_WEIGHT = 0.80


#: Robust lower-envelope statistic over the posterior pelvis/hip surface. A decile
#: suppresses isolated mesh vertices without averaging into the body's interior.
SEATED_SURFACE_QUANTILE = 0.10


def measure_smpl_stature(parser, betas: torch.Tensor, up_axis: int = 1) -> float:
    """Measure SMPL-H stature from rest-pose vertices.

    Args:
        parser: Initialized SMPL-H parser.
        betas: Shape parameters with shape ``(1, 16)``.
        up_axis: Vertex coordinate used as the vertical axis.

    Returns:
        Rest-pose vertex extent in meters along ``up_axis``.
    """
    pose = torch.zeros(1, 156).float()
    trans = torch.zeros(1, 3).float()
    verts, _ = parser.get_joints_verts(pose, th_betas=betas.float(), th_trans=trans)
    v = verts.detach().cpu().numpy()[0]
    return float(v[:, up_axis].max() - v[:, up_axis].min())


def get_robot_height(fitted_shape_path: str, smpl_model_path: str) -> float:
    """Calculate robot stature from the optimized SMPL shape cache.

    Args:
        fitted_shape_path: Optimized SMPL shape cache path.
        smpl_model_path: SMPL model directory.

    Returns:
        Robot stature in meters.
    """
    from loco_mujoco.smpl import SMPLH_Parser

    shape_new, scale, *_ = joblib.load(fitted_shape_path)
    betas = shape_new.detach() if hasattr(shape_new, "detach") else torch.as_tensor(shape_new)
    scale_f = float(np.asarray(scale.detach() if hasattr(scale, "detach") else scale).ravel()[0])

    parser = SMPLH_Parser(model_path=smpl_model_path, gender="neutral")
    return measure_smpl_stature(parser, betas) * scale_f


def amass_to_world_transforms(
    motion_data: dict,
    smpl_model_path: str,
    betas: torch.Tensor,
    skip: int = 1,
    *,
    parser=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate SMPL-H world transforms for an AMASS motion.

    Args:
        motion_data: Loaded AMASS pose and translation arrays.
        smpl_model_path: SMPL model directory.
        betas: SMPL-H shape parameters.
        skip: Positive frame stride.
        parser: Optional initialized parser, reused when the caller also needs the
            neutral calibration pose.

    Returns:
        World joint positions with shape ``(T, 52, 3)`` in OmniRetarget order
        and rotations with shape ``(T, 52, 3, 3)`` in SMPL-H bone order.
    """
    from loco_mujoco.smpl import SMPLH_Parser

    pose_aa = torch.from_numpy(motion_data["pose_aa"][::skip]).float()
    pose_aa = torch.cat([pose_aa, torch.zeros((pose_aa.shape[0], 156 - pose_aa.shape[1]))], dim=-1)
    trans = torch.from_numpy(motion_data["trans"][::skip]).float()
    n = pose_aa.shape[0]

    parser = parser or SMPLH_Parser(model_path=smpl_model_path, gender="neutral")
    transforms = parser.get_joint_transformations(pose_aa.reshape(n, -1, 3), betas.repeat(n, 1), trans)
    transforms = transforms.detach().cpu().numpy()

    joints = transforms[..., :3, 3][:, demo_joint_permutation(), :]  # (T, 52, 3)
    return joints, transforms[..., :3, :3]  # (T, 52, 3, 3)


def amass_to_world_joints(
    motion_data: dict,
    smpl_model_path: str,
    betas: torch.Tensor,
    skip: int = 1,
) -> np.ndarray:
    """Evaluate SMPL-H world joint positions for an AMASS motion.

    Args:
        motion_data: Loaded AMASS pose and translation arrays.
        smpl_model_path: SMPL model directory.
        betas: SMPL-H shape parameters.
        skip: Positive frame stride.

    Returns:
        World joint positions with shape ``(T, 52, 3)`` in OmniRetarget order.
    """
    return amass_to_world_transforms(motion_data, smpl_model_path, betas, skip)[0]


def apply_site_calibration(
    joints: np.ndarray,
    smpl_rotations: np.ndarray,
    position_offsets: np.ndarray,
    rotation_offsets: np.ndarray,
    calibration_smpl_rotations: np.ndarray,
) -> np.ndarray:
    """Convert SMPL-H joint centres to calibrated MyoFullBody mimic-site positions.

    The legacy fitted-shape cache stores each positional residual in the world frame of its
    neutral calibration pose. Convert that residual to the calibrated site frame first,
    then rotate it with the site's world orientation at every motion frame. This exactly
    reproduces the calibrated neutral pose without fixing an anatomical offset in world.

    Args:
        joints: SMPL-H positions in ``SMPLH_DEMO_JOINTS`` order, shape ``(T, J, 3)``.
        smpl_rotations: SMPL-H world rotations in bone order, shape ``(T, B, 3, 3)``.
        position_offsets: Neutral-pose world residuals from ``shape_optimized.pkl``, shape
            ``(S, 3)``.
        rotation_offsets: SMPL-to-site rotation offsets, shape ``(S, 3, 3)``.
        calibration_smpl_rotations: SMPL-H world rotations in the neutral pose used to
            create the cache, shape ``(B, 3, 3)``.

    Returns:
        A copy of ``joints`` whose calibrated entries represent MyoFullBody mimic sites.

    Raises:
        ValueError: If any input is inconsistent with the calibration schema.
    """
    joints = np.asarray(joints, dtype=float)
    smpl_rotations = np.asarray(smpl_rotations, dtype=float)
    position_offsets = np.asarray(position_offsets, dtype=float)
    rotation_offsets = np.asarray(rotation_offsets, dtype=float)
    calibration_smpl_rotations = np.asarray(calibration_smpl_rotations, dtype=float)
    n_sites = len(MYOFULLBODY_SITE_CALIBRATION)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joints must have shape (T, J, 3), got {joints.shape}")
    if smpl_rotations.shape != (len(joints), len(SMPLH_BONE_ORDER_NAMES), 3, 3):
        raise ValueError(
            "smpl_rotations must have shape "
            f"({len(joints)}, {len(SMPLH_BONE_ORDER_NAMES)}, 3, 3), got {smpl_rotations.shape}"
        )
    if position_offsets.shape != (n_sites, 3):
        raise ValueError(f"position calibration must have shape ({n_sites}, 3), got {position_offsets.shape}")
    if rotation_offsets.shape != (n_sites, 3, 3):
        raise ValueError(f"rotation calibration must have shape ({n_sites}, 3, 3), got {rotation_offsets.shape}")
    if calibration_smpl_rotations.shape != (len(SMPLH_BONE_ORDER_NAMES), 3, 3):
        raise ValueError(
            "calibration SMPL-H rotations must have shape "
            f"({len(SMPLH_BONE_ORDER_NAMES)}, 3, 3), got "
            f"{calibration_smpl_rotations.shape}"
        )
    if not all(
        np.isfinite(values).all()
        for values in (
            joints,
            smpl_rotations,
            position_offsets,
            rotation_offsets,
            calibration_smpl_rotations,
        )
    ):
        raise ValueError("site calibration inputs must be finite")

    demo_index = {name: index for index, name in enumerate(SMPLH_DEMO_JOINTS)}
    bone_index = {name: index for index, name in enumerate(SMPLH_BONE_ORDER_NAMES)}
    calibrated = joints.copy()
    for site_index, (_site, smpl_joint, _body) in enumerate(MYOFULLBODY_SITE_CALIBRATION):
        if smpl_joint not in demo_index or smpl_joint not in bone_index:
            raise ValueError(f"calibration joint {smpl_joint!r} is absent from SMPL-H ordering")
        site_rotation = np.einsum(
            "tij,jk->tik",
            smpl_rotations[:, bone_index[smpl_joint]],
            rotation_offsets[site_index],
        )
        calibration_site_rotation = calibration_smpl_rotations[bone_index[smpl_joint]] @ rotation_offsets[site_index]
        local_offset = calibration_site_rotation.T @ position_offsets[site_index]
        world_offset = np.einsum("tij,j->ti", site_rotation, local_offset)
        calibrated[:, demo_index[smpl_joint]] -= world_offset
    return calibrated


def _landmark_normalization(joints: np.ndarray, scale: float) -> dict[str, object]:
    """Describe the exact motion-only transform used by ``preprocess_motion_data``."""

    values = np.asarray(joints, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(SMPLH_DEMO_JOINTS), 3):
        raise ValueError(f"joints must have shape (T, {len(SMPLH_DEMO_JOINTS)}, 3), got {values.shape}")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization scale must be finite and positive")
    toe_indices = [SMPLH_DEMO_JOINTS.index(name) for name in ("L_Toe", "R_Toe")]
    source_floor_z = float(np.min(values[:, toe_indices, 2]))
    if source_floor_z >= NO_MAT_HEIGHT:
        source_floor_z -= NO_MAT_HEIGHT
    return {
        "method": "minimum_fitted_toe_height_then_uniform_scale",
        "source_floor_z_m": source_floor_z,
        "uniform_scale": float(scale),
        "source_to_normalized_translation_m": [
            0.0,
            0.0,
            float(-source_floor_z * scale),
        ],
        "equation": "normalized_xyz = source_xyz * uniform_scale + source_to_normalized_translation_m",
    }


def normalized_motion_landmarks(
    motion_data: dict,
    smpl_model_path: str,
    fitted_shape_path: str,
    robot_height: float,
    *,
    use_fitted_shape: bool = True,
    calibrate_sites: bool | None = None,
    return_normalization: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    """Build the canonical source landmarks shared by terrain, retargeting and metrics.

    ``calibrate_sites=None`` follows the shape convention: robot-fitted SMPL-H motions use
    the matching site calibration, while subject-shaped motions remain anatomical SMPL-H
    joints because the robot-fitted local offsets are not valid for another body shape.
    """
    shape_cache = joblib.load(fitted_shape_path)
    if len(shape_cache) < 4:
        raise ValueError(f"invalid fitted-shape cache at {fitted_shape_path}")
    shape_new, shape_scale, position_offsets, rotation_offsets, *_ = shape_cache
    calibrate_sites = use_fitted_shape if calibrate_sites is None else bool(calibrate_sites)
    if calibrate_sites and not use_fitted_shape:
        raise ValueError("MyoFullBody site calibration requires the matching robot-fitted SMPL-H shape")

    from loco_mujoco.smpl import SMPLH_Parser

    parser = SMPLH_Parser(model_path=smpl_model_path, gender="neutral")
    if use_fitted_shape:
        betas = shape_new.detach() if hasattr(shape_new, "detach") else torch.as_tensor(shape_new)
        smpl_scale = 1.0
    else:
        betas = torch.as_tensor(motion_data["betas"][:16]).reshape(1, 16).float()
        human_height = measure_smpl_stature(parser, betas)
        smpl_scale = robot_height / human_height

    joints, rotations = amass_to_world_transforms(motion_data, smpl_model_path, betas, parser=parser)
    if use_fitted_shape:
        scale = float(
            np.asarray(shape_scale.detach() if hasattr(shape_scale, "detach") else shape_scale).reshape(-1)[0]
        )
        root = joints[:, :1].copy()
        joints = (joints - root) * scale + root
    if calibrate_sites:
        calibration_pose = torch.zeros((1, len(SMPLH_BONE_ORDER_NAMES), 3)).float()
        calibration_pose[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = torch.from_numpy(
            Rotation.from_euler("xyz", [np.pi / 2, 0.0, np.pi / 2]).as_rotvec()
        ).float()
        calibration_transforms = parser.get_joint_transformations(calibration_pose, betas, torch.zeros((1, 3)).float())
        calibration_smpl_rotations = calibration_transforms[0, :, :3, :3].detach().cpu().numpy()
        joints = apply_site_calibration(
            joints,
            rotations,
            position_offsets,
            rotation_offsets,
            calibration_smpl_rotations,
        )
    normalization = _landmark_normalization(joints, smpl_scale)
    joints = preprocess_motion_data(
        joints,
        SimpleNamespace(demo_joints=list(SMPLH_DEMO_JOINTS)),
        ["L_Toe", "R_Toe"],
        scale=smpl_scale,
        mat_height=NO_MAT_HEIGHT,
    )
    result = joints, rotations, float(motion_data["fps"])
    return (*result, normalization) if return_normalization else result


def posed_seat_support_heights(
    motion_data: dict,
    smpl_model_path: str,
    fitted_shape_path: str,
    solver_joints: np.ndarray,
    demo_joints: Sequence[str],
    rests: Sequence[SeatRest],
    *,
    calibrate_sites: bool = True,
    min_skin_weight: float = SEATED_SURFACE_MIN_SKIN_WEIGHT,
    quantile: float = SEATED_SURFACE_QUANTILE,
) -> np.ndarray:
    """Estimate seated support from the posed posterior pelvis surface.

    The fitted robot-shape SMPL-H mesh is evaluated only at three representative
    frames per rest. Vertices predominantly skinned to the pelvis or proximal hips
    and lying behind the pelvis-to-feet direction form the anatomical contact patch;
    its lower decile is the support surface. The result is aligned to the same
    calibrated solver landmarks used by terrain reconstruction.

    Args:
        motion_data: Loaded SMPL-H motion dictionary.
        smpl_model_path: SMPL-H model directory.
        fitted_shape_path: Robot-fitted SMPL-H shape cache.
        solver_joints: Normalized, calibrated source landmarks with shape ``(T,J,3)``.
        demo_joints: Names corresponding to ``solver_joints``.
        rests: Detected seated intervals, in the same frame clock as ``solver_joints``.
        calibrate_sites: Whether ``solver_joints`` use the fitted MyoFullBody site
            calibration rather than raw SMPL-H joint centres.
        min_skin_weight: Required combined pelvis/left-hip/right-hip skinning weight.
        quantile: Robust lower-envelope quantile of the posterior surface.

    Returns:
        One solver-space support height per rest.
    """
    if not rests:
        return np.empty(0, dtype=float)
    solver_joints = np.asarray(solver_joints, dtype=float)
    names = list(demo_joints)
    if solver_joints.ndim != 3 or solver_joints.shape[1:] != (len(names), 3):
        raise ValueError(f"solver_joints must have shape (T, len(demo_joints), 3), got {solver_joints.shape}")
    if "Pelvis" not in names:
        raise ValueError("demo_joints must contain Pelvis")
    if not np.isfinite(min_skin_weight) or not 0.0 < min_skin_weight <= 1.0:
        raise ValueError("min_skin_weight must lie in (0, 1]")
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 0.5:
        raise ValueError("quantile must lie in [0, 0.5]")

    sample_frames: list[int] = []
    frames_by_rest: list[list[int]] = []
    for rest in rests:
        if rest.start < 0 or rest.end > len(solver_joints) or rest.end <= rest.start:
            raise ValueError(f"invalid seat rest [{rest.start}, {rest.end})")
        frames = sorted(
            {
                int(rest.start),
                int((rest.start + rest.end - 1) // 2),
                int(rest.end - 1),
            }
        )
        frames_by_rest.append(frames)
        sample_frames.extend(frames)
    selected = np.asarray(sorted(set(sample_frames)), dtype=int)
    selected_index = {int(frame): index for index, frame in enumerate(selected)}

    shape_cache = joblib.load(fitted_shape_path)
    if len(shape_cache) < 4:
        raise ValueError(f"invalid fitted-shape cache at {fitted_shape_path}")
    shape_new, shape_scale, position_offsets, rotation_offsets, *_ = shape_cache
    betas = shape_new.detach() if hasattr(shape_new, "detach") else torch.as_tensor(shape_new)
    betas = betas.reshape(1, -1).float()
    scale = float(np.asarray(shape_scale.detach() if hasattr(shape_scale, "detach") else shape_scale).reshape(-1)[0])

    from loco_mujoco.smpl import SMPLH_Parser

    parser = SMPLH_Parser(model_path=smpl_model_path, gender="neutral")
    pose_aa = torch.as_tensor(np.asarray(motion_data["pose_aa"])[selected]).float()
    if pose_aa.shape[1] > 156:
        raise ValueError(f"pose_aa has {pose_aa.shape[1]} values per frame; expected at most 156")
    if pose_aa.shape[1] < 156:
        pose_aa = torch.cat(
            (pose_aa, torch.zeros((len(pose_aa), 156 - pose_aa.shape[1]))),
            dim=1,
        )
    trans = torch.as_tensor(np.asarray(motion_data["trans"])[selected]).float()
    repeated_betas = betas.repeat(len(selected), 1)
    with torch.no_grad():
        transforms = parser.get_joint_transformations(
            pose_aa.reshape(len(selected), -1, 3),
            repeated_betas,
            trans,
        )
        vertices, _ = parser.get_joints_verts(
            pose_aa,
            th_betas=repeated_betas,
            th_trans=trans,
        )
    transforms = transforms.detach().cpu().numpy()
    vertices = vertices.detach().cpu().numpy()
    raw_joints = transforms[..., :3, 3][:, demo_joint_permutation(), :]
    rotations = transforms[..., :3, :3]
    root = raw_joints[:, :1].copy()
    raw_joints = (raw_joints - root) * scale + root
    vertices = (vertices - root) * scale + root

    reference_joints = raw_joints
    if calibrate_sites:
        calibration_pose = torch.zeros((1, len(SMPLH_BONE_ORDER_NAMES), 3)).float()
        calibration_pose[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = torch.from_numpy(
            Rotation.from_euler("xyz", [np.pi / 2, 0.0, np.pi / 2]).as_rotvec()
        ).float()
        with torch.no_grad():
            calibration = parser.get_joint_transformations(
                calibration_pose,
                betas,
                torch.zeros((1, 3)).float(),
            )
        reference_joints = apply_site_calibration(
            raw_joints,
            rotations,
            position_offsets,
            rotation_offsets,
            calibration[0, :, :3, :3].detach().cpu().numpy(),
        )

    weights = parser.lbs_weights.detach().cpu().numpy()
    anatomical = weights[:, :3].sum(axis=1) >= min_skin_weight
    if not np.any(anatomical):
        raise ValueError("no SMPL-H vertices satisfy the seated-support skinning rule")
    pelvis = names.index("Pelvis")
    heights = []
    for rest, frames in zip(rests, frames_by_rest, strict=True):
        direction = np.array([np.cos(rest.yaw), np.sin(rest.yaw)])
        samples = []
        for frame in frames:
            index = selected_index[frame]
            delta_xy = vertices[index, :, :2] - solver_joints[frame, pelvis, :2]
            posterior = delta_xy @ direction <= 0.0
            patch = anatomical & posterior
            if not np.any(patch):
                raise ValueError(f"seat rest [{rest.start}, {rest.end}) has no posterior support vertices")
            raw_surface = float(np.quantile(vertices[index, patch, 2], quantile))
            pelvis_to_surface = float(reference_joints[index, pelvis, 2] - raw_surface)
            samples.append(float(solver_joints[frame, pelvis, 2] - pelvis_to_surface))
        heights.append(float(np.median(samples)))
    return np.asarray(heights, dtype=float)


def motion_world_joints(
    motion_name: str,
    env_name: str = "MyoFullBody",
    use_fitted_shape: bool = True,
    motion_data: dict | None = None,
    calibrate_sites: bool | None = None,
    return_normalization: bool = False,
    *,
    smpl_model_path: str | os.PathLike[str] | None = None,
    fitted_shape_path: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, dict[str, object]]:
    """Load and normalize source joints for a retargeting solve.

    Args:
        motion_name: AMASS motion identifier.
        env_name: Environment associated with the fitted shape cache.
        use_fitted_shape: Whether to use the optimized robot-matched SMPL shape.
        motion_data: Preloaded AMASS motion. Source loading is kept at the dataset
            or API boundary so this low-level helper never consults hidden path defaults.
        calibrate_sites: Whether to convert matched SMPL-H joint centres to calibrated
            MyoFullBody mimic sites. ``None`` enables it exactly for robot-fitted shapes.
        smpl_model_path: Explicit SMPL-H model root, or ``TERRA_MODEL_ROOT`` when omitted.
        fitted_shape_path: Explicit optimized robot-shape artifact, or the direct
            cache below ``TERRA_ARTIFACT_ROOT`` when omitted.

    Returns:
        Solver-space joints in OmniRetarget order and their frame rate.
    """
    require_omniretarget()
    from terra.runtime import resolve_model_path, shape_cache_path

    base_env_name = env_name.replace("Mjx", "")
    if motion_data is None:
        raise ValueError(
            "motion_data is required; load the SMPL-H archive explicitly or use terra.api.retarget for a source path"
        )
    resolved_model_path = resolve_model_path(smpl_model_path)
    resolved_shape_path = (
        Path(fitted_shape_path).expanduser().resolve()
        if fitted_shape_path is not None
        else shape_cache_path(base_env_name, None)
    )
    data = dict(motion_data)

    normalized = normalized_motion_landmarks(
        data,
        str(resolved_model_path),
        str(resolved_shape_path),
        get_robot_height(str(resolved_shape_path), str(resolved_model_path)),
        use_fitted_shape=use_fitted_shape,
        calibrate_sites=calibrate_sites,
        return_normalization=return_normalization,
    )
    if return_normalization:
        joints, _rotations, fps, normalization = normalized
        return joints, fps, normalization
    joints, _rotations, fps = normalized
    return joints, fps


def flat_ground_scene(num_frames: int, ground_range: tuple[float, float] = (-10.0, 10.0), size: int = 10) -> dict:
    """Build an interaction-mesh scene for flat ground.

    Args:
        num_frames: Number of motion frames.
        ground_range: Inclusive grid extent in both horizontal axes, in meters.
        size: Number of samples per horizontal axis.

    Returns:
        Static scene dictionary consumed by OmniRetarget.
    """
    x = np.linspace(ground_range[0], ground_range[1], size)
    y = np.linspace(ground_range[0], ground_range[1], size)
    gx, gy = np.meshgrid(x, y)
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    return _scene_dict(pts, num_frames)


def _scene_dict(
    points: np.ndarray,
    num_frames: int,
) -> dict:
    """Pack static surface points into an OmniRetarget scene dictionary.

    Args:
        points: Static surface points with shape ``(N, 3)``.
        num_frames: Number of object-pose frames.

    Returns:
        Scene dictionary containing static object poses and surface points.
    """
    identity = np.tile(np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), (num_frames, 1))
    identity_src = np.tile(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), (num_frames, 1))
    points = np.asarray(points, dtype=float)
    return {
        "object_name": "ground",
        "object_poses": identity,
        "object_poses_augmented": identity.copy(),
        "object_poses_src": identity_src,
        "object_points_local_demo": points,
        "object_points_local": points,
    }


def terrain_scene(
    terrain,
    num_frames: int,
    spacing: float = 0.20,
    ground_range: tuple[float, float] = (-3.0, 3.0),
    ground_size: int = 8,
) -> dict:
    """Build an interaction-mesh scene from terrain surface samples.

    Args:
        terrain: Terrain specification to sample.
        num_frames: Number of motion frames.
        spacing: Approximate sample spacing on box top faces, in meters.
        ground_range: Inclusive floor-grid extent in both horizontal axes.
        ground_size: Number of floor-grid samples per horizontal axis.

    Returns:
        Static scene dictionary consumed by OmniRetarget.
    """
    pts = terrain.surface_points(spacing=spacing, ground_range=ground_range, ground_size=ground_size)
    return _scene_dict(pts, num_frames)
