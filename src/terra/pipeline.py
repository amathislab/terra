"""Run TERRA retargeting or the OmniRetarget comparison method.

The pipeline prepares SMPL-H landmarks, resolves terrain, configures solver terms,
runs sequential quadratic programming, and assembles the cached trajectory.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from musclemimic_models import get_xml_path
from omegaconf import DictConfig

from terra._musclemimic import Mujoco, Trajectory, TrajectoryCacheType, joint_couplers, materialize_trajectory
from terra.assembly import (
    NumericalEnvelope,
    SolveContext,
    SolveStage,
    apply_sole_offsets,
    apply_stance_landmark_clearance_cap,
    apply_swing_route,
    attach_foot_anchor,
    attach_leg_self_collision,
    attach_non_penetration,
    attach_orientation,
    attach_seat_contact_targets,
    attach_stance_height_targets,
    attach_swing_clearance,
    pad_for_warmup,
    source_contact_schedule,
    source_sticking_schedule,
)
from terra.baselines.omniretarget import (
    SMPLH_DEMO_JOINTS,
    extract_foot_sticking_sequence_velocity,
    require_omniretarget,
    transform_from_human_to_world,
)
from terra.constants import (
    ENV_TO_OMNIRETARGET_ROBOT,
    MYOFULLBODY_FOOT_LINKS,
    SITE_CALIBRATION_VERSION,
    SMPLH_TO_MYOFULLBODY,
)
from terra.defaults import (
    DEFAULT_MAX_ROOT_SOURCE_DEVIATION,
    DEFAULT_MAX_ROOT_STEP,
    DEFAULT_TERRAIN_PENETRATION_TOLERANCE,
)
from terra.postprocess import (
    align_to_ground,
    assemble_trajectory,
    landmark_error,
    repair_collisions,
    repair_materialized_tendon_transitions,
    resample_frame_values,
)
from terra.profiles import SolverConfig, active_qp_terms, all_active_qp_terms, resolve_solver_config
from terra.reconstruction import (
    ReconstructionRequest,
    add_posed_seat_support,
    implicit_calibration_evidence,
    reconstruct_terrain,
    smplh_terrain_fit_options,
)
from terra.retargeter import TerraRetargeter
from terra.robot import build_task_constants, swap_model_and_fix_limits
from terra.runtime import resolve_model_path, shape_cache_path
from terra.source import (
    flat_ground_scene,
    get_robot_height,
    motion_world_joints,
    normalized_motion_landmarks,
    terrain_scene,
)
from terra.terrain.metadata import TerrainMetadata


def _add_posed_seat_support(
    fit_options: dict,
    joints: np.ndarray,
    fps: float,
    motion_data: dict,
    smpl_model_path: str | os.PathLike,
    fitted_shape_path: str | os.PathLike,
    *,
    use_fitted_shape: bool,
    calibrate_sites: bool,
    ground_datum_correction_m: float = 0.0,
    logger: logging.Logger | None = None,
) -> dict:
    """Compatibility alias for the shared reconstruction input-preparation helper."""

    return add_posed_seat_support(
        fit_options,
        joints,
        fps,
        motion_data,
        smpl_model_path,
        fitted_shape_path,
        use_fitted_shape=use_fitted_shape,
        calibrate_sites=calibrate_sites,
        ground_datum_correction_m=ground_datum_correction_m,
        logger=logger,
    )


def terrain_for_motion(
    motion_name: str,
    env_name: str = "MyoFullBody",
    use_fitted_shape: bool = True,
    motion_data: dict | None = None,
    calibrate_sites: bool | None = None,
    *,
    smpl_model_path: str | os.PathLike | None = None,
    fitted_shape_path: str | os.PathLike | None = None,
    **fit_kwargs,
):
    """Fit and validate terrain for an AMASS motion.

    Args:
        motion_name: AMASS motion identifier.
        env_name: Environment associated with the fitted shape cache.
        use_fitted_shape: Whether to pose SMPL-H with the robot-fitted shape.
        motion_data: Preloaded AMASS motion. This helper does not consult an
            upstream AMASS path; use the dataset pipeline to load stored motions.
        calibrate_sites: Whether to use calibrated MyoFullBody mimic-site positions.
            ``None`` enables them exactly when ``use_fitted_shape`` is true.
        **fit_kwargs: Keyword arguments for :func:`fit_terrain_from_motion`.

    Returns:
        The fitted terrain, fit report, and validation report.
    """
    if motion_data is None:
        raise ValueError(
            "motion_data is required; load the SMPL-H archive explicitly or use the dataset terrain pipeline"
        )
    data = dict(motion_data)
    resolved_model_path = resolve_model_path(smpl_model_path)
    resolved_shape_path = (
        Path(fitted_shape_path).expanduser().resolve()
        if fitted_shape_path is not None
        else shape_cache_path(env_name, None)
    )
    joints, fps, normalization = motion_world_joints(
        motion_name,
        env_name,
        use_fitted_shape,
        data,
        calibrate_sites=calibrate_sites,
        return_normalization=True,
        smpl_model_path=resolved_model_path,
        fitted_shape_path=resolved_shape_path,
    )
    names = list(SMPLH_DEMO_JOINTS)
    calibrated = use_fitted_shape if calibrate_sites is None else bool(calibrate_sites)
    fit_kwargs = smplh_terrain_fit_options(
        fit_kwargs, use_fitted_shape=use_fitted_shape, calibrate_sites=calibrated
    )
    fit_kwargs = _add_posed_seat_support(
        fit_kwargs,
        joints,
        fps,
        data,
        resolved_model_path,
        resolved_shape_path,
        use_fitted_shape=use_fitted_shape,
        calibrate_sites=calibrated,
        ground_datum_correction_m=-float(normalization["source_to_normalized_translation_m"][2]),
    )
    site_calibration_version = (
        SITE_CALIBRATION_VERSION
        if (use_fitted_shape if calibrate_sites is None else calibrate_sites)
        else "uncalibrated_smplh_joints"
    )
    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=joints,
            joint_names=tuple(names),
            fps=fps,
            fit_options=fit_kwargs,
            calibration=implicit_calibration_evidence(fit_kwargs),
            report_fields={"site_calibration_version": site_calibration_version},
        )
    )
    return result.terrain, result.report, result.validation


def _resolve_terrain(
    terrain,
    human_joints: np.ndarray,
    fps: float,
    config: SolverConfig,
    logger: logging.Logger,
    *,
    motion_data: dict,
    smpl_model_path: str | os.PathLike,
    fitted_shape_path: str | os.PathLike,
    normalization: dict[str, object],
):
    """Resolve a terrain input to a terrain specification.

    Args:
        terrain: Terrain specification, dictionary, JSON path, ``"auto"``, or
            ``None``. A value of ``None`` uses the configuration fallback.
        human_joints: Preprocessed source joints with shape ``(T, J, 3)``.
        fps: Source frame rate in frames per second.
        config: Resolved solver configuration.
        logger: Logger for fit and validation results.
        motion_data: Loaded SMPL-H motion used to evaluate the posed body surface.
        smpl_model_path: SMPL-H model root.
        fitted_shape_path: Robot-fitted SMPL-H shape cache.
        normalization: Motion-derived source-to-landmark normalization metadata.

    Returns:
        Resolved terrain, or ``None`` for flat ground.

    """
    from terra._musclemimic import TerrainSpec

    if terrain is None:
        terrain = config.terrain
    if terrain is None:
        return None

    if isinstance(terrain, TerrainSpec):
        return terrain
    if isinstance(terrain, dict):
        return TerrainSpec.from_dict(terrain)
    if terrain != "auto":
        logger.info(f"Terrain: loading {terrain}")
        return TerrainSpec.load(terrain)

    calibrated = config.use_fitted_shape if config.calibrate_sites is None else config.calibrate_sites
    fit_cfg = _add_posed_seat_support(
        smplh_terrain_fit_options(
            config.terrain_fit, use_fitted_shape=config.use_fitted_shape, calibrate_sites=calibrated
        ),
        human_joints,
        fps,
        motion_data,
        smpl_model_path,
        fitted_shape_path,
        use_fitted_shape=config.use_fitted_shape,
        calibrate_sites=calibrated,
        ground_datum_correction_m=-float(normalization["source_to_normalized_translation_m"][2]),
        logger=logger,
    )
    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=human_joints,
            joint_names=tuple(SMPLH_DEMO_JOINTS),
            fps=fps,
            fit_options=fit_cfg,
            calibration=implicit_calibration_evidence(fit_cfg),
        )
    )
    spec = result.terrain
    report = result.report
    check = result.validation
    assert spec is not None
    for w in report["warnings"]:
        logger.warning(f"Terrain fit: {w}")
    logger.info(
        f"Terrain fit: {len(spec)} box(es) from {report['n_stance_events']} stance events; "
        f"internal support diagnostic {check['raised_contact_error_max'] * 1000:.1f} mm, "
        f"penetration diagnostic {check['max_penetration'] * 1000:.1f} mm"
    )
    return spec


def _source_joints(
    motion_data: dict,
    smpl_model_path: str,
    fitted_shape_path: str,
    robot_height: float,
    config: SolverConfig,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, object], SolverConfig]:
    """Convert an AMASS motion to normalized solver-space SMPL-H data.

    Args:
        motion_data: Loaded AMASS motion data.
        smpl_model_path: SMPL model directory.
        fitted_shape_path: Optimized SMPL shape cache path.
        robot_height: Robot stature in meters.
        config: Resolved solver configuration.
        logger: Logger for source-motion information.

    Returns:
        Solver-space joints, world rotations, and source frame rate. Joints use
        OmniRetarget order; rotations use SMPL-H bone order.
    """
    use_fitted_shape = config.use_fitted_shape
    calibrate_sites = use_fitted_shape if config.calibrate_sites is None else config.calibrate_sites
    human_joints, smpl_rotations, fps, normalization = normalized_motion_landmarks(
        motion_data,
        smpl_model_path,
        fitted_shape_path,
        robot_height,
        use_fitted_shape=use_fitted_shape,
        calibrate_sites=calibrate_sites,
        return_normalization=True,
    )
    calibration_version = SITE_CALIBRATION_VERSION if calibrate_sites else "uncalibrated_smplh_joints"
    config = config.for_source_calibration(calibrate_sites=calibrate_sites, version=calibration_version)
    logger.info(f"Source motion: {human_joints.shape[0]} frames @ {fps:.1f} Hz")
    logger.info(
        "Source landmarks: %s",
        calibration_version if calibrate_sites else "uncalibrated SMPL-H joints",
    )
    return human_joints, smpl_rotations, fps, normalization, config


def _build_environment(env_name: str, robot_conf: DictConfig, terrain, logger: logging.Logger):
    """Instantiate an environment containing the resolved terrain.

    Args:
        env_name: Registered environment name.
        robot_conf: Robot configuration containing environment parameters.
        terrain: Resolved terrain, or ``None`` for flat ground.
        logger: Logger for environment details.

    Returns:
        Instantiated MuscleMimic environment.
    """
    # MuscleMimic envs are only in `Mujoco.registered_envs` once their module is imported,
    # so resolve them directly. Mirrors `fit_gmr_motion`.
    th_params = {"random_start": False, "fixed_start_conf": (0, 0)}
    env_params = dict(robot_conf.env_params)
    if terrain is not None and not terrain.is_flat:
        env_params["terrain_type"] = "BoxTerrain"
        env_params["terrain_params"] = terrain.to_env_params()
        logger.info(f"Terrain: {len(terrain)} box(es) built into the environment model")

    if env_name == "MyoFullBody":
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        return MyoFullBody(**env_params, th_params=th_params)
    if env_name == "MjxMyoFullBody":
        from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody

        return MjxMyoFullBody(**env_params, th_params=th_params)
    return Mujoco.registered_envs[env_name](**env_params, th_params=th_params)


def _build_retargeter(
    ctx: SolveContext, robot_dof: int, robot_height: float, base_env_name: str, foot_mode: str
) -> TerraRetargeter:
    """Construct and configure a retargeter for the environment model.

    Args:
        ctx: State and configuration for the current solve.
        robot_dof: Number of robot coordinates after the floating root.
        robot_height: Robot stature in meters.
        base_env_name: Environment name without an ``Mjx`` prefix.
        foot_mode: Selected foot-constraint mode.

    Returns:
        Retargeter bound to the environment's MuJoCo model.

    Raises:
        ValueError: If the solver backend is invalid.
    """
    config = ctx.config
    constants = build_task_constants(
        robot_dof=robot_dof,
        robot_height=robot_height,
        robot_xml_path=str(get_xml_path(ENV_TO_OMNIRETARGET_ROBOT[base_env_name])),
        joints_mapping=ctx.joints_mapping,
        foot_links=MYOFULLBODY_FOOT_LINKS,
        object_name=ctx.scene["object_name"],
    )
    retargeter = TerraRetargeter(
        task_constants=constants,
        object_urdf_path=None,
        q_a_init_idx=config.q_a_init_idx,
        activate_joint_limits=config.activate_joint_limits,
        activate_obj_non_penetration=config.activate_obj_non_penetration,
        activate_foot_sticking=foot_mode == "omniretarget",
        penetration_tolerance=(
            config.penetration_tolerance
            if config.penetration_tolerance is not None
            else DEFAULT_TERRAIN_PENETRATION_TOLERANCE
            if ctx.on_terrain
            else 1e-3
        ),
        foot_sticking_tolerance=config.foot_sticking_tolerance,
        step_size=config.step_size,
        visualize=False,
        debug=config.debug,
    )
    solver_backend = config.solver_backend
    if solver_backend not in {"omniretarget", "native_clarabel"}:
        raise ValueError(
            f"solver_backend must be 'omniretarget' or 'native_clarabel', got {solver_backend!r}"
        )
    retargeter._solver_backend = solver_backend
    initial_step_size = float(config.initial_step_size)
    if not np.isfinite(initial_step_size) or initial_step_size < retargeter.step_size:
        raise ValueError(
            "initial_step_size must be finite and at least the ordinary step_size, got "
            f"{initial_step_size} < {retargeter.step_size}"
        )
    retargeter._initial_step_size = initial_step_size
    ctx.logger.info(
        f"QP solver backend: {solver_backend}; first-frame trust radius "
        f"{initial_step_size:g} (ordinary {retargeter.step_size:g})"
    )
    swap_model_and_fix_limits(
        retargeter,
        ctx.model,
        robot_dof,
        config.foot_links,
        ctx.logger,
        base_smooth_weight=config.smooth_weight,
        trunk_smooth_weight=config.trunk_smooth_weight,
        trunk_q_diag=config.trunk_q_diag,
        axial_smooth_weight=config.axial_smooth_weight,
        # Damped only where something pushes on the toes, i.e. alongside swing clearance, so
        # flat ground keeps the behaviour its results were measured under.
        mtp_smooth_weight=config.mtp_smooth_weight,
    )
    _bind_omniretarget_scene_collision(ctx, retargeter)
    return retargeter


def _bind_omniretarget_scene_collision(ctx: SolveContext, retargeter: TerraRetargeter) -> None:
    """Bind OmniRetarget's inherited object constraint to fitted terrain geoms.

    OmniRetarget selects object collisions by matching ``object_name`` against MuJoCo
    geom names.  TERRA's static interaction scene intentionally uses the identity
    ``ground`` frame, whereas its fitted primitives are named ``terrain_box_*``.  The
    binding therefore happens only after the environment model has replaced the
    constructor's plain robot model.  Foot mode, rather than the surrounding method
    profile, selects the inherited OmniRetarget constraints so the matched-core
    ablation can retain TERRA's solver and initialization settings.  The binding changes
    neither the interaction-mesh frame nor any TERRA objective.
    """
    if (
        not ctx.on_terrain
        or ctx.config.foot_mode != "omniretarget"
        or not ctx.config.activate_obj_non_penetration
    ):
        return
    terrain_prefix = "terrain_box"
    terrain_geoms = [
        geom_id
        for geom_id in range(ctx.model.ngeom)
        if (mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(terrain_prefix)
    ]
    if not terrain_geoms:
        raise ValueError("OmniRetarget received non-flat terrain but no terrain_box_* geoms")
    retargeter.object_name = terrain_prefix
    ctx.diagnostics.omniretarget_collision_geom_prefix = terrain_prefix
    ctx.diagnostics.omniretarget_collision_geom_count = len(terrain_geoms)
    ctx.logger.info(
        "OmniRetarget object non-penetration bound to %d %s_* geom(s)",
        len(terrain_geoms),
        terrain_prefix,
    )


def _solve(ctx: SolveContext, retargeter, foot_sticking, robot_dof: int) -> np.ndarray:
    """Run the frame-by-frame SQP solve and remove warm-up output.

    Args:
        ctx: State for the current solve. Source joints may be trimmed in place.
        retargeter: Fully configured retargeter.
        foot_sticking: OmniRetarget per-frame foot-contact flags.
        robot_dof: Number of robot coordinates after the floating root.

    Returns:
        Generalized positions with shape ``(T, nq)`` and warm-up frames removed.
    """
    if ctx.stage is not SolveStage.CONSTRAINTS_ATTACHED:
        raise RuntimeError(f"native solve requires constraints_attached stage, found {ctx.stage.value!r}")
    logger = ctx.logger
    # Initial base pose: position from the human pelvis, orientation estimated from frame 0.
    # An identity quaternion would leave the SQP to rotate the whole body from scratch under
    # a 0.2 trust region, which converges poorly.
    _, quat_init = transform_from_human_to_world(
        ctx.human_joints[0, 0, :], ctx.scene["object_poses_src"][0], np.zeros(3)
    )
    if ctx.model.nq != 7 + robot_dof:
        raise ValueError(f"robot_dof/model mismatch during initialization: {robot_dof} vs nq={ctx.model.nq}")
    q_init = np.asarray(ctx.model.qpos0, dtype=float).copy()
    q_init[:3] = ctx.human_joints[0, 0, :3]
    q_init[3:7] = quat_init
    # MyoFullBody's polynomial knee coordinates have non-zero declared ranges even though
    # its XML qpos0 leaves every scalar coordinate at zero. Start dependent coordinates on
    # their coupling manifold, then absorb sub-nanometre coefficient/range roundoff.
    for dependent, independent, polynomial in joint_couplers(ctx.model):
        q_init[dependent] = sum(polynomial[power] * q_init[independent] ** power for power in range(5))
    for joint_id in range(ctx.model.njnt):
        if ctx.model.jnt_limited[joint_id]:
            address = int(ctx.model.jnt_qposadr[joint_id])
            q_init[address] = np.clip(q_init[address], *ctx.model.jnt_range[joint_id])

    # `retarget_motion` always calls np.savez(dest_res_path, ...), so it cannot take None.
    # Write to a scratch file and discard it; the caller persists the Trajectory through the
    # normal cache path.
    scratch = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    scratch.close()

    t0 = time.perf_counter()
    try:
        qpos, _obj_demo, _obj, _tets = retargeter.retarget_motion(
            human_joint_motions=ctx.human_joints,
            object_poses=ctx.scene["object_poses"],
            object_poses_augmented=ctx.scene["object_poses_augmented"],
            object_points_local_demo=ctx.scene["object_points_local_demo"],
            object_points_local=ctx.scene["object_points_local"],
            foot_sticking_sequences=foot_sticking,
            q_a_init=q_init,
            q_nominal_list=None,
            original=True,
            dest_res_path=scratch.name,
        )
    finally:
        Path(scratch.name).unlink(missing_ok=True)
    elapsed = time.perf_counter() - t0
    ctx.diagnostics.retarget_fps = len(qpos) / elapsed if elapsed > 0 else float("inf")
    method_name = "TERRA" if ctx.config.method_profile == "terra" else "OmniRetarget"
    logger.info(f"[OK] {method_name}: {len(qpos)} frames in {elapsed:.2f}s ({ctx.diagnostics.retarget_fps:.2f} FPS)")
    relaxed = sorted(retargeter.nonpenetration_relaxed_frames)
    if relaxed:
        logger.warning(
            f"Non-penetration relaxed for one SQP iteration on {len(relaxed)} frame(s) "
            f"{relaxed[:8]}{' ...' if len(relaxed) > 8 else ''}; the QP was infeasible with "
            f"it. Check the penetration figures on those frames."
        )
    native_fallbacks = int(getattr(retargeter, "_native_fallback_count", 0))
    if native_fallbacks:
        frames = sorted(getattr(retargeter, "_native_fallback_frames", ()))
        logger.warning(
            f"Native Clarabel used the condensed CVXPY fallback for {native_fallbacks} "
            f"SQP iteration(s) on {len(frames)} frame(s) "
            f"{frames[:8]}{' ...' if len(frames) > 8 else ''}."
        )
    qpos = np.asarray(qpos)[:, : ctx.model.nq]
    ctx.diagnostics.numerical_envelope = validate_numerical_output(
        qpos,
        ctx.human_joints[:, 0, :3],
        max_root_step=float(ctx.config.max_root_step),
        max_root_source_deviation=float(ctx.config.max_root_source_deviation),
    )
    if ctx.warmup:
        qpos = qpos[ctx.warmup :]
        ctx.human_joints = ctx.human_joints[ctx.warmup - 1 :]
    ctx.advance(SolveStage.CONSTRAINTS_ATTACHED, SolveStage.SOLVED)
    return qpos


def validate_numerical_output(
    qpos: np.ndarray,
    source_pelvis: np.ndarray,
    *,
    max_root_step: float = DEFAULT_MAX_ROOT_STEP,
    max_root_source_deviation: float = DEFAULT_MAX_ROOT_SOURCE_DEVIATION,
) -> NumericalEnvelope:
    """Reject solver output that escaped the physical trajectory envelope.

    These checks are guards, not retargeting costs: thresholds are intentionally much wider
    than human gait so they do not alter a valid TERRA solution. They catch the two observed
    Clarabel fallback failures before a corrupt trajectory can enter the cache.

    Args:
        qpos: Solver generalized positions, with floating-root translation first.
        source_pelvis: Source pelvis positions on the same solver time base.
        max_root_step: Largest permitted root displacement between source frames.
        max_root_source_deviation: Largest permitted root-to-source-pelvis distance.

    Returns:
        Measured maximum root step and source deviation.

    Raises:
        ValueError: If shapes or thresholds are invalid, or the output is non-finite or
            outside the defensive envelope.
    """
    qpos = np.asarray(qpos, dtype=float)
    source_pelvis = np.asarray(source_pelvis, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] < 7 or not len(qpos):
        raise ValueError(f"retargeted qpos must have non-empty shape (T, nq>=7), got {qpos.shape}")
    if source_pelvis.shape != (len(qpos), 3):
        raise ValueError(
            f"source pelvis must match retargeted frames with shape ({len(qpos)}, 3), got {source_pelvis.shape}"
        )
    if not np.isfinite(max_root_step) or max_root_step <= 0:
        raise ValueError("max_root_step must be finite and positive")
    if not np.isfinite(max_root_source_deviation) or max_root_source_deviation <= 0:
        raise ValueError("max_root_source_deviation must be finite and positive")
    if not np.isfinite(qpos).all():
        bad = np.argwhere(~np.isfinite(qpos))[0]
        raise ValueError(f"retargeted qpos contains a non-finite value at frame {bad[0]}, coordinate {bad[1]}")
    if not np.isfinite(source_pelvis).all():
        raise ValueError("source pelvis contains non-finite values")

    root = qpos[:, :3]
    root_step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    source_deviation = np.linalg.norm(root - source_pelvis, axis=1)
    max_step = float(root_step.max(initial=0.0))
    max_deviation = float(source_deviation.max(initial=0.0))
    if max_step > max_root_step:
        frame = int(np.argmax(root_step)) + 1
        raise ValueError(
            "retargeting numerical envelope failed: root moved "
            f"{max_step:.3f} m into frame {frame} (limit {max_root_step:.3f} m)"
        )
    if max_deviation > max_root_source_deviation:
        frame = int(np.argmax(source_deviation))
        raise ValueError(
            "retargeting numerical envelope failed: root is "
            f"{max_deviation:.3f} m from source pelvis at frame {frame} "
            f"(limit {max_root_source_deviation:.3f} m)"
        )
    return NumericalEnvelope(
        max_root_step_m=max_step,
        max_root_source_deviation_m=max_deviation,
    )


@dataclass(frozen=True)
class _RobotAssets:
    """Resolved robot and SMPL-H assets for one retargeting run."""

    base_env_name: str
    smpl_model_path: str
    fitted_shape_path: str


@dataclass(frozen=True)
class _SolverSetup:
    """Solver components needed after constraint construction."""

    retargeter: TerraRetargeter
    foot_sticking: np.ndarray
    selfpen_mode: str


@dataclass(frozen=True)
class _PostprocessResult:
    """Materialized trajectory and diagnostics produced after the native solve."""

    trajectory: Trajectory
    site_names: np.ndarray
    position_error: np.ndarray
    solver_penetration: float


def _resolve_robot_assets(
    env_name: str,
    smpl_model_path: str | os.PathLike | None,
    fitted_shape_path: str | os.PathLike | None,
) -> _RobotAssets:
    """Resolve and validate the model assets needed by the solver."""
    base_env_name = env_name.replace("Mjx", "")
    if base_env_name not in ENV_TO_OMNIRETARGET_ROBOT:
        raise ValueError(f"OmniRetarget not configured for '{env_name}'. Supported: {list(ENV_TO_OMNIRETARGET_ROBOT)}")

    resolved_smpl_path = str(resolve_model_path(smpl_model_path))
    resolved_shape_path = str(
        Path(fitted_shape_path).expanduser().resolve()
        if fitted_shape_path is not None
        else shape_cache_path(base_env_name, None)
    )
    if not os.path.exists(resolved_shape_path):
        raise FileNotFoundError(
            f"Fitted SMPL shape not found at {resolved_shape_path}. Run `fit_smpl_shape` first "
            "(the `smpl` retargeting path creates it on first use)."
        )
    return _RobotAssets(base_env_name, resolved_smpl_path, resolved_shape_path)


def _interaction_scene(
    scene: dict | None,
    terrain,
    num_frames: int,
    config: SolverConfig,
    logger: logging.Logger,
) -> dict:
    """Return a caller-supplied scene or construct one for the solve."""
    if scene is not None:
        return scene

    spacing = config.terrain_point_spacing
    if terrain is not None and not terrain.is_flat:
        terrain_mesh = terrain_scene(
            terrain,
            num_frames,
            spacing=spacing,
            ground_range=config.ground_range or (-3.0, 3.0),
            ground_size=config.ground_size if config.ground_size is not None else 8,
        )
        logger.info(f"Interaction mesh: {len(terrain_mesh['object_points_local'])} terrain surface points")
        return terrain_mesh

    return flat_ground_scene(
        num_frames,
        config.ground_range or (-10.0, 10.0),
        config.ground_size if config.ground_size is not None else 10,
    )


def _create_solve_context(
    config: SolverConfig,
    logger: logging.Logger,
    model: mujoco.MjModel,
    terrain,
    fps: float,
    human_joints: np.ndarray,
    smpl_rotations: np.ndarray,
    scene: dict,
) -> SolveContext:
    """Build the shared solve context and resolve additional landmarks."""
    joints_mapping = {**SMPLH_TO_MYOFULLBODY, **dict(config.extra_landmarks)}
    if len(joints_mapping) != len(SMPLH_TO_MYOFULLBODY):
        logger.info(f"Position landmarks: {len(joints_mapping)} (added {dict(config.extra_landmarks)})")
    return SolveContext(
        config=config,
        logger=logger,
        model=model,
        terrain=terrain,
        fps=fps,
        joints_mapping=joints_mapping,
        human_joints=human_joints,
        smpl_rotations=smpl_rotations,
        scene=scene,
    )


def _configure_solver(
    ctx: SolveContext,
    env,
    robot_conf: DictConfig,
    assets: _RobotAssets,
    robot_dof: int,
    robot_height: float,
) -> _SolverSetup:
    """Construct the retargeter and attach all configured solve terms."""
    config, logger = ctx.config, ctx.logger
    foot_mode = config.foot_mode
    if foot_mode not in ("anchored", "omniretarget", "off"):
        raise ValueError(f"foot_mode must be 'anchored', 'omniretarget' or 'off', got {foot_mode!r}")

    retargeter = _build_retargeter(ctx, robot_dof, robot_height, assets.base_env_name, foot_mode)
    nonpen_mode = attach_non_penetration(ctx, retargeter)
    selfpen_mode = attach_leg_self_collision(ctx, retargeter)
    apply_sole_offsets(ctx)
    pad_for_warmup(ctx)
    attach_orientation(ctx, retargeter, env, robot_conf, assets.fitted_shape_path)

    toe_names = ["L_Toe", "R_Toe"]
    toe_contact = source_contact_schedule(ctx, retargeter, toe_names)
    toe_sticking = source_sticking_schedule(ctx, retargeter, toe_names)
    apply_stance_landmark_clearance_cap(ctx, toe_contact)
    attach_stance_height_targets(ctx, retargeter, toe_contact)
    attach_seat_contact_targets(ctx, retargeter)
    attach_swing_clearance(ctx, retargeter, toe_contact, nonpen_mode)

    # OmniRetarget thresholds per-frame toe displacement. Express the public setting as
    # speed using 0.01 m per source frame as the default.
    foot_speed = config.foot_planted_speed if config.foot_planted_speed is not None else 0.01 * ctx.fps
    foot_sticking = extract_foot_sticking_sequence_velocity(
        ctx.human_joints,
        retargeter.demo_joints,
        toe_names,
        velocity_threshold=foot_speed / ctx.fps,
    )

    coupler_weight = float(config.coupler_weight)
    if coupler_weight > 0:
        couplers = joint_couplers(ctx.model)
        retargeter.attach_joint_couplers(couplers, coupler_weight)
        logger.info(f"Joint couplers: {len(couplers)} active, weight {coupler_weight:g}")

    attach_foot_anchor(ctx, retargeter, toe_contact, toe_sticking, foot_sticking, toe_names, foot_mode)
    apply_swing_route(ctx, retargeter, toe_contact)
    ctx.advance(SolveStage.PREPARED, SolveStage.CONSTRAINTS_ATTACHED)
    return _SolverSetup(retargeter, foot_sticking, selfpen_mode)


def _postprocess_solution(
    ctx: SolveContext,
    env,
    env_name: str,
    retargeter: TerraRetargeter,
    qpos: np.ndarray,
    selfpen_mode: str,
) -> _PostprocessResult:
    """Align, materialize, repair, and measure a native solver result."""
    if ctx.stage is not SolveStage.SOLVED:
        raise RuntimeError(f"postprocessing requires solved stage, found {ctx.stage.value!r}")
    model = ctx.model
    data = mujoco.MjData(model)
    if not ctx.on_terrain:
        # Measure landmark error before ground alignment changes the root height.
        position_error = landmark_error(ctx, data, qpos, retargeter.demo_joints)[1:-1]
        solver_penetration = align_to_ground(ctx, data, qpos)
        trajectory, site_names = assemble_trajectory(ctx, env, env_name, data, qpos)
        ctx.advance(SolveStage.SOLVED, SolveStage.POSTPROCESSED)
        return _PostprocessResult(trajectory, site_names, position_error, solver_penetration)

    # Materialize at the environment control rate before repairing interpolation-induced
    # terrain collisions and tendon transitions. The shared outer materializer then leaves
    # this exact control-rate sequence unchanged.
    solver_penetration = align_to_ground(ctx, data, qpos)
    trajectory, site_names = assemble_trajectory(ctx, env, env_name, data, qpos)
    trajectory = materialize_trajectory(
        trajectory,
        model,
        env.dt,
        cache_type=TrajectoryCacheType.NONE,
        clip_joint_ranges=True,
    )
    final_qpos = np.asarray(trajectory.data.qpos)
    final_targets = resample_frame_values(ctx.human_joints[1:-1], len(final_qpos))
    final_qpos = repair_collisions(ctx, final_qpos, selfpen_mode)
    final_qpos = repair_materialized_tendon_transitions(ctx, final_qpos, final_targets, selfpen_mode)
    trajectory = Trajectory(
        trajectory.info,
        trajectory.data.replace(qpos=np.asarray(final_qpos, dtype=np.float32)),
    )
    trajectory = materialize_trajectory(
        trajectory,
        model,
        env.dt,
        cache_type=TrajectoryCacheType.NONE,
        clip_joint_ranges=True,
    )
    position_error = landmark_error(
        ctx,
        data,
        np.asarray(trajectory.data.qpos),
        retargeter.demo_joints,
        target_joints=final_targets,
    )
    ctx.advance(SolveStage.SOLVED, SolveStage.POSTPROCESSED)
    return _PostprocessResult(trajectory, site_names, position_error, solver_penetration)


def _motion_analysis(
    ctx: SolveContext,
    retargeter: TerraRetargeter,
    native_qpos: np.ndarray,
    result: _PostprocessResult,
    terrain,
) -> dict:
    """Serialize diagnostics and settings for one motion."""
    if ctx.stage is not SolveStage.POSTPROCESSED:
        raise RuntimeError(f"analysis requires postprocessed stage, found {ctx.stage.value!r}")
    config = ctx.config
    resolved_config = config.to_dict()
    term_config = resolved_config | {"seat_contact_active": ctx.diagnostics.seat_contact_active}
    envelope = ctx.diagnostics.numerical_envelope
    if envelope is None:
        raise RuntimeError("analysis requires numerical-envelope diagnostics")
    analysis = {
        "pos_error": result.position_error,
        "retarget_fps": ctx.diagnostics.retarget_fps,
        "site_names": result.site_names,
        "solver_penetration": result.solver_penetration,
        "native_fps": ctx.fps,
        "native_frame_count": len(native_qpos) - 2,
        "trim_start_frames": 1,
        "trim_end_frames": 1,
        "numerical_envelope": envelope.to_dict(),
        "method_profile": config.method_profile,
        "solver_backend": retargeter._solver_backend,
        "nonpen_relaxed_frames": np.array(
            sorted(retargeter.nonpenetration_relaxed_frames),
            dtype=int,
        ),
    }
    if retargeter._solver_backend == "native_clarabel":
        analysis["native_fallback_count"] = int(retargeter._native_fallback_count)
        analysis["native_fallback_frames"] = np.array(sorted(retargeter._native_fallback_frames), dtype=int)
        analysis["native_fallback_reasons_json"] = json.dumps(retargeter._native_fallback_reasons)
    analysis["resolved_config_json"] = json.dumps(resolved_config, sort_keys=True, default=str)
    analysis["active_qp_terms_json"] = json.dumps(all_active_qp_terms(term_config, on_terrain=ctx.on_terrain))
    analysis["terra_specific_qp_terms_json"] = json.dumps(active_qp_terms(term_config, on_terrain=ctx.on_terrain))
    analysis["seat_contact_active"] = ctx.diagnostics.seat_contact_active
    analysis["seat_contact_rest_count"] = ctx.diagnostics.seat_contact_rest_count
    analysis["seat_contact_seat_count"] = ctx.diagnostics.seat_contact_seat_count
    if ctx.diagnostics.omniretarget_collision_geom_prefix is not None:
        analysis["omniretarget_collision_geom_prefix"] = ctx.diagnostics.omniretarget_collision_geom_prefix
        analysis["omniretarget_collision_geom_count"] = ctx.diagnostics.omniretarget_collision_geom_count
    if ctx.on_terrain:
        analysis["terrain"] = TerrainMetadata.from_terrain(terrain).to_dict()
    return analysis


def fit_terra_motion(
    env_name: str,
    robot_conf: DictConfig,
    motion_data: dict,
    logger: logging.Logger,
    terra_config: dict | None = None,
    scene: dict | None = None,
    terrain=None,
    *,
    smpl_model_path: str | os.PathLike | None = None,
    fitted_shape_path: str | os.PathLike | None = None,
) -> tuple[Trajectory, dict]:
    """Retarget one AMASS motion to the requested robot environment.

    Args:
        env_name: Environment name configured in ``ENV_TO_OMNIRETARGET_ROBOT``.
        robot_conf: Robot configuration containing environment and mimic-site data.
        motion_data: Loaded AMASS motion data.
        logger: Logger for pipeline progress and diagnostics.
        terra_config: Optional solver, terrain, and postprocessing overrides.
        scene: Optional prebuilt interaction-mesh scene dictionary.
        terrain: Target terrain specification, dictionary, JSON path, or ``"auto"``.
        smpl_model_path: Optional SMPL-H model root. Defaults to ``TERRA_MODEL_ROOT``.
        fitted_shape_path: Optional robot-fitted SMPL-H shape cache. Defaults to
            the direct cache below ``TERRA_ARTIFACT_ROOT`` for ``env_name``.

    Returns:
        A retargeted trajectory and an analysis dictionary containing tracking,
        solver, timing, configuration, and optional terrain metadata.

    Raises:
        ValueError: If the environment, configuration, or terrain is invalid.
        FileNotFoundError: If the optimized SMPL shape cache is missing.
    """
    require_omniretarget()
    raw_cfg = terra_config or {}
    config = resolve_solver_config(raw_cfg.get("method_profile"), raw_cfg)
    assets = _resolve_robot_assets(env_name, smpl_model_path, fitted_shape_path)
    robot_height = get_robot_height(assets.fitted_shape_path, assets.smpl_model_path)
    human_joints, smpl_rotations, fps, normalization, config = _source_joints(
        motion_data,
        assets.smpl_model_path,
        assets.fitted_shape_path,
        robot_height,
        config,
        logger,
    )
    num_frames = human_joints.shape[0]
    terrain = _resolve_terrain(
        terrain,
        human_joints,
        fps,
        config,
        logger,
        motion_data=motion_data,
        smpl_model_path=assets.smpl_model_path,
        fitted_shape_path=assets.fitted_shape_path,
        normalization=normalization,
    )
    on_terrain = terrain is not None and not terrain.is_flat
    # Materialize every scene-dependent default before building SolveContext so the
    # Analysis metadata records the complete configuration that actually ran.
    config = config.for_scene(on_terrain=on_terrain)
    env = _build_environment(env_name, robot_conf, terrain, logger)
    model = env._model
    robot_dof = model.nq - 7
    logger.info(f"ROBOT_HEIGHT={robot_height:.4f} m, ROBOT_DOF={robot_dof}")
    scene = _interaction_scene(scene, terrain, num_frames, config, logger)
    ctx = _create_solve_context(
        config,
        logger,
        model,
        terrain,
        fps,
        human_joints,
        smpl_rotations,
        scene,
    )
    setup = _configure_solver(ctx, env, robot_conf, assets, robot_dof, robot_height)
    native_qpos = _solve(ctx, setup.retargeter, setup.foot_sticking, robot_dof)
    native_qpos = repair_collisions(ctx, native_qpos, setup.selfpen_mode)
    result = _postprocess_solution(
        ctx,
        env,
        env_name,
        setup.retargeter,
        native_qpos,
        setup.selfpen_mode,
    )
    analysis = _motion_analysis(
        ctx,
        setup.retargeter,
        native_qpos,
        result,
        terrain,
    )
    return result.trajectory, analysis
