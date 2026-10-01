"""Repair solver output and assemble cached retargeting trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.interpolate import interp1d

from terra._musclemimic import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
    joint_couplers,
    scene_geom_ids,
)
from terra.assembly import SolveContext
from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
from terra.collisions import (
    collidable_geoms,
    floor_clearance,
    max_penetration_with_geoms,
    repair_isolated_terrain_penetrations,
    self_collision_geom_pairs,
)
from terra.defaults import (
    DEFAULT_POSTHOC_PEN_THRESHOLD,
    DEFAULT_POSTHOC_TENDON_MAX_FRAMES,
    DEFAULT_POSTHOC_TENDON_MAX_LOCAL_FRAMES,
    DEFAULT_POSTHOC_TENDON_THRESHOLD,
    DEFAULT_POSTHOC_TRACKING_MAX_REGRESSION,
    DEFAULT_POSTHOC_TRACKING_MEAN_REGRESSION,
)
from terra.profiles import SolverConfig
from terra.tendon_repair import repair_tendon_discontinuities


def repair_collisions(ctx: SolveContext, qpos: np.ndarray, selfpen_mode: str) -> np.ndarray:
    """Repair short collision outliers in a terrain trajectory.

    Args:
        ctx: State and configuration for the completed solve.
        qpos: Generalized positions with shape ``(T, nq)``.
        selfpen_mode: Selected self-collision mode.

    Returns:
        Repaired generalized positions, or the original input when repair is inactive.
    """
    cfg, logger = ctx.config, ctx.logger
    if not (ctx.on_terrain and cfg.posthoc_repair):
        return qpos

    qpos, repaired = repair_isolated_terrain_penetrations(
        ctx.model,
        qpos,
        scene_geom_ids(ctx.model),
        threshold=float(cfg.posthoc_pen_threshold),
        max_run=int(cfg.posthoc_max_run),
        self_collision_body_pairs=(cfg.posthoc_selfpen_pairs if selfpen_mode == "legs" else ()),
    )
    if repaired:
        logger.info(f"Post-hoc collision repair: interpolated frame(s) {repaired}")
    else:
        logger.info("Post-hoc collision repair: no eligible isolated penetrations")
    return qpos


def resample_frame_values(values: np.ndarray, output_frames: int) -> np.ndarray:
    """Resample frame-indexed values on an endpoint-aligned grid.

    Args:
        values: Values whose first dimension indexes time.
        output_frames: Required number of output frames.

    Returns:
        Resampled values with ``output_frames`` entries on the first axis.

    Raises:
        ValueError: If the input or requested output is empty.
    """
    values = np.asarray(values)
    if output_frames < 1:
        raise ValueError(f"output_frames must be positive, got {output_frames}")
    if len(values) < 1:
        raise ValueError("cannot resample an empty frame sequence")
    if len(values) == output_frames:
        return values.copy()
    if len(values) == 1:
        return np.repeat(values, output_frames, axis=0)
    source_grid = np.arange(len(values))
    output_grid = np.linspace(0, len(values) - 1, output_frames, endpoint=True)
    kind = "cubic" if len(values) >= 4 else "linear"
    return np.asarray(interp1d(source_grid, values, kind=kind, axis=0)(output_grid))


@dataclass(frozen=True)
class _MaterializedTendonRepairPolicy:
    """Validated collision, tracking, and search limits for tendon repair."""

    penetration_threshold: float
    tracking_mean_regression: float
    tracking_max_regression: float
    tendon_threshold: float
    max_changed_frames: int
    max_local_changed_frames: int


@dataclass(frozen=True)
class _PoseQuality:
    """Safety and fidelity measurements for one materialized pose."""

    terrain_penetration: float
    self_penetration: float
    tracking_mean: float
    tracking_max: float
    coupler_error: float


def _nonnegative_float_config(config: SolverConfig, key: str, default: float) -> float:
    """Resolve one finite, non-negative floating-point configuration value."""
    try:
        value = float(getattr(config, key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be finite and non-negative") from error
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return value


def _nonnegative_int_config(config: SolverConfig, key: str, default: int) -> int:
    """Resolve one non-negative integer configuration value."""
    raw_value = getattr(config, key, default)
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a non-negative integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{key} must be a non-negative integer") from error
    if value != raw_value or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _materialized_tendon_repair_policy(config: SolverConfig) -> _MaterializedTendonRepairPolicy:
    """Resolve every materialized tendon-repair quality and search limit."""
    tendon_threshold = _nonnegative_float_config(
        config,
        "posthoc_tendon_threshold",
        DEFAULT_POSTHOC_TENDON_THRESHOLD,
    )
    if tendon_threshold == 0.0:
        raise ValueError("posthoc_tendon_threshold must be positive")
    return _MaterializedTendonRepairPolicy(
        penetration_threshold=_nonnegative_float_config(
            config,
            "posthoc_pen_threshold",
            DEFAULT_POSTHOC_PEN_THRESHOLD,
        ),
        tracking_mean_regression=_nonnegative_float_config(
            config,
            "posthoc_tracking_mean_regression",
            DEFAULT_POSTHOC_TRACKING_MEAN_REGRESSION,
        ),
        tracking_max_regression=_nonnegative_float_config(
            config,
            "posthoc_tracking_max_regression",
            DEFAULT_POSTHOC_TRACKING_MAX_REGRESSION,
        ),
        tendon_threshold=tendon_threshold,
        max_changed_frames=_nonnegative_int_config(
            config,
            "posthoc_tendon_max_frames",
            DEFAULT_POSTHOC_TENDON_MAX_FRAMES,
        ),
        max_local_changed_frames=_nonnegative_int_config(
            config,
            "posthoc_tendon_max_local_frames",
            DEFAULT_POSTHOC_TENDON_MAX_LOCAL_FRAMES,
        ),
    )


def _materialized_repair_inputs(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    target_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate materialized poses and targets, preserving storage precision."""
    # Production trajectory storage uses float32. A wrap close to its topology boundary
    # can change after that cast, so validate the exact precision evaluation will read.
    original = np.asarray(qpos, dtype=np.float32).astype(float)
    if original.ndim != 2 or original.shape[1] != model.nq:
        raise ValueError(f"materialized tendon repair qpos must have shape (T, {model.nq}), got {original.shape}")
    if not np.isfinite(original).all():
        raise ValueError("materialized tendon repair qpos must contain only finite values")

    targets = np.asarray(target_joints, dtype=float)
    if targets.ndim != 3 or targets.shape[2] != 3:
        raise ValueError(f"materialized tendon repair targets must have shape (T, J, 3), got {targets.shape}")
    if len(targets) != len(original):
        raise ValueError(f"materialized tendon repair target length must match qpos: {len(targets)} != {len(original)}")
    if not np.isfinite(targets).all():
        raise ValueError("materialized tendon repair targets must contain only finite values")
    return original, targets


def _materialized_landmark_mapping(
    model: mujoco.MjModel,
    joints_mapping: dict[str, str],
    target_joint_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve and validate the landmark indices used by the quality gate."""
    if not joints_mapping:
        raise ValueError("materialized tendon repair requires at least one landmark mapping")
    body_ids = []
    joint_indices = []
    for joint, body in joints_mapping.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            raise ValueError(f"materialized tendon repair body {body!r} does not exist in the model")
        try:
            joint_index = SMPLH_DEMO_JOINTS.index(joint)
        except ValueError as error:
            raise ValueError(f"materialized tendon repair landmark {joint!r} is not in SMPLH_DEMO_JOINTS") from error
        if joint_index >= target_joint_count:
            raise ValueError(f"materialized tendon repair target array has no coordinate for landmark {joint!r}")
        body_ids.append(body_id)
        joint_indices.append(joint_index)
    return np.asarray(body_ids, dtype=int), np.asarray(joint_indices, dtype=int)


def _joint_limits_satisfied(model: mujoco.MjModel, qpos: np.ndarray) -> bool:
    """Check scalar and ball-joint limits for one generalized-position vector."""
    tolerance = 1e-5
    for joint in range(model.njnt):
        if not model.jnt_limited[joint]:
            continue
        joint_type = int(model.jnt_type[joint])
        qpos_address = int(model.jnt_qposadr[joint])
        low, high = model.jnt_range[joint]
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            quaternion = qpos[qpos_address : qpos_address + 4]
            value = 2.0 * np.arctan2(np.linalg.norm(quaternion[1:]), abs(quaternion[0]))
        else:
            value = qpos[qpos_address]
        if value < low - tolerance or value > high + tolerance:
            return False
    return True


@dataclass
class _MaterializedTendonValidator:
    """Validate proposed tendon repairs against materialized-motion quality gates."""

    model: mujoco.MjModel
    data: mujoco.MjData
    environment_geoms: tuple[int, ...]
    self_pairs: list[tuple[int, int, str]]
    couplers: list[tuple[int, int, np.ndarray]]
    original: np.ndarray
    target_joints: np.ndarray
    mapped_body_ids: np.ndarray
    mapped_joint_indices: np.ndarray
    policy: _MaterializedTendonRepairPolicy
    baseline: dict[int, _PoseQuality] = field(default_factory=dict)

    def _measure(self, qpos: np.ndarray, frame: int) -> _PoseQuality:
        """Measure collision, tracking, and coupler error for one pose."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        terrain_penetration = max(
            0.0,
            -max_penetration_with_geoms(
                self.model,
                self.data,
                self.environment_geoms,
            ),
        )
        self_penetration = max(
            [0.0]
            + [
                -float(mujoco.mj_geomDistance(self.model, self.data, first, second, 0.05, None))
                for first, second, _label in self.self_pairs
            ]
        )
        target = self.target_joints[frame, self.mapped_joint_indices]
        tracking_error = np.linalg.norm(self.data.xpos[self.mapped_body_ids] - target, axis=1)
        coupler_error = max(
            [0.0]
            + [
                abs(qpos[dependent] - sum(poly[power] * qpos[independent] ** power for power in range(5)))
                for dependent, independent, poly in self.couplers
            ]
        )
        return _PoseQuality(
            terrain_penetration=terrain_penetration,
            self_penetration=self_penetration,
            tracking_mean=float(tracking_error.mean()),
            tracking_max=float(tracking_error.max()),
            coupler_error=coupler_error,
        )

    def _quality_is_preserved(self, before: _PoseQuality, after: _PoseQuality) -> bool:
        """Return whether a candidate stays within every regression budget."""
        return all(
            (
                after.terrain_penetration <= max(self.policy.penetration_threshold, before.terrain_penetration) + 1e-9,
                after.self_penetration <= max(self.policy.penetration_threshold, before.self_penetration) + 1e-9,
                after.tracking_mean <= before.tracking_mean + self.policy.tracking_mean_regression,
                after.tracking_max <= before.tracking_max + self.policy.tracking_max_regression,
                after.coupler_error <= max(np.radians(1.0), before.coupler_error + np.radians(0.1)),
            )
        )

    def __call__(self, candidate: np.ndarray, frames: range) -> bool:
        """Return whether every modified candidate frame is quality-safe."""
        for frame in frames:
            if frame not in self.baseline:
                self.baseline[frame] = self._measure(self.original[frame], frame)
            after = self._measure(candidate[frame], frame)
            if not self._quality_is_preserved(self.baseline[frame], after):
                return False
            if not _joint_limits_satisfied(self.model, candidate[frame]):
                return False
        return True


def _materialized_tendon_validator(
    ctx: SolveContext,
    original: np.ndarray,
    target_joints: np.ndarray,
    selfpen_mode: str,
    policy: _MaterializedTendonRepairPolicy,
) -> _MaterializedTendonValidator:
    """Build the quality validator for one materialized trajectory."""
    model = ctx.model
    mapped_body_ids, mapped_joint_indices = _materialized_landmark_mapping(
        model,
        ctx.joints_mapping,
        target_joints.shape[1],
    )
    self_collision_bodies = ctx.config.posthoc_selfpen_pairs if selfpen_mode == "legs" else ()
    return _MaterializedTendonValidator(
        model=model,
        data=mujoco.MjData(model),
        environment_geoms=tuple(scene_geom_ids(model)),
        self_pairs=self_collision_geom_pairs(model, self_collision_bodies),
        couplers=joint_couplers(model),
        original=original,
        target_joints=target_joints,
        mapped_body_ids=mapped_body_ids,
        mapped_joint_indices=mapped_joint_indices,
        policy=policy,
    )


def _log_materialized_tendon_repair(logger, frames: list[int], remaining: list[tuple[float, int, int]]) -> None:
    """Report accepted repairs or the reason unresolved events remain."""
    if frames:
        logger.info(f"Post-hoc tendon repair: interpolated frame(s) {frames}")
    if remaining:
        logger.warning(
            f"Post-hoc tendon repair left {len(remaining)} event(s); worst "
            f"{remaining[0][0]:.4f} at frame {remaining[0][1]} because no bounded "
            "quality-safe interpolation was found"
        )
    elif not frames:
        logger.info("Post-hoc tendon repair: no threshold-exceeding events")


def repair_materialized_tendon_transitions(
    ctx: SolveContext,
    qpos: np.ndarray,
    target_joints: np.ndarray,
    selfpen_mode: str,
) -> np.ndarray:
    """Repair isolated tendon discontinuities in a materialized trajectory.

    Candidate interpolations are validated against collision, tracking, coupler,
    and joint-limit constraints.

    Args:
        ctx: State and configuration for the completed solve.
        qpos: Materialized generalized positions with shape ``(T, nq)``.
        target_joints: Landmark targets on the same time base as ``qpos``.
        selfpen_mode: Selected self-collision mode.

    Returns:
        Generalized positions after accepted tendon repairs.

    Raises:
        ValueError: If the inputs, configured limits, or landmark mapping are invalid.
    """
    if not (ctx.on_terrain and ctx.config.posthoc_tendon_repair):
        return qpos

    original, target_joints = _materialized_repair_inputs(ctx.model, qpos, target_joints)
    policy = _materialized_tendon_repair_policy(ctx.config)
    validator = _materialized_tendon_validator(ctx, original, target_joints, selfpen_mode, policy)

    repaired, frames, remaining = repair_tendon_discontinuities(
        ctx.model,
        original,
        validator=validator,
        threshold=policy.tendon_threshold,
        max_changed_frames=policy.max_changed_frames,
        max_local_changed_frames=policy.max_local_changed_frames,
    )
    _log_materialized_tendon_repair(ctx.logger, frames, remaining)
    return repaired


def landmark_error(
    ctx: SolveContext,
    data: mujoco.MjData,
    qpos: np.ndarray,
    demo_joints: list[str],
    target_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Measure per-frame position error for mapped robot landmarks.

    Args:
        ctx: State for the completed solve.
        data: MuJoCo data whose generalized positions are overwritten.
        qpos: Generalized positions with shape ``(T, nq)``.
        demo_joints: Names indexing the target joint dimension.
        target_joints: Optional targets aligned to ``qpos``. Defaults to the
            targets stored in ``ctx``.

    Returns:
        Position errors in meters with shape ``(T, n_landmarks)`` and dtype
        ``float32``.

    Raises:
        ValueError: If the target sequence does not cover the trajectory.
    """
    model = ctx.model
    if target_joints is None:
        # Flat motions may have one extra source target after central-difference trimming.
        # Use the first ``len(qpos)`` entries for this analysis-only path.
        targets = np.asarray(ctx.human_joints)
        if len(targets) < len(qpos):
            raise ValueError(f"landmark target length must cover qpos: {len(targets)} < {len(qpos)}")
    else:
        targets = np.asarray(target_joints)
    if target_joints is not None and len(targets) != len(qpos):
        raise ValueError(f"landmark target length must match qpos: {len(targets)} != {len(qpos)}")
    mapped_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ctx.joints_mapping.values()]
    mapped_joint_idx = [demo_joints.index(j) for j in ctx.joints_mapping]
    pos_error = np.zeros((len(qpos), len(mapped_body_ids)), dtype=np.float32)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        pos_error[i] = np.linalg.norm(data.xpos[mapped_body_ids] - targets[i, mapped_joint_idx], axis=-1)
    ctx.logger.info(f"landmark error: mean={pos_error.mean() * 1000:.1f} mm, max={pos_error.max() * 1000:.1f} mm")
    return pos_error


def align_to_ground(ctx: SolveContext, data: mujoco.MjData, qpos: np.ndarray) -> float:
    """Align a flat-ground trajectory or measure residual terrain penetration.

    Args:
        ctx: State for the completed solve.
        data: MuJoCo data whose generalized positions are overwritten.
        qpos: Generalized positions with shape ``(T, nq)``. Flat-ground root
            heights are updated in place.

    Returns:
        Maximum solver penetration in meters before flat-ground correction.
    """
    from loco_mujoco.smpl.retargeting import max_penetration_with_floor

    model, logger = ctx.model, ctx.logger
    if ctx.on_terrain:
        env_geoms = scene_geom_ids(model)
        pen_per_frame = []
        for i in range(len(qpos)):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            pen_per_frame.append(max_penetration_with_geoms(model, data, env_geoms))
        residual_penetration = float(-min(pen_per_frame))
        logger.info(
            f"Terrain alignment: datum left untouched; residual penetration "
            f"max {residual_penetration * 1000:.1f} mm, "
            f"mean {float(-np.mean(pen_per_frame)) * 1000:.1f} mm"
        )
        return residual_penetration

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    probes = collidable_geoms(model, exclude=floor_id)
    clearance = np.array([floor_clearance(model, data, q, floor_id, probes) for q in qpos])
    if (gz := float(clearance.min())) > 0.0:
        qpos[:, 2] -= gz
        logger.info(f"Ground alignment: motion floats, lowered by {gz * 1000:.1f} mm")

    lift = np.zeros(len(qpos))
    for i in range(len(qpos)):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        pen, _, _ = max_penetration_with_floor(model, data)
        lift[i] = -pen if pen < 0.0 else 0.0
        if pen < 0.0:
            qpos[i, 2] -= pen
    logger.info(
        f"Ground alignment: solver penetration before correction "
        f"max {lift.max() * 1000:.1f} mm, mean {lift.mean() * 1000:.1f} mm, "
        f"on {np.mean(lift > 1e-3):.1%} of frames; the correction shifts the whole body "
        f"vertically by that much, changing up to {np.abs(np.diff(lift)).max() * 1000:.2f} mm "
        f"between frames"
    )
    return float(lift.max())


def assemble_trajectory(
    ctx: SolveContext,
    env,
    env_name: str,
    data: mujoco.MjData,
    qpos: np.ndarray,
) -> tuple[Trajectory, np.ndarray]:
    """Build a trajectory with velocities and mimic-site kinematics.

    Args:
        ctx: State for the completed solve.
        env: Environment that defines mimic sites and the root joint.
        env_name: Environment name used in validation messages.
        data: MuJoCo data whose generalized positions are overwritten.
        qpos: Ground-aligned generalized positions with shape ``(T, nq)``.

    Returns:
        The assembled trajectory and ordered mimic-site names. Central
        differencing removes one frame from each end of the input.
    """
    from loco_mujoco.smpl.retargeting import (
        _compute_qvel_from_qpos,
        _record_site_kinematics,
        _resolve_site_ids,
    )

    model = ctx.model
    site_names = getattr(env, "sites_for_mimic", None)
    if site_names is not None:
        site_ids = _resolve_site_ids(model, site_names, context=f"{env_name} OmniRetarget sites")
        site_xpos_full, site_xmat_full = _record_site_kinematics(model, data, qpos, site_ids)
    else:
        site_ids = None
        site_xpos_full = np.zeros((len(qpos), 0, 3), dtype=np.float32)
        site_xmat_full = np.zeros((len(qpos), 0, 9), dtype=np.float32)

    qpos, qvel = _compute_qvel_from_qpos(qpos, ctx.fps, env.root_free_joint_xml_name, model)
    site_xpos, site_xmat = site_xpos_full[1:-1], site_xmat_full[1:-1]

    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if site_ids is not None:
        ids = jnp.array(site_ids)
        traj_model = TrajectoryModel(
            njnt=model.njnt,
            jnt_type=jnp.array(model.jnt_type),
            nbody=model.nbody,
            body_rootid=jnp.array(model.body_rootid),
            body_weldid=jnp.array(model.body_weldid),
            body_mocapid=jnp.array(model.body_mocapid),
            body_pos=jnp.array(model.body_pos),
            body_quat=jnp.array(model.body_quat),
            body_ipos=jnp.array(model.body_ipos),
            body_iquat=jnp.array(model.body_iquat),
            nsite=len(site_ids),
            site_bodyid=jnp.array(model.site_bodyid)[ids],
            site_pos=jnp.array(model.site_pos)[ids],
            site_quat=jnp.array(model.site_quat)[ids],
        )
    else:
        traj_model = TrajectoryModel(model.njnt, jnp.array(model.jnt_type))

    traj_info = TrajectoryInfo(jnt_names, model=traj_model, frequency=ctx.fps, site_names=site_names)
    traj_data = TrajectoryData(
        jnp.array(qpos),
        jnp.array(qvel),
        site_xpos=jnp.array(site_xpos),
        site_xmat=jnp.array(site_xmat),
        split_points=jnp.array([0, len(qpos)]),
    )
    return Trajectory(traj_info, traj_data), site_names
