"""Configure the retargeting terms used to solve a single motion.

The helpers in this module update a shared :class:`SolveContext` and attach
collision, orientation, contact, and foot-motion terms to the retargeter.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np

from terra._musclemimic import scene_geom_ids, torso_frame
from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
from terra.clearance import (
    sole_clearance_targets,
    source_sole_clearance,
    swing_foot_target_offsets,
)
from terra.collisions import starts_on_raised_terrain, walkable_terrain
from terra.constants import (
    FOOT_LANDMARK_SIDES,
    FOOT_MIMIC_SITES,
    MYOFULLBODY_FOOT_LINKS,
    MYOFULLBODY_FOOT_SITES,
    MYOFULLBODY_SEAT_GEOMS,
    MYOFULLBODY_SOLE_GEOMS,
)
from terra.contacts import (
    contact_ramp,
    foot_orientation_offsets,
    foot_sole_offsets,
    robot_sole_offsets,
    source_foot_contact,
    source_foot_sticking,
)
from terra.defaults import (
    DEFAULT_FLAT_FOOT_ORIENT_WEIGHT,
    DEFAULT_FLAT_SELF_COLLISION_WEIGHT,
    DEFAULT_SELF_COLLISION_WEIGHT,
)
from terra.profiles import SolverConfig
from terra.targets import build_orientation_targets
from terra.terrain.seats import SEAT_GEOM_PREFIX, detect_seat_rests


class SolveStage(StrEnum):
    """Observable stages in the production retargeting pipeline."""

    PREPARED = "prepared"
    CONSTRAINTS_ATTACHED = "constraints_attached"
    SOLVED = "solved"
    POSTPROCESSED = "postprocessed"


@dataclass(frozen=True)
class NumericalEnvelope(Mapping[str, float]):
    """Measured defensive bounds for an accepted native trajectory."""

    max_root_step_m: float
    max_root_source_deviation_m: float

    def to_dict(self) -> dict[str, float]:
        """Return the stable analysis representation."""
        return {
            "max_root_step_m": self.max_root_step_m,
            "max_root_source_deviation_m": self.max_root_source_deviation_m,
        }

    def __getitem__(self, key: str) -> float:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 2


@dataclass
class SolveDiagnostics:
    """Runtime observations produced by solver assembly and execution."""

    numerical_envelope: NumericalEnvelope | None = None
    retarget_fps: float = 0.0
    seat_contact_active: bool = False
    seat_contact_rest_count: int = 0
    seat_contact_seat_count: int = 0
    omniretarget_collision_geom_prefix: str | None = None
    omniretarget_collision_geom_count: int = 0


@dataclass
class SolveContext:
    """Store the shared state for a single retargeting solve.

    Attributes:
        config: Resolved, immutable solver configuration.
        logger: Logger used for progress and diagnostic messages.
        model: MuJoCo environment model containing any terrain geometry.
        terrain: Target terrain, or ``None`` for flat ground.
        fps: Source frame rate.
        joints_mapping: Mapping from SMPL joint names to robot body names.
        human_joints: Source joint positions with shape ``(T, J, 3)``.
        smpl_rotations: Source rotations with shape ``(T, 52, 3, 3)``.
        scene: Interaction-mesh scene data indexed by solver frame.
        shifted_sole_offsets: Landmark-to-sole offsets after the landmark targets
            have been shifted into the robot convention.
        warmup: Number of static frames prepended to the motion.
        diagnostics: Runtime measurements populated by later solve stages.
        stage: Current production solve stage.
    """

    config: SolverConfig
    logger: logging.Logger
    model: mujoco.MjModel
    terrain: object | None
    fps: float
    joints_mapping: dict[str, str]
    human_joints: np.ndarray
    smpl_rotations: np.ndarray
    scene: dict
    shifted_sole_offsets: dict[str, float] | None = None
    warmup: int = 0
    diagnostics: SolveDiagnostics = field(default_factory=SolveDiagnostics)
    stage: SolveStage = SolveStage.PREPARED

    def __post_init__(self) -> None:
        """Validate public mapping inputs before any constraint can consume them."""
        if not isinstance(self.config, SolverConfig):
            self.config = SolverConfig.from_mapping(self.config).for_scene(on_terrain=self.on_terrain)

    def advance(self, expected: SolveStage, target: SolveStage) -> None:
        """Advance the pipeline only from the expected auditable stage."""
        if self.stage is not expected:
            raise RuntimeError(
                f"invalid solver stage transition: expected {expected.value!r}, "
                f"found {self.stage.value!r}, requested {target.value!r}"
            )
        self.stage = target

    @property
    def on_terrain(self) -> bool:
        """Return whether the target contains non-flat terrain."""
        return self.terrain is not None and not self.terrain.is_flat

    @property
    def demo_joints(self) -> list[str]:
        """Return joint names in the order used by ``human_joints``."""
        return list(SMPLH_DEMO_JOINTS)

    def warmup_frames(self) -> int:
        """Return the configured nonnegative warm-up length."""
        return max(int(self.config.warmup_frames), 0)

    def default_engage_frame(self) -> int:
        """Return the default frame for activating collision terms."""
        return max(self.warmup_frames() - 5, 0)


def attach_non_penetration(ctx: SolveContext, retargeter) -> str:
    """Attach static-environment non-penetration constraints.

    Args:
        ctx: State and configuration for the current solve.
        retargeter: Retargeter that receives the constraints.

    Returns:
        Selected non-penetration mode: ``"scene"`` or ``"off"``.

    Raises:
        ValueError: If ``nonpen_mode`` is not supported.
    """
    cfg, logger = ctx.config, ctx.logger
    nonpen_mode = cfg.nonpen_mode
    if nonpen_mode == "scene":
        env_geoms = scene_geom_ids(ctx.model)
        starts_raised = ctx.on_terrain and starts_on_raised_terrain(ctx.human_joints, SMPLH_DEMO_JOINTS, ctx.terrain)
        engage = cfg.nonpen_engage_frame
        if engage is None:
            engage = 0 if starts_raised else ctx.default_engage_frame()
        retargeter.attach_environment_geoms(
            env_geoms,
            max_recovery_per_iter=cfg.nonpen_max_recovery,
            engage_from_frame=engage,
        )
        logger.info(
            f"Non-penetration against {len(env_geoms)} environment geom(s), from frame {engage}"
            + (" (the motion starts on raised terrain)" if starts_raised else "")
        )
    elif nonpen_mode != "off":
        raise ValueError(f"nonpen_mode must be 'scene' or 'off', got {nonpen_mode!r}")
    return nonpen_mode


def attach_leg_self_collision(ctx: SolveContext, retargeter) -> str:
    """Attach the configured collision cost between the left and right legs.

    Args:
        ctx: State and configuration for the current solve.
        retargeter: Retargeter that receives the collision cost.

    Returns:
        Selected self-collision mode: ``"legs"`` or ``"off"``.

    Raises:
        ValueError: If ``selfpen_mode`` is not supported.
    """
    cfg, logger = ctx.config, ctx.logger
    selfpen_mode = cfg.selfpen_mode
    if selfpen_mode == "legs":
        weight = cfg.selfpen_weight
        if weight is None:  # Defensive for contexts constructed outside the production pipeline.
            weight = DEFAULT_SELF_COLLISION_WEIGHT if ctx.on_terrain else DEFAULT_FLAT_SELF_COLLISION_WEIGHT
        # Never engage before non-penetration does: this term assumes `mj_forward` has
        # already run for the current q, which the non-penetration override is what does.
        engage = max(
            int(cfg.selfpen_engage_frame if cfg.selfpen_engage_frame is not None else ctx.default_engage_frame()),
            int(cfg.nonpen_engage_frame if cfg.nonpen_engage_frame is not None else ctx.default_engage_frame()),
        )
        n_pairs = retargeter.attach_self_collision(
            cfg.selfpen_pairs,
            tolerance=cfg.selfpen_tolerance,
            weight=weight,
            max_recovery_per_iter=cfg.selfpen_max_recovery_per_iter,
            engage_from_frame=engage,
        )
        logger.info(f"Left/right self-collision on {n_pairs} geom pair(s), from frame {engage}")
    elif selfpen_mode != "off":
        raise ValueError(f"selfpen_mode must be 'legs' or 'off', got {selfpen_mode!r}")
    return selfpen_mode


def apply_sole_offsets(ctx: SolveContext) -> None:
    """Apply robot-sole offsets to the source foot landmarks in place.

    Args:
        ctx: Solve context whose ``human_joints`` are updated.

    Raises:
        ValueError: If ``sole_offset_mode`` is not supported.
    """
    cfg, logger = ctx.config, ctx.logger
    sole_offset_mode = cfg.sole_offset_mode or ("on" if ctx.on_terrain else "off")
    if sole_offset_mode not in ("on", "off"):
        raise ValueError(f"sole_offset_mode must be 'on' or 'off', got {sole_offset_mode!r}")
    if sole_offset_mode != "on":
        logger.info("Foot sole offset disabled")
        return

    source_offsets = cfg.source_sole_offsets
    offsets = foot_sole_offsets(
        ctx.model,
        ctx.human_joints,
        ctx.demo_joints,
        ctx.fps,
        ctx.joints_mapping,
        logger,
        human_offsets=source_offsets,
    )
    if not offsets:
        logger.warning("Foot sole offset requested but no offsets could be measured")
        return

    ctx.human_joints = ctx.human_joints.copy()
    for joint, dz in offsets.items():
        ctx.human_joints[:, SMPLH_DEMO_JOINTS.index(joint), 2] += dz
    if source_offsets is not None:
        # Downstream functions see these already-shifted targets, so their anatomical
        # offsets require the same shift: (p + dz) - (h + dz) == p - h.  Reusing ``h``
        # here makes every stance look like positive sole clearance.
        ctx.shifted_sole_offsets = {
            joint: float(source_offsets[joint]) + dz for joint, dz in offsets.items() if joint in source_offsets
        }
    logger.info(
        "Foot sole offset applied to landmark targets (mm): "
        + "  ".join(f"{j}={dz * 1000:+.0f}" for j, dz in sorted(offsets.items()))
    )


def pad_for_warmup(ctx: SolveContext) -> None:
    """Prepend static frames to every frame-indexed solve input.

    Args:
        ctx: Solve context to pad. The function updates ``warmup``,
            ``human_joints``, ``smpl_rotations``, and ``scene``.
    """
    warmup = ctx.warmup_frames()
    ctx.warmup = warmup
    if not warmup:
        return

    def pad(a: np.ndarray) -> np.ndarray:
        """Prepend copies of the first frame to an array."""
        return np.concatenate([np.repeat(a[:1], warmup, axis=0), a])

    ctx.human_joints = pad(ctx.human_joints)
    ctx.smpl_rotations = pad(ctx.smpl_rotations)
    ctx.scene = {
        **ctx.scene,
        **{k: pad(np.asarray(ctx.scene[k])) for k in ("object_poses", "object_poses_augmented", "object_poses_src")},
    }
    ctx.logger.info(f"Warm-up: {warmup} static frames prepended (discarded after the solve)")


UPPER_BODY_MIMIC_SITES = frozenset(
    {
        "head_mimic",
        "upper_body_mimic",
        "left_shoulder_mimic",
        "left_elbow_mimic",
        "left_hand_mimic",
        "right_shoulder_mimic",
        "right_elbow_mimic",
        "right_hand_mimic",
    }
)


def position_torso_orientation_targets(
    human_joints: np.ndarray,
    demo_joints: list[str],
    reference_target: np.ndarray,
) -> np.ndarray:
    """Transfer position-derived torso-frame dynamics onto one robot-site convention."""
    required = ("Pelvis", "L_Shoulder", "R_Shoulder")
    missing = [name for name in required if name not in demo_joints]
    if missing:
        raise ValueError(f"position-derived torso frame is missing {missing}")
    frames = torso_frame(
        human_joints[:, demo_joints.index("Pelvis")],
        human_joints[:, demo_joints.index("L_Shoulder")],
        human_joints[:, demo_joints.index("R_Shoulder")],
    )
    reference_target = np.asarray(reference_target, dtype=float)
    if reference_target.shape != (3, 3):
        raise ValueError("reference_target must have shape (3, 3)")
    dynamics = frames @ frames[0].T
    return dynamics @ reference_target


def _orientation_site_weights(
    site_names: list[str],
    orient_weight: float,
    foot_orient_weight: float | None,
    upper_orient_weight: float | None = None,
    torso_orient_weight: float | None = None,
) -> np.ndarray:
    """Build orientation weights for foot and non-foot sites.

    Args:
        site_names: Ordered orientation-site names.
        orient_weight: Weight assigned to non-foot sites.
        foot_orient_weight: Optional weight assigned to foot sites.
        upper_orient_weight: Optional weight assigned to head, trunk, and arm sites.
        torso_orient_weight: Optional override for ``upper_body_mimic`` alone.

    Returns:
        Per-site weights in the same order as ``site_names``.

    Raises:
        ValueError: If any site weight is negative or not finite.
    """
    if not np.isfinite(orient_weight) or orient_weight < 0:
        raise ValueError("orient_weight must be finite and non-negative")
    foot_weight = orient_weight if foot_orient_weight is None else float(foot_orient_weight)
    if not np.isfinite(foot_weight) or foot_weight < 0:
        raise ValueError("foot_orient_weight must be finite and non-negative")
    upper_weight = orient_weight if upper_orient_weight is None else float(upper_orient_weight)
    if not np.isfinite(upper_weight) or upper_weight < 0:
        raise ValueError("upper_orient_weight must be finite and non-negative")
    torso_weight = upper_weight if torso_orient_weight is None else float(torso_orient_weight)
    if not np.isfinite(torso_weight) or torso_weight < 0:
        raise ValueError("torso_orient_weight must be finite and non-negative")
    return np.array(
        [
            foot_weight
            if site in FOOT_MIMIC_SITES
            else torso_weight
            if site == "upper_body_mimic"
            else upper_weight
            if site in UPPER_BODY_MIMIC_SITES
            else orient_weight
            for site in site_names
        ],
        dtype=float,
    )


def _default_foot_orientation_weight(on_terrain: bool, orient_weight: float) -> float:
    """Return the default foot-orientation weight for the scene type.

    Args:
        on_terrain: Whether the solve uses non-flat terrain.
        orient_weight: General orientation weight.

    Returns:
        The default weight for foot orientation sites.
    """
    return orient_weight if on_terrain else DEFAULT_FLAT_FOOT_ORIENT_WEIGHT


def attach_orientation(ctx: SolveContext, retargeter, env, robot_conf, fitted_shape_path: str) -> None:
    """Build and attach orientation targets for the robot mimic sites.

    Args:
        ctx: State and configuration for the current solve.
        retargeter: Retargeter that receives the orientation targets.
        env: Environment used to resolve mimic sites.
        robot_conf: Robot configuration containing site-to-joint matches.
        fitted_shape_path: Path to the optimized SMPL shape parameters.

    Raises:
        ValueError: If ``foot_orient_mode`` or a site weight is invalid.
    """
    cfg, logger = ctx.config, ctx.logger
    orient_weight = float(cfg.orient_weight)
    default_foot_weight = _default_foot_orientation_weight(ctx.on_terrain, orient_weight)
    foot_orient_weight = float(cfg.foot_orient_weight if cfg.foot_orient_weight is not None else default_foot_weight)
    upper_orient_weight = float(cfg.upper_orient_weight if cfg.upper_orient_weight is not None else orient_weight)
    torso_frame_mode = cfg.torso_frame_mode
    if torso_frame_mode not in ("off", "positions"):
        raise ValueError(f"torso_frame_mode must be 'off' or 'positions', got {torso_frame_mode!r}")
    torso_orient_weight = float(cfg.torso_orient_weight) if torso_frame_mode == "positions" else None
    if (
        max(
            orient_weight,
            foot_orient_weight,
            upper_orient_weight,
            0.0 if torso_orient_weight is None else torso_orient_weight,
        )
        <= 0
    ):
        logger.info("Orientation cost disabled (all orientation weights are zero)")
        return

    orient = build_orientation_targets(env, robot_conf, fitted_shape_path, ctx.smpl_rotations)

    if torso_frame_mode == "positions":
        torso_index = orient["names"].index("upper_body_mimic")
        orient["targets"][:, torso_index] = position_torso_orientation_targets(
            ctx.human_joints,
            ctx.demo_joints,
            orient["targets"][0, torso_index],
        )
        logger.info(
            "Torso orientation from pelvis/shoulder positions, weight %g",
            torso_orient_weight,
        )

    foot_orient_mode = cfg.foot_orient_mode
    if foot_orient_mode not in ("on", "off"):
        raise ValueError(f"foot_orient_mode must be 'on' or 'off', got {foot_orient_mode!r}")
    if foot_orient_mode == "on":
        corrections = foot_orientation_offsets(
            ctx.model,
            orient["site_ids"],
            orient["names"],
            orient["targets"],
            ctx.human_joints,
            ctx.demo_joints,
            ctx.fps,
            ctx.terrain,
            logger=logger,
            source_offsets=ctx.shifted_sole_offsets,
        )
        for site, rotation in corrections.items():
            k = orient["names"].index(site)
            orient["targets"][:, k] = np.einsum("tij,jk->tik", orient["targets"][:, k], rotation)
    else:
        logger.info("Foot orientation offset disabled")

    # The four corrected foot targets are a much stiffer request than the other site
    # targets: unlike the original SMPL alignment, they deliberately rotate away from the
    # position-only solution. Give them an independent authority so fixing the foot datum
    # does not require weakening orientation tracking over the whole body. Terrain keeps
    # its validated global authority. Flat uses the separately validated
    # low-authority default; an explicit key always wins in either scene class.
    weights = _orientation_site_weights(
        orient["names"],
        orient_weight,
        foot_orient_weight,
        upper_orient_weight,
        torso_orient_weight,
    )
    retargeter.attach_orientation_targets(orient["site_ids"], orient["targets"], weights, orient["names"])
    logger.info(
        f"Orientation cost on {len(orient['names'])} sites, weight {orient_weight:g}; "
        f"foot-site weight {foot_orient_weight:g}; upper-site weight {upper_orient_weight:g}"
    )


def source_contact_schedule(ctx: SolveContext, retargeter, toe_names: list[str]) -> dict:
    """Compute the source contact schedule on the solver time base.

    Args:
        ctx: Solve context after warm-up padding.
        retargeter: Retargeter that defines the source joint order.
        toe_names: Toe joints to test for contact.

    Returns:
        Mapping from toe name to a Boolean contact array with shape ``(T,)``.
    """
    return source_foot_contact(
        ctx.human_joints,
        retargeter.demo_joints,
        toe_names,
        ctx.fps,
        speed_ms=ctx.config.contact_speed,
        height_m=ctx.config.contact_height,
        terrain=ctx.terrain,
    )


def source_sticking_schedule(ctx: SolveContext, retargeter, toe_names: list[str]) -> dict:
    """Compute source intervals for horizontal foot-velocity damping.

    Args:
        ctx: State and configuration for the current solve.
        retargeter: Retargeter that defines the source joint order.
        toe_names: Toe joints to evaluate.

    Returns:
        Mapping from toe name to a Boolean sticking array with shape ``(T,)``.
    """
    sticking = source_foot_sticking(
        ctx.human_joints,
        retargeter.demo_joints,
        toe_names,
        ctx.fps,
        speed_ms=ctx.config.contact_speed,
    )
    return sticking


@dataclass(frozen=True)
class _StanceLandmarkCapConfig:
    """Validated source-landmark clearance policy, in meters."""

    loaded_cap: float | None
    midstance_cap: float | None
    midstance_fraction: float
    probe_tolerance: float


@dataclass(frozen=True)
class _StanceLandmarkCapResult:
    """Corrected landmarks and diagnostics for one clearance-cap pass."""

    human_joints: np.ndarray
    corrected_frame_count: int
    max_lowering: float
    corrected_by_joint: dict[str, int]
    corrected_midstance_by_joint: dict[str, int]


def _optional_nonnegative_config(config: SolverConfig, key: str) -> float | None:
    """Read an optional finite, non-negative floating-point setting."""
    raw_value = getattr(config, key)
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be finite and non-negative") from error
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return value


def _stance_landmark_cap_config(config: SolverConfig) -> _StanceLandmarkCapConfig | None:
    """Resolve and validate the optional stance-landmark correction policy."""
    loaded_key = "stance_landmark_clearance_cap_m"
    midstance_key = "stance_landmark_midstance_both_cap_m"
    if getattr(config, loaded_key) is None and getattr(config, midstance_key) is None:
        return None

    loaded_cap = _optional_nonnegative_config(config, loaded_key)
    midstance_cap = _optional_nonnegative_config(config, midstance_key)
    try:
        midstance_fraction = float(config.stance_landmark_midstance_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("stance_landmark_midstance_fraction must be in (0, 1]") from error
    if not np.isfinite(midstance_fraction) or not 0.0 < midstance_fraction <= 1.0:
        raise ValueError("stance_landmark_midstance_fraction must be in (0, 1]")
    probe_tolerance = _optional_nonnegative_config(config, "stance_landmark_probe_tolerance_m")
    if probe_tolerance is None:
        probe_tolerance = 0.015
    return _StanceLandmarkCapConfig(
        loaded_cap=loaded_cap,
        midstance_cap=midstance_cap,
        midstance_fraction=midstance_fraction,
        probe_tolerance=probe_tolerance,
    )


def _stance_landmark_offsets(ctx: SolveContext) -> tuple[dict[str, float], dict[str, float]]:
    """Return complete source and target landmark-to-sole offsets."""
    source_offsets = ctx.shifted_sole_offsets or ctx.config.source_sole_offsets
    missing_source = sorted(set(FOOT_LANDMARK_SIDES) - set(source_offsets or {}))
    if missing_source:
        raise ValueError(f"stance landmark clearance cap requires shifted source sole offsets for {missing_source}")

    target_offsets = robot_sole_offsets(ctx.model, ctx.joints_mapping)
    missing_target = sorted(set(FOOT_LANDMARK_SIDES) - set(target_offsets))
    if missing_target:
        raise ValueError(f"stance landmark clearance cap requires robot sole offsets for {missing_target}")
    return source_offsets, target_offsets


def _surface_heights(surface, points: np.ndarray) -> np.ndarray:
    """Evaluate a walkable surface at landmark positions."""
    if surface is None:
        return np.zeros(len(points), dtype=float)
    return np.asarray(surface.height_at(points[:, 0], points[:, 1]), dtype=float)


def _foot_landmark_clearances(
    human_joints: np.ndarray,
    surface,
    sole_offsets: dict[str, float],
) -> dict[str, np.ndarray]:
    """Measure each source foot landmark above its local support surface."""
    clearances = {}
    for joint in FOOT_LANDMARK_SIDES:
        point = human_joints[:, SMPLH_DEMO_JOINTS.index(joint)]
        clearances[joint] = point[:, 2] - float(sole_offsets[joint]) - _surface_heights(surface, point)
    return clearances


def _midstance_mask(contact: np.ndarray, warmup: int, fraction: float) -> np.ndarray:
    """Keep the centered fraction of each observed contact interval.

    Recording boundaries are censored rather than observed foot transitions, so an
    interval clipped at either boundary retains authority through that boundary. A
    clipped initial interval also extends across the prepended static warm-up.
    """
    midstance = np.zeros(len(contact), dtype=bool)
    source_contact = contact[warmup:]
    edges = np.flatnonzero(np.diff(np.r_[0, source_contact.astype(np.int8), 0]) != 0).reshape(-1, 2)
    for start, end in edges:
        margin = int(np.floor(0.5 * (1.0 - fraction) * (end - start)))
        kept_start = start if start == 0 else min(start + margin, end - 1)
        kept_end = end if end == len(source_contact) else max(end - margin, kept_start + 1)
        midstance[warmup + kept_start : warmup + kept_end] = True
    if 0 < warmup < len(midstance) and midstance[warmup]:
        midstance[:warmup] = True
    return midstance


def _stance_contact_masks(
    toe_contact: dict[str, np.ndarray],
    frame_count: int,
    warmup: int,
    config: _StanceLandmarkCapConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Validate toe contact schedules and construct per-side midstance masks."""
    if not 0 <= warmup <= frame_count:
        raise ValueError(f"warmup must be between zero and the trajectory length, got {warmup}")
    masks = {}
    for toe, side in (("L_Toe", "l"), ("R_Toe", "r")):
        if toe not in toe_contact:
            raise ValueError(f"stance landmark clearance cap requires contact schedule for {toe!r}")
        contact = np.asarray(toe_contact[toe], dtype=bool)
        if contact.shape != (frame_count,):
            raise ValueError(f"contact schedule for {toe!r} must have shape ({frame_count},), got {contact.shape}")
        midstance = (
            _midstance_mask(contact, warmup, config.midstance_fraction)
            if config.midstance_cap is not None
            else np.zeros(frame_count, dtype=bool)
        )
        masks[side] = (contact, midstance)
    return masks


def _loaded_landmark_masks(
    source_clearance: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, np.ndarray]:
    """Select foot landmarks within tolerance of their side's lowest probe."""
    loaded = {}
    for joint, side in FOOT_LANDMARK_SIDES.items():
        side_joints = [name for name, joint_side in FOOT_LANDMARK_SIDES.items() if joint_side == side]
        minimum = np.min(np.stack([source_clearance[name] for name in side_joints]), axis=0)
        loaded[joint] = source_clearance[joint] <= minimum + tolerance
    return loaded


def _cap_stance_landmarks(
    ctx: SolveContext,
    toe_contact: dict[str, np.ndarray],
    config: _StanceLandmarkCapConfig,
    source_clearance: dict[str, np.ndarray],
    target_offsets: dict[str, float],
) -> _StanceLandmarkCapResult:
    """Apply the validated cap policy to a copy of the source landmarks."""
    corrected = ctx.human_joints.copy()
    corrected_frames = np.zeros(len(corrected), dtype=bool)
    corrected_by_joint = {}
    corrected_midstance_by_joint = {}
    max_lowering = 0.0
    target_surface = walkable_terrain(ctx.terrain)
    contacts = _stance_contact_masks(toe_contact, len(corrected), ctx.warmup, config)
    loaded = _loaded_landmark_masks(source_clearance, config.probe_tolerance)

    for joint, side in FOOT_LANDMARK_SIDES.items():
        contact, midstance = contacts[side]
        joint_index = SMPLH_DEMO_JOINTS.index(joint)
        point = corrected[:, joint_index]
        allowed = np.full(len(point), np.inf, dtype=float)
        if config.loaded_cap is not None:
            allowed[contact & loaded[joint]] = config.loaded_cap
        if config.midstance_cap is not None:
            allowed[contact & midstance] = np.minimum(allowed[contact & midstance], config.midstance_cap)

        ceiling = _surface_heights(target_surface, point) + float(target_offsets[joint]) + allowed
        lowering = point[:, 2] - ceiling
        mask = np.isfinite(allowed) & (lowering > 0.0)
        if not np.any(mask):
            continue
        corrected[mask, joint_index, 2] = ceiling[mask]
        corrected_frames |= mask
        max_lowering = max(max_lowering, float(np.max(lowering[mask])))
        corrected_by_joint[joint] = int(mask.sum())
        corrected_midstance_by_joint[joint] = int(np.sum(mask & midstance))

    return _StanceLandmarkCapResult(
        human_joints=corrected,
        corrected_frame_count=int(corrected_frames.sum()),
        max_lowering=max_lowering,
        corrected_by_joint=corrected_by_joint,
        corrected_midstance_by_joint=corrected_midstance_by_joint,
    )


def _stance_landmark_cap_message(
    config: _StanceLandmarkCapConfig,
    result: _StanceLandmarkCapResult,
) -> str:
    """Format a concise correction summary for the solve log."""
    message = f"Annotated stance landmark clearance cap: {result.corrected_frame_count} frame(s), "
    if config.loaded_cap is not None:
        message += f"loaded cap {config.loaded_cap * 1000:.0f} mm, "
    if config.midstance_cap is not None:
        message += f"both-end midstance cap {config.midstance_cap * 1000:.0f} mm over {config.midstance_fraction:.0%}, "
    message += f"max correction {result.max_lowering * 1000:.0f} mm"
    if result.corrected_by_joint:
        counts = ", ".join(f"{joint}={count}" for joint, count in sorted(result.corrected_by_joint.items()))
        message += f" ({counts})"
    if any(result.corrected_midstance_by_joint.values()):
        counts = ", ".join(f"{joint}={count}" for joint, count in sorted(result.corrected_midstance_by_joint.items()))
        message += f" (midstance {counts})"
    return message


def apply_stance_landmark_clearance_cap(ctx: SolveContext, toe_contact: dict[str, np.ndarray]) -> None:
    """Cap excess lift of the loaded source foot landmark during known stance.

    Marker-to-SMPL-H fitting can raise both target foot landmarks near the end of an
    annotated stance even though one end of the physical foot is still supporting the
    subject.  A stronger robot stance-height cost fights every position target and can
    trade a small floating improvement for substantial skating.  This optional processing
    correction instead lowers only the source landmark that is closest to its source
    support surface.  The other end remains untouched, preserving real heel or toe rise.

    The correction is disabled unless a loaded-foot or midstance cap is explicitly set.
    """
    config = _stance_landmark_cap_config(ctx.config)
    if config is None:
        return
    source_offsets, target_offsets = _stance_landmark_offsets(ctx)
    source_clearance = _foot_landmark_clearances(
        ctx.human_joints,
        walkable_terrain(ctx.terrain),
        source_offsets,
    )
    result = _cap_stance_landmarks(ctx, toe_contact, config, source_clearance, target_offsets)
    ctx.human_joints = result.human_joints
    ctx.logger.info(_stance_landmark_cap_message(config, result))


def attach_stance_height_targets(ctx: SolveContext, retargeter, toe_contact: dict[str, np.ndarray]) -> None:
    """Attach support-referenced vertical targets during an explicit stance policy.

    The contact annotation is independent evidence that the sole is planted.  Therefore
    the target cannot be copied from the fitted SMPL-H landmark: marker fitting can leave
    that landmark a few centimetres above the reconstructed surface, which would simply
    encode the source conversion's floating error into the robot.  Instead, place each
    tracked robot foot body at its measured default-pose height above the support surface.
    """
    weight = float(ctx.config.stance_height_weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("stance_height_weight must be finite and non-negative")
    if weight == 0:
        return
    ramp_frames = int(ctx.config.stance_height_ramp_frames)
    release_ramp_frames = int(
        ctx.config.stance_height_release_ramp_frames
        if ctx.config.stance_height_release_ramp_frames is not None
        else ramp_frames
    )
    if release_ramp_frames < 0:
        raise ValueError("stance_height_release_ramp_frames must be non-negative")
    probe_mode = ctx.config.stance_height_probe_mode
    if probe_mode not in {"all", "source_support"}:
        raise ValueError(f"stance_height_probe_mode must be 'all' or 'source_support', got {probe_mode!r}")
    probe_tolerance = float(ctx.config.stance_height_probe_tolerance_m)
    if not np.isfinite(probe_tolerance) or probe_tolerance < 0.0:
        raise ValueError("stance_height_probe_tolerance_m must be finite and non-negative")
    body_offsets = robot_sole_offsets(ctx.model, ctx.joints_mapping)
    missing = set(FOOT_LANDMARK_SIDES) - set(body_offsets)
    if missing:
        raise ValueError(f"annotated stance height requires robot sole offsets for {sorted(missing)}")
    support = walkable_terrain(ctx.terrain)
    targets = {}
    activation = {}
    source_support = walkable_terrain(ctx.terrain)
    source_clearance: dict[str, np.ndarray] = {}
    if probe_mode == "source_support":
        # Flat motions do not apply robot/source landmark shifts, so their original
        # calibrated offsets are already in the current landmark convention.
        source_offsets = ctx.shifted_sole_offsets or ctx.config.source_sole_offsets
        missing_offsets = sorted(set(FOOT_LANDMARK_SIDES) - set(source_offsets or {}))
        if missing_offsets:
            raise ValueError(f"source-support stance height requires shifted sole offsets for {missing_offsets}")
        source_clearance = _foot_landmark_clearances(
            ctx.human_joints,
            source_support,
            source_offsets,
        )

    selected_fraction = {}
    for joint, side in FOOT_LANDMARK_SIDES.items():
        toe = "L_Toe" if side == "l" else "R_Toe"
        body = ctx.joints_mapping[joint]
        source = ctx.human_joints[:, SMPLH_DEMO_JOINTS.index(joint)]
        surface_z = _surface_heights(support, source)
        targets[body] = surface_z + body_offsets[joint]
        authority = contact_ramp(
            toe_contact[toe],
            ramp_frames,
            release_ramp_frames=release_ramp_frames,
            preserve_clipped_boundaries=True,
        )
        if probe_mode == "source_support":
            side_joints = [name for name, joint_side in FOOT_LANDMARK_SIDES.items() if joint_side == side]
            minimum = np.min(np.stack([source_clearance[name] for name in side_joints]), axis=0)
            selected = source_clearance[joint] <= minimum + probe_tolerance
            authority = authority * selected
            contact_frames = np.asarray(toe_contact[toe], dtype=bool)
            selected_fraction[joint] = float(np.mean(selected[contact_frames]) if np.any(contact_frames) else 0.0)
        activation[body] = authority
    retargeter.attach_foot_stance_height(
        targets,
        activation,
        weight,
        float(ctx.config.stance_height_max_recovery_per_iter),
    )
    ctx.logger.info(
        f"Kinematic support-referenced stance height: {len(targets)} foot bodies, "
        f"weight {weight:g}, "
        f"touchdown/release ramps {ramp_frames}/{release_ramp_frames} frames, "
        f"probe mode {probe_mode}"
        + (
            " (" + ", ".join(f"{joint}={fraction:.0%}" for joint, fraction in sorted(selected_fraction.items())) + ")"
            if selected_fraction
            else ""
        )
    )


def attach_seat_contact_targets(ctx: SolveContext, retargeter) -> None:
    """Attach chair-only glute contact during motion-inferred seated rests.

    Seat reconstruction estimates the source human's physical support surface.  The robot
    has a different pelvis-to-glute morphology, so preserving the source pelvis height can
    leave its rigid glute ellipsoids in the air.  This term closes that representation gap
    against the already reconstructed seat and is a strict no-op for all other terrain.
    """
    cfg = ctx.config
    mode = cfg.seat_contact_mode
    if mode not in {"glute_distance", "off"}:
        raise ValueError(f"seat_contact_mode must be 'glute_distance' or 'off', got {mode!r}")
    ctx.diagnostics.seat_contact_active = False
    ctx.diagnostics.seat_contact_rest_count = 0
    ctx.diagnostics.seat_contact_seat_count = 0
    if mode == "off" or not ctx.on_terrain:
        return

    seats = [box for box in ctx.terrain.boxes if box.name.startswith(SEAT_GEOM_PREFIX)]
    if not seats:
        return
    rests = detect_seat_rests(ctx.human_joints, ctx.demo_joints, ctx.fps)
    if not rests:
        ctx.logger.warning(
            "Chair terrain has seat geometry but no seated rest was detected; contact calibration disabled"
        )
        return

    activation = {box.name: np.zeros(len(ctx.human_joints), dtype=float) for box in seats}
    assignments = []
    for rest in rests:
        xy = np.median(rest.xy, axis=0)
        containing = [box for box in seats if bool(box.contains_xy(xy[0], xy[1], margin=0.02))]
        candidates = containing or seats
        box = min(candidates, key=lambda value: float(np.linalg.norm(xy - np.asarray(value.pos[:2]))))
        activation[box.name][rest.start : rest.end] = 1.0
        assignments.append(box.name)

    ramp_frames = int(cfg.seat_contact_ramp_frames)
    if ramp_frames < 0:
        raise ValueError("seat_contact_ramp_frames must be non-negative")
    activation = {
        name: contact_ramp(
            values.astype(bool),
            ramp_frames,
            release_ramp_frames=ramp_frames,
            preserve_clipped_boundaries=True,
        )
        for name, values in activation.items()
        if np.any(values)
    }
    retargeter.attach_seat_contact(
        MYOFULLBODY_SEAT_GEOMS,
        activation,
        weight=float(cfg.seat_contact_weight),
        clearance=float(cfg.seat_contact_clearance_m),
        max_recovery_per_iter=float(cfg.seat_contact_max_recovery_per_iter),
    )
    ctx.diagnostics.seat_contact_active = True
    ctx.diagnostics.seat_contact_rest_count = len(rests)
    ctx.diagnostics.seat_contact_seat_count = len(activation)
    ctx.logger.info(
        "Chair morphology calibration: %d seated rest(s) assigned to %d seat(s), "
        "glute-distance weight %g, clearance %.1f mm, ramp %d frames",
        len(rests),
        len(set(assignments)),
        float(cfg.seat_contact_weight),
        1000.0 * float(cfg.seat_contact_clearance_m),
        ramp_frames,
    )


def attach_swing_clearance(ctx: SolveContext, retargeter, toe_contact: dict, nonpen_mode: str) -> None:
    """Attach per-frame sole-height targets for swing phases.

    Args:
        ctx: Solve context after warm-up padding.
        retargeter: Retargeter that receives the clearance targets.
        toe_contact: Source contact schedule keyed by toe name.
        nonpen_mode: Selected environment non-penetration mode.

    Raises:
        ValueError: If ``clearance_mode`` is invalid or no sole geoms are found.
    """
    cfg, logger = ctx.config, ctx.logger
    clearance_mode = cfg.clearance_mode or ("source" if ctx.on_terrain else "off")
    if clearance_mode == "off":
        return
    if clearance_mode != "source":
        raise ValueError(f"clearance_mode must be 'source' or 'off', got {clearance_mode!r}")
    if nonpen_mode != "scene":
        logger.warning("clearance_mode='source' needs nonpen_mode='scene'; clearance disabled")
        return

    frac = float(cfg.clearance_fraction)
    cap = float(cfg.clearance_cap)
    lookahead = float(cfg.clearance_lookahead)
    min_clear = float(cfg.min_swing_clearance)
    clear_ramp = int(cfg.clearance_ramp_frames)
    surface_ramp_seconds = float(cfg.clearance_surface_ramp_seconds)
    demo = ctx.demo_joints
    clear = source_sole_clearance(
        ctx.human_joints,
        demo,
        ctx.fps,
        ctx.terrain,
        lookahead=lookahead,
        surface_ramp_seconds=surface_ramp_seconds,
        source_offsets=ctx.shifted_sole_offsets,
    )

    # The warm-up frames repeat frame 0 and are not a swing, whatever the contact test says
    # about that pose; counting them as planted keeps the minimum inert while the solver is
    # still converging.
    swing_gate = {FOOT_LANDMARK_SIDES[toe]: c.copy() for toe, c in toe_contact.items()}
    for c in swing_gate.values():
        c[: ctx.warmup] = True
    targets = sole_clearance_targets(
        ctx.human_joints,
        demo,
        ctx.fps,
        ctx.terrain,
        lookahead=lookahead,
        fraction=frac,
        cap=cap,
        min_clearance=min_clear,
        contact=swing_gate,
        ramp_frames=clear_ramp,
        surface_ramp_seconds=surface_ramp_seconds,
        source_offsets=ctx.shifted_sole_offsets,
    )
    # A finite/-inf target boundary is another hard gate. Ease the QP term's authority at
    # every such boundary, including those caused by the source sole crossing a box top;
    # ramping only `min_clearance` left the main fractional target discontinuous.
    activation = {side: contact_ramp(np.isfinite(target), clear_ramp) for side, target in targets.items()}

    model = ctx.model
    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    sole_geoms = {
        side: [g for g, n in enumerate(geom_names) if n.startswith(prefixes)]
        for side, prefixes in MYOFULLBODY_SOLE_GEOMS.items()
    }
    missing = [s for s, g in sole_geoms.items() if not g]
    if missing:
        raise ValueError(f"No sole geoms matched for side(s) {missing}; check MYOFULLBODY_SOLE_GEOMS")

    retargeter.attach_foot_clearance(
        sole_geoms,
        targets,
        activation_by_side=activation,
        weight=float(cfg.clearance_weight),
        max_recovery_per_iter=float(cfg.clearance_max_recovery_per_iter),
    )
    logger.info(
        "Swing clearance: "
        + "  ".join(
            f"{s} active on {np.isfinite(t).mean() * 100:.0f}% of frames, "
            f"peak {np.clip(frac * clear[s], 0.0, cap).max() * 1000:.0f} mm "
            f"(source {clear[s].max() * 1000:.0f})"
            for s, t in sorted(targets.items())
        )
        + f", fraction {frac:g}, cap {cap * 1000:.0f} mm, lookahead {lookahead * 100:.0f} cm"
        + f", surface ramp {surface_ramp_seconds * 1000:.0f} ms"
        + (
            f", swing minimum {min_clear * 1000:.0f} mm over {clear_ramp} ramp frames"
            if min_clear > 0
            else ", no swing minimum"
        )
    )


def attach_foot_anchor(
    ctx: SolveContext,
    retargeter,
    toe_contact: dict,
    toe_sticking: dict,
    foot_sticking,
    toe_names: list[str],
    foot_mode: str,
) -> None:
    """Attach horizontal foot anchoring and velocity damping during stance.

    Args:
        ctx: State and configuration for the current solve.
        retargeter: Retargeter that receives the foot constraints.
        toe_contact: Height-and-speed contact schedule keyed by toe name.
        toe_sticking: Speed-only sticking schedule keyed by toe name.
        foot_sticking: Contact flags produced by the OmniRetarget retargeter.
        toe_names: Toe names ordered to match ``MYOFULLBODY_FOOT_LINKS``.
        foot_mode: Selected OmniRetarget foot-constraint mode.

    Raises:
        ValueError: If a velocity weight or limit is negative or not finite.
    """
    cfg, logger = ctx.config, ctx.logger
    if foot_mode != "anchored":
        logger.info(f"Foot mode '{foot_mode}'")
        return

    # `anchor_gate` selects only *when* to constrain; the anchoring and the ramp are
    # unaffected by it, so the two can be ablated separately.
    if cfg.anchor_gate == "omniretarget":
        contact = {t: np.array([bool(s[t]) for s in foot_sticking], dtype=bool) for t in toe_names}
    else:
        contact = {t: c.copy() for t, c in toe_contact.items()}
    # Never anchor during the warm-up: the pose is still converging, and an anchor captured
    # there would pin the opening stance to it for the whole contact. Clearing the flag also
    # makes the first real frame a fresh touchdown, so its anchor comes from the last
    # warm-up frame, which has settled.
    if ctx.warmup:
        for c in contact.values():
            c[: ctx.warmup] = False
        for c in toe_sticking.values():
            c[: ctx.warmup] = False

    label_of_toe = dict(zip(toe_names, MYOFULLBODY_FOOT_LINKS, strict=True))
    anchor_weight = cfg.foot_anchor_weight
    velocity_weight = float(cfg.foot_velocity_weight)
    if not np.isfinite(velocity_weight) or velocity_weight < 0.0:
        raise ValueError("foot_velocity_weight must be finite and non-negative")
    tracking_weight = float(cfg.foot_velocity_tracking_weight)
    if not np.isfinite(tracking_weight) or tracking_weight < 0.0:
        raise ValueError("foot_velocity_tracking_weight must be finite and non-negative")
    velocity_limit = float(cfg.foot_velocity_limit)
    if not np.isfinite(velocity_limit) or velocity_limit < 0.0:
        raise ValueError("foot_velocity_limit must be finite and non-negative")
    ramp_frames = cfg.foot_ramp_frames
    retargeter.attach_foot_anchoring(
        {label_of_toe[t]: c for t, c in contact.items()},
        weight=anchor_weight,
        ramp_frames=ramp_frames,
        preserve_clipped_boundaries=True,
        velocity_contact={label_of_toe[t]: c for t, c in toe_sticking.items()},
        velocity_sites=MYOFULLBODY_FOOT_SITES,
        velocity_weight=velocity_weight,
        velocity_tracking_weight=tracking_weight,
        velocity_tolerance_m=velocity_limit / ctx.fps,
    )
    omniretarget_frac = np.mean([[bool(s[t]) for t in toe_names] for s in foot_sticking])
    logger.info(
        f"Foot mode 'anchored': {np.mean([c.mean() for c in contact.values()]):.1%} of frames "
        f"in contact (OmniRetarget flag would give {omniretarget_frac:.1%}), "
        f"anchor weight {anchor_weight:g}, velocity tracking/excess weights "
        f"{tracking_weight:g}/{velocity_weight:g} above {velocity_limit:g} m/s, "
        f"ramp {ramp_frames} frames"
    )


def apply_swing_route(ctx: SolveContext, retargeter, toe_contact: dict) -> None:
    """Raise swing-foot landmarks and attach their routed height targets.

    Args:
        ctx: Solve context whose source landmarks are updated.
        retargeter: Retargeter that receives the route targets.
        toe_contact: Source contact schedule keyed by toe name.
    """
    cfg, logger = ctx.config, ctx.logger
    target_lift = float(cfg.swing_target_lift)
    target_ramp = int(cfg.swing_target_ramp_frames)
    if not (target_lift > 0.0 and ctx.on_terrain):
        logger.info("Whole-foot swing route disabled")
        return

    route = swing_foot_target_offsets(
        ctx.human_joints,
        ctx.demo_joints,
        ctx.fps,
        ctx.terrain,
        toe_contact,
        lift=target_lift,
        lookahead=float(cfg.clearance_lookahead),
        ramp_frames=target_ramp,
        source_offsets=ctx.shifted_sole_offsets,
    )
    route_targets = {}
    route_activation = {}
    for joint, side in FOOT_LANDMARK_SIDES.items():
        joint_idx = SMPLH_DEMO_JOINTS.index(joint)
        point = ctx.human_joints[:, joint_idx]
        z = point[:, 2] + route[side]
        body = ctx.joints_mapping[joint]
        route_targets[body] = np.where(route[side] > 0.0, z, -np.inf)
        route_activation[body] = route[side] / target_lift
    route_weight = float(cfg.swing_target_weight)
    retargeter.attach_foot_route(
        route_targets,
        route_weight,
        activation_by_body=route_activation,
        max_recovery_per_iter=float(cfg.swing_target_max_recovery_per_iter),
    )
    ctx.human_joints = ctx.human_joints.copy()
    for joint, side in FOOT_LANDMARK_SIDES.items():
        ctx.human_joints[:, SMPLH_DEMO_JOINTS.index(joint), 2] += route[side]
    logger.info(
        "Whole-foot swing route: "
        + "  ".join(
            f"{side} active on {np.mean(dz > 0):.1%} of frames, peak {dz.max() * 1000:.0f} mm"
            for side, dz in sorted(route.items())
        )
        + f", weight {route_weight:g}, ramp {target_ramp} frames, "
        f"recovery cap {retargeter.route_recovery_cap * 1000:.0f} mm"
    )
