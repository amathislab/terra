"""Detect source foot contacts and calibrate robot foot targets."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from terra.collisions import walkable_terrain
from terra.constants import (
    FOOT_LANDMARK_SIDES,
    FOOT_MIMIC_SITES,
    MYOFULLBODY_SOLE_GEOMS,
)
from terra.defaults import FLAT_STANCE_TOL, MAX_FOOT_ORIENT_OFFSET, MAX_SOLE_OFFSET
from terra.terrain import (
    detect_seat_rests,
    detect_stance_events,
    drop_seated_contacts,
    joint_surface_offsets,
    paired_sole_offsets,
)


def source_foot_contact(
    human_joints: np.ndarray,
    demo_joints: list,
    toe_names: list,
    fps: float,
    speed_ms: float = 0.30,
    height_m: float = 0.06,
    terrain=None,
) -> dict:
    """Detect source foot contact from toe speed and surface-relative height.

    Args:
        human_joints: Ground-aligned source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        toe_names: Joint names to test for contact.
        fps: Source frame rate in frames per second.
        speed_ms: Maximum horizontal contact speed in meters per second.
        height_m: Maximum contact height above the support surface in meters.
        terrain: Source terrain, or ``None`` to use the motion's lowest point.

    Returns:
        Boolean contact arrays keyed by toe name.
    """
    ground = walkable_terrain(terrain)
    flat = ground is None or ground.is_flat
    floor = float(human_joints[:, :, 2].min())
    out = {}
    for name in toe_names:
        p = human_joints[:, demo_joints.index(name)]
        speed = np.linalg.norm(np.gradient(p, axis=0)[:, :2], axis=1) * fps
        datum = floor if flat else ground.height_at(p[:, 0], p[:, 1])
        out[name] = (speed < speed_ms) & ((p[:, 2] - datum) < height_m)
    return out


def source_foot_sticking(
    human_joints: np.ndarray,
    demo_joints: list,
    toe_names: list,
    fps: float,
    speed_ms: float = 0.30,
    release_guard_frames: int = 1,
) -> dict[str, np.ndarray]:
    """Detect low-speed intervals used for horizontal foot damping.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        toe_names: Joint names to evaluate.
        fps: Source frame rate in frames per second.
        speed_ms: Maximum horizontal sticking speed in meters per second.
        release_guard_frames: Samples for which to extend each sticking interval.

    Returns:
        Boolean sticking arrays keyed by toe name.

    Raises:
        ValueError: If an input shape, threshold, or frame count is invalid.
    """
    human_joints = np.asarray(human_joints, dtype=float)
    if human_joints.ndim != 3 or human_joints.shape[2] != 3:
        raise ValueError("human_joints must have shape (T, J, 3)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    if not np.isfinite(speed_ms) or speed_ms <= 0:
        raise ValueError(f"speed_ms must be positive and finite, got {speed_ms}")
    if isinstance(release_guard_frames, bool) or int(release_guard_frames) != release_guard_frames:
        raise ValueError("release_guard_frames must be a non-negative integer")
    release_guard_frames = int(release_guard_frames)
    if release_guard_frames < 0:
        raise ValueError("release_guard_frames must be a non-negative integer")

    out = {}
    for name in toe_names:
        p = human_joints[:, demo_joints.index(name), :2]
        speed = np.full(len(p), np.inf, dtype=float)
        if len(p) > 1:
            speed[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
        sticking = speed <= speed_ms
        guarded = sticking.copy()
        for lag in range(1, release_guard_frames + 1):
            guarded[lag:] |= sticking[:-lag]
        out[name] = guarded
    return out


def contact_ramp(
    contact: np.ndarray,
    ramp_frames: int,
    *,
    release_ramp_frames: int | None = None,
    preserve_clipped_boundaries: bool = False,
) -> np.ndarray:
    """Create a symmetric engagement ramp for each contact interval.

    Args:
        contact: Boolean contact array with shape ``(T,)``.
        ramp_frames: Frames used to engage at each observed touchdown and, by default,
            release at each observed foot-off.
        release_ramp_frames: Optional independent release duration.  One keeps full
            authority through the final detected contact frame; ``None`` retains the
            symmetric ramp on both boundaries.
        preserve_clipped_boundaries: Keep a contact that is already active at the
            first sample or still active at the last sample fully engaged at that
            recording boundary.  Such boundaries are censoring, not observed
            touchdown/liftoff events, so tapering there would manufacture floating.

    Returns:
        Engagement values in ``[0, 1]`` with shape ``(T,)``.
    """
    contact = np.asarray(contact, dtype=bool)
    ramp = np.zeros(len(contact), dtype=float)
    release_ramp_frames = int(ramp_frames) if release_ramp_frames is None else int(release_ramp_frames)
    if release_ramp_frames < 0:
        raise ValueError("release_ramp_frames must be non-negative")
    if ramp_frames <= 0:
        return contact.astype(float)

    edges = np.flatnonzero(np.diff(np.r_[0, contact.astype(int), 0]) != 0).reshape(-1, 2)
    for a, b in edges:
        n = b - a
        i = np.arange(n)
        # Ramp from whichever *observed* end of the contact is nearer.  A run that
        # reaches a recording boundary has no measured transition at that end.
        up = np.ones(n) if preserve_clipped_boundaries and a == 0 else np.minimum(i + 1, ramp_frames) / ramp_frames
        down = (
            np.ones(n)
            if preserve_clipped_boundaries and b == len(contact)
            else np.ones(n)
            if release_ramp_frames == 0
            else np.minimum(n - i, release_ramp_frames) / release_ramp_frames
        )
        ramp[a:b] = np.minimum(up, down)
    return ramp


def flat_stance_frames(
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain=None,
    tol: float = FLAT_STANCE_TOL,
    source_offsets: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Find frames where both landmarks of each foot rest on a surface.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        terrain: Source terrain, or ``None`` for flat ground.
        tol: Maximum distance from the estimated support level in meters.
        source_offsets: Optional ankle/toe heights calibrated on flat ground.

    Returns:
        Frame-index arrays keyed by ``"l"`` and ``"r"``.
    """
    demo_joints = list(demo_joints)
    probes = tuple(j for j in FOOT_LANDMARK_SIDES if j in demo_joints)
    # A foot parked while the pelvis is on a seat is not a calibration reference - same rule
    # as `drop_seated_contacts`, applied here to both the offset it feeds and the frames this
    # returns, since a foot held at an angle through a sit is not "the same physical posture"
    # this function promises even where it happens to pass the flatness test.
    rests = detect_seat_rests(human_joints, demo_joints, fps)
    events, _ = drop_seated_contacts(detect_stance_events(human_joints, demo_joints, fps, contact_joints=probes), rests)
    offsets = (
        paired_sole_offsets(joint_surface_offsets(events))
        if source_offsets is None
        else {str(name): float(value) for name, value in source_offsets.items()}
    )

    seated = np.zeros(len(human_joints), dtype=bool)
    for r in rests:
        seated[r.start : r.end] = True

    ground = walkable_terrain(terrain)
    out = {}
    for side in ("l", "r"):
        flat = ~seated
        for joint in (j for j in probes if FOOT_LANDMARK_SIDES[j] == side):
            p = human_joints[:, demo_joints.index(joint)]
            surface = (
                np.zeros(len(p)) if ground is None else np.asarray(ground.height_at(p[:, 0], p[:, 1]), dtype=float)
            )
            flat &= np.abs(p[:, 2] - offsets.get(joint, 0.0) - surface) < tol
        out[side] = np.flatnonzero(flat)
    return out


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the shortest rotation vector between two unit vectors.

    Args:
        a: Initial unit vector.
        b: Target unit vector.

    Returns:
        Axis-angle rotation vector that maps ``a`` to ``b``.
    """
    axis = np.cross(a, b)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:  # parallel, or antiparallel and the axis is not defined
        return np.zeros(3)
    return axis / norm * float(np.arctan2(norm, float(np.dot(a, b))))


def _surface_into(
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    side: str,
    frames: np.ndarray,
    terrain=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the inward support normal beneath one foot.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        side: Foot side, ``"l"`` or ``"r"``.
        frames: Frame indices to evaluate.
        terrain: Source terrain, or ``None`` for flat ground.

    Returns:
        Inward unit normals with shape ``(len(frames), 3)`` and a Boolean mask
        identifying frames whose landmarks share one support plane.
    """
    down = np.array([0.0, 0.0, -1.0])
    terrain = walkable_terrain(terrain)
    if terrain is None or terrain.is_flat or not len(frames):
        # `not len(frames)` is the case where this side is never flat-footed on the source,
        # and it has to be caught here: the loop below builds its result by stacking a list,
        # and stacking nothing raises rather than giving an empty array of the right shape.
        # The caller already handles an empty return by skipping the correction; it just
        # cannot survive the exception. Reachable on any terrain motion where one foot never
        # goes flat, and common on chairs, where a foot spends the sit tucked under itself.
        return (np.repeat(down[None], len(frames), axis=0).reshape(len(frames), 3), np.ones(len(frames), dtype=bool))
    demo_joints = list(demo_joints)
    cols = [demo_joints.index(j) for j, s in FOOT_LANDMARK_SIDES.items() if s == side and j in demo_joints]
    into, usable = [], []
    for t in frames:
        xy = human_joints[t][cols, :2]
        per_landmark = [terrain.support_normal_at(p[None]) for p in xy]
        into.append(-terrain.support_normal_at(xy))
        usable.append(all(float(n @ per_landmark[0]) > 1.0 - 1e-9 for n in per_landmark))
    return np.stack(into), np.array(usable, dtype=bool)


def robot_sole_offsets(model: mujoco.MjModel, joints_mapping: dict[str, str]) -> dict[str, float]:
    """Measure robot foot-body heights above the sole in the default pose.

    Args:
        model: Environment model containing a ``"floor"`` geom.
        joints_mapping: Mapping from SMPL joint names to robot body names.

    Returns:
        Sole offsets in meters keyed by mapped SMPL foot landmark. Returns an
        empty dictionary when the model has no floor geom.
    """
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        return {}

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    out = {}
    for smpl_joint, side in FOOT_LANDMARK_SIDES.items():
        body = joints_mapping.get(smpl_joint)
        if body is None:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        sole = [g for g, n in enumerate(names) if n.startswith(MYOFULLBODY_SOLE_GEOMS[side])]
        if body_id < 0 or not sole:
            continue
        lowest = min(mujoco.mj_geomDistance(model, data, g, floor, 5.0, None) for g in sole)
        out[smpl_joint] = float(data.xpos[body_id][2] - lowest)
    return out


def foot_sole_offsets(
    model: mujoco.MjModel,
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    joints_mapping: dict[str, str],
    logger: logging.Logger | None = None,
    human_offsets: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Calculate vertical target corrections from human and robot sole offsets.

    Args:
        model: Environment model containing the robot and floor.
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        fps: Source frame rate in frames per second.
        joints_mapping: Mapping from SMPL joint names to robot body names.
        logger: Optional logger for skipped or clamped estimates.
        human_offsets: Optional anatomical offsets measured on a flat source clip.

    Returns:
        Vertical corrections in meters keyed by SMPL foot landmark.
    """
    robot = robot_sole_offsets(model, joints_mapping)
    if not robot:
        return {}

    events = detect_stance_events(human_joints, demo_joints, fps, contact_joints=tuple(robot))
    # A foot parked while the pelvis is on a seat is not a footfall. Left in, its low
    # pointed-toe reading biases this joint's offset and so every frame's target.
    events, _ = drop_seated_contacts(events, detect_seat_rests(human_joints, demo_joints, fps))
    human = (
        paired_sole_offsets(joint_surface_offsets(events), logger)
        if human_offsets is None
        else {str(name): float(value) for name, value in human_offsets.items()}
    )

    out = {}
    for joint, r in robot.items():
        if joint not in human:
            if logger:
                logger.warning(f"Sole offset for {joint}: no source stance events, correction skipped")
            continue
        dz = r - human[joint]
        if abs(dz) > MAX_SOLE_OFFSET:
            if logger:
                logger.warning(
                    f"Sole offset for {joint} is {dz * 1000:.0f} mm, beyond the "
                    f"{MAX_SOLE_OFFSET * 1000:.0f} mm cap; clamped. The source stance for "
                    f"this joint was probably not flat-footed."
                )
            dz = float(np.clip(dz, -MAX_SOLE_OFFSET, MAX_SOLE_OFFSET))
        out[joint] = float(dz)
    return out


def foot_orientation_offsets(
    model: mujoco.MjModel,
    site_ids: np.ndarray,
    site_names: Sequence[str],
    targets: np.ndarray,
    human_joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    terrain=None,
    cap: float = MAX_FOOT_ORIENT_OFFSET,
    logger: logging.Logger | None = None,
    source_offsets: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Estimate rotations that align flat-foot targets with support normals.

    Args:
        model: Environment model used to read the default sole direction.
        site_ids: Site IDs indexing the target array.
        site_names: Site names in the same order as ``site_ids``.
        targets: Orientation targets with shape ``(T, K, 3, 3)``.
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the source joint dimension.
        fps: Source frame rate in frames per second.
        terrain: Source terrain, or ``None`` for flat ground.
        cap: Maximum accepted correction angle in radians.
        logger: Optional logger for skipped and accepted estimates.
        source_offsets: Optional ankle/toe heights calibrated on flat ground.

    Returns:
        Correction rotation matrices keyed by foot site name. Sites without a
        usable flat-stance estimate are omitted.
    """
    from scipy.spatial.transform import Rotation

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0  # a flat-footed pose; see `robot_sole_offsets`
    mujoco.mj_forward(model, data)

    flat = flat_stance_frames(human_joints, demo_joints, fps, terrain, source_offsets=source_offsets)
    n = min(len(human_joints), targets.shape[0])
    down = np.array([0.0, 0.0, -1.0])

    out = {}
    for i, site in enumerate(site_names):
        side = FOOT_MIMIC_SITES.get(site)
        if side is None:
            continue
        frames = flat[side][flat[side] < n]
        into, usable = _surface_into(human_joints, demo_joints, side, frames, terrain)
        frames, into = frames[usable], into[usable]
        if not len(frames):
            if logger:
                logger.warning(
                    f"Foot orientation offset for {site}: the source foot is never flat on "
                    "one surface, correction skipped"
                )
            continue

        # Where the sole faces, in site coordinates. The site frames ride their bodies, so
        # this is a fixed property of the model once read at a flat-footed pose.
        sole = data.site_xmat[site_ids[i]].reshape(3, 3).T @ down
        # Median rather than mean: a stance detected a frame or two into a heel strike
        # contributes an outlier, and the estimate is a constant either way.
        rotvec = np.median(
            np.stack([_rotation_between(sole, targets[t, i].T @ into[k]) for k, t in enumerate(frames)]), axis=0
        )
        angle = float(np.linalg.norm(rotvec))
        if angle > cap:
            if logger:
                logger.warning(
                    f"Foot orientation offset for {site} is {np.degrees(angle):.0f} deg, beyond "
                    f"the {np.degrees(cap):.0f} deg cap; discarded. The source frames it was "
                    "estimated from were probably not flat-footed."
                )
            continue
        out[site] = Rotation.from_rotvec(rotvec).as_matrix()
        if logger:
            logger.info(
                f"Foot orientation offset for {site}: {np.degrees(angle):.1f} deg over "
                f"{len(frames)} flat-footed frame(s)"
            )
    return out
