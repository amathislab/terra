"""Select collision geoms, measure penetration, and repair short outliers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from terra._musclemimic import max_penetration_with_geoms
from terra.constants import RAISED_START_HEIGHT
from terra.defaults import DEFAULT_POSTHOC_MAX_RUN, DEFAULT_POSTHOC_PEN_THRESHOLD
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

CollisionMetric = Callable[[np.ndarray], float]


def collidable_geoms(model: mujoco.MjModel, exclude: int | None = None) -> list[int]:
    """Find geoms that participate in contact.

    Args:
        model: MuJoCo model to inspect.
        exclude: Optional geom ID to omit.

    Returns:
        Geom IDs with a nonzero contact type or affinity.
    """
    return [g for g in range(model.ngeom) if g != exclude and (model.geom_contype[g] or model.geom_conaffinity[g])]


def floor_clearance(
    model: mujoco.MjModel, data: mujoco.MjData, qpos_frame, floor_id: int, geom_ids, distmax: float = 10.0
) -> float:
    """Measure the minimum signed distance between robot geoms and the floor.

    Args:
        model: MuJoCo model containing the robot and floor.
        data: MuJoCo data whose generalized positions are overwritten.
        qpos_frame: Generalized positions for the frame to evaluate.
        floor_id: Floor geom ID.
        geom_ids: Robot geom IDs to measure.
        distmax: Maximum distance evaluated by ``mj_geomDistance``.

    Returns:
        Minimum signed distance in meters. Negative values indicate penetration.
    """
    data.qpos[:] = qpos_frame
    mujoco.mj_forward(model, data)
    return min(float(mujoco.mj_geomDistance(model, data, g, floor_id, distmax, None)) for g in geom_ids)


def self_collision_geom_pairs(
    model: mujoco.MjModel,
    body_pairs: Sequence[tuple[str, str]],
) -> list[tuple[int, int, str]]:
    """Resolve body pairs to their Cartesian product of collision geoms.

    Args:
        model: MuJoCo model containing the named bodies.
        body_pairs: Pairs of body names to test for self-collision.

    Returns:
        Tuples containing two geom IDs and a body-pair label.

    Raises:
        ValueError: If either body in a pair has no collision geoms.
    """
    by_body: dict[str, list[int]] = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        by_body.setdefault(body, []).append(g)

    pairs: list[tuple[int, int, str]] = []
    for a, b in body_pairs:
        ga, gb = by_body.get(a, ()), by_body.get(b, ())
        missing = [name for name, geoms in ((a, ga), (b, gb)) if not geoms]
        if missing:
            raise ValueError(
                f"Self-collision pair ({a!r}, {b!r}): no collision geoms on {', '.join(repr(name) for name in missing)}"
            )
        pairs.extend((x, y, f"{a}/{b}") for x in ga for y in gb)
    return pairs


@dataclass
class _CollisionEvaluator:
    """Evaluate terrain and selected self-collision depths with shared MuJoCo state."""

    model: mujoco.MjModel
    data: mujoco.MjData
    environment_geoms: tuple[int, ...]
    self_pairs: list[tuple[int, int, str]]

    def _forward(self, qpos: np.ndarray) -> None:
        """Run forward kinematics for one generalized-position vector."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def terrain_penetration(self, qpos: np.ndarray) -> float:
        """Return positive terrain-penetration depth for one pose."""
        self._forward(qpos)
        return max(0.0, -max_penetration_with_geoms(self.model, self.data, self.environment_geoms))

    def self_penetration(self, qpos: np.ndarray) -> float:
        """Return positive selected self-penetration depth for one pose."""
        if not self.self_pairs:
            return 0.0
        self._forward(qpos)
        pair_depth = max(
            -float(mujoco.mj_geomDistance(self.model, self.data, first, second, 0.05, None))
            for first, second, _label in self.self_pairs
        )
        return max(0.0, pair_depth)


def _validated_collision_repair_inputs(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    threshold: float,
    max_run: int,
) -> tuple[np.ndarray, float, int]:
    """Validate a collision-repair trajectory and its numerical limits."""
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape (T, {model.nq}), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos must contain only finite values")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(f"collision repair threshold must be finite and non-negative, got {threshold!r}") from error
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"collision repair threshold must be finite and non-negative, got {threshold!r}")
    if isinstance(max_run, (bool, np.bool_)):
        raise ValueError(f"max_run must be a non-negative integer, got {max_run!r}")
    try:
        resolved_max_run = int(max_run)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"max_run must be a non-negative integer, got {max_run!r}") from error
    if resolved_max_run != max_run or resolved_max_run < 0:
        raise ValueError(f"max_run must be a non-negative integer, got {max_run!r}")
    return qpos, threshold, resolved_max_run


def _validated_environment_geoms(model: mujoco.MjModel, geom_ids: Sequence[int]) -> tuple[int, ...]:
    """Resolve and range-check environment geom IDs."""
    resolved = tuple(int(geom_id) for geom_id in geom_ids)
    invalid = [geom_id for geom_id in resolved if not 0 <= geom_id < model.ngeom]
    if invalid:
        raise ValueError(f"environment geom IDs are outside [0, {model.ngeom}): {invalid}")
    return resolved


def _quaternion_qpos_slices(model: mujoco.MjModel) -> tuple[slice, ...]:
    """Return every free- or ball-joint quaternion slice in qpos order."""
    slices = []
    for joint in range(model.njnt):
        joint_type = int(model.jnt_type[joint])
        qpos_address = int(model.jnt_qposadr[joint])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            slices.append(slice(qpos_address + 3, qpos_address + 7))
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            slices.append(slice(qpos_address, qpos_address + 4))
    return tuple(slices)


def _interpolate_pose(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float,
    quaternion_slices: Sequence[slice],
) -> np.ndarray:
    """Interpolate a pose while respecting every quaternion hemisphere."""
    candidate = (1.0 - alpha) * before + alpha * after
    for quaternion_slice in quaternion_slices:
        # Equivalent quaternions can have opposite signs. Choose one hemisphere before
        # normalized lerp so a midpoint cannot collapse towards the zero quaternion.
        before_quaternion = before[quaternion_slice]
        after_quaternion = after[quaternion_slice]
        if np.dot(before_quaternion, after_quaternion) < 0.0:
            after_quaternion = -after_quaternion
        quaternion = (1.0 - alpha) * before_quaternion + alpha * after_quaternion
        candidate[quaternion_slice] = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    return candidate


def _collision_runs(depth: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Return half-open intervals whose penetration exceeds the threshold."""
    bad = depth > threshold
    edges = np.flatnonzero(np.diff(np.r_[False, bad, False].astype(int)) != 0).reshape(-1, 2)
    return [(int(start), int(end)) for start, end in edges]


def _run_is_eligible(
    start: int,
    end: int,
    depth: np.ndarray,
    threshold: float,
    max_run: int,
) -> bool:
    """Check that a short run is bounded by valid sampled poses."""
    if end - start > max_run or start == 0 or end == len(depth):
        return False
    return bool(depth[start - 1] <= threshold and depth[end] <= threshold)


def _interpolated_run(
    qpos: np.ndarray,
    start: int,
    end: int,
    quaternion_slices: Sequence[slice],
) -> list[np.ndarray]:
    """Interpolate all poses in one open interval between fixed neighbors."""
    before, after = qpos[start - 1], qpos[end]
    return [
        _interpolate_pose(
            before,
            after,
            (frame - start + 1) / (end - start + 1),
            quaternion_slices,
        )
        for frame in range(start, end)
    ]


def _candidate_is_safe(
    candidate: np.ndarray,
    original: np.ndarray,
    primary: CollisionMetric,
    secondary: CollisionMetric,
    threshold: float,
) -> bool:
    """Require primary clearance without deepening an existing secondary violation."""
    return bool(primary(candidate) <= threshold and secondary(candidate) <= max(threshold, secondary(original)) + 1e-9)


def _phase_shifted_candidate(
    before: np.ndarray,
    after: np.ndarray,
    original: np.ndarray,
    quaternion_slices: Sequence[slice],
    primary: CollisionMetric,
    secondary: CollisionMetric,
    threshold: float,
) -> np.ndarray | None:
    """Find the nearest collision-safe temporal phase for one recreated frame."""
    nominal_phase = 0.5
    phases = sorted(
        np.linspace(0.0, 1.0, 41),
        key=lambda phase: (abs(float(phase) - nominal_phase), float(phase)),
    )
    for phase in phases:
        candidate = _interpolate_pose(before, after, float(phase), quaternion_slices)
        if _candidate_is_safe(candidate, original, primary, secondary, threshold):
            return candidate
    return None


def _repair_run_candidates(
    qpos: np.ndarray,
    start: int,
    end: int,
    quaternion_slices: Sequence[slice],
    primary: CollisionMetric,
    secondary: CollisionMetric,
    threshold: float,
) -> list[np.ndarray] | None:
    """Build a safe straight interpolation or single-frame phase fallback."""
    candidates = _interpolated_run(qpos, start, end, quaternion_slices)
    if all(
        _candidate_is_safe(candidate, qpos[frame], primary, secondary, threshold)
        for frame, candidate in zip(range(start, end), candidates, strict=True)
    ):
        return candidates

    # Collision geometry moves nonlinearly between two valid sampled poses. A stair-edge
    # scrape can occur at the temporal midpoint even when neighboring samples are valid.
    # The fallback is a bounded time warp of at most one native interval, not a new IK
    # solution, and preserves both neighboring TERRA poses.
    if end - start != 1:
        return None
    candidate = _phase_shifted_candidate(
        qpos[start - 1],
        qpos[end],
        qpos[start],
        quaternion_slices,
        primary,
        secondary,
        threshold,
    )
    return None if candidate is None else [candidate]


def _repair_collision_runs(
    qpos: np.ndarray,
    repaired: set[int],
    primary: CollisionMetric,
    secondary: CollisionMetric,
    threshold: float,
    max_run: int,
    quaternion_slices: Sequence[slice],
) -> None:
    """Repair eligible runs identified by one collision metric in place."""
    depth = np.array([primary(pose) for pose in qpos])
    for start, end in _collision_runs(depth, threshold):
        if not _run_is_eligible(start, end, depth, threshold, max_run):
            continue
        candidates = _repair_run_candidates(
            qpos,
            start,
            end,
            quaternion_slices,
            primary,
            secondary,
            threshold,
        )
        if candidates is None:
            continue
        qpos[start:end] = candidates
        repaired.update(range(start, end))


def repair_isolated_terrain_penetrations(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    env_geom_ids: Sequence[int],
    threshold: float = DEFAULT_POSTHOC_PEN_THRESHOLD,
    max_run: int = DEFAULT_POSTHOC_MAX_RUN,
    self_collision_body_pairs: Sequence[tuple[str, str]] = (),
) -> tuple[np.ndarray, list[int]]:
    """Repair short collision runs by interpolating between valid poses.

    Args:
        model: MuJoCo model used to validate candidate poses.
        qpos: Generalized positions with shape ``(T, nq)``.
        env_geom_ids: Static environment geom IDs.
        threshold: Maximum accepted penetration depth in meters.
        max_run: Maximum number of consecutive frames to interpolate. Zero disables repair.
        self_collision_body_pairs: Optional body pairs to validate and repair.

    Returns:
        A repaired trajectory copy and the sorted indices of changed frames.

    Raises:
        ValueError: If the trajectory, collision threshold, run limit, or geom IDs are invalid.
    """
    qpos, threshold, max_run = _validated_collision_repair_inputs(model, qpos, threshold, max_run)
    environment_geoms = _validated_environment_geoms(model, env_geom_ids)
    self_pairs = self_collision_geom_pairs(model, self_collision_body_pairs)
    if len(qpos) < 3 or max_run == 0:
        return qpos.copy(), []

    evaluator = _CollisionEvaluator(
        model=model,
        data=mujoco.MjData(model),
        environment_geoms=environment_geoms,
        self_pairs=self_pairs,
    )
    quaternion_slices = _quaternion_qpos_slices(model)
    out = qpos.copy()
    repaired: set[int] = set()
    _repair_collision_runs(
        out,
        repaired,
        evaluator.terrain_penetration,
        evaluator.self_penetration,
        threshold,
        max_run,
        quaternion_slices,
    )
    if evaluator.self_pairs:
        _repair_collision_runs(
            out,
            repaired,
            evaluator.self_penetration,
            evaluator.terrain_penetration,
            threshold,
            max_run,
            quaternion_slices,
        )
    return out, sorted(repaired)


def walkable_terrain(terrain):
    """Return terrain surfaces on which a foot may stand.

    Args:
        terrain: Terrain specification, or ``None`` for flat ground.

    Returns:
        The terrain's walkable subset, or ``None``.
    """
    return None if terrain is None else terrain.walkable


def starts_on_raised_terrain(human_joints: np.ndarray, demo_joints, terrain) -> bool:
    """Check whether either foot starts above a raised support surface.

    Args:
        human_joints: Source joints with shape ``(T, J, 3)``.
        demo_joints: Names indexing the joint dimension.
        terrain: Resolved terrain specification.

    Returns:
        ``True`` when an opening toe or ankle is over terrain higher than
        ``RAISED_START_HEIGHT``.
    """
    names = list(demo_joints)
    probes = [joint for joint in DEFAULT_CONTACT_JOINTS if joint in names]
    if not probes:
        return False
    p = np.array([human_joints[0, names.index(j), :2] for j in probes])
    ground = walkable_terrain(terrain)
    return bool(np.max(ground.height_at(p[:, 0], p[:, 1])) > RAISED_START_HEIGHT)
