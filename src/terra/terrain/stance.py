"""Detect stance events and group them into terrain support levels."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


#: Joint names used as foot-contact probes.
DEFAULT_CONTACT_JOINTS = ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle")


#: Maximum probe-joint speed for a stance event, in meters per second.
DEFAULT_STANCE_SPEED = 0.30


#: Minimum number of frames in a stance event.
DEFAULT_MIN_STANCE_FRAMES = 5


#: Maximum height gap between contacts on the same support level, in meters.
DEFAULT_LEVEL_TOL = 0.04


#: Maximum gap for merging an under-evidenced support level, in meters.
WEAK_LEVEL_MERGE_GAP = 0.07


# Long repeated trials provide enough evidence to distinguish a physical support from a
# brief slow swing pause. This gate is deliberately inactive for ordinary single-pass
# clips, whose real stair/stone levels may be observed only once or twice.
REPEATED_SUPPORT_MIN_EVENTS = 100
REPEATED_SUPPORT_MIN_FRACTION = 0.10


@dataclass
class StanceEvent:
    """Describe an interval in which a contact joint rests on a surface.

    Attributes:
        joint: Name of the contact joint.
        start: Index of the first frame in the interval.
        end: Exclusive index of the final frame in the interval.
        z: Median joint height during the interval, in meters.
        xy: Horizontal joint positions during the interval.
    """

    joint: str
    start: int
    end: int  # exclusive
    z: float  # representative height, the median over the run
    xy: np.ndarray = field(repr=False)  # (n, 2) positions over the run

    @property
    def n_frames(self) -> int:
        """Return the number of frames in the stance interval."""
        return self.end - self.start


def detect_stance_events(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    contact_joints: Sequence[str] = DEFAULT_CONTACT_JOINTS,
    speed_ms: float = DEFAULT_STANCE_SPEED,
    min_frames: int = DEFAULT_MIN_STANCE_FRAMES,
    local_window_s: float = 0.3,
    local_tol: float = 0.05,
) -> list[StanceEvent]:
    """Detect intervals in which contact joints are locally stationary and low.

    Args:
        joints: World-space joint positions with shape ``(n_frames, n_joints, 3)``.
        demo_joints: Joint names corresponding to the second axis of ``joints``.
        fps: Motion frame rate in frames per second.
        contact_joints: Joint names to evaluate as contact probes.
        speed_ms: Maximum probe speed for a stance event, in meters per second.
        min_frames: Minimum number of frames in a stance event.
        local_window_s: Half-width of the local-height window in seconds.
        local_tol: Maximum height above the local minimum in meters.

    Returns:
        Detected stance events.
    """
    names = list(demo_joints)
    n_frames = joints.shape[0]
    half = max(int(local_window_s * fps), 1)
    out: list[StanceEvent] = []

    for joint in contact_joints:
        if joint not in names:
            logger.warning("Contact joint %r not in the source skeleton; skipped", joint)
            continue
        p = joints[:, names.index(joint)]
        speed = np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps
        speed = np.concatenate([speed[:1], speed])
        slow = speed < speed_ms

        i = 0
        while i < n_frames:
            if not slow[i]:
                i += 1
                continue
            j = i
            while j < n_frames and slow[j]:
                j += 1
            if j - i >= min_frames:
                z = float(np.median(p[i:j, 2]))
                lo = max(0, (i + j) // 2 - half)
                hi = min(n_frames, (i + j) // 2 + half)
                if z <= float(p[lo:hi, 2].min()) + local_tol:
                    out.append(StanceEvent(joint=joint, start=i, end=j, z=z, xy=p[i:j, :2].copy()))
            i = j

    return out


def _kinematic_interval_events(
    joints: np.ndarray,
    demo_joints: Sequence[str],
    fps: float,
    intervals_s: Mapping[str, Sequence[Sequence[float]]],
    contact_joints: Sequence[str] = DEFAULT_CONTACT_JOINTS,
) -> list[StanceEvent]:
    """Build validation events from intervals derived by a reconstruction method."""
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError("joints must have shape (T, J, 3)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    names = list(demo_joints)
    times = np.arange(len(joints), dtype=float) / fps
    out: list[StanceEvent] = []
    for joint in contact_joints:
        if joint not in names:
            logger.warning("Contact joint %r not in the source skeleton; skipped", joint)
            continue
        if joint not in intervals_s:
            raise ValueError(f"kinematic validation intervals are missing {joint}")
        p = joints[:, names.index(joint)]
        for interval in intervals_s[joint]:
            if len(interval) != 2:
                raise ValueError(f"kinematic {joint} interval must contain start/end seconds")
            start_s, end_s = map(float, interval)
            if not np.isfinite(start_s) or not np.isfinite(end_s) or start_s < 0 or end_s <= start_s:
                raise ValueError(f"invalid kinematic {joint} interval {interval}")
            selected = np.flatnonzero((times >= start_s) & (times < end_s))
            if not len(selected):
                continue
            start = int(selected[0])
            end = int(selected[-1] + 1)
            out.append(
                StanceEvent(
                    joint=joint,
                    start=start,
                    end=end,
                    z=float(np.median(p[start:end, 2])),
                    xy=p[start:end, :2].copy(),
                )
            )
    return out


def joint_surface_offsets(events: Sequence[StanceEvent], tol: float = DEFAULT_LEVEL_TOL) -> dict[str, float]:
    """Estimate each contact joint's height above its supporting surface.

    Args:
        events: Stance events from the motion.
        tol: Height gap above which contacts belong to different surfaces.

    Returns:
        Mapping from joint name to surface-relative height in meters.
    """
    offsets = {}
    for joint in {e.joint for e in events}:
        zs = sorted(e.z for e in events if e.joint == joint)
        lowest = [zs[0]]
        for z in zs[1:]:
            if z - lowest[-1] > tol:
                break
            lowest.append(z)
        offsets[joint] = float(np.mean(lowest))
    return offsets


def paired_sole_offsets(
    human: dict[str, float],
    logger: logging.Logger | None = None,
    max_disagreement: float = 0.05,
) -> dict[str, float]:
    """Reconcile paired left and right sole offsets to their lower estimate.

    Args:
        human: Surface-relative height for each source contact joint.
        logger: Logger used to report large left-right differences.
        max_disagreement: Difference above which a warning is emitted, in meters.

    Returns:
        A copy of the mapping with each complete left-right pair set to its minimum.
    """
    out = dict(human)
    for left, right in (("L_Ankle", "R_Ankle"), ("L_Toe", "R_Toe")):
        if left not in human or right not in human:
            continue
        lower = min(human[left], human[right])
        if logger and abs(human[left] - human[right]) > max_disagreement:
            higher = left if human[left] > human[right] else right
            logger.warning(
                f"Sole offset estimates differ by "
                f"{abs(human[left] - human[right]) * 1000:.0f} mm between sides "
                f"({left}={human[left] * 1000:.0f}, {right}={human[right] * 1000:.0f} mm); "
                f"{higher} never reached the lowest surface, so both take the lower."
            )
        out[left] = out[right] = lower
    return out


def cluster_levels(
    events: Sequence[StanceEvent],
    tol: float = DEFAULT_LEVEL_TOL,
    offsets: dict[str, float] | None = None,
    max_spread: float | None = None,
) -> list[list[StanceEvent]]:
    """Group stance events into distinct support-height levels.

    Args:
        events: Stance events to group.
        tol: Maximum adjacent height gap within one level, in meters.
        offsets: Surface-relative joint heights. Estimated from ``events`` when omitted.
        max_spread: Maximum total height range within one level, in meters.

    Returns:
        Event groups ordered from lowest to highest support level.
    """
    if not events:
        return []
    if offsets is None:
        offsets = joint_surface_offsets(events, tol)
    if max_spread is None:
        max_spread = 2 * tol

    key = lambda e: e.z - offsets.get(e.joint, 0.0)  # noqa: E731
    order = sorted(events, key=key)
    groups: list[list[StanceEvent]] = [[order[0]]]
    for e in order[1:]:
        if key(e) - key(groups[-1][-1]) > tol:
            groups.append([e])
        else:
            groups[-1].append(e)

    out: list[list[StanceEvent]] = []
    for group in groups:
        stack = [group]
        while stack:
            g = stack.pop()
            if len(g) < 2 or key(g[-1]) - key(g[0]) <= max_spread:
                out.append(g)
                continue
            gaps = [key(b) - key(a) for a, b in itertools.pairwise(g)]
            cut = int(np.argmax(gaps)) + 1
            stack.extend([g[cut:], g[:cut]])  # lower half last, so it pops first
    return out


def _level_height(level: Sequence[StanceEvent], offsets: dict[str, float]) -> tuple[float, dict]:
    """Estimate a support level's height above the floor.

    Args:
        level: Stance events assigned to one support surface.
        offsets: Surface-relative height for each contact joint.

    Returns:
        The estimated height in meters and diagnostics for the per-joint estimates.
    """
    per_joint = {}
    for joint in sorted({e.joint for e in level}):
        zs = [e.z for e in level if e.joint == joint]
        per_joint[joint] = float(np.mean(zs) - offsets.get(joint, 0.0))

    values = np.array(list(per_joint.values()))
    unreferenced = [j for j in per_joint if j not in offsets]
    return float(values.mean()), {
        "per_joint": per_joint,
        "spread": float(values.max() - values.min()) if len(values) > 1 else 0.0,
        "joints_without_ground_reference": unreferenced,
    }


def _accept_levels(
    heights: Sequence[float],
    levels: Sequence[Sequence[StanceEvent]],
    min_events: int,
    min_level_height: float,
    merge_gap: float = WEAK_LEVEL_MERGE_GAP,
) -> list[bool]:
    """Select support levels that should produce terrain geometry.

    Args:
        heights: Support-level heights ordered from lowest to highest.
        levels: Stance-event groups corresponding to ``heights``.
        min_events: Minimum number of events for unconditional acceptance.
        min_level_height: Height below which a level is treated as the floor.
        merge_gap: Maximum distance to an accepted level that explains a weak level.

    Returns:
        Boolean acceptance flags corresponding to the input levels.
    """
    counts = [len(level) for level in levels]
    raised_counts = [count for height, count in zip(heights, counts, strict=True) if height >= min_level_height]
    repeated_min = min_events
    repeated_trial = sum(counts) >= REPEATED_SUPPORT_MIN_EVENTS and bool(raised_counts)
    if repeated_trial:
        repeated_min = max(
            min_events,
            int(np.ceil(REPEATED_SUPPORT_MIN_FRACTION * max(raised_counts))),
        )
    accepted = [
        count >= repeated_min or height < min_level_height for height, count in zip(heights, counts, strict=True)
    ]
    for i, (h, ok) in enumerate(zip(heights, accepted, strict=True)):
        if ok:
            continue
        if repeated_trial:
            # An isolated novel height is evidence in a short traversal, but an outlier
            # in a trial that repeats every real support dozens of times.
            continue
        explained = any(j != i and accepted[j] and abs(h - heights[j]) <= merge_gap for j in range(len(heights)))
        accepted[i] = not explained
    return accepted
