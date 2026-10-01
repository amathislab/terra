"""Provide geometric helpers for fitting terrain boxes."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from terra._musclemimic import BoxSpec

#: Allowed body penetration below a fitted surface, in meters.
DEFAULT_FREE_SPACE_TOL = 0.03


#: Maximum motion-derived support residual accepted by the physical validator, in meters.
DEFAULT_CONTACT_TOL = 0.05


#: Contact-patch margin as ``(longitudinal, lateral)`` distances in meters.
DEFAULT_CONTACT_MARGIN = (0.10, 0.05)


#: Generic completion prior beyond the contact-derived footprint as
#: ``(longitudinal, lateral)`` meters.  A straight sequence of foot contacts cannot
#: observe a surface's lateral boundary; retaining a modest, dataset-independent prior
#: prevents platforms and other continuous supports from collapsing to sole-width pads.
#: Longitudinal growth remains evidence/free-space bounded.
DEFAULT_MAX_EXTENSION = (0.0, 0.15)


#: Gap above which contact patches are split into separate boxes, in meters.
DEFAULT_SPLIT_GAP = 0.60


#: Split gap used for support levels with inconsistent contact heights, in meters.
DEFAULT_AMBIGUOUS_SPLIT_GAP = 0.30


#: Whether fitted boxes exclude contacts assigned to lower surfaces.
DEFAULT_EXCLUDE_CLAIMED = True


#: Maximum height of a box eligible for conflict-based rejection, in meters.
CONFLICT_REJECT_MAX_HEIGHT = 0.12


#: Minimum conflicting-frame ratio used to reject a low box.
CONFLICT_REJECT_MIN_RATIO = 0.3


#: Samples per axis used to test whether one box's top face is buried under others.
BURIED_FACE_SAMPLES = 25


def _as_pair(value, name: str) -> tuple[float, float]:
    """Convert a scalar or two-value sequence to a float pair.

    Args:
        value: Scalar or sequence to convert.
        name: Parameter name used in validation errors.

    Returns:
        A pair of floats. Scalar values are repeated for both entries.

    Raises:
        ValueError: If a sequence does not contain exactly two values.
    """
    if np.isscalar(value):
        return (float(value), float(value))
    pair = tuple(float(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be a scalar or a (longitudinal, lateral) pair, got {value!r}")
    return pair


def _principal_yaw(xy: np.ndarray) -> float:
    """Compute the yaw of the dominant axis through horizontal points.

    Args:
        xy: Horizontal points with shape ``(n_points, 2)``.

    Returns:
        Dominant-axis yaw in radians, or zero when no axis can be determined.
    """
    if len(xy) < 2:
        return 0.0
    centred = xy - xy.mean(axis=0)
    if not np.any(np.abs(centred) > 1e-9):
        return 0.0
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return float(np.arctan2(vt[0, 1], vt[0, 0]))


def _split_by_gap(u: np.ndarray, gap: float) -> list[np.ndarray]:
    """Split positions into groups separated by a specified gap.

    Args:
        u: Positions along the level's major axis.
        gap: Separation above which adjacent positions form different groups.

    Returns:
        Arrays of indices into ``u``, one per group.
    """
    order = np.argsort(u)
    groups, current = [], [order[0]]
    for prev, cur in itertools.pairwise(order):
        if u[cur] - u[prev] > gap:
            groups.append(np.array(current))
            current = [cur]
        else:
            current.append(cur)
    groups.append(np.array(current))
    return groups


def _free_space_limit(
    body_points: np.ndarray,
    yaw: float,
    centre_xy: np.ndarray,
    v_half: float,
    top: float,
    tol: float,
    support_lo: float,
    support_hi: float,
    clearance: float,
) -> tuple[float, float]:
    """Find the box limits imposed by surrounding free space.

    Args:
        body_points: Joint positions flattened across frames, with shape ``(n_points, 3)``.
        yaw: Yaw of the box's major axis in radians.
        centre_xy: Horizontal center of the supporting contacts.
        v_half: Lateral half-width of the box in meters.
        top: Height of the box's top face in meters.
        tol: Required depth below the top face for a point to block the box.
        support_lo: Lower bound of the required support span along the major axis.
        support_hi: Upper bound of the required support span along the major axis.
        clearance: Margin between a box face and a blocking point in meters.

    Returns:
        Lower and upper limits along the major axis. An unblocked side is infinite.
    """
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = body_points[:, :2] - centre_xy
    u = c * d[:, 0] - s * d[:, 1]
    v = s * d[:, 0] + c * d[:, 1]

    blocking = (np.abs(v) <= v_half) & (body_points[:, 2] < top - tol)
    lo_limit, hi_limit = -np.inf, np.inf
    if np.any(blocking & (u > support_hi)):
        hi_limit = float(u[blocking & (u > support_hi)].min()) - clearance
    if np.any(blocking & (u < support_lo)):
        lo_limit = float(u[blocking & (u < support_lo)].max()) + clearance
    return lo_limit, hi_limit


def _exclude_claimed_points(
    extent: tuple[float, float, float, float],
    required: tuple[float, float, float, float],
    pu: np.ndarray,
    pv: np.ndarray,
    clearance: float,
) -> tuple[tuple[float, float, float, float], int]:
    """Shrink a box to exclude contacts assigned to lower surfaces.

    Args:
        extent: Current ``(u_lo, u_hi, v_lo, v_hi)``.
        required: Minimum extent required by the box's supporting contacts.
        pu: Major-axis coordinates of contacts to exclude.
        pv: Lateral-axis coordinates of contacts to exclude.
        clearance: Margin between a moved face and an excluded point.

    Returns:
        The adjusted extent and the number of contacts that could not be excluded.
    """
    u_lo, u_hi, v_lo, v_hi = extent
    ru_lo, ru_hi, rv_lo, rv_hi = required

    # Outermost point first: those cuts are the largest and land furthest from the
    # footfalls justifying this box, so they are least likely to be undone by a later one.
    unresolved = 0
    for i in np.argsort(-(np.abs(pu) + np.abs(pv))):
        a, b = float(pu[i]), float(pv[i])
        if not (u_lo < a < u_hi and v_lo < b < v_hi):
            continue  # outside already, possibly thanks to an earlier cut

        # (slack past the required span, face to move, where to move it). Slack measures
        # what the cut costs: the further beyond support the point sits, the less surface
        # is given up to exclude it. `max` therefore picks the cheapest cut.
        cuts = []
        if a >= ru_hi:
            cuts.append((a - ru_hi, "u_hi", a - clearance))
        if a <= ru_lo:
            cuts.append((ru_lo - a, "u_lo", a + clearance))
        if b >= rv_hi:
            cuts.append((b - rv_hi, "v_hi", b - clearance))
        if b <= rv_lo:
            cuts.append((rv_lo - b, "v_lo", b + clearance))
        if not cuts:
            unresolved += 1
            continue

        _, face, value = max(cuts)
        if face == "u_hi":
            u_hi = min(u_hi, max(value, ru_hi))
        elif face == "u_lo":
            u_lo = max(u_lo, min(value, ru_lo))
        elif face == "v_hi":
            v_hi = min(v_hi, max(value, rv_hi))
        else:
            v_lo = max(v_lo, min(value, rv_lo))

    return (u_lo, u_hi, v_lo, v_hi), unresolved


def _drop_buried_boxes(boxes: Sequence[BoxSpec]) -> tuple[list[BoxSpec], list[tuple[str, str]]]:
    """Remove boxes whose top face is completely covered by taller ones.

    Args:
        boxes: Fitted terrain boxes in build order.

    Returns:
        Retained boxes and pairs naming each removed box and its covering boxes.
    """
    kept, dropped = [], []
    for i, b in enumerate(boxes):
        # Sample the top face itself rather than a footprint reconstructed from the yaw: a
        # pitched box's face is longer than its ground projection by 1/cos(pitch), so a
        # grid built in world xy from `size[0]` would test points the face does not reach.
        face = b.top_face_points(2 * max(b.size[0], b.size[1]) / (BURIED_FACE_SAMPLES - 1))
        x, y = face[:, 0], face[:, 1]

        covered = np.zeros(x.shape, dtype=bool)
        over = []
        for j, o in enumerate(boxes):
            if j == i:
                continue
            # Buried means another box's surface is above this face *here*, which for a
            # sloped box varies over the face and cannot be settled by the scalar tops.
            hit = o.contains_xy(x, y) & (o.top_at(x, y) > b.top_at(x, y) + 1e-9)
            if hit.any():
                over.append(o.name or f"box {j}")
            covered |= hit
        if covered.all() and over:
            dropped.append((b.name or f"box {i}", ", ".join(over)))
        else:
            kept.append(b)
    return kept, dropped
