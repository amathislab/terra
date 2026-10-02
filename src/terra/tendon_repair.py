"""Detect and repair isolated musculotendon discontinuities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

CandidateValidator = Callable[[np.ndarray, range], bool]


def tendon_lengths(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Evaluate all tendon lengths along a trajectory.

    Args:
        model: MuJoCo model containing the tendons.
        qpos: Generalized positions with shape ``(T, nq)``.

    Returns:
        Tendon lengths with shape ``(T, ntendon)``.
    """
    data = mujoco.MjData(model)
    values = np.empty((len(qpos), model.ntendon), dtype=np.float64)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        values[frame] = data.ten_length
    return values


def adaptive_tendon_events(
    model: mujoco.MjModel,
    lengths: np.ndarray,
    *,
    threshold: float = 0.05,
    jump_factor: float = 10.0,
    min_rel_jump: float = 1e-3,
    ema_alpha: float = 0.01,
) -> list[tuple[float, int, int]]:
    """Detect relative tendon-length jumps with an adaptive threshold.

    Args:
        model: MuJoCo model containing tendon reference lengths.
        lengths: Tendon lengths with shape ``(T, ntendon)``.
        threshold: Minimum reported relative jump.
        jump_factor: Multiplier applied to the exponential moving average.
        min_rel_jump: Minimum adaptive detector threshold.
        ema_alpha: Exponential moving-average coefficient.

    Returns:
        Descending ``(relative_jump, frame, tendon_id)`` event tuples.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    if len(lengths) < 2 or model.ntendon == 0:
        return []
    ema = np.zeros(model.ntendon, dtype=np.float64)
    scale = np.maximum(np.asarray(model.tendon_length0, dtype=np.float64), 1e-6)
    events = []
    for frame in range(1, len(lengths)):
        relative = np.abs(lengths[frame] - lengths[frame - 1]) / scale
        ema = relative.copy() if frame == 1 else (1.0 - ema_alpha) * ema + ema_alpha * relative
        flagged = relative > np.maximum(jump_factor * ema, min_rel_jump)
        if np.any(flagged):
            tendon = int(np.argmax(np.where(flagged, relative, -np.inf)))
            jump = float(relative[tendon])
            if jump > threshold:
                events.append((jump, frame, tendon))
    return sorted(events, reverse=True)


def _repair_events(
    model: mujoco.MjModel,
    lengths: np.ndarray,
    threshold: float,
    tracked: set[tuple[int, int]],
) -> tuple[list[tuple[float, int, int]], set[tuple[int, int]]]:
    """Update detected events while retaining unresolved tracked transitions.

    Args:
        model: MuJoCo model containing tendon reference lengths.
        lengths: Tendon lengths with shape ``(T, ntendon)``.
        threshold: Maximum accepted relative jump.
        tracked: Previously detected ``(frame, tendon_id)`` pairs.

    Returns:
        Current descending event tuples and the updated tracked set.
    """
    detected = adaptive_tendon_events(model, lengths, threshold=threshold)
    tracked = tracked | {(frame, tendon) for _jump, frame, tendon in detected}
    scale = np.maximum(np.asarray(model.tendon_length0, dtype=np.float64), 1e-6)
    events = {(frame, tendon): jump for jump, frame, tendon in detected}
    for frame, tendon in tracked:
        if not 0 < frame < len(lengths):
            continue
        jump = float(abs(lengths[frame, tendon] - lengths[frame - 1, tendon]) / scale[tendon])
        if jump > threshold:
            events[(frame, tendon)] = max(jump, events.get((frame, tendon), 0.0))
    return sorted([(jump, frame, tendon) for (frame, tendon), jump in events.items()], reverse=True), tracked


def _interpolate_span(
    qpos: np.ndarray,
    left: int,
    right: int,
    *,
    preserve_prefix: int = 0,
    coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate generalized positions within an open frame interval.

    Args:
        qpos: Generalized positions with shape ``(T, nq)``.
        left: Left endpoint, which remains unchanged.
        right: Right endpoint, which remains unchanged.
        preserve_prefix: Leading coordinates preserved during full-pose interpolation.
        coordinates: Optional coordinate indices to interpolate exclusively.

    Returns:
        A copy of ``qpos`` with the selected interval interpolated.
    """
    out = qpos.copy()
    qa, qb = qpos[left, 3:7].copy(), qpos[right, 3:7].copy()
    if np.dot(qa, qb) < 0.0:
        qb = -qb
    for frame in range(left + 1, right):
        alpha = (frame - left) / (right - left)
        interpolated = (1.0 - alpha) * qpos[left] + alpha * qpos[right]
        if coordinates is not None:
            out[frame, coordinates] = interpolated[coordinates]
        else:
            out[frame] = interpolated
        if coordinates is None and preserve_prefix:
            out[frame, :preserve_prefix] = qpos[frame, :preserve_prefix]
        elif coordinates is None and qpos.shape[1] >= 7:
            quat = (1.0 - alpha) * qa + alpha * qb
            out[frame, 3:7] = quat / max(np.linalg.norm(quat), 1e-12)
    return out


def _tendon_qpos_dependencies(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, tendon: int) -> np.ndarray:
    """Find generalized-position coordinates that affect one tendon.

    Args:
        model: MuJoCo model containing the tendon.
        data: Reusable MuJoCo data.
        q: Generalized positions at which to evaluate the tendon Jacobian.
        tendon: Tendon ID.

    Returns:
        Sorted generalized-position indices with nonzero tendon derivatives.
    """
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    # MuJoCo 3.11 moved the sparse tendon-Jacobian layout arrays from MjData to
    # MjModel.  Accept both layouts so trajectory repair remains compatible with
    # caches and environments built against either API generation.
    jacobian_layout = data if hasattr(data, "ten_J_rownnz") else model
    row_nnz = int(jacobian_layout.ten_J_rownnz[tendon])
    if row_nnz:
        start = int(jacobian_layout.ten_J_rowadr[tendon])
        stop = start + row_nnz
        values = data.ten_J.reshape(-1)[start:stop]
        dofs = jacobian_layout.ten_J_colind.reshape(-1)[start:stop]
    else:
        values = np.asarray(data.ten_J[tendon])
        dofs = np.arange(model.nv)

    qpos_indices: set[int] = set()
    for dof in np.asarray(dofs)[np.abs(values) > 1e-9]:
        joint = int(model.dof_jntid[int(dof)])
        joint_type = int(model.jnt_type[joint])
        qadr = int(model.jnt_qposadr[joint])
        if joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            qpos_indices.add(qadr)
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            # MyoFullBody has no ball joints, but preserving the whole quaternion is the
            # only coherent fallback for direct users that do.
            qpos_indices.update(range(qadr, qadr + 4))
    return np.array(sorted(qpos_indices), dtype=int)


def _coupled_coordinate_closure(
    model: mujoco.MjModel, coordinates: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Close a coordinate set over active polynomial joint couplers.

    Args:
        model: MuJoCo model containing joint equalities.
        coordinates: Initial generalized-position indices.

    Returns:
        Expanded coordinate indices and the affected coupler rows.
    """
    selected = {int(i) for i in coordinates}
    all_rows = []
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[eq]:
            continue
        dep_joint, indep_joint = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        if indep_joint < 0:
            continue
        all_rows.append(
            (
                int(model.jnt_qposadr[dep_joint]),
                int(model.jnt_qposadr[indep_joint]),
                np.asarray(model.eq_data[eq][:5], dtype=float),
            )
        )

    changed = True
    while changed:
        changed = False
        for dep, indep, _poly in all_rows:
            if dep in selected or indep in selected:
                before = len(selected)
                selected.update((dep, indep))
                changed |= len(selected) != before
    rows = [(dep, indep, poly) for dep, indep, poly in all_rows if dep in selected or indep in selected]
    return np.array(sorted(selected), dtype=int), rows


def _project_couplers(qpos: np.ndarray, frames: range, rows: list[tuple[int, int, np.ndarray]]) -> None:
    """Project selected poses onto polynomial joint equalities in place.

    Args:
        qpos: Generalized-position trajectory to update.
        frames: Frame indices to project.
        rows: Dependent index, independent index, and polynomial coefficients.
    """
    for frame in frames:
        for dep, indep, poly in rows:
            value = qpos[frame, indep]
            qpos[frame, dep] = sum(poly[k] * value**k for k in range(5))


@dataclass(frozen=True)
class _RepairScope:
    """Coordinate subset and maximum span used by one repair search."""

    coordinates: np.ndarray | None
    coupler_rows: list[tuple[int, int, np.ndarray]]
    max_changed_frames: int


@dataclass(frozen=True)
class _RepairCandidate:
    """Accepted repair candidate and its updated event bookkeeping."""

    score: tuple[int, float]
    qpos: np.ndarray
    lengths: np.ndarray
    frames: range
    tracked: set[tuple[int, int]]


@dataclass(frozen=True)
class _RepairSearch:
    """Immutable state shared by every candidate for one tendon event."""

    model: mujoco.MjModel
    qpos: np.ndarray
    lengths: np.ndarray
    tracked: set[tuple[int, int]]
    preserve_prefix: int
    validator: CandidateValidator | None
    event_frame: int
    event_tendon: int
    length_scale: float
    threshold: float
    before_score: tuple[int, float]


def _validate_repair_limits(
    threshold: float,
    max_changed_frames: int,
    max_local_changed_frames: int | None,
) -> int:
    """Validate public limits and return the resolved tendon-local limit."""
    if threshold <= 0.0 or not np.isfinite(threshold):
        raise ValueError(f"tendon repair threshold must be positive and finite, got {threshold!r}")
    if max_changed_frames < 0:
        raise ValueError(f"max_changed_frames must be non-negative, got {max_changed_frames}")
    if max_local_changed_frames is None:
        max_local_changed_frames = max_changed_frames
    if max_local_changed_frames < 0:
        raise ValueError(f"max_local_changed_frames must be non-negative, got {max_local_changed_frames}")
    return max_local_changed_frames


def _preserved_qpos_prefix(model: mujoco.MjModel) -> int:
    """Return the leading free-root coordinates that repair must preserve."""
    if (
        getattr(model, "njnt", 0)
        and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
        and int(model.jnt_qposadr[0]) == 0
    ):
        return 7
    return 0


def _repair_scopes(
    model: mujoco.MjModel,
    dependency_data: mujoco.MjData | None,
    qpos: np.ndarray,
    event_frame: int,
    event_tendon: int,
    max_local_changed_frames: int,
    max_changed_frames: int,
) -> list[_RepairScope]:
    """Build the preferred tendon-local scope and whole-pose fallback."""
    coordinates = None
    if dependency_data is not None:
        # Use both sides of the transition: a wrap-topology change can make a coordinate
        # disappear from the tendon linearisation on only one side.
        coordinates = np.union1d(
            _tendon_qpos_dependencies(
                model,
                dependency_data,
                qpos[event_frame - 1],
                event_tendon,
            ),
            _tendon_qpos_dependencies(
                model,
                dependency_data,
                qpos[event_frame],
                event_tendon,
            ),
        )
        if not len(coordinates):
            coordinates = None

    coupler_rows = []
    if coordinates is not None:
        coordinates, coupler_rows = _coupled_coordinate_closure(model, coordinates)
    scopes = [
        _RepairScope(
            coordinates=coordinates,
            coupler_rows=coupler_rows,
            max_changed_frames=max_local_changed_frames,
        )
    ]
    if coordinates is not None:
        scopes.append(
            _RepairScope(
                coordinates=None,
                coupler_rows=[],
                max_changed_frames=max_changed_frames,
            )
        )
    return scopes


def _candidate_spans(
    event_frame: int,
    changed_frames: int,
    trajectory_frames: int,
) -> list[tuple[int, int]]:
    """Enumerate endpoint pairs by center, then by preceding-frame count."""
    spans = []
    for centre in (event_frame, event_frame - 1):
        for pre in range(1, changed_frames + 1):
            post = changed_frames - pre + 1
            left, right = centre - pre, centre + post
            if left >= 0 and right < trajectory_frames:
                spans.append((left, right))
    return spans


def _evaluate_candidate(
    search: _RepairSearch,
    scope: _RepairScope,
    left: int,
    right: int,
) -> _RepairCandidate | None:
    """Interpolate, validate, and score one candidate span."""
    candidate = _interpolate_span(
        search.qpos,
        left,
        right,
        preserve_prefix=search.preserve_prefix,
        coordinates=scope.coordinates,
    )
    frames = range(left + 1, right)
    _project_couplers(candidate, frames, scope.coupler_rows)
    if search.validator is not None and not search.validator(candidate, frames):
        return None

    candidate_lengths = search.lengths.copy()
    candidate_lengths[left + 1 : right] = tendon_lengths(search.model, candidate[left + 1 : right])
    selected_jump = (
        abs(
            candidate_lengths[search.event_frame, search.event_tendon]
            - candidate_lengths[search.event_frame - 1, search.event_tendon]
        )
        / search.length_scale
    )
    if selected_jump > search.threshold:
        return None

    after, candidate_tracked = _repair_events(
        search.model,
        candidate_lengths,
        search.threshold,
        search.tracked,
    )
    score = (len(after), after[0][0] if after else 0.0)
    if score >= search.before_score:
        return None
    return _RepairCandidate(
        score=score,
        qpos=candidate,
        lengths=candidate_lengths,
        frames=frames,
        tracked=candidate_tracked,
    )


def _best_candidate_in_scope(
    search: _RepairSearch,
    scope: _RepairScope,
) -> _RepairCandidate | None:
    """Return the best candidate at the smallest accepted span size."""
    for changed_frames in range(1, scope.max_changed_frames + 1):
        best = None
        for left, right in _candidate_spans(search.event_frame, changed_frames, len(search.qpos)):
            candidate = _evaluate_candidate(
                search,
                scope,
                left,
                right,
            )
            if candidate is not None and (best is None or candidate.score < best.score):
                best = candidate
        if best is not None:
            return best
    return None


def _find_repair_candidate(
    search: _RepairSearch,
    scopes: list[_RepairScope],
) -> _RepairCandidate | None:
    """Search the tendon-local scope before the whole-pose fallback."""
    for scope in scopes:
        candidate = _best_candidate_in_scope(search, scope)
        if candidate is not None:
            return candidate
    return None


def repair_tendon_discontinuities(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    validator: CandidateValidator | None = None,
    threshold: float = 0.05,
    max_changed_frames: int = 15,
    max_local_changed_frames: int | None = None,
    max_iterations: int = 50,
) -> tuple[np.ndarray, list[int], list[tuple[float, int, int]]]:
    """Repair tendon events using the smallest accepted interpolation spans.

    Args:
        model: MuJoCo model containing the tendons and joint equalities.
        qpos: Generalized positions with shape ``(T, nq)``.
        validator: Optional callback that validates a candidate and changed range.
        threshold: Maximum accepted relative tendon-length jump.
        max_changed_frames: Maximum full-pose interpolation span.
        max_local_changed_frames: Maximum tendon-local interpolation span.
        max_iterations: Maximum number of accepted repair searches.

    Returns:
        Repaired generalized positions, changed frame indices, and unresolved events.

    Raises:
        ValueError: If a threshold or frame limit is invalid.
    """
    max_local_changed_frames = _validate_repair_limits(
        threshold,
        max_changed_frames,
        max_local_changed_frames,
    )
    out = np.asarray(qpos, dtype=float).copy()
    if len(out) < 3 or max_changed_frames == 0 or model.ntendon == 0:
        current = adaptive_tendon_events(
            model,
            tendon_lengths(model, out),
            threshold=threshold,
        )
        return out, [], current

    values = tendon_lengths(model, out)
    preserve_prefix = _preserved_qpos_prefix(model)
    dependency_data = mujoco.MjData(model) if hasattr(model, "nv") else None
    tracked: set[tuple[int, int]] = set()
    changed: set[int] = set()
    for _iteration in range(max_iterations):
        before, tracked = _repair_events(model, values, threshold, tracked)
        if not before:
            return out, sorted(changed), []
        before_score = (len(before), before[0][0])
        _jump, event_frame, event_tendon = before[0]
        length_scale = max(float(model.tendon_length0[event_tendon]), 1e-6)
        scopes = _repair_scopes(
            model,
            dependency_data,
            out,
            event_frame,
            event_tendon,
            max_local_changed_frames,
            max_changed_frames,
        )
        search = _RepairSearch(
            model=model,
            qpos=out,
            lengths=values,
            tracked=tracked,
            preserve_prefix=preserve_prefix,
            validator=validator,
            event_frame=event_frame,
            event_tendon=event_tendon,
            length_scale=length_scale,
            threshold=threshold,
            before_score=before_score,
        )
        candidate = _find_repair_candidate(search, scopes)
        if candidate is None:
            break
        out = candidate.qpos
        values = candidate.lengths
        tracked = candidate.tracked
        changed.update(candidate.frames)

    remaining, _tracked = _repair_events(model, values, threshold, tracked)
    return out, sorted(changed), remaining
