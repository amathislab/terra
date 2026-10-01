"""Pure objective assembly shared by TERRA's condensed QP backends."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from terra._sqp import _LaplacianLinearization, _QuadraticTerm


def native_condensed_objective(
    *,
    nq_a: int,
    track_nominal_indices: Sequence[int],
    q_diag: np.ndarray,
    smooth_weight: float | np.ndarray,
    laplacian: _LaplacianLinearization,
    q_a_n_last: np.ndarray,
    dqa_smooth: np.ndarray,
    nominal_weight: float,
    q_a_nominal: np.ndarray | None,
    attached_terms: Sequence[_QuadraticTerm],
) -> tuple[list[tuple], list[tuple]]:
    """Assemble native least-squares and centered-quadratic terms."""
    least_squares = [laplacian.as_native_term()]
    if nominal_weight > 0 and q_a_nominal is not None:
        indices = np.array(track_nominal_indices, dtype=int)
        if indices.size > 0:
            selection = np.eye(nq_a)[indices]
            least_squares.append(
                (
                    selection,
                    q_a_nominal[indices] - q_a_n_last[indices],
                    nominal_weight,
                )
            )

    diagonal = np.asarray(q_diag, dtype=float).reshape(-1)
    least_squares.append((None, -q_a_n_last, diagonal))
    centered_quadratics = []
    if np.isscalar(smooth_weight):
        least_squares.append((None, dqa_smooth, smooth_weight))
    else:
        smoothing = np.asarray(smooth_weight, dtype=float)
        if smoothing.ndim == 1:
            least_squares.append((None, dqa_smooth, smoothing))
        else:
            centered_quadratics.append((smoothing, dqa_smooth))
    least_squares.extend(term.as_native_term() for term in attached_terms)
    return least_squares, centered_quadratics


def cvxpy_condensed_objective(
    cp,
    step,
    *,
    track_nominal_indices: Sequence[int],
    q_diag: np.ndarray,
    smooth_weight: float | np.ndarray,
    laplacian: _LaplacianLinearization,
    q_a_n_last: np.ndarray,
    dqa_smooth: np.ndarray,
    nominal_weight: float,
    q_a_nominal: np.ndarray | None,
    attached_terms: Sequence[_QuadraticTerm],
) -> list:
    """Build the condensed CVXPY objective in its established term order."""
    objective = [laplacian.cvxpy_expression(cp, step)]
    if nominal_weight > 0 and q_a_nominal is not None:
        indices = np.array(track_nominal_indices, dtype=int)
        if indices.size > 0:
            residual = step[indices] - (q_a_nominal[indices] - q_a_n_last[indices])
            objective.append(nominal_weight * cp.sum_squares(residual))

    diagonal = np.asarray(q_diag, dtype=float).reshape(-1)
    objective.append(cp.sum_squares(cp.multiply(np.sqrt(diagonal), step + q_a_n_last)))
    if np.isscalar(smooth_weight):
        objective.append(smooth_weight * cp.sum_squares(step - dqa_smooth))
    else:
        smoothing = np.asarray(smooth_weight, dtype=float)
        if smoothing.ndim == 1:
            objective.append(cp.sum_squares(cp.multiply(np.sqrt(smoothing), step - dqa_smooth)))
        else:
            objective.append(cp.quad_form(step - dqa_smooth, smoothing))
    objective.extend(term.cvxpy_expression(cp, step) for term in attached_terms)
    return objective


def updated_pose(
    q: np.ndarray,
    q_a_indices: np.ndarray,
    q_a_n_last: np.ndarray,
    step: np.ndarray,
) -> np.ndarray:
    """Apply an optimizer step and normalize the floating-root quaternion."""
    updated = np.copy(q)
    updated[q_a_indices] = step + q_a_n_last
    updated[3:7] /= np.linalg.norm(updated[3:7]) + 1e-12
    return updated


__all__ = [
    "cvxpy_condensed_objective",
    "native_condensed_objective",
    "updated_pose",
]
