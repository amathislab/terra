"""Internal quadratic-program assembly and Clarabel integration.

This module contains solver mechanics that are independent of musculoskeletal
retargeting.  The TERRA formulation remains in :mod:`terra.retargeter`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy import sparse as sp


def _add_weighted_least_squares(
    hessian: np.ndarray,
    linear: np.ndarray,
    constant: float,
    jacobian,
    target,
    weight,
) -> float:
    """Accumulate a weighted least-squares term in Clarabel QP form.

    Args:
        hessian: Hessian matrix updated in place.
        linear: Linear objective vector updated in place.
        constant: Current constant objective term.
        jacobian: Dense or sparse least-squares Jacobian.
        target: Least-squares target vector.
        weight: Scalar or per-row weights.

    Returns:
        Updated constant objective term.

    Raises:
        ValueError: If the Jacobian, target, and weights have inconsistent rows.
    """
    jacobian_is_sparse = sp.issparse(jacobian)
    jacobian = sp.csr_matrix(jacobian, dtype=float) if jacobian_is_sparse else np.asarray(jacobian, dtype=float)
    if not jacobian_is_sparse and jacobian.ndim == 1:
        jacobian = jacobian.reshape(1, -1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if jacobian.shape[0] != len(target):
        raise ValueError(f"least-squares rows {jacobian.shape[0]} != target length {len(target)}")
    weights = np.asarray(weight, dtype=float)
    if weights.ndim == 0:
        weights = np.full(len(target), float(weights))
    else:
        weights = weights.reshape(-1)
    if len(weights) != len(target):
        raise ValueError(f"weight length {len(weights)} != target length {len(target)}")
    weighted_jacobian = jacobian.multiply(weights[:, None]) if jacobian_is_sparse else jacobian * weights[:, None]
    normal_matrix = jacobian.T @ weighted_jacobian
    hessian += 2.0 * (normal_matrix.toarray() if sp.issparse(normal_matrix) else normal_matrix)
    linear -= 2.0 * np.asarray(jacobian.T @ (weights * target)).reshape(-1)
    return constant + float(np.dot(weights, target * target))


def _add_weighted_identity_least_squares(
    hessian: np.ndarray,
    linear: np.ndarray,
    constant: float,
    target,
    weight,
) -> float:
    """Accumulate a weighted identity least-squares term.

    Args:
        hessian: Hessian matrix updated in place.
        linear: Linear objective vector updated in place.
        constant: Current constant objective term.
        target: Least-squares target vector.
        weight: Scalar or per-row weights.

    Returns:
        Updated constant objective term.

    Raises:
        ValueError: If the target and weights do not match the QP dimension.
    """
    target = np.asarray(target, dtype=float).reshape(-1)
    if len(target) != len(linear):
        raise ValueError(f"identity least-squares rows {len(linear)} != target length {len(target)}")
    weights = np.asarray(weight, dtype=float)
    if weights.ndim == 0:
        weights = np.full(len(target), float(weights))
    else:
        weights = weights.reshape(-1)
    if len(weights) != len(target):
        raise ValueError(f"weight length {len(weights)} != target length {len(target)}")
    hessian[np.diag_indices(len(target))] += 2.0 * weights
    linear -= 2.0 * (weights * target)
    return constant + float(np.dot(weights, target * target))


@cache
def _upper_triangle_csc_coordinates(n_dof: int) -> tuple[np.ndarray, np.ndarray]:
    """Build column-major coordinates for an upper-triangular matrix.

    Args:
        n_dof: Matrix size.

    Returns:
        Row and column index arrays for the upper triangle.
    """
    columns = np.repeat(np.arange(n_dof), np.arange(1, n_dof + 1))
    rows = np.concatenate([np.arange(column + 1) for column in range(n_dof)])
    return rows, columns


def _symmetric_upper_csc(hessian: np.ndarray) -> sp.csc_matrix:
    """Convert a dense Hessian to Clarabel's symmetric upper CSC format.

    Args:
        hessian: Square dense Hessian matrix.

    Returns:
        Sparse upper-triangular symmetric matrix.

    Raises:
        ValueError: If ``hessian`` is not square.
    """
    hessian = np.asarray(hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"hessian must be square, got {hessian.shape}")
    n_dof = hessian.shape[0]
    rows, columns = _upper_triangle_csc_coordinates(n_dof)
    values = 0.5 * (hessian[rows, columns] + hessian[columns, rows])
    nonzero = values != 0.0
    counts = np.bincount(columns[nonzero], minlength=n_dof)
    indptr = np.empty(n_dof + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return sp.csc_matrix((values[nonzero], rows[nonzero], indptr), shape=(n_dof, n_dof))


def _dense_inequality_soc_csc(g_matrix: np.ndarray) -> sp.csc_matrix:
    """Build the inequality and trust-region matrix in CSC format.

    Args:
        g_matrix: Dense inequality matrix ``G``.

    Returns:
        Sparse matrix representing ``[-G; 0; -I]``.

    Raises:
        ValueError: If ``g_matrix`` is not two-dimensional.
    """
    g_matrix = np.asarray(g_matrix, dtype=float)
    if g_matrix.ndim != 2:
        raise ValueError(f"inequality matrix must be two-dimensional, got {g_matrix.shape}")
    n_rows, n_dof = g_matrix.shape
    transposed = g_matrix.T
    nonzero = transposed != 0.0
    columns, rows = np.nonzero(nonzero)
    values = -transposed[nonzero]
    counts = np.bincount(columns, minlength=n_dof)
    indptr = np.empty(n_dof + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts + 1, out=indptr[1:])

    data = np.empty(len(values) + n_dof, dtype=float)
    indices = np.empty(len(values) + n_dof, dtype=np.int64)
    inequality_positions = np.arange(len(values)) + columns
    data[inequality_positions] = values
    indices[inequality_positions] = rows
    soc_positions = indptr[1:] - 1
    data[soc_positions] = -1.0
    indices[soc_positions] = n_rows + 1 + np.arange(n_dof)
    return sp.csc_matrix((data, indices, indptr), shape=(n_rows + n_dof + 1, n_dof))


def _native_clarabel_qp(
    n_dof: int,
    least_squares: list[tuple],
    inequalities: tuple[np.ndarray, np.ndarray],
    step_size: float,
    *,
    centered_quadratics: list[tuple] | None = None,
    verbose: bool = False,
):
    """Solve the reduced TERRA QP directly with Clarabel.

    Args:
        n_dof: Number of optimization variables.
        least_squares: ``(jacobian, target, weight)`` objective terms. A
            ``None`` Jacobian represents the identity.
        inequalities: Matrix and right-hand side for ``G @ x >= h``.
        step_size: Radius of the second-order-cone trust region.
        centered_quadratics: Optional weight-matrix and center pairs.
        verbose: Whether to enable Clarabel logging.

    Returns:
        Optimizer step, full objective value, and Clarabel solution object.

    Raises:
        RuntimeError: If Clarabel does not return a solved status.
    """
    import clarabel

    hessian = np.zeros((n_dof, n_dof), dtype=float)
    linear = np.zeros(n_dof, dtype=float)
    constant = 0.0
    for jacobian, target, weight in least_squares:
        if jacobian is None:
            constant = _add_weighted_identity_least_squares(hessian, linear, constant, target, weight)
        else:
            constant = _add_weighted_least_squares(hessian, linear, constant, jacobian, target, weight)

    for weight_matrix, center in centered_quadratics or ():
        weight_matrix = np.asarray(weight_matrix, dtype=float)
        center = np.asarray(center, dtype=float).reshape(-1)
        hessian += 2.0 * weight_matrix
        linear -= 2.0 * weight_matrix @ center
        constant += float(center @ weight_matrix @ center)

    # Supplying only the upper triangle is Clarabel's documented QP convention.
    p_matrix = _symmetric_upper_csc(hessian)
    g_matrix, rhs = inequalities
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    g_is_sparse = sp.issparse(g_matrix)
    g_matrix = (
        sp.csr_matrix(g_matrix, shape=(len(rhs), n_dof), dtype=float)
        if g_is_sparse
        else np.asarray(g_matrix, dtype=float).reshape(len(rhs), n_dof)
    )
    sparse_a_blocks = []
    b_blocks = []
    cones = []
    if len(rhs):
        # Gx >= h  <=>  -Gx + s = -h, s in the non-negative cone.
        if g_is_sparse:
            sparse_a_blocks.append(-g_matrix)
        b_blocks.append(-rhs)
        cones.append(clarabel.NonnegativeConeT(len(rhs)))
    # [step_size; x] in SOC  <=>  [0; -I]x + s = [step_size; 0]. Native normal
    # equations are never used without this certificate; initialization infeasibility
    # routes to the independently validated lifted CVXPY fallback instead.
    if g_is_sparse:
        sparse_a_blocks.append(
            sp.vstack(
                [sp.csr_matrix((1, n_dof)), -sp.eye(n_dof, format="csr")],
                format="csr",
            )
        )
    b_blocks.append(np.r_[float(step_size), np.zeros(n_dof)])
    cones.append(clarabel.SecondOrderConeT(n_dof + 1))

    if g_is_sparse:
        a_matrix = sp.vstack(sparse_a_blocks, format="csc") if sparse_a_blocks else sp.csc_matrix((0, n_dof))
    else:
        a_matrix = _dense_inequality_soc_csc(g_matrix)
    b_vector = np.concatenate(b_blocks) if b_blocks else np.empty(0)
    settings = clarabel.DefaultSettings()
    settings.verbose = bool(verbose)
    solver = clarabel.DefaultSolver(p_matrix, linear, a_matrix, b_vector, cones, settings)
    solution = solver.solve()
    if solution.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
        raise RuntimeError(
            f"Clarabel solve failed: {solution.status}; iterations={solution.iterations}, "
            f"primal_residual={solution.r_prim:.3e}, dual_residual={solution.r_dual:.3e}"
        )
    solution_x = np.asarray(solution.x, dtype=float)
    return solution_x, float(solution.obj_val + constant), solution
