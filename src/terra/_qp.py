"""Internal quadratic-program assembly and Clarabel integration.

This module contains solver mechanics that are independent of musculoskeletal
retargeting.  The TERRA formulation remains in :mod:`terra.retargeter`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class _QuadraticTerm:
    """Represent one weighted least-squares term for every QP backend.

    The objective is ``weight * ||jacobian @ step - target||²`` for scalar
    weights. ``cvxpy_row_scale`` preserves an already-computed per-row square
    root while ``weight`` stores its square for the native normal equations.
    """

    jacobian: np.ndarray
    target: np.ndarray
    weight: float | np.ndarray
    cvxpy_row_scale: np.ndarray | None = None

    def as_native_term(self) -> tuple[np.ndarray, np.ndarray, float | np.ndarray]:
        """Return the native-solver representation without copying arrays."""
        return self.jacobian, self.target, self.weight

    def cvxpy_expression(self, cp, step):
        """Build the mathematically equivalent CVXPY expression."""
        residual = cp.Constant(self.jacobian) @ step - self.target
        if self.cvxpy_row_scale is not None:
            return cp.sum_squares(cp.multiply(self.cvxpy_row_scale, residual))
        return self.weight * cp.sum_squares(residual)


@dataclass(frozen=True)
class _LaplacianLinearization:
    """Linearized interaction-mesh Laplacian matching objective."""

    jacobian: np.ndarray
    current: np.ndarray
    target: np.ndarray
    row_scale: np.ndarray

    def as_native_term(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the native least-squares form ``J step ≈ target - current``."""
        return self.jacobian, self.target - self.current, self.row_scale**2

    def cvxpy_expression(self, cp, step):
        """Build the original condensed CVXPY Laplacian expression."""
        residual = cp.Constant(self.jacobian) @ step + self.current - self.target
        return cp.sum_squares(cp.multiply(self.row_scale, residual))


@dataclass
class _InequalityConstraints:
    """Keep equivalent CVXPY and native forms of ordered linear constraints."""

    n_dof: int
    _cvxpy_builders: list = field(default_factory=list, repr=False)
    _native_rows: list[np.ndarray] = field(default_factory=list, repr=False)
    _native_rhs: list[np.ndarray] = field(default_factory=list, repr=False)

    def _append_native(self, jacobian, rhs) -> None:
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.ndim == 1:
            jacobian = jacobian.reshape(1, -1)
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if len(rhs) == 1 and jacobian.shape[0] != 1:
            rhs = np.full(jacobian.shape[0], rhs.item())
        if jacobian.shape != (len(rhs), self.n_dof):
            raise ValueError(f"inequality shape {jacobian.shape} does not match ({len(rhs)}, {self.n_dof})")
        self._native_rows.append(jacobian)
        self._native_rhs.append(rhs)

    def add_lower_bound(self, jacobian, rhs) -> None:
        """Add ``jacobian @ step >= rhs`` in both backend representations."""
        self._cvxpy_builders.append(lambda step, j=jacobian, r=rhs: [j @ step >= r])
        self._append_native(jacobian, rhs)

    def add_two_sided(self, jacobian, lower, upper) -> None:
        """Add lower and upper bounds on one linear projection."""
        self._cvxpy_builders.append(
            lambda step, j=jacobian, lo=lower, hi=upper: [
                j @ step >= lo,
                j @ step <= hi,
            ]
        )
        self._append_native(jacobian, lower)
        self._append_native(-jacobian, -upper)

    def add_variable_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Add componentwise bounds without changing the CVXPY expression shape."""
        self._cvxpy_builders.append(lambda step, lo=lower, hi=upper: [step >= lo, step <= hi])
        identity = np.eye(self.n_dof)
        self._append_native(identity, lower)
        self._append_native(-identity, -upper)

    def native(self) -> tuple[np.ndarray, np.ndarray]:
        """Stack constraints in the row order required by the native solver."""
        rows = np.vstack(self._native_rows) if self._native_rows else np.empty((0, self.n_dof))
        rhs = np.concatenate(self._native_rhs) if self._native_rhs else np.empty(0)
        return rows, rhs

    def cvxpy(self, step) -> list:
        """Materialize CVXPY constraints in their original insertion order."""
        return [constraint for build in self._cvxpy_builders for constraint in build(step)]


def _row_quadratic_terms(raw_terms) -> list[_QuadraticTerm]:
    """Convert scalar one-row linearizations into quadratic terms."""
    return [
        _QuadraticTerm(
            np.asarray(jacobian).reshape(1, -1),
            np.atleast_1d(target),
            weight,
        )
        for weight, jacobian, target in raw_terms
    ]


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
