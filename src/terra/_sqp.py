"""Backend-neutral pieces of TERRA's sequential quadratic programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


class _CvxpyShim:
    """Proxy CVXPY while intercepting problem construction.

    Args:
        cvxpy_module: CVXPY module to proxy.
        hook: Callback that transforms an objective before problem construction.
    """

    def __init__(self, cvxpy_module, hook):
        """Initialize the proxy module and objective callback.

        Args:
            cvxpy_module: CVXPY module to proxy.
            hook: Objective transformation callback.
        """
        self._cp = cvxpy_module
        self._hook = hook

    def __getattr__(self, name):
        """Forward an attribute lookup to CVXPY.

        Args:
            name: Attribute name.

        Returns:
            The corresponding CVXPY attribute.
        """
        return getattr(self._cp, name)

    def Problem(self, objective, constraints=None):  # noqa: N802 - mirror cvxpy.Problem
        """Build a CVXPY problem with the transformed objective.

        Args:
            objective: Original CVXPY objective.
            constraints: Optional CVXPY constraints.

        Returns:
            A CVXPY problem.
        """
        return self._cp.Problem(self._hook(objective), constraints)


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


def _objective_with_quadratic_terms(cp, objective, terms: Sequence[_QuadraticTerm]):
    """Append TERRA least-squares terms to an OmniRetarget objective."""
    step = next((variable for variable in objective.variables() if variable.name() == "dqa"), None)
    if step is None:
        raise RuntimeError(
            "OmniRetarget's step variable 'dqa' is unavailable; the added costs "
            "cannot be attached. See TerraRetargeter."
        )
    return cp.Minimize(objective.args[0] + cp.sum([term.cvxpy_expression(cp, step) for term in terms]))
