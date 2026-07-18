"""ODM solvers provide direct, iterative, and legacy numerical routes over one
fixed coefficient objective without owning model or Technique Result semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from scipy import linalg, optimize

from crackpy.fracture_analysis.odm.assembly import LinearSystem
from crackpy.fracture_analysis.odm.results import CoefficientFitResult

__all__ = [
    "SolverRoute",
    "solve_direct",
    "solve_iterative",
    "solve_legacy",
    "solve_coefficient_fit",
    "to_optimize_result",
    "coefficient_fit_from_optimize_result",
]

SolverRoute = Literal["direct", "iterative", "legacy"]
ResidualFunction = Callable[[np.ndarray], np.ndarray]


def to_optimize_result(result: CoefficientFitResult) -> optimize.OptimizeResult:
    """Adapt an owned coefficient fit to the mutable SciPy-compatible facade.

    Args:
        result: Analysis-owned coefficient-fit result to adapt.

    Returns:
        A normalized ``OptimizeResult`` containing fresh mutable array copies.
    """
    return optimize.OptimizeResult(
        solver=result.solver,
        x=result.coefficients.copy(),
        fun=result.residual.copy(),
        cost=result.cost,
        jac=None if result.jacobian is None else result.jacobian.copy(),
        rank=result.rank,
        singular_values=(
            None if result.singular_values is None else result.singular_values.copy()
        ),
        success=result.success,
        message=result.message,
        status=result.status,
        nfev=result.nfev,
        njev=result.njev,
    )


def coefficient_fit_from_optimize_result(
    result: optimize.OptimizeResult,
) -> CoefficientFitResult:
    """Copy a normalized SciPy-compatible result into owned fit storage.

    Args:
        result: Facade result with the normalized ODM coefficient-fit fields.

    Returns:
        An immutable analysis-owned coefficient-fit result.
    """
    return CoefficientFitResult(
        solver=cast(SolverRoute, result.solver),
        coefficients=np.array(result.x, copy=True),
        residual=np.array(result.fun, copy=True),
        cost=result.cost,
        jacobian=None if result.jac is None else np.array(result.jac, copy=True),
        rank=result.rank,
        singular_values=(
            None
            if result.singular_values is None
            else np.array(result.singular_values, copy=True)
        ),
        success=result.success,
        message=result.message,
        status=result.status,
        nfev=result.nfev,
        njev=result.njev,
    )


def solve_direct(system: LinearSystem) -> CoefficientFitResult:
    """Solve a fixed ODM coefficient system through direct GELSS least squares.

    Args:
        system: Assembled residual-by-coefficient matrix and measured target.

    Returns:
        An owned normalized result containing GELSS coefficients and metadata.
    """
    coefficients, _, rank, singular_values = linalg.lstsq(
        system.matrix,
        system.target,
        lapack_driver="gelss",
    )
    residual = system.matrix @ coefficients - system.target
    return CoefficientFitResult(
        solver="direct",
        coefficients=coefficients,
        residual=residual,
        cost=0.5 * np.dot(residual, residual),
        jacobian=system.matrix,
        rank=rank,
        singular_values=singular_values,
        success=True,
        message="Solved by direct linear least squares.",
        status=1,
        nfev=1,
        njev=1,
    )


def solve_iterative(
    system: LinearSystem,
    *,
    method: str = "lm",
    init_coeffs: np.ndarray | None = None,
) -> CoefficientFitResult:
    """Solve a fixed ODM coefficient system through iterative least squares.

    Args:
        system: Assembled residual-by-coefficient matrix and measured target.
        method: SciPy least-squares algorithm control.
        init_coeffs: Optional initial coefficient vector.

    Returns:
        An owned normalized result from SciPy's iterative least-squares solve.
    """
    initial = (
        np.zeros(system.matrix.shape[1])
        if init_coeffs is None
        else np.array(init_coeffs, copy=True)
    )
    result = optimize.least_squares(
        lambda coefficients: system.matrix @ coefficients - system.target,
        x0=initial,
        jac=lambda _: system.matrix,
        method=method,
    )
    return CoefficientFitResult(
        solver="iterative",
        coefficients=result.x,
        residual=result.fun,
        cost=result.cost,
        jacobian=result.jac,
        rank=None,
        singular_values=None,
        success=result.success,
        message=result.message,
        status=result.status,
        nfev=result.nfev,
        njev=result.njev,
    )


def solve_legacy(
    *,
    residuals: ResidualFunction,
    jacobian: ResidualFunction,
    n_coefficients: int,
    method: str = "lm",
    init_coeffs: np.ndarray | None = None,
) -> CoefficientFitResult:
    """Solve an ODM coefficient fit through legacy residual callbacks.

    Args:
        residuals: Established callback evaluating the displacement residual.
        jacobian: Established callback evaluating the residual Jacobian.
        n_coefficients: Number of coefficients expected by both callbacks.
        method: SciPy least-squares algorithm control.
        init_coeffs: Optional initial coefficient vector.

    Returns:
        An owned normalized result from SciPy's callback-based solve.
    """
    initial = (
        np.random.rand(n_coefficients)
        if init_coeffs is None
        else np.array(init_coeffs, copy=True)
    )
    result = optimize.least_squares(
        fun=residuals,
        jac=jacobian,
        x0=initial,
        method=method,
    )
    return CoefficientFitResult(
        solver="legacy",
        coefficients=result.x,
        residual=result.fun,
        cost=result.cost,
        jacobian=result.jac,
        rank=None,
        singular_values=None,
        success=result.success,
        message=result.message,
        status=result.status,
        nfev=result.nfev,
        njev=result.njev,
    )


def solve_coefficient_fit(
    system: LinearSystem,
    *,
    solver: SolverRoute = "direct",
    method: str = "lm",
    init_coeffs: np.ndarray | None = None,
    residuals: ResidualFunction | None = None,
    jacobian: ResidualFunction | None = None,
) -> CoefficientFitResult:
    """Select one Numerical Solver Route for a fixed ODM coefficient fit.

    Args:
        system: Assembled residual-by-coefficient matrix and measured target.
        solver: Direct, iterative, or legacy Solver Route selector.
        method: SciPy algorithm forwarded to iterative and legacy routes.
        init_coeffs: Optional initial vector for iterative and legacy routes.
        residuals: Legacy displacement-residual callback.
        jacobian: Legacy residual-Jacobian callback.

    Returns:
        The selected route's immutable normalized coefficient-fit result.

    Raises:
        ValueError: If the route is unsupported or legacy callbacks are absent.
    """
    if solver == "direct":
        return solve_direct(system)
    if solver == "iterative":
        return solve_iterative(system, method=method, init_coeffs=init_coeffs)
    if solver == "legacy":
        if residuals is None or jacobian is None:
            raise ValueError(
                "Legacy Solver requires residual and Jacobian callbacks."
            )
        return solve_legacy(
            residuals=residuals,
            jacobian=jacobian,
            n_coefficients=system.matrix.shape[1],
            method=method,
            init_coeffs=init_coeffs,
        )
    raise ValueError(
        f"Unsupported Solver Route {solver!r}. Expected direct, iterative, or legacy."
    )
