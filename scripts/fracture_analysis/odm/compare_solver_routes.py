"""Compare the supported ODM Solver Routes on one deterministic displacement field."""

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

import numpy as np
from scipy.optimize import OptimizeResult

from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties
from crackpy.input.input_data import InputData

_SolverRoute = Literal["direct", "iterative", "legacy"]
_FitFunction = Callable[..., OptimizeResult]
_OUTPUT_FIELDS = (
    "fit",
    "solver",
    "success",
    "cost",
    "elapsed_ms",
    "max_abs_coefficient_delta",
    "max_abs_residual_delta",
    "coefficients",
)


@dataclass(frozen=True)
class _ComparisonRow:
    """Retain one facade result and its comparison with the direct route.

    Attributes:
        fit: Stable name of the fitted Williams displacement system.
        solver: Solver Route used for the fit.
        elapsed_ms: Observed wall-clock duration of this single fit in milliseconds.
        max_abs_coefficient_delta: Largest coefficient difference from the direct route.
        max_abs_residual_delta: Largest residual difference from the direct route.
        result: Complete result returned by the public ``Optimization`` facade.
    """

    fit: str
    solver: _SolverRoute
    elapsed_ms: float
    max_abs_coefficient_delta: float
    max_abs_residual_delta: float
    result: OptimizeResult


def _build_optimization() -> Optimization:
    """Build one deterministic in-memory ODM optimization problem."""
    axis = np.linspace(-1.5, 1.5, 9)
    coor_x, coor_y = np.meshgrid(axis, axis)
    data = InputData()
    data.coor_x = coor_x.ravel()
    data.coor_y = coor_y.ravel()
    data.coor_z = np.zeros(coor_x.size)
    data.disp_x = (0.04 + 0.02 * coor_x - 0.01 * coor_y + 0.003 * coor_x * coor_y).ravel()
    data.disp_y = (-0.02 + 0.01 * coor_x + 0.03 * coor_y - 0.002 * coor_x**2).ravel()
    data.disp_z = (0.01 - 0.015 * coor_x + 0.005 * coor_y).ravel()
    options = OptimizationProperties(
        angle_gap=25,
        min_radius=0.3,
        max_radius=1.1,
        tick_size=0.2,
        terms=[-1, 1, 2],
    )
    return Optimization(data, options=options)


def _maximum_absolute_delta(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return the largest elementwise absolute difference between two vectors."""
    if candidate.size == 0:
        return 0.0
    return float(np.max(np.abs(candidate - reference)))


def _run_route(
        fit: str,
        solver: _SolverRoute,
        fit_function: _FitFunction,
        direct_result: OptimizeResult | None = None) -> _ComparisonRow:
    """Run one public Solver Route and compare it with the direct result."""
    arguments = {"solver": solver}
    if solver != "direct":
        if direct_result is None:
            raise ValueError("A direct result is required for iterative and legacy comparisons.")
        arguments.update(method="lm", init_coeffs=np.zeros_like(direct_result.x))

    started = perf_counter()
    result = fit_function(**arguments)
    elapsed_ms = (perf_counter() - started) * 1_000.0
    reference = result if direct_result is None else direct_result
    return _ComparisonRow(
        fit=fit,
        solver=solver,
        elapsed_ms=elapsed_ms,
        max_abs_coefficient_delta=_maximum_absolute_delta(result.x, reference.x),
        max_abs_residual_delta=_maximum_absolute_delta(result.fun, reference.fun),
        result=result,
    )


def _require_equivalent_result(candidate: _ComparisonRow, direct: _ComparisonRow) -> None:
    """Require one iterative or legacy result to match its full-rank direct reference."""
    np.testing.assert_allclose(candidate.result.x, direct.result.x, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(candidate.result.fun, direct.result.fun, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(candidate.result.cost, direct.result.cost, rtol=0.0, atol=1e-12)


def _compare_solver_routes() -> list[_ComparisonRow]:
    """Run all Solver Routes for the in-plane and out-of-plane Williams fits."""
    optimization = _build_optimization()
    fits = (
        ("williams_xy", optimization.optimize_williams_displacements_xy),
        ("williams_z", optimization.optimize_williams_displacements_z),
    )
    rows = []
    for fit, fit_function in fits:
        direct = _run_route(fit, "direct", fit_function)
        if direct.result.rank != direct.result.x.size:
            raise AssertionError(
                f"{fit} direct system is not full rank: "
                f"rank={direct.result.rank}, coefficients={direct.result.x.size}."
            )
        rows.append(direct)
        for solver in ("iterative", "legacy"):
            candidate = _run_route(fit, solver, fit_function, direct.result)
            _require_equivalent_result(candidate, direct)
            rows.append(candidate)
    return rows


def _format_row(row: _ComparisonRow) -> str:
    """Format one comparison row as stable tab-separated fields."""
    values = (
        row.fit,
        row.solver,
        str(bool(row.result.success)),
        f"{row.result.cost:.12e}",
        f"{row.elapsed_ms:.6f}",
        f"{row.max_abs_coefficient_delta:.12e}",
        f"{row.max_abs_residual_delta:.12e}",
        json.dumps(row.result.x.tolist(), separators=(",", ":")),
    )
    return "\t".join(values)


def main() -> None:
    """Print the six Williams Solver Route comparison rows.

    Returns:
        None.
    """
    print("\t".join(_OUTPUT_FIELDS))
    for row in _compare_solver_routes():
        print(_format_row(row))


if __name__ == "__main__":
    main()
