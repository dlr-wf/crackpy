"""Compatibility evidence that public Optimization methods select each ODM Solver
Route while preserving established signatures and SciPy-shaped results.
"""

from unittest import mock

import numpy as np
import pytest

import crackpy.fracture_analysis.odm.solvers as solver_module
import crackpy.fracture_analysis.optimization as optimization_module
from crackpy.fracture_analysis.optimization import Optimization, OptimizationProperties
from crackpy.input.input_data import InputData


def _make_optimization() -> Optimization:
    """Return a small prepared optimization instance for Solver Route tests."""
    axis = np.linspace(-1.5, 1.5, 9)
    coor_x, coor_y = np.meshgrid(axis, axis)
    data = InputData()
    data.coor_x = coor_x.ravel()
    data.coor_y = coor_y.ravel()
    data.coor_z = np.zeros(coor_x.size)
    data.disp_x = (0.04 + 0.02 * coor_x - 0.01 * coor_y).ravel()
    data.disp_y = (-0.02 + 0.01 * coor_x + 0.03 * coor_y).ravel()
    data.disp_z = (0.01 - 0.015 * coor_x + 0.005 * coor_y).ravel()
    options = OptimizationProperties(
        angle_gap=25,
        min_radius=0.3,
        max_radius=1.1,
        tick_size=0.2,
        terms=[-1, 1, 2],
    )
    return Optimization(data, options=options)


def test_cjp_mode_i_direct_route_returns_normalized_facade_result() -> None:
    """CJP Mode I should adapt the centralized direct-route result."""
    optimization = _make_optimization()
    with (
        mock.patch.object(
            optimization_module,
            "solve_coefficient_fit",
            wraps=solver_module.solve_coefficient_fit,
        ) as solve,
        mock.patch.object(
            optimization_module,
            "to_optimize_result",
            wraps=solver_module.to_optimize_result,
        ) as adapt,
    ):
        result = optimization.optimize_cjp_displacements_modeI(
            method="trf",
            init_coeffs=np.ones(5),
            solver="direct",
        )

    solve.assert_called_once_with(
        optimization._cjp_assembly.mode_i,
        solver="direct",
    )
    adapt.assert_called_once()
    assert isinstance(adapt.call_args.args[0], solver_module.CoefficientFitResult)
    assert set(result.keys()) == {
        "solver",
        "x",
        "fun",
        "cost",
        "jac",
        "rank",
        "singular_values",
        "success",
        "message",
        "status",
        "nfev",
        "njev",
    }
    assert result.solver == "direct"
    assert "diagnostics" not in result


def test_cjp_mixed_mode_direct_route_uses_centralized_solver() -> None:
    """CJP mixed mode should solve its fixed system through the route selector."""
    optimization = _make_optimization()
    with mock.patch.object(
        optimization_module,
        "solve_coefficient_fit",
        wraps=solver_module.solve_coefficient_fit,
    ) as solve:
        result = optimization.optimize_cjp_displacements_mixedmode(
            solver="direct"
        )

    solve.assert_called_once_with(
        optimization._cjp_assembly.mixed_mode,
        solver="direct",
    )
    assert result.solver == "direct"
    assert "diagnostics" not in result


def test_williams_xy_direct_route_uses_centralized_solver() -> None:
    """In-plane Williams fitting should route its prepared fixed system."""
    optimization = _make_optimization()
    with mock.patch.object(
        optimization_module,
        "solve_coefficient_fit",
        wraps=solver_module.solve_coefficient_fit,
    ) as solve:
        result = optimization.optimize_williams_displacements_xy(solver="direct")

    solve.assert_called_once_with(
        optimization._williams_assembly.xy,
        solver="direct",
    )
    assert result.solver == "direct"
    assert "diagnostics" not in result


def test_williams_z_direct_route_uses_centralized_solver() -> None:
    """Out-of-plane Williams fitting should route its prepared fixed system."""
    optimization = _make_optimization()
    with mock.patch.object(
        optimization_module,
        "solve_coefficient_fit",
        wraps=solver_module.solve_coefficient_fit,
    ) as solve:
        result = optimization.optimize_williams_displacements_z(solver="direct")

    solve.assert_called_once_with(
        optimization._williams_assembly.z,
        solver="direct",
    )
    assert result.solver == "direct"
    assert "diagnostics" not in result


def test_iterative_route_forwards_method_and_copied_initialization() -> None:
    """Iterative fitting should receive the selected system and copied controls."""
    optimization = _make_optimization()
    initial = np.linspace(0.1, 0.6, 2 * len(optimization.terms))
    expected_initial = initial.copy()
    with mock.patch.object(
        optimization_module,
        "solve_coefficient_fit",
        wraps=solver_module.solve_coefficient_fit,
    ) as solve:
        result = optimization.optimize_williams_displacements_xy(
            method="trf",
            init_coeffs=initial,
            solver="iterative",
        )

    assert solve.call_count == 1
    system, = solve.call_args.args
    assert system is optimization._williams_assembly.xy
    assert solve.call_args.kwargs["solver"] == "iterative"
    assert solve.call_args.kwargs["method"] == "trf"
    forwarded_initial = solve.call_args.kwargs["init_coeffs"]
    assert forwarded_initial is not initial
    np.testing.assert_array_equal(forwarded_initial, expected_initial)
    assert "residuals" not in solve.call_args.kwargs
    assert "jacobian" not in solve.call_args.kwargs
    np.testing.assert_array_equal(initial, expected_initial)
    assert result.solver == "iterative"


@pytest.mark.parametrize(
    (
        "optimizer_name",
        "assembly_name",
        "system_name",
        "residual_name",
        "jacobian_name",
        "coefficient_count",
    ),
    (
        (
            "optimize_cjp_displacements_modeI",
            "_cjp_assembly",
            "mode_i",
            "residuals_cjp_displacements_modeI",
            "jacobian_cjp_displacements_modeI",
            5,
        ),
        (
            "optimize_cjp_displacements_mixedmode",
            "_cjp_assembly",
            "mixed_mode",
            "residuals_cjp_displacements_mixedmode",
            "jacobian_cjp_displacements_mixedmode",
            5,
        ),
        (
            "optimize_williams_displacements_xy",
            "_williams_assembly",
            "xy",
            "residuals_williams_displacements",
            "jacobian_williams_displacements",
            6,
        ),
        (
            "optimize_williams_displacements_z",
            "_williams_assembly",
            "z",
            "residuals_williams_displacements_z",
            "jacobian_williams_displacements_z",
            3,
        ),
    ),
)
def test_legacy_routes_forward_their_exact_systems_callbacks_and_controls(
    optimizer_name: str,
    assembly_name: str,
    system_name: str,
    residual_name: str,
    jacobian_name: str,
    coefficient_count: int,
) -> None:
    """Every legacy facade should forward its matching system and callbacks."""
    optimization = _make_optimization()
    initial = np.linspace(0.1, 0.5, coefficient_count)
    expected_initial = initial.copy()
    optimizer = getattr(optimization, optimizer_name)
    expected_system = getattr(getattr(optimization, assembly_name), system_name)
    expected_residual = getattr(optimization, residual_name)
    expected_jacobian = getattr(optimization, jacobian_name)
    with mock.patch.object(
        optimization_module,
        "solve_coefficient_fit",
        wraps=solver_module.solve_coefficient_fit,
    ) as solve:
        result = optimizer(
            method="trf",
            init_coeffs=initial,
            solver="legacy",
        )

    assert solve.call_count == 1
    system, = solve.call_args.args
    assert system is expected_system
    assert solve.call_args.kwargs["solver"] == "legacy"
    assert solve.call_args.kwargs["method"] == "trf"
    forwarded_initial = solve.call_args.kwargs["init_coeffs"]
    assert forwarded_initial is not initial
    np.testing.assert_array_equal(forwarded_initial, expected_initial)
    assert solve.call_args.kwargs["residuals"] == expected_residual
    assert solve.call_args.kwargs["jacobian"] == expected_jacobian
    np.testing.assert_array_equal(initial, expected_initial)
    assert result.solver == "legacy"


def test_facade_rejects_an_unsupported_solver_route() -> None:
    """The public facade should preserve centralized route validation."""
    optimization = _make_optimization()

    with pytest.raises(ValueError, match="Unsupported Solver Route 'unsupported'"):
        optimization.optimize_williams_displacements_z(solver="unsupported")


@pytest.mark.parametrize("formulation,n_coefficients", [
    ("cjp_displacements_modeI", 5),
    ("cjp_displacements_mixedmode", 5),
    ("williams_displacements_xy", 6),
    ("williams_displacements_z", 3),
])
@pytest.mark.parametrize("solver", ["direct", "iterative", "legacy"])
def test_typed_fits_match_public_results_and_keep_owned_arrays(
    formulation, n_coefficients, solver,
):
    optimization = _make_optimization()
    arguments = {"solver": solver, "init_coeffs": np.zeros(n_coefficients)}
    fit = getattr(optimization, "_fit_" + formulation)(**arguments)
    public = getattr(optimization, "optimize_" + formulation)(**arguments)

    for name in ("solver", "cost", "rank", "success", "message", "status", "nfev", "njev"):
        assert getattr(public, name) == getattr(fit, name)
    for public_name, fit_name in (
        ("x", "coefficients"), ("fun", "residual"),
        ("jac", "jacobian"), ("singular_values", "singular_values"),
    ):
        actual, expected = getattr(public, public_name), getattr(fit, fit_name)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)
            assert actual.flags.writeable
            assert not expected.flags.writeable
            retained = expected.copy()
            actual.fill(123.0)
            np.testing.assert_array_equal(expected, retained)
