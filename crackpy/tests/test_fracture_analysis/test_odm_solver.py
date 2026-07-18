"""Numerical Solver Route evidence for owned ODM fit results, GELSS parity,
iterative behavior, legacy callbacks, and SciPy facade adaptation.
"""

from unittest import mock

import numpy as np
import pytest
from scipy import linalg, optimize

import crackpy.fracture_analysis.odm.solvers as solver_module
from crackpy.fracture_analysis.odm.assembly import LinearSystem
from crackpy.fracture_analysis.odm.results import CoefficientFitResult
from crackpy.fracture_analysis.odm.solvers import (
    coefficient_fit_from_optimize_result,
    solve_coefficient_fit,
    solve_direct,
    solve_iterative,
    solve_legacy,
    to_optimize_result,
)


def _direct_oracle(system: LinearSystem) -> dict[str, object]:
    """Calculate normalized direct-route fields with a test-local GELSS call."""
    coefficients, _, rank, singular_values = linalg.lstsq(
        system.matrix,
        system.target,
        lapack_driver="gelss",
    )
    residual = system.matrix @ coefficients - system.target
    return {
        "solver": "direct",
        "coefficients": coefficients,
        "residual": residual,
        "cost": 0.5 * np.dot(residual, residual),
        "jacobian": system.matrix.copy(),
        "rank": rank,
        "singular_values": singular_values,
        "success": True,
        "message": "Solved by direct linear least squares.",
        "status": 1,
        "nfev": 1,
        "njev": 1,
    }


def _assert_fit_matches_fields(
    actual: CoefficientFitResult,
    expected: dict[str, object],
) -> None:
    """Assert every normalized coefficient-fit field against expected values."""
    np.testing.assert_allclose(actual.coefficients, expected["coefficients"])
    np.testing.assert_allclose(actual.residual, expected["residual"])
    np.testing.assert_allclose(actual.jacobian, expected["jacobian"])
    np.testing.assert_allclose(actual.singular_values, expected["singular_values"])
    assert actual.solver == expected["solver"]
    assert actual.cost == pytest.approx(expected["cost"])
    assert actual.rank == expected["rank"]
    assert actual.success is expected["success"]
    assert actual.message == expected["message"]
    assert actual.status == expected["status"]
    assert actual.nfev == expected["nfev"]
    assert actual.njev == expected["njev"]


def test_coefficient_fit_result_can_be_constructed() -> None:
    result = CoefficientFitResult(
        solver="direct",
        coefficients=np.array([1.0, 2.0]),
        residual=np.array([0.25]),
        cost=0.03125,
        jacobian=np.array([[1.0, 2.0]]),
        rank=1,
        singular_values=np.array([3.0]),
        success=True,
        message="complete",
        status=1,
        nfev=1,
        njev=1,
    )

    assert result.solver == "direct"
    np.testing.assert_array_equal(result.coefficients, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result.residual, np.array([0.25]))
    np.testing.assert_array_equal(result.jacobian, np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(result.singular_values, np.array([3.0]))


def test_coefficient_fit_result_owns_immutable_array_storage() -> None:
    coefficients = np.array([1.0, 2.0])
    residual = np.array([0.25, -0.5])
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0]])
    singular_values = np.array([2.0, 1.0])
    result = CoefficientFitResult(
        solver="direct",
        coefficients=coefficients,
        residual=residual,
        cost=0.15625,
        jacobian=jacobian,
        rank=2,
        singular_values=singular_values,
        success=True,
        message="complete",
        status=1,
        nfev=1,
        njev=1,
    )

    expected_arrays = tuple(
        array.copy() for array in (coefficients, residual, jacobian, singular_values)
    )
    coefficients.fill(10.0)
    residual.fill(10.0)
    jacobian.fill(10.0)
    singular_values.fill(10.0)

    authoritative_arrays = (
        result.coefficients,
        result.residual,
        result.jacobian,
        result.singular_values,
    )
    for authoritative, expected in zip(authoritative_arrays, expected_arrays, strict=True):
        assert authoritative is not None
        np.testing.assert_array_equal(authoritative, expected)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            authoritative.flags.writeable = True


def test_optimize_result_adapters_round_trip_the_normalized_fields() -> None:
    fit = CoefficientFitResult(
        solver="iterative",
        coefficients=np.array([1.0, 2.0]),
        residual=np.array([0.25, -0.5]),
        cost=0.15625,
        jacobian=np.array([[1.0, 0.0], [0.0, 1.0]]),
        rank=2,
        singular_values=np.array([2.0, 1.0]),
        success=True,
        message="complete",
        status=2,
        nfev=3,
        njev=2,
    )

    facade = to_optimize_result(fit)

    assert set(facade) == {
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
    assert "diagnostics" not in facade
    assert facade.x.flags.writeable
    assert facade.fun.flags.writeable
    assert facade.jac.flags.writeable
    assert facade.singular_values.flags.writeable

    round_tripped = coefficient_fit_from_optimize_result(facade)
    facade.x.fill(10.0)
    facade.fun.fill(10.0)
    facade.jac.fill(10.0)
    facade.singular_values.fill(10.0)

    assert round_tripped.solver == fit.solver
    np.testing.assert_array_equal(fit.coefficients, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(fit.residual, np.array([0.25, -0.5]))
    np.testing.assert_array_equal(fit.jacobian, np.eye(2))
    np.testing.assert_array_equal(fit.singular_values, np.array([2.0, 1.0]))
    np.testing.assert_array_equal(round_tripped.coefficients, fit.coefficients)
    np.testing.assert_array_equal(round_tripped.residual, fit.residual)
    np.testing.assert_array_equal(round_tripped.jacobian, fit.jacobian)
    assert round_tripped.rank == fit.rank
    np.testing.assert_array_equal(round_tripped.singular_values, fit.singular_values)
    assert round_tripped.cost == fit.cost
    assert round_tripped.success is fit.success
    assert round_tripped.message == fit.message
    assert round_tripped.status == fit.status
    assert round_tripped.nfev == fit.nfev
    assert round_tripped.njev == fit.njev


def test_direct_route_matches_full_column_rank_gelss() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )
    expected = _direct_oracle(system)

    with mock.patch("scipy.linalg.lstsq", wraps=linalg.lstsq) as least_squares:
        actual = solve_direct(system)

    assert least_squares.call_args.kwargs["lapack_driver"] == "gelss"
    _assert_fit_matches_fields(actual, expected)
    assert actual.jacobian is not system.matrix


@pytest.mark.parametrize(
    "system",
    [
        pytest.param(
            LinearSystem(matrix=np.empty((0, 2)), target=np.empty(0)),
            id="empty",
        ),
        pytest.param(
            LinearSystem(
                matrix=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
                target=np.array([2.0, 3.0]),
            ),
            id="underdetermined",
        ),
        pytest.param(
            LinearSystem(
                matrix=np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
                target=np.array([1.0, 2.0, 3.0]),
            ),
            id="rank-deficient",
        ),
    ],
)
def test_direct_route_preserves_gelss_edge_system_results(
    system: LinearSystem,
) -> None:
    expected = _direct_oracle(system)

    actual = solve_direct(system)

    _assert_fit_matches_fields(actual, expected)
    assert actual.jacobian is not system.matrix


def test_iterative_route_uses_fixed_residual_exact_jacobian_and_zero_initialization() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )
    expected = optimize.least_squares(
        lambda coefficients: system.matrix @ coefficients - system.target,
        x0=np.zeros(system.matrix.shape[1]),
        jac=lambda _: system.matrix,
        method="lm",
    )

    with mock.patch("scipy.optimize.least_squares", wraps=optimize.least_squares) as least_squares:
        actual = solve_iterative(system)

    residuals = least_squares.call_args.args[0]
    np.testing.assert_array_equal(
        residuals(np.array([0.5, -0.25])),
        system.matrix @ np.array([0.5, -0.25]) - system.target,
    )
    np.testing.assert_array_equal(
        least_squares.call_args.kwargs["jac"](np.array([0.5, -0.25])),
        system.matrix,
    )
    np.testing.assert_array_equal(
        least_squares.call_args.kwargs["x0"],
        np.zeros(system.matrix.shape[1]),
    )
    assert least_squares.call_args.kwargs["method"] == "lm"
    assert actual.solver == "iterative"
    np.testing.assert_allclose(actual.coefficients, expected.x)
    np.testing.assert_allclose(actual.residual, expected.fun)
    np.testing.assert_allclose(actual.jacobian, expected.jac)
    assert actual.cost == pytest.approx(expected.cost)
    assert actual.rank is None
    assert actual.singular_values is None
    assert actual.success is expected.success
    assert actual.message == expected.message
    assert actual.status == expected.status
    assert actual.nfev == expected.nfev
    assert actual.njev == expected.njev


def test_iterative_route_copies_explicit_initialization() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )
    initial = np.array([0.25, -0.5])
    expected_initial = initial.copy()

    with mock.patch("scipy.optimize.least_squares", wraps=optimize.least_squares) as least_squares:
        solve_iterative(system, init_coeffs=initial)

    passed_initial = least_squares.call_args.kwargs["x0"]
    np.testing.assert_array_equal(passed_initial, expected_initial)
    assert passed_initial is not initial
    np.testing.assert_array_equal(initial, expected_initial)


def test_iterative_route_forwards_method() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )

    with mock.patch("scipy.optimize.least_squares", wraps=optimize.least_squares) as least_squares:
        solve_iterative(system, method="trf")

    assert least_squares.call_args.kwargs["method"] == "trf"


def test_direct_and_iterative_routes_agree_for_full_column_rank_system() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )

    direct = solve_direct(system)
    iterative = solve_iterative(system, init_coeffs=np.array([0.1, 0.5]))

    np.testing.assert_allclose(iterative.coefficients, direct.coefficients, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(iterative.residual, direct.residual, rtol=1e-6, atol=1e-9)
    assert iterative.cost == pytest.approx(direct.cost, abs=1e-12)


def test_legacy_route_uses_callbacks_and_random_default_initialization() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    target = np.array([1.0, 4.0, 3.0])
    residuals = mock.Mock(side_effect=lambda coefficients: matrix @ coefficients - target)
    jacobian = mock.Mock(side_effect=lambda _: matrix)
    random_initial = np.array([0.25, -0.5])

    with mock.patch("numpy.random.rand", return_value=random_initial.copy()) as random, mock.patch(
        "scipy.optimize.least_squares",
        wraps=optimize.least_squares,
    ) as least_squares:
        actual = solve_legacy(
            residuals=residuals,
            jacobian=jacobian,
            n_coefficients=2,
        )

    random.assert_called_once_with(2)
    assert least_squares.call_args.kwargs["fun"] is residuals
    assert least_squares.call_args.kwargs["jac"] is jacobian
    np.testing.assert_array_equal(least_squares.call_args.kwargs["x0"], random_initial)
    assert residuals.call_count > 0
    assert jacobian.call_count > 0
    assert actual.solver == "legacy"
    np.testing.assert_allclose(actual.residual, residuals(actual.coefficients))
    np.testing.assert_allclose(actual.jacobian, matrix)
    assert actual.rank is None
    assert actual.singular_values is None


def test_legacy_route_copies_explicit_initialization() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    target = np.array([1.0, 4.0, 3.0])
    initial = np.array([0.25, -0.5])
    expected_initial = initial.copy()

    with mock.patch("numpy.random.rand", return_value=np.zeros(2)) as random, mock.patch(
        "scipy.optimize.least_squares",
        wraps=optimize.least_squares,
    ) as least_squares:
        solve_legacy(
            residuals=lambda coefficients: matrix @ coefficients - target,
            jacobian=lambda _: matrix,
            n_coefficients=2,
            init_coeffs=initial,
        )

    random.assert_not_called()
    passed_initial = least_squares.call_args.kwargs["x0"]
    np.testing.assert_array_equal(passed_initial, expected_initial)
    assert passed_initial is not initial
    np.testing.assert_array_equal(initial, expected_initial)


def test_legacy_route_forwards_method() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    target = np.array([1.0, 4.0, 3.0])

    with mock.patch("scipy.optimize.least_squares", wraps=optimize.least_squares) as least_squares:
        solve_legacy(
            residuals=lambda coefficients: matrix @ coefficients - target,
            jacobian=lambda _: matrix,
            n_coefficients=2,
            method="trf",
        )

    assert least_squares.call_args.kwargs["method"] == "trf"


def test_coefficient_fit_selects_direct_route() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )
    expected = solve_direct(system)

    with mock.patch.object(solver_module, "solve_direct", return_value=expected) as direct:
        actual = solve_coefficient_fit(system, solver="direct")

    assert actual is expected
    direct.assert_called_once_with(system)


def test_coefficient_fit_selects_iterative_route() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )
    initial = np.array([0.25, -0.5])
    expected = solve_iterative(system)

    with mock.patch.object(
        solver_module,
        "solve_iterative",
        return_value=expected,
    ) as iterative:
        actual = solve_coefficient_fit(
            system,
            solver="iterative",
            method="trf",
            init_coeffs=initial,
        )

    assert actual is expected
    iterative.assert_called_once_with(system, method="trf", init_coeffs=initial)


def test_coefficient_fit_selects_legacy_route() -> None:
    system = LinearSystem(
        matrix=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        target=np.array([1.0, 4.0, 3.0]),
    )

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        return system.matrix @ coefficients - system.target

    def jacobian(_: np.ndarray) -> np.ndarray:
        return system.matrix

    initial = np.array([0.25, -0.5])
    expected = solve_legacy(
        residuals=residuals,
        jacobian=jacobian,
        n_coefficients=2,
        init_coeffs=initial,
    )

    with mock.patch.object(
        solver_module,
        "solve_legacy",
        return_value=expected,
    ) as legacy:
        actual = solve_coefficient_fit(
            system,
            solver="legacy",
            method="trf",
            init_coeffs=initial,
            residuals=residuals,
            jacobian=jacobian,
        )

    assert actual is expected
    legacy.assert_called_once_with(
        residuals=residuals,
        jacobian=jacobian,
        n_coefficients=2,
        method="trf",
        init_coeffs=initial,
    )


def test_coefficient_fit_rejects_unsupported_route() -> None:
    system = LinearSystem(matrix=np.eye(2), target=np.ones(2))

    with pytest.raises(
        ValueError,
        match="Unsupported Solver Route 'unsupported'. Expected direct, iterative, or legacy.",
    ):
        solve_coefficient_fit(system, solver="unsupported")


@pytest.mark.parametrize(
    ("residuals", "jacobian"),
    [
        (None, lambda _: np.eye(2)),
        (lambda coefficients: coefficients - 1.0, None),
        (None, None),
    ],
)
def test_coefficient_fit_requires_both_legacy_callbacks(
    residuals: object,
    jacobian: object,
) -> None:
    system = LinearSystem(matrix=np.eye(2), target=np.ones(2))

    with pytest.raises(
        ValueError,
        match="Legacy Solver requires residual and Jacobian callbacks.",
    ):
        solve_coefficient_fit(
            system,
            solver="legacy",
            residuals=residuals,
            jacobian=jacobian,
        )
