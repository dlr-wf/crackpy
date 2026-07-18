"""ODM result-completion evidence for typed scientific payloads, failure states,
skips, and legacy compatibility projections.
"""

from dataclasses import astuple

import numpy as np
import pytest

from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    CjpMixedModeCoefficients,
    CjpMixedModeQuantities,
    CjpModeICoefficients,
    CjpModeIQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp.quantities import (
    derive_cjp_mixed_mode_fracture_quantities,
    derive_cjp_mode_i_fracture_quantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    WilliamsInPlaneCoefficients,
    WilliamsInPlaneQuantities,
    WilliamsOutOfPlaneCoefficients,
    WilliamsOutOfPlaneQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.quantities import (
    derive_williams_in_plane_fracture_quantities,
    derive_williams_out_of_plane_fracture_quantities,
)
from crackpy.fracture_analysis.odm._compatibility import (
    _project_cjp_mixed_mode_compatibility,
    _project_cjp_mode_i_compatibility,
    _project_williams_compatibility,
)
from crackpy.fracture_analysis.odm.results import CoefficientFitResult
from crackpy.fracture_analysis.odm.runners import (
    _build_cjp_mixed_mode_odm_result,
    _build_cjp_mode_i_odm_result,
    _build_williams_in_plane_odm_result,
    _build_williams_out_of_plane_odm_result,
)


def _fit(
    coefficients: list[float] | np.ndarray,
    *,
    cost: float = 2.5,
    success: bool = True,
) -> CoefficientFitResult:
    """Build one immutable coefficient-fit fixture."""
    coefficient_array = np.asarray(coefficients)
    return CoefficientFitResult(
        solver="direct",
        coefficients=coefficient_array,
        residual=np.asarray([1.0, -2.0]),
        cost=cost,
        jacobian=np.ones((2, coefficient_array.size)),
        rank=min(2, coefficient_array.size),
        singular_values=np.ones(min(2, coefficient_array.size)),
        success=success,
        message="fixture",
        status=1 if success else 0,
        nfev=1,
        njev=1,
    )


def _assert_dataclass_nan(payload) -> None:
    """Assert that every scalar or coefficient sequence in a payload is NaN."""
    for value in astuple(payload):
        if isinstance(value, tuple):
            if value and isinstance(value[0], int):
                continue
            assert all(np.isnan(item) for item in value)
        else:
            assert np.isnan(value)


def test_cjp_mode_i_builder_derives_typed_completed_payloads() -> None:
    fit = _fit([1.0, 2.0, 3.0, 4.0, 5.0], cost=1.25)

    result = _build_cjp_mode_i_odm_result(fit)
    expected = derive_cjp_mode_i_fracture_quantities(fit.coefficients)

    assert result.status == "completed"
    assert result.coefficient_fit is fit
    assert result.coefficients == CjpModeICoefficients(1.0, 2.0, 3.0, 4.0, 5.0)
    assert result.quantities == CjpModeIQuantities(*expected)
    assert result.cost == 1.25


def test_cjp_mixed_builder_derives_typed_completed_payloads() -> None:
    fit = _fit([1.0, 2.0, 3.0, 4.0, 5.0], cost=1.25)

    result = _build_cjp_mixed_mode_odm_result(fit)
    expected = derive_cjp_mixed_mode_fracture_quantities(fit.coefficients)

    assert result.status == "completed"
    assert result.coefficient_fit is fit
    assert result.coefficients == CjpMixedModeCoefficients(1.0, 2.0, 3.0, 4.0, 5.0)
    assert result.quantities == CjpMixedModeQuantities(*expected)
    assert result.cost == 1.25


@pytest.mark.parametrize(
    ("builder", "coefficient_type", "quantity_type"),
    [
        (_build_cjp_mode_i_odm_result, CjpModeICoefficients, CjpModeIQuantities),
        (
            _build_cjp_mixed_mode_odm_result,
            CjpMixedModeCoefficients,
            CjpMixedModeQuantities,
        ),
    ],
)
@pytest.mark.parametrize("returned_fit", [False, True])
def test_cjp_builders_derive_typed_failed_nan_payloads(
    builder,
    coefficient_type,
    quantity_type,
    returned_fit: bool,
) -> None:
    fit = _fit(np.arange(5.0), success=False) if returned_fit else None

    result = builder(fit)

    assert result.status == "failed"
    assert result.coefficient_fit is fit
    assert isinstance(result.coefficients, coefficient_type)
    assert isinstance(result.quantities, quantity_type)
    _assert_dataclass_nan(result.coefficients)
    _assert_dataclass_nan(result.quantities)
    assert np.isnan(result.cost)


def test_williams_in_plane_builder_derives_typed_completed_payloads() -> None:
    terms = (-1, 1, 2)
    fit = _fit([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], cost=1.5)

    result = _build_williams_in_plane_odm_result(terms, fit)
    expected = derive_williams_in_plane_fracture_quantities(
        terms,
        result.coefficients.a_n,
        result.coefficients.b_n,
    )

    assert result.status == "completed"
    assert result.coefficient_fit is fit
    assert result.coefficients == WilliamsInPlaneCoefficients(
        terms=terms,
        a_n=(10.0, 20.0, 30.0),
        b_n=(40.0, 50.0, 60.0),
    )
    assert result.quantities == WilliamsInPlaneQuantities(*expected)
    assert result.cost == 1.5


def test_williams_out_of_plane_builder_derives_typed_completed_payloads() -> None:
    terms = (-1, 1, 2)
    fit = _fit([70.0, 80.0, 90.0], cost=0.5)

    result = _build_williams_out_of_plane_odm_result(terms, fit, skipped=False)
    expected = derive_williams_out_of_plane_fracture_quantities(
        terms,
        result.coefficients.c_n,
    )

    assert result.status == "completed"
    assert result.coefficient_fit is fit
    assert result.coefficients == WilliamsOutOfPlaneCoefficients(
        terms=terms,
        c_n=(70.0, 80.0, 90.0),
    )
    assert result.quantities == WilliamsOutOfPlaneQuantities(*expected)
    assert result.cost == 0.5


def test_williams_builders_derive_each_supported_quantity_independently() -> None:
    term_one_fit = _fit([10.0, 20.0, 30.0, 40.0], cost=1.5)
    term_two_fit = _fit([30.0, 20.0, 60.0, 40.0], cost=1.25)
    out_of_plane_fit = _fit([70.0, 80.0], cost=0.5)

    term_one = _build_williams_in_plane_odm_result((1, 3), term_one_fit)
    term_two = _build_williams_in_plane_odm_result((2, 3), term_two_fit)
    out_of_plane = _build_williams_out_of_plane_odm_result(
        (2, 3),
        out_of_plane_fit,
        skipped=False,
    )

    assert term_one.status == "completed"
    assert term_one.coefficient_fit is term_one_fit
    assert term_one.coefficients == WilliamsInPlaneCoefficients(
        terms=(1, 3),
        a_n=(10.0, 20.0),
        b_n=(30.0, 40.0),
    )
    assert term_one.quantities.k_i == pytest.approx(0.7926654595212022)
    assert term_one.quantities.k_ii == pytest.approx(-2.3779963785636067)
    assert np.isnan(term_one.quantities.t_stress)
    assert term_one.cost == 1.5

    assert term_two.status == "completed"
    assert term_two.coefficient_fit is term_two_fit
    assert np.isnan(term_two.quantities.k_i)
    assert np.isnan(term_two.quantities.k_ii)
    assert term_two.quantities.t_stress == 120.0
    assert term_two.cost == 1.25

    assert out_of_plane.status == "completed"
    assert out_of_plane.coefficient_fit is out_of_plane_fit
    assert out_of_plane.coefficients == WilliamsOutOfPlaneCoefficients(
        terms=(2, 3),
        c_n=(70.0, 80.0),
    )
    _assert_dataclass_nan(out_of_plane.quantities)
    assert out_of_plane.cost == 0.5

    for result, expected in (
        (term_one, (0.7926654595212022, -2.3779963785636067, np.nan)),
        (term_two, (np.nan, np.nan, 120.0)),
    ):
        _, _, _, _, legacy = _project_williams_compatibility(result, out_of_plane)
        assert legacy["K_I"] == pytest.approx(expected[0], nan_ok=True)
        assert legacy["K_II"] == pytest.approx(expected[1], nan_ok=True)
        assert legacy["T"] == pytest.approx(expected[2], nan_ok=True)


@pytest.mark.parametrize(
    ("builder", "coefficient_count", "expected_count"),
    [
        (_build_cjp_mode_i_odm_result, 4, 5),
        (_build_cjp_mixed_mode_odm_result, 6, 5),
        (
            lambda fit: _build_williams_in_plane_odm_result((-1, 1, 2), fit),
            5,
            6,
        ),
        (
            lambda fit: _build_williams_out_of_plane_odm_result(
                (-1, 1, 2),
                fit,
                skipped=False,
            ),
            4,
            3,
        ),
    ],
)
@pytest.mark.parametrize("success", [True, False])
def test_builders_reject_fits_with_wrong_coefficient_count(
    builder,
    coefficient_count: int,
    expected_count: int,
    success: bool,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"requires exactly {expected_count} coefficients",
    ):
        builder(_fit(np.arange(float(coefficient_count)), success=success))


@pytest.mark.parametrize("returned_fit", [False, True])
def test_williams_builders_derive_independent_failed_nan_payloads(
    returned_fit: bool,
) -> None:
    in_plane_fit = _fit(np.arange(6.0), success=False) if returned_fit else None
    out_of_plane_fit = _fit(np.arange(3.0), success=False) if returned_fit else None

    in_plane = _build_williams_in_plane_odm_result(
        (-1, 1, 2),
        in_plane_fit,
    )
    out_of_plane = _build_williams_out_of_plane_odm_result(
        (-1, 1, 2),
        out_of_plane_fit,
        skipped=False,
    )

    assert in_plane.status == "failed"
    assert in_plane.coefficient_fit is in_plane_fit
    assert isinstance(in_plane.coefficients, WilliamsInPlaneCoefficients)
    assert isinstance(in_plane.quantities, WilliamsInPlaneQuantities)
    _assert_dataclass_nan(in_plane.coefficients)
    _assert_dataclass_nan(in_plane.quantities)
    assert out_of_plane.status == "failed"
    assert out_of_plane.coefficient_fit is out_of_plane_fit
    assert isinstance(out_of_plane.coefficients, WilliamsOutOfPlaneCoefficients)
    assert isinstance(out_of_plane.quantities, WilliamsOutOfPlaneQuantities)
    _assert_dataclass_nan(out_of_plane.coefficients)
    _assert_dataclass_nan(out_of_plane.quantities)


def test_williams_out_of_plane_builder_represents_explicit_skip() -> None:
    result = _build_williams_out_of_plane_odm_result(
        (-1, 1, 2),
        None,
        skipped=True,
    )

    assert result.status == "skipped"
    assert result.coefficient_fit is None
    _assert_dataclass_nan(result.coefficients)
    _assert_dataclass_nan(result.quantities)
    assert np.isnan(result.cost)


def test_williams_out_of_plane_builder_rejects_skip_with_fit() -> None:
    with pytest.raises(ValueError, match="skipped ODM result cannot retain"):
        _build_williams_out_of_plane_odm_result(
            (-1, 1, 2),
            _fit(np.arange(3.0)),
            skipped=True,
        )


def test_legacy_projections_preserve_order_shape_and_mutation_isolation() -> None:
    mode_i = _build_cjp_mode_i_odm_result(_fit(np.arange(5.0)))
    mixed = _build_cjp_mixed_mode_odm_result(_fit(np.arange(5.0)))
    in_plane = _build_williams_in_plane_odm_result(
        (-1, 1, 2),
        _fit(np.arange(6.0)),
    )
    out_of_plane = _build_williams_out_of_plane_odm_result(
        (-1, 1, 2),
        _fit(np.arange(3.0)),
        skipped=False,
    )

    mode_i_coefficients, mode_i_quantities = _project_cjp_mode_i_compatibility(
        mode_i
    )
    mixed_coefficients, mixed_quantities = (
        _project_cjp_mixed_mode_compatibility(mixed)
    )
    coefficients, a_n, b_n, c_n, quantities = _project_williams_compatibility(
        in_plane,
        out_of_plane,
    )

    assert list(mode_i_quantities) == ["Error", "K_F", "K_R", "K_S", "T_x", "T_y"]
    assert list(mixed_quantities) == ["Error", "K_F", "K_R", "K_S", "K_II", "T"]
    assert coefficients.shape == (9,)
    assert list(a_n) == list(in_plane.coefficients.terms)
    assert list(b_n) == list(in_plane.coefficients.terms)
    assert list(c_n) == list(out_of_plane.coefficients.terms)
    assert list(quantities) == ["Error_xy", "K_I", "K_II", "T", "Error_z", "K_III"]

    mode_i_coefficients[:] = -1
    mixed_coefficients[:] = -1
    coefficients[:] = -1
    a_n[-1] = -1
    mode_i_quantities["K_F"] = -1
    mixed_quantities["K_F"] = -1
    quantities["K_I"] = -1

    np.testing.assert_array_equal(
        _project_cjp_mode_i_compatibility(mode_i)[0],
        np.arange(5.0),
    )
    np.testing.assert_array_equal(
        _project_cjp_mixed_mode_compatibility(mixed)[0],
        np.arange(5.0),
    )
    np.testing.assert_array_equal(
        _project_williams_compatibility(in_plane, out_of_plane)[0],
        np.r_[np.arange(6.0), np.arange(3.0)],
    )
