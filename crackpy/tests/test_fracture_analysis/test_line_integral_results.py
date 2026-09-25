"""Behavior tests for Contour-Wise Line-Integral Results."""

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
)
from crackpy.fracture_analysis.line_integrals import (
    ContourWiseLineIntegralResult,
    IntegrationContourResultGeometry,
    LineIntegralQuantities,
)
from crackpy.fracture_analysis.line_integrals._compatibility import (
    mutable_integration_points,
    mutable_path_result,
    mutable_path_size,
    mutable_williams_a_n,
    mutable_williams_b_n,
    mutable_williams_coefficients,
)


def _quantities(*, t_stress_chen: float | None = 4.0) -> LineIntegralQuantities:
    return LineIntegralQuantities(
        j_integral=0.0,
        sif_k_j=1.0,
        sif_k_i=2.0,
        sif_k_ii=3.0,
        t_stress_chen=t_stress_chen,
        t_stress_sdm=5.0,
        t_stress_int=6.0,
        decomp_j_integral_i=7.0,
        decomp_j_integral_ii=8.0,
        decomp_j_integral_iii=9.0,
        decomp_j_integral_k_i=10.0,
        decomp_j_integral_k_ii=11.0,
        decomp_j_integral_k_iii=12.0,
    )


def test_result_owns_geometry_and_retains_williams_coefficient_contract():
    source_points = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    source_terms = np.asarray([1, 2])
    source_a_n = np.asarray([5.0, 6.0])
    source_b_n = np.asarray([7.0, 8.0])
    geometry = IntegrationContourResultGeometry(
        size_left=-1.0,
        size_right=2.0,
        size_bottom=-3.0,
        size_top=4.0,
        integration_points=source_points,
        number_of_nodes=10,
        tick_size=0.5,
    )
    coefficients = WilliamsInPlaneCoefficients(
        terms=source_terms,
        a_n=source_a_n,
        b_n=source_b_n,
    )
    result = ContourWiseLineIntegralResult(
        geometry=geometry,
        quantities=_quantities(),
        williams_coefficients=coefficients,
    )

    source_points[0, 0] = 99.0
    source_terms[0] = 99
    source_a_n[0] = 99.0
    source_b_n[0] = 99.0

    assert result.geometry.integration_points == ((1.0, 2.0), (3.0, 4.0))
    assert result.williams_coefficients is coefficients
    assert result.williams_coefficients == WilliamsInPlaneCoefficients(
        terms=(1, 2),
        a_n=(5.0, 6.0),
        b_n=(7.0, 8.0),
    )


def test_result_construction_preserves_precomputed_values_without_derivation():
    quantities = _quantities()
    result = ContourWiseLineIntegralResult(
        geometry=IntegrationContourResultGeometry(
            size_left=-1.0,
            size_right=1.0,
            size_bottom=-1.0,
            size_top=1.0,
            integration_points=(),
            number_of_nodes=0,
            tick_size=0.25,
        ),
        quantities=quantities,
        williams_coefficients=None,
    )

    assert result.quantities is quantities
    assert result.williams_coefficients is None


def _result_with_terms(
    terms: tuple[int, ...] | None,
    *,
    t_stress_chen: float | None,
) -> ContourWiseLineIntegralResult:
    coefficients = None
    if terms is not None:
        coefficients = WilliamsInPlaneCoefficients(
            terms=terms,
            a_n=tuple(10.0 + term for term in terms),
            b_n=tuple(20.0 + term for term in terms),
        )
    return ContourWiseLineIntegralResult(
        geometry=IntegrationContourResultGeometry(
            size_left=-1.0,
            size_right=2.0,
            size_bottom=-3.0,
            size_top=4.0,
            integration_points=((1.0, 2.0), (3.0, 4.0)),
            number_of_nodes=10,
            tick_size=0.5,
        ),
        quantities=_quantities(t_stress_chen=t_stress_chen),
        williams_coefficients=coefficients,
    )


def test_mutable_projection_preserves_exact_shapes_and_order():
    result = _result_with_terms((1, 3), t_stress_chen=42.0)
    expected_quantities = [float(index) for index in range(13)]
    expected_quantities[4] = 42.0

    assert mutable_path_result(result) == expected_quantities
    assert mutable_path_size(result) == [-1.0, 2.0, -3.0, 4.0]
    assert mutable_williams_a_n(result) == [11.0, 13.0]
    assert mutable_williams_b_n(result) == [21.0, 23.0]
    assert mutable_williams_coefficients(result) == [
        [1, 11.0, 21.0],
        [3, 13.0, 23.0],
    ]
    assert mutable_integration_points(result) == [[1.0, 3.0], [2.0, 4.0]]


def test_williams_projection_distinguishes_disabled_empty_and_requested_terms():
    disabled = _result_with_terms(None, t_stress_chen=None)
    excluding_two = _result_with_terms((1, 3), t_stress_chen=17.0)
    including_two = _result_with_terms((1, 2), t_stress_chen=17.0)

    assert mutable_williams_coefficients(disabled) == []
    assert mutable_path_result(disabled)[4] is None
    assert mutable_williams_coefficients(excluding_two)[0][0] == 1
    assert mutable_path_result(excluding_two)[4] == 17.0
    assert mutable_williams_coefficients(including_two)[1][0] == 2
    assert mutable_path_result(including_two)[4] == 17.0
