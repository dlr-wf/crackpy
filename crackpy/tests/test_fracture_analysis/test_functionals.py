"""Formula-level tests fix the scientific meaning of fracture-mechanics functionals."""

import numpy as np
import pytest

from crackpy.fracture_analysis.functionals import IntegrandTerms
from crackpy.fracture_analysis.functionals.bueckner_chen import (
    bueckner_chen_integral_terms,
    williams_coefficient_from_bueckner_chen_integral,
)
from crackpy.fracture_analysis.functionals.interaction_integral import (
    in_plane_sif_from_interaction_integral,
    interaction_integral_terms,
    t_stress_from_interaction_integral,
    t_stress_interaction_integral_terms,
)
from crackpy.fracture_analysis.functionals.j_integral import (
    in_plane_energy_equivalent_sif_from_j_integral,
    in_plane_j_integral_terms,
    in_plane_sif_magnitude_from_j_integral,
    mode_iii_j_integral_terms,
    mode_iii_sif_magnitude_from_j_integral,
)
from crackpy.fracture_analysis.functionals.stress_difference import (
    t_stress_from_stress_difference,
)
from crackpy.fracture_analysis.line_integrals.quadrature import (
    evaluate_contour_integral,
)


@pytest.fixture
def prepared_interaction_fields() -> dict[str, np.ndarray]:
    """Return one fixed measured-and-auxiliary field example."""
    return {
        "measured_stress": np.array(
            [
                [[2.0, 1.0], [1.0, 3.0]],
                [[4.0, -1.0], [-1.0, 2.0]],
            ]
        ),
        "measured_strain": np.array(
            [
                [[0.5, 0.2], [0.2, 0.4]],
                [[0.3, -0.1], [-0.1, 0.6]],
            ]
        ),
        "measured_displacement_gradient_x": np.array([[0.5, -0.25], [0.75, 1.25]]),
        "auxiliary_stress": np.array(
            [
                [[1.5, -0.5], [-0.5, 2.0]],
                [[2.0, 0.25], [0.25, 1.0]],
            ]
        ),
        "auxiliary_strain": np.array(
            [
                [[0.1, 0.3], [0.3, 0.2]],
                [[0.4, 0.2], [0.2, 0.5]],
            ]
        ),
        "auxiliary_displacement_gradient_x": np.array([[0.1, 0.9], [-0.4, 0.6]]),
        "normals": np.array([[0.6, 0.8], [-0.8, 0.6]]),
        "segment_dy": np.array([0.75, -0.5]),
        "segment_lengths": np.array([2.0, 1.5]),
    }


def test_in_plane_j_integral_terms_and_quadrature_match_worked_example() -> None:
    stress = np.array(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[1.0, 3.0], [3.0, 2.0]],
        ]
    )
    strain = np.array(
        [
            [[0.5, 0.0], [0.0, 0.25]],
            [[2.0, 0.5], [0.5, 1.0]],
        ]
    )
    normals = np.array([[1.0, 0.0], [0.0, 1.0]])
    displacement_gradient_x = np.array([[0.5, 1.5], [2.0, -1.0]])

    integrand_terms = in_plane_j_integral_terms(
        stress,
        strain,
        displacement_gradient_x,
        normals,
    )
    assert isinstance(integrand_terms, IntegrandTerms)

    expected_strain_energy_density = np.array(
        [
            0.5 * (2.0 * 0.5 + 4.0 * 0.25),
            0.5 * (1.0 * 2.0 + 3.0 * 0.5 + 3.0 * 0.5 + 2.0 * 1.0),
        ]
    )
    expected_traction_work_term = np.array(
        [
            (2.0 * 1.0 + 0.0 * 0.0) * 0.5 + (0.0 * 1.0 + 4.0 * 0.0) * 1.5,
            (1.0 * 0.0 + 3.0 * 1.0) * 2.0 + (3.0 * 0.0 + 2.0 * 1.0) * -1.0,
        ]
    )
    segment_dy = np.array([0.5, -1.0])
    segment_lengths = np.array([2.0, 3.0])
    expected_integral = np.sum(
        expected_strain_energy_density * segment_dy
        - expected_traction_work_term * segment_lengths
    )

    np.testing.assert_allclose(
        integrand_terms.integrated_over_dy[0],
        expected_strain_energy_density,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        integrand_terms.subtracted_over_ds[0],
        expected_traction_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=segment_dy,
        segment_lengths=segment_lengths,
    )
    np.testing.assert_allclose(
        result,
        expected_integral,
        rtol=1e-13,
        atol=1e-13,
    )


def test_mode_iii_j_uses_the_established_stress_gradient_expression() -> None:
    integrand_terms = mode_iii_j_integral_terms(
        out_of_plane_displacement_derivative_x=np.array([1.0, 2.0]),
        out_of_plane_displacement_derivative_y=np.array([3.0, 4.0]),
        sigma_xz=np.array([5.0, 6.0]),
        sigma_yz=np.array([7.0, 8.0]),
        contour_normals=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    expected_stress_gradient_contraction = np.array(
        [1.0 * 5.0 + 3.0 * 7.0, 2.0 * 6.0 + 4.0 * 8.0]
    )
    expected_traction_work_term = np.array(
        [
            5.0 * 1.0 * 1.0 + 7.0 * 1.0 * 0.0,
            6.0 * 2.0 * 0.0 + 8.0 * 2.0 * 1.0,
        ]
    )
    segment_dy = np.array([0.5, -0.25])
    segment_lengths = np.array([2.0, 3.0])
    expected_integral = np.sum(
        expected_stress_gradient_contraction * segment_dy
        - expected_traction_work_term * segment_lengths
    )

    np.testing.assert_allclose(
        integrand_terms.integrated_over_dy[0],
        expected_stress_gradient_contraction,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        integrand_terms.subtracted_over_ds[0],
        expected_traction_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=segment_dy,
        segment_lengths=segment_lengths,
    )
    np.testing.assert_allclose(
        result,
        expected_integral,
        rtol=1e-13,
        atol=1e-13,
    )


def test_interaction_integral_matches_worked_reciprocal_example(
    prepared_interaction_fields: dict[str, np.ndarray],
) -> None:
    fields = prepared_interaction_fields

    integrand_terms = interaction_integral_terms(
        fields["measured_stress"],
        fields["measured_displacement_gradient_x"],
        fields["auxiliary_stress"],
        fields["auxiliary_strain"],
        fields["auxiliary_displacement_gradient_x"],
        fields["normals"],
    )

    expected_interaction_strain_energy_density = np.array(
        [
            2.0 * 0.1 + 1.0 * 0.3 + 1.0 * 0.3 + 3.0 * 0.2,
            4.0 * 0.4 + -1.0 * 0.2 + -1.0 * 0.2 + 2.0 * 0.5,
        ]
    )
    expected_measured_traction_work_term = np.array(
        [
            (2.0 * 0.6 + 1.0 * 0.8) * 0.1 + (1.0 * 0.6 + 3.0 * 0.8) * 0.9,
            (4.0 * -0.8 + -1.0 * 0.6) * -0.4 + (-1.0 * -0.8 + 2.0 * 0.6) * 0.6,
        ]
    )
    expected_auxiliary_traction_work_term = np.array(
        [
            (1.5 * 0.6 + -0.5 * 0.8) * 0.5 + (-0.5 * 0.6 + 2.0 * 0.8) * -0.25,
            (2.0 * -0.8 + 0.25 * 0.6) * 0.75 + (0.25 * -0.8 + 1.0 * 0.6) * 1.25,
        ]
    )
    expected_integral = np.sum(
        expected_interaction_strain_energy_density * fields["segment_dy"]
        - expected_measured_traction_work_term * fields["segment_lengths"]
        - expected_auxiliary_traction_work_term * fields["segment_lengths"]
    )

    np.testing.assert_allclose(
        integrand_terms.integrated_over_dy[0],
        expected_interaction_strain_energy_density,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        integrand_terms.subtracted_over_ds[0],
        expected_measured_traction_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        integrand_terms.subtracted_over_ds[1],
        expected_auxiliary_traction_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=fields["segment_dy"],
        segment_lengths=fields["segment_lengths"],
    )
    np.testing.assert_allclose(result, expected_integral, rtol=1e-13, atol=1e-13)


def test_t_stress_interaction_matches_worked_reciprocal_example(
    prepared_interaction_fields: dict[str, np.ndarray],
) -> None:
    fields = prepared_interaction_fields

    integrand_terms = t_stress_interaction_integral_terms(
        fields["measured_stress"],
        fields["measured_strain"],
        fields["measured_displacement_gradient_x"],
        fields["auxiliary_stress"],
        fields["auxiliary_displacement_gradient_x"],
        fields["normals"],
    )

    expected_interaction_strain_energy_density = np.array(
        [
            1.5 * 0.5 + -0.5 * 0.2 + -0.5 * 0.2 + 2.0 * 0.4,
            2.0 * 0.3 + 0.25 * -0.1 + 0.25 * -0.1 + 1.0 * 0.6,
        ]
    )
    expected_measured_traction_work_term = np.array(
        [
            (2.0 * 0.6 + 1.0 * 0.8) * 0.1 + (1.0 * 0.6 + 3.0 * 0.8) * 0.9,
            (4.0 * -0.8 + -1.0 * 0.6) * -0.4 + (-1.0 * -0.8 + 2.0 * 0.6) * 0.6,
        ]
    )
    expected_auxiliary_traction_work_term = np.array(
        [
            (1.5 * 0.6 + -0.5 * 0.8) * 0.5 + (-0.5 * 0.6 + 2.0 * 0.8) * -0.25,
            (2.0 * -0.8 + 0.25 * 0.6) * 0.75 + (0.25 * -0.8 + 1.0 * 0.6) * 1.25,
        ]
    )
    expected_integral = np.sum(
        expected_interaction_strain_energy_density * fields["segment_dy"]
        - expected_measured_traction_work_term * fields["segment_lengths"]
        - expected_auxiliary_traction_work_term * fields["segment_lengths"]
    )

    np.testing.assert_allclose(
        integrand_terms.integrated_over_dy[0],
        expected_interaction_strain_energy_density,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        integrand_terms.subtracted_over_ds[0],
        expected_measured_traction_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        integrand_terms.subtracted_over_ds[1],
        expected_auxiliary_traction_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=fields["segment_dy"],
        segment_lengths=fields["segment_lengths"],
    )
    np.testing.assert_allclose(result, expected_integral, rtol=1e-13, atol=1e-13)


def test_bueckner_chen_term_matches_worked_reciprocal_example(
    prepared_interaction_fields: dict[str, np.ndarray],
) -> None:
    fields = prepared_interaction_fields
    measured_displacement = np.array([[1.0, 2.0], [-0.5, 3.0]])
    auxiliary_displacement = np.array([[0.25, -1.0], [2.0, 0.5]])

    integrand_terms = bueckner_chen_integral_terms(
        fields["measured_stress"],
        measured_displacement,
        fields["auxiliary_stress"],
        auxiliary_displacement,
        fields["normals"],
    )

    expected_reciprocal_work_term = np.array(
        [
            (2.0 * 0.6 + 1.0 * 0.8) * 0.25 + (1.0 * 0.6 + 3.0 * 0.8) * -1.0 - (1.5 * 0.6 + -0.5 * 0.8) * 1.0 - (-0.5 * 0.6 + 2.0 * 0.8) * 2.0,
            (4.0 * -0.8 + -1.0 * 0.6) * 2.0 + (-1.0 * -0.8 + 2.0 * 0.6) * 0.5 - (2.0 * -0.8 + 0.25 * 0.6) * -0.5 - (0.25 * -0.8 + 1.0 * 0.6) * 3.0,
        ]
    )
    expected_integral = np.sum(
        expected_reciprocal_work_term * fields["segment_lengths"]
    )

    assert integrand_terms.integrated_over_dy == ()
    np.testing.assert_allclose(
        integrand_terms.added_over_ds[0],
        expected_reciprocal_work_term,
        rtol=1e-13,
        atol=1e-13,
    )
    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=fields["segment_dy"],
        segment_lengths=fields["segment_lengths"],
    )
    np.testing.assert_allclose(result, expected_integral, rtol=1e-13, atol=1e-13)


def test_nan_integrand_term_propagates_through_contour_quadrature() -> None:
    stress = np.array([[[2.0, 0.0], [0.0, 4.0]], [[np.nan, 0.0], [0.0, 1.0]]])
    strain = np.array([[[0.5, 0.0], [0.0, 0.25]], [[1.0, 0.0], [0.0, 1.0]]])
    displacement_gradient_x = np.array([[0.5, 1.5], [1.0, 1.0]])
    normals = np.array([[0.6, 0.8], [-0.8, 0.6]])

    integrand_terms = in_plane_j_integral_terms(
        stress,
        strain,
        displacement_gradient_x,
        normals,
    )
    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=np.array([0.75, -0.5]),
        segment_lengths=np.array([2.0, 1.5]),
    )

    assert np.isnan(integrand_terms.integrated_over_dy[0][1])
    assert np.isnan(result)


def test_contour_quadrature_preserves_float64_functional_precision_with_float32_geometry() -> None:
    integrand_terms = IntegrandTerms(
        integrated_over_dy=(np.array([1.0], dtype=np.float64),),
        added_over_ds=(np.array([1.0e-8], dtype=np.float64),),
        subtracted_over_ds=(np.array([5.0e-9], dtype=np.float64),),
    )

    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=np.array([1.0], dtype=np.float32),
        segment_lengths=np.array([1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(
        result,
        1.000000005,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )


def test_contour_quadrature_accepts_integer_geometry_with_floating_functional_terms() -> None:
    integrand_terms = IntegrandTerms(
        integrated_over_dy=(np.array([0.5, 1.25]),),
        added_over_ds=(np.array([0.25, -0.5]),),
        subtracted_over_ds=(np.array([0.125, 0.75]),),
    )

    result = evaluate_contour_integral(
        integrand_terms,
        segment_dy=np.array([1, -1]),
        segment_lengths=np.array([2, 3]),
    )

    np.testing.assert_allclose(result, -4.25, rtol=0.0, atol=0.0)


def test_j_to_stress_intensity_mappings_preserve_total_and_modal_policy() -> None:
    expected_total = np.sqrt(np.abs(-4.0) / 1000.0 * 1000.0)
    expected_modal = np.sqrt(4.0 * 1000.0) / np.sqrt(1000.0)
    expected_mode_iii = np.sqrt(10.0 * 125.0 / (1.0 + 0.25)) / np.sqrt(1000.0)

    np.testing.assert_allclose(
        in_plane_energy_equivalent_sif_from_j_integral(
            -4.0,
            youngs_modulus=1000.0,
        ),
        expected_total,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        in_plane_sif_magnitude_from_j_integral(
            4.0,
            youngs_modulus=1000.0,
        ),
        expected_modal,
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.isnan(
        in_plane_sif_magnitude_from_j_integral(
            -4.0,
            youngs_modulus=1000.0,
        )
    )
    np.testing.assert_allclose(
        mode_iii_sif_magnitude_from_j_integral(
            10.0,
            youngs_modulus=125.0,
            poisson_ratio=0.25,
        ),
        expected_mode_iii,
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.isnan(
        mode_iii_sif_magnitude_from_j_integral(
            -10.0,
            youngs_modulus=125.0,
            poisson_ratio=0.25,
        )
    )


@pytest.mark.parametrize(
    ("interaction_integral", "auxiliary_intensity"),
    [
        pytest.param(4.0, 10.0, id="Mode I"),
        pytest.param(-3.0, 2.0, id="Mode II"),
    ],
)
def test_mode_i_and_ii_interaction_mappings_preserve_factor_two_and_units(
    interaction_integral: float,
    auxiliary_intensity: float,
) -> None:
    youngs_modulus = 100.0
    expected_stress_intensity = youngs_modulus / auxiliary_intensity * interaction_integral / 2.0 / np.sqrt(1000.0)

    measured_sif = in_plane_sif_from_interaction_integral(
        interaction_integral,
        youngs_modulus=youngs_modulus,
        auxiliary_sif=auxiliary_intensity,
    )

    np.testing.assert_allclose(
        measured_sif,
        expected_stress_intensity,
        rtol=1e-13,
        atol=1e-13,
    )


def test_t_stress_interaction_mappings_preserve_plane_policy() -> None:
    expected_plane_strain = 100.0 / (1.0 - 0.5**2) * 0.03
    expected_plane_stress = 100.0 * (0.03 + 0.25 * 0.04)

    plane_strain = t_stress_from_interaction_integral(
        0.03,
        youngs_modulus=100.0,
        poisson_ratio=0.5,
        plane_strain=True,
    )
    plane_stress = t_stress_from_interaction_integral(
        0.03,
        youngs_modulus=100.0,
        poisson_ratio=0.25,
        plane_strain=False,
        reference_out_of_plane_strain=0.04,
    )
    np.testing.assert_allclose(
        plane_strain,
        expected_plane_strain,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        plane_stress,
        expected_plane_stress,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    (
        "symmetric_auxiliary_amplitude",
        "antisymmetric_auxiliary_amplitude",
        "term",
        "integral_value",
    ),
    [
        (1.0, 0.0, 1, 2.0),
        (0.0, 2.0, 2, -3.0),
    ],
)
def test_williams_coefficient_mapping_preserves_auxiliary_branch_sign_and_normalization(
    symmetric_auxiliary_amplitude: float,
    antisymmetric_auxiliary_amplitude: float,
    term: int,
    integral_value: float,
) -> None:
    shear_modulus = 10.0
    kappa = 3.0
    auxiliary_amplitude = (
        symmetric_auxiliary_amplitude
        + antisymmetric_auxiliary_amplitude
    )
    expected_coefficient = -shear_modulus / (kappa + 1.0) / auxiliary_amplitude / (np.pi * term * (-1) ** (term + 1)) * integral_value

    coefficient = williams_coefficient_from_bueckner_chen_integral(
        integral_value,
        shear_modulus=shear_modulus,
        kappa=kappa,
        symmetric_auxiliary_amplitude=symmetric_auxiliary_amplitude,
        antisymmetric_auxiliary_amplitude=antisymmetric_auxiliary_amplitude,
        term=term,
    )

    np.testing.assert_allclose(
        coefficient,
        expected_coefficient,
        rtol=1e-13,
        atol=1e-13,
    )


def test_williams_coefficient_mapping_rejects_two_active_auxiliary_branches() -> None:

    with pytest.raises(
        ValueError,
        match="Select either the symmetric or antisymmetric auxiliary eigenfield",
    ):
        williams_coefficient_from_bueckner_chen_integral(
            2.0,
            shear_modulus=10.0,
            kappa=3.0,
            symmetric_auxiliary_amplitude=1.0,
            antisymmetric_auxiliary_amplitude=1.0,
            term=1,
        )


def test_stress_difference_and_nan_propagation() -> None:
    result = t_stress_from_stress_difference(
        np.array([5.0, np.nan]),
        np.array([2.0, 1.0]),
    )
    expected_finite_t_stress = np.array([5.0 - 2.0])
    np.testing.assert_allclose(
        result[:1],
        expected_finite_t_stress,
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.isnan(result[1])


@pytest.mark.parametrize(
    ("function", "citation_key", "locator"),
    [
        (in_plane_j_integral_terms, "rice_1968_j_integral", "equation 1"),
        (
            mode_iii_j_integral_terms,
            "molteno_becker_2015_j_integral_decomposition",
            "equations 8--11",
        ),
        (
            in_plane_energy_equivalent_sif_from_j_integral,
            "breitbarth_et_al_2019_dic_integrals",
            "equation 10",
        ),
        (
            in_plane_sif_magnitude_from_j_integral,
            "molteno_becker_2015_j_integral_decomposition",
            "equation 16",
        ),
        (
            mode_iii_sif_magnitude_from_j_integral,
            "molteno_becker_2015_j_integral_decomposition",
            "equation 17",
        ),
        (
            interaction_integral_terms,
            "kuna_fracture_mechanics",
            "equation 6.81",
        ),
        (
            in_plane_sif_from_interaction_integral,
            "molteno_becker_2015_j_integral_decomposition",
            "equation 16",
        ),
        (
            t_stress_interaction_integral_terms,
            "zhao_et_al_2001_corner_cracks",
            "equation 5",
        ),
        (
            t_stress_from_interaction_integral,
            "zhao_et_al_2001_corner_cracks",
            "equation 6",
        ),
        (
            bueckner_chen_integral_terms,
            "kuna_fracture_mechanics",
            "equation 6.92",
        ),
        (
            williams_coefficient_from_bueckner_chen_integral,
            "kuna_fracture_mechanics",
            "equations 6.91--6.94",
        ),
        (
            t_stress_from_stress_difference,
            "yang_ravi_chandar_1999_stress_difference",
            "equation 7",
        ),
    ],
)
def test_formula_names_its_scientific_reference_and_locator(
    function: object,
    citation_key: str,
    locator: str,
) -> None:
    docstring = getattr(function, "__doc__", None) or ""

    assert citation_key in docstring
    assert locator in docstring
    assert "https://doi.org/" in docstring


@pytest.mark.parametrize(
    "function",
    [
        in_plane_j_integral_terms,
        mode_iii_j_integral_terms,
        interaction_integral_terms,
        t_stress_interaction_integral_terms,
        bueckner_chen_integral_terms,
    ],
)
def test_contour_functional_documents_crack_tip_coordinate_convention(
    function: object,
) -> None:
    docstring = " ".join(
        (getattr(function, "__doc__", None) or "").lower().split()
    )

    assert "outward unit" in docstring
    assert "prospective crack extension" in docstring
    assert "normal to the crack plane" in docstring
