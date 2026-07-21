"""Bueckner-Chen kernels define reciprocal-work terms and their Williams-coefficient mapping."""

import numpy as np

from crackpy.fracture_analysis.functionals import IntegrandTerms


def bueckner_chen_integral_terms(
    measured_stress: np.ndarray,
    measured_displacement: np.ndarray,
    auxiliary_stress: np.ndarray,
    auxiliary_displacement: np.ndarray,
    contour_normals: np.ndarray,
) -> IntegrandTerms:
    """Evaluate the Bueckner-Chen reciprocal-work term on prepared fields.

    Args:
        measured_stress: Measured in-plane stress tensors with shape ``(n, 2, 2)``.
        measured_displacement: Measured displacement vectors with shape ``(n, 2)``.
        auxiliary_stress: Auxiliary Williams stress tensors with shape ``(n, 2, 2)``.
        auxiliary_displacement: Auxiliary Williams displacement vectors with shape ``(n, 2)``.
        contour_normals: Outward unit normal vectors with shape ``(n, 2)`` in the
            crack-tip Cartesian frame. The x-axis follows prospective crack
            extension and the y-axis is normal to the crack plane.

    Returns:
        Reciprocal-work term added over contour segment length.

    Notes:
        Kuna (2013), equation 6.92, gives this reciprocal-work contour form.
        DOI: https://doi.org/10.1007/978-94-007-6680-8.
        Citation key: ``kuna_fracture_mechanics``.
        The formulation originates with Chen (1985),
        DOI: https://doi.org/10.1016/0013-7944(85)90131-6,
        citation key ``chen_1985_path_independent_integrals``.
    """
    measured_traction = np.einsum(
        "nij,nj->ni",
        measured_stress,
        contour_normals,
    )
    auxiliary_traction = np.einsum(
        "nij,nj->ni",
        auxiliary_stress,
        contour_normals,
    )

    # Kuna (2013), equation 6.92, for the Bueckner-Chen contour integral.
    reciprocal_work_term = np.sum(
        measured_traction * auxiliary_displacement
        - auxiliary_traction * measured_displacement,
        axis=1,
    )

    return IntegrandTerms(added_over_ds=(reciprocal_work_term,))


def williams_coefficient_from_bueckner_chen_integral(
    integral_value: float,
    *,
    shear_modulus: float,
    kappa: float,
    symmetric_auxiliary_amplitude: float,
    antisymmetric_auxiliary_amplitude: float,
    term: int,
) -> float:
    """Map a Bueckner-Chen integral to one Williams coefficient.

    Args:
        integral_value: Evaluated Bueckner-Chen reciprocal-work integral.
        shear_modulus: Material shear modulus in MPa.
        kappa: Kolosov material constant.
        symmetric_auxiliary_amplitude:
            Amplitude ``c_m`` of the symmetric complementary auxiliary
            Williams eigenfield with order ``m = -term``.
        antisymmetric_auxiliary_amplitude:
            Amplitude ``d_m`` of the antisymmetric complementary auxiliary
            Williams eigenfield with order ``m = -term``.
        term: Requested Williams term number.

    Returns:
        Williams coefficient in MPa mm**(1 - term/2).

    Raises:
        ValueError: If both auxiliary eigenfield amplitudes are nonzero.

    Notes:
        Kuna (2013), equations 6.91--6.94, identifies the complementary
        auxiliary eigenfield order and maps the reciprocal-work integral to
        the selected Williams coefficient.
        See https://doi.org/10.1007/978-94-007-6680-8, citation key
        ``kuna_fracture_mechanics``, using the Williams normalization
        identified by citation key ``williams_1957``.
    """
    if (
        symmetric_auxiliary_amplitude != 0
        and antisymmetric_auxiliary_amplitude != 0
    ):
        raise ValueError(
            "Select either the symmetric or antisymmetric auxiliary eigenfield."
        )

    auxiliary_amplitude = (
        symmetric_auxiliary_amplitude
        + antisymmetric_auxiliary_amplitude
    )

    # Kuna equation 6.93 pairs target order n only with auxiliary order m = -n.
    # Selecting c_m isolates symmetric a_n; selecting d_m isolates
    # antisymmetric b_n.
    sign_and_term_normalization = np.pi * term * (-1) ** (term + 1)
    williams_coefficient = (
        -shear_modulus
        / (kappa + 1)
        / auxiliary_amplitude
        / sign_and_term_normalization
        * integral_value
    )
    return williams_coefficient
