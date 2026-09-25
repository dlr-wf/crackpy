"""Interaction-integral kernels relate measured and auxiliary fields to stress-intensity factors and T-stress."""

import numpy as np

from crackpy.fracture_analysis.functionals import IntegrandTerms


def interaction_integral_terms(
    measured_stress: np.ndarray,
    measured_displacement_gradient_x: np.ndarray,
    auxiliary_stress: np.ndarray,
    auxiliary_strain: np.ndarray,
    auxiliary_displacement_gradient_x: np.ndarray,
    contour_normals: np.ndarray,
) -> IntegrandTerms:
    """Evaluate the mutual-interaction terms for prepared fields.

    Args:
        measured_stress: Measured in-plane stress tensors with shape ``(n, 2, 2)``.
        measured_displacement_gradient_x: Measured ``(du_x/dx, du_y/dx)`` vectors.
        auxiliary_stress: Auxiliary in-plane stress tensors with shape ``(n, 2, 2)``.
        auxiliary_strain: Auxiliary in-plane strain tensors with shape ``(n, 2, 2)``.
        auxiliary_displacement_gradient_x: Auxiliary ``(du_x/dx, du_y/dx)`` vectors.
        contour_normals: Outward unit normal vectors with shape ``(n, 2)`` in the
            crack-tip Cartesian frame. The x-axis follows prospective crack
            extension and the y-axis is normal to the crack plane.

    Returns:
        Interaction strain-energy density for signed ``dy`` integration and
        two reciprocal traction-work terms for subtraction over ``ds``.

    Notes:
        Kuna (2013), equation 6.81, defines the interaction-integral cross
        terms. DOI: https://doi.org/10.1007/978-94-007-6680-8.
        Citation key: ``kuna_fracture_mechanics``.
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

    interaction_strain_energy_density = np.einsum(
        "nij,nij->n",
        measured_stress,
        auxiliary_strain,
    )
    measured_traction_work_term = np.einsum(
        "ni,ni->n",
        measured_traction,
        auxiliary_displacement_gradient_x,
    )
    auxiliary_traction_work_term = np.einsum(
        "ni,ni->n",
        auxiliary_traction,
        measured_displacement_gradient_x,
    )

    # Kuna (2013), equation 6.81, expressed with dy = n_x ds.
    return IntegrandTerms(
        integrated_over_dy=(interaction_strain_energy_density,),
        subtracted_over_ds=(
            measured_traction_work_term,
            auxiliary_traction_work_term,
        ),
    )


def t_stress_interaction_integral_terms(
    measured_stress: np.ndarray,
    measured_strain: np.ndarray,
    measured_displacement_gradient_x: np.ndarray,
    auxiliary_stress: np.ndarray,
    auxiliary_displacement_gradient_x: np.ndarray,
    contour_normals: np.ndarray,
) -> IntegrandTerms:
    """Evaluate Zhao auxiliary-field interaction terms for T-stress.

    Args:
        measured_stress: Measured in-plane stress tensors with shape ``(n, 2, 2)``.
        measured_strain: Measured in-plane strain tensors with shape ``(n, 2, 2)``.
        measured_displacement_gradient_x: Measured ``(du_x/dx, du_y/dx)`` vectors.
        auxiliary_stress: Zhao auxiliary stress tensors with shape ``(n, 2, 2)``.
        auxiliary_displacement_gradient_x: Zhao auxiliary ``(du_x/dx, du_y/dx)`` vectors.
        contour_normals: Outward unit normal vectors with shape ``(n, 2)`` in the
            crack-tip Cartesian frame. The x-axis follows prospective crack
            extension and the y-axis is normal to the crack plane.

    Returns:
        Interaction strain-energy density for signed ``dy`` integration and
        two reciprocal traction-work terms for subtraction over ``ds``.

    Notes:
        Zhao, Tong, and Byrne (2001), equations 4a--4b, define the auxiliary
        line-load field; equation 5 gives the corresponding interaction
        cross terms in domain form. This kernel evaluates their contour
        counterpart. DOI: https://doi.org/10.1023/A:1011016720630.
        Citation key: ``zhao_et_al_2001_corner_cracks``.
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

    interaction_strain_energy_density = np.einsum(
        "nij,nij->n",
        auxiliary_stress,
        measured_strain,
    )
    measured_traction_work_term = np.einsum(
        "ni,ni->n",
        measured_traction,
        auxiliary_displacement_gradient_x,
    )
    auxiliary_traction_work_term = np.einsum(
        "ni,ni->n",
        auxiliary_traction,
        measured_displacement_gradient_x,
    )

    # Zhao, Tong, and Byrne (2001), equation 5, reduced to contour terms.
    return IntegrandTerms(
        integrated_over_dy=(interaction_strain_energy_density,),
        subtracted_over_ds=(
            measured_traction_work_term,
            auxiliary_traction_work_term,
        ),
    )


def in_plane_sif_from_interaction_integral(
    interaction_integral: float,
    *,
    youngs_modulus: float,
    auxiliary_sif: float,
) -> float:
    """Recover the measured-field Stress Intensity Factor for the auxiliary mode.

    Args:
        interaction_integral:
            Measured--auxiliary interaction integral in N/mm.
        youngs_modulus:
            Young's modulus in MPa for the plane-stress relation.
        auxiliary_sif:
            Nonzero Mode I or Mode II auxiliary-field Stress Intensity Factor
            in MPa sqrt(mm).

    Returns:
        The signed measured-field Stress Intensity Factor for the
        auxiliary-field loading mode in MPa sqrt(m).

    Notes:
        Kuna (2013), equation 6.81, defines the interaction-integral cross
        terms used here; equations 6.84--6.86 give the pure-mode auxiliary
        mapping under the book's displayed energy normalization.
        Molteno and Becker (2015), equation 16, gives
        ``G_M = K_M**2 / E'`` for Modes I and II.
        Superposition under that normalization gives
        ``J_M**(1,2) = 2 K_M**(1) K_M**(2) / E'`` and therefore introduces
        the factor of one half in the recovered measured-field Stress
        Intensity Factor.
        See https://doi.org/10.1007/978-94-007-6680-8, citation key
        ``kuna_fracture_mechanics``, and https://doi.org/10.1111/str.12166,
        citation key ``molteno_becker_2015_j_integral_decomposition``.
        Breitbarth et al. (2019), equations 3 and 9, document the corresponding
        CrackPy line-integral form and pure auxiliary-mode mapping, while
        equation 10 uses ``K_J = sqrt(J E)``;
        https://doi.org/10.3221/IGF-ESIS.49.02, citation key
        ``breitbarth_et_al_2019_dic_integrals``.
        Kuna equations 6.84--6.86 and Breitbarth equation 9 display the
        no-half mapping because their preceding normalization is
        ``G = K**2 / (2 E')``. CrackPy retains the standard
        ``G = K**2 / E'`` normalization and its resulting factor of one half.
    """
    # Molteno and Becker (2015), equation 16, combined with superposition:
    # J_M**(1,2) = 2 K_M**(1) K_M**(2) / E' for M = I or II.
    measured_sif_mm = (
        youngs_modulus
        / auxiliary_sif
        * interaction_integral
        / 2
    )
    measured_sif = measured_sif_mm / np.sqrt(1000)
    return measured_sif


def t_stress_from_interaction_integral(
    interaction_integral: float,
    *,
    youngs_modulus: float,
    poisson_ratio: float,
    plane_strain: bool,
    reference_out_of_plane_strain: float | None = None,
) -> float:
    """Map the Zhao interaction integral to elastic T-stress.

    Args:
        interaction_integral: Evaluated T-stress interaction integral.
        youngs_modulus: Young's modulus in MPa.
        poisson_ratio: In-plane Poisson ratio.
        plane_strain: Whether to use the plane-strain mapping.
        reference_out_of_plane_strain: Out-of-plane strain at the crack-front
            reference point, required for the plane-stress mapping.

    Returns:
        T-stress in MPa.

    Raises:
        ValueError: If plane stress is selected without the out-of-plane
            reference strain.

    Notes:
        Zhao, Tong, and Byrne (2001), equation 6, maps the interaction
        integral and crack-front out-of-plane strain to elastic T-Stress.
        DOI: https://doi.org/10.1023/A:1011016720630.
        Citation key: ``zhao_et_al_2001_corner_cracks``.
    """
    if plane_strain:
        # Zhao, Tong, and Byrne (2001), equation 6, with eps_33 = 0.
        t_stress = (
            youngs_modulus
            / (1 - poisson_ratio ** 2)
            * interaction_integral
        )
    else:
        if reference_out_of_plane_strain is None:
            raise ValueError(
                "reference_out_of_plane_strain is required for the "
                "plane-stress T-stress mapping."
            )
        # Zhao, Tong, and Byrne (2001), equation 6, for plane stress.
        t_stress = youngs_modulus * (
            interaction_integral
            + poisson_ratio * reference_out_of_plane_strain
        )
    return t_stress
