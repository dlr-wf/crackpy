"""J-integral kernels define crack-driving-force terms and their stress-intensity mappings."""

import numpy as np

from crackpy.fracture_analysis.functionals import IntegrandTerms


def in_plane_j_integral_terms(
    stress_tensors: np.ndarray,
    strain_tensors: np.ndarray,
    displacement_gradient_x: np.ndarray,
    contour_normals: np.ndarray,
) -> IntegrandTerms:
    """Evaluate the in-plane J-integral terms on prepared contour samples.

    Args:
        stress_tensors: In-plane stress tensors with shape ``(n, 2, 2)`` in MPa.
        strain_tensors: In-plane strain tensors with shape ``(n, 2, 2)``.
        displacement_gradient_x: Vectors ``(du_x/dx, du_y/dx)`` with shape ``(n, 2)``.
        contour_normals: Outward unit normal vectors with shape ``(n, 2)`` in the
            crack-tip Cartesian frame. The x-axis follows prospective crack
            extension and the y-axis is normal to the crack plane.

    Returns:
        Strain-energy density for signed ``dy`` integration and a traction-work
        term for subtraction over ``ds``.

    Notes:
        Rice (1968), equation 1, defines this J-integral expression.
        DOI: https://doi.org/10.1115/1.3601206.
        Citation key: ``rice_1968_j_integral``.
    """
    traction_vectors = np.einsum(
        "nij,nj->ni",
        stress_tensors,
        contour_normals,
    )

    # For linear elasticity, the strain-energy density is one half of the
    # double contraction between stress and strain.
    strain_energy_density = 0.5 * np.einsum(
        "nij,nij->n",
        stress_tensors,
        strain_tensors,
    )

    # The traction work uses the crack-growth-direction displacement gradient.
    traction_work_term = np.einsum(
        "ni,ni->n",
        traction_vectors,
        displacement_gradient_x,
    )

    # Rice (1968), equation 1:
    # J = integral(W dy - traction · displacement_gradient_x ds)
    return IntegrandTerms(
        integrated_over_dy=(strain_energy_density,),
        subtracted_over_ds=(traction_work_term,),
    )


def mode_iii_j_integral_terms(
    out_of_plane_displacement_derivative_x: np.ndarray,
    out_of_plane_displacement_derivative_y: np.ndarray,
    sigma_xz: np.ndarray,
    sigma_yz: np.ndarray,
    contour_normals: np.ndarray,
) -> IntegrandTerms:
    """Evaluate CrackPy's Mode III J-integral expression.

    Args:
        out_of_plane_displacement_derivative_x:
            Crack-growth-direction derivative ``du_z/dx`` on the contour.
        out_of_plane_displacement_derivative_y:
            Crack-plane-normal derivative ``du_z/dy`` on the contour.
        sigma_xz: Xz shear-stress samples in MPa.
        sigma_yz: Yz shear-stress samples in MPa.
        contour_normals: Outward unit normal vectors with shape ``(n, 2)`` in the
            crack-tip Cartesian frame. The x-axis follows prospective crack
            extension and the y-axis is normal to the crack plane.

    Returns:
        Stress--displacement-gradient contraction for signed ``dy``
        integration and a traction-work term for subtraction over ``ds``.

    Notes:
        Molteno and Becker (2015), equations 8--11, define the Mode III
        contour expression, out-of-plane displacement derivatives, and shear
        stresses. DOI: https://doi.org/10.1111/str.12166.
        Citation key: ``molteno_becker_2015_j_integral_decomposition``.
        CrackPy pairs ``du_z/dx`` with both shear-stress traction components in
        the traction-work term.
    """
    stress_gradient_contraction = out_of_plane_displacement_derivative_x * sigma_xz + out_of_plane_displacement_derivative_y * sigma_yz

    # CrackPy's Mode III expression pairs du_z/dx with both
    # shear-stress components after projection onto the contour unit normal.
    traction_work_term = (
        sigma_xz * out_of_plane_displacement_derivative_x * contour_normals[:, 0]
        + sigma_yz * out_of_plane_displacement_derivative_x * contour_normals[:, 1]
    )

    return IntegrandTerms(
        integrated_over_dy=(stress_gradient_contraction,),
        subtracted_over_ds=(traction_work_term,),
    )


def in_plane_energy_equivalent_sif_from_j_integral(
    j_integral: float,
    *,
    youngs_modulus: float,
) -> float:
    """Calculate the in-plane energy-equivalent Stress Intensity Factor magnitude.

    Args:
        j_integral:
            Total in-plane J-integral in N/mm.
        youngs_modulus:
            Young's modulus in MPa for the plane-stress relation.

    Returns:
        The nonnegative energy-equivalent Stress Intensity Factor ``K_J`` in
        MPa sqrt(m).

    Notes:
        Breitbarth et al. (2019), equation 10, defines ``K_J = sqrt(J E)``
        as a representative or equivalent Stress Intensity Factor under
        mixed-mode loading; https://doi.org/10.3221/IGF-ESIS.49.02,
        citation key ``breitbarth_et_al_2019_dic_integrals``.
        For a nonnegative J-integral under linear-elastic plane stress,
        ``K_J = sqrt(E J) = sqrt(K_I**2 + K_II**2)``.
        The absolute-value evaluation produces a nonnegative magnitude for
        negative contour results.
    """
    # Breitbarth et al. (2019), equation 10, under in-plane linear-elastic
    # plane stress:
    # J = (K_I**2 + K_II**2) / E and K_J = sqrt(E * abs(J)).
    energy_equivalent_sif = np.sqrt(np.abs(j_integral) / 1000.0 * youngs_modulus)
    return energy_equivalent_sif


def in_plane_sif_magnitude_from_j_integral(
    modal_j_integral: float,
    *,
    youngs_modulus: float,
) -> float:
    """Calculate a Mode I or II Stress Intensity Factor magnitude from J.

    Args:
        modal_j_integral:
            Decomposed Mode I or Mode II J-integral in N/mm.
        youngs_modulus:
            Young's modulus in MPa for the plane-stress relation.

    Returns:
        The corresponding nonnegative modal Stress Intensity Factor magnitude
        in MPa sqrt(m), or NaN when the modal J-integral is negative.

    Notes:
        Molteno and Becker (2015), equation 16, gives
        ``J_M = G_M = K_M**2 / E`` for ``M = I, II`` under plane stress.
        The square-root mapping therefore defines the magnitude associated
        with the decomposed modal energy-release rate.
        See https://doi.org/10.1111/str.12166, citation key
        ``molteno_becker_2015_j_integral_decomposition``.
    """
    sanitized_modal_j = np.where(
        modal_j_integral >= 0,
        modal_j_integral,
        np.nan,
    )

    # Molteno and Becker equation 16 under plane stress:
    # J_M = G_M = K_M**2 / E for M = I, II.
    modal_sif_magnitude = np.sqrt(sanitized_modal_j * youngs_modulus) / np.sqrt(1000)
    return modal_sif_magnitude


def mode_iii_sif_magnitude_from_j_integral(
    mode_iii_j_integral: float,
    *,
    youngs_modulus: float,
    poisson_ratio: float,
) -> float:
    """Calculate the Mode III Stress Intensity Factor magnitude from J.

    Args:
        mode_iii_j_integral:
            Decomposed Mode III J-integral in N/mm.
        youngs_modulus:
            Young's modulus in MPa for the plane-stress relation.
        poisson_ratio:
            Poisson's ratio used to obtain the Mode III shear relation.

    Returns:
        The nonnegative Mode III Stress Intensity Factor magnitude in MPa
        sqrt(m), or NaN when the Mode III J-integral is negative.

    Notes:
        Molteno and Becker (2015), equation 17, gives
        ``J_III = G_III = K_III**2 * (1 + nu) / E`` under plane stress.
        See https://doi.org/10.1111/str.12166, citation key
        ``molteno_becker_2015_j_integral_decomposition``.
    """
    sanitized_mode_iii_j = np.where(
        mode_iii_j_integral >= 0,
        mode_iii_j_integral,
        np.nan,
    )

    # Molteno and Becker (2015), equation 17, under plane stress:
    # J_III = G_III = K_III**2 * (1 + nu) / E.
    mode_iii_sif_magnitude = np.sqrt(sanitized_mode_iii_j * youngs_modulus / (1 + poisson_ratio)) / np.sqrt(1000)
    return mode_iii_sif_magnitude
