"""CJP displacement bases define coefficient-separated Mode I and mixed-mode
crack-tip field responses in formulation order.
"""

import numpy as np

from crackpy.structure_elements.material import Material

###########################
# SHARED CJP BASIS INPUTS #
###########################


def _polar_coordinates(r: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return identically shaped CJP polar-coordinate arrays.

    Args:
        r: Positive radial coordinates in mm.
        phi: Angular coordinates in radians with the same shape as ``r``.

    Returns:
        NumPy views of ``r`` and ``phi``.

    Raises:
        ValueError: If the coordinate arrays do not have identical shapes.
    """
    r_values = np.asarray(r)
    phi_values = np.asarray(phi)
    if r_values.shape != phi_values.shape:
        raise ValueError("CJP polar-coordinate arrays must have identical shapes.")
    return r_values, phi_values


##################################################
# CHRISTOPHER-JAMES-PATTERSON (CJP) MODE I BASIS #
##################################################


def cjp_mode_i_displacement_basis(
    r: np.ndarray,
    phi: np.ndarray,
    material: Material,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the five CJP Mode I displacement basis columns.

    The equations follow Camacho-Reyes et al. (2023), formulas 10 and 11,
    DOI 10.3390/ma16165705.

    Args:
        r: Positive radial coordinates in mm.
        phi: Angular coordinates in radians with the same shape as ``r``.
        material: Material supplying shear modulus ``G`` and ``kappa``.

    Returns:
        X- and y-displacement arrays with shape ``(5, *r.shape)`` in
        ``(A, B, C, E, F)`` coefficient order.
    """
    r_values, phi_values = _polar_coordinates(r, phi)
    kappa = material.kappa
    sqrt_r = np.sqrt(r_values)
    log_r = np.log(r_values)
    phi_half = phi_values / 2
    phi_three_half = 3 * phi_half
    sin_phi_half = np.sin(phi_half)
    cos_phi_half = np.cos(phi_half)
    sin_phi_three_half = np.sin(phi_three_half)
    cos_phi_three_half = np.cos(phi_three_half)
    sin_phi = np.sin(phi_values)
    cos_phi = np.cos(phi_values)

    # Each row is the displacement response to one unit formulation coefficient.
    basis_x = np.asarray([
        -sqrt_r * cos_phi_half,
        sqrt_r * (cos_phi_three_half - 2 * kappa * cos_phi_half),
        -r_values * (1 + kappa) * cos_phi / 4,
        sqrt_r
        * (
            -2 * cos_phi_half
            + 2 * cos_phi_three_half
            - log_r * (cos_phi_three_half + (1 - 2 * kappa) * cos_phi_half)
            - phi_values
            * (sin_phi_three_half + (2 * kappa - 1) * sin_phi_half)
        ),
        r_values * (kappa - 3) * cos_phi / 4,
    ])
    basis_y = np.asarray([
        sqrt_r * sin_phi_half,
        sqrt_r * (sin_phi_three_half - 2 * kappa * sin_phi_half),
        r_values * (3 - kappa) * sin_phi / 4,
        sqrt_r
        * (
            2 * sin_phi_half
            + 2 * sin_phi_three_half
            + log_r * (sin_phi_three_half - (1 + 2 * kappa) * sin_phi_half)
            - phi_values
            * (cos_phi_three_half + (2 * kappa + 1) * cos_phi_half)
        ),
        r_values * (kappa + 1) * sin_phi / 4,
    ])
    displacement_scale = 1 / (2 * material.G)
    return basis_x * displacement_scale, basis_y * displacement_scale


######################################################
# CHRISTOPHER-JAMES-PATTERSON (CJP) MIXED-MODE BASIS #
######################################################


def cjp_mixed_mode_displacement_basis(
    r: np.ndarray,
    phi: np.ndarray,
    material: Material,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the five CJP mixed-mode displacement basis columns.

    The equations follow Christopher et al. (2013), formulas 10 and 11,
    DOI 10.3221/IGF-ESIS.25.23.

    Args:
        r: Positive radial coordinates in mm.
        phi: Angular coordinates in radians with the same shape as ``r``.
        material: Material supplying shear modulus ``G`` and ``kappa``.

    Returns:
        X- and y-displacement arrays with shape ``(5, *r.shape)`` in
        ``(A_r, B_r, B_i, C, E)`` coefficient order.
    """
    r_values, phi_values = _polar_coordinates(r, phi)
    kappa = material.kappa
    sqrt_r = np.sqrt(r_values)
    log_r = np.log(r_values)
    phi_half = phi_values / 2
    phi_three_half = 3 * phi_half
    sin_phi_half = np.sin(phi_half)
    cos_phi_half = np.cos(phi_half)
    sin_phi_three_half = np.sin(phi_three_half)
    cos_phi_three_half = np.cos(phi_three_half)
    sin_phi = np.sin(phi_values)
    cos_phi = np.cos(phi_values)

    # Rows preserve the published real/imaginary formulation coefficient order.
    basis_x = np.asarray([
        -sqrt_r * cos_phi_half,
        sqrt_r * (cos_phi_three_half - 2 * kappa * cos_phi_half),
        sqrt_r * ((2 * kappa - 3) * sin_phi_half - sin_phi_three_half),
        -r_values * (1 + kappa) * cos_phi / 4,
        sqrt_r
        * (
            -2 * cos_phi_half
            + 2 * cos_phi_three_half
            + log_r * (cos_phi_three_half + (1 - 2 * kappa) * cos_phi_half)
            + phi_values
            * (sin_phi_three_half + (1 + 2 * kappa) * sin_phi_half)
        ),
    ])
    basis_y = np.asarray([
        sqrt_r * sin_phi_half,
        sqrt_r * (sin_phi_three_half - 2 * kappa * sin_phi_half),
        sqrt_r * (cos_phi_three_half - (2 * kappa + 3) * cos_phi_half),
        r_values * (3 - kappa) * sin_phi / 4,
        sqrt_r
        * (
            2 * sin_phi_half
            + 2 * sin_phi_three_half
            + log_r * (sin_phi_three_half - (1 + 2 * kappa) * sin_phi_half)
            - phi_values
            * (cos_phi_three_half + (1 + 2 * kappa) * cos_phi_half)
        ),
    ])
    displacement_scale = 1 / (2 * material.G)
    return basis_x * displacement_scale, basis_y * displacement_scale
