"""Williams displacement bases define coefficient-separated in-plane and
out-of-plane expansion responses in selected-term order.
"""

from collections.abc import Sequence

import numpy as np

from crackpy.structure_elements.material import Material

################################
# SHARED WILLIAMS BASIS INPUTS #
################################


def _basis_inputs(
    r: np.ndarray,
    phi: np.ndarray,
    terms: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return Williams basis coordinates and selected terms.

    Args:
        r: Positive radial coordinates in mm.
        phi: Angular coordinates in radians with the same shape as ``r``.
        terms: Selected integer Williams term numbers.

    Returns:
        NumPy views of ``r``, ``phi``, and ``terms``.

    Raises:
        ValueError: If the coordinate arrays do not have identical shapes.
    """
    r_values = np.asarray(r)
    phi_values = np.asarray(phi)
    if r_values.shape != phi_values.shape:
        raise ValueError("Williams polar-coordinate arrays must have identical shapes.")
    return r_values, phi_values, np.asarray(terms)


########################################
# WILLIAMS IN-PLANE DISPLACEMENT BASIS #
########################################


def williams_in_plane_displacement_basis(
    r: np.ndarray,
    phi: np.ndarray,
    terms: Sequence[int] | np.ndarray,
    material: Material,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate in-plane Williams displacement basis columns.

    The equations follow Williams (1957), DOI 10.1115/1.4011454, and Kuna
    (2013), DOI 10.1007/978-94-007-6680-8.

    Args:
        r: Positive radial coordinates in mm.
        phi: Angular coordinates in radians with the same shape as ``r``.
        terms: Selected integer Williams term numbers.
        material: Material supplying shear modulus ``G`` and ``kappa``.

    Returns:
        X- and y-displacement arrays with shape
        ``(2 * len(terms), *r.shape)`` in all-``a_n`` then all-``b_n`` order.
    """
    r_values, phi_values, term_values = _basis_inputs(r, phi, terms)
    n_terms = len(term_values)
    basis_shape = (2 * n_terms, *r_values.shape)
    basis_x = np.empty(basis_shape)
    basis_y = np.empty(basis_shape)
    displacement_scale = 1 / (2 * material.G)

    for index, term in enumerate(term_values):
        half_term = term / 2
        sign_term = (-1.0) ** term
        radial_scale = displacement_scale * r_values**half_term
        phi_term = half_term * phi_values
        phi_term_minus_two = (half_term - 2) * phi_values

        # Symmetric a_n and antisymmetric b_n eigenfields share the radial scale.
        f_1 = (material.kappa + sign_term + half_term) * np.cos(phi_term)
        f_1 -= half_term * np.cos(phi_term_minus_two)
        g_1 = (-material.kappa + sign_term - half_term) * np.sin(phi_term)
        g_1 += half_term * np.sin(phi_term_minus_two)
        f_2 = (material.kappa - sign_term - half_term) * np.sin(phi_term)
        f_2 += half_term * np.sin(phi_term_minus_two)
        g_2 = (material.kappa + sign_term - half_term) * np.cos(phi_term)
        g_2 += half_term * np.cos(phi_term_minus_two)

        basis_x[index] = radial_scale * f_1
        basis_y[index] = radial_scale * f_2
        basis_x[n_terms + index] = radial_scale * g_1
        basis_y[n_terms + index] = radial_scale * g_2

    return basis_x, basis_y


############################################
# WILLIAMS OUT-OF-PLANE DISPLACEMENT BASIS #
############################################


def williams_out_of_plane_displacement_basis(
    r: np.ndarray,
    phi: np.ndarray,
    terms: Sequence[int] | np.ndarray,
    material: Material,
) -> np.ndarray:
    """Evaluate out-of-plane Williams displacement basis columns.

    The equations follow Kuna (2013), DOI 10.1007/978-94-007-6680-8,
    equations 3.52--3.55.

    Args:
        r: Positive radial coordinates in mm.
        phi: Angular coordinates in radians with the same shape as ``r``.
        terms: Selected integer Williams term numbers.
        material: Material supplying shear modulus ``G``.

    Returns:
        Z-displacement array with shape ``(len(terms), *r.shape)`` in selected
        ``c_n`` term order.
    """
    r_values, phi_values, term_values = _basis_inputs(r, phi, terms)
    basis_z = np.empty((len(term_values), *r_values.shape))
    displacement_scale = 1 / (2 * material.G)

    for index, term in enumerate(term_values):
        half_term = term / 2
        angular_response = (
            2 * np.cos(half_term * phi_values)
            if term % 2 == 0
            else 2 * np.sin(half_term * phi_values)
        )
        basis_z[index] = (
            displacement_scale * r_values**half_term * angular_response
        )

    return basis_z
