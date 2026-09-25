"""Scientific parity evidence for crack-tip-field displacement bases across every CJP
and Williams coefficient column.
"""

from collections.abc import Callable

import numpy as np
import pytest

from crackpy.fracture_analysis.crack_tip import (
    cjp_displ_field_mixedmode,
    cjp_displ_field_modeI,
    williams_displ_field_xy,
    williams_displ_field_z,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    cjp_mixed_mode_displacement_basis,
    cjp_mode_i_displacement_basis,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    williams_in_plane_displacement_basis,
    williams_out_of_plane_displacement_basis,
)
from crackpy.structure_elements.material import Material


@pytest.fixture
def polar_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Return positive radii and representative angles with identical shapes."""
    r = np.array([[0.25, 0.8, 1.7], [0.4, 1.2, 2.5]])
    phi = np.array([[-2.1, -0.7, 0.0], [0.35, 1.1, 2.4]])
    return r, phi


@pytest.mark.parametrize(
    ("basis_function", "analytical_function"),
    [
        (cjp_mode_i_displacement_basis, cjp_displ_field_modeI),
        (cjp_mixed_mode_displacement_basis, cjp_displ_field_mixedmode),
    ],
)
def test_cjp_basis_columns_match_unit_coefficient_analytical_fields(
    basis_function: Callable,
    analytical_function: Callable,
    polar_coordinates: tuple[np.ndarray, np.ndarray],
) -> None:
    r, phi = polar_coordinates
    material = Material(E=70000.0, nu_xy=0.29)

    basis_x, basis_y = basis_function(r, phi, material)

    assert basis_x.shape == (5, *r.shape)
    assert basis_y.shape == (5, *r.shape)
    for column, unit_coefficients in enumerate(np.eye(5)):
        expected_x, expected_y = analytical_function(
            unit_coefficients,
            phi,
            r,
            material,
        )
        np.testing.assert_allclose(basis_x[column], expected_x)
        np.testing.assert_allclose(basis_y[column], expected_y)


def test_williams_in_plane_basis_columns_match_all_a_then_all_b_order(
    polar_coordinates: tuple[np.ndarray, np.ndarray],
) -> None:
    r, phi = polar_coordinates
    terms = np.array([-1, 1, 2, 4])
    material = Material(E=70000.0, nu_xy=0.29)

    basis_x, basis_y = williams_in_plane_displacement_basis(
        r,
        phi,
        terms,
        material,
    )

    assert basis_x.shape == (2 * len(terms), *r.shape)
    assert basis_y.shape == (2 * len(terms), *r.shape)
    for column in range(2 * len(terms)):
        a_n = np.zeros(len(terms))
        b_n = np.zeros(len(terms))
        if column < len(terms):
            a_n[column] = 1.0
        else:
            b_n[column - len(terms)] = 1.0
        expected_x, expected_y = williams_displ_field_xy(
            a_n,
            b_n,
            terms,
            phi,
            r,
            material,
        )
        np.testing.assert_allclose(basis_x[column], expected_x)
        np.testing.assert_allclose(basis_y[column], expected_y)


def test_williams_out_of_plane_basis_columns_match_selected_term_order(
    polar_coordinates: tuple[np.ndarray, np.ndarray],
) -> None:
    r, phi = polar_coordinates
    terms = np.array([4, -1, 3, 1])
    material = Material(E=70000.0, nu_xy=0.29)

    basis_z = williams_out_of_plane_displacement_basis(
        r,
        phi,
        terms,
        material,
    )

    assert basis_z.shape == (len(terms), *r.shape)
    for column, unit_coefficients in enumerate(np.eye(len(terms))):
        expected_z = williams_displ_field_z(
            unit_coefficients,
            terms,
            phi,
            r,
            material,
        )
        np.testing.assert_allclose(basis_z[column], expected_z)


@pytest.mark.parametrize(
    "basis_call",
    [
        lambda r, phi, material: cjp_mode_i_displacement_basis(r, phi, material),
        lambda r, phi, material: cjp_mixed_mode_displacement_basis(r, phi, material),
        lambda r, phi, material: williams_in_plane_displacement_basis(
            r, phi, np.array([1, 2]), material
        ),
        lambda r, phi, material: williams_out_of_plane_displacement_basis(
            r, phi, np.array([1, 2]), material
        ),
    ],
)
def test_crack_tip_field_bases_require_identically_shaped_polar_coordinates(
    basis_call: Callable,
) -> None:
    with pytest.raises(ValueError):
        basis_call(
            np.ones((2, 1)),
            np.ones((1, 2)),
            Material(),
        )
