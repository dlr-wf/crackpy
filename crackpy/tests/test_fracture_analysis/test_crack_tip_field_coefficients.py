"""Verify Williams coefficient contracts own normalized coefficient snapshots."""

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.williams.coefficients import (
    WilliamsInPlaneCoefficients,
    WilliamsOutOfPlaneCoefficients,
)


def test_in_plane_coefficients_own_normalized_source_array_values():
    source_terms = np.asarray([1, 2], dtype=np.int32)
    source_a_n = np.asarray([3.0, 4.0], dtype=np.float32)
    source_b_n = np.asarray([5.0, 6.0], dtype=np.float32)

    coefficients = WilliamsInPlaneCoefficients(
        terms=source_terms,
        a_n=source_a_n,
        b_n=source_b_n,
    )

    source_terms[0] = 99
    source_a_n[0] = 99.0
    source_b_n[0] = 99.0

    assert coefficients.terms == (1, 2)
    assert coefficients.a_n == (3.0, 4.0)
    assert coefficients.b_n == (5.0, 6.0)
    assert all(type(term) is int for term in coefficients.terms)
    assert all(type(value) is float for value in coefficients.a_n)
    assert all(type(value) is float for value in coefficients.b_n)


def test_out_of_plane_coefficients_own_normalized_source_array_values():
    source_terms = np.asarray([1, 3], dtype=np.int32)
    source_c_n = np.asarray([7.0, 8.0], dtype=np.float32)

    coefficients = WilliamsOutOfPlaneCoefficients(
        terms=source_terms,
        c_n=source_c_n,
    )

    source_terms[0] = 99
    source_c_n[0] = 99.0

    assert coefficients.terms == (1, 3)
    assert coefficients.c_n == (7.0, 8.0)
    assert all(type(term) is int for term in coefficients.terms)
    assert all(type(value) is float for value in coefficients.c_n)
