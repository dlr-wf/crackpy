"""Contour quadrature numerically evaluates fracture-mechanics functionals on
an oriented Integration Contour.

Each evaluation reduces local energy and work expressions to one scalar
integral value for that contour.
"""

import numpy as np

from crackpy.fracture_analysis.functionals import IntegrandTerms


def evaluate_contour_integral(
    integrand_terms: IntegrandTerms,
    *,
    segment_dy: np.ndarray,
    segment_lengths: np.ndarray,
) -> float:
    """Evaluate a prepared fracture-mechanics functional on one Integration Contour.

    Args:
        integrand_terms: Physical expressions grouped by integration measure
            and contribution sign, in contour-segment order.
        segment_dy: Signed vertical increment for each contour segment in mm,
            ordered like the prepared integrand fields. For CrackPy's
            counter-clockwise contour order, ``dy = n_x ds`` for the outward
            unit normal.
        segment_lengths: Positive length ``ds`` of each contour segment in mm,
            in matching contour order.

    Returns:
        Scalar contour-integral value in the units implied by the prepared
        expressions and segment measures.
    """
    # Follow the widest geometry or functional dtype so quadrature preserves scientific precision.
    accumulator_dtype = np.result_type(
        segment_dy,
        segment_lengths,
        *integrand_terms.integrated_over_dy,
        *integrand_terms.added_over_ds,
        *integrand_terms.subtracted_over_ds,
    )
    contributions = np.zeros(segment_dy.shape, dtype=accumulator_dtype)
    # segment_dy is signed by contour orientation. segment_lengths is positive
    # ds; the functional decomposition determines whether each ds term is
    # added or subtracted.
    for term in integrand_terms.integrated_over_dy:
        contributions += term * segment_dy
    for term in integrand_terms.added_over_ds:
        contributions += term * segment_lengths
    for term in integrand_terms.subtracted_over_ds:
        contributions -= term * segment_lengths
    integral_value = np.sum(contributions)
    return integral_value
