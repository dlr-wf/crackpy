"""ODM result completion maps successful, failed, and skipped coefficient-fit
outcomes to authoritative Technique Results. Fits without displacement
observations retain their numerical evidence but have NaN accepted payloads.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    CjpMixedModeCoefficients,
    CjpMixedModeQuantities,
    CjpModeICoefficients,
    CjpModeIQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.cjp.quantities import (
    derive_cjp_mixed_mode_fracture_quantities,
    derive_cjp_mode_i_fracture_quantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    WilliamsInPlaneCoefficients,
    WilliamsInPlaneQuantities,
    WilliamsOutOfPlaneCoefficients,
    WilliamsOutOfPlaneQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams.quantities import (
    derive_williams_in_plane_fracture_quantities,
    derive_williams_out_of_plane_fracture_quantities,
)
from crackpy.fracture_analysis.odm.results import CoefficientFitResult, OdmFitResult

################################
# SHARED RESULT-FIT VALIDATION #
################################


def _require_coefficient_count(
    coefficient_fit: CoefficientFitResult | None,
    expected_count: int,
) -> None:
    """Reject returned fits that cannot represent the selected formulation."""
    if (
        coefficient_fit is not None
        and coefficient_fit.coefficients.size != expected_count
    ):
        raise ValueError(
            f"ODM result requires exactly {expected_count} coefficients; "
            f"received {coefficient_fit.coefficients.size}."
        )


##############################################################
# CHRISTOPHER-JAMES-PATTERSON (CJP) MODE I RESULT COMPLETION #
##############################################################


def _build_cjp_mode_i_odm_result(
    coefficient_fit: CoefficientFitResult | None,
) -> OdmFitResult[CjpModeICoefficients, CjpModeIQuantities]:
    """Build a CJP Mode I result from one realizable fit outcome.

    Args:
        coefficient_fit: Numerical fit returned by the attempted formulation,
            or ``None`` when solving raised before returning.

    Returns:
        A typed completed result for a successful fit, or a failed result with
        fixed-shape NaN payloads otherwise.
    """
    _require_coefficient_count(coefficient_fit, 5)
    if (
        coefficient_fit is None
        or not coefficient_fit.success
        or coefficient_fit.residual.size == 0
    ):
        nan = float("nan")
        return OdmFitResult(
            coefficient_fit,
            CjpModeICoefficients(nan, nan, nan, nan, nan),
            CjpModeIQuantities(nan, nan, nan, nan, nan),
        )

    a, b, c, e, f = coefficient_fit.coefficients
    k_f, k_r, k_s, t_x, t_y = derive_cjp_mode_i_fracture_quantities(
        (a, b, c, e, f)
    )
    return OdmFitResult(
        coefficient_fit,
        CjpModeICoefficients(a, b, c, e, f),
        CjpModeIQuantities(
            k_f=k_f,
            k_r=k_r,
            k_s=k_s,
            t_x=t_x,
            t_y=t_y,
        ),
    )


##################################################################
# CHRISTOPHER-JAMES-PATTERSON (CJP) MIXED-MODE RESULT COMPLETION #
##################################################################


def _build_cjp_mixed_mode_odm_result(
    coefficient_fit: CoefficientFitResult | None,
) -> OdmFitResult[CjpMixedModeCoefficients, CjpMixedModeQuantities]:
    """Build a CJP mixed-mode result from one realizable fit outcome.

    Args:
        coefficient_fit: Numerical fit returned by the attempted formulation,
            or ``None`` when solving raised before returning.

    Returns:
        A typed completed result for a successful fit, or a failed result with
        fixed-shape NaN payloads otherwise.
    """
    _require_coefficient_count(coefficient_fit, 5)
    if (
        coefficient_fit is None
        or not coefficient_fit.success
        or coefficient_fit.residual.size == 0
    ):
        nan = float("nan")
        return OdmFitResult(
            coefficient_fit,
            CjpMixedModeCoefficients(nan, nan, nan, nan, nan),
            CjpMixedModeQuantities(nan, nan, nan, nan, nan),
        )

    a_r, b_r, b_i, c, e = coefficient_fit.coefficients
    k_f, k_r, k_s, k_ii, t_stress = derive_cjp_mixed_mode_fracture_quantities(
        (a_r, b_r, b_i, c, e)
    )
    return OdmFitResult(
        coefficient_fit,
        CjpMixedModeCoefficients(a_r, b_r, b_i, c, e),
        CjpMixedModeQuantities(
            k_f=k_f,
            k_r=k_r,
            k_s=k_s,
            k_ii=k_ii,
            t_stress=t_stress,
        ),
    )


#######################################
# WILLIAMS IN-PLANE RESULT COMPLETION #
#######################################


def _build_williams_in_plane_odm_result(
    terms: Sequence[int],
    coefficient_fit: CoefficientFitResult | None,
) -> OdmFitResult[WilliamsInPlaneCoefficients, WilliamsInPlaneQuantities]:
    """Build an in-plane Williams result from one realizable fit outcome.

    Args:
        terms: Selected Williams terms in coefficient-sequence order.
        coefficient_fit: Numerical in-plane fit, or ``None`` when solving raised.

    Returns:
        A typed completed result for a successful fit, or a failed result with
        term-shaped NaN payloads otherwise.
    """
    ordered_terms = tuple(terms)
    _require_coefficient_count(coefficient_fit, 2 * len(ordered_terms))
    if (
        coefficient_fit is None
        or not coefficient_fit.success
        or coefficient_fit.residual.size == 0
    ):
        nan_values = tuple(np.nan for _ in ordered_terms)
        return OdmFitResult(
            coefficient_fit,
            WilliamsInPlaneCoefficients(
                terms=ordered_terms,
                a_n=nan_values,
                b_n=nan_values,
            ),
            WilliamsInPlaneQuantities(np.nan, np.nan, np.nan),
        )

    n_terms = len(ordered_terms)
    a_n = tuple(coefficient_fit.coefficients[:n_terms])
    b_n = tuple(coefficient_fit.coefficients[n_terms:])
    k_i, k_ii, t_stress = derive_williams_in_plane_fracture_quantities(
        ordered_terms,
        a_n,
        b_n,
    )
    return OdmFitResult(
        coefficient_fit,
        WilliamsInPlaneCoefficients(terms=ordered_terms, a_n=a_n, b_n=b_n),
        WilliamsInPlaneQuantities(
            k_i=k_i,
            k_ii=k_ii,
            t_stress=t_stress,
        ),
    )


###########################################
# WILLIAMS OUT-OF-PLANE RESULT COMPLETION #
###########################################


def _build_williams_out_of_plane_odm_result(
    terms: Sequence[int],
    coefficient_fit: CoefficientFitResult | None,
    *,
    skipped: bool,
) -> OdmFitResult[WilliamsOutOfPlaneCoefficients, WilliamsOutOfPlaneQuantities]:
    """Build an out-of-plane Williams result from a fit or real skip outcome.

    Args:
        terms: Selected Williams terms in coefficient-sequence order.
        coefficient_fit: Numerical out-of-plane fit, or ``None`` when solving
            raised or execution was skipped.
        skipped: Whether absent meaningful z displacement caused an intentional
            out-of-plane skip.

    Returns:
        A typed completed, failed, or skipped result with term-shaped payloads.

    Raises:
        ValueError: If an explicit skip is combined with a coefficient fit.
    """
    ordered_terms = tuple(terms)
    _require_coefficient_count(coefficient_fit, len(ordered_terms))
    if (
        skipped
        or coefficient_fit is None
        or not coefficient_fit.success
        or coefficient_fit.residual.size == 0
    ):
        nan_values = tuple(np.nan for _ in ordered_terms)
        return OdmFitResult(
            coefficient_fit,
            WilliamsOutOfPlaneCoefficients(
                terms=ordered_terms,
                c_n=nan_values,
            ),
            WilliamsOutOfPlaneQuantities(np.nan),
            skipped=skipped,
        )

    c_n = tuple(coefficient_fit.coefficients)
    (k_iii,) = derive_williams_out_of_plane_fracture_quantities(
        ordered_terms, c_n
    )
    return OdmFitResult(
        coefficient_fit,
        WilliamsOutOfPlaneCoefficients(terms=ordered_terms, c_n=c_n),
        WilliamsOutOfPlaneQuantities(k_iii=k_iii),
    )
