"""Legacy ODM projections define the one-way mapping from authoritative Technique
Results to established mutable CrackPy 1.x containers.
"""

import numpy as np

from crackpy.fracture_analysis.crack_tip_fields.cjp import (
    CjpMixedModeCoefficients,
    CjpMixedModeQuantities,
    CjpModeICoefficients,
    CjpModeIQuantities,
)
from crackpy.fracture_analysis.crack_tip_fields.williams import (
    WilliamsInPlaneCoefficients,
    WilliamsInPlaneQuantities,
    WilliamsOutOfPlaneCoefficients,
    WilliamsOutOfPlaneQuantities,
)
from crackpy.fracture_analysis.odm.results import OdmFitResult


def _project_cjp_mode_i_compatibility(
    result: OdmFitResult[CjpModeICoefficients, CjpModeIQuantities],
) -> tuple[np.ndarray, dict[str, float]]:
    """Project a CJP Mode I result into established mutable containers.

    Args:
        result: Authoritative CJP Mode I ODM Technique Result.

    Returns:
        A fresh coefficient array and ordered legacy quantity dictionary.
    """
    coefficients = result.coefficients
    quantities = result.quantities
    return np.array(
        (coefficients.a, coefficients.b, coefficients.c, coefficients.e, coefficients.f)
    ), {
        "Error": result.cost,
        "K_F": quantities.k_f,
        "K_R": quantities.k_r,
        "K_S": quantities.k_s,
        "T_x": quantities.t_x,
        "T_y": quantities.t_y,
    }


def _project_cjp_mixed_mode_compatibility(
    result: OdmFitResult[CjpMixedModeCoefficients, CjpMixedModeQuantities],
) -> tuple[np.ndarray, dict[str, float]]:
    """Project a CJP mixed-mode result into established mutable containers.

    Args:
        result: Authoritative CJP mixed-mode ODM Technique Result.

    Returns:
        A fresh coefficient array and ordered legacy quantity dictionary.
    """
    coefficients = result.coefficients
    quantities = result.quantities
    return np.array(
        (
            coefficients.a_r,
            coefficients.b_r,
            coefficients.b_i,
            coefficients.c,
            coefficients.e,
        )
    ), {
        "Error": result.cost,
        "K_F": quantities.k_f,
        "K_R": quantities.k_r,
        "K_S": quantities.k_s,
        "K_II": quantities.k_ii,
        "T": quantities.t_stress,
    }


def _project_williams_compatibility(
    in_plane_result: OdmFitResult[
        WilliamsInPlaneCoefficients,
        WilliamsInPlaneQuantities,
    ],
    out_of_plane_result: OdmFitResult[
        WilliamsOutOfPlaneCoefficients,
        WilliamsOutOfPlaneQuantities,
    ],
) -> tuple[
    np.ndarray,
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[str, float],
]:
    """Combine two Williams results into established mutable containers.

    Args:
        in_plane_result: Authoritative in-plane Williams ODM Technique Result.
        out_of_plane_result: Authoritative out-of-plane Williams ODM Technique
            Result.

    Returns:
        Fresh combined coefficients, term-keyed coefficient dictionaries, and
        the ordered legacy quantity dictionary.
    """
    in_plane_coefficients = in_plane_result.coefficients
    out_of_plane_coefficients = out_of_plane_result.coefficients
    in_plane_quantities = in_plane_result.quantities
    out_of_plane_quantities = out_of_plane_result.quantities
    return (
        np.array(
            in_plane_coefficients.a_n
            + in_plane_coefficients.b_n
            + out_of_plane_coefficients.c_n
        ),
        dict(zip(in_plane_coefficients.terms, in_plane_coefficients.a_n)),
        dict(zip(in_plane_coefficients.terms, in_plane_coefficients.b_n)),
        dict(zip(out_of_plane_coefficients.terms, out_of_plane_coefficients.c_n)),
        {
            "Error_xy": in_plane_result.cost,
            "K_I": in_plane_quantities.k_i,
            "K_II": in_plane_quantities.k_ii,
            "T": in_plane_quantities.t_stress,
            "Error_z": out_of_plane_result.cost,
            "K_III": out_of_plane_quantities.k_iii,
        },
    )
