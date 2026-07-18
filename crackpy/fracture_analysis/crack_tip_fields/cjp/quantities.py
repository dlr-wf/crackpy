"""CJP quantity contracts and coefficient transformations define the formulation's
reported fracture-mechanics values, signs, units, and scientific references.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_SQRT_MM_TO_SQRT_M = 1 / np.sqrt(1000)


@dataclass(frozen=True)
class CjpModeIQuantities:
    """Store derived fracture quantities for the CJP Mode I formulation.

    Attributes:
        k_f: Forward-driving intensity factor in MPa sqrt(m).
        k_r: Retardation intensity factor in MPa sqrt(m).
        k_s: Compatibility-induced shear intensity factor in MPa sqrt(m).
        t_x: Crack-growth-direction non-singular stress in MPa.
        t_y: Crack-opening-direction non-singular stress in MPa.
    """

    k_f: float
    k_r: float
    k_s: float
    t_x: float
    t_y: float


@dataclass(frozen=True)
class CjpMixedModeQuantities:
    """Store derived fracture quantities for the CJP mixed-mode formulation.

    Attributes:
        k_f: Forward-driving intensity factor in MPa sqrt(m).
        k_r: Retardation intensity factor in MPa sqrt(m).
        k_s: Crack-face shear intensity factor in MPa sqrt(m).
        k_ii: Mode II stress-intensity factor in MPa sqrt(m).
        t_stress: Non-singular T-stress in MPa.
    """

    k_f: float
    k_r: float
    k_s: float
    k_ii: float
    t_stress: float


def derive_cjp_mode_i_fracture_quantities(
    coefficients: Iterable[float],
) -> tuple[float, float, float, float, float]:
    """Derive CJP Mode I fracture quantities from formulation coefficients.

    Args:
        coefficients: Exactly five coefficients in ``(A, B, C, E, F)`` order.
            ``A``, ``B``, and ``E`` use MPa sqrt(mm), while ``C`` and ``F`` use
            MPa.

    Returns:
        ``(K_F, K_R, K_S, T_x, T_y)`` with intensity factors in MPa sqrt(m)
        and T-stresses in MPa.

    Raises:
        ValueError: If the coefficient iterable cannot be unpacked into exactly
            five values.
    Notes:
        The coefficient interpretation follows Camacho-Reyes et al., "Study of
        Effective Stress Intensity Factor through the CJP Model Using Full-Field
        Experimental Data" (2023), DOI 10.3390/ma16165705.
        This formula-level function is the future attachment point for structured
        scientific-reference metadata.
    """
    a, b, c, e, f = coefficients

    # A, B, and E govern the formulation's singular driving, retardation, and
    # compatibility-induced shear contributions. Their native length scale is
    # sqrt(mm), so only these stress-intensity quantities require conversion to
    # the public MPa sqrt(m) convention.
    k_f = np.sqrt(np.pi / 2) * (a - 3 * b - 8 * e) * _SQRT_MM_TO_SQRT_M
    k_r = -((2 * np.pi) ** (3 / 2)) * e * _SQRT_MM_TO_SQRT_M
    k_s = np.sqrt(np.pi / 2) * (a + b) * _SQRT_MM_TO_SQRT_M

    # C and F are already non-singular normal-stress coefficients in MPa.
    # The published CJP convention defines the reported stresses with the
    # opposite sign to those fitted coefficients.
    t_x = -c
    t_y = -f

    return (
        k_f,
        k_r,
        k_s,
        t_x,
        t_y,
    )


def derive_cjp_mixed_mode_fracture_quantities(
    coefficients: Iterable[float],
) -> tuple[float, float, float, float, float]:
    """Derive CJP mixed-mode fracture quantities from formulation coefficients.

    Args:
        coefficients: Exactly five coefficients in ``(A_r, B_r, B_i, C, E)``
            order. ``A_r``, ``B_r``, ``B_i``, and ``E`` use MPa sqrt(mm), while
            ``C`` uses MPa.

    Returns:
        ``(K_F, K_R, K_S, K_II, T)`` with intensity factors in MPa sqrt(m)
        and T-stress in MPa.

    Raises:
        ValueError: If the coefficient iterable cannot be unpacked into exactly
            five values.
    Notes:
        The coefficient interpretation follows Christopher et al., "Extension
        of the CJP Model to Mixed Mode I and Mode II" (2013), DOI
        10.3221/IGF-ESIS.25.23.
        This formula-level function is the future attachment point for structured
        scientific-reference metadata.
    """
    a_r, b_r, b_i, c, e = coefficients

    # A_r, B_r, and E retain the symmetric CJP driving and shielding terms.
    # B_i is the antisymmetric crack-face shear coefficient introduced by the
    # mixed Mode I/II formulation and contributes to both K_R and K_II.
    k_f = np.sqrt(np.pi / 2) * (a_r - 3 * b_r - 8 * e) * _SQRT_MM_TO_SQRT_M
    k_r = (
        -4
        * np.sqrt(np.pi / 2)
        * (2 * b_i + e * np.pi)
        * _SQRT_MM_TO_SQRT_M
    )

    # The 2013 formulation permits a crack-face-dependent sign for K_S.
    # CrackPy preserves its established negative-branch convention here.
    k_s = -np.sqrt(np.pi / 2) * (a_r + b_r) * _SQRT_MM_TO_SQRT_M
    k_ii = 2 * np.sqrt(2 * np.pi) * b_i * _SQRT_MM_TO_SQRT_M

    # C is the non-singular stress coefficient and already uses MPa.
    t_stress = -c

    return (
        k_f,
        k_r,
        k_s,
        k_ii,
        t_stress,
    )
