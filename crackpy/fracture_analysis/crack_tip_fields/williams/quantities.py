"""Williams quantity contracts and coefficient transformations define reported
fracture-mechanics values, term selection, signs, units, and scientific references.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_SQRT_MM_TO_SQRT_M = 1 / np.sqrt(1000)


@dataclass(frozen=True)
class WilliamsInPlaneQuantities:
    """Store derived in-plane fracture quantities for a Williams expansion.

    Attributes:
        k_i: Mode I stress-intensity factor in MPa sqrt(m).
        k_ii: Mode II stress-intensity factor in MPa sqrt(m).
        t_stress: Non-singular T-stress in MPa.
    """

    k_i: float
    k_ii: float
    t_stress: float


@dataclass(frozen=True)
class WilliamsOutOfPlaneQuantities:
    """Store the derived out-of-plane quantity for a Williams expansion.

    Attributes:
        k_iii: Mode III stress-intensity factor in MPa sqrt(m).
    """

    k_iii: float


def williams_coefficient_m_to_mm(
    quantity_in_m: float,
    *,
    term: int = 1,
) -> float:
    """Convert a Williams coefficient from metre- to millimetre-based units.

    Args:
        quantity_in_m: Coefficient in MPa m**(1 - n/2).
        term: Williams expansion term ``n``.

    Returns:
        Coefficient in MPa mm**(1 - n/2).
    """
    # Williams term n scales with length raised to the exponent 1 - n/2.
    length_unit_factor = 1000 ** (1 - term / 2)
    converted_coefficient = quantity_in_m * length_unit_factor
    return converted_coefficient


def williams_coefficient_mm_to_m(
    quantity_in_mm: float,
    *,
    term: int = 1,
) -> float:
    """Convert a Williams coefficient from millimetre- to metre-based units.

    Args:
        quantity_in_mm: Coefficient in MPa mm**(1 - n/2).
        term: Williams expansion term ``n``.

    Returns:
        Coefficient in MPa m**(1 - n/2).
    """
    # Williams term n scales with length raised to the exponent 1 - n/2.
    length_unit_factor = 1000 ** (1 - term / 2)
    converted_coefficient = quantity_in_mm / length_unit_factor
    return converted_coefficient


def t_stress_from_williams_coefficient(
    second_order_symmetric_coefficient: float,
) -> float:
    """Map the second-order symmetric Williams coefficient to T-stress.

    Args:
        second_order_symmetric_coefficient: The symmetric ``a_2`` coefficient
            in MPa.

    Returns:
        T-stress in MPa.

    Notes:
        The coefficient interpretation follows Williams, "On the Stress
        Distribution at the Base of a Stationary Crack" (1957), DOI
        10.1115/1.4011454. Kuna, "Finite Elements in Fracture Mechanics:
        Theory---Numerics---Applications" (2013), equations 3.41--3.44,
        gives ``sigma_xx = 4 a_2`` and ``sigma_yy = 0`` for ``n = 2``, hence
        ``T = 4 a_2``. DOI https://doi.org/10.1007/978-94-007-6680-8;
        Citation Key ``kuna_fracture_mechanics``.
    """
    # Kuna (2013), Eqs. 3.41--3.44: n=2 gives sigma_xx=4 a_2 and
    # sigma_yy=0, so the crack-parallel T-stress is T=4 a_2.
    t_stress = 4 * second_order_symmetric_coefficient
    return t_stress


def derive_williams_in_plane_fracture_quantities(
    terms: Iterable[int],
    a_n: Iterable[float],
    b_n: Iterable[float],
) -> tuple[float, float, float]:
    """Derive in-plane fracture quantities from Williams coefficients.

    Args:
        terms: Williams term numbers in the order shared by both coefficient
            iterables.
        a_n: Symmetric coefficients corresponding to ``terms``, each in
            MPa mm**(1 - n/2).
        b_n: Antisymmetric coefficients corresponding to ``terms``, each in
            MPa mm**(1 - n/2).

    Returns:
        ``(K_I, K_II, T)`` with intensity factors in MPa sqrt(m) and T-stress
        in MPa. A quantity is NaN when its required Williams term is absent.
    Notes:
        The coefficient interpretation follows Williams, "On the Stress
        Distribution at the Base of a Stationary Crack" (1957), DOI
        10.1115/1.4011454, and Kuna, "Finite Elements in Fracture Mechanics:
        Theory---Numerics---Applications" (2013), DOI
        10.1007/978-94-007-6680-8.
    """
    ordered_terms = tuple(terms)
    a_by_term = dict(zip(ordered_terms, a_n))
    b_by_term = dict(zip(ordered_terms, b_n))

    if 1 in a_by_term:
        # First-order symmetric and antisymmetric Williams coefficients encode
        # the Mode I and Mode II singular fields. Both use MPa sqrt(mm), while
        # CrackPy reports stress-intensity factors in MPa sqrt(m).
        k_i = np.sqrt(2 * np.pi) * a_by_term[1] * _SQRT_MM_TO_SQRT_M

        # CrackPy's antisymmetric eigenfield convention maps positive b_1 to
        # negative K_II.
        k_ii = -np.sqrt(2 * np.pi) * b_by_term[1] * _SQRT_MM_TO_SQRT_M
    else:
        k_i = np.nan
        k_ii = np.nan

    if 2 in a_by_term:
        # The second-order symmetric coefficient is the constant stress term.
        t_stress = t_stress_from_williams_coefficient(
            second_order_symmetric_coefficient=a_by_term[2]
        )
    else:
        t_stress = np.nan

    return (
        k_i,
        k_ii,
        t_stress,
    )


def derive_williams_out_of_plane_fracture_quantities(
    terms: Iterable[int],
    c_n: Iterable[float],
) -> tuple[float]:
    """Derive the out-of-plane fracture quantity from Williams coefficients.

    Args:
        terms: Williams term numbers in the order shared by ``c_n``.
        c_n: Out-of-plane coefficients corresponding to ``terms``, each in
            MPa mm**(1 - n/2).

    Returns:
        A one-value ``(K_III,)`` tuple with the intensity factor in MPa sqrt(m),
        or NaN when Williams term 1 is absent.
    Notes:
        The coefficient interpretation follows Williams, "On the Stress
        Distribution at the Base of a Stationary Crack" (1957), DOI
        10.1115/1.4011454, and Kuna, "Finite Elements in Fracture Mechanics:
        Theory---Numerics---Applications" (2013), DOI
        10.1007/978-94-007-6680-8.
    """
    c_by_term = dict(zip(terms, c_n))

    if 1 in c_by_term:
        # The first-order out-of-plane coefficient encodes the Mode III field in
        # MPa sqrt(mm). The length factor converts it to public MPa sqrt(m).
        k_iii = np.sqrt(0.5 * np.pi) * c_by_term[1] * _SQRT_MM_TO_SQRT_M
    else:
        k_iii = np.nan

    return (k_iii,)
