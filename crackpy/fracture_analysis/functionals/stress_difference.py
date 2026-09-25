"""The Stress-Difference Method estimates elastic T-Stress from normal stresses
sampled on the crack-extension line ahead of the crack tip.
"""

import numpy as np


def t_stress_from_stress_difference(
    crack_parallel_stress: np.ndarray,
    crack_opening_stress: np.ndarray,
) -> np.ndarray:
    """Estimate elastic T-Stress by the Stress-Difference Method.

    Args:
        crack_parallel_stress: Normal stress parallel to the crack plane,
            sampled on the crack-extension line ahead of the crack tip, in
            MPa.
        crack_opening_stress: Normal stress perpendicular to the crack plane,
            sampled at the same positions, in MPa.

    Returns:
        Pointwise elastic T-Stress estimates in MPa.

    Notes:
        Yang and Ravi-Chandar (1999), equation 7, define
        ``T = lim[r -> 0] (sigma_11 - sigma_22)`` at ``theta = 0``.
        DOI: https://doi.org/10.1016/S0013-7944(99)00082-X.
        Citation key: ``yang_ravi_chandar_1999_stress_difference``.
    """
    # Yang and Ravi-Chandar (1999), Eq. (7), on the crack-extension line
    # theta = 0: T = lim[r -> 0] (sigma_11 - sigma_22).
    t_stress = crack_parallel_stress - crack_opening_stress
    return t_stress
