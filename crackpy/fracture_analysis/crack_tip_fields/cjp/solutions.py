"""CJP stress and displacement fields for Mode I and mixed mode."""

from typing import Union

import numpy as np

from crackpy.structure_elements.material import Material


def cjp_stress_field_mixedmode(coeffs: Union[list, np.ndarray], phi: float, r: float) -> list:
    """Formula for the stress field around the crack tip in real polar coordinates by means of the **five-parameter CJP model**.
    [see formulas 3 in Christopher et al. Extension of the CJP model to mixed mode I and mode II (2013)]

    Args:
        coeffs: Z = (A_r, B_r, B_i, C, E) as in Christopher et al. '13
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]

    Returns:
        stresses sigma_x, sigma_y, and sigma_xy

    """
    A_r, B_r, B_i, C, E = coeffs
    sigma_x = (1 / r ** 0.5) * (
            -0.5 * (A_r + 4 * B_r + 8 * E) * np.cos(phi / 2) - 0.5 * B_r * np.cos(5 * phi / 2) +
            0.5 * B_i * (np.sin(5 * phi / 2) + 7 * np.sin(phi / 2)) -
            0.5 * E * (np.log(r) * (np.cos(5 * phi / 2) + 3 * np.cos(phi / 2)) + phi * (
            np.sin(5 * phi / 2) + 3 * np.sin(phi / 2)))
    ) - C

    sigma_y = (1 / r ** 0.5) * (
            0.5 * (A_r - 4 * B_r - 8 * E) * np.cos(phi / 2) + 0.5 * B_r * np.cos(5 * phi / 2) -
            0.5 * B_i * (np.sin(5 * phi / 2) - np.sin(phi / 2)) +
            0.5 * E * (np.log(r) * (np.cos(5 * phi / 2) - 5 * np.cos(phi / 2)) + phi * (
            np.sin(5 * phi / 2) - 5 * np.sin(phi / 2)))
    )

    sigma_xy = (1 / r ** 0.5) * (
            -0.5 * (A_r * np.sin(phi / 2) + B_r * np.sin(5 * phi / 2)) +
            0.5 * B_i * (np.cos(5 * phi / 2) + 3 * np.cos(phi / 2)) -
            - E * np.sin(phi) * (np.log(r) * np.cos(3 * phi / 2) + phi * np.sin(3 * phi / 2))
    )

    return [sigma_x, sigma_y, sigma_xy]


def cjp_displ_field_mixedmode(coeffs: Union[list, np.ndarray], phi: float, r: float, material: Material) -> tuple:
    """Displacement fields around the crack tip in real polar coordinates by means of the **five-parameter CJP model**.
    [see formulas 10 and 11 in Christopher et al. Extension of the CJP model to mixed mode I and mode II (2013)]

    Args:
        coeffs: Z = (A_r, B_r, B_i, C, E) as in Christopher et al. '13
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]
        material: obj of class Material used to calculate *kappa* and **G**

    Returns:
        displacements disp_x, disp_y

    """
    A_r, B_r, B_i, C, E = coeffs
    kappa = material.kappa
    disp_x = r ** 0.5 * (-A_r - 2 * B_r * kappa - 2 * E) * np.cos(phi / 2) + r ** 0.5 * (
            2 * B_i * kappa - 3 * B_i) * np.sin(phi / 2) + \
             r ** 0.5 * (B_r + 2 * E) * np.cos(3 * phi / 2) - r ** 0.5 * B_i * np.sin(3 * phi / 2) + \
             r ** 0.5 * E * (np.log(r) * (np.cos(3 * phi / 2) + (1 - 2 * kappa) * np.cos(phi / 2)) +
                             phi * (np.sin(3 * phi / 2) + (1 + 2 * kappa) * np.sin(phi / 2))) - \
             C / 4 * r * (1 + kappa) * np.cos(phi)
    disp_y = r ** 0.5 * (-2 * B_i * kappa - 3 * B_i) * np.cos(phi / 2) + r ** 0.5 * (
            A_r - 2 * B_r * kappa + 2 * E) * np.sin(phi / 2) + \
             r ** 0.5 * (B_r + 2 * E) * np.sin(3 * phi / 2) + r ** 0.5 * B_i * np.cos(3 * phi / 2) + \
             r ** 0.5 * E * (np.log(r) * (np.sin(3 * phi / 2) - (1 + 2 * kappa) * np.sin(phi / 2)) -
                             phi * (np.cos(3 * phi / 2) + (1 + 2 * kappa) * np.cos(phi / 2))) + \
             C / 4 * r * (3 - kappa) * np.sin(phi)
    disp_x = disp_x / (2 * material.G)
    disp_y = disp_y / (2 * material.G)

    return disp_x, disp_y


def cjp_stress_field_modeI(coeffs: Union[list, np.ndarray], phi: float, r: float) -> list:
    """Formula for the stress field around the crack tip in real polar coordinates by means of the **five-parameter CJP model**.
    This is the mode I formulation based on the original paper Christopher et al. 2007 and recited in James et al. 2013.
    [see formulas 1 and 12 in Camacho-Reyes et al. 2023 for the crack tip stress fields]

    Args:
        coeffs: Z = (A, B, C, E, F)
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]

    Returns:
        stresses sigma_x, sigma_y, and sigma_xy

    """
    A, B, C, E, F = coeffs
    sigma_x = -(1 / r ** 0.5) * (
            -0.5 * (A + 4 * B + 8 * E) * np.cos(phi / 2)
            - 0.5 * B * np.cos(5 * phi / 2) -
            0.5 * E * (np.log(r) * (np.cos(5 * phi / 2) + 3 * np.cos(phi / 2))
                       + phi * (np.sin(5 * phi / 2) + 3 * np.sin(phi / 2)))
    ) - C

    sigma_y = (1 / r ** 0.5) * (
            0.5 * (A - 4 * B - 8 * E) * np.cos(phi / 2)
            + 0.5 * B * np.cos(5 * phi / 2) +
            0.5 * E * (np.log(r) * (np.cos(5 * phi / 2) - 5 * np.cos(phi / 2))
                       + phi * (np.sin(5 * phi / 2) - 5 * np.sin(phi / 2)))
    )

    sigma_xy = (1 / r ** 0.5) * -0.5 * (
            A * np.sin(phi / 2) + B * np.sin(5 * phi / 2)
            - E * np.sin(phi) * (np.log(r) * np.cos(3 * phi / 2) + phi * np.sin(3 * phi / 2))
    )

    return [sigma_x, sigma_y, sigma_xy]


def cjp_displ_field_modeI(coeffs: Union[list, np.ndarray], phi: float, r: float, material: Material) -> tuple:
    """Displacement fields around the crack tip in real polar coordinates by means of the **five-parameter CJP model**.
    This is the mode I formulation based on the original paper Christopher et al. 2007 and recited in James et al. 2013.
    [see formulas 10 and 11 in Camacho-Reyes et al. 2023 for the crack tip displacement fields]

    Args:
        coeffs: Z = (A, B, C, E, F), following the parameter mapping in Camacho-Reyes et al. (2023)
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]
        material: obj of class Material used to calculate *kappa* and **G**

    Returns:
        displacements disp_x, disp_y

    """
    A, B, C, E, F = coeffs
    kappa = material.kappa
    disp_x = r ** 0.5 * (-A - 2 * B * kappa - 2 * E) * np.cos(phi / 2) + r ** 0.5 * (B + 2 * E) * np.cos(3 * phi / 2) - \
             r ** 0.5 * E * (np.log(r) * (np.cos(3 * phi / 2) + (1 - 2 * kappa) * np.cos(phi / 2)) + phi * (
            np.sin(3 * phi / 2) +
            (2 * kappa - 1) * np.sin(phi / 2))) - \
             r * C * (1 + kappa) * np.cos(phi) / 4 + \
             r * F * (kappa - 3) * np.cos(phi) / 4

    disp_y = r ** 0.5 * (A - 2 * B * kappa + 2 * E) * np.sin(phi / 2) + r ** 0.5 * (B + 2 * E) * np.sin(3 * phi / 2) + \
             r ** 0.5 * E * (np.log(r) * (np.sin(3 * phi / 2) - (1 + 2 * kappa) * np.sin(phi / 2)) - phi * (
            np.cos(3 * phi / 2) +
            (2 * kappa + 1) * np.cos(phi / 2))) + \
             r * C * (3 - kappa) * np.sin(phi) / 4 + \
             r * F * (kappa + 1) * np.sin(phi) / 4

    disp_x = disp_x / (2 * material.G)
    disp_y = disp_y / (2 * material.G)

    return disp_x, disp_y
