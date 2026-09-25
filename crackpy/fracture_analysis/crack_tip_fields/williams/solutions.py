"""Williams stress and displacement fields and coefficient units."""

from fractions import Fraction
from typing import Union

import numpy as np

from crackpy.structure_elements.material import Material


def williams_in_plane_stress_field(a: Union[list, np.ndarray], b: Union[list, np.ndarray], terms: Union[list, np.ndarray],
                          phi: float, r: float) -> list:
    """Formula for the stress field around the crack tip in polar coordinates by Williams.
    [Meinhard Kuna - Numerische Beanspruchungsanalyse formulas (3.41)-(3.42)]

    Args:
        a: Williams coefficient
        b: Williams coefficient
        terms: defines the used Williams coefficients
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]

    Returns:
        stresses sigma_x, sigma_y, and sigma_xy

    """
    sigma_x = 0.0
    sigma_y = 0.0
    sigma_xy = 0.0
    for index, n in enumerate(terms):
        sigma_x += n / 2 * r ** (n / 2 - 1) * (a[index] * ((2 + n / 2 + (-1) ** n) * np.cos((n / 2 - 1) * phi)
                                                           - (n / 2 - 1) * np.cos((n / 2 - 3) * phi))
                                               - b[index] * ((2 + n / 2 - (-1) ** n) * np.sin((n / 2 - 1) * phi)
                                                             - (n / 2 - 1) * np.sin((n / 2 - 3) * phi)))
        sigma_y += n / 2 * r ** (n / 2 - 1) * (a[index] * ((2 - n / 2 - (-1) ** n) * np.cos((n / 2 - 1) * phi)
                                                           + (n / 2 - 1) * np.cos((n / 2 - 3) * phi))
                                               - b[index] * ((2 - n / 2 + (-1) ** n) * np.sin((n / 2 - 1) * phi)
                                                             + (n / 2 - 1) * np.sin((n / 2 - 3) * phi)))
        sigma_xy += n / 2 * r ** (n / 2 - 1) * (a[index] * ((n / 2 - 1) * np.sin((n / 2 - 3) * phi)
                                                            - (n / 2 + (-1) ** n) * np.sin((n / 2 - 1) * phi))
                                                + b[index] * ((n / 2 - 1) * np.cos((n / 2 - 3) * phi)
                                                              - (n / 2 - (-1) ** n) * np.cos((n / 2 - 1) * phi)))
    return [sigma_x, sigma_y, sigma_xy]


def williams_in_plane_displacement_field(a: Union[list, np.ndarray], b: Union[list, np.ndarray], terms: Union[list, np.ndarray],
                         phi: float, r: float, material: Material) -> tuple:
    """Formula for the displacement fields in x- and y-direction around the crack tip in polar coordinates by Williams.
    [Meinhard Kuna - Numerische Beanspruchungsanalyse formulas (3.43)-(3.44)]

    Args:
        a: Williams coefficient
        b: Williams coefficient
        terms: defines the used Williams coefficients
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]
        material: obj of class Material used to calculate *kappa*

    Returns:
        displacements disp_x, disp_y

    """
    kappa = material.kappa
    disp_x = 0.0
    disp_y = 0.0
    for index, n in enumerate(terms):
        F_1 = (kappa + (-1.0) ** n + n / 2) * np.cos(n / 2 * phi) - n / 2 * np.cos((n / 2 - 2) * phi)
        G_1 = (-kappa + (-1.0) ** n - n / 2) * np.sin(n / 2 * phi) + n / 2 * np.sin((n / 2 - 2) * phi)
        F_2 = (kappa - (-1.0) ** n - n / 2) * np.sin(n / 2 * phi) + n / 2 * np.sin((n / 2 - 2) * phi)
        G_2 = (kappa + (-1.0) ** n - n / 2) * np.cos(n / 2 * phi) + n / 2 * np.cos((n / 2 - 2) * phi)

        disp_x += 1 / (2 * material.G) * r ** (n / 2) * (a[index] * F_1 + b[index] * G_1)
        disp_y += 1 / (2 * material.G) * r ** (n / 2) * (a[index] * F_2 + b[index] * G_2)

    return disp_x, disp_y


def williams_combined_stress_field(a: Union[list, np.ndarray], b: Union[list, np.ndarray], c: Union[list, np.ndarray],
                             terms: Union[list, np.ndarray],
                             phi: float, r: float) -> list:
    """Formula for the stress field around the crack tip in polar coordinates by Williams.
    [Meinhard Kuna - Numerische Beanspruchungsanalyse formulas (3.41)-(3.55)]

    Args:
        a: Williams coefficient
        b: Williams coefficient
        c: Williams coefficient
        terms: defines the used Williams coefficients
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]

    Returns:
        stresses sigma_x, sigma_y, sigma_xy, sigma_xz, and sigma_yz,

    """
    sigma_x = 0.0
    sigma_y = 0.0
    sigma_xy = 0.0
    sigma_xz = 0.0
    sigma_yz = 0.0
    for index, n in enumerate(terms):
        sigma_x += n / 2 * r ** (n / 2 - 1) * (a[index] * ((2 + n / 2 + (-1) ** n) * np.cos((n / 2 - 1) * phi)
                                                           - (n / 2 - 1) * np.cos((n / 2 - 3) * phi))
                                               - b[index] * ((2 + n / 2 - (-1) ** n) * np.sin((n / 2 - 1) * phi)
                                                             - (n / 2 - 1) * np.sin((n / 2 - 3) * phi)))
        sigma_y += n / 2 * r ** (n / 2 - 1) * (a[index] * ((2 - n / 2 - (-1) ** n) * np.cos((n / 2 - 1) * phi)
                                                           + (n / 2 - 1) * np.cos((n / 2 - 3) * phi))
                                               - b[index] * ((2 - n / 2 + (-1) ** n) * np.sin((n / 2 - 1) * phi)
                                                             + (n / 2 - 1) * np.sin((n / 2 - 3) * phi)))
        sigma_xy += n / 2 * r ** (n / 2 - 1) * (a[index] * ((n / 2 - 1) * np.sin((n / 2 - 3) * phi)
                                                            - (n / 2 + (-1) ** n) * np.sin((n / 2 - 1) * phi))
                                                - b[index] * ((n / 2 - 1) * np.cos((n / 2 - 3) * phi)
                                                              - (n / 2 - (-1) ** n) * np.cos((n / 2 - 1) * phi)))

        if n % 2 == 0:
            L_13 = n / 2 * np.cos(n / 2 - 1) * phi
            L_23 = -n / 2 * np.sin(n / 2 - 1) * phi

        else:
            L_13 = n / 2 * np.sin(n / 2 - 1) * phi
            L_23 = n / 2 * np.cos(n / 2 - 1) * phi

        sigma_xz += n / 2 * r ** (n / 2 - 1) * c[index] * L_13
        sigma_yz += n / 2 * r ** (n / 2 - 1) * c[index] * L_23

    return [sigma_x, sigma_y, sigma_xy, sigma_xz, sigma_yz]


def williams_out_of_plane_displacement_field(c: Union[list, np.ndarray],
                            terms: Union[list, np.ndarray],
                            phi: float, r: float, material: Material) -> float:
    """Formula for the displacement fields in z-direction around the crack tip in polar coordinates by Williams.
    [Meinhard Kuna - Numerische Beanspruchungsanalyse formulas (3.52)-(3.55)]

    Args:
        c: Williams coefficient
        terms: defines the used Williams coefficients
        phi: angle from polar coordinates [rad]
        r: radius from polar coordinates [mm]
        material: obj of class Material used to calculate *kappa*

    Returns:
        displacements disp_x, disp_y, disp_z

    """

    disp_z = 0.0
    for index, n in enumerate(terms):

        if n % 2 == 0:
            H_3 = 2 * np.cos(n / 2 * phi)

        else:
            H_3 = 2 * np.sin(n / 2 * phi)

        disp_z += 1 / (2 * material.G) * r ** (n / 2) * (c[index] * H_3)

    return disp_z


def williams_in_plane_eigenfunction(n: int, a_n: float, b_n: float, r: float, theta: float, material: Material) -> tuple:
    """The n-the eigenfunctions of the planar crack problem in real polar coordinates.
    [see Meinhard Kuna equations (3.41-3.44)]

    Args:
        n: order of series coefficient
        a_n: first coefficient (a_1 ~ K_I, a_2 ~ T-stress)
        b_n: second coefficent (b_2 ~ K_II)
        r: radius from polar coordinates [mm]
        theta: angle from polar coordinates [rad]
        material: obj of class Material used to calculate *kappa*

    Returns:
        sigma_x, sigma_y, sigma_xy, disp_x, disp_y
            of order n with coefficients a_n and b_n and angle theta and radius r

    """
    M_11 = n / 2 * ((2 + (-1) ** n + n / 2) * np.cos((n / 2 - 1) * theta) - (n / 2 - 1) * np.cos((n / 2 - 3) * theta))
    N_11 = n / 2 * ((-2 + (-1) ** n - n / 2) * np.sin((n / 2 - 1) * theta) + (n / 2 - 1) * np.sin((n / 2 - 3) * theta))
    M_22 = n / 2 * ((2 - (-1) ** n - n / 2) * np.cos((n / 2 - 1) * theta) + (n / 2 - 1) * np.cos((n / 2 - 3) * theta))
    N_22 = n / 2 * ((-2 - (-1) ** n + n / 2) * np.sin((n / 2 - 1) * theta) - (n / 2 - 1) * np.sin((n / 2 - 3) * theta))
    M_12 = n / 2 * ((n / 2 - 1) * np.sin((n / 2 - 3) * theta) - (n / 2 + (-1) ** n) * np.sin((n / 2 - 1) * theta))
    N_12 = n / 2 * ((n / 2 - 1) * np.cos((n / 2 - 3) * theta) - (n / 2 - (-1) ** n) * np.cos((n / 2 - 1) * theta))

    sigma_x_n = r ** (n / 2 - 1) * (a_n * M_11 + b_n * N_11)
    sigma_y_n = r ** (n / 2 - 1) * (a_n * M_22 + b_n * N_22)
    sigma_xy_n = r ** (n / 2 - 1) * (a_n * M_12 + b_n * N_12)

    kappa = material.kappa
    F_1 = (kappa + (-1) ** n + n / 2) * np.cos(n / 2 * theta) - n / 2 * np.cos((n / 2 - 2) * theta)
    G_1 = (-kappa + (-1) ** n - n / 2) * np.sin(n / 2 * theta) + n / 2 * np.sin((n / 2 - 2) * theta)
    F_2 = (kappa - (-1) ** n - n / 2) * np.sin(n / 2 * theta) + n / 2 * np.sin((n / 2 - 2) * theta)
    G_2 = (kappa + (-1) ** n - n / 2) * np.cos(n / 2 * theta) + n / 2 * np.cos((n / 2 - 2) * theta)

    u_x_n = 1 / (2 * material.G) * r ** (n / 2) * (a_n * F_1 + b_n * G_1)
    u_y_n = 1 / (2 * material.G) * r ** (n / 2) * (a_n * F_2 + b_n * G_2)

    return sigma_x_n, sigma_y_n, sigma_xy_n, u_x_n, u_y_n


def unit_of_williams_coefficients(n):
    """
    Returns the unit of the Williams coefficent.
    """
    if n == 2:
        unit = 'MPa'
    else:
        unit = f'MPa*mm^{{{Fraction(-n / 2 + 1)}}}'
    return unit
