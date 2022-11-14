from fractions import Fraction

import numpy as np

from crackpy.structure_elements.material import Material


def williams_stress_field(a: list or np.array, b: list or np.array, terms: list or np.array,
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
        sigma_x += n/2 * r**(n/2 - 1) * (a[index] * ((2 + n/2 + (-1)**n) * np.cos((n/2 - 1) * phi)
                                                 - (n / 2 - 1) * np.cos((n / 2 - 3) * phi))
                                         - b[index] * ((2 + n/2 - (-1)**n) * np.sin((n/2 - 1) * phi)
                                                   - (n/2 - 1) * np.sin((n/2 - 3) * phi)))
        sigma_y += n/2 * r**(n/2 - 1) * (a[index] * ((2 - n/2 - (-1)**n) * np.cos((n/2 - 1) * phi)
                                                 + (n/2 - 1) * np.cos((n/2 - 3) * phi))
                                         - b[index] * ((2 - n/2 + (-1)**n) * np.sin((n/2 - 1) * phi)
                                                   + (n/2 - 1) * np.sin((n/2 - 3) * phi)))
        sigma_xy += n/2 * r**(n/2 - 1) * (a[index] * ((n/2 - 1) * np.sin((n/2 - 3) * phi)
                                                  - (n/2 + (-1)**n) * np.sin((n/2 - 1) * phi))
                                          - b[index] * ((n/2 - 1) * np.cos((n/2 - 3) * phi)
                                                    - (n/2 - (-1)**n) * np.cos((n/2 - 1) * phi)))
    return [sigma_x, sigma_y, sigma_xy]


def cjp_displ_field(coeffs: list or np.array, phi: float, r: float, material: Material) -> tuple:
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
    disp_x = r**0.5 * (-A_r - 2*B_r*kappa - 2*E) * np.cos(phi/2) + r**0.5 * (2*B_i*kappa - 3*B_i) * np.sin(phi/2) + \
             r**0.5 * (B_r + 2*E) * np.cos(3*phi/2) - r**0.5 * B_i * np.sin(3*phi/2) + \
             r**0.5 * E * (np.log(r) * (np.cos(3*phi/2) + (1 - 2*kappa) * np.cos(phi/2)) +
                           phi * (np.sin(3*phi/2) + (1 + 2*kappa) * np.sin(phi/2))) - \
             C/4 * r * (1 + kappa) * np.cos(phi)
    disp_y = r**0.5 * (-2*B_i*kappa - 3*B_i) * np.cos(phi/2) + r**0.5 * (A_r - 2*B_r*kappa + 2*E) * np.sin(phi/2) + \
             r**0.5 * (B_r + 2*E) * np.sin(3*phi/2) + r**0.5 * B_i * np.cos(3*phi/2) + \
             r**0.5 * E * (np.log(r) * (np.sin(3*phi/2) - (1 + 2*kappa) * np.sin(phi/2)) -
                           phi * (np.cos(3*phi/2) + (1 + 2*kappa) * np.cos(phi/2))) + \
             C/4 * r * (3 - kappa) * np.sin(phi)
    disp_x = disp_x / (2*material.G)
    disp_y = disp_y / (2*material.G)

    return disp_x, disp_y


def williams_displ_field(a: list or np.array, b: list or np.array, terms: list or np.array,
                         phi: float, r: float, material: Material) -> tuple:
    """Formula for the displacement fields around the crack tip in polar coordinates by Williams.
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


def eigenfunction(n: int, a_n: float, b_n: float, r: float, theta: float, material: Material) -> tuple:
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
    M_11 = n/2 * ((2 + (-1)**n + n/2) * np.cos((n/2 - 1) * theta) - (n/2 - 1) * np.cos((n/2 - 3) * theta))
    N_11 = n/2 * ((-2 + (-1)**n - n/2) * np.sin((n/2 - 1) * theta) + (n/2 - 1) * np.sin((n/2 - 3) * theta))
    M_22 = n/2 * ((2 - (-1)**n - n/2) * np.cos((n/2 - 1) * theta) + (n/2 - 1) * np.cos((n/2 - 3) * theta))
    N_22 = n/2 * ((-2 - (-1) ** n + n/2) * np.sin((n/2 - 1) * theta) - (n/2 - 1) * np.sin((n/2 - 3) * theta))
    M_12 = n/2 * ((n/2 - 1) * np.sin((n/2 - 3) * theta) - (n/2 + (-1)**n) * np.sin((n/2 - 1) * theta))
    N_12 = n/2 * ((n/2 - 1) * np.cos((n/2 - 3) * theta) - (n/2 - (-1)**n) * np.cos((n/2 - 1) * theta))

    sigma_x_n = r ** (n/2 - 1) * (a_n * M_11 + b_n * N_11)
    sigma_y_n = r ** (n/2 - 1) * (a_n * M_22 + b_n * N_22)
    sigma_xy_n = r ** (n/2 - 1) * (a_n * M_12 + b_n * N_12)

    kappa = material.kappa
    F_1 = (kappa + (-1)**n + n/2) * np.cos(n/2 * theta) - n/2 * np.cos((n/2 - 2) * theta)
    G_1 = (-kappa + (-1)**n - n/2) * np.sin(n/2 * theta) + n/2 * np.sin((n/2 - 2) * theta)
    F_2 = (kappa - (-1)**n - n/2) * np.sin(n/2 * theta) + n/2 * np.sin((n/2 - 2) * theta)
    G_2 = (kappa + (-1)**n - n/2) * np.cos(n/2 * theta) + n/2 * np.cos((n/2 - 2) * theta)

    u_x_n = 1 / (2 * material.G) * r ** (n/2) * (a_n * F_1 + b_n * G_1)
    u_y_n = 1 / (2 * material.G) * r ** (n/2) * (a_n * F_2 + b_n * G_2)

    return sigma_x_n, sigma_y_n, sigma_xy_n, u_x_n, u_y_n


def get_crack_nearfield(k_i: float, k_ii: float, r: float, phi: float, material: Material) -> tuple:
    """Formula for the analytical stress and strain using the crack near field.
    [see Eq. 3,4 in Sladek et al. Contour integrals for mixed-mode crack analysis: effect of non-singular terms (1997)]

    Args:
        k_i: first stress intensity factor K_I
        k_ii: second stress intensity factor K_II
        r: radius from polar coordinates [mm]
        phi: angle from polar coordinates [rad]
        material: obj of class Material

    Returns:
        stress tensor, strain tensor, [displacement x, displacement y]

    """
    kappa = material.kappa

    sigma_x_ana = k_i / (np.sqrt(2 * np.pi * r)) * np.cos(phi / 2) * (1 - np.sin(phi / 2) * np.sin(3 * phi / 2)) \
                  - k_ii / (np.sqrt(2 * np.pi * r)) * np.sin(phi / 2) * (2 + np.cos(phi / 2) * np.cos(3 * phi / 2))

    sigma_y_ana = k_i / (np.sqrt(2 * np.pi * r)) * np.cos(phi / 2) * (1 + np.sin(phi / 2) * np.sin(3 * phi / 2)) \
                  + k_ii / (np.sqrt(2 * np.pi * r)) * np.sin(phi / 2) * np.cos(phi / 2) * np.cos(3 * phi / 2)

    sigma_xy_ana = k_i / (np.sqrt(2 * np.pi * r)) * np.sin(phi / 2) * np.cos(phi / 2) * np.cos(3 * phi / 2) \
                   + k_ii / (np.sqrt(2 * np.pi * r)) * np.cos(phi / 2) * (1 - np.sin(phi / 2) * np.sin(3 * phi / 2))

    # the following two formulas differ from the reference but are identical (by trigonometric identities)
    u_x_ana = k_i / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (np.cos(phi / 2) * (kappa - np.cos(phi))) \
              + k_ii / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (np.sin(phi / 2) * (kappa + 2 + np.cos(phi)))

    v_x_ana = k_i / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (np.sin(phi / 2) * (kappa - np.cos(phi))) \
              + k_ii / (2 * material.G) * np.sqrt(r / (2 * np.pi)) * (-np.cos(phi / 2) * (kappa - 2 + np.cos(phi)))

    eps_vector_ana = np.dot(np.linalg.inv(material.stiffness_matrix), [sigma_x_ana, sigma_y_ana, sigma_xy_ana])
    eps_tensor_ana = np.asarray([[eps_vector_ana[0], eps_vector_ana[2]], [eps_vector_ana[2], eps_vector_ana[1]]])
    sigma_tensor_ana = np.asarray([[sigma_x_ana, sigma_xy_ana], [sigma_xy_ana, sigma_y_ana]])

    return sigma_tensor_ana, eps_tensor_ana, [u_x_ana, v_x_ana]


def get_zhao_solutions(r: float, phi: float, material: Material, force: float = 1, dist: float = 1) -> tuple:
    """Returns stress and displacement test functions according to formulas (4a-b) in
    Zhao et al. 'Stress intensity factor K and the elastic T-stress for corner cracks' (2001)

    Args:
        r: radius from polar coordinates [mm]
        phi: angle from polar coordinates [rad]
        material: obj of class Material
        force: point force applied at the crack tip
        dist: reference distance from the crack tip on the x-axis

    Returns:
        stress_11, stress_22, stress_12, disp_x, disp_y

    """
    sigma_x = - force / (np.pi * r) * np.cos(phi)**3
    sigma_y = - force / (np.pi * r) * np.cos(phi) * np.sin(phi)**2
    sigma_xy = - force / (np.pi * r) * np.cos(phi)**2 * np.sin(phi)

    u_x = - (1 - material.nu_xy**2) / material.E * force / np.pi * (np.log(r / dist)
                                                                    + np.sin(phi)**2 / (2 * (1 - material.nu_xy)))
    u_y = - (1 + material.nu_xy) / (2 * material.E) * force / np.pi * ((1 - 2 * material.nu_xy) * phi
                                                                       - np.cos(phi) * np.sin(phi))

    return sigma_x, sigma_y, sigma_xy, u_x, u_y


def unit_of_williams_coefficients(n):
    """
    Returns the unit of the Williams coefficent.
    """
    if n == 2:
        unit = 'MPa'
    else:
        unit = f'MPa*mm^{{{Fraction(-n/2+1)}}}'
    return unit
