import numpy as np
from numpy import linalg as LA
import logging

"""
    This script compares formulas to compute the equivalent strain from the strain tensor.
"""

# Logging
logger = logging.getLogger(__name__)

# Example strain tensor from ARAMIS nodemap export
eps_xx = -0.048588741570711 / 100.0
eps_yy = 1.085232496261597 / 100.0
eps_xy = -0.003783278865740
eps_eqv = 1.301692605018616 / 100.0

eps = np.array([[eps_xx, eps_xy],
                [eps_xy, eps_yy]])

# Zeiss GOM Aramis implementation of equivalent strain using large strain theory
# See: https://techguide.zeiss.com/en/gom-software-2022/article/cmd_comparison_check_scalar_mises_strain.html

# large strains
w, _ = LA.eig(eps)
eps_1 = w[0]
eps_2 = w[1]
phi_1 = np.log(1 + eps_1)
phi_2 = np.log(1 + eps_2)
phi_3 = phi_1 + phi_2

phi_M = np.sqrt(2 / 3 * (phi_1 ** 2 + phi_2 ** 2 + phi_3 ** 2))
eps_M_large_strains = np.exp(phi_M) - 1

# small strains
w, _ = LA.eig(eps)
eps_1 = w[0]
eps_2 = w[1]
eps_3 = eps_1 + eps_2

eps_M_small_strains = np.sqrt(2 / 3 * (eps_1 ** 2 + eps_2 ** 2 + eps_3 ** 2))

# Wikipedia: A definition that is commonly used in the literature on plasticity is:
# https://en.wikipedia.org/wiki/Infinitesimal_strain_theory#Equivalent_strain

nu = 0.5  # Poisson's ratio set to 0.5 which means incompressible material
eps_zz = -nu / (1 - nu) * (eps_xx + eps_yy)

eps = np.array([[eps_xx, eps_xy,      0],
                [eps_xy, eps_yy,      0],
                [0,           0, eps_zz]])

eps_dev = eps - 1 / 3 * np.trace(eps) * np.eye(3)
eps_vm = np.sqrt(2 / 3 * np.trace(eps_dev @ eps_dev))  # @ is matrix multiplication

logger.info(f"Equivalent strain (Wikipedia, nu={nu}): {eps_vm}")
logger.info(f"Equivalent strain (ARAMIS nodemap export): {eps_eqv}")
logger.info(f"Equivalent strain (ARAMIS, large strains): {eps_M_large_strains}")
logger.info(f"Equivalent strain (ARAMIS, small strains): {eps_M_small_strains}")
