import unittest

import numpy as np
from numpy.testing import assert_allclose

from crackpy.fracture_analysis.crack_tip import williams_stress_field, get_crack_nearfield
from crackpy.structure_elements.material import Material


class CrackTipField(unittest.TestCase):

    def test_williams_field(self):
        a = [1, 1, 1]
        b = [1, 1, 1]
        terms = [1, 2, 3]
        phi = 0
        r = 1
        sigmas = williams_stress_field(a, b, terms, phi, r)
        self.assertEqual(sigmas, [8, 4, 4])

    def test_crack_tip_near_field(self):
        material = Material(E=3, nu_xy=0.5, sig_yield=1)
        k_i = 1
        k_ii = 1
        r = 1

        # first test
        phi_1 = 0
        sigma_tensor_1 = np.asarray([[0.39894228, 0.39894228],
                                     [0.39894228, 0.39894228]])
        eps_tensor_1 = np.asarray([[0.06649038, 0.19947114],
                                   [0.19947114, 0.06649038]])
        u_x_1, v_x_1 = 0.1329807601338109, -0.1329807601338109
        sigma_tensor_ana, eps_tensor_ana, [u_x_ana, v_x_ana] = get_crack_nearfield(k_i, k_ii, r, phi_1, material)
        self.assertIsNone(assert_allclose(sigma_tensor_1, sigma_tensor_ana, atol=1e-4))
        self.assertIsNone(assert_allclose(eps_tensor_1, eps_tensor_ana, atol=1e-4))
        self.assertAlmostEqual(u_x_1, u_x_ana)
        self.assertAlmostEqual(v_x_1, v_x_ana)

        # second test
        phi_1 = np.pi
        sigma_tensor_1 = np.asarray([[-0.7978846,  0],
                                     [0, 0]])
        eps_tensor_1 = np.asarray([[-2.65961520e-01, 0],
                                   [0, 1.32980760e-01]])
        u_x_1, v_x_1 = 0.5319230405352436, 0.5319230405352436
        sigma_tensor_ana, eps_tensor_ana, [u_x_ana, v_x_ana] = get_crack_nearfield(k_i, k_ii, r, phi_1, material)
        self.assertIsNone(assert_allclose(sigma_tensor_1, sigma_tensor_ana, atol=1e-4))
        self.assertIsNone(assert_allclose(eps_tensor_1, eps_tensor_ana, atol=1e-4))
        self.assertAlmostEqual(u_x_1, u_x_ana)
        self.assertAlmostEqual(v_x_1, v_x_ana)


if __name__ == '__main__':
    unittest.main()
