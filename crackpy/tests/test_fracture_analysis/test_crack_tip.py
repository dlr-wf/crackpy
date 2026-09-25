import unittest
from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_allclose

from crackpy.fracture_analysis.crack_tip import (
    get_crack_nearfield,
    williams_stress_field,
)
from crackpy.structure_elements.material import Material


class CrackTipField(unittest.TestCase):

    def test_williams_field(self):
        a = [1, 1, 1]
        b = [1, 1, 1]
        terms = [1, 2, 3]
        phi = 0
        r = 1
        sigmas = williams_stress_field(a, b, terms, phi, r)
        self.assertEqual(sigmas, [8, 4, -4])

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


@pytest.mark.parametrize("module, names", [
    ("williams.solutions", (
        "williams_stress_field", "williams_displ_field_xy", "williams_stress_field_3d",
        "williams_displ_field_z", "eigenfunction", "unit_of_williams_coefficients",
    )),
    ("cjp.solutions", (
        "cjp_stress_field_mixedmode", "cjp_displ_field_mixedmode",
        "cjp_stress_field_modeI", "cjp_displ_field_modeI",
    )),
    ("auxiliary", ("get_crack_nearfield", "get_zhao_solutions")),
])
def test_legacy_field_imports_preserve_functions(module, names):
    legacy = import_module("crackpy.fracture_analysis.crack_tip")
    owner = import_module("crackpy.fracture_analysis.crack_tip_fields." + module)
    renamed = {
        "williams_stress_field": "williams_in_plane_stress_field",
        "williams_displ_field_xy": "williams_in_plane_displacement_field",
        "williams_stress_field_3d": "williams_combined_stress_field",
        "williams_displ_field_z": "williams_out_of_plane_displacement_field",
        "eigenfunction": "williams_in_plane_eigenfunction",
    }
    for name in names:
        assert getattr(legacy, name) is getattr(owner, renamed.get(name, name))
