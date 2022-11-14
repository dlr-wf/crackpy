import unittest

import numpy as np
from numpy.testing import assert_allclose

from crackpy.structure_elements.material import Material


class TestMaterialProperties(unittest.TestCase):

    def setUp(self):
        self.material = Material(
            E=10,
            nu_xy=0.5,
            sig_yield=5
        )

    def test_stiffness(self):
        actual_stiffness = np.asarray([[13.33333333, 6.66666667, 0],
                                       [6.66666667, 13.33333333, 0],
                                       [0, 0, 6.66666667]])
        self.assertIsNone(assert_allclose(actual_stiffness, self.material.stiffness_matrix))

    def test_inverse_stiffness(self):
        actual_inverse = np.asarray([[0.1, -0.05, 0],
                                     [-0.05, 0.1, 0],
                                     [0, 0, 0.15]])
        self.assertIsNone(assert_allclose(actual_inverse, self.material.inverse_stiffness_matrix))


if __name__ == '__main__':
    unittest.main()
