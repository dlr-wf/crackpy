import unittest

import numpy as np

from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.material import Material


class TestInputData(unittest.TestCase):
    def setUp(self):
        coor_x = [-1, 0, 1]
        coor_y = [-1, 0, 1]
        disp_x = [0, 1, 2]
        disp_y = [0, 1, 2]
        eps_x = [1, 2, 3]
        eps_y = [-1, 2, 3]
        eps_xy = [5, 6, 7]

        self.data = InputData()
        self.data.set_data_manually(coor_x, coor_y, disp_x, disp_y, eps_x, eps_y, eps_xy)

    def test_calc_eps_vm(self):
        eps_vm_exp = np.asarray([5.88784058, 8., 10.06644591])
        self.data.calc_eps_vm()

        # check equality to calculated results
        self.assertAlmostEqual(np.sum((self.data.eps_vm - eps_vm_exp)).item(), 0, delta=1e-5)

    def test_calc_sig_vm(self):
        sig_vm_exp = np.asarray([478110.36, 602247.03, 731257.40])
        self.data.calc_stresses(material=Material())

        # check equality to calculated results
        self.assertAlmostEqual(np.sum((self.data.sig_vm - sig_vm_exp)).item(), 0, delta=1e-2)

    def test_transform_data(self):
        self.data.calc_stresses(material=Material())
        self.data.transform_data(x_shift=10, y_shift=-10, angle=45)

        # expected results
        coor_x_t = np.asarray([-1.41421356, 0, 1.41421356])
        coor_y_t = np.asarray([14.1421356, 14.1421356, 14.1421356])
        disp_x_t = np.asarray([0, 1.41421356, 2.828427])
        disp_y_t = np.asarray([0, 0, 0])
        eps_x_t = np.asarray([5, 8, 10])
        eps_y_t = np.asarray([-5, -4, -4])
        eps_xy_t = np.asarray([-1, 0, 0])

        # check equality to calculated results
        self.assertAlmostEqual(np.sum((self.data.coor_x - coor_x_t)**2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.coor_y - coor_y_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.disp_x - disp_x_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.disp_y - disp_y_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.eps_x - eps_x_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.eps_y - eps_y_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.eps_xy - eps_xy_t) ** 2).item(), 0, delta=1e-5)

    def test_transform_data_wo_stresses(self):
        self.data.transform_data(x_shift=10, y_shift=-10, angle=45)

        # expected results
        coor_x_t = np.asarray([-1.41421356, 0, 1.41421356])
        coor_y_t = np.asarray([14.1421356, 14.1421356, 14.1421356])
        disp_x_t = np.asarray([0, 1.41421356, 2.828427])
        disp_y_t = np.asarray([0, 0, 0])
        eps_x_t = np.asarray([5, 8, 10])
        eps_y_t = np.asarray([-5, -4, -4])
        eps_xy_t = np.asarray([-1, 0, 0])

        # check equality to calculated results
        self.assertAlmostEqual(np.sum((self.data.coor_x - coor_x_t)**2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.coor_y - coor_y_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.disp_x - disp_x_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.disp_y - disp_y_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.eps_x - eps_x_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.eps_y - eps_y_t) ** 2).item(), 0, delta=1e-5)
        self.assertAlmostEqual(np.sum((self.data.eps_xy - eps_xy_t) ** 2).item(), 0, delta=1e-5)


if __name__ == '__main__':
    unittest.main()
