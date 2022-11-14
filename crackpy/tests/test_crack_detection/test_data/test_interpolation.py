import os
import unittest
import numpy as np

import crackpy.crack_detection.data.datapreparation as dp
from crackpy.crack_detection.data.interpolation import interpolate_on_array
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums


class TestInterpolation(unittest.TestCase):

    def test_interpolation(self):
        origin = os.path.join(  # '..', '..', '..', '..',
                              'test_data', 'crack_detection', 'raw')
        side = 'left'
        size = 70

        stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(os.path.join(origin, 'Nodemaps'), ['7'])

        # import
        inputs, _ = dp.import_data(nodemaps=stages_to_nodemaps,
                                   data_path=origin,
                                   side=side,
                                   exists_target=False)
        # interpolate
        interp_size = size if side == 'right' else size * -1
        interp_coors, interp_disps, interp_eps_vm = interpolate_on_array(input_by_nodemap=inputs,
                                                                         interp_size=interp_size,
                                                                         pixels=256)
        # tests
        self.assertIsInstance(interp_coors, dict)
        self.assertIsInstance(interp_disps, dict)
        self.assertIsInstance(interp_eps_vm, dict)

        for coors in interp_coors.values():
            self.assertIsInstance(coors, np.ndarray)
            self.assertEqual(coors.shape, (2, 256, 256))
            self.assertAlmostEqual(coors[0, 0, 1], -0.2745098, delta=1e-6)

        for disps in interp_disps.values():
            self.assertIsInstance(disps, np.ndarray)
            self.assertEqual(disps.shape, (2, 256, 256))
            self.assertAlmostEqual(disps[0, 0, 0], -0.017540, delta=1e-6)

        for eps_vm in interp_eps_vm.values():
            self.assertIsInstance(eps_vm, np.ndarray)
            self.assertEqual(eps_vm.shape, (256, 256))
            self.assertAlmostEqual(eps_vm[0, 0], 0.000952, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
