import unittest
import os

import numpy as np

import crackpy.crack_detection.data.datapreparation as dp
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums


class TestDataImport(unittest.TestCase):

    def setUp(self):
        self.origin = os.path.join(  # '..', '..', '..', '..',
                                   'test_data', 'crack_detection', 'raw')
        self.side = 'left'
        self.key_list = [
            'AllDataPoints_1_7.txt_' + self.side,
            'AllDataPoints_1_407.txt_' + self.side,
            'AllDataPoints_1_807.txt_' + self.side
        ]
        self.stages_to_nodemaps, _ = \
            get_nodemaps_and_stage_nums(os.path.join(self.origin, 'Nodemaps'), 'All')

    def test_import_data(self):
        inputs, ground_truths = dp.import_data(nodemaps=self.stages_to_nodemaps,
                                               data_path=self.origin,
                                               side=self.side,
                                               exists_target=False)
        self.assertIsNone(ground_truths)

        for key, actual in zip(inputs.keys(), self.key_list):
            self.assertEqual(key, actual)
            self.assertEqual(inputs[key].__class__.__name__, 'InputData')

        # tests specific to the nodemaps 'AllDataPoints_1_7.txt'
        self.assertEqual(inputs[self.key_list[0]].coor_x.shape, (37845,))
        self.assertEqual(inputs[self.key_list[0]].coor_x[0], 79.9941056528)

    def test_ground_truth_import(self):
        _, ground_truths = dp.import_data(nodemaps=self.stages_to_nodemaps,
                                          data_path=self.origin,
                                          side=self.side,
                                          exists_target=True)
        self.assertIsNotNone(ground_truths)

        for data in ground_truths.values():
            self.assertIsInstance(data, np.ndarray)

        # tests specific to the ground truth 'AllDataPoints_1_7.txt_left'
        self.assertEqual(ground_truths[self.key_list[0]][129, 38], 2.0)


if __name__ == '__main__':
    unittest.main()
