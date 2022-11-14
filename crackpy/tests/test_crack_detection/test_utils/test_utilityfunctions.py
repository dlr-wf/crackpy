import os
import unittest

import torch

from crackpy.crack_detection.utils.utilityfunctions import \
    get_nodemaps_and_stage_nums, calculate_segmentation


class TestGetNodemapsAndStages(unittest.TestCase):

    def setUp(self):
        self.origin = os.path.join(  # '..', '..', '..', '..',
                                   'test_data', 'crack_detection', 'raw')
        self.nodemaps = ['7', '407', '807']

    def test_get_nodemaps_and_stages(self):
        stages_to_nodemaps, nodemaps_to_stages = \
            get_nodemaps_and_stage_nums(os.path.join(self.origin, 'Nodemaps'), self.nodemaps)
        stages_to_nodemaps_all, nodemaps_to_stages_all = \
            get_nodemaps_and_stage_nums(os.path.join(self.origin, 'Nodemaps'), 'All')
        self.assertIsInstance(stages_to_nodemaps, dict)
        self.assertIsInstance(nodemaps_to_stages, dict)
        self.assertIsInstance(stages_to_nodemaps_all, dict)
        self.assertIsInstance(nodemaps_to_stages_all, dict)
        self.assertDictEqual(stages_to_nodemaps, stages_to_nodemaps_all)
        self.assertDictEqual(nodemaps_to_stages, nodemaps_to_stages_all)


class TestSegmentationCalculation(unittest.TestCase):

    def test_calculate_segmentation(self):
        test_output = torch.tensor([[0.1, 0.5], [0.8, 0.499]])
        exp_seg = torch.tensor([[0., 1.], [1., 0.]])

        act_seg = calculate_segmentation(test_output)
        self.assertTrue(torch.equal(exp_seg, act_seg))


if __name__ == '__main__':
    unittest.main()
