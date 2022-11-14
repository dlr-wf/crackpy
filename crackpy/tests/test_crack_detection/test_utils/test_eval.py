import unittest

import math
import torch

from crackpy.crack_detection.utils import evaluate


class TestEvaluation(unittest.TestCase):

    def test_get_deviation(self):
        output = torch.tensor([[-0.5, -0.5], [0., 0.], [0.5, 0.5]])
        labels = torch.tensor([[0., -0.5], [0.5, 1.], [0., 0.]])

        exp_deviation = 25. + 5. * math.sqrt(125.) + math.sqrt(2) * 25.
        act_deviation = evaluate.get_deviation(output, labels, [100, 100])
        self.assertAlmostEqual(exp_deviation, act_deviation, delta=1e-5)


class TestEvalSegmentation(unittest.TestCase):

    def setUp(self):
        self.output = torch.tensor([
            [[0.0, 0.5],
             [0.7, 0.9]],

            [[0.4, 0.1],
             [0.3, 0.0]]
        ])
        self.label = torch.tensor([
            [[0., 0.],
             [0., 1.]],

            [[1., 0.],
             [0., 0.]]
        ])

    def test_get_segmentation_deviation(self):
        exp_deviation = 0.4714045
        act_deviation = evaluate.get_segmentation_deviation(self.output, self.label)
        self.assertAlmostEqual(act_deviation, exp_deviation, delta=1e-5)

    def test_get_reliability(self):
        exp_rel = 0.5
        act_rel = evaluate.get_reliability(self.output, self.label)
        self.assertEqual(act_rel, exp_rel)


if __name__ == '__main__':
    unittest.main()
