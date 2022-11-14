import unittest

import torch

from crackpy.crack_detection.deep_learning import loss


class TestLosses(unittest.TestCase):

    def setUp(self):
        self.test_tensors = [torch.tensor([[[[0., 0.],
                                             [1., 1.]]]]),
                             torch.tensor([[[[0., 0.],
                                             [0., 1.]]]])]

    def test_dice_loss(self):
        dice_wo_eps = loss.DiceLoss(eps=0)
        dice = loss.DiceLoss(eps=1)

        self.assertAlmostEqual(dice_wo_eps(self.test_tensors[0], self.test_tensors[1]).item(), 1/3)
        self.assertAlmostEqual(dice(self.test_tensors[0], self.test_tensors[1]).item(), 1/4)

    def test_mse_loss(self):
        mse = loss.MSELoss()
        mse_w_factor = loss.MSELoss(weight_factor=10)

        self.assertEqual(mse(self.test_tensors[0], self.test_tensors[1]), 1/4)
        self.assertEqual(mse_w_factor(self.test_tensors[0], self.test_tensors[1]), 2.5)


if __name__ == '__main__':
    unittest.main()
