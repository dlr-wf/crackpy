import unittest

import torch

from crackpy.crack_detection.data import preprocess


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.test_tensor = torch.tensor([[-2., 2.], [2., -2.]])
        self.test_tensor_normalized = torch.tensor([[-1., 1.], [1., -1.]])
        self.exp_crack_tip_pos = torch.tensor([[0., 1.], [1., 0.]])

    def test_normalize(self):
        act = preprocess.normalize(self.test_tensor)
        self.assertTrue(torch.equal(act, self.test_tensor_normalized))

    def test_target_to_crack_tip_position(self):
        act = preprocess.target_to_crack_tip_position(self.test_tensor)
        self.assertTrue(torch.equal(act, self.exp_crack_tip_pos))


if __name__ == '__main__':
    unittest.main()
