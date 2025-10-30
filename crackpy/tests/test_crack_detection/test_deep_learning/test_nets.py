import unittest
from urllib.error import HTTPError
import time
import torch

from crackpy.crack_detection.model import get_model


class TestParallelNets(unittest.TestCase):

    def test_parallel_nets(self):
        device = torch.device('cpu')

        # catching some rare HTTP 503 errors during model download
        for i in range(5):
            try:
                model = get_model('ParallelNets', map_location=device)
                break
            except HTTPError as e:
                if e.code == 503:
                    time.sleep(2)  # wait 2 seconds, then retry
                else:
                    raise
        else:
            raise RuntimeError("Failed to load model after 5 retries")

        in_tensor = torch.ones((1, 2, 256, 256)).to(device)
        out = model(in_tensor)

        self.assertIsInstance(out, tuple)
        self.assertEqual(list(out[0].size()), [1, 1, 256, 256])
        self.assertEqual(list(out[1].size()), [1, 2])


class TestUNet(unittest.TestCase):

    def test_unet(self):
        device = torch.device('cpu')

        # catching some rare HTTP 503 errors during model download
        for i in range(5):
            try:
                model = get_model('UNetPath', map_location=device)
                break
            except HTTPError as e:
                if e.code == 503:
                    time.sleep(2)  # wait 2 seconds, then retry
                else:
                    raise
        else:
            raise RuntimeError("Failed to load model after 5 retries")

        in_tensor = torch.ones((1, 2, 256, 256)).to(device)
        out = model(in_tensor)

        self.assertEqual(list(out.size()), [1, 1, 256, 256])

        with self.assertRaises(BaseException):
            model(torch.zeros((1, 3, 256, 256)))


if __name__ == '__main__':
    unittest.main()
