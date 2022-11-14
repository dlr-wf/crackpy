import unittest
import numpy as np
import torch
from numpy.testing import assert_array_equal

from crackpy.crack_detection.utils import basic


class TestBasic(unittest.TestCase):

    def test_concatenate_np_dicts(self):
        np_dict_1 = {'key1': np.asarray([[1, 2]]),
                     'key2': np.asarray([[3, 4]])}
        np_dict_2 = {'key1': np.asarray([5, 6]),
                     'key2': np.asarray([7, 8])}

        np_dict_exp = {'key1': np.asarray([[1, 2], [5, 6]]),
                       'key2': np.asarray([[3, 4], [7, 8]])}
        np_dict_act = basic.concatenate_np_dicts(np_dict_1, np_dict_2)

        for key, act in np_dict_act.items():
            exp = np_dict_exp[key]
            self.assertIsNone(assert_array_equal(act, exp))

    def test_numpy_to_tensor(self):
        np_dict = {'key1': np.asarray([1, 2]),
                   'key2': np.asarray([3, 4])}

        dict_exp = {'key1': torch.tensor([[1, 2]], dtype=torch.float32),
                    'key2': torch.tensor([[3, 4]], dtype=torch.float32)}
        dict_act = basic.numpy_to_tensor(np_dict, dtype=torch.float32)
        for key, act in dict_act.items():
            exp = dict_exp[key]
            self.assertTrue(torch.equal(exp, act))

    def test_dict_to_list(self):
        dictionary = {'key1': 3, 'key2': 'Value'}
        list_of_dict = [3, 'Value']
        self.assertListEqual(basic.dict_to_list(dictionary), list_of_dict)


if __name__ == '__main__':
    unittest.main()
