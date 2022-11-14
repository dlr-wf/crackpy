import os
import unittest

from crackpy.crack_detection.deep_learning.setup import Setup


class TestSetup(unittest.TestCase):

    def setUp(self):
        self.setup = Setup(data_path=os.path.join(  # '..', '..', '..', '..',
                                                  'test_data', 'crack_detection', 'raw'),
                           experiment='EBr10', side='left')
        self.setup.set_stages(['7', '407', '807'])

        self.assertEqual(list(self.setup.stages_to_nodemaps.keys()), [7, 407, 807])

    def test_load_data(self):
        inputs, targets = self.setup.load_data()

        self.assertIsInstance(inputs, dict)
        self.assertIsInstance(targets, dict)
        self.assertEqual(len(inputs), len(targets))
        self.assertEqual(list(next(iter(inputs.values())).shape), [1, 2, 256, 256])
        self.assertEqual(list(next(iter(targets.values())).shape), [1, 256, 256])

    def test_set_model(self):
        model_path = os.path.join(  # '..', '..', '..', '..',
                                  'crackpy', 'crack_detection', 'models')
        model_name = 'ParallelNets.pt'

        self.setup.set_model(model_path=model_path, model_name=model_name)
        self.assertEqual(self.setup.model_path, model_path)
        self.assertEqual(self.setup.model_name, model_name)

        with self.assertRaises(ValueError):
            self.setup.set_model(model_path='this_path_does_not_exist', model_name='')

    def test_set_output_path(self):
        pass


if __name__ == '__main__':
    unittest.main()
