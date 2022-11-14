import math
import os
import shutil
import tempfile
import unittest

import torch
import numpy as np

from crackpy.crack_detection.data import transforms
from crackpy.crack_detection.data.dataset import CrackTipDataset
from crackpy.crack_detection.data.interpolation import interpolate_on_array
from crackpy.crack_detection.utils.basic import numpy_to_tensor, dict_to_list
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums
from crackpy.crack_detection.data import datapreparation as dp


class TestTransformsWithRealSample(unittest.TestCase):

    def setUp(self):
        self.origin = os.path.join(  # '..', '..', '..', '..',
                                   'test_data', 'crack_detection', 'raw')
        self.side = 'left'
        self.size = 70

        # import test data
        # Do NOT change the nodemap! Otherwise tests will fail!
        stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(os.path.join(self.origin, 'Nodemaps'),
                                                            ['7'])
        inputs, ground_truths = dp.import_data(nodemaps=stages_to_nodemaps,
                                               data_path=self.origin,
                                               side=self.side,
                                               exists_target=True)
        # interpolate
        interp_size = self.size if self.side == 'right' else self.size * -1
        _, interp_disps, _ = interpolate_on_array(input_by_nodemap=inputs, interp_size=interp_size,
                                                  pixels=256)
        # convert
        inputs = numpy_to_tensor(interp_disps, dtype=torch.float32)
        inputs = dict_to_list(inputs)
        targets = numpy_to_tensor(ground_truths, dtype=torch.int64)
        targets = dict_to_list(targets)

        # save test data in temporary folder
        temp_dir = tempfile.mkdtemp()
        try:
            torch.save(inputs, os.path.join(temp_dir, 'input.pt'))
            torch.save(targets, os.path.join(temp_dir, 'target.pt'))

            # load data from tmp into dataset
            self.dataset = CrackTipDataset(inputs=os.path.join(temp_dir, 'input.pt'),
                                           labels=os.path.join(temp_dir, 'target.pt'))
            self.sample = self.dataset[0]
        finally:  # clean up tmp
            shutil.rmtree(temp_dir)

    def test_dataset(self):
        self.assertEqual(len(self.dataset), 1)
        self.assertIsInstance(self.sample, dict)
        self.assertListEqual(list(self.sample.keys()), ['input', 'target', 'tip'])
        self.assertEqual(list(self.sample['input'].shape), [2, 256, 256])
        self.assertEqual(list(self.sample['target'].shape), [256, 256])
        self.assertEqual(list(self.sample['tip']), [129., 38.])

    def test_add_zero_component(self):
        trsfm = transforms.AddZeroComponent()
        trsfm(self.sample)

        self.assertListEqual(list(self.sample.keys()), ['input', 'target', 'tip'])
        self.assertEqual(list(self.sample['input'].shape), [3, 256, 256])
        self.assertTrue(torch.equal(self.sample['input'][-1], torch.zeros((256, 256))))

    def test_crack_tip_normalization(self):
        trsfm = transforms.CrackTipNormalization()
        trsfm(self.sample)

        self.assertEqual(list(self.sample['tip']), [0.0078125, -0.703125])

    def test_input_normalization(self):
        sample_before_transform = self.sample.copy()
        trsfm = transforms.InputNormalization()
        trsfm(self.sample)

        # check if target and tip is unchanged
        self.assertTrue(torch.equal(sample_before_transform['target'], self.sample['target']))
        self.assertTrue(torch.equal(sample_before_transform['tip'], self.sample['tip']))

        # check if input channels have mean 0 and std 1
        img = np.asarray(self.sample['input'])
        self.assertTrue(np.allclose(img.mean(axis=(1, 2)), np.asarray([0., 0.]), atol=1e-7))
        self.assertTrue(np.allclose(img.std(axis=(1, 2)), np.asarray([1., 1.]), atol=1e-7))

    def test_enhance_tip(self):
        sample_before_transform = self.sample.copy()
        trsfm = transforms.EnhanceTip()
        trsfm(self.sample)

        # check if target and tip is unchanged
        self.assertTrue(torch.equal(sample_before_transform['input'], self.sample['input']))
        self.assertTrue(torch.equal(sample_before_transform['tip'], self.sample['tip']))

        # check that the correct pixels are labelled
        condition = torch.BoolTensor(self.sample['target'] == 2.)
        is_crack_tip = torch.where(condition, 1, 0)
        act_tip_pos = torch.nonzero(is_crack_tip, as_tuple=False)[:, -2:] / 1.
        exp_tip_pos = torch.tensor([[128., 37.], [128., 38.], [128., 39.],
                                    [129., 37.], [129., 38.], [129., 39.],
                                    [130., 37.], [130., 38.], [130., 39.]])
        self.assertTrue(torch.equal(act_tip_pos, exp_tip_pos))

    def test_random_crop_with_fixed_size(self):
        trsfm = transforms.RandomCrop(size=100, left=None)
        trsfm(self.sample)
        self.assertEqual(list(self.sample['input'].shape[-2:]), [100, 100])
        self.assertEqual(list(self.sample['target'].shape[-2:]), [100, 100])

    def test_random_crop_with_fixed_tuple_size(self):
        trsfm = transforms.RandomCrop(size=(128, 128), left=None)
        trsfm(self.sample)
        self.assertEqual(list(self.sample['input'].shape[-2:]), [128, 128])
        self.assertEqual(list(self.sample['target'].shape[-2:]), [128, 128])

    def test_random_crop_with_size_interval(self):
        trsfm = transforms.RandomCrop(size=[100, 150])
        trsfm(self.sample)
        act_size_input = list(self.sample['input'].shape[-2:])
        act_size_target = list(self.sample['target'].shape[-2:])
        self.assertEqual(act_size_input[0], act_size_input[1])
        self.assertEqual(act_size_target[0], act_size_target[1])
        self.assertEqual(act_size_input, act_size_target)
        self.assertTrue(100 <= act_size_input[0] <= 150)

    def test_random_crop_assertions(self):
        with self.assertRaises(AssertionError):
            transforms.RandomCrop(size='100')
            transforms.RandomCrop(size=100, left=10)
            transforms.RandomCrop(size=(100, 100, 100))

    def test_random_flip(self):
        always_flip = transforms.RandomFlip(flip_probability=1)
        never_flip = transforms.RandomFlip(flip_probability=0)
        sample_before_transform = self.sample

        self.assertTrue(torch.equal(sample_before_transform['input'],
                                    never_flip(self.sample)['input']))
        self.assertTrue(torch.equal(sample_before_transform['target'],
                                    never_flip(self.sample)['target']))
        self.assertTrue(torch.equal(torch.flip(self.sample['input'], dims=[1]),
                                    always_flip(self.sample)['input']))
        self.assertTrue(torch.equal(torch.flip(self.sample['target'], dims=[0]),
                                    always_flip(self.sample)['target']))

    def test_resize(self):
        with self.assertRaises(AssertionError):
            transforms.Resize(-1)
            transforms.Resize((10, -1))
            transforms.Resize((-2, 5))

        trsfm = transforms.Resize(128)
        sample_resized = trsfm(self.sample)
        self.assertEqual(list(sample_resized['input'].size()), [2, 128, 128])
        self.assertEqual(list(sample_resized['target'].size()), [128, 128])
        self.assertEqual(list(sample_resized['tip']), [64.5, 19.0])

    def test_to_crack_tip_masks(self):
        trsfm = transforms.ToCrackTipMasks()
        image, target = trsfm(self.sample)

        self.assertTrue(torch.equal(image, self.sample['input']))
        self.assertEqual(list(target.size()), [1, 256, 256])

        act_tip = torch.nonzero(target, as_tuple=False)[:, 1:3] / 1.
        self.assertTrue(torch.equal(act_tip.squeeze(), self.sample['tip']))

    def test_to_crack_tips(self):
        trsfm = transforms.ToCrackTips()
        image, tips = trsfm(self.sample)

        self.assertTrue(torch.equal(image, self.sample['input']))
        self.assertTrue(torch.equal(tips, self.sample['tip']))

    def test_to_crack_tips_and_masks(self):
        trsfm = transforms.ToCrackTipsAndMasks()
        image, (mask, tip) = trsfm(self.sample)

        self.assertTrue(torch.equal(image, self.sample['input']))
        exp_mask = mask = torch.where(self.sample['target'] == 2, 1, 0).unsqueeze(0)
        self.assertTrue(torch.equal(mask, exp_mask))
        self.assertTrue(torch.equal(tip, self.sample['tip']))


class TestTransformsBasic(unittest.TestCase):

    def test_denormalize_crack_tips(self):
        test_tensor = torch.tensor([0.0078125, -0.703125], dtype=torch.float32)
        test_size = [256, 256]

        test_tensor_denormalized = transforms.denormalize_crack_tips(test_tensor, test_size)
        self.assertTrue(torch.equal(test_tensor_denormalized, torch.tensor([129., 38.])))

    def test_rotate_point(self):
        origin = (0, 0)
        point = (1, 0)

        self.assertAlmostEqual(transforms.rotate_point(origin, point, math.pi / 2)[0], 0)
        self.assertAlmostEqual(transforms.rotate_point(origin, point, math.pi / 2)[1], 1)
        self.assertAlmostEqual(transforms.rotate_point(origin, point, math.pi)[0], -1)
        self.assertAlmostEqual(transforms.rotate_point(origin, point, math.pi)[1], 0)
        self.assertAlmostEqual(transforms.rotate_point(origin, point, 2*math.pi)[0], 1)
        self.assertAlmostEqual(transforms.rotate_point(origin, point, 2 * math.pi)[1], 0)

    def test_calculate_crop_size(self):
        in_size = 100

        self.assertEqual(transforms.calculate_crop_size(0, in_size), in_size)
        self.assertEqual(transforms.calculate_crop_size(math.pi / 2, in_size), in_size)
        self.assertEqual(transforms.calculate_crop_size(math.pi / 4, in_size), 70)

    def test_random_rotation(self):
        with self.assertRaises(AssertionError):
            transforms.RandomRotation(-1)
            transforms.RandomRotation(50)
            transforms.RandomRotation((-1, 1, 2))
            transforms.RandomRotation((1, -1))


if __name__ == '__main__':
    unittest.main()
