import random
import math

import torch
import numpy as np
import torchvision.transforms.functional as F


class AddZeroComponent:
    """Add the z-component equal to zero for 3 input channels."""
    def __call__(self, sample):
        img = sample['input']
        zeros_shape = 1, img.size()[1], img.size()[2]
        zeros = torch.zeros(zeros_shape)
        sample['input'] = torch.cat((img, zeros), dim=0)

        return sample


class CrackTipNormalization:
    """Normalize the crack tips."""
    def __call__(self, sample):
        target, tip = sample['target'], sample['tip']

        # Normalize crack tip label
        center = torch.tensor(target.shape, dtype=torch.float32) / 2.
        tip_normalized = (tip - center) / center
        sample['tip'] = tip_normalized

        return sample


def denormalize_crack_tips(tips: torch.Tensor, size: list) -> torch.Tensor:
    """Revert normalization of crack tips.

    Args:
        tips: tensor of size 2
        size: list of length 2

    Returns:
        tips_normalized tensor of size 2

    """
    assert isinstance(tips, torch.Tensor)
    assert isinstance(size, (list, torch.Size))
    center = torch.tensor(size, dtype=torch.float32) / 2.
    tips_denormalized = tips * center + center
    return tips_denormalized


class InputNormalization:
    """Normalize the input image."""
    def __init__(self, means=None, stds=None):
        self.means = np.asarray(means).reshape((-1, 1, 1)) if means is not None else None
        self.stds = np.asarray(stds).reshape((-1, 1, 1)) if stds is not None else None

    def __call__(self, sample):
        # Normalize input
        img = np.asarray(sample['input'])
        means = img.mean(axis=(1, 2), keepdims=True) if self.means is None else self.means
        stds = img.std(axis=(1, 2), keepdims=True) if self.stds is None else self.stds
        img = (img - means) / stds
        sample['input'] = torch.tensor(img, dtype=torch.float32)

        return sample


class EnhanceTip:
    """
    Enhance the crack tip position with width 1.
    This is necessary because otherwise the crack tip position might vanish with the
    transformations 'Resize' or 'RandomRotation' which involve interpolation.
    """

    def __init__(self, width=1):
        self.width = width

    def __call__(self, sample):
        target, tip = sample['target'], sample['tip']

        # Write '2's of width 'self.width' indicating crack tip around the target tip position
        # This produces 1+2*width x 1+2*width crack tip pixels instead of just 1
        for i in range(-self.width, self.width + 1):
            for j in range(-self.width, self.width + 1):
                target[int(tip[0].item() + i), int(tip[1].item() + j)] = 2

        sample['target'] = target

        return sample


class EnhancePath:
    """Make Ground Truth path thicker."""
    def __call__(self, sample):
        target = torch.where(sample['target'] == 1, 1, 0)
        size = len(target)
        path_pixels = torch.nonzero(target, as_tuple=False)

        for pos in path_pixels:
            if 0 < pos[0] < size and 0 < pos[1] < size:
                target[int(pos[0].item() - 1), int(pos[1].item() - 1):int(pos[1].item() + 2)] = 1
                target[int(pos[0].item()), int(pos[1].item() - 1):int(pos[1].item() + 2)] = 1
                target[int(pos[0].item() + 1), int(pos[1].item() - 1):int(pos[1].item() + 2)] = 1

        sample['target'] = target

        return sample


class RandomCrop:
    """Crop randomly the image & labels in a sample."""
    def __init__(self, size: int or tuple or list, left: list = None):
        """Crop randomly the image & labels in a sample.

        Args:
            size:
                    if int: crop size is ('size', 'size')
                    if tuple: crop size is 'size'
                    if list: crop size is randomly chosen in interval 'size'
            left:
                    if not None: left-side of the crop is randomly chosen in interval 'left'

        """
        if left is not None:
            assert isinstance(left, list) and len(left) == 2
        self.left = left
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            self.size = (size, size)
            self.random_size = False
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.size = size
            self.random_size = False
        else:
            assert len(size) == 2
            self.size = size
            self.random_size = True

    def __call__(self, sample):
        image, crack, tip = sample['input'], sample['target'], sample['tip']

        height, width = image.shape[1:3]
        if self.random_size:
            new_height = np.random.randint(self.size[0], self.size[1])
            new_width = new_height
        else:
            new_height, new_width = self.size

        top = np.random.randint(0, height - new_height)
        if self.left is not None:
            left = np.random.randint(self.left[0], min(self.left[1], width - new_width))
        else:
            left = np.random.randint(0, width - new_width)

        sample['input'] = image[:, top: top + new_height, left: left + new_width]
        sample['target'] = crack[top: top + new_height, left: left + new_width]
        sample['tip'] = tip - torch.tensor([top, left])

        return sample


class RandomFlip:
    """Flip randomly up/down of an image & label in a sample."""
    def __init__(self, flip_probability: float=0.5):
        """Flip randomly up/down of an image & label in a sample.

        Args:
            flip_probability: of random flip

        """
        self.flip_probability = flip_probability

    def __call__(self, sample):
        if random.random() <= self.flip_probability:
            sample['input'] = torch.flip(sample['input'], dims=[1])
            sample['target'] = torch.flip(sample['target'], dims=[0])
            sample['tip'][0] = sample['target'].shape[-1] - sample['tip'][0]

        return sample


def rotate_point(origin: tuple, point: tuple, angle_rad: float) -> list:
    """Rotate a point counterclockwise by an angle around an origin.

    Args:
        origin: (list of floats) of rotation axis
        point: (list of floats) point to rotate around the origin
        angle_rad: rotation angle [rad]

    Returns:
        x and y coordinates of rotated point

    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)

    return [qx, qy]


def calculate_crop_size(angle_rad: float, in_size: int) -> int:
    """Calculates the crop size if rotation is applied.

    Args:
        angle_rad: angle of rotation in radians
        in_size: size of original uncropped image

    Returns:
        crop size

    """
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if abs(sin_a - cos_a) < 1e-6:
        crop = in_size / (sin_a + cos_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        crop = (in_size * cos_a - in_size * sin_a) / cos_2a

    return int(crop)


class RandomRotation:
    """Rotate the input, target and crack tip randomly with the same angle."""
    def __init__(self, degrees: int or tuple):
        """Rotate the input, target and crack tip randomly with the same angle.

        Args:
            degrees: int or tuple of rotation degrees (not radian)

        """
        assert isinstance(degrees, (int, tuple))
        if isinstance(degrees, int):
            assert 0 <= degrees <= 90
            self.degree_min = -degrees
            self.degree_max = degrees
        else:
            assert len(degrees) == 2
            self.degree_min, self.degree_max = degrees
            assert self.degree_min >= -45 and self.degree_max <= 45

    def __call__(self, sample):
        image, crack, tip = sample['input'], sample['target'], sample['tip']
        in_size = crack.shape[0]

        # Rotate and crop to avoid padding at the corners
        angle = random.uniform(self.degree_min, self.degree_max)

        image = F.rotate(image, angle)
        crack = F.rotate(crack.unsqueeze(0), angle, interpolation=F.InterpolationMode.NEAREST).squeeze()
        tip_rot = rotate_point(origin=(in_size / 2., in_size / 2.),
                               point=(tip[0].item(), tip[1].item()),
                               angle_rad=np.deg2rad(angle))

        crop_size = calculate_crop_size(np.deg2rad(angle), in_size)
        image = F.center_crop(image, [crop_size, crop_size])
        crack = F.center_crop(crack, [crop_size, crop_size])
        tip = torch.tensor(tip_rot) - \
              torch.tensor([in_size - crop_size, in_size - crop_size]) / 2.

        return {'input': image, 'target': crack, 'tip': tip}


class Resize:
    """Resize input and target tensors."""
    def __init__(self, size: int or tuple):
        """Resize input and target tensors.

        Args:
            size: int or tuple of resize size

        """
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            assert size > 0
            self.out_sizes = [size, size]
        else:
            assert len(size) == 2 and isinstance(size[0], int) and isinstance(size[1], int)
            assert size[0] >= 0, size[1] >= 0
            self.out_sizes = list(size)

    def __call__(self, sample):
        image, crack, tip = sample['input'], sample['target'], sample['tip']
        in_sizes = torch.tensor(image.size()[1:])

        image_resized = F.resize(image, size=self.out_sizes)
        crack_resized = F.resize(img=crack.unsqueeze(0),
                                 size=self.out_sizes,
                                 interpolation=F.InterpolationMode.NEAREST).squeeze()
        tip_resized = tip * torch.tensor(self.out_sizes) / in_sizes

        return {'input': image_resized, 'target': crack_resized, 'tip': tip_resized}


class ToCrackTipMasks:
    """Extract input and crack tip mask."""
    def __call__(self, sample):
        image, target = sample['input'], sample['target']
        tip = torch.where(target == 2, 1, 0)
        tip = tip.unsqueeze(0)
        return image, tip


class ToCrackTips:
    """Extract input and crack tip coordinates."""
    def __call__(self, sample):
        return sample['input'], sample['tip']


class ToCrackPathMasks:
    """Extract input and crack path."""
    def __call__(self, sample):
        image, target = sample['input'], sample['target']
        path = torch.where(target == 1, 1, 0)
        path = path.unsqueeze(0)
        return image, path


class ToCrackTipsAndMasks:
    """Extract input and tuple of crack tip mask and coordinates."""
    def __call__(self, sample):
        image, mask, tip = sample['input'], sample['target'], sample['tip']
        mask = torch.where(mask == 2, 1, 0)
        mask = mask.unsqueeze(0)
        return image, (mask, tip)
