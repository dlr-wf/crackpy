import torch
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression

from crackpy.crack_detection.data import preprocess
from crackpy.crack_detection.deep_learning.nets import ParallelNets, UNet
from crackpy.crack_detection.data.interpolation import interpolate
from crackpy.crack_detection.utils.utilityfunctions import calculate_segmentation, find_most_likely_tip_pos
from crackpy.fracture_analysis.data_processing import InputData


class CrackDetection:
    """Crack detection setup class.

    Methods:
        * preprocess - prepare interpolated displacements for input to NN
        * interpolate - interpolate nodemap data on arrays (256 x 256 pixels)

    """

    def __init__(self, side: str = 'right', detection_window_size: float = 70, offset: tuple = (0, 0),
                 angle_det_radius: float = 10, device=None):
        """Initialize class arguments.

        Args:
            side: of the specimen ('left' or 'right')
            window: size of the detection window in mm
            offset: tuple of (x,y)-offset in mm
            angle_det_radius: radius in mm around crack tip considered for angle detection
            device: (torch.device)

        """
        self.side = side
        self.detection_window_size = detection_window_size
        self.offset = offset
        self.interp_size = self._get_interp_size()
        self.angle_det_radius = angle_det_radius
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def preprocess(interp_disps: np.ndarray) -> torch.Tensor:
        """Prepare interpolated displacements for input to NN.

        Args:
            interp_disps: interpolated displacements

        Returns:
            input data

        """
        input_ch = torch.tensor(interp_disps, dtype=torch.float32)
        input_ch = preprocess.normalize(input_ch).unsqueeze(0)

        return input_ch

    def interpolate(self, data: InputData):
        """Interpolate nodemap data on arrays (256 x 256 pixels)

        Args:
            data: nodemap data

        Returns:
            interpolated displacements, interpolated von Mises strains as arrays of size 256x256

        """
        _, interp_disps, interp_eps_vm = interpolate(data, self.interp_size, offset=self.offset, pixels=256)
        return interp_disps, interp_eps_vm

    def _get_interp_size(self):
        return self.detection_window_size if self.side == 'right' else -self.detection_window_size


class CrackTipDetection:
    """Crack tip detection class.

    Methods:
        * calculate_position_in_mm - converts crack tip position from pixels to mm
        * make_prediction - predict crack tips as segmented pixels
        * calculate_segmentation - calculates the crack tip positions of all segmented pixels of an output mask
        * find_most_likely_tip_pos - detects crack tip region and selects the one with the highest mean probability

    """

    def __init__(self, detection: CrackDetection, tip_detector: ParallelNets):
        """Initialize class arguments.

        Args:
            detection: crack detection setup
            tip_detector: crack tip detection model
        """
        self.detection = detection
        self.tip_detector = tip_detector

    def calculate_position_in_mm(self, crack_tip_px: list):
        """Converts the crack tip position from pixels to mm.

        Args:
            crack_tip_px: x- and y-coordinates of crack tip position [px]

        """
        # Transform to global coordinate system
        crack_tip_x = crack_tip_px[1] * self.detection.detection_window_size / 255
        crack_tip_y = crack_tip_px[0] * self.detection.detection_window_size / 255 - self.detection.detection_window_size / 2
        if self.detection.side == 'left':  # mirror x-value of crack tip position to left-hand side
            crack_tip_x *= -1
        crack_tip_x += self.detection.offset[0]
        crack_tip_y += self.detection.offset[1]

        return crack_tip_x, crack_tip_y

    def make_prediction(self, input_ch: torch.Tensor) -> torch.Tensor:
        """Predict crack tips as segmented pixels.

        Args:
            input_ch: input data

        Returns:
            crack tip segmentation output

        """
        device = self.detection.device
        self.tip_detector.to(device=device)
        self.tip_detector.eval()
        out = self.tip_detector(input_ch.to(device))
        pred = out[0].detach().to('cpu')

        return pred

    @staticmethod
    def calculate_segmentation(pred: torch.Tensor) -> torch.Tensor:
        """Calculates the crack tip positions of all segmented pixels of an output mask.

        Args:
            pred: tensor of shape (.. x H x W)

        Returns:
            crack tip segmentation tensor of shape num_of_seg x 2

        """
        return calculate_segmentation(pred)

    @staticmethod
    def find_most_likely_tip_pos(pred: torch.Tensor) -> list:
        """Detects crack tip region and selects the one with the highest mean probability.
        Returns the mean crack tip position on this region."""
        return find_most_likely_tip_pos(pred)


class CrackPathDetection:
    """Crack path detection class.

    Methods:
        * make_prediction - predict crack path as segmented pixels
        * predict_path - predict the crack path and return the segmentation and skeletonized path

    """
    def __init__(self, detection: CrackDetection, path_detector: UNet):
        """Initialize class arguments.

        Args:
            detection: crack detection setup
            path_detector: crack path detection model

        """
        self.detection = detection
        self.path_detector = path_detector

    def make_prediction(self, input_ch: torch.Tensor) -> torch.Tensor:
        """Predict crack path as segmented pixels.

        Args:
            input_ch: input data

        Returns:
            crack path segmentation output

        """
        device = self.detection.device
        self.path_detector.to(device)
        self.path_detector.eval()
        out = self.path_detector(input_ch.to(device))
        pred = out.detach().to('cpu')
        return pred

    def predict_path(self, input_ch: torch.Tensor) -> tuple:
        """Predict the crack path and return the segmentation and skeletonized path

        Args:
            input_ch: input data

        Returns:
            tuple of tensors of crack path segmentation ('1' = crack path)
            and skeletonized crack path as pixel positions

        """
        # make prediction
        pred = self.make_prediction(input_ch)

        # filter crack path prediction pixels
        condition = torch.BoolTensor(pred >= 0.5)
        is_crack_path = torch.where(condition, 1, 0).squeeze()

        # crack path skeleton
        skeleton = skeletonize(is_crack_path.numpy().astype('uint8'), method='lee')
        skeleton = torch.nonzero(torch.from_numpy(skeleton), as_tuple=False)

        return is_crack_path, skeleton


class CrackAngleEstimation:
    """Crack angle estimation class using linear regression on segmented crack path pixels around the crack tip.

    Methods:
        * apply_circular_mask - apply circular mask on the path segmentation around the crack tip position
        * get_largest_region - filter largest connected region in path segmentation
        * predict_angle - estimate the crack tip angle by means of a linear regression

    """
    def __init__(self, detection: CrackDetection, crack_tip_in_px: list):
        """Initialize class arguments.

        Args:
            detection: crack detection setup
            crack_tip_in_px: detected crack tip position as pixels

        """
        self.detection = detection
        self.tip = crack_tip_in_px

    def apply_circular_mask(self, path_segmentation: torch.Tensor) -> np.ndarray:
        """Apply circular mask on the path segmentation around the crack tip position.

        Args:
            path_segmentation: segmented crack path to apply mask on

        Returns:
            array of filtered crack path pixels ('1') and background ('0')

        """
        angle_det_radius_px = self.detection.angle_det_radius / self.detection.detection_window_size * 255
        Y, X = np.ogrid[:256, :256]
        dist_from_tip = np.sqrt((X - self.tip[1].item()) ** 2 +
                                (Y - self.tip[0].item()) ** 2)
        mask = dist_from_tip <= angle_det_radius_px
        masked_crack_path_seg = path_segmentation.numpy() * mask
        return masked_crack_path_seg

    @staticmethod
    def get_largest_region(path_segmentation: np.ndarray) -> np.ndarray:
        """Filter largest connected region in path segmentation.

        Args:
            path_segmentation: segmented crack path to apply filter on

        Returns:
            array of filtered crack path pixels ('1') and background ('0')

        """
        labels, num_of_labels = label(path_segmentation)
        largest_region_label = 0
        largest_region_pixels = 0
        for k in range(1, num_of_labels + 1):
            num_of_pixels = np.sum(labels == k)
            if num_of_pixels > largest_region_pixels:
                largest_region_label = k
                largest_region_pixels = num_of_pixels
        largest_region_path_segmentation = (labels == largest_region_label)
        return largest_region_path_segmentation

    def predict_angle(self, path_segmentation: np.ndarray) -> float or None:
        """Estimate the crack tip angle by means of a linear regression.

        Args:
            path_segmentation: crack path pixels ('1') and background ('0')

        Returns:
            crack path angle close to the crack tip

        """
        if np.sum(path_segmentation) > 0 and self.tip[0] is not np.nan:
            # linear regression for crack angle determination
            y, x = np.nonzero(path_segmentation)
            lin_model = LinearRegression()
            lin_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
            slope = lin_model.coef_[0, 0]
            angle = np.arctan(slope)  # slope to angle
            angle *= 180 / np.pi  # rad to deg
            if self.detection.side == 'left':
                angle = 180 - angle
            return angle
        return np.nan
