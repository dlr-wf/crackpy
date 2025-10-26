from pathlib import Path

import numpy as np
import torch
import logging

from crackpy.crack_detection.utils.plot import plot_prediction
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums
from crackpy.crack_detection.detection import CrackTipDetection, CrackPathDetection, CrackAngleEstimation, CrackDetection
from crackpy.input.input_data import InputData
from crackpy.structure_elements.data_files import Nodemap

logger = logging.getLogger(__name__)


class CrackDetectionSetup:
    """Wrapper for crack detection pipeline settings."""
    def __init__(
            self,
            specimen_size: float,
            sides: list = None,
            stage_nums: range = 'All',
            detection_window_size: float = None,
            detection_boundary: tuple = None,
            start_offset: tuple = (0, 0),
            angle_det_radius: float = 10
    ):
        """Wrapper for crack detection pipeline settings.

        Args:
            specimen_size: size of the whole specimen (currently optimized for MT-specimen with two sides!)
            sides: list of sides, e.g. ['left'] (Default: ['left', 'right'])
            stage_nums: numbers of stages to predict the crack tip (Default: 'All' uses all files of later given path)
            detection_window_size: window size used to predict the crack tip
                                   (if None: detection_window_size is equal to specimen_size / 2 - 10)
            detection_boundary: (x_min, x_max, y_min, y_max) hard boundary of the crack detection window (to avoid NaNs)
                                The 'left' side is mirrored to the 'right' side.
                                Example: (0, 70, -35, 35) for MT160 specimen
                                If None: detection_boundary is set to
                                (0, specimen_size / 2, -specimen_size / 4, specimen_size / 4)
            start_offset: (offset_x, offset_y) offset from the standard starting window
                                               adapted automatically during the pipeline
            angle_det_radius: radius (in mm) around the crack tip

        """

        self.specimen_size = specimen_size

        self.window_size = detection_window_size if detection_window_size is not None else (specimen_size / 2 - 10)
        self.start_offset = start_offset
        if detection_boundary is not None:
            self.detection_boundary = detection_boundary
        else:
            self.detection_boundary = (0, specimen_size / 2, -specimen_size / 4, specimen_size / 4)
        self.angle_det_radius = angle_det_radius

        self.stage_nums = stage_nums

        if sides is None or sides == ['left', 'right'] or sides == ['right', 'left']:
            self.sides = ['left', 'right']
        elif sides in (['left'], ['right']):
            self.sides = sides
        else:
            raise ValueError("Sides must be list of strings 'left' and/or 'right'.")


class CrackDetectionPipeline:
    """Crack detection pipeline.

    Methods:
        * filter_detection_stages - select nodemaps for crack tip detection by ``max_force`` out of ``setup.stage_nums``
        * run_detection - crack detection for the filtered detection stages
        * assign_remaining_stages - sort the non-detection stages to their corresponding detection stage by closest cycles
        * write_results - write results to ``filename`` under ``output_path``

    """
    def __init__(
            self,
            data_path: str,
            output_path: str,
            tip_detector_model,
            setup: CrackDetectionSetup,
            path_detector_model=None
    ):
        """Initialize class arguments.

        Args:
            data_path: path of the nodemap data files
            output_path: where the results are saved
            tip_detector_model: (torch.model) model used for tip detection
            setup: crack detection setup
            path_detector_model: (torch.model) model used for path detection

        """
        self.data_path = data_path
        self.output_path = self._make_path(output_path)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tip_detector = tip_detector_model.to(self.device)
        self.path_detector = path_detector_model
        if path_detector_model is not None:
            self.path_detector = self.path_detector.to(self.device)

        self.setup = setup
        self.stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(Path(self.data_path), self.setup.stage_nums)
        self.detection_stages = sorted(self.stages_to_nodemaps.keys())

        logger.debug(f"Crack detection pipeline initialized: {len(self.stages_to_nodemaps)} nodemaps found")
        logger.debug(f"Device: {self.device}, Sides: {self.setup.sides}, Window size: {self.setup.window_size}")
        logger.debug(f"Detection boundary: {self.setup.detection_boundary}")

        self.sides_to_results = None
        self.stages_to_det_stages = None
        self.det_cycles_to_stages = None
        self.stages_to_cycles = None

    def filter_detection_stages(self, max_force: float, tol: float = 20) -> list:
        """Stages used for crack detection are selected by ``max_force`` with lower tolerance ``tol``.

        Args:
            max_force: value of maximal force [N]
            tol: detection tolerance to maximal force [N]

        Returns:
            detection stage numbers

        """
        logger.info("Filtering detection stages based on max force and tolerance …")

        filtered_stages = []
        det_cycles_to_stages = {}
        stages_to_cycles = {}
        for i, stage in enumerate(self.stages_to_nodemaps):
            logger.info(f"Progress: {i+1}/{len(self.stages_to_nodemaps)}")
            input_nodemap = Nodemap(name=self.stages_to_nodemaps[stage], folder=self.data_path)
            data = InputData()
            data.set_nodemap_file(Path(input_nodemap.folder) / input_nodemap.name)
            data.read_header()

            if data.force is None:
                # no force -> add to filtered_stages
                filtered_stages.append(stage)
                logger.debug(f"Stage {stage}: no force data, added to filtered stages")
            elif data.force > max_force - tol:
                filtered_stages.append(stage)
                logger.debug(f"Stage {stage}: force={data.force:.2f} N (>= {max_force - tol:.2f} N), added to filtered stages")
                # get cycles to stages dictionaries
                if data.cycles is not None:
                    det_cycles_to_stages[data.cycles] = stage
            if data.cycles is not None:
                stages_to_cycles[stage] = data.cycles
        logger.info("Finished filtering detection stages.")
        self.detection_stages = filtered_stages
        self.det_cycles_to_stages = det_cycles_to_stages
        self.stages_to_cycles = stages_to_cycles

        return self.detection_stages

    def run_detection(self) -> dict:
        """Run crack detection on detection_stages and collect results to ``sides_to_results`` dictionary.

        Returns:
            results[side][stage][...]

        """
        # Init
        sides_to_results = {}

        for side in self.setup.sides:
            logger.info(f"Predicting crack tip/path for side: {side} …")

            # Init
            offset_x, offset_y = self.setup.start_offset
            if side == 'left':
                offset_x *= -1   # left side: offset in negative x-direction
            stages_to_results = {}

            for stage in sorted(self.detection_stages):
                # Init
                results = {}

                # Make new crack detection
                ##########################
                # Get input data
                nodemap = self.stages_to_nodemaps[stage]
                input_nodemap = Nodemap(name=nodemap, folder=self.data_path)
                data = InputData(input_nodemap)

                det = CrackDetection(
                    side=side,
                    detection_window_size=self.setup.window_size,
                    offset=(offset_x, offset_y),
                    angle_det_radius=self.setup.angle_det_radius,
                )

                # Interpolate data on arrays (256 x 256 pixels)
                interp_disps, interp_eps_vm = det.interpolate(data)

                # Preprocess input
                input_ch = det.preprocess(interp_disps)

                #####################
                # Tip detection
                #####################
                # Load crack tip detector
                ct_det = CrackTipDetection(detection=det, tip_detector=self.tip_detector)

                # Make prediction
                pred = ct_det.make_prediction(input_ch)
                crack_tip_seg = ct_det.calculate_segmentation(pred)

                # Calculate segmentation and most likely crack tip position
                crack_tip_pixels = ct_det.find_most_likely_tip_pos(pred)

                # Calculate global crack tip positions in mm
                crack_tip_x, crack_tip_y = ct_det.calculate_position_in_mm(crack_tip_pixels)
                logger.debug(f"Stage {stage} ({side}): crack tip detected at pixels {crack_tip_pixels}, "
                           f"position: ({crack_tip_x:.2f}, {crack_tip_y:.2f}) mm")

                #####################
                # Path detection
                #####################
                # load crack path detector model
                cp_det = CrackPathDetection(detection=det, path_detector=self.path_detector)

                # predict segmentation and path skeleton
                cp_segmentation, cp_skeleton = cp_det.predict_path(input_ch)

                ##########################
                # Crack angle estimation
                ##########################
                angle_est = CrackAngleEstimation(detection=det, crack_tip_in_px=crack_tip_pixels)

                if crack_tip_pixels == [np.nan, np.nan]:
                    cp_segmentation_masked = cp_segmentation
                else:
                    # Consider only crack path close to crack tip
                    cp_segmentation_masked = angle_est.apply_circular_mask(cp_segmentation)

                # Filter for largest connected crack path
                cp_segmentation_largest_region = angle_est.get_largest_region(cp_segmentation_masked)

                # Estimate the angle
                angle = angle_est.predict_angle(cp_segmentation_largest_region)

                # Adjust crack detection window
                ###############################
                x_min, x_max, y_min, y_max = self.setup.detection_boundary
                old_offset_x, old_offset_y = offset_x, offset_y

                # Case distinction for left and right side
                if side == 'right':
                    if offset_x <= x_max - self.setup.window_size:  # check detection boundary
                        if crack_tip_x > offset_x + self.setup.window_size / 2:  # check if crack tip passed middle
                            offset_x += (crack_tip_x - offset_x - self.setup.window_size / 2)
                if side == 'left':  # offset is negative
                    if offset_x >= x_min + self.setup.window_size:  # check detection boundary
                        if crack_tip_x < offset_x - self.setup.window_size / 2:
                            offset_x -= (offset_x - self.setup.window_size / 2 - crack_tip_x)
                if y_min + self.setup.window_size / 2 <= offset_y <= y_max - self.setup.window_size / 2:
                    if crack_tip_y > offset_y + self.setup.window_size / 8:
                        offset_y += (crack_tip_y - offset_y - self.setup.window_size / 8)
                    if crack_tip_y < offset_y - self.setup.window_size / 8:
                        offset_y -= (offset_y - self.setup.window_size / 8 - crack_tip_y)

                if offset_x != old_offset_x or offset_y != old_offset_y:
                    logger.debug(f"Stage {stage} ({side}): detection window adjusted from ({old_offset_x:.2f}, {old_offset_y:.2f}) "
                               f"to ({offset_x:.2f}, {offset_y:.2f})")

                # Write results to dictionary
                results['crack_tip_x'] = crack_tip_x
                results['crack_tip_y'] = crack_tip_y
                results['angle'] = angle
                stages_to_results[stage] = results
                logger.info(f"Stage {stage}/{sorted(self.detection_stages)[-1]} - crack tip: {crack_tip_x:.2f} mm, {crack_tip_y:.2f} mm, angle: {angle:.2f}°")

                # Plot crack detection
                plot_prediction(background=interp_eps_vm * 100,
                                interp_size=self.setup.window_size if side == 'right' else -self.setup.window_size,
                                offset=(offset_x, offset_y),
                                save_name=Path(nodemap).stem,
                                crack_tip_prediction=np.asarray([crack_tip_pixels]),
                                crack_tip_seg=crack_tip_seg,
                                crack_tip_label=None,
                                crack_path=cp_skeleton,
                                f_min=0,
                                f_max=0.68,
                                title=Path(nodemap).stem + f' - {side} side',
                                path=str(Path(self.output_path) / "plots" / f"{side}"),
                                label='Von Mises strain [%]')

            sides_to_results[side] = stages_to_results
        self.sides_to_results = sides_to_results

        return sides_to_results

    def assign_remaining_stages(self) -> dict:
        """Assign the remaining (non-detected) stages to the max-load (detected) stages using the cycles information.

        Returns:
            stages_to_det_stages - dictionary stage -> corresponding max load stage

        """
        logger.info("Assigning remaining (non-detected) stages to nearest max-load stages …")

        # assign stages to there detection stage
        stages_to_det_stages = {}
        for i, cycle_i in self.stages_to_cycles.items():
            det_cycles_array = np.asarray(list(self.det_cycles_to_stages.keys()))
            closest_det_cycle = np.argmin(np.abs(det_cycles_array - cycle_i))
            closest_det_stage = self.det_cycles_to_stages[det_cycles_array[closest_det_cycle]]
            stages_to_det_stages[i] = closest_det_stage

        # assign results
        for side in self.setup.sides:
            for i in stages_to_det_stages:
                if i not in self.sides_to_results[side]:
                    self.sides_to_results[side][i] = self.sides_to_results[side][stages_to_det_stages[i]]
        self.stages_to_det_stages = stages_to_det_stages

        return stages_to_det_stages

    def write_results(self, filename: str = 'crack_info_by_nodemap.txt'):
        """Write the crack detection results to a text file.

        Args:
            filename: name out output file (has to end with '.txt')

        """
        logger.info("Writing crack detection results to output file …")

        out_file = Path(self.output_path) / filename
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, mode="w") as file:
            file.write(
                f"{'Filename':>60},{'Crack Tip x [mm]':>17},{'Crack Tip y [mm]':>17},{'Crack Angle':>12},{'Side':>6}\n")
            for side in self.setup.sides:
                for i, stage in enumerate(sorted(self.stages_to_det_stages)):
                    logger.info(f"Progress: {i+1}/{len(self.stages_to_det_stages)}")
                    det_stage = self.stages_to_det_stages[stage]
                    ct_x = self.sides_to_results[side][det_stage]['crack_tip_x']
                    ct_y = self.sides_to_results[side][det_stage]['crack_tip_y']
                    ct_angle = self.sides_to_results[side][det_stage]['angle']
                    nodemap = self.stages_to_nodemaps[stage]

                    if not np.isnan(ct_x):
                        file.write(f"{nodemap:>60},{ct_x:>17.2f},{ct_y:>17.2f},{ct_angle:>12.2f},{side:>6}\n")

    @staticmethod
    def _make_path(path: str) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
