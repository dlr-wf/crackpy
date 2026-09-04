import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.pipeline.pipeline import (
    CrackDetectionPipeline,
    CrackDetectionSetup,
)


class TestCrackDetPipeline(unittest.TestCase):

    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        self.data_path = root / 'test_data' / 'crack_detection' / 'Nodemaps'
        self.crack_info_by_nodemap_file = root / 'test_data' / 'crack_detection' / 'crack_info_by_nodemap.txt'

        self.det_setup = CrackDetectionSetup(
            specimen_size=160,
            sides=['left', 'right'],
            detection_window_size=None,
            start_offset=(0, 0),
            angle_det_radius=13.725
        )

        # crack detectors
        self.tip_detector = get_model('ParallelNets')
        self.path_detector = get_model('UNetPath')

    def test_cd_pipeline(self):
        temp_dir = tempfile.mkdtemp()
        try:
            pipeline = CrackDetectionPipeline(
                data_path=str(self.data_path),
                output_path=temp_dir,
                tip_detector_model=self.tip_detector,
                path_detector_model=self.path_detector,
                setup=self.det_setup
            )

            ################################
            # necessary to run on GitLab
            pipeline.device = 'cpu'
            pipeline.tip_detector.to('cpu')
            pipeline.path_detector.to('cpu')
            ################################

            pipeline.filter_detection_stages(max_force=15000, tol=20)
            pipeline.run_detection()
            pipeline.assign_remaining_stages()
            pipeline.write_results()

            # check crack detection results
            exp_results = pd.read_csv(str(self.crack_info_by_nodemap_file))
            act_results = pd.read_csv(str(Path(temp_dir) / 'crack_info_by_nodemap.txt'))
            pd.testing.assert_frame_equal(exp_results, act_results)

        finally:
            shutil.rmtree(temp_dir)

    def test_tip_only_pipeline_skips_path_and_angle(self):
        class PathDetectorMustNotBeUsed:
            def to(self, device):
                raise AssertionError("tip-only pipeline touched the path detector")

        tip_only_setup = CrackDetectionSetup(
            specimen_size=160,
            sides=['right'],
            stage_nums=[52],
            detection_window_size=None,
            start_offset=(0, 0),
            angle_det_radius=13.725,
            tip_only=True,
        )

        temp_dir = tempfile.mkdtemp()
        try:
            pipeline = CrackDetectionPipeline(
                data_path=str(self.data_path),
                output_path=temp_dir,
                tip_detector_model=self.tip_detector,
                path_detector_model=PathDetectorMustNotBeUsed(),
                setup=tip_only_setup,
            )

            pipeline.device = 'cpu'
            pipeline.tip_detector.to('cpu')

            results = pipeline.run_detection()

            self.assertAlmostEqual(results['right'][52]['crack_tip_x'], 14.90, places=2)
            self.assertAlmostEqual(results['right'][52]['crack_tip_y'], 0.66, places=2)
            self.assertTrue(pd.isna(results['right'][52]['angle']))

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
