import os
import shutil
import tempfile
import unittest

import pandas as pd

from crackpy.crack_detection.model import get_model
from crackpy.crack_detection.pipeline.pipeline import CrackDetectionSetup, CrackDetectionPipeline


class TestCrackDetPipeline(unittest.TestCase):

    def setUp(self):
        self.data_path = os.path.join(  # '..', '..', '..', '..',
                                      'test_data', 'crack_detection', 'Nodemaps')
        self.crack_info_by_nodemap_file = os.path.join(  # '..', '..', '..', '..',
                                                       'test_data', 'crack_detection', 'crack_info_by_nodemap.txt')

        self.det_setup = CrackDetectionSetup(
            specimen_size=160,
            sides=['left'],
            detection_window_size=None,
            start_offset=(0, 0),
            angle_det_radius=13.725
        )

        # crack detectors
        self.tip_detector = get_model('ParallelNets')
        self.path_detector = get_model('UNetPath')

    def test_pipeline(self):
        temp_dir = tempfile.mkdtemp()
        try:
            pipeline = CrackDetectionPipeline(
                data_path=self.data_path,
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
            exp_results = pd.read_csv(self.crack_info_by_nodemap_file)
            act_results = pd.read_csv(os.path.join(temp_dir, 'crack_info_by_nodemap.txt'))
            pd.testing.assert_frame_equal(exp_results, act_results)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
