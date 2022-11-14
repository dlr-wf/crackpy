import os
import shutil
import tempfile
import unittest

from crackpy.crack_detection.data.interpolation import interpolate_on_array
from crackpy.crack_detection.utils.plot import plot_prediction
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums
from crackpy.crack_detection.data import datapreparation as dp


class TestPlotting(unittest.TestCase):

    def setUp(self):
        origin = os.path.join(  # '..', '..', '..', '..',
                              'test_data', 'crack_detection', 'raw')
        self.side = 'left'
        self.size = 70

        self.stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(os.path.join(origin, 'Nodemaps'),
                                                                 ['7'])

        # import
        inputs, ground_truths = dp.import_data(nodemaps=self.stages_to_nodemaps,
                                               data_path=origin,
                                               side=self.side,
                                               exists_target=True)
        # interpolate
        interp_size = self.size if self.side == 'right' else self.size * -1
        _, _, self.interp_eps_vm = interpolate_on_array(input_by_nodemap=inputs,
                                                        interp_size=interp_size,
                                                        pixels=256)

        self.ground_truths = ground_truths

    def test_plot_prediction(self):
        key = self.stages_to_nodemaps[7] + '_' + self.side
        background = self.interp_eps_vm[key]

        temp_dir = tempfile.mkdtemp()
        try:
            # test plotting without crack tips or path
            plot_prediction(background=background * 100,
                            interp_size=self.size,
                            save_name='test_file_wo_crack_tip',
                            path=temp_dir)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
