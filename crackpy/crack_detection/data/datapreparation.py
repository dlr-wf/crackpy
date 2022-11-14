import os
import numpy as np

from crackpy.fracture_analysis.data_processing import InputData
from crackpy.structure_elements.data_files import Nodemap


def ground_truth_import(path: str, side: str) -> np.ndarray:
    """Import of ground truth data.

    Args:
        path: path of ground truth data file. Expected is a path ending with '.txt' without any header
        side: indicating left or right-hand side of specimen

    Returns:
        np.ndarray

    """
    with open(path[:-4] + '_' + side + '.txt') as file:
        array2d = [[float(digit) for digit in line.split()] for line in file]
        array2d = np.asarray(array2d)
    if side == 'left':
        array2d = np.fliplr(array2d)
    return array2d


def import_data(nodemaps: dict, data_path: str, side: str ='', exists_target=True):
    """Create dictionaries for Nodemaps and ground truth data.

    Args:
        nodemaps: dict of nodemaps by stage numbers
        data_path: data folder must contain sub-folders 'Nodemaps', 'GroundTruth'
        side: indicating left or right-hand side of specimen
        exists_target: indicates whether or not a target is available

    Returns:
        inputs, ground_truths (dicts)

    """
    print('Data will be imported for ' + side + ' side of the specimen...')

    inputs = {}
    ground_truths = {}

    for _, nodemap in sorted(nodemaps.items()):
        print(f'\r- {nodemap}. {len(inputs.keys()) + 1}/{len(nodemaps)} imported.', end='')
        input_nodemap = Nodemap(name=nodemap, folder = os.path.join(data_path, "Nodemaps"))
        input_by_nodemap = InputData(input_nodemap)
        inputs.update({nodemap + '_' + side: input_by_nodemap})

        if exists_target:
            path = os.path.join(data_path, 'GroundTruth', nodemap)
            ground_truth_by_nodemap = ground_truth_import(path, side)
            ground_truths.update({nodemap + '_' + side: ground_truth_by_nodemap})
        else:
            ground_truths = None

    print('\n')

    return inputs, ground_truths
