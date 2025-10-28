import logging
from importlib import resources
from pathlib import Path
import torch

from crackpy.crack_detection.deep_learning.nets import ParallelNets, UNet

logger = logging.getLogger(__name__)


def get_model(model_name: str, map_location: torch.device = torch.device('cpu')) -> ParallelNets | UNet:
    """Return the trained crack detection model *model_name* as a PyTorch model.

    If the model is not found, it will be downloaded from Zenodo DOI:10.5281/zenodo.7245516 first.

    Args:
        model_name: 'ParallelNets' or 'UNetPath'
        map_location: map_location for torch.load()-function, e.g. torch.device('cpu') or torch.device('cuda:0')

    Returns:
        Trained PyTorch model (either ParallelNets or UNet)

    """
    # model urls on Zenodo
    model_urls = {'ParallelNets': 'https://zenodo.org/record/7245516/files/ParallelNets.pth?download=1',
                  'UNetPath': 'https://zenodo.org/record/7245516/files/UNetPath.pth?download=1'}

    # check if model_name is supported
    if model_name not in ['ParallelNets', 'UNetPath']:
        raise ValueError("Model name needs to be 'ParallelNets' or 'UNetPath'.")

    # Use importlib.resources.files() to get the model path
    model_folder = resources.files('crackpy').joinpath('crack_detection/models')
    model_path = Path(str(model_folder)) / f'{model_name}.pth'

    # check if model folder exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Model folder ensured: %s", model_path.parent)

    if not model_path.exists():
        logger.info("Downloading model file for %s …", model_name)
        torch.hub.download_url_to_file(model_urls[model_name], str(model_path))
    else:
        logger.debug("Loading existing model from %s", model_path)

    if model_name == 'ParallelNets':
        model = ParallelNets(in_ch=2, out_ch=1, init_features=64)
        model.load_state_dict(torch.load(str(model_path), map_location=map_location))
        logger.debug("ParallelNets model loaded successfully to %s", map_location)

    else:  # model_name == 'UNetPath'
        model = UNet(in_ch=2, out_ch=1, init_features=64)
        model.load_state_dict(torch.load(str(model_path), map_location=map_location))
        logger.debug("UNetPath model loaded successfully to %s", map_location)

    return model
