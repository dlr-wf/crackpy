import os

import pkg_resources
import torch

from crackpy.crack_detection.deep_learning.nets import ParallelNets, UNet


def get_model(model_name: str, map_location=torch.device('cpu')):
    """Return the trained crack detection model *model_name* as a PyTorch model.
    If the model is not found, it will be downloaded from Zenodo DOI:10.5281/zenodo.7245516 first.

    Args:
        model_name: 'ParallelNets' or 'UNetPath'
        map_location: map_location for torch.load()-function, e.g. torch.device('cpu') or torch.device('cuda:0')

    Returns:
        torch model

    """
    # model urls on Zenodo
    model_urls = {'ParallelNets': 'https://zenodo.org/record/7245516/files/ParallelNets.pth?download=1',
                  'UNetPath': 'https://zenodo.org/record/7245516/files/UNetPath.pth?download=1'}

    # check if model_name is supported
    if model_name not in ['ParallelNets', 'UNetPath']:
        raise ValueError("Model name needs to be 'ParallelNets' or 'UNetPath'.")

    model_path = pkg_resources.resource_filename('crackpy', f'crack_detection/models/{model_name}.pth')

    # check if model folder exists
    origin, _ = os.path.split(model_path)
    if not os.path.exists(origin):
        os.makedirs(origin)

    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        torch.hub.download_url_to_file(model_urls[model_name], model_path)

    if model_name == 'ParallelNets':
        model = ParallelNets(in_ch=2, out_ch=1, init_features=64)
        model.load_state_dict(torch.load(model_path, map_location=map_location))
    else:  # model_name == 'UNetPath'
        model = UNet(in_ch=2, out_ch=1, init_features=64)
        model.load_state_dict(torch.load(model_path, map_location=map_location))

    return model
