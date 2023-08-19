from torch import Tensor
import numpy as np

from .utils import plot_sim, plot_streetview_with_discrimination
from src.data.transformation import toNumpy


def batch_to_numpy(batch: Tensor) -> list[np.ndarray]:
    """
    Convert a batch of images to a list of nd.array, scaled back to
    range [0, 255].
    """
    batch_np = toNumpy()(batch)

    np_images = []
    for torch_image in batch_np:
        np_images.append(torch_image)

    return np_images


def multichannels_to_individuals(img: np.ndarray, passes_channels: dict[str, int]) -> list[np.ndarray]:
    """
    transform a multichannel image to individual, less than 3 channels
    images.

    Parameters
    ----------
    img : np.ndarray
        the multi spectral image.
    passes_channels : list[int]
        the list indicating the number of channels for each pass.
        You can get this list from the .passes_channel_nbr dataset property.

    Returns
    -------
    list[np.ndarray]
        the mono spectral images.
    """
    i_channel = 0

    individual_images = []
    for _, pass_channels in passes_channels.items():
        individual_images.append(img[:, :, i_channel: i_channel + pass_channels])

        i_channel += pass_channels

    return individual_images
