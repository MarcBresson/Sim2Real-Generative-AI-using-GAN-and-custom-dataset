from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


def plot_sim(images: list[np.ndarray], pass_names: list[str], suptitle: str = None, horizontal: bool = False) -> Figure:
    """
    plot a simulated image with all its channels.

    Parameters
    ----------
    images : list[np.ndarray]
        images of each channels. To get this list, you can use
        data.visualisation.multichannels_to_individuals() on your
        simulated image.
    pass_names : list[str]
        name of each pass for the plot title
    horizontal : bool, optional
        align the subplots horizontaly or verticaly, by default True

    Returns
    -------
    matplotlib.Figure
        the figure ready to be ploted
    """
    if horizontal:
        nrows = 1
        ncols = len(images)
    else:
        nrows = len(images)
        ncols = 1

    axs: np.ndarray[Axes]
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.ravel()

    if suptitle is not None:
        fig.suptitle(suptitle)

    for i, image in enumerate(images):
        # the image can have negative values
        image = np.abs(image.astype("uint8"))
        axs[i].imshow(image)
        axs[i].set_title(pass_names[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    return fig


def plot_streetview_with_discrimination(streetview: np.ndarray, discrimination: np.ndarray, target: np.ndarray, suptitle: str = None, horizontal: bool = False):
    if horizontal:
        nrows = 1
        ncols = 3
    else:
        nrows = 3
        ncols = 1

    axs: list[Axes]
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.ravel()

    if suptitle is not None:
        fig.suptitle(suptitle)

    imgs = {"generated\nstreetview": streetview, "associated\ndiscrimination": discrimination, "target": target}
    for i, (name, image) in enumerate(imgs.items()):
        image = np.abs(image.astype("uint8"))
        axs[i].imshow(image)
        axs[i].set_title(name)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    return fig
