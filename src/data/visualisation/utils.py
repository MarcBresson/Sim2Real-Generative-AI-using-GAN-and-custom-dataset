from typing import Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import matplotlib as mpl
import numpy as np


def plot_sim(images: list[np.ndarray], pass_names: list[str], suptitle: Union[str, None] = None, horizontal: bool = False) -> Figure:
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
    fig, axs = get_fig_axs(dim_0=len(images), horizontal=horizontal)

    if suptitle is not None:
        fig.suptitle(suptitle)

    for i, image in enumerate(images):
        show_rgb_image(axs[i], image, pass_names[i])

    return fig


def plot_streetview_with_discrimination(streetview: np.ndarray, discrimination: np.ndarray, target: np.ndarray, suptitle: Union[str, None] = None, horizontal: bool = False) -> Figure:
    """
    plot the result of the GAN with the generated sample, the discrimination and the target side by side.

    Parameters
    ----------
    streetview : np.ndarray
        image value of the generated sample.
    discrimination : np.ndarray
        value of the discriminator output.
    target : np.ndarray
        image value of the target.
    suptitle : Union[str, None], optional
        The plot suptitle, by default None.
    horizontal : bool, optional
        Whether to plot the images horizontally or vertically, by default False.

    Returns
    -------
    Figure
        return the figure with the outputs plotted on.
    """
    fig, axs = get_fig_axs(horizontal=horizontal)

    if suptitle is not None:
        fig.suptitle(suptitle)

    show_rgb_image(axs[0], streetview, "Generated")
    show_discrimination(axs[1], discrimination, "Discrimination")
    show_rgb_image(axs[2], target, "Target")

    return fig


def get_fig_axs(dim_0: int = 3, dim_1: int = 1, horizontal: bool = False) -> tuple[Figure, list[Axes]]:
    """
    get the Axes of a figure in a 1D array.

    Parameters
    ----------
    dim_0 : int, optional
        number of raws, by default 3.
    dim_1 : int, optional
        number of columns, by default 1.
    horizontal : bool, optional
        whether to invert raws and columns, by default False.

    Returns
    -------
    tuple[Figure, list[Axes]]
        a tuple with the figure and a 1D array containing figure Axes.
    """
    if horizontal:
        dim_0, dim_1 = dim_1, dim_0

    axs: Union[Axes, np.ndarray]
    fig, axs = plt.subplots(dim_0, dim_1)

    axs_1d: list[Axes] = []
    if isinstance(axs, np.ndarray):
        axs = axs.ravel()
        axs_1d = list(axs)
    else:
        axs_1d = [axs]

    return fig, axs_1d


def show_rgb_image(ax: Axes, image: np.ndarray, title: Union[str, None] = None):
    """
    display an image onto an axe.

    Parameters
    ----------
    ax : Axes
        axe to draw the image on.
    image : np.ndarray
        value of the image, can have 1 or 3 channels.
    title : Union[str, None], optional
        _description_, by default None
    """
    ax.imshow(image)

    if title is not None:
        ax.set_xlabel(title)
        ax.xaxis.set_label_position('top')

    remove_border(ax)


def show_discrimination(ax: Axes, image: np.ndarray, title: Union[str, None] = None, display_legend: bool = False):
    """
    display the discriminator output onto an axe.

    Parameters
    ----------
    ax : Axes
        axe to draw the discriminator output on.
    image : np.ndarray
        value of the discriminator output.
    title : Union[str, None], optional
        _description_, by default None
    display_legend : bool, optional
        _description_, by default False
    """
    cmap = mpl.colormaps["viridis"]

    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)

    if title is not None:
        ax.set_xlabel(title)
        ax.xaxis.set_label_position('top')

    if display_legend:
        legend = (Patch(facecolor=cmap(0), label='Fake'),
                  Patch(facecolor=cmap(1), label='Real'))
        ax.legend(handles=legend, prop={"size": 6})

    remove_border(ax)


def remove_border(ax: Axes):
    """
    remove all the borders of an Axe.

    Parameters
    ----------
    ax : Axes
        the Axe to remove borders from.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remove ticks but let axis labels
    ax.set_xticks([])
    ax.set_yticks([])
