from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import matplotlib as mpl
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
    fig, axs = get_fig_axs(dim_0=len(images), horizontal=horizontal)

    if suptitle is not None:
        fig.suptitle(suptitle)

    for i, image in enumerate(images):
        show_rgb_image(axs[i], image, pass_names[i])

    return fig


def plot_streetview_with_discrimination(streetview: np.ndarray, discrimination: np.ndarray, target: np.ndarray, suptitle: str = None, horizontal: bool = False):
    fig, axs = get_fig_axs(horizontal=horizontal)

    if suptitle is not None:
        fig.suptitle(suptitle)

    show_rgb_image(axs[0], streetview, "Generated")
    show_discrimination(axs[1], discrimination, "Discrimination")
    show_rgb_image(axs[2], target, "Target")

    return fig


def get_fig_axs(dim_0: int = 3, dim_1: int = 1, horizontal: bool = False):
    if horizontal:
        dim_0, dim_1 = dim_1, dim_0

    axs: np.ndarray[Axes]
    fig, axs = plt.subplots(dim_0, dim_1)
    axs = axs.ravel()

    return fig, axs


def show_rgb_image(ax: Axes, image: np.ndarray, title: str = None):
    ax.imshow(image)
    ax.set_xlabel(title)
    ax.xaxis.set_label_position('top')
    remove_border(ax)


def show_discrimination(ax: Axes, image: np.ndarray, title: str = None, display_legend: bool = False):
    cmap = mpl.colormaps["viridis"]

    ax.imshow(image, cmap=cmap, vmin=0, vmax=256)
    ax.set_xlabel(title)
    ax.xaxis.set_label_position('top')

    if display_legend:
        legend = (Patch(facecolor=cmap(0), label='Fake'),
                  Patch(facecolor=cmap(256), label='Real'))
        ax.legend(handles=legend, prop={"size": 6})

    remove_border(ax)


def remove_border(ax: Axes) -> Axes:
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remove ticks but let axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
