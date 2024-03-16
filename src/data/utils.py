from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch

from src.data.transformation import Remap


def dir_to_img_ids(dir_: Path) -> list[str]:
    """list all the ids of each file in a directory"""
    ids = []
    for file in dir_.iterdir():
        if not file.is_file():
            continue

        id_img = file.stem

        ids.append(id_img)

    return ids


def construct_img_path(
    dir_: Path, image_id: int | str, *, is_simulated: bool = False
) -> Path:
    """
    build the path to an image.

    Parameters
    ----------
    dir_ : Path
        directory in which is the image
    image_id : Union[int, str]
        id of the image. It is used as the file name.
    is_simulated : bool, optional
        whether or not it will be a npz file or a jpg file, by default False

    Returns
    -------
    Path
        path to the image
    """
    filepath = dir_ / str(image_id)

    if is_simulated:
        filepath = filepath.with_suffix(".npz")
    else:
        filepath = filepath.with_suffix(".jpg")

    return filepath


@overload
def get_simulated_image(
    simulated_dir: Path,
    image_id: int | str,
    pass_names: list[str] | None,
    return_nbr_of_channels_per_pass: Literal[True],
) -> dict[str, int]: ...


@overload
def get_simulated_image(
    simulated_dir: Path,
    image_id: int | str,
    pass_names: list[str] | None,
    return_nbr_of_channels_per_pass: Literal[False] = False,
) -> torch.Tensor: ...


def get_simulated_image(
    simulated_dir, image_id, pass_names=None, return_nbr_of_channels_per_pass=False
):
    """
    load a simulated image file from the disk.

    Parameters
    ----------
    dir_ : Path
        directory in which is the image
    image_id : Union[int, str]
        id of the image. It is used as the file name.
    pass_names : list[str]
        name of the passes to load. If none, will load all the passes
        in the file.
    return_nbr_of_channels_per_pass : bool, optional
        if True, returns a dict that gives the number of channels for
        each pass, by default False

    Returns
    -------
    Union[torch.Tensor, dict[str, int]]
        return the simulated image tensor or a dict that gives the number
        of channels for each pass
    """
    nbr_of_channels_per_pass = {}
    passes = []

    filepath = construct_img_path(simulated_dir, image_id, is_simulated=True)
    npz_file = np.load(filepath)

    if pass_names is None:
        pass_names = list(npz_file)

    for passname in pass_names:
        img = npz_file[passname]

        if passname == "Depth":
            img[img == np.inf] = 0
            img = torch.from_numpy(img)
            img = Remap(0, img.max(), 0, 1)(img)

        passes.append(img)
        nbr_of_channels_per_pass[passname] = img.shape[2]

    sim_image = np.concatenate(passes, 2)
    sim_image = np.transpose(sim_image, (2, 0, 1))
    sim_image = torch.from_numpy(sim_image).float()

    npz_file.close()

    if return_nbr_of_channels_per_pass:
        return nbr_of_channels_per_pass

    return sim_image
