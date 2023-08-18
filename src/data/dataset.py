"""
inspired from
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/vision/main/transforms.html
"""

from typing import Union
from pathlib import Path
from collections import Counter
import logging
import math

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.io import read_image

from src.data import transformation

logger = logging.getLogger(__name__)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: Path,
        streetview_dir: Path,
        blender_dir: Path,
        render_passes: dict[str, str] = None,
        transform=None,
    ):
        self.annotations = pd.read_feather(annotations_file)
        self.render_passes = render_passes

        self.streetview_dir = streetview_dir
        self.simulated_dir = blender_dir

        self.transform = transform

        self.filter_incomplete_rows()
        logging.info("the dataset has %s samples", len(self.annotations))

        self.set_passes_channel_nbr()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        sample = self.get_untransformed_sample(idx)

        if self.transform is not None:
            sample = self.transform_sample(sample)

        return sample

    def get_untransformed_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """
        get a raw sample from the dataset. Each time this function is called, the disk
        is read two times: one for the simulated images, and one for the ground truth.

        Parameters
        ----------
        idx : int
            the position of the sample in the annotation file.

        Returns
        -------
        dict[str, torch.Tensor]
            the raw sample.
        """
        image_id = self.annotations.iloc[idx]["image_id"]

        truth_img_path = self.streetview_dir / str(image_id)
        truth_img_path = truth_img_path.with_suffix(".jpg")
        truth_img = read_image(str(truth_img_path)).float()

        simul_img = get_simulated_image(self.simulated_dir, image_id, self.render_passes)

        sample = {"streetview": truth_img, "simulated": simul_img}

        return sample

    def transform_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply a transformation on a sample

        Parameters
        ----------
        sample : dict[str, torch.Tensor]
            the untransformed sample.

        Returns
        -------
        dict[str, torch.Tensor]
            the transformed sample

        Raises
        ------
        ValueError
            when this function is called but no transformation was given.
        """
        if self.transform is None:
            raise ValueError("No transformation given. Use CustomImageDataset"
                             "(..., transform=transform), to allow transformation.")

        sample = self.transform(sample)

        return sample

    def filter_incomplete_rows(self):
        """
        filters out rows that are missing either one of their
        simulated images or their streetview.
        """
        simulated_ids = dir_to_img_ids(self.simulated_dir)
        streetview_ids = dir_to_img_ids(self.streetview_dir)

        all_ids = []
        all_ids.extend(simulated_ids)
        all_ids.extend(streetview_ids)

        count_files = Counter(all_ids)

        valid_ids = []
        for img_id, count in count_files.items():
            if count == 2:
                valid_ids.append(img_id)

        before_deletion = len(self.annotations)
        self.annotations = self.annotations[self.annotations["image_id"].isin(valid_ids)]

        deleted_samples = before_deletion - len(self.annotations)
        logger.info("dataset - droped %s element because they were missing at least"
                    "one image. The dataset now has %s samples", deleted_samples,
                    len(self.annotations))

    def delete_incomplete_files(self):
        """
        if a row lacks at least one corresponding file on the disk, every related
        files are deleted.
        """
        simulated_ids = dir_to_img_ids(self.simulated_dir)
        streetview_ids = dir_to_img_ids(self.streetview_dir)

        all_ids = []
        all_ids.extend(simulated_ids)
        all_ids.extend(streetview_ids)

        count_files = Counter(all_ids)

        valid_ids = []
        for img_id, count in count_files.items():
            if count == 2:
                valid_ids.append(img_id)

        to_remove = self.annotations[~self.annotations["image_id"].isin(valid_ids)]
        to_remove_ids = to_remove["image_id"].to_list()

        for image_id in to_remove_ids:
            delete_image(self.streetview_dir, self.simulated_dir, image_id)

        logger.info("dataset - removed %s samples because they did not form a complete set", len(to_remove_ids))

    def delete_unloadable_data(self):
        """try to load every simulated images, and delete them if it fails."""
        index_to_drop = []

        deletion_progression = tqdm(self.annotations.itertuples(), desc="deleting unloadable data", total=len(self.annotations), miniters=int(len(self.annotations) / 100))
        for serie in deletion_progression:
            try:
                get_simulated_image(self.simulated_dir, serie.image_id, self.render_passes)
            except (RuntimeError, ValueError):
                # RuntimeError when file is empty
                # ValueError when it was pickled
                index_to_drop.append(serie.Index)

                delete_image(self.streetview_dir, self.simulated_dir, serie.image_id)

        self.annotations = self.annotations.drop(index=index_to_drop)
        logger.info("dataset - droped %s element because the data could not be loaded."
                    "The dataset now has %s samples", len(index_to_drop),
                    len(self.annotations))

    @property
    def passes_channel_nbr(self):
        """return the number of channels for each pass in a simulated image."""
        return self._passes_channel_nbr

    def set_passes_channel_nbr(self):
        """compute the number of channels for each pass in a simulated image."""
        image_id = self.annotations.iloc[0]["image_id"]

        channels_per_pass: dict
        channels_per_pass = get_simulated_image(self.simulated_dir, image_id, self.render_passes, return_nbr_of_channels_per_pass=True)

        self._passes_channel_nbr = channels_per_pass

        self.render_passes = list(channels_per_pass.keys())


def delete_image(street_view_dir: Path, simulated_dir: Path, image_id: Union[str, int]):
    img_path = construct_img_path(simulated_dir, image_id, is_simulated=True)
    img_path.unlink(missing_ok=True)
    img_path = construct_img_path(street_view_dir, image_id, is_simulated=False)
    img_path.unlink(missing_ok=True)


def get_simulated_image(simulated_dir: Path, image_id: int, pass_names: list[str], return_nbr_of_channels_per_pass: bool = False) -> Union[torch.Tensor, dict[str, int]]:
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
            img = transformation.Remap(0, img.max(), 0, 1)(img)

        passes.append(img)
        nbr_of_channels_per_pass[passname] = img.shape[2]

    sim_image = np.concatenate(passes, 2)
    sim_image = np.transpose(sim_image, (2, 0, 1))
    sim_image = torch.from_numpy(sim_image).float()

    npz_file.close()

    if return_nbr_of_channels_per_pass:
        return nbr_of_channels_per_pass

    return sim_image


def construct_img_path(dir_: Path, image_id: Union[int, str], *, is_simulated: bool = False) -> Path:
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


def dataset_split(dataset: Dataset, proportions: list[float]) -> list[Subset]:
    """
    split a dataset in regards in proportions. It can be a mix of integers
    and float numbers.

    Parameters
    ----------
    dataset : Dataset
        dataset to split
    proportions : list[float]
        the sum of the integers must be less than the length of the dataset,
        and the sum of the fractions must be <= 1. If integers and fractions
        are mixed up, it will create subsets with integres first, and will
        fractionate the remaining dataset.

    Returns
    -------
    list[Subset]
        all subsets with the same order of proportion.
    """
    integer_sum = sum([n for n in proportions if isinstance(n, int)])
    remaining_length = len(dataset) - integer_sum

    if remaining_length < 0:
        raise ValueError("There is not enough samples in the given dataset to satisfy"
                         " all the integers condtions.")

    if remaining_length != 0:
        int_divider = len(dataset)
        fraction_divider = len(dataset) / remaining_length

        for i, p in enumerate(proportions):
            if isinstance(p, int):
                proportions[i] = p / int_divider
            elif isinstance(p, float) and p <= 1:
                proportions[i] = p / fraction_divider

    # float division error can sometimes bring the sum to more than 1
    # so this bit of code spread the error on each fraction.
    if math.isclose(sum(proportions), 1) and sum(proportions) > 1:
        too_much = sum(proportions) - 1

        for i, p in enumerate(proportions):
            proportions[i] = p - too_much / len(proportions)

    discard_overflow = False
    if not math.isclose(sum(proportions), 1) and sum(proportions) < 1:
        discard_overflow = True
        proportions.append(1 - sum(proportions))

    subsets = random_split(dataset, proportions, torch.Generator().manual_seed(42))

    if discard_overflow:
        subsets.pop()

    return subsets


def dir_to_img_ids(dir_: Path) -> list[int]:
    """list all the ids of each file in a directory"""
    ids = []
    for file in dir_.iterdir():
        if not file.is_file():
            continue

        id_img = file.stem

        ids.append(int(id_img))

    return ids
