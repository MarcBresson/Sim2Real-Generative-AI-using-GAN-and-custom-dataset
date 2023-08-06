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

from src.data.acquisition.array_compresser import load_compressed_array
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
        to_device: torch.device = None
    ):
        self.annotations = pd.read_feather(annotations_file)
        self.render_passes = set_render_passes(render_passes, blender_dir)

        self.streetview_dir = streetview_dir
        self.simulated_dir = blender_dir

        self.transform = transform

        self.to_device = to_device

        self.filter_incomplete_rows()
        logging.info("the dataset has %s samples", len(self.annotations))

        self.set_passes_channel_nbr()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        sample = self.get_untransformed_sample(idx)

        if self.transform is not None:
            sample = self.transform_sample(sample)

        if self.to_device is not None:
            sample["streetview"].to(self.to_device)
            sample["simulated"].to(self.to_device)

        return sample

    def get_untransformed_sample(self, idx) -> dict[str, torch.Tensor]:
        image_id = self.annotations.iloc[idx]["image_id"]

        truth_img_path = self.streetview_dir / str(image_id)
        truth_img_path = truth_img_path.with_suffix(".jpg")
        truth_img = read_image(str(truth_img_path)).float()

        simul_img, _ = get_simulated_image(self.simulated_dir, image_id, self.render_passes)

        sample = {"streetview": truth_img, "simulated": simul_img}

        return sample

    def transform_sample(self, sample):
        if self.transform is None:
            raise ValueError("No transformation given. Use CustomImageDataset"
                             "(..., transform=transform), to allow transformation.")

        # the transform operation converts single sample to a batched sample
        sample = self.transform(sample)

        # we debatch the sample for the dataloader
        # sample = {"streetview": sample["streetview"][0], "simulated": sample["simulated"][0]}

        return sample

    def filter_incomplete_rows(self):
        """
        filters out rows that are missing either one of their
        simulated images or their streetview.
        """
        simulated_ids = dir_to_passes_id(self.simulated_dir)
        streetview_ids = dir_to_passes_id(self.streetview_dir)

        all_ids = []
        for passname in simulated_ids.keys():
            all_ids.extend(simulated_ids[passname]["ids"])
        all_ids.extend(streetview_ids[""]["ids"])

        count_passes = Counter(all_ids)
        target_nbr_passes = len(simulated_ids) + len(streetview_ids)

        rows_to_delete = 0
        valid_ids = []
        for img_id, count in count_passes.items():
            if count == target_nbr_passes:
                valid_ids.append(img_id)
            else:
                rows_to_delete += 1

        self.annotations = self.annotations[self.annotations["image_id"].isin(valid_ids)]

        logger.info("dataset - droped %s element because they were missing at least"
                    "one image. The dataset now has %s samples", rows_to_delete,
                    len(self.annotations))

    def delete_incomplete_files(self):
        simulated_ids = dir_to_passes_id(self.simulated_dir)
        streetview_ids = dir_to_passes_id(self.streetview_dir)

        all_ids = []
        for passname in simulated_ids.keys():
            all_ids.extend(simulated_ids[passname]["ids"])
        all_ids.extend(streetview_ids[""]["ids"])

        count_passes = Counter(all_ids)
        target_nbr_passes = len(simulated_ids) + len(streetview_ids)

        rows_to_delete = 0
        valid_ids = []
        for img_id, count in count_passes.items():
            if count == target_nbr_passes:
                valid_ids.append(img_id)
            else:
                rows_to_delete += 1

        to_remove = self.annotations[~self.annotations["image_id"].isin(valid_ids)]
        to_remove_ids = to_remove["image_id"].to_list()

        nbr_deleted = 0
        for image_id in to_remove_ids:
            for passname, ext in self.render_passes.items():
                img_path = construct_img_path(self.simulated_dir, image_id, ext, passname)
                try:
                    img_path.unlink()
                    nbr_deleted += 1
                except FileNotFoundError:
                    pass

        logger.info("dataset - deleted %s files because they did not form a complete set", nbr_deleted)

    def delete_unloadable_data(self):
        """try to load every simulated images, and delete them if it fails."""
        index_to_drop = []

        deletion_progression = tqdm(self.annotations.itertuples(), desc="deleting unloadable data", total=len(self.annotations))
        for serie in deletion_progression:
            try:
                get_simulated_image(self.simulated_dir, serie.image_id, self.render_passes)
            except (RuntimeError, ValueError):
                # RuntimeError when file is empty
                # ValueError when it was pickled
                index_to_drop.append(serie.Index)

                for passname, ext in self.render_passes.items():
                    img_path = construct_img_path(self.simulated_dir, serie.image_id, ext, passname)
                    img_path.unlink()

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

        _, passes_channel_nbr = get_simulated_image(self.simulated_dir, image_id, self.render_passes)

        self._passes_channel_nbr = tuple(passes_channel_nbr)


def get_simulated_image(simulated_dir: Path, image_id: int, render_passes: dict[str, str]) -> tuple[torch.Tensor, list[int]]:
    images = []
    nbr_channels = []

    for passname, ext in render_passes.items():
        img_path = construct_img_path(simulated_dir, image_id, ext, passname)

        if img_path.suffix.lower() == ".npy":
            img = torch.from_numpy(np.load(img_path))
        if img_path.suffix.lower() == ".npz":
            img = torch.from_numpy(load_compressed_array(img_path))
        elif img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = read_image(str(img_path))
        else:
            raise ValueError(f"{img_path.suffix} as image file is not supported. Use npy, jpg or png.")

        if len(img.shape) == 2:
            nbr_channels.append(1)
        else:
            nbr_channels.append(img.shape[0])

        images.append(img)

    sim_image = concat_channels(images).float()

    return sim_image, nbr_channels


def concat_channels(images: list[torch.Tensor]) -> torch.Tensor:
    """
    concat the channels of a list of images. If you have red blue and
    green in three different files, you can use this function to create
    a single rgb image.
    """
    for i, img in enumerate(images):
        # if img has only two dim (one channel), burry the channel in a separate dim
        if len(img.shape) == 2:
            images[i] = torch.unsqueeze(img, 0)

    return torch.cat(images)


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


def set_render_passes(render_passes: list[str], blender_dir: Path) -> dict[str, str]:
    """if render_passes is None, will include all the render_passes found in the directory"""
    if render_passes is not None:
        return render_passes

    render_passes_ = {passname: value["ext"] for passname, value in dir_to_passes_id(blender_dir).items()}

    return render_passes_


def dir_to_passes_id(dir: Path):
    """list all the ids for each pass in a directory"""
    ids = {}
    for file in dir.iterdir():
        if not file.is_file():
            continue

        id_img = file.stem.split("_")[0]
        if "_" in file.stem:
            pass_name = file.stem.split("_")[1]
        else:
            pass_name = ""

        if pass_name not in ids:
            extension = file.suffix
            ids[pass_name] = {"ext": extension, "ids": []}

        ids[pass_name]["ids"].append(int(id_img))

    return ids


def construct_img_path(dir: Path, image_id: Union[int, str], extension: str, passname: str = None):
    """construct the path to an image"""
    if passname is not None:
        img_filename = str(image_id) + "_" + passname
    else:
        img_filename = str(image_id)

    img_path = dir / img_filename
    img_path = img_path.with_suffix(extension)

    return img_path
