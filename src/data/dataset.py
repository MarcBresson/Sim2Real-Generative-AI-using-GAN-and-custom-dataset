"""
inspired from
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/vision/main/transforms.html
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.io import read_image

logger = logging.getLogger(__name__)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: Path,
        streetview_dir: Path,
        blender_dir: Path,
        render_passes: list[str],
    ):
        self.annotations = pd.read_feather(annotations_file)
        self.filter_incomplete_rows()

        self.streetview_dir = streetview_dir
        self.simulated_dir = blender_dir

        self.render_passes = render_passes
        self.set_passes_channel_nbr()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        image_id = self.annotations.iloc[idx]["image_id"]

        truth_img_path = self.streetview_dir / str(image_id)
        truth_img_path = truth_img_path.with_suffix(".jpg")
        truth_img = read_image(str(truth_img_path)) / 255

        simul_img, _ = get_simulated_image(self.render_passes, self.simulated_dir, image_id)

        sample = {"streetview": truth_img, "simulated": simul_img}

        return sample

    def filter_incomplete_rows(self):
        """
        will filter out rows that are missing either one of their
        simulated images or their streetview.
        """
        rows_to_delete = []

        for index, row in self.annotations.iterrows():
            img_id = str(row["image_id"])
            if (
                not image_exists(self.simulated_dir, img_id + "_", expected_length=len(self.render_passes))
                and not image_exists(self.streetview_dir, img_id + ".")
            ):
                rows_to_delete.append(index)

        self.annotations.drop(rows_to_delete)

        logger.info("dataset - droped %s element because they were missing at least"
                    "one image. The dataset now has %s samples", len(rows_to_delete),
                    len(self.annotations))

    @property
    def passes_channel_nbr(self):
        """return the number of channels for each pass in a simulated image."""
        return self._passes_channel_nbr

    def set_passes_channel_nbr(self):
        """compute the number of channels for each pass in a simulated image."""
        image_id = self.annotations.iloc[0]["image_id"]

        _, passes_channel_nbr = get_simulated_image(self.render_passes, self.simulated_dir, image_id)

        self._passes_channel_nbr = tuple(passes_channel_nbr)


def image_exists(search_dir: Path, prefix: str, expected_length: int = 1):
    search_results = list(search_dir.glob(f"{prefix}*"))

    return len(search_results) == expected_length


def get_simulated_image(simulated_dir: Path, image_id: int, render_passes: list[str] = None) -> tuple[torch.Tensor, list[int]]:
    images = []
    nbr_channels = []

    for img_path in simulated_dir.glob(f"{image_id}_*"):
        render_pass_name = img_path.stem.split("_")[1]

        if render_passes is None or render_pass_name in render_passes:
            if img_path.suffix.lower() == ".npy":
                img = torch.from_numpy(np.load(img_path))
            elif img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img = read_image(str(img_path))
            else:
                raise ValueError(f"{img_path.suffix} as image file is not supported. Use npy, jpg or png.")

            if len(img.shape) == 2:
                nbr_channels.append(1)
            else:
                nbr_channels.append(img.shape[0])

            images.append(img)

    return concat_channels(images), nbr_channels


def concat_channels(images: list[torch.Tensor]) -> torch.Tensor:
    """
    concat the channels of a list of images. If you have red blue and
    green in three different files, you can use this function to create
    a single rgb image.
    """
    for img in images:
        # if img has only two dim (one channel), burry the channel in a separate dim
        if len(img.shape) == 2:
            img = torch.unsqueeze(img, 0)

    return torch.cat(images)


def split_train_test_val(dataset: Dataset, proportions: tuple[float, float, float]) -> tuple[Subset, Subset, Subset]:
    if sum(proportions) != 1:
        raise ValueError("The proportions of the splitting should sum up to 1.")

    if len(proportions) != 3:
        raise ValueError("You must specify 3 fractions for the train, test and validation subsets.")

    return random_split(dataset, proportions, torch.Generator().manual_seed(42))
