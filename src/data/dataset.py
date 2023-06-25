"""
inspired from
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/vision/main/transforms.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: Path,
        streetview_dir: Path,
        blender_dir: Path,
        render_passes: list[str],
    ):
        self.annotations = pd.read_feather(annotations_file)

        self.streetview_dir = streetview_dir
        self.simulated_dir = blender_dir

        self.render_passes = render_passes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        image_id = self.annotations.iloc[idx]["image_id"]

        truth_img_path = self.streetview_dir / str(image_id)
        truth_img_path = truth_img_path.with_suffix(".jpg")
        truth_img = read_image(str(truth_img_path)) / 255

        simul_img = get_simulated_image(self.render_passes, self.simulated_dir, image_id) / 255

        sample = {"streetview": truth_img, "simulated": simul_img}

        return sample


def get_simulated_image(simulated_dir: Path, image_id: int, render_passes: list[str] = None) -> torch.Tensor:
    images = []

    for img_path in simulated_dir.glob(f"{image_id}_*"):
        render_pass_name = img_path.stem.split("_")[1]

        if render_passes is None or render_pass_name in render_passes:
            if img_path.suffix.lower() == ".npy":
                img = torch.from_numpy(np.load(img_path))
                img = img.unsqueeze(0)
            elif img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img = read_image(str(img_path))
            else:
                raise ValueError(f"{img_path.suffix} as image file is not supported. Use npy, jpg or png.")

            images.append(img)

    return concat_channels(images)


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


def split_train_test_val():
    pass
