"""
inspired from
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/vision/main/transforms.html
"""

from typing import Union
from pathlib import Path
from collections import Counter
import threading
import logging
import math
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.io import read_image

from src.data.utils import dir_to_img_ids, construct_img_path, get_simulated_image
from src.data import transformation
from src.data.acquisition.utils import download_image

logger = logging.getLogger(__name__)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: Path,
        streetview_dir: Path,
        blender_dir: Path,
        render_passes: list[str] = None,
        *,
        transform=None,
        download_missing_mapillary: bool = False,
        delete_unused_files: bool = False,
        filter_incomplete_rows: bool = True,
    ):
        self.annotations = pd.read_feather(annotations_file)
        self.render_passes = render_passes

        self.streetview_dir = streetview_dir
        self.simulated_dir = blender_dir

        self.transform = transform

        if download_missing_mapillary:
            self.download_missing_streetviews()
        if delete_unused_files:
            self.delete_unused_files()
        if filter_incomplete_rows:
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

        try:
            truth_img = read_image(str(truth_img_path)).float()
        except RuntimeError as exc:
            raise RuntimeError(f"Image at {truth_img_path} could not be loaded.") from exc
        truth_img = transformation.Remap(0, 255, 0, 1)(truth_img)

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
        logger.info("dataset - droped %s element because they were missing at least "
                    "one image. The dataset now has %s samples.", deleted_samples,
                    len(self.annotations))

    def download_missing_streetviews(self):
        simulated_ids = dir_to_img_ids(self.simulated_dir)
        streetview_ids = dir_to_img_ids(self.streetview_dir)

        s = set(streetview_ids)
        ids_to_download = [x for x in simulated_ids if x not in s]

        if len(ids_to_download) > 0:
            logging.info("%s street views to download.", len(ids_to_download))
            rows_to_download = self.annotations[self.annotations["image_id"].isin(ids_to_download)]

            for _, row in tqdm(rows_to_download.iterrows(), total=len(rows_to_download)):
                img_id = row["image_id"]
                url = row["thumb_2048_url"]
                time.sleep(0.3)  # throttle down
                download_image(self.streetview_dir, img_id, url, use_thread=True)
        else:
            logging.info("no street view to download.")

    def delete_unused_files(self):
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

    def delete_unloadable_files(self, streetviews_only: bool = True, simulated_only: bool = False):
        """try to load every simulated and streetview images, and delete them if it fails."""
        if streetviews_only and simulated_only:
            raise ValueError("cannot be only streetviews and only simulated at the same time. "
                             "To do both, set streetviews_only to `False`.")

        def attempt_loading_simulated(image_id):
            try:
                get_simulated_image(self.simulated_dir, image_id, self.render_passes)
                return True
            except (RuntimeError, ValueError):
                # RuntimeError when file is empty
                # ValueError when it was pickled
                return False

        def attempt_loading_streetview(image_id):
            try:
                truth_img_path = self.streetview_dir / str(image_id)
                truth_img_path = truth_img_path.with_suffix(".jpg")
                read_image(str(truth_img_path)).float()
                return True
            except RuntimeError:
                # RuntimeError when jpeg is corrupted
                return False

        def check_image_id(image_id, count_element_removed: list):
            if not simulated_only:
                success_loading_streetview = attempt_loading_streetview(image_id)

                if not success_loading_streetview:
                    img_path = construct_img_path(self.streetview_dir, image_id, is_simulated=False)
                    img_path.unlink()
                    count_element_removed[0] += 1

            if not streetviews_only:
                success_loading_simulated = attempt_loading_simulated(image_id)

                if not success_loading_simulated:
                    img_path = construct_img_path(self.simulated_dir, image_id, is_simulated=True)
                    img_path.unlink()
                    count_element_removed[1] += 1

        count_element_removed = [0, 0]

        deletion_progression = tqdm(self.annotations.itertuples(), desc="deleting unloadable data", total=len(self.annotations))
        for serie in deletion_progression:
            time.sleep(0.2)  # throtlle down
            threading.Thread(
                target=check_image_id,
                args=(serie.image_id, count_element_removed)
            ).start()

        logger.info("dataset - removed %s files because they could not be loaded.", sum(count_element_removed))

    @property
    def passes_channel_nbr(self):
        """return the number of channels for each pass in a simulated image."""
        return self._passes_channel_nbr

    def set_passes_channel_nbr(self):
        """compute the number of channels for each pass in a simulated image."""
        image_id = self.annotations.iloc[0]["image_id"]

        channels_per_pass = get_simulated_image(self.simulated_dir, image_id, self.render_passes, return_nbr_of_channels_per_pass=True)

        self._passes_channel_nbr = channels_per_pass

        self.render_passes = list(channels_per_pass.keys())


def delete_image(street_view_dir: Path, simulated_dir: Path, image_id: Union[str, int]):
    img_path = construct_img_path(simulated_dir, image_id, is_simulated=True)
    img_path.unlink(missing_ok=True)
    img_path = construct_img_path(street_view_dir, image_id, is_simulated=False)
    img_path.unlink(missing_ok=True)


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
        are mixed up, it will create subsets with integers first, and will
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
