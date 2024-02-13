from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.utils import dir_to_img_ids, get_simulated_image


class InferenceDataset(Dataset):
    def __init__(
        self,
        view_folder: Path
    ):
        self.view_folder = view_folder
        self.view_ids = dir_to_img_ids(view_folder)

    def __getitem__(self, idx: int) -> torch.Tensor:
        view_id = self.view_ids[idx]

        sample = get_simulated_image(self.view_folder, view_id, return_nbr_of_channels_per_pass=False)

        return sample

    def __len__(self):
        return len(self.view_ids)
