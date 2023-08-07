from typing import Union

import torch
from torch import Tensor


class ToDevice():
    """
    extends torchvision.transforms.Resize to suits this project
    """
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, imgs: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
        if isinstance(imgs, dict):
            imgs["streetview"] = imgs["streetview"].to(device=self.device)
            imgs["simulated"] = imgs["simulated"].to(device=self.device)
        elif isinstance(imgs, Tensor):
            imgs = imgs.to(device=self.device)

        return imgs
