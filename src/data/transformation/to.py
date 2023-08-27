from typing import Union

import torch
from torch import Tensor

from src import get_device


class To():
    """
    extends torchvision.transforms.Resize to suits this project
    """
    def __init__(
        self,
        device: Union[torch.device, str, None] = None,
        dtype: Union[torch.dtype, str, None] = None
    ) -> None:
        if isinstance(device, str):
            device = get_device(device)
        self.device = device

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype

    def __call__(
        self,
        imgs: Union[dict[str, Tensor], Tensor]
    ) -> Union[dict[str, Tensor], Tensor]:
        if isinstance(imgs, dict):
            imgs["streetview"] = imgs["streetview"].to(device=self.device, dtype=self.dtype)
            imgs["simulated"] = imgs["simulated"].to(device=self.device, dtype=self.dtype)

        elif isinstance(imgs, Tensor):
            imgs = imgs.to(device=self.device, dtype=self.dtype)

        else:
            raise TypeError(f"type {type(imgs)} is not supported. Please use a Tensor or a "
                            "dict with keys `simulated` and `streetview`.")

        return imgs
