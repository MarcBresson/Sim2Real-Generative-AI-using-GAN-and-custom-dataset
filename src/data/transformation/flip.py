from typing import Union

import torch
import torchvision
from torch import Tensor

torch.manual_seed(972000)


class RandomHorizontalFlip():
    """
    extends torchvision.transforms.RandomHorizontalFlip to suits this project
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.transform = torchvision.transforms.RandomHorizontalFlip(*args, **kwargs)

    def __call__(self, imgs: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
        if isinstance(imgs, dict):
            imgs["streetview"] = self.transform(imgs["streetview"])
            imgs["simulated"] = self.transform(imgs["simulated"])
        elif isinstance(imgs, Tensor):
            imgs = self.transform(imgs)
        else:
            raise TypeError(f"type {type(imgs)} is not supported. Please use a Tensor or a "
                            "dict with keys `simulated` and `streetview`.")

        return imgs
