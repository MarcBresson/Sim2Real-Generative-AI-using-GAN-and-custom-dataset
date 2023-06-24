import torch
import torchvision
from torch import Tensor

from .batcher import batcher

torch.manual_seed(972000)


class Resize():
    """
    extends torchvision.transforms.Resize to suits this project
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.transform = torchvision.transforms.Resize(*args, **kwargs)

    def __call__(self, imgs: dict[str, Tensor]) -> dict[str, Tensor]:
        imgs = batcher(imgs)
        imgs["streetview"] = self.transform(imgs["streetview"])
        imgs["simulated"] = self.transform(imgs["simulated"])

        return imgs
