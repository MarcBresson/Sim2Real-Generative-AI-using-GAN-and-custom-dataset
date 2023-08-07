from typing import Union

from torch import Tensor
import numpy as np


class NormalizeChannels():
    def __init__(self) -> None:
        self.remap = Remap(0, 255)

    def __call__(self, imgs: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
        if isinstance(imgs, dict):
            imgs["streetview"] = self.remap(imgs["streetview"])
            imgs["simulated"] = self.remap(imgs["simulated"])
        elif isinstance(imgs, Tensor):
            imgs = self.remap(imgs)

        return imgs


class toNumpy():
    def __init__(self) -> None:
        self.remap = Remap(-1, 1, 0, 255)

    def __call__(self, imgs: Tensor) -> np.ndarray:
        imgs = self.remap(imgs)

        # remove autograd function and move to cpu
        imgs = imgs.detach().cpu()

        # invert channels in numpy's order
        if len(imgs.shape) == 3:
            imgs_norm = imgs.permute(1, 2, 0)
        elif len(imgs.shape) == 4:
            imgs_norm = imgs.permute(0, 2, 3, 1)

        # convert to numpy and to uint8
        imgs_np = imgs_norm.numpy().astype(np.uint8)

        return imgs_np


class Remap():
    def __init__(self, in_min: float, in_max: float, out_min: float = -1, out_max: float = 1) -> None:
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, imgs: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
        scale_factor = (self.out_max - self.out_min) / (self.in_max - self.in_min)

        if isinstance(imgs, dict):
            imgs["streetview"] = (imgs["streetview"] - self.in_min) * scale_factor + self.out_min
            imgs["simulated"] = (imgs["simulated"] - self.in_min) * scale_factor + self.out_min
        elif isinstance(imgs, Tensor):
            imgs = (imgs - self.in_min) * scale_factor + self.out_min

        return imgs
