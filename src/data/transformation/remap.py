from typing import overload

import numpy as np
from torch import Tensor


class toNumpy:
    def __call__(self, imgs: Tensor) -> np.ndarray:
        # remove autograd function and move to cpu
        imgs = imgs.detach().cpu()

        # invert channels in numpy's order
        if len(imgs.shape) == 3:
            imgs_norm = imgs.permute(1, 2, 0)
        elif len(imgs.shape) == 4:
            imgs_norm = imgs.permute(0, 2, 3, 1)

        imgs_np = imgs_norm.numpy().astype(np.float32)

        return imgs_np


class Remap:
    def __init__(
        self,
        in_min: float | Tensor,
        in_max: float | Tensor,
        out_min: float | Tensor = -1,
        out_max: float | Tensor = 1,
    ) -> None:
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max

    @overload
    def __call__(self, imgs: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @overload
    def __call__(self, imgs: Tensor) -> Tensor: ...

    def __call__(self, imgs):
        scale_factor = (self.out_max - self.out_min) / (self.in_max - self.in_min)

        if isinstance(imgs, dict):
            imgs["streetview"] = (
                imgs["streetview"] - self.in_min
            ) * scale_factor + self.out_min
            imgs["simulated"] = (
                imgs["simulated"] - self.in_min
            ) * scale_factor + self.out_min

        elif isinstance(imgs, Tensor):
            imgs = (imgs - self.in_min) * scale_factor + self.out_min

        else:
            raise TypeError(
                f"type {type(imgs)} is not supported. Please use a Tensor or a "
                "dict with keys `simulated` and `streetview`."
            )

        return imgs
