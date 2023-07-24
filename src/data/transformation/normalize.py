from torch import Tensor

import numpy as np

from .batcher import batcher


class NormalizeChannels():
    """
    extends torchvision.transforms.Resize to suits this project
    """
    def __call__(self, imgs: dict[str, Tensor]) -> dict[str, Tensor]:
        imgs = batcher(imgs)
        imgs["streetview"] = imgs["streetview"] / 255
        imgs["simulated"] = imgs["simulated"] / 255

        return imgs


class toNumpy():
    """
    extends torchvision.transforms.Resize to suits this project
    """
    def __call__(self, imgs: Tensor) -> np.ndarray:
        imgs = imgs.detach().cpu()
        imgs_norm = (imgs * 255).permute(0, 2, 3, 1)
        imgs_np = imgs_norm.numpy().astype(np.uint8)
        return imgs_np
