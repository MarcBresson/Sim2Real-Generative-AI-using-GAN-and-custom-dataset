from torch import Tensor

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


class UnNormalize():
    """
    extends torchvision.transforms.Resize to suits this project
    """
    def __call__(self, imgs: dict[str, Tensor]) -> dict[str, Tensor]:
        imgs = batcher(imgs)
        imgs["streetview"] = imgs["streetview"] * 255
        imgs["simulated"] = imgs["simulated"] * 255

        return imgs
