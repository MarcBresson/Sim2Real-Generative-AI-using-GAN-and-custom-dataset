import torch
from torch import Tensor

torch.manual_seed(972000)


class Concat():
    def __call__(self, imgs: dict[str, Tensor]) -> Tensor:
        """transform a dict sample or a batched dict sample to a concatenated sample or concatenated batched sample"""
        if len(imgs["simulated"].shape) == 3:
            concat_imgs = torch.concat((imgs["streetview"], imgs["simulated"]), dim=0)

        elif len(imgs["simulated"].shape) == 4:
            concat_imgs = torch.concat((imgs["streetview"], imgs["simulated"]), dim=1)

        else:
            raise ValueError("Couldn't concat the dict into a single sample/batch. "
                             "Ensure that the value of the key `simulated` has 3 or 4 dimensions.")

        return concat_imgs


class unConcat():
    def __call__(self, imgs: Tensor) -> dict[str, Tensor]:
        """transform a batched sample to a batched dict sample"""
        dict_imgs = {}

        if len(imgs.shape) == 3:
            dict_imgs["streetview"] = imgs[:3]
            dict_imgs["simulated"] = imgs[3:]

        elif len(imgs.shape) == 4:
            dict_imgs["streetview"] = imgs[:, :3]
            dict_imgs["simulated"] = imgs[:, 3:]

        else:
            raise ValueError("Couldn't convert the batched sample to a batched dict sample. "
                             "Ensure that the input has 3 or 4 dimensions.")

        return dict_imgs
