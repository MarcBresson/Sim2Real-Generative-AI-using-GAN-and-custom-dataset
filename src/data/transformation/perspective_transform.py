from typing import Sequence

import torch
from torch import Tensor

from .equirec2perspective import Equirec
from .batcher import batcher

torch.manual_seed(972000)


class RandomPerspective():
    """
    Randomly pick a perspective view in the 360 image.
    """
    def __init__(
        self,
        yaw: float | tuple[float, float],
        pitch: float | tuple[float, float],
        w_fov: int | tuple[int, int]
    ):
        """
        Parameters
        ----------
        yaw : float
            yaw angle (left/right angle) in degree, or range of allowed
            yaw angles
        pitch : float
            pitch angle (up/down angle) in degree, or range of allowed
            pitch angles
        w_fov : int, optional
            horizontal Field Of View (FOV) angle in degree, or range of
            allowed FOV angles
        """
        assert isinstance(yaw, (int, float, Sequence))
        assert isinstance(pitch, (int, float, Sequence))
        assert isinstance(w_fov, (int, float, Sequence))

        self.yaw = yaw
        self.pitch = pitch
        self.w_fov = w_fov

    def __call__(self, equirec_imgs: dict[str, Tensor]) -> dict[str, Tensor]:
        yaw = self.yaw
        if isinstance(yaw, Sequence):
            yaw = rng_range(self.yaw[0], self.yaw[1])

        pitch = self.pitch
        if isinstance(pitch, Sequence):
            pitch = rng_range(self.pitch[0], self.pitch[1])

        w_fov = self.w_fov
        if isinstance(w_fov, Sequence):
            w_fov = rng_range(self.w_fov[0], self.w_fov[1])

        equirec_imgs = batcher(equirec_imgs)

        equirec_imgs["streetview"] = Equirec(equirec_imgs["streetview"]).to_persp(yaw, pitch, w_fov)
        equirec_imgs["simulated"] = Equirec(equirec_imgs["simulated"]).to_persp(yaw, pitch, w_fov)

        return equirec_imgs


def rng_range(mmin: float, mmax: float):
    rng = torch.rand(1) * (mmax - mmin) + mmin
    return rng.tolist()[0]
