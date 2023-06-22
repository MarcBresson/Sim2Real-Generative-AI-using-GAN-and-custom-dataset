"""
from https://github.com/timy90022/Perspective-and-Equirectangular/blob/master/lib/Equirec2Perspec.py
"""
from numpy import radians, tan
import numpy as np
import cv2
import torch


class Equirec:
    def __init__(self, img_batch: torch.Tensor, device: str = "cuda:0"):
        """
        Parameters
        ----------
        img_batch : torch.Tensor
            the image batch with shape (N, C, H, W):
                - N is the number of image in the batch
                - C is the number of channels in the images
                - H is the height of the images
                - W is the width of the images
        device : str, optional
            device to make the computation on, by default "cuda:0"
        """
        if len(img_batch.shape) != 4:
            raise ValueError(f"img_batch should be of dimension 4. Found {len(img_batch.shape)}."
                             "in case of single image, you can use `img.unsqueeze(0)`.")
        self.device = device

        self.img_batch = img_batch.to(device)
        self.batch_size = img_batch.shape[0]

        self._width = img_batch.shape[2]
        self._height = img_batch.shape[1]

    def to_persp(self, yaw: float, pitch: float, w_fov: int = 90, aspect_ratio: float = 1) -> torch.Tensor:
        """
        Parameters
        ----------
        yaw : float
            yaw angle, in degree
        pitch : float
            pitch angle, in degree
        w_fov : int, optional
            horizontal Field Of View, by default 90
        aspect_ratio : float, optional
            aspect ratio of the output image, by default 1
        out_tensor : bool, optional
            if True, output a torch Tensor object, by default False
        """
        h_fov = w_fov / aspect_ratio

        lon, lat = self.compute_maps(yaw, pitch, w_fov, h_fov)

        grid = torch.stack((lon / 1024 - 1, lat / 512 - 1), dim=2)
        grid = grid.unsqueeze(0).repeat((self.batch_size, 1, 1, 1))  # we use the same grid for the whole batch

        persp = torch.nn.functional.grid_sample(self.img_batch, grid)

        return persp

    def compute_maps(self, yaw: float, pitch: float, w_fov: int = 90, h_fov: int = 90) -> tuple[torch.Tensor, torch.Tensor]:
        """
        the first matrix indicates where to find the source x coordinate of a (x, y) point.
        the second matrix indicates where to find the source y coordinate of a (x, y) point.

        Parameters
        ----------
        yaw : float
            yaw angle (left/right angle), in degree
        pitch : float
            pitch angle (up/down angle), in degree
        w_fov : int, optional
            horizontal Field Of View, by default 90
        h_fov : int, optional
            vertical Field Of View, by default 90

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            lon matrix and lat matrix
        """
        w_len = tan(radians(w_fov / 2.0))  # =1 for fov=90
        h_len = tan(radians(h_fov / 2.0))

        width = int(self._width / 2 * w_len)
        height = int(self._width / 2 * h_len)

        x_map = torch.ones([height, width], device=self.device)
        y_map = torch.tile(torch.linspace(-w_len, w_len, width, device=self.device), [height, 1])
        z_map = torch.tile(torch.linspace(h_len, -h_len, height, device=self.device), [width, 1]).transpose(0, 1)

        distance: torch.Tensor = torch.sqrt(x_map**2 + y_map**2 + z_map**2)
        distance = distance.unsqueeze(2).repeat((1, 1, 3))

        xyz: torch.Tensor = torch.stack((x_map, y_map, z_map), axis=2) / distance
        xyz = xyz.reshape([height * width, 3]).T

        r1, r2 = rot_matrices(yaw, pitch, self.device)

        xyz = torch.mm(r1, xyz)
        xyz = torch.mm(r2, xyz).T

        lat = torch.arcsin(xyz[:, 2])
        lon = torch.arctan2(xyz[:, 1], xyz[:, 0])

        lon = torch.rad2deg(lon.reshape([height, width]))
        lat = torch.rad2deg(-lat.reshape([height, width]))

        equ_cx = (self._width - 1) / 2
        equ_cy = (self._height - 1) / 2

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        return lon, lat


def rot_matrices(yaw: float, pitch: float, device: str, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """return the two rotation matrices for a yaw and a pitch"""
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)

    r1, _ = cv2.Rodrigues(z_axis * radians(yaw))
    r2, _ = cv2.Rodrigues(np.dot(r1, y_axis) * radians(-pitch))

    if inverse:
        r1 = np.linalg.inv(r1)
        r2 = np.linalg.inv(r2)

    r1 = torch.from_numpy(r1).to(device)
    r2 = torch.from_numpy(r2).to(device)

    return r1, r2
