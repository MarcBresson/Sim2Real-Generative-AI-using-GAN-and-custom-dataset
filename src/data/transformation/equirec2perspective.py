"""
from https://github.com/timy90022/Perspective-and-Equirectangular/blob/master/lib/Equirec2Perspec.py
"""
from numpy import radians, degrees, tan, arcsin, arctan2, sqrt
import numpy as np
import cv2
import torch


class Equirec:
    def __init__(self, img: np.ndarray | torch.Tensor):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        self.img = img

        self._width = img.shape[1]
        self._height = img.shape[0]

    def to_persp(self, yaw: float, pitch: float, w_fov: int = 90, aspect_ratio: float = 1, out_tensor: bool = False) -> np.ndarray | torch.Tensor:
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

        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)

        # cv2.INTER_CUBIC doesn't work for more than 4 channels
        persp = cv2.remap(self.img, lon, lat, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        if out_tensor:
            persp = torch.Tensor(persp).permute(2, 0, 1)

        return persp

    def compute_maps(self, yaw: float, pitch: float, w_fov: int = 90, h_fov: int = 90) -> tuple[np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray]
            lon matrix and lat matrix
        """
        w_len = tan(radians(w_fov / 2.0))  # =1 for fov=90
        h_len = tan(radians(h_fov / 2.0))

        width = int(self._width / 2 * w_len)
        height = int(self._width / 2 * h_len)

        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
        z_map = np.tile(np.linspace(h_len, -h_len, height), [width, 1]).T

        distance = sqrt(x_map**2 + y_map**2 + z_map**2)
        distance = np.repeat(distance[:, :, np.newaxis], 3, axis=2)

        xyz = np.stack((x_map, y_map, z_map), axis=2) / distance
        xyz = xyz.reshape([height * width, 3]).T

        r1, r2 = rot_matrices(yaw, pitch)

        xyz = np.dot(r1, xyz)
        xyz = np.dot(r2, xyz).T

        lat = arcsin(xyz[:, 2])
        lon = arctan2(xyz[:, 1], xyz[:, 0])

        lon = degrees(lon.reshape([height, width]))
        lat = degrees(-lat.reshape([height, width]))

        equ_cx = (self._width - 1) / 2
        equ_cy = (self._height - 1) / 2

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        return lon, lat


class Perspective:
    def __init__(self, img: np.ndarray, yaw: float, pitch: float, w_fov: int = 90):
        self.img = img

        self._width = img.shape[1]
        self._height = img.shape[0]

        self.yaw = yaw
        self.pitch = pitch

        aspect_ratio = self._width / self._height
        h_fov = w_fov / aspect_ratio

        self.w_len = tan(radians(w_fov / 2.0))
        self.h_len = tan(radians(h_fov / 2.0))

    def to_equirec(self, width: int) -> np.ndarray:
        """
        Parameters
        ----------
        width : int
            width of the output equirectangular image.
        """
        lon, lat = self.compute_maps(width)

        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)

        persp = cv2.remap(self.img, lon, lat, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

    def compute_maps(self, out_width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        the first matrix indicates where to find the source x coordinate of a (x, y) point.
        the second matrix indicates where to find the source y coordinate of a (x, y) point.

        Parameters
        ----------
        yaw : float
            yaw angle (left/right angle), in degree
        pitch : float
            pitch angle (up/down angle), in degree
        out_width : int
            width of the output image.
        w_fov : int, optional
            horizontal Field Of View, by default 90
        h_fov : int, optional
            vertical Field Of View, by default 90

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            lon matrix and lat matrix
        """
        out_height = out_width * 2

        x, y = np.meshgrid(np.linspace(-180, 180, out_width), np.linspace(90, -90, out_height))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map), axis=2)
        xyz = xyz.reshape((out_height * out_width, 3)).T

        r1, r2 = rot_matrices(self.yaw, self.pitch, inverse=True)

        xyz = np.dot(r2, xyz)
        xyz = np.dot(r1, xyz).T

        xyz = xyz.reshape((out_height, out_width, 3))
        xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

        mask = np.where((-self.w_len < xyz[:, :, 1] < self.w_len) & (-self.h_len < xyz[:, :, 2] < self.h_len), 1, 0)

        lon_map = mask * (xyz[:, :, 1] + self.w_len) / 2 / self.w_len * self._width
        lat_map = mask * (-xyz[:, :, 2] + self.h_len) / 2 / self.h_len * self._height

        return lon_map, lat_map


def rot_matrices(yaw: float, pitch: float, inverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """return the two rotation matrices for a yaw and a pitch"""
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)

    r1, _ = cv2.Rodrigues(z_axis * np.radians(yaw))
    r2, _ = cv2.Rodrigues(np.dot(r1, y_axis) * np.radians(-pitch))

    if inverse:
        r1 = np.linalg.inv(r1)
        r2 = np.linalg.inv(r2)

    return r1, r2
