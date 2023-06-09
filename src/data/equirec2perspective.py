"""
from https://github.com/timy90022/Perspective-and-Equirectangular/blob/master/lib/Equirec2Perspec.py
"""
from numpy import radians, degrees, tan, arcsin, arctan2, sqrt
import numpy as np
import cv2


class Equirec:
    def __init__(self, img: np.ndarray):
        self.img = img

        self._width = img.shape[1]
        self._height = img.shape[0]

    def to_persp(self, yaw: float, pitch: float, w_fov: int = 90, aspect_ratio: float = 1) -> np.ndarray:
        """
        _summary_

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

        Returns
        -------
        torch.Tensor
            _description_
        """
        h_fov = w_fov / aspect_ratio

        lon, lat = self.compute_maps(yaw, pitch, w_fov, h_fov)

        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)

        persp = cv2.remap(self.img, lon, lat, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

    def compute_maps(self, yaw: float, pitch: float, w_fov: int = 90, h_fov: int = 90) -> tuple[np.ndarray, np.ndarray]:
        """
        the first matrix indicates where to find the source x coordinate of a (x, y) point.
        the second matrix indicates where to find the source y coordinate of a (x, y) point.

        Parameters
        ----------
        yaw : float
            yaw angle, in degree
        pitch : float
            pitch angle, in degree
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
        z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

        d = sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(d[:, :, np.newaxis], 3, axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)

        r1, _ = cv2.Rodrigues(z_axis * np.radians(yaw))
        r2, _ = cv2.Rodrigues(np.dot(r1, y_axis) * np.radians(-pitch))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(r1, xyz)
        xyz = np.dot(r2, xyz).T
        lat = arcsin(xyz[:, 2])
        lon = arctan2(xyz[:, 1], xyz[:, 0])

        lon = degrees(lon.reshape([height, width]))
        lat = degrees(-lat.reshape([height, width]))

        equ_cx = (self._width - 1) / 2.0
        equ_cy = (self._height - 1) / 2.0

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        return lon, lat
