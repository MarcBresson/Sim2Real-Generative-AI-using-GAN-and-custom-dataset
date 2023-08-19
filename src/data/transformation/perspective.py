"""
from https://github.com/timy90022/Perspective-and-Equirectangular/blob/master/lib/Equirec2Perspec.py
"""
from typing import Sequence, Union
import logging

from numpy import radians, tan
import torch
from torch import Tensor

torch.manual_seed(972000)


class RandomPerspective():
    """
    Randomly pick a perspective view in the 360 image.
    """
    def __init__(
        self,
        yaw: Union[float, tuple[float, float]],
        pitch: Union[float, tuple[float, float]],
        w_fov: Union[int, tuple[int, int]],
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

        self.last_yaw = None
        self.last_pitch = None
        self.last_w_fov = None

    def __call__(
            self,
            equirec_imgs: Union[Tensor, dict[str, Tensor]],
            *,
            max_retry: int = 5,
            retry: int = 0
    ) -> dict[str, Tensor]:
        yaw, pitch, w_fov = self.pick_parameters()
        self.last_yaw = yaw
        self.last_pitch = pitch
        self.last_w_fov = w_fov

        if isinstance(equirec_imgs, dict):
            persp_imgs = self.transform(equirec_imgs, yaw, pitch, w_fov)

        elif isinstance(equirec_imgs, Tensor):
            persp_imgs = self.transform_concatenated(equirec_imgs, yaw, pitch, w_fov)

        else:
            raise TypeError(f"type {type(equirec_imgs)} is not supported. Please use a Tensor or a "
                            "dict with keys `simulated` and `streetview`.")

        if retry < max_retry and self.persp_has_nan(persp_imgs):
            retry += 1
            self(equirec_imgs, max_retry=max_retry, retry=retry)

        if retry == max_retry:
            logging.debug("Could not compute a non-corrupted perspective view.")
            return None

        if retry > 0:
            print(retry)
            logging.debug("Had to retry %s times to compute a perspective view.", retry)

        return persp_imgs

    def pick_parameters(self) -> tuple[float, float, float]:
        """
        pick a set of yaw, pitch and horizontal FOV that are in the
        range given or that corresponds to the number given.

        Returns
        -------
        tuple[float, float, float]
            yaw, pitch, w_fov
        """
        yaw = self.yaw
        if isinstance(yaw, Sequence):
            yaw = rng_range(self.yaw[0], self.yaw[1])

        pitch = self.pitch
        if isinstance(pitch, Sequence):
            pitch = rng_range(self.pitch[0], self.pitch[1])

        w_fov = self.w_fov
        if isinstance(w_fov, Sequence):
            w_fov = rng_range(self.w_fov[0], self.w_fov[1])

        return yaw, pitch, w_fov

    def transform_concatenated(self, concat_imgs: Tensor, yaw: float, pitch: float, w_fov: float):
        """given already concatenated images, transform them."""
        persp_imgs = Equirec(concat_imgs).to_persp(yaw, pitch, w_fov)

        return persp_imgs

    def transform(self, equirec_imgs: dict[str, Tensor], yaw: float, pitch: float, w_fov: float) -> dict[str, Tensor]:
        """transform a raw batch"""
        persp_imgs = {}
        persp_imgs["streetview"] = Equirec(equirec_imgs["streetview"]).to_persp(yaw, pitch, w_fov)
        persp_imgs["simulated"] = Equirec(equirec_imgs["simulated"]).to_persp(yaw, pitch, w_fov)

        return equirec_imgs

    def persp_has_nan(self, persp_imgs: Union[Tensor, dict[str, Tensor]]) -> bool:
        """
        check if the transformed perspective images has NaN values.

        Parameters
        ----------
        persp_imgs : Union[Tensor, dict[str, Tensor]]
            the transformed perspective images.

        Returns
        -------
        bool
            whether there are NaN values in the transformed perspective images.
        """
        if isinstance(persp_imgs, dict):
            return persp_imgs["simulated"].isnan().any() or persp_imgs["streetview"].isnan().any()

        elif isinstance(persp_imgs, Tensor):
            return persp_imgs.isnan().any()

        raise TypeError("Unknown error while checking for NaN values. The given type "
                        f"{type(persp_imgs)} was not suppose to be given.")

    def __str__(self) -> str:
        s = "("
        if isinstance(self.yaw, Sequence):
            s += f"yaw in [{self.yaw[0]}; {self.yaw[1]}] "
        else:
            s += f"yaw={self.yaw} "

        if isinstance(self.pitch, Sequence):
            s += f"pitch in [{self.pitch[0]}; {self.pitch[1]}] "
        else:
            s += f"pitch={self.pitch} "

        if isinstance(self.w_fov, Sequence):
            s += f"w_fov in [{self.w_fov[0]}; {self.w_fov[1]}]"
        else:
            s += f"w_fov={self.w_fov}"

        s += ")"

        if self.last_pitch is not None:
            s += "; last transformation: ("
            s += f"yaw={self.last_yaw}"
            s += f"pitch={self.last_pitch}"
            s += f"w_fov={self.last_w_fov}"
            s += ")"

        return s


def rng_range(mmin: float, mmax: float):
    rng = torch.rand(1) * (mmax - mmin) + mmin
    return rng.tolist()[0]


class Equirec:
    def __init__(self, img_batch: torch.Tensor):
        """
        Parameters
        ----------
        img_batch : torch.Tensor
            the image batch with shape (N, C, H, W):
                - N is the number of image in the batch
                - C is the number of channels in the images
                - H is the height of the images
                - W is the width of the images
        """
        if len(img_batch.shape) != 4:
            raise ValueError(f"img_batch should be of dimension 4. Found {len(img_batch.shape)}."
                             "in case of single image, you can use `img.unsqueeze(0)`.")
        self.device = img_batch.device

        self.img_batch = img_batch.to(self.device)
        self.batch_size = img_batch.shape[0]

        self._width = img_batch.shape[3]
        self._height = img_batch.shape[2]

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

        persp = torch.nn.functional.grid_sample(self.img_batch, grid, align_corners=False)

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

        width = int(self._height * w_fov / 180)
        height = int(self._height * h_fov / 180)

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


# from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L461

def rot_matrices(yaw: float, pitch: float, device: str, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """return the two rotation matrices for a yaw and a pitch"""
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, radians(yaw)], dtype=torch.float32, device=device)

    r1 = axis_angle_to_matrix(z_axis)
    r2 = axis_angle_to_matrix((r1 * y_axis).sum(-1) * radians(-pitch))

    if inverse:
        r1 = torch.linalg.inv(r1)
        r2 = torch.linalg.inv(r2)

    return r1, r2


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions
