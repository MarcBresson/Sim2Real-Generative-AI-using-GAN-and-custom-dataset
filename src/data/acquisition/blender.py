"""
Install OpenEXR on windows:
    https://github.com/AcademySoftwareFoundation/openexr
    ensure that it is in your program files ("C:\Program Files (x86)\OpenEXR\bin")
    if not, change the value of line 13 accordingly

Install python modules:
    https://stackoverflow.com/questions/11161901/how-to-install-python-modules-in-blender
    (installing the pip package of openexr will install Imath too)
"""

from pathlib import Path
from math import radians, sin, cos, log, atan, tan, pi
import os
os.add_dll_directory(r"C:\Program Files (x86)\OpenEXR\bin")

import OpenEXR
import Imath
import mathutils
import pandas as pd
import bpy
import numpy as np


class TransverseMercator:
    """
    see conversion formulas at
    http://en.wikipedia.org/wiki/Transverse_Mercator_projection
    and
    http://mathworld.wolfram.com/MercatorProjection.html
    """
    radius = 6378137

    def __init__(self, lat: float = 0, lon: float = 0, scale: float = 1):
        self.lat = radians(lat)
        self.lon = radians(lon)
        self.scale = scale

    def from_geographic(self, lat: float, lon: float):
        lat = radians(lat)
        lon = radians(lon) - self.lon

        B = sin(lon) * cos(lat)

        x = 0.5 * self.scale * self.radius * log((1 + B) / (1 - B))
        y = self.scale * self.radius * (atan(tan(lat) / cos(lon)) - self.lat)

        return x, y


def read_data(dataset: Path):
    data = pd.read_feather(dataset)

    for _, row in data.iterrows():
        data = row[["image_id", "lon", "lat", "computed_altitude"]]
        angle_axis_rot = row[["rot_x", "rot_y", "rot_z"]]

        data.append(angle_axis_rot)
        yield data


def create_camera(
    width: int = 2048,
    aspect_ratio: float = 2
):
    # clear any existing camera
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    bpy.context.scene.render.engine = 'CYCLES'

    bpy.ops.object.camera_add(
        location=(0, 0, 0),
        rotation=(0, 0, 0)
    )
    camera = bpy.context.object

    camera.data.type = 'PANO'
    camera.data.cycles.panorama_type = 'EQUIRECTANGULAR'
    camera.data.cycles.panorama_resolution = width

    bpy.context.view_layer.objects.active = camera
    bpy.context.scene.camera = camera

    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = int(width / aspect_ratio)

    bpy.context.scene.render.image_settings.file_format = 'JPEG'


def mapillary_to_euler(angle_axis_rot: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    convert the mapillary rotation vector to a blender rotation vector

    Parameters
    ----------
    angle_axis_rot : tuple[float, float, float]
        angle axis rotation vector from mapillary

    Returns
    -------
    tuple[float, float, float]
        eularian rotation vector for blender
    """
    e = angle_axis_rot / np.linalg.norm(angle_axis_rot)
    teta = np.linalg.norm(angle_axis_rot)

    axis_angle_rot = mathutils.Matrix.Rotation(-teta, 3, e)

    # this is to correctly align axis in blender according to https://opensfm.org/docs/cam_coord_system.html
    camera_rotation = mathutils.Euler([0, pi, pi])

    camera_rotation.rotate(axis_angle_rot)

    return camera_rotation


def place_camera(
    xyz: tuple[float, float, float],
    angle_axis_rot: tuple[float, float, float],
    offset_altitude: float = 0.3
):
    """
    Parameters
    ----------
    xyz : tuple[float, float, float]
        in meters
    angle_axis_rot : tuple[float, float, float]
        in degree
    offset_altitude : float
        move the z point up or down to account for streets not being at 0
    """
    camera = bpy.context.scene.camera

    x, y, z = xyz
    camera.location = (x, y, z + offset_altitude)
    camera.rotation_euler = mapillary_to_euler(angle_axis_rot)


def exr_to_numpy(filepath, passdata: dict[str, dict]):
    exrfile = OpenEXR.InputFile(filepath)

    displaywindow = exrfile.header()['displayWindow']
    height = displaywindow.max.y + 1 - displaywindow.min.y
    width = displaywindow.max.x + 1 - displaywindow.min.x

    pass_arrays = {}
    for passname, pass_data in passdata.items():
        channel_names = [f"{passname}.{channel}" for channel in pass_data["channels"]]

        channels_raw: list[bytes]
        channels_raw = exrfile.channels(channel_names, Imath.PixelType(Imath.PixelType.FLOAT))

        channels = []
        for channel in channels_raw:
            # the type conversion must happen after the frombuffer loading in float32
            channel_values = np.frombuffer(channel).astype(pass_data["dtype"])

            channel_values = np.reshape(channel_values, (height, width, -1))

            channels.append(channel_values)

        full_pass = np.concatenate(channels, 2)

        pass_arrays[passname] = full_pass

    return pass_arrays



def image_exists(search_dir: Path, prefix: str, expected_length: int = 1):
    search_results = list(search_dir.glob(f"{prefix}*"))

    return len(search_results) == expected_length


def main():
    render_passes = {"Depth": {"channels": "V", "dtype": "float16"}, "Normal": {"channels": "XYZ", "dtype": "float16"}, "DiffCol": {"channels": "RGB", "dtype": "uint8"}}

    root_dir = Path(r"C:\Users\marco\Documents\Cours\Individual Research Project - IRP\code\data")

    dataset_path = root_dir / "annotations.arrow"
    tmpdir = root_dir / "tmp"
    savedir = root_dir / "ttt"
    savedir.mkdir(exist_ok=True)

    create_camera()

    scene = bpy.context.scene
    projection = TransverseMercator(lat=scene["lat"], lon=scene["lon"])

    for streetview_data in read_data(dataset_path):
        image_id, lon, lat, alt, angle_axis_rot = streetview_data

        if image_exists(savedir, str(image_id) + "_", len(render_passes)):
            continue

        x, y = projection.from_geographic(lat, lon)
        place_camera((x, y, alt), angle_axis_rot)

        bpy.ops.render.render(write_still=True)

        pass_arrays = exr_to_numpy(tmpdir / "tmp0001.exr", render_passes)

        np.savez_compressed(savedir / str(image_id), **pass_arrays)

        break


main()
