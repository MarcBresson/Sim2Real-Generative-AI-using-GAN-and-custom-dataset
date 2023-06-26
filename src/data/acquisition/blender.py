from pathlib import Path
from numpy import deg2rad
import pandas as pd
import bpy
import numpy as np
from math import radians, sin, cos, log, atan, tan


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
        yield row[["image_id", "lon", "lat", "computed_altitude", "computed_compass_angle"]]


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


def mapillary_to_euler(compass_orientation: float) -> tuple[float, float, float]:
    euler_rotation = [90, 0, -compass_orientation]

    return deg2rad(euler_rotation)


def create_nodes(tmp_dir: Path, render_pass_names: list[str], render_pass_options: list[dict]) -> list[str]:
    """
    create the node tree to save each pass in a different folder.

    Parameters
    ----------
    render_pass_names : list[str]
        list of the render passes to save

    Returns
    -------
    list[str]
        path to where the passes are saved.
    """
    tree = bpy.context.scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    renderlayers_node = tree.nodes.new("CompositorNodeRLayers")
    renderlayers_node.location = (0, 0)
    renderlayers_node.name = "RenderLayers"

    save_dirs = [""] * len(render_pass_names)

    for i, render_pass_name in enumerate(render_pass_names):
        save_dirs[i] = tmp_dir / render_pass_name

        render_pass = renderlayers_node.outputs[render_pass_name]

        if "use_viewer" in render_pass_options[i]:
            viewer_node = tree.nodes.new("CompositorNodeViewer")
            output_node = viewer_node
        else:
            for key, value in render_pass_options[i].items():
                setattr(bpy.context.scene.render.image_settings, key, value)
            fileoutput_node = tree.nodes.new("CompositorNodeOutputFile")
            fileoutput_node.location = (400, -120 * i)
            fileoutput_node.name = render_pass_name

            fileoutput_node.base_path = str(save_dirs[i])
            output_node = fileoutput_node

        tree.links.new(render_pass, output_node.inputs[0])

    return save_dirs


def place_camera(
    xyz: tuple[float, float, float],
    computed_compass_angle: float,
    offset_altitude: float = 0.3
):
    """
    Parameters
    ----------
    xyz : tuple[float, float, float]
        in meters
    computed_compass_angle : float
        in degree
    offset_altitude : float
        move the z point up or down to account for streets not being at 0
    """
    camera = bpy.context.scene.camera

    x, y, z = xyz
    camera.location = (x, y, z + offset_altitude)
    camera.rotation_euler = mapillary_to_euler(computed_compass_angle)


def move_render_passes(save_dirs: list[Path], output_dir: Path):
    image_id = output_dir.stem

    for dir_ in save_dirs:
        pass_img_path: Path = list(dir_.glob("*.*"))[0]
        pass_name = pass_img_path.parent.stem
        pass_ext = pass_img_path.suffix

        new_filepath = output_dir.with_stem(f"{image_id}_{pass_name}").with_suffix(pass_ext)
        new_filepath.unlink(missing_ok=True)

        pass_img_path.rename(new_filepath)


def save_viewernode(save_dir: Path):
    viewer = np.array(bpy.data.images["Viewer Node"].pixels)
    viewer = viewer[0::4].reshape((1024, 2048))  # remove duplicated channels
    viewer = viewer[::-1, :]  # flip horizontally
    viewer = viewer * (viewer < 1.0e+10)  # remove the sky
    viewer = viewer.astype(np.float16)
    np.save(save_dir / "raw", viewer)


def image_exists(search_dir: Path, prefix: str, expected_length: int = 1):
    search_results = list(search_dir.glob(f"{prefix}*"))

    return len(search_results) == expected_length


def main():
    render_pass_names = ["Depth", "Normal", "DiffCol"]
    render_pass_options = [{"use_viewer": True}, {"file_format": "PNG", "color_depth": "8", "color_mode": "RGB"}, {"file_format": "PNG", "color_depth": "8", "color_mode": "RGB"}]

    root_dir = Path(r"C:\Users\marco\Documents\Cours\Individual Research Project - IRP\code\data")

    tmp_dir = root_dir / "tmp"
    dataset_path = root_dir / "dataset.arrow"
    savedir = root_dir / "blender"

    create_camera()
    save_dirs = create_nodes(tmp_dir, render_pass_names, render_pass_options)

    scene = bpy.context.scene
    projection = TransverseMercator(lat=scene["lat"], lon=scene["lon"])

    for streetview_data in read_data(dataset_path):
        image_id, lon, lat, alt, computed_compass_angle = streetview_data

        if image_exists(savedir, str(image_id) + "_", len(render_pass_names)):
            continue

        x, y = projection.from_geographic(lat, lon)
        place_camera((x, y, alt), computed_compass_angle)

        bpy.ops.render.render(write_still=True)

        save_viewernode(save_dirs[0])
        move_render_passes(save_dirs, savedir / str(image_id))


main()
