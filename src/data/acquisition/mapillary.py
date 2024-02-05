from pathlib import Path
import threading

from tqdm import tqdm
import requests
import pandas
import mercantile
from vt2geojson.tools import vt_bytes_to_geojson

from src.data.acquisition.utils import download_image


north =            48.9219
west, east = 2.1976,     2.4699
south =            48.8072


def is_camera_360(img_data: dict):
    return (
        "camera_type" in img_data
        and img_data["camera_type"] in ["equirectangular", "spherical"]
    )


def process_img_data(image_id: int, image_data: dict):
    data = [image_id]
    data.append(image_data["camera_type"])
    data.append(image_data["captured_at"])
    data.append(image_data["computed_compass_angle"])

    data.append(image_data["computed_altitude"])
    # lon, lat
    data.extend(image_data["computed_geometry"]["coordinates"])
    # rot z, rot y, rot x. See https://opensfm.org/docs/cam_coord_system.html
    data.extend(image_data["computed_rotation"])

    data.append(image_data["thumb_2048_url"])

    return data


def download_image_data(
    data_set: list,
    image_id: int,
    fields_to_dl: set[str],
    headers: dict,
    image_dir: Path
):
    """launch a request to query data for an image"""
    url = f'https://graph.mapillary.com/{image_id}?fields={",".join(fields_to_dl)}'
    response = requests.get(url, headers=headers)
    image_data = response.json()

    if is_camera_360(image_data) and "computed_compass_angle" in image_data:
        img_data = process_img_data(image_id, image_data)
        data_set.append(img_data)

        image_url = image_data['thumb_2048_url']
        download_image(image_dir, image_id, image_url)


def is_inside_bbox(lon_lat, region_nesw) -> bool:
    """ensure lon and lat are inside the bounding box since tiles
    can expand beyond"""
    lon, lat = lon_lat
    north, east, south, west = region_nesw
    return west < lon < east and south < lat < north


def image_already_exists(image_id: int, dataset: pandas.DataFrame):
    """return True if the image is in the dataset"""
    if dataset.empty:
        return False
    return image_id in dataset["image_id"].values


def handle_tile(
    access_token: str,
    image_dir: Path,
    existing_dataset: pandas.DataFrame,
    dataset: list,
    tile: mercantile.Tile,
    region_nesw: tuple[float, float, float, float],
    fields: set[str]
):
    """
    handles the download of images in a tile.

    Parameters
    ----------
    access_token : str
        Mapillary's access token.
    image_dir : Path
        the directory where to save the images.
    existing_dataset : pandas.DataFrame
        the already existing dataframe (can be an empty
        dataframe if none existed beforehand).
    dataset : list
        the current dataset to grow with the new data.
    tile : mercantile.Tile
        the tile to download features in.
    region_nesw: tuple[float, float, float, float]
        the region to download street views in with North, Est,
        South, and West coordinates.
    fields : set[str]
        the fields to download and save in the growing dataframe.
    """
    headers = {"Authorization": f"OAuth {access_token}"}

    tile_url = f'https://tiles.mapillary.com/maps/vtp/mly1_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}'
    response = requests.get(tile_url, headers=headers, timeout=20)
    tile_data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer="image")

    for feature in tile_data['features']:
        feature_lon = feature['geometry']['coordinates'][0]
        feature_lat = feature['geometry']['coordinates'][1]

        image_id = feature['properties']['id']

        if feature['properties']['is_pano'] is False:
            continue

        if image_already_exists(image_id, existing_dataset):
            continue

        if is_inside_bbox((feature_lon, feature_lat), region_nesw):
            threading.Thread(
                target=download_image_data,
                args=(dataset, feature['properties']['id'], fields, headers, image_dir)
            ).start()


def main(
    access_token: str,
    data_dir: Path,
    region_nesw: tuple[float, float, float, float]
):
    """
    download mapillary's street views

    Parameters
    ----------
    access_token : str
        mapillary access token. See more at
        https://www.mapillary.com/dashboard/developers
    data_dir : Path
        the data directory where the mapillary street views will be saved
        in the mapillary folder.
    region_nesw: tuple[float, float, float, float]
        the region to download street views in with North, Est,
        South, and West coordinates.
    """
    fields = set(["thumb_2048_url", "camera_type", "captured_at", "computed_altitude", "computed_compass_angle", "computed_geometry", "computed_rotation"])

    image_dir = data_dir / "mapillary"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = data_dir / "annotations.csv"

    # if a dataset already exists, we load it to avoid redownloading the same
    # street views and allow for step by step loading.
    if annotations_file.exists():
        existing_dataset = pandas.read_feather(annotations_file)
    else:
        existing_dataset = pandas.DataFrame()

    # we store everything in an array to avoid growing a pd dataframe.
    dataset = []

    # this try statement is just to allow for the dataset to be saved in case
    # of unknown error.
    try:
        total_tiles = sum(1 for _ in mercantile.tiles(west, south, east, north, 14))

        tiles_progress_bar = tqdm(
            mercantile.tiles(west, south, east, north, 14),
            total=total_tiles
        )
        for tile in tiles_progress_bar:
            # gives more info in the progress bar.
            tiles_progress_bar.set_postfix(tile_z=tile.z, tile_x=tile.x, tile_y=tile.y)

            handle_tile(access_token, image_dir, existing_dataset, dataset, tile, region_nesw, fields)

    finally:
        print(f"{len(dataset)} new samples")
        new_dataset_pd = pandas.DataFrame(dataset, columns=["image_id", "camera_type", "capture_at", "computed_altitude", "computed_compass_angle", "lon", "lat", "rot x", "rot y", "rot z", "thumb_2048_url"])

        if annotations_file.exists():
            existing_dataset = pandas.concat([new_dataset_pd, existing_dataset], ignore_index=True)
        else:
            existing_dataset = new_dataset_pd

        existing_dataset.to_csv(annotations_file)
