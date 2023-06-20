from pathlib import Path
import threading

from tqdm import tqdm
import requests
import pandas
import mercantile
from vt2geojson.tools import vt_bytes_to_geojson


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
    data.append(image_data["computed_altitude"])
    data.append(image_data["computed_compass_angle"])
    # lon, lat
    data.extend(image_data["computed_geometry"]["coordinates"])
    # rot z, rot y, rot x. See https://opensfm.org/docs/cam_coord_system.html
    data.extend(image_data["computed_rotation"])

    return data


def download_image_data(
    data_set: list,
    image_id: int,
    fields_to_dl: list[str],
    headers: dict,
    image_dir: Path
):
    """launch a request to query data for an image"""
    url = f'https://graph.mapillary.com/{image_id}?fields={",".join(fields_to_dl)}'
    response = requests.get(url, headers=headers)
    image_data = response.json()

    if is_camera_360(image_data):
        img_data = process_img_data(image_id, image_data)
        data_set.append(img_data)

        image_url = image_data['thumb_2048_url']
        download_image(image_dir, image_id, image_url)


def download_image(image_dir: Path, image_id: int, image_url: str):
    """launch a request to download an image"""
    image_path = image_dir / f"{image_id}.jpg"

    with image_path.open("wb") as handler:
        img = requests.get(image_url, stream=True).content
        handler.write(img)


def inside_bbox(lon_lat, west_south_east_north) -> bool:
    """ensure lon and lat are inside the bounding box since tiles
    can expand beyond"""
    lon, lat = lon_lat
    west, south, east, north = west_south_east_north
    return west < lon < east and south < lat < north


def image_exists(image_id: int, dataset: pandas.DataFrame):
    """return True if the image is in the dataset"""
    return image_id in dataset["image_id"].values


def main(access_token, data_dir: Path, dataset: list):
    fields = ["thumb_2048_url", "camera_type", "captured_at", "computed_altitude", "computed_compass_angle", "computed_geometry", "computed_rotation"]
    headers = {"Authorization": f"OAuth {access_token}"}

    image_dir = data_dir / "mapillary2"
    annotations_file = data_dir / "annotations.arrow"

    dataset_pd = pandas.read_feather(annotations_file)

    try:
        total_tiles = sum(1 for _ in mercantile.tiles(west, south, east, north, 14))

        tiles_bar = tqdm(
            mercantile.tiles(west, south, east, north, 14),
            total=total_tiles
        )
        for tile in tiles_bar:
            tiles_bar.set_postfix(tile_z=tile.z, tile_x=tile.x, tile_y=tile.y)

            tile_url = f'https://tiles.mapillary.com/maps/vtp/mly1_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}'
            response = requests.get(tile_url, headers=headers)
            tile_data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer="image")

            for feature in tile_data['features']:
                lon = feature['geometry']['coordinates'][0]
                lat = feature['geometry']['coordinates'][1]

                image_id = feature['properties']['id']

                if image_exists(image_id, dataset_pd):
                    continue

                if inside_bbox((lon, lat), (west, south, east, north)):
                    threading.Thread(
                        target=download_image_data,
                        args=(dataset, feature['properties']['id'], fields, headers, image_dir)
                    ).start()
    finally:
        new_dataset_pd = pandas.DataFrame(dataset, columns=["image_id", "camera_type", "capture_at", "computed_altitude", "computed_compass_angle", "lon", "lat", "rot x", "rot y", "rot z"])

        dataset_pd = pandas.concat([new_dataset_pd, dataset_pd], ignore_index=True)

        dataset_pd.to_feather(annotations_file)
