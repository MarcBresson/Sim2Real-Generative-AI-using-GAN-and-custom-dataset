"""
the goal here is to provide the simplest training pipeline to allow
for multiple speed evaluation depending on the batch size.

Every stats will be recorded by pyinstruments and outputted in an
HTML file where the profiling will be in a form of a tree. You will
see the execution time of each functions with their sub functions.
"""

from pathlib import Path

import torch
import yaml
from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data import CustomImageDataset, dataset_split, transformation
from src.models import GAN, init_weights


def train(dataloader_train: DataLoader, batch_transform, model: GAN):
    for _, batch in enumerate(dataloader_train):
        batch = batch_transform(batch)

        model.fit_sample(batch)


def run(
    out_data_path: Path,
    batch_transform,
    dataloader_train: DataLoader,
    model: GAN,
    batch_size: int,
):
    pr = Profiler(interval=1e-2, async_mode="enabled")
    pr.start()

    train(dataloader_train, batch_transform, model)

    pr.stop()

    html = pr.output(HTMLRenderer(show_all=True))
    stats_file = out_data_path / f"profile_stats_batch_size={batch_size}.html"
    stats_file.write_text(html, encoding="utf8")


def load_config(config_name: str, dir: Path = Path("config")):
    filepath = dir / (config_name + ".yml")

    with filepath.open() as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


if __name__ == "__main__":
    cfg = load_config("test")

    in_data_path = Path(cfg["train"]["in_data"])
    annotations_file = in_data_path / Path(r"annotations.arrow")
    streetview_dir = in_data_path / Path(r"mapillary")
    simulated_dir = in_data_path / Path(r"blender_numpy")

    sample_transform = Compose(
        [
            transformation.Sample2Batch(),
            transformation.RandomPerspective(
                yaw=(0, 360), pitch=(-40, 60), w_fov=(60, 120)
            ),
            transformation.Resize((256, 256), antialias=True),
            transformation.To(dtype=cfg["data"]["dtype"]),
            transformation.RandomHorizontalFlip(),
            transformation.Batch2Sample(),
        ]
    )

    dataset = CustomImageDataset(
        annotations_file, streetview_dir, simulated_dir, transform=sample_transform
    )
    dataset_train, _ = dataset_split(dataset, [240, 2])

    out_data_path = Path(cfg["train"]["out_data"])
    out_data_path.mkdir(parents=True, exist_ok=True)

    batch_transform = Compose(
        [
            transformation.To(torch.device("cuda:0")),
        ]
    )

    for batch_size in [1, 1, 2, 4, 8, 12, 16, 20, 24]:
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        model = GAN(
            dtype=cfg["data"]["dtype"],
            input_channels=cfg["data"]["input_channels"],
            output_channels=cfg["data"]["output_channels"],
            generator_config=cfg["network"]["generator"],
            discriminator_config=cfg["network"]["discriminator"],
        )
        init_weights(model, init_type="xavier", gain=0.2)
        run(out_data_path, batch_transform, dataloader_train, model, batch_size)
