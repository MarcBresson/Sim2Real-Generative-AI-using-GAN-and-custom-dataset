import logging
from collections.abc import MutableSequence
from pathlib import Path

import torch
from ignite.metrics import SSIM
from matplotlib import pyplot as plt
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from config import TrainConfig
from src.data import CustomImageDataset, dataset_split, transformation
from src.data.visualisation import (
    Visualisation,
    batch_to_numpy,
    multichannels_to_individuals,
    plot_sim,
)
from src.logger import Checkpointer, MetricLogger
from src.models import GAN, init_weights

logging.getLogger().setLevel(logging.INFO)


def train(
    dataloader_train: DataLoader,
    batch_transform,
    model: GAN,
    metric_logger: MetricLogger,
):
    for _, batch in enumerate(dataloader_train):
        batch = batch_transform(batch)

        model.fit_sample(batch)

        metric_logger.update_model_losses("train", model.loss_values)
        metric_logger.get("train_generator_ssim").update(
            (model.fake_streetviews, model.real_streetviews)
        )


def validate(
    dataloader_val: DataLoader, batch_transform, model: GAN, metric_logger: MetricLogger
):
    for _, batch in enumerate(dataloader_val):
        batch = batch_transform(batch)

        model.test(batch)

        metric_logger.update_model_losses("validate", model.loss_values)
        metric_logger.get("test_generator_ssim").update(
            (model.fake_streetviews, model.real_streetviews)
        )


def run(
    n_epoch: int,
    batch_transform,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    visualisation: Visualisation,
    model: GAN,
    checkpointer: Checkpointer,
    metric_logger: MetricLogger,
):
    for epoch in tqdm(range(n_epoch)):
        train(dataloader_train, batch_transform, model, metric_logger)
        validate(dataloader_val, batch_transform, model, metric_logger)
        visualisation.visualize(model, epoch)

        checkpointer.step(epoch, "epoch")
        metric_logger.step()


def create_dataloaders(
    dataset: Dataset,
    dataset_split_proportions: MutableSequence[float | int],
    batch_size: int,
):
    dataset_train, dataset_val, dataset_viz = dataset_split(
        dataset, dataset_split_proportions
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    return dataloader_train, dataloader_val, dataset_viz


def create_initial_viz(dataset_viz):
    viz_dir: Path = dataset_viz.out_dir

    for i in range(len(dataset_viz)):
        (viz_dir / f"sample_{i}").mkdir(parents=True, exist_ok=True)

    dataset = dataset_viz.dataset

    for i_sample, idx in enumerate(dataset_viz.indices):
        sample = dataset.get_untransformed_sample(idx)
        sim_img = dataset_viz.transform(sample["simulated"])
        sim_img = batch_to_numpy(sim_img)[0]

        channels = multichannels_to_individuals(sim_img, dataset.passes_channel_nbr)
        save_path = viz_dir / f"sample_{i_sample}" / "real_simulated"
        fig = plot_sim(channels, dataset.render_passes)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()


def get_metric_logger(out_data_path: Path):
    metrics = {
        "train_generator_ssim": SSIM(1.0, device="cuda:0"),
        "test_generator_ssim": SSIM(1.0, device="cuda:0"),
    }
    metric_logger = MetricLogger(
        out_data_path / "metrics.log", metrics, torch.device("cuda:0")
    )

    return metric_logger


def get_dataset(cfg: TrainConfig) -> CustomImageDataset:
    input_size = cfg.network.generator.input_size
    sample_transform = Compose(
        [
            transformation.Sample2Batch(),
            transformation.RandomPerspective(
                yaw=(0, 360), pitch=(0, 60), w_fov=(60, 120)
            ),
            transformation.Resize((input_size, input_size), antialias=True),
            transformation.RandomHorizontalFlip(),
            transformation.To(dtype=cfg.data.dtype),
            transformation.Batch2Sample(),
        ]
    )

    annotations_file = cfg.train.in_data_path / "annotations.arrow"
    streetview_dir = cfg.train.in_data_path / "mapillary"
    simulated_dir = cfg.train.in_data_path / "blender_numpy"

    dataset = CustomImageDataset(
        annotations_file, streetview_dir, simulated_dir, transform=sample_transform
    )

    return dataset


def main():
    cfg = TrainConfig.load("train")

    dataset = get_dataset(cfg)

    split_with_viz = cfg.train.dataset_split + [cfg.train.visualisation.n_samples]
    dataset_train, dataset_val, dataset_viz = dataset_split(dataset, split_with_viz)

    dataloader_train = DataLoader(dataset_train, **cfg.train.dataloader.model_dump())
    dataloader_val = DataLoader(dataset_val, **cfg.train.dataloader.model_dump())

    input_size = cfg.network.generator.input_size
    viz_transform = Compose(
        [
            transformation.Sample2Batch(),
            transformation.RandomPerspective(yaw=0, pitch=30, w_fov=100),
            transformation.To(dtype=cfg.data.dtype),
            transformation.Resize((input_size, input_size), antialias=True),
            transformation.To(torch.device("cuda:0")),
        ]
    )

    visualisation = Visualisation(
        dataset_viz, viz_transform, cfg.train.out_data_path / "visualisation"
    )

    create_initial_viz(dataset_viz)

    model = GAN(
        dtype=cfg.data.dtype,
        input_channels=cfg.data.input_channels,
        output_channels=cfg.data.output_channels,
        generator_kwargs=dict(cfg.network.generator),
        discriminator_kwargs=dict(cfg.network.discriminator),
    )
    init_weights(model, init_type="xavier", gain=0.2)

    checkpointer = Checkpointer(
        model, cfg.train.out_data_path / "checkpoints", cfg.train.checkpointer.period
    )

    metric_logger = get_metric_logger(cfg.train.out_data_path)

    batch_transform = Compose(
        [
            transformation.To(torch.device("cuda:0")),
        ]
    )

    if cfg.train.profile:
        pr = Profiler(interval=1e-2, async_mode="enabled")
        pr.start()

    run(
        cfg.train.n_epochs,
        batch_transform,
        dataloader_train,
        dataloader_val,
        visualisation,
        model,
        checkpointer,
        metric_logger,
    )

    if cfg.train.profile:
        pr.stop()

        html = pr.output(HTMLRenderer(show_all=True))
        stats_file = cfg.train.out_data_path / "profile_stats.html"
        stats_file.write_text(html, encoding="utf8")


if __name__ == "__main__":
    main()
