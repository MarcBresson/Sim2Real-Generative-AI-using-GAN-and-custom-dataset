import logging
from pathlib import Path
from typing import MutableSequence

from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose
from ignite.metrics import SSIM
from matplotlib import pyplot as plt

from src.logger import Checkpointer, MetricLogger
from src.models import GAN, init_weights
from src.data import CustomImageDataset, dataset_split
from src.data import transformation
from src.data.visualisation import batch_to_numpy, plot_streetview_with_discrimination, plot_sim, multichannels_to_individuals
from config import TrainConfig

logging.getLogger().setLevel(logging.INFO)


def train(dataloader_train: DataLoader, batch_transform, model: GAN, metric_logger: MetricLogger):
    for _, batch in enumerate(dataloader_train):
        batch = batch_transform(batch)

        model.fit_sample(batch)

        metric_logger.update_model_losses("train", model.loss_values)
        metric_logger.get("train_generator_ssim").update((model.fake_streetviews, model.real_streetviews))


def validate(dataloader_val: DataLoader, batch_transform, model: GAN, metric_logger: MetricLogger):
    for _, batch in enumerate(dataloader_val):
        batch = batch_transform(batch)

        model.test(batch)

        metric_logger.update_model_losses("validate", model.loss_values)
        metric_logger.get("test_generator_ssim").update((model.fake_streetviews, model.real_streetviews))


def visualize(dataset_viz: Subset, model: GAN, epoch: int):
    viz_dir = dataset_viz.out_dir

    for i_sample, idx in enumerate(dataset_viz.indices):
        sample = dataset_viz.dataset.get_untransformed_sample(idx)
        # to_perspective transformation transforms a sample into a batch
        batch = dataset_viz.transform(sample)
        model.test(batch)

        streetview_img = batch_to_numpy(model.fake_streetviews)[0].clip(0, 1)
        target_img = batch_to_numpy(model.real_streetviews)[0]
        discrimination_img = batch_to_numpy(torch.nn.Sigmoid()(model.discriminated_strt_fake))[0]

        save_path = viz_dir / f"sample_{i_sample}" / f"strtviw_discrim&epoch_{epoch}"
        fig = plot_streetview_with_discrimination(streetview_img, discrimination_img, target_img)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()


def run(cfg: dict, out_data_path: Path, batch_transform, dataloader_train: DataLoader, dataloader_val: DataLoader, dataset_viz: Dataset, model: GAN, checkpointer: Checkpointer, metric_logger: MetricLogger):
    profile = cfg["train"]["profile"]
    if profile:
        pr = Profiler(interval=1e-2, async_mode="enabled")
        pr.start()

    for epoch in tqdm(range(cfg["train"]["n_epochs"])):
        train(dataloader_train, batch_transform, model, metric_logger)
        validate(dataloader_val, batch_transform, model, metric_logger)
        visualize(dataset_viz, model, epoch)

        checkpointer.step(epoch, "epoch")
        metric_logger.step()

    if profile:
        pr.stop()
        # pr.print(show_all=True)

        html = pr.output(HTMLRenderer(show_all=True))
        stats_file = out_data_path / "profile_stats.html"
        stats_file.write_text(html, encoding="utf8")


def create_dataloaders(dataset: Dataset, dataset_split_proportions: MutableSequence[float | int], batch_size: int):
    dataset_train, dataset_val, dataset_viz = dataset_split(dataset, dataset_split_proportions)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

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
    metric_logger = MetricLogger(out_data_path / "metrics.log", metrics, torch.device("cuda:0"))

    return metric_logger


if __name__ == "__main__":
    cfg = TrainConfig.load("train")

    in_data_path = cfg.train.in_data_path
    annotations_file = in_data_path / Path(r"annotations.arrow")
    streetview_dir = in_data_path / Path(r"mapillary")
    simulated_dir = in_data_path / Path(r"blender_numpy")

    sample_transform = Compose([
        transformation.Sample2Batch(),
        transformation.RandomPerspective(yaw=(0, 360), pitch=(0, 60), w_fov=(60, 120)),
        transformation.Resize((256, 256), antialias=True),
        transformation.RandomHorizontalFlip(),
        transformation.To(dtype=cfg.data.dtype),
        transformation.Batch2Sample(),
    ])

    dataset = CustomImageDataset(
        annotations_file,
        streetview_dir,
        simulated_dir,
        transform=sample_transform
    )

    split_with_viz = cfg.train.dataset_split + [cfg.train.visualisation.n_samples]
    dataset_train, dataset_val, dataset_viz = dataset_split(dataset, split_with_viz)

    dataloader_train = DataLoader(dataset_train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    viz_transform = Compose([
        transformation.Sample2Batch(),
        transformation.RandomPerspective(yaw=0, pitch=30, w_fov=100),
        transformation.To(dtype=cfg.data.dtype),
        transformation.Resize((256, 256), antialias=True),
        transformation.To(torch.device("cuda:0")),
    ])

    out_data_path = Path(cfg.train.out_data_path)
    out_data_path.mkdir(parents=True, exist_ok=True)
    viz_dir = out_data_path / Path(r"visualisation")

    dataset_viz.transform = viz_transform
    dataset_viz.out_dir = viz_dir

    create_initial_viz(dataset_viz)

    model = GAN(
        dtype=cfg.data.dtype,
        input_channels=cfg.data.input_channels,
        output_channels=cfg.data.output_channels,
        generator_kwargs=cfg.network.generator.model_dump(),
        discriminator_kwargs=cfg.network.discriminator.model_dump()
    )
    init_weights(model, init_type="xavier", gain=0.2)

    checkpoints_dir = out_data_path / Path(r"checkpoints")
    checkpointer = Checkpointer(model, checkpoints_dir, cfg.train.checkpointer.period)

    metric_logger = get_metric_logger(out_data_path)

    batch_transform = Compose([
        transformation.To(torch.device("cuda:0")),
    ])

    run(cfg, out_data_path, batch_transform, dataloader_train, dataloader_val, dataset_viz, model, checkpointer, metric_logger)
