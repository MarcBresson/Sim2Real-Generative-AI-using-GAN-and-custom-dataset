from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import Subset
from torchvision.transforms import Compose

from src.data.dataset import CustomImageDataset
from src.data.transformation import toNumpy
from src.data.visualisation import (
    batch_to_numpy,
    multichannels_to_individuals,
    plot_sim,
    plot_streetview_with_discrimination,
)
from src.models import GAN


@dataclass
class Visualisation:
    subset: Subset
    transform: Compose
    out_directory: Path
    every_nth_epoch: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.create_sample_directories()

    def create_initial(self):
        """create the first visualisation with the input passes."""
        dataset: CustomImageDataset = self.subset.dataset

        for i_sample, torch_sim_batch in enumerate(self.transformed_samples()):
            sim_img = toNumpy()(torch_sim_batch)[0]
            channels = multichannels_to_individuals(sim_img, dataset.passes_channel_nbr)

            save_path = self.get_sample_dir(i_sample) / "real_simulated"
            fig = plot_sim(channels, dataset.render_passes)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()

    def step(self, model: GAN, epoch: int):
        # starting from epoch 1, we visualize once every nth epoch
        if (epoch - 1) % self.every_nth_epoch != 0:
            return

        for i_sample, torch_sim_batch in enumerate(self.transformed_samples()):
            model.test(torch_sim_batch)

            # due to computational errors, sometimes values were slightly negative
            # (-0.002) or slightly superior to 1 (1.001)
            streetview_img = batch_to_numpy(model.fake_streetviews)[0].clip(0, 1)
            target_img = batch_to_numpy(model.real_streetviews)[0]
            discrimination_img = batch_to_numpy(
                nn.Sigmoid()(model.discriminated_strt_fake)
            )[0]

            save_path = self.get_sample_dir(i_sample) / f"strtviw_discrim&epoch_{epoch}"
            fig = plot_streetview_with_discrimination(
                streetview_img, discrimination_img, target_img
            )
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()

    def create_sample_directories(self):
        """create one folder per sample to visualize."""
        for i_sample in range(len(self.subset)):
            self.get_sample_dir(i_sample).mkdir(parents=True, exist_ok=True)

    def get_sample_dir(self, i_sample: int) -> Path:
        return self.out_directory / f"sample_{i_sample}"

    def transformed_samples(self) -> Iterator[Tensor]:
        dataset: CustomImageDataset = self.subset.dataset

        for idx in self.subset.indices:
            sample = dataset.get_untransformed_sample(idx)

            torch_sim_batch = self.transform(sample["simulated"])

            yield torch_sim_batch
