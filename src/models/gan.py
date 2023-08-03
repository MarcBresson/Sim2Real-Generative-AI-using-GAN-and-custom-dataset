from pathlib import Path

import torch
from torch import Tensor
from torch import nn

from src.models.discriminator_patchgan import PatchGAN
from src.models.generator_unet import UnetGenerator
from src.eval.gan_loss import BCEWithLogitsLoss


class GAN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gen_Sim2Strtview = UnetGenerator(7, 3, num_downs=8, ngf=64, lr=2e-4)
        self.discriminator_Strtview = PatchGAN(3)

        self.fooling_loss = BCEWithLogitsLoss()
        self.gen_loss = nn.L1Loss()

    def set_input(self, sample: dict[str, Tensor]):
        self.real_streetviews = sample["streetview"]
        self.real_simulated = sample["simulated"]

    def forward(self, train: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if train:
            self.gen_Sim2Strtview.train()
        else:
            self.gen_Sim2Strtview.eval()

        self.gen_streetviews = self.gen_Sim2Strtview(self.real_simulated)

    def compute_generator_loss(self):
        self.discriminator_Strtview.eval()

        # we want to optimize the generator so that the discriminator is wrong more often.
        # we compute the loss with a target being the opposite of what we expect.
        self.loss_fooling_discrim_strtview = self.fooling_loss(self.discriminator_Strtview(self.gen_streetviews), True)

        self.loss_gen_strtview = self.gen_loss(self.gen_streetviews, self.real_streetviews)

        self.generator_loss_value = self.loss_fooling_discrim_strtview + self.loss_gen_strtview

    def fit_sample(self, sample: dict[str, Tensor]):
        self.set_input(sample)

        self.gen_Sim2Strtview.optimizer.zero_grad()
        self.forward()
        self.compute_generator_loss()
        self.generator_loss_value.backward()
        self.gen_Sim2Strtview.optimizer.step()

        self.discriminator_Strtview.fit(self.real_streetviews, self.gen_streetviews)

    def test(self, sample: dict[str, Tensor]):
        self.set_input(sample)
        self.forward(train=False)

        self.compute_generator_loss()

        self.discriminator_Strtview.test(self.real_streetviews, self.gen_streetviews)

    def save(self, dir: Path, prefix: str):
        torch.save(self.state_dict(), dir / f"GAN_{prefix}.pth")
