from pathlib import Path

import torch
from torch import Tensor
from torch import nn

from src.models.discriminator_patchgan import PatchGAN
from src.models.generator_unet import UnetGenerator
from src.eval.gan_loss import BCEWithLogitsLoss


class CycleGAN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gen_Sim2Strtview = UnetGenerator(7, 3, num_downs=7, lr=1e-2)
        self.gen_Strtview2Sim = UnetGenerator(3, 7, num_downs=7, lr=1e-2)
        self.discriminator_Sim = PatchGAN(7)
        self.discriminator_Strtview = PatchGAN(3)

        self.fooling_loss = BCEWithLogitsLoss()
        self.cycle_loss = nn.L1Loss()

    def set_input(self, sample: dict[str, Tensor]):
        self.real_streetviews = sample["streetview"]
        self.real_simulated = sample["simulated"]

    def forward(self, train: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if train:
            self.gen_Sim2Strtview.train()
            self.gen_Strtview2Sim.train()
        else:
            self.gen_Sim2Strtview.eval()
            self.gen_Strtview2Sim.eval()

        self.gen_streetviews = self.gen_Sim2Strtview(self.real_simulated)
        self.gen_simulated = self.gen_Strtview2Sim(self.real_streetviews)

        self.cycled_simulated = self.gen_Strtview2Sim(self.gen_streetviews)
        self.cycled_streetviews = self.gen_Sim2Strtview(self.gen_simulated)

    def compute_generators_loss(self):
        self.discriminator_Sim.eval()
        self.discriminator_Strtview.eval()

        # we want to optimize the generator so that the discriminator is wrong more often.
        # we compute the loss with a target being the opposite of what we expect.
        loss_fooling_discrim_strtview = self.fooling_loss(self.discriminator_Strtview(self.gen_streetviews), True)
        loss_fooling_discrim_sim = self.fooling_loss(self.discriminator_Sim(self.gen_simulated), True)

        loss_cycle_strtview = self.cycle_loss(self.cycled_streetviews, self.real_streetviews)
        loss_cycle_sim = self.cycle_loss(self.cycled_simulated, self.real_simulated)

        self.loss_value = loss_fooling_discrim_strtview + loss_fooling_discrim_sim + loss_cycle_strtview + loss_cycle_sim

    def fit_sample(self, sample: dict[str, Tensor]):
        self.set_input(sample)
        self.forward()

        self.gen_Sim2Strtview.optimizer.zero_grad()
        self.gen_Strtview2Sim.optimizer.zero_grad()
        self.compute_generators_loss()
        self.loss_value.backward(retain_graph=True)
        self.gen_Sim2Strtview.optimizer.step()
        self.gen_Strtview2Sim.optimizer.step()

        self.discriminator_Strtview.fit(self.real_streetviews, self.gen_streetviews)
        self.discriminator_Sim.fit(self.real_simulated, self.gen_simulated)

    def test(self, sample: dict[str, Tensor]):
        self.set_input(sample)
        self.forward(train=False)

        self.compute_generators_loss()

        self.discriminator_Strtview.test(self.real_streetviews, self.gen_streetviews)
        self.discriminator_Sim.test(self.real_simulated, self.gen_simulated)

    def save(self, dir: Path, prefix: str):
        torch.save(self.state_dict(), dir / f"cycleGAN_{prefix}.pth")
