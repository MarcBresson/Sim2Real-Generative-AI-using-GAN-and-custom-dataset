from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.eval.gan_loss import BCEWithLogitsLoss
from src.models.discriminator_patchgan import PatchGAN
from src.models.generator_spade import SPADEGenerator


class CycleGAN(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype | str = "float32",
        input_channels: int = 7,
        output_channels: int = 3,
        device: torch.device = torch.device("cuda:0"),
        generator_kwargs: dict[str, Any] | None = None,
        discriminator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        dtype_: torch.dtype
        if isinstance(dtype, str):
            dtype_ = getattr(torch, dtype)
        else:
            dtype_ = dtype

        if generator_kwargs is None:
            generator_kwargs = {}
        if discriminator_kwargs is None:
            discriminator_kwargs = {}

        self.generator_strt = SPADEGenerator(
            input_channels,
            output_channels,
            device=device,
            dtype=dtype_,
            **generator_kwargs,
        )
        self.generator_simu = SPADEGenerator(
            output_channels,
            input_channels,
            device=device,
            dtype=dtype_,
            **generator_kwargs,
        )
        self.discriminator_strt = PatchGAN(
            input_channels,
            output_channels,
            device=device,
            dtype=dtype_,
            **discriminator_kwargs,
        )
        self.discriminator_simu = PatchGAN(
            output_channels,
            input_channels,
            device=device,
            dtype=dtype_,
            **discriminator_kwargs,
        )

        self.fooling_loss = BCEWithLogitsLoss()
        self.cycle_loss = nn.L1Loss()

        self.init_loss_values()

    def init_loss_values(self):
        self.loss_values: dict[str, Any] = {}
        self.loss_values["generator"] = {}
        self.loss_values["discriminator"] = {}

    def set_input(self, sample: dict[str, Tensor]):
        self.real_streetviews = sample["streetview"]
        self.real_simulated = sample["simulated"]

    def forward(self):
        self.forward_G()

    def forward_G(self):
        self.fake_streetviews = self.generator_strt(self.real_simulated)
        self.fake_simulated = self.generator_simu(self.real_streetviews)

        self.cycled_streetviews = self.generator_strt(self.fake_simulated)
        self.cycled_simulated = self.generator_simu(self.fake_streetviews)

    def forward_D(self):
        self.discriminated_strt_real = self.discriminator_strt(self.real_streetviews)
        self.discriminated_strt_fake = self.discriminator_strt(
            self.fake_streetviews.detach()
        )

        self.discriminated_simu_real = self.discriminator_simu(self.real_simulated)
        self.discriminated_simu_fake = self.discriminator_simu(
            self.fake_simulated.detach()
        )

    def backward_G(self, only_compute_loss: bool = False):
        # we want to optimize the generator so that the discriminator is wrong more often.
        # we compute the loss with a target being the opposite of what we expect.
        discriminated_strt_fake = self.discriminator_strt(self.fake_streetviews)
        fooling_strt_loss_value = self.fooling_loss(discriminated_strt_fake, True)

        discriminated_simu_fake = self.discriminator_simu(self.fake_simulated)
        fooling_simu_loss_value = self.fooling_loss(discriminated_simu_fake, True)

        cycled_strt_loss_value = self.cycle_loss(
            self.cycled_streetviews, self.real_streetviews
        )
        cycled_simu_loss_value = self.cycle_loss(
            self.cycled_simulated, self.real_simulated
        )

        loss_value: Tensor = (
            fooling_strt_loss_value
            + fooling_simu_loss_value
            + cycled_strt_loss_value
            + cycled_simu_loss_value
        )

        if not only_compute_loss:
            loss_value.backward()

        self.loss_values["generator"][
            "fooling_strt_loss_value"
        ] = fooling_strt_loss_value
        self.loss_values["generator"][
            "fooling_simu_loss_value"
        ] = fooling_simu_loss_value
        self.loss_values["generator"]["cycled_strt_loss_value"] = cycled_strt_loss_value
        self.loss_values["generator"]["cycled_simu_loss_value"] = cycled_simu_loss_value
        self.loss_values["generator"]["loss_value"] = loss_value

    def backward_D(self, only_compute_loss: bool = False):
        loss_real_strt_value = self.fooling_loss(self.discriminated_strt_real, True)
        loss_gen_strt_value = self.fooling_loss(self.discriminated_strt_fake, False)
        loss_strt_value = (loss_real_strt_value + loss_gen_strt_value) * 0.5

        loss_real_simu_value = self.fooling_loss(self.discriminated_simu_real, True)
        loss_gen_simu_value = self.fooling_loss(self.discriminated_simu_fake, False)
        loss_simu_value = (loss_real_simu_value + loss_gen_simu_value) * 0.5

        if not only_compute_loss:
            loss_strt_value.backward()
            loss_simu_value.backward()

        self.loss_values["discriminator"]["loss_real_strt_value"] = loss_real_strt_value
        self.loss_values["discriminator"]["loss_gen_strt_value"] = loss_gen_strt_value
        self.loss_values["discriminator"]["loss_strt_value"] = loss_strt_value
        self.loss_values["discriminator"]["loss_real_simu_value"] = loss_real_simu_value
        self.loss_values["discriminator"]["loss_gen_simu_value"] = loss_gen_simu_value
        self.loss_values["discriminator"]["loss_simu_value"] = loss_simu_value

    def fit_sample(self, sample: dict[str, Tensor]):
        self.train()

        self.set_input(sample)

        self.generator_strt.optimizer.zero_grad()
        self.generator_simu.optimizer.zero_grad()
        self.forward_G()
        self.backward_G()
        self.generator_strt.optimizer.step()
        self.generator_simu.optimizer.step()

        self.discriminator_strt.optimizer.zero_grad()
        self.discriminator_simu.optimizer.zero_grad()
        self.forward_D()
        self.backward_D()
        self.discriminator_strt.optimizer.step()
        self.discriminator_simu.optimizer.step()

    def test(self, sample: dict[str, Tensor]):
        self.eval()

        self.set_input(sample)

        self.forward_G()
        self.backward_G(True)

        self.forward_D()
        self.backward_D(True)

    def save(self, save_dir: Path, prefix: str):
        torch.save(self.state_dict(), save_dir / f"GAN_{prefix}.pth")
