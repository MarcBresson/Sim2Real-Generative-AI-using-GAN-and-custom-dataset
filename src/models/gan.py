from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from config import DiscriminatorConfig, GeneratorConfig, get_config_to_dict
from src.eval.gan_loss import HingeLoss
from src.models.discriminator_patchgan import PatchGAN
from src.models.generator_spade import SPADEGenerator


class GAN(nn.Module):
    real_streetviews: Tensor
    real_simulated: Tensor
    fake_streetviews: Tensor
    discriminated_strt_real: Tensor
    discriminated_strt_fake: Tensor

    def __init__(
        self,
        dtype: torch.dtype | str = "float32",
        input_channels: int = 7,
        output_channels: int = 3,
        device: torch.device = torch.device("cuda:0"),
        generator_config: GeneratorConfig | None = None,
        discriminator_config: DiscriminatorConfig | None = None,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str):
            _dtype: torch.dtype = getattr(torch, dtype)

        generator_kwargs = get_config_to_dict(generator_config)
        discriminator_kwargs = get_config_to_dict(discriminator_config)

        self.generator = SPADEGenerator(
            input_channels,
            output_channels,
            device=device,
            dtype=_dtype,
            **generator_kwargs,
        )
        self.discriminator = PatchGAN(
            output_channels, device=device, dtype=_dtype, **discriminator_kwargs
        )

        self.fooling_loss = HingeLoss()
        self.gen_loss = nn.L1Loss()

        self.init_loss_values()

    def init_loss_values(self):
        self.loss_values: dict[str, Any] = {}
        self.loss_values["generator"] = {}
        self.loss_values["discriminator"] = {}

    def set_input(self, sample: dict[str, Tensor]):
        self.real_streetviews = sample["streetview"]
        self.real_simulated = sample["simulated"]

    def forward(self, real_simulated: Tensor) -> Tensor:
        self.real_simulated = real_simulated
        self.forward_G()
        return self.fake_streetviews

    def forward_G(self):
        self.fake_streetviews: Tensor = self.generator(self.real_simulated)

    def forward_D(self):
        self.discriminated_strt_real = self.discriminator(self.real_streetviews)
        self.discriminated_strt_fake = self.discriminator(
            self.fake_streetviews.detach()
        )

    def backward_G(self, only_compute_loss: bool = False):
        # we want to optimize the generator so that the discriminator is wrong more often.
        # we compute the loss with a target being the opposite of what we expect.
        discriminated_fake = self.discriminator(self.fake_streetviews)
        fooling_loss_value = self.fooling_loss(discriminated_fake, True)

        gen_loss_value = self.gen_loss(self.fake_streetviews, self.real_streetviews)

        loss_value: Tensor = gen_loss_value + fooling_loss_value

        if not only_compute_loss:
            loss_value.backward()

        self.loss_values["generator"]["fooling_loss_value"] = fooling_loss_value
        self.loss_values["generator"]["gen_loss_value"] = gen_loss_value
        self.loss_values["generator"]["loss_value"] = loss_value

    def backward_D(self, only_compute_loss: bool = False):
        loss_real_value = self.fooling_loss(self.discriminated_strt_real, True)
        loss_gen_value = self.fooling_loss(self.discriminated_strt_fake, False)
        loss_value = (loss_gen_value + loss_real_value) * 0.5

        if not only_compute_loss:
            loss_value.backward()

        self.loss_values["discriminator"]["loss_real_value"] = loss_real_value
        self.loss_values["discriminator"]["loss_gen_value"] = loss_gen_value
        self.loss_values["discriminator"]["loss_value"] = loss_value

    def fit_sample(self, sample: dict[str, Tensor]):
        self.train()

        self.set_input(sample)

        self.generator.optimizer.zero_grad()
        self.forward_G()
        self.backward_G()
        self.generator.optimizer.step()

        self.discriminator.optimizer.zero_grad()
        self.forward_D()
        self.backward_D()
        self.discriminator.optimizer.step()

    def test(self, sample: dict[str, Tensor]):
        self.eval()

        self.set_input(sample)

        self.forward_G()
        self.backward_G(True)

        self.forward_D()
        self.backward_D(True)

    def save(self, save_dir: Path, prefix: str):
        torch.save(self.state_dict(), save_dir / f"GAN_{prefix}.pth")
