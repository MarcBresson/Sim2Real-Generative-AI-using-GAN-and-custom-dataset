import torch
from torch import Tensor, nn

from config import OptimizerKwargs, get_config_to_dict


class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        output_nc: int,
        n_filters: int = 64,
        n_layers: int = 3,
        norm_layer=nn.BatchNorm2d,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float32,
        optimizer_config: OptimizerKwargs | None = None,
    ):
        """
        Construct a PatchGAN discriminator

        Parameters
        ----------
        input_nc : int
            the number of channels in input images
        n_filters : int, optional
            the number of filters in the first conv layer, by default 64
        n_layers : int, optional
            the number of conv layers in the discriminator, by default 3
        norm_layer : _type_, optional
            normalization layer, by default nn.BatchNorm2d
        """
        super().__init__()

        self.model = self.build(output_nc, n_filters, n_layers, norm_layer)
        self.model.to(device=device, dtype=dtype)

        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        optimizer_kwargs = get_config_to_dict(optimizer_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)

    def build(
        self,
        output_nc: int,
        n_filters: int = 64,
        n_layers: int = 3,
        norm_layer=nn.BatchNorm2d,
    ):
        """build the discriminator"""
        use_bias = True
        if isinstance(
            norm_layer, nn.BatchNorm2d
        ):  # batchnorm contains affine parametres
            use_bias = False

        sequence = [
            nn.Conv2d(output_nc, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        in_nbr_filter = n_filters

        for n in range(1, n_layers + 1):  # gradually increase the number of filters
            nbr_filt_mult = min(2**n, 8)
            out_nbr_filter = n_filters * nbr_filt_mult

            if n == n_layers:
                stride = 1
            else:
                stride = 2

            sequence += [
                nn.Conv2d(
                    in_nbr_filter,
                    out_nbr_filter,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(out_nbr_filter),
                nn.LeakyReLU(0.2, True),
            ]

            in_nbr_filter = n_filters * nbr_filt_mult

        sequence += [
            nn.Conv2d(in_nbr_filter, 1, kernel_size=4, stride=1, padding=1)
        ]  # output 1 channel prediction map

        return nn.Sequential(*sequence)

    def forward(self, input_: Tensor):
        return self.model(input_)
