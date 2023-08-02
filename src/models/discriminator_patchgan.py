import torch
from torch import Tensor, nn, device as Device

from src.eval.gan_loss import BCEWithLogitsLoss


class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc: int,
        nbr_filters: int = 64,
        n_layers: int = 3,
        norm_layer=nn.BatchNorm2d,
        device: Device = Device("cuda:0")
    ):
        """
        Construct a PatchGAN discriminator

        Parameters
        ----------
        input_nc : int
            the number of channels in input images
        nbr_filters : int, optional
            the number of filters in the first conv layer, by default 64
        n_layers : int, optional
            the number of conv layers in the discriminator, by default 3
        norm_layer : _type_, optional
            normalization layer, by default nn.BatchNorm2d
        """
        super().__init__()

        self.loss = BCEWithLogitsLoss()
        self.model = self.build(input_nc, nbr_filters, n_layers, norm_layer)
        self.model.to(device)

        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def build(self, input_nc: int, nbr_filters: int = 64, n_layers: int = 3, norm_layer=nn.BatchNorm2d):
        """build the discriminator"""
        use_bias = True
        if isinstance(norm_layer, nn.BatchNorm2d):  # batchnorm contains affine parametres
            use_bias = False

        sequence = [
            nn.Conv2d(input_nc, nbr_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        in_nbr_filter = nbr_filters

        for n in range(1, n_layers + 1):  # gradually increase the number of filters
            nbr_filt_mult = min(2 ** n, 8)
            out_nbr_filter = nbr_filters * nbr_filt_mult

            if n == n_layers:
                stride = 1
            else:
                stride = 2

            sequence += [
                nn.Conv2d(in_nbr_filter, out_nbr_filter, kernel_size=4, stride=stride, padding=1, bias=use_bias),
                norm_layer(out_nbr_filter),
                nn.LeakyReLU(0.2, True)
            ]

            in_nbr_filter = nbr_filters * nbr_filt_mult

        sequence += [nn.Conv2d(in_nbr_filter, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map

        return nn.Sequential(*sequence)

    def forward(self, input_: Tensor):
        return self.model(input_)

    def compute_loss(self, real_input: Tensor, generated_input: Tensor):
        loss_real = self.loss(self.forward(real_input), True)
        loss_gen = self.loss(self.forward(generated_input.detach()), False)

        self.loss_value = (loss_real + loss_gen) * 0.5

    def fit(self, real_input: Tensor, generated_input: Tensor):
        """
        fit the discriminator with the real sample and the sample
        that's coming out of the generator.
        """
        # switch the model to train mode
        self.train()

        self.optimizer.zero_grad()

        self.compute_loss(real_input, generated_input)
        self.loss_value.backward()

        self.optimizer.step()

    def test(self, real_input: Tensor, generated_input: Tensor):
        # switch the model to eval mode
        self.eval()

        self.real_discrim = self.forward(real_input)
        self.gene_discrim = self.forward(generated_input)

        self.compute_loss(real_input, generated_input)
