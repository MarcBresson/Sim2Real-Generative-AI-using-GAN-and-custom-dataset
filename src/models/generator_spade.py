"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


class SPADEGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        n_filters: int = 64,
        input_size: int = 128,
        input_aspect_ratio: float = 1,
        z_dim: int = 256,
        num_upsampling_layers: Literal["normal", "more", "most"] = "normal",
        use_vae: bool = False,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float32,
        lr: float = 1e-4, beta1: float = 0.0, beta2: float = 0.999,
    ):
        "If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator"
        "dimension of the latent z vector"

        super().__init__()
        self.n_filters = n_filters
        self.input_size = input_size
        self.aspect_ratio = input_aspect_ratio
        self.input_nc = input_nc
        self.num_upsampling_layers = num_upsampling_layers
        self.use_vae = use_vae
        self.z_dim = z_dim

        self.sw, self.sh = self.compute_latent_vector_size()

        if use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(z_dim, 16 * n_filters * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.input_nc, 16 * n_filters, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * n_filters, 16 * n_filters, input_nc=input_nc)

        self.G_middle_0 = SPADEResnetBlock(16 * n_filters, 16 * n_filters, input_nc=input_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * n_filters, 16 * n_filters, input_nc=input_nc)

        self.up_0 = SPADEResnetBlock(16 * n_filters, 8 * n_filters, input_nc=input_nc)  # has learned shortcut
        self.up_1 = SPADEResnetBlock(8 * n_filters, 4 * n_filters, input_nc=input_nc)   # has learned shortcut
        self.up_2 = SPADEResnetBlock(4 * n_filters, 2 * n_filters, input_nc=input_nc)   # has learned shortcut
        self.up_3 = SPADEResnetBlock(2 * n_filters, 1 * n_filters, input_nc=input_nc)   # has learned shortcut

        final_nc = n_filters

        self.num_upsampling_layers = num_upsampling_layers
        if num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * n_filters, n_filters // 2, input_nc=input_nc)
            final_nc = n_filters // 2

        self.conv_img = nn.Conv2d(final_nc, output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.to(device=device, dtype=dtype)

        lr = float(lr)
        beta1 = float(beta1)
        beta2 = float(beta2)
        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))

    def compute_latent_vector_size(self):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('self.num_upsampling_layers [%s] not recognized' %
                             self.num_upsampling_layers)

        sw = self.input_size // (2**num_up_layers)
        sh = round(sw / self.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None) -> torch.Tensor:
        seg = input

        if self.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.n_filters, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.num_upsampling_layers == 'more' or \
           self.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, input_nc, norm_G: str = "spectralinstance"):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(fin, input_nc, norm_type=spade_config_str)
        self.norm_1 = SPADE(fmiddle, input_nc, norm_type=spade_config_str)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, input_nc, norm_type=spade_config_str)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size: int = 3, norm_type: str = "instance"):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
