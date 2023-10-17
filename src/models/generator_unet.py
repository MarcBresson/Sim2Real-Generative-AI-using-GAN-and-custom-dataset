import torch
from torch import nn


class UnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        n_levels: int = 7,
        n_filters: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        lr: float = 1e-3,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        lr = float(lr)
        beta1 = float(beta1)
        beta2 = float(beta2)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(n_filters * 8, n_filters * 8, norm_layer=norm_layer, innermost=True)
        for _ in range(n_levels - 5):
            unet_block = UnetSkipConnectionBlock(n_filters * 8, n_filters * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(n_filters * 4, n_filters * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(n_filters * 2, n_filters * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(n_filters, n_filters * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, n_filters, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.model.to(device=device, dtype=dtype)

        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))

    def forward(self, _input):
        return self.model(_input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, _input):
        if self.outermost:
            return self.model(_input)

        return torch.cat([_input, self.model(_input)], 1)
