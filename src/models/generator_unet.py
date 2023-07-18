import torch
from torch import nn, Tensor, device

from src.models.blocks.shortcut_connection import UnetSkipConnectionBlock


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, device: device = device("cuda:0")):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.model.to(device)

        # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def forward(self, input):
        return self.model(input)

    def compute_loss(self, real_input: Tensor, generated_input: Tensor) -> Tensor:
        """maximize discriminator error"""
        predic_real = self(real_input)
        predic_gene = self(generated_input)

        target_real = torch.Tensor(1.).expand_as(predic_real)
        target_gene = torch.Tensor(0.).expand_as(predic_gene)

        loss = self.loss(
            torch.cat([predic_real, predic_gene]),
            torch.cat([target_real, target_gene])
        )
        return loss

    def backward(self, real_input: Tensor, generated_input: Tensor) -> Tensor:
        loss_value = self.compute_loss(real_input, generated_input)
        loss_value.backward()
        return loss_value

    def fit(self, real_input: Tensor, generated_input: Tensor):
        """
        fit the discriminator with the real sample and the sample
        that's coming out of the generator.
        """
        self.optimizer.zero_grad()
        self.backward(real_input, generated_input)
        self.optimizer.step()
