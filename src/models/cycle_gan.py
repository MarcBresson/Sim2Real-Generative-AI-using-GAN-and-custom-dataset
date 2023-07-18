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

        self.gen_Sim2Strtview = UnetGenerator(7, 3, num_downs=7).to("cuda:0")
        self.gen_Strtview2Sim = UnetGenerator(3, 7, num_downs=7).to("cuda:0")
        self.discriminator_Sim = PatchGAN(7).to("cuda:0")
        self.discriminator_Strtview = PatchGAN(3).to("cuda:0")

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

    def backward_generators(self):
        self.loss_fn()
        self.loss_value.backward(retain_graph=True)

    def loss_fn(self):
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
        self.backward_generators()
        self.gen_Sim2Strtview.optimizer.step()
        self.gen_Strtview2Sim.optimizer.step()

        self.discriminator_Strtview.fit(self.real_streetviews, self.gen_streetviews)
        self.discriminator_Sim.fit(self.real_simulated, self.gen_simulated)

    def test(self, sample: dict[str, Tensor]):
        self.set_input(sample)
        self.forward(train=False)

        self.loss_fn()

        self.discriminator_Strtview.test(self.real_streetviews, self.gen_streetviews)
        self.discriminator_Sim.test(self.real_simulated, self.gen_simulated)

    def save(self, dir: Path, prefix: str):
        torch.save(self.state_dict(), dir / f"cycleGAN_{prefix}.pth")


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
