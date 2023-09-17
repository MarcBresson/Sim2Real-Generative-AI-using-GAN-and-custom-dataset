import torch
from torch import Tensor
from torch import nn


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """extension class to work with simplified inputs"""
    def __call__(self, input_: Tensor, is_target_real: bool) -> Tensor:
        if is_target_real:
            target = torch.ones_like(input_)
        else:
            target = torch.zeros_like(input_)

        loss_value = super().__call__(input_, target)
        return loss_value


class HingeLoss(nn.HingeEmbeddingLoss):
    """extension class to work with simplified inputs"""
    def __call__(self, input_: Tensor, is_target_real: bool) -> Tensor:
        target = torch.ones_like(input_)

        if not is_target_real:
            # target must be 1 or -1
            target = -target

        loss_value = super().__call__(input_, target)
        return loss_value
