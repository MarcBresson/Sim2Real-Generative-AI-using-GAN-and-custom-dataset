import torch
from torch import Tensor
from torch import nn


class BCEWithLogitsLoss():
    """extension class to work with simplified inputs"""
    def __init__(self, weight: Tensor = None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight: Tensor = None) -> None:
        self.loss = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)

    def __call__(self, input: Tensor, is_target_real: bool) -> Tensor:
        if is_target_real:
            target = torch.ones_like(input)
        else:
            target = torch.zeros_like(input)

        loss_value = self.loss(input, target)
        return loss_value
