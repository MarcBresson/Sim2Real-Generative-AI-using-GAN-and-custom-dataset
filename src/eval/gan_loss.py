import torch
from torch import Tensor
from torch import nn


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """extension class to work with simplified inputs"""
    def __call__(self, input: Tensor, is_target_real: bool) -> Tensor:
        bool_as_float = [float(is_target_real)]
        target = torch.tensor(bool_as_float, device=input.device).expand_as(input)

        loss = super().__call__(input, target)
        return loss
