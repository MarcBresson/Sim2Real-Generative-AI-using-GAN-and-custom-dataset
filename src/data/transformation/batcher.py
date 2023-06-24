import torch


def batcher(input_: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    in case individual images were provided, convert them to batches
    """
    if len(input_["streetview"].shape) == 3:
        input_["streetview"] = input_["streetview"].unsqueeze(0)
    if len(input_["simulated"].shape) == 3:
        input_["simulated"] = input_["simulated"].unsqueeze(0)

    return input_
