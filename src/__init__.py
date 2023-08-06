import torch


def get_device(device: str):
    is_gpu = "cuda" in device

    if is_gpu and not torch.cuda.is_available():
        raise RuntimeError(f"Target device {device} seems to be a GPU, but torch cannot find any cuda devices.")

    return torch.device(device)
