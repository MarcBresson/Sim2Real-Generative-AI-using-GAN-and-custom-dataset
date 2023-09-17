from pathlib import Path

from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


class Checkpointer:
    def __init__(self, model: nn.Module, save_dir: Path, period: int):
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model

        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)

        self.period = period

    def step(self, iteration: int, prefix: str = "iteration"):
        self.model.save(self.save_dir, f"latest_{prefix}")

        if iteration % self.period == 0:
            self.model.save(self.save_dir, f"{iteration}_{prefix}")
