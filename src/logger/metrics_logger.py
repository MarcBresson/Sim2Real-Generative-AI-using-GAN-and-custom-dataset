from typing import Any
from pathlib import Path
import json

from torch import Tensor
from ignite.metrics import Metric


class MetricLogger:
    def __init__(self, save_file: Path, metrics: dict[str, Metric]) -> None:
        self.metrics = metrics

        self.metres: dict[str, list] = {}
        for name in self.metrics.keys():
            self.metres[name] = []

        self.save_file = save_file
        self.save_file.parent.mkdir(exist_ok=True)

    def step(self):
        for name, metric in self.metrics.items():
            value = metric.compute()

            self._append(name, value)
            metric.reset()

        json.dump(self.metres, self.save_file.open(mode="w"))

    def _append(self, name: str, value: float | int | Tensor):
        if name not in self.metres:
            self.metres[name] = [None] * len(self.metres)

        if isinstance(value, Tensor):
            value = value.item()

        self.metres[name].append(value)

    def get(self, metric_name: str):
        return self.metrics[metric_name]
