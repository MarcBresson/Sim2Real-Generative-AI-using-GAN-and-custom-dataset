from typing import Union, Any
from pathlib import Path
import json

import torch
from torch import Tensor
from ignite.metrics import Metric, Average
from ignite.exceptions import NotComputableError


class MetricLogger:
    def __init__(self, save_file: Path, metrics: dict[str, Metric], device: torch.device) -> None:
        self.metrics = metrics

        self.metres: dict[str, list] = {}
        for name in self.metrics.keys():
            self.metres[name] = []

        self.device = device

        self.save_file = save_file
        self.save_file.parent.mkdir(exist_ok=True)

        self.number_of_steps = 0

    def step(self):

        for name, metric in self.metrics.items():
            try:
                value = metric.compute()
                self._append(name, value)
                metric.reset()
            except NotComputableError:
                pass

        self.number_of_steps += 1
        json.dump(self.metres, self.save_file.open(mode="w"))

    def _append(self, name: str, value: Union[float, int, Tensor]):
        if name not in self.metres:
            self.metres[name] = [None] * self.number_of_steps

        if isinstance(value, Tensor):
            value = value.item()

        self.metres[name].append(value)

    def update_model_losses(self, key_prefix: str, loss_values: dict[str, Any]):
        """
        update metrics from a dictionnary of values that will be explored recursively.

        Parameters
        ----------
        key_prefix : str
            prefix for the metrics in the dict. Can be "train", or "test".
        loss_values : dict[str, Any]
            metrics value in a recursive dict that has a metric value on each of its lowest levels.
        """
        for k, v in loss_values.items():
            key_prefix_k = key_prefix + "-" + k

            if isinstance(v, dict):
                self.update_model_losses(key_prefix_k, v)
            else:
                self.update_metric(key_prefix_k, output=v)

    def update_metric(self, metric_name: str, output):
        """
        update a metric or create one if it does not exist yet.

        Parameters
        ----------
        metric_name : str
            name of the metric to update/create.
        output : _type_
            value of the metric.
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].update(output)
        else:
            self.metrics[metric_name] = Average(device=self.device)

    def get(self, metric_name: str):
        return self.metrics[metric_name]
