from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from .client import ClientData
from .server import Server


@dataclass
class FlDataset:
    id_data: dict[Hashable, ClientData]
    name: str
    num_class: int
    fractions: tuple[float, float, float]
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.num_clients = len(self.id_data)
        self.clients = list(self.id_data.keys())
        self.datasize = sum(client_data.datasize for client_data in self.id_data.values())


class FederatedLearning[S: Server](ABC):
    def __init__(
        self,
        *,
        server: S,
        logdir: str | Path,
        max_epoch: int,
        tensorboard: bool = True,
    ):
        """
        The base class of a simulated federated learning system. Support parallel training.

        :param server: A federated learning server.
        :param log_dir: The log directory of logging file, e.g., val.jsonl.
        :param max_epoch: The maximum number of global epochs.
        :param tensorboard: Logging evaluation metrics to tensorboard.
        :param max_workers: The maximum number of workers in parallel training.
        """
        self.server = server
        self.logdir = Path(logdir)
        self.max_epoch = max_epoch
        self.tensorboard = SummaryWriter(logdir) if tensorboard else None
        self.logdir.mkdir(parents=True, exist_ok=True)

    def to_brief_metrics(self, data: dict[Hashable, dict[str, float]]) -> dict[str, float]:
        # expected layout:
        # {
        #   "0":  <-- client id
        #     {
        #       "accuracy": 0.1, <-- metrics
        #       "loss": 0.1
        #       ...
        #     }
        # }
        names = sorted(data[next(iter(data))].keys())
        name_values = {name: torch.tensor([item[name] for item in data.values()]) for name in names}
        name_mean = {name: values.mean().item() for name, values in name_values.items()}
        name_std = {name: values.std().item() for name, values in name_values.items()}

        result = {f"{name}-mean": value for name, value in name_mean.items()}
        result |= {f"{name}-std": value for name, value in name_std.items()}

        return result

    def save_metrics(self, epoch: int, data: Any, filename: str | Path):
        with open(self.logdir / filename, "a+") as f:
            json.dump(
                {
                    "protocol": self.__class__.__name__,
                    "datetime": datetime.now().isoformat(),
                    "epoch": epoch,
                    "metrics": data,
                },
                f,
                indent=None,
                ensure_ascii=False,
            )
            f.write("\n")

        if not self.tensorboard:
            return

        brief_metrics = self.to_brief_metrics(data)
        for name, value in brief_metrics.items():
            self.tensorboard.add_scalar(f"{filename}-{name}", value, epoch)

    @abstractmethod
    def algorithm(self) -> nn.Module:
        """
        The abstract method of federated learning algorithm.

        :return: The trained model.
        """
        raise NotImplementedError

    def run(self):
        """
        Run the simulated federated learning algorithm.

        :return: The trained model.
        """
        if len(self.server.registered_clients) == 0:
            raise ValueError("There are no registered clients in server.")

        model = self.algorithm()

        return model

    def __repr__(self) -> str:
        info = f"logdir={self.logdir},tensorboard={self.tensorboard}"
        return f"<{self.__class__.__name__} [{info}]>"
