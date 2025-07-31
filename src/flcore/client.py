from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from asyncio import Server
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Hashable, NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from .utils.model import move_parameters

if TYPE_CHECKING:
    from .server import Server


ClientMetrics = dict[str, int | float]


class ClientData(NamedTuple):
    train_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader
    datasize: int


class ClientModel(NamedTuple):
    device: torch.device
    model: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module
    personalized_model: nn.Module | None
    personalized_optimizer: optim.Optimizer | None
    personalized_criterion: nn.Module | None


def connection_required(method):
    @functools.wraps(method)
    def wrapper(self: Client, *args, **kwargs):
        if not self.connected:
            raise RuntimeError(f"Client {self.id} is not connected.")
        return method(self, *args, **kwargs)

    return wrapper


class Client(ABC):
    def __init__(self, id: Hashable, workdir: str | PathLike) -> None:
        """
        Abstract class for a client in the federated learning system.

        :param id: The unique identifier of the client.
        :param workdir: The directory to store the state of the client.
        """
        self.id = id
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.connected = False

    @abstractmethod
    def load_data(self, server: Server) -> ClientData:
        """
        Load the `DataLoader` for training, validation and testing.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, server: Server) -> ClientModel:
        """
        Prepare the local model and optimzier for training.
        """
        raise NotImplementedError

    @connection_required
    def receive_parameters(self, global_model: nn.Module):
        """
        Receive the global model (cpu) from the server, and move its parameters to the local model (local device).
        """
        move_parameters(global_model, self.model, buffer=False, zero_grad=True)

    @connection_required
    def send_parameters(self) -> nn.Module:
        """
        Send the local model (cpu) to the server.
        """
        return copy.deepcopy(self.model).cpu()

    @abstractmethod
    def on_train(self):
        """
        Train the local model and personalized model.
        """
        raise NotImplementedError

    @abstractmethod
    def on_evaluation(self) -> ClientMetrics:
        """
        Evaluate the local model and personalized model on the validation set.
        """
        raise NotImplementedError

    @abstractmethod
    def on_test(self) -> ClientMetrics:
        """
        Evaluate the local model and personalized model on the test set.
        """
        raise NotImplementedError

    @connection_required
    def train(self):
        """
        Train the local model and personalized model.
        """
        self.on_train()

    @connection_required
    def evaluate(self) -> ClientMetrics:
        """
        Evaluate the local model and personalized model on the validation set.
        """
        return self.on_evaluation()

    @connection_required
    def test(self) -> ClientMetrics:
        """
        Evaluate the local model and personalized model on the test set.
        """
        return self.on_test()

    def connect(self, server: Server):
        """
        Connect to the client. Prepare the data, model and optimizer.
        """
        if self.connected:
            logger.debug(f"Client {self.id} is already connected.")
            return

        (
            self.train_dataloader,
            self.validation_dataloader,
            self.test_dataloader,
            self.datasize,
        ) = self.load_data(server)
        (
            self.device,
            self.model,
            self.optimizer,
            self.criterion,
            self.personalized_model,
            self.personalized_optimizer,
            self.personalized_criterion,
        ) = self.load_model(server)

        self.connected = True

    def disconnect(self):
        """
        Disconnect from the client. Clean up the data, model and optimizer.
        """
        if not self.connected:
            logger.debug(f"Client {self.id} is already disconnected.")
            return

        del self.train_dataloader
        del self.validation_dataloader
        del self.test_dataloader
        del self.datasize

        del self.device
        del self.model
        del self.optimizer
        del self.criterion
        del self.personalized_model
        del self.personalized_optimizer
        del self.personalized_criterion

        self.connected = False

    def __repr__(self) -> str:
        info = f"id={self.id},workdir={self.workdir},connected={self.connected}"
        return f"<{self.__class__.__name__} [{info}]>"
