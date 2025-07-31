from __future__ import annotations

import math
import os
import platform
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from typing import TYPE_CHECKING, Hashable, Iterable, Literal, Sequence

import torch.nn as nn
from loguru import logger

from .client import Client, ClientMetrics
from .utils.model import aggregate_model

if TYPE_CHECKING:
    from .robust import RobustFn

CollectedClientMetrics = dict[Hashable, ClientMetrics]


def parallel_do(
    clients: Iterable[Client],
    do: Literal["train", "evaluate", "test"],
    max_workers: int | None = None,
    executor: ThreadPoolExecutor | None = None,
):
    """
    Train multiple clients in parallel.
    """
    if (
        platform.system() == "Linux"
        and ("MKL_NUM_THREADS" not in os.environ and "OMP_NUM_THREADS" not in os.environ)
        and max_workers != 1
    ):
        warnings.warn(
            "Currently PyTorch has serious memory leak with multi-threading on Linux. "
            "We temporarily stop using multi-threading to avoid the issue. "
            "A Temporary solution is to set MKL_NUM_THREADS and OMP_NUM_THREADS to a proper number."
            "For more information see https://github.com/pytorch/pytorch/issues/64412 "
            "and https://github.com/pytorch/pytorch/issues/64535."
        )
        max_workers = 1

    if max_workers == 1:
        return [getattr(client, do)() for client in clients]

    def _worker(client: Client):
        return getattr(client, do)()

    if executor:
        results = executor.map(_worker, clients)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_worker, clients)

    return list(results)  # consume the generator sothat we can catch the exceptions


class Server[C: Client]:
    def __init__(
        self,
        *,
        model: nn.Module,
        select_ratio: float,
        clients: Iterable[C],
        learning_rate: float = 1.0,
        robust_fn: RobustFn | None = None,
    ):
        """
        The server in federated learning. It works with some clients in ``FederatedLearning``.

        You can specify global learning rate in `learning_rate`. The global learning rate may distinctly slow down the
        convergence time but take smoother loss variance in return.

        :param select_ratio: Client select ratio of each round in federated learning.
        :param learning_rate: The global learning rate, this works in aggregation.
        :param robust_fn: The robust aggregation function.
        """
        self.model = model.cpu()
        self.select_ratio = select_ratio
        self.learning_rate = learning_rate
        self.robust_fn = robust_fn
        self.registered_clients = {client.id: client for client in clients}

    def select_clients(self) -> list[C]:
        """
        Randomly select ``int(select_ratio * num_registered_clients)`` clients.

        :return: Selected clients
        """
        k = int(len(self.registered_clients) * self.select_ratio)
        selected_clients = random.sample(list(self.registered_clients.values()), k=k)

        return selected_clients

    def aggregate(self, models: Sequence[nn.Module], weights: Sequence[float] | None = None):
        """
        Aggregate local models to global model by aggregating updates (delta of models). Each model update will be
        multiplied by server's learning rate and corresponding weight to perform aggregation.

        :param models: Models to be aggregated.
        :param weights: Weight that corresponding model update, expected the sum equals to 1.

        :raises ValueError: When no models to be aggregated.
        :raises ValueError: When the length of weights and models are not the same.
        :raises ValueError: When sum of weights is not 1.
        """
        if len(models) == 0:
            raise ValueError("Not enough models to perform aggregation.")

        if weights is None:
            weights = [1 / len(models)] * len(models)

        if len(weights) != len(models):
            raise ValueError(f"The length of weights ({len(weights)}) and clients ({len(models)}) are not the same.")

        if not math.isclose(math.fsum(weights), 1.0):
            raise ValueError(f"The sum of weights should be closed to 1, got {sum(weights)}.")

        if self.robust_fn:
            models, weights = self.robust_fn(self.model, models, weights)

        weights = [weight * self.learning_rate for weight in weights]
        self.model = aggregate_model(self.model, models, weights, buffer=True)

    def connect(self, clients: Iterable[C]):
        """
        Connect client to server.

        :param client: The client to be connected.
        """
        for client in clients:
            if client.id not in self.registered_clients:
                logger.warning(f"Client {client.id} is not registered.")
                self.registered_clients[client.id] = client

            logger.debug(f"Connecting client {client.id}.")
            client.connect(self)

    def send_parameters(self, clients: Iterable[C]):
        """
        Send the global model to clients.

        :param clients: The clients to be sent.
        """
        for client in clients:
            if client.id not in self.registered_clients:
                logger.warning(f"Client {client.id} is not registered.")
                self.registered_clients[client.id] = client

            if not client.connected:
                logger.warning(f"Client {client.id} is not connected.")
                client.connect(self)

            client.receive_parameters(self.model)

    def disconnect(self, clients: Iterable[C]):
        """
        Disconnect client from server.

        :param client: The client to be disconnected.
        """
        for client in clients:
            logger.debug(f"Disconnecting client {client.id}.")
            client.disconnect()

    def train(
        self,
        clients: Iterable[C],
        *,
        concurrent: int | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> tuple[list[nn.Module], list[int]]:
        self.connect(clients)
        self.send_parameters(clients)

        logger.debug(f"Training clients in parallel with {concurrent} workers.")
        parallel_do(clients, "train", max_workers=concurrent, executor=executor)
        local_models = [client.send_parameters() for client in clients]
        sizes = [client.datasize for client in clients]
        self.disconnect(clients)
        return local_models, sizes

    def evaluate(self, concurrent: int, executor: ThreadPoolExecutor | None = None) -> CollectedClientMetrics:
        collected: CollectedClientMetrics = {}
        for clients in batched(self.registered_clients.values(), concurrent):
            self.connect(clients)
            self.send_parameters(clients)
            results = parallel_do(clients, "evaluate", max_workers=concurrent, executor=executor)
            collected |= {client.id: result for client, result in zip(clients, results)}
            self.disconnect(clients)
        return collected

    def test(self, concurrent: int, executor: ThreadPoolExecutor | None = None) -> CollectedClientMetrics:
        collected: CollectedClientMetrics = {}
        for clients in batched(self.registered_clients.values(), concurrent):
            self.connect(clients)
            self.send_parameters(clients)
            results = parallel_do(clients, "test", max_workers=concurrent, executor=executor)
            collected |= {client.id: result for client, result in zip(clients, results)}
            self.disconnect(clients)
        return collected

    def __repr__(self) -> str:
        info = f"clients={len(self.registered_clients)},select_ratio={self.select_ratio},learning_rate={self.learning_rate},robust_fn={self.robust_fn.__class__.__name__}"
        return f"<{self.__class__.__name__} [{info}]>"
