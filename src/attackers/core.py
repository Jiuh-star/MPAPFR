from __future__ import annotations

import copy
import time
import weakref
from functools import cached_property
from random import sample
from typing import Iterable, Sequence

import flcore
import torch
from flcore.utils import (
    aggregate_model,
    aggregate_update,
    aggregate_vector,
    model_to_vector,
)
from loguru import logger


class AttackContext[FL: flcore.FederatedLearning, S: flcore.Server, C: flcore.Client]:
    def __init__(
        self,
        system: FL,
        number: int,
        device: torch.device,
    ):
        self.system: FL = weakref.proxy(system)
        self.number = number
        self.device = device

        self.global_epoch = -1
        self.local_models: list[torch.nn.Module] = []
        self.weights: list[float] = []

        # set malicious_clients
        registered_clients = list(self.server.registered_clients.values())
        malicious_clients: list[C] = sample(registered_clients, self.number)
        self.registered_malicious_clients = malicious_clients

        self.attacked = False

    def update(
        self,
        global_epoch: int,
        selected_clients: Sequence[C],
        local_models: Sequence[torch.nn.Module],
        weights: Sequence[float],
    ):
        self.global_epoch = global_epoch
        self.selected_clients = list(selected_clients)
        self.local_models = list(local_models)
        self.weights = list(weights)

        self.attacked = False

        # clear cached properties
        if "global_model" in self.__dict__:
            del self.global_model
        if "benign_models" in self.__dict__:
            del self.benign_models
        if "reference_model" in self.__dict__:
            del self.reference_model

    def setup_strategy(self, strategy: AttackStrategy):
        self.strategy = strategy

    def inject_attackers(self, clients: Iterable[C]) -> list[C]:
        selected_clients = list(clients)

        if self.strategy.continuous:
            nm = len(self.registered_malicious_clients)
            assert nm < len(selected_clients), (
                "When enable continuous attack, the number of malicious clients"
                " must be less than the number of selected clients."
            )
            attacker_ids = [attacker.id for attacker in self.registered_malicious_clients]
            selected_benign_clients = [client for client in selected_clients if client.id not in attacker_ids]
            selected_clients = (self.registered_malicious_clients + selected_benign_clients)[: len(selected_clients)]
            logger.debug(f"Enable continuous attack, replace {nm} malicious clients to selected clients.")

        return selected_clients

    def execute_attack(self) -> tuple[list[torch.nn.Module], list[float]]:
        # execute attack only when malicious clients are selected
        if not self.selected_malicious_clients:
            logger.debug("No malicious clients selected, nothing to do.")
            return self.local_models, self.weights

        # connect malicious clients
        self.server.connect(self.registered_malicious_clients)

        # execute attack algorithm
        poisoned_model = self.strategy.execute(self)

        # disconnect malicious clients
        self.server.disconnect(self.registered_malicious_clients)

        # do nothing
        if poisoned_model is None:
            logger.debug("The attack strategy returned None, nothing to do.")
            return self.local_models, self.weights

        # replace malicious client's local model to poisoned model
        local_models = list(self.local_models)

        for i, client in enumerate(self.selected_clients):
            if client in self.selected_malicious_clients:
                local_models[i] = poisoned_model.cpu()
                logger.debug(f"Replace the local model of malicious client: {client.id} to poisoned model.")

        self.attacked = True

        return local_models, self.weights

    def evaluate_attack(self) -> None:
        # NOTE: default no connection for speeding up
        self.strategy.evaluate(self)

    @property
    def server(self) -> S:
        return self.system.server

    @property
    def robust_fn(self):
        return self.system.server.robust_fn

    @property
    def registered_benign_clients(self):
        return [
            client
            for client in self.server.registered_clients.values()
            if client not in self.registered_malicious_clients
        ]

    @property
    def selected_benign_clients(self):
        return [client for client in self.selected_clients if client not in self.registered_malicious_clients]

    @property
    def selected_malicious_clients(self):
        return [client for client in self.selected_clients if client in self.registered_malicious_clients]

    @cached_property
    def global_model(self) -> torch.nn.Module:
        return copy.deepcopy(self.server.model).to(self.device)

    @property
    def global_vector(self):
        return model_to_vector(self.global_model)

    @cached_property
    def benign_models(self):
        return [
            copy.deepcopy(model).to(self.device)
            for model, client in zip(self.local_models, self.selected_clients)
            if client not in self.registered_malicious_clients
        ]

    @property
    def benign_vectors(self):
        return [model_to_vector(model) for model in self.benign_models]

    @property
    def benign_updates(self):
        return [vector - self.global_vector for vector in self.benign_vectors]

    @property
    def benign_weights(self):
        return [
            weight
            for weight, client in zip(self.weights, self.selected_benign_clients)
            if client not in self.registered_malicious_clients
        ]

    @cached_property
    def reference_models(self):
        return [
            copy.deepcopy(model).to(self.device)
            for model, client in zip(self.local_models, self.selected_clients)
            if client in self.registered_malicious_clients
        ]

    @property
    def reference_weights(self):
        return [
            weight
            for weight, client in zip(self.weights, self.selected_clients)
            if client in self.registered_malicious_clients
        ]

    @cached_property
    def reference_model(self):
        return aggregate_model(
            global_model=copy.deepcopy(self.global_model),
            local_models=self.reference_models,
            weights=self.reference_weights,
        )

    @property
    def reference_vector(self):
        return aggregate_vector(
            global_vector=self.global_vector,
            local_vectors=self.benign_vectors,
            weights=self.benign_weights,
        )

    @property
    def reference_update(self):
        return aggregate_update(
            updates=self.benign_updates,
            weights=self.benign_weights,
        )

    def reference_dataloader(self):
        for client in self.registered_malicious_clients:
            for x, y in client.train_dataloader:
                yield x, y


class AttackStrategy:
    def __init__(
        self,
        attack_range: range | None = None,
        once: bool = False,
        threshold: int = -1,
        continuous: bool = False,
        **kwargs,
    ):
        self.range = attack_range if attack_range is not None else range(9999)
        self.once = once
        self.threshold = threshold
        self.continuous = continuous
        self.attacked = False

    def attack_algorithm(self, ctx: AttackContext) -> torch.nn.Module | None:
        pass

    def evaluate(self, ctx: AttackContext) -> None:
        pass

    def execute(self, ctx: AttackContext) -> torch.nn.Module | None:
        flag = False

        # some simple attack strategy that might be useful

        # continuous attack check
        if self.continuous:
            flag = True

        # threshold check
        if len(ctx.selected_malicious_clients) < self.threshold:
            flag = False

        # attack once check
        if self.once and self.attacked:
            flag = False

        # attack range check
        if ctx.global_epoch not in self.range:
            flag = False

        if flag is False:
            return

        # execute the attack algorithm and measure the time
        logger.debug(f"Execute attack algorithm: {self.__class__.__name__}")
        start_time = time.perf_counter()

        poisoned_model = self.attack_algorithm(ctx)

        end_time = time.perf_counter()
        logger.debug(f"Attack algorithm executed in {end_time - start_time:.4f}s")

        return poisoned_model
