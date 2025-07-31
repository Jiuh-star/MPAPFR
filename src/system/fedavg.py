from __future__ import annotations

import copy
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Hashable, NamedTuple, Self, override

import torch
import torch.nn as nn
import torch.optim as optim
from box import Box
from flcore import (
    Client,
    ClientData,
    ClientMetrics,
    ClientModel,
    FederatedLearning,
    FlDataset,
    Server,
)
from flcore.utils import dump, load, proper_call
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric, MetricCollection

from . import optimizers
from .helper import g, get_ids, get_optimal_cuda_device, to_fixed

if TYPE_CHECKING:
    from attackers import AttackContext


class ClientStateDict(NamedTuple):
    optimizer: dict
    criterion: dict
    personalized_model: dict | None
    personalized_optimizer: dict | None
    personalized_criterion: dict | None


class FedAvgClient(Client):
    personalized = False

    def __init__(
        self,
        id: Hashable,
        workdir: str | Path,
        max_epoch: int,
        optimizer: Box,
        criterion: Box,
        personalized_max_epoch: int | None = None,
        personalized_optimizer: Box | None = None,
        personalized_criterion: Box | None = None,
        **kwargs,
    ):
        super().__init__(id=id, workdir=workdir)
        self.args = Box(
            max_epoch=max_epoch,
            optimizer=optimizer,
            criterion=criterion,
            personalized_max_epoch=personalized_max_epoch,
            personalized_optimizer=personalized_optimizer,
            personalized_criterion=personalized_criterion,
        )
        self.init(**kwargs)

    def init(self, **kwargs):
        pass

    @override
    def load_data(self, server: Server) -> ClientData:
        fl_dataset: FlDataset = g.fl_dataset
        self.num_class = fl_dataset.num_class
        self.metrics = MetricCollection(
            {
                "loss": MeanMetric(nan_strategy="error"),
                "accuracy": Accuracy("multiclass", num_classes=self.num_class, average="micro"),
            }
        )

        if self.personalized:
            self.personalized_metrics = MetricCollection(
                {
                    "per_loss": MeanMetric(nan_strategy="error"),
                    "per_accuracy": Accuracy("multiclass", num_classes=self.num_class, average="micro"),
                }
            )

        return fl_dataset.id_data[self.id]

    @override
    def load_model(self, server: Server) -> ClientModel:
        optimal_device = get_optimal_cuda_device()
        model = copy.deepcopy(server.model).to(optimal_device)
        # load optimizer from two sources
        if hasattr(optim, self.args.optimizer.type):
            optimizer_type = getattr(optim, self.args.optimizer.type)
        else:
            optimizer_type = getattr(optimizers, self.args.optimizer.type)
        optimizer = proper_call(optimizer_type, params=model.parameters(), **self.args.optimizer)
        criterion_type = getattr(nn, self.args.criterion.type)
        criterion = proper_call(criterion_type, **self.args.criterion)

        # trace the finalizer
        weakref.finalize(model, logger.debug, f"[{self.id}:{optimal_device}] Removed model.")
        weakref.finalize(optimizer, logger.debug, f"[{self.id}:{optimal_device}] Removed optimizer.")
        weakref.finalize(criterion, logger.debug, f"[{self.id}:{optimal_device}] Removed criterion.")

        if self.personalized:
            per_model = copy.deepcopy(server.model).to(optimal_device)

            # load optimizer from two sources
            if hasattr(optim, self.args.personalized_optimizer.type):
                per_optimizer_type = getattr(optim, self.args.personalized_optimizer.type)
            else:
                per_optimizer_type = getattr(optimizers, self.args.personalized_optimizer.type)

            per_optimizer = proper_call(
                per_optimizer_type,
                params=per_model.parameters(),
                **self.args.personalized_optimizer,
            )
            per_criterion_type = getattr(nn, self.args.personalized_criterion.type)
            per_criterion = proper_call(per_criterion_type, **self.args.personalized_criterion)

            # trace the finalizer
            weakref.finalize(
                per_model,
                logger.debug,
                f"[{self.id}:{optimal_device}] Removed personalized_model.",
            )
            weakref.finalize(
                per_optimizer,
                logger.debug,
                f"[{self.id}:{optimal_device}] Removed personalized_optimizer.",
            )
            weakref.finalize(
                per_criterion,
                logger.debug,
                f"[{self.id}:{optimal_device}] Removed personalized_criterion.",
            )
        else:
            per_model = None
            per_optimizer = None
            per_criterion = None

        # load state_dict if exists
        if state_dict := self.load_state_dict(raise_error=False):
            # we restore the state dict for optimzier for better training. e.g., momentum, running_var, etc.
            optimizer.load_state_dict(state_dict.optimizer)
            criterion.load_state_dict(state_dict.criterion)

            if self.personalized:
                assert per_model and state_dict.personalized_model is not None
                assert per_optimizer and state_dict.personalized_optimizer is not None
                assert per_criterion and state_dict.personalized_criterion is not None

                per_model.load_state_dict(state_dict.personalized_model)
                per_optimizer.load_state_dict(state_dict.personalized_optimizer)
                per_criterion.load_state_dict(state_dict.personalized_criterion)

            del state_dict

        client_model = ClientModel(
            device=optimal_device,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            personalized_model=per_model,
            personalized_optimizer=per_optimizer,
            personalized_criterion=per_criterion,
        )

        return client_model

    @override
    def disconnect(self):
        self.save_state_dict()
        super().disconnect()

    def load_state_dict(self, raise_error: bool = True) -> ClientStateDict:
        return load(self.workdir / "model.pkl", raise_error=raise_error)

    def save_state_dict(self):
        state_dict = ClientStateDict(
            optimizer=self.optimizer.state_dict(),
            criterion=self.criterion.state_dict(),
            personalized_model=(self.personalized_model.state_dict() if self.personalized_model else None),
            personalized_optimizer=(self.personalized_optimizer.state_dict() if self.personalized_optimizer else None),
            personalized_criterion=(self.personalized_criterion.state_dict() if self.personalized_criterion else None),
        )

        dump(state_dict, self.workdir / "model.pkl", replace=True)

        del state_dict

    @override
    def on_train(self):
        self.model.train()

        for i in range(self.args.max_epoch):
            losses = []
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.detach().item())

            avg_loss = sum(losses) / len(losses)
            logger.debug(f"[{self.id}:{self.device}:{i}/{self.args.max_epoch - 1}] Training loss: {avg_loss}")

            del losses

    @torch.inference_mode()
    def _eval(self, dataloader: DataLoader, eval_personal: bool = False) -> ClientMetrics:
        if eval_personal and self.personalized:
            metrics = self.personalized_metrics
            model = self.personalized_model
            criterion = self.personalized_criterion
            assert model
            assert criterion
        else:
            metrics = self.metrics
            model = self.model
            criterion = self.criterion

        metrics.reset()
        metrics.to(self.device)
        model.eval()

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            metrics.update(value=loss, target=y, preds=y_hat)

        eval_result = {name: value.item() for name, value in metrics.compute().items()}

        logger.debug(f"[{self.id}:{self.device}] Evaluation result: {eval_result}")

        metrics.reset()  # incase memory leak

        return eval_result

    @override
    def on_evaluation(self) -> ClientMetrics:
        eval_result = self._eval(self.validation_dataloader)
        if self.personalized:
            eval_result |= self._eval(self.validation_dataloader, eval_personal=True)

        return eval_result

    @override
    def on_test(self) -> ClientMetrics:
        eval_result = self._eval(self.test_dataloader)
        if self.personalized:
            eval_result |= self._eval(self.test_dataloader, eval_personal=True)

        return eval_result


class FedAvgServer[C: FedAvgClient](Server[C]):
    pass


class FedAvg(FederatedLearning[FedAvgServer[FedAvgClient]]):
    def __init__(
        self,
        *,
        server: FedAvgServer,
        logdir: str | Path,
        max_epoch: int,
        tensorboard: bool = True,
        eval_interval: int = 1,
        concurrent: int | None = None,
    ) -> None:
        super().__init__(server=server, logdir=logdir, max_epoch=max_epoch, tensorboard=tensorboard)
        self.eval_interval = eval_interval
        self.concurrent = concurrent or int(self.server.select_ratio * len(self.server.registered_clients))
        self.executor = ThreadPoolExecutor(max_workers=self.concurrent, thread_name_prefix="ThreadedClient")

    def setup_context(self, ctx: AttackContext) -> None:
        self.ctx: AttackContext[Self, FedAvgServer, FedAvgClient] = ctx

    @override
    def algorithm(self) -> nn.Module:
        logger.info(f"[BEGIN] Start of {self.__class__.__name__} algorithm.")
        logger.info(f"[BEGIN] The malicious clients are {get_ids(self.ctx.registered_malicious_clients)}")

        for global_epoch in range(self.max_epoch):
            prefix = f"[{global_epoch}/{self.max_epoch - 1}]"

            # select client to participate in the global training
            selected_clients = self.server.select_clients()
            selected_clients = self.ctx.inject_attackers(selected_clients)

            # train the selected clients
            logger.info(f"{prefix} Training the selected clients: {get_ids(selected_clients)}")
            local_models, sizes = self.server.train(
                selected_clients, concurrent=self.concurrent, executor=self.executor
            )
            weights = [size / sum(sizes) for size in sizes]
            logger.debug(f"{prefix} Training done, weights: {to_fixed(weights)}")

            # execute the attack strategy
            logger.debug(f"{prefix} Executing the attack strategy.")
            self.ctx.update(
                global_epoch=global_epoch,
                selected_clients=selected_clients,
                local_models=local_models,
                weights=weights,
            )
            local_models, weights = self.ctx.execute_attack()

            if self.ctx.attacked:
                logger.info(
                    f"{prefix} Successfully executed the attack strategy, "
                    f"{len(self.ctx.selected_malicious_clients)} malicious clients submitted the poisoned models."
                )
            else:
                logger.info(f"{prefix} No attack executed.")

            # aggregate the local models
            logger.debug(f"{prefix} Aggregating the local models.")
            self.server.aggregate(local_models, weights)

            # evaluate the global model
            if global_epoch % self.eval_interval == 0:
                logger.info(f"{prefix} Evaluating the global model.")
                collected_metrics = self.server.evaluate(self.concurrent, self.executor)

                # log the evaluation metrics
                brief_metrics = self.to_brief_metrics(collected_metrics)
                for name, value in brief_metrics.items():
                    logger.info(f"{prefix} Evaluation metric: {name} = {to_fixed(value)}")
                self.save_metrics(global_epoch, collected_metrics, "val.jsonl")

                self.ctx.evaluate_attack()

        # test the global model
        logger.info("[END] Testing the global model.")

        collected_metrics = self.server.test(self.concurrent, self.executor)
        self.save_metrics(global_epoch + 1, collected_metrics, "test.json")
        brief_metrics = self.to_brief_metrics(collected_metrics)
        for name, value in brief_metrics.items():
            logger.info(f"[END] Test metric: {name} = {value}")

        # return the trained model
        logger.info(f"[END] End of {self.__class__.__name__} algorithm.")
        return self.server.model
