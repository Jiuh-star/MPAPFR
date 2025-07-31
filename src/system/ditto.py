from __future__ import annotations

from typing import override

from loguru import logger

from .fedavg import FedAvg, FedAvgClient, FedAvgServer
from .optimizers import PerturbedGradientDescent


class DittoClient(FedAvgClient):
    """
    Ditto: Fair and Robust Federated Learning Through Personalization (ICML, 2021)

    The global model follows the FedAvg algorithm.
    """
    personalized = True

    @override
    def on_train(self):
        # train personalized model
        assert self.personalized_model
        assert self.personalized_optimizer
        assert self.personalized_criterion
        assert self.personalized_metrics
        assert isinstance(self.personalized_optimizer, PerturbedGradientDescent)

        self.personalized_model.train()

        for i in range(self.args.personalized_max_epoch):
            losses = []
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.personalized_optimizer.zero_grad()
                y_hat = self.personalized_model(x)
                loss = self.personalized_criterion(y_hat, y)
                loss.backward()
                # self.model synced with global model
                self.personalized_optimizer.step(self.model.parameters(), self.device)

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            logger.debug(
                f"[{self.id}:{self.device}:{i}/{self.args.personalized_max_epoch - 1}] Personalized training loss: {avg_loss}"
            )

            del losses

        # train local model
        super().on_train()


class DittoServer(FedAvgServer[DittoClient]):
    pass


class Ditto(FedAvg):
    pass
