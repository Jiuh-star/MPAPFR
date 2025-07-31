from __future__ import annotations

from typing import override

from flcore import ClientMetrics
from flcore.utils import move_parameters
from loguru import logger

from .fedavg import FedAvg, FedAvgClient, FedAvgServer


class FedAvgFTClient(FedAvgClient):
    personalized = True

    def finetune(self):
        assert self.personalized_model
        assert self.personalized_optimizer
        assert self.personalized_criterion
        assert self.personalized_metrics

        # load the global model
        move_parameters(self.model, self.personalized_model, buffer=True)

        self.personalized_model.train()

        for i in range(self.args.personalized_max_epoch):
            losses = []
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.personalized_optimizer.zero_grad()
                y_hat = self.personalized_model(x)
                loss = self.personalized_criterion(y_hat, y)
                loss.backward()
                self.personalized_optimizer.step()

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            logger.debug(
                f"[{self.id}:{self.device}:{i}/{self.args.personalized_max_epoch - 1}] Personalized training loss: {avg_loss}"
            )

            del losses

    @override
    def on_evaluation(self) -> ClientMetrics:
        self.finetune()
        return super().on_evaluation()

    @override
    def on_test(self) -> ClientMetrics:
        self.finetune()
        return super().on_test()


class FedAvgFTServer(FedAvgServer[FedAvgFTClient]):
    pass


class FedAvgFT(FedAvg):
    pass
