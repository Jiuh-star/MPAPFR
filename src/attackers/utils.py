from __future__ import annotations

import copy
import random
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Self

import torch
from box import Box
from flcore import Client, FederatedLearning, FlDataset, Server
from flcore.utils import proper_call
from loguru import logger
from system.helper import g
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as tvf

if TYPE_CHECKING:
    from .core import AttackContext


def normalize(batch: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    return tvf.normalize(batch, mean, std)


def denormalize(batch: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    batch_mean = torch.tensor(mean).view(-1, 1, 1).to(batch.device)
    batch_std = torch.tensor(std).view(-1, 1, 1).to(batch.device)
    batch = batch * batch_std + batch_mean
    return batch


class BackdoorDataLoader:
    # NOTE: The trigger pattern is from *BapFL: You can Backdoor Personalized Federated Learning*
    pixel_position = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 6],
        [3, 7],
        [3, 8],
        [3, 9],
    ]

    # "racing stripe": [2180,2771,3233,4932,6241,6813,6869,9476,11395,11744,14209,14238,18716,
    #                   19793,20781,21529,31311,40518,40633,42119,42663,49392]
    # "green cars": [389,561,874,1605,3378,3678,4528,9744,19165,19500,21422,22984,32941,34287,34385,
    #                36005,37365,37533,38658,38735,39824,40138,41336,41861,47001,47026,48003,48030,
    #                49163,49588]
    # "vertical stripes": [330,568,3934,12336,30560,30696,33105,33615,33907,36848,40713,41706]
    cifar10_trigger_indices = [
        389,
        561,
        874,
        1605,
        3378,
        3678,
        4528,
        9744,
        19165,
        19500,
        21422,
        22984,
        32941,
        34287,
        34385,
        36005,
        37365,
        37533,
        38658,
        38735,
        39824,
        40138,
        41336,
        41861,
        47001,
        47026,
        48003,
        48030,
        49163,
        49588,
    ]

    def __init__(
        self,
        trigger: Literal["pixel", "physical-cifar10"] | str,
        target: Any,
        fraction: float,
        dataloader: DataLoader | Callable[[], DataLoader | Generator],
        normalize_fn: Callable[[torch.Tensor], torch.Tensor],
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
        fl_dataset: FlDataset,
    ):
        self._dataloader = dataloader
        self.trigger = trigger
        self.fraction = fraction
        self.target = target
        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn

        self._fl_dataset = fl_dataset

        self.replace_history = []

    def __iter__(self):
        yield from self.train_dataloader()

    def dataloader(self):
        if callable(self._dataloader):
            return self._dataloader()
        return self._dataloader

    def train_dataloader(self):
        for data in self.dataloader():
            backdoor_indices = random.sample(range(len(data[0])), int(len(data[0]) * self.fraction))
            x, y = data

            if backdoor_indices:
                self.replace_history.append(backdoor_indices)

                for i in backdoor_indices:
                    x[i] = self.trigger_fn(x[i])
                    y[i] = self.target

            yield data, backdoor_indices

    def test_dataloader(self):
        for x, y in self.dataloader():
            for i in range(len(x)):
                x[i] = self.trigger_fn(x[i])
                y[i] = self.target

            yield x, y

    def get_data(self, index: int) -> Dataset:
        assert "root_dataset" in self._fl_dataset.extras, '"root_dataset" is required in FlDataset.extras.'
        root_dataset = self._fl_dataset.extras["root_dataset"]
        return root_dataset[index]

    def trigger_fn(self, batch: torch.Tensor) -> torch.Tensor:
        match self.trigger:
            case "pixel":
                return self.add_pixel_trigger(batch)
            case "physical-cifar10":
                return self.add_physical_cifar10_trigger(batch)
            case _:
                raise ValueError(f"Unknown trigger type: {self.trigger}")

    def add_pixel_trigger(self, batch: torch.Tensor) -> torch.Tensor:
        height, width = batch.shape[-2:]
        pattern = torch.zeros(height, width)
        for y, x in self.pixel_position:
            pattern[y, x] = 1.0
        pattern = pattern.reshape(-1, height, width)

        batch = self.denormalize_fn(batch)
        batch = batch.add(pattern).clamp(0, 1).to(batch.device)
        batch = self.normalize_fn(batch)

        return batch

    def add_physical_cifar10_trigger(self, batch: torch.Tensor) -> torch.Tensor:
        index = random.choice(self.cifar10_trigger_indices)
        backdoor = self.get_data(index)[0].to(batch.device)
        return backdoor

    def debug(self) -> None:
        import matplotlib.pyplot as plt
        from torchvision.transforms.functional import to_pil_image
        from torchvision.utils import make_grid

        for (x, y), _ in self.test_dataloader():
            history = self.replace_history[-1]
            fmt_y = ", ".join([f"<red>{y_}</red>" if i in history else str(y_) for i, y_ in enumerate(y.tolist())])
            logger.opt(colors=True).debug(f"Targets: [{fmt_y}], Backdoor indices: {history}")

            x = self.denormalize_fn(x)
            grid = make_grid(x)
            img = to_pil_image(grid)
            plt.imshow(img)

            plt.show()

    @classmethod
    def from_config_and_context(cls, config: dict | Box, ctx: AttackContext) -> Self:
        config = Box(config)

        normalize_fn = partial(normalize, mean=config.mean, std=config.std)
        denormalize_fn = partial(denormalize, mean=config.mean, std=config.std)
        fl_dataset = g.fl_dataset
        dataloader = ctx.reference_dataloader

        return proper_call(
            cls,
            **config
            | {
                "normalize_fn": normalize_fn,
                "denormalize_fn": denormalize_fn,
                "fl_dataset": fl_dataset,
                "dataloader": dataloader,
            },
        )


@torch.no_grad()
def evaluate(ctx: AttackContext[FederatedLearning, Server, Client], dataloader_args: Box) -> None:
    # evaluation strategy #1: evaluate the trigger-injected data on maliclious clients and measure the ASR.
    backdoor_dataloader = BackdoorDataLoader.from_config_and_context(dataloader_args, ctx)

    ctx.server.connect(ctx.registered_malicious_clients)

    model = copy.deepcopy(ctx.server.model).to(ctx.device)
    model.eval()

    correct, total = 0, 0
    for x, y in backdoor_dataloader.test_dataloader():
        x, y = x.to(ctx.device), y.to(ctx.device)
        y_hat = model(x)
        y_hat = torch.max(y_hat, dim=1).indices
        correct += (y_hat == y).sum().item()
        total += y.size(0)

    asr = correct / total if total != 0 else 0
    logger.info(f"[{ctx.global_epoch}/{ctx.system.max_epoch}] ASR#1: {asr:.4%}")

    ctx.server.disconnect(ctx.registered_malicious_clients)

    # evaluation strategy #2: evaluate the side-effect of the backdoor attack on the benign clients. i.e.., the accuracy of the benign clients.
    # This can be calculated in the val.jsonl
    pass

    # evaluation strategy #3: evaluate on the personalized model
    correct, total = 0, 0
    for client in ctx.registered_benign_clients:
        client: Client
        client.connect(ctx.server)

        model = client.personalized_model if client.personalized_model is not None else client.model
        model.eval()

        # I know, ugly, but I just get tired.
        backdoor_dataloader._dataloader = client.test_dataloader  # type: ignore
        for x, y in backdoor_dataloader.test_dataloader():
            x, y = x.to(ctx.device), y.to(ctx.device)
            y_hat = model(x)
            y_hat = torch.max(y_hat, dim=1).indices
            correct += (y_hat == y).sum().item()
            total += len(y)

        client.disconnect()

    asr = correct / total if total != 0 else 0
    logger.info(f"[{ctx.global_epoch}/{ctx.system.max_epoch}] ASR#3: {asr:.4%}")
