from __future__ import annotations

import copy
import random
from typing import Iterable, override

import flcore
import torch
import torch.nn as nn
import torch.optim as optim
from flcore.utils import extract_features, model_to_vector, vector_to_model
from loguru import logger
from models import HEAD_NAME

from .core import AttackContext, AttackStrategy


class MpapfrOptimizer(optim.Optimizer):
    """Mask & Poison Concealment"""

    def __init__(self, params, lr: float, lambda_: float, beta: float = 0.5):
        default = {"lr": lr, "lambda": lambda_, "beta": beta}
        super().__init__(params, default)

    @torch.no_grad()
    def step(
        self, device: torch.device, ref_per_params: Iterable, ref_global_params: Iterable, masks: Iterable[torch.Tensor]
    ):
        for group in self.param_groups:
            for param, g, p, m in zip(group["params"], ref_global_params, ref_per_params, masks):
                if param.grad is None:
                    continue

                zeros = torch.zeros_like(param.grad.data)

                # poison cealment
                g, p = g.to(device), p.to(device)
                d_p = param.grad.data + group["lambda"] * (
                    (1 - group["beta"]) * (param.data - g.data) + group["beta"] * (param.data - p.data)
                )

                # apply the mask to the gradietns
                d_p = torch.where(m, d_p, zeros)

                param.data.add_(d_p, alpha=-group["lr"])


class MpapfrAttackStrategy(AttackStrategy):
    def __init__(
        self,
        lr: float,  # just let it same as the benign
        max_epoch: int,  # has edge effect
        max_scale: float,  # the larger, the more aggressive, but easier to be detected
        normalize: bool,  # for contrastive loss, True is better in most cases
        beta: float,  # adjust the balance between the two distance losses
        lambda_: float,  # the weight of poison concealment
        alpha: float,  # adjust the balance between the core loss and assist loss
        mask_ratio: float,  # the ratio of the mask
        warmup: int,  # warm up steps before poison, to balance the contrastive loss and poison loss
        core_index: int,  # the malicious client in a priority "group" as core attacker to optimize the negative loss in poison loss
        invert_mask: bool = False,  # invert the mask, for full model-sharing
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.max_epoch = max_epoch
        self.max_scale = max_scale
        self.normalize = normalize
        self.beta = beta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.warmup = warmup
        self.mask_ratio = mask_ratio
        self.core_index = core_index
        self.invert_mask = invert_mask

    @override
    def attack_algorithm(self, ctx: AttackContext) -> nn.Module | None:
        # NOTE: The implementation only support PFL

        # NOTE: With core attacker (demoted client) and assist attackers (promoted clients),
        # we can make the implementation much simpler.
        self.core_id = ctx.registered_malicious_clients[self.core_index].id
        core = self.get_core(ctx)
        logger.debug(f"[Attacker]: Core Attacker {core.id}.")

        poisoned_model = copy.deepcopy(ctx.global_model)
        ref_per_model = copy.deepcopy(core.personalized_model)
        criterion = nn.CrossEntropyLoss()

        assert ref_per_model  # just for typing hint

        # ========== Poison Concealment ==========
        optimizer = MpapfrOptimizer(poisoned_model.parameters(), lr=self.lr, lambda_=self.lambda_, beta=self.beta)

        # ========== Mask Construction ==========
        # NOTE: Both work. Due to not continuous back propagation,
        # option 1 requires more training tricks (e.g., AdamW, warmup, more epochs, etc.)

        # option 1: mask for each layer
        # masks = self.construct_mask_1(ctx)
        # option 2: mask for all parameters
        masks = self.construct_mask_2(ctx)

        # ========== Feature Separation ==========

        # NOTE: Full code will be submitted to the repository once the paper is accepted

        return poisoned_model

    def get_core(self, ctx: AttackContext) -> flcore.Client:
        core = [client for client in ctx.registered_malicious_clients if client.id == self.core_id][0]
        core.model.to(ctx.device)
        core.personalized_model.to(ctx.device)
        return core

    def core_dataloader(self, ctx: AttackContext):
        for x, y in self.get_core(ctx).train_dataloader:
            yield x, y

    def assist_dataloader(self, ctx: AttackContext):
        assists = [client for client in ctx.registered_malicious_clients if client.id != self.core_id]
        random.shuffle(assists)  # so that the assist attackers are not in a fixed order

        for client in assists:
            for x, y in client.train_dataloader:
                yield x, y

    @torch.no_grad()
    def construct_mask_2(self, ctx: AttackContext) -> list[torch.Tensor]:
        ...

    @torch.no_grad()
    def construct_mask_1(self, ctx: AttackContext) -> list[torch.Tensor]:
        ...

    def separate_features(
        self,
        ctx: AttackContext,
        optimizer: MpapfrOptimizer,
        poisoned_model: nn.Module,
        ref_per_model: nn.Module,
        masks: list[torch.Tensor],
    ) -> float:
        ...

    def optimize_poison(
        self,
        ctx: AttackContext,
        optimizer: MpapfrOptimizer,
        poisoned_model: nn.Module,
        ref_per_model: nn.Module,
        criterion: nn.Module,
        masks: list[torch.Tensor],
    ) -> float:
       ...

