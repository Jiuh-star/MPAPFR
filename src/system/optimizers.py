from typing import Iterable

import torch
import torch.optim as optim


class PerturbedGradientDescent(optim.Optimizer):
    def __init__(self, params, lr: float, mu: float):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params: Iterable, device: torch.device):
        for group in self.param_groups:
            for p, g in zip(group["params"], global_params):
                g = g.to(device)
                d_p = p.grad.data + group["mu"] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group["lr"])
