from __future__ import annotations

import inspect
import keyword
import os
import random
from typing import Callable

import numpy as np
import torch

__all__ = [
    "fix_random_seed",
    "get_optimal_cuda_device",
    "proper_call",
]


def fix_random_seed(seed: int) -> None:
    """
    Fix the random seed for reproducibility.

    :param seed: The random seed.
    """
    # fix seed of hash algo
    os.environ["PYTHONHASHSEED"] = str(seed)
    # fix seed of random module
    random.seed(seed)
    # fix seed of numpy module
    np.random.seed(seed)
    # fix seed of torch module
    torch.random.manual_seed(seed)

    # NOTE: import cuda runtime on windows will cause memory leaks whrn training with threadings.
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_optimal_cuda_device() -> torch.device:
    """
    Dynamically select the CUDA device, which has the most memory, for training.
    Support CUDA_VISIBLE_DEVICES environment variable.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    gpu_memory = []
    for gpu_id in gpu_ids:
        free, _ = torch.cuda.mem_get_info(device=torch.device(f"cuda:{gpu_id}"))
        gpu_memory.append(free)

    optimal_id = gpu_ids[np.argmax(gpu_memory)]
    return torch.device(f"cuda:{optimal_id}")


def proper_call[T](call: Callable[..., T], **kwargs) -> T:
    sig = inspect.signature(call)
    kinds = [param.kind for param in sig.parameters.values()]

    mapped_kwargs = {}
    has_var_keyword = inspect.Parameter.VAR_KEYWORD in kinds
    for k, v in kwargs.items():
        if k in keyword.kwlist:
            k = f"{k}_"

        kind = sig.parameters.get(k, None)
        kind = kind.kind if kind else inspect.Parameter.VAR_KEYWORD

        if not has_var_keyword and kind == inspect.Parameter.VAR_KEYWORD:
            continue

        mapped_kwargs[k] = v

    result = call(**mapped_kwargs)

    return result
