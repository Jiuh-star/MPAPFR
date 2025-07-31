from __future__ import annotations

import itertools
import threading
from typing import Hashable, Sequence

import torch
from box import Box
from flcore import Client

g = Box()  # NOTE: race condition may occur here, use with caution.
lock = threading.Lock()


def get_optimal_cuda_device() -> torch.device:
    """
    This is a replacement for `flcore.utils.misc.get_optimal_cuda_device()`, which
    may cause weird large memory occupation when importing torch.cuda runtime.
    """
    with lock:
        it = g.setdefault("_device_cycle", itertools.cycle(g.devices))
        return next(it)


def to_fixed(obj, decimal=4) -> str:
    if isinstance(obj, (tuple, list)):
        return f"[{', '.join([to_fixed(o, decimal) for o in obj])}]"
    return f"{obj:.{decimal}f}"


def get_ids(clients: Sequence[Client]) -> list[Hashable]:
    return [client.id for client in clients]
