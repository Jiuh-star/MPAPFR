from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from flcore.robust import (
    Krum,
    NormBound,
    RobustFn,
    RobustFnReturn,
)
from flcore.utils import model_to_vector

__all__ = ["Krum","NormBound", "Mesas"]


class Mesas(RobustFn):
    def __init__(self, stt_conf: float, stv_conf: float, std_conf: float) -> None:
        super().__init__()
        self.stt_conf = stt_conf
        self.stv_conf = stv_conf
        self.std_conf = std_conf

    def __call__(
        self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float]
    ) -> RobustFnReturn:
        checkers = {
            "EUCL": self.get_magnitudes,
            "COS": self.get_directions,
            "COUNT": self.get_orientations,
            "VAR": self.get_vars,
            "MIN": self.get_mins,
            "MAX": self.get_maxs,
        }

        # NOTE: Full code will be submitted to the repository once the paper is accepted

        return local_models, weights

    def get_magnitudes(self, global_model: nn.Module, local_models: Sequence[nn.Module]) -> list[float]:
        vector = model_to_vector(global_model)
        local_vectors = (model_to_vector(model) for model in local_models)

        return [local_vector.dist(vector).item() for local_vector in local_vectors]

    def get_directions(self, global_model: nn.Module, local_models: Sequence[nn.Module]) -> list[float]:
        vector = model_to_vector(global_model)
        local_vectors = (model_to_vector(model) for model in local_models)

        return [nn.functional.cosine_similarity(local_vector, vector, dim=0).item() for local_vector in local_vectors]

    def get_orientations(self, global_model: nn.Module, local_models: Sequence[nn.Module]) -> list[float]:
        vector = model_to_vector(global_model)
        local_vectors = (model_to_vector(model) for model in local_models)

        return [nn.functional.relu((local_vector - vector).sign()).sum().item() for local_vector in local_vectors]

    def get_vars(self, global_model: nn.Module, local_models: Sequence[nn.Module]) -> list[float]:
        local_vectors = (model_to_vector(model) for model in local_models)
        return [local_vector.var().item() for local_vector in local_vectors]

    def get_mins(self, global_model: nn.Module, local_models: Sequence[nn.Module]) -> list[float]:
        vector = model_to_vector(global_model)
        local_vectors = (model_to_vector(model) for model in local_models)
        diffs = ((local_vector - vector).abs() for local_vector in local_vectors)
        return [diff[diff != 0].min().item() for diff in diffs]

    def get_maxs(self, global_model: nn.Module, local_models: Sequence[nn.Module]) -> list[float]:
        vector = model_to_vector(global_model)
        local_vectors = (model_to_vector(model) for model in local_models)
        return [(local_vector - vector).abs().max().item() for local_vector in local_vectors]
