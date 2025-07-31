from __future__ import annotations

import copy
import typing as T
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .utils import model as model_utils

RobustFnReturn: T.TypeAlias = tuple[list[nn.Module], list[float]]


class RobustFn(ABC):
    """
    The abstract class of the robust aggregation function.
    """

    @abstractmethod
    def __call__(
        self,
        global_model: nn.Module,
        local_models: T.Sequence[nn.Module],
        weights: T.Sequence[float],
    ) -> RobustFnReturn: ...


class Krum(RobustFn):
    def __init__(self, *, num_remove: int):
        """
        The Krum robust aggregation function.

        :param num_remove: How many models to remove.
        """
        assert num_remove > 0
        self.num_remove = num_remove

    def __call__(
        self,
        global_model: nn.Module,
        local_models: T.Sequence[nn.Module],
        weights: T.Sequence[float],
    ) -> RobustFnReturn:
        if self.num_remove >= len(local_models):
            raise ValueError(f"Can not remove {self.num_remove} models when there are only {len(local_models)} models.")

        if 2 * self.num_remove + 2 >= len(local_models):
            warnings.warn(
                f"There are only {len(local_models)} models, "
                f"which not satisfy Krum/MultiKrum needs {2 * self.num_remove + 2}."
            )

        local_models = list(local_models)
        weights = list(weights)
        local_vectors = [model_utils.model_to_vector(model) for model in local_models]

        # multi-krum
        num_select = len(local_models) - self.num_remove
        selected_models = []
        selected_weights = []
        for _ in range(num_select):
            index = self._krum(local_vectors)
            local_vectors.pop(index)
            selected_models.append(local_models.pop(index))
            selected_weights.append(weights.pop(index))

        return selected_models, selected_weights

    def _krum(self, vectors: T.Sequence[torch.Tensor]) -> int:
        # Calculate distance between any two vectors
        distances = [torch.stack([vector.dist(other) for other in vectors]) for vector in vectors]

        # Calculate their scores
        num_select = len(vectors) - self.num_remove - 2
        # torch.sort() return a 2-element tuple, we only need 0th.
        # The 0th is the distance between itself in the sorted distances
        scores = [distance.sort()[0][1 : num_select + 1].sum() for distance in distances]

        # Select the minimal
        index = int(torch.tensor(scores).argmin().item())

        return index


class NormBound(RobustFn):
    def __init__(self, *, threshold: float):
        """
        The NormBound robust aggregation function.

        :param threshold: The threshold of the norm of the gradient.
        """
        assert threshold > 0
        self.threshold = threshold

    def __call__(
        self,
        global_model: nn.Module,
        local_models: T.Sequence[nn.Module],
        weights: T.Sequence[float],
    ) -> RobustFnReturn:
        global_vector = model_utils.model_to_vector(global_model)

        models = []
        for model in local_models:
            vector = model_utils.model_to_vector(model)
            update = vector - global_vector
            update = update / max(1, update.norm().item() / self.threshold)

            model = copy.deepcopy(model)
            model_utils.vector_to_model(global_vector + update, model)
            models.append(model)

        return models, list(weights)
