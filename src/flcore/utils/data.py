from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence, TypeVar

import more_itertools
import torch
import torch.distributions as distributions
from torch.utils.data import ConcatDataset, Dataset, Subset

__all__ = [
    "Feature",
    "Target",
    "Dataset",
    "default_get_target",
    "get_targets",
    "group_targets",
    "random_split",
    "partition_dataset",
    "generate_dirichlet_subsets",
    "generate_p_degree_subsets",
    "generate_iid_subsets",
]

T = TypeVar("T")
Feature = TypeVar("Feature")
Target = TypeVar("Target")


def default_get_target(item: tuple[Feature, Target]) -> Target | Any:
    """
    Default function that receives an item from dataset, and then return its target.

    :param item: An item from dataset.
    :return: Target of the item.
    """
    target = item[-1]
    if isinstance(target, torch.Tensor):
        return target.tolist()
    return target


def get_targets(
    dataset: Dataset,
    get_target: Callable[[Any], Target] = default_get_target,
) -> list[Target]:
    """
    Get list of targets from `dataset`.

    :param dataset: The dataset.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: A list of targets of dataset.
    """
    # dataset from torchvision may have attribute targets
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        # some targets are Tensor type
        targets = targets.tolist() if isinstance(targets, torch.Tensor) else targets
        return list(targets)

    # some user may concat torchvision dataset
    if isinstance(dataset, ConcatDataset):
        targets = []
        for subset in dataset.datasets:
            targets.extend(get_targets(subset, get_target))
        return targets

    # deal with Subset, this can accelerate the process of above when concatted subset are from the same dataset
    if isinstance(dataset, Subset):
        targets = get_targets(dataset.dataset, get_target)  # get whole targets from whole dataset
        targets = [targets[index] for index in dataset.indices]
        return targets

    # fail back to normal process, this can be slow
    targets = [get_target(item) for item in dataset]
    # cache targets to dataset
    setattr(dataset, "targets", targets)

    return targets


def group_targets(targets: Sequence[Target]) -> dict[Target, list[Target]]:
    """
    Collect and group the `targets` to a dict. In dict, the keys are targets and value are corresponding indexes.

    :param targets: A sequence of target.
    :return: Groupped targets.
    """
    target_indexes = {}
    for index, target in enumerate(targets):
        target_indexes.setdefault(target, []).append(index)
    return target_indexes


def random_split(values: Sequence[T], fractions: Sequence[float]) -> list[list[T]]:
    """
    Randomly split `values` into some list of value.

    :param fractions: Fractions of subsequence length. Should be closed to 1 enough.
    :return: A list of sublist of values.

    :raises ValueError: If lengths or fractions are not legal.
    """
    if not math.isclose(math.fsum(fractions), 1.0, rel_tol=1e-5):
        raise ValueError("Sum of fractions is not close enough to 1.")

    # fractions to lengths
    lengths = []
    for i, frac in enumerate(fractions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Fraction at index {i} is not between 0 and 1")
        length = math.floor(len(values) * frac)
        lengths.append(length)

    # sum(lengths) may not exactly equals to len(values)
    remainder = len(values) - sum(lengths)
    step = 1 if remainder >= 0 else -1

    # put remainder into lengths
    for i in range(abs(remainder)):
        lengths[i % len(lengths)] += step

    values = list(values)
    random.shuffle(values)
    split_values = list(more_itertools.split_into(values, lengths))

    return split_values


def partition_dataset(
    dataset: Dataset,
    fraction_fn: Callable[[], list[float]],
    get_target: Callable[[Any], Target] = default_get_target,
) -> list[Subset]:
    """
    Partition dataset into many subset, the size of each subset follow the return value of `fraction_fn`.

    :param dataset: The dataset.
    :param fraction_fn: A callable that return the corresponding fractions.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: A list of subsets.

    :raises ValueError: If the length of partition_func didn't return the right fractions.
    """
    targets = get_targets(dataset, get_target)
    target_indexes = group_targets(targets)
    target_subindexes = {}

    for target, indexes in target_indexes.items():
        # get the partition fractions of each class
        fractions = fraction_fn()
        target_subindexes[target] = random_split(indexes, fractions=fractions)

    # check if the length of fraction_fn didn't return the right fractions
    partition_set = set(map(len, target_subindexes.values()))
    assert len(partition_set) == 1
    partition = partition_set.pop()

    # make subsets
    subsets = []
    for _ in range(partition):
        indexes = []
        # put sub indexes of each target to a list
        for subindexes in target_subindexes.values():
            indexes.extend(subindexes.pop())

        subset = Subset(dataset, indexes)
        subsets.append(subset)

    return subsets


def generate_dirichlet_subsets(
    dataset: Dataset,
    alphas: Sequence[float],
    get_target: Callable[[Any], Target] = default_get_target,
    min_data: int = 10,
    max_retry: int = 10,
) -> list[Subset]:
    """
    Generate subsets that follow dirichlet distribution.

    :param dataset: The source dataset.
    :param alphas: The parameter of dirichlet distribution.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :param min_data: The minimal dataset size of each subset.
    :param max_retry: Max retry of sample from dirichlet distribution.
    :return: The subsets that follow dirichlet distribution.

    :raises TimeoutError: If unable to sample from dirichlet distribution within max_retry retries.
    """

    def dirichlet() -> list[float]:
        m = distributions.Dirichlet(torch.tensor(alphas, dtype=torch.float))
        sample = m.sample().tolist()
        return sample

    # generate the subsets and verify min_data was satisfied
    for _ in range(max_retry):
        subsets = partition_dataset(
            dataset,
            fraction_fn=dirichlet,
            get_target=get_target,
        )
        if all([len(subset) >= min_data for subset in subsets]):
            break
    else:
        raise TimeoutError(
            f"Unable to sample from dirichlet distributions (alpha = {alphas}) within {max_retry} retries "
            f"to satisfy that each client holds {min_data} data at least."
        )

    return subsets


def generate_p_degree_subsets(
    dataset: Dataset,
    p: float,
    num_clients: int,
    get_target: Callable[[Any], Target] = default_get_target,
) -> list[Subset]:
    """
    Generate subsets that follow p degree of non-IID. See `Local Model Poisoning Attacks to Byzantine Robust Federated
    Learning`.

    :param dataset: The source dataset.
    :param p: The parameter of p degree of non-IID.
    :param num_clients: The number of clients.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: A list of subsets.
    """
    assert 0 < p < 1

    called = 0
    group_size = num_clients // len(set(get_targets(dataset)))

    def p_degree() -> list[float]:
        nonlocal called

        lth_group = [p / group_size] * group_size
        no_lth_group = [(1 - p) / (num_clients - group_size)] * (num_clients - group_size)
        fractions = no_lth_group[: called * group_size] + lth_group + no_lth_group[called * group_size :]
        called += 1

        return fractions

    return partition_dataset(dataset, fraction_fn=p_degree, get_target=get_target)


def generate_iid_subsets(
    dataset: Dataset,
    num_clients: int,
    get_target: Callable[[Any], Target] = default_get_target,
) -> list[Subset]:
    """
    Generate subsets that follow IID.

    :param dataset: The source dataset.
    :param num_clients: The quantity of partition, namely the number of clients.
    :param get_target: A callable that receives an item from dataset and then return its target.
    :return: An IID subsets.
    """
    return partition_dataset(
        dataset,
        fraction_fn=lambda: [1 / num_clients] * num_clients,
        get_target=get_target,
    )
