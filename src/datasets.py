from typing import Any, Sequence, Sized


import torchvision as tv
from flcore.client import ClientData
from flcore.system import FlDataset
from flcore.utils.data import (
    generate_dirichlet_subsets,
    generate_p_degree_subsets,
    get_targets,
)
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    random_split,
)


def _tv_dataset(
    dataset: Dataset,
    name: str,
    num_clients: int,
    min_data: int = 40,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    alpha: float | None = 1.0,
    p_degree: float | None = None,
    dataloader_args: dict[str, Any] | None = None,
) -> FlDataset:
    num_classes = len(set(get_targets(dataset)))
    dataloader_args = dataloader_args or {}

    if p_degree:
        subsets = generate_p_degree_subsets(dataset, p_degree, num_clients=num_clients)
    elif alpha:
        subsets = generate_dirichlet_subsets(
            dataset,
            alphas=[alpha] * num_clients,
            min_data=min_data,
            max_retry=100,
        )
    else:
        raise ValueError("Either alpha or p_degree must be provided.")

    pfl_dataset = _generate_fl_dataset(
        subsets=subsets,
        name=name,
        num_classes=num_classes,
        fractions=fractions,
        dataloader_args=dataloader_args,
        extras={"root_dataset": dataset},
    )

    return pfl_dataset


def _generate_fl_dataset(
    subsets: Sequence[Dataset],
    name: str,
    num_classes: int,
    fractions: tuple[float, float, float],
    dataloader_args: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> FlDataset:
    id_data = {}
    extras = extras or {}
    for i, subset in enumerate(subsets):
        assert isinstance(subset, Sized)

        train_set, vali_set, test_set, *_ = random_split(subset, fractions)

        train_dataloader = DataLoader(train_set, **dataloader_args)
        validation_dataloader = DataLoader(vali_set, **dataloader_args)
        test_dataloader = DataLoader(test_set, **dataloader_args)

        id_data[i] = ClientData(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            test_dataloader=test_dataloader,
            datasize=len(subset),
        )

    pfl_dataset = FlDataset(
        id_data=id_data,
        name=name,
        num_class=num_classes,
        fractions=fractions,
        extras=extras,
    )

    return pfl_dataset


def cifar10(
    num_client: int,
    resize: tuple[int, int] = (32, 32),
    min_data: int = 40,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    alpha: float | None = 1.0,
    p_degree: float | None = None,
    dataloader_args: dict[str, Any] | None = None,
) -> FlDataset:
    name = "CIFAR-10"
    dataloader_args = dataloader_args or {}
    transforms = tv.transforms.Compose(
        [
            tv.transforms.Resize(resize, antialias=True),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = ConcatDataset(
        [
            tv.datasets.CIFAR10("data", download=True, train=True, transform=transforms),
            tv.datasets.CIFAR10("data", download=True, train=False, transform=transforms),
        ]
    )

    return _tv_dataset(
        dataset=dataset,
        name=name,
        num_clients=num_client,
        min_data=min_data,
        fractions=fractions,
        alpha=alpha,
        p_degree=p_degree,
        dataloader_args=dataloader_args,
    )
