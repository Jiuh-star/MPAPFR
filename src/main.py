import sys
from pathlib import Path
from typing import Annotated, Optional

import toml
from box import Box
from loguru import logger
from rich.pretty import pprint
from typer import Option, Typer, confirm, secho
from typer.colors import GREEN, RED, YELLOW

app = Typer(add_completion=False, no_args_is_help=True)


@app.command(name="gendata", help="Generate dataset for federated learning.")
def gendata(
    config_file: Annotated[Path, Option("-c", "--config", help="The config file.", exists=True)],
):
    import datasets
    from flcore import FlDataset
    from flcore.utils import dump, fix_random_seed, proper_call

    with open(config_file) as f:
        config = toml.load(f)
        config = Box(config)

    # echo config details
    pprint(config.to_dict(), expand_all=True)

    if (seed := getattr(config, "seed", None)) is not None:
        fix_random_seed(seed)

    # load fl dataset
    dataset_fn = getattr(datasets, config.dataset.source)
    fl_dataset: FlDataset = proper_call(dataset_fn, **config.dataset)

    # save fl dataset
    dump(fl_dataset, config.output, replace=True)

    secho(f"The federated dataset is saved to {config.output}.", fg=GREEN)


@app.command(name="plot", help="Plot the distribution of the federated dataset.")
def plot(
    dataset: Annotated[Path, Option("-d", "--dataset", help="The FlDataset file.", exists=True)],
    sort: Annotated[bool, Option(help="Sort the data distribution by the number of samples.")] = False,
):
    import itertools

    import matplotlib.colors
    import matplotlib.pyplot as plt
    from flcore import FlDataset
    from flcore.utils import get_targets, group_targets, load
    from rich.progress import track

    fl_dataset: FlDataset = load(dataset)

    secho("Analyzing, this may take a long while.", fg=GREEN)

    colors = itertools.cycle(list(matplotlib.colors.TABLEAU_COLORS.keys()))
    class_color = {}

    # plot
    plt.figure(figsize=(10, 10))

    id_data = fl_dataset.id_data
    if sort:
        id_data = dict(sorted(id_data.items(), key=lambda x: x[1].datasize))

    # for each client
    max_height = 0
    for i, data_info in track(enumerate(id_data.values()), description="Plotting", total=len(id_data)):
        targets = [
            *get_targets(data_info.train_dataloader.dataset),
            *get_targets(data_info.validation_dataloader.dataset),
            *get_targets(data_info.test_dataloader.dataset),
        ]

        if len(targets) > max_height:
            max_height = len(targets)

        # for each class
        class_targets = group_targets(targets)
        offset = 0
        for class_ in sorted(class_targets.keys()):
            if class_ not in class_color:
                class_color[class_] = next(colors)

            num_targets = len(class_targets[class_])
            plt.fill_between([i, i + 1], offset, offset + num_targets, facecolor=class_color[class_])
            offset += num_targets

    secho("Done.", fg=GREEN)

    plt.title("Data Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.xlim(0, len(fl_dataset.id_data))
    plt.ylim(0, max_height)

    plt.savefig(dataset.with_suffix(".png"))

    plt.show()


@app.command(name="vis", help="Visualize the personalized models with t-SNE.")
def visual(
    config_file: Annotated[Path, Option("-c", "--config", help="The config file.", exists=True)],
    number: Annotated[int, Option("-n", "--number", help="The number of clients to visualize.")] = -1,
    clients: Annotated[str, Option(help='The clients to visualize, each id split by ",".')] = "",
    subset: Annotated[str, Option(help="The subset to visualize. i.e., train, validation or test.")] = "test",
    perplexity: Annotated[int, Option(help="The perplexity of t-SNE.")] = 5,
    early_exaggeration: Annotated[int, Option(help="The early exaggeration of t-SNE.")] = 12,
    max_iter: Annotated[int, Option(help="The max iteration of t-SNE.")] = 5_000,
):
    import itertools

    import matplotlib.pyplot as plt
    import models
    import numpy as np
    import torch
    from box import Box
    from flcore import FlDataset
    from flcore.utils import extract_features, fix_random_seed, load, proper_call
    from rich.progress import track
    from sklearn.manifold import TSNE
    from system.fedavg import ClientStateDict

    with open(config_file) as f:
        config = toml.load(f)
        config = Box(config)

    workdir = Path(config.workdir)

    # fix the seed if provided
    seed = config.get("seed", None)
    if seed is not None:
        secho(f"Fixing random seed to {seed}.", fg=GREEN)
        fix_random_seed(seed)

    # load federated dataset
    secho(f"Loading federated dataset from {config.dataset}.", fg=GREEN)
    fl_dataset: FlDataset = load(config.dataset)

    if clients:
        client_ids = [int(id) for id in clients.strip().split(",")]
        fl_dataset.id_data = {id: data for id, data in fl_dataset.id_data.items() if hash(id) in client_ids}

    if number > 0:
        fl_dataset.id_data = dict(itertools.islice(fl_dataset.id_data.items(), number))

    # prepare model
    secho(f"Preparing model {config.server.model.type}.", fg=GREEN)
    device = torch.device((config.get("devices", []) or ["cpu"])[0])
    model_type = getattr(models, config.server.model.type)
    model: torch.nn.Module = proper_call(
        model_type,
        **config.server.model | dict(num_class=fl_dataset.num_class),
    ).to(device)

    model.eval()
    id_result = {}

    with torch.inference_mode():
        secho("Evaluating the personalized model.", fg=GREEN)

        for index, client_data in track(fl_dataset.id_data.items(), transient=True):
            client_workdir = workdir / str(index)
            client_state: ClientStateDict = load(client_workdir / "model.pkl", map_location=device)

            if client_state.personalized_model is None:
                secho(f"Client {index} has no personalized model, exited.", fg=RED)
                exit(1)

            id_result[index] = {"features": [], "targets": []}
            model.load_state_dict(client_state.personalized_model)

            # visualize the model
            for x, y in getattr(client_data, f"{subset}_dataloader"):
                x = x.to(device)
                features = extract_features(model, x, models.HEAD_NAME)[0].clone().detach().cpu()
                id_result[index]["features"].extend(features)  # (BATCH, FEATURE...) -> [(FEATURE...) * BATCH]
                id_result[index]["targets"].extend(y)

            del client_state

    # execute t-SNE
    secho("Executing t-SNE. This may take a while...", fg=GREEN)

    tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, random_state=seed, max_iter=max_iter, verbose=True)  # type: ignore
    id_index = {id: i for i, id in enumerate(id_result.keys())}
    index_id = {i: id for i, id in enumerate(id_result.keys())}

    features = torch.concat([torch.stack(result["features"]) for result in id_result.values()]).numpy()
    targets = torch.concat([torch.stack(result["targets"]) for result in id_result.values()]).numpy()
    indices = torch.concat(
        [torch.tensor([id_index[id]] * len(result["features"])) for id, result in id_result.items()]
    ).numpy()

    tsne_features = tsne.fit_transform(features)

    # plot
    secho("Plotting.", fg=GREEN)
    plt.figure(figsize=(10, 10))
    all_markers = ["o", "x", "s", "v", "^", "<", ">", "p", "P", "*", "h", "H", "+", "X", "D", "d", "|", "_"]
    id_marker = {id: marker for id, marker in zip(id_index.values(), itertools.cycle(all_markers))}
    colors = plt.cm.tab20.colors  # type: ignore

    class_handlers = {}
    id_handlers = {}

    for index in track(np.unique(indices), transient=True):
        for class_ in np.unique(targets):
            mask = (indices == index) & (targets == class_)
            color = colors[class_ % len(colors)]
            marker = id_marker[index]
            plt.scatter(
                tsne_features[mask, 0],
                tsne_features[mask, 1],
                color=color,
                marker=marker,
            )

            handler = plt.plot([], [], "o", color=color, label=str(class_), ls="")[0]
            class_handlers.setdefault(class_, handler)
            handler = plt.plot([], [], marker=marker, color="black", label=str(index), ls="")[0]
            id_handlers.setdefault(index_id[index], handler)

    legend = plt.legend(handles=list(class_handlers.values()), title="Class", loc="upper right")
    plt.gca().add_artist(legend)
    plt.legend(handles=list(id_handlers.values()), labels=list(id_handlers.keys()), title="Client", loc="lower right")

    plt.savefig(workdir / "tsne.png")
    plt.show()


@app.command(help="Run federated learning.")
def run(
    config_file: Annotated[Path, Option("-c", "--config", help="The config file.", exists=True)],
    verbose: Annotated[bool, Option("-v", "--verbose", help="Verbose mode.")] = False,
    overwrites: Annotated[
        Optional[list[str]], Option("-o", "--overwrite", help="Overwrite the config file, e.g. client.max_epoch=3")
    ] = None,
):
    import importlib as imp
    import json
    import shutil

    import attackers
    import models
    import robusts
    import torch
    from flcore import FlDataset
    from flcore.utils import dump, fix_random_seed, load, proper_call, set_io_policy
    from system.helper import g

    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    with open(config_file) as f:
        config = toml.load(f)
        config = Box(config)

    if overwrites:
        parsed = [item.split("=", maxsplit=1) for item in overwrites]
        for key, value in parsed:
            walked_config = config
            path = key.split(".")
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = value

            for k in path[:-1]:
                walked_config = walked_config.get(k, Box())
            walked_config[path[-1]] = value

            secho(f"Overwrite {key} to {value}.", fg=GREEN)

    # echo current config
    pprint(config.to_dict(), expand_all=True)

    # fix the seed if provided
    if (seed := config.get("seed", None)) is not None:
        secho(f"Fixing random seed to {seed}.", fg=GREEN)
        fix_random_seed(seed)

    # set io policy
    policy = config.get("io_policy", "real")
    secho(f"Setting IO policy to {policy}.", fg=GREEN)
    set_io_policy(policy)

    # load protocol
    secho(f"Loading protocol {config.protocol}.", fg=GREEN)
    module_name = f"system.{config.protocol.lower()}"
    module = imp.import_module(module_name)
    client_type = getattr(module, config.protocol + "Client")
    server_type = getattr(module, config.protocol + "Server")
    system_type = getattr(module, config.protocol)

    # load federated dataset
    secho(f"Loading federated dataset from {config.dataset}.", fg=GREEN)
    fl_dataset: FlDataset = load(config.dataset)

    # prepare workdir
    workdir = Path(config.workdir)
    if workdir.exists():
        confirm("Workdir existed, remove it and then continue?", default=True, abort=True)
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # prepare g
    g.fl_dataset = fl_dataset
    g.devices = [torch.device(device) for device in config.get("devices", ["cpu"])]

    # prepare local data and create clients
    secho(f"Creating {len(fl_dataset.id_data)} clients.", fg=GREEN)
    clients = []
    for id, data in fl_dataset.id_data.items():
        client_workdir = workdir / str(id)
        client_workdir.mkdir(parents=True, exist_ok=True)

        # prepare params
        optimizer_args = Box(config.client.optimizer)
        criterion_args = Box(config.client.criterion)
        personalized_optimizer_args = Box(config.client.get("personalized_optimizer", {}))
        personalized_criterion_args = Box(config.client.get("personalized_criterion", {}))

        client = proper_call(
            client_type,
            **config.client  # custom args
            | dict(
                id=id,
                workdir=client_workdir,
                max_epoch=config.client.max_epoch,
                optimizer=optimizer_args,
                criterion=criterion_args,
                personalized_max_epoch=config.client.get("personalized_max_epoch", None),
                personalized_optimizer=personalized_optimizer_args,
                personalized_criterion=personalized_criterion_args,
            ),
        )

        clients.append(client)

    # prepare model
    secho(f"Preparing model {config.server.model.type}.", fg=GREEN)
    model_type = getattr(models, config.server.model.type)
    model = proper_call(model_type, **config.server.model | dict(num_class=fl_dataset.num_class)).cpu()

    # prepare robust function
    secho(f"Preparing robust function {config.server.robust_fn}.", fg=GREEN)
    robust_fn = None
    if robust_args := config.server.robust_fn:
        robust_type = getattr(robusts, robust_args.type)
        robust_fn = proper_call(robust_type, **robust_args)

    # create server
    secho("Creating server.", fg=GREEN)
    params = config.server.to_dict() | {
        "model": model,
        "clients": clients,
        "robust_fn": robust_fn,
    }
    server = proper_call(server_type, **params)

    # create system
    secho("Creating system.", fg=GREEN)
    params = config.to_dict() | {"server": server, "logdir": workdir}
    system = proper_call(system_type, **params)

    # perpare logger
    logging_file = workdir / "run.log"
    secho(f"Preparing logger to {logging_file}.", fg=GREEN)
    logger.add(logging_file)

    # prepare attack context
    secho("Preparing attack context.", fg=GREEN)
    context = attackers.AttackContext(system=system, number=config.attacker.number, device=g.devices[0])

    # prepare attack strategy
    secho("Preparing attack strategy.", fg=GREEN)
    attack_config: Box = config.attacker.copy()

    # type casting
    if "attack_range" in attack_config:
        attack_config.attack_range = range(*attack_config.attack_range)

    attack_strategy = proper_call(getattr(attackers, config.attacker.strategy), **attack_config)
    context.setup_strategy(attack_strategy)
    system.setup_context(context)

    # write configuration to workdir
    secho(f"Writting the final config file to workdir {workdir}.", fg=GREEN)
    with open(workdir / "config.toml", "w") as f:
        toml.dump(config, f)

    secho("Running system.", fg=GREEN)
    secho(" BEGIN ".center(80, "="), fg=YELLOW)

    with logger.catch():
        model = system.run()

    secho(" END ".center(80, "="), fg=YELLOW)

    set_io_policy("real")

    secho(f"Model is saved to {workdir / 'model.pt'}.", fg=GREEN)
    dump(model, workdir / "model.pt", replace=True)


if __name__ == "__main__":
    app()
