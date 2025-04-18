""" Contains the core functionality of the lightly Python package. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os
from typing import List, Tuple

import numpy as np
import yaml

import lightly.cli as cli
from lightly.cli.embed_cli import _embed_cli
from lightly.cli.lightly_cli import _lightly_cli
from lightly.cli.train_cli import _train_cli


def _get_config_path(config_path):
    """Find path to yaml config file

    Args:
        config_path: (str) Path to config.yaml file

    Returns:
        Path to config.yaml if specified else default config.yaml

    Raises:
        ValueError: If the config_path is not None but doesn't exist

    """
    if config_path is None:
        dirname = os.path.dirname(cli.__file__)
        config_path = os.path.join(dirname, "config/config.yaml")
    if not os.path.exists(config_path):
        raise ValueError("Config path {} does not exist!".format(config_path))

    return config_path


def _load_config_file(config_path):
    """Load a yaml config file

    Args:
        config_path: (str) Path to config.yaml file

    Returns:
        Dictionary with configs from config.yaml

    """
    Loader = yaml.FullLoader
    with open(config_path, "r") as config_file:
        cfg = yaml.load(config_file, Loader=Loader)

    return cfg


def _add_kwargs(cfg, kwargs):
    """Add keyword arguments to config

    Args:
        cfg: (dict) Dictionary of configs from config.yaml
        kwargs: (dict) Dictionary of keyword arguments

    Returns:
        Union of cfg and kwargs

    """
    for key, item in kwargs.items():
        if isinstance(item, dict):
            if key in cfg:
                cfg[key] = _add_kwargs(cfg[key], item)
            else:
                cfg[key] = item
        else:
            cfg[key] = item
    return cfg


def train_model_and_embed_images(
    config_path: str = None, **kwargs
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Train a self-supervised model and use it to embed images.

    First trains a modle using the _train_cli(),
    then embeds with the _embed_cli().
    All arguments passed to the CLI functions
    can also be passed to this function (see below for an example).

    Args:
        config_path:
            Path to config.yaml. If None, the default configs will be used.
        **kwargs:
            Overwrite default configs py passing keyword arguments.

    Returns:
        Embeddings, labels, and filenames of the images.
        Embeddings are of shape (n_samples, embedding_size)
        len(labels) = len(filenames) = n_samples

    Examples:
        >>> import lightly
        >>>
        >>> # train a model and embed images with default configs
        >>> embeddings, _, _ = lightly.train_model_and_embed_images(
        >>>     input_dir='path/to/data')
        >>>
        >>> # train a model and embed images with separate config file
        >>> my_config_path = 'my/config/file.yaml'
        >>> embeddings, _, _ = lightly.train_model_and_embed_images(
        >>>     input_dir='path/to/data', config_path=my_config_path)
        >>>
        >>> # train a model and embed images with default settings + overwrites
        >>> my_trainer = {max_epochs: 10}
        >>> embeddings, _, _ = lightly.train_model_and_embed_images(
        >>>     input_dir='path/to/data', trainer=my_trainer)

    """
    config_path = _get_config_path(config_path)
    config_args = _load_config_file(config_path)
    config_args = _add_kwargs(config_args, kwargs)

    checkpoint = _train_cli(config_args, is_cli_call=False)
    config_args["checkpoint"] = checkpoint
    embeddings, labels, filenames = _embed_cli(config_args, is_cli_call=False)
    return embeddings, labels, filenames


def train_embedding_model(config_path: str = None, **kwargs):
    """Train a self-supervised model.

    Calls the same function as lightly-ssl-train. All arguments passed to
    lightly-ssl-train can also be passed to this function (see below for an
    example).

    Args:
        config_path:
            Path to config.yaml. If None, the default configs will be used.
        **kwargs:
            Overwrite default configs py passing keyword arguments.

    Returns:
        Path to checkpoint of the trained embedding model.

    Examples:
        >>> import lightly
        >>>
        >>> # train a model with default configs
        >>> checkpoint_path = lightly.train_embedding_model(
        >>>     input_dir='path/to/data')
        >>>
        >>> # train a model with separate config file
        >>> my_config_path = 'my/config/file.yaml'
        >>> checkpoint_path = lightly.train_embedding_model(
        >>>     input_dir='path/to/data', config_path=my_config_path)
        >>>
        >>> # train a model with default settings and overwrites: large batch
        >>> # sizes are benefitial for self-supervised training and more
        >>> # workers speed up the dataloading process.
        >>> my_loader = {
        >>>     batch_size: 100,
        >>>     num_workers: 8,
        >>> }
        >>> checkpoint_path = lightly.train_embedding_model(
        >>>     input_dir='path/to/data', loader=my_loader)
        >>> # the command above is equivalent to:
        >>> # lightly-ssl-train input_dir='path/to/data' loader.batch_size=100 loader.num_workers=8
    """
    config_path = _get_config_path(config_path)
    config_args = _load_config_file(config_path)
    config_args = _add_kwargs(config_args, kwargs)

    return _train_cli(config_args, is_cli_call=False)


def embed_images(checkpoint: str, config_path: str = None, **kwargs):
    """Embed images with a self-supervised model.

    Calls the same function as lightly-embed. All arguments passed to
    lightly-embed can also be passed to this function (see below for an
    example).

    Args:
        checkpoint:
            Path to the checkpoint file for the embedding model.
        config_path:
            Path to config.yaml. If None, the default configs will be used.
        **kwargs:
            Overwrite default configs py passing keyword arguments.

    Returns:
        Embeddings, labels, and filenames of the images.

    Examples:
        >>> import lightly
        >>> my_checkpoint_path = 'path/to/checkpoint.ckpt'
        >>>
        >>> # embed images with default configs
        >>> embeddings, _, _ = lightly.embed_images(
        >>>     my_checkpoint_path, input_dir='path/to/data')
        >>>
        >>> # embed images with separate config file
        >>> my_config_path = 'my/config/file.yaml'
        >>> embeddings, _, _ = lightly.embed_images(
        >>>     my_checkpoint_path, input_dir='path/to/data', config_path=my_config_path)
        >>>
        >>> # embed images with default settings and overwrites: at inference,
        >>> # we can use larger input_sizes because it requires less memory.
        >>> my_collate = {input_size: 256}
        >>> embeddings, _, _ = lightly.embed_images(
        >>>     my_checkpoint_path, input_dir='path/to/data', collate=my_collate)
        >>> # the command above is equivalent to:
        >>> # lightly-embed input_dir='path/to/data' collate.input_size=256

    """
    config_path = _get_config_path(config_path)
    config_args = _load_config_file(config_path)
    config_args = _add_kwargs(config_args, kwargs)

    config_args["checkpoint"] = checkpoint

    return _embed_cli(config_args, is_cli_call=False)
