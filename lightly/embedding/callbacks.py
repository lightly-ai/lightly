import os
from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary


def create_checkpoint_callback(
    save_last: bool = False,
    save_top_k: int = 0,
    monitor: str = "loss",
    dirpath: Optional[str] = None,
) -> ModelCheckpoint:
    """Initializes the checkpoint callback.

    Args:
        save_last:
            Whether or not to save the checkpoint of the last epoch.
        save_top_k:
            Save the top_k model checkpoints.
        monitor:
            Which quantity to monitor.
        dirpath:
            Where to save the checkpoint.

    Returns:
        ModelCheckpoint: The initialized checkpoint callback.

    """
    return ModelCheckpoint(
        dirpath=os.getcwd() if dirpath is None else dirpath,
        filename="lightly_epoch_{epoch:d}",
        save_last=save_last,
        save_top_k=save_top_k,
        monitor=monitor,
        auto_insert_metric_name=False,
    )


def create_summary_callback(summary_callback_config: DictConfig) -> ModelSummary:
    """Creates a model summary callback based on the configuration.

    Args:
        summary_callback_config:
            Configuration dictionary for the summary callback.

    Returns:
        ModelSummary: The model summary callback.

    """
    return _create_summary_callback(summary_callback_config["max_depth"])


def _create_summary_callback(max_depth: int) -> ModelSummary:
    """Initializes the model summary callback.
    See `ModelSummary reference documentation
    <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelSummary.html?highlight=ModelSummary>`.

    Args:
        max_depth:
            The maximum depth of layer nesting that the summary will include.

    Returns:
        ModelSummary: The initialized model summary callback.

    """
    return ModelSummary(max_depth=max_depth)
