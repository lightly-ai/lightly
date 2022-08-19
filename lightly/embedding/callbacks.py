import os

from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from lightly.cli import _helpers


def create_checkpoint_callback(
    save_last=False,
    save_top_k=0,
    monitor='loss',
    dirpath=None,
):
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

    """
    return ModelCheckpoint(
        dirpath=os.getcwd() if dirpath is None else dirpath,
        filename='lightly_epoch_{epoch:d}',
        save_last=save_last,
        save_top_k=save_top_k,
        monitor=monitor,
        auto_insert_metric_name=False)


def create_summary_callback(summary_callback_config: DictConfig, trainer_config: DictConfig):
    """Creates a summary callback.

    If the deprecated argument ``weights_summary`` is present
    it is removed from ``trainer_config``.
    """
    # TODO: Drop support for the "weights_summary" argument.
    weights_summary = trainer_config.get("weights_summary", None)
    if weights_summary not in [None, "None"]:
        summary_callback = _create_summary_callback_deprecated(weights_summary)
    else:
        summary_callback = _create_summary_callback(**summary_callback_config)

    if "weights_summary" in trainer_config:
        del trainer_config["weights_summary"]

    return summary_callback


def _create_summary_callback(max_depth: int):
    """Initializes the model summary callback.
    See `ModelSummary reference documentation <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelSummary.html?highlight=ModelSummary>`.

    Args:
        max_depth:
            The maximum depth of layer nesting that the summary will include.
    """
    return ModelSummary(max_depth=max_depth)


def _create_summary_callback_deprecated(weights_summary: str):
    """Constructs summary callback from the deprecated ``weights_summary`` argument.

    The ``weights_summary`` trainer argument was deprecated with the release
    of pytorch lightning 1.7 in 08/2022. Support for this will be removed
    in the future.
    """
    _helpers.print_as_warning(
        "The configuration parameter 'trainer.weights_summary' is deprecated."
        " Please use 'trainer.weights_summary: True' and set"
        " 'checkpoint_callback.max_depth' to value 1 for the option 'top'"
        " or -1 for the option 'full'."
    )
    if weights_summary == "top":
        max_depth = 1
    elif weights_summary == "full":
        max_depth = -1
    else:
        raise ValueError(
            "Invalid value for the deprecated trainer.weights_summary"
            " configuration parameter."
        )
    _create_summary_callback(max_depth=max_depth)
