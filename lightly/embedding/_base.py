""" BaseEmbeddings """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os

import pytorch_lightning as pl
import pytorch_lightning.core.lightning as lightning
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from lightly.cli import _helpers


class BaseEmbedding(lightning.LightningModule):
    """All trainable embeddings must inherit from BaseEmbedding.

    """

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 dataloader,
                 scheduler=None):
        """ Constructor

        Args:
            model: (torch.nn.Module)
            criterion: (torch.nn.Module)
            optimizer: (torch.optim.Optimizer)
            dataloader: (torch.utils.data.DataLoader)

        """

        super(BaseEmbedding, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.checkpoint = None
        self.cwd = os.getcwd()

        self._checkpoint_callback = None
        self._summary_callback = None

    def forward(self, x0, x1):
        return self.model(x0, x1)

    def training_step(self, batch, batch_idx):

        # get the two image transformations
        (x0, x1), _, _ = batch
        # forward pass of the transformations
        y0, y1 = self(x0, x1)
        # calculate loss
        loss = self.criterion(y0, y1)
        # log loss and return
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return self.dataloader

    def train_embedding(self, **kwargs):
        """ Train the model on the provided dataset.

        Args:
            **kwargs: pylightning_trainer arguments, examples include:
                min_epochs: (int) Minimum number of epochs to train
                max_epochs: (int) Maximum number of epochs to train
                gpus: (int) Number of gpus to use
                enable_model_summary: (bool) Whether to enable model summarisation.
                weights_summary: (str) DEPRECATED. How to print a summary of the model and weights.

        Returns:
            A trained encoder, ready for embedding datasets.

        """
        # TODO: Drop support for the "weights_summary" argument.
        if "weights_summary" in kwargs:
            self._init_summary_callback_from_trainer_arguments(kwargs["weights_summary"])
            del kwargs["weights_summary"]

        callbacks = [
            callback for callback in [self._checkpoint_callback, self._summary_callback]
            if callback is not None
        ]
        trainer = pl.Trainer(**kwargs, callbacks=callbacks)

        trainer.fit(self)

        self.checkpoint = self._checkpoint_callback.best_model_path
        self.checkpoint = os.path.join(self.cwd, self.checkpoint)

    def embed(self, *args, **kwargs):
        """Must be implemented by classes which inherit from BaseEmbedding.

        """
        raise NotImplementedError()

    def init_checkpoint_callback(self,
                                 save_last=False,
                                 save_top_k=0,
                                 monitor='loss',
                                 dirpath=None):
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
        self._checkpoint_callback = ModelCheckpoint(
            dirpath=self.cwd if dirpath is None else dirpath,
            filename='lightly_epoch_{epoch:d}',
            save_last=save_last,
            save_top_k=save_top_k,
            monitor=monitor,
            auto_insert_metric_name=False)

    def init_summary_callback(self, max_depth: int):
        """Initializes the model summary callback.
        See `ModelSummary reference documentation <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelSummary.html?highlight=ModelSummary>`.

        Args:
            max_depth:
                The maximum depth of layer nesting that the summary will include.
                A value of 0 turns the layer summary off. Options ``top`` and ``full``
                from previous pytorch lightning versions correspond to values
                1 and -1 respectively.
        """
        self._summary_callback = ModelSummary(max_depth=max_depth)

    def _init_summary_callback_from_trainer_arguments(self, weights_summary: str):
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
        elif weights_summary is None or weights_summary == "None":
            max_depth = 0
        else:
            raise ValueError(
                "Invalid value for the deprecated trainer.weights_summary"
                " configuration parameter."
            )
        self.init_summary_callback(max_depth=max_depth)
