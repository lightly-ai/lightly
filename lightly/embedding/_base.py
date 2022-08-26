""" BaseEmbeddings """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import copy
import os


import omegaconf
from omegaconf import DictConfig
import pytorch_lightning as pl
import pytorch_lightning.core.lightning as lightning

from lightly.embedding import callbacks


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

    def train_embedding(
        self,
        trainer_config: DictConfig,
        checkpoint_callback_config: DictConfig,
        summary_callback_config: DictConfig,
    ):
        """ Train the model on the provided dataset.

        Args:
            trainer_config: pylightning_trainer arguments, examples include:
                min_epochs: (int) Minimum number of epochs to train
                max_epochs: (int) Maximum number of epochs to train
                gpus: (int) Number of gpus to use
                enable_model_summary: (bool) Whether to enable model summarisation.
                weights_summary: (str) DEPRECATED. How to print a summary of the model and weights.
            checkpoint_callback_config: ModelCheckpoint callback arguments
            summary_callback_config: ModelSummary callback arguments

        Returns:
            A trained encoder, ready for embedding datasets.

        """
        trainer_callbacks = []

        checkpoint_cb = callbacks.create_checkpoint_callback(**checkpoint_callback_config)
        trainer_callbacks.append(checkpoint_cb)

        summary_cb = callbacks.create_summary_callback(
            summary_callback_config=summary_callback_config,
            trainer_config=trainer_config,
        )
        if summary_cb is not None:
            trainer_callbacks.append(summary_cb)

        # Remove weights_summary from trainer_config now that the summary callback
        # has been created. TODO: Drop support for the "weights_summary" argument.
        trainer_config_copy = copy.deepcopy(trainer_config)
        if "weights_summary" in trainer_config_copy:
            with omegaconf.open_dict(trainer_config_copy):
                del trainer_config_copy["weights_summary"]

        trainer = pl.Trainer(**trainer_config_copy, callbacks=trainer_callbacks)

        trainer.fit(self)

        if checkpoint_cb.best_model_path != "":
            self.checkpoint = os.path.join(self.cwd, checkpoint_cb.best_model_path)

    def embed(self, *args, **kwargs):
        """Must be implemented by classes which inherit from BaseEmbedding.

        """
        raise NotImplementedError()
