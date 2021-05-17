""" BaseEmbeddings """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os
import copy

import pytorch_lightning as pl
import pytorch_lightning.core.lightning as lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch.nn as nn

from lightly.embedding._callback import CustomModelCheckpoint


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

        self.checkpoint_callback = None
        self.init_checkpoint_callback()

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
                gpus: (int) number of gpus to use

        Returns:
            A trained encoder, ready for embedding datasets.

        """

        trainer = pl.Trainer(**kwargs, callbacks=[self.checkpoint_callback])

        trainer.fit(self)

        self.checkpoint = self.checkpoint_callback.best_model_path
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

        if pl.__version__[:3] in ['1.0', '1.1', '1.2']:
            # initialize custom model checkpoint
            self.checkpoint_callback = CustomModelCheckpoint()
            self.checkpoint_callback.save_last = save_last
            self.checkpoint_callback.save_top_k = save_top_k
            self.checkpoint_callback.monitor = monitor

            dirpath = self.cwd if dirpath is None else dirpath
            self.checkpoint_callback.dirpath = dirpath
        else:
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=self.cwd if dirpath is None else dirpath,
                filename='lightly_epoch_{epoch:d}',
                save_last=save_last,
                save_top_k=save_top_k,
                monitor=monitor,
                auto_insert_metric_name=False)
