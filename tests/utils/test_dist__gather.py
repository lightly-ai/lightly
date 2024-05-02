import os
from typing import Any
import unittest
from unittest import mock

import pytest
from pytest_mock import MockerFixture
import torch
from pytest import CaptureFixture
import torch.multiprocessing as mp
import torch.distributed as torch_dist
from lightly.utils import dist as lightly_dist


from typing import Any
from torch.nn import Linear, MSELoss
from torch.optim import SGD
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import torch
from pytorch_lightning.strategies.ddp import DDPStrategy


pl.seed_everything(0, workers=True)


class Model(LightningModule):
    def __init__(self, gather: bool):
        super().__init__()
        self.model = Linear(5, 2, bias=False)
        self.model.weight.data = torch.Tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
            ]
        )
        self.criterion = MSELoss()
        self.gather = gather

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x = batch[0]
        y = batch[1]
        x = self.model(x)
        if self.gather:
            x = torch.cat(lightly_dist.gather(x))
            with torch.no_grad():
                y = torch.cat(lightly_dist.gather(y))
        loss = self.criterion(x, y)
        return loss

    def configure_optimizers(self) -> Any:
        return SGD(self.parameters(), lr=0.0001)


class TestGatherLayer:


    def test(self) -> None:
        """
        Tests that the gather layer works as expected.

        Because using a DDPStrategy causes the whole script to be executed multiple
        times, running a proper test is difficult. The approach used here:

        1. This test was run once with n_devices=1 and gather=False. The resulting
        parameters after 100 epochs are saved as expected_params__10_epochs__no_gather.

        2. This test was run once with n_devices=2 and gather=True. The resulting
        parameters are checked to be the same.

        Note that the results would not be the same for n_devices=1 and n_devices=2 if
        there was:
        - any randomness in the transform, as the order the samples are processed and
          thus the random seed differ
        - any batch normalization in the model, as the batch size is split between
          the devices and there is no gather layer for the batch normalization
        """
        n_samples = 8
        n_devices = 8
        batch_size = int(n_samples / n_devices)
        gather = n_devices > 1


        xs = torch.arange(n_samples * 2 * 5).reshape(n_samples, 2, 5).float()
        ys = torch.arange(n_samples * 2 * 2).reshape(n_samples, 2, 2).float()
        dataset = TensorDataset(xs, ys)

        model = Model(gather=gather)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        trainer = Trainer(
            devices=n_devices,
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            max_epochs=10,
        )
        trainer.fit(model=model, train_dataloaders=dataloader)

        params = list(model.parameters())[0]
        expected_params__10_epochs__no_gather = torch.Tensor(
            [
                [-0.1172, -0.0219,  0.0734,  0.1687,  0.2639],
                [-0.0949, -0.0087,  0.0775,  0.1637,  0.2499]
            ]
        )
        assert torch.allclose(params, expected_params__10_epochs__no_gather, rtol=1e-2)