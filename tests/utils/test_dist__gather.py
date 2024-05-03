from typing import Any

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import Tensor
from torch.nn import Linear, MSELoss
from torch.optim import SGD
from torch.utils.data import TensorDataset

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.utils import dist as lightly_dist

"""
WARNING: 
Using a DDPStrategy causes the file to be executed once per device.
Thus this test needs to be in a separate file.
"""


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
        self.criterion = NTXentLoss(gather_distributed=gather)
        self.gather = gather

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x = batch[0]
        y = batch[1]
        x = self.model(x).flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        loss = self.criterion(x, y)
        return loss

    def configure_optimizers(self) -> Any:
        return SGD(self.parameters(), lr=0.01)


@pytest.fixture
def close_torch_distributed() -> None:
    yield None
    torch.distributed.destroy_process_group()


class TestGatherLayer:
    def test(self, close_torch_distributed) -> None:
        """
        Tests that the gather layer works as expected.

        Because using a DDPStrategy causes the whole script to be executed multiple
        times, running a proper test is difficult. The approach used here:

        1. This test was run once with n_devices=1 and gather=False. The resulting
        parameters after 10 epochs are saved as expected_params__10_epochs__no_gather.

        2. This test is now run with n_devices=2 and gather=True. The resulting
        parameters are asserted to be the same ad with n_devices=1 and gather=False.

        Note that the results would not be the same for n_devices=1 and n_devices=2 if
        there was:
        - Any randomness in the transform, as the order the samples are processed and
          thus the random seed differ.
        - Any batch normalization in the model, as the batch size is split between
          the devices and there is no gather layer for the batch normalization.
        """
        n_devices = 2
        n_samples = 8
        batch_size = int(n_samples / n_devices)
        gather = n_devices > 1

        pl.seed_everything(0, workers=True)

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
                [0.1076, 0.2076, 0.3077, 0.4077, 0.5078],
                [0.5976, 0.6973, 0.7971, 0.8969, 0.9967],
            ]
        )
        assert torch.allclose(params, expected_params__10_epochs__no_gather, rtol=1e-2)
