from typing import Any, Generator, Tuple

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import Tensor
from torch.nn import Linear, Module
from torch.optim import SGD
from torch.utils.data import TensorDataset

from lightly.loss.dcl_loss import DCLLoss
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.loss.tico_loss import TiCoLoss
from lightly.loss.vicreg_loss import VICRegLoss

"""
WARNING: 
Using a DDPStrategy causes the file to be executed once per device.
Thus this test needs to be in a separate file.
"""


class Model(LightningModule):
    def __init__(self, gather: bool, criterion: Module, learning_rate: float):
        super().__init__()
        self.model = Linear(5, 2, bias=False)
        self.model.weight.data = torch.Tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
            ]
        )
        self.criterion = criterion
        self.gather = gather
        self.learning_rate = learning_rate

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x = batch[0]
        y = batch[1]
        x = self.model(x).flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        loss: Tensor = self.criterion(x, y)
        return loss

    def configure_optimizers(self) -> Any:
        return SGD(self.parameters(), lr=self.learning_rate)


@pytest.fixture
def close_torch_distributed() -> Generator[None, None, None]:
    yield None
    torch.distributed.destroy_process_group()


class TestGatherLayer_Losses:
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

    def test_loss_ntxent(self, close_torch_distributed: None) -> None:
        self._test_ddp(
            NTXentLoss,
            torch.Tensor(
                [
                    [0.1675, 0.2676, 0.3678, 0.4679, 0.5680],
                    [0.5763, 0.6742, 0.7721, 0.8700, 0.9679],
                ]
            ),
            learning_rate=0.1,
        )

    def test_loss_tico(self, close_torch_distributed: None) -> None:
        self._test_ddp(
            TiCoLoss,
            torch.Tensor(
                [
                    [0.1985, 0.3004, 0.4022, 0.5041, 0.6060],
                    [0.5694, 0.6627, 0.7559, 0.8492, 0.9424],
                ]
            ),
            learning_rate=0.1,
        )

    def test_loss_vicreg(self, close_torch_distributed: None) -> None:
        self._test_ddp(
            VICRegLoss,
            torch.Tensor(
                [
                    [-0.2382, -0.1381, -0.0381, 0.0619, 0.1620],
                    [-0.0494, 0.0505, 0.1504, 0.2503, 0.3503],
                ]
            ),
            learning_rate=1e-8,
        )

    def test_loss_dcl(self, close_torch_distributed: None) -> None:
        self._test_ddp(
            DCLLoss,
            torch.Tensor(
                [
                    [0.1484, 0.2335, 0.3185, 0.4036, 0.4886],
                    [0.6198, 0.7093, 0.7988, 0.8884, 0.9779],
                ]
            ),
            learning_rate=0.1,
        )

    def _test_ddp(
        self,
        loss: Module,
        expected_params__10_epochs__no_gather: Tensor,
        learning_rate: float,
    ) -> None:
        n_devices = 1
        n_samples = 8
        batch_size = int(n_samples / n_devices)
        gather = n_devices > 1

        pl.seed_everything(0, workers=True)

        xs = torch.arange(n_samples * 2 * 5).reshape(n_samples, 2, 5).float()
        ys = torch.arange(n_samples * 2 * 2).reshape(n_samples, 2, 2).float()
        dataset = TensorDataset(xs, ys)

        model = Model(
            gather=gather,
            criterion=loss(gather_distributed=gather),
            learning_rate=learning_rate,
        )

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
        assert torch.allclose(params, expected_params__10_epochs__no_gather, rtol=1e-3)
