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
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from lightly.data import LightlyDataset
from lightly.loss.dcl_loss import DCLLoss
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.loss.tico_loss import TiCoLoss
from lightly.loss.vicreg_loss import VICRegLoss
from tests.utils.benchmarking.test_benchmark_module import (
    _DummyModel as _BenchmarkDummyModel,
)

"""
WARNING: 
Using a DDPStrategy causes the file to be executed once per device.
Thus this test needs to be in a separate file.
"""


@pytest.fixture
def close_torch_distributed() -> Generator[None, None, None]:
    yield None
    torch.distributed.destroy_process_group()


class TestGatherLayer_BenchmarkModule:
    """
    Tests that the gather layer works as expected.

    Because using a DDPStrategy causes the whole script to be executed multiple
    times, running a proper test is difficult. The approach used here:

    1. This test was run once with n_devices=1 and gather=False.
    The resulting knn accuracy of 0.953125 is hardcoded in the assertion

    2. This test is now run with n_devices=2 and gather=True. The resulting
    parameters are asserted to be the same ad with n_devices=1 and gather=False.

    Note that the results would not be the same for n_devices=1 and n_devices=2 if
    there was:
    - Any randomness in the transform, as the order the samples are processed and
    thus the random seed differ.
    - Any batch normalization in the model, as the batch size is split between
    the devices and there is no gather layer for the batch normalization.
    """

    def test__benchmark_module(self, close_torch_distributed: None) -> None:
        n_devices = 2
        n_samples = 32
        num_classes = 10
        batch_size = int(n_samples / n_devices) // 2

        pl.seed_everything(0, workers=True)

        dataset_train = LightlyDataset.from_torch_dataset(
            FakeData(
                size=n_samples,
                image_size=(3, 32, 32),
                num_classes=num_classes,
                transform=ToTensor(),
            )
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
        dataset_val = LightlyDataset.from_torch_dataset(
            FakeData(
                size=100,
                image_size=(3, 32, 32),
                num_classes=num_classes,
                transform=ToTensor(),
                random_offset=10,
            )
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )

        model = _BenchmarkDummyModel(
            dataloader_kNN=dataloader_train, knn_k=3, num_classes=num_classes
        )

        trainer = Trainer(
            devices=n_devices,
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            max_epochs=3,
        )
        trainer.fit(
            model,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val,
        )

        assert model.max_accuracy == 0.75
