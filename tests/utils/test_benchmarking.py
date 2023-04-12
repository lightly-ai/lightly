import unittest

import torch
from pytorch_lightning import Trainer
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from lightly.data import LightlyDataset
from lightly.utils.benchmarking import BenchmarkModule


class TestBenchmarkModule:
    def test(self, accelerator: str = "cpu") -> None:
        torch.manual_seed(0)
        dataset = LightlyDataset.from_torch_dataset(
            FakeData(
                size=10, image_size=(3, 32, 32), num_classes=2, transform=ToTensor()
            )
        )
        dataloader = DataLoader(dataset, batch_size=2)

        model = _DummyModel(dataloader_kNN=dataloader)
        trainer = Trainer(max_epochs=2, accelerator=accelerator)
        trainer.fit(
            model,
            train_dataloaders=dataloader,
            val_dataloaders=dataloader,
        )
        assert model.max_accuracy == 1.0  # accuracy is 1.0 because knn_k=1

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_cuda(self) -> None:
        self.test(accelerator="cuda")

    def test_knn_train_val(self) -> None:
        torch.manual_seed(0)
        dataset_train = LightlyDataset.from_torch_dataset(
            FakeData(
                size=10, image_size=(3, 32, 32), num_classes=2, transform=ToTensor()
            )
        )
        dataloader_train = DataLoader(dataset_train, batch_size=2)
        dataset_val = LightlyDataset.from_torch_dataset(
            FakeData(
                size=10,
                image_size=(3, 32, 32),
                num_classes=2,
                transform=ToTensor(),
                random_offset=10,
            )
        )
        dataloader_val = DataLoader(dataset_val, batch_size=2)

        model = _DummyModel(dataloader_kNN=dataloader_train, knn_k=3)
        trainer = Trainer(max_epochs=2)
        trainer.fit(
            model,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val,
        )
        assert (
            model.max_accuracy < 1.0
        )  # accuracy is <1.0 because train val are different


class _DummyModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, knn_k=1):
        super().__init__(dataloader_kNN, num_classes=2, knn_k=knn_k)
        self.backbone = Sequential(
            Flatten(),
            Linear(3 * 32 * 32, 2),
        )
        self.criterion = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        predictions = self.backbone(images)
        loss = self.criterion(predictions, targets)
        return loss

    def configure_optimizers(self):
        return SGD(self.backbone.parameters(), lr=0.1)
