import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.utils.benchmarking import MetricCallback


class TestMetricCallback:
    def test(self) -> None:
        callback = MetricCallback()
        trainer = Trainer(accelerator="cpu", callbacks=[callback], max_epochs=3)
        dataset = FakeData(
            size=10, image_size=(3, 32, 32), num_classes=10, transform=T.ToTensor()
        )
        train_dataloader = DataLoader(dataset, batch_size=2)
        val_dataloader = DataLoader(dataset, batch_size=2)
        trainer.fit(
            _DummyModule(),
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        assert callback.train_metrics["train_epoch"] == [0, 1, 2]
        assert callback.train_metrics["train_epoch_dict"] == [0, 1, 2]
        # test logs 2 * epoch in validation step
        assert callback.val_metrics["val_epoch"] == [0, 2, 4]
        assert callback.val_metrics["val_epoch_dict"] == [0, 2, 4]


class _DummyModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx) -> None:
        self.log("train_epoch", self.trainer.current_epoch)
        self.log_dict({"train_epoch_dict": self.trainer.current_epoch})

    def validation_step(self, batch, batch_idx) -> None:
        self.log("val_epoch", self.trainer.current_epoch * 2)
        self.log_dict({"val_epoch_dict": self.trainer.current_epoch * 2})

    def configure_optimizers(self) -> None:
        return None
