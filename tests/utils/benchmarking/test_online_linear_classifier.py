import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.utils.benchmarking import OnlineLinearClassifier


class TestOnlineLinearClassifier:
    def test__cpu(self) -> None:
        self._test(accelerator="cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test__cuda(self) -> None:
        self._test(accelerator="gpu")

    def _test(self, accelerator: str) -> None:
        dataset = FakeData(
            size=10, image_size=(3, 8, 8), num_classes=5, transform=T.ToTensor()
        )
        train_dataloader = DataLoader(dataset, batch_size=2)
        val_dataloader = DataLoader(dataset, batch_size=2)
        model = _DummyModule()
        trainer = Trainer(
            max_epochs=1, accelerator=accelerator, devices=1, log_every_n_steps=1
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        assert trainer.callback_metrics["train_online_cls_loss"].item() >= 0
        assert trainer.callback_metrics["train_online_cls_top1"].item() >= 0
        assert (
            trainer.callback_metrics["train_online_cls_top5"].item()
            >= trainer.callback_metrics["train_online_cls_top1"].item()
        )
        assert trainer.callback_metrics["train_online_cls_top5"].item() <= 1
        assert trainer.callback_metrics["val_online_cls_loss"].item() >= 0
        assert trainer.callback_metrics["val_online_cls_top1"].item() >= 0
        assert (
            trainer.callback_metrics["val_online_cls_top5"].item()
            >= trainer.callback_metrics["val_online_cls_top1"].item()
        )
        assert trainer.callback_metrics["val_online_cls_top5"].item() <= 1


class _DummyModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
        self.online_classifier = OnlineLinearClassifier(feature_dim=3, num_classes=5)

    def training_step(self, batch, batch_idx) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.linear(images)
        cls_loss, cls_log = self.online_classifier.training_step(
            (features, targets), batch_idx
        )
        self.log_dict(cls_log)
        return cls_loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.linear(images)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features, targets), batch_idx
        )
        self.log_dict(cls_log)
        return cls_loss

    def configure_optimizers(self) -> SGD:
        return SGD(self.parameters(), lr=0.1)
