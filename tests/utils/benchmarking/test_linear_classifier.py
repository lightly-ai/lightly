import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from lightly.utils.benchmarking import FinetuneClassifier, LinearClassifier


class TestFinetuneClassifier:
    def test__finetune(self) -> None:
        """Test the classifier in a finetune evaluation setting.

        The test verifies that the model and classification head are updated.
        """
        torch.manual_seed(0)
        dataset = FakeData(
            size=10, image_size=(3, 8, 8), num_classes=5, transform=ToTensor()
        )
        train_dataloader = DataLoader(dataset, batch_size=2)
        val_dataloader = DataLoader(dataset, batch_size=2)
        linear = nn.Linear(3 * 8 * 8, 4)
        model = nn.Sequential(nn.Flatten(), linear)
        initial_weights = linear.weight.clone()
        finetune_classifier = FinetuneClassifier(
            model=model,
            batch_size_per_device=2,
            feature_dim=4,
            num_classes=5,
        )
        initial_head_weights = finetune_classifier.classification_head.weight.clone()
        trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1)
        trainer.fit(finetune_classifier, train_dataloader, val_dataloader)
        assert trainer.callback_metrics["train_loss"].item() > 0
        assert trainer.callback_metrics["train_top1"].item() >= 0
        assert (
            trainer.callback_metrics["train_top5"].item()
            >= trainer.callback_metrics["train_top1"].item()
        )
        assert trainer.callback_metrics["train_top5"].item() <= 1
        assert trainer.callback_metrics["val_loss"].item() > 0
        assert trainer.callback_metrics["val_top1"].item() >= 0
        assert (
            trainer.callback_metrics["val_top5"].item()
            >= trainer.callback_metrics["val_top1"].item()
        )
        assert trainer.callback_metrics["val_top5"].item() <= 1

        # Verify that weights were updated.
        assert not torch.all(torch.eq(initial_weights, linear.weight))
        # Verify that head weights were updated.
        assert not torch.all(
            torch.eq(
                initial_head_weights, finetune_classifier.classification_head.weight
            )
        )


class TestLinearClassifier:
    def test__linear(self) -> None:
        """Test the classifier in a linear evaluation setting.

        The test verifies that only the classification head is updated and the model
        remains unchanged.
        """
        torch.manual_seed(0)
        dataset = FakeData(
            size=10, image_size=(3, 8, 8), num_classes=5, transform=ToTensor()
        )
        train_dataloader = DataLoader(dataset, batch_size=2)
        val_dataloader = DataLoader(dataset, batch_size=2)
        linear = nn.Linear(3 * 8 * 8, 4)
        batch_norm = nn.BatchNorm1d(4)
        model = nn.Sequential(nn.Flatten(), linear, batch_norm)
        initial_weights = linear.weight.clone()
        initial_bn_weights = batch_norm.weight.clone()
        linear_classifier = LinearClassifier(
            model=model,
            batch_size_per_device=2,
            feature_dim=4,
            num_classes=5,
        )
        initial_head_weights = linear_classifier.classification_head.weight.clone()
        trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1)
        trainer.fit(linear_classifier, train_dataloader, val_dataloader)
        assert trainer.callback_metrics["train_loss"].item() > 0
        assert trainer.callback_metrics["train_top1"].item() >= 0
        assert (
            trainer.callback_metrics["train_top5"].item()
            >= trainer.callback_metrics["train_top1"].item()
        )
        assert trainer.callback_metrics["train_top5"].item() <= 1
        assert trainer.callback_metrics["val_loss"].item() > 0
        assert trainer.callback_metrics["val_top1"].item() >= 0
        assert (
            trainer.callback_metrics["val_top5"].item()
            >= trainer.callback_metrics["val_top1"].item()
        )
        assert trainer.callback_metrics["val_top5"].item() <= 1

        # Verify that model weights were not updated.
        assert torch.all(torch.eq(initial_weights, linear.weight))
        assert torch.all(torch.eq(initial_bn_weights, batch_norm.weight))
        # Verify that batch norm statistics were not updated. Note that even though the
        # running mean was not updated, it is still initialized to zero.
        assert batch_norm.running_mean is not None
        assert torch.all(
            torch.eq(batch_norm.running_mean, torch.zeros_like(batch_norm.running_mean))
        )
        # Verify that head weights were updated.
        assert not torch.all(
            torch.eq(initial_head_weights, linear_classifier.classification_head.weight)
        )
