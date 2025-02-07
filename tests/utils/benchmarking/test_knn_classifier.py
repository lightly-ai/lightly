from typing import Tuple

import pytest
import torch
from pytest_mock import MockerFixture
from pytorch_lightning import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lightly.utils.benchmarking import KNNClassifier
from lightly.utils.benchmarking.knn_classifier import F


class TestKNNClassifier:
    def test(self) -> None:
        # Define 4 training points from 4 classes.
        train_features = torch.tensor(
            [
                [0.0, -1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        train_targets = torch.tensor([0, 1, 2, 3])
        train_dataset = _FeaturesDataset(features=train_features, targets=train_targets)

        # Define 3 validation points.
        # Their expected predicted labels are their closest training points in order.
        val_features = torch.tensor(
            [
                [0.0, -0.4],  # predicted_labels = [0, 1, 2, 3]
                [0.6, 0.7],  # predicted_labels = [3, 1, 2, 0]
                [0.6, 0.3],  # predicted_labels = [2, 3, 1, 0]
            ]
        )
        val_targets = torch.tensor([0, 0, 1])
        val_dataset = _FeaturesDataset(features=val_features, targets=val_targets)

        train_dataloader = DataLoader(train_dataset, batch_size=2)
        val_dataloader = DataLoader(val_dataset, batch_size=2)

        # Run KNN classifier.
        model = nn.Identity()
        classifier = KNNClassifier(model, num_classes=4, knn_k=3, topk=(1, 2, 3, 4))
        trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1)
        trainer.validate(
            model=classifier,
            dataloaders=[train_dataloader, val_dataloader],
        )
        assert trainer.callback_metrics["val_top1"].item() == pytest.approx(1 / 3)
        assert trainer.callback_metrics["val_top2"].item() == pytest.approx(1 / 3)
        assert trainer.callback_metrics["val_top3"].item() == pytest.approx(2 / 3)
        assert trainer.callback_metrics["val_top4"].item() == pytest.approx(3 / 3)

    def test__cpu(self) -> None:
        self._test__accelerator(accelerator="cpu", expected_device="cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda available")
    def test__cuda(self) -> None:
        self._test__accelerator(accelerator="gpu", expected_device="cuda")

    def _test__accelerator(self, accelerator: str, expected_device: str) -> None:
        torch.manual_seed(0)
        linear = nn.Linear(3, 2)
        batch_norm = nn.BatchNorm1d(2)
        model = nn.Sequential(linear, batch_norm)
        initial_weights = linear.weight.clone()
        initial_bn_weights = batch_norm.weight.clone()
        classifier = KNNClassifier(model, num_classes=10, knn_k=20)
        trainer = Trainer(max_epochs=1, accelerator=accelerator, devices=1)
        train_features = torch.randn(40, 3)
        train_targets = torch.randint(0, 10, (40,))
        train_dataset = _FeaturesDataset(features=train_features, targets=train_targets)
        val_features = torch.randn(10, 3)
        val_targets = torch.randint(0, 10, (10,))
        val_dataset = _FeaturesDataset(features=val_features, targets=val_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=3)
        val_dataloader = DataLoader(val_dataset, batch_size=3)
        trainer.validate(
            model=classifier,
            dataloaders=[train_dataloader, val_dataloader],
        )
        assert trainer.callback_metrics["val_top1"].item() >= 0.0
        assert (
            trainer.callback_metrics["val_top5"].item()
            >= trainer.callback_metrics["val_top1"].item()
        )
        assert trainer.callback_metrics["val_top5"].item() <= 1.0
        assert classifier._train_features == []
        assert classifier._train_targets == []
        assert classifier._train_features_tensor is not None
        assert classifier._train_targets_tensor is not None
        assert classifier._train_features_tensor.shape == (2, 40)
        assert classifier._train_targets_tensor.shape == (40,)
        assert classifier._train_features_tensor.dtype == torch.float32
        assert classifier._train_features_tensor.device.type == expected_device
        assert classifier._train_targets_tensor.device.type == expected_device

        # Verify that model weights were not updated.
        assert torch.all(torch.eq(initial_weights, linear.weight))
        assert torch.all(torch.eq(initial_bn_weights, batch_norm.weight))
        # Verify that batch norm statistics were not updated. Note that even though the
        # running mean was not updated, it is still initialized to zero.
        assert batch_norm.running_mean is not None
        assert torch.all(
            torch.eq(batch_norm.running_mean, torch.zeros_like(batch_norm.running_mean))
        )

    def test__features_dtype(self) -> None:
        model = nn.Identity()
        # Set feature_dtype to torch.int to test if classifier correctly changes dtype.
        # We cannot test for torch.float16 because it is not supported on cpu.
        classifier = KNNClassifier(
            model, num_classes=10, knn_k=3, feature_dtype=torch.int
        )
        trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1)
        train_features = torch.randn(4, 3)
        train_targets = torch.randint(0, 10, (4,))
        train_dataset = _FeaturesDataset(features=train_features, targets=train_targets)
        val_features = torch.randn(4, 3)
        val_targets = torch.randint(0, 10, (4,))
        val_dataset = _FeaturesDataset(features=val_features, targets=val_targets)
        train_dataloader = DataLoader(train_dataset)
        val_dataloader = DataLoader(val_dataset)
        trainer.validate(
            model=classifier,
            dataloaders=[train_dataloader, val_dataloader],
        )
        assert classifier._train_features_tensor is not None
        assert classifier._train_features_tensor.dtype == torch.int

    def test__normalize(self, mocker: MockerFixture) -> None:
        train_features = torch.randn(4, 3)
        train_targets = torch.randint(0, 10, (4,))
        train_dataset = _FeaturesDataset(features=train_features, targets=train_targets)
        val_features = torch.randn(4, 3)
        val_targets = torch.randint(0, 10, (4,))
        val_dataset = _FeaturesDataset(features=val_features, targets=val_targets)
        train_dataloader = DataLoader(train_dataset)
        val_dataloader = DataLoader(val_dataset)
        trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1)
        spy_normalize = mocker.spy(F, "normalize")

        # Test that normalize is called when normalize=True.
        classifier = KNNClassifier(
            nn.Identity(), num_classes=10, knn_k=3, normalize=True
        )
        trainer.validate(
            model=classifier,
            dataloaders=[train_dataloader, val_dataloader],
        )
        spy_normalize.assert_called()
        spy_normalize.reset_mock()

        trainer.validate(model=classifier, dataloaders=val_dataloader)
        spy_normalize.assert_called()
        spy_normalize.reset_mock()

        # Test that normalize is not called when normalize=False.
        classifier = KNNClassifier(
            nn.Identity(), num_classes=10, knn_k=3, normalize=False
        )
        trainer.validate(
            model=classifier,
            dataloaders=[train_dataloader, val_dataloader],
        )
        spy_normalize.assert_not_called()
        spy_normalize.reset_mock()

        trainer.validate(model=classifier, dataloaders=val_dataloader)
        spy_normalize.assert_not_called()
        spy_normalize.reset_mock()


class _FeaturesDataset(Dataset):
    def __init__(self, features: Tensor, targets) -> None:
        super().__init__()
        self.features = features
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.features)
