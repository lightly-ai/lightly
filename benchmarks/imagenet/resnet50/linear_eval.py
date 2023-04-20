from pathlib import Path

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import LinearClassifier


class LinearEvalClassifier(LinearClassifier):
    def __init__(self, model: Module, batch_size: int) -> None:
        super().__init__(
            model=model, feature_dim=2048, num_classes=1000, batch_size=batch_size
        )

    def on_fit_start(self) -> None:
        # Freeze model weights.
        deactivate_requires_grad(model=self.model)

    def on_fit_end(self) -> None:
        # Unfreeze model weights.
        activate_requires_grad(model=self.model)

    def configure_optimizers(self):
        optimizer = SGD(
            self.classification_head.parameters(),
            lr=0.1
            * self.batch_size
            * self.trainer.num_devices
            * self.trainer.num_nodes
            / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        return optimizer


def linear_eval(
    model: Module,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
) -> None:
    print("Running linear evaluation...")

    # Setup training data.
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Setup validation data.
    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Train linear classifier.
    classifier = LinearEvalClassifier(model=model, batch_size=batch_size)
    trainer = Trainer(
        max_epochs=90,
        accelerator=accelerator,
        devices=devices,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="linear_eval"),
        precision=precision,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
