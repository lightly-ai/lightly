from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import KNNClassifier, MetricCallback


def knn_eval(
    model: LightningModule,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    num_classes: int,
) -> None:
    print("Running KNN evaluation...")

    # Setup trainer.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=str(log_dir), name="knn_eval"),
        callbacks=[
            DeviceStatsMonitor(),
            metric_callback,
        ],
        strategy="ddp_find_unused_parameters_true",
    )

    # Setup training data.
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=transform)
    assert batch_size % trainer.world_size == 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // trainer.world_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Setup validation data.
    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size // trainer.world_size,
        shuffle=False,
        num_workers=num_workers,
    )

    classifier = KNNClassifier(
        model=model,
        num_classes=num_classes,
        feature_dtype=torch.float16,
    )

    # Run KNN evaluation.
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    for metric in ["val_top1", "val_top5"]:
        print(f"knn {metric}: {max(metric_callback.val_metrics[metric])}")
