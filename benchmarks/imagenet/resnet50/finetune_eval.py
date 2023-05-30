from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from lightly.utils.scheduler import CosineWarmupScheduler


class FinetuneLinearClassifier(LinearClassifier):
    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def finetune_eval(
    model: Module,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    num_classes: int,
) -> None:
    """Runs fine-tune evaluation on the given model.

    Parameters follow SimCLR settings.
    """
    print("Running fine-tune evaluation...")

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
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
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
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # Train linear classifier.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=30,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="finetune_eval"),
        precision=precision,
        strategy="ddp_find_unused_parameters_true",
    )
    classifier = FinetuneLinearClassifier(
        model=model,
        batch_size_per_device=batch_size_per_device,
        feature_dim=2048,
        num_classes=num_classes,
        freeze_model=False,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    for metric in ["val_top1", "val_top5"]:
        print(f"max finetune {metric}: {max(metric_callback.val_metrics[metric])}")
