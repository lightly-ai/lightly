from pathlib import Path
from typing import Dict, Tuple

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from timm.data import create_transform
from timm.data.mixup import Mixup
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.models.utils import add_stochastic_depth_to_blocks
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler


class FinetuneEvalClassifier(LinearClassifier):
    # Parameters follow MAE settings.
    # Adapt initialization to include mixup.
    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        feature_dim: int,
        num_classes: int,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        super().__init__(
            model, batch_size_per_device, feature_dim, num_classes, topk, freeze_model
        )
        self.mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            label_smoothing=0.1,
            num_classes=num_classes,
        )

    # Adapt step to include mixup.
    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        if self.trainer.state.stage == "train":
            images, targets = self.mixup(images, targets)
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        # Pass targets without mixup for topk accuracy calculation.
        topk = mean_topk_accuracy(predicted_labels, batch[1], k=self.topk)
        return loss, topk

    # Adapt optimizer to match MAE settings.
    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        parameters += self.model.parameters()
        optimizer = AdamW(
            parameters,
            lr=1e-3 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.999),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 5
                ),
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

    Parameters follow MAE settings.
    """
    print("Running fine-tune evaluation...")

    # Add path dropout.
    add_stochastic_depth_to_blocks(model, prob=0.1)

    # Setup training data.
    # NOTE: We use transforms from the timm library here as they are the default in MAE
    # and torchvision does not provide all required parameters.
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_NORMALIZE["mean"],
        std=IMAGENET_NORMALIZE["std"],
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
        max_epochs=100,
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
    classifier = FinetuneEvalClassifier(
        model=model,
        batch_size_per_device=batch_size_per_device,
        feature_dim=768,
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
