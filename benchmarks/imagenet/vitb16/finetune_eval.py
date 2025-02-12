from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from timm.data import create_transform
from timm.data.mixup import Mixup
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.models.utils import (
    add_stochastic_depth_to_blocks,
    get_named_leaf_modules,
    get_weight_decay_parameters,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import FinetuneClassifier, MetricCallback
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.dist import print_rank_zero
from lightly.utils.scheduler import CosineWarmupScheduler


class FinetuneClassifierMAE(FinetuneClassifier):
    # Parameters follow MAE settings.
    # Adapt initialization to include mixup.
    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        lr: float = 5e-4,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__(
            model,
            batch_size_per_device,
            lr,
            feature_dim,
            num_classes,
            topk,
        )
        # TODO(Ersi, 2/24): Add path dropout for TIMM.

        # Add path dropout.
        add_stochastic_depth_to_blocks(self.model, prob=0.1)
        # Add mixup and cutmix.
        self.mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            label_smoothing=0.1,
            num_classes=num_classes,
        )

    # Adapt step to include mixup.
    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        if self.trainer.state.stage == "train":
            images, targets = self.mixup(images, targets)
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        # Pass targets without mixup for topk accuracy calculation.
        topk = mean_topk_accuracy(predicted_labels, batch[1], k=self.topk)

        return loss, topk

    # Adapt optimizer to match MAE settings. Parameters follow the original code from
    # the authors: https://github.com/facebookresearch/mae/blob/main/FINETUNE.md#fine-tuning
    # Note that lr and layerwise_lr_decay for ViT-B/16 are 1e-3 and 0.75 in the paper
    # but 5e-4 and 0.65 in the code.
    # Type ignore is needed because return type of LightningModule.configure_optimizers
    # is complicated and typing changes between versions.
    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        lr = self.get_effective_lr()
        layerwise_lr_decay = 0.65

        # Group parameters by weight decay and learning rate.
        param_groups: Dict[str, Dict[str, Any]] = {}
        for name, module in get_named_leaf_modules(self.model).items():
            if "encoder_layer_" in name:
                layer_index = int(name.split("encoder_layer_")[1].split(".")[0])
                group_name = f"vit_layer_{layer_index}"
                # ViT-B has 12 layers. LR increases from first layer with index 0 to
                # last layer with index 11.
                group_lr = lr * (layerwise_lr_decay ** (11 - layer_index))
            else:
                group_name = "vit"
                group_lr = lr
            params, params_no_weight_decay = get_weight_decay_parameters([module])
            group = param_groups.setdefault(
                group_name,
                {
                    "name": group_name,
                    "params": [],
                    "lr": group_lr,
                    "weight_decay": 0.05,
                },
            )
            group["params"].extend(params)
            group_no_weight_decay = param_groups.setdefault(
                f"{group_name}_no_weight_decay",
                {
                    "name": f"{group_name}_no_weight_decay",
                    "params": [],
                    "lr": group_lr,
                    "weight_decay": 0.0,
                },
            )
            group_no_weight_decay["params"].extend(params_no_weight_decay)
        param_groups["classification_head"] = {
            "name": "classification_head",
            "params": self.classification_head.parameters(),
            "weight_decay": 0.0,
        }
        optimizer = AdamW(
            list(param_groups.values()),
            betas=(0.9, 0.999),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 5
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def finetune_eval(
    model: Module,
    eval_method: str,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    strategy: str,
    num_classes: int,
) -> Dict[str, float]:
    """Runs fine-tune evaluation on the given model."""
    print_rank_zero("Running fine-tune evaluation...")

    # Setup training data.

    if eval_method == "mae":
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
        print_rank_zero("Using MAE training transform.")
    else:
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )
        print_rank_zero("Using default training transform.")

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
        strategy=strategy,
        num_sanity_val_steps=0,  # NOTE: prevent problems from warmup schedule or validation metrics
    )
    if eval_method == "mae":
        classifier = FinetuneClassifierMAE(
            model=model,
            batch_size_per_device=batch_size_per_device,
            feature_dim=model.online_classifier.feature_dim,
            num_classes=num_classes,
        )
        print_rank_zero("Using MAE finetune classifier.")
    else:
        classifier = FinetuneClassifier(
            model=model,
            batch_size_per_device=batch_size_per_device,
            feature_dim=model.online_classifier.feature_dim,
            num_classes=num_classes,
        )
        print_rank_zero("Using default finetune classifier.")

    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    metrics_dict: Dict[str, float] = dict()
    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(
            f"max finetune {metric}: {max(metric_callback.val_metrics[metric])}"
        )
        metrics_dict[metric] = max(metric_callback.val_metrics[metric])
    return metrics_dict
