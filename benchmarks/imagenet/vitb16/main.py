from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import aim
import finetune_eval
import knn_eval
import linear_eval
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback

parser = ArgumentParser("ImageNet ViT-B/16 Benchmarks")
parser.add_argument("--train-dir", type=Path, default="/datasets/imagenet/train")
parser.add_argument("--val-dir", type=Path, default="/datasets/imagenet/val")
parser.add_argument("--log-dir", type=Path, default="benchmark_logs")
parser.add_argument("--batch-size-per-device", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--num-classes", type=int, default=1000)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")
parser.add_argument("--float32-matmul-precision", type=str, default="high")
parser.add_argument("--strategy", default="ddp_find_unused_parameters_true")

METHODS = {
    "aim": {"model": aim.AIM, "transform": aim.transform},
}


def main(
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    compile_model: bool,
    methods: Union[Sequence[str], None],
    num_classes: int,
    skip_knn_eval: bool,
    skip_linear_eval: bool,
    skip_finetune_eval: bool,
    float32_matmul_precision: str,
    strategy: str,
) -> None:
    torch.set_float32_matmul_precision(float32_matmul_precision)

    method_names = methods or METHODS.keys()

    for method in method_names:
        method_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        model = METHODS[method]["model"](
            batch_size_per_device=batch_size_per_device, num_classes=num_classes
        )

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print("Compiling model...")
            model = torch.compile(model)

        if epochs <= 0:
            print("Epochs <= 0, skipping pretraining.")
        else:
            pretrain(
                model=model,
                method=method,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                epochs=epochs,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                strategy=strategy,
            )

        if skip_knn_eval:
            print("Skipping KNN eval.")
        else:
            knn_eval.knn_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
            )

        if skip_linear_eval:
            print("Skipping linear eval.")
        else:
            linear_eval.linear_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
            )

        if skip_finetune_eval:
            print("Skipping fine-tune eval.")
        else:
            finetune_eval.finetune_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
            )


def pretrain(
    model: LightningModule,
    method: str,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    strategy: str,
) -> None:
    print(f"Running pretraining for {method}...")

    # Setup training data.
    train_transform = METHODS[method]["transform"]
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=MultiViewCollate(),
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
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
    )

    # Train model.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
        precision=precision,
        strategy=strategy,
        sync_batchnorm=True,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
        print(f"max {metric}: {max(metric_callback.val_metrics[metric])}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
