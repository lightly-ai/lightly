from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Union, Optional
import barlowtwins
import byol
import dcl
import dclw
import dino
import finetune_eval
import knn_eval
import linear_eval
import mocov2
import simclr_muti
import simclr
import swav
import tico
import torch
import vicreg

import numpy as np
import os
import traceback
from tqdm import tqdm

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset, MultiLabelDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero

parser = ArgumentParser("ImageNet ResNet50 Benchmarks")
parser.add_argument("--train-dir", type=Path, default="/datasets/imagenet/train")
parser.add_argument("--val-dir", type=Path, default="/datasets/imagenet/val")
parser.add_argument("--log-dir", type=Path, default="benchmark_logs")
parser.add_argument("--batch-size-per-device", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=2)
parser.add_argument("--precision", type=str, default="16-mixed") 
parser.add_argument("--ckpt-path", type=Path, default=None)
parser.add_argument("--resume-from", type=Path, default=None)
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--num-classes", type=int, default=8)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")
parser.add_argument("--float32-matmul-precision", type=str, default="high")
parser.add_argument("--strategy", default="ddp_find_unused_parameters_true")
parser.add_argument("--seed", type=int, default=None)

METHODS = {
    "barlowtwins": {
        "model": barlowtwins.BarlowTwins,
        "transform": barlowtwins.transform,
    },
    "byol": {"model": byol.BYOL, "transform": byol.transform},
    "dcl": {"model": dcl.DCL, "transform": dcl.transform},
    "dclw": {"model": dclw.DCLW, "transform": dclw.transform},
    "dino": {"model": dino.DINO, "transform": dino.transform},
    "mocov2": {"model": mocov2.MoCoV2, "transform": mocov2.transform},
    "simclr": {"model": simclr_muti.SimCLR, "transform": simclr_muti.transform},
    "swav": {"model": swav.SwAV, "transform": swav.transform},
    "tico": {"model": tico.TiCo, "transform": tico.transform},
    "vicreg": {"model": vicreg.VICReg, "transform": vicreg.transform},
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
    ckpt_path: Union[Path, None],
    resume_from: Union[Path, None],
    float32_matmul_precision: str,
    strategy: str,
    seed: Optional[int] = None,
) -> None:
    print_rank_zero(f"Args: {locals()}")
    # seed_everything(seed, workers=True, verbose=True)
    torch.set_float32_matmul_precision(float32_matmul_precision)

    method_names = methods or METHODS.keys()

    for method in method_names:
        method_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        print_rank_zero(f"Logging to {method_dir}")
        # if resume_from is not None:
        #     checkpoint = torch.load(ckpt_path, weights_only=False)
        #     model.load_state_dict(checkpoint['state_dict'])
        model = METHODS[method]["model"](
            batch_size_per_device=batch_size_per_device, num_classes=num_classes
        )

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print_rank_zero("Compiling model...")
            model = torch.compile(model)

        # epoch迭代结束保存权重
        if epochs <= 0:
            print_rank_zero("Epochs <= 0, skipping pretraining.")
            if ckpt_path is not None:
                model.load_state_dict(torch.load(ckpt_path))["state_dict"]
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
                ckpt_path=ckpt_path,
                resume_from = resume_from,
                strategy=strategy,
            )
        eval_metrics: Dict[str, Dict[str, float]] = dict()
        if skip_knn_eval:
            print_rank_zero("Skipping KNN eval.")
        else:
            eval_metrics["knn"] = knn_eval.knn_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                strategy=strategy,
            )

        if skip_linear_eval:
            print_rank_zero("Skipping linear eval.")
        else:
            eval_metrics["linear"] = linear_eval.linear_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                strategy=strategy,
                precision=precision,
            )

        if skip_finetune_eval:
            print_rank_zero("Skipping fine-tune eval.")
        else:
            eval_metrics["finetune"] = finetune_eval.finetune_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                strategy=strategy,
                precision=precision,
            )

        if eval_metrics:
            print_rank_zero(f"Results for {method}:")
            print_rank_zero(eval_metrics_to_markdown(eval_metrics))


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
    ckpt_path: Union[Path, None],
    resume_from: Union[Path, None],
    strategy: str,
) -> None:
    print_rank_zero(f"Running pretraining for {method}...")
    all_classes = ['痰湿质','气郁质','气虚质','平和质','湿热质','阳虚质','阴虚质','血瘀质']
    # Setup training data.
    train_transform = METHODS[method]["transform"]
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
    )
    # train_transform = T.Compose(
    #     [
    #         T.Resize(256),
    #         # T.Resize(224),
    #         T.CenterCrop(224),
    #         T.ToTensor(),
    #         T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    #     ]
    # )
    # train_dataset = MultiLabelDataset(
    #     root_dir=str(val_dir),  # 验证集根目录
    #     all_classes=all_classes,
    #     transform=train_transform
    # )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size_per_device,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     drop_last=True,
    #     pin_memory=False
    # )

    # Setup validation data.
    val_transform = T.Compose(
        [
            T.Resize(256),
            # T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    # val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size_per_device,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     persistent_workers=True,
    # )
    # 多标签验证
    
    val_dataset = MultiLabelDataset(
        root_dir=str(val_dir),  # 验证集根目录
        all_classes=all_classes,
        transform=val_transform
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False
    )

    # Train model.
    if resume_from is not None:
        checkpoint = torch.load(resume_from, weights_only=False)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()
    
        # 过滤出预训练字典中与当前模型匹配的键
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
    
        model_dict.update(pretrained_dict) # 更新当前模型状态字典
        model.load_state_dict(model_dict)
        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")

    metric_callback = MetricCallback()
    # 权重保存逻辑
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/simclr/pretrain",  # 保存目录
        filename="simclr_shezhen-{epoch}-{step}",  # 文件名格式
        monitor="train_loss",         # 监控的指标（可选）
        save_top_k=1,                 # 保留最好的k个模型
        mode="min",                   # 监控模式（min/max）
        save_last=True,               # 额外保存最后一个epoch的模型
    )
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            # Stop if training loss diverges.
            EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
            DeviceStatsMonitor(),
            metric_callback,
            checkpoint_callback,
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
        precision=precision,
        strategy="ddp",
        # find_unused_parameters=True,
        sync_batchnorm=accelerator != "cpu",  # Sync batchnorm is not supported on CPU.
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
        print_rank_zero(
            f"max {metric}: {max(metric_callback.val_metrics.get(metric, [-1]))}"
        )


def eval_metrics_to_markdown(metrics: Dict[str, Dict[str, float]]) -> str:
    EVAL_NAME_COLUMN_NAME = "Eval Name"
    METRIC_COLUMN_NAME = "Metric Name"
    VALUE_COLUMN_NAME = "Value"

    eval_name_max_len = max(
        len(eval_name) for eval_name in list(metrics.keys()) + [EVAL_NAME_COLUMN_NAME]
    )
    metric_name_max_len = max(
        len(metric_name)
        for metric_dict in metrics.values()
        for metric_name in list(metric_dict.keys()) + [METRIC_COLUMN_NAME]
    )
    value_max_len = max(
        len(metric_value)
        for metric_dict in metrics.values()
        for metric_value in list(f"{value:.2f}" for value in metric_dict.values())
        + [VALUE_COLUMN_NAME]
    )

    header = f"| {EVAL_NAME_COLUMN_NAME.ljust(eval_name_max_len)} | {METRIC_COLUMN_NAME.ljust(metric_name_max_len)} | {VALUE_COLUMN_NAME.ljust(value_max_len)} |"
    separator = f"|:{'-' * (eval_name_max_len)}:|:{'-' * (metric_name_max_len)}:|:{'-' * (value_max_len)}:|"

    lines = [header, separator] + [
        f"| {eval_name.ljust(eval_name_max_len)} | {metric_name.ljust(metric_name_max_len)} | {f'{metric_value:.2f}'.ljust(value_max_len)} |"
        for eval_name, metric_dict in metrics.items()
        for metric_name, metric_value in metric_dict.items()
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
