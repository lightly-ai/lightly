import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    log += "".join(traceback.format_stack())
    print(log)


warnings.showwarning = warn_with_traceback


from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import barlowtwins
import byol
import dcl
import dclw
import dino
import finetune_eval
import knn_eval
import linear_eval
import mocov2
import simclr
import swav
import tico
import torch
import vicreg
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from lightly.data import LightlyDataset
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
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
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--ckpt-path", type=Path, default=None)
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--num-classes", type=int, default=1000)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--knn-k", type=int, default=200)
parser.add_argument("--knn-t", type=float, default=0.1)
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")
parser.add_argument("--float32-matmul-precision", type=str, default="high")
parser.add_argument("--strategy", default="ddp_find_unused_parameters_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--verbose", action="store_true", help="Print full configuration before training"
)

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
    "simclr": {"model": simclr.SimCLR, "transform": simclr.transform},
    "swav": {"model": swav.SwAV, "transform": swav.transform},
    "tico": {"model": tico.TiCo, "transform": tico.transform},
    "vicreg": {"model": vicreg.VICReg, "transform": vicreg.transform},
}


def create_full_config(
    args_dict: Dict,
    method: str,
    method_dir: Path,
    world_size: int,
) -> Dict:
    """Create full resolved configuration with derived values.

    Args:
        args_dict: Dictionary of parsed arguments.
        method: SSL method name.
        method_dir: Directory for logging this method.
        world_size: Number of devices/processes.

    Returns:
        Full configuration dictionary with derived values.
    """
    config = args_dict.copy()

    # Add derived values
    config["method"] = method
    config["world_size"] = world_size
    config["global_batch_size"] = args_dict["batch_size_per_device"] * world_size

    # Add effective learning rate (typical SimCLR scaling: lr * sqrt(batch_size))
    # Note: Actual LR is computed in each method's configure_optimizers
    base_lr = 0.075  # SimCLR base learning rate
    config["effective_lr_approx"] = base_lr * (config["global_batch_size"] ** 0.5)

    # Add environment info
    config["pytorch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    config["timestamp"] = datetime.now().isoformat()
    config["log_directory"] = str(method_dir)

    # Convert Path objects to strings for serialization
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)

    return config


def save_config(config: Dict, output_dir: Path) -> None:
    """Save configuration to a text file.

    Args:
        config: Configuration dictionary to save.
        output_dir: Directory to save config file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as plain text
    config_path = output_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write("FULL RESOLVED CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print_rank_zero(f"  💾 Config saved to: {config_path}")


def print_config(config: Dict) -> None:
    """Pretty print the full configuration.

    Args:
        config: Configuration dictionary to print.
    """
    print_rank_zero("\n" + "=" * 80)
    print_rank_zero("🔧 FULL RESOLVED CONFIGURATION")
    print_rank_zero("=" * 80)

    # Group by category
    categories = {
        "📂 Paths": ["train_dir", "val_dir", "log_dir", "log_directory", "ckpt_path"],
        "🧠 Method": ["method", "methods"],
        "📊 Data": [
            "num_classes",
            "batch_size_per_device",
            "global_batch_size",
            "num_workers",
        ],
        "🎓 Training": ["epochs", "effective_lr_approx", "float32_matmul_precision"],
        "📈 Evaluation": [
            "skip_knn_eval",
            "skip_linear_eval",
            "skip_finetune_eval",
            "knn_k",
            "knn_t",
        ],
        "💻 Hardware": [
            "accelerator",
            "devices",
            "world_size",
            "precision",
            "strategy",
            "cuda_available",
        ],
        "🔧 Other": ["seed", "compile_model", "pytorch_version", "timestamp"],
    }

    for category, keys in categories.items():
        matching_items = [(k, v) for k, v in config.items() if k in keys]
        if matching_items:
            print_rank_zero(f"\n{category}:")
            for key, value in matching_items:
                print_rank_zero(f"  {key}: {value}")

    # Print any remaining keys not categorized
    categorized_keys = set(k for keys in categories.values() for k in keys)
    remaining = [(k, v) for k, v in config.items() if k not in categorized_keys]
    if remaining:
        print_rank_zero("\n📝 Additional:")
        for key, value in remaining:
            print_rank_zero(f"  {key}: {value}")

    print_rank_zero("=" * 80 + "\n")


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
    knn_k: int,
    knn_t: float,
    skip_knn_eval: bool,
    skip_linear_eval: bool,
    skip_finetune_eval: bool,
    ckpt_path: Union[Path, None],
    float32_matmul_precision: str,
    strategy: str,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> None:
    # Store args for config creation
    args_dict = locals().copy()

    seed_everything(seed, workers=True, verbose=True)
    torch.set_float32_matmul_precision(float32_matmul_precision)

    method_names = methods or METHODS.keys()

    for method in method_names:
        method_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        print_rank_zero(f"Logging to {method_dir}")

        # Create and save full resolved config
        world_size = devices if accelerator != "cpu" else 1
        config = create_full_config(args_dict, method, method_dir, world_size)

        # Print config if verbose flag is set
        if verbose:
            print_config(config)

        # Save config files
        save_config(config, method_dir)

        model = METHODS[method]["model"](
            batch_size_per_device=batch_size_per_device, num_classes=num_classes
        )

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print_rank_zero("Compiling model...")
            model = torch.compile(model)

        if epochs <= 0:
            print_rank_zero("Epochs <= 0, skipping pretraining.")
            if ckpt_path is not None:
                model.load_state_dict(torch.load(ckpt_path)["state_dict"])
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
                knn_k=knn_k,
                knn_t=knn_t,
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
    strategy: str,
) -> None:
    print_rank_zero(f"Running pretraining for {method}...")

    # Setup training data.
    train_transform = METHODS[method]["transform"]
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
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
        persistent_workers=num_workers > 0,
    )

    # Train model.
    metric_callback = MetricCallback()
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
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
        precision=precision,
        strategy=strategy,
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
