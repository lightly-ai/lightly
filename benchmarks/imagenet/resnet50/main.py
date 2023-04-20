from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import knn_eval
import linear_eval
import simclr
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.utils import IMAGENET_NORMALIZE

parser = ArgumentParser("ImageNet ResNet50 Benchmarks")
parser.add_argument("--train-dir", type=Path, required=True)
parser.add_argument("--val-dir", type=Path, required=True)
parser.add_argument("--log-dir", type=Path, default="benchmark_logs")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16")
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--no-knn-eval", action="store_true")
parser.add_argument("--no-linear-eval", action="store_true")
parser.add_argument("--no-finetune-eval", action="store_true")

METHODS = {
    "simclr": {"model": simclr.SimCLR, "transform": simclr.transform},
}

def main(
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    compile_model: bool,
    methods: Union[Sequence[str], None],
    no_knn_eval: bool,
    no_linear_eval: bool,
    no_finetune_eval: bool,
) -> None:
    method_names = methods or METHODS.keys()

    for method in method_names:
        print(f"Running pretraining for {method}...")
        # Setup training data.
        train_transform = METHODS[method]["transform"]
        train_dataset = LightlyDataset(
            input_dir=str(train_dir), transform=train_transform
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
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
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )
        val_dataset = LightlyDataset(
            input_dir=str(val_dir), transform=val_transform
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Train model.
        log_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        callbacks = [LearningRateMonitor(logging_interval="step")]
        if not no_knn_eval:
            print("Adding KNN eval callback...")
            callbacks.append(
                knn_eval.get_knn_eval_callback(
                    train_dir=train_dir,
                    val_dir=val_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    devices=devices,
                )
            )

        trainer = Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
            precision=precision,
        )
        model = METHODS[method]["model"](batch_size=batch_size, epochs=epochs)

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print("Compiling model...")
            model = torch.compile(model)

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        if not no_linear_eval:
            linear_eval.linear_eval(
                model=model,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=log_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
            )

        if not no_finetune_eval:
            # TODO: Implement finetune eval.
            print("Finetune eval is not yet implemented.")
            pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
