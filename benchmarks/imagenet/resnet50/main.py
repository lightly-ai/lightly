from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

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

parser = ArgumentParser("ImageNet ResNet50")
parser.add_argument("--train-dir", type=Path, required=True)
parser.add_argument("--val-dir", type=Path, required=True)
parser.add_argument(
    "--log-dir",
    type=Path,
    default=Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "benchmark_logs"
    / "imagenet"
    / "resnet50",
)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--no-knn-eval", action="store_true")
parser.add_argument("--no-linear-eval", action="store_true")
parser.add_argument("--no-finetune-eval", action="store_true")

METHODS = {
    "simclr": {"model": simclr.SimCLR, "transform": simclr.transform},
}


if __name__ == "__main__":
    args = parser.parse_args()
    method_names = args.methods or METHODS.keys()

    for method in method_names:
        print(f"Running pretraining for {method}...")
        # Setup training data.
        train_transform = METHODS[method]["transform"]
        train_dataset = LightlyDataset(
            input_dir=str(args.train_dir), transform=train_transform
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
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
            input_dir=str(args.val_dir), transform=val_transform
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Train model.
        log_dir = (
            args.log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        callbacks = [LearningRateMonitor(logging_interval="step")]
        if not args.no_knn_eval:
            print("Adding KNN eval callback...")
            callbacks.append(
                knn_eval.get_knn_eval_callback(
                    train_dir=args.train_dir,
                    val_dir=args.val_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    devices=args.devices,
                )
            )

        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
            log_every_n_steps=1,  # TODO: remove
        )
        model = METHODS[method]["model"](batch_size=args.batch_size, epochs=args.epochs)

        if hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print("Compiling model...")
            model = torch.compile(model)

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        if not args.no_linear_eval:
            linear_eval.linear_eval(
                model=model,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                log_dir=log_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                accelerator=args.accelerator,
                devices=args.devices,
            )

        if not args.no_finetune_eval:
            # TODO: Implement finetune eval.
            print("Finetune eval is not yet implemented.")
            pass
