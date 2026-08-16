"""SimCLR on Lightning, parameterised by one row of datasets.py.

The same method as ``examples/simclr.py``, written for the other job. The example
is the file to read; this one has to survive a week on a preemptible cluster, so
it hands the loop, the checkpoint and the step counter to Lightning. What it adds
on top of the example is the schedule, the probes, the logger, bf16, a checkpoint
and the dataset axis.

The two files restate each other on purpose, and ``tests/test_simclr_agrees.py``
is what holds them to the same method. They do not share numbers: the example is
one small configuration, the ``imagenet`` row is the paper's.

Run it with::

    torchrun --nproc_per_node=8 -m benchmarks.simclr.benchmark \\
        --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val

    python -m benchmarks.simclr.benchmark --dataset cifar10 \\
        --train-dir ./cifar/train --val-dir ./cifar/val
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, no_grad
from torch.optim import SGD, Optimizer
from torchvision.models import resnet18, resnet50

from benchmarks.bench import ImageDataModule
from benchmarks.bench.probes import KNNProbe
from benchmarks.simclr.datasets import SETTINGS, Setting
from lightly.backbones import TorchvisionResNetBackbone, small_image_stem
from lightly.data.sample import Sample
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.optim import LARS, CosineWarmupScheduler, param_groups
from lightly.transforms import SimCLRTransform
from lightly.utils.benchmarking import OnlineLinearClassifier

BACKBONES = {"resnet18": resnet18, "resnet50": resnet50}
OPTIMIZERS = {"lars": LARS, "sgd": SGD}


def backbone(setting: Setting) -> TorchvisionResNetBackbone:
    """Builds the row's backbone. Restated from the example; the gate compares it."""
    net = BACKBONES[setting.backbone]()
    return TorchvisionResNetBackbone(
        small_image_stem(net) if setting.stem == "small_image" else net
    )


def transform(setting: Setting) -> SimCLRTransform:
    """Builds the row's transform. Restated from the example; the gate compares it."""
    return SimCLRTransform(
        input_size=setting.size,
        cj_strength=setting.color_jitter_strength,
        gaussian_blur=setting.blur,
        normalize=setting.normalize,
    )


class SimCLR(LightningModule):
    """SimCLR, for one row.

    Args:
        setting: The row from ``datasets.py`` to run.
    """

    def __init__(self, setting: Setting) -> None:
        super().__init__()
        self.save_hyperparameters({"setting": setting.name})
        self.setting = setting
        self.backbone = backbone(setting)
        self.head = SimCLRProjectionHead(
            setting.feature_dim, setting.hidden_dim, setting.out_dim
        )
        self.criterion = NTXentLoss(
            temperature=setting.temperature, gather_distributed=True
        )

        # The probes stay inside the module until eval/ is public. Their optimiser
        # is this one, which is what the published numbers were produced with.
        self.online_classifier = OnlineLinearClassifier(
            feature_dim=setting.feature_dim, num_classes=setting.num_classes
        )
        self.knn_probe = KNNProbe(
            num_classes=setting.num_classes, k=setting.knn_k, t=setting.knn_t
        )

    def forward(self, sample: Sample) -> Tensor:
        # Both views go through the backbone and the head in one call, so every
        # BatchNorm sees 2N samples. Running one view at a time gives it N, which
        # is a different optimisation problem: the gradients of the two agree at
        # cosine similarity 0.0837.
        images = torch.cat([view.data for view in sample.views])
        z0, z1 = self.head(self.backbone.embed(images)).chunk(len(sample.views))
        loss: Tensor = self.criterion(z0, z1)
        return loss

    def training_step(self, sample: Sample, batch_idx: int) -> Tensor:
        loss = self(sample)
        targets = sample.meta["target"]
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # The probe reads the encoder, not the projection, so it takes its own
        # forward. One view under no_grad, where v1 reused both views from the
        # SSL pass, so val_online_cls_top* moves against the published number.
        with no_grad():
            features = self.backbone.embed(sample.views[0].data)
        cls_loss, cls_log = self.online_classifier.training_step(
            (features, targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> Tensor | None:
        images, targets = batch[0], batch[1]
        with no_grad():
            features = self.backbone.embed(images)

        if dataloader_idx == 0:
            # Loader 0 is the training set under the validation transform: the
            # kNN bank. Loader 1 is the validation set, scored against it.
            if batch_idx == 0:
                self.knn_probe.reset()
            self.knn_probe.add(features, targets)
            return None

        if batch_idx == 0:
            self.knn_probe.build(gather=self.all_gather, device=self.device)
        knn_log = self.knn_probe.score(features, targets)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features, targets), batch_idx
        )
        self.log_dict(
            {**knn_log, **cls_log},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )
        return cls_loss

    def on_validation_epoch_end(self) -> None:
        self.knn_probe.reset()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        setting = self.setting
        groups = param_groups(
            self.backbone, self.head, weight_decay=setting.weight_decay
        )
        groups.append(
            {
                "name": "online_classifier",
                "params": list(self.online_classifier.parameters()),
                "weight_decay": 0.0,
            }
        )
        optimizer = OPTIMIZERS[setting.optimizer](
            groups, lr=setting.lr, momentum=setting.momentum
        )
        steps_per_epoch = int(
            self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=steps_per_epoch * setting.warmup_epochs,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def parse_args() -> Namespace:
    parser = ArgumentParser("SimCLR benchmark")
    parser.add_argument(
        "--dataset", type=str, default="imagenet", choices=list(SETTINGS)
    )
    parser.add_argument("--train-dir", type=Path, default="/datasets/imagenet/train")
    parser.add_argument("--val-dir", type=Path, default="/datasets/imagenet/val")
    parser.add_argument("--out", type=Path, default=Path("benchmark_logs"))
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument(
        "--strategy", type=str, default="ddp_find_unused_parameters_true"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed, workers=True)
    setting = SETTINGS[args.dataset]

    # Setting.batch_size is the total across ranks, so the loaders get a share.
    world_size = args.devices if args.devices > 0 else max(torch.cuda.device_count(), 1)
    if setting.batch_size % world_size:
        raise ValueError(
            f"the {setting.name} row's batch size {setting.batch_size} does not "
            f"divide into {world_size} devices"
        )
    batch_size_per_device = setting.batch_size // world_size

    Trainer(
        max_epochs=setting.epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision="bf16-mixed",
        default_root_dir=str(args.out),
        logger=TensorBoardLogger(save_dir=str(args.out), name=setting.name),
        callbacks=[
            ModelCheckpoint(save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        fast_dev_run=args.fast_dev_run,
    ).fit(
        SimCLR(setting),
        ImageDataModule(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            transform=transform(setting),
            batch_size=batch_size_per_device,
            size=setting.size,
            normalize=setting.normalize,
            num_workers=args.num_workers,
        ),
        # Resume is one argument.
        ckpt_path="last" if not args.fast_dev_run else None,
    )
