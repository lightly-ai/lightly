from pathlib import Path
from typing import Dict

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from lightly.data import LightlyDataset
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero

class KNNEvalCallback(Callback):
    """
    A PyTorch Lightning Callback that runs a KNN evaluation at the end 
    of every validation epoch during pretraining.
    
    This allows us to track the model's feature-learning progress over time 
    without waiting for the full pretraining process to finish.
    """
    def __init__(
        self, 
        train_dir: Path,
        val_dir: Path,
        batch_size_per_device: int,
        num_workers: int,
        accelerator: str,
        log_dir: Path,
        devices: int,
        strategy: str,
        num_classes: int, 
        knn_k: int, 
        knn_t: float):
        """
        Args:
            train_dir: Path to the training data directory.
            val_dir: Path to the validation data directory.
            batch_size_per_device: Number of images per batch per GPU.
            num_workers: Number of workers for the dataloaders.
            accelerator: Hardware accelerator (e.g., 'gpu').
            log_dir: Directory to save TensorBoard logs.
            devices: Number of devices to use.
            strategy: Distributed training strategy.
            num_classes: Total number of classes in the dataset.
            knn_k: Number of nearest neighbors to retrieve.
            knn_t: Temperature parameter to weight the neighbors.
        """

        super().__init__()

        self.accelerator=accelerator
        self.log_dir=log_dir
        self.devices=devices
        self.strategy=strategy
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        print_rank_zero("Running KNN evaluation...")

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
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_device,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        self.train_dataloader = train_dataloader

        # Setup validation data.
        val_dataset = LightlyDataset(input_dir=str(val_dir), transform=transform)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_per_device,
            shuffle=False,
            num_workers=num_workers,
        )
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Triggers automatically when the main pretraining validation loop ends.
        Evaluates the current state of the model using KNN and logs the metrics.
        """

        classifier = KNNClassifier(
        model=pl_module,
        num_classes=self.num_classes,
        knn_k=self.knn_k,
        knn_t=self.knn_t,
        )

        # Run KNN evaluation.
        metric_callback = MetricCallback()
        knn_trainer = Trainer(
            max_epochs=1,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=TensorBoardLogger(save_dir=str(self.log_dir), name="knn_eval"),
            callbacks=[
                DeviceStatsMonitor(),
                metric_callback,
            ],
            strategy=self.strategy,
            num_sanity_val_steps=0,  # NOTE: save shared memory usage
        )

        # Initialize a temporary, lightweight trainer just for this KNN evaluation step
        knn_trainer.validate(
        model=classifier,
        dataloaders=[self.train_dataloader, self.val_dataloader],
        verbose=False,
        )

        # Broadcast the scores back to the main training process
        metrics_dict: dict[str, float] = dict()
        for metric in ["val_top1", "val_top5"]:
            for name, value in metric_callback.val_metrics.items():
                if name.startswith(metric):
                    print_rank_zero(f"knn {name}: {max(value)}")
                    metrics_dict[name] = max(value)

                    # Logging the metrics
                    pl_module.log(
                        f"knn_{name}", 
                        max(value), 
                        sync_dist=True
                    )
