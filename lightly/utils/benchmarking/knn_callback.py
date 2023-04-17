from typing import Sequence

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from lightly.utils.benchmarking.knn import knn_predict
from lightly.utils.benchmarking.topk import topk_accuracy


class KNNCallback(Callback):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
        topk: Sequence[int] = (1, 5),
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.topk = topk
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs > 1 and trainer.current_epoch % self.every_n_epochs != 0:
            return

        train_features = []
        train_targets = []
        for batch in self.train_dataloader:
            images, targets, _ = batch
            images = images.to(pl_module.device)
            targets = targets.to(pl_module.device)
            features = pl_module.forward(images).flatten(start_dim=1)
            features = F.normalize(features, dim=1)
            train_features.append(features)
            train_targets.append(targets)

        all_train_features = (
            torch.cat(pl_module.all_gather(train_features), dim=0).t().contiguous()
        )
        all_train_targets = (
            torch.cat(pl_module.all_gather(train_targets), dim=0).t().contiguous()
        )

        accuracies = {"top{k}": [] for k in self.topk}
        for batch in self.val_dataloader:
            images, targets, _ = batch
            images = images.to(pl_module.device)
            targets = targets.to(pl_module.device)
            features = pl_module.forward(images).flatten(start_dim=1)
            features = F.normalize(features, dim=1)
            predictions = knn_predict(
                feature=features,
                feature_bank=all_train_features,
                feature_labels=all_train_targets,
                num_classes=self.num_classes,
                knn_k=self.knn_k,
                knn_t=self.knn_t,
            )
            acc = topk_accuracy(predictions=predictions, targets=targets, k=self.topk)
            for k, a in zip(self.topk, acc):
                accuracies[f"top{k}"].append(a.cpu())

        if trainer.global_rank == 0:
            all_accuracies = {
                "top{k}_val": torch.cat(
                    pl_module.all_gather(accuracies["top{k}"]), dim=0
                )
            }
            mean_accuracies = {k: torch.mean(v) for k, v in all_accuracies.items()}
            pl_module.log_dict(
                mean_accuracies, prog_bar=True, rank_zero_only=True,
            )
