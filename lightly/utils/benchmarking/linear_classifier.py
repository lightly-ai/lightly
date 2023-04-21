import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from typing import Tuple

from lightly.utils.benchmarking.topk import topk_accuracy


class LinearClassifier(LightningModule):
    def __init__(
        self,
        model: Module,
        feature_dim: int,
        num_classes: int,
        batch_size: int,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.topk = topk

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

        self._accuracy = {
            "train": {k: [] for k in self.topk},
            "val": {k: [] for k in self.topk},
        }
        self._loss = {"train": [], "val": []}

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.forward(x).flatten(start_dim=1)
        return self.classification_head(features)

    def shared_step(self, batch, batch_idx, name: str) -> Tensor:
        images, targets, _ = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        topk = topk_accuracy(predictions.detach(), targets.detach(), k=self.topk)

        log_dict = {
            f"{name}_lin_cls_top{k}_step": acc.mean() for k, acc in topk.items()
        }
        log_dict[f"{name}_lin_cls_loss_step"] = loss
        self.log_dict(log_dict, batch_size=self.batch_size)
        for k, acc in topk.items():
            self._accuracy[name][k].append(acc.cpu())
        self._loss[name].extend([loss.detach().cpu()] * len(images))
        return loss

    def shared_on_epoch_end(self, name: str) -> None:
        topk = {
            k: self.all_gather(torch.cat(acc))
            for k, acc in self._accuracy[name].items()
        }
        loss = self.all_gather(torch.cat(self._loss[name]))
        log_dict = {
            f"{name}_lin_cls_top{k}_epoch": acc.mean() for k, acc in topk.items()
        }
        log_dict[f"{name}_lin_cls_loss_epoch"] = loss.mean()
        self.log_dict(log_dict, prog_bar=True)
        for val in self._accuracy[name].values():
            val.clear()
        self._loss[name].clear()

    def training_step(self, batch, batch_idx) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, name="train")

    def validation_step(self, batch, batch_idx) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, name="val")

    def on_train_epoch_end(self) -> None:
        return self.shared_on_epoch_end(name="train")

    def on_validation_epoch_end(self) -> None:
        return self.shared_on_epoch_end(name="val")
