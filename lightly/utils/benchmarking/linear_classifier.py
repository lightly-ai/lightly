import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from typing import Dict, Tuple
from torch.optim import SGD
from lightly.models.utils import activate_requires_grad, deactivate_requires_grad

from lightly.utils.benchmarking.topk import topk_accuracy


class LinearClassifier(LightningModule):
    def __init__(
        self,
        model: Module,
        feature_dim: int,
        num_classes: int,
        batch_size: int,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.topk = topk
        self.freeze_model = freeze_model

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

        self._outputs = {
            "train": {"loss": [], "topk": {k: [] for k in self.topk}},
            "val": {"loss": [], "topk": {k: [] for k in self.topk}},
        }

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.forward(x).flatten(start_dim=1)
        return self.classification_head(features)

    def shared_step(
        self, batch, batch_idx, name: str
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        topk = topk_accuracy(predictions.detach(), targets.detach(), k=self.topk)
        self._outputs[name]["loss"].append(loss.detach().repeat(len(images)).cpu())
        for k, acc in topk.items():
            self._outputs[name]["topk"][k].append(acc.cpu())
        return loss, topk

    def shared_on_epoch_end(self, name: str) -> None:
        loss = self.all_gather(torch.cat(self._outputs[name]["loss"]))
        topk = {
            k: self.all_gather(torch.cat(acc))
            for k, acc in self._outputs[name]["topk"].items()
        }
        log_dict = {f"{name}_loss": loss.mean()}
        log_dict.update({f"{name}_top{k}": acc.mean() for k, acc in topk.items()})
        self.log_dict(log_dict, prog_bar=True)
        self._outputs[name]["loss"].clear()
        for acc in self._outputs[name]["topk"].values():
            acc.clear()

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx, name="train")
        log_dict = {"train_loss_step": loss}
        log_dict.update({f"train_top{k}_step": acc.mean() for k, acc in topk.items()})
        self.log_dict(log_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss, _ = self.shared_step(batch=batch, batch_idx=batch_idx, name="val")
        return loss

    def on_train_epoch_end(self) -> None:
        return self.shared_on_epoch_end(name="train")

    def on_validation_epoch_end(self) -> None:
        return self.shared_on_epoch_end(name="val")

    def configure_optimizers(self):
        optimizer = SGD(
            self.classification_head.parameters(),
            lr=0.1
            * self.batch_size
            * self.trainer.num_devices
            * self.trainer.num_nodes
            / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        return optimizer

    def on_fit_start(self) -> None:
        # Freeze model weights.
        if self.freeze_model:
            deactivate_requires_grad(model=self.model)

    def on_fit_end(self) -> None:
        # Unfreeze model weights.
        if self.freeze_model:
            activate_requires_grad(model=self.model)
