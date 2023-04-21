from typing import Dict, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import SGD

from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking.topk import mean_topk_accuracy


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

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.forward(x).flatten(start_dim=1)
        return self.classification_head(features)

    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        predictions = self.forward(features)
        loss = self.criterion(predictions, targets)
        topk = mean_topk_accuracy(predictions.detach(), targets.detach(), k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

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
