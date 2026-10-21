from typing import Dict, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear

from lightly.utils.benchmarking.topk import mean_topk_accuracy


class OnlineLinearClassifier(LightningModule):
    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        predictions: Tensor = self.classification_head(x.detach().flatten(start_dim=1))
        return predictions

    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        predictions = self.forward(features)
        loss = self.criterion(predictions, targets)
        _, predicted_classes = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_classes, targets, k=self.topk)
        return loss, topk

    # type ignore is needed because LightningModule.training_step is typed to return
    # STEP_OUTPUT. this classifier is not run by a Trainer directly; the parent module
    # calls it and logs the returned dict itself.
    def training_step(  # type: ignore[override]
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"train_online_cls_loss": loss}
        log_dict.update({f"train_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict

    # type ignore is needed because LightningModule.validation_step is typed to return
    # STEP_OUTPUT. this classifier is not run by a Trainer directly; the parent module
    # calls it and logs the returned dict itself.
    def validation_step(  # type: ignore[override]
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"val_online_cls_loss": loss}
        log_dict.update({f"val_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict
