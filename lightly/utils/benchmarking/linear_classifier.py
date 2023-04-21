import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module

from lightly.utils.benchmarking.topk import topk_accuracy


class LinearClassifier(LightningModule):
    def __init__(
        self, model: Module, feature_dim: int, num_classes: int, batch_size: int
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

        self._accuracy = {
            "train": {"top1": [], "top5": []},
            "val": {"top1": [], "top5": []},
        }

    def forward(self, x: Tensor) -> Tensor:
        features = self.model.forward(x).flatten(start_dim=1)
        return self.classification_head(features)

    def shared_step(self, batch, batch_idx, name: str) -> Tensor:
        images, targets, _ = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        top1, top5 = topk_accuracy(predictions, targets, k=(1, 5))
        self.log_dict(
            {
                f"{name}_loss": loss,
                f"{name}_top1_step": top1.mean(),  # mean over batch
                f"{name}_top5_step": top5.mean(),  # mean over batch
            },
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self._accuracy[name]["top1"].append(top1.cpu())
        self._accuracy[name]["top5"].append(top5.cpu())
        return loss

    def shared_on_epoch_end(self, name: str) -> None:
        top1 = self.all_gather(torch.cat(self._accuracy[name]["top1"]))
        top5 = self.all_gather(torch.cat(self._accuracy[name]["top5"]))
        self.log_dict(
            {
                f"{name}_top1_epoch": top1.mean(),
                f"{name}_top5_epoch": top5.mean(),
            },
            prog_bar=True,
        )
        for val in self._accuracy[name].values():
            val.clear()

    def training_step(self, batch, batch_idx) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, name="train")

    def validation_step(self, batch, batch_idx) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, name="val")

    def on_train_epoch_end(self) -> None:
        return self.shared_on_epoch_end(name="train")

    def on_validation_epoch_end(self) -> None:
        return self.shared_on_epoch_end(name="val")
