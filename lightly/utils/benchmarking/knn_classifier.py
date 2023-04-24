from typing import Tuple, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module

from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking.knn import knn_predict
from lightly.utils.benchmarking.topk import mean_topk_accuracy


class KNNClassifier(LightningModule):
    def __init__(
        self,
        model: Module,
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
        topk: Tuple[int, ...] = (1, 5),
        feature_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "num_classes": num_classes,
                "knn_k": knn_k,
                "knn_t": knn_t,
                "topk": topk,
                "feature_dtype": str(feature_dtype),
            }
        )
        self.model = model
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.topk = topk
        self.feature_dtype = feature_dtype

        self._train_features = []
        self._train_targets = []
        self._train_features_tensor: Union[Tensor, None] = None
        self._train_targets_tensor: Union[Tensor, None] = None

    def training_step(self, batch, batch_idx) -> None:
        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        features = F.normalize(features, dim=1).to(self.feature_dtype)
        self._train_features.append(features.cpu())
        self._train_targets.append(targets.cpu())

    def validation_step(self, batch, batch_idx) -> None:
        if self._train_features_tensor is None or self._train_targets_tensor is None:
            return

        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        features = F.normalize(features, dim=1).to(self.feature_dtype)
        predicted_classes = knn_predict(
            feature=features,
            feature_bank=self._train_features_tensor,
            feature_labels=self._train_targets_tensor,
            num_classes=self.num_classes,
            knn_k=self.knn_k,
            knn_t=self.knn_t,
        )
        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=targets, k=self.topk
        )
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=len(targets))

    def on_validation_epoch_start(self) -> None:
        if self._train_features and self._train_targets:
            # Features and targets have size (world_size, batch_size, dim) and
            # (world_size, batch_size) after gather. For non-distributed training,
            # features and targets have size (batch_size, dim) and (batch_size,).
            features = self.all_gather(torch.cat(self._train_features, dim=0))
            self._train_features = []
            targets = self.all_gather(torch.cat(self._train_targets, dim=0))
            self._train_targets = []
            # Reshape to (dim, world_size * batch_size)
            features = features.flatten(end_dim=-2).t().contiguous()
            self._train_features_tensor = features.to(self.device)
            # Reshape to (world_size * batch_size,)
            targets = targets.flatten().t().contiguous()
            self._train_targets_tensor = targets.to(self.device)

    def on_fit_start(self) -> None:
        # Freeze model weights.
        deactivate_requires_grad(model=self.model)

    def on_fit_end(self) -> None:
        # Unfreeze model weights.
        activate_requires_grad(model=self.model)

    def configure_optimizers(self) -> None:
        pass
