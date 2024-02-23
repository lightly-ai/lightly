from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module

from lightly.utils.benchmarking import knn_predict
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
        normalize: bool = True,
    ):
        """KNN classifier for benchmarking.

        Settings based on InstDisc [0]. Code adapted from MoCo [1].

        - [0]: InstDisc, 2018, https://arxiv.org/pdf/1805.01978v1.pdf
        - [1]: MoCo, 2019, https://github.com/facebookresearch/moco

        Args:
            model:
                Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            num_classes:
                Number of classes in the dataset.
            knn_k:
                Number of neighbors used for KNN search.
            knn_t:
                Temperature parameter to reweights similarities.
            topk:
                Tuple of integers defining the top-k accuracy metrics to compute.
            feature_dtype:
                Torch data type of the features used for KNN search. Reduce to float16
                for memory-efficient KNN search.
            normalize:
                Whether to normalize the features for KNN search.

        Examples:
            >>> from pytorch_lightning import Trainer
            >>> from torch import nn
            >>> import torchvision
            >>> from lightly.models import LinearClassifier
            >>> from lightly.modles.modules import SimCLRProjectionHead
            >>>
            >>> class SimCLR(nn.Module):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.backbone = torchvision.models.resnet18()
            >>>         self.backbone.fc = nn.Identity() # Ignore classification layer
            >>>         self.projection_head = SimCLRProjectionHead(512, 512, 128)
            >>>
            >>>     def forward(self, x):
            >>>         # Forward must return image features.
            >>>         features = self.backbone(x).flatten(start_dim=1)
            >>>         return features
            >>>
            >>> # Initialize a model.
            >>> model = SimCLR()
            >>>
            >>>
            >>> # Wrap it with a KNNClassifier.
            >>> knn_classifier = KNNClassifier(resnet, num_classes=10)
            >>>
            >>> # Extract features and evaluate.
            >>> trainer = Trainer(max_epochs=1)
            >>> trainer.fit(knn_classifier, train_dataloder, val_dataloader)

        """
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
        self.normalize = normalize

        self._train_features = []
        self._train_targets = []
        self._train_features_tensor: Optional[Tensor] = None
        self._train_targets_tensor: Optional[Tensor] = None

    @torch.no_grad()
    def training_step(self, batch, batch_idx) -> None:
        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
        self._train_features.append(features.cpu())
        self._train_targets.append(targets.cpu())

    def validation_step(self, batch, batch_idx) -> None:
        if self._train_features_tensor is None or self._train_targets_tensor is None:
            return

        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
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

    def on_train_epoch_start(self) -> None:
        # Set model to eval mode to disable norm layer updates.
        self.model.eval()

        # Reset features and targets.
        self._train_features = []
        self._train_targets = []
        self._train_features_tensor = None
        self._train_targets_tensor = None

    def configure_optimizers(self) -> None:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass
