import copy
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import (
    get_weight_decay_parameters,
    update_momentum,
    batch_shuffle,
    batch_unshuffle,
)
from lightly.transforms import MoCoV2Transform
from lightly.utils.benchmarking import OnlineLinearClassifier
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler


class MoCoV2(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = MoCoProjectionHead()
        self.query_backbone = copy.deepcopy(self.backbone)
        self.query_projection_head = MoCoProjectionHead()
        self.criterion = NTXentLoss(temperature=0.2, memory_bank_size=65536)

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def forward_key_encoder(self, x: Tensor) -> Tensor:
        x, idx = batch_shuffle(batch=x, distributed=self.trainer.num_devices > 1)
        features = self.forward(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        features = batch_unshuffle(batch=features, idx=idx)
        projections = batch_unshuffle(batch=projections, idx=idx)
        return features, projections

    def forward_query_encoder(self, x: Tensor) -> Tensor:
        features = self.query_backbone(x).flatten(start_dim=1)
        projections = self.query_projection_head(features)
        return projections

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        query_projections = self.forward_query_encoder(views[1])
        update_momentum(self.query_backbone, self.backbone, m=0.999)
        update_momentum(self.query_projection_head, self.projection_head, m=0.999)
        key_features, key_projections = self.forward_key_encoder(views[0])
        loss = self.criterion(query_projections, key_projections)

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # Online linear evaluation.
        cls_loss, cls_log = self.online_classifier.training_step(
            (key_features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        # NOTE: The original implementation from the authors uses weight decay for all
        # parameters.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = SGD(
            [
                {"name": "mocov2", "params": params},
                {
                    "name": "mocov2_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = MoCoV2Transform()
