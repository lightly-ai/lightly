import copy
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.transforms import BYOLTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule


class TiCo(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = TiCoProjectionHead()

        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = copy.deepcopy(self.projection_head)

        self.criterion = TiCoLoss(beta=0.9, rho=8.0, gather_distributed=True)

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        return features, projections

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        return projections

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.99,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        # Forward pass and loss calculation.
        views, targets = batch[0], batch[1]
        teacher_features, teacher_projections = self.forward_teacher(views[0])
        student_projections = self.forward_student(views[1])

        loss, transformative_invariance_loss, covariance_contrast_loss = self.criterion(
            teacher_projections, student_projections
        )

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )
        self.log(
            "train_transformative_invariance_loss",
            transformative_invariance_loss,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(targets),
        )
        self.log(
            "train_covariance_contrast_loss",
            covariance_contrast_loss,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online linear evaluation.
        cls_loss, cls_log = self.online_classifier.training_step(
            (teacher_features.detach(), targets), batch_idx
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
        params, params_no_weight_decay = get_weight_decay_parameters(
            [
                self.student_backbone,
                self.student_projection_head,
            ]
        )
        optimizer = LARS(
            [
                {"name": "tico", "params": params},
                {
                    "name": "tico_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.02 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1.5e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


# TiCo uses BYOL augmentations.
transform = BYOLTransform()
