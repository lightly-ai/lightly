import math
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.transforms import SimCLRTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
import copy


class BYOL(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = BYOLProjectionHead()
        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = BYOLProjectionHead()
        self.student_prediction_head = BYOLPredictionHead()
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x)
        projections = self.student_projection_head(features)
        predictions = self.student_prediction_head(projections)
        return predictions

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        views = torch.cat(batch[0])
        targets = batch[1]
        with torch.no_grad():
            teacher_features = self.forward(views).flatten(start_dim=1)
            teacher_projections = self.projection_head(teacher_features).chunk(2)

        student_predictions = self.forward_student(views).chunk(2)
        loss = 0.5 * (
            self.criterion(student_predictions[0], teacher_projections[1])
            + self.criterion(student_predictions[1], teacher_projections[0])
        )
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_loss, cls_log = self.online_classifier.training_step(
            (teacher_features.chunk(2)[0].detach(), targets), batch_idx
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

    # TODO: Update optimizer configuration.
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {"name": "byol", "params": params},
                {
                    "name": "byol_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            #   lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * math.sqrt(self.batch_size_per_device * self.trainer.world_size),
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


# BYOL uses same transform as SimCLR.
transform = SimCLRTransform()
