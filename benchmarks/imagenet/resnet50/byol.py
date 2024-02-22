import copy
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.transforms import BYOLTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule


class BYOL(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = BYOLProjectionHead()
        self.prediction_head = BYOLPredictionHead()
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_projection_head = BYOLProjectionHead()
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        predictions = self.prediction_head(projections)
        return features, predictions

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        features = self.teacher_backbone(x).flatten(start_dim=1)
        projections = self.teacher_projection_head(features)
        return projections

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Momentum update teacher.
        # Settings follow original code for 100 epochs which are slightly different
        # from the paper, see:
        # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.99,
            end_value=1.0,
        )
        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)

        # Forward pass and loss calculation.
        views, targets = batch[0], batch[1]
        teacher_projections_0 = self.forward_teacher(views[0])
        teacher_projections_1 = self.forward_teacher(views[1])
        student_features_0, student_predictions_0 = self.forward_student(views[0])
        _, student_predictions_1 = self.forward_student(views[1])
        # NOTE: Factor 2 because: L2(norm(x), norm(y)) = 2 - 2 * cossim(x, y)
        loss_0 = 2 * self.criterion(teacher_projections_0, student_predictions_1)
        loss_1 = 2 * self.criterion(teacher_projections_1, student_predictions_0)
        # NOTE: No mean because original code only takes mean over batch dimension, not
        # views.
        loss = loss_0 + loss_1
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # Online linear evaluation.
        cls_loss, cls_log = self.online_classifier.training_step(
            (student_features_0.detach(), targets), batch_idx
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
                self.backbone,
                self.projection_head,
                self.prediction_head,
            ]
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
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.45 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-6,
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


# BYOL uses a slight modification of the SimCLR transforms.
# It uses asymmetric augmentation and solarize.
# Check table 6 in the BYOL paper for more info.
transform = BYOLTransform()
