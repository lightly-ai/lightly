import copy
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD
from torchvision.models import resnet50

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import (
    activate_requires_grad,
    deactivate_requires_grad,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule


class DINO(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = DINOProjectionHead(freeze_last_layer=1)
        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = copy.deepcopy(self.projection_head)
        self.criterion = DINOLoss()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        return projections

    def on_train_start(self) -> None:
        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.projection_head)

    def on_train_end(self) -> None:
        activate_requires_grad(self.backbone)
        activate_requires_grad(self.projection_head)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        teacher_features = self.forward(global_views).flatten(start_dim=1)
        teacher_projections = self.projection_head(teacher_features)
        student_projections = torch.cat(
            [self.forward_student(global_views), self.forward_student(local_views)]
        )
        self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

        loss = self.criterion(
            teacher_projections.chunk(2),
            student_projections.chunk(len(views)),
            epoch=self.current_epoch,
        )
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_loss, cls_log = self.online_classifier.training_step(
            (teacher_features.chunk(2)[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))

        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)
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
            [self.student_backbone, self.student_projection_head]
        )
        # For ResNet50 we use SGD instead of AdamW/LARS as recommended by the authors:
        # https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
        optimizer = SGD(
            [
                {"name": "dino", "params": params},
                {
                    "name": "dino_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size,
            momentum=0.9,
            weight_decay=1e-4,
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


# For ResNet50 we use global crop scale (0.14, 1) instead of (0.4, 1) as recommended
# by the authors:
# https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
transform = DINOTransform(global_crop_scale=(0.14, 1))
