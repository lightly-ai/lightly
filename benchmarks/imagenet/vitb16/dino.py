from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.optim import AdamW

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule


class DINO(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        vit = vit_small_patch16_224(dynamic_img_size=True)
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.projection_head = DINOProjectionHead(input_dim=384, norm_last_layer=False)

        vit_student = vit_small_patch16_224(dynamic_img_size=True, drop_path_rate=0.1)
        self.student_backbone = MaskedVisionTransformerTIMM(vit=vit_student)
        self.student_projection_head = DINOProjectionHead(
            input_dim=384, freeze_last_layer=1, norm_last_layer=False
        )

        self.criterion = DINOLoss()
        self.online_classifier = OnlineLinearClassifier(
            feature_dim=384, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        return projections

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

        views, targets = batch[0], batch[1]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        with torch.no_grad():
            teacher_features = self.forward(global_views).flatten(start_dim=1)
            teacher_projections = self.projection_head(teacher_features)

        student_projections = torch.cat(
            [self.forward_student(global_views), self.forward_student(local_views)]
        )

        loss = self.criterion(
            teacher_out=teacher_projections.chunk(2),
            student_out=student_projections.chunk(len(views)),
            teacher_temp=0.04,  # for benchmarking we use a constant temperature of 0.04
        )
        self.log_dict(
            {"train_loss": loss, "ema_momentum": momentum},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
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

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.student_backbone, self.student_projection_head]
        )

        optimizer = AdamW(
            [
                {"name": "dino", "params": params, "weight_decay": 0.04},
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
            lr=0.0005 * self.batch_size_per_device * self.trainer.world_size / 256,
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

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.04,
            end_value=0.4,
        )
        update_param_groups(
            optimizer, updates=[{"name": "dino", "weight_decay": weight_decay}]
        )


# From https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt
# For 800 epochs training.
# transform = DINOTransform(
#     global_crop_scale=(0.25, 1),
#     local_crop_scale=(0.05, 0.25),
#     n_local_views=10,
# )

# Default settings from https://github.com/facebookresearch/dino/blob/main/main_dino.py
# For vanilla training: https://github.com/facebookresearch/dino?tab=readme-ov-file#vanilla-dino-training-sauropod
transform = DINOTransform(
    global_crop_scale=(0.4, 1),
    local_crop_scale=(0.05, 0.4),
    n_local_views=8,
)
