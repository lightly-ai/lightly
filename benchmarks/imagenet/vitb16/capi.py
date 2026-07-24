from __future__ import annotations

import copy

import torch
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_base_patch16_224
from torch import Tensor
from torch.optim import AdamW

from lightly.loss import CAPILoss
from lightly.models import utils
from lightly.models.modules import (
    CAPIPredictorTIMM,
    CAPIProjectionHead,
    MaskedVisionTransformerTIMM,
)
from lightly.transforms import CAPITransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule


class CAPI(LightningModule):
    def __init__(
        self,
        batch_size_per_device: int,
        num_classes: int,
        num_clusters: int = 16384,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.mask_ratio = 0.65
        self.prediction_subsampling = 0.05

        vit = vit_base_patch16_224(img_size=256, reg_tokens=16, dynamic_img_size=True)
        self.student_backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.student_backbone.sequence_length
        self.num_prefix_tokens = vit.num_prefix_tokens
        self.embed_dim = vit.embed_dim
        grid_size = vit.patch_embed.grid_size[0]

        self.student_head = CAPIProjectionHead(
            input_dim=self.embed_dim, num_clusters=num_clusters, weight_norm=True
        )
        self.predictor = CAPIPredictorTIMM(
            embed_dim=self.embed_dim, grid_size=grid_size, depth=12, num_heads=12
        )

        # The teacher backbone is an exponential moving average of the student; the
        # teacher head is trained by the clustering loss.
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = CAPIProjectionHead(
            input_dim=self.embed_dim, num_clusters=num_clusters, bias=True
        )
        utils.deactivate_requires_grad(self.teacher_backbone)

        # Stochastic depth on the student backbone, following the reference (0.2).
        utils.update_drop_path_rate(
            self.student_backbone.vit, drop_path_rate=0.2, mode="uniform"
        )

        self.criterion = CAPILoss(gather_distributed=True)

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=self.embed_dim, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(images=x)

    def forward_student(
        self, images: Tensor, idx_keep: Tensor, idx_mask: Tensor
    ) -> Tensor:
        visible_tokens = self.student_backbone.encode(images=images, idx_keep=idx_keep)
        num_prefix_tokens = self.num_prefix_tokens
        predicted_tokens = self.predictor(
            context=visible_tokens[:, num_prefix_tokens:],
            context_positions=idx_keep[:, num_prefix_tokens:] - num_prefix_tokens,
            query_positions=idx_mask - num_prefix_tokens,
        )
        return self.student_head(predicted_tokens)

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        images = images[0]  # images is a list containing only one view
        idx_keep, idx_mask = utils.random_inverse_block_mask(
            size=(images.shape[0], self.sequence_length),
            mask_ratio=self.mask_ratio,
            num_prefix_tokens=self.num_prefix_tokens,
            device=images.device,
        )
        # Predict only a random subset of the masked patches, following the
        # reference's prediction subsampling.
        num_predict = int(idx_mask.shape[1] * self.prediction_subsampling)
        subsample = torch.argsort(
            torch.rand(idx_mask.shape, device=idx_mask.device), dim=1
        )[:, :num_predict]
        idx_predict = torch.gather(idx_mask, dim=1, index=subsample)

        student_logits = self.forward_student(
            images=images, idx_keep=idx_keep, idx_mask=idx_predict
        )
        with torch.no_grad():
            teacher_features = self.teacher_backbone.encode(images=images)
        # All patch tokens, so the online clustering normalizes over the full grid;
        # the masked targets are selected in the loss via teacher_index.
        teacher_logits = self.teacher_head(teacher_features)[
            :, self.num_prefix_tokens :
        ]

        # The student loss trains the student on the masked patches; the clustering
        # loss trains the teacher head's prototypes over all patches.
        student_loss = self.criterion(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            teacher_index=idx_predict - self.num_prefix_tokens,
        )
        clustering_loss = self.criterion(
            teacher_logits=teacher_logits, student_logits=teacher_logits
        )
        loss = student_loss + clustering_loss
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        cls_features = teacher_features[:, 0]
        cls_loss, cls_log = self.online_classifier.training_step(
            (cls_features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        cls_features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (cls_features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # Momentum update of the teacher backbone.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        utils.update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)

    def configure_optimizers(self):
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.student_backbone, self.student_head, self.predictor]
        )
        # The predictor's mask token is a learned token and is not weight-decayed.
        params = [p for p in params if p is not self.predictor.mask_token]
        params_no_weight_decay.append(self.predictor.mask_token)
        # The online-clustering (teacher) head is trained at half the backbone
        # learning rate, following the reference's separate clustering optimizer.
        cluster_params, cluster_no_weight_decay = utils.get_weight_decay_parameters(
            [self.teacher_head]
        )
        lr = 1.5e-4 * self.batch_size_per_device * self.trainer.world_size / 256
        optimizer = AdamW(
            [
                {"name": "capi", "params": params},
                {
                    "name": "capi_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {"name": "capi_clustering", "params": cluster_params, "lr": 0.5 * lr},
                {
                    "name": "capi_clustering_no_weight_decay",
                    "params": cluster_no_weight_decay,
                    "lr": 0.5 * lr,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=lr,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        max_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(int(max_steps / self.trainer.max_epochs * 40), max_steps)
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_steps,
                max_epochs=max_steps,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


transform = CAPITransform(input_size=256)
