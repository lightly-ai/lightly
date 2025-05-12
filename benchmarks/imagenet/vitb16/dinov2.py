from __future__ import annotations

import copy
import math
import re

import torch
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.optim import AdamW, Optimizer

from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import (
    CosineWarmupScheduler,
    cosine_schedule,
    linear_warmup_schedule,
)


class DINOv2(LightningModule):
    def __init__(
        self,
        batch_size_per_device: int,
        num_classes: int,
        ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        # Teacher
        vit_teacher = vit_small_patch16_224(
            pos_embed="learn",
            dynamic_img_size=True,
            init_values=1e-5,
        )
        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=vit_teacher,
            antialias=False,
            pos_embed_initialization="skip",
        )

        self.teacher_dino_head = DINOProjectionHead(
            input_dim=384, norm_last_layer=False
        )
        if ibot_separate_head:
            self.teacher_ibot_head = DINOProjectionHead(
                input_dim=384, norm_last_layer=False
            )
        else:
            self.teacher_ibot_head = self.teacher_dino_head

        # Student
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=0.1,  # we recommend using smaller rates like 0.1 for vit-s-14
            mode="uniform",
        )

        self.student_dino_head = DINOProjectionHead(
            input_dim=384, freeze_last_layer=1, norm_last_layer=False
        )
        if ibot_separate_head:
            self.student_ibot_head = DINOProjectionHead(
                input_dim=384, freeze_last_layer=1, norm_last_layer=False
            )
        else:
            self.student_ibot_head = self.student_dino_head

        # Losses
        self.dino_criterion = DINOLoss()
        self.ibot_criterion = IBOTPatchLoss()
        self.koleo_criterion = KoLeoLoss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=384, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.992,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)

        views, targets = batch[0], batch[1]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        # Masking
        B = len(global_views)
        sequence_length = self.teacher_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
        # Mask patches except class token.
        H, W = self.teacher_backbone.vit.patch_embed.grid_size
        assert (
            H * W == sequence_length - 1
        ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
        block_mask = random_block_mask(size=(B, H, W), device=mask.device)
        mask[:, 1:] = block_mask.flatten(start_dim=1)

        # Teacher forward
        with torch.no_grad():
            teacher_cls_token, teacher_features = self.forward_teacher(global_views)
            teacher_cls_out = self.teacher_dino_head(teacher_cls_token)
            teacher_masked_out = self.teacher_ibot_head(teacher_features[mask])

        # Student forward
        student_global_cls_token, student_global_masked_features = self.forward_student(
            global_views, mask=mask
        )
        student_global_cls_out = self.student_dino_head(student_global_cls_token)
        student_global_masked_out = self.student_ibot_head(
            student_global_masked_features
        )

        student_local_cls_token, _ = self.forward_student(local_views, mask=None)
        student_local_cls_out = self.student_dino_head(student_local_cls_token)
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=int(
                30 / self.trainer.max_epochs * self.trainer.estimated_stepping_batches
            ),
            start_value=0.04,
            end_value=0.07,
        )
        dino_loss = self.dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        koleo_loss = 0.1 * sum(
            self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )
        loss = dino_loss + ibot_loss + koleo_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_dino_loss": dino_loss,
                "train_ibot_loss": ibot_loss,
                "train_koleo_loss": koleo_loss,
                "ema_momentum": momentum,
                "teacher_temp": teacher_temp,
            },
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (teacher_cls_token.chunk(2)[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        cls_token = self.forward(images)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (cls_token.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        lr_scale = math.sqrt(
            self.batch_size_per_device * self.trainer.world_size / 1024
        )
        lr = 0.004 * lr_scale
        num_layers = len(self.student_backbone.vit.blocks)

        def lr_layer(layer_idx: int) -> float:
            return 0.9 ** (num_layers + 1 - layer_idx)  # layer_scale defaults to 0.9

        param_groups = []
        for name, param in self.named_parameters():
            if not "student" in name:
                continue  # Ignore teacher parameters

            group = {
                "name": name,
                "params": [param],
                "lr": lr,
                "weight_decay": 0.04,
            }

            # Update lr
            if any(
                s in name
                for s in [
                    "pos_embed",
                    "mask_token",
                    "cls_token",
                    "register_tokens",
                ]
            ):
                group["lr"] = lr * lr_layer(0)
            elif "patch_embed" in name:
                group["lr"] = lr * lr_layer(0) * 0.2
            elif "residual" in name:
                group["lr"] = lr
            elif "blocks" in name:
                layer_idx = int(re.search(r"blocks\.(\d+)\.", name).group(1))
                group["lr"] = lr * lr_layer(layer_idx + 1)
            elif "vit.norm" in name:
                pass  # Do not update vit.norm parameters
            elif "head" in name:
                pass  # Do not update classification and dino/ibot head parameters
            else:
                assert False, f"Unknown parameter: {name}"

            # Update weight_decay
            if name.endswith(".bias") or ".norm" in name or "gamma" in name:
                group["weight_decay"] = 0.0

            # Ignore ViT classification head
            if not "vit.head" in name:
                param_groups.append(group)

        param_groups.append(
            {
                "name": "online_classifier",
                "params": self.online_classifier.parameters(),
                "lr": lr,
                "weight_decay": 0.0,
            }
        )

        optimizer = AdamW(param_groups, lr=lr)
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
        self.student_dino_head.cancel_last_layer_gradients(self.current_epoch)
        self.student_ibot_head.cancel_last_layer_gradients(self.current_epoch)

        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.04,
            end_value=0.4,
        )
        updates = []
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                updates.append({"name": group["name"], "weight_decay": weight_decay})
        update_param_groups(optimizer, updates=updates)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=3.0,
            gradient_clip_algorithm="norm",
        )


transform = DINOTransform(
    global_crop_scale=(0.32, 1),
    local_crop_scale=(0.05, 0.32),
    n_local_views=8,
)
