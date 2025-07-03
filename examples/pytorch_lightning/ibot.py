# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
from functools import partial

import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from lightly.loss import DINOLoss, IBOTPatchLoss
from lightly.models.modules import DINOProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.transforms.ibot_transform import IBOTTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule


def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class IBOT(pl.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.output_dim = 8192

        # Backbones
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
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=0.1,
            mode="uniform",
        )
        freeze_eval_module(self.teacher_backbone)
        self.embed_dim = self.student_backbone.vit.embed_dim

        # Projection heads
        projection_head = partial(
            DINOProjectionHead,
            input_dim=self.embed_dim,
            output_dim=self.output_dim,
        )

        self.student_head = projection_head(norm_last_layer=False)
        self.student_cls_head = self.student_patch_head = self.student_head

        self.teacher_head = projection_head()
        self.teacher_cls_head = self.teacher_patch_head = self.teacher_head

        freeze_eval_module(self.teacher_head)

        # Losses
        self.cls_criterion = DINOLoss(
            output_dim=self.output_dim,
            teacher_temp=0.07,
        )
        self.patch_criterion = IBOTPatchLoss(
            output_dim=self.output_dim,
            teacher_temp=0.07,
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
            teacher_cls_out = self.teacher_cls_head.forward(teacher_cls_token)
            teacher_masked_out = self.teacher_patch_head.forward(teacher_features[mask])

        # Student forward
        student_global_cls_token, student_global_masked_features = self.forward_student(
            global_views, mask=mask
        )
        student_global_cls_out = self.student_cls_head.forward(student_global_cls_token)
        student_global_masked_out = self.student_patch_head.forward(
            student_global_masked_features
        )

        student_local_cls_token, _ = self.forward_student(local_views, mask=None)
        student_local_cls_out = self.student_head.forward(student_local_cls_token)
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=int(
                30 / self.trainer.max_epochs * self.trainer.estimated_stepping_batches
            ),
            start_value=0.04,
            end_value=0.07,
        )
        cls_loss = self.cls_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        patch_loss = self.patch_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        loss = cls_loss + patch_loss

        return loss

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=0.001)
        return optim

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        # Optionally zero out the learning rate of the last layer.
        if self.current_epoch < 1:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group:
                    param_group["lr"] = 0.0

        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.04,
            end_value=0.4,
        )
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        return super().on_train_batch_end(outputs, batch, batch_idx)


model = IBOT()

transform = IBOTTransform(
    global_crop_scale=(0.32, 1),
    local_crop_scale=(0.05, 0.32),
    n_local_views=8,
)


# we ignore object detection annotations by setting target_transform to return 0
def target_transform(t):
    return 0


dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=50, devices=1, accelerator=accelerator)
trainer.fit(model=model, train_dataloaders=dataloader)
