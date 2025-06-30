# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
from functools import partial

import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule


def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(Module):
    def __init__(
        self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(Module):
    def __init__(
        self,
        ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()

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
            drop_path_rate=0.1,  # we recommend using smaller rates like 0.1 for vit-s-14
            mode="uniform",
        )

        freeze_eval_module(self.teacher_backbone)

        # Heads
        dino_head = partial(
            DINOv2ProjectionHead,
            input_dim=384,
        )

        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(
            DINOv2ProjectionHead,
            input_dim=384,
        )

        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head,
            ibot_head=teacher_ibot_head,
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head,
            ibot_head=student_ibot_head,
        )

        freeze_eval_module(self.teacher_head)

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


model = DINOv2()

transform = DINOTransform(
    global_crop_scale=(0.32, 1),
    local_crop_scale=(0.05, 0.32),
    n_local_views=8,
)


# We ignore object detection annotations by setting target_transform to return 0.
def target_transform(t):
    return 0


device = "cuda" if torch.cuda.is_available() else "mps"
model.to(device)

dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)
# Or create a dataset from a folder containing images or videos.
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# Create the loss functions.
dino_criterion = DINOLoss()
ibot_criterion = IBOTPatchLoss()
koleo_criterion = KoLeoLoss()

# Move loss to correct device because it also contains parameters.
dino_criterion = dino_criterion.to(device)
ibot_criterion = ibot_criterion.to(device)
koleo_criterion = koleo_criterion.to(device)

optimizer = AdamW(model.parameters(), lr=0.001)

epochs = 50
num_batches = len(dataloader)
total_steps = epochs * num_batches

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        views = batch[0]
        views = [view.to(device) for view in views]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        # Masking
        B = len(global_views)
        sequence_length = model.teacher_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

        # Mask patches except class token.
        H, W = model.teacher_backbone.vit.patch_embed.grid_size
        assert (
            H * W == sequence_length - 1
        ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
        block_mask = random_block_mask(size=(B, H, W), device=mask.device)
        mask[:, 1:] = block_mask.flatten(start_dim=1)

        # Teacher forward
        with torch.no_grad():
            teacher_cls_token, teacher_features = model.forward_teacher(global_views)
            teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = model.teacher_head.ibot_head.forward(
                teacher_features[mask]
            )

        # Student forward
        (
            student_global_cls_token,
            student_global_masked_features,
        ) = model.forward_student(global_views, mask=mask)
        student_global_cls_out = model.student_head.dino_head.forward(
            student_global_cls_token
        )
        student_global_masked_out = model.student_head.ibot_head.forward(
            student_global_masked_features
        )
        student_local_cls_token, _ = model.forward_student(local_views, mask=None)
        student_local_cls_out = model.student_head.dino_head.forward(
            student_local_cls_token
        )
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

        # Calculate current global step based on epoch and batch index.
        global_step = epoch * num_batches + batch_idx

        # Calculate the loss.
        teacher_temp = linear_warmup_schedule(
            step=global_step,
            warmup_steps=int(30 / epochs * total_steps),
            start_value=0.04,
            end_value=0.07,
        )
        dino_loss = dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        ibot_loss = ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        koleo_loss = 0.1 * sum(
            koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )
        loss = dino_loss + ibot_loss + koleo_loss

        total_loss += loss.detach()
        loss.backward()

        # Optionally zero out the learning rate of the last layer.
        if epoch < 1:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group:
                    param_group["lr"] = 0.0

        # Apply weight decay schedule.
        weight_decay = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=0.04,
            end_value=0.4,
        )

        # Update weight decay directly for all parameter groups.
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

        optimizer.step()
        optimizer.zero_grad()

        # Momentum update teacher.
        momentum = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=0.992,
            end_value=1.0,
        )
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
        update_momentum(model.student_head, model.teacher_head, m=momentum)

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
