# This example requires the following dependencies to be installed:
# pip install "lightly[timm]"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU: fewer clusters, a shallow predictor,
# and a small backbone. The encoder is a standard masked ViT; the reference
# additionally uses rotary position embeddings in the encoder.

import copy

import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor

from lightly.loss import CAPILoss
from lightly.models import utils
from lightly.models.modules import (
    CAPIPredictorTIMM,
    CAPIProjectionHead,
    MaskedVisionTransformerTIMM,
)
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms import CAPITransform
from lightly.utils.scheduler import cosine_schedule


class CAPI(pl.LightningModule):
    def __init__(self, num_clusters: int = 1024) -> None:
        super().__init__()
        self.mask_ratio = 0.65
        vit = vit_small_patch16_224(img_size=256, reg_tokens=7)
        self.student_backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.student_backbone.sequence_length
        self.num_prefix_tokens = vit.num_prefix_tokens
        embed_dim = vit.embed_dim
        grid_size = vit.patch_embed.grid_size[0]

        self.student_head = CAPIProjectionHead(
            input_dim=embed_dim, num_clusters=num_clusters, weight_norm=True
        )
        self.predictor = CAPIPredictorTIMM(
            embed_dim=embed_dim, grid_size=grid_size, depth=2, num_heads=6
        )

        # The teacher backbone is an exponential moving average of the student; the
        # teacher head is trained by the clustering loss.
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = CAPIProjectionHead(
            input_dim=embed_dim, num_clusters=num_clusters, bias=True
        )
        deactivate_requires_grad(self.teacher_backbone)

        # Enable synchronization of the Sinkhorn normalization across processes.
        self.criterion = CAPILoss(gather_distributed=True)

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

    def forward_teacher(self, images: Tensor) -> Tensor:
        # Return the cluster logits of all patch tokens so the online clustering
        # normalizes over the full grid; the masked targets are selected in the loss.
        with torch.no_grad():
            features = self.teacher_backbone.encode(images=images)
        logits = self.teacher_head(features)
        return logits[:, self.num_prefix_tokens :]

    def training_step(self, batch, batch_idx) -> Tensor:
        momentum = cosine_schedule(
            step=self.current_epoch, max_steps=10, start_value=0.996, end_value=1.0
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)

        views = batch[0]
        images = views[0]  # views contains only a single view
        idx_keep, idx_mask = utils.random_inverse_block_mask(
            size=(images.shape[0], self.sequence_length),
            mask_ratio=self.mask_ratio,
            num_prefix_tokens=self.num_prefix_tokens,
            device=images.device,
        )
        student_logits = self.forward_student(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask
        )
        teacher_logits = self.forward_teacher(images=images)
        student_loss = self.criterion(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            teacher_index=idx_mask - self.num_prefix_tokens,
        )
        clustering_loss = self.criterion(
            teacher_logits=teacher_logits, student_logits=teacher_logits
        )
        return student_loss + clustering_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1.5e-4)


model = CAPI()

transform = CAPITransform(input_size=256)
# We ignore object detection annotations by setting target_transform to return 0.


def target_transform(t):
    return 0


dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)
# Or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# Train with DDP on multiple gpus. Distributed sampling is also enabled with
# use_distributed_sampler=True.
trainer = pl.Trainer(
    max_epochs=10,
    devices="auto",
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
)
trainer.fit(model=model, train_dataloaders=dataloader)
