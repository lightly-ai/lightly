# This example requires the following dependencies to be installed:
# pip install "lightly[timm]"

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU: fewer clusters, a shallow predictor,
# and a small backbone. The encoder is a standard masked ViT; the reference
# additionally uses rotary position embeddings in the encoder.

import copy

import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor, nn

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


class CAPI(nn.Module):
    def __init__(self, vit: nn.Module, num_clusters: int = 1024) -> None:
        super().__init__()
        self.mask_ratio = 0.65
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

        # The teacher backbone is an exponential moving average of the student.
        # The teacher head is trained by the clustering loss (see the training loop).
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = CAPIProjectionHead(
            input_dim=embed_dim, num_clusters=num_clusters, bias=True
        )
        deactivate_requires_grad(self.teacher_backbone)

    def forward_student(
        self, images: Tensor, idx_keep: Tensor, idx_mask: Tensor
    ) -> Tensor:
        # Encode only the visible patches, then predict the masked patches from them.
        visible_tokens = self.student_backbone.encode(images=images, idx_keep=idx_keep)
        num_prefix_tokens = self.num_prefix_tokens
        predicted_tokens = self.predictor(
            context=visible_tokens[:, num_prefix_tokens:],
            context_positions=idx_keep[:, num_prefix_tokens:] - num_prefix_tokens,
            query_positions=idx_mask - num_prefix_tokens,
        )
        return self.student_head(predicted_tokens)

    def forward_teacher(self, images: Tensor) -> Tensor:
        # The teacher encodes the full image; its backbone is frozen (EMA-updated).
        # Return the cluster logits of all patch tokens so the online clustering
        # normalizes over the full grid; the masked targets are selected in the loss.
        with torch.no_grad():
            features = self.teacher_backbone.encode(images=images)
        logits = self.teacher_head(features)
        return logits[:, self.num_prefix_tokens :]


vit = vit_small_patch16_224(img_size=256, reg_tokens=7)
model = CAPI(vit)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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

criterion = CAPILoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

epochs = 10

print("Starting Training")
for epoch in range(epochs):
    momentum = cosine_schedule(
        step=epoch, max_steps=epochs, start_value=0.996, end_value=1.0
    )
    total_loss = 0
    for batch in dataloader:
        views = batch[0]
        images = views[0].to(device)  # views contains only a single view

        idx_keep, idx_mask = utils.random_inverse_block_mask(
            size=(images.shape[0], model.sequence_length),
            mask_ratio=model.mask_ratio,
            num_prefix_tokens=model.num_prefix_tokens,
            device=images.device,
        )
        student_logits = model.forward_student(
            images=images, idx_keep=idx_keep, idx_mask=idx_mask
        )
        teacher_logits = model.forward_teacher(images=images)

        # The student loss trains the student on the masked patches; the clustering
        # loss trains the teacher head's prototypes over all patches. The teacher
        # targets are balanced over the full grid by the Sinkhorn normalization, and
        # the masked positions are selected via teacher_index.
        student_loss = criterion(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            teacher_index=idx_mask - model.num_prefix_tokens,
        )
        clustering_loss = criterion(
            teacher_logits=teacher_logits, student_logits=teacher_logits
        )
        loss = student_loss + clustering_loss

        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Momentum update of the teacher backbone.
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
