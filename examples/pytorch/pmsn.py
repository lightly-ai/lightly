# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import copy

import torch
import torchvision
from torch import nn

from lightly.loss import PMSNLoss
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTorchvision
from lightly.models.modules.heads import MSNProjectionHead
from lightly.transforms import MSNTransform


class PMSN(nn.Module):
    def __init__(self, vit):
        super().__init__()

        self.mask_ratio = 0.15
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        self.projection_head = MSNProjectionHead(384)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight

    def forward(self, images):
        out = self.backbone(images=images)
        return self.projection_head(out)

    def forward_masked(self, images):
        batch_size, _, _, width = images.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        out = self.anchor_backbone(images=images, idx_keep=idx_keep)
        return self.anchor_projection_head(out)


# ViT small configuration (ViT-S/16)
vit = torchvision.models.VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=384 * 4,
)
model = PMSN(vit)
# # or use a torchvision ViT backbone:
# vit = torchvision.models.vit_b_32(pretrained=False)
# model = PMSN(vit)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = MSNTransform()
# we ignore object detection annotations by setting target_transform to return 0
dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=lambda t: 0,
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

criterion = PMSNLoss()

params = [
    *list(model.anchor_backbone.parameters()),
    *list(model.anchor_projection_head.parameters()),
    model.prototypes,
]
optimizer = torch.optim.AdamW(params, lr=1.5e-4)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        views = batch[0]
        utils.update_momentum(model.anchor_backbone, model.backbone, 0.996)
        utils.update_momentum(
            model.anchor_projection_head, model.projection_head, 0.996
        )

        views = [view.to(device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = model.backbone(images=targets)
        targets_out = model.projection_head(targets_out)
        anchors_out = model.forward_masked(anchors)
        anchors_focal_out = model.forward_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = criterion(anchors_out, targets_out, model.prototypes.data)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
