# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from lightly.loss import LeJEPALoss
from lightly.models.modules import LeJEPAEncoder, LeJEPAProjectionHead
from lightly.transforms.dino_transform import DINOTransform


class LeJEPA(Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.backbone = vit_small_patch16_224(
            pretrained=False,
            pos_embed="learn",
            num_classes=512,
            dynamic_img_size=True,
            drop_path_rate=0.1,
        )

        self.encoder = LeJEPAEncoder(
            self.backbone,
            projection_head=LeJEPAProjectionHead(input_dim=512),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


model = LeJEPA()

transform = DINOTransform(
    global_crop_scale=(0.3, 1),
    local_crop_scale=(0.05, 0.3),
    gaussian_blur=(0.5, 0.5, 0.5),
    n_local_views=6,
)


# We ignore object detection annotations by setting target_transform to return 0.
def target_transform(t):
    return 0


device = "cuda" if torch.cuda.is_available() else "mps"
device = "mps"
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
    num_workers=0,
)

# Create the loss functions.
lejepa_criterion = LeJEPALoss()

# Move loss to correct device because it also contains parameters.
lejepa_criterion = lejepa_criterion.to(device)

optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=5e-2)

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

        global_proj = model(global_views)
        local_proj = model(local_views)

        loss = lejepa_criterion(global_proj, local_proj)
        print(
            f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{num_batches}] Loss [{loss.item():.4f}]"
        )

        optimizer.zero_grad()
        loss.backward()
