# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import torch
import torchvision
from torch import nn

from lightly.loss import MMCRLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.mmcr_transform import MMCRTransform
from lightly.utils.scheduler import cosine_schedule


class MMCR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = MMCR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = MMCRTransform(k=8, input_size=32, gaussian_blur=0.0)
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=True, transform=transform
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = MMCRLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

epochs = 10

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    for batch in dataloader:
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=momentum_val
        )
        z_o = [model(x.to(device)) for x in batch[0]]
        z_m = [model.forward_momentum(x.to(device)) for x in batch[0]]

        # Switch dimensions to (batch_size, k, embedding_size)
        z_o = torch.stack(z_o, dim=1)
        z_m = torch.stack(z_m, dim=1)

        loss = criterion(z_o, z_m)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
