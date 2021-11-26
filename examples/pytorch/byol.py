import torch
from torch import nn
import torchvision
import copy

import lightly
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x0, x1):
        y0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(y0)
        p0 = self.prediction_head(z0)
        y1 = self.backbone_momentum(x1).flatten(start_dim=1)
        z1 = self.projection_head_momentum(y1).detach()
        return p0, z1


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BYOL(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = lightly.data.LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = lightly.data.LightlyDataset("path/to/folder")

collate_fn = lightly.data.SimCLRCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = lightly.loss.NegativeCosineSimilarity()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x0, x1), file_name, label in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        p0, z1 = model(x0, x1)
        p1, z0 = model(x1, x0)
        loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        update_momentum(model.backbone, model.backbone_momentum, m=0.99)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=0.99
        )
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")