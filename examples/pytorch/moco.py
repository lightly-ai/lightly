import torch
from torch import nn
import torchvision
import copy

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x_q, x_k):
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        k = self.backbone_momentum(x_k).flatten(start_dim=1)
        k = self.projection_head_momentum(k).detach()
        return q, k


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = MoCo(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = MoCoCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss(memory_bank_size=4096)
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x0, x1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        q, k = model(x0, x1)
        loss = criterion(q, k)
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
