import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data.collate import VICRegLCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjector

## The projection head is the same as the Barlow Twins one
from lightly.loss.vicreg_loss import VICRegLoss

class VICReg(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        self.local_projector = VicRegLLocalProjector(512, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z
    
    def forward_local(self, x):
        x = self.backbone(x).transpose(1, 3).transpose(1, 2)
        z = self.local_projector(x)
        return z



resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-2])
model = VICReg(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("home/ubuntu/datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = VICRegLCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256, #2048 from the paper if enough memory
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = VICRegLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x0, x1, grid0, grid1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1, )
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
