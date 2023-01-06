import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data.collate import VICRegLCollateFunction
## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.loss import VICRegLLoss

class VICRegL(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        self.local_projection_head = VicRegLLocalProjectionHead(512, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1) # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)         
        return z, z_local

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-2])
model = VICRegL(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = VICRegLCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128, #2048 from the paper if enough memory
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = VICRegLLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (view_global, view_local, grid_global, grid_local), _, _ in dataloader:
        view_global = view_global.to(device)
        view_local = view_local.to(device)
        grid_global = grid_global.to(device)
        grid_local = grid_local.to(device)
        z_global, z_global_local_features = model(view_global)
        z_local, z_local_local_features = model(view_local)
        loss = criterion(
            z_a=z_global, 
            z_b=z_local, 
            z_a_local=z_global_local_features, 
            z_b_local=z_local_local_features, 
            location_a=grid_global, 
            location_b=grid_local
        )
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
