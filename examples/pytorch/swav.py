import torch
from torch import nn
import torchvision

import lightly
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes


class SwaV(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SwaV(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = lightly.data.LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = lightly.data.LightlyDataset("path/to/folder")

collate_fn = lightly.data.SwaVCollateFunction(crop_sizes=[32], crop_counts=[2])

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = lightly.loss.SwaVLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch, file_name, label in dataloader:
        multi_crop_features = [model(x.to(device)) for x in batch]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = criterion(high_resolution, low_resolution)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
