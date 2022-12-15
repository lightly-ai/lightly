# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models import SwaV


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SwaV(backbone, num_ftrs=512, out_dim=128, n_prototypes=512)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# we ignore object detection annotations by setting target_transform to return 0
pascal_voc = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc", download=True, target_transform=lambda t: 0
)
dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SwaVCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = SwaVLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch, _, _ in dataloader:
        batch = [x.to(device) for x in batch]
        high_resolution, low_resolution = batch[:2], batch[2:]
        high_resolution, low_resolution = model(high_resolution, low_resolution)
        loss = criterion(high_resolution, low_resolution)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
