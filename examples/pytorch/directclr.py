# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import CIFAR10

from lightly.loss import DirectCLRLoss
from lightly.transforms.simclr_transform import SimCLRTransform

resnet = models.resnet18()
model = nn.Sequential(*list(resnet.children())[:-1])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
dataset = CIFAR10("datasets/cifar10", download=True, transform=transform)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = DirectCLRLoss(loss_dim=32)
optimizer = SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        x = torch.cat([x0, x1]).to(device)
        z0, z1 = model(x).chunk(2, dim=0)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
