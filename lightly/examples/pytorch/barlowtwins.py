import torch
from torch import nn
import torchvision

import lightly
from lightly.models.modules import BarlowTwinsProjectionHead

dataset_dir = "/datasets/clothing-dataset"
batch_size = 128
num_workers = 8
lr = 0.06


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BarlowTwins(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = lightly.data.LightlyDataset(input_dir=dataset_dir)
collate_fn = lightly.data.ImageCollateFunction(input_size=224)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

criterion = lightly.loss.BarlowTwinsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

print('Starting Training')
for epoch in range(10):
    total_loss = 0
    for (x0, x1), file_name, label in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
