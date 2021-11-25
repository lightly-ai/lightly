import torch
from torch import nn
import torchvision

import lightly
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead
from lightly.models.modules import NNMemoryBankModule

dataset_dir = "/datasets/clothing-dataset/images"


class NNCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = NNCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

memory_bank = NNMemoryBankModule(size=4096)
memory_bank.to(device)

dataset = lightly.data.LightlyDataset(input_dir=dataset_dir)
collate_fn = lightly.data.SimCLRCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for (x0, x1), file_name, label in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0, p0 = model(x0)
        z1, p1 = model(x1)
        z0 = memory_bank(z0, update=False)
        z1 = memory_bank(z1, update=True)
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")