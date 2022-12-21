# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision
import copy

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import schedule_momentum
from torch.autograd import Variable

class TiCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = TiCoProjectionHead(512, 1024, 256)

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
model = TiCo(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cifar10 = torchvision.datasets.CIFAR10("/home/ubuntu/datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SimCLRCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = TiCoLoss(
    beta_param = 0.9,
    ro_param = 20.0,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

C_prev = Variable(torch.zeros(256, 256), requires_grad=True).to(device)
C_prev = C_prev.detach()

epochs = 10

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = schedule_momentum(epoch, epochs, m = 0.99)
    for (x0, x1), _, _ in dataloader:
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model.forward_momentum(x1)
        loss, C = criterion(C_prev, z0, z1)
        C_prev = C.detach()
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")