# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules import DenseCLProjectionHead
from lightly.transforms import DenseCLTransform
from lightly.utils.scheduler import cosine_schedule


class DenseCL(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head_global = DenseCLProjectionHead(512, 512, 128)
        self.projection_head_local = DenseCLProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(
            self.projection_head_global
        )
        self.projection_head_local_momentum = copy.deepcopy(self.projection_head_local)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_global_momentum)
        utils.deactivate_requires_grad(self.projection_head_local_momentum)

    def forward(self, x):
        query_features = self.backbone(x)
        query_global = self.pool(query_features).flatten(start_dim=1)
        query_global = self.projection_head_global(query_global)
        query_features = query_features.flatten(start_dim=2).permute(0, 2, 1)
        query_local = self.projection_head_local(query_features)
        # Shapes: (B, H*W, C), (B, D), (B, H*W, D)
        return query_features, query_global, query_local

    def forward_momentum(self, x):
        key_features = self.backbone(x)
        key_global = self.pool(key_features).flatten(start_dim=1)
        key_global = self.projection_head_global(key_global)
        key_features = key_features.flatten(start_dim=2).permute(0, 2, 1)
        key_local = self.projection_head_local(key_features)
        return key_features, key_global, key_local


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-2])
model = DenseCL(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


transform = DenseCLTransform(input_size=32)
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=True, transform=transform
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion_global = NTXentLoss(memory_bank_size=(4096, 128))
criterion_local = NTXentLoss(memory_bank_size=(4096, 128))
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

epochs = 10

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum = cosine_schedule(epoch, epochs, 0.996, 1)
    for batch in dataloader:
        x_query, x_key = batch[0]
        utils.update_momentum(model.backbone, model.backbone_momentum, m=momentum)
        utils.update_momentum(
            model.projection_head_global,
            model.projection_head_global_momentum,
            m=momentum,
        )
        utils.update_momentum(
            model.projection_head_local,
            model.projection_head_local_momentum,
            m=momentum,
        )
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        query_features, query_global, query_local = model(x_query)
        key_features, key_global, key_local = model.forward_momentum(x_key)

        key_local = utils.select_most_similar(query_features, key_features, key_local)
        query_local = query_local.flatten(end_dim=1)
        key_local = key_local.flatten(end_dim=1)

        loss_global = criterion_global(query_global, key_global)
        loss_local = criterion_local(query_local, key_local)
        loss = 0.5 * (loss_global + loss_local)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
