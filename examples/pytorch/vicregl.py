import torch
import torchvision
from torch import nn

from lightly.loss import VICRegLLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.transforms.vicregl_transform import VICRegLTransform


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
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z, z_local


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-2])
model = VICRegL(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = VICRegLTransform(n_local_views=0)
dataset = pascal_voc = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc", download=True, transform=transform
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = VICRegLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for views_and_grids, _, _ in dataloader:
        views_and_grids = [x.to(device) for x in views_and_grids]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2 :]
        features = [model(view) for view in views]
        loss = criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
