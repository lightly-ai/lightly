# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import VICRegLLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.transforms.vicregl_transform import VICRegLTransform


class VICRegL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        self.local_projection_head = VicRegLLocalProjectionHead(512, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()

    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z, z_local

    def training_step(self, batch, batch_index):
        views_and_grids = batch[0]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2 :]
        features = [self.forward(view) for view in views]
        loss = self.criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        return optim


model = VICRegL()

pascal_voc = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc", download=True, target_transform=lambda t: 0
)
transform = VICRegLTransform(n_local_views=0)
dataset = LightlyDataset.from_torch_dataset(pascal_voc, transform=transform)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
trainer.fit(model=model, train_dataloaders=dataloader)
