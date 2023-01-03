# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data.collate import VICRegLCollateFunction
## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjector
from lightly.loss import VICRegLLoss



class VICRegL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        self.local_projector = VicRegLLocalProjector(512, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()

    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z
    
    def forward_local(self, x):
        x = self.backbone(x).transpose(1, 3).transpose(1, 2)
        z = self.local_projector(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1, grid0, grid1), _, _ = batch
        z0 = model.forward(x=x0)
        z1 = model.forward(x=x1)
        z0_local = model.forward_local(x0)
        z1_local = model.forward_local(x1)
        loss = self.criterion.forward(
            z_a=z0, 
            z_b=z1, 
            z_a_local=z0_local, 
            z_b_local=z1_local, 
            location_a=grid0, 
            location_b=grid1
            )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = VICRegL()

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = VICRegLCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gpus = torch.cuda.device_count()

# train with DDP and use Synchronized Batch Norm for a more accurate batch norm
# calculation
trainer = pl.Trainer(
    max_epochs=10, 
    gpus=gpus,
    strategy='ddp',
    sync_batchnorm=True,
)
trainer.fit(model=model, train_dataloaders=dataloader)
