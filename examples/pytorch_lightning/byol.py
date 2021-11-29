import torch
from torch import nn
import torchvision
import copy
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class BYOL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x0, x1):
        y0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(y0)
        p0 = self.prediction_head(z0)
        y1 = self.backbone_momentum(x1).flatten(start_dim=1)
        z1 = self.projection_head_momentum(y1).detach()
        return p0, z1

    def training_step(self, batch, batch_idx):
        update_momentum(model.backbone, model.backbone_momentum, m=0.99)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=0.99
        )
        (x0, x1), filename, label = batch
        p0, z1 = self.forward(x0, x1)
        p1, z0 = self.forward(x1, x0)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)


model = BYOL()

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SimCLRCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(max_epochs=10, gpus=gpus)
trainer.fit(model=model, train_dataloaders=dataloader)
