import torch
from torch import nn
import torchvision
import copy
import pytorch_lightning as pl

import lightly
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

dataset_dir = "/datasets/clothing-dataset/images"


class BYOL(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()

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


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BYOL(backbone)

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

gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(max_epochs=10, gpus=gpus)
trainer.fit(model=model, train_dataloaders=dataloader)
