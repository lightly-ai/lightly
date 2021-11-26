import torch
from torch import nn
import torchvision
import copy
import pytorch_lightning as pl

import lightly
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class MoCo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NTXentLoss(memory_bank_size=4096)

    def forward(self, x_query, x_key):
        query = self.backbone(x_query).flatten(start_dim=1)
        query = self.projection_head(query)

        key = self.backbone_momentum(x_key).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return query, key

    def training_step(self, batch, batch_idx):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(
            self.projection_head, self.projection_head_momentum, m=0.99
        )
        (x_query, x_key), filename, label = batch
        query, key = self.forward(x_query, x_key)
        loss = self.criterion(query, key)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = MoCo()

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
dataset = lightly.data.LightlyDataset.from_torch_dataset(cifar10)
# or create a dataset from a folder containing images or videos:
# dataset = lightly.data.LightlyDataset("path/to/folder")

collate_fn = lightly.data.MoCoCollateFunction(input_size=32)

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
