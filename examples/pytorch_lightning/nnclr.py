import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

import lightly
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead
from lightly.models.modules import NNMemoryBankModule

dataset_dir = "/datasets/clothing-dataset/images"


class NNCLR(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)
        self.memory_bank = NNMemoryBankModule(size=4096)

        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), filename, label = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = NNCLR(backbone)

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
