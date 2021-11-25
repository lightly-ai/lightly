import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

import lightly
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes

dataset_dir = "/datasets/clothing-dataset/images"


class SwaV(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512)
        self.criterion = lightly.loss.SwaVLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        crops, filename, label = batch
        multi_crop_features = [model(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        return optim


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SwaV(backbone)

dataset = lightly.data.LightlyDataset(input_dir=dataset_dir)
collate_fn = lightly.data.SwaVCollateFunction()

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
