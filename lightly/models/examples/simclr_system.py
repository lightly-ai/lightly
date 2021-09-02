import pytorch_lightning as pl
import torch

import lightly
from lightly.models.modules.heads import SimCLRProjectionHead


class SimClrSystem(pl.LightningModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a simclr model based on ResNet
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        # remove the classification head
        self.backbone = torch.nn.Sequential(
            *list(resnet.children())[:-1],
            torch.nn.AdaptiveAvgPool2d(1),
        )
        # use a projection head instead
        num_ftrs = 512
        out_dim = 128
        self.projection_head = \
            SimCLRProjectionHead(num_ftrs, num_ftrs, out_dim)

        # define the criterion / loss function
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        out = self.projection_head(features)
        return out

    def training_step(self, batch, batch_idx):
        # We assume that x0 and x1 are different augmentations of the same
        # image each
        (x0, x1), _, _ = batch
        # Run each augmentation separately through the model
        out0 = self.forward(x0)
        out1 = self.forward(x1)
        # Compute the loss between the different representations of each
        # augmentation
        loss = self.criterion(out0, out1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simclr.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        max_epochs = 200
        scheduler = \
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]