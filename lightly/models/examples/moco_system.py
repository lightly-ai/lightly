import pytorch_lightning as pl
import torch

from lightly.models.modules.heads import MoCoProjectionHead

import lightly
from lightly.models.modules.momentum_encoder import MomentumEncoder


class MocoSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = torch.nn.Sequential(
            *list(resnet.children())[:-1],
            torch.nn.AdaptiveAvgPool2d(1),
        )
        # create a moco model based on ResNet
        num_ftrs = 512
        out_dim = 128
        self.projection_head = MoCoProjectionHead(num_ftrs, num_ftrs, out_dim)

        # define the momentum encoded model
        self.momentum_encoder = \
            MomentumEncoder(backbone=self.backbone,
                            projection_head=self.projection_head,
                            momentum=0.999,
                            batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=4096)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        out = self.projection_head(features)
        return out

    def training_step(self, batch, batch_idx):
        # We assume that x0 and x1 are different augmentations of the same
        # image each
        (x0, x1), _, _ = batch
        # Run each augmentation separately through the model or momentum model
        out0 = self.forward(x0)
        out1 = self.momentum_encoder.forward(x1)

        # We use a symmetric loss
        # (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/
        # blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(out0, out1)
        loss_2 = self.criterion(out1, out0)
        loss = 0.5 * (loss_1 + loss_2)

        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        max_epochs = 200
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max_epochs)
        return [optim], [scheduler]
