# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import copy

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data.collate import MSNCollateFunction
from lightly.loss import MSNLoss
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models import utils
from lightly.models.modules.masked_autoencoder import MAEBackbone


class MSN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # ViT small configuration (ViT-S/16)
        self.mask_ratio = 0.15
        self.backbone = MAEBackbone(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )
        # or use a torchvision ViT backbone:
        # vit = torchvision.models.vit_b_32(pretrained=False)
        # self.backbone = MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(384)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight

        # set gather_distributed to True for distributed training
        self.criterion = MSNLoss(gather_distributed=True)

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, _, _ = batch
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(anchors, idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(params, lr=1.5e-4)
        return optim


model = MSN()

# we ignore object detection annotations by setting target_transform to return 0
pascal_voc = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc", download=True, target_transform=lambda t: 0
)
dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = MSNCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gpus = torch.cuda.device_count()

# Train with DDP on multiple gpus. Distributed sampling is also enabled with 
# replace_sampler_ddp=True.
trainer = pl.Trainer(
    max_epochs=10, 
    gpus=gpus,
    strategy='ddp',
    replace_sampler_ddp=True,
)
trainer.fit(model=model, train_dataloaders=dataloader)
