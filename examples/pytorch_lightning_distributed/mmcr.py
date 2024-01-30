# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import MMCRLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.mmcr_transform import MMCRTransform
from lightly.utils.scheduler import cosine_schedule


class MMCR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = MMCRLoss()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        z_o = [model(x) for x in batch[0]]
        z_m = [model.forward_momentum(x) for x in batch[0]]

        # Switch dimensions to (batch_size, k, embedding_size)
        z_o = torch.stack(z_o, dim=1)
        z_m = torch.stack(z_m, dim=1)

        loss = self.criterion(z_o, z_m)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)


model = MMCR()

# We disable resizing and gaussian blur for cifar10.
transform = MMCRTransform(k=8, input_size=32, gaussian_blur=0.0)
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=True, transform=transform
)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

# Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
# calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
if __name__ == "__main__":
    trainer = pl.Trainer(
        max_epochs=10,
        devices="auto",
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
    )
    trainer.fit(model=model, train_dataloaders=dataloader)
