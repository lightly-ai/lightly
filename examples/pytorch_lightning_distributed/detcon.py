# This example requires the following dependencies to be installed:
# pip install lightly
# Note: This example requires torchvision >= 0.17 for tv_tensors support.
# To run with unsupervised segmentation masks instead of a grid:
# pip install lightly scikit-image

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.transforms.v2 import ToImage
from torchvision.tv_tensors import Mask

from lightly.loss import DetConSLoss
from lightly.models import utils
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import DetConSTransform

try:
    from skimage.segmentation import felzenszwalb

    SCIKIT_IMAGE_INSTALLED = True
except ImportError:
    print("scikit-image is not installed, running with grid masks.")
    SCIKIT_IMAGE_INSTALLED = False


class DetConS(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.num_cls = 25
        self.criterion = DetConSLoss(gather_distributed=True)

    def forward(self, x, mask):
        features = self.backbone(x)
        h, w = features.shape[2], features.shape[3]
        mask_down = (
            F.interpolate(mask.float(), size=(h, w), mode="nearest").long().squeeze(1)
        )
        pooled = utils.pool_masked(features, mask_down, num_cls=self.num_cls)
        pooled = pooled.permute(0, 2, 1)
        b, m, d = pooled.shape
        z = self.projection_head(pooled.reshape(b * m, d))
        return z.reshape(b, m, -1)

    def training_step(self, batch, batch_idx):
        (x0, mask0), (x1, mask1) = batch[0]
        z0 = self.forward(x0, mask0)
        z1 = self.forward(x1, mask1)
        idx = (
            torch.arange(self.num_cls, device=self.device)
            .unsqueeze(0)
            .expand(x0.shape[0], -1)
        )
        loss = self.criterion(z0, z1, idx, idx)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = DetConS()

if SCIKIT_IMAGE_INSTALLED:
    _detcons_transform = DetConSTransform(input_size=96)
else:
    _detcons_transform = DetConSTransform(grid_size=(5, 5), input_size=96)

_to_image = ToImage()


def transform(pil_img):
    tv_img = _to_image(pil_img)
    if SCIKIT_IMAGE_INSTALLED:
        segments = felzenszwalb(
            np.array(pil_img), scale=100, sigma=0.5, min_size=20
        ).astype(np.int64)
        segments = np.clip(segments, 0, model.num_cls - 1)
        mask = Mask(torch.from_numpy(segments).unsqueeze(0))
    else:
        mask = Mask(torch.zeros(1, *tv_img.shape[-2:], dtype=torch.int64))
    return _detcons_transform(tv_img, mask)


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
trainer = pl.Trainer(
    max_epochs=10,
    devices="auto",
    accelerator="gpu",
    strategy="ddp",
    sync_batchnorm=True,
    use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
)
trainer.fit(model=model, train_dataloaders=dataloader)
