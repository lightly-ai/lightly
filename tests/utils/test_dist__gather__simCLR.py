from typing import Any
import numpy as np
from torch.nn import Linear, MSELoss
from torch.optim import SGD
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from lightly.utils import dist
import torch.distributed as torch_dist
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightly.loss import NTXentLoss
import argparse


import torchvision
from torch import nn


from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
import torch
from torch.utils.data import Dataset
from PIL import Image
from reprlib import repr



import torchvision.transforms as transforms


class SimCLR(pl.LightningModule):
    def __init__(self, gather: bool):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = NTXentLoss(gather_distributed=gather)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        self.print_(
            "training_step before loss",
            z0_shape=z0.shape,
            z1_shape=z1.shape,
            x0_shape=x0.shape,
            x1_shape=x1.shape,
            z0=z0[:,:4], z1=z1[:,:4],
            #x0=x0[:,:4], x1=x1[:,:4]
        )
        loss = self.criterion(z0, z1)
        self.print_(
            "training_step after gather",
            loss=loss.item(),
            z0_shape=z0.shape,
            z1_shape=z1.shape,
            x0_shape=x0.shape,
            x1_shape=x1.shape,
            z0=z0[:,:4], z1=z1[:,:4],
            #x0=x0[:,:4], x1=x1[:,:4]
        )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=1.0)
        return optim
    
    def on_after_backward(self) -> None:
        result = super().on_after_backward()
        self.print_("on_after_backward")
        return result
    
    def print_(self, step, **kwargs):
        p = list(self.parameters())[0]
        if True:
            print(
                f"{step}, {self.local_rank=}, {self.global_rank=}, {repr(p.grad)=}, {repr(p)=}, {kwargs=}"
            )

    def configure_optimizers(self) -> Any:
        return SGD(self.parameters(), lr=0.1)
    


class RandomImageDataset(Dataset):
    def __init__(self, num_images, transform):
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        array = np.random.rand(32, 32, 3) * 255
        image = Image.fromarray(array.astype("uint8")).convert("RGB")
        return self.transform(image), 0

def test_gather() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", type=int, default=1, help="Number of devices")
    args = parser.parse_args()
    n_devices = args.n_devices
    gather = n_devices > 1

    n_samples = 8
    
    pl.seed_everything(42, workers=True)
    batch_size = n_samples / n_devices
    assert int(batch_size) == batch_size
    batch_size = int(batch_size)


    model = SimCLR(gather=gather)

    transform = SimCLRTransform(input_size=32)
    if False:
        dataset = RandomImageDataset(n_samples, transform)
    else:
        dataset = torchvision.datasets.CIFAR10(
            "datasets/cifar10", download=True, transform=transform
        )
        dataset.data = dataset.data[:n_samples]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    trainer = Trainer(
        devices=n_devices,
        accelerator="cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=1,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    test_gather()