"""SimCLR: a simple framework for contrastive learning of visual representations.

Two augmented views of the same image are pulled together and pushed away from
every other image in the batch, with NT-Xent. Reference: Chen et al. 2020,
https://arxiv.org/abs/2002.05709.

This file is one configuration, written out. It is ResNet-18 on CIFAR-10 at 32
pixels, so it runs on one GPU and finishes; the paper's ImageNet settings are the
``imagenet`` row of ``benchmarks/simclr/datasets.py``, and that row is the one the
published number belongs to. The settings below are the ones this repository
benchmarked SimCLR on CIFAR-10 with.

What changes for a different dataset is the block of constants and the four
lines that consume them. ``benchmarks/simclr/datasets.py`` is that change made
once per dataset, and ``tests/test_simclr_agrees.py`` is what holds the two files
to the same method.

Run it against CIFAR-10 with::

    python examples/simclr.py
"""

import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from lightly.backbones import TorchvisionResNetBackbone, small_image_stem
from lightly.data.sample import Sample, collate
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.optim import param_groups
from lightly.transforms import SimCLRTransform
from lightly.transforms.utils import CIFAR10_NORMALIZE

# ResNet-18, CIFAR-10, 32 pixels. Small enough to run on one GPU.
BATCH_SIZE, EPOCHS, LR, MOMENTUM = 256, 100, 0.06, 0.9
TEMPERATURE, WEIGHT_DECAY = 0.5, 5e-4
INPUT_SIZE, FEATURE_DIM, HIDDEN_DIM, OUT_DIM = 32, 512, 512, 128
COLOR_JITTER_STRENGTH, GAUSSIAN_BLUR, NORMALIZE = 0.5, 0.0, CIFAR10_NORMALIZE

backbone = TorchvisionResNetBackbone(small_image_stem(resnet18()))
head = SimCLRProjectionHead(FEATURE_DIM, HIDDEN_DIM, OUT_DIM)
criterion = NTXentLoss(temperature=TEMPERATURE)
transform = SimCLRTransform(
    input_size=INPUT_SIZE,
    cj_strength=COLOR_JITTER_STRENGTH,
    gaussian_blur=GAUSSIAN_BLUR,
    normalize=NORMALIZE,
)


def forward(sample: Sample) -> Tensor:
    # Both views go through the backbone and the head in one call, so every
    # BatchNorm sees 2N samples. Running one view at a time gives it N, which is
    # a different optimisation problem: the gradients of the two agree at cosine
    # similarity 0.0837.
    images = torch.cat([view.data for view in sample.views])
    z0, z1 = head(backbone.embed(images)).chunk(len(sample.views))
    return criterion(z0, z1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    head.to(device)

    optimizer = SGD(
        param_groups(backbone, head, weight_decay=WEIGHT_DECAY),
        lr=LR,
        momentum=MOMENTUM,
    )
    dataset = CIFAR10("datasets/cifar10", download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        collate_fn=collate,
    )

    print("Starting Training")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for sample in dataloader:
            for view in sample.views:
                view.data = view.data.to(device)
            loss = forward(sample)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
        print(f"epoch: {epoch:>02}, loss: {total_loss / len(dataloader):.5f}")
