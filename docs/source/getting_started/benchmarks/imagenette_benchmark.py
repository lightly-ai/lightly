# -*- coding: utf-8 -*-
"""
Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)
You can download the ImageNette dataset from here: https://github.com/fastai/imagenette

Code has been tested on a V100 GPU with 16GBytes of video memory.

Code to reproduce the benchmark results:

Results (5.3.2022):
------------------------------------------------------------------------------------------
| Model         | Batch Size | Epochs |  KNN Test Accuracy |       Time | Peak GPU Usage |
------------------------------------------------------------------------------------------
| BarlowTwins   |        256 |    200 |              0.587 |   86.2 Min |      4.0 GByte |
| BYOL          |        256 |    200 |              0.619 |   88.6 Min |      4.3 GByte |
| DCL (*)       |        256 |    200 |              0.762 |   53.3 Min |      4.3 GByte |
| DCLW (*)      |        256 |    200 |              0.755 |   53.7 Min |      4.3 GByte |
| DINO (Res18)  |        256 |    200 |              0.736 |   86.5 Min |      4.1 GByte |
| MSN (ViT-S)   |        256 |    200 |              0.741 |   92.7 Min |     16.3 GByte |
| SMoG          |        128 |    200 |              0.698 |  220.9 Min |     14.3 GByte |
| Moco          |        256 |    200 |              0.727 |   87.3 Min |      4.3 GByte |
| NNCLR         |        256 |    200 |              0.726 |   86.8 Min |      4.2 GByte |
| SimCLR        |        256 |    200 |              0.771 |   82.2 Min |      3.9 GByte |
| SimSiam       |        256 |    200 |              0.669 |   78.6 Min |      3.9 GByte |
| SwaV          |        256 |    200 |              0.748 |   77.6 Min |      4.0 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        256 |    800 |              0.789 |  330.9 Min |      4.0 GByte |
| BYOL          |        256 |    800 |              0.851 |  332.7 Min |      4.3 GByte |
| DCL (*)       |        256 |    800 |              0.816 |  213.1 Min |      4.3 GByte |
| DCLW (*)      |        256 |    800 |              0.827 |  213.1 Min |      4.3 GByte |
| DINO (Res18)  |        256 |    800 |              0.881 |  613.9 Min |      6.7 GByte |
| MSN (ViT-S)   |        256 |    800 |              0.834 |  376.1 Min |     16.3 GByte |
| Moco          |        256 |    800 |              0.832 |  322.8 Min |      4.2 GByte |
| NNCLR         |        256 |    800 |              0.848 |  341.4 Min |      4.2 GByte |
| SimCLR        |        256 |    800 |              0.858 |  324.8 Min |      3.9 GByte |
| SimSiam       |        256 |    800 |              0.852 |  316.0 Min |      3.9 GByte |
| SwaV          |        256 |    800 |              0.899 |  554.7 Min |      6.6 GByte |
------------------------------------------------------------------------------------------

(*): Different runtime and memory requirements due to different hardware settings
and pytorch version. Runtime and memory requirements are comparable to SimCLR
with the default settings.

"""
import copy
import math
import os

import time
import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models.modules import masked_autoencoder
from lightly.models import utils
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger

logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')

num_workers = 12
memory_bank_size = 4096

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 200
knn_t = 0.1
classes = 10
input_size = 128

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True). 
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating 
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all 
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False 

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_size = 256
lr_factor = batch_size / 256 # scales the learning rate linearly with batch size


# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if distributed:
    distributed_backend = 'ddp'
    # reduce batch size for distributed training
    batch_size = batch_size // gpus
else:
    distributed_backend = None
    # limit to single gpu if not using distributed training
    gpus = min(gpus, 1)

# The dataset structure should be like this:

path_to_train = '/datasets/imagenette2-160/train/'
path_to_test = '/datasets/imagenette2-160/val/'

# Use SimCLR augmentations
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
)

# Multi crop augmentation for SwAV
swav_collate_fn = lightly.data.SwaVCollateFunction(
    crop_sizes=[128, 64],
    crop_counts=[2, 6] # 2 crops @ 128x128px and 6 crops @ 64x64px
)

# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_collate_fn = lightly.data.DINOCollateFunction(
    global_crop_size=128,
    local_crop_size=64,
)

# Two crops for SMoG
smog_collate_function = lightly.data.collate.SMoGCollateFunction(
    crop_sizes=[128, 128],
    crop_counts=[1, 1],
    crop_min_scales=[0.2, 0.2],
    crop_max_scales=[1.0, 1.0],
)

# Single crop augmentation for MAE
mae_collate_fn = lightly.data.MAECollateFunction()

# Multi crop augmentation for MSN
msn_collate_fn = lightly.data.MSNCollateFunction(random_size=128, focal_size=64)

normalize_transform = torchvision.transforms.Normalize(
    mean=lightly.data.collate.imagenet_normalize['mean'],
    std=lightly.data.collate.imagenet_normalize['std'],
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(128),
    torchvision.transforms.ToTensor(),
    normalize_transform,
])

dataset_train_ssl = lightly.data.LightlyDataset(
    input_dir=path_to_train
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train,
    transform=test_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)

def get_data_loaders(batch_size: int, model):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    col_fn = collate_fn
    if model == SwaVModel:
        col_fn = swav_collate_fn
    elif model == DINOModel:
        col_fn = dino_collate_fn
    elif model == MAEModel:
        col_fn = mae_collate_fn
    elif model == MSNModel:
        col_fn = msn_collate_fn
    elif model == SMoGModel:
        col_fn = smog_collate_function
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        num_splits = 0 if sync_batchnorm else 8
        # TODO: Add split batch norm to the resnet model
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(feature_dim, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = utils.batch_shuffle(x1_, distributed=distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=distributed)
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters())
        optim = torch.optim.SGD(
            params, 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimSiamProjectionHead(feature_dim, 2048, 2048)
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        self.criterion = lightly.loss.NegativeCosineSimilarity()
            
    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2, # no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = heads.BarlowTwinsProjectionHead(feature_dim, 2048, 2048)

        self.criterion = lightly.loss.BarlowTwinsLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
        self.prediction_head = heads.BYOLPredictionHead(256, 4096, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) \
            + list(self.projection_head.parameters()) \
            + list(self.prediction_head.parameters())
        optim = torch.optim.SGD(
            params, 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class NNCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.NNCLRProjectionHead(feature_dim, 2048, 256)
        self.prediction_head = heads.NNCLRPredictionHead(256, 4096, 256)

        self.criterion = lightly.loss.NTXentLoss()
        self.memory_bank = modules.NNMemoryBankModule(size=4096)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = heads.SwaVProjectionHead(feature_dim, 2048, 128)
        self.prototypes = heads.SwaVPrototypes(128, 3000) # use 3000 prototypes

        self.criterion = lightly.loss.SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):

        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(
            high_resolution_features,
            low_resolution_features
        )

        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * lr_factor,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DINOModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = heads.DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(feature_dim, 2048, 256, 2048, batch_norm=True)

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = lightly.loss.DINOLoss(output_dim=2048)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        param = list(self.backbone.parameters()) \
            + list(self.head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DCL(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.DCLLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DCLW(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.DCLWLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class MAEModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        
        decoder_dim = 512
        vit = torchvision.models.vit_b_32(pretrained=False)

        self.warmup_epochs = 40 if max_epochs >= 800 else 20
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size ** 2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images, _, _ = batch
        
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        
        loss = self.criterion(x_pred, target)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_with_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, self.scale_lr)
        return [optim], [cosine_with_warmup_scheduler]

    def scale_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs 
        else:
            return 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (max_epochs - self.warmup_epochs)))

class MSNModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        self.warmup_epochs = 15
        # ViT small configuration (ViT-S/16)
        self.mask_ratio = 0.15
        self.backbone = masked_autoencoder.MAEBackbone(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )
        self.projection_head = heads.MSNProjectionHead(384)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight
        self.criterion = lightly.loss.MSNLoss()

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
        self.log('train_loss_ssl', loss)
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
        optim = torch.optim.AdamW(
            params=params,
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_with_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, self.scale_lr)
        return [optim], [cosine_with_warmup_scheduler]

    def scale_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs 
        else:
            return 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (max_epochs - self.warmup_epochs)))



from sklearn.cluster import KMeans

class SMoGModel(BenchmarkModule):

    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a model based on ResNet
        self.projection_head = heads.SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = heads.SMoGPredictionHead(128, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # smog
        self.n_groups = 300 # 2% malus vs optimal setting of 3000 groups
        memory_bank_size = 300 * batch_size # because we reset the group features every 300 iterations
        self.memory_bank = lightly.loss.memory_bank.MemoryBankModule(size=memory_bank_size)
        # create our loss
        group_features = torch.nn.functional.normalize(
            torch.rand(self.n_groups, 128), dim=1
        ).to(self.device)
        self.smog = heads.SMoGPrototypes(group_features=group_features, beta=0.99)
        self.criterion = nn.CrossEntropyLoss()

    def _reset_group_features(self):
        # see Table 7b)
        features = self.memory_bank.bank
        if features is not None:
            features = features.t().cpu().numpy()
            kmeans = KMeans(self.n_groups).fit(features)
            new_features = torch.from_numpy(kmeans.cluster_centers_).float()
            new_features = torch.nn.functional.normalize(new_features, dim=1)
            self.smog.group_features = new_features.cuda()

    def _reset_momentum_weights(self):
        # see Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def training_step(self, batch, batch_idx):

        if self.global_step > 0 and self.global_step % 300 == 0:
            # reset group features and weights every 300 iterations
            self._reset_group_features()
            self._reset_momentum_weights()
        else:
            # update momentum
            utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
            utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        (x0, x1), _, _ = batch
        if batch_idx % 2:
            tmp = x1
            x1 = x0
            x0 = tmp

        x0_features = self.backbone(x0).flatten(start_dim=1)
        x0_encoded = self.projection_head(x0_features)
        x0_predicted = self.prediction_head(x0_encoded)
        x1_features = self.backbone_momentum(x1).flatten(start_dim=1)
        x1_encoded = self.projection_head_momentum(x1_features)

        # update group features and get group assignments
        assignments = self.smog.assign_groups(x1_encoded)
        self.smog.update_groups(x0_encoded)

        logits = self.smog(x0_predicted, temperature=0.1)
        loss = self.criterion(logits, assignments)

        # use memory bank to periodically reset the group features with k-means
        self.memory_bank(x0_encoded, update=True)

        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters()) + list(self.prediction_head.parameters())        
        optim = torch.optim.SGD(
            params, 
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]




models = [
    BarlowTwinsModel,
    BYOLModel,
    DCL,
    DCLW,
    DINOModel,
    # MAEModel, # disabled by default because MAE uses larger images with size 224
    # MSNModel, # disabled by default because MSN uses larger images with size 224
    MocoModel,
    NNCLRModel,
    SimCLRModel,
    SimSiamModel,
    SwaVModel,
    SMoGModel
]
bench_results = dict()

experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    runs = []
    model_name = BenchmarkModel.__name__.replace('Model', '')
    for seed in range(n_runs):
        pl.seed_everything(seed)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, 
            model=BenchmarkModel,
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f'{model_name}/run{seed}'
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, 'imagenette'),
            name='',
            sub_dir=sub_dir,
            version=experiment_version,
        )
        if experiment_version is None:
            # Save results of all models under same version directory
            experiment_version = logger.version
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, 'checkpoints')
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs, 
            gpus=gpus,
            default_root_dir=logs_root_dir,
            strategy=distributed_backend,
            sync_batchnorm=sync_batchnorm,
            logger=logger,
            callbacks=[checkpoint_callback]
        )
        start = time.time()
        trainer.fit(
            benchmark_model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test
        )
        end = time.time()
        run = {
            'model': model_name,
            'batch_size': batch_size,
            'epochs': max_epochs,
            'max_accuracy': benchmark_model.max_accuracy,
            'runtime': end - start,
            'gpu_memory_usage': torch.cuda.max_memory_allocated(),
            'seed': seed,
        }
        runs.append(run)
        print(run)

        # delete model and trainer + free up cuda memory
        del benchmark_model
        del trainer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    bench_results[model_name] = runs

# print results table
header = (
    f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print('-' * len(header))
print(header)
print('-' * len(header))
for model, results in bench_results.items():
    runtime = np.array([result['runtime'] for result in results])
    runtime = runtime.mean() / 60 # convert to min
    accuracy = np.array([result['max_accuracy'] for result in results])
    gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy.mean():>18.3f}"

    print(
        f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True
    )
print('-' * len(header))
