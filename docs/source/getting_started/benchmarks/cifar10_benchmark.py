# -*- coding: utf-8 -*-
"""

Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)


Code to reproduce the benchmark results:

| Model   | Epochs | Batch Size | Test Accuracy | Peak GPU usage |
|---------|--------|------------|---------------|----------------|
| MoCo    |  200   | 128        | 0.83          | 2.1 GBytes     |
| SimCLR  |  200   | 128        | 0.78          | 2.0 GBytes     |
| SimSiam |  200   | 128        | 0.73          | 3.0 GBytes     |
| MoCo    |  200   | 512        | 0.85          | 7.4 GBytes     |
| SimCLR  |  200   | 512        | 0.83          | 7.8 GBytes     |
| SimSiam |  200   | 512        | 0.81          | 7.0 GBytes     |
| MoCo    |  800   | 512        | 0.90          | 7.2 GBytes     |
| SimCLR  |  800   | 512        | 0.89          | 7.7 GBytes     |
| SimSiam |  800   | 512        | 0.91          | 6.9 GBytes     |

"""
import copy
import os

import time
import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.modules.heads import ProjectionHead
from lightly.models.modules.heads import SwaVProjectionHead
from lightly.models.modules.heads import SwaVPrototypes
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger

num_workers = 8
memory_bank_size = 4096

logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 200
knn_t = 0.1
classes = 10

# distributed training settings
distributed = False
sync_batchnorm = False
gather_distributed = False # gather features from all gpus before calculating loss

# benchmark
n_runs = 5 # optional, increase to create multiple runs and report mean + std
batch_sizes = [128, 512]

# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if distributed:
    distributed_backend = 'ddp'
    # reduce batch size for distributed training
    batch_sizes = [size // gpus for size in batch_sizes]
else:
    distributed_backend = None
    # limit to single gpu if not using distributed training
    gpus = min(gpus, 1)

# Adapted from our MoCo Tutorial on CIFAR-10
#
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from kaggle 
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
path_to_train = '/datasets/cifar10/train/'
path_to_test = '/datasets/cifar10/test/'

# Use SimCLR augmentations, additionally, disable blur for cifar10
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# Multi crop augmentation for SwAV
swav_collate_fn = lightly.data.SwaVCollateFunction(
    crop_sizes=[32],
    crop_counts=[2], # 2 crops @ 32x32px
    crop_min_scales=[0.14]
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
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

def get_data_loaders(batch_size: int, multi_crops: bool = False):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn if not multi_crops else swav_collate_fn,
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
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=num_splits)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

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
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = batch_shuffle(x1_, distributed=distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = batch_unshuffle(x1_, shuffle, distributed=distributed)
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
            lr=6e-2,
            momentum=0.9, 
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=512)
        self.criterion = lightly.loss.NTXentLoss(gather_distributed=gather_distributed)
            
    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_simclr.parameters(), 
            lr=6e-2,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512)
        # replace the 3-layer projection head by a 2-layer projection head
        self.resnet_simsiam.projection_mlp = ProjectionHead([
            (
                self.resnet_simsiam.num_ftrs,
                self.resnet_simsiam.proj_hidden_dim,
                nn.BatchNorm1d(self.resnet_simsiam.proj_hidden_dim),
                nn.ReLU(inplace=True)
            ),
            (
                self.resnet_simsiam.proj_hidden_dim,
                self.resnet_simsiam.out_dim,
                nn.BatchNorm1d(self.resnet_simsiam.out_dim),
                None
            )
        ])
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_simsiam.parameters(), 
            lr=6e-2,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        # create a barlow twins model based on ResNet
        self.resnet_barlowtwins = \
            lightly.models.BarlowTwins(
                self.backbone, 
                num_ftrs=512,
                proj_hidden_dim=2048,
                out_dim=2048,
            )
        # replace the 3-layer projection head by a 2-layer projection head
        self.resnet_barlowtwins.projection_mlp = ProjectionHead([
            (
                self.resnet_barlowtwins.num_ftrs,
                self.resnet_barlowtwins.proj_hidden_dim,
                nn.BatchNorm1d(self.resnet_barlowtwins.proj_hidden_dim),
                nn.ReLU(inplace=True)
            ),
            (
                self.resnet_barlowtwins.proj_hidden_dim,
                self.resnet_barlowtwins.out_dim,
                None,
                None
            )
        ])
        self.criterion = lightly.loss.BarlowTwinsLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        self.resnet_barlowtwins(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_barlowtwins(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_barlowtwins.parameters(), 
            lr=6e-2,
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256,1024,256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)
            x0_ = self.prediction_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            return x0_, x1_

        p0, z1 = step(x0, x1)
        p1, z0 = step(x1, x0)
        
        loss = self.criterion((z0, p0), (z1, p1))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) \
            + list(self.projection_head.parameters()) \
            + list(self.prediction_head.parameters())
        optim = torch.optim.SGD(
            params, 
            lr=6e-2,
            momentum=0.9, 
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512) # use 512 prototypes

        self.criterion = lightly.loss.SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):

        # normalize the prototypes so they are on the unit sphere
        lightly.models.utils.normalize_weight(
            self.prototypes.layers.weight
        )

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
            lr=1e-3,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

models = [MocoModel, SimCLRModel, SimSiamModel, BarlowTwinsModel, BYOLModel, SwaVModel]
bench_results = dict()

# loop through configurations and train models
for batch_size in batch_sizes:
    for BenchmarkModel in models:
        runs = []
        model_name = BenchmarkModel.__name__
        for seed in range(n_runs):
            pl.seed_everything(seed)
            dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
            benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

            logger = TensorBoardLogger('cifar10_runs', version=model_name)

            trainer = pl.Trainer(
                max_epochs=max_epochs, 
                gpus=gpus,
                default_root_dir=logs_root_dir,
                strategy=distributed_backend,
                sync_batchnorm=sync_batchnorm,
            )
            start = time.time()
            trainer.fit(
                benchmark_model,
                train_dataloaders=dataloader_train_ssl,
                val_dataloaders=dataloader_test
            )
            end = time.time()
            run = {
                'seed': seed,
                'runtime': end - start,
                'max_accuracy': benchmark_model.max_accuracy,
                'gpu_memory_usage': torch.cuda.max_memory_allocated(),
            }
            runs.append(run)

            # delete model and trainer + free up cuda memory
            del benchmark_model
            del trainer
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        bench_results[model_name] = runs

for model, results in bench_results.items():
    runtime = np.array([result['runtime'] for result in results])
    accuracy = np.array([result['max_accuracy'] for result in results])
    gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])

    print(
        f'{model}: {accuracy.mean():.3f} +- {accuracy.std():.3f}'
        f', GPU used: {gpu_memory_usage.max() / (1024.0**3):.1f} GByte'
        f', Time: {runtime.mean() // 60} min',
        flush=True
    )
