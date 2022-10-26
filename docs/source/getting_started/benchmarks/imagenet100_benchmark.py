# -*- coding: utf-8 -*-
"""
Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)
Code has been tested on a V100 GPU with 16GBytes of video memory.
"""
import copy
import os

import time
import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import BenchmarkModule
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.loggers import TensorBoardLogger

logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')

num_workers = 12
memory_bank_size = 2**16

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 20
knn_t = 0.1
classes = 100
input_size=224

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True). 
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating 
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all 
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False 

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_size = 256
lr_factor = batch_size / 256 # scales the learning rate linearly with batch size


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

path_to_train = '/datasets/imagenet100/train/'
path_to_test = '/datasets/imagenet100/val/'

# Use SimCLR augmentations
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
)

# Multi crop augmentation for SwAV
swav_collate_fn = lightly.data.SwaVCollateFunction()

# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_collate_fn = lightly.data.DINOCollateFunction()

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(224),
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

steps_per_epoch = len(dataset_train_ssl) // batch_size


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
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)

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
            temperature=0.07,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.999)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.999)

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
            lr=0.03 * lr_factor,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss(temperature=0.1)

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
        optim = LARS(
            self.parameters(), 
            lr=0.3 * lr_factor,
            momentum=0.9, 
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": LambdaLR(
                optimizer=optim,
                lr_lambda=linear_warmup_decay(
                    warmup_steps=steps_per_epoch * 10, 
                    total_steps=steps_per_epoch * max_epochs, 
                    cosine=True,
                )
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]

class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        self.projection_head = heads.SimSiamProjectionHead(feature_dim, 512, 2048)
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
            lr=0.05 * lr_factor,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
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
        optim = LARS(
            self.parameters(), 
            lr=0.2 * lr_factor,
            momentum=0.9, 
            weight_decay=1.5 * 1e-6,
        )
        scheduler = {
            "scheduler": LambdaLR(
                optimizer=optim,
                lr_lambda=linear_warmup_decay(
                    warmup_steps=steps_per_epoch * 10, 
                    total_steps=steps_per_epoch * max_epochs, 
                    cosine=True,
                )
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]

class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
        self.prediction_head = heads.BYOLProjectionHead(256, 4096, 256)

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
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.999)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, m=0.999)
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
        optim = LARS(
            params,
            lr=0.2 * lr_factor,
            momentum=0.9,
            weight_decay=1.5 * 1e-6,
        )
        scheduler = {
            "scheduler": LambdaLR(
                optimizer=optim,
                lr_lambda=linear_warmup_decay(
                    warmup_steps=steps_per_epoch * 10, 
                    total_steps=steps_per_epoch * max_epochs, 
                    cosine=True,
                )
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]


class NNCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = heads.NNCLRPredictionHead(256, 4096, 256)
        self.projection_head = heads.NNCLRProjectionHead(feature_dim, 4096, 256)

        self.criterion = lightly.loss.NTXentLoss()
        self.memory_bank = modules.NNMemoryBankModule(size=memory_bank_size)

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
        optim = LARS(
            self.parameters(), 
            lr=0.3 * lr_factor,
            momentum=0.9, 
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
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
        optim = LARS(
            self.parameters(),
            lr=4.8 * lr_factor,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": LambdaLR(
                optimizer=optim,
                lr_lambda=linear_warmup_decay(
                    warmup_steps=steps_per_epoch * 10, 
                    total_steps=steps_per_epoch * max_epochs, 
                    cosine=True,
                )
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]


class DINOModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes, knn_k=knn_k, knn_t=knn_t)
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
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.999)
        utils.update_momentum(self.head, self.teacher_head, m=0.999)
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
        optim = LARS(
            param,
            lr=0.3 * lr_factor,
            weight_decay=1e-6,
            momentum=0.9,
        )
        scheduler = {
            "scheduler": LambdaLR(
                optimizer=optim,
                lr_lambda=linear_warmup_decay(
                    warmup_steps=steps_per_epoch * 10, 
                    total_steps=steps_per_epoch * max_epochs, 
                    cosine=True,
                )
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]


models = [
    BarlowTwinsModel, 
    BYOLModel,
    DINOModel,
    MocoModel,
    NNCLRModel,
    SimCLRModel,
    SimSiamModel,
    SwaVModel,
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

        # Save logs to: {CWD}/benchmark_logs/imagenet/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f'{model_name}/run{seed}'
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, 'imagenet'),
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

# print results table
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
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte

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