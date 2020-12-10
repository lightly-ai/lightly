# -*- coding: utf-8 -*-
"""

.. _lightly-moco-tutorial-2:

Tutorial 2: Train MoCo on CIFAR-10
==============================================

In this tutorial, we will train a model based on the MoCo Paper
`Momentum Contrast for Unsupervised Visual Representation Learning <https://arxiv.org/abs/1911.05722>`_.

When training self-supervised models using contrastive loss we
usually face one big problem. To get good results, we need
many negative examples for the contrastive loss to work. Therefore,
we need a large batch size. However, not everyone has access to a cluster
full of GPUs or TPUs. To solve this problem, alternative approaches have been developed.
Some of them use a memory bank to store old negative examples we can query 
to compensate for the smaller batch size. MoCo takes this approach
one step further by including a momentum encoder.

We use the **CIFAR-10** dataset for this tutorial.

In this tutorial you will learn:

- How to use lightly to load a dataset and train a model

- How to create a MoCo model with a memory bank

- How to use the pre-trained model after self-supervised learning for a 
  transfer learning task

"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
# Make sure you have lightly installed.
#
# .. code-block:: console
#
#   pip install lightly

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly

# %%
# Configuration
# -------------
# 
# We set some configuration parameters for our experiment.
# Feel free to change them and analyze the effect.

num_workers = 8
batch_size = 256
memory_bank_size = 4096
seed = 1
max_epochs = 50

# %%
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

# %%
# Let's set the seed to ensure reproducibility of the experiments
pl.seed_everything(seed)


# %%
# Setup data augmentations and loaders
# ------------------------------------
#
# We start with our data preprocessing pipeline. We can implement augmentations
# from the MOCO paper using the collate functions provided by lightly. For MoCo v2,
# we can use the same augmentations as SimCLR but override the input size and blur.
# Images from the CIFAR-10 dataset have a resolution of 32x32 pixels. Let's use
# this resolution to train our model. 
#
# .. note::  We could use a higher input resolution to train our model. However, 
#   since the original resolution of CIFAR-10 images is low there is no real value
#   in increasing the resolution. A higher resolution results in higher memory
#   consumption and to compensate for that we would need to reduce the batch size.

# MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# %%
# We don't want any augmentation for our test data. Therefore,
# we create custom, torchvision based data transformations.
# Let's ensure the size is correct and we normalize the data in
# the same way as we do with the training data.

# Augmentations typically used to train on cifar-10
train_classifier_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# We use the moco augmentations for training moco
dataset_train_moco = lightly.data.LightlyDataset(
    from_folder=path_to_train
)

# Since we also train a linear classifier on the pre-trained moco model we
# reuse the test augmentations here (MoCo augmentations are very strong and 
# usually reduce accuracy of models which are not used for contrastive learning.
# Our linear layer will be trained using cross entropy loss and labels provided
# by the dataset. Therefore we chose light augmentations.)
dataset_train_classifier = lightly.data.LightlyDataset(
    from_folder=path_to_train,
    transform=train_classifier_transforms
)

dataset_test = lightly.data.LightlyDataset(
    from_folder=path_to_test,
    transform=test_transforms
)

# %%
# Create the dataloaders to load and preprocess the data 
# in the background.

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# %%
# Create the MoCo Lightning Module
# ---------------------------
# Now we create our MoCo model. We use PyTorch Lightning to train
# our model. We follow the specification of the lightning module.
# In this example we set the number of features for the hidden dimension to 512.
# The momentum for the Momentum Encoder is set to 0.99 (default is 0.999) since
# other reports show that this works better for Cifar-10.
#
# .. note:: We use a split batch norm to simulate multi-gpu behaviour. Combined
#   with the use of batch shuffling, this prevents the model from communicating 
#   through the batch norm layers.

class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        projection = self.resnet_moco(x)
        loss = self.criterion(projection)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        return torch.optim.SGD(self.resnet_moco.parameters(), lr=3e-2,
                               momentum=0.9, weight_decay=5e-4)


# %%
# Create the Classifier Lightning Module
# ---------------------------
# We create a linear classifier using the features we extract using MoCo
# and train it on the dataset

class Classifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        # create a moco based on ResNet
        self.resnet_moco = model

        # freeze the layers of moco
        for p in self.resnet_moco.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_moco.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.fc.parameters(), lr=10.0)
        

# %%
# Train the MoCo model
# ---------------
#
# We can instantiate the model and train it using the
# lightning trainer.

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

model = MocoModel()
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=100)
trainer.fit(
    model,
    dataloader_train_moco
)

# %%
# Train the Classifier
classifier = Classifier(model.resnet_moco)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=100)
trainer.fit(
    classifier,
    dataloader_train_classifier,
    dataloader_test
)

# %%
# Checkout the tensorboard logs while the model is training.
#
# Run `tensorboard --logdir lightning_logs/` to start tensorboard
# 
# .. note:: If you run the code on a remote machine you can't just
#   access the tensorboard logs. You need to forward the port.
#   You can do this by using an editor such as Visual Studio Code
#   which has a port forwarding functionality (make sure
#   the remote extensions are installed and are connected with your machine).
#   
#   Or you can use a shell command similar to this one to forward port
#   6006 from your remote machine to your local machine:
#
#   `ssh username:host -N -L localhost:6006:localhost:6006`
