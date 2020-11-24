# -*- coding: utf-8 -*-
"""

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

"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
# You can install all of them using pip.
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
max_epochs = 1 #100

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
# from the MOCO paper using the following collate function provided by lightly.
# We can overwrite any default parameter such as in this example the input_size.
# Images from the CIFAR-10 dataset have a resolution of 32x32 pixels. Let's use
# this resolution to train our model. 
#
# .. note::  We could use a higher input resolution to train our model. However, 
#   since the original resolution of CIFAR-10 images is low there is no real value
#   in increasing the resolution. A higher resolution results in higher memory
#   consumption and to compensate for that we would need to reduce the batch size.

collate_fn = lightly.data.MoCoCollateFunction(
    input_size=32,
)

# %%
# We don't want any augmentation for our test data. Therefore,
# we create custom torchvision based data transformations.
# Let's ensure the size is correct and we normalize the data in
# the same way as we do with the training data.
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train = lightly.data.LightlyDataset(
    from_folder=path_to_train
)

dataset_test = lightly.data.LightlyDataset(
    from_folder=path_to_test,
    transform=test_transforms
)

# %%
# Create the dataloaders to load and preprocess the data 
# in the background.

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
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
# Create the Lightning Module
# ---------------------------
# Now we create our model. We use PyTorch Lightning to train
# our model. We follow the specification of the lightning module.

class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # create a moco based on ResNet
        self.model = lightly.models.ResNetMoCo(num_ftrs=128)

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(128, 10)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            memory_bank_size=memory_bank_size)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        self.model(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, _ = batch

        if optimizer_idx == 0:  # moco model
            projection = self.model(x)
            loss = self.criterion(projection)
            self.log('train_loss_ssl', loss, prog_bar=True)
        else:  # linear layer
            y_hat = self.model.features(x).squeeze().detach()
            y_hat = self.fc(y_hat)
            loss = nn.functional.cross_entropy(y_hat, y)
            self.log('train_loss_fc', loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model.features(x).squeeze()
        y_hat = self.fc(y_hat)

        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return [
            torch.optim.SGD(self.model.parameters(), lr=0.3,
                            momentum=0.9, weight_decay=1e-4),
            torch.optim.SGD(self.fc.parameters(), lr=1e-3,
                            momentum=0.9, weight_decay=1e-4)
        ]

# %%
# Train the model
# ---------------
#
# We can instantiate the model and train it using the
# lightning trainer.

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

model = MocoModel()
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
trainer.fit(model, dataloader_train, dataloader_test)


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

# %%
# The plots in tensorboard should look similar to the ones here:
#
# .. image:: images/moco_memory_bank/training_loss.png
#   :alt: Training loss for MoCo with different memory banks
#
# .. image:: images/moco_memory_bank/validation_accuracy.png
#   :alt: Validation accuracy for MoCo with different memory banks
