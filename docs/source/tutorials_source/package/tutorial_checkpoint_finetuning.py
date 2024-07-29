"""
.. _lightly-checkpoint-finetuning-tutorial-7:

Tutorial 7: Finetuning Lightly Checkpoints
===========================================

LightlySSL provides pre-trained models on various datasets such as ImageNet1k,
ImageNet100, Imagenette, and CIFAR-10. All these models' weights along with their
hyperparameter configurations are available here :ref:`lightly-benchmarks`.

In this tutorial, we will learn how to use these pre-trained model checkpoints
to fine-tune an image classification model for the Food-101 dataset using
PyTorch Lightning and Weights & Biases.
"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
# Make sure you have the necessary packages installed.
#
# .. code-block:: console
#
#   pip install lightly torchmetrics wandb


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import wandb
from lightning.pytorch.loggers import WandbLogger

from lightly.transforms.utils import IMAGENET_NORMALIZE

# %%
# Downloading Model Checkpoint
# -----------------------------
#
# Let's use the resnet50 model pre-trained on ImageNet1k using
# the `SimCLR <https://arxiv.org/abs/2002.05709>`_ method. We will
# download the model weights from the S3 bucket and store them in memory
#
# .. code-block:: console
#
#   wget -O weights.ckpt https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt
#
# You can browse other model checkpoints at :ref:`lightly-benchmarks`.

# %%
# Configure Weights & Biases
# ---------------------------
#
# We will use the [WandbLogger](https://docs.wandb.ai/guides/integrations/lightning)
# to log our experiments.


wandb.login()
wandb.init(
    project="lightly_finetuning",
    job_type="finetune",
)

wandb_logger = WandbLogger()

# %%
# Configuration
# -------------
# Let's set the configuration parameters for our experiments.
#
# We use a batch size of 32 and an input size of 128.
#
# We only train for 5 epochs because the focus of this tutorial is on
# finetuning lightly checkpoints.

learning_rate = 0.001
num_workers = 8
batch_size = 3
input_size = 128

seed = 42
num_train_epochs = 5

# use cuda if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Setup data augmentations and loaders
# -------------------------------------
# For this tutorial, we'll use the Food-101 dataset of 101 food categories
# with 101,000 images. For each class, 250 manually reviewed test images are
# provided as well as 750 training images. On purpose, the training images
# were not cleaned, and thus still contain some amount of noise. This comes
# mostly in the form of intense colors and sometimes wrong labels.
# All images were rescaled to have a maximum side length of 512 pixels.
#
# We will also use some minimal augmentations for the train and test subsets.
# To learn more about data pipelines in LightlySSL you can refer to :ref:`input-structure-label`
# and to learn more about the different augmentations and learned invariances please refer to
# :ref:`lightly-advanced`.

num_classes = 101

# Training Transformations
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ]
)
train_dataset = torchvision.datasets.Food101(
    "datasets/food101", split="train", download=True, transform=train_transform
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

# Test Transformations
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ]
)
test_dataset = torchvision.datasets.Food101(
    "datasets/food101", split="test", download=True, transform=test_transform
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

# %%
# Load Model Checkpoint
# -----------------------
# Having chosen a model checkpoint to load, we will now create a
# PyTorch Module using the pre-trained weights.


# Initialize the ResNet-50 model
model = torchvision.models.resnet50()
num_feats = model.fc.in_features

# Load the checkpoint
checkpoint = torch.load("/content/weights.ckpt", map_location="cpu")

# Remove 'backbone.' prefix from keys in state_dict
backbone_state_dict = {
    k.replace("backbone.", ""): v
    for k, v in checkpoint["state_dict"].items()
    if k.startswith("backbone.")
}

# Load the state_dict into the model
model.load_state_dict(backbone_state_dict, strict=False)

# Add a classification head
model.fc = nn.Linear(num_feats, num_classes)

# %%
# Create a FineTuning Lightning Module
# --------------------------------------
# Now let's creat a PyTorch Lightning Module to finetune our pre-trained model.


from torchmetrics import Accuracy


class FinetuningModel(pl.LightningModule):
    def __init__(self, model, num_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        _, preds = torch.max(logits, 1)
        acc = self.train_accuracy(preds, y)
        self.log("train_accuracy", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


# %%
# Training
# ----------
#
# Let's instantiate our Lightning Module and train it using the Lightning Trainer.


finetuning_model = FinetuningModel(model, lr=learning_rate, num_classes=num_classes)

trainer = pl.Trainer(
    max_epochs=num_train_epochs, devices=1, logger=wandb_logger, accelerator=device
)

trainer.fit(model=finetuning_model, train_dataloaders=train_dataloader)

wandb.finish()

# %%
# Next Steps
# ------------
#
# Interested in pre-training your own self-supervised models? Check out our other
# tutorials:
#
# - :ref:`input-structure-label`
# - :ref:`lightly-moco-tutorial-2`
# - :ref:`lightly-simsiam-tutorial-4`
# - :ref:`lightly-custom-augmentation-5`
# - :ref:`lightly-detectron-tutorial-6`
#
