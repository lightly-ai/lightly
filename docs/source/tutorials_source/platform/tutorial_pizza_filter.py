# -*- coding: utf-8 -*-

"""
This documentation accompanies the video tutorial: `youtube link <https://youtu.be/imQWZ0HhYjk>`_

##############################################################################

Tutorial 1: Curate Pizza Images
===============================

In this tutorial, you will learn how to upload a dataset to the Lightly platform,
curate the data, and finally use the curated data to train a model.

What you will learn
-------------------

* Create and upload a new dataset via the web frontend
* Curate a dataset using simple image metrics such as Width, Height, Sharpness, Signal-to-Noise ratio, File Size
* Download images based on a tag from a dataset
* Train an image classifier with the filtered dataset


Requirements
------------

You can use your dataset or use the one we provide with this tutorial: 
:download:`pizzas.zip <../../../_data/pizzas.zip>`. 
If you use your dataset, please make sure the images are smaller than 
2048 pixels with width and height, and you use less than 1000 images.

.. note::  For this tutorial, we provide you with a small dataset of pizza images.
    We chose a small dataset because it's easy to ship and train.

Upload the data
---------------

We start by uploading the dataset to the `Lightly Platform <https://app.lightly.ai>`_.

Create a new account if you do not have one yet and create a new dataset. You can upload images
using drag and drop from your local machine.

Filter the dataset using metadata
---------------------------------

Once the dataset is created and the
images uploaded, you can head to 'Histogram' under the 'Analyze & Filter' menu.

Move the sliders below the histograms to define filter rules for the dataset.
Once you are satisfied with the filtered dataset, create a new tag using the tag menu
on the left side.

Download the curated dataset
----------------------------

We have filtered the dataset and want to download it now to train a model.
Therefore, click on the download menu on the left.

We can now download the filtered images by clicking on the 'DOWNLOAD IMAGES' button.
In our case, the images are stored in the 'pizzas' folder. We now have to 
annotate the images. We can do this by moving the individual images to 
subfolders corresponding to the class. E.g. we move salami pizza images to the
'salami' folder and Margherita pizza images to the 'margherita' folder.

##############################################################################

Training a model using the curated data
---------------------------------------

"""


# %%
# Now we can start training our model using PyTorch Lightning
# We start by importing the necessary dependencies
import os
import torch
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18


# %%
# We use a small batch size to make sure we can run the training on all kinds 
# of machines. Feel free to adjust the value to one that works on your machine.
batch_size = 8
seed = 42

# %%
# Set the seed to make the experiment reproducible
pl.seed_everything(seed)

#%%
# Let's set up the augmentations for the train and the test data.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# we don't do any resizing or mirroring for the test data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# %%
# We load our data and split it into train/test with a 70/30 ratio.

# Please make sure the data folder contains subfolders for each class
#
# pizzas
#  L salami
#  L margherita
dset = ImageFolder('pizzas', transform=train_transform)

# to use the random_split method we need to obtain the length
# of the train and test set
full_len = len(dset)
train_len = int(full_len * 0.7)
test_len = int(full_len - train_len)
dataset_train, dataset_test = random_split(dset, [train_len, test_len])
dataset_test.transforms = test_transform

print('Training set consists of {} images'.format(len(dataset_train)))
print('Test set consists of {} images'.format(len(dataset_test)))

# %%
# We can create our data loaders to fetch the data from the training and test
# set and pack them into batches.
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

# %%
# PyTorch Lightning allows us to pack the loss as well as the 
# optimizer into a single module.
class MyModel(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.save_hyperparameters()

        # load a pretrained resnet from torchvision
        self.model = resnet18(pretrained=True)

        # add new linear output layer (transfer learning)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.accuracy.compute(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)


# %%
# Finally, we can create the model and use the Trainer
# to train our model.
model = MyModel()
trainer = pl.Trainer(max_epochs=4)
trainer.fit(model, dataloader_train, dataloader_test)
